//! One-click automatic refusal-edit optimization.
//!
//! Heretic-style closed-loop search over (strength, block) edit candidates:
//! every candidate is really edited into a staged GGUF, really measured for
//! refusal behaviour and held-out functional capability, and the selection is
//! delegated to the existing measured-score chooser. Losers are deleted; the
//! winner is published atomically. No score is ever estimated.

use crate::gguf::{inspect_gguf, inspect_gguf_tensors};
use crate::gguf_edit::{prepare_gguf_abliteration_value, run_gguf_abliteration};
use crate::jobs::run_with_log;
use crate::llama::{prepare_gguf, quantize_gguf_with_log, Precision};
use crate::method_select::choose_edit_method;
use crate::refusal::{prepare_refusal_eval_with_server, run_refusal_eval};
use crate::task_eval::{prepare_eval_with_server, run_eval_reusing_baseline};
use serde_json::{json, Value};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn invalid(s: impl Into<String>) -> io::Error { io::Error::new(ErrorKind::InvalidInput, s.into()) }
fn recipe_receipt(path: &Path) -> PathBuf {
    let mut name = path.as_os_str().to_os_string();
    name.push(".ojon-recipe.json");
    PathBuf::from(name)
}

#[derive(Clone, Debug, PartialEq)]
pub struct UncensorAutoRequest {
    pub source: PathBuf,
    pub output_dir: PathBuf,
    /// ojon.refusal_suite.v1 probe suite (harmful + optional benign controls).
    pub refusal_probes: PathBuf,
    /// ojon.task_suite.v1 held-out functional suite; REQUIRED so capability
    /// retention is always measured, never assumed.
    pub capability_suite: PathBuf,
    /// Folder with paired positive.txt / negative.txt behaviour examples.
    pub paired_prompts: PathBuf,
    pub cvector_generator: PathBuf,
    pub runtime: PathBuf,
    /// Optional persistent llama-server: one model load per candidate.
    pub server_runtime: Option<PathBuf>,
    pub gpu_layers: i32,
    pub device: Option<String>,
    pub refusal_tokens: u32,
    pub task_tokens: u32,
    /// Sweep grid. Defaults: strengths [0.10, 0.20, 0.35] and the three
    /// middle-depth attention blocks.
    pub strengths: Vec<f64>,
    pub blocks: Option<Vec<usize>>,
    pub max_capability_loss: f64,
    pub max_refusal_rate: Option<f64>,
    /// Optional final quantization of the published winner.
    pub quantize: Option<Precision>,
    pub quantizer: PathBuf,
    /// Proceed when disk/RAM telemetry is unavailable or below policy.
    pub allow_resource_risk: bool,
}
impl UncensorAutoRequest {
    pub fn new(source: PathBuf, output_dir: PathBuf) -> Self {
        Self { source, output_dir, refusal_probes: PathBuf::new(), capability_suite: PathBuf::new(),
            paired_prompts: PathBuf::new(), cvector_generator: PathBuf::from("llama-cvector-generator"),
            runtime: PathBuf::from("llama-cli"), server_runtime: None, gpu_layers: 0, device: None,
            refusal_tokens: 96, task_tokens: 128, strengths: vec![0.10, 0.20, 0.35], blocks: None,
            max_capability_loss: 0.02, max_refusal_rate: None, quantize: None,
            quantizer: PathBuf::from("llama-quantize"), allow_resource_risk: false }
    }

    pub fn with_auto_hardware(mut self) -> Self {
        if self.device.is_none() && self.gpu_layers == 0 {
            let (_, rt, layers) = crate::resource::resolve_auto_runtime_device("auto", None, 0);
            if let Some(rt_id) = rt {
                self.device = Some(rt_id);
                self.gpu_layers = layers as i32;
            }
        }
        self
    }
}

#[derive(Clone, Debug)]
struct Candidate {
    name: String,
    method: String,
    block: usize,
    strength: f64,
    edits: Vec<(usize, f64)>,
    preserve_row_norms: bool,
    staged: PathBuf,
}

pub struct UncensorAutoPlan {
    request: UncensorAutoRequest,
    source: PathBuf,
    output: PathBuf,
    control_vector_output: PathBuf,
    quantized_output: Option<PathBuf>,
    positive: PathBuf,
    negative: PathBuf,
    candidates: Vec<Candidate>,
    strengths: Vec<f64>,
    blocks: Vec<usize>,
}

fn paired_file(path: &Path) -> io::Result<usize> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > 2 * 1024 * 1024 {
        return Err(invalid("paired prompt files must be 1 byte..2 MiB each"));
    }
    let mut count = 0;
    for row in BufReader::new(File::open(path)?).lines() {
        let row = row?;
        if row.trim().is_empty() || row.len() > 8192 {
            return Err(invalid("prompt lines must be nonblank and <=8192 bytes"));
        }
        count += 1;
        if count > 256 { return Err(invalid("paired prompt files may have at most 256 lines")); }
    }
    if count < 2 { return Err(invalid("need at least two paired prompt lines")); }
    Ok(count)
}

fn attn_blocks(source: &Path) -> io::Result<Vec<usize>> {
    let mut blocks = inspect_gguf_tensors(source)?.into_iter().filter_map(|tensor| {
        let layer = tensor.name.strip_prefix("blk.")?.strip_suffix(".attn_output.weight")?;
        (tensor.shape.len() == 2).then(|| layer.parse::<usize>().ok()).flatten()
    }).collect::<Vec<_>>();
    blocks.sort_unstable(); blocks.dedup();
    if blocks.len() < 3 { return Err(invalid("no supported middle-layer attention output matrix")); }
    Ok(blocks)
}

fn default_sweep_blocks(blocks: &[usize]) -> Vec<usize> {
    let middle = blocks[blocks.len() * 2 / 3];
    [middle.wrapping_sub(1), middle, middle + 1].into_iter()
        .filter(|b| blocks.contains(b))
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter().collect()
}

pub fn prepare_uncensor_auto(mut request: UncensorAutoRequest) -> io::Result<UncensorAutoPlan> {
    if request.cvector_generator.as_os_str().is_empty() || !request.cvector_generator.is_file() {
        if let Some(tool) = crate::discovery::discovered_path(crate::discovery::BackendId::LlamaCvectorGenerator) {
            request.cvector_generator = tool;
        }
    }
    if !request.cvector_generator.is_file() {
        return Err(invalid("automatic uncensoring requires an installed llama-cvector-generator executable"));
    }
    if request.runtime.as_os_str().is_empty() || !request.runtime.is_file() {
        if let Some(tool) = crate::discovery::discovered_path(crate::discovery::BackendId::LlamaCli) {
            request.runtime = tool;
        }
    }
    if request.server_runtime.is_none() {
        request.server_runtime = crate::discovery::discovered_path(crate::discovery::BackendId::LlamaServer);
    }
    if request.server_runtime.as_ref().is_some_and(|p| !p.is_file()) {
        return Err(invalid("configured llama-server executable does not exist"));
    }
    if request.quantize.is_some() && (!request.quantizer.is_file() || request.quantizer.as_os_str().is_empty()) {
        if let Some(tool) = crate::discovery::discovered_path(crate::discovery::BackendId::LlamaQuantize) {
            request.quantizer = tool;
        }
    }
    let source = request.source.canonicalize()?;
    let info = inspect_gguf(&source)?;
    if !matches!(info.file_type, Some(0 | 1 | 32)) || info.split_count.unwrap_or(1) > 1 {
        return Err(invalid("uncensor sweep needs an original single-file F16/BF16/F32 GGUF"));
    }
    let output_dir = request.output_dir.canonicalize()?;
    if !output_dir.is_dir() { return Err(invalid("output folder must exist")); }
    if request.refusal_probes.as_os_str().is_empty() {
        if let Some(bundled) = crate::discovery::bundled_data_file("refusal-probes.json") {
            request.refusal_probes = bundled;
        }
    }
    if request.capability_suite.as_os_str().is_empty() {
        if let Some(bundled) = crate::discovery::bundled_data_file("sanity-tasks.json") {
            request.capability_suite = bundled;
        }
    }
    if request.refusal_probes.as_os_str().is_empty() || request.capability_suite.as_os_str().is_empty() {
        return Err(invalid("refusal probes and capability suite files are both required"));
    }
    // Validate both suites now so a broken file never reaches the sweep.
    crate::refusal::parse_refusal_suite(&request.refusal_probes)?;
    crate::task_eval::parse_eval_suite(&request.capability_suite)?;
    if request.paired_prompts.as_os_str().is_empty() {
        if let Some(bundled) = crate::discovery::bundled_data_dir("paired")
            .or_else(|| crate::discovery::bundled_data_file("positive.txt").and_then(|p| p.parent().map(Path::to_path_buf))) {
            request.paired_prompts = bundled;
        }
    }
    let folder = request.paired_prompts.canonicalize()?;
    let positive = folder.join("positive.txt"); let negative = folder.join("negative.txt");
    if paired_file(&positive)? != paired_file(&negative)? {
        return Err(invalid("positive.txt and negative.txt need the same number of lines"));
    }
    if !(1..=512).contains(&request.refusal_tokens) || !(1..=512).contains(&request.task_tokens)
        || !(-1..=512).contains(&request.gpu_layers) {
        return Err(invalid("token counts must be 1..=512 and gpu layers -1..=512"));
    }
    if let Some(d) = &request.device {
        if d.is_empty() || d.len() > 80 || !d.bytes().all(|b|
            b.is_ascii_alphanumeric() || b == b'_' || b == b'-' || b == b'/') {
            return Err(invalid("invalid device"));
        }
    }
    let strengths = if request.strengths.is_empty() { vec![0.10, 0.20, 0.35] } else { request.strengths.clone() };
    if strengths.len() > 6 { return Err(invalid("at most 6 strengths per sweep")); }
    for strength in &strengths {
        if !strength.is_finite() || *strength <= 0.0 || *strength > 1.0 {
            return Err(invalid("each strength must be >0 and <=1"));
        }
    }
    let available = attn_blocks(&source)?;
    let blocks = match &request.blocks {
        Some(list) => {
            if list.is_empty() || list.len() > 4 { return Err(invalid("1..=4 blocks per sweep")); }
            let mut sorted = list.clone(); sorted.sort_unstable(); sorted.dedup();
            for block in &sorted {
                if !available.contains(block) {
                    return Err(invalid(format!("block {block} has no supported attention output matrix")));
                }
            }
            sorted
        }
        None => default_sweep_blocks(&available),
    };
    let candidate_count = strengths.len().checked_mul(blocks.len())
        .ok_or_else(|| invalid("grid overflow"))?;
    if candidate_count > 12 { return Err(invalid("strength x block grid must produce at most 12 candidates")); }
    if !request.max_capability_loss.is_finite()
        || !(0.0..=1.0).contains(&request.max_capability_loss) {
        return Err(invalid("max capability loss must be a finite fraction 0..1"));
    }
    if let Some(rate) = request.max_refusal_rate {
        if !rate.is_finite() || !(0.0..=1.0).contains(&rate) {
            return Err(invalid("max refusal rate must be a finite fraction 0..1"));
        }
    }
    if let Some(precision) = request.quantize {
        if !matches!(precision, Precision::Q4KM | Precision::Q5KM | Precision::Q6K) {
            return Err(invalid("optional final quantization supports Q4_K_M, Q5_K_M or Q6_K"));
        }
    }
    let stem = source.file_stem().ok_or_else(|| invalid("source name missing"))?.to_string_lossy();
    let output = output_dir.join(format!("{stem}-uncensored.gguf"));
    if output.symlink_metadata().is_ok() { return Err(io::Error::new(ErrorKind::AlreadyExists, "uncensored output exists")); }
    if recipe_receipt(&output).symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "uncensored recipe receipt exists"));
    }
    let control_vector_output = output_dir.join(format!("{stem}-control-vector.gguf"));
    let quantized_output = match request.quantize {
        Some(precision) => {
            let path = output_dir.join(format!("{stem}-uncensored-{}.gguf", precision.as_str()));
            if path.symlink_metadata().is_ok() {
                return Err(io::Error::new(ErrorKind::AlreadyExists, "quantized output exists"));
            }
            Some(path)
        }
        None => None,
    };
    let mut candidates = Vec::new();
    for strength in &strengths {
        for block in &blocks {
            let name = format!("s{:03}-b{:03}", (strength * 1000.0).round() as i64, block);
            let staged = output_dir.join(format!("{stem}-uncensored-staged-{name}.gguf"));
            candidates.push(Candidate {
                name,
                method: "norm_preserving_ablation".into(),
                block: *block,
                strength: *strength,
                edits: vec![(*block, *strength)],
                preserve_row_norms: true,
                staged,
            });
        }
    }
    // When running automatic multi-block discovery (blocks was None) and grid has room (< 12),
    // also add standard_ablation (no row-norm rescaling) and multi_layer_heretic (tapered multi-block)
    // candidates so Ojon evaluates multiple editing methods individually.
    if request.blocks.is_none() && !strengths.is_empty() && !blocks.is_empty() {
        let mid_block = blocks[blocks.len() / 2];
        let mid_strength = strengths[strengths.len() / 2];
        if candidates.len() < 12 {
            let name = format!("std-s{:03}-b{:03}", (mid_strength * 1000.0).round() as i64, mid_block);
            let staged = output_dir.join(format!("{stem}-uncensored-staged-{name}.gguf"));
            candidates.push(Candidate {
                name,
                method: "standard_ablation".into(),
                block: mid_block,
                strength: mid_strength,
                edits: vec![(mid_block, mid_strength)],
                preserve_row_norms: false,
                staged,
            });
        }
        if blocks.len() >= 2 && candidates.len() < 12 {
            let multi_edits: Vec<(usize, f64)> = blocks
                .iter()
                .map(|&b| {
                    let s = if b == mid_block {
                        mid_strength
                    } else {
                        (mid_strength * 0.65).clamp(0.02, 1.0)
                    };
                    (b, s)
                })
                .collect();
            let name = format!("heretic-multi-s{:03}", (mid_strength * 1000.0).round() as i64);
            let staged = output_dir.join(format!("{stem}-uncensored-staged-{name}.gguf"));
            candidates.push(Candidate {
                name,
                method: "projected_multi_layer_heretic".into(),
                block: mid_block,
                strength: mid_strength,
                edits: multi_edits,
                preserve_row_norms: true,
                staged,
            });
        }
    }
    // The sweep keeps at most the current best and the candidate under test.
    // Quantization starts after the losing copy has been removed.
    let source_bytes = fs::metadata(&source)?.len();
    let needed = source_bytes as u128 * 2;
    if let Some(free) = crate::resource::free_disk(&output_dir) {
        if (needed + 2 * crate::resource::GIB as u128) > free as u128 {
            return Err(invalid(format!(
                "sweep needs ~{} MiB staged plus output; volume has {} MiB free",
                needed / (1024 * 1024), free / (1024 * 1024))));
        }
    } else if !request.allow_resource_risk {
        return Err(invalid("free disk space could not be measured; pass --allow-resource-risk to proceed"));
    }
    for candidate in &candidates {
        if candidate.staged.symlink_metadata().is_ok() {
            return Err(io::Error::new(ErrorKind::AlreadyExists,
                format!("staged candidate {} exists; remove it first", candidate.staged.display())));
        }
    }
    Ok(UncensorAutoPlan { request, source, output, control_vector_output, quantized_output, positive, negative,
        candidates, strengths, blocks })
}

impl UncensorAutoPlan {
    pub fn as_json(&self) -> Value {
        json!({"schema":"ojon.uncensor_auto_plan.v1",
            "source":self.source.to_string_lossy(),"output":self.output.to_string_lossy(),
            "detachable_control_vector":self.control_vector_output.to_string_lossy(),
            "quantized_output":self.quantized_output.as_ref().map(|p|p.to_string_lossy().to_string()),
            "refusal_probes":self.request.refusal_probes.to_string_lossy(),
            "capability_suite":self.request.capability_suite.to_string_lossy(),
            "candidate_count":self.candidates.len(),
            "grid":{"strengths":self.strengths,"blocks":self.blocks},
            "candidates":self.candidates.iter().map(|c|json!({
                "name":c.name,"method":c.method,"block":c.block,"strength":c.strength,
                "edits":c.edits,"preserve_row_norms":c.preserve_row_norms,
                "staged":c.staged.to_string_lossy()}))
                .collect::<Vec<_>>(),
            "max_capability_loss":self.request.max_capability_loss,
            "max_refusal_rate":self.request.max_refusal_rate,
            "final_quantization":self.request.quantize.as_ref().map(|p|p.as_str().to_owned()),
            "refusal_effect":"NOT_EVALUATED","quality_retention":"NOT_EVALUATED",
            "status":"PLAN_ONLY"})
    }
}

struct Stage(PathBuf);
impl Drop for Stage { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
struct CandidateCleanup(Vec<PathBuf>);
impl Drop for CandidateCleanup {
    fn drop(&mut self) {
        for path in &self.0 {
            let _ = fs::remove_file(path);
            let _ = fs::remove_file(recipe_receipt(path));
        }
    }
}
fn remove_candidate(path: &Path) -> io::Result<()> {
    if path.is_file() { fs::remove_file(path)?; }
    let receipt = recipe_receipt(path);
    if receipt.is_file() { fs::remove_file(receipt)?; }
    Ok(())
}
fn stage(parent: &Path) -> io::Result<Stage> {
    for retry in 0..16 {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH)
            .map_err(|_| invalid("invalid clock"))?.as_nanos();
        let path = parent.join(format!(".ojon-uncensor-{}-{stamp}-{retry}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(Stage(path)),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(invalid("cannot create sweep staging folder"))
}

/// Capability metrics are stored as percentage points 0..100 in the task
/// receipt; the measured-score chooser consumes fractions 0..1.
fn metric_fraction(metrics: &Value, key: &str) -> io::Result<f64> {
    let value = metrics.get(key).and_then(Value::as_f64)
        .ok_or_else(|| invalid(format!("capability receipt lacks measured {key}")))?;
    if !value.is_finite() || !(0.0..=100.0).contains(&value) {
        return Err(invalid(format!("measured {key} out of range")));
    }
    Ok(value / 100.0)
}

fn trial_scores(
    name: &str,
    method: Option<&str>,
    artifact: &Path,
    refusal_rate: f64,
    benign_overrefusal_rate: Option<f64>,
    metrics: &Value,
) -> io::Result<Value> {
    let mut obj = json!({"name":name,"artifact":artifact.to_string_lossy(),
        "bytes":fs::metadata(artifact)?.len(),
        "refusal_rate":refusal_rate,
        "benign_overrefusal_rate":benign_overrefusal_rate,
        "coding":metric_fraction(metrics,"coding_accuracy")?,
        "reasoning":metric_fraction(metrics,"reasoning_accuracy")?,
        "tools":metric_fraction(metrics,"tool_call_accuracy")?});
    if let Some(m) = method {
        obj["method"] = json!(m);
    }
    Ok(obj)
}

pub fn run_uncensor_auto(plan: &UncensorAutoPlan, mut log: impl FnMut(&str)) -> io::Result<Value> {
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "output appeared after planning"));
    }
    let parent = plan.output.parent().ok_or_else(|| invalid("output folder missing"))?;
    let staging = stage(parent)?;
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
    // 1. One measured control vector from the paired examples.
    let vector = staging.0.join("control-vector.gguf");
    let mut command = Command::new(&plan.request.cvector_generator);
    command.arg("-m").arg(&plan.source)
        .arg("-ngl").arg(plan.request.gpu_layers.max(0).to_string())
        .arg("--positive-file").arg(&plan.positive)
        .arg("--negative-file").arg(&plan.negative)
        .arg("--method").arg("mean").arg("-o").arg(&vector);
    log("Capturing paired model directions from GGUF");
    run_with_log(&mut command, |line| log(line))?;
    // Export detachable control-vector GGUF alongside the output if not already present.
    if !plan.control_vector_output.exists() {
        let _ = fs::copy(&vector, &plan.control_vector_output);
    }
    let ranked_blocks = crate::cvector_import::rank_control_vector_blocks(&vector, &plan.blocks)
        .unwrap_or_default()
        .into_iter()
        .map(|(b, norm)| json!({"block": b, "direction_l2_norm": norm}))
        .collect::<Vec<_>>();
    // 2. Baseline refusal measurement on the untouched original.
    let baseline_refusal_path = staging.0.join("baseline-refusal.json");
    let baseline_refusal_plan = prepare_refusal_eval_with_server(&plan.source,
        &plan.request.refusal_probes, &plan.request.runtime, plan.request.server_runtime.as_deref(),
        plan.request.gpu_layers, plan.request.device.as_deref(), plan.request.refusal_tokens,
        &baseline_refusal_path)?;
    log("Measuring refusal behaviour of the original model");
    let baseline_refusal = run_refusal_eval(&baseline_refusal_plan, &mut log)?;
    let baseline_refusal_rate = crate::refusal::refusal_rate_from_report(&baseline_refusal)?;
    let baseline_benign_overrefusal = baseline_refusal["benign_overrefusal_rate"].as_f64();
    // 3. Measure candidates one at a time. Retain only the current measured
    // winner, so a large-model sweep needs at most two edited copies on disk.
    let mut cleanup = CandidateCleanup(Vec::new());
    let mut capability: Option<Value> = None;
    let mut measured = Vec::new();
    let mut candidate_scores = Vec::new();
    let mut best_name: Option<String> = None;
    for (index, candidate) in plan.candidates.iter().enumerate() {
        let recipe = crate::cvector_import::recipe_from_control_vector_with_options(
            &vector, &candidate.edits, candidate.preserve_row_norms, &[],
        )?;
        let edit = prepare_gguf_abliteration_value(&plan.source, &recipe, &candidate.staged)?;
        log(&format!("Editing candidate {}/{} ({}, method={}): block {} strength {:.2}",
            index + 1, plan.candidates.len(), candidate.name, candidate.method, candidate.block, candidate.strength));
        run_gguf_abliteration(&edit, |line| log(line))?;
        cleanup.0.push(candidate.staged.clone());

        let capability_path = staging.0.join(format!("{}-capability.json", candidate.name));
        let capability_plan = prepare_eval_with_server(&plan.request.capability_suite,
            &plan.source, &[(candidate.name.clone(), candidate.staged.clone())],
            &plan.request.runtime, plan.request.server_runtime.as_deref(),
            plan.request.gpu_layers, plan.request.device.as_deref(), plan.request.task_tokens,
            &capability_path)?;
        log(&format!("Measuring held-out capability of candidate {}", candidate.name));
        let one = run_eval_reusing_baseline(&capability_plan, capability.as_ref(), &mut log)?;
        let metrics = one["candidates"][0]["metrics"].clone();
        if let Some(all) = capability.as_mut() {
            all["candidates"].as_array_mut()
                .ok_or_else(|| invalid("capability candidate list missing"))?
                .push(one["candidates"][0].clone());
        } else {
            capability = Some(one);
        }

        let report_path = staging.0.join(format!("{}-refusal.json", candidate.name));
        let refusal_plan = prepare_refusal_eval_with_server(&candidate.staged,
            &plan.request.refusal_probes, &plan.request.runtime, plan.request.server_runtime.as_deref(),
            plan.request.gpu_layers, plan.request.device.as_deref(), plan.request.refusal_tokens,
            &report_path)?;
        log(&format!("Measuring refusal behaviour of candidate {}", candidate.name));
        let report = run_refusal_eval(&refusal_plan, &mut log)?;
        let rate = crate::refusal::refusal_rate_from_report(&report)?;
        let benign_over = report["benign_overrefusal_rate"].as_f64();
        candidate_scores.push(trial_scores(&candidate.name, Some(&candidate.method),
            &candidate.staged, rate, benign_over, &metrics)?);
        measured.push((candidate, rate, benign_over));

        let baseline_metrics = &capability.as_ref()
            .ok_or_else(|| invalid("capability baseline missing"))?["baseline"]["metrics"];
        let scores = json!({"schema":"ojon.edit_method_scores.v1",
            "max_capability_loss":plan.request.max_capability_loss,
            "max_refusal_rate":plan.request.max_refusal_rate,
            "baseline":trial_scores("original", None, &plan.source, baseline_refusal_rate,
                baseline_benign_overrefusal, baseline_metrics)?,
            "candidates":candidate_scores});
        let provisional = choose_edit_method(&scores)?;
        let next_best = provisional.report["winner"]["method"].as_str().map(str::to_owned);
        if next_best != best_name {
            if let Some(previous) = best_name.as_ref() {
                let path = &plan.candidates.iter().find(|c| &c.name == previous)
                    .ok_or_else(|| invalid("previous winner missing"))?.staged;
                remove_candidate(path)?;
            }
            best_name = next_best;
        }
        if best_name.as_deref() != Some(candidate.name.as_str()) {
            remove_candidate(&candidate.staged)?;
        }
    }
    // 4. Final selection is recomputed from the complete measured score set.
    let capability = capability.ok_or_else(|| invalid("no capability measurements"))?;
    let baseline_metrics = capability["baseline"]["metrics"].clone();
    let scores = json!({"schema":"ojon.edit_method_scores.v1",
        "max_capability_loss":plan.request.max_capability_loss,
        "max_refusal_rate":plan.request.max_refusal_rate,
        "baseline":trial_scores("original", None, &plan.source, baseline_refusal_rate, baseline_benign_overrefusal, &baseline_metrics)?,
        "candidates":candidate_scores});
    let choice = choose_edit_method(&scores)?;
    let winner_name = choice.report["winner"]["method"].as_str().map(str::to_owned);
    if winner_name != best_name {
        return Err(invalid("incremental winner differs from final measured selection"));
    }
    // 7. Publish the winner; delete losers and staging.
    let mut kept = Vec::new();
    let mut removed = Vec::new();
    let mut published = false;
    let mut winner_method_kind = Value::Null;
    let mut winner_refusal_rate = None;
    let mut winner_metrics: Option<Value> = None;
    if let Some(winner) = winner_name.as_ref() {
        let candidate = plan.candidates.iter().find(|c| &c.name == winner)
            .ok_or_else(|| invalid("chooser selected an unknown candidate"))?;
        winner_method_kind = json!(candidate.method);
        winner_refusal_rate = measured.iter().find(|(c, _, _)| c.name == *winner).map(|(_, r, _)| *r);
        winner_metrics = capability["candidates"]
            .as_array()
            .and_then(|list| list.iter().find(|c| c["label"] == json!(candidate.name)))
            .and_then(|c| c.get("metrics"))
            .cloned();
        let staged_receipt = recipe_receipt(&candidate.staged);
        let published_receipt = recipe_receipt(&plan.output);
        fs::rename(&staged_receipt, &published_receipt)?;
        if let Err(error) = fs::rename(&candidate.staged, &plan.output) {
            let _ = fs::rename(&published_receipt, &staged_receipt);
            return Err(error);
        }
        published = true;
    }
    for candidate in &plan.candidates {
        if published && winner_name.as_deref() == Some(candidate.name.as_str()) {
            kept.push(candidate.name.clone());
        } else {
            let _ = fs::remove_file(&candidate.staged);
            let _ = fs::remove_file(recipe_receipt(&candidate.staged));
            removed.push(candidate.name.clone());
        }
    }
    let edited_status = if published { "PUBLISHED_FROM_MEASURED_WINNER" } else { "NOT_PUBLISHED_NO_MEASURED_WINNER" };
    // 8. Reload the saved edited model and retest independently.
    let edited_reload_verification = if published {
        log("Reloading saved edited model from disk for independent verification");
        match crate::smoke::smoke_gguf_capture_with_device(
            &plan.output,
            &plan.request.runtime,
            "/no_think\nReply with the single word READY.",
            24,
            plan.request.gpu_layers.max(0) as u32,
            plan.request.device.as_deref(),
        ) {
            Ok(cap) if crate::smoke::matches_single_word_reply(&cap.stdout, "READY")
                || cap.stdout.to_ascii_uppercase().contains("READY") => json!({
                "status": "RELOADED_AND_VERIFIED",
                "artifact": plan.output.to_string_lossy(),
                "file_bytes": fs::metadata(&plan.output).map(|m| m.len()).unwrap_or(0),
            }),

            Ok(cap) => json!({
                "status": "RELOAD_LOADED_NON_EXACT_PROBE",
                "artifact": plan.output.to_string_lossy(),
                "stdout_preview": cap.stdout.chars().take(80).collect::<String>(),
            }),
            Err(e) => json!({"status": "RELOAD_FAILED", "reason": e.to_string()}),
        }
    } else {
        json!({"status": "NOT_RUN_NO_PUBLISHED_WINNER"})
    };

    // 9. Optionally quantize the accepted edited teacher and verify the final
    //    quantized artifact against BOTH the original model and the edited teacher.
    let mut quantized_status = "NOT_REQUESTED".to_string();
    let mut quantized_verification = Value::Null;
    if published {
        if let Some(precision) = plan.request.quantize {
            let quant_out = plan.quantized_output.as_ref()
                .ok_or_else(|| invalid("quantized output path missing"))?;
            let spine_recipe = crate::adaptive::sensitive_spine_pinning_recipe(
                &inspect_gguf_tensors(&plan.output).unwrap_or_default(),
            );
            let mut quant = prepare_gguf(&plan.output, quant_out, &plan.request.quantizer,
                precision, threads)?;
            if !spine_recipe.is_empty() {
                let recipe_path = staging.0.join("quant-spine.tensor-types.txt");
                fs::write(&recipe_path, spine_recipe.as_bytes())?;
                quant = quant.with_tensor_type_file(&recipe_path)?;
            }
            log(&format!("Quantizing accepted edited teacher to {}", precision.as_str()));
            quantize_gguf_with_log(&quant, |line| log(line))?;

            log("Reloading quantized uncensored artifact and verifying against BOTH original and edited teacher");
            let q_topology = crate::gguf::inspect_model_topology(quant_out)
                .map(|t| t.as_json())
                .unwrap_or(Value::Null);
            let q_refusal_path = staging.0.join("quantized-refusal.json");
            let q_refusal_plan = prepare_refusal_eval_with_server(
                quant_out,
                &plan.request.refusal_probes,
                &plan.request.runtime,
                plan.request.server_runtime.as_deref(),
                plan.request.gpu_layers,
                plan.request.device.as_deref(),
                plan.request.refusal_tokens,
                &q_refusal_path,
            )?;
            let q_refusal_report = run_refusal_eval(&q_refusal_plan, &mut log)?;
            let q_refusal_rate = crate::refusal::refusal_rate_from_report(&q_refusal_report)?;

            let q_cap_path = staging.0.join("quantized-capability.json");
            let q_cap_plan = prepare_eval_with_server(
                &plan.request.capability_suite,
                &plan.source,
                &[(String::from("quantized_final"), quant_out.clone())],
                &plan.request.runtime,
                plan.request.server_runtime.as_deref(),
                plan.request.gpu_layers,
                plan.request.device.as_deref(),
                plan.request.task_tokens,
                &q_cap_path,
            )?;
            let q_cap_report = crate::task_eval::run_eval_reusing_baseline(
                &q_cap_plan,
                Some(&capability),
                &mut log,
            )?;
            let q_metrics = &q_cap_report["candidates"][0]["metrics"];
            let mut vs_original = Vec::new();
            let mut vs_teacher = Vec::new();
            let mut passes_both = true;
            let mut orig_total_cap = 0.0;
            let allowed_drop_pts = plan.request.max_capability_loss * 100.0;
            for metric in ["coding_accuracy", "reasoning_accuracy", "tool_call_accuracy"] {
                let orig_v = baseline_metrics[metric].as_f64().unwrap_or(0.0);
                orig_total_cap += orig_v;
                let teach_v = winner_metrics.as_ref().and_then(|m| m[metric].as_f64()).unwrap_or(orig_v);
                let quant_v = q_metrics[metric].as_f64().unwrap_or(0.0);
                let pass_orig = orig_v > 0.0 && quant_v + 1e-9 >= orig_v - allowed_drop_pts;
                let pass_teach = teach_v > 0.0 && quant_v + 1e-9 >= teach_v - allowed_drop_pts;

                if !pass_orig || !pass_teach {
                    passes_both = false;
                }
                vs_original.push(json!({
                    "metric": metric,
                    "original": orig_v,
                    "quantized": quant_v,
                    "pass": pass_orig
                }));
                vs_teacher.push(json!({
                    "metric": metric,
                    "edited_teacher": teach_v,
                    "quantized": quant_v,
                    "pass": pass_teach
                }));
            }
            if orig_total_cap <= 0.0 {
                passes_both = false;
            }
            let dual_gate_passed = passes_both && q_refusal_rate <= baseline_refusal_rate;
            quantized_status = if dual_gate_passed {
                "QUANTIZED_AND_DUAL_VERIFIED"
            } else {
                "QUANTIZED_DUAL_GATE_FAILED"
            }.to_string();
            quantized_verification = json!({
                "artifact": quant_out.to_string_lossy(),
                "file_bytes": fs::metadata(quant_out).map(|m| m.len()).unwrap_or(0),
                "topology": q_topology,
                "refusal_rate": q_refusal_rate,
                "original_refusal_rate": baseline_refusal_rate,
                "edited_teacher_refusal_rate": winner_refusal_rate,
                "benign_overrefusal_rate": q_refusal_report["benign_overrefusal_rate"],
                "capability_metrics": q_metrics,
                "vs_original": vs_original,
                "vs_edited_teacher": vs_teacher,
                "passes_dual_reference_gate": dual_gate_passed,
            });
        }
    }
    let result = json!({"schema":"ojon.uncensor_auto_result.v1",
        "source":plan.source.to_string_lossy(),
        "output_edited":if published {json!(plan.output.to_string_lossy())} else {Value::Null},
        "detachable_control_vector":if plan.control_vector_output.is_file() {
            Value::String(plan.control_vector_output.to_string_lossy().into_owned())
        } else { Value::Null },
        "output_quantized":if quantized_status == "QUANTIZED_AND_DUAL_VERIFIED" {
            plan.quantized_output.as_ref().map(|p|p.to_string_lossy().to_string())
        } else {None},
        "grid":{"strengths":plan.strengths,"blocks":plan.blocks},
        "discovered_refusal_directions":ranked_blocks,
        "baseline":{"model":plan.source.to_string_lossy(),
            "refusal_rate":baseline_refusal_rate,
            "benign_overrefusal_rate":baseline_benign_overrefusal,
            "capability_metrics":capability["baseline"]["metrics"]},
        "candidates":choice.report["tested"],
        "method_choice":choice.report,
        "selected_edit_strategy":winner_method_kind,
        "published_candidates":kept,"removed_candidates":removed,
        "edited_status":edited_status,
        "edited_reload_verification":edited_reload_verification,
        "quantization_status":quantized_status,
        "quantized_verification":quantized_verification,
        "refusal_measurement":"bounded_phrase_heuristic_v1",
        "quality_scope":"held-out exact/JSON functional tasks; not full intelligence parity",
        "status":if !published {"NO_QUALITY_PASSING_CANDIDATE"}
            else if quantized_status == "QUANTIZED_DUAL_GATE_FAILED" {"QUANTIZED_RESULT_FAILED_GATE"}
            else {"SELECTED_FROM_MEASURED_SCORES"},
        "limitation":"Selection compares only the supplied measured refusal and functional metrics. Phrase-based refusal detection and small functional suites cannot prove behavioural or intelligence equivalence."});
    let receipt_path = match &plan.quantized_output {
        Some(path) => path, None => &plan.output,
    };
    let mut receipt_name = receipt_path.as_os_str().to_os_string();
    receipt_name.push(".ojon-uncensor-auto.json");
    let mut receipt = OpenOptions::new().write(true).create_new(true).open(PathBuf::from(receipt_name))?;
    serde_json::to_writer_pretty(&mut receipt, &result)?;
    receipt.flush()?; receipt.sync_all()?;
    drop(staging);
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Seek;

    fn write_mock_model(path: &Path) {
        fn write_str(f: &mut File, value: &str) {
            f.write_all(&(value.len() as u64).to_le_bytes()).unwrap();
            f.write_all(value.as_bytes()).unwrap();
        }
        let mut f = File::create(path).unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&3u64.to_le_bytes()).unwrap(); // 3 tensors
        f.write_all(&2u64.to_le_bytes()).unwrap(); // 2 kv metadata
        write_str(&mut f, "general.architecture");
        f.write_all(&8u32.to_le_bytes()).unwrap();
        write_str(&mut f, "llama");
        write_str(&mut f, "general.file_type");
        f.write_all(&4u32.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap(); // 1 = F16
        // Tensor 0: blk.0.attn_output.weight
        write_str(&mut f, "blk.0.attn_output.weight");
        f.write_all(&2u32.to_le_bytes()).unwrap(); // 2 dims
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap(); // F16
        f.write_all(&0u64.to_le_bytes()).unwrap();
        // Tensor 1: blk.1.attn_output.weight
        write_str(&mut f, "blk.1.attn_output.weight");
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&32u64.to_le_bytes()).unwrap();
        // Tensor 2: blk.2.attn_output.weight
        write_str(&mut f, "blk.2.attn_output.weight");
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&64u64.to_le_bytes()).unwrap();

        let pos = f.stream_position().unwrap();
        let pad = ((32 - pos % 32) % 32) as usize;
        f.write_all(&vec![0u8; pad]).unwrap();
        f.write_all(&[0u8; 96]).unwrap(); // 96 bytes aligned data
    }

    #[test]
    fn uncensor_auto_plan_generates_grid_with_bundled_defaults() {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("ojon-uncensor-test-{}-{stamp}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let model = dir.join("test-f16.gguf");
        write_mock_model(&model);

        let mut request = UncensorAutoRequest::new(model, dir.clone());
        request.allow_resource_risk = true;
        // Test with explicit blocks and strengths
        request.strengths = vec![0.15, 0.30];
        request.blocks = Some(vec![1]);

        let plan = prepare_uncensor_auto(request).unwrap();
        assert_eq!(plan.candidates.len(), 2);
        assert_eq!(plan.strengths, vec![0.15, 0.30]);
        assert_eq!(plan.blocks, vec![1]);

        let json = plan.as_json();
        assert_eq!(json["schema"], "ojon.uncensor_auto_plan.v1");
        assert_eq!(json["status"], "PLAN_ONLY");
        assert_eq!(json["candidate_count"], 2);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn uncensor_auto_plan_rejects_invalid_parameters() {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("ojon-uncensor-err-test-{}-{stamp}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let model = dir.join("test-f16.gguf");
        write_mock_model(&model);

        // 1. Invalid strengths (> 1.0)
        let mut req_bad_strength = UncensorAutoRequest::new(model.clone(), dir.clone());
        req_bad_strength.allow_resource_risk = true;
        req_bad_strength.strengths = vec![1.5];
        assert!(prepare_uncensor_auto(req_bad_strength).is_err());

        // 2. Invalid tokens (0)
        let mut req_bad_tokens = UncensorAutoRequest::new(model.clone(), dir.clone());
        req_bad_tokens.allow_resource_risk = true;
        req_bad_tokens.refusal_tokens = 0;
        assert!(prepare_uncensor_auto(req_bad_tokens).is_err());

        // 3. Invalid blocks (> 4 blocks)
        let mut req_bad_blocks = UncensorAutoRequest::new(model, dir.clone());
        req_bad_blocks.allow_resource_risk = true;
        req_bad_blocks.blocks = Some(vec![0, 1, 2, 3, 4]);
        assert!(prepare_uncensor_auto(req_bad_blocks).is_err());

        let _ = fs::remove_dir_all(&dir);
    }
}

