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
use crate::task_eval::{prepare_eval_with_server, run_eval};
use serde_json::{json, Value};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn invalid(s: impl Into<String>) -> io::Error { io::Error::new(ErrorKind::InvalidInput, s.into()) }

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
}

#[derive(Clone, Debug)]
struct Candidate { name: String, block: usize, strength: f64, staged: PathBuf }

pub struct UncensorAutoPlan {
    request: UncensorAutoRequest,
    source: PathBuf,
    output: PathBuf,
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
    // Every staged candidate is a full edited copy of the source: refuse to
    // start when the volume cannot hold the whole sweep plus output.
    let source_bytes = fs::metadata(&source)?.len();
    let needed = source_bytes as u128 * (candidate_count as u128 + 1)
        + source_bytes as u128; // control-vector generation reads the original; edits read it too
    if let Some(free) = crate::resource::free_disk(&output_dir) {
        if (needed + 2 * crate::resource::GIB as u128) > free as u128 {
            return Err(invalid(format!(
                "sweep needs ~{} MiB staged plus output; volume has {} MiB free",
                needed / (1024 * 1024), free / (1024 * 1024))));
        }
    } else if !request.allow_resource_risk {
        return Err(invalid("free disk space could not be measured; pass --allow-resource-risk to proceed"));
    }
    let mut candidates = Vec::new();
    for strength in &strengths {
        for block in &blocks {
            let name = format!("s{:03}-b{:03}", (strength * 1000.0).round() as i64, block);
            let staged = output_dir.join(format!("{stem}-uncensored-staged-{name}.gguf"));
            candidates.push(Candidate { name, block: *block, strength: *strength, staged });
        }
    }
    for candidate in &candidates {
        if candidate.staged.symlink_metadata().is_ok() {
            return Err(io::Error::new(ErrorKind::AlreadyExists,
                format!("staged candidate {} exists; remove it first", candidate.staged.display())));
        }
    }
    Ok(UncensorAutoPlan { request, source, output, quantized_output, positive, negative,
        candidates, strengths, blocks })
}

impl UncensorAutoPlan {
    pub fn as_json(&self) -> Value {
        json!({"schema":"ojon.uncensor_auto_plan.v1",
            "source":self.source.to_string_lossy(),"output":self.output.to_string_lossy(),
            "quantized_output":self.quantized_output.as_ref().map(|p|p.to_string_lossy().to_string()),
            "refusal_probes":self.request.refusal_probes.to_string_lossy(),
            "capability_suite":self.request.capability_suite.to_string_lossy(),
            "candidate_count":self.candidates.len(),
            "grid":{"strengths":self.strengths,"blocks":self.blocks},
            "candidates":self.candidates.iter().map(|c|json!({
                "name":c.name,"block":c.block,"strength":c.strength,"staged":c.staged.to_string_lossy()}))
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

fn trial_scores(name: &str, artifact: &Path, refusal_rate: f64, metrics: &Value) -> io::Result<Value> {
    Ok(json!({"name":name,"artifact":artifact.to_string_lossy(),
        "bytes":fs::metadata(artifact)?.len(),
        "refusal_rate":refusal_rate,
        "coding":metric_fraction(metrics,"coding_accuracy")?,
        "reasoning":metric_fraction(metrics,"reasoning_accuracy")?,
        "tools":metric_fraction(metrics,"tool_call_accuracy")?}))
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
    // 2. Baseline refusal measurement on the untouched original.
    let baseline_refusal_path = staging.0.join("baseline-refusal.json");
    let baseline_refusal_plan = prepare_refusal_eval_with_server(&plan.source,
        &plan.request.refusal_probes, &plan.request.runtime, plan.request.server_runtime.as_deref(),
        plan.request.gpu_layers, plan.request.device.as_deref(), plan.request.refusal_tokens,
        &baseline_refusal_path)?;
    log("Measuring refusal behaviour of the original model");
    let baseline_refusal = run_refusal_eval(&baseline_refusal_plan, &mut log)?;
    let baseline_refusal_rate = crate::refusal::refusal_rate_from_report(&baseline_refusal)?;
    // 3. Stage every candidate as a real edited GGUF copy.
    for (index, candidate) in plan.candidates.iter().enumerate() {
        let recipe = crate::cvector_import::recipe_from_control_vector_blocks(&vector,
            &[(candidate.block, candidate.strength)])?;
        let edit = prepare_gguf_abliteration_value(&plan.source, &recipe, &candidate.staged)?;
        log(&format!("Editing candidate {}/{} ({}): block {} strength {:.2}",
            index + 1, plan.candidates.len(), candidate.name, candidate.block, candidate.strength));
        run_gguf_abliteration(&edit, |line| log(line))?;
    }
    // 4. Held-out functional capability for original + all candidates in one
    //    shared evaluation receipt.
    let capability_report = staging.0.join("capability.json");
    let capability_plan = prepare_eval_with_server(&plan.request.capability_suite,
        &plan.source,
        &plan.candidates.iter().map(|c| (c.name.clone(), c.staged.clone())).collect::<Vec<_>>(),
        &plan.request.runtime, plan.request.server_runtime.as_deref(),
        plan.request.gpu_layers, plan.request.device.as_deref(), plan.request.task_tokens,
        &capability_report)?;
    log("Measuring held-out functional capability for baseline and candidates");
    let capability = run_eval(&capability_plan, &mut log)?;
    let baseline_metrics = capability["baseline"]["metrics"].as_object()
        .ok_or_else(|| invalid("capability receipt lacks baseline metrics"))?
        .clone().into();
    // 5. Refusal measurement for every candidate.
    let mut measured = Vec::new();
    for candidate in &plan.candidates {
        let report_path = staging.0.join(format!("{}-refusal.json", candidate.name));
        let refusal_plan = prepare_refusal_eval_with_server(&candidate.staged,
            &plan.request.refusal_probes, &plan.request.runtime, plan.request.server_runtime.as_deref(),
            plan.request.gpu_layers, plan.request.device.as_deref(), plan.request.refusal_tokens,
            &report_path)?;
        log(&format!("Measuring refusal behaviour of candidate {}", candidate.name));
        let report = run_refusal_eval(&refusal_plan, &mut log)?;
        let rate = crate::refusal::refusal_rate_from_report(&report)?;
        measured.push((candidate, rate));
    }
    // 6. Selection from measured scores only.
    let mut candidate_scores = Vec::new();
    for (candidate, rate) in &measured {
        candidate_scores.push(trial_scores(&candidate.name, &candidate.staged, *rate, &capability["candidates"]
            .as_array().and_then(|list| list.iter().find(|c| c["label"] == json!(candidate.name)))
            .ok_or_else(|| invalid("capability receipt lacks a measured candidate"))?
            .get("metrics").ok_or_else(|| invalid("candidate metrics missing"))?.clone())?);
    }
    let scores = json!({"schema":"ojon.edit_method_scores.v1",
        "max_capability_loss":plan.request.max_capability_loss,
        "max_refusal_rate":plan.request.max_refusal_rate,
        "baseline":trial_scores("original", &plan.source, baseline_refusal_rate, &baseline_metrics)?,
        "candidates":candidate_scores});
    let choice = choose_edit_method(&scores)?;
    let winner_name = choice.report["winner"]["method"].as_str().map(str::to_owned);
    // 7. Publish the winner; delete losers and staging.
    let mut kept = Vec::new();
    let mut removed = Vec::new();
    let mut published = false;
    if let Some(winner) = winner_name.as_ref() {
        let candidate = plan.candidates.iter().find(|c| &c.name == winner)
            .ok_or_else(|| invalid("chooser selected an unknown candidate"))?;
        fs::rename(&candidate.staged, &plan.output)?;
        published = true;
    }
    for candidate in &plan.candidates {
        if published && winner_name.as_deref() == Some(candidate.name.as_str()) {
            kept.push(candidate.name.clone());
        } else {
            let _ = fs::remove_file(&candidate.staged);
            removed.push(candidate.name.clone());
        }
    }
    let edited_status = if published { "PUBLISHED_FROM_MEASURED_WINNER" } else { "NOT_PUBLISHED_NO_MEASURED_WINNER" };
    let mut quantized_status = "NOT_REQUESTED".to_string();
    if published {
        if let Some(precision) = plan.request.quantize {
            let quant_out = plan.quantized_output.as_ref()
                .ok_or_else(|| invalid("quantized output path missing"))?;
            let quant = prepare_gguf(&plan.output, quant_out, &plan.request.quantizer,
                precision.clone(), threads)?;
            log(&format!("Quantizing measured winner to {}", precision.as_str()));
            quantize_gguf_with_log(&quant, |line| log(line))?;
            quantized_status = "QUANTIZED_FROM_PUBLISHED_WINNER".to_string();
        }
    }
    let result = json!({"schema":"ojon.uncensor_auto_result.v1",
        "source":plan.source.to_string_lossy(),
        "output_edited":plan.output.to_string_lossy(),
        "output_quantized":plan.quantized_output.as_ref().map(|p|p.to_string_lossy().to_string()),
        "grid":{"strengths":plan.strengths,"blocks":plan.blocks},
        "baseline":{"model":plan.source.to_string_lossy(),
            "refusal_rate":baseline_refusal_rate,
            "capability_metrics":capability["baseline"]["metrics"]},
        "candidates":choice.report["tested"],
        "method_choice":choice.report,
        "published_candidates":kept,"removed_candidates":removed,
        "edited_status":edited_status,"quantization_status":quantized_status,
        "refusal_measurement":"bounded_phrase_heuristic_v1",
        "quality_scope":"held-out exact/JSON functional tasks; not full intelligence parity",
        "status":if published {"SELECTED_FROM_MEASURED_SCORES"}
            else {"NO_QUALITY_PASSING_CANDIDATE"},
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

