//! Executor for the KLD-gated recipe search (`recipe-plan` / `recipe-run`).
//!
//! Default is PLAN ONLY: it inspects the original GGUF, lists every probe the
//! run would quantize and measure, and writes `recipe-plan.json`. Executing
//! requires an explicit confirmation because a run quantizes the model once per
//! probe (tens of conversions for a large MoE) and holds one probe file at a
//! time on disk.
//!
//! Measurement protocol, all through public llama.cpp tools and the existing
//! Ojon adapters (`prepare_gguf`, `quantize_gguf_with_log`, `kld`):
//!   reference  = original logits over the corpus slice
//!   base       = every searched class pinned to `base` (a full tensor-type
//!                file, so the converter's own mixtures do not blur the measure)
//!   probe(c,l) = base with only class `c` at level `l`
//!   recipe     = greedy selection, quantized again and MEASURED again
//! The receipt reports `measured_within_budget` from that final measurement,
//! never from the prediction.
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::time::Instant;

use crate::gguf::{inspect_gguf, inspect_gguf_tensors, inspect_sharded_gguf, GgufTensor};
use crate::llama::{prepare_gguf, quantize_gguf_with_log, Precision};
use crate::quant::kld::{measure_kld, save_reference_logits, KldSettings, KldStats};
use crate::quant::recipe::{
    adopt_boundary, boundary_blocks, class_elements, ladder_index, probe_matrix, recipe_json, render_boundary_lines,
    select_recipe, Measured, Probe, Recipe, TensorClass, LADDER,
};

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, s.into())
}

/// Paths for receipts: canonicalize() on Windows yields `\\?\C:\...`, which is
/// correct for the OS but noise for a human reading JSON.
pub fn plain(path: &Path) -> String {
    let s = path.to_string_lossy();
    s.strip_prefix(r"\\?\").map(str::to_owned).unwrap_or_else(|| s.into_owned())
}

/// Ladder levels that can also be the converter's main format. The main format
/// only decides tensors outside the searched classes; everything searched is
/// pinned explicitly.
pub fn main_precision_for(base: &str) -> io::Result<Precision> {
    let name = match base {
        "q4_k" => "Q4_K_M",
        "q5_k" => "Q5_K_M",
        "q6_k" => "Q6_K",
        other => {
            return Err(invalid(format!(
                "base must be one of q4_k, q5_k, q6_k (got '{other}'); lower bases leave no room to measure"
            )))
        }
    };
    Precision::from_str(name).map_err(invalid)
}

#[derive(Debug, Clone)]
pub struct RecipeJob {
    pub source: PathBuf,
    pub output_dir: PathBuf,
    pub base: &'static str,
    pub floor: &'static str,
    pub budget_kld: f64,
    pub quantize_tool: PathBuf,
    pub imatrix: Option<PathBuf>,
    /// When set and `imatrix` is None, an importance matrix is computed first
    /// from the same corpus/context with this `llama-imatrix` executable and
    /// used for every quantization. Low-bit levels depend on it.
    pub imatrix_tool: Option<PathBuf>,
    pub threads: usize,
    pub kld: KldSettings,
    pub keep_probes: bool,
    /// Optional tail gate: the final recipe must also keep the 99th-percentile
    /// per-token KLD within `base_p99 + budget_p99`. Mean KLD hides rare but
    /// large divergences on individual tokens; this catches them.
    pub budget_p99: Option<f64>,
    /// Continue a run in an output directory that already holds a plan, the
    /// reference logits, the base candidate and a progress ledger. Finished
    /// probes are reused from `recipe-progress.jsonl`; failed ones are retried.
    pub resume: bool,
    /// Held-out text for the KLD measurements. When None, `kld.corpus` is split
    /// deterministically: the first half calibrates (imatrix), the second half
    /// evaluates. Measuring on the calibration text overstates quality.
    pub eval_corpus: Option<PathBuf>,
    /// After the recipe is measured, try one more file that keeps the first two
    /// and last two blocks of every lowered class at the base level; adopt it
    /// only if it measures better and is not more than 2% bigger.
    pub refine_boundaries: bool,
    /// Select the final recipe at this FILE size (bytes) instead of the KLD budget:
    /// the best measured recipe that fits, e.g. to compare with another quant at
    /// exactly its size. Probes are the same, so a resumed run can re-target cheaply.
    pub target_bytes: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct RecipePlan {
    pub job: RecipeJob,
    pub present: BTreeMap<TensorClass, u64>,
    pub boundary: Option<(Vec<u32>, Vec<u32>)>,
    pub probes: Vec<Probe>,
    pub source_bytes: u64,
    pub tensor_count: u64,
}

fn ladder_name(level: &str) -> io::Result<&'static str> {
    ladder_index(level)
        .map(|i| LADDER[i].0)
        .ok_or_else(|| invalid(format!("unknown ladder level '{level}'")))
}


/// The recipe search is defined for transformer language towers. A vision
/// projector (`mmproj`, architecture `clip`) or any GGUF without attention or
/// FFN blocks must be refused up front, not after an hour of probes: its
/// tensors would still match some class patterns by name.
pub fn not_a_language_model(architecture: Option<&str>, present: &BTreeMap<TensorClass, u64>) -> Option<String> {
    if matches!(architecture, Some("clip") | Some("mmproj")) {
        return Some(format!(
            "{} is a vision projector (architecture {}); run the recipe search on the language model GGUF and keep the projector as is",
            "this file", architecture.unwrap_or("?")));
    }
    let has_attention = present.contains_key(&TensorClass::AttnQk) || present.contains_key(&TensorClass::AttnVo);
    let has_ffn = [TensorClass::FfnGateUp, TensorClass::FfnDown, TensorClass::ExpertGateUp, TensorClass::ExpertDown]
        .iter().any(|c| present.contains_key(c));
    if !has_attention || !has_ffn {
        return Some("no transformer attention/FFN blocks recognised; the recipe search needs a language model GGUF".to_string());
    }
    None
}

/// Bytes below which the calibration text is too thin for an importance
/// matrix to mean much. Public dynamic-quant pipelines calibrate on hundreds
/// of thousands to over a million tokens of chat-style text; a few kilobytes
/// of README prose only demonstrates the mechanism.
pub const SMALL_CORPUS_BYTES: u64 = 1024 * 1024;

/// Deterministic half/half split on a paragraph boundary near the middle, so
/// calibration and evaluation never share text. Returns (calibration, eval).
pub fn split_corpus(text: &str) -> (String, String) {
    let mut mid = text.len() / 2;
    while mid < text.len() && !text.is_char_boundary(mid) {
        mid += 1;
    }
    let cut = text[..mid].rfind("\n\n").map(|i| i + 2)
        .or_else(|| text[..mid].rfind('\n').map(|i| i + 1))
        .unwrap_or(mid);
    let cut = if cut == 0 || cut >= text.len() { mid } else { cut };
    // never cut inside a UTF-8 sequence
    let mut c = cut;
    while c < text.len() && !text.is_char_boundary(c) {
        c += 1;
    }
    (text[..c].to_string(), text[c..].to_string())
}

/// Seed `out` from a finished recipe folder `previous`: copy the plan and the probe
/// ledger, and hard-link (copy when linking fails) the reference logits, imatrix, base
/// and level quants. Existing files in `out` are kept. Returns how many files were placed.
pub fn seed_from(previous: &Path, out: &Path) -> io::Result<usize> {
    fs::create_dir_all(out)?;
    let mut placed = 0;
    for entry in fs::read_dir(previous)? {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else { continue };
        let copy = name == "recipe-plan.json" || name == "recipe-progress.jsonl";
        let link = name == "reference.logits"
            || name.ends_with("imatrix.gguf")
            || (name.starts_with("base-") || name.starts_with("level-")) && name.ends_with(".gguf");
        let dest = out.join(name);
        if !(copy || link) || dest.exists() || !path.is_file() {
            continue;
        }
        if copy || fs::hard_link(&path, &dest).is_err() {
            fs::copy(&path, &dest)?;
        }
        placed += 1;
    }
    Ok(placed)
}

pub fn prepare_recipe(job: RecipeJob) -> io::Result<RecipePlan> {
    prepare_recipe_inner(job, false)
}

/// For `group-refine`, which runs in a FINISHED class-search directory (it reuses
/// that search's reference, base quant and probes, and writes group-* files only).
pub fn prepare_refine(job: RecipeJob) -> io::Result<RecipePlan> {
    prepare_recipe_inner(RecipeJob { resume: true, ..job }, true)
}

fn prepare_recipe_inner(job: RecipeJob, allow_finished: bool) -> io::Result<RecipePlan> {
    if job.threads == 0 {
        return Err(invalid("threads must be positive"));
    }
    if !(job.budget_kld.is_finite() && (0.0..=1.0).contains(&job.budget_kld)) {
        return Err(invalid("KLD budget must be a finite number in 0..=1 (typical: 0.005 strict, 0.02 relaxed)"));
    }
    main_precision_for(job.base)?;
    let floor_ix = ladder_index(job.floor).ok_or_else(|| invalid("unknown ladder floor"))?;
    if floor_ix >= ladder_index(job.base).unwrap() {
        return Err(invalid("floor must be below base"));
    }
    job.kld.validate()?;
    let source = job.source.canonicalize()?;
    // A split original is read through its first shard; llama-quantize,
    // llama-imatrix and llama-perplexity all follow the shards from there.
    let (info, tensors) = inspect_sharded_gguf(&source)?;
    if info.tensor_count == 0 {
        return Err(invalid("source GGUF has no tensors"));
    }
    if !matches!(info.file_type, Some(0 | 1 | 32)) {
        return Err(invalid(
            "recipe search requires an original F32/F16/BF16 GGUF; never requantize an already compressed model",
        ));
    }
    if !job.output_dir.is_dir() {
        return Err(invalid("output directory must exist"));
    }
    let output_dir = job.output_dir.canonicalize()?;
    for name in ["recipe-receipt.json", "recipe.gguf"] {
        if !allow_finished && output_dir.join(name).exists() {
            return Err(invalid(format!("{name} already exists in the output directory; this search is finished, use a new directory")));
        }
    }
    if !job.resume {
        for name in ["recipe-plan.json", "reference.logits", PROGRESS_FILE] {
            if output_dir.join(name).exists() {
                return Err(invalid(format!("{name} already exists in the output directory; use an empty directory or --resume")));
            }
        }
    }
    if let Some(im) = &job.imatrix {
        if !im.is_file() || im.metadata()?.len() == 0 {
            return Err(invalid("importance matrix must be a nonempty local file"));
        }
    }
    if let Some(tool) = &job.imatrix_tool {
        if tool.as_os_str().is_empty() {
            return Err(invalid("imatrix executable missing"));
        }
        if !job.resume && output_dir.join("calibration.imatrix.gguf").exists() {
            return Err(invalid("calibration.imatrix.gguf already exists in the output directory"));
        }
    }
    if let Some(ec) = &job.eval_corpus {
        if !ec.is_file() || ec.metadata()?.len() < 100 {
            return Err(invalid("eval corpus must be a local readable text file of at least 100 bytes"));
        }
        if ec.canonicalize()? == job.kld.corpus.canonicalize()? {
            return Err(invalid("eval corpus must differ from the calibration corpus; measuring on calibration text overstates quality"));
        }
    }
    if let Some(p99) = job.budget_p99 {
        if !(p99.is_finite() && (0.0..=10.0).contains(&p99)) {
            return Err(invalid("p99 budget must be a finite number in 0..=10"));
        }
    }
    let present = class_elements(&tensors);
    if let Some(reason) = not_a_language_model(info.architecture.as_deref(), &present) {
        return Err(invalid(reason));
    }
    let probes = probe_matrix(job.base, &present, job.floor)?;
    if probes.is_empty() {
        return Err(invalid("nothing to search: no ladder level between floor and base for the classes present"));
    }
    let boundary = boundary_blocks(&tensors);
    let job = RecipeJob { source, output_dir, ..job };
    Ok(RecipePlan { job, present, boundary, probes, source_bytes: info.file_bytes, tensor_count: info.tensor_count })
}

impl RecipePlan {
    pub fn plan_json(&self) -> Value {
        json!({
            "schema": "ojon.kld_recipe_plan.v1",
            "source": plain(&self.job.source),
            "source_bytes": self.source_bytes,
            "tensors": self.tensor_count,
            "output_dir": plain(&self.job.output_dir),
            "base_level": self.job.base,
            "floor_level": self.job.floor,
            "budget_kld_delta": self.job.budget_kld,
            "classes_present": self.present.iter().map(|(c, e)| {
                let searched = TensorClass::SEARCHED.contains(c);
                json!({"class": c.name(), "elements": e, "searched": searched,
                       "floor": if searched { Some(c.floor()) } else { None }})
            }).collect::<Vec<_>>(),
            "probes": self.probes.iter().map(|p| json!({"class": p.class.name(), "level": p.level})).collect::<Vec<_>>(),
            "quantizations_required": self.probes.len() + 2,
            "corpus": plain(&self.job.kld.corpus),
            "eval_corpus": self.job.eval_corpus.as_ref().map(|p| plain(p)),
            "corpus_policy": if self.job.eval_corpus.is_some() { "separate calibration and evaluation files" } else { "single file split in half: first half calibrates, second half evaluates" },
            "corpus_warning": if fs::metadata(&self.job.kld.corpus).map(|m| m.len() < SMALL_CORPUS_BYTES).unwrap_or(false) {
                Some(format!("calibration text under {} MiB; results demonstrate the mechanism, not a production-grade importance matrix", SMALL_CORPUS_BYTES / (1024 * 1024)))
            } else { None },
            "context": self.job.kld.context,
            "chunks": self.job.kld.chunks,
            "quantize_tool": plain(&self.job.quantize_tool),
            "quant_buffer_policy": "adaptive 32/64/128 MiB from live free RAM; requires llama-quantize --max-buffer-size",
            "perplexity_tool": plain(&self.job.kld.tool),
            "imatrix": self.job.imatrix.as_ref().map(|p| plain(p)),
            "imatrix_auto_tool": self.job.imatrix_tool.as_ref().map(|p| plain(p)),
            "budget_p99": self.job.budget_p99,
            "resume": self.job.resume,
            "refine_boundaries": self.job.refine_boundaries && self.boundary.is_some(),
            "boundary_blocks": self.boundary.as_ref().map(|(h, t)| json!({"head": h, "tail": t})),
            "execution": "plan_only",
            "claim": "none; every number in the receipt is measured, and KLD is a distribution distance, not a task score"
        })
    }

    pub fn write_plan(&self) -> io::Result<PathBuf> {
        let path = self.job.output_dir.join("recipe-plan.json");
        if self.job.resume && path.exists() {
            return Ok(path);
        }
        write_new(&path, &serde_json::to_vec_pretty(&self.plan_json())?)?;
        Ok(path)
    }
}

/// Full tensor-type file: every present searched class pinned explicitly.
pub fn render_full(levels: &BTreeMap<TensorClass, &'static str>) -> String {
    let mut out = String::new();
    for (class, level) in levels {
        if TensorClass::SEARCHED.contains(class) {
            out.push_str(&format!("{}={}\n", class.pattern(), level));
        }
    }
    out
}

fn write_new(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut f = fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    f.write_all(bytes)
}

fn probe_stem(class: TensorClass, level: &str) -> String {
    format!("probe-{}-{}", class.name(), level)
}

struct Quantized {
    path: PathBuf,
    bytes: u64,
    seconds: f64,
}

fn quantize_with<F>(plan: &RecipePlan, stem: &str, levels: &BTreeMap<TensorClass, &'static str>, log: &mut F) -> io::Result<Quantized>
where
    F: FnMut(&str),
{
    quantize_with_prefix(plan, stem, "", levels, log)
}

/// `prefix` lines go first in the tensor-type file; llama-quantize applies the
/// first matching pattern, so block-specific pins must precede class lines.
fn quantize_with_prefix<F>(plan: &RecipePlan, stem: &str, prefix: &str, levels: &BTreeMap<TensorClass, &'static str>, log: &mut F) -> io::Result<Quantized>
where
    F: FnMut(&str),
{
    let dir = &plan.job.output_dir;
    let tt_path = dir.join(format!("{stem}.tensor-types.txt"));
    let out_path = dir.join(format!("{stem}.gguf"));
    if out_path.exists() {
        if plan.job.resume && stem.starts_with("probe-") {
            // a probe the previous run never finished measuring: redo it
            fs::remove_file(&out_path)?;
        } else {
            return Err(invalid(format!("refusing to overwrite {}", out_path.display())));
        }
    }
    let mtp_pin = mtp_block_pin(&plan.job.source, plan.job.base)?;
    fs::write(&tt_path, format!("{mtp_pin}{prefix}{}", render_full(levels)))?;
    let started = Instant::now();
    let mut q = prepare_gguf(&plan.job.source, &out_path, &plan.job.quantize_tool, main_precision_for(plan.job.base)?, plan.job.threads)?
        .with_tensor_type_file(&tt_path)?;
    let slab = crate::resource::current_gguf_slab_mib();
    q = q.with_max_buffer_mib(slab)?;
    log(&format!("[ojon] adaptive CPU quantisation slab: {slab} MiB (the actual process may use more memory)"));
    if let Some(im) = &plan.job.imatrix {
        q = q.with_imatrix(im)?;
    }
    let done = quantize_gguf_with_log(&q, |line| log(line))?;
    Ok(Quantized { path: done.output, bytes: done.output_info.file_bytes, seconds: started.elapsed().as_secs_f64() })
}

/// Tensor-type lines that keep MTP (next-token prediction) blocks at the base level.
/// Those blocks (`<arch>.nextn_predict_layers` last of `<arch>.block_count`) never run
/// in a normal forward pass, so the imatrix has no data for them and llama-quantize
/// refuses very-low-bit types there ("Missing importance matrix for tensor
/// blk.64.attn_k.weight", Qwen3.8-27B). Their `nextn.*` tensors stay in the Mtp class.
/// First matching pattern wins, so these lines go first.
fn mtp_block_pin(source: &Path, base_level: &str) -> io::Result<String> {
    let meta = crate::models::gguf::inspect_gguf_metadata(source)?;
    Ok(mtp_pin_lines(
        meta.str("general.architecture").and_then(|arch| {
            Some((
                meta.int(&format!("{arch}.block_count"))?,
                meta.int(&format!("{arch}.nextn_predict_layers"))?,
            ))
        }),
        base_level,
    ))
}

fn mtp_pin_lines(blocks_and_nextn: Option<(i64, i64)>, base_level: &str) -> String {
    match blocks_and_nextn {
        Some((blocks, nextn)) if nextn > 0 && blocks > nextn => {
            let ids: Vec<String> = (blocks - nextn..blocks).map(|b| b.to_string()).collect();
            format!("^blk\\.({})\\.(?!nextn\\.).*\\.weight$={base_level}\n", ids.join("|"))
        }
        _ => String::new(),
    }
}

/// Build the probe "base + `class` at `level`" by splicing (see the probe loop).
#[allow(clippy::too_many_arguments)]
fn splice_probe<F>(
    plan: &RecipePlan,
    base_path: &Path,
    all_base: &BTreeMap<TensorClass, &'static str>,
    class: TensorClass,
    level: &'static str,
    stem: &str,
    level_quants: &mut BTreeMap<&'static str, PathBuf>,
    log: &mut F,
) -> io::Result<Quantized>
where
    F: FnMut(&str),
{
    if !level_quants.contains_key(level) {
        // Every class the plan probes at this level, set to it, in one quantization.
        let mut levels = all_base.clone();
        for p in plan.probes.iter().filter(|p| p.level == level) {
            levels.insert(p.class, level);
        }
        let level_stem = format!("level-{level}");
        let existing = plan.job.output_dir.join(format!("{level_stem}.gguf"));
        let path = if plan.job.resume && existing.exists() && crate::models::gguf::inspect_gguf(&existing).is_ok() {
            log(&format!("[ojon] resume: reusing level quant {}", plain(&existing)));
            existing
        } else {
            if existing.exists() {
                fs::remove_file(&existing)?;
            }
            log(&format!("[ojon] level quant: every probed class at {level} (spliced into each probe)"));
            quantize_with(plan, &level_stem, &levels, log)?.path
        };
        level_quants.insert(level, path);
    }
    let out = plan.job.output_dir.join(format!("{stem}.gguf"));
    if out.exists() {
        fs::remove_file(&out)?;
    }
    let started = Instant::now();
    let report = crate::quant::gguf_splice::splice(base_path, &level_quants[level], |t| {
        t.shape.len() >= 2 && TensorClass::of_name(&t.name) == class
    }, &out)?;
    log(&format!("[ojon]   spliced {} {} tensors from the {level} quant in {:.0} s",
        report.from_donor, class.name(), started.elapsed().as_secs_f64()));
    Ok(Quantized { path: out, bytes: report.bytes, seconds: started.elapsed().as_secs_f64() })
}

fn kld_json(s: &KldStats) -> Value {
    json!({"mean": s.mean, "mean_uncertainty": s.mean_uncertainty, "median": s.median, "p99": s.p99, "max": s.max, "same_top_p": s.same_top_p, "seconds": s.seconds})
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BudgetChecks {
    mean_within: bool,
    p99_within: Option<bool>,
}

impl BudgetChecks {
    fn accepted(self) -> bool {
        self.mean_within && self.p99_within.unwrap_or(true)
    }
}

/// Always evaluate the checks against the candidate actually selected for
/// publication. A boundary refinement must not leave stale receipt flags.
fn check_quality_budgets(
    base: &KldStats,
    candidate: &KldStats,
    budget_mean: f64,
    budget_p99: Option<f64>,
) -> BudgetChecks {
    let mean_within = candidate.mean <= base.mean + budget_mean + 1e-9;
    let p99_within = match (budget_p99, base.p99, candidate.p99) {
        (Some(budget), Some(base_p99), Some(final_p99)) => {
            Some(final_p99 <= base_p99 + budget + 1e-9)
        }
        (Some(_), _, _) => Some(false),
        (None, _, _) => None,
    };
    BudgetChecks { mean_within, p99_within }
}

fn should_adopt_refinement(
    plain: &KldStats,
    plain_bytes: u64,
    plain_accepted: bool,
    refined: &KldStats,
    refined_bytes: u64,
    refined_accepted: bool,
) -> bool {
    adopt_boundary(
        plain.mean, plain.mean_uncertainty.unwrap_or(0.0),
        refined.mean, refined.mean_uncertainty.unwrap_or(0.0),
        plain_bytes, refined_bytes,
    ) && (!plain_accepted || refined_accepted)
}

pub const PROGRESS_FILE: &str = "recipe-progress.jsonl";

/// Append-only ledger of finished steps, one JSON object per line, so a run
/// that dies after hours keeps every measured probe.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct Progress {
    pub base: Option<(u64, KldStats)>,
    pub probes: BTreeMap<(TensorClass, &'static str), (u64, KldStats)>,
}

fn kld_from_json(v: &Value) -> Option<KldStats> {
    Some(KldStats {
        mean: v.get("mean")?.as_f64()?,
        mean_uncertainty: v.get("mean_uncertainty").and_then(Value::as_f64),
        median: v.get("median").and_then(Value::as_f64),
        p99: v.get("p99").and_then(Value::as_f64),
        max: v.get("max").and_then(Value::as_f64),
        same_top_p: v.get("same_top_p").and_then(Value::as_f64),
        seconds: v.get("seconds").and_then(Value::as_f64).unwrap_or(0.0),
    })
}

/// Parse a ledger. Malformed lines fail the whole load: a half-written line
/// means the previous run died mid-write and the user should look, not trust.
pub fn parse_progress(text: &str) -> io::Result<Progress> {
    let mut out = Progress::default();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| invalid(format!("{PROGRESS_FILE} line {}: {e}", n + 1)))?;
        let bytes = v.get("bytes").and_then(Value::as_u64).ok_or_else(|| invalid(format!("{PROGRESS_FILE} line {}: missing bytes", n + 1)))?;
        let kld = v.get("kld").and_then(kld_from_json).ok_or_else(|| invalid(format!("{PROGRESS_FILE} line {}: missing kld", n + 1)))?;
        match v.get("kind").and_then(Value::as_str) {
            Some("base") => out.base = Some((bytes, kld)),
            Some("probe") => {
                let class = v.get("class").and_then(Value::as_str).and_then(TensorClass::parse)
                    .ok_or_else(|| invalid(format!("{PROGRESS_FILE} line {}: bad class", n + 1)))?;
                let level = v.get("level").and_then(Value::as_str).map(ladder_name).transpose()?
                    .ok_or_else(|| invalid(format!("{PROGRESS_FILE} line {}: bad level", n + 1)))?;
                out.probes.insert((class, level), (bytes, kld));
            }
            _ => return Err(invalid(format!("{PROGRESS_FILE} line {}: unknown kind", n + 1))),
        }
    }
    Ok(out)
}

fn load_progress(dir: &Path) -> io::Result<Progress> {
    let path = dir.join(PROGRESS_FILE);
    if !path.exists() {
        return Ok(Progress::default());
    }
    parse_progress(&fs::read_to_string(path)?)
}

fn block_of(name: &str) -> Option<u32> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

/// Contiguous layer groups over `n_layer` transformer blocks: group g covers
/// [g*n/G, (g+1)*n/G). Returns the group of `block`.
fn group_of(block: u32, n_layer: u32, groups: u32) -> u32 {
    ((u64::from(block) * u64::from(groups)) / u64::from(n_layer.max(1))).min(u64::from(groups - 1)) as u32
}

/// Per-layer-group allocation, the step after the per-class search. Runs in the class
/// search's output directory (reference logits, base quant, imatrix and class probes
/// present) and reuses its level quants. Each (class, layer group) is probed by
/// splicing just its tensors from a level quant; levels are then chosen per unit by
/// bytes saved per unit of KLD until `target_bytes` (file bytes), the file is
/// assembled by splicing several level quants and measured.
///
/// Units are measured one at a time against the base, so the selection assumes their
/// KLD effects add up; the final measurement of the assembled file is what counts.
pub fn run_group_refine<F>(plan: &RecipePlan, groups: u32, target_bytes: u64, mut log: F) -> io::Result<PathBuf>
where
    F: FnMut(&str),
{
    if !(2..=16).contains(&groups) {
        return Err(invalid("groups must be 2..=16"));
    }
    let dir = plan.job.output_dir.clone();
    let base_level = ladder_name(plan.job.base)?;
    let base_path = dir.join(format!("base-{base_level}.gguf"));
    let dump = dir.join("reference.logits");
    let progress = load_progress(&dir)?;
    let (Some((_, base_kld)), true, true) = (&progress.base, base_path.is_file(), dump.is_file()) else {
        return Err(invalid("group refinement needs a finished class search in this directory (base quant, reference.logits, progress)"));
    };
    let mut plan_used = plan.clone();
    plan_used.job.kld.corpus = match &plan.job.eval_corpus {
        Some(ec) => ec.clone(),
        None => dir.join("corpus-eval.txt"),
    };
    if plan_used.job.imatrix.is_none() {
        let auto = dir.join("calibration.imatrix.gguf");
        plan_used.job.imatrix = auto.is_file().then_some(auto);
    }
    let plan = &plan_used;

    let base_tensors = inspect_gguf_tensors(&base_path)?;
    let base_file_bytes = fs::metadata(&base_path)?.len();
    let mtp_blocks: Vec<u32> = base_tensors.iter().filter(|t| t.name.contains(".nextn.")).filter_map(|t| block_of(&t.name)).collect();
    let n_layer = base_tensors.iter().filter_map(|t| block_of(&t.name)).filter(|b| !mtp_blocks.contains(b)).max().map_or(0, |m| m + 1);
    if n_layer < groups {
        return Err(invalid("model has fewer blocks than groups"));
    }
    // unit key: (class, group); classes without blocks (token_embd, output) are one unit.
    let unit_of = |t: &GgufTensor| -> Option<(TensorClass, u32)> {
        if t.shape.len() < 2 {
            return None;
        }
        let class = TensorClass::of(t);
        if !TensorClass::SEARCHED.contains(&class) {
            return None;
        }
        match block_of(&t.name) {
            Some(b) if mtp_blocks.contains(&b) => None, // MTP block stays pinned at base
            Some(b) => Some((class, group_of(b, n_layer, groups))),
            None => Some((class, 0)),
        }
    };

    // Levels each class was measured at in the class search, highest precision first.
    let mut class_levels: BTreeMap<TensorClass, Vec<&'static str>> = BTreeMap::new();
    for (class, level) in progress.probes.keys() {
        class_levels.entry(*class).or_default().push(level);
    }
    for v in class_levels.values_mut() {
        v.sort_by_key(|l| std::cmp::Reverse(ladder_index(l).unwrap_or(0)));
    }
    let mut needed: Vec<&'static str> = class_levels.values().flatten().copied().collect();
    needed.sort_by_key(|l| std::cmp::Reverse(ladder_index(l).unwrap_or(0)));
    needed.dedup();

    // Level quants: every class measured at L set to L (reused when present).
    let mut level_paths: BTreeMap<&'static str, PathBuf> = BTreeMap::new();
    for level in &needed {
        let path = dir.join(format!("level-{level}.gguf"));
        if !(path.is_file() && inspect_gguf(&path).is_ok()) {
            let mut levels: BTreeMap<TensorClass, &'static str> = plan.present.keys().map(|c| (*c, base_level)).collect();
            for (class, ls) in &class_levels {
                if ls.contains(level) {
                    levels.insert(*class, level);
                }
            }
            log(&format!("[ojon] group refine: building level quant {level}"));
            if path.exists() {
                fs::remove_file(&path)?;
            }
            quantize_with(plan, &format!("level-{level}"), &levels, &mut log)?;
        }
        level_paths.insert(level, path);
    }

    // Exact byte cost of every unit at every level, from the real tensor types.
    let mut unit_bytes: BTreeMap<((TensorClass, u32), &'static str), u64> = BTreeMap::new();
    for t in &base_tensors {
        if let Some(u) = unit_of(t) {
            *unit_bytes.entry((u, base_level)).or_insert(0) += crate::quant::gguf_diff::tensor_bytes(t).unwrap_or(0);
        }
    }
    for (level, path) in &level_paths {
        for t in inspect_gguf_tensors(path)? {
            if let Some(u) = unit_of(&t) {
                if class_levels.get(&u.0).is_some_and(|ls| ls.contains(level)) {
                    *unit_bytes.entry((u, level)).or_insert(0) += crate::quant::gguf_diff::tensor_bytes(&t).unwrap_or(0);
                }
            }
        }
    }
    let units: Vec<(TensorClass, u32)> = {
        let mut u: Vec<_> = unit_bytes.keys().map(|(u, _)| *u).collect();
        u.sort();
        u.dedup();
        u
    };

    // Probe every (unit, level); resumable through group-progress.jsonl.
    let gp_path = dir.join("group-progress.jsonl");
    let mut measured: BTreeMap<((TensorClass, u32), &'static str), f64> = BTreeMap::new();
    if let Ok(text) = fs::read_to_string(&gp_path) {
        for line in text.lines() {
            let Ok(v) = serde_json::from_str::<Value>(line) else { continue };
            let (Some(c), Some(g), Some(l), Some(k)) = (
                v["class"].as_str().and_then(TensorClass::parse),
                v["group"].as_u64(),
                v["level"].as_str().and_then(|l| ladder_index(l).map(|i| LADDER[i].0)),
                v["kld_mean"].as_f64(),
            ) else { continue };
            measured.insert(((c, g as u32), l), k);
        }
    }
    let total = units.iter().map(|u| class_levels.get(&u.0).map_or(0, Vec::len)).sum::<usize>();
    let mut done = 0;
    for unit in &units {
        for level in class_levels.get(&unit.0).cloned().unwrap_or_default() {
            done += 1;
            if measured.contains_key(&(*unit, level)) {
                continue;
            }
            let probe = dir.join(format!("gprobe-{}-g{}-{level}.gguf", unit.0.name(), unit.1));
            if probe.exists() {
                fs::remove_file(&probe)?;
            }
            log(&format!("[ojon] group probe {done}/{total}: {} group {} at {level}", unit.0.name(), unit.1));
            let u = *unit;
            crate::quant::gguf_splice::splice(&base_path, &level_paths[level], |t| unit_of(t) == Some(u), &probe)?;
            let k = measure_kld(&probe, &dump, &plan.job.kld, &mut log);
            let _ = fs::remove_file(&probe);
            let k = k?;
            log(&format!("[ojon]   mean KLD {:.6} (delta {:+.6})", k.mean, k.mean - base_kld.mean));
            let mut f = fs::OpenOptions::new().create(true).append(true).open(&gp_path)?;
            writeln!(f, "{}", json!({"class": unit.0.name(), "group": unit.1, "level": level, "kld_mean": k.mean}))?;
            measured.insert((*unit, level), k.mean);
        }
    }

    // Greedy per unit: each step lowers one unit to its next measured level, best bytes
    // saved per KLD added first, until the predicted file fits the target.
    let delta = |u: (TensorClass, u32), l: &'static str| -> f64 {
        if l == base_level { 0.0 } else { (measured[&(u, l)] - base_kld.mean).max(0.0) }
    };
    let mut current: BTreeMap<(TensorClass, u32), &'static str> = units.iter().map(|u| (*u, base_level)).collect();
    let mut predicted_bytes = base_file_bytes;
    let mut predicted_kld = base_kld.mean;
    while predicted_bytes > target_bytes {
        let mut best: Option<(f64, (TensorClass, u32), &'static str, u64, f64)> = None;
        for u in &units {
            let cur = current[u];
            let ladder = class_levels.get(&u.0).cloned().unwrap_or_default();
            let next = ladder.iter().copied().find(|l| ladder_index(l) < ladder_index(cur) && measured.contains_key(&(*u, *l)));
            let Some(next) = next else { continue };
            let (Some(b_cur), Some(b_next)) = (unit_bytes.get(&(*u, cur)), unit_bytes.get(&(*u, next))) else { continue };
            let saved = b_cur.saturating_sub(*b_next);
            if saved == 0 {
                continue;
            }
            let added = (delta(*u, next) - delta(*u, cur)).max(0.0);
            let score = if added <= 0.0 { f64::INFINITY } else { saved as f64 / added };
            if best.as_ref().is_none_or(|b| score > b.0) {
                best = Some((score, *u, next, saved, added));
            }
        }
        let Some((_, u, next, saved, added)) = best else {
            return Err(invalid(format!("measured group probes cannot reach {target_bytes} bytes (smallest {predicted_bytes})")));
        };
        current.insert(u, next);
        predicted_bytes -= saved;
        predicted_kld += added;
    }

    // Assemble from the level quants and measure the real file.
    let used: Vec<&'static str> = {
        let mut v: Vec<_> = current.values().copied().filter(|l| *l != base_level).collect();
        v.sort();
        v.dedup();
        v
    };
    let donors: Vec<&Path> = used.iter().map(|l| level_paths[l].as_path()).collect();
    let out = dir.join("group-recipe.gguf");
    if out.exists() {
        fs::remove_file(&out)?;
    }
    crate::quant::gguf_splice::splice_many(&base_path, &donors, |t| {
        let u = unit_of(t)?;
        let level = current.get(&u)?;
        used.iter().position(|l| l == level)
    }, &out)?;
    let final_bytes = fs::metadata(&out)?.len();
    let final_kld = measure_kld(&out, &dump, &plan.job.kld, &mut log)?;
    log(&format!("[ojon] group recipe: {final_bytes} bytes, mean KLD {:.6} (predicted {:.6}; base {:.6})", final_kld.mean, predicted_kld, base_kld.mean));
    let receipt = json!({
        "schema": "ojon.group_recipe.v1",
        "groups": groups,
        "n_layer": n_layer,
        "target_bytes": target_bytes,
        "base": {"level": base_level, "bytes": base_file_bytes, "kld": kld_json(base_kld)},
        "units": current.iter().map(|((c, g), l)| json!({"class": c.name(), "group": g, "level": l})).collect::<Vec<_>>(),
        "predicted": {"bytes": predicted_bytes, "kld_mean": predicted_kld},
        "final": {"path": plain(&out), "bytes": final_bytes, "kld": kld_json(&final_kld)},
        "claim": "none; unit effects were measured one at a time and assumed additive; the final row is the measurement",
    });
    let receipt_path = dir.join("group-receipt.json");
    fs::write(&receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    Ok(receipt_path)
}

fn append_progress(dir: &Path, entry: &Value) -> io::Result<()> {
    let mut f = fs::OpenOptions::new().create(true).append(true).open(dir.join(PROGRESS_FILE))?;
    f.write_all(serde_json::to_string(entry)?.as_bytes())?;
    f.write_all(b"\n")
}

/// A budget is relative to the base, so a base that already moved the
/// distribution a lot makes "within budget" a weak statement. Say so.
pub fn base_advisory(base_level: &str, base_mean_kld: f64, budget_kld: f64) -> Option<String> {
    if base_mean_kld > 0.05 || (budget_kld > 0.0 && base_mean_kld > 5.0 * budget_kld) {
        let higher = match base_level { "q4_k" => "q5_k", "q5_k" => "q6_k", _ => "q8_0" };
        Some(format!(
            "the all-{base_level} base already measures mean KLD {base_mean_kld:.4}, {:.0}x the budget of {budget_kld};              the recipe is judged relative to that base, so consider --base {higher}, a larger/held-out corpus, or more --chunks before trusting the result",
            if budget_kld > 0.0 { base_mean_kld / budget_kld } else { f64::INFINITY }))
    } else {
        None
    }
}

/// llama-perplexity reports the mean KLD with a standard error. A budget that
/// is not comfortably above twice that error cannot separate a real step from
/// sampling noise; probes then come out below the base by chance.
pub fn noise_advisory(base_uncertainty: Option<f64>, budget_kld: f64, chunks: u32) -> Option<String> {
    let unc = base_uncertainty?;
    if unc <= 0.0 || budget_kld >= 2.0 * unc {
        return None;
    }
    let want = if chunks == 0 { 0 } else { ((2.0 * unc / budget_kld).powi(2) * chunks as f64).ceil() as u32 };
    Some(format!(
        "the base KLD standard error is {unc:.4}, so a budget of {budget_kld} is inside measurement noise (need > {:.4});          increase --chunks (about {} for this budget) or raise the budget before reading per-class deltas as real",
        2.0 * unc, if want == 0 { "all available".to_string() } else { want.to_string() }))
}

/// Run the whole search. Returns the receipt path. Every step streams to `log`.
pub fn run_recipe<F>(plan: &RecipePlan, mut log: F) -> io::Result<PathBuf>
where
    F: FnMut(&str),
{
    let dir = plan.job.output_dir.clone();
    let started = Instant::now();
    let base_level = ladder_name(plan.job.base)?;

    // Optional calibration first: an importance matrix from the same corpus makes
    // every low-bit level materially better, so a search without one would
    // understate how low the classes can go.
    let mut plan_used = plan.clone();
    let mut imatrix_auto: Option<PathBuf> = None;
    // Calibration and evaluation text must not overlap.
    let (calibration_corpus, eval_corpus) = match &plan.job.eval_corpus {
        Some(ec) => (plan.job.kld.corpus.clone(), ec.clone()),
        None => {
            let cal_path = dir.join("corpus-calibration.txt");
            let eval_path = dir.join("corpus-eval.txt");
            if !(plan.job.resume && cal_path.is_file() && eval_path.is_file()) {
                let text = fs::read_to_string(&plan.job.kld.corpus)?;
                let (cal, eval) = split_corpus(&text);
                if cal.len() < 100 || eval.len() < 100 {
                    return Err(invalid("corpus too small to split into calibration and evaluation halves; pass --eval-corpus"));
                }
                fs::write(&cal_path, cal)?;
                fs::write(&eval_path, eval)?;
            }
            log("[ojon] corpus split: first half calibrates, second half evaluates (no overlap)");
            (cal_path, eval_path)
        }
    };
    plan_used.job.kld.corpus = eval_corpus.clone();
    if plan.job.imatrix.is_none() {
        if let Some(tool) = &plan.job.imatrix_tool {
            let out = dir.join("calibration.imatrix.gguf");
            if plan.job.resume && out.is_file() {
                log(&format!("[ojon] resume: reusing importance matrix {}", plain(&out)));
                plan_used.job.imatrix = Some(out.clone());
                imatrix_auto = Some(out);
            } else {
            log(&format!("[ojon] importance matrix from the corpus with {}", plain(tool)));
            let im_plan = crate::quant::calibration::prepare_imatrix(
                &plan.job.source, &calibration_corpus, &out, tool,
                if plan.job.kld.fit { crate::quant::calibration::GPU_LAYERS_FIT } else { plan.job.kld.gpu_layers },
                plan.job.kld.context)?;
            let mut im_plan = im_plan;
            im_plan.device = plan.job.kld.device.clone();
            let saved = crate::quant::calibration::run_imatrix(&im_plan, |line| log(line))?;
            plan_used.job.imatrix = Some(saved.clone());
            imatrix_auto = Some(saved);
            }
        }
    }
    let plan = &plan_used;
    let progress = if plan.job.resume { load_progress(&dir)? } else { Progress::default() };

    let dump = if plan.job.resume && dir.join("reference.logits").is_file() {
        log("[ojon] resume: reusing reference.logits");
        dir.join("reference.logits")
    } else {
        log(&format!("[ojon] reference logits for {} ({} chunks of {} ctx)", plain(&plan.job.source), plan.job.kld.chunks, plan.job.kld.context));
        save_reference_logits(&plan.job.source, &dir.join("reference.logits"), &plan.job.kld, &mut log)?
    };

    let all_base: BTreeMap<TensorClass, &'static str> = plan.present.keys().map(|c| (*c, base_level)).collect();
    let base_path = dir.join(format!("base-{base_level}.gguf"));
    let (base, base_kld) = match (&progress.base, base_path.is_file()) {
        (Some((bytes, kld)), true) if plan.job.resume => {
            log(&format!("[ojon] resume: reusing base candidate {}", plain(&base_path)));
            (Quantized { path: base_path.clone(), bytes: *bytes, seconds: 0.0 }, kld.clone())
        }
        _ => {
            log(&format!("[ojon] base candidate: every class at {base_level}"));
            let base = quantize_with(plan, &format!("base-{base_level}"), &all_base, &mut log)?;
            let kld = measure_kld(&base.path, &dump, &plan.job.kld, &mut log)?;
            append_progress(&dir, &json!({"kind": "base", "level": base_level, "bytes": base.bytes, "kld": kld_json(&kld)}))?;
            (base, kld)
        }
    };
    log(&format!("[ojon] base: {:.2}% of source, mean KLD {:.6}", 100.0 * base.bytes as f64 / plan.source_bytes as f64, base_kld.mean));
    let base_advisory = base_advisory(base_level, base_kld.mean, plan.job.budget_kld);
    if let Some(note) = &base_advisory {
        log(&format!("[ojon] advisory: {note}"));
    }
    let noise_advisory = noise_advisory(base_kld.mean_uncertainty, plan.job.budget_kld, plan.job.kld.chunks);
    if let Some(note) = &noise_advisory {
        log(&format!("[ojon] advisory: {note}"));
    }

    let mut rows = Vec::new();
    let mut measured = Vec::new();
    // Probe by splicing: one quantization per level (every class probed at that level
    // set to it), then each probe copies just its class out of that level quant onto
    // the base quant. llama-quantize quantizes each tensor independently under a
    // tensor-type file, so the bytes equal a full requantization; it replaces one
    // ~15 min 27B requantization per probe with one per level. OJON_SPLICE_PROBES=0
    // restores the per-probe requantization.
    let splice_probes = std::env::var("OJON_SPLICE_PROBES").map_or(true, |v| v.trim() != "0");
    let mut level_quants: BTreeMap<&'static str, PathBuf> = BTreeMap::new();
    // OJON_MEASURED_ONLY=1 with --resume: no new probes; the recipe is chosen from the
    // probes already in the ledger (unmeasured levels are simply never lowered to).
    let measured_only = plan.job.resume && std::env::var("OJON_MEASURED_ONLY").is_ok_and(|v| v.trim() == "1");
    for (i, probe) in plan.probes.iter().enumerate() {
        let mut levels = all_base.clone();
        levels.insert(probe.class, probe.level);
        let stem = probe_stem(probe.class, probe.level);
        if let Some((bytes, k)) = progress.probes.get(&(probe.class, probe.level)) {
            log(&format!("[ojon] probe {}/{}: {} at {} (resumed: mean KLD {:.6})", i + 1, plan.probes.len(), probe.class.name(), probe.level, k.mean));
            rows.push(json!({"class": probe.class.name(), "level": probe.level, "status": "measured", "resumed": true, "bytes": bytes,
                             "kld": kld_json(k), "kld_delta_vs_base": k.mean - base_kld.mean}));
            measured.push(Measured { class: probe.class, level: probe.level, mean_kld: k.mean });
            continue;
        }
        if measured_only {
            // Cut a long search short: select from what is already measured.
            log(&format!("[ojon] probe {}/{}: {} at {} (skipped: OJON_MEASURED_ONLY)", i + 1, plan.probes.len(), probe.class.name(), probe.level));
            continue;
        }
        log(&format!("[ojon] probe {}/{}: {} at {}", i + 1, plan.probes.len(), probe.class.name(), probe.level));
        let built = if splice_probes {
            splice_probe(plan, &base.path, &all_base, probe.class, probe.level, &stem, &mut level_quants, &mut log)
        } else {
            quantize_with(plan, &stem, &levels, &mut log)
        };
        let outcome = built.and_then(|q| {
            let k = measure_kld(&q.path, &dump, &plan.job.kld, &mut log)?;
            Ok((q, k))
        });
        match outcome {
            Ok((q, k)) => {
                log(&format!("[ojon]   mean KLD {:.6} (delta {:+.6}), {:.2}% of source", k.mean, k.mean - base_kld.mean, 100.0 * q.bytes as f64 / plan.source_bytes as f64));
                rows.push(json!({"class": probe.class.name(), "level": probe.level, "status": "measured", "bytes": q.bytes,
                                 "quantize_seconds": q.seconds, "kld": kld_json(&k), "kld_delta_vs_base": k.mean - base_kld.mean}));
                measured.push(Measured { class: probe.class, level: probe.level, mean_kld: k.mean });
                append_progress(&dir, &json!({"kind": "probe", "class": probe.class.name(), "level": probe.level,
                                              "bytes": q.bytes, "quantize_seconds": q.seconds, "kld": kld_json(&k)}))?;
                if !plan.job.keep_probes {
                    let _ = fs::remove_file(&q.path);
                }
            }
            Err(e) => {
                log(&format!("[ojon]   probe failed: {e}"));
                rows.push(json!({"class": probe.class.name(), "level": probe.level, "status": "failed", "error": e.to_string()}));
                let _ = fs::remove_file(dir.join(format!("{stem}.gguf")));
            }
        }
    }

    if !plan.job.keep_probes {
        for path in level_quants.values() {
            let _ = fs::remove_file(path);
        }
    }

    let recipe: Recipe = match plan.job.target_bytes {
        Some(target_file_bytes) => {
            // Recipe bytes are nominal weight bytes; the file also holds untouched
            // tensors and metadata. Convert with the base quant's measured overhead.
            let base_nominal = select_recipe(plan.job.base, &plan.present, base_kld.mean, &[], 0.0)?.base_bytes;
            let overhead = base.bytes.saturating_sub(base_nominal);
            let target_nominal = target_file_bytes.saturating_sub(overhead);
            log(&format!("[ojon] size target: {target_file_bytes} file bytes (~{target_nominal} weight bytes after {overhead} of non-recipe data)"));
            crate::quant::recipe::select_recipe_for_size(plan.job.base, &plan.present, base_kld.mean, &measured, target_nominal)?
        }
        None => select_recipe(plan.job.base, &plan.present, base_kld.mean, &measured, plan.job.budget_kld)?,
    };
    log(&format!("[ojon] selected recipe: {} steps, predicted KLD delta {:.6}, predicted {:.1}% of base bytes",
        recipe.steps.len(), recipe.predicted_kld_delta, 100.0 * recipe.predicted_bytes as f64 / recipe.base_bytes.max(1) as f64));
    let final_q = quantize_with(plan, "recipe", &recipe.levels, &mut log)?;
    let final_kld = measure_kld(&final_q.path, &dump, &plan.job.kld, &mut log)?;
    let mut checks = check_quality_budgets(
        &base_kld, &final_kld, plan.job.budget_kld, plan.job.budget_p99,
    );
    let within = checks.accepted();
    log(&format!("[ojon] recipe: {:.2}% of source, mean KLD {:.6} (base {:.6}, budget +{:.6}){} -> {}",
        100.0 * final_q.bytes as f64 / plan.source_bytes as f64, final_kld.mean, base_kld.mean, plan.job.budget_kld,
        match (plan.job.budget_p99, final_kld.p99) {
            (Some(b), Some(p)) => format!(", p99 {:.6} (budget +{:.6})", p, b),
            _ => String::new(),
        },
        if within { "within budget" } else { "OVER budget; not accepted" }));
    if !within {
        if let Some(target) = plan.job.target_bytes {
            let added = final_kld.mean - base_kld.mean;
            log(&format!(
                "[ojon] advisory: the size target ({target} bytes) forced this recipe {:.1}x over the KLD budget (+{:.6} vs +{:.6}); raise --target-bytes, or drop it to get the smallest recipe within budget",
                added / plan.job.budget_kld, added, plan.job.budget_kld
            ));
        }
    }

    // Boundary refinement: one more quantization, adopted only on measurement.
    let mut final_q = final_q;
    let mut final_kld = final_kld;
    let mut within = within;
    let mut boundary_json = Value::Null;
    if plan.job.refine_boundaries && !recipe.steps.is_empty() {
        if let Some((head, tail)) = &plan.boundary {
            let blocks: Vec<u32> = head.iter().chain(tail.iter()).copied().collect();
            let prefix = render_boundary_lines(&recipe, base_level, &blocks);
            if !prefix.is_empty() {
                log(&format!("[ojon] boundary refinement: blocks {:?} of every lowered class back to {base_level}", blocks));
                match quantize_with_prefix(plan, "recipe-boundary", &prefix, &recipe.levels, &mut log)
                    .and_then(|q| measure_kld(&q.path, &dump, &plan.job.kld, &mut log).map(|k| (q, k)))
                {
                    Ok((bq, bk)) => {
                        let refined_checks = check_quality_budgets(
                            &base_kld, &bk, plan.job.budget_kld, plan.job.budget_p99,
                        );
                        // Never replace a passing plain recipe with a refinement
                        // that fails its tail or mean quality gate.
                        let adopt = should_adopt_refinement(
                            &final_kld, final_q.bytes, within,
                            &bk, bq.bytes, refined_checks.accepted(),
                        );
                        log(&format!("[ojon]   boundary file: mean KLD {:.6} vs {:.6}, {:.2}% vs {:.2}% of source -> {}",
                            bk.mean, final_kld.mean, 100.0 * bq.bytes as f64 / plan.source_bytes as f64,
                            100.0 * final_q.bytes as f64 / plan.source_bytes as f64, if adopt { "adopted" } else { "kept plain recipe" }));
                        boundary_json = json!({"blocks": blocks, "path": plain(&bq.path), "bytes": bq.bytes, "kld": kld_json(&bk), "adopted": adopt});
                        if adopt {
                            // the adopted file becomes recipe.gguf; the plain one is kept for inspection
                            let plain_path = dir.join("recipe-plain.gguf");
                            fs::rename(&final_q.path, &plain_path)?;
                            fs::rename(&bq.path, &final_q.path)?;
                            checks = refined_checks;
                            within = checks.accepted();
                            final_q = Quantized { path: final_q.path.clone(), bytes: bq.bytes, seconds: bq.seconds };
                            final_kld = bk;
                            boundary_json["plain_recipe_kept_as"] = json!(plain(&plain_path));
                        }
                    }
                    Err(e) => {
                        log(&format!("[ojon]   boundary refinement failed: {e}"));
                        boundary_json = json!({"blocks": blocks, "status": "failed", "error": e.to_string()});
                    }
                }
            }
        }
    }

    let receipt = json!({
        "schema": "ojon.kld_recipe_receipt.v1",
        "plan": plan.plan_json(),
        "reference_logits": plain(&dump),
        "base": {"level": base_level, "path": plain(&base.path), "bytes": base.bytes, "kld": kld_json(&base_kld)},
        "probes": rows,
        "recipe": recipe_json(&recipe, plan.job.base, base_kld.mean, plan.job.budget_kld),
        "final": {"path": plain(&final_q.path), "bytes": final_q.bytes,
                  "size_ratio_vs_source": final_q.bytes as f64 / plan.source_bytes as f64,
                  "size_ratio_vs_base": final_q.bytes as f64 / base.bytes.max(1) as f64,
                  "kld": kld_json(&final_kld), "kld_delta_vs_base": final_kld.mean - base_kld.mean,
                  "mean_within_budget": checks.mean_within,
                  "p99_budget": plan.job.budget_p99,
                  "p99_within_budget": checks.p99_within,
                  "measured_within_budget": within},
        "boundary_refinement": boundary_json,
        "base_advisory": base_advisory,
        "noise_advisory": noise_advisory,
        "imatrix": plan.job.imatrix.as_ref().map(|p| plain(p)),
        "imatrix_computed_by_this_run": imatrix_auto.as_ref().map(|p| plain(p)),
        "calibration_corpus": plain(&calibration_corpus),
        "eval_corpus": plain(&eval_corpus),
        "elapsed_seconds": started.elapsed().as_secs_f64(),
        "claim": if within {
            "measured mean KLD within budget on this corpus slice; run the task suite before any capability claim"
        } else {
            "rejected: measured mean KLD exceeds budget; the base candidate remains the accepted output"
        }
    });
    let path = dir.join("recipe-receipt.json");
    write_new(&path, &serde_json::to_vec_pretty(&receipt)?)?;
    Ok(path)
}

/// Read the per-class levels and base out of a `recipe-receipt.json` (or a
/// bare `ojon.kld_recipe.v1` object), so a measured recipe can be reused on
/// another original of the same family without re-probing.
pub fn levels_from_receipt(doc: &Value) -> io::Result<(&'static str, BTreeMap<TensorClass, &'static str>)> {
    let recipe = if doc.get("schema").and_then(Value::as_str) == Some("ojon.kld_recipe_receipt.v1") {
        doc.get("recipe").ok_or_else(|| invalid("receipt has no recipe object"))?
    } else {
        doc
    };
    if recipe.get("schema").and_then(Value::as_str) != Some("ojon.kld_recipe.v1") {
        return Err(invalid("expected an ojon.kld_recipe.v1 recipe or an ojon.kld_recipe_receipt.v1 receipt"));
    }
    let base = recipe.get("base_level").and_then(Value::as_str).ok_or_else(|| invalid("recipe missing base_level"))?;
    let base = ladder_name(base)?;
    let mut levels = BTreeMap::new();
    let map = recipe.get("levels").and_then(Value::as_object).ok_or_else(|| invalid("recipe missing levels"))?;
    for (name, level) in map {
        let class = TensorClass::parse(name).ok_or_else(|| invalid(format!("unknown tensor class '{name}' in recipe")))?;
        let level = level.as_str().ok_or_else(|| invalid(format!("level for {name} must be a string")))?;
        levels.insert(class, ladder_name(level)?);
    }
    if levels.is_empty() {
        return Err(invalid("recipe has no class levels"));
    }
    Ok((base, levels))
}

/// Apply a measured recipe to `source` (an original F16/BF16 GGUF of the same
/// family) without re-probing. When `verify` is given, a fresh reference is
/// built from this source and the output's KLD is measured and reported; the
/// transferred recipe is then only called within budget on that measurement.
#[allow(clippy::too_many_arguments)]
pub fn apply_recipe<F>(
    source: &Path,
    output_dir: &Path,
    receipt: &Path,
    quantize_tool: &Path,
    imatrix: Option<&Path>,
    threads: usize,
    verify: Option<(&KldSettings, f64)>,
    mut log: F,
) -> io::Result<PathBuf>
where
    F: FnMut(&str),
{
    let doc: Value = serde_json::from_slice(&fs::read(receipt)?).map_err(|e| invalid(format!("invalid recipe JSON: {e}")))?;
    let (base, levels) = levels_from_receipt(&doc)?;
    let source = source.canonicalize()?;
    let info = inspect_gguf(&source)?;
    if !matches!(info.file_type, Some(0 | 1 | 32)) {
        return Err(invalid("recipe-apply requires an original F32/F16/BF16 GGUF"));
    }
    if !output_dir.is_dir() {
        return Err(invalid("output directory must exist"));
    }
    let output_dir = output_dir.canonicalize()?;
    let out_path = output_dir.join("recipe-applied.gguf");
    if out_path.exists() || output_dir.join("recipe-applied-receipt.json").exists() {
        return Err(invalid("recipe-applied output already exists in the output directory"));
    }
    // Only classes present in THIS model are pinned; a class the recipe knows but
    // the model lacks is harmless, a class the model has but the recipe never saw
    // stays at the base and is reported.
    let present = class_elements(&inspect_gguf_tensors(&source)?);
    let mut applied = BTreeMap::new();
    let mut unseen = Vec::new();
    for class in present.keys() {
        match levels.get(class) {
            Some(level) => {
                applied.insert(*class, *level);
            }
            None => {
                applied.insert(*class, base);
                unseen.push(class.name());
            }
        }
    }
    let tt_path = output_dir.join("recipe-applied.tensor-types.txt");
    // Replay what the measured run adopted beyond the class levels: the MTP-block pin
    // and, when the boundary refinement won, its first/last-block pins (they must come
    // first: llama-quantize applies the first matching pattern).
    let mut prefix = mtp_block_pin(&source, base)?;
    let br = &doc["boundary_refinement"];
    if br["adopted"].as_bool() == Some(true) {
        let blocks: Vec<u32> = br["blocks"].as_array().into_iter().flatten().filter_map(|b| b.as_u64().map(|b| b as u32)).collect();
        let lowered = Recipe { levels: applied.clone(), predicted_kld_delta: 0.0, predicted_bytes: 0, base_bytes: 0, steps: Vec::new() };
        prefix.push_str(&render_boundary_lines(&lowered, base, &blocks));
        log(&format!("[ojon] replaying adopted boundary refinement: blocks {blocks:?} kept at {base}"));
    }
    fs::write(&tt_path, format!("{prefix}{}", render_full(&applied)))?;
    log(&format!("[ojon] applying recipe (base {base}) to {}", plain(&source)));
    let started = Instant::now();
    let mut q = prepare_gguf(&source, &out_path, quantize_tool, main_precision_for(base)?, threads)?.with_tensor_type_file(&tt_path)?;
    let slab = crate::resource::current_gguf_slab_mib();
    q = q.with_max_buffer_mib(slab)?;
    log(&format!("[ojon] adaptive CPU quantisation slab: {slab} MiB (the actual process may use more memory)"));
    if let Some(im) = imatrix {
        q = q.with_imatrix(im)?;
    }
    let done = quantize_gguf_with_log(&q, |line| log(line))?;
    let mut receipt_json = json!({
        "schema": "ojon.kld_recipe_applied.v1",
        "source": plain(&source),
        "source_bytes": info.file_bytes,
        "recipe_from": plain(receipt),
        "base_level": base,
        "levels": applied.iter().map(|(c, l)| (c.name().to_string(), json!(l))).collect::<serde_json::Map<_, _>>(),
        "classes_not_in_recipe_left_at_base": unseen,
        "output": plain(&done.output),
        "output_bytes": done.output_info.file_bytes,
        "size_ratio_vs_source": done.output_info.file_bytes as f64 / info.file_bytes.max(1) as f64,
        "quantize_seconds": started.elapsed().as_secs_f64(),
        "quant_buffer_mib": slab,
        "verification": "none",
        "claim": "transferred recipe; not measured on this model unless verification is present"
    });
    if let Some((settings, budget)) = verify {
        let dump = save_reference_logits(&source, &output_dir.join("reference.logits"), settings, &mut log)?;
        let k = measure_kld(&done.output, &dump, settings, &mut log)?;
        // No base candidate here: the budget is absolute mean KLD against the original.
        let within = k.mean <= budget + 1e-9;
        log(&format!("[ojon] applied recipe: mean KLD {:.6} vs absolute budget {:.6} -> {}", k.mean, budget, if within { "within" } else { "OVER" }));
        receipt_json["verification"] = json!({"kld": kld_json(&k), "absolute_mean_kld_budget": budget, "measured_within_budget": within});
        receipt_json["claim"] = json!(if within {
            "measured mean KLD within the absolute budget on this corpus slice; run the task suite before any capability claim"
        } else {
            "rejected: measured mean KLD exceeds the absolute budget; re-run recipe-run on this model"
        });
    }
    let path = output_dir.join("recipe-applied-receipt.json");
    write_new(&path, &serde_json::to_vec_pretty(&receipt_json)?)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_from_reuses_measurements_and_skips_finals() {
        let root = std::env::temp_dir().join(format!("ojon-seed-{}", std::process::id()));
        let (prev, out) = (root.join("prev"), root.join("out"));
        fs::create_dir_all(&prev).unwrap();
        for f in ["recipe-plan.json", "recipe-progress.jsonl", "reference.logits", "ojon.imatrix.gguf", "base-q5_k.gguf", "level-q4_k.gguf", "recipe.gguf", "recipe-receipt.json"] {
            fs::write(prev.join(f), f).unwrap();
        }
        assert_eq!(seed_from(&prev, &out).unwrap(), 6);
        assert!(out.join("level-q4_k.gguf").exists() && out.join("recipe-plan.json").exists());
        assert!(!out.join("recipe.gguf").exists() && !out.join("recipe-receipt.json").exists());
        assert_eq!(seed_from(&prev, &out).unwrap(), 0);
        let _ = fs::remove_dir_all(&root);
    }

    fn measured(mean: f64, p99: Option<f64>) -> KldStats {
        KldStats {
            mean,
            mean_uncertainty: Some(0.0001),
            p99,
            ..KldStats::default()
        }
    }

    #[test]
    fn boundary_refinement_cannot_replace_a_passing_recipe_with_a_tail_regression() {
        let base = measured(0.020, Some(0.100));
        let plain = measured(0.028, Some(0.140));
        let refined_bad_tail = measured(0.023, Some(0.190));
        let plain_checks = check_quality_budgets(&base, &plain, 0.010, Some(0.050));
        let refined_checks = check_quality_budgets(&base, &refined_bad_tail, 0.010, Some(0.050));
        assert_eq!(plain_checks, BudgetChecks { mean_within: true, p99_within: Some(true) });
        assert_eq!(refined_checks, BudgetChecks { mean_within: true, p99_within: Some(false) });
        assert!(plain_checks.accepted());
        assert!(!refined_checks.accepted());
        assert!(!should_adopt_refinement(
            &plain, 1000, plain_checks.accepted(),
            &refined_bad_tail, 1010, refined_checks.accepted(),
        ));

        let refined_good = measured(0.023, Some(0.120));
        let good_checks = check_quality_budgets(&base, &refined_good, 0.010, Some(0.050));
        assert!(good_checks.accepted());
        assert!(should_adopt_refinement(
            &plain, 1000, plain_checks.accepted(),
            &refined_good, 1010, good_checks.accepted(),
        ));
    }

    #[test]
    fn selected_refinement_checks_are_computed_from_selected_kld() {
        let base = measured(0.020, Some(0.100));
        let plain = measured(0.032, Some(0.160));
        let refined = measured(0.026, Some(0.125));
        let plain_checks = check_quality_budgets(&base, &plain, 0.010, Some(0.050));
        let refined_checks = check_quality_budgets(&base, &refined, 0.010, Some(0.050));
        assert_eq!(plain_checks, BudgetChecks { mean_within: false, p99_within: Some(false) });
        assert_eq!(refined_checks, BudgetChecks { mean_within: true, p99_within: Some(true) });
        assert!(should_adopt_refinement(
            &plain, 1000, plain_checks.accepted(), &refined, 1010, refined_checks.accepted(),
        ));
        assert!(refined_checks.accepted());

        // A missing p99 must fail closed when the user requested a tail gate.
        let missing = measured(0.021, None);
        assert_eq!(
            check_quality_budgets(&base, &missing, 0.010, Some(0.050)).p99_within,
            Some(false),
        );
        assert_eq!(
            check_quality_budgets(&base, &missing, 0.010, None).p99_within,
            None,
        );
    }

    #[test]
    fn levels_from_receipt_accepts_recipe_or_full_receipt_and_rejects_junk() {
        let recipe = json!({"schema": "ojon.kld_recipe.v1", "base_level": "q5_k",
                            "levels": {"expert_down": "iq4_xs", "attn_vo": "q5_k"}});
        let (base, levels) = levels_from_receipt(&recipe).unwrap();
        assert_eq!(base, "q5_k");
        assert_eq!(levels[&TensorClass::ExpertDown], "iq4_xs");
        let receipt = json!({"schema": "ojon.kld_recipe_receipt.v1", "recipe": recipe});
        assert_eq!(levels_from_receipt(&receipt).unwrap().1.len(), 2);
        assert!(levels_from_receipt(&json!({"schema": "other"})).is_err());
        assert!(levels_from_receipt(&json!({"schema": "ojon.kld_recipe.v1", "base_level": "q5_k", "levels": {"bogus": "q4_k"}})).is_err());
        assert!(levels_from_receipt(&json!({"schema": "ojon.kld_recipe.v1", "base_level": "q5_k", "levels": {"attn_vo": "q9_z"}})).is_err());
        assert!(levels_from_receipt(&json!({"schema": "ojon.kld_recipe.v1", "base_level": "q5_k", "levels": {}})).is_err());
    }

    #[test]
    fn vision_projectors_and_blockless_files_are_refused() {
        let mut lm = BTreeMap::new();
        lm.insert(TensorClass::AttnQk, 10u64);
        lm.insert(TensorClass::FfnDown, 10u64);
        assert!(not_a_language_model(Some("qwen3"), &lm).is_none());
        assert!(not_a_language_model(Some("clip"), &lm).unwrap().contains("vision projector"));
        let mut proj = BTreeMap::new();
        proj.insert(TensorClass::AttnQk, 10u64); // mmproj attention without FFN classes
        assert!(not_a_language_model(Some("qwen2vl"), &proj).is_some());
        assert!(not_a_language_model(None, &BTreeMap::new()).is_some());
    }

    #[test]
    fn progress_ledger_roundtrips_and_fails_closed_on_damage() {
        let base = json!({"kind": "base", "level": "q5_k", "bytes": 100, "kld": {"mean": 0.02, "p99": 0.1, "seconds": 3.0}});
        let probe = json!({"kind": "probe", "class": "attn_vo", "level": "q4_k", "bytes": 90, "kld": {"mean": 0.03}});
        let text = format!("{}{}{}{}", base, '\n', probe, '\n');
        let p = parse_progress(&text).unwrap();
        assert_eq!(p.base.as_ref().unwrap().0, 100);
        assert_eq!(p.base.as_ref().unwrap().1.p99, Some(0.1));
        assert_eq!(p.probes[&(TensorClass::AttnVo, "q4_k")].1.mean, 0.03);
        assert!(parse_progress(r#"{"kind": "probe", "class": "nope", "level": "q4_k", "bytes": 1, "kld": {"mean": 0.1}}"#).is_err());
        assert!(parse_progress(r#"{"kind": "base", "bytes": 1"#).is_err()); // truncated line
        assert!(parse_progress("").unwrap().base.is_none());
    }

    #[test]
    fn corpus_split_is_deterministic_non_overlapping_and_paragraph_aligned() {
        let text = "alpha one\nalpha two\n\nbeta one\nbeta two\n\ngamma one\ngamma two\n";
        let (a, b) = split_corpus(text);
        assert_eq!(format!("{a}{b}"), text);
        assert!(a.ends_with("\n\n"), "{a:?}");
        assert!(!b.is_empty());
        let (a2, b2) = split_corpus(text);
        assert_eq!((a, b), (a2, b2));
        let (c, d) = split_corpus("no breaks at all here");
        assert_eq!(c.len() + d.len(), "no breaks at all here".len());
        let (e, f) = split_corpus("ééééééé");
        assert!(e.is_char_boundary(e.len()) && !f.is_empty());
    }

    #[test]
    fn base_advisory_fires_only_when_the_base_is_already_lossy() {
        assert!(base_advisory("q5_k", 0.023, 0.01).is_none());
        assert!(base_advisory("q5_k", 0.12, 0.01).unwrap().contains("--base q6_k"));
        assert!(base_advisory("q4_k", 0.06, 0.05).unwrap().contains("--base q5_k"));
        assert!(base_advisory("q6_k", 0.3, 0.0).unwrap().contains("--base q8_0"));
        assert!(base_advisory("q5_k", 0.03, 0.01).is_none()); // 3x budget, under 0.05: fine
    }

    #[test]
    fn noise_advisory_compares_budget_to_twice_the_standard_error() {
        assert!(noise_advisory(Some(0.001), 0.01, 12).is_none());
        let n = noise_advisory(Some(0.008), 0.01, 6).unwrap();
        assert!(n.contains("inside measurement noise") && n.contains("--chunks"), "{n}");
        assert!(noise_advisory(None, 0.01, 6).is_none());
        assert!(noise_advisory(Some(0.0), 0.01, 6).is_none());
    }

    #[test]
    fn main_precision_maps_only_measurable_bases() {
        assert_eq!(main_precision_for("q5_k").unwrap().as_str(), "Q5_K_M");
        assert_eq!(main_precision_for("q4_k").unwrap().as_str(), "Q4_K_M");
        assert_eq!(main_precision_for("q6_k").unwrap().as_str(), "Q6_K");
        assert!(main_precision_for("q3_k").is_err());
        assert!(main_precision_for("q8_0").is_err());
    }

    #[test]
    fn full_render_pins_every_searched_class_and_skips_router() {
        let mut levels = BTreeMap::new();
        levels.insert(TensorClass::AttnVo, "q5_k");
        levels.insert(TensorClass::ExpertDown, "q4_k");
        levels.insert(TensorClass::Router, "q5_k");
        let body = render_full(&levels);
        assert!(body.contains("\\.attn_(v|output)\\.weight$=q5_k\n"));
        assert!(body.contains("\\.ffn_down_exps\\.weight$=q4_k\n"));
        assert!(!body.contains("ffn_gate_inp"));
        assert_eq!(body.lines().count(), 2);
    }

    #[test]
    fn plain_strips_only_the_windows_verbatim_prefix() {
        assert_eq!(plain(Path::new(r"\\?\C:\m\x.gguf")), r"C:\m\x.gguf");
        assert_eq!(plain(Path::new("/srv/models/x.gguf")), "/srv/models/x.gguf");
    }

    #[test]
    fn layer_groups_are_contiguous_and_cover_every_block() {
        // 24 blocks, 4 groups -> 0..6, 6..12, 12..18, 18..24
        assert_eq!((0..24).map(|b| group_of(b, 24, 4)).collect::<Vec<_>>(),
            [vec![0; 6], vec![1; 6], vec![2; 6], vec![3; 6]].concat());
        // uneven: 10 blocks, 4 groups, still 4 non-empty groups in order
        let g: Vec<u32> = (0..10).map(|b| group_of(b, 10, 4)).collect();
        assert_eq!(g.first(), Some(&0));
        assert_eq!(g.last(), Some(&3));
        assert!(g.windows(2).all(|w| w[0] <= w[1]));
        assert_eq!(block_of("blk.17.ffn_down.weight"), Some(17));
        assert_eq!(block_of("token_embd.weight"), None);
    }

    #[test]
    fn mtp_blocks_are_pinned_to_base_except_their_nextn_tensors() {
        // Qwen3.8-27B: 65 blocks, the last one is the MTP layer.
        let pin = mtp_pin_lines(Some((65, 1)), "q5_k");
        assert_eq!(pin, "^blk\\.(64)\\.(?!nextn\\.).*\\.weight$=q5_k\n");
        assert_eq!(mtp_pin_lines(Some((48, 2)), "q6_k"), "^blk\\.(46|47)\\.(?!nextn\\.).*\\.weight$=q6_k\n");
        // No MTP layers, or no metadata: nothing pinned.
        assert_eq!(mtp_pin_lines(Some((32, 0)), "q5_k"), "");
        assert_eq!(mtp_pin_lines(None, "q5_k"), "");
    }

    #[test]
    fn probe_file_names_are_stable_and_filesystem_safe() {
        let s = probe_stem(TensorClass::ExpertGateUp, "iq4_xs");
        assert_eq!(s, "probe-expert_gate_up-iq4_xs");
        assert!(s.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_'));
    }

    #[test]
    fn prepare_rejects_bad_budget_floor_and_threads_before_touching_files() {
        let dir = std::env::temp_dir().join(format!("ojon-recipe-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let corpus = dir.join("c.txt");
        fs::write(&corpus, "y".repeat(300)).unwrap();
        let kld = KldSettings { tool: PathBuf::from("llama-perplexity"), corpus, gpu_layers: 99, fit: false, context: 512, chunks: 10, device: None };
        let job = RecipeJob {
            source: dir.join("missing.gguf"), output_dir: dir.clone(), base: "q5_k", floor: "iq2_s", budget_kld: 0.01,
            quantize_tool: PathBuf::from("llama-quantize"), imatrix: None, imatrix_tool: None, threads: 4, kld,
            keep_probes: false, budget_p99: None, resume: false, eval_corpus: None, refine_boundaries: true, target_bytes: None,
        };
        assert!(prepare_recipe(RecipeJob { budget_p99: Some(-0.5), ..job.clone() }).is_err());
        assert!(prepare_recipe(RecipeJob { budget_kld: 2.0, ..job.clone() }).is_err());
        assert!(prepare_recipe(RecipeJob { floor: "q6_k", ..job.clone() }).is_err());
        assert!(prepare_recipe(RecipeJob { threads: 0, ..job.clone() }).is_err());
        assert!(prepare_recipe(RecipeJob { base: "iq2_s", ..job.clone() }).is_err());
        // valid settings but no source file: fails at the source, not earlier
        let err = prepare_recipe(job).unwrap_err();
        assert!(err.kind() == io::ErrorKind::NotFound || err.to_string().contains("source") || err.to_string().contains("system cannot find"), "{err}");
        let _ = fs::remove_dir_all(&dir);
    }
}
