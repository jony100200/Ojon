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

use crate::gguf::{inspect_gguf, inspect_gguf_tensors};
use crate::llama::{prepare_gguf, quantize_gguf_with_log, Precision};
use crate::quant::kld::{measure_kld, save_reference_logits, KldSettings, KldStats};
use crate::quant::recipe::{
    class_elements, ladder_index, probe_matrix, recipe_json, select_recipe, Measured, Probe, Recipe,
    TensorClass, LADDER,
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
}

#[derive(Debug, Clone)]
pub struct RecipePlan {
    pub job: RecipeJob,
    pub present: BTreeMap<TensorClass, u64>,
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

pub fn prepare_recipe(job: RecipeJob) -> io::Result<RecipePlan> {
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
    let info = inspect_gguf(&source)?;
    if info.tensor_count == 0 {
        return Err(invalid("source GGUF has no tensors"));
    }
    if info.split_count.unwrap_or(1) > 1 {
        return Err(invalid("split GGUF source unsupported; merge the shards first"));
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
    for name in ["recipe-plan.json", "recipe-receipt.json", "recipe.gguf", "reference.logits"] {
        if output_dir.join(name).exists() {
            return Err(invalid(format!("{name} already exists in the output directory; use an empty directory")));
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
        if output_dir.join("calibration.imatrix.gguf").exists() {
            return Err(invalid("calibration.imatrix already exists in the output directory"));
        }
    }
    if let Some(p99) = job.budget_p99 {
        if !(p99.is_finite() && (0.0..=10.0).contains(&p99)) {
            return Err(invalid("p99 budget must be a finite number in 0..=10"));
        }
    }
    let tensors = inspect_gguf_tensors(&source)?;
    let present = class_elements(&tensors);
    if let Some(reason) = not_a_language_model(info.architecture.as_deref(), &present) {
        return Err(invalid(reason));
    }
    let probes = probe_matrix(job.base, &present, job.floor)?;
    if probes.is_empty() {
        return Err(invalid("nothing to search: no ladder level between floor and base for the classes present"));
    }
    let job = RecipeJob { source, output_dir, ..job };
    Ok(RecipePlan { job, present, probes, source_bytes: info.file_bytes, tensor_count: info.tensor_count })
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
            "context": self.job.kld.context,
            "chunks": self.job.kld.chunks,
            "quantize_tool": plain(&self.job.quantize_tool),
            "perplexity_tool": plain(&self.job.kld.tool),
            "imatrix": self.job.imatrix.as_ref().map(|p| plain(p)),
            "imatrix_auto_tool": self.job.imatrix_tool.as_ref().map(|p| plain(p)),
            "budget_p99": self.job.budget_p99,
            "execution": "plan_only",
            "claim": "none; every number in the receipt is measured, and KLD is a distribution distance, not a task score"
        })
    }

    pub fn write_plan(&self) -> io::Result<PathBuf> {
        let path = self.job.output_dir.join("recipe-plan.json");
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
    let dir = &plan.job.output_dir;
    let tt_path = dir.join(format!("{stem}.tensor-types.txt"));
    let out_path = dir.join(format!("{stem}.gguf"));
    if out_path.exists() {
        return Err(invalid(format!("refusing to overwrite {}", out_path.display())));
    }
    fs::write(&tt_path, render_full(levels))?;
    let started = Instant::now();
    let mut q = prepare_gguf(&plan.job.source, &out_path, &plan.job.quantize_tool, main_precision_for(plan.job.base)?, plan.job.threads)?
        .with_tensor_type_file(&tt_path)?;
    if let Some(im) = &plan.job.imatrix {
        q = q.with_imatrix(im)?;
    }
    let done = quantize_gguf_with_log(&q, |line| log(line))?;
    Ok(Quantized { path: done.output, bytes: done.output_info.file_bytes, seconds: started.elapsed().as_secs_f64() })
}

fn kld_json(s: &KldStats) -> Value {
    json!({"mean": s.mean, "mean_uncertainty": s.mean_uncertainty, "median": s.median, "p99": s.p99, "max": s.max, "seconds": s.seconds})
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
    if plan.job.imatrix.is_none() {
        if let Some(tool) = &plan.job.imatrix_tool {
            let out = dir.join("calibration.imatrix.gguf");
            log(&format!("[ojon] importance matrix from the corpus with {}", plain(tool)));
            let im_plan = crate::quant::calibration::prepare_imatrix(
                &plan.job.source, &plan.job.kld.corpus, &out, tool,
                if plan.job.kld.fit { crate::quant::calibration::GPU_LAYERS_FIT } else { plan.job.kld.gpu_layers },
                plan.job.kld.context)?;
            let saved = crate::quant::calibration::run_imatrix(&im_plan, |line| log(line))?;
            plan_used.job.imatrix = Some(saved.clone());
            imatrix_auto = Some(saved);
        }
    }
    let plan = &plan_used;

    log(&format!("[ojon] reference logits for {} ({} chunks of {} ctx)", plain(&plan.job.source), plan.job.kld.chunks, plan.job.kld.context));
    let dump = save_reference_logits(&plan.job.source, &dir.join("reference.logits"), &plan.job.kld, &mut log)?;

    let all_base: BTreeMap<TensorClass, &'static str> = plan.present.keys().map(|c| (*c, base_level)).collect();
    log(&format!("[ojon] base candidate: every class at {base_level}"));
    let base = quantize_with(plan, &format!("base-{base_level}"), &all_base, &mut log)?;
    let base_kld = measure_kld(&base.path, &dump, &plan.job.kld, &mut log)?;
    log(&format!("[ojon] base: {:.2}% of source, mean KLD {:.6}", 100.0 * base.bytes as f64 / plan.source_bytes as f64, base_kld.mean));

    let mut rows = Vec::new();
    let mut measured = Vec::new();
    for (i, probe) in plan.probes.iter().enumerate() {
        let mut levels = all_base.clone();
        levels.insert(probe.class, probe.level);
        let stem = probe_stem(probe.class, probe.level);
        log(&format!("[ojon] probe {}/{}: {} at {}", i + 1, plan.probes.len(), probe.class.name(), probe.level));
        let outcome = quantize_with(plan, &stem, &levels, &mut log).and_then(|q| {
            let k = measure_kld(&q.path, &dump, &plan.job.kld, &mut log)?;
            Ok((q, k))
        });
        match outcome {
            Ok((q, k)) => {
                log(&format!("[ojon]   mean KLD {:.6} (delta {:+.6}), {:.2}% of source", k.mean, k.mean - base_kld.mean, 100.0 * q.bytes as f64 / plan.source_bytes as f64));
                rows.push(json!({"class": probe.class.name(), "level": probe.level, "status": "measured", "bytes": q.bytes,
                                 "quantize_seconds": q.seconds, "kld": kld_json(&k), "kld_delta_vs_base": k.mean - base_kld.mean}));
                measured.push(Measured { class: probe.class, level: probe.level, mean_kld: k.mean });
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

    let recipe: Recipe = select_recipe(plan.job.base, &plan.present, base_kld.mean, &measured, plan.job.budget_kld)?;
    log(&format!("[ojon] selected recipe: {} steps, predicted KLD delta {:.6}, predicted {:.1}% of base bytes",
        recipe.steps.len(), recipe.predicted_kld_delta, 100.0 * recipe.predicted_bytes as f64 / recipe.base_bytes.max(1) as f64));
    let final_q = quantize_with(plan, "recipe", &recipe.levels, &mut log)?;
    let final_kld = measure_kld(&final_q.path, &dump, &plan.job.kld, &mut log)?;
    let mean_within = final_kld.mean <= base_kld.mean + plan.job.budget_kld + 1e-9;
    // Tail gate: only when both p99 values were reported and a p99 budget was set.
    let p99_within = match (plan.job.budget_p99, base_kld.p99, final_kld.p99) {
        (Some(budget), Some(base_p99), Some(final_p99)) => Some(final_p99 <= base_p99 + budget + 1e-9),
        (Some(_), _, _) => Some(false), // asked for a tail gate but the tool did not report p99: fail closed
        (None, _, _) => None,
    };
    let within = mean_within && p99_within.unwrap_or(true);
    log(&format!("[ojon] recipe: {:.2}% of source, mean KLD {:.6} (base {:.6}, budget +{:.6}){} -> {}",
        100.0 * final_q.bytes as f64 / plan.source_bytes as f64, final_kld.mean, base_kld.mean, plan.job.budget_kld,
        match (plan.job.budget_p99, final_kld.p99) {
            (Some(b), Some(p)) => format!(", p99 {:.6} (budget +{:.6})", p, b),
            _ => String::new(),
        },
        if within { "within budget" } else { "OVER budget; not accepted" }));

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
                  "mean_within_budget": mean_within,
                  "p99_budget": plan.job.budget_p99,
                  "p99_within_budget": p99_within,
                  "measured_within_budget": within},
        "imatrix": plan.job.imatrix.as_ref().map(|p| plain(p)),
        "imatrix_computed_by_this_run": imatrix_auto.as_ref().map(|p| plain(p)),
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
    fs::write(&tt_path, render_full(&applied))?;
    log(&format!("[ojon] applying recipe (base {base}) to {}", plain(&source)));
    let started = Instant::now();
    let mut q = prepare_gguf(&source, &out_path, quantize_tool, main_precision_for(base)?, threads)?.with_tensor_type_file(&tt_path)?;
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
        let kld = KldSettings { tool: PathBuf::from("llama-perplexity"), corpus, gpu_layers: 99, fit: false, context: 512, chunks: 10 };
        let job = RecipeJob {
            source: dir.join("missing.gguf"), output_dir: dir.clone(), base: "q5_k", floor: "iq2_s", budget_kld: 0.01,
            quantize_tool: PathBuf::from("llama-quantize"), imatrix: None, imatrix_tool: None, threads: 4, kld,
            keep_probes: false, budget_p99: None,
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
