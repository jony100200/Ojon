//! Optional *real* perplexity-based GGUF candidate comparison.
//!
//! Uses an independently installed public llama-perplexity executable,
//! original model + identical held-out text corpus, and sequential evaluation.
//! PPL preservation is NOT the same as preserving general reasoning/tool/vision
//! quality. A successful result is therefore marked "perplexity_only".

use crate::gguf::inspect_gguf;
use crate::jobs::run_with_log;
use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

/// Extracts only upstream's final estimate, not intermediate chunk estimates.
pub fn parse_final_perplexity(line: &str) -> Option<f64> {
    let marker = "Final estimate: PPL =";
    let raw = line.split_once(marker)?.1.trim();
    let numeric = raw.split_whitespace().next()?.trim();
    let n = numeric.parse::<f64>().ok()?;
    if n.is_finite() && n > 0.0 {
        Some(n)
    } else {
        None
    }
}
#[derive(Clone, Debug)]
pub struct PplSettings {
    pub tool: PathBuf,
    pub corpus: PathBuf,
    pub gpu_layers: u32,
    pub context: u32,
    pub max_relative_degradation_pct: f64,
}
impl PplSettings {
    pub fn validate(&self) -> io::Result<()> {
        if !(256..=32768).contains(&self.context) || !self.context.is_multiple_of(32) {
            return Err(invalid("context must be 256..=32768 and divisible by 32"));
        }
        if self.gpu_layers > 512 {
            return Err(invalid("GPU layers must be 0..=512"));
        }
        if !self.max_relative_degradation_pct.is_finite()
            || !(0.0..=100.0).contains(&self.max_relative_degradation_pct)
        {
            return Err(invalid(
                "quality threshold must be a finite 0..=100 percent",
            ));
        }
        if !self.corpus.is_file() || self.corpus.metadata()?.len() < 100 {
            return Err(invalid(
                "held-out corpus must be a local readable text file of at least 100 bytes",
            ));
        }
        if self.tool.as_os_str().is_empty() {
            return Err(invalid("perplexity executable missing"));
        }
        Ok(())
    }
}
pub(crate) fn measure<F>(
    model: &Path,
    settings: &PplSettings,
    log: &mut F,
) -> io::Result<(f64, f64)>
where
    F: FnMut(&str),
{
    let info = inspect_gguf(model)?;
    if info.tensor_count == 0 {
        return Err(invalid("empty GGUF tensor catalog"));
    }
    let started = Instant::now();
    let mut estimate = None;
    let mut cmd = crate::jobs::tool_command(&settings.tool);
    cmd.arg("-m")
        .arg(model)
        .arg("-f")
        .arg(&settings.corpus)
        .arg("-c")
        .arg(settings.context.to_string())
        .arg("-ngl")
        .arg(settings.gpu_layers.to_string());
    run_with_log(&mut cmd, |line| {
        if let Some(value) = parse_final_perplexity(line) {
            estimate = Some(value);
        }
        log(line);
    })?;
    let ppl = estimate.ok_or_else(|| {
        io::Error::other(
        "upstream perplexity tool did not emit 'Final estimate: PPL = ...'; cannot compare safely"
    )
    })?;
    Ok((ppl, started.elapsed().as_secs_f64()))
}

fn read_sweep_manifest(path: &Path) -> io::Result<Value> {
    let meta = fs::metadata(path)?;
    if meta.len() == 0 || meta.len() > 2 * 1024 * 1024 {
        return Err(invalid(
            "sweep manifest must be a JSON file of at most 2MiB",
        ));
    }
    let raw = fs::read(path)?;
    let data: Value =
        serde_json::from_slice(&raw).map_err(|e| invalid(format!("invalid manifest: {e}")))?;
    if data["schema"].as_str() != Some("ojon.sweep_result.v1") {
        return Err(invalid("expected an Ojon completed sweep manifest"));
    }
    Ok(data)
}
pub fn evaluate_sweep<F>(manifest: &Path, settings: &PplSettings, mut log: F) -> io::Result<PathBuf>
where
    F: FnMut(&str),
{
    settings.validate()?;
    let content = read_sweep_manifest(manifest)?;
    let source = content["plan"]["source"]
        .as_str()
        .ok_or_else(|| invalid("manifest missing source path"))?;
    let source = PathBuf::from(source);
    let output_dir = manifest
        .parent()
        .ok_or_else(|| invalid("manifest has no parent"))?;
    let rows = content["candidates"]
        .as_array()
        .ok_or_else(|| invalid("manifest missing candidate list"))?;
    if rows.is_empty() || rows.len() > 12 {
        return Err(invalid("invalid candidate count"));
    }
    log(&format!(
        "[ojon] evaluating original {} (PPL)",
        source.display()
    ));
    let (baseline, base_time) = measure(&source, settings, &mut log)?;
    let mut results = Vec::new();
    let mut best: Option<(String, u64)> = None;
    for row in rows {
        let name = row["format"].as_str().unwrap_or("unknown");
        if row["status"].as_str() != Some("converted_catalog_verified") {
            results.push(json!({"format":name,"status":"skipped_conversion_failed"}));
            continue;
        }
        let file = row["output"]
            .as_str()
            .ok_or_else(|| invalid("candidate path missing"))?;
        let path = PathBuf::from(file);
        // Only measure candidates residing in the original sweep output dir.
        let real = path.canonicalize()?;
        let canonical_dir = output_dir.canonicalize()?;
        if !real.starts_with(&canonical_dir) {
            return Err(invalid(
                "manifest references a candidate outside the sweep directory",
            ));
        }
        log(&format!("[ojon] evaluating {}: {}", name, real.display()));
        match measure(&real, settings, &mut log) {
            Ok((ppl, seconds)) => {
                let delta = (ppl / baseline - 1.0) * 100.0;
                let allowed = delta <= settings.max_relative_degradation_pct + 1e-10;
                let size = fs::metadata(&real)?.len();
                if allowed && best.as_ref().is_none_or(|(_, bytes)| size < *bytes) {
                    best = Some((name.to_owned(), size));
                }
                results.push(json!({
                    "format":name,"output":real.to_string_lossy(),
                    "output_bytes":size,
                    "perplexity":ppl,
                    "relative_perplexity_change_pct":delta,
                    "elapsed_seconds":seconds,
                    "ppl_threshold_pass":allowed,
                    "quality_status":"PERPLEXITY_ONLY_NOT_FULL_TASK_VALIDATION",
                    "runtime_load_verified_for_perplexity":true,
                }));
            }
            Err(e) => {
                log(&format!("[ojon] {name}: evaluation failed: {e}"));
                results.push(
                    json!({"format":name,"status":"evaluation_failed","error":e.to_string()}),
                );
            }
        }
    }
    let report = json!({
        "schema":"ojon.ppl_sweep.v1",
        "original_model":source.to_string_lossy(),
        "baseline_perplexity":baseline,
        "baseline_elapsed_seconds":base_time,
        "same_corpus":settings.corpus.to_string_lossy(),
        "perplexity_tool":settings.tool.to_string_lossy(),
        "context":settings.context,
        "gpu_layers":settings.gpu_layers,
        "allowed_relative_ppl_degradation_pct":settings.max_relative_degradation_pct,
        "candidates":results,
        "smallest_ppl_qualifying_format":best.as_ref().map(|x|&x.0),
        "full_task_quality_verified":false,
        "recommendation_status":"PPL_ONLY_PROVISIONAL_NEEDS_TASK_TESTS",
        "warning":"PPL is a model-specific proxy. Do not publish this result as negligible overall quality loss. Also compare real text generation, tool use, coding, multilingual and vision tasks as applicable."
    });
    let output = output_dir.join(format!(
        "ojon-ppl-evaluation-{}.json",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| invalid("clock"))?
            .as_nanos()
    ));
    let mut writer = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&output)?;
    serde_json::to_writer_pretty(&mut writer, &report)
        .map_err(|e| io::Error::other(format!("JSON error: {e}")))?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(output)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_only_final_ppl() {
        assert_eq!(
            parse_final_perplexity("[stderr] Final estimate: PPL = 7.2510 +/- 0.023"),
            Some(7.251)
        );
        assert_eq!(parse_final_perplexity("chunk 2: PPL = 4.2"), None);
        assert_eq!(parse_final_perplexity("Final estimate: PPL = inf"), None);
    }
}
