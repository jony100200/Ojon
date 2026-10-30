//! Benchmark predownloaded lower-bit GGUFs against an original or Q4 reference.
//! Requires NO high-precision source, NO weights loaded during plan, and NO
//! quantization process. Runtime tests run only with explicit confirmation.

use crate::gguf::{inspect_gguf, inspect_gguf_tensors};
use crate::ppl::{measure as measure_perplexity, PplSettings};
use crate::quant_parity::{
    task_parity, vision_score, vision_suite, QuantParityOptions, VisionSuite,
};
use crate::task_eval::{
    parse_eval_suite, prepare_eval_with_server, run_eval_reusing_baseline, EvalRequest,
};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::fs::OpenOptions;
use std::io::{self, ErrorKind, Write};
use std::path::{Path, PathBuf};

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, s.into())
}
#[derive(Clone, Debug)]
pub struct ExistingQuantPlan {
    baseline: PathBuf,
    baseline_bytes: u64,
    candidates: Vec<(String, PathBuf, u64)>,
    suite: PathBuf,
    report: PathBuf,
    options: QuantParityOptions,
    vision: Option<VisionSuite>,
}
impl ExistingQuantPlan {
    pub fn as_json(&self) -> Value {
        json!({"schema":"ojon.existing_quant_plan.v1",
            "reference":self.baseline.to_string_lossy(),
            "reference_bytes":self.baseline_bytes,
            "candidates":self.candidates.iter().map(|(n,p,bytes)|json!({
                "name":n,"path":p.to_string_lossy(),"bytes":bytes,
                "savings_pct":100.0*(1.0-*bytes as f64/self.baseline_bytes as f64)
            })).collect::<Vec<_>>(),
            "task_suite":self.suite.to_string_lossy(),
            "vision_suite_count":self.vision.as_ref().map(VisionSuite::count),
            "perplexity_corpus":self.options.ppl_corpus.as_ref().map(|x|x.to_string_lossy().to_string()),
            "max_loss_points":self.options.max_loss_points,
            "min_saving_percent":self.options.min_saving_percent,
            "max_ppl_regression_pct":self.options.max_ppl_regression_pct,
            "report":self.report.to_string_lossy(),
            "no_quantization_or_weights_loaded":true,
            "status":"PLAN_ONLY"})
    }
}
pub fn prepare_existing_quant_comparison(
    baseline: &Path,
    candidates: &[(String, PathBuf)],
    suite: &Path,
    report: &Path,
    mut options: QuantParityOptions,
) -> io::Result<ExistingQuantPlan> {
    let baseline = baseline.canonicalize()?;
    let b = inspect_gguf(&baseline)?;
    if b.tensor_count == 0 || b.split_count.unwrap_or(1) > 1 || b.file_type.is_none() {
        return Err(invalid("reference must be a nonempty single-file GGUF"));
    }
    let b_arch = b
        .architecture
        .ok_or_else(|| invalid("reference architecture missing"))?;
    let b_tensors: BTreeMap<_, _> = inspect_gguf_tensors(&baseline)?
        .into_iter()
        .map(|t| (t.name, t.shape))
        .collect();
    if candidates.is_empty() || candidates.len() > 6 {
        return Err(invalid("need 1..6 already downloaded quantized candidates"));
    }
    let mut seen = HashSet::new();
    let mut rows = Vec::new();
    for (name, path) in candidates {
        if name.is_empty()
            || name.len() > 80
            || !name
                .bytes()
                .all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-')
            || !seen.insert(name.clone())
        {
            return Err(invalid("invalid or duplicated candidate label"));
        }
        let path = path.canonicalize()?;
        if path == baseline || rows.iter().any(|(_, p, _)| p == &path) {
            return Err(invalid(
                "reference and candidate file paths must be distinct",
            ));
        }
        let c = inspect_gguf(&path)?;
        if c.tensor_count != b.tensor_count
            || c.split_count.unwrap_or(1) > 1
            || c.architecture.as_deref() != Some(b_arch.as_str())
            || c.file_type.is_none()
            || matches!(c.file_type, Some(0 | 1 | 32))
        {
            return Err(invalid(format!(
                "candidate {name} is not a matching compressed GGUF"
            )));
        }
        let shapes: BTreeMap<_, _> = inspect_gguf_tensors(&path)?
            .into_iter()
            .map(|t| (t.name, t.shape))
            .collect();
        if shapes != b_tensors {
            return Err(invalid(format!(
                "{name} tensor names/shapes do not match reference"
            )));
        }
        rows.push((name.clone(), path, c.file_bytes));
    }
    if !options.max_loss_points.is_finite()
        || !(0.0..=100.0).contains(&options.max_loss_points)
        || !options.min_saving_percent.is_finite()
        || !(0.0..=95.0).contains(&options.min_saving_percent)
    {
        return Err(invalid("quality loss or size savings threshold invalid"));
    }
    if !(1..=1024).contains(&options.tokens) || !(-1..=512).contains(&options.gpu_layers) {
        return Err(invalid("invalid evaluation generation configuration"));
    }
    if let Some(d) = &options.device {
        if d.is_empty()
            || d.len() > 80
            || !d
                .bytes()
                .all(|x| x.is_ascii_alphanumeric() || matches!(x, b'_' | b'-' | b'/' | b',' | b':'))
        {
            return Err(invalid("invalid evaluation GPU device"));
        }
    }
    let suite = suite.canonicalize()?;
    parse_eval_suite(&suite)?;
    let vision = if let Some(path) = &options.vision_suite {
        let mm = options
            .mmproj
            .as_ref()
            .ok_or_else(|| invalid("vision suite requires --mmproj"))?;
        let mm = mm.canonicalize()?;
        if inspect_gguf(&mm)?.tensor_count == 0 {
            return Err(invalid("mmproj empty"));
        }
        options.mmproj = Some(mm);
        Some(vision_suite(path)?)
    } else {
        if options.mmproj.is_some() {
            return Err(invalid(
                "mmproj without visual tests cannot prove VLM parity",
            ));
        }
        None
    };
    if let Some(path) = options.ppl_corpus.take() {
        let path = path.canonicalize()?;
        let settings = PplSettings {
            tool: options.ppl_tool.clone(),
            corpus: path.clone(),
            gpu_layers: options.gpu_layers.max(0) as u32,
            context: options.ppl_context,
            max_relative_degradation_pct: options.max_ppl_regression_pct,
        };
        settings.validate()?;
        options.ppl_corpus = Some(path);
    }
    let parent = report
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    if report.extension().and_then(|s| s.to_str()) != Some("json") {
        return Err(invalid("report must be .json"));
    }
    let report = parent.join(
        report
            .file_name()
            .ok_or_else(|| invalid("report filename missing"))?,
    );
    if report.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "report exists"));
    }
    Ok(ExistingQuantPlan {
        baseline,
        baseline_bytes: b.file_bytes,
        candidates: rows,
        suite,
        report,
        options,
        vision,
    })
}
pub fn run_existing_quant_comparison(
    plan: &ExistingQuantPlan,
    mut log: impl FnMut(&str),
) -> io::Result<Value> {
    if plan.report.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "report exists"));
    }
    let mut results = Vec::new();
    let mut winner: Option<(String, PathBuf, u64)> = None;
    let mut baseline_vis: Option<Value> = None;
    let mut baseline_ppl: Option<f64> = None;
    let mut baseline_tasks: Option<Value> = None;
    for (idx, (label, path, bytes)) in plan.candidates.iter().enumerate() {
        let savings = 100.0 * (1.0 - *bytes as f64 / plan.baseline_bytes as f64);
        if *bytes >= plan.baseline_bytes || savings + 1e-10 < plan.options.min_saving_percent {
            results.push(json!({"name":label,"bytes":bytes,"savings_pct":savings,
                "status":"NOT_SMALL_ENOUGH"}));
            continue;
        }
        // JSON evidence from run_eval is unavoidable because it records all
        // deterministic answers. Never overwrite preexisting files.
        let evidence = plan.report.with_file_name(format!(
            "{}-candidate-{idx}-evidence.json",
            plan.report
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("parity")
        ));
        let test = (|| {
            let v = vec![(label.clone(), path.clone())];
            let prep = prepare_eval_with_server(EvalRequest {
                suite: &plan.suite,
                baseline: &plan.baseline,
                candidates: &v,
                tool: &plan.options.eval_tool,
                server_tool: plan.options.server_tool.as_deref(),
                gpu_layers: plan.options.gpu_layers,
                device: plan.options.device.as_deref(),
                tokens: plan.options.tokens,
                report: &evidence,
            })?;
            run_eval_reusing_baseline(&prep, baseline_tasks.as_ref(), |msg| log(msg))
        })();
        let scored = match test {
            Ok(v) => v,
            Err(e) => {
                results.push(json!({"name":label,"bytes":bytes,"savings_pct":savings,
                    "status":"TEXT_EVALUATION_FAILED","error":e.to_string()}));
                continue;
            }
        };
        if baseline_tasks.is_none() {
            baseline_tasks = Some(scored.clone());
        }
        let (text_pass, text_scores) = task_parity(&scored, plan.options.max_loss_points);
        let mut image_ok = true;
        let mut image_scores = Value::Null;
        if text_pass {
            if let Some(vsuite) = &plan.vision {
                let test = (|| {
                    let mmproj = plan
                        .options
                        .mmproj
                        .as_deref()
                        .ok_or_else(|| invalid("mmproj missing"))?;
                    if baseline_vis.is_none() {
                        baseline_vis = Some(vision_score(
                            &plan.options,
                            vsuite,
                            &plan.baseline,
                            mmproj,
                            &mut log,
                        )?);
                    }
                    vision_score(&plan.options, vsuite, path, mmproj, &mut log)
                })();
                match test {
                    Ok(candidate) => {
                        let reference = baseline_vis.as_ref().unwrap();
                        let b = reference["accuracy_pct"].as_f64().unwrap_or(0.0);
                        let c = candidate["accuracy_pct"].as_f64().unwrap_or(-1.0);
                        image_ok = b > 0.0 && c + plan.options.max_loss_points + 1e-10 >= b;
                        image_scores = json!({"baseline":reference,"candidate":candidate,
                            "pass":image_ok});
                    }
                    Err(e) => {
                        image_ok = false;
                        image_scores = json!({"status":"VISION_EVALUATION_FAILED",
                            "error":e.to_string()});
                    }
                }
            }
        }
        let mut ppl_ok = true;
        let mut ppl_scores = Value::Null;
        if text_pass && image_ok {
            if let Some(corpus) = &plan.options.ppl_corpus {
                let cfg = PplSettings {
                    tool: plan.options.ppl_tool.clone(),
                    corpus: corpus.clone(),
                    gpu_layers: plan.options.gpu_layers.max(0) as u32,
                    context: plan.options.ppl_context,
                    max_relative_degradation_pct: plan.options.max_ppl_regression_pct,
                };
                let test = (|| {
                    if baseline_ppl.is_none() {
                        baseline_ppl = Some(measure_perplexity(&plan.baseline, &cfg, &mut log)?.0);
                    }
                    measure_perplexity(path, &cfg, &mut log)
                })();
                match test {
                    Ok((ppl, seconds)) => {
                        let reference = baseline_ppl.unwrap();
                        let regression = 100.0 * (ppl / reference - 1.0);
                        ppl_ok = regression <= plan.options.max_ppl_regression_pct + 1e-10;
                        ppl_scores = json!({"baseline":reference,"candidate":ppl,
                            "relative_regression_pct":regression,
                            "seconds":seconds,"pass":ppl_ok});
                    }
                    Err(e) => {
                        ppl_ok = false;
                        ppl_scores =
                            json!({"status":"PPL_EVALUATION_FAILED","error":e.to_string()});
                    }
                }
            }
        }
        let pass = text_pass && image_ok && ppl_ok;
        if pass && winner.as_ref().is_none_or(|(_, _, n)| bytes < n) {
            winner = Some((label.clone(), path.clone(), *bytes));
        }
        results.push(json!({"name":label,"path":path.to_string_lossy(),
            "bytes":bytes,"savings_pct":savings,
            "status":if pass{"NO_DETECTED_TEST_REGRESSION"}else{"FAILED_QUALITY_GATE"},
            "text_scores":text_scores,"vision_scores":image_scores,
            "perplexity_scores":ppl_scores,
            "text_evidence":evidence.to_string_lossy(),
            "vision_tested":plan.vision.is_some(),"ppl_tested":plan.options.ppl_corpus.is_some(),
            "globally_lossless_proven":false}));
    }
    let report = json!({"schema":"ojon.existing_quant_comparison.v1",
        "reference":plan.baseline.to_string_lossy(),
        "reference_bytes":plan.baseline_bytes,
        "results":results,
        "provisional_smallest_candidate":winner.as_ref().map(|(n,path,bytes)|
            json!({"name":n,"path":path.to_string_lossy(),"bytes":bytes})),
        "whole_model_quality_equivalence_proven":false,
        "real_hardware_fit_measured":false,
        "scope":"Actual same-suite results, not general capability proof",
        "warning":"For VLMs require real image suites; for coding agents, benchmark whole coding workflows; no zero-loss guarantee."});
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&plan.report)?;
    f.write_all(
        serde_json::to_string_pretty(&report)
            .map_err(io::Error::other)?
            .as_bytes(),
    )?;
    f.sync_all()?;
    Ok(report)
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    fn tiny_gguf(path: &Path, file_type: u32) {
        use std::io::{Seek, Write};
        fn val_string(f: &mut fs::File, s: &str) {
            f.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
            f.write_all(s.as_bytes()).unwrap();
        }
        let mut f = fs::File::create(path).unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap(); // 1 tensor
        f.write_all(&2u64.to_le_bytes()).unwrap(); // 2 metadata keys
        val_string(&mut f, "general.architecture");
        f.write_all(&8u32.to_le_bytes()).unwrap();
        val_string(&mut f, "qwen3");
        val_string(&mut f, "general.file_type");
        f.write_all(&4u32.to_le_bytes()).unwrap();
        f.write_all(&file_type.to_le_bytes()).unwrap();
        val_string(&mut f, "blk.0.attn_output.weight");
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        let pos = f.stream_position().unwrap();
        f.write_all(&vec![0u8; ((32 - pos % 32) % 32) as usize])
            .unwrap();
        f.write_all(&[0u8; 32]).unwrap();
    }
    #[test]
    fn compare_existing_quants_preflight_uses_metadata_only() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let tick = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("ojon-existing-quant-{}-{tick}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let q4 = dir.join("reference.gguf");
        tiny_gguf(&q4, 15);
        let q3 = dir.join("candidate.gguf");
        tiny_gguf(&q3, 27);
        let suite = dir.join("tasks.json");
        let tasks = (0..6)
            .map(|i| {
                json!({
                    "id":format!("t{i}"),
                    "category":if i<2{"coding"}else if i<4{"reasoning"}else{"tool_call"},
                    "prompt":"Return 42","expected":"42","match":"exact"
                })
            })
            .collect::<Vec<_>>();
        fs::write(
            &suite,
            serde_json::to_vec(&json!({
                "schema":"ojon.task_suite.v1","id":"fixture","tasks":tasks
            }))
            .unwrap(),
        )
        .unwrap();
        let report = dir.join("compare.json");
        let plan = prepare_existing_quant_comparison(
            &q4,
            &[("iq3m".into(), q3.clone())],
            &suite,
            &report,
            QuantParityOptions::default(),
        )
        .unwrap();
        assert_eq!(plan.as_json()["status"], "PLAN_ONLY");
        assert!(plan.as_json()["no_quantization_or_weights_loaded"]
            .as_bool()
            .unwrap());
        assert!(!report.exists());
        assert!(prepare_existing_quant_comparison(
            &q4,
            &[("duplicate".into(), q4.clone())],
            &suite,
            &report,
            QuantParityOptions::default()
        )
        .is_err());
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn default_existing_comparison_is_cpu_and_strict() {
        let options = QuantParityOptions::default();
        assert_eq!(options.gpu_layers, 0);
        assert_eq!(options.max_loss_points, 0.0);
        assert_eq!(options.min_saving_percent, 5.0);
    }
}
