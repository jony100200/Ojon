//! Reusable edit -> compress -> load -> evaluate -> select pipeline (`ojon.pipeline.v1`).
//!
//! One config describes a run; the same stages execute for any model size:
//!   1. apply verified weight patches in order (`weight_patch`, hashed, exact rollback)
//!   2. convert the (patched) HF folder to a high-precision GGUF (llama.cpp convert script)
//!   3. imatrix on the calibration text
//!   4. quant ladder (llama-quantize with the imatrix), optional unpatched baseline quant
//!   5. KLD of every candidate against the high-precision GGUF on held-out text
//!   6. each configured test command on each candidate (writes a metrics JSON)
//!   7. `quality::recommend` picks the smallest candidate within tolerance, or none
//! Every stage is resumable: finished outputs are recorded in `pipeline-state.json` with their sha256
//! and skipped on rerun. Nothing here simulates inference; tests are real external commands.

use crate::quality::{parse_quality_report, recommend};
use crate::weight_patch::{prepare_weight_patch, run_weight_patch, sha256_file};
use serde_json::{json, Map, Value};
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::Command;

pub const PIPELINE_SCHEMA: &str = "ojon.pipeline.v1";

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, s.into())
}

fn s<'a>(v: &'a Value, key: &str) -> io::Result<&'a str> {
    v.get(key).and_then(Value::as_str).ok_or_else(|| invalid(format!("pipeline config needs string `{key}`")))
}

/// A parsed pipeline configuration.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    pub raw: Value,
    pub name: String,
    pub source: PathBuf,
    pub work: PathBuf,
    pub patches: Vec<PathBuf>,
    pub python: String,
    pub convert_script: PathBuf,
    pub outtype: String,
    pub quantize: PathBuf,
    pub imatrix: PathBuf,
    pub perplexity: PathBuf,
    pub calibration: PathBuf,
    pub heldout: PathBuf,
    pub ladder: Vec<String>,
    pub baseline_precisions: Vec<String>,
    pub gpu_layers: u32,
    pub tests: Vec<Value>,
    pub reference: String,
    pub tolerance_pct: f64,
    pub max_file_bytes: Option<u64>,
    pub hardware: Option<Value>,
}

pub fn load_pipeline_config(path: &Path) -> io::Result<PipelineConfig> {
    let raw: Value = serde_json::from_slice(&fs::read(path)?).map_err(|e| invalid(format!("pipeline config: {e}")))?;
    if raw.get("schema").and_then(Value::as_str) != Some(PIPELINE_SCHEMA) {
        return Err(invalid(format!("pipeline config schema must be {PIPELINE_SCHEMA}")));
    }
    let tools = raw.get("tools").ok_or_else(|| invalid("pipeline config needs `tools`"))?;
    let strings = |key: &str| -> Vec<String> {
        raw.get(key).and_then(Value::as_array).map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect()).unwrap_or_default()
    };
    let ladder = strings("ladder");
    if ladder.is_empty() {
        return Err(invalid("pipeline `ladder` needs at least one precision"));
    }
    let tests = raw.get("tests").and_then(Value::as_array).cloned().unwrap_or_default();
    for t in &tests {
        s(t, "name")?;
        s(t, "metric")?;
        if t.get("command").and_then(Value::as_array).map_or(true, |c| c.is_empty()) {
            return Err(invalid("each test needs a non-empty `command` array"));
        }
    }
    let tolerance_pct = raw.get("tolerance_pct").and_then(Value::as_f64).unwrap_or(5.0);
    if !(0.0..=100.0).contains(&tolerance_pct) {
        return Err(invalid("tolerance_pct must be within 0..=100"));
    }
    Ok(PipelineConfig {
        name: s(&raw, "name")?.to_owned(),
        source: PathBuf::from(s(&raw, "source")?),
        work: PathBuf::from(s(&raw, "work_dir")?),
        patches: strings("patches").into_iter().map(PathBuf::from).collect(),
        python: tools.get("python").and_then(Value::as_str).unwrap_or("python").to_owned(),
        convert_script: PathBuf::from(s(tools, "convert_hf_to_gguf")?),
        outtype: raw.get("outtype").and_then(Value::as_str).unwrap_or("bf16").to_owned(),
        quantize: PathBuf::from(s(tools, "llama_quantize")?),
        imatrix: PathBuf::from(s(tools, "llama_imatrix")?),
        perplexity: PathBuf::from(s(tools, "llama_perplexity")?),
        calibration: PathBuf::from(s(&raw, "calibration")?),
        heldout: PathBuf::from(s(&raw, "heldout")?),
        ladder,
        baseline_precisions: strings("baseline_precisions"),
        gpu_layers: raw.get("gpu_layers").and_then(Value::as_u64).unwrap_or(99) as u32,
        tests,
        reference: raw.get("reference").and_then(Value::as_str).unwrap_or("").to_owned(),
        tolerance_pct,
        max_file_bytes: raw.get("max_file_bytes").and_then(Value::as_u64),
        hardware: raw.get("hardware").cloned(),
        raw,
    })
}

/// What the run will do, without doing it.
pub fn plan_pipeline(cfg: &PipelineConfig) -> Value {
    let mut cands: Vec<String> = cfg.ladder.iter().map(|q| format!("edited_{q}")).collect();
    cands.extend(cfg.baseline_precisions.iter().map(|q| format!("base_{q}")));
    json!({"schema": "ojon.pipeline_plan.v1", "name": cfg.name, "source": cfg.source, "work_dir": cfg.work,
           "patches": cfg.patches, "ladder": cfg.ladder, "baseline_precisions": cfg.baseline_precisions,
           "candidates": cands, "tests": cfg.tests.iter().map(|t| t["name"].clone()).collect::<Vec<_>>(),
           "reference": cfg.reference, "tolerance_pct": cfg.tolerance_pct, "max_file_bytes": cfg.max_file_bytes,
           "stages": ["patch", "convert", "imatrix", "quantize", "kld", "tests", "recommend"], "status": "PLAN_ONLY"})
}

struct State {
    path: PathBuf,
    done: Map<String, Value>,
}

impl State {
    fn load(work: &Path) -> io::Result<Self> {
        let path = work.join("pipeline-state.json");
        let done = if path.is_file() {
            serde_json::from_slice::<Value>(&fs::read(&path)?)?.as_object().cloned().unwrap_or_default()
        } else {
            Map::new()
        };
        Ok(State { path, done })
    }
    /// A stage is done when it was recorded and every recorded file still has its hash.
    fn is_done(&self, key: &str) -> bool {
        let Some(v) = self.done.get(key) else { return false };
        v.get("files").and_then(Value::as_array).map_or(true, |files| {
            files.iter().all(|f| {
                let p = PathBuf::from(f["path"].as_str().unwrap_or_default());
                p.is_file() && sha256_file(&p).ok().as_deref() == f["sha256"].as_str()
            })
        })
    }
    fn record(&mut self, key: &str, mut entry: Value, files: &[&Path]) -> io::Result<()> {
        let mut hashed = Vec::new();
        for f in files {
            hashed.push(json!({"path": f, "sha256": sha256_file(f)?, "bytes": fs::metadata(f)?.len()}));
        }
        entry["files"] = Value::Array(hashed);
        self.done.insert(key.to_owned(), entry);
        fs::write(&self.path, serde_json::to_vec_pretty(&Value::Object(self.done.clone()))?)
    }
}

fn run(cmd: &mut Command, log: &Path, what: &str) -> io::Result<String> {
    let out = cmd.output().map_err(|e| invalid(format!("{what}: cannot start: {e}")))?;
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    fs::write(log, &text)?;
    if !out.status.success() {
        return Err(invalid(format!("{what} failed ({}); log {}", out.status, log.display())));
    }
    Ok(text)
}

/// Pull "Mean    KLD:   0.012345 ± ..." out of llama-perplexity output.
pub fn parse_mean_kld(text: &str) -> Option<f64> {
    text.lines().find_map(|l| {
        let t = l.trim_start_matches(|c: char| c.is_ascii_digit() || c == '.' || c == ' ' || c == 'I');
        let t = t.trim();
        t.strip_prefix("Mean")?.trim_start().strip_prefix("KLD:")?.split_whitespace().next()?.parse().ok()
    })
}

fn fill(template: &[Value], gguf: &Path, out: &Path, label: &str) -> io::Result<Vec<String>> {
    template
        .iter()
        .map(|a| {
            a.as_str()
                .map(|s| s.replace("{gguf}", &gguf.to_string_lossy()).replace("{out}", &out.to_string_lossy()).replace("{label}", label))
                .ok_or_else(|| invalid("test command items must be strings"))
        })
        .collect()
}

/// Execute every stage (resumable) and return the final receipt.
pub fn run_pipeline(cfg: &PipelineConfig, mut log: impl FnMut(&str)) -> io::Result<Value> {
    fs::create_dir_all(&cfg.work)?;
    let logs = cfg.work.join("logs");
    fs::create_dir_all(&logs)?;
    let mut st = State::load(&cfg.work)?;

    // 1. patches, applied in order; each step produces a new folder
    let mut model_dir = cfg.source.clone();
    let mut patch_receipts = Vec::new();
    for (i, manifest) in cfg.patches.iter().enumerate() {
        let out = cfg.work.join(format!("patched-{i}"));
        let key = format!("patch_{i}");
        if !(st.is_done(&key) && out.is_dir()) {
            if out.exists() {
                return Err(invalid(format!("{} exists from an unfinished run; remove it to redo patch {i}", out.display())));
            }
            log(&format!("[pipeline] patch {i}: {}", manifest.display()));
            let r = run_weight_patch(&prepare_weight_patch(&model_dir, manifest, &out, false)?, |m| log(m))?;
            st.record(&key, json!({"manifest": manifest, "status": r["status"]}), &[&out.join(crate::weight_patch::RECEIPT_FILE)])?;
        }
        patch_receipts.push(serde_json::from_slice::<Value>(&fs::read(out.join(crate::weight_patch::RECEIPT_FILE))?)?);
        model_dir = out;
    }

    // 2. convert patched + (if baselines are asked for) unpatched source
    let gg = cfg.work.join("gguf");
    fs::create_dir_all(&gg)?;
    let mut convert = |tag: &str, dir: &Path, st: &mut State, log: &mut dyn FnMut(&str)| -> io::Result<PathBuf> {
        let out = gg.join(format!("{tag}-{}.gguf", cfg.outtype));
        let key = format!("convert_{tag}");
        if !st.is_done(&key) {
            log(&format!("[pipeline] convert {tag}"));
            let _ = fs::remove_file(&out);
            run(Command::new(&cfg.python).arg(&cfg.convert_script).arg(dir).args(["--outtype", &cfg.outtype, "--outfile"]).arg(&out),
                &logs.join(format!("convert_{tag}.log")), "convert_hf_to_gguf")?;
            st.record(&key, json!({"from": dir}), &[&out])?;
        }
        Ok(out)
    };
    let edited_hp = convert("edited", &model_dir, &mut st, &mut log)?;
    let base_hp = if cfg.baseline_precisions.is_empty() || cfg.patches.is_empty() { None } else { Some(convert("base", &cfg.source, &mut st, &mut log)?) };

    // 3. imatrix per high-precision model
    let mut imat = |tag: &str, hp: &Path, st: &mut State, log: &mut dyn FnMut(&str)| -> io::Result<PathBuf> {
        let out = gg.join(format!("{tag}.imatrix.gguf"));
        let key = format!("imatrix_{tag}");
        if !st.is_done(&key) {
            log(&format!("[pipeline] imatrix {tag}"));
            run(Command::new(&cfg.imatrix).arg("-m").arg(hp).arg("-f").arg(&cfg.calibration).arg("-o").arg(&out)
                    .args(["--output-format", "gguf", "-c", "512", "-ngl", &cfg.gpu_layers.to_string()]),
                &logs.join(format!("imatrix_{tag}.log")), "llama-imatrix")?;
            st.record(&key, json!({}), &[&out])?;
        }
        Ok(out)
    };
    let edited_im = imat("edited", &edited_hp, &mut st, &mut log)?;
    let base_im = match &base_hp { Some(b) => Some(imat("base", b, &mut st, &mut log)?), None => None };

    // 4. quantize
    let mut candidates: Vec<(String, PathBuf, PathBuf)> = Vec::new(); // (label, gguf, its high-precision reference)
    let mut jobs: Vec<(String, &Path, &Path, &str)> = cfg.ladder.iter().map(|q| (format!("edited_{q}"), edited_hp.as_path(), edited_im.as_path(), q.as_str())).collect();
    if let (Some(b), Some(bi)) = (&base_hp, &base_im) {
        jobs.extend(cfg.baseline_precisions.iter().map(|q| (format!("base_{q}"), b.as_path(), bi.as_path(), q.as_str())));
    }
    for (label, hp, im, q) in jobs {
        let out = gg.join(format!("{label}.gguf"));
        let key = format!("quant_{label}");
        if !st.is_done(&key) {
            log(&format!("[pipeline] quantize {label}"));
            run(Command::new(&cfg.quantize).arg("--imatrix").arg(im).arg(hp).arg(&out).arg(q), &logs.join(format!("quant_{label}.log")), "llama-quantize")?;
            st.record(&key, json!({"precision": q}), &[&out])?;
        }
        candidates.push((label, out, hp.to_path_buf()));
    }

    // 5. KLD vs the matching high-precision GGUF on held-out text
    let mut kld = Map::new();
    for hp in [Some(&edited_hp), base_hp.as_ref()].into_iter().flatten() {
        let tag = hp.file_stem().unwrap_or_default().to_string_lossy().to_string();
        let refl = gg.join(format!("{tag}.kldref"));
        let key = format!("kldref_{tag}");
        if !st.is_done(&key) {
            log(&format!("[pipeline] KLD reference {tag}"));
            run(Command::new(&cfg.perplexity).arg("-m").arg(hp).arg("-f").arg(&cfg.heldout).args(["-c", "512", "-ngl", &cfg.gpu_layers.to_string()])
                    .arg("--kl-divergence-base").arg(&refl), &logs.join(format!("kldref_{tag}.log")), "llama-perplexity")?;
            st.record(&key, json!({}), &[&refl])?;
        }
    }
    for (label, gguf, hp) in &candidates {
        let key = format!("kld_{label}");
        if !st.is_done(&key) {
            let tag = hp.file_stem().unwrap_or_default().to_string_lossy().to_string();
            log(&format!("[pipeline] KLD {label}"));
            let text = run(Command::new(&cfg.perplexity).arg("-m").arg(gguf).arg("-f").arg(&cfg.heldout).args(["-c", "512", "-ngl", &cfg.gpu_layers.to_string()])
                               .arg("--kl-divergence-base").arg(gg.join(format!("{tag}.kldref"))).arg("--kl-divergence"),
                           &logs.join(format!("kld_{label}.log")), "llama-perplexity --kl-divergence")?;
            st.record(&key, json!({"mean_kld": parse_mean_kld(&text), "against": hp}), &[])?;
        }
        kld.insert(label.clone(), st.done[&key]["mean_kld"].clone());
    }

    // 6. tests: each writes {"metrics": {...}, "runtime": {...}} to {out}
    let mut metrics: Map<String, Value> = Map::new();
    let mut runtime: Map<String, Value> = Map::new();
    for (label, gguf, _) in &candidates {
        let mut m = Map::new();
        for t in &cfg.tests {
            let name = t["name"].as_str().unwrap_or_default();
            let out = cfg.work.join("tests").join(format!("{name}_{label}.json"));
            fs::create_dir_all(out.parent().unwrap_or(&cfg.work))?;
            let key = format!("test_{name}_{label}");
            if !st.is_done(&key) {
                log(&format!("[pipeline] test {name} on {label}"));
                let argv = fill(t["command"].as_array().map(Vec::as_slice).unwrap_or_default(), gguf, &out, label)?;
                run(Command::new(&argv[0]).args(&argv[1..]), &logs.join(format!("test_{name}_{label}.log")), name)?;
                st.record(&key, json!({}), &[&out])?;
            }
            let r: Value = serde_json::from_slice(&fs::read(&out)?)?;
            let metric = t["metric"].as_str().unwrap_or_default();
            let v = r.get("metrics").and_then(|x| x.get(metric)).cloned().ok_or_else(|| invalid(format!("{} has no metrics.{metric}", out.display())))?;
            m.insert(metric.to_owned(), v);
            if let Some(rt) = r.get("runtime") {
                runtime.insert(label.clone(), rt.clone());
            }
        }
        metrics.insert(label.clone(), Value::Object(m));
    }

    // 7. selection through quality::recommend (smallest within tolerance of the reference, or none)
    let reference = if cfg.reference.is_empty() { format!("edited_{}", cfg.ladder[0]) } else { cfg.reference.clone() };
    let base_metrics = metrics.get(&reference).cloned().ok_or_else(|| invalid(format!("reference {reference} has no metrics")))?;
    let requirements: Vec<Value> = cfg.tests.iter().map(|t| json!({"name": t["metric"], "direction": t.get("direction").cloned().unwrap_or(json!("higher")),
                                                                   "max_relative_degradation_pct": cfg.tolerance_pct})).collect();
    let mut rows = Vec::new();
    let mut excluded = Vec::new();
    for (label, gguf, _) in &candidates {
        let bytes = fs::metadata(gguf)?.len();
        if cfg.max_file_bytes.is_some_and(|max| bytes > max) {
            excluded.push(json!({"label": label, "bytes": bytes, "reason": "over max_file_bytes"}));
            continue;
        }
        let rt = runtime.get(label).cloned().unwrap_or(Value::Null);
        rows.push(json!({"label": label, "bytes": bytes, "evaluation_verified": true, "runtime_load_verified": !rt.is_null(),
                         "metrics": metrics[label], "runtime_measurements": rt}));
    }
    let mut evidence = json!({"schema": "ojon.quality_evidence.v1", "evaluation_suite": cfg.name, "baseline": {"metrics": base_metrics},
                              "requirements": requirements, "candidates": rows});
    if let Some(hw) = &cfg.hardware {
        evidence["hardware_requirements"] = hw.clone();
    }
    let evp = cfg.work.join("quality_evidence.json");
    fs::write(&evp, serde_json::to_vec_pretty(&evidence)?)?;
    let rec = if evidence["candidates"].as_array().is_some_and(|a| !a.is_empty()) {
        recommend(&parse_quality_report(&evp)?)
    } else {
        json!({"recommendation": null, "reason": "no candidate within max_file_bytes"})
    };

    let mut files = Vec::new();
    for (label, gguf, _) in &candidates {
        files.push(json!({"label": label, "path": gguf, "bytes": fs::metadata(gguf)?.len(), "sha256": sha256_file(gguf)?, "mean_kld": kld.get(label)}));
    }
    let receipt = json!({
        "schema": "ojon.pipeline_receipt.v1", "name": cfg.name, "tool": format!("ojon {}", env!("CARGO_PKG_VERSION")),
        "source": cfg.source, "patches": patch_receipts, "high_precision": {"edited": edited_hp, "base": base_hp},
        "candidates": files, "excluded": excluded, "metrics": metrics, "runtime": runtime,
        "reference": reference, "tolerance_pct": cfg.tolerance_pct, "recommendation": rec["recommendation"],
        "selection": rec, "config": cfg.raw,
        "status": if rec["recommendation"].is_null() { "NO_WINNER" } else { "CANDIDATE_SELECTED" },
    });
    fs::write(cfg.work.join("pipeline-receipt.json"), serde_json::to_vec_pretty(&receipt)?)?;
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_kld_line_from_llama_perplexity() {
        let text = "chunk stuff\n0.03.1 I Mean    KLD:   0.013657 ±   0.000210\n0.03.1 I Maximum KLD:  4.2\n";
        assert_eq!(parse_mean_kld(text), Some(0.013657));
        assert_eq!(parse_mean_kld("Mean    KLD:   0.5 ± 0.1"), Some(0.5));
        assert_eq!(parse_mean_kld("no kld here"), None);
    }

    #[test]
    fn config_validation_and_plan() {
        let dir = std::env::temp_dir().join(format!("ojon-pipe-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("cfg.json");
        let mut cfg = json!({"schema": PIPELINE_SCHEMA, "name": "t", "source": "S", "work_dir": "W",
            "tools": {"convert_hf_to_gguf": "c.py", "llama_quantize": "q", "llama_imatrix": "i", "llama_perplexity": "p"},
            "calibration": "cal.txt", "heldout": "held.txt", "ladder": ["Q5_K_M", "Q4_K_M"], "baseline_precisions": ["Q5_K_M"],
            "patches": ["patch.json"],
            "tests": [{"name": "math", "metric": "accuracy", "command": ["python", "eval.py", "{gguf}", "{out}"]}]});
        fs::write(&p, serde_json::to_vec(&cfg).unwrap()).unwrap();
        let c = load_pipeline_config(&p).unwrap();
        let plan = plan_pipeline(&c);
        assert_eq!(plan["candidates"], json!(["edited_Q5_K_M", "edited_Q4_K_M", "base_Q5_K_M"]));
        assert_eq!(fill(&c.tests[0]["command"].as_array().unwrap().clone(), Path::new("m.gguf"), Path::new("o.json"), "x").unwrap()[2], "m.gguf");
        cfg["ladder"] = json!([]);
        fs::write(&p, serde_json::to_vec(&cfg).unwrap()).unwrap();
        assert!(load_pipeline_config(&p).is_err());
        cfg["ladder"] = json!(["Q4_K_M"]);
        cfg["tolerance_pct"] = json!(150.0);
        fs::write(&p, serde_json::to_vec(&cfg).unwrap()).unwrap();
        assert!(load_pipeline_config(&p).is_err());
        fs::remove_dir_all(dir).ok();
    }
}
