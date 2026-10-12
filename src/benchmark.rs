//! GGUF prefill/decode benchmarks driven by a user-installed public llama-bench.
//! Ojon does not invent throughput, quality, VRAM, or original-model baselines.
//! This module is separate from quantization and never downloads any weights.

use crate::gguf::inspect_gguf;
use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::{self, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

pub const RECEIPT_SCHEMA: &str = "ojon.llama_bench_receipt.v1";
const MAX_BENCH_OUTPUT: usize = 8 * 1024 * 1024;
const MAX_RECEIPT_SIZE: u64 = 2 * 1024 * 1024;

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, message.into())
}

#[derive(Clone, Debug)]
pub struct BenchPlan {
    pub model: PathBuf,
    pub tool: PathBuf,
    pub prompt_tokens: u32,
    pub gen_tokens: u32,
    pub repetitions: u32,
    pub gpu_layers: i32,
    pub model_bytes: u64,
    pub architecture: String,
}

impl BenchPlan {
    pub fn as_json(&self) -> Value {
        json!({
            "schema": "ojon.llama_bench_plan.v1",
            "model": self.model.to_string_lossy(),
            "model_bytes": self.model_bytes,
            "architecture": self.architecture,
            "tool": self.tool.to_string_lossy(),
            "prompt_tokens": self.prompt_tokens,
            "generation_tokens": self.gen_tokens,
            "repetitions": self.repetitions,
            "gpu_layers": self.gpu_layers,
            "external_backend": "llama-bench",
            "weights_loaded": false,
            "memory_fit_verified": false,
            "backend_compatibility_verified": false,
            "runtime_quality_verified": false,
            "original_model_not_required": true,
            "message": "Benchmark runs actual llama.cpp prefill and decode kernels only; not model quality, full chat latency, TTFT, or memory residency."
        })
    }
    pub fn command(&self) -> Command {
        let mut command = Command::new(&self.tool);
        command
            .arg("-m").arg(&self.model)
            .arg("-p").arg(self.prompt_tokens.to_string())
            .arg("-n").arg(self.gen_tokens.to_string())
            .arg("-r").arg(self.repetitions.to_string())
            .arg("-ngl").arg(self.gpu_layers.to_string())
            .arg("-o").arg("json");
        command
    }
}

/// Reads GGUF catalog only. Bench execution is explicitly separate, and large
/// originals need not be loaded in order to measure a quantized candidate.
pub fn prepare_bench(
    model: &Path, tool: &Path, prompt_tokens: u32, gen_tokens: u32,
    repetitions: u32, gpu_layers: i32,
) -> io::Result<BenchPlan> {
    if !(16..=8192).contains(&prompt_tokens) { return Err(invalid("prompt tokens must be 16..=8192")); }
    if !(16..=2048).contains(&gen_tokens) { return Err(invalid("generation tokens must be 16..=2048")); }
    if !(1..=10).contains(&repetitions) { return Err(invalid("repetitions must be 1..=10")); }
    if !(-1..=512).contains(&gpu_layers) { return Err(invalid("GPU layers must be -1..=512")); }
    if tool.as_os_str().is_empty() { return Err(invalid("llama-bench executable must be specified")); }
    let model = model.canonicalize()?;
    let meta = inspect_gguf(&model)?;
    if meta.tensor_count == 0 { return Err(invalid("GGUF has no tensor catalog")); }
    Ok(BenchPlan {
        model, tool: tool.to_path_buf(), prompt_tokens, gen_tokens,
        repetitions, gpu_layers, model_bytes: meta.file_bytes,
        architecture: meta.architecture.unwrap_or_else(|| "unknown".into()),
    })
}

fn positive_number(row: &Value, name: &str) -> io::Result<f64> {
    let value = row.get(name).and_then(Value::as_f64)
        .ok_or_else(|| invalid(format!("llama-bench JSON lacks numeric {name}")))?;
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(format!("llama-bench {name} must be finite and positive")));
    }
    Ok(value)
}
fn nonnegative_number(row: &Value, name: &str) -> io::Result<f64> {
    let value = row.get(name).and_then(Value::as_f64)
        .ok_or_else(|| invalid(format!("llama-bench JSON lacks numeric {name}")))?;
    if !value.is_finite() || value < 0.0 {
        return Err(invalid(format!("llama-bench {name} must be finite and nonnegative")));
    }
    Ok(value)
}
fn must_u64(row: &Value, field: &str) -> io::Result<u64> {
    row.get(field).and_then(Value::as_u64)
        .ok_or_else(|| invalid(format!("llama-bench JSON lacks unsigned {field}")))
}
fn must_str<'a>(row: &'a Value, field: &str) -> io::Result<&'a str> {
    row.get(field).and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| invalid(format!("llama-bench JSON lacks text {field}")))
}
fn metric(row: &Value, expected: u32, kind: &str) -> io::Result<Value> {
    Ok(json!({
        "kind": kind,
        "tokens": expected,
        "tokens_per_second": positive_number(row, "avg_ts")?,
        "stddev_tokens_per_second": nonnegative_number(row, "stddev_ts")?,
        "mean_nanoseconds": must_u64(row, "avg_ns")?,
        "samples_tokens_per_second": row.get("samples_ts").cloned().unwrap_or(Value::Null),
    }))
}
/// Strictly interpret llama-bench's native JSON array. Reject missing, duplicate
/// or mismatched pp/tg measurements rather than inventing results from a log.
pub fn parse_bench_json(raw: &str, plan: &BenchPlan) -> io::Result<Value> {
    if raw.len() > MAX_BENCH_OUTPUT { return Err(invalid("llama-bench JSON exceeded 8 MiB")); }
    let value: Value = serde_json::from_str(raw).map_err(|e| invalid(format!("invalid llama-bench JSON: {e}")))?;
    let rows = value.as_array().ok_or_else(|| invalid("llama-bench output must be an array"))?;
    if rows.len() != 2 { return Err(invalid("expected exactly two llama-bench pp and tg rows")); }
    let pp = rows.iter().find(|r| r["n_prompt"].as_u64() == Some(plan.prompt_tokens as u64)
        && r["n_gen"].as_u64() == Some(0)).ok_or_else(|| invalid("missing matching pp row"))?;
    let tg = rows.iter().find(|r| r["n_prompt"].as_u64() == Some(0)
        && r["n_gen"].as_u64() == Some(plan.gen_tokens as u64)).ok_or_else(|| invalid("missing matching tg row"))?;
    // Compare true configuration emitted by the backend, not our requested CLI.
    for field in ["build_commit", "cpu_info", "gpu_info", "backends", "model_filename",
                  "devices", "split_mode", "type_k", "type_v"] {
        if pp.get(field) != tg.get(field) { return Err(invalid(format!("pp/tg disagree on {field}"))); }
    }
    for field in ["n_gpu_layers", "n_batch", "n_ubatch", "n_threads", "model_n_params"] {
        if pp.get(field) != tg.get(field) { return Err(invalid(format!("pp/tg disagree on {field}"))); }
    }
    let mut requested_gpu_valid = true;
    if let Some(actual) = pp.get("n_gpu_layers").and_then(Value::as_i64) {
        // Some builds report -1 for "all" layers regardless of -ngl 99.
        // Record differences rather than claiming exact GPU residency.
        requested_gpu_valid = plan.gpu_layers < 0 || actual == plan.gpu_layers as i64;
    }
    if !requested_gpu_valid { return Err(invalid("observed GPU layer count differs from request")); }
    let backend_size = must_u64(pp, "model_size")?;
    if backend_size == 0 { return Err(invalid("backend returned zero model size")); }
    let n_params = must_u64(pp, "model_n_params")?;
    if n_params == 0 { return Err(invalid("backend returned zero model parameters")); }
    // Size from GGUF file and backend may legitimately differ for split GGUF;
    // preserve both and do not claim model identification by this measurement.
    let config = json!({
        "build_commit": must_str(pp, "build_commit")?,
        "cpu_info": must_str(pp, "cpu_info")?,
        "gpu_info": must_str(pp, "gpu_info")?,
        "backends": must_str(pp, "backends")?,
        "devices": must_str(pp, "devices")?,
        "split_mode": must_str(pp, "split_mode")?,
        "n_gpu_layers": pp.get("n_gpu_layers").cloned().unwrap_or(Value::Null),
        "n_batch": pp.get("n_batch").cloned().unwrap_or(Value::Null),
        "n_ubatch": pp.get("n_ubatch").cloned().unwrap_or(Value::Null),
        "n_threads": pp.get("n_threads").cloned().unwrap_or(Value::Null),
        "type_k": must_str(pp, "type_k")?,
        "type_v": must_str(pp, "type_v")?,
        "model_n_params": n_params,
        "model_type": pp.get("model_type").cloned().unwrap_or(Value::Null),
        "backend_model_bytes": backend_size,
        "n_cpu_moe": pp.get("n_cpu_moe").cloned().unwrap_or(Value::Null),
        "flash_attn": pp.get("flash_attn").cloned().unwrap_or(Value::Null),
        "use_mmap": pp.get("use_mmap").cloned().unwrap_or(Value::Null),
        "tensor_split": pp.get("tensor_split").cloned().unwrap_or(Value::Null),
        "main_gpu": pp.get("main_gpu").cloned().unwrap_or(Value::Null),
        "no_kv_offload": pp.get("no_kv_offload").cloned().unwrap_or(Value::Null),
    });
    Ok(json!({
        "schema": RECEIPT_SCHEMA,
        "model": plan.model.to_string_lossy(),
        "model_file_bytes": plan.model_bytes,
        "model_architecture": plan.architecture,
        "requested": plan.as_json(),
        "observed": config,
        "prompt_processing": metric(pp, plan.prompt_tokens, "pp")?,
        "token_generation": metric(tg, plan.gen_tokens, "tg")?,
        "benchmark_completed": true,
        "original_baseline_available": false,
        "runtime_load_observed": true,
        "task_quality_verified": false,
        "time_to_first_token_ms": Value::Null,
        "peak_ram_bytes": Value::Null,
        "peak_vram_bytes": Value::Null,
        "metric_scope": "native_llama_bench_prefill_decode_only",
        "warning": "Real kernel throughput is not ChatGPT-style streaming speed, first-token latency, VLM correctness, or quality retention."
    }))
}
/// Explicitly run installed llama-bench (which loads the model) and create a
/// new receipt without overwriting or modifying the original/checkpoint.
pub fn run_bench(plan: &BenchPlan, receipt: &Path) -> io::Result<Value> {
    let target = receipt.to_path_buf();
    if target.extension().and_then(|v| v.to_str()) != Some("json") {
        return Err(invalid("receipt must be a new .json file"));
    }
    let parent = target.parent().filter(|p| !p.as_os_str().is_empty())
        .ok_or_else(|| invalid("receipt must have an existing parent directory"))?;
    if !parent.is_dir() { return Err(invalid("receipt parent directory must already exist")); }
    if target.exists() { return Err(io::Error::new(ErrorKind::AlreadyExists, "receipt already exists; refusing overwrite")); }
    eprintln!("Ojon: real llama-bench loads model weights. Results require an installed backend and sufficient real RAM/VRAM.");
    let output = plan.command().stdin(Stdio::null()).output()?;
    if !output.status.success() {
        let tail = String::from_utf8_lossy(&output.stderr).chars().take(1200).collect::<String>();
        return Err(io::Error::other(format!("llama-bench failed: {}: {tail}", output.status)));
    }
    if output.stdout.len() > MAX_BENCH_OUTPUT { return Err(invalid("llama-bench JSON exceeded 8 MiB")); }
    let stdout = String::from_utf8(output.stdout).map_err(|_| invalid("llama-bench JSON is not UTF-8"))?;
    let parsed = parse_bench_json(&stdout, plan)?;
    // create_new is atomic with respect to collisions. No incomplete file is
    // reported as a successful receipt on a failed write.
    let bytes = serde_json::to_vec_pretty(&parsed).map_err(io::Error::other)?;
    let mut file = OpenOptions::new().write(true).create_new(true).open(&target)?;
    if let Err(e) = file.write_all(&bytes).and_then(|()| file.sync_all()) {
        drop(file);
        let _ = fs::remove_file(&target);
        return Err(e);
    }
    Ok(parsed)
}
fn read_receipt(path: &Path) -> io::Result<Value> {
    let meta = fs::metadata(path)?;
    if meta.len() == 0 || meta.len() > MAX_RECEIPT_SIZE {
        return Err(invalid("benchmark receipt must be 1 byte..2 MiB"));
    }
    let raw: Value = serde_json::from_slice(&fs::read(path)?)
        .map_err(|e| invalid(format!("bad benchmark receipt JSON: {e}")))?;
    if raw["schema"].as_str() != Some(RECEIPT_SCHEMA) || raw["benchmark_completed"] != true {
        return Err(invalid("not a completed Ojon benchmark receipt"));
    }
    for name in ["prompt_processing", "token_generation"] {
        positive_number(&raw[name], "tokens_per_second")?;
        must_u64(&raw[name], "tokens")?;
    }
    Ok(raw)
}
pub fn compare_bench_files(reference: &Path, candidate: &Path) -> io::Result<Value> {
    let base = read_receipt(reference)?;
    let next = read_receipt(candidate)?;
    let mut differences = Vec::new();
    if base["model_architecture"] != next["model_architecture"] {
        differences.push("model_architecture".to_string());
    }
    let same = ["build_commit", "cpu_info", "gpu_info", "backends", "devices", "split_mode",
                "n_gpu_layers", "n_batch", "n_ubatch", "n_threads", "type_k", "type_v",
                "model_n_params", "n_cpu_moe", "flash_attn", "use_mmap", "tensor_split",
                "main_gpu", "no_kv_offload"];
    for item in same {
        if base["observed"][item] != next["observed"][item] {
            differences.push(item.into());
        }
    }
    for item in ["prompt_tokens", "generation_tokens", "repetitions", "gpu_layers"] {
        if base["requested"][item] != next["requested"][item] {
            differences.push(item.into());
        }
    }
    let comparable = differences.is_empty();
    let ratio = |name: &str| -> Option<f64> {
        let left = base[name]["tokens_per_second"].as_f64()?;
        let right = next[name]["tokens_per_second"].as_f64()?;
        if left > 0.0 && right > 0.0 { Some(right / left) } else { None }
    };
    Ok(json!({
        "schema": "ojon.llama_bench_comparison.v1",
        "reference": base["model"],
        "candidate": next["model"],
        "reference_bytes": base["model_file_bytes"],
        "candidate_bytes": next["model_file_bytes"],
        "candidate_size_reduction_pct": base["model_file_bytes"].as_u64()
            .zip(next["model_file_bytes"].as_u64()).and_then(|(a,b)|
                if a>0 {Some((1.0-b as f64/a as f64)*100.0)} else {None}),
        "benchmark_conditions_match": comparable,
        "mismatched_conditions": differences,
        "prefill_speed_ratio_candidate_over_reference": if comparable {ratio("prompt_processing")} else {None},
        "decode_speed_ratio_candidate_over_reference": if comparable {ratio("token_generation")} else {None},
        "original_and_quantized_task_accuracy_compared": false,
        "quality_retention_verified": false,
        "winner": Value::Null,
        "warning": "Performance alone never selects a quantization winner. Compare matched held-out task quality before recommending any precision."
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plan() -> BenchPlan {
        BenchPlan {model:"test.gguf".into(), tool:"llama-bench".into(),
            prompt_tokens:512,gen_tokens:128,repetitions:3,gpu_layers:0,
            model_bytes:2400000000,architecture:"qwen3".into()}
    }
    fn example() -> String {
        let template = json!({
            "build_commit":"abc123", "cpu_info":"test CPU", "gpu_info":"test GPU",
            "backends":"CPU", "model_filename":"test.gguf", "model_type":"qwen3 Q4",
            "model_size":2400000000u64, "model_n_params":4000000000u64,
            "n_batch":2048, "n_ubatch":512,"n_threads":8,"devices":"CPU",
            "split_mode":"layer","type_k":"f16","type_v":"f16","n_gpu_layers":0,
            "n_cpu_moe":0,"flash_attn":0,"use_mmap":true,"tensor_split":"0",
            "main_gpu":0,"no_kv_offload":false,"avg_ns":90000000u64,
            "avg_ts":100.0,"stddev_ts":1.0
        });
        let mut a = template.clone();
        a["n_prompt"] = json!(512); a["n_gen"] = json!(0);
        let mut b = template;
        b["n_prompt"] = json!(0); b["n_gen"] = json!(128);
        b["avg_ts"] = json!(44.5);
        json!([a,b]).to_string()
    }
    #[test] fn reads_only_real_json_metrics_not_quality() {
        let receipt=parse_bench_json(&example(),&plan()).unwrap();
        assert_eq!(receipt["prompt_processing"]["tokens_per_second"],100.0);
        assert_eq!(receipt["token_generation"]["tokens_per_second"],44.5);
        assert_eq!(receipt["task_quality_verified"], false);
        assert!(receipt["peak_vram_bytes"].is_null());
    }
    #[test] fn rejects_missing_or_mismatched_benchmark_rows() {
        let value: Value=serde_json::from_str(&example()).unwrap();
        let mut rows=value.as_array().unwrap().clone();
        rows[1]["n_gen"]=json!(42);
        assert!(parse_bench_json(&Value::Array(rows).to_string(), &plan()).is_err());
        let mut rows=value.as_array().unwrap().clone();
        rows[1]["backends"]=json!("Vulkan");
        assert!(parse_bench_json(&Value::Array(rows).to_string(), &plan()).is_err());
        let mut rows=value.as_array().unwrap().clone();
        rows[0]["avg_ts"]=json!(-1.0);
        assert!(parse_bench_json(&Value::Array(rows).to_string(), &plan()).is_err());
    }
    #[test] fn comparison_requires_same_real_hardware_and_test_settings() {
        let baseline=parse_bench_json(&example(),&plan()).unwrap();
        let mut candidate=baseline.clone();
        candidate["prompt_processing"]["tokens_per_second"]=json!(200.0);
        candidate["token_generation"]["tokens_per_second"]=json!(80.0);
        let mut path=std::env::temp_dir();
        path.push(format!("ojon_bench_test_{}_{}.json", std::process::id(),
            std::thread::current().name().unwrap_or("t").replace(':', "_")));
        let mut other=path.clone();
        other.set_extension("candidate.json");
        fs::write(&path,serde_json::to_vec(&baseline).unwrap()).unwrap();
        fs::write(&other,serde_json::to_vec(&candidate).unwrap()).unwrap();
        let result=compare_bench_files(&path,&other).unwrap();
        assert_eq!(result["benchmark_conditions_match"],true);
        assert_eq!(result["prefill_speed_ratio_candidate_over_reference"],2.0);
        assert!(result["winner"].is_null());
        candidate["observed"]["gpu_info"]=json!("different card");
        fs::write(&other,serde_json::to_vec(&candidate).unwrap()).unwrap();
        let result=compare_bench_files(&path,&other).unwrap();
        assert_eq!(result["benchmark_conditions_match"],false);
        assert!(result["decode_speed_ratio_candidate_over_reference"].is_null());
        let _=fs::remove_file(path);let _=fs::remove_file(other);
    }
}
