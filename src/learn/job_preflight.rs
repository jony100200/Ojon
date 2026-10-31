//! Ojon Learn V1 preflight: every check reports PASS / FAIL / NOT_RUN with an
//! actionable reason. Nothing here starts a trainer, downloads, or mutates inputs.

use super::knowledge::sha256_hex;
use crate::core::resource::{assess_resources, ResourceMode};
use crate::eval::task_eval::parse_eval_suite;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};

pub const SCHEMA: &str = "ojon.learn.v1";
const MAX_JSONL_BYTES: u64 = 256 * 1024 * 1024;
const MIN_SCRATCH_BYTES: u64 = 512 * 1024 * 1024;
const MIN_ADAPTER_BYTES: u64 = 256 * 1024 * 1024;
/// Quantized formats that cannot be LoRA-trained in a one-click flow.
const NON_TRAINABLE_QUANT: [&str; 8] = [
    "gptq", "awq", "gguf", "exl2", "exl3", "aqlm", "marlin", "hqq",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CheckStatus {
    Pass,
    Fail,
    NotRun,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckResult {
    pub name: String,
    pub status: CheckStatus,
    pub reason: String,
}

fn chk(name: &str, status: CheckStatus, reason: impl Into<String>) -> CheckResult {
    CheckResult {
        name: name.into(),
        status,
        reason: reason.into(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceInfo {
    pub vendor: String,
    pub name: String,
    pub vram_bytes: u64,
    pub integrated: bool,
}

/// One chosen device. Devices are never pooled (Intel 24 GB + NVIDIA 16 GB != 40 GB).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeviceChoice {
    pub family: String,
    pub name: String,
    pub vram_bytes: u64,
    pub trainer_path: String,
}

/// Test/CI injection points; `None` means measure the real host.
#[derive(Clone, Debug, Default)]
pub struct PreflightEnv {
    pub devices: Option<Vec<DeviceInfo>>,
    pub free_disk_bytes: Option<u64>,
    /// Set only after an Intel XPU forward+backward+optimizer gate has been proven.
    pub xpu_gate_proven: bool,
}

#[derive(Clone, Debug)]
pub struct PreflightInput {
    pub model_dir: PathBuf,
    pub dataset_file: PathBuf,
    pub heldout_suite: Option<PathBuf>,
    pub work_root: PathBuf,
    pub device_request: String,
    pub trainer_is_mock: bool,
    pub trainer_label: String,
    pub min_accepted: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PreflightReport {
    pub schema: String,
    pub passed: bool,
    pub checks: Vec<CheckResult>,
    pub model_fingerprint: Option<String>,
    pub model_weight_bytes: u64,
    pub dataset_sha256: Option<String>,
    pub heldout_suite_sha256: Option<String>,
    pub accepted_rows: Option<u64>,
    pub rejected_rows: Option<u64>,
    pub device: Option<DeviceChoice>,
    pub required_disk_bytes: u64,
    pub free_disk_bytes: Option<u64>,
    pub offline: bool,
    pub trainer: String,
    pub real_gpu_training: String,
}

/// Accepted training rows, normalised to instruction/output.
pub struct DatasetScan {
    pub rows: Vec<(String, String)>,
    pub rejected: u64,
    pub sha256: String,
    pub is_parquet: bool,
}

fn normalize_row(v: &Value) -> Option<(String, String)> {
    // Unverified traces are not successful labels.
    if v.get("verified") == Some(&Value::Bool(false)) {
        return None;
    }
    let s = |k: &str| {
        v.get(k)
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|x| !x.is_empty())
    };
    for (ik, ok) in [("instruction", "output"), ("prompt", "response")] {
        if let (Some(i), Some(o)) = (s(ik), s(ok)) {
            return match s("input") {
                Some(extra) => Some((format!("{i}\n{extra}"), o.into())),
                None => Some((i.into(), o.into())),
            };
        }
    }
    let msgs = v.get("messages")?.as_array()?;
    let last = |role: &str| {
        msgs.iter()
            .rev()
            .filter(|m| m.get("role").and_then(Value::as_str) == Some(role))
            .filter_map(|m| m.get("content").and_then(Value::as_str))
            .map(str::trim)
            .find(|c| !c.is_empty())
            .map(str::to_owned)
    };
    Some((last("user")?, last("assistant")?))
}

pub fn scan_dataset(path: &Path) -> Result<DatasetScan, String> {
    let meta = fs::metadata(path).map_err(|e| format!("dataset not readable: {e}"))?;
    if !meta.is_file() {
        return Err("dataset path must be a file (.jsonl or .parquet)".into());
    }
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if !matches!(ext.as_str(), "jsonl" | "parquet") {
        return Err("dataset must be .jsonl or .parquet".into());
    }
    if meta.len() == 0 || meta.len() > MAX_JSONL_BYTES {
        return Err("dataset must be 1 byte..256 MiB in V1".into());
    }
    let bytes = fs::read(path).map_err(|e| format!("dataset read failed: {e}"))?;
    let sha256 = sha256_hex(&bytes);
    if ext == "parquet" {
        if !(bytes.starts_with(b"PAR1") && bytes.ends_with(b"PAR1")) {
            return Err("file has .parquet extension but no PAR1 magic".into());
        }
        return Ok(DatasetScan {
            rows: Vec::new(),
            rejected: 0,
            sha256,
            is_parquet: true,
        });
    }
    let text = String::from_utf8_lossy(&bytes);
    let (mut rows, mut rejected) = (Vec::new(), 0u64);
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        match serde_json::from_str::<Value>(line)
            .ok()
            .as_ref()
            .and_then(normalize_row)
        {
            Some(r) => rows.push(r),
            None => rejected += 1,
        }
    }
    Ok(DatasetScan {
        rows,
        rejected,
        sha256,
        is_parquet: false,
    })
}

pub struct ModelScan {
    pub weight_bytes: u64,
    pub fingerprint: String,
}

fn is_weight_file(n: &str) -> bool {
    n.ends_with(".safetensors") || (n.starts_with("pytorch_model") && n.ends_with(".bin"))
}

/// Returns checks plus scan data when the directory is a trainable HF checkpoint.
pub fn scan_model(dir: &Path) -> (Vec<CheckResult>, Option<ModelScan>) {
    let name = "model_trainable";
    let fail = |why: String| (vec![chk(name, CheckStatus::Fail, why)], None);
    if dir.is_file() {
        let gguf = dir
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("gguf"));
        return fail(if gguf {
            "GGUF is inference-only: Ojon Learn needs a trainable base (HF checkpoint: config.json + tokenizer + safetensors), not a one-click conversion"
                .into()
        } else {
            "model path must be a Hugging Face checkpoint directory".into()
        });
    }
    let Ok(entries) = fs::read_dir(dir) else {
        return fail("model directory does not exist or is unreadable".into());
    };
    let mut files: Vec<(String, u64)> = entries
        .flatten()
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            m.is_file()
                .then(|| (e.file_name().to_string_lossy().into_owned(), m.len()))
        })
        .collect();
    files.sort();
    let has = |f: &dyn Fn(&str) -> bool| files.iter().any(|(n, _)| f(&n.to_ascii_lowercase()));
    if has(&|n| n.ends_with(".gguf")) && !has(&is_weight_file) {
        return fail(
            "directory holds only GGUF: inference-only. Needs a trainable base, not a one-click conversion"
                .into(),
        );
    }
    let mut missing = Vec::new();
    if !has(&|n| n == "config.json") {
        missing.push("config.json");
    }
    if !has(&|n| n == "tokenizer.json" || n == "tokenizer.model" || n == "tokenizer_config.json") {
        missing.push("tokenizer (tokenizer.json/tokenizer.model/tokenizer_config.json)");
    }
    if !has(&is_weight_file) {
        missing.push("weights (*.safetensors or pytorch_model*.bin)");
    }
    if !missing.is_empty() {
        return fail(format!(
            "not a trainable HF checkpoint; missing: {}",
            missing.join(", ")
        ));
    }
    let cfg_bytes = fs::read(dir.join("config.json")).unwrap_or_default();
    let Ok(cfg) = serde_json::from_slice::<Value>(&cfg_bytes) else {
        return fail("config.json is not valid JSON".into());
    };
    let method = cfg["quantization_config"]["quant_method"]
        .as_str()
        .unwrap_or("")
        .to_ascii_lowercase();
    if NON_TRAINABLE_QUANT.contains(&method.as_str()) {
        return fail(format!(
            "pre-quantized '{method}' checkpoint is not LoRA-trainable; needs a trainable base, not a one-click conversion"
        ));
    }
    let weight_bytes = files
        .iter()
        .filter(|(n, _)| is_weight_file(&n.to_ascii_lowercase()))
        .map(|f| f.1)
        .sum();
    let mut manifest = cfg_bytes;
    for (n, len) in &files {
        manifest.extend_from_slice(format!("\n{n}:{len}").as_bytes());
    }
    let scan = ModelScan {
        weight_bytes,
        fingerprint: sha256_hex(&manifest),
    };
    let why = "HF checkpoint with config.json, tokenizer and weights (fingerprint = config bytes + file name/size manifest, not a full weight hash)";
    (vec![chk(name, CheckStatus::Pass, why)], Some(scan))
}

pub fn choose_device(
    request: &str,
    devices: &[DeviceInfo],
    mock: bool,
    xpu_proven: bool,
) -> (CheckResult, Option<DeviceChoice>) {
    let name = "device";
    let best = |vendor: &str| {
        devices
            .iter()
            .filter(|d| d.vendor.eq_ignore_ascii_case(vendor) && !d.integrated)
            .max_by_key(|d| d.vram_bytes)
    };
    let family = match request {
        "auto" if best("nvidia").is_some() => "cuda",
        "auto" if best("intel").is_some() => "xpu",
        "auto" => "cpu",
        "cuda" | "xpu" | "cpu" => request,
        _ => {
            let why = "device must be auto, cuda, xpu or cpu";
            return (chk(name, CheckStatus::Fail, why), None);
        }
    };
    let mk = |dev: Option<&DeviceInfo>, path: &str| DeviceChoice {
        family: family.into(),
        name: dev.map_or("host CPU".into(), |d| d.name.clone()),
        vram_bytes: dev.map_or(0, |d| d.vram_bytes),
        trainer_path: path.into(),
    };
    match family {
        "cuda" => match best("nvidia") {
            Some(d) => {
                let why = format!(
                    "NVIDIA {} ({} bytes VRAM); single device, never pooled with other GPUs",
                    d.name, d.vram_bytes
                );
                (
                    chk(name, CheckStatus::Pass, why),
                    Some(mk(Some(d), "cuda_supported")),
                )
            }
            None => (
                chk(
                    name,
                    CheckStatus::Fail,
                    "no discrete NVIDIA GPU detected for device=cuda",
                ),
                None,
            ),
        },
        "xpu" => match best("intel") {
            Some(d) if xpu_proven => (
                chk(
                    name,
                    CheckStatus::Pass,
                    format!("Intel {} with proven XPU gate", d.name),
                ),
                Some(mk(Some(d), "xpu_proven")),
            ),
            Some(d) => {
                let why = format!(
                    "Intel {} detected but the XPU forward+backward+optimizer trainer gate is UNPROVEN; train on NVIDIA or run CPU prep only",
                    d.name
                );
                (
                    chk(name, CheckStatus::Fail, why),
                    Some(mk(Some(d), "xpu_unproven")),
                )
            }
            None => (
                chk(
                    name,
                    CheckStatus::Fail,
                    "no discrete Intel GPU detected for device=xpu",
                ),
                None,
            ),
        },
        _ if mock => (
            chk(
                name,
                CheckStatus::Pass,
                "CPU is prep-only: acceptable for mock/dry-run, never a real training pass",
            ),
            Some(mk(None, "cpu_prep_only")),
        ),
        _ => (
            chk(
                name,
                CheckStatus::Fail,
                "CPU is prep-only: real training needs an NVIDIA GPU",
            ),
            Some(mk(None, "cpu_prep_only")),
        ),
    }
}

fn nearest_existing(path: &Path) -> PathBuf {
    let mut p = path.to_path_buf();
    while !p.exists() {
        if !p.pop() {
            return PathBuf::from(".");
        }
    }
    p
}

fn parse_devices(assessed: &[Value]) -> Vec<DeviceInfo> {
    assessed
        .iter()
        .map(|d| DeviceInfo {
            vendor: d["vendor"].as_str().unwrap_or("").to_ascii_lowercase(),
            name: d["device_name"].as_str().unwrap_or("unknown").into(),
            vram_bytes: d["total_vram_bytes"].as_u64().unwrap_or(0),
            integrated: d["is_integrated"].as_bool().unwrap_or(false),
        })
        .collect()
}

pub fn required_disk(weight_bytes: u64) -> u64 {
    (weight_bytes / 8).max(MIN_ADAPTER_BYTES) + MIN_SCRATCH_BYTES
}

fn heldout_checks(input: &PreflightInput, scan: Option<&DatasetScan>, rep: &mut PreflightReport) {
    let name = "heldout_suite";
    let Some(path) = &input.heldout_suite else {
        let why = "no frozen held-out suite supplied; Verify needs an ojon.task_suite.v1 file";
        rep.checks.push(chk(name, CheckStatus::Fail, why));
        return;
    };
    let suite = match parse_eval_suite(path) {
        Ok(s) => s,
        Err(e) => {
            rep.checks.push(chk(
                name,
                CheckStatus::Fail,
                format!("held-out suite invalid: {e}"),
            ));
            return;
        }
    };
    rep.heldout_suite_sha256 = fs::read(path).ok().map(|b| sha256_hex(&b));
    let why = format!(
        "suite '{}' with {} tasks, hash frozen",
        suite.id,
        suite.tasks.len()
    );
    rep.checks.push(chk(name, CheckStatus::Pass, why));
    let overlap = "heldout_overlap";
    let Some(scan) = scan.filter(|s| !s.is_parquet) else {
        let why = "dataset rows not inspectable (parquet or unreadable): overlap cannot be ruled out; convert to JSONL";
        rep.checks.push(chk(overlap, CheckStatus::NotRun, why));
        return;
    };
    let rows: Vec<String> = scan.rows.iter().map(|r| r.0.to_lowercase()).collect();
    let hits: Vec<&str> = suite
        .tasks
        .iter()
        .filter(|t| {
            let p = t.prompt.trim().to_lowercase();
            rows.iter()
                .any(|r| r.contains(&p) || p.contains(r.as_str()))
        })
        .map(|t| t.id.as_str())
        .collect();
    rep.checks.push(if hits.is_empty() {
        chk(
            overlap,
            CheckStatus::Pass,
            "no held-out prompt appears in the training rows (exact/containment check)",
        )
    } else {
        chk(
            overlap,
            CheckStatus::Fail,
            format!(
                "held-out tasks leak into training data: {}",
                hits.join(", ")
            ),
        )
    });
}

fn blank_report(input: &PreflightInput) -> PreflightReport {
    PreflightReport {
        schema: SCHEMA.into(),
        passed: false,
        checks: Vec::new(),
        model_fingerprint: None,
        model_weight_bytes: 0,
        dataset_sha256: None,
        heldout_suite_sha256: None,
        accepted_rows: None,
        rejected_rows: None,
        device: None,
        required_disk_bytes: 0,
        free_disk_bytes: None,
        offline: true,
        trainer: input.trainer_label.clone(),
        real_gpu_training: "NOT_RUN".into(),
    }
}

fn dataset_checks(input: &PreflightInput, rep: &mut PreflightReport) -> Option<DatasetScan> {
    let scan = match scan_dataset(&input.dataset_file) {
        Ok(s) => s,
        Err(e) => {
            rep.checks.push(chk("dataset", CheckStatus::Fail, e));
            return None;
        }
    };
    rep.dataset_sha256 = Some(scan.sha256.clone());
    if scan.is_parquet {
        let why = "valid parquet container; rows not counted in V1 (accepted/rejected NOT_RUN)";
        rep.checks.push(chk("dataset", CheckStatus::Pass, why));
    } else {
        rep.accepted_rows = Some(scan.rows.len() as u64);
        rep.rejected_rows = Some(scan.rejected);
        let ok = scan.rows.len() as u64 >= input.min_accepted;
        let why = format!(
            "{} accepted, {} rejected (need >= {} accepted; unverified traces are rejected)",
            scan.rows.len(),
            scan.rejected,
            input.min_accepted
        );
        let status = if ok {
            CheckStatus::Pass
        } else {
            CheckStatus::Fail
        };
        rep.checks.push(chk("dataset", status, why));
    }
    Some(scan)
}

pub fn run_preflight(input: &PreflightInput, env: &PreflightEnv) -> PreflightReport {
    let mut rep = blank_report(input);
    let (mchecks, model) = scan_model(&input.model_dir);
    rep.checks.extend(mchecks);
    if let Some(m) = &model {
        rep.model_fingerprint = Some(m.fingerprint.clone());
        rep.model_weight_bytes = m.weight_bytes;
    }
    let scan = dataset_checks(input, &mut rep);
    heldout_checks(input, scan.as_ref(), &mut rep);
    let measured = if env.devices.is_none() || env.free_disk_bytes.is_none() {
        let parent = nearest_existing(&input.work_root);
        assess_resources(rep.model_weight_bytes, &parent, ResourceMode::Pack, "cpu").ok()
    } else {
        None
    };
    let devices = env.devices.clone().unwrap_or_else(|| {
        measured
            .as_ref()
            .map(|a| parse_devices(&a.gpu_devices))
            .unwrap_or_default()
    });
    let (dcheck, dev) = choose_device(
        &input.device_request,
        &devices,
        input.trainer_is_mock,
        env.xpu_gate_proven,
    );
    rep.checks.push(dcheck);
    rep.device = dev;
    rep.required_disk_bytes = required_disk(rep.model_weight_bytes);
    rep.free_disk_bytes = env
        .free_disk_bytes
        .or_else(|| measured.as_ref().and_then(|a| a.free_disk_bytes));
    let need = rep.required_disk_bytes;
    rep.checks.push(match rep.free_disk_bytes {
        Some(free) if free >= need => chk(
            "disk_space",
            CheckStatus::Pass,
            format!("{free} bytes free >= {need} required"),
        ),
        Some(free) => chk(
            "disk_space",
            CheckStatus::Fail,
            format!("only {free} bytes free, {need} required at the destination"),
        ),
        None => chk(
            "disk_space",
            CheckStatus::Fail,
            "free disk space could not be measured",
        ),
    });
    let why = "local absolute model/dataset paths only; trainer is launched with HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 (env-enforced, not a network sandbox); no downloads";
    rep.checks.push(chk("offline", CheckStatus::Pass, why));
    rep.passed = rep.checks.iter().all(|c| c.status == CheckStatus::Pass);
    rep
}
