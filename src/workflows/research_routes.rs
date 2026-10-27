//! Independently-installed HQQ and OpenVINO/NNCF quantization routes.
//! Do not imply GGUF compatibility or that an installed Python runtime was tested.
//! Every run requires explicit user confirmation in the CLI.
use crate::hf::{inspect_hf_folder, HfFolderInfo};
use crate::jobs::run_with_log;
use crate::resource::{assess_resources, ResourceMode};
use serde_json::{json, Value};
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn bad(msg: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, msg.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResearchKind { Hqq, OpenVino }

impl ResearchKind {
    pub fn parse(input: &str) -> io::Result<Self> {
        match input.to_ascii_lowercase().as_str() {
            "hqq" => Ok(Self::Hqq),
            "openvino" | "nncf" | "openvino-nncf" => Ok(Self::OpenVino),
            _ => Err(bad("research backend must be hqq or openvino")),
        }
    }
    fn id(self) -> &'static str {
        match self { Self::Hqq => "hqq", Self::OpenVino => "openvino_nncf" }
    }
    fn format(self) -> &'static str {
        match self {
            Self::Hqq => "HQQ native qmodel.pt (NOT GGUF)",
            Self::OpenVino => "OpenVINO IR .xml/.bin (NOT GGUF)",
        }
    }
}

#[derive(Clone, Debug)]
pub struct ResearchPlan {
    pub kind: ResearchKind,
    pub source: HfFolderInfo,
    pub output: PathBuf,
    pub executable: PathBuf,
    pub bits: u32,
    pub group: u32,
    pub ratio: f64,
    pub device: String,
}

impl ResearchPlan {
    pub fn as_json(&self) -> Value {
        json!({
            "schema": "ojon.research_route.v1",
            "backend": self.kind.id(),
            "source": self.source.directory.to_string_lossy(),
            "source_weight_bytes": self.source.source_weight_bytes,
            "output": self.output.to_string_lossy(),
            "bits": self.bits,
            "group_size": self.group,
            "ratio": if self.kind == ResearchKind::OpenVino { json!(self.ratio) } else { Value::Null },
            "device_requested": self.device,
            "tool": self.executable.to_string_lossy(),
            "output_format": self.kind.format(),
            "compatible_loader": match self.kind {
                ResearchKind::Hqq => "installed HQQ AutoHQQHFModel.from_quantized, exact model architecture unverified",
                ResearchKind::OpenVino => "OpenVINO Runtime / Optimum Intel, exact model architecture unverified",
            },
            "status": "PLAN_ONLY",
            "model_compatibility_verified": false,
            "runtime_load_verified": false,
            "task_quality_verified": false,
            "device_verified": false,
            "peak_ram_bytes": null,
            "peak_vram_bytes": null,
            "host_memory_preflight_verified": false,
            "source_weights_exceed_28_gib": self.source.source_weight_bytes > 28u64 * 1024 * 1024 * 1024,
            "full_checkpoint_requires_memory_budget": true,
            "downloaded": false,
            "weights_loaded": false,
            "files_written": false,
            "confirmation_required": true,
            "upstream_reference": match self.kind {
                ResearchKind::Hqq => "https://github.com/dropbox/hqq",
                ResearchKind::OpenVino => "https://github.com/huggingface/optimum-intel/blob/main/docs/source/openvino/export.mdx",
            },
            "limitations": match self.kind {
                ResearchKind::Hqq => "HQQ native archive is not GGUF; full source loads into host RAM; no 1-bit/5-bit or VLM path in this adapter.",
                ResearchKind::OpenVino => "OpenVINO INT4 support depends on model, operator and device. Quantization device is managed by upstream, not selected by Ojon.",
            }
        })
    }
}

/// Inspect an ORIGINAL local Hugging Face checkpoint; never load the model.
/// The caller must provide a new external destination. Avoid overwriting source.
pub fn prepare_research(
    kind: ResearchKind, source: &Path, output: &Path, executable: &Path,
    bits: u32, group: u32, ratio: f64, device: &str
) -> io::Result<ResearchPlan> {
    let source = inspect_hf_folder(source)?;
    // This first bridge is causal-LM only. Refuse to drop a VLM's processor
    // or vision tower while exporting what looks like a complete model.
    let config_file = source.directory.join("config.json");
    if fs::metadata(&config_file)?.len() > 2_000_000 {
        return Err(bad("config.json too large for bounded research capability check"));
    }
    let config: Value = serde_json::from_slice(&fs::read(config_file)?)
        .map_err(|_| bad("invalid source config.json"))?;
    let kind_hint = config["model_type"].as_str().unwrap_or("").to_ascii_lowercase();
    if config["vision_config"].is_object()
        || config["vision_tower"].is_string()
        || config["mm_projector_type"].is_string()
        || kind_hint.contains("_vl") || kind_hint.contains("vision")
    {
        return Err(bad("multimodal source requires vision/processor-preserving export; HQQ/OpenVINO causal-only research route refused"));
    }
    if !output.is_absolute() {
        return Err(bad("output directory must be absolute"));
    }
    if !output.file_name().is_some_and(|n| !n.is_empty()) {
        return Err(bad("output directory needs a name"));
    }
    if output.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "output already exists"));
    }
    let parent = output.parent().ok_or_else(|| bad("missing output parent"))?.canonicalize()?;
    if !parent.is_dir() {
        return Err(bad("output parent must be an existing directory"));
    }
    let name = output.file_name().ok_or_else(|| bad("missing output filename"))?;
    let output = parent.join(name);
    if output.starts_with(&source.directory) {
        return Err(bad("output may not be inside the source checkpoint"));
    }
    if !matches!(group, 32 | 64 | 128) {
        return Err(bad("group must be 32, 64 or 128"));
    }
    match kind {
        ResearchKind::Hqq if !matches!(bits, 2 | 3 | 4) => {
            return Err(bad("HQQ route supports 2/3/4 bit only"));
        }
        ResearchKind::OpenVino if bits != 4 => {
            return Err(bad("this NNCF route supports INT4 weight-only only"));
        }
        _ => {}
    }
    if kind == ResearchKind::OpenVino {
        if !ratio.is_finite() || !(0.5..=1.0).contains(&ratio) {
            return Err(bad("OpenVINO INT4 ratio must be 0.5..=1.0"));
        }
        if device != "cpu" {
            return Err(bad("OpenVINO CLI selects execution devices separately; set --device cpu for conversion planning"));
        }
    } else if !matches!(device, "cpu" | "cuda" | "xpu") {
        return Err(bad("HQQ device must be cpu, cuda or xpu"));
    }
    if executable.as_os_str().is_empty() {
        return Err(bad("Python or optimum-cli executable must be given"));
    }
    if executable.is_absolute() && !executable.is_file() {
        return Err(bad("specified tool executable does not exist"));
    }
    Ok(ResearchPlan {
        kind, source, output, executable: executable.to_path_buf(),
        bits, group, ratio, device: device.to_owned(),
    })
}

struct Stage(PathBuf);
impl Drop for Stage { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
fn stage_directory(parent: &Path) -> io::Result<Stage> {
    for attempt in 0..12 {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)
            .map_err(|_| bad("invalid clock"))?.as_nanos();
        let root = parent.join(format!(".ojon-research-{}-{now}-{attempt}", std::process::id()));
        match fs::create_dir(&root) {
            Ok(()) => return Ok(Stage(root)),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(bad("could not create unique staging directory"))
}

fn verified_output(kind: ResearchKind, root: &Path) -> io::Result<()> {
    match kind {
        ResearchKind::Hqq => {
            let weights = root.join("qmodel.pt");
            if !root.join("config.json").is_file() || !weights.is_file() || fs::metadata(weights)?.len() == 0 {
                return Err(io::Error::other("HQQ output missing nonempty qmodel.pt or config.json"));
            }
        }
        ResearchKind::OpenVino => {
            let mut matched = false;
            for entry in fs::read_dir(root)?.take(512) {
                let path = entry?.path();
                if path.extension().and_then(|s| s.to_str()) == Some("xml") {
                    let bin = path.with_extension("bin");
                    if bin.is_file() && fs::metadata(bin)?.len() > 0 {
                        matched = true;
                        break;
                    }
                }
            }
            if !matched {
                return Err(io::Error::other("OpenVINO export missing matching XML and BIN weights"));
            }
        }
    }
    Ok(())
}

/// Run exactly one backend. Ojon does NOT install Python packages or model weights.
/// A successful export is only a format-artifact check, not loaded-runtime validation.
pub fn run_research(plan: &ResearchPlan) -> io::Result<()> {
    run_research_with_policy(plan, false)
}

/// Opt-in escape hatch for deliberately risky/unknown hardware telemetry.
/// Never turns an unverified resource estimate into a compatibility claim.
pub fn run_research_with_policy(plan: &ResearchPlan, allow_resource_risk: bool) -> io::Result<()> {
    let mode = match plan.kind {
        ResearchKind::Hqq => ResourceMode::Hqq,
        ResearchKind::OpenVino => ResourceMode::OpenVino,
    };
    let parent = plan.output.parent().ok_or_else(|| bad("missing output parent"))?;
    let preflight = assess_resources(
        plan.source.source_weight_bytes, parent, mode, &plan.device
    )?;
    eprintln!("Ojon resource preflight: {}", preflight.as_json());
    preflight.enforce(allow_resource_risk)?;
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "output already exists"));
    }
    let parent = plan.output.parent().ok_or_else(|| bad("no parent"))?;
    let staging = stage_directory(parent)?;
    let partial = staging.0.join("candidate");
    let mut command = match plan.kind {
        ResearchKind::Hqq => {
            let mut c = Command::new(&plan.executable);
            c.arg("-c").arg(include_str!("hqq_bridge.py"))
                .arg("--source").arg(&plan.source.directory)
                .arg("--output").arg(&partial)
                .arg("--bits").arg(plan.bits.to_string())
                .arg("--group").arg(plan.group.to_string())
                .arg("--device").arg(&plan.device);
            c
        }
        ResearchKind::OpenVino => {
            let mut c = Command::new(&plan.executable);
            c.arg("export").arg("openvino")
                .arg("--model").arg(&plan.source.directory)
                .arg("--task").arg("text-generation-with-past")
                .arg("--weight-format").arg("int4")
                .arg("--group-size").arg(plan.group.to_string())
                .arg("--ratio").arg(plan.ratio.to_string())
                .arg(&partial);
            c
        }
    };
    command.env("HF_HUB_OFFLINE", "1")
        .env("HF_DATASETS_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("HF_HUB_DISABLE_TELEMETRY", "1");
    run_with_log(&mut command, |line| println!("{line}"))?;
    verified_output(plan.kind, &partial)?;
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "output was created concurrently"));
    }
    fs::rename(&partial, &plan.output)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> (PathBuf, PathBuf) {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("ojon-research-test-{}-{stamp}", std::process::id()));
        let source = root.join("model");
        fs::create_dir_all(&source).unwrap();
        fs::write(source.join("config.json"), r#"{"model_type":"llama"}"#).unwrap();
        fs::write(source.join("model.safetensors"), b"placeholder weights").unwrap();
        (root, source)
    }
    #[test]
    fn independent_routes_and_strict_output_contract() {
        let (root, source) = fixture();
        let out = root.join("hqq-output");
        let hqq = prepare_research(ResearchKind::Hqq, &source, &out, Path::new("python"), 4, 64, 1.0, "xpu").unwrap();
        assert_eq!(hqq.as_json()["output_format"], "HQQ native qmodel.pt (NOT GGUF)");
        assert_eq!(hqq.as_json()["runtime_load_verified"], false);
        assert!(prepare_research(ResearchKind::Hqq, &source, &source.join("packed"), Path::new("python"), 4, 64, 1.0, "cpu").is_err());
        assert!(prepare_research(ResearchKind::Hqq, &source, &out, Path::new("python"), 5, 64, 1.0, "cpu").is_err());
        let ov = prepare_research(ResearchKind::OpenVino, &source, &out, Path::new("optimum-cli"), 4, 128, 0.8, "cpu").unwrap();
        assert_eq!(ov.as_json()["output_format"], "OpenVINO IR .xml/.bin (NOT GGUF)");
        assert!(prepare_research(ResearchKind::OpenVino, &source, &out, Path::new("optimum-cli"), 4, 128, 1.3, "cpu").is_err());
        assert!(prepare_research(ResearchKind::OpenVino, &source, &out, Path::new("optimum-cli"), 4, 128, 1.0, "xpu").is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn require_real_output_artifacts() {
        let (root, _) = fixture();
        assert!(verified_output(ResearchKind::Hqq, &root).is_err());
        fs::write(root.join("qmodel.pt"), b"quantized").unwrap();
        fs::write(root.join("config.json"), b"{}").unwrap();
        assert!(verified_output(ResearchKind::Hqq, &root).is_ok());
        fs::write(root.join("model.xml"), b"<net/>").unwrap();
        assert!(verified_output(ResearchKind::OpenVino, &root).is_err());
        fs::write(root.join("model.bin"), b"weights").unwrap();
        assert!(verified_output(ResearchKind::OpenVino, &root).is_ok());
        fs::remove_dir_all(root).unwrap();
    }
}
