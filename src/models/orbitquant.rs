//! MiniMax-H3 FL2VA / Ref2VA W4A4 conversion via an installed OrbitQuant.
 //! A valid packed artifact is not proof of video quality or runtime speed.
use crate::jobs::run_with_log;
use serde_json::{json, Value};
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MiniMaxComponent {
    Fl2va,
    Ref2va,
    TextEncoder,
}

impl MiniMaxComponent {
    pub fn parse(value: &str) -> io::Result<Self> {
        match value {
            "fl2va" => Ok(Self::Fl2va),
            "ref2va" => Ok(Self::Ref2va),
            "text_encoder" => Ok(Self::TextEncoder),
            _ => Err(invalid("component must be fl2va, ref2va or text_encoder")),
        }
    }
    pub fn directory(self) -> &'static str {
        match self {
            Self::Fl2va => "transformer",
            Self::Ref2va => "transformer_ref",
            Self::TextEncoder => "text_encoder",
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Fl2va => "fl2va",
            Self::Ref2va => "ref2va",
            Self::TextEncoder => "text_encoder",
        }
    }
}

#[derive(Clone, Debug)]
pub struct MiniMaxW4A4Plan {
    pub source: PathBuf,
    pub output: PathBuf,
    pub python: PathBuf,
    pub component: MiniMaxComponent,
    pub device: String,
}

impl MiniMaxW4A4Plan {
    pub fn json(&self) -> Value {
        json!({
            "schema": "ojon.minimax_h3_w4a4.v1",
            "status": "PLAN_ONLY",
            "source": self.source,
            "output": self.output,
            "component": self.component.name(),
            "orbitquant_component": self.component.directory(),
            "python": self.python,
            "device_requested": self.device,
            "bits": {"weights": 4, "activations": 4},
            "backend": "orbitquant",
            "export": "OrbitQuant packed artifact directory; NOT GGUF",
            "visual_quality": "NOT_EVALUATED",
            "runtime_inference": "NOT_TESTED",
            "offline": true,
        })
    }
}

fn bounded_text(path: &Path) -> io::Result<String> {
    if fs::metadata(path)?.len() > 2 * 1024 * 1024 {
        return Err(invalid("model JSON configuration exceeds 2 MiB"));
    }
    fs::read_to_string(path)
}

fn has_safetensors(directory: &Path) -> io::Result<bool> {
    let mut count = 0;
    for entry in fs::read_dir(directory)? {
        count += 1;
        if count > 16384 {
            return Err(invalid("too many component files"));
        }
        let entry = entry?;
        if entry.file_type()?.is_file()
            && entry.path().extension().and_then(|v| v.to_str()) == Some("safetensors")
        {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Plan is local and read-only: no Python, network, or GPU operation.
pub fn prepare_minimax_w4a4(
    source: &Path,
    output: &Path,
    python: &Path,
    component: MiniMaxComponent,
    device: &str,
) -> io::Result<MiniMaxW4A4Plan> {
    let allowed_device = matches!(device, "auto" | "cpu" | "cuda" | "xpu")
        || ["cuda:", "xpu:"].iter().any(|prefix| {
            device.strip_prefix(prefix).is_some_and(|suffix| {
                !suffix.is_empty() && suffix.bytes().all(|v| v.is_ascii_digit())
            })
        });
    if !allowed_device {
        return Err(invalid("device must be auto, cpu, cuda[:N] or xpu[:N]"));
    }
    if python.as_os_str().is_empty() || (python.is_absolute() && !python.is_file()) {
        return Err(invalid("configured Python executable is unavailable"));
    }
    let source = source.canonicalize()?;
    if !source.is_dir() {
        return Err(invalid("MiniMax-H3 W4A4 requires a local Diffusers model directory"));
    }
    let manifest = ["modular_model_index.json", "model_index.json"]
        .iter()
        .map(|name| source.join(name))
        .find(|path| path.is_file())
        .ok_or_else(|| invalid("expected model_index.json or modular_model_index.json; unpack the full local MiniMax-H3 Diffusers pipeline"))?;
    let text = bounded_text(&manifest)?.to_ascii_lowercase();
    if !text.contains("minimax") {
        return Err(invalid("model index does not identify MiniMax-H3"));
    }
    let component_dir = source.join(component.directory());
    if !component_dir.is_dir()
        || !component_dir.join("config.json").is_file()
        || !has_safetensors(&component_dir)?
    {
        return Err(invalid(format!(
            "missing {}/config.json or local safetensors weights; FL2VA and Ref2VA are separate checkpoints",
            component.directory()
        )));
    }
    if let Ok(config) = bounded_text(&component_dir.join("config.json")) {
        if matches!(component, MiniMaxComponent::Fl2va | MiniMaxComponent::Ref2va)
            && !config.to_ascii_lowercase().contains("minimax")
        {
            return Err(invalid("selected denoiser config is not MiniMax-H3"));
        }
    }
    let parent = output.parent().filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new(".")).canonicalize()?;
    let filename = output.file_name().filter(|v| !v.is_empty())
        .ok_or_else(|| invalid("missing output directory name"))?;
    let output = parent.join(filename);
    if output.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "output already exists"));
    }
    if output.starts_with(&source) {
        return Err(invalid("output cannot be placed inside the source model"));
    }
    Ok(MiniMaxW4A4Plan {
        source, output, python: python.to_path_buf(), component, device: device.to_owned(),
    })
}

struct StagedDirectory(PathBuf);
impl Drop for StagedDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Invokes the real, independently installed OrbitQuant conversion and validator.
/// No network downloads, shell strings, overwrite or synthetic success receipt.
pub fn run_minimax_w4a4(
    plan: &MiniMaxW4A4Plan,
    mut log: impl FnMut(&str),
) -> io::Result<Value> {
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "output already exists"));
    }
    let nonce = SystemTime::now().duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?.as_nanos();
    let name = plan.output.file_name().unwrap().to_string_lossy();
    let stage = StagedDirectory(plan.output.with_file_name(format!(".{name}.ojon-{}-{nonce}.partial", std::process::id())));
    if stage.0.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "staging path exists"));
    }
    let mut convert = Command::new(&plan.python);
    convert.arg("-m").arg("orbitquant.cli.main")
        .arg("quantize")
        .arg("--model-id").arg(&plan.source)
        .arg("--output").arg(&stage.0)
        .arg("--component").arg(plan.component.directory())
        .arg("--weight-bits").arg("4")
        .arg("--activation-bits").arg("4")
        .arg("--target-policy").arg("auto")
        .arg("--staging-mode").arg("streaming")
        .arg("--dtype").arg("bfloat16")
        .arg("--device").arg(&plan.device)
        .env("HF_HUB_OFFLINE", "1")
        .env("HF_DATASETS_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("WANDB_MODE", "offline");
    log("MiniMax-H3 W4A4: running installed OrbitQuant conversion");
    run_with_log(&mut convert, &mut log)?;
    if !stage.0.is_dir() || !has_safetensors(&stage.0)? {
        return Err(io::Error::new(ErrorKind::InvalidData,
            "OrbitQuant returned without a packed safetensors artifact"));
    }
    let mut verify = Command::new(&plan.python);
    verify.arg("-m").arg("orbitquant.cli.main")
        .arg("validate-artifact").arg("--artifact").arg(&stage.0)
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1");
    log("MiniMax-H3 W4A4: validating packed artifact with OrbitQuant");
    run_with_log(&mut verify, &mut log)?;
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "output created concurrently"));
    }
    fs::rename(&stage.0, &plan.output)?;
    Ok(json!({
        "schema": "ojon.minimax_h3_w4a4.v1",
        "status": "PACKED_ARTIFACT_VALIDATED",
        "component": plan.component.name(),
        "model": plan.output,
        "backend": "orbitquant",
        "weight_bits": 4,
        "activation_bits": 4,
        "runtime_inference": "NOT_TESTED",
        "visual_quality": "NOT_EVALUATED",
        "note": "Conversion and artifact validation passed. Load and generate with a compatible OrbitQuant runtime before quality certification."
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Scratch(PathBuf);
    impl Drop for Scratch { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
    fn fixture() -> Scratch {
        let nonce = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("ojon-h3-w4a4-{}-{nonce}", std::process::id()));
        fs::create_dir_all(root.join("model/transformer")).unwrap();
        fs::write(root.join("model/model_index.json"), r#"{"_class_name":"MiniMaxH3Pipeline"}"#).unwrap();
        fs::write(root.join("model/transformer/config.json"),
            r#"{"_class_name":"MiniMaxH3Transformer3DModel"}"#).unwrap();
        fs::write(root.join("model/transformer/model.safetensors"), [0u8; 16]).unwrap();
        Scratch(root)
    }
    #[test]
    fn fl2va_plan_is_read_only_and_not_quality_certified() {
        let s = fixture();
        let result = prepare_minimax_w4a4(&s.0.join("model"), &s.0.join("output"),
            Path::new("python"), MiniMaxComponent::Fl2va, "cuda").unwrap();
        assert_eq!(result.json()["visual_quality"], "NOT_EVALUATED");
        assert_eq!(result.json()["orbitquant_component"], "transformer");
        assert!(!result.output.exists());
    }
    #[test]
    fn rejects_non_h3_and_missing_variant() {
        let s = fixture();
        assert!(prepare_minimax_w4a4(&s.0.join("model"), &s.0.join("result"),
            Path::new("python"), MiniMaxComponent::Ref2va, "cuda").is_err());
        fs::write(s.0.join("model/model_index.json"), r#"{"_class_name":"OtherModel"}"#).unwrap();
        assert!(prepare_minimax_w4a4(&s.0.join("model"), &s.0.join("result"),
            Path::new("python"), MiniMaxComponent::Fl2va, "cuda").is_err());
    }
    #[test]
    fn refuses_existing_output_and_source_tree() {
        let s = fixture();
        fs::create_dir(s.0.join("output")).unwrap();
        assert!(prepare_minimax_w4a4(&s.0.join("model"), &s.0.join("output"),
            Path::new("python"), MiniMaxComponent::Fl2va, "cuda").is_err());
        assert!(prepare_minimax_w4a4(&s.0.join("model"), &s.0.join("model/nested"),
            Path::new("python"), MiniMaxComponent::Fl2va, "cuda").is_err());
    }
}
