//! Optional external LoRA/QLoRA SFT runner for an installed LLaMA-Factory CLI.
//!
//! This is an orchestrator, NOT a trainer. It never generates arbitrary
//! training datasets, alters existing model weights, installs packages or
//! initiates work during preflight. Training starts only via an explicit
//! caller-approved execution, and model/data license decisions remain user's.
//! Knowledge distillation is separate and is NOT implemented here.

use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, s.into())
}
fn string<'a>(yaml: &'a serde_yaml::Value, name: &str) -> io::Result<&'a str> {
    yaml.get(name).and_then(serde_yaml::Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| invalid(format!("{name} must be a nonempty YAML string")))
}
#[derive(Debug)]
pub struct TrainPlan {
    pub config: PathBuf,
    pub trainer: PathBuf,
    pub model: PathBuf,
    pub data_dir: PathBuf,
    pub dataset: String,
    pub output_dir: PathBuf,
    pub quantization_bit: Option<u64>,
}
impl TrainPlan {
    pub fn json(&self) -> Value {
        json!({
            "schema": "ojon.train_plan.v1",
            "backend": "llamafactory-cli",
            "operation": if self.quantization_bit.is_some() {"qlora_sft"} else {"lora_sft"},
            "config": self.config.to_string_lossy(),
            "trainer": self.trainer.to_string_lossy(),
            "model_directory": self.model.to_string_lossy(),
            "data_directory": self.data_dir.to_string_lossy(),
            "dataset": self.dataset,
            "output_directory": self.output_dir.to_string_lossy(),
            "quantization_bit": self.quantization_bit,
            "peak_ram": "unknown",
            "peak_vram": "unknown",
            "network_access": "controlled by external training configuration; review dataset/model references",
            "status": "plan_only",
            "model_supported": "not_verified",
            "training_data_licensed": "user_must_verify",
            "quality_benchmark": "not_run",
            "teacher_student_distillation": "not_implemented",
        })
    }
}

/// Prepare a *local* LoRA/QLoRA supervised fine-tune; the referenced checkpoint
/// and dataset directory must exist locally to avoid implicit model downloads.
/// The actual trainer validates architecture/dataset schema when started.
pub fn prepare_train(config: &Path, trainer: &Path) -> io::Result<TrainPlan> {
    let config = config.canonicalize()?;
    if !config.is_file() { return Err(invalid("training config must be a YAML file")); }
    if !matches!(config.extension().and_then(|v|v.to_str()),Some("yaml"|"yml")) {
        return Err(invalid("training config must be .yaml or .yml"));
    }
    let len = fs::metadata(&config)?.len();
    if len == 0 || len > 1024 * 1024 {
        return Err(invalid("training config must be nonempty and at most 1 MiB"));
    }
    let mut text = String::new();
    File::open(&config)?.read_to_string(&mut text)?;
    let doc: serde_yaml::Value = serde_yaml::from_str(&text)
        .map_err(|e| invalid(format!("invalid training YAML: {e}")))?;
    if !doc.is_mapping() { return Err(invalid("training YAML must contain mapping keys")); }
    let stage = string(&doc,"stage")?;
    if stage != "sft" { return Err(invalid("only supervised fine-tuning stage: sft is enabled")); }
    let method = string(&doc,"finetuning_type")?;
    if method != "lora" { return Err(invalid("only LoRA/QLoRA is enabled; full training is not recommended")); }
    if doc.get("trust_remote_code").and_then(serde_yaml::Value::as_bool).unwrap_or(false) {
        return Err(invalid("trust_remote_code=true is not permitted in Ojon's guarded training route"));
    }
    let model = Path::new(string(&doc,"model_name_or_path")?);
    if !model.is_absolute() || !model.is_dir() {
        return Err(invalid("model_name_or_path must be a local absolute checkpoint directory"));
    }
    if !model.join("config.json").is_file() {
        return Err(invalid("model directory lacks config.json"));
    }
    let model = model.canonicalize()?;
    let data_dir = Path::new(string(&doc,"dataset_dir")?);
    if !data_dir.is_absolute() || !data_dir.is_dir() {
        return Err(invalid("dataset_dir must be a local absolute directory"));
    }
    let data_dir = data_dir.canonicalize()?;
    let dataset = string(&doc,"dataset")?.to_string();
    if !data_dir.join("dataset_info.json").is_file() {
        return Err(invalid("dataset_dir must contain dataset_info.json for LLaMA-Factory"));
    }
    let output_dir = Path::new(string(&doc,"output_dir")?);
    if !output_dir.is_absolute() { return Err(invalid("output_dir must be an absolute directory path")); }
    if output_dir.starts_with(&model) || output_dir.starts_with(&data_dir) {
        return Err(invalid("output must not be inside source model or dataset directory"));
    }
    if output_dir.symlink_metadata().is_ok() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists,
            "output already exists: choose a new directory; Ojon does not overwrite training results"));
    }
    let parent = output_dir.parent().ok_or_else(|| invalid("output directory needs parent"))?;
    if !parent.is_dir() { return Err(invalid("output parent directory must already exist")); }
    let quantization_bit = match doc.get("quantization_bit") {
        None | Some(serde_yaml::Value::Null) => None,
        Some(value) => match value.as_u64() {
            Some(4) => Some(4),
            Some(8) => Some(8),
            _ => return Err(invalid("QLoRA quantization_bit must be 4 or 8")),
        },
    };
    // Plan does not execute, install or probe the trainer binary. A relative
    // executable name is accepted and resolved by the OS when the job starts.
    if trainer.as_os_str().is_empty() { return Err(invalid("trainer executable missing")); }
    Ok(TrainPlan { config, trainer: trainer.to_path_buf(), model, data_dir, dataset, output_dir: output_dir.to_path_buf(), quantization_bit })
}

/// Run only after the caller deliberately approves the job. Runs an installed
/// trusted binary without invoking a shell. Training may need hours and large
/// RAM/VRAM; call on a worker thread in GUI integrations.
pub fn run_train(plan: &TrainPlan) -> io::Result<()> {
    if plan.output_dir.symlink_metadata().is_ok() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists,"training output already exists"));
    }
    eprintln!("Ojon: launching external LLaMA-Factory SFT. Resources/quality not yet predicted.");
    let status = Command::new(&plan.trainer)
        .arg("train")
        .arg(&plan.config)
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;
    if !status.success() {
        return Err(io::Error::other(format!("external LLaMA-Factory trainer failed: {status}")));
    }
    if !plan.output_dir.is_dir() {
        return Err(io::Error::other("trainer exited successfully but expected output directory was not created"));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reject_missing_yaml() {
        let result = prepare_train(Path::new("nonexistent-ojon-training-configuration.yaml"),
            Path::new("llamafactory-cli"));
        assert!(result.is_err());
    }
}
