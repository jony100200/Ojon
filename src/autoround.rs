//! Intel AutoRound (Adaptive / Mixed-Bit Quantization) Adapter.
//!
//! Orchestrates external AutoRound optimization for Intel XPU / CUDA / CPU.
//! Produces high-fidelity adaptive low-bit GGUF models.
//!
//! Design Constraints:
//! - Operates strictly on external paths specified by the caller.
//! - Read-only plan inspection; execution requires explicit caller confirmation.
//! - Checks local model files and Python executable without side effects.
//! - Does not overwrite existing output directories.

use crate::jobs::run_with_log;
use serde_json::{json, Value};
use std::fs;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::process::Command;

fn bad_input(msg: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, msg.into())
}

/// AutoRound quantization device target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoRoundDevice {
    /// Intel Arc / Data Center GPU via Intel XPU OneAPI / SYCL
    Xpu,
    /// NVIDIA GPU via CUDA
    Cuda,
    /// Host CPU fallback
    Cpu,
}

impl AutoRoundDevice {
    pub fn parse(s: &str) -> io::Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "xpu" | "intel" => Ok(AutoRoundDevice::Xpu),
            "cuda" | "nvidia" => Ok(AutoRoundDevice::Cuda),
            "cpu" => Ok(AutoRoundDevice::Cpu),
            _ => Err(bad_input(format!("unsupported device: {s}; must be 'xpu', 'cuda', or 'cpu'"))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            AutoRoundDevice::Xpu => "xpu",
            AutoRoundDevice::Cuda => "cuda",
            AutoRoundDevice::Cpu => "cpu",
        }
    }
}

/// Plan for running external Intel AutoRound quantization.
#[derive(Clone, Debug)]
pub struct AutoRoundPlan {
    pub model_dir: PathBuf,
    pub output_dir: PathBuf,
    pub bits: u32,
    pub device: AutoRoundDevice,
    pub format: String,
    pub python: PathBuf,
    pub nsamples: u32,
    pub seqlen: u32,
    pub arguments: Vec<String>,
}

impl AutoRoundPlan {
    pub fn json(&self) -> Value {
        json!({
            "schema": "ojon.autoround_plan.v1",
            "backend": "intel_autoround",
            "operation": "adaptive_quantization",
            "model_directory": self.model_dir.to_string_lossy(),
            "output_directory": self.output_dir.to_string_lossy(),
            "bits": self.bits,
            "device": self.device.as_str(),
            "format": self.format,
            "python": self.python.to_string_lossy(),
            "nsamples": self.nsamples,
            "seqlen": self.seqlen,
            "arguments": self.arguments,
            "status": "PLAN_ONLY",
            "requires_explicit_confirm": true,
            "runtime_hardware": match self.device {
                AutoRoundDevice::Xpu => "Intel Arc / XPU required",
                AutoRoundDevice::Cuda => "NVIDIA CUDA required",
                AutoRoundDevice::Cpu => "Host CPU",
            }
        })
    }
}

/// Inspect and validate inputs to prepare an AutoRound execution plan.
pub fn prepare_autoround_plan(
    model_dir: &Path,
    output_dir: &Path,
    bits: u32,
    device_str: &str,
    python: &Path,
    nsamples: u32,
    seqlen: u32,
) -> io::Result<AutoRoundPlan> {
    if !model_dir.is_dir() {
        return Err(bad_input(format!("model directory does not exist: {}", model_dir.display())));
    }
    let model_dir = model_dir.canonicalize()?;
    if !model_dir.join("config.json").is_file() {
        return Err(bad_input(format!("model directory lacks config.json: {}", model_dir.display())));
    }

    if bits != 2 && bits != 3 && bits != 4 && bits != 8 {
        return Err(bad_input(format!("bits must be 2, 3, 4, or 8; received {bits}")));
    }

    let device = AutoRoundDevice::parse(device_str)?;

    if !output_dir.is_absolute() {
        return Err(bad_input("output_dir must be an absolute path"));
    }
    if output_dir.exists() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            format!("output directory already exists: {}; Ojon will not overwrite", output_dir.display()),
        ));
    }
    let parent = output_dir.parent().ok_or_else(|| bad_input("output directory needs parent"))?;
    if !parent.is_dir() {
        return Err(bad_input("output parent directory does not exist"));
    }

    let python = if python.is_absolute() {
        if !python.is_file() {
            return Err(bad_input(format!("Python executable does not exist: {}", python.display())));
        }
        python.to_path_buf()
    } else {
        python.to_path_buf()
    };

    let nsamples = if nsamples == 0 { 128 } else { nsamples };
    let seqlen = if seqlen == 0 { 2048 } else { seqlen };

    let arguments = vec![
        "-m".into(),
        "auto_round".into(),
        "--model".into(),
        model_dir.to_string_lossy().to_string(),
        "--bits".into(),
        bits.to_string(),
        "--format".into(),
        "auto_round:gguf".into(),
        "--device".into(),
        device.as_str().into(),
        "--nsamples".into(),
        nsamples.to_string(),
        "--seqlen".into(),
        seqlen.to_string(),
        "--output_dir".into(),
        output_dir.to_string_lossy().to_string(),
    ];

    Ok(AutoRoundPlan {
        model_dir,
        output_dir: output_dir.to_path_buf(),
        bits,
        device,
        format: "auto_round:gguf".into(),
        python,
        nsamples,
        seqlen,
        arguments,
    })
}

/// Execute AutoRound after explicit user confirmation.
pub fn run_autoround(plan: &AutoRoundPlan) -> io::Result<PathBuf> {
    if plan.output_dir.exists() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            format!("output directory already exists: {}", plan.output_dir.display()),
        ));
    }

    eprintln!(
        "Ojon: launching Intel AutoRound adaptive quantization on {} (bits: {})...",
        plan.device.as_str(),
        plan.bits
    );

    let mut cmd = Command::new(&plan.python);
    cmd.args(&plan.arguments);

    let (status, log_path) = run_with_log(&mut cmd, "autoround")?;
    if !status.success() {
        return Err(io::Error::other(format!(
            "AutoRound failed with exit status: {}; see log: {}",
            status,
            log_path.display()
        )));
    }

    if !plan.output_dir.is_dir() {
        return Err(io::Error::other(format!(
            "AutoRound completed but expected output directory was not created: {}",
            plan.output_dir.display()
        )));
    }

    Ok(log_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autoround_device_parse() {
        assert_eq!(AutoRoundDevice::parse("xpu").unwrap(), AutoRoundDevice::Xpu);
        assert_eq!(AutoRoundDevice::parse("intel").unwrap(), AutoRoundDevice::Xpu);
        assert_eq!(AutoRoundDevice::parse("cuda").unwrap(), AutoRoundDevice::Cuda);
        assert_eq!(AutoRoundDevice::parse("cpu").unwrap(), AutoRoundDevice::Cpu);
        assert!(AutoRoundDevice::parse("tpu").is_err());
    }

    #[test]
    fn test_autoround_plan_rejects_missing_model() {
        let res = prepare_autoround_plan(
            Path::new("nonexistent-model-folder"),
            Path::new("C:\\Outputs\\out"),
            4,
            "xpu",
            Path::new("python"),
            128,
            2048,
        );
        assert!(res.is_err());
    }
}
