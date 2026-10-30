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

use crate::gguf::inspect_gguf;
use crate::hf::inspect_hf_folder;
use crate::jobs::run_with_log;
use crate::resource::{assess_resources, ResourceMode};
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
            _ => Err(bad_input(format!(
                "unsupported device: {s}; must be 'xpu', 'cuda', or 'cpu'"
            ))),
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
    pub iters: u32,
    pub dataset: Option<PathBuf>,
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
            "method": if self.iters == 0 { "optimized_rtn" } else { "calibrated_auto_round" },
            "iterations": self.iters,
            "local_dataset": self.dataset.as_ref().map(|p| p.to_string_lossy().to_string()),
            "network_downloads_blocked_at_execution": true,
            "method_reference": "https://github.com/intel/auto-round/blob/main/docs/step_by_step.md",
            "export_status": "experimental_upstream_gguf",
            "loader_required": "compatible GGUF runtime; not verified by plan",
            "model_compatibility_verified": false,
            "runtime_load_verified": false,
            "task_quality_verified": false,
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
    // Preserve existing API and GUI calls: 0-iteration optimized RTN by default.
    prepare_autoround_recipe_plan(AutoRoundRecipeRequest {
        model_dir,
        output_dir,
        bits,
        device_str,
        python,
        nsamples,
        seqlen,
        requested_format: None,
        iters: 0,
        dataset: None,
    })
}

/// Inputs for [`prepare_autoround_recipe_plan`].
#[derive(Clone, Copy, Debug)]
pub struct AutoRoundRecipeRequest<'a> {
    pub model_dir: &'a Path,
    pub output_dir: &'a Path,
    pub bits: u32,
    pub device_str: &'a str,
    pub python: &'a Path,
    pub nsamples: u32,
    pub seqlen: u32,
    pub requested_format: Option<&'a str>,
    pub iters: u32,
    pub dataset: Option<&'a Path>,
}
/// Explicit paper-backed recipe: local original HF -> experimental GGUF export.
/// GGUF formats come from intel/auto-round docs/step_by_step.md.
/// Calibration iterations require an operator-supplied local JSON dataset.
pub fn prepare_autoround_recipe_plan(
    request: AutoRoundRecipeRequest<'_>,
) -> io::Result<AutoRoundPlan> {
    let AutoRoundRecipeRequest {
        model_dir,
        output_dir,
        bits,
        device_str,
        python,
        nsamples,
        seqlen,
        requested_format,
        iters,
        dataset,
    } = request;
    if !model_dir.is_dir() {
        return Err(bad_input(format!(
            "model directory does not exist: {}",
            model_dir.display()
        )));
    }
    let model_dir = model_dir.canonicalize()?;
    if !model_dir.join("config.json").is_file() {
        return Err(bad_input(format!(
            "model directory lacks config.json: {}",
            model_dir.display()
        )));
    }

    let default_format = match bits {
        2 => "gguf:q2_k_s",
        3 => "gguf:q3_k_m",
        4 => "gguf:q4_k_m",
        5 => "gguf:q5_k_m",
        6 => "gguf:q6_k",
        8 => "gguf:q8_0", // Explicit legacy opt-in only, never a default.
        _ => {
            return Err(bad_input(
                "unsupported AutoRound bit width (2, 3, 4, 5, 6 or 8)",
            ))
        }
    };
    let selected = requested_format
        .unwrap_or(default_format)
        .to_ascii_lowercase();
    let format_bits =
        match selected.as_str() {
            "gguf:q2_k_s" => 2,
            "gguf:q3_k_s" | "gguf:q3_k_m" | "gguf:q3_k_l" => 3,
            "gguf:q4_k_s" | "gguf:q4_k_m" | "gguf:q4_0" | "gguf:q4_1" => 4,
            "gguf:q5_k_s" | "gguf:q5_k_m" | "gguf:q5_0" | "gguf:q5_1" => 5,
            "gguf:q6_k" => 6,
            "gguf:q8_0" => 8,
            _ => return Err(bad_input(
                "unsupported AutoRound GGUF export format; check upstream 'auto_round list format'",
            )),
        };
    if format_bits != bits {
        return Err(bad_input(format!(
            "bits={bits} does not match export format {selected}"
        )));
    }
    if iters > 1000 {
        return Err(bad_input("AutoRound iterations must be in 0..=1000"));
    }
    if iters > 0 && dataset.is_none() {
        return Err(bad_input("calibrated AutoRound needs --dataset <local.json>; remote default datasets are disabled"));
    }
    let dataset = if let Some(dataset) = dataset {
        let resolved = dataset.canonicalize()?;
        if !resolved.is_file() || fs::metadata(&resolved)?.len() == 0 {
            return Err(bad_input(
                "calibration dataset must be a nonempty local file",
            ));
        }
        if !resolved
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|s| s.eq_ignore_ascii_case("json"))
        {
            return Err(bad_input(
                "AutoRound local calibration input must be a .json file",
            ));
        }
        Some(resolved)
    } else {
        None
    };
    let device = AutoRoundDevice::parse(device_str)?;

    if !output_dir.is_absolute() {
        return Err(bad_input("output_dir must be an absolute path"));
    }
    if output_dir.exists() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            format!(
                "output directory already exists: {}; Ojon will not overwrite",
                output_dir.display()
            ),
        ));
    }
    let parent = output_dir
        .parent()
        .ok_or_else(|| bad_input("output directory needs parent"))?;
    if !parent.is_dir() {
        return Err(bad_input("output parent directory does not exist"));
    }
    if parent.canonicalize()?.starts_with(&model_dir) {
        return Err(bad_input(
            "output must be outside the original model directory",
        ));
    }

    let python = if python.is_absolute() {
        if !python.is_file() {
            return Err(bad_input(format!(
                "Python executable does not exist: {}",
                python.display()
            )));
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
        selected.clone(),
        "--iters".into(),
        iters.to_string(),
        "--device".into(),
        device.as_str().into(),
        "--nsamples".into(),
        nsamples.to_string(),
        "--seqlen".into(),
        seqlen.to_string(),
        "--output_dir".into(),
        output_dir.to_string_lossy().to_string(),
    ];
    let mut arguments = arguments;
    if let Some(path) = &dataset {
        arguments.push("--dataset".into());
        arguments.push(path.to_string_lossy().to_string());
    }

    Ok(AutoRoundPlan {
        model_dir,
        output_dir: output_dir.to_path_buf(),
        bits,
        device,
        format: selected,
        iters,
        dataset,
        python,
        nsamples,
        seqlen,
        arguments,
    })
}

/// Execute AutoRound after explicit user confirmation.
pub fn run_autoround(plan: &AutoRoundPlan) -> io::Result<()> {
    run_autoround_with_policy(plan, false)
}

/// Like run_autoround but permits explicitly accepted resource uncertainty.
pub fn run_autoround_with_policy(
    plan: &AutoRoundPlan,
    allow_resource_risk: bool,
) -> io::Result<()> {
    let source = inspect_hf_folder(&plan.model_dir)?;
    let parent = plan
        .output_dir
        .parent()
        .ok_or_else(|| bad_input("output has no parent"))?;
    let preflight = assess_resources(
        source.source_weight_bytes,
        parent,
        ResourceMode::AutoRound,
        plan.device.as_str(),
    )?;
    eprintln!("Ojon resource preflight: {}", preflight.as_json());
    preflight.enforce(allow_resource_risk)?;
    if plan.output_dir.exists() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            format!(
                "output directory already exists: {}",
                plan.output_dir.display()
            ),
        ));
    }

    eprintln!(
        "Ojon: launching Intel AutoRound adaptive quantization on {} (bits: {})...",
        plan.device.as_str(),
        plan.bits
    );

    let mut cmd = Command::new(&plan.python);
    cmd.args(&plan.arguments);
    // Do not let upstream silently download default datasets/models.
    cmd.env("HF_HUB_OFFLINE", "1")
        .env("HF_DATASETS_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1");

    run_with_log(&mut cmd, |line| println!("{line}"))?;

    if !plan.output_dir.is_dir() {
        return Err(io::Error::other(format!(
            "AutoRound completed but expected output directory was not created: {}",
            plan.output_dir.display()
        )));
    }

    // Directory creation alone is not proof that GGUF export succeeded.
    // Cap metadata traversal; do not load tensor payloads.
    let mut discovered = 0usize;
    let mut inspected = 0usize;
    let mut directories = vec![(plan.output_dir.clone(), 0usize)];
    while let Some((directory, depth)) = directories.pop() {
        for entry in fs::read_dir(directory)? {
            discovered += 1;
            if discovered > 4096 {
                return Err(io::Error::other(
                    "AutoRound output contains too many entries to validate safely",
                ));
            }
            let entry = entry?;
            let filetype = entry.file_type()?;
            if filetype.is_symlink() {
                continue;
            }
            if filetype.is_dir() && depth < 2 {
                directories.push((entry.path(), depth + 1));
            } else if filetype.is_file()
                && entry
                    .path()
                    .extension()
                    .and_then(|s| s.to_str())
                    .is_some_and(|s| s.eq_ignore_ascii_case("gguf"))
            {
                inspect_gguf(&entry.path())?;
                inspected += 1;
            }
        }
    }
    if inspected == 0 {
        return Err(io::Error::other("AutoRound exited without a valid GGUF output; result must not be presented as converted"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autoround_device_parse() {
        assert_eq!(AutoRoundDevice::parse("xpu").unwrap(), AutoRoundDevice::Xpu);
        assert_eq!(
            AutoRoundDevice::parse("intel").unwrap(),
            AutoRoundDevice::Xpu
        );
        assert_eq!(
            AutoRoundDevice::parse("cuda").unwrap(),
            AutoRoundDevice::Cuda
        );
        assert_eq!(AutoRoundDevice::parse("cpu").unwrap(), AutoRoundDevice::Cpu);
        assert!(AutoRoundDevice::parse("tpu").is_err());
    }

    #[test]
    fn test_research_gguf_formats_and_local_calibration() {
        use std::time::{SystemTime, UNIX_EPOCH};
        let id = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("ojon-ar-{id}-{}", std::process::id()));
        let model = root.join("source");
        fs::create_dir_all(&model).unwrap();
        fs::write(model.join("config.json"), "{}").unwrap();
        let out = root.join("out");
        let rt = prepare_autoround_recipe_plan(AutoRoundRecipeRequest {
            model_dir: &model,
            output_dir: &out,
            bits: 4,
            device_str: "cpu",
            python: Path::new("python"),
            nsamples: 128,
            seqlen: 2048,
            requested_format: None,
            iters: 0,
            dataset: None,
        })
        .unwrap();
        assert_eq!(rt.format, "gguf:q4_k_m");
        assert!(rt
            .arguments
            .windows(2)
            .any(|w| w[0] == "--iters" && w[1] == "0"));
        let path = root.join("calibration.json");
        fs::write(&path, "[]").unwrap();
        let calibrated = prepare_autoround_recipe_plan(AutoRoundRecipeRequest {
            model_dir: &model,
            output_dir: &out,
            bits: 5,
            device_str: "xpu",
            python: Path::new("python"),
            nsamples: 128,
            seqlen: 2048,
            requested_format: Some("gguf:q5_k_m"),
            iters: 200,
            dataset: Some(&path),
        })
        .unwrap();
        assert_eq!(calibrated.format, "gguf:q5_k_m");
        assert_eq!(calibrated.json()["method"], "calibrated_auto_round");
        assert!(calibrated
            .arguments
            .windows(2)
            .any(|w| w.first().map(String::as_str) == Some("--dataset")));
        assert!(prepare_autoround_recipe_plan(AutoRoundRecipeRequest {
            model_dir: &model,
            output_dir: &out,
            bits: 4,
            device_str: "cpu",
            python: Path::new("python"),
            nsamples: 128,
            seqlen: 2048,
            requested_format: Some("auto_round:gguf"),
            iters: 0,
            dataset: None,
        },)
        .is_err());
        assert!(prepare_autoround_recipe_plan(AutoRoundRecipeRequest {
            model_dir: &model,
            output_dir: &out,
            bits: 4,
            device_str: "cpu",
            python: Path::new("python"),
            nsamples: 128,
            seqlen: 2048,
            requested_format: Some("gguf:q5_k_m"),
            iters: 0,
            dataset: None,
        },)
        .is_err());
        assert!(prepare_autoround_recipe_plan(AutoRoundRecipeRequest {
            model_dir: &model,
            output_dir: &out,
            bits: 4,
            device_str: "cpu",
            python: Path::new("python"),
            nsamples: 128,
            seqlen: 2048,
            requested_format: None,
            iters: 200,
            dataset: None,
        },)
        .is_err());
        let _ = fs::remove_dir_all(root);
    }
    #[test]
    fn test_autoround_plan_rejects_missing_model() {
        let res = prepare_autoround_plan(
            Path::new("nonexistent-model-folder"),
            Path::new("outputs/out"),
            4,
            "xpu",
            Path::new("python"),
            128,
            2048,
        );
        assert!(res.is_err());
    }
}
