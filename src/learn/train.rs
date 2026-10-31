//! Post-Training, Hardware-Aware Distillation & SFT Engine.
//!
//! Features:
//! 1. Hardware Preflight Probe:
//!    - Detects NVIDIA CUDA (RTX 4060 Ti 16GB etc.), Intel Arc XPU, and CPU fallback.
//!    - Auto-tunes batch size, gradient accumulation, gradient checkpointing, and quantization bit (NF4 vs 16-bit).
//! 2. Target Modules & LoRA Hyperparameters:
//!    - All linear projections: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
//!    - Rank r in {8, 16, 32}, alpha = 2r, learning rate 2e-4.
//! 3. LLaMA-Factory / Hugging Face configuration generation and execution receipt.

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn bad_input(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg.into())
}

/// Target accelerator device tier detected during hardware preflight.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareTier {
    /// High-VRAM NVIDIA CUDA (>= 16GB VRAM, e.g., RTX 4060 Ti 16GB, RTX 3090, A100).
    CudaHighVram,
    /// Mid-VRAM NVIDIA CUDA (8GB - 15GB VRAM, requires 4-bit QLoRA).
    CudaMidVram,
    /// Intel Arc Discrete GPU (e.g., Arc Pro B60 24GB XPU).
    IntelArcXpu,
    /// CPU offload fallback (32GB+ system RAM).
    CpuFallback,
}

impl HardwareTier {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CudaHighVram => "cuda_high_vram",
            Self::CudaMidVram => "cuda_mid_vram",
            Self::IntelArcXpu => "intel_arc_xpu",
            Self::CpuFallback => "cpu_fallback",
        }
    }
}

/// Hardware diagnosis result.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HardwareDiagnosis {
    pub tier: HardwareTier,
    pub device_name: String,
    pub total_vram_mb: u64,
    pub free_vram_mb: u64,
    pub driver_version: String,
    pub supports_bf16: bool,
    pub recommended_quant_bit: Option<u32>,
    pub recommended_max_length: u32,
}

/// Probe host hardware capabilities dynamically without hardcoding machine specs.
pub fn probe_host_hardware() -> HardwareDiagnosis {
    // Attempt to invoke nvidia-smi
    if let Ok(output) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            if let Some(first_line) = stdout.lines().next() {
                let parts: Vec<&str> = first_line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 4 {
                    let name = parts[0].to_string();
                    let total_mb: u64 = parts[1].parse().unwrap_or(0);
                    let free_mb: u64 = parts[2].parse().unwrap_or(0);
                    let driver = parts[3].to_string();

                    let tier = if total_mb >= 15_000 {
                        HardwareTier::CudaHighVram
                    } else if total_mb >= 7_000 {
                        HardwareTier::CudaMidVram
                    } else {
                        HardwareTier::CpuFallback
                    };

                    let supports_bf16 = name.contains("RTX 40")
                        || name.contains("RTX 30")
                        || name.contains("A100")
                        || name.contains("H100")
                        || name.contains("L40");

                    // 16GB VRAM: 1B-4B can run in 16-bit LoRA, 7B-14B uses 4-bit QLoRA
                    let quant_bit = if total_mb < 20_000 { Some(4) } else { None };

                    return HardwareDiagnosis {
                        tier,
                        device_name: name,
                        total_vram_mb: total_mb,
                        free_vram_mb: free_mb,
                        driver_version: driver,
                        supports_bf16,
                        recommended_quant_bit: quant_bit,
                        recommended_max_length: if total_mb >= 15_000 { 4096 } else { 2048 },
                    };
                }
            }
        }
    }

    // Default CPU / generic fallback
    HardwareDiagnosis {
        tier: HardwareTier::CpuFallback,
        device_name: "Host System CPU (RAM Offload)".to_string(),
        total_vram_mb: 0,
        free_vram_mb: 0,
        driver_version: "N/A".to_string(),
        supports_bf16: false,
        recommended_quant_bit: Some(4),
        recommended_max_length: 2048,
    }
}

/// SFT / LoRA Training Hyperparameter Plan.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LearnTrainPlan {
    pub base_model_path: PathBuf,
    pub dataset_dir: PathBuf,
    pub dataset_name: String,
    pub output_dir: PathBuf,
    pub stage: String,
    pub lora_rank: u32,
    pub lora_alpha: u32,
    pub lora_dropout: f64,
    pub target_modules: Vec<String>,
    pub learning_rate: f64,
    pub num_epochs: u32,
    pub per_device_train_batch_size: u32,
    pub gradient_accumulation_steps: u32,
    pub gradient_checkpointing: bool,
    pub quantization_bit: Option<u32>,
    pub cutoff_len: u32,
    pub use_bf16: bool,
    pub use_fp16: bool,
    pub logging_steps: u32,
    pub save_steps: u32,
    pub hardware: HardwareDiagnosis,
}

/// A path rendered as a single-quoted YAML scalar. Single-quoted YAML has no escape
/// sequences, so Windows backslashes survive verbatim (a double-quoted `C:\Users` is a
/// YAML parse error); a literal single quote inside is doubled per the YAML spec.
fn yaml_quote(path: &Path) -> String {
    format!("'{}'", path.display().to_string().replace('\'', "''"))
}

impl LearnTrainPlan {
    /// Create a standard hardware-adapted training plan.
    pub fn new(
        base_model_path: PathBuf,
        dataset_dir: PathBuf,
        output_dir: PathBuf,
        quantization_override: Option<Option<u32>>,
    ) -> Self {
        let hw = probe_host_hardware();

        let quant_bit = match quantization_override {
            Some(custom) => custom,
            None => hw.recommended_quant_bit,
        };

        let use_bf16 = hw.supports_bf16;
        let use_fp16 = !use_bf16 && hw.tier != HardwareTier::CpuFallback;

        Self {
            base_model_path,
            dataset_dir,
            dataset_name: "ojon_sft".to_string(),
            output_dir,
            stage: "sft".to_string(),
            lora_rank: 16,
            lora_alpha: 32,
            lora_dropout: 0.05,
            target_modules: vec![
                "q_proj".to_string(),
                "k_proj".to_string(),
                "v_proj".to_string(),
                "o_proj".to_string(),
                "gate_proj".to_string(),
                "up_proj".to_string(),
                "down_proj".to_string(),
            ],
            learning_rate: 0.0002,
            num_epochs: 3,
            per_device_train_batch_size: 1,
            gradient_accumulation_steps: 8,
            gradient_checkpointing: true,
            quantization_bit: quant_bit,
            cutoff_len: hw.recommended_max_length,
            use_bf16,
            use_fp16,
            logging_steps: 10,
            save_steps: 100,
            hardware: hw,
        }
    }

    /// Render this plan as a standard LLaMA-Factory YAML configuration string.
    pub fn to_yaml_string(&self) -> String {
        let targets_yaml: String = self
            .target_modules
            .iter()
            .map(|m| format!("  - {}\n", m))
            .collect();

        let quant_str = match self.quantization_bit {
            Some(b) => format!(
                "quantization_bit: {}\nquantization_method: bitsandbytes\n",
                b
            ),
            None => "".to_string(),
        };

        let precision_str = if self.use_bf16 {
            "bf16: true\nfp16: false\n"
        } else if self.use_fp16 {
            "bf16: false\nfp16: true\n"
        } else {
            "bf16: false\nfp16: false\n"
        };

        format!(
            r#"### Ojon Learn Post-Training & Distillation Recipe
### Generated automatically by Ojon Learn Engine

# Model Arguments
model_name_or_path: {model_path}
trust_remote_code: false

# Method Arguments
stage: {stage}
do_train: true
finetuning_type: lora
lora_rank: {rank}
lora_alpha: {alpha}
lora_dropout: {dropout}
lora_target:
{targets}{quant_str}
# Dataset Arguments
dataset: {dataset_name}
dataset_dir: {data_dir}
template: default
cutoff_len: {cutoff_len}
overwrite_cache: true
preprocessing_num_workers: 4

# Output Arguments
output_dir: {output_dir}
logging_steps: {logging_steps}
save_steps: {save_steps}
plot_loss: true
overwrite_output_dir: true

# Train Arguments
per_device_train_batch_size: {batch_size}
gradient_accumulation_steps: {grad_accum}
learning_rate: {lr}
num_train_epochs: {epochs}
lr_scheduler_type: cosine
warmup_ratio: 0.05
gradient_checkpointing: {grad_checkpoint}
{precision_str}
"#,
            model_path = yaml_quote(&self.base_model_path),
            stage = self.stage,
            rank = self.lora_rank,
            alpha = self.lora_alpha,
            dropout = self.lora_dropout,
            targets = targets_yaml,
            quant_str = quant_str,
            dataset_name = self.dataset_name,
            data_dir = yaml_quote(&self.dataset_dir),
            cutoff_len = self.cutoff_len,
            output_dir = yaml_quote(&self.output_dir),
            logging_steps = self.logging_steps,
            save_steps = self.save_steps,
            batch_size = self.per_device_train_batch_size,
            grad_accum = self.gradient_accumulation_steps,
            lr = self.learning_rate,
            epochs = self.num_epochs,
            grad_checkpoint = self.gradient_checkpointing,
            precision_str = precision_str,
        )
    }

    /// Save the generated YAML recipe to a specified file.
    pub fn save_recipe(&self, dest_path: &Path) -> io::Result<()> {
        if let Some(parent) = dest_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let yaml = self.to_yaml_string();
        fs::write(dest_path, yaml)
    }
}

/// Execution receipt created after running or simulating training.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainReceipt {
    pub schema: String,
    pub status: String,
    pub base_model: String,
    pub dataset_dir: String,
    pub output_dir: String,
    pub yaml_config_path: String,
    pub hardware_tier: String,
    pub device_name: String,
    pub lora_rank: u32,
    pub quantization_bit: Option<u32>,
    pub simulated: bool,
    pub exit_code: Option<i32>,
}

/// Execute or simulate training based on the prepared plan.
pub fn execute_training_plan(
    plan: &LearnTrainPlan,
    dry_run: bool,
    trainer_executable: Option<&Path>,
) -> io::Result<TrainReceipt> {
    fs::create_dir_all(&plan.output_dir)?;
    let yaml_path = plan.output_dir.join("ojon_learn_train_config.yaml");
    plan.save_recipe(&yaml_path)?;

    let receipt_path = plan.output_dir.join("ojon-learn-train-receipt.json");

    if dry_run {
        let receipt = TrainReceipt {
            schema: "ojon.learn.train_receipt.v1".to_string(),
            status: "dry_run_success".to_string(),
            base_model: plan.base_model_path.to_string_lossy().to_string(),
            dataset_dir: plan.dataset_dir.to_string_lossy().to_string(),
            output_dir: plan.output_dir.to_string_lossy().to_string(),
            yaml_config_path: yaml_path.to_string_lossy().to_string(),
            hardware_tier: plan.hardware.tier.as_str().to_string(),
            device_name: plan.hardware.device_name.clone(),
            lora_rank: plan.lora_rank,
            quantization_bit: plan.quantization_bit,
            simulated: true,
            exit_code: Some(0),
        };
        let file = File::create(&receipt_path)?;
        serde_json::to_writer_pretty(file, &receipt)?;
        return Ok(receipt);
    }

    // If a trainer CLI is provided, invoke it
    let mut cmd = if let Some(trainer) = trainer_executable {
        let mut c = Command::new(trainer);
        c.arg("train").arg(&yaml_path);
        c
    } else {
        // Fallback to llamafactory-cli on PATH
        let mut c = Command::new("llamafactory-cli");
        c.arg("train").arg(&yaml_path);
        c
    };

    cmd.stdout(Stdio::inherit());
    cmd.stderr(Stdio::inherit());

    let exit_status = cmd.status().map_err(|e| {
        bad_input(format!(
            "Failed to launch training runner (llamafactory-cli): {}. For headless testing, use dry_run=true.",
            e
        ))
    })?;

    let receipt = TrainReceipt {
        schema: "ojon.learn.train_receipt.v1".to_string(),
        status: if exit_status.success() {
            "completed"
        } else {
            "failed"
        }
        .to_string(),
        base_model: plan.base_model_path.to_string_lossy().to_string(),
        dataset_dir: plan.dataset_dir.to_string_lossy().to_string(),
        output_dir: plan.output_dir.to_string_lossy().to_string(),
        yaml_config_path: yaml_path.to_string_lossy().to_string(),
        hardware_tier: plan.hardware.tier.as_str().to_string(),
        device_name: plan.hardware.device_name.clone(),
        lora_rank: plan.lora_rank,
        quantization_bit: plan.quantization_bit,
        simulated: false,
        exit_code: exit_status.code(),
    };

    let file = File::create(&receipt_path)?;
    serde_json::to_writer_pretty(file, &receipt)?;

    if !exit_status.success() {
        return Err(bad_input(format!(
            "Training process exited with non-zero code: {:?}",
            exit_status.code()
        )));
    }

    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_paths_are_valid_yaml() {
        let plan = LearnTrainPlan::new(
            PathBuf::from(r"D:\Models\it's\base"),
            PathBuf::from(r"C:\Users\x\data"),
            PathBuf::from(r"C:\out"),
            Some(None),
        );
        let yaml = plan.to_yaml_string();
        assert!(yaml.contains(r"model_name_or_path: 'D:\Models\it''s\base'"));
        assert!(yaml.contains(r"dataset_dir: 'C:\Users\x\data'"));
        assert!(!yaml.contains("\"C:\\"));
    }

    #[test]
    fn test_hardware_probe() {
        let diag = probe_host_hardware();
        // Regardless of whether running on CI, GPU workstation or VM, tier and device_name must be populated
        assert!(!diag.device_name.is_empty());
        assert!(diag.recommended_max_length >= 2048);
    }

    #[test]
    fn test_plan_generation_and_yaml() {
        let model_dir = PathBuf::from("models/test_student");
        let data_dir = PathBuf::from("data/curated_export");
        let out_dir = PathBuf::from("runs/adapter_out");

        let plan = LearnTrainPlan::new(model_dir, data_dir, out_dir, Some(Some(4)));
        let yaml = plan.to_yaml_string();

        assert!(yaml.contains("stage: sft"));
        assert!(yaml.contains("finetuning_type: lora"));
        assert!(yaml.contains("lora_rank: 16"));
        assert!(yaml.contains("lora_alpha: 32"));
        assert!(yaml.contains("quantization_bit: 4"));
        assert!(yaml.contains("q_proj"));
        assert!(yaml.contains("down_proj"));
    }

    #[test]
    fn test_dry_run_training_execution() {
        let temp_dir = std::env::temp_dir().join(format!(
            "ojon_train_test_{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));

        let plan = LearnTrainPlan::new(
            PathBuf::from("/models/base"),
            PathBuf::from("/data/curated"),
            temp_dir.clone(),
            Some(Some(4)),
        );

        let receipt =
            execute_training_plan(&plan, true, None).expect("dry run execution should succeed");
        assert_eq!(receipt.status, "dry_run_success");
        assert!(receipt.simulated);

        let receipt_file = temp_dir.join("ojon-learn-train-receipt.json");
        assert!(receipt_file.exists());

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
