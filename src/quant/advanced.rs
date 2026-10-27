//! Optional public GSQ/RCO optimizer adapters.
//!
//! These orchestrate the upstream Python programs, not their ML kernels.
//! The owning user supplies a trusted local checkout, Python environment,
//! source checkpoint and pre-existing calibration/database inputs.
//! Plan = read-only. Run = explicit confirmation at the CLI boundary.
//! Offline environment is requested, but is NOT a network sandbox.
//! Never claim usable 16/24GB-GPU support without a real test.

use crate::jobs::run_with_log;
use serde_json::{json, Value};
use serde_yaml::Value as YamlValue;
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::process::Command;

fn invalid(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg.into())
}
fn local_repo(root: &Path, script: &str) -> io::Result<(PathBuf, PathBuf)> {
    let root = root.canonicalize()?;
    if !root.is_dir() { return Err(invalid("backend checkout must be a directory")); }
    let path = root.join(script);
    if !path.is_file() { return Err(invalid(format!("missing upstream entry point: {script}"))); }
    Ok((root, path))
}
fn local_hf_model(input: &Path) -> io::Result<PathBuf> {
    let path = input.canonicalize()?;
    if !path.is_dir() || !path.join("config.json").is_file() {
        return Err(invalid("model must be an existing local Hugging Face checkpoint with config.json"));
    }
    // Require at least one actual local weight file, without touching its payload.
    crate::hf::inspect_hf_folder(&path)?;
    Ok(path)
}
fn new_output_directory(path: &Path) -> io::Result<PathBuf> {
    if !path.is_absolute() { return Err(invalid("output must be an absolute local directory path")); }
    if path.symlink_metadata().is_ok() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, "output exists; will not overwrite"));
    }
    let parent = path.parent().ok_or_else(||invalid("output parent missing"))?.canonicalize()?;
    let name = path.file_name().ok_or_else(||invalid("output name missing"))?;
    Ok(parent.join(name))
}
fn validate_python(python: &Path) -> io::Result<PathBuf> {
    if python.as_os_str().is_empty() { return Err(invalid("Python executable must be specified")); }
    // Plain "python"/"python3" may resolve through the local environment's PATH.
    // Absolute paths must resolve to an existing file, without executing it.
    if python.is_absolute() && !python.is_file() {
        return Err(invalid("Python executable path does not exist"));
    }
    Ok(python.to_path_buf())
}
#[derive(Debug)]
pub struct AdvancedPlan {
    pub backend: &'static str,
    pub stage: &'static str,
    pub root: PathBuf,
    pub python: PathBuf,
    pub script: PathBuf,
    pub arguments: Vec<String>,
    pub expected_output: Option<PathBuf>,
    pub model: Option<PathBuf>,
    pub warnings: Vec<&'static str>,
}
impl AdvancedPlan {
    pub fn json(&self) -> Value {
        json!({
            "schema":"ojon.advanced_plan.v1",
            "backend":self.backend,
            "stage":self.stage,
            "checkout":self.root.to_string_lossy(),
            "python":self.python.to_string_lossy(),
            "script":self.script.to_string_lossy(),
            "argv":self.arguments,
            "expected_output":self.expected_output.as_ref().map(|p|p.to_string_lossy().to_string()),
            "model":self.model.as_ref().map(|p|p.to_string_lossy().to_string()),
            "status":"PLAN_ONLY",
            "runtime_tested":false,
            "model_compatibility_verified":false,
            "quality_evaluated":false,
            "gpu_ram_scratch_requirements":"unknown_until_local_preflight",
            "warnings":self.warnings,
        })
    }
}
/// GSQ quantization: starts the public entrypoint with a limited number of
/// layers by default. Use None only after deliberate all-layers selection.
pub fn plan_gsq(
    checkout:&Path, config:&Path, python:&Path, max_layers:Option<u32>,
) -> io::Result<AdvancedPlan> {
    if max_layers.is_some_and(|n| n==0 || n>512){
        return Err(invalid("GSQ layer smoke-test must be 1..=512"));
    }
    let (root,script)=local_repo(checkout,"main.py")?;
    let config=config.canonicalize()?;
    if !config.is_file() || !matches!(config.extension().and_then(|e|e.to_str()),Some("yaml"|"yml")) {
        return Err(invalid("GSQ configuration must be a local YAML file"));
    }
    let size=fs::metadata(&config)?.len();
    if size==0 || size>1024*1024 {return Err(invalid("GSQ config must be nonempty and at most 1 MiB"));}
    let mut text=String::new();
    File::open(&config)?.read_to_string(&mut text)?;
    let value:YamlValue=serde_yaml::from_str(&text)
        .map_err(|e|invalid(format!("malformed GSQ YAML: {e}")))?;
    let model=value.get("model").and_then(|m|m.get("name")).and_then(YamlValue::as_str)
        .ok_or_else(||invalid("GSQ config must set model.name to a local HF directory"))?;
    let model=local_hf_model(Path::new(model))?;
    let mut arguments=vec![script.to_string_lossy().to_string(),"--config".into(),
        config.to_string_lossy().to_string()];
    if let Some(layers)=max_layers {
        arguments.extend(["--max-layers".to_owned(),layers.to_string()]);
    }
    Ok(AdvancedPlan {
        backend:"gsq",stage:"quantize",root,python:validate_python(python)?,script,
        arguments,expected_output:None,model:Some(model),
        warnings:vec![
            "GSQ GPU/RAM/scratch compatibility is not established on 16GB/24GB local cards",
            "Quantization produces training/checkpoint artifacts; a separately verified model assembly step may be necessary",
            "External YAML and local project files must be trusted; offline environment variables are not network isolation",
        ]
    })
}
pub fn plan_gsq_assemble(
    checkout:&Path, config:&Path, output:&Path, python:&Path,
) -> io::Result<AdvancedPlan> {
    let (root,script)=local_repo(checkout,"save_model.py")?;
    let config=config.canonicalize()?;
    if !config.is_file(){return Err(invalid("existing GSQ config required"));}
    let output=new_output_directory(output)?;
    Ok(AdvancedPlan {
        backend:"gsq",stage:"assemble",root,python:validate_python(python)?,script:script.clone(),
        arguments:vec![
            script.to_string_lossy().to_string(),"--config".into(),config.to_string_lossy().to_string(),
            "--out-dir".into(),output.to_string_lossy().to_string()
        ],
        expected_output:Some(output),model:None,
        warnings:vec![
            "Only assemble a completed, locally verified GSQ run",
            "The resulting format and target inference loader still require compatibility and quality tests",
        ]
    })
}
/// Optional CUDA-only RCO database preparation for a local dense Qwen3
/// model using the public quantizer arguments. This runs under
/// torch.distributed.run with one local worker; it still needs CUDA/RAM
/// preflight on the user's machine. The 2/3/4-bit database is input to RCO
/// mixed-precision search, not a runnable model.
pub fn plan_rco_database(
    checkout:&Path,model:&Path,output:&Path,python:&Path,calibration_tokens:u32
) -> io::Result<AdvancedPlan> {
    if !(2048..=524288).contains(&calibration_tokens) {
        return Err(invalid("RCO calibration token count must be 2048..=524288"));
    }
    let (root,script)=local_repo(checkout,"run_quantize.py")?;
    let model=local_hf_model(model)?;
    let config=fs::read(model.join("config.json"))?;
    if config.len()>1024*1024 {return Err(invalid("model config too large"));}
    let cfg:Value=serde_json::from_slice(&config)
        .map_err(|e| invalid(format!("bad model config: {e}")))?;
    if cfg.get("model_type").and_then(Value::as_str)!=Some("qwen3") {
        return Err(invalid("first RCO database adapter is limited to original dense Qwen3 model_type=qwen3"));
    }
    let output=new_output_directory(output)?;
    if output.starts_with(&model) || output.starts_with(&root) {
        return Err(invalid("database output cannot be inside source model or executable checkout"));
    }
    let arguments=vec![
        "-m".into(),"torch.distributed.run".into(),
        "--standalone".into(),"--nproc-per-node=1".into(),
        script.to_string_lossy().to_string(),
        "--model_name_or_path".into(),model.to_string_lossy().to_string(),
        "--quantizable_modules".into(),".*(q|k|v|o|gate|up|down)_proj$".into(),
        "--pre_block_modules".into(),"model.embed_tokens".into(),
        "--block_modules".into(),"model.layers".into(),
        "--post_block_modules".into(),"model.norm".into(),"lm_head".into(),
        "--calibration_data".into(),"fineweb_edu".into(),
        "--calibration_tokens".into(),calibration_tokens.to_string(),
        "--calibration_sequence_length".into(),"512".into(),
        "--bitwidth_options".into(),"2".into(),"3".into(),"4".into(),
        "--calibration_bitwidth".into(),"4".into(),
        "--group_size".into(),"128".into(),
        "--cpu_offload_modules".into(),"--cpu_offload_activations".into(),
        "--low_cpu_mem_usage".into(),
        "--save_dir".into(),output.to_string_lossy().to_string(),
    ];
    Ok(AdvancedPlan{
        backend:"rco",stage:"database",root,python:validate_python(python)?,script,
        arguments,expected_output:Some(output),model:Some(model),
        warnings:vec![
            "Requires working CUDA PyTorch; Intel Arc XPU is NOT supported for this stage",
            "Qwen3-only entrypoint untested on 16GB consumer GPUs; offload may still require more RAM/VRAM",
            "fineweb_edu calibration must already be cached locally in offline mode",
            "Upstream writes multiple per-bit tensors into nested folders; budget substantial scratch space",
            "Local checkout and input data must be trusted; offline environment variables are not a network sandbox",
        ]
    })
}

/// RCO performs precision search **after** upstream has built a per-layer,
/// multi-bit GPTQ database. This is not a complete checkpoint converter.
pub fn plan_rco_search(
    checkout:&Path,model:&Path,layer_database:&Path,
    target_bits:f64,python:&Path,steps:u32,
) -> io::Result<AdvancedPlan> {
    if !target_bits.is_finite() || !(2.0..=8.0).contains(&target_bits){
        return Err(invalid("target average bits must be a number from 2 to 8"));
    }
    if !(1..=3000).contains(&steps){return Err(invalid("RCO steps must be 1..=3000"));}
    let (root,script)=local_repo(checkout,"rco_search_quant.py")?;
    let model=local_hf_model(model)?;
    let database=layer_database.canonicalize()?;
    if !database.is_dir() || fs::read_dir(&database)?.next().is_none() {
        return Err(invalid("RCO requires an existing, populated per-layer bitwidth database"));
    }
    let arguments=vec![
        script.to_string_lossy().to_string(),
        "--model".into(),model.to_string_lossy().to_string(),
        "--layer-dir".into(),database.to_string_lossy().to_string(),
        "--bitwidths".into(),"2,3,4".into(),
        "--target-avg-bits".into(),target_bits.to_string(),
        "--n-steps".into(),steps.to_string(),
        "--calibration-data".into(),"fineweb_edu".into(),
        "--calibration-samples".into(),"32".into(),
        "--calibration-seq-length".into(),"512".into(),
        "--batch-size".into(),"1".into(),
        "--device-map".into(),"single".into(),
        "--cpu-deltas".into(),
        "--weightstore-cache".into(),"lazy".into(),
        "--save-json".into(),
    ];
    Ok(AdvancedPlan {
        backend:"rco",stage:"search",root,python:validate_python(python)?,script,
        arguments,expected_output:None,model:Some(model),
        warnings:vec![
            "RCO search requires a complete upstream multi-precision GPTQ database; Ojon does not produce that database yet",
            "Calibration dataset fineweb_edu must be cached locally when offline mode is enabled",
            "This bounded-steps profile is exploratory and not a published quality/compatibility result",
            "RCO assignment is NOT an exported runnable model; assemble separately with a matching loader",
        ],
    })
}
pub fn plan_rco_assemble(
    checkout:&Path,model:&Path,layer_database:&Path,
    assignment:&Path,output:&Path,python:&Path,
) -> io::Result<AdvancedPlan> {
    let (root,script)=local_repo(checkout,"run_build_checkpoint.py")?;
    let model=local_hf_model(model)?;
    let database=layer_database.canonicalize()?;
    if !database.is_dir(){return Err(invalid("RCO layer database missing"));}
    let assignment=assignment.canonicalize()?;
    if !assignment.is_file() || !matches!(assignment.extension().and_then(|s|s.to_str()),Some("json"|"txt")) {
        return Err(invalid("RCO needs an existing assignment JSON/TXT from upstream search"));
    }
    let output=new_output_directory(output)?;
    if output.starts_with(&model) || output.starts_with(&database) {
        return Err(invalid("RCO assembled output must be outside the model and database directories"));
    }
    let arguments=vec![
        script.to_string_lossy().to_string(),"quant".into(),
        "--model".into(),model.to_string_lossy().to_string(),
        "--config".into(),assignment.to_string_lossy().to_string(),
        "--layer-dir".into(),database.to_string_lossy().to_string(),
        "--output".into(),output.to_string_lossy().to_string(),
        "--format".into(),"compressed-tensors".into(),
    ];
    Ok(AdvancedPlan {
        backend:"rco",stage:"assemble",root,python:validate_python(python)?,script,
        arguments,expected_output:Some(output),model:Some(model),
        warnings:vec![
            "Assembles a compressed-tensors HF checkpoint, NOT GGUF",
            "Do not assume that any llama.cpp build can directly load this checkpoint",
            "Model architecture, quantized loader support, memory usage and actual quality remain unverified",
        ],
    })
}
pub fn run_advanced<F:FnMut(&str)>(plan:&AdvancedPlan, mut log:F)->io::Result<()>{
    if let Some(output)=&plan.expected_output {
        if output.symlink_metadata().is_ok(){
            return Err(io::Error::new(io::ErrorKind::AlreadyExists, "output already exists"));
        }
    }
    let mut cmd=Command::new(&plan.python);
    cmd.args(&plan.arguments).current_dir(&plan.root)
        .env("HF_HUB_OFFLINE","1")
        .env("HF_DATASETS_OFFLINE","1")
        .env("TRANSFORMERS_OFFLINE","1")
        .env("WANDB_MODE","offline")
        .env("TOKENIZERS_PARALLELISM","false");
    log(&format!("[ojon] running external public {} stage {}",plan.backend,plan.stage));
    run_with_log(&mut cmd, |line|log(line))?;
    if let Some(out)=&plan.expected_output {
        if !out.is_dir() {return Err(io::Error::other("backend exited but output directory is missing"));}
        if plan.stage=="assemble" && !out.join("config.json").is_file(){
            return Err(io::Error::other("output missing config.json; resulting checkpoint not verified"));
        }
    }
    log("[ojon] external command exited; model load and retained task quality still require real evaluation");
    Ok(())
}

/// Checks only declared checkout locations, without executing Python or
/// treating unrelated files named main.py as GSQ.
pub fn discover_advanced() -> Value {
    let entries = [
        ("gsq","OJON_GSQ_DIR","https://github.com/IST-DASLab/GSQ",&["main.py","save_model.py"][..]),
        ("rco","OJON_RCO_DIR","https://github.com/IST-DASLab/RCO",&["run_quantize.py","rco_search_quant.py","run_build_checkpoint.py"][..]),
    ];
    let tools: Vec<Value> = entries.iter().map(|(name,envname,url,files)|{
        let root=std::env::var_os(envname).map(PathBuf::from);
        let detected=root.as_ref().is_some_and(|p|p.is_dir()&&files.iter().all(|f|p.join(f).is_file()));
        json!({
            "backend":name,"environment_variable":envname,
            "checkout":root.as_ref().map(|p|p.to_string_lossy().into_owned()),
            "source_code_present":detected,
            "scripts_required":files,
            "official_repository":url,
            "python_dependencies_installed":"not_checked",
            "hardware_compatible":"not_verified",
            "runtime_load_verified":false,
            "quality_tested":false,
        })
    }).collect();
    json!({"schema":"ojon.optional_optimizers.v1","optimizers":tools})
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn rejects_unusable_precision_budgets(){
        assert!(plan_rco_search(Path::new("."),Path::new("."),Path::new("."),1.2,Path::new("python"),30).is_err());
        assert!(plan_gsq(Path::new("."),Path::new("."),Path::new("python"),Some(0)).is_err());
    }
}
