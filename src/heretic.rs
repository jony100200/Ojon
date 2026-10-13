//! Local-only Heretic 2.x adapter. Heretic itself (AGPL-3.0-or-later) must be
//! installed separately. No Heretic source code is bundled with Ojon.
//!
//! This edits alignment behaviour; it does not quantize, verify intelligence,
//! or ensure an edited model is runnable on the requested GPU.
use crate::hf::inspect_hf_folder;
use crate::jobs::run_with_log;
use crate::resource::{assess_resources, ResourceMode};
use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::{self, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message.into())
}

#[derive(Clone, Debug)]
pub struct HereticPrompts {
    pub good_train: PathBuf,
    pub bad_train: PathBuf,
    pub good_eval: PathBuf,
    pub bad_eval: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HereticExport { Merge, Adapter }
impl HereticExport {
    pub fn parse(s: &str) -> io::Result<Self> {
        match s {"merge"=>Ok(Self::Merge),"adapter"=>Ok(Self::Adapter),
            _=>Err(invalid("Heretic export must be merge or adapter"))}
    }
    fn as_str(self)->&'static str {
        match self {Self::Merge=>"merge",Self::Adapter=>"adapter"}
    }
}

#[derive(Clone,Debug)]
pub struct HereticPlan {
    pub model: PathBuf,
    pub model_weight_bytes:u64,
    pub output:PathBuf,
    pub work:PathBuf,
    pub prompts:HereticPrompts,
    pub executable:PathBuf,
    pub device:String,
    pub quantization:String,
    pub export:HereticExport,
    pub trials:u32,
    pub resume:bool,
}

/// External prompt content is never fetched. A nonempty local UTF-8 line
/// corresponds to one prompt, per Heretic's own text-file loader.
fn checked_prompts(path:&Path)->io::Result<PathBuf>{
    let path=path.canonicalize()?;
    let metadata=fs::metadata(&path)?;
    if !metadata.is_file()||metadata.len()==0||metadata.len()>4*1024*1024 {
        return Err(invalid("local Heretic prompt file must be nonempty and <=4 MiB"));
    }
    let data=fs::read_to_string(&path)?;
    if data.lines().filter(|line|!line.trim().is_empty()).take(4).count()<4 {
        return Err(invalid("each Heretic prompt file needs >=4 nonempty UTF-8 lines"));
    }
    Ok(path)
}
fn new_destination(path:&Path, label:&str)->io::Result<PathBuf>{
    let parent=path.parent().filter(|p|!p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let parent=parent.canonicalize()?;
    if !parent.is_dir(){return Err(invalid(format!("{label} parent is not a directory")));}
    let name=path.file_name().ok_or_else(||invalid("missing destination name"))?;
    let path=parent.join(name);
    if path.symlink_metadata().is_ok(){
        return Err(io::Error::new(ErrorKind::AlreadyExists,format!("{label} already exists")));
    }
    Ok(path)
}

/// Plan using only preexisting checkpoint and text prompt files. No downloads,
/// model loading, output writes or execution are performed here.
pub fn prepare_heretic(
    model:&Path, output:&Path, work:&Path, prompts:&HereticPrompts,
    executable:&Path,device:&str, quantization:&str, export:HereticExport,
    trials:u32, resume:bool,
)->io::Result<HereticPlan>{
    let info=inspect_hf_folder(model)?;
    if !matches!(device,"xpu"|"cuda"|"cpu"){
        return Err(invalid("Heretic device must be xpu, cuda or cpu"));
    }
    if !matches!(quantization,"none"|"bnb_4bit"){
        return Err(invalid("Heretic quantization must be none or bnb_4bit"));
    }
    if !(1..=2000).contains(&trials){return Err(invalid("Heretic trials must be 1..=2000"));}
    if executable.as_os_str().is_empty(){return Err(invalid("Heretic executable required"));}
    let output=new_destination(output,"Heretic model output")?;
    let work=if resume{
        let old=work.canonicalize()?;
        if !old.is_dir()||!old.join("config.toml").is_file(){
            return Err(invalid("resume requires a prior Ojon Heretic work folder"));
        }
        old
    }else{new_destination(work,"Heretic work folder")?};
    let prompts=HereticPrompts{
        good_train:checked_prompts(&prompts.good_train)?,
        bad_train:checked_prompts(&prompts.bad_train)?,
        good_eval:checked_prompts(&prompts.good_eval)?,
        bad_eval:checked_prompts(&prompts.bad_eval)?,
    };
    // Keep the training/calibration and held-out evaluation sets independent.
    if prompts.good_train==prompts.good_eval||prompts.bad_train==prompts.bad_eval{
        return Err(invalid("training and evaluation prompt files must differ"));
    }
    if output.starts_with(&info.directory)||work.starts_with(&info.directory)
        ||output.starts_with(&work)||work.starts_with(&output){
        return Err(invalid("model source, work folder and output must be separate"));
    }
    let plan=HereticPlan{
        model:info.directory,model_weight_bytes:info.source_weight_bytes,
        output,work,prompts,executable:executable.to_path_buf(),device:device.into(),
        quantization:quantization.into(),export,trials,resume,
    };
    if resume{
        let existing=fs::read_to_string(plan.work.join("config.toml"))?;
        if existing!=plan.config_toml(){
            return Err(invalid("resume configuration does not match original Ojon Heretic inputs"));
        }
    }
    Ok(plan)
}

fn toml_str(path:&Path)->String{
    // JSON string escaping is compatible with TOML quoted basic strings.
    serde_json::to_string(&path.to_string_lossy().to_string()).unwrap_or_default()
}
impl HereticPlan{
    pub fn config_toml(&self)->String{
        let device_map=match self.device.as_str(){
            "xpu"=>"xpu:0","cuda"=>"cuda:0",_=>"cpu",
        };
        let initial_trials=self.trials.min(30);
        format!(concat!(
            "# Generated by Ojon; offline local-only Heretic 2.x configuration.\n",
            "model = {model}\n",
            "device_map = \"{device_map}\"\n",
            "quantization = \"{quantization}\"\n",
            "n_trials = {trials}\n",
            "n_startup_trials = {initial_trials}\n",
            "trial_index = 0\n",
            "model_action = \"save\"\n",
            "export_strategy = \"{export}\"\n",
            "save_directory = {output}\n",
            "checkpoint_action = \"{checkpoint}\"\n",
            "study_checkpoint_dir = \"checkpoints\"\n",
            "offload_outputs_to_cpu = true\n",
            "scorers = [\n",
            "  {{ plugin = \"heretic.scorers.keyword_rate.KeywordRate\", optimization = \"minimize\" }},\n",
            "  {{ plugin = \"heretic.scorers.kl_divergence.KLDivergence\", optimization = \"minimize\" }},\n",
            "]\n",
            "modifiers = [{{ plugin = \"heretic.modifiers.ara.ARA\" }}]\n",
            "\n[batch_size_test_prompts]\ndataset = {good_train}\n",
            "\n[[response_prefix_test_prompts]]\ndataset = {good_eval}\n",
            "\n[[response_prefix_test_prompts]]\ndataset = {bad_eval}\n",
            "\n[scorer.KeywordRate.prompts]\ndataset = {bad_eval}\n",
            "\n[scorer.KLDivergence.prompts]\ndataset = {good_eval}\n",
            "\n[modifier.ARA.good_prompts]\ndataset = {good_train}\n",
            "\n[modifier.ARA.bad_prompts]\ndataset = {bad_train}\n"
        ),
        model=toml_str(&self.model),device_map=device_map,
        quantization=self.quantization,trials=self.trials,initial_trials=initial_trials,
        export=self.export.as_str(),output=toml_str(&self.output),
        checkpoint=if self.resume {"continue"}else{"restart"},
        good_train=toml_str(&self.prompts.good_train),bad_train=toml_str(&self.prompts.bad_train),
        good_eval=toml_str(&self.prompts.good_eval),bad_eval=toml_str(&self.prompts.bad_eval))
    }
    pub fn as_json(&self)->Value{
        json!({
            "schema":"ojon.heretic_plan.v1",
            "backend":"Heretic 2.x, separately installed, AGPL-3.0-or-later",
            "source_local_hf":self.model.to_string_lossy(),
            "source_weight_bytes":self.model_weight_bytes,
            "output_new_directory":self.output.to_string_lossy(),
            "work_directory":self.work.to_string_lossy(),
            "prompt_files":{
                "good_train":self.prompts.good_train.to_string_lossy(),
                "bad_train":self.prompts.bad_train.to_string_lossy(),
                "good_eval":self.prompts.good_eval.to_string_lossy(),
                "bad_eval":self.prompts.bad_eval.to_string_lossy(),
            },
            "device":self.device,"quantization":self.quantization,
            "export":self.export.as_str(),"trials":self.trials,"resume":self.resume,
            "model_action":"save_only_no_upload",
            "offline_environment":true,
            "tool":self.executable.to_string_lossy(),
            "status":"PLAN_ONLY",
            "runtime_load_verified":false,
            "quality_preservation_verified":false,
            "guaranteed_network_sandbox":false,
            "next_step":if self.export==HereticExport::Merge{
                "Feed saved HF checkpoint to Ojon quantize-hf, then run held-out tests."
            }else{
                "LoRA adapter needs a compatible local runtime or verified merge before GGUF quantization."
            }
        })
    }
}

fn inspect_heretic_export(plan:&HereticPlan)->io::Result<(String,u64)>{
    if !plan.output.is_dir(){return Err(invalid("Heretic returned but output directory is absent"));}
    match plan.export{
        HereticExport::Merge=>{
            let info=inspect_hf_folder(&plan.output)
                .map_err(|e|invalid(format!("Heretic output is not a complete HF weight directory: {e}")))?;
            if !plan.output.join("tokenizer_config.json").is_file()
                && !plan.output.join("tokenizer.json").is_file(){
                return Err(invalid("merged Heretic model lacks tokenizer information"));
            }
            Ok(("merged_hf_checkpoint_catalog_verified".into(),info.source_weight_bytes))
        }
        HereticExport::Adapter=>{
            if !plan.output.join("adapter_config.json").is_file(){
                return Err(invalid("Heretic adapter missing adapter_config.json"));
            }
            let mut bytes=0u64;let mut found=false;
            for entry in fs::read_dir(&plan.output)?{
                let entry=entry?;let path=entry.path();
                let name=entry.file_name().to_string_lossy().to_string();
                if name.starts_with("adapter_model.")&&(name.ends_with(".safetensors")||name.ends_with(".bin")){
                    bytes=bytes.checked_add(fs::metadata(path)?.len()).ok_or_else(||invalid("adapter bytes overflow"))?;
                    found=true;
                }
            }
            if !found||bytes==0{return Err(invalid("Heretic adapter weights missing"));}
            Ok(("lora_adapter_catalog_verified_not_merged".into(),bytes))
        }
    }
}

/// Actually run installed Heretic, never through a shell. Offline flags avoid
/// Hugging Face network calls; these flags are NOT an OS network sandbox.
pub fn run_heretic(plan:&HereticPlan,allow_resource_risk:bool,mut log:impl FnMut(&str))->io::Result<Value>{
    if plan.output.symlink_metadata().is_ok(){
        return Err(io::Error::new(ErrorKind::AlreadyExists,"Heretic output already exists"));
    }
    let parent=plan.output.parent().ok_or_else(||invalid("missing output parent"))?;
    let preflight=assess_resources(plan.model_weight_bytes,parent,ResourceMode::Heretic,&plan.device)?;
    log(&format!("Heretic resource preflight: {}",preflight.as_json()));
    preflight.enforce(allow_resource_risk)?;
    if !plan.resume {
        fs::create_dir(&plan.work)?;
        let mut config=OpenOptions::new().write(true).create_new(true).open(plan.work.join("config.toml"))?;
        config.write_all(plan.config_toml().as_bytes())?;
        config.sync_all()?;
    }
    let mut cmd=Command::new(&plan.executable);
    cmd.current_dir(&plan.work).arg("--model").arg(&plan.model);
    cmd.env("HF_HUB_OFFLINE","1")
        .env("HF_DATASETS_OFFLINE","1")
        .env("TRANSFORMERS_OFFLINE","1")
        .env("HF_HUB_DISABLE_TELEMETRY","1")
        .env("WANDB_MODE","offline")
        .env("WANDB_DISABLED","true")
        .env("DO_NOT_TRACK","1");
    // Avoid host HERETIC_* configuration overriding the locally generated TOML.
    for (name,_) in std::env::vars_os(){
        if name.to_string_lossy().to_ascii_uppercase().starts_with("HERETIC_"){
            cmd.env_remove(name);
        }
    }
    cmd.env_remove("HF_TOKEN").env_remove("HUGGING_FACE_HUB_TOKEN");
    log("Ojon: launching installed Heretic using local HF model and four local prompt files.");
    run_with_log(&mut cmd,|line|log(line))?;
    // Heretic catches some export errors internally; only a verified output
    // directory lets Ojon report success.
    let (validation,output_bytes)=inspect_heretic_export(plan)?;
    let receipt=json!({
        "schema":"ojon.heretic_result.v1","status":validation,
        "model":plan.model.to_string_lossy(),
        "output":plan.output.to_string_lossy(),"output_bytes":output_bytes,
        "work":plan.work.to_string_lossy(),"export":plan.export.as_str(),
        "device_requested":plan.device,"quantization_requested":plan.quantization,
        "offline_environment_requested":true,
        "internet_blocked_by_os":false,
        "output_catalog_verified":true,
        "runtime_load_verified":false,"task_quality_verified":false,
        "model_refusal_rate_verified":false,
        "format":if plan.export==HereticExport::Merge {"HF safetensors/bin"}else{"PEFT LoRA adapter"},
        "warning":"Export validation is not proof of behavioral results, quality parity, inference compatibility or B60 residency."
    });
    let receipt_file=plan.work.join("ojon-heretic-result.json");
    let mut file=OpenOptions::new().write(true).create_new(true).open(&receipt_file)?;
    file.write_all(&serde_json::to_vec_pretty(&receipt).map_err(io::Error::other)?)?;
    file.sync_all()?;
    Ok(receipt)
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn export_and_source_validation(){
        assert_eq!(HereticExport::parse("merge").unwrap(),HereticExport::Merge);
        assert!(HereticExport::parse("upload").is_err());
        let good=PathBuf::from("C:\\tmp\\model");
        assert_eq!(toml_str(&good),r#""C:\\tmp\\model""#);
    }
    #[test]
    fn config_disables_upload_and_uses_local_inputs(){
        let plan=HereticPlan{model:PathBuf::from("/tmp/model"),model_weight_bytes:1,
            output:PathBuf::from("/tmp/out"),work:PathBuf::from("/tmp/work"),
            prompts:HereticPrompts{
                good_train:"/tmp/train-good.txt".into(),bad_train:"/tmp/train-bad.txt".into(),
                good_eval:"/tmp/eval-good.txt".into(),bad_eval:"/tmp/eval-bad.txt".into()},
            executable:"heretic".into(),device:"xpu".into(),quantization:"none".into(),
            export:HereticExport::Merge,trials:30,resume:false};
        let config=plan.config_toml();
        assert!(config.contains("model_action = \"save\""));
        assert!(config.contains("export_strategy = \"merge\""));
        assert!(config.contains("device_map = \"xpu:0\""));
        assert!(config.contains("dataset = \"/tmp/eval-bad.txt\""));
        assert!(!config.contains("mlabonne/"));
    }
}
