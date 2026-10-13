//! Local compressed-base + LoRA runtime smoke verification.
//!
//! Preflight is CPU-only metadata inspection; execution requires a separately
//! installed llama-cli, and may run entirely on CPU (-ngl 0) or a selected GPU.
//! Successful inference does NOT demonstrate lower refusals, preserved coding
//! ability, or VLM recognition quality.

use crate::gguf::{inspect_gguf, inspect_gguf_tensors};
use crate::jobs::run_with_log;
use serde_json::{json, Value};
use std::fs::{self, File, OpenOptions};
use std::io::{self, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn invalid(s: impl Into<String>) -> io::Error { io::Error::new(ErrorKind::InvalidInput, s.into()) }
const LIMIT: u64 = 2 * 1024 * 1024;

#[derive(Debug)]
pub struct AdapterSmokePlan {
    bundle: PathBuf,
    base: PathBuf,
    adapter: PathBuf,
    mmproj: Option<PathBuf>,
    tool: PathBuf,
    image: Option<PathBuf>,
    prompt: String,
    report: PathBuf,
    gpu_layers: i32,
    device: Option<String>,
    tokens: u32,
    adapter_tensors: usize,
}

impl AdapterSmokePlan {
    pub fn as_json(&self) -> Value {
        json!({
            "schema":"ojon.adapter_smoke_plan.v1",
            "bundle":self.bundle.to_string_lossy(),
            "model":self.base.to_string_lossy(),
            "adapter":self.adapter.to_string_lossy(),
            "mmproj":self.mmproj.as_ref().map(|p|p.to_string_lossy().to_string()),
            "image":self.image.as_ref().map(|p|p.to_string_lossy().to_string()),
            "llama_cli":self.tool.to_string_lossy(),
            "prompt":self.prompt,
            "gpu_layers":self.gpu_layers,
            "device":self.device,
            "tokens":self.tokens,
            "adapter_tensors":self.adapter_tensors,
            "report":self.report.to_string_lossy(),
            "status":"PREFLIGHT_ONLY",
            "model_loaded":false,
            "refusal_test":"NOT_RUN",
            "vision_quality":"NOT_EVALUATED"
        })
    }
    fn command(&self) -> Command {
        let mut cmd=Command::new(&self.tool);
        cmd.arg("-m").arg(&self.base)
            .arg("--lora").arg(&self.adapter)
            .arg("--prompt").arg(&self.prompt)
            .arg("--gpu-layers").arg(self.gpu_layers.to_string())
            .arg("--temp").arg("0")
            .arg("-s").arg("42")
            .arg("-n").arg(self.tokens.to_string())
            .arg("--no-display-prompt")
            .arg("--simple-io")
            .arg("--single-turn")
            .arg("--no-show-timings")
            .arg("--offline");
        if let Some(dev)=&self.device {cmd.arg("--device").arg(dev);}
        if let Some(mm)=&self.mmproj {cmd.arg("--mmproj").arg(mm);}
        if let Some(image)=&self.image {cmd.arg("--image").arg(image);}
        cmd
    }
}
fn manifest_path(value:&str)->io::Result<PathBuf>{
    if value.trim().is_empty() {return Err(invalid("bundle referenced empty path"));}
    let p=Path::new(value);
    if !p.is_absolute() {return Err(invalid("bundle external model paths must be absolute"));}
    p.canonicalize()
}

/// Verify the bundle's references, GGUF structure, LoRA architecture and tensor
/// naming before a model/GPU allocation. Does not load original tensor payload.
pub fn prepare_adapter_smoke(
    bundle:&Path, report:&Path, tool:&Path,
    gpu_layers:i32, device:Option<&str>, image:Option<&Path>,
    prompt:Option<&str>,tokens:u32,
)->io::Result<AdapterSmokePlan> {
    if !(-1..=512).contains(&gpu_layers) {return Err(invalid("gpu_layers must be -1..512"));}
    if !(1..=1024).contains(&tokens) {return Err(invalid("tokens must be 1..1024"));}
    if tool.as_os_str().is_empty() {return Err(invalid("llama-cli tool path is empty"));}
    if let Some(d)=device {
        if d.is_empty()||d.len()>80||!d.bytes().all(|b|
            b.is_ascii_alphanumeric()||matches!(b,b'_'|b'-'|b'/'|b','|b':')) {
            return Err(invalid("device must be a literal local backend device ID"));
        }
    }
    let text=prompt.unwrap_or("Reply with the single word READY.");
    if text.is_empty()||text.len()>8192 {return Err(invalid("prompt must be 1..8192 characters"));}
    let bundle=bundle.canonicalize()?;
    if !bundle.is_dir(){return Err(invalid("bundle must be a directory"));}
    let manifest=bundle.join("ojon-compressed-bundle.json");
    let len=fs::metadata(&manifest)?.len();
    if len==0||len>LIMIT {return Err(invalid("compressed bundle manifest must be 1B..2MiB"));}
    let raw:Value=serde_json::from_slice(&fs::read(manifest)?)
        .map_err(|e|invalid(format!("bundle manifest JSON is invalid: {e}")))?;
    if raw.get("schema").and_then(Value::as_str)!=Some("ojon.compressed_model_bundle.v1") {
        return Err(invalid("not an Ojon compressed model bundle"));
    }
    let base=manifest_path(raw.get("original_quantized_base")
        .and_then(Value::as_str).ok_or_else(||invalid("manifest lacks base model path"))?)?;
    let base_summary=inspect_gguf(&base)?;
    if base_summary.tensor_count==0 || base_summary.split_count.unwrap_or(1)>1 {
        return Err(invalid("invalid or split base GGUF"));
    }
    if raw.get("base_bytes").and_then(Value::as_u64)!=Some(base_summary.file_bytes) {
        return Err(invalid("base GGUF size changed since adapter creation"));
    }
    let adapter=bundle.join("adapter.gguf").canonicalize()?;
    if !adapter.is_file() {return Err(invalid("adapter.gguf missing"));}
    let summary=inspect_gguf(&adapter)?;
    if summary.tensor_count==0||summary.big_endian||summary.split_count.unwrap_or(1)>1 {
        return Err(invalid("invalid compressed LoRA adapter GGUF"));
    }
    if summary.architecture!=base_summary.architecture {
        return Err(invalid("adapter architecture differs from base architecture"));
    }
    if raw.get("adapter_bytes").and_then(Value::as_u64)!=Some(summary.file_bytes) {
        return Err(invalid("adapter GGUF size differs from bundle manifest"));
    }
    let list=inspect_gguf_tensors(&adapter)?;
    if list.iter().any(|x| !x.name.ends_with(".lora_a") && !x.name.ends_with(".lora_b")) {
        return Err(invalid("adapter GGUF contains unexpected non-LoRA tensor"));
    }
    if list.len()%2!=0||list.iter().any(|x|x.storage_type!=0||x.shape.len()!=2) {
        return Err(invalid("adapter GGUF requires paired F32 2D LoRA matrices"));
    }
    let mmproj=match raw.get("mmproj").and_then(Value::as_str) {
        Some(p) => {
            let p=manifest_path(p)?;
            if inspect_gguf(&p)?.tensor_count==0 {
                return Err(invalid("mmproj GGUF has no tensors"));
            }
            Some(p)
        }
        None=>None
    };
    let image=match image {
        Some(p)=>{
            let p=p.canonicalize()?;
            if !p.is_file() {return Err(invalid("image must be a readable local file"));}
            if mmproj.is_none() {return Err(invalid("cannot test image without a vision mmproj"));}
            if fs::metadata(&p)?.len()==0 {return Err(invalid("test image is empty"));}
            Some(p)
        }
        None=>None
    };
    // Existing report stays untouched, even when another process writes while
    // this plan is running. Run uses create_new for final output.
    if report.extension().and_then(|p|p.to_str())!=Some("json"){
        return Err(invalid("smoke report filename must end in .json"));
    }
    let parent=report.parent().filter(|p|!p.as_os_str().is_empty())
        .unwrap_or(Path::new(".")).canonicalize()?;
    let report=parent.join(report.file_name().ok_or_else(||invalid("report filename missing"))?);
    if report.symlink_metadata().is_ok(){
        return Err(io::Error::new(ErrorKind::AlreadyExists,"smoke report already exists"));
    }
    Ok(AdapterSmokePlan{
        bundle,base,adapter,mmproj,tool:tool.to_path_buf(),image,prompt:text.to_owned(),
        report,gpu_layers,device:device.map(str::to_owned),tokens,adapter_tensors:list.len()
    })
}

#[derive(Debug)]
pub struct AdapterSmokeResult {
    pub report:PathBuf,
    pub runtime_exit_success:bool,
    pub image_supplied:bool,
}
impl AdapterSmokeResult{
    pub fn as_json(&self)->Value {
        json!({"schema":"ojon.adapter_smoke_result.v1",
            "report":self.report.to_string_lossy(),
            "runtime_exit_success":self.runtime_exit_success,
            "image_supplied":self.image_supplied,
            "quality_evaluated":false,
            "refusal_rate":"NOT_EVALUATED",
            "image_understanding":"NOT_EVALUATED"})
    }
}

/// Run one actual local llama-cli prompt. Records successful process exit and
/// actual logged lines but DOES NOT interpret model text as a quality metric.
/// Failure leaves no misleading success receipt.
pub fn run_adapter_smoke(plan:&AdapterSmokePlan,mut log:impl FnMut(&str))->io::Result<AdapterSmokeResult>{
    if plan.report.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists,"smoke report exists"));
    }
    let mut lines=Vec::<String>::new();
    let mut truncated=false;
    let mut received=0usize;
    let start=SystemTime::now().duration_since(UNIX_EPOCH)
        .map_err(|_|invalid("system clock invalid"))?.as_secs();
    let mut cmd=plan.command();
    run_with_log(&mut cmd,|line|{
        log(line);
        received+=1;
        if lines.len()<400 && lines.iter().map(|s|s.len()).sum::<usize>() < 128*1024 {
            lines.push(line.to_owned());
        } else {truncated=true;}
    })?;
    let receipt=json!({
        "schema":"ojon.adapter_smoke_receipt.v1",
        "created_utc_unix":start,
        "bundle":plan.bundle.to_string_lossy(),
        "base":plan.base.to_string_lossy(),
        "adapter":plan.adapter.to_string_lossy(),
        "mmproj":plan.mmproj.as_ref().map(|p|p.to_string_lossy().to_string()),
        "image":plan.image.as_ref().map(|p|p.to_string_lossy().to_string()),
        "gpu_layers_requested":plan.gpu_layers,
        "device_requested":plan.device,
        "tokens_requested":plan.tokens,
        "runtime_command":"llama-cli (model + LoRA + optional mmproj/image)",
        "runtime_exit_success":true,
        "output_lines":received,
        "captured_log_lines":lines,
        "captured_log_truncated":truncated,
        "refusal_rate":"NOT_EVALUATED",
        "coding_quality":"NOT_EVALUATED",
        "vision_qa":"NOT_EVALUATED",
        "status":"INFERENCE_PROCESS_EXITED_SUCCESSFULLY"
    });
    let bytes=serde_json::to_vec_pretty(&receipt)
        .map_err(|e|invalid(format!("cannot serialize receipt: {e}")))?;
    let mut file=OpenOptions::new().write(true).create_new(true).open(&plan.report)?;
    if let Err(err)=file.write_all(&bytes).and_then(|_|file.sync_all()) {
        drop(file);let _=fs::remove_file(&plan.report);return Err(err);
    }
    Ok(AdapterSmokeResult{report:plan.report.clone(),
        runtime_exit_success:true,image_supplied:plan.image.is_some()})
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn only_genuine_local_run_may_claim_success(){
        let result=AdapterSmokeResult{report:PathBuf::from("test.json"),
            runtime_exit_success:true,image_supplied:true};
        assert_eq!(result.as_json()["quality_evaluated"],false);
        assert_eq!(result.as_json()["image_understanding"],"NOT_EVALUATED");
    }
    fn tiny_gguf(path:&Path) {
        let mut f=File::create(path).unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        fn strfield(f:&mut File,txt:&str){
            f.write_all(&(txt.len() as u64).to_le_bytes()).unwrap();
            f.write_all(txt.as_bytes()).unwrap();
        }
        strfield(&mut f,"general.architecture");
        f.write_all(&8u32.to_le_bytes()).unwrap();
        strfield(&mut f,"qwen3");
        strfield(&mut f,"blk.0.attn_output.weight");
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        let pos=f.stream_position().unwrap();
        let padding=(32-pos%32)%32;
        f.write_all(&vec![0u8;padding as usize]).unwrap();
        f.write_all(&[0u8;32]).unwrap();
        f.sync_all().unwrap();
    }
    #[test]
    fn adapter_bundle_preflight_needs_no_model_inference(){
        use crate::compressed_edit::{prepare_compressed_edit,run_compressed_edit};
        let stamp=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root=std::env::temp_dir().join(format!("ojon-adapter-smoke-{}-{stamp}",std::process::id()));
        fs::create_dir(&root).unwrap();
        let base=root.join("quantized.gguf");
        tiny_gguf(&base);
        let recipe=root.join("recipe.json");
        fs::write(&recipe,serde_json::to_vec(&json!({
            "schema":"ojon.abliteration.v1","preserve_row_norms":false,
            "edits":[{
                "tensor":"model.layers.0.self_attn.o_proj.weight",
                "strength":1.0,
                "lora":{"a":[[1.0,0.0]],"b":[[0.0],[1.0]]}
            }]
        })).unwrap()).unwrap();
        let bundle=root.join("adapter-bundle");
        let edit_plan=prepare_compressed_edit(&base,&recipe,&bundle,None).unwrap();
        run_compressed_edit(&edit_plan,|_|{}).unwrap();
        let report=root.join("check.json");
        let smoke=prepare_adapter_smoke(&bundle,&report,Path::new("llama-cli"),
            0,None,None,None,32).unwrap();
        assert_eq!(smoke.as_json()["adapter_tensors"],2);
        assert_eq!(smoke.as_json()["status"],"PREFLIGHT_ONLY");
        assert!(!report.exists());
        assert!(prepare_adapter_smoke(&bundle,&report,Path::new("llama-cli"),
            0,None,Some(&recipe),None,32).is_err(),"image without mmproj must fail");
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn rejects_blank_tool_and_invalid_backend_device(){
        let nonexistent=Path::new("this-test-path-does-not-exist");
        assert!(prepare_adapter_smoke(nonexistent,nonexistent,Path::new(""),
            0,None,None,None,32).is_err());
        assert!(prepare_adapter_smoke(nonexistent,nonexistent,Path::new("llama-cli"),
            0,Some("GPU $BAD"),None,None,32).is_err());
    }
}
