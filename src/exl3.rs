//! Optional EXL3 conversion using a user-installed ExLlamaV3 checkout.
//! No tensors are loaded by Ojon. EXL3 is not GGUF and requires its own loader.
//! Interrupted work is deliberately retained for upstream resumable conversion.
use crate::hf::{inspect_hf_folder, HfFolderInfo};
use crate::jobs::run_with_log;
use crate::resource::{assess_resources, ResourceMode};
use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::{self, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

const STATE_NAME: &str = ".ojon-exl3-job.json";

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message.into())
}

#[derive(Clone, Debug)]
pub struct Exl3Plan {
    pub source: HfFolderInfo,
    pub output: PathBuf,
    pub work_dir: PathBuf,
    pub converter: PathBuf,
    pub python: PathBuf,
    pub bits: f64,
    pub high_quality: bool,
    pub resume: bool,
}

impl Exl3Plan {
    fn job_record(&self) -> Value {
        json!({
            "schema":"ojon.exl3_job.v1",
            "source":self.source.directory.to_string_lossy(),
            "output":self.output.to_string_lossy(),
            "work_dir":self.work_dir.to_string_lossy(),
            "converter":self.converter.to_string_lossy(),
            "bits":self.bits,
            "high_quality":self.high_quality,
        })
    }
    pub fn as_json(&self) -> Value {
        json!({
            "schema":"ojon.exl3_plan.v1",
            "backend":"exllamav3",
            "output_format":"EXL3 (NOT GGUF)",
            "source":self.source.directory.to_string_lossy(),
            "source_weight_bytes":self.source.source_weight_bytes,
            "output":self.output.to_string_lossy(),
            "work_dir":self.work_dir.to_string_lossy(),
            "converter":self.converter.to_string_lossy(),
            "python":self.python.to_string_lossy(),
            "target_bits_per_weight":self.bits,
            "high_quality":self.high_quality,
            "resume":self.resume,
            "device":"cuda",
            "status":"PLAN_ONLY",
            "runtime_load_verified":false,
            "task_quality_verified":false,
            "model_compatibility_verified":false,
            "expected_peak_ram_bytes":null,
            "expected_peak_vram_bytes":null,
            "conversion_writes_user_owned_work_dir":true,
            "output_may_be_incomplete_after_failure":true,
            "recovery":"Keep work directory and rerun with --resume; successful conversion still needs matching EXL3 loader and quality tests.",
            "upstream":"https://github.com/turboderp-org/exllamav3",
        })
    }
}

/// Normalize a destination without creating it. Only an existing parent is allowed.
fn destination(path: &Path) -> io::Result<PathBuf> {
    if !path.is_absolute() { return Err(invalid("output and work paths must be absolute")); }
    let parent=path.parent().ok_or_else(||invalid("path has no parent"))?.canonicalize()?;
    if !parent.is_dir() { return Err(invalid("destination parent must be an existing directory")); }
    let name=path.file_name().ok_or_else(||invalid("destination needs a filename"))?;
    Ok(parent.join(name))
}

/// Plans only. Verify input is a local unquantized Hugging Face checkpoint.
pub fn prepare_exl3(
    source: &Path, output: &Path, work_dir: &Path,
    checkout: &Path, python: &Path, bits: f64, high_quality: bool, resume: bool,
) -> io::Result<Exl3Plan> {
    if !bits.is_finite() || !(2.0..=8.0).contains(&bits) {
        return Err(invalid("EXL3 target bitrate must be finite and between 2.0 and 8.0"));
    }
    let source=inspect_hf_folder(source)?;
    if !source.directory.join("tokenizer.json").is_file() {
        return Err(invalid("EXL3 source needs tokenizer.json"));
    }
    let config_file=source.directory.join("config.json");
    if fs::metadata(&config_file)?.len()>2_000_000 {
        return Err(invalid("config.json exceeds 2 MiB"));
    }
    let config:Value=serde_json::from_slice(&fs::read(config_file)?)
        .map_err(|_|invalid("source config.json is not valid JSON"))?;
    if config.get("quantization_config").is_some() {
        return Err(invalid("EXL3 requires original unquantized HF weights"));
    }
    let safetensors=fs::read_dir(&source.directory)?.filter_map(Result::ok)
        .any(|item| item.file_name().to_string_lossy().ends_with(".safetensors")
            && item.metadata().is_ok_and(|meta|meta.is_file() && meta.len()>0));
    if !safetensors { return Err(invalid("EXL3 requires nonempty safetensors weights")); }
    let converter=if checkout.is_dir() {checkout.join("convert.py")} else {checkout.to_path_buf()};
    let converter=converter.canonicalize()?;
    if converter.file_name().and_then(|s|s.to_str())!=Some("convert.py") || !converter.is_file() {
        return Err(invalid("supply an installed ExLlamaV3 checkout containing convert.py"));
    }
    if python.as_os_str().is_empty() || (python.is_absolute() && !python.is_file()) {
        return Err(invalid("Python executable must exist when passed as an absolute path"));
    }
    let output=destination(output)?;
    let work_dir=destination(work_dir)?;
    if output==work_dir || output.starts_with(&work_dir) || work_dir.starts_with(&output) {
        return Err(invalid("output and work directories must be separate"));
    }
    if output.starts_with(&source.directory) || work_dir.starts_with(&source.directory) {
        return Err(invalid("output/work directories must not be inside original source"));
    }
    if resume {
        if !work_dir.is_dir() { return Err(invalid("resume requires an existing Ojon work directory")); }
    } else {
        if output.symlink_metadata().is_ok() || work_dir.symlink_metadata().is_ok() {
            return Err(io::Error::new(ErrorKind::AlreadyExists, "new EXL3 output and work paths must not exist"));
        }
    }
    Ok(Exl3Plan {source,output,work_dir,converter,python:python.to_path_buf(),
        bits,high_quality,resume})
}

fn existing_job_matches(plan: &Exl3Plan) -> io::Result<()> {
    let marker=plan.work_dir.join(STATE_NAME);
    if fs::metadata(&marker)?.len()>65536 {return Err(invalid("EXL3 work marker exceeds limit")); }
    let stored:Value=serde_json::from_slice(&fs::read(marker)?)
        .map_err(|_|invalid("EXL3 resume marker is malformed"))?;
    if stored!=plan.job_record() {
        return Err(invalid("EXL3 resume does not match original source, output, converter or recipe"));
    }
    Ok(())
}

fn verify_bundle(out: &Path) -> io::Result<u64> {
    if !out.join("config.json").is_file() {
        return Err(io::Error::other("EXL3 result is missing config.json"));
    }
    let mut weight_bytes=0u64;
    for entry in fs::read_dir(out)?.take(4096) {
        let entry=entry?;
        if entry.file_name().to_string_lossy().ends_with(".safetensors") {
            let metadata=entry.metadata()?;
            if metadata.is_file() {
                weight_bytes=weight_bytes.checked_add(metadata.len())
                    .ok_or_else(||io::Error::other("output size overflow"))?;
            }
        }
    }
    if weight_bytes==0 {return Err(io::Error::other("EXL3 output missing nonempty safetensors")); }
    Ok(weight_bytes)
}

/// Launch installed upstream converter. A failed/interrupted job retains partial
/// output and its separate work directory so the upstream -r mechanism can resume.
pub fn run_exl3(plan: &Exl3Plan, allow_resource_risk: bool) -> io::Result<Value> {
    let parent=plan.output.parent().ok_or_else(||invalid("missing output parent"))?;
    let preflight=assess_resources(plan.source.source_weight_bytes,parent,ResourceMode::Exl3,"cuda")?;
    eprintln!("Ojon EXL3 resource preflight: {}",preflight.as_json());
    preflight.enforce(allow_resource_risk)?;

    if plan.resume {
        existing_job_matches(plan)?;
    } else {
        if plan.output.symlink_metadata().is_ok() || plan.work_dir.symlink_metadata().is_ok() {
            return Err(io::Error::new(ErrorKind::AlreadyExists,"EXL3 job destination already exists"));
        }
        fs::create_dir(&plan.work_dir)?;
        let mut marker=OpenOptions::new().write(true).create_new(true)
            .open(plan.work_dir.join(STATE_NAME))?;
        marker.write_all(serde_json::to_string_pretty(&plan.job_record())
            .map_err(io::Error::other)?.as_bytes())?;
        marker.flush()?;
    }
    let mut command=Command::new(&plan.python);
    command.arg(&plan.converter);
    if plan.resume {
        command.arg("-w").arg(&plan.work_dir).arg("-r");
    } else {
        command.arg("-i").arg(&plan.source.directory)
            .arg("-o").arg(&plan.output)
            .arg("-w").arg(&plan.work_dir)
            .arg("-b").arg(plan.bits.to_string());
        if plan.high_quality { command.arg("-hq"); }
    }
    command.env("HF_HUB_OFFLINE","1")
        .env("HF_DATASETS_OFFLINE","1")
        .env("TRANSFORMERS_OFFLINE","1")
        .env("HF_HUB_DISABLE_TELEMETRY","1");
    run_with_log(&mut command, |line| println!("{line}"))?;
    let output_weight_bytes=verify_bundle(&plan.output)?;
    Ok(json!({
        "schema":"ojon.exl3_result.v1",
        "status":"EXPORTED_FORMAT_ONLY",
        "output":plan.output.to_string_lossy(),
        "work_dir":plan.work_dir.to_string_lossy(),
        "output_weight_bytes":output_weight_bytes,
        "runtime_load_verified":false,
        "task_quality_verified":false,
        "warning":"EXL3 output files were detected after successful conversion; runtime compatibility, memory fit and task quality are NOT verified."
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};
    fn fixture() -> (PathBuf,PathBuf,PathBuf) {
        let stamp=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root=std::env::temp_dir().join(format!("ojon-exl3-{}-{stamp}",std::process::id()));
        let source=root.join("original");
        let checkout=root.join("checkout");
        fs::create_dir_all(&source).unwrap();
        fs::create_dir_all(&checkout).unwrap();
        fs::write(source.join("config.json"),r#"{"model_type":"llama"}"#).unwrap();
        fs::write(source.join("tokenizer.json"),b"{}").unwrap();
        fs::write(source.join("model.safetensors"),b"full-precision-fixture").unwrap();
        fs::write(checkout.join("convert.py"),b"# fixture").unwrap();
        (root,source,checkout)
    }
    #[test]
    fn plan_checks_unquantized_input_and_safe_paths() {
        let (root,source,checkout)=fixture();
        let out=root.join("exl3");
        let work=root.join("work");
        let p=prepare_exl3(&source,&out,&work,&checkout,Path::new("python"),3.5,true,false).unwrap();
        assert_eq!(p.as_json()["output_format"],"EXL3 (NOT GGUF)");
        assert_eq!(p.as_json()["runtime_load_verified"],false);
        assert!(prepare_exl3(&source,&out,&work,&checkout,Path::new("python"),0.5,false,false).is_err());
        assert!(prepare_exl3(&source,&source.join("out"),&work,&checkout,Path::new("python"),3.0,false,false).is_err());
        assert!(prepare_exl3(&source,&out,&out,&checkout,Path::new("python"),3.0,false,false).is_err());
        fs::write(source.join("config.json"),r#"{"quantization_config":{"bits":4}}"#).unwrap();
        assert!(prepare_exl3(&source,&out,&work,&checkout,Path::new("python"),3.0,false,false).is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn resume_refuses_unknown_work_directory_or_changed_recipe() {
        let (root,source,checkout)=fixture();
        let out=root.join("exl3");
        let work=root.join("work");
        let plan=prepare_exl3(&source,&out,&work,&checkout,Path::new("python"),3.0,false,false).unwrap();
        fs::create_dir_all(&work).unwrap();
        fs::write(work.join(STATE_NAME),serde_json::to_vec(&plan.job_record()).unwrap()).unwrap();
        let resumed=prepare_exl3(&source,&out,&work,&checkout,Path::new("python"),3.0,false,true).unwrap();
        assert!(existing_job_matches(&resumed).is_ok());
        let changed=prepare_exl3(&source,&out,&work,&checkout,Path::new("python"),4.0,false,true).unwrap();
        assert!(existing_job_matches(&changed).is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn incomplete_bundle_not_reported_as_verified() {
        let (root,_,_)=fixture();
        assert!(verify_bundle(&root).is_err());
        fs::write(root.join("config.json"),b"{}").unwrap();
        assert!(verify_bundle(&root).is_err());
        fs::write(root.join("quant.safetensors"),b"abc").unwrap();
        assert_eq!(verify_bundle(&root).unwrap(),3);
        fs::remove_dir_all(root).unwrap();
    }
}
