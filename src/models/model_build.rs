//! Stage-safe original HF -> local edit -> GGUF quantization -> VLM mmproj.
//!
//! Model editing uses native Rust. GGUF conversion is *currently* delegated to
//! the explicitly installed upstream llama.cpp Python converter and quantizer.
//! This workflow does not download models or perform quality evaluation.

use crate::hf::{inspect_hf_folder, prepare_hf, quantize_hf_with_log};
use crate::llama::Precision;
use crate::mmproj::{config_has_vision_tower, convert_mmproj_into, MmprojPrecision};
use crate::model_edit::{prepare_model_abliteration, run_model_abliteration};
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::{self, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, s.into())
}

#[derive(Debug)]
pub struct ModelBuildPlan {
    source: PathBuf,
    recipe: PathBuf,
    output: PathBuf,
    converter: PathBuf,
    python: PathBuf,
    quantizer: PathBuf,
    precision: Precision,
    threads: usize,
    is_vlm: bool,
    source_weight_bytes: u64,
}

impl ModelBuildPlan {
    pub fn as_json(&self) -> Value {
        json!({
            "schema":"ojon.model_build_plan.v1",
            "original_model":self.source.to_string_lossy(),
            "recipe":self.recipe.to_string_lossy(),
            "output_directory":self.output.to_string_lossy(),
            "source_weight_bytes":self.source_weight_bytes,
            "precision":self.precision.to_string(),
            "threads":self.threads,
            "vision_detected":self.is_vlm,
            "language_output":"model.gguf",
            "vision_output":if self.is_vlm {Some("mmproj.gguf")} else {None},
            "edited_hf_output":"edited-hf/",
            "editing_runtime":"RUST_NATIVE",
            "gguf_conversion_runtime":"EXTERNAL_INSTALLED_LLAMA_CPP_PYTHON",
            "quantization_runtime":"EXTERNAL_INSTALLED_LLAMA_QUANTIZE",
            "vision_qa":"NOT_RUN",
            "model_inference":"NOT_RUN",
            "refusal_optimization":"NOT_RUN",
            "status":"PLAN_ONLY"
        })
    }
}

/// Inputs for [`prepare_model_build`].
#[derive(Clone, Copy, Debug)]
pub struct ModelBuildRequest<'a> {
    pub source: &'a Path,
    pub recipe: &'a Path,
    pub output: &'a Path,
    pub converter: &'a Path,
    pub python: &'a Path,
    pub quantizer: &'a Path,
    pub precision: Precision,
    pub threads: usize,
}
/// Folder must contain local HF model files and original safetensors. This
/// preview does not mutate source, download weights, or start any subprocess.
pub fn prepare_model_build(request: ModelBuildRequest<'_>) -> io::Result<ModelBuildPlan> {
    let ModelBuildRequest {
        source,
        recipe,
        output,
        converter,
        python,
        quantizer,
        precision,
        threads,
    } = request;
    if threads == 0 {
        return Err(invalid("threads must be positive"));
    }
    let info = inspect_hf_folder(source)?;
    let source = info.directory;
    let cfg = source.join("config.json");
    let size = fs::metadata(&cfg)?.len();
    if size == 0 || size > 8 * 1024 * 1024 {
        return Err(invalid("model config.json missing or too large"));
    }
    let mut raw = Vec::with_capacity(size as usize);
    File::open(cfg)?
        .take(8 * 1024 * 1024 + 1)
        .read_to_end(&mut raw)?;
    let metadata: Value = serde_json::from_slice(&raw)
        .map_err(|e| invalid(format!("invalid model config.json: {e}")))?;
    let is_vlm = config_has_vision_tower(&metadata);
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    let output = parent.join(
        output
            .file_name()
            .ok_or_else(|| invalid("missing output directory name"))?,
    );
    if output.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "output folder exists",
        ));
    }
    if output.starts_with(&source) || source.starts_with(&output) {
        return Err(invalid("output folder must be outside source checkpoint"));
    }
    let recipe = recipe.canonicalize()?;
    if !recipe.is_file() {
        return Err(invalid("edit recipe must be an existing JSON file"));
    }
    // Reuse full model planner to verify all weight-map references and exact
    // allowed tensor edits BEFORE provisioning temp storage.
    let preflight = parent.join(".ojon-build-preflight-unused");
    prepare_model_abliteration(&source, &recipe, &preflight)?;
    let converter = if converter.is_dir() {
        converter.join("convert_hf_to_gguf.py")
    } else {
        converter.to_path_buf()
    };
    let converter = converter.canonicalize()?;
    if converter.file_name().and_then(|s| s.to_str()) != Some("convert_hf_to_gguf.py") {
        return Err(invalid(
            "converter must be official llama.cpp convert_hf_to_gguf.py",
        ));
    }
    if !converter.is_file() {
        return Err(invalid("converter not found"));
    }
    Ok(ModelBuildPlan {
        source,
        recipe,
        output,
        converter,
        python: python.to_path_buf(),
        quantizer: quantizer.to_path_buf(),
        precision,
        threads,
        is_vlm,
        source_weight_bytes: info.source_weight_bytes,
    })
}

struct Temp(PathBuf);
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn stage(parent: &Path) -> io::Result<Temp> {
    for attempt in 0..12 {
        let ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| invalid("system time unavailable"))?
            .as_nanos();
        let path = parent.join(format!(".ojon-build-{}-{ns}-{attempt}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(Temp(path)),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(invalid("cannot create new model build staging folder"))
}

#[derive(Debug)]
pub struct ModelBuildResult {
    pub directory: PathBuf,
    pub has_mmproj: bool,
    pub model_bytes: u64,
    pub mmproj_bytes: Option<u64>,
}
impl ModelBuildResult {
    pub fn as_json(&self) -> Value {
        json!({
            "schema":"ojon.model_build_result.v1",
            "directory":self.directory.to_string_lossy(),
            "language_gguf":self.directory.join("model.gguf").to_string_lossy(),
            "mmproj_gguf":if self.has_mmproj { Some(self.directory.join("mmproj.gguf").to_string_lossy().to_string()) }else{None},
            "language_bytes":self.model_bytes,
            "mmproj_bytes":self.mmproj_bytes,
            "model_load":"NOT_TESTED",
            "vision_quality":"NOT_TESTED",
            "refusal_rate":"NOT_EVALUATED",
            "coding_quality":"NOT_EVALUATED",
            "status":"UNVERIFIED_LOCAL_ARTIFACTS_CREATED"
        })
    }
}

/// Copies/edits original HF shard set into a new temp folder, quantizes the
/// edited language model and converts an independent high-precision vision
/// GGUF if the config indicates a vision model. Output stays unpublished if
/// any essential step fails (unsupported multimodal converter = hard error).
pub fn run_model_build(
    plan: &ModelBuildPlan,
    mut log: impl FnMut(&str),
) -> io::Result<ModelBuildResult> {
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "output directory exists",
        ));
    }
    let stage = stage(
        plan.output
            .parent()
            .ok_or_else(|| invalid("missing build parent"))?,
    )?;
    let edited = stage.0.join("edited-hf");
    let edit_plan = prepare_model_abliteration(&plan.source, &plan.recipe, &edited)?;
    log("[ojon] native Rust editing the Hugging Face checkpoint");
    run_model_abliteration(&edit_plan, |line| log(line))?;
    let gguf = stage.0.join("model.gguf");
    let quant_plan = prepare_hf(
        &edited,
        &gguf,
        &plan.converter,
        &plan.python,
        &plan.quantizer,
        plan.precision,
        plan.threads,
    )?;
    log("[ojon] converting edited language checkpoint and quantizing GGUF");
    let quantized = quantize_hf_with_log(&quant_plan, |line| log(line))?;
    if quantized.info.tensor_count == 0 {
        return Err(invalid("no tensors in quantized language GGUF"));
    }
    let mut mmproj_bytes = None;
    if plan.is_vlm {
        log("[ojon] converting VLM projector (Q8_0, separate from language GGUF)");
        let expected = stage.0.join("mmproj.gguf");
        let info = convert_mmproj_into(
            &plan.python,
            &plan.converter,
            &edited,
            &expected,
            MmprojPrecision::Q8_0,
            &mut log,
        )?;
        mmproj_bytes = Some(info.file_bytes);
    }
    let receipt = json!({
        "schema":"ojon.model_build_receipt.v1",
        "source":plan.source.to_string_lossy(),
        "edited_hf":"edited-hf/",
        "language_gguf":"model.gguf",
        "language_precision":plan.precision.to_string(),
        "vision_mmproj":if plan.is_vlm {Some("mmproj.gguf")} else {None},
        "vision_precision":if plan.is_vlm {Some("Q8_0")} else {None},
        "language_bytes":quantized.info.file_bytes,
        "mmproj_bytes":mmproj_bytes,
        "model_load_test":"NOT_RUN",
        "image_inference_test":"NOT_RUN",
        "refusal_quality_test":"NOT_RUN",
        "quantization_backend":"llama.cpp",
        "model_edit_backend":"ojon_native_rust_directions"
    });
    let receipt_path = stage.0.join("ojon-build-receipt.json");
    let mut receipt_file = File::create(&receipt_path)?;
    receipt_file.write_all(
        serde_json::to_string_pretty(&receipt)
            .map_err(|e| invalid(format!("receipt: {e}")))?
            .as_bytes(),
    )?;
    receipt_file.sync_all()?;
    drop(receipt_file); // Required before Windows directory rename.
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "output directory appeared during build",
        ));
    }
    fs::rename(&stage.0, &plan.output)?;
    log(&format!(
        "[ojon] model artifacts published: {}",
        plan.output.display()
    ));
    Ok(ModelBuildResult {
        directory: plan.output.clone(),
        has_mmproj: plan.is_vlm,
        model_bytes: quantized.info.file_bytes,
        mmproj_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn result_does_not_claim_model_verified() {
        let r = ModelBuildResult {
            directory: PathBuf::from("models/out"),
            has_mmproj: true,
            model_bytes: 123,
            mmproj_bytes: Some(34),
        };
        assert_eq!(r.as_json()["status"], "UNVERIFIED_LOCAL_ARTIFACTS_CREATED");
        assert_eq!(r.as_json()["vision_quality"], "NOT_TESTED");
        assert_eq!(r.as_json()["mmproj_bytes"], 34);
    }
}
