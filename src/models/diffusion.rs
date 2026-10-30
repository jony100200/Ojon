//! Image / video SVDQuant weights using an independently installed
//! public diffuse-compressor Python backend. Ojon never runs code on inspection,
//! downloads models, overwrites source weights, or invents quality scores.
//!
//! Route: local Diffusers pipeline -> SVDQuant INT4 backbone.
//! Native Ojon weight-only compression lives separately in packed.rs.
//! A converted diffusion model still needs real loader and image quality tests.

use crate::jobs::run_with_log;
use crate::probe::probe_model;
use serde_json::{json, Value};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn bad(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, s.into())
}
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DiffusionEngine {
    Svdq,
}
impl DiffusionEngine {
    pub fn from_name(name: &str) -> io::Result<Self> {
        match name {
            "svdq" => Ok(Self::Svdq),
            _ => Err(bad(
                "diffusion backend must be svdq; use pack-plan for native weights",
            )),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Svdq => "svdq",
        }
    }
}
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DiffusionKind {
    Image,
    Video,
}
impl DiffusionKind {
    pub fn from_name(name: &str) -> io::Result<Self> {
        match name {
            "image" => Ok(Self::Image),
            "video" => Ok(Self::Video),
            _ => Err(bad("model kind must be image or video")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
        }
    }
}
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum DiffusionFormat {
    Int4,
}
impl DiffusionFormat {
    pub fn from_name(name: &str) -> io::Result<Self> {
        match name {
            "int4" => Ok(Self::Int4),
            _ => Err(bad(
                "diffusion SVDQ format must be int4; use pack-plan for native 2/3/4-bit files",
            )),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Int4 => "int4",
        }
    }
}
#[derive(Debug)]
pub struct DiffusionPlan {
    pub engine: DiffusionEngine,
    pub kind: DiffusionKind,
    pub format: DiffusionFormat,
    pub source: PathBuf,
    pub source_bytes: u64,
    pub output: PathBuf,
    pub checkout: PathBuf,
    pub script: PathBuf,
    pub python: PathBuf,
    pub samples: u32,
}
impl DiffusionPlan {
    pub fn json(&self) -> Value {
        json!({
            "schema":"ojon.diffusion_plan.v1",
            "model_family":self.kind.name(),
            "backend":self.engine.name(),
            "precision":self.format.name(),
            "source":self.source.to_string_lossy(),
            "source_weight_bytes":self.source_bytes,
            "output":self.output.to_string_lossy(),
            "script":self.script.to_string_lossy(),
            "python":self.python.to_string_lossy(),
            "calibration_samples":if self.engine==DiffusionEngine::Svdq{Some(self.samples)}else{None},
            "status":"PLAN_ONLY",
            "estimated_peak_ram":"unknown",
            "estimated_peak_vram":"unknown",
            "output_load_verified":false,
            "visual_quality_evaluated":false,
            "warnings":[
                "Only confirmed compatible inference loaders can use the resulting weights",
                "Source, tokenizer, text encoder and VAE may remain necessary at their original precision",
                "Output file size alone cannot prove image or video generation quality",
                "No automatic model downloads; Python may still execute code from a user-supplied checkout",
            ],
        })
    }
}
fn script_from_checkout(checkout: &Path, relative: &str) -> io::Result<(PathBuf, PathBuf)> {
    let root = checkout.canonicalize()?;
    if !root.is_dir() {
        return Err(bad("backend checkout must be an existing directory"));
    }
    let script = root.join(relative);
    if !script.is_file() {
        return Err(bad(format!("upstream backend script missing: {relative}")));
    }
    Ok((root, script))
}
/// Read safetensors tensor descriptors only; do not materialize weights.
fn inspect_source_safetensors(source: &Path) -> io::Result<u64> {
    let found = probe_model(source)?;
    if found.format != "safetensors_single_file" {
        return Err(bad(
            "transformer weights must be original single-file safetensors shards",
        ));
    }
    let mut f = File::open(source)?;
    let mut prefix = [0u8; 8];
    f.read_exact(&mut prefix)?;
    let length = u64::from_le_bytes(prefix);
    if length == 0 || length > 32 * 1024 * 1024 {
        return Err(bad("oversized safetensors header"));
    }
    let mut header = vec![0; length as usize];
    f.read_exact(&mut header)?;
    let fields: Value = serde_json::from_slice(&header)
        .map_err(|e| bad(format!("invalid safetensors header: {e}")))?;
    let map = fields
        .as_object()
        .ok_or_else(|| bad("no tensor metadata"))?;
    let mut high = 0usize;
    for (name, tensor) in map.iter().filter(|(k, _)| k.as_str() != "__metadata__") {
        if name.ends_with(".qdata") || name.ends_with(".codebook") {
            return Err(bad("already packed quantized source; do not requantize"));
        }
        if matches!(tensor["dtype"].as_str(), Some("F16" | "BF16" | "F32")) {
            high += 1;
        }
    }
    if high == 0 {
        return Err(bad("source has no high-precision tensors to compress"));
    }
    Ok(found.bytes)
}
fn inspect_diffusers(source: &Path) -> io::Result<u64> {
    let manifest = source.join("model_index.json");
    let bytes = fs::read(&manifest)
        .map_err(|_| bad("SVDQ needs a local Diffusers pipeline with model_index.json"))?;
    if bytes.is_empty() || bytes.len() > 2 * 1024 * 1024 {
        return Err(bad("Diffusers model index is missing or oversized"));
    }
    let v: Value =
        serde_json::from_slice(&bytes).map_err(|e| bad(format!("bad model_index.json: {e}")))?;
    if !v.is_object() || v.get("transformer").is_none() {
        return Err(bad("SVDQ initial route requires a Diffusers transformer component; UNet-only models are not yet supported"));
    }
    let transformer = source.join("transformer");
    if !transformer.is_dir() {
        return Err(bad("Diffusers transformer directory missing"));
    }
    let mut total = 0u64;
    let mut weights = 0;
    // Sharded transformer snapshots can contain many weight shards, but all
    // metadata probes remain shallow and bounded.
    for e in fs::read_dir(transformer)?.take(4096) {
        let e = e?;
        let name = e.file_name();
        let name = name.to_string_lossy();
        if (name.ends_with(".safetensors") || name == "diffusion_pytorch_model.bin")
            && e.file_type()?.is_file()
        {
                if name.ends_with(".safetensors") {
                    // Reject malformed or already packed weights before scheduling
                    // any memory-intensive external Python processing.
                    inspect_source_safetensors(&e.path())?;
                }
                weights += 1;
                total = total
                    .checked_add(e.metadata()?.len())
                    .ok_or_else(|| bad("weight sizes overflow"))?;
            }
        }
    }
    if weights == 0 {
        return Err(bad(
            "Diffusers transformer weight shards not present locally",
        ));
    }
    Ok(total)
}
pub fn plan_diffusion(
    source: &Path,
    output: &Path,
    checkout: &Path,
    python: &Path,
    engine: DiffusionEngine,
    kind: DiffusionKind,
    format: DiffusionFormat,
    samples: u32,
) -> io::Result<DiffusionPlan> {
    if samples == 0 || samples > 256 {
        return Err(bad("calibration sample count must be 1..=256"));
    }
    if python.as_os_str().is_empty() {
        return Err(bad("Python executable not configured"));
    }
    if python.is_absolute() && !python.is_file() {
        return Err(bad("Python executable path does not exist"));
    }
    if output.extension().and_then(|x| x.to_str()) != Some("safetensors") {
        return Err(bad("output filename must end in .safetensors"));
    }
    let path = source.canonicalize()?;
    let (root, script, bytes) = match engine {
        DiffusionEngine::Svdq => {
            if format != DiffusionFormat::Int4 {
                return Err(bad("SVDQ on Ada: choose int4, not NVFP4 or W4A8 codebook"));
            }
            if !path.is_dir() {
                return Err(bad("SVDQ expects a local Diffusers pipeline folder"));
            }
            let (r, s) = script_from_checkout(
                checkout,
                match kind {
                    DiffusionKind::Image => "examples/text_to_image/quantize_hf.py",
                    DiffusionKind::Video => "examples/text_to_video/quantize_hf.py",
                },
            )?;
            (r, s, inspect_diffusers(&path)?)
        }
    };
    let name = output
        .file_name()
        .ok_or_else(|| bad("missing output filename"))?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let parent = parent.canonicalize()?;
    let out = parent.join(name);
    if out.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "quantized output already exists",
        ));
    }
    if out == path || (path.is_dir() && out.starts_with(&path)) {
        return Err(bad(
            "output cannot overwrite the source or be inside the source model directory",
        ));
    }
    // Adjacent metadata and experimental report are treated as outputs, too.
    let sidecar = out.with_extension("config.yaml");
    if engine == DiffusionEngine::Svdq && sidecar.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "quantized sidecar already exists",
        ));
    }
    Ok(DiffusionPlan {
        engine,
        kind,
        format,
        source: path,
        source_bytes: bytes,
        output: out,
        checkout: root,
        script,
        python: python.to_path_buf(),
        samples,
    })
}
fn command(plan: &DiffusionPlan, target: Option<&Path>, inspect: bool) -> Command {
    let mut cmd = Command::new(&plan.python);
    cmd.arg(&plan.script).current_dir(&plan.checkout);
    match plan.engine {
        DiffusionEngine::Svdq => {
            cmd.arg(&plan.source)
                .arg("--precision")
                .arg("int4")
                .arg("--rank")
                .arg("32")
                .arg("--num-samples")
                .arg(plan.samples.to_string())
                .arg("--cache-num-samples")
                .arg(plan.samples.to_string())
                .arg("--batch-size")
                .arg("1")
                .arg("--sample-batch-size")
                .arg("16")
                .arg("--height")
                .arg("512")
                .arg("--width")
                .arg("512")
                .arg("--device")
                .arg("cuda")
                .arg("--compute-device")
                .arg("cuda")
                .arg("--offload-model")
                .arg("--pipeline-offload")
                .arg("model");
            if plan.kind == DiffusionKind::Video {
                cmd.arg("--num-frames").arg("17").arg("--fps").arg("16");
            }
            if inspect {
                cmd.arg("--inspect-config");
            }
            if let Some(out) = target {
                cmd.arg("--output").arg(out);
            }
        }
    }
    cmd.env("HF_HUB_OFFLINE", "1")
        .env("HF_DATASETS_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("WANDB_MODE", "offline")
        .env("TOKENIZERS_PARALLELISM", "false");
    cmd
}
pub fn inspect_diffusion_recipe<F: FnMut(&str)>(
    plan: &DiffusionPlan,
    mut log: F,
) -> io::Result<()> {
    log("[ojon] SVDQ inspect-config invokes externally supplied Python and may load model weights");
    run_with_log(&mut command(plan, None, true), |line| log(line))?;
    log("[ojon] inspect completed; does not verify quantized output or GPU memory budget");
    Ok(())
}
struct Stage(PathBuf);
impl Drop for Stage {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn create_stage(parent: &Path) -> io::Result<Stage> {
    for retry in 0..8 {
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| bad("invalid system clock"))?
            .as_nanos();
        let dir = parent.join(format!(
            ".ojon-diffusion-{}-{t}-{retry}",
            std::process::id()
        ));
        match fs::create_dir(&dir) {
            Ok(()) => return Ok(Stage(dir)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(bad("cannot allocate temporary diffusion output"))
}
#[derive(Debug)]
pub struct DiffusionResult {
    pub output: PathBuf,
    pub output_bytes: u64,
    pub elapsed_seconds: f64,
    pub sidecar: Option<PathBuf>,
    pub receipt: Option<PathBuf>,
}
pub fn run_diffusion<F: FnMut(&str)>(
    plan: &DiffusionPlan,
    mut log: F,
) -> io::Result<DiffusionResult> {
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "output now exists",
        ));
    }
    let parent = plan
        .output
        .parent()
        .ok_or_else(|| bad("output parent missing"))?;
    let stage = create_stage(parent)?;
    let file = stage.0.join("quantized.safetensors");
    let sidecar = stage.0.join("quantized.config.yaml");
    log("[ojon] externally installed converter running; no model or output quality claims yet");
    let started = Instant::now();
    run_with_log(&mut command(plan, Some(&file), false), |line| log(line))?;
    // Format verification reads only bounded metadata and tensor descriptors.
    let inspect = probe_model(&file)?;
    if inspect.format != "safetensors_single_file" || inspect.bytes == 0 {
        return Err(bad(
            "external converter returned a missing or invalid safetensors output",
        ));
    }
    let published_sidecar = plan.output.with_extension("config.yaml");
    let has_sidecar = sidecar.is_file();
    if plan.engine == DiffusionEngine::Svdq && !has_sidecar {
        return Err(bad(
            "SVDQ converter did not produce required config.yaml metadata",
        ));
    }
    if has_sidecar && published_sidecar.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "output sidecar already exists",
        ));
    }
    // Hard links publish atomically without replacing existing files and keep
    // the staging file until both artifacts have been successfully published.
    fs::hard_link(&file, &plan.output)?;
    if has_sidecar {
        if let Err(e) = fs::hard_link(&sidecar, &published_sidecar) {
            let _ = fs::remove_file(&plan.output);
            return Err(e);
        }
    }
    let seconds = started.elapsed().as_secs_f64();
    let receipt_path = plan.output.with_extension("ojon.json");
    let receipt = json!({
        "schema":"ojon.diffusion_result.v1",
        "backend":plan.engine.name(),
        "kind":plan.kind.name(),
        "format":plan.format.name(),
        "source":plan.source.to_string_lossy(),
        "source_weight_bytes":plan.source_bytes,
        "output":plan.output.to_string_lossy(),
        "output_bytes":inspect.bytes,
        "elapsed_seconds":seconds,
        "safetensors_structure_checked":true,
        "runtime_load_verified":false,
        "quality_evaluated":false,
        "warning":"Check intended diffusion runtime, conditioning components and matched-seed image/video output against the original."
    });
    let mut manifest = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&receipt_path)
        .ok();
    if let Some(f) = manifest.as_mut() {
        if serde_json::to_writer_pretty(&mut *f, &receipt).is_err() || writeln!(f).is_err() {
            let _ = fs::remove_file(&receipt_path);
            manifest = None;
        }
    }
    log(&format!(
        "[ojon] format-valid output: {} bytes; runtime and visual quality NOT TESTED",
        inspect.bytes
    ));
    Ok(DiffusionResult {
        output: plan.output.clone(),
        output_bytes: inspect.bytes,
        elapsed_seconds: seconds,
        sidecar: has_sidecar.then_some(published_sidecar),
        receipt: manifest.map(|_| receipt_path),
    })
}
pub fn discover_diffusion() -> Value {
    let checks = [(
        "svdq",
        "OJON_SVDQ_DIR",
        "https://github.com/rootonchair/diffuse-compressor",
        &[
            "examples/text_to_image/quantize_hf.py",
            "examples/text_to_video/quantize_hf.py",
        ][..],
    )];
    let entries: Vec<Value> = checks
        .iter()
        .map(|(name, env, url, scripts)| {
            let root = std::env::var_os(env).map(PathBuf::from);
            let found = root
                .as_ref()
                .is_some_and(|r| r.is_dir() && scripts.iter().all(|s| r.join(s).is_file()));
            json!({
                "backend":name,"checkout":root.as_ref().map(|p|p.to_string_lossy().into_owned()),
                "environment_variable":env,"upstream":url,"scripts":scripts,
                "source_present":found,
                "python_dependencies_verified":false,
                "gpu_compatible":"not_verified",
                "loader_compatible":"not_verified",
                "image_video_quality":"not_measured"
            })
        })
        .collect();
    json!({"schema":"ojon.diffusion_backends.v1","backends":entries})
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_format_and_backend_must_match() {
        assert_eq!(
            DiffusionFormat::from_name("int4").unwrap(),
            DiffusionFormat::Int4
        );
        assert!(DiffusionFormat::from_name("nvfp4").is_err());
        assert!(DiffusionKind::from_name("audio").is_err());
        assert!(DiffusionEngine::from_name("codebook").is_err());
        assert!(DiffusionFormat::from_name("w4a8").is_err());
    }
    fn sample_weights(path: &Path) {
        let metadata = json!({"transformer_blocks.0.attn.weight":{
            "dtype":"F16","shape":[2,2],"data_offsets":[0,8]
        }})
        .to_string();
        let mut output = File::create(path).unwrap();
        output
            .write_all(&(metadata.len() as u64).to_le_bytes())
            .unwrap();
        output.write_all(metadata.as_bytes()).unwrap();
        output.write_all(&[0u8; 8]).unwrap();
    }
    struct Scratch(PathBuf);
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn scratch() -> Scratch {
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("ojon-diffusion-test-{}-{t}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        Scratch(dir)
    }
    #[test]
    fn plans_real_diffusers_folder_with_local_transformer_weights() {
        let dir = scratch();
        let source = dir.0.join("pipeline");
        fs::create_dir(&source).unwrap();
        fs::write(source.join("model_index.json"),r#"{"_class_name":"FluxPipeline","transformer":["diffusers","FluxTransformer2DModel"]}"#).unwrap();
        fs::create_dir(source.join("transformer")).unwrap();
        sample_weights(&source.join("transformer/diffusion_pytorch_model.safetensors"));
        let checkout = dir.0.join("compressor");
        fs::create_dir_all(checkout.join("examples/text_to_image")).unwrap();
        fs::write(checkout.join("examples/text_to_image/quantize_hf.py"), "").unwrap();
        let output = dir.0.join("denoiser-int4.safetensors");
        let plan = plan_diffusion(
            &source,
            &output,
            &checkout,
            Path::new("python"),
            DiffusionEngine::Svdq,
            DiffusionKind::Image,
            DiffusionFormat::Int4,
            16,
        )
        .unwrap();
        assert_eq!(plan.format, DiffusionFormat::Int4);
        assert!(!output.exists());
        assert!(plan_diffusion(
            &source,
            &output,
            &checkout,
            Path::new("python"),
            DiffusionEngine::Svdq,
            DiffusionKind::Image,
            DiffusionFormat::Int4,
            0
        )
        .is_err());
    }
    #[test]
    fn invalid_samples_rejected_before_any_external_commands() {
        let result = plan_diffusion(
            Path::new("missing"),
            Path::new("bad.safetensors"),
            Path::new("missing"),
            Path::new("python"),
            DiffusionEngine::Svdq,
            DiffusionKind::Image,
            DiffusionFormat::Int4,
            0,
        );
        assert!(result.is_err());
    }
}
