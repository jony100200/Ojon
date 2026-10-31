//! Working image/video GGUF conversion and runtime smoke path through the
//! user's installed stable-diffusion.cpp sd-cli. No Python or downloaded weights.
//! A valid container or generated preview is not a visual quality benchmark.
use crate::gguf::{inspect_gguf, inspect_gguf_tensors, GgufTensor};
use crate::jobs::run_with_log;
use crate::probe::probe_model;
use serde_json::{json, Value};
use std::fs::{self, File, OpenOptions};
use std::io::{self, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn bad(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message.into())
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MediaKind {
    Image,
    Video,
}
impl MediaKind {
    pub fn parse(text: &str) -> io::Result<Self> {
        match text {
            "image" => Ok(Self::Image),
            "video" => Ok(Self::Video),
            _ => Err(bad("kind must be image or video")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Video => "video",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputLayout {
    Checkpoint,
    Diffusion,
}
impl InputLayout {
    pub fn parse(text: &str) -> io::Result<Self> {
        match text {
            "checkpoint" => Ok(Self::Checkpoint),
            "diffusion" => Ok(Self::Diffusion),
            _ => Err(bad("input must be checkpoint or diffusion")),
        }
    }
    fn flag(self) -> &'static str {
        match self {
            Self::Checkpoint => "-m",
            Self::Diffusion => "--diffusion-model",
        }
    }
    fn name(self) -> &'static str {
        match self {
            Self::Checkpoint => "checkpoint",
            Self::Diffusion => "diffusion",
        }
    }
}
/// Which external converter writes the GGUF. `sd-cli` covers the architectures
/// stable-diffusion.cpp knows. The ComfyUI-GGUF converter (molbal fork,
/// `tools/convert.py`) covers what ComfyUI loads, FIBO 2 included, and
/// quantizes in the same Python pass: `--src <safetensors> --dst <gguf>
/// --streamed --quant-type <TYPE>` (streamed = one tensor at a time, low RAM).
#[derive(Clone, Debug, PartialEq)]
pub enum GgufConverter {
    SdCli,
    ComfyGguf { python: PathBuf },
}

/// sd-cli formats with nominal bits per weight (ordering only).
pub const SD_CLI_FORMATS: [(&str, f64); 5] = [("q8_0", 8.5), ("q5_0", 5.5), ("q4_K", 4.5), ("q4_0", 4.5), ("q3_K", 3.4)];
/// ComfyUI-GGUF converter quant types (its `QUANT_TYPE_MAP`, minus the CR
/// layouts that need the fork's own loader), with nominal bits per weight.
pub const COMFY_GGUF_FORMATS: [(&str, f64); 14] = [
    ("BF16", 16.0), ("F16", 16.0), ("Q8_0", 8.5), ("Q6_K", 6.6), ("Q5_1", 6.0), ("Q5_K_M", 5.7), ("Q5_K_S", 5.5),
    ("Q5_K", 5.5), ("Q5_0", 5.5), ("Q4_1", 5.0), ("Q4_K_M", 4.8), ("Q4_K_S", 4.6), ("Q4_K", 4.5), ("Q4_0", 4.5),
];

impl GgufConverter {
    pub fn name(&self) -> &'static str {
        match self {
            Self::SdCli => "stable-diffusion.cpp",
            Self::ComfyGguf { .. } => "comfyui-gguf convert.py",
        }
    }
    pub fn parse(text: &str, python: &Path) -> io::Result<Self> {
        match text {
            "sd-cli" | "sdcpp" | "stable-diffusion.cpp" => Ok(Self::SdCli),
            "comfy-gguf" | "comfyui-gguf" => Ok(Self::ComfyGguf { python: python.to_path_buf() }),
            _ => Err(bad("converter must be sd-cli or comfy-gguf")),
        }
    }
    pub fn formats(&self) -> &'static [(&'static str, f64)] {
        match self {
            Self::SdCli => &SD_CLI_FORMATS,
            Self::ComfyGguf { .. } => &COMFY_GGUF_FORMATS,
        }
    }
    pub fn nominal_bits(&self, format: &str) -> Option<f64> {
        self.formats().iter().find(|(f, _)| *f == format).map(|(_, b)| *b)
    }
    /// The GGML storage type id the converter must have produced for `format`
    /// (mixed presets are judged by their dominant type).
    pub fn storage_type(&self, format: &str) -> io::Result<u32> {
        let id = match (self, format) {
            (Self::SdCli, "q8_0") | (Self::ComfyGguf { .. }, "Q8_0") => 8,
            (Self::SdCli, "q5_0") | (Self::ComfyGguf { .. }, "Q5_0") => 6,
            (Self::SdCli, "q4_0") | (Self::ComfyGguf { .. }, "Q4_0") => 2,
            (Self::SdCli, "q4_K") | (Self::ComfyGguf { .. }, "Q4_K" | "Q4_K_S" | "Q4_K_M") => 12,
            (Self::SdCli, "q3_K") => 11,
            (Self::ComfyGguf { .. }, "Q5_K" | "Q5_K_S" | "Q5_K_M") => 13,
            (Self::ComfyGguf { .. }, "Q6_K") => 14,
            (Self::ComfyGguf { .. }, "Q5_1") => 7,
            (Self::ComfyGguf { .. }, "Q4_1") => 3,
            (Self::ComfyGguf { .. }, "F16") => 1,
            (Self::ComfyGguf { .. }, "BF16") => 30,
            _ => return Err(bad(format!("{format} is not a format of {}", self.name()))),
        };
        Ok(id)
    }
}

fn precision(text: &str) -> io::Result<&str> {
    match text {
        "q8_0" | "q5_0" | "q4_0" | "q4_K" | "q3_K" => Ok(text),
        _ => Err(bad("GGUF precision must be q8_0, q5_0, q4_0, q4_K or q3_K")),
    }
}
// Refuse packed SVDQ/Nunchaku checkpoints before passing them to a
// converter expecting ordinary floating-point denoiser tensors. GGUF
// output is weight-only; already packed modules need their native loader.
fn verify_dense_media_source(path: &Path) -> io::Result<()> {
    let mut file = File::open(path)?;
    let size = file.metadata()?.len();
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)?;
    let header_bytes = u64::from_le_bytes(prefix);
    if header_bytes == 0 || header_bytes > 32 * 1024 * 1024 || header_bytes > size.saturating_sub(8)
    {
        return Err(bad("invalid or excessive safetensors header"));
    }
    let mut metadata = vec![0u8; header_bytes as usize];
    file.read_exact(&mut metadata)?;
    let value: Value =
        serde_json::from_slice(&metadata).map_err(|_| bad("invalid safetensors tensor catalog"))?;
    let catalog = value
        .as_object()
        .ok_or_else(|| bad("invalid safetensors header object"))?;
    let mut dense = 0usize;
    for (name, entry) in catalog
        .iter()
        .filter(|(name, _)| name.as_str() != "__metadata__")
    {
        if [".qweight", ".qdata", ".qzeros", ".codebook", ".wscales"]
            .iter()
            .any(|suffix| name.ends_with(suffix))
        {
            return Err(bad("packed low-bit checkpoint detected; use the matching W4A4 runtime instead of sd-cli GGUF conversion"));
        }
        if matches!(entry["dtype"].as_str(), Some("F16" | "BF16" | "F32")) {
            dense += 1;
        }
    }
    if dense == 0 {
        return Err(bad(
            "GGUF conversion requires original floating-point model tensors",
        ));
    }
    Ok(())
}

// GGML tensor types, not activation quantization or Nunchaku W4A4.
// The target backend may keep sensitive tensors at a higher precision.
fn requested_storage_type(name: &str) -> io::Result<u32> {
    GgufConverter::SdCli.storage_type(precision(name)?)
}
fn count_requested_tensors(tensors: &[GgufTensor], name: &str, converter: &GgufConverter) -> io::Result<usize> {
    let requested = converter.storage_type(name)?;
    let count = tensors
        .iter()
        .filter(|t| t.storage_type == requested)
        .count();
    if count == 0 {
        return Err(bad(format!(
            "the converter returned a GGUF without any {name} tensors; refusing an unquantized or incompatible artifact"
        )));
    }
    Ok(count)
}
fn target(path: &Path, extension: &str) -> io::Result<PathBuf> {
    if path.extension().and_then(|s| s.to_str()) != Some(extension) {
        return Err(bad(format!("output must end in .{extension}")));
    }
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    if !parent.is_dir() {
        return Err(bad("output parent must be a directory"));
    }
    let filename = path
        .file_name()
        .ok_or_else(|| bad("missing output filename"))?;
    let output = parent.join(filename);
    if output.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "output already exists",
        ));
    }
    if receipt_path(&output).symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "output receipt already exists",
        ));
    }
    Ok(output)
}
fn receipt_path(path: &Path) -> PathBuf {
    PathBuf::from(format!("{}.ojon.json", path.to_string_lossy()))
}
fn checked_tool(tool: &Path) -> io::Result<PathBuf> {
    if tool.as_os_str().is_empty() {
        return Err(bad("sd-cli executable is required"));
    }
    if (tool.is_absolute() || tool.components().count() > 1) && !tool.is_file() {
        return Err(bad("configured sd-cli executable does not exist"));
    }
    Ok(tool.to_path_buf())
}
struct Stage(PathBuf);
impl Drop for Stage {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn stage(parent: &Path) -> io::Result<Stage> {
    for attempt in 0..12 {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| bad("clock before UNIX epoch"))?
            .as_nanos();
        let p = parent.join(format!(
            ".ojon-sdcpp-{}-{nanos}-{attempt}",
            std::process::id()
        ));
        match fs::create_dir(&p) {
            Ok(()) => return Ok(Stage(p)),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(bad("could not allocate a unique staging directory"))
}
fn atomic_receipt(output: &Path, value: &Value) -> io::Result<PathBuf> {
    let dest = receipt_path(output);
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&dest)?;
    if let Err(e) = serde_json::to_writer_pretty(&mut file, value)
        .map_err(io::Error::other)
        .and_then(|_| writeln!(file))
        .and_then(|_| file.sync_all())
    {
        drop(file);
        let _ = fs::remove_file(&dest);
        return Err(e);
    }
    Ok(dest)
}
#[derive(Debug)]
pub struct GgufConvertPlan {
    pub source: PathBuf,
    pub output: PathBuf,
    pub tool: PathBuf,
    pub layout: InputLayout,
    pub kind: MediaKind,
    pub format: String,
    pub source_bytes: u64,
    /// Which program writes the file; `tool` is sd-cli for SdCli and the
    /// `convert.py` script for ComfyGguf.
    pub converter: GgufConverter,
}
pub fn prepare_gguf_convert(
    source: &Path,
    output: &Path,
    tool: &Path,
    kind: MediaKind,
    layout: InputLayout,
    format: &str,
) -> io::Result<GgufConvertPlan> {
    prepare_gguf_convert_with(source, output, tool, kind, layout, format, GgufConverter::SdCli)
}

pub fn prepare_gguf_convert_with(
    source: &Path,
    output: &Path,
    tool: &Path,
    kind: MediaKind,
    layout: InputLayout,
    format: &str,
    converter: GgufConverter,
) -> io::Result<GgufConvertPlan> {
    if converter.nominal_bits(format).is_none() {
        return Err(bad(format!(
            "{format} is not a format of {}; accepted: {}",
            converter.name(),
            converter.formats().iter().map(|(f, _)| *f).collect::<Vec<_>>().join(", ")
        )));
    }
    let source = source.canonicalize()?;
    if !source.is_file() || source.extension().and_then(|s| s.to_str()) != Some("safetensors") {
        return Err(bad(
            "first GGUF media milestone accepts local standalone .safetensors files",
        ));
    }
    let info = probe_model(&source)?;
    if info.format != "safetensors_single_file" || info.tensor_count == Some(0) {
        return Err(bad("source lacks valid safetensors tensor descriptors"));
    }
    verify_dense_media_source(&source)?;
    if kind == MediaKind::Video && layout == InputLayout::Checkpoint {
        return Err(bad("video requires a standalone diffusion denoiser"));
    }
    let output = target(output, "gguf")?;
    if source == output {
        return Err(bad("output cannot replace source"));
    }
    let tool = match &converter {
        GgufConverter::SdCli => checked_tool(tool)?,
        GgufConverter::ComfyGguf { python } => {
            if !tool.is_file() || tool.extension().and_then(|e| e.to_str()) != Some("py") {
                return Err(bad("comfy-gguf converter needs the path to ComfyUI-GGUF tools/convert.py (--convert-script)"));
            }
            if python.as_os_str().is_empty() {
                return Err(bad("comfy-gguf converter needs a Python interpreter with torch and gguf"));
            }
            tool.to_path_buf()
        }
    };
    Ok(GgufConvertPlan {
        source,
        output,
        tool,
        kind,
        layout,
        format: format.into(),
        source_bytes: info.bytes,
        converter,
    })
}
impl GgufConvertPlan {
    pub fn json(&self) -> Value {
        json!({"schema":"ojon.diffusion_gguf_plan.v1","status":"PLAN_ONLY",
        "backend":self.converter.name(),"mode":"convert","kind":self.kind.name(),
        "input_layout":self.layout.name(),"source":self.source.to_string_lossy(),
        "source_bytes":self.source_bytes,"output":self.output.to_string_lossy(),
        "tool":self.tool.to_string_lossy(),"precision":self.format,
        "model_loaded":false,"runtime_load_verified":false,
        "quality_verified":false,
        "quantization_semantics":"weight_only; NOT W4A4",
        "comfyui_gguf_runtime_verified":false,
        "cross_runtime_recommended_formats":["q4_0","q5_0","q8_0"],
        "warnings":[
            "The output contains converted diffusion weights, not automatically the text encoders or VAE.",
            "Architecture-specific conversion support depends on the installed sd-cli build.",
            "Convert first, then run diffusion-gguf-smoke-run with the required conditioning components.",
            "Checkpoint size and GGUF validity do not establish visual quality or actual VRAM usage."
        ]})
    }
    /// The exact argv, for receipts and tests.
    pub fn command_args(&self, dest: &Path) -> Vec<String> {
        let p = |x: &Path| x.to_string_lossy().into_owned();
        match &self.converter {
            GgufConverter::SdCli => vec![
                "-M".into(), "convert".into(), self.layout.flag().into(), p(&self.source),
                "-o".into(), p(dest), "--type".into(), self.format.clone(),
            ],
            GgufConverter::ComfyGguf { .. } => vec![
                p(&self.tool), "--src".into(), p(&self.source), "--dst".into(), p(dest),
                "--streamed".into(), "--quant-type".into(), self.format.clone(),
            ],
        }
    }
    fn command(&self, dest: &Path) -> Command {
        let mut cmd = match &self.converter {
            GgufConverter::SdCli => Command::new(&self.tool),
            GgufConverter::ComfyGguf { python } => crate::jobs::tool_command(python),
        };
        cmd.args(self.command_args(dest));
        cmd
    }
}
pub fn run_gguf_convert(plan: &GgufConvertPlan, mut log: impl FnMut(&str)) -> io::Result<Value> {
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "output already exists",
        ));
    }
    if receipt_path(&plan.output).symlink_metadata().is_ok() {
        return Err(bad("receipt exists"));
    }
    let parent = plan
        .output
        .parent()
        .ok_or_else(|| bad("missing output parent"))?;
    let stage = stage(parent)?;
    let candidate = stage.0.join("candidate.gguf");
    let start = Instant::now();
    log(&format!("[ojon] starting real GGUF conversion with {}; no visual-quality claim", plan.converter.name()));
    run_with_log(&mut plan.command(&candidate), |line| log(line))?;
    let info = inspect_gguf(&candidate)?;
    if info.tensor_count == 0 || info.split_count.unwrap_or(1) > 1 {
        return Err(bad("converter did not produce a nonempty single-file GGUF"));
    }
    // A structurally valid GGUF can still contain only F16/F32 weights if
    // the selected converter ignored or fell back from the requested type.
    // Read the bounded tensor catalog to ensure the requested GGML type exists
    // before publishing, while still permitting mixed high-precision tensors.
    let tensors = inspect_gguf_tensors(&candidate)?;
    let requested_tensor_count = count_requested_tensors(&tensors, &plan.format, &plan.converter)?;
    let output_bytes = info.file_bytes;
    fs::hard_link(&candidate, &plan.output)?;
    let report = json!({"schema":"ojon.diffusion_gguf_conversion.v1",
        "backend":plan.converter.name(),"status":"FORMAT_VALID_NOT_RUNTIME_TESTED",
        "source":plan.source.to_string_lossy(),"output":plan.output.to_string_lossy(),
        "kind":plan.kind.name(),"input_layout":plan.layout.name(),"precision":plan.format,
        "source_bytes":plan.source_bytes,"output_bytes":output_bytes,
        "size_saving_pct":100.0*(1.0-output_bytes as f64/plan.source_bytes as f64),
        "tensor_count":info.tensor_count,"requested_quantized_tensor_count":requested_tensor_count,
        "requested_ggml_storage_type":requested_storage_type(&plan.format)?,
        "gguf_architecture":info.architecture,
        "quantization_semantics":"weight_only; NOT W4A4",
        "elapsed_seconds":start.elapsed().as_secs_f64(),
        "gguf_structure_checked":true,"requested_quantization_present":true,
        "runtime_load_verified":false,"comfyui_gguf_runtime_verified":false,
        "peak_vram_measured":false,"image_video_quality_verified":false,
        "warning":"GGUF contains requested weight quantization but activation quantization is not W4A4. ComfyUI-GGUF compatibility and SD.cpp generation need independent real-load tests and matched-seed comparison."});
    if let Err(e) = atomic_receipt(&plan.output, &report) {
        let _ = fs::remove_file(&plan.output);
        return Err(e);
    }
    log("[ojon] GGUF container validated and published; actual generation still untested");
    Ok(report)
}
fn component_flag(name: &str) -> Option<&'static str> {
    match name {
        "vae" => Some("--vae"),
        "audio-vae" => Some("--audio-vae"),
        "clip_l" => Some("--clip_l"),
        "clip_g" => Some("--clip_g"),
        "t5xxl" => Some("--t5xxl"),
        "llm" => Some("--llm"),
        "llm_vision" => Some("--llm_vision"),
        "high-noise-diffusion-model" => Some("--high-noise-diffusion-model"),
        _ => None,
    }
}
#[derive(Debug)]
pub struct GgufSmokePlan {
    pub model: PathBuf,
    pub output: PathBuf,
    pub tool: PathBuf,
    pub kind: MediaKind,
    pub layout: InputLayout,
    pub components: Vec<(String, PathBuf)>,
    pub prompt: String,
    pub seed: i64,
    pub steps: u32,
    pub width: u32,
    pub height: u32,
    pub frames: u32,
    pub fps: u32,
    pub offload: bool,
    pub backend: Option<String>,
}
#[allow(clippy::too_many_arguments)]
pub fn prepare_gguf_smoke(
    model: &Path,
    output: &Path,
    tool: &Path,
    kind: MediaKind,
    layout: InputLayout,
    components: &[(String, PathBuf)],
    prompt: &str,
    seed: i64,
    steps: u32,
    width: u32,
    height: u32,
    frames: u32,
    fps: u32,
    offload: bool,
    backend: Option<&str>,
) -> io::Result<GgufSmokePlan> {
    if prompt.is_empty() || prompt.len() > 4096 {
        return Err(bad("prompt must be 1..4096 bytes"));
    }
    if !(1..=100).contains(&steps)
        || width < 128
        || height < 128
        || width > 1536
        || height > 1536
        || !width.is_multiple_of(64)
        || !height.is_multiple_of(64)
    {
        return Err(bad(
            "steps 1..100, width/height 128..1536 and multiples of 64 required",
        ));
    }
    if seed < 0 {
        return Err(bad("fixed nonnegative seed required for reproducibility"));
    }
    if kind == MediaKind::Video && (!(2..=257).contains(&frames) || !(1..=60).contains(&fps)) {
        return Err(bad("video requires 2..257 frames and 1..60 FPS"));
    }
    let model = model.canonicalize()?;
    match model.extension().and_then(|s| s.to_str()) {
        Some("gguf") => {
            let info = inspect_gguf(&model)?;
            if info.tensor_count == 0 {
                return Err(bad("model GGUF has no tensors"));
            }
        }
        Some("safetensors") => {
            let info = probe_model(&model)?;
            if info.format != "safetensors_single_file" || info.tensor_count == Some(0) {
                return Err(bad("baseline source lacks valid safetensors metadata"));
            }
        }
        _ => {
            return Err(bad(
                "runtime smoke requires standalone GGUF or safetensors model",
            ))
        }
    }
    if kind == MediaKind::Video && layout == InputLayout::Checkpoint {
        return Err(bad("video smoke requires a standalone diffusion denoiser"));
    }
    if let Some(name) = backend {
        if name.is_empty()
            || name.len() > 150
            || !name
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'=' | b',' | b'&'))
        {
            return Err(bad("invalid sd-cli backend assignment"));
        }
    }
    let output = target(
        output,
        if kind == MediaKind::Image {
            "png"
        } else {
            "webm"
        },
    )?;
    if model == output {
        return Err(bad("smoke output cannot overwrite model"));
    }
    let mut verified = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (name, path) in components {
        if component_flag(name).is_none() || !seen.insert(name) {
            return Err(bad("invalid or duplicate conditioning component name"));
        }
        let path = path.canonicalize()?;
        if !path.is_file() {
            return Err(bad("conditioning component must be a local file"));
        }
        verified.push((name.clone(), path));
    }
    Ok(GgufSmokePlan {
        model,
        output,
        tool: checked_tool(tool)?,
        kind,
        layout,
        components: verified,
        prompt: prompt.into(),
        seed,
        steps,
        width,
        height,
        frames,
        fps,
        offload,
        backend: backend.map(str::to_owned),
    })
}
impl GgufSmokePlan {
    pub fn json(&self) -> Value {
        json!({"schema":"ojon.diffusion_gguf_smoke_plan.v1","status":"PLAN_ONLY",
            "backend":"stable-diffusion.cpp","model":self.model.to_string_lossy(),
            "kind":self.kind.name(),"input_layout":self.layout.name(),"output":self.output.to_string_lossy(),
            "tool":self.tool.to_string_lossy(),"components":self.components.iter()
                .map(|(k,v)|json!({"name":k,"path":v.to_string_lossy()})).collect::<Vec<_>>(),
            "seed":self.seed,"prompt":self.prompt,"steps":self.steps,
            "width":self.width,"height":self.height,
            "frames":if self.kind==MediaKind::Video{Some(self.frames)}else{None::<u32>},
            "fps":if self.kind==MediaKind::Video{Some(self.fps)}else{None::<u32>},
            "offload_to_cpu":self.offload,"runtime_backend":self.backend,"model_loaded":false,
            "warning":"Smoke only proves the installed sd-cli produced a structurally recognizable media file; compare visual output with a matched-seed original before quality approval."})
    }
    fn command(&self, dest: &Path) -> Command {
        let mut cmd = Command::new(&self.tool);
        cmd.arg("-M")
            .arg(if self.kind == MediaKind::Image {
                "img_gen"
            } else {
                "vid_gen"
            })
            .arg(self.layout.flag())
            .arg(&self.model)
            .arg("-o")
            .arg(dest)
            .arg("-p")
            .arg(&self.prompt)
            .arg("-s")
            .arg(self.seed.to_string())
            .arg("--steps")
            .arg(self.steps.to_string())
            .arg("-W")
            .arg(self.width.to_string())
            .arg("-H")
            .arg(self.height.to_string());
        for (name, path) in &self.components {
            if let Some(flag) = component_flag(name) {
                cmd.arg(flag).arg(path);
            }
        }
        if self.kind == MediaKind::Video {
            cmd.arg("--video-frames")
                .arg(self.frames.to_string())
                .arg("--fps")
                .arg(self.fps.to_string());
        }
        if self.offload {
            cmd.arg("--offload-to-cpu");
        }
        if let Some(name) = &self.backend {
            cmd.arg("--backend").arg(name);
        }
        cmd
    }
}
fn verify_media(path: &Path, kind: MediaKind) -> io::Result<u64> {
    let mut f = File::open(path)?;
    let size = f.metadata()?.len();
    if size < 32 {
        return Err(bad("runtime produced missing or empty media output"));
    }
    let mut prefix = [0u8; 8];
    f.read_exact(&mut prefix)?;
    let valid = match kind {
        MediaKind::Image => prefix == [137, 80, 78, 71, 13, 10, 26, 10],
        MediaKind::Video => prefix[..4] == [0x1a, 0x45, 0xdf, 0xa3],
    };
    if !valid {
        return Err(bad(
            "sd-cli did not write a recognizable PNG/WebM media file",
        ));
    }
    Ok(size)
}
pub fn run_gguf_smoke(plan: &GgufSmokePlan, mut log: impl FnMut(&str)) -> io::Result<Value> {
    if plan.output.symlink_metadata().is_ok()
        || receipt_path(&plan.output).symlink_metadata().is_ok()
    {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "smoke output or receipt already exists",
        ));
    }
    let stage = stage(
        plan.output
            .parent()
            .ok_or_else(|| bad("missing output parent"))?,
    )?;
    let preview = stage.0.join(if plan.kind == MediaKind::Image {
        "preview.png"
    } else {
        "preview.webm"
    });
    let start = Instant::now();
    log("[ojon] real diffusion generation is starting; this can load large model weights");
    run_with_log(&mut plan.command(&preview), |line| log(line))?;
    let bytes = verify_media(&preview, plan.kind)?;
    fs::hard_link(&preview, &plan.output)?;
    let report = json!({"schema":"ojon.diffusion_gguf_smoke_result.v1",
        "status":"RUNTIME_GENERATION_COMPLETED_NOT_QUALITY_VERIFIED",
        "backend":"stable-diffusion.cpp","kind":plan.kind.name(),
        "input_layout":plan.layout.name(),"runtime_backend":plan.backend,
        "model":plan.model.to_string_lossy(),"output":plan.output.to_string_lossy(),
        "output_bytes":bytes,"elapsed_seconds":start.elapsed().as_secs_f64(),
        "seed":plan.seed,"prompt":plan.prompt,"steps":plan.steps,
        "width":plan.width,"height":plan.height,
        "frames":if plan.kind==MediaKind::Video{Some(plan.frames)}else{None::<u32>},
        "fps":if plan.kind==MediaKind::Video{Some(plan.fps)}else{None::<u32>},
        "conditioning_components":plan.components.iter().map(|(k,v)|
            json!({"name":k,"path":v.to_string_lossy()})).collect::<Vec<_>>(),
        "runtime_generation_verified":true,
        "file_signature_checked":true,
        "peak_vram_measured":false,"matched_seed_quality_verified":false,
        "visual_quality_verified":false,
        "warning":"A successful generated preview is not evidence of no visual degradation; compare baseline and quantized output with matching sampler, prompts and seed."});
    if let Err(e) = atomic_receipt(&plan.output, &report) {
        let _ = fs::remove_file(&plan.output);
        return Err(e);
    }
    log("[ojon] real media file created and validated; visual fidelity still NOT VERIFIED");
    Ok(report)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_wrong_formats_without_running_tools() {
        assert!(precision("nvfp4").is_err());
        assert!(MediaKind::parse("audio").is_err());
        assert!(InputLayout::parse("random").is_err());
        assert!(prepare_gguf_convert(
            Path::new("does_not_exist.safetensors"),
            Path::new("out.gguf"),
            Path::new("sd-cli"),
            MediaKind::Image,
            InputLayout::Diffusion,
            "q4_0"
        )
        .is_err());
    }
    #[test]
    fn dense_input_allowed_but_prepacked_w4a4_rejected() {
        fn make(path: &Path, packed: bool) {
            let mut map = serde_json::Map::new();
            map.insert(
                "transformer.blocks.0.attn.weight".into(),
                json!({"dtype":"F16","shape":[2,2],"data_offsets":[0,8]}),
            );
            if packed {
                map.insert(
                    "transformer.blocks.0.attn.qweight".into(),
                    json!({"dtype":"I8","shape":[2,2],"data_offsets":[8,12]}),
                );
            }
            let header = serde_json::to_vec(&Value::Object(map)).unwrap();
            let mut file = File::create(path).unwrap();
            file.write_all(&(header.len() as u64).to_le_bytes())
                .unwrap();
            file.write_all(&header).unwrap();
            file.write_all(&[0u8; 12]).unwrap();
        }
        let root =
            std::env::temp_dir().join(format!("ojon-diff-gguf-dense-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let dense = root.join("dense.safetensors");
        let packed = root.join("packed.safetensors");
        make(&dense, false);
        make(&packed, true);
        assert!(verify_dense_media_source(&dense).is_ok());
        assert!(verify_dense_media_source(&packed).is_err());
        let _ = fs::remove_dir_all(root);
    }
    #[test]
    fn requested_quantization_is_checked_in_tensor_catalog() {
        let make = |name: &str, kind: u32| GgufTensor {
            name: name.into(),
            elements: 4096,
            shape: vec![64, 64],
            storage_type: kind,
            relative_offset: 0,
        };
        let mixed = vec![make("w1", 2), make("norm", 1), make("w2", 2)];
        assert_eq!(count_requested_tensors(&mixed, "q4_0", &GgufConverter::SdCli).unwrap(), 2);
        assert!(count_requested_tensors(&mixed, "q8_0", &GgufConverter::SdCli).is_err());
        assert_eq!(requested_storage_type("q4_K").unwrap(), 12);
        assert_eq!(requested_storage_type("q3_K").unwrap(), 11);
        assert!(requested_storage_type("w4a4").is_err());
    }
    #[test]
    fn verifies_real_media_headers_not_just_extensions() {
        let dir = std::env::temp_dir().join(format!("ojon-media-signature-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("test.png");
        fs::write(&file, vec![0u8; 128]).unwrap();
        assert!(verify_media(&file, MediaKind::Image).is_err());
        let mut pixels = vec![0u8; 128];
        pixels[..8].copy_from_slice(&[137, 80, 78, 71, 13, 10, 26, 10]);
        fs::write(&file, &pixels).unwrap();
        assert!(verify_media(&file, MediaKind::Image).is_ok());
        pixels[..4].copy_from_slice(&[0x1a, 0x45, 0xdf, 0xa3]);
        fs::write(&file, &pixels).unwrap();
        assert!(verify_media(&file, MediaKind::Video).is_ok());
        fs::remove_dir_all(dir).unwrap();
    }
}

#[cfg(test)]
mod converter_tests {
    use super::*;

    #[test]
    fn comfy_converter_renders_streamed_quant_type_command() {
        let plan = GgufConvertPlan {
            source: PathBuf::from("fibo2_dev_bf16.safetensors"),
            output: PathBuf::from("out/fibo2_dev-Q5_K_M.gguf"),
            tool: PathBuf::from("tools/convert.py"),
            layout: InputLayout::Diffusion,
            kind: MediaKind::Image,
            format: "Q5_K_M".into(),
            source_bytes: 1,
            converter: GgufConverter::ComfyGguf { python: PathBuf::from("python") },
        };
        let args = plan.command_args(Path::new("stage/candidate.gguf"));
        assert_eq!(args[0], "tools/convert.py");
        assert!(args.windows(2).any(|w| w[0] == "--src" && w[1] == "fibo2_dev_bf16.safetensors"));
        assert!(args.windows(2).any(|w| w[0] == "--dst" && w[1] == "stage/candidate.gguf"));
        assert!(args.contains(&"--streamed".to_string()));
        assert!(args.windows(2).any(|w| w[0] == "--quant-type" && w[1] == "Q5_K_M"));
        assert_eq!(plan.json()["backend"], "comfyui-gguf convert.py");
        let sd = GgufConvertPlan { converter: GgufConverter::SdCli, tool: PathBuf::from("sd-cli"), format: "q5_0".into(), ..plan };
        let args = sd.command_args(Path::new("c.gguf"));
        assert_eq!(args[..2], ["-M".to_string(), "convert".to_string()]);
        assert!(args.windows(2).any(|w| w[0] == "--type" && w[1] == "q5_0"));
    }

    #[test]
    fn converter_formats_and_storage_types_are_per_converter() {
        let comfy = GgufConverter::parse("comfy-gguf", Path::new("python")).unwrap();
        assert_eq!(comfy.storage_type("Q4_K_M").unwrap(), 12);
        assert_eq!(comfy.storage_type("Q5_K_S").unwrap(), 13);
        assert_eq!(comfy.storage_type("Q6_K").unwrap(), 14);
        assert!(comfy.storage_type("q4_K").is_err(), "sd-cli spelling is not a comfy format");
        assert!(GgufConverter::SdCli.storage_type("Q4_K_M").is_err());
        assert_eq!(GgufConverter::SdCli.storage_type("q3_K").unwrap(), 11);
        assert!(comfy.nominal_bits("Q8_0").unwrap() > comfy.nominal_bits("Q4_0").unwrap());
        assert!(GgufConverter::parse("gguf-magic", Path::new("python")).is_err());
    }
}
