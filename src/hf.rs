//! Local Hugging Face checkpoint -> GGUF -> quantized GGUF adapter.
//! Delegates all tensor math to installed llama.cpp scripts and binaries.
//! Large intermediates may require significant RAM and free disk space.

use crate::gguf::{inspect_gguf, GgufSummary};
use crate::llama::{prepare_gguf, quantize_gguf_with_log, Precision};
use crate::jobs::run_with_log;
use std::ffi::OsString;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[derive(Clone, Debug)]
pub struct HfFolderInfo {
    pub directory: PathBuf,
    pub source_weight_bytes: u64,
    pub weight_files: usize,
}
pub fn inspect_hf_folder(folder: &Path) -> io::Result<HfFolderInfo> {
    let directory = folder.canonicalize()?;
    if !directory.is_dir() { return Err(invalid("Hugging Face input must be a directory")); }
    if !directory.join("config.json").is_file() {
        return Err(invalid("missing config.json in model folder"));
    }
    let mut source_weight_bytes = 0u64;
    let mut weight_files = 0usize;
    for entry in fs::read_dir(&directory)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if !meta.is_file() { continue; }
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".safetensors")
            || name == "pytorch_model.bin"
            || (name.starts_with("pytorch_model-") && name.ends_with(".bin"))
        {
            weight_files += 1;
            source_weight_bytes = source_weight_bytes.checked_add(meta.len())
                .ok_or_else(|| invalid("source weight size overflow"))?;
        }
    }
    if weight_files == 0 {
        return Err(invalid("no local safetensors or pytorch_model*.bin weight files"));
    }
    Ok(HfFolderInfo { directory, source_weight_bytes, weight_files })
}

#[derive(Clone, Debug)]
pub struct HfQuantizePlan {
    pub source: HfFolderInfo,
    pub output: PathBuf,
    pub converter_script: PathBuf,
    pub python: PathBuf,
    pub quantizer: PathBuf,
    pub precision: Precision,
    pub threads: usize,
}
fn resolve_script(path: &Path) -> io::Result<PathBuf> {
    let candidate = if path.is_dir() { path.join("convert_hf_to_gguf.py") } else { path.to_path_buf() };
    let path = candidate.canonicalize()?;
    if !path.is_file() || path.file_name().and_then(|s| s.to_str()) != Some("convert_hf_to_gguf.py") {
        return Err(invalid("provide the official convert_hf_to_gguf.py script or its containing directory"));
    }
    Ok(path)
}
pub fn prepare_hf(
    source: &Path, output: &Path, converter_script: &Path,
    python: &Path, quantizer: &Path, precision: Precision, threads: usize,
) -> io::Result<HfQuantizePlan> {
    if threads == 0 { return Err(invalid("thread count must be positive")); }
    let source = inspect_hf_folder(source)?;
    if !output.extension().and_then(|s| s.to_str()).is_some_and(|s| s.eq_ignore_ascii_case("gguf")) {
        return Err(invalid("output must end in .gguf"));
    }
    let name = output.file_name().ok_or_else(|| invalid("missing output file name"))?;
    let parent = output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let parent = parent.canonicalize()?;
    let output = parent.join(name);
    if output.symlink_metadata().is_ok() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, "output exists; never overwrite it"));
    }
    if output.starts_with(&source.directory) {
        return Err(invalid("output must be outside the source model directory"));
    }
    Ok(HfQuantizePlan {
        source, output, converter_script: resolve_script(converter_script)?,
        python: python.to_path_buf(), quantizer: quantizer.to_path_buf(), precision, threads,
    })
}

struct Stage(PathBuf);
impl Drop for Stage { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
fn create_stage(parent: &Path) -> io::Result<Stage> {
    for retry in 0..8 {
        let nano = SystemTime::now().duration_since(UNIX_EPOCH)
            .map_err(|_| invalid("invalid clock"))?.as_nanos();
        let path = parent.join(format!(".ojon-hf-{}-{nano}-{retry}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(Stage(path)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(invalid("unable to create staging directory"))
}
#[derive(Debug)]
pub struct HfQuantizeResult {
    pub output: PathBuf,
    pub info: GgufSummary,
    pub receipt: Option<PathBuf>,
}
pub fn quantize_hf(plan: &HfQuantizePlan) -> io::Result<HfQuantizeResult> {
    quantize_hf_with_log(plan, |line| eprintln!("{line}"))
}

/// Convert a compatible local HF folder and stream the real subprocess logs.
pub fn quantize_hf_with_log<F>(plan: &HfQuantizePlan, mut on_log: F) -> io::Result<HfQuantizeResult>
where F: FnMut(&str) {
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, "output exists"));
    }
    let parent = plan.output.parent().ok_or_else(|| invalid("missing output parent"))?;
    let stage = create_stage(parent)?;
    let intermediate = stage.0.join("intermediate.gguf");
    let reduced = stage.0.join("quantized.gguf");
    eprintln!("Ojon: running official HF -> GGUF converter. Memory and temporary disk use may be large.");
    let mut convert = Command::new(&plan.python);
    convert.arg(&plan.converter_script).arg(&plan.source.directory)
        .arg("--outfile").arg(&intermediate).arg("--outtype").arg("auto");
    run_with_log(&mut convert, |line| on_log(line))?;
    let interim = inspect_gguf(&intermediate)?;
    if interim.tensor_count == 0 || interim.split_count.unwrap_or(1) > 1 {
        return Err(invalid("unsupported GGUF intermediate (empty or split)"));
    }
    eprintln!("Ojon: running installed GGUF quantizer.");
    let qplan = prepare_gguf(&intermediate, &reduced, &plan.quantizer, plan.precision, plan.threads)?;
    let result = quantize_gguf_with_log(&qplan, |line| on_log(line))?;
    fs::hard_link(&result.output, &plan.output)?;
    let receipt = write_receipt(plan, &result.output_info).ok();
    Ok(HfQuantizeResult { output: plan.output.clone(), info: result.output_info, receipt })
}
fn write_receipt(plan: &HfQuantizePlan, info: &GgufSummary) -> io::Result<PathBuf> {
    let mut path_os: OsString = plan.output.as_os_str().to_os_string();
    path_os.push(".ojon-receipt.txt");
    let path = PathBuf::from(path_os);
    let mut file = OpenOptions::new().create_new(true).write(true).open(&path)?;
    let stamp = SystemTime::now().duration_since(UNIX_EPOCH)
        .map_err(|_| invalid("invalid clock"))?.as_secs();
    writeln!(file, "Ojon HF/GGUF quantization receipt v1")?;
    writeln!(file, "created_utc_unix={stamp}")?;
    writeln!(file, "source={}", plan.source.directory.display())?;
    writeln!(file, "source_weight_bytes={}", plan.source.source_weight_bytes)?;
    writeln!(file, "source_weight_files={}", plan.source.weight_files)?;
    writeln!(file, "converter_script={}", plan.converter_script.display())?;
    writeln!(file, "quantizer={}", plan.quantizer.display())?;
    writeln!(file, "precision={}", plan.precision)?;
    writeln!(file, "output={}", plan.output.display())?;
    writeln!(file, "output_bytes={}", info.file_bytes)?;
    writeln!(file, "output_tensors={}", info.tensor_count)?;
    writeln!(file, "verification=GGUF_METADATA_AND_TENSOR_CATALOG_ONLY")?;
    writeln!(file, "runtime_load_test=NOT_RUN")?;
    file.flush()?;
    Ok(path)
}
