//! Public llama.cpp GGUF quantizer adapter.
//!
//! This module calls a separately installed llama-quantize executable.
//! It implements NO quantization math. Ojon keeps original weights untouched,
//! uses a private staging directory, validates the output catalog, and publishes
//! the resulting artifact only if the converter succeeds.

use crate::gguf::{inspect_gguf, GgufSummary};
use crate::resource::{assess_resources, ResourceMode};
use crate::jobs::run_with_log;
use std::ffi::OsString;
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::str::FromStr;
use std::time::{SystemTime, UNIX_EPOCH};

fn bad(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Precision {
    Q3KM,
    Q4KM,
    Q5KM,
    Q6K,
    IQ4XS,
    IQ3M,
    IQ3S,
    IQ3XS,
    IQ2M,
    IQ2S,
    IQ2XS,
    IQ2XXS,
}
impl Default for Precision {
    fn default() -> Self { Self::Q5KM }
}
impl Precision {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Q3KM => "Q3_K_M",
            Self::Q4KM => "Q4_K_M",
            Self::Q5KM => "Q5_K_M",
            Self::Q6K => "Q6_K",
            Self::IQ4XS => "IQ4_XS",
            Self::IQ3M => "IQ3_M",
            Self::IQ3S => "IQ3_S",
            Self::IQ3XS => "IQ3_XS",
            Self::IQ2M => "IQ2_M",
            Self::IQ2S => "IQ2_S",
            Self::IQ2XS => "IQ2_XS",
            Self::IQ2XXS => "IQ2_XXS",
        }
    }
}
impl fmt::Display for Precision {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(self.as_str()) }
}
impl FromStr for Precision {
    type Err = String;
    fn from_str(input: &str) -> Result<Self, Self::Err> {
        match input.to_ascii_uppercase().as_str() {
            "Q3_K_M" => Ok(Self::Q3KM),
            "Q4_K_M" => Ok(Self::Q4KM),
            "Q5_K_M" => Ok(Self::Q5KM),
            "Q6_K" => Ok(Self::Q6K),
            "IQ4_XS" => Ok(Self::IQ4XS),
            "IQ3_M" => Ok(Self::IQ3M),
            "IQ3_S" => Ok(Self::IQ3S),
            "IQ3_XS" => Ok(Self::IQ3XS),
            "IQ2_M" => Ok(Self::IQ2M),
            "IQ2_S" => Ok(Self::IQ2S),
            "IQ2_XS" => Ok(Self::IQ2XS),
            "IQ2_XXS" => Ok(Self::IQ2XXS),
            other => Err(format!(
                "unsupported GGUF precision: {other}; supported: Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, IQ4_XS, IQ3_M, IQ3_S, IQ3_XS, IQ2_M, IQ2_S, IQ2_XS, IQ2_XXS"
            )),
        }
    }
}

/// A checked recipe that the GUI or CLI can run.
/// This stage does NOT assert that the backend supports every GGUF architecture.
#[derive(Clone, Debug)]
pub struct QuantizePlan {
    pub source: PathBuf,
    pub output: PathBuf,
    pub executable: PathBuf,
    pub precision: Precision,
    pub threads: usize,
    pub imatrix: Option<PathBuf>,
    pub tensor_type_file: Option<PathBuf>,
    pub source_info: GgufSummary,
}
#[derive(Debug)]
pub struct QuantizeResult {
    pub output: PathBuf,
    pub output_info: GgufSummary,
    pub receipt: Option<PathBuf>,
}

/// Preflight read-only metadata and destination safety. A successful preflight
/// does not mean the converter is installed or this model family is supported.
pub fn prepare_gguf(
    source: &Path,
    output: &Path,
    executable: &Path,
    precision: Precision,
    threads: usize,
) -> io::Result<QuantizePlan> {
    if threads == 0 { return Err(bad("thread count must be greater than zero")); }
    if output.extension().and_then(|x| x.to_str()).map(|x| x.eq_ignore_ascii_case("gguf")) != Some(true) {
        return Err(bad("output filename must end in .gguf"));
    }
    if source.extension().and_then(|x| x.to_str()).map(|x| x.eq_ignore_ascii_case("gguf")) != Some(true) {
        return Err(bad("this adapter only supports a GGUF source; first use a compatible upstream model converter"));
    }
    let canonical_source = source.canonicalize()?;
    if !canonical_source.is_file() { return Err(bad("source must be a regular GGUF file")); }
    let source_info = inspect_gguf(&canonical_source)?;
    if source_info.tensor_count == 0 { return Err(bad("source GGUF has no tensors")); }
    validate_original_precision(source_info.file_type)?;
    if source_info.split_count.unwrap_or(1) > 1 {
        return Err(bad("this first adapter supports single-file GGUF only; multi-shard GGUF needs a dedicated recipe"));
    }
    let output_filename = output.file_name().ok_or_else(|| bad("missing output filename"))?;
    let parent = output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let canonical_parent = parent.canonicalize()?;
    if !canonical_parent.is_dir() { return Err(bad("output parent is not a directory")); }
    let canonical_output = canonical_parent.join(output_filename);
    if canonical_output == canonical_source {
        return Err(bad("cannot overwrite source weights"));
    }
    if canonical_output.symlink_metadata().is_ok() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, "output already exists; choose another path"));
    }
    Ok(QuantizePlan {
        source: canonical_source,
        output: canonical_output,
        executable: executable.to_path_buf(),
        precision,
        threads,
        imatrix: None,
        tensor_type_file: None,
        source_info,
    })
}

/// Prevent accidental double quantization; unknown GGUF precision fails closed.
/// GGUF general.file_type: 0=F32, 1=F16, 32=BF16.
pub(crate) fn validate_original_precision(file_type: Option<u32>) -> io::Result<()> {
    if !matches!(file_type, Some(0 | 1 | 32)) {
        return Err(bad("source must be original F32/F16/BF16 GGUF with known general.file_type; refusing lossy requantization"));
    }
    Ok(())
}

impl QuantizePlan {
    /// Optional importance matrix computed separately with the public
    /// llama-imatrix tool and representative calibration data. Never claims
    /// guaranteed quality; actual task evaluation remains mandatory.
    /// Apply explicit tensor-level bit assignments supported by llama-quantize.
    /// Must be a preexisting nonempty text file produced by a trusted planner.
    pub fn with_tensor_type_file(mut self, path: &Path) -> io::Result<Self> {
        let resolved = path.canonicalize()?;
        if !resolved.is_file() || !(1..=1024*1024).contains(&resolved.metadata()?.len()) {
            return Err(bad("tensor-type file must be 1 byte..1 MiB"));
        }
        self.tensor_type_file = Some(resolved);
        Ok(self)
    }
    pub fn with_imatrix(mut self, path: &Path) -> io::Result<Self> {
        let real = path.canonicalize()?;
        if !real.is_file() || real.metadata()?.len()==0 {
            return Err(bad("importance matrix file must exist and be nonempty"));
        }
        self.imatrix = Some(real);
        Ok(self)
    }
}

struct StagingDirectory(PathBuf);
impl Drop for StagingDirectory {
    fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); }
}

fn make_staging_directory(output: &Path) -> io::Result<StagingDirectory> {
    let parent = output.parent().ok_or_else(|| bad("missing output directory"))?;
    let stem = output.file_name().unwrap_or_default().to_string_lossy();
    for attempt in 0..8 {
        let now = SystemTime::now().duration_since(UNIX_EPOCH)
            .map_err(|_| bad("system time before Unix epoch"))?.as_nanos();
        let path = parent.join(format!(".{stem}.ojon-work-{}-{now}-{attempt}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(StagingDirectory(path)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(bad("could not create a unique staging directory"))
}

/// Run the configured quantizer as an argv-based subprocess (never through a shell).
/// This method is blocking; desktop callers MUST execute it on a worker thread.
pub fn quantize_gguf(plan: &QuantizePlan) -> io::Result<QuantizeResult> {
    quantize_gguf_with_log(plan, |line| eprintln!("{line}"))
}

/// Same safe conversion operation, forwarding actual converter log lines.
/// No progress percentages or VRAM values are inferred.
pub fn quantize_gguf_with_log<F>(plan: &QuantizePlan, mut on_log: F) -> io::Result<QuantizeResult>
where F: FnMut(&str) {
    // Recheck because output could have appeared between preflight and execution.
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, "output already exists"));
    }
    validate_original_precision(inspect_gguf(&plan.source)?.file_type)?;
    let parent = plan.output.parent().ok_or_else(|| bad("missing output directory"))?;
    let resources = assess_resources(plan.source_info.file_bytes, parent, ResourceMode::Gguf, "cpu")?;
    resources.enforce(false)?;
    let staging = make_staging_directory(&plan.output)?;
    let stage_file = staging.0.join("quantized.gguf");
    eprintln!("Ojon: launching configured llama.cpp quantizer; source weights are read-only.");
    let mut cmd = Command::new(&plan.executable);
    if let Some(matrix) = &plan.imatrix {
        cmd.arg("--imatrix").arg(matrix);
    }
    if let Some(overrides) = &plan.tensor_type_file {
        cmd.arg("--tensor-type-file").arg(overrides);
    }
    cmd.arg(&plan.source)
        .arg(&stage_file)
        .arg(plan.precision.as_str())
        .arg(plan.threads.to_string());
    run_with_log(&mut cmd, |line| on_log(line))?;
    // Container metadata + catalog checked; NOT a full tensor check or runtime test.
    let info = inspect_gguf(&stage_file)?;
    if info.tensor_count == 0 {
        return Err(bad("converter output contains no tensors"));
    }
    if info.architecture != plan.source_info.architecture || info.tensor_count != plan.source_info.tensor_count {
        return Err(bad("quantized output changed architecture or tensor count; refusing publication"));
    }
    if !matches!(info.file_type, Some(v) if !matches!(v, 0 | 1 | 32)) {
        return Err(bad("converter did not produce an identifiable quantized GGUF; refusing publication"));
    }
    if info.split_count.unwrap_or(1) > 1 {
        return Err(bad("converter emitted a multi-shard model; this adapter cannot publish all shards safely yet"));
    }

    // Stage and destination share a parent filesystem. Hard-linking publishes
    // with no replacement on either Windows or Linux; existing output fails.
    // The staging directory is cleaned by Drop, leaving only the final link.
    fs::hard_link(&stage_file, &plan.output)?;
    let receipt = write_receipt(plan, &info).ok();
    Ok(QuantizeResult {
        output: plan.output.clone(),
        output_info: info,
        receipt,
    })
}
fn write_receipt(plan: &QuantizePlan, info: &GgufSummary) -> io::Result<PathBuf> {
    let mut receipt_os: OsString = plan.output.as_os_str().to_os_string();
    receipt_os.push(".ojon-receipt.txt");
    let path = PathBuf::from(receipt_os);
    let mut file = OpenOptions::new().write(true).create_new(true).open(&path)?;
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)
        .map_err(|_| bad("system time before Unix epoch"))?.as_secs();
    writeln!(file, "Ojon GGUF conversion receipt v1")?;
    writeln!(file, "created_utc_unix={timestamp}")?;
    writeln!(file, "adapter=llama.cpp-quantize")?;
    writeln!(file, "tensor_type_file={}", plan.tensor_type_file.as_ref().map(|p|p.display().to_string()).unwrap_or_else(||"none".into()))?;
    writeln!(file, "tool={}", plan.executable.display())?;
    writeln!(file, "source={}", plan.source.display())?;
    writeln!(file, "source_bytes={}", plan.source_info.file_bytes)?;
    writeln!(file, "output={}", plan.output.display())?;
    writeln!(file, "output_bytes={}", info.file_bytes)?;
    writeln!(file, "precision={}", plan.precision)?;
    writeln!(file, "threads={}", plan.threads)?;
    writeln!(file, "imatrix={}", plan.imatrix.as_ref().map(|p|p.display().to_string()).unwrap_or_else(||"none".into()))?;
    writeln!(file, "output_tensors={}", info.tensor_count)?;
    writeln!(file, "verification=GGUF_METADATA_AND_TENSOR_CATALOG_ONLY")?;
    writeln!(file, "runtime_load_test=NOT_RUN")?;
    writeln!(file, "task_quality_test=NOT_RUN")?;
    file.flush()?;
    Ok(path)
}

#[cfg(test)]
mod precision_tests {
    use super::*;
    #[test]
    fn iq_and_k_precisions_round_trip() {
        for variant in [
            Precision::Q3KM, Precision::Q4KM, Precision::Q5KM, Precision::Q6K,
            Precision::IQ4XS, Precision::IQ3M, Precision::IQ3S, Precision::IQ3XS,
            Precision::IQ2M, Precision::IQ2S, Precision::IQ2XS, Precision::IQ2XXS,
        ] {
            assert_eq!(Precision::from_str(variant.as_str()).unwrap(), variant);
        }
        assert!(Precision::from_str("Q8_0").is_err());
    }
}

#[cfg(test)]
mod conversion_safety_tests {
    use super::*;
    #[test]
    fn originals_only() {
        for ty in [Some(0), Some(1), Some(32)] {
            assert!(validate_original_precision(ty).is_ok());
        }
        for ty in [None, Some(2), Some(7), Some(12), Some(40)] {
            assert!(validate_original_precision(ty).is_err());
        }
    }
}
