//! Reproducible, local GGUF candidate sweep. The launcher never benchmarks
//! quality or invents a "winner"; only external task tests can establish it.
//!
//! Default is PLAN ONLY. Executing converters requires an explicit confirmation.
//! Each candidate uses the validated llama.cpp adapter, its safe staging policy,
//! and produces a receipt. Pre-existing destination files are NEVER overwritten.

use crate::gguf::inspect_gguf;
use crate::llama::{prepare_gguf, quantize_gguf_with_log, Precision};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, s.into())
}

#[derive(Debug)]
pub struct SweepPlan {
    pub source: PathBuf,
    pub destination: PathBuf,
    pub tool: PathBuf,
    pub imatrix: Option<PathBuf>,
    pub threads: usize,
    pub precisions: Vec<Precision>,
    pub source_bytes: u64,
    pub source_format: Option<u32>,
    pub tensor_count: u64,
}
impl SweepPlan {
    fn filename(&self, precision: Precision) -> String {
        let stem = self
            .source
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy();
        format!("{stem}-{}.gguf", precision.as_str())
    }
    pub fn output_for(&self, precision: Precision) -> PathBuf {
        self.destination.join(self.filename(precision))
    }
    pub fn json(&self) -> Value {
        json!({
            "schema": "ojon.sweep_plan.v1",
            "source": self.source.to_string_lossy(),
            "original_source_bytes": self.source_bytes,
            "source_general_file_type": self.source_format,
            "tensors": self.tensor_count,
            "tool": self.tool.to_string_lossy(),
            "threads": self.threads,
            "imatrix": self.imatrix.as_ref().map(|p|p.to_string_lossy().to_string()),
            "candidate_formats":self.precisions.iter().map(|p|p.as_str()).collect::<Vec<_>>(),
            "outputs":self.precisions.iter().map(|p|json!({
                "format":p.as_str(),
                "path":self.output_for(*p).to_string_lossy(),
                "status":"not_started",
                "estimated_memory":"unknown",
                "estimated_disk":"unknown",
                "quality":"not_evaluated"
            })).collect::<Vec<_>>(),
            "output_kind":"GGUF",
            "source_safety":"source is not changed; no output overwritten",
            "execution":"plan_only",
            "recommendation":"not_available_without_measured_quality",
        })
    }
}

pub fn parse_precisions(spec: &str) -> io::Result<Vec<Precision>> {
    use std::str::FromStr;
    if spec.len() > 512 {
        return Err(invalid("precision list is too long"));
    }
    let mut methods = Vec::new();
    let mut seen = HashSet::new();
    for name in spec.split(',') {
        let name = name.trim();
        if name.is_empty() {
            return Err(invalid("empty precision in list"));
        }
        let precision = Precision::from_str(name).map_err(invalid)?;
        if !seen.insert(precision.as_str()) {
            return Err(invalid(format!("duplicate precision: {name}")));
        }
        methods.push(precision);
    }
    if methods.is_empty() || methods.len() > 12 {
        return Err(invalid("sweep must have 1..=12 candidate precisions"));
    }
    Ok(methods)
}

/// Refuses to sweep a compressed source. 0 = F32, 1 = F16, 32 = BF16 in
/// llama.cpp GGUF general.file_type enumeration. A custom GGUF missing that
/// metadata should use the standalone direct quantizer with manual review.
pub fn prepare_sweep(
    source: &Path,
    destination: &Path,
    tool: &Path,
    methods: &[Precision],
    imatrix: Option<&Path>,
    threads: usize,
) -> io::Result<SweepPlan> {
    if threads == 0 {
        return Err(invalid("threads must be positive"));
    }
    if methods.is_empty() || methods.len() > 12 {
        return Err(invalid("sweep requires 1..=12 precisions"));
    }
    let mut seen = HashSet::new();
    for method in methods {
        if !seen.insert(method.as_str()) {
            return Err(invalid("duplicate formats waste space and are not allowed"));
        }
    }
    let source = source.canonicalize()?;
    let source_info = inspect_gguf(&source)?;
    if source_info.tensor_count == 0 {
        return Err(invalid("source GGUF has no tensors"));
    }
    if source_info.split_count.unwrap_or(1) > 1 {
        return Err(invalid("split GGUF source unsupported by this sweep"));
    }
    if !matches!(source_info.file_type, Some(0 | 1 | 32)) {
        return Err(invalid(
            "quality-first sweep requires an original F32/F16/BF16 GGUF with general.file_type; \
             do not repeatedly quantize an already compressed model",
        ));
    }
    if !destination.is_dir() {
        return Err(invalid("output directory must exist"));
    }
    let destination = destination.canonicalize()?;
    if source.parent() == Some(destination.as_path()) {
        // Same parent is valid as long as each target name is different.
        // Never change the original file; output names remain explicit.
    }
    let imatrix = match imatrix {
        Some(path) => {
            let p = path.canonicalize()?;
            if !p.is_file() || p.metadata()?.len() == 0 {
                return Err(invalid("importance matrix must be a nonempty local file"));
            }
            Some(p)
        }
        None => None,
    };
    if methods.iter().any(|p| {
        matches!(
            p,
            Precision::IQ2M | Precision::IQ2S | Precision::IQ2XS | Precision::IQ2XXS
        )
    }) && imatrix.is_none()
    {
        return Err(invalid(
            "ultra-low IQ2 variants require an imatrix for this quality-first sweep",
        ));
    }
    let plan = SweepPlan {
        source: source.clone(),
        destination,
        tool: tool.to_path_buf(),
        imatrix,
        threads,
        precisions: methods.to_vec(),
        source_bytes: source_info.file_bytes,
        source_format: source_info.file_type,
        tensor_count: source_info.tensor_count,
    };
    for method in &plan.precisions {
        let output = plan.output_for(*method);
        if output == source {
            return Err(invalid("sweep output collides with source"));
        }
        if output.symlink_metadata().is_ok() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("sweep destination already exists: {}", output.display()),
            ));
        }
    }
    Ok(plan)
}

fn unique_manifest(destination: &Path) -> io::Result<(PathBuf, std::fs::File)> {
    for index in 0..8u32 {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| invalid("system clock error"))?
            .as_nanos();
        let path = destination.join(format!(
            "ojon-sweep-{}-{nanos}-{index}.json",
            std::process::id()
        ));
        match OpenOptions::new().create_new(true).write(true).open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(invalid("cannot allocate a unique manifest"))
}

/// Runs candidates sequentially to avoid concurrent large VRAM/RAM pressure.
/// Each failed candidate is recorded; the original remains intact.
pub fn execute_sweep<F>(plan: &SweepPlan, mut log: F) -> io::Result<PathBuf>
where
    F: FnMut(&str),
{
    let (manifest_path, mut manifest_file) = unique_manifest(&plan.destination)?;
    let mut rows = Vec::new();
    for precision in &plan.precisions {
        let path = plan.output_for(*precision);
        log(&format!(
            "[ojon] starting {} -> {}",
            precision,
            path.display()
        ));
        let started = Instant::now();
        let result = (|| {
            let mut checked =
                prepare_gguf(&plan.source, &path, &plan.tool, *precision, plan.threads)?;
            if let Some(imatrix) = &plan.imatrix {
                checked = checked.with_imatrix(imatrix)?;
            }
            quantize_gguf_with_log(&checked, |line| log(line))
        })();
        let elapsed = started.elapsed().as_secs_f64();
        match result {
            Ok(done) => {
                let ratio = done.output_info.file_bytes as f64 / plan.source_bytes as f64;
                log(&format!(
                    "[ojon] {} done: {:.2}% source size, {:.1}s; quality not tested",
                    precision,
                    100.0 * ratio,
                    elapsed
                ));
                rows.push(json!({
                    "format":precision.as_str(),"status":"converted_catalog_verified",
                    "output":done.output.to_string_lossy(),
                    "output_bytes":done.output_info.file_bytes,
                    "ratio_to_source":ratio,
                    "elapsed_seconds":elapsed,
                    "runtime_load_verified":false,"task_quality_evaluated":false,
                }));
            }
            Err(e) => {
                let error = e.to_string();
                log(&format!("[ojon] {} failed: {error}", precision));
                rows.push(json!({
                    "format":precision.as_str(),"status":"failed",
                    "output":path.to_string_lossy(),
                    "error":error,"elapsed_seconds":elapsed,
                    "runtime_load_verified":false,"task_quality_evaluated":false
                }));
            }
        }
    }
    let finished = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| invalid("invalid clock"))?
        .as_secs();
    let report = json!({
        "schema":"ojon.sweep_result.v1",
        "timestamp_unix":finished,
        "plan":plan.json(),
        "candidates":rows,
        "quality_recommendation":Value::Null,
        "quality_status":"NO_TASK_EVALUATION",
        "warning":"A smaller GGUF is not necessarily a better model. Run baseline + candidate quality and loader tests before a recommendation. No results were simulated.",
    });
    serde_json::to_writer_pretty(&mut manifest_file, &report)
        .map_err(|e| io::Error::other(format!("manifest JSON error: {e}")))?;
    writeln!(manifest_file)?;
    manifest_file.flush()?;
    Ok(manifest_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn accepts_bounded_nonduplicate_precision_list() {
        let formats = parse_precisions("Q5_K_M,Q4_K_M,IQ4_XS,IQ3_M").unwrap();
        assert_eq!(formats.len(), 4);
        assert!(parse_precisions("Q4_K_M,Q4_K_M").is_err());
        assert!(parse_precisions("Q8_0").is_err());
        assert!(parse_precisions("Q5_K_M,").is_err());
    }
}
