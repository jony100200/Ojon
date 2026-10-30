//! Generate a real, reusable GGUF importance matrix with public llama.cpp.
//! Calibration is independent of held-out task evaluation and does not make
//! a "quality preserved" claim. The original GGUF and corpus are read-only.

use crate::gguf::inspect_gguf;
use crate::jobs::run_with_log;
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, s.into())
}
#[derive(Debug)]
pub struct ImatrixPlan {
    pub model: PathBuf,
    pub corpus: PathBuf,
    pub output: PathBuf,
    pub tool: PathBuf,
    pub gpu_layers: u32,
    pub context: u32,
    pub model_bytes: u64,
    pub corpus_bytes: u64,
}
impl ImatrixPlan {
    pub fn json(&self) -> Value {
        json!({
            "schema":"ojon.imatrix_plan.v1",
            "model":self.model.to_string_lossy(),
            "model_bytes":self.model_bytes,
            "calibration_corpus":self.corpus.to_string_lossy(),
            "corpus_bytes":self.corpus_bytes,
            "output":self.output.to_string_lossy(),
            "tool":self.tool.to_string_lossy(),
            "gpu_layers":self.gpu_layers,
            "context":self.context,
            "format":"gguf-imatrix",
            "status":"PLAN_ONLY",
            "ram_vram_budget":"unknown_until_local_measurement",
            "quality":"not_evaluated",
            "requirement":"use a separate held-out corpus to evaluate actual task quality",
            "warning":"original model is loaded by the explicitly installed public llama-imatrix; GPU offloading only if its binary supports that backend"
        })
    }
}
pub fn prepare_imatrix(
    model: &Path,
    corpus: &Path,
    output: &Path,
    tool: &Path,
    gpu_layers: u32,
    context: u32,
) -> io::Result<ImatrixPlan> {
    if !(256..=32768).contains(&context) || context % 32 != 0 {
        return Err(invalid("context must be 256..32768 and divisible by 32"));
    }
    if gpu_layers > 512 {
        return Err(invalid("GPU offload layers must be 0..=512"));
    }
    if tool.as_os_str().is_empty() {
        return Err(invalid("llama-imatrix binary path required"));
    }
    if tool.is_absolute() && !tool.is_file() {
        return Err(invalid("selected llama-imatrix tool does not exist"));
    }
    let model = model.canonicalize()?;
    if !model.is_file() {
        return Err(invalid("original GGUF path must be a file"));
    }
    let m = inspect_gguf(&model)?;
    if m.tensor_count == 0 {
        return Err(invalid("empty GGUF tensor catalog"));
    }
    if m.split_count.unwrap_or(1) > 1 {
        return Err(invalid("split GGUF source unsupported"));
    }
    if !matches!(m.file_type, Some(0 | 1 | 32)) {
        return Err(invalid("importance calibration expects original F32, F16 or BF16 GGUF, not an already quantized model"));
    }
    let corpus = corpus.canonicalize()?;
    if !corpus.is_file() {
        return Err(invalid("calibration corpus must be an existing file"));
    }
    let corpus_bytes = fs::metadata(&corpus)?.len();
    if !(1024..=64 * 1024 * 1024).contains(&corpus_bytes) {
        return Err(invalid("calibration text file must be 1KiB to 64MiB"));
    }
    let mut sample = [0u8; 4096];
    let n = File::open(&corpus)?.read(&mut sample)?;
    // Require a UTF-8 text corpus rather than accidentally passing a GGUF or binary.
    if std::str::from_utf8(&sample[..n]).is_err() && n < 4096 {
        return Err(invalid("calibration must be UTF-8 text"));
    }
    if sample[..n].iter().any(|b| *b == 0) {
        return Err(invalid("calibration contains binary NUL bytes"));
    }
    if output.extension().and_then(|s| s.to_str()) != Some("gguf") {
        return Err(invalid("new importance matrix must use .gguf extension"));
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let output = parent.canonicalize()?.join(
        output
            .file_name()
            .ok_or_else(|| invalid("output filename required"))?,
    );
    if output == model || output == corpus || output.symlink_metadata().is_ok() {
        return Err(invalid(
            "imatrix output already exists or conflicts with input files",
        ));
    }
    Ok(ImatrixPlan {
        model,
        corpus,
        output,
        tool: tool.to_path_buf(),
        gpu_layers,
        context,
        model_bytes: m.file_bytes,
        corpus_bytes,
    })
}
struct Stage(PathBuf);
impl Drop for Stage {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn allocate_stage(parent: &Path) -> io::Result<Stage> {
    for n in 0..8 {
        let tick = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| invalid("system clock invalid"))?
            .as_nanos();
        let path = parent.join(format!(".ojon-imatrix-{}-{tick}-{n}", std::process::id()));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(Stage(path)),
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(invalid("cannot allocate calibration staging folder"))
}
pub fn run_imatrix<F: FnMut(&str)>(plan: &ImatrixPlan, mut log: F) -> io::Result<PathBuf> {
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "importance matrix output already exists",
        ));
    }
    let parent = plan
        .output
        .parent()
        .ok_or_else(|| invalid("output parent missing"))?;
    let stage = allocate_stage(parent)?;
    let tmp = stage.0.join("imatrix.gguf");
    let mut cmd = crate::jobs::tool_command(&plan.tool);
    cmd.arg("-m")
        .arg(&plan.model)
        .arg("-f")
        .arg(&plan.corpus)
        .arg("-o")
        .arg(&tmp)
        .arg("--output-format")
        .arg("gguf")
        .arg("--no-ppl")
        .arg("-ngl")
        .arg(plan.gpu_layers.to_string())
        .arg("-c")
        .arg(plan.context.to_string())
        .current_dir(&stage.0);
    log("[ojon] generating calibrated importance matrix with installed public llama-imatrix");
    run_with_log(&mut cmd, |line| log(line))?;
    let bytes = fs::metadata(&tmp)?.len();
    if bytes < 64 {
        return Err(io::Error::other(
            "importance matrix is empty or suspiciously small",
        ));
    }
    let mut magic = [0; 4];
    File::open(&tmp)?.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(io::Error::other(
            "upstream output is not a GGUF importance matrix",
        ));
    }
    // The llama.cpp imatrix file contains statistics rather than ordinary
    // model tensors, so do not try to prove model quality by parsing this file.
    fs::hard_link(&tmp, &plan.output)?;
    log(&format!(
        "[ojon] saved {} calibration bytes; task quality not evaluated",
        bytes
    ));
    Ok(plan.output.clone())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_invalid_context_before_backend_or_model_access() {
        assert!(prepare_imatrix(
            Path::new("missing.gguf"),
            Path::new("missing.txt"),
            Path::new("output.gguf"),
            Path::new("llama-imatrix"),
            0,
            0
        )
        .is_err());
        assert!(prepare_imatrix(
            Path::new("missing.gguf"),
            Path::new("missing.txt"),
            Path::new("output.gguf"),
            Path::new("llama-imatrix"),
            999,
            2048
        )
        .is_err());
    }
}
