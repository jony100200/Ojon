//! Optional real runtime smoke check using a user-installed llama-cli.
//! This does NOT quantify quality retention, check vision components, or
//! establish parity with an original model.

use crate::gguf::inspect_gguf;
use std::io::{self, ErrorKind};
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::Duration;

/// Executes a short inference run with the actual selected llama.cpp runtime.
/// The prompt and output are visible in the terminal; Ojon records only whether
/// the CLI process exited successfully. Never call this on a UI event thread.
pub fn smoke_gguf(
    model: &Path,
    runtime: &Path,
    prompt: &str,
    tokens: u32,
    gpu_layers: u32,
) -> io::Result<()> {
    if tokens == 0 || tokens > 512 {
        return Err(io::Error::new(ErrorKind::InvalidInput, "token count must be 1..=512"));
    }
    if prompt.is_empty() || prompt.len() > 4096 {
        return Err(io::Error::new(ErrorKind::InvalidInput, "prompt must be 1..=4096 bytes"));
    }
    let source = model.canonicalize()?;
    let summary = inspect_gguf(&source)?;
    if summary.tensor_count == 0 {
        return Err(io::Error::new(ErrorKind::InvalidData, "GGUF contains no tensors"));
    }
    eprintln!("Ojon: invoking real llama.cpp inference; loading may require substantial memory.");
    eprintln!("Ojon: CLI success is a smoke check, not a model-quality score.");
    let status = Command::new(runtime)
        .arg("-m").arg(&source)
        .arg("-p").arg(prompt)
        .arg("-n").arg(tokens.to_string())
        .arg("-ngl").arg(gpu_layers.to_string())
        .arg("-st")
        .stdin(Stdio::null())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()?;
    if !status.success() {
        return Err(io::Error::new(ErrorKind::Other, format!("llama-cli inference failed (status: {status})")));
    }
    Ok(())
}

/// Captured output of a short real inference run. Only an exit code is not a
/// quality metric; the user must inspect the generated answer.
#[derive(Debug)]
pub struct SmokeCapture {
    pub stdout: String,
    pub stderr: String,
}

/// Runs the installed CLI and captures its real output for the Kathamo Test pane.
/// Keep token count small: Command::output buffers the child's output in RAM.
pub fn smoke_gguf_capture(
    model:&Path,runtime:&Path,prompt:&str,tokens:u32,gpu_layers:u32
)->io::Result<SmokeCapture>{
    smoke_gguf_capture_with_device(model,runtime,prompt,tokens,gpu_layers,None)
}
/// Explicit device ID is passed straight to the selected llama-cli. A success
/// proves this specific backend/device ran, not free VRAM or model quality.
pub fn smoke_gguf_capture_with_device(
    model: &Path,
    runtime: &Path,
    prompt: &str,
    tokens: u32,
    gpu_layers: u32,
    device:Option<&str>,
) -> io::Result<SmokeCapture> {
    if !(1..=128).contains(&tokens) {
        return Err(io::Error::new(ErrorKind::InvalidInput, "capture mode allows 1..=128 tokens"));
    }
    if prompt.is_empty() || prompt.len() > 4096 {
        return Err(io::Error::new(ErrorKind::InvalidInput, "prompt must be 1..=4096 bytes"));
    }
    let model = model.canonicalize()?;
    if inspect_gguf(&model)?.tensor_count == 0 {
        return Err(io::Error::new(ErrorKind::InvalidData, "GGUF has no tensors"));
    }
    let mut cmd = Command::new(runtime);
    cmd.arg("-m").arg(&model)
        .arg("-p").arg(prompt)
        .arg("-n").arg(tokens.to_string())
        .arg("-ngl").arg(gpu_layers.to_string())
        .arg("--no-display-prompt")
        .arg("--simple-io")
        .arg("-st")
        ;
    if let Some(device)=device{
        if device.is_empty()||device.len()>80||!device.bytes().all(|b|
            b.is_ascii_alphanumeric()||matches!(b,b'_'|b'-'|b'/'))
        {return Err(io::Error::new(ErrorKind::InvalidInput,"invalid explicit backend device")); }
        cmd.arg("-dev").arg(device);
    }
    let result=crate::jobs::capture_with_timeout(&mut cmd,128*1024,Duration::from_secs(600))?;
    let stderr = String::from_utf8_lossy(&result.stderr);
    if !result.status.success() {
        return Err(io::Error::new(ErrorKind::Other, format!(
            "llama-cli exited with {}: {}",
            result.status, stderr.chars().take(1000).collect::<String>()
        )));
    }
    Ok(SmokeCapture {
        stdout: String::from_utf8_lossy(&result.stdout).chars().take(16_384).collect(),
        stderr: stderr.chars().take(4_096).collect(),
    })
}
