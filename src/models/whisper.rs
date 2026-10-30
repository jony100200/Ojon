//! whisper.cpp model quantization through its public `whisper-quantize` tool.
//!
//! Upstream usage, verified in whisper.cpp `examples/quantize/quantize.cpp`
//! (master, 2026-10-30): `whisper-quantize model-f32.bin model-quant.bin type`,
//! where `type` is parsed by `ggml_parse_ftype` (`examples/common-ggml.cpp`).
//! The accepted names are lower-case (`q4_0 q4_1 q5_0 q5_1 q8_0 q2_k .. q6_k`);
//! an upper-case name is silently read as the integer 0, so Ojon always passes
//! the lower-case spelling. The tool reads the GGML `.bin` container only
//! (`whisper_model_quantize` checks `GGML_FILE_MAGIC` and rejects anything
//! else), so a GGUF input is refused here before the tool is spawned. The
//! CMake target is `whisper-quantize`; older checkouts built it as `quantize`.
//!
//! Plan writes nothing. Run writes `<stem>-<type>.bin` per format and
//! `whisper-quant-receipt.json` with sizes; nothing is ever overwritten.
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{json, Value};

use crate::jobs::{run_with_log, tool_command};
use crate::probe::{probe_model, ModelCategory};
use crate::recipe_run::plain;

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, s.into())
}

pub const WHISPER_QUANT_RECEIPT: &str = "whisper-quant-receipt.json";
pub const WHISPER_QUANT_SCHEMA: &str = "ojon.whisper_quant.v1";
pub const WHISPER_QUANT_CLAIM: &str = "size reduction only; nothing here measures transcription quality, run whisper-parity-run on held-out clips before using a candidate";

/// The whisper.cpp scalar formats Ojon offers. K-quants exist upstream too but
/// are not exposed until one has been checked through the WER gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WhisperFormat {
    Q8_0,
    Q5_0,
    Q5_1,
    Q4_0,
    Q4_1,
}

impl WhisperFormat {
    pub const ALL: [WhisperFormat; 5] = [Self::Q8_0, Self::Q5_1, Self::Q5_0, Self::Q4_1, Self::Q4_0];
    pub const DEFAULT: [WhisperFormat; 3] = [Self::Q8_0, Self::Q5_0, Self::Q4_0];

    /// Accepts any case; the tool argument is always `as_str()` (lower-case).
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "q8_0" => Some(Self::Q8_0),
            "q5_0" => Some(Self::Q5_0),
            "q5_1" => Some(Self::Q5_1),
            "q4_0" => Some(Self::Q4_0),
            "q4_1" => Some(Self::Q4_1),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Q8_0 => "q8_0",
            Self::Q5_0 => "q5_0",
            Self::Q5_1 => "q5_1",
            Self::Q4_0 => "q4_0",
            Self::Q4_1 => "q4_1",
        }
    }
}

/// Parse `q8_0,q5_0` (comma separated, any case, duplicates dropped).
pub fn parse_whisper_formats(text: &str) -> io::Result<Vec<WhisperFormat>> {
    let mut out: Vec<WhisperFormat> = Vec::new();
    for part in text.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        let f = WhisperFormat::parse(part).ok_or_else(|| {
            invalid(format!(
                "unknown whisper format '{part}' (accepted: {})",
                WhisperFormat::ALL.iter().map(|f| f.as_str()).collect::<Vec<_>>().join(", ")
            ))
        })?;
        if !out.contains(&f) {
            out.push(f);
        }
    }
    if out.is_empty() {
        return Err(invalid("at least one whisper format is required"));
    }
    Ok(out)
}

#[derive(Clone, Debug)]
pub struct WhisperQuantPlan {
    pub source: PathBuf,
    pub output_dir: PathBuf,
    pub tool: PathBuf,
    pub formats: Vec<WhisperFormat>,
    pub source_bytes: u64,
    pub identification: String,
}

impl WhisperQuantPlan {
    fn stem(&self) -> String {
        self.source.file_stem().unwrap_or_default().to_string_lossy().into_owned()
    }

    pub fn output_for(&self, format: WhisperFormat) -> PathBuf {
        self.output_dir.join(format!("{}-{}.bin", self.stem(), format.as_str()))
    }

    pub fn receipt_path(&self) -> PathBuf {
        self.output_dir.join(WHISPER_QUANT_RECEIPT)
    }

    pub fn tool_args(&self, format: WhisperFormat) -> Vec<String> {
        vec![plain(&self.source), plain(&self.output_for(format)), format.as_str().to_owned()]
    }

    pub fn command_line(&self, format: WhisperFormat) -> String {
        let mut parts = vec![plain(&self.tool)];
        parts.extend(self.tool_args(format));
        parts.into_iter().map(|p| if p.contains(' ') { format!("\"{p}\"") } else { p }).collect::<Vec<_>>().join(" ")
    }

    pub fn plan_json(&self) -> Value {
        json!({
            "schema": "ojon.whisper_quant_plan.v1",
            "source": plain(&self.source),
            "source_bytes": self.source_bytes,
            "source_identification": self.identification,
            "tool": plain(&self.tool),
            "tool_usage": "whisper-quantize <model-f32.bin> <model-quant.bin> <type>",
            "output_dir": plain(&self.output_dir),
            "outputs": self.formats.iter().map(|f| json!({
                "format": f.as_str(),
                "path": plain(&self.output_for(*f)),
                "command": self.command_line(*f),
            })).collect::<Vec<_>>(),
            "receipt": plain(&self.receipt_path()),
            "writes": "nothing until whisper-quant-run --confirm",
        })
    }
}

/// Validate the input (must probe as a whisper GGML `.bin`), the formats and
/// the destination (no planned file may already exist). Writes nothing.
pub fn prepare_whisper_quant(
    source: &Path,
    output_dir: &Path,
    tool: &Path,
    formats: &[WhisperFormat],
) -> io::Result<WhisperQuantPlan> {
    if formats.is_empty() {
        return Err(invalid("at least one whisper format is required"));
    }
    if tool.as_os_str().is_empty() {
        return Err(invalid("whisper-quantize executable missing"));
    }
    let probe = probe_model(source).map_err(|e| invalid(format!("source {}: {e}", source.display())))?;
    if probe.category != ModelCategory::Speech {
        return Err(invalid(format!(
            "source probes as {}, not a speech model; whisper-quantize needs a whisper.cpp GGML .bin",
            probe.category.as_str()
        )));
    }
    if probe.format != "ggml_bin" {
        return Err(invalid(format!(
            "source is {}; whisper.cpp's quantize reads the GGML .bin container only (it checks GGML_FILE_MAGIC), not {}",
            probe.format,
            source.display()
        )));
    }
    if output_dir.is_file() {
        return Err(invalid("output path is a file, not a directory"));
    }
    let mut formats_dedup: Vec<WhisperFormat> = Vec::new();
    for f in formats {
        if !formats_dedup.contains(f) {
            formats_dedup.push(*f);
        }
    }
    let plan = WhisperQuantPlan {
        source: probe.path.clone(),
        output_dir: output_dir.to_path_buf(),
        tool: tool.to_path_buf(),
        formats: formats_dedup,
        source_bytes: probe.bytes,
        identification: probe.identification.clone(),
    };
    let mut planned = vec![plan.receipt_path()];
    planned.extend(plan.formats.iter().map(|f| plan.output_for(*f)));
    if let Some(existing) = planned.iter().find(|p| p.exists()) {
        return Err(invalid(format!("{} already exists; Ojon never overwrites", existing.display())));
    }
    Ok(plan)
}

fn write_new(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut f = fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    f.write_all(bytes)
}

/// Run the tool once per format. A failed format is recorded in the receipt
/// and the next one still runs; the receipt is written when every format has
/// been attempted. Returns the receipt path.
pub fn run_whisper_quant<F>(plan: &WhisperQuantPlan, mut log: F) -> io::Result<PathBuf>
where
    F: FnMut(&str),
{
    fs::create_dir_all(&plan.output_dir)?;
    let receipt_path = plan.receipt_path();
    if receipt_path.exists() {
        return Err(invalid(format!("{} already exists", receipt_path.display())));
    }
    let mut outputs = Vec::with_capacity(plan.formats.len());
    for format in &plan.formats {
        let out = plan.output_for(*format);
        if out.exists() {
            return Err(invalid(format!("{} already exists; Ojon never overwrites", out.display())));
        }
        let command = plan.command_line(*format);
        log(&format!("[whisper-quant] {}: {command}", format.as_str()));
        let started = Instant::now();
        let mut cmd = tool_command(&plan.tool);
        cmd.args(plan.tool_args(*format));
        let result = run_with_log(&mut cmd, |line| log(line)).and_then(|_| {
            let bytes = out.metadata().map(|m| m.len()).unwrap_or(0);
            if bytes == 0 {
                return Err(io::Error::other("tool exited but wrote no output file"));
            }
            // The output must still read as a whisper GGML container.
            let probe = probe_model(&out)?;
            if probe.category != ModelCategory::Speech || probe.format != "ggml_bin" {
                return Err(io::Error::other("output does not probe as a whisper GGML .bin"));
            }
            Ok(bytes)
        });
        let seconds = started.elapsed().as_secs_f64();
        match result {
            Ok(bytes) => {
                log(&format!(
                    "[whisper-quant] {} ok: {} bytes ({:.1}% of source)",
                    format.as_str(),
                    bytes,
                    100.0 * bytes as f64 / plan.source_bytes.max(1) as f64
                ));
                outputs.push(json!({
                    "format": format.as_str(),
                    "path": plain(&out),
                    "command": command,
                    "status": "ok",
                    "output_bytes": bytes,
                    "size_ratio": bytes as f64 / plan.source_bytes.max(1) as f64,
                    "seconds": seconds,
                }));
            }
            Err(e) => {
                log(&format!("[whisper-quant] {} FAILED: {e}", format.as_str()));
                outputs.push(json!({
                    "format": format.as_str(),
                    "path": plain(&out),
                    "command": command,
                    "status": "failed",
                    "error": e.to_string(),
                    "output_bytes": out.metadata().map(|m| m.len()).unwrap_or(0),
                    "seconds": seconds,
                }));
            }
        }
    }
    let ok = outputs.iter().filter(|o| o["status"] == "ok").count();
    let receipt = json!({
        "schema": WHISPER_QUANT_SCHEMA,
        "created_unix": SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        "source": plain(&plan.source),
        "source_bytes": plan.source_bytes,
        "source_identification": plan.identification,
        "tool": plain(&plan.tool),
        "output_dir": plain(&plan.output_dir),
        "outputs": outputs,
        "formats_ok": ok,
        "formats_failed": plan.formats.len() - ok,
        "quality": "NOT_EVALUATED",
        "claim": WHISPER_QUANT_CLAIM,
    });
    write_new(&receipt_path, &serde_json::to_vec_pretty(&receipt)?)?;
    log(&format!("[whisper-quant] {ok}/{} formats written -> {}", plan.formats.len(), receipt_path.display()));
    Ok(receipt_path)
}

/// 16-byte whisper-style GGML header (magic + multilingual `n_vocab`), enough
/// for `probe_model` to classify the file as Speech. Test support only.
#[cfg(test)]
pub(crate) fn write_fake_whisper_bin(path: &Path) {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&0x6767_6d6cu32.to_le_bytes());
    bytes.extend_from_slice(&51_865i32.to_le_bytes());
    bytes.extend_from_slice(&[0u8; 8]);
    fs::write(path, bytes).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ojon-whisper-{tag}-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn formats_parse_any_case_and_emit_lowercase() {
        let f = parse_whisper_formats("Q8_0, q5_0,q8_0").unwrap();
        assert_eq!(f, vec![WhisperFormat::Q8_0, WhisperFormat::Q5_0]);
        assert_eq!(WhisperFormat::Q4_1.as_str(), "q4_1");
        assert!(parse_whisper_formats("q4_k").is_err(), "k-quants not exposed");
        assert!(parse_whisper_formats("").is_err());
        assert_eq!(WhisperFormat::DEFAULT.len(), 3);
    }

    #[test]
    fn plan_requires_whisper_ggml_and_refuses_existing_outputs() {
        let dir = temp_dir("plan");
        let model = dir.join("ggml-base.bin");
        write_fake_whisper_bin(&model);
        let out = dir.join("out");
        let tool = PathBuf::from("whisper-quantize");
        let plan = prepare_whisper_quant(&model, &out, &tool, &[WhisperFormat::Q8_0, WhisperFormat::Q5_0, WhisperFormat::Q8_0]).unwrap();
        assert_eq!(plan.formats, vec![WhisperFormat::Q8_0, WhisperFormat::Q5_0]);
        assert_eq!(plan.source_bytes, 16);
        assert!(plan.output_for(WhisperFormat::Q5_0).ends_with("ggml-base-q5_0.bin"));
        let args = plan.tool_args(WhisperFormat::Q8_0);
        assert_eq!(args.len(), 3);
        assert_eq!(args[2], "q8_0");
        assert!(args[0].ends_with("ggml-base.bin"));
        let j = plan.plan_json();
        assert_eq!(j["schema"], "ojon.whisper_quant_plan.v1");
        assert_eq!(j["outputs"].as_array().unwrap().len(), 2);
        assert!(j["outputs"][0]["command"].as_str().unwrap().ends_with(" q8_0"));

        // An output that already exists blocks the plan.
        fs::create_dir_all(&out).unwrap();
        fs::write(out.join("ggml-base-q5_0.bin"), b"x").unwrap();
        assert!(prepare_whisper_quant(&model, &out, &tool, &[WhisperFormat::Q5_0]).is_err());
        assert!(prepare_whisper_quant(&model, &out, &tool, &[WhisperFormat::Q8_0]).is_ok());

        // Not a whisper container: refused before any tool runs.
        let pickle = dir.join("pytorch_model.bin");
        fs::write(&pickle, b"PK..............").unwrap();
        assert!(prepare_whisper_quant(&pickle, &out, &tool, &[WhisperFormat::Q8_0]).is_err());
        assert!(prepare_whisper_quant(&model, &out, &tool, &[]).is_err());
        assert!(prepare_whisper_quant(&model, &out, &PathBuf::new(), &[WhisperFormat::Q8_0]).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn run_refuses_existing_receipt() {
        let dir = temp_dir("run");
        let model = dir.join("ggml-tiny.bin");
        write_fake_whisper_bin(&model);
        let out = dir.join("out");
        let plan = prepare_whisper_quant(&model, &out, Path::new("whisper-quantize"), &[WhisperFormat::Q8_0]).unwrap();
        fs::create_dir_all(&out).unwrap();
        fs::write(out.join(WHISPER_QUANT_RECEIPT), b"{}").unwrap();
        assert!(run_whisper_quant(&plan, |_| {}).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
