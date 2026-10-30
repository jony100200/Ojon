//! Speech (whisper.cpp) commands: quantize through `whisper-quantize`, and the
//! WER parity gate through `whisper-cli` on held-out clips.

use crate::cli::common::{invalid, next_argument};
use ojon_backend::discovery::{resolve_backend_command, BackendId};
use ojon_backend::whisper::{parse_whisper_formats, prepare_whisper_quant, run_whisper_quant, WhisperFormat};
use ojon_backend::whisper_parity::{prepare_whisper_parity, run_whisper_parity, WhisperParityJob, WhisperParitySettings};
use std::ffi::OsString;
use std::io;
use std::path::PathBuf;

fn text(v: OsString, what: &str) -> io::Result<String> {
    v.to_str().map(str::to_owned).ok_or_else(|| invalid(format!("{what} requires UTF-8")))
}

pub fn handle(cmd: &str, mut args: impl Iterator<Item = OsString>) -> io::Result<bool> {
    match cmd {
        "whisper-quant-plan" | "whisper-quant-run" => {
            // whisper.cpp GGML .bin -> q8_0 / q5_0 / q4_0 (...) through the
            // public whisper-quantize tool. Plan writes nothing.
            let execute = cmd == "whisper-quant-run";
            let model = PathBuf::from(next_argument(&mut args, "whisper.cpp GGML model (.bin)")?);
            let output_dir = PathBuf::from(next_argument(&mut args, "output directory")?);
            let mut formats: Vec<WhisperFormat> = WhisperFormat::DEFAULT.to_vec();
            let mut tool = resolve_backend_command(BackendId::WhisperQuantize, "whisper-quantize")?;
            let (mut confirm, mut json) = (false, false);
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--formats") => formats = parse_whisper_formats(&text(next_argument(&mut args, "formats, e.g. q8_0,q5_0,q4_0")?, "formats")?)?,
                    Some("--tool") => tool = PathBuf::from(next_argument(&mut args, "whisper-quantize executable")?),
                    Some("--json") => json = true,
                    Some("--confirm") if execute => confirm = true,
                    _ => return Err(invalid(format!("unknown whisper-quant option: {}", option.to_string_lossy()))),
                }
            }
            let plan = prepare_whisper_quant(&model, &output_dir, &tool, &formats)?;
            if json {
                println!("{}", serde_json::to_string_pretty(&plan.plan_json())?);
            } else {
                println!("source: {} ({} bytes, {})", ojon_backend::recipe_run::plain(&plan.source), plan.source_bytes, plan.identification);
                println!("tool: {}", plan.tool.display());
                for f in &plan.formats {
                    println!("output {}: {}", f.as_str(), plan.command_line(*f));
                }
            }
            if !execute {
                println!("writes: nothing");
                println!("next_step: ojon whisper-quant-run <same arguments> --confirm");
                return Ok(true);
            }
            if !confirm {
                return Err(invalid("whisper-quant-run writes one .bin per format and a receipt; add --confirm to execute"));
            }
            let receipt = run_whisper_quant(&plan, |line| eprintln!("{line}"))?;
            if json {
                println!("{}", std::fs::read_to_string(&receipt)?);
            } else {
                let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
                println!("receipt: {}", receipt.display());
                for o in doc["outputs"].as_array().into_iter().flatten() {
                    println!(
                        "{} {}: {} bytes{}",
                        o["format"].as_str().unwrap_or("?"),
                        o["status"].as_str().unwrap_or("?"),
                        o["output_bytes"],
                        o["error"].as_str().map(|e| format!(" ({e})")).unwrap_or_default()
                    );
                }
                println!("claim: {}", doc["claim"]);
            }
            Ok(true)
        }
        "whisper-parity-plan" | "whisper-parity-run" => {
            // Original and candidates transcribe the same clips through
            // whisper-cli; corpus WER per model, candidate-minus-original delta
            // gated by --max-wer-delta. Plan writes nothing.
            let execute = cmd == "whisper-parity-run";
            let original = PathBuf::from(next_argument(&mut args, "original whisper.cpp model (.bin)")?);
            let mut candidates: Vec<PathBuf> = Vec::new();
            let mut settings = WhisperParitySettings {
                whisper_cli: resolve_backend_command(BackendId::WhisperCpp, "whisper-cli")?,
                ..Default::default()
            };
            let mut output_dir: Option<PathBuf> = None;
            let (mut confirm, mut json) = (false, false);
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--clips") => settings.clips = PathBuf::from(next_argument(&mut args, "clips manifest (.jsonl)")?),
                    Some("--max-wer-delta") => {
                        settings.max_wer_delta = text(next_argument(&mut args, "maximum WER increase")?, "max-wer-delta")?
                            .parse::<f64>()
                            .map_err(|_| invalid("max-wer-delta must be a number"))?
                    }
                    Some("--language") => settings.language = text(next_argument(&mut args, "language code")?, "language")?,
                    Some("--threads") => {
                        settings.threads = text(next_argument(&mut args, "threads")?, "threads")?
                            .parse::<usize>()
                            .map_err(|_| invalid("threads must be a number"))?
                    }
                    Some("--whisper-cli") => settings.whisper_cli = PathBuf::from(next_argument(&mut args, "whisper-cli executable")?),
                    Some("--out-dir") => output_dir = Some(PathBuf::from(next_argument(&mut args, "empty output directory")?)),
                    Some("--json") => json = true,
                    Some("--confirm") if execute => confirm = true,
                    Some("--confirm") => return Err(invalid("--confirm only applies to whisper-parity-run")),
                    Some(other) if other.starts_with("--") => {
                        return Err(invalid(format!("unknown whisper-parity option: {other}")))
                    }
                    _ => candidates.push(PathBuf::from(option)),
                }
            }
            if settings.clips.as_os_str().is_empty() {
                return Err(invalid("--clips manifest.jsonl is required"));
            }
            // Default output folder sits beside the manifest; a second run must
            // name a fresh --out-dir because nothing is overwritten.
            let output_dir = output_dir.unwrap_or_else(|| {
                settings
                    .clips
                    .parent()
                    .filter(|p| !p.as_os_str().is_empty())
                    .map(|p| p.join("whisper-parity"))
                    .unwrap_or_else(|| PathBuf::from("whisper-parity"))
            });
            let plan = prepare_whisper_parity(WhisperParityJob { original, candidates, output_dir, settings })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&plan.plan_json())?);
            } else {
                for m in &plan.models {
                    println!("{}: {} ({} bytes)", m.role, m.model.display(), m.bytes);
                }
                println!(
                    "clips {} | language {} | threads {} | gate max WER delta {}",
                    plan.clips.len(),
                    plan.job.settings.language,
                    plan.job.settings.threads,
                    plan.job.settings.max_wer_delta
                );
                println!("output_dir: {}", plan.job.output_dir.display());
                let commands = plan.commands();
                for c in commands.iter().take(2) {
                    println!("command: {c}");
                }
                if commands.len() > 2 {
                    println!("... {} commands in total (--json lists all)", commands.len());
                }
            }
            if !execute {
                println!("writes: nothing");
                println!("next_step: ojon whisper-parity-run <same arguments> --confirm");
                return Ok(true);
            }
            if !confirm {
                return Err(invalid("whisper-parity-run transcribes every clip with every model and writes transcripts and a receipt; add --confirm to execute"));
            }
            let receipt = run_whisper_parity(&plan, |line| eprintln!("{line}"))?;
            if json {
                println!("{}", std::fs::read_to_string(&receipt)?);
            } else {
                let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
                println!("receipt: {}", receipt.display());
                for r in doc["results"].as_array().into_iter().flatten() {
                    println!("{} corpus WER: {}", r["role"].as_str().unwrap_or("?"), r["corpus_wer"]);
                }
                println!("gate: {}", doc["gate"]);
                println!("claim: {}", doc["claim"]);
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
