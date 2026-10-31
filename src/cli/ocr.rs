//! OCR parity gate: character error rate of an original VLM and its quantized
//! candidates on a manifest of pages with known text, through `llama-mtmd-cli`.

use crate::cli::common::{invalid, next_argument};
use ojon_backend::discovery::{resolve_backend_command, BackendId};
use ojon_backend::ocr_parity::{prepare_ocr_parity, run_ocr_parity, OcrModel, OcrParityJob, OcrParitySettings, DEFAULT_PROMPT};
use ojon_backend::recipe_run::plain;
use std::ffi::OsString;
use std::io;
use std::path::PathBuf;

fn text(v: OsString, what: &str) -> io::Result<String> {
    v.to_str().map(str::to_owned).ok_or_else(|| invalid(format!("{what} requires UTF-8")))
}

pub fn handle(cmd: &str, mut args: impl Iterator<Item = OsString>) -> io::Result<bool> {
    match cmd {
        "ocr-parity-plan" | "ocr-parity-run" => {
            let execute = cmd == "ocr-parity-run";
            let original = PathBuf::from(next_argument(&mut args, "original VLM GGUF")?);
            let mut candidates: Vec<PathBuf> = Vec::new();
            let mut mmproj: Option<PathBuf> = None;
            let mut mmproj_candidate: Option<PathBuf> = None;
            let mut pages: Option<PathBuf> = None;
            let mut out_dir: Option<PathBuf> = None;
            let mut settings = OcrParitySettings {
                tool: resolve_backend_command(BackendId::LlamaMtmdCli, "llama-mtmd-cli")?,
                pages: PathBuf::new(),
                prompt: DEFAULT_PROMPT.to_owned(),
                tokens: 512,
                gpu_layers: 99,
                device: None,
                max_cer_delta: 0.02,
            };
            let (mut confirm, mut json) = (false, false);
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--mmproj") => mmproj = Some(PathBuf::from(next_argument(&mut args, "projector GGUF")?)),
                    Some("--mmproj-candidate") => mmproj_candidate = Some(PathBuf::from(next_argument(&mut args, "candidate projector GGUF")?)),
                    Some("--pages") => pages = Some(PathBuf::from(next_argument(&mut args, "page manifest JSONL")?)),
                    Some("--prompt") => settings.prompt = text(next_argument(&mut args, "prompt")?, "prompt")?,
                    Some("--tokens") => settings.tokens = text(next_argument(&mut args, "max new tokens")?, "tokens")?.parse().map_err(|_| invalid("tokens must be a number"))?,
                    Some("--ngl") => settings.gpu_layers = text(next_argument(&mut args, "gpu layers")?, "ngl")?.parse().map_err(|_| invalid("ngl must be a number"))?,
                    Some("--device") => settings.device = Some(text(next_argument(&mut args, "backend device list, e.g. Vulkan0")?, "device")?),
                    Some("--max-cer-delta") => settings.max_cer_delta = text(next_argument(&mut args, "max CER delta")?, "max-cer-delta")?.parse().map_err(|_| invalid("max-cer-delta must be a number"))?,
                    Some("--tool") => settings.tool = PathBuf::from(next_argument(&mut args, "llama-mtmd-cli executable")?),
                    Some("--out-dir") => out_dir = Some(PathBuf::from(next_argument(&mut args, "output directory")?)),
                    Some("--json") => json = true,
                    Some("--confirm") if execute => confirm = true,
                    Some(s) if !s.starts_with("--") => candidates.push(PathBuf::from(s)),
                    _ => return Err(invalid(format!("unknown ocr-parity option: {}", option.to_string_lossy()))),
                }
            }
            let mmproj = mmproj.ok_or_else(|| invalid("--mmproj <projector.gguf> is required"))?;
            settings.pages = pages.ok_or_else(|| invalid("--pages <pages.jsonl> is required ({\"image\":..., \"text\":...} per line)"))?;
            let output_dir = out_dir.unwrap_or_else(|| settings.pages.parent().map(|p| p.join("ocr-parity")).unwrap_or_else(|| PathBuf::from("ocr-parity")));
            let cand_proj = mmproj_candidate.clone().unwrap_or_else(|| mmproj.clone());
            let plan = prepare_ocr_parity(OcrParityJob {
                original: OcrModel { model: original, mmproj: mmproj.clone() },
                candidates: candidates.into_iter().map(|m| OcrModel { model: m, mmproj: cand_proj.clone() }).collect(),
                output_dir,
                settings,
            })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&plan.plan_json())?);
            } else {
                println!("pages: {} from {}", plan.pages.len(), plain(&plan.job.settings.pages));
                for m in &plan.models {
                    println!("{}: {} ({} bytes) + {}", m.role, plain(&m.model), m.bytes, plain(&m.mmproj));
                }
                println!("runs: {} (first: {})", plan.models.len() * plan.pages.len(), plan.commands().first().cloned().unwrap_or_default());
                println!("gate: candidate CER - original CER <= {}", plan.job.settings.max_cer_delta);
            }
            if !execute {
                println!("writes: nothing");
                println!("next_step: ojon ocr-parity-run <same arguments> --confirm");
                return Ok(true);
            }
            if !confirm {
                return Err(invalid("ocr-parity-run reads every page with every model and writes transcripts plus a receipt; add --confirm"));
            }
            let receipt = run_ocr_parity(&plan, |line| eprintln!("{line}"))?;
            if json {
                println!("{}", std::fs::read_to_string(&receipt)?);
            } else {
                let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
                println!("receipt: {}", receipt.display());
                for m in doc["models"].as_array().into_iter().flatten() {
                    println!(
                        "{}: CER {:.4} (delta {:+.4}) {} | {} bytes ({:.1}%)",
                        m["role"].as_str().unwrap_or(""),
                        m["corpus_cer"].as_f64().unwrap_or(f64::NAN),
                        m["cer_delta_vs_original"].as_f64().unwrap_or(0.0),
                        if m["within_gate"].as_bool().unwrap_or(false) { "within gate" } else { "OVER gate" },
                        m["bytes"],
                        m["percent_of_original"].as_f64().unwrap_or(0.0)
                    );
                }
                println!("claim: {}", doc["claim"].as_str().unwrap_or(""));
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
