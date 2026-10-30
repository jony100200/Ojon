//! LoRA-then-quant: merge an adapter into its high-precision original through
//! llama.cpp's `llama-export-lora`; the receipt names the task-suite gate and
//! the `auto-quant` step that follow.

use crate::cli::common::{invalid, next_argument};
use ojon_backend::discovery::{discovered_path, resolve_backend_command, BackendId};
use ojon_backend::lora_merge::{prepare_lora_merge, run_lora_merge, LoraMergeJob};
use ojon_backend::recipe_run::plain;
use std::ffi::OsString;
use std::io;
use std::path::PathBuf;

fn text(v: OsString, what: &str) -> io::Result<String> {
    v.to_str().map(str::to_owned).ok_or_else(|| invalid(format!("{what} requires UTF-8")))
}

pub fn handle(cmd: &str, mut args: impl Iterator<Item = OsString>) -> io::Result<bool> {
    match cmd {
        "lora-merge-plan" | "lora-merge-run" => {
            let execute = cmd == "lora-merge-run";
            let base = PathBuf::from(next_argument(&mut args, "high-precision original GGUF (F16/BF16/F32)")?);
            let adapter = PathBuf::from(next_argument(&mut args, "LoRA adapter (GGUF file or PEFT directory)")?);
            let output = PathBuf::from(next_argument(&mut args, "new merged GGUF path")?);
            let mut export_tool = resolve_backend_command(BackendId::LlamaExportLora, "llama-export-lora")?;
            // convert_lora_to_gguf.py ships beside convert_hf_to_gguf.py in every llama.cpp checkout.
            let mut convert_script = discovered_path(BackendId::HfConverter)
                .map(|p| p.with_file_name("convert_lora_to_gguf.py"))
                .filter(|p| p.is_file());
            let mut python = PathBuf::from("python");
            let mut hf_base: Option<PathBuf> = None;
            let mut scale: Option<f64> = None;
            let mut task_suite: Option<PathBuf> = None;
            let (mut confirm, mut json) = (false, false);
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--export-tool") => export_tool = PathBuf::from(next_argument(&mut args, "llama-export-lora executable")?),
                    Some("--convert-script") => convert_script = Some(PathBuf::from(next_argument(&mut args, "convert_lora_to_gguf.py")?)),
                    Some("--python") => python = PathBuf::from(next_argument(&mut args, "Python executable")?),
                    Some("--hf-base") => hf_base = Some(PathBuf::from(next_argument(&mut args, "Hugging Face base model folder")?)),
                    Some("--scale") => {
                        let s: f64 = text(next_argument(&mut args, "adapter scale")?, "scale")?.parse().map_err(|_| invalid("scale must be a number"))?;
                        if !(s.is_finite() && s > 0.0) {
                            return Err(invalid("scale must be a positive number"));
                        }
                        scale = Some(s);
                    }
                    Some("--task-suite") => task_suite = Some(PathBuf::from(next_argument(&mut args, "held-out task suite JSON")?)),
                    Some("--json") => json = true,
                    Some("--confirm") if execute => confirm = true,
                    _ => return Err(invalid(format!("unknown lora-merge option: {}", option.to_string_lossy()))),
                }
            }
            let plan = prepare_lora_merge(LoraMergeJob { base, adapter, output, export_tool, scale, convert_script, python, hf_base, task_suite })?;
            if json {
                println!("{}", serde_json::to_string_pretty(&plan.plan_json())?);
            } else {
                println!("base: {} ({} bytes, file_type {})", plain(&plan.job.base), plan.base_bytes, plan.base_file_type);
                println!("adapter: {} ({})", plain(&plan.job.adapter), plan.adapter_kind.as_str());
                if let Some(c) = plan.convert_command_line() {
                    println!("convert: {c}");
                }
                println!("export: {}", plan.export_command_line());
                println!("output: {}", plain(&plan.job.output));
                println!("after merge, gate: {}", plan.gate_command());
                println!("after gate, quantize: {}", plan.quant_command());
            }
            if !execute {
                println!("writes: nothing");
                println!("next_step: ojon lora-merge-run <same arguments> --confirm");
                return Ok(true);
            }
            if !confirm {
                return Err(invalid("lora-merge-run writes the merged GGUF (and a converted adapter for a PEFT folder) plus a receipt; add --confirm"));
            }
            let receipt = run_lora_merge(&plan, |line| eprintln!("{line}"))?;
            if json {
                println!("{}", std::fs::read_to_string(&receipt)?);
            } else {
                let doc: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt)?)?;
                println!("receipt: {}", receipt.display());
                println!("merged: {} ({} bytes)", doc["merged"]["path"].as_str().unwrap_or(""), doc["merged"]["bytes"]);
                println!("claim: {}", doc["claim"].as_str().unwrap_or(""));
                println!("next: {}", doc["next_steps"]["gate"].as_str().unwrap_or(""));
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
