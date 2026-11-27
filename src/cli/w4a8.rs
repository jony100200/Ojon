//! `w4a8-plan` / `w4a8-run`: floating safetensors -> comfy-kitchen W4A8 or W6A8 safetensors.

use crate::cli::common::{invalid, next_argument};
use ojon_backend::quant::w4a8::{describe_plan, h3_default_layers, run, wan_default_layers, Bits, Request};
use std::ffi::OsString;
use std::io;
use std::path::PathBuf;

const HELP: &str = "w4a8-plan|w4a8-run <source.safetensors> <output.safetensors> [--bits 4|6] [--profile h3|wan] [--layers glob,glob] [--threads N] [--confirm]";

pub fn handle(cmd: &str, mut args: impl Iterator<Item = OsString>) -> io::Result<bool> {
    if cmd != "w4a8-plan" && cmd != "w4a8-run" {
        return Ok(false);
    }
    let source = PathBuf::from(next_argument(&mut args, "source safetensors")?);
    let output = PathBuf::from(next_argument(&mut args, "output safetensors")?);
    let mut req = Request { source, output, bits: Bits::Four, layers: h3_default_layers(), threads: default_threads(), };
    let mut confirm = false;
    let mut profile = String::from("h3");
    let mut explicit_layers = None;
    while let Some(opt) = args.next() {
        match opt.to_str() {
            Some("--bits") => {
                req.bits = match text(&mut args, "4 or 6")?.as_str() {
                    "4" => Bits::Four,
                    "6" => Bits::Six,
                    other => return Err(invalid(format!("--bits must be 4 or 6, got {other}"))),
                }
            }
            Some("--profile") => profile = text(&mut args, "h3 or wan")?,
            Some("--layers") => explicit_layers = Some(text(&mut args, "comma-separated layer globs")?.split(',').map(|s| s.trim().to_owned()).collect()),
            Some("--threads") => req.threads = text(&mut args, "thread count")?.parse().map_err(|_| invalid("--threads must be a number"))?,
            Some("--confirm") => confirm = true,
            _ => return Err(invalid(format!("unknown option {}; usage: {HELP}", opt.to_string_lossy()))),
        }
    }
    let defaults = match profile.as_str() {
        "h3" => h3_default_layers(),
        "wan" => wan_default_layers(),
        other => return Err(invalid(format!("--profile must be h3 or wan, got {other}"))),
    };
    req.layers = explicit_layers.unwrap_or(defaults);
    if cmd == "w4a8-plan" {
        println!("{}", serde_json::to_string_pretty(&describe_plan(&req)?)?);
        return Ok(true);
    }
    if !confirm {
        return Err(invalid("w4a8-run writes a new checkpoint and requires --confirm"));
    }
    let receipt = run(&req, |m| eprintln!("{m}"))?;
    println!("{}", serde_json::to_string_pretty(&receipt)?);
    Ok(true)
}

fn text(args: &mut impl Iterator<Item = OsString>, what: &str) -> io::Result<String> {
    next_argument(args, what)?.into_string().map_err(|_| invalid(format!("{what} must be UTF-8")))
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get().saturating_sub(2).max(1)).unwrap_or(4)
}
