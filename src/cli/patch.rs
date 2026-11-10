//! Verified weight patches (delta -> new HF checkpoint) with exact rollback, and the reusable pipeline.

use crate::cli::common::{invalid, next_argument};
use ojon_backend::pipeline::{load_pipeline_config, plan_pipeline, run_pipeline};
use ojon_backend::weight_patch::{
    prepare_weight_patch, rollback_weight_patch, run_weight_patch, verify_weight_patch,
};
use std::ffi::OsString;
use std::io;
use std::path::PathBuf;

fn print(v: &serde_json::Value) -> io::Result<()> {
    println!("{}", serde_json::to_string_pretty(v)?);
    Ok(())
}

pub fn handle(cmd: &str, mut args: impl Iterator<Item = OsString>) -> io::Result<bool> {
    match cmd {
        "weight-patch-plan" | "weight-patch-run" => {
            let execute = cmd == "weight-patch-run";
            let source = PathBuf::from(next_argument(&mut args, "original Hugging Face model folder")?);
            let manifest = PathBuf::from(next_argument(&mut args, "ojon.weight_patch.v1 manifest JSON")?);
            let output = PathBuf::from(next_argument(&mut args, "new patched model folder")?);
            let (mut confirm, mut link) = (false, false);
            for option in args.by_ref() {
                match option.to_str() {
                    Some("--confirm") => confirm = true,
                    Some("--link-unchanged") => link = true,
                    _ => return Err(invalid(format!("unknown option: {}", option.to_string_lossy()))),
                }
            }
            let plan = prepare_weight_patch(&source, &manifest, &output, link)?;
            if !execute {
                print(&plan.as_json())?;
                return Ok(true);
            }
            if !confirm {
                return Err(invalid("weight-patch-run writes a new checkpoint and requires --confirm"));
            }
            print(&run_weight_patch(&plan, |m| eprintln!("{m}"))?)?;
            Ok(true)
        }
        "weight-patch-verify" => {
            let dir = PathBuf::from(next_argument(&mut args, "patched model folder")?);
            let r = verify_weight_patch(&dir)?;
            print(&r)?;
            if r["status"] != "VERIFIED" {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "patched folder does not match its receipt"));
            }
            Ok(true)
        }
        "weight-patch-rollback" => {
            let dir = PathBuf::from(next_argument(&mut args, "patched model folder")?);
            let output = PathBuf::from(next_argument(&mut args, "new restored model folder")?);
            let confirm = args.any(|a| a == "--confirm");
            if !confirm {
                return Err(invalid("weight-patch-rollback writes a new folder and requires --confirm"));
            }
            print(&rollback_weight_patch(&dir, &output)?)?;
            Ok(true)
        }
        "pipeline-plan" | "pipeline-run" => {
            let cfg = load_pipeline_config(&PathBuf::from(next_argument(&mut args, "ojon.pipeline.v1 config JSON")?))?;
            if cmd == "pipeline-plan" {
                print(&plan_pipeline(&cfg))?;
                return Ok(true);
            }
            if !args.any(|a| a == "--confirm") {
                return Err(invalid("pipeline-run writes models and runs tests; it requires --confirm"));
            }
            print(&run_pipeline(&cfg, |m| eprintln!("{m}"))?)?;
            Ok(true)
        }
        _ => Ok(false),
    }
}
