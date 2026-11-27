//! EchoMimicV3-Flash model inspection and CPU-only weight compression.
use crate::cli::common::{invalid, next_argument};
use ojon_backend::models::echo::{self, Bits};
use std::ffi::OsString;
use std::io;
use std::path::PathBuf;

pub fn handle(cmd: &str, mut args: impl Iterator<Item = OsString>) -> io::Result<bool> {
    if !matches!(cmd, "echo-inspect" | "echo-quant-plan" | "echo-quant-run") {
        return Ok(false);
    }
    let root = PathBuf::from(next_argument(&mut args, "Echo Flash weights root")?);
    let destination = if cmd == "echo-inspect" {
        None
    } else {
        Some(PathBuf::from(next_argument(
            &mut args,
            "new output folder",
        )?))
    };
    let mut bits = Bits::Eight;
    let mut confirm = false;
    while let Some(arg) = args.next() {
        match arg.to_str() {
            Some("--bits") => {
                let text = next_argument(&mut args, "8 or 4")?;
                bits = Bits::parse(text.to_str().ok_or_else(|| invalid("bits must be UTF-8"))?)?;
            }
            Some("--confirm") => confirm = true,
            _ => {
                return Err(invalid(format!(
                    "unknown Echo option {}",
                    arg.to_string_lossy()
                )))
            }
        }
    }
    if confirm && cmd != "echo-quant-run" {
        return Err(invalid("--confirm is only for echo-quant-run"));
    }
    if cmd == "echo-quant-run" && !confirm {
        return Err(invalid("echo-quant-run requires --confirm"));
    }
    let plan = echo::inspect(&root, bits)?;
    if cmd == "echo-quant-run" {
        let output = destination.expect("run has output");
        let receipt = echo::run(&root, &output, bits)?;
        println!(
            "{}",
            serde_json::json!({"status":receipt["status"],"manifest":output.join("echo-manifest.json"),
            "runtime_verified":false,"quality_verified":false})
        );
    } else {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "plan":plan,"output":destination,"next":"echo-quant-run <flash-root> <new-output-dir> --bits 8 --confirm"
            }))?
        );
    }
    Ok(true)
}
