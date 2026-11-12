//! GGUF metadata and tensor repair commands (native replacements for the P-H3 Python scripts):
//!   gguf-set-kv <src.gguf> <out.gguf> key=value|key=@file.json ... --confirm
//!   gguf-restore-norms <src.gguf> <reference.safetensors> <out.gguf> [--confirm]
//!   gguf-verify-restore <original.gguf> <restored.gguf> <reference.safetensors>

use crate::cli::common::{invalid, next_argument};
use ojon_backend::quant::gguf_norms::{plan, restore, verify};
use ojon_backend::quant::gguf_rewrite::{rewrite, string_kv, Edits};
use serde_json::json;
use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::PathBuf;

fn path(args: &mut impl Iterator<Item = OsString>, what: &str) -> io::Result<PathBuf> {
    Ok(PathBuf::from(next_argument(args, what)?))
}

fn rest(args: impl Iterator<Item = OsString>) -> io::Result<(Vec<String>, bool)> {
    let mut items = Vec::new();
    let mut confirm = false;
    for a in args {
        let s = a.into_string().map_err(|_| invalid("arguments must be UTF-8"))?;
        if s == "--confirm" {
            confirm = true;
        } else {
            items.push(s);
        }
    }
    Ok((items, confirm))
}

pub fn handle(cmd: &str, mut args: impl Iterator<Item = OsString>) -> io::Result<bool> {
    match cmd {
        "gguf-set-kv" => {
            let (src, out) = (path(&mut args, "source GGUF")?, path(&mut args, "output GGUF")?);
            let (pairs, confirm) = rest(args)?;
            if pairs.is_empty() {
                return Err(invalid("give at least one key=value (or key=@file)"));
            }
            let mut edits = Edits::default();
            for p in &pairs {
                let (k, v) = p.split_once('=').ok_or_else(|| invalid(format!("expected key=value, got {p}")))?;
                let value = match v.strip_prefix('@') {
                    Some(file) => fs::read_to_string(file)?.trim().to_owned(),
                    None => v.to_owned(),
                };
                edits.kvs.push(string_kv(k, &value));
            }
            if !confirm {
                println!("{}", json!({"source": src, "output": out, "set": pairs, "run": "add --confirm to write"}));
                return Ok(true);
            }
            let bytes = rewrite(&src, &out, &edits)?;
            println!("{}", json!({"output": out, "output_bytes": bytes, "keys_set": edits.kvs.iter().map(|k| &k.key).collect::<Vec<_>>()}));
            Ok(true)
        }
        "gguf-restore-norms" => {
            let (src, reference, out) = (path(&mut args, "source GGUF")?, path(&mut args, "reference safetensors")?, path(&mut args, "output GGUF")?);
            let (_, confirm) = rest(args)?;
            if !confirm {
                let p = plan(&src, &reference)?;
                println!("{}", json!({"tensors": p.tensors, "restore_as_f32": p.restore.len(), "names": p.restore, "run": "add --confirm to write"}));
                return Ok(true);
            }
            let (p, bytes) = restore(&src, &reference, &out)?;
            println!("{}", json!({"output": out, "output_bytes": bytes, "tensors": p.tensors, "restored_as_f32": p.restore.len()}));
            Ok(true)
        }
        "gguf-verify-restore" => {
            let (orig, restored, reference) = (path(&mut args, "original GGUF")?, path(&mut args, "restored GGUF")?, path(&mut args, "reference safetensors")?);
            let v = verify(&orig, &restored, &reference)?;
            println!("{}", json!({"status": "VERIFIED", "tensors": v.tensors, "restored_exactly": v.restored_exactly,
                "unchanged_checked": v.unchanged_checked, "quantized_vectors_left": v.quantized_vectors_left}));
            Ok(true)
        }
        _ => Ok(false),
    }
}
