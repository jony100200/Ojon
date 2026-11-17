//! `3d-plan` / `3d-quant`: image-to-3D flow DiT safetensors -> Q4_K_M / Q5_K GGUF for ggml loaders.

use crate::cli::common::{invalid, next_argument};
use ojon_backend::models::three_d::{describe, prepare, run, Overrides, ThreeDQuant};
use std::ffi::OsString;
use std::io;
use std::path::{Path, PathBuf};

const HELP: &str = "3d-plan|3d-quant <source.safetensors | dir> <output dir> --quant Q4_K_M|Q5_K|Q6_K [--threads N] [--keep-blocks 0,29] [--keep-kinds mlp.mlp.2,to_kv] [--only-blocks 7] [--tag -name] [--prefix model.img2shape_512] [--name stem] [--confirm]";

pub fn handle(cmd: &str, mut args: impl Iterator<Item = OsString>) -> io::Result<bool> {
    if cmd != "3d-plan" && cmd != "3d-quant" {
        return Ok(false);
    }
    let source = PathBuf::from(next_argument(&mut args, "source safetensors or model folder")?);
    let out_dir = PathBuf::from(next_argument(&mut args, "output directory")?);
    let (mut quant, mut threads, mut confirm) = (None, default_threads(), false);
    let mut ov = Overrides::default();
    while let Some(opt) = args.next() {
        match opt.to_str() {
            Some("--quant") => quant = Some(ThreeDQuant::from_name(&text(&mut args, "Q4_K_M, Q5_K or Q6_K")?)?),
            Some("--threads") => threads = text(&mut args, "thread count")?.parse().map_err(|_| invalid("--threads must be a number"))?,
            Some("--keep-blocks") => ov.keep_blocks = list(&text(&mut args, "block indices")?)?,
            Some("--only-blocks") => ov.only_blocks = Some(list(&text(&mut args, "block indices")?)?),
            Some("--keep-kinds") => ov.keep_kinds = text(&mut args, "tensor name suffixes")?.split(',').map(str::to_owned).collect(),
            Some("--prefix") => ov.prefix = text(&mut args, "tensor name prefix")?,
            Some("--name") => ov.name = Some(text(&mut args, "output file stem")?),
            Some("--tag") => ov.tag = text(&mut args, "file name tag")?,
            Some("--confirm") => confirm = true,
            _ => return Err(invalid(format!("unknown option {}; usage: {HELP}", opt.to_string_lossy()))),
        }
    }
    let quant = quant.ok_or_else(|| invalid(format!("--quant is required; usage: {HELP}")))?;
    if cmd == "3d-quant" && !confirm {
        return Err(invalid("3d-quant writes new files and requires --confirm"));
    }
    for file in flow_files(&source)? {
        let plan = prepare(&file, &out_dir, quant, threads, ov.clone())?;
        if cmd == "3d-plan" {
            println!("{}", serde_json::to_string_pretty(&describe(&plan)?)?);
        } else {
            std::fs::create_dir_all(&out_dir)?;
            let receipt = run(&plan, |m| eprintln!("{m}"))?;
            println!("{}", serde_json::to_string_pretty(&receipt)?);
        }
    }
    Ok(true)
}

/// A single file, or every `*_dit_*.safetensors` flow checkpoint in a model folder (or its `ckpts`).
fn flow_files(source: &Path) -> io::Result<Vec<PathBuf>> {
    if source.is_file() {
        return Ok(vec![source.to_path_buf()]);
    }
    let dir = if source.join("ckpts").is_dir() { source.join("ckpts") } else { source.to_path_buf() };
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors") && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.contains("_dit_")))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(invalid(format!("no *_dit_*.safetensors flow checkpoints in {}", dir.display())));
    }
    Ok(files)
}

fn text(args: &mut impl Iterator<Item = OsString>, what: &str) -> io::Result<String> {
    next_argument(args, what)?.into_string().map_err(|_| invalid(format!("{what} must be UTF-8")))
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get().saturating_sub(2).clamp(1, 6)).unwrap_or(2)
}

fn list(text: &str) -> io::Result<Vec<usize>> {
    text.split(',').map(|n| n.trim().parse().map_err(|_| invalid("block list must be comma-separated numbers"))).collect()
}
