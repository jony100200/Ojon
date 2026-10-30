//! Compare any GGUF to its original with the same KL-divergence measurement the
//! recipe search uses: `kld-reference` saves the original's logits on a corpus
//! once, `kld-eval` scores one or more quantized files against that dump. This
//! is how an online quant and an Ojon recipe are put on the same scale: same
//! original, same held-out text, same context, same tool.

use crate::cli::common::{invalid, next_argument};
use ojon_backend::discovery::{resolve_backend_command, BackendId};
use ojon_backend::gguf::{inspect_gguf, inspect_sharded_gguf};
use ojon_backend::kld::{measure_kld, save_reference_logits, KldSettings};
use ojon_backend::recipe_run::plain;
use serde_json::json;
use std::ffi::OsString;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const KLD_EVAL_SCHEMA: &str = "ojon.kld_eval.v1";
pub const KLD_REFERENCE_SCHEMA: &str = "ojon.kld_reference.v1";
pub const KLD_EVAL_CLAIM: &str = "mean/p99/max KL divergence against the saved original logits on this corpus slice; a similarity measurement between two files, not a capability claim; run the task suite before any such claim";

fn text(v: OsString, what: &str) -> io::Result<String> {
    v.to_str().map(str::to_owned).ok_or_else(|| invalid(format!("{what} requires UTF-8")))
}

fn write_new(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut f = fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    f.write_all(bytes)
}

fn now_unix() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Shared measurement options; `--corpus` is required for both commands so the
/// receipt always names the text the numbers belong to.
struct Options {
    settings: KldSettings,
    json: bool,
    confirm: bool,
    out: Option<PathBuf>,
}

fn parse_options(args: &mut impl Iterator<Item = OsString>, what: &str, extra: &mut Vec<PathBuf>) -> io::Result<Options> {
    let mut settings = KldSettings {
        tool: resolve_backend_command(BackendId::LlamaPerplexity, "llama-perplexity")?,
        corpus: PathBuf::new(),
        gpu_layers: 99,
        fit: true,
        context: 512,
        chunks: 0,
        device: None,
    };
    let (mut json, mut confirm, mut out) = (false, false, None);
    while let Some(option) = args.next() {
        match option.to_str() {
            Some("--corpus") => settings.corpus = PathBuf::from(next_argument(args, "corpus text file")?),
            Some("--ctx") => settings.context = text(next_argument(args, "context tokens")?, "ctx")?.parse().map_err(|_| invalid("ctx must be a number"))?,
            Some("--chunks") => settings.chunks = text(next_argument(args, "chunk limit")?, "chunks")?.parse().map_err(|_| invalid("chunks must be a number"))?,
            Some("--ngl") => {
                let v = text(next_argument(args, "gpu layers or 'auto'")?, "ngl")?;
                if v == "auto" {
                    settings.fit = true;
                } else {
                    settings.fit = false;
                    settings.gpu_layers = v.parse().map_err(|_| invalid("ngl must be a number or 'auto'"))?;
                }
            }
            Some("--device") => settings.device = Some(text(next_argument(args, "backend device list, e.g. Vulkan0")?, "device")?),
            Some("--tool") => settings.tool = PathBuf::from(next_argument(args, "llama-perplexity executable")?),
            Some("--out") => out = Some(PathBuf::from(next_argument(args, "receipt JSON path")?)),
            Some("--json") => json = true,
            Some("--confirm") => confirm = true,
            Some(s) if !s.starts_with("--") => extra.push(PathBuf::from(s)),
            _ => return Err(invalid(format!("unknown {what} option: {}", option.to_string_lossy()))),
        }
    }
    if settings.corpus.as_os_str().is_empty() {
        return Err(invalid("--corpus <text file> is required (use the recipe run's corpus-eval.txt to compare against a recipe)"));
    }
    settings.validate()?;
    Ok(Options { settings, json, confirm, out })
}

pub fn handle(cmd: &str, mut args: impl Iterator<Item = OsString>) -> io::Result<bool> {
    match cmd {
        "kld-reference" => {
            let original = PathBuf::from(next_argument(&mut args, "original GGUF (F16/BF16/F32)")?);
            let dump = PathBuf::from(next_argument(&mut args, "new reference logits path")?);
            let mut extra = Vec::new();
            let opt = parse_options(&mut args, "kld-reference", &mut extra)?;
            if !extra.is_empty() {
                return Err(invalid(format!("unexpected argument {}", extra[0].display())));
            }
            let summary = inspect_gguf(&original).map_err(|e| invalid(format!("original {}: {e}", original.display())))?;
            if dump.exists() {
                return Err(invalid(format!("{} already exists; Ojon never overwrites", dump.display())));
            }
            let receipt_path = opt.out.clone().unwrap_or_else(|| dump.with_extension("logits.receipt.json"));
            println!("original: {} ({} bytes, file_type {:?})", plain(&original), summary.file_bytes, summary.file_type);
            println!("corpus: {} ({} bytes), ctx {}, chunks {}, {}", plain(&opt.settings.corpus), fs::metadata(&opt.settings.corpus)?.len(), opt.settings.context, if opt.settings.chunks == 0 { "all".to_string() } else { opt.settings.chunks.to_string() }, if opt.settings.fit { "--fit on".to_string() } else { format!("-ngl {}", opt.settings.gpu_layers) });
            println!("writes: {} (one float per vocab entry per token; large) and {}", plain(&dump), plain(&receipt_path));
            if !opt.confirm {
                println!("next_step: ojon kld-reference <same arguments> --confirm");
                return Ok(true);
            }
            let started = std::time::Instant::now();
            let mut log = |line: &str| eprintln!("{line}");
            let written = save_reference_logits(&original, &dump, &opt.settings, &mut log)?;
            let receipt = json!({
                "schema": KLD_REFERENCE_SCHEMA,
                "created_unix": now_unix(),
                "original": {"path": plain(&original), "bytes": summary.file_bytes, "file_type": summary.file_type, "architecture": summary.architecture},
                "corpus": {"path": plain(&opt.settings.corpus), "bytes": fs::metadata(&opt.settings.corpus)?.len()},
                "settings": {"context": opt.settings.context, "chunks": opt.settings.chunks, "fit": opt.settings.fit, "gpu_layers": opt.settings.gpu_layers, "device": opt.settings.device, "tool": plain(&opt.settings.tool)},
                "logits": {"path": plain(&written), "bytes": fs::metadata(&written).map(|m| m.len()).unwrap_or(0)},
                "seconds": started.elapsed().as_secs_f64(),
            });
            write_new(&receipt_path, serde_json::to_string_pretty(&receipt)?.as_bytes())?;
            if opt.json {
                println!("{}", serde_json::to_string_pretty(&receipt)?);
            } else {
                println!("reference: {} ({} bytes) in {:.1} s", plain(&written), receipt["logits"]["bytes"], started.elapsed().as_secs_f64());
                println!("receipt: {}", plain(&receipt_path));
            }
            Ok(true)
        }
        "kld-eval" => {
            // kld-eval <candidate.gguf> [more.gguf ...] --reference <logits> --corpus <same text> [...]
            let first = PathBuf::from(next_argument(&mut args, "candidate GGUF")?);
            let mut candidates = vec![first];
            let mut reference: Option<PathBuf> = None;
            let mut original_bytes: Option<u64> = None;
            // Pull --reference / --original out first; the rest is shared options.
            let mut rest: Vec<OsString> = Vec::new();
            let mut it = args;
            while let Some(a) = it.next() {
                match a.to_str() {
                    Some("--reference") => reference = Some(PathBuf::from(next_argument(&mut it, "reference logits path")?)),
                    Some("--original") => {
                        let p = PathBuf::from(next_argument(&mut it, "original GGUF, for size percentages")?);
                        original_bytes = Some(inspect_sharded_gguf(&p).map_err(|e| invalid(format!("original {}: {e}", p.display())))?.0.file_bytes);
                    }
                    _ => rest.push(a),
                }
            }
            let opt = parse_options(&mut rest.into_iter(), "kld-eval", &mut candidates)?;
            let reference = reference.ok_or_else(|| invalid("--reference <logits> is required (from kld-reference or a recipe run's reference.logits)"))?;
            if !reference.is_file() {
                return Err(invalid(format!("reference logits {} not found", reference.display())));
            }
            let mut rows = Vec::new();
            let mut log = |line: &str| eprintln!("{line}");
            for c in &candidates {
                let summary = inspect_gguf(c).map_err(|e| invalid(format!("candidate {}: {e}", c.display())))?;
                let stats = measure_kld(c, &reference, &opt.settings, &mut log)?;
                let pct = original_bytes.map(|o| 100.0 * summary.file_bytes as f64 / o as f64);
                if !opt.json {
                    println!(
                        "{}: {} bytes{} | mean KLD {:.6}{} | p99 {} | max {} | {:.1} s",
                        plain(c),
                        summary.file_bytes,
                        pct.map(|p| format!(" ({p:.2}% of original)")).unwrap_or_default(),
                        stats.mean,
                        stats.mean_uncertainty.map(|u| format!(" ± {u:.6}")).unwrap_or_default(),
                        stats.p99.map(|v| format!("{v:.6}")).unwrap_or_else(|| "-".into()),
                        stats.max.map(|v| format!("{v:.6}")).unwrap_or_else(|| "-".into()),
                        stats.seconds
                    );
                }
                rows.push(json!({
                    "path": plain(c), "bytes": summary.file_bytes, "file_type": summary.file_type, "percent_of_original": pct,
                    "kld": {"mean": stats.mean, "mean_uncertainty": stats.mean_uncertainty, "median": stats.median, "p99": stats.p99, "max": stats.max, "seconds": stats.seconds},
                }));
            }
            let receipt = json!({
                "schema": KLD_EVAL_SCHEMA,
                "created_unix": now_unix(),
                "reference_logits": plain(&reference),
                "original_bytes": original_bytes,
                "corpus": {"path": plain(&opt.settings.corpus), "bytes": fs::metadata(&opt.settings.corpus)?.len()},
                "settings": {"context": opt.settings.context, "chunks": opt.settings.chunks, "fit": opt.settings.fit, "gpu_layers": opt.settings.gpu_layers, "device": opt.settings.device, "tool": plain(&opt.settings.tool)},
                "candidates": rows,
                "claim": KLD_EVAL_CLAIM,
            });
            if let Some(out) = &opt.out {
                write_new(out, serde_json::to_string_pretty(&receipt)?.as_bytes())?;
                if !opt.json {
                    println!("receipt: {}", plain(out));
                }
            }
            if opt.json {
                println!("{}", serde_json::to_string_pretty(&receipt)?);
            } else {
                println!("claim: {KLD_EVAL_CLAIM}");
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}
