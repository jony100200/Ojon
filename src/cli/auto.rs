//! `auto-quant`: the one-command front door.
//!
//! Load an original, pick a preset, press. Everything below is the existing
//! measured pipeline (importance matrix from a hold-out split, per-class KLD
//! probes, greedy recipe, boundary refinement, final measurement, receipt)
//! with defaults chosen so the common case needs no flags. Every default is
//! printed in plan mode and written into the receipt, so "easy" never means
//! "hidden".
use crate::cli::common::{invalid, next_argument};
use ojon_backend::discovery::{resolve_backend_command, BackendId};
use ojon_backend::kld::KldSettings;
use ojon_backend::recipe_run::{plain, prepare_recipe, run_recipe, RecipeJob};
use std::ffi::OsString;
use std::io;
use std::path::PathBuf;

/// Presets are the three answers people actually give to "how much loss is ok?".
/// They set the base precision every class starts from, the lowest level the
/// search may try, and the mean / tail KLD budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Preset {
    Smallest,
    Balanced,
    Safest,
}

impl Preset {
    pub fn parse(s: &str) -> io::Result<Preset> {
        match s {
            "smallest" => Ok(Preset::Smallest),
            "balanced" => Ok(Preset::Balanced),
            "safest" => Ok(Preset::Safest),
            other => Err(invalid(format!("unknown preset '{other}'; use smallest, balanced or safest"))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Preset::Smallest => "smallest",
            Preset::Balanced => "balanced",
            Preset::Safest => "safest",
        }
    }

    /// (base, floor, mean KLD budget over base, p99 budget over base)
    pub fn knobs(self) -> (&'static str, &'static str, f64, f64) {
        match self {
            Preset::Smallest => ("q4_k", "iq2_s", 0.02, 0.08),
            Preset::Balanced => ("q5_k", "iq3_s", 0.01, 0.05),
            Preset::Safest => ("q6_k", "q4_k", 0.005, 0.03),
        }
    }

    pub fn meaning(self) -> &'static str {
        match self {
            Preset::Smallest => "smallest file that keeps the output distribution within a relaxed budget; expect visible changes on hard prompts",
            Preset::Balanced => "default: near-q5 quality at the smallest size the measurements allow",
            Preset::Safest => "q6 base, tight budget; for models you will not re-test downstream",
        }
    }
}

pub fn handle(cmd: &str, mut args: impl Iterator<Item = OsString>) -> io::Result<bool> {
    if cmd != "auto-quant" {
        return Ok(false);
    }
    let source = PathBuf::from(next_argument(&mut args, "original F16/BF16 GGUF model")?);
    let output_dir = PathBuf::from(next_argument(&mut args, "empty output directory")?);
    let mut preset = Preset::Balanced;
    let mut corpus: Option<PathBuf> = None;
    let mut eval_corpus: Option<PathBuf> = None;
    let mut chunks = 24u32;
    let mut threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4);
    let (mut confirm, mut resume, mut json) = (false, false, false);
    let text = |v: OsString, what: &str| -> io::Result<String> {
        v.to_str().map(str::to_owned).ok_or_else(|| invalid(format!("{what} requires UTF-8")))
    };
    while let Some(option) = args.next() {
        match option.to_str() {
            Some("--preset") => preset = Preset::parse(&text(next_argument(&mut args, "preset")?, "preset")?)?,
            Some("--corpus") => corpus = Some(PathBuf::from(next_argument(&mut args, "calibration text")?)),
            Some("--eval-corpus") => eval_corpus = Some(PathBuf::from(next_argument(&mut args, "held-out text")?)),
            Some("--chunks") => chunks = text(next_argument(&mut args, "chunks")?, "chunks")?.parse().map_err(|_| invalid("chunks must be a number"))?,
            Some("--threads") => threads = text(next_argument(&mut args, "threads")?, "threads")?.parse().map_err(|_| invalid("threads must be a number"))?,
            Some("--confirm") => confirm = true,
            Some("--resume") => resume = true,
            Some("--json") => json = true,
            _ => return Err(invalid(format!("unknown auto-quant option: {}", option.to_string_lossy()))),
        }
    }
    // Default corpus: the bundled calibration text. It is small; the plan says so
    // and points at corpus-build, because calibration text is the biggest lever.
    let corpus = corpus.unwrap_or_else(|| PathBuf::from("data").join("calibration.txt"));
    let (base, floor, budget, p99) = preset.knobs();
    let quantize_tool = resolve_backend_command(BackendId::LlamaQuantize, "llama-quantize")?;
    let perplexity_tool = resolve_backend_command(BackendId::LlamaPerplexity, "llama-perplexity")?;
    let imatrix_tool = resolve_backend_command(BackendId::LlamaImatrix, "llama-imatrix")?;

    let plan = prepare_recipe(RecipeJob {
        source,
        output_dir,
        base,
        floor,
        budget_kld: budget,
        quantize_tool,
        imatrix: None,
        imatrix_tool: Some(imatrix_tool),
        threads,
        kld: KldSettings { tool: perplexity_tool, corpus, gpu_layers: 99, fit: true, context: 512, chunks },
        keep_probes: false,
        budget_p99: Some(p99),
        resume,
        eval_corpus,
        refine_boundaries: true,
    })?;
    let plan_path = plan.write_plan()?;
    let plan_json = plan.plan_json();
    if json {
        let mut v = plan_json.clone();
        v["preset"] = serde_json::json!({"name": preset.name(), "meaning": preset.meaning()});
        println!("{}", serde_json::to_string_pretty(&v)?);
    } else {
        println!("auto-quant preset: {} ({})", preset.name(), preset.meaning());
        println!("  base {base}, floor {floor}, budget +{budget} mean KLD, +{p99} p99 KLD, imatrix auto, GPU fit auto, boundary refinement on");
        println!("  model: {} | classes {} | probes {} | quantizations {}", plain(&plan.job.source), plan.present.len(), plan.probes.len(), plan.probes.len() + 2 + 1);
        if let Some(w) = plan_json.get("corpus_warning").and_then(|v| v.as_str()) {
            println!("  warning: {w}; build a bigger one with: ojon corpus-build <out.txt> <your chats/docs/code>");
        }
        println!("  plan: {}", plain(&plan_path));
    }
    if !confirm {
        println!("next_step: add --confirm to run (one quantization per probe; hours for large MoE models; --resume continues a stopped run)");
        return Ok(true);
    }
    let receipt_path = run_recipe(&plan, |line| eprintln!("{line}"))?;
    let receipt: serde_json::Value = serde_json::from_slice(&std::fs::read(&receipt_path)?)?;
    let f = &receipt["final"];
    println!("receipt: {}", plain(&receipt_path));
    println!(
        "result: {} | {:.1}% of the original | mean KLD {} (base {}) | {}",
        f["path"].as_str().unwrap_or("?"),
        f["size_ratio_vs_source"].as_f64().unwrap_or(0.0) * 100.0,
        f["kld"]["mean"],
        receipt["base"]["kld"]["mean"],
        if f["measured_within_budget"].as_bool().unwrap_or(false) { "WITHIN BUDGET" } else { "OVER BUDGET: keep the base candidate" }
    );
    for key in ["base_advisory", "noise_advisory"] {
        if let Some(note) = receipt.get(key).and_then(|v| v.as_str()) {
            println!("advisory: {note}");
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn presets_are_ordered_from_loose_to_tight() {
        let (b1, f1, k1, p1) = Preset::Smallest.knobs();
        let (b2, f2, k2, p2) = Preset::Balanced.knobs();
        let (b3, f3, k3, p3) = Preset::Safest.knobs();
        assert!(k1 > k2 && k2 > k3 && p1 > p2 && p2 > p3);
        assert_eq!((b1, b2, b3), ("q4_k", "q5_k", "q6_k"));
        assert_eq!((f1, f2, f3), ("iq2_s", "iq3_s", "q4_k"));
        assert_eq!(Preset::parse("balanced").unwrap(), Preset::Balanced);
        assert!(Preset::parse("fastest").is_err());
    }
}
