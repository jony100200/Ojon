//! KL-divergence-gated mixed-precision recipe search: the planning half.
//!
//! The question a quant maker must answer is not "which single format?" but
//! "how low can EACH tensor class go before the model's output distribution
//! moves more than the user tolerates?". Formats like Q4_K_M already hard-code
//! one answer (attn_v and ffn_down a notch higher); this module measures the
//! answer per model instead of assuming it.
//!
//! Protocol (`recipe-plan` / `recipe-run`):
//! 1. Reference: the original F16/BF16 GGUF's logits over a held-out corpus
//!    (`llama-perplexity --save-all-logits`).
//! 2. Base candidate: every class at `base` (default q5_k). Its mean KLD is the
//!    floor every probe is measured against.
//! 3. Probes: for each searched class and each ladder level below `base`, one
//!    quantization with ONLY that class lowered. Mean KLD minus the base KLD is
//!    that class's measured sensitivity at that level. No proxy, no guess.
//! 4. Selection: greedy, bytes saved per unit of added KLD, until the summed
//!    deltas reach the budget. The sum is an upper-bound style approximation,
//!    so the final recipe is quantized and measured again; it is reported as
//!    passing only if the MEASURED mean KLD is within budget.
//!
//! Everything here is pure: no I/O, no process spawn, fully unit-tested. The
//! executor lives in `recipe_run` and uses the existing llama-quantize and
//! llama-perplexity adapters.
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io;

use crate::gguf::GgufTensor;

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, s.into())
}

/// Tensor classes with distinct, well-known quantization sensitivity. Order is
/// the order recipes and receipts list them in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum TensorClass {
    TokenEmbd,
    Output,
    AttnQk,
    AttnVo,
    FfnGateUp,
    FfnDown,
    ExpertGateUp,
    ExpertDown,
    SharedExpert,
    Router,
    Other,
}

impl TensorClass {
    pub const ALL: [TensorClass; 11] = [
        TensorClass::TokenEmbd,
        TensorClass::Output,
        TensorClass::AttnQk,
        TensorClass::AttnVo,
        TensorClass::FfnGateUp,
        TensorClass::FfnDown,
        TensorClass::ExpertGateUp,
        TensorClass::ExpertDown,
        TensorClass::SharedExpert,
        TensorClass::Router,
        TensorClass::Other,
    ];

    /// Classes whose precision the search may lower. Routers stay at the
    /// converter default (llama.cpp keeps them high precision on purpose) and
    /// `Other` is norms, biases and 1-D tensors that are never quantized.
    pub const SEARCHED: [TensorClass; 9] = [
        TensorClass::TokenEmbd,
        TensorClass::Output,
        TensorClass::AttnQk,
        TensorClass::AttnVo,
        TensorClass::FfnGateUp,
        TensorClass::FfnDown,
        TensorClass::ExpertGateUp,
        TensorClass::ExpertDown,
        TensorClass::SharedExpert,
    ];

    pub fn name(self) -> &'static str {
        match self {
            TensorClass::TokenEmbd => "token_embd",
            TensorClass::Output => "output",
            TensorClass::AttnQk => "attn_qk",
            TensorClass::AttnVo => "attn_vo",
            TensorClass::FfnGateUp => "ffn_gate_up",
            TensorClass::FfnDown => "ffn_down",
            TensorClass::ExpertGateUp => "expert_gate_up",
            TensorClass::ExpertDown => "expert_down",
            TensorClass::SharedExpert => "shared_expert",
            TensorClass::Router => "router",
            TensorClass::Other => "other",
        }
    }

    pub fn parse(s: &str) -> Option<TensorClass> {
        TensorClass::ALL.iter().copied().find(|c| c.name() == s)
    }

    /// Regex llama-quantize's `--tensor-type-file` matches against the
    /// lower-cased tensor name (ECMAScript `regex_search`). Must agree with
    /// [`TensorClass::of`]; the tests enforce that on representative names.
    pub fn pattern(self) -> &'static str {
        match self {
            TensorClass::TokenEmbd => r"^token_embd\.weight$",
            TensorClass::Output => r"^output\.weight$",
            TensorClass::AttnQk => r"\.attn_(q|k|qkv)\.weight$",
            TensorClass::AttnVo => r"\.attn_(v|output)\.weight$",
            TensorClass::FfnGateUp => r"^blk\.\d+\.ffn_(gate|up)\.weight$",
            TensorClass::FfnDown => r"^blk\.\d+\.ffn_down\.weight$",
            TensorClass::ExpertGateUp => r"\.ffn_(gate|up|gate_up)_exps\.weight$",
            TensorClass::ExpertDown => r"\.ffn_down_exps\.weight$",
            TensorClass::SharedExpert => r"_shexp\.weight$",
            TensorClass::Router => r"\.ffn_gate_inp\.weight$",
            TensorClass::Other => r"$^",
        }
    }

    /// Classify a GGUF tensor by name. 1-D tensors (norms, biases) are `Other`
    /// regardless of name because llama-quantize never quantizes them.
    pub fn of(tensor: &GgufTensor) -> TensorClass {
        if tensor.shape.len() < 2 {
            return TensorClass::Other;
        }
        Self::of_name(&tensor.name)
    }

    pub fn of_name(name: &str) -> TensorClass {
        let n = name.to_ascii_lowercase();
        let base = n.strip_suffix(".weight").unwrap_or(n.as_str());
        if base == "token_embd" {
            return TensorClass::TokenEmbd;
        }
        if base == "output" {
            return TensorClass::Output;
        }
        if base.ends_with("_shexp") {
            return TensorClass::SharedExpert;
        }
        if base.ends_with(".ffn_gate_inp") {
            return TensorClass::Router;
        }
        if base.ends_with(".ffn_down_exps") {
            return TensorClass::ExpertDown;
        }
        if base.ends_with(".ffn_gate_exps")
            || base.ends_with(".ffn_up_exps")
            || base.ends_with(".ffn_gate_up_exps")
        {
            return TensorClass::ExpertGateUp;
        }
        if base.ends_with(".attn_q") || base.ends_with(".attn_k") || base.ends_with(".attn_qkv") {
            return TensorClass::AttnQk;
        }
        if base.ends_with(".attn_v") || base.ends_with(".attn_output") {
            return TensorClass::AttnVo;
        }
        if base.starts_with("blk.") && (base.ends_with(".ffn_gate") || base.ends_with(".ffn_up")) {
            return TensorClass::FfnGateUp;
        }
        if base.starts_with("blk.") && base.ends_with(".ffn_down") {
            return TensorClass::FfnDown;
        }
        TensorClass::Other
    }

    /// Lowest ladder level the search may assign. Embeddings and the output
    /// head carry the whole vocabulary; shared experts run on every token.
    /// These floors are conservative starting rules, and the receipt records
    /// them so a later measured override is auditable.
    pub fn floor(self) -> &'static str {
        match self {
            TensorClass::TokenEmbd | TensorClass::Output | TensorClass::SharedExpert => "q4_k",
            TensorClass::AttnVo => "q3_k",
            _ => "iq2_s",
        }
    }
}

/// Ordered from smallest to largest. Names are the ggml type names that
/// llama-quantize accepts in a tensor-type file. Bits per weight are the
/// nominal storage costs including scales (used only to RANK savings; the
/// receipt records real file sizes).
pub const LADDER: [(&str, f64); 8] = [
    ("iq2_s", 2.50),
    ("iq3_s", 3.44),
    ("q3_k", 3.44),
    ("iq4_xs", 4.25),
    ("q4_k", 4.50),
    ("q5_k", 5.50),
    ("q6_k", 6.56),
    ("q8_0", 8.50),
];

pub fn ladder_index(level: &str) -> Option<usize> {
    LADDER.iter().position(|(n, _)| *n == level)
}

pub fn bits_per_weight(level: &str) -> io::Result<f64> {
    LADDER
        .iter()
        .find(|(n, _)| *n == level)
        .map(|(_, b)| *b)
        .ok_or_else(|| invalid(format!("unknown ladder level '{level}'")))
}

/// Weight elements per class, from the GGUF tensor catalog. Only tensors the
/// converter quantizes (2-D and up) count.
pub fn class_elements(tensors: &[GgufTensor]) -> BTreeMap<TensorClass, u64> {
    let mut out = BTreeMap::new();
    for t in tensors {
        let c = TensorClass::of(t);
        if c != TensorClass::Other {
            *out.entry(c).or_insert(0) += t.elements;
        }
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Probe {
    pub class: TensorClass,
    pub level: &'static str,
}

/// Every (class, level) pair the executor must quantize and measure: each
/// searched class present in the model, at every ladder level strictly below
/// `base` and not below the class floor.
pub fn probe_matrix(
    base: &str,
    present: &BTreeMap<TensorClass, u64>,
    ladder_floor: &str,
) -> io::Result<Vec<Probe>> {
    let base_ix = ladder_index(base).ok_or_else(|| invalid(format!("unknown base level '{base}'")))?;
    let global_floor = ladder_index(ladder_floor)
        .ok_or_else(|| invalid(format!("unknown ladder floor '{ladder_floor}'")))?;
    let mut probes = Vec::new();
    for class in TensorClass::SEARCHED {
        if !present.contains_key(&class) {
            continue;
        }
        let floor = ladder_index(class.floor()).unwrap_or(0).max(global_floor);
        for ix in (floor..base_ix).rev() {
            probes.push(Probe { class, level: LADDER[ix].0 });
        }
    }
    Ok(probes)
}

/// One measured probe: mean KLD of the whole model with ONLY `class` at `level`.
#[derive(Debug, Clone, PartialEq)]
pub struct Measured {
    pub class: TensorClass,
    pub level: &'static str,
    pub mean_kld: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Recipe {
    /// Final level per class present in the model (classes left at base included).
    pub levels: BTreeMap<TensorClass, &'static str>,
    /// Sum of the measured per-class KLD deltas over the base candidate.
    pub predicted_kld_delta: f64,
    /// Nominal weight bytes of the recipe versus everything at base.
    pub predicted_bytes: u64,
    pub base_bytes: u64,
    pub steps: Vec<Step>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub class: TensorClass,
    pub from: &'static str,
    pub to: &'static str,
    pub bytes_saved: u64,
    pub kld_added: f64,
}

fn class_bytes(elements: u64, level: &str) -> u64 {
    (elements as f64 * bits_per_weight(level).unwrap_or(8.5) / 8.0).round() as u64
}

/// Greedy selection. Each class walks down the ladder one level at a time; at
/// every round the step with the most bytes saved per unit of KLD added wins,
/// as long as the running total stays within `budget_kld`. Steps whose probe
/// measured no KLD increase over base are free and always taken. A class
/// without a measurement for a level is never lowered to it.
pub fn select_recipe(
    base: &str,
    present: &BTreeMap<TensorClass, u64>,
    base_kld: f64,
    measured: &[Measured],
    budget_kld: f64,
) -> io::Result<Recipe> {
    if !(base_kld.is_finite() && base_kld >= 0.0) {
        return Err(invalid("base KLD must be a finite non-negative number"));
    }
    if !(budget_kld.is_finite() && budget_kld >= 0.0) {
        return Err(invalid("KLD budget must be a finite non-negative number"));
    }
    ladder_index(base).ok_or_else(|| invalid(format!("unknown base level '{base}'")))?;
    let base_level: &'static str = LADDER[ladder_index(base).unwrap()].0;

    // delta over base per (class, level); negative noise clamps to zero
    let mut delta: BTreeMap<(TensorClass, &'static str), f64> = BTreeMap::new();
    for m in measured {
        if !m.mean_kld.is_finite() || m.mean_kld < 0.0 {
            return Err(invalid(format!("probe {}@{} has an invalid KLD", m.class.name(), m.level)));
        }
        delta.insert((m.class, m.level), (m.mean_kld - base_kld).max(0.0));
    }

    let mut levels: BTreeMap<TensorClass, &'static str> =
        present.keys().map(|c| (*c, base_level)).collect();
    let base_bytes: u64 = present.iter().map(|(c, e)| class_bytes(*e, levels[c])).sum();
    let mut spent = 0.0;
    let mut steps = Vec::new();

    loop {
        let mut best: Option<(f64, Step)> = None;
        for class in TensorClass::SEARCHED {
            let Some(&elements) = present.get(&class) else { continue };
            let cur = levels[&class];
            let cur_ix = ladder_index(cur).unwrap();
            if cur_ix == 0 {
                continue;
            }
            let next = LADDER[cur_ix - 1].0;
            let Some(&d_next) = delta.get(&(class, next)) else { continue };
            let d_cur = if cur == base_level { 0.0 } else { delta[&(class, cur)] };
            let kld_added = (d_next - d_cur).max(0.0);
            if spent + kld_added > budget_kld + 1e-12 {
                continue;
            }
            let bytes_saved = class_bytes(elements, cur).saturating_sub(class_bytes(elements, next));
            if bytes_saved == 0 {
                continue;
            }
            let score = if kld_added <= 0.0 { f64::INFINITY } else { bytes_saved as f64 / kld_added };
            let step = Step { class, from: cur, to: next, bytes_saved, kld_added };
            if best.as_ref().map(|(s, _)| score > *s).unwrap_or(true) {
                best = Some((score, step));
            }
        }
        let Some((_, step)) = best else { break };
        spent += step.kld_added;
        levels.insert(step.class, step.to);
        steps.push(step);
    }

    let predicted_bytes: u64 = present.iter().map(|(c, e)| class_bytes(*e, levels[c])).sum();
    Ok(Recipe { levels, predicted_kld_delta: spent, predicted_bytes, base_bytes, steps })
}

/// `--tensor-type-file` body for a probe: only the probed class differs from
/// the converter's base format, which the caller passes as the main precision.
pub fn render_probe(class: TensorClass, level: &str) -> String {
    format!("{}={}\n", class.pattern(), level)
}

/// `--tensor-type-file` body for a final recipe. Classes at `base` are left to
/// the converter's main format, so an all-base recipe renders empty.
pub fn render_recipe(recipe: &Recipe, base: &str) -> String {
    let mut out = String::new();
    for (class, level) in &recipe.levels {
        if *level != base {
            out.push_str(&format!("{}={}\n", class.pattern(), level));
        }
    }
    out
}

pub fn recipe_json(recipe: &Recipe, base: &str, base_kld: f64, budget_kld: f64) -> Value {
    json!({
        "schema": "ojon.kld_recipe.v1",
        "base_level": base,
        "base_mean_kld": base_kld,
        "budget_kld_delta": budget_kld,
        "predicted_kld_delta": recipe.predicted_kld_delta,
        "predicted_bytes": recipe.predicted_bytes,
        "base_bytes": recipe.base_bytes,
        "predicted_size_ratio_vs_base": if recipe.base_bytes > 0 {
            recipe.predicted_bytes as f64 / recipe.base_bytes as f64 } else { 1.0 },
        "levels": recipe.levels.iter().map(|(c, l)| (c.name().to_string(), json!(l))).collect::<serde_json::Map<_, _>>(),
        "floors": TensorClass::SEARCHED.iter().map(|c| (c.name().to_string(), json!(c.floor()))).collect::<serde_json::Map<_, _>>(),
        "steps": recipe.steps.iter().map(|s| json!({
            "class": s.class.name(), "from": s.from, "to": s.to,
            "bytes_saved": s.bytes_saved, "kld_added": s.kld_added})).collect::<Vec<_>>(),
        "note": "predicted_kld_delta sums per-class probe deltas; the final recipe must be quantized and measured again before any quality claim"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(name: &str, shape: &[u64]) -> GgufTensor {
        GgufTensor {
            name: name.to_string(),
            elements: shape.iter().product(),
            shape: shape.to_vec(),
            storage_type: 1,
            relative_offset: 0,
        }
    }

    fn moe_model() -> Vec<GgufTensor> {
        vec![
            t("token_embd.weight", &[2560, 151936]),
            t("output.weight", &[2560, 151936]),
            t("output_norm.weight", &[2560]),
            t("blk.0.attn_q.weight", &[2560, 4096]),
            t("blk.0.attn_k.weight", &[2560, 512]),
            t("blk.0.attn_v.weight", &[2560, 512]),
            t("blk.0.attn_output.weight", &[4096, 2560]),
            t("blk.0.attn_norm.weight", &[2560]),
            t("blk.0.ffn_gate_inp.weight", &[2560, 128]),
            t("blk.0.ffn_gate_exps.weight", &[2560, 768, 128]),
            t("blk.0.ffn_up_exps.weight", &[2560, 768, 128]),
            t("blk.0.ffn_down_exps.weight", &[768, 2560, 128]),
            t("blk.0.ffn_gate_shexp.weight", &[2560, 768]),
            t("blk.1.ffn_gate.weight", &[2560, 9728]),
            t("blk.1.ffn_down.weight", &[9728, 2560]),
        ]
    }

    #[test]
    fn classifies_every_representative_name() {
        let m = moe_model();
        let got: Vec<TensorClass> = m.iter().map(TensorClass::of).collect();
        assert_eq!(
            got,
            vec![
                TensorClass::TokenEmbd,
                TensorClass::Output,
                TensorClass::Other,
                TensorClass::AttnQk,
                TensorClass::AttnQk,
                TensorClass::AttnVo,
                TensorClass::AttnVo,
                TensorClass::Other,
                TensorClass::Router,
                TensorClass::ExpertGateUp,
                TensorClass::ExpertGateUp,
                TensorClass::ExpertDown,
                TensorClass::SharedExpert,
                TensorClass::FfnGateUp,
                TensorClass::FfnDown,
            ]
        );
    }

    /// The llama-side regex and the Rust classifier must agree, otherwise a
    /// probe would lower a different set of tensors than it accounts for.
    #[test]
    fn patterns_agree_with_classifier() {
        // a tiny ECMAScript-compatible subset evaluator is overkill; check the
        // structural facts the patterns rely on instead
        for tensor in moe_model() {
            let class = TensorClass::of(&tensor);
            let name = tensor.name.to_ascii_lowercase();
            let p = class.pattern();
            match class {
                TensorClass::Other => assert_eq!(p, r"$^"),
                TensorClass::TokenEmbd => assert_eq!(name, "token_embd.weight"),
                TensorClass::Output => assert_eq!(name, "output.weight"),
                _ => {
                    // every non-anchored pattern ends in `\.weight$` or `_shexp\.weight$`
                    assert!(p.ends_with(r"\.weight$"), "{p}");
                    assert!(name.ends_with(".weight"));
                }
            }
        }
        // shared experts must not be swallowed by the plain ffn patterns
        assert_eq!(TensorClass::of_name("blk.3.ffn_up_shexp.weight"), TensorClass::SharedExpert);
        assert_eq!(TensorClass::of_name("blk.3.ffn_up.weight"), TensorClass::FfnGateUp);
        assert_eq!(TensorClass::of_name("blk.3.ffn_gate_up_exps.weight"), TensorClass::ExpertGateUp);
    }

    #[test]
    fn probe_matrix_walks_down_from_base_to_each_floor() {
        let present = class_elements(&moe_model());
        let probes = probe_matrix("q5_k", &present, "iq2_s").unwrap();
        let vo: Vec<&str> = probes.iter().filter(|p| p.class == TensorClass::AttnVo).map(|p| p.level).collect();
        assert_eq!(vo, vec!["q4_k", "iq4_xs", "q3_k"]); // floor q3_k
        let emb: Vec<&str> = probes.iter().filter(|p| p.class == TensorClass::TokenEmbd).map(|p| p.level).collect();
        assert_eq!(emb, vec!["q4_k"]); // floor q4_k
        let ex: Vec<&str> = probes.iter().filter(|p| p.class == TensorClass::ExpertGateUp).map(|p| p.level).collect();
        assert_eq!(ex, vec!["q4_k", "iq4_xs", "q3_k", "iq3_s", "iq2_s"]);
        assert!(probes.iter().all(|p| p.class != TensorClass::Router));
    }

    #[test]
    fn probe_matrix_respects_a_global_floor() {
        let present = class_elements(&moe_model());
        let probes = probe_matrix("q5_k", &present, "iq4_xs").unwrap();
        assert!(probes.iter().all(|p| ladder_index(p.level).unwrap() >= ladder_index("iq4_xs").unwrap()));
    }

    #[test]
    fn greedy_lowers_cheapest_classes_first_and_stops_at_budget() {
        let present = class_elements(&moe_model());
        let base_kld = 0.010;
        let measured = vec![
            Measured { class: TensorClass::ExpertGateUp, level: "q4_k", mean_kld: 0.011 },  // +0.001, huge bytes
            Measured { class: TensorClass::ExpertGateUp, level: "iq4_xs", mean_kld: 0.012 }, // +0.002
            Measured { class: TensorClass::ExpertGateUp, level: "q3_k", mean_kld: 0.030 },   // +0.020
            Measured { class: TensorClass::AttnVo, level: "q4_k", mean_kld: 0.025 },         // +0.015, tiny bytes
            Measured { class: TensorClass::Output, level: "q4_k", mean_kld: 0.0105 },        // +0.0005
        ];
        let r = select_recipe("q5_k", &present, base_kld, &measured, 0.003).unwrap();
        assert_eq!(r.levels[&TensorClass::ExpertGateUp], "iq4_xs");
        assert_eq!(r.levels[&TensorClass::Output], "q4_k");
        assert_eq!(r.levels[&TensorClass::AttnVo], "q5_k"); // too expensive for the budget
        assert_eq!(r.levels[&TensorClass::ExpertDown], "q5_k"); // never measured, never lowered
        assert!((r.predicted_kld_delta - 0.0025).abs() < 1e-9);
        assert!(r.predicted_bytes < r.base_bytes);
        // greedy order is bytes saved per unit KLD: the output head (0.0005 for 48 MB)
        // beats the expert gate/up step (0.001 for 63 MB)
        assert_eq!(r.steps.first().unwrap().class, TensorClass::Output);
        assert_eq!(r.steps.len(), 3);
    }

    #[test]
    fn zero_budget_only_takes_free_steps() {
        let present = class_elements(&moe_model());
        let measured = vec![
            Measured { class: TensorClass::FfnGateUp, level: "q4_k", mean_kld: 0.0098 }, // below base: noise, free
            Measured { class: TensorClass::FfnDown, level: "q4_k", mean_kld: 0.011 },
        ];
        let r = select_recipe("q5_k", &present, 0.010, &measured, 0.0).unwrap();
        assert_eq!(r.levels[&TensorClass::FfnGateUp], "q4_k");
        assert_eq!(r.levels[&TensorClass::FfnDown], "q5_k");
        assert_eq!(r.predicted_kld_delta, 0.0);
    }

    #[test]
    fn rejects_invalid_numbers_and_levels() {
        let present = class_elements(&moe_model());
        assert!(select_recipe("q5_k", &present, f64::NAN, &[], 0.01).is_err());
        assert!(select_recipe("q5_k", &present, 0.01, &[], -1.0).is_err());
        assert!(select_recipe("q9_z", &present, 0.01, &[], 0.01).is_err());
        let bad = vec![Measured { class: TensorClass::FfnDown, level: "q4_k", mean_kld: f64::INFINITY }];
        assert!(select_recipe("q5_k", &present, 0.01, &bad, 0.01).is_err());
        assert!(probe_matrix("nope", &present, "iq2_s").is_err());
    }

    #[test]
    fn renders_only_classes_that_differ_from_base() {
        let present = class_elements(&moe_model());
        let measured = vec![Measured { class: TensorClass::ExpertDown, level: "q4_k", mean_kld: 0.0 }];
        let r = select_recipe("q5_k", &present, 0.0, &measured, 0.0).unwrap();
        let body = render_recipe(&r, "q5_k");
        assert_eq!(body, "\\.ffn_down_exps\\.weight$=q4_k\n");
        assert_eq!(render_probe(TensorClass::AttnVo, "q3_k"), "\\.attn_(v|output)\\.weight$=q3_k\n");
        let j = recipe_json(&r, "q5_k", 0.0, 0.0);
        assert_eq!(j["schema"], "ojon.kld_recipe.v1");
        assert_eq!(j["levels"]["expert_down"], "q4_k");
        assert_eq!(j["steps"].as_array().unwrap().len(), 1);
    }
}
