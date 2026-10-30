//! Adaptive precision search. Real GGUF quantization + held-out task evaluation.
//! Tensor sensitivities are externally measured inputs and must be auditable;
//! no heuristic guess is described as observed activation sensitivity.
use crate::gguf::{inspect_gguf, inspect_gguf_tensors, read_imatrix_energy};
use crate::llama::{prepare_gguf, quantize_gguf_with_log, Precision};
use crate::task_eval::{prepare_eval_with_server, run_eval_reusing_baseline, EvalRequest};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{self, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, s.into())
}

const TYPES: [&str; 5] = ["q2_k", "q3_k", "q4_k", "q5_k", "q6_k"];
const BPW: [f64; 5] = [2.5625, 3.4375, 4.5, 5.5, 6.5625];

#[derive(Clone, Debug)]
pub struct TensorScore {
    pub name: String,
    pub elements: u64,
    pub sensitivity: f64,
}

#[derive(Clone, Debug)]
pub struct AdaptivePlan {
    pub source: PathBuf,
    pub scores_file: PathBuf,
    pub scores: Vec<TensorScore>,
    pub output_dir: PathBuf,
    pub suite: PathBuf,
    pub quantizer: PathBuf,
    pub eval_tool: PathBuf,
    pub server_tool: Option<PathBuf>,
    pub imatrix: Option<PathBuf>,
    pub targets: Vec<f64>,
    pub threads: usize,
    pub gpu_layers: i32,
    pub device: Option<String>,
    pub tokens: u32,
    pub max_absolute_loss_pct: f64,
    pub reference: PathBuf,
}

/// Derive a ranked importance proxy directly from a real GGUF llama-imatrix.
/// No source weights are loaded; this is an activation-energy proxy, not actual
/// perturbation sensitivity. Every candidate still needs held-out task tests.
pub fn extract_sensitivity_from_imatrix(
    source: &Path,
    imatrix: &Path,
    output: &Path,
) -> io::Result<Value> {
    let source = source.canonicalize()?;
    let info = inspect_gguf(&source)?;
    if !matches!(info.file_type, Some(0 | 1 | 32)) {
        return Err(invalid(
            "sensitivity source must be original F32/F16/BF16 GGUF",
        ));
    }
    let energy = read_imatrix_energy(imatrix)?;
    let by_name: HashMap<String, f64> = energy.into_iter().collect();
    let catalog = inspect_gguf_tensors(&source)?;
    let mut rows = Vec::new();
    for t in catalog {
        if t.shape.len() < 2
            || t.elements < 65536
            || t.shape[0] % 256 != 0
            || !t.name.ends_with(".weight")
            || !t.name.starts_with("blk.")
        {
            continue;
        }
        if let Some(&score) = by_name.get(&t.name) {
            rows.push((t.name, score));
        }
    }
    if rows.is_empty() {
        return Err(invalid(
            "imatrix has no matched K-quantizable source tensors",
        ));
    }
    // Stable rank-based normalization avoids pretending raw activation units
    // from unrelated tensor families share one calibrated loss scale.
    rows.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let n = rows.len();
    let tensors = rows
        .into_iter()
        .enumerate()
        .map(|(idx, (name, energy))| {
            let scaled = (idx + 1) as f64 / n as f64;
            json!({"name":name,"sensitivity":scaled,"activation_energy_proxy":energy})
        })
        .collect::<Vec<_>>();
    let result = json!({
        "schema":"ojon.tensor_sensitivity.v1",
        "source_file_bytes":info.file_bytes,
        "architecture":info.architecture,
        "calibration_protocol":"GGUF llama-imatrix mean squared activation-energy RANK proxy; not direct perturbation sensitivity",
        "calibration_imatrix":imatrix.canonicalize()?.to_string_lossy(),
        "verified_task_sensitivity":false,
        "tensors":tensors
    });
    if output.extension().and_then(|x| x.to_str()) != Some("json") {
        return Err(invalid("sensitivity output must be a new JSON file"));
    }
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)?;
    file.write_all(&serde_json::to_vec_pretty(&result).map_err(io::Error::other)?)?;
    file.sync_all()?;
    Ok(result)
}

pub fn parse_tensor_sensitivity(source: &Path, file: &Path) -> io::Result<Vec<TensorScore>> {
    let source = source.canonicalize()?;
    let base = inspect_gguf(&source)?;
    if !matches!(base.file_type, Some(0 | 1 | 32)) {
        return Err(invalid(
            "adaptive source must be original F32/F16/BF16 GGUF",
        ));
    }
    let size = fs::metadata(file)?.len();
    if size == 0 || size > 4 * 1024 * 1024 {
        return Err(invalid("sensitivity file must be 1 byte..4 MiB"));
    }
    let raw: Value = serde_json::from_slice(&fs::read(file)?).map_err(io::Error::other)?;
    if raw["schema"] != "ojon.tensor_sensitivity.v1" {
        return Err(invalid(
            "tensor sensitivity schema must be ojon.tensor_sensitivity.v1",
        ));
    }
    if raw["source_file_bytes"].as_u64() != Some(base.file_bytes)
        || raw["architecture"].as_str() != base.architecture.as_deref()
    {
        return Err(invalid(
            "sensitivity evidence does not match original model bytes/architecture",
        ));
    }
    if raw["calibration_protocol"]
        .as_str()
        .is_none_or(|s| s.trim().is_empty())
    {
        return Err(invalid(
            "sensitivity provenance requires calibration_protocol",
        ));
    }
    let tensors = raw["tensors"]
        .as_array()
        .filter(|a| !a.is_empty() && a.len() <= 4096)
        .ok_or_else(|| invalid("require 1..=4096 sensitivity entries"))?;
    let catalog = inspect_gguf_tensors(&source)?;
    let allowed: HashMap<String, (u64, bool)> = catalog
        .into_iter()
        .map(|t| {
            let eligible = t.shape.len() >= 2
                && t.elements >= 65536
                && t.shape[0] % 256 == 0
                && t.name.ends_with(".weight")
                && t.name.starts_with("blk.");
            (t.name, (t.elements, eligible))
        })
        .collect();
    let mut seen = HashSet::new();
    let mut scores = Vec::new();
    for item in tensors {
        let name = item["name"]
            .as_str()
            .ok_or_else(|| invalid("tensor name missing"))?;
        if !seen.insert(name.to_owned()) {
            return Err(invalid("duplicate sensitivity entry"));
        }
        let &(elements, eligible) = allowed
            .get(name)
            .ok_or_else(|| invalid(format!("tensor {name} does not exist")))?;
        if !eligible {
            return Err(invalid(format!(
                "tensor {name} is not safe for K-quant override"
            )));
        }
        let sensitivity = item["sensitivity"]
            .as_f64()
            .filter(|v| v.is_finite() && *v > 0.0 && *v <= 1.0)
            .ok_or_else(|| invalid("sensitivity must be finite and in (0,1]"))?;
        scores.push(TensorScore {
            name: name.into(),
            elements,
            sensitivity,
        });
    }
    Ok(scores)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationStrategy {
    GreedySensitivity,
    RcoLagrangian,
    LearnedRoleAware,
}

impl AllocationStrategy {
    pub fn parse(s: &str) -> io::Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "greedy" | "greedy_sensitivity" => Ok(Self::GreedySensitivity),
            "rco" | "rco_lagrangian" | "lagrangian" => Ok(Self::RcoLagrangian),
            "learned" | "learned_role_aware" | "shapelearn" => Ok(Self::LearnedRoleAware),
            _ => Err(invalid("strategy must be greedy, rco, or learned")),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::GreedySensitivity => "greedy_sensitivity",
            Self::RcoLagrangian => "rco_lagrangian",
            Self::LearnedRoleAware => "learned_role_aware",
        }
    }
}

fn tensor_role_multiplier(name: &str, max_block: usize) -> (f64, usize) {
    let mut mult = 1.0;
    let mut min_level = 0usize; // 0 = q2_k, 1 = q3_k, 2 = q4_k, 3 = q5_k, 4 = q6_k
    if name.contains("_shexp") || name.contains("shared_expert") {
        mult *= 2.4;
        min_level = 2; // at least q4_k when budget allows
    } else if name.contains("attn_v") || name.contains("attn_output") || name.contains("ssm_out") {
        mult *= 1.85;
        min_level = 1;
    } else if name.contains("attn_q") || name.contains("attn_k") {
        mult *= 1.45;
        min_level = 1;
    } else if name.contains("ffn_down") && !name.contains("_exps") {
        mult *= 1.35;
    } else if name.contains("_exps") {
        mult *= 0.85;
    }
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot) = rest.find('.') {
            if let Ok(blk) = rest[..dot].parse::<usize>() {
                // Boundary blocks (first 2 and last 2) carry higher representation sensitivity
                if blk < 2 || (max_block > 4 && blk + 2 >= max_block) {
                    mult *= 1.35;
                    min_level = min_level.max(1);
                }
            }
        }
    }
    (mult, min_level)
}

pub fn allocate_bits_with_strategy(
    scores: &[TensorScore],
    target: f64,
    strategy: AllocationStrategy,
) -> io::Result<Value> {
    if scores.is_empty() || !target.is_finite() || !(BPW[0]..=BPW[4]).contains(&target) {
        return Err(invalid(
            "target must be in 2.5625..=6.5625 BPW with nonempty scores",
        ));
    }
    let total = scores
        .iter()
        .try_fold(0u64, |a, v| a.checked_add(v.elements))
        .ok_or_else(|| invalid("total tensor elements overflow"))?;
    if total == 0 {
        return Err(invalid("empty tensors"));
    }
    let max_block = scores
        .iter()
        .filter_map(|t| {
            t.name
                .strip_prefix("blk.")
                .and_then(|r| r.split('.').next())
                .and_then(|s| s.parse::<usize>().ok())
        })
        .max()
        .map_or(0, |m| m + 1);

    let budget = total as f64 * target;
    let mut levels = vec![0usize; scores.len()];
    let mut current = total as f64 * BPW[0];

    match strategy {
        AllocationStrategy::GreedySensitivity => {
            for _ in 0..(4 * scores.len()) {
                let mut best: Option<(usize, f64)> = None;
                for (i, t) in scores.iter().enumerate() {
                    let level = levels[i];
                    if level + 1 >= BPW.len() {
                        continue;
                    }
                    let incremental = t.elements as f64 * (BPW[level + 1] - BPW[level]);
                    if current + incremental > budget + 0.0001 {
                        continue;
                    }
                    let prev_error = (-BPW[level]).exp();
                    let next_error = (-BPW[level + 1]).exp();
                    let gain =
                        t.sensitivity * (prev_error - next_error) / (BPW[level + 1] - BPW[level]);
                    if best.as_ref().is_none_or(|(_, s)| gain > *s) {
                        best = Some((i, gain));
                    }
                }
                let Some((index, _)) = best else { break };
                let level = levels[index];
                current += scores[index].elements as f64 * (BPW[level + 1] - BPW[level]);
                levels[index] += 1;
            }
        }
        AllocationStrategy::RcoLagrangian => {
            // RCO rate-distortion model: D_i(b) = Omega_i * 2^(-2*b)
            // First solve Lagrangian relaxation via bisection on lambda > 0:
            // argmin_l [ Omega_i * 2^(-2 * BPW[l]) + lambda * BPW[l] ]
            let omegas: Vec<(f64, usize)> = scores
                .iter()
                .map(|t| {
                    let (mult, min_lvl) = tensor_role_multiplier(&t.name, max_block);
                    ((t.sensitivity.max(1e-4)).powf(1.25) * mult, min_lvl)
                })
                .collect();

            let mut lo = 1e-9f64;
            let mut hi = 10.0f64;
            for _ in 0..48 {
                let mid = 0.5 * (lo + hi);
                let mut trial_bits = 0.0f64;
                for (i, t) in scores.iter().enumerate() {
                    let (omega, min_lvl) = omegas[i];
                    let floor = if target >= 3.8 { min_lvl } else { 0 };
                    let mut best_l = floor;
                    let mut best_obj = f64::INFINITY;
                    for (l, &b) in BPW.iter().enumerate().skip(floor) {
                        let obj = omega * (-2.0 * std::f64::consts::LN_2 * b).exp() + mid * b;
                        if obj < best_obj {
                            best_obj = obj;
                            best_l = l;
                        }
                    }
                    trial_bits += t.elements as f64 * BPW[best_l];
                }
                if trial_bits > budget {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            // Assign at conservative `hi` so total bits <= budget
            current = 0.0;
            for (i, t) in scores.iter().enumerate() {
                let (omega, min_lvl) = omegas[i];
                let floor = if target >= 3.8 { min_lvl } else { 0 };
                let mut best_l = floor;
                let mut best_obj = f64::INFINITY;
                for (l, &b) in BPW.iter().enumerate().skip(floor) {
                    let obj = omega * (-2.0 * std::f64::consts::LN_2 * b).exp() + hi * b;
                    if obj < best_obj {
                        best_obj = obj;
                        best_l = l;
                    }
                }
                levels[i] = best_l;
                current += t.elements as f64 * BPW[best_l];
            }
            // If floors pushed current above budget, step down least sensitive first
            while current > budget + 0.0001 {
                let mut worst: Option<(usize, f64)> = None;
                for (i, _t) in scores.iter().enumerate() {
                    let l = levels[i];
                    if l == 0 {
                        continue;
                    }
                    let (omega, _) = omegas[i];
                    let d_cur = omega * (-2.0 * std::f64::consts::LN_2 * BPW[l]).exp();
                    let d_prev = omega * (-2.0 * std::f64::consts::LN_2 * BPW[l - 1]).exp();
                    let penalty_per_bit = (d_prev - d_cur) / (BPW[l] - BPW[l - 1]);
                    if worst.as_ref().is_none_or(|(_, p)| penalty_per_bit < *p) {
                        worst = Some((i, penalty_per_bit));
                    }
                }
                let Some((idx, _)) = worst else { break };
                let l = levels[idx];
                current -= scores[idx].elements as f64 * (BPW[l] - BPW[l - 1]);
                levels[idx] -= 1;
            }
            // Polish remaining slack with marginal RCO distortion reduction
            for _ in 0..(4 * scores.len()) {
                let mut best: Option<(usize, f64)> = None;
                for (i, t) in scores.iter().enumerate() {
                    let l = levels[i];
                    if l + 1 >= BPW.len() {
                        continue;
                    }
                    let inc = t.elements as f64 * (BPW[l + 1] - BPW[l]);
                    if current + inc > budget + 0.0001 {
                        continue;
                    }
                    let (omega, _) = omegas[i];
                    let d_cur = omega * (-2.0 * std::f64::consts::LN_2 * BPW[l]).exp();
                    let d_next = omega * (-2.0 * std::f64::consts::LN_2 * BPW[l + 1]).exp();
                    let gain = (d_cur - d_next) / (BPW[l + 1] - BPW[l]);
                    if best.as_ref().is_none_or(|(_, g)| gain > *g) {
                        best = Some((i, gain));
                    }
                }
                let Some((idx, _)) = best else { break };
                let l = levels[idx];
                current += scores[idx].elements as f64 * (BPW[l + 1] - BPW[l]);
                levels[idx] += 1;
            }
        }
        AllocationStrategy::LearnedRoleAware => {
            // ShapeLearn / AutoScheme-inspired role & position-aware coordinate optimization
            let weights: Vec<f64> = scores
                .iter()
                .map(|t| {
                    let (mult, _) = tensor_role_multiplier(&t.name, max_block);
                    (0.35 + 0.65 * t.sensitivity) * mult
                })
                .collect();
            for _ in 0..(4 * scores.len()) {
                let mut best: Option<(usize, f64)> = None;
                for (i, t) in scores.iter().enumerate() {
                    let l = levels[i];
                    if l + 1 >= BPW.len() {
                        continue;
                    }
                    let inc = t.elements as f64 * (BPW[l + 1] - BPW[l]);
                    if current + inc > budget + 0.0001 {
                        continue;
                    }
                    let prev_err = (-1.6 * BPW[l]).exp();
                    let next_err = (-1.6 * BPW[l + 1]).exp();
                    let gain = weights[i] * (prev_err - next_err) / (BPW[l + 1] - BPW[l]);
                    if best.as_ref().is_none_or(|(_, g)| gain > *g) {
                        best = Some((i, gain));
                    }
                }
                let Some((idx, _)) = best else { break };
                let l = levels[idx];
                current += scores[idx].elements as f64 * (BPW[l + 1] - BPW[l]);
                levels[idx] += 1;
            }
        }
    }

    let mut distortion_proxy = 0.0f64;
    for (i, t) in scores.iter().enumerate() {
        let (mult, _) = tensor_role_multiplier(&t.name, max_block);
        let w = t.elements as f64 / total as f64;
        distortion_proxy +=
            w * t.sensitivity * mult * (-2.0 * std::f64::consts::LN_2 * BPW[levels[i]]).exp();
    }

    let assignments = scores
        .iter()
        .enumerate()
        .map(|(i, t)| {
            json!({
                "name": t.name,
                "elements": t.elements,
                "sensitivity": t.sensitivity,
                "type": TYPES[levels[i]],
                "nominal_bpw": BPW[levels[i]]
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "schema": "ojon.tensor_allocation.v1",
        "strategy": strategy.name(),
        "target_measured_tensor_bpw": target,
        "estimated_measured_tensor_bpw": current / total as f64,
        "rate_distortion_proxy": distortion_proxy,
        "scored_tensor_count": scores.len(),
        "scored_weight_elements": total,
        "assignments": assignments,
        "warning": "Size estimate covers only listed tensors; actual GGUF size, group overhead, fallback types and runtime must be measured."
    }))
}

/// Greedy measured-importance allocation under a weighted bits-per-weight
/// budget for the selected tensors; NOT total model BPW or predicted output size.
pub fn allocate_bits(scores: &[TensorScore], target: f64) -> io::Result<Value> {
    allocate_bits_with_strategy(scores, target, AllocationStrategy::GreedySensitivity)
}

/// Compare greedy, RCO-style Lagrangian, and learned role-aware allocation strategies
/// under the same target bit budget.
pub fn compare_allocation_strategies(scores: &[TensorScore], target: f64) -> io::Result<Value> {
    let greedy =
        allocate_bits_with_strategy(scores, target, AllocationStrategy::GreedySensitivity)?;
    let rco = allocate_bits_with_strategy(scores, target, AllocationStrategy::RcoLagrangian)?;
    let learned =
        allocate_bits_with_strategy(scores, target, AllocationStrategy::LearnedRoleAware)?;
    let strategies = vec![greedy, rco, learned];
    let best_strategy = strategies
        .iter()
        .min_by(|a, b| {
            let da = a["rate_distortion_proxy"].as_f64().unwrap_or(f64::INFINITY);
            let db = b["rate_distortion_proxy"].as_f64().unwrap_or(f64::INFINITY);
            da.total_cmp(&db)
        })
        .and_then(|v| v["strategy"].as_str())
        .unwrap_or("rco_lagrangian")
        .to_string();
    Ok(json!({
        "schema": "ojon.strategy_comparison.v1",
        "target_bpw": target,
        "lowest_distortion_strategy": best_strategy,
        "strategies": strategies,
    }))
}

/// Generate a protective `--tensor-type-file` tail that preserves sensitive
/// embeddings, output head, MoE routers, and shared experts when present.
pub fn sensitive_spine_pinning_recipe(tensors: &[crate::gguf::GgufTensor]) -> String {
    let mut extra = crate::llama::moe_router_pinning_recipe(tensors).unwrap_or_default();
    for t in tensors {
        if (t.name == "token_embd.weight"
            || t.name == "output.weight"
            || t.name.contains("_shexp.weight"))
            && t.shape.len() >= 2
            && t.shape[0] % 256 == 0
        {
            let rule = format!("{}=q6_k\n", regex_literal(&t.name));
            if !extra.contains(&rule) {
                extra.push_str(&rule);
            }
        }
    }
    extra
}

pub fn prepare_adaptive(
    source: &Path,
    scores: &Path,
    output_dir: &Path,
    suite: &Path,
    quantizer: &Path,
    eval_tool: &Path,
    imatrix: Option<&Path>,
    targets: &[f64],
    threads: usize,
    gpu_layers: i32,
    device: Option<&str>,
    tokens: u32,
    max_absolute_loss_pct: f64,
) -> io::Result<AdaptivePlan> {
    let source = source.canonicalize()?;
    let scores_file = scores.canonicalize()?;
    let parsed = parse_tensor_sensitivity(&source, &scores_file)?;
    if !output_dir.is_dir() {
        return Err(invalid("adaptive output directory must already exist"));
    }
    let output_dir = output_dir.canonicalize()?;
    if output_dir == source.parent().unwrap_or(Path::new("")) {
        return Err(invalid(
            "adaptive outputs require a separate directory from source",
        ));
    }
    if targets.is_empty() || targets.len() > 8 {
        return Err(invalid("require 1..=8 budget targets"));
    }
    let mut bits = Vec::new();
    for &b in targets {
        allocate_bits(&parsed, b)?;
        if bits.iter().any(|n: &f64| (*n - b).abs() < 0.00001) {
            return Err(invalid("duplicate adaptive bitrate target"));
        }
        bits.push(b);
    }
    bits.sort_by(|a, b| a.total_cmp(b));
    if threads == 0 {
        return Err(invalid("threads must be >0"));
    }
    if !max_absolute_loss_pct.is_finite() || !(0.0..=10.0).contains(&max_absolute_loss_pct) {
        return Err(invalid(
            "max absolute accuracy loss must be 0..=10 percentage points",
        ));
    }
    let imatrix = imatrix.map(Path::canonicalize).transpose()?;
    if let Some(i) = &imatrix {
        if !i.is_file() || i.metadata()?.len() == 0 {
            return Err(invalid("imatrix missing/empty"));
        }
    }
    // Validates the same task suite used later for every bitrate.
    let suite = suite.canonicalize()?;
    crate::task_eval::parse_eval_suite(&suite)?;
    if quantizer.as_os_str().is_empty() || eval_tool.as_os_str().is_empty() {
        return Err(invalid("quantizer and llama-cli must be provided"));
    }
    if !(1..=1024).contains(&tokens) || !(-1..=512).contains(&gpu_layers) {
        return Err(invalid("evaluation token or layer count out of range"));
    }
    if let Some(d) = device {
        if d.is_empty()
            || d.len() > 80
            || !d
                .bytes()
                .all(|v| v.is_ascii_alphanumeric() || v == b'_' || v == b'-' || v == b'/')
        {
            return Err(invalid("invalid device"));
        }
    }
    Ok(AdaptivePlan {
        reference: source.clone(),
        source,
        scores_file,
        scores: parsed,
        output_dir,
        suite,
        quantizer: quantizer.into(),
        eval_tool: eval_tool.into(),
        server_tool: None,
        imatrix,
        targets: bits,
        threads,
        gpu_layers,
        device: device.map(str::to_owned),
        tokens,
        max_absolute_loss_pct,
    })
}

/// Guided searches use the existing quantized Q4 benchmark reference.
pub fn prepare_adaptive_against_reference(
    source: &Path,
    reference: &Path,
    scores: &Path,
    output_dir: &Path,
    suite: &Path,
    quantizer: &Path,
    eval_tool: &Path,
    imatrix: Option<&Path>,
    targets: &[f64],
    threads: usize,
    gpu_layers: i32,
    device: Option<&str>,
    tokens: u32,
    max_absolute_loss_pct: f64,
) -> io::Result<AdaptivePlan> {
    let mut plan = prepare_adaptive(
        source,
        scores,
        output_dir,
        suite,
        quantizer,
        eval_tool,
        imatrix,
        targets,
        threads,
        gpu_layers,
        device,
        tokens,
        max_absolute_loss_pct,
    )?;
    let (_, matched, _) = crate::quant_parity::inspect_compatible_models(source, reference)?;
    plan.reference = matched;
    Ok(plan)
}
impl AdaptivePlan {
    pub fn as_json(&self) -> Value {
        json!({
            "schema":"ojon.adaptive_plan.v1","source":self.source.to_string_lossy(),
            "score_source":self.scores_file.to_string_lossy(),
            "sensitivity_verified_by_ojon":false,
            "output_dir":self.output_dir.to_string_lossy(),
            "suite":self.suite.to_string_lossy(),
            "evaluation_server":self.server_tool.as_ref().map(|p|p.to_string_lossy().to_string()),
            "quality_reference":self.reference.to_string_lossy(),
            "targets":self.targets.iter().map(|&t|allocate_bits(&self.scores,t).unwrap_or(Value::Null)).collect::<Vec<_>>(),
            "default_format":"GGUF with upstream Q2_K through Q6_K tensor overrides",
            "quality_tolerance_absolute_percentage_points":self.max_absolute_loss_pct,
            "runtime_memory_fit_verified":false,
            "status":"PLAN_ONLY"
        })
    }
}

fn regex_literal(name: &str) -> String {
    let mut pattern = String::from("^");
    for ch in name.chars() {
        if matches!(
            ch,
            '.' | '+' | '*' | '?' | '(' | ')' | '[' | ']' | '{' | '}' | '|' | '^' | '$' | '\\'
        ) {
            pattern.push('\\');
        }
        pattern.push(ch);
    }
    pattern.push('$');
    pattern
}
fn write_recipe(path: &Path, allocation: &Value, extra: &str) -> io::Result<()> {
    let entries = allocation["assignments"]
        .as_array()
        .ok_or_else(|| invalid("missing assignments"))?;
    let mut text = String::new();
    for row in entries {
        let name = row["name"]
            .as_str()
            .ok_or_else(|| invalid("missing tensor"))?;
        let q = row["type"]
            .as_str()
            .ok_or_else(|| invalid("missing tensor precision"))?;
        text.push_str(&regex_literal(name));
        text.push('=');
        text.push_str(q);
        text.push('\n');
    }
    // MoE router preservation: pin routing tensors to Q6_K so a guided search
    // can never make expert selection a regression source. Empty for dense.
    text.push_str(extra);
    let mut file = OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(text.as_bytes())?;
    file.sync_all()
}

/// Run real quantizations in ascending budget order, run same held-out suite
/// against the original and every candidate, keep the smallest measured pass.
pub fn run_adaptive(plan: &AdaptivePlan, mut log: impl FnMut(&str)) -> io::Result<Value> {
    let started = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| invalid("system time invalid"))?
        .as_nanos();
    let session = format!("ojon-adaptive-{}-{started}", std::process::id());
    let suite_id = crate::task_eval::parse_eval_suite(&plan.suite)?.id;
    let source_tensors = crate::gguf::inspect_gguf_tensors(&plan.source)?;
    let router_recipe = sensitive_spine_pinning_recipe(&source_tensors);
    let mut rows = Vec::new();
    let mut best: Option<(String, u64)> = None;
    let mut baseline_tasks: Option<Value> = None;
    for (idx, target) in plan.targets.iter().enumerate() {
        let label = format!("adaptive_{idx}");
        let prefix = format!("{session}-{:03}", (target * 100.0).round() as u32);
        let output = plan.output_dir.join(format!("{prefix}.gguf"));
        let recipe = plan.output_dir.join(format!("{prefix}.tensor-types.txt"));
        let evidence = plan.output_dir.join(format!("{prefix}.quality.json"));
        let strategy_cmp = compare_allocation_strategies(&plan.scores, *target)?;
        let best_strat_name = strategy_cmp["lowest_distortion_strategy"]
            .as_str()
            .unwrap_or("rco_lagrangian");
        let chosen_strategy =
            AllocationStrategy::parse(best_strat_name).unwrap_or(AllocationStrategy::RcoLagrangian);
        let alloc = allocate_bits_with_strategy(&plan.scores, *target, chosen_strategy)?;
        // Resume: reuse an existing verified candidate instead of re-quantizing.
        let reusable = match crate::quant_parity::verify_reusable_candidate(&plan.source, &output) {
            Ok(reusable) => reusable,
            Err(e) => {
                rows.push(
                    json!({"target":target,"status":"resume_verification_failed",
                    "reason":e.to_string(),"output":output.to_string_lossy()}),
                );
                continue;
            }
        };
        let size = if let Some(bytes) = reusable {
            log(&format!(
                "[ojon] resume: reusing existing candidate {}",
                output.display()
            ));
            bytes
        } else {
            write_recipe(&recipe, &alloc, &router_recipe)?;
            log(&format!(
                "[ojon] quantizing tensor budget {:.3} BPW using strategy {}",
                target,
                chosen_strategy.name()
            ));
            let converted = (|| {
                let mut q = prepare_gguf(
                    &plan.source,
                    &output,
                    &plan.quantizer,
                    Precision::Q4KM,
                    plan.threads,
                )?
                .with_tensor_type_file(&recipe)?;
                if let Some(i) = &plan.imatrix {
                    q = q.with_imatrix(i)?;
                }
                quantize_gguf_with_log(&q, |line| log(line))
            })();
            match converted {
                Err(e) => {
                    rows.push(
                        json!({"target":target,"status":"conversion_failed","reason":e.to_string(),
                        "recipe":recipe.to_string_lossy()}),
                    );
                    continue;
                }
                Ok(done) => done.output_info.file_bytes,
            }
        };
        let topology_json = crate::gguf::inspect_model_topology(&output)
            .map(|t| t.as_json())
            .unwrap_or(Value::Null);
        let prior = match crate::quant_parity::verify_prior_evidence(
            &evidence,
            &plan.reference,
            &suite_id,
            &label,
            &output,
        ) {
            Ok(prior) => prior,
            Err(e) => {
                rows.push(
                    json!({"target":target,"status":"existing_evidence_not_reusable",
                    "reason":e.to_string(),"bytes":size,"output":output.to_string_lossy()}),
                );
                continue;
            }
        };
        let prior_used = prior.is_some();
        let evaluated = if let Some(report) = prior {
            log("[ojon] resume: reusing verified held-out task evidence (baseline unchanged)");
            Ok(report)
        } else {
            (|| {
                let candidate = vec![(label.clone(), output.clone())];
                let test = prepare_eval_with_server(EvalRequest {
                    suite: &plan.suite,
                    baseline: &plan.reference,
                    candidates: &candidate,
                    tool: &plan.eval_tool,
                    server_tool: plan.server_tool.as_deref(),
                    gpu_layers: plan.gpu_layers,
                    device: plan.device.as_deref(),
                    tokens: plan.tokens,
                    report: &evidence,
                })?;
                run_eval_reusing_baseline(&test, baseline_tasks.as_ref(), |line| log(line))
            })()
        };
        match evaluated {
            Err(e) => rows.push(json!({"target":target,"status":"quality_evaluation_failed",
                "reason":e.to_string(),"bytes":size,
                "output":output.to_string_lossy()})),
            Ok(report) => {
                if baseline_tasks.is_none() {
                    baseline_tasks = Some(report.clone());
                }
                let base = &report["baseline"]["metrics"];
                let cand = &report["candidates"][0]["metrics"];
                let mut comparisons = Vec::new();
                let mut passes = true;
                for metric in [
                    "coding_accuracy",
                    "reasoning_accuracy",
                    "tool_call_accuracy",
                ] {
                    let b = base[metric].as_f64().unwrap_or(0.0);
                    let c = cand[metric].as_f64().unwrap_or(0.0);
                    // A suite the baseline cannot solve is not evidence of preserved ability.
                    let pass = b > 0.0 && c + 1e-10 >= b - plan.max_absolute_loss_pct;
                    if !pass {
                        passes = false;
                    }
                    comparisons.push(json!({"metric":metric,"baseline":b,"candidate":c,
                        "allowed_drop_points":plan.max_absolute_loss_pct,"pass":pass}));
                }
                if passes && best.as_ref().is_none_or(|(_, bytes)| size < *bytes) {
                    best = Some((output.to_string_lossy().to_string(), size));
                }
                rows.push(json!({"target":target,"status":"measured_task_suite",
                    "strategy":chosen_strategy.name(),
                    "strategy_comparison":strategy_cmp,
                    "output":output.to_string_lossy(),"bytes":size,
                    "effective_bpw":topology_json["effective_bpw"],
                    "artifact_topology":topology_json,
                    "task_quality_pass":passes,
                    "resumed_quantization":reusable.is_some(),
                    "resumed_task_evidence":prior_used,
                    "moe_router_pinned":!router_recipe.is_empty(),
                    "comparisons":comparisons,"quality_receipt":evidence.to_string_lossy(),
                    "tensor_type_file":recipe.to_string_lossy()}));
            }
        }
    }
    let report = json!({"schema":"ojon.adaptive_search_result.v1",
        "source":plan.source.to_string_lossy(),"quality_reference":plan.reference.to_string_lossy(),"suite":plan.suite.to_string_lossy(),
        "candidates":rows,"provisional_smallest_task_passing_output":best.as_ref().map(|p|p.0.clone()),
        "provisional_smallest_bytes":best.map(|p|p.1),
        "hardware_residency_verified":false,"near_lossless_generalization_verified":false,
        "warning":"Winner passes only requested held-out deterministic tasks; verify target GPU RAM/VRAM and larger real coding tasks before deployment."});
    let receipt = plan.output_dir.join(format!("{session}.json"));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&receipt)?;
    file.write_all(&serde_json::to_vec_pretty(&report).map_err(io::Error::other)?)?;
    file.sync_all()?;
    Ok(report)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn sample() -> Vec<TensorScore> {
        vec![
            TensorScore {
                name: "blk.0.attn_q.weight".into(),
                elements: 65536,
                sensitivity: 0.95,
            },
            TensorScore {
                name: "blk.0.ffn_up.weight".into(),
                elements: 65536,
                sensitivity: 0.1,
            },
            TensorScore {
                name: "blk.1.attn_q.weight".into(),
                elements: 65536,
                sensitivity: 0.6,
            },
        ]
    }
    #[test]
    fn budget_and_priority() {
        let scores = sample();
        let low = allocate_bits(&scores, 2.5625).unwrap();
        assert_eq!(low["assignments"][0]["type"], "q2_k");
        let med = allocate_bits(&scores, 3.4375).unwrap();
        let bpw = med["estimated_measured_tensor_bpw"].as_f64().unwrap();
        assert!(bpw <= 3.4375 + 1e-8 && bpw > 2.5625);
        assert!(allocate_bits(&scores, 1.0).is_err());
        assert!(allocate_bits(&scores, f64::NAN).is_err());
    }
    #[test]
    fn rco_and_learned_strategies_respect_budget_and_compare() {
        let scores = sample();
        let cmp = compare_allocation_strategies(&scores, 4.0).unwrap();
        assert_eq!(cmp["schema"], "ojon.strategy_comparison.v1");
        let strats = cmp["strategies"].as_array().unwrap();
        assert_eq!(strats.len(), 3);
        for s in strats {
            let bpw = s["estimated_measured_tensor_bpw"].as_f64().unwrap();
            assert!((2.5625..=4.0 + 1e-6).contains(&bpw));
        }
    }
    #[test]
    fn escape_tensor_regex() {
        assert_eq!(
            regex_literal("blk.0.attn_q.weight"),
            r"^blk\.0\.attn_q\.weight$"
        );
    }
}
