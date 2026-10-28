//! Dependency-free first-stage ARA-like low-rank adapter optimizer.
//!
//! Takes previously captured linear-module input/output pairs. It does NOT
//! perform live forward passes, autodiff through a transformer, L-BFGS or TPE.
//! Its measured loss is a LOCAL MODULE PROXY, not refusal/coding/vision quality.
//! Exports the ojon.abliteration.v1 low-rank edit recipe for the Rust editor.

use serde_json::{json, Value};
use std::fs::{self, OpenOptions};
use std::io::{self, ErrorKind, Write};
use std::path::{Path, PathBuf};

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, s.into())
}
fn number(v: &Value, name: &str) -> io::Result<f64> {
    let n = v
        .get(name)
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid(format!("missing numeric {name}")))?;
    if !n.is_finite() {
        return Err(invalid(format!("{name} must be finite")));
    }
    Ok(n)
}
fn vector(v: &Value, name: &str) -> io::Result<Vec<f64>> {
    let a = v
        .as_array()
        .ok_or_else(|| invalid(format!("{name} must be an array")))?;
    if a.is_empty() || a.len() > 8192 {
        return Err(invalid(format!("invalid {name} length")));
    }
    a.iter()
        .map(|x| {
            let n = x
                .as_f64()
                .ok_or_else(|| invalid(format!("nonnumeric {name}")))?;
            if !n.is_finite() {
                return Err(invalid(format!("nonfinite {name}")));
            }
            Ok(n)
        })
        .collect()
}
#[derive(Clone)]
struct Sample {
    input: Vec<f64>,
    output: Vec<f64>,
}
fn samples(v: &Value, name: &str) -> io::Result<Vec<Sample>> {
    let a = v
        .as_array()
        .ok_or_else(|| invalid(format!("{name} must be an array")))?;
    if a.is_empty() || a.len() > 128 {
        return Err(invalid(format!("{name} sample count must be 1..128")));
    }
    a.iter()
        .map(|v| {
            Ok(Sample {
                input: vector(
                    v.get("input")
                        .ok_or_else(|| invalid("sample input missing"))?,
                    "input",
                )?,
                output: vector(
                    v.get("output")
                        .ok_or_else(|| invalid("sample output missing"))?,
                    "output",
                )?,
            })
        })
        .collect()
}
struct Config {
    tensor: String,
    rank: usize,
    steps: usize,
    rate: f64,
    good: Vec<Sample>,
    bad: Vec<Sample>,
    ins: usize,
    outs: usize,
    preserve: f64,
    steer: f64,
    overcorrect: f64,
    row_norms: bool,
}
fn config(v: &Value) -> io::Result<Config> {
    if v.get("schema").and_then(Value::as_str) != Some("ojon.ara_fit.v1") {
        return Err(invalid("expected ojon.ara_fit.v1"));
    }
    let tensor = v
        .get("tensor")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty() && s.len() <= 512)
        .ok_or_else(|| invalid("invalid target weight tensor name"))?
        .to_owned();
    let rank = v
        .get("rank")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("missing rank"))? as usize;
    let steps = v
        .get("steps")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid("missing steps"))? as usize;
    let rate = number(v, "learning_rate")?;
    if rank == 0 || rank > 32 || steps == 0 || steps > 500 || rate <= 0.0 || rate > 0.25 {
        return Err(invalid(
            "rank 1..32, steps 1..500, learning_rate >0 .. 0.25",
        ));
    }
    let preserve = number(v, "preserve_good_weight")?;
    let steer = number(v, "steer_bad_weight")?;
    let overcorrect = number(v, "overcorrect_weight")?;
    if preserve < 0.0 || steer <= 0.0 || overcorrect < 0.0 || overcorrect > 1.0 {
        return Err(invalid("invalid ARA loss weight"));
    }
    let row_norms = v
        .get("preserve_row_norms")
        .and_then(Value::as_bool)
        .ok_or_else(|| invalid("preserve_row_norms must be boolean"))?;
    let good = samples(
        v.get("good")
            .ok_or_else(|| invalid("good samples missing"))?,
        "good",
    )?;
    let bad = samples(
        v.get("bad").ok_or_else(|| invalid("bad samples missing"))?,
        "bad",
    )?;
    let ins = good[0].input.len();
    let outs = good[0].output.len();
    if (ins + outs)
        .checked_mul(rank)
        .ok_or_else(|| invalid("size overflow"))?
        > 1_000_000
    {
        return Err(invalid("adapter too large"));
    }
    if good
        .iter()
        .chain(&bad)
        .any(|s| s.input.len() != ins || s.output.len() != outs)
    {
        return Err(invalid(
            "inconsistent paired module input/output dimensions",
        ));
    }
    Ok(Config {
        tensor,
        rank,
        steps,
        rate,
        good,
        bad,
        ins,
        outs,
        preserve,
        steer,
        overcorrect,
        row_norms,
    })
}

struct Grad {
    a: Vec<f64>,
    b: Vec<f64>,
    loss: f64,
}
fn gradient(cfg: &Config, a: &[f64], b: &[f64]) -> io::Result<Grad> {
    let (mut ga, mut gb) = (vec![0.0; a.len()], vec![0.0; b.len()]);
    let mut loss = 0.0f64;
    let mut forward = |sample: &Sample, good: bool| {
        let mut z = vec![0.0; cfg.rank];
        for r in 0..cfg.rank {
            for i in 0..cfg.ins {
                z[r] += a[r * cfg.ins + i] * sample.input[i];
            }
        }
        let mut delta = vec![0.0; cfg.outs];
        for o in 0..cfg.outs {
            for r in 0..cfg.rank {
                delta[o] += b[o * cfg.rank + r] * z[r];
            }
        }
        let mut ddelta = vec![0.0; cfg.outs];
        if good {
            let coeff = cfg.preserve / (cfg.good.len() * cfg.outs) as f64;
            for o in 0..cfg.outs {
                loss += coeff * delta[o] * delta[o];
                ddelta[o] = 2.0 * coeff * delta[o];
            }
        } else {
            // Heretic-inspired nearest-neighbor pull towards original good
            // outputs, optional push away from the original bad output.
            let mut nearest = &cfg.good[0].output;
            let mut min_dist = f64::INFINITY;
            for target in &cfg.good {
                let dist = target
                    .output
                    .iter()
                    .enumerate()
                    .map(|(i, y)| (sample.output[i] + delta[i] - y).powi(2))
                    .sum::<f64>();
                if dist < min_dist {
                    min_dist = dist;
                    nearest = &target.output;
                }
            }
            let dist = (min_dist + 1e-12).sqrt();
            let baseline_dist = (delta.iter().map(|v| v * v).sum::<f64>() + 1e-12).sqrt();
            let coeff = cfg.steer / (cfg.bad.len() as f64);
            loss += coeff * (dist - cfg.overcorrect * baseline_dist);
            for o in 0..cfg.outs {
                ddelta[o] = coeff
                    * ((sample.output[o] + delta[o] - nearest[o]) / dist
                        - cfg.overcorrect * delta[o] / baseline_dist);
            }
        }
        for o in 0..cfg.outs {
            for r in 0..cfg.rank {
                gb[o * cfg.rank + r] += ddelta[o] * z[r];
            }
        }
        for r in 0..cfg.rank {
            let mut dz = 0.0;
            for o in 0..cfg.outs {
                dz += ddelta[o] * b[o * cfg.rank + r];
            }
            for i in 0..cfg.ins {
                ga[r * cfg.ins + i] += dz * sample.input[i];
            }
        }
    };
    for item in &cfg.good {
        forward(item, true);
    }
    for item in &cfg.bad {
        forward(item, false);
    }
    // Small coefficient regularization guards unbounded overcorrection.
    let wd = 1e-6;
    for (g, &v) in ga.iter_mut().zip(a) {
        loss += wd * v * v;
        *g += 2.0 * wd * v;
    }
    for (g, &v) in gb.iter_mut().zip(b) {
        loss += wd * v * v;
        *g += 2.0 * wd * v;
    }
    if !loss.is_finite() || ga.iter().chain(&gb).any(|x| !x.is_finite()) {
        return Err(invalid("nonfinite ARA objective/gradient"));
    }
    Ok(Grad { a: ga, b: gb, loss })
}

fn adam(
    weights: &mut [f64],
    gradient: &[f64],
    m: &mut [f64],
    v: &mut [f64],
    time: usize,
    rate: f64,
) -> io::Result<()> {
    let t = time as f64;
    let k1 = 1.0 - 0.9f64.powf(t);
    let k2 = 1.0 - 0.999f64.powf(t);
    for i in 0..weights.len() {
        // Clip individual gradients for stability of this dependency-free
        // prototype; Heretic's original optimizer uses L-BFGS instead.
        let g = gradient[i].clamp(-10.0, 10.0);
        m[i] = 0.9 * m[i] + 0.1 * g;
        v[i] = 0.999 * v[i] + 0.001 * g * g;
        weights[i] -= rate * (m[i] / k1) / ((v[i] / k2).sqrt() + 1e-8);
        if !weights[i].is_finite() {
            return Err(invalid("ARA optimizer diverged"));
        }
    }
    Ok(())
}

#[derive(Debug)]
pub struct AraFitResult {
    pub tensor: String,
    pub rank: usize,
    pub steps: usize,
    pub initial_loss: f64,
    pub final_loss: f64,
    pub recipe: Value,
}
impl AraFitResult {
    pub fn as_json(&self) -> Value {
        json!({"schema":"ojon.ara_fit_result.v1","tensor":self.tensor,"rank":self.rank,
            "steps":self.steps,"initial_module_loss":self.initial_loss,
            "final_module_loss":self.final_loss,
            "optimizer":"RUST_ADAM_FIRST_STAGE_NOT_LBFGS",
            "method":"ARA_INSPIRED_LOW_RANK",
            "refusal_rate":"NOT_EVALUATED",
            "vision_quality":"NOT_EVALUATED",
            "recipe":self.recipe
        })
    }
}
pub fn fit_ara(v: &Value) -> io::Result<AraFitResult> {
    let cfg = config(v)?;
    let mut a = vec![0.0; cfg.rank * cfg.ins];
    let mut b = vec![0.0; cfg.outs * cfg.rank];
    // Deterministic nonzero A and zero B initializes a no-op low-rank
    // update but permits learning B at the very first optimization step.
    for (i, entry) in a.iter_mut().enumerate() {
        *entry = 0.02 * ((i + 1) as f64 * 0.6180339887).sin() / (cfg.ins as f64).sqrt();
    }
    let initial = gradient(&cfg, &a, &b)?.loss;
    let mut ma = vec![0.0; a.len()];
    let mut va = vec![0.0; a.len()];
    let mut mb = vec![0.0; b.len()];
    let mut vb = vec![0.0; b.len()];
    for step in 1..=cfg.steps {
        let g = gradient(&cfg, &a, &b)?;
        adam(&mut a, &g.a, &mut ma, &mut va, step, cfg.rate)?;
        adam(&mut b, &g.b, &mut mb, &mut vb, step, cfg.rate)?;
    }
    let final_loss = gradient(&cfg, &a, &b)?.loss;
    let array_a: Vec<Vec<f64>> = a.chunks_exact(cfg.ins).map(|r| r.to_vec()).collect();
    let array_b: Vec<Vec<f64>> = b.chunks_exact(cfg.rank).map(|r| r.to_vec()).collect();
    let recipe = json!({
        "schema":"ojon.abliteration.v1",
        "preserve_row_norms":cfg.row_norms,
        "edits":[{
            "tensor":cfg.tensor,
            "strength":1.0,
            "lora":{"a":array_a,"b":array_b}
        }]
    });
    Ok(AraFitResult {
        tensor: cfg.tensor,
        rank: cfg.rank,
        steps: cfg.steps,
        initial_loss: initial,
        final_loss,
        recipe,
    })
}
pub fn fit_ara_file(path: &Path) -> io::Result<AraFitResult> {
    let size = fs::metadata(path)?.len();
    if size == 0 || size > 64 * 1024 * 1024 {
        return Err(invalid("ARA calibration JSON exceeds 64 MiB"));
    }
    let value: Value = serde_json::from_slice(&fs::read(path)?)
        .map_err(|e| invalid(format!("invalid ARA calibration JSON: {e}")))?;
    fit_ara(&value)
}
pub fn save_ara_recipe(result: &AraFitResult, output: &Path) -> io::Result<PathBuf> {
    if output.extension().and_then(|s| s.to_str()) != Some("json") {
        return Err(invalid("output must be .json"));
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    let dest = parent.join(
        output
            .file_name()
            .ok_or_else(|| invalid("missing output name"))?,
    );
    let mut f = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&dest)?;
    let bytes = serde_json::to_vec_pretty(&result.recipe)
        .map_err(|e| invalid(format!("recipe serialization: {e}")))?;
    if let Err(e) = f.write_all(&bytes).and_then(|_| f.sync_all()) {
        drop(f);
        let _ = fs::remove_file(&dest);
        return Err(e);
    }
    Ok(dest)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fitter_produces_low_rank_edit_with_finite_loss() {
        let raw = json!({
          "schema":"ojon.ara_fit.v1",
          "tensor":"model.layers.0.self_attn.o_proj.weight",
          "rank":1,"steps":80,"learning_rate":0.03,
          "preserve_good_weight":0.6,"steer_bad_weight":1.0,
          "overcorrect_weight":0.0,"preserve_row_norms":false,
          "good":[{"input":[1.0,0.0],"output":[2.0,0.0]}],
          "bad":[{"input":[0.0,1.0],"output":[0.0,2.0]}]
        });
        let fit = fit_ara(&raw).unwrap();
        assert!(fit.final_loss.is_finite());
        assert!(
            fit.final_loss < fit.initial_loss,
            "optimizer should reduce synthetic proxy objective"
        );
        assert_eq!(
            fit.recipe["edits"][0]["lora"]["a"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            fit.recipe["edits"][0]["lora"]["b"]
                .as_array()
                .unwrap()
                .len(),
            2
        );
    }
    #[test]
    fn refuses_missing_samples_or_wrong_dimensions() {
        let bad = json!({"schema":"ojon.ara_fit.v1","tensor":"w","rank":1,
          "steps":10,"learning_rate":0.01,"preserve_good_weight":1.0,
          "steer_bad_weight":1.0,"overcorrect_weight":0.0,
          "preserve_row_norms":false,
          "good":[{"input":[1,2],"output":[1,2]}],
          "bad":[{"input":[1],"output":[1,2]}]});
        assert!(fit_ara(&bad).is_err());
    }
}
