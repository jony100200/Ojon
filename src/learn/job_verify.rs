//! Ojon Learn V1 verification: base vs candidate on the SAME frozen held-out suite.

use super::job_preflight::CheckStatus;
use crate::eval::task_eval::{prepare_eval, run_eval, EvalRequest};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};

/// A candidate may not lose more than this many points on any metric.
pub const MAX_REGRESSION_POINTS: f64 = 1.0;

pub type Scores = BTreeMap<String, f64>;

/// Scores base and candidate with identical settings. Implementations must not
/// fabricate numbers; return Err when evaluation cannot run.
pub trait Evaluator {
    fn id(&self) -> &str;
    fn is_mock(&self) -> bool;
    fn evaluate(
        &self,
        suite: &Path,
        base: &Path,
        candidate: &Path,
        report: &Path,
    ) -> io::Result<(Scores, Scores)>;
}

/// Real evaluator: llama.cpp-driven `task_eval` over single-file GGUF models.
pub struct TaskEvalEvaluator {
    pub tool: PathBuf,
    pub server_tool: Option<PathBuf>,
    pub gpu_layers: i32,
    pub device: Option<String>,
    pub tokens: u32,
}

fn scores_of(metrics: &Value) -> Scores {
    metrics
        .as_object()
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| Some((k.clone(), v.as_f64()?)))
                .collect()
        })
        .unwrap_or_default()
}

impl Evaluator for TaskEvalEvaluator {
    fn id(&self) -> &str {
        "task_eval"
    }
    fn is_mock(&self) -> bool {
        false
    }
    fn evaluate(
        &self,
        suite: &Path,
        base: &Path,
        candidate: &Path,
        report: &Path,
    ) -> io::Result<(Scores, Scores)> {
        let candidates = [("candidate".to_string(), candidate.to_path_buf())];
        let plan = prepare_eval(EvalRequest {
            suite,
            baseline: base,
            candidates: &candidates,
            tool: &self.tool,
            server_tool: self.server_tool.as_deref(),
            gpu_layers: self.gpu_layers,
            device: self.device.as_deref(),
            tokens: self.tokens,
            report,
        })?;
        let out = run_eval(&plan, |_| {})?;
        Ok((
            scores_of(&out["baseline"]["metrics"]),
            scores_of(&out["candidates"][0]["metrics"]),
        ))
    }
}

/// What to compare. `candidate_model` is `None` until the adapter has been
/// merged and converted into an evaluable model.
#[derive(Clone, Debug)]
pub struct VerifyInputs {
    pub base_model: PathBuf,
    pub candidate_model: Option<PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MetricResult {
    pub name: String,
    pub base: Option<f64>,
    pub candidate: Option<f64>,
    pub delta: Option<f64>,
    pub status: CheckStatus,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyReport {
    pub schema: String,
    pub status: CheckStatus,
    pub reason: String,
    pub suite_sha256: String,
    pub evaluator: String,
    pub evaluator_is_mock: bool,
    pub trainer_is_mock: bool,
    pub real_gpu_training: String,
    pub metrics: Vec<MetricResult>,
    pub evidence_paths: Vec<String>,
}

pub fn compare(base: &Scores, cand: &Scores) -> (Vec<MetricResult>, CheckStatus, String) {
    let mut names: Vec<&String> = base.keys().chain(cand.keys()).collect();
    names.sort();
    names.dedup();
    let mut metrics = Vec::new();
    let (mut regressed, mut improved, mut missing) = (Vec::new(), false, false);
    for n in names {
        let (b, c) = (base.get(n).copied(), cand.get(n).copied());
        let (delta, status) = match (b, c) {
            (Some(b), Some(c)) => {
                let d = c - b;
                improved |= d > 0.0;
                if d < -MAX_REGRESSION_POINTS {
                    regressed.push(n.clone());
                    (Some(d), CheckStatus::Fail)
                } else {
                    (Some(d), CheckStatus::Pass)
                }
            }
            _ => {
                missing = true;
                (None, CheckStatus::NotRun)
            }
        };
        metrics.push(MetricResult {
            name: n.clone(),
            base: b,
            candidate: c,
            delta,
            status,
        });
    }
    let (status, why) = if metrics.is_empty() || missing {
        (
            CheckStatus::NotRun,
            "metrics missing for base or candidate".to_string(),
        )
    } else if !regressed.is_empty() {
        (
            CheckStatus::Fail,
            format!(
                "regression beyond {MAX_REGRESSION_POINTS} pt on: {}",
                regressed.join(", ")
            ),
        )
    } else if !improved {
        (
            CheckStatus::Fail,
            "no metric improved over the untouched base".to_string(),
        )
    } else {
        (
            CheckStatus::Pass,
            "no regression and at least one metric improved on the frozen suite".to_string(),
        )
    };
    (metrics, status, why)
}
