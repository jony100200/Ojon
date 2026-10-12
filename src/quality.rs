//! Select the smallest *verified* quantized variant that meets a user's
//! quality tolerance. No inference is simulated; every metric is supplied
//! by a real, externally performed task evaluation and must share a suite.
//!
//! This module refuses to rank untested outputs or silently equate structural
//! GGUF validity with preserved task quality.

use serde_json::{json, Value};
use std::fs;
use std::io::{self, ErrorKind};
use std::path::Path;

fn bad(reason: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, reason.into())
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Direction { Higher, Lower }
impl Direction {
    fn parse(x: &str) -> io::Result<Self> {
        match x { "higher" => Ok(Self::Higher), "lower" => Ok(Self::Lower),
            _ => Err(bad("direction must be 'higher' or 'lower'")) }
    }
}
#[derive(Clone, Debug)]
pub struct Requirement {
    pub name: String,
    pub direction: Direction,
    pub max_relative_degradation_pct: f64,
}
#[derive(Clone, Debug)]
pub struct Candidate {
    pub label: String,
    pub bytes: u64,
    pub runtime_load_verified: bool,
    pub evaluation_verified: bool,
    pub metrics: Value,
}
#[derive(Clone, Debug)]
pub struct QualityReport {
    pub evaluation_suite: String,
    pub baseline: Value,
    pub requirements: Vec<Requirement>,
    pub candidates: Vec<Candidate>,
}
fn parse_score(metric: &Value, key: &str) -> io::Result<f64> {
    let v=metric.get(key).and_then(Value::as_f64)
        .ok_or_else(|| bad(format!("missing numeric metric {key}")))?;
    if !v.is_finite() { return Err(bad("metric must be finite")); }
    Ok(v)
}
pub fn parse_quality_report(path: &Path) -> io::Result<QualityReport> {
    let meta=fs::metadata(path)?;
    if meta.len()==0 || meta.len()>1024*1024 {
        return Err(bad("quality report must be 1 byte to 1 MiB"));
    }
    let contents=fs::read(path)?;
    let raw:Value=serde_json::from_slice(&contents).map_err(|e|bad(format!("invalid JSON: {e}")))?;
    if raw.get("schema").and_then(Value::as_str)!=Some("ojon.quality_evidence.v1") {
        return Err(bad("quality evidence requires schema ojon.quality_evidence.v1"));
    }
    let suite=raw.get("evaluation_suite").and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty()).ok_or_else(||bad("missing evaluation_suite"))?.to_owned();
    let baseline=raw.get("baseline").and_then(|v|v.get("metrics"))
        .ok_or_else(||bad("baseline.metrics missing"))?.clone();
    let reqs=raw.get("requirements").and_then(Value::as_array)
        .ok_or_else(||bad("requirements must be an array"))?;
    if reqs.is_empty() || reqs.len()>32 { return Err(bad("require 1..=32 quality requirements")); }
    let mut requirements=Vec::new();
    for entry in reqs {
        let name=entry.get("name").and_then(Value::as_str)
            .filter(|s| !s.is_empty()).ok_or_else(||bad("requirement name missing"))?.to_string();
        let direction=Direction::parse(entry.get("direction").and_then(Value::as_str).unwrap_or(""))?;
        let pct=entry.get("max_relative_degradation_pct").and_then(Value::as_f64)
            .ok_or_else(||bad("max_relative_degradation_pct missing"))?;
        if !pct.is_finite() || !(0.0..=100.0).contains(&pct) {
            return Err(bad("quality tolerance must be between 0 and 100 percent"));
        }
        let baseline_score=parse_score(&baseline, &name)?;
        if baseline_score<=0.0 {
            return Err(bad(format!("baseline metric {name} must be positive for relative comparison")));
        }
        if requirements.iter().any(|r: &Requirement|r.name==name) {
            return Err(bad(format!("duplicate metric requirement: {name}")));
        }
        requirements.push(Requirement {name, direction, max_relative_degradation_pct:pct});
    }
    let rows=raw.get("candidates").and_then(Value::as_array)
        .ok_or_else(||bad("candidates array missing"))?;
    if rows.is_empty() || rows.len()>256 { return Err(bad("require 1..=256 candidates")); }
    let mut candidates=Vec::new();
    for row in rows {
        let label=row.get("label").and_then(Value::as_str)
            .filter(|v|!v.is_empty()).ok_or_else(||bad("candidate label missing"))?.to_string();
        let bytes=row.get("bytes").and_then(Value::as_u64)
            .filter(|v| *v>0).ok_or_else(||bad("candidate bytes must be positive"))?;
        let verified=row.get("evaluation_verified").and_then(Value::as_bool).unwrap_or(false);
        let runtime=row.get("runtime_load_verified").and_then(Value::as_bool).unwrap_or(false);
        let metrics=row.get("metrics").cloned().unwrap_or(Value::Null);
        candidates.push(Candidate {label,bytes,runtime_load_verified:runtime,evaluation_verified:verified,metrics});
    }
    Ok(QualityReport { evaluation_suite:suite,baseline,requirements,candidates })
}

pub fn recommend(report: &QualityReport) -> Value {
    let mut ranked=Vec::new();
    for candidate in &report.candidates {
        let mut reasons:Vec<String>=Vec::new();
        if !candidate.runtime_load_verified { reasons.push("runtime load not verified".into()); }
        if !candidate.evaluation_verified { reasons.push("task evaluation not verified".into()); }
        let mut metrics=Vec::new();
        for req in &report.requirements {
            let reference=parse_score(&report.baseline,&req.name).unwrap_or(f64::NAN);
            match parse_score(&candidate.metrics,&req.name) {
                Ok(measured) => {
                    let degradation=match req.direction {
                        Direction::Higher => ((reference-measured)/reference)*100.0,
                        Direction::Lower => ((measured-reference)/reference)*100.0,
                    };
                    let pass=degradation<=req.max_relative_degradation_pct+1e-12;
                    if !pass {reasons.push(format!("{} quality degraded {:.4}% vs allowed {:.4}%",
                        req.name,degradation,req.max_relative_degradation_pct));}
                    metrics.push(json!({"name":req.name,"baseline":reference,"measured":measured,
                        "relative_degradation_pct":degradation,"allowed_pct":req.max_relative_degradation_pct,
                        "passes":pass}));
                }
                Err(_) => reasons.push(format!("missing measured metric {}",req.name)),
            }
        }
        ranked.push(json!({"label":candidate.label,"bytes":candidate.bytes,
            "eligible":reasons.is_empty(),"reasons":reasons,"comparisons":metrics}));
    }
    ranked.sort_by(|a,b|a["bytes"].as_u64().cmp(&b["bytes"].as_u64()));
    let best=ranked.iter().find(|v|v["eligible"].as_bool()==Some(true))
        .map(|v|v["label"].clone());
    json!({"schema":"ojon.quality_recommendation.v1",
        "evaluation_suite":report.evaluation_suite,
        "selection_policy":"smallest_runtime_verified_candidate_meeting_all_supplied_quality_thresholds",
        "recommendation":best,
        "candidates":ranked,
        "warning":"User-supplied evidence is not independently verified by Ojon; benchmark protocol, sample size, and comparability must be audited. No model was run to compute this ranking.",
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn choose_smallest_verified_quality_preserving_variant() {
        let report=QualityReport {
            evaluation_suite:"fixed_task_suite".into(),
            baseline:json!({"accuracy":0.90}),
            requirements:vec![Requirement {name:"accuracy".into(),direction:Direction::Higher,max_relative_degradation_pct:1.0}],
            candidates:vec![
                Candidate {label:"Q3".into(),bytes:3,runtime_load_verified:true,evaluation_verified:true,metrics:json!({"accuracy":0.85})},
                Candidate {label:"Q4".into(),bytes:4,runtime_load_verified:true,evaluation_verified:true,metrics:json!({"accuracy":0.895})},
                Candidate {label:"Q5".into(),bytes:5,runtime_load_verified:true,evaluation_verified:true,metrics:json!({"accuracy":0.899})},
                Candidate {label:"Q2_UNTESTED".into(),bytes:2,runtime_load_verified:false,evaluation_verified:false,metrics:json!({"accuracy":0.9})},
            ],
        };
        let result=recommend(&report);
        assert_eq!(result["recommendation"],"Q4");
    }
    #[test]
    fn reject_untested() {
        let report=QualityReport {
            evaluation_suite:"vision_ocr".into(),baseline:json!({"word_accuracy":0.94}),
            requirements:vec![Requirement {name:"word_accuracy".into(),direction:Direction::Higher,max_relative_degradation_pct:0.5}],
            candidates:vec![Candidate {label:"Q5".into(),bytes:8,runtime_load_verified:false,evaluation_verified:false,metrics:Value::Null}],
        };
        let result=recommend(&report);
        assert!(result["recommendation"].is_null());
    }
}
