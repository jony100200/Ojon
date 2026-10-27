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
    /// Explicitly measured runtime evidence; null does not establish hardware fit.
    pub runtime_measurements: Value,
}

#[derive(Clone, Debug)]
pub struct HardwareRequirements {
    pub backend: Option<String>,
    pub max_peak_ram_bytes: Option<u64>,
    pub max_peak_vram_bytes: Option<u64>,
    pub min_decode_tps: Option<f64>,
    pub min_prefill_tps: Option<f64>,
}

fn parse_hardware_requirements(value: &Value) -> io::Result<HardwareRequirements> {
    let object = value.as_object().ok_or_else(||bad("hardware_requirements must be an object"))?;
    if object.is_empty() { return Err(bad("hardware_requirements cannot be empty")); }
    for key in object.keys() {
        if !matches!(key.as_str(),
            "backend" | "max_peak_ram_bytes" | "max_peak_vram_bytes"
            | "min_decode_tps" | "min_prefill_tps")
        { return Err(bad(format!("unknown hardware requirement: {key}"))); }
    }
    let backend = match object.get("backend") {
        None => None,
        Some(v) => Some(v.as_str().filter(|s|!s.trim().is_empty())
            .ok_or_else(||bad("backend must be a nonempty string"))?.to_owned()),
    };
    fn memory(value:&Value, key:&str) -> io::Result<Option<u64>> {
        match value.get(key) {
            None => Ok(None),
            Some(v) => Ok(Some(v.as_u64().filter(|n|*n>0)
                .ok_or_else(||bad(format!("{key} must be a positive integer")))?)),
        }
    }
    fn speed(value:&Value, key:&str) -> io::Result<Option<f64>> {
        match value.get(key) {
            None => Ok(None),
            Some(v) => Ok(Some(v.as_f64().filter(|n|n.is_finite() && *n>0.0)
                .ok_or_else(||bad(format!("{key} must be a positive finite number")))?)),
        }
    }
    Ok(HardwareRequirements {
        backend,
        max_peak_ram_bytes: memory(value,"max_peak_ram_bytes")?,
        max_peak_vram_bytes: memory(value,"max_peak_vram_bytes")?,
        min_decode_tps: speed(value,"min_decode_tps")?,
        min_prefill_tps: speed(value,"min_prefill_tps")?,
    })
}
#[derive(Clone, Debug)]
pub struct QualityReport {
    pub evaluation_suite: String,
    pub baseline: Value,
    pub requirements: Vec<Requirement>,
    pub candidates: Vec<Candidate>,
    /// Optional strict local hardware limits; absent retains v1 behavior.
    pub hardware_requirements: Option<HardwareRequirements>,
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
    let hardware_requirements=raw.get("hardware_requirements")
        .map(parse_hardware_requirements).transpose()?;
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
        let runtime_measurements=row.get("runtime_measurements").cloned().unwrap_or(Value::Null);
        if !runtime_measurements.is_null() && !runtime_measurements.is_object() {
            return Err(bad("runtime_measurements must be an object when supplied"));
        }
        candidates.push(Candidate {label,bytes,runtime_load_verified:runtime,
            evaluation_verified:verified,metrics,runtime_measurements});
    }
    Ok(QualityReport { evaluation_suite:suite,baseline,requirements,candidates,hardware_requirements })
}

/// Evaluate separately measured hardware requirements. Unknown readings fail closed.
fn hardware_checks(
    candidate: &Candidate, requirements: &HardwareRequirements, reasons: &mut Vec<String>
) -> Vec<Value> {
    let runtime=&candidate.runtime_measurements;
    if runtime.get("measured").and_then(Value::as_bool)!=Some(true) {
        reasons.push("runtime measurements not verified".into());
    }
    let mut checks=Vec::new();
    if let Some(expected)=requirements.backend.as_ref() {
        let actual=runtime.get("backend").and_then(Value::as_str);
        let passes=actual==Some(expected.as_str());
        if !passes { reasons.push(format!("runtime backend must be {expected}")); }
        checks.push(json!({"name":"backend","required":expected,"measured":actual,"passes":passes}));
    }
    fn check_memory(
        runtime:&Value, key:&str, limit:Option<u64>, reasons:&mut Vec<String>,
        checks:&mut Vec<Value>
    ) {
        if let Some(maximum)=limit {
            let observed=runtime.get(key).and_then(Value::as_u64);
            let passes=observed.is_some_and(|n|n<=maximum);
            if !passes { reasons.push(format!("{key} missing or exceeds budget {maximum}")); }
            checks.push(json!({"name":key,"max":maximum,"measured":observed,"passes":passes}));
        }
    }
    fn check_speed(
        runtime:&Value, key:&str, limit:Option<f64>, reasons:&mut Vec<String>,
        checks:&mut Vec<Value>
    ) {
        if let Some(minimum)=limit {
            let observed=runtime.get(key).and_then(Value::as_f64)
                .filter(|v|v.is_finite() && *v>=0.0);
            let passes=observed.is_some_and(|n|n>=minimum);
            if !passes { reasons.push(format!("{key} missing or below minimum {minimum}")); }
            checks.push(json!({"name":key,"min":minimum,"measured":observed,"passes":passes}));
        }
    }
    check_memory(runtime,"peak_ram_bytes",requirements.max_peak_ram_bytes,reasons,&mut checks);
    check_memory(runtime,"peak_vram_bytes",requirements.max_peak_vram_bytes,reasons,&mut checks);
    check_speed(runtime,"decode_tps",requirements.min_decode_tps,reasons,&mut checks);
    check_speed(runtime,"prefill_tps",requirements.min_prefill_tps,reasons,&mut checks);
    checks
}

/// All runners apply their held-out task/vision/PPL gates before reaching this
/// selector. Never permit status-only artifacts from unsupported routes.
pub fn rank_quality_gated_candidates(kind:&str,result:&Value)->Vec<Value>{
    let rows=if kind=="adaptive_quant_comparison"{&result["candidates"]}
        else{&result["results"]};
    let mut ranked=Vec::<(u64,Value)>::new();
    if let Some(rows)=rows.as_array(){
        for row in rows{
            let passed=match kind{
                "existing_quant_comparison"=>row["status"]=="NO_DETECTED_TEST_REGRESSION",
                "high_precision_quant_parity"=>row["status"]=="NO_DETECTED_TASK_REGRESSION",
                "adaptive_quant_comparison"=>row["status"]=="measured_task_suite"&&row["task_quality_pass"]==true,
                _=>false
            };
            if !passed{continue;}
            let path=if kind=="existing_quant_comparison"{row["path"].as_str()}
                else{row["output"].as_str()};
            if let (Some(path),Some(bytes))=(path,row["bytes"].as_u64()){
                if bytes>0&&!path.is_empty(){
                    ranked.push((bytes,json!({"name":row["name"].as_str().or_else(||row["type"].as_str()).unwrap_or("candidate"),
                        "path":path,"bytes":bytes})));
                }
            }
        }
    }
    ranked.sort_by(|a,b|a.0.cmp(&b.0).then_with(||a.1["path"].as_str().cmp(&b.1["path"].as_str())));
    ranked.into_iter().map(|(_,row)|row).collect()
}

/// Choose a measured tradeoff after strict held-out gates and backend loading.
/// Candidate list must be sorted smallest-first; benchmarks are from the SAME
/// settings and observed device, as checked by compare_bench_files.
/// Balanced policy: smallest variant within 85% of the best measured decode
/// throughput; Fastest: highest measured decode throughput.
pub fn select_workflow_goal(loadable:&[Value],benchmarks:&[Value],goal:&str)->(Value,&'static str){
    select_workflow_goal_with_ratio(loadable,benchmarks,goal,0.85)
}
pub fn select_workflow_goal_with_ratio(
    loadable:&[Value],benchmarks:&[Value],goal:&str,min_speed_ratio:f64
)->(Value,&'static str){
    if !min_speed_ratio.is_finite()||!(0.50..=1.0).contains(&min_speed_ratio){
        return (Value::Null,"invalid_balanced_speed_ratio");
    }
    if loadable.is_empty(){return (Value::Null,"no_loadable_quality_passing_candidate");}
    if goal=="smallest"{return (loadable[0].clone(),"smallest_quality_passing_loadable");}
    let measured:Vec<(&Value,f64)>=benchmarks.iter().filter_map(|bench|{
        if bench["status"]!="MEASURED"{return None;}
        let speed=bench["metrics"]["token_generation"]["tokens_per_second"].as_f64()?;
        if speed>0.0&&speed.is_finite(){Some((bench,speed))}else{None}
    }).collect();
    if measured.is_empty(){
        return if goal=="balanced"{(loadable[0].clone(),"size_only_no_comparable_speed_evidence")}
            else{(Value::Null,"fastest_requires_measured_comparable_benchmarks")};
    }
    let max_decode=measured.iter().map(|(_,speed)|*speed).fold(0.0f64,f64::max);
    let threshold=if goal=="fastest"{max_decode}else{max_decode*min_speed_ratio};
    let picked=loadable.iter().find(|candidate|{
        measured.iter().any(|(bench,speed)|bench["path"]==candidate["path"]&&*speed+1e-9>=threshold)
    });
    (picked.cloned().unwrap_or(Value::Null),
        if goal=="fastest"{"fastest_measured_decode_quality_passing"}
        else{"smallest_within_configured_fraction_of_fastest_decode"})
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
        let runtime_comparisons=report.hardware_requirements.as_ref()
            .map(|limits| hardware_checks(candidate,limits,&mut reasons)).unwrap_or_default();
        ranked.push(json!({"label":candidate.label,"bytes":candidate.bytes,
            "eligible":reasons.is_empty(),"reasons":reasons,
            "comparisons":metrics,"runtime_comparisons":runtime_comparisons}));
    }
    ranked.sort_by(|a,b|a["bytes"].as_u64().cmp(&b["bytes"].as_u64()));
    let best=ranked.iter().find(|v|v["eligible"].as_bool()==Some(true))
        .map(|v|v["label"].clone());
    json!({"schema":"ojon.quality_recommendation.v1",
        "evaluation_suite":report.evaluation_suite,
        "selection_policy":if report.hardware_requirements.is_some() {
            "smallest_runtime_verified_candidate_meeting_quality_and_hardware_requirements"
        } else { "smallest_runtime_verified_candidate_meeting_all_supplied_quality_thresholds" },
        "hardware_constraints_applied":report.hardware_requirements.is_some(),
        "recommendation":best,
        "candidates":ranked,
        "warning":"User-supplied evidence is not independently verified by Ojon; benchmark protocol, sample size, and comparability must be audited. No model was run to compute this ranking.",
    })
}

/// Evaluate whether any quality- and hardware-verified candidate also achieves
/// the required minimum byte-size reduction against `baseline_bytes`.
pub fn evaluate_optimization_readiness(
    report: &QualityReport,
    baseline_bytes: u64,
    min_saving_percent: f64,
) -> io::Result<Value> {
    if baseline_bytes == 0 {
        return Err(bad("baseline_bytes must be positive"));
    }
    if !min_saving_percent.is_finite() || !(0.0..=95.0).contains(&min_saving_percent) {
        return Err(bad("min_saving_percent must be within 0..=95 percent"));
    }
    let rec = recommend(report);
    let ranked = rec["candidates"].as_array().cloned().unwrap_or_default();
    let mut evaluated = Vec::with_capacity(ranked.len());
    let mut any_quality_eligible = false;
    for item in ranked {
        let bytes = item["bytes"].as_u64().unwrap_or(0);
        let eligible = item["eligible"].as_bool() == Some(true);
        if eligible {
            any_quality_eligible = true;
        }
        let saving_bytes = baseline_bytes.saturating_sub(bytes);
        let saving_percent = if bytes <= baseline_bytes {
            ((baseline_bytes - bytes) as f64 / baseline_bytes as f64) * 100.0
        } else {
            -(((bytes - baseline_bytes) as f64 / baseline_bytes as f64) * 100.0)
        };
        let saving_passes = bytes < baseline_bytes && saving_percent + 1e-12 >= min_saving_percent;
        let readiness_pass = eligible && saving_passes;
        evaluated.push(json!({
            "label": item["label"],
            "bytes": bytes,
            "quality_and_hardware_eligible": eligible,
            "saving_bytes": saving_bytes,
            "saving_percent": saving_percent,
            "saving_passes": saving_passes,
            "readiness_pass": readiness_pass,
            "reasons": item["reasons"]
        }));
    }
    let selected = evaluated
        .iter()
        .find(|c| c["readiness_pass"].as_bool() == Some(true))
        .cloned();
    let verdict = if selected.is_some() {
        "PASS"
    } else if any_quality_eligible {
        "REJECT_INSUFFICIENT_SAVING"
    } else {
        "REJECT_QUALITY_OR_HARDWARE"
    };
    Ok(json!({
        "schema": "ojon.optimization_readiness_gate.v1",
        "evaluation_suite": report.evaluation_suite,
        "baseline_bytes": baseline_bytes,
        "min_saving_percent": min_saving_percent,
        "ready": selected.is_some(),
        "verdict": verdict,
        "selected_candidate": selected,
        "candidates": evaluated
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn requirement() -> Requirement {
        Requirement { name:"accuracy".into(), direction:Direction::Higher,
            max_relative_degradation_pct:1.0 }
    }
    fn candidate(label:&str, bytes:u64, accuracy:f64) -> Candidate {
        Candidate { label:label.into(), bytes, runtime_load_verified:true,
            evaluation_verified:true, metrics:json!({"accuracy":accuracy}),
            runtime_measurements:Value::Null }
    }
    fn report(candidates:Vec<Candidate>) -> QualityReport {
        QualityReport { evaluation_suite:"held_out_code_and_tool_tests".into(),
            baseline:json!({"accuracy":0.90}), requirements:vec![requirement()],
            candidates, hardware_requirements:None }
    }

    #[test]
    fn choose_smallest_verified_quality_preserving_variant() {
        let mut r=report(vec![
            candidate("Q3",3,0.85), candidate("Q4",4,0.895),candidate("Q5",5,0.899),
            candidate("Q2_UNTESTED",2,0.9)
        ]);
        r.candidates[3].runtime_load_verified=false;
        r.candidates[3].evaluation_verified=false;
        assert_eq!(recommend(&r)["recommendation"],"Q4");
        assert_eq!(recommend(&r)["hardware_constraints_applied"],false);
    }

    #[test]
    fn configurable_balanced_tradeoff_uses_same_measured_evidence(){
        let loadable=vec![json!({"path":"small","bytes":80}),json!({"path":"fast","bytes":90})];
        let results=vec![
            json!({"path":"small","status":"MEASURED","metrics":{"token_generation":{"tokens_per_second":90.0}}}),
            json!({"path":"fast","status":"MEASURED","metrics":{"token_generation":{"tokens_per_second":100.0}}})
        ];
        assert_eq!(select_workflow_goal_with_ratio(&loadable,&results,"balanced",0.85).0["path"],"small");
        assert_eq!(select_workflow_goal_with_ratio(&loadable,&results,"balanced",0.95).0["path"],"fast");
        assert!(select_workflow_goal_with_ratio(&loadable,&results,"balanced",f64::NAN).0.is_null());
    }
    #[test]
    fn guided_selector_fails_closed_and_applies_measured_balanced_policy(){
        let report=json!({"results":[
            {"status":"FAILED_QUALITY_GATE","name":"bad","bytes":1,"path":"bad.gguf"},
            {"status":"NO_DETECTED_TEST_REGRESSION","name":"small","bytes":100,"path":"small.gguf"},
            {"status":"NO_DETECTED_TEST_REGRESSION","name":"faster","bytes":120,"path":"faster.gguf"}
        ]});
        let ranked=rank_quality_gated_candidates("existing_quant_comparison",&report);
        assert_eq!(ranked.len(),2);
        let measured=vec![
            json!({"path":"small.gguf","status":"MEASURED",
                "metrics":{"token_generation":{"tokens_per_second":80.0}}}),
            json!({"path":"faster.gguf","status":"MEASURED",
                "metrics":{"token_generation":{"tokens_per_second":100.0}}})
        ];
        assert_eq!(select_workflow_goal(&ranked,&measured,"smallest").0["path"],"small.gguf");
        assert_eq!(select_workflow_goal(&ranked,&measured,"balanced").0["path"],"faster.gguf");
        assert_eq!(select_workflow_goal(&ranked,&measured,"fastest").0["path"],"faster.gguf");
        assert!(select_workflow_goal(&ranked,&[],"fastest").0.is_null());
        let adaptive=json!({"candidates":[
            {"output":"adaptive.gguf","bytes":80,"status":"measured_task_suite","task_quality_pass":true},
            {"output":"failed.gguf","bytes":50,"status":"measured_task_suite","task_quality_pass":false}
        ]});
        assert_eq!(rank_quality_gated_candidates("adaptive_quant_comparison",&adaptive).len(),1);
    }
    #[test]
    fn reject_untested() {
        let mut r=report(vec![candidate("Q5",8,0.9)]);
        r.candidates[0].runtime_load_verified=false;
        r.candidates[0].evaluation_verified=false;
        assert!(recommend(&r)["recommendation"].is_null());
    }

    #[test]
    fn actual_hardware_fit_filters_smaller_quality_passing_models() {
        let mut r=report(vec![
            candidate("Q3-no-bench",3,0.9),
            candidate("Q4-too-slow",4,0.898),
            candidate("Q5-fits",5,0.899)
        ]);
        r.hardware_requirements=Some(HardwareRequirements {
            backend:Some("intel-vulkan".into()),max_peak_ram_bytes:Some(30_000),
            max_peak_vram_bytes:Some(20_000),min_decode_tps:Some(10.0),
            min_prefill_tps:Some(100.0)
        });
        r.candidates[1].runtime_measurements=json!({"measured":true,"backend":"intel-vulkan",
            "peak_ram_bytes":9000,"peak_vram_bytes":12000,"decode_tps":7.0,"prefill_tps":200.0});
        r.candidates[2].runtime_measurements=json!({"measured":true,"backend":"intel-vulkan",
            "peak_ram_bytes":8000,"peak_vram_bytes":15000,"decode_tps":14.0,"prefill_tps":125.0});
        let result=recommend(&r);
        assert_eq!(result["recommendation"],"Q5-fits");
        assert_eq!(result["hardware_constraints_applied"],true);
        assert_eq!(result["candidates"][0]["eligible"],false);
        assert_eq!(result["candidates"][1]["eligible"],false);
        assert_eq!(result["candidates"][2]["eligible"],true);
    }

    #[test]
    fn incompatible_backend_and_excess_vram_fail_closed() {
        let mut r=report(vec![candidate("Q4",4,0.9)]);
        r.hardware_requirements=Some(HardwareRequirements {
            backend:Some("intel-sycl".into()),max_peak_ram_bytes:None,
            max_peak_vram_bytes:Some(16),min_decode_tps:None,min_prefill_tps:None
        });
        r.candidates[0].runtime_measurements=json!({"measured":true,"backend":"cuda",
            "peak_vram_bytes":17});
        assert!(recommend(&r)["recommendation"].is_null());
    }

    #[test]
    fn parse_hardware_limit_rejects_unknown_fields_and_invalid_values() {
        assert!(parse_hardware_requirements(&json!({"max_peak_vram":123})).is_err());
        assert!(parse_hardware_requirements(&json!({"min_decode_tps":-1.0})).is_err());
        assert!(parse_hardware_requirements(&json!({})).is_err());
        assert!(parse_hardware_requirements(&json!({"max_peak_ram_bytes":100})).is_ok());
    }

    #[test]
    fn parse_report_accepts_hardware_evidence_and_rejects_unmeasured_runtime() {
        let now=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path=std::env::temp_dir().join(format!("ojon-quality-hardware-{}-{now}.json",std::process::id()));
        let payload=json!({"schema":"ojon.quality_evidence.v1",
            "evaluation_suite":"held_out_code",
            "baseline":{"metrics":{"accuracy":0.9}},
            "requirements":[{"name":"accuracy","direction":"higher","max_relative_degradation_pct":1.0}],
            "hardware_requirements":{"backend":"intel-vulkan","max_peak_vram_bytes":20000},
            "candidates":[{"label":"Q4","bytes":3000,"runtime_load_verified":true,
                "evaluation_verified":true,"metrics":{"accuracy":0.9},
                "runtime_measurements":{"backend":"intel-vulkan","peak_vram_bytes":10000}}]
        });
        fs::write(&path,serde_json::to_vec(&payload).unwrap()).unwrap();
        let result=parse_quality_report(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert!(recommend(&result)["recommendation"].is_null());
    }

    #[test]
    fn readiness_gate_selects_smallest_quality_and_size_passing_candidate() {
        let r = report(vec![
            candidate("Q3_DEGRADED", 400, 0.80),
            candidate("Q4_PASS", 700, 0.895),
            candidate("Q8_TOO_LARGE", 980, 0.90),
        ]);
        let gate = evaluate_optimization_readiness(&r, 1000, 20.0).unwrap();
        assert_eq!(gate["schema"], "ojon.optimization_readiness_gate.v1");
        assert_eq!(gate["ready"], true);
        assert_eq!(gate["verdict"], "PASS");
        assert_eq!(gate["selected_candidate"]["label"], "Q4_PASS");
        assert_eq!(gate["selected_candidate"]["saving_bytes"], 300);
    }

    #[test]
    fn exact_boundary_saving_passes_readiness_gate() {
        let r = report(vec![candidate("Q4_BOUNDARY", 750, 0.898)]);
        let gate = evaluate_optimization_readiness(&r, 1000, 25.0).unwrap();
        assert_eq!(gate["ready"], true);
        assert_eq!(gate["verdict"], "PASS");
        assert_eq!(gate["selected_candidate"]["label"], "Q4_BOUNDARY");
        let rejected = evaluate_optimization_readiness(&r, 1000, 30.0).unwrap();
        assert_eq!(rejected["ready"], false);
        assert_eq!(rejected["verdict"], "REJECT_INSUFFICIENT_SAVING");
    }
}
