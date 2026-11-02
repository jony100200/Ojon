//! Strict validation of Ojon's own live task-evaluation receipts.
//!
//! This compares real recorded task rows rather than trusting aggregate scores.
//! It is a receipt consistency check, not a cryptographic provenance or an
//! independent replay of the model.

use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::{self, ErrorKind};
use std::path::Path;

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, s.into())
}
fn text<'a>(v: &'a Value, field: &str) -> io::Result<&'a str> {
    v.get(field)
        .and_then(Value::as_str)
        .filter(|v| !v.trim().is_empty())
        .ok_or_else(|| invalid(format!("missing nonempty {field}")))
}
fn yes_no(v: &Value, field: &str) -> io::Result<bool> {
    v.get(field)
        .and_then(Value::as_bool)
        .ok_or_else(|| invalid(format!("missing boolean {field}")))
}
#[derive(Default)]
struct Counts {
    correct: usize,
    seen: usize,
}
fn rows<'a>(v: &'a Value) -> io::Result<&'a [Value]> {
    v.get("results")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .filter(|x| !x.is_empty() && x.len() <= 10000)
        .ok_or_else(|| invalid("requires 1..10000 task result rows"))
}
fn parse_rows(v: &Value) -> io::Result<BTreeMap<String, (String, bool)>> {
    let mut out = BTreeMap::new();
    for row in rows(v)? {
        let id = text(row, "id")?;
        let category = text(row, "category")?;
        let pass = yes_no(row, "pass")?;
        if !matches!(category, "coding" | "reasoning" | "tool_call") {
            return Err(invalid(format!("unsupported category in task {id}")));
        }
        if row.get("answer").and_then(Value::as_str).is_none() {
            return Err(invalid(format!("missing generated answer for task {id}")));
        }
        if !row
            .get("elapsed_seconds")
            .and_then(Value::as_f64)
            .is_some_and(|time| time.is_finite() && time >= 0.0)
        {
            return Err(invalid(format!("invalid elapsed time for task {id}")));
        }
        if out.insert(id.into(), (category.into(), pass)).is_some() {
            return Err(invalid(format!("duplicate task id {id}")));
        }
    }
    Ok(out)
}
fn metrics_for(rows: &BTreeMap<String, (String, bool)>) -> BTreeMap<&'static str, Counts> {
    let mut metrics = BTreeMap::new();
    for category in ["coding", "reasoning", "tool_call"] {
        metrics.insert(category, Counts::default());
    }
    for (category, passed) in rows.values() {
        let c = metrics.get_mut(category.as_str()).unwrap();
        c.seen += 1;
        c.correct += usize::from(*passed);
    }
    metrics
}
fn validate_claimed_metrics(
    tests: &Value,
    rows: &BTreeMap<String, (String, bool)>,
    context: &str,
) -> io::Result<()> {
    let measured = metrics_for(rows);
    for (category, metric) in measured {
        if metric.seen == 0 {
            return Err(invalid(format!("{context}: missing category {category}")));
        }
        let key = match category {
            "coding" => "coding_accuracy",
            "reasoning" => "reasoning_accuracy",
            _ => "tool_call_accuracy",
        };
        let claimed = tests
            .get("metrics")
            .and_then(|m| m.get(key))
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite())
            .ok_or_else(|| invalid(format!("{context}: missing {key}")))?;
        let actual = 100.0 * metric.correct as f64 / metric.seen as f64;
        if (actual - claimed).abs() > 0.000001 {
            return Err(invalid(format!(
                "{context}: {key} disagrees with recorded rows"
            )));
        }
    }
    Ok(())
}

pub fn audit_task_receipt(v: &Value) -> io::Result<Value> {
    if v.get("schema").and_then(Value::as_str) != Some("ojon.quality_evidence.v1") {
        return Err(invalid("expected ojon.quality_evidence.v1 receipt"));
    }
    let suite = text(v, "evaluation_suite")?;
    let config = v
        .get("evaluation_config")
        .filter(|x| x.is_object())
        .ok_or_else(|| invalid("missing evaluation_config"))?;
    // The current task evaluator emits this fingerprint plus runtime settings.
    if config
        .get("suite_fingerprint_noncrypto")
        .and_then(Value::as_u64)
        .is_none()
        || config.get("tokens").and_then(Value::as_u64).is_none()
        || config.get("tool").and_then(Value::as_str).is_none()
    {
        return Err(invalid("missing suite/runtime fingerprint in task receipt"));
    }
    let baseline = v
        .get("baseline")
        .ok_or_else(|| invalid("missing baseline"))?;
    text(baseline, "model")?;
    let baseline_tests = baseline
        .get("tests")
        .ok_or_else(|| invalid("missing baseline tests"))?;
    let baseline_rows = parse_rows(baseline_tests)?;
    validate_claimed_metrics(baseline_tests, &baseline_rows, "baseline")?;
    let candidates = v
        .get("candidates")
        .and_then(Value::as_array)
        .filter(|a| !a.is_empty() && a.len() <= 64)
        .ok_or_else(|| invalid("needs 1..64 candidate summaries"))?;
    let details = v
        .get("detailed_candidate_results")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("missing detailed_candidate_results"))?;
    if details.len() != candidates.len() {
        return Err(invalid("candidate summary/detail length mismatch"));
    }
    let mut seen = BTreeSet::new();
    let mut reports = Vec::new();
    for (candidate, detail) in candidates.iter().zip(details) {
        let name = text(candidate, "label")?;
        if !seen.insert(name.to_string()) || text(detail, "label")? != name {
            return Err(invalid("candidate labels are duplicate or mismatched"));
        }
        text(detail, "model")?;
        if !yes_no(candidate, "runtime_load_verified")?
            || !yes_no(candidate, "evaluation_verified")?
        {
            return Err(invalid(format!("{name}: runtime/evaluation not verified")));
        }
        let tests = detail
            .get("tests")
            .ok_or_else(|| invalid(format!("{name}: missing tests")))?;
        let candidate_rows = parse_rows(tests)?;
        if baseline_rows.len() != candidate_rows.len() {
            return Err(invalid(format!(
                "{name}: incomplete candidate task results"
            )));
        }
        validate_claimed_metrics(tests, &candidate_rows, name)?;
        if tests.get("metrics") != candidate.get("metrics") {
            return Err(invalid(format!("{name}: summary metrics mismatch detail")));
        }
        let mut counts = BTreeMap::<&str, (usize, usize, usize, usize)>::new();
        for category in ["coding", "reasoning", "tool_call"] {
            counts.insert(category, (0, 0, 0, 0));
        }
        let mut original_correct = 0usize;
        let mut retained = 0usize;
        let mut gained = 0usize;
        let mut degraded = 0usize;
        for (id, (category, original_pass)) in &baseline_rows {
            let (candidate_category, candidate_pass) = candidate_rows
                .get(id)
                .ok_or_else(|| invalid(format!("{name}: missing task {id}")))?;
            if candidate_category != category {
                return Err(invalid(format!("{name}: category changed for {id}")));
            }
            let entry = counts.get_mut(category.as_str()).unwrap();
            entry.0 += 1;
            entry.1 += usize::from(*original_pass);
            entry.2 += usize::from(*candidate_pass);
            entry.3 += usize::from(*original_pass && !candidate_pass);
            original_correct += usize::from(*original_pass);
            retained += usize::from(*original_pass && *candidate_pass);
            gained += usize::from(!original_pass && *candidate_pass);
            degraded += usize::from(*original_pass && !candidate_pass);
        }
        let retention = if original_correct == 0 {
            0.0
        } else {
            retained as f64 / original_correct as f64
        };
        let distribution = counts
            .iter()
            .map(|(category, (n, base, after, lost))| {
                (
                    category.to_string(),
                    json!({"count":n, "baseline_pass":base,
                "candidate_pass":after, "regressions":lost}),
                )
            })
            .collect::<serde_json::Map<String, Value>>();
        let sample_large_enough =
            baseline_rows.len() >= 120 && counts.values().all(|(n, _, _, _)| *n >= 20);
        let no_major_regression = original_correct > 0 && retention >= 0.95;
        let status = if !no_major_regression {
            "REGRESSION_DETECTED"
        } else if !sample_large_enough {
            "SMOKE_ONLY_INSUFFICIENT_COVERAGE"
        } else {
            "TASK_SCREEN_PASS_NOT_INDEPENDENTLY_ATTESTED"
        };
        reports.push(json!({"label":name, "status":status,
            "task_count":baseline_rows.len(), "originally_correct":original_correct,
            "retained_correct":retained,"newly_correct":gained,"regressions":degraded,
            "retention":retention,"sample_large_enough":sample_large_enough,
            "category_counts":distribution}));
    }
    Ok(json!({"schema":"ojon.task_receipt_audit.v1",
        "status": if reports.iter().any(|r|r["status"]=="REGRESSION_DETECTED") {
            "REGRESSION_DETECTED"
        } else if reports.iter().any(|r|r["status"]=="SMOKE_ONLY_INSUFFICIENT_COVERAGE") {
            "SMOKE_ONLY"
        } else {"TASK_SCREEN_PASS_NOT_INDEPENDENTLY_ATTESTED"},
        "suite":suite,
        "evaluation_config":config,
        "baseline_rows":baseline_rows.len(),
        "candidates":reports,
        "limitations":"External model output and scorer judgements are not independently attested. Suite fingerprint is non-cryptographic. This receipt covers exact/JSON functional tasks only, not refusals, multilingual, vision or general reasoning."}))
}

pub fn audit_task_receipt_file(path: &Path) -> io::Result<Value> {
    let metadata = fs::metadata(path)?;
    if metadata.len() == 0 || metadata.len() > 64 * 1024 * 1024 {
        return Err(invalid("task receipt must be 1 byte..64 MiB"));
    }
    let bytes = fs::read(path)?;
    let bytes = bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes);
    let input: Value = serde_json::from_slice(bytes)
        .map_err(|err| invalid(format!("invalid task receipt JSON: {err}")))?;
    audit_task_receipt(&input)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Value {
        let mut rows = Vec::new();
        for i in 0..120 {
            rows.push(json!({"id":format!("task-{i}"),
                "category":(["coding","reasoning","tool_call"][i%3]),
                "pass":true,"answer":"ok","elapsed_seconds":0.5}));
        }
        let metrics = json!({"coding_accuracy":100.0,
            "reasoning_accuracy":100.0,"tool_call_accuracy":100.0});
        json!({"schema":"ojon.quality_evidence.v1","evaluation_suite":"fixture",
            "evaluation_config":{"suite_fingerprint_noncrypto":1234,"tokens":128,"tool":"llama-cli"},
            "baseline":{"model":"original.gguf","tests":{"results":rows,"metrics":metrics}},
            "candidates":[{"label":"q4","metrics":metrics,
                "runtime_load_verified":true,"evaluation_verified":true}],
            "detailed_candidate_results":[{"label":"q4","model":"candidate.gguf",
                "tests":{"results":rows,"metrics":metrics}}]})
    }
    #[test]
    fn matched_receipt_checks_all_rows() {
        let v = audit_task_receipt(&fixture()).unwrap();
        assert_eq!(v["status"], "TASK_SCREEN_PASS_NOT_INDEPENDENTLY_ATTESTED");
        assert_eq!(v["candidates"][0]["retention"], 1.0);
    }
    #[test]
    fn small_set_is_smoke_only() {
        let mut v = fixture();
        v["baseline"]["tests"]["results"]
            .as_array_mut()
            .unwrap()
            .truncate(6);
        v["detailed_candidate_results"][0]["tests"]["results"]
            .as_array_mut()
            .unwrap()
            .truncate(6);
        let v = audit_task_receipt(&v).unwrap();
        assert_eq!(v["status"], "SMOKE_ONLY");
    }
    #[test]
    fn corrupted_details_and_tampered_scores_rejected() {
        let mut v = fixture();
        v["detailed_candidate_results"][0]["tests"]["results"][1]["id"] = json!("task-0");
        assert!(audit_task_receipt(&v).is_err());
        let mut v = fixture();
        v["candidates"][0]["metrics"]["coding_accuracy"] = json!(80.0);
        assert!(audit_task_receipt(&v).is_err());
        let mut v = fixture();
        v["detailed_candidate_results"][0]["tests"]["results"][1]["category"] = json!("coding");
        assert!(audit_task_receipt(&v).is_err());
        let mut v = fixture();
        v["detailed_candidate_results"][0]["tests"]["results"]
            .as_array_mut()
            .unwrap()
            .pop();
        assert!(audit_task_receipt(&v).is_err());
    }
    #[test]
    fn ability_regression_is_not_a_pass() {
        let mut v = fixture();
        for i in 0..8 {
            v["detailed_candidate_results"][0]["tests"]["results"][i]["pass"] = json!(false);
        }
        for cat in ["coding", "reasoning", "tool_call"] {
            let drop = match cat {
                "coding" => 3,
                "reasoning" => 3,
                _ => 2,
            };
            v["detailed_candidate_results"][0]["tests"]["metrics"][format!("{cat}_accuracy")] =
                json!(100.0 - 2.5 * drop as f64);
            v["candidates"][0]["metrics"][format!("{cat}_accuracy")] =
                json!(100.0 - 2.5 * drop as f64);
        }
        let v = audit_task_receipt(&v).unwrap();
        assert_eq!(v["status"], "REGRESSION_DETECTED");
    }
}
