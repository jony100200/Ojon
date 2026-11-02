//! Fail-closed audit for local community-model screening receipts.
//! No network/model access. This never promotes a model from phrase detectors.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

fn invalid(why: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, why.into())
}
fn field<'a>(v: &'a Value, name: &str) -> io::Result<&'a str> {
    v[name]
        .as_str()
        .filter(|s| !s.trim().is_empty())
        .ok_or_else(|| invalid(format!("missing or empty {name}")))
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[derive(Clone)]
struct Input {
    model: String,
    receipt_sha256: String,
    settings: Value,
    cases: BTreeMap<String, Value>,
    fingerprint: String,
}
fn parse(path: &Path) -> io::Result<Input> {
    let md = fs::metadata(path)?;
    if md.len() == 0 || md.len() > 32 * 1024 * 1024 {
        return Err(invalid(
            "community screen receipt must be nonempty and at most 32 MiB",
        ));
    }
    let bytes = fs::read(path)?;
    let root: Value = serde_json::from_slice(&bytes)?;
    if root["schema"] != "ks.community_model_screen.v1" {
        return Err(invalid("unsupported community-model screen schema"));
    }
    let model = field(&root, "model")?.to_owned();
    let settings = root["setting"]
        .as_object()
        .filter(|s| !s.is_empty())
        .ok_or_else(|| invalid("missing inference settings"))?;
    for required in [
        "api",
        "context",
        "reasoning",
        "max_output_tokens",
        "temperature",
    ] {
        if !settings.contains_key(required) {
            return Err(invalid(format!(
                "missing setting {required}; incomplete evidence"
            )));
        }
    }
    let case_list = root["cases"]
        .as_array()
        .filter(|v| !v.is_empty() && v.len() <= 10_000)
        .ok_or_else(|| invalid("cases must have 1..10000 rows"))?;
    let mut cases = BTreeMap::new();
    let mut identity = BTreeMap::new();
    for case in case_list {
        let id = field(case, "id")?.to_owned();
        let kind = field(case, "kind")?;
        let prompt = field(case, "prompt")?;
        let _ = case["answer"]
            .as_str()
            .ok_or_else(|| invalid(format!("{id}: missing answer string")))?;
        if !case["refusal_phrase_at_start"].is_boolean() {
            return Err(invalid(format!("{id}: refusal flag must be a boolean")));
        }
        let expected = case["expected"].as_str();
        if cases.insert(id.clone(), case.clone()).is_some() {
            return Err(invalid(format!("duplicate task ID: {id}")));
        }
        identity.insert(
            id,
            json!({"kind": kind, "prompt":prompt, "expected":expected}),
        );
    }
    Ok(Input {
        model,
        receipt_sha256: digest(&bytes),
        settings: Value::Object(settings.clone()),
        cases,
        fingerprint: digest(&serde_json::to_vec(&identity)?),
    })
}

fn checked_summary(input: &Input) -> Value {
    let mut numbers_checked = 0;
    let mut numbers_exact = 0;
    let mut json_checked = 0;
    let mut json_valid = 0;
    let mut ungraded = 0;
    let mut leading_flags = 0;
    let mut nonempty = 0;
    let mut errors = 0;
    let mut by_kind: BTreeMap<String, usize> = BTreeMap::new();
    for row in input.cases.values() {
        let kind = row["kind"].as_str().unwrap_or("unknown");
        *by_kind.entry(kind.to_owned()).or_default() += 1;
        let answer = row["answer"].as_str().unwrap_or("");
        if !answer.trim().is_empty() {
            nonempty += 1;
        }
        if row["error"].as_str().is_some_and(|s| !s.is_empty()) {
            errors += 1;
        }
        if row["refusal_phrase_at_start"] == true {
            leading_flags += 1;
        }
        if kind == "reasoning" && row["expected"].is_string() {
            numbers_checked += 1;
            if answer.trim() == row["expected"].as_str().unwrap_or("") {
                numbers_exact += 1;
            }
        } else if kind == "tools" {
            json_checked += 1;
            if serde_json::from_str::<Value>(answer).is_ok_and(|v| v.is_object()) {
                json_valid += 1;
            }
        } else {
            // Never treat a regex match in model-authored code as compilation.
            // Nor is absence of a leading refusal an independently verified helpful answer.
            ungraded += 1;
        }
    }
    json!({
        "model":input.model,
        "receipt_sha256":input.receipt_sha256,
        "task_count":input.cases.len(),
        "by_kind":by_kind,
        "nonempty":nonempty,
        "errors":errors,
        "leading_refusal_phrase_hits_UNVERIFIED":leading_flags,
        "strict_expected_answer":{"checked":numbers_checked,"passed":numbers_exact},
        "json_object_format":{"checked":json_checked,"passed":json_valid},
        "not_independently_graded":ungraded,
        "coding_compile_verified":false,
        "semantic_helpfulness_verified":false,
        "uncensored_verified":false
    })
}

/// Comparisons fail closed on missing/inconsistent tasks or inference settings.
/// Even when comparable, output remains an exploratory screen, never a release gate.
pub fn audit_community_screens(paths: &[PathBuf]) -> io::Result<Value> {
    if !(2..=32).contains(&paths.len()) {
        return Err(invalid(
            "compare between two and 32 community screening receipts",
        ));
    }
    let inputs = paths
        .iter()
        .map(|p| parse(p))
        .collect::<io::Result<Vec<_>>>()?;
    let mut names = HashSet::new();
    for input in &inputs {
        if !names.insert(input.model.clone()) {
            return Err(invalid(format!(
                "duplicate model identifier: {}",
                input.model
            )));
        }
    }
    let baseline = &inputs[0];
    for input in inputs.iter().skip(1) {
        if input.fingerprint != baseline.fingerprint {
            return Err(invalid(
                "task identity/prompt/expected mismatch: cannot compare screens",
            ));
        }
    }
    let same_settings = inputs.iter().all(|i| i.settings == baseline.settings);
    let summaries = inputs.iter().map(checked_summary).collect::<Vec<_>>();
    let complete = summaries
        .iter()
        .all(|s| s["errors"] == 0 && s["nonempty"] == s["task_count"]);
    let settings = inputs
        .iter()
        .map(|i| json!({"model":i.model, "setting":i.settings}))
        .collect::<Vec<_>>();
    Ok(json!({
        "schema":"ojon.community_challenge_audit.v1",
        "status":if !same_settings {"INCOMPARABLE_SETTINGS"} else if !complete {"INCOMPLETE_EVIDENCE"} else {"COMPARABLE_SCREEN_ONLY"},
        "suite_fingerprint_sha256":baseline.fingerprint,
        "settings_match":same_settings,
        "all_responses_complete":complete,
        "settings":settings,
        "models":summaries,
        "release_accepted":false,
        "independent_benchmark":false,
        "semantic_helpfulness_verified":false,
        "code_execution_covered_by_this_audit":false,
        "warning":"Same prompts and settings are necessary, not sufficient. Leading refusal phrases are not semantic helpfulness. Regex coding checks are not compilation. No independent verified model-quality or uncensorship winner is selected."
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn receipt(model: &str, ctx: u32, answer: &str) -> Value {
        json!({"schema":"ks.community_model_screen.v1","model":model,
        "setting":{"api":"LMStudio /api/v1/chat","context":ctx,"reasoning":"off","max_output_tokens":320,"temperature":0},
        "cases":[
            {"id":"math","kind":"reasoning","prompt":"Compute 2 + 2.","expected":"4","answer":answer,"refusal_phrase_at_start":false,"objective_pass":true,"error":null},
            {"id":"rust","kind":"coding","prompt":"Write Rust function.","expected":null,"answer":"fn broken() {}","refusal_phrase_at_start":false,"objective_pass":true,"error":null}
        ]})
    }
    fn write_temp(model: &str, ctx: u32, answer: &str, nonce: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!(
            "ojon-challenge-{}-{}-{nonce}.json",
            std::process::id(),
            model
        ));
        fs::write(
            &p,
            serde_json::to_vec(&receipt(model, ctx, answer)).unwrap(),
        )
        .unwrap();
        p
    }
    #[test]
    fn same_inputs_and_settings_are_only_a_screen_not_release_proof() {
        let a = write_temp("alpha", 2048, "4", "a");
        let b = write_temp("beta", 2048, "wrong", "a");
        let r = audit_community_screens(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(r["status"], "COMPARABLE_SCREEN_ONLY");
        assert_eq!(r["models"][0]["strict_expected_answer"]["passed"], 1);
        assert_eq!(r["models"][1]["strict_expected_answer"]["passed"], 0);
        assert_eq!(r["models"][0]["not_independently_graded"], 1);
        assert_eq!(r["release_accepted"], false);
        fs::remove_file(a).unwrap();
        fs::remove_file(b).unwrap();
    }
    #[test]
    fn mismatched_context_blocks_comparability() {
        let a = write_temp("alpha", 4096, "4", "b");
        let b = write_temp("beta", 2048, "4", "b");
        let r = audit_community_screens(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(r["status"], "INCOMPARABLE_SETTINGS");
        assert_eq!(r["settings_match"], false);
        fs::remove_file(a).unwrap();
        fs::remove_file(b).unwrap();
    }
    #[test]
    fn changed_task_and_fake_score_are_not_accepted() {
        let a = write_temp("alpha", 2048, "4", "c");
        let b = write_temp("beta", 2048, "wrong", "c");
        let mut r: Value = serde_json::from_slice(&fs::read(&b).unwrap()).unwrap();
        r["cases"][1]["prompt"] = json!("Different question.");
        fs::write(&b, serde_json::to_vec(&r).unwrap()).unwrap();
        assert!(audit_community_screens(&[a.clone(), b.clone()]).is_err());
        r["cases"][1]["prompt"] = json!("Write Rust function.");
        r["cases"][0]["answer"] = json!("wrong");
        r["cases"][0]["objective_pass"] = json!(true);
        fs::write(&b, serde_json::to_vec(&r).unwrap()).unwrap();
        let result = audit_community_screens(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(result["models"][1]["strict_expected_answer"]["passed"], 0);
        fs::remove_file(a).unwrap();
        fs::remove_file(b).unwrap();
    }
    #[test]
    fn duplicate_case_ids_rejected() {
        let a = write_temp("alpha", 2048, "4", "d");
        let b = write_temp("beta", 2048, "4", "d");
        let mut r: Value = serde_json::from_slice(&fs::read(&a).unwrap()).unwrap();
        r["cases"][1]["id"] = json!("math");
        fs::write(&a, serde_json::to_vec(&r).unwrap()).unwrap();
        assert!(audit_community_screens(&[a.clone(), b.clone()]).is_err());
        fs::remove_file(a).unwrap();
        fs::remove_file(b).unwrap();
    }
    #[test]
    fn empty_or_error_responses_never_count_as_comparable() {
        let a = write_temp("alpha", 2048, "4", "e");
        let b = write_temp("beta", 2048, "4", "e");
        let mut r: Value = serde_json::from_slice(&fs::read(&b).unwrap()).unwrap();
        r["cases"][0]["answer"] = json!("");
        r["cases"][0]["error"] = json!("timeout");
        fs::write(&b, serde_json::to_vec(&r).unwrap()).unwrap();
        let result = audit_community_screens(&[a.clone(), b.clone()]).unwrap();
        assert_eq!(result["status"], "INCOMPLETE_EVIDENCE");
        assert_eq!(result["all_responses_complete"], false);
        fs::remove_file(a).unwrap();
        fs::remove_file(b).unwrap();
    }
}
