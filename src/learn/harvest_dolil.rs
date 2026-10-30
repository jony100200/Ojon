//! Harvest Dolil agent runs (`<state>/agent_runs/*.jsonl`) into Ojon Learn episodes.
//!
//! Each file is one run: an event stream with one JSON object per line and an
//! `event` key. The events that carry the conversation:
//!   started              { goal }                      the user prompt
//!   assistant_tool_calls { turn, message{tool_calls[{id, function{name, arguments}}]} }
//!   tool_result          { turn, message{tool_call_id, content: JSON string of
//!                          receipts [{action, status: succeeded|failed, output|error}] } }
//!   assistant_final      { turn, content }             the assistant's answer
//!   model_error          { turn, error }               a route/server failure
//! Everything else (budgets, token usage, recall, routing) is bookkeeping.
//!
//! One file is one episode. Outcome: the transcript does not say whether the
//! answer was correct (Dolil's hand-judged labels live elsewhere), so an
//! episode with a final answer is `Unverified`; a run that ended in
//! `model_error` with no answer is `FailedRetry` (the route failed, which the
//! curator keeps as a negative), and tool receipt statuses are recorded as
//! counts for later labelling.
use serde_json::Value;
use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use super::collector::{CollectedEpisode, EpisodeOutcome, ToolCallRecord};

const MAX_RESPONSE_BYTES: usize = 64 * 1024;

/// Sniff: first non-empty line is a JSON object with an `event` string and no
/// Claude Code `sessionId`/`type` pair.
pub fn is_dolil_run(path: &Path) -> bool {
    let Ok(file) = fs::File::open(path) else { return false };
    for line in BufReader::new(file).lines().take(5) {
        let Ok(line) = line else { return false };
        if line.trim().is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(&line) else { return false };
        return v.get("event").and_then(Value::as_str).is_some() && v.get("sessionId").is_none();
    }
    false
}

fn receipt_statuses(content: &str) -> (usize, usize) {
    let Ok(v) = serde_json::from_str::<Value>(content) else { return (0, 0) };
    let mut ok = 0;
    let mut failed = 0;
    if let Some(items) = v.as_array() {
        for r in items {
            match r.get("status").and_then(Value::as_str) {
                Some("succeeded") => ok += 1,
                Some("failed") => failed += 1,
                _ => {}
            }
        }
    }
    (ok, failed)
}

pub fn parse_dolil_run(path: &Path) -> io::Result<Vec<CollectedEpisode>> {
    let file = fs::File::open(path)?;
    let run_id = path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "run".into());
    let mut goal: Option<String> = None;
    let mut finals: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCallRecord> = Vec::new();
    let (mut ok, mut failed) = (0usize, 0usize);
    let mut model_errors: Vec<String> = Vec::new();
    for line in BufReader::new(file).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
        match v.get("event").and_then(Value::as_str) {
            Some("started") => goal = v.get("goal").and_then(Value::as_str).map(|s| s.trim().to_string()),
            Some("assistant_tool_calls") => {
                if let Some(calls) = v.pointer("/message/tool_calls").and_then(Value::as_array) {
                    for c in calls {
                        let f = c.get("function").cloned().unwrap_or(Value::Null);
                        let name = f.get("name").and_then(Value::as_str).unwrap_or("unknown").to_string();
                        let args_raw = f.get("arguments").and_then(Value::as_str).unwrap_or("{}");
                        let arguments = serde_json::from_str::<Value>(args_raw).unwrap_or_else(|_| Value::String(args_raw.to_string()));
                        tool_calls.push(ToolCallRecord {
                            id: c.get("id").and_then(Value::as_str).unwrap_or("").to_string(),
                            name,
                            arguments,
                        });
                    }
                }
            }
            Some("tool_result") => {
                if let Some(content) = v.pointer("/message/content").and_then(Value::as_str) {
                    let (o, f) = receipt_statuses(content);
                    ok += o;
                    failed += f;
                }
            }
            Some("assistant_final") => {
                if let Some(c) = v.get("content").and_then(Value::as_str) {
                    if !c.trim().is_empty() {
                        finals.push(c.trim().to_string());
                    }
                }
            }
            Some("model_error") => {
                if let Some(e) = v.get("error").and_then(Value::as_str) {
                    model_errors.push(e.chars().take(300).collect());
                }
            }
            _ => {}
        }
    }
    let Some(goal) = goal.filter(|g| !g.is_empty()) else { return Ok(Vec::new()) };
    let mut response = finals.join("\n");
    if response.len() > MAX_RESPONSE_BYTES {
        let mut cut = MAX_RESPONSE_BYTES;
        while !response.is_char_boundary(cut) {
            cut -= 1;
        }
        response.truncate(cut);
        response.push_str("\n[... truncated]");
    }
    if response.is_empty() && tool_calls.is_empty() && model_errors.is_empty() {
        return Ok(Vec::new());
    }
    let answered = !response.is_empty();
    let mut ep = CollectedEpisode::new(format!("dolil:{run_id}"), run_id, "dolil", "dolil-local-model", goal, response);
    ep.tool_calls = tool_calls;
    ep.tool_results = Some(vec![serde_json::json!({
        "receipts_succeeded": ok, "receipts_failed": failed, "model_errors": model_errors,
        "outcome_note": "correctness not in transcript; see Dolil labels.json for hand-judged runs"
    })]);
    ep.outcome = if answered { EpisodeOutcome::Unverified } else { EpisodeOutcome::FailedRetry };
    ep.unit_tests_passed = None;
    ep.scalar_reward = ep.outcome.default_reward();
    Ok(vec![ep])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_run(dir: &Path, name: &str, lines: &[&str]) -> std::path::PathBuf {
        let p = dir.join(name);
        let mut f = fs::File::create(&p).unwrap();
        for l in lines {
            writeln!(f, "{l}").unwrap();
        }
        p
    }

    #[test]
    fn sniff_and_parse_a_full_run() {
        let dir = std::env::temp_dir().join(format!("ojon-dolil-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = write_run(&dir, "v3-1.jsonl", &[
            r#"{"event": "started", "goal": "List the crates directory.", "tools_exposed": 4}"#,
            r#"{"event": "request_budget", "max_turns": 6}"#,
            r#"{"event": "assistant_tool_calls", "message": {"content": "", "role": "assistant", "tool_calls": [{"function": {"arguments": "{\"path\": \"crates\"}", "name": "file.list"}, "id": "K1", "type": "function"}]}, "turn": 0}"#,
            r#"{"event": "tool_result", "message": {"content": "[{\"action\":\"file.list\",\"status\":\"succeeded\",\"output\":\"dir bidda\"},{\"action\":\"x\",\"status\":\"failed\",\"error\":\"nope\"}]", "role": "tool", "tool_call_id": "K1"}, "turn": 0}"#,
            r#"{"content": "The crates directory contains bidda.", "event": "assistant_final", "turn": 1}"#,
        ]);
        assert!(is_dolil_run(&p));
        let eps = parse_dolil_run(&p).unwrap();
        assert_eq!(eps.len(), 1);
        let e = &eps[0];
        assert_eq!(e.episode_id, "dolil:v3-1");
        assert_eq!(e.app_id, "dolil");
        assert_eq!(e.user_prompt, "List the crates directory.");
        assert_eq!(e.assistant_response, "The crates directory contains bidda.");
        assert_eq!(e.tool_calls.len(), 1);
        assert_eq!(e.tool_calls[0].name, "file.list");
        assert_eq!(e.tool_calls[0].arguments["path"], "crates");
        let tr = &e.tool_results.as_ref().unwrap()[0];
        assert_eq!(tr["receipts_succeeded"], 1);
        assert_eq!(tr["receipts_failed"], 1);
        assert_eq!(e.outcome, EpisodeOutcome::Unverified);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn model_error_without_answer_is_a_failed_run_and_claude_jsonl_is_not_dolil() {
        let dir = std::env::temp_dir().join(format!("ojon-dolil2-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = write_run(&dir, "v3-2.jsonl", &[
            r#"{"event": "started", "goal": "Do a thing.", "tools_exposed": 4}"#,
            r#"{"error": "local model HTTP 503: no eligible route", "event": "model_error", "turn": 0}"#,
        ]);
        let eps = parse_dolil_run(&p).unwrap();
        assert_eq!(eps.len(), 1);
        assert_eq!(eps[0].outcome, EpisodeOutcome::FailedRetry);
        assert!(eps[0].tool_results.as_ref().unwrap()[0]["model_errors"].as_array().unwrap().len() == 1);
        let claude = write_run(&dir, "c.jsonl", &[r#"{"type":"user","sessionId":"S","message":{"role":"user","content":"hi"}}"#]);
        assert!(!is_dolil_run(&claude));
        let empty = write_run(&dir, "e.jsonl", &[r#"{"event": "request_budget"}"#]);
        assert!(parse_dolil_run(&empty).unwrap().is_empty());
        let _ = fs::remove_dir_all(&dir);
    }
}
