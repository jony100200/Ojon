//! Harvest Claude Code session transcripts into Ojon Learn episodes.
//!
//! Claude Code writes one JSONL file per session under
//! `~/.claude/projects/<project>/<session>.jsonl`. The lines that matter are
//! `type == "user"` and `type == "assistant"`, each carrying `message.content`
//! as either a string or a list of blocks (`text`, `tool_use`, `tool_result`).
//! Everything else (`summary`, `api-request*`, `file-history-snapshot`,
//! `permission-mode`, ...) is bookkeeping and is skipped.
//!
//! An episode is one user turn plus every assistant turn until the next user
//! turn that carries human text. Tool calls become `ToolCallRecord`s; tool
//! results are counted, not copied (they are mostly file dumps). Outcome is
//! `Unverified` on purpose: a transcript proves what was said, not that the
//! work was correct; curation and the task suite decide that later. The
//! harvester's `sanitize()` redacts secrets after this parser returns.
use serde_json::Value;
use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use super::collector::{CollectedEpisode, EpisodeOutcome, ToolCallRecord};

const MAX_PROMPT_BYTES: usize = 32 * 1024;
const MAX_RESPONSE_BYTES: usize = 96 * 1024;
const MAX_TOOL_ARGS_BYTES: usize = 8 * 1024;

/// Sniff the first few lines: Claude Code transcripts have `type` + `sessionId`
/// on nearly every line and `message` on the conversational ones.
pub fn is_claude_transcript(path: &Path) -> bool {
    let Ok(file) = fs::File::open(path) else { return false };
    let reader = BufReader::new(file);
    let mut seen_session = false;
    for (i, line) in reader.lines().enumerate() {
        if i >= 40 {
            break;
        }
        let Ok(line) = line else { return false };
        let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
        if v.get("sessionId").and_then(Value::as_str).is_some() {
            seen_session = true;
        }
        let t = v.get("type").and_then(Value::as_str).unwrap_or("");
        if (t == "user" || t == "assistant") && v.get("message").is_some() && seen_session {
            return true;
        }
    }
    false
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut cut = max;
    while cut > 0 && !s.is_char_boundary(cut) {
        cut -= 1;
    }
    format!("{}\n[... truncated {} bytes]", &s[..cut], s.len() - cut)
}

/// Text blocks of a user message. Tool results come back as `user` lines too;
/// those carry no human text and must not start a new episode.
fn user_text(content: &Value) -> Option<String> {
    match content {
        Value::String(s) => {
            let t = s.trim();
            (!t.is_empty()).then(|| t.to_string())
        }
        Value::Array(blocks) => {
            let text: Vec<&str> = blocks
                .iter()
                .filter(|b| b.get("type").and_then(Value::as_str) == Some("text"))
                .filter_map(|b| b.get("text").and_then(Value::as_str))
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .collect();
            (!text.is_empty()).then(|| text.join("\n"))
        }
        _ => None,
    }
}

fn count_tool_results(content: &Value) -> usize {
    content
        .as_array()
        .map(|blocks| blocks.iter().filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_result")).count())
        .unwrap_or(0)
}

/// Text inside tool_result blocks (string or nested text blocks), lower-cased.
fn tool_result_text(content: &Value) -> String {
    let mut out = String::new();
    if let Some(blocks) = content.as_array() {
        for b in blocks.iter().filter(|b| b.get("type").and_then(Value::as_str) == Some("tool_result")) {
            match b.get("content") {
                Some(Value::String(s)) => out.push_str(s),
                Some(Value::Array(inner)) => {
                    for t in inner {
                        if let Some(s) = t.get("text").and_then(Value::as_str) {
                            out.push_str(s);
                            out.push('\n');
                        }
                    }
                }
                _ => {}
            }
            out.push('\n');
        }
    }
    out.to_ascii_lowercase()
}

/// Execution signal from a tool result: Some(true) for an explicit pass line,
/// Some(false) for an explicit failure, None for neither. These are the markers
/// of the tools the agents actually run (cargo, pytest, cmake/MSBuild, Python,
/// Rust panics). Only explicit markers count; prose is never a signal.
pub fn execution_signal(text: &str) -> Option<bool> {
    const PASS: [&str; 6] = ["test result: ok.", " passed in ", "build_exit=0", "exit=0", "all tests passed", "finished `dev` profile"];
    const FAIL: [&str; 9] = ["test result: failed", " failed in ", "error[e", "panicked at", "traceback (most recent call last)", "build_exit=1", "assertionerror", "exit=1", "error: could not compile"];
    let pass = PASS.iter().filter_map(|m| text.rfind(m)).max();
    let fail = FAIL.iter().filter_map(|m| text.rfind(m)).max();
    match (pass, fail) {
        (None, None) => None,
        (Some(_), None) => Some(true),
        (None, Some(_)) => Some(false),
        (Some(p), Some(f)) => Some(p > f), // the LAST signal in the result wins
    }
}

struct Draft {
    episode_id: String,
    session_id: String,
    app_id: String,
    model: String,
    timestamp_epoch_s: u64,
    user_prompt: String,
    assistant_text: Vec<String>,
    tool_calls: Vec<ToolCallRecord>,
    tool_results: usize,
    /// last execution signal seen in this episode's tool results
    signal: Option<bool>,
}

impl Draft {
    fn finish(self) -> Option<CollectedEpisode> {
        let response = self.assistant_text.join("\n").trim().to_string();
        if response.is_empty() && self.tool_calls.is_empty() {
            return None;
        }
        let mut ep = CollectedEpisode::new(
            self.episode_id,
            self.session_id,
            self.app_id,
            "claude-code",
            truncate(&self.user_prompt, MAX_PROMPT_BYTES),
            truncate(&response, MAX_RESPONSE_BYTES),
        );
        ep.model_id = self.model;
        if self.timestamp_epoch_s > 0 {
            ep.timestamp_epoch_s = self.timestamp_epoch_s;
        }
        ep.tool_calls = self.tool_calls;
        ep.tool_results = (self.tool_results > 0).then(|| vec![serde_json::json!({"tool_result_count": self.tool_results})]);
        // Outcome from the last explicit execution signal in the episode's own tool
        // results; a transcript with no such signal stays Unverified.
        ep.outcome = match self.signal {
            Some(true) => EpisodeOutcome::VerifiedSuccess,
            Some(false) => EpisodeOutcome::FailedRetry,
            None => EpisodeOutcome::Unverified,
        };
        ep.unit_tests_passed = self.signal;
        ep.scalar_reward = ep.outcome.default_reward();
        Some(ep)
    }
}

fn epoch_seconds(ts: Option<&str>) -> u64 {
    // RFC 3339 like 2026-10-30T01:23:45.678Z; a plain date arithmetic avoids a dependency.
    let Some(ts) = ts else { return 0 };
    let b = ts.as_bytes();
    if b.len() < 19 {
        return 0;
    }
    let num = |s: &str| s.parse::<i64>().unwrap_or(0);
    let (y, mo, d) = (num(&ts[0..4]), num(&ts[5..7]), num(&ts[8..10]));
    let (h, mi, s) = (num(&ts[11..13]), num(&ts[14..16]), num(&ts[17..19]));
    if y < 1970 || !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
        return 0;
    }
    // days from civil (Howard Hinnant)
    let yy = if mo <= 2 { y - 1 } else { y };
    let era = yy.div_euclid(400);
    let yoe = yy - era * 400;
    let mp = (mo + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    (days * 86400 + h * 3600 + mi * 60 + s).max(0) as u64
}

pub fn parse_claude_transcript(path: &Path) -> io::Result<Vec<CollectedEpisode>> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let project = path
        .parent()
        .and_then(|p| p.file_name())
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown-project".to_string());
    let mut episodes = Vec::new();
    let mut draft: Option<Draft> = None;
    let mut turn = 0usize;
    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(&line) else { continue };
        let t = v.get("type").and_then(Value::as_str).unwrap_or("");
        if t != "user" && t != "assistant" {
            continue;
        }
        let Some(message) = v.get("message") else { continue };
        let content = message.get("content").unwrap_or(&Value::Null);
        let session_id = v.get("sessionId").and_then(Value::as_str).unwrap_or("no-session").to_string();
        if t == "user" {
            if v.get("isMeta").and_then(Value::as_bool).unwrap_or(false) {
                continue;
            }
            match user_text(content) {
                Some(text) => {
                    if let Some(d) = draft.take() {
                        episodes.extend(d.finish());
                    }
                    turn += 1;
                    let uuid = v.get("uuid").and_then(Value::as_str).map(str::to_string).unwrap_or_else(|| format!("turn-{turn}"));
                    let cwd = v.get("cwd").and_then(Value::as_str).unwrap_or("");
                    let app = Path::new(cwd).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| project.clone());
                    draft = Some(Draft {
                        episode_id: format!("{session_id}:{uuid}"),
                        session_id,
                        app_id: format!("claude-code:{app}"),
                        model: String::new(),
                        timestamp_epoch_s: epoch_seconds(v.get("timestamp").and_then(Value::as_str)),
                        user_prompt: text,
                        assistant_text: Vec::new(),
                        tool_calls: Vec::new(),
                        tool_results: 0,
                        signal: None,
                    });
                }
                None => {
                    if let Some(d) = draft.as_mut() {
                        d.tool_results += count_tool_results(content);
                        if let Some(sig) = execution_signal(&tool_result_text(content)) {
                            d.signal = Some(sig);
                        }
                    }
                }
            }
            continue;
        }
        // assistant
        let Some(d) = draft.as_mut() else { continue };
        if let Some(model) = message.get("model").and_then(Value::as_str) {
            if d.model.is_empty() {
                d.model = model.to_string();
            }
        }
        match content {
            Value::String(s) => {
                if !s.trim().is_empty() {
                    d.assistant_text.push(s.trim().to_string());
                }
            }
            Value::Array(blocks) => {
                for b in blocks {
                    match b.get("type").and_then(Value::as_str) {
                        Some("text") => {
                            if let Some(s) = b.get("text").and_then(Value::as_str) {
                                if !s.trim().is_empty() {
                                    d.assistant_text.push(s.trim().to_string());
                                }
                            }
                        }
                        Some("tool_use") => {
                            let args = b.get("input").cloned().unwrap_or(Value::Null);
                            let args = if serde_json::to_string(&args).map(|s| s.len()).unwrap_or(0) > MAX_TOOL_ARGS_BYTES {
                                serde_json::json!({"truncated": true, "keys": args.as_object().map(|o| o.keys().cloned().collect::<Vec<_>>()).unwrap_or_default()})
                            } else {
                                args
                            };
                            d.tool_calls.push(ToolCallRecord {
                                id: b.get("id").and_then(Value::as_str).unwrap_or("").to_string(),
                                name: b.get("name").and_then(Value::as_str).unwrap_or("unknown").to_string(),
                                arguments: args,
                            });
                        }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    if let Some(d) = draft.take() {
        episodes.extend(d.finish());
    }
    Ok(episodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_transcript(dir: &Path) -> std::path::PathBuf {
        let p = dir.join("sess.jsonl");
        let mut f = fs::File::create(&p).unwrap();
        let lines = [
            r#"{"type":"summary","summary":"x","sessionId":"S1"}"#,
            r#"{"type":"user","uuid":"u1","sessionId":"S1","cwd":"D:\\KSAppDev\\Ojon","timestamp":"2026-10-30T01:02:03.000Z","message":{"role":"user","content":"rename helper_17 in main.rs"}}"#,
            r#"{"type":"assistant","sessionId":"S1","message":{"role":"assistant","model":"claude-x","content":[{"type":"text","text":"Renaming it now."},{"type":"tool_use","id":"t1","name":"Edit","input":{"file_path":"main.rs","old_string":"helper_17","new_string":"renamed_helper_17"}}]}}"#,
            r#"{"type":"user","sessionId":"S1","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1","content":"ok"}]}}"#,
            r#"{"type":"assistant","sessionId":"S1","message":{"role":"assistant","model":"claude-x","content":[{"type":"text","text":"Done: helper_17 is now renamed_helper_17."}]}}"#,
            r#"{"type":"user","uuid":"u2","isMeta":true,"sessionId":"S1","message":{"role":"user","content":"<local-command-caveat>ignored</local-command-caveat>"}}"#,
            r#"{"type":"user","uuid":"u3","sessionId":"S1","cwd":"D:\\KSAppDev\\Ojon","message":{"role":"user","content":[{"type":"text","text":"now run the tests"}]}}"#,
            r#"{"type":"assistant","sessionId":"S1","message":{"role":"assistant","content":"All 5 tests pass."}}"#,
            r#"{"type":"api-request-blob","sessionId":"S1","message":"junk"}"#,
        ];
        for l in lines {
            writeln!(f, "{l}").unwrap();
        }
        p
    }

    #[test]
    fn sniffs_claude_transcripts_and_rejects_other_jsonl() {
        let dir = std::env::temp_dir().join(format!("ojon-claude-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = write_transcript(&dir);
        assert!(is_claude_transcript(&p));
        let other = dir.join("other.jsonl");
        fs::write(&other, "{\"user_prompt\":\"a\",\"assistant_response\":\"b\"}\n").unwrap();
        assert!(!is_claude_transcript(&other));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn pairs_user_turns_with_assistant_text_and_tool_calls() {
        let dir = std::env::temp_dir().join(format!("ojon-claude2-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = write_transcript(&dir);
        let eps = parse_claude_transcript(&p).unwrap();
        assert_eq!(eps.len(), 2, "{eps:?}");
        let e = &eps[0];
        assert_eq!(e.episode_id, "S1:u1");
        assert_eq!(e.session_id, "S1");
        assert_eq!(e.app_id, "claude-code:Ojon");
        assert_eq!(e.model_id, "claude-x");
        assert_eq!(e.user_prompt, "rename helper_17 in main.rs");
        assert!(e.assistant_response.starts_with("Renaming it now.") && e.assistant_response.contains("renamed_helper_17"));
        assert_eq!(e.tool_calls.len(), 1);
        assert_eq!(e.tool_calls[0].name, "Edit");
        assert_eq!(e.tool_results.as_ref().unwrap()[0]["tool_result_count"], 1);
        assert_eq!(e.outcome, EpisodeOutcome::Unverified); // "ok" alone is not an execution marker
        assert_eq!(e.unit_tests_passed, None);
        assert_eq!(e.timestamp_epoch_s, 1793322123); // 2026-10-30T01:02:03Z
        let e2 = &eps[1];
        assert_eq!(e2.episode_id, "S1:u3");
        assert_eq!(e2.assistant_response, "All 5 tests pass.");
        assert!(e2.tool_calls.is_empty());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn execution_signals_use_the_last_explicit_marker() {
        assert_eq!(execution_signal("running 3 tests\ntest result: ok. 3 passed; 0 failed"), Some(true));
        assert_eq!(execution_signal("test result: failed. 2 passed; 1 failed"), Some(false));
        assert_eq!(execution_signal("error[e0308]: mismatched types\nerror: could not compile `x`"), Some(false));
        assert_eq!(execution_signal("test result: failed\n...fixed...\ntest result: ok. 5 passed"), Some(true));
        assert_eq!(execution_signal("test result: ok. 5 passed\nthread 'main' panicked at src/x.rs"), Some(false));
        assert_eq!(execution_signal("file written"), None);
        assert_eq!(execution_signal(""), None);
    }

    #[test]
    fn tool_result_signals_label_the_episode() {
        let dir = std::env::temp_dir().join(format!("ojon-claude3-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let p = dir.join("s.jsonl");
        let lines = [
            r#"{"type":"user","uuid":"a","sessionId":"S","cwd":"D:\\x","message":{"role":"user","content":"fix it"}}"#,
            r#"{"type":"assistant","sessionId":"S","message":{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"Bash","input":{"command":"cargo test"}}]}}"#,
            r#"{"type":"user","sessionId":"S","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t","content":"test result: FAILED. 1 passed; 1 failed"}]}}"#,
            r#"{"type":"assistant","sessionId":"S","message":{"role":"assistant","content":[{"type":"tool_use","id":"t2","name":"Bash","input":{"command":"cargo test"}}]}}"#,
            r#"{"type":"user","sessionId":"S","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t2","content":[{"type":"text","text":"test result: ok. 2 passed; 0 failed"}]}]}}"#,
            r#"{"type":"assistant","sessionId":"S","message":{"role":"assistant","content":"Fixed and verified."}}"#,
            r#"{"type":"user","uuid":"b","sessionId":"S","cwd":"D:\\x","message":{"role":"user","content":"and this one"}}"#,
            r#"{"type":"assistant","sessionId":"S","message":{"role":"assistant","content":[{"type":"tool_use","id":"t3","name":"Bash","input":{"command":"pytest"}}]}}"#,
            r#"{"type":"user","sessionId":"S","message":{"role":"user","content":[{"type":"tool_result","tool_use_id":"t3","content":"Traceback (most recent call last):\nAssertionError"}]}}"#,
            r#"{"type":"assistant","sessionId":"S","message":{"role":"assistant","content":"It fails."}}"#,
        ];
        fs::write(&p, lines.join("\n") + "\n").unwrap();
        let eps = parse_claude_transcript(&p).unwrap();
        assert_eq!(eps.len(), 2);
        assert_eq!(eps[0].outcome, EpisodeOutcome::VerifiedSuccess);
        assert_eq!(eps[0].unit_tests_passed, Some(true));
        assert!(eps[0].scalar_reward > 0.0);
        assert_eq!(eps[1].outcome, EpisodeOutcome::FailedRetry);
        assert_eq!(eps[1].unit_tests_passed, Some(false));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncation_keeps_char_boundaries_and_notes_the_cut() {
        let s = "é".repeat(40_000);
        let t = truncate(&s, MAX_PROMPT_BYTES);
        assert!(t.len() <= MAX_PROMPT_BYTES + 64);
        assert!(t.contains("[... truncated"));
    }

    #[test]
    fn epoch_parser_matches_known_dates() {
        assert_eq!(epoch_seconds(Some("1970-01-01T00:00:00Z")), 0);
        assert_eq!(epoch_seconds(Some("2000-03-01T00:00:00.000Z")), 951868800);
        assert_eq!(epoch_seconds(Some("garbage")), 0);
        assert_eq!(epoch_seconds(None), 0);
    }
}
