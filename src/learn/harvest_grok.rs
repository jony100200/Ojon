//! Harvest Grok Build `chat_history.jsonl` sessions.
//!
//! One episode is one kept user turn plus the assistant turns that follow it.
//! Kept turns are human messages and `compaction_meta` summaries. Reasoning
//! rows, `encrypted_content`, system reminders, and project-instruction rows
//! are skipped. Tool results are counted, not copied. `record_episode` redacts
//! secrets after this parser returns.

use serde_json::Value;
use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::time::UNIX_EPOCH;

use super::collector::{CollectedEpisode, EpisodeOutcome, EpisodeProvenance, ToolCallRecord};
use super::harvest_claude::execution_signal;
use super::knowledge::sha256_hex;

const MAX_PROMPT_BYTES: usize = 32 * 1024;
const MAX_RESPONSE_BYTES: usize = 96 * 1024;
const MAX_TOOL_ARGS_BYTES: usize = 8 * 1024;
const MAX_SIGNAL_BYTES: usize = 48 * 1024;

fn truncate(text: &str, max: usize) -> String {
    if text.len() <= max {
        return text.to_string();
    }
    let mut cut = max;
    while cut > 0 && !text.is_char_boundary(cut) {
        cut -= 1;
    }
    format!(
        "{}\n[... truncated {} bytes]",
        &text[..cut],
        text.len() - cut
    )
}

fn text_of(content: &Value) -> String {
    match content {
        Value::String(text) => text.trim().to_string(),
        Value::Array(blocks) => blocks
            .iter()
            .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn capped_args(raw: &Value) -> Value {
    let text = match raw {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).unwrap_or_default(),
    };
    if text.len() > MAX_TOOL_ARGS_BYTES {
        return serde_json::json!({ "truncated": true, "bytes": text.len() });
    }
    serde_json::from_str(&text).unwrap_or(Value::String(text))
}

fn signal_window(text: &str) -> String {
    if text.len() <= MAX_SIGNAL_BYTES {
        return text.to_ascii_lowercase();
    }
    let mut start = text.len() - MAX_SIGNAL_BYTES;
    while start < text.len() && !text.is_char_boundary(start) {
        start += 1;
    }
    text[start..].to_ascii_lowercase()
}

fn file_mtime_s(path: &Path) -> u64 {
    fs::metadata(path)
        .ok()
        .and_then(|meta| meta.modified().ok())
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|elapsed| elapsed.as_secs())
        .unwrap_or(0)
}

/// True when the file is a Grok `chat_history.jsonl` transcript.
pub fn is_grok_chat(path: &Path) -> bool {
    if path.file_name().and_then(|name| name.to_str()) != Some("chat_history.jsonl") {
        return false;
    }
    let Ok(file) = fs::File::open(path) else {
        return false;
    };
    for (index, line) in BufReader::new(file).lines().enumerate() {
        if index >= 20 {
            break;
        }
        let Ok(line) = line else {
            return false;
        };
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if matches!(
            value.get("type").and_then(Value::as_str),
            Some("system" | "user" | "assistant" | "reasoning" | "tool_result")
        ) {
            return true;
        }
    }
    false
}

fn kept_user(value: &Value) -> bool {
    if value.get("type").and_then(Value::as_str) != Some("user") {
        return false;
    }
    match value.get("synthetic_reason").and_then(Value::as_str) {
        None => true,
        Some(reason) if reason.trim().is_empty() || reason == "compaction_meta" => true,
        Some(_) => false,
    }
}

struct Draft {
    episode_id: String,
    session_id: String,
    model: String,
    timestamp_epoch_s: u64,
    user_prompt: String,
    assistant_text: Vec<String>,
    tool_calls: Vec<ToolCallRecord>,
    tool_results: usize,
    signal: Option<bool>,
    source_line_start: u64,
    source_line_end: u64,
}

impl Draft {
    fn finish(self) -> Option<CollectedEpisode> {
        let response = self.assistant_text.join("\n").trim().to_string();
        if response.is_empty() && self.tool_calls.is_empty() {
            return None;
        }
        let mut episode = CollectedEpisode::new(
            self.episode_id,
            self.session_id.clone(),
            format!("grok:{}", self.session_id),
            "grok",
            truncate(&self.user_prompt, MAX_PROMPT_BYTES),
            truncate(&response, MAX_RESPONSE_BYTES),
        );
        if !self.model.is_empty() {
            episode.model_id = self.model.clone();
            episode.claimed_teacher = self.model;
        }
        if self.timestamp_epoch_s > 0 {
            episode.timestamp_epoch_s = self.timestamp_epoch_s;
        }
        episode.tool_calls = self.tool_calls;
        episode.tool_results = (self.tool_results > 0)
            .then(|| vec![serde_json::json!({ "tool_result_count": self.tool_results })]);
        episode.outcome = match self.signal {
            Some(true) => EpisodeOutcome::VerifiedSuccess,
            Some(false) => EpisodeOutcome::FailedRetry,
            None => EpisodeOutcome::Unverified,
        };
        episode.unit_tests_passed = self.signal;
        episode.scalar_reward = episode.outcome.default_reward();
        episode.provenance = Some(EpisodeProvenance {
            source_format: "grok.chat_history.v1".to_string(),
            source_session_id: self.session_id.clone(),
            source_file: "chat_history.jsonl".to_string(),
            source_record_sha256: sha256_hex(
                format!(
                    "{}:{}:{}",
                    self.session_id, self.source_line_start, self.source_line_end
                )
                .as_bytes(),
            ),
            source_line_start: self.source_line_start,
            source_line_end: self.source_line_end,
            source_event_ids: Vec::new(),
            workspace_id: None,
            repository_id: None,
            task_family_id: "session".to_string(),
        });
        Some(episode)
    }
}

/// Parse one Grok `chat_history.jsonl`. Reasoning text is not copied.
pub fn parse_grok_chat(path: &Path) -> io::Result<Vec<CollectedEpisode>> {
    let file = fs::File::open(path)?;
    let session_id = path
        .parent()
        .and_then(|parent| parent.file_name())
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| !name.is_empty())
        .unwrap_or_else(|| "grok-session".to_string());
    let stamp = file_mtime_s(path);
    let mut episodes = Vec::new();
    let mut draft: Option<Draft> = None;

    for (index, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if value.get("encrypted_content").is_some() {
            continue;
        }
        let line_no = (index + 1) as u64;
        match value.get("type").and_then(Value::as_str) {
            Some("user") if kept_user(&value) => {
                let text = text_of(value.get("content").unwrap_or(&Value::Null));
                if text.is_empty() {
                    continue;
                }
                if let Some(finished) = draft.take().and_then(Draft::finish) {
                    episodes.push(finished);
                }
                draft = Some(Draft {
                    episode_id: format!("{session_id}:{line_no}"),
                    session_id: session_id.clone(),
                    model: String::new(),
                    timestamp_epoch_s: stamp,
                    user_prompt: text,
                    assistant_text: Vec::new(),
                    tool_calls: Vec::new(),
                    tool_results: 0,
                    signal: None,
                    source_line_start: line_no,
                    source_line_end: line_no,
                });
            }
            Some("assistant") => {
                let Some(current) = draft.as_mut() else {
                    continue;
                };
                current.source_line_end = line_no;
                if current.model.is_empty() {
                    if let Some(model) = value.get("model_id").and_then(Value::as_str) {
                        current.model = model.to_string();
                    }
                }
                let text = text_of(value.get("content").unwrap_or(&Value::Null));
                if !text.is_empty() {
                    current.assistant_text.push(text);
                }
                if let Some(calls) = value.get("tool_calls").and_then(Value::as_array) {
                    for call in calls {
                        current.tool_calls.push(ToolCallRecord {
                            id: call
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string(),
                            name: call
                                .get("name")
                                .and_then(Value::as_str)
                                .unwrap_or("unknown")
                                .to_string(),
                            arguments: capped_args(call.get("arguments").unwrap_or(&Value::Null)),
                        });
                    }
                }
            }
            Some("tool_result") => {
                let Some(current) = draft.as_mut() else {
                    continue;
                };
                current.source_line_end = line_no;
                current.tool_results += 1;
                let text = text_of(value.get("content").unwrap_or(&Value::Null));
                if let Some(signal) = execution_signal(&signal_window(&text)) {
                    current.signal = Some(signal);
                }
            }
            _ => {}
        }
    }
    if let Some(finished) = draft.take().and_then(Draft::finish) {
        episodes.push(finished);
    }
    Ok(episodes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(dir: &Path) -> std::path::PathBuf {
        let session = dir.join("01a194c5-session");
        fs::create_dir_all(&session).unwrap();
        let path = session.join("chat_history.jsonl");
        let rows = [
            r#"{"type":"system","content":"system prompt sk-testsecretvalue000000"}"#,
            r#"{"type":"user","synthetic_reason":"compaction_meta","content":[{"type":"text","text":"earlier session summary"}]}"#,
            r#"{"type":"assistant","model_id":"grok-test","content":"Continued from the summary.","tool_calls":[]}"#,
            r#"{"type":"user","synthetic_reason":"system_reminder","content":[{"type":"text","text":"do not store this reminder"}]}"#,
            r#"{"type":"user","content":[{"type":"text","text":"record this session"}]}"#,
            r#"{"type":"reasoning","id":"r1","status":"done","summary":"hidden-chain-of-thought","encrypted_content":"CIPHERTEXT-NOT-STORED"}"#,
            r#"{"type":"assistant","model_id":"grok-test","content":"I updated the watcher.","tool_calls":[{"id":"c1","name":"read_file","arguments":{"target_file":"src/lib.rs"}}]}"#,
            r#"{"type":"tool_result","tool_call_id":"c1","content":"test result: ok. 2 passed; 0 failed"}"#,
            r#"{"type":"assistant","model_id":"grok-test","content":"The tests passed.","tool_calls":[]}"#,
        ];
        fs::write(&path, rows.join("\n") + "\n").unwrap();
        path
    }

    #[test]
    fn skips_reasoning_synthetic_turns_and_tool_dumps() {
        let dir = std::env::temp_dir().join(format!("ojon-grok-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = fixture(&dir);
        assert!(is_grok_chat(&path));
        let episodes = parse_grok_chat(&path).unwrap();
        assert_eq!(episodes.len(), 2);
        assert_eq!(episodes[0].user_prompt, "earlier session summary");
        assert!(episodes[0]
            .assistant_response
            .contains("Continued from the summary."));
        let episode = &episodes[1];
        assert_eq!(episode.session_id, "01a194c5-session");
        assert_eq!(episode.app_id, "grok:01a194c5-session");
        assert_eq!(episode.user_prompt, "record this session");
        assert!(episode
            .assistant_response
            .contains("I updated the watcher."));
        assert!(episode.assistant_response.contains("The tests passed."));
        assert_eq!(episode.model_id, "grok-test");
        assert_eq!(episode.tool_calls.len(), 1);
        assert_eq!(episode.tool_calls[0].name, "read_file");
        assert_eq!(
            episode.tool_results.as_ref().unwrap()[0]["tool_result_count"],
            1
        );
        assert_eq!(episode.outcome, EpisodeOutcome::VerifiedSuccess);
        let rendered = serde_json::to_string(&episodes).unwrap();
        assert!(!rendered.contains("CIPHERTEXT"));
        assert!(!rendered.contains("hidden-chain-of-thought"));
        assert!(!rendered.contains("do not store this reminder"));
        assert!(rendered.contains("earlier session summary"));
        assert!(!rendered.contains("sk-testsecretvalue"));
        assert!(!rendered.contains("encrypted_content"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_other_jsonl_names() {
        let dir = std::env::temp_dir().join(format!("ojon-grok-name-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("updates.jsonl");
        fs::write(&path, "{\"type\":\"assistant\",\"content\":\"nope\"}\n").unwrap();
        assert!(!is_grok_chat(&path));
        let _ = fs::remove_dir_all(&dir);
    }
}
