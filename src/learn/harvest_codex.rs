//! Harvest Codex CLI `rollout-*.jsonl` sessions.
//!
//! Reasoning items and `encrypted_content` are skipped. Developer instruction
//! rows are skipped. Tool outputs are counted, not copied. Files larger than
//! 64 MiB are left unread so a multi-gigabyte rollout cannot fill the collector.

use serde_json::Value;
use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use super::collector::{CollectedEpisode, EpisodeOutcome, EpisodeProvenance, ToolCallRecord};
use super::harvest_claude::{epoch_seconds, execution_signal};
use super::knowledge::sha256_hex;

const MAX_PROMPT_BYTES: usize = 32 * 1024;
const MAX_RESPONSE_BYTES: usize = 96 * 1024;
const MAX_TOOL_ARGS_BYTES: usize = 8 * 1024;
const MAX_SIGNAL_BYTES: usize = 48 * 1024;
pub(crate) const MAX_SESSION_BYTES: u64 = 64 * 1024 * 1024;

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

fn message_text(content: &Value) -> String {
    match content {
        Value::String(text) => text.trim().to_string(),
        Value::Array(blocks) => blocks
            .iter()
            .filter(|block| {
                matches!(
                    block.get("type").and_then(Value::as_str),
                    Some("input_text" | "output_text" | "text")
                )
            })
            .filter_map(|block| block.get("text").and_then(Value::as_str))
            .map(str::trim)
            .filter(|text| !text.is_empty())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

fn output_text(output: &Value) -> String {
    match output {
        Value::String(text) => text.to_string(),
        Value::Array(blocks) => blocks
            .iter()
            .filter_map(|block| {
                block
                    .get("text")
                    .and_then(Value::as_str)
                    .or_else(|| block.as_str())
            })
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

fn rollout_name(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name.starts_with("rollout-") && name.to_ascii_lowercase().ends_with(".jsonl")
        })
}

/// True when the file is a Codex rollout transcript.
pub fn is_codex_rollout(path: &Path) -> bool {
    if !rollout_name(path) {
        return false;
    }
    let Ok(file) = fs::File::open(path) else {
        return false;
    };
    for (index, line) in BufReader::new(file).lines().enumerate() {
        if index >= 15 {
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
            Some("session_meta" | "response_item" | "turn_context" | "event_msg")
        ) {
            return true;
        }
    }
    false
}

struct Draft {
    episode_id: String,
    session_id: String,
    app_name: String,
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
            format!("codex:{}", self.app_name),
            "codex",
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
            source_format: "codex.rollout.v1".to_string(),
            source_session_id: self.session_id.clone(),
            source_file: "rollout.jsonl".to_string(),
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

fn note_tool(current: &mut Draft, line_no: u64, call_id: &str, name: &str, arguments: &Value) {
    current.source_line_end = line_no;
    current.tool_calls.push(ToolCallRecord {
        id: call_id.to_string(),
        name: if name.is_empty() {
            "unknown".to_string()
        } else {
            name.to_string()
        },
        arguments: capped_args(arguments),
    });
}

fn note_output(current: &mut Draft, line_no: u64, output: &Value) {
    current.source_line_end = line_no;
    current.tool_results += 1;
    if let Some(signal) = execution_signal(&signal_window(&output_text(output))) {
        current.signal = Some(signal);
    }
}

/// Parse one Codex rollout. Oversized files return no episodes.
pub fn parse_codex_rollout(path: &Path) -> io::Result<Vec<CollectedEpisode>> {
    let meta = fs::metadata(path)?;
    if meta.len() > MAX_SESSION_BYTES {
        return Ok(Vec::new());
    }
    let file = fs::File::open(path)?;
    let mut session_id = path
        .file_stem()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "codex-session".to_string());
    let mut app_name = String::from("workspace");
    let mut model = String::new();
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
        let line_no = (index + 1) as u64;
        let kind = value.get("type").and_then(Value::as_str).unwrap_or("");
        let payload = value.get("payload").cloned().unwrap_or(Value::Null);
        if payload.get("encrypted_content").is_some()
            || payload.get("type").and_then(Value::as_str) == Some("reasoning")
        {
            continue;
        }
        match kind {
            "session_meta" => {
                if let Some(id) = payload
                    .get("session_id")
                    .and_then(Value::as_str)
                    .or_else(|| payload.get("id").and_then(Value::as_str))
                {
                    session_id = id.to_string();
                }
                if let Some(cwd) = payload.get("cwd").and_then(Value::as_str) {
                    if let Some(name) = Path::new(cwd).file_name() {
                        let name = name.to_string_lossy().into_owned();
                        if !name.is_empty() {
                            app_name = name;
                        }
                    }
                }
            }
            "turn_context" => {
                if model.is_empty() {
                    if let Some(found) = payload.get("model").and_then(Value::as_str) {
                        model = found.to_string();
                    }
                }
            }
            "response_item" => {
                let item_type = payload.get("type").and_then(Value::as_str).unwrap_or("");
                match item_type {
                    "message" => {
                        let role = payload.get("role").and_then(Value::as_str).unwrap_or("");
                        let text = message_text(payload.get("content").unwrap_or(&Value::Null));
                        if role == "user" {
                            if text.is_empty() {
                                continue;
                            }
                            if let Some(finished) = draft.take().and_then(Draft::finish) {
                                episodes.push(finished);
                            }
                            let message_id =
                                payload.get("id").and_then(Value::as_str).unwrap_or("turn");
                            draft = Some(Draft {
                                episode_id: format!("{session_id}:{message_id}"),
                                session_id: session_id.clone(),
                                app_name: app_name.clone(),
                                model: model.clone(),
                                timestamp_epoch_s: epoch_seconds(
                                    value.get("timestamp").and_then(Value::as_str),
                                ),
                                user_prompt: text,
                                assistant_text: Vec::new(),
                                tool_calls: Vec::new(),
                                tool_results: 0,
                                signal: None,
                                source_line_start: line_no,
                                source_line_end: line_no,
                            });
                        } else if role == "assistant" {
                            let Some(current) = draft.as_mut() else {
                                continue;
                            };
                            current.source_line_end = line_no;
                            if !text.is_empty() {
                                current.assistant_text.push(text);
                            }
                        }
                    }
                    "function_call" | "custom_tool_call" => {
                        let Some(current) = draft.as_mut() else {
                            continue;
                        };
                        let arguments = if item_type == "function_call" {
                            payload.get("arguments").unwrap_or(&Value::Null)
                        } else {
                            payload.get("input").unwrap_or(&Value::Null)
                        };
                        note_tool(
                            current,
                            line_no,
                            payload.get("call_id").and_then(Value::as_str).unwrap_or(""),
                            payload.get("name").and_then(Value::as_str).unwrap_or(""),
                            arguments,
                        );
                    }
                    "function_call_output" | "custom_tool_call_output" => {
                        let Some(current) = draft.as_mut() else {
                            continue;
                        };
                        note_output(
                            current,
                            line_no,
                            payload.get("output").unwrap_or(&Value::Null),
                        );
                    }
                    _ => {}
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

    #[test]
    fn skips_reasoning_developer_rows_and_tool_dumps() {
        let dir = std::env::temp_dir().join(format!("ojon-codex-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("rollout-test.jsonl");
        let rows = [
            r#"{"type":"session_meta","timestamp":"2026-10-31T23:19:53.000Z","payload":{"session_id":"sess-1","cwd":"D:\\KSAppDev\\Ojon","id":"sess-1"}}"#,
            r#"{"type":"turn_context","timestamp":"2026-10-31T23:19:53.000Z","payload":{"model":"codex-test","turn_id":"t1"}}"#,
            r#"{"type":"response_item","timestamp":"2026-10-31T23:19:54.000Z","payload":{"type":"message","role":"developer","id":"dev","content":[{"type":"input_text","text":"hidden developer instructions"}]}}"#,
            r#"{"type":"response_item","timestamp":"2026-10-31T23:19:55.000Z","payload":{"type":"message","role":"user","id":"u1","content":[{"type":"input_text","text":"run the tests"}]}}"#,
            r#"{"type":"response_item","timestamp":"2026-10-31T23:19:56.000Z","payload":{"type":"reasoning","id":"r1","encrypted_content":"CIPHERTEXT-NOT-STORED","summary":[]}}"#,
            r#"{"type":"response_item","timestamp":"2026-10-31T23:19:57.000Z","payload":{"type":"function_call","id":"c1","call_id":"call-1","name":"shell","arguments":"{\"cmd\":\"cargo test\"}"}}"#,
            r#"{"type":"response_item","timestamp":"2026-10-31T23:19:58.000Z","payload":{"type":"function_call_output","call_id":"call-1","output":[{"type":"text","text":"test result: failed. 1 passed; 1 failed"}]}}"#,
            r#"{"type":"response_item","timestamp":"2026-10-31T23:19:59.000Z","payload":{"type":"message","role":"assistant","id":"a1","content":[{"type":"output_text","text":"The test failed."}]}}"#,
        ];
        fs::write(&path, rows.join("\n") + "\n").unwrap();
        assert!(is_codex_rollout(&path));
        let episodes = parse_codex_rollout(&path).unwrap();
        assert_eq!(episodes.len(), 1);
        let episode = &episodes[0];
        assert_eq!(episode.episode_id, "sess-1:u1");
        assert_eq!(episode.app_id, "codex:Ojon");
        assert_eq!(episode.user_prompt, "run the tests");
        assert_eq!(episode.assistant_response, "The test failed.");
        assert_eq!(episode.model_id, "codex-test");
        assert_eq!(episode.tool_calls.len(), 1);
        assert_eq!(episode.tool_calls[0].name, "shell");
        assert_eq!(episode.outcome, EpisodeOutcome::FailedRetry);
        assert_eq!(
            episode.timestamp_epoch_s,
            epoch_seconds(Some("2026-10-31T23:19:55.000Z"))
        );
        let rendered = serde_json::to_string(episode).unwrap();
        assert!(!rendered.contains("CIPHERTEXT"));
        assert!(!rendered.contains("hidden developer"));
        assert!(!rendered.contains("test result: failed"));
        assert!(!rendered.contains("encrypted_content"));
        let _ = fs::remove_dir_all(&dir);
    }
}
