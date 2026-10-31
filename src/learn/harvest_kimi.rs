//! Harvest Kimi Code `wire.jsonl` sessions without retaining reasoning traces.
use serde_json::Value;
use std::collections::HashSet;
use std::fs;
use std::io::{self, BufRead, BufReader};
use std::path::Path;

use super::collector::{CollectedEpisode, EpisodeOutcome, EpisodeProvenance, ToolCallRecord};
use super::harvest_claude::execution_signal;
use super::knowledge::sha256_hex;

const MAX_PROMPT_BYTES: usize = 32 * 1024;
const MAX_RESPONSE_BYTES: usize = 96 * 1024;
const MAX_TOOL_BYTES: usize = 16 * 1024;
const MAX_EVENT_IDS: usize = 256;

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

fn text_blocks(content: &Value) -> String {
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

pub fn is_kimi_wire(path: &Path) -> bool {
    let Ok(file) = fs::File::open(path) else {
        return false;
    };
    let mut metadata = false;
    for (index, line) in BufReader::new(file).lines().enumerate() {
        if index >= 40 {
            break;
        }
        let Ok(line) = line else {
            return false;
        };
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        if value.get("type").and_then(Value::as_str) == Some("metadata")
            && value.get("protocol_version").is_some()
        {
            metadata = true;
        }
        if metadata
            && matches!(
                value.get("type").and_then(Value::as_str),
                Some("agent.message.appended" | "context.append_message")
            )
        {
            return true;
        }
    }
    false
}

struct Draft {
    episode_id: String,
    session_id: String,
    task_family_id: String,
    timestamp_epoch_s: u64,
    user_prompt: String,
    assistant_text: Vec<String>,
    tool_calls: Vec<ToolCallRecord>,
    tool_results: Vec<Value>,
    signal: Option<bool>,
    repository_id: Option<String>,
    source_line_start: u64,
    source_line_end: u64,
    source_event_ids: Vec<String>,
    event_hashes: Vec<String>,
}

impl Draft {
    fn record_event(&mut self, line: &str, line_no: u64, event_id: Option<String>) {
        self.source_line_end = line_no;
        self.event_hashes.push(sha256_hex(line.as_bytes()));
        if self.source_event_ids.len() < MAX_EVENT_IDS {
            if let Some(id) = event_id.filter(|id| !id.is_empty()) {
                self.source_event_ids.push(id);
            }
        }
    }

    fn finish(self, context: &SessionContext) -> Option<CollectedEpisode> {
        let response = self.assistant_text.join("\n").trim().to_string();
        if response.is_empty() && self.tool_calls.is_empty() {
            return None;
        }
        let mut episode = CollectedEpisode::new(
            self.episode_id,
            self.session_id,
            format!(
                "kimi-code:{}",
                self.repository_id.as_deref().unwrap_or("workspace")
            ),
            &context.model_id,
            truncate(&self.user_prompt, MAX_PROMPT_BYTES),
            truncate(&response, MAX_RESPONSE_BYTES),
        );
        episode.model_id = context.model_id.clone();
        episode.timestamp_epoch_s = self.timestamp_epoch_s;
        episode.tool_calls = self.tool_calls;
        episode.tool_results = (!self.tool_results.is_empty()).then_some(self.tool_results);
        episode.outcome = match self.signal {
            Some(true) => EpisodeOutcome::VerifiedSuccess,
            Some(false) => EpisodeOutcome::FailedRetry,
            None => EpisodeOutcome::Unverified,
        };
        episode.unit_tests_passed = self.signal;
        episode.scalar_reward = episode.outcome.default_reward();
        episode.provenance = Some(EpisodeProvenance {
            source_format: context.source_format.clone(),
            source_session_id: context.source_session_id.clone(),
            source_file: context.source_file.clone(),
            source_record_sha256: sha256_hex(self.event_hashes.join("").as_bytes()),
            source_line_start: self.source_line_start,
            source_line_end: self.source_line_end,
            source_event_ids: self.source_event_ids,
            workspace_id: context.workspace_id.clone(),
            repository_id: self.repository_id,
            task_family_id: self.task_family_id,
        });
        Some(episode)
    }
}

struct SessionContext {
    source_format: String,
    source_session_id: String,
    source_file: String,
    workspace_id: Option<String>,
    model_id: String,
}

fn session_context(path: &Path) -> SessionContext {
    let source_session_id = path
        .components()
        .map(|part| part.as_os_str().to_string_lossy())
        .find(|part| part.starts_with("session_"))
        .unwrap_or_else(|| "session-unknown".into());
    SessionContext {
        source_format: "kimi.wire.v1".into(),
        source_session_id: source_session_id.into_owned(),
        source_file: path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .into_owned(),
        workspace_id: None,
        model_id: "kimi-code".into(),
    }
}

fn user_identity(value: &Value, line_no: u64) -> String {
    value
        .pointer("/message/id")
        .or_else(|| value.pointer("/message/meta/userMessageId"))
        .or_else(|| value.pointer("/message/meta/promptId"))
        .or_else(|| value.pointer("/message/origin/promptId"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| format!("line-{line_no}"))
}

fn tool_arguments(call: &Value) -> Value {
    let raw = call.get("arguments").cloned().unwrap_or(Value::Null);
    let parsed = match raw {
        Value::String(text) => serde_json::from_str(&text).unwrap_or(Value::String(text)),
        value => value,
    };
    if serde_json::to_string(&parsed).map_or(0, |text| text.len()) > MAX_TOOL_BYTES {
        serde_json::json!({
            "truncated": true,
            "keys": parsed.as_object().map(|object| object.keys().collect::<Vec<_>>()).unwrap_or_default()
        })
    } else {
        parsed
    }
}

fn note_repository(draft: &mut Draft, arguments: &Value) {
    if draft.repository_id.is_some() {
        return;
    }
    if let Some(cwd) = arguments.get("cwd").and_then(Value::as_str) {
        draft.repository_id = Path::new(cwd)
            .file_name()
            .map(|name| name.to_string_lossy().into_owned());
    }
}

pub fn parse_kimi_wire(path: &Path) -> io::Result<Vec<CollectedEpisode>> {
    let mut context = session_context(path);
    let mut episodes = Vec::new();
    let mut draft: Option<Draft> = None;
    let mut seen_users = HashSet::new();
    for (index, line) in BufReader::new(fs::File::open(path)?).lines().enumerate() {
        let line_no = index as u64 + 1;
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        match value.get("type").and_then(Value::as_str) {
            Some("metadata") => {
                if let Some(version) = value.get("protocol_version").and_then(Value::as_str) {
                    context.source_format = format!("kimi.wire.{version}");
                }
            }
            Some("runtime.set_binding") => {
                context.workspace_id = value
                    .get("workspaceId")
                    .and_then(Value::as_str)
                    .map(str::to_string);
            }
            Some("profile.bind") => {
                if let Some(model) = value.get("modelAlias").and_then(Value::as_str) {
                    context.model_id = model.to_string();
                }
            }
            Some("context.append_message")
                if value.pointer("/message/role").and_then(Value::as_str) == Some("user") =>
            {
                let prompt = text_blocks(value.pointer("/message/content").unwrap_or(&Value::Null));
                let identity = user_identity(&value, line_no);
                if prompt.is_empty() || !seen_users.insert(identity.clone()) {
                    continue;
                }
                if let Some(finished) = draft.take().and_then(|draft| draft.finish(&context)) {
                    episodes.push(finished);
                }
                draft = Some(Draft {
                    episode_id: format!("kimi:{}:{identity}", context.source_session_id),
                    session_id: context.source_session_id.clone(),
                    task_family_id: identity.clone(),
                    timestamp_epoch_s: value.get("time").and_then(Value::as_u64).unwrap_or(0)
                        / 1000,
                    user_prompt: prompt,
                    assistant_text: Vec::new(),
                    tool_calls: Vec::new(),
                    tool_results: Vec::new(),
                    signal: None,
                    repository_id: None,
                    source_line_start: line_no,
                    source_line_end: line_no,
                    source_event_ids: Vec::new(),
                    event_hashes: Vec::new(),
                });
                if let Some(current) = draft.as_mut() {
                    current.record_event(&line, line_no, Some(identity));
                }
            }
            Some("agent.message.appended") => {
                let message = value.pointer("/message/message").unwrap_or(&Value::Null);
                match message.get("role").and_then(Value::as_str) {
                    Some("assistant") => {
                        let Some(current) = draft.as_mut() else {
                            continue;
                        };
                        let text = text_blocks(message.get("content").unwrap_or(&Value::Null));
                        if !text.is_empty() {
                            current.assistant_text.push(text);
                        }
                        let mut event_ids = Vec::new();
                        if let Some(id) = value
                            .pointer("/message/meta/messageId")
                            .and_then(Value::as_str)
                        {
                            event_ids.push(id.to_string());
                        }
                        for call in message
                            .get("toolCalls")
                            .and_then(Value::as_array)
                            .into_iter()
                            .flatten()
                        {
                            let arguments = tool_arguments(call);
                            note_repository(current, &arguments);
                            let id = call
                                .get("id")
                                .and_then(Value::as_str)
                                .unwrap_or("")
                                .to_string();
                            event_ids.push(id.clone());
                            current.tool_calls.push(ToolCallRecord {
                                id,
                                name: call
                                    .get("name")
                                    .and_then(Value::as_str)
                                    .unwrap_or("unknown")
                                    .to_string(),
                                arguments,
                            });
                        }
                        current.record_event(&line, line_no, event_ids.first().cloned());
                    }
                    Some("tool") => {
                        let Some(current) = draft.as_mut() else {
                            continue;
                        };
                        let text = text_blocks(message.get("content").unwrap_or(&Value::Null));
                        if let Some(signal) = execution_signal(&text.to_ascii_lowercase()) {
                            current.signal = Some(signal);
                        }
                        let id = message
                            .get("toolCallId")
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string();
                        current.tool_results.push(serde_json::json!({
                            "tool_call_id": id,
                            "content": truncate(&text, MAX_TOOL_BYTES)
                        }));
                        current.record_event(&line, line_no, Some(id));
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }
    if let Some(finished) = draft.take().and_then(|draft| draft.finish(&context)) {
        episodes.push(finished);
    }
    Ok(episodes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture(dir: &Path) -> std::path::PathBuf {
        let path = dir.join("session_test/agents/main/wire.jsonl");
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        let rows = [
            r#"{"type":"metadata","protocol_version":"1.5"}"#,
            r#"{"type":"runtime.set_binding","workspaceId":"wd_test"}"#,
            r#"{"type":"profile.bind","modelAlias":"kimi-code/test-model"}"#,
            r#"{"type":"context.append_message","time":1793452960279,"message":{"id":"msg-one","role":"user","content":[{"type":"text","text":"run the tests"}]}}"#,
            r#"{"type":"agent.message.appended","message":{"message":{"role":"assistant","content":[{"type":"think","think":"private reasoning"},{"type":"text","text":"Running them."}],"toolCalls":[{"type":"function","id":"tool-one","name":"Bash","arguments":"{\"command\":\"cargo test\",\"cwd\":\"D:\\\\work\\\\Bunon\"}"}]},"meta":{"messageId":"assistant-one"}}}"#,
            r#"{"type":"agent.message.appended","message":{"message":{"role":"tool","toolCallId":"tool-one","content":[{"type":"text","text":"test result: ok. 3 passed; 0 failed"}]}}}"#,
            r#"{"type":"agent.message.appended","message":{"message":{"role":"assistant","content":[{"type":"text","text":"All three tests pass."}]}}}"#,
        ];
        fs::write(&path, rows.join("\n") + "\n").unwrap();
        path
    }

    #[test]
    fn parses_tool_sessions_with_complete_source_provenance() {
        let dir = std::env::temp_dir().join(format!("ojon-kimi-{}", std::process::id()));
        let path = fixture(&dir);
        assert!(is_kimi_wire(&path));
        let episodes = parse_kimi_wire(&path).unwrap();
        assert_eq!(episodes.len(), 1);
        let episode = &episodes[0];
        assert_eq!(episode.outcome, EpisodeOutcome::VerifiedSuccess);
        assert_eq!(episode.model_id, "kimi-code/test-model");
        assert_eq!(episode.tool_calls[0].name, "Bash");
        assert!(!episode.assistant_response.contains("private reasoning"));
        let provenance = episode.provenance.as_ref().unwrap();
        assert_eq!(provenance.source_format, "kimi.wire.1.5");
        assert_eq!(provenance.source_session_id, "session_test");
        assert_eq!(provenance.source_line_start, 4);
        assert_eq!(provenance.source_line_end, 7);
        assert_eq!(provenance.source_event_ids.len(), 3);
        assert_eq!(provenance.workspace_id.as_deref(), Some("wd_test"));
        assert_eq!(provenance.repository_id.as_deref(), Some("Bunon"));
        assert_eq!(provenance.source_record_sha256.len(), 64);
        let _ = fs::remove_dir_all(dir);
    }
}
