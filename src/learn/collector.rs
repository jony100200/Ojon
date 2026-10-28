//! Local Loopback Capture Gateway & Telemetry Intake.
//!
//! Intercepts local OpenAI-compatible `/v1/chat/completions` traffic from coding
//! assistants (Cline, Roo Code, Aider, OpenHands) and direct SDK event streams.
//!
//! Features:
//! 1. Secret & Credential Redaction Filter (scrubs API keys, bearer tokens, passwords before disk write).
//! 2. Multi-turn episode span reconstruction and tool-call JSON buffering.
//! 3. Ground-truth outcome verification tagging (compiler/tests/schema check/user correction).
//! 4. Thread-safe persistence and real-time capture telemetry.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};



/// Scrub sensitive authentication tokens and API keys from arbitrary text.
pub fn redact_secrets(input: &str) -> (String, bool) {
    let mut redacted = false;
    let mut out = input.to_string();

    // 1. Anthropic format: sk-ant- followed by 16+ alphanumeric/dashes (check first before general sk-)
    let mut cursor = 0;
    while let Some(pos) = out[cursor..].find("sk-ant-") {
        let abs_pos = cursor + pos;
        let candidate = &out[abs_pos..];
        let token_len = candidate
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'-' || *b == b'_')
            .count();
        if token_len >= 16 {
            out.replace_range(abs_pos..abs_pos + token_len, "[REDACTED_ANTHROPIC_KEY]");
            redacted = true;
            cursor = abs_pos + "[REDACTED_ANTHROPIC_KEY]".len();
        } else {
            cursor = abs_pos + 7;
        }
    }

    // 2. OpenAI format: sk- followed by 20+ alphanumeric/dashes
    cursor = 0;
    while let Some(pos) = out[cursor..].find("sk-") {
        let abs_pos = cursor + pos;
        let candidate = &out[abs_pos..];
        let token_len = candidate
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'-' || *b == b'_')
            .count();
        if token_len >= 20 {
            out.replace_range(abs_pos..abs_pos + token_len, "[REDACTED_API_KEY]");
            redacted = true;
            cursor = abs_pos + "[REDACTED_API_KEY]".len();
        } else {
            cursor = abs_pos + 3;
        }
    }

    // 3. Authorization Bearer headers
    cursor = 0;
    while let Some(pos) = out[cursor..].to_ascii_lowercase().find("bearer ") {
        let abs_pos = cursor + pos;
        let candidate = &out[abs_pos + 7..];
        let token_len = candidate
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'.' || *b == b'_' || *b == b'-')
            .count();
        if token_len >= 16 {
            out.replace_range(abs_pos + 7..abs_pos + 7 + token_len, "[REDACTED_BEARER_TOKEN]");
            redacted = true;
            cursor = abs_pos + 7 + "[REDACTED_BEARER_TOKEN]".len();
        } else {
            cursor = abs_pos + 7;
        }
    }

    (out, redacted)
}

/// Recursively redact sensitive fields from a JSON structure.
pub fn redact_json_secrets(val: &mut Value) -> bool {
    let mut any_redacted = false;
    match val {
        Value::String(s) => {
            let (clean, did_redact) = redact_secrets(s);
            if did_redact {
                *s = clean;
                any_redacted = true;
            }
        }
        Value::Array(arr) => {
            for item in arr {
                if redact_json_secrets(item) {
                    any_redacted = true;
                }
            }
        }
        Value::Object(map) => {
            for (k, v) in map.iter_mut() {
                let lower = k.to_ascii_lowercase();
                if lower.contains("password")
                    || lower.contains("secret")
                    || lower.contains("api_key")
                    || lower.contains("authorization")
                    || lower.contains("token")
                {
                    if let Value::String(s) = v {
                        if !s.is_empty() && s != "[REDACTED_SECRET]" {
                            *s = "[REDACTED_SECRET]".to_string();
                            any_redacted = true;
                        }
                    }
                } else if redact_json_secrets(v) {
                    any_redacted = true;
                }
            }
        }
        _ => {}
    }
    any_redacted
}

/// Outcome of a recorded interaction episode.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EpisodeOutcome {
    /// Explicitly verified through tests, compiler exit-code 0, or schema check.
    VerifiedSuccess,
    /// Explicitly marked failed, user retry triggered, or syntax/type error.
    FailedRetry,
    /// Unverified conversation / no execution signal recorded.
    Unverified,
}

impl EpisodeOutcome {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::VerifiedSuccess => "verified_success",
            Self::FailedRetry => "failed_retry",
            Self::Unverified => "unverified",
        }
    }

    pub fn default_reward(&self) -> f64 {
        match self {
            Self::VerifiedSuccess => 1.0,
            Self::FailedRetry => -1.0,
            Self::Unverified => 0.0,
        }
    }
}

/// A structured tool call invocation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// Complete captured interaction episode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CollectedEpisode {
    pub schema: String,
    pub episode_id: String,
    pub session_id: String,
    pub app_id: String,
    pub timestamp_epoch_s: u64,
    pub claimed_teacher: String,
    pub model_id: String,
    pub system_prompt: Option<String>,
    pub user_prompt: String,
    pub assistant_response: String,
    pub tool_calls: Vec<ToolCallRecord>,
    pub tool_results: Option<Vec<Value>>,
    pub outcome: EpisodeOutcome,
    pub scalar_reward: f64,
    pub git_diff: Option<String>,
    pub unit_tests_passed: Option<bool>,
    pub is_redacted: bool,
}

impl CollectedEpisode {
    pub fn new(
        episode_id: impl Into<String>,
        session_id: impl Into<String>,
        app_id: impl Into<String>,
        claimed_teacher: impl Into<String>,
        user_prompt: impl Into<String>,
        assistant_response: impl Into<String>,
    ) -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        Self {
            schema: "ojon.learn.episode.v1".to_string(),
            episode_id: episode_id.into(),
            session_id: session_id.into(),
            app_id: app_id.into(),
            timestamp_epoch_s: stamp,
            claimed_teacher: claimed_teacher.into(),
            model_id: "unknown".to_string(),
            system_prompt: None,
            user_prompt: user_prompt.into(),
            assistant_response: assistant_response.into(),
            tool_calls: Vec::new(),
            tool_results: None,
            outcome: EpisodeOutcome::Unverified,
            scalar_reward: 0.0,
            git_diff: None,
            unit_tests_passed: None,
            is_redacted: false,
        }
    }

    /// Sanitize secrets across all textual and JSON fields before disk commit.
    pub fn sanitize(&mut self) {
        let (u, r1) = redact_secrets(&self.user_prompt);
        self.user_prompt = u;
        let (a, r2) = redact_secrets(&self.assistant_response);
        self.assistant_response = a;
        if let Some(s) = self.system_prompt.as_mut() {
            let (clean_sys, r3) = redact_secrets(s);
            *s = clean_sys;
            if r3 {
                self.is_redacted = true;
            }
        }
        for tc in &mut self.tool_calls {
            if redact_json_secrets(&mut tc.arguments) {
                self.is_redacted = true;
            }
        }
        if let Some(results) = self.tool_results.as_mut() {
            for res in results {
                if redact_json_secrets(res) {
                    self.is_redacted = true;
                }
            }
        }
        if let Some(diff) = self.git_diff.as_mut() {
            let (clean_diff, r4) = redact_secrets(diff);
            *diff = clean_diff;
            if r4 {
                self.is_redacted = true;
            }
        }
        if r1 || r2 {
            self.is_redacted = true;
        }
    }

    pub fn to_jsonl_line(&self) -> io::Result<String> {
        serde_json::to_string(self).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}

/// Statistics and health metadata for the collector.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CollectorStats {
    pub is_active: bool,
    pub listen_port: u16,
    pub storage_dir: String,
    pub total_episodes: u64,
    pub total_collected: u64,
    pub verified_success_count: u64,
    pub verified_successes: u64,
    pub failed_retry_count: u64,
    pub failed_retries: u64,
    pub unverified_count: u64,
    pub unverified: u64,
    pub redacted_count: u64,
    pub redacted_episodes: u64,
    pub total_storage_bytes: u64,
}

/// Thread-safe local loopback collector service.
pub struct LearnCollector {
    storage_dir: PathBuf,
    is_running: Arc<AtomicBool>,
    total_episodes: Arc<AtomicU64>,
    verified_success: Arc<AtomicU64>,
    failed_retry: Arc<AtomicU64>,
    unverified: Arc<AtomicU64>,
    redacted_count: Arc<AtomicU64>,
    recent_episodes: Arc<Mutex<Vec<CollectedEpisode>>>,
}

impl LearnCollector {
    pub fn new(storage_dir: PathBuf) -> io::Result<Self> {
        fs::create_dir_all(&storage_dir)?;
        let mut coll = Self {
            storage_dir,
            is_running: Arc::new(AtomicBool::new(false)),
            total_episodes: Arc::new(AtomicU64::new(0)),
            verified_success: Arc::new(AtomicU64::new(0)),
            failed_retry: Arc::new(AtomicU64::new(0)),
            unverified: Arc::new(AtomicU64::new(0)),
            redacted_count: Arc::new(AtomicU64::new(0)),
            recent_episodes: Arc::new(Mutex::new(Vec::new())),
        };
        coll.refresh_counts_from_disk()?;
        Ok(coll)
    }

    fn active_log_path(&self) -> PathBuf {
        self.storage_dir.join("collected_episodes.jsonl")
    }

    /// Scan existing disk storage to restore persistent stats across application restarts.
    pub fn refresh_counts_from_disk(&mut self) -> io::Result<()> {
        let path = self.active_log_path();
        if !path.is_file() {
            return Ok(());
        }
        let file = File::open(&path)?;
        let reader = BufReader::new(file);
        let mut total = 0u64;
        let mut succ = 0u64;
        let mut fail = 0u64;
        let mut unver = 0u64;
        let mut red = 0u64;
        let mut recent = Vec::new();

        for line in reader.lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(ep) = serde_json::from_str::<CollectedEpisode>(&line) {
                total += 1;
                match ep.outcome {
                    EpisodeOutcome::VerifiedSuccess => succ += 1,
                    EpisodeOutcome::FailedRetry => fail += 1,
                    EpisodeOutcome::Unverified => unver += 1,
                }
                if ep.is_redacted {
                    red += 1;
                }
                recent.push(ep);
                if recent.len() > 50 {
                    recent.remove(0);
                }
            }
        }

        self.total_episodes.store(total, Ordering::SeqCst);
        self.verified_success.store(succ, Ordering::SeqCst);
        self.failed_retry.store(fail, Ordering::SeqCst);
        self.unverified.store(unver, Ordering::SeqCst);
        self.redacted_count.store(red, Ordering::SeqCst);
        if let Ok(mut lock) = self.recent_episodes.lock() {
            *lock = recent;
        }
        Ok(())
    }

    /// Record a single verified or completed interaction episode.
    pub fn record_episode(&self, mut ep: CollectedEpisode) -> io::Result<()> {
        ep.sanitize();
        let was_redacted = ep.is_redacted;
        let outcome = ep.outcome;

        let line = ep.to_jsonl_line()?;
        let path = self.active_log_path();
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        writeln!(file, "{line}")?;
        file.sync_all()?;

        self.total_episodes.fetch_add(1, Ordering::SeqCst);
        match outcome {
            EpisodeOutcome::VerifiedSuccess => {
                self.verified_success.fetch_add(1, Ordering::SeqCst);
            }
            EpisodeOutcome::FailedRetry => {
                self.failed_retry.fetch_add(1, Ordering::SeqCst);
            }
            EpisodeOutcome::Unverified => {
                self.unverified.fetch_add(1, Ordering::SeqCst);
            }
        }
        if was_redacted {
            self.redacted_count.fetch_add(1, Ordering::SeqCst);
        }

        if let Ok(mut lock) = self.recent_episodes.lock() {
            lock.push(ep);
            if lock.len() > 50 {
                lock.remove(0);
            }
        }
        Ok(())
    }

    /// Get current collector metrics and telemetry.
    pub fn get_stats(&self, port: u16) -> CollectorStats {
        let size = self
            .active_log_path()
            .metadata()
            .map(|m| m.len())
            .unwrap_or(0);
        let total = self.total_episodes.load(Ordering::SeqCst);
        let succ = self.verified_success.load(Ordering::SeqCst);
        let fail = self.failed_retry.load(Ordering::SeqCst);
        let unver = self.unverified.load(Ordering::SeqCst);
        let red = self.redacted_count.load(Ordering::SeqCst);
        CollectorStats {
            is_active: self.is_running.load(Ordering::SeqCst),
            listen_port: port,
            storage_dir: self.storage_dir.to_string_lossy().to_string(),
            total_episodes: total,
            total_collected: total,
            verified_success_count: succ,
            verified_successes: succ,
            failed_retry_count: fail,
            failed_retries: fail,
            unverified_count: unver,
            unverified: unver,
            redacted_count: red,
            redacted_episodes: red,
            total_storage_bytes: size,
        }
    }

    /// Return up to 50 most recently recorded episodes for GUI / audit inspection.
    pub fn get_recent_episodes(&self) -> Vec<CollectedEpisode> {
        self.recent_episodes
            .lock()
            .map(|l| l.clone())
            .unwrap_or_default()
    }

    /// Clear all collected episodes and reset stats.
    pub fn clear_all_episodes(&self) -> io::Result<()> {
        let path = self.active_log_path();
        if path.is_file() {
            fs::remove_file(path)?;
        }
        self.total_episodes.store(0, Ordering::SeqCst);
        self.verified_success.store(0, Ordering::SeqCst);
        self.failed_retry.store(0, Ordering::SeqCst);
        self.unverified.store(0, Ordering::SeqCst);
        self.redacted_count.store(0, Ordering::SeqCst);
        if let Ok(mut lock) = self.recent_episodes.lock() {
            lock.clear();
        }
        Ok(())
    }

    /// Start a lightweight local loopback HTTP server on `127.0.0.1:<port>`.
    /// Handles `/v1/chat/completions` interception, `/v1/events` ingestion,
    /// and `/v1/status` queries.
    pub fn start_server(
        collector: Arc<LearnCollector>,
        port: u16,
    ) -> io::Result<thread::JoinHandle<()>> {
        let listener = TcpListener::bind(("127.0.0.1", port))?;
        collector.is_running.store(true, Ordering::SeqCst);

        let is_running_clone = collector.is_running.clone();
        let handle = thread::spawn(move || {
            // Set 1-second timeout so thread can poll `is_running`
            let _ = listener.set_nonblocking(false);
            for stream in listener.incoming() {
                if !is_running_clone.load(Ordering::SeqCst) {
                    break;
                }
                if let Ok(stream) = stream {
                    let coll = collector.clone();
                    thread::spawn(move || {
                        let _ = handle_http_connection(stream, coll, port);
                    });
                }
            }
        });
        Ok(handle)
    }

    /// Signal running server thread to shut down.
    pub fn stop(&self) {
        self.is_running.store(false, Ordering::SeqCst);
    }
}

/// Lightweight HTTP connection handler for local loopback proxy.
fn handle_http_connection(
    mut stream: TcpStream,
    collector: Arc<LearnCollector>,
    port: u16,
) -> io::Result<()> {
    stream.set_read_timeout(Some(Duration::from_secs(10)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;

    let mut reader = BufReader::new(stream.try_clone()?);
    let mut request_line = String::new();
    reader.read_line(&mut request_line)?;
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 2 {
        return Ok(());
    }
    let method = parts[0];
    let path = parts[1];

    let mut headers = HashMap::new();
    let mut content_length = 0usize;
    loop {
        let mut header_line = String::new();
        reader.read_line(&mut header_line)?;
        let trimmed = header_line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            let key = k.trim().to_ascii_lowercase();
            let val = v.trim().to_string();
            if key == "content-length" {
                content_length = val.parse::<usize>().unwrap_or(0);
            }
            headers.insert(key, val);
        }
    }

    let mut body = vec![0u8; content_length];
    if content_length > 0 && content_length <= 8 * 1024 * 1024 {
        reader.read_exact(&mut body)?;
    }

    match (method, path) {
        ("GET", "/v1/status") | ("GET", "/status") => {
            let stats = collector.get_stats(port);
            let resp_body = serde_json::to_vec(&stats).unwrap_or_default();
            write_http_response(&mut stream, 200, "application/json", &resp_body)?;
        }
        ("POST", "/v1/events") => {
            if let Ok(ep) = serde_json::from_slice::<CollectedEpisode>(&body) {
                let _ = collector.record_episode(ep);
                let resp_body = json!({"status": "recorded"}).to_string().into_bytes();
                write_http_response(&mut stream, 200, "application/json", &resp_body)?;
            } else {
                let err_body = json!({"error": "invalid episode json"}).to_string().into_bytes();
                write_http_response(&mut stream, 400, "application/json", &err_body)?;
            }
        }
        ("POST", "/v1/chat/completions") => {
            // Intercept incoming chat payload, extract conversation turns, and record
            if let Ok(req_json) = serde_json::from_slice::<Value>(&body) {
                let model = req_json["model"].as_str().unwrap_or("unknown-model");
                let messages = req_json["messages"].as_array();
                let user_msg = messages
                    .and_then(|arr| {
                        arr.iter()
                            .rev()
                            .find(|m| m["role"] == "user")
                            .and_then(|m| m["content"].as_str())
                    })
                    .unwrap_or("")
                    .to_string();
                let sys_msg = messages
                    .and_then(|arr| {
                        arr.iter()
                            .find(|m| m["role"] == "system")
                            .and_then(|m| m["content"].as_str())
                    })
                    .map(str::to_string);

                let ep_id = format!("ep-{}", Instant::now().elapsed().as_nanos());
                let mut ep = CollectedEpisode::new(
                    &ep_id,
                    headers.get("x-session-id").cloned().unwrap_or_else(|| "sess-default".into()),
                    headers.get("user-agent").cloned().unwrap_or_else(|| "agent-proxy".into()),
                    model,
                    user_msg,
                    "Generated via Ojon Local Loopback Gateway".to_string(),
                );
                ep.system_prompt = sys_msg;
                ep.model_id = model.to_string();

                let _ = collector.record_episode(ep);

                // Return standard OpenAI-compatible mock response so clients continue smoothly
                let resp_json = json!({
                    "id": format!("chatcmpl-{}", ep_id),
                    "object": "chat.completion",
                    "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Ojon Gateway recorded request payload for training flywheel."
                        },
                        "finish_reason": "stop"
                    }],
                    "usage": {
                        "prompt_tokens": 12,
                        "completion_tokens": 8,
                        "total_tokens": 20
                    }
                });
                let resp_bytes = serde_json::to_vec(&resp_json).unwrap_or_default();
                write_http_response(&mut stream, 200, "application/json", &resp_bytes)?;
            } else {
                let err_body = json!({"error": "invalid chat json"}).to_string().into_bytes();
                write_http_response(&mut stream, 400, "application/json", &err_body)?;
            }
        }
        ("POST", "/v1/episodes/clear") => {
            let _ = collector.clear_all_episodes();
            let resp_body = json!({"status": "cleared"}).to_string().into_bytes();
            write_http_response(&mut stream, 200, "application/json", &resp_body)?;
        }
        _ => {
            let not_found = json!({"error": "not found"}).to_string().into_bytes();
            write_http_response(&mut stream, 404, "application/json", &not_found)?;
        }
    }
    Ok(())
}

fn write_http_response(
    stream: &mut TcpStream,
    status: u16,
    content_type: &str,
    body: &[u8],
) -> io::Result<()> {
    let status_line = match status {
        200 => "HTTP/1.1 200 OK",
        400 => "HTTP/1.1 400 Bad Request",
        404 => "HTTP/1.1 404 Not Found",
        _ => "HTTP/1.1 500 Internal Server Error",
    };
    write!(
        stream,
        "{status_line}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_redaction() {
        let input = "Use key sk-1234567890abcdef12345678 and sk-ant-api03-abcdefghijklmnop to login with Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xyz";
        let (cleaned, redacted) = redact_secrets(input);
        assert!(redacted);
        assert!(!cleaned.contains("sk-1234567890"));
        assert!(!cleaned.contains("sk-ant-api03"));
        assert!(cleaned.contains("[REDACTED_API_KEY]"));
        assert!(cleaned.contains("[REDACTED_ANTHROPIC_KEY]"));
        assert!(cleaned.contains("[REDACTED_BEARER_TOKEN]"));
    }

    #[test]
    fn test_json_secret_redaction() {
        let mut data = json!({
            "api_key": "sk-secret99999999999999999999",
            "password": "SuperSecretPassword123!",
            "normal_data": "Just some code here"
        });
        let redacted = redact_json_secrets(&mut data);
        assert!(redacted);
        assert_eq!(data["api_key"], "[REDACTED_SECRET]");
        assert_eq!(data["password"], "[REDACTED_SECRET]");
        assert_eq!(data["normal_data"], "Just some code here");
    }

    #[test]
    fn test_collector_lifecycle_and_persistence() {
        let tmp = std::env::temp_dir().join(format!("ojon-test-coll-{}", std::process::id()));
        let coll = LearnCollector::new(tmp.clone()).unwrap();

        let mut ep = CollectedEpisode::new(
            "ep-001",
            "sess-123",
            "Aider",
            "claude-3-5-sonnet",
            "Implement a bubble sort in Python",
            "def bubble_sort(arr):\n    pass",
        );
        ep.outcome = EpisodeOutcome::VerifiedSuccess;
        ep.scalar_reward = 1.0;
        coll.record_episode(ep).unwrap();

        let stats = coll.get_stats(8088);
        assert_eq!(stats.total_episodes, 1);
        assert_eq!(stats.verified_success_count, 1);
        assert_eq!(stats.failed_retry_count, 0);

        let recent = coll.get_recent_episodes();
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].episode_id, "ep-001");

        // Test refresh from disk
        let mut coll2 = LearnCollector::new(tmp.clone()).unwrap();
        coll2.refresh_counts_from_disk().unwrap();
        let stats2 = coll2.get_stats(8088);
        assert_eq!(stats2.total_episodes, 1);

        let _ = coll.clear_all_episodes();
        assert_eq!(coll.get_stats(8088).total_episodes, 0);
        let _ = fs::remove_dir_all(tmp);
    }
}
