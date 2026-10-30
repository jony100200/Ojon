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
            out.replace_range(
                abs_pos + 7..abs_pos + 7 + token_len,
                "[REDACTED_BEARER_TOKEN]",
            );
            redacted = true;
            cursor = abs_pos + 7 + "[REDACTED_BEARER_TOKEN]".len();
        } else {
            cursor = abs_pos + 7;
        }
    }

    // 4. GitHub tokens (ghp_, gho_, ghu_, ghs_, ghr_, github_pat_)
    for prefix in ["github_pat_", "ghp_", "gho_", "ghu_", "ghs_", "ghr_"] {
        cursor = 0;
        while let Some(pos) = out[cursor..].find(prefix) {
            let abs_pos = cursor + pos;
            let candidate = &out[abs_pos..];
            let token_len = candidate
                .bytes()
                .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_')
                .count();
            if token_len >= prefix.len() + 16 {
                out.replace_range(abs_pos..abs_pos + token_len, "[REDACTED_GITHUB_TOKEN]");
                redacted = true;
                cursor = abs_pos + "[REDACTED_GITHUB_TOKEN]".len();
            } else {
                cursor = abs_pos + prefix.len();
            }
        }
    }

    // 5. HuggingFace tokens (hf_ followed by 20+ alphanumeric chars)
    cursor = 0;
    while let Some(pos) = out[cursor..].find("hf_") {
        let abs_pos = cursor + pos;
        let candidate = &out[abs_pos..];
        let token_len = 3 + candidate[3..]
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric())
            .count();
        if token_len >= 23 {
            out.replace_range(abs_pos..abs_pos + token_len, "[REDACTED_HF_TOKEN]");
            redacted = true;
            cursor = abs_pos + "[REDACTED_HF_TOKEN]".len();
        } else {
            cursor = abs_pos + 3;
        }
    }

    // 6. AWS Access Key IDs (AKIA followed by 16 uppercase alphanumeric chars)
    cursor = 0;
    while let Some(pos) = out[cursor..].find("AKIA") {
        let abs_pos = cursor + pos;
        let candidate = &out[abs_pos..];
        let token_len = candidate
            .bytes()
            .take_while(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
            .count();
        if token_len >= 20 {
            out.replace_range(abs_pos..abs_pos + token_len, "[REDACTED_AWS_KEY]");
            redacted = true;
            cursor = abs_pos + "[REDACTED_AWS_KEY]".len();
        } else {
            cursor = abs_pos + 4;
        }
    }

    // 7. Google API Keys (AIza followed by 25+ alphanumeric/dash/underscore chars)
    cursor = 0;
    while let Some(pos) = out[cursor..].find("AIza") {
        let abs_pos = cursor + pos;
        let candidate = &out[abs_pos..];
        let token_len = candidate
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'-' || *b == b'_')
            .count();
        if token_len >= 29 {
            out.replace_range(abs_pos..abs_pos + token_len, "[REDACTED_GOOGLE_KEY]");
            redacted = true;
            cursor = abs_pos + "[REDACTED_GOOGLE_KEY]".len();
        } else {
            cursor = abs_pos + 4;
        }
    }

    // 8. URL basic-auth credentials (://user:password@host)
    cursor = 0;
    while let Some(pos) = out[cursor..].find("://") {
        let abs_pos = cursor + pos + 3;
        let rest = &out[abs_pos..];
        let authority_end = rest.find('/').unwrap_or(rest.len());
        let authority = &rest[..authority_end];
        if let Some(at_pos) = authority.find('@') {
            let user_info = &authority[..at_pos];
            if user_info.contains(':') && !user_info.contains("[REDACTED_CREDENTIALS]") {
                out.replace_range(abs_pos..abs_pos + at_pos, "[REDACTED_CREDENTIALS]");
                redacted = true;
                cursor = abs_pos + "[REDACTED_CREDENTIALS]".len() + 1;
                continue;
            }
        }
        cursor = abs_pos;
    }

    // 9. OS User Profile Paths (C:\Users\<name>, /Users/<name>, /home/<name>)
    for prefix in [
        "C:\\Users\\",
        "c:\\Users\\",
        "C:/Users/",
        "/Users/",
        "/home/",
    ] {
        cursor = 0;
        while let Some(pos) = out[cursor..].find(prefix) {
            let user_start = cursor + pos + prefix.len();
            let rest = &out[user_start..];
            let user_len = rest
                .bytes()
                .take_while(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'-' || *b == b'.')
                .count();
            if user_len > 0 {
                let username = &rest[..user_len];
                if username != "[REDACTED_USER]" && username != "Public" && username != "Default" {
                    out.replace_range(user_start..user_start + user_len, "[REDACTED_USER]");
                    redacted = true;
                    cursor = user_start + "[REDACTED_USER]".len();
                    continue;
                }
            }
            cursor = user_start;
        }
    }

    // 10. Email addresses (local@domain.tld)
    cursor = 0;
    while let Some(pos) = out[cursor..].find('@') {
        let at_pos = cursor + pos;
        let before = &out[..at_pos];
        let local_len = before
            .bytes()
            .rev()
            .take_while(|b| {
                b.is_ascii_alphanumeric() || *b == b'.' || *b == b'_' || *b == b'-' || *b == b'+'
            })
            .count();
        let after = &out[at_pos + 1..];
        let domain_len = after
            .bytes()
            .take_while(|b| b.is_ascii_alphanumeric() || *b == b'.' || *b == b'-')
            .count();
        let domain = &after[..domain_len];
        if local_len >= 2
            && domain_len >= 4
            && domain.contains('.')
            && !domain.starts_with('.')
            && !domain.ends_with('.')
        {
            let start = at_pos - local_len;
            let end = at_pos + 1 + domain_len;
            out.replace_range(start..end, "[REDACTED_EMAIL]");
            redacted = true;
            cursor = start + "[REDACTED_EMAIL]".len();
        } else {
            cursor = at_pos + 1;
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
    upstream_endpoint: Arc<Mutex<Option<String>>>,
    auth_token: Arc<Mutex<Option<String>>>,
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
            upstream_endpoint: Arc::new(Mutex::new(None)),
            auth_token: Arc::new(Mutex::new(None)),
        };
        coll.refresh_counts_from_disk()?;
        Ok(coll)
    }

    /// Configure an upstream OpenAI-compatible server (e.g. `http://127.0.0.1:1234/v1`)
    /// so `/v1/chat/completions` forwards traffic transparently and captures real responses.
    pub fn with_upstream_endpoint(self, url: impl Into<String>) -> Self {
        self.set_upstream_endpoint(Some(url.into()));
        self
    }

    pub fn set_upstream_endpoint(&self, url: Option<String>) {
        if let Ok(mut lock) = self.upstream_endpoint.lock() {
            *lock = url;
        }
    }

    pub fn upstream_endpoint(&self) -> Option<String> {
        self.upstream_endpoint.lock().ok().and_then(|g| g.clone())
    }

    /// Configure an authentication token required for destructive management endpoints.
    pub fn with_auth_token(self, token: impl Into<String>) -> Self {
        self.set_auth_token(Some(token.into()));
        self
    }

    pub fn set_auth_token(&self, token: Option<String>) {
        if let Ok(mut lock) = self.auth_token.lock() {
            *lock = token;
        }
    }

    pub fn auth_token(&self) -> Option<String> {
        self.auth_token.lock().ok().and_then(|g| g.clone())
    }

    /// Ingest OpenTelemetry `gen_ai.*` span payloads (`resourceSpans` or flat `spans` array).
    pub fn ingest_otel_genai_spans(&self, payload: &Value) -> io::Result<usize> {
        let mut spans: Vec<&Value> = Vec::new();
        if let Some(res_spans) = payload["resourceSpans"].as_array() {
            for rs in res_spans {
                if let Some(scope_spans) = rs["scopeSpans"].as_array() {
                    for ss in scope_spans {
                        if let Some(sp_arr) = ss["spans"].as_array() {
                            spans.extend(sp_arr.iter());
                        }
                    }
                }
            }
        } else if let Some(sp_arr) = payload["spans"].as_array() {
            spans.extend(sp_arr.iter());
        }

        let mut recorded = 0usize;
        for sp in spans {
            let span_id = sp["spanId"]
                .as_str()
                .or_else(|| sp["span_id"].as_str())
                .unwrap_or("otel-span");
            let trace_id = sp["traceId"]
                .as_str()
                .or_else(|| sp["trace_id"].as_str())
                .unwrap_or("otel-trace");

            let mut attrs = HashMap::new();
            if let Some(attr_arr) = sp["attributes"].as_array() {
                for a in attr_arr {
                    if let Some(k) = a["key"].as_str() {
                        let val_str = a["value"]["stringValue"]
                            .as_str()
                            .map(str::to_string)
                            .or_else(|| a["value"]["intValue"].as_i64().map(|n| n.to_string()))
                            .or_else(|| a["value"]["boolValue"].as_bool().map(|b| b.to_string()))
                            .or_else(|| a["value"].as_str().map(str::to_string));
                        if let Some(v) = val_str {
                            attrs.insert(k.to_string(), v);
                        }
                    }
                }
            } else if let Some(attr_obj) = sp["attributes"].as_object() {
                for (k, v) in attr_obj {
                    if let Some(s) = v.as_str() {
                        attrs.insert(k.clone(), s.to_string());
                    } else {
                        attrs.insert(k.clone(), v.to_string());
                    }
                }
            }

            let prompt = attrs
                .get("gen_ai.prompt")
                .or_else(|| attrs.get("gen_ai.input.messages"))
                .cloned()
                .unwrap_or_default();
            let completion = attrs
                .get("gen_ai.completion")
                .or_else(|| attrs.get("gen_ai.output.messages"))
                .cloned()
                .unwrap_or_default();
            if prompt.is_empty() && completion.is_empty() {
                continue;
            }
            let model = attrs
                .get("gen_ai.request.model")
                .or_else(|| attrs.get("gen_ai.response.model"))
                .cloned()
                .unwrap_or_else(|| "otel-model".to_string());
            let app_id = attrs
                .get("gen_ai.system")
                .or_else(|| attrs.get("service.name"))
                .cloned()
                .unwrap_or_else(|| "otel-agent".to_string());

            let mut ep = CollectedEpisode::new(
                format!("otel-{span_id}"),
                trace_id,
                app_id,
                &model,
                prompt,
                completion,
            );
            ep.model_id = model;
            if let Some(passed) = attrs.get("ojon.unit_tests_passed") {
                if passed == "true" {
                    ep.unit_tests_passed = Some(true);
                    ep.outcome = EpisodeOutcome::VerifiedSuccess;
                    ep.scalar_reward = 1.0;
                } else if passed == "false" {
                    ep.unit_tests_passed = Some(false);
                    ep.outcome = EpisodeOutcome::FailedRetry;
                    ep.scalar_reward = -1.0;
                }
            }
            if let Some(diff) = attrs.get("ojon.git_diff") {
                if !diff.trim().is_empty() {
                    ep.git_diff = Some(diff.clone());
                }
            }
            self.record_episode(ep)?;
            recorded += 1;
        }
        Ok(recorded)
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
        let _ = listener.set_nonblocking(true);
        collector.is_running.store(true, Ordering::SeqCst);

        let is_running_clone = collector.is_running.clone();
        let handle = thread::spawn(move || {
            while is_running_clone.load(Ordering::SeqCst) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        let _ = stream.set_nonblocking(false);
                        let coll = collector.clone();
                        thread::spawn(move || {
                            let _ = handle_http_connection(stream, coll, port);
                        });
                    }
                    Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(50));
                    }
                    Err(_) => break,
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
                let err_body = json!({"error": "invalid episode json"})
                    .to_string()
                    .into_bytes();
                write_http_response(&mut stream, 400, "application/json", &err_body)?;
            }
        }
        ("POST", "/v1/traces") => {
            if let Ok(trace_json) = serde_json::from_slice::<Value>(&body) {
                match collector.ingest_otel_genai_spans(&trace_json) {
                    Ok(count) => {
                        let resp_body = json!({"status": "recorded", "spans_ingested": count})
                            .to_string()
                            .into_bytes();
                        write_http_response(&mut stream, 200, "application/json", &resp_body)?;
                    }
                    Err(e) => {
                        let err_body = json!({"error": e.to_string()}).to_string().into_bytes();
                        write_http_response(&mut stream, 500, "application/json", &err_body)?;
                    }
                }
            } else {
                let err_body = json!({"error": "invalid otel trace json"})
                    .to_string()
                    .into_bytes();
                write_http_response(&mut stream, 400, "application/json", &err_body)?;
            }
        }
        ("GET", "/v1/utility-models") => {
            let models = default_utility_models();
            let resp_body = serde_json::to_vec(&models).unwrap_or_default();
            write_http_response(&mut stream, 200, "application/json", &resp_body)?;
        }
        ("POST", "/v1/chat/completions") => {
            // Intercept incoming chat payload, optionally forward to upstream LLM server,
            // extract conversation turns & tool_calls, and record to flywheel store.
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

                let ep_id = format!(
                    "ep-{}-{}",
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis(),
                    Instant::now().elapsed().as_nanos()
                );

                let upstream_target = headers
                    .get("x-ojon-upstream")
                    .cloned()
                    .or_else(|| collector.upstream_endpoint());

                let (resp_json, assistant_text, tool_calls) = if let Some(ref up_url) =
                    upstream_target
                {
                    match forward_chat_completion_to_upstream(up_url, &body) {
                        Ok(up_resp) => {
                            let choice_msg = &up_resp["choices"][0]["message"];
                            let content = choice_msg["content"].as_str().unwrap_or("").to_string();
                            let mut parsed_tools = Vec::new();
                            if let Some(tc_arr) = choice_msg["tool_calls"].as_array() {
                                for tc in tc_arr {
                                    let id = tc["id"].as_str().unwrap_or("call_0").to_string();
                                    let name = tc["function"]["name"]
                                        .as_str()
                                        .or_else(|| tc["name"].as_str())
                                        .unwrap_or("unknown_tool")
                                        .to_string();
                                    let args_val = if let Some(arg_str) =
                                        tc["function"]["arguments"].as_str()
                                    {
                                        serde_json::from_str::<Value>(arg_str)
                                            .unwrap_or_else(|_| Value::String(arg_str.to_string()))
                                    } else {
                                        tc["function"]["arguments"].clone()
                                    };
                                    parsed_tools.push(ToolCallRecord {
                                        id,
                                        name,
                                        arguments: args_val,
                                    });
                                }
                            }
                            (up_resp, content, parsed_tools)
                        }
                        Err(_) => {
                            let fallback = json!({
                                "id": format!("chatcmpl-{}", ep_id),
                                "object": "chat.completion",
                                "created": SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs(),
                                "model": model,
                                "choices": [{
                                    "index": 0,
                                    "message": {
                                        "role": "assistant",
                                        "content": "Ojon Gateway recorded request payload for training flywheel (upstream unreachable)."
                                    },
                                    "finish_reason": "stop"
                                }],
                                "usage": {
                                    "prompt_tokens": 12,
                                    "completion_tokens": 8,
                                    "total_tokens": 20
                                }
                            });
                            (
                                fallback,
                                "Generated via Ojon Local Loopback Gateway".to_string(),
                                Vec::new(),
                            )
                        }
                    }
                } else {
                    let mock = json!({
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
                    (
                        mock,
                        "Generated via Ojon Local Loopback Gateway".to_string(),
                        Vec::new(),
                    )
                };

                let mut ep = CollectedEpisode::new(
                    &ep_id,
                    headers
                        .get("x-session-id")
                        .cloned()
                        .unwrap_or_else(|| "sess-default".into()),
                    headers
                        .get("user-agent")
                        .cloned()
                        .unwrap_or_else(|| "agent-proxy".into()),
                    model,
                    user_msg,
                    assistant_text,
                );
                ep.system_prompt = sys_msg;
                ep.model_id = model.to_string();
                ep.tool_calls = tool_calls;

                let _ = collector.record_episode(ep);

                let resp_bytes = serde_json::to_vec(&resp_json).unwrap_or_default();
                write_http_response(&mut stream, 200, "application/json", &resp_bytes)?;
            } else {
                let err_body = json!({"error": "invalid chat json"})
                    .to_string()
                    .into_bytes();
                write_http_response(&mut stream, 400, "application/json", &err_body)?;
            }
        }
        ("POST", "/v1/episodes/clear") => {
            let body_confirmed = serde_json::from_slice::<Value>(&body)
                .ok()
                .and_then(|v| v["confirm"].as_bool())
                .unwrap_or(false);
            let header_confirmed = headers
                .get("x-ojon-confirm")
                .map(|v| v.eq_ignore_ascii_case("true") || v == "1")
                .unwrap_or(false);
            let token_valid = if let Some(expected_tok) = collector.auth_token() {
                headers
                    .get("authorization")
                    .and_then(|h| {
                        h.strip_prefix("Bearer ")
                            .or_else(|| h.strip_prefix("bearer "))
                    })
                    .map(|t| t.trim() == expected_tok)
                    .unwrap_or(false)
            } else {
                true
            };

            if !(body_confirmed || header_confirmed) || !token_valid {
                let err_body = json!({
                    "error": "confirmation or authorization required to clear episodes",
                    "hint": "pass {\"confirm\": true} in request body or X-Ojon-Confirm: true header"
                })
                .to_string()
                .into_bytes();
                write_http_response(&mut stream, 403, "application/json", &err_body)?;
            } else {
                let _ = collector.clear_all_episodes();
                let resp_body = json!({"status": "cleared"}).to_string().into_bytes();
                write_http_response(&mut stream, 200, "application/json", &resp_body)?;
            }
        }
        _ => {
            let not_found = json!({"error": "not found"}).to_string().into_bytes();
            write_http_response(&mut stream, 404, "application/json", &not_found)?;
        }
    }
    Ok(())
}

/// Forward a `/v1/chat/completions` JSON payload to an upstream local HTTP endpoint (e.g. `http://127.0.0.1:1234/v1`).
fn forward_chat_completion_to_upstream(upstream_base: &str, body: &[u8]) -> io::Result<Value> {
    let trimmed = upstream_base.trim().trim_end_matches('/');
    let without_scheme = trimmed.strip_prefix("http://").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "upstream must start with http://",
        )
    })?;
    let (host_port, base_path) = match without_scheme.find('/') {
        Some(idx) => (&without_scheme[..idx], &without_scheme[idx..]),
        None => (without_scheme, ""),
    };
    let target_path = if base_path.ends_with("/chat/completions") {
        base_path.to_string()
    } else if base_path.ends_with("/v1") {
        format!("{base_path}/chat/completions")
    } else if base_path.is_empty() {
        "/v1/chat/completions".to_string()
    } else {
        format!("{base_path}/chat/completions")
    };

    let addr = if host_port.contains(':') {
        host_port.to_string()
    } else {
        format!("{host_port}:80")
    };

    let mut stream = TcpStream::connect(addr)?;
    stream.set_read_timeout(Some(Duration::from_secs(30)))?;
    stream.set_write_timeout(Some(Duration::from_secs(10)))?;

    write!(
        stream,
        "POST {target_path} HTTP/1.1\r\nHost: {host_port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    stream.write_all(body)?;
    stream.flush()?;

    let mut resp_bytes = Vec::new();
    stream.read_to_end(&mut resp_bytes)?;
    let resp_str = String::from_utf8_lossy(&resp_bytes);
    let (_, json_part) = resp_str
        .split_once("\r\n\r\n")
        .or_else(|| resp_str.split_once("\n\n"))
        .ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "invalid upstream HTTP response")
        })?;

    serde_json::from_str::<Value>(json_part.trim()).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("invalid upstream JSON: {e}"),
        )
    })
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
        403 => "HTTP/1.1 403 Forbidden",
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

/// Specification for a small specialised utility model used by Ojon Learn
/// to extract, embed, rerank, format, or scrub agent traces without requiring a heavy LLM.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UtilityModelSpec {
    pub id: String,
    pub name: String,
    pub repo_or_engine: String,
    pub parameter_scale: String,
    pub role: String,
    pub recommendation: String,
    pub cpu_friendly: bool,
}

/// Canonical registry of the 6 specialised utility models for Ojon Learn.
pub fn default_utility_models() -> Vec<UtilityModelSpec> {
    vec![
        UtilityModelSpec {
            id: "gliner2.5-small-74m".to_string(),
            name: "GLiNER2.5 Small (74M)".to_string(),
            repo_or_engine: "fastino/gliner2.5-small".to_string(),
            parameter_scale: "74M".to_string(),
            role: "Extract tasks, errors, file names, fixes and outcome labels from agent logs"
                .to_string(),
            recommendation: "First choice for CPU-friendly extraction".to_string(),
            cpu_friendly: true,
        },
        UtilityModelSpec {
            id: "qwen3-embedding-0.6b".to_string(),
            name: "Qwen3-Embedding-0.6B".to_string(),
            repo_or_engine: "Qwen/Qwen3-Embedding-0.6B".to_string(),
            parameter_scale: "0.6B".to_string(),
            role: "Create embeddings for deduplication, clustering, similarity search and memory indexing"
                .to_string(),
            recommendation: "First choice for dataset hygiene".to_string(),
            cpu_friendly: true,
        },
        UtilityModelSpec {
            id: "qwen3-reranker-0.6b".to_string(),
            name: "Qwen3-Reranker-0.6B".to_string(),
            repo_or_engine: "Qwen/Qwen3-Reranker-0.6B".to_string(),
            parameter_scale: "0.6B".to_string(),
            role: "Score which agent traces, patches or answers are highest quality".to_string(),
            recommendation: "First choice for filtering before training".to_string(),
            cpu_friendly: true,
        },
        UtilityModelSpec {
            id: "qwen3.5-0.8b".to_string(),
            name: "Qwen3.5-0.8B".to_string(),
            repo_or_engine: "Qwen/Qwen3.5-0.8B".to_string(),
            parameter_scale: "0.8B".to_string(),
            role: "Fast local summariser, instruction cleaner and training-sample formatter"
                .to_string(),
            recommendation: "Good lightweight generative helper".to_string(),
            cpu_friendly: true,
        },
        UtilityModelSpec {
            id: "nuextract-2.0-2b".to_string(),
            name: "NuExtract 2.0 (2B)".to_string(),
            repo_or_engine: "numind/NuExtract-2.0-2B".to_string(),
            parameter_scale: "2B".to_string(),
            role: "Turn messy logs, diffs, docs and chat transcripts into clean JSON training records"
                .to_string(),
            recommendation: "Best when logs are noisy".to_string(),
            cpu_friendly: false,
        },
        UtilityModelSpec {
            id: "presidio-pii-scrubber".to_string(),
            name: "Microsoft Presidio + Secret Scrubber".to_string(),
            repo_or_engine: "microsoft/presidio".to_string(),
            parameter_scale: "Deterministic + NER".to_string(),
            role: "Remove API keys, paths, emails, tokens and personal data before saving datasets"
                .to_string(),
            recommendation: "Mandatory before dataset export".to_string(),
            cpu_friendly: true,
        },
    ]
}

/// Structured metadata extracted from a `CollectedEpisode` (GLiNER2.5 / Reranker schema).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExtractedEpisodeMetadata {
    pub episode_id: String,
    pub task_category: String,
    pub detected_files: Vec<String>,
    pub detected_errors: Vec<String>,
    pub detected_tools: Vec<String>,
    pub quality_score: f64,
    pub recommended_route: String,
}

/// CPU-fast structured entity & quality extractor for recorded agent episodes.
pub fn extract_episode_metadata(ep: &CollectedEpisode) -> ExtractedEpisodeMetadata {
    let combined = format!(
        "{}\n{}\n{}",
        ep.user_prompt,
        ep.assistant_response,
        ep.git_diff.as_deref().unwrap_or("")
    );
    let lower = combined.to_ascii_lowercase();

    let task_category =
        if lower.contains("fix") || lower.contains("bug") || lower.contains("error") {
            "bugfix"
        } else if lower.contains("test") || lower.contains("assert") {
            "testing"
        } else if lower.contains("refactor") || lower.contains("clean") {
            "refactoring"
        } else if lower.contains("doc") || lower.contains("readme") {
            "documentation"
        } else {
            "feature"
        }
        .to_string();

    let mut detected_files = Vec::new();
    let exts = [
        ".rs", ".py", ".ts", ".tsx", ".js", ".json", ".toml", ".yaml", ".yml", ".md", ".c", ".cpp",
        ".h",
    ];
    for token in combined.split(|c: char| {
        c.is_whitespace()
            || matches!(
                c,
                '"' | '\'' | '`' | '(' | ')' | '[' | ']' | '{' | '}' | ',' | ';' | ':'
            )
    }) {
        let trimmed = token.trim_matches(|c: char| c == '.' || c == '/');
        if trimmed.len() >= 3
            && exts.iter().any(|ext| trimmed.ends_with(ext))
            && !detected_files.contains(&trimmed.to_string())
        {
            detected_files.push(trimmed.to_string());
        }
    }

    let mut detected_errors = Vec::new();
    for line in combined.lines() {
        let t = line.trim();
        if t.contains("error[E")
            || t.contains("panicked at")
            || t.contains("Traceback (most recent call last)")
            || t.contains("SyntaxError:")
            || t.contains("AssertionError")
            || t.contains("FAILED")
        {
            let snippet: String = t.chars().take(120).collect();
            if !detected_errors.contains(&snippet) {
                detected_errors.push(snippet);
            }
        }
    }

    let mut detected_tools = Vec::new();
    for tc in &ep.tool_calls {
        if !detected_tools.contains(&tc.name) {
            detected_tools.push(tc.name.clone());
        }
    }

    let has_verif = ep.unit_tests_passed == Some(true)
        || ep
            .git_diff
            .as_ref()
            .map(|d| !d.trim().is_empty())
            .unwrap_or(false);

    let mut quality_score: f64 = match ep.outcome {
        EpisodeOutcome::VerifiedSuccess if has_verif => 0.85,
        EpisodeOutcome::VerifiedSuccess => 0.45,
        EpisodeOutcome::FailedRetry => 0.30,
        EpisodeOutcome::Unverified => 0.25,
    };
    if ep.unit_tests_passed == Some(true) {
        quality_score += 0.10;
    }
    if ep
        .git_diff
        .as_ref()
        .map(|d| d.contains("@@"))
        .unwrap_or(false)
    {
        quality_score += 0.05;
    }
    quality_score = quality_score.clamp(0.0, 1.0);

    let recommended_route = if ep.outcome == EpisodeOutcome::VerifiedSuccess && has_verif {
        "sft_positive"
    } else if ep.outcome == EpisodeOutcome::FailedRetry {
        "dpo_rejected"
    } else {
        "discard_unverified"
    }
    .to_string();

    ExtractedEpisodeMetadata {
        episode_id: ep.episode_id.clone(),
        task_category,
        detected_files,
        detected_errors,
        detected_tools,
        quality_score,
        recommended_route,
    }
}

/// Summary receipt returned when harvesting local agent session logs into Ojon Learn.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarvestReceipt {
    pub source_dir: String,
    pub files_scanned: usize,
    pub episodes_harvested: usize,
    pub verified_success_count: usize,
    pub failed_retry_count: usize,
    pub unverified_count: usize,
    pub duplicates_skipped: usize,
    pub redacted_count: usize,
}

/// Harvest real agent run artifacts from a workspace or log directory (supports Pothik `.pothik/runs/*.json`,
/// Dolil `agent_runs/*.jsonl`, and OpenAI/ShareGPT JSONL traces) into the `LearnCollector` with strict
/// deterministic verification gates and automatic PII/secret scrubbing.
pub fn harvest_agent_sessions(
    source_dir: &std::path::Path,
    collector: &LearnCollector,
) -> io::Result<HarvestReceipt> {
    let mut receipt = HarvestReceipt {
        source_dir: source_dir.to_string_lossy().to_string(),
        ..Default::default()
    };
    if !source_dir.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "harvest source path does not exist: {}",
                source_dir.display()
            ),
        ));
    }

    let mut existing_ids = std::collections::HashSet::new();
    let active_path = collector.active_log_path();
    if active_path.is_file() {
        if let Ok(file) = File::open(&active_path) {
            for line in BufReader::new(file).lines().map_while(Result::ok) {
                if let Ok(ep) = serde_json::from_str::<CollectedEpisode>(&line) {
                    existing_ids.insert(ep.episode_id);
                }
            }
        }
    }

    let mut candidate_files = Vec::new();
    collect_harvest_files(source_dir, 0, 4, &mut candidate_files)?;

    for file_path in candidate_files {
        if file_path == active_path {
            continue;
        }
        receipt.files_scanned += 1;
        let ext = file_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        let parsed_episodes = if ext == "json" {
            parse_json_agent_run(&file_path).unwrap_or_default()
        } else if ext == "jsonl" {
            // Claude Code session transcripts share the extension but not the schema.
            if crate::learn::harvest_claude::is_claude_transcript(&file_path) {
                crate::learn::harvest_claude::parse_claude_transcript(&file_path).unwrap_or_default()
            } else if crate::learn::harvest_dolil::is_dolil_run(&file_path) {
                crate::learn::harvest_dolil::parse_dolil_run(&file_path).unwrap_or_default()
            } else {
                parse_jsonl_agent_run(&file_path).unwrap_or_default()
            }
        } else {
            Vec::new()
        };

        for mut ep in parsed_episodes {
            if existing_ids.contains(&ep.episode_id) {
                receipt.duplicates_skipped += 1;
                continue;
            }
            ep.sanitize();
            existing_ids.insert(ep.episode_id.clone());
            match ep.outcome {
                EpisodeOutcome::VerifiedSuccess => receipt.verified_success_count += 1,
                EpisodeOutcome::FailedRetry => receipt.failed_retry_count += 1,
                EpisodeOutcome::Unverified => receipt.unverified_count += 1,
            }
            if ep.is_redacted {
                receipt.redacted_count += 1;
            }
            collector.record_episode(ep)?;
            receipt.episodes_harvested += 1;
        }
    }

    Ok(receipt)
}

fn collect_harvest_files(
    dir: &std::path::Path,
    depth: usize,
    max_depth: usize,
    out: &mut Vec<PathBuf>,
) -> io::Result<()> {
    if depth > max_depth {
        return Ok(());
    }
    if dir.is_file() {
        out.push(dir.to_path_buf());
        return Ok(());
    }
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if matches!(name, ".git" | "target" | "node_modules" | ".venv") {
                continue;
            }
            collect_harvest_files(&path, depth + 1, max_depth, out)?;
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if ext.eq_ignore_ascii_case("json") || ext.eq_ignore_ascii_case("jsonl") {
                // If transcript_full.jsonl exists alongside transcript.jsonl, prefer transcript_full.jsonl
                // so edit_file / write_file code payloads are never truncated.
                if path.file_name().and_then(|n| n.to_str()) == Some("transcript.jsonl") {
                    let full_sibling = path.with_file_name("transcript_full.jsonl");
                    if full_sibling.is_file() {
                        continue;
                    }
                }
                out.push(path);
            }
        }
    }
    Ok(())
}

fn parse_json_agent_run(path: &std::path::Path) -> io::Result<Vec<CollectedEpisode>> {
    let raw = fs::read_to_string(path)?;
    let val: Value =
        serde_json::from_str(&raw).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    if let Ok(ep) = serde_json::from_value::<CollectedEpisode>(val.clone()) {
        return Ok(vec![ep]);
    }

    // Pothik / generic agent run JSON receipt
    let prompt = val["prompt"]
        .as_str()
        .or_else(|| val["user_prompt"].as_str())
        .or_else(|| val["task"].as_str())
        .or_else(|| val["goal"].as_str())
        .unwrap_or("")
        .to_string();
    let response = val["response"]
        .as_str()
        .or_else(|| val["assistant_response"].as_str())
        .or_else(|| val["output"].as_str())
        .or_else(|| val["final_output"].as_str())
        .or_else(|| val["summary"].as_str())
        .unwrap_or("")
        .to_string();

    if prompt.trim().is_empty() || response.trim().is_empty() {
        return Ok(Vec::new());
    }

    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("run");
    let run_id = val["run_id"]
        .as_str()
        .or_else(|| val["episode_id"].as_str())
        .or_else(|| val["id"].as_str())
        .unwrap_or(stem);
    let session_id = val["session_id"].as_str().unwrap_or(run_id).to_string();
    let app_id = val["agent"]
        .as_str()
        .or_else(|| val["app_id"].as_str())
        .unwrap_or("pothik")
        .to_string();
    let model = val["model"]
        .as_str()
        .or_else(|| val["model_id"].as_str())
        .unwrap_or("local-agent-model")
        .to_string();

    let mut ep = CollectedEpisode::new(
        format!("harvest-{run_id}"),
        session_id,
        app_id,
        &model,
        prompt,
        response,
    );
    ep.model_id = model;

    let tests_passed = val["unit_tests_passed"]
        .as_bool()
        .or_else(|| val["tests_passed"].as_bool())
        .or_else(|| val["verified"].as_bool())
        .or_else(|| val["exit_code"].as_i64().map(|c| c == 0));
    let git_diff = val["git_diff"]
        .as_str()
        .or_else(|| val["diff"].as_str())
        .map(str::to_string);
    ep.unit_tests_passed = tests_passed;
    ep.git_diff = git_diff;

    let status_str = val["status"]
        .as_str()
        .or_else(|| val["verdict"].as_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let has_deterministic_proof = ep.unit_tests_passed == Some(true)
        || ep
            .git_diff
            .as_ref()
            .map(|d| !d.trim().is_empty())
            .unwrap_or(false);

    if ep.unit_tests_passed == Some(false)
        || status_str == "failed"
        || status_str == "fail"
        || status_str == "error"
    {
        ep.outcome = EpisodeOutcome::FailedRetry;
        ep.scalar_reward = -1.0;
    } else if has_deterministic_proof
        && (ep.unit_tests_passed == Some(true)
            || status_str == "pass"
            || status_str == "passed"
            || status_str == "completed"
            || status_str == "success"
            || status_str.is_empty())
    {
        ep.outcome = EpisodeOutcome::VerifiedSuccess;
        ep.scalar_reward = 1.0;
    } else {
        ep.outcome = EpisodeOutcome::Unverified;
        ep.scalar_reward = 0.0;
    }

    Ok(vec![ep])
}

fn parse_jsonl_agent_run(path: &std::path::Path) -> io::Result<Vec<CollectedEpisode>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut episodes = Vec::new();
    let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("trace");

    let mut pending_agy_user: Option<String> = None;
    let mut pending_agy_step_idx: usize = 0;
    let mut last_agy_test_passed: Option<bool> = None;

    for (idx, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let Ok(val) = serde_json::from_str::<Value>(&line) else {
            continue;
        };

        // Support AGY transcript.jsonl step streams (USER_INPUT -> PLANNER_RESPONSE / TOOL_RESPONSE)
        if let Some(step_type) = val["type"].as_str() {
            match step_type {
                "USER_INPUT" => {
                    if let Some(c) = val["content"].as_str() {
                        if !c.trim().is_empty() {
                            pending_agy_user = Some(c.to_string());
                            pending_agy_step_idx =
                                val["step_index"].as_u64().unwrap_or(idx as u64) as usize;
                        }
                    }
                    continue;
                }
                "GENERIC" | "TOOL_RESPONSE" | "CORTEX_STEP_TYPE_M14_TOOL_RESPONSE" => {
                    let s = val["content"].as_str().unwrap_or(line.as_str());
                    if s.contains("test result: ok")
                        || s.contains("Command exited with code 0")
                        || s.contains("OVERALL VERDICT: PASS")
                        || s.contains("Made 1 replacement in ")
                    {
                        last_agy_test_passed = Some(true);
                    } else if s.contains("test result: FAILED")
                        || s.contains("Command exited with code 1")
                    {
                        last_agy_test_passed = Some(false);
                    }
                    continue;
                }
                "PLANNER_RESPONSE" => {
                    let content = val["content"].as_str().unwrap_or("").trim().to_string();
                    let thinking = val["thinking"].as_str().unwrap_or("").trim().to_string();
                    let tool_calls_arr = val["tool_calls"].as_array();
                    let has_tools = tool_calls_arr.map(|a| !a.is_empty()).unwrap_or(false);
                    if let Some(ref user_prompt) = pending_agy_user {
                        if !content.is_empty() || has_tools {
                            let step_no = val["step_index"].as_u64().unwrap_or(idx as u64);
                            let ep_id =
                                format!("harvest-agy-{stem}-{pending_agy_step_idx}-{step_no}");
                            let mut parsed_tools = Vec::new();
                            let mut synthesized_diffs = Vec::new();
                            if let Some(tc_list) = tool_calls_arr {
                                for (t_i, tc) in tc_list.iter().enumerate() {
                                    let name = tc["name"]
                                        .as_str()
                                        .or_else(|| tc["tool_name"].as_str())
                                        .unwrap_or("tool")
                                        .to_string();
                                    let args = if !tc["arguments"].is_null() {
                                        tc["arguments"].clone()
                                    } else {
                                        tc["args"].clone()
                                    };
                                    if name == "edit_file" {
                                        let p = args["Path"]
                                            .as_str()
                                            .unwrap_or("file")
                                            .trim_matches('"');
                                        let tgt =
                                            args["Target"].as_str().unwrap_or("").trim_matches('"');
                                        let rep = args["Replacement"]
                                            .as_str()
                                            .unwrap_or("")
                                            .trim_matches('"');
                                        if !tgt.is_empty() || !rep.is_empty() {
                                            synthesized_diffs.push(format!(
                                                "--- a/{p}\n+++ b/{p}\n@@ -1 +1 @@\n-{tgt}\n+{rep}"
                                            ));
                                        }
                                    } else if name == "write_file" {
                                        let p = args["Path"]
                                            .as_str()
                                            .unwrap_or("file")
                                            .trim_matches('"');
                                        let body = args["Content"]
                                            .as_str()
                                            .unwrap_or("")
                                            .trim_matches('"');
                                        if !body.is_empty() {
                                            synthesized_diffs.push(format!(
                                                "--- /dev/null\n+++ b/{p}\n@@ -0,0 +1 @@\n+{body}"
                                            ));
                                        }
                                    }
                                    parsed_tools.push(ToolCallRecord {
                                        id: format!("call_{step_no}_{t_i}"),
                                        name,
                                        arguments: args,
                                    });
                                }
                            }
                            let resp_text = if !content.is_empty() {
                                if !thinking.is_empty() {
                                    format!("<think>\n{thinking}\n</think>\n{content}")
                                } else {
                                    content
                                }
                            } else if !synthesized_diffs.is_empty() {
                                let diff_joined = synthesized_diffs.join("\n\n");
                                if !thinking.is_empty() {
                                    format!("<think>\n{thinking}\n</think>\n```diff\n{diff_joined}\n```")
                                } else {
                                    format!("```diff\n{diff_joined}\n```")
                                }
                            } else if !thinking.is_empty() {
                                thinking
                            } else {
                                format!("Executed {} tool call(s)", parsed_tools.len())
                            };
                            let mut ep = CollectedEpisode::new(
                                ep_id,
                                format!("{stem}-task-{pending_agy_step_idx}"),
                                "agy",
                                "agy-coding-agent",
                                user_prompt.clone(),
                                resp_text,
                            );
                            ep.tool_calls = parsed_tools;
                            if !synthesized_diffs.is_empty() {
                                ep.git_diff = Some(synthesized_diffs.join("\n\n"));
                            }
                            ep.unit_tests_passed = last_agy_test_passed;
                            if last_agy_test_passed == Some(true) || ep.git_diff.is_some() {
                                ep.outcome = EpisodeOutcome::VerifiedSuccess;
                                ep.scalar_reward = 1.0;
                            } else if last_agy_test_passed == Some(false) {
                                ep.outcome = EpisodeOutcome::FailedRetry;
                                ep.scalar_reward = -1.0;
                            }
                            episodes.push(ep);
                        }
                    }
                    continue;
                }
                _ => {}
            }
        }
        if let Ok(mut ep) = serde_json::from_value::<CollectedEpisode>(val.clone()) {
            // Enforce deterministic verification gate on harvested episodes
            if ep.outcome == EpisodeOutcome::VerifiedSuccess {
                let has_proof = ep.unit_tests_passed == Some(true)
                    || ep
                        .git_diff
                        .as_ref()
                        .map(|d| !d.trim().is_empty())
                        .unwrap_or(false);
                if !has_proof {
                    ep.outcome = EpisodeOutcome::Unverified;
                    ep.scalar_reward = 0.0;
                }
            }
            episodes.push(ep);
            continue;
        }

        // Parse messages array or Dolil turn record
        let (prompt, response, sys_prompt) = if let Some(msgs) = val["messages"].as_array() {
            let u = msgs
                .iter()
                .rev()
                .find(|m| m["role"] == "user")
                .and_then(|m| m["content"].as_str())
                .unwrap_or("")
                .to_string();
            let a = msgs
                .iter()
                .rev()
                .find(|m| m["role"] == "assistant")
                .and_then(|m| m["content"].as_str())
                .unwrap_or("")
                .to_string();
            let s = msgs
                .iter()
                .find(|m| m["role"] == "system")
                .and_then(|m| m["content"].as_str())
                .map(str::to_string);
            (u, a, s)
        } else {
            let u = val["user_prompt"]
                .as_str()
                .or_else(|| val["prompt"].as_str())
                .unwrap_or("")
                .to_string();
            let a = val["assistant_response"]
                .as_str()
                .or_else(|| val["response"].as_str())
                .unwrap_or("")
                .to_string();
            let s = val["system_prompt"].as_str().map(str::to_string);
            (u, a, s)
        };

        if prompt.trim().is_empty() || response.trim().is_empty() {
            continue;
        }

        let ep_id = val["episode_id"]
            .as_str()
            .map(str::to_string)
            .unwrap_or_else(|| format!("harvest-{stem}-{idx}"));
        let session_id = val["session_id"].as_str().unwrap_or(stem).to_string();
        let app_id = val["app_id"]
            .as_str()
            .or_else(|| val["agent"].as_str())
            .unwrap_or("dolil")
            .to_string();
        let model = val["model"]
            .as_str()
            .or_else(|| val["model_id"].as_str())
            .unwrap_or("local-agent-model")
            .to_string();

        let mut ep = CollectedEpisode::new(ep_id, session_id, app_id, &model, prompt, response);
        ep.system_prompt = sys_prompt;
        ep.model_id = model;
        ep.unit_tests_passed = val["unit_tests_passed"].as_bool();
        ep.git_diff = val["git_diff"].as_str().map(str::to_string);

        let has_proof = ep.unit_tests_passed == Some(true)
            || ep
                .git_diff
                .as_ref()
                .map(|d| !d.trim().is_empty())
                .unwrap_or(false);
        if ep.unit_tests_passed == Some(false) {
            ep.outcome = EpisodeOutcome::FailedRetry;
            ep.scalar_reward = -1.0;
        } else if has_proof {
            ep.outcome = EpisodeOutcome::VerifiedSuccess;
            ep.scalar_reward = 1.0;
        }
        episodes.push(ep);
    }

    Ok(episodes)
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

    #[test]
    fn test_extended_pii_and_credential_redaction() {
        let sample = "Token ghp_1234567890abcdefghijklmnopqrstuvwx and hf_1234567890abcdefghijklmnop and AKIAIOSFODNN7EXAMPLE at https://admin:s3cr3t@db.internal:5432/prod from /home/alice/repo by dev@example.com";
        let (clean, redacted) = redact_secrets(sample);
        assert!(redacted);
        assert!(clean.contains("[REDACTED_GITHUB_TOKEN]"));
        assert!(clean.contains("[REDACTED_HF_TOKEN]"));
        assert!(clean.contains("[REDACTED_AWS_KEY]"));
        assert!(clean.contains("https://[REDACTED_CREDENTIALS]@db.internal:5432/prod"));
        assert!(clean.contains("/home/[REDACTED_USER]/repo"));
        assert!(clean.contains("[REDACTED_EMAIL]"));
    }

    #[test]
    fn test_utility_models_and_structured_extractor() {
        let models = default_utility_models();
        assert_eq!(models.len(), 6);
        assert!(models.iter().any(|m| m.id == "gliner2.5-small-74m"));
        assert!(models.iter().any(|m| m.id == "qwen3-embedding-0.6b"));
        assert!(models.iter().any(|m| m.id == "qwen3-reranker-0.6b"));
        assert!(models.iter().any(|m| m.id == "qwen3.5-0.8b"));
        assert!(models.iter().any(|m| m.id == "nuextract-2.0-2b"));
        assert!(models.iter().any(|m| m.id == "presidio-pii-scrubber"));

        let mut ep = CollectedEpisode::new(
            "ep-meta-1",
            "sess-1",
            "pothik",
            "qwen3-coder",
            "Fix error[E0308]: mismatched types in src/parser.rs",
            "Updated src/parser.rs and tests/parser_test.rs",
        );
        ep.outcome = EpisodeOutcome::VerifiedSuccess;
        ep.unit_tests_passed = Some(true);
        ep.git_diff = Some("@@ -10,3 +10,3 @@\n-let x: u32 = \"a\";\n+let x: u32 = 42;".into());
        ep.tool_calls.push(ToolCallRecord {
            id: "c1".into(),
            name: "edit_file".into(),
            arguments: json!({"path": "src/parser.rs"}),
        });

        let meta = extract_episode_metadata(&ep);
        assert_eq!(meta.task_category, "bugfix");
        assert!(meta.detected_files.iter().any(|f| f.ends_with("parser.rs")));
        assert!(!meta.detected_errors.is_empty());
        assert_eq!(meta.detected_tools, vec!["edit_file".to_string()]);
        assert!(meta.quality_score >= 0.90);
        assert_eq!(meta.recommended_route, "sft_positive");
    }

    #[test]
    fn test_otel_genai_span_ingestion_and_session_harvester() {
        let tmp = std::env::temp_dir().join(format!("ojon-test-harvest-{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        let coll_dir = tmp.join("collector");
        let runs_dir = tmp.join("runs");
        fs::create_dir_all(&runs_dir).unwrap();

        let coll = LearnCollector::new(coll_dir).unwrap();

        // 1. OTel span ingestion
        let otel_payload = json!({
            "resourceSpans": [{
                "scopeSpans": [{
                    "spans": [{
                        "spanId": "sp-101",
                        "traceId": "tr-900",
                        "attributes": [
                            {"key": "gen_ai.system", "value": {"stringValue": "pothik"}},
                            {"key": "gen_ai.request.model", "value": {"stringValue": "qwen3-8b"}},
                            {"key": "gen_ai.prompt", "value": {"stringValue": "Write a prime check in Rust"}},
                            {"key": "gen_ai.completion", "value": {"stringValue": "pub fn is_prime(n: u64) -> bool { n > 1 }"}},
                            {"key": "ojon.unit_tests_passed", "value": {"stringValue": "true"}}
                        ]
                    }]
                }]
            }]
        });
        let count = coll.ingest_otel_genai_spans(&otel_payload).unwrap();
        assert_eq!(count, 1);

        // 2. Harvest Pothik JSON & Dolil JSONL runs
        let pothik_run = json!({
            "run_id": "run-pothik-01",
            "agent": "pothik",
            "model": "qwen3-coder",
            "prompt": "Fix off-by-one in src/lib.rs with token ghp_1234567890abcdefghijklmnopqrstuvwx",
            "response": "Fixed bounds check in src/lib.rs.",
            "unit_tests_passed": true,
            "status": "passed"
        });
        fs::write(
            runs_dir.join("run-01.json"),
            serde_json::to_string_pretty(&pothik_run).unwrap(),
        )
        .unwrap();

        // Unverified claim (claims status=passed without unit_tests_passed or git_diff)
        let unverified_run = json!({
            "run_id": "run-unverified-02",
            "agent": "pothik",
            "prompt": "Explain ownership",
            "response": "Ownership is...",
            "status": "passed"
        });
        fs::write(
            runs_dir.join("run-02.json"),
            serde_json::to_string_pretty(&unverified_run).unwrap(),
        )
        .unwrap();

        let receipt = harvest_agent_sessions(&runs_dir, &coll).unwrap();
        assert_eq!(receipt.episodes_harvested, 2);
        assert_eq!(receipt.verified_success_count, 1);
        assert_eq!(receipt.unverified_count, 1);
        assert_eq!(receipt.redacted_count, 1);

        // Re-harvesting is idempotent
        let receipt2 = harvest_agent_sessions(&runs_dir, &coll).unwrap();
        assert_eq!(receipt2.episodes_harvested, 0);
        assert_eq!(receipt2.duplicates_skipped, 2);

        let _ = fs::remove_dir_all(&tmp);
    }
}
