//! Data Curation, Dual-Buffer Partitioning & Training Export Pipeline.
//!
//! Implements:
//! 1. Dual-Buffer Partitioning (Anti-Catastrophic Forgetting & Anti-Catastrophic Remembering):
//!    - Static Foundational Anchor Buffer (5-10% of batch containing immutable syntax, math,
//!      and explicit tool-decline demonstrations).
//!    - Recency Sliding Window (K=3 generations) discarding obsolete workspace states.
//! 2. DPO / Preference Pairing (pairs verified successes against failed retries).
//! 3. Standard LLaMA-Factory / ShareGPT export generation (`dataset_info.json`, `sft_chat.jsonl`, `dpo_pairs.jsonl`).

use crate::learn::collector::{CollectedEpisode, EpisodeOutcome, ToolCallRecord};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};



/// Configuration options for the Dual-Buffer split.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DualBufferConfig {
    /// Fraction of training data reserved for static foundational anchors (e.g., 0.08 = 8%).
    pub anchor_ratio: f64,
    /// Maximum number of active workspace generations / sessions in the recency window (K=3).
    pub recency_window_k: usize,
    /// Minimum confidence / scalar reward for inclusion in SFT training.
    pub min_sft_reward: f64,
    /// Ensure explicit tool-decline demonstrations are included in the anchor set.
    pub include_tool_decline_anchors: bool,
}

impl Default for DualBufferConfig {
    fn default() -> Self {
        Self {
            anchor_ratio: 0.08,
            recency_window_k: 3,
            min_sft_reward: 0.5,
            include_tool_decline_anchors: true,
        }
    }
}

/// A curated batch divided cleanly into static anchors and recency window.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CuratedBatch {
    pub total_source_episodes: usize,
    pub anchor_episodes: Vec<CollectedEpisode>,
    pub recency_episodes: Vec<CollectedEpisode>,
    pub dpo_pairs: Vec<DpoPreferencePair>,
}

/// A pairwise preference sample for Direct Preference Optimization (DPO).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DpoPreferencePair {
    pub pair_id: String,
    pub session_id: String,
    pub system_prompt: Option<String>,
    pub user_prompt: String,
    pub chosen_response: String,
    pub rejected_response: String,
    pub chosen_tool_calls: Vec<ToolCallRecord>,
    pub rejected_tool_calls: Vec<ToolCallRecord>,
    pub reward_delta: f64,
}

/// Summary report returned after curating and exporting datasets.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CurationReceipt {
    pub total_input_episodes: usize,
    pub anchor_count: usize,
    pub recency_count: usize,
    pub sft_sample_count: usize,
    pub dpo_pair_count: usize,
    pub export_dir: PathBuf,
    pub sft_file: PathBuf,
    pub dpo_file: PathBuf,
    pub dataset_info_file: PathBuf,
}

/// Synthetic / immutable foundational tool-decline anchor episodes.
/// Teaches the student model when *not* to invoke tools (solving silent tool-decline collapse).
pub fn generate_tool_decline_anchors() -> Vec<CollectedEpisode> {
    vec![
        CollectedEpisode {
            schema: "ojon.learn.v1".to_string(),
            episode_id: "anchor-decline-01".to_string(),
            session_id: "static-anchor-session".to_string(),
            app_id: "anchor".to_string(),
            timestamp_epoch_s: 1_700_000_000,
            claimed_teacher: "curated_ground_truth".to_string(),
            model_id: "anchor".to_string(),
            system_prompt: Some("You are an expert programming assistant with access to file and shell tools.".to_string()),
            user_prompt: "What is the difference between an Arc and an Rc in Rust?".to_string(),
            assistant_response: "In Rust, `Rc<T>` is a reference-counting pointer for single-threaded use. It does not implement `Send` or `Sync`, making reference increments and decrements fast via non-atomic operations. In contrast, `Arc<T>` stands for Atomically Reference Counted and uses atomic operations, making it thread-safe to share across multiple threads at the cost of slight atomic synchronization overhead.".to_string(),
            tool_calls: Vec::new(),
            tool_results: None,
            outcome: EpisodeOutcome::VerifiedSuccess,
            scalar_reward: 1.0,
            git_diff: None,
            unit_tests_passed: Some(true),
            is_redacted: false,
        },
        CollectedEpisode {
            schema: "ojon.learn.v1".to_string(),
            episode_id: "anchor-decline-02".to_string(),
            session_id: "static-anchor-session".to_string(),
            app_id: "anchor".to_string(),
            timestamp_epoch_s: 1_700_000_001,
            claimed_teacher: "curated_ground_truth".to_string(),
            model_id: "anchor".to_string(),
            system_prompt: Some("You are an autonomous coding assistant.".to_string()),
            user_prompt: "Write a pure function in Rust that computes the Fibonacci number iteratively.".to_string(),
            assistant_response: "```rust\npub fn fibonacci(n: u32) -> u64 {\n    match n {\n        0 => 0,\n        1 => 1,\n        _ => {\n            let mut a = 0u64;\n            let mut b = 1u64;\n            for _ in 2..=n {\n                let next = a.saturating_add(b);\n                a = b;\n                b = next;\n            }\n            b\n        }\n    }\n}\n```".to_string(),
            tool_calls: Vec::new(),
            tool_results: None,
            outcome: EpisodeOutcome::VerifiedSuccess,
            scalar_reward: 1.0,
            git_diff: None,
            unit_tests_passed: Some(true),
            is_redacted: false,
        },
        CollectedEpisode {
            schema: "ojon.learn.v1".to_string(),
            episode_id: "anchor-decline-03".to_string(),
            session_id: "static-anchor-session".to_string(),
            app_id: "anchor".to_string(),
            timestamp_epoch_s: 1_700_000_002,
            claimed_teacher: "curated_ground_truth".to_string(),
            model_id: "anchor".to_string(),
            system_prompt: Some("You have file search and edit tools.".to_string()),
            user_prompt: "Hello! Who are you and how can you help me today?".to_string(),
            assistant_response: "Hello! I am your AI coding assistant. I can help you inspect code, debug issues, write tests, optimize models, and perform repository refactorings. Let me know what task you'd like to work on!".to_string(),
            tool_calls: Vec::new(),
            tool_results: None,
            outcome: EpisodeOutcome::VerifiedSuccess,
            scalar_reward: 1.0,
            git_diff: None,
            unit_tests_passed: Some(true),
            is_redacted: false,
        },
    ]
}

/// Verify that an episode claiming VerifiedSuccess has deterministic verification evidence
/// (passing unit tests, non-empty verified git diff, or canonical static anchor).
pub fn has_deterministic_verification(ep: &CollectedEpisode) -> bool {
    if ep.unit_tests_passed == Some(false) {
        return false;
    }
    ep.unit_tests_passed == Some(true)
        || ep.git_diff.as_ref().is_some_and(|d| !d.trim().is_empty())
        || ep.app_id == "anchor"
        || ep.session_id == "static-anchor-session"
}

/// Partition an episode collection using the Dual-Buffer algorithm.
pub fn partition_dual_buffer(
    episodes: &[CollectedEpisode],
    config: &DualBufferConfig,
) -> CuratedBatch {
    let mut verified_episodes: Vec<CollectedEpisode> = Vec::new();
    let mut session_map: HashMap<String, Vec<CollectedEpisode>> = HashMap::new();

    for raw_ep in episodes {
        let mut ep = raw_ep.clone();
        if ep.outcome == EpisodeOutcome::VerifiedSuccess && !has_deterministic_verification(&ep) {
            if ep.unit_tests_passed == Some(false) {
                ep.outcome = EpisodeOutcome::FailedRetry;
                ep.scalar_reward = -1.0;
            } else {
                ep.outcome = EpisodeOutcome::Unverified;
                ep.scalar_reward = 0.0;
            }
        }
        session_map
            .entry(ep.session_id.clone())
            .or_default()
            .push(ep.clone());

        if ep.outcome == EpisodeOutcome::VerifiedSuccess && ep.scalar_reward >= config.min_sft_reward {
            verified_episodes.push(ep);
        }
    }

    // Sort sessions chronologically by the max timestamp of their episodes
    let mut session_timestamps: Vec<(String, u64)> = session_map
        .iter()
        .map(|(sid, list)| {
            let max_ts = list.iter().map(|e| e.timestamp_epoch_s).max().unwrap_or(0);
            (sid.clone(), max_ts)
        })
        .collect();
    session_timestamps.sort_by_key(|(_, ts)| *ts);

    // Identify the K most recent sessions for the Recency Window
    let recent_session_ids: HashSet<String> = if session_timestamps.len() > config.recency_window_k {
        let skip_count = session_timestamps.len() - config.recency_window_k;
        session_timestamps[skip_count..]
            .iter()
            .map(|(sid, _)| sid.clone())
            .collect()
    } else {
        session_timestamps.iter().map(|(sid, _)| sid.clone()).collect()
    };

    // Separate into anchor and recency candidates
    let mut anchor_episodes: Vec<CollectedEpisode> = Vec::new();
    let mut recency_episodes: Vec<CollectedEpisode> = Vec::new();

    if config.include_tool_decline_anchors {
        anchor_episodes.extend(generate_tool_decline_anchors());
    }

    for ep in verified_episodes {
        if ep.app_id == "anchor" || ep.session_id == "static-anchor-session" {
            anchor_episodes.push(ep);
        } else if recent_session_ids.contains(&ep.session_id) {
            recency_episodes.push(ep);
        } else {
            // Older session: if anchor ratio target is not yet met, keep as anchor candidate
            let target_anchor_count = ((recency_episodes.len() + 1) as f64 * config.anchor_ratio).ceil() as usize;
            if anchor_episodes.len() < target_anchor_count {
                anchor_episodes.push(ep);
            }
        }
    }

    // Pairwise Preference / DPO Extraction
    let mut dpo_pairs: Vec<DpoPreferencePair> = Vec::new();
    for (session_id, list) in &session_map {
        let successes: Vec<&CollectedEpisode> = list
            .iter()
            .filter(|e| e.outcome == EpisodeOutcome::VerifiedSuccess)
            .collect();
        let retries: Vec<&CollectedEpisode> = list
            .iter()
            .filter(|e| e.outcome == EpisodeOutcome::FailedRetry)
            .collect();

        for succ in &successes {
            for ret in &retries {
                // If they share a similar user prompt or are within the same session
                if succ.user_prompt.trim() == ret.user_prompt.trim()
                    || (!succ.assistant_response.is_empty() && !ret.assistant_response.is_empty())
                {
                    dpo_pairs.push(DpoPreferencePair {
                        pair_id: format!("pair-{}-{}", succ.episode_id, ret.episode_id),
                        session_id: session_id.clone(),
                        system_prompt: succ.system_prompt.clone().or_else(|| ret.system_prompt.clone()),
                        user_prompt: succ.user_prompt.clone(),
                        chosen_response: succ.assistant_response.clone(),
                        rejected_response: ret.assistant_response.clone(),
                        chosen_tool_calls: succ.tool_calls.clone(),
                        rejected_tool_calls: ret.tool_calls.clone(),
                        reward_delta: succ.scalar_reward - ret.scalar_reward,
                    });
                }
            }
        }
    }

    CuratedBatch {
        total_source_episodes: episodes.len(),
        anchor_episodes,
        recency_episodes,
        dpo_pairs,
    }
}

/// Convert an episode into a standardized OpenAI / LLaMA-Factory messages list.
fn episode_to_messages(ep: &CollectedEpisode) -> Value {
    let mut messages = Vec::new();

    if let Some(sys) = &ep.system_prompt {
        if !sys.trim().is_empty() {
            messages.push(json!({
                "role": "system",
                "content": sys
            }));
        }
    }

    messages.push(json!({
        "role": "user",
        "content": ep.user_prompt
    }));

    if !ep.tool_calls.is_empty() {
        let calls_json: Vec<Value> = ep
            .tool_calls
            .iter()
            .map(|tc| {
                json!({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments.to_string()
                    }
                })
            })
            .collect();

        messages.push(json!({
            "role": "assistant",
            "content": ep.assistant_response,
            "tool_calls": calls_json
        }));
        if let Some(results) = &ep.tool_results {
            for (call, result) in ep.tool_calls.iter().zip(results) {
                if result.is_null() { continue; }
                let content = result.as_str().map(str::to_owned)
                    .unwrap_or_else(|| result.to_string());
                messages.push(json!({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": content
                }));
            }
        }
    } else {
        messages.push(json!({
            "role": "assistant",
            "content": ep.assistant_response
        }));
    }

    json!({ "messages": messages })
}

/// Export a curated batch to a standard LLaMA-Factory compatible directory.
pub fn export_to_llamafactory(
    batch: &CuratedBatch,
    output_dir: &Path,
) -> io::Result<CurationReceipt> {
    fs::create_dir_all(output_dir)?;

    let sft_path = output_dir.join("sft_chat.jsonl");
    let sft_train_path = output_dir.join("sft_train.jsonl");
    let sft_eval_path = output_dir.join("sft_eval.jsonl");
    let dpo_path = output_dir.join("dpo_pairs.jsonl");
    let info_path = output_dir.join("dataset_info.json");

    // Determine session-disjoint held-out eval session(s) from non-anchor sessions
    let mut non_anchor_sessions: Vec<String> = batch
        .recency_episodes
        .iter()
        .chain(batch.anchor_episodes.iter())
        .filter(|ep| ep.app_id != "anchor" && ep.session_id != "static-anchor-session")
        .map(|ep| ep.session_id.clone())
        .collect();
    non_anchor_sessions.sort();
    non_anchor_sessions.dedup();
    let eval_session_set: HashSet<String> = if non_anchor_sessions.len() >= 2 {
        let eval_count = (non_anchor_sessions.len() / 5).max(1);
        non_anchor_sessions[non_anchor_sessions.len() - eval_count..]
            .iter()
            .cloned()
            .collect()
    } else {
        HashSet::new()
    };

    // 1. Write SFT Dataset + Session-Disjoint Train/Eval Splits
    let mut sft_file = File::create(&sft_path)?;
    let mut sft_train_file = File::create(&sft_train_path)?;
    let mut sft_eval_file = File::create(&sft_eval_path)?;
    let mut sft_count = 0;

    // Anchor first, then recency
    for ep in batch.anchor_episodes.iter().chain(batch.recency_episodes.iter()) {
        let msg_obj = episode_to_messages(ep);
        writeln!(sft_file, "{}", msg_obj)?;
        if eval_session_set.contains(&ep.session_id) {
            writeln!(sft_eval_file, "{}", msg_obj)?;
        } else {
            writeln!(sft_train_file, "{}", msg_obj)?;
        }
        sft_count += 1;
    }
    sft_file.flush()?;
    sft_train_file.flush()?;
    sft_eval_file.flush()?;

    // 2. Write DPO Dataset
    let mut dpo_file = File::create(&dpo_path)?;
    let mut dpo_count = 0;

    for pair in &batch.dpo_pairs {
        let mut convs = Vec::new();
        if let Some(sys) = &pair.system_prompt {
            convs.push(json!({ "role": "system", "content": sys }));
        }
        convs.push(json!({ "role": "user", "content": pair.user_prompt }));

        let dpo_obj = json!({
            "conversations": convs,
            "chosen": {
                "role": "assistant",
                "content": pair.chosen_response
            },
            "rejected": {
                "role": "assistant",
                "content": pair.rejected_response
            }
        });
        writeln!(dpo_file, "{}", dpo_obj)?;
        dpo_count += 1;
    }
    dpo_file.flush()?;

    // 3. Write dataset_info.json manifest for LLaMA-Factory
    let info_manifest = json!({
        "ojon_sft": {
            "file_name": "sft_chat.jsonl",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages"
            }
        },
        "ojon_sft_train": {
            "file_name": "sft_train.jsonl",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages"
            }
        },
        "ojon_sft_eval": {
            "file_name": "sft_eval.jsonl",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages"
            }
        },
        "ojon_dpo": {
            "file_name": "dpo_pairs.jsonl",
            "ranking": true,
            "formatting": "sharegpt",
            "columns": {
                "prompt": "conversations",
                "chosen": "chosen",
                "rejected": "rejected"
            }
        }
    });

    let mut info_file = File::create(&info_path)?;
    writeln!(info_file, "{}", serde_json::to_string_pretty(&info_manifest)?)?;
    info_file.flush()?;

    Ok(CurationReceipt {
        total_input_episodes: batch.total_source_episodes,
        anchor_count: batch.anchor_episodes.len(),
        recency_count: batch.recency_episodes.len(),
        sft_sample_count: sft_count,
        dpo_pair_count: dpo_count,
        export_dir: output_dir.to_path_buf(),
        sft_file: sft_path,
        dpo_file: dpo_path,
        dataset_info_file: info_path,
    })
}

/// Read all collected episodes from a JSONL file.
pub fn load_episodes_from_jsonl(path: &Path) -> io::Result<Vec<CollectedEpisode>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut out = Vec::new();

    for (line_idx, line_res) in reader.lines().enumerate() {
        let line = line_res?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<CollectedEpisode>(trimmed) {
            Ok(ep) => out.push(ep),
            Err(e) => {
                eprintln!("[warn] load_episodes_from_jsonl: skipping line {}: {}", line_idx + 1, e);
            }
        }
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_ep(id: &str, session: &str, outcome: EpisodeOutcome, ts: u64, is_anchor: bool) -> CollectedEpisode {
        CollectedEpisode {
            schema: "ojon.learn.v1".to_string(),
            episode_id: id.to_string(),
            session_id: session.to_string(),
            app_id: if is_anchor { "anchor".to_string() } else { "test-app".to_string() },
            timestamp_epoch_s: ts,
            claimed_teacher: "claude-3-5-sonnet".to_string(),
            model_id: "claude-3-5-sonnet".to_string(),
            system_prompt: Some("System prompt test".to_string()),
            user_prompt: format!("User prompt for {}", id),
            assistant_response: format!("Assistant answer for {}", id),
            tool_calls: Vec::new(),
            tool_results: None,
            outcome,
            scalar_reward: outcome.default_reward(),
            git_diff: None,
            unit_tests_passed: Some(outcome == EpisodeOutcome::VerifiedSuccess),
            is_redacted: false,
        }
    }

    #[test]
    fn test_dual_buffer_partitioning() {
        let mut episodes = Vec::new();

        // Older session 1 (failed)
        episodes.push(dummy_ep("ep-01", "session-1", EpisodeOutcome::FailedRetry, 100, false));
        // Older session 2 (verified)
        episodes.push(dummy_ep("ep-02", "session-2", EpisodeOutcome::VerifiedSuccess, 200, false));
        // Recent session 3 (verified)
        episodes.push(dummy_ep("ep-03", "session-3", EpisodeOutcome::VerifiedSuccess, 300, false));
        // Recent session 4 (verified + failed retry)
        episodes.push(dummy_ep("ep-04-fail", "session-4", EpisodeOutcome::FailedRetry, 400, false));
        episodes.push(dummy_ep("ep-04-succ", "session-4", EpisodeOutcome::VerifiedSuccess, 401, false));
        // Recent session 5 (verified)
        episodes.push(dummy_ep("ep-05", "session-5", EpisodeOutcome::VerifiedSuccess, 500, false));

        let config = DualBufferConfig {
            anchor_ratio: 0.10,
            recency_window_k: 3, // sessions 3, 4, 5
            min_sft_reward: 0.5,
            include_tool_decline_anchors: true,
        };

        let batch = partition_dual_buffer(&episodes, &config);

        // Tool decline anchors (3) should be present
        assert!(batch.anchor_episodes.len() >= 3);
        // Recency episodes should include verified from sessions 3, 4, 5
        assert_eq!(batch.recency_episodes.len(), 3);
        // DPO pairs from session 4
        assert_eq!(batch.dpo_pairs.len(), 1);
        assert_eq!(batch.dpo_pairs[0].chosen_response, "Assistant answer for ep-04-succ");
        assert_eq!(batch.dpo_pairs[0].rejected_response, "Assistant answer for ep-04-fail");
        assert_eq!(batch.dpo_pairs[0].reward_delta, 2.0); // 1.0 - (-1.0)
    }

    #[test]
    fn test_llamafactory_export() {
        let mut episodes = Vec::new();
        episodes.push(dummy_ep("ep-succ", "session-dpo", EpisodeOutcome::VerifiedSuccess, 10, false));
        episodes.push(dummy_ep("ep-fail", "session-dpo", EpisodeOutcome::FailedRetry, 11, false));

        let config = DualBufferConfig::default();
        let batch = partition_dual_buffer(&episodes, &config);

        let temp_dir = std::env::temp_dir().join(format!("ojon_curate_test_{}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let receipt = export_to_llamafactory(&batch, &temp_dir).expect("export should succeed");

        assert!(receipt.sft_file.exists());
        assert!(receipt.dpo_file.exists());
        assert!(receipt.dataset_info_file.exists());

        // Validate dataset_info.json is valid JSON
        let info_str = fs::read_to_string(&receipt.dataset_info_file).unwrap();
        let info_val: Value = serde_json::from_str(&info_str).unwrap();
        assert!(info_val.get("ojon_sft").is_some());
        assert!(info_val.get("ojon_dpo").is_some());

        // Cleanup
        let _ = fs::remove_dir_all(&temp_dir);
    }
}
