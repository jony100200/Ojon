//! Curated Hugging Face & Community Distillation Dataset Registry.
//!
//! Provides an immutable catalog of high-priority verifiable SWE/agent trajectories,
//! authentic developer traces, and quarantine handling for unverified dumps.
//!
//! Rules from Roadmap & Papers:
//! 1. The student's held-out task score takes precedence over dataset brand names.
//! 2. Unverified / claimed-teacher dumps are quarantined and flagged with provenance confidence.
//! 3. Bounded streaming import with deduplication and PII scrubbing.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;

/// Category / domain of the distillation dataset.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetCategory {
    SweCodingAgent,
    RealWorldDeveloperTrace,
    GeneralReasoningAndPlanning,
    QuarantinedCommunityDump,
}

impl DatasetCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SweCodingAgent => "swe_coding_agent",
            Self::RealWorldDeveloperTrace => "real_world_developer_trace",
            Self::GeneralReasoningAndPlanning => "general_reasoning_and_planning",
            Self::QuarantinedCommunityDump => "quarantined_community_dump",
        }
    }
}

/// Metadata entry for a curated public distillation dataset.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatasetCatalogEntry {
    pub id: &'static str,
    pub name: &'static str,
    pub hf_repo: &'static str,
    pub category: DatasetCategory,
    pub claimed_teacher: &'static str,
    pub documented_generator: &'static str,
    pub estimated_rows: u64,
    pub has_executable_tests: bool,
    pub is_quarantined: bool,
    pub confidence_tier: &'static str,
    pub description: &'static str,
    pub recommended_student_scale: &'static str,
}

/// The official curated catalog defined in Ojon Learn Roadmap v2026-10-27.
pub const CURATED_DATASETS: &[DatasetCatalogEntry] = &[
    // 1. Highest-Priority Verifiable SWE/Coding-Agent Data
    DatasetCatalogEntry {
        id: "si2ca-trajectories",
        name: "SI2CA Training Trajectories",
        hf_repo: "Self-Improving-Coding-Agents/SI2CA-Training-Trajectories",
        category: DatasetCategory::SweCodingAgent,
        claimed_teacher: "122B Teacher Model",
        documented_generator: "SI2CA Coupled Verification Harness",
        estimated_rows: 32340,
        has_executable_tests: true,
        is_quarantined: false,
        confidence_tier: "High",
        description: "32,340 teacher trajectories on 10,780 software tasks; SFT yields 59.6 SWE-bench Verified.",
        recommended_student_scale: "3B-35B",
    },
    DatasetCatalogEntry {
        id: "open-swe-traces",
        name: "NVIDIA Open-SWE-Traces",
        hf_repo: "nvidia/Open-SWE-Traces",
        category: DatasetCategory::SweCodingAgent,
        claimed_teacher: "Frontier SWE Models",
        documented_generator: "NVIDIA Multi-Harness Agent Fleet",
        estimated_rows: 150000,
        has_executable_tests: true,
        is_quarantined: false,
        confidence_tier: "High",
        description: "Large-scale agent trajectories with reported test resolution and environment traces.",
        recommended_student_scale: "4B-14B",
    },
    DatasetCatalogEntry {
        id: "nemotron-swe-v3",
        name: "NVIDIA Nemotron SFT SWE v3",
        hf_repo: "nvidia/Nemotron-SFT-SWE-v3",
        category: DatasetCategory::SweCodingAgent,
        claimed_teacher: "Nemotron / Llama-3 70B",
        documented_generator: "NVIDIA Repo Navigation & Debugging Pipeline",
        estimated_rows: 85000,
        has_executable_tests: true,
        is_quarantined: false,
        confidence_tier: "High",
        description: "Repository navigation, bug localization, and multi-file code editing trajectories.",
        recommended_student_scale: "4B-14B",
    },
    DatasetCatalogEntry {
        id: "swe-smith",
        name: "SWE-smith Trajectories",
        hf_repo: "SWE-bench/SWE-smith-trajectories",
        category: DatasetCategory::SweCodingAgent,
        claimed_teacher: "Synthetic Execution Agents",
        documented_generator: "SWE-bench Verified Framework",
        estimated_rows: 45000,
        has_executable_tests: true,
        is_quarantined: false,
        confidence_tier: "High",
        description: "Execution-based synthetic bug fixing designed for automated grading and verification.",
        recommended_student_scale: "1B-8B",
    },
    DatasetCatalogEntry {
        id: "swe-hero",
        name: "SWE-Hero OpenHands Trajectories",
        hf_repo: "nvidia/SWE-Hero-openhands-trajectories",
        category: DatasetCategory::SweCodingAgent,
        claimed_teacher: "Claude 3.5 Sonnet / OpenHands",
        documented_generator: "OpenHands Event-Stream Runtime",
        estimated_rows: 28000,
        has_executable_tests: true,
        is_quarantined: false,
        confidence_tier: "High",
        description: "Multi-turn software engineering trajectories with exact tool calls and git patch diffs.",
        recommended_student_scale: "3B-14B",
    },
    // 2. Authentic-World GPT/Claude/Codex Traces
    DatasetCatalogEntry {
        id: "swe-chat",
        name: "SWE-chat Developer Sessions",
        hf_repo: "SALT-NLP/SWE-chat",
        category: DatasetCategory::RealWorldDeveloperTrace,
        claimed_teacher: "Commercial AI Coding Agents",
        documented_generator: "Real-world developer coding interaction logs",
        estimated_rows: 17968,
        has_executable_tests: false,
        is_quarantined: false,
        confidence_tier: "Medium",
        description: "~17,968 real-world developer coding sessions and 229,909 user turns with tool interactions.",
        recommended_student_scale: "4B-8B",
    },
    DatasetCatalogEntry {
        id: "share-codex",
        name: "Share-Codex Trajectories",
        hf_repo: "nmuendler/share-codex",
        category: DatasetCategory::RealWorldDeveloperTrace,
        claimed_teacher: "Codex & Claude Code",
        documented_generator: "Exported CLI & IDE tool traces",
        estimated_rows: 42000,
        has_executable_tests: false,
        is_quarantined: false,
        confidence_tier: "Medium",
        description: "Public Codex and Claude Code trajectories with multi-step bash and editor tool calls.",
        recommended_student_scale: "1B-8B",
    },
    DatasetCatalogEntry {
        id: "trace-commons",
        name: "Trace Commons Agent Traces",
        hf_repo: "trace-commons/agent-traces",
        category: DatasetCategory::RealWorldDeveloperTrace,
        claimed_teacher: "Claude Code / Cursor / Codex",
        documented_generator: "Community donated developer interaction logs",
        estimated_rows: 64000,
        has_executable_tests: false,
        is_quarantined: false,
        confidence_tier: "Medium",
        description: "Real-world developer traces covering tool structure, self-correction, and long horizons.",
        recommended_student_scale: "3B-14B",
    },
    // 3. Complementary General & Reasoning Data
    DatasetCatalogEntry {
        id: "openthoughts3",
        name: "OpenThoughts 3 Reasoning",
        hf_repo: "open-thoughts/OpenThoughts3-1.2M",
        category: DatasetCategory::GeneralReasoningAndPlanning,
        claimed_teacher: "DeepSeek-R1 / QwQ",
        documented_generator: "OpenThoughts multi-step CoT generator",
        estimated_rows: 1200000,
        has_executable_tests: true,
        is_quarantined: false,
        confidence_tier: "High",
        description: "Long-horizon deliberation, mathematical derivation, and algorithmic reasoning traces.",
        recommended_student_scale: "4B-14B",
    },
    DatasetCatalogEntry {
        id: "nemotron-agentic-v2",
        name: "Nemotron SFT Agentic v2",
        hf_repo: "nvidia/Nemotron-SFT-Agentic-v2",
        category: DatasetCategory::GeneralReasoningAndPlanning,
        claimed_teacher: "Llama-3.1 70B / 405B",
        documented_generator: "NVIDIA Synthetic Planning Engine",
        estimated_rows: 50000,
        has_executable_tests: true,
        is_quarantined: false,
        confidence_tier: "High",
        description: "Tool calling, schema adherence, and multi-turn planning examples.",
        recommended_student_scale: "1B-8B",
    },
    // 4. Quarantined Community Dumps
    DatasetCatalogEntry {
        id: "fable-5-traces",
        name: "Complete FABLE.5 Traces",
        hf_repo: "Crownelius/Complete-FABLE.5-traces-2M",
        category: DatasetCategory::QuarantinedCommunityDump,
        claimed_teacher: "FABLE.5 / GPT-5 class",
        documented_generator: "Unauthenticated community web scraper",
        estimated_rows: 2000000,
        has_executable_tests: false,
        is_quarantined: true,
        confidence_tier: "Low",
        description: "QUARANTINED: Unauthenticated teacher claims; requires PII and benchmark-leakage audit.",
        recommended_student_scale: "Quarantined",
    },
    DatasetCatalogEntry {
        id: "fable-gpt-5-5-distill",
        name: "Fable GPT-5.5 Distillation Traces",
        hf_repo: "RESMP-DEV/Fable-GPT-5.5-Distillation-Traces",
        category: DatasetCategory::QuarantinedCommunityDump,
        claimed_teacher: "GPT-5.5 (Claimed)",
        documented_generator: "Synthetic prompt expansion",
        estimated_rows: 500000,
        has_executable_tests: false,
        is_quarantined: true,
        confidence_tier: "Low",
        description: "QUARANTINED: Uploader titles do not prove underlying model identity.",
        recommended_student_scale: "Quarantined",
    },
];

/// Retrieve catalog entries by optional category filter.
pub fn list_curated_datasets(
    category: Option<DatasetCategory>,
) -> Vec<&'static DatasetCatalogEntry> {
    CURATED_DATASETS
        .iter()
        .filter(|e| category.map_or(true, |c| e.category == c))
        .collect()
}

/// Find catalog entry by exact ID.
pub fn get_dataset_by_id(id: &str) -> Option<&'static DatasetCatalogEntry> {
    CURATED_DATASETS.iter().find(|e| e.id == id)
}

/// Summary receipt from streaming import.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ImportReceipt {
    pub schema: String,
    pub source_path: String,
    pub output_path: String,
    pub total_input_rows: u64,
    pub accepted_rows: u64,
    pub rejected_syntax_errors: u64,
    pub rejected_duplicates: u64,
    pub redacted_secrets_count: u64,
    pub total_tokens_estimate: u64,
}

/// Stream a local JSONL dataset file, validate basic conversation structure,
/// scrub secrets, deduplicate identical queries, and write out clean records.
pub fn import_and_sanitize_jsonl(
    source_path: &Path,
    output_path: &Path,
    max_records: Option<usize>,
) -> io::Result<ImportReceipt> {
    if !source_path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            "source dataset file missing",
        ));
    }
    let source_file = File::open(source_path)?;
    let reader = BufReader::new(source_file);

    let mut out_file = File::create(output_path)?;
    let mut seen_hashes = HashSet::new();

    let mut total_rows = 0u64;
    let mut accepted = 0u64;
    let mut rejected_syntax = 0u64;
    let mut rejected_dups = 0u64;
    let mut redacted_count = 0u64;
    let mut token_estimate = 0u64;

    for line_res in reader.lines() {
        let line = line_res?;
        if line.trim().is_empty() {
            continue;
        }
        total_rows += 1;

        if let Some(max) = max_records {
            if accepted >= max as u64 {
                break;
            }
        }

        let parsed: Result<Value, _> = serde_json::from_str(&line);
        let mut row = match parsed {
            Ok(v) => v,
            Err(_) => {
                rejected_syntax += 1;
                continue;
            }
        };

        // Extract prompt / user text for deduplication hash
        let prompt_text: String = row
            .get("user_prompt")
            .or_else(|| row.get("instruction"))
            .or_else(|| row.get("prompt"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .or_else(|| {
                // Standard chat "messages": find user message content
                row.get("messages")
                    .and_then(|m| m.as_array())
                    .and_then(|arr| {
                        arr.iter()
                            .find(|msg| msg.get("role").and_then(|r| r.as_str()) == Some("user"))
                            .and_then(|msg| msg.get("content").and_then(|c| c.as_str()))
                            .map(|s| s.to_string())
                    })
            })
            .or_else(|| {
                // ShareGPT "conversations": find human/user message
                row.get("conversations")
                    .and_then(|c| c.as_array())
                    .and_then(|arr| {
                        arr.iter()
                            .find(|msg| {
                                let from = msg
                                    .get("from")
                                    .or_else(|| msg.get("role"))
                                    .and_then(|r| r.as_str());
                                matches!(from, Some("human") | Some("user"))
                            })
                            .and_then(|msg| {
                                msg.get("value")
                                    .or_else(|| msg.get("content"))
                                    .and_then(|c| c.as_str())
                            })
                            .map(|s| s.to_string())
                    })
            })
            .unwrap_or_default();

        let prompt_hash = {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            if !prompt_text.is_empty() {
                std::hash::Hash::hash(&prompt_text, &mut hasher);
            } else {
                std::hash::Hash::hash(&line, &mut hasher);
            }
            std::hash::Hasher::finish(&hasher)
        };

        if !seen_hashes.insert(prompt_hash) {
            rejected_dups += 1;
            continue;
        }

        // Scrub credentials
        let did_redact = crate::learn::collector::redact_json_secrets(&mut row);
        if did_redact {
            redacted_count += 1;
        }

        let row_str = serde_json::to_string(&row).map_err(io::Error::other)?;
        writeln!(out_file, "{row_str}")?;
        accepted += 1;
        token_estimate += (row_str.len() / 4) as u64;
    }
    out_file.sync_all()?;

    Ok(ImportReceipt {
        schema: "ojon.learn.import_receipt.v1".to_string(),
        source_path: source_path.to_string_lossy().to_string(),
        output_path: output_path.to_string_lossy().to_string(),
        total_input_rows: total_rows,
        accepted_rows: accepted,
        rejected_syntax_errors: rejected_syntax,
        rejected_duplicates: rejected_dups,
        redacted_secrets_count: redacted_count,
        total_tokens_estimate: token_estimate,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_curated_registry_entries() {
        let entries = list_curated_datasets(None);
        assert!(entries.len() >= 10);

        let si2ca = get_dataset_by_id("si2ca-trajectories").unwrap();
        assert_eq!(si2ca.category, DatasetCategory::SweCodingAgent);
        assert!(si2ca.has_executable_tests);
        assert!(!si2ca.is_quarantined);

        let fable = get_dataset_by_id("fable-5-traces").unwrap();
        assert!(fable.is_quarantined);
        assert_eq!(fable.confidence_tier, "Low");
    }

    #[test]
    fn test_import_and_sanitize_jsonl() {
        let tmp = std::env::temp_dir().join(format!("ojon-test-reg-{}", std::process::id()));
        fs::create_dir_all(&tmp).unwrap();
        let src = tmp.join("raw_input.jsonl");
        let dst = tmp.join("clean_output.jsonl");

        let raw_lines = vec![
            r#"{"instruction": "Calculate 2+2", "response": "4 with key sk-abcdef1234567890abcdef"}"#,
            r#"{"instruction": "Calculate 2+2", "response": "Duplicate"}"#, // duplicate
            r#"{"broken json"#,                                             // syntax error
            r#"{"instruction": "Write a python func", "response": "def foo(): pass"}"#,
        ];
        fs::write(&src, raw_lines.join("\n")).unwrap();

        let receipt = import_and_sanitize_jsonl(&src, &dst, None).unwrap();
        assert_eq!(receipt.total_input_rows, 4);
        assert_eq!(receipt.accepted_rows, 2);
        assert_eq!(receipt.rejected_duplicates, 1);
        assert_eq!(receipt.rejected_syntax_errors, 1);
        assert_eq!(receipt.redacted_secrets_count, 1);

        let clean_content = fs::read_to_string(&dst).unwrap();
        assert!(clean_content.contains("[REDACTED_API_KEY]"));
        assert!(!clean_content.contains("sk-abcdef1234567890"));

        let _ = fs::remove_dir_all(tmp);
    }
}
