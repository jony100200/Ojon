//! Candidate Evaluation, Silent Tool-Decline Auditing & Promotion Gate.
//!
//! Features:
//! 1. Four-Way Candidate Comparison:
//!    - Baseline (unfine-tuned / teacher baseline)
//!    - Student Adapter (LoRA adapter)
//!    - Merged High-Precision (base + adapter fused)
//!    - Quantized GGUF candidates (Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q3_K_M)
//! 2. Silent Tool-Decline Verification:
//!    - Specifically detects over-quantization collapse where models hallucinate tool calls
//!      on pure conversational/conceptual prompts (requires >= 80% decline accuracy).
//! 3. Promotion Gates & Rollback Pointer:
//!    - Gate 1: Size reduction >= 15% (or configured threshold).
//!    - Gate 2: Must-pass task score drop <= 1.0 point vs merged high-precision model.
//!    - Gate 3: Silent tool-decline accuracy >= 80%.
//!    - Maintains an immutable rollback pointer on disk.

use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn bad_input(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg.into())
}

/// Type of candidate model under evaluation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CandidateType {
    Baseline,
    Adapter,
    MergedFp16,
    QuantizedGguf,
}

impl CandidateType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::Adapter => "adapter",
            Self::MergedFp16 => "merged_fp16",
            Self::QuantizedGguf => "quantized_gguf",
        }
    }
}

/// Evaluation record for an individual candidate model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateModelEvaluation {
    pub candidate_id: String,
    pub candidate_type: CandidateType,
    pub file_path: Option<PathBuf>,
    pub size_mb: f64,
    pub size_reduction_pct: f64,
    pub bpw: Option<f64>,
    pub must_pass_task_score: f64,
    pub must_pass_retention_pct: f64,
    pub tool_decline_accuracy: f64,
    pub avg_tokens_per_sec: f64,
    pub passed_gates: bool,
}

/// Explicit test prompt to evaluate silent tool-decline behavior.
#[derive(Clone, Debug)]
pub struct ToolDeclineTestPrompt {
    pub prompt_id: &'static str,
    pub prompt: &'static str,
    pub description: &'static str,
}

/// Standard canonical suite of tool-decline test cases.
pub const CANONICAL_TOOL_DECLINE_SUITE: &[ToolDeclineTestPrompt] = &[
    ToolDeclineTestPrompt {
        prompt_id: "decline-01-rust-concept",
        prompt: "What is the difference between Arc and Rc in Rust?",
        description: "Conceptual question; must not trigger file/shell inspection tools.",
    },
    ToolDeclineTestPrompt {
        prompt_id: "decline-02-pure-algo",
        prompt: "Write a pure function in Rust that computes the factorial of an integer.",
        description: "Pure algorithmic code generation; requires no file system reading.",
    },
    ToolDeclineTestPrompt {
        prompt_id: "decline-03-api-design",
        prompt: "Explain what makes an API endpoint idempotent and provide examples of HTTP methods.",
        description: "General software engineering definition.",
    },
    ToolDeclineTestPrompt {
        prompt_id: "decline-04-creative-story",
        prompt: "Write a short creative story about a compiler optimizing away dead code.",
        description: "Creative writing task; zero tool dependencies.",
    },
    ToolDeclineTestPrompt {
        prompt_id: "decline-05-general-qa",
        prompt: "Who was Alan Turing and what was the significance of the Bombe machine?",
        description: "Historical QA prompt.",
    },
];

/// Gate check individual verdict.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GateCheckResult {
    pub gate_name: String,
    pub passed: bool,
    pub target: String,
    pub observed: String,
    pub reason: String,
}

/// Promotion policy thresholds.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromotionPolicy {
    /// Minimum size reduction percentage compared to the base/merged model (e.g. 15.0%).
    pub min_size_reduction_pct: f64,
    /// Maximum allowed drop in must-pass task score vs the merged high-precision model (e.g. 1.0).
    pub max_must_pass_score_drop: f64,
    /// Minimum required tool-decline accuracy percentage (e.g. 80.0%).
    pub min_tool_decline_accuracy: f64,
}

impl Default for PromotionPolicy {
    fn default() -> Self {
        Self {
            min_size_reduction_pct: 15.0,
            max_must_pass_score_drop: 1.0,
            min_tool_decline_accuracy: 80.0,
        }
    }
}

/// Final promotion decision with full gate audit trail.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromotionDecision {
    pub schema: String,
    pub decision: String, // "promoted" or "rejected"
    pub candidate_id: String,
    pub rollback_pointer: Option<String>,
    pub previous_promoted_candidate: Option<String>,
    pub gates: Vec<GateCheckResult>,
    pub timestamp_epoch_s: u64,
}

/// Persistent pointer record on disk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromotedModelPointer {
    pub schema: String,
    pub active_model_id: String,
    pub active_model_path: Option<String>,
    pub candidate_type: String,
    pub size_mb: f64,
    pub must_pass_task_score: f64,
    pub tool_decline_accuracy: f64,
    pub promoted_at_epoch_s: u64,
    pub rollback_model_id: Option<String>,
    pub rollback_model_path: Option<String>,
}

/// Evaluate candidate models and check if a target candidate passes all promotion gates.
pub fn evaluate_and_promote(
    candidates: &[CandidateModelEvaluation],
    target_candidate_id: &str,
    policy: &PromotionPolicy,
    pointer_path: &Path,
) -> io::Result<PromotionDecision> {
    let target = candidates
        .iter()
        .find(|c| c.candidate_id == target_candidate_id)
        .ok_or_else(|| bad_input(format!("Candidate '{}' not found in evaluation set", target_candidate_id)))?;

    // Find merged model or baseline for comparison
    let reference_score = candidates
        .iter()
        .find(|c| c.candidate_type == CandidateType::MergedFp16)
        .map(|c| c.must_pass_task_score)
        .or_else(|| {
            candidates
                .iter()
                .find(|c| c.candidate_type == CandidateType::Baseline)
                .map(|c| c.must_pass_task_score)
        })
        .unwrap_or(target.must_pass_task_score);

    let mut gates = Vec::new();

    // Gate 1: Size reduction
    let size_passed = target.size_reduction_pct >= policy.min_size_reduction_pct;
    gates.push(GateCheckResult {
        gate_name: "Size Reduction Gate".to_string(),
        passed: size_passed,
        target: format!(">= {:.1}%", policy.min_size_reduction_pct),
        observed: format!("{:.1}%", target.size_reduction_pct),
        reason: if size_passed {
            "Sufficient compression achieved".to_string()
        } else {
            "Model size does not meet minimum compression threshold".to_string()
        },
    });

    // Gate 2: Must-pass score retention
    let score_drop = (reference_score - target.must_pass_task_score).max(0.0);
    let score_passed = score_drop <= policy.max_must_pass_score_drop;
    gates.push(GateCheckResult {
        gate_name: "Must-Pass Capability Retention Gate".to_string(),
        passed: score_passed,
        target: format!("Drop <= {:.1} pts (from reference {:.1})", policy.max_must_pass_score_drop, reference_score),
        observed: format!("{:.1} pts (drop {:.1} pts)", target.must_pass_task_score, score_drop),
        reason: if score_passed {
            "High capability retention on held-out tasks".to_string()
        } else {
            "Excessive degradation on must-pass verification tasks".to_string()
        },
    });

    // Gate 3: Silent Tool-Decline Accuracy
    let decline_passed = target.tool_decline_accuracy >= policy.min_tool_decline_accuracy;
    gates.push(GateCheckResult {
        gate_name: "Silent Tool-Decline Gate".to_string(),
        passed: decline_passed,
        target: format!(">= {:.1}%", policy.min_tool_decline_accuracy),
        observed: format!("{:.1}%", target.tool_decline_accuracy),
        reason: if decline_passed {
            "Model correctly refrains from invoking tools on non-tool queries".to_string()
        } else {
            "Model suffers from silent tool-decline collapse (hallucinating unnecessary tools)".to_string()
        },
    });

    let all_passed = gates.iter().all(|g| g.passed);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // Read previous pointer if it exists
    let previous_pointer: Option<PromotedModelPointer> = if pointer_path.exists() {
        fs::read_to_string(pointer_path)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
    } else {
        None
    };

    let previous_promoted_id = previous_pointer.as_ref().map(|p| p.active_model_id.clone());
    let rollback_pointer = previous_pointer.as_ref().map(|p| p.active_model_id.clone());

    if all_passed {
        // Update persistent pointer on disk
        if let Some(parent) = pointer_path.parent() {
            fs::create_dir_all(parent)?;
        }

        let new_pointer = PromotedModelPointer {
            schema: "ojon.learn.promoted_pointer.v1".to_string(),
            active_model_id: target.candidate_id.clone(),
            active_model_path: target.file_path.as_ref().map(|p| p.to_string_lossy().to_string()),
            candidate_type: target.candidate_type.as_str().to_string(),
            size_mb: target.size_mb,
            must_pass_task_score: target.must_pass_task_score,
            tool_decline_accuracy: target.tool_decline_accuracy,
            promoted_at_epoch_s: now,
            rollback_model_id: previous_promoted_id.clone(),
            rollback_model_path: previous_pointer.and_then(|p| p.active_model_path),
        };

        let file = File::create(pointer_path)?;
        serde_json::to_writer_pretty(file, &new_pointer)?;

        Ok(PromotionDecision {
            schema: "ojon.learn.promotion_decision.v1".to_string(),
            decision: "promoted".to_string(),
            candidate_id: target.candidate_id.clone(),
            rollback_pointer,
            previous_promoted_candidate: previous_promoted_id,
            gates,
            timestamp_epoch_s: now,
        })
    } else {
        Ok(PromotionDecision {
            schema: "ojon.learn.promotion_decision.v1".to_string(),
            decision: "rejected".to_string(),
            candidate_id: target.candidate_id.clone(),
            rollback_pointer,
            previous_promoted_candidate: previous_promoted_id,
            gates,
            timestamp_epoch_s: now,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_promotion_gates_accepted() {
        let temp_dir = std::env::temp_dir().join(format!(
            "ojon_eval_test_{}",
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        let pointer_file = temp_dir.join("promoted_model_pointer.json");

        let candidates = vec![
            CandidateModelEvaluation {
                candidate_id: "base-fp16".to_string(),
                candidate_type: CandidateType::Baseline,
                file_path: None,
                size_mb: 8000.0,
                size_reduction_pct: 0.0,
                bpw: Some(16.0),
                must_pass_task_score: 82.0,
                must_pass_retention_pct: 100.0,
                tool_decline_accuracy: 90.0,
                avg_tokens_per_sec: 25.0,
                passed_gates: true,
            },
            CandidateModelEvaluation {
                candidate_id: "student-fused".to_string(),
                candidate_type: CandidateType::MergedFp16,
                file_path: None,
                size_mb: 8000.0,
                size_reduction_pct: 0.0,
                bpw: Some(16.0),
                must_pass_task_score: 88.5,
                must_pass_retention_pct: 100.0,
                tool_decline_accuracy: 92.0,
                avg_tokens_per_sec: 25.0,
                passed_gates: true,
            },
            CandidateModelEvaluation {
                candidate_id: "student-q4_k_m".to_string(),
                candidate_type: CandidateType::QuantizedGguf,
                file_path: Some(PathBuf::from("/models/student-q4_k_m.gguf")),
                size_mb: 2600.0,
                size_reduction_pct: 67.5,
                bpw: Some(4.5),
                must_pass_task_score: 88.0,
                must_pass_retention_pct: 99.4,
                tool_decline_accuracy: 88.0,
                avg_tokens_per_sec: 78.0,
                passed_gates: true,
            },
        ];

        let policy = PromotionPolicy::default();
        let decision = evaluate_and_promote(&candidates, "student-q4_k_m", &policy, &pointer_file)
            .expect("evaluation should run cleanly");

        assert_eq!(decision.decision, "promoted");
        assert!(pointer_file.exists());

        let _ = fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_promotion_gates_rejected_on_tool_decline_collapse() {
        let temp_dir = std::env::temp_dir().join(format!(
            "ojon_eval_fail_test_{}",
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        let pointer_file = temp_dir.join("promoted_model_pointer.json");

        let candidates = vec![
            CandidateModelEvaluation {
                candidate_id: "student-fused".to_string(),
                candidate_type: CandidateType::MergedFp16,
                file_path: None,
                size_mb: 8000.0,
                size_reduction_pct: 0.0,
                bpw: Some(16.0),
                must_pass_task_score: 88.5,
                must_pass_retention_pct: 100.0,
                tool_decline_accuracy: 92.0,
                avg_tokens_per_sec: 25.0,
                passed_gates: true,
            },
            CandidateModelEvaluation {
                candidate_id: "student-q2_k".to_string(),
                candidate_type: CandidateType::QuantizedGguf,
                file_path: Some(PathBuf::from("/models/student-q2_k.gguf")),
                size_mb: 1800.0,
                size_reduction_pct: 77.5,
                bpw: Some(2.6),
                must_pass_task_score: 85.0,
                must_pass_retention_pct: 96.0,
                tool_decline_accuracy: 40.0, // Fails: Silent tool-decline collapse!
                avg_tokens_per_sec: 95.0,
                passed_gates: false,
            },
        ];

        let policy = PromotionPolicy::default();
        let decision = evaluate_and_promote(&candidates, "student-q2_k", &policy, &pointer_file)
            .expect("evaluation should run cleanly");

        assert_eq!(decision.decision, "rejected");
        // Gate for tool decline should have failed
        let decline_gate = decision.gates.iter().find(|g| g.gate_name.contains("Tool-Decline")).unwrap();
        assert!(!decline_gate.passed);
        assert!(!pointer_file.exists()); // Was not created because rejection occurred

        let _ = fs::remove_dir_all(&temp_dir);
    }
}
