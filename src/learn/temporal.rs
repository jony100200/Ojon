//! Time-Aware Knowledge Engine, Temporal Retrieval & Augmented Prompt Synthesis.
//!
//! Implements Chronos / ERASE / FreshQA patterns:
//! 1. Temporal Interval Indexing: tracks [valid_from, valid_to) for every stored fact.
//! 2. Present-Day Queries: prioritizes currently active, authoritative evidence; suppresses superseded facts.
//! 3. Historical Queries: resolves facts valid as of an explicit target timestamp (e.g. "Unity in 2021" vs "Unity in 2026").
//! 4. Verifiable Citations & Honest Weight Disclaimers:
//!    - Strictly distinguishes retrieved current knowledge from underlying model weights.
//!    - Never falsely claims a model's intrinsic training cutoff has moved.

use crate::learn::knowledge::KnowledgeItem;
use serde::{Deserialize, Serialize};
use std::io;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

/// Query parameters for temporal retrieval.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalRetrievalQuery {
    pub query_text: String,
    /// Target date to evaluate fact validity (None = Present day / latest valid).
    pub as_of_epoch_s: Option<u64>,
    pub domain_filter: Option<String>,
    pub limit: usize,
}

/// A retrieved fact snippet scored with temporal relevance and citations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalSearchResult {
    pub item_id: String,
    pub title: String,
    pub source_url: String,
    pub snippet: String,
    pub domain: String,
    pub valid_from_epoch_s: u64,
    pub valid_to_epoch_s: Option<u64>,
    pub is_currently_valid: bool,
    pub software_version: Option<String>,
    pub reliability_tier: String,
    pub relevance_score: f64,
    pub citations: Vec<String>,
}

/// Synthesized response augmenting a user prompt with verified temporal citations.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AugmentedPromptResponse {
    pub original_prompt: String,
    pub augmented_prompt: String,
    pub retrieved_evidence: Vec<TemporalSearchResult>,
    pub has_current_knowledge: bool,
    pub target_date_iso: String,
    pub disclaimer: String,
}

/// Time-Aware Knowledge Retrieval Engine.
pub struct TemporalKnowledgeEngine {
    items: Vec<KnowledgeItem>,
}

impl TemporalKnowledgeEngine {
    pub fn new(items: Vec<KnowledgeItem>) -> Self {
        Self { items }
    }

    pub fn load_from_store(store_path: &Path) -> io::Result<Self> {
        let items = crate::learn::knowledge::load_knowledge_store(store_path)?;
        Ok(Self::new(items))
    }

    pub fn total_items(&self) -> usize {
        self.items.len()
    }

    /// Query the knowledge base with explicit temporal constraints.
    pub fn search(&self, query: &TemporalRetrievalQuery) -> Vec<TemporalSearchResult> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let target_time = query.as_of_epoch_s.unwrap_or(now);
        let query_lower = query.query_text.to_ascii_lowercase();
        let query_words: Vec<&str> = query_lower
            .split_whitespace()
            .filter(|w| w.len() >= 3)
            .collect();

        let mut scored_results: Vec<(f64, TemporalSearchResult)> = Vec::new();

        for item in &self.items {
            // Optional domain filtering
            if let Some(df) = &query.domain_filter {
                if !item.domain.eq_ignore_ascii_case(df) {
                    continue;
                }
            }

            // 1. Temporal Validity Check:
            // The item must have been published on or before target_time.
            if item.valid_from_epoch_s > target_time {
                // Item is from the future relative to target_time
                continue;
            }

            // If an expiration / supersession date exists, target_time must be strictly before it.
            let is_valid_at_target = match item.valid_to_epoch_s {
                Some(to) => target_time < to,
                None => true,
            };

            if !is_valid_at_target && query.as_of_epoch_s.is_none() {
                // For present-day queries, reject superseded / expired facts completely.
                continue;
            }

            // 2. Keyword & Title Relevance Scoring:
            let title_lower = item.title.to_ascii_lowercase();
            let content_lower = item.content.to_ascii_lowercase();

            let mut keyword_score = 0.0f64;
            for word in &query_words {
                if title_lower.contains(word) {
                    keyword_score += 3.0;
                }
                if content_lower.contains(word) {
                    keyword_score += 1.0;
                }
            }

            if keyword_score <= 0.0 {
                continue;
            }

            // Temporal ranking boost:
            let mut final_score = keyword_score;

            if is_valid_at_target {
                final_score += 5.0; // Boost currently active facts
            } else {
                final_score -= 2.0; // Demote expired historical facts
            }

            // Reliability tier boost
            if item.reliability_tier == "Authoritative" {
                final_score += 2.0;
            }

            let is_current = item.valid_to_epoch_s.is_none() && !item.is_superseded;

            // Generate representative snippet
            let snippet = if item.content.len() > 300 {
                format!("{}...", &item.content[..297])
            } else {
                item.content.clone()
            };

            scored_results.push((
                final_score,
                TemporalSearchResult {
                    item_id: item.item_id.clone(),
                    title: item.title.clone(),
                    source_url: item.source_url.clone(),
                    snippet,
                    domain: item.domain.clone(),
                    valid_from_epoch_s: item.valid_from_epoch_s,
                    valid_to_epoch_s: item.valid_to_epoch_s,
                    is_currently_valid: is_current,
                    software_version: item.software_version.clone(),
                    reliability_tier: item.reliability_tier.clone(),
                    relevance_score: final_score,
                    citations: item.citations.clone(),
                },
            ));
        }

        // Sort descending by relevance score
        scored_results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        scored_results
            .into_iter()
            .take(query.limit)
            .map(|(_, res)| res)
            .collect()
    }

    /// Construct an augmented prompt injecting retrieved temporal knowledge.
    /// Strictly states that underlying model weights remain untouched.
    pub fn build_augmented_prompt(
        &self,
        base_prompt: &str,
        as_of_epoch_s: Option<u64>,
        limit: usize,
    ) -> AugmentedPromptResponse {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let query_time = as_of_epoch_s.unwrap_or(now);

        let query = TemporalRetrievalQuery {
            query_text: base_prompt.to_string(),
            as_of_epoch_s,
            domain_filter: None,
            limit,
        };

        let results = self.search(&query);
        let has_knowledge = !results.is_empty();

        let date_iso = format_epoch_iso(query_time);

        let disclaimer = if has_knowledge {
            format!(
                "[Ojon Retrieved Knowledge as of {} | Model Weights Unmodified | Verifiable Citations Attached]",
                date_iso
            )
        } else {
            "[Native Model Knowledge - No external retrieval found]".to_string()
        };

        let augmented_prompt = if has_knowledge {
            let mut context_block = String::new();
            context_block.push_str("### RETRIEVED VERIFIED KNOWLEDGE (Time-Aware Engine)\n");
            context_block.push_str(&format!("Evaluation Date: {}\n\n", date_iso));

            for (idx, item) in results.iter().enumerate() {
                let (safe_snippet, _) =
                    crate::learn::knowledge::sanitize_untrusted_knowledge(&item.snippet);
                context_block.push_str(&format!(
                    "[{}] \"{}\" (Source: {}, Version: {}) [EXTERNAL_REFERENCE_DATA_ONLY]\n",
                    idx + 1,
                    item.title,
                    item.source_url,
                    item.software_version.as_deref().unwrap_or("N/A")
                ));
                context_block.push_str(&format!("Evidence: {}\n", safe_snippet));
                if !item.citations.is_empty() {
                    context_block.push_str(&format!("Citations: {}\n", item.citations.join("; ")));
                }
                context_block.push('\n');
            }

            format!("{}\n### USER INSTRUCTION:\n{}", context_block, base_prompt)
        } else {
            base_prompt.to_string()
        };

        AugmentedPromptResponse {
            original_prompt: base_prompt.to_string(),
            augmented_prompt,
            retrieved_evidence: results,
            has_current_knowledge: has_knowledge,
            target_date_iso: date_iso,
            disclaimer,
        }
    }
}

/// Convert seconds since epoch into a compact YYYY-MM-DD string.
pub fn format_epoch_iso(epoch_s: u64) -> String {
    // Basic calendar approximation for zero-dependency portability
    let days = epoch_s / 86400;
    let mut year = 1970i64;
    let mut d = days as i64;

    loop {
        let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        let days_in_year = if leap { 366 } else { 365 };
        if d < days_in_year {
            break;
        }
        d -= days_in_year;
        year += 1;
    }

    let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let month_days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];

    let mut month = 1;
    for &md in &month_days {
        if d < md {
            break;
        }
        d -= md;
        month += 1;
    }
    let day = d + 1;

    format!("{:04}-{:02}-{:02}", year, month, day)
}

/// Parse a YYYY-MM-DD or YYYY/MM/DD date string into epoch seconds (UTC midnight).
pub fn parse_iso_date_to_epoch(date_str: &str) -> Option<u64> {
    let clean = date_str.trim().replace('/', "-");
    let parts: Vec<&str> = clean.split('-').collect();
    if parts.len() != 3 {
        return None;
    }

    let year: i64 = parts[0].parse().ok()?;
    let month: usize = parts[1].parse().ok()?;
    let day: usize = parts[2].parse().ok()?;

    if year < 1970 || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return None;
    }

    let mut total_days: i64 = 0;
    for y in 1970..year {
        let leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
        total_days += if leap { 366 } else { 365 };
    }

    let is_leap_year = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let month_days = [
        31,
        if is_leap_year { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];

    for m in 1..month {
        total_days += month_days[m - 1] as i64;
    }

    total_days += (day - 1) as i64;
    if total_days < 0 {
        return None;
    }

    Some((total_days as u64) * 86400)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::learn::knowledge::{
        apply_supersession, create_knowledge_item, KnowledgeItemInput, KnowledgeSourceKind,
    };

    #[test]
    fn test_temporal_retrieval_distinguishes_present_vs_past() {
        // 1. Fact valid from 2021 to 2026 (old Mesh API)
        let old_doc = create_knowledge_item(KnowledgeItemInput {
            item_id: "doc-unity-mesh-2021",
            source_kind: KnowledgeSourceKind::SoftwareDocumentation,
            source_url: "https://docs.unity.com/2021/mesh",
            title: "Unity Mesh Generation",
            domain: "unity",
            raw_content: "In Unity 2021, use Mesh.SetVertices(List<Vector3>) with CPU allocation.",
            published_epoch_s: 1_609_459_200, // 2021-01-01
            software_version: Some("2021.3"),
            reliability_tier: "Authoritative",
            license: "Public",
            citations: vec!["Unity 2021 Manual".to_string()],
        });

        // 2. Fact valid from 2026 onwards (Unity 6 GPU direct buffer)
        let new_doc = create_knowledge_item(
            KnowledgeItemInput {
                item_id: "doc-unity-mesh-2026",
                source_kind: KnowledgeSourceKind::SoftwareDocumentation,
                source_url: "https://docs.unity.com/6.0/mesh",
                title: "Unity Mesh Generation",
                domain: "unity",
                raw_content: "In Unity 6 (2026), use Mesh.SetVertexBufferData with native GraphicsBuffer direct GPU upload.",
                published_epoch_s: 1_767_225_600, // 2026-01-01
                software_version: Some("6.0"),
                reliability_tier: "Authoritative",
                license: "Public",
                citations: vec!["Unity 6 Manual".to_string()],
            },
        );

        let mut corpus = vec![old_doc, new_doc.clone()];
        let _ = apply_supersession(&mut corpus, &new_doc);

        let engine = TemporalKnowledgeEngine::new(corpus);

        // Case A: Present-day query (after 2026).
        // Must return the 2026 documentation and suppress the superseded 2021 documentation.
        let query_present = TemporalRetrievalQuery {
            query_text: "Unity Mesh Generation".to_string(),
            as_of_epoch_s: Some(1_790_000_000), // Mid 2026
            domain_filter: None,
            limit: 5,
        };
        let res_present = engine.search(&query_present);
        assert!(!res_present.is_empty());
        assert_eq!(res_present[0].item_id, "doc-unity-mesh-2026");
        assert!(res_present[0]
            .snippet
            .contains("GraphicsBuffer direct GPU upload"));

        // Case B: Historical query (as of 2023).
        // Must return the 2021 documentation (which was valid in 2023) and NOT the 2026 API.
        let query_past = TemporalRetrievalQuery {
            query_text: "Unity Mesh Generation".to_string(),
            as_of_epoch_s: Some(1_672_531_200), // 2023-01-01
            domain_filter: None,
            limit: 5,
        };
        let res_past = engine.search(&query_past);
        assert!(!res_past.is_empty());
        assert_eq!(res_past[0].item_id, "doc-unity-mesh-2021");
        assert!(res_past[0].snippet.contains("Mesh.SetVertices"));
    }

    #[test]
    fn test_augmented_prompt_synthesis_and_disclaimer() {
        let doc = create_knowledge_item(KnowledgeItemInput {
            item_id: "doc-rust-2026",
            source_kind: KnowledgeSourceKind::SoftwareDocumentation,
            source_url: "https://doc.rust-lang.org/edition-2024",
            title: "Rust 2024 Edition",
            domain: "rust",
            raw_content: "Rust 2024 stabilizes gen blocks and async closures.",
            published_epoch_s: 1_735_689_600, // 2025-01-01
            software_version: Some("1.85"),
            reliability_tier: "Authoritative",
            license: "MIT/Apache",
            citations: vec!["The Rust Edition Guide".to_string()],
        });

        let engine = TemporalKnowledgeEngine::new(vec![doc]);
        let aug =
            engine.build_augmented_prompt("How do async closures work in Rust 2024?", None, 3);

        assert!(aug.has_current_knowledge);
        assert!(aug
            .augmented_prompt
            .contains("RETRIEVED VERIFIED KNOWLEDGE"));
        assert!(aug.augmented_prompt.contains("Rust 2024 Edition"));
        assert!(aug.disclaimer.contains("Model Weights Unmodified"));
    }

    #[test]
    fn test_iso_date_roundtrip() {
        let epoch = 1_767_225_600; // 2026-01-01
        let formatted = format_epoch_iso(epoch);
        assert_eq!(formatted, "2026-01-01");
        let parsed = parse_iso_date_to_epoch(&formatted).expect("must parse");
        assert_eq!(parsed, epoch);

        let parsed_slash = parse_iso_date_to_epoch("2026/01/01").expect("must parse slash");
        assert_eq!(parsed_slash, epoch);
    }
}
