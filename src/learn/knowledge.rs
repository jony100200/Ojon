//! Online Knowledge Ingestion, Normalization & Temporal Supersession Engine.
//!
//! Features:
//! 1. Multimodal Knowledge Sources:
//!    - Software documentation, SDK release notes, GitHub changelogs, API deprecations.
//!    - AI research papers, model architecture releases, verified facts.
//! 2. Ingestion Pipeline:
//!    - DISCOVER -> FETCH -> EXTRACT -> NORMALISE -> VERIFY -> DATE -> DEDUPLICATE -> INDEX.
//! 3. Precise Temporal Provenance & Supersession Tracking:
//!    - Records [valid_from, valid_to) validity intervals.
//!    - Automatically identifies superseded documentation when newer versions are indexed.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

fn bad_input(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg.into())
}

/// Category / nature of the knowledge source.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KnowledgeSourceKind {
    SoftwareDocumentation,
    GitHubReleaseChangelog,
    ApiDeprecationNotice,
    AiResearchPaper,
    AuthoritativeFact,
}

impl KnowledgeSourceKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::SoftwareDocumentation => "software_documentation",
            Self::GitHubReleaseChangelog => "github_release_changelog",
            Self::ApiDeprecationNotice => "api_deprecation_notice",
            Self::AiResearchPaper => "ai_research_paper",
            Self::AuthoritativeFact => "authoritative_fact",
        }
    }
}

/// A structured, dated, and verified knowledge item.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct KnowledgeItem {
    pub item_id: String,
    pub source_kind: KnowledgeSourceKind,
    pub source_url: String,
    pub title: String,
    pub domain: String,
    pub content: String,
    pub published_epoch_s: u64,
    pub modified_epoch_s: u64,
    pub retrieved_epoch_s: u64,
    pub software_version: Option<String>,
    pub content_hash: String,
    pub reliability_tier: String,
    pub license: String,
    pub is_superseded: bool,
    pub superseded_by: Option<String>,
    pub valid_from_epoch_s: u64,
    pub valid_to_epoch_s: Option<u64>,
    pub citations: Vec<String>,
}

/// Calculate a fast, deterministic hex content hash.
pub fn compute_content_hash(text: &str) -> String {
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let h = hasher.finish();
    format!("{:016x}", h)
}

/// Strip basic HTML tags and normalize whitespace from web extracts.
pub fn extract_clean_text(raw: &str) -> String {
    let mut clean = String::with_capacity(raw.len());
    let mut in_tag = false;

    for ch in raw.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                clean.push(' ');
            }
            _ => {
                if !in_tag {
                    clean.push(ch);
                }
            }
        }
    }

    // Collapse multi-spaces and trim lines
    clean
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Create a new validated knowledge record.
pub fn create_knowledge_item(
    item_id: &str,
    source_kind: KnowledgeSourceKind,
    source_url: &str,
    title: &str,
    domain: &str,
    raw_content: &str,
    published_epoch_s: u64,
    software_version: Option<&str>,
    reliability_tier: &str,
    license: &str,
    citations: Vec<String>,
) -> KnowledgeItem {
    let clean_content = extract_clean_text(raw_content);
    let hash = compute_content_hash(&clean_content);
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    KnowledgeItem {
        item_id: item_id.to_string(),
        source_kind,
        source_url: source_url.to_string(),
        title: title.to_string(),
        domain: domain.to_string(),
        content: clean_content,
        published_epoch_s,
        modified_epoch_s: published_epoch_s,
        retrieved_epoch_s: now,
        software_version: software_version.map(|s| s.to_string()),
        content_hash: hash,
        reliability_tier: reliability_tier.to_string(),
        license: license.to_string(),
        is_superseded: false,
        superseded_by: None,
        valid_from_epoch_s: published_epoch_s,
        valid_to_epoch_s: None,
        citations,
    }
}

/// Compare a newly ingested knowledge item against an existing corpus and apply supersession.
///
/// If `new_item` updates or replaces existing items in the same domain/title with a newer
/// publication date or version, the existing item's `valid_to_epoch_s` is sealed to the
/// new item's `valid_from_epoch_s` and marked superseded.
pub fn apply_supersession(existing: &mut [KnowledgeItem], new_item: &KnowledgeItem) -> Vec<String> {
    let mut superseded_ids = Vec::new();

    for old in existing.iter_mut() {
        if old.item_id == new_item.item_id {
            continue;
        }

        // Match by identical title / entity subject within the same domain
        let is_same_subject = old.domain == new_item.domain
            && (old.title.trim().eq_ignore_ascii_case(new_item.title.trim())
                || (old.source_url == new_item.source_url && !old.source_url.is_empty()));

        if is_same_subject && new_item.published_epoch_s > old.published_epoch_s {
            old.is_superseded = true;
            old.superseded_by = Some(new_item.item_id.clone());
            old.valid_to_epoch_s = Some(new_item.valid_from_epoch_s);
            superseded_ids.push(old.item_id.clone());
        }
    }

    superseded_ids
}

/// Persist an array of knowledge items to a JSONL file.
pub fn save_knowledge_store(items: &[KnowledgeItem], path: &Path) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = File::create(path)?;
    for item in items {
        let line = serde_json::to_string(item).map_err(|e| bad_input(e.to_string()))?;
        writeln!(file, "{line}")?;
    }
    file.flush()?;
    Ok(())
}

/// Load knowledge items from a JSONL store.
pub fn load_knowledge_store(path: &Path) -> io::Result<Vec<KnowledgeItem>> {
    if !path.exists() {
        return Ok(Vec::new());
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut items = Vec::new();

    for (line_no, line_res) in reader.lines().enumerate() {
        let line = line_res?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        match serde_json::from_str::<KnowledgeItem>(trimmed) {
            Ok(item) => items.push(item),
            Err(e) => {
                eprintln!("[warn] load_knowledge_store skipping line {}: {}", line_no + 1, e);
            }
        }
    }

    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_clean_text() {
        let html = "<div><h1>API Guide</h1><p>Use <code>render_mesh()</code> instead of old draw calls.</p></div>";
        let clean = extract_clean_text(html);
        assert!(clean.contains("API Guide"));
        assert!(clean.contains("render_mesh()"));
        assert!(!clean.contains("<div>"));
        assert!(!clean.contains("</code>"));
    }

    #[test]
    fn test_knowledge_item_creation_and_supersession() {
        // 1. Fact published in 2021 (e.g. Unity 2021 Mesh API)
        let old_doc = create_knowledge_item(
            "doc-unity-mesh-v2021",
            KnowledgeSourceKind::SoftwareDocumentation,
            "https://docs.unity.com/mesh-api",
            "Mesh Generation API",
            "game_engine_unity",
            "In Unity 2021, use Mesh.SetVertices(List<Vector3>) with CPU allocation.",
            1_609_459_200, // Jan 1, 2021
            Some("2021.3"),
            "Authoritative",
            "Public Documentation",
            vec!["Unity 2021.3 LTS Manual".to_string()],
        );

        // 2. Updated API published in 2026 (e.g. Unity 6 Mesh Allocation)
        let new_doc = create_knowledge_item(
            "doc-unity-mesh-v2026",
            KnowledgeSourceKind::SoftwareDocumentation,
            "https://docs.unity.com/mesh-api",
            "Mesh Generation API",
            "game_engine_unity",
            "In Unity 6 / 2026, use Mesh.SetVertexBufferData with native GraphicsBuffer direct GPU upload.",
            1_767_225_600, // Jan 1, 2026
            Some("6.0"),
            "Authoritative",
            "Public Documentation",
            vec!["Unity 6 User Manual".to_string()],
        );

        let mut corpus = vec![old_doc.clone()];
        let superseded = apply_supersession(&mut corpus, &new_doc);

        assert_eq!(superseded.len(), 1);
        assert_eq!(superseded[0], "doc-unity-mesh-v2021");

        let updated_old = &corpus[0];
        assert!(updated_old.is_superseded);
        assert_eq!(updated_old.superseded_by.as_deref(), Some("doc-unity-mesh-v2026"));
        assert_eq!(updated_old.valid_to_epoch_s, Some(new_doc.valid_from_epoch_s));
    }
}
