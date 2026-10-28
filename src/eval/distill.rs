//! Knowledge Distillation & Fine-Tuning Dataset Curation Module.
//!
//! Provides deterministic, local-first dataset inspection, drift auditing,
//! knowledge decay scoring, and sub-millisecond recipe retrieval.
//!
//! Design Constraints:
//! - Operates strictly on external paths specified by the caller.
//! - Never modifies or creates files inside the Ojon repository itself.
//! - Zero network access, zero GPU/VRAM requirement, zero runtime dependencies.
//! - Implements the 4 Pillars of Drift-Immunity:
//!     1. Evergreen vs. Volatile Knowledge Taxonomy (Exponential Half-Life)
//!     2. Filesystem Liveness Verification (Disk Ground Truth)
//!     3. Curated High-Signal Training Splits (Teacher-Student Preparation)
//!     4. Tombstone & Supersede Auditing (Anti-Zombie Propagation)

use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn bad_input(msg: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, msg.into())
}

/// Knowledge decay tiers based on fundamental volatility.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KnowledgeTier {
    /// Invariant mathematics, algorithmic blueprints, core formats (Infinite half-life).
    Evergreen,
    /// Workflows, operational conventions, tooling pipelines (90-day half-life).
    Operational,
    /// Version workarounds, volatile dependencies, library patches (30-day half-life).
    Volatile,
    /// Transient scratch notes, single-session logs (7-day half-life).
    Ephemeral,
}

impl KnowledgeTier {
    pub fn parse(s: &str) -> Self {
        let lower = s.to_ascii_lowercase();
        if lower.contains("evergreen")
            || lower.contains("math")
            || lower.contains("algorithm")
            || lower.contains("blueprint")
            || lower.contains("format")
            || lower.contains("principle")
            || lower.contains("spec")
        {
            KnowledgeTier::Evergreen
        } else if lower.contains("volatile")
            || lower.contains("patch")
            || lower.contains("workaround")
            || lower.contains("hotfix")
            || lower.contains("temp")
        {
            KnowledgeTier::Volatile
        } else if lower.contains("ephemeral")
            || lower.contains("log")
            || lower.contains("scratch")
            || lower.contains("session")
        {
            KnowledgeTier::Ephemeral
        } else {
            KnowledgeTier::Operational
        }
    }

    pub fn half_life_days(&self) -> Option<f64> {
        match self {
            KnowledgeTier::Evergreen => None,
            KnowledgeTier::Operational => Some(90.0),
            KnowledgeTier::Volatile => Some(30.0),
            KnowledgeTier::Ephemeral => Some(7.0),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            KnowledgeTier::Evergreen => "evergreen",
            KnowledgeTier::Operational => "operational",
            KnowledgeTier::Volatile => "volatile",
            KnowledgeTier::Ephemeral => "ephemeral",
        }
    }
}

/// Exact astronomical civil day count from year, month (1..=12), day (1..=31).
pub fn days_since_epoch(year: i32, month: u32, day: u32) -> i64 {
    let y = if month <= 2 { year - 1 } else { year } as i64;
    let m = if month <= 2 { month + 12 } else { month } as i64;
    let d = day as i64;
    let era = if y >= 0 { y / 400 } else { (y - 399) / 400 };
    let yoe = y - era * 400;
    let doy = (153 * (m - 3) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Parse ISO date string ("YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS...") into epoch days.
pub fn parse_iso_date_to_days(iso_str: &str) -> Option<i64> {
    let trimmed = iso_str.trim();
    if trimmed.len() < 10 {
        return None;
    }
    let y: i32 = trimmed.get(0..4)?.parse().ok()?;
    let sep1 = trimmed.get(4..5)?;
    if sep1 != "-" {
        return None;
    }
    let m: u32 = trimmed.get(5..7)?.parse().ok()?;
    let sep2 = trimmed.get(7..8)?;
    if sep2 != "-" {
        return None;
    }
    let d: u32 = trimmed.get(8..10)?.parse().ok()?;
    if m < 1 || m > 12 || d < 1 || d > 31 {
        return None;
    }
    Some(days_since_epoch(y, m, d))
}

/// Compute exponential decay factor $D(t) = 0.5^{(\Delta t / \tau)}$.
pub fn compute_decay_factor(tier: KnowledgeTier, record_date: Option<&str>, now_days: i64) -> f64 {
    let half_life = match tier.half_life_days() {
        None => return 1.0, // Evergreen knowledge never decays
        Some(hl) => hl,
    };
    let rec_days = match record_date.and_then(parse_iso_date_to_days) {
        None => return 1.0, // Default to no decay if date unparseable
        Some(d) => d,
    };
    let elapsed = (now_days - rec_days).max(0) as f64;
    0.5f64.powf(elapsed / half_life)
}

/// Extract referenced paths from text or metadata and audit existence on disk.
pub fn audit_referenced_paths(val: &Value) -> (usize, usize) {
    let mut live = 0;
    let mut dead = 0;

    fn check_path_str(p_str: &str, live: &mut usize, dead: &mut usize) {
        let clean = p_str
            .trim_matches(|c| c == '"' || c == '\'' || c == '`' || c == '<' || c == '>')
            .replace("file:///", "")
            .replace("file://", "");
        let clean = clean.split('#').next().unwrap_or(&clean);
        if clean.len() < 3 {
            return;
        }
        // Check for common file extension or separator
        let has_sep = clean.contains('/') || clean.contains('\\');
        let has_ext = clean.contains('.');
        if (has_sep || has_ext) && !clean.starts_with("http://") && !clean.starts_with("https://") {
            let p = Path::new(clean);
            if p.exists() {
                *live += 1;
            } else if p.is_absolute() {
                // Only count as dead if it's an absolute path that definitely doesn't exist
                *dead += 1;
            }
        }
    }

    // Direct path fields
    for field in &["path", "source_path", "file", "target_file", "source_file"] {
        if let Some(s) = val.get(*field).and_then(Value::as_str) {
            check_path_str(s, &mut live, &mut dead);
        }
    }

    // Direct source field if it's a file path
    if let Some(src) = val.get("source").and_then(Value::as_str) {
        let clean = src.strip_prefix("worklog:").unwrap_or(src);
        check_path_str(clean, &mut live, &mut dead);
    }

    // Helper to scan markdown text for backticked paths and FILES: lines
    let scan_text = |text: &str, live: &mut usize, dead: &mut usize| {
        for part in text.split('`') {
            let trimmed = part.trim();
            if (trimmed.contains('/') || trimmed.contains('\\'))
                && trimmed.contains('.')
                && !trimmed.contains('\n')
                && trimmed.len() < 200
            {
                check_path_str(trimmed, live, dead);
            }
        }
        for line in text.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed
                .strip_prefix("FILES:")
                .or_else(|| trimmed.strip_prefix("File:"))
            {
                for p in rest.split(',') {
                    check_path_str(p.trim(), live, dead);
                }
            }
        }
    };

    if let Some(meta) = val.get("metadata").and_then(Value::as_object) {
        for field in &["path", "source_path", "file"] {
            if let Some(s) = meta.get(*field).and_then(Value::as_str) {
                check_path_str(s, &mut live, &mut dead);
            }
        }
    }

    // Scan messages content if ChatML / OpenAI format
    if let Some(msgs) = val.get("messages").and_then(Value::as_array) {
        for msg in msgs {
            if let Some(content) = msg.get("content").and_then(Value::as_str) {
                scan_text(content, &mut live, &mut dead);
            }
        }
    }

    (live, dead)
}

/// Calculate liveness factor from live/dead reference counts.
pub fn calculate_liveness_factor(live: usize, dead: usize) -> f64 {
    if live == 0 && dead == 0 {
        1.0 // Neutral
    } else if dead == 0 {
        1.30 // +30% Live ground-truth bonus
    } else {
        // Penalty scaled by proportion of dead references
        let total = (live + dead * 3) as f64;
        (live as f64 / total).max(0.30) // Min floor 0.30 (-70% dead penalty)
    }
}

/// A parsed entry from an external dataset (supports SFT, Alpaca, ShareGPT, ChatML formats).
#[derive(Clone, Debug)]
pub struct DatasetEntry {
    pub id: String,
    pub title: String,
    pub instruction: String,
    pub input: String,
    pub output: String,
    pub tier: KnowledgeTier,
    pub date: Option<String>,
    pub is_superseded: bool,
    pub superseded_by: Option<String>,
    pub live_refs: usize,
    pub dead_refs: usize,
    pub liveness_factor: f64,
    pub decay_factor: f64,
    pub composite_score: f64,
    pub raw: Value,
}

impl DatasetEntry {
    pub fn from_json_value(raw: Value, now_days: i64) -> Option<Self> {
        let obj = raw.as_object()?;

        // Detect Tombstone / Supersede (Pillar 4)
        let is_superseded = obj
            .get("status")
            .and_then(Value::as_str)
            .map(|s| s.eq_ignore_ascii_case("superseded"))
            .unwrap_or(false)
            || obj
                .get("superseded")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            || obj.get("superseded_by").is_some();

        let superseded_by = obj
            .get("superseded_by")
            .and_then(Value::as_str)
            .map(String::from);

        let id = obj
            .get("id")
            .and_then(Value::as_str)
            .or_else(|| obj.get("source").and_then(Value::as_str))
            .or_else(|| obj.get("key").and_then(Value::as_str))
            .map(String::from)
            .unwrap_or_else(|| "unnamed".into());

        // Extract instruction / input / output (Alpaca, ShareGPT, ChatML, or generic prompt/response)
        let mut instruction = obj
            .get("instruction")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();

        let input = obj
            .get("input")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();

        let mut output = obj
            .get("output")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();

        if instruction.is_empty() {
            if let Some(p) = obj.get("prompt").and_then(Value::as_str) {
                instruction = p.to_string();
            }
        }
        if output.is_empty() {
            if let Some(r) = obj.get("response").and_then(Value::as_str) {
                output = r.to_string();
            } else if let Some(c) = obj.get("content").and_then(Value::as_str) {
                output = c.to_string();
            }
        }

        // Support ChatML / OpenAI messages format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        if instruction.is_empty() {
            if let Some(msgs) = obj.get("messages").and_then(Value::as_array) {
                for msg in msgs {
                    let role = msg.get("role").and_then(Value::as_str).unwrap_or_default();
                    let content = msg
                        .get("content")
                        .and_then(Value::as_str)
                        .unwrap_or_default();
                    if role == "user" && instruction.is_empty() {
                        instruction = content.to_string();
                    } else if (role == "assistant" || role == "model") && output.is_empty() {
                        output = content.to_string();
                    }
                }
            }
        }

        // Support ShareGPT format: {"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}
        if instruction.is_empty() {
            if let Some(convs) = obj.get("conversations").and_then(Value::as_array) {
                for msg in convs {
                    let from = msg.get("from").and_then(Value::as_str).unwrap_or_default();
                    let val = msg.get("value").and_then(Value::as_str).unwrap_or_default();
                    if from == "human" && instruction.is_empty() {
                        instruction = val.to_string();
                    } else if (from == "gpt" || from == "model") && output.is_empty() {
                        output = val.to_string();
                    }
                }
            }
        }

        let mut title = obj
            .get("title")
            .and_then(Value::as_str)
            .map(String::from)
            .unwrap_or_default();

        if title.is_empty() && !instruction.is_empty() {
            let first_line = instruction.lines().next().unwrap_or("").trim();
            if first_line.len() > 80 {
                title = format!("{}...", &first_line[..80]);
            } else {
                title = first_line.to_string();
            }
        }

        // Extract Tier (Pillar 1)
        let tier_str = obj
            .get("tier")
            .and_then(Value::as_str)
            .or_else(|| obj.get("taxonomy").and_then(Value::as_str))
            .or_else(|| obj.get("category").and_then(Value::as_str))
            .unwrap_or_else(|| {
                if !title.is_empty() {
                    &title
                } else {
                    &instruction
                }
            });
        let tier = KnowledgeTier::parse(tier_str);

        // Date & Decay
        let mut date = obj
            .get("date")
            .and_then(Value::as_str)
            .or_else(|| obj.get("timestamp").and_then(Value::as_str))
            .or_else(|| obj.get("created_at").and_then(Value::as_str))
            .map(String::from);

        if date.is_none() {
            if let Some(src) = obj.get("source").and_then(Value::as_str) {
                for part in src.split(|c| c == '_' || c == ':' || c == '/' || c == '\\') {
                    if parse_iso_date_to_days(part).is_some() {
                        date = Some(part.to_string());
                        break;
                    }
                }
            }
        }
        let decay_factor = compute_decay_factor(tier, date.as_deref(), now_days);

        // Liveness (Pillar 2)
        let (live_refs, dead_refs) = audit_referenced_paths(&raw);
        let liveness_factor = calculate_liveness_factor(live_refs, dead_refs);

        // Composite Quality Score
        let base_score = obj
            .get("quality_score")
            .and_then(Value::as_f64)
            .unwrap_or(0.85);
        let tombstone_mult = if is_superseded { 0.0 } else { 1.0 };
        let composite_score = base_score * liveness_factor * decay_factor * tombstone_mult;

        Some(DatasetEntry {
            id,
            title,
            instruction,
            input,
            output,
            tier,
            date,
            is_superseded,
            superseded_by,
            live_refs,
            dead_refs,
            liveness_factor,
            decay_factor,
            composite_score,
            raw,
        })
    }
}

/// Dataset drift audit report.
#[derive(Clone, Debug)]
pub struct DatasetAuditReport {
    pub source_path: PathBuf,
    pub total_records: usize,
    pub active_records: usize,
    pub superseded_records: usize,
    pub total_live_refs: usize,
    pub total_dead_refs: usize,
    pub evergreen_count: usize,
    pub operational_count: usize,
    pub volatile_count: usize,
    pub ephemeral_count: usize,
    pub avg_decay_factor: f64,
    pub avg_composite_score: f64,
}

impl DatasetAuditReport {
    pub fn json(&self) -> Value {
        json!({
            "schema": "ojon.dataset_audit.v1",
            "source_path": self.source_path.to_string_lossy(),
            "total_records": self.total_records,
            "active_records": self.active_records,
            "superseded_records": self.superseded_records,
            "references": {
                "live": self.total_live_refs,
                "dead": self.total_dead_refs,
                "liveness_health_pct": if self.total_live_refs + self.total_dead_refs > 0 {
                    (self.total_live_refs as f64 / (self.total_live_refs + self.total_dead_refs) as f64) * 100.0
                } else {
                    100.0
                }
            },
            "taxonomy_distribution": {
                "evergreen": self.evergreen_count,
                "operational": self.operational_count,
                "volatile": self.volatile_count,
                "ephemeral": self.ephemeral_count
            },
            "quality_metrics": {
                "avg_decay_factor": (self.avg_decay_factor * 1000.0).round() / 1000.0,
                "avg_composite_score": (self.avg_composite_score * 1000.0).round() / 1000.0
            }
        })
    }
}

/// Audit an external dataset file (.jsonl or .json) for drift, dead links, and staleness.
pub fn audit_dataset_file(path: &Path) -> io::Result<DatasetAuditReport> {
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let now_days = (now_secs / 86400) as i64;

    let entries = read_dataset_entries(path, now_days)?;
    if entries.is_empty() {
        return Err(bad_input(format!(
            "dataset is empty or contains no valid JSON objects: {}",
            path.display()
        )));
    }

    let mut active_records = 0;
    let mut superseded_records = 0;
    let mut total_live_refs = 0;
    let mut total_dead_refs = 0;
    let mut evergreen_count = 0;
    let mut operational_count = 0;
    let mut volatile_count = 0;
    let mut ephemeral_count = 0;
    let mut sum_decay = 0.0;
    let mut sum_score = 0.0;

    for e in &entries {
        if e.is_superseded {
            superseded_records += 1;
        } else {
            active_records += 1;
        }
        total_live_refs += e.live_refs;
        total_dead_refs += e.dead_refs;

        match e.tier {
            KnowledgeTier::Evergreen => evergreen_count += 1,
            KnowledgeTier::Operational => operational_count += 1,
            KnowledgeTier::Volatile => volatile_count += 1,
            KnowledgeTier::Ephemeral => ephemeral_count += 1,
        }
        sum_decay += e.decay_factor;
        sum_score += e.composite_score;
    }

    let n = entries.len() as f64;
    Ok(DatasetAuditReport {
        source_path: path.to_path_buf(),
        total_records: entries.len(),
        active_records,
        superseded_records,
        total_live_refs,
        total_dead_refs,
        evergreen_count,
        operational_count,
        volatile_count,
        ephemeral_count,
        avg_decay_factor: sum_decay / n,
        avg_composite_score: sum_score / n,
    })
}

/// Read all dataset entries from a .jsonl or .json file streaming line-by-line or parsing array.
pub fn read_dataset_entries(path: &Path, now_days: i64) -> io::Result<Vec<DatasetEntry>> {
    let meta = fs::metadata(path)?;
    if meta.len() == 0 {
        return Err(bad_input("dataset file is empty"));
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut entries = Vec::new();

    // Check if JSON Lines (.jsonl) or regular JSON (.json)
    let is_jsonl = path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|s| s.eq_ignore_ascii_case("jsonl"))
        .unwrap_or(false);

    if is_jsonl {
        for line_res in reader.lines() {
            let line = line_res?;
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            if let Ok(raw) = serde_json::from_str::<Value>(trimmed) {
                if let Some(entry) = DatasetEntry::from_json_value(raw, now_days) {
                    entries.push(entry);
                }
            }
        }
    } else {
        // Parse entire JSON buffer (array or object)
        let full_val: Value = serde_json::from_reader(reader)
            .map_err(|e| bad_input(format!("failed to parse JSON: {e}")))?;
        if let Some(arr) = full_val.as_array() {
            for item in arr {
                if let Some(entry) = DatasetEntry::from_json_value(item.clone(), now_days) {
                    entries.push(entry);
                }
            }
        } else if let Some(entry) = DatasetEntry::from_json_value(full_val, now_days) {
            entries.push(entry);
        }
    }

    Ok(entries)
}

/// Distillation execution plan.
#[derive(Clone, Debug)]
pub struct DistillPlan {
    pub source_dataset: PathBuf,
    pub output_dir: PathBuf,
    pub total_input_rows: usize,
    pub curated_rows: usize,
    pub min_score_threshold: f64,
    pub live_reference_ratio: f64,
    pub curated_split_path: PathBuf,
}

impl DistillPlan {
    pub fn json(&self) -> Value {
        json!({
            "schema": "ojon.distill_plan.v1",
            "operation": "curate_and_distill",
            "source_dataset": self.source_dataset.to_string_lossy(),
            "output_directory": self.output_dir.to_string_lossy(),
            "total_input_rows": self.total_input_rows,
            "curated_rows": self.curated_rows,
            "curated_split_file": self.curated_split_path.to_string_lossy(),
            "min_score_threshold": self.min_score_threshold,
            "live_reference_ratio": (self.live_reference_ratio * 1000.0).round() / 1000.0,
            "status": "plan_ready",
            "quality_guarantee": "pruned_superseded_and_decayed_rows"
        })
    }
}

/// Prepare a curated distillation dataset and execution plan in an external destination directory.
pub fn prepare_distill_plan(
    source_dataset: &Path,
    output_dir: &Path,
    min_score: f64,
) -> io::Result<DistillPlan> {
    if !source_dataset.is_file() {
        return Err(bad_input(format!(
            "source dataset must be an existing file: {}",
            source_dataset.display()
        )));
    }
    if !output_dir.is_absolute() {
        return Err(bad_input("output_dir must be an absolute directory path"));
    }
    if output_dir.exists() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            format!(
                "output directory already exists: {}; Ojon refuses to overwrite existing directories",
                output_dir.display()
            ),
        ));
    }

    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let now_days = (now_secs / 86400) as i64;

    let entries = read_dataset_entries(source_dataset, now_days)?;
    let total_rows = entries.len();

    let mut curated_entries = Vec::new();
    let mut live_refs = 0;
    let mut dead_refs = 0;

    for e in entries {
        if e.is_superseded {
            continue; // Filter out dead knowledge
        }
        if e.composite_score >= min_score {
            live_refs += e.live_refs;
            dead_refs += e.dead_refs;
            curated_entries.push(e);
        }
    }

    // Sort curated rows by composite score descending
    curated_entries.sort_by(|a, b| {
        b.composite_score
            .partial_cmp(&a.composite_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Create the output directory
    fs::create_dir_all(output_dir)?;

    let curated_split_path = output_dir.join("distilled_train.jsonl");
    let mut out_file = File::create(&curated_split_path)?;
    for e in &curated_entries {
        let serialized = serde_json::to_string(&e.raw).map_err(|err| bad_input(err.to_string()))?;
        writeln!(out_file, "{}", serialized)?;
    }

    let plan = DistillPlan {
        source_dataset: source_dataset.to_path_buf(),
        output_dir: output_dir.to_path_buf(),
        total_input_rows: total_rows,
        curated_rows: curated_entries.len(),
        min_score_threshold: min_score,
        live_reference_ratio: if live_refs + dead_refs > 0 {
            live_refs as f64 / (live_refs + dead_refs) as f64
        } else {
            1.0
        },
        curated_split_path,
    };

    // Write manifest
    let manifest_path = output_dir.join("distill_plan.json");
    let manifest_json =
        serde_json::to_string_pretty(&plan.json()).map_err(|err| bad_input(err.to_string()))?;
    fs::write(manifest_path, manifest_json)?;

    Ok(plan)
}

/// Result of a sub-millisecond native recipe query.
#[derive(Clone, Debug)]
pub struct RecipeMatch {
    pub id: String,
    pub title: String,
    pub tier: &'static str,
    pub match_score: f64,
    pub liveness_factor: f64,
    pub decay_factor: f64,
    pub instruction: String,
    pub summary_or_output: String,
}

impl RecipeMatch {
    pub fn json(&self) -> Value {
        json!({
            "id": self.id,
            "title": self.title,
            "tier": self.tier,
            "score": (self.match_score * 1000.0).round() / 1000.0,
            "liveness": (self.liveness_factor * 100.0).round() / 100.0,
            "decay": (self.decay_factor * 100.0).round() / 100.0,
            "instruction": self.instruction,
            "output_preview": if self.summary_or_output.len() > 300 {
                format!("{}...", &self.summary_or_output[..300])
            } else {
                self.summary_or_output.clone()
            }
        })
    }
}

/// Ultra-fast native recipe retrieval across an external dataset.
///
/// Features:
/// - Sub-millisecond execution in native compiled Rust.
/// - Keyword matching boosted by Pillars 1 (Decay) & 2 (Liveness).
/// - Excludes superseded/tombstoned entries (Pillar 4).
pub fn retrieve_recipes(
    dataset_path: &Path,
    query: &str,
    limit: usize,
    min_score: f64,
) -> io::Result<Vec<RecipeMatch>> {
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let now_days = (now_secs / 86400) as i64;

    let entries = read_dataset_entries(dataset_path, now_days)?;
    let query_lower = query.to_ascii_lowercase();
    let query_tokens: Vec<&str> = query_lower
        .split_whitespace()
        .filter(|t| t.len() >= 2)
        .collect();

    if query_tokens.is_empty() {
        return Err(bad_input(
            "query must contain at least one token (length >= 2)",
        ));
    }

    let mut matches = Vec::new();

    for e in entries {
        if e.is_superseded {
            continue; // Dropped by Pillar 4
        }

        // Search space: title, instruction, output, id
        let title_lower = e.title.to_ascii_lowercase();
        let inst_lower = e.instruction.to_ascii_lowercase();
        let out_lower = e.output.to_ascii_lowercase();
        let id_lower = e.id.to_ascii_lowercase();

        let mut token_hits = 0;
        let mut title_exact_bonus = 0.0;

        for &tok in &query_tokens {
            let hit_title = title_lower.contains(tok) || id_lower.contains(tok);
            let hit_inst = inst_lower.contains(tok);
            let hit_out = out_lower.contains(tok);

            if hit_title {
                token_hits += 1;
                title_exact_bonus += 0.5;
            } else if hit_inst {
                token_hits += 1;
                title_exact_bonus += 0.2;
            } else if hit_out {
                token_hits += 1;
            }
        }

        if token_hits > 0 {
            let token_ratio = token_hits as f64 / query_tokens.len() as f64;
            let raw_relevance = token_ratio + title_exact_bonus;
            // Weighted by Pillar 1 (time decay) and Pillar 2 (liveness)
            let final_score = raw_relevance * e.liveness_factor * e.decay_factor;

            if final_score >= min_score {
                matches.push(RecipeMatch {
                    id: e.id,
                    title: if e.title.is_empty() {
                        e.instruction.clone()
                    } else {
                        e.title
                    },
                    tier: e.tier.as_str(),
                    match_score: final_score,
                    liveness_factor: e.liveness_factor,
                    decay_factor: e.decay_factor,
                    instruction: e.instruction,
                    summary_or_output: e.output,
                });
            }
        }
    }

    // Sort descending by final score
    matches.sort_by(|a, b| {
        b.match_score
            .partial_cmp(&a.match_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    matches.truncate(limit);
    Ok(matches)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_days_since_epoch_accuracy() {
        // 1970-01-01 is day 0
        assert_eq!(days_since_epoch(1970, 1, 1), 0);
        // 2000-01-01 is day 10957
        assert_eq!(days_since_epoch(2000, 1, 1), 10957);
    }

    #[test]
    fn test_parse_iso_date() {
        assert_eq!(parse_iso_date_to_days("1970-01-01"), Some(0));
        assert_eq!(parse_iso_date_to_days("2000-01-01T12:00:00Z"), Some(10957));
        assert_eq!(parse_iso_date_to_days("invalid-date"), None);
    }

    #[test]
    fn test_knowledge_decay_half_life() {
        let now_days = days_since_epoch(2026, 10, 12);

        // Evergreen tier never decays
        let eg_decay = compute_decay_factor(KnowledgeTier::Evergreen, Some("2020-01-01"), now_days);
        assert!((eg_decay - 1.0).abs() < 1e-6);

        // Volatile tier (30-day half-life): 30 days elapsed = 0.5 decay
        let vol_decay = compute_decay_factor(KnowledgeTier::Volatile, Some("2026-09-12"), now_days);
        assert!((vol_decay - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_liveness_factor() {
        // Neutral
        assert_eq!(calculate_liveness_factor(0, 0), 1.0);
        // Live bonus
        assert_eq!(calculate_liveness_factor(3, 0), 1.30);
        // Dead penalty
        let penalty = calculate_liveness_factor(0, 2);
        assert_eq!(penalty, 0.30);
    }

    #[test]
    fn test_tombstone_detection() {
        let raw = json!({
            "id": "rec_01",
            "instruction": "Test recipe",
            "output": "Some result",
            "status": "superseded",
            "superseded_by": "rec_02"
        });
        let entry = DatasetEntry::from_json_value(raw, 0).unwrap();
        assert!(entry.is_superseded);
        assert_eq!(entry.superseded_by.as_deref(), Some("rec_02"));
        assert_eq!(entry.composite_score, 0.0);
    }
}
