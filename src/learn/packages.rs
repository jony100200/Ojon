//! Versioned Model Update Packages & Atomic Rollback Manager.
//!
//! Implements Software-Update Parity for AI Models:
//! 1. Three Concrete Package Tiers:
//!    - `KnowledgePack`: Verified facts, temporal search index, source citations (zero weight changes).
//!    - `SkillPack`: Fine-tuning LoRA adapter, tool-use behaviors, compatibility signature, benchmark delta.
//!    - `ModelRelease`: Merged checkpoint / Ojon-quantized GGUF, hardware specs, runtime verification.
//! 2. Integrity Verification:
//!    - SHA-256 / content hash verification before installation.
//! 3. Atomic Installation & Rollback:
//!    - Automatically archives the previous active release.
//!    - Allows instant, one-click rollback if runtime verification or loading fails.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::fs::{self, File};
use std::hash::{Hash, Hasher};
use std::io;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn bad_input(msg: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, msg.into())
}

/// Category of the update package.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PackageKind {
    /// Incremental verified online facts and citations for retrieval.
    KnowledgePack,
    /// Targeted LoRA / QLoRA behavioral adapter.
    SkillPack,
    /// Full merged checkpoint or quantized GGUF release.
    ModelRelease,
}

impl PackageKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::KnowledgePack => "knowledge_pack",
            Self::SkillPack => "skill_pack",
            Self::ModelRelease => "model_release",
        }
    }
}

/// Metadata manifest for a versioned Ojon update package.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PackageManifest {
    pub schema: String,
    pub package_id: String,
    pub name: String,
    pub version: String,
    pub kind: PackageKind,
    pub base_model_compatibility: Option<String>,
    pub created_at_epoch_s: u64,
    pub content_hash: String,
    pub size_bytes: u64,
    pub description: String,
    pub author: String,
    pub license: String,
    pub changelog: Vec<String>,
    pub benchmark_scores: HashMap<String, f64>,
}

/// Record of an installed package on disk.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstallReceipt {
    pub schema: String,
    pub status: String,
    pub package_id: String,
    pub version: String,
    pub kind: PackageKind,
    pub active_path: PathBuf,
    pub rollback_available: bool,
    pub rollback_path: Option<PathBuf>,
    pub installed_at_epoch_s: u64,
}

/// Record of a rollback operation.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RollbackReceipt {
    pub schema: String,
    pub status: String,
    pub restored_package_id: String,
    pub restored_version: String,
    pub kind: PackageKind,
    pub restored_at_epoch_s: u64,
}

/// Calculate a deterministic SHA-256 hex hash of file contents.
pub fn hash_file_contents(path: &Path) -> io::Result<String> {
    let bytes = fs::read(path)?;
    Ok(format!("sha256:{}", crate::learn::knowledge::sha256_hex(&bytes)))
}

/// Verify package manifest, payload existence, SHA-256 content hash, and optional base-model compatibility.
pub fn verify_package_integrity(
    package_dir: &Path,
    expected_base_model: Option<&str>,
) -> io::Result<PackageManifest> {
    let manifest_path = package_dir.join("ojon_package_manifest.json");
    if !manifest_path.is_file() {
        return Err(bad_input(format!(
            "Missing ojon_package_manifest.json in {}",
            package_dir.display()
        )));
    }

    let manifest_text = fs::read_to_string(&manifest_path)?;
    let manifest: PackageManifest = serde_json::from_str(&manifest_text)
        .map_err(|e| bad_input(format!("Invalid package manifest: {e}")))?;

    if manifest.schema != "ojon.package_manifest.v1" {
        return Err(bad_input(format!(
            "Unsupported package manifest schema '{}'",
            manifest.schema
        )));
    }

    if let (Some(expected), Some(required)) = (
        expected_base_model.map(str::trim).filter(|s| !s.is_empty()),
        manifest.base_model_compatibility.as_deref().map(str::trim).filter(|s| !s.is_empty()),
    ) {
        let exp_low = expected.to_ascii_lowercase();
        let req_low = required.to_ascii_lowercase();
        if req_low != "universal" && exp_low != req_low && !exp_low.contains(&req_low) {
            return Err(bad_input(format!(
                "Base model compatibility mismatch: package requires '{}', target is '{}'",
                required, expected
            )));
        }
    }

    let mut payload_files = Vec::new();
    for entry_res in fs::read_dir(package_dir)? {
        let entry = entry_res?;
        if entry.file_type()?.is_file() && entry.file_name() != "ojon_package_manifest.json" {
            payload_files.push(entry.path());
        }
    }

    if payload_files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Package directory contains no payload artifact files",
        ));
    }

    payload_files.sort();
    let primary_payload = &payload_files[0];
    let actual_size = fs::metadata(primary_payload)?.len();
    if actual_size != manifest.size_bytes {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Package size mismatch for '{}': manifest expects {} bytes, found {} bytes",
                primary_payload.display(),
                manifest.size_bytes,
                actual_size
            ),
        ));
    }

    let actual_hash = hash_file_contents(primary_payload)?;
    let legacy_siphash = {
        let bytes = fs::read(primary_payload)?;
        let mut hasher = DefaultHasher::new();
        bytes.hash(&mut hasher);
        format!("{:016x}", hasher.finish())
    };
    if actual_hash != manifest.content_hash && legacy_siphash != manifest.content_hash {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "Package integrity check failed for '{}': expected hash '{}', computed '{}'",
                primary_payload.display(),
                manifest.content_hash,
                actual_hash
            ),
        ));
    }

    Ok(manifest)
}

/// Create a new KnowledgePack containing verified JSONL facts.
pub fn create_knowledge_pack(
    package_id: &str,
    name: &str,
    version: &str,
    facts_jsonl: &Path,
    dest_dir: &Path,
    description: &str,
    author: &str,
    license: &str,
    changelog: Vec<String>,
) -> io::Result<PackageManifest> {
    fs::create_dir_all(dest_dir)?;

    let target_facts = dest_dir.join("knowledge_store.jsonl");
    fs::copy(facts_jsonl, &target_facts)?;

    let size_bytes = fs::metadata(&target_facts)?.len();
    let content_hash = hash_file_contents(&target_facts)?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let manifest = PackageManifest {
        schema: "ojon.package_manifest.v1".to_string(),
        package_id: package_id.to_string(),
        name: name.to_string(),
        version: version.to_string(),
        kind: PackageKind::KnowledgePack,
        base_model_compatibility: None, // Universal retrieval
        created_at_epoch_s: now,
        content_hash,
        size_bytes,
        description: description.to_string(),
        author: author.to_string(),
        license: license.to_string(),
        changelog,
        benchmark_scores: HashMap::new(),
    };

    let manifest_path = dest_dir.join("ojon_package_manifest.json");
    let file = File::create(&manifest_path)?;
    serde_json::to_writer_pretty(file, &manifest)?;

    Ok(manifest)
}

/// Create a new SkillPack containing an adapter and benchmark scores.
pub fn create_skill_pack(
    package_id: &str,
    name: &str,
    version: &str,
    adapter_file: &Path,
    base_model_compatibility: &str,
    dest_dir: &Path,
    benchmark_scores: HashMap<String, f64>,
    description: &str,
    author: &str,
    license: &str,
    changelog: Vec<String>,
) -> io::Result<PackageManifest> {
    fs::create_dir_all(dest_dir)?;

    let file_name = adapter_file
        .file_name()
        .ok_or_else(|| bad_input("invalid adapter path"))?;
    let target_adapter = dest_dir.join(file_name);
    fs::copy(adapter_file, &target_adapter)?;

    let size_bytes = fs::metadata(&target_adapter)?.len();
    let content_hash = hash_file_contents(&target_adapter)?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let manifest = PackageManifest {
        schema: "ojon.package_manifest.v1".to_string(),
        package_id: package_id.to_string(),
        name: name.to_string(),
        version: version.to_string(),
        kind: PackageKind::SkillPack,
        base_model_compatibility: Some(base_model_compatibility.to_string()),
        created_at_epoch_s: now,
        content_hash,
        size_bytes,
        description: description.to_string(),
        author: author.to_string(),
        license: license.to_string(),
        changelog,
        benchmark_scores,
    };

    let manifest_path = dest_dir.join("ojon_package_manifest.json");
    let file = File::create(&manifest_path)?;
    serde_json::to_writer_pretty(file, &manifest)?;

    Ok(manifest)
}

/// Create a new ModelRelease containing a full model checkpoint or GGUF.
pub fn create_model_release(
    package_id: &str,
    name: &str,
    version: &str,
    model_file: &Path,
    dest_dir: &Path,
    benchmark_scores: HashMap<String, f64>,
    description: &str,
    author: &str,
    license: &str,
    changelog: Vec<String>,
) -> io::Result<PackageManifest> {
    fs::create_dir_all(dest_dir)?;

    let file_name = model_file
        .file_name()
        .ok_or_else(|| bad_input("invalid model path"))?;
    let target_model = dest_dir.join(file_name);
    fs::copy(model_file, &target_model)?;

    let size_bytes = fs::metadata(&target_model)?.len();
    let content_hash = hash_file_contents(&target_model)?;
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let manifest = PackageManifest {
        schema: "ojon.package_manifest.v1".to_string(),
        package_id: package_id.to_string(),
        name: name.to_string(),
        version: version.to_string(),
        kind: PackageKind::ModelRelease,
        base_model_compatibility: None,
        created_at_epoch_s: now,
        content_hash,
        size_bytes,
        description: description.to_string(),
        author: author.to_string(),
        license: license.to_string(),
        changelog,
        benchmark_scores,
    };

    let manifest_path = dest_dir.join("ojon_package_manifest.json");
    let file = File::create(&manifest_path)?;
    serde_json::to_writer_pretty(file, &manifest)?;

    Ok(manifest)
}

/// Install an update package atomically into the local workstation environment.
/// Automatically snapshots any previously active package of the same kind for rollback.
pub fn install_package(package_dir: &Path, base_install_root: &Path) -> io::Result<InstallReceipt> {
    install_package_for_model(package_dir, base_install_root, None)
}

/// Install an update package with optional base-model compatibility enforcement and atomic staging.
pub fn install_package_for_model(
    package_dir: &Path,
    base_install_root: &Path,
    expected_base_model: Option<&str>,
) -> io::Result<InstallReceipt> {
    // 0. Strictly verify package manifest, payload size, SHA-256 hash, and base-model compatibility
    // BEFORE touching active/ or rollback/.
    let manifest = verify_package_integrity(package_dir, expected_base_model)?;

    let kind_str = manifest.kind.as_str();
    let active_dir = base_install_root.join("active").join(kind_str);
    let rollback_dir = base_install_root.join("rollback").join(kind_str);
    let now_nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let staging_dir = base_install_root
        .join("staging")
        .join(format!("{}_{}_{}", kind_str, std::process::id(), now_nanos));

    if staging_dir.exists() {
        fs::remove_dir_all(&staging_dir)?;
    }
    fs::create_dir_all(&staging_dir)?;

    // 1. Copy into isolated staging directory and verify staged integrity first
    for entry_res in fs::read_dir(package_dir)? {
        let entry = entry_res?;
        let target = staging_dir.join(entry.file_name());
        if entry.file_type()?.is_file() {
            fs::copy(entry.path(), &target)?;
        }
    }
    if let Err(e) = verify_package_integrity(&staging_dir, expected_base_model) {
        let _ = fs::remove_dir_all(&staging_dir);
        return Err(e);
    }

    // 2. If an active package exists, archive it into rollback directory
    let had_previous = active_dir.is_dir() && active_dir.join("ojon_package_manifest.json").is_file();
    if had_previous {
        if rollback_dir.exists() {
            fs::remove_dir_all(&rollback_dir)?;
        }
        if let Some(parent) = rollback_dir.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::rename(&active_dir, &rollback_dir)?;
    }

    // 3. Atomically promote staged directory into active directory
    if active_dir.exists() {
        fs::remove_dir_all(&active_dir)?;
    }
    if let Some(parent) = active_dir.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::rename(&staging_dir, &active_dir)?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let receipt = InstallReceipt {
        schema: "ojon.install_receipt.v1".to_string(),
        status: "installed_successfully".to_string(),
        package_id: manifest.package_id,
        version: manifest.version,
        kind: manifest.kind,
        active_path: active_dir,
        rollback_available: had_previous,
        rollback_path: if had_previous { Some(rollback_dir) } else { None },
        installed_at_epoch_s: now,
    };

    let receipt_path = base_install_root.join("last_install_receipt.json");
    let file = File::create(&receipt_path)?;
    serde_json::to_writer_pretty(file, &receipt)?;

    Ok(receipt)
}

/// List all currently installed active packages across all 3 tiers.
pub fn list_active_packages(base_install_root: &Path) -> io::Result<Vec<PackageManifest>> {
    let mut out = Vec::new();
    for kind in [
        PackageKind::KnowledgePack,
        PackageKind::SkillPack,
        PackageKind::ModelRelease,
    ] {
        let manifest_path = base_install_root
            .join("active")
            .join(kind.as_str())
            .join("ojon_package_manifest.json");
        if manifest_path.is_file() {
            if let Ok(txt) = fs::read_to_string(&manifest_path) {
                if let Ok(m) = serde_json::from_str::<PackageManifest>(&txt) {
                    out.push(m);
                }
            }
        }
    }
    Ok(out)
}

/// Roll back to the previous working package of the specified kind.
pub fn rollback_package(kind: PackageKind, base_install_root: &Path) -> io::Result<RollbackReceipt> {
    let kind_str = kind.as_str();
    let active_dir = base_install_root.join("active").join(kind_str);
    let rollback_dir = base_install_root.join("rollback").join(kind_str);

    if !rollback_dir.is_dir() || !rollback_dir.join("ojon_package_manifest.json").is_file() {
        return Err(bad_input(format!(
            "No rollback snapshot available for kind '{}'",
            kind_str
        )));
    }

    let rollback_manifest_text = fs::read_to_string(rollback_dir.join("ojon_package_manifest.json"))?;
    let manifest: PackageManifest = serde_json::from_str(&rollback_manifest_text)
        .map_err(|e| bad_input(format!("Corrupt rollback manifest: {e}")))?;

    // Swap rollback back to active
    if active_dir.exists() {
        fs::remove_dir_all(&active_dir)?;
    }
    fs::rename(&rollback_dir, &active_dir)?;

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let receipt = RollbackReceipt {
        schema: "ojon.rollback_receipt.v1".to_string(),
        status: "rollback_restored".to_string(),
        restored_package_id: manifest.package_id,
        restored_version: manifest.version,
        kind,
        restored_at_epoch_s: now,
    };

    let receipt_path = base_install_root.join("last_rollback_receipt.json");
    let file = File::create(&receipt_path)?;
    serde_json::to_writer_pretty(file, &receipt)?;

    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_knowledge_pack_lifecycle_and_rollback() {
        let temp_root = std::env::temp_dir().join(format!(
            "ojon_pkg_test_{}",
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        let pkg_v1_dir = temp_root.join("src_pkg_v1");
        let pkg_v2_dir = temp_root.join("src_pkg_v2");
        let dummy_facts_v1 = temp_root.join("facts_v1.jsonl");
        let dummy_facts_v2 = temp_root.join("facts_v2.jsonl");

        fs::create_dir_all(&temp_root).unwrap();
        fs::write(&dummy_facts_v1, "{\"item_id\":\"fact-1\",\"content\":\"v1 content\"}\n").unwrap();
        fs::write(&dummy_facts_v2, "{\"item_id\":\"fact-1\",\"content\":\"v2 content\"}\n").unwrap();

        // 1. Create Knowledge Pack v1.0.0
        let m1 = create_knowledge_pack(
            "kp-game-apis",
            "Game Dev Verified APIs",
            "1.0.0",
            &dummy_facts_v1,
            &pkg_v1_dir,
            "Game dev APIs v1",
            "Kalponic R&D",
            "MIT",
            vec!["Initial release".to_string()],
        )
        .unwrap();
        assert_eq!(m1.version, "1.0.0");

        // 2. Install v1.0.0
        let install_root = temp_root.join("workstation_packages");
        let receipt1 = install_package(&pkg_v1_dir, &install_root).unwrap();
        assert_eq!(receipt1.status, "installed_successfully");
        assert!(!receipt1.rollback_available); // First install, no rollback yet

        // 3. Create and Install Knowledge Pack v2.0.0
        let _m2 = create_knowledge_pack(
            "kp-game-apis",
            "Game Dev Verified APIs",
            "2.0.0",
            &dummy_facts_v2,
            &pkg_v2_dir,
            "Game dev APIs v2",
            "Kalponic R&D",
            "MIT",
            vec!["Updated to 2026 specs".to_string()],
        )
        .unwrap();

        let receipt2 = install_package(&pkg_v2_dir, &install_root).unwrap();
        assert_eq!(receipt2.version, "2.0.0");
        assert!(receipt2.rollback_available); // v1.0.0 preserved in rollback

        // Verify active directory has v2
        let active_manifest = fs::read_to_string(install_root.join("active/knowledge_pack/ojon_package_manifest.json")).unwrap();
        assert!(active_manifest.contains("\"version\": \"2.0.0\""));

        // 4. Trigger atomic Rollback
        let rb_receipt = rollback_package(PackageKind::KnowledgePack, &install_root).unwrap();
        assert_eq!(rb_receipt.status, "rollback_restored");
        assert_eq!(rb_receipt.restored_version, "1.0.0");

        // Verify active directory was restored back to v1.0.0
        let restored_manifest = fs::read_to_string(install_root.join("active/knowledge_pack/ojon_package_manifest.json")).unwrap();
        assert!(restored_manifest.contains("\"version\": \"1.0.0\""));

        let _ = fs::remove_dir_all(&temp_root);
    }

    #[test]
    fn test_corrupted_package_and_incompatible_base_model_rejected_before_touching_active() {
        let temp_root = std::env::temp_dir().join(format!(
            "ojon_pkg_tamper_{}",
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()
        ));
        fs::create_dir_all(&temp_root).unwrap();
        let install_root = temp_root.join("install");
        let adapter_file = temp_root.join("adapter.bin");
        fs::write(&adapter_file, b"valid-lora-weights-v1").unwrap();

        let pkg_v1 = temp_root.join("skill_v1");
        create_skill_pack(
            "sp-coder",
            "Rust Coder Skill",
            "1.0.0",
            &adapter_file,
            "Qwen3-4B",
            &pkg_v1,
            HashMap::new(),
            "Skill v1",
            "KS",
            "Apache-2.0",
            vec!["v1".into()],
        )
        .unwrap();

        // Install v1 for Qwen3-4B succeeds
        install_package_for_model(&pkg_v1, &install_root, Some("Qwen3-4B-Instruct")).unwrap();
        assert_eq!(list_active_packages(&install_root).unwrap().len(), 1);

        // Incompatible base model is rejected before touching active/
        let bad_model_res = install_package_for_model(&pkg_v1, &install_root, Some("Llama-3-8B"));
        assert!(bad_model_res.is_err());

        // Tamper with payload bytes in pkg_v2
        let pkg_v2 = temp_root.join("skill_v2_tampered");
        create_skill_pack(
            "sp-coder",
            "Rust Coder Skill",
            "2.0.0",
            &adapter_file,
            "Qwen3-4B",
            &pkg_v2,
            HashMap::new(),
            "Skill v2",
            "KS",
            "Apache-2.0",
            vec!["v2".into()],
        )
        .unwrap();
        // Corrupt payload bytes while keeping same length
        fs::write(pkg_v2.join("adapter.bin"), b"corrupted-weights-v2!").unwrap();
        let corrupt_res = install_package_for_model(&pkg_v2, &install_root, Some("Qwen3-4B"));
        assert!(corrupt_res.is_err());

        // Verify v1.0.0 is still intact in active/
        let active = list_active_packages(&install_root).unwrap();
        assert_eq!(active[0].version, "1.0.0");

        let _ = fs::remove_dir_all(&temp_root);
    }
}
