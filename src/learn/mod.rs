//! Ojon Learn — Distillation, Fine-Tuning & Data Flywheel System.
//!
//! Re-exports:
//! - `collector`: Local loopback capture gateway, telemetry intake & secret redaction.
//! - `daemon`: Universal headless background service & bounded resource guard.
//! - `registry`: Curated Hugging Face & community distillation dataset registry & streaming importer.
//! - `curate`: Dual-buffer splitter (anti-catastrophic remembering/forgetting) & LLaMA-Factory exporter.
//! - `knowledge`: Multimodal online knowledge ingestion, text normalization & supersession tracker.
//! - `temporal`: Time-aware knowledge engine, temporal retrieval & augmented prompt synthesis.
//! - `packages`: Versioned update packages (KnowledgePack, SkillPack, ModelRelease) & atomic rollback.
//! - `train`: Hardware-aware preflight probe, LoRA/QLoRA plan generator & execution runner.
//! - `eval`: Candidate evaluation matrix, silent tool-decline audit & promotion/rollback gates.

pub mod collector;
pub mod curate;
pub mod daemon;
pub mod eval;
pub mod knowledge;
pub mod packages;
pub mod registry;
pub mod temporal;
pub mod train;

pub use collector::{
    default_utility_models, extract_episode_metadata, harvest_agent_sessions, redact_json_secrets,
    redact_secrets, CollectedEpisode, CollectorStats, EpisodeOutcome, ExtractedEpisodeMetadata,
    HarvestReceipt, LearnCollector, ToolCallRecord, UtilityModelSpec,
};
pub use curate::{
    export_to_llamafactory, has_deterministic_verification, load_episodes_from_jsonl,
    partition_dual_buffer, CuratedBatch, CurationReceipt, DpoPreferencePair, DualBufferConfig,
};
pub use daemon::{DaemonConfig, DaemonStatus, LearnDaemon};
pub use eval::{
    evaluate_and_promote, CandidateModelEvaluation, CandidateType, GateCheckResult,
    PromotedModelPointer, PromotionDecision, PromotionPolicy, CANONICAL_TOOL_DECLINE_SUITE,
};
pub use knowledge::{
    apply_supersession, compute_content_hash, create_knowledge_item, extract_clean_text,
    fetch_online_knowledge_url, load_knowledge_store, sanitize_untrusted_knowledge,
    save_knowledge_store, sha256_bytes, sha256_hex, KnowledgeItem, KnowledgeSourceKind,
};
pub use packages::{
    create_knowledge_pack, create_model_release, create_skill_pack, hash_file_contents,
    install_package, install_package_for_model, list_active_packages, rollback_package,
    verify_package_integrity, InstallReceipt, PackageKind, PackageManifest, RollbackReceipt,
};
pub use registry::{
    get_dataset_by_id, import_and_sanitize_jsonl, list_curated_datasets, DatasetCatalogEntry,
    DatasetCategory, ImportReceipt, CURATED_DATASETS,
};
pub use temporal::{
    format_epoch_iso, parse_iso_date_to_epoch, AugmentedPromptResponse, TemporalKnowledgeEngine,
    TemporalRetrievalQuery, TemporalSearchResult,
};
pub use train::{
    execute_training_plan, probe_host_hardware, HardwareDiagnosis, HardwareTier, LearnTrainPlan,
    TrainReceipt,
};
