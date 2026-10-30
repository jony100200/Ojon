//! Ojon's independent Rust backend: native scalar weight packing plus optional public ML engines.
//! No GUI, Python, network access, or model weights are required for inspection.

pub mod core;
pub mod eval;
pub mod learn;
pub mod models;
pub mod quant;
pub mod uncensor;
pub mod workflows;

// Root-level module re-exports for transparent backward compatibility
// Core
pub use self::core::discovery;
pub use self::core::jobs;
pub use self::core::probe;
pub use self::core::resource;
pub use self::core::settings;
pub use self::core::smoke;

// Quant
pub use self::quant::adaptive;
pub use self::quant::advanced;
pub use self::quant::ara;
pub use self::quant::autoround;
pub use self::quant::calibration;
pub use self::quant::corpus;
pub use self::quant::exl3;
pub use self::quant::kld;
pub use self::quant::packed;
pub use self::quant::ppl;
pub use self::quant::quant_compare;
pub use self::quant::quant_parity;
pub use self::quant::recipe;
pub use self::quant::recipe_run;
pub use self::quant::sweep;

// Uncensor
pub use self::uncensor::abliteration;
pub use self::uncensor::activation_trace;
pub use self::uncensor::compressed_edit;
pub use self::uncensor::cvector_import;
pub use self::uncensor::gguf_edit;
pub use self::uncensor::model_edit;
pub use self::uncensor::refusal;
pub use self::uncensor::uncensor_auto;
pub use self::uncensor::uncensor_trace;

// Models
pub use self::models::diffusion;
pub use self::models::diffusion_gguf;
pub use self::models::gguf;
pub use self::models::hf;
pub use self::models::llama;
pub use self::models::model_build;

// Eval
pub use self::eval::benchmark;
pub use self::eval::distill;
pub use self::eval::eval_session;
pub use self::eval::method_select;
pub use self::eval::quality;
pub use self::eval::task_eval;

// Workflows
pub use self::workflows::adapter_smoke;
pub use self::workflows::optimize_workflow;
pub use self::workflows::options;
pub use self::workflows::research_routes;
pub use self::workflows::simple_workflow;
pub use self::workflows::train;

// Re-export specific public items
pub use self::workflows::flint_pipeline;
pub use self::workflows::flint_pipeline::{
    assemble_flint_hybrid_gguf, assess_flint_hardware_fit, compare_flint_logits_files,
    evaluate_flint_candidate_logits, generate_flint_roco_recipe, inspect_flint_moe,
    parity_gate_passes, plan_flint_pipeline, plan_flint_uncensor, promote_flint_candidate,
    router_storage_type_match_pct, run_flint_functional_suite, verify_moe_router_storage,
    FlintDirectLogitsComparison, FlintHardwareFit, FlintMoETopology, FlintParityReceipt,
    FlintPipelinePlan, FlintUncensorPlan,
};
pub use abliteration::{
    direction_from_activations, prepare_abliteration, run_abliteration, AblationPlan,
    AblationResult,
};
pub use activation_trace::{compile_activation_recipe, save_activation_recipe, ActivationRecipe};
pub use adapter_smoke::{
    prepare_adapter_smoke, run_adapter_smoke, AdapterSmokePlan, AdapterSmokeResult,
};
pub use adaptive::{
    allocate_bits, allocate_bits_with_strategy, compare_allocation_strategies,
    extract_sensitivity_from_imatrix, parse_tensor_sensitivity, prepare_adaptive, run_adaptive,
    sensitive_spine_pinning_recipe, AdaptivePlan, AllocationStrategy, TensorScore,
};
pub use advanced::{
    discover_advanced, plan_gsq, plan_gsq_assemble, plan_rco_assemble, plan_rco_database,
    plan_rco_search, run_advanced, AdvancedPlan,
};
pub use ara::{fit_ara, fit_ara_file, save_ara_recipe, AraFitResult};
pub use autoround::{prepare_autoround_plan, run_autoround, AutoRoundDevice, AutoRoundPlan};
pub use benchmark::{compare_bench_files, prepare_bench, run_bench, BenchPlan};
pub use calibration::{prepare_imatrix, run_imatrix, ImatrixPlan};
pub use compressed_edit::{
    prepare_compressed_edit, prepare_compressed_edit_with_recipe, run_compressed_edit,
    CompressedEditPlan, CompressedEditResult,
};
pub use cvector_import::{
    rank_control_vector_blocks, recipe_from_control_vector, recipe_from_control_vector_with_options,
};
pub use diffusion::{
    discover_diffusion, inspect_diffusion_recipe, plan_diffusion, run_diffusion, DiffusionEngine,
    DiffusionFormat, DiffusionKind, DiffusionPlan, DiffusionResult,
};
pub use discovery::{
    backend_specs, bundled_data_dir, bundled_data_file, discover_backends, discovered_path,
    hf_python_path, open_official_download_page, probe_backend_capabilities,
    probe_quantizer_capabilities, probe_quantizer_formats, probe_runtime_devices, BackendId,
    FoundBackend, Integration,
};
pub use distill::{
    audit_dataset_file, prepare_distill_plan, retrieve_recipes, DatasetAuditReport, DatasetEntry,
    DistillPlan, KnowledgeTier, RecipeMatch,
};
pub use exl3::{prepare_exl3, run_exl3, Exl3Plan};
pub use gguf::{
    classify_tensor_role, ggml_type_block_info, ggml_type_name, inspect_gguf, inspect_gguf_tensors,
    inspect_model_topology, GgufSummary, GgufTensor, ModelTopology, RuntimeMemoryEstimate,
};
pub use gguf_edit::{
    prepare_gguf_abliteration, prepare_gguf_abliteration_value, run_gguf_abliteration,
    GgufAblationPlan,
};
pub use hf::{
    inspect_hf_folder, prepare_hf, quantize_hf, quantize_hf_with_log, HfFolderInfo, HfQuantizePlan,
    HfQuantizeResult,
};
pub use jobs::{
    clear_job_cancel, finish_job_history, is_job_cancelled, job_history_path, load_job_history,
    record_job_start, request_job_cancel, save_job_history, update_job_stage, JobHistoryEntry,
};
pub use llama::{
    prepare_gguf, quantize_gguf, quantize_gguf_with_log, Precision, QuantizePlan, QuantizeResult,
};
pub use method_select::{choose_edit_method, choose_edit_method_file, MethodChoice};
pub use model_build::{prepare_model_build, run_model_build, ModelBuildPlan, ModelBuildResult};
pub use model_edit::{
    prepare_model_abliteration, run_model_abliteration, ModelAblationPlan, ModelAblationResult,
};
pub use optimize_workflow::{
    plan_optimize_workflow, run_optimize_workflow, OptimizeGoal, OptimizeWorkflowRequest,
};
pub use options::plan_options;
pub use packed::{plan_pack, run_pack, unpack, PackPlan, PackResult};
pub use ppl::{evaluate_sweep, parse_final_perplexity, PplSettings};
pub use probe::{probe_model, ModelCategory, ModelProbe};
pub use quality::{
    evaluate_optimization_readiness, parse_quality_report, recommend, QualityReport,
};
pub use quant_compare::{
    prepare_existing_quant_comparison, run_existing_quant_comparison, ExistingQuantPlan,
};
pub use quant_parity::{
    prepare_quant_parity, run_quant_parity, QuantParityOptions, QuantParityPlan,
};
pub use research_routes::{prepare_research, run_research, ResearchKind, ResearchPlan};
pub use resource::{
    assess_resources, discover_runtime_gpu_devices, is_integrated_gpu_name,
    resolve_auto_runtime_device, resource_plan, ResourceAssessment, ResourceMode, RuntimeGpuDevice,
};
pub use settings::{load_settings, save_settings, settings_path, LocalSettings};
pub use smoke::{smoke_gguf, smoke_gguf_capture, SmokeCapture};
pub use sweep::{execute_sweep, parse_precisions, prepare_sweep, SweepPlan};
pub use task_eval::{
    inspect_task_suite, parse_eval_suite, prepare_eval, run_eval, EvalPlan, EvalRequest, EvalSuite,
};
pub use train::{prepare_train, run_train, TrainPlan};
pub use uncensor_auto::{prepare_uncensor_auto, run_uncensor_auto, UncensorAutoRequest};
pub use uncensor_trace::{prepare_uncensor_from_trace, run_uncensor_from_trace, UncensorTracePlan};

// Learn
pub use self::learn::{
    apply_supersession, collector, compute_content_hash, create_knowledge_item,
    create_knowledge_pack, create_model_release, create_skill_pack, curate, daemon,
    default_utility_models, eval as learn_eval, evaluate_and_promote, execute_training_plan,
    export_to_llamafactory, extract_clean_text, extract_episode_metadata,
    fetch_online_knowledge_url, format_epoch_iso, get_dataset_by_id, harvest_agent_sessions,
    has_deterministic_verification, hash_file_contents, import_and_sanitize_jsonl, install_package,
    install_package_for_model, knowledge, list_active_packages, list_curated_datasets,
    load_episodes_from_jsonl, load_knowledge_store, packages, parse_iso_date_to_epoch,
    partition_dual_buffer, probe_host_hardware, redact_json_secrets, redact_secrets, registry,
    rollback_package, sanitize_untrusted_knowledge, save_knowledge_store, sha256_bytes, sha256_hex,
    temporal, train as learn_train, verify_package_integrity, AugmentedPromptResponse,
    CandidateModelEvaluation, CandidateType, CollectedEpisode, CollectorStats, CuratedBatch,
    CurationReceipt, DaemonConfig, DaemonStatus, DatasetCatalogEntry, DatasetCategory,
    DpoPreferencePair, DualBufferConfig, EpisodeOutcome, ExtractedEpisodeMetadata, GateCheckResult,
    HardwareDiagnosis, HardwareTier, HarvestReceipt, ImportReceipt, InstallReceipt, KnowledgeItem,
    KnowledgeSourceKind, LearnCollector, LearnDaemon, LearnTrainPlan, PackageKind, PackageManifest,
    PromotedModelPointer, PromotionDecision, PromotionPolicy, RollbackReceipt,
    TemporalKnowledgeEngine, TemporalRetrievalQuery, TemporalSearchResult, ToolCallRecord,
    TrainReceipt, UtilityModelSpec, CANONICAL_TOOL_DECLINE_SUITE, CURATED_DATASETS,
};
