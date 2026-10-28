//! Ojon's independent Rust backend: native scalar weight packing plus optional public ML engines.
//! No GUI, Python, network access, or model weights are required for inspection.

pub mod core;
pub mod quant;
pub mod uncensor;
pub mod models;
pub mod eval;
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
pub use self::quant::exl3;
pub use self::quant::packed;
pub use self::quant::ppl;
pub use self::quant::quant_compare;
pub use self::quant::quant_parity;
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
pub use settings::{LocalSettings, load_settings, save_settings, settings_path};
pub use discovery::{
    backend_specs, discover_backends, discovered_path, hf_python_path,
    open_official_download_page, BackendId, FoundBackend, Integration,
    probe_backend_capabilities, probe_quantizer_formats, probe_runtime_devices,
    probe_quantizer_capabilities, bundled_data_file, bundled_data_dir,
};
pub use jobs::{
    request_job_cancel, clear_job_cancel, is_job_cancelled,
    job_history_path, load_job_history, save_job_history,
    record_job_start, update_job_stage, finish_job_history, JobHistoryEntry,
};
pub use gguf::{
    inspect_gguf, inspect_gguf_tensors, inspect_model_topology,
    ggml_type_name, ggml_type_block_info, classify_tensor_role,
    GgufSummary, GgufTensor, ModelTopology, RuntimeMemoryEstimate,
};
pub use hf::{
    inspect_hf_folder, prepare_hf, quantize_hf, quantize_hf_with_log,
    HfFolderInfo, HfQuantizePlan, HfQuantizeResult,
};
pub use llama::{
    quantize_gguf, quantize_gguf_with_log, prepare_gguf, Precision,
    QuantizePlan, QuantizeResult,
};
pub use smoke::{smoke_gguf, smoke_gguf_capture, SmokeCapture};
pub use probe::{probe_model, ModelCategory, ModelProbe};
pub use train::{prepare_train, run_train, TrainPlan};
pub use quality::{evaluate_optimization_readiness, parse_quality_report, recommend, QualityReport};
pub use sweep::{execute_sweep, parse_precisions, prepare_sweep, SweepPlan};
pub use ppl::{evaluate_sweep, parse_final_perplexity, PplSettings};
pub use advanced::{
    plan_gsq, plan_gsq_assemble, plan_rco_database, plan_rco_search,
    plan_rco_assemble, run_advanced, discover_advanced, AdvancedPlan,
};
pub use distill::{
    audit_dataset_file, prepare_distill_plan, retrieve_recipes,
    DatasetAuditReport, DatasetEntry, DistillPlan, KnowledgeTier, RecipeMatch,
};
pub use autoround::{prepare_autoround_plan, run_autoround, AutoRoundDevice, AutoRoundPlan};
pub use diffusion::{
    plan_diffusion, run_diffusion, inspect_diffusion_recipe, discover_diffusion,
    DiffusionPlan, DiffusionResult, DiffusionEngine, DiffusionKind, DiffusionFormat,
};
pub use packed::{plan_pack, run_pack, unpack, PackPlan, PackResult};
pub use calibration::{prepare_imatrix, run_imatrix, ImatrixPlan};
pub use options::plan_options;
pub use research_routes::{prepare_research, run_research, ResearchKind, ResearchPlan};
pub use resource::{
    assess_resources, resource_plan, discover_runtime_gpu_devices,
    resolve_auto_runtime_device, is_integrated_gpu_name,
    ResourceAssessment, ResourceMode, RuntimeGpuDevice,
};
pub use benchmark::{prepare_bench, run_bench, compare_bench_files, BenchPlan};
pub use exl3::{prepare_exl3, run_exl3, Exl3Plan};
pub use task_eval::{parse_eval_suite, inspect_task_suite, prepare_eval, run_eval, EvalPlan, EvalSuite};
pub use quant_compare::{prepare_existing_quant_comparison, run_existing_quant_comparison, ExistingQuantPlan};
pub use quant_parity::{prepare_quant_parity, run_quant_parity, QuantParityOptions, QuantParityPlan};
pub use adapter_smoke::{prepare_adapter_smoke, run_adapter_smoke, AdapterSmokePlan, AdapterSmokeResult};
pub use compressed_edit::{
    prepare_compressed_edit, prepare_compressed_edit_with_recipe,
    run_compressed_edit, CompressedEditPlan, CompressedEditResult,
};
pub use ara::{fit_ara, fit_ara_file, save_ara_recipe, AraFitResult};
pub use method_select::{choose_edit_method, choose_edit_method_file, MethodChoice};
pub use model_build::{prepare_model_build, run_model_build, ModelBuildPlan, ModelBuildResult};
pub use activation_trace::{compile_activation_recipe, save_activation_recipe, ActivationRecipe};
pub use uncensor_trace::{prepare_uncensor_from_trace, run_uncensor_from_trace, UncensorTracePlan};
pub use gguf_edit::{prepare_gguf_abliteration, prepare_gguf_abliteration_value, run_gguf_abliteration, GgufAblationPlan};
pub use model_edit::{prepare_model_abliteration, run_model_abliteration, ModelAblationPlan, ModelAblationResult};
pub use abliteration::{direction_from_activations, prepare_abliteration, run_abliteration, AblationPlan, AblationResult};
pub use cvector_import::{
    rank_control_vector_blocks, recipe_from_control_vector,
    recipe_from_control_vector_with_options,
};
pub use adaptive::{
    allocate_bits, allocate_bits_with_strategy, compare_allocation_strategies,
    sensitive_spine_pinning_recipe, parse_tensor_sensitivity,
    extract_sensitivity_from_imatrix, prepare_adaptive, run_adaptive,
    AdaptivePlan, AllocationStrategy, TensorScore,
};
pub use optimize_workflow::{
    plan_optimize_workflow, run_optimize_workflow, OptimizeWorkflowRequest, OptimizeGoal,
};
pub use uncensor_auto::{
    prepare_uncensor_auto, run_uncensor_auto, UncensorAutoRequest,
};
pub use self::workflows::flint_pipeline;
pub use self::workflows::flint_pipeline::{
    plan_flint_pipeline, inspect_flint_moe, assess_flint_hardware_fit,
    promote_flint_candidate, evaluate_flint_candidate_logits,
    run_flint_functional_suite, generate_flint_roco_recipe,
    plan_flint_uncensor, FlintPipelinePlan, FlintMoETopology,
    FlintHardwareFit, FlintParityReceipt, FlintUncensorPlan,
    FlintDirectLogitsComparison,
    assemble_flint_hybrid_gguf, verify_moe_router_storage,
    compare_flint_logits_files, router_storage_type_match_pct, parity_gate_passes,
};
