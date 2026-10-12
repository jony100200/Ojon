//! Ojon's independent Rust backend: native scalar weight packing plus optional public ML engines.
//! No GUI, Python, network access, or model weights are required for inspection.

pub mod discovery;
pub mod gguf;
pub mod hf;
pub mod llama;
pub mod jobs;
pub mod smoke;
pub mod probe;
pub mod train;
pub mod quality;
pub mod sweep;
pub mod ppl;
pub mod calibration;
pub mod advanced;
pub mod diffusion;
pub mod packed;
pub mod distill;
pub mod autoround;

pub use discovery::{backend_specs, discover_backends, discovered_path, hf_python_path, open_official_download_page, BackendId, FoundBackend, Integration};
pub use gguf::{inspect_gguf, GgufSummary};
pub use hf::{inspect_hf_folder, prepare_hf, quantize_hf, quantize_hf_with_log, HfFolderInfo, HfQuantizePlan, HfQuantizeResult};
pub use llama::{quantize_gguf, quantize_gguf_with_log, prepare_gguf, Precision, QuantizePlan, QuantizeResult};
pub use smoke::{smoke_gguf, smoke_gguf_capture, SmokeCapture};

pub use probe::{probe_model, ModelCategory, ModelProbe};
pub use train::{prepare_train, run_train, TrainPlan};
pub use quality::{parse_quality_report, recommend, QualityReport};
pub use sweep::{execute_sweep, parse_precisions, prepare_sweep, SweepPlan};
pub use ppl::{evaluate_sweep, parse_final_perplexity, PplSettings};
pub use advanced::{AdvancedPlan, plan_gsq, plan_gsq_assemble, plan_rco_database, plan_rco_search, plan_rco_assemble, run_advanced, discover_advanced};
pub use distill::{
    audit_dataset_file, prepare_distill_plan, retrieve_recipes, DatasetAuditReport, DatasetEntry,
    DistillPlan, KnowledgeTier, RecipeMatch,
};
pub use autoround::{prepare_autoround_plan, run_autoround, AutoRoundDevice, AutoRoundPlan};

pub use diffusion::{plan_diffusion, run_diffusion, inspect_diffusion_recipe, discover_diffusion, DiffusionPlan, DiffusionResult, DiffusionEngine, DiffusionKind, DiffusionFormat};

pub use packed::{plan_pack,run_pack,unpack,PackPlan,PackResult};

pub use calibration::{prepare_imatrix,run_imatrix,ImatrixPlan};
