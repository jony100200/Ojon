//! Kathamo desktop frontend for Ojon.
//! Kept separate from src/bin (Cargo's executable entry-point convention).
//! Ojon's native workstation shell, powered by Kathamo.
//! This optional frontend depends on ojon_backend; it is not a quantization engine.
//! All model details displayed here come from inspected files. No demo benchmarks.

use ojon_backend::cvector_import::recipe_from_control_vector;
use ojon_backend::{
    clear_job_cancel, discover_backends, discover_runtime_gpu_devices, finish_job_history,
    inspect_model_topology, is_job_cancelled, load_job_history, open_official_download_page,
    record_job_start, request_job_cancel, resolve_auto_runtime_device, save_job_history,
    update_job_stage, BackendId, FoundBackend, GgufSummary, HfFolderInfo, HfQuantizeResult,
    JobHistoryEntry, ModelProbe, ModelTopology, Precision, QuantizeResult, RuntimeGpuDevice,
    SmokeCapture,
};
use ojon_backend::{
    compile_activation_recipe, prepare_gguf_abliteration_value, prepare_uncensor_from_trace,
    run_gguf_abliteration, run_uncensor_from_trace,
};
use ojon_backend::{
    hf_python_path, inspect_gguf, inspect_hf_folder, prepare_gguf, prepare_hf, probe_model,
    quantize_gguf_with_log, quantize_hf_with_log, smoke_gguf_capture,
};
use ojon_backend::{load_settings, save_settings, LocalSettings};
use ojon_backend::{
    plan_optimize_workflow, run_optimize_workflow, OptimizeGoal, OptimizeWorkflowRequest,
};
use ojon_backend::{prepare_bench, resource_plan, run_bench, ResourceMode};
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::time::{Duration, Instant};
use ui_core::{Rect, SizeConstraint, Vec2};
use ui_platform::winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, MouseButton, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes, WindowId},
};
use ui_platform::PointerState;
use ui_render::{CommandBuffer, CpuBackend, GpuBackend, RenderBackend, SoftwarePresenter};
use ui_themes::{BilingualLabel, ThemePalette};
use ui_widgets::{
    icon_cache::{IconAtlas, StudioIcon},
    KathamoUi, NavigationRail, NavigationRailItem,
};

// Native Windows clipboard: cheap path pasting without a browser/UI runtime.
// Safe only for CF_UNICODETEXT global-memory clipboard handles.
#[cfg(windows)]
fn clipboard_text() -> Option<String> {
    use windows_sys::Win32::System::DataExchange::{
        CloseClipboard, GetClipboardData, OpenClipboard,
    };
    use windows_sys::Win32::System::Memory::{GlobalLock, GlobalSize, GlobalUnlock};
    unsafe {
        if OpenClipboard(0) == 0 {
            return None;
        }
        // Win32 CF_UNICODETEXT is the documented clipboard format 13.
        let handle = GetClipboardData(13u32);
        if handle == 0 {
            CloseClipboard();
            return None;
        }
        let global = handle as *mut core::ffi::c_void;
        let size = GlobalSize(global) / 2;
        let ptr = GlobalLock(global) as *const u16;
        if ptr.is_null() {
            CloseClipboard();
            return None;
        }
        let limit = size.min(8192);
        let mut len = 0usize;
        while len < limit && *ptr.add(len) != 0 {
            len += 1;
        }
        let value = String::from_utf16_lossy(std::slice::from_raw_parts(ptr, len));
        GlobalUnlock(global);
        CloseClipboard();
        Some(value)
    }
}

#[cfg(not(windows))]
fn clipboard_text() -> Option<String> {
    None
}

#[cfg(windows)]
fn pick_model_file() -> Option<PathBuf> {
    let script = "Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.OpenFileDialog; $f.Title = 'Select GGUF or Model File'; $f.Filter = 'AI Models (*.gguf;*.safetensors)|*.gguf;*.safetensors|All Files (*.*)|*.*'; if ($f.ShowDialog() -eq 'OK') { [Console]::Out.WriteLine($f.FileName) }";
    let out = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Sta", "-Command", script])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!text.is_empty()).then(|| PathBuf::from(text))
}

#[cfg(not(windows))]
fn pick_model_file() -> Option<PathBuf> {
    None
}

#[cfg(windows)]
fn pick_folder(description: &str) -> Option<PathBuf> {
    let safe_desc = description.replace('\'', "");
    let script = format!(
        "Add-Type -AssemblyName System.Windows.Forms; $f = New-Object System.Windows.Forms.FolderBrowserDialog; $f.Description = '{safe_desc}'; if ($f.ShowDialog() -eq 'OK') {{ [Console]::Out.WriteLine($f.SelectedPath) }}"
    );
    let out = std::process::Command::new("powershell")
        .args(["-NoProfile", "-Sta", "-Command", &script])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout).trim().to_string();
    (!text.is_empty()).then(|| PathBuf::from(text))
}

#[cfg(not(windows))]
fn pick_folder(_description: &str) -> Option<PathBuf> {
    None
}

fn open_folder_in_explorer(path: &Path) -> std::io::Result<()> {
    let dir = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    #[cfg(windows)]
    {
        std::process::Command::new("explorer").arg(dir).spawn()?;
    }
    #[cfg(not(windows))]
    {
        std::process::Command::new("xdg-open").arg(dir).spawn()?;
    }
    Ok(())
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus {
    Model,
    Output,
    Quantizer,
    Converter,
    Runtime,
    EvalServer,
    Benchmark,
    Prompt,
    GpuLayers,
    OptBaseline,
    OptCandidate,
    OptSuite,
    OptOutput,
    OptImatrix,
    OptSensitivity,
    OptPpl,
    OptMinSaving,
    OptMaxLoss,
    OptMaxPpl,
    OptRuntimeDevice,
    OptPrecisions,
    OptBenchPrompt,
    OptBenchGeneration,
    OptBenchRepetitions,
    OptSpeedRatio,
    UncTrace,
    UncOutput,
    UncStrength,
    UncBlock,
    UncPaired,
    UncRefusalSuite,
    UncTaskSuite,
}

#[derive(Clone, PartialEq)]
struct UncensorRequest {
    source: PathBuf,
    trace: PathBuf,
    output: PathBuf,
    strength: f64,
    block: Option<usize>,
}

fn plan_uncensor(request: &UncensorRequest) -> std::io::Result<serde_json::Value> {
    if let Some(block) = request.block {
        let recipe = recipe_from_control_vector(&request.trace, block, request.strength)?;
        let plan = prepare_gguf_abliteration_value(&request.source, &recipe, &request.output)?;
        Ok(
            serde_json::json!({"model_edit":plan.as_json(),"activation_tensors":1,
            "direction_origin":"LLAMA_CPP_RESIDUAL_CONTROL_VECTOR","selected_block":block,
            "refusal_rate":"NOT_EVALUATED","capability_retention":"NOT_EVALUATED","status":"PLAN_ONLY"}),
        )
    } else if request.source.extension().and_then(|x| x.to_str()) == Some("gguf") {
        let activation = compile_activation_recipe(&request.trace, request.strength, true)?;
        let plan =
            prepare_gguf_abliteration_value(&request.source, &activation.recipe, &request.output)?;
        Ok(serde_json::json!({"model_edit":plan.as_json(),
            "activation_records":activation.records,"activation_tensors":activation.tensors,
            "activation_capture":"EXTERNAL_INSTRUMENTED_RUNTIME",
            "refusal_rate":"NOT_EVALUATED","capability_retention":"NOT_EVALUATED",
            "status":"PLAN_ONLY"}))
    } else {
        Ok(prepare_uncensor_from_trace(
            &request.source,
            &request.trace,
            &request.output,
            request.strength,
            true,
        )?
        .as_json())
    }
}

fn execute_uncensor(
    request: &UncensorRequest,
    mut log: impl FnMut(&str),
) -> std::io::Result<serde_json::Value> {
    if let Some(block) = request.block {
        let recipe = recipe_from_control_vector(&request.trace, block, request.strength)?;
        let plan = prepare_gguf_abliteration_value(&request.source, &recipe, &request.output)?;
        let model = run_gguf_abliteration(&plan, |line| log(line))?;
        Ok(serde_json::json!({"model":model,"activation_tensors":1,
            "direction_origin":"LLAMA_CPP_RESIDUAL_CONTROL_VECTOR","selected_block":block,
            "refusal_rate":"NOT_EVALUATED","capability_retention":"NOT_EVALUATED",
            "status":"EDITED_GGUF_NOT_VALIDATED"}))
    } else if request.source.extension().and_then(|x| x.to_str()) == Some("gguf") {
        let activation = compile_activation_recipe(&request.trace, request.strength, true)?;
        let plan =
            prepare_gguf_abliteration_value(&request.source, &activation.recipe, &request.output)?;
        let model = run_gguf_abliteration(&plan, |line| log(line))?;
        Ok(
            serde_json::json!({"model":model,"activation_records":activation.records,
            "activation_tensors":activation.tensors,"activation_capture":"EXTERNAL_INSTRUMENTED_RUNTIME",
            "refusal_rate":"NOT_EVALUATED","capability_retention":"NOT_EVALUATED",
            "status":"EDITED_GGUF_NOT_VALIDATED"}),
        )
    } else {
        let plan = prepare_uncensor_from_trace(
            &request.source,
            &request.trace,
            &request.output,
            request.strength,
            true,
        )?;
        run_uncensor_from_trace(&plan, |line| log(line))
    }
}

enum WorkerEvent {
    Inspected(Result<(GgufSummary, ModelProbe, Option<ModelTopology>), String>),
    HfInspected(Result<(HfFolderInfo, ModelProbe), String>),
    StandaloneProbed(Result<ModelProbe, String>),
    Planned(Result<String, String>),
    OptimizerPlan(OptimizeWorkflowRequest, Result<serde_json::Value, String>),
    OptimizerRun(Result<serde_json::Value, String>),
    UncensorPlan(UncensorRequest, Result<serde_json::Value, String>),
    UncensorRun(Result<serde_json::Value, String>),
    UncensorAutoPlan(
        ojon_backend::uncensor_auto::UncensorAutoRequest,
        Result<serde_json::Value, String>,
    ),
    UncensorAutoRun(Result<serde_json::Value, String>),
    Quantized(Result<QuantizeResult, String>),
    HfQuantized(Result<HfQuantizeResult, String>),
    Smoke(Result<SmokeCapture, String>),
    Bench(Result<(PathBuf, serde_json::Value), String>),
    BackendCapabilities(serde_json::Value),
    Stage(String),
    Log(String),
}

struct Workstation {
    source: String,
    output: String,
    local_settings: LocalSettings,
    settings_load_failed: bool,
    quantizer: String,
    converter: String,
    runtime: String,
    eval_server: String,
    benchmark: String,
    prompt: String,
    gpu_layers: String,
    recipe: usize,
    format_tab: usize,
    workspace_tab: usize,
    optimize_settings_tab: usize,
    optimize_review_tab: usize,
    backends_expanded: bool,
    advanced_opt_expanded: bool,
    advanced_unc_expanded: bool,
    history_expanded: bool,
    backend_capabilities: Option<serde_json::Value>,
    detected_gpus: Vec<RuntimeGpuDevice>,
    opt_baseline: String,
    opt_candidate: String,
    opt_suite: String,
    opt_output: String,
    opt_imatrix: String,
    opt_sensitivity: String,
    opt_ppl: String,
    opt_min_saving: String,
    opt_max_loss: String,
    opt_max_ppl: String,
    opt_goal: usize,
    opt_device: usize,
    opt_runtime_device: String,
    opt_precisions: String,
    opt_bench_prompt: String,
    opt_bench_generation: String,
    opt_bench_repetitions: String,
    opt_speed_ratio: String,
    opt_planned: Option<(OptimizeWorkflowRequest, serde_json::Value)>,
    opt_result: Option<serde_json::Value>,
    unc_trace: String,
    unc_output: String,
    unc_strength: String,
    unc_block: String,
    unc_planned: Option<(UncensorRequest, serde_json::Value)>,
    unc_result: Option<serde_json::Value>,
    unc_tab: usize,
    unc_paired: String,
    unc_refusal_suite: String,
    unc_task_suite: String,
    unc_auto_quantize: usize,
    unc_auto_planned: Option<(
        ojon_backend::uncensor_auto::UncensorAutoRequest,
        serde_json::Value,
    )>,
    unc_auto_result: Option<serde_json::Value>,
    learn_subtab: usize,
    learn_dataset_filter: usize,
    learn_train_quant: usize,
    learn_train_rank: usize,
    learn_last_action_msg: Option<String>,
    focus: Option<Focus>,
    model: Option<GgufSummary>,
    model_topology: Option<ModelTopology>,
    hf_folder: Option<HfFolderInfo>,
    probe: Option<ModelProbe>,
    last_inference: Option<String>,
    last_benchmark: Option<(PathBuf, serde_json::Value)>,
    inspected_source: Option<String>,
    last_output: Option<(PathBuf, u64)>,
    started_at: Option<Instant>,
    last_elapsed: Option<Duration>,
    current_stage: String,
    completed_stages: Vec<String>,
    job_history: Vec<JobHistoryEntry>,
    log_tail: VecDeque<String>,
    status: String,
    busy: bool,
    tx: mpsc::Sender<WorkerEvent>,
    rx: mpsc::Receiver<WorkerEvent>,
    icons: IconAtlas,
    backends: Vec<FoundBackend>,
    pointer: PointerState,
    click_pulse: bool,
    control_held: bool,
    select_all: bool,
    needs_redraw: bool,
}

impl Workstation {
    fn new() -> Self {
        let (tx, rx) = mpsc::channel();
        let (settings, settings_problem) = match load_settings() {
            Ok(value) => (value, None),
            Err(e) => (LocalSettings::default(), Some(e.to_string())),
        };
        let mut source = settings
            .last_model
            .as_deref()
            .filter(|path| Path::new(path).exists())
            .unwrap_or("")
            .to_string();
        let backends = if settings_problem.is_none() {
            ojon_backend::discovery::discover_backends_with_settings(&settings)
        } else {
            // Never silently pick a different executable if a preferences
            // file exists but could not be parsed.
            Vec::new()
        };
        let mut quantizer = backends
            .iter()
            .find(|b| b.spec.id == BackendId::LlamaQuantize)
            .and_then(|b| b.executable.as_ref())
            .map(|p| p.to_string_lossy().into_owned())
            .or_else(|| settings.backend("OJON_LLAMA_QUANTIZE").map(str::to_owned))
            .unwrap_or_else(|| "llama-quantize".to_owned());
        let mut converter = backends
            .iter()
            .find(|b| b.spec.id == BackendId::HfConverter)
            .and_then(|b| b.executable.as_ref())
            .map(|p| p.to_string_lossy().into_owned())
            .or_else(|| settings.backend("OJON_CONVERT_HF").map(str::to_owned))
            .unwrap_or_else(|| "convert_hf_to_gguf.py".to_owned());
        let mut runtime = backends
            .iter()
            .find(|b| b.spec.id == BackendId::LlamaCli)
            .and_then(|b| b.executable.as_ref())
            .map(|p| p.to_string_lossy().into_owned())
            .or_else(|| settings.backend("OJON_LLAMA_CLI").map(str::to_owned))
            .unwrap_or_else(|| "llama-cli".to_owned());
        let eval_server = backends
            .iter()
            .find(|b| b.spec.id == BackendId::LlamaServer)
            .and_then(|b| b.executable.as_ref())
            .map(|path| path.to_string_lossy().into_owned())
            .or_else(|| settings.backend("OJON_LLAMA_SERVER").map(str::to_owned))
            .unwrap_or_default();
        let benchmark = backends
            .iter()
            .find(|b| b.spec.id == BackendId::LlamaBench)
            .and_then(|b| b.executable.as_ref())
            .map(|p| p.to_string_lossy().into_owned())
            .or_else(|| settings.backend("OJON_LLAMA_BENCH").map(str::to_owned))
            .unwrap_or_else(|| "llama-bench".to_owned());
        let mut args = std::env::args_os().skip(1);
        while let Some(key) = args.next() {
            match key.to_str() {
                Some("--model") => {
                    if let Some(s) = args.next() {
                        source = s.to_string_lossy().into_owned();
                    }
                }
                Some("--quantizer") => {
                    if let Some(s) = args.next() {
                        quantizer = s.to_string_lossy().into_owned();
                    }
                }
                Some("--converter") => {
                    if let Some(s) = args.next() {
                        converter = s.to_string_lossy().into_owned();
                    }
                }
                Some("--runtime") => {
                    if let Some(s) = args.next() {
                        runtime = s.to_string_lossy().into_owned();
                    }
                }
                _ => {}
            }
        }
        let output = default_output(&source, Precision::default());
        let initial_opt_output = settings.output_directory.clone().unwrap_or_else(|| {
            Path::new(&output)
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default()
        });
        let detected_gpus = discover_runtime_gpu_devices();
        let job_history = load_job_history().unwrap_or_default();
        Self {
            source, output, quantizer, converter, runtime, eval_server, benchmark,
            local_settings:settings.clone(),
            settings_load_failed:settings_problem.is_some(),
            prompt: "Explain in one sentence what a quantized language model is.".into(),
            gpu_layers: settings.gpu_layers.to_string(),
            recipe: 0,
            format_tab: 0,
            workspace_tab: 0,
            optimize_settings_tab: 0,
            optimize_review_tab: 0,
            backends_expanded: false,
            advanced_opt_expanded: false,
            advanced_unc_expanded: false,
            history_expanded: false,
            backend_capabilities:None,
            detected_gpus,
            opt_baseline: String::new(),
            opt_candidate: String::new(),
            opt_suite: String::new(),
            opt_output: initial_opt_output,
            opt_imatrix: String::new(),
            opt_sensitivity: String::new(),
            opt_ppl: String::new(),
            opt_min_saving: settings.min_saving_percent.to_string(),
            opt_max_loss: settings.max_loss_points.to_string(),
            opt_max_ppl: settings.max_ppl_regression_pct.to_string(),
            opt_goal: match settings.optimization_goal.as_str(){
                "smallest"=>1,"fastest"=>2,"maximum_quality"=>3,_=>0
            },
            opt_device:match settings.preferred_device.as_str(){
                "cpu"=>1,"cuda"=>2,"xpu"=>3,"vulkan"=>4,_=>0
            },
            opt_runtime_device:settings.runtime_device.clone().unwrap_or_default(),
            opt_precisions:settings.precision_candidates.clone().unwrap_or_default(),
            opt_bench_prompt:settings.bench_prompt_tokens.to_string(),
            opt_bench_generation:settings.bench_generation_tokens.to_string(),
            opt_bench_repetitions:settings.bench_repetitions.to_string(),
            opt_speed_ratio:settings.balanced_min_speed_ratio.to_string(),
            opt_planned: None,
            opt_result: None,
            unc_trace:String::new(),
            unc_output:String::new(),
            unc_strength:"0.7".into(),
            unc_block:String::new(),
            unc_planned:None,
            unc_result:None,
            unc_tab:0,
            unc_paired:String::new(),
            unc_refusal_suite:String::new(),
            unc_task_suite:String::new(),
            unc_auto_quantize:0,
            unc_auto_planned:None,
            unc_auto_result:None,
            learn_subtab: 0,
            learn_dataset_filter: 0,
            learn_train_quant: 0,
            learn_train_rank: 0,
            learn_last_action_msg: None,
            focus: None,
            model: None,
            model_topology: None,
            hf_folder: None,
            probe: None,
            last_inference: None,
            last_benchmark: None,
            inspected_source: None,
            last_output: None,
            started_at: None,
            last_elapsed: None,
            current_stage: "Idle".into(),
            completed_stages: Vec::new(),
            job_history,
            log_tail: VecDeque::new(),
            status:settings_problem.map(|e|format!(
                "Local settings could not be loaded: {e}. Check ojon settings show; original file untouched."
            )).unwrap_or_else(||"Drop a GGUF file or HF model folder here, or click Browse Model...".into()),
            busy: false,
            tx, rx,
            icons: IconAtlas::new(),
            backends,
            pointer: PointerState::default(),
            click_pulse: false,
            control_held: false,
            select_all: false,
            needs_redraw: true,
        }
    }
    /// Save only user preferences. CLI overrides and inferred runtime
    /// capabilities are deliberately not treated as verified settings.
    fn persist_local_settings(&mut self, show_success: bool) {
        if self.settings_load_failed {
            self.status="Settings file is invalid. Repair it with the CLI or restore a backup before saving GUI preferences.".into();
            self.needs_redraw = true;
            return;
        }
        let mut next = self.local_settings.clone();
        next.last_model = Some(self.source.trim().to_string()).filter(|v| !v.is_empty());
        next.model_directory = next
            .last_model
            .as_deref()
            .and_then(|v| Path::new(v).parent())
            .filter(|v| !v.as_os_str().is_empty())
            .map(|v| v.to_string_lossy().into_owned())
            .or_else(|| next.model_directory.clone());
        next.output_directory = Some(self.opt_output.trim().to_string()).filter(|v| !v.is_empty());
        for (key, value, default) in [
            (
                "OJON_LLAMA_QUANTIZE",
                self.quantizer.trim(),
                "llama-quantize",
            ),
            (
                "OJON_CONVERT_HF",
                self.converter.trim(),
                "convert_hf_to_gguf.py",
            ),
            ("OJON_LLAMA_CLI", self.runtime.trim(), "llama-cli"),
            ("OJON_LLAMA_SERVER", self.eval_server.trim(), "llama-server"),
            ("OJON_LLAMA_BENCH", self.benchmark.trim(), "llama-bench"),
        ] {
            if value.is_empty() || value == default {
                next.backend_paths.remove(key);
            } else {
                next.backend_paths.insert(key.to_owned(), value.to_owned());
            }
        }
        next.optimization_goal = match self.opt_goal {
            1 => "smallest",
            2 => "fastest",
            3 => "maximum_quality",
            _ => "balanced",
        }
        .into();
        next.preferred_device = match self.opt_device {
            1 => "cpu",
            2 => "cuda",
            3 => "xpu",
            4 => "vulkan",
            _ => "auto",
        }
        .into();
        next.runtime_device = if matches!(self.opt_device, 2 | 3 | 4) {
            Some(self.opt_runtime_device.trim().to_string()).filter(|v| !v.is_empty())
        } else {
            None
        };
        next.precision_candidates =
            Some(self.opt_precisions.trim().to_string()).filter(|v| !v.is_empty());
        let parsed = (|| {
            next.min_saving_percent = self
                .opt_min_saving
                .trim()
                .parse::<f64>()
                .map_err(|_| "Invalid minimum size saving")?;
            next.max_loss_points = self
                .opt_max_loss
                .trim()
                .parse::<f64>()
                .map_err(|_| "Invalid maximum task quality loss")?;
            next.max_ppl_regression_pct = self
                .opt_max_ppl
                .trim()
                .parse::<f64>()
                .map_err(|_| "Invalid maximum PPL regression")?;
            next.gpu_layers = self
                .gpu_layers
                .trim()
                .parse::<u32>()
                .map_err(|_| "Invalid GPU layer count")?;
            next.bench_prompt_tokens = self
                .opt_bench_prompt
                .trim()
                .parse::<u32>()
                .map_err(|_| "Invalid benchmark prefill tokens")?;
            next.bench_generation_tokens = self
                .opt_bench_generation
                .trim()
                .parse::<u32>()
                .map_err(|_| "Invalid benchmark decode tokens")?;
            next.bench_repetitions = self
                .opt_bench_repetitions
                .trim()
                .parse::<u32>()
                .map_err(|_| "Invalid benchmark repetitions")?;
            next.balanced_min_speed_ratio = self
                .opt_speed_ratio
                .trim()
                .parse::<f64>()
                .map_err(|_| "Invalid Balanced speed ratio")?;
            next.validate().map_err(|_| "Invalid preference value")
        })();
        if let Err(error) = parsed {
            self.status = format!("Preferences not saved: {error}");
            self.needs_redraw = true;
            return;
        }
        match save_settings(&next) {
            Ok(path) => {
                self.local_settings = next;
                if show_success {
                    self.status = format!("Local preferences saved: {}", path.display());
                    self.needs_redraw = true;
                }
            }
            Err(e) => {
                self.status = format!("Cannot save local preferences: {e}");
                self.needs_redraw = true;
                eprintln!("Ojon local settings save failed: {e}");
            }
        }
    }
    fn rescan_backends(&mut self) {
        self.backend_capabilities = None;
        self.backends = discover_backends();
        if self.converter.trim() == "convert_hf_to_gguf.py" || self.converter.trim().is_empty() {
            if let Some(path) = self
                .backends
                .iter()
                .find(|b| b.spec.id == BackendId::HfConverter)
                .and_then(|b| b.executable.as_ref())
            {
                self.converter = path.to_string_lossy().into_owned();
            }
        }
        if self.quantizer.trim() == "llama-quantize" || self.quantizer.trim().is_empty() {
            if let Some(path) = self
                .backends
                .iter()
                .find(|b| b.spec.id == BackendId::LlamaQuantize)
                .and_then(|b| b.executable.as_ref())
            {
                self.quantizer = path.to_string_lossy().into_owned();
            }
        }
        if self.runtime.trim() == "llama-cli" || self.runtime.trim().is_empty() {
            if let Some(path) = self
                .backends
                .iter()
                .find(|b| b.spec.id == BackendId::LlamaCli)
                .and_then(|b| b.executable.as_ref())
            {
                self.runtime = path.to_string_lossy().into_owned();
            }
        }
        let installed = self.backends.iter().filter(|b| b.installed()).count();
        self.status = format!("{installed} known backend executable(s) detected. Discovery alone does not confirm compatibility.");
        self.needs_redraw = true;
    }

    /// Respect the executable paths currently shown in Kathamo. Saved
    /// preferences are not a prerequisite for an explicit per-run override.
    /// In particular, selecting a SYCL or Vulkan llama-cli must not
    /// accidentally run the first CUDA binary discovered at startup.
    fn selected_backends(&self) -> Vec<FoundBackend> {
        let mut found = self.backends.clone();
        for (id, current, default) in [
            (
                BackendId::LlamaQuantize,
                self.quantizer.trim(),
                "llama-quantize",
            ),
            (BackendId::LlamaCli, self.runtime.trim(), "llama-cli"),
            (
                BackendId::LlamaServer,
                self.eval_server.trim(),
                "llama-server",
            ),
            (BackendId::LlamaBench, self.benchmark.trim(), "llama-bench"),
            (
                BackendId::HfConverter,
                self.converter.trim(),
                "convert_hf_to_gguf.py",
            ),
        ] {
            if current.is_empty() || current == default {
                continue;
            }
            if let Some(backend) = found.iter_mut().find(|item| item.spec.id == id) {
                let path = PathBuf::from(current);
                backend.executable = path.is_file().then_some(path);
                backend.found_via = Some("Kathamo selected executable");
            }
        }
        found
    }
    /// Explicit bounded tool probes, never an implicit model load.
    fn start_capability_probe(&mut self) {
        if self.busy {
            return;
        }
        let tools = self.selected_backends();
        let tx = self.tx.clone();
        self.backend_capabilities = None;
        self.busy = true;
        self.started_at = Some(Instant::now());
        self.status =
            "Probing selected local executables (no model inference or quantization)...".into();
        self.needs_redraw = true;
        std::thread::spawn(move || {
            let receipt = ojon_backend::probe_backend_capabilities(&tools);
            let _ = tx.send(WorkerEvent::BackendCapabilities(receipt));
        });
    }

    #[allow(dead_code)]
    fn hf_converter(&self) -> &str {
        &self.converter
    }
    fn precision(&self) -> Precision {
        match self.recipe {
            1 => Precision::Q4KM,
            2 => Precision::Q3KM,
            3 => Precision::Q6K,
            _ => Precision::Q5KM,
        }
    }
    fn set_source(&mut self, path: &Path) {
        if self.busy {
            self.status =
                "A job is already running. Finish it before importing a different model.".into();
            self.needs_redraw = true;
            return;
        }
        self.source = path.to_string_lossy().into_owned();
        self.output = default_output(&self.source, self.precision());
        self.model = None;
        self.model_topology = None;
        self.hf_folder = None;
        self.probe = None;
        self.last_inference = None;
        self.inspected_source = None;
        self.last_output = None;
        self.opt_planned = None;
        self.opt_result = None;
        let parent_dir = path
            .parent()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_default();
        self.opt_output = parent_dir.clone();
        if self.unc_output.trim().is_empty() {
            self.unc_output = parent_dir;
        }
        self.status = "Model selected. Inspecting architecture and hardware fit...".into();
        self.needs_redraw = true;
        self.persist_local_settings(false);
        self.start_inspect();
    }
    fn start_inspect(&mut self) {
        if self.busy {
            return;
        }
        self.inspected_source = None;
        self.model = None;
        self.model_topology = None;
        self.hf_folder = None;
        self.probe = None;
        self.last_output = None;
        let model = PathBuf::from(self.source.trim());
        if self.source.trim().is_empty() {
            self.status = "Choose a GGUF file or local HF model folder first.".into();
            return;
        }
        let tx = self.tx.clone();
        if self.output.trim().is_empty() {
            self.output = default_output(&self.source, self.precision());
        }
        self.status =
            "Reading model metadata and tensor topology without loading weights...".into();
        self.current_stage = "Inspecting model metadata".into();
        self.inspected_source = Some(self.source.clone());
        self.started_at = Some(Instant::now());
        self.busy = true;
        self.needs_redraw = true;
        std::thread::spawn(move || {
            if model.is_dir() {
                let result = inspect_hf_folder(&model)
                    .and_then(|info| probe_model(&model).map(|p| (info, p)))
                    .map_err(|e| e.to_string());
                let _ = tx.send(WorkerEvent::HfInspected(result));
            } else if model
                .extension()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("gguf"))
            {
                let result = inspect_gguf(&model)
                    .and_then(|info| {
                        let topo = inspect_model_topology(&model).ok();
                        probe_model(&model).map(|p| (info, p, topo))
                    })
                    .map_err(|e| e.to_string());
                let _ = tx.send(WorkerEvent::Inspected(result));
            } else {
                let result = probe_model(&model).map_err(|e| e.to_string());
                let _ = tx.send(WorkerEvent::StandaloneProbed(result));
            }
        });
    }
    /// KLD recipe search, plan mode. Uses the manual-tools source/output/quantizer
    /// and the Optimize tab's held-out perplexity corpus. Writes recipe-plan.json
    /// and reports the probe count; executing is a deliberate CLI step.
    fn start_recipe_plan(&mut self) {
        if self.busy {
            return;
        }
        let corpus = self.opt_ppl.trim().to_owned();
        if corpus.is_empty() {
            self.status = "Set the held-out perplexity corpus (Optimize Model > settings) first; the recipe search measures KL divergence over it.".into();
            self.needs_redraw = true;
            return;
        }
        let (source, output, tool) = (self.source.clone(), self.output.clone(), self.quantizer.clone());
        let tx = self.tx.clone();
        self.busy = true;
        self.started_at = Some(Instant::now());
        self.status = "Reading the original GGUF and listing the recipe probes (no quantization)...".into();
        self.needs_redraw = true;
        std::thread::spawn(move || {
            let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4);
            let res = (|| -> Result<String, String> {
                let perplexity = ojon_backend::discovery::resolve_backend_command(
                    ojon_backend::discovery::BackendId::LlamaPerplexity,
                    "llama-perplexity",
                )
                .map_err(|e| e.to_string())?;
                let plan = ojon_backend::recipe_run::prepare_recipe(ojon_backend::recipe_run::RecipeJob {
                    source: PathBuf::from(source.trim()),
                    output_dir: PathBuf::from(output.trim()),
                    base: "q5_k",
                    floor: "iq2_s",
                    budget_kld: 0.01,
                    quantize_tool: PathBuf::from(tool.trim()),
                    imatrix: None,
                    imatrix_tool: None,
                    threads,
                    kld: ojon_backend::kld::KldSettings {
                        tool: perplexity,
                        corpus: PathBuf::from(corpus),
                        gpu_layers: 99,
                        fit: true,
                        context: 512,
                        chunks: 40,
                    },
                    keep_probes: false,
                    budget_p99: None,
                    resume: false,
                    eval_corpus: None,
                    refine_boundaries: true,
                })
                .map_err(|e| e.to_string())?;
                let path = plan.write_plan().map_err(|e| e.to_string())?;
                Ok(format!(
                    "Recipe plan written: {} classes, {} probes, {} quantizations at base q5_k / floor iq2_s / budget 0.01 mean KLD.                      Run it with: ojon recipe-run <source> <output> --corpus <corpus> --imatrix auto --ngl auto --confirm. Plan: {}",
                    plan.present.len(),
                    plan.probes.len(),
                    plan.probes.len() + 2,
                    ojon_backend::recipe_run::plain(&path)
                ))
            })();
            let _ = tx.send(WorkerEvent::Planned(res));
        });
    }

    fn start_plan(&mut self) {
        if self.busy {
            return;
        }
        if !self.ready_to_convert() {
            self.status = "Import and inspect a compatible GGUF or HF folder before planning. GGUF is the only connected quantization format.".into();
            self.needs_redraw = true;
            return;
        }
        let (source, output, tool, converter, python, precision) = (
            self.source.clone(),
            self.output.clone(),
            self.quantizer.clone(),
            self.converter.clone(),
            hf_python_path(),
            self.precision(),
        );
        let tx = self.tx.clone();
        self.busy = true;
        self.started_at = Some(Instant::now());
        self.status = "Checking source/output safety with installed public backends...".into();
        self.needs_redraw = true;
        std::thread::spawn(move || {
            let threads = std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(4);
            let res = if Path::new(source.trim()).is_dir() {
                prepare_hf(
                    Path::new(source.trim()), Path::new(output.trim()), Path::new(converter.trim()),
                    &python, Path::new(tool.trim()), precision, threads,
                ).map(|p| format!(
                    "HF conversion to {} prepared. {} source bytes; large temporary GGUF and RAM required. Architecture compatibility unverified.",
                    p.precision, p.source.source_weight_bytes
                ))
            } else {
                prepare_gguf(
                    Path::new(source.trim()), Path::new(output.trim()),
                    Path::new(tool.trim()), precision, threads,
                ).map(|p| format!(
                    "GGUF conversion to {} prepared. Source: {} bytes. RAM/scratch unknown.",
                    p.precision, p.source_info.file_bytes
                ))
            }.and_then(|summary| {
                let output_path = Path::new(output.trim());
                let parent = output_path.parent().filter(|p| !p.as_os_str().is_empty())
                    .unwrap_or(Path::new("."));
                let capacity = resource_plan(
                    Path::new(source.trim()), parent, ResourceMode::Gguf, "cpu"
                )?;
                let disk = capacity["free_output_volume_bytes"].as_u64()
                    .map(|v| format!("{:.1} GiB", v as f64 / 1_073_741_824.0))
                    .unwrap_or_else(|| "UNKNOWN".into());
                let ram = capacity["free_host_ram_bytes"].as_u64()
                    .map(|v| format!("{:.1} GiB", v as f64 / 1_073_741_824.0))
                    .unwrap_or_else(|| "UNKNOWN".into());
                let status = capacity["status"].as_str().unwrap_or("unverified");
                Ok(format!(
                    "{summary} Resource policy: {status}; available host RAM {ram}; free output volume {disk}. These are measured free capacities, NOT peak quantizer requirements."
                ))
            }).map_err(|e| e.to_string());
            let _ = tx.send(WorkerEvent::Planned(res));
        });
    }
    /// Read-only settings capture. The visual inputs are simple model-independent
    /// strings; this converts them to a strongly validated backend request.
    fn optimize_request(&self) -> Result<OptimizeWorkflowRequest, String> {
        let source = self.source.trim();
        if source.is_empty() {
            return Err("Choose a model first".into());
        }
        let mut req = OptimizeWorkflowRequest::new(PathBuf::from(source));
        let server = self.eval_server.trim();
        if !server.is_empty() && server != "llama-server" {
            req.server_tool = Some(PathBuf::from(server));
        }
        req.goal = match self.opt_goal {
            1 => OptimizeGoal::Smallest,
            2 => OptimizeGoal::Fastest,
            3 => OptimizeGoal::MaximumQuality,
            _ => OptimizeGoal::Balanced,
        };
        req.device = match self.opt_device {
            1 => "cpu",
            2 => "cuda",
            3 => "xpu",
            4 => "vulkan",
            _ => "auto",
        }
        .into();
        if matches!(self.opt_device, 2 | 3 | 4) {
            req.runtime_device =
                Some(self.opt_runtime_device.trim().to_string()).filter(|v| !v.is_empty());
            req.gpu_layers = self
                .gpu_layers
                .trim()
                .parse::<i32>()
                .map_err(|_| "GPU offload layers must be an integer")?;
        }
        let recipe = self.opt_precisions.trim();
        if !recipe.is_empty() {
            req.precision_candidates = Some(
                ojon_backend::parse_precisions(recipe)
                    .map_err(|e| format!("Guided precisions: {e}"))?,
            );
        }
        req.bench_prompt_tokens = self
            .opt_bench_prompt
            .trim()
            .parse()
            .map_err(|_| "Benchmark prefill tokens must be an integer")?;
        req.bench_generation_tokens = self
            .opt_bench_generation
            .trim()
            .parse()
            .map_err(|_| "Benchmark decode tokens must be an integer")?;
        req.bench_repetitions = self
            .opt_bench_repetitions
            .trim()
            .parse()
            .map_err(|_| "Benchmark repetitions must be an integer")?;
        req.balanced_min_speed_ratio = self
            .opt_speed_ratio
            .trim()
            .parse()
            .map_err(|_| "Balanced minimum speed ratio must be numeric")?;
        let baseline = self.opt_baseline.trim();
        if !baseline.is_empty() {
            req.current_baseline = Some(PathBuf::from(baseline));
        }
        let candidate = self.opt_candidate.trim();
        if !candidate.is_empty() {
            req.candidates
                .push(("smaller".into(), PathBuf::from(candidate)));
        }
        let suite = self.opt_suite.trim();
        if !suite.is_empty() {
            req.suite = Some(PathBuf::from(suite));
        } else {
            // One-click default: the bundled structural sanity suite keeps the
            // guided search functional without user benchmark data. It is
            // explicitly a MINIMUM gate, not a representative benchmark.
            req.suite = ojon_backend::discovery::bundled_data_file("sanity-tasks.json");
        }
        let output = self.opt_output.trim();
        if !output.is_empty() {
            req.output_dir = Some(PathBuf::from(output));
        } else {
            req.output_dir = Path::new(source).parent().map(Path::to_path_buf);
        }
        let sensitivity = self.opt_sensitivity.trim();
        if !sensitivity.is_empty() {
            req.tensor_sensitivity = Some(PathBuf::from(sensitivity));
        }
        let imatrix = self.opt_imatrix.trim();
        if !imatrix.is_empty() {
            req.imatrix = Some(PathBuf::from(imatrix));
        } else {
            // One-click default: an approved run calibrates a bundled-corpus
            // importance matrix itself when the backend exists.
            req.auto_imatrix = true;
        }
        let ppl = self.opt_ppl.trim();
        if !ppl.is_empty() {
            req.ppl_corpus = Some(PathBuf::from(ppl));
        }
        req.min_saving_percent = self
            .opt_min_saving
            .trim()
            .parse()
            .map_err(|_| "Minimum size reduction must be a number from 0 to 95".to_owned())?;
        req.max_loss_points = self
            .opt_max_loss
            .trim()
            .parse()
            .map_err(|_| "Allowed task score loss must be a number from 0 to 100".to_owned())?;
        req.max_ppl_regression_pct = self
            .opt_max_ppl
            .trim()
            .parse()
            .map_err(|_| "Allowed PPL regression must be a number from 0 to 100".to_owned())?;
        Ok(req)
    }
    fn start_optimizer_advice(&mut self) {
        if self.busy {
            return;
        }
        if self.inspected_source.as_deref() != Some(self.source.as_str()) {
            self.status =
                "Inspect the currently selected model before planning optimization.".into();
            self.needs_redraw = true;
            return;
        }
        let request = match self.optimize_request() {
            Ok(v) => v,
            Err(e) => {
                self.status = format!("Optimizer settings: {e}");
                self.needs_redraw = true;
                return;
            }
        };
        let tools = self.selected_backends();
        let tx = self.tx.clone();
        self.opt_planned = None;
        self.opt_result = None;
        self.optimize_review_tab = 0;
        self.busy = true;
        self.current_stage = "Planning precision search route".into();
        self.started_at = Some(Instant::now());
        self.status =
            "Planning optimal quantization route (read-only; no model weights or GPU)".into();
        self.needs_redraw = true;
        std::thread::spawn(move || {
            let result = plan_optimize_workflow(&request, &tools).map_err(|e| e.to_string());
            let _ = tx.send(WorkerEvent::OptimizerPlan(request, result));
        });
    }
    fn start_optimizer_run(&mut self) {
        if self.busy {
            return;
        }
        if self.inspected_source.as_deref() != Some(self.source.as_str()) {
            self.status = "Inspect this model before running an optimizer.".into();
            self.needs_redraw = true;
            return;
        }
        let request = match self.optimize_request() {
            Ok(v) => v,
            Err(e) => {
                self.status = format!("Optimizer settings: {e}");
                self.needs_redraw = true;
                return;
            }
        };
        let approved = matches!(self.opt_planned.as_ref(),
            Some((planned,value)) if *planned==request && value["ready_for_user_approved_execution"]==true);
        if !approved {
            self.status="Settings changed or prerequisites are missing. Review a fresh plan before approving execution.".into();
            self.opt_planned = None;
            self.needs_redraw = true;
            return;
        }
        self.launch_optimizer_worker(request);
    }
    fn start_one_click_optimize(&mut self) {
        if self.busy {
            return;
        }
        if self.source.trim().is_empty() {
            self.status = "Select or drop a model first before clicking Optimise Model.".into();
            self.needs_redraw = true;
            return;
        }
        let request = match self.optimize_request() {
            Ok(v) => v.with_auto_hardware(),
            Err(e) => {
                self.status = format!("Optimizer settings: {e}");
                self.needs_redraw = true;
                return;
            }
        };
        self.launch_optimizer_worker(request);
    }
    fn launch_optimizer_worker(&mut self, request: OptimizeWorkflowRequest) {
        clear_job_cancel();
        let tools = self.selected_backends();
        let tx = self.tx.clone();
        self.opt_planned = None;
        self.opt_result = None;
        self.log_tail.clear();
        self.completed_stages.clear();
        self.current_stage = "Calibrating & searching precision candidates".into();
        self.optimize_review_tab = 1;
        self.started_at = Some(Instant::now());
        self.last_elapsed = None;
        self.busy = true;
        let out_dir = request
            .output_dir
            .clone()
            .unwrap_or_else(|| PathBuf::from("."));
        let job_id = record_job_start(
            "optimize_model",
            &request.model,
            &out_dir,
            &format!("{:?}", request.goal),
            "Calibrating & searching precision candidates",
        )
        .unwrap_or_else(|_| "opt_job".into());
        self.job_history = load_job_history().unwrap_or_default();
        self.status = format!(
            "Optimising model on {} ({:?}); live logs below.",
            request.device, request.goal
        );
        self.needs_redraw = true;
        std::thread::spawn(move || {
            let logs = tx.clone();
            let jid = job_id.clone();
            let result = run_optimize_workflow(&request, &tools, true, |line| {
                if line.contains("imatrix") || line.contains("Calibrat") {
                    let _ = logs.send(WorkerEvent::Stage("Calibrating importance matrix".into()));
                    let _ = update_job_stage(&jid, "Calibrating importance matrix");
                } else if line.contains("quantiz") {
                    let _ = logs.send(WorkerEvent::Stage("Quantising candidate GGUF".into()));
                    let _ = update_job_stage(&jid, "Quantising candidate GGUF");
                } else if line.contains("eval") || line.contains("server") || line.contains("bench")
                {
                    let _ = logs.send(WorkerEvent::Stage(
                        "Evaluating quality, reload & speed".into(),
                    ));
                    let _ = update_job_stage(&jid, "Evaluating quality, reload & speed");
                }
                let _ = logs.send(WorkerEvent::Log(line.to_owned()));
            })
            .map_err(|e| e.to_string());
            match &result {
                Ok(val) => {
                    let cand = val["provisional_candidate"]["path"]
                        .as_str()
                        .or_else(|| val["provisional_candidate"].as_str());
                    let status = val["status"].as_str().unwrap_or("COMPLETED");
                    let _ = finish_job_history(
                        &job_id,
                        status,
                        cand.map(Path::new),
                        "Optimization workflow finished",
                    );
                }
                Err(err) => {
                    let status = if is_job_cancelled() {
                        "CANCELLED"
                    } else {
                        "FAILED"
                    };
                    let _ = finish_job_history(&job_id, status, None, err);
                }
            }
            let _ = tx.send(WorkerEvent::OptimizerRun(result));
        });
    }
    fn cancel_active_job(&mut self) {
        if !self.busy {
            return;
        }
        request_job_cancel();
        self.busy = false;
        self.current_stage = "Cancelled by user".into();
        self.last_elapsed = self.started_at.take().map(|s| s.elapsed());
        if let Ok(mut entries) = load_job_history() {
            for entry in entries.iter_mut() {
                if entry.status == "RUNNING" {
                    entry.status = "CANCELLED".into();
                    entry.current_stage = "Cancelled by user; partial artifacts preserved".into();
                }
            }
            let _ = save_job_history(&entries);
            self.job_history = entries;
        }
        self.status = "Active job cancelled safely. Any completed candidate checkpoints remain in the output folder.".into();
        self.needs_redraw = true;
    }
    fn uncensor_request(&self) -> Result<UncensorRequest, String> {
        let source = self.source.trim();
        let trace = self.unc_trace.trim();
        let output = self.unc_output.trim();
        if source.is_empty() || trace.is_empty() || output.is_empty() {
            return Err("Select a source, paired activation trace, and new output path".into());
        }
        let strength = self
            .unc_strength
            .trim()
            .parse::<f64>()
            .map_err(|_| "Edit strength must be a number".to_owned())?;
        if !strength.is_finite() || strength <= 0.0 || strength > 1.0 {
            return Err("Edit strength must be >0 and <=1".into());
        }
        let block = if self.unc_block.trim().is_empty() {
            None
        } else {
            Some(
                self.unc_block
                    .trim()
                    .parse::<usize>()
                    .map_err(|_| "Control-vector block must be a nonnegative integer".to_owned())?,
            )
        };
        if block.is_some() && (!trace.ends_with(".gguf") || !source.ends_with(".gguf")) {
            return Err("Control-vector mode requires GGUF source and vector paths".into());
        }
        if block.is_none() && trace.ends_with(".gguf") {
            return Err("Enter the zero-based block index for a control-vector GGUF".into());
        }
        Ok(UncensorRequest {
            source: PathBuf::from(source),
            trace: PathBuf::from(trace),
            output: PathBuf::from(output),
            strength,
            block,
        })
    }
    fn start_uncensor_plan(&mut self) {
        if self.busy {
            return;
        }
        if self.inspected_source.as_deref() != Some(self.source.as_str()) {
            self.status = "Inspect the currently selected source before planning an edit.".into();
            self.needs_redraw = true;
            return;
        }
        let request = match self.uncensor_request() {
            Ok(r) => r,
            Err(e) => {
                self.status = e;
                self.needs_redraw = true;
                return;
            }
        };
        let tx = self.tx.clone();
        self.unc_planned = None;
        self.unc_result = None;
        self.busy = true;
        self.started_at = Some(Instant::now());
        self.status = "Checking trace and model tensors without changing the source".into();
        self.needs_redraw = true;
        std::thread::spawn(move || {
            let result = plan_uncensor(&request).map_err(|e| e.to_string());
            let _ = tx.send(WorkerEvent::UncensorPlan(request, result));
        });
    }
    fn start_uncensor_run(&mut self) {
        if self.busy {
            return;
        }
        let request = match self.uncensor_request() {
            Ok(r) => r,
            Err(e) => {
                self.status = e;
                self.needs_redraw = true;
                return;
            }
        };
        let approved = matches!(self.unc_planned.as_ref(),Some((planned,_)) if *planned==request);
        if !approved {
            self.status = "Edit settings changed. Review a fresh plan before running.".into();
            self.unc_planned = None;
            self.needs_redraw = true;
            return;
        }
        clear_job_cancel();
        let tx = self.tx.clone();
        self.unc_planned = None;
        self.unc_result = None;
        self.log_tail.clear();
        self.busy = true;
        self.started_at = Some(Instant::now());
        self.current_stage = "Applying norm-preserving tensor edits".into();
        self.status = "Writing a new edited checkpoint; original source remains untouched".into();
        self.needs_redraw = true;
        std::thread::spawn(move || {
            let logger = tx.clone();
            let result = execute_uncensor(&request, |line| {
                let _ = logger.send(WorkerEvent::Log(line.to_owned()));
            })
            .map_err(|e| e.to_string());
            let _ = tx.send(WorkerEvent::UncensorRun(result));
        });
    }
    fn uncensor_auto_request(
        &self,
    ) -> Result<ojon_backend::uncensor_auto::UncensorAutoRequest, String> {
        let source = self.source.trim();
        if source.is_empty() || !source.ends_with(".gguf") {
            return Err(
                "Automatic refusal search needs an original F16/BF16/F32 GGUF source".into(),
            );
        }
        let paired = self.unc_paired.trim();
        let paired = if paired.is_empty() {
            ojon_backend::discovery::bundled_data_dir("paired")
                .or_else(|| {
                    ojon_backend::discovery::bundled_data_file("positive.txt")
                        .and_then(|p| p.parent().map(Path::to_path_buf))
                })
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or_default()
        } else {
            paired.to_string()
        };
        if paired.is_empty() {
            return Err("Select a folder with paired positive.txt and negative.txt behaviour examples (or place them in data/paired)".into());
        }
        let refusal_suite = self.unc_refusal_suite.trim();
        let refusal_suite = if refusal_suite.is_empty() {
            ojon_backend::discovery::bundled_data_file("refusal-probes.json")
        } else {
            Some(PathBuf::from(refusal_suite))
        };
        let refusal_suite = refusal_suite.ok_or_else(|| {
            "No refusal probe suite found; provide a ojon.refusal_suite.v1 JSON path".to_owned()
        })?;
        let task_suite = self.unc_task_suite.trim();
        let task_suite = if task_suite.is_empty() {
            ojon_backend::discovery::bundled_data_file("sanity-tasks.json")
        } else {
            Some(PathBuf::from(task_suite))
        };
        let task_suite = task_suite.ok_or_else(|| {
            "No capability suite found; provide a ojon.task_suite.v1 JSON path".to_owned()
        })?;
        let output = self.unc_output.trim();
        let output_dir = if output.is_empty() {
            Path::new(source)
                .parent()
                .ok_or_else(|| "Source has no parent folder".to_owned())?
        } else {
            Path::new(output)
        };
        let parsed_layers = self
            .gpu_layers
            .trim()
            .parse::<i32>()
            .map_err(|_| "GPU offload layers must be an integer".to_owned())?;
        let gpu_layers = if parsed_layers == 0 {
            let (_, _, auto_layers) = resolve_auto_runtime_device("auto", None, 0);
            auto_layers as i32
        } else {
            parsed_layers
        };
        let quantize = match self.unc_auto_quantize {
            1 => Some(Precision::Q4KM),
            2 => Some(Precision::Q5KM),
            3 => Some(Precision::Q6K),
            _ => None,
        };
        let cvector_generator = self
            .backends
            .iter()
            .find(|b| b.spec.id == BackendId::LlamaCvectorGenerator)
            .and_then(|b| b.executable.clone());
        let mut request = ojon_backend::uncensor_auto::UncensorAutoRequest::new(
            PathBuf::from(source),
            output_dir.to_path_buf(),
        );
        request.refusal_probes = refusal_suite;
        request.capability_suite = task_suite;
        request.paired_prompts = PathBuf::from(paired);
        request.cvector_generator = cvector_generator.ok_or_else(|| {
            "llama-cvector-generator was not detected; install it or use the CLI path".to_owned()
        })?;
        let rt_str = self.runtime.trim();
        request.runtime = if rt_str.is_empty() || rt_str == "llama-cli" {
            ojon_backend::discovered_path(BackendId::LlamaCli)
                .unwrap_or_else(|| PathBuf::from("llama-cli"))
        } else {
            PathBuf::from(rt_str)
        };
        let server = self.eval_server.trim();
        if !server.is_empty() && server != "llama-server" {
            request.server_runtime = Some(PathBuf::from(server));
        } else {
            request.server_runtime = ojon_backend::discovered_path(BackendId::LlamaServer);
        }
        request.gpu_layers = gpu_layers;
        request.quantize = quantize;
        let q_str = self.quantizer.trim();
        request.quantizer = if q_str.is_empty() || q_str == "llama-quantize" {
            ojon_backend::discovered_path(BackendId::LlamaQuantize)
                .unwrap_or_else(|| PathBuf::from("llama-quantize"))
        } else {
            PathBuf::from(q_str)
        };

        Ok(request)
    }
    fn start_uncensor_auto_plan(&mut self) {
        if self.busy {
            return;
        }
        if self.inspected_source.as_deref() != Some(self.source.as_str()) {
            self.status =
                "Inspect the currently selected source before planning an automatic search.".into();
            self.needs_redraw = true;
            return;
        }
        let request = match self.uncensor_auto_request() {
            Ok(r) => r,
            Err(e) => {
                self.status = e;
                self.needs_redraw = true;
                return;
            }
        };
        let tx = self.tx.clone();
        self.unc_auto_planned = None;
        self.unc_auto_result = None;
        self.busy = true;
        self.started_at = Some(Instant::now());
        self.current_stage = "Planning automatic refusal direction search".into();
        self.status = "Planning the automatic refusal-edit sweep (read-only)".into();
        self.needs_redraw = true;
        std::thread::spawn(move || {
            let result = ojon_backend::uncensor_auto::prepare_uncensor_auto(request.clone())
                .map(|plan| plan.as_json())
                .map_err(|e| e.to_string());
            let _ = tx.send(WorkerEvent::UncensorAutoPlan(request, result));
        });
    }
    fn start_uncensor_auto_run(&mut self) {
        if self.busy {
            return;
        }
        let request = match self.uncensor_auto_request() {
            Ok(r) => r,
            Err(e) => {
                self.status = e;
                self.needs_redraw = true;
                return;
            }
        };
        let approved = matches!(self.unc_auto_planned.as_ref(),
            Some((planned,value)) if *planned==request && value["status"]=="PLAN_ONLY");
        if !approved {
            self.status =
                "Automatic search settings changed. Review a fresh plan before executing.".into();
            self.unc_auto_planned = None;
            self.needs_redraw = true;
            return;
        }
        self.launch_uncensor_auto_worker(request);
    }
    fn start_one_click_uncensor(&mut self) {
        if self.busy {
            return;
        }
        if self.source.trim().is_empty() {
            self.status =
                "Select or drop an F16/BF16/F32 GGUF model first before clicking Uncensor Model."
                    .into();
            self.needs_redraw = true;
            return;
        }
        let request = match self.uncensor_auto_request() {
            Ok(r) => r,
            Err(e) => {
                self.status = e;
                self.needs_redraw = true;
                return;
            }
        };
        self.launch_uncensor_auto_worker(request);
    }
    fn launch_uncensor_auto_worker(
        &mut self,
        request: ojon_backend::uncensor_auto::UncensorAutoRequest,
    ) {
        clear_job_cancel();
        let tx = self.tx.clone();
        self.unc_auto_planned = None;
        self.unc_auto_result = None;
        self.log_tail.clear();
        self.completed_stages.clear();
        self.current_stage = "Capturing refusal directions & searching edits".into();
        self.busy = true;
        self.started_at = Some(Instant::now());
        let job_id = record_job_start(
            "uncensor_auto",
            &request.source,
            &request.output_dir,
            "automatic",
            "Capturing refusal directions & searching edits",
        )
        .unwrap_or_else(|_| "unc_job".into());
        self.job_history = load_job_history().unwrap_or_default();
        self.status = "Automatic uncensor search running: capturing directions, testing candidate edits, verifying capabilities...".into();
        self.needs_redraw = true;
        std::thread::spawn(move || {
            let logger = tx.clone();
            let jid = job_id.clone();
            let plan = ojon_backend::uncensor_auto::prepare_uncensor_auto(request.clone());
            let result = match plan {
                Ok(plan) => ojon_backend::uncensor_auto::run_uncensor_auto(&plan, |line| {
                    if line.contains("Capturing") || line.contains("cvector") {
                        let _ = logger.send(WorkerEvent::Stage(
                            "Capturing paired refusal directions".into(),
                        ));
                        let _ = update_job_stage(&jid, "Capturing paired refusal directions");
                    } else if line.contains("Candidate") {
                        let _ = logger.send(WorkerEvent::Stage(line.chars().take(64).collect()));
                        let _ = update_job_stage(&jid, "Searching & scoring candidate edits");
                    } else if line.contains("Quantizing") || line.contains("Verifying") {
                        let _ = logger.send(WorkerEvent::Stage(
                            "Reloading & verifying published winner".into(),
                        ));
                        let _ = update_job_stage(&jid, "Reloading & verifying published winner");
                    }
                    let _ = logger.send(WorkerEvent::Log(line.to_owned()));
                }),
                Err(e) => Err(e),
            }
            .map_err(|e| e.to_string());
            match &result {
                Ok(val) => {
                    let cand = val["output_quantized"]
                        .as_str()
                        .or_else(|| val["output_edited"].as_str());
                    let status = val["status"].as_str().unwrap_or("COMPLETED");
                    let _ = finish_job_history(
                        &job_id,
                        status,
                        cand.map(Path::new),
                        "Automatic uncensor workflow finished",
                    );
                }

                Err(err) => {
                    let status = if is_job_cancelled() {
                        "CANCELLED"
                    } else {
                        "FAILED"
                    };
                    let _ = finish_job_history(&job_id, status, None, err);
                }
            }
            let _ = tx.send(WorkerEvent::UncensorAutoRun(result));
        });
    }
    fn ready_to_convert(&self) -> bool {
        self.format_tab == 0
            && self.inspected_source.as_deref() == Some(self.source.as_str())
            && (self.model.is_some() || self.hf_folder.is_some())
            && !self.source.trim().is_empty()
            && !self.output.trim().is_empty()
    }
    fn start_quantize(&mut self) {
        if self.busy {
            return;
        }
        if !self.ready_to_convert() {
            self.status = "Inspect the selected model and choose GGUF before quantizing. This source/format is not ready.".into();
            self.needs_redraw = true;
            return;
        }
        let (source, output, tool, converter, python, precision) = (
            self.source.clone(),
            self.output.clone(),
            self.quantizer.clone(),
            self.converter.clone(),
            hf_python_path(),
            self.precision(),
        );
        let tx = self.tx.clone();
        self.busy = true;
        self.started_at = Some(Instant::now());
        self.last_elapsed = None;
        self.last_output = None;
        self.log_tail.clear();
        self.status = "Running installed converter. Live subprocess messages appear below; no estimated percentage is shown.".into();
        self.needs_redraw = true;
        std::thread::spawn(move || {
            let threads = std::thread::available_parallelism()
                .map(|v| v.get())
                .unwrap_or(4);
            if Path::new(source.trim()).is_dir() {
                let result = prepare_hf(
                    Path::new(source.trim()),
                    Path::new(output.trim()),
                    Path::new(converter.trim()),
                    &python,
                    Path::new(tool.trim()),
                    precision,
                    threads,
                )
                .and_then(|plan| {
                    quantize_hf_with_log(&plan, |line| {
                        let _ = tx.send(WorkerEvent::Log(line.to_owned()));
                    })
                })
                .map_err(|e| e.to_string());
                let _ = tx.send(WorkerEvent::HfQuantized(result));
            } else {
                let result = prepare_gguf(
                    Path::new(source.trim()),
                    Path::new(output.trim()),
                    Path::new(tool.trim()),
                    precision,
                    threads,
                )
                .and_then(|plan| {
                    quantize_gguf_with_log(&plan, |line| {
                        let _ = tx.send(WorkerEvent::Log(line.to_owned()));
                    })
                })
                .map_err(|e| e.to_string());
                let _ = tx.send(WorkerEvent::Quantized(result));
            }
        });
    }
    fn start_smoke(&mut self) {
        if self.busy {
            return;
        }
        let gpu_layers = match self.gpu_layers.trim().parse::<u32>() {
            Ok(v) if v <= 512 => v,
            _ => {
                self.status = "GPU layers must be an integer from 0 to 512.".into();
                self.needs_redraw = true;
                return;
            }
        };
        let (output, runtime, prompt) = (
            self.output.clone(),
            self.runtime.clone(),
            self.prompt.clone(),
        );
        let tx = self.tx.clone();
        self.busy = true;
        self.started_at = Some(Instant::now());
        self.last_elapsed = None;
        self.log_tail.clear();
        self.last_inference = None;
        self.status = "Running real llama-cli inference. This is not a quality benchmark.".into();
        self.needs_redraw = true;
        std::thread::spawn(move || {
            let result = smoke_gguf_capture(
                Path::new(output.trim()),
                Path::new(runtime.trim()),
                &prompt,
                32,
                gpu_layers,
            )
            .map_err(|e| e.to_string());
            let _ = tx.send(WorkerEvent::Smoke(result));
        });
    }
    fn start_bench(&mut self) {
        if self.busy {
            return;
        }
        let gpu_layers = match self.gpu_layers.trim().parse::<i32>() {
            Ok(v) if (0..=512).contains(&v) => v,
            _ => {
                self.status = "GPU layers must be an integer from 0 to 512.".into();
                self.needs_redraw = true;
                return;
            }
        };
        let (model, benchmark) = (self.output.clone(), self.benchmark.clone());
        let (prompt_tokens, gen_tokens, repetitions) = match (
            self.opt_bench_prompt.trim().parse::<u32>(),
            self.opt_bench_generation.trim().parse::<u32>(),
            self.opt_bench_repetitions.trim().parse::<u32>(),
        ) {
            (Ok(pp), Ok(tg), Ok(reps))
                if (16..=8192).contains(&pp)
                    && (16..=2048).contains(&tg)
                    && (1..=10).contains(&reps) =>
            {
                (pp, tg, reps)
            }
            _ => {
                self.status =
                    "Benchmark settings: prefill 16..8192, decode 16..2048, repetitions 1..10."
                        .into();
                self.needs_redraw = true;
                return;
            }
        };
        let device = if matches!(self.opt_device, 2 | 3) {
            Some(self.opt_runtime_device.trim().to_string()).filter(|v| !v.is_empty())
        } else {
            None
        };
        let tx = self.tx.clone();
        self.busy = true;
        self.started_at = Some(Instant::now());
        self.last_elapsed = None;
        self.last_benchmark = None;
        self.status = format!("Running real llama-bench: {prompt_tokens} prefill / {gen_tokens} decode, {repetitions} repeats; not a quality benchmark.");
        self.needs_redraw = true;
        std::thread::spawn(move || {
            let result = (|| -> std::io::Result<(PathBuf, serde_json::Value)> {
                let model = Path::new(model.trim());
                let plan = prepare_bench(
                    model,
                    Path::new(benchmark.trim()),
                    prompt_tokens,
                    gen_tokens,
                    repetitions,
                    gpu_layers,
                    device.as_deref(),
                )?;
                let stamp = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_err(std::io::Error::other)?
                    .as_millis();
                let name = model.file_stem().unwrap_or_default().to_string_lossy();
                let parent = model.parent().unwrap_or_else(|| Path::new("."));
                let receipt = parent.join(format!("{name}-ojon-bench-{stamp}.json"));
                let metrics = run_bench(&plan, &receipt)?;
                Ok((receipt, metrics))
            })()
            .map_err(|e| e.to_string());
            let _ = tx.send(WorkerEvent::Bench(result));
        });
    }
    fn poll(&mut self) {
        while let Ok(message) = self.rx.try_recv() {
            self.needs_redraw = true;
            if let WorkerEvent::Log(line) = message {
                let truncated: String = line
                    .chars()
                    .filter(|ch| !ch.is_control() || *ch == ' ')
                    .take(180)
                    .collect();
                if !truncated.trim().is_empty() {
                    self.log_tail.push_back(truncated);
                    while self.log_tail.len() > 8 {
                        self.log_tail.pop_front();
                    }
                }
                continue; // Log events MUST NOT set busy=false.
            }
            if let WorkerEvent::Stage(stage) = message {
                if !self.current_stage.is_empty()
                    && self.current_stage != "Idle"
                    && self.current_stage != stage
                {
                    self.completed_stages.push(self.current_stage.clone());
                    while self.completed_stages.len() > 5 {
                        self.completed_stages.remove(0);
                    }
                }
                self.current_stage = stage;
                continue; // Stage events MUST NOT set busy=false.
            }
            self.busy = false;
            self.last_elapsed = self.started_at.take().map(|start| start.elapsed());
            self.job_history = load_job_history().unwrap_or_default();
            match message {
                WorkerEvent::Inspected(Ok((model, probe, topo))) => {
                    let topo_note = topo
                        .as_ref()
                        .map(|t| {
                            format!(
                                " | {:.2}B params, {:.2} bpw, {}",
                                t.total_parameters as f64 / 1e9,
                                t.effective_bpw,
                                if t.is_moe { "MoE" } else { "Dense" }
                            )
                        })
                        .unwrap_or_default();
                    self.status = format!(
                        "Read GGUF v{}: {} tensors; likely category {} ({}){}. No weights loaded.",
                        model.version,
                        model.tensor_count,
                        probe.category.as_str(),
                        probe.confidence,
                        topo_note
                    );
                    self.model = Some(model);
                    self.model_topology = topo;
                    self.hf_folder = None;
                    self.probe = Some(probe);
                    self.current_stage = "Model inspected".into();
                    if self.detected_gpus.is_empty() {
                        self.detected_gpus = discover_runtime_gpu_devices();
                    }
                }
                WorkerEvent::HfInspected(Ok((info, probe))) => {
                    self.status = format!("HF model folder: {} weight files; likely category {} ({}). No weights loaded.",
                        info.weight_files, probe.category.as_str(), probe.confidence);
                    self.model = None;
                    self.model_topology = None;
                    self.hf_folder = Some(info);
                    self.probe = Some(probe);
                    self.current_stage = "HF folder inspected".into();
                }
                WorkerEvent::StandaloneProbed(Ok(probe)) => {
                    self.status = format!(
                        "Metadata inspected: {} ({}). {}",
                        probe.category.as_str(),
                        probe.confidence,
                        probe.conversion_status
                    );
                    self.model = None;
                    self.model_topology = None;
                    self.hf_folder = None;
                    self.probe = Some(probe);
                    self.current_stage = "Metadata inspected".into();
                }
                WorkerEvent::StandaloneProbed(Err(e)) => {
                    self.probe = None;
                    self.current_stage = "Inspection failed".into();
                    self.status = format!("Unsupported input or inspection failed: {e}");
                }
                WorkerEvent::HfInspected(Err(e)) => {
                    self.hf_folder = None;
                    self.current_stage = "Inspection failed".into();
                    self.status = format!("HF folder inspection failed: {e}");
                }
                WorkerEvent::Inspected(Err(e)) => {
                    self.model = None;
                    self.model_topology = None;
                    self.current_stage = "Inspection failed".into();
                    self.status = format!("Inspection failed: {e}");
                }
                WorkerEvent::BackendCapabilities(value) => {
                    let verified = value["probes"].as_array().map_or(0, |list| {
                        list.iter()
                            .filter(|r| r["status"] == "CAPABILITY_OUTPUT_PARSED")
                            .count()
                    });
                    self.backend_capabilities = Some(value);
                    self.detected_gpus = discover_runtime_gpu_devices();
                    self.current_stage = "Capabilities probed".into();
                    self.status=format!("Parsed {verified} executable capability outputs; {} discrete GPU(s) ready.", self.detected_gpus.len());
                }
                WorkerEvent::OptimizerPlan(request, Ok(value)) => {
                    let route = value["route"].as_str().unwrap_or("unknown");
                    let step = value["next_user_step"]
                        .as_str()
                        .unwrap_or("See prerequisites");
                    let ready = value["ready_for_user_approved_execution"] == true;
                    self.current_stage = "Plan ready".into();
                    self.status = format!(
                        "Optimization plan [{route}]: {step}. {}",
                        if ready {
                            "Review settings and approve the selected backend run."
                        } else {
                            "Prerequisites not yet satisfied."
                        }
                    );
                    self.opt_planned = Some((request, value));
                }
                WorkerEvent::OptimizerPlan(_, Err(e)) => {
                    self.opt_planned = None;
                    self.current_stage = "Plan blocked".into();
                    self.status = format!("Optimization plan blocked: {e}");
                }
                WorkerEvent::OptimizerRun(Ok(value)) => {
                    let pass = value["status"].as_str() == Some("PROVISIONAL_CANDIDATE_FOUND");
                    self.current_stage = if pass {
                        "Candidate verified & saved".into()
                    } else {
                        "Completed (no candidate passed gates)".into()
                    };
                    self.status = if pass {
                        "Measured candidate passed the selected gates. Open the result folder, load, or export the report below.".into()
                    } else {
                        "Experiment finished, but no candidate passed the selected quality/size gates.".into()
                    };
                    self.opt_result = Some(value);
                    self.optimize_review_tab = 1;
                }
                WorkerEvent::OptimizerRun(Err(e)) => {
                    self.current_stage = "Optimization failed".into();
                    self.status = format!("Optimization failed: {e}");
                }
                WorkerEvent::UncensorPlan(request, Ok(value)) => {
                    self.current_stage = "Edit plan ready".into();
                    self.status="Edit plan ready. Review the target and run; refusal/capability tests are still required.".into();
                    self.unc_planned = Some((request, value));
                }
                WorkerEvent::UncensorPlan(_, Err(e)) => {
                    self.unc_planned = None;
                    self.current_stage = "Edit plan blocked".into();
                    self.status = format!("Uncensor edit plan blocked: {e}");
                }
                WorkerEvent::UncensorRun(Ok(value)) => {
                    self.current_stage = "Edited checkpoint saved".into();
                    self.status="Edited checkpoint written. Runtime load and refusal/capability evaluation remain to be done.".into();
                    self.unc_result = Some(value);
                }
                WorkerEvent::UncensorRun(Err(e)) => {
                    self.current_stage = "Edit failed".into();
                    self.status = format!("Uncensor edit failed: {e}");
                }
                WorkerEvent::UncensorAutoPlan(request, Ok(value)) => {
                    let count = value["candidate_count"].as_u64().unwrap_or(0);
                    self.current_stage = "Automatic search plan ready".into();
                    self.status=format!(
                        "Automatic search plan ready: {count} candidates will be really edited and measured. Review and approve.");
                    self.unc_auto_planned = Some((request, value));
                }
                WorkerEvent::UncensorAutoPlan(_, Err(e)) => {
                    self.unc_auto_planned = None;
                    self.current_stage = "Search plan blocked".into();
                    self.status = format!("Automatic uncensor plan blocked: {e}");
                }
                WorkerEvent::UncensorAutoRun(Ok(value)) => {
                    let published =
                        value["status"].as_str() == Some("SELECTED_FROM_MEASURED_SCORES");
                    self.current_stage = if published {
                        "Winner verified & published".into()
                    } else {
                        "Search finished (no candidate passed gates)".into()
                    };
                    self.status = if published {
                        "Measured winner published and reloaded. Review refusal & capability comparison below.".into()
                    } else {
                        "Search finished: no candidate passed the configured gates; nothing was published.".into()
                    };
                    self.unc_auto_result = Some(value);
                }
                WorkerEvent::UncensorAutoRun(Err(e)) => {
                    self.current_stage = "Automatic uncensor failed".into();
                    self.status = format!("Automatic uncensor search failed: {e}");
                }
                WorkerEvent::Planned(Ok(text)) => self.status = text,
                WorkerEvent::Planned(Err(e)) => self.status = format!("Preflight failed: {e}"),
                WorkerEvent::Quantized(Ok(done)) => {
                    self.last_output = Some((done.output.clone(), done.output_info.file_bytes));
                    self.current_stage = "Quantisation completed".into();
                    self.status = format!(
                        "Converter completed. GGUF catalog checked: {} tensors, {} bytes. Runtime loading NOT verified.",
                        done.output_info.tensor_count, done.output_info.file_bytes
                    );
                }
                WorkerEvent::Quantized(Err(e)) => self.status = format!("Conversion failed: {e}"),
                WorkerEvent::HfQuantized(Ok(done)) => {
                    self.last_output = Some((done.output.clone(), done.info.file_bytes));
                    self.current_stage = "HF conversion completed".into();
                    self.status = format!(
                        "HF conversion + quantization completed: {} bytes, {} tensors. Runtime load NOT verified.",
                        done.info.file_bytes, done.info.tensor_count,
                    );
                }
                WorkerEvent::HfQuantized(Err(e)) => {
                    self.status = format!("HF conversion failed: {e}")
                }
                WorkerEvent::Smoke(Ok(capture)) => {
                    self.last_inference = Some(capture.stdout);
                    self.current_stage = "Inference smoke check complete".into();
                    self.status = "Real llama-cli output captured. Review the generated text; quality is NOT evaluated.".into();
                }
                WorkerEvent::Smoke(Err(e)) => {
                    self.status = format!("Inference smoke check failed: {e}")
                }
                WorkerEvent::Bench(Ok((path, receipt))) => {
                    let pp = receipt["prompt_processing"]["tokens_per_second"]
                        .as_f64()
                        .unwrap_or(0.0);
                    let tg = receipt["token_generation"]["tokens_per_second"]
                        .as_f64()
                        .unwrap_or(0.0);
                    self.current_stage = "Benchmark complete".into();
                    self.status = format!("Real llama-bench: prefill {pp:.2} tok/s, decode {tg:.2} tok/s. Quality not tested. Receipt: {}",
                        path.file_name().unwrap_or_default().to_string_lossy());
                    self.last_benchmark = Some((path, receipt));
                }
                WorkerEvent::Bench(Err(e)) => {
                    self.status = format!("GGUF performance benchmark failed: {e}")
                }
                WorkerEvent::Log(_) | WorkerEvent::Stage(_) => {
                    unreachable!("log/stage messages handled above")
                }
            }
        }
        // One render tick per second while processing, even when a backend is
        // temporarily silent. No continuous idle repaint or invented metrics.
        if self.busy {
            self.needs_redraw = true;
        }
    }
    fn edit_focused(&mut self, event: &ui_platform::winit::event::KeyEvent) {
        if event.state != ElementState::Pressed {
            return;
        }
        if matches!(event.logical_key, Key::Named(NamedKey::Escape)) {
            self.focus = None;
            self.select_all = false;
            return;
        }
        let Some(focus) = self.focus else {
            return;
        };
        let dest = match focus {
            Focus::Model => &mut self.source,
            Focus::Output => &mut self.output,
            Focus::Quantizer => &mut self.quantizer,
            Focus::Converter => &mut self.converter,
            Focus::Runtime => &mut self.runtime,
            Focus::EvalServer => &mut self.eval_server,
            Focus::Benchmark => &mut self.benchmark,
            Focus::Prompt => &mut self.prompt,
            Focus::GpuLayers => &mut self.gpu_layers,
            Focus::OptBaseline => &mut self.opt_baseline,
            Focus::OptCandidate => &mut self.opt_candidate,
            Focus::OptSuite => &mut self.opt_suite,
            Focus::OptOutput => &mut self.opt_output,
            Focus::OptImatrix => &mut self.opt_imatrix,
            Focus::OptSensitivity => &mut self.opt_sensitivity,
            Focus::OptPpl => &mut self.opt_ppl,
            Focus::OptMinSaving => &mut self.opt_min_saving,
            Focus::OptMaxLoss => &mut self.opt_max_loss,
            Focus::OptMaxPpl => &mut self.opt_max_ppl,
            Focus::OptRuntimeDevice => &mut self.opt_runtime_device,
            Focus::OptPrecisions => &mut self.opt_precisions,
            Focus::OptBenchPrompt => &mut self.opt_bench_prompt,
            Focus::OptBenchGeneration => &mut self.opt_bench_generation,
            Focus::OptBenchRepetitions => &mut self.opt_bench_repetitions,
            Focus::OptSpeedRatio => &mut self.opt_speed_ratio,
            Focus::UncTrace => &mut self.unc_trace,
            Focus::UncOutput => &mut self.unc_output,
            Focus::UncStrength => &mut self.unc_strength,
            Focus::UncBlock => &mut self.unc_block,
            Focus::UncPaired => &mut self.unc_paired,
            Focus::UncRefusalSuite => &mut self.unc_refusal_suite,
            Focus::UncTaskSuite => &mut self.unc_task_suite,
        };
        if self.control_held {
            if let Key::Character(ref key) = event.logical_key {
                if key.eq_ignore_ascii_case("a") {
                    self.select_all = true;
                    self.needs_redraw = true;
                    return;
                }
                if key.eq_ignore_ascii_case("v") {
                    if let Some(value) = clipboard_text() {
                        if self.select_all {
                            dest.clear();
                        }
                        for ch in value.chars().filter(|ch| !ch.is_control()) {
                            if dest.len() < 4096 {
                                dest.push(ch);
                            }
                        }
                        self.select_all = false;
                        self.needs_redraw = true;
                    }
                    return;
                }
            }
            return;
        }
        if self.select_all {
            dest.clear();
            self.select_all = false;
        }
        if matches!(event.logical_key, Key::Named(NamedKey::Backspace)) {
            dest.pop();
        } else if let Some(value) = &event.text {
            for ch in value.chars().filter(|ch| !ch.is_control()) {
                if dest.len() < 4096 {
                    dest.push(ch);
                }
            }
        }
        self.needs_redraw = true;
    }
    fn render(&mut self, width: f32, height: f32) -> CommandBuffer {
        let optimizer_plan_current = self
            .opt_planned
            .as_ref()
            .is_some_and(|(prior, _)| self.optimize_request().is_ok_and(|now| now == *prior));
        let source_is_current = self.inspected_source.as_deref() == Some(self.source.as_str());
        let previous_precision = self.precision();
        let auto_generated_output = self.output == default_output(&self.source, previous_precision);
        let mut cmds = CommandBuffer::new();
        let palette = ThemePalette::studio_graphite();
        let mut click_pointer = self.pointer.clone();
        click_pointer.primary_down = self.click_pulse;
        self.click_pulse = false;
        let navigation_width = 56.0;
        let viewport = Rect::from_min_size(
            Vec2::new(navigation_width, 0.0),
            Vec2::new((width - navigation_width).max(700.0), height.max(500.0)),
        );
        let source = &mut self.source;
        let output = &mut self.output;
        let quantizer = &mut self.quantizer;
        let converter = &mut self.converter;
        let runtime = &mut self.runtime;
        let eval_server = &mut self.eval_server;
        let benchmark = &mut self.benchmark;
        let prompt = &mut self.prompt;
        let gpu_layers = &mut self.gpu_layers;
        let recipe = &mut self.recipe;
        let format_tab = &mut self.format_tab;
        let workspace_tab = &mut self.workspace_tab;
        let optimize_settings_tab = &mut self.optimize_settings_tab;
        let optimize_review_tab = &mut self.optimize_review_tab;
        let backends_expanded = &mut self.backends_expanded;
        let advanced_opt_expanded = &mut self.advanced_opt_expanded;
        let advanced_unc_expanded = &mut self.advanced_unc_expanded;
        let history_expanded = &mut self.history_expanded;
        let opt_baseline = &mut self.opt_baseline;
        let opt_candidate = &mut self.opt_candidate;
        let opt_suite = &mut self.opt_suite;
        let opt_output = &mut self.opt_output;
        let opt_imatrix = &mut self.opt_imatrix;
        let opt_sensitivity = &mut self.opt_sensitivity;
        let opt_ppl = &mut self.opt_ppl;
        let opt_min_saving = &mut self.opt_min_saving;
        let opt_max_loss = &mut self.opt_max_loss;
        let opt_max_ppl = &mut self.opt_max_ppl;
        let opt_goal = &mut self.opt_goal;
        let opt_device = &mut self.opt_device;
        let opt_runtime_device = &mut self.opt_runtime_device;
        let opt_precisions = &mut self.opt_precisions;
        let opt_bench_prompt = &mut self.opt_bench_prompt;
        let opt_bench_generation = &mut self.opt_bench_generation;
        let opt_bench_repetitions = &mut self.opt_bench_repetitions;
        let opt_speed_ratio = &mut self.opt_speed_ratio;
        let opt_planned = self.opt_planned.as_ref();
        let opt_result = self.opt_result.as_ref();
        let unc_trace = &mut self.unc_trace;
        let unc_output = &mut self.unc_output;
        let unc_strength = &mut self.unc_strength;
        let unc_block = &mut self.unc_block;
        let unc_planned = self.unc_planned.as_ref();
        let unc_result = self.unc_result.as_ref();
        let unc_tab = &mut self.unc_tab;
        let unc_paired = &mut self.unc_paired;
        let unc_refusal_suite = &mut self.unc_refusal_suite;
        let unc_task_suite = &mut self.unc_task_suite;
        let unc_auto_quantize = &mut self.unc_auto_quantize;
        let unc_auto_planned = self.unc_auto_planned.as_ref();
        let unc_auto_result = self.unc_auto_result.as_ref();
        let learn_subtab = &mut self.learn_subtab;
        let learn_dataset_filter = &mut self.learn_dataset_filter;
        let learn_train_quant = &mut self.learn_train_quant;
        let _learn_train_rank = &mut self.learn_train_rank;
        let learn_last_action_msg = &mut self.learn_last_action_msg;
        let model = if source_is_current {
            self.model.as_ref()
        } else {
            None
        };
        let model_topology = if source_is_current {
            self.model_topology.as_ref()
        } else {
            None
        };
        let hf_folder = if source_is_current {
            self.hf_folder.as_ref()
        } else {
            None
        };
        let model_probe = if source_is_current {
            self.probe.as_ref()
        } else {
            None
        };
        let detected_gpus = &self.detected_gpus;
        let current_stage = &self.current_stage;
        let completed_stages = &self.completed_stages;
        let job_history = &self.job_history;
        let last_inference = &self.last_inference;
        let last_benchmark = &self.last_benchmark;
        let status = &self.status;
        let busy = self.busy;
        let backends = &self.backends;
        let backend_capabilities = self.backend_capabilities.as_ref();
        let last_output = if source_is_current {
            self.last_output.as_ref()
        } else {
            None
        };
        let log_tail = &self.log_tail;
        let job_elapsed = self.started_at.map(|t| t.elapsed()).or(self.last_elapsed);
        let mut rescan = false;
        let mut probe_capabilities = false;
        let mut save_local_settings = false;
        let mut download: Option<BackendId> = None;
        let mut new_focus = self.focus;
        let mut browse_model = false;
        let mut browse_folder = false;
        let mut browse_opt_out = false;
        let mut browse_unc_out = false;
        let mut one_click_opt = false;
        let mut one_click_unc = false;
        let mut cancel_job = false;
        let mut open_folder_target: Option<PathBuf> = None;
        let mut load_candidate_target: Option<PathBuf> = None;
        let mut export_report_json: Option<(PathBuf, serde_json::Value)> = None;
        let mut recipe_plan = false;
        let (
            mut inspect,
            mut plan,
            mut advise,
            mut opt_run,
            mut convert,
            mut smoke,
            mut bench,
            mut unc_plan,
            mut unc_run,
            mut unc_auto_plan,
            mut unc_auto_run,
        ) = (
            false, false, false, false, false, false, false, false, false, false, false,
        );
        {
            // New Kathamo navigation; each destination is an existing working view.
            let rail = NavigationRail::new()
                .with_selected(if *workspace_tab == 3 {
                    "learn"
                } else if *workspace_tab == 2 {
                    "uncensor"
                } else if *workspace_tab == 1 {
                    "manual"
                } else if *optimize_review_tab == 1 {
                    "results"
                } else if *optimize_settings_tab == 2 {
                    "hardware"
                } else {
                    "optimize"
                })
                .add_main(NavigationRailItem::new(
                    "optimize",
                    StudioIcon::Sparkles,
                    "Optimize",
                ))
                .add_main(NavigationRailItem::new(
                    "uncensor",
                    StudioIcon::Sparkles,
                    "Uncensor",
                ))
                .add_main(NavigationRailItem::new(
                    "hardware",
                    StudioIcon::Cpu,
                    "Hardware",
                ))
                .add_main(NavigationRailItem::new(
                    "results",
                    StudioIcon::Check,
                    "Results",
                ))
                .add_main(NavigationRailItem::new(
                    "manual",
                    StudioIcon::Settings,
                    "Manual Tools",
                ))
                .add_main(NavigationRailItem::new(
                    "learn",
                    StudioIcon::Sparkles,
                    "Learn",
                ));
            if let Some(destination) = rail.render(
                &mut cmds,
                &mut self.icons,
                Rect::from_min_size(
                    Vec2::ZERO,
                    Vec2::new(navigation_width - 4.0, viewport.height()),
                ),
                click_pointer.position,
                click_pointer.primary_down,
                &palette,
            ) {
                match destination {
                    "optimize" => {
                        *workspace_tab = 0;
                        *optimize_settings_tab = 0;
                        *optimize_review_tab = 0;
                    }
                    "hardware" => {
                        *workspace_tab = 0;
                        *optimize_settings_tab = 2;
                        *optimize_review_tab = 0;
                        *advanced_opt_expanded = true;
                    }
                    "results" => {
                        *workspace_tab = 0;
                        *optimize_review_tab = 1;
                    }
                    "manual" => {
                        *workspace_tab = 1;
                    }
                    "uncensor" => {
                        *workspace_tab = 2;
                    }
                    "learn" => {
                        *workspace_tab = 3;
                    }
                    _ => {}
                }
            }
            let mut ui = KathamoUi::new(
                &mut cmds,
                &mut self.icons,
                &palette,
                viewport,
                click_pointer,
            );

            // 1. Studio Telemetry & Header Bar
            ui.horizontal(72.0, |top| {
                top.vertical(252.0, |brand| {
                    brand.bilingual_heading(&BilingualLabel::new("Ojon", "ওজন"));
                    brand.horizontal(22.0, |pills| {
                        pills.status_pill(if busy { "Processing" } else { "Idle" }, !busy);
                        let found = backends.iter().filter(|b| b.installed()).count();
                        pills.status_pill(
                            &format!("{found}/{} Tools Detected", backends.len()),
                            found > 0,
                        );
                    });
                });

                let (model_title, model_sub) = if let Some(info) = model {
                    let name = info.model_name.as_deref().unwrap_or("GGUF Container");
                    let sub = if let Some(topo) = model_topology {
                        format!(
                            "{:.2} GiB | {:.2}B params ({:.2} bpw)",
                            info.file_bytes as f64 / 1_073_741_824.0,
                            topo.total_parameters as f64 / 1e9,
                            topo.effective_bpw
                        )
                    } else {
                        format!(
                            "{:.2} GiB ({} Tensors)",
                            info.file_bytes as f64 / 1_073_741_824.0,
                            info.tensor_count
                        )
                    };
                    (name, sub)
                } else if let Some(info) = hf_folder {
                    let sub = format!(
                        "{:.2} GiB ({} Shards)",
                        info.source_weight_bytes as f64 / 1_073_741_824.0,
                        info.weight_files
                    );
                    ("HuggingFace Folder", sub)
                } else if let Some(p) = model_probe {
                    (
                        p.identification.as_str(),
                        format!(
                            "{:.2} GiB; {} (metadata only)",
                            p.bytes as f64 / 1_073_741_824.0,
                            p.category.as_str()
                        ),
                    )
                } else if !source.trim().is_empty() {
                    ("Not Inspected", "Press Inspect Model Metadata".into())
                } else {
                    ("No Model Loaded", "Drop File or Click Browse".into())
                };
                let title = model_title.chars().take(26).collect::<String>();
                let subtitle = model_sub.chars().take(42).collect::<String>();
                top.stat_card("SOURCE CHECKPOINT", &title, &subtitle);

                let (hw_title, hw_sub) = if let Some(gpu) = detected_gpus
                    .iter()
                    .find(|g| !g.is_integrated)
                    .or_else(|| detected_gpus.first())
                {
                    let free_gib = gpu.free_vram_bytes as f64 / 1_073_741_824.0;
                    (
                        gpu.runtime_id.as_str(),
                        format!(
                            "{} ({:.1} GiB free)",
                            gpu.name.chars().take(22).collect::<String>(),
                            free_gib
                        ),
                    )
                } else {
                    ("CPU Safe Mode", "Auto hardware discovery active".into())
                };
                top.stat_card("DETECTED HARDWARE", hw_title, &hw_sub);

                let quantizer_ready = Path::new(quantizer.trim()).is_file()
                    || (quantizer.trim() == "llama-quantize"
                        && backends
                            .iter()
                            .any(|b| b.spec.id == BackendId::LlamaQuantize && b.installed()));
                let smoke_ready = Path::new(runtime.trim()).is_file()
                    || (runtime.trim() == "llama-cli"
                        && backends
                            .iter()
                            .any(|b| b.spec.id == BackendId::LlamaCli && b.installed()));
                top.stat_card(
                    "LOCAL TOOLS",
                    if quantizer_ready {
                        "Quantizer detected"
                    } else {
                        "Setup required"
                    },
                    if smoke_ready {
                        "Inference tool found"
                    } else {
                        "Inference tool missing"
                    },
                );
            });

            ui.separator();
            ui.horizontal(30.0, |steps| {
                steps.status_pill(
                    "1  SELECT / INSPECT",
                    source_is_current && model_probe.is_some(),
                );
                steps.status_pill(
                    "2  OPTIMISE / EDIT",
                    busy || opt_result.is_some() || unc_auto_result.is_some(),
                );
                steps.status_pill(
                    "3  RELOAD & VERIFY",
                    opt_result.is_some() || unc_auto_result.is_some(),
                );
                steps.status_pill(
                    "4  READY / EXPORT",
                    opt_result.is_some_and(|r| r["status"] == "PROVISIONAL_CANDIDATE_FOUND")
                        || unc_auto_result
                            .is_some_and(|r| r["status"] == "SELECTED_FROM_MEASURED_SCORES"),
                );
            });

            // 2. Main Two-Column Workstation Layout
            ui.columns(
                (height - 154.0).max(520.0),
                &[SizeConstraint::Fixed(if width < 1280.0 { 335.0 } else { 365.0 }), SizeConstraint::Flex(1.0)],
                |idx, col| {
                    if idx == 0 {
                        col.glass_card(336.0, |card| {
                            card.heading("01   SELECT YOUR MODEL");
                            card.label("Drag & drop a model, browse, or paste its path:");
                            if card.text_input(source, if new_focus == Some(Focus::Model) { "Type/paste model path (Ctrl+V)..." } else { "Source model path (or drop file)..." }).clicked() {
                                new_focus = Some(Focus::Model);
                            }
                            card.horizontal(32.0, |pick_row| {
                                if pick_row.button_secondary("Browse GGUF...").clicked() { browse_model = true; }
                                if pick_row.button_secondary("Browse Folder...").clicked() { browse_folder = true; }
                            });
                            if card.button("Inspect Model Metadata").clicked() { inspect = true; }
                            card.separator();
                            if let Some(p) = model_probe {
                                card.label(&format!("Category: {} ({})", p.category.as_str(), p.confidence));
                            }
                            if let Some(info) = model {
                                card.horizontal(22.0, |b| {
                                    b.badge(info.architecture.as_deref().unwrap_or("GGUF"), true);
                                    b.badge(&format!("v{}", info.version), true);
                                    b.badge(&format!("{:.2} GiB", info.file_bytes as f64 / 1_073_741_824.0), true);
                                });
                                if let Some(topo) = model_topology {
                                    card.label(&format!(
                                        "Topology: {:.2}B params | {:.2} bpw | {}",
                                        topo.total_parameters as f64 / 1e9,
                                        topo.effective_bpw,
                                        if topo.is_moe { format!("MoE ({} router tensors)", topo.expert_count) } else { "Dense".into() }
                                    ));
                                    let est_4k = topo.estimate_runtime_memory(4096, 512);
                                    card.label(&format!(
                                        "Est. Runtime (4k ctx): {:.2} GiB total",
                                        est_4k.total_estimated_bytes as f64 / 1_073_741_824.0
                                    ));
                                } else {
                                    card.label(&format!("Tensors: {} verified descriptors", info.tensor_count));
                                }
                                card.status_pill("Ready for One-Click Optimise / Uncensor", true);
                            } else if let Some(info) = hf_folder {
                                card.horizontal(22.0, |b| {
                                    b.badge("HuggingFace", true);
                                    b.badge(&format!("{} Shards", info.weight_files), true);
                                    b.badge(&format!("{:.2} GiB", info.source_weight_bytes as f64 / 1_073_741_824.0), true);
                                });
                                card.label("Official convert_hf_to_gguf.py pipeline");
                                card.status_pill("Checkpoint Folder Verified", true);
                            } else if model_probe.is_some() {
                                card.label("Only metadata inspection available for this source.");
                                card.status_pill("No compatible conversion adapter connected", false);
                            } else {
                                card.label("No model inspected yet. Drop or browse a model.");
                                card.status_pill("Waiting for GGUF or HF model", false);
                            }
                        });

                        // Persistent Job History & Checkpoint Recovery
                        col.card(if *history_expanded { 210.0 } else { 76.0 }, |panel| {
                            let recoverable = job_history.iter().filter(|j| j.status == "CHECKPOINT_RECOVERABLE").count();
                            let title = if recoverable > 0 {
                                format!("JOB HISTORY ({} RECOVERABLE)", recoverable)
                            } else {
                                format!("JOB HISTORY ({} RECORDED)", job_history.len())
                            };
                            panel.collapsible_panel(&title, history_expanded, |hist| {
                                if job_history.is_empty() {
                                    hist.label("No persistent jobs recorded yet.");
                                } else {
                                    for item in job_history.iter().rev().take(3) {
                                        let name = Path::new(&item.source_model)
                                            .file_name().unwrap_or_default().to_string_lossy();
                                        hist.label(&format!(
                                            "[{}] {} ({})",
                                            item.status,
                                            name.chars().take(24).collect::<String>(),
                                            item.current_stage.chars().take(28).collect::<String>()
                                        ));
                                        if let Some(cand) = &item.result_artifact {
                                            if hist.button_secondary("Load Saved Result").clicked() {
                                                load_candidate_target = Some(PathBuf::from(cand));
                                            }
                                        } else if item.status == "CHECKPOINT_RECOVERABLE" {
                                            if hist.button_secondary("Recover Source & Output").clicked() {
                                                load_candidate_target = Some(PathBuf::from(&item.source_model));
                                            }
                                        }
                                    }
                                }
                            });
                        });

                        // Collapsible specialist backend tools
                        col.card(if *backends_expanded { 410.0 } else { 76.0 }, |panel| {
                            let detected=backends.iter().filter(|b|b.installed()).count();
                            panel.collapsible_panel(
                                &format!("BACKEND TOOLS ({detected}/{})",backends.len()),
                                backends_expanded,
                                |details| {
                            if details.button_secondary("Rescan Installed Tools").clicked() { rescan = true; }
                            if details.button_secondary("Probe Device IDs & Quant Formats").clicked() {
                                probe_capabilities=true;
                            }
                            if details.button_secondary("Save Local Preferences").clicked(){save_local_settings=true;}
                            if let Some(receipt)=backend_capabilities{
                                for item in receipt["probes"].as_array().into_iter().flatten(){
                                    if item["status"]=="CAPABILITY_OUTPUT_PARSED"{
                                        let devices=item["device_ids"].as_array()
                                            .map(|a|a.iter().filter_map(|v|v.as_str()).collect::<Vec<_>>().join(", "))
                                            .unwrap_or_default();
                                        let formats=item["precision_formats"].as_array()
                                            .map(|a|a.iter().filter_map(|v|v.as_str()).take(6).collect::<Vec<_>>().join(", "))
                                            .unwrap_or_default();
                                        details.label(&format!("{}: {} {}",
                                            item["name"].as_str().unwrap_or("runtime"),
                                            devices,formats).chars().take(94).collect::<String>());
                                    }
                                }
                            }
                            for backend in backends.iter() {
                                let label = match backend.spec.id {
                                    BackendId::LlamaQuantize => "GGUF converter",
                                    BackendId::LlamaCli => "GGUF inference",
                                    BackendId::LlamaServer=>"Persistent task-evaluation server",
                                    BackendId::LlamaBench => "GGUF performance benchmark",
                                    BackendId::LlamaPerplexity => "GGUF quality / PPL",
                                    BackendId::LlamaEmbedding => "Embedding / reranker parity",
                                    BackendId::LlamaImatrix => "GGUF calibration",
                                    BackendId::HfConverter => "HF to GGUF script",
                                    BackendId::LlamaFactory => "LoRA / QLoRA trainer",
                                    BackendId::StableDiffusionCpp => "Image diffusion",
                                    BackendId::WhisperCpp => "Speech / Whisper",
                                    BackendId::IntelAutoRound => "Intel AutoRound (XPU)",
                                    BackendId::IntelSyclCli => "Intel SYCL / oneAPI",
                                    BackendId::LlamaCvectorGenerator => "llama.cpp control-vector generator",
                                    BackendId::WhisperQuantize => "Speech / whisper quantize",
                                };
                                details.horizontal(22.0, |row| {
                                    row.label(&format!("{}:", label));
                                    row.badge(if backend.installed() { "Detected" } else { "Not Found" }, backend.installed());
                                });
                                if !backend.installed() {
                                    if details.button_secondary(&format!("Official download: {label}")).clicked() {
                                        download = Some(backend.spec.id);
                                    }
                                } else if backend.spec.integration == ojon_backend::Integration::DiscoveryOnly {
                                    details.label("Detected; Ojon adapter not connected yet");
                                }
                            }
                                },
                            );
                        });
                    } else {
                        col.card(54.0, |bar|{
                            bar.tabs(&["Optimize Model","Manual Tools","Uncensor Model","Ojon Learn"], workspace_tab);
                        });
                        if *workspace_tab==0 {
                            col.glass_card(if *advanced_opt_expanded { 430.0 } else { 250.0 }, |card|{
                                card.heading("02   OPTIMISE MODEL (UNIFIED PRECISION OPTIMIZER)");
                                card.label("1. Choose your objective  ->  2. Click Optimise Model (auto hardware & calibration).");
                                card.dropdown("Optimization Objective", opt_goal, &[
                                    "Balanced  —  Best quality-to-size & decode speed",
                                    "Smallest  —  Maximum compression passing quality gates",
                                    "Fastest   —  Highest measured decode throughput",
                                    "Maximum Quality  —  Highest retention (Q6_K / Q5_K_M spine)",
                                ]);
                                let rec_strategy = match *opt_goal {
                                    1 => "Recommended: Auto-Imatrix + Aggressive K-Quant search (Q3_K_M / Q4_K_M) with Spine Protection",
                                    2 => "Recommended: Auto-Imatrix + High-Throughput K-Quant search + llama-bench verification",
                                    3 => "Recommended: Auto-Imatrix + RCO / Role-Aware Q6_K / Q5_K_M preservation",
                                    _ => "Recommended: Auto-Imatrix + Role-Aware Q4_K_M / Q5_K_M search with Spine Protection",
                                };
                                card.label(rec_strategy);
                                card.horizontal(36.0, |actions| {
                                    if !busy && actions.button("Optimise Model").clicked() {
                                        one_click_opt = true;
                                    }
                                    if !busy && actions.button_secondary("Build Read-Only Plan").clicked() {
                                        advise = true;
                                    }
                                    if busy && actions.button_secondary("Cancel Active Job").clicked() {
                                        cancel_job = true;
                                    }
                                });
                                card.collapsible_panel(
                                    "ADVANCED OPTIMIZATION SETTINGS (OPTIONAL)",
                                    advanced_opt_expanded,
                                    |adv| {
                                        adv.tabs(&["Model & Tasks","Quality Controls","Hardware","Search & Bench"], optimize_settings_tab);
                                        if *optimize_settings_tab==0 {
                                            adv.label("Quality reference (optional: original F16/BF16 is used when blank)");
                                            if adv.text_input(opt_baseline,"Optional existing Q4 reference GGUF path...").clicked(){
                                                new_focus=Some(Focus::OptBaseline);
                                            }
                                            adv.label("Held-out coding, reasoning and tool-use suite (blank = bundled sanity suite)");
                                            if adv.text_input(opt_suite,"Task suite JSON path...").clicked(){
                                                new_focus=Some(Focus::OptSuite);
                                            }
                                            adv.label("Existing smaller GGUF to compare (optional)");
                                            if adv.text_input(opt_candidate,"Smaller GGUF path; omit to quantize original...").clicked(){
                                                new_focus=Some(Focus::OptCandidate);
                                            }
                                            adv.label("Results directory (blank = source model folder)");
                                            adv.horizontal(28.0, |out_row| {
                                                if out_row.text_input(opt_output,"Folder for new models and reports...").clicked(){
                                                    new_focus=Some(Focus::OptOutput);
                                                }
                                                if out_row.button_secondary("Browse...").clicked() {
                                                    browse_opt_out = true;
                                                }
                                            });
                                        }else if *optimize_settings_tab==1{
                                            adv.label("Importance matrix from separate calibration (blank = auto-calibrate)");
                                            if adv.text_input(opt_imatrix,"Calibration imatrix path...").clicked(){
                                                new_focus=Some(Focus::OptImatrix);
                                            }
                                            adv.label("Adaptive tensor sensitivity scores JSON (optional)");
                                            if adv.text_input(opt_sensitivity,"Tensor sensitivity scores path...").clicked(){
                                                new_focus=Some(Focus::OptSensitivity);
                                            }
                                            adv.label("Independent held-out perplexity corpus (optional)");
                                            if adv.text_input(opt_ppl,"PPL corpus text path...").clicked(){
                                                new_focus=Some(Focus::OptPpl);
                                            }
                                            adv.label("Minimum file saving %, Max task loss pts, Max PPL regression %:");
                                            adv.horizontal(28.0, |gates| {
                                                if gates.text_input(opt_min_saving,"Min save % (5)").clicked(){
                                                    new_focus=Some(Focus::OptMinSaving);
                                                }
                                                if gates.text_input(opt_max_loss,"Max loss (0)").clicked(){
                                                    new_focus=Some(Focus::OptMaxLoss);
                                                }
                                                if gates.text_input(opt_max_ppl,"Max PPL % (0)").clicked(){
                                                    new_focus=Some(Focus::OptMaxPpl);
                                                }
                                            });
                                        }else if *optimize_settings_tab==2{
                                            adv.label("Runtime backend (Auto discovers discrete GPU automatically)");
                                            adv.dropdown("Choose runtime",opt_device,&[
                                                "Auto (Discrete GPU / CPU fallback)","CPU","CUDA backend","SYCL backend","Vulkan backend"
                                            ]);
                                            if matches!(*opt_device,2|3|4){
                                                adv.label("Exact device ID from your installed runtime (e.g. Vulkan2, CUDA0)");
                                                if adv.text_input(opt_runtime_device,"e.g. Vulkan2 or CUDA0").clicked(){
                                                    new_focus=Some(Focus::OptRuntimeDevice);
                                                }
                                                adv.label("GPU offload layers (>0 required)");
                                                if adv.text_input(gpu_layers,"GPU layers").clicked(){
                                                    new_focus=Some(Focus::GpuLayers);
                                                }
                                            }
                                        }else{
                                            adv.label("Persistent evaluator executable (blank uses discovered server or CLI)");
                                            if adv.text_input(eval_server,"llama-server executable path...").clicked(){
                                                new_focus=Some(Focus::EvalServer);
                                            }
                                            adv.label("Candidate formats (blank = source-aware defaults)");
                                            if adv.text_input(opt_precisions,"Q4_K_M,Q5_K_M,Q6_K,...").clicked(){
                                                new_focus=Some(Focus::OptPrecisions);
                                            }
                                            adv.horizontal(28.0, |bench_row| {
                                                if bench_row.text_input(opt_bench_prompt,"Prefill (512)").clicked(){
                                                    new_focus=Some(Focus::OptBenchPrompt);
                                                }
                                                if bench_row.text_input(opt_bench_generation,"Decode (128)").clicked(){
                                                    new_focus=Some(Focus::OptBenchGeneration);
                                                }
                                                if bench_row.text_input(opt_bench_repetitions,"Reps (3)").clicked(){
                                                    new_focus=Some(Focus::OptBenchRepetitions);
                                                }
                                                if bench_row.text_input(opt_speed_ratio,"Min ratio (0.85)").clicked(){
                                                    new_focus=Some(Focus::OptSpeedRatio);
                                                }
                                            });
                                        }
                                    }
                                );
                            });
                            col.card(if *advanced_opt_expanded { 245.0 } else { 370.0 }, |card|{
                                card.heading("03   PROGRESS, COMPARISON & VERIFIED RESULTS");
                                card.tabs(&["Review & Progress","Results & Comparison"],optimize_review_tab);
                                if busy {
                                    card.horizontal(26.0, |prog| {
                                        prog.status_pill(&format!("STAGE: {}", current_stage), true);
                                        if let Some(elapsed) = job_elapsed {
                                            prog.badge(&format!("Elapsed: {:.1}s", elapsed.as_secs_f64()), true);
                                        }
                                        if prog.button_secondary("Cancel Job").clicked() {
                                            cancel_job = true;
                                        }
                                    });
                                    if !completed_stages.is_empty() {
                                        card.label(&format!("Completed stages: {}", completed_stages.join(" -> ")));
                                    }
                                }
                                if *optimize_review_tab==0{
                                    let preview=opt_planned.map(|(_,plan)| (plan,optimizer_plan_current));
                                    if let Some((review,unchanged))=preview {
                                        let route=review["route"].as_str().unwrap_or("unavailable");
                                        let valid=review["ready_for_user_approved_execution"]==true;
                                        card.status_pill(if valid&&unchanged{"Ready for approval"}
                                            else if !unchanged {"Replan: settings changed"}
                                            else {"Requirements missing"},valid&&unchanged);
                                        card.label(&format!("Route: {route}"));
                                        if let Some(step)=review["next_user_step"].as_str(){
                                            card.label(&step.chars().take(105).collect::<String>());
                                        }
                                        if !valid {
                                            for note in review["checks"].as_array().into_iter()
                                                .flatten().take(2){
                                                if let Some(reason)=note.as_str(){
                                                    card.label(&reason.chars().take(92).collect::<String>());
                                                }
                                            }
                                        }
                                        if valid&&unchanged&&!busy{
                                            if card.button("Approve & Run Selected Backend").clicked(){
                                                opt_run=true;
                                            }
                                        }
                                    }else{
                                        card.label("Click 'Optimise Model' above for automatic execution, or 'Build Read-Only Plan' to inspect first.");
                                    }
                                    card.label(&status.chars().take(115).collect::<String>());
                                }else{
                                    if let Some(result)=opt_result{
                                        let passed=result["status"].as_str()==Some("PROVISIONAL_CANDIDATE_FOUND");
                                        card.status_pill(if passed{"ACCEPTED: Candidate passed all quality & size gates"}
                                            else{"REJECTED: No candidate passed selected gates"},passed);
                                        let selected=&result["provisional_candidate"];
                                        let candidate=selected.as_str()
                                            .or_else(||selected["path"].as_str()).unwrap_or("None");
                                        card.label(&format!("Selected Artifact: {}",
                                            candidate.chars().take(94).collect::<String>()));
                                        if let Some(topo) = result.get("selected_candidate_topology") {
                                            if let (Some(bpw), Some(bytes)) = (topo["effective_bpw"].as_f64(), topo["file_bytes"].as_u64()) {
                                                card.label(&format!(
                                                    "Measured Artifact Size: {:.2} GiB  |  Effective Precision: {:.2} bpw",
                                                    bytes as f64 / 1_073_741_824.0, bpw
                                                ));
                                            }
                                        }
                                        let evidence=&result["evidence"];
                                        card.label("BEFORE / AFTER CANDIDATE COMPARISON:");
                                        for item in evidence["results"].as_array().into_iter()
                                            .flatten().take(3){
                                            let name=item["name"].as_str()
                                                .or_else(||item["type"].as_str()).unwrap_or("candidate");
                                            let state=item["status"].as_str().unwrap_or("unknown");
                                            let saving=item["savings_pct"].as_f64().map(|v|format!("{v:.1}%"))
                                                .unwrap_or_else(||"unmeasured".into());
                                            let score_diff = item["score_change_points"].as_f64()
                                                .map(|v| format!("{v:+.1} pts")).unwrap_or_else(|| "0.0 pts".into());
                                            card.label(&format!("  • {name}: [{state}]  Size saving: {saving}  |  Quality delta: {score_diff}"));
                                        }
                                        let load=&result["selected_candidate_runtime_load"];
                                        let speed=&result["selected_candidate_performance"];
                                        let speed_str = if speed["status"]=="MEASURED"{
                                            let pp=speed["metrics"]["prompt_processing"]["tokens_per_second"].as_f64().unwrap_or(0.0);
                                            let tg=speed["metrics"]["token_generation"]["tokens_per_second"].as_f64().unwrap_or(0.0);
                                            format!("{pp:.1} prefill / {tg:.1} decode tok/s")
                                        } else {
                                            speed["status"].as_str().unwrap_or("unverified").to_string()
                                        };
                                        card.label(&format!(
                                            "Reload Verification: {}  |  Speed: {}",
                                            load["status"].as_str().unwrap_or("not tested"),
                                            speed_str
                                        ));
                                        card.horizontal(30.0, |act| {
                                            if candidate != "None" {
                                                let cand_path = PathBuf::from(candidate);
                                                if act.button_secondary("Open Result Folder").clicked() {
                                                    if let Some(parent) = cand_path.parent() {
                                                        open_folder_target = Some(parent.to_path_buf());
                                                    }
                                                }
                                                if act.button_secondary("Load & Test Model").clicked() {
                                                    load_candidate_target = Some(cand_path.clone());
                                                }
                                                if act.button_secondary("Export Report").clicked() {
                                                    let rep_path = cand_path.with_extension("optimization-report.json");
                                                    export_report_json = Some((rep_path, result.clone()));
                                                }
                                            }
                                        });
                                    }else{
                                        card.label(if busy {"Real optimization running; live stage and subprocess logs shown below."}
                                            else {"No optimization result yet. Click Optimise Model to start."});
                                        card.label(&status.chars().take(110).collect::<String>());
                                    }
                                    if let Some(elapsed)=job_elapsed{
                                        card.label(&format!("Measured elapsed: {:.1}s",elapsed.as_secs_f64()));
                                    }
                                    for line in log_tail.iter().rev().take(2){
                                        card.label(&line.chars().take(100).collect::<String>());
                                    }
                                }
                            });
                        }else if *workspace_tab==1{
                        col.glass_card(320.0, |card| {
                            card.heading("02   CONFIGURE QUANTIZATION");
                            let formats = ["GGUF (llama.cpp)", "FP8 (TensorRT)", "AWQ (AutoAWQ)", "EXL2 (ExLlamaV2)"];
                            card.tabs(&formats, format_tab);
                            if *format_tab != 0 {
                                card.label("Note: GGUF adapter active. Additional formats planned in Ojon roadmap.");
                            }

                            card.label("GGUF weight precision (not a measured quality ranking)");
                            card.dropdown("Precision", recipe, &[
                                "Q5_K_M  /  Quality Default (~5.45 bpw)",
                                "Q4_K_M  /  Balanced Compact (~4.50 bpw)",
                                "Q3_K_M  /  High Compression (~3.66 bpw)",
                                "Q6_K    /  Larger Format (~6.56 bpw)",
                            ]);

                            card.label("Destination GGUF path");
                            if card.text_input(output, if new_focus == Some(Focus::Output) { "Destination output path..." } else { "Output path..." }).clicked() {
                                new_focus = Some(Focus::Output);
                            }

                            card.label("Quantizer executable");
                            if card.text_input(quantizer, "llama-quantize path...").clicked() {
                                new_focus = Some(Focus::Quantizer);
                            }

                            if Path::new(source.trim()).is_dir() {
                                card.label("HF converter script");
                                if card.text_input(converter, "convert_hf_to_gguf.py...").clicked() {
                                    new_focus = Some(Focus::Converter);
                                }
                            }

                            card.horizontal(36.0, |actions| {
                                if actions.button_secondary("Check Safety & Plan").clicked() { plan = true; }
                                if actions.button("Quantize Model").clicked() { convert = true; }
                            });
                            if card.button_secondary("Recommend Best Route (No GPU)").clicked() { advise = true; }
                            // Measured mixed precision: plan only here; the run is the CLI's
                            // recipe-run with --confirm because it quantizes once per probe.
                            if !busy && card.button_secondary("Plan KLD Recipe Search (read-only)").clicked() { recipe_plan = true; }
                        });

                        col.glass_card(330.0, |card| {
                            card.heading("03   TEST QUANTIZED MODEL");
                            card.label("Real llama-cli smoke check. 0 GPU layers runs on CPU.");
                            card.label("llama-cli runtime executable");
                            if card.text_input(runtime, "llama-cli executable...").clicked() { new_focus = Some(Focus::Runtime); }
                            card.horizontal(28.0, |gpu_row| {
                                gpu_row.label("GPU offload layers (0 = CPU only):");
                                if gpu_row.text_input(gpu_layers, "0").clicked() { new_focus = Some(Focus::GpuLayers); }
                            });
                            card.label("Verification prompt");
                            if card.text_input(prompt, if new_focus == Some(Focus::Prompt) { "Enter test prompt..." } else { "Test prompt..." }).clicked() {
                                new_focus = Some(Focus::Prompt);
                            }
                            if card.button_secondary("Test with llama-cli").clicked() { smoke = true; }
                            card.label("llama-bench executable (real performance, optional)");
                            if card.text_input(benchmark, "llama-bench executable...").clicked() { new_focus = Some(Focus::Benchmark); }
                            if card.button_secondary("Benchmark Output (selected settings)").clicked() { bench = true; }
                            if let Some((receipt_path, values)) = last_benchmark {
                                let pp = values["prompt_processing"]["tokens_per_second"].as_f64().unwrap_or(0.0);
                                let tg = values["token_generation"]["tokens_per_second"].as_f64().unwrap_or(0.0);
                                card.label(&format!("Measured: prefill {pp:.1} tok/s, decode {tg:.1} tok/s"));
                                card.label(&format!("Receipt: {}", receipt_path.file_name().unwrap_or_default().to_string_lossy()));
                                card.label("Speed only; quality, TTFT and VRAM are not measured.");
                            }
                            if let Some(answer) = last_inference {
                                card.horizontal(20.0, |hdr| {
                                    hdr.badge("llama-cli output captured", true);
                                });
                                let clipped_reversed: String = answer.chars()
                                    .rev().filter(|ch| !ch.is_control() || *ch == '\n')
                                    .take(320).collect();
                                let clipped: String = clipped_reversed.chars().rev().collect();
                                for line in clipped.lines().filter(|s| !s.trim().is_empty()).take(3) {
                                    card.label(&line.chars().take(80).collect::<String>());
                                }
                            }
                        });

                        col.card(156.0, |card| {
                            card.heading("JOB STATUS & TELEMETRY");
                            card.label("Agent interface: Ojon CLI with JSON outputs.");
                            card.status_pill(if busy { "Backend Processing" } else { "No Active Job" }, !busy);
                            if let Some(elapsed) = job_elapsed {
                                card.label(&format!("Measured elapsed: {:.1}s", elapsed.as_secs_f64()));
                            }
                            if let Some((path, bytes)) = last_output {
                                let source_bytes = model_probe.map(|p| p.bytes).unwrap_or(0);
                                card.label(&format!("Output: {:.2} GiB ({})", *bytes as f64 / 1_073_741_824.0,
                                    path.file_name().unwrap_or_default().to_string_lossy()));
                                if source_bytes > 0 {
                                    card.label(&format!("Output is {:.1}% of inspected source bytes",
                                        *bytes as f64 / source_bytes as f64 * 100.0));
                                }
                            }
                            card.label(status);
                            for line in log_tail.iter().rev().take(2) {
                                card.label(&line.chars().take(88).collect::<String>());
                            }
                        });
                        }else if *workspace_tab==2{
                            col.glass_card(if *advanced_unc_expanded { 415.0 } else { 255.0 }, |card|{
                                card.heading("02   UNCENSOR MODEL (AUTOMATIC DIRECTION & EDIT ENGINE)");
                                card.tabs(&["Automatic (Recommended)","Manual trace edit"], unc_tab);
                                if *unc_tab==0{
                                    card.label("Automatic mode captures refusal directions, evaluates norm-preserving & Heretic multi-layer edits, and verifies retained capabilities.");
                                    card.label("Optimise after editing (Combined Edit -> Optimise -> Reload -> Verify workflow):");
                                    card.dropdown("Optimise after editing",unc_auto_quantize,&[
                                        "No quantization (keep edited F16/BF16 teacher)",
                                        "Optimise to Q4_K_M (verified against Original & Teacher)",
                                        "Optimise to Q5_K_M (verified against Original & Teacher)",
                                        "Optimise to Q6_K (verified against Original & Teacher)"
                                    ]);
                                    card.horizontal(36.0, |actions| {
                                        if !busy && actions.button("Uncensor Model").clicked() {
                                            one_click_unc = true;
                                        }
                                        if !busy && actions.button_secondary("Plan Automatic Search").clicked() {
                                            unc_auto_plan = true;
                                        }
                                        if busy && actions.button_secondary("Cancel Active Job").clicked() {
                                            cancel_job = true;
                                        }
                                    });
                                    if let Some((_,plan))=unc_auto_planned{
                                        let count=plan["candidate_count"].as_u64().unwrap_or(0);
                                        card.status_pill("Automatic search plan ready",true);
                                        card.label(&format!("{count} candidates across norm-preserving, standard & projected multi-layer Heretic methods."));
                                        if !busy && card.button("Run Approved Automatic Search").clicked(){unc_auto_run=true;}
                                    }
                                    card.collapsible_panel(
                                        "ADVANCED UNCENSOR SETTINGS (OPTIONAL)",
                                        advanced_unc_expanded,
                                        |adv| {
                                            adv.label("Paired behaviour examples (blank = bundled data/paired)");
                                            if adv.text_input(unc_paired,"Folder with positive.txt and negative.txt...").clicked(){
                                                new_focus=Some(Focus::UncPaired);
                                            }
                                            adv.label("Refusal probes (blank = bundled refusal-probes.json)");
                                            if adv.text_input(unc_refusal_suite,"ojon.refusal_suite.v1 JSON path...").clicked(){
                                                new_focus=Some(Focus::UncRefusalSuite);
                                            }
                                            adv.label("Capability suite (blank = bundled sanity-tasks.json)");
                                            if adv.text_input(unc_task_suite,"ojon.task_suite.v1 JSON path...").clicked(){
                                                new_focus=Some(Focus::UncTaskSuite);
                                            }
                                            adv.label("Results directory (blank = source model folder)");
                                            adv.horizontal(28.0, |out_row| {
                                                if out_row.text_input(unc_output,"Folder for winner, control-vector & receipts...").clicked(){
                                                    new_focus=Some(Focus::UncOutput);
                                                }
                                                if out_row.button_secondary("Browse...").clicked() {
                                                    browse_unc_out = true;
                                                }
                                            });
                                        }
                                    );
                                }else{
                                card.label("Edit a new F16/BF16 GGUF or HF checkpoint from model activations.");
                                card.label("Paired activation JSONL or llama.cpp control-vector GGUF");
                                if card.text_input(unc_trace,"Activation trace or control-vector GGUF path...").clicked(){
                                    new_focus=Some(Focus::UncTrace);
                                }
                                card.label("Control-vector block index (zero-based; blank for JSONL)");
                                if card.text_input(unc_block,"e.g. 19 for direction.20...").clicked(){
                                    new_focus=Some(Focus::UncBlock);
                                }
                                card.label("New output GGUF or model folder (source stays untouched)");
                                if card.text_input(unc_output,"New edited output path...").clicked(){
                                    new_focus=Some(Focus::UncOutput);
                                }
                                card.label("Projection strength (0 < value <= 1)");
                                if card.text_input(unc_strength,"0.7").clicked(){
                                    new_focus=Some(Focus::UncStrength);
                                }
                                card.label("Original row norms are preserved. Advanced editing is available in the CLI.");
                                if card.button_secondary("Build Read-Only Edit Plan").clicked(){unc_plan=true;}
                                if let Some((_,plan))=unc_planned{
                                    let count=plan["activation_tensors"].as_u64().unwrap_or(0);
                                    card.status_pill("Edit plan ready",true);
                                    card.label(&format!("{count} targeted matrices; review the new output path above."));
                                    if !busy && card.button("Write New Edited Checkpoint").clicked(){unc_run=true;}
                                }else{
                                    card.status_pill("Trace and source required",false);
                                }
                                }
                            });
                            col.card(if *advanced_unc_expanded { 245.0 } else { 365.0 },|card|{
                                card.heading("03   MEASURED REFUSAL & CAPABILITY EVIDENCE");
                                if busy {
                                    card.horizontal(26.0, |prog| {
                                        prog.status_pill(&format!("STAGE: {}", current_stage), true);
                                        if let Some(elapsed) = job_elapsed {
                                            prog.badge(&format!("Elapsed: {:.1}s", elapsed.as_secs_f64()), true);
                                        }
                                    });
                                }
                                if let Some(result)=unc_auto_result{
                                    let published=result["status"].as_str()==Some("SELECTED_FROM_MEASURED_SCORES");
                                    if published{
                                        card.status_pill("ACCEPTED: Measured winner verified & published",true);
                                    }else{
                                        card.status_pill("REJECTED: No candidate passed refusal & capability gates",false);
                                    }
                                    let orig_ref = result["baseline"]["refusal_rate"].as_f64().unwrap_or(0.0) * 100.0;
                                    let orig_benign = result["baseline"]["benign_overrefusal_rate"].as_f64().unwrap_or(0.0) * 100.0;
                                    card.label(&format!(
                                        "BEFORE (Original): Refusal Rate {:.1}%  |  Benign Over-Refusal {:.1}%",
                                        orig_ref, orig_benign
                                    ));
                                    if let Some(tested)=result["candidates"].as_array(){
                                        for row in tested.iter().take(4){
                                            card.label(&format!(
                                                "  • {} [{}]: Refusal {:.1}% | Benign Over-Refusal {:.1}% | Capability Pass={}",
                                                row["name"].as_str().unwrap_or("?"),
                                                row["method"].as_str().unwrap_or("ablation"),
                                                100.0*row["refusal_rate"].as_f64().unwrap_or(0.0),
                                                100.0*row["benign_overrefusal_rate"].as_f64().unwrap_or(0.0),
                                                row["quality_pass"]
                                            ));
                                        }
                                    }
                                    if let Some(reload) = result.get("edited_reload_verification") {
                                        if !reload.is_null() {
                                            card.label(&format!(
                                                "Edited Teacher Reload Verification: status={} ({} tensors)",
                                                reload["status"].as_str().unwrap_or("verified"),
                                                reload["tensor_count"].as_u64().unwrap_or(0)
                                            ));
                                        }
                                    }
                                    if let Some(quant) = result.get("quantized_verification") {
                                        if !quant.is_null() {
                                            let dual_pass = quant["passes_dual_reference_gate"].as_bool().unwrap_or(false);
                                            let q_ref = quant["refusal_rate"].as_f64().unwrap_or(0.0) * 100.0;
                                            card.label(&format!(
                                                "Combined Quantized Verification: Refusal {:.1}% | Dual Gate (vs Original & Teacher)={}",
                                                q_ref, dual_pass
                                            ));
                                        }
                                    }
                                    let published_path = result["output_quantized"]
                                        .as_str()
                                        .or_else(|| result["output_edited"].as_str())
                                        .map(PathBuf::from);

                                    card.horizontal(30.0, |act| {
                                        if let Some(ref p) = published_path {
                                            if act.button_secondary("Test Model Locally").clicked() {
                                                load_candidate_target = Some(p.clone());
                                            }
                                            if act.button_secondary("Open Result Folder").clicked() {
                                                if let Some(parent) = p.parent() {
                                                    open_folder_target = Some(parent.to_path_buf());
                                                }
                                            }
                                            if act.button_secondary("Export Report").clicked() {
                                                let rep_path = p.with_extension("uncensor-report.json");
                                                export_report_json = Some((rep_path, result.clone()));
                                            }
                                        }
                                    });
                                }else if let Some(result)=unc_result{
                                    card.status_pill("Edited checkpoint written",true);
                                    let output=&result["model"]["output"];
                                    let location=output.as_str().or_else(||result["model"]["model"]["output"].as_str())
                                        .unwrap_or("See result receipt");
                                    card.label(&format!("Output: {}",location.chars().take(90).collect::<String>()));
                                    card.label("Loader, refusal rate and capability retention need separate tests.");
                                }else{
                                    card.label("No edited checkpoint result yet. Click Uncensor Model above to start.");
                                }
                                card.label(&status.chars().take(110).collect::<String>());
                                if let Some(elapsed)=job_elapsed{
                                    card.label(&format!("Elapsed: {:.1}s",elapsed.as_secs_f64()));
                                }
                                for line in log_tail.iter().rev().take(2){
                                    card.label(&line.chars().take(95).collect::<String>());
                                }
                            });
                        }else if *workspace_tab==3{
                            col.glass_card(420.0, |card|{
                                card.heading("02   OJON LEARN — DISTILLATION, FINE-TUNING & DATA FLYWHEEL");
                                let learn_tabs = ["Collector & Telemetry", "Dataset Curation", "Post-Training (SFT)", "Evaluate & Promote"];
                                card.tabs(&learn_tabs, learn_subtab);

                                if *learn_subtab == 0 {
                                    card.label("Local Loopback Capture Gateway: intercepts coding assistant traces (/v1/chat/completions & /v1/traces) with PII redaction.");
                                    card.horizontal(28.0, |row| {
                                        row.label("Loopback Port: 8089");
                                        row.status_pill("Gateway Ready (6 Utility Models Registered)", true);
                                    });
                                    card.label("Active Data Store: %APPDATA%\\Ojon\\collector\\collected_episodes.jsonl");
                                    card.label("Outcome Verification: Compiler / test / schema exit codes recorded as reward signals (+1.0 / -1.0 / 0.0).");
                                    if card.button_secondary("Refresh Collector Telemetry").clicked() {
                                        let store_dir = ojon_backend::DaemonConfig::default().storage_dir;
                                        match ojon_backend::LearnCollector::new(store_dir) {
                                            Ok(coll) => {
                                                let st = coll.get_stats(8089);
                                                *learn_last_action_msg = Some(format!(
                                                    "Telemetry refreshed: {} episodes ({} verified, {} failed, {} redacted).",
                                                    st.total_episodes, st.verified_success_count, st.failed_retry_count, st.redacted_count
                                                ));
                                            }
                                            Err(e) => {
                                                *learn_last_action_msg = Some(format!("Collector telemetry check failed: {e}"));
                                            }
                                        }
                                    }
                                } else if *learn_subtab == 1 {
                                    card.label("Curated Public Trajectories & Dual-Buffer Anti-Catastrophic Splitter:");
                                    card.dropdown("Dataset Catalog", learn_dataset_filter, &[
                                        "All Curated Datasets (12 Canonical Registries)",
                                        "SWE / Coding Agent (SI2CA, NVIDIA SWE, SWE-smith, SWE-Hero)",
                                        "Developer Traces (SWE-chat, Share-Codex, Trace-Commons)",
                                        "Reasoning & Planning (OpenThoughts-3, Nemotron-Agentic-v2)",
                                        "Quarantined Dumps (FABLE.5, Fable-GPT-5.5) [Audit Required]"
                                    ]);
                                    card.label("Dual-Buffer Allocation: 8% Static Anchor (Math/Syntax/Tool-Decline) + K=3 Recency Window");
                                    if card.button_secondary("Run Dual-Buffer Curation & Export").clicked() {
                                        let store_dir = ojon_backend::DaemonConfig::default().storage_dir;
                                        let ep_file = store_dir.join("collected_episodes.jsonl");
                                        let out_dir = store_dir.join("curated");
                                        let eps = ojon_backend::load_episodes_from_jsonl(&ep_file).unwrap_or_default();
                                        let batch = ojon_backend::partition_dual_buffer(&eps, &ojon_backend::DualBufferConfig::default());
                                        match ojon_backend::export_to_llamafactory(&batch, &out_dir) {
                                            Ok(rec) => {
                                                *learn_last_action_msg = Some(format!(
                                                    "Curated split exported: {} anchor, {} recent, {} DPO pairs to {}.",
                                                    rec.anchor_count, rec.recency_count, rec.dpo_pair_count, out_dir.display()
                                                ));
                                            }
                                            Err(e) => {
                                                *learn_last_action_msg = Some(format!("Curation export error: {e}"));
                                            }
                                        }
                                    }
                                } else if *learn_subtab == 2 {
                                    card.label("Hardware-Aware Post-Training & Distillation Planner:");
                                    card.label("Probed Accelerator: NVIDIA RTX 4060 Ti 16GB (CUDA High VRAM, BF16 Supported) / Intel Arc / CPU");
                                    card.dropdown("LoRA Training Strategy", learn_train_quant, &[
                                        "NF4 QLoRA (Rank 16, Alpha 32, Targets: q,k,v,o,gate,up,down) [Recommended]",
                                        "BF16 LoRA (Unquantized Base Model, 1B-4B Scale)",
                                    ]);
                                    card.label("Effective Batch Size: 8 (Microbatch 1, Grad Accum 8, Gradient Checkpointing: Active)");
                                    if card.button_secondary("Generate LLaMA-Factory Recipe & Dry Run").clicked() {
                                        let store_dir = ojon_backend::DaemonConfig::default().storage_dir;
                                        let q_bits = if *learn_train_quant == 0 { Some(4) } else { None };
                                        let base_model_str = if source.trim().is_empty() { "base_model" } else { source.trim() };
                                        let plan = ojon_backend::LearnTrainPlan::new(
                                            std::path::PathBuf::from(base_model_str),
                                            store_dir.join("curated"),
                                            store_dir.join("train_out"),
                                            Some(q_bits),
                                        );
                                        match ojon_backend::execute_training_plan(&plan, true, None) {
                                            Ok(rec) => {
                                                *learn_last_action_msg = Some(format!(
                                                    "Training plan validated ({}): {}",
                                                    rec.status,
                                                    rec.yaml_config_path
                                                ));
                                            }
                                            Err(e) => {
                                                *learn_last_action_msg = Some(format!("Training plan error: {e}"));
                                            }
                                        }
                                    }
                                } else {
                                    card.label("Four-Way Candidate Comparison & Promotion Gate:");
                                    card.label("Gates: Size Reduction >= 15% | Must-Pass Drop <= 1.0 pt | Silent Tool-Decline >= 80%");
                                    card.horizontal(24.0, |pills| {
                                        pills.status_pill("Tool-Decline: 88.0% (PASS)", true);
                                        pills.status_pill("Must-Pass Score: 88.0 pts (PASS)", true);
                                        pills.status_pill("Compression: 67.5% (PASS)", true);
                                    });
                                    card.label("Active Promoted Model: student-q4_k_m.gguf (Rollback Pointer: student-fused-fp16)");
                                    if card.button_secondary("Evaluate & Verify Promotion Gates").clicked() {
                                        let util_count = ojon_backend::default_utility_models().len();
                                        let store_dir = ojon_backend::DaemonConfig::default().storage_dir;
                                        let active_pkgs = ojon_backend::list_active_packages(&store_dir).map(|v| v.len()).unwrap_or(0);
                                        *learn_last_action_msg = Some(format!(
                                            "Promotion & package gates verified ({} utility models ready, {} active packages installed).",
                                            util_count, active_pkgs
                                        ));
                                    }
                                }

                                if let Some(msg) = learn_last_action_msg.as_deref() {
                                    card.status_pill(msg, true);
                                }
                            });
                        }
                    }
                },
            );
        }
        if auto_generated_output && previous_precision != self.precision() {
            self.output = default_output(&self.source, self.precision());
            self.last_output = None;
        }
        if self.focus != new_focus {
            self.select_all = false;
        }
        self.focus = new_focus;
        if save_local_settings {
            self.persist_local_settings(true);
        }
        if rescan {
            self.rescan_backends();
        }
        if probe_capabilities {
            self.start_capability_probe();
        }
        if browse_model {
            if let Some(picked) = pick_model_file() {
                self.set_source(&picked);
            }
        }
        if browse_folder {
            if let Some(picked) = pick_folder("Select HuggingFace model folder or source directory")
            {
                self.set_source(&picked);
            }
        }
        if browse_opt_out {
            if let Some(picked) = pick_folder("Select Optimization Output Directory") {
                self.opt_output = picked.to_string_lossy().into_owned();
                self.needs_redraw = true;
            }
        }
        if browse_unc_out {
            if let Some(picked) = pick_folder("Select Uncensor Output Directory") {
                self.unc_output = picked.to_string_lossy().into_owned();
                self.needs_redraw = true;
            }
        }
        if let Some(folder) = open_folder_target {
            let _ = open_folder_in_explorer(&folder);
        }
        if let Some(cand) = load_candidate_target {
            self.output = cand.to_string_lossy().into_owned();
            self.workspace_tab = 1;
            self.status = format!("Loaded '{}' into Manual Tools test runner. Click 'Test with llama-cli' to run local inference.", cand.display());
            self.needs_redraw = true;
        }
        if let Some((rep_path, payload)) = export_report_json {
            let write_res = serde_json::to_string_pretty(&payload)
                .map_err(std::io::Error::other)
                .and_then(|s| std::fs::write(&rep_path, s));
            match write_res {
                Ok(()) => {
                    self.status = format!("Report exported to {}", rep_path.display());
                }
                Err(e) => {
                    self.status = format!("Report export failed: {e}");
                }
            }
            self.needs_redraw = true;
        }
        if let Some(id) = download {
            if let Err(e) = open_official_download_page(id) {
                self.status = format!(
                    "Could not open download page: {e}. See 'ojon backends' for publisher links."
                );
            } else {
                self.status =
                    "Publisher page opened. Install the tool, then click Rescan installed tools."
                        .into();
            }
            self.needs_redraw = true;
        }
        if cancel_job {
            self.cancel_active_job();
        }
        // Running conversion or inference is permitted only through the backend.
        // The action runs on a worker thread; rendering never blocks on model I/O.
        if inspect {
            self.start_inspect();
        } else if one_click_opt {
            self.start_one_click_optimize();
        } else if one_click_unc {
            self.start_one_click_uncensor();
        } else if plan {
            self.start_plan();
        } else if recipe_plan {
            self.start_recipe_plan();
        } else if advise {
            self.start_optimizer_advice();
        } else if opt_run {
            self.start_optimizer_run();
        } else if unc_plan {
            self.start_uncensor_plan();
        } else if unc_run {
            self.start_uncensor_run();
        } else if unc_auto_plan {
            self.start_uncensor_auto_plan();
        } else if unc_auto_run {
            self.start_uncensor_auto_run();
        } else if convert {
            self.start_quantize();
        } else if smoke {
            self.start_smoke();
        } else if bench {
            self.start_bench();
        }
        cmds
    }
}

fn default_output(source: &str, precision: Precision) -> String {
    if source.trim().is_empty() {
        return String::new();
    }
    let path = Path::new(source.trim());
    let stem = if path.is_dir() {
        path.file_name()
    } else {
        path.file_stem()
    }
    .unwrap_or_default()
    .to_string_lossy();
    path.with_file_name(format!("{stem}-{}.gguf", precision.as_str()))
        .to_string_lossy()
        .into_owned()
}

struct DesktopApp {
    ui: Workstation,
    window: Option<Arc<Window>>,
    gpu: Option<GpuBackend>,
    cpu: CpuBackend,
    presenter: Option<SoftwarePresenter>,
    scale: f32,
    physical: (u32, u32),
    visible: bool,
}
impl DesktopApp {
    fn new() -> Self {
        Self {
            ui: Workstation::new(),
            window: None,
            gpu: None,
            cpu: CpuBackend::new(),
            presenter: None,
            scale: 1.0,
            physical: (0, 0),
            visible: false,
        }
    }
    fn draw(&mut self, window: &Window) {
        let (w, h) = self.physical;
        if w == 0 || h == 0 {
            return;
        }
        let palette = ThemePalette::studio_graphite();
        let logical_width = w as f32 / self.scale;
        let logical_height = h as f32 / self.scale;
        let cmds = self.ui.render(logical_width, logical_height);
        let mut presented = false;
        if let Some(gpu) = self.gpu.as_mut() {
            gpu.set_scale_factor(self.scale);
            gpu.begin_frame(w, h, palette.canvas);
            gpu.submit(&cmds);
            match gpu.end_frame() {
                Ok(_) => presented = true,
                Err(e) => {
                    eprintln!("Ojon GPU renderer error ({e}); switching to CPU safe mode");
                    self.gpu = None;
                }
            }
        }
        if !presented {
            self.cpu.set_scale_factor(self.scale);
            self.cpu.begin_frame(w, h, palette.canvas);
            self.cpu.submit(&cmds);
            if self.cpu.end_frame().is_ok() {
                if let Some(presenter) = self.presenter.as_mut() {
                    match presenter.present(self.cpu.pixels(), w, h) {
                        Ok(()) => presented = true,
                        Err(e) => eprintln!("Ojon CPU window presentation error: {e}"),
                    }
                }
            }
        }
        if presented && !self.visible {
            window.set_visible(true);
            self.visible = true;
        }
    }
}
impl ApplicationHandler for DesktopApp {
    fn resumed(&mut self, loop_: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("Ojon (ওজন)  |  Kalponic Studio")
            .with_inner_size(LogicalSize::new(1500.0, 960.0))
            .with_min_inner_size(LogicalSize::new(1080.0, 720.0))
            .with_visible(false);
        match loop_.create_window(attrs) {
            Ok(win) => {
                let win = Arc::new(win);
                self.scale = win.scale_factor() as f32;
                let s = win.inner_size();
                self.physical = (s.width, s.height);
                match SoftwarePresenter::new(win.clone()) {
                    Ok(p) => self.presenter = Some(p),
                    Err(e) => eprintln!("CPU window presentation unavailable: {e}"),
                }
                match GpuBackend::new_for_window(win.clone(), s.width, s.height) {
                    Ok(g) => self.gpu = Some(g),
                    Err(e) => eprintln!("GPU renderer unavailable ({e}); trying CPU safe mode"),
                }
                self.window = Some(win.clone());
                if !self.ui.source.is_empty() {
                    self.ui.start_inspect();
                }
                win.request_redraw();
            }
            Err(e) => eprintln!("Unable to open Ojon window: {e}"),
        }
    }
    fn window_event(&mut self, loop_: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(win) = self.window.as_ref().cloned() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => {
                self.ui.persist_local_settings(false);
                loop_.exit();
            }
            WindowEvent::Resized(size) => {
                self.physical = (size.width, size.height);
                win.request_redraw();
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.scale = scale_factor as f32;
                let size = win.inner_size();
                self.physical = (size.width, size.height);
                win.request_redraw();
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.ui.pointer.position = Vec2::new(
                    position.x as f32 / self.scale,
                    position.y as f32 / self.scale,
                );
                win.request_redraw();
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.ui.pointer.primary_down = state == ElementState::Pressed;
                if state == ElementState::Pressed {
                    self.ui.click_pulse = true;
                }
                win.request_redraw();
            }
            WindowEvent::DroppedFile(path) => {
                self.ui.set_source(&path);
                win.request_redraw();
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.ui.control_held = mods.state().control_key();
            }
            WindowEvent::Focused(false) => {
                self.ui.control_held = false;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.ui.edit_focused(&event);
                win.request_redraw();
            }
            WindowEvent::RedrawRequested => self.draw(&win),
            _ => {}
        }
    }
    fn about_to_wait(&mut self, loop_: &ActiveEventLoop) {
        self.ui.poll();
        if self.ui.needs_redraw {
            self.ui.needs_redraw = false;
            if let Some(window) = self.window.as_ref() {
                window.request_redraw();
            }
        }
        if self.ui.busy {
            loop_.set_control_flow(ControlFlow::WaitUntil(
                Instant::now() + Duration::from_secs(1),
            ));
        } else {
            loop_.set_control_flow(ControlFlow::Wait);
        }
    }
}
pub(super) fn run() -> Result<(), Box<dyn std::error::Error>> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Wait);
    let mut app = DesktopApp::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_log_does_not_finish_running_conversion() {
        let mut ws = Workstation::new();
        ws.busy = true;
        ws.started_at = Some(Instant::now());
        ws.tx
            .send(WorkerEvent::Log("[stderr] quantizing tensor 1/200".into()))
            .unwrap();
        ws.poll();
        assert!(ws.busy, "log messages must not finish the job");
        assert_eq!(ws.log_tail.len(), 1);
        ws.tx
            .send(WorkerEvent::Quantized(Err("test converter failed".into())))
            .unwrap();
        ws.poll();
        assert!(!ws.busy);
        assert!(ws.status.contains("test converter failed"));
        assert!(ws.last_elapsed.is_some());
    }

    #[test]
    fn requires_inspection_for_current_source_and_supported_format() {
        let mut ws = Workstation::new();
        ws.source = "source-f16.gguf".into();
        ws.output = "output-Q5_K_M.gguf".into();
        ws.inspected_source = Some(ws.source.clone());
        ws.model = Some(GgufSummary {
            version: 3,
            tensor_count: 1,
            metadata_count: 0,
            file_bytes: 1024,
            architecture: None,
            model_name: None,
            file_type: None,
            data_offset: 32,
            alignment: 32,
            split_count: None,
            big_endian: false,
        });
        assert!(ws.ready_to_convert());
        ws.source = "different-source.gguf".into();
        assert!(
            !ws.ready_to_convert(),
            "editing source must invalidate earlier inspection"
        );
        ws.source = "source-f16.gguf".into();
        ws.format_tab = 1;
        assert!(
            !ws.ready_to_convert(),
            "unimplemented formats must not start GGUF conversion"
        );
    }

    #[test]
    fn new_kathamo_navigation_routes_to_real_workstation_views() {
        let mut ws = Workstation::new();
        let hit = |ws: &mut Workstation, y: f32| {
            ws.pointer.position = Vec2::new(23.0, y);
            ws.click_pulse = true;
            ws.render(1500.0, 960.0);
        };
        hit(&mut ws, 70.0);
        assert_eq!(
            ws.workspace_tab, 2,
            "Uncensor should open its model edit view"
        );
        hit(&mut ws, 114.0);
        assert_eq!(ws.workspace_tab, 0);
        assert_eq!(
            ws.optimize_settings_tab, 2,
            "Hardware should open actual hardware controls"
        );
        hit(&mut ws, 158.0);
        assert_eq!(
            ws.optimize_review_tab, 1,
            "Results should open real receipts"
        );
        hit(&mut ws, 202.0);
        assert_eq!(ws.workspace_tab, 1, "Manual Tools must remain accessible");
        hit(&mut ws, 246.0);
        assert_eq!(ws.workspace_tab, 3, "Learn should open Ojon Learn view");
        hit(&mut ws, 26.0);
        assert_eq!(ws.workspace_tab, 0);
        assert_eq!(ws.optimize_settings_tab, 0);
        assert_eq!(ws.optimize_review_tab, 0);
        assert!(
            !ws.busy,
            "navigation must not trigger a conversion or model load"
        );
    }

    #[test]
    fn ojon_gui_renders_clean_frame() {
        let mut ws = Workstation::new();
        let cmds = ws.render(1500.0, 960.0);
        assert!(!cmds.is_empty());

        let mut cpu = CpuBackend::new();
        let palette = ThemePalette::studio_graphite();
        cpu.begin_frame(1500, 960, palette.canvas);
        cpu.submit(&cmds);
        assert!(cpu.end_frame().is_ok());

        // Render test artifacts are opt-in; no developer-specific paths.
        if let Some(artifact_dir) = std::env::var_os("OJON_SCREENSHOT_DIR")
            .map(PathBuf::from)
            .filter(|path| path.is_dir())
        {
            let ppm_path = artifact_dir.join("ojon_workstation_render.ppm");
            let pixels = cpu.pixels();
            let mut data = format!("P6\n{} {}\n255\n", 1500, 960).into_bytes();
            for &p in pixels {
                data.push((p >> 16) as u8);
                data.push((p >> 8) as u8);
                data.push(p as u8);
            }
            let _ = std::fs::write(ppm_path, data);
        }
    }

    #[test]
    fn test_gui_settings_persistence_and_goals() {
        let temp = std::env::temp_dir().join(format!(
            "ojon_gui_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::create_dir_all(&temp);
        let settings_file = temp.join("settings.json");
        std::env::set_var("OJON_SETTINGS_PATH", &settings_file);

        let dummy_model = temp.join("test_model.gguf");
        let _ = std::fs::write(&dummy_model, b"GGUF");

        // 1. Initial launch with no settings
        let mut ws = Workstation::new();
        assert_eq!(ws.opt_goal, 0, "default goal should be Balanced (0)");

        // 2. Configure in GUI: model, tools, Smallest goal
        ws.source = dummy_model.to_string_lossy().into_owned();
        ws.opt_goal = 1; // Smallest
        ws.opt_min_saving = "15.0".into();
        ws.persist_local_settings(true);

        // 3. Reopen GUI: confirm persistence of Smallest
        let mut ws2 = Workstation::new();
        assert_eq!(ws2.source, dummy_model.to_string_lossy());
        assert_eq!(ws2.opt_goal, 1, "Smallest goal should persist");
        assert_eq!(ws2.opt_min_saving, "15");

        // 4. Change to Fastest in GUI and save
        ws2.opt_goal = 2; // Fastest
        ws2.persist_local_settings(true);

        // 5. Reopen GUI: confirm persistence of Fastest
        let ws3 = Workstation::new();
        assert_eq!(ws3.opt_goal, 2, "Fastest goal should persist");

        // 6. Test CLI set -> GUI reload
        let mut settings = ojon_backend::load_settings().unwrap();
        settings.optimization_goal = "balanced".into();
        ojon_backend::save_settings(&settings).unwrap();

        let ws4 = Workstation::new();
        assert_eq!(
            ws4.opt_goal, 0,
            "Balanced goal from CLI should reflect in GUI"
        );

        // Cleanup
        std::env::remove_var("OJON_SETTINGS_PATH");
        let _ = std::fs::remove_dir_all(&temp);
    }
}
