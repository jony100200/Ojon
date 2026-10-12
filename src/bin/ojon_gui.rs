//! Ojon's native workstation shell, powered by Kathamo.
//! This is an optional GUI binary, not a model quantization engine.
//! All model details displayed here come from inspected files. No demo benchmarks.

use ojon_backend::{inspect_gguf, prepare_gguf, quantize_gguf_with_log, inspect_hf_folder, prepare_hf, quantize_hf_with_log, hf_python_path, smoke_gguf_capture, probe_model};
use ojon_backend::{GgufSummary, HfFolderInfo, ModelProbe, HfQuantizeResult, SmokeCapture, Precision, QuantizeResult, FoundBackend, BackendId, discover_backends, open_official_download_page};
use std::path::{Path, PathBuf};
use std::collections::VecDeque;
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
use ui_widgets::{icon_cache::IconAtlas, KathamoUi};


// Native Windows clipboard: cheap path pasting without a browser/UI runtime.
// Safe only for CF_UNICODETEXT global-memory clipboard handles.
#[cfg(windows)]
fn clipboard_text() -> Option<String> {
    use windows_sys::Win32::System::DataExchange::{
        CloseClipboard, GetClipboardData, OpenClipboard,
    };
    use windows_sys::Win32::System::Memory::{GlobalLock, GlobalSize, GlobalUnlock};
    unsafe {
        if OpenClipboard(0) == 0 { return None; }
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
        while len < limit && *ptr.add(len) != 0 { len += 1; }
        let value = String::from_utf16_lossy(std::slice::from_raw_parts(ptr, len));
        GlobalUnlock(global);
        CloseClipboard();
        Some(value)
    }
}

#[cfg(not(windows))]
fn clipboard_text() -> Option<String> { None }

#[derive(Clone, Copy, PartialEq, Eq)]
enum Focus { Model, Output, Quantizer, Converter, Runtime, Prompt, GpuLayers }

enum WorkerEvent {
    Inspected(Result<(GgufSummary, ModelProbe), String>),
    HfInspected(Result<(HfFolderInfo, ModelProbe), String>),
    StandaloneProbed(Result<ModelProbe, String>),
    Planned(Result<String, String>),
    Quantized(Result<QuantizeResult, String>),
    HfQuantized(Result<HfQuantizeResult, String>),
    Smoke(Result<SmokeCapture, String>),
    Log(String),
}

struct Workstation {
    source: String,
    output: String,
    quantizer: String,
    converter: String,
    runtime: String,
    prompt: String,
    gpu_layers: String,
    recipe: usize,
    format_tab: usize,
    focus: Option<Focus>,
    model: Option<GgufSummary>,
    hf_folder: Option<HfFolderInfo>,
    probe: Option<ModelProbe>,
    last_inference: Option<String>,
    inspected_source: Option<String>,
    last_output: Option<(PathBuf, u64)>,
    started_at: Option<Instant>,
    last_elapsed: Option<Duration>,
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
        let mut source = String::new();
        let backends = discover_backends();
        let mut quantizer = backends.iter()
            .find(|b| b.spec.id == BackendId::LlamaQuantize)
            .and_then(|b| b.executable.as_ref())
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| "llama-quantize".to_owned());
        let mut converter = backends.iter()
            .find(|b| b.spec.id == BackendId::HfConverter)
            .and_then(|b| b.executable.as_ref())
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| "convert_hf_to_gguf.py".to_owned());
        let mut runtime = backends.iter()
            .find(|b| b.spec.id == BackendId::LlamaCli)
            .and_then(|b| b.executable.as_ref())
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| "llama-cli".to_owned());
        let mut args = std::env::args_os().skip(1);
        while let Some(key) = args.next() {
            match key.to_str() {
                Some("--model") => if let Some(s) = args.next() { source = s.to_string_lossy().into_owned(); },
                Some("--quantizer") => if let Some(s) = args.next() { quantizer = s.to_string_lossy().into_owned(); },
                Some("--converter") => if let Some(s) = args.next() { converter = s.to_string_lossy().into_owned(); },
                Some("--runtime") => if let Some(s) = args.next() { runtime = s.to_string_lossy().into_owned(); },
                _ => {}
            }
        }
        let output = default_output(&source, Precision::default());
        Self {
            source, output, quantizer, converter, runtime,
            prompt: "Explain in one sentence what a quantized language model is.".into(),
            gpu_layers: "0".into(),
            recipe: 0,
            format_tab: 0,
            focus: None,
            model: None,
            hf_folder: None,
            probe: None,
            last_inference: None,
            inspected_source: None,
            last_output: None,
            started_at: None,
            last_elapsed: None,
            log_tail: VecDeque::new(),
            status: "Drop a GGUF file or HF model folder here, or enter a path.".into(),
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
    fn rescan_backends(&mut self) {
        self.backends = discover_backends();
        if self.converter.trim() == "convert_hf_to_gguf.py" || self.converter.trim().is_empty() {
            if let Some(path) = self.backends.iter()
                .find(|b| b.spec.id == BackendId::HfConverter)
                .and_then(|b| b.executable.as_ref()) {
                self.converter = path.to_string_lossy().into_owned();
            }
        }
        if self.quantizer.trim() == "llama-quantize" || self.quantizer.trim().is_empty() {
            if let Some(path) = self.backends.iter()
                .find(|b| b.spec.id == BackendId::LlamaQuantize)
                .and_then(|b| b.executable.as_ref()) {
                self.quantizer = path.to_string_lossy().into_owned();
            }
        }
        if self.runtime.trim() == "llama-cli" || self.runtime.trim().is_empty() {
            if let Some(path) = self.backends.iter()
                .find(|b| b.spec.id == BackendId::LlamaCli)
                .and_then(|b| b.executable.as_ref()) {
                self.runtime = path.to_string_lossy().into_owned();
            }
        }
        let installed = self.backends.iter().filter(|b| b.installed()).count();
        self.status = format!("{installed} known backend executable(s) detected. Discovery alone does not confirm compatibility.");
        self.needs_redraw = true;
    }

    #[allow(dead_code)]
    fn hf_converter(&self) -> &str { &self.converter }
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
            self.status = "A job is already running. Finish it before importing a different model.".into();
            self.needs_redraw = true;
            return;
        }
        self.source = path.to_string_lossy().into_owned();
        self.output = default_output(&self.source, self.precision());
        self.model = None;
        self.hf_folder = None;
        self.probe = None;
        self.last_inference = None;
        self.inspected_source = None;
        self.last_output = None;
        self.status = "Model selected. Inspect it before converting.".into();
        self.needs_redraw = true;
        self.start_inspect();
    }
    fn start_inspect(&mut self) {
        if self.busy { return; }
        self.inspected_source = None;
        self.model = None;
        self.hf_folder = None;
        self.probe = None;
        self.last_output = None;
        let model = PathBuf::from(self.source.trim());
        if self.source.trim().is_empty() { self.status = "Choose a GGUF file or local HF model folder first.".into(); return; }
        let tx = self.tx.clone();
        if self.output.trim().is_empty() {
            self.output = default_output(&self.source, self.precision());
        }
        self.status = "Reading model metadata without loading tensor weights...".into();
        self.inspected_source = Some(self.source.clone());
        self.started_at = Some(Instant::now());
        self.busy = true;
        self.needs_redraw = true;
        std::thread::spawn(move || {
            if model.is_dir() {
                let result = inspect_hf_folder(&model)
                    .and_then(|info| probe_model(&model).map(|p| (info,p)))
                    .map_err(|e| e.to_string());
                let _ = tx.send(WorkerEvent::HfInspected(result));
            } else if model.extension().and_then(|s| s.to_str())
                .is_some_and(|s| s.eq_ignore_ascii_case("gguf")) {
                let result = inspect_gguf(&model)
                    .and_then(|info| probe_model(&model).map(|p|(info,p)))
                    .map_err(|e| e.to_string());
                let _ = tx.send(WorkerEvent::Inspected(result));
            } else {
                let result = probe_model(&model).map_err(|e|e.to_string());
                let _ = tx.send(WorkerEvent::StandaloneProbed(result));
            }
        });
    }
    fn start_plan(&mut self) {
        if self.busy { return; }
        if !self.ready_to_convert() {
            self.status = "Import and inspect a compatible GGUF or HF folder before planning. GGUF is the only connected quantization format.".into();
            self.needs_redraw = true;
            return;
        }
        let (source, output, tool, converter, python, precision) = (
            self.source.clone(), self.output.clone(), self.quantizer.clone(),
            self.converter.clone(), hf_python_path(), self.precision(),
        );
        let tx = self.tx.clone();
        self.busy = true;
        self.started_at = Some(Instant::now());
        self.status = "Checking source/output safety with installed public backends...".into();
        self.needs_redraw = true;
        std::thread::spawn(move || {
            let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4);
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
            }.map_err(|e| e.to_string());
            let _ = tx.send(WorkerEvent::Planned(res));
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
        if self.busy { return; }
        if !self.ready_to_convert() {
            self.status = "Inspect the selected model and choose GGUF before quantizing. This source/format is not ready.".into();
            self.needs_redraw = true;
            return;
        }
        let (source, output, tool, converter, python, precision) = (
            self.source.clone(), self.output.clone(), self.quantizer.clone(),
            self.converter.clone(), hf_python_path(), self.precision(),
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
            let threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4);
            if Path::new(source.trim()).is_dir() {
                let result = prepare_hf(
                    Path::new(source.trim()), Path::new(output.trim()), Path::new(converter.trim()),
                    &python, Path::new(tool.trim()), precision, threads,
                ).and_then(|plan| quantize_hf_with_log(&plan, |line| {
                    let _ = tx.send(WorkerEvent::Log(line.to_owned()));
                })).map_err(|e| e.to_string());
                let _ = tx.send(WorkerEvent::HfQuantized(result));
            } else {
                let result = prepare_gguf(
                    Path::new(source.trim()), Path::new(output.trim()),
                    Path::new(tool.trim()), precision, threads,
                ).and_then(|plan| quantize_gguf_with_log(&plan, |line| {
                    let _ = tx.send(WorkerEvent::Log(line.to_owned()));
                })).map_err(|e| e.to_string());
                let _ = tx.send(WorkerEvent::Quantized(result));
            }
        });
    }
    fn start_smoke(&mut self) {
        if self.busy { return; }
        let gpu_layers = match self.gpu_layers.trim().parse::<u32>() {
            Ok(v) if v <= 512 => v,
            _ => {
                self.status = "GPU layers must be an integer from 0 to 512.".into();
                self.needs_redraw = true;
                return;
            }
        };
        let (output, runtime, prompt) =
            (self.output.clone(), self.runtime.clone(), self.prompt.clone());
        let tx = self.tx.clone();
        self.busy = true;
        self.started_at = Some(Instant::now());
        self.last_elapsed = None;
        self.log_tail.clear();
        self.last_inference = None;
        self.status = "Running real llama-cli inference. This is not a quality benchmark.".into();
        self.needs_redraw = true;
        std::thread::spawn(move || {
            let result = smoke_gguf_capture(Path::new(output.trim()), Path::new(runtime.trim()), &prompt, 32, gpu_layers)
                .map_err(|e| e.to_string());
            let _ = tx.send(WorkerEvent::Smoke(result));
        });
    }
    fn poll(&mut self) {
        while let Ok(message) = self.rx.try_recv() {
            self.needs_redraw = true;
            if let WorkerEvent::Log(line) = message {
                let truncated: String = line.chars()
                    .filter(|ch| !ch.is_control() || *ch == ' ')
                    .take(180).collect();
                if !truncated.trim().is_empty() {
                    self.log_tail.push_back(truncated);
                    while self.log_tail.len() > 8 { self.log_tail.pop_front(); }
                }
                continue; // Log events MUST NOT set busy=false.
            }
            self.busy = false;
            self.last_elapsed = self.started_at.take().map(|start| start.elapsed());
            match message {
                WorkerEvent::Inspected(Ok((model, probe))) => {
                    self.status = format!("Read GGUF v{}: {} tensors; likely category {} ({}). No weights loaded.",
                        model.version, model.tensor_count, probe.category.as_str(), probe.confidence);
                    self.model = Some(model);
                    self.hf_folder = None;
                    self.probe = Some(probe);
                }
                WorkerEvent::HfInspected(Ok((info, probe))) => {
                    self.status = format!("HF model folder: {} weight files; likely category {} ({}). No weights loaded.",
                        info.weight_files, probe.category.as_str(), probe.confidence);
                    self.model = None;
                    self.hf_folder = Some(info);
                    self.probe = Some(probe);
                }
                WorkerEvent::StandaloneProbed(Ok(probe)) => {
                    self.status = format!("Metadata inspected: {} ({}). {}",
                        probe.category.as_str(), probe.confidence, probe.conversion_status);
                    self.model = None;
                    self.hf_folder = None;
                    self.probe = Some(probe);
                }
                WorkerEvent::StandaloneProbed(Err(e)) => {
                    self.probe = None;
                    self.status = format!("Unsupported input or inspection failed: {e}");
                }
                WorkerEvent::HfInspected(Err(e)) => {
                    self.hf_folder = None;
                    self.status = format!("HF folder inspection failed: {e}");
                }
                WorkerEvent::Inspected(Err(e)) => {
                    self.model = None;
                    self.status = format!("Inspection failed: {e}");
                }
                WorkerEvent::Planned(Ok(text)) => self.status = text,
                WorkerEvent::Planned(Err(e)) => self.status = format!("Preflight failed: {e}"),
                WorkerEvent::Quantized(Ok(done)) => {
                    self.last_output = Some((done.output.clone(), done.output_info.file_bytes));
                    self.status = format!(
                        "Converter completed. GGUF catalog checked: {} tensors, {} bytes. Runtime loading NOT verified.",
                        done.output_info.tensor_count, done.output_info.file_bytes
                    );
                }
                WorkerEvent::Quantized(Err(e)) => self.status = format!("Conversion failed: {e}"),
                WorkerEvent::HfQuantized(Ok(done)) => {
                    self.last_output = Some((done.output.clone(), done.info.file_bytes));
                    self.status = format!(
                        "HF conversion + quantization completed: {} bytes, {} tensors. Runtime load NOT verified.",
                        done.info.file_bytes, done.info.tensor_count,
                    );
                }
                WorkerEvent::HfQuantized(Err(e)) => self.status = format!("HF conversion failed: {e}"),
                WorkerEvent::Smoke(Ok(capture)) => {
                    self.last_inference = Some(capture.stdout);
                    self.status = "Real llama-cli output captured. Review the generated text; quality is NOT evaluated.".into();
                }
                WorkerEvent::Smoke(Err(e)) => self.status = format!("Inference smoke check failed: {e}"),
                WorkerEvent::Log(_) => unreachable!("log messages handled above"),
            }
        }
        // One render tick per second while processing, even when a backend is
        // temporarily silent. No continuous idle repaint or invented metrics.
        if self.busy { self.needs_redraw = true; }
    }
    fn edit_focused(&mut self, event: &ui_platform::winit::event::KeyEvent) {
        if event.state != ElementState::Pressed { return; }
        if matches!(event.logical_key, Key::Named(NamedKey::Escape)) { self.focus = None; self.select_all = false; return; }
        let Some(focus) = self.focus else { return; };
        let dest = match focus {
            Focus::Model => &mut self.source,
            Focus::Output => &mut self.output,
            Focus::Quantizer => &mut self.quantizer,
            Focus::Converter => &mut self.converter,
            Focus::Runtime => &mut self.runtime,
            Focus::Prompt => &mut self.prompt,
            Focus::GpuLayers => &mut self.gpu_layers,
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
                        if self.select_all { dest.clear(); }
                        for ch in value.chars().filter(|ch| !ch.is_control()) {
                            if dest.len() < 4096 { dest.push(ch); }
                        }
                        self.select_all = false;
                        self.needs_redraw = true;
                    }
                    return;
                }
            }
            return;
        }
        if self.select_all { dest.clear(); self.select_all = false; }
        if matches!(event.logical_key, Key::Named(NamedKey::Backspace)) {
            dest.pop();
        } else if let Some(value) = &event.text {
            for ch in value.chars().filter(|ch| !ch.is_control()) {
                if dest.len() < 4096 { dest.push(ch); }
            }
        }
        self.needs_redraw = true;
    }
    fn render(&mut self, width: f32, height: f32) -> CommandBuffer {
        let source_is_current = self.inspected_source.as_deref() == Some(self.source.as_str());
        let previous_precision = self.precision();
        let auto_generated_output = self.output == default_output(&self.source, previous_precision);
        let mut cmds = CommandBuffer::new();
        let palette = ThemePalette::studio_graphite();
        let mut click_pointer = self.pointer;
        click_pointer.primary_down = self.click_pulse;
        self.click_pulse = false;
        let viewport = Rect::from_min_size(Vec2::ZERO, Vec2::new(width.max(700.0), height.max(500.0)));
        let source = &mut self.source;
        let output = &mut self.output;
        let quantizer = &mut self.quantizer;
        let converter = &mut self.converter;
        let runtime = &mut self.runtime;
        let prompt = &mut self.prompt;
        let gpu_layers = &mut self.gpu_layers;
        let recipe = &mut self.recipe;
        let format_tab = &mut self.format_tab;
        let model = if source_is_current { self.model.as_ref() } else { None };
        let hf_folder = if source_is_current { self.hf_folder.as_ref() } else { None };
        let model_probe = if source_is_current { self.probe.as_ref() } else { None };
        let last_inference = &self.last_inference;
        let status = &self.status;
        let busy = self.busy;
        let backends = &self.backends;
        let last_output = if source_is_current { self.last_output.as_ref() } else { None };
        let log_tail = &self.log_tail;
        let job_elapsed = self.started_at.map(|t| t.elapsed()).or(self.last_elapsed);
        let mut rescan = false;
        let mut download: Option<BackendId> = None;
        let mut new_focus = self.focus;
        let (mut inspect, mut plan, mut convert, mut smoke) = (false, false, false, false);
        {
            let mut ui = KathamoUi::new(&mut cmds, &mut self.icons, &palette, viewport, click_pointer);

            // 1. Studio Telemetry & Header Bar
            ui.horizontal(72.0, |top| {
                top.vertical(280.0, |brand| {
                    brand.bilingual_heading(&BilingualLabel::new("Ojon", "ওজন"));
                    brand.horizontal(22.0, |pills| {
                        pills.status_pill(if busy { "Processing" } else { "Idle" }, !busy);
                        let found = backends.iter().filter(|b| b.installed()).count();
                        pills.status_pill(&format!("{found}/{} Tools Detected", backends.len()), found > 0);
                    });
                });

                let (model_title, model_sub) = if let Some(info) = model {
                    let name = info.model_name.as_deref().unwrap_or("GGUF Container");
                    let sub = format!("{:.2} GiB ({} Tensors)", info.file_bytes as f64 / 1_073_741_824.0, info.tensor_count);
                    (name, sub)
                } else if let Some(info) = hf_folder {
                    let sub = format!("{:.2} GiB ({} Shards)", info.source_weight_bytes as f64 / 1_073_741_824.0, info.weight_files);
                    ("HuggingFace Folder", sub)
                } else if let Some(p) = model_probe {
                    (p.identification.as_str(), format!("{:.2} GiB; {} (metadata only)", p.bytes as f64 / 1_073_741_824.0, p.category.as_str()))
                } else if !source.trim().is_empty() {
                    ("Not Inspected", "Press Inspect Model Metadata".into())
                } else {
                    ("No Model Loaded", "Drop File or Enter Path".into())
                };
                top.stat_card("SOURCE CHECKPOINT", model_title, &model_sub);

                let target_precision = match *recipe {
                    1 => "Q4_K_M (Selected)",
                    2 => "Q3_K_M (Selected)",
                    3 => "Q6_K (Selected)",
                    _ => "Q5_K_M (Default)",
                };
                top.stat_card("TARGET PRECISION",
                    if *format_tab == 0 { target_precision } else { "Adapter not connected" },
                    if *format_tab == 0 { "Not quality-benchmarked" } else { "GGUF only currently" });

                let quantizer_ready = Path::new(quantizer.trim()).is_file()
                    || (quantizer.trim() == "llama-quantize"
                        && backends.iter().any(|b| b.spec.id == BackendId::LlamaQuantize && b.installed()));
                top.stat_card("QUANTIZER TOOL", "llama-quantize", if quantizer_ready { "Executable Detected" } else { "Not Found / Unverified" });

                let smoke_ready = Path::new(runtime.trim()).is_file()
                    || (runtime.trim() == "llama-cli"
                        && backends.iter().any(|b| b.spec.id == BackendId::LlamaCli && b.installed()));
                top.stat_card("INFERENCE ENGINE", "llama-cli", if smoke_ready { "Executable Detected" } else { "Optional / Not Found" });
            });

            ui.separator();

            // 2. Main Two-Column Workstation Layout
            ui.columns(
                (height - 118.0).max(520.0),
                &[SizeConstraint::Fixed(385.0), SizeConstraint::Flex(1.0)],
                |idx, col| {
                    if idx == 0 {
                        col.glass_card(300.0, |card| {
                            card.heading("01   IMPORT & INSPECT");
                            card.label("Drop a GGUF, SafeTensors, or HF checkpoint folder.");
                            if card.text_input(source, if new_focus == Some(Focus::Model) { "Type/paste model path (Ctrl+V)..." } else { "Source model path (or drop file)..." }).clicked() {
                                new_focus = Some(Focus::Model);
                            }
                            if card.button("Inspect Model Metadata").clicked() { inspect = true; }
                            card.separator();
                            if let Some(p) = model_probe {
                                card.label(&format!("Category: {} ({})", p.category.as_str(), p.confidence));
                                card.label(&format!("Route: {}", p.conversion_status));
                            }
                            if let Some(info) = model {
                                card.horizontal(22.0, |b| {
                                    b.badge(info.architecture.as_deref().unwrap_or("GGUF"), true);
                                    b.badge(&format!("v{}", info.version), true);
                                    b.badge(&format!("{:.2} GiB", info.file_bytes as f64 / 1_073_741_824.0), true);
                                });
                                card.label(&format!("Model: {}", info.model_name.as_deref().unwrap_or("Unnamed")));
                                card.label(&format!("Tensors: {} verified descriptors", info.tensor_count));
                                card.status_pill("Metadata Read (Zero GPU Weights Loaded)", true);
                            } else if let Some(info) = hf_folder {
                                card.horizontal(22.0, |b| {
                                    b.badge("HuggingFace", true);
                                    b.badge(&format!("{} Shards", info.weight_files), true);
                                    b.badge(&format!("{:.2} GiB", info.source_weight_bytes as f64 / 1_073_741_824.0), true);
                                });
                                card.label("Official convert_hf_to_gguf.py pipeline");
                                card.status_pill("Checkpoint Folder Structure Verified", true);
                            } else if model_probe.is_some() {
                                card.label("Only metadata inspection available for this source.");
                                card.status_pill("No compatible conversion adapter connected", false);
                            } else {
                                card.label("No model inspected yet.");
                                card.status_pill("Waiting for GGUF or HF model", false);
                            }
                        });

                        col.card(370.0, |panel| {
                            panel.heading("BACKENDS  /  AUTO-DETECT");
                            if panel.button_secondary("Rescan Installed Tools").clicked() { rescan = true; }
                            for backend in backends.iter() {
                                let label = match backend.spec.id {
                                    BackendId::LlamaQuantize => "GGUF converter",
                                    BackendId::LlamaCli => "GGUF inference",
                                    BackendId::LlamaPerplexity => "GGUF quality / PPL",
                                    BackendId::HfConverter => "HF to GGUF script",
                                    BackendId::LlamaFactory => "LoRA / QLoRA trainer",
                                    BackendId::StableDiffusionCpp => "Image diffusion",
                                    BackendId::WhisperCpp => "Speech / Whisper",
                                };
                                panel.horizontal(22.0, |row| {
                                    row.label(&format!("{}:", label));
                                    row.badge(if backend.installed() { "Detected" } else { "Not Found" }, backend.installed());
                                });
                                if !backend.installed() {
                                    if panel.button_secondary(&format!("Official download: {label}")).clicked() {
                                        download = Some(backend.spec.id);
                                    }
                                } else if backend.spec.integration == ojon_backend::Integration::DiscoveryOnly {
                                    panel.label("Detected; Ojon adapter not connected yet");
                                }
                            }
                        });
                    } else {
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
                        });

                        col.glass_card(260.0, |card| {
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
                    }
                },
            );
        }
        if auto_generated_output && previous_precision != self.precision() {
            self.output = default_output(&self.source, self.precision());
            self.last_output = None;
        }
        if self.focus != new_focus { self.select_all = false; }
        self.focus = new_focus;
        if rescan { self.rescan_backends(); }
        if let Some(id) = download {
            if let Err(e) = open_official_download_page(id) {
                self.status = format!("Could not open download page: {e}. See 'ojon backends' for publisher links.");
            } else {
                self.status = "Publisher page opened. Install the tool, then click Rescan installed tools.".into();
            }
            self.needs_redraw = true;
        }
        // Running conversion or inference is permitted only through the backend.
        // The action runs on a worker thread; rendering never blocks on model I/O.
        if inspect { self.start_inspect(); }
        else if plan { self.start_plan(); }
        else if convert { self.start_quantize(); }
        else if smoke { self.start_smoke(); }
        cmds
    }
}

fn default_output(source: &str, precision: Precision) -> String {
    if source.trim().is_empty() { return String::new(); }
    let path = Path::new(source.trim());
    let stem = if path.is_dir() { path.file_name() } else { path.file_stem() }
        .unwrap_or_default().to_string_lossy();
    path.with_file_name(format!("{stem}-{}.gguf", precision.as_str()))
        .to_string_lossy().into_owned()
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
            ui: Workstation::new(), window: None, gpu: None,
            cpu: CpuBackend::new(), presenter: None,
            scale: 1.0, physical: (0, 0), visible: false,
        }
    }
    fn draw(&mut self, window: &Window) {
        let (w, h) = self.physical;
        if w == 0 || h == 0 { return; }
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
        if presented && !self.visible { window.set_visible(true); self.visible = true; }
    }
}
impl ApplicationHandler for DesktopApp {
    fn resumed(&mut self, loop_: &ActiveEventLoop) {
        if self.window.is_some() { return; }
        let attrs = WindowAttributes::default()
            .with_title("Ojon (ওজন)  |  Kalponic Studio")
            .with_inner_size(LogicalSize::new(1340.0, 840.0))
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
                if !self.ui.source.is_empty() { self.ui.start_inspect(); }
                win.request_redraw();
            }
            Err(e) => eprintln!("Unable to open Ojon window: {e}"),
        }
    }
    fn window_event(&mut self, loop_: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(win) = self.window.as_ref().cloned() else { return; };
        match event {
            WindowEvent::CloseRequested => loop_.exit(),
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
                self.ui.pointer.position =
                    Vec2::new(position.x as f32 / self.scale, position.y as f32 / self.scale);
                win.request_redraw();
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.ui.pointer.primary_down = state == ElementState::Pressed;
                if state == ElementState::Pressed { self.ui.click_pulse = true; }
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
            if let Some(window) = self.window.as_ref() { window.request_redraw(); }
        }
        if self.ui.busy {
            loop_.set_control_flow(ControlFlow::WaitUntil(Instant::now() + Duration::from_secs(1)));
        } else {
            loop_.set_control_flow(ControlFlow::Wait);
        }
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
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
        ws.tx.send(WorkerEvent::Log("[stderr] quantizing tensor 1/200".into())).unwrap();
        ws.poll();
        assert!(ws.busy, "log messages must not finish the job");
        assert_eq!(ws.log_tail.len(), 1);
        ws.tx.send(WorkerEvent::Quantized(Err("test converter failed".into()))).unwrap();
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
            version:3, tensor_count: 1, metadata_count: 0, file_bytes: 1024,
            architecture:None, model_name:None, file_type:None,
            data_offset:32, alignment:32, split_count:None, big_endian:false
        });
        assert!(ws.ready_to_convert());
        ws.source = "different-source.gguf".into();
        assert!(!ws.ready_to_convert(), "editing source must invalidate earlier inspection");
        ws.source = "source-f16.gguf".into();
        ws.format_tab = 1;
        assert!(!ws.ready_to_convert(), "unimplemented formats must not start GGUF conversion");
    }

    #[test]
    fn ojon_gui_renders_clean_frame() {
        let mut ws = Workstation::new();
        let cmds = ws.render(1340.0, 840.0);
        assert!(!cmds.is_empty());

        let mut cpu = CpuBackend::new();
        let palette = ThemePalette::studio_graphite();
        cpu.begin_frame(1340, 840, palette.canvas);
        cpu.submit(&cmds);
        assert!(cpu.end_frame().is_ok());

        // Render test artifacts are opt-in; no developer-specific paths.
        if let Some(artifact_dir) = std::env::var_os("OJON_SCREENSHOT_DIR").map(PathBuf::from)
            .filter(|path| path.is_dir()) {
            let ppm_path = artifact_dir.join("ojon_workstation_render.ppm");
            let pixels = cpu.pixels();
            let mut data = format!("P6\n{} {}\n255\n", 1340, 840).into_bytes();
            for &p in pixels {
                data.push((p >> 16) as u8);
                data.push((p >> 8) as u8);
                data.push(p as u8);
            }
            let _ = std::fs::write(ppm_path, data);
        }
    }
}
