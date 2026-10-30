//! Lightweight, read-only discovery of independently installed AI backends.
//!
//! Detecting an executable is NOT proof of its model/format compatibility.
//! No binary is executed, installed, downloaded or recursively searched for.

use crate::jobs::capture_with_timeout;
use crate::llama::Precision;
use crate::settings::{load_settings, LocalSettings};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Duration;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Integration {
    /// Ojon already knows how to call this tool for one verified format recipe.
    Wired,
    /// Display and download support only; conversion/test adapter is not implemented.
    DiscoveryOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BackendId {
    LlamaQuantize,
    LlamaCli,
    LlamaMtmdCli,
    LlamaServer,
    LlamaBench,
    LlamaPerplexity,
    LlamaEmbedding,
    LlamaImatrix,
    HfConverter,
    LlamaFactory,
    StableDiffusionCpp,
    WhisperCpp,
    LlamaCvectorGenerator,
    IntelAutoRound,
    IntelSyclCli,
}

#[derive(Clone, Copy, Debug)]
pub struct BackendSpec {
    pub id: BackendId,
    pub name: &'static str,
    pub executable_names: &'static [&'static str],
    pub override_env: &'static str,
    pub website: &'static str,
    pub integration: Integration,
    pub role: &'static str,
}

const SPECS: [BackendSpec; 15] = [
    BackendSpec {
        id: BackendId::LlamaQuantize,
        name: "llama.cpp quantizer",
        executable_names: &["llama-quantize", "llama-quantize.exe"],
        override_env: "OJON_LLAMA_QUANTIZE",
        website: "https://github.com/ggml-org/llama.cpp/releases",
        integration: Integration::Wired,
        role: "GGUF quantization: Q3_K_M / Q4_K_M / Q5_K_M / Q6_K",
    },
    BackendSpec {
        id: BackendId::LlamaCli,
        name: "llama.cpp inference CLI",
        executable_names: &["llama-cli", "llama-cli.exe"],
        override_env: "OJON_LLAMA_CLI",
        website: "https://github.com/ggml-org/llama.cpp/releases",
        integration: Integration::Wired,
        role: "GGUF text-model inference smoke check",
    },
    BackendSpec {
        id: BackendId::LlamaMtmdCli,
        name: "llama.cpp multimodal CLI",
        executable_names: &["llama-mtmd-cli", "llama-mtmd-cli.exe"],
        override_env: "OJON_LLAMA_MTMD_CLI",
        website: "https://github.com/ggml-org/llama.cpp/blob/master/tools/mtmd/README.md",
        integration: Integration::Wired,
        role: "VLM vision-suite scoring with --mmproj and --image; preferred over llama-cli for quant-parity vision tests",
    },
    BackendSpec {
        id: BackendId::LlamaServer,
        name: "llama.cpp persistent evaluation server",
        executable_names: &["llama-server", "llama-server.exe"],
        override_env: "OJON_LLAMA_SERVER",
        website: "https://github.com/ggml-org/llama.cpp",
        integration: Integration::Wired,
        role: "Local loopback session, one GGUF load per held-out task suite",
    },
    BackendSpec {
        id: BackendId::LlamaBench,
        name: "llama.cpp performance benchmark",
        executable_names: &["llama-bench", "llama-bench.exe"],
        override_env: "OJON_LLAMA_BENCH",
        website: "https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md",
        integration: Integration::Wired,
        role: "Real GGUF prefill/decode benchmark with comparable JSON receipts",
    },
    BackendSpec {
        id: BackendId::LlamaPerplexity,
        name: "llama.cpp perplexity evaluator",
        executable_names: &["llama-perplexity", "llama-perplexity.exe"],
        override_env: "OJON_LLAMA_PERPLEXITY",
        website: "https://github.com/ggml-org/llama.cpp",
        integration: Integration::Wired,
        role: "Real held-out text perplexity comparison between original and candidate GGUF variants",
    },
    BackendSpec {
        id: BackendId::LlamaEmbedding,
        name: "llama.cpp embedding and reranker scorer",
        executable_names: &["llama-embedding", "llama-embedding.exe"],
        override_env: "OJON_LLAMA_EMBEDDING",
        website: "https://github.com/ggml-org/llama.cpp",
        integration: Integration::Wired,
        role: "Embedding / reranker parity: cosine, neighbour overlap and ranking agreement between original and candidate GGUF",
    },
    BackendSpec {
        id: BackendId::LlamaImatrix,
        name: "llama.cpp calibration imatrix",
        executable_names: &["llama-imatrix", "llama-imatrix.exe"],
        override_env: "OJON_LLAMA_IMATRIX",
        website: "https://github.com/ggml-org/llama.cpp",
        integration: Integration::Wired,
        role: "GGUF importance-matrix calibration for quality-aware low-bit quantization",
    },
    BackendSpec {
        id: BackendId::HfConverter,
        name: "llama.cpp Hugging Face to GGUF converter",
        executable_names: &["convert_hf_to_gguf.py"],
        override_env: "OJON_CONVERT_HF",
        website: "https://github.com/ggml-org/llama.cpp",
        integration: Integration::Wired,
        role: "Hugging Face model folder -> intermediate GGUF (requires Python and converter dependencies)",
    },
    BackendSpec {
        id: BackendId::LlamaFactory,
        name: "LLaMA-Factory LoRA/QLoRA trainer",
        executable_names: &["llamafactory-cli", "llamafactory-cli.exe", "llamafactory-cli.cmd"],
        override_env: "OJON_LLAMAFAC_CLI",
        website: "https://github.com/hiyouga/LlamaFactory",
        integration: Integration::Wired,
        role: "Optional local LLM/VLM LoRA or QLoRA supervised fine-tuning using user-supplied YAML",
    },
    BackendSpec {
        id: BackendId::StableDiffusionCpp,
        name: "stable-diffusion.cpp",
        executable_names: &["sd-cli", "sd-cli.exe"],
        override_env: "OJON_SD_CLI",
        website: "https://github.com/leejet/stable-diffusion.cpp",
        integration: Integration::Wired,
        role: "Compatible image/video safetensors -> GGUF conversion and real preview smoke via sd-cli",
    },
    BackendSpec {
        id: BackendId::WhisperCpp,
        name: "whisper.cpp",
        executable_names: &["whisper-cli", "whisper-cli.exe"],
        override_env: "OJON_WHISPER_CLI",
        website: "https://github.com/ggml-org/whisper.cpp",
        integration: Integration::DiscoveryOnly,
        role: "Speech support requires a separate compatible adapter",
    },
    BackendSpec {
        id: BackendId::LlamaCvectorGenerator,
        name: "llama.cpp control vector generator",
        executable_names: &["llama-cvector-generator", "llama-cvector-generator.exe"],
        override_env: "OJON_LLAMA_CVECTOR",
        website: "https://github.com/ggml-org/llama.cpp",
        integration: Integration::Wired,
        role: "Measured residual control vectors from paired GGUF prompts for directional edits",
    },
    BackendSpec {
        id: BackendId::IntelAutoRound,
        name: "Intel AutoRound (Adaptive / Mixed-Bit Quantizer)",
        executable_names: &["auto-round", "auto-round.exe"],
        override_env: "OJON_AUTOROUND",
        website: "https://github.com/intel/auto-round",
        integration: Integration::Wired,
        role: "Intel AutoRound adaptive / mixed-bit quantization for Intel XPU / CUDA",
    },
    BackendSpec {
        id: BackendId::IntelSyclCli,
        name: "llama.cpp SYCL (Intel Arc XMX)",
        executable_names: &["llama-cli", "llama-cli.exe"],
        override_env: "OJON_LLAMA_SYCL",
        website: "https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/SYCL.md",
        integration: Integration::Wired,
        role: "GGUF inference accelerated on Intel Arc GPU via native oneAPI SYCL and XMX engines",
    },
];

pub fn backend_specs() -> &'static [BackendSpec] {
    &SPECS
}

#[derive(Clone, Debug)]
pub struct FoundBackend {
    pub spec: BackendSpec,
    pub executable: Option<PathBuf>,
    pub found_via: Option<&'static str>,
}
impl FoundBackend {
    pub fn installed(&self) -> bool {
        self.executable.is_some()
    }
}

fn binary_at(path: &Path) -> bool {
    path.is_file()
}

/// Collect common executable folders without scanning entire disks.
/// OJON_BACKENDS_DIR can point to a parent with one directory of packaged tools.
/// The user can always enter a full executable path when the scan misses a tool.
fn search_dirs(prefs: Option<&LocalSettings>) -> Vec<(PathBuf, &'static str)> {
    let mut dirs: Vec<(PathBuf, &'static str)> = Vec::new();
    if let Some(settings) = prefs {
        for path in settings.backend_paths.values() {
            if let Some(parent) = Path::new(path).parent() {
                dirs.push((parent.to_path_buf(), "settings backend folder"));
            }
        }
    }
    if let Some(path) = env::var_os("PATH") {
        dirs.extend(
            env::split_paths(&path)
                .filter(|p| !p.as_os_str().is_empty())
                .map(|p| (p, "PATH")),
        );
    }
    if let Some(home) = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE")) {
        let home = PathBuf::from(home);
        for sub in [
            "bin",
            ".local/bin",
            "scoop/shims",
            "scoop/apps/llama.cpp/current/bin",
            "AppData/Local/Microsoft/WinGet/Links",
        ] {
            dirs.push((home.join(sub), "known user folder"));
        }
    }
    if let Some(program_files) = env::var_os("ProgramFiles") {
        dirs.push((
            PathBuf::from(program_files).join("llama.cpp"),
            "known install folder",
        ));
    }
    if let Some(local) = env::var_os("LOCALAPPDATA") {
        dirs.push((
            PathBuf::from(local).join("Programs").join("llama.cpp"),
            "known install folder",
        ));
    }
    if let Some(root) = env::var_os("OJON_LLAMA_CPP_DIR") {
        let root = PathBuf::from(root);
        dirs.push((root.clone(), "OJON_LLAMA_CPP_DIR"));
        dirs.push((root.join("bin"), "OJON_LLAMA_CPP_DIR"));
        dirs.push((root.join("build").join("bin"), "OJON_LLAMA_CPP_DIR"));
        dirs.push((
            root.join("build").join("bin").join("Release"),
            "OJON_LLAMA_CPP_DIR",
        ));
    }
    if let Some(root) = env::var_os("OJON_BACKENDS_DIR") {
        let root = PathBuf::from(root);
        dirs.push((root.clone(), "OJON_BACKENDS_DIR"));
        // Bounded search under a user explicitly chosen converter directory.
        // Includes common layouts such as <root>/llama.cpp/build/bin.
        if let Ok(iter) = fs::read_dir(&root) {
            for folder in iter
                .flatten()
                .take(64)
                .map(|e| e.path())
                .filter(|p| p.is_dir())
            {
                dirs.push((folder.clone(), "OJON_BACKENDS_DIR"));
                dirs.push((folder.join("bin"), "OJON_BACKENDS_DIR"));
                dirs.push((folder.join("build").join("bin"), "OJON_BACKENDS_DIR"));
                dirs.push((
                    folder.join("build").join("bin").join("Release"),
                    "OJON_BACKENDS_DIR",
                ));
            }
        }
    }
    // Check adjacent peer directories when repos are cloned side-by-side
    let mut base_roots = Vec::new();
    if let Ok(cwd) = env::current_dir() {
        base_roots.push(cwd.clone());
        if let Some(parent) = cwd.parent() {
            base_roots.push(parent.to_path_buf());
        }
    }
    if let Ok(exe) = env::current_exe() {
        let mut cur = exe.parent();
        for _ in 0..4 {
            if let Some(dir) = cur {
                if !base_roots.contains(&dir.to_path_buf()) {
                    base_roots.push(dir.to_path_buf());
                }
                cur = dir.parent();
            }
        }
    }
    for root in base_roots {
        for sub in [
            "bin",
            "llama.cpp",
            "llama.cpp/bin",
            "llama.cpp/build/bin",
            "llama.cpp/build/bin/Release",
        ] {
            dirs.push((root.join(sub), "adjacent workspace folder"));
        }
    }
    dirs
}

fn discover_one(
    spec: BackendSpec,
    dirs: &[(PathBuf, &'static str)],
    prefs: &LocalSettings,
) -> FoundBackend {
    // Explicit override is authoritative: do not secretly choose another installation.
    // An invalid override is reported as missing so the user can correct it.
    if let Some(override_path) = env::var_os(spec.override_env) {
        let candidate = PathBuf::from(override_path);
        return FoundBackend {
            spec,
            executable: if binary_at(&candidate) {
                Some(candidate)
            } else {
                None
            },
            found_via: Some("explicit environment override"),
        };
    }
    if let Some(saved) = prefs.backend(spec.override_env) {
        let candidate = PathBuf::from(saved);
        // Explicit local choice is authoritative even when temporarily missing.
        return FoundBackend {
            spec,
            executable: if binary_at(&candidate) {
                Some(candidate)
            } else {
                None
            },
            found_via: Some("local settings file"),
        };
    }
    let mut seen = HashSet::new();
    for (dir, origin) in dirs {
        if !seen.insert(dir.clone()) {
            continue;
        }
        for executable in spec.executable_names {
            let path = dir.join(executable);
            if binary_at(&path) {
                // Ensure SYCL backend only selects a genuine SYCL-enabled build directory
                if spec.id == BackendId::IntelSyclCli
                    && !dir.join("ggml-sycl.dll").is_file()
                    && !dir.join("sycl-ls.exe").is_file()
                {
                    continue;
                }
                return FoundBackend {
                    spec,
                    executable: Some(path),
                    found_via: Some(origin),
                };
            }
        }
    }
    FoundBackend {
        spec,
        executable: None,
        found_via: None,
    }
}

/// Identifies known binaries in PATH/common folders, without executing any.
/// "Installed" here means a candidate file was found, not validated compatibility.
/// Discover using an already loaded file, so GUI and CLI share the same
/// preference source and command-line flags remain independent overrides.
pub fn discover_backends_with_settings(prefs: &LocalSettings) -> Vec<FoundBackend> {
    let dirs = search_dirs(Some(prefs));
    SPECS
        .iter()
        .copied()
        .map(|s| discover_one(s, &dirs, prefs))
        .collect()
}
pub fn discover_backends() -> Vec<FoundBackend> {
    match load_settings() {
        Ok(prefs) => discover_backends_with_settings(&prefs),
        Err(e) => {
            // Fail closed: corrupt/future settings never silently select a
            // different executable than the user's previously chosen backend.
            eprintln!("Ojon settings unavailable ({e}); backend discovery blocked until repaired");
            SPECS
                .iter()
                .copied()
                .map(|spec| FoundBackend {
                    spec,
                    executable: None,
                    found_via: Some("invalid local settings"),
                })
                .collect()
        }
    }
}

/// Capabilities are live, read-only *executable probes*, never inferred from
/// a filename, GPU vendor or release folder. No models are opened.
/// Only run this at user request or immediately before an approved job.
fn probe_help(executable: &Path, arg: &str) -> std::io::Result<String> {
    let mut command = Command::new(executable);
    command.arg(arg);
    let output = capture_with_timeout(&mut command, 256 * 1024, Duration::from_secs(12))?;
    // A runtime that rejected --list-devices cannot claim a verified list.
    // Quantizer --help is different: several compatible releases print usage
    // and exit nonzero, so accept it only when precise format rows are found.
    if arg == "--list-devices" && !output.status.success() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "selected runtime rejected --list-devices",
        ));
    }
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push('\n');
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    // llama-quantize --help sometimes returns a nonzero exit code; clear,
    // bounded human-readable output is useful but does not prove execution.
    if combined.trim().is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "backend capability probe returned no output",
        ));
    }
    Ok(combined)
}

/// Quantizer usage lines contain an index, "or" and the supported GGUF type.
/// Do not parse unrelated prose that happens to mention a known precision.
pub fn quantizer_formats_from_help(help: &str) -> Vec<String> {
    let mut formats = Vec::new();
    for line in help.lines() {
        let words = line.split_whitespace().collect::<Vec<_>>();
        let name = if words.len() >= 3
            && words[0].parse::<u16>().is_ok()
            && words[1].eq_ignore_ascii_case("or")
        {
            words[2]
        } else {
            continue;
        };
        if let Ok(precision) = name.parse::<Precision>() {
            let canonical = precision.as_str();
            if !formats.contains(&canonical.to_owned()) {
                formats.push(canonical.to_owned());
            }
        }
    }
    formats
}
/// Parse only explicit device identifiers at the start of runtime list lines.
/// A random GPU name or reference to a device elsewhere in stdout is not proof.
pub fn runtime_devices_from_list(output: &str) -> Vec<String> {
    let mut devices = Vec::new();
    for line in output.lines() {
        let line = line.trim();
        let Some((key, description)) = line.split_once(':') else {
            continue;
        };
        let key = key.trim();
        if key.len() > 80
            || key.len() < 3
            || description.trim().is_empty()
            || !key.starts_with(|c: char| c.is_ascii_alphabetic())
            || !key.bytes().any(|b| b.is_ascii_digit())
            || !key
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-'))
        {
            continue;
        }
        if !devices.iter().any(|id| id == key) {
            devices.push(key.to_string());
        }
    }
    devices
}
fn probe_binary(found: &FoundBackend) -> Value {
    let Some(path) = found.executable.as_deref() else {
        return json!({"name":found.spec.name,"status":"NOT_INSTALLED",
            "path":Value::Null,"capability_evidence":"none"});
    };
    let probe = match found.spec.id {
        BackendId::LlamaQuantize => Some("--help"),
        BackendId::LlamaCli | BackendId::IntelSyclCli => Some("--list-devices"),
        BackendId::LlamaBench | BackendId::LlamaServer => Some("--help"),
        _ => None,
    };
    let Some(flag) = probe else {
        return json!({"name":found.spec.name,"status":"NOT_PROBED_NO_STANDARD_ADAPTER",
            "path":path.to_string_lossy(),"capability_evidence":"none"});
    };
    match probe_help(path, flag) {
        Ok(output) => {
            let formats = if found.spec.id == BackendId::LlamaQuantize {
                quantizer_formats_from_help(&output)
            } else {
                Vec::new()
            };
            let devices = if matches!(found.spec.id, BackendId::LlamaCli | BackendId::IntelSyclCli)
            {
                runtime_devices_from_list(&output)
            } else {
                Vec::new()
            };
            json!({"name":found.spec.name,"path":path.to_string_lossy(),
                "status":if matches!(found.spec.id,BackendId::LlamaBench|BackendId::LlamaServer){
                    "HELP_OUTPUT_ONLY"
                }else if (found.spec.id==BackendId::LlamaQuantize&&formats.is_empty())
                     ||(matches!(found.spec.id,BackendId::LlamaCli|BackendId::IntelSyclCli)
                        &&devices.is_empty()){"INCONCLUSIVE"}else{"CAPABILITY_OUTPUT_PARSED"},
                "probe_argument":flag,"precision_formats":formats,
                "device_ids":devices,
                "backend_executable_ran":true,
                "model_family_compatibility_verified":false,
                "runtime_model_load_verified":false})
        }
        Err(error) => json!({"name":found.spec.name,
            "path":path.to_string_lossy(),"status":"PROBE_FAILED",
            "reason":error.to_string(),"capability_evidence":"none"}),
    }
}
/// Caller explicitly requests capability checks; this can execute selected
/// local binaries with bounded time/output but does not load or quantize models.
pub fn probe_backend_capabilities(found: &[FoundBackend]) -> Value {
    json!({"schema":"ojon.backend_capabilities.v1",
        "read_only_no_model_weights_loaded":true,
        "probes":found.iter().map(probe_binary).collect::<Vec<_>>()})
}
pub fn probe_quantizer_capabilities(executable: &Path) -> std::io::Result<Value> {
    let help = probe_help(executable, "--help")?;
    let formats = quantizer_formats_from_help(&help);
    if formats.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "installed quantizer did not advertise supported GGUF formats; use manual workflow",
        ));
    }
    Ok(json!({"status":"PARSED_FROM_INSTALLED_QUANTIZER",
        "executable":executable.to_string_lossy(),
        "precision_formats":formats,
        "supports_imatrix":help.contains("--imatrix"),
        "supports_tensor_type":help.contains("--tensor-type"),
        "model_compatibility_verified":false}))
}
pub fn probe_quantizer_formats(executable: &Path) -> std::io::Result<Vec<String>> {
    let result = probe_quantizer_capabilities(executable)?;
    Ok(result["precision_formats"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|x| x.as_str().map(str::to_owned))
        .collect())
}
pub fn probe_runtime_devices(executable: &Path) -> std::io::Result<Vec<String>> {
    let devices = runtime_devices_from_list(&probe_help(executable, "--list-devices")?);
    if devices.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "installed llama runtime returned no parseable device list",
        ));
    }
    Ok(devices)
}

/// The optional Python interpreter for llama.cpp's official HF converter.
/// This is not executed during discovery; installed Python packages are not
/// checked until the user explicitly starts a conversion.
pub fn hf_python_path() -> PathBuf {
    env::var_os("OJON_PYTHON")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("python"))
}

/// Executable selection for actual CLI execution. An explicit environment or
/// saved path must not silently fall back to a different PATH executable.
pub fn resolve_backend_command(id: BackendId, fallback: &str) -> std::io::Result<PathBuf> {
    let spec = SPECS
        .iter()
        .find(|spec| spec.id == id)
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "unknown backend"))?;
    if let Some(path) = env::var_os(spec.override_env) {
        return Ok(PathBuf::from(path));
    }
    let prefs = load_settings()?;
    if let Some(path) = prefs.backend(spec.override_env) {
        return Ok(PathBuf::from(path));
    }
    Ok(discovered_path(id).unwrap_or_else(|| PathBuf::from(fallback)))
}
pub fn discovered_path(id: BackendId) -> Option<PathBuf> {
    discover_backends()
        .into_iter()
        .find(|b| b.spec.id == id)?
        .executable
}

/// Resolves a bundled read-only fixture directory next to the running executable
/// or in the project root (`data/<name>`).
pub fn bundled_data_dir(name: &str) -> Option<PathBuf> {
    if name.is_empty()
        || name.len() > 128
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
    {
        return None;
    }
    let mut candidates = Vec::new();
    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("data").join(name));
            if let Some(p2) = parent.parent() {
                candidates.push(p2.join("data").join(name));
                if let Some(p3) = p2.parent() {
                    candidates.push(p3.join("data").join(name));
                }
            }
        }
    }
    if let Ok(manifest) = env::var("CARGO_MANIFEST_DIR") {
        candidates.push(PathBuf::from(manifest).join("data").join(name));
    }
    if let Ok(cwd) = env::current_dir() {
        candidates.push(cwd.join("data").join(name));
    }
    candidates.into_iter().find(|p| p.is_dir())
}

/// Resolves a bundled read-only fixture next to the running executable
/// (`data/<name>`), in the project workspace, or None when it is absent. Fixtures let one-click flows
/// run without user-supplied files; Ojon never downloads or generates them
/// implicitly, and user-provided paths always win.
pub fn bundled_data_file(name: &str) -> Option<PathBuf> {
    if name.is_empty()
        || name.len() > 128
        || !name
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'_' | b'-' | b'.'))
    {
        return None;
    }
    let mut candidates = Vec::new();
    if let Ok(exe) = env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join("data").join(name));
            if let Some(p2) = parent.parent() {
                candidates.push(p2.join("data").join(name));
                if let Some(p3) = p2.parent() {
                    candidates.push(p3.join("data").join(name));
                }
            }
        }
    }
    if let Ok(manifest) = env::var("CARGO_MANIFEST_DIR") {
        candidates.push(PathBuf::from(manifest).join("data").join(name));
    }
    if let Ok(cwd) = env::current_dir() {
        candidates.push(cwd.join("data").join(name));
    }
    candidates.into_iter().find(|p| p.is_file())
}

/// Open the publisher's trusted website, only following an explicit user action.
/// Never forms a URL from a model name, file path or untrusted model metadata.
pub fn open_official_download_page(id: BackendId) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    use std::process::{Command, Stdio};
    let url = SPECS
        .iter()
        .find(|spec| spec.id == id)
        .map(|s| s.website)
        .ok_or_else(|| Error::new(ErrorKind::NotFound, "unknown backend"))?;
    let mut cmd = if cfg!(windows) {
        let mut p = Command::new("explorer.exe");
        p.arg(url);
        p
    } else if cfg!(target_os = "macos") {
        let mut p = Command::new("open");
        p.arg(url);
        p
    } else {
        let mut p = Command::new("xdg-open");
        p.arg(url);
        p
    };
    let status = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    // Browser helpers may launch in a separate process; successful spawn is not a download.
    drop(status);
    Ok(())
}

#[cfg(test)]
mod capability_tests {
    use super::*;
    #[test]
    fn quantizer_only_accepts_advertised_format_table() {
        let output="usage: llama-quantize\n  11 or Q3_K_M : mixed\n  13 or Q3_K_L : mixed\n  30 or IQ4_XS : mixed\n Q3_K_S mentioned in a random message";
        assert_eq!(
            quantizer_formats_from_help(output),
            vec!["Q3_K_M", "Q3_K_L", "IQ4_XS"]
        );
    }
    #[test]
    fn device_list_does_not_infer_vendor_from_prose() {
        let list="Available devices:\n CUDA0: NVIDIA GPU\n SYCL0: Intel Arc GPU\n Vulkan2: adapter\n CUDA0: same again\n Please try CUDA1 in a sample\n";
        assert_eq!(
            runtime_devices_from_list(list),
            vec!["CUDA0", "SYCL0", "Vulkan2"]
        );
    }
}
