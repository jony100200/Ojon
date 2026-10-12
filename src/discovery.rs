//! Lightweight, read-only discovery of independently installed AI backends.
//!
//! Detecting an executable is NOT proof of its model/format compatibility.
//! No binary is executed, installed, downloaded or recursively searched for.

use std::collections::HashSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

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
    LlamaPerplexity,
    HfConverter,
    LlamaFactory,
    StableDiffusionCpp,
    WhisperCpp,
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

const SPECS: [BackendSpec; 7] = [
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
        id: BackendId::LlamaPerplexity,
        name: "llama.cpp perplexity evaluator",
        executable_names: &["llama-perplexity", "llama-perplexity.exe"],
        override_env: "OJON_LLAMA_PERPLEXITY",
        website: "https://github.com/ggml-org/llama.cpp",
        integration: Integration::Wired,
        role: "Real held-out text perplexity comparison between original and candidate GGUF variants",
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
        integration: Integration::DiscoveryOnly,
        role: "Diffusion support requires a separate compatible adapter",
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
];

pub fn backend_specs() -> &'static [BackendSpec] { &SPECS }

#[derive(Clone, Debug)]
pub struct FoundBackend {
    pub spec: BackendSpec,
    pub executable: Option<PathBuf>,
    pub found_via: Option<&'static str>,
}
impl FoundBackend {
    pub fn installed(&self) -> bool { self.executable.is_some() }
}

fn binary_at(path: &Path) -> bool {
    path.is_file()
}

/// Collect common executable folders without scanning entire disks.
/// OJON_BACKENDS_DIR can point to a parent with one directory of packaged tools.
/// The user can always enter a full executable path when the scan misses a tool.
fn search_dirs() -> Vec<(PathBuf, &'static str)> {
    let mut dirs: Vec<(PathBuf, &'static str)> = Vec::new();
    if let Some(path) = env::var_os("PATH") {
        dirs.extend(env::split_paths(&path).filter(|p| !p.as_os_str().is_empty())
            .map(|p| (p, "PATH")));
    }
    if let Some(home) = env::var_os("HOME").or_else(|| env::var_os("USERPROFILE")) {
        let home = PathBuf::from(home);
        for sub in ["bin", ".local/bin", "scoop/shims", "scoop/apps/llama.cpp/current/bin",
                    "AppData/Local/Microsoft/WinGet/Links"] {
            dirs.push((home.join(sub), "known user folder"));
        }
    }
    if let Some(program_files) = env::var_os("ProgramFiles") {
        dirs.push((PathBuf::from(program_files).join("llama.cpp"), "known install folder"));
    }
    if let Some(local) = env::var_os("LOCALAPPDATA") {
        dirs.push((PathBuf::from(local).join("Programs").join("llama.cpp"), "known install folder"));
    }
    if let Some(root) = env::var_os("OJON_LLAMA_CPP_DIR") {
        let root = PathBuf::from(root);
        dirs.push((root.clone(), "OJON_LLAMA_CPP_DIR"));
        dirs.push((root.join("build").join("bin"), "OJON_LLAMA_CPP_DIR"));
    }
    if let Some(root) = env::var_os("OJON_BACKENDS_DIR") {
        let root = PathBuf::from(root);
        dirs.push((root.clone(), "OJON_BACKENDS_DIR"));
        // Bounded search under a user explicitly chosen converter directory.
        // Includes common layouts such as <root>/llama.cpp/build/bin.
        if let Ok(iter) = fs::read_dir(&root) {
            for folder in iter.flatten().take(64).map(|e| e.path()).filter(|p| p.is_dir()) {
                dirs.push((folder.clone(), "OJON_BACKENDS_DIR"));
                dirs.push((folder.join("bin"), "OJON_BACKENDS_DIR"));
                dirs.push((folder.join("build").join("bin"), "OJON_BACKENDS_DIR"));
            }
        }
    }
    dirs
}

fn discover_one(spec: BackendSpec, dirs: &[(PathBuf, &'static str)]) -> FoundBackend {
    // Explicit override is authoritative: do not secretly choose another installation.
    // An invalid override is reported as missing so the user can correct it.
    if let Some(override_path) = env::var_os(spec.override_env) {
        let candidate = PathBuf::from(override_path);
        return FoundBackend {
            spec,
            executable: if binary_at(&candidate) { Some(candidate) } else { None },
            found_via: Some("explicit environment override"),
        };
    }
    let mut seen = HashSet::new();
    for (dir, origin) in dirs {
        if !seen.insert(dir.clone()) { continue; }
        for executable in spec.executable_names {
            let path = dir.join(executable);
            if binary_at(&path) {
                return FoundBackend { spec, executable: Some(path), found_via: Some(origin) };
            }
        }
    }
    FoundBackend { spec, executable: None, found_via: None }
}

/// Identifies known binaries in PATH/common folders, without executing any.
/// "Installed" here means a candidate file was found, not validated compatibility.
pub fn discover_backends() -> Vec<FoundBackend> {
    let dirs = search_dirs();
    SPECS.iter().copied().map(|s| discover_one(s, &dirs)).collect()
}

/// The optional Python interpreter for llama.cpp's official HF converter.
/// This is not executed during discovery; installed Python packages are not
/// checked until the user explicitly starts a conversion.
pub fn hf_python_path() -> PathBuf {
    env::var_os("OJON_PYTHON").map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("python"))
}

pub fn discovered_path(id: BackendId) -> Option<PathBuf> {
    discover_backends().into_iter().find(|b| b.spec.id == id)?.executable
}

/// Open the publisher's trusted website, only following an explicit user action.
/// Never forms a URL from a model name, file path or untrusted model metadata.
pub fn open_official_download_page(id: BackendId) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    use std::process::{Command, Stdio};
    let url = SPECS.iter().find(|spec| spec.id == id)
        .map(|s| s.website).ok_or_else(|| Error::new(ErrorKind::NotFound, "unknown backend"))?;
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
    let status = cmd.stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn()?;
    // Browser helpers may launch in a separate process; successful spawn is not a download.
    drop(status);
    Ok(())
}
