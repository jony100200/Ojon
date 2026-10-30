//! Non-invasive model identification, deliberately distinct from format support.
//!
//! Reads GGUF headers, HF config.json or bounded safetensors headers. Never
//! opens tensor payloads, runs Python, downloads files or executes a converter.
//! "Likely" classifications are hints, NOT permission to claim compatibility.

use crate::gguf::inspect_gguf;
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, s.into())
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelCategory {
    Language,
    VisionLanguage,
    Diffusion,
    Speech,
    EmbeddingOrReranker,
    Unknown,
}
impl ModelCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Language => "llm",
            Self::VisionLanguage => "vlm",
            Self::Diffusion => "image_video_diffusion",
            Self::Speech => "audio_speech",
            Self::EmbeddingOrReranker => "embedding_reranker",
            Self::Unknown => "unknown",
        }
    }
}
#[derive(Debug)]
pub struct ModelProbe {
    pub path: PathBuf,
    pub format: &'static str,
    pub category: ModelCategory,
    pub identification: String,
    pub confidence: &'static str,
    pub bytes: u64,
    pub tensor_count: Option<u64>,
    pub conversion_route: &'static str,
    pub conversion_status: &'static str,
    pub notes: Vec<String>,
}
impl ModelProbe {
    pub fn as_json(&self) -> Value {
        json!({
            "schema": "ojon.probe.v1",
            "source": self.path.to_string_lossy(),
            "format": self.format,
            "category": self.category.as_str(),
            "model_identifier": self.identification,
            "category_confidence": self.confidence,
            "source_bytes": self.bytes,
            "tensor_count": self.tensor_count,
            "candidate_conversion_route": self.conversion_route,
            "conversion_status": self.conversion_status,
            "notes": self.notes,
            "weights_loaded": false,
            "compatibility_verified": false
        })
    }
}
/// whisper.cpp / legacy ggml container magic: the u32 `0x67676d6c` ("ggml") read
/// little-endian from offset 0. GGUF uses "GGUF" instead and is handled separately.
const GGML_FILE_MAGIC: u32 = 0x6767_6d6c;

/// Split a descriptor into lowercase alphanumeric tokens so that short names
/// such as `dit`, `e5` or `clip` only match as whole words, never as substrings
/// of `ForConditionalGeneration`, `gemma3` or `eclipse`.
fn tokens(lower: &str) -> impl Iterator<Item = &str> {
    lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|t| !t.is_empty())
}
fn has_token(lower: &str, words: &[&str]) -> bool {
    tokens(lower).any(|t| words.contains(&t))
}
/// GGUF `general.architecture = "clip"` is a multimodal projector (mmproj), not a
/// standalone model. Token match, so `siglip` or `eclipse` do not qualify.
fn is_vision_projector(architecture: &str) -> bool {
    has_token(&architecture.to_ascii_lowercase(), &["clip"])
}
fn projector_note(architecture: &str) -> String {
    format!(
        "architecture '{architecture}' is a vision projector (mmproj) for a language GGUF, not a standalone model; quantizing it is not wired here"
    )
}
fn category_from(text: &str, has_vision_config: bool) -> (ModelCategory, &'static str) {
    let s = text.to_ascii_lowercase();
    // These are only hints. Never authorize a conversion based on substring matching.
    // Order matters: Speech and Embedding run before Diffusion so encoder-decoder
    // names (`*ForConditionalGeneration`) are never mistaken for a DiT.
    if has_vision_config
        || [
            "qwen2_vl",
            "qwen3_vl",
            "qwen2vl",
            "qwen3vl",
            "llava",
            "internvl",
            "pixtral",
            "idefics",
            "molmo",
            "minicpmv",
            "vision2seq",
            "paligemma",
            "gui-owl",
            "gui_owl",
            "florence",
            "smolvlm",
        ]
        .iter()
        .any(|x| s.contains(x))
        || has_token(&s, &["clip"])
    {
        return (ModelCategory::VisionLanguage, "heuristic");
    }
    if [
        "whisper",
        "encodec",
        "wav2vec",
        "speecht5",
        "parler_tts",
        "vits",
        "bark",
        "f5_tts",
    ]
    .iter()
    .any(|x| s.contains(x))
    {
        return (ModelCategory::Speech, "heuristic");
    }
    if [
        "reranker",
        "embedding",
        "sentence_transformer",
        "nomic-embed",
        "nomic_embed",
        "bert", // BertModel, XLMRobertaModel, RobertaModel, ModernBert, DistilBert
        "bge_",
        "e5_",
        "gte_",
    ]
    .iter()
    .any(|x| s.contains(x))
        || has_token(&s, &["bge", "e5", "gte"])
    {
        return (ModelCategory::EmbeddingOrReranker, "heuristic");
    }
    if [
        "flux",
        "wan",
        "ltx",
        "diffusion",
        "unet",
        "stable_diffusion",
        "hunyuanvideo",
        "hunyuan_video",
        "z_image",
        "qwen_image",
    ]
    .iter()
    .any(|x| s.contains(x))
        || has_token(&s, &["dit"])
    {
        return (ModelCategory::Diffusion, "heuristic");
    }
    if !s.is_empty() && s != "unknown" {
        return (ModelCategory::Language, "weak_guess");
    }
    (ModelCategory::Unknown, "unknown")
}
/// Classification for a named source (HF folder, GGUF model name / file stem).
/// The name is consulted only when the metadata says plain Language or Unknown,
/// and only for the words `embedding` / `reranker`: Qwen3-Embedding ships an
/// ordinary `Qwen3ForCausalLM` config and the folder name is the sole hint.
/// No other name word can override the architecture.
fn category_for_source(
    descriptor: &str,
    name: &str,
    has_vision_config: bool,
) -> (ModelCategory, &'static str) {
    let (category, confidence) = category_from(descriptor, has_vision_config);
    if matches!(category, ModelCategory::Language | ModelCategory::Unknown) {
        let lower = name.to_ascii_lowercase();
        if lower.contains("embedding") || lower.contains("reranker") {
            return (ModelCategory::EmbeddingOrReranker, "heuristic");
        }
    }
    (category, confidence)
}
fn load_json(path: &Path, max: u64) -> io::Result<Value> {
    let size = fs::metadata(path)?.len();
    if size == 0 || size > max {
        return Err(invalid("metadata header exceeds inspection bound"));
    }
    let mut file = File::open(path)?;
    let mut buf = Vec::with_capacity(size as usize);
    file.read_to_end(&mut buf)?;
    serde_json::from_slice(&buf).map_err(|e| invalid(format!("invalid JSON metadata: {e}")))
}
fn inspect_safetensors(path: &Path) -> io::Result<ModelProbe> {
    const MAX_HEADER: u64 = 32 * 1024 * 1024;
    let file_size = fs::metadata(path)?.len();
    if file_size < 10 {
        return Err(invalid("short safetensors file"));
    }
    let mut file = File::open(path)?;
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)?;
    let n = u64::from_le_bytes(prefix);
    if n == 0 || n > MAX_HEADER || n > file_size.saturating_sub(8) {
        return Err(invalid("safetensors header length invalid or too large"));
    }
    let mut header = vec![0u8; n as usize];
    file.read_exact(&mut header)?;
    let fields: Value = serde_json::from_slice(&header)
        .map_err(|e| invalid(format!("bad safetensors header: {e}")))?;
    let map = fields
        .as_object()
        .ok_or_else(|| invalid("safetensors header must be an object"))?;
    let tensor_names: Vec<&str> = map
        .keys()
        .filter(|k| k.as_str() != "__metadata__")
        .take(32)
        .map(String::as_str)
        .collect();
    let count = map
        .len()
        .saturating_sub(usize::from(map.contains_key("__metadata__")));
    if count == 0 {
        return Err(invalid("no tensor descriptors found"));
    }
    // Bound and validate each tensor payload range; never load the payload itself.
    let payload_size = file_size - 8 - n;
    for (name, desc) in map
        .iter()
        .filter(|(name, _)| name.as_str() != "__metadata__")
    {
        let pair = desc
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid(format!("tensor {name} lacks data_offsets")))?;
        if pair.len() != 2 {
            return Err(invalid("malformed tensor offsets"));
        }
        let from = pair[0]
            .as_u64()
            .ok_or_else(|| invalid("non-integer tensor offset"))?;
        let to = pair[1]
            .as_u64()
            .ok_or_else(|| invalid("non-integer tensor offset"))?;
        if from > to || to > payload_size {
            return Err(invalid("tensor offset outside source file"));
        }
    }
    let joined = tensor_names.join(" ").to_ascii_lowercase();
    let (category, confidence) = if joined.contains("diffusion_model.")
        || joined.contains("transformer_blocks.")
        || joined.contains("double_blocks.")
        || joined.contains("model.diffusion_model.")
    {
        (ModelCategory::Diffusion, "heuristic")
    } else if joined.contains("vision_tower.") || joined.contains("vision_model.") {
        (ModelCategory::VisionLanguage, "heuristic")
    } else {
        (ModelCategory::Unknown, "unknown")
    };
    Ok(ModelProbe {
        path: path.to_path_buf(), format:"safetensors_single_file", category,
        identification: if tensor_names.is_empty() { "unknown".into() } else {
            format!("{} tensors; first keys: {}", count, tensor_names.iter().take(3).copied().collect::<Vec<_>>().join(", "))
        }, confidence, bytes:file_size, tensor_count:Some(count as u64),
        conversion_route:if category==ModelCategory::Diffusion{"stable_diffusion_cpp_gguf_candidate"}else{"none"},
        conversion_status:if category==ModelCategory::Diffusion{"sd_cli_conversion_requires_model_specific_runtime_test"}else{"not_implemented_for_this_standalone_safetensors"},
        notes:vec![
            "Standalone diffusion safetensors may be eligible for stable-diffusion.cpp GGUF conversion; installed backend support and real image/video generation must be checked.".into(),
            "Classification from tensor names is advisory; preserve architecture-specific metadata and sensitive tensors.".into(),
        ],
    })
}

/// Diffusers pipelines use model_index.json instead of HF language config.json.
/// Probe only shallow component folders and local weight-file metadata.
fn inspect_diffusers_pipeline(directory: &Path) -> io::Result<ModelProbe> {
    let index = load_json(&directory.join("model_index.json"), 2 * 1024 * 1024)?;
    let map = index
        .as_object()
        .ok_or_else(|| invalid("Diffusers model_index.json must be an object"))?;
    let class = index
        .get("_class_name")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let mut total = 0u64;
    let mut files = 0usize;
    let mut components = Vec::new();
    for (name, value) in map.iter().filter(|(key, _)| !key.starts_with('_')).take(32) {
        if !value.is_array() {
            continue;
        }
        components.push(name.clone());
        let path = directory.join(name);
        if !path.is_dir() {
            continue;
        }
        for entry in fs::read_dir(path)?.take(4096) {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let file = entry.file_name();
            let file = file.to_string_lossy();
            if file.ends_with(".safetensors") || file.ends_with(".bin") {
                total = total
                    .checked_add(entry.metadata()?.len())
                    .ok_or_else(|| invalid("weight sizes overflow"))?;
                files += 1;
            }
        }
    }
    if files == 0 {
        return Err(invalid(
            "Diffusers pipeline has no local component weights in expected shallow directories",
        ));
    }
    let is_video = ["video", "wan", "ltx", "hunyuanvideo", "mochi", "cogvideo"]
        .iter()
        .any(|word| class.to_ascii_lowercase().contains(word));
    let supported = directory.join("transformer").is_dir() && map.contains_key("transformer");
    Ok(ModelProbe {
        path:directory.to_path_buf(),
        format:"diffusers_pipeline_directory",
        category:ModelCategory::Diffusion,
        identification:format!("{class} ({} components)",components.len()),
        confidence:"metadata_pipeline_class",
        bytes:total,tensor_count:None,
        conversion_route:if supported{"public_diffuse_compressor_optional_svdq_int4"}else{"none"},
        conversion_status:if supported{"experimental_requires_external_cuda_checkout_and_loader_test"}else{"no_diffusion_adapter_for_this_component_layout"},
        notes:vec![
            format!("Diffusers class: {class}; inferred category: {}",
                if is_video{"video"}else{"image_or_unspecified"}),
            format!("{} local component weight files found; total size excludes cache/intermediate files",files),
            format!("model components: {}",components.join(", ")),
            "SVDQ quantizes eligible transformer weights, not every text encoder, VAE or full pipeline component.".into(),
            "Compatible target kernel, GPU memory and visual fidelity have not been checked.".into(),
        ],
    })
}

/// Read-only inspection of Ojon owned weight-only .ojonq files.
fn inspect_ojonq(path: &Path) -> io::Result<ModelProbe> {
    let mut f = File::open(path)?;
    let size = f.metadata()?.len();
    if size < 18 {
        return Err(invalid("short Ojon packed file"));
    }
    let mut prefix = [0u8; 16];
    f.read_exact(&mut prefix)?;
    if &prefix[..8] != b"OJONQ001" {
        return Err(invalid("invalid Ojon packed container magic"));
    }
    let n = u64::from_le_bytes(prefix[8..16].try_into().unwrap());
    if n == 0 || n > 32 * 1024 * 1024 || n > size - 16 {
        return Err(invalid("invalid Ojon packed header"));
    }
    let mut header = vec![0u8; n as usize];
    f.read_exact(&mut header)?;
    let value: Value =
        serde_json::from_slice(&header).map_err(|e| invalid(format!("bad packed header: {e}")))?;
    if value["format"].as_str() != Some("ojon-packed-v1") {
        return Err(invalid("unsupported packed format version"));
    }
    let tensors = value["tensors"]
        .as_array()
        .ok_or_else(|| invalid("packed tensor manifest missing"))?;
    if tensors.is_empty() || tensors.len() > 100000 {
        return Err(invalid("packed tensor count invalid"));
    }
    let bits = value["bits"]
        .as_u64()
        .ok_or_else(|| invalid("packed bit count missing"))?;
    if !matches!(bits, 2..=4) {
        return Err(invalid("unsupported packed bit count"));
    }
    let mut count = 0u64;
    let mut names = String::new();
    for item in tensors {
        let name = item["name"]
            .as_str()
            .ok_or_else(|| invalid("invalid packed tensor name"))?;
        if names.len() < 256 {
            names.push_str(name);
            names.push(' ');
        }
        if item["quantized"].as_bool() == Some(true) {
            count += 1;
        }
    }
    let (category, confidence) = category_from(&names, false);
    Ok(ModelProbe{
        path:path.to_path_buf(),format:"ojonq_packed_weights",category,
        identification:format!("Ojon packed weights v1; {bits} bits; {count} packed tensors"),
        confidence,bytes:size,tensor_count:Some(tensors.len() as u64),
        conversion_route:"native_ojon_unpack",
        conversion_status:"weight_storage_only_gpu_loader_not_implemented",
        notes:vec![
            "Ojon packed weights store 2/3/4-bit scalar matrices with group scales and biases.".into(),
            "Native unpack restores ordinary safetensors, but restored weights still contain quantization error.".into(),
            "No directly compatible ComfyUI or Diffusers GPU loading kernel has been implemented yet.".into(),
        ],
    })
}

/// whisper.cpp GGML `.bin`: accepted by magic only. Reads 8 bytes (magic + the
/// first hparam, whisper's `n_vocab`); never the tensors. The wired route is
/// whisper.cpp's own `whisper-quantize` (`whisper-quant-plan|run`).
fn inspect_ggml_bin(path: &Path) -> io::Result<ModelProbe> {
    let mut f = File::open(path)?;
    let size = f.metadata()?.len();
    let mut head = [0u8; 8];
    if size < head.len() as u64 {
        return Err(invalid("short .bin file; not a GGML container"));
    }
    f.read_exact(&mut head)?;
    let magic = u32::from_le_bytes(head[..4].try_into().unwrap());
    if magic != GGML_FILE_MAGIC {
        return Err(invalid(
            "unsupported .bin; only whisper.cpp GGML (magic 'ggml') is recognised, PyTorch pickles are not",
        ));
    }
    // whisper vocabularies: 51864 (English-only), 51865 (multilingual), 51866 (large-v3).
    let n_vocab = i32::from_le_bytes(head[4..8].try_into().unwrap());
    let whisper_vocab = matches!(n_vocab, 51_864..=51_866);
    let (identification, confidence) = if whisper_vocab {
        (format!("whisper ggml (n_vocab={n_vocab})"), "heuristic")
    } else {
        ("ggml container (whisper-style assumed)".to_owned(), "weak_guess")
    };
    Ok(ModelProbe {
        path: path.to_path_buf(),
        format: "ggml_bin",
        category: ModelCategory::Speech,
        identification,
        confidence,
        bytes: size,
        tensor_count: None,
        conversion_route: "public_whisper_cpp_whisper_quantize",
        conversion_status: "candidate_only_requires_installed_backend_compatibility",
        notes: vec![
            "GGML magic matched; hparams beyond the first field and tensor payloads were not read.".into(),
            "The 'ggml' magic is shared with legacy pre-GGUF llama containers; whisper is inferred from the vocabulary size.".into(),
            "Quantize with whisper-quant-plan|run (whisper.cpp whisper-quantize); gate the result with whisper-parity-plan|run (WER on held-out clips).".into(),
        ],
    })
}

pub fn probe_model(input: &Path) -> io::Result<ModelProbe> {
    let path = input.canonicalize()?;
    let metadata = fs::metadata(&path)?;
    if metadata.is_dir() {
        if path.join("model_index.json").is_file() {
            return inspect_diffusers_pipeline(&path);
        }
        let conf = path.join("config.json");
        let value = load_json(&conf, 2 * 1024 * 1024)?;
        let model_type = value
            .get("model_type")
            .and_then(Value::as_str)
            .unwrap_or("");
        let architectures = value
            .get("architectures")
            .and_then(Value::as_array)
            .map(|list| {
                list.iter()
                    .filter_map(Value::as_str)
                    .take(12)
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_default();
        let descriptor = format!("{model_type} {architectures}");
        let folder_name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let (category, confidence) = category_for_source(
            &descriptor,
            &folder_name,
            value.get("vision_config").is_some(),
        );
        let mut total = 0u64;
        let mut count = 0usize;
        for entry in fs::read_dir(&path)?.take(8192) {
            let entry = entry?;
            let n = entry.file_name();
            let n = n.to_string_lossy();
            if n.ends_with(".safetensors") || n.starts_with("pytorch_model") && n.ends_with(".bin")
            {
                let meta = entry.metadata()?;
                if meta.is_file() {
                    total = total
                        .checked_add(meta.len())
                        .ok_or_else(|| invalid("source size overflow"))?;
                    count += 1;
                }
            }
        }
        if count == 0 {
            return Err(invalid("no model weights found in checkpoint directory"));
        }
        let candidate = matches!(
            category,
            ModelCategory::Language | ModelCategory::VisionLanguage
        );
        Ok(ModelProbe {
            path, format:"hf_checkpoint_directory", category,
            identification: descriptor.trim().to_owned(), confidence,
            bytes:total, tensor_count:None,
            conversion_route: if candidate { "public_llama_cpp_hf_to_gguf_then_quantize" } else { "none" },
            conversion_status: if candidate { "candidate_only_must_check_llama_cpp_architecture_support" } else { "no_wired_converter" },
            notes:vec![
                format!("{count} local model weight file(s) detected; shards/index completeness not fully validated"),
                "Training requires a supported full checkpoint and compatible backend; quantized GGUF files are not assumed trainable.".into(),
            ],
        })
    } else {
        match path
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase()
            .as_str()
        {
            "gguf" => {
                let gguf = inspect_gguf(&path)?;
                let desc = gguf
                    .architecture
                    .clone()
                    .unwrap_or_else(|| "unknown".to_owned());
                let name_hint = format!(
                    "{} {}",
                    gguf.model_name.clone().unwrap_or_default(),
                    path.file_stem().map(|s| s.to_string_lossy()).unwrap_or_default()
                );
                let (category, confidence) = category_for_source(&desc, &name_hint, false);
                let mut notes = vec![
                    "GGUF header and tensor catalog inspected; tensor bytes and runtime compatibility not verified.".into(),
                    "Quantizing an already heavily quantized source may not be beneficial; prefer F16/BF16 input.".into(),
                ];
                if is_vision_projector(&desc) {
                    notes.insert(0, projector_note(&desc));
                }
                Ok(ModelProbe {
                    path,format:"gguf",category,identification:desc,confidence,
                    bytes:gguf.file_bytes,tensor_count:Some(gguf.tensor_count),
                    conversion_route:"public_llama_cpp_llama_quantize",
                    conversion_status:"candidate_only_requires_installed_backend_compatibility",
                    notes,
                })
            }
            "safetensors" => inspect_safetensors(&path),
            "ojonq" => inspect_ojonq(&path),
            "bin" => inspect_ggml_bin(&path),
            _ => Err(invalid(
                "unsupported model source; use GGUF, a safetensors file, a whisper.cpp GGML .bin or HF checkpoint directory",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn categorize_without_promising_compatibility() {
        assert_eq!(
            category_from("Qwen3VLForConditionalGeneration", true).0,
            ModelCategory::VisionLanguage
        );
        assert_eq!(
            category_from("WanVideoModel", false).0,
            ModelCategory::Diffusion
        );
        assert_eq!(category_from("unknown", false).0, ModelCategory::Unknown);
    }

    // Task 1.1: `dit` must be a whole token, and Speech / Embedding are checked
    // before Diffusion, so `ForConditionalGeneration` no longer reads as DiT.
    #[test]
    fn conditional_generation_is_not_diffusion() {
        assert_eq!(
            category_from("whisper WhisperForConditionalGeneration", false).0,
            ModelCategory::Speech
        );
        // T5 / BART are encoder-decoder text models: Language (weak guess), never Diffusion.
        let (t5, t5_conf) = category_from("t5 T5ForConditionalGeneration", false);
        assert_eq!(t5, ModelCategory::Language);
        assert_eq!(t5_conf, "weak_guess");
        assert_eq!(
            category_from("bart BartForConditionalGeneration", false).0,
            ModelCategory::Language
        );
        // A DiT architecture on its own is still diffusion, in any case.
        assert_eq!(category_from("DiT", false).0, ModelCategory::Diffusion);
        assert_eq!(category_from("dit", false).0, ModelCategory::Diffusion);
        assert_eq!(category_from("pixart_dit", false).0, ModelCategory::Diffusion);
        // `dit` buried inside another word is not a token.
        assert_eq!(category_from("editor", false).0, ModelCategory::Language);
    }

    // Task 1.2: a GGUF mmproj carries general.architecture = "clip".
    #[test]
    fn clip_projector_is_vision_language() {
        let (category, confidence) = category_from("clip", false);
        assert_eq!(category, ModelCategory::VisionLanguage);
        assert_eq!(confidence, "heuristic");
        assert!(is_vision_projector("clip"));
        assert!(is_vision_projector("CLIP"));
        assert!(!is_vision_projector("qwen3"));
        let note = projector_note("clip");
        assert!(note.contains("projector"));
    }

    // Task 1.3: encoder / embedding / reranker hints.
    #[test]
    fn embedding_and_reranker_hints() {
        assert_eq!(
            category_from("xlm-roberta XLMRobertaModel", false).0,
            ModelCategory::EmbeddingOrReranker
        );
        assert_eq!(
            category_from("bert BertModel", false).0,
            ModelCategory::EmbeddingOrReranker
        );
        assert_eq!(
            category_from("roberta RobertaModel", false).0,
            ModelCategory::EmbeddingOrReranker
        );
        for name in ["bge-m3", "e5-large-v2", "gte-large", "nomic-embed-text-v1.5", "BGE_RERANKER"] {
            assert_eq!(
                category_from(name, false).0,
                ModelCategory::EmbeddingOrReranker,
                "{name}"
            );
        }
        // `e5` / `gte` / `bge` are whole tokens only.
        assert_eq!(category_from("gemma3", false).0, ModelCategory::Language);
        assert_eq!(category_from("qwen2.5", false).0, ModelCategory::Language);
    }

    #[test]
    fn embedding_name_hint_wins_over_causal_lm_architecture_only_for_those_words() {
        // Qwen3-Embedding ships a plain Qwen3ForCausalLM config; the folder name is the hint.
        let (category, confidence) =
            category_for_source("qwen3 Qwen3ForCausalLM", "Qwen3-Embedding-0.6B", false);
        assert_eq!(category, ModelCategory::EmbeddingOrReranker);
        assert_eq!(confidence, "heuristic");
        assert_eq!(
            category_for_source("qwen3 Qwen3ForCausalLM", "Qwen3-Reranker-0.6B", false).0,
            ModelCategory::EmbeddingOrReranker
        );
        // Any other folder word does not override the architecture.
        assert_eq!(
            category_for_source("qwen3 Qwen3ForCausalLM", "Qwen3-0.6B-Instruct", false).0,
            ModelCategory::Language
        );
        assert_eq!(
            category_for_source("qwen3 Qwen3ForCausalLM", "flux-dit-folder", false).0,
            ModelCategory::Language
        );
        // The name hint never demotes a stronger classification.
        assert_eq!(
            category_for_source("llava LlavaForConditionalGeneration", "my-embedding", true).0,
            ModelCategory::VisionLanguage
        );
    }

    // Task 1.4: whisper.cpp GGML `.bin` is accepted by magic and classified as Speech.
    #[test]
    fn ggml_bin_probes_as_speech_without_converter() {
        let dir = std::env::temp_dir().join(format!(
            "ojon-probe-ggml-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        fs::create_dir_all(&dir).unwrap();
        let file = dir.join("ggml-base.en.bin");
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(&GGML_FILE_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&51_864i32.to_le_bytes()); // whisper n_vocab
        bytes.extend_from_slice(&[0u8; 8]);
        assert_eq!(bytes.len(), 16);
        fs::write(&file, &bytes).unwrap();

        let probe = probe_model(&file).unwrap();
        assert_eq!(probe.format, "ggml_bin");
        assert_eq!(probe.category, ModelCategory::Speech);
        assert_eq!(probe.conversion_route, "public_whisper_cpp_whisper_quantize");
        assert_eq!(probe.conversion_status, "candidate_only_requires_installed_backend_compatibility");
        assert_eq!(probe.bytes, 16);
        assert!(probe.identification.contains("whisper"));
        assert_eq!(probe.as_json()["category"], "audio_speech");

        // A `.bin` without the GGML magic is still rejected at the door.
        let other = dir.join("pytorch_model.bin");
        fs::write(&other, b"PK................").unwrap();
        assert!(probe_model(&other).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
