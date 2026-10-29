//! Single-file MiniMax-H3 checkpoints: native ComfyUI NVFP4 export using an
//! independently installed convert-to-quant (ctq) CLI. The output is a packed
//! safetensors file, not an Ojon-private or automatically quality-certified format.

use crate::jobs::run_with_log;
use serde_json::{json, Value};
use std::fs::{self, File};
use std::io::{self, ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

const HEADER_LIMIT: u64 = 32 * 1024 * 1024;
const MAX_TENSORS: usize = 100_000;
// Keep timestep conditioning, heads, patch embeddings, token refiner, output
// projections and boundary blocks in their source precision. This is a
// conservative *recipe*, not a claim of zero-loss or measured sensitivity.
const H3_PROTECTED: &str =
    r"^(blocks\.(0|1|47|48|49)\.|token_refiner\.|.*(adaln|norm|patch_proj|final_layer|condition_proj|time_embedder|attn\.out_proj))";

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message.into())
}

#[derive(Clone, Debug)]
pub struct MiniMaxCheckpointPlan {
    pub source: PathBuf,
    pub output: PathBuf,
    pub converter: PathBuf,
    pub device: String,
    pub source_bytes: u64,
    pub source_tensors: usize,
}

impl MiniMaxCheckpointPlan {
    pub fn json(&self) -> Value {
        json!({
            "schema": "ojon.minimax_h3_checkpoint.v1",
            "status": "PLAN_ONLY",
            "source": self.source,
            "output": self.output,
            "source_bytes": self.source_bytes,
            "source_tensors": self.source_tensors,
            "converter": self.converter,
            "device": self.device,
            "output_format": "ComfyUI native NVFP4 safetensors",
            "weight_format": "NVFP4",
            "activation_precision": "runtime-dependent; NOT W4A4 certified",
            "protected_layer_regex": H3_PROTECTED,
            "quality_verified": false,
            "comfyui_runtime_verified": false,
            "intel_runtime_verified": false,
            "notes": "Four-bit checkpoint storage, not proof of four-bit accelerated inference or visual fidelity."
        })
    }
}

/// Reads a bounded safetensors catalog without loading tensor payloads.
fn catalog(path: &Path) -> io::Result<(u64, serde_json::Map<String, Value>)> {
    let mut file = File::open(path)?;
    let bytes = file.metadata()?.len();
    if bytes < 16 {
        return Err(invalid("safetensors checkpoint is too small"));
    }
    let mut size = [0u8; 8];
    file.read_exact(&mut size)?;
    let header_len = u64::from_le_bytes(size);
    if header_len == 0 || header_len > HEADER_LIMIT || header_len > bytes.saturating_sub(8) {
        return Err(invalid("invalid or oversized safetensors header"));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)?;
    let value: Value = serde_json::from_slice(&header)
        .map_err(|err| invalid(format!("invalid safetensors JSON: {err}")))?;
    let mut object = value.as_object().ok_or_else(|| invalid("safetensors header is not an object"))?.clone();
    object.remove("__metadata__");
    if object.is_empty() || object.len() > MAX_TENSORS {
        return Err(invalid("unexpected safetensors tensor count"));
    }
    // Validate every advertised byte range against the actual source payload.
    let payload = bytes - 8 - header_len;
    for (name, value) in &object {
        let offsets = value.get("data_offsets").and_then(Value::as_array)
            .ok_or_else(|| invalid(format!("tensor {name} lacks offsets")))?;
        if offsets.len() != 2 {
            return Err(invalid(format!("invalid offsets for tensor {name}")));
        }
        let start = offsets[0].as_u64().ok_or_else(|| invalid("invalid tensor start"))?;
        let end = offsets[1].as_u64().ok_or_else(|| invalid("invalid tensor end"))?;
        if start > end || end > payload {
            return Err(invalid(format!("tensor {name} exceeds file payload")));
        }
    }
    Ok((bytes, object))
}

fn is_h3_checkpoint(tensors: &serde_json::Map<String, Value>) -> bool {
    let h3_blocks = tensors.keys().filter(|name| {
        name.starts_with("blocks.")
            && (name.contains(".mlp.") || name.contains(".attn."))
            && name.ends_with(".weight")
    }).count();
    let has_token_refiner = tensors.keys().any(|name| name.starts_with("token_refiner."));
    let has_video = tensors.keys().any(|name| name.contains("video_patch_proj"));
    h3_blocks >= 100 && has_token_refiner && has_video
}

fn contains_quantized_storage(tensors: &serde_json::Map<String, Value>) -> bool {
    tensors.keys().any(|name| {
        name.ends_with(".comfy_quant")
            || name.ends_with(".weight_scale")
            || name.ends_with(".weight_scale_2")
    })
}

pub fn prepare_minimax_checkpoint(
    source: &Path,
    output: &Path,
    converter: &Path,
    device: &str,
) -> io::Result<MiniMaxCheckpointPlan> {
    if !matches!(device, "auto" | "cpu" | "cuda" | "cuda:0" | "cuda:1" | "xpu" | "xpu:0" | "xpu:1") {
        return Err(invalid("device must be auto, cpu, cuda[:0|1], or xpu[:0|1]"));
    }
    if converter.as_os_str().is_empty() || (converter.is_absolute() && !converter.is_file()) {
        return Err(invalid("ctq converter is missing"));
    }
    let source = source.canonicalize()?;
    if !source.is_file() || source.extension().and_then(|e| e.to_str()) != Some("safetensors") {
        return Err(invalid("input must be an existing single .safetensors checkpoint"));
    }
    let (bytes, tensors) = catalog(&source)?;
    if !is_h3_checkpoint(&tensors) {
        return Err(invalid("source does not match the single-file MiniMax-H3 transformer layout"));
    }
    if contains_quantized_storage(&tensors) {
        return Err(invalid("source is already quantized; always start from the original checkpoint"));
    }
    let parent = output.parent().filter(|x| !x.as_os_str().is_empty())
        .unwrap_or(Path::new(".")).canonicalize()?;
    let filename = output.file_name().ok_or_else(|| invalid("missing output filename"))?;
    if output.extension().and_then(|e| e.to_str()) != Some("safetensors") {
        return Err(invalid("output must end in .safetensors"));
    }
    let output = parent.join(filename);
    if output == source || output.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "output already exists or equals source"));
    }
    Ok(MiniMaxCheckpointPlan {
        source,
        output,
        converter: converter.to_owned(),
        device: device.to_owned(),
        source_bytes: bytes,
        source_tensors: tensors.len(),
    })
}

struct Staged(PathBuf);
impl Drop for Staged {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

pub fn run_minimax_checkpoint(
    plan: &MiniMaxCheckpointPlan,
    mut log: impl FnMut(&str),
) -> io::Result<Value> {
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "output already exists"));
    }
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)
        .map_err(io::Error::other)?.as_nanos();
    let parent = plan.output.parent().ok_or_else(|| invalid("output missing parent"))?;
    let stage = Staged(parent.join(format!(
        ".ojon-minimax-nvfp4-{}-{timestamp}.partial.safetensors",
        std::process::id()
    )));
    if stage.0.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "staging path exists"));
    }
    let mut command = Command::new(&plan.converter);
    command
        .arg("-i").arg(&plan.source)
        .arg("-o").arg(&stage.0)
        .arg("--nvfp4")
        .arg("--comfy_quant")
        .arg("--save-quant-metadata")
        .arg("--low-memory")
        .arg("--simple")
        .arg("--exclude-layers").arg(H3_PROTECTED)
        .env("HF_HUB_OFFLINE", "1")
        .env("TRANSFORMERS_OFFLINE", "1")
        .env("WANDB_MODE", "offline");
    if plan.device != "auto" {
        command.arg("--device").arg(&plan.device);
    }
    log("MiniMax-H3: quantizing original single-file checkpoint to ComfyUI NVFP4");
    run_with_log(&mut command, &mut log)?;
    if !stage.0.is_file() {
        return Err(io::Error::new(ErrorKind::InvalidData, "converter produced no checkpoint"));
    }
    let (output_bytes, packed) = catalog(&stage.0)?;
    let quantized = packed.keys().filter(|name| name.ends_with(".comfy_quant")).count();
    let scales = packed.keys().filter(|name| name.ends_with(".weight_scale")).count();
    if quantized < 20 || scales < 20 || output_bytes >= plan.source_bytes {
        return Err(io::Error::new(ErrorKind::InvalidData,
            "NVFP4 output is not demonstrably quantized and smaller"));
    }
    if !is_h3_checkpoint(&packed) {
        return Err(io::Error::new(ErrorKind::InvalidData,
            "quantization output lost the H3 transformer tensor structure"));
    }
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "output created concurrently"));
    }
    fs::rename(&stage.0, &plan.output)?;
    Ok(json!({
        "schema": "ojon.minimax_h3_checkpoint.v1",
        "status": "PACKED_NVFP4_VALIDATED",
        "output": plan.output,
        "original_bytes": plan.source_bytes,
        "compressed_bytes": output_bytes,
        "quantized_modules": quantized,
        "scale_tensors": scales,
        "format": "ComfyUI NVFP4 safetensors",
        "quality_verified": false,
        "comfyui_runtime_verified": false,
        "note": "Packed structure verified; actual ComfyUI video generation and visual parity remain untested."
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn make(path: &Path, valid_h3: bool, quantized: bool) {
        let mut tensors = serde_json::Map::new();
        if valid_h3 {
            for i in 0..50 {
                for suffix in ["mlp.fc1.weight", "mlp.fc2.weight", "attn.qkv_proj.weight"] {
                    let name = format!("blocks.{i}.{suffix}");
                    tensors.insert(name, json!({"dtype":"F16","shape":[1],"data_offsets":[0,2]}));
                }
            }
            tensors.insert("token_refiner.blocks.0.attn.weight".into(),
                json!({"dtype":"F16","shape":[1],"data_offsets":[0,2]}));
            tensors.insert("video_patch_proj.weight".into(),
                json!({"dtype":"F16","shape":[1],"data_offsets":[0,2]}));
        } else {
            tensors.insert("unrelated.weight".into(),
                json!({"dtype":"F16","shape":[1],"data_offsets":[0,2]}));
        }
        if quantized {
            tensors.insert("blocks.1.attn.comfy_quant".into(),
                json!({"dtype":"U8","shape":[1],"data_offsets":[0,1]}));
        }
        let header = serde_json::to_vec(&tensors).unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&(header.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&[0u8; 2]);
        fs::write(path, bytes).unwrap();
    }
    struct Scratch(PathBuf);
    impl Drop for Scratch { fn drop(&mut self) { let _ = fs::remove_dir_all(&self.0); } }
    fn fixture() -> Scratch {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let path = std::env::temp_dir().join(format!("ojon-minimax-checkpoint-{}-{nanos}", std::process::id()));
        fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
    #[test]
    fn plans_real_layout_without_execution() {
        let root = fixture();
        let source = root.0.join("10Eros_Max_h3_8stepTURBO_beta7.safetensors");
        make(&source, true, false);
        let plan = prepare_minimax_checkpoint(&source, &root.0.join("out.safetensors"),
            Path::new("ctq"), "cuda").unwrap();
        assert_eq!(plan.json()["status"], "PLAN_ONLY");
        assert_eq!(plan.json()["quality_verified"], false);
        assert!(!plan.output.exists());
    }
    #[test]
    fn rejects_non_h3_and_requantization() {
        let root = fixture();
        let source = root.0.join("bad.safetensors");
        make(&source, false, false);
        assert!(prepare_minimax_checkpoint(&source, &root.0.join("out.safetensors"),
            Path::new("ctq"), "cpu").is_err());
        make(&source, true, true);
        assert!(prepare_minimax_checkpoint(&source, &root.0.join("out.safetensors"),
            Path::new("ctq"), "cpu").is_err());
    }
    #[test]
    fn refuses_source_overwrite_and_missing_parent() {
        let root = fixture();
        let source = root.0.join("beta7.safetensors");
        make(&source, true, false);
        assert!(prepare_minimax_checkpoint(&source, &source, Path::new("ctq"), "auto").is_err());
        assert!(prepare_minimax_checkpoint(&source, &root.0.join("missing/out.safetensors"),
            Path::new("ctq"), "auto").is_err());
    }
}