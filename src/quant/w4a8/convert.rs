//! Model-level converter: stream a floating safetensors checkpoint into comfy-kitchen W4A8/W6A8
//! layout (loads in ComfyUI, Rupokar and Probaho). Selected 2-D linear weights become
//! weight (I8 packed) + weight_s_rel (F8_E4M3) + weight_s_channel (F32) [+ weight_codebook] +
//! comfy_quant (JSON); every other tensor and the file metadata are copied unchanged.

use super::matrix::{check_shape, quantize_matrix};
use super::quantize::{Bits, GROUP};
use super::st_writer::{header, Entry};
use crate::models::weight_patch::{read_safetensors_header, TensorLoc};
use serde_json::{json, Map, Value};
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct Request {
    pub source: PathBuf,
    pub output: PathBuf,
    pub bits: Bits,
    /// Dot-separated globs; `*` matches one name segment. A layer is quantized when
    /// `<pattern>.weight` matches the tensor name.
    pub layers: Vec<String>,
    pub threads: usize,
}

/// Default for MiniMax H3 / H3-Image, matching Comfy-Org's published W4A8 file.
pub fn h3_default_layers() -> Vec<String> {
    ["blocks.*.attn.qkv_proj", "blocks.*.attn.out_proj", "blocks.*.mlp.fc1", "blocks.*.mlp.fc2"].map(String::from).to_vec()
}

fn glob(pattern: &str, name: &str) -> bool {
    let (p, n): (Vec<&str>, Vec<&str>) = (pattern.split('.').collect(), name.split('.').collect());
    p.len() == n.len() && p.iter().zip(&n).all(|(a, b)| *a == "*" || a == b)
}

fn format_name(bits: Bits) -> &'static str {
    match bits {
        Bits::Four => "asym_w4a8_int8",
        Bits::Six => "w6a8_int8",
    }
}

fn selected(req: &Request, name: &str, t: &TensorLoc) -> bool {
    t.shape.len() == 2
        && name.strip_suffix(".weight").is_some_and(|p| req.layers.iter().any(|g| glob(g, p)))
        && check_shape(t.shape[0], t.shape[1], req.bits).is_ok()
}

fn read_metadata(path: &Path) -> io::Result<Map<String, Value>> {
    let mut f = File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let mut h = vec![0u8; u64::from_le_bytes(len) as usize];
    f.read_exact(&mut h)?;
    let v: Value = serde_json::from_slice(&h).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    Ok(v.get("__metadata__").and_then(Value::as_object).cloned().unwrap_or_default())
}

fn quant_entries(prefix: &str, n: usize, k: usize, bits: Bits) -> Vec<Entry> {
    let conf = json!({"format": format_name(bits)}).to_string();
    let mut v = vec![
        Entry { name: format!("{prefix}.comfy_quant"), dtype: "U8", shape: vec![conf.len()], bytes: conf.len() as u64 },
        Entry { name: format!("{prefix}.weight"), dtype: "I8", shape: vec![n, k * bits.count() / 8], bytes: (n * k * bits.count() / 8) as u64 },
        Entry { name: format!("{prefix}.weight_s_channel"), dtype: "F32", shape: vec![n], bytes: (n * 4) as u64 },
        Entry { name: format!("{prefix}.weight_s_rel"), dtype: "F8_E4M3", shape: vec![n, k / GROUP], bytes: (n * k / GROUP) as u64 },
    ];
    if bits == Bits::Four {
        v.push(Entry { name: format!("{prefix}.weight_codebook"), dtype: "F32", shape: vec![16], bytes: 64 });
    }
    v
}

const DTYPES: [&str; 12] = ["BF16", "F16", "F32", "F64", "I8", "U8", "I16", "I32", "I64", "BOOL", "F8_E4M3", "F8_E5M2"];

fn static_dtype(d: &str) -> io::Result<&'static str> {
    DTYPES.iter().copied().find(|&s| s == d).ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, format!("unsupported dtype {d}")))
}

/// Plan, then stream. Returns a JSON receipt. Never overwrites `output`.
pub fn run(req: &Request, mut progress: impl FnMut(&str)) -> io::Result<Value> {
    if req.output.exists() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{} exists", req.output.display())));
    }
    let tensors = read_safetensors_header(&req.source)?;
    let metadata = read_metadata(&req.source)?;
    let mut entries = Vec::new();
    let mut plan = Vec::new(); // (source name, quantize?)
    for (name, t) in &tensors {
        if selected(req, name, t) {
            entries.extend(quant_entries(name.trim_end_matches(".weight"), t.shape[0], t.shape[1], req.bits));
            plan.push((name.clone(), true));
        } else {
            entries.push(Entry { name: name.clone(), dtype: static_dtype(&t.dtype)?, shape: t.shape.clone(), bytes: t.bytes });
            plan.push((name.clone(), false));
        }
    }
    let quantized = plan.iter().filter(|p| p.1).count();
    if quantized == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "no tensor matched the layer patterns"));
    }
    let part = req.output.with_extension("safetensors.part");
    let started = Instant::now();
    let mut src = File::open(&req.source)?;
    let mut out = BufWriter::with_capacity(16 << 20, File::create(&part)?);
    out.write_all(&header(&entries, &metadata))?;
    let mut fitted = Vec::new();
    for (i, (name, quant)) in plan.iter().enumerate() {
        let t = &tensors[name];
        let mut buf = vec![0u8; t.bytes as usize];
        src.seek(SeekFrom::Start(t.offset))?;
        src.read_exact(&mut buf)?;
        if !quant {
            out.write_all(&buf)?;
            continue;
        }
        let (n, k) = (t.shape[0], t.shape[1]);
        let p = quantize_matrix(&buf, &t.dtype, n, k, req.bits, req.threads).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("{name}: {e}")))?;
        out.write_all(json!({"format": format_name(req.bits)}).to_string().as_bytes())?;
        out.write_all(&p.qdata)?;
        out.write_all(&p.s_channel.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
        out.write_all(&p.s_rel)?;
        if let Some(cb) = &p.codebook {
            out.write_all(&cb.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())?;
        }
        if p.codebook_fitted {
            fitted.push(name.clone());
        }
        progress(&format!("[{}/{}] {name} {n}x{k} -> {}", i + 1, plan.len(), format_name(req.bits)));
    }
    out.flush()?;
    drop(out);
    fs::rename(&part, &req.output)?;
    Ok(json!({
        "schema": "ojon.w4a8_convert.v1",
        "source": req.source, "output": req.output, "format": format_name(req.bits),
        "tensors_quantized": quantized, "tensors_copied": plan.len() - quantized,
        "source_bytes": fs::metadata(&req.source)?.len(), "output_bytes": fs::metadata(&req.output)?.len(),
        "fitted_codebooks": fitted, "seconds": started.elapsed().as_secs_f64().round(),
        "layout": "comfy-kitchen grouped INT8 (ConvRot 256, group 16, fp8 e4m3 s_rel, f32 s_channel)",
        "runtime_verified": false
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_globs_match_one_segment_per_star() {
        assert!(glob("blocks.*.mlp.fc1", "blocks.12.mlp.fc1"));
        assert!(!glob("blocks.*.mlp.fc1", "token_refiner.blocks.0.mlp.fc1"));
        assert!(!glob("blocks.*.mlp.fc1", "blocks.1.mlp.fc2"));
    }
}
