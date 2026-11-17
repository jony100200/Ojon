//! `3d` model family: image-to-3D flow DiTs (TRELLIS.2 / Pixal3D style safetensors) -> GGUF with
//! Q4_K_M or Q5_K linear weights. Norms, biases, embeddings, modulation and the small input/output
//! layers keep their source dtype. Tensor names and shapes are unchanged (ggml shape = reversed
//! safetensors shape), so a ggml-based loader binds the file exactly like the source checkpoint.
//! The source is streamed tensor by tensor: peak RAM is a few of the largest tensors, not the file.

use crate::quant::kquant::{quantize_row, KQuant, QK_K};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const ALIGN: u64 = 32;
const GGML_F32: u32 = 0;
const GGML_F16: u32 = 1;
const GGML_BF16: u32 = 30;
const GGML_I8: u32 = 24;

fn bad(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum ThreeDQuant {
    Q4KM,
    Q5K,
    Q6K,
}

impl ThreeDQuant {
    pub fn from_name(name: &str) -> io::Result<Self> {
        match name {
            "Q4_K_M" => Ok(Self::Q4KM),
            "Q5_K" => Ok(Self::Q5K),
            "Q6_K" => Ok(Self::Q6K),
            _ => Err(bad("3d quant must be Q4_K_M, Q5_K or Q6_K (Q8 is not offered)")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Q4KM => "Q4_K_M",
            Self::Q5K => "Q5_K",
            Self::Q6K => "Q6_K",
        }
    }
    fn general_file_type(self) -> u32 {
        match self {
            Self::Q4KM => 15,
            Self::Q5K => 16,
            Self::Q6K => 18,
        }
    }
}

#[derive(Debug, Clone)]
struct SrcTensor {
    name: String,
    dtype: String,
    shape: Vec<u64>,
    begin: u64,
    end: u64,
    /// ComfyUI int8 source: per-row f32 scale offsets and the ConvRot group size (0 = none).
    scale: Option<(u64, u64)>,
    rotate: usize,
}

impl SrcTensor {
    fn elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

/// What happens to one tensor.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Fate {
    Quant(KQuant),
    Keep(u32),
}

fn dtype_info(dtype: &str) -> io::Result<(u32, u64)> {
    match dtype {
        "F32" => Ok((GGML_F32, 4)),
        "F16" => Ok((GGML_F16, 2)),
        "BF16" => Ok((GGML_BF16, 2)),
        "I8" => Ok((GGML_I8, 1)),
        other => Err(bad(format!("unsupported source dtype {other}"))),
    }
}

/// llama.cpp's "more bits" block schedule: first and last eighth, then every third block.
fn more_bits(index: usize, blocks: usize) -> bool {
    index < blocks / 8 || index >= blocks - blocks / 8 || (index - blocks / 8) % 3 == 2
}

/// Block-linear weights worth quantizing: attention and MLP matrices inside `blocks.N.`.
fn block_linear(name: &str) -> Option<(usize, &str)> {
    let rest = name.strip_prefix("blocks.")?;
    let (index, tail) = rest.split_once('.')?;
    let tail = tail.strip_suffix(".weight")?;
    let linear = matches!(
        tail,
        "self_attn.to_qkv" | "self_attn.to_out" | "cross_attn.to_q" | "cross_attn.to_kv" | "cross_attn.to_out" | "mlp.mlp.0" | "mlp.mlp.2"
            | "cross_attn.cross_attn_block.to_q" | "cross_attn.cross_attn_block.to_kv" | "cross_attn.cross_attn_block.to_out"
    );
    if !linear {
        return None;
    }
    Some((index.parse().ok()?, tail))
}

/// Mixed-precision overrides: blocks or tensor kinds left at the source dtype, or a block subset
/// to quantize (sensitivity scans quantize one block at a time).
#[derive(Debug, Clone, Default)]
pub struct Overrides {
    pub keep_blocks: Vec<usize>,
    pub keep_kinds: Vec<String>,
    pub only_blocks: Option<Vec<usize>>,
    pub tag: String,
    /// Read only tensors under this prefix (ComfyUI bundles several models in one file).
    pub prefix: String,
    /// Replaces the source file stem in the output name.
    pub name: Option<String>,
}

fn decide(t: &SrcTensor, plan: &ThreeDPlan, blocks: usize) -> io::Result<Fate> {
    let quant = plan.quant;
    let keep = Fate::Keep(dtype_info(&t.dtype)?.0);
    if t.dtype == "I8" && (t.shape.len() != 2 || t.scale.is_none() || block_linear(&t.name).is_none()) {
        return Err(bad(format!("int8 tensor {} is not a recognised block linear", t.name)));
    }
    let Some((index, tail)) = block_linear(&t.name) else { return Ok(keep) };
    let o = &plan.overrides;
    if t.dtype == "I8" && (o.keep_blocks.contains(&index) || o.keep_kinds.iter().any(|k| tail.ends_with(k.as_str()))) {
        return Err(bad("keep overrides cannot apply to int8 sources (no source precision to keep)"));
    }
    if o.keep_blocks.contains(&index) || o.keep_kinds.iter().any(|k| tail.ends_with(k.as_str())) || o.only_blocks.as_ref().is_some_and(|b| !b.contains(&index)) {
        return Ok(keep);
    }
    if t.shape.len() != 2 || t.shape[1] as usize % QK_K != 0 {
        return Ok(keep);
    }
    let sensitive = tail.ends_with("mlp.mlp.2") || tail.ends_with("to_kv");
    Ok(Fate::Quant(match quant {
        ThreeDQuant::Q4KM if sensitive && more_bits(index, blocks) => KQuant::Q6K,
        ThreeDQuant::Q4KM => KQuant::Q4K,
        ThreeDQuant::Q5K => KQuant::Q5K,
        ThreeDQuant::Q6K => KQuant::Q6K,
    }))
}

fn out_bytes(t: &SrcTensor, fate: Fate) -> io::Result<u64> {
    Ok(match fate {
        Fate::Quant(k) => t.shape[0] * k.row_bytes(t.shape[1] as usize) as u64,
        Fate::Keep(_) => t.elements() * dtype_info(&t.dtype)?.1,
    })
}

fn read_header(path: &Path, prefix: &str) -> io::Result<Vec<SrcTensor>> {
    let mut file = File::open(path)?;
    let mut len = [0u8; 8];
    file.read_exact(&mut len)?;
    let len = u64::from_le_bytes(len);
    if len == 0 || len > 64 << 20 {
        return Err(bad("implausible safetensors header length"));
    }
    let mut raw = vec![0u8; len as usize];
    file.read_exact(&mut raw)?;
    let header: BTreeMap<String, Value> = serde_json::from_slice(&raw).map_err(|e| bad(format!("safetensors header: {e}")))?;
    let base = 8 + len;
    let lead = if prefix.is_empty() { String::new() } else { format!("{prefix}.") };
    let span = |v: &Value| -> (u64, u64) {
        let off = v["data_offsets"].as_array();
        let get = |i: usize| off.and_then(|o| o.get(i)).and_then(Value::as_u64).unwrap_or(0);
        (base + get(0), base + get(1))
    };
    let (mut scales, mut quant_meta) = (BTreeMap::new(), BTreeMap::new());
    let mut tensors = Vec::new();
    for (full, v) in header.iter().filter(|(k, _)| *k != "__metadata__" && k.starts_with(&lead)) {
        let name = full[lead.len()..].to_owned();
        if let Some(stem) = name.strip_suffix(".weight_scale") {
            scales.insert(stem.to_owned(), span(v));
        } else if let Some(stem) = name.strip_suffix(".comfy_quant") {
            quant_meta.insert(stem.to_owned(), span(v));
        } else {
            let dtype = v["dtype"].as_str().ok_or_else(|| bad("tensor without dtype"))?.to_owned();
            let shape = v["shape"].as_array().ok_or_else(|| bad("tensor without shape"))?.iter().filter_map(Value::as_u64).collect();
            let (begin, end) = span(v);
            tensors.push(SrcTensor { name, dtype, shape, begin, end, scale: None, rotate: 0 });
        }
    }
    for t in tensors.iter_mut().filter(|t| t.dtype == "I8") {
        let stem = t.name.strip_suffix(".weight").unwrap_or(&t.name).to_owned();
        t.scale = scales.get(&stem).copied();
        if let Some(&(b, e)) = quant_meta.get(&stem) {
            let mut meta = vec![0u8; (e - b) as usize];
            file.seek(SeekFrom::Start(b))?;
            file.read_exact(&mut meta)?;
            let meta: Value = serde_json::from_slice(&meta).map_err(|e| bad(format!("comfy_quant: {e}")))?;
            if meta["format"] != "int8_tensorwise" {
                return Err(bad(format!("unsupported comfy_quant format {}", meta["format"])));
            }
            if meta["convrot"] == true {
                t.rotate = meta["convrot_groupsize"].as_u64().unwrap_or(256) as usize;
            }
        }
        if let Some((b, e)) = t.scale {
            if e - b != t.shape[0] * 4 {
                return Err(bad(format!("{} needs a per-row f32 scale", t.name)));
            }
        }
    }
    tensors.sort_by_key(|t| t.begin);
    let size = fs::metadata(path)?.len();
    for t in &tensors {
        let (_, width) = dtype_info(&t.dtype)?;
        if t.end > size || t.end < t.begin || t.end - t.begin != t.elements() * width {
            return Err(bad(format!("tensor {} has inconsistent offsets", t.name)));
        }
    }
    if tensors.is_empty() {
        return Err(bad("no tensors found (check --prefix)"));
    }
    Ok(tensors)
}

fn block_count(tensors: &[SrcTensor]) -> usize {
    tensors.iter().filter_map(|t| t.name.strip_prefix("blocks")?.strip_prefix('.')?.split('.').next()?.parse::<usize>().ok()).max().map_or(0, |m| m + 1)
}

/// `<stem without precision tag>-Ojon-<Quant>.gguf`.
pub fn output_name(source: &Path, quant: ThreeDQuant, tag: &str, name: Option<&str>) -> io::Result<String> {
    let stem = name.or_else(|| source.file_stem().and_then(|s| s.to_str())).ok_or_else(|| bad("source file name is not UTF-8"))?;
    let stem = ["_bf16", "_fp16", "_f16", "_f32"].iter().find_map(|tag| stem.strip_suffix(tag)).unwrap_or(stem);
    Ok(format!("{stem}-Ojon-{}{tag}.gguf", quant.name()))
}

#[derive(Debug)]
pub struct ThreeDPlan {
    pub source: PathBuf,
    pub output: PathBuf,
    pub quant: ThreeDQuant,
    pub threads: usize,
    pub overrides: Overrides,
}

pub fn prepare(source: &Path, out_dir: &Path, quant: ThreeDQuant, threads: usize, overrides: Overrides) -> io::Result<ThreeDPlan> {
    if source.extension().and_then(|e| e.to_str()) != Some("safetensors") || !source.is_file() {
        return Err(bad("3d source must be an existing .safetensors file"));
    }
    let output = out_dir.join(output_name(source, quant, &overrides.tag, overrides.name.as_deref())?);
    if output.exists() {
        return Err(bad(format!("{} already exists; Ojon never overwrites", output.display())));
    }
    if let (Ok(a), Ok(b)) = (source.canonicalize(), out_dir.canonicalize()) {
        if a.parent() == Some(b.as_path()) && output.file_name() == source.file_name() {
            return Err(bad("output would overwrite the source"));
        }
    }
    Ok(ThreeDPlan { source: source.to_path_buf(), output, quant, threads: threads.max(1), overrides })
}

/// Read-only plan: per-type tensor counts, output size and the streaming RAM bound.
pub fn describe(plan: &ThreeDPlan) -> io::Result<Value> {
    let tensors = read_header(&plan.source, &plan.overrides.prefix)?;
    let blocks = block_count(&tensors);
    let mut kinds: BTreeMap<&'static str, (u64, u64)> = BTreeMap::new();
    let (mut src_bytes, mut dst_bytes, mut largest) = (0u64, 0u64, 0u64);
    for t in &tensors {
        let fate = decide(t, plan, blocks)?;
        let label = match fate {
            Fate::Quant(k) => k.name(),
            Fate::Keep(GGML_F32) => "F32",
            Fate::Keep(GGML_F16) => "F16",
            Fate::Keep(_) => "BF16",
        };
        let size = out_bytes(t, fate)?;
        let e = kinds.entry(label).or_default();
        e.0 += 1;
        e.1 += size;
        src_bytes += t.end - t.begin;
        dst_bytes += size;
        largest = largest.max((t.end - t.begin) + t.elements() * 4);
    }
    Ok(json!({
        "schema": "ojon.3d_plan.v1",
        "model_family": "3d",
        "source": plan.source.to_string_lossy(),
        "output": plan.output.to_string_lossy(),
        "quant": plan.quant.name(),
        "blocks": blocks,
        "tensors": tensors.len(),
        "tensor_types": kinds.iter().map(|(k, (n, b))| (k.to_string(), json!({"tensors": n, "bytes": b}))).collect::<BTreeMap<_, _>>(),
        "source_bytes": src_bytes,
        "output_bytes": dst_bytes,
        "peak_ram_bound_bytes": largest * 2,
        "status": "PLAN_ONLY",
        "output_load_verified": false,
    }))
}

fn put_str(w: &mut impl Write, s: &str) -> io::Result<()> {
    w.write_all(&(s.len() as u64).to_le_bytes())?;
    w.write_all(s.as_bytes())
}

fn put_kv_str(w: &mut impl Write, key: &str, value: &str) -> io::Result<()> {
    put_str(w, key)?;
    w.write_all(&8u32.to_le_bytes())?;
    put_str(w, value)
}

fn put_kv_u32(w: &mut impl Write, key: &str, value: u32) -> io::Result<()> {
    put_str(w, key)?;
    w.write_all(&4u32.to_le_bytes())?;
    w.write_all(&value.to_le_bytes())
}

fn pad(w: &mut impl Write, from: u64) -> io::Result<u64> {
    let to = from.div_ceil(ALIGN) * ALIGN;
    w.write_all(&vec![0u8; (to - from) as usize])?;
    Ok(to)
}

fn write_header(w: &mut impl Write, plan: &ThreeDPlan, tensors: &[SrcTensor], fates: &[Fate]) -> io::Result<()> {
    let mut h: Vec<u8> = Vec::new();
    h.extend_from_slice(b"GGUF");
    h.extend_from_slice(&3u32.to_le_bytes());
    h.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    h.extend_from_slice(&6u64.to_le_bytes());
    let name = plan.output.file_stem().and_then(|s| s.to_str()).unwrap_or("3d");
    put_kv_str(&mut h, "general.architecture", "trellis-dit")?;
    put_kv_str(&mut h, "general.name", name)?;
    put_kv_u32(&mut h, "general.file_type", plan.quant.general_file_type())?;
    put_kv_u32(&mut h, "general.quantization_version", 2)?;
    put_kv_str(&mut h, "ojon.family", "3d")?;
    put_kv_str(&mut h, "ojon.quant", plan.quant.name())?;
    let mut offset = 0u64;
    for (t, fate) in tensors.iter().zip(fates) {
        let ggml_type = match fate {
            Fate::Quant(k) => k.ggml_type(),
            Fate::Keep(ty) => *ty,
        };
        put_str(&mut h, &t.name)?;
        h.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
        for dim in t.shape.iter().rev() {
            h.extend_from_slice(&dim.to_le_bytes());
        }
        h.extend_from_slice(&ggml_type.to_le_bytes());
        h.extend_from_slice(&offset.to_le_bytes());
        offset += out_bytes(t, *fate)?.div_ceil(ALIGN) * ALIGN;
    }
    w.write_all(&h)?;
    pad(w, h.len() as u64).map(|_| ())
}

fn to_f32(raw: &[u8], dtype: &str) -> Vec<f32> {
    match dtype {
        "F32" => raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect(),
        "F16" => raw.chunks_exact(2).map(|c| crate::quant::kquant::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
        _ => raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
    }
}

/// Inverse of ConvRot's grouped regular-Hadamard rotation (H is symmetric and orthonormal, so
/// H^-1 = H): one order-4 butterfly per base-4 digit, scaled by 1/sqrt(group).
fn unrotate(values: &mut [f32], group: usize) {
    let norm = 1.0 / (group as f32).sqrt();
    for chunk in values.chunks_exact_mut(group) {
        let mut stride = 1;
        while stride < group {
            for base in (0..group).step_by(4 * stride) {
                for off in 0..stride {
                    let i = base + off;
                    let (a, b, c, d) = (chunk[i], chunk[i + stride], chunk[i + 2 * stride], chunk[i + 3 * stride]);
                    chunk[i] = a + b + c - d;
                    chunk[i + stride] = a + b - c + d;
                    chunk[i + 2 * stride] = a - b + c + d;
                    chunk[i + 3 * stride] = -a + b + c + d;
                }
            }
            stride *= 4;
        }
        chunk.iter_mut().for_each(|v| *v *= norm);
    }
}

/// Source values as f32; int8 sources are dequantized by their row scale and un-rotated.
fn source_f32(raw: &[u8], t: &SrcTensor, scales: &[u8]) -> Vec<f32> {
    if t.dtype != "I8" {
        return to_f32(raw, &t.dtype);
    }
    let cols = t.shape[1] as usize;
    let row_scale = |r: usize| f32::from_le_bytes(scales[r * 4..r * 4 + 4].try_into().unwrap());
    let mut values: Vec<f32> = raw.iter().enumerate().map(|(i, &q)| q as i8 as f32 * row_scale(i / cols)).collect();
    if t.rotate > 0 {
        unrotate(&mut values, t.rotate);
    }
    values
}

fn quantize_tensor(raw: &[u8], scales: &[u8], t: &SrcTensor, kind: KQuant, threads: usize) -> Vec<u8> {
    let (rows, cols) = (t.shape[0] as usize, t.shape[1] as usize);
    let values = source_f32(raw, t, scales);
    let row_bytes = kind.row_bytes(cols);
    let mut out = vec![0u8; rows * row_bytes];
    let per = rows.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (chunk, dst) in out.chunks_mut(per * row_bytes).enumerate() {
            let src = &values[chunk * per * cols..];
            scope.spawn(move || {
                for (r, d) in dst.chunks_exact_mut(row_bytes).enumerate() {
                    quantize_row(kind, &src[r * cols..(r + 1) * cols], d);
                }
            });
        }
    });
    out
}

pub fn run(plan: &ThreeDPlan, mut log: impl FnMut(&str)) -> io::Result<Value> {
    let started = Instant::now();
    let tensors = read_header(&plan.source, &plan.overrides.prefix)?;
    let blocks = block_count(&tensors);
    let fates = tensors.iter().map(|t| decide(t, plan, blocks)).collect::<io::Result<Vec<_>>>()?;
    let part = plan.output.with_extension("gguf.part");
    let mut input = File::open(&plan.source)?;
    let mut out = BufWriter::with_capacity(1 << 20, File::create(&part)?);
    let result = (|| -> io::Result<()> {
        write_header(&mut out, plan, &tensors, &fates)?;
        for (i, (t, fate)) in tensors.iter().zip(&fates).enumerate() {
            let mut raw = vec![0u8; (t.end - t.begin) as usize];
            input.seek(SeekFrom::Start(t.begin))?;
            input.read_exact(&mut raw)?;
            let payload = match fate {
                Fate::Quant(kind) => {
                    let mut scales = Vec::new();
                    if let Some((b, e)) = t.scale {
                        scales = vec![0u8; (e - b) as usize];
                        input.seek(SeekFrom::Start(b))?;
                        input.read_exact(&mut scales)?;
                    }
                    quantize_tensor(&raw, &scales, t, *kind, plan.threads)
                }
                Fate::Keep(_) => raw,
            };
            debug_assert_eq!(payload.len() as u64, out_bytes(t, *fate)?);
            out.write_all(&payload)?;
            pad(&mut out, payload.len() as u64)?;
            if i % 50 == 0 {
                log(&format!("{}/{} {}", i + 1, tensors.len(), t.name));
            }
        }
        out.flush()
    })();
    drop(out);
    if let Err(e) = result {
        let _ = fs::remove_file(&part);
        return Err(e);
    }
    fs::rename(&part, &plan.output)?;
    let out_size = fs::metadata(&plan.output)?.len();
    Ok(json!({
        "schema": "ojon.3d_receipt.v1",
        "source": plan.source.to_string_lossy(),
        "output": plan.output.to_string_lossy(),
        "quant": plan.quant.name(),
        "tensors": tensors.len(),
        "quantized_tensors": fates.iter().filter(|f| matches!(f, Fate::Quant(_))).count(),
        "output_bytes": out_size,
        "seconds": started.elapsed().as_secs_f64(),
        "status": "WRITTEN",
        "output_load_verified": false,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quant::kquant::dequantize_row;

    fn bf16(v: f32) -> [u8; 2] {
        ((v.to_bits() >> 16) as u16).to_le_bytes()
    }

    fn write_source(dir: &Path) -> PathBuf {
        let specs: [(&str, Vec<u64>); 3] =
            [("blocks.0.mlp.mlp.0.weight", vec![4, 256]), ("blocks.0.mlp.mlp.0.bias", vec![4]), ("blocks.0.norm2.weight", vec![6])];
        let (mut header, mut data, mut off) = (serde_json::Map::new(), Vec::new(), 0usize);
        for (name, shape) in &specs {
            let n: u64 = shape.iter().product();
            for i in 0..n {
                data.extend_from_slice(&bf16(((i as f32) * 0.37).sin() * 0.1));
            }
            let end = data.len();
            header.insert(name.to_string(), json!({"dtype": "BF16", "shape": shape, "data_offsets": [off, end]}));
            off = end;
        }
        let raw = serde_json::to_vec(&header).unwrap();
        let path = dir.join("tiny_dit_bf16.safetensors");
        let mut f = File::create(&path).unwrap();
        f.write_all(&(raw.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&raw).unwrap();
        f.write_all(&data).unwrap();
        path
    }

    fn u64_at(b: &[u8], p: usize) -> u64 {
        u64::from_le_bytes(b[p..p + 8].try_into().unwrap())
    }

    #[test]
    fn writes_gguf_with_quantized_linear_and_kept_small_tensors() {
        let dir = std::env::temp_dir().join(format!("ojon_3d_test_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let src = write_source(&dir);
        let plan = prepare(&src, &dir, ThreeDQuant::Q5K, 2, Overrides::default()).unwrap();
        assert!(plan.output.file_name().unwrap().to_str().unwrap() == "tiny_dit-Ojon-Q5_K.gguf");
        let plan_json = describe(&plan).unwrap();
        assert_eq!(plan_json["tensor_types"]["Q5_K"]["tensors"], 1);
        run(&plan, |_| {}).unwrap();
        let b = fs::read(&plan.output).unwrap();
        assert_eq!(&b[..4], b"GGUF");
        assert_eq!(u64_at(&b, 8), 3);
        let mut p = 24;
        for _ in 0..6 {
            p += 8 + u64_at(&b, p) as usize;
            let ty = u32::from_le_bytes(b[p..p + 4].try_into().unwrap());
            p += 4;
            if ty == 8 {
                p += 8 + u64_at(&b, p) as usize;
            } else {
                p += 4;
            }
        }
        let mut infos = Vec::new();
        for _ in 0..3 {
            let name = String::from_utf8(b[p + 8..p + 8 + u64_at(&b, p) as usize].to_vec()).unwrap();
            p += 8 + name.len();
            let nd = u32::from_le_bytes(b[p..p + 4].try_into().unwrap()) as usize;
            p += 4 + 8 * nd;
            let ty = u32::from_le_bytes(b[p..p + 4].try_into().unwrap());
            infos.push((name, ty, u64_at(&b, p + 4) as usize));
            p += 12;
        }
        let base = p.div_ceil(32) * 32;
        let w = infos.iter().find(|i| i.0 == "blocks.0.mlp.mlp.0.weight").unwrap();
        assert_eq!(w.1, 13);
        let bias = infos.iter().find(|i| i.0 == "blocks.0.mlp.mlp.0.bias").unwrap();
        assert_eq!(bias.1, 30);
        let mut y = vec![0f32; 256];
        dequantize_row(KQuant::Q5K, &b[base + w.2..base + w.2 + 176], &mut y);
        let expect: Vec<f32> = (0..256).map(|i| ((i as f32) * 0.37).sin() * 0.1).collect();
        let err: f32 = y.iter().zip(&expect).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(err < 0.01, "max error {err}");
        assert_eq!(b.len() % 32, 0);
        assert!(prepare(&src, &dir, ThreeDQuant::Q5K, 2, Overrides::default()).is_err(), "must refuse to overwrite");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn q4_km_schedule_adds_bits_to_edges_only() {
        assert!(more_bits(0, 30) && more_bits(29, 30) && more_bits(5, 30));
        assert!(!more_bits(6, 30) && !more_bits(7, 30) && more_bits(8, 30));
        assert_eq!(ThreeDQuant::from_name("Q8_0").is_err(), true);
    }
}
