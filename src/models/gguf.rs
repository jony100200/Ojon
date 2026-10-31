//! Bounded, read-only GGUF container inspection.
//!
//! Reads only metadata and tensor descriptors, never tensor payloads. This is
//! a *container/catalog check*, NOT inference, tensor checksum, or quality QA.
//! v2 little-endian and v3 little-/big-endian headers are supported.

use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub fn ggml_type_name(storage_type: u32) -> &'static str {
    match storage_type {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        34 => "TQ1_0",
        35 => "TQ2_0",
        39 => "MXFP4",
        _ => "UNKNOWN",
    }
}

/// Returns `(block_elements, block_bytes)` for standard GGML storage types.
pub fn ggml_type_block_info(storage_type: u32) -> Option<(u64, u64)> {
    match storage_type {
        0 => Some((1, 4)),      // F32
        1 => Some((1, 2)),      // F16
        2 => Some((32, 18)),    // Q4_0
        3 => Some((32, 20)),    // Q4_1
        6 => Some((32, 22)),    // Q5_0
        7 => Some((32, 24)),    // Q5_1
        8 => Some((32, 34)),    // Q8_0
        9 => Some((32, 36)),    // Q8_1
        10 => Some((256, 84)),  // Q2_K
        11 => Some((256, 110)), // Q3_K
        12 => Some((256, 144)), // Q4_K
        13 => Some((256, 176)), // Q5_K
        14 => Some((256, 210)), // Q6_K
        15 => Some((256, 292)), // Q8_K
        16 => Some((256, 66)),  // IQ2_XXS
        17 => Some((256, 74)),  // IQ2_XS
        18 => Some((256, 98)),  // IQ3_XXS
        19 => Some((256, 50)),  // IQ1_S
        20 => Some((32, 18)),   // IQ4_NL
        21 => Some((256, 110)), // IQ3_S
        22 => Some((256, 82)),  // IQ2_S
        23 => Some((256, 136)), // IQ4_XS
        24 => Some((1, 1)),     // I8
        25 => Some((1, 2)),     // I16
        26 => Some((1, 4)),     // I32
        27 => Some((1, 8)),     // I64
        28 => Some((1, 8)),     // F64
        29 => Some((256, 56)),  // IQ1_M
        30 => Some((1, 2)),     // BF16
        34 => Some((256, 54)),  // TQ1_0
        35 => Some((256, 66)),  // TQ2_0
        39 => Some((32, 17)),   // MXFP4
        _ => None,
    }
}

pub fn classify_tensor_role(name: &str) -> &'static str {
    if name.contains("ffn_gate_inp") || name.contains("exp_probs_b") {
        "router"
    } else if name.contains("_shexp") || name.contains("shared_expert") {
        "shared_expert"
    } else if name.contains("_exps") || name.contains(".experts.") {
        "routed_expert"
    } else if name.contains("token_embd") || name == "output.weight" || name.contains("lm_head") {
        "embeddings_and_head"
    } else if name.contains("norm")
        || name.ends_with(".bias")
        || name.contains("ssm_a")
        || name.contains("ssm_d")
    {
        "norms_and_biases"
    } else if name.contains("attn_") || name.contains("ssm_") {
        "attention"
    } else if name.contains("ffn_") {
        "ffn_dense"
    } else {
        "other"
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RuntimeMemoryEstimate {
    pub context_tokens: u32,
    pub batch_size: u32,
    pub weights_bytes: u64,
    pub kv_cache_bytes: u64,
    pub compute_buffer_bytes: u64,
    pub runtime_overhead_bytes: u64,
    pub total_estimated_bytes: u64,
}

impl RuntimeMemoryEstimate {
    pub fn as_json(&self) -> Value {
        json!({
            "context_tokens": self.context_tokens,
            "batch_size": self.batch_size,
            "weights_bytes": self.weights_bytes,
            "kv_cache_bytes": self.kv_cache_bytes,
            "compute_buffer_bytes": self.compute_buffer_bytes,
            "runtime_overhead_bytes": self.runtime_overhead_bytes,
            "total_estimated_bytes": self.total_estimated_bytes,
            "total_estimated_mib": self.total_estimated_bytes / (1024 * 1024),
        })
    }
}

#[derive(Clone, Debug)]
pub struct ModelTopology {
    pub summary: GgufSummary,
    pub total_parameters: u64,
    pub payload_bytes: u64,
    pub effective_bpw: f64,
    pub block_count: usize,
    pub embedding_length: u64,
    pub expert_count: usize,
    pub is_moe: bool,
    pub is_high_precision_source: bool,
    pub dominant_storage_type: String,
    pub storage_type_histogram: BTreeMap<String, usize>,
    pub role_type_breakdown: BTreeMap<String, BTreeMap<String, usize>>,
}

impl ModelTopology {
    pub fn estimate_runtime_memory(
        &self,
        context_tokens: u32,
        batch_size: u32,
    ) -> RuntimeMemoryEstimate {
        let weights_bytes = self.payload_bytes;
        let blocks = (self.block_count as u64).max(1);
        let embd = self.embedding_length.max(512);
        // F16 KV cache: 2 (K+V) * blocks * embd * 2 bytes/elem * context_tokens
        // (For GQA models this is an upper-bound safety estimate)
        let kv_cache_bytes = 2u64
            .saturating_mul(blocks)
            .saturating_mul(embd)
            .saturating_mul(2)
            .saturating_mul(context_tokens as u64);
        let moe_extra = if self.is_moe { 256 * 1024 * 1024 } else { 0 };
        let compute_buffer_bytes = ((batch_size as u64).saturating_mul(embd).saturating_mul(32))
            .max(128 * 1024 * 1024)
            .saturating_add(moe_extra);
        let runtime_overhead_bytes = 384 * 1024 * 1024;
        let total_estimated_bytes = weights_bytes
            .saturating_add(kv_cache_bytes)
            .saturating_add(compute_buffer_bytes)
            .saturating_add(runtime_overhead_bytes);
        RuntimeMemoryEstimate {
            context_tokens,
            batch_size,
            weights_bytes,
            kv_cache_bytes,
            compute_buffer_bytes,
            runtime_overhead_bytes,
            total_estimated_bytes,
        }
    }

    pub fn as_json(&self) -> Value {
        let est_4k = self.estimate_runtime_memory(4096, 512);
        json!({
            "architecture": self.summary.architecture,
            "model_name": self.summary.model_name,
            "file_bytes": self.summary.file_bytes,
            "payload_bytes": self.payload_bytes,
            "tensor_count": self.summary.tensor_count,
            "total_parameters": self.total_parameters,
            "effective_bpw": (self.effective_bpw * 1000.0).round() / 1000.0,
            "block_count": self.block_count,
            "embedding_length": self.embedding_length,
            "expert_count": self.expert_count,
            "is_moe": self.is_moe,
            "is_high_precision_source": self.is_high_precision_source,
            "dominant_storage_type": self.dominant_storage_type,
            "storage_type_histogram": self.storage_type_histogram,
            "role_type_breakdown": self.role_type_breakdown,
            "runtime_memory_estimate_4k": est_4k.as_json(),
        })
    }
}

pub fn inspect_model_topology(path: &Path) -> io::Result<ModelTopology> {
    let (summary, tensors) = inspect_gguf_inner(path, true)?;
    let mut total_parameters = 0u64;
    let mut max_block_idx: Option<usize> = None;
    let mut embedding_length = 0u64;
    let mut expert_count = 0usize;
    let mut storage_type_histogram: BTreeMap<String, usize> = BTreeMap::new();
    let mut storage_type_elements: BTreeMap<String, u64> = BTreeMap::new();
    let mut role_type_breakdown: BTreeMap<String, BTreeMap<String, usize>> = BTreeMap::new();

    for t in &tensors {
        total_parameters = total_parameters.saturating_add(t.elements);
        if let Some(rest) = t.name.strip_prefix("blk.") {
            if let Some(dot_idx) = rest.find('.') {
                if let Ok(idx) = rest[..dot_idx].parse::<usize>() {
                    max_block_idx = Some(max_block_idx.map_or(idx, |m| m.max(idx)));
                }
            }
        }
        if t.name == "token_embd.weight" && !t.shape.is_empty() {
            embedding_length = t.shape[0];
        }
        if (t.name.contains("_exps.") || t.name.contains("ffn_gate_exps")) && t.shape.len() >= 3 {
            let n_exp = t.shape[2] as usize;
            if n_exp > expert_count {
                expert_count = n_exp;
            }
        } else if t.name.contains("ffn_gate_inp.weight") && t.shape.len() >= 2 {
            let n_exp = t.shape[1] as usize;
            if n_exp > expert_count {
                expert_count = n_exp;
            }
        }
        let ty_name = ggml_type_name(t.storage_type).to_string();
        *storage_type_histogram.entry(ty_name.clone()).or_insert(0) += 1;
        *storage_type_elements.entry(ty_name.clone()).or_insert(0) += t.elements;
        let role = classify_tensor_role(&t.name).to_string();
        *role_type_breakdown
            .entry(role)
            .or_default()
            .entry(ty_name)
            .or_insert(0) += 1;
    }

    let payload_bytes = summary.file_bytes.saturating_sub(summary.data_offset);
    let effective_bpw = if total_parameters > 0 {
        (payload_bytes as f64 * 8.0) / (total_parameters as f64)
    } else {
        0.0
    };
    let dominant_storage_type = storage_type_elements
        .iter()
        .max_by_key(|(_, elems)| *elems)
        .map(|(k, _)| k.clone())
        .unwrap_or_else(|| "UNKNOWN".to_string());
    let is_high_precision_source = matches!(dominant_storage_type.as_str(), "F32" | "F16" | "BF16");
    let is_moe = expert_count > 1
        || role_type_breakdown.contains_key("router")
        || role_type_breakdown.contains_key("routed_expert");

    Ok(ModelTopology {
        summary,
        total_parameters,
        payload_bytes,
        effective_bpw,
        block_count: max_block_idx.map_or(0, |m| m + 1),
        embedding_length,
        expert_count,
        is_moe,
        is_high_precision_source,
        dominant_storage_type,
        storage_type_histogram,
        role_type_breakdown,
    })
}

#[derive(Clone, Debug)]
pub struct GgufSummary {
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_count: u64,
    pub file_bytes: u64,
    pub architecture: Option<String>,
    pub model_name: Option<String>,
    pub file_type: Option<u32>,
    pub data_offset: u64,
    pub alignment: u32,
    pub split_count: Option<u16>,
    /// Zero-based shard index from `split.no`; None for a single-file GGUF.
    pub split_no: Option<u16>,
    pub big_endian: bool,
}

#[derive(Clone, Debug)]
pub struct GgufTensor {
    pub name: String,
    pub elements: u64,
    pub shape: Vec<u64>,
    pub storage_type: u32,
    pub relative_offset: u64,
}
fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

struct Reader {
    file: File,
    size: u64,
    big_endian: bool,
}
impl Reader {
    fn position(&mut self) -> io::Result<u64> {
        self.file.stream_position()
    }
    fn ensure(&mut self, bytes: u64) -> io::Result<()> {
        let end = self
            .position()?
            .checked_add(bytes)
            .ok_or_else(|| invalid("offset overflow"))?;
        if end > self.size {
            return Err(invalid("GGUF metadata or tensor catalog is truncated"));
        }
        Ok(())
    }
    fn raw<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        self.ensure(N as u64)?;
        let mut buf = [0; N];
        self.file.read_exact(&mut buf)?;
        Ok(buf)
    }
    #[allow(dead_code)]
    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.raw::<1>()?[0])
    }
    fn u16(&mut self) -> io::Result<u16> {
        let b = self.raw::<2>()?;
        Ok(if self.big_endian {
            u16::from_be_bytes(b)
        } else {
            u16::from_le_bytes(b)
        })
    }
    fn u32(&mut self) -> io::Result<u32> {
        let b = self.raw::<4>()?;
        Ok(if self.big_endian {
            u32::from_be_bytes(b)
        } else {
            u32::from_le_bytes(b)
        })
    }
    fn u64(&mut self) -> io::Result<u64> {
        let b = self.raw::<8>()?;
        Ok(if self.big_endian {
            u64::from_be_bytes(b)
        } else {
            u64::from_le_bytes(b)
        })
    }
    fn skip(&mut self, bytes: u64) -> io::Result<()> {
        self.ensure(bytes)?;
        let next = self.position()? + bytes;
        self.file.seek(SeekFrom::Start(next))?;
        Ok(())
    }
    // Large metadata strings (e.g. chat templates) can be skipped without RAM allocations.
    fn string(&mut self, capture_limit: usize) -> io::Result<Option<String>> {
        let n = self.u64()?;
        self.ensure(n)?;
        if n > capture_limit as u64 {
            self.skip(n)?;
            return Ok(None);
        }
        let mut bytes = vec![0u8; n as usize];
        self.file.read_exact(&mut bytes)?;
        String::from_utf8(bytes)
            .map(Some)
            .map_err(|_| invalid("invalid UTF-8 GGUF string"))
    }
    fn skip_value(&mut self, ty: u32) -> io::Result<()> {
        if ty == 8 {
            let n = self.u64()?;
            return self.skip(n);
        }
        if ty == 9 {
            let elem_type = self.u32()?;
            let count = self.u64()?;
            if elem_type == 8 {
                // Each string consumes at least its eight-byte length field.
                if count > self.size / 8 {
                    return Err(invalid("invalid GGUF string array length"));
                }
                for _ in 0..count {
                    let len = self.u64()?;
                    self.skip(len)?;
                }
                return Ok(());
            }
            let width = scalar_width(elem_type)
                .ok_or_else(|| invalid("unsupported GGUF array element type"))?;
            let bytes = count
                .checked_mul(width)
                .ok_or_else(|| invalid("GGUF array overflow"))?;
            return self.skip(bytes);
        }
        let width = scalar_width(ty).ok_or_else(|| invalid("unsupported GGUF metadata type"))?;
        self.skip(width)
    }
}

fn scalar_width(ty: u32) -> Option<u64> {
    match ty {
        0 | 1 | 7 => Some(1),    // unsigned/signed byte, bool
        2 | 3 => Some(2),        // uint16/int16
        4..=6 => Some(4),    // uint32/int32/f32
        10..=12 => Some(8), // uint64/int64/f64
        _ => None,
    }
}
fn align_up(pos: u64, align: u64) -> io::Result<u64> {
    let padding = (align - (pos % align)) % align;
    pos.checked_add(padding)
        .ok_or_else(|| invalid("alignment overflow"))
}

/// Inspects a local GGUF file without allocating or loading model tensors.
///
/// Verifies the header, metadata traversal, tensor descriptors, alignment and
/// basic in-file data offsets. Full payload integrity and runtime loadability
/// require the actual target runtime and are NOT established here.
fn inspect_gguf_inner(path: &Path, collect: bool) -> io::Result<(GgufSummary, Vec<GgufTensor>)> {
    let file = File::open(path)?;
    if !fs::metadata(path)?.is_file() {
        return Err(invalid("not a regular file"));
    }
    let size = file.metadata()?.len();
    if size < 24 {
        return Err(invalid("GGUF file is shorter than its header"));
    }
    let mut reader = Reader {
        file,
        size,
        big_endian: false,
    };
    if &reader.raw::<4>()? != b"GGUF" {
        return Err(invalid("not a GGUF file (wrong magic)"));
    }
    let ver = reader.raw::<4>()?;
    let (version, big_endian) = match (u32::from_le_bytes(ver), u32::from_be_bytes(ver)) {
        (2, _) => (2, false),
        (3, _) => (3, false),
        (_, 3) => (3, true),
        _ => return Err(invalid("unsupported GGUF version (requires v2 or v3)")),
    };
    reader.big_endian = big_endian;
    let tensor_count = reader.u64()?;
    let metadata_count = reader.u64()?;
    // Reject impossible counts before looping over an untrusted file.
    if metadata_count > size / 12 || tensor_count > size / 24 {
        return Err(invalid("impossible GGUF item counts for file size"));
    }
    let mut architecture = None;
    let mut model_name = None;
    let mut file_type = None;
    let mut split_count = None;
    let mut split_no = None;
    let mut alignment = 32u32;
    for _ in 0..metadata_count {
        let key = reader
            .string(65_535)?
            .ok_or_else(|| invalid("oversized GGUF metadata key"))?;
        let ty = reader.u32()?;
        match (key.as_str(), ty) {
            ("general.architecture", 8) => architecture = reader.string(256)?,
            ("general.name", 8) => model_name = reader.string(4096)?,
            ("general.file_type", 4) => file_type = Some(reader.u32()?),
            ("general.alignment", 4) => alignment = reader.u32()?,
            ("split.count", 2) => split_count = Some(reader.u16()?),
            ("split.no", 2) => split_no = Some(reader.u16()?),
            _ => reader.skip_value(ty)?,
        }
    }
    if alignment == 0 || !alignment.is_power_of_two() || alignment > 1_048_576 {
        return Err(invalid("invalid GGUF tensor alignment"));
    }
    let mut max_tensor_offset = 0u64;
    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        let name = reader
            .string(4096)?
            .ok_or_else(|| invalid("oversized tensor name"))?;
        let dimensions = reader.u32()?;
        if !(1..=4).contains(&dimensions) {
            return Err(invalid("unsupported tensor dimension count"));
        }
        let mut shape = Vec::new();
        let mut elements = 1u64;
        for _ in 0..dimensions {
            let dim = reader.u64()?;
            if dim == 0 {
                return Err(invalid("zero-sized tensor dimension"));
            }
            elements = elements
                .checked_mul(dim)
                .ok_or_else(|| invalid("tensor elements overflow"))?;
            if collect {
                shape.push(dim);
            }
        }
        let storage_type = reader.u32()?;
        let offset = reader.u64()?;
        if collect {
            tensors.push(GgufTensor {
                name,
                elements,
                shape,
                storage_type,
                relative_offset: offset,
            });
        }
        if offset % u64::from(alignment) != 0 {
            return Err(invalid("misaligned GGUF tensor offset"));
        }
        max_tensor_offset = max_tensor_offset.max(offset);
    }
    let data_offset = align_up(reader.position()?, u64::from(alignment))?;
    if tensor_count > 0 {
        let payload = data_offset
            .checked_add(max_tensor_offset)
            .ok_or_else(|| invalid("GGUF tensor data offset overflow"))?;
        if payload >= size {
            return Err(invalid("tensor payload offset outside GGUF file"));
        }
    } else if data_offset > size {
        return Err(invalid("missing GGUF data section"));
    }
    Ok((
        GgufSummary {
            version,
            tensor_count,
            metadata_count,
            file_bytes: size,
            architecture,
            model_name,
            file_type,
            data_offset,
            alignment,
            split_count,
            split_no,
            big_endian,
        },
        tensors,
    ))
}

pub fn inspect_gguf(path: &Path) -> io::Result<GgufSummary> {
    Ok(inspect_gguf_inner(path, false)?.0)
}

/// Read names/shapes/storage types without loading tensor data.
pub fn inspect_gguf_tensors(path: &Path) -> io::Result<Vec<GgufTensor>> {
    Ok(inspect_gguf_inner(path, true)?.1)
}

/// For GGUF-format llama-imatrix files: read bounded F32 in_sum2 and counts
/// arrays and derive the mean squared activation-energy proxy per tensor.
/// This is a proxy, not measured post-quantization task sensitivity.
pub fn read_imatrix_energy(path: &Path) -> io::Result<Vec<(String, f64)>> {
    let summary = inspect_gguf(path)?;
    if summary.big_endian {
        return Err(invalid("big-endian imatrix F32 unsupported"));
    }
    let descriptors = inspect_gguf_tensors(path)?;
    let mut sums = std::collections::HashMap::new();
    let mut counts = std::collections::HashMap::new();
    let mut file = File::open(path)?;
    for tensor in descriptors {
        let (name, is_sum) = if let Some(base) = tensor.name.strip_suffix(".in_sum2") {
            (base.to_owned(), true)
        } else if let Some(base) = tensor.name.strip_suffix(".counts") {
            (base.to_owned(), false)
        } else {
            continue;
        };
        if tensor.storage_type != 0 || tensor.elements > 2_000_000 {
            return Err(invalid(
                "importance matrix requires bounded F32 tensor arrays",
            ));
        }
        let len = tensor
            .elements
            .checked_mul(4)
            .ok_or_else(|| invalid("imatrix overflow"))?;
        let offset = summary
            .data_offset
            .checked_add(tensor.relative_offset)
            .ok_or_else(|| invalid("imatrix offset overflow"))?;
        if offset
            .checked_add(len)
            .is_none_or(|end| end > summary.file_bytes)
        {
            return Err(invalid("truncated imatrix tensor"));
        }
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes = [0u8; 4];
        let mut total = 0.0f64;
        for _ in 0..tensor.elements {
            file.read_exact(&mut bytes)?;
            let sample = f32::from_le_bytes(bytes) as f64;
            if !sample.is_finite() || sample < 0.0 {
                return Err(invalid("invalid imatrix F32 energy/count"));
            }
            total += sample;
        }
        if is_sum {
            sums.insert(name, (total, tensor.elements));
        } else {
            counts.insert(name, total);
        }
    }
    let mut result = Vec::new();
    for (name, (sum, nval)) in sums {
        if let Some(count) = counts.get(&name) {
            if *count > 0.0 && nval > 0 {
                let energy = sum / (*count * nval as f64);
                if energy.is_finite() && energy > 0.0 {
                    result.push((name, energy));
                }
            }
        }
    }
    if result.is_empty() {
        return Err(invalid("no valid GGUF imatrix in_sum2/counts pairs"));
    }
    result.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(result)
}

/// The shard files of a split GGUF, first shard first. A single-file GGUF
/// returns itself. The path must be the first shard (`split.no` 0, named
/// `<stem>-00001-of-NNNNN.gguf`), which is the file llama.cpp tools take;
/// every sibling must exist. Nothing is merged: `llama-quantize`,
/// `llama-imatrix` and `llama-perplexity` read all shards from the first.
pub fn shard_paths(path: &Path) -> io::Result<Vec<PathBuf>> {
    let summary = inspect_gguf(path)?;
    let count = summary.split_count.unwrap_or(1);
    if count <= 1 {
        return Ok(vec![path.to_path_buf()]);
    }
    if summary.split_no.unwrap_or(0) != 0 {
        return Err(invalid(format!(
            "{} is shard {} of {}; pass the first shard (-00001-of-{:05}.gguf)",
            path.display(),
            summary.split_no.unwrap_or(0) + 1,
            count,
            count
        )));
    }
    let name = path.file_name().and_then(|n| n.to_str()).ok_or_else(|| invalid("shard name is not UTF-8"))?;
    let stem = name.strip_suffix(".gguf").ok_or_else(|| invalid("shard name must end in .gguf"))?;
    let (head, total) = stem.rsplit_once("-of-").ok_or_else(|| invalid("split GGUF shard must be named <stem>-00001-of-NNNNN.gguf"))?;
    let (base, first) = head.rsplit_once('-').ok_or_else(|| invalid("split GGUF shard must be named <stem>-00001-of-NNNNN.gguf"))?;
    if first != "00001" || total.parse::<u16>().ok() != Some(count) {
        return Err(invalid(format!(
            "shard name {name} does not match split.count {count} (expected <stem>-00001-of-{count:05}.gguf)"
        )));
    }
    let dir = path.parent().map(Path::to_path_buf).unwrap_or_default();
    let mut shards = Vec::with_capacity(count as usize);
    for i in 1..=count {
        let p = dir.join(format!("{base}-{i:05}-of-{count:05}.gguf"));
        if !p.is_file() {
            return Err(invalid(format!("missing shard {}", p.display())));
        }
        shards.push(p);
    }
    Ok(shards)
}

/// Inspect a possibly split GGUF as one model: the first shard's metadata with
/// `file_bytes` and `tensor_count` summed over all shards, and the full
/// tensor catalog in shard order. Single files pass straight through.
pub fn inspect_sharded_gguf(path: &Path) -> io::Result<(GgufSummary, Vec<GgufTensor>)> {
    let shards = shard_paths(path)?;
    let mut summary = inspect_gguf(&shards[0])?;
    let mut tensors = inspect_gguf_tensors(&shards[0])?;
    for shard in &shards[1..] {
        let s = inspect_gguf(shard)?;
        summary.file_bytes = summary.file_bytes.checked_add(s.file_bytes).ok_or_else(|| invalid("shard size overflow"))?;
        summary.tensor_count = summary.tensor_count.checked_add(s.tensor_count).ok_or_else(|| invalid("shard tensor count overflow"))?;
        tensors.extend(inspect_gguf_tensors(shard)?);
    }
    Ok((summary, tensors))
}

#[cfg(test)]
mod shard_tests {
    use super::*;
    use std::fs;
    use std::io::{Seek, Write};

    fn shard(path: &Path, no: u16, count: u16, tensor: &str) {
        fn val_string(f: &mut fs::File, s: &str) {
            f.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
            f.write_all(s.as_bytes()).unwrap();
        }
        let mut f = fs::File::create(path).unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        f.write_all(&4u64.to_le_bytes()).unwrap();
        val_string(&mut f, "general.architecture");
        f.write_all(&8u32.to_le_bytes()).unwrap();
        val_string(&mut f, "qwen3");
        val_string(&mut f, "general.file_type");
        f.write_all(&4u32.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        val_string(&mut f, "split.count");
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&count.to_le_bytes()).unwrap();
        val_string(&mut f, "split.no");
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&no.to_le_bytes()).unwrap();
        val_string(&mut f, tensor);
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        let pos = f.stream_position().unwrap();
        f.write_all(&vec![0u8; ((32 - pos % 32) % 32) as usize]).unwrap();
        f.write_all(&[0u8; 32]).unwrap();
    }

    #[test]
    fn sharded_inspection_sums_shards_and_requires_the_first_one() {
        let tick = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!("ojon-shards-{}-{tick}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let a = dir.join("m-00001-of-00002.gguf");
        let b = dir.join("m-00002-of-00002.gguf");
        shard(&a, 0, 2, "blk.0.attn_q.weight");
        let err = shard_paths(&a).unwrap_err().to_string();
        assert!(err.contains("missing shard"), "{err}");
        shard(&b, 1, 2, "blk.1.attn_q.weight");
        assert_eq!(shard_paths(&a).unwrap(), vec![a.clone(), b.clone()]);
        let err = shard_paths(&b).unwrap_err().to_string();
        assert!(err.contains("pass the first shard"), "{err}");
        let (summary, tensors) = inspect_sharded_gguf(&a).unwrap();
        assert_eq!(summary.tensor_count, 2);
        assert_eq!(summary.split_count, Some(2));
        assert_eq!(summary.file_bytes, fs::metadata(&a).unwrap().len() + fs::metadata(&b).unwrap().len());
        assert_eq!(tensors.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(), ["blk.0.attn_q.weight", "blk.1.attn_q.weight"]);
        let single = dir.join("single.gguf");
        shard(&single, 0, 1, "token_embd.weight");
        assert_eq!(shard_paths(&single).unwrap(), vec![single.clone()]);
        let _ = fs::remove_dir_all(&dir);
    }
}
