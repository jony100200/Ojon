//! Restore quantized 1-D tensors (norm vectors) of a diffusion GGUF as exact F32 from the
//! floating source checkpoint. sd.cpp's converter quantizes RMSNorm weights whose length is a
//! multiple of the block size; ComfyUI-GGUF then reads the packed bytes as plain weights.
//! Ported from scripts/repair_ph3_q4_norms.py and verify_ph3_normsafe.py.

use super::gguf_rewrite::{read_layout, rewrite, Edits, TensorInfo};
use super::w4a8::numeric::{decode, source_width};
use crate::models::weight_patch::{read_safetensors_header, TensorLoc};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

const F32: u32 = 0;
const F16: u32 = 1;
const BF16: u32 = 30;
/// Prefix sd.cpp/ComfyUI use for diffusion weights; reference checkpoints usually lack it.
const PREFIX: &str = "model.diffusion_model.";

fn invalid(m: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, m.into())
}

fn is_quantized_vector(t: &TensorInfo) -> bool {
    t.dims.len() == 1 && ![F32, F16, BF16].contains(&t.ggml_type)
}

fn reference_for<'a>(name: &str, refs: &'a BTreeMap<String, TensorLoc>) -> Option<&'a TensorLoc> {
    refs.get(name).or_else(|| refs.get(name.strip_prefix(PREFIX)?))
}

/// The reference vector as little-endian F32 bytes (exact for BF16/F16/F32 sources).
fn f32_bytes(src: &mut File, loc: &TensorLoc, len: u64) -> io::Result<Vec<u8>> {
    let width = source_width(&loc.dtype).ok_or_else(|| invalid(format!("reference dtype {} is not floating", loc.dtype)))?;
    if loc.shape.len() != 1 || loc.shape[0] as u64 != len {
        return Err(invalid(format!("reference shape {:?} does not match length {len}", loc.shape)));
    }
    let mut raw = vec![0u8; loc.bytes as usize];
    src.seek(SeekFrom::Start(loc.offset))?;
    src.read_exact(&mut raw)?;
    Ok(raw.chunks_exact(width).flat_map(|b| decode(&loc.dtype, b).to_le_bytes()).collect())
}

pub struct Plan {
    pub restore: Vec<String>,
    pub tensors: usize,
}

/// Which tensors would be restored. Fails if a quantized vector has no floating reference.
pub fn plan(gguf: &Path, reference: &Path) -> io::Result<Plan> {
    let layout = read_layout(gguf)?;
    let refs = read_safetensors_header(reference)?;
    let mut restore = Vec::new();
    for t in layout.tensors.iter().filter(|t| is_quantized_vector(t)) {
        reference_for(&t.name, &refs).ok_or_else(|| invalid(format!("no reference tensor for {}", t.name)))?;
        restore.push(t.name.clone());
    }
    Ok(Plan { restore, tensors: layout.tensors.len() })
}

/// Write `out` with every quantized 1-D tensor replaced by its exact F32 reference.
pub fn restore(gguf: &Path, reference: &Path, out: &Path) -> io::Result<(Plan, u64)> {
    let p = plan(gguf, reference)?;
    let layout = read_layout(gguf)?;
    let refs = read_safetensors_header(reference)?;
    let mut src = File::open(reference)?;
    let mut edits = Edits::default();
    for t in layout.tensors.iter().filter(|t| p.restore.contains(&t.name)) {
        let bytes = f32_bytes(&mut src, reference_for(&t.name, &refs).unwrap(), t.dims[0])?;
        edits.tensors.insert(t.name.clone(), (F32, bytes));
    }
    let size = rewrite(gguf, out, &edits)?;
    Ok((p, size))
}

pub struct Verified {
    pub tensors: usize,
    pub restored_exactly: usize,
    pub unchanged_checked: usize,
    pub quantized_vectors_left: usize,
}

fn read_at(f: &mut File, at: u64, n: u64) -> io::Result<Vec<u8>> {
    let mut b = vec![0u8; n as usize];
    f.seek(SeekFrom::Start(at))?;
    f.read_exact(&mut b)?;
    Ok(b)
}

/// Every restored vector equals its reference exactly; every other tensor keeps its type and
/// matches the original at the first and last 128 bytes.
pub fn verify(original: &Path, restored: &Path, reference: &Path) -> io::Result<Verified> {
    let (a, b) = (read_layout(original)?, read_layout(restored)?);
    if a.tensors.len() != b.tensors.len() {
        return Err(invalid("tensor counts differ"));
    }
    let refs = read_safetensors_header(reference)?;
    let (mut fa, mut fb, mut fr) = (File::open(original)?, File::open(restored)?, File::open(reference)?);
    let mut v = Verified { tensors: a.tensors.len(), restored_exactly: 0, unchanged_checked: 0, quantized_vectors_left: 0 };
    for (x, y) in a.tensors.iter().zip(&b.tensors) {
        if x.name != y.name || x.dims != y.dims {
            return Err(invalid(format!("tensor table differs at {}", x.name)));
        }
        if is_quantized_vector(x) && y.ggml_type == F32 {
            let want = f32_bytes(&mut fr, reference_for(&x.name, &refs).ok_or_else(|| invalid(format!("no reference for {}", x.name)))?, x.dims[0])?;
            if read_at(&mut fb, b.data_offset + y.offset, want.len() as u64)? != want {
                return Err(invalid(format!("{} differs from its reference", x.name)));
            }
            v.restored_exactly += 1;
            continue;
        }
        if x.ggml_type != y.ggml_type {
            return Err(invalid(format!("{} changed type", x.name)));
        }
        let n = x.span.min(y.span).min(128);
        let ends = |f: &mut File, base: u64, t: &TensorInfo| -> io::Result<(Vec<u8>, Vec<u8>)> {
            Ok((read_at(f, base + t.offset, n)?, read_at(f, base + t.offset + t.span.min(y.span) - n, n)?))
        };
        if ends(&mut fa, a.data_offset, x)? != ends(&mut fb, b.data_offset, y)? {
            return Err(invalid(format!("{} payload differs", x.name)));
        }
        v.unchanged_checked += 1;
    }
    v.quantized_vectors_left = b.tensors.iter().filter(|t| is_quantized_vector(t)).count();
    Ok(v)
}
