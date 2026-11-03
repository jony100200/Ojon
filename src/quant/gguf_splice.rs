//! Build a GGUF from two quantizations of the SAME original: every tensor comes from
//! `base`, except the ones `take` selects, which come byte-for-byte from `donor`.
//!
//! Why: the recipe search probes one tensor class at a time. With an explicit
//! tensor-type file llama-quantize quantizes each tensor independently, so a probe
//! "base + class C at level L" equals splicing class C out of a full level-L quant.
//! One quantization per level plus fast splices replaces one full requantization
//! per probe (14.5 min each on Qwen3.8-27B BF16).
//!
//! Metadata is copied verbatim from `base` (so `general.file_type` keeps the base's
//! value). Single-file GGUF v3, little-endian only.
use crate::models::gguf::{inspect_gguf, inspect_gguf_tensors, GgufTensor};
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

fn invalid(m: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, m.into())
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpliceReport {
    pub tensors: usize,
    pub from_donor: usize,
    pub bytes: u64,
}

struct Source {
    file: File,
    data_offset: u64,
    /// name -> (relative offset, payload bytes incl. alignment padding, tensor)
    tensors: HashMap<String, (u64, u64, GgufTensor)>,
}

fn open(path: &Path) -> io::Result<(Source, Vec<GgufTensor>, u64, u32)> {
    let summary = inspect_gguf(path)?;
    if summary.big_endian || summary.version != 3 {
        return Err(invalid("splice supports little-endian GGUF v3 only"));
    }
    if summary.split_count.unwrap_or(1) > 1 {
        return Err(invalid("splice supports single-file GGUF only"));
    }
    let list = inspect_gguf_tensors(path)?;
    let file_len = summary.file_bytes;
    let mut by_offset: Vec<&GgufTensor> = list.iter().collect();
    by_offset.sort_by_key(|t| t.relative_offset);
    let payload_end = file_len - summary.data_offset;
    let mut tensors = HashMap::new();
    for (i, t) in by_offset.iter().enumerate() {
        let end = by_offset.get(i + 1).map_or(payload_end, |n| n.relative_offset);
        tensors.insert(t.name.clone(), (t.relative_offset, end - t.relative_offset, (*t).clone()));
    }
    let file = File::open(path)?;
    Ok((Source { file, data_offset: summary.data_offset, tensors }, list, summary.data_offset, summary.alignment))
}

fn info_bytes(t: &GgufTensor) -> u64 {
    8 + t.name.len() as u64 + 4 + 8 * t.shape.len() as u64 + 4 + 8
}

/// Byte offset where the tensor-info table starts (end of metadata). The reader does
/// not record it, so try every position the alignment allows and confirm it by the
/// first tensor's name.
fn info_start(file: &mut File, tensors: &[GgufTensor], data_offset: u64, alignment: u32) -> io::Result<u64> {
    let table: u64 = tensors.iter().map(info_bytes).sum();
    let first = tensors.first().ok_or_else(|| invalid("GGUF has no tensors"))?;
    for pad in 0..u64::from(alignment) {
        let Some(start) = data_offset.checked_sub(pad + table) else { continue };
        if start < 24 {
            continue;
        }
        file.seek(SeekFrom::Start(start))?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        if u64::from_le_bytes(len) != first.name.len() as u64 {
            continue;
        }
        let mut name = vec![0u8; first.name.len()];
        file.read_exact(&mut name)?;
        if name == first.name.as_bytes() {
            return Ok(start);
        }
    }
    Err(invalid("could not locate the GGUF tensor-info table"))
}

pub fn splice(base: &Path, donor: &Path, take: impl Fn(&GgufTensor) -> bool, out: &Path) -> io::Result<SpliceReport> {
    if out.exists() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, "splice output already exists"));
    }
    let (mut b, base_list, data_offset, alignment) = open(base)?;
    let (mut d, _, _, _) = open(donor)?;
    let align = u64::from(alignment);
    let meta_end = info_start(&mut b.file, &base_list, data_offset, alignment)?;

    // Plan: per tensor (in base order) pick a source, its type and payload size.
    let mut plan = Vec::with_capacity(base_list.len());
    let mut from_donor = 0;
    for t in &base_list {
        if take(t) {
            let (off, len, dt) = d.tensors.get(&t.name).ok_or_else(|| invalid(format!("donor lacks tensor {}", t.name)))?;
            if dt.shape != t.shape {
                return Err(invalid(format!("shape differs for {}", t.name)));
            }
            from_donor += 1;
            plan.push((t.clone(), dt.storage_type, true, *off, *len));
        } else {
            let (off, len, _) = b.tensors[&t.name].clone();
            plan.push((t.clone(), t.storage_type, false, off, len));
        }
    }
    if from_donor == 0 {
        return Err(invalid("selection matched no tensors"));
    }

    let tmp = out.with_extension("splice-part");
    let mut w = BufWriter::with_capacity(8 << 20, File::create(&tmp)?);
    // Header + metadata copied verbatim (counts unchanged).
    b.file.seek(SeekFrom::Start(0))?;
    io::copy(&mut (&b.file).take(meta_end), &mut w)?;
    // New tensor-info table with fresh, aligned offsets.
    let mut offset = 0u64;
    let mut offsets = Vec::with_capacity(plan.len());
    for (t, ty, _, _, len) in &plan {
        w.write_all(&(t.name.len() as u64).to_le_bytes())?;
        w.write_all(t.name.as_bytes())?;
        w.write_all(&(t.shape.len() as u32).to_le_bytes())?;
        for dim in &t.shape {
            w.write_all(&dim.to_le_bytes())?;
        }
        w.write_all(&ty.to_le_bytes())?;
        w.write_all(&offset.to_le_bytes())?;
        offsets.push(offset);
        offset += len.div_ceil(align) * align;
    }
    let mut pos = meta_end + plan.iter().map(|(t, ..)| info_bytes(t)).sum::<u64>();
    while pos % align != 0 {
        w.write_all(&[0])?;
        pos += 1;
    }
    // Payloads.
    for ((_, _, donor_side, off, len), rel) in plan.iter().zip(&offsets) {
        let src = if *donor_side { &mut d } else { &mut b };
        src.file.seek(SeekFrom::Start(src.data_offset + off))?;
        let copied = io::copy(&mut (&src.file).take(*len), &mut w)?;
        if copied != *len {
            return Err(invalid("short read while splicing"));
        }
        let padded = len.div_ceil(align) * align;
        w.write_all(&vec![0u8; (padded - len) as usize])?;
        debug_assert_eq!(*rel % align, 0);
    }
    w.flush()?;
    drop(w);
    let bytes = std::fs::metadata(&tmp)?.len();
    // Re-read before publishing: the result must parse as a valid GGUF.
    let check = inspect_gguf_tensors(&tmp)?;
    if check.len() != base_list.len() {
        let _ = std::fs::remove_file(&tmp);
        return Err(invalid("spliced GGUF failed re-inspection"));
    }
    std::fs::rename(&tmp, out)?;
    Ok(SpliceReport { tensors: base_list.len(), from_donor, bytes })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny GGUF v3: one string KV, tensors with given type and filled payload byte.
    fn write(path: &Path, tensors: &[(&str, u64, u32, u8)]) {
        let mut b: Vec<u8> = Vec::new();
        let s = |b: &mut Vec<u8>, v: &str| {
            b.extend((v.len() as u64).to_le_bytes());
            b.extend(v.as_bytes());
        };
        b.extend(b"GGUF");
        b.extend(3u32.to_le_bytes());
        b.extend((tensors.len() as u64).to_le_bytes());
        b.extend(1u64.to_le_bytes());
        s(&mut b, "general.architecture");
        b.extend(8u32.to_le_bytes());
        s(&mut b, "qwen35");
        let bytes_of = |elems: u64, ty: u32| if ty == 1 { elems * 2 } else { elems * 4 };
        let mut offset = 0u64;
        for (name, elems, ty, _) in tensors {
            s(&mut b, name);
            b.extend(1u32.to_le_bytes());
            b.extend(elems.to_le_bytes());
            b.extend(ty.to_le_bytes());
            b.extend(offset.to_le_bytes());
            offset += bytes_of(*elems, *ty).div_ceil(32) * 32;
        }
        while b.len() % 32 != 0 {
            b.push(0);
        }
        for (_, elems, ty, fill) in tensors {
            let n = bytes_of(*elems, *ty);
            b.extend(vec![*fill; n as usize]);
            b.extend(vec![0u8; (n.div_ceil(32) * 32 - n) as usize]);
        }
        std::fs::write(path, b).unwrap();
    }

    fn payload(path: &Path, name: &str) -> (u32, Vec<u8>) {
        let s = inspect_gguf(path).unwrap();
        let ts = inspect_gguf_tensors(path).unwrap();
        let t = ts.iter().find(|t| t.name == name).unwrap();
        let n = if t.storage_type == 1 { t.elements * 2 } else { t.elements * 4 };
        let mut f = File::open(path).unwrap();
        f.seek(SeekFrom::Start(s.data_offset + t.relative_offset)).unwrap();
        let mut buf = vec![0u8; n as usize];
        f.read_exact(&mut buf).unwrap();
        (t.storage_type, buf)
    }

    #[test]
    fn splice_takes_selected_tensors_from_donor_and_keeps_the_rest() {
        let dir = std::env::temp_dir().join(format!("ojon-splice-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (base, donor, out) = (dir.join("base.gguf"), dir.join("donor.gguf"), dir.join("out.gguf"));
        // Base: everything F32 (type 0). Donor: same tensors, F16 (type 1), other bytes.
        write(&base, &[("blk.0.attn_q.weight", 10, 0, 0xAA), ("blk.0.ssm_out.weight", 7, 0, 0xBB), ("output.weight", 3, 0, 0xCC)]);
        write(&donor, &[("blk.0.attn_q.weight", 10, 1, 0x11), ("blk.0.ssm_out.weight", 7, 1, 0x22), ("output.weight", 3, 1, 0x33)]);
        let report = splice(&base, &donor, |t| t.name.contains("ssm_out"), &out).unwrap();
        assert_eq!((report.tensors, report.from_donor), (3, 1));
        assert_eq!(payload(&out, "blk.0.ssm_out.weight"), (1, vec![0x22; 14]));
        assert_eq!(payload(&out, "blk.0.attn_q.weight"), (0, vec![0xAA; 40]));
        assert_eq!(payload(&out, "output.weight"), (0, vec![0xCC; 12]));
        // Metadata kept from base; never overwrite an existing output.
        assert_eq!(inspect_gguf(&out).unwrap().architecture.as_deref(), Some("qwen35"));
        assert!(splice(&base, &donor, |_| true, &out).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn splice_refuses_mismatched_models_and_empty_selection() {
        let dir = std::env::temp_dir().join(format!("ojon-splice-bad-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (base, donor) = (dir.join("base.gguf"), dir.join("donor.gguf"));
        write(&base, &[("a.weight", 8, 0, 1), ("b.weight", 8, 0, 2)]);
        write(&donor, &[("a.weight", 9, 1, 3), ("b.weight", 8, 1, 4)]);
        assert!(splice(&base, &donor, |t| t.name == "a.weight", &dir.join("x.gguf")).is_err());
        assert!(splice(&base, &donor, |_| false, &dir.join("y.gguf")).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
