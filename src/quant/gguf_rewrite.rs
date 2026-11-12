//! Rewrite a single-file little-endian GGUF v3: set or replace string metadata and swap chosen
//! tensors for new payloads (e.g. restore quantized 1-D norms as exact F32). Everything else,
//! metadata order and tensor order included, is copied byte for byte; offsets are re-laid with
//! the file's alignment. The source is never modified; output goes to a new file via `.part`.

use crate::models::gguf::inspect_gguf;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

const GGUF_STRING: u32 = 8;
const GGUF_ARRAY: u32 = 9;

fn invalid(m: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, m.into())
}

pub struct Kv {
    pub key: String,
    /// Type tag followed by the raw value bytes, exactly as stored.
    pub raw: Vec<u8>,
}

pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
    /// Bytes from this tensor's offset to the next one (payload plus padding).
    pub span: u64,
}

pub struct Layout {
    pub version: u32,
    pub alignment: u64,
    pub data_offset: u64,
    pub kvs: Vec<Kv>,
    pub tensors: Vec<TensorInfo>,
}

struct Cursor<'a> {
    b: &'a [u8],
    p: usize,
}

impl Cursor<'_> {
    fn take(&mut self, n: usize) -> io::Result<&[u8]> {
        let s = self.b.get(self.p..self.p + n).ok_or_else(|| invalid("truncated GGUF header"))?;
        self.p += n;
        Ok(s)
    }
    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> io::Result<String> {
        let n = self.u64()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| invalid("GGUF string is not UTF-8"))
    }
    fn skip_value(&mut self, ty: u32) -> io::Result<()> {
        match ty {
            0 | 1 | 7 => self.take(1).map(|_| ()),
            2 | 3 => self.take(2).map(|_| ()),
            4..=6 => self.take(4).map(|_| ()),
            10..=12 => self.take(8).map(|_| ()),
            GGUF_STRING => self.string().map(|_| ()),
            GGUF_ARRAY => {
                let (et, n) = (self.u32()?, self.u64()?);
                (0..n).try_for_each(|_| self.skip_value(et))
            }
            _ => Err(invalid(format!("unknown GGUF value type {ty}"))),
        }
    }
}

/// Parse the header, metadata (raw) and tensor table of `path`.
pub fn read_layout(path: &Path) -> io::Result<Layout> {
    let summary = inspect_gguf(path)?; // validates the whole header and tensor table
    if summary.big_endian || summary.version != 3 || summary.split_count.unwrap_or(1) > 1 {
        return Err(invalid("only single-file little-endian GGUF v3 is supported"));
    }
    let mut head = vec![0u8; summary.data_offset as usize];
    File::open(path)?.read_exact(&mut head)?;
    let mut c = Cursor { b: &head, p: 4 };
    let version = c.u32()?;
    let (n_tensors, n_kv) = (c.u64()?, c.u64()?);
    let mut kvs = Vec::with_capacity(n_kv as usize);
    for _ in 0..n_kv {
        let key = c.string()?;
        let start = c.p;
        let ty = c.u32()?;
        c.skip_value(ty)?;
        kvs.push(Kv { key, raw: head[start..c.p].to_vec() });
    }
    let mut tensors = Vec::with_capacity(n_tensors as usize);
    for _ in 0..n_tensors {
        let name = c.string()?;
        let nd = c.u32()?;
        let dims = (0..nd).map(|_| c.u64()).collect::<io::Result<Vec<_>>>()?;
        let (ggml_type, offset) = (c.u32()?, c.u64()?);
        tensors.push(TensorInfo { name, dims, ggml_type, offset, span: 0 });
    }
    let payload = summary.file_bytes - summary.data_offset;
    let mut order: Vec<usize> = (0..tensors.len()).collect();
    order.sort_by_key(|&i| tensors[i].offset);
    for (k, &i) in order.iter().enumerate() {
        let next = order.get(k + 1).map_or(payload, |&j| tensors[j].offset);
        tensors[i].span = next.checked_sub(tensors[i].offset).filter(|&s| s > 0).ok_or_else(|| invalid("overlapping tensor offsets"))?;
    }
    Ok(Layout { version, alignment: u64::from(summary.alignment), data_offset: summary.data_offset, kvs, tensors })
}

/// A string KV in GGUF encoding (type tag + length + bytes).
pub fn string_kv(key: &str, value: &str) -> Kv {
    let mut raw = GGUF_STRING.to_le_bytes().to_vec();
    raw.extend((value.len() as u64).to_le_bytes());
    raw.extend(value.as_bytes());
    Kv { key: key.to_owned(), raw }
}

#[derive(Default)]
pub struct Edits {
    /// Replace the value of an existing key in place, or append it.
    pub kvs: Vec<Kv>,
    /// tensor name -> (new ggml type, new payload bytes)
    pub tensors: HashMap<String, (u32, Vec<u8>)>,
}

fn align(n: u64, a: u64) -> u64 {
    n.div_ceil(a) * a
}

/// Write `src` with `edits` applied to `out`. Returns the output size in bytes.
pub fn rewrite(src: &Path, out: &Path, edits: &Edits) -> io::Result<u64> {
    if out.exists() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("{} exists", out.display())));
    }
    let layout = read_layout(src)?;
    if let Some(missing) = edits.tensors.keys().find(|n| !layout.tensors.iter().any(|t| &&t.name == n)) {
        return Err(invalid(format!("no tensor named {missing}")));
    }
    let mut kvs: Vec<&Kv> = layout.kvs.iter().collect();
    for kv in &edits.kvs {
        match kvs.iter().position(|k| k.key == kv.key) {
            Some(i) => kvs[i] = kv,
            None => kvs.push(kv),
        }
    }
    let a = layout.alignment;
    let mut head = b"GGUF".to_vec();
    head.extend(layout.version.to_le_bytes());
    head.extend((layout.tensors.len() as u64).to_le_bytes());
    head.extend((kvs.len() as u64).to_le_bytes());
    for kv in &kvs {
        head.extend((kv.key.len() as u64).to_le_bytes());
        head.extend(kv.key.as_bytes());
        head.extend(&kv.raw);
    }
    let mut cursor = 0u64;
    let mut placed = Vec::with_capacity(layout.tensors.len());
    for t in &layout.tensors {
        let (ty, size) = match edits.tensors.get(&t.name) {
            Some((ty, bytes)) => (*ty, bytes.len() as u64),
            None => (t.ggml_type, t.span),
        };
        let off = align(cursor, a);
        cursor = off + size;
        head.extend((t.name.len() as u64).to_le_bytes());
        head.extend(t.name.as_bytes());
        head.extend((t.dims.len() as u32).to_le_bytes());
        t.dims.iter().for_each(|d| head.extend(d.to_le_bytes()));
        head.extend(ty.to_le_bytes());
        head.extend(off.to_le_bytes());
        placed.push(off);
    }
    head.resize(align(head.len() as u64, a) as usize, 0);
    let part = out.with_extension("gguf.part");
    let mut r = File::open(src)?;
    let mut w = BufWriter::with_capacity(16 << 20, File::create(&part)?);
    w.write_all(&head)?;
    let mut written = 0u64;
    for (t, &off) in layout.tensors.iter().zip(&placed) {
        w.write_all(&vec![0u8; (off - written) as usize])?;
        match edits.tensors.get(&t.name) {
            Some((_, bytes)) => {
                w.write_all(bytes)?;
                written = off + bytes.len() as u64;
            }
            None => {
                r.seek(SeekFrom::Start(layout.data_offset + t.offset))?;
                io::copy(&mut (&mut r).take(t.span), &mut w)?;
                written = off + t.span;
            }
        }
    }
    w.flush()?;
    drop(w);
    let size = fs::metadata(&part)?.len();
    if size != head.len() as u64 + written {
        return Err(invalid("output size does not match the planned layout"));
    }
    fs::rename(&part, out)?;
    Ok(size)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Minimal GGUF v3: one string KV, two F32 vectors of 8 elements, 32-byte alignment.
    fn tiny_gguf(path: &Path) {
        let mut h = b"GGUF".to_vec();
        h.extend(3u32.to_le_bytes());
        h.extend(2u64.to_le_bytes());
        h.extend(1u64.to_le_bytes());
        let kv = string_kv("general.architecture", "x");
        h.extend((kv.key.len() as u64).to_le_bytes());
        h.extend(kv.key.as_bytes());
        h.extend(&kv.raw);
        for (i, name) in ["a", "b"].iter().enumerate() {
            h.extend((name.len() as u64).to_le_bytes());
            h.extend(name.as_bytes());
            h.extend(1u32.to_le_bytes());
            h.extend(8u64.to_le_bytes());
            h.extend(0u32.to_le_bytes());
            h.extend(((i * 32) as u64).to_le_bytes());
        }
        h.resize(align(h.len() as u64, 32) as usize, 0);
        for v in 0..16 {
            h.extend((v as f32).to_le_bytes());
        }
        fs::write(path, h).unwrap();
    }

    #[test]
    fn rewrite_sets_kv_and_replaces_one_tensor() {
        let dir = std::env::temp_dir().join(format!("ojon_gguf_rewrite_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let (src, out) = (dir.join("in.gguf"), dir.join("out.gguf"));
        let _ = fs::remove_file(&out);
        tiny_gguf(&src);
        let mut edits = Edits::default();
        edits.kvs.push(string_kv("general.architecture", "wan"));
        edits.kvs.push(string_kv("config", "{}"));
        edits.tensors.insert("b".into(), (0, [9f32; 8].iter().flat_map(|v| v.to_le_bytes()).collect()));
        rewrite(&src, &out, &edits).unwrap();
        let l = read_layout(&out).unwrap();
        assert_eq!(l.kvs.iter().map(|k| k.key.as_str()).collect::<Vec<_>>(), ["general.architecture", "config"]);
        assert_eq!(l.kvs[0].raw, string_kv("general.architecture", "wan").raw);
        let data = fs::read(&out).unwrap();
        let at = |t: &TensorInfo| (l.data_offset + t.offset) as usize;
        assert_eq!(&data[at(&l.tensors[0])..at(&l.tensors[0]) + 4], &0f32.to_le_bytes());
        assert_eq!(&data[at(&l.tensors[1])..at(&l.tensors[1]) + 4], &9f32.to_le_bytes());
        assert!(rewrite(&src, &out, &edits).is_err(), "never overwrites");
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn string_kv_encodes_type_length_and_bytes() {
        let kv = string_kv("general.architecture", "wan");
        assert_eq!(&kv.raw[..4], &GGUF_STRING.to_le_bytes());
        assert_eq!(u64::from_le_bytes(kv.raw[4..12].try_into().unwrap()), 3);
        assert_eq!(&kv.raw[12..], b"wan");
        assert_eq!(align(33, 32), 64);
        assert_eq!(align(64, 32), 64);
    }
}
