//! Bounded, read-only GGUF container inspection.
//!
//! Reads only metadata and tensor descriptors, never tensor payloads. This is
//! a *container/catalog check*, NOT inference, tensor checksum, or quality QA.
//! v2 little-endian and v3 little-/big-endian headers are supported.

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

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
    pub big_endian: bool,
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
        let end = self.position()?.checked_add(bytes).ok_or_else(|| invalid("offset overflow"))?;
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
    fn u8(&mut self) -> io::Result<u8> { Ok(self.raw::<1>()?[0]) }
    fn u16(&mut self) -> io::Result<u16> {
        let b = self.raw::<2>()?;
        Ok(if self.big_endian { u16::from_be_bytes(b) } else { u16::from_le_bytes(b) })
    }
    fn u32(&mut self) -> io::Result<u32> {
        let b = self.raw::<4>()?;
        Ok(if self.big_endian { u32::from_be_bytes(b) } else { u32::from_le_bytes(b) })
    }
    fn u64(&mut self) -> io::Result<u64> {
        let b = self.raw::<8>()?;
        Ok(if self.big_endian { u64::from_be_bytes(b) } else { u64::from_le_bytes(b) })
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
        String::from_utf8(bytes).map(Some).map_err(|_| invalid("invalid UTF-8 GGUF string"))
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
                if count > self.size / 8 { return Err(invalid("invalid GGUF string array length")); }
                for _ in 0..count {
                    let len = self.u64()?;
                    self.skip(len)?;
                }
                return Ok(());
            }
            let width = scalar_width(elem_type)
                .ok_or_else(|| invalid("unsupported GGUF array element type"))?;
            let bytes = count.checked_mul(width).ok_or_else(|| invalid("GGUF array overflow"))?;
            return self.skip(bytes);
        }
        let width = scalar_width(ty).ok_or_else(|| invalid("unsupported GGUF metadata type"))?;
        self.skip(width)
    }
}

fn scalar_width(ty: u32) -> Option<u64> {
    match ty {
        0 | 1 | 7 => Some(1),          // unsigned/signed byte, bool
        2 | 3 => Some(2),              // uint16/int16
        4 | 5 | 6 => Some(4),          // uint32/int32/f32
        10 | 11 | 12 => Some(8),       // uint64/int64/f64
        _ => None,
    }
}
fn align_up(pos: u64, align: u64) -> io::Result<u64> {
    let padding = (align - (pos % align)) % align;
    pos.checked_add(padding).ok_or_else(|| invalid("alignment overflow"))
}

/// Inspects a local GGUF file without allocating or loading model tensors.
///
/// Verifies the header, metadata traversal, tensor descriptors, alignment and
/// basic in-file data offsets. Full payload integrity and runtime loadability
/// require the actual target runtime and are NOT established here.
pub fn inspect_gguf(path: &Path) -> io::Result<GgufSummary> {
    let file = File::open(path)?;
    if !fs::metadata(path)?.is_file() { return Err(invalid("not a regular file")); }
    let size = file.metadata()?.len();
    if size < 24 { return Err(invalid("GGUF file is shorter than its header")); }
    let mut reader = Reader { file, size, big_endian: false };
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
    let mut alignment = 32u32;
    for _ in 0..metadata_count {
        let key = reader.string(65_535)?.ok_or_else(|| invalid("oversized GGUF metadata key"))?;
        let ty = reader.u32()?;
        match (key.as_str(), ty) {
            ("general.architecture", 8) => architecture = reader.string(256)?,
            ("general.name", 8) => model_name = reader.string(4096)?,
            ("general.file_type", 4) => file_type = Some(reader.u32()?),
            ("general.alignment", 4) => alignment = reader.u32()?,
            ("split.count", 2) => split_count = Some(reader.u16()?),
            _ => reader.skip_value(ty)?,
        }
    }
    if alignment == 0 || !alignment.is_power_of_two() || alignment > 1_048_576 {
        return Err(invalid("invalid GGUF tensor alignment"));
    }
    let mut max_tensor_offset = 0u64;
    for _ in 0..tensor_count {
        reader.string(4096)?.ok_or_else(|| invalid("oversized tensor name"))?;
        let dimensions = reader.u32()?;
        if !(1..=4).contains(&dimensions) {
            return Err(invalid("unsupported tensor dimension count"));
        }
        for _ in 0..dimensions {
            if reader.u64()? == 0 { return Err(invalid("zero-sized tensor dimension")); }
        }
        reader.u32()?; // ggml storage type: backend checks its own supported types
        let offset = reader.u64()?;
        if offset % u64::from(alignment) != 0 {
            return Err(invalid("misaligned GGUF tensor offset"));
        }
        max_tensor_offset = max_tensor_offset.max(offset);
    }
    let data_offset = align_up(reader.position()?, u64::from(alignment))?;
    if tensor_count > 0 {
        let payload = data_offset.checked_add(max_tensor_offset)
            .ok_or_else(|| invalid("GGUF tensor data offset overflow"))?;
        if payload >= size { return Err(invalid("tensor payload offset outside GGUF file")); }
    } else if data_offset > size {
        return Err(invalid("missing GGUF data section"));
    }
    Ok(GgufSummary {
        version, tensor_count, metadata_count, file_bytes: size,
        architecture, model_name, file_type, data_offset, alignment, split_count, big_endian,
    })
}
