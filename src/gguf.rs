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
fn inspect_gguf_inner(path: &Path, collect: bool) -> io::Result<(GgufSummary, Vec<GgufTensor>)> {
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
    let mut tensors = Vec::new();
    for _ in 0..tensor_count {
        let name=reader.string(4096)?.ok_or_else(|| invalid("oversized tensor name"))?;
        let dimensions=reader.u32()?;
        if !(1..=4).contains(&dimensions) {return Err(invalid("unsupported tensor dimension count"));}
        let mut shape=Vec::new();
        let mut elements=1u64;
        for _ in 0..dimensions {
            let dim=reader.u64()?;
            if dim==0 {return Err(invalid("zero-sized tensor dimension"));}
            elements=elements.checked_mul(dim).ok_or_else(||invalid("tensor elements overflow"))?;
            if collect {shape.push(dim);}
        }
        let storage_type=reader.u32()?;
        let offset = reader.u64()?;
        if collect {tensors.push(GgufTensor{name,elements,shape,storage_type,relative_offset:offset});}
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
    Ok((GgufSummary {
        version, tensor_count, metadata_count, file_bytes: size,
        architecture, model_name, file_type, data_offset, alignment, split_count, big_endian,
    }, tensors))
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
    let summary=inspect_gguf(path)?;
    if summary.big_endian {return Err(invalid("big-endian imatrix F32 unsupported"));}
    let descriptors=inspect_gguf_tensors(path)?;
    let mut sums=std::collections::HashMap::new();
    let mut counts=std::collections::HashMap::new();
    let mut file=File::open(path)?;
    for tensor in descriptors {
        let (name,is_sum)=if let Some(base)=tensor.name.strip_suffix(".in_sum2"){
            (base.to_owned(),true)
        }else if let Some(base)=tensor.name.strip_suffix(".counts"){
            (base.to_owned(),false)
        }else{continue;};
        if tensor.storage_type!=0 || tensor.elements>2_000_000 {
            return Err(invalid("importance matrix requires bounded F32 tensor arrays"));
        }
        let len=tensor.elements.checked_mul(4).ok_or_else(||invalid("imatrix overflow"))?;
        let offset=summary.data_offset.checked_add(tensor.relative_offset)
            .ok_or_else(||invalid("imatrix offset overflow"))?;
        if offset.checked_add(len).is_none_or(|end|end>summary.file_bytes){
            return Err(invalid("truncated imatrix tensor"));
        }
        file.seek(SeekFrom::Start(offset))?;
        let mut bytes=[0u8;4];
        let mut total=0.0f64;
        for _ in 0..tensor.elements {
            file.read_exact(&mut bytes)?;
            let sample=f32::from_le_bytes(bytes) as f64;
            if !sample.is_finite()||sample<0.0{return Err(invalid("invalid imatrix F32 energy/count"));}
            total+=sample;
        }
        if is_sum {sums.insert(name,(total,tensor.elements));}
        else {counts.insert(name,total);}
    }
    let mut result=Vec::new();
    for (name,(sum,nval)) in sums {
        if let Some(count)=counts.get(&name) {
            if *count>0.0&&nval>0 {
                let energy=sum/(*count*nval as f64);
                if energy.is_finite()&&energy>0.0{result.push((name,energy));}
            }
        }
    }
    if result.is_empty(){return Err(invalid("no valid GGUF imatrix in_sum2/counts pairs"));}
    result.sort_by(|a,b|a.0.cmp(&b.0));
    Ok(result)
}
