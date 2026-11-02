//! Independent read-only Bahon v1 Q8 pack inspector for Ojon.
//! Does not reuse Bahon code or modify Ojon's existing OJONQ001 format.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

const MAX_MANIFEST: u64 = 4 * 1024 * 1024;
const MAX_WEIGHTS: u64 = 32 * 1024 * 1024 * 1024;
const MAX_ELEMENTS: u64 = 100_000_000;

fn bad(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
fn open_regular(path: &Path) -> io::Result<File> {
    let meta = fs::symlink_metadata(path)?;
    if !meta.is_file() || meta.file_type().is_symlink() {
        return Err(bad("Bahon pack requires regular non-symlink files"));
    }
    File::open(path)
}
fn sha256(path: &Path) -> io::Result<String> {
    let mut input = open_regular(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = input.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}
fn shape(value: &Value) -> io::Result<Vec<u64>> {
    let items = value
        .as_array()
        .ok_or_else(|| bad("missing tensor shape"))?;
    if !(1..=2).contains(&items.len()) {
        return Err(bad("only rank 1 or 2 supported"));
    }
    let dims: Vec<u64> = items
        .iter()
        .map(|v| {
            v.as_u64()
                .filter(|n| *n > 0)
                .ok_or_else(|| bad("invalid tensor dimension"))
        })
        .collect::<io::Result<_>>()?;
    let count = dims
        .iter()
        .try_fold(1u64, |a, b| a.checked_mul(*b))
        .ok_or_else(|| bad("tensor element overflow"))?;
    if count > MAX_ELEMENTS {
        return Err(bad("tensor exceeds element limit"));
    }
    Ok(dims)
}
fn encoded_len(kind: &str, dims: &[u64]) -> io::Result<u64> {
    match (kind, dims) {
        ("f32", [n]) => n.checked_mul(4).ok_or_else(|| bad("F32 byte overflow")),
        ("q8_g64", [rows, cols]) => {
            let weight_count = rows
                .checked_mul(*cols)
                .ok_or_else(|| bad("Q8 weight overflow"))?;
            let groups = (cols / 64) + u64::from(cols % 64 != 0);
            let scale_bytes = rows
                .checked_mul(groups)
                .and_then(|v| v.checked_mul(4))
                .ok_or_else(|| bad("Q8 scale overflow"))?;
            weight_count
                .checked_add(scale_bytes)
                .ok_or_else(|| bad("Q8 byte overflow"))
        }
        _ => Err(bad("unsupported Bahon tensor encoding and shape")),
    }
}
fn valid_name(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 180
        && value
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'.' | b'_' | b'-'))
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BahonPackReport {
    pub weight_bytes: u64,
    pub q8_matrices: usize,
    pub f32_vectors: usize,
    pub tensors: usize,
    pub weights_sha256: String,
}
impl BahonPackReport {
    pub fn as_json(&self) -> Value {
        json!({
            "status": "PASS_CONTAINER_INTEGRITY", "format": "bahon-tensor-pack",
            "version": 1, "profile": "q8", "weight_bytes": self.weight_bytes,
            "q8_matrices": self.q8_matrices, "f32_vectors": self.f32_vectors,
            "tensors": self.tensors, "weights_sha256": self.weights_sha256,
            "scope": "independent Ojon pack integrity/encoding check, not tensor-generation or model inference"
        })
    }
}
pub fn inspect_bahon_pack(folder: &Path) -> io::Result<BahonPackReport> {
    let manifest_path = folder.join("bahon-pack.json");
    let metadata = fs::symlink_metadata(&manifest_path)?;
    if !metadata.is_file()
        || metadata.file_type().is_symlink()
        || metadata.len() == 0
        || metadata.len() > MAX_MANIFEST
    {
        return Err(bad("missing or oversized Bahon pack manifest"));
    }
    let doc: Value = serde_json::from_reader(open_regular(&manifest_path)?)?;
    if doc["format"] != "bahon-tensor-pack" || doc["version"] != 1 || doc["profile"] != "q8" {
        return Err(bad("unsupported Bahon pack format/version/profile"));
    }
    for (filename, field) in [
        ("config.json", "config_sha256"),
        ("tokenizer.json", "tokenizer_sha256"),
        ("weights.bin", "weights_sha256"),
    ] {
        let expected = doc[field].as_str().ok_or_else(|| bad("missing digest"))?;
        if expected.len() != 64
            || !expected.bytes().all(|b| b.is_ascii_hexdigit())
            || !expected.eq_ignore_ascii_case(&sha256(&folder.join(filename))?)
        {
            return Err(bad(format!("{filename} SHA-256 mismatch")));
        }
    }
    let path = folder.join("weights.bin");
    let total = fs::symlink_metadata(&path)?.len();
    if total > MAX_WEIGHTS || doc["weights_bytes"].as_u64() != Some(total) {
        return Err(bad("packed weight byte length mismatch"));
    }
    let entries = doc["tensors"]
        .as_object()
        .ok_or_else(|| bad("missing tensor index"))?;
    if entries.is_empty() || entries.len() > 100_000 {
        return Err(bad("invalid tensor count"));
    }
    let mut intervals = BTreeMap::new();
    let (mut q8, mut f32_count) = (0, 0);
    let mut file = open_regular(&path)?;
    for (name, entry) in entries {
        if !valid_name(name) {
            return Err(bad("invalid tensor name"));
        }
        let kind = entry["kind"]
            .as_str()
            .ok_or_else(|| bad("missing encoding"))?;
        let dims = shape(&entry["shape"])?;
        let offset = entry["offset"]
            .as_u64()
            .ok_or_else(|| bad("missing offset"))?;
        let length = entry["length"]
            .as_u64()
            .ok_or_else(|| bad("missing length"))?;
        if encoded_len(kind, &dims)? != length
            || offset.checked_add(length).is_none_or(|end| end > total)
        {
            return Err(bad("invalid tensor byte interval or shape"));
        }
        if intervals.insert(offset, length).is_some() {
            return Err(bad("duplicate tensor offset"));
        }
        file.seek(SeekFrom::Start(offset))?;
        if kind == "q8_g64" {
            q8 += 1;
            let weight_count = dims[0]
                .checked_mul(dims[1])
                .ok_or_else(|| bad("Q8 overflow"))?;
            let mut remaining = weight_count;
            let mut buffer = [0u8; 65536];
            while remaining > 0 {
                let n = (remaining.min(buffer.len() as u64)) as usize;
                file.read_exact(&mut buffer[..n])?;
                if buffer[..n].contains(&128) {
                    return Err(bad("invalid Q8 -128 code"));
                }
                remaining -= n as u64;
            }
            for _ in 0..(length - weight_count) / 4 {
                let mut bytes = [0u8; 4];
                file.read_exact(&mut bytes)?;
                let scale = f32::from_le_bytes(bytes);
                if !scale.is_finite() || scale <= 0.0 {
                    return Err(bad("invalid Q8 scale"));
                }
            }
        } else {
            f32_count += 1;
            for _ in 0..length / 4 {
                let mut bytes = [0u8; 4];
                file.read_exact(&mut bytes)?;
                if !f32::from_le_bytes(bytes).is_finite() {
                    return Err(bad("nonfinite F32 tensor"));
                }
            }
        }
    }
    let mut end = 0u64;
    for (offset, length) in intervals {
        if offset != end {
            return Err(bad("tensor intervals overlap or contain a gap"));
        }
        end = end
            .checked_add(length)
            .ok_or_else(|| bad("interval overflow"))?;
    }
    if end != total {
        return Err(bad("trailing weight data"));
    }
    Ok(BahonPackReport {
        weight_bytes: total,
        q8_matrices: q8,
        f32_vectors: f32_count,
        tensors: entries.len(),
        weights_sha256: doc["weights_sha256"]
            .as_str()
            .ok_or_else(|| bad("missing weights digest"))?
            .to_owned(),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Temp(std::path::PathBuf);
    impl Drop for Temp {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
    fn fixture() -> Temp {
        let path = std::env::temp_dir().join(format!(
            "ojon-bahon-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        fs::write(path.join("config.json"), b"{}").unwrap();
        fs::write(path.join("tokenizer.json"), b"{}").unwrap();
        let mut bytes = vec![1, 2, 3, 4, 5, 6]; // Q8 2x3
        bytes.extend_from_slice(&0.5f32.to_le_bytes());
        bytes.extend_from_slice(&0.25f32.to_le_bytes());
        bytes.extend_from_slice(&1.0f32.to_le_bytes()); // F32 2-vector
        bytes.extend_from_slice(&2.0f32.to_le_bytes());
        fs::write(path.join("weights.bin"), &bytes).unwrap();
        let manifest = json!({
            "format": "bahon-tensor-pack", "version": 1, "profile": "q8",
            "config_sha256": sha256(&path.join("config.json")).unwrap(),
            "tokenizer_sha256": sha256(&path.join("tokenizer.json")).unwrap(),
            "weights_sha256": sha256(&path.join("weights.bin")).unwrap(),
            "weights_bytes": bytes.len(),
            "tensors": {
                "tensor.q8": {"kind": "q8_g64", "shape": [2, 3], "offset": 0, "length": 14},
                "tensor.f32": {"kind": "f32", "shape": [2], "offset": 14, "length": 8}
            }
        });
        fs::write(path.join("bahon-pack.json"), manifest.to_string()).unwrap();
        Temp(path)
    }
    #[test]
    fn validates_independent_layout_and_rejects_tampering() {
        let tmp = fixture();
        let report = inspect_bahon_pack(&tmp.0).unwrap();
        assert_eq!(
            (report.q8_matrices, report.f32_vectors, report.weight_bytes),
            (1, 1, 22)
        );
        let weight_path = tmp.0.join("weights.bin");
        let mut bytes = fs::read(&weight_path).unwrap();
        bytes[0] = 128;
        fs::write(&weight_path, &bytes).unwrap();
        assert!(inspect_bahon_pack(&tmp.0).is_err());
        bytes[0] = 1;
        fs::write(&weight_path, &bytes).unwrap();
        let manifest_path = tmp.0.join("bahon-pack.json");
        let original = fs::read(&manifest_path).unwrap();
        let mut doc: Value = serde_json::from_slice(&original).unwrap();
        doc["tensors"]["tensor.f32"]["offset"] = json!(13);
        fs::write(&manifest_path, doc.to_string()).unwrap();
        assert!(inspect_bahon_pack(&tmp.0).is_err());
        doc["tensors"]["tensor.f32"]["offset"] = json!(14);
        doc["version"] = json!(2);
        fs::write(&manifest_path, doc.to_string()).unwrap();
        assert!(inspect_bahon_pack(&tmp.0).is_err());
        fs::write(&manifest_path, original).unwrap();
        assert!(inspect_bahon_pack(&tmp.0).is_ok());
        // Recalculate hashes to prove payload validation is not merely
        // checking the manifest checksum.
        let mut bytes = fs::read(&weight_path).unwrap();
        bytes[0] = 128;
        fs::write(&weight_path, &bytes).unwrap();
        let mut doc: Value = serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
        doc["weights_sha256"] = json!(sha256(&weight_path).unwrap());
        fs::write(&manifest_path, doc.to_string()).unwrap();
        assert!(inspect_bahon_pack(&tmp.0).is_err());
        bytes[0] = 1;
        bytes[6..10].copy_from_slice(&f32::NAN.to_le_bytes());
        fs::write(&weight_path, &bytes).unwrap();
        doc["weights_sha256"] = json!(sha256(&weight_path).unwrap());
        fs::write(&manifest_path, doc.to_string()).unwrap();
        assert!(inspect_bahon_pack(&tmp.0).is_err());
    }
}
