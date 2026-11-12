//! Minimal streaming safetensors writer: plan every entry's size first, write the header, then
//! append payloads in the same order.

use serde_json::{json, Map, Value};

pub struct Entry {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    pub bytes: u64,
}

/// Header bytes including the 8-byte length prefix, padded with spaces to 8-byte alignment.
pub fn header(entries: &[Entry], metadata: &Map<String, Value>) -> Vec<u8> {
    let mut root = Map::new();
    if !metadata.is_empty() {
        root.insert("__metadata__".into(), Value::Object(metadata.clone()));
    }
    let mut off = 0u64;
    for e in entries {
        root.insert(e.name.clone(), json!({"dtype": e.dtype, "shape": e.shape, "data_offsets": [off, off + e.bytes]}));
        off += e.bytes;
    }
    let mut body = serde_json::to_vec(&Value::Object(root)).expect("header serializes");
    while body.len() % 8 != 0 {
        body.push(b' ');
    }
    let mut out = (body.len() as u64).to_le_bytes().to_vec();
    out.extend(body);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_aligned_and_offsets_are_contiguous() {
        let entries = vec![
            Entry { name: "a".into(), dtype: "I8", shape: vec![2, 3], bytes: 6 },
            Entry { name: "b".into(), dtype: "F32", shape: vec![2], bytes: 8 },
        ];
        let h = header(&entries, &Map::new());
        let len = u64::from_le_bytes(h[..8].try_into().unwrap()) as usize;
        assert_eq!(len % 8, 0);
        let v: Value = serde_json::from_slice(&h[8..8 + len]).unwrap();
        assert_eq!(v["b"]["data_offsets"], json!([6, 14]));
    }
}
