//! Per-tensor quant-type diff of two GGUF files (e.g. Ojon vs Unsloth): which
//! tensors got which ggml type, and where the bytes go per tensor class.
use crate::models::gguf::{ggml_type_block_info, ggml_type_name, GgufTensor};
use crate::quant::recipe::TensorClass;
use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct TensorDiff {
    pub name: String,
    pub class: &'static str,          // TensorClass::of_name(&name).name()
    pub a_type: Option<&'static str>, // None when the tensor is only in b
    pub b_type: Option<&'static str>, // None when the tensor is only in a
    pub a_bytes: Option<u64>,
    pub b_bytes: Option<u64>,
}

/// Bytes a tensor occupies: elements / block_elements * block_bytes (None for unknown types).
pub fn tensor_bytes(t: &GgufTensor) -> Option<u64> {
    let (block_elements, block_bytes) = ggml_type_block_info(t.storage_type)?;
    if block_elements == 0 {
        return None;
    }
    Some((t.elements / block_elements) as u64 * block_bytes)
}

/// One row per tensor name present in either list, sorted by name.
pub fn diff_tensors(a: &[GgufTensor], b: &[GgufTensor]) -> Vec<TensorDiff> {
    let mut names: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for t in a { names.insert(t.name.clone()); }
    for t in b { names.insert(t.name.clone()); }

    let mut a_map: std::collections::HashMap<String, &GgufTensor> = a.iter().map(|t| (t.name.clone(), t)).collect();
    let mut b_map: std::collections::HashMap<String, &GgufTensor> = b.iter().map(|t| (t.name.clone(), t)).collect();

    names.into_iter().map(|name| {
        let ta = a_map.remove(&name);
        let tb = b_map.remove(&name);

        let class_name = TensorClass::of_name(&name).name();

        let (a_type, a_bytes) = if let Some(t) = ta {
            (Some(ggml_type_name(t.storage_type)), tensor_bytes(t))
        } else {
            (None, None)
        };

        let (b_type, b_bytes) = if let Some(t) = tb {
            (Some(ggml_type_name(t.storage_type)), tensor_bytes(t))
        } else {
            (None, None)
        };

        TensorDiff {
            name,
            class: class_name,
            a_type,
            b_type,
            a_bytes,
            b_bytes,
        }
    }).collect()
}

/// Per class: (class name, total a bytes, total b bytes, number of tensors whose type differs).
pub fn class_summary(rows: &[TensorDiff]) -> Vec<(&'static str, u64, u64, usize)> {
    let mut map: BTreeMap<&'static str, (u64, u64, usize)> = BTreeMap::new();

    for row in rows {
        let entry = map.entry(row.class).or_insert((0, 0, 0));
        if let Some(ab) = row.a_bytes { entry.0 += ab; }
        if let Some(bb) = row.b_bytes { entry.1 += bb; }
        if row.a_type != row.b_type {
            entry.2 += 1;
        }
    }

    map.into_iter().map(|(name, (a, b, diff))| (name, a, b, diff)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_bytes() {
        let t = GgufTensor {
            name: "test".to_string(),
            elements: 10,
            shape: vec![10],
            storage_type: 0, // F32: (1, 4)
            relative_offset: 0,
        };
        assert_eq!(tensor_bytes(&t), Some(40));
    }

    #[test]
    fn test_diff_tensors() {
        let a = vec![
            GgufTensor { name: "x".to_string(), elements: 10, shape: vec![10], storage_type: 0, relative_offset: 0 },
            GgufTensor { name: "y".to_string(), elements: 10, shape: vec![10], storage_type: 0, relative_offset: 0 },
        ];
        let b = vec![
            GgufTensor { name: "y".to_string(), elements: 256, shape: vec![256], storage_type: 12, relative_offset: 0 }, // Q4_K: (256, 144)
            GgufTensor { name: "z".to_string(), elements: 10, shape: vec![10], storage_type: 0, relative_offset: 0 },
        ];

        let diff = diff_tensors(&a, &b);
        assert_eq!(diff.len(), 3);
        assert_eq!(diff[0].name, "x");
        assert_eq!(diff[0].b_type, None);
        assert_eq!(diff[1].name, "y");
        assert_eq!(diff[1].a_type, Some(ggml_type_name(0)));
        assert_eq!(diff[2].name, "z");
        assert_eq!(diff[2].a_type, None);
    }

    #[test]
    fn test_class_summary() {
        let rows = vec![
            TensorDiff { name: "x".to_string(), class: "class1", a_type: Some("f32"), b_type: Some("f32"), a_bytes: Some(10), b_bytes: Some(10), },
            TensorDiff { name: "y".to_string(), class: "class1", a_type: Some("f32"), b_type: Some("q4"), a_bytes: Some(10), b_bytes: Some(5), },
        ];
        let summary = class_summary(&rows);
        assert_eq!(summary[0].0, "class1");
        assert_eq!(summary[0].1, 20); // a bytes: 10 + 10
        assert_eq!(summary[0].2, 15); // b bytes: 10 + 5
        assert_eq!(summary[0].3, 1); // one tensor changed type
    }
}
