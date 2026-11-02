//! Read-only per-layer GGUF compatibility survey.
//! This reports exact tensor evidence, not proof that an editing strategy works.

use crate::gguf::{inspect_gguf, inspect_gguf_tensors};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::Path;

#[derive(Default)]
struct Layer {
    has_attention_output: bool,
    has_ffn_down: bool,
    linear_indicators: usize,
    expert_tensors: usize,
    all_tensors: usize,
    eligible_float_matrices: usize,
}

fn layer_number(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?
        .split('.')
        .next()?
        .parse::<usize>()
        .ok()
}

fn is_supported_projection(name: &str) -> bool {
    let Some(tail) = name.strip_prefix("blk.") else {
        return false;
    };
    let Some((layer, module)) = tail.split_once('.') else {
        return false;
    };
    layer.parse::<usize>().is_ok()
        && matches!(
            module,
            "attn_output.weight" | "ffn_down.weight" | "ffn_down_shexp.weight"
        )
}

fn text_layer_count(path: &Path) -> io::Result<usize> {
    let bytes = fs::read(path)?;
    if bytes.is_empty() || bytes.len() > 2 * 1024 * 1024 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "HF config must be 1 byte..2 MiB",
        ));
    }
    let config: Value =
        serde_json::from_slice(bytes.strip_prefix(&[0xef, 0xbb, 0xbf]).unwrap_or(&bytes))
            .map_err(io::Error::other)?;
    let count = config
        .pointer("/text_config/num_hidden_layers")
        .or_else(|| config.get("num_hidden_layers"))
        .and_then(Value::as_u64)
        .filter(|n| (1..=4096).contains(n))
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "HF config must declare num_hidden_layers",
            )
        })?;
    Ok(count as usize)
}

/// Report measured tensor-layout eligibility only; does not edit any weights.
pub fn inspect_edit_coverage(path: &Path) -> io::Result<Value> {
    inspect_edit_coverage_with_config(path, None)
}

/// An optional HF config separates declared language blocks from auxiliary
/// prediction blocks. It does not prove the function of the extra blocks.
pub fn inspect_edit_coverage_with_config(
    path: &Path,
    hf_config: Option<&Path>,
) -> io::Result<Value> {
    let summary = inspect_gguf(path)?;
    let tensors = inspect_gguf_tensors(path)?;
    let language_layers = hf_config.map(text_layer_count).transpose()?;
    let high_precision = matches!(summary.file_type, Some(0 | 1 | 32));
    let source_supported =
        !summary.big_endian && summary.split_count.unwrap_or(1) == 1 && high_precision;
    let mut layers = BTreeMap::<usize, Layer>::new();
    let mut eligible_targets = Vec::new();
    let mut auxiliary_targets = Vec::new();
    let mut uneditable_projections = Vec::new();
    let mut linear_indicators = 0usize;
    let mut experts = 0usize;
    for tensor in &tensors {
        let Some(layer_num) = layer_number(&tensor.name) else {
            continue;
        };
        let layer = layers.entry(layer_num).or_default();
        layer.all_tensors += 1;
        if tensor.name.ends_with(".attn_output.weight") {
            layer.has_attention_output = true;
        }
        if tensor.name.ends_with(".ffn_down.weight") {
            layer.has_ffn_down = true;
        }
        if tensor.name.contains(".ssm_")
            || tensor.name.contains(".linear_attn")
            || tensor.name.contains(".attn_gate")
        {
            layer.linear_indicators += 1;
            linear_indicators += 1;
        }
        if tensor.name.contains("_exps")
            || tensor.name.contains("_shexp")
            || tensor.name.contains(".experts.")
        {
            layer.expert_tensors += 1;
            experts += 1;
        }
        if is_supported_projection(&tensor.name) {
            let exact_float = matches!(tensor.storage_type, 0 | 1 | 30);
            let eligible = source_supported && tensor.shape.len() == 2 && exact_float;
            if eligible {
                if language_layers.is_some_and(|n| layer_num >= n) {
                    auxiliary_targets.push(tensor.name.as_str());
                } else {
                    layer.eligible_float_matrices += 1;
                    eligible_targets.push(tensor.name.as_str());
                }
            } else {
                uneditable_projections.push(tensor.name.as_str());
            }
        }
    }
    let per_layer = layers
        .iter()
        .map(|(n, layer)| {
            json!({"layer":n, "tensors":layer.all_tensors,
            "attention_output_present":layer.has_attention_output,
            "ffn_down_present":layer.has_ffn_down,
            "linear_path_indicators":layer.linear_indicators,
            "expert_tensors":layer.expert_tensors,
            "eligible_float_matrices":layer.eligible_float_matrices})
        })
        .collect::<Vec<_>>();
    let attention_layers = layers
        .iter()
        .filter(|(n, layer)| {
            layer.has_attention_output && language_layers.is_none_or(|count| **n < count)
        })
        .map(|(n, _)| *n)
        .collect::<Vec<_>>();
    let uncovered_layers = layers
        .iter()
        .filter(|(n, layer)| {
            layer.eligible_float_matrices == 0 && language_layers.is_none_or(|count| **n < count)
        })
        .map(|(n, _)| *n)
        .collect::<Vec<_>>();
    let auxiliary_block_indices = layers
        .keys()
        .copied()
        .filter(|n| language_layers.is_some_and(|limit| *n >= limit))
        .collect::<Vec<_>>();
    Ok(json!({
        "schema":"ojon.model_edit_coverage.v1",
        "status":"STRUCTURAL_PREFLIGHT_ONLY",
        "architecture":summary.architecture,
        "file_type":summary.file_type,
        "file_bytes":summary.file_bytes,
        "source_eligible":source_supported,
        "layer_count_detected":layers.len(),
        "declared_language_layers":language_layers,
        "auxiliary_or_out_of_range_block_indices":auxiliary_block_indices,
        "auxiliary_eligible_projection_count":auxiliary_targets.len(),
        "auxiliary_projection_names_excluded":auxiliary_targets,
        "conventional_attention_layers":attention_layers,
        "layers_without_eligible_target":uncovered_layers,
        "linear_path_indicators":linear_indicators,
        "expert_tensors":experts,
        "eligible_projection_count":eligible_targets.len(),
        "eligible_projection_names":eligible_targets,
        "ineligible_projection_names":uneditable_projections,
        "layers":per_layer,
        "model_loaded":false,
        "behaviour_effect_measured":false,
        "capability_retention_measured":false,
        "warning":"Eligible tensors are only exact shape and storage matches for the existing GGUF float-matrix editor. A supplied HF config excludes out-of-range layers from eligible counts; this is descriptive, not an editing enforcement gate. Do not infer support for linear-attention, experts, vision, MTP or behavioural effectiveness."
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn only_explicit_projection_names_are_eligible() {
        assert!(is_supported_projection("blk.19.attn_output.weight"));
        assert!(is_supported_projection("blk.3.ffn_down.weight"));
        assert!(is_supported_projection("blk.4.ffn_down_shexp.weight"));
        assert!(!is_supported_projection("blk.7.ssm_a"));
        assert!(!is_supported_projection("blk.7.attn_q.weight"));
        assert!(!is_supported_projection("blk.foo.ffn_down.weight"));
        assert!(!is_supported_projection("vision.blk.7.attn_output.weight"));
        assert!(!is_supported_projection("blk.7.ffn_down.weight.extra"));
    }
    #[test]
    fn parses_real_layer_number_only() {
        assert_eq!(layer_number("blk.31.attn_output.weight"), Some(31));
        assert_eq!(layer_number("blk.3.ssm_a"), Some(3));
        assert_eq!(layer_number("blk.foo.ssm_a"), None);
        assert_eq!(layer_number("vision.blk.3.ssm_a"), None);
    }
    #[test]
    fn hf_config_identifies_language_layer_bounds() {
        let file = std::env::temp_dir().join(format!("ojon-hf-config-{}.json", std::process::id()));
        fs::write(&file, br#"{"text_config":{"num_hidden_layers":32}}"#).unwrap();
        assert_eq!(text_layer_count(&file).unwrap(), 32);
        fs::write(&file, br#"{"num_hidden_layers":8}"#).unwrap();
        assert_eq!(text_layer_count(&file).unwrap(), 8);
        fs::write(&file, br#"{"text_config":{"num_hidden_layers":0}}"#).unwrap();
        assert!(text_layer_count(&file).is_err());
        fs::remove_file(file).unwrap();
    }
}
