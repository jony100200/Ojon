//! Multi-token-prediction (MTP / NextN) and speculative-draft compatibility.
//!
//! Ownership: Bunon trains or fine-tunes MTP heads and draft models; Ojon only
//! DETECTS, PRESERVES and PACKAGES them; KS llama.cpp / Twarok executes and
//! benchmarks speculative decoding. Ojon never creates MTP weights: a model
//! without trained MTP tensors is reported `absent`, and the runtime falls back
//! to normal decoding.
//!
//! GGUF convention (llama.cpp `llama-arch.cpp`): `<arch>.nextn_predict_layers`
//! counts the trailing blocks that carry `blk.N.nextn.*` tensors (eh_proj,
//! embed_tokens, enorm, hnorm, shared_head_head, shared_head_norm, ...).
//! Hugging Face convention: `num_nextn_predict_layers` (DeepSeek-V3, GLM-4.5,
//! MiMo) or `mtp_num_hidden_layers` (Qwen3-Next) in `config.json`, with `mtp.*`
//! / `model.mtp_layers.*` tensors or extra `model.layers.N` beyond the base depth.

use super::gguf::{ggml_type_block_info, ggml_type_name, inspect_gguf_metadata, inspect_gguf_tensors, GgufMetadata};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
use std::path::Path;

pub const INVENTORY_SCHEMA: &str = "ojon.mtp_inventory.v1";
pub const PRESERVE_SCHEMA: &str = "ojon.mtp_preservation.v1";
pub const DRAFT_SCHEMA: &str = "ojon.draft_compat.v1";

const HF_MTP_KEYS: [&str; 4] = [
    "num_nextn_predict_layers",
    "mtp_num_hidden_layers",
    "num_mtp_layers",
    "n_mtp_layers",
];

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn tensor_bytes(storage_type: u32, elements: u64) -> Option<u64> {
    let (block_elems, block_bytes) = ggml_type_block_info(storage_type)?;
    Some(elements.div_ceil(block_elems) * block_bytes)
}

/// `present` | `absent` | `declared_missing` (metadata says MTP, tensors gone)
/// | `orphan_tensors` (tensors without the layer-count declaration).
fn status(declared: u64, tensors: usize) -> &'static str {
    match (declared > 0, tensors > 0) {
        (true, true) => "present",
        (false, false) => "absent",
        (true, false) => "declared_missing",
        (false, true) => "orphan_tensors",
    }
}

/// Tokenizer identity used by both MTP packaging and draft compatibility.
pub fn gguf_tokenizer_identity(meta: &GgufMetadata) -> Value {
    let arr = |k: &str| {
        meta.arrays
            .get(k)
            .map(|d| json!({"count": d.count, "sha256": d.sha256}))
            .unwrap_or(Value::Null)
    };
    json!({
        "model": meta.str("tokenizer.ggml.model"),
        "pre": meta.str("tokenizer.ggml.pre"),
        "tokens": arr("tokenizer.ggml.tokens"),
        "token_type": arr("tokenizer.ggml.token_type"),
        "merges": arr("tokenizer.ggml.merges"),
        "bos_token_id": meta.int("tokenizer.ggml.bos_token_id"),
        "eos_token_id": meta.int("tokenizer.ggml.eos_token_id"),
    })
}

pub fn inventory_gguf(path: &Path) -> io::Result<Value> {
    let meta = inspect_gguf_metadata(path)?;
    let tensors = inspect_gguf_tensors(path)?;
    let arch = meta.str("general.architecture").unwrap_or("").to_string();
    let declared = meta
        .int(&format!("{arch}.nextn_predict_layers"))
        .unwrap_or(0)
        .max(0) as u64;
    let block_count = meta.int(&format!("{arch}.block_count"));
    let mut layers = BTreeSet::new();
    let mut components = Vec::new();
    let mut mtp_bytes = 0u64;
    for t in tensors.iter().filter(|t| t.name.contains("nextn.")) {
        if let Some(layer) = t
            .name
            .strip_prefix("blk.")
            .and_then(|r| r.split('.').next())
            .and_then(|n| n.parse::<u64>().ok())
        {
            layers.insert(layer);
        }
        let bytes = tensor_bytes(t.storage_type, t.elements);
        mtp_bytes += bytes.unwrap_or(0);
        components.push(json!({
            "name": t.name,
            "shape": t.shape,
            "storage_type": ggml_type_name(t.storage_type),
            "bytes": bytes,
        }));
    }
    Ok(json!({
        "schema": INVENTORY_SCHEMA,
        "format": "gguf",
        "architecture": arch,
        "block_count": block_count,
        "declared_mtp_layers": declared,
        "mtp_layer_indices": layers,
        "mtp_tensor_count": components.len(),
        "mtp_tensor_bytes": mtp_bytes,
        "status": status(declared, components.len()),
        "components": components,
        "tokenizer": gguf_tokenizer_identity(&meta),
        "note": "inventory only; runtime loading, acceptance rate and speed are measured by KS llama.cpp/Twarok",
    }))
}

fn sha256_file(path: &Path) -> io::Result<String> {
    Ok(format!("{:x}", Sha256::digest(fs::read(path)?)))
}

pub fn inventory_hf(dir: &Path) -> io::Result<Value> {
    let config: Value = serde_json::from_slice(&fs::read(dir.join("config.json"))?)
        .map_err(|e| invalid(format!("config.json: {e}")))?;
    let cfg = if config.get("text_config").is_some() { &config["text_config"] } else { &config };
    let (key, declared) = HF_MTP_KEYS
        .iter()
        .find_map(|k| cfg[*k].as_u64().map(|v| (Some(*k), v)))
        .unwrap_or((None, 0));
    let base_layers = cfg["num_hidden_layers"].as_u64();
    let mut names: Vec<String> = Vec::new();
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        let idx: Value = serde_json::from_slice(&fs::read(&index)?)
            .map_err(|e| invalid(format!("safetensors index: {e}")))?;
        if let Some(map) = idx["weight_map"].as_object() {
            names.extend(map.keys().cloned());
        }
    } else {
        for entry in fs::read_dir(dir)? {
            let p = entry?.path();
            if p.extension().is_some_and(|e| e == "safetensors") {
                names.extend(safetensors_names(&p)?);
            }
        }
    }
    let is_mtp = |n: &str| {
        n.contains("mtp") || n.contains("nextn") || base_layers.is_some_and(|base| {
            n.strip_prefix("model.layers.")
                .and_then(|r| r.split('.').next())
                .and_then(|i| i.parse::<u64>().ok())
                .is_some_and(|i| i >= base)
        })
    };
    let mtp: Vec<&String> = names.iter().filter(|n| is_mtp(n)).collect();
    let tok = dir.join("tokenizer.json");
    Ok(json!({
        "schema": INVENTORY_SCHEMA,
        "format": "hf",
        "architecture": cfg["model_type"].as_str().or(config["model_type"].as_str()),
        "architectures": config["architectures"],
        "base_hidden_layers": base_layers,
        "declared_mtp_layers": declared,
        "declaration_key": key,
        "mtp_tensor_count": mtp.len(),
        "status": status(declared, mtp.len()),
        "components": mtp,
        "tokenizer": {"tokenizer_json_sha256": if tok.is_file() { Some(sha256_file(&tok)?) } else { None }},
        "note": "inventory only; Ojon never synthesises MTP weights",
    }))
}

/// Tensor names from a safetensors header (8-byte length + JSON), no weights read.
fn safetensors_names(path: &Path) -> io::Result<Vec<String>> {
    use std::io::Read;
    let mut f = fs::File::open(path)?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let n = u64::from_le_bytes(len);
    if n > 100 * 1024 * 1024 {
        return Err(invalid("safetensors header too large"));
    }
    let mut header = vec![0u8; n as usize];
    f.read_exact(&mut header)?;
    let v: Value = serde_json::from_slice(&header).map_err(|e| invalid(e.to_string()))?;
    Ok(v.as_object()
        .map(|m| m.keys().filter(|k| *k != "__metadata__").cloned().collect())
        .unwrap_or_default())
}

pub fn inventory(path: &Path) -> io::Result<Value> {
    if path.is_dir() {
        inventory_hf(path)
    } else {
        inventory_gguf(path)
    }
}

fn component_names(inv: &Value) -> BTreeSet<String> {
    inv["components"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| c.as_str().or(c["name"].as_str()).map(str::to_string))
        .collect()
}

/// Source vs optimised output: every source MTP layer must survive.
/// Names are compared only when both sides are GGUF (HF and GGUF names differ).
pub fn verify_preserved(source: &Path, output: &Path) -> io::Result<Value> {
    let src = inventory(source)?;
    let out = inventory(output)?;
    let src_layers = src["declared_mtp_layers"].as_u64().unwrap_or(0);
    let out_layers = out["declared_mtp_layers"].as_u64().unwrap_or(0);
    let mut problems = Vec::new();
    let status = if src["status"] == "absent" {
        "NOT_APPLICABLE"
    } else if src["status"] != "present" {
        problems.push(format!("source MTP status is {}", src["status"]));
        "FAIL"
    } else {
        if out["status"] != "present" {
            problems.push(format!("output MTP status is {}", out["status"]));
        }
        if out_layers != src_layers {
            problems.push(format!("MTP layers {src_layers} -> {out_layers}"));
        }
        if src["format"] == "gguf" && out["format"] == "gguf" {
            let missing: Vec<String> = component_names(&src)
                .difference(&component_names(&out))
                .cloned()
                .collect();
            if !missing.is_empty() {
                problems.push(format!("missing MTP tensors: {}", missing.join(", ")));
            }
            if src["tokenizer"] != out["tokenizer"] {
                problems.push("tokenizer identity changed".into());
            }
        }
        if problems.is_empty() { "PASS" } else { "FAIL" }
    };
    let storage: BTreeMap<String, Value> = out["components"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| Some((c["name"].as_str()?.to_string(), c["storage_type"].clone())))
        .collect();
    Ok(json!({
        "schema": PRESERVE_SCHEMA,
        "status": status,
        "source_mtp_layers": src_layers,
        "output_mtp_layers": out_layers,
        "output_mtp_storage_types": storage,
        "output_mtp_tensor_bytes": out["mtp_tensor_bytes"],
        "problems": problems,
    }))
}

/// Speculative draft compatibility: llama.cpp drafts must share the target's
/// vocabulary. Exact tokenizer identity is required here; anything else is
/// reported INCOMPATIBLE (the runtime may still tolerate small tail differences,
/// which must then be proven by a Twarok/KS llama.cpp benchmark).
pub fn draft_compat(target: &Path, draft: &Path) -> io::Result<Value> {
    let t = gguf_tokenizer_identity(&inspect_gguf_metadata(target)?);
    let d = gguf_tokenizer_identity(&inspect_gguf_metadata(draft)?);
    let mut reasons = Vec::new();
    if t["tokens"].is_null() || d["tokens"].is_null() {
        reasons.push("tokenizer.ggml.tokens missing".to_string());
    }
    for key in ["model", "pre", "tokens", "token_type", "merges", "bos_token_id", "eos_token_id"] {
        if t[key] != d[key] {
            reasons.push(format!("{key} differs"));
        }
    }
    Ok(json!({
        "schema": DRAFT_SCHEMA,
        "status": if reasons.is_empty() { "COMPATIBLE" } else { "INCOMPATIBLE" },
        "reasons": reasons,
        "target_tokenizer": t,
        "draft_tokenizer": d,
        "runtime_acceptance": "NOT_RUN: acceptance rate, tokens/s and memory overhead are measured by KS llama.cpp/Twarok",
    }))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

    pub enum Kv<'a> {
        Str(&'a str),
        U32(u32),
        StrArr(&'a [&'a str]),
    }

    /// Minimal GGUF v3 writer: metadata + F32 tensor descriptors + zeroed payload.
    pub fn write_gguf(path: &Path, kvs: &[(&str, Kv)], tensors: &[(&str, u64)]) {
        let mut b: Vec<u8> = Vec::new();
        let s = |b: &mut Vec<u8>, v: &str| {
            b.extend((v.len() as u64).to_le_bytes());
            b.extend(v.as_bytes());
        };
        b.extend(b"GGUF");
        b.extend(3u32.to_le_bytes());
        b.extend((tensors.len() as u64).to_le_bytes());
        b.extend((kvs.len() as u64).to_le_bytes());
        for (k, v) in kvs {
            s(&mut b, k);
            match v {
                Kv::Str(x) => { b.extend(8u32.to_le_bytes()); s(&mut b, x); }
                Kv::U32(x) => { b.extend(4u32.to_le_bytes()); b.extend(x.to_le_bytes()); }
                Kv::StrArr(xs) => {
                    b.extend(9u32.to_le_bytes());
                    b.extend(8u32.to_le_bytes());
                    b.extend((xs.len() as u64).to_le_bytes());
                    for x in *xs { s(&mut b, x); }
                }
            }
        }
        let mut offset = 0u64;
        for (name, elems) in tensors {
            s(&mut b, name);
            b.extend(1u32.to_le_bytes());
            b.extend(elems.to_le_bytes());
            b.extend(0u32.to_le_bytes()); // F32
            b.extend(offset.to_le_bytes());
            offset += (elems * 4).div_ceil(32) * 32;
        }
        while b.len() % 32 != 0 { b.push(0); }
        b.extend(vec![0u8; offset.max(32) as usize]);
        fs::File::create(path).unwrap().write_all(&b).unwrap();
    }

    fn model(path: &Path, nextn: Option<u32>, mtp_tensors: bool, vocab: &[&str]) {
        let mut kvs = vec![
            ("general.architecture", Kv::Str("glm4moe")),
            ("glm4moe.block_count", Kv::U32(3)),
            ("tokenizer.ggml.model", Kv::Str("gpt2")),
            ("tokenizer.ggml.tokens", Kv::StrArr(vocab)),
        ];
        if let Some(n) = nextn {
            kvs.push(("glm4moe.nextn_predict_layers", Kv::U32(n)));
        }
        let mut tensors = vec![("token_embd.weight", 8u64), ("blk.0.attn_q.weight", 8)];
        if mtp_tensors {
            tensors.push(("blk.2.nextn.eh_proj.weight", 8));
            tensors.push(("blk.2.nextn.shared_head_norm.weight", 4));
        }
        write_gguf(path, &kvs, &tensors);
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("ojon-mtp-{}-{name}", std::process::id()));
        let _ = fs::create_dir_all(&d);
        d
    }

    #[test]
    fn detects_present_absent_and_stripped_mtp() {
        let d = tmp("detect");
        let v = ["a", "b"];
        model(&d.join("with.gguf"), Some(1), true, &v);
        model(&d.join("plain.gguf"), None, false, &v);
        model(&d.join("stripped.gguf"), Some(1), false, &v);
        let inv = inventory(&d.join("with.gguf")).unwrap();
        assert_eq!(inv["status"], "present");
        assert_eq!(inv["mtp_tensor_count"], 2);
        assert_eq!(inv["mtp_layer_indices"][0], 2);
        assert_eq!(inventory(&d.join("plain.gguf")).unwrap()["status"], "absent");
        assert_eq!(inventory(&d.join("stripped.gguf")).unwrap()["status"], "declared_missing");
    }

    #[test]
    fn preservation_fails_when_quant_drops_heads_and_skips_plain_models() {
        let d = tmp("preserve");
        let v = ["a", "b"];
        model(&d.join("src.gguf"), Some(1), true, &v);
        model(&d.join("good.gguf"), Some(1), true, &v);
        model(&d.join("bad.gguf"), Some(1), false, &v);
        model(&d.join("plain.gguf"), None, false, &v);
        assert_eq!(verify_preserved(&d.join("src.gguf"), &d.join("good.gguf")).unwrap()["status"], "PASS");
        let bad = verify_preserved(&d.join("src.gguf"), &d.join("bad.gguf")).unwrap();
        assert_eq!(bad["status"], "FAIL");
        assert!(bad["problems"].to_string().contains("declared_missing"));
        assert_eq!(verify_preserved(&d.join("plain.gguf"), &d.join("plain.gguf")).unwrap()["status"], "NOT_APPLICABLE");
    }

    #[test]
    fn draft_requires_identical_tokenizer() {
        let d = tmp("draft");
        model(&d.join("target.gguf"), None, false, &["a", "b", "c"]);
        model(&d.join("same.gguf"), None, false, &["a", "b", "c"]);
        model(&d.join("other.gguf"), None, false, &["a", "x", "c"]);
        assert_eq!(draft_compat(&d.join("target.gguf"), &d.join("same.gguf")).unwrap()["status"], "COMPATIBLE");
        let bad = draft_compat(&d.join("target.gguf"), &d.join("other.gguf")).unwrap();
        assert_eq!(bad["status"], "INCOMPATIBLE");
        assert!(bad["reasons"].to_string().contains("tokens differs"));
    }

    #[test]
    fn hf_inventory_reads_config_and_index() {
        let d = tmp("hf");
        fs::write(d.join("config.json"), r#"{"model_type":"deepseek_v3","num_hidden_layers":2,"num_nextn_predict_layers":1}"#).unwrap();
        fs::write(d.join("model.safetensors.index.json"), r#"{"weight_map":{"model.layers.0.mlp.w":"a","model.layers.2.eh_proj.weight":"a","model.layers.2.enorm.weight":"a"}}"#).unwrap();
        let inv = inventory(&d).unwrap();
        assert_eq!(inv["status"], "present");
        assert_eq!(inv["mtp_tensor_count"], 2);
        assert_eq!(inv["declaration_key"], "num_nextn_predict_layers");
    }
}
