//! Recipe library: a measured recipe is reused for every model with the same
//! architecture and tensor shapes (fine-tunes, behaviour edits, re-releases of
//! one base), so only the first model of a family pays for the probe search.
//! The next one is one quantization plus an optional quick KLD check.
//!
//! Layout: `<library>/<signature>/<preset>/{recipe-receipt.json, imatrix.gguf, entry.json}`.
//! The library lives next to Ojon's settings (`OJON_RECIPE_LIBRARY` overrides).
use crate::models::gguf::{inspect_gguf_metadata, inspect_sharded_gguf};
use sha2::{Digest, Sha256};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

fn invalid(m: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, m.into())
}

pub fn library_dir() -> io::Result<PathBuf> {
    if let Some(p) = std::env::var_os("OJON_RECIPE_LIBRARY") {
        return Ok(PathBuf::from(p));
    }
    let settings = crate::settings::settings_path()?;
    Ok(settings.parent().ok_or_else(|| invalid("settings path has no parent"))?.join("recipes"))
}

/// Architecture + every tensor's name and shape (types ignored). Equal for a base model,
/// its fine-tunes and its BF16 edits; different for any other size or architecture.
pub fn signature_of(arch: &str, tensors: impl IntoIterator<Item = (String, Vec<u64>)>) -> String {
    let mut lines: Vec<String> = tensors
        .into_iter()
        .map(|(name, shape)| format!("{name}:{}", shape.iter().map(u64::to_string).collect::<Vec<_>>().join("x")))
        .collect();
    lines.sort();
    let mut h = Sha256::new();
    h.update(arch.as_bytes());
    for l in &lines {
        h.update(b"\n");
        h.update(l.as_bytes());
    }
    format!("{arch}-{}", &format!("{:x}", h.finalize())[..16])
}

pub fn model_signature(path: &Path) -> io::Result<String> {
    let (_, tensors) = inspect_sharded_gguf(path)?;
    let arch = inspect_gguf_metadata(path)?
        .str("general.architecture")
        .unwrap_or("unknown")
        .to_owned();
    Ok(signature_of(&arch, tensors.into_iter().map(|t| (t.name, t.shape))))
}

fn safe(component: &str) -> io::Result<&str> {
    if component.is_empty() || !component.chars().all(|c| c.is_ascii_alphanumeric() || "-_".contains(c)) {
        return Err(invalid(format!("unsafe library component '{component}'")));
    }
    Ok(component)
}

pub struct Entry {
    pub receipt: PathBuf,
    pub imatrix: Option<PathBuf>,
}

pub fn lookup(library: &Path, signature: &str, preset: &str) -> io::Result<Option<Entry>> {
    let dir = library.join(safe(signature)?).join(safe(preset)?);
    let receipt = dir.join("recipe-receipt.json");
    if !receipt.is_file() {
        return Ok(None);
    }
    let imatrix = Some(dir.join("imatrix.gguf")).filter(|p| p.is_file());
    Ok(Some(Entry { receipt, imatrix }))
}

/// Store a measured recipe (and the imatrix it was measured with). Never overwrites an
/// existing entry: a family's first measured recipe stays until removed on purpose.
pub fn store(library: &Path, signature: &str, preset: &str, receipt: &Path, imatrix: Option<&Path>, source_name: &str) -> io::Result<PathBuf> {
    let dir = library.join(safe(signature)?).join(safe(preset)?);
    if dir.join("recipe-receipt.json").exists() {
        return Err(io::Error::new(io::ErrorKind::AlreadyExists, format!("library already has {signature}/{preset}")));
    }
    fs::create_dir_all(&dir)?;
    fs::copy(receipt, dir.join("recipe-receipt.json"))?;
    if let Some(im) = imatrix {
        fs::copy(im, dir.join("imatrix.gguf"))?;
    }
    let entry = serde_json::json!({
        "schema": "ojon.recipe_library_entry.v1",
        "signature": signature,
        "preset": preset,
        "measured_on": source_name,
        "imatrix_included": imatrix.is_some(),
        "note": "recipe measured on this model; reused for same-architecture, same-shape models (fine-tunes, edits)"
    });
    fs::write(dir.join("entry.json"), serde_json::to_vec_pretty(&entry)?)?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(name: &str, shape: &[u64]) -> (String, Vec<u64>) {
        (name.to_owned(), shape.to_vec())
    }

    #[test]
    fn signature_ignores_order_and_types_but_not_shapes_or_arch() {
        let a = signature_of("qwen35", vec![t("blk.0.attn_q.weight", &[5120, 6144]), t("output.weight", &[5120, 248320])]);
        let b = signature_of("qwen35", vec![t("output.weight", &[5120, 248320]), t("blk.0.attn_q.weight", &[5120, 6144])]);
        assert_eq!(a, b);
        assert!(a.starts_with("qwen35-"));
        assert_ne!(a, signature_of("qwen35", vec![t("blk.0.attn_q.weight", &[4096, 6144]), t("output.weight", &[5120, 248320])]));
        assert_ne!(a, signature_of("llama", vec![t("blk.0.attn_q.weight", &[5120, 6144]), t("output.weight", &[5120, 248320])]));
    }

    #[test]
    fn store_then_lookup_and_never_overwrite() {
        let dir = std::env::temp_dir().join(format!("ojon-lib-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        let receipt = dir.join("r.json");
        fs::write(&receipt, br#"{"final":{}}"#).unwrap();
        let lib = dir.join("lib");
        assert!(lookup(&lib, "qwen35-abc", "balanced").unwrap().is_none());
        store(&lib, "qwen35-abc", "balanced", &receipt, None, "model.gguf").unwrap();
        let e = lookup(&lib, "qwen35-abc", "balanced").unwrap().unwrap();
        assert!(e.receipt.is_file() && e.imatrix.is_none());
        assert!(store(&lib, "qwen35-abc", "balanced", &receipt, None, "x").is_err());
        assert!(lookup(&lib, "../escape", "balanced").is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
