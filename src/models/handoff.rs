//! Intake verification for Bunon's `bunon.model_handoff.v1` manifests.
//!
//! Bunon produces the manifest; Ojon never trusts it blindly. Intake re-hashes
//! every listed file and adds model-level checks only Ojon can do: GGUF
//! validity and architecture, MTP preservation against the base, and draft
//! tokenizer compatibility. Bunon's verification receipts are reported as
//! `source_reported`, never as independently reproduced passes.

use super::gguf::inspect_gguf;
use super::mtp;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{self, Read};
use std::path::{Path, PathBuf};

pub const MANIFEST_SCHEMA: &str = "bunon.model_handoff.v1";
pub const INTAKE_SCHEMA: &str = "ojon.handoff_intake.v1";

fn sha256_file(path: &Path) -> io::Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            return Ok(format!("{:x}", h.finalize()));
        }
        h.update(&buf[..n]);
    }
}

/// Same rule as Bunon: a directory hashes sorted (relative path, file hash) pairs.
fn hash_entry(path: &Path) -> io::Result<String> {
    if path.is_file() {
        return sha256_file(path);
    }
    let mut files = Vec::new();
    collect(path, path, &mut files)?;
    files.sort();
    let mut agg = Sha256::new();
    for rel in files {
        agg.update(rel.as_bytes());
        agg.update(sha256_file(&path.join(&rel))?.as_bytes());
    }
    Ok(format!("{:x}", agg.finalize()))
}

fn collect(root: &Path, dir: &Path, out: &mut Vec<String>) -> io::Result<()> {
    for e in fs::read_dir(dir)? {
        let p = e?.path();
        if p.is_dir() {
            collect(root, &p, out)?;
        } else {
            out.push(p.strip_prefix(root).unwrap_or(&p).to_string_lossy().replace('\\', "/"));
        }
    }
    Ok(())
}

fn resolve(dir: &Path, p: &str) -> PathBuf {
    let path = PathBuf::from(p);
    if path.is_absolute() { path } else { dir.join(path) }
}

fn is_gguf(c: &Value, path: &Path) -> bool {
    c["format"] == "gguf" || path.extension().is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
}

fn component<'a>(m: &'a Value, role: &str) -> Option<&'a Value> {
    m["components"].as_array()?.iter().find(|c| c["role"] == role)
}

pub fn verify(manifest_path: &Path) -> io::Result<Value> {
    let m: Value = serde_json::from_slice(&fs::read(manifest_path)?)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?;
    let dir = manifest_path.parent().unwrap_or(Path::new("."));
    let mut problems: Vec<String> = Vec::new();
    let mut checks: Vec<Value> = Vec::new();
    if m["schema"] != MANIFEST_SCHEMA {
        problems.push(format!("unsupported manifest schema {} (expected {MANIFEST_SCHEMA})", m["schema"]));
        return Ok(report(problems, checks, &m));
    }
    for k in ["id", "architecture", "revision"] {
        if m["base_model"][k].as_str().is_none_or(str::is_empty) {
            problems.push(format!("base_model.{k} missing"));
        }
    }
    // 1. Every file is re-hashed by Ojon.
    let mut hashed = 0;
    for key in ["tokenizer", "components", "datasets", "checkpoints"] {
        for e in m[key].as_array().into_iter().flatten() {
            let Some(p) = e["path"].as_str() else { continue };
            match hash_entry(&resolve(dir, p)) {
                Ok(s) if Some(s.as_str()) == e["sha256"].as_str() => hashed += 1,
                Ok(_) => problems.push(format!("{key} hash mismatch: {p}")),
                Err(err) => problems.push(format!("{key} unreadable: {p}: {err}")),
            }
        }
    }
    checks.push(json!({"check": "file_hashes", "files_verified": hashed}));
    // 2. GGUF components load as GGUF and report an architecture.
    for c in m["components"].as_array().into_iter().flatten() {
        let Some(p) = c["path"].as_str() else { continue };
        let path = resolve(dir, p);
        if is_gguf(c, &path) {
            match inspect_gguf(&path) {
                Ok(s) => checks.push(json!({"check": "gguf_header", "role": c["role"], "architecture": s.architecture, "tensors": s.tensor_count})),
                Err(e) => problems.push(format!("{} is not a valid GGUF: {e}", c["role"])),
            }
        }
    }
    // 3. MTP: declared heads must exist; heads in the base must survive.
    let gguf_of = |role: &str| {
        component(&m, role)
            .and_then(|c| c["path"].as_str().map(|p| (c, resolve(dir, p))))
            .filter(|(c, p)| is_gguf(c, p))
            .map(|(_, p)| p)
    };
    let model = gguf_of("merged_model").or_else(|| gguf_of("base_model"));
    if let Some(model_path) = &model {
        let inv = mtp::inventory(model_path)?;
        let declared = m["mtp"]["status"].as_str().unwrap_or("unknown");
        if matches!(declared, "present" | "trained") && inv["status"] != "present" {
            problems.push(format!("manifest declares MTP '{declared}' but model status is {}", inv["status"]));
        }
        checks.push(json!({"check": "mtp_inventory", "declared": declared, "observed": inv["status"], "layers": inv["declared_mtp_layers"]}));
        if let (Some(base), Some(_)) = (gguf_of("base_model"), gguf_of("merged_model")) {
            let pres = mtp::verify_preserved(&base, model_path)?;
            if pres["status"] == "FAIL" {
                problems.push(format!("MTP not preserved: {}", pres["problems"]));
            }
            checks.push(json!({"check": "mtp_preserved_vs_base", "status": pres["status"]}));
        }
        if let Some(draft) = gguf_of("draft_model") {
            let compat = mtp::draft_compat(model_path, &draft)?;
            if compat["status"] != "COMPATIBLE" {
                problems.push(format!("draft model incompatible: {}", compat["reasons"]));
            }
            checks.push(json!({"check": "draft_tokenizer", "status": compat["status"]}));
        }
    } else {
        checks.push(json!({"check": "mtp_inventory", "status": "NOT_RUN", "reason": "no GGUF model component (convert with Ojon first)"}));
    }
    // 4. Receipts: PASS needs evidence whose hash still matches.
    for r in m["verification"].as_array().into_iter().flatten() {
        let status = r["status"].as_str().unwrap_or("");
        if status == "PASS" {
            match (r["evidence"].as_str(), r["evidence_sha256"].as_str()) {
                (Some(ev), Some(sha)) if sha256_file(&resolve(dir, ev)).ok().as_deref() == Some(sha) => {}
                _ => problems.push(format!("PASS receipt '{}' lacks matching evidence", r["kind"].as_str().unwrap_or("?"))),
            }
        }
    }
    Ok(report(problems, checks, &m))
}

fn report(problems: Vec<String>, checks: Vec<Value>, m: &Value) -> Value {
    json!({
        "schema": INTAKE_SCHEMA,
        "status": if problems.is_empty() { "PASS" } else { "FAIL" },
        "manifest_schema": m["schema"],
        "base_model": m["base_model"],
        "training_method": m["training"]["method"],
        "checks": checks,
        "problems": problems,
        "source_reported_receipts": m["verification"],
        "note": "receipts are Bunon's claims; Ojon re-verified hashes, GGUF headers, MTP and draft tokenizer only. Quality must be re-measured with ojon task-eval-run / Porikkha.",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::mtp::tests::{write_gguf, Kv};

    fn setup(name: &str, merged_has_mtp: bool) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ojon-handoff-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        let kv = |n| vec![
            ("general.architecture", Kv::Str("qwen35")),
            ("qwen35.nextn_predict_layers", Kv::U32(n)),
            ("tokenizer.ggml.tokens", Kv::StrArr(&["a", "b"])),
        ];
        write_gguf(&d.join("base.gguf"), &kv(1), &[("token_embd.weight", 8), ("blk.1.nextn.eh_proj.weight", 8)]);
        let merged_t: Vec<(&str, u64)> = if merged_has_mtp {
            vec![("token_embd.weight", 8), ("blk.1.nextn.eh_proj.weight", 8)]
        } else {
            vec![("token_embd.weight", 8)]
        };
        write_gguf(&d.join("merged.gguf"), &kv(if merged_has_mtp { 1 } else { 0 }), &merged_t);
        fs::write(d.join("tokenizer.json"), b"{}").unwrap();
        let h = |f: &str| sha256_file(&d.join(f)).unwrap();
        let m = json!({
            "schema": MANIFEST_SCHEMA,
            "base_model": {"id": "base", "revision": "r1", "architecture": "qwen35"},
            "tokenizer": [{"path": "tokenizer.json", "sha256": h("tokenizer.json")}],
            "components": [
                {"role": "base_model", "path": "base.gguf", "format": "gguf", "sha256": h("base.gguf")},
                {"role": "merged_model", "path": "merged.gguf", "format": "gguf", "sha256": h("merged.gguf")}
            ],
            "training": {"method": "qlora_sft", "job_id": "j1"},
            "mtp": {"status": "present"},
            "verification": [{"kind": "heldout", "status": "NOT_RUN"}],
        });
        fs::write(d.join("handoff.json"), serde_json::to_vec(&m).unwrap()).unwrap();
        d
    }

    #[test]
    fn accepts_intact_handoff_with_preserved_mtp() {
        let d = setup("ok", true);
        let r = verify(&d.join("handoff.json")).unwrap();
        assert_eq!(r["status"], "PASS", "{r}");
    }

    #[test]
    fn rejects_dropped_mtp_and_tampered_files() {
        let d = setup("bad", false);
        let r = verify(&d.join("handoff.json")).unwrap();
        assert_eq!(r["status"], "FAIL");
        assert!(r["problems"].to_string().contains("MTP"));
        let d = setup("tamper", true);
        fs::write(d.join("tokenizer.json"), b"{\"x\":1}").unwrap();
        let r = verify(&d.join("handoff.json")).unwrap();
        assert!(r["problems"].to_string().contains("hash mismatch"));
    }
}
