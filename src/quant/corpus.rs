//! Calibration corpus builder.
//!
//! The importance matrix and every KLD measurement are only as good as the text
//! they see. Public dynamic-quant pipelines calibrate instruct models on
//! hundreds of thousands to over a million tokens of chat-style text and avoid
//! encyclopedia-only corpora, which overfit. Most users have the right material
//! lying around (their own prompts, chat exports, docs, code) but no tool that
//! turns it into one clean file. This does that: concatenate text inputs,
//! drop exact-duplicate paragraphs, strip control characters, and report what
//! came out so the plan can say whether it is enough.
//!
//! Pure text handling; no model is touched.
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, s.into())
}

/// Extensions accepted as text. JSON/JSONL chat exports are flattened to their
/// string values so role markup does not dominate the corpus.
pub const TEXT_EXTENSIONS: [&str; 8] = ["txt", "md", "markdown", "rst", "json", "jsonl", "py", "rs"];

/// Below this the plan warns; see `recipe_run::SMALL_CORPUS_BYTES`.
pub const RECOMMENDED_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Default, Clone, PartialEq)]
pub struct CorpusReport {
    pub inputs: Vec<PathBuf>,
    pub skipped: Vec<(PathBuf, String)>,
    pub paragraphs_in: usize,
    pub paragraphs_kept: usize,
    pub duplicates_dropped: usize,
    pub bytes_out: u64,
    pub approx_tokens: u64,
}

impl CorpusReport {
    pub fn as_json(&self) -> Value {
        json!({
            "schema": "ojon.corpus_build.v1",
            "inputs": self.inputs.iter().map(|p| p.to_string_lossy().to_string()).collect::<Vec<_>>(),
            "skipped": self.skipped.iter().map(|(p, why)| json!({"path": p.to_string_lossy(), "reason": why})).collect::<Vec<_>>(),
            "paragraphs_in": self.paragraphs_in,
            "paragraphs_kept": self.paragraphs_kept,
            "duplicates_dropped": self.duplicates_dropped,
            "bytes_out": self.bytes_out,
            "approx_tokens": self.approx_tokens,
            "recommended_bytes": RECOMMENDED_BYTES,
            "enough_for_calibration": self.bytes_out >= RECOMMENDED_BYTES,
            "note": "approx_tokens assumes ~4 bytes per token; chat-style, task-representative text calibrates instruct models better than encyclopedia prose",
        })
    }
}

/// Flatten JSON (or JSONL) into the human text it contains: every string value
/// that looks like prose (8+ bytes with a space), one paragraph each. Keys,
/// structure and single-word labels such as roles are dropped.
pub fn json_strings(text: &str) -> Vec<String> {
    fn walk(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::String(s) => {
                // keep prose, drop labels such as roles, ids and enum values
                let t = s.trim();
                if t.len() >= 8 && t.contains(' ') {
                    out.push(t.to_string());
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            Value::Object(o) => o.values().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    let mut out = Vec::new();
    if let Ok(v) = serde_json::from_str::<Value>(text) {
        walk(&v, &mut out);
        return out;
    }
    for line in text.lines() {
        if let Ok(v) = serde_json::from_str::<Value>(line) {
            walk(&v, &mut out);
        }
    }
    out
}

/// Split plain text into paragraphs on blank lines, trimming each and dropping
/// control characters other than tab and newline.
pub fn paragraphs(text: &str) -> Vec<String> {
    let cleaned: String = text
        .chars()
        .filter(|c| !c.is_control() || *c == '\n' || *c == '\t')
        .collect();
    cleaned
        .split("\n\n")
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .map(|p| p.to_string())
        .collect()
}

/// Deduplicate exact paragraphs (after whitespace normalisation), keeping first
/// occurrences in order.
pub fn dedup(paras: Vec<String>) -> (Vec<String>, usize) {
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    let mut dropped = 0;
    for p in paras {
        let key: String = p.split_whitespace().collect::<Vec<_>>().join(" ");
        if seen.insert(key) {
            out.push(p);
        } else {
            dropped += 1;
        }
    }
    (out, dropped)
}

fn collect_inputs(paths: &[PathBuf], report: &mut CorpusReport) -> io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for p in paths {
        if p.is_dir() {
            let mut entries: Vec<PathBuf> = fs::read_dir(p)?.filter_map(|e| e.ok().map(|e| e.path())).collect();
            entries.sort();
            for e in entries {
                if e.is_file() {
                    files.push(e);
                }
            }
        } else if p.is_file() {
            files.push(p.clone());
        } else {
            report.skipped.push((p.clone(), "not found".into()));
        }
    }
    Ok(files)
}

/// Build `output` from `inputs` (files or directories, one level). Refuses to
/// overwrite. Returns the report; the same report is written as
/// `<output>.report.json` next to the corpus.
pub fn build_corpus(inputs: &[PathBuf], output: &Path, max_file_bytes: u64) -> io::Result<CorpusReport> {
    if output.exists() {
        return Err(invalid(format!("refusing to overwrite {}", output.display())));
    }
    if output.extension().and_then(|e| e.to_str()).map(|e| e.eq_ignore_ascii_case("txt")) != Some(true) {
        return Err(invalid("output must end in .txt"));
    }
    let mut report = CorpusReport::default();
    let files = collect_inputs(inputs, &mut report)?;
    let mut all = Vec::new();
    for f in files {
        let ext = f.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).unwrap_or_default();
        if !TEXT_EXTENSIONS.contains(&ext.as_str()) {
            report.skipped.push((f, format!("extension .{ext} not accepted")));
            continue;
        }
        let len = f.metadata()?.len();
        if len == 0 || len > max_file_bytes {
            report.skipped.push((f, format!("size {len} outside 1..={max_file_bytes} bytes")));
            continue;
        }
        let text = match fs::read_to_string(&f) {
            Ok(t) => t,
            Err(e) => {
                report.skipped.push((f, format!("not UTF-8 text: {e}")));
                continue;
            }
        };
        let paras = if ext == "json" || ext == "jsonl" { json_strings(&text) } else { paragraphs(&text) };
        if paras.is_empty() {
            report.skipped.push((f, "no usable text".into()));
            continue;
        }
        report.paragraphs_in += paras.len();
        report.inputs.push(f);
        all.extend(paras);
    }
    if all.is_empty() {
        return Err(invalid("no usable text in the inputs"));
    }
    let (kept, dropped) = dedup(all);
    report.duplicates_dropped = dropped;
    report.paragraphs_kept = kept.len();
    let body = kept.join("\n\n") + "\n";
    let mut f = fs::OpenOptions::new().write(true).create_new(true).open(output)?;
    f.write_all(body.as_bytes())?;
    report.bytes_out = body.len() as u64;
    report.approx_tokens = report.bytes_out / 4;
    let report_path = output.with_extension("txt.report.json");
    fs::write(&report_path, serde_json::to_vec_pretty(&report.as_json())?)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paragraphs_split_trim_and_strip_control_chars() {
        let p = paragraphs("one\u{0007} line\n\n\n  two\tlines\nhere  \n\n");
        assert_eq!(p, vec!["one line".to_string(), "two\tlines\nhere".to_string()]);
    }

    #[test]
    fn dedup_keeps_first_and_normalises_whitespace() {
        let (out, dropped) = dedup(vec!["a  b".into(), "a b".into(), "c".into(), "a\nb".into()]);
        assert_eq!(out, vec!["a  b".to_string(), "c".to_string()]);
        assert_eq!(dropped, 2);
    }

    #[test]
    fn json_and_jsonl_flatten_to_strings_only() {
        let j = r#"{"messages":[{"role":"user","content":"please write a function"},{"role":"assistant","content":"def f(): pass  # ok"}],"n":3}"#;
        let s = json_strings(j);
        assert_eq!(s, vec!["please write a function".to_string(), "def f(): pass  # ok".to_string()]);
        let jl = "{\"text\":\"first record here\"}\n{\"text\":\"second record here\"}\nnot json\n";
        assert_eq!(json_strings(jl).len(), 2);
    }

    #[test]
    fn build_writes_corpus_and_report_and_refuses_overwrite() {
        let dir = std::env::temp_dir().join(format!("ojon-corpus-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(dir.join("in")).unwrap();
        fs::write(dir.join("in/a.md"), "Alpha paragraph.\n\nShared paragraph.\n").unwrap();
        fs::write(dir.join("in/b.txt"), "Shared paragraph.\n\nBeta paragraph.\n").unwrap();
        fs::write(dir.join("in/c.bin"), [0u8, 1, 2]).unwrap();
        fs::write(dir.join("in/d.jsonl"), "{\"content\":\"a chat turn with enough text\"}\n").unwrap();
        let out = dir.join("corpus.txt");
        let r = build_corpus(&[dir.join("in")], &out, 10 * 1024 * 1024).unwrap();
        assert_eq!(r.inputs.len(), 3);
        assert_eq!(r.skipped.len(), 1);
        assert_eq!(r.duplicates_dropped, 1);
        assert_eq!(r.paragraphs_kept, 4);
        let text = fs::read_to_string(&out).unwrap();
        assert_eq!(text.matches("Shared paragraph.").count(), 1);
        assert!(out.with_extension("txt.report.json").is_file());
        assert!(!r.as_json()["enough_for_calibration"].as_bool().unwrap());
        assert!(build_corpus(&[dir.join("in")], &out, 1024).is_err());
        assert!(build_corpus(&[dir.join("in")], &dir.join("x.gguf"), 1024).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
