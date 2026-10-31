//! OCR parity gate: character error rate (CER) of an original and its quantized
//! candidates against a manifest of pages with known text, read through
//! `llama-mtmd-cli` (model + projector + image + prompt). The gate is the
//! candidate-minus-original CER delta on these pages only; it is a parity
//! measurement, not a general OCR accuracy claim.
//!
//! Manifest: JSONL, one `{"image": "<path>", "text": "<exact text>"}` per line;
//! relative image paths are joined to the manifest's folder.
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::gguf::inspect_gguf;
use crate::jobs::tool_command;
use crate::recipe_run::plain;
use crate::wer::{char_error, normalise_chars, WerCounts};

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, s.into())
}

pub const OCR_PARITY_RECEIPT: &str = "ocr-parity-receipt.json";
pub const OCR_PARITY_SCHEMA: &str = "ojon.ocr_parity.v1";
pub const OCR_PARITY_CLAIM: &str = "character error rate of each model against the manifest texts on these pages only, and the candidate-minus-original delta; a parity measurement on a user-chosen page set, not a general OCR accuracy or benchmark claim";
pub const DEFAULT_PROMPT: &str = "Transcribe all the text in this image exactly as written. Output only the text.";
pub const MAX_PAGES: usize = 512;
pub const MAX_CANDIDATES: usize = 16;
pub const MAX_TEXT_BYTES: usize = 65_536;
pub const TRANSCRIPT_DIR: &str = "transcripts";

#[derive(Clone, Debug, PartialEq)]
pub struct OcrPage {
    pub image: PathBuf,
    pub text: String,
}

#[derive(Deserialize)]
struct ManifestLine {
    image: String,
    text: String,
}

/// Parse manifest text; no filesystem access. Blank lines are skipped; a
/// reference with no characters after normalisation is refused (CER undefined).
pub fn parse_page_manifest(text: &str, base: Option<&Path>) -> io::Result<Vec<OcrPage>> {
    let mut pages = Vec::new();
    for (n, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row: ManifestLine = serde_json::from_str(line).map_err(|e| invalid(format!("manifest line {}: {e}", n + 1)))?;
        if row.text.len() > MAX_TEXT_BYTES {
            return Err(invalid(format!("manifest line {}: text over {MAX_TEXT_BYTES} bytes", n + 1)));
        }
        if normalise_chars(&row.text).is_empty() {
            return Err(invalid(format!("manifest line {}: reference text has no characters", n + 1)));
        }
        let image = PathBuf::from(&row.image);
        let image = match (image.is_relative(), base) {
            (true, Some(b)) => b.join(image),
            _ => image,
        };
        pages.push(OcrPage { image, text: row.text });
    }
    if pages.is_empty() || pages.len() > MAX_PAGES {
        return Err(invalid(format!("manifest must list 1..={MAX_PAGES} pages")));
    }
    Ok(pages)
}

pub fn load_page_manifest(path: &Path) -> io::Result<Vec<OcrPage>> {
    let text = fs::read_to_string(path).map_err(|e| invalid(format!("manifest {}: {e}", path.display())))?;
    let pages = parse_page_manifest(&text, path.parent())?;
    for p in &pages {
        if !p.image.is_file() {
            return Err(invalid(format!("page image {} not found", p.image.display())));
        }
    }
    Ok(pages)
}

#[derive(Clone, Debug)]
pub struct OcrParitySettings {
    pub tool: PathBuf,
    pub pages: PathBuf,
    pub prompt: String,
    pub tokens: u32,
    pub gpu_layers: i32,
    pub device: Option<String>,
    pub max_cer_delta: f64,
}

impl OcrParitySettings {
    pub fn validate(&self) -> io::Result<()> {
        if self.tool.as_os_str().is_empty() {
            return Err(invalid("llama-mtmd-cli executable missing"));
        }
        if !self.pages.is_file() {
            return Err(invalid("page manifest must be a local file"));
        }
        if self.prompt.trim().is_empty() {
            return Err(invalid("prompt must not be empty"));
        }
        if !(16..=8192).contains(&self.tokens) {
            return Err(invalid("tokens must be 16..=8192"));
        }
        if !(self.max_cer_delta.is_finite() && (0.0..=1.0).contains(&self.max_cer_delta)) {
            return Err(invalid("max CER delta must be a finite number in 0..=1"));
        }
        Ok(())
    }

    /// `llama-mtmd-cli -m M --mmproj P --image I -p PROMPT -n N -ngl G --temp 0 -s 42 --offline [-dev D]`
    pub fn tool_args(&self, model: &Path, mmproj: &Path, image: &Path) -> Vec<String> {
        let mut a = vec![
            "-m".to_owned(), plain(model), "--mmproj".to_owned(), plain(mmproj), "--image".to_owned(), plain(image),
            "-p".to_owned(), self.prompt.clone(), "-n".to_owned(), self.tokens.to_string(),
            "-ngl".to_owned(), self.gpu_layers.to_string(), "--temp".to_owned(), "0".to_owned(),
            "-s".to_owned(), "42".to_owned(), "--offline".to_owned(),
        ];
        if let Some(d) = &self.device {
            a.push("-dev".to_owned());
            a.push(d.clone());
        }
        a
    }

    pub fn command_line(&self, model: &Path, mmproj: &Path, image: &Path) -> String {
        let mut parts = vec![plain(&self.tool)];
        parts.extend(self.tool_args(model, mmproj, image));
        parts.into_iter().map(|p| if p.contains(' ') { format!("\"{p}\"") } else { p }).collect::<Vec<_>>().join(" ")
    }
}

/// Candidate CER minus original CER, and whether it is within `max_delta`.
/// A candidate better than the original (negative delta) passes.
pub fn cer_delta_gate(original_cer: f64, candidate_cer: f64, max_delta: f64) -> (f64, bool) {
    let delta = candidate_cer - original_cer;
    (delta, delta <= max_delta)
}

#[derive(Clone, Debug)]
pub struct OcrModel {
    pub model: PathBuf,
    pub mmproj: PathBuf,
}

#[derive(Clone, Debug)]
pub struct OcrParityJob {
    pub original: OcrModel,
    pub candidates: Vec<OcrModel>,
    pub output_dir: PathBuf,
    pub settings: OcrParitySettings,
}

#[derive(Clone, Debug)]
pub struct PlannedOcrModel {
    pub role: String,
    pub model: PathBuf,
    pub mmproj: PathBuf,
    pub bytes: u64,
    pub transcript_dir: PathBuf,
}

#[derive(Clone, Debug)]
pub struct OcrParityPlan {
    pub job: OcrParityJob,
    pub pages: Vec<OcrPage>,
    pub models: Vec<PlannedOcrModel>,
}

fn gguf_bytes(path: &Path, what: &str) -> io::Result<u64> {
    let s = inspect_gguf(path).map_err(|e| invalid(format!("{what} {}: {e}", path.display())))?;
    if s.tensor_count == 0 {
        return Err(invalid(format!("{what} {} has no tensors", path.display())));
    }
    Ok(s.file_bytes)
}

/// Validate settings, manifest and models; describe the runs. Writes nothing.
pub fn prepare_ocr_parity(job: OcrParityJob) -> io::Result<OcrParityPlan> {
    job.settings.validate()?;
    if job.candidates.is_empty() {
        return Err(invalid("at least one candidate model is required"));
    }
    if job.candidates.len() > MAX_CANDIDATES {
        return Err(invalid(format!("at most {MAX_CANDIDATES} candidates per run")));
    }
    let pages = load_page_manifest(&job.settings.pages)?;
    let original_canon = job.original.model.canonicalize()?;
    let mut models = vec![PlannedOcrModel {
        role: "original".into(),
        model: job.original.model.clone(),
        mmproj: job.original.mmproj.clone(),
        bytes: gguf_bytes(&job.original.model, "original")?,
        transcript_dir: job.output_dir.join(TRANSCRIPT_DIR).join("original"),
    }];
    gguf_bytes(&job.original.mmproj, "original projector")?;
    for (i, c) in job.candidates.iter().enumerate() {
        if c.model.canonicalize()? == original_canon {
            return Err(invalid(format!("candidate {} is the original model", c.model.display())));
        }
        gguf_bytes(&c.mmproj, "candidate projector")?;
        models.push(PlannedOcrModel {
            role: format!("candidate-{}", i + 1),
            model: c.model.clone(),
            mmproj: c.mmproj.clone(),
            bytes: gguf_bytes(&c.model, "candidate")?,
            transcript_dir: job.output_dir.join(TRANSCRIPT_DIR).join(format!("candidate-{}", i + 1)),
        });
    }
    if job.output_dir.join(OCR_PARITY_RECEIPT).exists() {
        return Err(invalid(format!("{} already exists in the output directory", OCR_PARITY_RECEIPT)));
    }
    Ok(OcrParityPlan { job, pages, models })
}

impl OcrParityPlan {
    pub fn page_transcript(&self, model: &PlannedOcrModel, index: usize) -> PathBuf {
        model.transcript_dir.join(format!("page-{index:04}.txt"))
    }

    pub fn commands(&self) -> Vec<String> {
        let s = &self.job.settings;
        self.models
            .iter()
            .flat_map(|m| self.pages.iter().map(move |p| s.command_line(&m.model, &m.mmproj, &p.image)))
            .collect()
    }

    pub fn plan_json(&self) -> Value {
        json!({
            "schema": "ojon.ocr_parity_plan.v1",
            "tool": plain(&self.job.settings.tool),
            "pages": self.pages.len(),
            "manifest": plain(&self.job.settings.pages),
            "prompt": self.job.settings.prompt,
            "models": self.models.iter().map(|m| json!({"role": m.role, "model": plain(&m.model), "mmproj": plain(&m.mmproj), "bytes": m.bytes})).collect::<Vec<_>>(),
            "runs": self.models.len() * self.pages.len(),
            "gate": {"max_cer_delta": self.job.settings.max_cer_delta, "rule": "candidate corpus CER minus original corpus CER must be <= max_cer_delta"},
            "writes": "nothing until ocr-parity-run --confirm",
        })
    }
}

#[derive(Clone, Debug)]
pub struct PageResult {
    pub index: usize,
    pub image: PathBuf,
    pub transcript: PathBuf,
    pub hypothesis: String,
    pub counts: WerCounts,
    pub seconds: f64,
}

#[derive(Clone, Debug)]
pub struct OcrModelResult {
    pub role: String,
    pub model: PathBuf,
    pub pages: Vec<PageResult>,
    pub corpus: WerCounts,
    pub seconds: f64,
}

impl OcrModelResult {
    pub fn corpus_cer(&self) -> f64 {
        self.corpus.wer().unwrap_or(f64::NAN)
    }
}

/// The model's answer is whatever mtmd-cli printed to stdout; logs go to stderr.
pub fn parse_answer(stdout: &str) -> String {
    stdout.trim().to_owned()
}

fn read_all<F>(plan: &OcrParityPlan, model: &PlannedOcrModel, log: &mut F) -> io::Result<OcrModelResult>
where
    F: FnMut(&str),
{
    fs::create_dir_all(&model.transcript_dir)?;
    let s = &plan.job.settings;
    let started = Instant::now();
    let mut pages = Vec::with_capacity(plan.pages.len());
    let mut corpus = WerCounts::default();
    for (i, page) in plan.pages.iter().enumerate() {
        let transcript = plan.page_transcript(model, i);
        if transcript.exists() {
            return Err(invalid(format!("{} already exists", transcript.display())));
        }
        log(&format!("[ocr-parity] {} page {}/{}: {}", model.role, i + 1, plan.pages.len(), s.command_line(&model.model, &model.mmproj, &page.image)));
        let t0 = Instant::now();
        let mut cmd = tool_command(&s.tool);
        cmd.args(s.tool_args(&model.model, &model.mmproj, &page.image)).stdin(Stdio::null());
        let output = cmd.output()?;
        if !output.status.success() {
            return Err(io::Error::other(format!(
                "{} page {} failed: {}",
                model.role,
                i + 1,
                String::from_utf8_lossy(&output.stderr).chars().take(400).collect::<String>()
            )));
        }
        if output.stdout.len() > 2 * 1024 * 1024 {
            return Err(invalid("OCR output exceeded 2 MiB"));
        }
        let hypothesis = parse_answer(&String::from_utf8_lossy(&output.stdout));
        fs::write(&transcript, &hypothesis)?;
        let counts = char_error(&page.text, &hypothesis);
        corpus.add(&counts);
        pages.push(PageResult { index: i, image: page.image.clone(), transcript, hypothesis, counts, seconds: t0.elapsed().as_secs_f64() });
    }
    Ok(OcrModelResult { role: model.role.clone(), model: model.model.clone(), pages, corpus, seconds: started.elapsed().as_secs_f64() })
}

fn write_new(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut f = fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    f.write_all(bytes)
}

fn counts_json(c: &WerCounts) -> Value {
    json!({"reference_chars": c.reference_words, "substitutions": c.substitutions, "insertions": c.insertions, "deletions": c.deletions, "cer": c.wer()})
}

/// Read every page with every model, original first; write transcripts and the
/// receipt. Returns the receipt path.
pub fn run_ocr_parity<F>(plan: &OcrParityPlan, mut log: F) -> io::Result<PathBuf>
where
    F: FnMut(&str),
{
    fs::create_dir_all(&plan.job.output_dir)?;
    let receipt_path = plan.job.output_dir.join(OCR_PARITY_RECEIPT);
    if receipt_path.exists() {
        return Err(invalid(format!("{} already exists", receipt_path.display())));
    }
    let started = Instant::now();
    let mut results = Vec::with_capacity(plan.models.len());
    for m in &plan.models {
        results.push(read_all(plan, m, &mut log)?);
    }
    let original_cer = results[0].corpus_cer();
    let max_delta = plan.job.settings.max_cer_delta;
    let models: Vec<Value> = results
        .iter()
        .zip(&plan.models)
        .map(|(r, m)| {
            let (delta, pass) = if r.role == "original" { (0.0, true) } else { cer_delta_gate(original_cer, r.corpus_cer(), max_delta) };
            json!({
                "role": r.role, "model": plain(&r.model), "mmproj": plain(&m.mmproj), "bytes": m.bytes,
                "percent_of_original": 100.0 * m.bytes as f64 / plan.models[0].bytes as f64,
                "corpus": counts_json(&r.corpus), "corpus_cer": r.corpus_cer(),
                "cer_delta_vs_original": delta, "within_gate": pass, "seconds": r.seconds,
                "pages": r.pages.iter().map(|p| json!({
                    "index": p.index, "image": plain(&p.image), "transcript": plain(&p.transcript),
                    "hypothesis": p.hypothesis.chars().take(4096).collect::<String>(), "counts": counts_json(&p.counts), "seconds": p.seconds,
                })).collect::<Vec<_>>(),
            })
        })
        .collect();
    let all_pass = models.iter().all(|m| m["within_gate"].as_bool().unwrap_or(false));
    let receipt = json!({
        "schema": OCR_PARITY_SCHEMA,
        "created_unix": SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        "tool": plain(&plan.job.settings.tool),
        "manifest": plain(&plan.job.settings.pages),
        "pages": plan.pages.len(),
        "prompt": plan.job.settings.prompt,
        "settings": {"tokens": plan.job.settings.tokens, "gpu_layers": plan.job.settings.gpu_layers, "device": plan.job.settings.device, "max_cer_delta": max_delta},
        "models": models,
        "all_candidates_within_gate": all_pass,
        "seconds": started.elapsed().as_secs_f64(),
        "claim": OCR_PARITY_CLAIM,
    });
    write_new(&receipt_path, serde_json::to_string_pretty(&receipt)?.as_bytes())?;
    Ok(receipt_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn manifest_parses_joins_relative_paths_and_refuses_empty_text() {
        let text = "{\"image\": \"a.png\", \"text\": \"Hello, World 1\"}\n\n{\"image\": \"/abs/b.png\", \"text\": \"second\"}\n";
        let pages = parse_page_manifest(text, Some(Path::new("/pages"))).unwrap();
        assert_eq!(pages.len(), 2);
        assert_eq!(pages[0].image, PathBuf::from("/pages").join("a.png"));
        assert_eq!(pages[1].image, PathBuf::from("/abs/b.png"));
        let err = parse_page_manifest("{\"image\": \"x.png\", \"text\": \"   \"}", None).unwrap_err().to_string();
        assert!(err.contains("no characters"), "{err}");
        assert!(parse_page_manifest("", None).is_err());
    }

    #[test]
    fn tool_args_and_gate() {
        let s = OcrParitySettings {
            tool: PathBuf::from("llama-mtmd-cli"), pages: PathBuf::from("pages.jsonl"), prompt: "Read it".into(),
            tokens: 256, gpu_layers: 99, device: Some("Vulkan0".into()), max_cer_delta: 0.02,
        };
        let a = s.tool_args(Path::new("m.gguf"), Path::new("p.gguf"), Path::new("img.png"));
        assert!(a.windows(2).any(|w| w[0] == "--mmproj" && w[1] == "p.gguf"));
        assert!(a.windows(2).any(|w| w[0] == "--image" && w[1] == "img.png"));
        assert!(a.windows(2).any(|w| w[0] == "--temp" && w[1] == "0"));
        assert!(a.windows(2).any(|w| w[0] == "-dev" && w[1] == "Vulkan0"));
        assert!(a.contains(&"--offline".to_string()));
        let (d, ok) = cer_delta_gate(0.05, 0.06, 0.02);
        assert!((d - 0.01).abs() < 1e-9 && ok);
        assert!(!cer_delta_gate(0.05, 0.09, 0.02).1);
        assert!(cer_delta_gate(0.05, 0.03, 0.0).1, "better than the original passes");
        assert_eq!(parse_answer("  The quick brown fox\n"), "The quick brown fox");
    }
}
