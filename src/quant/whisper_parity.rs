//! Transcription parity between an original whisper.cpp model and quantized
//! candidates: word error rate on held-out clips, gated on the DELTA between
//! candidate and original.
//!
//! Every model transcribes every clip through the public `whisper-cli`:
//!
//! `whisper-cli -m <model> -f <clip.wav> -nt -np -l <lang> -t <threads> -otxt -of <stem>`
//!
//! Flags verified in whisper.cpp `examples/cli/cli.cpp` (master, 2026-10-30):
//! `-m/--model`, `-f/--file`, `-nt/--no-timestamps`, `-np/--no-prints`,
//! `-l/--language`, `-t/--threads`, `-otxt/--output-txt`, `-of/--output-file`
//! (a path WITHOUT extension; the text lands in `<stem>.txt`, one line per
//! segment, leading whitespace stripped). `-f` also accepts non-WAV input when
//! the build has ffmpeg; Ojon only checks that the clip file exists.
//!
//! The manifest is JSONL: one `{"audio": "clip.wav", "text": "reference"}` per
//! line, relative audio paths resolved against the manifest's folder.
//!
//! Pure parts (manifest parsing, transcript parsing, gate) are unit-tested;
//! the runner needs whisper.cpp and real clips and is verified on the first
//! live clip set.
use std::fs;
use std::io::{self, Write};
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::jobs::{run_with_log, tool_command};
use crate::probe::{probe_model, ModelCategory};
use crate::recipe_run::plain;
use crate::wer::{normalise_words, word_error, WerCounts};

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, s.into())
}

pub const WHISPER_PARITY_RECEIPT: &str = "whisper-parity-receipt.json";
pub const WHISPER_PARITY_SCHEMA: &str = "ojon.whisper_parity.v1";
pub const WHISPER_PARITY_CLAIM: &str = "word error rate of each model against the manifest transcripts on these clips only, and the candidate-minus-original delta; this is a parity measurement on a user-chosen clip set, not a general accuracy, language or benchmark claim";
/// Clips accepted in one manifest.
pub const CLIP_RANGE: RangeInclusive<usize> = 1..=512;
/// Longest reference transcript Ojon accepts (bytes).
pub const MAX_TEXT_BYTES: usize = 65_536;
/// Candidates accepted in one run.
pub const MAX_CANDIDATES: usize = 16;
/// Folder under the output directory where `whisper-cli` writes `<stem>.txt`.
pub const TRANSCRIPT_DIR: &str = "transcripts";

/// One held-out clip: the audio file and the reference transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WhisperClip {
    pub audio: PathBuf,
    pub text: String,
}

#[derive(Deserialize)]
struct ManifestLine {
    audio: String,
    text: String,
}

/// Parse manifest text. No filesystem access: relative audio paths are joined
/// to `base` when given, otherwise kept as written. Blank lines are skipped;
/// a reference whose normalised word list is empty is refused because its WER
/// would be undefined.
pub fn parse_clip_manifest(text: &str, base: Option<&Path>) -> io::Result<Vec<WhisperClip>> {
    let mut clips = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        let entry: ManifestLine = serde_json::from_str(line)
            .map_err(|e| invalid(format!("manifest line {}: {e} (expected {{\"audio\": \"clip.wav\", \"text\": \"...\"}})", i + 1)))?;
        if entry.audio.trim().is_empty() {
            return Err(invalid(format!("manifest line {}: empty audio path", i + 1)));
        }
        if entry.text.len() > MAX_TEXT_BYTES {
            return Err(invalid(format!("manifest line {}: text exceeds {MAX_TEXT_BYTES} bytes", i + 1)));
        }
        if normalise_words(&entry.text).is_empty() {
            return Err(invalid(format!("manifest line {}: reference text has no words after normalisation", i + 1)));
        }
        let audio = PathBuf::from(entry.audio.trim());
        let audio = match base {
            Some(b) if audio.is_relative() => b.join(audio),
            _ => audio,
        };
        clips.push(WhisperClip { audio, text: entry.text });
    }
    if !CLIP_RANGE.contains(&clips.len()) {
        return Err(invalid(format!(
            "manifest must hold {}..={} clips, found {}",
            CLIP_RANGE.start(),
            CLIP_RANGE.end(),
            clips.len()
        )));
    }
    Ok(clips)
}

/// Read the manifest, resolve relative audio paths against its folder and
/// require every audio file to exist.
pub fn load_clip_manifest(path: &Path) -> io::Result<Vec<WhisperClip>> {
    let text = fs::read_to_string(path).map_err(|e| invalid(format!("clips manifest {}: {e}", path.display())))?;
    let base = path.parent().filter(|p| !p.as_os_str().is_empty()).map(Path::to_path_buf).unwrap_or_else(|| PathBuf::from("."));
    let clips = parse_clip_manifest(&text, Some(&base))?;
    for (i, clip) in clips.iter().enumerate() {
        if !clip.audio.is_file() {
            return Err(invalid(format!("clip {} audio missing: {}", i + 1, clip.audio.display())));
        }
    }
    Ok(clips)
}

/// `whisper-cli -otxt` output: one line per segment. Joined with spaces; the
/// WER normaliser handles case, punctuation and spacing afterwards.
pub fn parse_transcript(text: &str) -> String {
    text.lines().map(str::trim).filter(|l| !l.is_empty()).collect::<Vec<_>>().join(" ")
}

#[derive(Clone, Debug)]
pub struct WhisperParitySettings {
    /// `whisper-cli` executable.
    pub whisper_cli: PathBuf,
    /// JSONL manifest of clips.
    pub clips: PathBuf,
    /// `-l`; `auto` lets the model detect it per clip.
    pub language: String,
    /// `-t`.
    pub threads: usize,
    /// Allowed corpus WER increase of a candidate over the original.
    pub max_wer_delta: f64,
}

impl Default for WhisperParitySettings {
    fn default() -> Self {
        Self {
            whisper_cli: PathBuf::from("whisper-cli"),
            clips: PathBuf::new(),
            language: "en".to_owned(),
            threads: std::thread::available_parallelism().map(|v| v.get().min(16)).unwrap_or(4),
            max_wer_delta: 0.02,
        }
    }
}

impl WhisperParitySettings {
    pub fn validate(&self) -> io::Result<()> {
        if self.whisper_cli.as_os_str().is_empty() {
            return Err(invalid("whisper-cli executable missing"));
        }
        if self.clips.as_os_str().is_empty() {
            return Err(invalid("--clips manifest.jsonl is required"));
        }
        let lang = self.language.trim();
        if lang.is_empty() || lang.len() > 8 || !lang.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            return Err(invalid("language must be a short code such as en, de, ja or auto"));
        }
        if !(1..=256).contains(&self.threads) {
            return Err(invalid("threads must be 1..=256"));
        }
        if !self.max_wer_delta.is_finite() || !(0.0..=1.0).contains(&self.max_wer_delta) {
            return Err(invalid("max-wer-delta must be within 0..=1"));
        }
        Ok(())
    }

    /// Arguments after the executable. `out_stem` has no extension; the tool
    /// appends `.txt`.
    pub fn tool_args(&self, model: &Path, audio: &Path, out_stem: &Path) -> Vec<String> {
        vec![
            "-m".to_owned(),
            plain(model),
            "-f".to_owned(),
            plain(audio),
            "-nt".to_owned(),
            "-np".to_owned(),
            "-l".to_owned(),
            self.language.trim().to_ascii_lowercase(),
            "-t".to_owned(),
            self.threads.to_string(),
            "-otxt".to_owned(),
            "-of".to_owned(),
            plain(out_stem),
        ]
    }

    pub fn command_line(&self, model: &Path, audio: &Path, out_stem: &Path) -> String {
        let mut parts = vec![plain(&self.whisper_cli)];
        parts.extend(self.tool_args(model, audio, out_stem));
        parts.into_iter().map(|p| if p.contains(' ') { format!("\"{p}\"") } else { p }).collect::<Vec<_>>().join(" ")
    }
}

/// Candidate corpus WER minus original corpus WER, and whether it is within
/// `max_delta`. A candidate that is better than the original (negative delta)
/// passes.
pub fn wer_delta_gate(original_wer: f64, candidate_wer: f64, max_delta: f64) -> (f64, bool) {
    let delta = candidate_wer - original_wer;
    (delta, delta.is_finite() && delta <= max_delta)
}

#[derive(Clone, Debug)]
pub struct WhisperParityJob {
    pub original: PathBuf,
    pub candidates: Vec<PathBuf>,
    pub output_dir: PathBuf,
    pub settings: WhisperParitySettings,
}

/// One model in the run: role `original` or `candidate-N`.
#[derive(Clone, Debug)]
pub struct PlannedModel {
    pub role: String,
    pub model: PathBuf,
    pub bytes: u64,
    /// Folder for this model's `<clip>.txt` transcripts.
    pub transcript_dir: PathBuf,
}

#[derive(Clone, Debug)]
pub struct WhisperParityPlan {
    pub job: WhisperParityJob,
    pub clips: Vec<WhisperClip>,
    pub models: Vec<PlannedModel>,
}

fn probe_speech(path: &Path, what: &str) -> io::Result<u64> {
    let probe = probe_model(path).map_err(|e| invalid(format!("{what} {}: {e}", path.display())))?;
    if probe.category != ModelCategory::Speech {
        return Err(invalid(format!("{what} {} probes as {}, not a speech model", path.display(), probe.category.as_str())));
    }
    Ok(probe.bytes)
}

/// Validate settings, manifest and models; describe the runs. Writes nothing.
pub fn prepare_whisper_parity(job: WhisperParityJob) -> io::Result<WhisperParityPlan> {
    job.settings.validate()?;
    if job.candidates.is_empty() {
        return Err(invalid("at least one candidate model is required"));
    }
    if job.candidates.len() > MAX_CANDIDATES {
        return Err(invalid(format!("at most {MAX_CANDIDATES} candidates per run")));
    }
    let clips = load_clip_manifest(&job.settings.clips)?;
    let original_bytes = probe_speech(&job.original, "original")?;
    let original_canon = job.original.canonicalize()?;
    let mut models = vec![PlannedModel {
        role: "original".to_owned(),
        model: job.original.clone(),
        bytes: original_bytes,
        transcript_dir: job.output_dir.join(TRANSCRIPT_DIR).join("original"),
    }];
    for (i, candidate) in job.candidates.iter().enumerate() {
        let bytes = probe_speech(candidate, "candidate")?;
        if candidate.canonicalize()? == original_canon {
            return Err(invalid(format!("candidate {} is the original model", candidate.display())));
        }
        let role = format!("candidate-{}", i + 1);
        models.push(PlannedModel {
            transcript_dir: job.output_dir.join(TRANSCRIPT_DIR).join(&role),
            role,
            model: candidate.clone(),
            bytes,
        });
    }
    if job.output_dir.is_file() {
        return Err(invalid("output path is a file, not a directory"));
    }
    let receipt = job.output_dir.join(WHISPER_PARITY_RECEIPT);
    if receipt.exists() {
        return Err(invalid(format!("{} already exists; use an empty output directory", receipt.display())));
    }
    if job.output_dir.join(TRANSCRIPT_DIR).exists() {
        return Err(invalid(format!("{} already exists; use an empty output directory", job.output_dir.join(TRANSCRIPT_DIR).display())));
    }
    Ok(WhisperParityPlan { job, clips, models })
}

impl WhisperParityPlan {
    fn clip_stem(&self, model: &PlannedModel, index: usize) -> PathBuf {
        model.transcript_dir.join(format!("clip-{:03}", index + 1))
    }

    /// Every command the run will execute, in order.
    pub fn commands(&self) -> Vec<String> {
        let s = &self.job.settings;
        let mut out = Vec::with_capacity(self.models.len() * self.clips.len());
        for model in &self.models {
            for (i, clip) in self.clips.iter().enumerate() {
                out.push(s.command_line(&model.model, &clip.audio, &self.clip_stem(model, i)));
            }
        }
        out
    }

    fn settings_json(&self) -> Value {
        let s = &self.job.settings;
        json!({
            "whisper_cli": plain(&s.whisper_cli),
            "clips_manifest": plain(&s.clips),
            "clips": self.clips.len(),
            "language": s.language.trim().to_ascii_lowercase(),
            "threads": s.threads,
            "max_wer_delta": s.max_wer_delta,
            "flags": "-nt -np -otxt -of <stem> (whisper.cpp examples/cli)",
        })
    }

    fn models_json(&self) -> Value {
        Value::Array(
            self.models
                .iter()
                .map(|m| json!({ "role": m.role, "model": plain(&m.model), "bytes": m.bytes, "transcript_dir": plain(&m.transcript_dir) }))
                .collect(),
        )
    }

    pub fn plan_json(&self) -> Value {
        json!({
            "schema": "ojon.whisper_parity_plan.v1",
            "original": plain(&self.job.original),
            "candidates": self.job.candidates.iter().map(|p| plain(p)).collect::<Vec<_>>(),
            "output_dir": plain(&self.job.output_dir),
            "settings": self.settings_json(),
            "models": self.models_json(),
            "clips": self.clips.iter().map(|c| json!({"audio": plain(&c.audio), "reference_words": normalise_words(&c.text).len()})).collect::<Vec<_>>(),
            "commands": self.commands(),
            "writes": "nothing until whisper-parity-run --confirm",
        })
    }
}

/// One clip transcribed by one model.
#[derive(Clone, Debug)]
pub struct ClipResult {
    pub index: usize,
    pub audio: PathBuf,
    pub transcript: PathBuf,
    pub hypothesis: String,
    pub counts: WerCounts,
    pub seconds: f64,
}

/// One model over all clips.
#[derive(Clone, Debug)]
pub struct ModelResult {
    pub role: String,
    pub model: PathBuf,
    pub clips: Vec<ClipResult>,
    pub corpus: WerCounts,
    pub seconds: f64,
}

impl ModelResult {
    pub fn corpus_wer(&self) -> f64 {
        self.corpus.wer().unwrap_or(f64::NAN)
    }

    fn as_json(&self) -> Value {
        json!({
            "role": self.role,
            "model": plain(&self.model),
            "corpus_wer": self.corpus_wer(),
            "reference_words": self.corpus.reference_words,
            "substitutions": self.corpus.substitutions,
            "insertions": self.corpus.insertions,
            "deletions": self.corpus.deletions,
            "seconds": self.seconds,
            "clips": self.clips.iter().map(|c| json!({
                "clip": c.index + 1,
                "audio": plain(&c.audio),
                "transcript": plain(&c.transcript),
                "wer": c.counts.wer(),
                "reference_words": c.counts.reference_words,
                "substitutions": c.counts.substitutions,
                "insertions": c.counts.insertions,
                "deletions": c.counts.deletions,
                "seconds": c.seconds,
            })).collect::<Vec<_>>(),
        })
    }
}

fn transcribe_all<F>(plan: &WhisperParityPlan, model: &PlannedModel, log: &mut F) -> io::Result<ModelResult>
where
    F: FnMut(&str),
{
    fs::create_dir_all(&model.transcript_dir)?;
    let s = &plan.job.settings;
    let started = Instant::now();
    let mut clips = Vec::with_capacity(plan.clips.len());
    let mut corpus = WerCounts::default();
    for (i, clip) in plan.clips.iter().enumerate() {
        let stem = plan.clip_stem(model, i);
        let transcript = stem.with_extension("txt");
        if transcript.exists() {
            return Err(invalid(format!("{} already exists", transcript.display())));
        }
        log(&format!("[whisper-parity] {} clip {}/{}: {}", model.role, i + 1, plan.clips.len(), s.command_line(&model.model, &clip.audio, &stem)));
        let clip_started = Instant::now();
        let mut cmd = tool_command(&s.whisper_cli);
        cmd.args(s.tool_args(&model.model, &clip.audio, &stem));
        run_with_log(&mut cmd, |line| log(line))?;
        let raw = fs::read_to_string(&transcript).map_err(|e| {
            io::Error::other(format!("whisper-cli did not write {} ({e}); check -otxt/-of support in this build", transcript.display()))
        })?;
        let hypothesis = parse_transcript(&raw);
        let counts = word_error(&clip.text, &hypothesis);
        corpus.add(&counts);
        clips.push(ClipResult {
            index: i,
            audio: clip.audio.clone(),
            transcript,
            hypothesis,
            counts,
            seconds: clip_started.elapsed().as_secs_f64(),
        });
    }
    Ok(ModelResult {
        role: model.role.clone(),
        model: model.model.clone(),
        clips,
        corpus,
        seconds: started.elapsed().as_secs_f64(),
    })
}

fn write_new(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut f = fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    f.write_all(bytes)
}

/// Execute the plan: transcribe every clip with every model, score, gate,
/// write `whisper-parity-receipt.json`. Returns its path.
pub fn run_whisper_parity<F>(plan: &WhisperParityPlan, mut log: F) -> io::Result<PathBuf>
where
    F: FnMut(&str),
{
    let dir = &plan.job.output_dir;
    fs::create_dir_all(dir)?;
    let receipt_path = dir.join(WHISPER_PARITY_RECEIPT);
    if receipt_path.exists() {
        return Err(invalid(format!("{} already exists", receipt_path.display())));
    }
    let mut results = Vec::with_capacity(plan.models.len());
    for model in &plan.models {
        results.push(transcribe_all(plan, model, &mut log)?);
    }
    let original = &results[0];
    let original_wer = original.corpus_wer();
    let max_delta = plan.job.settings.max_wer_delta;
    let mut all_pass = true;
    let mut gates = Vec::new();
    for candidate in &results[1..] {
        let (delta, pass) = wer_delta_gate(original_wer, candidate.corpus_wer(), max_delta);
        all_pass &= pass;
        log(&format!(
            "[whisper-parity] {} corpus WER {:.4} vs original {:.4}: delta {:+.4} -> {}",
            candidate.role,
            candidate.corpus_wer(),
            original_wer,
            delta,
            if pass { "PASS" } else { "FAIL" }
        ));
        gates.push(json!({
            "role": candidate.role,
            "model": plain(&candidate.model),
            "candidate_wer": candidate.corpus_wer(),
            "original_wer": original_wer,
            "wer_delta": delta,
            "max_wer_delta": max_delta,
            "pass": pass,
        }));
    }
    let receipt = json!({
        "schema": WHISPER_PARITY_SCHEMA,
        "created_unix": SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        "original": plain(&plan.job.original),
        "candidates": plan.job.candidates.iter().map(|p| plain(p)).collect::<Vec<_>>(),
        "output_dir": plain(dir),
        "settings": plan.settings_json(),
        "models": plan.models_json(),
        "commands": plan.commands(),
        "results": results.iter().map(ModelResult::as_json).collect::<Vec<_>>(),
        "gate": { "max_wer_delta": max_delta, "candidates": gates, "pass": all_pass },
        "claim": WHISPER_PARITY_CLAIM,
    });
    write_new(&receipt_path, &serde_json::to_vec_pretty(&receipt)?)?;
    log(&format!("[whisper-parity] {} -> {}", if all_pass { "PASS" } else { "FAIL" }, receipt_path.display()));
    Ok(receipt_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::whisper::write_fake_whisper_bin;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ojon-wparity-{tag}-{}-{}",
            std::process::id(),
            SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn manifest_parses_jsonl_and_resolves_relative_paths() {
        let absolute = std::env::temp_dir().join("b.wav");
        let text = format!(
            "{{\"audio\": \"a.wav\", \"text\": \"Hello world.\"}}

{{\"audio\": {}, \"text\": \"second clip\"}}
",
            serde_json::to_string(&absolute.to_string_lossy()).unwrap()
        );
        let clips = parse_clip_manifest(&text, Some(Path::new("clips"))).unwrap();
        assert_eq!(clips.len(), 2);
        assert_eq!(clips[0].audio, Path::new("clips").join("a.wav"));
        assert_eq!(clips[0].text, "Hello world.");
        assert_eq!(clips[1].audio, absolute, "absolute paths are kept as written");
        let unresolved = parse_clip_manifest(&text, None).unwrap();
        assert_eq!(unresolved[0].audio, Path::new("a.wav"), "no base: relative path kept");
    }

    #[test]
    fn manifest_fails_closed() {
        assert!(parse_clip_manifest("", None).is_err(), "no clips");
        assert!(parse_clip_manifest("not json", None).is_err());
        assert!(parse_clip_manifest(r#"{"audio": "a.wav"}"#, None).is_err(), "missing text");
        assert!(parse_clip_manifest(r#"{"audio": "", "text": "x"}"#, None).is_err(), "empty audio");
        assert!(parse_clip_manifest(r#"{"audio": "a.wav", "text": "..."}"#, None).is_err(), "no words");
        let too_many: String = (0..513).map(|i| format!("{{\"audio\": \"{i}.wav\", \"text\": \"w\"}}\n")).collect();
        assert!(parse_clip_manifest(&too_many, None).is_err());
    }

    #[test]
    fn transcript_segments_join_with_spaces() {
        assert_eq!(parse_transcript(" Hello there.\n\n General Kenobi!\n"), "Hello there. General Kenobi!");
        assert_eq!(parse_transcript(""), "");
    }

    #[test]
    fn settings_validate_and_emit_verified_flags() {
        let dir = temp_dir("settings");
        let manifest = dir.join("clips.jsonl");
        fs::write(&manifest, "{\"audio\": \"a.wav\", \"text\": \"x\"}\n").unwrap();
        let ok = WhisperParitySettings { clips: manifest.clone(), threads: 4, ..Default::default() };
        assert!(ok.validate().is_ok());
        assert!(WhisperParitySettings { language: String::new(), ..ok.clone() }.validate().is_err());
        assert!(WhisperParitySettings { language: "en us!".into(), ..ok.clone() }.validate().is_err());
        assert!(WhisperParitySettings { threads: 0, ..ok.clone() }.validate().is_err());
        assert!(WhisperParitySettings { max_wer_delta: 1.5, ..ok.clone() }.validate().is_err());
        assert!(WhisperParitySettings { max_wer_delta: f64::NAN, ..ok.clone() }.validate().is_err());
        assert!(WhisperParitySettings { whisper_cli: PathBuf::new(), ..ok.clone() }.validate().is_err());
        assert!(WhisperParitySettings { clips: PathBuf::new(), ..ok.clone() }.validate().is_err());

        let args = ok.tool_args(Path::new("m.bin"), Path::new("c.wav"), Path::new("out/clip-001"));
        assert_eq!(
            args,
            vec!["-m", "m.bin", "-f", "c.wav", "-nt", "-np", "-l", "en", "-t", "4", "-otxt", "-of", "out/clip-001"]
        );
        assert!(ok.command_line(Path::new("m.bin"), Path::new("c.wav"), Path::new("o")).starts_with("whisper-cli -m m.bin"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn delta_gate_is_candidate_minus_original() {
        let (delta, pass) = wer_delta_gate(0.10, 0.11, 0.02);
        assert!((delta - 0.01).abs() < 1e-12 && pass);
        let (delta, pass) = wer_delta_gate(0.10, 0.13, 0.02);
        assert!((delta - 0.03).abs() < 1e-12 && !pass);
        assert!(wer_delta_gate(0.10, 0.08, 0.0).1, "a better candidate passes a zero budget");
        assert!(wer_delta_gate(0.10, 0.10, 0.0).1, "equal passes a zero budget");
        assert!(!wer_delta_gate(0.10, f64::NAN, 0.5).1, "undefined WER never passes");
    }

    #[test]
    fn plan_checks_models_clips_and_destination() {
        let dir = temp_dir("plan");
        let original = dir.join("ggml-base.bin");
        let candidate = dir.join("ggml-base-q5_0.bin");
        write_fake_whisper_bin(&original);
        write_fake_whisper_bin(&candidate);
        let wav = dir.join("a.wav");
        fs::write(&wav, b"RIFF....WAVE").unwrap();
        let manifest = dir.join("clips.jsonl");
        fs::write(&manifest, "{\"audio\": \"a.wav\", \"text\": \"hello world\"}\n{\"audio\": \"a.wav\", \"text\": \"again\"}\n").unwrap();
        let out = dir.join("out");
        let job = WhisperParityJob {
            original: original.clone(),
            candidates: vec![candidate.clone()],
            output_dir: out.clone(),
            settings: WhisperParitySettings { clips: manifest.clone(), threads: 2, ..Default::default() },
        };
        let plan = prepare_whisper_parity(job.clone()).unwrap();
        assert_eq!(plan.clips.len(), 2);
        assert_eq!(plan.models.len(), 2);
        assert_eq!(plan.models[1].role, "candidate-1");
        assert_eq!(plan.commands().len(), 4);
        assert!(plan.commands()[3].contains("-otxt -of"));
        let j = plan.plan_json();
        assert_eq!(j["schema"], "ojon.whisper_parity_plan.v1");
        assert_eq!(j["clips"][0]["reference_words"], 2);

        // Candidate equal to the original, no candidates, missing audio, used output dir.
        assert!(prepare_whisper_parity(WhisperParityJob { candidates: vec![original.clone()], ..job.clone() }).is_err());
        assert!(prepare_whisper_parity(WhisperParityJob { candidates: vec![], ..job.clone() }).is_err());
        let bad_manifest = dir.join("bad.jsonl");
        fs::write(&bad_manifest, "{\"audio\": \"missing.wav\", \"text\": \"x\"}\n").unwrap();
        assert!(prepare_whisper_parity(WhisperParityJob {
            settings: WhisperParitySettings { clips: bad_manifest, ..job.settings.clone() },
            ..job.clone()
        })
        .is_err());
        let pickle = dir.join("pytorch_model.bin");
        fs::write(&pickle, b"PK..............").unwrap();
        assert!(prepare_whisper_parity(WhisperParityJob { candidates: vec![pickle], ..job.clone() }).is_err());
        fs::create_dir_all(&out).unwrap();
        fs::write(out.join(WHISPER_PARITY_RECEIPT), b"{}").unwrap();
        assert!(prepare_whisper_parity(job.clone()).is_err());
        fs::remove_file(out.join(WHISPER_PARITY_RECEIPT)).unwrap();
        assert!(prepare_whisper_parity(job.clone()).is_ok());
        // The runner also refuses an existing receipt without spawning anything.
        fs::write(out.join(WHISPER_PARITY_RECEIPT), b"{}").unwrap();
        let plan = WhisperParityPlan { job: job.clone(), clips: vec![], models: vec![] };
        assert!(run_whisper_parity(&plan, |_| {}).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
