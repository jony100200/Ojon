//! Measured KL divergence between a quantized candidate and its original,
//! through the public `llama-perplexity` executable.
//!
//! Two steps, both read-only on the models:
//! 1. `save_reference_logits`: original GGUF + held-out corpus ->
//!    `--save-all-logits` dump (compressed per-position log-probabilities).
//! 2. `measure_kld`: candidate GGUF + the same corpus and context ->
//!    `--kl-divergence --kl-divergence-base <dump>`; the tool prints the
//!    distribution of per-token KLD, which is parsed here.
//!
//! Mean KLD is the gate the recipe search budgets against. It is a measured
//! distance between output distributions, not a quality score: a passing KLD
//! still needs the task suite before any capability claim.
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

use crate::gguf::inspect_gguf;
use crate::jobs::{run_with_log, tool_command};

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, s.into())
}

#[derive(Clone, Debug)]
pub struct KldSettings {
    pub tool: PathBuf,
    pub corpus: PathBuf,
    pub gpu_layers: u32,
    pub context: u32,
    /// Limit the corpus to this many chunks (0 = whole corpus). Probes are many,
    /// so a bounded, identical slice per run keeps the search tractable and fair.
    pub chunks: u32,
}

impl KldSettings {
    pub fn validate(&self) -> io::Result<()> {
        if !(256..=32768).contains(&self.context) || !self.context.is_multiple_of(32) {
            return Err(invalid("context must be 256..=32768 and divisible by 32"));
        }
        if self.gpu_layers > 512 {
            return Err(invalid("GPU layers must be 0..=512"));
        }
        if self.chunks > 4096 {
            return Err(invalid("chunk limit must be 0..=4096"));
        }
        if !self.corpus.is_file() || self.corpus.metadata()?.len() < 100 {
            return Err(invalid("held-out corpus must be a local readable text file of at least 100 bytes"));
        }
        if self.tool.as_os_str().is_empty() {
            return Err(invalid("perplexity executable missing"));
        }
        Ok(())
    }

    fn common_args(&self, cmd: &mut Command, model: &Path) {
        cmd.arg("-m")
            .arg(model)
            .arg("-f")
            .arg(&self.corpus)
            .arg("-c")
            .arg(self.context.to_string())
            .arg("-ngl")
            .arg(self.gpu_layers.to_string());
        if self.chunks > 0 {
            cmd.arg("--chunks").arg(self.chunks.to_string());
        }
    }
}

/// The lines llama-perplexity prints after `--kl-divergence`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct KldStats {
    pub mean: f64,
    pub mean_uncertainty: Option<f64>,
    pub median: Option<f64>,
    pub p99: Option<f64>,
    pub max: Option<f64>,
    pub seconds: f64,
}

/// Parse one `--kl-divergence` summary line. Accepts the exact upstream
/// formats, e.g. `Mean    KLD:   0.012345 ±   0.000123`, `99.0%   KLD:   0.1`,
/// `Maximum KLD:   1.5`, `Median  KLD:   0.01`.
pub fn parse_kld_line(line: &str) -> Option<(&'static str, f64, Option<f64>)> {
    let line = line.trim();
    let (label, rest) = line.split_once("KLD:")?;
    // Ojon's log streamer prefixes lines with "[stdout] " / "[stderr] "; the tool
    // itself may prefix a timestamp. Only the last word before "KLD:" matters.
    let label = label.rsplit(char::is_whitespace).find(|w| !w.is_empty()).unwrap_or("");
    let key = match label {
        "Mean" => "mean",
        "Median" => "median",
        "Maximum" => "max",
        "99.0%" => "p99",
        _ => return None,
    };
    let mut parts = rest.split('±');
    let value = parts.next()?.trim().parse::<f64>().ok()?;
    let unc = parts.next().and_then(|u| u.trim().parse::<f64>().ok());
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    Some((key, value, unc))
}

fn absorb(stats: &mut KldStats, line: &str) {
    if let Some((key, value, unc)) = parse_kld_line(line) {
        match key {
            "mean" => {
                stats.mean = value;
                stats.mean_uncertainty = unc;
            }
            "median" => stats.median = Some(value),
            "max" => stats.max = Some(value),
            "p99" => stats.p99 = Some(value),
            _ => {}
        }
    }
}

/// Write the reference logits of `original` to `dump`. Refuses to overwrite.
pub fn save_reference_logits<F>(original: &Path, dump: &Path, settings: &KldSettings, log: &mut F) -> io::Result<PathBuf>
where
    F: FnMut(&str),
{
    settings.validate()?;
    if inspect_gguf(original)?.tensor_count == 0 {
        return Err(invalid("empty GGUF tensor catalog"));
    }
    if dump.exists() {
        return Err(invalid(format!("reference dump already exists: {}", dump.display())));
    }
    let mut cmd = tool_command(&settings.tool);
    settings.common_args(&mut cmd, original);
    cmd.arg("--save-all-logits").arg(dump);
    run_with_log(&mut cmd, |line| log(line))?;
    if !dump.is_file() || dump.metadata()?.len() < 64 {
        return Err(io::Error::other("perplexity tool did not write a reference logits dump"));
    }
    Ok(dump.to_path_buf())
}

/// Mean/median/p99/max KLD of `candidate` against `dump`.
pub fn measure_kld<F>(candidate: &Path, dump: &Path, settings: &KldSettings, log: &mut F) -> io::Result<KldStats>
where
    F: FnMut(&str),
{
    settings.validate()?;
    if inspect_gguf(candidate)?.tensor_count == 0 {
        return Err(invalid("empty GGUF tensor catalog"));
    }
    if !dump.is_file() {
        return Err(invalid("reference logits dump missing"));
    }
    let started = Instant::now();
    let mut stats = KldStats::default();
    let mut seen_mean = false;
    let mut cmd = tool_command(&settings.tool);
    settings.common_args(&mut cmd, candidate);
    cmd.arg("--kl-divergence").arg("--kl-divergence-base").arg(dump);
    run_with_log(&mut cmd, |line| {
        if parse_kld_line(line).map(|(k, _, _)| k == "mean").unwrap_or(false) {
            seen_mean = true;
        }
        absorb(&mut stats, line);
        log(line);
    })?;
    if !seen_mean {
        return Err(io::Error::other(
            "perplexity tool did not emit 'Mean KLD'; the dump may not match this corpus/context",
        ));
    }
    stats.seconds = started.elapsed().as_secs_f64();
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_upstream_summary_lines() {
        assert_eq!(parse_kld_line("Mean    KLD:   0.012345 ±   0.000123"), Some(("mean", 0.012345, Some(0.000123))));
        assert_eq!(parse_kld_line("Maximum KLD:   1.500000"), Some(("max", 1.5, None)));
        assert_eq!(parse_kld_line("99.0%   KLD:   0.100000"), Some(("p99", 0.1, None)));
        assert_eq!(parse_kld_line("Median  KLD:   0.010000"), Some(("median", 0.01, None)));
        assert_eq!(parse_kld_line("[stdout] Mean    KLD:   0.043846 ±   0.001000"), Some(("mean", 0.043846, Some(0.001))));
        assert_eq!(parse_kld_line("[stderr] 99.0%   KLD:   0.262591"), Some(("p99", 0.262591, None)));
        assert_eq!(parse_kld_line("95.0%   KLD:   0.050000"), None);
        assert_eq!(parse_kld_line("Mean    KLD:   nan"), None);
        assert_eq!(parse_kld_line("[1]4.1234,[2]4.2"), None);
    }

    #[test]
    fn absorbs_a_full_block_in_any_order() {
        let mut s = KldStats::default();
        for line in [
            "Maximum KLD:   0.900000",
            "Mean    KLD:   0.020000 ±   0.000500",
            "99.0%   KLD:   0.300000",
            "Median  KLD:   0.004000",
            "Minimum KLD:   0.000000",
        ] {
            absorb(&mut s, line);
        }
        assert_eq!(s.mean, 0.02);
        assert_eq!(s.mean_uncertainty, Some(0.0005));
        assert_eq!(s.p99, Some(0.3));
        assert_eq!(s.max, Some(0.9));
        assert_eq!(s.median, Some(0.004));
    }

    #[test]
    fn settings_validation_fails_closed() {
        let dir = std::env::temp_dir().join(format!("ojon-kld-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let corpus = dir.join("corpus.txt");
        std::fs::write(&corpus, "x".repeat(200)).unwrap();
        let ok = KldSettings { tool: PathBuf::from("llama-perplexity"), corpus: corpus.clone(), gpu_layers: 99, context: 512, chunks: 20 };
        assert!(ok.validate().is_ok());
        assert!(KldSettings { context: 500, ..ok.clone() }.validate().is_err());
        assert!(KldSettings { gpu_layers: 999, ..ok.clone() }.validate().is_err());
        assert!(KldSettings { chunks: 9999, ..ok.clone() }.validate().is_err());
        assert!(KldSettings { tool: PathBuf::new(), ..ok.clone() }.validate().is_err());
        assert!(KldSettings { corpus: dir.join("missing.txt"), ..ok }.validate().is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
