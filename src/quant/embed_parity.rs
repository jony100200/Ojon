//! Embedding and reranker parity between an original GGUF and a quantized
//! candidate, through the public `llama-embedding` executable.
//!
//! Both models embed the same held-out lines (`-f <file>`, one prompt per
//! line, `--embd-output-format json`), and Ojon compares the two outputs:
//!
//! * per-text cosine between reference and candidate vectors (min, mean, p05);
//! * top-k neighbour overlap: for every text, the k nearest other texts under
//!   the reference versus under the candidate;
//! * with `--queries` and `--documents`: Recall@k and NDCG@10 of the candidate's
//!   query->document ranking against the reference's ranking;
//! * `--pooling rank` (rerankers): the tool scores every query/document pair,
//!   and the same Recall@k / NDCG@10 are computed on the score matrices, with
//!   the cosine gate applied to each query's vector of scores over documents.
//!
//! Everything is a similarity between two models' outputs on this set. There
//! is no labelled relevance, so the reference ranking stands in for truth: a
//! passing gate says the candidate agrees with the original, not that either
//! is good at retrieval.
use std::fs;
use std::io::{self, Write};
use std::ops::RangeInclusive;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::{json, Value};

use crate::gguf::inspect_gguf;
use crate::jobs::{run_with_log_to_file, tool_command};
use crate::recipe_run::plain;

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, s.into())
}

/// Lines accepted in the held-out text file.
pub const TEXT_LINES: RangeInclusive<usize> = 8..=4096;
/// Lines accepted in the query file.
pub const QUERY_LINES: RangeInclusive<usize> = 1..=1024;
/// Lines accepted in the document file.
pub const DOCUMENT_LINES: RangeInclusive<usize> = 2..=4096;
/// Reranker mode scores every query x document pair; this bounds that work.
pub const MAX_PAIRS: usize = 16384;
/// Longest prompt line (bytes) Ojon forwards to the tool.
pub const MAX_PROMPT_BYTES: usize = 16384;
/// Ranking depth of the NDCG metric.
pub const NDCG_K: usize = 10;
pub const RECEIPT_NAME: &str = "embed-parity-receipt.json";
pub const SCHEMA: &str = "ojon.embed_parity.v1";
pub const CLAIM: &str = "similarity measurement between the candidate's and the reference's embeddings or reranker scores on this held-out set; the reference ranking stands in for relevance labels, so this is not a downstream retrieval, classification or reranking benchmark result";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pooling {
    Mean,
    Last,
    Cls,
    /// Reranker head: the tool scores `query<TAB>document` pairs.
    Rank,
}

impl Pooling {
    pub fn parse(text: &str) -> Option<Self> {
        match text.trim().to_ascii_lowercase().as_str() {
            "mean" => Some(Self::Mean),
            "last" => Some(Self::Last),
            "cls" => Some(Self::Cls),
            "rank" => Some(Self::Rank),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Last => "last",
            Self::Cls => "cls",
            Self::Rank => "rank",
        }
    }
}

#[derive(Clone, Debug)]
pub struct EmbedSettings {
    /// `llama-embedding` executable.
    pub tool: PathBuf,
    /// Held-out texts, one per line. Required unless `pooling == Rank`.
    pub texts: Option<PathBuf>,
    /// Optional retrieval split; both or neither. Required for `Rank`.
    pub queries: Option<PathBuf>,
    pub documents: Option<PathBuf>,
    pub pooling: Pooling,
    /// `--embd-normalize 2` (euclidean) when true, `-1` when false. Rank
    /// pooling always disables normalisation: a one-dimensional score vector
    /// normalised to unit length would collapse to its sign.
    pub normalise: bool,
    pub gpu_layers: u32,
    /// `--fit on` instead of a fixed `-ngl`.
    pub fit: bool,
    pub context: u32,
    /// `-b`; 0 leaves it to the tool, which raises the batch to the context.
    pub batch: u32,
    /// Neighbour depth for overlap and Recall@k.
    pub top_k: usize,
}

impl Default for EmbedSettings {
    fn default() -> Self {
        Self {
            tool: PathBuf::from("llama-embedding"),
            texts: None,
            queries: None,
            documents: None,
            pooling: Pooling::Mean,
            normalise: true,
            gpu_layers: 99,
            fit: false,
            context: 512,
            batch: 0,
            top_k: 5,
        }
    }
}

/// Pass thresholds. `min_cosine` is applied to the MIN per-text cosine.
#[derive(Clone, Debug, PartialEq)]
pub struct EmbedGate {
    pub min_cosine: f64,
    pub min_recall: Option<f64>,
    pub min_ndcg: Option<f64>,
}

impl Default for EmbedGate {
    fn default() -> Self {
        Self { min_cosine: 0.99, min_recall: None, min_ndcg: None }
    }
}

impl EmbedGate {
    pub fn validate(&self, has_retrieval: bool) -> io::Result<()> {
        let unit = |v: f64, what: &str| -> io::Result<()> {
            if !(0.0..=1.0).contains(&v) || !v.is_finite() {
                return Err(invalid(format!("{what} must be within 0..=1")));
            }
            Ok(())
        };
        unit(self.min_cosine, "min-cosine")?;
        if let Some(r) = self.min_recall {
            unit(r, "min-recall")?;
        }
        if let Some(n) = self.min_ndcg {
            unit(n, "min-ndcg")?;
        }
        if !has_retrieval && (self.min_recall.is_some() || self.min_ndcg.is_some()) {
            return Err(invalid("--min-recall / --min-ndcg need --queries and --documents"));
        }
        Ok(())
    }
}

/// The held-out lines after validation (CR stripped, no empty lines).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct EmbedInputs {
    pub texts: Vec<String>,
    pub queries: Vec<String>,
    pub documents: Vec<String>,
}

impl EmbedInputs {
    pub fn has_retrieval(&self) -> bool {
        !self.queries.is_empty() && !self.documents.is_empty()
    }
    /// Reranker prompts: `query<TAB>document`, query-major.
    pub fn pairs(&self) -> Vec<String> {
        let mut out = Vec::with_capacity(self.queries.len() * self.documents.len());
        for q in &self.queries {
            for d in &self.documents {
                out.push(format!("{q}\t{d}"));
            }
        }
        out
    }
}

/// Read one prompt per line. A single trailing newline is allowed; any other
/// empty line would shift every later embedding against its text, so it fails.
pub fn load_lines(path: &Path, what: &str, lines: RangeInclusive<usize>, forbid_tab: bool) -> io::Result<Vec<String>> {
    let raw = fs::read_to_string(path).map_err(|e| invalid(format!("{what} file {}: {e}", path.display())))?;
    let mut out: Vec<String> = raw.split('\n').map(|l| l.trim_end_matches('\r').to_owned()).collect();
    if out.last().map(|l| l.is_empty()).unwrap_or(false) {
        out.pop();
    }
    for (i, line) in out.iter().enumerate() {
        if line.trim().is_empty() {
            return Err(invalid(format!("{what} line {} is empty", i + 1)));
        }
        if line.len() > MAX_PROMPT_BYTES {
            return Err(invalid(format!("{what} line {} exceeds {MAX_PROMPT_BYTES} bytes", i + 1)));
        }
        if forbid_tab && line.contains('\t') {
            return Err(invalid(format!("{what} line {} contains a TAB, which rank pooling uses as the query/document separator", i + 1)));
        }
    }
    if !lines.contains(&out.len()) {
        return Err(invalid(format!("{what} must have {}..={} lines, found {}", lines.start(), lines.end(), out.len())));
    }
    Ok(out)
}

impl EmbedSettings {
    pub fn validate(&self) -> io::Result<()> {
        self.load_inputs().map(|_| ())
    }

    /// Validate every setting and read the held-out files.
    pub fn load_inputs(&self) -> io::Result<EmbedInputs> {
        if self.tool.as_os_str().is_empty() {
            return Err(invalid("embedding executable missing"));
        }
        if !(64..=32768).contains(&self.context) || !self.context.is_multiple_of(32) {
            return Err(invalid("context must be 64..=32768 and divisible by 32"));
        }
        if self.batch != 0 && !(self.context..=65536).contains(&self.batch) {
            return Err(invalid("batch must be 0 (tool default) or context..=65536"));
        }
        if self.gpu_layers > 512 {
            return Err(invalid("GPU layers must be 0..=512"));
        }
        if !(1..=64).contains(&self.top_k) {
            return Err(invalid("top-k must be 1..=64"));
        }
        let rank = self.pooling == Pooling::Rank;
        let mut inputs = EmbedInputs::default();
        match (&self.queries, &self.documents) {
            (Some(q), Some(d)) => {
                inputs.queries = load_lines(q, "queries", QUERY_LINES, rank)?;
                inputs.documents = load_lines(d, "documents", DOCUMENT_LINES, rank)?;
                if inputs.documents.len() < self.top_k {
                    return Err(invalid(format!("documents must have at least top-k ({}) lines", self.top_k)));
                }
                if rank && inputs.queries.len() * inputs.documents.len() > MAX_PAIRS {
                    return Err(invalid(format!("rank pooling scores every query x document pair; {} x {} exceeds {MAX_PAIRS}", inputs.queries.len(), inputs.documents.len())));
                }
            }
            (None, None) => {}
            _ => return Err(invalid("--queries and --documents must be given together")),
        }
        match (&self.texts, rank) {
            (Some(_), true) => return Err(invalid("--pooling rank scores query/document pairs; drop --texts and give --queries and --documents")),
            (None, true) if !inputs.has_retrieval() => return Err(invalid("--pooling rank needs --queries and --documents")),
            (None, true) => {}
            (None, false) => return Err(invalid("--texts <held-out file> is required")),
            (Some(t), false) => {
                inputs.texts = load_lines(t, "texts", TEXT_LINES, false)?;
                if inputs.texts.len() <= self.top_k {
                    return Err(invalid(format!("texts must have more than top-k ({}) lines for neighbour overlap", self.top_k)));
                }
            }
        }
        Ok(inputs)
    }

    /// The exact `llama-embedding` arguments, flag names as in the tool's
    /// `--help` (`-m`, `-f`, `--pooling`, `--embd-output-format`,
    /// `--embd-normalize`, `-c`, `-b`, `--fit`, `-ngl`).
    pub fn tool_args(&self, model: &Path, prompts: &Path) -> Vec<String> {
        let normalise = if self.normalise && self.pooling != Pooling::Rank { "2" } else { "-1" };
        let mut args = vec![
            "-m".to_owned(),
            plain(model),
            "-f".to_owned(),
            plain(prompts),
            "--pooling".to_owned(),
            self.pooling.as_str().to_owned(),
            "--embd-output-format".to_owned(),
            "json".to_owned(),
            "--embd-normalize".to_owned(),
            normalise.to_owned(),
            "-c".to_owned(),
            self.context.to_string(),
        ];
        if self.batch > 0 {
            args.push("-b".to_owned());
            args.push(self.batch.to_string());
        }
        if self.fit {
            args.push("--fit".to_owned());
            args.push("on".to_owned());
        } else {
            args.push("-ngl".to_owned());
            args.push(self.gpu_layers.to_string());
        }
        args
    }

    pub fn command_line(&self, model: &Path, prompts: &Path) -> String {
        let quote = |s: &str| if s.contains(' ') { format!("\"{s}\"") } else { s.to_owned() };
        let mut parts = vec![quote(&plain(&self.tool))];
        parts.extend(self.tool_args(model, prompts).iter().map(|s| quote(s)));
        parts.join(" ")
    }

    /// Run the tool once: embed every line of `prompts` with `model`, keep the
    /// raw JSON at `capture` (never overwritten), return one row per line.
    pub fn embed<F>(&self, model: &Path, prompts: &Path, capture: &Path, expected: usize, log: &mut F) -> io::Result<Vec<Vec<f32>>>
    where
        F: FnMut(&str),
    {
        let mut cmd = tool_command(&self.tool);
        cmd.args(self.tool_args(model, prompts));
        run_with_log_to_file(&mut cmd, capture, |line| log(line))?;
        let rows = parse_embedding_output(&fs::read(capture)?)?;
        if rows.len() != expected {
            return Err(io::Error::other(format!("llama-embedding returned {} embeddings for {expected} prompts ({})", rows.len(), capture.display())));
        }
        Ok(rows)
    }
}

#[derive(Deserialize)]
struct EmbeddingItem {
    index: Option<usize>,
    embedding: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum EmbeddingOutput {
    List { data: Vec<EmbeddingItem> },
    Array(Vec<Vec<f64>>),
}

/// Parse `--embd-output-format json` (OpenAI-style list) or `array` output.
/// Fails closed on empty output, non-finite values, mixed dimensions or an
/// index set that is not exactly `0..n`.
pub fn parse_embedding_output(bytes: &[u8]) -> io::Result<Vec<Vec<f32>>> {
    let parsed: EmbeddingOutput = serde_json::from_slice(bytes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("llama-embedding output is not the expected JSON: {e}")))?;
    let rows: Vec<Vec<f64>> = match parsed {
        EmbeddingOutput::Array(rows) => rows,
        EmbeddingOutput::List { data } => {
            let mut items: Vec<(usize, Vec<f64>)> = data
                .into_iter()
                .enumerate()
                .map(|(pos, item)| (item.index.unwrap_or(pos), item.embedding))
                .collect();
            items.sort_by_key(|(i, _)| *i);
            if items.iter().enumerate().any(|(pos, (i, _))| pos != *i) {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "embedding indices are not a contiguous 0..n sequence"));
            }
            items.into_iter().map(|(_, e)| e).collect()
        }
    };
    if rows.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "llama-embedding output holds no embeddings"));
    }
    let dim = rows[0].len();
    if dim == 0 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "embedding dimension is zero"));
    }
    let mut out = Vec::with_capacity(rows.len());
    for (i, row) in rows.into_iter().enumerate() {
        if row.len() != dim {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("embedding {i} has {} values, embedding 0 has {dim}", row.len())));
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err(io::Error::new(io::ErrorKind::InvalidData, format!("embedding {i} contains a non-finite value")));
        }
        out.push(row.into_iter().map(|v| v as f32).collect());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Pure metrics
// ---------------------------------------------------------------------------

/// Cosine similarity; `None` on dimension mismatch or a zero-norm vector.
pub fn cosine(a: &[f32], b: &[f32]) -> Option<f64> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        let (x, y) = (*x as f64, *y as f64);
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na <= 0.0 || nb <= 0.0 {
        return None;
    }
    Some((dot / (na.sqrt() * nb.sqrt())).clamp(-1.0, 1.0))
}

/// Row-by-row cosine between two embedding sets of equal length and dimension.
pub fn pairwise_cosines(reference: &[Vec<f32>], candidate: &[Vec<f32>]) -> io::Result<Vec<f64>> {
    if reference.len() != candidate.len() {
        return Err(invalid(format!("reference has {} embeddings, candidate {}", reference.len(), candidate.len())));
    }
    reference
        .iter()
        .zip(candidate)
        .enumerate()
        .map(|(i, (r, c))| {
            cosine(r, c).ok_or_else(|| {
                if r.len() != c.len() {
                    invalid(format!("embedding {i}: reference dimension {} differs from candidate dimension {}", r.len(), c.len()))
                } else {
                    invalid(format!("embedding {i} has zero norm in one model"))
                }
            })
        })
        .collect()
}

#[derive(Clone, Debug, PartialEq)]
pub struct CosineStats {
    pub count: usize,
    pub min: f64,
    pub mean: f64,
    /// Nearest-rank 5th percentile.
    pub p05: f64,
}

/// Nearest-rank percentile of an ascending slice, `q` in 0..=1.
pub fn percentile(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let rank = (q.clamp(0.0, 1.0) * sorted.len() as f64).ceil() as usize;
    Some(sorted[rank.clamp(1, sorted.len()) - 1])
}

pub fn cosine_stats(values: &[f64]) -> Option<CosineStats> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Some(CosineStats {
        count: values.len(),
        min: sorted[0],
        mean: values.iter().sum::<f64>() / values.len() as f64,
        p05: percentile(&sorted, 0.05)?,
    })
}

/// Cosine of every row of `a` against every row of `b`.
pub fn cross_similarity(a: &[Vec<f32>], b: &[Vec<f32>]) -> io::Result<Vec<Vec<f64>>> {
    a.iter()
        .enumerate()
        .map(|(i, x)| {
            b.iter()
                .enumerate()
                .map(|(j, y)| cosine(x, y).ok_or_else(|| invalid(format!("cannot compare rows {i} and {j}: dimension mismatch or zero norm"))))
                .collect()
        })
        .collect()
}

/// Full self-similarity matrix of one embedding set.
pub fn similarity_matrix(rows: &[Vec<f32>]) -> io::Result<Vec<Vec<f64>>> {
    cross_similarity(rows, rows)
}

/// Indices of the `k` highest scores, best first; ties break on the lower
/// index so both models are ranked by the same deterministic rule.
pub fn top_k(scores: &[f64], k: usize, exclude: Option<usize>) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).filter(|i| Some(*i) != exclude).collect();
    order.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
    order.truncate(k);
    order
}

fn overlap(a: &[usize], b: &[usize]) -> usize {
    a.iter().filter(|i| b.contains(i)).count()
}

#[derive(Clone, Debug, PartialEq)]
pub struct OverlapStats {
    pub k: usize,
    pub mean: f64,
    pub min: f64,
}

/// For each text, the fraction of its k nearest other texts (reference) that
/// are also among its k nearest under the candidate.
pub fn neighbour_overlap(reference: &[Vec<f64>], candidate: &[Vec<f64>], k: usize) -> Option<OverlapStats> {
    let n = reference.len();
    if n == 0 || candidate.len() != n || k == 0 || n <= k {
        return None;
    }
    let mut total = 0.0;
    let mut min = 1.0f64;
    for i in 0..n {
        let r = top_k(&reference[i], k, Some(i));
        let c = top_k(&candidate[i], k, Some(i));
        let v = overlap(&r, &c) as f64 / k as f64;
        total += v;
        min = min.min(v);
    }
    Some(OverlapStats { k, mean: total / n as f64, min })
}

/// Mean over queries of |reference top-k ∩ candidate top-k| / k, the reference
/// top-k standing in for the relevant set.
pub fn recall_at_k(reference: &[Vec<f64>], candidate: &[Vec<f64>], k: usize) -> Option<f64> {
    let q = reference.len();
    if q == 0 || candidate.len() != q || k == 0 {
        return None;
    }
    let mut total = 0.0;
    for (r, c) in reference.iter().zip(candidate) {
        if r.len() != c.len() || r.len() < k {
            return None;
        }
        total += overlap(&top_k(r, k, None), &top_k(c, k, None)) as f64 / k as f64;
    }
    Some(total / q as f64)
}

/// NDCG@k of the candidate's ranking of one query against the reference's.
/// Graded gains come from the reference order: its rank-1 document gains k,
/// rank-2 gains k-1, ... rank-k gains 1, everything else 0. 1.0 means the
/// candidate's top-k is the reference's top-k in the same order.
pub fn ndcg_at_k(reference: &[f64], candidate: &[f64], k: usize) -> Option<f64> {
    let d = reference.len();
    if d == 0 || candidate.len() != d || k == 0 {
        return None;
    }
    let k = k.min(d);
    let ideal = top_k(reference, k, None);
    let mut gain = vec![0.0f64; d];
    for (rank, doc) in ideal.iter().enumerate() {
        gain[*doc] = (k - rank) as f64;
    }
    let discount = |pos: usize| 1.0 / ((pos + 2) as f64).log2();
    let dcg: f64 = top_k(candidate, k, None).iter().enumerate().map(|(pos, doc)| gain[*doc] * discount(pos)).sum();
    let idcg: f64 = ideal.iter().enumerate().map(|(pos, doc)| gain[*doc] * discount(pos)).sum();
    if idcg <= 0.0 {
        return None;
    }
    Some((dcg / idcg).clamp(0.0, 1.0))
}

pub fn mean_ndcg_at_k(reference: &[Vec<f64>], candidate: &[Vec<f64>], k: usize) -> Option<f64> {
    if reference.is_empty() || reference.len() != candidate.len() {
        return None;
    }
    let mut total = 0.0;
    for (r, c) in reference.iter().zip(candidate) {
        total += ndcg_at_k(r, c, k)?;
    }
    Some(total / reference.len() as f64)
}

// ---------------------------------------------------------------------------
// Evaluation of one model pair
// ---------------------------------------------------------------------------

/// Everything one model produced over the planned prompt sets.
#[derive(Clone, Debug, Default)]
pub struct EmbeddedSets {
    pub texts: Vec<Vec<f32>>,
    pub queries: Vec<Vec<f32>>,
    pub documents: Vec<Vec<f32>>,
    /// Rank pooling: one score row per `query<TAB>document` pair, query-major.
    pub pairs: Vec<Vec<f32>>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParityMetrics {
    pub dimension: usize,
    /// What the cosine was taken over: `embeddings` or `rerank_score_rows`.
    pub cosine_basis: &'static str,
    pub cosine: CosineStats,
    pub neighbour_overlap: Option<OverlapStats>,
    pub recall_at_k: Option<(usize, f64)>,
    pub ndcg_at_10: Option<f64>,
}

impl ParityMetrics {
    pub fn as_json(&self) -> Value {
        json!({
            "dimension": self.dimension,
            "cosine": {
                "basis": self.cosine_basis,
                "count": self.cosine.count,
                "min": self.cosine.min,
                "mean": self.cosine.mean,
                "p05": self.cosine.p05,
            },
            "neighbour_overlap": self.neighbour_overlap.as_ref().map(|o| json!({"k": o.k, "mean": o.mean, "min": o.min})),
            "recall_at_k": self.recall_at_k.map(|(k, v)| json!({"k": k, "value": v})),
            "ndcg_at_10": self.ndcg_at_10,
        })
    }
}

/// Score matrix (queries x documents) from rerank pair rows, which must be
/// single scores.
pub fn rerank_matrix(pairs: &[Vec<f32>], queries: usize, documents: usize) -> io::Result<Vec<Vec<f64>>> {
    if pairs.len() != queries * documents {
        return Err(invalid(format!("expected {} pair scores ({queries} x {documents}), found {}", queries * documents, pairs.len())));
    }
    let mut rows = Vec::with_capacity(queries);
    for q in 0..queries {
        let mut row = Vec::with_capacity(documents);
        for d in 0..documents {
            let cell = &pairs[q * documents + d];
            if cell.len() != 1 {
                return Err(invalid(format!("rank pooling returned {} values per pair; expected one relevance score", cell.len())));
            }
            row.push(cell[0] as f64);
        }
        rows.push(row);
    }
    Ok(rows)
}

pub fn evaluate(settings: &EmbedSettings, inputs: &EmbedInputs, reference: &EmbeddedSets, candidate: &EmbeddedSets) -> io::Result<ParityMetrics> {
    let k = settings.top_k;
    if settings.pooling == Pooling::Rank {
        let (q, d) = (inputs.queries.len(), inputs.documents.len());
        let r = rerank_matrix(&reference.pairs, q, d)?;
        let c = rerank_matrix(&candidate.pairs, q, d)?;
        let as_f32 = |m: &Vec<Vec<f64>>| -> Vec<Vec<f32>> { m.iter().map(|row| row.iter().map(|v| *v as f32).collect()).collect() };
        let cos = pairwise_cosines(&as_f32(&r), &as_f32(&c))?;
        return Ok(ParityMetrics {
            dimension: 1,
            cosine_basis: "rerank_score_rows",
            cosine: cosine_stats(&cos).ok_or_else(|| invalid("no queries"))?,
            neighbour_overlap: None,
            recall_at_k: Some((k, recall_at_k(&r, &c, k).ok_or_else(|| invalid("Recall@k needs at least k documents"))?)),
            ndcg_at_10: Some(mean_ndcg_at_k(&r, &c, NDCG_K).ok_or_else(|| invalid("NDCG needs at least one document"))?),
        });
    }
    let mut cos = pairwise_cosines(&reference.texts, &candidate.texts)?;
    cos.extend(pairwise_cosines(&reference.queries, &candidate.queries)?);
    cos.extend(pairwise_cosines(&reference.documents, &candidate.documents)?);
    let dimension = reference.texts.first().map(|r| r.len()).unwrap_or(0);
    let neighbour_overlap = neighbour_overlap(&similarity_matrix(&reference.texts)?, &similarity_matrix(&candidate.texts)?, k)
        .ok_or_else(|| invalid("neighbour overlap needs more than k texts"))?;
    let (mut recall, mut ndcg) = (None, None);
    if inputs.has_retrieval() {
        let r = cross_similarity(&reference.queries, &reference.documents)?;
        let c = cross_similarity(&candidate.queries, &candidate.documents)?;
        recall = Some((k, recall_at_k(&r, &c, k).ok_or_else(|| invalid("Recall@k needs at least k documents"))?));
        ndcg = Some(mean_ndcg_at_k(&r, &c, NDCG_K).ok_or_else(|| invalid("NDCG needs at least one document"))?);
    }
    Ok(ParityMetrics {
        dimension,
        cosine_basis: "embeddings",
        cosine: cosine_stats(&cos).ok_or_else(|| invalid("no texts"))?,
        neighbour_overlap: Some(neighbour_overlap),
        recall_at_k: recall,
        ndcg_at_10: ndcg,
    })
}

/// Apply the gate. Returns pass and the human-readable failures.
pub fn gate_result(gate: &EmbedGate, metrics: &ParityMetrics) -> (bool, Vec<String>) {
    let mut failures = Vec::new();
    if metrics.cosine.min < gate.min_cosine {
        failures.push(format!("min cosine {:.6} < {:.6}", metrics.cosine.min, gate.min_cosine));
    }
    if let Some(min) = gate.min_recall {
        match metrics.recall_at_k {
            Some((_, v)) if v >= min => {}
            Some((k, v)) => failures.push(format!("Recall@{k} {v:.4} < {min:.4}")),
            None => failures.push("Recall@k not measured".to_owned()),
        }
    }
    if let Some(min) = gate.min_ndcg {
        match metrics.ndcg_at_10 {
            Some(v) if v >= min => {}
            Some(v) => failures.push(format!("NDCG@{NDCG_K} {v:.4} < {min:.4}")),
            None => failures.push(format!("NDCG@{NDCG_K} not measured")),
        }
    }
    (failures.is_empty(), failures)
}

// ---------------------------------------------------------------------------
// Plan and run
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct EmbedParityJob {
    pub reference: PathBuf,
    pub candidate: PathBuf,
    pub output_dir: PathBuf,
    pub settings: EmbedSettings,
    pub gate: EmbedGate,
}

#[derive(Clone, Debug)]
pub struct PlannedRun {
    pub role: &'static str,
    pub set: &'static str,
    pub model: PathBuf,
    pub prompts: PathBuf,
    pub capture: PathBuf,
    pub command: String,
    pub lines: usize,
}

#[derive(Clone, Debug)]
pub struct EmbedParityPlan {
    pub job: EmbedParityJob,
    pub inputs: EmbedInputs,
    pub runs: Vec<PlannedRun>,
}

fn prompt_sets(inputs: &EmbedInputs, rank: bool) -> Vec<(&'static str, Vec<String>)> {
    let mut sets = Vec::new();
    if rank {
        sets.push(("pairs", inputs.pairs()));
    } else {
        sets.push(("texts", inputs.texts.clone()));
        if inputs.has_retrieval() {
            sets.push(("queries", inputs.queries.clone()));
            sets.push(("documents", inputs.documents.clone()));
        }
    }
    sets
}

/// Validate everything and describe the runs. Writes nothing.
pub fn prepare_embed_parity(job: EmbedParityJob) -> io::Result<EmbedParityPlan> {
    let inputs = job.settings.load_inputs()?;
    job.gate.validate(inputs.has_retrieval())?;
    for (what, model) in [("reference", &job.reference), ("candidate", &job.candidate)] {
        if inspect_gguf(model).map_err(|e| invalid(format!("{what} model {}: {e}", model.display())))?.tensor_count == 0 {
            return Err(invalid(format!("{what} model has an empty GGUF tensor catalog")));
        }
    }
    if job.output_dir.is_file() {
        return Err(invalid("output path is a file, not a directory"));
    }
    let rank = job.settings.pooling == Pooling::Rank;
    let mut runs = Vec::new();
    let mut planned_files = vec![job.output_dir.join(RECEIPT_NAME)];
    for (set, lines) in prompt_sets(&inputs, rank) {
        let prompts = job.output_dir.join(format!("prompts-{set}.txt"));
        planned_files.push(prompts.clone());
        for (role, model) in [("reference", &job.reference), ("candidate", &job.candidate)] {
            let capture = job.output_dir.join(format!("{role}-{set}.json"));
            planned_files.push(capture.clone());
            runs.push(PlannedRun {
                role,
                set,
                model: model.clone(),
                prompts: prompts.clone(),
                capture,
                command: job.settings.command_line(model, &prompts),
                lines: lines.len(),
            });
        }
    }
    if let Some(existing) = planned_files.iter().find(|p| p.exists()) {
        return Err(invalid(format!("{} already exists; use an empty output directory", existing.display())));
    }
    Ok(EmbedParityPlan { job, inputs, runs })
}

impl EmbedParityPlan {
    fn settings_json(&self) -> Value {
        let s = &self.job.settings;
        json!({
            "tool": plain(&s.tool),
            "pooling": s.pooling.as_str(),
            "normalise": s.normalise && s.pooling != Pooling::Rank,
            "context": s.context,
            "batch": if s.batch == 0 { Value::Null } else { json!(s.batch) },
            "gpu_layers": if s.fit { Value::String("fit".to_owned()) } else { json!(s.gpu_layers) },
            "top_k": s.top_k,
        })
    }

    fn inputs_json(&self) -> Value {
        let s = &self.job.settings;
        json!({
            "texts": s.texts.as_ref().map(|p| json!({"path": plain(p), "lines": self.inputs.texts.len()})),
            "queries": s.queries.as_ref().map(|p| json!({"path": plain(p), "lines": self.inputs.queries.len()})),
            "documents": s.documents.as_ref().map(|p| json!({"path": plain(p), "lines": self.inputs.documents.len()})),
            "pairs": if s.pooling == Pooling::Rank { json!(self.inputs.queries.len() * self.inputs.documents.len()) } else { Value::Null },
        })
    }

    fn gate_json(&self) -> Value {
        json!({
            "min_cosine": self.job.gate.min_cosine,
            "min_recall": self.job.gate.min_recall,
            "min_ndcg": self.job.gate.min_ndcg,
            "cosine_gate_applies_to": "minimum per-text cosine",
        })
    }

    fn runs_json(&self, seconds: Option<&[f64]>) -> Value {
        Value::Array(
            self.runs
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    json!({
                        "role": r.role,
                        "set": r.set,
                        "model": plain(&r.model),
                        "prompts": plain(&r.prompts),
                        "capture": plain(&r.capture),
                        "lines": r.lines,
                        "command": r.command,
                        "seconds": seconds.and_then(|s| s.get(i)),
                    })
                })
                .collect(),
        )
    }

    pub fn plan_json(&self) -> Value {
        json!({
            "schema": "ojon.embed_parity_plan.v1",
            "reference": plain(&self.job.reference),
            "candidate": plain(&self.job.candidate),
            "output_dir": plain(&self.job.output_dir),
            "settings": self.settings_json(),
            "inputs": self.inputs_json(),
            "gate": self.gate_json(),
            "runs": self.runs_json(None),
            "writes": "nothing until embed-parity-run --confirm",
        })
    }
}

fn write_new(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let mut f = fs::OpenOptions::new().write(true).create_new(true).open(path)?;
    f.write_all(bytes)
}

/// Execute the plan: write the prompt files, run the tool once per model and
/// prompt set, evaluate, write `embed-parity-receipt.json`. Returns its path.
pub fn run_embed_parity<F>(plan: &EmbedParityPlan, mut log: F) -> io::Result<PathBuf>
where
    F: FnMut(&str),
{
    let dir = &plan.job.output_dir;
    fs::create_dir_all(dir)?;
    let receipt_path = dir.join(RECEIPT_NAME);
    if receipt_path.exists() {
        return Err(invalid(format!("{} already exists", receipt_path.display())));
    }
    let rank = plan.job.settings.pooling == Pooling::Rank;
    for (set, lines) in prompt_sets(&plan.inputs, rank) {
        write_new(&dir.join(format!("prompts-{set}.txt")), lines.join("\n").as_bytes())?;
    }
    let mut reference = EmbeddedSets::default();
    let mut candidate = EmbeddedSets::default();
    let mut seconds = Vec::with_capacity(plan.runs.len());
    for run in &plan.runs {
        log(&format!("[embed-parity] {} {}: {}", run.role, run.set, run.command));
        let started = Instant::now();
        let rows = plan.job.settings.embed(&run.model, &run.prompts, &run.capture, run.lines, &mut log)?;
        seconds.push(started.elapsed().as_secs_f64());
        let target = if run.role == "reference" { &mut reference } else { &mut candidate };
        match run.set {
            "texts" => target.texts = rows,
            "queries" => target.queries = rows,
            "documents" => target.documents = rows,
            _ => target.pairs = rows,
        }
    }
    let metrics = evaluate(&plan.job.settings, &plan.inputs, &reference, &candidate)?;
    let (pass, failures) = gate_result(&plan.job.gate, &metrics);
    let mut gate = plan.gate_json();
    gate["pass"] = json!(pass);
    gate["failures"] = json!(failures);
    let receipt = json!({
        "schema": SCHEMA,
        "created_unix": SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0),
        "reference": plain(&plan.job.reference),
        "candidate": plain(&plan.job.candidate),
        "output_dir": plain(dir),
        "settings": plan.settings_json(),
        "inputs": plan.inputs_json(),
        "runs": plan.runs_json(Some(&seconds)),
        "metrics": metrics.as_json(),
        "gate": gate,
        "claim": CLAIM,
    });
    write_new(&receipt_path, &serde_json::to_vec_pretty(&receipt)?)?;
    log(&format!("[embed-parity] {} min cosine {:.6} -> {}", if pass { "PASS" } else { "FAIL" }, metrics.cosine.min, receipt_path.display()));
    Ok(receipt_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn parses_openai_style_json_and_array_output() {
        let json = br#"{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 1,
      "embedding": [0.0000000,1.0000000,0.0000000]
    },
    {
      "object": "embedding",
      "index": 0,
      "embedding": [1.0000000,0.0000000,0.0000000]
    }
  ]
}

"#;
        let rows = parse_embedding_output(json).unwrap();
        assert_eq!(rows, vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]]);
        let rows = parse_embedding_output(b"[[0.5,0.5],[0.25,-0.75]]\n").unwrap();
        assert_eq!(rows, vec![vec![0.5, 0.5], vec![0.25, -0.75]]);
    }

    #[test]
    fn parser_fails_closed() {
        assert!(parse_embedding_output(b"").is_err());
        assert!(parse_embedding_output(b"[]").is_err());
        assert!(parse_embedding_output(b"[[]]").is_err());
        assert!(parse_embedding_output(b"[[1.0,2.0],[1.0]]").is_err(), "mixed dimensions");
        assert!(parse_embedding_output(b"[[1.0,nan]]").is_err(), "nan is not JSON");
        assert!(parse_embedding_output(br#"{"object":"list","data":[{"index":0,"embedding":[1.0]},{"index":0,"embedding":[1.0]}]}"#).is_err(), "duplicate index");
        assert!(parse_embedding_output(b"embedding 0:  0.1 0.2").is_err(), "default text format is rejected");
    }

    #[test]
    fn cosine_and_stats() {
        assert!(close(cosine(&[1.0, 0.0], &[1.0, 0.0]).unwrap(), 1.0));
        assert!(close(cosine(&[1.0, 0.0], &[0.0, 1.0]).unwrap(), 0.0));
        assert!(close(cosine(&[1.0, 0.0], &[-2.0, 0.0]).unwrap(), -1.0));
        assert!(close(cosine(&[3.0, 4.0], &[6.0, 8.0]).unwrap(), 1.0));
        assert_eq!(cosine(&[1.0], &[1.0, 2.0]), None);
        assert_eq!(cosine(&[0.0, 0.0], &[1.0, 2.0]), None);
        let s = cosine_stats(&[0.99, 0.95, 1.0, 0.97]).unwrap();
        assert_eq!(s.count, 4);
        assert!(close(s.min, 0.95));
        assert!(close(s.mean, 0.9775));
        assert!(close(s.p05, 0.95));
        let twenty: Vec<f64> = (1..=20).map(|i| i as f64 / 20.0).collect();
        assert!(close(cosine_stats(&twenty).unwrap().p05, 0.05));
        assert_eq!(cosine_stats(&[]), None);
    }

    #[test]
    fn pairwise_cosines_reject_shape_mismatch() {
        let a = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
        let b = vec![vec![1.0f32, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        assert!(pairwise_cosines(&a, &b).is_err());
        assert!(pairwise_cosines(&a, &a[..1]).is_err());
        let c = pairwise_cosines(&a, &a).unwrap();
        assert!(c.iter().all(|v| close(*v, 1.0)));
    }

    #[test]
    fn top_k_breaks_ties_on_index_and_excludes_self() {
        assert_eq!(top_k(&[0.5, 0.9, 0.9, 0.1], 2, None), vec![1, 2]);
        assert_eq!(top_k(&[1.0, 0.9, 0.8, 0.7], 2, Some(0)), vec![1, 2]);
        assert_eq!(top_k(&[0.1, 0.2], 5, None), vec![1, 0]);
    }

    fn unit_rows() -> Vec<Vec<f32>> {
        // Eight points on the plane; neighbours by angle.
        (0..8)
            .map(|i| {
                let t = i as f32 * std::f32::consts::PI / 4.0;
                vec![t.cos(), t.sin()]
            })
            .collect()
    }

    #[test]
    fn neighbour_overlap_is_one_for_identical_sets_and_drops_when_shuffled() {
        let rows = unit_rows();
        let m = similarity_matrix(&rows).unwrap();
        let same = neighbour_overlap(&m, &m, 2).unwrap();
        assert!(close(same.mean, 1.0) && close(same.min, 1.0));
        let mut shuffled = rows.clone();
        shuffled.swap(0, 4); // opposite points swap places
        let s = similarity_matrix(&shuffled).unwrap();
        let moved = neighbour_overlap(&m, &s, 2).unwrap();
        assert!(moved.mean < 1.0 && moved.min == 0.0, "{moved:?}");
        assert!(neighbour_overlap(&m, &m, 8).is_none(), "k must leave at least one other text");
        assert!(neighbour_overlap(&m, &m[..7], 2).is_none());
    }

    #[test]
    fn recall_and_ndcg_against_reference_ranking() {
        let reference = vec![vec![0.9, 0.8, 0.7, 0.1, 0.0]];
        assert!(close(recall_at_k(&reference, &reference, 3).unwrap(), 1.0));
        assert!(close(mean_ndcg_at_k(&reference, &reference, 10).unwrap(), 1.0));
        // Same top-3 set, swapped order: recall stays 1, NDCG drops below 1.
        let reordered = vec![vec![0.7, 0.8, 0.9, 0.1, 0.0]];
        assert!(close(recall_at_k(&reference, &reordered, 3).unwrap(), 1.0));
        let n = mean_ndcg_at_k(&reference, &reordered, 10).unwrap();
        assert!(n < 1.0 && n > 0.5, "{n}");
        // Reversed ranking: the reference top-3 are now last.
        let reversed = vec![vec![0.0, 0.1, 0.7, 0.8, 0.9]];
        assert!(close(recall_at_k(&reference, &reversed, 3).unwrap(), 1.0 / 3.0));
        assert!(mean_ndcg_at_k(&reference, &reversed, 10).unwrap() < n);
        // k larger than the document count folds to the document count.
        assert!(close(ndcg_at_k(&[0.2, 0.1], &[0.2, 0.1], 10).unwrap(), 1.0));
        assert_eq!(recall_at_k(&reference, &reference, 6), None);
        assert_eq!(ndcg_at_k(&[], &[], 10), None);
        assert_eq!(ndcg_at_k(&[0.1, 0.2], &[0.1], 10), None);
    }

    #[test]
    fn ndcg_hand_computed() {
        // Reference order: d0 > d1 > d2 (gains 3,2,1). Candidate: d1 > d0 > d2.
        let r = [0.9, 0.8, 0.7];
        let c = [0.8, 0.9, 0.7];
        let d = |p: f64| 1.0 / (p + 2.0).log2();
        let dcg = 2.0 * d(0.0) + 3.0 * d(1.0) + 1.0 * d(2.0);
        let idcg = 3.0 * d(0.0) + 2.0 * d(1.0) + 1.0 * d(2.0);
        assert!(close(ndcg_at_k(&r, &c, 10).unwrap(), dcg / idcg));
    }

    #[test]
    fn rerank_matrix_requires_single_scores() {
        let pairs: Vec<Vec<f32>> = (0..6).map(|i| vec![i as f32]).collect();
        let m = rerank_matrix(&pairs, 2, 3).unwrap();
        assert_eq!(m, vec![vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0]]);
        assert!(rerank_matrix(&pairs, 2, 2).is_err());
        assert!(rerank_matrix(&[vec![0.0, 1.0]], 1, 1).is_err());
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ojon-embed-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn write_lines(path: &Path, n: usize, prefix: &str) {
        let body: Vec<String> = (0..n).map(|i| format!("{prefix} line {i}")).collect();
        fs::write(path, body.join("\n") + "\n").unwrap();
    }

    #[test]
    fn load_lines_strips_cr_and_rejects_empty_lines() {
        let dir = temp_dir("lines");
        let p = dir.join("t.txt");
        fs::write(&p, "a\r\nb\r\nc\r\n").unwrap();
        assert_eq!(load_lines(&p, "t", 1..=10, false).unwrap(), vec!["a", "b", "c"]);
        fs::write(&p, "a\n\nb\n").unwrap();
        assert!(load_lines(&p, "t", 1..=10, false).is_err());
        fs::write(&p, "a\tb\n").unwrap();
        assert!(load_lines(&p, "t", 1..=10, true).is_err());
        assert!(load_lines(&p, "t", 1..=10, false).is_ok());
        assert!(load_lines(&p, "t", 2..=10, false).is_err(), "too few lines");
        assert!(load_lines(&dir.join("missing.txt"), "t", 1..=10, false).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn settings_validation_fails_closed() {
        let dir = temp_dir("settings");
        let texts = dir.join("texts.txt");
        let queries = dir.join("queries.txt");
        let documents = dir.join("docs.txt");
        write_lines(&texts, 8, "text");
        write_lines(&queries, 2, "query");
        write_lines(&documents, 6, "doc");
        let ok = EmbedSettings { texts: Some(texts.clone()), ..EmbedSettings::default() };
        let inputs = ok.load_inputs().unwrap();
        assert_eq!(inputs.texts.len(), 8);
        assert!(!inputs.has_retrieval());
        assert!(EmbedSettings { texts: None, ..ok.clone() }.validate().is_err(), "texts required");
        assert!(EmbedSettings { context: 500, ..ok.clone() }.validate().is_err());
        assert!(EmbedSettings { context: 32, ..ok.clone() }.validate().is_err());
        assert!(EmbedSettings { batch: 256, ..ok.clone() }.validate().is_err(), "batch below context");
        assert!(EmbedSettings { batch: 1024, ..ok.clone() }.validate().is_ok());
        assert!(EmbedSettings { gpu_layers: 999, ..ok.clone() }.validate().is_err());
        assert!(EmbedSettings { top_k: 0, ..ok.clone() }.validate().is_err());
        assert!(EmbedSettings { top_k: 8, ..ok.clone() }.validate().is_err(), "k must be below the text count");
        assert!(EmbedSettings { tool: PathBuf::new(), ..ok.clone() }.validate().is_err());
        assert!(EmbedSettings { queries: Some(queries.clone()), ..ok.clone() }.validate().is_err(), "queries without documents");
        let retrieval = EmbedSettings { queries: Some(queries.clone()), documents: Some(documents.clone()), ..ok.clone() };
        assert!(retrieval.load_inputs().unwrap().has_retrieval());
        assert!(EmbedSettings { top_k: 7, ..retrieval.clone() }.validate().is_err(), "k must fit the document count");
        // Rank pooling: pairs only, texts refused.
        let rank = EmbedSettings { pooling: Pooling::Rank, texts: None, ..retrieval.clone() };
        let pairs = rank.load_inputs().unwrap().pairs();
        assert_eq!(pairs.len(), 12);
        assert_eq!(pairs[7], "query line 1\tdoc line 1");
        assert!(EmbedSettings { texts: Some(texts.clone()), ..rank.clone() }.validate().is_err());
        assert!(EmbedSettings { queries: None, documents: None, ..rank.clone() }.validate().is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn gate_validation_and_result() {
        let gate = EmbedGate::default();
        assert!(gate.validate(false).is_ok());
        assert!(EmbedGate { min_cosine: 1.5, ..gate.clone() }.validate(false).is_err());
        assert!(EmbedGate { min_recall: Some(0.9), ..gate.clone() }.validate(false).is_err());
        assert!(EmbedGate { min_recall: Some(0.9), ..gate.clone() }.validate(true).is_ok());
        let metrics = ParityMetrics {
            dimension: 4,
            cosine_basis: "embeddings",
            cosine: CosineStats { count: 8, min: 0.985, mean: 0.999, p05: 0.99 },
            neighbour_overlap: None,
            recall_at_k: Some((5, 0.8)),
            ndcg_at_10: Some(0.95),
        };
        let (pass, failures) = gate_result(&gate, &metrics);
        assert!(!pass && failures.len() == 1, "{failures:?}");
        let (pass, _) = gate_result(&EmbedGate { min_cosine: 0.98, ..gate.clone() }, &metrics);
        assert!(pass);
        let strict = EmbedGate { min_cosine: 0.98, min_recall: Some(0.9), min_ndcg: Some(0.9) };
        let (pass, failures) = gate_result(&strict, &metrics);
        assert!(!pass && failures.len() == 1 && failures[0].starts_with("Recall@5"), "{failures:?}");
    }

    #[test]
    fn tool_args_follow_llama_embedding_help() {
        let s = EmbedSettings { texts: Some(PathBuf::from("t.txt")), ..EmbedSettings::default() };
        let args = s.tool_args(Path::new("m.gguf"), Path::new("p.txt"));
        assert_eq!(
            args,
            vec!["-m", "m.gguf", "-f", "p.txt", "--pooling", "mean", "--embd-output-format", "json", "--embd-normalize", "2", "-c", "512", "-ngl", "99"]
        );
        let fit = EmbedSettings { fit: true, batch: 2048, pooling: Pooling::Rank, ..s.clone() };
        let args = fit.tool_args(Path::new("m.gguf"), Path::new("p.txt"));
        assert!(args.windows(2).any(|w| w == ["--fit", "on"]));
        assert!(args.windows(2).any(|w| w == ["-b", "2048"]));
        assert!(args.windows(2).any(|w| w == ["--embd-normalize", "-1"]), "rank never normalises: {args:?}");
        assert!(!args.iter().any(|a| a == "-ngl"));
        let no_norm = EmbedSettings { normalise: false, ..s.clone() };
        assert!(no_norm.tool_args(Path::new("m.gguf"), Path::new("p.txt")).windows(2).any(|w| w == ["--embd-normalize", "-1"]));
        let line = s.command_line(Path::new("my model.gguf"), Path::new("p.txt"));
        assert!(line.contains("\"my model.gguf\""), "{line}");
    }

    /// Irregular angles: no ties among the neighbours any assertion below depends on.
    fn skew_rows() -> Vec<Vec<f32>> {
        [0.0f32, 0.6, 1.5, 2.1, 3.0, 3.9, 4.4, 5.5].iter().map(|t| vec![t.cos(), t.sin()]).collect()
    }

    #[test]
    fn evaluate_identical_models_pass_and_perturbed_models_measure_below_one() {
        let dir = temp_dir("eval");
        let texts = dir.join("texts.txt");
        let queries = dir.join("q.txt");
        let documents = dir.join("d.txt");
        write_lines(&texts, 8, "text");
        write_lines(&queries, 2, "query");
        write_lines(&documents, 6, "doc");
        let settings = EmbedSettings { texts: Some(texts), queries: Some(queries), documents: Some(documents), top_k: 2, ..EmbedSettings::default() };
        let inputs = settings.load_inputs().unwrap();
        let rows = skew_rows();
        let reference = EmbeddedSets { texts: rows.clone(), queries: rows[..2].to_vec(), documents: rows[2..].to_vec(), pairs: vec![] };
        let m = evaluate(&settings, &inputs, &reference, &reference).unwrap();
        assert_eq!(m.dimension, 2);
        assert_eq!(m.cosine.count, 16);
        assert!(close(m.cosine.min, 1.0));
        assert!(close(m.neighbour_overlap.as_ref().unwrap().mean, 1.0));
        assert_eq!(m.recall_at_k, Some((2, 1.0)));
        assert!(close(m.ndcg_at_10.unwrap(), 1.0));
        assert!(gate_result(&EmbedGate::default(), &m).0);
        // Rotate every candidate vector by 30 degrees: cosines drop to cos(30), rankings are preserved.
        let rot = |v: &Vec<f32>| {
            let (c, s) = (30f32.to_radians().cos(), 30f32.to_radians().sin());
            vec![v[0] * c - v[1] * s, v[0] * s + v[1] * c]
        };
        let candidate = EmbeddedSets {
            texts: rows.iter().map(rot).collect(),
            queries: rows[..2].iter().map(rot).collect(),
            documents: rows[2..].iter().map(rot).collect(),
            pairs: vec![],
        };
        let m = evaluate(&settings, &inputs, &reference, &candidate).unwrap();
        assert!((m.cosine.min - 30f64.to_radians().cos()).abs() < 1e-5, "{}", m.cosine.min);
        assert!(close(m.neighbour_overlap.as_ref().unwrap().mean, 1.0));
        assert_eq!(m.recall_at_k, Some((2, 1.0)));
        assert!(close(m.ndcg_at_10.unwrap(), 1.0));
        let (pass, failures) = gate_result(&EmbedGate::default(), &m);
        assert!(!pass && failures[0].starts_with("min cosine"));
        // Dimension mismatch between the models fails closed.
        let wide = EmbeddedSets { texts: rows.iter().map(|v| vec![v[0], v[1], 0.0]).collect(), ..reference.clone() };
        assert!(evaluate(&settings, &inputs, &reference, &wide).is_err());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn evaluate_rank_mode_uses_score_rows() {
        let dir = temp_dir("rank");
        let queries = dir.join("q.txt");
        let documents = dir.join("d.txt");
        write_lines(&queries, 2, "query");
        write_lines(&documents, 4, "doc");
        let settings = EmbedSettings { pooling: Pooling::Rank, queries: Some(queries), documents: Some(documents), top_k: 2, ..EmbedSettings::default() };
        let inputs = settings.load_inputs().unwrap();
        let scores = |v: &[f32]| -> Vec<Vec<f32>> { v.iter().map(|s| vec![*s]).collect() };
        let reference = EmbeddedSets { pairs: scores(&[3.0, 2.0, 1.0, 0.0, 0.0, 1.0, 2.0, 3.0]), ..Default::default() };
        let m = evaluate(&settings, &inputs, &reference, &reference).unwrap();
        assert_eq!(m.cosine_basis, "rerank_score_rows");
        assert_eq!(m.cosine.count, 2);
        assert_eq!(m.recall_at_k, Some((2, 1.0)));
        assert!(close(m.ndcg_at_10.unwrap(), 1.0));
        assert!(m.neighbour_overlap.is_none());
        // Second query's ranking flipped in the candidate.
        let candidate = EmbeddedSets { pairs: scores(&[3.0, 2.0, 1.0, 0.0, 3.0, 2.0, 1.0, 0.0]), ..Default::default() };
        let m = evaluate(&settings, &inputs, &reference, &candidate).unwrap();
        assert!(close(m.recall_at_k.unwrap().1, 0.5), "{:?}", m.recall_at_k);
        assert!(m.ndcg_at_10.unwrap() < 1.0);
        assert!(m.cosine.min < 0.5, "{}", m.cosine.min);
        let _ = fs::remove_dir_all(&dir);
    }

    fn tiny_gguf(path: &Path) {
        use std::io::Seek;
        fn val_string(f: &mut fs::File, s: &str) {
            f.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
            f.write_all(s.as_bytes()).unwrap();
        }
        let mut f = fs::File::create(path).unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap(); // 1 tensor
        f.write_all(&1u64.to_le_bytes()).unwrap(); // 1 metadata key
        val_string(&mut f, "general.architecture");
        f.write_all(&8u32.to_le_bytes()).unwrap();
        val_string(&mut f, "bert");
        val_string(&mut f, "token_embd.weight");
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&0u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        let pos = f.stream_position().unwrap();
        f.write_all(&vec![0u8; ((32 - pos % 32) % 32) as usize]).unwrap();
        f.write_all(&[0u8; 32]).unwrap();
    }

    #[test]
    fn prepare_describes_runs_and_refuses_existing_outputs_and_bad_models() {
        let dir = temp_dir("prepare");
        let texts = dir.join("texts.txt");
        let queries = dir.join("q.txt");
        let documents = dir.join("d.txt");
        write_lines(&texts, 8, "text");
        write_lines(&queries, 2, "query");
        write_lines(&documents, 6, "doc");
        let reference = dir.join("ref.gguf");
        let candidate = dir.join("cand.gguf");
        tiny_gguf(&reference);
        tiny_gguf(&candidate);
        let job = EmbedParityJob {
            reference: reference.clone(),
            candidate: candidate.clone(),
            output_dir: dir.join("out"),
            settings: EmbedSettings { texts: Some(texts.clone()), queries: Some(queries.clone()), documents: Some(documents.clone()), ..EmbedSettings::default() },
            gate: EmbedGate::default(),
        };
        let plan = prepare_embed_parity(job.clone()).unwrap();
        assert_eq!(plan.runs.len(), 6, "texts, queries, documents x two models");
        assert_eq!(plan.runs[0].role, "reference");
        assert_eq!(plan.runs[1].role, "candidate");
        assert!(plan.runs[0].command.contains("--embd-output-format json"), "{}", plan.runs[0].command);
        assert!(plan.runs[0].command.contains("prompts-texts.txt"));
        assert!(!dir.join("out").exists(), "plan writes nothing");
        let doc = plan.plan_json();
        assert_eq!(doc["schema"], "ojon.embed_parity_plan.v1");
        assert_eq!(doc["inputs"]["texts"]["lines"], 8);
        assert_eq!(doc["runs"].as_array().unwrap().len(), 6);
        // Rank mode: one pair set per model.
        let rank = EmbedParityJob {
            settings: EmbedSettings { pooling: Pooling::Rank, texts: None, ..job.settings.clone() },
            ..job.clone()
        };
        let plan = prepare_embed_parity(rank).unwrap();
        assert_eq!(plan.runs.len(), 2);
        assert_eq!(plan.runs[0].lines, 12);
        assert!(plan.runs[0].command.contains("--pooling rank") && plan.runs[0].command.contains("--embd-normalize -1"));
        // A leftover capture or receipt in the output directory is refused.
        fs::create_dir_all(dir.join("out")).unwrap();
        fs::write(dir.join("out").join("candidate-texts.json"), b"{}").unwrap();
        assert!(prepare_embed_parity(job.clone()).is_err());
        fs::remove_file(dir.join("out").join("candidate-texts.json")).unwrap();
        assert!(prepare_embed_parity(job.clone()).is_ok());
        fs::write(dir.join("out").join(RECEIPT_NAME), b"{}").unwrap();
        assert!(prepare_embed_parity(job.clone()).is_err());
        // Models must be readable GGUF files.
        let missing = EmbedParityJob { candidate: dir.join("missing.gguf"), output_dir: dir.join("out2"), ..job.clone() };
        assert!(prepare_embed_parity(missing).is_err());
        // A retrieval threshold without a retrieval split fails before anything runs.
        let no_split = EmbedParityJob {
            output_dir: dir.join("out3"),
            settings: EmbedSettings { queries: None, documents: None, ..job.settings.clone() },
            gate: EmbedGate { min_recall: Some(0.9), ..EmbedGate::default() },
            ..job.clone()
        };
        assert!(prepare_embed_parity(no_split).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
