//! Native CPU-first directional weight ablation on original safetensors shards.
//!
//! This is a mathematical building block, NOT automatic Heretic/ARA, a refusal
//! guarantee, a vision evaluation, or a fully converted HF model. Callers must
//! supply measured directions or local per-module activation samples.
//! Memory: O(largest modified row + its projection), not O(model weights).
//! No Python, internet, GPU, or third-party tensor framework dependencies.

use crate::packed::{decode_float, encode_float};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const HEADER_MAX: u64 = 32 * 1024 * 1024;
const RECIPE_MAX: u64 = 64 * 1024 * 1024;
const MAX_ROW_BYTES: usize = 16 * 1024 * 1024;
const MAX_EDITS: usize = 256;

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, s.into())
}
fn corrupt(s: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, s.into())
}

#[derive(Clone, Debug)]
struct Edit {
    tensor: String,
    offset: u64,
    dtype: String,
    rows: usize,
    cols: usize,
    width: usize,
    direction: Vec<f64>,
    strength: f64,
    lora: Option<LowRank>,
}

#[derive(Clone, Debug)]
struct LowRank {
    a: Vec<Vec<f64>>,
    b: Vec<Vec<f64>>,
}

#[derive(Debug)]
pub struct AblationPlan {
    source: PathBuf,
    destination: PathBuf,
    original_bytes: u64,
    edits: Vec<Edit>,
    preserve_row_norms: bool,
}

impl AblationPlan {
    pub fn as_json(&self) -> Value {
        json!({
            "schema": "ojon.abliteration_plan.v1",
            "source": self.source.to_string_lossy(),
            "output": self.destination.to_string_lossy(),
            "original_bytes": self.original_bytes,
            "edited_tensors": self.edits.iter().map(|e| json!({
                "name": e.tensor, "rows": e.rows, "columns": e.cols,
                "dtype": e.dtype, "strength": e.strength,
                "method": if e.lora.is_some(){"LOW_RANK_DELTA"}else{"DIRECTIONAL_PROJECTION"}
            })).collect::<Vec<_>>(),
            "preserve_row_norms": self.preserve_row_norms,
            "model_inference": "NOT_RUN",
            "refusal_optimization": "NOT_RUN",
            "vision_qa": "NOT_RUN",
            "status": "PLAN_ONLY"
        })
    }
}

fn read_bounded_json(path: &Path, limit: u64) -> io::Result<Value> {
    let len = fs::metadata(path)?.len();
    if len == 0 || len > limit {
        return Err(invalid("JSON input missing or exceeds size bound"));
    }
    let mut buf = Vec::with_capacity(len as usize);
    File::open(path)?.take(limit + 1).read_to_end(&mut buf)?;
    serde_json::from_slice(&buf).map_err(|e| corrupt(format!("invalid JSON: {e}")))
}

fn parse_vector(v: &Value, label: &str) -> io::Result<Vec<f64>> {
    let arr = v
        .as_array()
        .ok_or_else(|| invalid(format!("{label} must be a numeric array")))?;
    if arr.is_empty() || arr.len() > 131072 {
        return Err(invalid(format!("{label} has invalid length")));
    }
    arr.iter()
        .map(|x| {
            let n = x
                .as_f64()
                .ok_or_else(|| invalid(format!("{label} has a nonnumber")))?;
            if !n.is_finite() {
                return Err(invalid(format!("{label} has nonfinite values")));
            }
            Ok(n)
        })
        .collect()
}

/// Difference of good/bad activation means with equal-dimensional samples.
/// This consumes EXTERNALLY CAPTURED data; it does not run a model or collect
/// module activations itself. The sign is immaterial to orthogonal projection.
pub fn direction_from_activations(good: &[Vec<f64>], bad: &[Vec<f64>]) -> io::Result<Vec<f64>> {
    if good.is_empty() || bad.is_empty() || good.len() > 4096 || bad.len() > 4096 {
        return Err(invalid("need bounded nonempty good/bad activation groups"));
    }
    let size = good[0].len();
    if size == 0 || size > 131072 {
        return Err(invalid("invalid activation width"));
    }
    let mut g = vec![0f64; size];
    let mut b = vec![0f64; size];
    for (samples, sum) in [(good, &mut g), (bad, &mut b)] {
        for row in samples {
            if row.len() != size || row.iter().any(|x| !x.is_finite()) {
                return Err(invalid(
                    "activation vectors have mismatched dimensions or nonfinite values",
                ));
            }
            for (s, &x) in sum.iter_mut().zip(row) {
                *s += x / samples.len() as f64;
            }
        }
    }
    let result: Vec<f64> = b.iter().zip(&g).map(|(x, y)| x - y).collect();
    if result.iter().any(|x| !x.is_finite()) {
        return Err(invalid("activation mean overflow"));
    }
    Ok(result)
}

fn parse_activation_samples(v: &Value, label: &str) -> io::Result<Vec<Vec<f64>>> {
    let arr = v
        .as_array()
        .ok_or_else(|| invalid(format!("{label} must be an array of vectors")))?;
    if arr.is_empty() || arr.len() > 4096 {
        return Err(invalid("invalid activation sample count"));
    }
    arr.iter().map(|row| parse_vector(row, label)).collect()
}

fn unit_direction(v: &Value, rows: usize) -> io::Result<Vec<f64>> {
    let direction = if let Some(d) = v.get("direction") {
        if v.get("good_activations").is_some() || v.get("bad_activations").is_some() {
            return Err(invalid(
                "supply either direction OR activation samples, not both",
            ));
        }
        parse_vector(d, "direction")?
    } else {
        let good = parse_activation_samples(
            v.get("good_activations")
                .ok_or_else(|| invalid("missing direction or good_activations"))?,
            "good_activations",
        )?;
        let bad = parse_activation_samples(
            v.get("bad_activations")
                .ok_or_else(|| invalid("missing bad_activations"))?,
            "bad_activations",
        )?;
        direction_from_activations(&good, &bad)?
    };
    if direction.len() != rows {
        return Err(invalid(format!(
            "direction length {} != tensor output rows {rows}",
            direction.len()
        )));
    }
    // f64::hypot avoids overflow during norm accumulation.
    let norm = direction.iter().fold(0.0_f64, |sum, &x| sum.hypot(x));
    if !norm.is_finite() || norm < 1e-30 {
        return Err(invalid("zero, nearly zero or invalid refusal direction"));
    }
    Ok(direction.iter().map(|x| x / norm).collect())
}

pub(crate) fn normalized_direction(value: &Value, rows: usize) -> io::Result<Vec<f64>> {
    unit_direction(value, rows)
}

/// Byte location and shape of one float matrix inside a GGUF tensor payload.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MatrixRegion<'a> {
    pub offset: u64,
    pub rows: usize,
    pub cols: usize,
    pub dtype: &'a str,
}

pub(crate) fn apply_directional_matrix(
    file: &mut File,
    region: MatrixRegion<'_>,
    direction: &[f64],
    strength: f64,
    preserve_row_norms: bool,
) -> io::Result<()> {
    let MatrixRegion {
        offset,
        rows,
        cols,
        dtype,
    } = region;
    if direction.len() != rows
        || !strength.is_finite()
        || !(0.0..=1.0).contains(&strength)
        || strength == 0.0
    {
        return Err(invalid("invalid directional matrix edit"));
    }
    let width = weight_width(dtype)?;
    let row_bytes = cols
        .checked_mul(width)
        .ok_or_else(|| invalid("matrix row overflow"))?;
    if rows == 0 || cols == 0 || rows > 131072 || row_bytes > MAX_ROW_BYTES {
        return Err(invalid("directional matrix exceeds supported dimensions"));
    }
    let edit = Edit {
        tensor: String::new(),
        offset,
        dtype: dtype.to_owned(),
        rows,
        cols,
        width,
        direction: direction.to_vec(),
        strength,
        lora: None,
    };
    apply_edit(file, &edit, preserve_row_norms)
}

/// A = rank x input columns, B = output rows x rank.
fn parse_lora(value: &Value, rows: usize, cols: usize) -> io::Result<LowRank> {
    let a = value
        .get("a")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("LoRA A must be an array of rank rows"))?;
    let b = value
        .get("b")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("LoRA B must be an array of output rows"))?;
    let rank = a.len();
    if rank == 0 || rank > 64 || b.len() != rows {
        return Err(invalid("invalid LoRA rank or B output shape"));
    }
    if (rows + cols)
        .checked_mul(rank)
        .ok_or_else(|| invalid("LoRA dimensions overflow"))?
        > 1_000_000
    {
        return Err(invalid("LoRA adapter exceeds 1 million element limit"));
    }
    let parse_row = |v: &Value, n: usize| -> io::Result<Vec<f64>> {
        let array = v
            .as_array()
            .ok_or_else(|| invalid("LoRA entries must be vectors"))?;
        if array.len() != n {
            return Err(invalid("LoRA matrix dimension mismatch"));
        }
        array
            .iter()
            .map(|x| {
                let n = x
                    .as_f64()
                    .ok_or_else(|| invalid("LoRA element is not numeric"))?;
                if !n.is_finite() {
                    return Err(invalid("LoRA element not finite"));
                }
                Ok(n)
            })
            .collect()
    };
    Ok(LowRank {
        a: a.iter()
            .map(|v| parse_row(v, cols))
            .collect::<io::Result<Vec<_>>>()?,
        b: b.iter()
            .map(|v| parse_row(v, rank))
            .collect::<io::Result<Vec<_>>>()?,
    })
}

fn weight_width(dtype: &str) -> io::Result<usize> {
    match dtype {
        "F16" | "BF16" => Ok(2),
        "F32" => Ok(4),
        _ => Err(invalid(format!(
            "only original F16/BF16/F32 weight edits supported: {dtype}"
        ))),
    }
}
fn safe_target(name: &str) -> bool {
    let n = name.to_ascii_lowercase();
    if [
        "vision",
        "visual",
        "mmproj",
        "projector",
        "image_encoder",
        "vision_tower",
    ]
    .iter()
    .any(|term| n.contains(term))
    {
        return false;
    }
    (n.ends_with(".self_attn.o_proj.weight") || n.ends_with(".mlp.down_proj.weight"))
        && (n.contains(".layers.") || n.starts_with("layers."))
}

/// Plan a targeted projection. Exact tensor names only: never glob image
/// tower/processor tensors and never silently edit unexpected architectures.
pub fn prepare_abliteration(
    source: &Path,
    recipe: &Path,
    output: &Path,
) -> io::Result<AblationPlan> {
    let json = read_bounded_json(recipe, RECIPE_MAX)?;
    prepare_abliteration_with_recipe(source, &json, output)
}

/// Plan shard edits from a pre-validated in-memory recipe. This avoids
/// temporary recipe files when editing a complete multi-shard HF checkpoint.
pub(crate) fn prepare_abliteration_with_recipe(
    source: &Path,
    json: &Value,
    output: &Path,
) -> io::Result<AblationPlan> {
    let source = source.canonicalize()?;
    if !source.is_file() || source.extension().and_then(|s| s.to_str()) != Some("safetensors") {
        return Err(invalid("source must be an original .safetensors file"));
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let destination = parent.canonicalize()?.join(
        output
            .file_name()
            .ok_or_else(|| invalid("missing output filename"))?,
    );
    if destination.extension().and_then(|s| s.to_str()) != Some("safetensors") {
        return Err(invalid("output must end in .safetensors"));
    }
    if destination == source || destination.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "output exists or equals source; refusing overwrite",
        ));
    }
    if json.get("schema").and_then(Value::as_str) != Some("ojon.abliteration.v1") {
        return Err(invalid("expected ojon.abliteration.v1 recipe schema"));
    }
    let preserve_row_norms = json
        .get("preserve_row_norms")
        .and_then(Value::as_bool)
        .ok_or_else(|| invalid("preserve_row_norms must be explicitly true or false"))?;
    let req = json
        .get("edits")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("edits must be an array"))?;
    if req.is_empty() || req.len() > MAX_EDITS {
        return Err(invalid("invalid edit count"));
    }
    let original_bytes = fs::metadata(&source)?.len();
    if original_bytes < 10 {
        return Err(corrupt("truncated safetensors file"));
    }
    let mut file = File::open(&source)?;
    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len == 0 || header_len > HEADER_MAX || header_len > original_bytes - 8 {
        return Err(corrupt("invalid safetensors header size"));
    }
    let mut header = vec![0u8; header_len as usize];
    file.read_exact(&mut header)?;
    let root: Value =
        serde_json::from_slice(&header).map_err(|e| corrupt(format!("bad tensor header: {e}")))?;
    let root = root
        .as_object()
        .ok_or_else(|| corrupt("tensor header must be an object"))?;
    if root.is_empty() || root.len() > 100000 {
        return Err(corrupt("invalid number of tensor descriptors"));
    }
    let payload_len = original_bytes - 8 - header_len;
    let mut intervals: Vec<(u64, u64)> = Vec::new();
    for (name, value) in root {
        if name == "__metadata__" {
            continue;
        }
        let offsets = value
            .get("data_offsets")
            .and_then(Value::as_array)
            .ok_or_else(|| corrupt("tensor missing data_offsets"))?;
        if offsets.len() != 2 {
            return Err(corrupt("invalid tensor offsets"));
        }
        let (a, b) = (
            offsets[0]
                .as_u64()
                .ok_or_else(|| corrupt("invalid offset"))?,
            offsets[1]
                .as_u64()
                .ok_or_else(|| corrupt("invalid offset"))?,
        );
        if a > b || b > payload_len {
            return Err(corrupt("tensor payload out of bounds"));
        }
        intervals.push((a, b));
    }
    intervals.sort_unstable();
    for pair in intervals.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(corrupt("overlapping tensor payloads"));
        }
    }

    let mut names = HashSet::new();
    let mut edits = Vec::new();
    for entry in req {
        let name = entry
            .get("tensor")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("each edit needs tensor name"))?;
        if !names.insert(name.to_owned()) {
            return Err(invalid("duplicate tensor edit"));
        }
        if !safe_target(name) {
            return Err(invalid(format!("unsupported or protected tensor: {name}")));
        }
        let tensor = root
            .get(name)
            .ok_or_else(|| invalid(format!("tensor not present: {name}")))?;
        let dtype = tensor
            .get("dtype")
            .and_then(Value::as_str)
            .ok_or_else(|| corrupt("missing tensor dtype"))?;
        let width = weight_width(dtype)?;
        let shape = tensor
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| corrupt("missing tensor shape"))?;
        if shape.len() != 2 {
            return Err(invalid("abliteration target must be a 2D weight matrix"));
        }
        let rows = usize::try_from(shape[0].as_u64().ok_or_else(|| corrupt("bad rows"))?)
            .map_err(|_| invalid("row dimension too large"))?;
        let cols = usize::try_from(shape[1].as_u64().ok_or_else(|| corrupt("bad columns"))?)
            .map_err(|_| invalid("column dimension too large"))?;
        if rows == 0 || cols == 0 || rows > 131072 {
            return Err(invalid("invalid matrix dimensions"));
        }
        let row_bytes = cols
            .checked_mul(width)
            .ok_or_else(|| invalid("matrix row overflow"))?;
        if row_bytes > MAX_ROW_BYTES {
            return Err(invalid("matrix row exceeds 16 MiB memory bound"));
        }
        let off = tensor
            .get("data_offsets")
            .and_then(Value::as_array)
            .unwrap();
        let start = off[0].as_u64().unwrap();
        let stop = off[1].as_u64().unwrap();
        let length = (rows as u64)
            .checked_mul(row_bytes as u64)
            .ok_or_else(|| corrupt("matrix size overflow"))?;
        if stop - start != length {
            return Err(corrupt("matrix bytes do not match dtype and shape"));
        }
        let strength = entry
            .get("strength")
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid("each edit must specify strength"))?;
        if !strength.is_finite() || !(0.0..=1.0).contains(&strength) || strength == 0.0 {
            return Err(invalid("strength must be > 0 and <= 1"));
        }
        let lora = if let Some(value) = entry.get("lora") {
            if entry.get("direction").is_some()
                || entry.get("good_activations").is_some()
                || entry.get("bad_activations").is_some()
            {
                return Err(invalid("cannot combine LoRA with a directional projection"));
            }
            Some(parse_lora(value, rows, cols)?)
        } else {
            None
        };
        let direction = if lora.is_some() {
            Vec::new()
        } else {
            unit_direction(entry, rows)?
        };
        edits.push(Edit {
            tensor: name.to_owned(),
            offset: start + 8 + header_len,
            dtype: dtype.to_owned(),
            rows,
            cols,
            width,
            direction,
            strength,
            lora,
        });
    }
    Ok(AblationPlan {
        source,
        destination,
        original_bytes,
        edits,
        preserve_row_norms,
    })
}

struct TempFile(PathBuf);
impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}
fn create_stage(parent: &Path) -> io::Result<(TempFile, File)> {
    for retry in 0..16 {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| invalid("invalid system time"))?
            .as_nanos();
        let candidate = parent.join(format!(
            ".ojon-abliterate-{}-{nanos}-{retry}.tmp",
            std::process::id()
        ));
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => return Ok((TempFile(candidate), file)),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(invalid("unable to allocate staging file"))
}

fn read_row(
    file: &mut File,
    edit: &Edit,
    row: usize,
    buf: &mut [u8],
    vals: &mut [f64],
) -> io::Result<()> {
    let off = edit
        .offset
        .checked_add(
            (row as u64)
                .checked_mul(buf.len() as u64)
                .ok_or_else(|| corrupt("row offset overflow"))?,
        )
        .ok_or_else(|| corrupt("row offset overflow"))?;
    file.seek(SeekFrom::Start(off))?;
    file.read_exact(buf)?;
    for (i, chunk) in buf.chunks_exact(edit.width).enumerate() {
        vals[i] = f64::from(decode_float(chunk, &edit.dtype)?);
    }
    Ok(())
}

/// Orthogonal output-direction projection: W' = W - alpha * d * (d^T W)
/// for unit d, followed optionally by ORIGINAL row-L2-norm preservation.
/// Deliberately streams each row twice instead of materializing full matrices.
fn apply_edit(file: &mut File, edit: &Edit, preserve: bool) -> io::Result<()> {
    let row_bytes = edit.cols * edit.width;
    let mut bytes = vec![0u8; row_bytes];
    let mut row = vec![0f64; edit.cols];
    let mut projection = vec![0f64; edit.cols];
    if edit.lora.is_none() {
        for i in 0..edit.rows {
            read_row(file, edit, i, &mut bytes, &mut row)?;
            for (p, &v) in projection.iter_mut().zip(&row) {
                *p += edit.direction[i] * v;
            }
        }
        if projection.iter().any(|v| !v.is_finite()) {
            return Err(invalid("projection overflow"));
        }
    }
    for i in 0..edit.rows {
        read_row(file, edit, i, &mut bytes, &mut row)?;
        let original_norm = if preserve {
            row.iter().fold(0f64, |s, &v| s.hypot(v))
        } else {
            0.0
        };
        if let Some(lora) = &edit.lora {
            for (column, v) in row.iter_mut().enumerate() {
                let mut delta = 0.0_f64;
                for rank in 0..lora.a.len() {
                    delta += lora.b[i][rank] * lora.a[rank][column];
                }
                *v += edit.strength * delta;
            }
        } else {
            for (v, &p) in row.iter_mut().zip(&projection) {
                *v -= edit.strength * edit.direction[i] * p;
            }
        }
        if preserve {
            let new_norm = row.iter().fold(0f64, |s, &v| s.hypot(v));
            if original_norm == 0.0 {
                // Exactly preserve originally zero rows, even if a global
                // projection would otherwise introduce nonzero weights.
                row.fill(0.0);
            } else {
                if new_norm <= 1e-30 {
                    return Err(invalid("cannot preserve nonzero row norm after projection; disable preserve_row_norms or reduce strength"));
                }
                let scale = original_norm / new_norm;
                for value in &mut row {
                    *value *= scale;
                }
            }
        }
        for (chunk, &value) in bytes.chunks_exact_mut(edit.width).zip(&row) {
            if !value.is_finite() || value.abs() > f32::MAX as f64 {
                return Err(invalid("edited weight outside finite f32 range"));
            }
            let encoded = encode_float(value as f32, &edit.dtype)?;
            if !decode_float(&encoded, &edit.dtype)?.is_finite() {
                return Err(invalid("edited weight overflows original tensor dtype"));
            }
            chunk.copy_from_slice(&encoded);
        }
        let off = edit.offset + (i as u64) * (row_bytes as u64);
        file.seek(SeekFrom::Start(off))?;
        file.write_all(&bytes)?;
    }
    Ok(())
}

#[derive(Debug)]
pub struct AblationResult {
    pub destination: PathBuf,
    pub bytes: u64,
    pub edited_tensors: usize,
}
impl AblationResult {
    pub fn as_json(&self) -> Value {
        json!({
            "schema": "ojon.abliteration_result.v1",
            "output": self.destination.to_string_lossy(),
            "bytes": self.bytes,
            "edited_tensors": self.edited_tensors,
            "output_format": "safetensors_shard",
            "refusal_rate": "NOT_EVALUATED",
            "coding_quality": "NOT_EVALUATED",
            "vision_qa": "NOT_EVALUATED",
            "complete_hf_checkpoint": false,
            "status": "TENSOR_EDIT_WRITTEN"
        })
    }
}

/// Copy-on-write transform: never edits original checkpoint, and publishes
/// output only after all requested matrices are successfully processed.
pub fn run_abliteration(
    plan: &AblationPlan,
    mut log: impl FnMut(&str),
) -> io::Result<AblationResult> {
    if plan.destination.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "output already exists",
        ));
    }
    let source_file = File::open(&plan.source)?;
    if source_file.metadata()?.len() != plan.original_bytes {
        return Err(invalid("input safetensors changed size after planning"));
    }
    let (staging, mut stage) = create_stage(
        plan.destination
            .parent()
            .ok_or_else(|| invalid("missing parent"))?,
    )?;
    let copied = io::copy(&mut source_file.take(plan.original_bytes), &mut stage)?;
    if copied != plan.original_bytes {
        return Err(corrupt("short copy of model source"));
    }
    for edit in &plan.edits {
        apply_edit(&mut stage, edit, plan.preserve_row_norms)?;
        log(&format!(
            "[ojon] applied directional projection to {}",
            edit.tensor
        ));
    }
    stage.sync_all()?;
    drop(stage);
    // Hard link is atomic and fails instead of overwriting a concurrent output.
    fs::hard_link(&staging.0, &plan.destination)?;
    Ok(AblationResult {
        destination: plan.destination.clone(),
        bytes: copied,
        edited_tensors: plan.edits.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scratch() -> PathBuf {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        loop {
            let n = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let serial = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "ojon-ablation-test-{}-{n}-{serial}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return path,
                Err(error) if error.kind() == ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("cannot create ablation test directory: {error}"),
            }
        }
    }
    fn fixture(dir: &Path, dtype: &str, matrix: &[f32], other: &[u8]) -> PathBuf {
        let w = weight_width(dtype).unwrap();
        let name = "model.layers.0.self_attn.o_proj.weight";
        let mut h = serde_json::Map::new();
        h.insert(
            name.to_owned(),
            json!({"dtype":dtype,"shape":[2,2],"data_offsets":[0,4*w]}),
        );
        h.insert(
            "vision_tower.weight".into(),
            json!({"dtype":"U8","shape":[other.len()],"data_offsets":[4*w,4*w+other.len()]}),
        );
        h.insert("__metadata__".into(), json!({"test":"preserve"}));
        let header = serde_json::to_vec(&Value::Object(h)).unwrap();
        let path = dir.join("model.safetensors");
        let mut f = File::create(&path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&header).unwrap();
        for x in matrix {
            f.write_all(&encode_float(*x, dtype).unwrap()).unwrap();
        }
        f.write_all(other).unwrap();
        path
    }
    fn recipe(dir: &Path, body: Value, preserve: bool) -> PathBuf {
        let p = dir.join("recipe.json");
        fs::write(&p,serde_json::to_vec(&json!({"schema":"ojon.abliteration.v1","preserve_row_norms":preserve,"edits":[body]})).unwrap()).unwrap();
        p
    }
    fn tensor_bytes(path: &Path) -> Vec<u8> {
        fs::read(path).unwrap()
    }
    #[test]
    fn difference_of_means_correct() {
        let direction = direction_from_activations(
            &[vec![0.0, 2.0], vec![2.0, 2.0]],
            &[vec![3.0, 2.0], vec![5.0, 2.0]],
        )
        .unwrap();
        assert_eq!(direction, vec![3.0, 0.0]);
        assert!(direction_from_activations(&[vec![1.0, 2.0]], &[vec![1.0]]).is_err());
    }
    #[test]
    fn f32_projection_only_changes_target_bytes() {
        let dir = scratch();
        let src = fixture(&dir, "F32", &[2.0, 0.0, 0.0, 2.0], &[11, 12, 13]);
        let cfg = recipe(
            &dir,
            json!({"tensor":"model.layers.0.self_attn.o_proj.weight",
          "direction":[1.0,0.0],"strength":1.0}),
            false,
        );
        let output = dir.join("ablated.safetensors");
        let plan = prepare_abliteration(&src, &cfg, &output).unwrap();
        assert_eq!(plan.as_json()["status"], "PLAN_ONLY");
        let result = run_abliteration(&plan, |_| {}).unwrap();
        assert_eq!(result.edited_tensors, 1);
        let bytes = tensor_bytes(&output);
        let old = tensor_bytes(&src);
        let offset = 8 + u64::from_le_bytes(old[..8].try_into().unwrap()) as usize;
        let edited: Vec<f32> = bytes[offset..offset + 16]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(edited, vec![0.0, 0.0, 0.0, 2.0]);
        assert_eq!(&old[offset + 16..], &bytes[offset + 16..]);
        assert_eq!(&old[..offset], &bytes[..offset]);
        assert!(run_abliteration(&plan, |_| {}).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn bf16_and_f16_projection_and_row_norms() {
        for dtype in ["BF16", "F16"] {
            let dir = scratch();
            let src = fixture(&dir, dtype, &[2.0, 0.0, 0.0, 2.0], &[0xa1]);
            let cfg = recipe(
                &dir,
                json!({"tensor":"model.layers.0.self_attn.o_proj.weight",
                "good_activations":[[0.0,0.0],[0.0,0.0]],
                "bad_activations":[[1.0,1.0],[1.0,1.0]],
                "strength":1.0}),
                true,
            );
            let output = dir.join("edited.safetensors");
            let plan = prepare_abliteration(&src, &cfg, &output).unwrap();
            run_abliteration(&plan, |_| {}).unwrap();
            let bytes = tensor_bytes(&output);
            let offset = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
            let width = weight_width(dtype).unwrap();
            let values: Vec<f64> = bytes[offset..offset + 4 * width]
                .chunks_exact(width)
                .map(|b| decode_float(b, dtype).unwrap() as f64)
                .collect();
            for row in values.chunks_exact(2) {
                let norm = row.iter().fold(0f64, |acc, &v| acc.hypot(v));
                assert!((norm - 2.0).abs() < 0.02, "{dtype} norm={norm}");
            }
            fs::remove_dir_all(dir).unwrap();
        }
    }
    #[test]
    fn failed_norm_preservation_never_publishes_partial_output() {
        let dir = scratch();
        let src = fixture(&dir, "F32", &[2.0, 0.0, 0.0, 2.0], &[19]);
        let before = tensor_bytes(&src);
        let recipe = recipe(
            &dir,
            json!({
                "tensor": "model.layers.0.self_attn.o_proj.weight",
                "direction": [1.0,0.0], "strength": 1.0
            }),
            true,
        );
        let dest = dir.join("must-not-appear.safetensors");
        let plan = prepare_abliteration(&src, &recipe, &dest).unwrap();
        assert!(run_abliteration(&plan, |_| {}).is_err());
        assert!(
            !dest.exists(),
            "failed edit must not publish a partial model"
        );
        assert_eq!(
            tensor_bytes(&src),
            before,
            "original weights must remain unchanged"
        );
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn low_rank_delta_edits_only_named_tensor() {
        let dir = scratch();
        let src = fixture(&dir, "F32", &[2.0, 0.0, 0.0, 2.0], &[0x41, 0x42]);
        let config = recipe(
            &dir,
            json!({
                "tensor":"model.layers.0.self_attn.o_proj.weight",
                "lora":{"a":[[1.0,0.0]],"b":[[1.0],[0.0]]},
                "strength":1.0
            }),
            false,
        );
        let out = dir.join("lowrank.safetensors");
        let plan = prepare_abliteration(&src, &config, &out).unwrap();
        let receipt = plan.as_json();
        assert_eq!(receipt["edited_tensors"][0]["method"], "LOW_RANK_DELTA");
        run_abliteration(&plan, |_| {}).unwrap();
        let bytes = fs::read(&out).unwrap();
        let pos = 8 + u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!(
            f32::from_le_bytes(bytes[pos..pos + 4].try_into().unwrap()),
            3.0
        );
        assert_eq!(&bytes[pos + 16..], &[0x41, 0x42]);
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn guards_unknown_tensor_bad_dims_bad_direction_and_vision() {
        let dir = scratch();
        let src = fixture(&dir, "F32", &[1.0, 0.0, 0.0, 1.0], &[7, 8]);
        let o = dir.join("out.safetensors");
        for body in [
            json!({"tensor":"vision_tower.weight","direction":[1.0,0.0],"strength":1.0}),
            json!({"tensor":"model.layers.0.self_attn.o_proj.weight","direction":[1.0],"strength":1.0}),
            json!({"tensor":"model.layers.0.self_attn.o_proj.weight","direction":[0.0,0.0],"strength":1.0}),
            json!({"tensor":"model.layers.0.self_attn.o_proj.weight","direction":[1.0,0.0],"strength":2.0}),
        ] {
            let cfg = recipe(&dir, body, true);
            assert!(prepare_abliteration(&src, &cfg, &o).is_err());
            assert!(!o.exists());
        }
        fs::remove_dir_all(dir).unwrap();
    }
}
