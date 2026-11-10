//! Verified weight patches on original Hugging Face safetensors checkpoints.
//!
//! A producer (training, editing or refusal-removal tool) supplies a delta safetensors file plus a manifest
//! (`ojon.weight_patch.v1`). Ojon checks both hashes, then writes a NEW model
//! folder: untouched files are copied (or hard-linked on request), edited shards
//! are copied and patched row by row. Memory is O(one row + low-rank B factor),
//! never O(model). The original folder is only read.
//!
//! The output folder carries `ojon-weight-patch-receipt.json` (hashes of the
//! manifest, the delta, every edited tensor before and after, every output file)
//! and `ojon-rollback.safetensors` (the original bytes of every edited tensor),
//! so the patch can be checked (`weight-patch-verify`) and undone exactly
//! (`weight-patch-rollback`) from the output folder alone.
//!
//! This applies a delta someone else measured. It does not discover edits and
//! says nothing about model quality: the receipt status is
//! `PATCH_WRITTEN_NOT_EVALUATED` until an evaluation gate runs.

use crate::packed::{decode_float, encode_float};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, ErrorKind, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const MANIFEST_SCHEMA: &str = "ojon.weight_patch.v1";
pub const RECEIPT_SCHEMA: &str = "ojon.weight_patch_receipt.v1";
pub const RECEIPT_FILE: &str = "ojon-weight-patch-receipt.json";
pub const ROLLBACK_FILE: &str = "ojon-rollback.safetensors";

const HEADER_MAX: u64 = 64 * 1024 * 1024;
const JSON_MAX: u64 = 16 * 1024 * 1024;
const MAX_EDITS: usize = 1024;
const MAX_RANK: usize = 256;
const MAX_ROW_BYTES: usize = 64 * 1024 * 1024;
const MAX_B_BYTES: usize = 256 * 1024 * 1024;
const MAX_FILES: usize = 4096;

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, s.into())
}
fn corrupt(s: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, s.into())
}

fn width(dtype: &str) -> io::Result<usize> {
    match dtype {
        "F16" | "BF16" => Ok(2),
        "F32" => Ok(4),
        _ => Err(invalid(format!("only F16/BF16/F32 tensors can be patched, found {dtype}"))),
    }
}

/// Location of one tensor inside a safetensors file.
#[derive(Clone, Debug)]
pub struct TensorLoc {
    pub file: PathBuf,
    /// Absolute byte offset of the tensor payload in `file`.
    pub offset: u64,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: u64,
}

impl TensorLoc {
    /// Matrix view: 2-D tensors as (rows, cols); 1-D as one row.
    fn matrix(&self) -> io::Result<(usize, usize)> {
        match self.shape.as_slice() {
            [n] => Ok((1, *n)),
            [r, c] => Ok((*r, *c)),
            _ => Err(invalid(format!("only 1-D or 2-D tensors can be patched, shape {:?}", self.shape))),
        }
    }
}

/// Parse a safetensors header with bounds and overlap checks.
pub fn read_safetensors_header(path: &Path) -> io::Result<BTreeMap<String, TensorLoc>> {
    let len = fs::metadata(path)?.len();
    if len < 10 {
        return Err(corrupt(format!("truncated safetensors file {}", path.display())));
    }
    let mut f = File::open(path)?;
    let mut prefix = [0u8; 8];
    f.read_exact(&mut prefix)?;
    let hlen = u64::from_le_bytes(prefix);
    if hlen == 0 || hlen > HEADER_MAX || hlen > len - 8 {
        return Err(corrupt(format!("invalid safetensors header size in {}", path.display())));
    }
    let mut header = vec![0u8; hlen as usize];
    f.read_exact(&mut header)?;
    let root: Value = serde_json::from_slice(&header).map_err(|e| corrupt(format!("bad safetensors header: {e}")))?;
    let root = root.as_object().ok_or_else(|| corrupt("safetensors header must be an object"))?;
    let base = 8 + hlen;
    let payload = len - base;
    let mut out = BTreeMap::new();
    let mut spans = Vec::new();
    for (name, v) in root {
        if name == "__metadata__" {
            continue;
        }
        let dtype = v.get("dtype").and_then(Value::as_str).ok_or_else(|| corrupt("tensor missing dtype"))?.to_owned();
        let shape: Vec<usize> = v
            .get("shape")
            .and_then(Value::as_array)
            .ok_or_else(|| corrupt("tensor missing shape"))?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize).ok_or_else(|| corrupt("bad shape")))
            .collect::<io::Result<_>>()?;
        let off = v.get("data_offsets").and_then(Value::as_array).ok_or_else(|| corrupt("tensor missing data_offsets"))?;
        if off.len() != 2 {
            return Err(corrupt("invalid data_offsets"));
        }
        let (a, b) = (
            off[0].as_u64().ok_or_else(|| corrupt("bad offset"))?,
            off[1].as_u64().ok_or_else(|| corrupt("bad offset"))?,
        );
        if a > b || b > payload {
            return Err(corrupt(format!("tensor {name} payload out of bounds")));
        }
        if let Ok(w) = width(&dtype) {
            let elems = shape.iter().try_fold(1u64, |acc, &d| acc.checked_mul(d as u64)).ok_or_else(|| corrupt("shape overflow"))?;
            if elems * w as u64 != b - a {
                return Err(corrupt(format!("tensor {name} size does not match shape and dtype")));
            }
        }
        spans.push((a, b));
        out.insert(name.clone(), TensorLoc { file: path.to_path_buf(), offset: base + a, dtype, shape, bytes: b - a });
    }
    spans.sort_unstable();
    if spans.windows(2).any(|w| w[1].0 < w[0].1) {
        return Err(corrupt("overlapping tensor payloads"));
    }
    Ok(out)
}

fn read_json(path: &Path) -> io::Result<(Value, Vec<u8>)> {
    let len = fs::metadata(path)?.len();
    if len > JSON_MAX {
        return Err(invalid(format!("{} is larger than {JSON_MAX} bytes", path.display())));
    }
    let bytes = fs::read(path)?;
    let v = serde_json::from_slice(&bytes).map_err(|e| invalid(format!("invalid JSON {}: {e}", path.display())))?;
    Ok((v, bytes))
}

pub fn sha256_file(path: &Path) -> io::Result<String> {
    let mut f = File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// SHA-256 of one tensor's raw payload bytes (streamed).
pub fn sha256_tensor(loc: &TensorLoc) -> io::Result<String> {
    let mut f = File::open(&loc.file)?;
    f.seek(SeekFrom::Start(loc.offset))?;
    let mut h = Sha256::new();
    let mut left = loc.bytes;
    let mut buf = vec![0u8; 8 << 20];
    while left > 0 {
        let n = left.min(buf.len() as u64) as usize;
        f.read_exact(&mut buf[..n])?;
        h.update(&buf[..n]);
        left -= n as u64;
    }
    Ok(hex(&h.finalize()))
}

/// Map every tensor name in a model folder to its location (index-aware).
fn model_tensors(dir: &Path) -> io::Result<BTreeMap<String, TensorLoc>> {
    let mut shards: BTreeSet<PathBuf> = BTreeSet::new();
    let index = dir.join("model.safetensors.index.json");
    if index.is_file() {
        let (v, _) = read_json(&index)?;
        let map = v.get("weight_map").and_then(Value::as_object).ok_or_else(|| corrupt("index.json without weight_map"))?;
        for shard in map.values() {
            let s = shard.as_str().ok_or_else(|| corrupt("bad weight_map entry"))?;
            if s.contains(['/', '\\']) || s.contains("..") {
                return Err(corrupt(format!("unsafe shard name {s}")));
            }
            shards.insert(dir.join(s));
        }
    } else {
        for e in fs::read_dir(dir)? {
            let p = e?.path();
            if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                shards.insert(p);
            }
        }
    }
    if shards.is_empty() {
        return Err(invalid("model folder has no safetensors shards"));
    }
    let mut all = BTreeMap::new();
    for s in shards {
        for (name, loc) in read_safetensors_header(&s)? {
            if all.insert(name.clone(), loc).is_some() {
                return Err(corrupt(format!("tensor {name} appears in more than one shard")));
            }
        }
    }
    Ok(all)
}

#[derive(Clone, Debug)]
enum DeltaKind {
    Full(TensorLoc),
    LowRank { a: TensorLoc, b: TensorLoc, rank: usize },
}

#[derive(Clone, Debug)]
struct PlannedEdit {
    tensor: String,
    target: TensorLoc,
    delta: DeltaKind,
    scale: f64,
}

#[derive(Debug)]
pub struct PatchPlan {
    source: PathBuf,
    output: PathBuf,
    manifest: Value,
    manifest_sha256: String,
    delta_path: PathBuf,
    delta_sha256: String,
    edits: Vec<PlannedEdit>,
    files: Vec<PathBuf>,
    link_unchanged: bool,
}

impl PatchPlan {
    pub fn as_json(&self) -> Value {
        let edits: Vec<Value> = self
            .edits
            .iter()
            .map(|e| {
                json!({"tensor": e.tensor, "shard": e.target.file.file_name().map(|s| s.to_string_lossy().to_string()),
                       "dtype": e.target.dtype, "shape": e.target.shape, "scale": e.scale,
                       "op": match &e.delta { DeltaKind::Full(_) => "add".to_owned(), DeltaKind::LowRank{rank, ..} => format!("add_low_rank(r={rank})") }})
            })
            .collect();
        json!({"schema": "ojon.weight_patch_plan.v1", "source": self.source, "output": self.output,
               "manifest_sha256": self.manifest_sha256, "delta_sha256": self.delta_sha256,
               "edits": edits, "files": self.files.len(), "link_unchanged": self.link_unchanged,
               "status": "PLAN_ONLY"})
    }
}

fn canonical_output(output: &Path) -> io::Result<PathBuf> {
    let parent = output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let name = output.file_name().ok_or_else(|| invalid("output folder needs a name"))?;
    Ok(parent.canonicalize()?.join(name))
}

/// Validate manifest, delta file and target model; nothing is written.
pub fn prepare_weight_patch(source: &Path, manifest_path: &Path, output: &Path, link_unchanged: bool) -> io::Result<PatchPlan> {
    let source = source.canonicalize()?;
    if !source.is_dir() {
        return Err(invalid("source must be a Hugging Face model folder"));
    }
    let output = canonical_output(output)?;
    if output.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "output exists; refusing overwrite"));
    }
    if output.starts_with(&source) {
        return Err(invalid("output must not be inside the source folder"));
    }
    let manifest_path = manifest_path.canonicalize()?;
    let (manifest, raw) = read_json(&manifest_path)?;
    if manifest.get("schema").and_then(Value::as_str) != Some(MANIFEST_SCHEMA) {
        return Err(invalid(format!("manifest schema must be {MANIFEST_SCHEMA}")));
    }
    let manifest_sha256 = hex(&Sha256::digest(&raw));
    let delta_rel = manifest.get("delta_file").and_then(Value::as_str).ok_or_else(|| invalid("manifest needs delta_file"))?;
    let delta_path = manifest_path.parent().unwrap_or(Path::new(".")).join(delta_rel).canonicalize()?;
    let want = manifest.get("delta_sha256").and_then(Value::as_str).ok_or_else(|| invalid("manifest needs delta_sha256"))?;
    let delta_sha256 = sha256_file(&delta_path)?;
    if !want.eq_ignore_ascii_case(&delta_sha256) {
        return Err(invalid(format!("delta_sha256 mismatch: manifest {want}, file {delta_sha256}")));
    }
    let deltas = read_safetensors_header(&delta_path)?;
    let tensors = model_tensors(&source)?;
    let req = manifest.get("edits").and_then(Value::as_array).ok_or_else(|| invalid("manifest needs edits array"))?;
    if req.is_empty() || req.len() > MAX_EDITS {
        return Err(invalid("edit count must be 1..=1024"));
    }
    let mut seen = BTreeSet::new();
    let mut edits = Vec::new();
    for e in req {
        let name = e.get("tensor").and_then(Value::as_str).ok_or_else(|| invalid("edit needs tensor"))?;
        if !seen.insert(name.to_owned()) {
            return Err(invalid(format!("tensor {name} edited twice")));
        }
        let target = tensors.get(name).ok_or_else(|| invalid(format!("tensor {name} not in source model")))?.clone();
        width(&target.dtype)?;
        let (rows, cols) = target.matrix()?;
        if cols * width(&target.dtype)? > MAX_ROW_BYTES {
            return Err(invalid(format!("row of {name} too large to stream")));
        }
        let scale = e.get("scale").and_then(Value::as_f64).unwrap_or(1.0);
        if !scale.is_finite() || scale == 0.0 {
            return Err(invalid("scale must be finite and non-zero"));
        }
        let get = |key: &str| -> io::Result<TensorLoc> {
            let n = e.get(key).and_then(Value::as_str).ok_or_else(|| invalid(format!("edit {name} needs {key}")))?;
            let loc = deltas.get(n).ok_or_else(|| invalid(format!("delta tensor {n} missing from delta file")))?.clone();
            width(&loc.dtype)?;
            Ok(loc)
        };
        let op = e.get("op").and_then(Value::as_str).unwrap_or("add");
        let delta = match op {
            "add" => {
                let d = get("delta")?;
                if d.shape != target.shape {
                    return Err(invalid(format!("delta for {name} has shape {:?}, target {:?}", d.shape, target.shape)));
                }
                DeltaKind::Full(d)
            }
            "add_low_rank" => {
                let (a, b) = (get("a")?, get("b")?);
                let (ar, ak) = a.matrix()?;
                let (bk, bc) = b.matrix()?;
                if a.shape.len() != 2 || b.shape.len() != 2 || ar != rows || bc != cols || ak != bk || ak == 0 || ak > MAX_RANK {
                    return Err(invalid(format!("low-rank factors for {name} must be A[{rows},r] B[r,{cols}] with 1<=r<={MAX_RANK}")));
                }
                if ak * bc * 8 > MAX_B_BYTES {
                    return Err(invalid("low-rank B factor too large to hold"));
                }
                DeltaKind::LowRank { a, b, rank: ak }
            }
            other => return Err(invalid(format!("unknown op {other}; use add or add_low_rank"))),
        };
        if let Some(expect) = e.get("source_sha256").and_then(Value::as_str) {
            let got = sha256_tensor(&target)?;
            if !expect.eq_ignore_ascii_case(&got) {
                return Err(invalid(format!("source tensor {name} hash {got} differs from manifest {expect}: wrong base model")));
            }
        }
        edits.push(PlannedEdit { tensor: name.to_owned(), target, delta, scale });
    }
    let mut files = Vec::new();
    for e in fs::read_dir(&source)? {
        let e = e?;
        if e.file_type()?.is_file() {
            files.push(PathBuf::from(e.file_name()));
        }
    }
    if files.len() > MAX_FILES {
        return Err(invalid("too many files in source folder"));
    }
    files.sort();
    if files.iter().any(|f| f == Path::new(RECEIPT_FILE) || f == Path::new(ROLLBACK_FILE)) {
        return Err(invalid("source already carries an Ojon patch receipt; patch its own original instead"));
    }
    Ok(PatchPlan { source, output, manifest, manifest_sha256, delta_path, delta_sha256, edits, files, link_unchanged })
}

struct Staging(PathBuf, bool);
impl Drop for Staging {
    fn drop(&mut self) {
        if !self.1 {
            let _ = fs::remove_dir_all(&self.0);
        }
    }
}
fn staging_dir(parent: &Path) -> io::Result<Staging> {
    for retry in 0..16 {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_err(|_| invalid("bad clock"))?.as_nanos();
        let p = parent.join(format!(".ojon-weight-patch-{}-{nanos}-{retry}", std::process::id()));
        match fs::create_dir(&p) {
            Ok(()) => return Ok(Staging(p, false)),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(invalid("cannot create staging folder"))
}

fn read_f64_row(f: &mut File, loc: &TensorLoc, row: usize, cols: usize, buf: &mut Vec<u8>, out: &mut [f64]) -> io::Result<()> {
    let w = width(&loc.dtype)?;
    buf.resize(cols * w, 0);
    f.seek(SeekFrom::Start(loc.offset + (row * cols * w) as u64))?;
    f.read_exact(buf)?;
    for (o, c) in out.iter_mut().zip(buf.chunks_exact(w)) {
        *o = f64::from(decode_float(c, &loc.dtype)?);
    }
    Ok(())
}

struct EditStats {
    source_sha256: String,
    output_sha256: String,
    elements: u64,
    elements_changed: u64,
    max_abs_change: f64,
    sum_sq_change: f64,
    sum_sq_source: f64,
}

/// Patch one tensor in the staged shard; also streams the original bytes into `rollback`.
fn apply_one(stage: &mut File, edit: &PlannedEdit, delta_file: &mut File, rollback: &mut impl Write) -> io::Result<EditStats> {
    let (rows, cols) = edit.target.matrix()?;
    let w = width(&edit.target.dtype)?;
    let row_bytes = cols * w;
    let b_factor: Option<Vec<f64>> = match &edit.delta {
        DeltaKind::LowRank { b, rank, .. } => {
            let mut all = vec![0f64; rank * cols];
            let mut tmp = Vec::new();
            for k in 0..*rank {
                read_f64_row(delta_file, b, k, cols, &mut tmp, &mut all[k * cols..(k + 1) * cols])?;
            }
            Some(all)
        }
        DeltaKind::Full(_) => None,
    };
    let (mut src_h, mut out_h) = (Sha256::new(), Sha256::new());
    let mut raw = vec![0u8; row_bytes];
    let mut vals = vec![0f64; cols];
    let mut dvals = vec![0f64; cols];
    let mut dbuf = Vec::new();
    let mut st = EditStats { source_sha256: String::new(), output_sha256: String::new(), elements: (rows * cols) as u64,
                             elements_changed: 0, max_abs_change: 0.0, sum_sq_change: 0.0, sum_sq_source: 0.0 };
    for i in 0..rows {
        let off = edit.target.offset + (i * row_bytes) as u64;
        stage.seek(SeekFrom::Start(off))?;
        stage.read_exact(&mut raw)?;
        src_h.update(&raw);
        rollback.write_all(&raw)?;
        for (v, c) in vals.iter_mut().zip(raw.chunks_exact(w)) {
            *v = f64::from(decode_float(c, &edit.target.dtype)?);
        }
        match &edit.delta {
            DeltaKind::Full(d) => read_f64_row(delta_file, d, i, cols, &mut dbuf, &mut dvals)?,
            DeltaKind::LowRank { a, rank, .. } => {
                let mut arow = vec![0f64; *rank];
                read_f64_row(delta_file, a, i, *rank, &mut dbuf, &mut arow)?;
                let bf = b_factor.as_ref().expect("low-rank B loaded");
                dvals.fill(0.0);
                for (k, &ak) in arow.iter().enumerate() {
                    if ak != 0.0 {
                        for (d, &bv) in dvals.iter_mut().zip(&bf[k * cols..(k + 1) * cols]) {
                            *d += ak * bv;
                        }
                    }
                }
            }
        }
        for ((chunk, v), d) in raw.chunks_exact_mut(w).zip(&vals).zip(&dvals) {
            let new = v + edit.scale * d;
            if !new.is_finite() || new.abs() > f32::MAX as f64 {
                return Err(invalid(format!("patched value of {} is not finite", edit.tensor)));
            }
            let enc = encode_float(new as f32, &edit.target.dtype)?;
            let stored = f64::from(decode_float(&enc, &edit.target.dtype)?);
            let change = stored - v;
            if change != 0.0 {
                st.elements_changed += 1;
            }
            st.max_abs_change = st.max_abs_change.max(change.abs());
            st.sum_sq_change += change * change;
            st.sum_sq_source += v * v;
            chunk.copy_from_slice(&enc);
        }
        out_h.update(&raw);
        stage.seek(SeekFrom::Start(off))?;
        stage.write_all(&raw)?;
    }
    st.source_sha256 = hex(&src_h.finalize());
    st.output_sha256 = hex(&out_h.finalize());
    Ok(st)
}

/// Build a safetensors header for the rollback file (original dtype/shape of each edited tensor).
fn rollback_header(edits: &[PlannedEdit]) -> io::Result<Vec<u8>> {
    let mut m = Map::new();
    let mut off = 0u64;
    for e in edits {
        m.insert(e.tensor.clone(), json!({"dtype": e.target.dtype, "shape": e.target.shape, "data_offsets": [off, off + e.target.bytes]}));
        off += e.target.bytes;
    }
    m.insert("__metadata__".into(), json!({"format": "pt", "ojon": "original bytes of patched tensors"}));
    let mut h = serde_json::to_vec(&Value::Object(m))?;
    while h.len() % 8 != 0 {
        h.push(b' ');
    }
    Ok(h)
}

fn copy_or_link(src: &Path, dst: &Path, link: bool) -> io::Result<()> {
    if link && fs::hard_link(src, dst).is_ok() {
        return Ok(());
    }
    fs::copy(src, dst).map(|_| ())
}

/// Write the patched model folder; publishes it only after every step succeeded.
pub fn run_weight_patch(plan: &PatchPlan, mut log: impl FnMut(&str)) -> io::Result<Value> {
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "output exists"));
    }
    let mut stage = staging_dir(plan.output.parent().ok_or_else(|| invalid("output needs a parent"))?)?;
    let edited_shards: BTreeSet<PathBuf> = plan.edits.iter().map(|e| e.target.file.clone()).collect();
    for f in &plan.files {
        let (src, dst) = (plan.source.join(f), stage.0.join(f));
        if edited_shards.contains(&src) {
            fs::copy(&src, &dst)?;
        } else {
            copy_or_link(&src, &dst, plan.link_unchanged)?;
        }
    }
    log(&format!("[ojon] staged {} files; patching {} tensors", plan.files.len(), plan.edits.len()));
    let rb_path = stage.0.join(ROLLBACK_FILE);
    let mut rb = BufWriter::new(File::create(&rb_path)?);
    let header = rollback_header(&plan.edits)?;
    rb.write_all(&(header.len() as u64).to_le_bytes())?;
    rb.write_all(&header)?;
    let mut delta_file = File::open(&plan.delta_path)?;
    let mut edits_out = Vec::new();
    let (mut changed, mut total) = (0u64, 0u64);
    for e in &plan.edits {
        let rel = e.target.file.file_name().ok_or_else(|| invalid("bad shard name"))?;
        let mut shard = OpenOptions::new().read(true).write(true).open(stage.0.join(rel))?;
        let st = apply_one(&mut shard, e, &mut delta_file, &mut rb)?;
        shard.sync_all()?;
        changed += st.elements_changed;
        total += st.elements;
        let rms_rel = if st.sum_sq_source > 0.0 { (st.sum_sq_change / st.sum_sq_source).sqrt() } else { 0.0 };
        log(&format!("[ojon] patched {} ({} of {} values changed after {} rounding)", e.tensor, st.elements_changed, st.elements, e.target.dtype));
        edits_out.push(json!({
            "tensor": e.tensor, "shard": rel.to_string_lossy(), "dtype": e.target.dtype, "shape": e.target.shape,
            "op": match &e.delta { DeltaKind::Full(_) => json!("add"), DeltaKind::LowRank{rank, ..} => json!({"add_low_rank": rank}) },
            "scale": e.scale, "source_sha256": st.source_sha256, "output_sha256": st.output_sha256,
            "elements": st.elements, "elements_changed": st.elements_changed,
            "max_abs_change": st.max_abs_change, "relative_rms_change": rms_rel,
        }));
    }
    rb.flush()?;
    rb.into_inner().map_err(|e| e.into_error())?.sync_all()?;
    // The original must still be byte-identical where we read it.
    for (e, out) in plan.edits.iter().zip(&edits_out) {
        if sha256_tensor(&e.target)? != out["source_sha256"].as_str().unwrap_or_default() {
            return Err(corrupt(format!("source tensor {} changed during the patch", e.tensor)));
        }
    }
    let mut files = Vec::new();
    for f in plan.files.iter().map(PathBuf::as_path).chain([Path::new(ROLLBACK_FILE)]) {
        let p = stage.0.join(f);
        files.push(json!({"file": f.to_string_lossy(), "bytes": fs::metadata(&p)?.len(), "sha256": sha256_file(&p)?}));
    }
    let receipt = json!({
        "schema": RECEIPT_SCHEMA,
        "tool": format!("ojon {}", env!("CARGO_PKG_VERSION")),
        "source": plan.source, "output": plan.output,
        "manifest_sha256": plan.manifest_sha256, "delta_sha256": plan.delta_sha256,
        "producer": plan.manifest.get("producer").cloned().unwrap_or(Value::Null),
        "evidence": plan.manifest.get("evidence").cloned().unwrap_or(Value::Null),
        "edits": edits_out, "elements_changed": changed, "elements_in_edited_tensors": total,
        "files": files, "rollback_file": ROLLBACK_FILE, "link_unchanged": plan.link_unchanged,
        "original_untouched": true, "model_load": "NOT_TESTED", "quality": "NOT_EVALUATED",
        "status": if changed == 0 { "PATCH_HAD_NO_EFFECT_AFTER_ROUNDING" } else { "PATCH_WRITTEN_NOT_EVALUATED" },
    });
    fs::write(stage.0.join(RECEIPT_FILE), serde_json::to_vec_pretty(&receipt)?)?;
    fs::rename(&stage.0, &plan.output)?;
    stage.1 = true;
    Ok(receipt)
}

fn load_receipt(dir: &Path) -> io::Result<Value> {
    let (v, _) = read_json(&dir.join(RECEIPT_FILE))?;
    if v.get("schema").and_then(Value::as_str) != Some(RECEIPT_SCHEMA) {
        return Err(invalid("not an Ojon weight-patch output folder"));
    }
    Ok(v)
}

/// Re-hash every output file and every patched tensor against the receipt.
pub fn verify_weight_patch(dir: &Path) -> io::Result<Value> {
    let r = load_receipt(dir)?;
    let mut bad = Vec::new();
    for f in r["files"].as_array().ok_or_else(|| corrupt("receipt without files"))? {
        let name = f["file"].as_str().unwrap_or_default();
        let got = sha256_file(&dir.join(name))?;
        if Some(got.as_str()) != f["sha256"].as_str() {
            bad.push(json!({"file": name, "expected": f["sha256"], "found": got}));
        }
    }
    for e in r["edits"].as_array().ok_or_else(|| corrupt("receipt without edits"))? {
        let shard = dir.join(e["shard"].as_str().unwrap_or_default());
        let name = e["tensor"].as_str().unwrap_or_default();
        let loc = read_safetensors_header(&shard)?.remove(name).ok_or_else(|| corrupt(format!("{name} missing from output")))?;
        let got = sha256_tensor(&loc)?;
        if Some(got.as_str()) != e["output_sha256"].as_str() {
            bad.push(json!({"tensor": name, "expected": e["output_sha256"], "found": got}));
        }
    }
    Ok(json!({"schema": "ojon.weight_patch_verify.v1", "folder": dir, "mismatches": bad,
              "status": if bad.is_empty() { "VERIFIED" } else { "MISMATCH" }}))
}

/// Write a new folder equal to the patch's original: restores every patched
/// tensor from the rollback file and proves each hash equals the receipt's source hash.
pub fn rollback_weight_patch(dir: &Path, output: &Path) -> io::Result<Value> {
    let dir = dir.canonicalize()?;
    let r = load_receipt(&dir)?;
    let output = canonical_output(output)?;
    if output.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "output exists; refusing overwrite"));
    }
    let mut stage = staging_dir(output.parent().ok_or_else(|| invalid("output needs a parent"))?)?;
    let originals = read_safetensors_header(&dir.join(ROLLBACK_FILE))?;
    for e in fs::read_dir(&dir)? {
        let e = e?;
        let n = e.file_name();
        if e.file_type()?.is_file() && n != RECEIPT_FILE && n != ROLLBACK_FILE {
            fs::copy(e.path(), stage.0.join(&n))?;
        }
    }
    let mut restored = Vec::new();
    let mut buf = vec![0u8; 8 << 20];
    for e in r["edits"].as_array().ok_or_else(|| corrupt("receipt without edits"))? {
        let name = e["tensor"].as_str().unwrap_or_default();
        let shard = stage.0.join(e["shard"].as_str().unwrap_or_default());
        let dst = read_safetensors_header(&shard)?.remove(name).ok_or_else(|| corrupt(format!("{name} missing")))?;
        let src = originals.get(name).ok_or_else(|| corrupt(format!("{name} missing from rollback file")))?;
        if src.bytes != dst.bytes {
            return Err(corrupt(format!("rollback size mismatch for {name}")));
        }
        let (mut fi, mut fo) = (File::open(&src.file)?, OpenOptions::new().write(true).open(&shard)?);
        fi.seek(SeekFrom::Start(src.offset))?;
        fo.seek(SeekFrom::Start(dst.offset))?;
        let mut left = src.bytes;
        while left > 0 {
            let n = left.min(buf.len() as u64) as usize;
            fi.read_exact(&mut buf[..n])?;
            fo.write_all(&buf[..n])?;
            left -= n as u64;
        }
        fo.sync_all()?;
        let got = sha256_tensor(&TensorLoc { file: shard.clone(), ..dst })?;
        let exact = Some(got.as_str()) == e["source_sha256"].as_str();
        if !exact {
            return Err(corrupt(format!("restored {name} does not match the original hash")));
        }
        restored.push(json!({"tensor": name, "sha256": got}));
    }
    fs::rename(&stage.0, &output)?;
    stage.1 = true;
    Ok(json!({"schema": "ojon.weight_patch_rollback.v1", "patched": dir, "output": output,
              "restored": restored, "status": "ROLLBACK_EXACT"}))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let n = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let p = std::env::temp_dir().join(format!("ojon-wpatch-{tag}-{}-{n}", std::process::id()));
        fs::create_dir_all(&p).unwrap();
        p
    }

    /// Write a safetensors file from (name, dtype, shape, f32 values).
    fn write_st(path: &Path, tensors: &[(&str, &str, Vec<usize>, Vec<f32>)]) {
        let mut m = Map::new();
        let mut payload = Vec::new();
        for (name, dtype, shape, vals) in tensors {
            let start = payload.len() as u64;
            for v in vals {
                payload.extend(encode_float(*v, dtype).unwrap());
            }
            m.insert((*name).into(), json!({"dtype": dtype, "shape": shape, "data_offsets": [start, payload.len() as u64]}));
        }
        let mut h = serde_json::to_vec(&Value::Object(m)).unwrap();
        while h.len() % 8 != 0 {
            h.push(b' ');
        }
        let mut f = File::create(path).unwrap();
        f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&h).unwrap();
        f.write_all(&payload).unwrap();
    }

    fn read_vals(path: &Path, name: &str) -> Vec<f32> {
        let loc = read_safetensors_header(path).unwrap().remove(name).unwrap();
        let w = width(&loc.dtype).unwrap();
        let mut f = File::open(path).unwrap();
        f.seek(SeekFrom::Start(loc.offset)).unwrap();
        let mut b = vec![0u8; loc.bytes as usize];
        f.read_exact(&mut b).unwrap();
        b.chunks_exact(w).map(|c| decode_float(c, &loc.dtype).unwrap()).collect()
    }

    fn model(dir: &Path) {
        write_st(&dir.join("model.safetensors"), &[
            ("layers.0.mlp.down_proj.weight", "BF16", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("layers.0.norm.weight", "F32", vec![3], vec![1.0, 1.0, 1.0]),
            ("embed.weight", "BF16", vec![2, 2], vec![0.5, 0.5, 0.5, 0.5]),
        ]);
        fs::write(dir.join("config.json"), b"{}").unwrap();
    }

    fn manifest(dir: &Path, delta: &Path, edits: Value) -> PathBuf {
        let m = json!({"schema": MANIFEST_SCHEMA, "delta_file": delta.file_name().unwrap().to_string_lossy(),
                       "delta_sha256": sha256_file(delta).unwrap(), "producer": {"name": "test"}, "edits": edits});
        let p = dir.join("patch.json");
        fs::write(&p, serde_json::to_vec(&m).unwrap()).unwrap();
        p
    }

    #[test]
    fn full_and_low_rank_patch_verify_and_exact_rollback() {
        let root = scratch("ok");
        let src = root.join("src");
        fs::create_dir(&src).unwrap();
        model(&src);
        let before = fs::read(src.join("model.safetensors")).unwrap();
        let pdir = root.join("patch");
        fs::create_dir(&pdir).unwrap();
        let delta = pdir.join("delta.safetensors");
        write_st(&delta, &[
            ("d_norm", "F32", vec![3], vec![0.5, 0.0, -0.25]),
            ("a", "F32", vec![2, 1], vec![1.0, 2.0]),
            ("b", "F32", vec![1, 3], vec![1.0, 0.0, -1.0]),
        ]);
        let m = manifest(&pdir, &delta, json!([
            {"tensor": "layers.0.norm.weight", "op": "add", "delta": "d_norm"},
            {"tensor": "layers.0.mlp.down_proj.weight", "op": "add_low_rank", "a": "a", "b": "b", "scale": 0.5},
        ]));
        let out = root.join("out");
        let plan = prepare_weight_patch(&src, &m, &out, false).unwrap();
        let r = run_weight_patch(&plan, |_| {}).unwrap();
        assert_eq!(r["status"], "PATCH_WRITTEN_NOT_EVALUATED");
        assert_eq!(read_vals(&out.join("model.safetensors"), "layers.0.norm.weight"), vec![1.5, 1.0, 0.75]);
        // down += 0.5 * [[1,0,-1],[2,0,-2]]
        assert_eq!(read_vals(&out.join("model.safetensors"), "layers.0.mlp.down_proj.weight"), vec![1.5, 2.0, 2.5, 5.0, 5.0, 5.0]);
        assert_eq!(read_vals(&out.join("model.safetensors"), "embed.weight"), vec![0.5; 4]);
        assert_eq!(fs::read(src.join("model.safetensors")).unwrap(), before, "original untouched");
        assert!(out.join("config.json").is_file());
        assert_eq!(verify_weight_patch(&out).unwrap()["status"], "VERIFIED");
        let back = root.join("back");
        let rb = rollback_weight_patch(&out, &back).unwrap();
        assert_eq!(rb["status"], "ROLLBACK_EXACT");
        assert_eq!(fs::read(back.join("model.safetensors")).unwrap(), before);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn rejects_wrong_delta_hash_wrong_base_bad_shape_and_overwrite() {
        let root = scratch("bad");
        let src = root.join("src");
        fs::create_dir(&src).unwrap();
        model(&src);
        let pdir = root.join("patch");
        fs::create_dir(&pdir).unwrap();
        let delta = pdir.join("delta.safetensors");
        write_st(&delta, &[("d", "F32", vec![3], vec![1.0, 1.0, 1.0]), ("d2", "F32", vec![2], vec![1.0, 1.0])]);
        let out = root.join("out");
        // tampered delta hash
        let m = manifest(&pdir, &delta, json!([{"tensor": "layers.0.norm.weight", "delta": "d"}]));
        let mut v: Value = serde_json::from_slice(&fs::read(&m).unwrap()).unwrap();
        v["delta_sha256"] = json!("00");
        fs::write(&m, serde_json::to_vec(&v).unwrap()).unwrap();
        assert!(prepare_weight_patch(&src, &m, &out, false).unwrap_err().to_string().contains("delta_sha256 mismatch"));
        // wrong base model
        let m = manifest(&pdir, &delta, json!([{"tensor": "layers.0.norm.weight", "delta": "d", "source_sha256": "ab"}]));
        assert!(prepare_weight_patch(&src, &m, &out, false).unwrap_err().to_string().contains("wrong base model"));
        // shape mismatch
        let m = manifest(&pdir, &delta, json!([{"tensor": "layers.0.norm.weight", "delta": "d2"}]));
        assert!(prepare_weight_patch(&src, &m, &out, false).is_err());
        // refuse overwrite
        fs::create_dir(&out).unwrap();
        let m = manifest(&pdir, &delta, json!([{"tensor": "layers.0.norm.weight", "delta": "d"}]));
        assert_eq!(prepare_weight_patch(&src, &m, &out, false).unwrap_err().kind(), ErrorKind::AlreadyExists);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn tiny_delta_lost_to_bf16_rounding_is_reported() {
        let root = scratch("round");
        let src = root.join("src");
        fs::create_dir(&src).unwrap();
        model(&src);
        let pdir = root.join("patch");
        fs::create_dir(&pdir).unwrap();
        let delta = pdir.join("delta.safetensors");
        write_st(&delta, &[("d", "F32", vec![2, 2], vec![1e-4; 4])]);
        let m = manifest(&pdir, &delta, json!([{"tensor": "embed.weight", "delta": "d"}]));
        let out = root.join("out");
        let r = run_weight_patch(&prepare_weight_patch(&src, &m, &out, false).unwrap(), |_| {}).unwrap();
        assert_eq!(r["status"], "PATCH_HAD_NO_EFFECT_AFTER_ROUNDING");
        assert_eq!(r["elements_changed"], 0);
        fs::remove_dir_all(root).ok();
    }

    #[test]
    fn verify_detects_tampering() {
        let root = scratch("tamper");
        let src = root.join("src");
        fs::create_dir(&src).unwrap();
        model(&src);
        let pdir = root.join("patch");
        fs::create_dir(&pdir).unwrap();
        let delta = pdir.join("delta.safetensors");
        write_st(&delta, &[("d", "F32", vec![3], vec![1.0, 1.0, 1.0])]);
        let m = manifest(&pdir, &delta, json!([{"tensor": "layers.0.norm.weight", "delta": "d"}]));
        let out = root.join("out");
        run_weight_patch(&prepare_weight_patch(&src, &m, &out, true).unwrap(), |_| {}).unwrap();
        let shard = out.join("model.safetensors");
        let loc = read_safetensors_header(&shard).unwrap().remove("layers.0.norm.weight").unwrap();
        let mut f = OpenOptions::new().write(true).open(&shard).unwrap();
        f.seek(SeekFrom::Start(loc.offset)).unwrap();
        f.write_all(&9.0f32.to_le_bytes()).unwrap();
        drop(f);
        assert_eq!(verify_weight_patch(&out).unwrap()["status"], "MISMATCH");
        fs::remove_dir_all(root).ok();
    }
}
