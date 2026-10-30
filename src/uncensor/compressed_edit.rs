//! Compressed-first model editing without merging/dequantizing the base.
//!
//! Takes an EXISTING quantized GGUF and an Ojon low-rank edit recipe, checks
//! adapter shapes against the base GGUF tensor catalog without reading base
//! weights, and writes an independent native GGUF LoRA adapter. Runtime loads
//! base + adapter; no full-precision base/merge is required.
//!
//! This is NOT a quantizer, automatic activation extractor, or ARA trainer.
//! Reject directional-only recipes, unsupported naming, ambiguous devices,
//! vision/projector weight edits and unverified base tensor dimensions.

use crate::gguf::{inspect_gguf, inspect_gguf_tensors, GgufTensor};
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const RECIPE_LIMIT: u64 = 128 * 1024 * 1024;
const MAX_EDITS: usize = 256;
const MAX_ELEMENTS: usize = 1_000_000;
const MAX_RANK: usize = 64;
const ALIGN: u64 = 32;

fn invalid(s: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, s.into())
}
fn corruption(s: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidData, s.into())
}
fn le32(w: &mut impl Write, n: u32) -> io::Result<()> {
    w.write_all(&n.to_le_bytes())
}
fn le64(w: &mut impl Write, n: u64) -> io::Result<()> {
    w.write_all(&n.to_le_bytes())
}
fn gg_string(w: &mut impl Write, s: &str) -> io::Result<()> {
    le64(w, s.len() as u64)?;
    w.write_all(s.as_bytes())
}
fn kv_string(w: &mut impl Write, k: &str, v: &str) -> io::Result<()> {
    gg_string(w, k)?;
    le32(w, 8)?;
    gg_string(w, v)
}
fn kv_f32(w: &mut impl Write, k: &str, v: f32) -> io::Result<()> {
    gg_string(w, k)?;
    le32(w, 6)?;
    w.write_all(&v.to_le_bytes())
}
fn pad(w: &mut (impl Write + io::Seek), align: u64) -> io::Result<()> {
    let pos = w.stream_position()?;
    let extra = (align - pos % align) % align;
    if extra > 0 {
        w.write_all(&[0u8; 32][..extra as usize])?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct AdapterTensor {
    gguf_name: String,
    dims: [u64; 2],
    data: Vec<f32>,
}
#[derive(Debug)]
pub struct CompressedEditPlan {
    pub base: PathBuf,
    pub mmproj: Option<PathBuf>,
    pub output: PathBuf,
    architecture: String,
    base_bytes: u64,
    adapter: Vec<AdapterTensor>,
    rank: usize,
    edit_count: usize,
}
impl CompressedEditPlan {
    pub fn as_json(&self) -> Value {
        json!({
          "schema":"ojon.compressed_edit_plan.v1",
          "base_gguf":self.base.to_string_lossy(),
          "base_bytes":self.base_bytes,
          "architecture":self.architecture,
          "mmproj_gguf":self.mmproj.as_ref().map(|x|x.to_string_lossy().to_string()),
          "output_folder":self.output.to_string_lossy(),
          "adapter_tensors":self.adapter.len(),
          "modified_matrices":self.edit_count,
          "rank":self.rank,
          "additional_base_ram_required_bytes":0,
          "full_base_copy":false,
          "full_precision_merge":false,
          "original_model_modified":false,
          "model_inference":"NOT_RUN",
          "refusal_rate":"NOT_EVALUATED",
          "vision_qa":"NOT_EVALUATED",
          "status":"PLAN_ONLY"
        })
    }
}
#[derive(Debug)]
pub struct CompressedEditResult {
    pub output: PathBuf,
    pub adapter_bytes: u64,
    pub rank: usize,
    pub matrices: usize,
    pub vision_projector: bool,
}
impl CompressedEditResult {
    pub fn as_json(&self) -> Value {
        json!({
            "schema":"ojon.compressed_edit_result.v1",
            "output_folder":self.output.to_string_lossy(),
            "adapter_gguf":self.output.join("adapter.gguf").to_string_lossy(),
            "adapter_bytes":self.adapter_bytes,
            "rank":self.rank,
            "edited_matrices":self.matrices,
            "vision_projector":self.vision_projector,
            "full_precision_base_loaded":false,
            "merged_weights":false,
            "model_inference":"NOT_RUN",
            "refusal_rate":"NOT_EVALUATED",
            "vision_qa":"NOT_EVALUATED",
            "status":"NATIVE_GGUF_ADAPTER_WRITTEN_UNTESTED"
        })
    }
}
fn read_recipe(path: &Path) -> io::Result<Value> {
    let size = fs::metadata(path)?.len();
    if size == 0 || size > RECIPE_LIMIT {
        return Err(invalid("edit recipe empty or exceeds 128 MiB"));
    }
    let mut bytes = Vec::with_capacity(size as usize);
    File::open(path)?
        .take(RECIPE_LIMIT + 1)
        .read_to_end(&mut bytes)?;
    serde_json::from_slice(&bytes).map_err(|e| invalid(format!("edit recipe JSON invalid: {e}")))
}
fn numbers(v: &Value, label: &str, expected: usize) -> io::Result<Vec<f32>> {
    let arr = v
        .as_array()
        .ok_or_else(|| invalid(format!("{label} must be an array")))?;
    if arr.len() != expected {
        return Err(invalid(format!("{label} dimension mismatch")));
    }
    arr.iter()
        .map(|x| {
            let value = x
                .as_f64()
                .ok_or_else(|| invalid(format!("{label} entry not numeric")))?;
            if !value.is_finite() || value.abs() > f32::MAX as f64 {
                return Err(invalid(format!("{label} entry not finite f32")));
            }
            Ok(value as f32)
        })
        .collect()
}
fn safe_name(name: &str) -> io::Result<(String, String)> {
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
    .any(|x| n.contains(x))
    {
        return Err(invalid(
            "vision/projector weights cannot be modified by compressed language LoRA",
        ));
    }
    let target = if n.ends_with(".self_attn.o_proj.weight") {
        "attn_output"
    } else if n.ends_with(".mlp.down_proj.weight") {
        "ffn_down"
    } else {
        return Err(invalid(format!("unsupported target module name: {name}")));
    };
    // Supports text-model and VLM prefixes where the language backbone
    // exposes ...layers.<N>.<attention/MLP>. These are NOT vision layers.
    let prefix = name
        .strip_suffix(".self_attn.o_proj.weight")
        .or_else(|| name.strip_suffix(".mlp.down_proj.weight"))
        .ok_or_else(|| invalid("unknown target suffix"))?;
    let (lead, layer) = prefix
        .rsplit_once(".layers.")
        .or_else(|| prefix.strip_prefix("layers.").map(|x| ("", x)))
        .ok_or_else(|| invalid("expected language transformer layers.N"))?;
    if !(lead == "model"
        || lead == "model.language_model"
        || lead == "language_model"
        || lead == "language_model.model"
        || lead == "model.text_model"
        || lead == "model.language_model.model"
        || lead.is_empty())
    {
        return Err(invalid(format!(
            "unrecognized language model prefix: {lead}"
        )));
    }
    if layer.is_empty() || layer.len() > 5 || !layer.bytes().all(|b| b.is_ascii_digit()) {
        return Err(invalid("layer ID must be a small unsigned number"));
    }
    let id: u32 = layer.parse().map_err(|_| invalid("layer index invalid"))?;
    Ok((format!("blk.{id}.{target}.weight"), target.to_owned()))
}
fn verify_dims(base: &GgufTensor, rows: usize, cols: usize) -> io::Result<()> {
    if base.shape.as_slice() != [cols as u64, rows as u64] {
        return Err(invalid(format!(
            "adapter dimension mismatch for {}: base GGUF dims {:?}, recipe expects [{cols},{rows}]",
            base.name,base.shape
        )));
    }
    Ok(())
}
/// Validate compact low-rank edit against actual GGUF tensor names and dimensions.
/// The function reads GGUF metadata and adapter JSON only, never model weights.
pub fn prepare_compressed_edit(
    base: &Path,
    recipe: &Path,
    output: &Path,
    mmproj: Option<&Path>,
) -> io::Result<CompressedEditPlan> {
    let recipe = read_recipe(recipe)?;
    prepare_compressed_edit_with_recipe(base, &recipe, output, mmproj)
}

/// Build an adapter plan from an in-memory LoRA recipe. Allows the small
/// CPU-only ARA fitter to feed the GGUF exporter without intermediary files.
pub fn prepare_compressed_edit_with_recipe(
    base: &Path,
    recipe: &Value,
    output: &Path,
    mmproj: Option<&Path>,
) -> io::Result<CompressedEditPlan> {
    let base = base.canonicalize()?;
    if !base.is_file() {
        return Err(invalid("base GGUF path is not a regular file"));
    }
    let summary = inspect_gguf(&base)?;
    if summary.big_endian {
        return Err(invalid("big-endian GGUF adapters not supported"));
    }
    if summary.split_count.unwrap_or(1) > 1 {
        return Err(invalid(
            "split GGUF base unsupported: provide a single quantized GGUF",
        ));
    }
    let architecture = summary
        .architecture
        .ok_or_else(|| invalid("GGUF has no architecture"))?;
    // Check that this is actually a low-precision base, not a BF16 file.
    // Some GGUFs lack a reliable general.file_type, so inspect tensors.
    let tensors = inspect_gguf_tensors(&base)?;
    if tensors.is_empty() {
        return Err(invalid("GGUF base has no tensors"));
    }
    let quant_count = tensors
        .iter()
        .filter(|t| t.storage_type != 0 && t.storage_type != 1 && t.storage_type != 30)
        .count();
    if quant_count == 0 {
        return Err(invalid(
            "base appears all F32/F16/BF16: supply a quantized GGUF first",
        ));
    }
    let catalog: HashMap<&str, &GgufTensor> =
        tensors.iter().map(|t| (t.name.as_str(), t)).collect();
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    let name = output
        .file_name()
        .ok_or_else(|| invalid("missing output folder name"))?;
    let output = parent.join(name);
    if output.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists, "output exists"));
    }
    if output.starts_with(&base) {
        return Err(invalid("output folder must differ from model"));
    }
    let mmproj = match mmproj {
        Some(path) => {
            let path = path.canonicalize()?;
            if path == base {
                return Err(invalid("mmproj cannot equal language GGUF"));
            }
            let info = inspect_gguf(&path)?;
            if info.tensor_count == 0 {
                return Err(invalid("mmproj GGUF has no tensor catalog"));
            }
            Some(path)
        }
        None => None,
    };
    if recipe.get("schema").and_then(Value::as_str) != Some("ojon.abliteration.v1") {
        return Err(invalid("expected ojon.abliteration.v1 recipe"));
    }
    if recipe.get("preserve_row_norms").and_then(Value::as_bool) != Some(false) {
        return Err(invalid("compressed LoRA does not implement row-norm preservation; use preserve_row_norms:false and validate retained quality"));
    }
    let edits = recipe
        .get("edits")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("recipe edits must be an array"))?;
    if edits.is_empty() || edits.len() > MAX_EDITS {
        return Err(invalid("recipe edit count must be 1..256"));
    }
    let mut rank = None;
    let mut seen = HashSet::<String>::new();
    let mut out = Vec::<AdapterTensor>::new();
    for edit in edits {
        let tensor = edit
            .get("tensor")
            .and_then(Value::as_str)
            .ok_or_else(|| invalid("edit missing tensor name"))?;
        let (mapped, _) = safe_name(tensor)?;
        if !seen.insert(mapped.clone()) {
            return Err(invalid("duplicate mapped adapter tensor"));
        }
        let base_tensor = catalog.get(mapped.as_str()).ok_or_else(|| {
            invalid(format!(
                "base GGUF does not contain mapped tensor {mapped}; refusing unverified adapter"
            ))
        })?;
        let lowrank=edit.get("lora").ok_or_else(||invalid(
            "compressed-first requires ARA/LoRA recipe (lora.A/B); directional ablation alone cannot be applied without a base-weight projection"))?;
        let a = lowrank
            .get("a")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("LoRA A must be rank x input array"))?;
        let b = lowrank
            .get("b")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("LoRA B must be output x rank array"))?;
        let r = a.len();
        if r == 0 || r > MAX_RANK {
            return Err(invalid("rank must be 1..64"));
        }
        if let Some(prev) = rank {
            if prev != r {
                return Err(invalid(
                    "all LoRA edits must have the same rank for one GGUF adapter",
                ));
            }
        }
        rank = Some(r);
        let rows = b.len();
        let cols = a
            .first()
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("LoRA A first row missing"))?
            .len();
        if rows == 0 || cols == 0 || rows > 131072 || cols > 131072 {
            return Err(invalid("invalid LoRA row/column count"));
        }
        if (rows + cols)
            .checked_mul(r)
            .ok_or_else(|| invalid("LoRA dimension overflow"))?
            > MAX_ELEMENTS
        {
            return Err(invalid("LoRA exceeds 1 million coefficients per module"));
        }
        verify_dims(base_tensor, rows, cols)?;
        let strength = edit
            .get("strength")
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid("edit strength missing"))?;
        if !strength.is_finite() || strength <= 0.0 || strength > 1.0 {
            return Err(invalid("LoRA strength must be >0 and <=1"));
        }
        let mut a_values = Vec::with_capacity(r * cols);
        for row in a {
            a_values.extend(numbers(row, "LoRA A row", cols)?);
        }
        let mut b_values = Vec::with_capacity(rows * r);
        for row in b {
            for x in numbers(row, "LoRA B row", r)? {
                let scaled = x as f64 * strength;
                if !scaled.is_finite() || scaled.abs() > f32::MAX as f64 {
                    return Err(invalid("scaled LoRA B exceeds f32 limits"));
                }
                b_values.push(scaled as f32);
            }
        }
        out.push(AdapterTensor {
            gguf_name: format!("{mapped}.lora_a"),
            dims: [cols as u64, r as u64],
            data: a_values,
        });
        out.push(AdapterTensor {
            gguf_name: format!("{mapped}.lora_b"),
            dims: [r as u64, rows as u64],
            data: b_values,
        });
    }
    out.sort_by(|a, b| a.gguf_name.cmp(&b.gguf_name));
    Ok(CompressedEditPlan {
        base,
        mmproj,
        output,
        architecture,
        base_bytes: summary.file_bytes,
        adapter: out,
        rank: rank.unwrap(),
        edit_count: edits.len(),
    })
}

struct Stage(PathBuf);
impl Drop for Stage {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn make_stage(parent: &Path) -> io::Result<Stage> {
    for retry in 0..16 {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| invalid("clock invalid"))?
            .as_nanos();
        let path = parent.join(format!(
            ".ojon-compressed-{}-{stamp}-{retry}",
            std::process::id()
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(Stage(path)),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    }
    Err(invalid("cannot create compressed adapter stage"))
}
/// GGUF v3 writer matching llama.cpp's general.type=adapter and .lora_a/b
/// tensor naming, F32 storage and alpha=rank for exact BA coefficient.
fn write_adapter(plan: &CompressedEditPlan, path: &Path) -> io::Result<u64> {
    let mut f = OpenOptions::new().write(true).create_new(true).open(path)?;
    let count =
        u64::try_from(plan.adapter.len()).map_err(|_| invalid("too many adapter tensors"))?;
    f.write_all(b"GGUF")?;
    le32(&mut f, 3)?;
    le64(&mut f, count)?;
    le64(&mut f, 4)?;
    kv_string(&mut f, "general.architecture", &plan.architecture)?;
    kv_string(&mut f, "general.type", "adapter")?;
    kv_string(&mut f, "adapter.type", "lora")?;
    kv_f32(&mut f, "adapter.lora.alpha", plan.rank as f32)?;
    let mut offset = 0u64;
    for tensor in &plan.adapter {
        gg_string(&mut f, &tensor.gguf_name)?;
        le32(&mut f, 2)?;
        le64(&mut f, tensor.dims[0])?;
        le64(&mut f, tensor.dims[1])?;
        le32(&mut f, 0)?; // GGML F32
        le64(&mut f, offset)?;
        let bytes = (tensor.data.len() as u64)
            .checked_mul(4)
            .ok_or_else(|| invalid("adapter byte size overflow"))?;
        offset = offset
            .checked_add(bytes)
            .and_then(|n| n.checked_add((ALIGN - n % ALIGN) % ALIGN))
            .ok_or_else(|| invalid("adapter offset overflow"))?;
    }
    pad(&mut f, ALIGN)?;
    for tensor in &plan.adapter {
        for x in &tensor.data {
            f.write_all(&x.to_le_bytes())?;
        }
        pad(&mut f, ALIGN)?;
    }
    f.sync_all()?;
    drop(f);
    let info = inspect_gguf(path)?;
    if info.tensor_count != count || info.architecture.as_deref() != Some(&plan.architecture) {
        return Err(corruption(
            "written native LoRA GGUF did not pass metadata verification",
        ));
    }
    Ok(info.file_bytes)
}

/// Never duplicates the base file: writes adapter.gguf and launch manifest.
/// Requires original quantized GGUF (and optional mmproj) to remain present.
pub fn run_compressed_edit(
    plan: &CompressedEditPlan,
    mut log: impl FnMut(&str),
) -> io::Result<CompressedEditResult> {
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "output folder already exists",
        ));
    }
    if fs::metadata(&plan.base)?.len() != plan.base_bytes {
        return Err(invalid("base GGUF changed after plan; rerun preflight"));
    }
    let stage = make_stage(
        plan.output
            .parent()
            .ok_or_else(|| invalid("output parent missing"))?,
    )?;
    let adapter_path = stage.0.join("adapter.gguf");
    log("[ojon] writing native LoRA GGUF without loading compressed base weights");
    let bytes = write_adapter(plan, &adapter_path)?;
    let manifest = json!({
        "schema":"ojon.compressed_model_bundle.v1",
        "original_quantized_base":plan.base.to_string_lossy(),
        "base_bytes":plan.base_bytes,
        "adapter":"adapter.gguf",
        "adapter_bytes":bytes,
        "mmproj":plan.mmproj.as_ref().map(|x|x.to_string_lossy().to_string()),
        "architecture":plan.architecture,
        "rank":plan.rank,
        "edited_matrices":plan.edit_count,
        "launch_flags":"-m <original-base.gguf> --lora <adapter.gguf> [--mmproj <vision.gguf>]",
        "no_full_precision_base_loaded":true,
        "no_merged_model":true,
        "model_inference":"NOT_RUN",
        "vision_qa":"NOT_EVALUATED",
        "refusal_rate":"NOT_EVALUATED",
        "quality_retention":"NOT_EVALUATED"
    });
    let mut m = File::create(stage.0.join("ojon-compressed-bundle.json"))?;
    m.write_all(
        serde_json::to_string_pretty(&manifest)
            .map_err(|e| invalid(format!("bundle serialization: {e}")))?
            .as_bytes(),
    )?;
    m.sync_all()?;
    drop(m); // Windows rename requires all handles closed.
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "output folder appeared during processing",
        ));
    }
    fs::rename(&stage.0, &plan.output)?;
    log("[ojon] published adapter bundle (unverified inference); base remains untouched");
    Ok(CompressedEditResult {
        output: plan.output.clone(),
        adapter_bytes: bytes,
        rank: plan.rank,
        matrices: plan.edit_count,
        vision_projector: plan.mmproj.is_some(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scratch() -> PathBuf {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!(
            "ojon-compressed-test-{}-{stamp}",
            std::process::id()
        ));
        fs::create_dir(&p).unwrap();
        p
    }
    fn gguf(path: &Path) {
        let mut f = File::create(path).unwrap();
        f.write_all(b"GGUF").unwrap();
        le32(&mut f, 3).unwrap();
        le64(&mut f, 2).unwrap();
        le64(&mut f, 1).unwrap();
        kv_string(&mut f, "general.architecture", "qwen3vl").unwrap();
        for (name, shape, offset) in [
            ("blk.0.attn_output.weight", [2u64, 2u64], 0u64),
            ("blk.1.ffn_down.weight", [2u64, 2u64], 32u64),
        ] {
            gg_string(&mut f, name).unwrap();
            le32(&mut f, 2).unwrap();
            le64(&mut f, shape[0]).unwrap();
            le64(&mut f, shape[1]).unwrap();
            le32(&mut f, 2).unwrap(); // quantized Q4_0 (fake payload for metadata-only test)
            le64(&mut f, offset).unwrap();
        }
        pad(&mut f, ALIGN).unwrap();
        f.write_all(&[0u8; 64]).unwrap();
    }
    #[test]
    fn export_native_adapter_without_loading_quantized_base() {
        let dir = scratch();
        let model = dir.join("base.gguf");
        gguf(&model);
        let input = dir.join("recipe.json");
        fs::write(
            &input,
            serde_json::to_vec(&json!({
                "schema":"ojon.abliteration.v1","preserve_row_norms":false,
                "edits":[
                  {"tensor":"model.language_model.layers.0.self_attn.o_proj.weight",
                    "strength":1.0,"lora":{"a":[[1.0,0.0]],"b":[[2.0],[0.0]]}},
                  {"tensor":"model.language_model.layers.1.mlp.down_proj.weight",
                    "strength":0.5,"lora":{"a":[[0.0,1.0]],"b":[[0.0],[2.0]]}}
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        let output = dir.join("out");
        let plan = prepare_compressed_edit(&model, &input, &output, None).unwrap();
        assert_eq!(plan.rank, 1);
        assert_eq!(plan.adapter.len(), 4);
        assert!(!output.exists());
        let result = run_compressed_edit(&plan, |_| {}).unwrap();
        assert!(result.adapter_bytes < 2048);
        let gg = inspect_gguf(&output.join("adapter.gguf")).unwrap();
        assert_eq!(gg.tensor_count, 4);
        assert_eq!(gg.architecture.as_deref(), Some("qwen3vl"));
        assert!(output.join("ojon-compressed-bundle.json").is_file());
        assert!(model.is_file());
        assert!(run_compressed_edit(&plan, |_| {}).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
    #[test]
    fn rejects_direction_only_and_wrong_matrix_shape() {
        let dir = scratch();
        let model = dir.join("base.gguf");
        gguf(&model);
        let recipe = dir.join("recipe.json");
        let out = dir.join("out");
        fs::write(
            &recipe,
            serde_json::to_vec(&json!({
                "schema":"ojon.abliteration.v1","preserve_row_norms":false,
                "edits":[{"tensor":"model.layers.0.self_attn.o_proj.weight",
                    "strength":1.0,"direction":[1.0,0.0]}]
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(prepare_compressed_edit(&model, &recipe, &out, None).is_err());
        fs::write(
            &recipe,
            serde_json::to_vec(&json!({
                "schema":"ojon.abliteration.v1","preserve_row_norms":false,
                "edits":[{"tensor":"model.layers.0.self_attn.o_proj.weight",
                    "strength":1.0,"lora":{"a":[[1.0,0.0,0.0]],"b":[[1.0],[0.0]]}}]
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(prepare_compressed_edit(&model, &recipe, &out, None).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
