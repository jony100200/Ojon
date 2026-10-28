//! Directional editing of an original F16/BF16/F32 GGUF checkpoint.
//! The source remains read-only. Float matrices are patched in a staged copy;
//! a loader and refusal/capability evaluation must still validate the result.

use crate::abliteration::{apply_directional_matrix, normalized_direction};
use crate::gguf::{inspect_gguf, inspect_gguf_tensors};
use crate::resource::free_disk;
use serde_json::{json, Value};
use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message.into())
}

struct MatrixEdit {
    name: String,
    offset: u64,
    rows: usize,
    cols: usize,
    dtype: &'static str,
    direction: Vec<f64>,
    strength: f64,
}

pub struct GgufAblationPlan {
    source: PathBuf,
    output: PathBuf,
    bytes: u64,
    modified: SystemTime,
    edits: Vec<MatrixEdit>,
    preserve_row_norms: bool,
    recipe: Value,
}

impl GgufAblationPlan {
    pub fn as_json(&self) -> Value {
        json!({
            "schema":"ojon.gguf_abliteration_plan.v1",
            "source":self.source.to_string_lossy(),
            "output":self.output.to_string_lossy(),
            "source_bytes":self.bytes,
            "edited_tensors":self.edits.iter().map(|e|json!({"name":e.name,"rows":e.rows,
                "columns":e.cols,"dtype":e.dtype,"strength":e.strength})).collect::<Vec<_>>(),
            "preserve_row_norms":self.preserve_row_norms,
            "original_model_modified":false,
            "activation_capture":"NOT_RUN",
            "refusal_rate":"NOT_EVALUATED",
            "capability_retention":"NOT_EVALUATED",
            "model_load":"NOT_TESTED",
            "status":"PLAN_ONLY"
        })
    }
}

fn safe_target(name: &str) -> bool {
    let mut parts = name.split('.');
    matches!(parts.next(), Some("blk"))
        && parts
            .next()
            .is_some_and(|part| !part.is_empty() && part.bytes().all(|b| b.is_ascii_digit()))
        && matches!(
            parts.next(),
            Some("attn_output" | "ffn_down" | "ffn_down_shexp")
        )
        && matches!(parts.next(), Some("weight"))
        && parts.next().is_none()
}

pub fn prepare_gguf_abliteration(
    source: &Path,
    recipe_path: &Path,
    output: &Path,
) -> io::Result<GgufAblationPlan> {
    let size = fs::metadata(recipe_path)?.len();
    if size == 0 || size > 64 * 1024 * 1024 {
        return Err(invalid("recipe must be 1 byte..64 MiB"));
    }
    let recipe: Value = serde_json::from_slice(&fs::read(recipe_path)?)
        .map_err(|e| invalid(format!("invalid direction recipe: {e}")))?;
    prepare_gguf_abliteration_value(source, &recipe, output)
}

pub fn prepare_gguf_abliteration_value(
    source: &Path,
    recipe: &Value,
    output: &Path,
) -> io::Result<GgufAblationPlan> {
    let source = source.canonicalize()?;
    if source.extension().and_then(|x| x.to_str()) != Some("gguf") {
        return Err(invalid("source must be an original .gguf file"));
    }
    let source_meta = fs::symlink_metadata(&source)?;
    if !source_meta.is_file() || source_meta.file_type().is_symlink() {
        return Err(invalid("source must be a regular GGUF file"));
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."))
        .canonicalize()?;
    let scratch_required = source_meta
        .len()
        .checked_add((source_meta.len() / 20).max(512 * 1024 * 1024))
        .ok_or_else(|| invalid("GGUF output space budget overflow"))?;
    let free =
        free_disk(&parent).ok_or_else(|| invalid("cannot measure free output disk space"))?;
    if free < scratch_required {
        return Err(invalid(format!(
            "edited GGUF needs {scratch_required} free bytes on output volume; found {free}"
        )));
    }
    let output = parent.join(
        output
            .file_name()
            .ok_or_else(|| invalid("output filename missing"))?,
    );
    if output.extension().and_then(|x| x.to_str()) != Some("gguf") || output == source {
        return Err(invalid("output must be a distinct .gguf file"));
    }
    if output.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "output GGUF already exists",
        ));
    }
    let summary = inspect_gguf(&source)?;
    if summary.tensor_count == 0
        || summary.split_count.unwrap_or(1) > 1
        || summary.big_endian
        || !matches!(summary.file_type, Some(0 | 1 | 32))
    {
        return Err(invalid(
            "directional GGUF edit requires a single-file little-endian F32/F16/BF16 original",
        ));
    }
    if recipe["schema"] != "ojon.abliteration.v1" {
        return Err(invalid("expected ojon.abliteration.v1 recipe"));
    }
    let preserve_row_norms = recipe["preserve_row_norms"]
        .as_bool()
        .ok_or_else(|| invalid("preserve_row_norms must be boolean"))?;
    let entries = recipe["edits"]
        .as_array()
        .filter(|v| !v.is_empty() && v.len() <= 256)
        .ok_or_else(|| invalid("recipe needs 1..256 tensor edits"))?;
    let tensors = inspect_gguf_tensors(&source)?
        .into_iter()
        .map(|tensor| (tensor.name.clone(), tensor))
        .collect::<HashMap<_, _>>();
    let mut seen = HashSet::new();
    let mut edits = Vec::with_capacity(entries.len());
    for entry in entries {
        let name = entry["tensor"]
            .as_str()
            .ok_or_else(|| invalid("edit tensor name missing"))?;
        if !safe_target(name) || !seen.insert(name) {
            return Err(invalid(format!(
                "unsupported or duplicate GGUF edit target: {name}"
            )));
        }
        let tensor = tensors
            .get(name)
            .ok_or_else(|| invalid(format!("tensor absent from GGUF: {name}")))?;
        if tensor.shape.len() != 2 {
            return Err(invalid("GGUF edit target must be a 2D matrix"));
        }
        // GGUF dims are [input columns, output rows], unlike safetensors.
        let cols =
            usize::try_from(tensor.shape[0]).map_err(|_| invalid("input width too large"))?;
        let rows =
            usize::try_from(tensor.shape[1]).map_err(|_| invalid("output width too large"))?;
        let (dtype, width) = match tensor.storage_type {
            0 => ("F32", 4u64),
            1 => ("F16", 2u64),
            30 => ("BF16", 2u64),
            _ => {
                return Err(invalid(format!(
                    "{name} is quantized or unsupported; edit an original float GGUF"
                )))
            }
        };
        let row_bytes = (cols as u64)
            .checked_mul(width)
            .ok_or_else(|| invalid("row byte size overflow"))?;
        let bytes = (rows as u64)
            .checked_mul(row_bytes)
            .ok_or_else(|| invalid("matrix byte size overflow"))?;
        let offset = summary
            .data_offset
            .checked_add(tensor.relative_offset)
            .ok_or_else(|| invalid("tensor offset overflow"))?;
        if rows == 0
            || rows > 131072
            || row_bytes > 16 * 1024 * 1024
            || offset
                .checked_add(bytes)
                .is_none_or(|end| end > summary.file_bytes)
        {
            return Err(invalid(
                "matrix exceeds supported dimensions or source bounds",
            ));
        }
        let strength = entry["strength"]
            .as_f64()
            .ok_or_else(|| invalid("edit strength missing"))?;
        if !strength.is_finite() || strength <= 0.0 || strength > 1.0 {
            return Err(invalid("edit strength must be >0 and <=1"));
        }
        let direction = normalized_direction(entry, rows)?;
        edits.push(MatrixEdit {
            name: name.to_owned(),
            offset,
            rows,
            cols,
            dtype,
            direction,
            strength,
        });
    }
    Ok(GgufAblationPlan {
        source,
        output,
        bytes: source_meta.len(),
        modified: source_meta.modified()?,
        edits,
        preserve_row_norms,
        recipe: recipe.clone(),
    })
}

struct Stage(PathBuf);
impl Drop for Stage {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
}

pub fn run_gguf_abliteration(
    plan: &GgufAblationPlan,
    mut log: impl FnMut(&str),
) -> io::Result<Value> {
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "output GGUF already exists",
        ));
    }
    let mut receipt_name = plan.output.as_os_str().to_os_string();
    receipt_name.push(".ojon-recipe.json");
    let receipt = PathBuf::from(receipt_name);
    if receipt.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "GGUF recipe receipt already exists",
        ));
    }
    let current = fs::metadata(&plan.source)?;
    if current.len() != plan.bytes || current.modified()? != plan.modified {
        return Err(invalid("source GGUF changed after planning"));
    }
    let parent = plan
        .output
        .parent()
        .ok_or_else(|| invalid("output parent missing"))?;
    let (stage, mut output) = loop {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| invalid("invalid clock"))?
            .as_nanos();
        let path = parent.join(format!(
            ".ojon-gguf-edit-{}-{stamp}.tmp",
            std::process::id()
        ));
        match OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(file) => break (Stage(path), file),
            Err(e) if e.kind() == ErrorKind::AlreadyExists => continue,
            Err(e) => return Err(e),
        }
    };
    let mut input = File::open(&plan.source)?;
    let mut buffer = vec![0u8; 4 * 1024 * 1024];
    let mut copied = 0u64;
    loop {
        let n = input.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        output.write_all(&buffer[..n])?;
        copied = copied
            .checked_add(n as u64)
            .ok_or_else(|| invalid("copy size overflow"))?;
    }
    if copied != plan.bytes || fs::metadata(&plan.source)?.modified()? != plan.modified {
        return Err(invalid("source GGUF changed while staging"));
    }
    for (index, edit) in plan.edits.iter().enumerate() {
        log(&format!(
            "[ojon] editing GGUF float matrix {}/{}: {}",
            index + 1,
            plan.edits.len(),
            edit.name
        ));
        apply_directional_matrix(
            &mut output,
            edit.offset,
            edit.rows,
            edit.cols,
            edit.dtype,
            &edit.direction,
            edit.strength,
            plan.preserve_row_norms,
        )?;
    }
    output.flush()?;
    output.sync_all()?;
    drop(output);
    let checked = inspect_gguf(&stage.0)?;
    if checked.file_bytes != plan.bytes || checked.tensor_count == 0 {
        return Err(invalid("edited GGUF failed container verification"));
    }
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(
            ErrorKind::AlreadyExists,
            "output GGUF appeared during edit",
        ));
    }
    let result = json!({
        "schema":"ojon.gguf_abliteration_result.v1",
        "source":plan.source.to_string_lossy(),
        "output":plan.output.to_string_lossy(),
        "recipe_receipt":receipt.to_string_lossy(),
        "bytes":plan.bytes,
        "edited_tensors":plan.edits.len(),
        "original_model":"PRESERVED",
        "container_verification":"PASSED",
        "model_load":"NOT_TESTED",
        "refusal_rate":"NOT_EVALUATED",
        "capability_retention":"NOT_EVALUATED",
        "status":"EDITED_GGUF_NOT_VALIDATED"
    });
    let receipt_stage = parent.join(format!(
        ".ojon-gguf-receipt-{}-{}.tmp",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| invalid("invalid clock"))?
            .as_nanos()
    ));
    let receipt_stage = Stage(receipt_stage);
    let mut receipt_file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&receipt_stage.0)?;
    serde_json::to_writer_pretty(
        &mut receipt_file,
        &json!({"result":result,"recipe":plan.recipe}),
    )?;
    receipt_file.sync_all()?;
    drop(receipt_file);
    fs::rename(&stage.0, &plan.output)?;
    if let Err(error) = fs::rename(&receipt_stage.0, &receipt) {
        let _ = fs::remove_file(&plan.output);
        return Err(error);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Seek, SeekFrom};

    fn gguf(path: &Path, storage: u32, file_type: u32) {
        fn string(f: &mut File, s: &str) {
            f.write_all(&(s.len() as u64).to_le_bytes()).unwrap();
            f.write_all(s.as_bytes()).unwrap();
        }
        let mut f = File::create(path).unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        string(&mut f, "general.architecture");
        f.write_all(&8u32.to_le_bytes()).unwrap();
        string(&mut f, "qwen3");
        string(&mut f, "general.file_type");
        f.write_all(&4u32.to_le_bytes()).unwrap();
        f.write_all(&file_type.to_le_bytes()).unwrap();
        string(&mut f, "blk.0.attn_output.weight");
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&storage.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        let pad = (32 - f.stream_position().unwrap() % 32) % 32;
        f.write_all(&vec![0u8; pad as usize]).unwrap();
        for value in [2.0f32, 0.0, 0.0, 2.0] {
            if storage == 30 {
                f.write_all(&((value.to_bits() >> 16) as u16).to_le_bytes())
                    .unwrap();
            } else {
                f.write_all(&value.to_le_bytes()).unwrap();
            }
        }
    }

    #[test]
    fn float_gguf_projection_preserves_source_and_valid_container() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("ojon-gguf-edit-{}-{stamp}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let source = dir.join("original.gguf");
        let output = dir.join("edited.gguf");
        gguf(&source, 0, 0);
        let original = fs::read(&source).unwrap();
        let recipe = json!({"schema":"ojon.abliteration.v1","preserve_row_norms":false,
            "edits":[{"tensor":"blk.0.attn_output.weight","direction":[1.0,0.0],"strength":1.0}]});
        let plan = prepare_gguf_abliteration_value(&source, &recipe, &output).unwrap();
        assert_eq!(plan.as_json()["edited_tensors"][0]["rows"], 2);
        let result = run_gguf_abliteration(&plan, |_| {}).unwrap();
        assert_eq!(result["status"], "EDITED_GGUF_NOT_VALIDATED");
        let saved: Value =
            serde_json::from_slice(&fs::read(result["recipe_receipt"].as_str().unwrap()).unwrap())
                .unwrap();
        assert_eq!(saved["recipe"], recipe);
        assert_eq!(saved["result"]["output"], result["output"]);
        assert_eq!(fs::read(&source).unwrap(), original);
        assert!(inspect_gguf(&output).is_ok());
        let mut file = File::open(&output).unwrap();
        file.seek(SeekFrom::Start(plan.edits[0].offset)).unwrap();
        let mut data = [0u8; 16];
        file.read_exact(&mut data).unwrap();
        let numbers = data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect::<Vec<_>>();
        assert_eq!(numbers, [0.0, 0.0, 0.0, 2.0]);
        assert!(run_gguf_abliteration(&plan, |_| {}).is_err());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn refuses_compressed_and_unprotected_targets() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("ojon-gguf-reject-{}-{stamp}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let source = dir.join("model.gguf");
        let output = dir.join("out.gguf");
        gguf(&source, 0, 15);
        let recipe = json!({"schema":"ojon.abliteration.v1","preserve_row_norms":false,
            "edits":[{"tensor":"blk.0.attn_output.weight","direction":[1.0,0.0],"strength":1.0}]});
        assert!(prepare_gguf_abliteration_value(&source, &recipe, &output).is_err());
        gguf(&source, 0, 0);
        let protected = json!({"schema":"ojon.abliteration.v1","preserve_row_norms":false,
            "edits":[{"tensor":"blk.0.ffn_gate_inp.weight","direction":[1.0,0.0],"strength":1.0}]});
        assert!(prepare_gguf_abliteration_value(&source, &protected, &output).is_err());
        assert!(!output.exists());
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn bf16_original_is_editable_without_safetensors() {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir =
            std::env::temp_dir().join(format!("ojon-gguf-bf16-{}-{stamp}", std::process::id()));
        fs::create_dir(&dir).unwrap();
        let source = dir.join("teacher.gguf");
        let output = dir.join("edited.gguf");
        gguf(&source, 30, 32);
        let original = fs::read(&source).unwrap();
        let recipe = json!({"schema":"ojon.abliteration.v1","preserve_row_norms":false,
            "edits":[{"tensor":"blk.0.attn_output.weight","direction":[1.0,0.0],"strength":1.0}]});
        let plan = prepare_gguf_abliteration_value(&source, &recipe, &output).unwrap();
        run_gguf_abliteration(&plan, |_| {}).unwrap();
        assert_eq!(fs::read(&source).unwrap(), original);
        let mut file = File::open(&output).unwrap();
        file.seek(SeekFrom::Start(plan.edits[0].offset)).unwrap();
        let mut data = [0u8; 8];
        file.read_exact(&mut data).unwrap();
        let numbers = data
            .chunks_exact(2)
            .map(|b| f32::from_bits((u16::from_le_bytes(b.try_into().unwrap()) as u32) << 16))
            .collect::<Vec<_>>();
        assert_eq!(numbers, [0.0, 0.0, 0.0, 2.0]);
        fs::remove_dir_all(dir).unwrap();
    }
}
