//! Native whole-folder HF model editing: safetensors index-aware and VLM-safe.
//! This is a copy-on-write checkpoint packager around the stage-1 tensor editor.
//! No model inference, activation capture, ARA or quantization happens here.

use crate::abliteration::{prepare_abliteration_with_recipe, run_abliteration};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::fs::{self, File};
use std::io::{self, ErrorKind, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_JSON_BYTES: u64 = 64 * 1024 * 1024;
const MAX_FILES: usize = 20000;
const MAX_EDITS: usize = 256;
const MAX_DEPTH: usize = 16;

fn invalid(s: impl Into<String>) -> io::Error { io::Error::new(ErrorKind::InvalidInput, s.into()) }
fn corrupt(s: impl Into<String>) -> io::Error { io::Error::new(ErrorKind::InvalidData, s.into()) }

#[derive(Debug)]
struct SourceFile {
    relative: PathBuf,
    bytes: u64,
}

#[derive(Debug)]
struct ShardEdits {
    relative: PathBuf,
    recipe: Value,
    count: usize,
}

#[derive(Debug)]
pub struct ModelAblationPlan {
    source: PathBuf,
    output: PathBuf,
    files: Vec<SourceFile>,
    shards: Vec<ShardEdits>,
    source_bytes: u64,
    vision_config_detected: bool,
    indexed: bool,
    total_edits: usize,
}

impl ModelAblationPlan {
    pub fn as_json(&self) -> Value {
        json!({
            "schema": "ojon.model_abliteration_plan.v1",
            "source": self.source.to_string_lossy(),
            "output": self.output.to_string_lossy(),
            "source_files": self.files.len(),
            "source_bytes": self.source_bytes,
            "edited_shards": self.shards.iter().map(|s| s.relative.to_string_lossy().to_string()).collect::<Vec<_>>(),
            "edited_tensors": self.total_edits,
            "indexed_safetensors": self.indexed,
            "vision_config_detected": self.vision_config_detected,
            "vision_weights": "PRESERVED_UNCHANGED",
            "activation_capture": "NOT_IMPLEMENTED",
            "auto_optimization": "NOT_IMPLEMENTED",
            "model_load": "NOT_TESTED",
            "refusal_rate": "NOT_EVALUATED",
            "vision_qa": "NOT_EVALUATED",
            "status": "PLAN_ONLY"
        })
    }
}

fn read_json(path: &Path, max: u64) -> io::Result<Value> {
    let len = fs::metadata(path)?.len();
    if len == 0 || len > max { return Err(invalid(format!("missing or oversized JSON: {}",path.display()))); }
    let mut buf = Vec::with_capacity(len as usize);
    File::open(path)?.take(max + 1).read_to_end(&mut buf)?;
    serde_json::from_slice(&buf).map_err(|e| corrupt(format!("invalid JSON {}: {e}", path.display())))
}

fn enumerate(root: &Path, dir: &Path, depth: usize, files: &mut Vec<SourceFile>) -> io::Result<()> {
    if depth > MAX_DEPTH { return Err(invalid("model directory nesting too deep")); }
    let mut entries = fs::read_dir(dir)?.collect::<io::Result<Vec<_>>>()?;
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let p = entry.path();
        let meta = fs::symlink_metadata(&p)?;
        if meta.file_type().is_symlink() {
            return Err(invalid(format!("model folder contains symlink (materialize files first): {}",p.display())));
        }
        let rel = p.strip_prefix(root).map_err(|_| invalid("file path escaped model root"))?.to_path_buf();
        if meta.is_dir() {
            // Avoid copying Git/HF cache internals into published model checkpoints.
            let name = entry.file_name();
            if name == ".git" || name == ".cache" { continue; }
            enumerate(root, &p, depth + 1, files)?;
        } else if meta.is_file() {
            if files.len() >= MAX_FILES { return Err(invalid("too many model files")); }
            files.push(SourceFile { relative: rel, bytes: meta.len() });
        } else {
            return Err(invalid("model source contains non-file/directory entry"));
        }
    }
    Ok(())
}

fn exact_shard_name(shard: &str) -> io::Result<PathBuf> {
    let p = Path::new(shard);
    if shard.is_empty() || shard.starts_with('.') || shard.contains('\\') ||
        p.components().count() != 1 || p.file_name().and_then(|s|s.to_str()) != Some(shard) ||
        !shard.ends_with(".safetensors") {
        return Err(invalid(format!("unsafe safetensors index shard path: {shard}")));
    }
    Ok(p.to_path_buf())
}

fn shard_tensors(path: &Path) -> io::Result<BTreeSet<String>> {
    let size = fs::metadata(path)?.len();
    if size < 10 { return Err(corrupt("truncated safetensors shard")); }
    let mut f=File::open(path)?;
    let mut prefix=[0u8;8];f.read_exact(&mut prefix)?;
    let hlen=u64::from_le_bytes(prefix);
    if hlen == 0 || hlen > 32*1024*1024 || hlen > size-8 { return Err(corrupt("invalid safetensors header size")); }
    let mut header=vec![0;hlen as usize];f.read_exact(&mut header)?;
    let map:Value=serde_json::from_slice(&header).map_err(|e|corrupt(format!("invalid safetensors header: {e}")))?;
    let obj=map.as_object().ok_or_else(||corrupt("invalid safetensors tensor map"))?;
    if obj.len() > 100000 { return Err(corrupt("too many tensors in shard")); }
    Ok(obj.keys().filter(|n|n.as_str()!="__metadata__").cloned().collect())
}

/// Create a safe model-folder plan. Index coverage is verified before any edit
/// is published; the recipe must explicitly identify tensors to change.
pub fn prepare_model_abliteration(source: &Path, recipe_path: &Path, output: &Path) -> io::Result<ModelAblationPlan> {
    let source = source.canonicalize()?;
    if !source.is_dir() { return Err(invalid("source must be an existing HF model directory")); }
    if !source.join("config.json").is_file() { return Err(invalid("Hugging Face config.json missing")); }
    let parent=output.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new(".")).canonicalize()?;
    if !parent.is_dir() { return Err(invalid("output parent must exist")); }
    let name=output.file_name().ok_or_else(||invalid("output folder name missing"))?;
    if name == "." || name == ".." { return Err(invalid("invalid output folder name")); }
    let output=parent.join(name);
    if output.symlink_metadata().is_ok() { return Err(io::Error::new(ErrorKind::AlreadyExists,"output directory already exists")); }
    if output.starts_with(&source) || source.starts_with(&output) {
        return Err(invalid("output must not contain or reside inside the source model directory"));
    }

    let mut files=Vec::new();
    enumerate(&source, &source, 0, &mut files)?;
    let file_names:HashSet<PathBuf>=files.iter().map(|f|f.relative.clone()).collect();
    let cfg=read_json(&source.join("config.json"),MAX_JSON_BYTES)?;
    let vision_config_detected=cfg.get("vision_config").is_some() ||
        cfg.get("text_config").is_some() && cfg.get("model_type").and_then(Value::as_str)
            .is_some_and(|t|t.contains("vl") || t.contains("vision"));
    let mut source_bytes=0u64;
    for file in &files { source_bytes=source_bytes.checked_add(file.bytes).ok_or_else(||invalid("source model file size overflow"))?; }

    let index_file=Path::new("model.safetensors.index.json");
    let indexed=file_names.contains(index_file);
    let mut weight_map=BTreeMap::<String,PathBuf>::new();
    let mut referenced=BTreeMap::<PathBuf,BTreeSet<String>>::new();
    if indexed {
        let idx=read_json(&source.join(index_file),MAX_JSON_BYTES)?;
        let mapping=idx.get("weight_map").and_then(Value::as_object)
            .ok_or_else(||invalid("safetensors index lacks weight_map"))?;
        if mapping.is_empty() || mapping.len()>200000 { return Err(invalid("invalid safetensors index weight_map size")); }
        for (tensor,file) in mapping {
            let name=file.as_str().ok_or_else(||invalid("non-string shard in weight_map"))?;
            let shard=exact_shard_name(name)?;
            if !file_names.contains(&shard) { return Err(invalid(format!("missing indexed weight shard: {name}"))); }
            referenced.entry(shard.clone()).or_default().insert(tensor.clone());
            weight_map.insert(tensor.clone(),shard);
        }
        // Full coverage: an index pointing to the wrong tensor or a missing
        // named tensor is an invalid model, even if that tensor is not edited.
        for (shard, expected) in &referenced {
            let observed=shard_tensors(&source.join(shard))?;
            if &observed != expected {
                return Err(corrupt(format!("weight_map and shard tensor names disagree: {}",shard.display())));
            }
        }
    } else {
        let shard=PathBuf::from("model.safetensors");
        if !file_names.contains(&shard) {
            return Err(invalid("need model.safetensors or model.safetensors.index.json; PyTorch .bin editing is unsupported"));
        }
        for tensor in shard_tensors(&source.join(&shard))? {
            weight_map.insert(tensor,shard.clone());
        }
    }

    let recipe=read_json(recipe_path,MAX_JSON_BYTES)?;
    if recipe.get("schema").and_then(Value::as_str)!=Some("ojon.abliteration.v1") {
        return Err(invalid("expected ojon.abliteration.v1 recipe"));
    }
    let preserve=recipe.get("preserve_row_norms").and_then(Value::as_bool)
        .ok_or_else(||invalid("preserve_row_norms must be boolean"))?;
    let entries=recipe.get("edits").and_then(Value::as_array)
        .ok_or_else(||invalid("recipe edits must be an array"))?;
    if entries.is_empty() || entries.len()>MAX_EDITS { return Err(invalid("invalid number of model edits")); }
    let mut seen=HashSet::new();
    let mut grouped=BTreeMap::<PathBuf,Vec<Value>>::new();
    for edit in entries {
        let tensor=edit.get("tensor").and_then(Value::as_str).ok_or_else(||invalid("edit missing tensor name"))?;
        if !seen.insert(tensor.to_owned()) { return Err(invalid("duplicate edit tensor name")); }
        let shard=weight_map.get(tensor).ok_or_else(||invalid(format!("unknown tensor in safetensors index: {tensor}")))?;
        grouped.entry(shard.clone()).or_default().push(edit.clone());
    }
    let mut shards=Vec::new();
    for (relative, edits) in grouped {
        let grouped_recipe=json!({"schema":"ojon.abliteration.v1","preserve_row_norms":preserve,"edits":edits});
        // Preflight exactly the same tensor-validation logic that runs later.
        // Temporary output doesn't exist and is never created at plan time.
        let synthetic_output=parent.join(format!(".ojon-preflight-{}",relative.display()));
        prepare_abliteration_with_recipe(&source.join(&relative),&grouped_recipe,&synthetic_output)?;
        shards.push(ShardEdits { relative, count: edits.len(), recipe: grouped_recipe });
    }
    Ok(ModelAblationPlan {
        source,output,files,shards,source_bytes,vision_config_detected,indexed,total_edits:seen.len()
    })
}

struct StagingDir(PathBuf);
impl Drop for StagingDir {
    fn drop(&mut self) { let _=fs::remove_dir_all(&self.0); }
}
fn stage(parent:&Path)->io::Result<StagingDir> {
    for attempt in 0..16 {
        let stamp=SystemTime::now().duration_since(UNIX_EPOCH)
            .map_err(|_|invalid("invalid clock"))?.as_nanos();
        let path=parent.join(format!(".ojon-hf-edit-{}-{stamp}-{attempt}",std::process::id()));
        match fs::create_dir(&path) {
            Ok(())=>return Ok(StagingDir(path)),
            Err(e) if e.kind()==ErrorKind::AlreadyExists=>continue,
            Err(e)=>return Err(e)
        }
    }
    Err(invalid("cannot create output staging directory"))
}

#[derive(Debug)]
pub struct ModelAblationResult {
    pub output: PathBuf,
    pub files: usize,
    pub source_bytes: u64,
    pub edited_tensors: usize,
    pub edited_shards: usize,
}
impl ModelAblationResult {
    pub fn as_json(&self)->Value {
        json!({
            "schema":"ojon.model_abliteration_result.v1",
            "output":self.output.to_string_lossy(),
            "model_files":self.files,
            "source_bytes":self.source_bytes,
            "edited_tensors":self.edited_tensors,
            "edited_shards":self.edited_shards,
            "checkpoint_packaged":true,
            "model_load":"NOT_TESTED",
            "refusal_rate":"NOT_EVALUATED",
            "vision_qa":"NOT_EVALUATED",
            "status":"MODEL_FOLDER_WRITTEN"
        })
    }
}

/// Copy every regular file and edit the selected weight shards in a staging
/// directory. Vision/projector weights and processor metadata remain unchanged.
pub fn run_model_abliteration(plan:&ModelAblationPlan,mut log:impl FnMut(&str))->io::Result<ModelAblationResult> {
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists,"output model folder already exists"));
    }
    let staging=stage(plan.output.parent().ok_or_else(||invalid("output parent missing"))?)?;
    let edited:BTreeMap<&Path,&ShardEdits>=plan.shards.iter()
        .map(|s|(s.relative.as_path(),s)).collect();
    for file in &plan.files {
        let src=plan.source.join(&file.relative);
        let dst=staging.0.join(&file.relative);
        let parent=dst.parent().ok_or_else(||invalid("bad model sidecar path"))?;
        fs::create_dir_all(parent)?;
        let current=fs::symlink_metadata(&src)?;
        if !current.is_file() || current.file_type().is_symlink() || current.len()!=file.bytes {
            return Err(invalid(format!("model source file changed after planning: {}",src.display())));
        }
        if let Some(change)=edited.get(file.relative.as_path()) {
            let patch=prepare_abliteration_with_recipe(&src,&change.recipe,&dst)?;
            run_abliteration(&patch,|msg|log(msg))?;
        } else {
            let copied=fs::copy(&src,&dst)?;
            if copied!=file.bytes { return Err(corrupt("model sidecar copy incomplete")); }
        }
    }
    // Put an honest receipt next to the model, without changing HF's weight
    // map or other metadata. The editor does NOT claim model-load validation.
    let receipt=staging.0.join("ojon-edit-receipt.json");
    if receipt.exists() { return Err(invalid("source checkpoint contains reserved Ojon receipt filename")); }
    let mut file=File::create(&receipt)?;
    file.write_all(serde_json::to_string_pretty(&json!({
        "schema":"ojon.model_edit_receipt.v1",
        "source":plan.source.to_string_lossy(),
        "edited_tensors":plan.total_edits,
        "edited_shards":plan.shards.iter().map(|s|json!({"shard":s.relative.to_string_lossy(),"edits":s.count})).collect::<Vec<_>>(),
        "all_untouched_files_copied":true,
        "original_weights_untouched":true,
        "inference":"NOT_TESTED",
        "refusal_rate":"NOT_EVALUATED",
        "vision_qa":"NOT_EVALUATED",
        "quantization":"NOT_RUN"
    }))?.as_bytes())?;
    file.sync_all()?;
    if plan.output.symlink_metadata().is_ok() {
        return Err(io::Error::new(ErrorKind::AlreadyExists,"output folder appeared while editing"));
    }
    fs::rename(&staging.0,&plan.output)?;
    log(&format!("[ojon] published complete HF folder: {}",plan.output.display()));
    Ok(ModelAblationResult {
        output:plan.output.clone(),files:plan.files.len()+1,source_bytes:plan.source_bytes,
        edited_tensors:plan.total_edits,edited_shards:plan.shards.len()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scratch()->PathBuf {
        let stamp=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let d=std::env::temp_dir().join(format!("ojon-hf-folder-{}-{stamp}",std::process::id()));
        fs::create_dir(&d).unwrap();d
    }
    fn shard(path:&Path, name:&str, vals:&[f32]) {
        let h=json!({name:{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}});
        let header=serde_json::to_vec(&h).unwrap();
        let mut f=File::create(path).unwrap();
        f.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&header).unwrap();
        for val in vals { f.write_all(&val.to_le_bytes()).unwrap(); }
    }
    fn setup(root:&Path)->(PathBuf,PathBuf) {
        let src=root.join("original");fs::create_dir(&src).unwrap();
        fs::write(src.join("config.json"),r#"{"model_type":"qwen3_vl","vision_config":{"hidden_size":8}}"#).unwrap();
        fs::write(src.join("tokenizer.json"),r#"{"dummy":true}"#).unwrap();
        fs::create_dir(src.join("vision")).unwrap();
        fs::write(src.join("vision/processor_config.json"),r#"{"image":true}"#).unwrap();
        fs::write(src.join("vision/visual.bin"),[1u8,2,3,4]).unwrap();
        let a="model.layers.0.self_attn.o_proj.weight";
        let b="model.layers.1.mlp.down_proj.weight";
        let s1="model-00001-of-00002.safetensors";
        let s2="model-00002-of-00002.safetensors";
        shard(&src.join(s1),a,&[2.0,0.0,0.0,2.0]);
        shard(&src.join(s2),b,&[3.0,0.0,0.0,3.0]);
        fs::write(src.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({"metadata":{"total_size":32},"weight_map":{a:s1,b:s2}})).unwrap()).unwrap();
        let recipe=root.join("directions.json");
        fs::write(&recipe,serde_json::to_vec(&json!({
            "schema":"ojon.abliteration.v1","preserve_row_norms":false,
            "edits":[
                {"tensor":a,"direction":[1.0,0.0],"strength":1.0},
                {"tensor":b,"direction":[0.0,1.0],"strength":1.0}
            ]
        })).unwrap()).unwrap();
        (src,recipe)
    }
    #[test]
    fn indexed_vlm_folder_preserves_vision_and_index_and_edits_both_shards() {
        let root=scratch();let (src,recipe)=setup(&root);
        let output=root.join("edited");
        let plan=prepare_model_abliteration(&src,&recipe,&output).unwrap();
        assert_eq!(plan.total_edits,2);
        assert_eq!(plan.shards.len(),2);
        assert!(plan.as_json()["vision_config_detected"].as_bool().unwrap());
        assert!(!output.exists());
        let result=run_model_abliteration(&plan,|_|{}).unwrap();
        assert_eq!(result.edited_shards,2);
        assert!(result.output.join("ojon-edit-receipt.json").is_file());
        for path in ["vision/visual.bin","vision/processor_config.json","tokenizer.json",
            "config.json","model.safetensors.index.json"] {
            assert_eq!(fs::read(src.join(path)).unwrap(),fs::read(output.join(path)).unwrap());
        }
        for (name,p) in [
            ("model-00001-of-00002.safetensors",0usize),
            ("model-00002-of-00002.safetensors",1usize)
        ] {
            let old=fs::read(src.join(name)).unwrap();
            let new=fs::read(output.join(name)).unwrap();
            assert_ne!(old,new);
            let n=8+u64::from_le_bytes(new[0..8].try_into().unwrap()) as usize;
            let values=new[n..n+16].chunks_exact(4)
                .map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect::<Vec<_>>();
            if p==0 { assert_eq!(values,[0.0,0.0,0.0,2.0]); }
            else { assert_eq!(values,[3.0,0.0,0.0,0.0]); }
        }
        assert!(run_model_abliteration(&plan,|_|{}).is_err());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn missing_indexed_tensor_or_bad_source_never_creates_output() {
        let root=scratch();let (src,recipe)=setup(&root);
        let output=root.join("edited");
        let idx=src.join("model.safetensors.index.json");
        fs::write(&idx,serde_json::to_vec(&json!({"weight_map":{
            "nonexistent.tensor":"model-00001-of-00002.safetensors"
        }})).unwrap()).unwrap();
        assert!(prepare_model_abliteration(&src,&recipe,&output).is_err());
        assert!(!output.exists());
        fs::remove_dir_all(root).unwrap();
    }
    #[test]
    fn rejects_output_nested_in_input_and_invalid_tensor() {
        let root=scratch();let (src,recipe)=setup(&root);
        assert!(prepare_model_abliteration(&src,&recipe,&src.join("nested")).is_err());
        let bad=root.join("bad.json");
        fs::write(&bad,serde_json::to_vec(&json!({
            "schema":"ojon.abliteration.v1","preserve_row_norms":false,
            "edits":[{"tensor":"vision_tower.weight","direction":[1.0],"strength":1.0}]
        })).unwrap()).unwrap();
        assert!(prepare_model_abliteration(&src,&bad,&root.join("out")).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
