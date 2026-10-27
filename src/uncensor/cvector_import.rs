//! Import a measured llama.cpp residual-stream control vector as an Ojon edit recipe.
//! The matching output-projection edit is an experiment, not validated refusal reduction.

use crate::gguf::{inspect_gguf, inspect_gguf_tensors};
use serde_json::{json, Value};
use std::fs::File;
use std::io::{self, ErrorKind, Read, Seek, SeekFrom};
use std::path::Path;

fn invalid(message:impl Into<String>)->io::Error {
    io::Error::new(ErrorKind::InvalidInput,message.into())
}

/// `direction.1` is the callback's first layer output and maps to block 0.
/// The residual direction is projected from the chosen block's attention
/// output matrix only when its output width matches the vector exactly.
pub fn recipe_from_control_vector(path:&Path,block:usize,strength:f64)->io::Result<Value> {
    recipe_from_control_vector_blocks(path,&[(block,strength)])
}

/// Multi-block recipe: one edit per (block, strength) pair. Blocks must be
/// unique, and gguf_edit's recipe limit of 256 edits still applies.
pub fn recipe_from_control_vector_blocks(path:&Path,edits:&[(usize,f64)])->io::Result<Value> {
    if edits.is_empty() || edits.len()>256 {
        return Err(invalid("require 1..=256 (block, strength) edit pairs"));
    }
    let mut seen=std::collections::HashSet::new();
    for (block,strength) in edits {
        if *block>1023 || !strength.is_finite() || *strength<=0.0 || *strength>1.0 {
            return Err(invalid("block must be <=1023 and strength must be >0 and <=1"));
        }
        if !seen.insert(*block){return Err(invalid("duplicate block in edit list"));}
    }
    let summary=inspect_gguf(path)?;
    if summary.architecture.as_deref()!=Some("controlvector") || summary.big_endian
        || summary.tensor_count==0 || summary.tensor_count>1024 {
        return Err(invalid("expected little-endian llama.cpp controlvector GGUF"));
    }
    let tensors=inspect_gguf_tensors(path)?;
    let mut out=Vec::new();
    for (block,strength) in edits {
        let name=format!("direction.{}",block+1);
        let tensor=tensors.iter().find(|t|t.name==name)
            .ok_or_else(||invalid(format!("control vector lacks {name}")))?;
        if tensor.storage_type!=0 || tensor.shape.len()!=1 || tensor.elements<2
            || tensor.elements>131072 {
            return Err(invalid("control direction must be a bounded 1D F32 tensor"));
        }
        let size=tensor.elements.checked_mul(4).ok_or_else(||invalid("control vector byte overflow"))?;
        let offset=summary.data_offset.checked_add(tensor.relative_offset)
            .ok_or_else(||invalid("control vector offset overflow"))?;
        if offset.checked_add(size).is_none_or(|end|end>summary.file_bytes) {
            return Err(invalid("control direction exceeds GGUF bounds"));
        }
        let mut file=File::open(path)?;
        file.seek(SeekFrom::Start(offset))?;
        let mut values=Vec::with_capacity(tensor.elements as usize);
        let mut bytes=[0u8;4];
        let mut norm=0.0f64;
        for _ in 0..tensor.elements {
            file.read_exact(&mut bytes)?;
            let value=f32::from_le_bytes(bytes) as f64;
            if !value.is_finite(){return Err(invalid("nonfinite control direction"));}
            norm=norm.hypot(value);
            values.push(value);
        }
        if !norm.is_finite() || norm<1e-30 {
            return Err(invalid("control direction is zero or invalid"));
        }
        let direction=values.into_iter().map(|value|value/norm).collect::<Vec<_>>();
        out.push(json!({"tensor":format!("blk.{block}.attn_output.weight"),
            "direction":direction,"strength":strength}));
    }
    Ok(json!({"schema":"ojon.abliteration.v1","preserve_row_norms":true,
        "edits":out}))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime,UNIX_EPOCH};

    #[test]
    fn imports_selected_layer_and_rejects_missing_or_zero_direction(){
        fn string(file:&mut File,text:&str){
            file.write_all(&(text.len() as u64).to_le_bytes()).unwrap();
            file.write_all(text.as_bytes()).unwrap();
        }
        let stamp=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir=std::env::temp_dir().join(format!("ojon-cvector-{}-{stamp}",std::process::id()));
        fs::create_dir(&dir).unwrap();
        let path=dir.join("control.gguf");
        let mut file=File::create(&path).unwrap();
        file.write_all(b"GGUF").unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&1u64.to_le_bytes()).unwrap();
        file.write_all(&1u64.to_le_bytes()).unwrap();
        string(&mut file,"general.architecture");
        file.write_all(&8u32.to_le_bytes()).unwrap();
        string(&mut file,"controlvector");
        string(&mut file,"direction.3");
        file.write_all(&1u32.to_le_bytes()).unwrap();
        file.write_all(&2u64.to_le_bytes()).unwrap();
        file.write_all(&0u32.to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        let pos=file.stream_position().unwrap();
        file.write_all(&vec![0u8;((32-pos%32)%32) as usize]).unwrap();
        file.write_all(&3.0f32.to_le_bytes()).unwrap();
        file.write_all(&4.0f32.to_le_bytes()).unwrap();
        drop(file);
        let recipe=recipe_from_control_vector(&path,2,0.5).unwrap();
        assert_eq!(recipe["edits"][0]["tensor"],"blk.2.attn_output.weight");
        assert_eq!(recipe["edits"][0]["direction"],json!([0.6,0.8]));
        assert!(recipe_from_control_vector(&path,1,0.5).is_err());
        assert!(recipe_from_control_vector(&path,2,1.5).is_err());
        fs::remove_dir_all(dir).unwrap();
    }
}
