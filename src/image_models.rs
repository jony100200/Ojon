//! Explicit, reviewable community image checkpoint references.
//! Public Rebel AI quantizations are derivative diffusion weights. No code
//! or weights are bundled, and no network request runs during discovery.
//! A user-approved fetch invokes their installed Hugging Face CLI.
use crate::probe::{probe_model,ModelCategory};
use crate::jobs::run_with_log;
use serde_json::{json,Value};
use std::fs;
use std::io::{self,ErrorKind};
use std::path::{Path,PathBuf};
use std::process::Command;

fn invalid(msg:impl Into<String>)->io::Error{
    io::Error::new(ErrorKind::InvalidInput,msg.into())
}
#[derive(Clone,Copy)]
struct Reference {
    key:&'static str,
    repo:&'static str,
    file:&'static str,
    category:&'static str,
    weight_format:&'static str,
    source_license:&'static str,
    model_family:&'static str,
}
const REBEL_REFERENCES:[Reference;5]=[
    Reference{key:"iris-3b-q3",repo:"realrebelai/iris-3b_GGUFs",
        file:"Iris-3B-Q3_K_S.gguf",category:"text_to_image",
        weight_format:"gguf",source_license:"apache-2.0",model_family:"iris"},
    Reference{key:"iris-3b-q4",repo:"realrebelai/iris-3b_GGUFs",
        file:"Iris-3B-Q4_K_S.gguf",category:"text_to_image",
        weight_format:"gguf",source_license:"apache-2.0",model_family:"iris"},
    Reference{key:"iris-3b-q5",repo:"realrebelai/iris-3b_GGUFs",
        file:"Iris-3B-Q5_K_S.gguf",category:"text_to_image",
        weight_format:"gguf",source_license:"apache-2.0",model_family:"iris"},
    Reference{key:"iris-3b-q6",repo:"realrebelai/iris-3b_GGUFs",
        file:"Iris-3B-Q6_K.gguf",category:"text_to_image",
        weight_format:"gguf",source_license:"apache-2.0",model_family:"iris"},
    Reference{key:"iris-3b-w4a8",repo:"realrebelai/iris-3b_GGUFs",
        file:"Iris-3B-W4A8.safetensors",category:"text_to_image",
        weight_format:"safetensors_w4a8",source_license:"apache-2.0",model_family:"iris"}
];
fn entry(r:Reference)->Value{
    json!({"id":r.key,"publisher":"Rebel AI (community quantization)",
        "upstream_model":"speridlabs/iris-3b","repository":r.repo,
        "filename":r.file,"weight_format":r.weight_format,
        "family":r.model_family,"category":r.category,
        "source_license":r.source_license,
        "model_page":format!("https://huggingface.co/{}",r.repo),
        "runtime":"ComfyUI with independently verified Iris-compatible nodes",
        "text_encoder_vae_scheduler_included":false,
        "quantization_quality_verified_by_ojon":false,
        "inference_fit_verified":false})
}
/// The catalog records upstream references, never machine-specific choices.
pub fn catalog()->Value{
    json!({"schema":"ojon.community_image_models.v1",
        "source":"https://huggingface.co/realrebelai/iris-3b_GGUFs",
        "upstream_research":"https://huggingface.co/speridlabs/iris-3b",
        "known_community_checkpoints":REBEL_REFERENCES.iter().map(|r|entry(*r)).collect::<Vec<_>>(),
        "additional_rebel_image_references":[
            "https://huggingface.co/realrebelai/Qwen-Image-2.1-Turbo_GGUFs",
            "https://github.com/RealRebelAI/ComfyUI-GGUF_KREA-2",
            "https://github.com/RealRebelAI/RebelUI"
        ],
        "status":"REFERENCE_ONLY_NOT_DOWNLOADED",
        "note":"The public file catalog is not evidence that any specific GGUF loader works on this PC."})
}
fn lookup(id:&str)->io::Result<Reference>{
    REBEL_REFERENCES.iter().copied().find(|r|r.key==id)
        .ok_or_else(||invalid("unknown community image checkpoint ID; use image-models"))
}
pub fn fetch_plan(id:&str,outdir:&Path,hf_tool:&Path)->io::Result<Value>{
    let model=lookup(id)?;
    if hf_tool.as_os_str().is_empty(){return Err(invalid("Hugging Face CLI tool path missing"));}
    let dir=outdir.canonicalize()?;
    if !dir.is_dir(){return Err(invalid("model destination must be an existing directory"));}
    let destination=dir.join(model.file);
    if destination.symlink_metadata().is_ok(){
        return Err(io::Error::new(ErrorKind::AlreadyExists,
            "image checkpoint already exists; inspect it rather than overwriting"));
    }
    Ok(json!({"schema":"ojon.image_fetch_plan.v1",
        "id":id,"model":entry(model),
        "destination":destination.to_string_lossy(),
        "command_argv":[hf_tool.to_string_lossy().to_string(),"download".to_string(),
            model.repo.to_string(),model.file.to_string(),"--local-dir".to_string(),
            dir.to_string_lossy().to_string()],
        "requires_explicit_confirmation":true,
        "network_used":false,
        "status":"PLAN_ONLY",
        "note":"Review the current model card, licensing, checksum and installed loader before deployment."}))
}
pub fn fetch_image(
    id:&str,outdir:&Path,hf_tool:&Path,approved:bool,mut log:impl FnMut(&str)
)->io::Result<Value>{
    if !approved{
        return Err(io::Error::new(ErrorKind::PermissionDenied,
            "image checkpoint downloads require explicit --confirm"));
    }
    let plan=fetch_plan(id,outdir,hf_tool)?;
    let model=lookup(id)?;
    let dir=outdir.canonicalize()?;
    let path=dir.join(model.file);
    log("[ojon] Fetching selected public Rebel AI image weights via locally installed hf CLI");
    let mut command=Command::new(hf_tool);
    command.args(["download",model.repo,model.file,"--local-dir"]).arg(&dir);
    run_with_log(&mut command,|line|log(line))?;
    let info=probe_model(&path)?;
    if info.category!=ModelCategory::Diffusion
        ||!(info.format=="gguf"||info.format=="safetensors_single_file"){
        return Err(invalid("downloaded file could not be identified as diffusion weights; no inference claim"));
    }
    Ok(json!({"schema":"ojon.image_fetch_result.v1",
        "id":id,"source":plan["model"]["model_page"],
        "destination":path.to_string_lossy(),
        "bytes":fs::metadata(&path)?.len(),
        "format":info.format,"category":info.category.as_str(),
        "download_completed":true,"loader_verified":false,
        "visual_quality_verified":false}))
}
/// Matched-image comparison is an *experiment plan*, not a metric. The user
/// must render actual images in an architecture-compatible pipeline before
/// any comparison claims can be made.
pub fn image_comparison_plan(
    baseline:&Path,candidate:&Path,workflow:&Path,recipe:&Path
)->io::Result<Value>{
    let a=probe_model(baseline)?;
    let b=probe_model(candidate)?;
    if a.path==b.path{return Err(invalid("baseline and candidate must be different files"));}
    if a.category!=ModelCategory::Diffusion||b.category!=ModelCategory::Diffusion{
        return Err(invalid("image comparison needs two recognized diffusion checkpoints"));
    }
    let workflow=workflow.canonicalize()?;
    let recipe=recipe.canonicalize()?;
    if !workflow.is_file()||!recipe.is_file(){return Err(invalid("workflow and recipe must be existing JSON files"));}
    let read=|path:&Path|->io::Result<Value>{
        if fs::metadata(path)?.len()>4*1024*1024{return Err(invalid("image experiment JSON exceeds 4MiB"));}
        serde_json::from_slice(&fs::read(path)?)
            .map_err(|e|invalid(format!("invalid image experiment JSON: {e}")))
    };
    let graph=read(&workflow)?;
    if !graph.is_object(){return Err(invalid("ComfyUI workflow must be a JSON object"));}
    let config=read(&recipe)?;
    if config["schema"]!="ojon.image_experiment.v1"{
        return Err(invalid("recipe schema must be ojon.image_experiment.v1"));
    }
    let str_field=|key:&str|->io::Result<&str>{
        config[key].as_str().filter(|x|!x.trim().is_empty()&&x.len()<=8192)
            .ok_or_else(||invalid(format!("missing or invalid image experiment field {key}")))
    };
    for key in ["prompt","sampler","scheduler","text_encoder","vae"]{str_field(key)?;}
    let steps=config["steps"].as_u64().filter(|x|(1..=200).contains(x))
        .ok_or_else(||invalid("image experiment steps must be 1..200"))?;
    let cfg=config["cfg"].as_f64().filter(|v|v.is_finite()&&(0.0..=30.0).contains(v))
        .ok_or_else(||invalid("image experiment cfg must be 0..30"))?;
    let width=config["width"].as_u64().filter(|v|(64..=4096).contains(v)&&v%8==0)
        .ok_or_else(||invalid("image width must be 64..4096 divisible by 8"))?;
    let height=config["height"].as_u64().filter(|v|(64..=4096).contains(v)&&v%8==0)
        .ok_or_else(||invalid("image height must be 64..4096 divisible by 8"))?;
    let seeds=config["seeds"].as_array()
        .filter(|x|!x.is_empty()&&x.len()<=32&&x.iter().all(|v|v.as_u64().is_some()))
        .ok_or_else(||invalid("recipe seeds must be 1..32 unsigned integers"))?;
    Ok(json!({"schema":"ojon.image_comparison_plan.v1",
        "baseline":{"path":a.path.to_string_lossy(),"format":a.format,"bytes":a.bytes},
        "candidate":{"path":b.path.to_string_lossy(),"format":b.format,"bytes":b.bytes},
        "size_change_percent":100.0*(b.bytes as f64/a.bytes as f64-1.0),
        "matched_parameters":{"prompt":config["prompt"],"seeds":seeds,
            "sampler":config["sampler"],"scheduler":config["scheduler"],
            "steps":steps,"cfg":cfg,"width":width,"height":height,
            "text_encoder":config["text_encoder"],"vae":config["vae"]},
        "workflow_path":workflow.to_string_lossy(),
        "rendering_started":false,
        "paired_images_rendered":false,
        "quality_verified":false,
        "source_model_family_verified":false,
        "remaining_requirements":[
            "Verify both checkpoints are derivatives of the same source architecture",
            "Verify ComfyUI checkpoint loader for each exact weight format",
            "Render matched prompts/seeds with the same encoder, VAE and sampling schedule",
            "Compare color, structure, details and prompt fidelity; record measured VRAM/time"
        ]}))
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn published_references_are_explicit_not_machine_paths(){
        assert!(catalog()["known_community_checkpoints"].as_array().unwrap().len()>=4);
        assert!(lookup("iris-3b-q4").is_ok());
        assert!(lookup("../evil").is_err());
    }
    #[test]
    fn nonexistent_destination_rejected_before_network(){
        assert!(fetch_plan("iris-3b-q4",Path::new("not-a-real-folder-ojon"),
            Path::new("hf")).is_err());
    }
}
