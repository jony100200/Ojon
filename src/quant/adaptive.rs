//! Adaptive precision search. Real GGUF quantization + held-out task evaluation.
//! Tensor sensitivities are externally measured inputs and must be auditable;
//! no heuristic guess is described as observed activation sensitivity.
use crate::gguf::{inspect_gguf, inspect_gguf_tensors, read_imatrix_energy};
use crate::llama::{prepare_gguf, quantize_gguf_with_log, Precision};
use crate::task_eval::{prepare_eval_with_server, run_eval_reusing_baseline};
use serde_json::{json,Value};
use std::collections::{HashMap,HashSet};
use std::fs::{self,OpenOptions};
use std::io::{self,ErrorKind,Write};
use std::path::{Path,PathBuf};
use std::time::{SystemTime,UNIX_EPOCH};

fn invalid(s:impl Into<String>)->io::Error{io::Error::new(ErrorKind::InvalidInput,s.into())}

const TYPES:[&str;5]=["q2_k","q3_k","q4_k","q5_k","q6_k"];
const BPW:[f64;5]=[2.5625,3.4375,4.5,5.5,6.5625];

#[derive(Clone,Debug)]
pub struct TensorScore {pub name:String,pub elements:u64,pub sensitivity:f64}

#[derive(Clone,Debug)]
pub struct AdaptivePlan{
    pub source:PathBuf,pub scores_file:PathBuf,pub scores:Vec<TensorScore>,
    pub output_dir:PathBuf,pub suite:PathBuf,pub quantizer:PathBuf,
    pub eval_tool:PathBuf,pub server_tool:Option<PathBuf>,pub imatrix:Option<PathBuf>,pub targets:Vec<f64>,
    pub threads:usize,pub gpu_layers:i32,pub device:Option<String>,pub tokens:u32,
    pub max_absolute_loss_pct:f64,
    pub reference:PathBuf,
}

/// Derive a ranked importance proxy directly from a real GGUF llama-imatrix.
/// No source weights are loaded; this is an activation-energy proxy, not actual
/// perturbation sensitivity. Every candidate still needs held-out task tests.
pub fn extract_sensitivity_from_imatrix(
    source:&Path,imatrix:&Path,output:&Path
)->io::Result<Value>{
    let source=source.canonicalize()?;
    let info=inspect_gguf(&source)?;
    if !matches!(info.file_type,Some(0|1|32)){
        return Err(invalid("sensitivity source must be original F32/F16/BF16 GGUF"));
    }
    let energy=read_imatrix_energy(imatrix)?;
    let by_name:HashMap<String,f64>=energy.into_iter().collect();
    let catalog=inspect_gguf_tensors(&source)?;
    let mut rows=Vec::new();
    for t in catalog{
        if t.shape.len()<2||t.elements<65536||t.shape[0]%256!=0
            ||!t.name.ends_with(".weight")||!t.name.starts_with("blk."){continue;}
        if let Some(&score)=by_name.get(&t.name) {
            rows.push((t.name,score));
        }
    }
    if rows.is_empty(){return Err(invalid("imatrix has no matched K-quantizable source tensors"));}
    // Stable rank-based normalization avoids pretending raw activation units
    // from unrelated tensor families share one calibrated loss scale.
    rows.sort_by(|a,b|a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let n=rows.len();
    let tensors=rows.into_iter().enumerate().map(|(idx,(name,energy))|{
        let scaled=(idx+1) as f64/n as f64;
        json!({"name":name,"sensitivity":scaled,"activation_energy_proxy":energy})
    }).collect::<Vec<_>>();
    let result=json!({
        "schema":"ojon.tensor_sensitivity.v1",
        "source_file_bytes":info.file_bytes,
        "architecture":info.architecture,
        "calibration_protocol":"GGUF llama-imatrix mean squared activation-energy RANK proxy; not direct perturbation sensitivity",
        "calibration_imatrix":imatrix.canonicalize()?.to_string_lossy(),
        "verified_task_sensitivity":false,
        "tensors":tensors
    });
    if output.extension().and_then(|x|x.to_str())!=Some("json"){
        return Err(invalid("sensitivity output must be a new JSON file"));
    }
    let mut file=OpenOptions::new().create_new(true).write(true).open(output)?;
    file.write_all(&serde_json::to_vec_pretty(&result).map_err(io::Error::other)?)?;
    file.sync_all()?;
    Ok(result)
}

pub fn parse_tensor_sensitivity(source:&Path, file:&Path)->io::Result<Vec<TensorScore>>{
    let source=source.canonicalize()?;
    let base=inspect_gguf(&source)?;
    if !matches!(base.file_type,Some(0|1|32)){
        return Err(invalid("adaptive source must be original F32/F16/BF16 GGUF"));
    }
    let size=fs::metadata(file)?.len();
    if size==0||size>4*1024*1024{return Err(invalid("sensitivity file must be 1 byte..4 MiB"));}
    let raw:Value=serde_json::from_slice(&fs::read(file)?).map_err(io::Error::other)?;
    if raw["schema"]!="ojon.tensor_sensitivity.v1"{
        return Err(invalid("tensor sensitivity schema must be ojon.tensor_sensitivity.v1"));
    }
    if raw["source_file_bytes"].as_u64()!=Some(base.file_bytes)
        || raw["architecture"].as_str()!=base.architecture.as_deref(){
        return Err(invalid("sensitivity evidence does not match original model bytes/architecture"));
    }
    if raw["calibration_protocol"].as_str().is_none_or(|s|s.trim().is_empty()){
        return Err(invalid("sensitivity provenance requires calibration_protocol"));
    }
    let tensors=raw["tensors"].as_array().filter(|a|!a.is_empty()&&a.len()<=4096)
        .ok_or_else(||invalid("require 1..=4096 sensitivity entries"))?;
    let catalog=inspect_gguf_tensors(&source)?;
    let allowed:HashMap<String,(u64,bool)>=catalog.into_iter()
        .map(|t|{
            let eligible=t.shape.len()>=2&&t.elements>=65536&&t.shape[0]%256==0
                &&t.name.ends_with(".weight") &&t.name.starts_with("blk.");
            (t.name,(t.elements,eligible))
        }).collect();
    let mut seen=HashSet::new();let mut scores=Vec::new();
    for item in tensors{
        let name=item["name"].as_str().ok_or_else(||invalid("tensor name missing"))?;
        if !seen.insert(name.to_owned()){return Err(invalid("duplicate sensitivity entry"));}
        let &(elements,eligible)=allowed.get(name).ok_or_else(||invalid(format!("tensor {name} does not exist")))?;
        if !eligible{return Err(invalid(format!("tensor {name} is not safe for K-quant override")));}
        let sensitivity=item["sensitivity"].as_f64().filter(|v|v.is_finite()&&*v>0.0&&*v<=1.0)
            .ok_or_else(||invalid("sensitivity must be finite and in (0,1]"))?;
        scores.push(TensorScore{name:name.into(),elements,sensitivity});
    }
    Ok(scores)
}

/// Greedy measured-importance allocation under a weighted bits-per-weight
/// budget for the selected tensors; NOT total model BPW or predicted output size.
pub fn allocate_bits(scores:&[TensorScore],target:f64)->io::Result<Value>{
    if scores.is_empty()||!target.is_finite()||!(BPW[0]..=BPW[4]).contains(&target){
        return Err(invalid("target must be in 2.5625..=6.5625 BPW with nonempty scores"));
    }
    let total=scores.iter().try_fold(0u64,|a,v|a.checked_add(v.elements))
        .ok_or_else(||invalid("total tensor elements overflow"))?;
    if total==0{return Err(invalid("empty tensors"));}
    let mut levels=vec![0usize;scores.len()];
    let budget=total as f64*target;
    let mut current=total as f64*BPW[0];
    for _ in 0..(4*scores.len()){
        let mut best:Option<(usize,f64)>=None;
        for (i,t) in scores.iter().enumerate(){
            let level=levels[i];if level+1>=BPW.len(){continue;}
            let incremental=t.elements as f64*(BPW[level+1]-BPW[level]);
            if current+incremental>budget+0.0001 {continue;}
            // Diminishing quality gains for each successive precision upgrade.
            let prev_error=(-BPW[level]).exp();
            let next_error=(-BPW[level+1]).exp();
            let gain=t.sensitivity*(prev_error-next_error)/(BPW[level+1]-BPW[level]);
            if best.as_ref().is_none_or(|(_,s)|gain>*s){best=Some((i,gain));}
        }
        let Some((index,_))=best else{break;};
        let level=levels[index];
        current+=scores[index].elements as f64*(BPW[level+1]-BPW[level]);
        levels[index]+=1;
    }
    let assignments=scores.iter().enumerate().map(|(i,t)|json!({
        "name":t.name,"elements":t.elements,"sensitivity":t.sensitivity,
        "type":TYPES[levels[i]],"nominal_bpw":BPW[levels[i]]
    })).collect::<Vec<_>>();
    Ok(json!({
        "schema":"ojon.tensor_allocation.v1","target_measured_tensor_bpw":target,
        "estimated_measured_tensor_bpw":current/total as f64,
        "scored_tensor_count":scores.len(),"scored_weight_elements":total,
        "assignments":assignments,
        "warning":"Size estimate covers only listed tensors; actual GGUF size, group overhead, fallback types and runtime must be measured."
    }))
}

pub fn prepare_adaptive(
    source:&Path,scores:&Path,output_dir:&Path,suite:&Path,
    quantizer:&Path,eval_tool:&Path,imatrix:Option<&Path>,
    targets:&[f64],threads:usize,gpu_layers:i32,device:Option<&str>,tokens:u32,
    max_absolute_loss_pct:f64
)->io::Result<AdaptivePlan>{
    let source=source.canonicalize()?;
    let scores_file=scores.canonicalize()?;
    let parsed=parse_tensor_sensitivity(&source,&scores_file)?;
    if !output_dir.is_dir(){return Err(invalid("adaptive output directory must already exist"));}
    let output_dir=output_dir.canonicalize()?;
    if output_dir==source.parent().unwrap_or(Path::new("")) {
        return Err(invalid("adaptive outputs require a separate directory from source"));
    }
    if targets.is_empty()||targets.len()>8{return Err(invalid("require 1..=8 budget targets"));}
    let mut bits=Vec::new();
    for &b in targets{
        allocate_bits(&parsed,b)?;
        if bits.iter().any(|n:&f64|(*n-b).abs()<0.00001){return Err(invalid("duplicate adaptive bitrate target"));}
        bits.push(b);
    }
    bits.sort_by(|a,b|a.total_cmp(b));
    if threads==0{return Err(invalid("threads must be >0"));}
    if !max_absolute_loss_pct.is_finite()||!(0.0..=10.0).contains(&max_absolute_loss_pct){
        return Err(invalid("max absolute accuracy loss must be 0..=10 percentage points"));
    }
    let imatrix=imatrix.map(Path::canonicalize).transpose()?;
    if let Some(i)=&imatrix {
        if !i.is_file()||i.metadata()?.len()==0{return Err(invalid("imatrix missing/empty"));}
    }
    // Validates the same task suite used later for every bitrate.
    let suite=suite.canonicalize()?;
    crate::task_eval::parse_eval_suite(&suite)?;
    if quantizer.as_os_str().is_empty()||eval_tool.as_os_str().is_empty(){
        return Err(invalid("quantizer and llama-cli must be provided"));
    }
    if !(1..=1024).contains(&tokens)||!(-1..=512).contains(&gpu_layers){
        return Err(invalid("evaluation token or layer count out of range"));
    }
    if let Some(d)=device {
        if d.is_empty()||d.len()>80||!d.bytes().all(|v|
            v.is_ascii_alphanumeric()||v==b'_'||v==b'-'||v==b'/'){
            return Err(invalid("invalid device"));
        }
    }
    Ok(AdaptivePlan{reference:source.clone(),source,scores_file,scores:parsed,output_dir,suite,
        quantizer:quantizer.into(),eval_tool:eval_tool.into(),server_tool:None,imatrix,targets:bits,
        threads,gpu_layers,device:device.map(str::to_owned),tokens,max_absolute_loss_pct})
}

/// Guided searches use the existing quantized Q4 benchmark reference.
pub fn prepare_adaptive_against_reference(
    source:&Path,reference:&Path,scores:&Path,output_dir:&Path,suite:&Path,
    quantizer:&Path,eval_tool:&Path,imatrix:Option<&Path>,
    targets:&[f64],threads:usize,gpu_layers:i32,device:Option<&str>,tokens:u32,
    max_absolute_loss_pct:f64
)->io::Result<AdaptivePlan>{
    let mut plan=prepare_adaptive(source,scores,output_dir,suite,quantizer,eval_tool,
        imatrix,targets,threads,gpu_layers,device,tokens,max_absolute_loss_pct)?;
    let (_,matched,_)=crate::quant_parity::inspect_compatible_models(source,reference)?;
    plan.reference=matched;
    Ok(plan)
}
impl AdaptivePlan{
    pub fn as_json(&self)->Value{
        json!({
            "schema":"ojon.adaptive_plan.v1","source":self.source.to_string_lossy(),
            "score_source":self.scores_file.to_string_lossy(),
            "sensitivity_verified_by_ojon":false,
            "output_dir":self.output_dir.to_string_lossy(),
            "suite":self.suite.to_string_lossy(),
            "evaluation_server":self.server_tool.as_ref().map(|p|p.to_string_lossy().to_string()),
            "quality_reference":self.reference.to_string_lossy(),
            "targets":self.targets.iter().map(|&t|allocate_bits(&self.scores,t).unwrap_or(Value::Null)).collect::<Vec<_>>(),
            "default_format":"GGUF with upstream Q2_K through Q6_K tensor overrides",
            "quality_tolerance_absolute_percentage_points":self.max_absolute_loss_pct,
            "runtime_memory_fit_verified":false,
            "status":"PLAN_ONLY"
        })
    }
}

fn regex_literal(name:&str)->String{
    let mut pattern=String::from("^");
    for ch in name.chars(){
        if matches!(ch,'.'|'+'|'*'|'?'|'('|')'|'['|']'|'{'|'}'|'|'|'^'|'$'|'\\'){
            pattern.push('\\');
        }
        pattern.push(ch);
    }
    pattern.push('$');pattern
}
fn write_recipe(path:&Path,allocation:&Value,extra:&str)->io::Result<()>{
    let entries=allocation["assignments"].as_array().ok_or_else(||invalid("missing assignments"))?;
    let mut text=String::new();
    for row in entries{
        let name=row["name"].as_str().ok_or_else(||invalid("missing tensor"))?;
        let q=row["type"].as_str().ok_or_else(||invalid("missing tensor precision"))?;
        text.push_str(&regex_literal(name));text.push('=');text.push_str(q);text.push('\n');
    }
    // MoE router preservation: pin routing tensors to Q6_K so a guided search
    // can never make expert selection a regression source. Empty for dense.
    text.push_str(extra);
    let mut file=OpenOptions::new().create_new(true).write(true).open(path)?;
    file.write_all(text.as_bytes())?;
    file.sync_all()
}

/// Run real quantizations in ascending budget order, run same held-out suite
/// against the original and every candidate, keep the smallest measured pass.
pub fn run_adaptive(plan:&AdaptivePlan,mut log:impl FnMut(&str))->io::Result<Value>{
    let started=SystemTime::now().duration_since(UNIX_EPOCH)
        .map_err(|_|invalid("system time invalid"))?.as_nanos();
    let session=format!("ojon-adaptive-{}-{started}",std::process::id());
    let suite_id=crate::task_eval::parse_eval_suite(&plan.suite)?.id;
    let router_recipe=crate::llama::moe_router_pinning_recipe(
        &crate::gguf::inspect_gguf_tensors(&plan.source)?).unwrap_or_default();
    let mut rows=Vec::new();
    let mut best:Option<(String,u64)>=None;
    let mut baseline_tasks:Option<Value>=None;
    for (idx,target) in plan.targets.iter().enumerate(){
        let label=format!("adaptive_{idx}");
        let prefix=format!("{session}-{:03}",(target*100.0).round() as u32);
        let output=plan.output_dir.join(format!("{prefix}.gguf"));
        let recipe=plan.output_dir.join(format!("{prefix}.tensor-types.txt"));
        let evidence=plan.output_dir.join(format!("{prefix}.quality.json"));
        let alloc=allocate_bits(&plan.scores,*target)?;
        // Resume: reuse an existing verified candidate instead of re-quantizing.
        let reusable=match crate::quant_parity::verify_reusable_candidate(&plan.source,&output){
            Ok(reusable)=>reusable,
            Err(e)=>{
                rows.push(json!({"target":target,"status":"resume_verification_failed",
                    "reason":e.to_string(),"output":output.to_string_lossy()}));
                continue;
            }
        };
        let size=if let Some(bytes)=reusable{
            log(&format!("[ojon] resume: reusing existing candidate {}",output.display()));
            bytes
        }else{
            write_recipe(&recipe,&alloc,&router_recipe)?;
            log(&format!("[ojon] quantizing tensor budget {:.3} BPW",target));
            let converted=(||{
                let mut q=prepare_gguf(&plan.source,&output,&plan.quantizer,Precision::Q4KM,plan.threads)?
                    .with_tensor_type_file(&recipe)?;
                if let Some(i)=&plan.imatrix {q=q.with_imatrix(i)?;}
                quantize_gguf_with_log(&q,|line|log(line))
            })();
            match converted{
                Err(e)=>{
                    rows.push(json!({"target":target,"status":"conversion_failed","reason":e.to_string(),
                        "recipe":recipe.to_string_lossy()}));
                    continue;
                }
                Ok(done)=>done.output_info.file_bytes,
            }
        };
        let prior=match crate::quant_parity::verify_prior_evidence(&evidence,
            &plan.reference,&suite_id,&label,&output){
            Ok(prior)=>prior,
            Err(e)=>{
                rows.push(json!({"target":target,"status":"existing_evidence_not_reusable",
                    "reason":e.to_string(),"bytes":size,"output":output.to_string_lossy()}));
                continue;
            }
        };
        let prior_used=prior.is_some();
        let evaluated=if let Some(report)=prior{
            log("[ojon] resume: reusing verified held-out task evidence (baseline unchanged)");
            Ok(report)
        }else{
            (||{
                let candidate=vec![(label.clone(),output.clone())];
                let test=prepare_eval_with_server(&plan.suite,&plan.reference,&candidate,&plan.eval_tool,
                plan.server_tool.as_deref(),
                    plan.gpu_layers,plan.device.as_deref(),plan.tokens,&evidence)?;
                run_eval_reusing_baseline(&test,baseline_tasks.as_ref(),|line|log(line))
            })()
        };
        match evaluated{
            Err(e)=>rows.push(json!({"target":target,"status":"quality_evaluation_failed",
                "reason":e.to_string(),"bytes":size,
                "output":output.to_string_lossy()})),
            Ok(report)=>{
                if baseline_tasks.is_none(){baseline_tasks=Some(report.clone());}
                let base=&report["baseline"]["metrics"];
                let cand=&report["candidates"][0]["metrics"];
                let mut comparisons=Vec::new();let mut passes=true;
                for metric in ["coding_accuracy","reasoning_accuracy","tool_call_accuracy"] {
                    let b=base[metric].as_f64().unwrap_or(0.0);
                    let c=cand[metric].as_f64().unwrap_or(0.0);
                    // A suite the baseline cannot solve is not evidence of preserved ability.
                    let pass=b>0.0 && c+1e-10>=b-plan.max_absolute_loss_pct;
                    if !pass{passes=false;}
                    comparisons.push(json!({"metric":metric,"baseline":b,"candidate":c,
                        "allowed_drop_points":plan.max_absolute_loss_pct,"pass":pass}));
                }
                if passes&&best.as_ref().is_none_or(|(_,bytes)|size<*bytes){
                    best=Some((output.to_string_lossy().to_string(),size));
                }
                rows.push(json!({"target":target,"status":"measured_task_suite",
                    "output":output.to_string_lossy(),"bytes":size,"task_quality_pass":passes,
                    "resumed_quantization":reusable.is_some(),
                    "resumed_task_evidence":prior_used,
                    "moe_router_pinned":!router_recipe.is_empty(),
                    "comparisons":comparisons,"quality_receipt":evidence.to_string_lossy(),
                    "tensor_type_file":recipe.to_string_lossy()}));
            }
        }
    }
    let report=json!({"schema":"ojon.adaptive_search_result.v1",
        "source":plan.source.to_string_lossy(),"quality_reference":plan.reference.to_string_lossy(),"suite":plan.suite.to_string_lossy(),
        "candidates":rows,"provisional_smallest_task_passing_output":best.as_ref().map(|p|p.0.clone()),
        "provisional_smallest_bytes":best.map(|p|p.1),
        "hardware_residency_verified":false,"near_lossless_generalization_verified":false,
        "warning":"Winner passes only requested held-out deterministic tasks; verify target GPU RAM/VRAM and larger real coding tasks before deployment."});
    let receipt=plan.output_dir.join(format!("{session}.json"));
    let mut file=OpenOptions::new().create_new(true).write(true).open(&receipt)?;
    file.write_all(&serde_json::to_vec_pretty(&report).map_err(io::Error::other)?)?;
    file.sync_all()?;
    Ok(report)
}
#[cfg(test)]
mod tests{
    use super::*;
    fn sample()->Vec<TensorScore>{vec![
        TensorScore{name:"blk.0.attn_q.weight".into(),elements:65536,sensitivity:0.95},
        TensorScore{name:"blk.0.ffn_up.weight".into(),elements:65536,sensitivity:0.1},
        TensorScore{name:"blk.1.attn_q.weight".into(),elements:65536,sensitivity:0.6}
    ]}
    #[test]
    fn budget_and_priority(){
        let scores=sample();let low=allocate_bits(&scores,2.5625).unwrap();
        assert_eq!(low["assignments"][0]["type"],"q2_k");
        let med=allocate_bits(&scores,3.4375).unwrap();
        let bpw=med["estimated_measured_tensor_bpw"].as_f64().unwrap();
        assert!(bpw<=3.4375+1e-8&&bpw>2.5625);
        assert!(allocate_bits(&scores,1.0).is_err());
        assert!(allocate_bits(&scores,f64::NAN).is_err());
    }
    #[test]
    fn escape_tensor_regex(){
        assert_eq!(regex_literal("blk.0.attn_q.weight"),r"^blk\.0\.attn_q\.weight$");
    }
}
