//! Compare smaller GGUF candidates against a *user's existing quantized*
//! reference model. Never uses lossy Q4 -> Q3 requantization as the source.
//!
//! Quantization uses a separately installed recent llama-quantize with optional
//! bounded slab buffers. Tasks run only when the user explicitly confirms.
//! Pass means "no detected regression on these particular held-out tasks",
//! NOT mathematically lossless weights or general model equivalence.

use crate::gguf::{inspect_gguf, inspect_gguf_tensors};
use crate::llama::{prepare_gguf, quantize_gguf_with_log, Precision};
use crate::ppl::{measure as measure_perplexity, PplSettings};
use crate::task_eval::{parse_eval_suite, prepare_eval, run_eval};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{self, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

fn invalid(s:impl Into<String>)->io::Error {io::Error::new(ErrorKind::InvalidInput,s.into())}
const MAX_VISION_SUITE:u64=2*1024*1024;

#[derive(Clone, Debug)]
struct VisionTask {
    id:String,
    image:PathBuf,
    prompt:String,
    expected:String,
    contains:bool,
}
#[derive(Clone, Debug)]
struct VisionSuite {tasks:Vec<VisionTask>}
fn vision_suite(path:&Path)->io::Result<VisionSuite>{
    let path=path.canonicalize()?;
    let len=fs::metadata(&path)?.len();
    if len==0||len>MAX_VISION_SUITE{return Err(invalid("vision suite must be 1 byte..2MiB"));}
    let data:Value=serde_json::from_slice(&fs::read(&path)?)
        .map_err(|e|invalid(format!("invalid vision suite JSON: {e}")))?;
    if data.get("schema").and_then(Value::as_str)!=Some("ojon.vision_suite.v1"){
        return Err(invalid("expected ojon.vision_suite.v1"));
    }
    let arr=data.get("tasks").and_then(Value::as_array)
        .ok_or_else(||invalid("vision tasks missing"))?;
    if arr.len()<3||arr.len()>128{return Err(invalid("need 3..128 real image tasks"));}
    let parent=path.parent().ok_or_else(||invalid("vision suite has no parent"))?;
    let mut seen=HashSet::new();
    let mut tasks=Vec::new();
    for row in arr {
        let id=row.get("id").and_then(Value::as_str)
            .filter(|s|!s.is_empty()&&s.len()<=128)
            .ok_or_else(||invalid("vision task missing ID"))?.to_owned();
        if !seen.insert(id.clone()){return Err(invalid("duplicate vision task ID"));}
        let rel=row.get("image").and_then(Value::as_str)
            .ok_or_else(||invalid("vision task missing image path"))?;
        let path=Path::new(rel);
        let candidate=if path.is_absolute(){path.to_path_buf()}else{parent.join(path)};
        let image=candidate.canonicalize()?;
        if !image.is_file()||fs::metadata(&image)?.len()==0{
            return Err(invalid("image missing or empty"));
        }
        let prompt=row.get("prompt").and_then(Value::as_str)
            .filter(|s|!s.is_empty()&&s.len()<=8192)
            .ok_or_else(||invalid("vision prompt missing/too long"))?.to_owned();
        let expected=row.get("expected").and_then(Value::as_str)
            .filter(|s|!s.is_empty()&&s.len()<=8192)
            .ok_or_else(||invalid("vision reference answer missing"))?.to_owned();
        let contains=match row.get("match").and_then(Value::as_str).unwrap_or("exact"){
            "exact"=>false,"contains"=>true,
            _=>return Err(invalid("vision match must be exact or contains"))
        };
        tasks.push(VisionTask{id,image,prompt,expected,contains});
    }
    Ok(VisionSuite{tasks})
}

fn inspect_compatible_models(source:&Path,baseline:&Path)->io::Result<(PathBuf,PathBuf,u64)>{
    let source=source.canonicalize()?;
    let baseline=baseline.canonicalize()?;
    if source==baseline{return Err(invalid("original and Q4 reference paths must differ"));}
    let s=inspect_gguf(&source)?;
    let b=inspect_gguf(&baseline)?;
    if !matches!(s.file_type,Some(0|1|32)){
        return Err(invalid("original source must be full-precision F32/F16/BF16 GGUF; do NOT requantize existing Q4"));
    }
    if b.file_type.is_none()||matches!(b.file_type,Some(0|1|32)){
        return Err(invalid("quality reference must be an already quantized GGUF (such as Q4_K_M)"));
    }
    if s.tensor_count==0||s.split_count.unwrap_or(1)>1||b.split_count.unwrap_or(1)>1{
        return Err(invalid("need single-file GGUF original and baseline with tensors"));
    }
    if s.architecture.is_none()||s.architecture!=b.architecture{
        return Err(invalid("original and baseline GGUF architecture mismatch"));
    }
    // Reject comparisons between different model checkpoints, even if they
    // share an architecture: all tensor names and dimensions must match.
    let original:BTreeMap<_,_>=inspect_gguf_tensors(&source)?.into_iter()
        .map(|t|(t.name,t.shape)).collect();
    let reference:BTreeMap<_,_>=inspect_gguf_tensors(&baseline)?.into_iter()
        .map(|t|(t.name,t.shape)).collect();
    if original!=reference || s.tensor_count!=b.tensor_count {
        return Err(invalid("source and quantized reference tensor names/shapes disagree"));
    }
    Ok((source,baseline,b.file_bytes))
}

#[derive(Clone, Debug)]
pub struct QuantParityOptions {
    pub precisions:Vec<Precision>,
    pub imatrix:Option<PathBuf>,
    pub quantizer:PathBuf,
    pub eval_tool:PathBuf,
    pub threads:usize,
    pub max_buffer_mib:Option<u32>,
    pub gpu_layers:i32,
    pub device:Option<String>,
    pub tokens:u32,
    pub max_loss_points:f64,
    pub min_saving_percent:f64,
    pub vision_suite:Option<PathBuf>,
    pub mmproj:Option<PathBuf>,
    /// Optional same-corpus perplexity gate against the existing Q4 reference.
    pub ppl_corpus:Option<PathBuf>,
    pub ppl_tool:PathBuf,
    pub ppl_context:u32,
    pub max_ppl_regression_pct:f64,
}
impl Default for QuantParityOptions {
    fn default()->Self{
        Self{
            precisions:vec![Precision::IQ4XS,Precision::IQ3M,Precision::Q3KM],
            imatrix:None,
            quantizer:PathBuf::from("llama-quantize"),
            eval_tool:PathBuf::from("llama-cli"),
            threads:std::thread::available_parallelism().map(|n|n.get()).unwrap_or(4),
            max_buffer_mib:Some(512),
            gpu_layers:0,
            device:None,
            tokens:128,
            max_loss_points:0.0,
            min_saving_percent:5.0,
            vision_suite:None,
            mmproj:None,
            ppl_corpus:None,
            ppl_tool:PathBuf::from("llama-perplexity"),
            ppl_context:2048,
            max_ppl_regression_pct:0.0,
        }
    }
}
#[derive(Clone, Debug)]
pub struct QuantParityPlan {
    source:PathBuf,baseline:PathBuf,baseline_bytes:u64,
    suite:PathBuf,output_dir:PathBuf,options:QuantParityOptions,
    vision:Option<VisionSuite>,
}
impl QuantParityPlan{
    pub fn as_json(&self)->Value{
        json!({
            "schema":"ojon.quant_parity_plan.v1",
            "high_precision_source":self.source.to_string_lossy(),
            "current_quantized_baseline":self.baseline.to_string_lossy(),
            "baseline_bytes":self.baseline_bytes,
            "suite":self.suite.to_string_lossy(),
            "output_directory":self.output_dir.to_string_lossy(),
            "precisions":self.options.precisions.iter().map(|p|p.to_string()).collect::<Vec<_>>(),
            "imatrix":self.options.imatrix.as_ref().map(|p|p.to_string_lossy().to_string()),
            "quantizer":self.options.quantizer.to_string_lossy(),
            "eval_tool":self.options.eval_tool.to_string_lossy(),
            "max_buffer_mib":self.options.max_buffer_mib,
            "threads":self.options.threads,
            "gpu_layers":self.options.gpu_layers,
            "device":self.options.device,
            "tokens":self.options.tokens,
            "max_loss_points":self.options.max_loss_points,
            "min_saving_percent":self.options.min_saving_percent,
            "vision_suite_tasks":self.vision.as_ref().map(|v|v.tasks.len()),
            "mmproj":self.options.mmproj.as_ref().map(|p|p.to_string_lossy().to_string()),
            "ppl_corpus":self.options.ppl_corpus.as_ref().map(|p|p.to_string_lossy().to_string()),
            "max_relative_ppl_regression_pct":self.options.max_ppl_regression_pct,
            "quality_claim":"NO_EXPERIMENT_RUN",
            "memory_claim":"NONE; max_buffer_mib limits only the upstream row-slab buffer",
            "status":"PLAN_ONLY"
        })
    }
}
pub fn prepare_quant_parity(
    original:&Path,baseline:&Path,suite:&Path,output_dir:&Path,
    mut options:QuantParityOptions,
)->io::Result<QuantParityPlan>{
    let(source,baseline,baseline_bytes)=inspect_compatible_models(original,baseline)?;
    let suite=suite.canonicalize()?;
    parse_eval_suite(&suite)?;
    let output_dir=output_dir.canonicalize()?;
    if !output_dir.is_dir(){return Err(invalid("output directory must already exist"));}
    if options.threads==0||options.threads>512{return Err(invalid("threads must be 1..512"));}
    if options.precisions.is_empty()||options.precisions.len()>6{
        return Err(invalid("use 1..6 distinct lower-bit candidate types"));
    }
    let mut seen=HashSet::new();
    for precision in &options.precisions{
        if !seen.insert(precision.as_str()){return Err(invalid("duplicate candidate quantization")); }
        if !matches!(precision,
            Precision::IQ4XS|Precision::IQ3M|Precision::Q3KM|
            Precision::IQ3S|Precision::IQ3XS) {
            return Err(invalid("quant-parity only accepts candidate formats normally smaller than Q4_K_M"));
        }
    }
    if !options.max_loss_points.is_finite() || !(0.0..=100.0).contains(&options.max_loss_points){
        return Err(invalid("max capability loss must be 0..100 percentage points"));
    }
    if !options.min_saving_percent.is_finite()||!(0.0..=95.0).contains(&options.min_saving_percent){
        return Err(invalid("minimum saving percent must be 0..95"));
    }
    if !(1..=1024).contains(&options.tokens)||!(-1..=512).contains(&options.gpu_layers){
        return Err(invalid("invalid tokens or gpu-layers"));
    }
    if let Some(v)=options.max_buffer_mib{
        if !(32..=8192).contains(&v){return Err(invalid("slab buffer must be 32..8192 MiB"));}
    }
    if let Some(d)=options.device.as_deref(){
        if d.is_empty()||d.len()>80||!d.bytes().all(|b|
            b.is_ascii_alphanumeric()||matches!(b,b'_'|b'-'|b'/'|b','|b':')){
            return Err(invalid("invalid device"));
        }
    }
    if let Some(p)=options.imatrix.take(){
        let p=p.canonicalize()?;
        if !p.is_file()||fs::metadata(&p)?.len()==0{
            return Err(invalid("importance-matrix file missing"));
        }
        options.imatrix=Some(p);
    }
    let vision=if let Some(path)=options.vision_suite.as_ref(){
        let mm=options.mmproj.as_ref()
            .ok_or_else(||invalid("vision suite requires matching --mmproj GGUF"))?;
        let mm=mm.canonicalize()?;
        if inspect_gguf(&mm)?.tensor_count==0{return Err(invalid("mmproj GGUF empty"));}
        options.mmproj=Some(mm);
        Some(vision_suite(path)?)
    }else{
        if options.mmproj.is_some(){
            return Err(invalid("mmproj without vision suite cannot prove visual parity"));
        }
        None
    };
    if let Some(p)=options.ppl_corpus.take(){
        let p=p.canonicalize()?;
        let cfg=PplSettings{
            tool:options.ppl_tool.clone(),corpus:p.clone(),
            gpu_layers:options.gpu_layers.max(0) as u32,
            context:options.ppl_context,
            max_relative_degradation_pct:options.max_ppl_regression_pct,
        };
        cfg.validate()?;
        options.ppl_corpus=Some(p);
    }
    if options.quantizer.as_os_str().is_empty()||options.eval_tool.as_os_str().is_empty(){
        return Err(invalid("quantizer and evaluation CLI paths must be nonempty"));
    }
    Ok(QuantParityPlan{source,baseline,baseline_bytes,suite,output_dir,options,vision})
}
fn matches_vision(task:&VisionTask,answer:&str)->bool {
    if task.contains {
        answer.to_lowercase().contains(&task.expected.to_lowercase())
    }else{
        answer.trim().replace("\r\n","\n")==task.expected.trim().replace("\r\n","\n")
    }
}
fn vision_score(plan:&QuantParityPlan,model:&Path,log:&mut impl FnMut(&str))->io::Result<Value>{
    let vision=plan.vision.as_ref().ok_or_else(||invalid("vision suite missing"))?;
    let mm=plan.options.mmproj.as_ref().ok_or_else(||invalid("mmproj missing"))?;
    let mut passed=0usize;
    let mut details=Vec::new();
    for task in &vision.tasks {
        log(&format!("[ojon] vision test {} using {}",task.id,model.display()));
        let mut cmd=Command::new(&plan.options.eval_tool);
        cmd.arg("-m").arg(model)
            .arg("--mmproj").arg(mm)
            .arg("--image").arg(&task.image)
            .arg("-p").arg(&task.prompt)
            .arg("-n").arg(plan.options.tokens.to_string())
            .arg("-ngl").arg(plan.options.gpu_layers.to_string())
            .arg("--temp").arg("0").arg("-s").arg("42")
            .arg("--no-display-prompt").arg("--simple-io")
            .arg("--offline").stdin(Stdio::null());
        if let Some(d)=&plan.options.device{cmd.arg("-dev").arg(d);}
        let start=Instant::now();
        let output=cmd.output()?;
        if !output.status.success(){
            return Err(io::Error::other(format!(
                "vision test {} failed: {}",task.id,
                String::from_utf8_lossy(&output.stderr).chars().take(300).collect::<String>())));
        }
        if output.stdout.len()>2*1024*1024||output.stderr.len()>2*1024*1024{
            return Err(invalid("vision model output exceeded 2 MiB"));
        }
        let answer=String::from_utf8(output.stdout)
            .map_err(|_|invalid("vision stdout is not UTF-8"))?;
        let pass=matches_vision(task,&answer);
        if pass{passed+=1;}
        details.push(json!({"id":task.id,"pass":pass,
            "elapsed_seconds":start.elapsed().as_secs_f64(),
            "answer":answer.chars().take(8192).collect::<String>()}));
    }
    Ok(json!({"passed":passed,"total":vision.tasks.len(),
        "accuracy_pct":100.0*passed as f64/vision.tasks.len() as f64,"tests":details}))
}
fn task_parity(report:&Value,max_loss:f64)->(bool,Value){
    let base=&report["baseline"]["metrics"];
    let current=&report["candidates"][0]["metrics"];
    let mut pass=true;
    let mut out=Vec::new();
    for key in ["coding_accuracy","reasoning_accuracy","tool_call_accuracy"]{
        let b=base[key].as_f64();
        let c=current[key].as_f64();
        let okay=matches!((b,c),(Some(b),Some(c))if b>0.0&&c+max_loss+1e-10>=b);
        pass&=okay;
        out.push(json!({"metric":key,"reference_pct":b,
            "candidate_pct":c,"max_loss_points":max_loss,"pass":okay}));
    }
    (pass,Value::Array(out))
}
pub fn run_quant_parity(plan:&QuantParityPlan,mut log:impl FnMut(&str))->io::Result<Value>{
    let timestamp=SystemTime::now().duration_since(UNIX_EPOCH)
        .map_err(|_|invalid("system clock invalid"))?.as_nanos();
    let prefix=format!("ojon-parity-{}-{timestamp}",std::process::id());
    let mut rows=Vec::new();
    let mut best:Option<(PathBuf,u64)>=None;
    let mut base_vision:Option<Value>=None;
    let mut baseline_ppl:Option<f64>=None;
    for(ordinal,precision)in plan.options.precisions.iter().enumerate(){
        let key=format!("{prefix}-{:02}-{}",ordinal,precision.as_str());
        let output=plan.output_dir.join(format!("{key}.gguf"));
        let evidence=plan.output_dir.join(format!("{key}.task-evidence.json"));
        log(&format!("[ojon] candidate {} from full-precision source (NOT from Q4 reference)",precision));
        let converted=(||{
            let mut quant=prepare_gguf(&plan.source,&output,&plan.options.quantizer,*precision,plan.options.threads)?;
            if let Some(path)=&plan.options.imatrix{quant=quant.with_imatrix(path)?;}
            if let Some(mib)=plan.options.max_buffer_mib{quant=quant.with_max_buffer_mib(mib)?;}
            quantize_gguf_with_log(&quant,|line|log(line))
        })();
        let done=match converted{
            Ok(done)=>done,
            Err(e)=>{
                rows.push(json!({"type":precision.as_str(),"status":"QUANTIZATION_FAILED",
                    "error":e.to_string()}));
                continue;
            }
        };
        let size=done.output_info.file_bytes;
        let savings=100.0*(1.0-size as f64/plan.baseline_bytes as f64);
        let size_pass=size<plan.baseline_bytes &&
            savings+1e-10>=plan.options.min_saving_percent;
        if !size_pass{
            rows.push(json!({"type":precision.as_str(),"status":"NOT_SMALL_ENOUGH",
                "bytes":size,"savings_pct":savings,"output":output.to_string_lossy(),
                "reference_bytes":plan.baseline_bytes}));
            continue;
        }
        let results=(||{
            let candidates=vec![(format!("smaller_{ordinal}"),output.clone())];
            let eval=prepare_eval(&plan.suite,&plan.baseline,&candidates,
                &plan.options.eval_tool,plan.options.gpu_layers,
                plan.options.device.as_deref(),plan.options.tokens,&evidence)?;
            run_eval(&eval,|line|log(line))
        })();
        let task_report=match results{
            Ok(r)=>r,
            Err(e)=>{
                rows.push(json!({"type":precision.as_str(),"status":"TASK_EVALUATION_FAILED",
                    "error":e.to_string(),"bytes":size,"output":output.to_string_lossy()}));
                continue;
            }
        };
        let(task_pass,task_metrics)=task_parity(&task_report,plan.options.max_loss_points);
        let mut vision_pass=true;
        let mut vision_evidence=Value::Null;
        if task_pass && plan.vision.is_some(){
            let test=(||{
                if base_vision.is_none(){
                    base_vision=Some(vision_score(plan,&plan.baseline,&mut log)?);
                }
                vision_score(plan,&output,&mut log)
            })();
            match test{
                Ok(candidate)=>{
                    let reference=base_vision.as_ref().unwrap();
                    let b=reference["accuracy_pct"].as_f64().unwrap_or(0.0);
                    let c=candidate["accuracy_pct"].as_f64().unwrap_or(-1.0);
                    vision_pass=b>0.0 && c+plan.options.max_loss_points+1e-10>=b;
                    vision_evidence=json!({"reference":reference,"candidate":candidate,
                        "pass":vision_pass});
                }
                Err(e)=>{
                    vision_pass=false;
                    vision_evidence=json!({"status":"VISION_EVALUATION_FAILED","error":e.to_string()});
                }
            }
        }
        let mut ppl_pass=true;
        let mut ppl_evidence=Value::Null;
        if task_pass&&vision_pass {
            if let Some(corpus)=&plan.options.ppl_corpus {
                let settings=PplSettings{
                    tool:plan.options.ppl_tool.clone(),corpus:corpus.clone(),
                    gpu_layers:plan.options.gpu_layers.max(0) as u32,
                    context:plan.options.ppl_context,
                    max_relative_degradation_pct:plan.options.max_ppl_regression_pct,
                };
                let measured=(||{
                    if baseline_ppl.is_none(){
                        baseline_ppl=Some(measure_perplexity(&plan.baseline,&settings,&mut log)?.0);
                    }
                    measure_perplexity(&output,&settings,&mut log)
                })();
                match measured{
                    Ok((candidate,seconds))=>{
                        let reference=baseline_ppl.unwrap();
                        let regression_pct=100.0*(candidate/reference-1.0);
                        ppl_pass=regression_pct<=plan.options.max_ppl_regression_pct+1e-10;
                        ppl_evidence=json!({"baseline_perplexity":reference,
                            "candidate_perplexity":candidate,
                            "relative_regression_pct":regression_pct,
                            "elapsed_seconds":seconds,"pass":ppl_pass});
                    }
                    Err(e)=>{
                        ppl_pass=false;
                        ppl_evidence=json!({"status":"PPL_EVALUATION_FAILED","error":e.to_string()});
                    }
                }
            }
        }
        let measured_pass=task_pass&&vision_pass&&ppl_pass;
        let scope=if plan.vision.is_some(){
            "TEXT_AND_VISION_TASK_SUITE_ONLY"
        }else{"TEXT_TASK_SUITE_ONLY_VISION_NOT_VERIFIED"};
        if measured_pass&&best.as_ref().is_none_or(|(_,n)|size<*n){
            best=Some((output.clone(),size));
        }
        rows.push(json!({"type":precision.as_str(),
            "status":if measured_pass {"NO_DETECTED_TASK_REGRESSION"}else{"CAPABILITY_REGRESSION"},
            "output":output.to_string_lossy(),"bytes":size,
            "savings_pct":savings,"reference_bytes":plan.baseline_bytes,
            "text_tasks":task_metrics,"vision":vision_evidence,
            "perplexity":ppl_evidence,
            "evidence_scope":scope,"task_report":evidence.to_string_lossy(),
            "candidate_verified_lossless":false}));
    }
    let report=json!({
        "schema":"ojon.quant_parity_result.v1",
        "baseline":plan.baseline.to_string_lossy(),
        "high_precision_source":plan.source.to_string_lossy(),
        "reference_bytes":plan.baseline_bytes,
        "heldout_suite":plan.suite.to_string_lossy(),
        "imatrix":plan.options.imatrix.as_ref().map(|p|p.to_string_lossy().to_string()),
        "ppl_corpus":plan.options.ppl_corpus.as_ref().map(|p|p.to_string_lossy().to_string()),
        "baseline_perplexity":baseline_ppl,
        "vision_suite_count":plan.vision.as_ref().map(|x|x.tasks.len()),
        "results":rows,
        "provisional_smallest_measured_candidate":best.as_ref()
            .map(|(p,_)|p.to_string_lossy().to_string()),
        "provisional_smallest_bytes":best.as_ref().map(|(_,size)|*size),
        "no_quality_loss_everywhere_proven":false,
        "hardware_runtime_fit_proven":false,
        "warning":"Pass means no detected regression on these specific tasks ONLY. Check perplexity, diverse prompts, real VLM use, coding agents and performance before accepting model."
    });
    let result_file=plan.output_dir.join(format!("{prefix}.report.json"));
    let mut f=OpenOptions::new().write(true).create_new(true).open(&result_file)?;
    f.write_all(serde_json::to_string_pretty(&report).map_err(io::Error::other)?.as_bytes())?;
    f.sync_all()?;
    Ok(report)
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn numerical_parity_checks_baseline_not_full_precision(){
        let identical=json!({"baseline":{"metrics":{"coding_accuracy":90.0,
            "reasoning_accuracy":85.0,"tool_call_accuracy":80.0}},
            "candidates":[{"metrics":{"coding_accuracy":90.0,
                "reasoning_accuracy":85.0,"tool_call_accuracy":80.0}}]});
        assert!(task_parity(&identical,0.0).0);
        let losing=json!({"baseline":{"metrics":{"coding_accuracy":90.0,
            "reasoning_accuracy":85.0,"tool_call_accuracy":80.0}},
            "candidates":[{"metrics":{"coding_accuracy":89.0,
                "reasoning_accuracy":85.0,"tool_call_accuracy":80.0}}]});
        assert!(!task_parity(&losing,0.0).0);
        assert!(task_parity(&losing,1.0).0);
    }
    #[test]
    fn vision_exact_and_contains_are_separate(){
        let t=VisionTask{id:"a".into(),image:PathBuf::new(),
            prompt:"test".into(),expected:"BLUE".into(),contains:true};
        assert!(matches_vision(&t,"the item is blue"));
        let e=VisionTask{contains:false,..t};
        assert!(!matches_vision(&e,"the item is blue"));
    }
}
