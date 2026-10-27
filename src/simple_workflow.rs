//! One Start action for an original checkpoint: optional directional edit,
//! GGUF quantization, and a real loader smoke check. Quality is a separate gate.

use crate::cvector_import::recipe_from_control_vector;
use crate::discovery::hf_python_path;
use crate::gguf::{inspect_gguf, inspect_gguf_tensors};
use crate::gguf_edit::{prepare_gguf_abliteration_value,run_gguf_abliteration};
use crate::hf::{inspect_hf_folder,prepare_hf,quantize_hf_with_log};
use crate::jobs::run_with_log;
use crate::llama::{prepare_gguf,quantize_gguf_with_log,Precision};
use crate::smoke::smoke_gguf_capture;
use serde_json::{json,Value};
use std::fs::{self,File,OpenOptions};
use std::io::{self,BufRead,BufReader,ErrorKind,Write};
use std::path::{Path,PathBuf};
use std::process::Command;
use std::time::{SystemTime,UNIX_EPOCH};

fn invalid(s:impl Into<String>)->io::Error{io::Error::new(ErrorKind::InvalidInput,s.into())}

#[derive(Clone,Debug)]
pub struct SimpleRequest {
    pub source:PathBuf,
    pub output_dir:PathBuf,
    pub precision:Precision,
    pub quantizer:PathBuf,
    pub runtime:PathBuf,
    pub converter:PathBuf,
    pub python:PathBuf,
    pub gpu_layers:u32,
    /// Directory containing positive.txt and negative.txt, in paired line order.
    pub refusal_examples:Option<PathBuf>,
    pub cvector_generator:Option<PathBuf>,
    pub edit_strength:f64,
    pub edit_block:Option<usize>,
    /// Optional ojon.refusal_suite.v1: refusal rate is really measured.
    pub refusal_suite:Option<PathBuf>,
    /// Optional ojon.task_suite.v1: functional capability really measured.
    pub task_suite:Option<PathBuf>,
}
impl SimpleRequest {
    pub fn new(source:PathBuf,output_dir:PathBuf)->Self{
        Self{source,output_dir,precision:Precision::Q5KM,
            quantizer:PathBuf::from("llama-quantize"),runtime:PathBuf::from("llama-cli"),
            converter:PathBuf::from("convert_hf_to_gguf.py"),python:hf_python_path(),
            gpu_layers:0,refusal_examples:None,cvector_generator:None,
            edit_strength:0.1,edit_block:None,refusal_suite:None,task_suite:None}
    }
}

#[derive(Clone,Debug)]
enum Source { Gguf(PathBuf), Hf(PathBuf) }
pub struct SimplePlan { request:SimpleRequest, source:Source, output:PathBuf,
    positive:Option<PathBuf>, negative:Option<PathBuf> }

fn resolve_source(path:&Path)->io::Result<Source>{
    let path=path.canonicalize()?;
    if path.is_file(){
        let info=inspect_gguf(&path)?;
        if !matches!(info.file_type,Some(0|1|32))||info.split_count.unwrap_or(1)>1{
            return Err(invalid("choose an original single-file F16/BF16/F32 GGUF"));
        }
        return Ok(Source::Gguf(path));
    }
    if !path.is_dir(){return Err(invalid("model source must be a GGUF file or model folder"));}
    let mut originals=Vec::new();
    for (index,entry) in fs::read_dir(&path)?.enumerate(){
        if index>=2048{return Err(invalid("model folder has too many entries"));}
        let entry=entry?;
        if entry.file_type()?.is_file() && entry.path().extension().and_then(|s|s.to_str())
            .is_some_and(|s|s.eq_ignore_ascii_case("gguf")){
            let info=inspect_gguf(&entry.path())?;
            if matches!(info.file_type,Some(0|1|32))&&info.split_count.unwrap_or(1)==1{
                originals.push(entry.path());
            }
        }
    }
    match originals.len(){
        1=>Ok(Source::Gguf(originals.remove(0))),
        n if n>1=>Err(invalid("folder has multiple original GGUFs; select the exact file")),
        _=>{inspect_hf_folder(&path)?;Ok(Source::Hf(path))}
    }
}

fn paired_file(path:&Path)->io::Result<usize>{
    let metadata=fs::metadata(path)?;
    if !metadata.is_file()||metadata.len()==0||metadata.len()>2*1024*1024{
        return Err(invalid("paired prompt files must be 1 byte..2 MiB each"));
    }
    let mut count=0;
    for row in BufReader::new(File::open(path)?).lines(){
        let row=row?;
        if row.trim().is_empty()||row.len()>8192{return Err(invalid("prompt lines must be nonblank and <=8192 bytes"));}
        count+=1;
        if count>256{return Err(invalid("paired prompt files may have at most 256 lines"));}
    }
    if count<2{return Err(invalid("need at least two paired prompt lines"));}
    Ok(count)
}

fn recommended_block(source:&Path)->io::Result<usize>{
    let mut blocks=inspect_gguf_tensors(source)?.into_iter().filter_map(|tensor|{
        let layer=tensor.name.strip_prefix("blk.")?.strip_suffix(".attn_output.weight")?;
        (tensor.shape.len()==2).then(||layer.parse::<usize>().ok()).flatten()
    }).collect::<Vec<_>>();
    blocks.sort_unstable();blocks.dedup();
    if blocks.len()<3{return Err(invalid("no supported middle-layer attention output matrix"));}
    Ok(blocks[(blocks.len()*2/3).min(blocks.len()-2)])
}

pub fn prepare_simple_run(request:SimpleRequest)->io::Result<SimplePlan>{
    if !matches!(request.precision,Precision::Q4KM|Precision::Q5KM){
        return Err(invalid("guided run supports Q4_K_M or Q5_K_M"));
    }
    if request.gpu_layers>512{return Err(invalid("GPU layers must be 0..512"));}
    let source=resolve_source(&request.source)?;
    let output_dir=request.output_dir.canonicalize()?;
    if !output_dir.is_dir(){return Err(invalid("output folder must exist"));}
    if let Source::Hf(path)=&source {
        if output_dir.starts_with(path){return Err(invalid("output folder must be outside HF source"));}
    }
    let stem=match &source{Source::Gguf(p)=>p.file_stem(),Source::Hf(p)=>p.file_name()}
        .ok_or_else(||invalid("source name missing"))?.to_string_lossy();
    let edit=request.refusal_examples.is_some();
    if let Some(suite)=&request.refusal_suite{
        crate::refusal::parse_refusal_suite(suite)?;
    }
    if let Some(suite)=&request.task_suite{
        crate::task_eval::parse_eval_suite(suite)?;
    }
    let suffix=if edit{"-refusal-edited"}else{""};
    let mut output=None;
    for index in 0..1000{
        let tail=if index==0{String::new()}else{format!("-{index:03}")};
        let path=output_dir.join(format!("{stem}{suffix}-{}{tail}.gguf",request.precision.as_str()));
        if path.symlink_metadata().is_err(){output=Some(path);break;}
    }
    let output=output.ok_or_else(||invalid("no free output filename after 1000 runs"))?;
    let (positive,negative)=if let Some(folder)=&request.refusal_examples{
        if !request.edit_strength.is_finite()||request.edit_strength<=0.0||request.edit_strength>1.0{
            return Err(invalid("edit strength must be >0 and <=1"));
        }
        if request.cvector_generator.as_ref().is_none_or(|p|!p.is_file()){
            return Err(invalid("refusal editing requires an installed llama-cvector-generator executable"));
        }
        let folder=folder.canonicalize()?;
        let positive=folder.join("positive.txt");let negative=folder.join("negative.txt");
        if paired_file(&positive)?!=paired_file(&negative)?{
            return Err(invalid("positive.txt and negative.txt need the same number of lines"));
        }
        (Some(positive),Some(negative))
    }else{(None,None)};
    Ok(SimplePlan{request,source,output,positive,negative})
}

impl SimplePlan {
    pub fn as_json(&self)->Value{
        json!({"schema":"ojon.simple_plan.v1","source":match &self.source{
            Source::Gguf(p)|Source::Hf(p)=>p.to_string_lossy()},
            "source_kind":match &self.source{Source::Gguf(_)=>"original_gguf",Source::Hf(_)=>"hf_checkpoint"},
            "output":self.output.to_string_lossy(),"precision":self.request.precision.as_str(),
            "refusal_edit_requested":self.positive.is_some(),
            "refusal_suite":self.request.refusal_suite.as_ref().map(|p|p.to_string_lossy().to_string()),
            "task_suite":self.request.task_suite.as_ref().map(|p|p.to_string_lossy().to_string()),
            "refusal_effect":"NOT_EVALUATED","quality_retention":"NOT_EVALUATED",
            "runtime_load":"NOT_TESTED","status":"PLAN_ONLY"})
    }
}

struct Stage(PathBuf);
impl Drop for Stage {fn drop(&mut self){let _=fs::remove_dir_all(&self.0);}}
fn stage(parent:&Path)->io::Result<Stage>{
    for retry in 0..16{
        let stamp=SystemTime::now().duration_since(UNIX_EPOCH)
            .map_err(|_|invalid("invalid clock"))?.as_nanos();
        let path=parent.join(format!(".ojon-simple-{}-{stamp}-{retry}",std::process::id()));
        match fs::create_dir(&path){
            Ok(())=>return Ok(Stage(path)),
            Err(e) if e.kind()==ErrorKind::AlreadyExists=>continue,
            Err(e)=>return Err(e),
        }
    }
    Err(invalid("cannot create run staging folder"))
}

const REFUSAL_TOKENS:u32=96;
const TASK_TOKENS:u32=128;

fn report_path(output:&Path,tag:&str)->PathBuf{
    let mut name=output.as_os_str().to_os_string();name.push(tag);
    PathBuf::from(name)
}

pub fn run_simple(plan:&SimplePlan,mut log:impl FnMut(&str))->io::Result<Value>{
    if plan.output.symlink_metadata().is_ok(){return Err(io::Error::new(ErrorKind::AlreadyExists,"output appeared after planning"));}
    let parent=plan.output.parent().ok_or_else(||invalid("output folder missing"))?;
    let stage=stage(parent)?;
    let threads=std::thread::available_parallelism().map_or(4,|n|n.get());
    let mut source=match &plan.source{
        Source::Gguf(path)=>path.clone(),
        Source::Hf(path)=>{
            if plan.positive.is_none(){
                log("Converting HF checkpoint and quantizing to GGUF");
                let hf=prepare_hf(path,&plan.output,&plan.request.converter,&plan.request.python,
                    &plan.request.quantizer,plan.request.precision,threads)?;
                quantize_hf_with_log(&hf,|line|log(line))?;
                return finish(plan,"not_requested",&mut log,None);
            }
            let intermediate=stage.0.join("original.gguf");
            let info=inspect_hf_folder(path)?;
            crate::resource::assess_resources(info.source_weight_bytes,parent,
                crate::resource::ResourceMode::Gguf,"cpu")?.enforce(false)?;
            let mut command=Command::new(&plan.request.python);
            command.arg(&plan.request.converter).arg(path).arg("--outfile")
                .arg(&intermediate).arg("--outtype").arg("auto");
            log("Converting HF checkpoint to original GGUF for refusal edit");
            run_with_log(&mut command,|line|log(line))?;
            let converted=inspect_gguf(&intermediate)?;
            if !matches!(converted.file_type,Some(0|1|32))||converted.tensor_count==0{
                return Err(invalid("HF converter did not make an original float GGUF"));
            }
            intermediate
        }
    };
    // Baseline for measurement: the untouched original float GGUF (before any
    // directional edit and before quantization).
    let baseline=source.clone();
    if let (Some(positive),Some(negative))=(&plan.positive,&plan.negative){
        let tool=plan.request.cvector_generator.as_ref().ok_or_else(||invalid("cvector tool missing"))?;
        let vector=stage.0.join("control-vector.gguf");
        let mut command=Command::new(tool);
        command.arg("-m").arg(&source).arg("-ngl").arg(plan.request.gpu_layers.to_string())
            .arg("--positive-file").arg(positive).arg("--negative-file").arg(negative)
            .arg("--method").arg("mean").arg("-o").arg(&vector);
        log("Capturing paired model directions from GGUF");
        run_with_log(&mut command,|line|log(line))?;
        let block=match plan.request.edit_block{Some(n)=>n,None=>recommended_block(&source)?};
        let recipe=recipe_from_control_vector(&vector,block,plan.request.edit_strength)?;
        let edited=stage.0.join("edited-original.gguf");
        let edit=prepare_gguf_abliteration_value(&source,&recipe,&edited)?;
        log(&format!("Editing attention output at block {block}"));
        run_gguf_abliteration(&edit,|line|log(line))?;
        source=edited;
    }
    let quant=prepare_gguf(&source,&plan.output,&plan.request.quantizer,plan.request.precision,threads)?;
    log(&format!("Quantizing edited or original weights to {}",plan.request.precision.as_str()));
    quantize_gguf_with_log(&quant,|line|log(line))?;
    finish(plan,if plan.positive.is_some(){"edited"}else{"not_requested"},&mut log,Some(&baseline))
}

fn finish(plan:&SimplePlan,edit_status:&str,log:&mut impl FnMut(&str),
    baseline:Option<&Path>)->io::Result<Value>{
    let info=inspect_gguf(&plan.output)?;
    log("Loading final GGUF with the configured inference backend");
    let smoke=smoke_gguf_capture(&plan.output,&plan.request.runtime,
        "Reply with one short sentence.",16,plan.request.gpu_layers);
    let (load_status,load_error)=match smoke{
        Ok(_)=>("LOADED_AND_GENERATED",None),
        Err(e)=>("LOAD_FAILED",Some(e.to_string()))
    };
    // Optional real refusal measurement: before (original) and after (output).
    let mut refusal_summary=json!("NOT_EVALUATED");
    let mut task_summary=json!("NOT_EVALUATED");
    if let Some(suite)=&plan.request.refusal_suite{
        let baseline=baseline.ok_or_else(||invalid("refusal measurement needs a GGUF baseline"))?;
        let gpu=plan.request.gpu_layers.min(512) as i32;
        let before_plan=crate::refusal::prepare_refusal_eval_with_server(baseline,suite,
            &plan.request.runtime,None,gpu,None,REFUSAL_TOKENS,
            &report_path(&plan.output,".ojon-refusal-before.json"))?;
        log("Measuring refusal behaviour of the original model");
        let before=crate::refusal::run_refusal_eval(&before_plan,&mut *log)?;
        let after_plan=crate::refusal::prepare_refusal_eval_with_server(&plan.output,suite,
            &plan.request.runtime,None,gpu,None,REFUSAL_TOKENS,
            &report_path(&plan.output,".ojon-refusal-after.json"))?;
        log("Measuring refusal behaviour of the produced model");
        let after=crate::refusal::run_refusal_eval(&after_plan,&mut *log)?;
        let before_rate=crate::refusal::refusal_rate_from_report(&before)?;
        let after_rate=crate::refusal::refusal_rate_from_report(&after)?;
        refusal_summary=json!({"measurement":"MEASURED",
            "refusal_rate_before":before_rate,"refusal_rate_after":after_rate,
            "reduction":before_rate-after_rate,
            "before_receipt":before,"after_receipt":after});
    }
    if let Some(suite)=&plan.request.task_suite{
        let baseline=baseline.ok_or_else(||invalid("capability measurement needs a GGUF baseline"))?;
        let report=crate::task_eval::prepare_eval(suite,baseline,
            &[("produced".to_owned(),plan.output.clone())],
            &plan.request.runtime,plan.request.gpu_layers.min(512) as i32,None,TASK_TOKENS,
            &report_path(&plan.output,".ojon-tasks.json"))?;
        log("Measuring held-out functional capability of the original and produced models");
        let eval=crate::task_eval::run_eval(&report,|line|log(line))?;
        task_summary=json!({"measurement":"MEASURED_FUNCTIONAL_SUITE",
            "suite":eval["evaluation_suite"],"baseline_metrics":eval["baseline"]["metrics"],
            "produced_metrics":eval["candidates"][0]["metrics"],"receipt":report_path(&plan.output,".ojon-tasks.json")});
    }
    let result=json!({"schema":"ojon.simple_result.v1","source":plan.as_json()["source"],
        "output":plan.output.to_string_lossy(),"bytes":info.file_bytes,
        "precision":plan.request.precision.as_str(),"refusal_edit":edit_status,
        "runtime_load":load_status,"load_error":load_error,
        "refusal_effect":refusal_summary,"quality_retention":task_summary,
        "status":if load_status=="LOADED_AND_GENERATED"{"MODEL_READY_FOR_QUALITY_REVIEW"}else{"MODEL_WRITTEN_LOAD_FAILED"}});
    let mut name=plan.output.as_os_str().to_os_string();name.push(".ojon-run.json");
    let mut receipt=OpenOptions::new().write(true).create_new(true).open(PathBuf::from(name))?;
    serde_json::to_writer_pretty(&mut receipt,&result)?;
    receipt.flush()?;receipt.sync_all()?;
    Ok(result)
}
