//! Real deterministic GGUF task comparison using either llama-cli or persistent local llama-server.
//! Strict text/JSON answers; generated code is never executed.
use crate::gguf::inspect_gguf;
use serde_json::{json, Value};
use std::collections::HashSet;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash,Hasher};
use std::time::UNIX_EPOCH;
use std::fs::{self, OpenOptions};
use std::io::{self, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Instant,Duration};

fn invalid(s: impl Into<String>) -> io::Error { io::Error::new(ErrorKind::InvalidInput,s.into()) }

#[derive(Clone,Debug)]
pub struct EvalTask { pub id:String, pub category:String, pub prompt:String, pub expected:String, pub matching:String }
#[derive(Clone,Debug)]
pub struct EvalSuite { pub id:String, pub tasks:Vec<EvalTask> }

pub fn parse_eval_suite(path:&Path)->io::Result<EvalSuite> {
    let size=fs::metadata(path)?.len();
    if size==0 || size>2*1024*1024 {return Err(invalid("task suite must be 1 byte..2 MiB"));}
    let raw:Value=serde_json::from_slice(&fs::read(path)?).map_err(io::Error::other)?;
    if raw["schema"]!="ojon.task_suite.v1" {return Err(invalid("suite schema must be ojon.task_suite.v1"));}
    let id=raw["id"].as_str().filter(|v|!v.is_empty()&&v.len()<=128)
        .ok_or_else(||invalid("missing suite id"))?.to_owned();
    let rows=raw["tasks"].as_array().filter(|v|(6..=256).contains(&v.len()))
        .ok_or_else(||invalid("require 6..=256 tasks"))?;
    let mut ids=HashSet::new(); let mut counts=[0usize;3]; let mut tasks=Vec::new();
    for row in rows {
        let id=row["id"].as_str().filter(|v|!v.is_empty()&&v.len()<=128)
            .ok_or_else(||invalid("invalid task id"))?.to_owned();
        if !ids.insert(id.clone()) {return Err(invalid("duplicate task id"));}
        let category=row["category"].as_str().unwrap_or("");
        let category_idx=match category {"coding"=>0,"reasoning"=>1,"tool_call"=>2,
            _=>return Err(invalid("category must be coding, reasoning or tool_call"))};
        counts[category_idx]+=1;
        let matching=row["match"].as_str().unwrap_or("exact");
        if !matches!(matching,"exact"|"json") {return Err(invalid("match must be exact or json"));}
        let prompt=row["prompt"].as_str().filter(|v|!v.is_empty()&&v.len()<=16384)
            .ok_or_else(||invalid("prompt must be <=16384 characters"))?;
        let expected=row["expected"].as_str().filter(|v|!v.is_empty()&&v.len()<=16384)
            .ok_or_else(||invalid("expected must be <=16384 characters"))?;
        if matching=="json" {serde_json::from_str::<Value>(expected)
            .map_err(|_|invalid("expected JSON is invalid"))?;}
        tasks.push(EvalTask{id,category:category.into(),prompt:prompt.into(),
            expected:expected.into(),matching:matching.into()});
    }
    if counts.iter().any(|n|*n<2) {return Err(invalid("minimum two tasks per category"));}
    Ok(EvalSuite{id,tasks})
}

fn answer_matches(task:&EvalTask,answer:&str)->bool {
    let text=answer.trim();
    if task.matching=="json" {
        const FENCE: &str = "\u{60}\u{60}\u{60}";
        let candidate = if text.starts_with(FENCE) && text.ends_with(FENCE) {
            text.split_once('\n')
                .and_then(|(_, s)| s.rsplit_once(FENCE).map(|v| v.0))
                .unwrap_or("")
        } else { text };
        match (serde_json::from_str::<Value>(candidate.trim()),
               serde_json::from_str::<Value>(&task.expected)) {
            (Ok(a),Ok(b))=>a==b,_=>false,
        }
    }else{
        text.replace("\r\n","\n")==task.expected.trim().replace("\r\n","\n")
    }
}

#[derive(Clone,Debug)]
pub struct EvalPlan {
    pub suite:EvalSuite,pub baseline:PathBuf,pub candidates:Vec<(String,PathBuf)>,
    pub tool:PathBuf,pub gpu_layers:i32,pub device:Option<String>,
    pub tokens:u32,pub report:PathBuf,
    /// When present, keep each model loaded for all suite prompts.
    pub server_tool:Option<PathBuf>
}
fn checked_model(path:&Path)->io::Result<PathBuf> {
    let path=path.canonicalize()?;let info=inspect_gguf(&path)?;
    if info.tensor_count==0 || info.split_count.unwrap_or(1)>1 {
        return Err(invalid("task evaluation supports nonempty single-file GGUF only"));
    }
    Ok(path)
}
pub fn prepare_eval(
    suite:&Path,baseline:&Path,candidates:&[(String,PathBuf)],tool:&Path,
    gpu_layers:i32,device:Option<&str>,tokens:u32,report:&Path
)->io::Result<EvalPlan>{
    prepare_eval_with_server(suite,baseline,candidates,tool,None,
        gpu_layers,device,tokens,report)
}
pub fn prepare_eval_with_server(
    suite:&Path,baseline:&Path,candidates:&[(String,PathBuf)],tool:&Path,
    server_tool:Option<&Path>,gpu_layers:i32,device:Option<&str>,tokens:u32,report:&Path
)->io::Result<EvalPlan>{
    let suite=parse_eval_suite(suite)?;
    if candidates.is_empty()||candidates.len()>12{return Err(invalid("require 1..=12 candidates"));}
    if !(1..=1024).contains(&tokens)||!(-1..=512).contains(&gpu_layers){
        return Err(invalid("tokens or GPU layer count out of range"));
    }
    if tool.as_os_str().is_empty(){return Err(invalid("llama-cli required"));}
    if server_tool.is_some_and(|p|p.as_os_str().is_empty()){
        return Err(invalid("llama-server executable cannot be empty"));
    }
    if let Some(d)=device{
        if d.is_empty()||d.len()>80||!d.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'_'||b==b'-'||b==b'/'){
            return Err(invalid("invalid device"));
        }
    }
    let baseline=checked_model(baseline)?;
    let baseline_arch=inspect_gguf(&baseline)?.architecture;
    let mut names=HashSet::new();let mut paths=HashSet::new();paths.insert(baseline.clone());
    let mut verified=Vec::new();
    for (name,path) in candidates {
        if name.is_empty()||name.len()>80||!name.bytes().all(|b|b.is_ascii_alphanumeric()||b==b'-'||b==b'_')
            ||!names.insert(name.clone()){return Err(invalid("duplicate/invalid candidate label"));}
        let path=checked_model(path)?;
        if !paths.insert(path.clone()){return Err(invalid("candidate paths must differ"));}
        if inspect_gguf(&path)?.architecture!=baseline_arch {
            return Err(invalid("candidate architecture differs from baseline"));
        }
        verified.push((name.clone(),path));
    }
    if report.extension().and_then(|e|e.to_str())!=Some("json"){return Err(invalid("report must end in .json"));}
    let parent=report.parent().filter(|p|!p.as_os_str().is_empty()).unwrap_or(Path::new(".")).canonicalize()?;
    let report=parent.join(report.file_name().ok_or_else(||invalid("missing filename"))?);
    if report.symlink_metadata().is_ok(){return Err(io::Error::new(ErrorKind::AlreadyExists,"report exists"));}
    Ok(EvalPlan{suite,baseline,candidates:verified,tool:tool.into(),
        gpu_layers,device:device.map(str::to_owned),tokens,report,
        server_tool:server_tool.map(Path::to_path_buf)})
}
impl EvalPlan {
    pub fn as_json(&self)->Value{
        json!({"schema":"ojon.task_eval_plan.v1","suite":self.suite.id,"tasks":self.suite.tasks.len(),
          "baseline":self.baseline.to_string_lossy(),
          "candidates":self.candidates.iter().map(|(name,path)|json!({"name":name,"path":path.to_string_lossy()})).collect::<Vec<_>>(),
          "tool":self.tool.to_string_lossy(),
          "evaluation_mode":if self.server_tool.is_some(){"persistent_llama_server"}else{"llama_cli_per_task"},
          "server_tool":self.server_tool.as_ref().map(|p|p.to_string_lossy().to_string()),
          "gpu_layers":self.gpu_layers,
          "device":self.device,"tokens":self.tokens,"output":self.report.to_string_lossy(),
          "status":"PLAN_ONLY","quality_scope":"deterministic exact/json functional tests, NOT general intelligence"})
    }
}
fn execute_task(plan:&EvalPlan,model:&Path,task:&EvalTask)->io::Result<(String,f64)>{
    let mut command=Command::new(&plan.tool);
    command.arg("-m").arg(model).arg("-p").arg(&task.prompt)
        .arg("-n").arg(plan.tokens.to_string())
        .arg("-ngl").arg(plan.gpu_layers.to_string())
        .arg("--temp").arg("0").arg("-s").arg("42")
        .arg("--no-display-prompt").arg("-st").arg("--simple-io");
    if let Some(d)=&plan.device {command.arg("-dev").arg(d);}
    let start=Instant::now();
    let result=crate::jobs::capture_with_timeout(&mut command,2*1024*1024,Duration::from_secs(1200))?;
    if !result.status.success(){
        return Err(io::Error::other(format!("llama-cli task {} failed ({}): {}",
            task.id,result.status,String::from_utf8_lossy(&result.stderr).chars().take(500).collect::<String>())));
    }
    if result.stdout.len()>2*1024*1024||result.stderr.len()>2*1024*1024 {
        return Err(invalid("model stdout/stderr exceeded 2MiB"));
    }
    let answer=String::from_utf8(result.stdout).map_err(|_|invalid("response not utf8"))?;
    Ok((answer,start.elapsed().as_secs_f64()))
}
fn metric_key(category:&str)->&'static str{
    match category {"coding"=>"coding_accuracy","reasoning"=>"reasoning_accuracy",
        _=>"tool_call_accuracy"}
}
fn evaluate_one(plan:&EvalPlan,model:&Path,log:&mut impl FnMut(&str))->io::Result<Value>{
    let mut counts=[0usize;3];let mut passed=[0usize;3];let mut detail=Vec::new();
    // One runtime process per model, rather than one expensive model reload
    // for every task. Baseline results are still reused across candidate calls.
    let mut session=match plan.server_tool.as_deref(){
        Some(server)=>Some(crate::eval_session::EvalSession::launch(
            server,model,plan.gpu_layers,plan.device.as_deref())?),
        None=>None,
    };
    for task in &plan.suite.tasks {
        log(&format!("[ojon] {} — {}",model.display(),task.id));
        let (answer,seconds)=if let Some(session)=session.as_mut(){
            let started=Instant::now();
            let answer=session.completion(&task.prompt,plan.tokens)?;
            (answer,started.elapsed().as_secs_f64())
        }else{execute_task(plan,model,task)?};
        let correct=answer_matches(task,&answer);
        let idx=match task.category.as_str(){"coding"=>0,"reasoning"=>1,_=>2};
        counts[idx]+=1;if correct{passed[idx]+=1;}
        detail.push(json!({"id":task.id,"category":task.category,"pass":correct,
            "elapsed_seconds":seconds,"answer":answer.chars().take(8192).collect::<String>()}));
    }
    let mut metrics=serde_json::Map::new();
    for (idx,cat) in ["coding","reasoning","tool_call"].iter().enumerate(){
        metrics.insert(metric_key(cat).into(),json!(100.0*passed[idx] as f64/counts[idx] as f64));
    }
    Ok(json!({"metrics":metrics,"results":detail,"correct":passed,"total":counts}))
}
pub fn run_eval(plan:&EvalPlan,log:impl FnMut(&str))->io::Result<Value>{
    run_eval_reusing_baseline(plan,None,log)
}
/// Only reuse evidence within this invocation, after checking the canonical
/// baseline model and all task generation parameters. Never trust a disk receipt.
pub(crate) fn run_eval_reusing_baseline(
    plan:&EvalPlan,prior:Option<&Value>,mut log:impl FnMut(&str)
)->io::Result<Value>{
    let mut fingerprint=DefaultHasher::new();
    for task in &plan.suite.tasks{
        task.id.hash(&mut fingerprint);task.category.hash(&mut fingerprint);
        task.prompt.hash(&mut fingerprint);task.expected.hash(&mut fingerprint);
        task.matching.hash(&mut fingerprint);
    }
    // Same-run invalidation, not cryptographic model provenance.
    let meta=fs::metadata(&plan.baseline)?;
    let modified=meta.modified().ok()
        .and_then(|t|t.duration_since(UNIX_EPOCH).ok())
        .map(|d|d.as_nanos().to_string());
    let config=json!({"tool":plan.tool.to_string_lossy(),
        "server_tool":plan.server_tool.as_ref().map(|p|p.to_string_lossy().to_string()),
        "evaluation_mode":if plan.server_tool.is_some(){"persistent_llama_server"}else{"llama_cli_per_task"},
        "gpu_layers":plan.gpu_layers,"device":plan.device,
        "tokens":plan.tokens,"task_count":plan.suite.tasks.len(),
        "suite_fingerprint_noncrypto":fingerprint.finish(),
        "baseline_file_bytes":meta.len(),"baseline_modified_ns":modified});
    let baseline=if let Some(cached)=prior{
        if cached["baseline"]["model"]!=json!(plan.baseline.to_string_lossy())
            ||cached["evaluation_suite"]!=json!(plan.suite.id)
            ||cached["evaluation_config"]!=config
            ||!cached["baseline"]["tests"]["metrics"].is_object()
        {return Err(invalid("same-run baseline cache is not compatible"));}
        log("[ojon] Reusing matched baseline tests; no repeated baseline model launches");
        cached["baseline"]["tests"].clone()
    }else{evaluate_one(plan,&plan.baseline,&mut log)?};
    let mut candidates=Vec::new();let mut details=Vec::new();
    for (name,path) in &plan.candidates{
        let tests=evaluate_one(plan,path,&mut log)?;
        candidates.push(json!({"label":name,"bytes":fs::metadata(path)?.len(),
            "runtime_load_verified":true,"evaluation_verified":true,
            "metrics":tests["metrics"],"runtime_measurements":null}));
        details.push(json!({"label":name,"model":path.to_string_lossy(),"tests":tests}));
    }
    let requirements=["coding_accuracy","reasoning_accuracy","tool_call_accuracy"].iter()
        .map(|name|json!({"name":name,"direction":"higher","max_relative_degradation_pct":1.0}))
        .collect::<Vec<_>>();
    let result=json!({
        "schema":"ojon.quality_evidence.v1","evaluation_suite":plan.suite.id,
        "evaluation_config":config,"baseline_reused":prior.is_some(),
        "model_load_policy":if plan.server_tool.is_some(){"one_server_per_model_per_suite"}else{"cli_per_task"},
        "baseline":{"model":plan.baseline.to_string_lossy(),"metrics":baseline["metrics"],"tests":baseline},
        "requirements":requirements,
        "candidates":candidates,"detailed_candidate_results":details,
        "quality_scope":"strict functional tasks only; no statistical or full agentic parity claim",
        "hardware_fit_verified":false,"peak_vram_measured":false,"calibration_overlap_checked":false
    });
    let bytes=serde_json::to_vec_pretty(&result).map_err(io::Error::other)?;
    let mut file=OpenOptions::new().create_new(true).write(true).open(&plan.report)?;
    if let Err(e)=file.write_all(&bytes).and_then(|_|file.sync_all()){
        drop(file);let _=fs::remove_file(&plan.report);return Err(e);
    }
    Ok(result)
}
#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn compare_strict_responses(){
        let mut task=EvalTask{id:"t".into(),category:"tool_call".into(),
            prompt:"p".into(),expected:r#"{"name":"read_file","arguments":{"path":"src/main.rs"}}"#.into(),matching:"json".into()};
        assert!(answer_matches(&task,r#"{"arguments":{"path":"src/main.rs"},"name":"read_file"}"#));
        assert!(!answer_matches(&task,"read_file src/main.rs"));
        task.matching="exact".into();task.expected="42".into();
        assert!(answer_matches(&task,"42\r\n"));
        assert!(!answer_matches(&task,"The answer is 42"));
    }
    #[test]
    fn malformed_suite_rejected(){
        let f=std::env::temp_dir().join(format!("ojon-eval-{}.json",std::process::id()));
        fs::write(&f,b"{}").unwrap();assert!(parse_eval_suite(&f).is_err());
        let tasks=(0..6).map(|i|json!({"id":format!("{i}"),"category":
            if i<2 {"coding"}else if i<4 {"reasoning"}else {"tool_call"},
            "prompt":"Return 42","expected":"42","match":"exact"})).collect::<Vec<_>>();
        fs::write(&f,serde_json::to_vec(&json!({"schema":"ojon.task_suite.v1","id":"fixture","tasks":tasks})).unwrap()).unwrap();
        assert_eq!(parse_eval_suite(&f).unwrap().tasks.len(),6);
        fs::remove_file(f).unwrap();
    }
}
