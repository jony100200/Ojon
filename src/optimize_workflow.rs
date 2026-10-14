//! Metadata-only, UX/agent-ready next-step selector for Ojon optimization.
//!
//! A single readable plan bridges model inspection, a user's current quantized
//! reference, a held-out task suite, installed upstream backends and exact
//! existing CLI routes. This does NOT run quantizers, load weights, benchmark
//! models, allocate GPU memory, download dependencies or promise a winner.

use crate::discovery::{BackendId, FoundBackend,probe_quantizer_capabilities,probe_runtime_devices};
use crate::gguf::inspect_gguf;
use crate::llama::Precision;
use crate::quant_compare::run_existing_quant_comparison;
use crate::quant_parity::run_quant_parity;
use crate::probe::{probe_model, ModelCategory};
use crate::resource::{assess_resources,ResourceMode};
use crate::quant_compare::prepare_existing_quant_comparison;
use crate::quant_parity::{prepare_quant_parity, QuantParityOptions};
use crate::task_eval::parse_eval_suite;
use crate::smoke::smoke_gguf_capture_with_device;
use crate::benchmark::{prepare_bench,run_bench,compare_bench_files};
use crate::quality::{rank_quality_gated_candidates,select_workflow_goal_with_ratio};
use crate::adaptive::{prepare_adaptive_against_reference,run_adaptive};
use std::time::{SystemTime,UNIX_EPOCH};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::{self, ErrorKind};
use std::path::{Path,PathBuf};

fn invalid(s:impl Into<String>)->io::Error{io::Error::new(ErrorKind::InvalidInput,s.into())}

#[derive(Debug,Clone,Copy,PartialEq,Eq)]
pub enum OptimizeGoal {
    Smallest,
    Balanced,
    Fastest,
}
impl OptimizeGoal {
    pub fn parse(s:&str)->io::Result<Self>{
        match s{
            "smallest"=>Ok(Self::Smallest),
            "balanced"=>Ok(Self::Balanced),
            "fastest"=>Ok(Self::Fastest),
            _=>Err(invalid("goal must be smallest, balanced or fastest"))
        }
    }
    pub fn label(self)->&'static str{
        match self{Self::Smallest=>"smallest",Self::Balanced=>"balanced",Self::Fastest=>"fastest"}
    }
}
#[derive(Debug,Clone,PartialEq)]
pub struct OptimizeWorkflowRequest {
    pub model:PathBuf,
    pub current_baseline:Option<PathBuf>,
    pub candidates:Vec<(String,PathBuf)>,
    pub suite:Option<PathBuf>,
    pub output_dir:Option<PathBuf>,
    /// Optional calibration; used only when producing new quantizations.
    pub imatrix:Option<PathBuf>,
    pub tensor_sensitivity:Option<PathBuf>,
    /// Held-out corpus for a second independent quantitative quality gate.
    pub ppl_corpus:Option<PathBuf>,
    pub max_ppl_regression_pct:f64,
    pub goal:OptimizeGoal,
    pub device:String,
    /// Concrete llama.cpp device ID; never inferred from vendor name alone.
    pub runtime_device:Option<String>,
    pub gpu_layers:i32,
    pub min_saving_percent:f64,
    pub max_loss_points:f64,
    /// Optional user-selected order, restricted to the existing Q4 parity formats.
    pub precision_candidates:Option<Vec<Precision>>,
    pub bench_prompt_tokens:u32,
    pub bench_generation_tokens:u32,
    pub bench_repetitions:u32,
    pub balanced_min_speed_ratio:f64,
    /// Explicit persistent evaluator, if selected; None discovers installed server.
    pub server_tool:Option<PathBuf>,
}
impl OptimizeWorkflowRequest{
    pub fn new(model:PathBuf)->Self{
        Self{
            model,current_baseline:None,candidates:Vec::new(),suite:None,
            output_dir:None,imatrix:None,tensor_sensitivity:None,ppl_corpus:None,max_ppl_regression_pct:0.0,
            goal:OptimizeGoal::Balanced,device:"auto".into(),runtime_device:None,gpu_layers:0,
            min_saving_percent:5.0,max_loss_points:0.0,
            precision_candidates:None,bench_prompt_tokens:512,bench_generation_tokens:128,
            bench_repetitions:3,balanced_min_speed_ratio:0.85,server_tool:None
        }
    }
}
fn path_text(p:&Path)->String{p.to_string_lossy().into_owned()}
fn executable(backends:&[FoundBackend],id:BackendId)->Option<PathBuf>{
    backends.iter().find(|x|x.spec.id==id && x.installed())
        .and_then(|x|x.executable.clone())
}
fn stage(id:&str,title:&str,status:&str,reason:&str,command:Option<Vec<String>>)->Value{
    json!({"id":id,"title":title,"status":status,"reason":reason,
        "command_argv":command,"completed":false,"quality_verified":false})
}
/// Pick a free receipt filename without deleting old benchmark evidence.
/// The downstream report writer retains exclusive create-new semantics.
fn next_compare_report(folder:&Path)->io::Result<PathBuf>{
    for ordinal in 0..1000 {
        let name=if ordinal==0 {"ojon-quant-comparison.json".to_owned()}
            else {format!("ojon-quant-comparison-{ordinal:03}.json")};
        let p=folder.join(name);
        if !p.exists(){return Ok(p);}
    }
    Err(io::Error::new(ErrorKind::AlreadyExists,
        "1000 comparison receipts already exist; use another results directory"))
}

/// Resolve the one candidate list shared by planning, generation and receipts.
fn precisions_for(input:&OptimizeWorkflowRequest)->Vec<Precision>{
    input.precision_candidates.clone().unwrap_or_else(||match input.goal{
        OptimizeGoal::Smallest=>vec![Precision::IQ3M,Precision::Q3KM,Precision::Q3KL,Precision::IQ4XS],
        OptimizeGoal::Balanced=>vec![Precision::IQ4XS,Precision::Q3KL,Precision::IQ3M,Precision::Q3KM],
        OptimizeGoal::Fastest=>vec![Precision::Q3KM,Precision::Q3KL,Precision::IQ4XS,Precision::IQ3M],
    })
}
fn precision_csv(input:&OptimizeWorkflowRequest)->String{
    precisions_for(input).iter().map(|p|p.as_str()).collect::<Vec<_>>().join(",")
}
/// Resolved settings shared by planning and execution; no tool invocation here.
fn configured_options(input:&OptimizeWorkflowRequest,backends:&[FoundBackend])->QuantParityOptions {
    let mut options=QuantParityOptions::default();
    options.min_saving_percent=input.min_saving_percent;
    options.max_loss_points=input.max_loss_points;
    options.imatrix=input.imatrix.clone();
    options.ppl_corpus=input.ppl_corpus.clone();
    options.max_ppl_regression_pct=input.max_ppl_regression_pct;
    if let Some(bin)=executable(backends,BackendId::LlamaPerplexity){options.ppl_tool=bin;}
    options.gpu_layers=input.gpu_layers;
    options.device=input.runtime_device.clone();
    options.precisions=precisions_for(input);
    if let Some(bin)=executable(backends,BackendId::LlamaQuantize){options.quantizer=bin;}
    if let Some(bin)=executable(backends,BackendId::LlamaCli){options.eval_tool=bin;}
    options.server_tool=input.server_tool.clone().or_else(||executable(backends,BackendId::LlamaServer));
    options
}

/// A read-only front door for the existing pipeline. No shell command runs.
pub fn plan_optimize_workflow(
    input:&OptimizeWorkflowRequest,backends:&[FoundBackend],
)->io::Result<Value>{
    if !matches!(input.device.as_str(),"auto"|"cpu"|"cuda"|"xpu"){
        return Err(invalid("device must be auto, cpu, cuda or xpu"));
    }
    let gpu_target=matches!(input.device.as_str(),"cuda"|"xpu");
    if gpu_target{
        if !(1..=512).contains(&input.gpu_layers){
            return Err(invalid("GPU optimization requires explicit --gpu-layers 1..512"));
        }
        let device=input.runtime_device.as_deref().unwrap_or("");
        if device.is_empty()||device.len()>80
            ||!device.bytes().all(|b|b.is_ascii_alphanumeric()||matches!(b,b'_'|b'-'|b'/'))
        {return Err(invalid("GPU optimization requires an explicitly selected runtime device ID (inspect with ojon backends --probe); device compatibility is checked at execution")); }
    }else if input.runtime_device.is_some()||input.gpu_layers!=0{
        return Err(invalid("CPU/auto guided optimization requires zero GPU layers and no runtime device"));
    }
    if !input.min_saving_percent.is_finite()
        ||!(0.0..=95.0).contains(&input.min_saving_percent){
        return Err(invalid("minimum saving must be within 0..95 percent"));
    }
    if !input.max_loss_points.is_finite()
        ||!(0.0..=100.0).contains(&input.max_loss_points){
        return Err(invalid("maximum quality loss must be within 0..100 percentage points"));
    }
    if !input.max_ppl_regression_pct.is_finite()
        ||!(0.0..=100.0).contains(&input.max_ppl_regression_pct){
        return Err(invalid("maximum relative perplexity regression must be 0..100 percent"));
    }
    if input.server_tool.as_ref().is_some_and(|p|p.as_os_str().is_empty()){
        return Err(invalid("persistent evaluator path cannot be empty"));
    }
    // A previously chosen server missing from disk is NOT permission to use
    // an unrelated llama-cli backend without informing the user.
    if backends.iter().any(|b|b.spec.id==BackendId::LlamaServer
        &&b.executable.is_none()
        &&matches!(b.found_via,Some("local settings file"|"explicit environment override"|"Kathamo selected executable"))){
        return Err(invalid("configured persistent evaluator is missing; repair or clear its local path"));
    }
    if input.candidates.len()>6{return Err(invalid("at most six existing candidate GGUFs"));}
    let formats=precisions_for(input);
    if formats.is_empty()||formats.len()>6{
        return Err(invalid("guided search requires 1..=6 precision candidates"));
    }
    let mut unique=HashSet::new();
    for precision in &formats{
        if !crate::quant_parity::allowed_parity_precision(*precision)
            ||!unique.insert(precision.as_str()){
            return Err(invalid("guided precisions must be distinct smaller-than-Q4 parity formats"));
        }
    }
    if !(16..=8192).contains(&input.bench_prompt_tokens)
        ||!(16..=2048).contains(&input.bench_generation_tokens)
        ||!(1..=10).contains(&input.bench_repetitions)
        ||!input.balanced_min_speed_ratio.is_finite()
        ||!(0.50..=1.0).contains(&input.balanced_min_speed_ratio){
        return Err(invalid("invalid benchmark length, repetitions, or Balanced speed fraction"));
    }
    let mut names=HashSet::new();
    for(label,_)in &input.candidates{
        if label.is_empty()||label.len()>80||
            !label.bytes().all(|v|v.is_ascii_alphanumeric()||matches!(v,b'_'|b'-'))||
            !names.insert(label.clone()){
            return Err(invalid("candidate names must be unique ASCII letters/numbers/-/_"));
        }
    }
    let probe=probe_model(&input.model)?;
    let source=probe.path.clone();
    let original_gguf=if probe.format=="gguf"{
        let g=inspect_gguf(&source)?;
        matches!(g.file_type,Some(0|1|32))&&g.split_count.unwrap_or(1)==1
    }else{false};
    let compressed_gguf=if probe.format=="gguf"{!original_gguf}else{false};
    let baseline=match &input.current_baseline{
        Some(p)=>Some(p.canonicalize()?),
        None if compressed_gguf=>Some(source.clone()),
        None=>None,
    };
    let baseline=if let Some(p)=baseline{
        let b=inspect_gguf(&p)?;
        if b.tensor_count==0||b.split_count.unwrap_or(1)>1||
            b.file_type.is_none()||matches!(b.file_type,Some(0|1|32)){
            return Err(invalid("current quality baseline must be a compressed single-file GGUF"));
        }
        Some(p)
    }else{None};
    let suite=match &input.suite{
        Some(p)=>{
            let p=p.canonicalize()?;
            parse_eval_suite(&p)?;
            Some(p)
        }
        None=>None
    };
    let out=match &input.output_dir{
        Some(p)=>{
            let p=p.canonicalize()?;
            if !p.is_dir(){return Err(invalid("output must be an existing directory"));}
            Some(p)
        }
        None=>None
    };
    let options=configured_options(input,backends);
    let quantizer_available=executable(backends,BackendId::LlamaQuantize).is_some();
    let evaluator_available=executable(backends,BackendId::LlamaCli).is_some();
    let mut stages=vec![
        stage("inspect","Inspect model and components","ready",
              "Model metadata inspected; no weight payload loaded",None)
    ];
    let mut checks=Vec::<String>::new();
    if probe.category==ModelCategory::VisionLanguage{
        checks.push("Vision model: matching mmproj and image-input evaluation are mandatory before claiming quality parity. This basic workflow cannot yet automate those requirements.".into());
    }
    let mut gpu_ready=!gpu_target;
    if gpu_target{
        if let Some(parent)=out.as_deref(){
            match assess_resources(probe.bytes,parent,ResourceMode::Gguf,&input.device){
                Ok(capacity)=>{
                    gpu_ready=capacity.device_measured&&capacity.disk_safe!=Some(false)
                        &&capacity.ram_safe!=Some(false);
                    if !gpu_ready{checks.push("Selected GPU lacks reliable live vendor telemetry or basic host capacity; use manual backend commands.".into());}
                },
                Err(e)=>checks.push(format!("GPU preflight failed: {e}")),
            }
        }else{checks.push("GPU verification needs an existing output directory.".into());}
        if input.ppl_corpus.is_some(){
            gpu_ready=false;
            checks.push("GPU-specific perplexity device pinning is unimplemented; refusing to silently test PPL on another device.".into());
        }
        if executable(backends,BackendId::LlamaBench).is_none(){
            gpu_ready=false;
            checks.push("GPU selection requires llama-bench for verified backend throughput.".into());
        }
    }
    if suite.is_none(){
        checks.push("No held-out test suite supplied; result cannot prove quality parity.".into());
    }
    if input.ppl_corpus.is_some() && executable(backends,BackendId::LlamaPerplexity).is_none(){
        checks.push("Perplexity corpus provided but llama-perplexity not detected. Execution requires this backend.".into());
    }
    if input.ppl_corpus.is_none(){
        checks.push("Optional held-out perplexity gate not configured; the task suite alone may miss language-model degradation.".into());
    }
    if input.imatrix.is_none() && original_gguf{
        checks.push("No calibrated importance matrix supplied; low-bit candidates will use uncalibrated recipes.".into());
    }
    // Metadata + read-only host resource telemetry; never infer quantizer peak
    // RSS from GGUF size. Unknown resources do not authorize automatic writes.
    let mut capacity_pass=false;
    let mut resource_snapshot=Value::Null;
    if original_gguf && probe.category==ModelCategory::Language{
        if let Some(parent)=out.as_deref(){
            match assess_resources(probe.bytes,parent,ResourceMode::Gguf,"cpu"){
                Ok(info)=>{
                    capacity_pass=info.ram_safe==Some(true)&&info.disk_safe==Some(true);
                    if !capacity_pass {
                        checks.push(format!("Automatic quantization resource policy not verified: {}. Free at least the stated RAM and scratch budget; use advanced manual tools for explicitly accepted risks.",info.status()));
                    }
                    resource_snapshot=info.as_json();
                },
                Err(e)=>checks.push(format!("RAM/scratch resource preflight failed: {e}")),
            }
        }
    }
    let base_text=baseline.as_ref().map(|p|path_text(p));
    let output_text=out.as_ref().map(|p|path_text(p));
    let mut next=None::<String>;
    let mut route="requires_input";
    let mut action="blocked";
    let mut permitted=false;
    let mut argv:Option<Vec<String>>=None;

    // Avoid misleadingly proposing text-only benchmarking as full VLM proof.
    let text_like=probe.category==ModelCategory::Language;
    if !text_like {
        checks.push("No fully verified one-click quality optimization path for this model category. Use the individual model-family workflow.".into());
    }else if input.goal==OptimizeGoal::Fastest
        && executable(backends,BackendId::LlamaBench).is_none(){
        route="speed_benchmark_required";
        next=Some("Install llama-bench to compare matching-device throughput".into());
        checks.push("Fastest requires installed llama-bench; size is not a speed proxy.".into());
    }else if gpu_target&&!gpu_ready{
        route="explicit_gpu_verification_required";
        next=Some("Provide a real CUDA/SYCL llama.cpp device and working live GPU telemetry".into());
    }else if !input.candidates.is_empty(){
        route="existing_candidate_comparison";
        if let (Some(base),Some(suite),Some(out))=(&baseline,&suite,&out){
            let report=next_compare_report(out)?;
            let valid=prepare_existing_quant_comparison(
                base,&input.candidates,suite,&report,options.clone());
            match valid {
                Ok(_)=>{
                    if !evaluator_available{
                        checks.push("llama-cli not discovered; cannot execute quality comparison.".into());
                    }else if input.ppl_corpus.is_some()
                        && executable(backends,BackendId::LlamaPerplexity).is_none(){
                        // A requested quality gate must never silently be skipped.
                    }else {
                        let mut cmd=vec!["ojon".into(),"quant-compare-run".into(),
                            path_text(base),path_text(suite),path_text(&report)];
                        for(label,path)in &input.candidates {
                            cmd.push("--candidate".into());
                            cmd.push(format!("{label}={}",path_text(&path.canonicalize()?)));
                        }
                        cmd.extend(["--eval-tool".into(),path_text(&options.eval_tool),
                            "--max-loss-points".into(),input.max_loss_points.to_string(),
                            "--min-saving-percent".into(),input.min_saving_percent.to_string()]);
                        if let Some(corpus)=options.ppl_corpus.as_ref(){
                            cmd.extend(["--ppl-corpus".into(),path_text(corpus),
                                "--ppl-tool".into(),path_text(&options.ppl_tool),
                                "--max-ppl-regression-pct".into(),
                                options.max_ppl_regression_pct.to_string()]);
                        }
                        cmd.push("--confirm".into());
                        argv=Some(cmd);
                        permitted=true;
                        action="compare_existing_quants";
                        next=Some("Compare existing quantized candidates to the current baseline".into());
                    }
                },
                Err(e)=>checks.push(format!("Existing candidate comparison cannot run: {e}"))
            }
        }else{
            checks.push("To compare existing quants, provide current Q4 baseline, held-out suite and output directory.".into());
        }
    }else if original_gguf && input.tensor_sensitivity.is_some() {
        route="adaptive_quantize_and_compare";
        if let (Some(base),Some(suite),Some(out),Some(scores))=
            (&baseline,&suite,&out,&input.tensor_sensitivity)
        {
            if !capacity_pass{
                checks.push("Adaptive workflow requires verified host RAM/scratch budgets.".into());
            }else if input.ppl_corpus.is_some(){
                checks.push("Adaptive runner cannot evaluate the requested PPL gate; use ordinary parity comparison.".into());
            }else if !quantizer_available||!evaluator_available{
                checks.push("Adaptive workflow requires installed quantizer and llama-cli.".into());
            }else{
                match prepare_adaptive_against_reference(
                    &source,base,scores,out,suite,&options.quantizer,&options.eval_tool,
                    options.imatrix.as_deref(),&[3.0,3.5,4.0],options.threads,0,None,
                    options.tokens,input.max_loss_points)
                {
                    Ok(_)=>{permitted=true;action="adaptive_quantize_and_compare";
                        next=Some("Run native tensor-specific adaptive quantization with Q4-referenced held-out tasks".into());},
                    Err(e)=>checks.push(format!("Adaptive workflow preflight blocked: {e}")),
                }
            }
        }else{
            checks.push("Adaptive workflow requires high-precision original, Q4 baseline, tensor sensitivity, suite and output directory.".into());
        }
    }else if original_gguf {
        if let (Some(base),Some(suite),Some(out))=(&baseline,&suite,&out){
            route="quantize_and_compare";
            if !capacity_pass {
                // Never start an automatic 3-candidate sweep on unverified disk/RAM.
            }else if !quantizer_available||!evaluator_available {
                checks.push("Both llama-quantize and llama-cli must be discovered for end-to-end candidate search.".into());
            }else if input.ppl_corpus.is_some()
                && executable(backends,BackendId::LlamaPerplexity).is_none(){
                // Requested PPL gate is mandatory for the guided workflow.
            }else{
                match prepare_quant_parity(&source,base,suite,out,options.clone()){
                    Ok(_)=>{
                        let mut cmd=vec!["ojon".into(),"quant-parity-run".into(),
                            path_text(&source),path_text(base),path_text(suite),path_text(out),
                            "--precisions".into(),precision_csv(input),
                            "--quantizer".into(),path_text(&options.quantizer),
                            "--eval-tool".into(),path_text(&options.eval_tool),
                            "--max-buffer-mib".into(),"512".into(),
                            "--min-saving-percent".into(),input.min_saving_percent.to_string(),
                            "--max-loss-points".into(),input.max_loss_points.to_string(),
                            "--gpu-layers".into(),options.gpu_layers.to_string()]; 
                        if let Some(device)=options.device.as_deref(){
                            cmd.extend(["--device".into(),device.into()]);
                        }
                        if let Some(imatrix)=options.imatrix.as_ref(){
                            cmd.extend(["--imatrix".into(),path_text(imatrix)]);
                        }
                        if let Some(corpus)=options.ppl_corpus.as_ref(){
                            cmd.extend(["--ppl-corpus".into(),path_text(corpus),
                                "--ppl-tool".into(),path_text(&options.ppl_tool),
                                "--max-ppl-regression-pct".into(),
                                options.max_ppl_regression_pct.to_string()]);
                        }
                        cmd.push("--confirm".into());
                        argv=Some(cmd);permitted=true;
                        action="quantize_and_compare";
                        next=Some("Generate and test smaller precision candidates from the original weights".into());
                    }
                    Err(e)=>checks.push(format!("Quantization parity preflight blocked: {e}"))
                }
            }
        }else{
            route="build_baseline_first";
            checks.push("Provide an existing Q4_K_M baseline and a held-out task suite to establish whether smaller results preserve quality.".into());
            if out.is_none(){checks.push("Select a writable output directory for any future conversion.".into());}
            next=Some("Select a current Q4 baseline, held-out suite and output folder".into());
        }
    }else if compressed_gguf{
        route="download_or_select_candidate";
        checks.push("Existing GGUF is already compressed. Do not turn Q4 into Q3 by lossy requantization.".into());
        next=Some("Add a smaller GGUF from the SAME model revision for a measured comparison".into());
    }else if probe.format=="hf_checkpoint_directory"{
        route="convert_hf_first";
        checks.push("HF checkpoint must first be exported to compatible high-precision GGUF with original tokenizer and model components; no automatic conversion performed.".into());
        next=Some("Prepare supported HF-to-GGUF source conversion and a quantized reference".into());
    }else{
        route="unsupported_source";
        checks.push("No verified quantization workflow for this source format and architecture.".into());
    }

    if let Some(c)=argv.clone(){
        stages.push(stage("verify_prerequisites","Check backend and output compatibility",
            "ready","Backend filenames detected; real runtime compatibility not yet established",None));
        stages.push(stage("run","Evaluate optimization candidates","awaiting_approval",
            "Requires explicit user approval; command will perform real model inference and/or quantization",Some(c)));
        stages.push(stage("verify_quality","Compare size, capability and hardware","not_started",
            "Accept only measured task-parity results and separately confirm runtime memory and speed",None));
        stages.push(stage("optional_lora","Optional LoRA quality repair and behavioural customization",
            "optional_not_automated",
            "Quantization recovery and refusal-reduction are separate user-approved experiments",None));
    }else{
        stages.push(stage("resolve","Resolve missing prerequisites","blocked",
            "A real optimization run is not authorized until the listed checks are satisfied",None));
    }
    // NOTE: .ojonq, EXL3, OpenVINO, and VLM are excluded until
    // an exact tested model-family+runtime recipe exists.
    Ok(json!({
        "schema":"ojon.optimize_workflow.v1",
        "model":probe.as_json(),
        "current_baseline":base_text,
        "heldout_suite":suite.map(|p|path_text(&p)),
        "calibration_imatrix":input.imatrix.as_ref().map(|p|path_text(p)),
        "tensor_sensitivity":input.tensor_sensitivity.as_ref().map(|p|path_text(p)),
        "heldout_perplexity_corpus":input.ppl_corpus.as_ref().map(|p|path_text(p)),
        "max_relative_perplexity_regression_pct":input.max_ppl_regression_pct,
        "output_directory":output_text,
        "minimum_cpu_resource_preflight":resource_snapshot,
        "goal":input.goal.label(),
        "target_device":input.device,
        "runtime_gpu_allocated":false,
        "weights_loaded":false,
        "downloads_or_writes_performed":false,
        "route":route,
        "next_action":action,
        "next_user_step":next,
        "ready_for_user_approved_execution":permitted,
        "recommended_command_argv":argv,
        "recommended_precision_order":formats.iter().map(|v|v.as_str()).collect::<Vec<_>>(),
        "precision_selection":if input.precision_candidates.is_some(){"user_configured"}else{"conservative_default"},
        "evaluation_engine":if options.server_tool.is_some(){"persistent_llama_server"}else{"llama_cli_per_task"},
        "evaluation_server_path":options.server_tool.as_ref().map(|p|path_text(p)),
        "benchmark_settings":{"prompt_tokens":input.bench_prompt_tokens,
            "generation_tokens":input.bench_generation_tokens,
            "repetitions":input.bench_repetitions,
            "balanced_min_speed_ratio":input.balanced_min_speed_ratio},
        "quality_thresholds":{"max_loss_points":input.max_loss_points,
             "min_saving_percent":input.min_saving_percent},
        "stages":stages,
        "checks":checks,
        "result":"NOT_TESTED",
        "winner":Value::Null,
        "general_quality_preservation_proven":false,
        "warning":"Read-only next-step plan, not an optimization run. No quality, free-VRAM, model revision identity, or runtime capability is proven."
    }))
}

/// Execute the exact existing Ojon quantization/verification adapters after
/// explicit approval. Unlike invoking the suggested command in a shell,
/// this calls typed Rust APIs directly; filenames never become shell syntax.
/// The original model, reference GGUF and candidate files remain unchanged.
/// The upstream runners write evidence with exclusive-create semantics.
pub fn run_optimize_workflow(
    request:&OptimizeWorkflowRequest,
    backends:&[FoundBackend],
    approved:bool,
    mut log:impl FnMut(&str),
)->io::Result<Value>{
    if !approved{
        return Err(io::Error::new(ErrorKind::PermissionDenied,
            "optimizer execution requires explicit --confirm; no model or GPU touched"));
    }
    let decision=plan_optimize_workflow(request,backends)?;
    if decision["ready_for_user_approved_execution"]!=true{
        return Err(invalid(format!(
            "optimizer cannot run: {}; prerequisites: {}",
            decision["route"],
            decision["checks"]
        )));
    }
    // Plan validated concrete CUDA/SYCL selection and free-memory telemetry.
    // Real runtime compatibility still requires the load and benchmark checks below.
    let suite=request.suite.as_ref()
        .ok_or_else(||invalid("a held-out task suite is mandatory"))?;
    let out=request.output_dir.as_ref()
        .ok_or_else(||invalid("an existing output directory is mandatory"))?;
    let mut options=configured_options(request,backends);
    let gpu_target=matches!(request.device.as_str(),"cuda"|"xpu");
    // Metadata-only planning never runs local binaries. Once approved, query
    // the EXACT executable we will call; no vendor-to-device-ID guessing.
    let runtime_device_evidence=if gpu_target{
        let wanted=options.device.as_deref()
            .ok_or_else(||invalid("explicit runtime device is required for GPU execution"))?;
        let detected=probe_runtime_devices(&options.eval_tool)?;
        if !detected.iter().any(|id|id==wanted){
            return Err(invalid(format!("runtime '{}' does not advertise selected device '{wanted}'",
                options.eval_tool.display())));
        }
        json!({"runtime":options.eval_tool.to_string_lossy(),
            "selected_device":wanted,"advertised_device_ids":detected,
            "device_id_advertised":true,
            "physical_gpu_identity_verified":false})
    }else{json!({"status":"CPU_ONLY_NO_GPU_DEVICE_NEEDED"})};
    let action=decision["next_action"].as_str().unwrap_or("");
    let quantizer_evidence=if matches!(action,"quantize_and_compare"|"adaptive_quantize_and_compare"){
        let advertised=probe_quantizer_capabilities(&options.quantizer)?;
        if options.imatrix.is_some() && advertised["supports_imatrix"]!=true{
            return Err(invalid("selected quantizer did not advertise --imatrix; calibration cannot be silently skipped"));
        }
        if action=="adaptive_quantize_and_compare" && advertised["supports_tensor_type"]!=true{
            return Err(invalid("selected quantizer did not advertise mixed-precision tensor controls"));
        }
        if action=="quantize_and_compare"{
            let supported=advertised["precision_formats"].as_array()
                .ok_or_else(||invalid("no quantizer format evidence"))?;
            let original_count=options.precisions.len();
            options.precisions.retain(|kind|supported.iter().any(|t|t.as_str()==Some(kind.as_str())));
            if options.precisions.len()!=original_count && request.precision_candidates.is_some(){
                return Err(invalid("a user-selected precision is not advertised by this installed quantizer"));
            }
            if options.precisions.is_empty(){
                return Err(invalid("installed quantizer supports none of the permitted guided precisions"));
            }
            if options.precisions.len()!=original_count{
                log("[ojon] Skipping candidate formats not advertised by the selected quantizer");
            }
        }
        advertised
    }else{json!({"status":"NOT_APPLICABLE_NO_NEW_QUANTIZATION"})};
    log("[ojon] User approved optimizer; backend capabilities checked without loading model weights");
    let (kind,result)=match decision["next_action"].as_str(){
        Some("compare_existing_quants")=>{
            let baseline=match request.current_baseline.as_ref(){
                Some(path)=>path,
                None=>&request.model,
            };
            let report=next_compare_report(out)?;
            let plan=prepare_existing_quant_comparison(
                baseline,&request.candidates,suite,&report,options.clone())?;
            log("[ojon] Running real matched task comparisons; this does not generate new quantized weights");
            ("existing_quant_comparison",run_existing_quant_comparison(&plan,|line|log(line))?)
        },
        Some("adaptive_quantize_and_compare")=>{
            let baseline=request.current_baseline.as_ref()
                .ok_or_else(||invalid("adaptive workflow needs a current Q4 baseline"))?;
            let scores=request.tensor_sensitivity.as_ref()
                .ok_or_else(||invalid("adaptive workflow needs sensitivity data"))?;
            let mut plan=prepare_adaptive_against_reference(&request.model,baseline,scores,out,suite,
                &options.quantizer,&options.eval_tool,options.imatrix.as_deref(),
                &[3.0,3.5,4.0],options.threads,options.gpu_layers,
                options.device.as_deref(),options.tokens,request.max_loss_points)?;
            plan.server_tool=options.server_tool.clone();
            log("[ojon] Using existing adaptive allocator, Q4-referenced quality evaluation and tensor recipes");
            ("adaptive_quant_comparison",run_adaptive(&plan,|line|log(line))?)
        },
        Some("quantize_and_compare")=>{
            let baseline=request.current_baseline.as_ref()
                .ok_or_else(||invalid("quantization parity requires an existing Q4 reference"))?;
            let plan=prepare_quant_parity(
                &request.model,baseline,suite,out,options.clone())?;
            log("[ojon] Generating candidates from original high-precision weights, not from the Q4 reference");
            ("high_precision_quant_parity",run_quant_parity(&plan,|line|log(line))?)
        },
        _=>return Err(invalid("unsupported execution route; no candidate work started")),
    };
    let ranked=rank_quality_gated_candidates(kind,&result);
    let mut attempts=Vec::new();
    let mut loadable=Vec::new();
    for possible in &ranked{
        let Some(path)=possible["path"].as_str() else {continue;};
        log("[ojon] Checking measured-quality candidate against the actual selected llama.cpp backend");
        let attempt=match smoke_gguf_capture_with_device(
            Path::new(path),&options.eval_tool,
            "Reply with the single word READY.",16,options.gpu_layers as u32,
            options.device.as_deref()
        ){
            Ok(capture) if !capture.stdout.trim().is_empty()=>json!({
                "path":path,"status":"LOAD_AND_GENERATION_PASSED",
                "backend":"llama-cli","device":options.device,
                "gpu_layers":options.gpu_layers,
                "captured_output_chars":capture.stdout.chars().count()
            }),
            Ok(_)=>json!({"path":path,"status":"LOAD_CHECK_FAILED",
                "reason":"runtime exited but generated no response"}),
            Err(e)=>json!({"path":path,"status":"LOAD_CHECK_FAILED","reason":e.to_string()}),
        };
        if attempt["status"]=="LOAD_AND_GENERATION_PASSED"{
            loadable.push(possible.clone());
        }
        attempts.push(attempt);
        // The smallest-only policy does not load larger candidates needlessly.
        if request.goal==OptimizeGoal::Smallest&&!loadable.is_empty(){break;}
    }
    let mut benchmark_results=Vec::new();
    let mut comparison_reference:Option<PathBuf>=None;
    if let Some(tool)=executable(backends,BackendId::LlamaBench){
        for (idx,possible) in loadable.iter().enumerate(){
            let Some(path)=possible["path"].as_str() else{continue;};
            let stamp=SystemTime::now().duration_since(UNIX_EPOCH)
                .map(|d|d.as_nanos()).unwrap_or(0);
            let receipt=out.join(format!("ojon-bench-{}-{stamp}-{idx}.json",std::process::id()));
            log("[ojon] Measuring matched prefill/decode performance of quality-passing candidate");
            let entry=match prepare_bench(Path::new(path),&tool,
                request.bench_prompt_tokens,request.bench_generation_tokens,
                request.bench_repetitions,
                options.gpu_layers,options.device.as_deref())
                .and_then(|plan|run_bench(&plan,&receipt))
            {
                Ok(metrics)=>{
                    let comparison=comparison_reference.as_ref()
                        .map(|reference|compare_bench_files(reference,&receipt))
                        .transpose();
                    match comparison{
                        Ok(check) if check.as_ref().is_some_and(|v|v["benchmark_conditions_match"]!=true)=>
                            json!({"path":path,"status":"INCOMPARABLE_BENCHMARK","comparison":check}),
                        Err(e)=>json!({"path":path,"status":"BENCHMARK_COMPARISON_FAILED","reason":e.to_string()}),
                        _=>{
                            if comparison_reference.is_none(){comparison_reference=Some(receipt.clone());}
                            json!({"path":path,"status":"MEASURED","receipt":receipt.to_string_lossy(),
                                "metrics":metrics})
                        }
                    }
                },
                Err(e)=>json!({"path":path,"status":"BENCHMARK_FAILED","reason":e.to_string()}),
            };
            benchmark_results.push(entry);
        }
    }
    let (mut candidate,selection_policy)=select_workflow_goal_with_ratio(
        &loadable,&benchmark_results,request.goal.label(),
        request.balanced_min_speed_ratio);
    // A GPU run without comparable measured throughput cannot claim the target
    // device was benchmark-certified. CPU smallest may run without llama-bench.
    if gpu_target && candidate["path"].as_str().is_some()
        && !benchmark_results.iter().any(|b|b["path"]==candidate["path"]&&b["status"]=="MEASURED")
    {
        candidate=Value::Null;
    }
    let load_check=if let Some(path)=candidate["path"].as_str(){
        attempts.iter().find(|a|a["path"]==path).cloned()
            .unwrap_or(json!({"status":"LOAD_EVIDENCE_MISSING"}))
    }else if attempts.is_empty(){
        json!({"status":"NOT_RUN_NO_QUALITY_PASSING_CANDIDATE"})
    }else{
        json!({"status":"NO_CANDIDATE_PASSED_BACKEND_AND_BENCHMARK_GATES"})
    };
    let performance=if let Some(path)=candidate["path"].as_str(){
        benchmark_results.iter().find(|b|b["path"]==path).cloned()
            .unwrap_or(json!({"status":"NOT_RUN_LLAMA_BENCH_UNAVAILABLE"}))
    }else{json!({"status":"NO_SELECTED_CANDIDATE"})};
    Ok(json!({
        "schema":"ojon.optimize_workflow_result.v1",
        "mode":if gpu_target{"user_confirmed_explicit_gpu"}else{"user_confirmed_cpu_only"},
        "route":kind,
        "status":if candidate.is_null(){"NO_CANDIDATE_PASSED"}else{"PROVISIONAL_CANDIDATE_FOUND"},
        "model_source":request.model.to_string_lossy(),
        "goal":request.goal.label(),
        "quality_reference":decision["current_baseline"],
        "evaluation_engine":if options.server_tool.is_some(){"persistent_llama_server"}else{"llama_cli_per_task"},
        "evaluation_server_path":options.server_tool.as_ref().map(|p|path_text(p)),
        "backend_capability_evidence":{"quantizer":quantizer_evidence,
            "runtime_device":runtime_device_evidence},
        "source_provenance":{"source_path":request.model.to_string_lossy(),
            "reference_path":decision["current_baseline"],
            "sensitivity":request.tensor_sensitivity.as_ref().map(|p|p.to_string_lossy().to_string()),
            "revision_identity":"NOT_CRYPTOGRAPHICALLY_VERIFIED"},
        "evaluated_tasks_only":true,
        "heldout_perplexity_requested":request.ppl_corpus.is_some(),
        "runtime_speed_not_automatically_tested":performance["status"]!="MEASURED",
        "global_quality_lossless_proven":false,
        "hardware_peak_ram_vram_fit_proven":false,
        "hardware_preflight_measured":decision["minimum_cpu_resource_preflight"],
        "provisional_candidate":candidate,
        "selected_candidate_runtime_load":load_check,
        "candidate_load_attempts":attempts,
        "quality_passing_candidate_count":ranked.len(),
        "selected_candidate_performance":performance,
        "candidate_benchmarks":benchmark_results,
        "requested_backend_device":options.device,
        "requested_gpu_layers":options.gpu_layers,
        "selection_policy":selection_policy,
        "precision_candidates":options.precisions.iter().map(|v|v.as_str()).collect::<Vec<_>>(),
        "requested_precisions":precisions_for(request).iter().map(|v|v.as_str()).collect::<Vec<_>>(),
        "benchmark_settings":{"prompt_tokens":request.bench_prompt_tokens,
            "generation_tokens":request.bench_generation_tokens,
            "repetitions":request.bench_repetitions,
            "balanced_min_speed_ratio":request.balanced_min_speed_ratio},
        "runtime_load_verified_on_cpu":!gpu_target && load_check["status"]=="LOAD_AND_GENERATION_PASSED",
        "selected_backend_load_verified":load_check["status"]=="LOAD_AND_GENERATION_PASSED",
        "evidence":result,
        "note":"Inspect all candidate task scores, size, RAM, VRAM and latency before deploying. There may be no candidate that passes."
    }))
}

#[cfg(test)]
mod tests{
    use super::*;
    fn tiny_model(path:&Path,kind:u32){
        use std::fs::{self,File};
        use std::io::{Seek,Write};
        fn s(f:&mut File,value:&str){
            f.write_all(&(value.len() as u64).to_le_bytes()).unwrap();
            f.write_all(value.as_bytes()).unwrap();
        }
        let mut f=File::create(path).unwrap();
        f.write_all(b"GGUF").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        s(&mut f,"general.architecture");f.write_all(&8u32.to_le_bytes()).unwrap();
        s(&mut f,"qwen3");
        s(&mut f,"general.file_type");f.write_all(&4u32.to_le_bytes()).unwrap();
        f.write_all(&kind.to_le_bytes()).unwrap();
        s(&mut f,"blk.0.attn_output.weight");
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&2u64.to_le_bytes()).unwrap();
        f.write_all(&2u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        let pos=f.stream_position().unwrap();
        f.write_all(&vec![0;((32-pos%32)%32) as usize]).unwrap();
        f.write_all(&[0;32]).unwrap();
        drop(f);
        assert!(fs::metadata(path).unwrap().len()>0);
    }
    #[test]
    fn selects_compressed_reference_without_requantization(){
        use std::fs;
        use std::time::{SystemTime,UNIX_EPOCH};
        let stamp=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir=std::env::temp_dir().join(format!("ojon-workflow-test-{}-{stamp}",std::process::id()));
        fs::create_dir(&dir).unwrap();
        let base=dir.join("model-Q4.gguf");
        tiny_model(&base,15);
        let request=OptimizeWorkflowRequest::new(base.clone());
        let plan=plan_optimize_workflow(&request,&[]).unwrap();
        assert_eq!(plan["route"],"download_or_select_candidate");
        assert_eq!(plan["ready_for_user_approved_execution"],false);
        assert_eq!(plan["downloads_or_writes_performed"],false);
        assert_eq!(plan["weights_loaded"],false);
        assert_eq!(plan["result"],"NOT_TESTED");
        let orig=dir.join("original-f16.gguf");
        tiny_model(&orig,1);
        let plan=plan_optimize_workflow(&OptimizeWorkflowRequest::new(orig),&[]).unwrap();
        assert_eq!(plan["route"],"build_baseline_first");
        fs::remove_dir_all(&dir).unwrap();
    }
    #[test]
    fn prepares_valid_existing_comparison_and_never_runs_it(){
        use std::fs;
        use std::time::{SystemTime,UNIX_EPOCH};
        let stamp=SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir=std::env::temp_dir().join(format!("ojon-workflow-compare-{}-{stamp}",std::process::id()));
        fs::create_dir(&dir).unwrap();
        let base=dir.join("baseline.gguf");tiny_model(&base,15);
        let smaller=dir.join("smaller.gguf");tiny_model(&smaller,27);
        let suite=dir.join("suite.json");
        let tasks=(0..6).map(|i|json!({
            "id":format!("w-{i}"),
            "category":if i<2{"coding"}else if i<4{"reasoning"}else{"tool_call"},
            "prompt":"Return 42","expected":"42","match":"exact"
        })).collect::<Vec<_>>();
        fs::write(&suite,serde_json::to_vec(&json!({
            "schema":"ojon.task_suite.v1","id":"workflow-fixture","tasks":tasks
        })).unwrap()).unwrap();
        let mut req=OptimizeWorkflowRequest::new(base);
        req.candidates=vec![("smaller".into(),smaller)];
        req.suite=Some(suite);
        req.output_dir=Some(dir.clone());
        let cli_spec=*crate::discovery::backend_specs().iter()
            .find(|s|s.id==BackendId::LlamaCli).unwrap();
        let result=plan_optimize_workflow(&req,&[FoundBackend{
            spec:cli_spec,executable:Some(PathBuf::from("llama-cli")),
            found_via:Some("test fixture")
        }]).unwrap();
        assert_eq!(result["route"],"existing_candidate_comparison");
        assert_eq!(result["next_action"],"compare_existing_quants");
        assert_eq!(result["ready_for_user_approved_execution"],true);
        assert_eq!(result["recommended_command_argv"][1],"quant-compare-run");
        assert_eq!(result["result"],"NOT_TESTED");
        assert!(!dir.join("ojon-quant-comparison.json").exists());
        fs::remove_dir_all(&dir).unwrap();
    }
    #[test]
    fn ranked_candidates_skip_failures_and_prefer_smallest(){
        let report=json!({"results":[
            {"name":"failed","path":"bad.gguf","bytes":1,
             "status":"FAILED_QUALITY_GATE"},
            {"name":"large","path":"large.gguf","bytes":300,
             "status":"NO_DETECTED_TEST_REGRESSION"},
            {"name":"small","path":"small.gguf","bytes":100,
             "status":"NO_DETECTED_TEST_REGRESSION"}
        ]});
        let ranked=rank_quality_gated_candidates("existing_quant_comparison",&report);
        assert_eq!(ranked.len(),2);
        assert_eq!(ranked[0]["path"],"small.gguf");
        assert_eq!(ranked[1]["path"],"large.gguf");
        let generated=json!({"results":[
            {"type":"IQ3_M","output":"out.gguf","bytes":200,
             "status":"NO_DETECTED_TASK_REGRESSION"}
        ]});
        assert_eq!(rank_quality_gated_candidates("high_precision_quant_parity",&generated)[0]["path"],"out.gguf");
    }
    #[test]
    fn execution_requires_explicit_approval_even_if_inputs_are_missing(){
        let request=OptimizeWorkflowRequest::new(PathBuf::from("not-loaded.gguf"));
        let error=run_optimize_workflow(&request,&[],false,|_|{}).unwrap_err();
        assert_eq!(error.kind(),ErrorKind::PermissionDenied);
    }
    #[test]
    fn runtime_precision_choices_equal_planner_choices(){
        let input=OptimizeWorkflowRequest::new(PathBuf::new());
        let opt=configured_options(&input,&[]);
        let types=opt.precisions.iter().map(|p|p.as_str()).collect::<Vec<_>>().join(",");
        assert_eq!(types,precision_csv(&input));
    }
    #[test]
    fn goals_and_limits_are_explicit(){
        assert_eq!(OptimizeGoal::parse("smallest").unwrap(),OptimizeGoal::Smallest);
        assert!(OptimizeGoal::parse("magical").is_err());
        let mut request=OptimizeWorkflowRequest::new(PathBuf::new());
        request.goal=OptimizeGoal::Smallest;
        assert_eq!(precision_csv(&request),"IQ3_M,Q3_K_M,Q3_K_L,IQ4_XS");
    }
    #[test]
    fn explicit_gpu_routes_reject_unknown_device_and_offload(){
        let mut req=OptimizeWorkflowRequest::new(PathBuf::from("no-model.gguf"));
        req.device="cuda".into();
        req.gpu_layers=99;
        assert!(plan_optimize_workflow(&req,&[]).is_err());
        req.runtime_device=Some("SYCL0".into());
        assert!(plan_optimize_workflow(&req,&[]).is_err());
        req.runtime_device=Some("CUDA0".into());
        req.gpu_layers=0;
        assert!(plan_optimize_workflow(&req,&[]).is_err());
        req.device="cpu".into();
        req.gpu_layers=99;
        assert!(plan_optimize_workflow(&req,&[]).is_err());
    }
    #[test]
    fn rejects_unsupported_guided_precisions_and_benchmark_knobs_before_model_access(){
        let mut request=OptimizeWorkflowRequest::new(PathBuf::from("missing.gguf"));
        request.precision_candidates=Some(vec![Precision::Q5KM]);
        assert!(plan_optimize_workflow(&request,&[]).is_err());
        request.precision_candidates=Some(vec![Precision::IQ3M,Precision::IQ3M]);
        assert!(plan_optimize_workflow(&request,&[]).is_err());
        request.precision_candidates=Some(vec![Precision::IQ3M]);
        request.bench_repetitions=0;
        assert!(plan_optimize_workflow(&request,&[]).is_err());
        request.bench_repetitions=3;
        request.balanced_min_speed_ratio=f64::NAN;
        assert!(plan_optimize_workflow(&request,&[]).is_err());
        request.balanced_min_speed_ratio=0.90;
        assert_eq!(precision_csv(&request),"IQ3_M");
    }
    #[test]
    fn invalid_arguments_do_not_invent_a_plan(){
        let mut req=OptimizeWorkflowRequest::new(PathBuf::from("does-not-exist.gguf"));
        req.max_loss_points=f64::NAN;
        assert!(plan_optimize_workflow(&req,&[]).is_err());
        req.max_loss_points=0.0;
        req.device="pooled-40gb".into();
        assert!(plan_optimize_workflow(&req,&[]).is_err());
        req.device="cpu".into();
        req.candidates=vec![("same".into(),PathBuf::from("a")),
            ("same".into(),PathBuf::from("b"))];
        assert!(plan_optimize_workflow(&req,&[]).is_err());
    }
    #[test]
    fn stages_do_not_claim_gpu_or_quality(){
        let s=stage("run","Quantize","awaiting_approval","requires consent",None);
        assert_eq!(s["completed"],false);
        assert_eq!(s["quality_verified"],false);
    }
}
