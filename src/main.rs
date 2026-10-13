//! Ojon backend CLI. The eventual Kathamo GUI imports ojon_backend directly.
use ojon_backend::{
    audit_dataset_file, discover_advanced, discover_backends, discovered_path, evaluate_sweep,
    execute_sweep, hf_python_path, inspect_gguf, inspect_hf_folder, parse_precisions,
    parse_quality_report, plan_gsq, plan_gsq_assemble, plan_rco_assemble, plan_rco_database,
    plan_rco_search, prepare_distill_plan, prepare_gguf, prepare_hf, prepare_sweep, prepare_train,
    probe_model, quantize_gguf, quantize_hf, recommend, retrieve_recipes, run_advanced, run_train,
    smoke_gguf, BackendId, GgufSummary, Integration, PplSettings, Precision, QuantizePlan,
    plan_diffusion, run_diffusion, inspect_diffusion_recipe, discover_diffusion,
    DiffusionEngine, DiffusionKind, DiffusionFormat,
    plan_pack, run_pack, unpack, prepare_imatrix, run_imatrix,
    prepare_autoround_plan, run_autoround,
    prepare_research, ResearchKind,
    resource_plan, ResourceMode, prepare_bench, run_bench, compare_bench_files,
    prepare_exl3, run_exl3, prepare_eval, run_eval,
};
use std::env;
use std::ffi::OsString;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::str::FromStr;

fn usage() {
    eprintln!(r#"Ojon (ওজন) — native quantization backend (early development)

Commands:
  ojon inspect   <model.gguf>
  ojon verify    <model.gguf>
  ojon backends [--json]     Detect installed engines and show official download links
  ojon probe <model-path> [--json]   Identify model format/family without loading weights
  ojon optimize-plan <model-path> [--device auto|cpu|cuda|xpu] [--json]   Compare candidate routes without converting
  ojon resource-plan <model-path> <existing-output-parent> [--mode hqq|openvino|autoround|gguf|pack|exl3] [--device cpu|cuda|xpu|auto] [--json]
  ojon dataset-audit <dataset.jsonl> [--json]   Audit external dataset for drift, dead paths, staleness
  ojon distill-plan  <dataset.jsonl> <output-dir> [--min-score 0.5] [--json]  Curate SFT split
  ojon retrieve      <dataset.jsonl> <query> [--limit 5] [--min-score 0.4] [--json] [--markdown]
  ojon train-plan <config.yaml> [--tool <llamafactory-cli>] [--json]
  ojon train-run  <config.yaml> [--tool <llamafactory-cli>] --confirm
  ojon recommend <quality-evidence.json> [--json]  Smallest verified model within tolerance
  ojon task-eval-plan <suite.json> <new-report.json> --baseline <original.gguf> --candidate <label=variant.gguf> [--candidate ...] [--tool llama-cli] [--gpu-layers 0] [--device Vulkan2] [--tokens 128]
  ojon task-eval-run <suite.json> <new-report.json> --baseline <original.gguf> --candidate <label=variant.gguf> [options] --confirm
  ojon bench-plan <model.gguf> [--tool llama-bench] [--prompt-tokens 512] [--gen-tokens 128] [--repetitions 3] [--gpu-layers 0] [--device SYCL0|CUDA0|Vulkan0] [--json]
  ojon bench-run <model.gguf> <new-receipt.json> [same options] --confirm
  ojon bench-compare <reference-receipt.json> <candidate-receipt.json> [--json]
  ojon sweep-plan <source.gguf> <existing-output-dir> [--precisions Q5_K_M,Q4_K_M,IQ4_XS,IQ3_M] [--imatrix path] [--tool path] [--threads N]
  ojon sweep-run <source.gguf> <existing-output-dir> [options] --confirm
  ojon eval-sweep <sweep-manifest.json> <held-out-corpus.txt> [--tool llama-perplexity] [--gpu-layers N] [--context 2048] [--max-degradation 1.0] --confirm
  ojon imatrix-plan <original-f16.gguf> <calibration.txt> <new-imatrix.gguf> [--gpu-layers 0] [--context 2048] [--tool path]
  ojon imatrix-run  <original-f16.gguf> <calibration.txt> <new-imatrix.gguf> [options] --confirm
  ojon diffusion-plan <local-source> <output.safetensors> --checkout <upstream> [--backend svdq] [--kind image|video] [--format int4] [--python python] [--samples 32]
  ojon pack-plan <original.safetensors> <output.ojonq> [--bits 4] [--group 64]
  ojon pack-run  <original.safetensors> <output.ojonq> [options] --confirm
  ojon unpack    <packed.ojonq> <restored.safetensors> --confirm
  ojon exl3-plan <original-hf-dir> <new-output-dir> <new-work-dir> --checkout <exllamav3> [--bits 3.5] [--hq] [--python python] [--json]
  ojon exl3-run <original-hf-dir> <new-output-dir> <work-dir> --checkout <exllamav3> [--bits 3.5] [--hq] [--python python] [--resume] --confirm [--allow-resource-risk]
  ojon research-plan <hqq|openvino> <original-hf-dir> <new-output-dir> [--bits 4] [--group 64|128] [--ratio 0.8|1.0] [--device cpu|cuda|xpu] [--tool python|optimum-cli] [--json]
  ojon research-run  <hqq|openvino> <original-hf-dir> <new-output-dir> [options] --confirm [--allow-resource-risk]
  ojon autoround-plan <model-dir> <output-dir> [--bits 4] [--format gguf:q4_k_m|gguf:q5_k_m] [--iters 0|200] [--dataset local.json] [--device xpu|cuda|cpu] [--python python] [--nsamples 128] [--seqlen 2048] [--json]
  ojon autoround-run  <model-dir> <output-dir> [options] --confirm [--allow-resource-risk]
  ojon diffusion-inspect <local-source> <output.safetensors> [options] --confirm
  ojon diffusion-run <local-source> <output.safetensors> [options] --confirm
  ojon gsq-plan <checkout> <config.yaml> [--python python] [--layers 1 | --all-layers]
  ojon gsq-run <checkout> <config.yaml> [options] --confirm
  ojon gsq-assemble-plan <checkout> <config.yaml> <new-output-dir> [--python python]
  ojon gsq-assemble <checkout> <config.yaml> <new-output-dir> --confirm
  ojon rco-db-plan <checkout> <local-qwen3-model-dir> <new-db-dir> [--tokens 16384] [--python python]
  ojon rco-db-run <checkout> <local-qwen3-model-dir> <new-db-dir> [options] --confirm
  ojon rco-plan <checkout> <local-model-dir> <layer-db-dir> <target-bits> [--steps 30] [--python python]
  ojon rco-run <checkout> <local-model-dir> <layer-db-dir> <target-bits> [options] --confirm
  ojon rco-assemble-plan <checkout> <local-model-dir> <layer-db-dir> <assignment.json> <new-output-dir>
  ojon rco-assemble <checkout> <local-model-dir> <layer-db-dir> <assignment.json> <new-output-dir> --confirm
  ojon inspect-hf <model-folder>
  ojon quantize-hf <model-folder> <output.gguf> [options]
  ojon smoke     <model.gguf> [--tool <llama-cli>] [--tokens 16] [--gpu-layers 0] [--prompt text]
  ojon plan      <input.gguf> <output.gguf> [options]
  ojon quantize  <input.gguf> <output.gguf> [options]

Quantization options:
  --tool <path>          Installed llama-quantize executable (default: llama-quantize in PATH)
  --precision <type>     Q5_K_M (default), Q4_K_M, Q3_K_M, Q6_K,
                         IQ4_XS, IQ3_M, IQ3_S, IQ3_XS, IQ2_M, IQ2_S, IQ2_XS, IQ2_XXS
  --threads <N>          CPU thread count (default: system logical parallelism)
  --imatrix <file>       Optional llama-imatrix calibration output (GGUF plan/quantize)

Hugging Face conversion options (quantize-hf only):
  --converter <path>      Installed upstream convert_hf_to_gguf.py
  --python <path>         Python executable with llama.cpp converter dependencies
  --quantizer <path>      Installed llama-quantize binary

Important:
  - Direct GGUF quantize takes a GGUF source; quantize-hf takes a local HF checkpoint folder.
  - HF source needs config.json, model weights, llama.cpp converter script, Python dependencies,
    and substantial scratch space for an intermediate GGUF. No installation is automated.
  - Only models supported by your installed llama.cpp quantizer can be converted.
  - 'verify' checks GGUF metadata and catalog; it does NOT run inference.
  - Existing outputs are never overwritten; no backend is installed automatically.
  - 'smoke' optionally runs real llama-cli inference; exit status is NOT a quality assessment.
  - The native Kathamo interface is in the optional ojon-gui binary.
  - Only externally configured LLaMA-Factory LoRA/QLoRA SFT is currently wired.
  - Teacher/student distillation and automatic data generation are NOT implemented.
  - Recommendation reads independently measured task metrics. It does not fabricate quality.
  - Agent probe/plan commands are read-only. Quantization sweeps and training need explicit --confirm.
  - Sweep conversion DOES NOT evaluate quality; it writes real results for later baseline comparison.
  - GSQ and RCO require user-installed trusted public Python checkouts; no local GPU compatibility is claimed.
  - Diffusion SVDQ INT4 uses a separate public diffuse-compressor CUDA checkout.
    Built-in Ojon pack-plan/pack-run/unpack use native Rust, with no RebelUI or Python.
    Ojon .ojonq output requires unpacking or a future Ojon-aware GPU loader.
    Conversion and inspection require explicit --confirm. No image/video quality is implied.
  - RCO searches an EXISTING upstream per-bit GPTQ database; it does not create that database yet.
  - GSQ/RCO assembly must be explicitly requested. Their artifacts are NOT guaranteed GGUF or load-tested.
"#);
}
fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message.into())
}
fn next_argument(iter: &mut impl Iterator<Item = OsString>, label: &str) -> io::Result<OsString> {
    iter.next().ok_or_else(|| invalid(format!("missing {label}")))
}
fn show_gguf(path: &Path, info: &GgufSummary) {
    println!("path: {}", path.display());
    println!("format: GGUF v{} ({})", info.version, if info.big_endian { "big-endian" } else { "little-endian" });
    println!("model: {}", info.model_name.as_deref().unwrap_or("(not specified)"));
    println!("architecture: {}", info.architecture.as_deref().unwrap_or("(not specified)"));
    println!("file_bytes: {}", info.file_bytes);
    println!("tensors: {}", info.tensor_count);
    println!("metadata_entries: {}", info.metadata_count);
    println!("tensor_data_offset: {}", info.data_offset);
    println!("alignment: {}", info.alignment);
    println!("general.file_type: {}", info.file_type.map(|n| n.to_string()).unwrap_or_else(|| "unknown".into()));
}
fn show_plan(plan: &QuantizePlan) {
    println!("adapter: llama.cpp-quantize (external executable)");
    println!("source: {}", plan.source.display());
    println!("source_bytes: {}", plan.source_info.file_bytes);
    println!("source_architecture: {}", plan.source_info.architecture.as_deref().unwrap_or("unknown; backend determines compatibility"));
    println!("output: {}", plan.output.display());
    println!("precision: {}", plan.precision);
    println!("threads: {}", plan.threads);
    println!("importance_matrix: {}", plan.imatrix.as_ref().map(|p|p.display().to_string()).unwrap_or_else(||"none".into()));
    println!("tool: {}", plan.executable.display());
    println!("peak_ram: UNKNOWN (depends on backend/model)");
    println!("peak_vram: UNKNOWN (typically CPU quantization, backend dependent)");
    println!("scratch_storage: UNKNOWN (output is staged alongside destination)");
    println!("compatibility: NOT CONFIRMED until the configured converter runs");
    println!("runtime_load: NOT TESTED");
}
fn parse_plan(mut args: impl Iterator<Item = OsString>) -> io::Result<QuantizePlan> {
    let source = PathBuf::from(next_argument(&mut args, "input GGUF file")?);
    let output = PathBuf::from(next_argument(&mut args, "output GGUF file")?);
    let mut precision = Precision::default();
    let mut matrix: Option<PathBuf> = None;
    let mut tool = ojon_backend::discovered_path(BackendId::LlamaQuantize)
        .unwrap_or_else(|| PathBuf::from("llama-quantize"));
    let mut threads = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4);
    while let Some(option) = args.next() {
        match option.to_str() {
            Some("--precision") => {
                let value = next_argument(&mut args, "precision value")?;
                let s = value.to_str().ok_or_else(|| invalid("precision must be text"))?;
                precision = Precision::from_str(s).map_err(invalid)?;
            }
            Some("--tool") => tool = PathBuf::from(next_argument(&mut args, "quantizer executable path")?),
            Some("--imatrix") => matrix = Some(PathBuf::from(next_argument(&mut args, "importance matrix file")?)),
            Some("--threads") => {
                let value = next_argument(&mut args, "thread count")?;
                let s = value.to_str().ok_or_else(|| invalid("thread count must be text"))?;
                threads = s.parse::<usize>().map_err(|_| invalid("thread count must be a positive integer"))?;
            }
            _ => return Err(invalid(format!("unknown option: {}", option.to_string_lossy()))),
        }
    }
    let plan = prepare_gguf(&source, &output, &tool, precision, threads)?;
    match matrix { Some(path) => plan.with_imatrix(&path), None => Ok(plan) }
}
fn run() -> io::Result<()> {
    let mut args = env::args_os().skip(1);
    let cmd = args.next().unwrap_or_else(|| OsString::from("--help"));
    match cmd.to_str() {
        Some("bench-plan") | Some("bench-run") => {
            let execute=cmd=="bench-run";
            let model=PathBuf::from(next_argument(&mut args,"GGUF model for performance measurement")?);
            let receipt=if execute {
                Some(PathBuf::from(next_argument(&mut args,"new JSON receipt path")?))
            } else { None };
            let mut tool=discovered_path(BackendId::LlamaBench)
                .unwrap_or_else(||PathBuf::from("llama-bench"));
            let (mut prompt_tokens,mut gen_tokens,mut reps,mut gpu_layers)=(512u32,128u32,3u32,0i32);
            let mut confirmed=false;
            let mut device:Option<String>=None;
            while let Some(arg)=args.next() {
                match arg.to_str() {
                    Some("--tool")=>tool=PathBuf::from(next_argument(&mut args,"llama-bench executable")?),
                    Some("--prompt-tokens")=>prompt_tokens=next_argument(&mut args,"prompt token count")?
                        .to_str().ok_or_else(||invalid("prompt count must be text"))?
                        .parse().map_err(|_|invalid("prompt tokens require integer"))?,
                    Some("--gen-tokens")=>gen_tokens=next_argument(&mut args,"generation token count")?
                        .to_str().ok_or_else(||invalid("generation count must be text"))?
                        .parse().map_err(|_|invalid("generation tokens require integer"))?,
                    Some("--repetitions")=>reps=next_argument(&mut args,"repetitions")?
                        .to_str().ok_or_else(||invalid("repetitions must be text"))?
                        .parse().map_err(|_|invalid("repetitions require integer"))?,
                    Some("--gpu-layers")=>gpu_layers=next_argument(&mut args,"GPU layer count")?
                        .to_str().ok_or_else(||invalid("GPU layers must be text"))?
                        .parse().map_err(|_|invalid("GPU layers require integer"))?,
                    Some("--device")=>device=Some(next_argument(&mut args,"explicit llama-bench device")?
                        .to_str().ok_or_else(||invalid("device must be text"))?.to_owned()),
                    Some("--confirm") if execute=>confirmed=true,
                    Some("--json")=>{},
                    _=>return Err(invalid(format!("unknown benchmark option: {}",arg.to_string_lossy()))),
                }
            }
            let plan=prepare_bench(&model,&tool,prompt_tokens,gen_tokens,reps,gpu_layers,device.as_deref())?;
            if !execute {println!("{}",serde_json::to_string_pretty(&plan.as_json()).map_err(io::Error::other)?);}
            else {
                if !confirmed {return Err(invalid("real llama-bench execution loads weights; add --confirm after reviewing model, installed tool and available RAM/VRAM"));}
                let path=receipt.expect("bench-run supplies a receipt");
                let record=run_bench(&plan,&path)?;
                println!("{}",serde_json::to_string_pretty(&serde_json::json!({
                    "saved_receipt": path.to_string_lossy(),
                    "benchmark":record
                })).map_err(io::Error::other)?);
            }
        }
        Some("bench-compare") => {
            let reference=PathBuf::from(next_argument(&mut args,"baseline receipt JSON")?);
            let candidate=PathBuf::from(next_argument(&mut args,"candidate receipt JSON")?);
            for arg in args {
                if arg != "--json" {return Err(invalid(format!("unknown benchmark comparison option: {}",arg.to_string_lossy())));}
            }
            let value=compare_bench_files(&reference,&candidate)?;
            println!("{}",serde_json::to_string_pretty(&value).map_err(io::Error::other)?);
        }
        Some("gsq-plan") | Some("gsq-run") => {
            let execute=cmd=="gsq-run";
            let checkout=PathBuf::from(next_argument(&mut args,"public GSQ repository checkout")?);
            let config=PathBuf::from(next_argument(&mut args,"GSQ local YAML configuration")?);
            let mut python=PathBuf::from("python");
            let mut max_layers=Some(1u32);
            let mut confirm=false;
            while let Some(arg)=args.next(){
                match arg.to_str(){
                    Some("--python")=>python=PathBuf::from(next_argument(&mut args,"Python executable")?),
                    Some("--layers")=>max_layers=Some(next_argument(&mut args,"GSQ max layers")?
                        .to_str().ok_or_else(||invalid("layers require integer"))?
                        .parse().map_err(|_|invalid("invalid GSQ layer count"))?),
                    Some("--all-layers")=>max_layers=None,
                    Some("--confirm") if execute=>confirm=true,
                    _=>return Err(invalid(format!("unknown GSQ option: {}",arg.to_string_lossy()))),
                }
            }
            let plan=plan_gsq(&checkout,&config,&python,max_layers)?;
            if !execute {println!("{}",plan.json());}
            else {
                if !confirm{return Err(invalid("GSQ execution requires --confirm after reviewing source, dataset, VRAM and output settings"));}
                println!("{}",plan.json());
                run_advanced(&plan,|line|eprintln!("{line}"))?;
                println!("gsq_stage: external_process_exited; model_quality: NOT_TESTED");
            }
        }
        Some("gsq-assemble-plan") | Some("gsq-assemble") => {
            let execute=cmd=="gsq-assemble";
            let checkout=PathBuf::from(next_argument(&mut args,"public GSQ checkout")?);
            let config=PathBuf::from(next_argument(&mut args,"GSQ YAML config")?);
            let output=PathBuf::from(next_argument(&mut args,"new output directory")?);
            let mut python=PathBuf::from("python");
            let mut confirm=false;
            while let Some(arg)=args.next(){
                match arg.to_str(){
                    Some("--python")=>python=PathBuf::from(next_argument(&mut args,"Python")?),
                    Some("--confirm") if execute=>confirm=true,
                    _=>return Err(invalid(format!("unknown GSQ assembly option: {}",arg.to_string_lossy()))),
                }
            }
            let plan=plan_gsq_assemble(&checkout,&config,&output,&python)?;
            if !execute{println!("{}",plan.json());}
            else {
                if !confirm{return Err(invalid("assembly writes a model checkpoint; pass --confirm"));}
                println!("{}",plan.json());
                run_advanced(&plan,|line|eprintln!("{line}"))?;
                println!("gsq_output: {}",output.display());
                println!("loadability_and_quality: NOT_VERIFIED");
            }
        }
        Some("rco-db-plan") | Some("rco-db-run") => {
            let execute=cmd=="rco-db-run";
            let checkout=PathBuf::from(next_argument(&mut args,"public RCO checkout")?);
            let model=PathBuf::from(next_argument(&mut args,"original local dense Qwen3 model")?);
            let output=PathBuf::from(next_argument(&mut args,"new RCO database root directory")?);
            let mut python=PathBuf::from("python");
            let mut tokens=16384u32;
            let mut confirm=false;
            while let Some(arg)=args.next(){
                match arg.to_str(){
                    Some("--python")=>python=PathBuf::from(next_argument(&mut args,"Python executable")?),
                    Some("--tokens")=>tokens=next_argument(&mut args,"calibration token count")?
                        .to_str().ok_or_else(||invalid("tokens must be integer"))?
                        .parse().map_err(|_|invalid("invalid calibration token count"))?,
                    Some("--confirm") if execute=>confirm=true,
                    _=>return Err(invalid(format!("unknown RCO database option: {}",arg.to_string_lossy()))),
                }
            }
            let plan=plan_rco_database(&checkout,&model,&output,&python,tokens)?;
            if !execute{println!("{}",plan.json());}
            else{
                if !confirm{return Err(invalid("RCO database construction uses CUDA and substantial resources; pass --confirm after verifying the plan"));}
                println!("{}",plan.json());
                run_advanced(&plan,|line|eprintln!("{line}"))?;
                println!("rco_database: external_backend_exited; database_contents: REVIEW_BEFORE_SEARCH");
            }
        }
        Some("rco-plan") | Some("rco-run") => {
            let execute=cmd=="rco-run";
            let checkout=PathBuf::from(next_argument(&mut args,"public RCO checkout")?);
            let model=PathBuf::from(next_argument(&mut args,"original local HF model folder")?);
            let database=PathBuf::from(next_argument(&mut args,"prebuilt per-bit layer database")?);
            let bits=next_argument(&mut args,"target average bits")?
                .to_str().ok_or_else(||invalid("target bits must be a number"))?
                .parse::<f64>().map_err(|_|invalid("target bits must be numeric"))?;
            let mut python=PathBuf::from("python");
            let mut steps=30u32;
            let mut confirm=false;
            while let Some(arg)=args.next(){
                match arg.to_str(){
                    Some("--python")=>python=PathBuf::from(next_argument(&mut args,"Python executable")?),
                    Some("--steps")=>steps=next_argument(&mut args,"optimization steps")?
                        .to_str().ok_or_else(||invalid("steps require an integer"))?
                        .parse().map_err(|_|invalid("RCO steps must be numeric"))?,
                    Some("--confirm") if execute=>confirm=true,
                    _=>return Err(invalid(format!("unknown RCO option: {}",arg.to_string_lossy()))),
                }
            }
            let plan=plan_rco_search(&checkout,&model,&database,bits,&python,steps)?;
            if !execute{println!("{}",plan.json());}
            else {
                if !confirm{return Err(invalid("RCO search uses local compute/disk; pass --confirm"));}
                println!("{}",plan.json());
                run_advanced(&plan,|line|eprintln!("{line}"))?;
                println!("rco_search: completed_external_process; output_assignment: inspect_layer_database");
                println!("model_checkpoint_and_quality: NOT_VERIFIED");
            }
        }
        Some("rco-assemble-plan") | Some("rco-assemble") => {
            let execute=cmd=="rco-assemble";
            let checkout=PathBuf::from(next_argument(&mut args,"public RCO checkout")?);
            let model=PathBuf::from(next_argument(&mut args,"original local HF model folder")?);
            let database=PathBuf::from(next_argument(&mut args,"prebuilt layer database")?);
            let assignment=PathBuf::from(next_argument(&mut args,"RCO search assignment JSON/TXT")?);
            let output=PathBuf::from(next_argument(&mut args,"new output directory")?);
            let mut python=PathBuf::from("python");
            let mut confirm=false;
            while let Some(arg)=args.next(){
                match arg.to_str(){
                    Some("--python")=>python=PathBuf::from(next_argument(&mut args,"Python executable")?),
                    Some("--confirm") if execute=>confirm=true,
                    _=>return Err(invalid(format!("unknown RCO assembly option: {}",arg.to_string_lossy()))),
                }
            }
            let plan=plan_rco_assemble(&checkout,&model,&database,&assignment,&output,&python)?;
            if !execute{println!("{}",plan.json());}
            else {
                if !confirm{return Err(invalid("assembly writes compressed-tensors checkpoint; pass --confirm"));}
                println!("{}",plan.json());
                run_advanced(&plan,|line|eprintln!("{line}"))?;
                println!("rco_output: {}",output.display());
                println!("format: HF_COMPRESSED_TENSORS; GGUF_NOT_IMPLIED; quality: NOT_TESTED");
            }
        }
        Some("pack-plan") | Some("pack-run") => {
            let execute=cmd=="pack-run";
            let source=PathBuf::from(next_argument(&mut args,"original full-precision safetensors")?);
            let output=PathBuf::from(next_argument(&mut args,"new Ojon .ojonq model file")?);
            let mut bits=4u8;let mut group=64usize;let mut confirmed=false;
            while let Some(option)=args.next(){
                match option.to_str(){
                    Some("--bits")=>{
                        bits=next_argument(&mut args,"2,3,4 bit precision")?.to_str()
                            .ok_or_else(||invalid("bits must be text"))?.parse()
                            .map_err(|_|invalid("invalid bits"))?;
                    }
                    Some("--group")=>{
                        group=next_argument(&mut args,"32,64,128,256 group size")?.to_str()
                            .ok_or_else(||invalid("group must be text"))?.parse()
                            .map_err(|_|invalid("invalid group size"))?;
                    }
                    Some("--confirm") if execute=>confirmed=true,
                    _=>return Err(invalid(format!("unknown native pack option: {}",option.to_string_lossy())))
                }
            }
            let plan=plan_pack(&source,&output,bits,group)?;
            if !execute{println!("{}",plan.json());}
            else {
                if !confirmed{return Err(invalid("native pack-run writes a model file; pass --confirm"));}
                let report=run_pack(&plan,|line|eprintln!("{line}"))?;
                println!("{}",serde_json::json!({
                    "output":report.output.to_string_lossy(),
                    "original_bytes":report.original_bytes,
                    "packed_bytes":report.output_bytes,
                    "quantized_tensors":report.packed_tensors,
                    "quantized_elements":report.elements,
                    "quantization_mse":report.mse,
                    "gpu_loader":"NOT_IMPLEMENTED",
                    "task_quality":"NOT_EVALUATED"
                }));
            }
        }
        Some("unpack") => {
            let src=PathBuf::from(next_argument(&mut args,"Ojon .ojonq packed model")?);
            let out=PathBuf::from(next_argument(&mut args,"new ordinary .safetensors path")?);
            let confirm=next_argument(&mut args,"--confirm")?;
            if confirm!="--confirm"||args.next().is_some(){
                return Err(invalid("unpack writes a restored checkpoint: --confirm is required"));
            }
            let saved=unpack(&src,&out)?;
            println!("restored_safetensors: {}",saved.display());
            println!("warning: model weights are lossily reconstructed; no full task-quality claim");
        }
        Some("diffusion-plan") | Some("diffusion-inspect") | Some("diffusion-run") => {
            let stage = cmd.to_string_lossy().to_string();
            let source=PathBuf::from(next_argument(&mut args,"local source model or Diffusers pipeline")?);
            let output=PathBuf::from(next_argument(&mut args,"new quantized safetensors output")?);
            let mut engine=DiffusionEngine::Svdq;
            let mut kind=DiffusionKind::Image;
            let mut precision=DiffusionFormat::Int4;
            let mut checkout:Option<PathBuf>=None;
            let mut python=PathBuf::from("python");
            let mut samples=32u32;
            let mut confirm=false;
            while let Some(option)=args.next(){
                match option.to_str(){
                    Some("--checkout")=>checkout=Some(PathBuf::from(next_argument(&mut args,"upstream Python repository checkout")?)),
                    Some("--python")=>python=PathBuf::from(next_argument(&mut args,"Python interpreter")?),
                    Some("--backend")=>{
                        let name=next_argument(&mut args,"backend svdq")?;
                        engine=DiffusionEngine::from_name(name.to_str().ok_or_else(||invalid("invalid backend name"))?)?;
                    }
                    Some("--kind")=>{
                        let name=next_argument(&mut args,"image|video")?;
                        kind=DiffusionKind::from_name(name.to_str().ok_or_else(||invalid("invalid model kind"))?)?;
                    }
                    Some("--format")=>{
                        let name=next_argument(&mut args,"int4")?;
                        precision=DiffusionFormat::from_name(name.to_str().ok_or_else(||invalid("invalid format"))?)?;
                    }
                    Some("--samples")=>{
                        samples=next_argument(&mut args,"calibration sample count")?
                            .to_str().ok_or_else(||invalid("samples must be a number"))?
                            .parse().map_err(|_|invalid("samples must be an integer"))?;
                    }
                    Some("--confirm") if stage!="diffusion-plan"=>confirm=true,
                    _=>return Err(invalid(format!("unknown diffusion option: {}",option.to_string_lossy()))),
                }
            }
            let checkout=checkout.or_else(||std::env::var_os("OJON_SVDQ_DIR")
                .map(PathBuf::from)).ok_or_else(||invalid(
                "set --checkout to diffuse-compressor or OJON_SVDQ_DIR"
            ))?;
            let plan=plan_diffusion(&source,&output,&checkout,&python,engine,kind,precision,samples)?;
            if stage=="diffusion-plan" {
                println!("{}",plan.json());
            } else {
                if !confirm {
                    return Err(invalid("diffusion inspection/conversion may load large models and run third-party Python; explicitly pass --confirm"));
                }
                println!("{}",plan.json());
                if stage=="diffusion-inspect" {
                    inspect_diffusion_recipe(&plan,|line|eprintln!("{line}"))?;
                    println!("diffusion_recipe: INSPECTED_BY_UPSTREAM; model_conversion: NOT_STARTED");
                } else {
                    let outcome=run_diffusion(&plan,|line|eprintln!("{line}"))?;
                    println!("output: {}",outcome.output.display());
                    println!("output_bytes: {}",outcome.output_bytes);
                    println!("conversion_seconds: {:.2}",outcome.elapsed_seconds);
                    println!("runtime_load: NOT_TESTED");
                    println!("image_video_quality: NOT_EVALUATED");
                    if let Some(config)=outcome.sidecar{println!("config: {}",config.display());}
                    if let Some(receipt)=outcome.receipt{println!("receipt: {}",receipt.display());}
                }
            }
        }
        Some("imatrix-plan") | Some("imatrix-run") => {
            let execute = cmd=="imatrix-run";
            let model=PathBuf::from(next_argument(&mut args,"original F16/BF16 GGUF model")?);
            let corpus=PathBuf::from(next_argument(&mut args,"local UTF-8 calibration text")?);
            let output=PathBuf::from(next_argument(&mut args,"new imatrix.gguf output")?);
            let mut tool=discovered_path(BackendId::LlamaImatrix)
                .unwrap_or_else(||PathBuf::from("llama-imatrix"));
            let mut gpu_layers=0u32;
            let mut context=2048u32;
            let mut confirmed=false;
            while let Some(option)=args.next(){
                match option.to_str(){
                    Some("--tool")=>tool=PathBuf::from(next_argument(&mut args,"public llama-imatrix executable")?),
                    Some("--gpu-layers")=>{
                        gpu_layers=next_argument(&mut args,"GPU layer offload count")?
                            .to_str().ok_or_else(||invalid("GPU layer count must be numeric"))?
                            .parse().map_err(|_|invalid("GPU layers must be integer"))?;
                    }
                    Some("--context")=>{
                        context=next_argument(&mut args,"calibration context")?
                            .to_str().ok_or_else(||invalid("context must be numeric"))?
                            .parse().map_err(|_|invalid("context must be integer"))?;
                    }
                    Some("--confirm") if execute=>confirmed=true,
                    _=>return Err(invalid(format!("unknown imatrix option: {}",option.to_string_lossy()))),
                }
            }
            let plan=prepare_imatrix(&model,&corpus,&output,&tool,gpu_layers,context)?;
            if !execute{println!("{}",plan.json());}
            else{
                if !confirmed{
                    return Err(invalid("importance-matrix calibration loads the original model and can consume significant RAM/VRAM; pass --confirm"));
                }
                println!("{}",plan.json());
                let saved=run_imatrix(&plan,|line|eprintln!("{line}"))?;
                println!("imatrix: {}",saved.display());
                println!("next_step: use --imatrix with ojon sweep-plan / sweep-run");
                println!("quality: NOT_EVALUATED; held_out_task_tests_required");
            }
        }
        Some("eval-sweep") => {
            let manifest = PathBuf::from(next_argument(&mut args, "completed Ojon sweep manifest")?);
            let corpus = PathBuf::from(next_argument(&mut args, "held-out evaluation text corpus")?);
            let mut settings = PplSettings {
                tool: discovered_path(BackendId::LlamaPerplexity)
                    .unwrap_or_else(|| PathBuf::from("llama-perplexity")),
                corpus,
                gpu_layers: 0,
                context: 2048,
                max_relative_degradation_pct: 1.0,
            };
            let mut confirmed = false;
            while let Some(arg) = args.next() {
                match arg.to_str() {
                    Some("--tool") => settings.tool = PathBuf::from(next_argument(&mut args,"perplexity executable")?),
                    Some("--gpu-layers") => {
                        settings.gpu_layers = next_argument(&mut args, "GPU offload layer count")?
                            .to_str().ok_or_else(|| invalid("GPU layers must be UTF-8 digits"))?
                            .parse().map_err(|_| invalid("GPU layers must be an integer"))?;
                    }
                    Some("--context") => {
                        settings.context = next_argument(&mut args, "context size")?
                            .to_str().ok_or_else(|| invalid("context must be UTF-8 digits"))?
                            .parse().map_err(|_| invalid("context size must be an integer"))?;
                    }
                    Some("--max-degradation") => {
                        settings.max_relative_degradation_pct=next_argument(&mut args, "max relative PPL increase")?
                            .to_str().ok_or_else(|| invalid("degradation must be text"))?
                            .parse().map_err(|_| invalid("degradation must be numeric"))?;
                    }
                    Some("--confirm") => confirmed = true,
                    _ => return Err(invalid(format!("unknown eval-sweep option: {}",arg.to_string_lossy()))),
                }
            }
            if !confirmed {
                return Err(invalid("eval-sweep loads all sweep variants sequentially and may use substantial RAM/VRAM; pass --confirm"));
            }
            settings.validate()?;
            let result = evaluate_sweep(&manifest, &settings, |line| eprintln!("{line}"))?;
            println!("perplexity_report: {}",result.display());
            println!("overall_task_quality_status: NOT_EVALUATED_BY_THIS_COMMAND");
            println!("recommendation: PPL_ONLY_PROVISIONAL");
        }
        Some("sweep-plan") | Some("sweep-run") => {
            let execute = cmd == "sweep-run";
            let source = PathBuf::from(next_argument(&mut args, "original F16/BF16 GGUF model")?);
            let destination = PathBuf::from(next_argument(&mut args, "existing output directory")?);
            let mut formats = String::from("Q5_K_M,Q4_K_M,IQ4_XS,IQ3_M");
            let mut imatrix: Option<PathBuf> = None;
            let mut tool = discovered_path(BackendId::LlamaQuantize)
                .unwrap_or_else(|| PathBuf::from("llama-quantize"));
            let mut threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4);
            let mut confirm = false;
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--precisions") => {
                        let v = next_argument(&mut args, "comma-separated precision list")?;
                        formats = v.to_str().ok_or_else(|| invalid("precisions require UTF-8"))?.to_owned();
                    }
                    Some("--imatrix") => {
                        imatrix = Some(PathBuf::from(next_argument(&mut args, "importance matrix file")?));
                    }
                    Some("--tool") => tool = PathBuf::from(next_argument(&mut args, "converter executable")?),
                    Some("--threads") => {
                        let value = next_argument(&mut args, "thread count")?;
                        threads = value.to_str().ok_or_else(|| invalid("invalid thread count"))?
                            .parse::<usize>().map_err(|_| invalid("thread count must be a number"))?;
                    }
                    Some("--confirm") if execute => confirm = true,
                    _ => return Err(invalid(format!("unknown sweep option: {}", option.to_string_lossy()))),
                }
            }
            let types = parse_precisions(&formats)?;
            let plan = prepare_sweep(
                &source, &destination, &tool, &types, imatrix.as_deref(), threads
            )?;
            if !execute {
                println!("{}", plan.json());
            } else {
                if !confirm {
                    return Err(invalid("sweep-run executes multiple model conversions and uses significant RAM/disk; pass --confirm to proceed"));
                }
                eprintln!("Ojon: executing {} sequential model conversions. Original source is read-only.", types.len());
                let path = execute_sweep(&plan, |line| eprintln!("{line}"))?;
                println!("sweep_manifest: {}",path.display());
                println!("quantization_task_quality: NOT_EVALUATED");
                println!("recommended_variant: NONE_UNTIL_REAL_BENCHMARKS");
            }
        }
        Some("task-eval-plan") | Some("task-eval-run") => {
            let execute=cmd=="task-eval-run";
            let suite=PathBuf::from(next_argument(&mut args,"held-out task suite JSON")?);
            let output=PathBuf::from(next_argument(&mut args,"new quality evidence JSON")?);
            let mut baseline:Option<PathBuf>=None;
            let mut candidates:Vec<(String,PathBuf)>=Vec::new();
            let mut tool=PathBuf::from("llama-cli");
            let mut gpu_layers=0i32;
            let mut device:Option<String>=None;
            let mut tokens=128u32;
            let mut confirm=false;
            let mut json=false;
            while let Some(option)=args.next() {
                match option.to_str() {
                    Some("--baseline") => baseline=Some(PathBuf::from(next_argument(&mut args,"baseline GGUF")?)),
                    Some("--candidate") => {
                        let pair=next_argument(&mut args,"label=model.gguf")?;
                        let pair=pair.to_str().ok_or_else(||invalid("candidate must be UTF-8"))?;
                        let (name,path)=pair.split_once('=').ok_or_else(||invalid("candidate must be label=path"))?;
                        candidates.push((name.to_owned(),PathBuf::from(path)));
                    }
                    Some("--tool") => tool=PathBuf::from(next_argument(&mut args,"llama-cli executable")?),
                    Some("--gpu-layers") => gpu_layers=next_argument(&mut args,"GPU layer count")?.to_str()
                        .ok_or_else(||invalid("invalid layers"))?.parse().map_err(|_|invalid("invalid layers"))?,
                    Some("--device") => device=Some(next_argument(&mut args,"Vulkan2 or SYCL0")?
                        .to_str().ok_or_else(||invalid("invalid device"))?.to_owned()),
                    Some("--tokens") => tokens=next_argument(&mut args,"token limit")?.to_str()
                        .ok_or_else(||invalid("invalid tokens"))?.parse().map_err(|_|invalid("invalid tokens"))?,
                    Some("--confirm") if execute => confirm=true,
                    Some("--json") => json=true,
                    _ => return Err(invalid(format!("unknown task evaluation option: {}",option.to_string_lossy()))),
                }
            }
            let baseline=baseline.ok_or_else(||invalid("--baseline is required"))?;
            let plan=prepare_eval(&suite,&baseline,&candidates,&tool,gpu_layers,device.as_deref(),tokens,&output)?;
            if execute {
                if !confirm{return Err(invalid("task evaluation loads real models; pass --confirm"));}
                let result=run_eval(&plan,|line|eprintln!("{line}"))?;
                if json {println!("{result}");}
                else {println!("task_evaluation_receipt: {}",plan.report.display());}
            } else {println!("{}",plan.as_json());}
        }
        Some("recommend") => {
            let file = PathBuf::from(next_argument(&mut args, "quality evidence JSON file")?);
            let json = match args.next() {
                None => false,
                Some(flag) if flag == "--json" => true,
                _ => return Err(invalid("recommend supports only optional --json")),
            };
            if args.next().is_some() { return Err(invalid("too many recommend options")); }
            let report = parse_quality_report(&file)?;
            let result = recommend(&report);
            if json { println!("{result}"); }
            else {
                println!("evaluation_suite: {}", report.evaluation_suite);
                if let Some(best) = result["recommendation"].as_str() {
                    println!("recommended_smallest_quality_preserving_variant: {best}");
                } else {
                    println!("recommendation: NONE; no measured candidate meets all requirements");
                }
                println!("evidence: USER_SUPPLIED_NOT_INDEPENDENTLY_VERIFIED");
                for candidate in result["candidates"].as_array().into_iter().flatten() {
                    println!("{}: {} bytes, eligible={}", candidate["label"].as_str().unwrap_or("?"),
                        candidate["bytes"], candidate["eligible"]);
                }
            }
        }
        Some("resource-plan") => {
            let source = PathBuf::from(next_argument(&mut args, "source model path")?);
            let output_parent = PathBuf::from(next_argument(&mut args, "existing destination parent directory")?);
            let mut mode = ResourceMode::Gguf;
            let mut device = String::from("cpu");
            let mut json = false;
            while let Some(arg) = args.next() {
                match arg.to_str() {
                    Some("--mode") => {
                        let value = next_argument(&mut args, "preflight mode")?;
                        mode = ResourceMode::parse(value.to_str().ok_or_else(|| invalid("mode must be text"))?)?;
                    }
                    Some("--device") => device = next_argument(&mut args, "cpu|cuda|xpu|auto")?
                        .to_str().ok_or_else(|| invalid("device must be text"))?.to_owned(),
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown resource-plan option: {}", arg.to_string_lossy()))),
                }
            }
            let result = resource_plan(&source, &output_parent, mode, &device)?;
            if json {
                println!("{result}");
            } else {
                println!("method: {}", result["mode"]);
                println!("source_bytes: {}", result["source_weight_bytes"]);
                println!("host_free_ram_bytes: {}", result["free_host_ram_bytes"]);
                println!("scratch_free_bytes: {}", result["free_output_volume_bytes"]);
                println!("host_safety_budget_bytes: {}", result["host_ram_safety_budget_bytes"]);
                println!("disk_safety_budget_bytes: {}", result["output_volume_safety_budget_bytes"]);
                println!("status: {}", result["status"]);
                println!("note: budget is a conservative policy floor, not converter peak-memory prediction");
                println!("note: Intel GPU free VRAM is measured only when installed xpu-smi returns a valid sample; otherwise UNKNOWN");
            }
        }
        Some("exl3-plan") | Some("exl3-run") => {
            let execute=cmd=="exl3-run";
            let source=PathBuf::from(next_argument(&mut args,"local unquantized HF model directory")?);
            let output=PathBuf::from(next_argument(&mut args,"new EXL3 output directory")?);
            let work_dir=PathBuf::from(next_argument(&mut args,"dedicated EXL3 work directory")?);
            let mut checkout:Option<PathBuf>=None;
            let mut python=PathBuf::from("python");
            let mut bits=3.5f64;
            let mut high_quality=false;
            let mut resume=false;
            let mut confirm=false;
            let mut allow_resource_risk=false;
            let mut json=false;
            while let Some(arg)=args.next() {
                match arg.to_str() {
                    Some("--checkout") => checkout=Some(PathBuf::from(next_argument(&mut args,"EXL3 checkout")?)),
                    Some("--python") => python=PathBuf::from(next_argument(&mut args,"Python interpreter")?),
                    Some("--bits") => bits=next_argument(&mut args,"target bitrate")?
                        .to_str().ok_or_else(||invalid("bitrate must be text"))?
                        .parse().map_err(|_|invalid("invalid EXL3 bitrate"))?,
                    Some("--hq") => high_quality=true,
                    Some("--resume") => resume=true,
                    Some("--confirm") if execute => confirm=true,
                    Some("--allow-resource-risk") if execute => allow_resource_risk=true,
                    Some("--json") => json=true,
                    _ => return Err(invalid(format!("unknown EXL3 option: {}",arg.to_string_lossy()))),
                }
            }
            let checkout=checkout.ok_or_else(||invalid("EXL3 --checkout is required"))?;
            let plan=prepare_exl3(&source,&output,&work_dir,&checkout,&python,bits,high_quality,resume)?;
            if !execute {
                if json { println!("{}",plan.as_json()); }
                else {
                    println!("backend: EXL3 / ExLlamaV3 (CUDA, not GGUF)");
                    println!("plan: {}",plan.as_json());
                }
            } else {
                if !confirm {return Err(invalid("EXL3 conversion writes model data; pass --confirm"));}
                println!("{}",plan.as_json());
                let result=run_exl3(&plan,allow_resource_risk)?;
                println!("{}",result);
            }
        }
        Some("research-plan") | Some("research-run") => {
            let execute = cmd == "research-run";
            let kind_str = next_argument(&mut args, "hqq or openvino backend")?;
            let kind = ResearchKind::parse(kind_str.to_str().ok_or_else(|| invalid("backend must be text"))?)?;
            let source = PathBuf::from(next_argument(&mut args, "local original HF checkpoint directory")?);
            let output = PathBuf::from(next_argument(&mut args, "new absolute output directory")?);
            let mut bits = 4u32;
            let mut group = if kind == ResearchKind::Hqq { 64u32 } else { 128u32 };
            let mut ratio = 1.0f64;
            let mut device = String::from("cpu");
            let mut tool = PathBuf::from(if kind == ResearchKind::Hqq { "python" } else { "optimum-cli" });
            let mut confirm = false;
            let mut allow_resource_risk = false;
            let mut json = false;
            while let Some(arg) = args.next() {
                match arg.to_str() {
                    Some("--bits") => bits = next_argument(&mut args, "bit width")?
                        .to_str().ok_or_else(|| invalid("bits must be text"))?
                        .parse().map_err(|_| invalid("invalid bits"))?,
                    Some("--group") => group = next_argument(&mut args, "group size")?
                        .to_str().ok_or_else(|| invalid("group must be text"))?
                        .parse().map_err(|_| invalid("invalid group size"))?,
                    Some("--ratio") => ratio = next_argument(&mut args, "OpenVINO 4-bit layer ratio")?
                        .to_str().ok_or_else(|| invalid("ratio must be text"))?
                        .parse().map_err(|_| invalid("invalid ratio"))?,
                    Some("--device") => device = next_argument(&mut args, "device")?
                        .to_str().ok_or_else(|| invalid("device must be text"))?.to_owned(),
                    Some("--tool") => tool = PathBuf::from(next_argument(&mut args, "backend executable")?),
                    Some("--json") => json = true,
                    Some("--confirm") if execute => confirm = true,
                    Some("--allow-resource-risk") if execute => allow_resource_risk = true,
                    _ => return Err(invalid(format!("unknown research option: {}", arg.to_string_lossy()))),
                }
            }
            let plan = prepare_research(kind, &source, &output, &tool, bits, group, ratio, &device)?;
            if !execute {
                if json { println!("{}", plan.as_json()); }
                else {
                    println!("backend: {}", plan.as_json()["backend"]);
                    println!("output_format: {}", plan.as_json()["output_format"]);
                    println!("source: {}", plan.source.directory.display());
                    println!("output: {}", plan.output.display());
                    println!("status: PLAN_ONLY; no model weights loaded");
                    println!("run: repeat the same options using research-run and --confirm");
                }
            } else {
                if !confirm { return Err(invalid("research conversion writes output; pass --confirm")); }
                println!("{}", plan.as_json());
                ojon_backend::research_routes::run_research_with_policy(&plan, allow_resource_risk)?;
                println!("research_status: EXPORTED_FORMAT_ONLY");
                println!("runtime_load_test: NOT_RUN");
                println!("held_out_task_quality: NOT_TESTED");
            }
        }
        Some("autoround-plan") | Some("autoround-run") => {
            let execute = cmd == "autoround-run";
            let model_dir = PathBuf::from(next_argument(&mut args, "local HF model directory")?);
            let output_dir = PathBuf::from(next_argument(&mut args, "new output directory")?);
            let mut bits = 4u32;
            let mut format: Option<String> = None;
            let mut iters = 0u32;
            let mut dataset: Option<PathBuf> = None;
            let mut device = String::from("xpu");
            let mut python = PathBuf::from("python");
            let mut nsamples = 128u32;
            let mut seqlen = 2048u32;
            let mut json = false;
            let mut confirm = false;
            let mut allow_resource_risk = false;
            while let Some(arg) = args.next() {
                match arg.to_str() {
                    Some("--bits") => {
                        bits = next_argument(&mut args, "bits (2, 3, 4, 5, 6, 8)")?
                            .to_str().ok_or_else(|| invalid("bits require integer"))?
                            .parse().map_err(|_| invalid("invalid bit width"))?;
                    }
                    Some("--format") => {
                        format = Some(next_argument(&mut args, "GGUF export format")?
                            .to_str().ok_or_else(|| invalid("format requires UTF-8"))?.to_owned());
                    }
                    Some("--iters") => {
                        iters = next_argument(&mut args, "AutoRound iterations")?
                            .to_str().ok_or_else(|| invalid("iterations require integer"))?
                            .parse().map_err(|_| invalid("invalid iterations"))?;
                    }
                    Some("--dataset") => {
                        dataset = Some(PathBuf::from(next_argument(&mut args, "local calibration JSON")?));
                    }
                    Some("--device") => {
                        device = next_argument(&mut args, "device (xpu, cuda, cpu)")?
                            .to_string_lossy().to_string();
                    }
                    Some("--python") => {
                        python = PathBuf::from(next_argument(&mut args, "Python executable")?);
                    }
                    Some("--nsamples") => {
                        nsamples = next_argument(&mut args, "calibration samples")?
                            .to_str().ok_or_else(|| invalid("nsamples require integer"))?
                            .parse().map_err(|_| invalid("invalid nsamples"))?;
                    }
                    Some("--seqlen") => {
                        seqlen = next_argument(&mut args, "sequence length")?
                            .to_str().ok_or_else(|| invalid("seqlen requires integer"))?
                            .parse().map_err(|_| invalid("invalid seqlen"))?;
                    }
                    Some("--json") => json = true,
                    Some("--confirm") if execute => confirm = true,
                    Some("--allow-resource-risk") if execute => allow_resource_risk = true,
                    _ => return Err(invalid(format!("unknown autoround option: {}", arg.to_string_lossy()))),
                }
            }
            let plan = ojon_backend::autoround::prepare_autoround_recipe_plan(
                &model_dir, &output_dir, bits, &device, &python, nsamples, seqlen,
                format.as_deref(), iters, dataset.as_deref()
            )?;
            if !execute {
                if json {
                    println!("{}", plan.json());
                } else {
                    println!("backend: intel_autoround");
                    println!("model_directory: {}", plan.model_dir.display());
                    println!("output_directory: {}", plan.output_dir.display());
                    println!("bits: {}", plan.bits);
                    println!("device: {}", plan.device.as_str());
                    println!("format: {}", plan.format);
                    println!("method: {}", if plan.iters == 0 { "optimized_rtn" } else { "calibrated_auto_round" });
                    println!("iterations: {}", plan.iters);
                    println!("dataset: {}", plan.dataset.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| "none (offline RTN)".into()));
                    println!("python: {}", plan.python.display());
                    println!("nsamples: {}", plan.nsamples);
                    println!("seqlen: {}", plan.seqlen);
                    println!("status: PLAN_ONLY");
                }
            } else {
                if !confirm {
                    return Err(invalid("autoround execution writes a quantized model; pass --confirm"));
                }
                println!("{}", plan.json());
                ojon_backend::autoround::run_autoround_with_policy(&plan, allow_resource_risk)?;
                println!("autoround_status: COMPLETED");
            }
        }
        Some("optimize-plan") => {
            let path = PathBuf::from(next_argument(&mut args, "local model file or directory")?);
            let mut device = String::from("auto");
            let mut json = false;
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--device") => {
                        device = next_argument(&mut args, "device")?
                            .to_str().ok_or_else(|| invalid("device must be UTF-8"))?.to_owned();
                    }
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown optimize-plan option: {}", option.to_string_lossy()))),
                }
            }
            let probe = probe_model(&path)?;
            let plan = ojon_backend::plan_options(&probe, &device, &discover_backends())?;
            if json {
                println!("{}", plan);
            } else {
                println!("source: {}", probe.path.display());
                println!("requested_device: {}", device);
                println!("mode: READ_ONLY / MULTIPLE ALTERNATIVES / NO WINNER");
                println!("actual_free_ram_vram_scratch: NOT_PROBED");
                for item in plan["options"].as_array().into_iter().flatten() {
                    println!("{} [{}] {} / {}",
                        item["id"].as_str().unwrap_or("?"),
                        item["state"].as_str().unwrap_or("?"),
                        item["backend"].as_str().unwrap_or("?"),
                        item["output_format"].as_str().unwrap_or("?"));
                    println!("  check: {}", item["reason"].as_str().unwrap_or("unknown"));
                }
                println!("warning: {}", plan["warning"].as_str().unwrap_or("not tested"));
            }
        }
        Some("probe") => {
            let path = PathBuf::from(next_argument(&mut args, "local model file or directory")?);
            let json = match args.next() {
                None => false,
                Some(value) if value == "--json" => true,
                _ => return Err(invalid("probe accepts only an optional --json")),
            };
            if args.next().is_some() { return Err(invalid("too many probe arguments")); }
            let report = probe_model(&path)?;
            if json {
                println!("{}", report.as_json());
            } else {
                println!("source: {}", report.path.display());
                println!("format: {}", report.format);
                println!("category: {} (confidence: {})", report.category.as_str(), report.confidence);
                println!("identification: {}", report.identification);
                println!("bytes: {}", report.bytes);
                println!("conversion_route: {}", report.conversion_route);
                println!("conversion_status: {}", report.conversion_status);
                for note in report.notes { println!("note: {note}"); }
            }
        }
        Some("train-plan") | Some("train-run") => {
            let run = cmd == "train-run";
            let config = PathBuf::from(next_argument(&mut args, "training YAML config")?);
            let mut trainer = discovered_path(BackendId::LlamaFactory)
                .unwrap_or_else(|| PathBuf::from("llamafactory-cli"));
            let mut json = false;
            let mut confirm = false;
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--tool") => trainer = PathBuf::from(next_argument(&mut args, "trainer executable")?),
                    Some("--json") if !run => json = true,
                    Some("--confirm") if run => confirm = true,
                    _ => return Err(invalid(format!("unknown train option: {}", option.to_string_lossy()))),
                }
            }
            let plan = prepare_train(&config, &trainer)?;
            if !run {
                if json { println!("{}", plan.json()); }
                else {
                    println!("trainer: {}", plan.trainer.display());
                    println!("config: {}", plan.config.display());
                    println!("model: {}", plan.model.display());
                    println!("dataset: {} (from {})", plan.dataset, plan.data_dir.display());
                    println!("output: {}", plan.output_dir.display());
                    println!("training: {}", if plan.quantization_bit.is_some() {"QLoRA"} else {"LoRA"});
                    println!("model_compatibility: NOT_VERIFIED; peak_VRAM: UNKNOWN");
                    println!("status: PLAN_ONLY; no external process launched");
                }
            } else {
                if !confirm {
                    return Err(invalid("training may use substantial RAM/VRAM and write output; explicitly pass --confirm to start"));
                }
                eprintln!("Ojon: user-confirmed external training job; examine YAML/data permissions before executing.");
                run_train(&plan)?;
                println!("trainer_exit: SUCCESS; output_dir: {}", plan.output_dir.display());
                println!("quality_benchmark: NOT_RUN; distillation: NOT_IMPLEMENTED");
            }
        }
        Some("dataset-audit") => {
            let path = PathBuf::from(next_argument(&mut args, "dataset file (.jsonl or .json)")?);
            let mut json = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown dataset-audit option: {}", opt.to_string_lossy()))),
                }
            }
            let report = audit_dataset_file(&path)?;
            if json {
                println!("{}", report.json());
            } else {
                println!("dataset: {}", report.source_path.display());
                println!("total_records: {}", report.total_records);
                println!("active_records: {}", report.active_records);
                println!("superseded_records: {}", report.superseded_records);
                println!("references_live: {}", report.total_live_refs);
                println!("references_dead: {}", report.total_dead_refs);
                println!("evergreen_records: {}", report.evergreen_count);
                println!("operational_records: {}", report.operational_count);
                println!("volatile_records: {}", report.volatile_count);
                println!("ephemeral_records: {}", report.ephemeral_count);
                println!("avg_decay_factor: {:.3}", report.avg_decay_factor);
                println!("avg_composite_score: {:.3}", report.avg_composite_score);
            }
        }
        Some("distill-plan") => {
            let source = PathBuf::from(next_argument(&mut args, "source dataset file (.jsonl)")?);
            let out_dir = PathBuf::from(next_argument(&mut args, "new output directory")?);
            let mut min_score = 0.50f64;
            let mut json = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--min-score") => {
                        let os = next_argument(&mut args, "minimum score threshold")?;
                        let s = os.to_str().ok_or_else(|| invalid("min-score must be text"))?;
                        min_score = s.parse::<f64>().map_err(|_| invalid("invalid min-score float"))?;
                    }
                    Some("--json") => json = true,
                    _ => return Err(invalid(format!("unknown distill-plan option: {}", opt.to_string_lossy()))),
                }
            }
            let plan = prepare_distill_plan(&source, &out_dir, min_score)?;
            if json {
                println!("{}", plan.json());
            } else {
                println!("source: {}", plan.source_dataset.display());
                println!("output_directory: {}", plan.output_dir.display());
                println!("total_input_rows: {}", plan.total_input_rows);
                println!("curated_rows: {}", plan.curated_rows);
                println!("curated_split: {}", plan.curated_split_path.display());
                println!("min_score_threshold: {:.2}", plan.min_score_threshold);
                println!("live_reference_ratio: {:.1}%", plan.live_reference_ratio * 100.0);
                println!("status: CURATED_SPLIT_AND_PLAN_WRITTEN");
            }
        }
        Some("retrieve") => {
            let dataset = PathBuf::from(next_argument(&mut args, "dataset file (.jsonl)")?);
            let query = next_argument(&mut args, "query search text")?
                .to_str()
                .ok_or_else(|| invalid("query must be valid Unicode"))?
                .to_owned();
            let mut limit = 5usize;
            let mut min_score = 0.35f64;
            let mut json = false;
            let mut markdown = false;
            while let Some(opt) = args.next() {
                match opt.to_str() {
                    Some("--limit") => {
                        let os = next_argument(&mut args, "match limit")?;
                        let s = os.to_str().ok_or_else(|| invalid("limit must be text"))?;
                        limit = s.parse::<usize>().map_err(|_| invalid("invalid limit integer"))?;
                    }
                    Some("--min-score") => {
                        let os = next_argument(&mut args, "minimum score threshold")?;
                        let s = os.to_str().ok_or_else(|| invalid("min-score must be text"))?;
                        min_score = s.parse::<f64>().map_err(|_| invalid("invalid min-score float"))?;
                    }
                    Some("--json") => json = true,
                    Some("--markdown") => markdown = true,
                    _ => return Err(invalid(format!("unknown retrieve option: {}", opt.to_string_lossy()))),
                }
            }
            let matches = retrieve_recipes(&dataset, &query, limit, min_score)?;
            if json {
                let arr: Vec<_> = matches.iter().map(|m| m.json()).collect();
                println!("{}", serde_json::json!({"query": query, "dataset": dataset.to_string_lossy(), "count": arr.len(), "matches": arr}));
            } else if markdown {
                println!("### Distilled Recipes: \"{}\" ({} matches)\n", query, matches.len());
                for (idx, m) in matches.iter().enumerate() {
                    println!("{}. **{}** `[{}]` (Score: {:.2}, Live: {:.0}%, Decay: {:.0}%)",
                        idx + 1, m.title, m.tier, m.match_score, m.liveness_factor * 100.0, m.decay_factor * 100.0);
                    println!("   - **Intent**: {}", m.instruction.lines().next().unwrap_or(""));
                    let preview = m.summary_or_output.lines().take(3).collect::<Vec<_>>().join("\n     ");
                    println!("   - **Recipe**:\n     {}\n", preview);
                }
            } else {
                println!("Query: \"{}\" | Matches: {}", query, matches.len());
                for (idx, m) in matches.iter().enumerate() {
                    println!("[{}] {} (tier: {}, score: {:.2})", idx + 1, m.title, m.tier, m.match_score);
                    println!("    Instruction: {}", m.instruction.lines().next().unwrap_or(""));
                    let snippet = if m.summary_or_output.len() > 160 {
                        format!("{}...", &m.summary_or_output[..160])
                    } else {
                        m.summary_or_output.clone()
                    };
                    println!("    Preview: {}\n", snippet.replace('\n', " "));
                }
            }
        }
        Some("inspect") | Some("verify") => {
            let path = PathBuf::from(next_argument(&mut args, "GGUF file")?);
            if args.next().is_some() { return Err(invalid("too many arguments")); }
            let summary = inspect_gguf(&path)?;
            show_gguf(&path, &summary);
            println!("verification: GGUF_METADATA_AND_TENSOR_CATALOG_ONLY");
            println!("runtime_load_test: NOT_RUN");
            println!("task_quality_test: NOT_RUN");
        }
        Some("inspect-hf") => {
            let path = PathBuf::from(next_argument(&mut args, "HF model folder")?);
            if args.next().is_some() { return Err(invalid("too many arguments")); }
            let source = inspect_hf_folder(&path)?;
            println!("source: {}", source.directory.display());
            println!("weight_files: {}", source.weight_files);
            println!("weight_bytes: {}", source.source_weight_bytes);
            println!("conversion_script: must support this model family");
            println!("inspection: SOURCE_FILES_ONLY; no tensor payloads loaded");
        }
        Some("quantize-hf") => {
            let source = PathBuf::from(next_argument(&mut args, "HF model directory")?);
            let output = PathBuf::from(next_argument(&mut args, "output GGUF path")?);
            let mut converter = discovered_path(BackendId::HfConverter)
                .unwrap_or_else(|| PathBuf::from("convert_hf_to_gguf.py"));
            let mut python = hf_python_path();
            let mut quantizer = discovered_path(BackendId::LlamaQuantize)
                .unwrap_or_else(|| PathBuf::from("llama-quantize"));
            let mut precision = Precision::default();
            let mut threads = std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4);
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--converter") => converter = PathBuf::from(next_argument(&mut args, "HF converter script")?),
                    Some("--python") => python = PathBuf::from(next_argument(&mut args, "Python executable")?),
                    Some("--quantizer") => quantizer = PathBuf::from(next_argument(&mut args, "quantizer executable")?),
                    Some("--precision") => {
                        let v = next_argument(&mut args, "GGUF precision")?;
                        let v = v.to_str().ok_or_else(|| invalid("precision must be text"))?;
                        precision = Precision::from_str(v).map_err(invalid)?;
                    }
                    Some("--threads") => {
                        let v = next_argument(&mut args, "thread count")?;
                        threads = v.to_str().ok_or_else(|| invalid("invalid thread count"))?
                            .parse().map_err(|_| invalid("invalid thread count"))?;
                    }
                    _ => return Err(invalid(format!("unknown quantize-hf option: {}", option.to_string_lossy()))),
                }
            }
            let plan = prepare_hf(&source, &output, &converter, &python, &quantizer, precision, threads)?;
            println!("model_folder: {}", plan.source.directory.display());
            println!("source_weight_bytes: {}", plan.source.source_weight_bytes);
            println!("converter: {}", plan.converter_script.display());
            println!("quantizer: {}", plan.quantizer.display());
            println!("python: {}", plan.python.display());
            println!("precision: {}", plan.precision);
            println!("scratch_space: LARGE / NOT ESTIMATED; intermediate F16/BF16 GGUF required");
            println!("runtime_load: NOT_TESTED");
            let done = quantize_hf(&plan)?;
            println!("status: HF_CONVERTER_AND_QUANTIZER_EXITED_SUCCESSFULLY");
            println!("output: {}", done.output.display());
            println!("output_bytes: {}", done.info.file_bytes);
            println!("verification: GGUF_METADATA_AND_TENSOR_CATALOG_ONLY");
            if let Some(receipt) = done.receipt { println!("receipt: {}", receipt.display()); }
        }
        Some("backends") => {
            let json = match args.next() {
                None => false,
                Some(value) if value == "--json" => true,
                _ => return Err(invalid("backends accepts only optional --json")),
            };
            if args.next().is_some() { return Err(invalid("too many backends options")); }
            let detected = discover_backends();
            if json {
                let arr: Vec<_> = detected.iter().map(|item| serde_json::json!({
                    "name": item.spec.name,
                    "role": item.spec.role,
                    "found": item.executable.is_some(),
                    "path": item.executable.as_ref().map(|p| p.to_string_lossy().into_owned()),
                    "source": item.found_via,
                    "interface_wired": item.spec.integration == Integration::Wired,
                    "compatibility_verified": false,
                    "official_url": item.spec.website,
                    "override_environment": item.spec.override_env,
                })).collect();
                println!("{}", serde_json::json!({"schema":"ojon.backends.v1","backends":arr,"optional_optimizers":discover_advanced()["optimizers"],"diffusion_backends":discover_diffusion()["backends"]}));
                return Ok(());
            }
            for item in detected {
                let status = match (&item.executable, item.spec.integration) {
                    (Some(_), Integration::Wired) => "DETECTED / ADAPTER CONNECTED",
                    (Some(_), Integration::DiscoveryOnly) => "DETECTED / ADAPTER NOT YET CONNECTED",
                    (None, Integration::Wired) => "NOT FOUND / MANUAL PATH OR DOWNLOAD",
                    (None, Integration::DiscoveryOnly) => "NOT FOUND / FUTURE ADAPTER",
                };
                println!("{}: {}", item.spec.name, status);
                println!("  purpose: {}", item.spec.role);
                if let Some(path) = &item.executable { println!("  executable: {}", path.display()); }
                if let Some(origin) = item.found_via { println!("  discovery: {origin}"); }
                println!("  publisher: {}", item.spec.website);
                println!("  override: {}", item.spec.override_env);
            }
            for opt in discover_advanced()["optimizers"].as_array().into_iter().flatten() {
                println!("optional optimizer {}: checkout={}, hardware=UNVERIFIED",
                    opt["backend"].as_str().unwrap_or("?"),
                    if opt["source_code_present"].as_bool()==Some(true){"DETECTED"}else{"NOT FOUND"});
                println!("  publisher: {}",opt["official_repository"].as_str().unwrap_or("?"));
                println!("  configure: {}",opt["environment_variable"].as_str().unwrap_or("?"));
            }
            for backend in discover_diffusion()["backends"].as_array().into_iter().flatten() {
                println!("diffusion backend {}: {}, image/video loader and quality NOT VERIFIED",
                    backend["backend"].as_str().unwrap_or("?"),
                    if backend["source_present"].as_bool()==Some(true){"CHECKOUT DETECTED"}else{"NOT FOUND"});
                println!("  configure: {}",backend["environment_variable"].as_str().unwrap_or("?"));
                println!("  publisher: {}",backend["upstream"].as_str().unwrap_or("?"));
            }
            println!("Native Ojon 2/3/4-bit model pack/unpack: built in, no RebelUI, Python or GPU required; direct GPU loading not yet implemented.");
            println!("Supported GGUF: public llama-quantize via K/IQ precision methods; optional GSQ/RCO local Python stages require explicit user configuration.");
            println!("Detected means filename found; binary version, model compatibility and runtime must still be verified.");
            println!("Add custom tool directories via OJON_BACKENDS_DIR or explicit --tool path.");
        }
        Some("smoke") => {
            let model = PathBuf::from(next_argument(&mut args, "GGUF model file")?);
            let mut tool = ojon_backend::discovered_path(BackendId::LlamaCli)
                .unwrap_or_else(|| PathBuf::from("llama-cli"));
            let mut tokens = 16u32;
            let mut gpu_layers = 0u32;
            let mut prompt = String::from("Reply with one short sentence: what is two plus two?");
            while let Some(option) = args.next() {
                match option.to_str() {
                    Some("--tool") => tool = PathBuf::from(next_argument(&mut args, "llama-cli executable")?),
                    Some("--tokens") => {
                        let os = next_argument(&mut args, "token count")?;
                        let str_value = os.to_str().ok_or_else(|| invalid("token count must be text"))?;
                        tokens = str_value.parse::<u32>().map_err(|_| invalid("invalid token count"))?;
                    }
                    Some("--gpu-layers") => {
                        let os = next_argument(&mut args, "GPU layer count")?;
                        let str_value = os.to_str().ok_or_else(|| invalid("GPU layer count must be text"))?;
                        gpu_layers = str_value.parse::<u32>().map_err(|_| invalid("invalid GPU layer count"))?;
                    }
                    Some("--prompt") => {
                        let os = next_argument(&mut args, "prompt text")?;
                        prompt = os.to_str().ok_or_else(|| invalid("prompt must be valid Unicode"))?.to_owned();
                    }
                    _ => return Err(invalid(format!("unknown smoke option: {}", option.to_string_lossy()))),
                }
            }
            smoke_gguf(&model, &tool, &prompt, tokens, gpu_layers)?;
            println!("runtime_smoke: LLAMA_CLI_EXITED_SUCCESSFULLY");
            println!("quality_retention: NOT_EVALUATED");
            println!("direct_baseline: NOT_RUN");
        }
        Some("plan") => {
            let plan = parse_plan(args)?;
            show_plan(&plan);
            println!("status: PLANNED_ONLY; no weights modified");
        }
        Some("quantize") => {
            let plan = parse_plan(args)?;
            show_plan(&plan);
            let result = quantize_gguf(&plan)?;
            println!("status: CONVERTER_EXITED_SUCCESSFULLY");
            println!("output: {}", result.output.display());
            println!("output_bytes: {}", result.output_info.file_bytes);
            println!("verification: GGUF_METADATA_AND_TENSOR_CATALOG_ONLY");
            println!("runtime_load_test: NOT_RUN");
            println!("task_quality_test: NOT_RUN");
            if let Some(path) = result.receipt {
                println!("receipt: {}", path.display());
            } else {
                eprintln!("warning: output was created, but the optional receipt could not be written");
            }
        }
        Some("--help") | Some("-h") | Some("help") => usage(),
        _ => { usage(); return Err(invalid("unknown command")); }
    }
    Ok(())
}
fn main() {
    if let Err(error) = run() {
        eprintln!("Ojon error: {error}");
        eprintln!("Run 'ojon --help' for usage.");
        std::process::exit(1);
    }
}
