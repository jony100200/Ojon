//! Common CLI helpers and usage text.

use ojon_backend::{prepare_gguf, BackendId, GgufSummary, Precision, QuantizePlan};
use std::ffi::OsString;
use std::io::{self, ErrorKind};
use std::path::{Path, PathBuf};
use std::str::FromStr;

pub fn usage() {
    eprintln!(
        r#"Ojon (ওজন) — native quantization backend (early development)

Commands:
  ojon inspect   <model.gguf>
  ojon verify    <model.gguf>
  ojon backends [--json]     Detect installed engines and show official download links
  ojon backends --probe [--json]     Explicitly probe installed runtime device IDs and quantizer formats\n  ojon settings path|show|set <key> <value>|clear <key>   Manage local per-user preferences
  ojon probe <model-path> [--json]   Identify model format/family without loading weights
  ojon flint-inspect <model.gguf|dir> [--json]   Inspect Flint MoE 256-expert topology
  ojon flint-pipeline-plan <model.gguf|dir> [--json]   Plan multi-stage RoCo quantization and hardware fit
  ojon flint-evaluate <candidate.pending> [--reference-logits <path>] [--tool <perplexity>] [--gpu-layers N] [--json]
  ojon flint-functional <model.gguf> [--tool <llama-cli>] [--gpu-layers N] [--json]
  ojon flint-uncensor-plan <model.gguf> <output.gguf> [--strength 0.1] [--blocks 20,24,28,32] [--json]
  ojon flint-promote <candidate.pending> --receipt <receipt.json> --confirm
  ojon optimize-workflow-run <model-path> [--baseline Q4.gguf] [--candidate name=smaller.gguf] [--suite tasks.json] [--output-dir folder] [--imatrix calibration.imatrix] [--tensor-sensitivity scores.json] [--ppl-corpus heldout.txt] [--max-ppl-regression-pct 0] [--goal balanced|smallest] --confirm [--json]
  ojon optimize-workflow-plan <model-path> [--baseline current-Q4.gguf] [--candidate name=smaller.gguf] [--suite tasks.json] [--output-dir folder] [--server-tool path-to-llama-server] [--goal balanced|smallest|fastest] [--precisions IQ4_XS,Q3_K_M] [--bench-prompt-tokens 512] [--bench-gen-tokens 128] [--bench-repetitions 3] [--balanced-speed-ratio 0.85] [--device auto|cpu|cuda|xpu] [--runtime-device SYCL0|CUDA0] [--gpu-layers 99] [--min-saving-percent 5] [--max-loss-points 0] [--json]
  ojon optimize-plan <model-path> [--device auto|cpu|cuda|xpu] [--json]   Compare candidate routes without converting
  ojon resource-plan <model-path> <existing-output-parent> [--mode hqq|openvino|autoround|gguf|pack|exl3] [--device cpu|cuda|xpu|auto] [--json]
  ojon dataset-audit <dataset.jsonl> [--json]   Audit external dataset for drift, dead paths, staleness
  ojon distill-plan  <dataset.jsonl> <output-dir> [--min-score 0.5] [--json]  Curate SFT split
  ojon retrieve      <dataset.jsonl> <query> [--limit 5] [--min-score 0.4] [--json] [--markdown]
  ojon train-plan <config.yaml> [--tool <llamafactory-cli>] [--json]
  ojon train-run  <config.yaml> [--tool <llamafactory-cli>] --confirm
  ojon recommend <quality-evidence.json> [--json]  Smallest verified model within tolerance
  ojon ara-fit <module-io-calibration.json> <new-lowrank-edit-recipe.json> --confirm [--json]
  ojon method-select <measured-quality-scores.json> [--json]
  ojon adapter-smoke-plan <bundle-folder> <new-receipt.json> [--tool llama-cli] [--gpu-layers 0] [--device CUDA0|Vulkan0] [--image image.jpg] [--tokens 64] [--prompt text] [--json]
  ojon adapter-smoke-run  <bundle-folder> <new-receipt.json> [same options] --confirm
  ojon compressed-ara-plan <already-quantized-base.gguf> <module-io-calibration.json> <new-adapter-folder> [--mmproj vision.gguf] [--json]
  ojon compressed-ara-run <already-quantized-base.gguf> <module-io-calibration.json> <new-adapter-folder> [--mmproj vision.gguf] --confirm [--json]
  ojon compressed-edit-plan <already-quantized-base.gguf> <lowrank-recipe.json> <new-adapter-folder> [--mmproj vision.gguf] [--json]
  ojon compressed-edit-run <already-quantized-base.gguf> <lowrank-recipe.json> <new-adapter-folder> [--mmproj vision.gguf] --confirm [--json]
  ojon model-build-plan <original-HF-dir> <directions.json> <new-output-folder> --converter <convert_hf_to_gguf.py> [--python python] [--quantizer llama-quantize] [--precision Q4_K_M] [--threads N] [--json]
  ojon model-build-run <original-HF-dir> <directions.json> <new-output-folder> [options] --confirm
  ojon activation-compile <trace.jsonl> <new-directions.json> [--strength 0.7] [--no-preserve] [--json] --confirm
  ojon uncensor-trace-plan <original-HF-folder-or-F16-GGUF> <paired-activations.jsonl> <new-model-folder-or-GGUF> [--strength 0.7] [--no-preserve] [--json]
  ojon uncensor-trace-run <original-HF-folder-or-F16-GGUF> <paired-activations.jsonl> <new-model-folder-or-GGUF> [same options] --confirm
  ojon abliterate-gguf-plan <original-F16-BF16-GGUF> <direction-recipe.json> <new-model.gguf> [--json]
  ojon abliterate-gguf-run <original-F16-BF16-GGUF> <direction-recipe.json> <new-model.gguf> --confirm [--json]
  ojon uncensor-cvector-plan <original-F16-BF16-GGUF> <control-vector.gguf> <new-model.gguf> --block N [--strength 0.7] [--json]
  ojon uncensor-cvector-run <original-F16-BF16-GGUF> <control-vector.gguf> <new-model.gguf> --block N [--strength 0.7] --confirm [--json]
  ojon uncensor-auto-plan <original-F16-GGUF> <output-dir> --refusal-suite probes.json --task-suite tasks.json --paired prompts-folder [--strengths 0.1,0.2,0.35] [--blocks 13,14,15] [--max-capability-loss 0.02] [--max-refusal-rate 0.05] [--quantize Q5_K_M] [--server llama-server] [--gpu-layers N] [--device ID] [--json]
  ojon uncensor-auto-run  <original-F16-GGUF> <output-dir> [same options] --confirm [--json]   Automatic refusal-edit sweep: every candidate is really measured; smallest-refusal quality-passing winner is published
  ojon refusal-eval-plan <model.gguf> <probes.json> <report.json> [--tool llama-cli] [--server llama-server] [--gpu-layers N] [--device ID] [--tokens 96] [--json]
  ojon refusal-eval-run  <model.gguf> <probes.json> <report.json> [same options] --confirm [--json]   Real probe run + bounded phrase classification
  ojon simple-plan <source> <output-dir> [--precision Q4_K_M|Q5_K_M] [--refusal-edit prompts-folder] [--strength 0.1] [--block N] [--refusal-suite probes.json] [--task-suite tasks.json] [--gpu-layers N] [--json]
  ojon simple-run  <source> <output-dir> [same options] --confirm [--json]   One click: optional refusal edit + quantize + real load smoke (+= measured suites when provided)
  ojon abliterate-model-plan <original-HF-folder> <direction-recipe.json> <new-model-folder> [--json]
  ojon abliterate-model-run <original-HF-folder> <direction-recipe.json> <new-model-folder> --confirm [--json]
  ojon abliterate-plan <original-shard.safetensors> <direction-recipe.json> <new-shard.safetensors> [--json]
  ojon abliterate-run <original-shard.safetensors> <direction-recipe.json> <new-shard.safetensors> --confirm [--json]
  ojon sensitivity-extract <original.gguf> <calibration.imatrix> <new-tensor-sensitivity.json> --confirm
  ojon quant-compare-plan <original-or-Q4-reference.gguf> <suite.json> <new-report.json> --candidate name=lower-bit.gguf [--candidate ...] [--vision-suite vision.json --mmproj vision.gguf] [--ppl-corpus heldout.txt] [--json]
  ojon quant-compare-run  <original-or-Q4-reference.gguf> <suite.json> <new-report.json> --candidate name=lower-bit.gguf [--candidate ...] [same options] --confirm
  ojon quant-parity-plan <original-F16.gguf> <original-or-Q4-reference.gguf> <suite.json> <output-dir> [--precisions Q4_K_M,Q5_K_M] [--imatrix path] [--max-buffer-mib 512] [--gpu-layers 0] [--vision-suite vision.json --mmproj vision.gguf] [--ppl-corpus heldout.txt --max-ppl-regression-pct 0] [--json]
  ojon quant-parity-run  <original-F16.gguf> <original-or-Q4-reference.gguf> <suite.json> <output-dir> [same options] --confirm
  ojon adaptive-plan <original.gguf> <tensor-sensitivity.json> <existing-output-dir> <suite.json> [--targets 3.0,3.5,4.0] [--quantizer llama-quantize] [--eval-tool llama-cli] [--imatrix path] [--device Vulkan2] [--gpu-layers 99] [--max-loss 1.0] [--json]
  ojon adaptive-run <original.gguf> <tensor-sensitivity.json> <existing-output-dir> <suite.json> [options] --confirm
  ojon task-suite-check <suite.json> [--json]   Validate held-out suite structure without loading models or exposing prompts
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
  ojon diffusion-gguf-plan <denoiser.safetensors> <output.gguf> [--tool sd-cli] [--kind image|video] [--layout diffusion|checkpoint] [--format q4_0|q5_0|q8_0|q4_K|q3_K]
  ojon diffusion-gguf-run <denoiser.safetensors> <output.gguf> [same options] --confirm
  ojon diffusion-gguf-smoke-plan <denoiser.gguf> <output.png|webm> [--kind image|video] [--component vae=vae.safetensors] [--component llm=encoder.gguf] [--prompt text] [--seed 42] [--steps 4]
  ojon diffusion-gguf-smoke-run <denoiser.gguf> <output.png|webm> [same options] --confirm
  ojon diffusion-gguf-comfy-plan <denoiser.gguf> <api-workflow.json> <receipt.json> [--kind image|video] [--python python] [--node-id ID] [--server http://127.0.0.1:8188]
  ojon diffusion-gguf-comfy-run  <denoiser.gguf> <api-workflow.json> <receipt.json> [same options] --confirm
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
  ojon learn-collector-status [--data-dir <path>] [--json]
  ojon learn-collector-start  [--port <port>] [--data-dir <path>] [--non-blocking]
  ojon learn-dataset-list     [--category <name>] [--json]
  ojon learn-dataset-import   <input.jsonl> <output.jsonl> [--json]
  ojon learn-curate           <episodes.jsonl> <output-dir> [--anchor-ratio 0.08] [--recency-k 3] [--json]
  ojon learn-train-plan       <model-dir> <dataset-dir> <output-dir> [--quant 4] [--json]
  ojon learn-train-run        <model-dir> <dataset-dir> <output-dir> [--quant 4] [--dry-run] [--confirm]
  ojon learn-promote          <eval-matrix.json> <candidate-id> <pointer-file> [--json]
  ojon learn-knowledge-add    <store.jsonl> <source-url> <title> <domain> <content-file|text> [options]
  ojon learn-temporal-query   <store.jsonl> <query-text> [--as-of <epoch|YYYY-MM-DD>] [--domain <d>] [--augment] [--json]
  ojon learn-package-create   <knowledge|skill|release> <id> <name> <version> <input-path> <dest-dir> [options]
  ojon learn-package-install  <pkg-dir> <install-root> [--json]
  ojon learn-package-rollback <knowledge|skill|release> <install-root> [--json]
  ojon learn-daemon-run       [--port <port>] [--data-dir <path>] [--max-disk-mb <N>] [--timeout-s <N>]

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
"#
    );
}
pub fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(ErrorKind::InvalidInput, message.into())
}
pub fn next_argument(
    iter: &mut impl Iterator<Item = OsString>,
    label: &str,
) -> io::Result<OsString> {
    iter.next()
        .ok_or_else(|| invalid(format!("missing {label}")))
}
pub fn show_gguf(path: &Path, info: &GgufSummary) {
    println!("path: {}", path.display());
    println!(
        "format: GGUF v{} ({})",
        info.version,
        if info.big_endian {
            "big-endian"
        } else {
            "little-endian"
        }
    );
    println!(
        "model: {}",
        info.model_name.as_deref().unwrap_or("(not specified)")
    );
    println!(
        "architecture: {}",
        info.architecture.as_deref().unwrap_or("(not specified)")
    );
    println!("file_bytes: {}", info.file_bytes);
    println!("tensors: {}", info.tensor_count);
    println!("metadata_entries: {}", info.metadata_count);
    println!("tensor_data_offset: {}", info.data_offset);
    println!("alignment: {}", info.alignment);
    println!(
        "general.file_type: {}",
        info.file_type
            .map(|n| n.to_string())
            .unwrap_or_else(|| "unknown".into())
    );
}
pub fn show_plan(plan: &QuantizePlan) {
    println!("adapter: llama.cpp-quantize (external executable)");
    println!("source: {}", plan.source.display());
    println!("source_bytes: {}", plan.source_info.file_bytes);
    println!(
        "source_architecture: {}",
        plan.source_info
            .architecture
            .as_deref()
            .unwrap_or("unknown; backend determines compatibility")
    );
    println!("output: {}", plan.output.display());
    println!("precision: {}", plan.precision);
    if let Some(ot) = &plan.output_tensor_type {
        println!("output_tensor_type: {ot}");
    }
    if let Some(et) = &plan.token_embedding_type {
        println!("token_embedding_type: {et}");
    }
    if let Some(tf) = &plan.tensor_type_file {
        println!("tensor_type_file: {}", tf.display());
    }
    println!("threads: {}", plan.threads);
    println!(
        "importance_matrix: {}",
        plan.imatrix
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".into())
    );
    println!("tool: {}", plan.executable.display());
    println!("peak_ram: UNKNOWN (depends on backend/model)");
    println!("peak_vram: UNKNOWN (typically CPU quantization, backend dependent)");
    println!("scratch_storage: UNKNOWN (output is staged alongside destination)");
    println!("compatibility: NOT CONFIRMED until the configured converter runs");
    println!("runtime_load: NOT TESTED");
}
pub fn parse_plan(mut args: impl Iterator<Item = OsString>) -> io::Result<QuantizePlan> {
    let source = PathBuf::from(next_argument(&mut args, "input GGUF file")?);
    let output = PathBuf::from(next_argument(&mut args, "output GGUF file")?);
    let mut precision = Precision::default();
    let mut matrix: Option<PathBuf> = None;
    let mut tensor_types: Option<PathBuf> = None;
    let mut output_type: Option<String> = None;
    let mut embed_type: Option<String> = None;
    let mut max_buffer: Option<u32> = None;
    let mut tool = ojon_backend::discovered_path(BackendId::LlamaQuantize)
        .unwrap_or_else(|| PathBuf::from("llama-quantize"));
    let mut threads = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(4);
    while let Some(option) = args.next() {
        match option.to_str() {
            Some("--precision") => {
                let value = next_argument(&mut args, "precision value")?;
                let s = value
                    .to_str()
                    .ok_or_else(|| invalid("precision must be text"))?;
                precision = Precision::from_str(s).map_err(invalid)?;
            }
            Some("--tool") => {
                tool = PathBuf::from(next_argument(&mut args, "quantizer executable path")?)
            }
            Some("--imatrix") => {
                matrix = Some(PathBuf::from(next_argument(
                    &mut args,
                    "importance matrix file",
                )?))
            }
            Some("--tensor-type-file") => {
                tensor_types = Some(PathBuf::from(next_argument(&mut args, "tensor type file")?))
            }
            Some("--output-tensor-type") => {
                let val = next_argument(&mut args, "output tensor type")?;
                output_type = Some(
                    val.to_str()
                        .ok_or_else(|| invalid("invalid output tensor type"))?
                        .to_owned(),
                );
            }
            Some("--token-embedding-type") => {
                let val = next_argument(&mut args, "token embedding type")?;
                embed_type = Some(
                    val.to_str()
                        .ok_or_else(|| invalid("invalid token embedding type"))?
                        .to_owned(),
                );
            }
            Some("--max-buffer-size") => {
                let val = next_argument(&mut args, "max buffer size in MiB")?;
                max_buffer = Some(
                    val.to_str()
                        .ok_or_else(|| invalid("invalid buffer size"))?
                        .parse()
                        .map_err(|_| invalid("invalid buffer size integer"))?,
                );
            }
            Some("--threads") => {
                let value = next_argument(&mut args, "thread count")?;
                let s = value
                    .to_str()
                    .ok_or_else(|| invalid("thread count must be text"))?;
                threads = s
                    .parse::<usize>()
                    .map_err(|_| invalid("thread count must be a positive integer"))?;
            }
            _ => {
                return Err(invalid(format!(
                    "unknown option: {}",
                    option.to_string_lossy()
                )))
            }
        }
    }
    let mut plan = prepare_gguf(&source, &output, &tool, precision, threads)?;
    if let Some(path) = &matrix {
        plan = plan.with_imatrix(path)?;
    }
    if let Some(path) = &tensor_types {
        plan = plan.with_tensor_type_file(path)?;
    }
    if let Some(ot) = &output_type {
        plan = plan.with_output_tensor_type(ot)?;
    }
    if let Some(et) = &embed_type {
        plan = plan.with_token_embedding_type(et)?;
    }
    if let Some(mb) = max_buffer {
        plan = plan.with_max_buffer_mib(mb)?;
    }
    Ok(plan)
}
