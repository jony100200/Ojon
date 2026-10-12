# Ojon (ওজন)

**Quantize, evaluate, and work with AI models locally — in one place.**

Ojon is an open-source, Rust-native application from **Kalponic Studio**, with a desktop interface built using the public [Kathamo](https://github.com/jony100200/Kathamo) UI framework.

Ojon brings existing, independent model tools under one roof. Install only the public converters you need; Ojon helps discover them, run compatible workflows, and report actual results. **No private repositories or cloud service are required** for locally available models, data, and conversion tools.

**Status:** pre-alpha. Windows CI compiles and tests the project, but individual models and backends still need practical testing on the intended hardware. Not all model families can be quantized by the same engine.

## What works today

- Inspect GGUF files, Hugging Face checkpoint folders, and standalone safetensors metadata without loading all model weights.
- Detect installed public conversion and inference executables, with links to their official projects.
- Convert compatible original Hugging Face checkpoints to GGUF using public llama.cpp's Python script.
- Quantize compatible GGUF models using an installed public llama.cpp `llama-quantize`.
- Select Q5_K_M (default), Q4_K_M, Q3_K_M, Q6_K and additional supported IQ4/IQ3/IQ2 formats; calibration via an optional llama.cpp importance matrix.
- Plan and run a *sequential* GGUF candidate sweep, recording actual output sizes, durations, status, and errors.
- Compare completed GGUF variants against an original via the optional public `llama-perplexity` executable. **Perplexity is only one quality proxy**, not a guarantee of retained task performance.
- Run a basic inference smoke test with an installed `llama-cli`.
- Preflight and explicitly launch an optional local LoRA/QLoRA fine-tune using installed LLaMA-Factory.
- Curate and prepare high-signal distillation and SFT training splits from external datasets (`ojon distill-plan`), filtering decayed or superseded rows.
- Audit external datasets for drift, dead file references, and knowledge decay (`ojon dataset-audit`).
- Retrieve relevant distilled recipes with sub-millisecond native Rust lookup (`ojon retrieve`), weighted by disk liveness and knowledge half-life.
- Plan and explicitly launch optional upstream [GSQ](https://github.com/IST-DASLab/GSQ) and [RCO](https://github.com/IST-DASLab/RCO) optimization stages, using user-installed Python environments. GSQ assembly and RCO compressed-tensors assembly remain separate explicit operations; their hardware compatibility and results are not yet tested.
- **Experimental image/video:** plan and run public [diffuse-compressor](https://github.com/rootonchair/diffuse-compressor) SVDQ INT4 conversions, or a user-installed [RebelUI](https://github.com/RealRebelAI/RebelUI) single-file W4/W3/W2A8 codebook converter. Model loader compatibility and visual quality remain unverified.
- Expose JSON model probes, backend discovery, candidate planning, and quality-report ranking for local agents.

**Not yet universal:** image/video/speech/utility quantization adapters, full multimodal evaluation, automatic quality benchmarking across all model families, cancellation/resume, and trustworthy hardware-resource planning remain in development. Merely detecting a backend does not mean it is compatible.

## Install

Install [Rust](https://www.rust-lang.org/tools/install), then from this repository:

```powershell
cargo build --release
cargo run --release -- backends
cargo run --release --features gui --bin ojon-gui
```

For GGUF conversion, install or build public [llama.cpp](https://github.com/ggml-org/llama.cpp) independently. If it isn't found, configure its executable path explicitly. For Hugging Face checkpoint-folder conversion, install the public `convert_hf_to_gguf.py` dependencies separately. Ojon doesn't bundle Python environments, giant backend packages, model weights, or GPU drivers.

### Convert an existing GGUF model

```powershell
ojon inspect "source-f16.gguf"
ojon plan "source-f16.gguf" "model-Q5_K_M.gguf" --precision Q5_K_M --tool "C:\Tools\llama.cpp\llama-quantize.exe"
ojon quantize "source-f16.gguf" "model-Q5_K_M.gguf" --precision Q5_K_M --tool "C:\Tools\llama.cpp\llama-quantize.exe"
ojon verify "model-Q5_K_M.gguf"
```

Use a high-precision source where possible. The source is not overwritten.

### Compare several sizes

```powershell
# Output directory must exist and source must be an original F32/F16/BF16 GGUF.
ojon sweep-plan "source-f16.gguf" "C:\Models\outputs"
ojon sweep-run "source-f16.gguf" "C:\Models\outputs" --confirm

# Supply the actual generated ojon-sweep-*.json manifest and held-out corpus.
ojon eval-sweep "C:\Models\outputs\ojon-sweep-ACTUAL-ID.json" "C:\Models\eval.txt" --confirm
```

The sweep writes real conversion measurements. The perplexity evaluator runs *only if you explicitly invoke it* with compatible public tools and a held-out corpus. It reports a **provisional PPL-only** comparison, not a complete model-quality verdict. For any important task, test actual generated outputs, tool usage, code, OCR/grounding or the model's other intended capability.

### Distill, audit, and retrieve external recipes

Ojon operates strictly on external paths specified by the caller and never touches files in the repository itself.

```powershell
# Audit an external dataset for drift, dead file references, and knowledge decay
ojon dataset-audit "C:\Datasets\sft_train.jsonl"

# Curate high-signal rows into a new output split directory
ojon distill-plan "C:\Datasets\sft_train.jsonl" "C:\Datasets\distilled_split" --min-score 0.65

# Sub-millisecond recipe lookup from an external dataset (supports --json and --markdown)
ojon retrieve "C:\Datasets\sft_train.jsonl" "quantize q4_k_m" --limit 5
```

## Image and video quantization (experimental)

Ojon accepts two independent, optionally installed backend checkouts, without bundling their Python or GPU dependencies:

- **SVDQ:** INT4 for a complete local Diffusers pipeline directory (model_index.json and transformer weights) via [diffuse-compressor](https://github.com/rootonchair/diffuse-compressor). Select image or video and test CUDA compatibility before claiming the result runnable.
- **Codebook:** W4A8, W3A8, W2A8 from a single original high-precision safetensors file using RebelUI's separately installed streaming conversion script. It requires an appropriate model-specific loader.

```powershell
# Read-only plan
ojon diffusion-plan "C:\Models\DiffusersModel" "C:\Models\outputs\denoiser-int4.safetensors" --checkout "C:\Tools\diffuse-compressor" --backend svdq --kind image --format int4
# Optional inspection can load a model: always explicit
ojon diffusion-inspect "C:\Models\DiffusersModel" "C:\Models\outputs\denoiser-int4.safetensors" --checkout "C:\Tools\diffuse-compressor" --backend svdq --kind image --format int4 --confirm
# Start actual conversion, preserving originals
ojon diffusion-run "C:\Models\DiffusersModel" "C:\Models\outputs\denoiser-int4.safetensors" --checkout "C:\Tools\diffuse-compressor" --backend svdq --kind image --format int4 --confirm
# For a standalone diffusion safetensors source
ojon diffusion-plan "C:\Models\original.safetensors" "C:\Models\outputs\model-w4a8.safetensors" --checkout "C:\Tools\RebelUI" --backend codebook --kind video --format w4a8
```

The output is staged and safetensors metadata validated before safe publication. Actual file bytes are recorded. A successful conversion is **not** a verified running image/video pipeline or a quality-preserved quantization. A CUDA GPU is required for the SVDQ recipe; Intel Arc support is not claimed. See the User Guide for limitations.
## Optional advanced optimizers

These require **separate public upstream repository checkouts** and a compatible Python environment. They do not install CUDA, PyTorch or models. GPU/RAM requirements are unknown until local verification.

```powershell
# GSQ: review an existing local YAML first. Limits work to one layer by default.
ojon gsq-plan "C:\Tools\GSQ" "C:\Models\gsq-local.yaml"
ojon gsq-run "C:\Tools\GSQ" "C:\Models\gsq-local.yaml" --layers 1 --confirm

# RCO: for a supported local dense Qwen3, build a small CUDA database first.
ojon rco-db-plan "C:\Tools\RCO" "C:\Models\hf-original" "C:\Models\new-layer-db"
ojon rco-db-run "C:\Tools\RCO" "C:\Models\hf-original" "C:\Models\new-layer-db" --confirm

# Then point the next command at the actual per-model database folder.
ojon rco-plan "C:\Tools\RCO" "C:\Models\hf-original" "C:\Models\rco-layer-db" 2.5
ojon rco-run "C:\Tools\RCO" "C:\Models\hf-original" "C:\Models\rco-layer-db" 2.5 --confirm
```

`gsq-assemble` and `rco-assemble` are separate model-package steps. RCO assembly creates a **compressed-tensors Hugging Face checkpoint, not automatically a GGUF file**. RCO database building currently supports only a local dense Qwen3 architecture through upstream CUDA/PyTorch and is **not verified on consumer GPUs**. These are development integrations; they are **not** verified to fit any particular GPU. See the User Guide for boundaries and prerequisites.

## Design and hardware

Ojon's desktop shell is [Kathamo](https://github.com/jony100200/Kathamo). The Rust backend and CLI can also be used without the GUI.

Actual capability depends on the selected model, converter, target loader, GPU/CPU, available RAM/VRAM and scratch storage. Separate GPUs do not automatically pool their VRAM. Python remains appropriate for specialized public ML backends; Rust handles model inspection, orchestration, job control, and reporting.

No conversion output is represented as *quality preserved* merely because the file was written. The goal is to find the **smallest model that still passes representative tests on real hardware**, not to promise mathematically zero quality loss.

See [USER_GUIDE.md](USER_GUIDE.md) for more commands and troubleshooting. Build checks are in [GitHub Actions](https://github.com/jony100200/Ojon/actions). License: [MIT](LICENSE).
