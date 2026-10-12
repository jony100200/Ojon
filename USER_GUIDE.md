# Ojon (ওজন) — User Guide

> **Pre-alpha guide.** Ojon now has Rust backend and a Kathamo native GUI source on `main`, but no packaged installer or confirmed real-model workflow on the maintainer's PC. The Windows compilation workflow is present. Other model families remain future adapter candidates.

## 1. What Ojon does

Ojon is designed to be **one place to inspect, quantize, and test models** using existing engines. It does not train a model or invent a new universal quantization format.

Intended path: **Import → Inspect → Choose supported conversion → Preflight → Quantize → Test → Export**.

## Model discovery, fine-tuning and best-variant recommendation (early backend features)

Ojon aims to make one **local** workflow from existing public backends, not create another quantization or training algorithm. For new AI model families, Ojon should first identify the checkpoint and the target runtime, then connect only a compatible upstream tool. Classification is a suggestion, **not an automatic conversion permission**.

```bash
ojon probe /models/my-model.gguf --json
ojon probe /models/my-model.safetensors --json
ojon probe /models/my-hf-model-folder --json
ojon backends --json
```

For optional LoRA/QLoRA supervised fine-tuning, install public [LLaMA-Factory](https://github.com/hiyouga/LlamaFactory) in a separate compatible Python environment. Supply a local Hugging Face model checkpoint, a local dataset with `dataset_info.json`, and a reviewed LLaMA-Factory YAML config. Ojon only preflights and launches its **existing** trainer; it does not implement training math.

```bash
ojon train-plan /data/train-lora.yaml --json
# Only after manually approving model, dataset and resource use:
ojon train-run /data/train-lora.yaml --confirm
```

Teacher–student distillation is a future **separate** workflow; it is **not currently implemented**. QLoRA is parameter-efficient fine-tuning, not a substitute for a tested deployable quantized model.

For supported GGUF LLMs, a quality-aware converter may use an **importance matrix** made from representative text with the official `llama-imatrix` binary:

```bash
llama-imatrix -m /models/source-f16.gguf -f /data/representative-text.txt -o /models/imatrix.gguf
ojon plan /models/source-f16.gguf /models/output-Q4_K_M.gguf --precision Q4_K_M --imatrix /models/imatrix.gguf
ojon quantize /models/source-f16.gguf /models/output-Q4_K_M.gguf --precision Q4_K_M --imatrix /models/imatrix.gguf
```

The importance matrix may improve quality retention, but it is not a benchmark or a guarantee of negligible loss. Do not apply this LLM-specific calibration route to arbitrary diffusion/video/speech models.

For the smallest acceptable model, benchmark the original and each quantization on an appropriate **identical task suite** and record the actual measurements. Then provide actual values in the JSON quality-evidence format accepted by `ojon recommend`. You can use `ojon recommend --help` for CLI details; do not mark a result verified without performing real tests.

```bash
ojon recommend /reports/real-quality-evidence.json --json
```

Ojon chooses only among candidates reported to have passed real runtime-loading and task-evaluation checks and every user-supplied tolerance. If no candidate passes, it recommends none. The evidence is user-supplied and **not independently authenticated by Ojon**. For VLMs test OCR/grounding; for image/video test relevant generation fidelity; for speech test transcription or speech quality; for embeddings test retrieval quality—not an unrelated text benchmark.

Local agents may call the JSON commands. They cannot control your computer through this repository alone: actual execution requires a locally installed agent or an explicitly authorized bridge. Treat conversion and training as user-approved write operations.

---

## Image and video quantization (experimental)

Image/video diffusion quantization has different model components and loader requirements than llama.cpp language-model GGUF.

**SVDQ / diffuse-compressor:** Clone the upstream [Apache-2.0 diffuse-compressor](https://github.com/rootonchair/diffuse-compressor) project independently. Install its Python and Diffusers dependencies in an isolated environment. Supply a fully downloaded local Diffusers model folder with model_index.json, transformer weights and applicable VAE/text-encoder components. Ojon initially supports the INT4 CUDA path and deliberately refuses NVFP4-native claims on RTX 40-series hardware. Use --kind image or --kind video to select the upstream example script. Low-memory offload is requested; it does **not** prove this specific model will fit 16 GB VRAM.

**Standalone codebook:** Supply a user-installed [RebelUI](https://github.com/RealRebelAI/RebelUI) checkout and original single-file safetensors. Ojon calls its streaming tools/w4a8_convert.py with bounded chunks and conservative exclusions for sensitive layers. W4A8/W3A8/W2A8 are backend-specific *experimental* weight formats, not guaranteed ComfyUI/Diffusers drop-in replacements. RebelUI did not publish a LICENSE file in the version inspected: Ojon does not copy or redistribute its code.

All pathways provide read-only planning first. They require --confirm before externally supplied Python runs. Provide --python pointing to the appropriate isolated environment if plain python is not configured.

```powershell
ojon backends --json
ojon diffusion-plan "C:\Models\FluxKlein4B" "C:\Models\out\klein-int4.safetensors" --checkout "C:\Tools\diffuse-compressor" --backend svdq --kind image --format int4 --samples 32
ojon diffusion-inspect "C:\Models\FluxKlein4B" "C:\Models\out\klein-int4.safetensors" --checkout "C:\Tools\diffuse-compressor" --backend svdq --kind image --format int4 --confirm
ojon diffusion-run "C:\Models\FluxKlein4B" "C:\Models\out\klein-int4.safetensors" --checkout "C:\Tools\diffuse-compressor" --backend svdq --kind image --format int4 --confirm
# For a single original .safetensors source, instead use --backend codebook --format w4a8
```

Substitute your real local paths; these examples have **not** been run on your hardware. Avoid remote model IDs: Ojon's initial diffusion adapter only accepts local files and directories. It requests Hugging Face offline mode, which is not a security sandbox. It does not install backend dependencies.

After conversion, test the result with an independently confirmed compatible loader. For image/video quality, compare matched **prompts, seeds, schedulers, image dimensions, CFG, sampling steps, text encoders and VAE**; for video also compare number of frames, motion consistency and temporal structure. Ojon currently verifies only safetensors structure and actual output bytes, not full visual fidelity, GPU consumption or runtime loading. Unchanged original source files are never overwritten.

---
## Advanced model optimization: GSQ and RCO (experimental)

Ojon can also orchestrate two independent, public Apache-2.0 projects: [GSQ](https://github.com/IST-DASLab/GSQ) for learned low-bit scalar weights and [RCO](https://github.com/IST-DASLab/RCO) for layer-wise precision search under an average-bit budget. These are **separate optional Python programs**, not bundled Ojon quantizers. Their output still requires runtime and quality testing.

**GSQ:** install a trusted copy of the upstream source and its dependencies into your own isolated Python environment. Supply an existing local YAML configuration with `model.name` pointing to a local HF model folder. The safe default is a **one-layer smoke test**; a full run requires `--all-layers`.

```powershell
ojon gsq-plan "C:\Tools\GSQ" "C:\Data\gsq-config.yaml" --python "C:\Venvs\gsq\Scripts\python.exe"
ojon gsq-run "C:\Tools\GSQ" "C:\Data\gsq-config.yaml" --python "C:\Venvs\gsq\Scripts\python.exe" --layers 1 --confirm
# If quantization completes and the checkpoint is ready:
ojon gsq-assemble-plan "C:\Tools\GSQ" "C:\Data\gsq-config.yaml" "C:\Models\gsq-output"
ojon gsq-assemble "C:\Tools\GSQ" "C:\Data\gsq-config.yaml" "C:\Models\gsq-output" --confirm
```

**RCO:** for an original **dense Qwen3** Hugging Face checkpoint, Ojon can launch the public upstream CUDA/PyTorch stage that creates a per-layer 2/3/4-bit GPTQ database. This stage is experimental and requires a suitable CUDA environment, calibration data already cached locally, and substantial scratch memory. It has not been tested on typical consumer GPUs. After the database has been built, Ojon can run a bounded precision search and assemble the result. A search assignment is not a ready-to-load model.

```powershell
ojon rco-db-plan "C:\Tools\RCO" "C:\Models\hf-original" "C:\Data\rco-new-db" --tokens 16384
ojon rco-db-run "C:\Tools\RCO" "C:\Models\hf-original" "C:\Data\rco-new-db" --confirm
# Use the populated output directory actually created by the upstream script:
ojon rco-plan "C:\Tools\RCO" "C:\Models\hf-original" "C:\Data\rco-layer-db" 2.5
ojon rco-run "C:\Tools\RCO" "C:\Models\hf-original" "C:\Data\rco-layer-db" 2.5 --confirm
# After the upstream search produces an assignment:
ojon rco-assemble-plan "C:\Tools\RCO" "C:\Models\hf-original" "C:\Data\rco-layer-db" "C:\Data\rco-layer-db\assignment.json" "C:\Models\rco-packed"
ojon rco-assemble "C:\Tools\RCO" "C:\Models\hf-original" "C:\Data\rco-layer-db" "C:\Data\rco-layer-db\assignment.json" "C:\Models\rco-packed" --confirm
```

Replace all placeholder paths with your actual existing files, including the real assignment file name generated by RCO. RCO assembly produces a **compressed-tensors Hugging Face checkpoint**, not automatically GGUF. Both integrations request offline Hugging Face and Weights & Biases modes; this is **not an operating-system network sandbox**. No claim is made that these researchers' Python trainers fit a specific 16 GB or 24 GB GPU. Run planning first and inspect source, Python environment, configuration, calibration data, scratch space and output.

---

## Using your installed quantization tools

**You install only the external backends you want.** Ojon does not bundle llama.cpp, diffusion runtimes, Python environments or conversion models. For local HF folders, Ojon can call the installed official llama.cpp Python conversion script, then quantize the resulting intermediate GGUF.

1. Download or build the standard, public [llama.cpp](https://github.com/ggml-org/llama.cpp) tools. The initial GGUF workflow uses `llama-quantize` for conversion and optionally `llama-cli` for local inference testing. No private or modified version is required.
2. Ojon searches `PATH`, a few common executable folders and the optional `OJON_BACKENDS_DIR` directory when it starts. The native UI displays **Found** or **Not found**. Choose **Rescan installed tools** after installing a tool.
3. If Ojon cannot find a custom build, enter its exact executable path in the relevant field or set `OJON_LLAMA_QUANTIZE` / `OJON_LLAMA_CLI`. The CLI accepts `--tool` explicitly.
4. Use **Official download** in the GUI to open the publisher's project/release page. Users choose and install the package themselves. No installation is performed by Ojon.
5. A detected file is **not** a successful compatibility test. The current workflow supports compatible single-file GGUF input or a Hugging Face model folder using llama.cpp's public conversion script. Image/speech tools are detected for convenience, but their conversion adapters are not connected.

Use `cargo run -- backends` for the list of detection results, actual found paths and publisher links. If a tool lives outside all recognized folders, configure it manually. The detector intentionally does not scan the entire computer or run executables during startup.

For Windows PowerShell with a custom tools folder:

```powershell
$env:OJON_BACKENDS_DIR = "C:\\Tools\\AI"
cargo run -- backends
cargo run --features gui --bin ojon-gui
```

For other quantization approaches, see the public upstream projects [NNCF](https://github.com/openvinotoolkit/nncf), [Model Optimizer](https://github.com/NVIDIA/Model-Optimizer), [Quark](https://github.com/amd/Quark) and [CTranslate2](https://github.com/OpenNMT/CTranslate2). **These are future adapter candidates, not currently connected Ojon conversion features.**

---

## Current Rust backend (pre-alpha)

The Rust backend is in `src/`. Inspecting GGUF metadata does not require Kathamo, Python or a GPU. The optional `quantize-hf` operation requires Python and the public llama.cpp converter script and its dependencies; the actual quantization uses the standard llama.cpp `llama-quantize` executable.

Install Rust/Cargo and your own compatible `llama-quantize`, then try:

```bash
cargo build --release
cargo run --release -- inspect /models/source-f16.gguf
cargo run --release -- plan /models/source-f16.gguf /models/output-Q5_K_M.gguf --precision Q5_K_M --tool /tools/llama-quantize
cargo run --release -- quantize /models/source-f16.gguf /models/output-Q5_K_M.gguf --precision Q5_K_M --tool /tools/llama-quantize
cargo run --release -- verify /models/output-Q5_K_M.gguf
# Two-stage conversion from an existing local HF checkpoint folder:
cargo run --release -- inspect-hf /models/my-checkpoint
cargo run --release -- quantize-hf /models/my-checkpoint /models/model-Q5_K_M.gguf --converter /tools/llama.cpp/convert_hf_to_gguf.py --python python --quantizer /tools/llama-quantize
cargo run --release -- smoke /models/output-Q5_K_M.gguf --tool /tools/llama-cli --tokens 16 --gpu-layers 0
```

These are examples, **not evidence that conversion has been successfully run**. Use a small, disposable, high-precision model as the first validation. Substitute Windows executable paths as appropriate. Default quantization is Q5_K_M; Q4_K_M, Q3_K_M and Q6_K are also offered. The direct GGUF adapter accepts **single-file GGUF input**. The HF path accepts a local model directory containing `config.json` plus safetensors or `pytorch_model*.bin`, where the upstream conversion script supports that model family. Split GGUF intermediates are not supported. The HF path requires substantial RAM and scratch space for an intermediate GGUF, and may fail on constrained hardware.

A successful conversion means the external converter exited successfully and the resulting file passed Ojon's bounded GGUF header/metadata/tensor-catalog check. The output is staged and never replaces an existing destination. An optional receipt is created alongside it. **This does not verify every tensor payload or guarantee the target runtime can load it.**

`smoke` optionally calls a real installed `llama-cli`; default `--gpu-layers 0` uses CPU mode. Success means the inference command exited successfully, not that the answer was correct or accuracy was preserved. It may still need substantial RAM. No model download, GPU support installation, advanced calibration, cancellation service or full graphical model-quality evaluation is bundled in this stage. The Kathamo window exposes inspection, conversion and a basic inference smoke action.

---

## 2. Import a model

Planned inputs include supported local model folders, safetensors/GGUF bundles, ONNX models and optional user-approved Hugging Face snapshots. Ojon should examine headers, configs, indexes, tokenizer/processor files, required components and shard completeness **without loading the full source model just to display its information**.

An architecture that cannot be confidently identified should show **Needs review**, not a generic conversion button.

## 3. Pick the target runtime before precision

The model's target loader matters as much as its size. A GGUF quantization suitable for one llama.cpp build is not automatically compatible with a diffusion runtime or another VLM loader.

After inspecting a model, Ojon should list only compatible backends and recipes. Precision labels will include their output format and runtime:

- **Q5_K_M / Q4_K_M:** typical quality/size choices for compatible GGUF language models, with eligible Q3/Q6 as alternatives.
- **FP8 / INT8:** when supported by the model architecture and exporter/loader.
- **W4A8 / AWQ / GPTQ / AutoRound:** model-family-specific methods and packing; calibration may be required.
- **NVFP4 / MXFP4:** only on supported models, kernels and target hardware; these are not interchangeable with GGUF Q4.

Unavailable selections must display a reason: no converter, unsupported architecture, incompatible runtime, insufficient memory, missing calibration data or unverified combination.

## 4. Review the preflight plan

Before any write operation, Ojon should show source size, estimated final size, peak CPU RAM, peak VRAM **per physical device**, temporary/scratch storage, converter version, output directory and confidence/unknowns.

Conversions still read the original model weights. Some methods can process weights tensor by tensor or shard by shard; not every calibrated method can. A 30 GB output may still need significantly more scratch space or memory during conversion.

Ojon should never overwrite the original checkpoint. Conversion outputs will be staged separately and published only after checks.

## 5. Convert and follow progress

The current native **Job Status & Telemetry** panel displays bounded real converter output lines, job elapsed time, and, after successful conversion, actual output bytes compared with source bytes. The interface reports **detected executables**, not 'online' backends, and requires the current source to be inspected before starting conversion. **A true completion percentage, cancel/recovery workflow, hardware telemetry and automated task-quality benchmarks are not implemented yet.** Those must never be fabricated, and all source weights remain separate from staged output.

Actual worker binaries may need to be installed or configured separately. No optional backend is assumed present solely because Ojon recognizes its name.

## 6. Test the smaller model

The intended **Test** page uses the target runtime, not a simulation:

| Type | Test experience |
| --- | --- |
| LLM | Prompt/chat, latency, tokens/s and resource use |
| VLM/OCR/grounding | Image input, visual questions, OCR or coordinates |
| Image generator | Prompt/seed/sampler, generated preview and resource use |
| Video generator | Short generated clip and temporal inspection |
| Speech/STT/TTS | Audio transcription or generated speech playback |
| Embedding/reranker | Similarity, rankings and retrieval examples |
| Other utility | The actual task's output and appropriate metrics |

If the converted model still does not fit available hardware, Ojon must say **Not locally load-tested**. Structural validation alone does not establish runtime compatibility or retained model quality.

## 7. Compare against an original-model reference

When both original and quantized models can run, paired local tests are preferable. When the original cannot run, a reputable published benchmark may be used as a **reference only**—with model version, benchmark dataset, prompt template and evaluation conditions recorded. Different settings can make the comparison invalid.

Report statuses should clearly separate: **Converted**, **Format-valid**, **Runtime-loadable**, **Task-tested**, **Direct baseline comparison**, and **External reference comparison**.

## 8. Hardware and larger models

Ojon uses the hardware available on **your machine**. Model size, conversion method, RAM, storage and available CPU/GPU resources determine what can run. There is no requirement to match the developer's hardware.

More VRAM does not make every converter compatible. Multiple GPUs are not one pooled memory device unless the chosen backend explicitly supports that arrangement. CPU-only processing can be a valid alternative for certain converters.

Compatibility reports should include the model, converter version, runtime, operating system and tested hardware so results can be reproduced.

## 9. Errors and safety

- **Unsupported model:** check model family, file completeness, target runtime and converter compatibility.
- **Not enough memory/disk:** change recipe, backend or execution device where the selected method supports it; otherwise use larger hardware.
- **Missing components:** verify the original tokenizer, processor, projector, vision encoder and shard index as appropriate.
- **Conversion succeeded but won't load:** treat as a failed runtime verification, not a successful runnable model.
- **Low-quality result:** run a task-relevant evaluation; do not equate a smaller file with preserved quality.

Local files remain local unless you explicitly select a remote source or external service. Source checkpoints must be read-only during conversion.

## 10. Status and availability

**Pre-alpha:** the public Rust backend and native [Kathamo](https://github.com/jony100200/Kathamo) GUI are on `main`. Windows CI compilation has passed, but successful real-model conversion and inference results have not yet been verified on the developer's own hardware. There is no stable installer. **Ojon requires no private repositories or unpublished studio tools**; users supply compatible public conversion and inference tools.
