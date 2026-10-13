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
- Select Q5_K_M (default), Q4_K_M, Q3_K_M, Q6_K and supported IQ4/IQ3/IQ2 formats. Generate an optional GGUF importance matrix with a confirmed public `llama-imatrix` run.
- Plan and run a *sequential* GGUF candidate sweep, recording actual output sizes, durations, status, and errors.
- Compare completed GGUF variants against an original via the optional public `llama-perplexity` executable. **Perplexity is only one quality proxy**, not a guarantee of retained task performance.
- Run a basic inference smoke test with an installed `llama-cli`.
- Preflight and explicitly launch an optional local LoRA/QLoRA fine-tune using installed LLaMA-Factory.
- Curate and prepare high-signal distillation and SFT training splits from external datasets (`ojon distill-plan`), filtering decayed or superseded rows.
- Audit external datasets for drift, dead file references, and knowledge decay (`ojon dataset-audit`).
- Retrieve relevant distilled recipes with sub-millisecond native Rust lookup (`ojon retrieve`), weighted by disk liveness and knowledge half-life.
- Plan and explicitly launch optional upstream [GSQ](https://github.com/IST-DASLab/GSQ) and [RCO](https://github.com/IST-DASLab/RCO) optimization stages, using user-installed Python environments. GSQ assembly and RCO compressed-tensors assembly remain separate explicit operations; their hardware compatibility and results are not yet tested.
- **Native Rust image/video weight storage:** inspect, pack and restore original safetensors models into 2/3/4-bit Ojon-packed containers; experimental SVDQuant INT4 route remains available separately.
- Expose JSON model probes, backend discovery, candidate planning, and quality-report ranking for local agents.

**Not yet universal:** image/video/speech/utility quantization adapters, full multimodal evaluation, automatic quality benchmarking across all model families, cancellation/resume, universal family-specific tests, and real peak memory prediction remain in development. Merely detecting a backend does not mean it is compatible.

## Install

Install [Rust](https://www.rust-lang.org/tools/install), then from this repository:

```powershell
cargo build --release
cargo run --release -- backends
cargo run --release --features gui --bin ojon-gui
```



### Verify a compressed GGUF + LoRA bundle when your GPU is free

No Heretic/PyTorch/Optuna installation is needed to **check an existing
Ojon compressed adapter bundle**. The CPU-only `adapter-smoke-plan`
validates model and adapter catalogs, architecture and file references
without loading model weights. A real smoke test needs only a working
`llama-cli` executable (built for CPU, Vulkan, CUDA, or your backend);
no download or installation is performed by Ojon.

```powershell
# Inspect what will run; does not load weights or consume VRAM:
ojon adapter-smoke-plan "D:\\Models\\edited-adapter" "D:\\Models\\smoke.json" --json

# Run a quick CPU-only load/generation test when ready:
ojon adapter-smoke-run "D:\\Models\\edited-adapter" "D:\\Models\\smoke.json" --tool "C:\\llama.cpp\\build\\bin\\Release\\llama-cli.exe" --gpu-layers 0 --confirm

# Check which GPU IDs llama.cpp actually exposes (mapping varies by build):
llama-cli --list-devices

# Optionally use only a selected *available* GPU, e.g. its reported Vulkan0:
ojon adapter-smoke-run "D:\\Models\\edited-adapter" "D:\\Models\\vision-smoke.json" --tool "C:\\llama.cpp\\build\\bin\\Release\\llama-cli.exe" --gpu-layers 12 --device Vulkan0 --image "D:\\Images\\sample.jpg" --prompt "Describe the image briefly." --confirm
```

The `--image` option requires an mmproj path already present in the
compressed bundle manifest. The runner passes `--mmproj`, `--lora`,
`-m`, deterministic sampling, offline mode and bounded output tokens
to the selected local `llama-cli`. GPU offload is **opt-in**; a large
model can still exceed CPU RAM or free VRAM even when the adapter is tiny.
The receipt records only successful runtime exit and captured logs,
**not measured visual understanding or refusal improvements**.

**Activation capture for ARA** is a distinct prerequisite. Official
llama.cpp's normal CLI does not expose arbitrary linear-module input
and output activations. For now Ojon can import an instrumented runtime's
captured activation JSONL and fit a low-rank adapter from paired module
input/output samples. Fully unattended Heretic-style capture/search
requires adding an inference-backend hook or compatible native library.
This is not a need to install Heretic at Ojon runtime.

### Quantization target: smaller than your CURRENT Q4_K_M, no measured regression

**Fastest first experiment: compare existing compressed downloads without
the original high-precision checkpoint.** Download or reuse a Q3/IQ3 GGUF
of the **same exact base model revision** and its Q4_K_M reference, then:

```powershell
ojon quant-compare-plan 'D:\Models\my-Q4_K_M.gguf' 'D:\Bench\coding-suite.json' 'D:\Bench\compare.json' --candidate iq3m='D:\Models\candidate-IQ3_M.gguf' --json
ojon quant-compare-run 'D:\Models\my-Q4_K_M.gguf' 'D:\Bench\coding-suite.json' 'D:\Bench\compare.json' --candidate iq3m='D:\Models\candidate-IQ3_M.gguf' --confirm
```

For VLMs, append both `--vision-suite vision-suite.json --mmproj mmproj.gguf`.
To require unchanged perplexity too, append `--ppl-corpus heldout.txt`.
The program validates tensor catalog compatibility, but cannot prove
checkpoint lineage from dimensions alone; inspect publisher provenance.

These are **two independent routes**: `quant-compare-*` compares files
that already exist (no source F16 or quantizer required), while
`quant-parity-*` generates new candidates from original high-precision
GGUF. Both require actual model inference only for `*-run`, not `*-plan`.



**Quantization takes priority over editing.** The success reference is the
actual model GGUF you already run (for example, a Q4_K_M), not a theoretical
F16 score. Ojon's `quant-parity-*` workflow quantizes a separate high-precision
GGUF and compares the candidate on the **same** deterministic coding,
reasoning and tool-call suite, with optional vision image tasks and PPL.

**Important:** Do **not** use an existing Q4_K_M as the quantization input.
Requantizing low-precision weights typically compounds error. The original
F16/BF16/F32 GGUF is needed as source; the existing Q4_K_M is the reference.
If you only have the Q4 file, retain it while preparing or downloading a
matching original checkpoint. Ojon checks GGUF architecture and full
tensor-name/shape agreement; this does not cryptographically verify that
both files came from the same model revision.

```powershell
# Plan without loading model weights or using a GPU:
ojon quant-parity-plan 'D:\Models\original-F16.gguf' 'D:\Models\my-Q4_K_M.gguf' 'D:\Bench\coding-suite.json' 'D:\Bench\candidates' --precisions IQ4_XS,IQ3_M,Q3_K_M --max-buffer-mib 512 --gpu-layers 0 --json

# Later: compute candidates, run the same tasks against reference and candidates:
ojon quant-parity-run 'D:\Models\original-F16.gguf' 'D:\Models\my-Q4_K_M.gguf' 'D:\Bench\coding-suite.json' 'D:\Bench\candidates' --precisions IQ4_XS,IQ3_M,Q3_K_M --imatrix 'D:\Bench\calibration.imatrix.gguf' --max-buffer-mib 512 --gpu-layers 0 --confirm

# For VLMs, pass BOTH a matching vision-projector GGUF and vision suite:
ojon quant-parity-run 'D:\Models\original-F16.gguf' 'D:\Models\my-Q4_K_M.gguf' 'D:\Bench\coding-suite.json' 'D:\Bench\candidates' --vision-suite 'D:\Bench\vision-suite.json' --mmproj 'D:\Models\mmproj.gguf' --max-buffer-mib 512 --confirm
```

Set `--ppl-corpus D:\Bench\heldout.txt` (at least 100 bytes),
`--ppl-tool llama-perplexity` and optional `--max-ppl-regression-pct 0`
to also reject higher perplexity on the exact same text corpus.
`--max-loss-points 0` is the default for every text category and vision
accuracy; `--min-saving-percent 5` requires meaningful size reduction.
Results mark this as **no observed regression on the chosen suite**, never
mathematically lossless. Candidates that fail the size gate are skipped
without using the GPU for quality scoring.

The text suite uses `ojon.task_suite.v1`: 6–256 tasks, at least two each
from `coding`, `reasoning` and `tool_call`. Exact/JSON-answer matching
does **not** execute generated code and is not a full agentic benchmark.
Optional vision suite example (real image files must already exist):

```json
{
  "schema": "ojon.vision_suite.v1",
  "tasks": [
    {"id":"ocr-1","image":"images/receipt-1.png","prompt":"Return only the receipt total.","expected":"$15.98","match":"exact"},
    {"id":"screen-1","image":"images/screen-1.png","prompt":"Which button is highlighted?","expected":"Settings","match":"contains"},
    {"id":"chart-1","image":"images/chart-1.png","prompt":"Give only the label of the tallest bar.","expected":"June","match":"exact"}
  ]
}
```

**Large-model memory finding:** Recent upstream `llama-quantize` on
Windows/Linux memory-maps original GGUF and processes tensor rows in slabs.
Its `--max-buffer-size MiB` accepts a per-tensor working-buffer cap;
Ojon uses an opt-in setting in its quantizer and defaults to 512 MiB
for the parity search. This limit is **not** total process RAM usage, and
it requires a llama.cpp build supporting that flag. On-disk source and
intermediate GGUF files remain large; HF → GGUF conversion has its own
memory/disk demands. Check actual resources before a 27B/35B run.

Comparison priorities: **IQ4_XS** (gentler), **IQ3_M** and **Q3_K_M**;
importance-matrix calibration can improve lower-bit results, but never
guarantees parity. We should only accept a smaller candidate after
demonstrating matching visual QA, coding/agent tasks and throughput on
the real hardware. This is separate from refusal-reduction.

### Quantize first, then edit with a small native Rust LoRA (experimental)

Ojon now supports a **compressed-first** workflow that does not load or rewrite
the original BF16 model when producing the final adapter. Start with an
**already quantized, single-file GGUF**, such as a Q4_K_M model downloaded
from Hugging Face or produced by a separate supported quantization run.
The quantizer itself can still need substantial system RAM: this command
**does not** create an enormous quantized GGUF from raw 70 GB BF16 checkpoints.

1. Generate an `ojon.abliteration.v1` recipe with actual **low-rank**
   `lora.a` (rank x input features) and `lora.b` (output features x rank)
   matrices, e.g. from `ojon ara-fit`. Set `preserve_row_norms` to `false`:
   separate LoRA adapters cannot reproduce the post-merge row-renormalization
   performed by the full-precision editor. Direction-vector-only recipes
   are rejected, because computing their projection requires base weights.
2. Run the low-memory adapter exporter:

```powershell
ojon compressed-edit-plan 'C:\Models\Qwen3VL-Q4_K_M.gguf' 'lowrank.json' 'C:\Models\Qwen3VL-edited' --mmproj 'C:\Models\mmproj-Q8.gguf' --json
ojon compressed-edit-run 'C:\Models\Qwen3VL-Q4_K_M.gguf' 'lowrank.json' 'C:\Models\Qwen3VL-edited' --mmproj 'C:\Models\mmproj-Q8.gguf' --confirm
```

3. The output contains **only** `adapter.gguf` and
   `ojon-compressed-bundle.json`. It references the original compressed
   GGUF and optional mmproj by absolute path; neither model file is copied.
   The adapter is a native GGUF v3 LoRA with `general.type=adapter`,
   `adapter.type=lora`, and matching GGUF matrix names and shapes.

```powershell
llama-cli -m 'C:\Models\Qwen3VL-Q4_K_M.gguf' --lora 'C:\Models\Qwen3VL-edited\adapter.gguf' --mmproj 'C:\Models\mmproj-Q8.gguf' --image 'C:\Images\test.jpg' -p 'Describe this picture.'
```

**Compatibility and limits:** Native export currently maps only
`*.self_attn.o_proj.weight` and `*.mlp.down_proj.weight` names to their
`blk.N.attn_output.weight` and `blk.N.ffn_down.weight` GGUF counterparts.
It validates real tensor dimensions and requires equal adapter rank.
Runtime loading, refusal effectiveness, VLM understanding, and model-family
coverage remain **unverified until locally tested**. Real ARA fitting still
needs the corresponding model's per-module good/bad input/output activations;
the exporter alone does not collect them or magically reduce refusals.
For a model larger than available RAM, use an existing quantized GGUF or
a separately validated out-of-core quantizer; Ojon's current llama.cpp
conversion/quantization pipeline is not guaranteed to be streaming.

### Native editing and VLM build pipeline (experimental)

The native Rust editor can **now** modify full safetensors Hugging Face
checkpoint folders (including multiple indexed shards) while copying
tokenizer, config, processor and vision files unchanged. It supports
directional projection or a bounded low-rank adapter from the experimental
ARA-like fitter. This is NOT yet live automatic activation capture or full
Heretic ARA: runtime hooks and model/vision-quality evaluation are still
needed. No additional Rust crates are required.

**Compile externally captured activations** (JSONL records must be captured
from the *output* of the corresponding named linear module):
```json
{"tensor":"model.layers.0.self_attn.o_proj.weight","class":"good","values":[0.1,0.2]}
{"tensor":"model.layers.0.self_attn.o_proj.weight","class":"bad","values":[0.3,0.4]}
```
```powershell
ojon activation-compile "trace.jsonl" "directions.json" --strength 0.7 --confirm
ojon abliterate-model-run "C:\Models\OriginalHF" "directions.json" "C:\Models\EditedHF" --confirm
```

**Optional ARA-like low-rank experiment:** run
`ojon ara-fit module-io.json directions.json --confirm` instead, where
`module-io.json` contains `ojon.ara_fit.v1` with the target tensor,
rank, steps, learning rate, good/bad paired module input/output samples and
loss weights. The fitter currently uses Rust Adam, **not** Heretic's exact
L-BFGS/TPE procedure. Its numerical module loss is not a refusal benchmark.
The resulting JSON uses the same edit-recipe format as the native editor.

**Build edited + quantized model (and vision mmproj when applicable):**
```powershell
ojon model-build-plan "C:\Models\OriginalHF" "directions.json" "C:\Models\Final" --converter "C:\llama.cpp\convert_hf_to_gguf.py" --precision Q4_K_M
ojon model-build-run "C:\Models\OriginalHF" "directions.json" "C:\Models\Final" --converter "C:\llama.cpp\convert_hf_to_gguf.py" --quantizer "C:\llama.cpp\build\bin\Release\llama-quantize.exe" --python python --precision Q4_K_M --confirm
```
This combines the **native Rust edit** with **user-installed llama.cpp**
Python GGUF conversion and native GGUF quantization tools. Only the
conversion stage still needs Python. Output is a new folder containing
`edited-hf/`, `model.gguf`, and—only for compatible VLM architectures—
`mmproj.gguf` in Q8_0. If the installed converter cannot create the vision
projector, the entire staged build fails rather than pretending to produce
a working VLM.

Once local model evaluations are available, call
`ojon method-select measured-scores.json --json` to choose the lowest
refusal-rate candidate **that passes per-capability quality floors**;
unknown vision scores are not accepted for VLM selection.

All output receipts explicitly mark inference, refusal, coding, tool-use and
vision testing as **NOT RUN**. For 27B/35B weights, a full-precision edited
checkpoint plus intermediate GGUF may exceed available RAM/disk; B60 24 GB
quantized inference feasibility does not establish large-model editing
feasibility.

### Experimental native Rust directional ablation — stage 1

This is an **independent CPU-based math-and-weight-editing primitive**, not an
automatic Heretic/ARA port, a model trainer, or an end-to-end uncensored VLM
builder. No Python, Torch, Optuna, Heretic, model download, GPU or new Rust
dependency is used at runtime. It requires a prepared local JSON recipe with
**measured residual/output activation directions** for named transformer
weights. Merely supplying a prompt does not collect those activations.

```powershell
# The recipe and target tensor name/direction length must match the original shard.
ojon abliterate-plan "original.safetensors" "directions.json" "edited.safetensors" --json

# Explicitly run; source file is never modified or overwritten.
ojon abliterate-run "original.safetensors" "directions.json" "edited.safetensors" --confirm
```

Recipe `ojon.abliteration.v1` (the short vectors are *illustrative*, for a
2-row synthetic tensor, **not** values that should be applied to a real model):

```json
{
  "schema": "ojon.abliteration.v1",
  "preserve_row_norms": true,
  "edits": [
    {
      "tensor": "model.layers.0.self_attn.o_proj.weight",
      "direction": [1.0, 0.0],
      "strength": 0.5
    }
  ]
}
```

Instead of `"direction"`, an entry may contain arrays of
`"good_activations"` and `"bad_activations"`, each a list of equally sized,
**externally captured** per-module output vectors. Ojon computes and normalizes
the mean difference. This is **not** an activation-capture implementation.

The only enabled edit targets in stage 1 are explicitly named
`*.self_attn.o_proj.weight` and `*.mlp.down_proj.weight` in transformer
`layers`. The editor rejects vision/projector targets, supports original
F32/F16/BF16 matrices, uses two bounded-memory passes per edited matrix,
and retains all untouched bytes and safetensors metadata. It publishes the new
shard only after every edit succeeds; existing outputs are never overwritten.

**Important limitations:** A saved shard alone is not a complete loadable
Hugging Face or VLM checkpoint. Other original model shards, config, tokenizer,
vision tower and processor must remain available. A real VLM workflow must
package them and test image inputs before claiming VLM support. Stage 1 does
not optimize directions against refusals, select layers automatically, perform
Heretic ARA/L-BFGS/TPE, run inference, assess tool/coding quality, or establish
refusal reduction. The edited shard also needs to be placed in a **new,
complete model directory** with the original sidecars to be loadable; do not
replace original weights. Benchmark it against the original and then quantize
the complete verified model through the compatible Ojon/HF conversion route.


### Keep Rust builds and executables small

Ojon defaults to line-only debug information for its own Rust code and no
dependency debug symbols. Normal `--release` builds remain speed-optimized, but
use thin link-time optimization and symbol stripping to reduce executable size.
The CLI still has no GUI/tensor-framework dependency unless explicitly enabled.

```powershell
# Smallest distributable CLI (potentially slower than normal --release):
cargo build --profile release-small --bin ojon

# Smallest GUI build, only when actually packaging the GUI:
cargo build --profile release-small --features gui --bin ojon-gui

# When full debugger information is actually required:
cargo build --profile debugging
```

Cargo's `target/` directory is a **rebuildable cache**, not Ojon's model storage.
On Windows, you can move build outputs to another disk:

```powershell
$env:CARGO_TARGET_DIR = "D:\\RustBuild\\Ojon"
cargo build --release --bin ojon
```

To **reclaim existing** build space in the repository (which a profile change
does not automatically delete), run `cargo clean --target-dir .\\target` from
the Ojon root, after making sure no Cargo job is running. This deletes *only*
the Rust build cache under that target directory and will require recompilation.
For an additional disk-versus-rebuild-speed trade-off, set
`$env:CARGO_INCREMENTAL = "0"` for a build session. Avoid forcing this for
everyone: incremental builds can make rapid Rust development substantially faster.


For GGUF conversion, install or build public [llama.cpp](https://github.com/ggml-org/llama.cpp) independently. If it isn't found, configure its executable path explicitly. For Hugging Face checkpoint-folder conversion, install the public `convert_hf_to_gguf.py` dependencies separately. Ojon doesn't bundle Python environments, giant backend packages, model weights, or GPU drivers.

### Convert an existing GGUF model

```powershell
ojon inspect "source-f16.gguf"
ojon plan "source-f16.gguf" "model-Q5_K_M.gguf" --precision Q5_K_M --tool "C:\Tools\llama.cpp\llama-quantize.exe"
ojon quantize "source-f16.gguf" "model-Q5_K_M.gguf" --precision Q5_K_M --tool "C:\Tools\llama.cpp\llama-quantize.exe"
ojon verify "model-Q5_K_M.gguf"
```

Use a high-precision source where possible. The source is not overwritten.

### Calibrate for better low-bit quality

```powershell
# Plan only, then explicitly run a calibrated importance matrix from original F16/BF16 GGUF
ojon imatrix-plan "original-f16.gguf" "calibration.txt" "imatrix.gguf"
ojon imatrix-run "original-f16.gguf" "calibration.txt" "imatrix.gguf" --confirm
# Reuse imatrix.gguf for all compatible GGUF Q4/IQ3 candidate sweeps using --imatrix
```

The text must represent the model's real tasks. Use different data for subsequent held-out evaluation. Calibration loads the original model and may need substantial RAM/VRAM; the file alone does not verify task quality.

### Benchmark GGUF outputs with real measurements

Use the independently installed public [llama-bench](https://github.com/ggml-org/llama.cpp/blob/master/tools/llama-bench/README.md). Ojon runs actual, separate prompt-processing (pp) and token-generation (tg) benchmarks, and stores a reviewable JSON receipt. The original BF16 checkpoint is **optional**; if it exceeds available hardware capacity, benchmark the quantized variant by itself instead of claiming a fictional baseline.

```powershell
# Metadata-only preview
ojon bench-plan "M:\Models\my-Q5_K_M.gguf" --gpu-layers 0 --json

# Intentional REAL benchmark; parent folder must exist
ojon bench-run "M:\Models\my-Q5_K_M.gguf" "M:\Reports\q5-bench.json" --tool "C:\Tools\llama.cpp\llama-bench.exe" --gpu-layers 99 --confirm

# Compare two completed receipts only when machine, runtime and test settings match
ojon bench-compare "M:\Reports\bf16-bench.json" "M:\Reports\q5-bench.json" --json
```

Kathamo's **Test Quantized Model** pane now includes an optional **Benchmark Output** button. It measures the current output GGUF on a worker thread (512 prompt / 128 generation tokens, three repetitions), shows both measured speeds, and creates a unique report next to the model. Conversion and inference smoke tests remain separate.

Benchmark receipts capture real backend build, CPU/GPU, test configuration, architecture, output size and pp/tg throughput. Mismatched measured settings suppress comparisons. **This is not model-quality retention, first-token latency, VLM/OCR accuracy, or a direct measure of peak VRAM.** The official `llama-bench` excludes tokenization and sampling overhead. Use independent held-out task evidence before selecting the smallest acceptable quantization.

## Compare several sizes

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

## Compare quantization alternatives (read-only)

Use the multi-route planner before choosing a precision or backend:

```powershell
ojon optimize-plan "C:\Models\original-hf-folder" --device xpu --json
ojon optimize-plan "C:\Models\original-f16.gguf" --device cpu --json
```

The planner lists up to three existing or experimental routes for relevant source families: GGUF Q5/Q4/IQ precision candidates; HF-to-GGUF and Intel AutoRound; diffusion SVDQuant, native Ojon storage packing, and an explicitly unwired GGUF research route. It does not convert, download, run Python, test GPU availability, or pick a winner. Missing backend executables, unsupported sources, experimental loaders and unmeasured resource requirements remain visible in JSON. **Detected is not verified.** Preserve alternatives until matched runtime and held-out task benchmarks permit a meaningful comparison.

## AutoRound research recipes: executable GGUF candidates

Intel AutoRound's [documented GGUF export](https://github.com/intel/auto-round/blob/main/docs/step_by_step.md) is **experimental**. Ojon exposes its actual `gguf:q4_k_m` / `gguf:q5_k_m` formats, never the older invalid `auto_round:gguf`. Only an independently installed AutoRound Python environment is used. Source stays unchanged; output must be a new directory outside the source. Command execution blocks Hugging Face automatic downloads and checks GGUF metadata after conversion, but **does not prove model loadability, GPU acceleration or task accuracy**.

```powershell
# Fast initial alternative: optimized RTN, no tuning iterations
ojon autoround-plan "C:\Models\Qwen3-4B-original" "C:\Models\out\q4-rtn" --bits 4 --format gguf:q4_k_m --iters 0 --device cpu --json
# Calibration-aware alternative: explicit LOCAL calibration JSON (must already exist)
ojon autoround-plan "C:\Models\Qwen3-4B-original" "C:\Models\out\q5-tuned" --bits 5 --format gguf:q5_k_m --iters 200 --dataset "C:\Datasets\calibration.json" --device xpu --json
# Only after reviewing resource needs and backend compatibility, use autoround-run with same arguments and --confirm.
```

Run real held-out quality, load and performance tests before comparing AutoRound candidates against llama.cpp's Q4_K_M, Q5_K_M and IQ/imatrix methods. The `--device xpu` selection is a requested target, not proof the Intel GPU or Python dependencies are available. The complete public method remains owned by Intel's separate tool; Ojon does not reimplement its quantization mathematics.

## Compare HQQ and OpenVINO/NNCF against GGUF

HQQ and OpenVINO/NNCF are **independent optional public implementations**. Ojon does not reimplement either quantizer, install Python packages, or use KS llama.cpp to produce the model. These experimental adapters currently accept locally stored, original Hugging Face checkpoints intended for **causal language models**. Do not treat their outputs as GGUF or automatically compatible with your existing GGUF inference engine.

- [HQQ](https://github.com/dropbox/hqq) quantizes weights without calibration using an independently installed `hqq`/PyTorch/Transformers environment. The output is a native HQQ `qmodel.pt` archive for HQQ's own loader. XPU is only a *requested* target and requires a real compatible torch.xpu environment.
- [OpenVINO/NNCF through Optimum Intel](https://github.com/huggingface/optimum-intel/blob/main/docs/source/openvino/export.mdx) exports INT4 weight-only OpenVINO IR `.xml/.bin` files via an independently installed `optimum-cli`. Setting `--ratio 0.8` keeps some backup INT8 layers, which may help quality at a larger final size. It is not a GGUF conversion.

```powershell
ojon research-plan hqq "C:\Models\hf-original" "C:\Models\hqq-q4" --bits 4 --group 64 --device xpu --json
ojon research-plan openvino "C:\Models\hf-original" "C:\Models\ov-int4" --bits 4 --group 128 --ratio 1.0 --json
ojon research-plan openvino "C:\Models\hf-original" "C:\Models\ov-int4-mixed" --bits 4 --group 128 --ratio 0.8 --json
# After reviewing one plan, change research-plan to research-run and add --confirm.
```

Both routes stage conversion beside the destination and publish only after checking expected output files, without overwriting originals or downloading dependencies. They set Hugging Face offline flags during execution. **Successful export is not successful inference**: loader compatibility, model correctness, speed, VRAM, machine RAM and held-out quality still need measured tests. In particular, the original HF checkpoint is loaded into memory for HQQ quantization and may not fit a 32GB RAM workstation. Runtime performance on Intel B60 is unverified.

## Resource preflight — measure capacity before quantization

Ojon measures **available host RAM** and **free space on the actual destination volume**, plus per-physical-device NVIDIA free VRAM where `nvidia-smi` is installed. Intel Arc Pro B60 free VRAM is read only from valid, optional Intel `xpu-smi` telemetry. When it is absent, unsupported on the host or returns `N/A`, capacity remains **unknown** rather than assuming the advertised 24 GB is free.

```powershell
ojon resource-plan "C:\Models\hf-source" "M:\OjonOutputs" --mode hqq --device xpu --json
ojon resource-plan "C:\Models\hf-source" "M:\OjonOutputs" --mode openvino --device cpu --json
ojon resource-plan "C:\Models\hf-source" "M:\OjonOutputs" --mode autoround --device xpu --json
ojon resource-plan "M:\Models\original-f16.gguf" "M:\OjonOutputs" --mode gguf --device cpu --json
```

The three statuses are `passes_minimum_policy_only`, `budget_below_policy`, and `unverified`. Budgets are deliberately conservative **safety floors**, not predicted converter peak memory. Unknown measurements remain null, never zero. For GGUF streaming and native tensor packers, host-peak usage stays unknown.

`research-run` and `autoround-run` use this resource check before launching their external tools. Insufficient or missing measurements stop execution by default; `--allow-resource-risk` with an explicitly confirmed run is an operator-only override, not proof of safety or device support. No model tensor payloads are loaded by preflight.

## Image and video quantization

- **Native Ojon scalar packing:** Pack a local original F16/BF16/F32 safetensors model's eligible weight matrices into a small, independently specified Ojon `.ojonq` container, using only Rust. 2-, 3- and 4-bit groupwise affine precision; sensitive/unmatched tensors remain at original precision. The matching built-in `unpack` restores a **lossily reconstructed** standard safetensors file. A compatible GPU loader is **not yet implemented**; this is a storage/algorithm prototype, not a ready-to-run low-VRAM image model.
- **SVDQuant INT4:** Optional installed [diffuse-compressor](https://github.com/rootonchair/diffuse-compressor) pipeline for compatible image/video Diffusers transformers. Its CUDA dependencies, runtime loader, model family support and quality remain subject to independent verification.

```powershell
# Read-only; no weights loaded or written
ojon pack-plan "C:\Models\original.safetensors" "C:\Models\out\model-q4.ojonq" --bits 4 --group 64
# Bounded-memory native Rust quantization, original unchanged
ojon pack-run "C:\Models\original.safetensors" "C:\Models\out\model-q4.ojonq" --bits 4 --group 64 --confirm
# Restore a conventional safetensors checkpoint (precision loss remains)
ojon unpack "C:\Models\out\model-q4.ojonq" "C:\Models\out\model-q4-restored.safetensors" --confirm
```

A smaller weight file or a low tensor reconstruction MSE is **not** proof of unchanged generated images/video, nor does the restored model use less VRAM. No model download or Python is required for Ojon's pack/unpack functions.
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
