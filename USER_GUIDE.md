# Ojon user guide

Ojon provides a Rust command-line application and an optional native Kathamo desktop interface. It works with installed model conversion and inference tools.

## Model compatibility examples

Ojon provides source-first conversion workflows for supported language, vision-language, image and video model formats. These are real examples, not a universal compatibility guarantee.

| Model | Conversion / functionality | Current verification |
| --- | --- | --- |
| **Qwen3-0.6B (LLM)** | F16 GGUF to a measured mixed-precision recipe (Q4_K embeddings, Q5_K elsewhere); 1.51 GB to 503 MB | Prior local conversion and held-out distribution measurements documented; broad task-quality retention is not established |
| **FrogNano 4B (LLM)** | Separately edited BF16 model converted to Q4_K_M GGUF (2.78 GB) | Previous local quantized inference and small task comparison documented; the behavioral edit was **not** shown to reduce refusals |
| **P-H3 Image** | BF16 safetensors to Q4_K GGUF: **40.47 GB to 11.62 GB (71.28% smaller)** | 532 tensors retained, 314 Q4_K tensors; output reopened and inspected; **image generation and visual-quality parity not yet tested** |
| **MiniMax H3 (video)** | W4A4/OrbitQuant component and NVFP4 checkpoint adapter commands plus ComfyUI video testing | A W4A8 MiniMax video was demonstrated using external weights; Ojon-origin quantized weights and a fully verified release still require their own conversion and playback checks |
| **Multimodal VLMs** | GGUF quantization with optional vision-projector conversion plus llama.cpp multimodal parity hooks | Command paths available; each architecture, projector, runtime load and vision quality needs a model-specific test |

### Try the supported workflows

For an F16/BF16 text-model GGUF, use [automatic quantization](#automatic-quantization). For the P-H3 image-model family, use a compatible stable-diffusion.cpp `sd-cli` build:

```sh
ojon diffusion-gguf-plan original-image.safetensors output-image.gguf --tool sd-cli --kind image --format q4_K
ojon diffusion-gguf-run original-image.safetensors output-image.gguf --tool sd-cli --kind image --format q4_K --confirm
```

The command writes a quantized diffusion-model file, **not** its separate text encoder or VAE. Actual image generation requires the matching components and a compatible runtime. For video conversion options, see the [MiniMax H3 guide](docs/MINIMAX_H3.md). For multimodal models, use `ojon --help` to inspect the `quantize-hf` / projector and parity options.

A successful format conversion proves that Ojon created and inspected an artifact; it does not, by itself, prove runtime loading, equal image/video quality, lower VRAM use or faster generation.

## Install

1. Install Rust and Cargo.
2. Clone the repository and build the CLI.
3. Use `backends` to see which model-processing tools are available.

```bash
git clone https://github.com/jony100200/Ojon.git
cd Ojon
cargo build --release --locked --bin ojon
cargo run --release --locked --bin ojon -- backends
```

Use `cargo run --release --locked --bin ojon -- --help` for the complete command reference.

## Inspect a model

```bash
cargo run --release --locked --bin ojon -- inspect path/to/model.gguf
cargo run --release --locked --bin ojon -- probe path/to/model.gguf --json
```

These commands read model metadata without executing inference. For Hugging Face model directories, use `inspect-hf`.

## Automatic quantization

Supply an original F16/BF16 GGUF, a representative calibration corpus, and an independent held-out evaluation corpus.

```bash
cargo run --release --locked --bin ojon -- auto-quant model-f16.gguf output/ --preset balanced --corpus calibration.txt --eval-corpus heldout.txt
```

The first command previews the configuration and probe count. To execute:

```bash
cargo run --release --locked --bin ojon -- auto-quant model-f16.gguf output/ --preset balanced --corpus calibration.txt --eval-corpus heldout.txt --confirm
```

Available presets:

| Preset | Objective |
| --- | --- |
| `safest` | Retain higher precision with a tighter divergence budget |
| `balanced` | Balance file size and measured distribution change |
| `smallest` | Explore lower precision under a looser budget |

Advanced options:

- `--target-bytes N`: target an output size in bytes.
- `--search`: search again instead of applying a stored family recipe.
- `--resume`: continue a previously interrupted search.
- `--device DEVICE`: select a supported runtime device.

The initial search creates and evaluates multiple candidates; larger models require additional memory, disk space and execution time. Inspect the generated receipt before choosing a variant.

## Measured recipe search

To control the mixed-precision search directly, use the recipe tools:

```bash
cargo run --release --locked --bin ojon -- recipe-plan model-f16.gguf output/ --corpus calibration.txt --eval-corpus heldout.txt --base q5_k --floor q4_k --budget 0.01
```

Review the plan before running `recipe-run` with the same settings, `--resume` and `--confirm`. Use separate calibration and evaluation corpora; compare the model's actual quality and size against the source.

To list saved recipes:

```bash
cargo run --release --locked --bin ojon -- recipe-library list
```

## Apply a weight delta, then compress

To change a model before quantizing it, give Ojon a delta file and an `ojon.weight_patch.v1` manifest (delta hash, target tensor names, optional source tensor hashes). Ojon writes a new folder and only reads the original:

```bash
cargo run --release --locked --bin ojon -- weight-patch-plan path/to/hf-model patch.json patched-model
cargo run --release --locked --bin ojon -- weight-patch-run  path/to/hf-model patch.json patched-model --confirm
cargo run --release --locked --bin ojon -- weight-patch-verify patched-model
cargo run --release --locked --bin ojon -- weight-patch-rollback patched-model restored-model --confirm
```

The output folder holds a receipt with every hash and the original bytes of the patched tensors, so the change can be checked and undone exactly.

To run the whole path from one config (patches, GGUF conversion, imatrix, a quant ladder, KLD, your own test commands, then the smallest candidate within your tolerance or no winner):

```bash
cargo run --release --locked --bin ojon -- pipeline-plan pipeline.json
cargo run --release --locked --bin ojon -- pipeline-run  pipeline.json --confirm
```

Each test is a command that writes `{"metrics": {...}, "runtime": {...}}`. Runs resume where they stopped.

## Hardware and quality checks

```bash
cargo run --release --locked --bin ojon -- backends --probe
cargo run --release --locked --bin ojon -- resource-plan path/to/model path/to/output-parent --json
```

Ojon can inspect resources and gather quality/runtime receipts using supported backends. Model-file inspection, successful conversion, runtime loading and actual task-quality preservation are separate checks.

Compare size, loadability, accuracy on held-out tasks and generation speed against the original model before accepting an output.

## Desktop UI

```bash
cargo run --release --locked --features gui --bin ojon-gui
```

The GUI uses [Kathamo](https://github.com/jony100200/Kathamo). The Rust CLI remains available independently of the GUI.

| Screen | What it is for |
|---|---|
| Home | Start a job, see the running job and recent models. |
| Quantize | Choose a model, choose a goal (Best Quality, Balanced, Smallest Practical), review and start. Shows the stage, time and last log line while it runs, then the accepted model with its size and measured quality. |
| Compare answers | On a finished result: send one question to the original and to the smaller model and read both replies. Nothing is scored. |
| Uncensor | The same three steps. The original file is never changed, and a result is kept only if the measurements pass; a rejected run says so and lists each candidate's measured rates. |
| Queue | The running job and past jobs, with Load result where a job left one. |
| Models | Models you opened, models Ojon made, and GGUF files in your folders. |
| Tools, Backends & Setup | The detailed workstation pages and tool paths. |

The setup assistant lists the tools Ojon drives; it does not install them. A repeat run on a model that already has a finished result writes to a new numbered folder.

## Image and video quantization

Ojon supports diffusion-model GGUF conversion through compatible installed backends and native Rust conversion of supported safetensors linear layers into comfy-kitchen W4A8 or W6A8 format. A converted checkpoint still needs its matching text encoder, VAE, loader and workflow to generate an image or video.

For an H3-family safetensors denoiser with compatible linear layers, preview and run a 4-bit conversion:

```bash
ojon w4a8-plan original.safetensors output-w4a8.safetensors --bits 4
ojon w4a8-run original.safetensors output-w4a8.safetensors --bits 4 --confirm
```

For the corresponding 6-bit weight / 8-bit activation format:

```bash
ojon w4a8-plan original.safetensors output-w6a8.safetensors --bits 6
ojon w4a8-run original.safetensors output-w6a8.safetensors --bits 6 --confirm
```

Use `--layers` for a compatible subset of linear-layer names and `--threads` to limit worker threads. The default layer policy targets MiniMax H3 attention/MLP projection layers. The Rust writer produces comfy-kitchen-compatible packed safetensors; it does not automatically package the encoder or VAE. Actual inference speed and image quality depend on the target GPU and loader. A W4A8 checkpoint and a Q4 GGUF use different quantization formats and must not be treated as interchangeable.

For compatible GGUF diffusion checkpoints, use `diffusion-gguf-plan` / `diffusion-gguf-run` (see `ojon --help`). Ojon also provides `gguf-set-kv`, `gguf-restore-norms`, and `gguf-verify-restore` to prepare and verify applicable model metadata and norm tensors. These tools do not guarantee a runnable model without a compatible inference backend.

MiniMax H3 **video** conversion uses a separate family-specific workflow. See [MiniMax H3 conversion guide](docs/MINIMAX_H3.md). MiniMax H3 **Image** and H3 **Video** have different components and must be tested separately.

## Specialized workflows

The CLI also exposes experimental model editing, optional image/video quantization adapters, OCR and audio comparisons, and model-format-specific operations. Check `--help` for supported commands and the selected backend's own requirements.

For MiniMax H3 conversion examples, see [MiniMax H3](docs/MINIMAX_H3.md). These adapters use external conversion tools and require model-specific runtime validation.

## Development

```bash
cargo fmt --all -- --check
cargo check --locked --bin ojon
cargo test --locked --lib --bin ojon
cargo check --locked --features gui --bin ojon-gui
```

For bugs or reproducible compatibility problems, use [GitHub issues](https://github.com/jony100200/Ojon/issues).
