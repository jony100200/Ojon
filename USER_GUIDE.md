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
