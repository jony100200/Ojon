# Ojon user guide

Ojon provides a Rust command-line application and an optional native Kathamo desktop interface. It works with installed model conversion and inference tools.

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
