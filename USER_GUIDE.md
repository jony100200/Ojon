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

Review the plan before running `recipe-run` with the same settings, `--resume` and `--confirm`. See [recipe search](docs/RECIPE_SEARCH.md) for calibration, KL divergence and result interpretation.

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

For evaluation criteria, see [Quality strategy](QUALITY_STRATEGY.md).

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

See [Contributing](CONTRIBUTING.md) for changes and test reports.
