<div align="center">

# Ojon (ওজন)

**Make local AI models smaller. Measure what matters.**

A Rust-based model optimization workstation for quantization, mixed-precision search, and quality evaluation.

[![Build](https://github.com/jony100200/Ojon/actions/workflows/rust-check.yml/badge.svg)](https://github.com/jony100200/Ojon/actions/workflows/rust-check.yml)
[![Rust](https://img.shields.io/badge/built_with-Rust-222222?logo=rust)](Cargo.toml)

[Get started](#get-started) · [Features](#features) · [Documentation](#documentation) · [Contribute](CONTRIBUTING.md)

</div>

## What is Ojon?

Ojon helps developers optimize local AI models for the hardware they actually have. Instead of selecting a quantization format by guesswork, it can search mixed-precision recipes and compare the resulting models against a measured quality budget.

The goal is straightforward: **the smallest useful model, not just the smallest file.**

## Features

| | Capability |
| --- | --- |
| **Model inspection** | Inspect GGUF files, identify model components, and discover available backends. |
| **Quantization** | Convert and quantize supported models through compatible local tools. |
| **Automatic optimization** | Search for mixed-precision recipes and reuse results on compatible model families. |
| **Quality evaluation** | Compare output distributions, task results, file sizes, and runtime performance. |
| **Model editing** | Experiment with model behavior changes and evaluate the result against a baseline. |
| **Desktop interface** | Native GUI built with [Kathamo](https://github.com/jony100200/Kathamo). |

Ojon integrates existing conversion and inference backends rather than replacing them. Backend and model-format support depends on the selected workflow.

## Get started

**Requirements:** [Rust and Cargo](https://rustup.rs/). Install a compatible quantizer or inference backend when running model-processing operations.

```bash
git clone https://github.com/jony100200/Ojon.git
cd Ojon
cargo build --release --locked --bin ojon
```

Discover available backends and inspect a GGUF model:

```bash
cargo run --release --locked --bin ojon -- backends
cargo run --release --locked --bin ojon -- inspect path/to/model.gguf
```

### Optimize a model

Use an original F16/BF16 GGUF file, calibration text and a separate evaluation corpus:

```bash
# Preview the optimization plan.
cargo run --release --locked --bin ojon -- auto-quant model-f16.gguf output/ --preset balanced --corpus calibration.txt --eval-corpus heldout.txt

# Execute the plan.
cargo run --release --locked --bin ojon -- auto-quant model-f16.gguf output/ --preset balanced --corpus calibration.txt --eval-corpus heldout.txt --confirm
```

Available presets: `smallest`, `balanced`, and `safest`. Use `--target-bytes N` to search toward a specific file-size target or `--resume` to continue an interrupted search. Run `cargo run --release --locked --bin ojon -- --help` for more commands.

### Desktop application

```bash
cargo run --release --locked --features gui --bin ojon-gui
```

## Quality first

Ojon tracks model size **and** quality. An accepted candidate needs an appropriate source reference, held-out evaluation, and runtime evidence for the target hardware.

The current development results demonstrate working optimization and evaluation workflows; they do **not** establish general superiority over existing community quantizations.

## Project status

**In development.** The Rust CLI and desktop source are available. Installer packages and a stable release are not yet published. External backends, model architectures and advanced editing features have different levels of validation.

## Documentation

- [User guide](USER_GUIDE.md) — commands, setup and workflow examples
- [Quality strategy](QUALITY_STRATEGY.md) — evaluation and acceptance criteria
- [Mixed-precision recipe search](docs/RECIPE_SEARCH.md) — algorithm and reproducibility
- [MiniMax H3 guide](docs/MINIMAX_H3.md) — optional W4A4/NVFP4 workflows

## Contributing

Bug reports, reproducible model tests, backend integrations and Rust improvements are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

<div align="center">

**Ojon · Kalponic Studio**

[GitHub issues](https://github.com/jony100200/Ojon/issues) · [Builds](https://github.com/jony100200/Ojon/actions) · [Source history](https://github.com/jony100200/Ojon/commits/main)

</div>
