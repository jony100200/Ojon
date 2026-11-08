# Ojon (ওজন)

**Quantize models. Uncensor models. Run AI locally.**

[![Windows Rust Checks](https://github.com/jony100200/Ojon/actions/workflows/rust-check.yml/badge.svg)](https://github.com/jony100200/Ojon/actions/workflows/rust-check.yml) [![Rust](https://img.shields.io/badge/Built_with-Rust-202020?style=flat-square&logo=rust&logoColor=white)](Cargo.toml) [![GGUF](https://img.shields.io/badge/Model_format-GGUF-4C51BF?style=flat-square)](USER_GUIDE.md) [![Native GUI](https://img.shields.io/badge/Native_GUI-Kathamo-2563EB?style=flat-square)](https://github.com/jony100200/Kathamo) [![MIT](https://img.shields.io/badge/License-MIT-0D8065?style=flat-square)](LICENSE)

[Quick start](#build-and-start) · [Quantize](#quantize) · [Uncensor](#uncensor) · [Desktop GUI](#desktop-application) · [User guide](USER_GUIDE.md)

Ojon is a Rust-based local model workstation focused on two jobs:

- **Quantize:** Reduce the size of supported language and video models with selectable precision, mixed-precision search and measurable quality checks. Image-model support is an ongoing target.
- **Uncensor:** Edit supported GGUF model behavior, search refusal-reduction candidates and evaluate results against the original model.

Both are implemented workflows; compatibility and measured results depend on the model and selected backend.

## Build and start

Install [Rust/Cargo](https://rustup.rs/) and clone the repository:

```bash
git clone https://github.com/jony100200/Ojon.git
cd Ojon
cargo build --release --locked --bin ojon
```

Inspect a model and discover available tools:

```bash
cargo run --release --locked --bin ojon -- backends
cargo run --release --locked --bin ojon -- inspect path/to/model.gguf
```

### Quantize

Start with an F16/BF16 GGUF model, calibration text, a separate evaluation corpus and an output folder:

```bash
cargo run --release --locked --bin ojon -- auto-quant model-f16.gguf output/ --preset balanced --corpus calibration.txt --eval-corpus heldout.txt
cargo run --release --locked --bin ojon -- auto-quant model-f16.gguf output/ --preset balanced --corpus calibration.txt --eval-corpus heldout.txt --confirm
```

Presets: `smallest`, `balanced`, `safest`. Compatible quantization/inference tools are required only for operations that use them.

### Uncensor

Start with an original F16/BF16/F32 GGUF model and an existing output directory. The automatic workflow uses compatible local `llama-cli` and `llama-cvector-generator` tools.

```bash
cargo run --release --locked --bin ojon -- uncensor-auto-plan model-f16.gguf output/
cargo run --release --locked --bin ojon -- uncensor-auto-run model-f16.gguf output/ --confirm
```

Ojon reports actual candidate outcomes; it does not assume an edited model is better.

### Desktop application

```bash
cargo run --release --locked --features gui --bin ojon-gui
```

Settings are stored in the user's configuration directory. Model files and optimized outputs use user-controlled locations.

**Release status:** The CLI and native desktop source can be built today. A packaged Windows installer is not yet published.

[User guide](USER_GUIDE.md) · [MiniMax H3 conversion guide](docs/MINIMAX_H3.md) · [Issue tracker](https://github.com/jony100200/Ojon/issues)

**Kalponic Studio**
