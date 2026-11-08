# Contributing

Contributions to Ojon are welcome: Rust improvements, backend integrations, reproducible bug reports, model-compatibility fixes and performance measurements.

## Build and test

```bash
cargo fmt --all -- --check
cargo check --locked --bin ojon
cargo test --locked --lib --bin ojon
cargo check --locked --features gui --bin ojon-gui
```

Model-specific changes should also include a reproducible input, expected output and the exact backend version used.

## Pull requests

Keep changes focused. Explain the problem, summarize the fix, and list tests performed. Distinguish unit tests from actual model/runtime benchmarks.

Create an [issue](https://github.com/jony100200/Ojon/issues) for a bug report or feature request.

Do not commit model weights, generated checkpoints, credentials or private data.
