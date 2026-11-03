# Contributing to Ojon

Thanks for helping make local AI model optimisation easier to use and easier to verify.

Ojon is a pre-alpha Rust project. Contributions are most useful when they include a focused change, a reproducible test and an honest account of what was actually measured.

## Before you start

- Check [open issues](https://github.com/jony100200/Ojon/issues) for known bugs and planned work.
- For substantial new workflows, open a feature request first so the design and supported backends are clear.
- Avoid presenting a conversion, GGUF metadata inspection or mock test as proof of real model quality.
- Keep user-supplied models, private prompts, tokens and machine-specific paths out of commits and logs.

## Local development

```bash
git clone https://github.com/jony100200/Ojon.git
cd Ojon

cargo fmt --all -- --check
cargo check --locked --bin ojon
cargo test --locked --lib --bin ojon
cargo check --locked --features gui --bin ojon-gui
```

The GUI depends on [Kathamo](https://github.com/jony100200/Kathamo). External quantisation and inference workflows require their compatible backends; these Rust commands do not prove GPU or model-runtime support.

Run only the checks that match your change, and say which were not run. When changing Python-based optional integrations, run the relevant unit tests in `scripts/`. On Windows, CI also checks the CLI, GUI and offline adapter tests.

## Changes we welcome

- Backend compatibility fixes with exact tool versions and reproducible examples.
- Tests for model detection, quantisation plans, integrity, quality receipts and regression rejection.
- Better UI clarity, accessibility, documentation and actionable errors.
- Independent before/after benchmark evidence and reproducible measurement scripts.

## Pull requests

Keep each pull request focused. Include:

1. **What changed** and why it is useful.
2. **How to reproduce** the problem and verify the fix.
3. **Tests run** and actual results, including anything not tested.
4. **Model/runtime details**, if relevant, without private paths or logs.

Do not add large checkpoints or generated model weights. Keep CI green. Never change an existing user's model or remove files as a side effect of a preflight operation.

For non-public vulnerability reports, see [SECURITY.md](SECURITY.md). For regular questions, use [GitHub Discussions or Issues](https://github.com/jony100200/Ojon/issues), as available.
