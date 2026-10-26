# Ojon (ওজন)

### Make AI models smaller — without guessing what you lose.

**Ojon is a local-first model optimization workstation by [Kalponic Studio](https://github.com/jony100200).** It brings model inspection, quantization, comparison, and performance testing into one Rust-powered application, with a native desktop interface built on [Kathamo](https://github.com/jony100200/Kathamo).

The goal is straightforward: **keep the capability you need, use less memory, and run more models on the hardware you already own.**

Whether you are compressing an LLM for your laptop, testing a VLM on an Intel or NVIDIA GPU, or experimenting with smaller image-model weights, Ojon helps you turn a collection of scripts into a repeatable workflow.

> **Pre-alpha:** The desktop app and CLI are available as source. Supported operations call real, separately installed backends. Model compatibility and performance still depend on your hardware and the chosen engine. Not every planned workflow is finished.

## Why Ojon?

A smaller file is not automatically a better model. A successful conversion doesn't prove that it can load, run quickly, or answer just as well.

Ojon is built around the questions that matter:

- **How small can this model get?**
- **Does it still handle my coding, reasoning, tools, or visual tasks?**
- **Will the result actually run on my machine?**

Instead of declaring one quantization format “best,” Ojon aims to compare actual candidates and select the smallest model that meets your quality and runtime requirements.

## What you can do

| | Capability |
|---|---|
| **Explore models** | Inspect local GGUF, Safetensors, and Hugging Face checkpoints without loading all the weights. |
| **Make smaller models** | Convert supported checkpoints and create GGUF variants with formats such as Q5, Q4, Q3, and IQ, using installed llama.cpp tools. |
| **Tune the trade-off** | Use importance matrices and experimental per-tensor mixed-precision recipes instead of relying solely on one global bit setting. |
| **Compare results** | Run model-loading checks, held-out functional evaluations, and speed benchmarks, with results you can inspect. |
| **Experiment beyond LLMs** | Try compatible image/video compression backends and research-stage model-editing workflows, with their limitations made explicit. |
| **Stay local** | Bring your own models, tools, hardware, and folders. No account or hosted service is required for local workflows. |

### One workstation, multiple backends

Ojon does not try to replace every inference or quantization engine. It connects compatible tools—including **llama.cpp**, optional **Intel AutoRound**, and selected diffusion tooling—behind one workflow. Install only what you need.

The native GUI is for creators who don't want to manage everything in a terminal. The CLI and JSON output are available for scripting and local agents.

## Get started

You need [Rust](https://www.rust-lang.org/tools/install) and whichever model backend you plan to use.

```bash
git clone https://github.com/jony100200/Ojon.git
cd Ojon

# Open the native workstation
cargo run --locked --features gui --bin ojon-gui

# Or discover locally installed model tools
cargo run --locked --bin ojon -- backends
```

Select an existing model in the GUI to inspect it and prepare a workflow. Quantization and evaluation are explicit operations; Ojon won't silently rewrite your model files.

See the **[User Guide](USER_GUIDE.md)** for supported inputs, setup, real examples, and current limitations. Developers can find the detailed [technical reference](docs/TECHNICAL_REFERENCE.md).

## Roadmap

- **Now — Solid LLM/GGUF workflows:** Improve the workstation, runtime compatibility checks, calibrated quantization, comparison tools, and reproducible results.
- **Next — Quality-first optimization:** Connect model-edit evidence, accepted high-precision teachers, learned mixed-precision search, and stronger capability-preservation tests. See the [quality strategy](docs/QUALITY_STRATEGY.md).
- **Next — Better GPU coverage:** Validate and improve real workflows on Intel Arc, NVIDIA, and CPU, with trustworthy memory and speed results.
- **Later — More model families:** Expand practical VLM evaluation and image, video, audio, and utility-model compression where supported runtimes exist.
- **Toward release:** Improve packaging, documentation, usability, and real-world test coverage.

## Build with us

Ojon is **open source (MIT)** and under active development. Contributions are welcome—from a bug report or supported-model test to a new backend adapter, benchmark suite, interface improvement, or documentation fix.

If you use local AI and want smaller, more practical models, **try Ojon, open an issue, or contribute a pull request.** If the project is useful or the direction interests you, consider **starring the repository** to help others find it.

---

**Less memory. More models. Measured results.**

Made by **Kalponic Studio** · [License](LICENSE) · [User Guide](USER_GUIDE.md) · [Development reference](docs/TECHNICAL_REFERENCE.md)
