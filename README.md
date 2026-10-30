# Ojon (ওজন)

### Smaller AI models. More freedom. The quality you need.

**Your AI models should fit your hardware — not the other way around.**

Large models demand more storage, memory and expensive hardware. Ojon is a free, open-source workstation from **[Kalponic Studio](https://github.com/jony100200)** built to help you make AI models smaller, test what you actually lose, and get more out of the computer you already own.

Instead of juggling a dozen separate utilities, **Ojon aims to bring model optimisation, uncensoring, fine-tuning and distillation into one straightforward desktop app.**

**[Explore the features](#what-ojon-is-for) · [Get started](#get-started) · [Help build Ojon](#help-build-ojon)**

> **Current status: pre-alpha.** Ojon's desktop app, command-line tools and several working model inspection, conversion and testing workflows are available as source. Universal model support, automatic uncensoring, and one-click fine-tuning/distillation are **goals in development**, not completed features. Results depend on the model, installed tools and target hardware.

## Why use Ojon?

- **Run more AI on your own machine.** Smaller models can mean less RAM, VRAM and disk usage.
- **Keep the intelligence that matters.** Compare outputs and performance instead of assuming a tiny model is still good.
- **Stop hopping between tools.** One home for compatible model conversion, evaluation and optimisation workflows.
- **Choose what matters to you.** Prioritise the smallest file, practical speed or the best measured balance.
- **Work locally.** Use your own models, computer and supported backends without a mandatory cloud account.
- **See the evidence.** Understand the size, speed and quality trade-offs before keeping a result.

## What Ojon is for

| Feature | What it means for you | Progress |
| --- | --- | --- |
| **Optimise Model** | Find a smaller version that still performs well on the tasks you care about. | Available workflows; expanding |
| **Measured Mixed Precision** | Find how low each tensor class can go within a KL-divergence budget, per model, and build the smallest recipe that stays inside it ([method](docs/RECIPE_SEARCH.md)). | Available (`recipe-plan` / `recipe-run` / `recipe-apply`) |
| **Image Parity Gate** | Compare a diffusion GGUF quant against its reference on a fixed-seed ComfyUI workflow: PSNR and pixel differences, pass/fail, receipt. | Available (`diffusion-parity`) |
| **Compare Before & After** | Check size, model behaviour and performance rather than guessing. | Available workflows; expanding |
| **Uncensor Model** | Explore reduced refusal behaviour while testing whether useful abilities are preserved. | Experimental; automation in development |
| **Fine-tune & Distil** | Optionally specialise a model, teach a smaller one from a larger one or recover performance. | Planned/experimental |
| **Desktop Workstation** | Inspect models and manage supported workflows through a native graphical interface. | Pre-alpha |
| **Local-first & Open Source** | Bring your own supported tools and models, with no required hosted account. | Available |

### One place for different kinds of AI

Ojon's long-term goal is to support **all major AI model families**, using the right tools for each rather than pretending one converter works for everything.

| Models | Examples |
| --- | --- |
| **Language (LLM)** | Chat, coding, reasoning and assistants |
| **Vision-language (VLM)** | Models that understand both images and text |
| **Image** | Image generation and editing |
| **Video** | Video generation and editing |
| **Audio** | Speech recognition, voice, sound and music |
| **Utility** | OCR, embeddings, rerankers, detection, segmentation and other specialist AI |

**Support varies by model and backend.** Today, the most mature workflows centre on language models and GGUF. Image, video, audio and specialist-model support are being developed and validated incrementally.

## How it should work

1. **Choose a model.** Ojon identifies what it is and checks compatible tools.
2. **Choose your goal.** Make it smaller; optionally edit refusal behaviour, fine-tune or distil when supported.
3. **Find the right approach.** Ojon compares suitable methods for your model and hardware.
4. **Test the results.** Check whether it loads, how much memory it uses, how fast it runs and how well it performs.
5. **Keep the best version.** Export the smallest candidate that passes your chosen tests — or get an honest report if none does.

*This is the intended end-to-end experience. Some stages currently require manual setup or separate backends.*

## Small doesn't have to mean blindly worse

A successful conversion is not proof that a model is still useful. Ojon is being built around **measured quality**, not just impressive compression numbers.

The target is the **smallest practical, runnable model within your chosen quality limits**. No tool can promise identical quality at every possible file size, so Ojon should say clearly when a result has not been verified.

## Get started

Ojon is currently source-first. You'll need [Rust](https://www.rust-lang.org/tools/install) and the compatible model tools for the operation you want to run.

```bash
git clone https://github.com/jony100200/Ojon.git
cd Ojon

# Launch the desktop workstation
cargo run --locked --features gui --bin ojon-gui

# See the locally available model backends
cargo run --locked --bin ojon -- backends
```

Read the [User Guide](USER_GUIDE.md) for setup and supported workflows. For video quantisation, follow the [MiniMax H3 W4A4/NVFP4 guide](docs/MINIMAX_H3.md), which distinguishes packed W4A4 components from ComfyUI-compatible NVFP4 safetensors. More detailed developer information lives in the [Technical Reference](docs/TECHNICAL_REFERENCE.md).

## Where we're heading

Our priorities are simple: **finish reliable model optimisation and meaningful before/after comparisons; make model editing practical; extend verified support to image, video, audio and utility models; and add optional fine-tuning and distillation.**

Every new model family must produce files that **actually work in their intended applications**, not just smaller files on disk.

## Help build Ojon

Ojon is an early open-source project. If you want AI tools that are easier to use and better suited to everyday hardware, you can help by:

- **Starring this repository** to support its development.
- Trying a supported model and sharing real-world results.
- Reporting bugs, missing model support or confusing workflows.
- Contributing fixes, tests, backend integrations or interface improvements.

**[Star Ojon on GitHub](https://github.com/jony100200/Ojon)** and follow its progress.

---

**Less memory. More models. Measured results.**

Made by **Kalponic Studio**.
