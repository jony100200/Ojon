# Smallest quality-preserving GGUF — implementation roadmap

**Status:** research/engineering roadmap, not a claim of model-level validation.  
**Source:** `2026-10-26_Ojon_Lowest_Size_Uncensored_Quality_Strategy.md` (KS Smart System strategy).

## Goal

Produce the **smallest practical GGUF** that meets an explicitly defined refusal-behavior target, preserves useful model capabilities, and runs efficiently on the **Intel Arc Pro B60 (SYCL)** and **NVIDIA RTX 4060 Ti (CUDA)** through the intended KS llama.cpp backend.

"Near-BF16" means passing **measured equivalence margins** against the accepted high-precision teacher. It never means mathematically identical behavior.

## Correct production order

1. **Establish an original BF16/F16 checkpoint and evidence.** Freeze source identification, tokenizer/chat template, model configuration, generation parameters and independent test datasets. Work from original Safetensors if weight editing is required.
2. **Evaluate high-precision model edits.** Capture activation examples from a compatible instrumented runtime, generate trial directions, create independently edited Safetensors checkpoints, and compare refusal behavior with capability retention. Only accept a high-precision edited teacher when the defined thresholds pass. Run cross-language and model-specific tests where relevant.
3. **Convert the accepted teacher to GGUF.** Quantize **from the accepted full-precision teacher** for every candidate, never from Q4/IQ3. Preserve model identity and verify compatibility with the chosen llama.cpp build.
4. **Collect quantization evidence.** Generate representative importance-matrix statistics, sensitivity/perturbation measurements, held-out task scores, and—where available—response-token divergence against the accepted teacher. Calibration examples must be distinct from held-out tests.
5. **Search mixed precision.** Use available GGUF per-tensor type controls to allocate more precision to sensitive tensors and fewer bits elsewhere. Compare measured artifact bytes and output quality instead of assuming a Q-label will win.
6. **Apply hard gates before selecting.** Check refusal target, capabilities (coding/reasoning/tool calls/instruction following), token divergence, relevant long-context and multimodal evaluations, and real runtime behavior on **both** GPUs. Reject incomplete or unsupported evidence.
7. **Pick the smallest passing candidate.** Use throughput, startup/latency and memory use as tie breakers, and publish reproducible measured receipts. If none passes, report no winner.

## Existing foundation (audit of main, 2026-10-26)

| Part | Existing source | Verified scope / gap |
|---|---|---|
| HF Safetensors model editing | `src/abliteration.rs`, `src/model_edit.rs`, `src/model_build.rs` | Recipe-driven edits and checkpoint packaging exist; no confirmed automatic model-specific direction capture/end-to-end teacher acceptance |
| Activation evidence | `src/activation_trace.rs`, `src/ara.rs` | Trace import and recipe fitting exist; tracing still requires an instrumented runtime |
| Edit candidate selector | `src/method_select.rs` | Uses externally measured scores; now supports optional `max_refusal_rate`. The final ordered pipeline must **require** an explicit threshold |
| Full-precision GGUF conversion | `src/hf.rs`, `src/llama.rs` | Uses installed llama.cpp converters; does not guarantee checkpoint family support |
| Importance/sensitivity | `src/calibration.rs`, `src/adaptive.rs` | Imatrix-derived ranking is an activation-energy **proxy**, not verified tensor perturbation loss |
| Mixed-precision GGUF | `src/adaptive.rs` | Per-tensor recipes with current Q2_K–Q6_K choices; IQ variants, hardware-aware search and divergence-aware scoring still need implementation and validation |
| Quality/runtime measurements | `src/task_eval.rs`, `src/quality.rs`, `src/quant_parity.rs`, `src/benchmark.rs` | Functional suites and bench adapters exist; no single orchestration enforces **two-reference** quality checks or both-device parity |

## Incremental next milestone

**Make a fully traceable high-precision teacher acceptance stage before optimizing bitrate.**

- Validate an explicit edit trial set and predeclared `max_refusal_rate` plus per-capability loss margins; do not silently accept a lowest-but-still-high refusal rate.
- Require trace provenance and independent held-out evaluations. Preserve unedited model identity and accepted edited teacher separately.
- Store measured source/teacher comparison receipts and artifact hashes. Avoid copying or committing machine-local model paths.
- Add candidate-to-**teacher** gates (not merely Q4-reference gates), then implement token-divergence evidence and the second Intel/NVIDIA hardware requirement.

The refusal threshold support added to `method_select.rs` is **one narrow step**, not completion of the automatic uncensor/quantize pipeline.

## Acceptance rules

A model is not a quality-preserving winner merely because it converts, loads, produces a low perplexity score, or passes a handful of exact-match prompts. Require independently measured capability, refusal-target and device evidence. Report `NOT_TESTED` for anything not actually run. Benchmark on real B60 and RTX 4060 Ti hardware before declaring the release ready.

## Research starting points

- [ByteShape ShapeLearn](https://byteshape.com/technology/) — constrained mixed-precision search inspiration
- [Intel AutoRound](https://github.com/intel/auto-round), [SignRoundV2](https://arxiv.org/abs/2512.04746), [EfficientQAT](https://arxiv.org/abs/2407.11062), [SpinQuant](https://arxiv.org/abs/2405.16406)
- [llama.cpp quantize](https://github.com/ggml-org/llama.cpp/tree/master/tools/quantize) and [imatrix](https://github.com/ggml-org/llama.cpp/tree/master/tools/imatrix)
- [Refusal direction research](https://arxiv.org/abs/2406.11717) and [Heretic](https://github.com/p-e-w/heretic)
