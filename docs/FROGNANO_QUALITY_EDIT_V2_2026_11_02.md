# FrogNano — Quality-Preserving Edit Pilot (2026-11-02)

**Status: REJECTED — editing succeeded, no measured legitimate-answer improvement.**

## Objective
Reduce unnecessary refusals while retaining useful coding, reasoning, tool-use, vision and MTP behaviour. Model Baker should eventually own behavioural edits; Ojon currently provides the prototype editing and quantisation/evaluation path.

Community references:
- Heretic: https://github.com/p-e-w/heretic — activation-based modifiers and multi-objective optimisation including KL divergence.
- llama.cpp cvector generator: https://github.com/ggml-org/llama.cpp/blob/master/tools/cvector-generator/README.md

Ojon's current row-norm-preserving GGUF edit is NOT full Heretic ARA.

## Engineering fixes implemented

1. Selector previously ranked candidates by lowering refusal of HARMFUL controls, which was the wrong target. It now requires a measurable reduction in BENIGN over-refusal, retains refusal of harmful controls, and rejects coding/reasoning/tool/vision regressions beyond the configured tolerance.
2. CLI now supports --max-benign-overrefusal-rate. The old --max-refusal-rate flag is retained as a backwards-compatible alias and means BENIGN over-refusal in the updated code.
3. The old bundled probe suite wrongly labelled defensive security explanations harmful. New development screen: 16 clearly legitimate prompts and eight separately labelled harmful controls. NOT a held-out benchmark.
4. Replaced generic positive/negative openings with 16 paired calibration examples sharing EXACTLY the same legitimate user intent and contrasting substantive vs unnecessary-refusal answers. Automatic planning now rejects mismatched pairs and small suites.
5. Corrected the post-quantisation gate: harmful-control refusal retention must not decrease, and benign over-refusal must not regress against the original and the edited teacher. Edited candidates remain experimental; release_accepted=false.

## Real edit experiment

- Untouched source: FrogNano 4B BF16, 8,665,620,032 bytes.
- Extracted a 318,976-byte residual control vector using original Q4_K_M, CUDA cvector generator, 16 matched prompts, mean method.
- Vulkan cvector executable crashed even with -h (Windows 0xC0000005). Rebuilt the CUDA generator successfully from existing KS llama.cpp sources.
- Created separate BF16 GGUF candidate using strength 0.10, attention tensor blk.19.attn_output.weight, preserving row norms and source.
- Quantised the edited BF16 to Q4_K_M (2,783,446,592 bytes) with a 256 MiB buffer and four CPU threads.
- Tested with KS llama.cpp CUDA, 99 GPU layers, 2048 context, deterministic generation, thinking disabled; no existing LM Studio session unloaded.

## Before/after measurements

| Screen | Original Q4 | New edited Q4 |
| --- | ---: | ---: |
| Coding accuracy | 4/5 | 4/5 |
| Reasoning accuracy | 4/4 | 4/4 |
| JSON tool-call accuracy | 3/3 | 3/3 |
| Direct/blunt legitimate prompts flagged | 2/32 | 2/32 |
| OR-Bench-Hard benign-labelled prompts flagged | 30/40 | 30/40 |
| Separate harmful controls flagged | 8/8 | 8/8 |

There were zero changed detector decisions on the 48-case hard screen. Two direct/blunt decisions changed, but manual examination found no verified useful-answer recovery: the original fictional answer contained a false-positive quoted phrase, while the edited version added unnecessary hedging on a fictional political critique. OR-Bench-Hard labels can be ambiguous and were not all independently reviewed. Phrase detection is NOT a semantic judge.

**Decision:** Not a release candidate. The twelve-task capability suite found no regression, but is far too small to prove general intelligence retention. No verified over-refusal improvement; no Hugging Face upload.

## Reproducible local evidence

Folder: <private-evaluation-vault>/frognano-2026-11-02/quality_edit_v2/

| File | SHA-256 |
| --- | --- |
| calibration-q4-cuda.gguf | cac763d07e68d39420cd1d2951c25263cf21c5e569262856e0aed711a478fad8 |
| frognano-bf16-edit-s010-b019.gguf | 4e092f12a06a5dd866b171c42ac5762fd4be48a4297479be20e6d912d34847fc |
| frognano-q4-edit-s010-b019.gguf | 3deee7f0237a981f4efc5ff1fe7c384d20fad7111672c05175f6fc592e3044f5 |
| task_parity_q4.json | 5dad0d96b03827a2b646a74365acc407b6a1d02de51818459303fef08cd1d3f3 |
| direct_blunt_q4.json | 139c9c8ad88bca2951be584de52ab987de7b84a5e6152fa7659d6f5eb6c87ac8 |
| orbench_hard_q4.json | 1adb36f3ea10173b9fcd9530cd85f9da5fb8c891006386abde6fff404993a0c5 |

## Next milestone

Replace the limited directional proxy with genuine model-activation measurement and a quality-preserving objective, including KL on a separate benign holdout and independent semantic grading of changed replies. Preserve MTP/vision where applicable. Benchmark the edited teacher BEFORE publishing or quantisation selection.

Large checkpoints, exact responses, and raw logs remain local. Original checkpoint remains unchanged.
