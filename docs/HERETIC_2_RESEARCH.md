# Heretic 2.0 / FrogNano research and implementation gates

Research snapshot: 2026-11-02. This is a curated reading list and engineering plan, **not** evidence that Ojon has achieved Heretic 2.0 parity or successfully modified FrogNano behaviour.

## Upstream implementation to inspect before porting

- Official Heretic repository: https://github.com/p-e-w/heretic
- ARA modifier (source): https://github.com/p-e-w/heretic/blob/master/src/heretic/modifiers/ara.py
- ARA design discussion: https://github.com/p-e-w/heretic/pull/211
- Configuration and scorer architecture: https://heretic-project.org/configuration

Heretic 2.x defaults to Arbitrary-Rank Ablation (ARA) and uses Optuna TPE over pluggable scorers. Its ARA captures module inputs/outputs and optimises the change to those modules. Ojon's `src/quant/ara.rs` is a **local-module low-rank proxy**, with no live transformer forward pass, L-BFGS, or TPE. Keep that distinction visible in the UI and evaluation receipts.

## Relevant papers (Hugging Face paper page + arXiv full text)

1. Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (2024).
   - https://huggingface.co/papers/2406.11717
   - https://arxiv.org/abs/2406.11717
   - Motivation for baseline directional editing, **not** a proof all model families use one direction.
2. Wollschläger et al., *The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence* (2025).
   - https://huggingface.co/papers/2502.17420
   - https://arxiv.org/abs/2502.17420
   - Motivates evaluating multidimensional/independent directions.
3. Joad et al., *There Is More to Refusal in Large Language Models than a Single Direction* (2026).
   - https://huggingface.co/papers/2602.02132
   - https://arxiv.org/abs/2602.02132
   - Distinguishes refusal, non-compliance, and benign over-refusal; do not collapse them into one score.
4. Fafuła, *Abliteration Is Not a Scalpel* (2026).
   - https://huggingface.co/papers/2607.17427
   - https://arxiv.org/abs/2607.17427
   - Motivates held-out decision, confidence, uncertainty, and provenance tests beyond refusal and standard accuracy.

Links are enough for reproducible reference; do **not** check multi-megabyte PDFs or model weights into this Git repo. Archive copies and extracted notes separately if a full-text offline library is needed.

## FrogNano / Qwen3.5 hybrid architecture

Local `E:\LMStudio\models\microsoft\FrogNano-4B-2609\config.json` reports 32 language layers and `full_attention_interval=4`. Full attention occurs at zero-based layers 3, 7, 11, 15, 19, 23, 27, 31. The remaining 24 use linear attention. This is **local config evidence only**.

- The current GGUF directional editor supports `blk.N.attn_output.weight` and `blk.N.ffn_down.weight` (plus the explicitly supported shared-expert down target); this is **not** general DeltaNet or vision tensor editing.
- The auto-sweep currently picks conventional `attn_output` blocks. It must never imply that the unedited 24 linear-attention pathways have been covered.
- The November 2026 fix in `default_sweep_blocks` selects adjacent **available** blocks by index rather than assuming layer IDs are consecutive. On the 32-layer hybrid it chooses 19, 23, 27 instead of only 23.
- A checkpoint name containing `uncensored` or a valid GGUF header does not establish measured refusal reduction.

## Next controlled milestone

1. **Baseline:** independently evaluate the original BF16 FrogNano (and original Q4 if available) using the same chat template, exact prompt sets, runtime, and sampling configuration.
2. **Coverage audit:** enumerate actual GGUF tensor names and shapes by block and classify full-attention, linear-attention, FFN, vision, embeddings, and MTP. Fail closed on unsupported writer matrices; never silently claim full hybrid support.
3. **ARA backend experiment:** compare existing Rust directional/proxy candidates with the upstream Heretic 2.x ARA implementation on the same small, supported model. Keep the external backend optional; evaluate whether a pure Rust equivalent is justified **after** measured benefit.
4. **Candidate search:** jointly score unnecessary refusals on legitimate held-out requests, refusal categories, benign answer drift/KL, coding, reasoning, tools, calibration, and model load. Keep a strict independent holdout separate from optimisation probes.
5. **Acceptance:** original and edited and quantised checkpoints must each load successfully; report exact hashes, settings, token counts, scores, and confidence intervals. Default target: <=1% *unnecessary* refusals with >=95% baseline capability on sufficiently powered legitimate suites. A target is never an achieved claim.

Do not re-label Ojon's proxy as full Heretic ARA until activation capture, live optimisation, evaluation, and hardware tests demonstrate parity.
