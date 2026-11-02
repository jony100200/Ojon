# Ojon community-model challenge — reproducible screening audit

**Status:** verification tool implemented. No model promoted; no new training/quantisation run.

## Product boundary

- **Bunon:** model training, fine-tuning, data curation and behavioural model edits.
- **Ojon:** independently verify model receipts, quantisation, MTP-component preservation and package a release only after quality gates pass.
- **Porikkha:** external/task-disjoint, independently reviewed capability and semantic-answer evaluation.
- **Twarok and KS llama.cpp:** inference, speculative decoding and repeatable throughput tests.

## Problem

A model card reporting zero refusals cannot establish that an edited model retained reasoning, coding, instruction following and linguistic quality. An exploratory ten-question local comparison is a screening result, not an independent benchmark. Superficial regex checks can mislabel broken Rust code as correct, and a different inference setting invalidates head-to-head conclusions.

The new **`ojon community-challenge-audit`** command reads existing `ks.community_model_screen.v1` JSON receipts without making model calls. It:

1. Requires two or more distinct model receipts, matching task IDs, prompts, kinds, expected answers and recorded inference settings.
2. Recomputes strict exact-answer scores and parses tool JSON instead of trusting the caller's `objective_pass` flag.
3. Does **not** count uncompiled model-authored code as a coding pass, or interpret absence of refusal phrases as semantic helpfulness.
4. Detects duplicate task IDs, incomplete/error outputs, unsupported schemas, or mismatched settings.
5. Includes source SHA-256 hashes and a common suite fingerprint. It reports `COMPARABLE_SCREEN_ONLY`, `INCOMPARABLE_SETTINGS` or `INCOMPLETE_EVIDENCE`; **release_accepted remains false**.

Example with local model receipts (paths are illustrative; no model data is bundled):

```sh
ojon community-challenge-audit screen-model-a.json screen-model-b.json --json
```

## Real local pilot, 2026-11-02

Three independently released community-edited local models were screened with ten authored questions covering reasoning, JSON, C#/Rust and five legitimate sensitive-topic tasks. The raw responses and actual compiler checks are retained in the owner's local evaluation vault.

| Result | Qwen3.8 4B Heretic | Gemma4 12B Balanced | Ornith1.5 9B |
| --- | --- | --- | --- |
| Strict maths exact answers | 1/2 | 1/2 | 1/2 |
| Tool answer parses as JSON object | 1/1 | 1/1 | 1/1 |
| Leading refusal phrase detector | 0/5 | 0/5 | 0/5 |
| Rust compilation and three executed assertions (separate tool) | Pass | Pass | Fail |
| Independent human semantic or editorial evaluation | **Not run** | **Not run** | **Not run** |

Gemma corrected itself to the right probability fraction in a verbose answer, which failed the answer-only format. A separate Gemma run **with reasoning enabled and a larger budget** produced the strict correct `7/22`, but **must not be counted in the reasoning-disabled, matched comparison**.

The Ojon audit verifies that the Qwen/Ornith original JSON receipts have matching task identity and recorded settings. It does not independently verify that the saved settings reflect the host runtime, GPU, quantisation variant or a source-model baseline. The observed results favour Gemma as an **operational, provisional user choice**, not an empirically certified best-in-class model or a proven fully uncensored model.

## Expanded test readiness

Bunon's development-authored frozen English-quality suite currently contains 48 tasks across professional English, dialogue, dark fantasy, adult romance, long-form continuity and adult erotica (eight per group). An opt-in local runner was prepared for 40 non-erotica tasks, with per-task JSON receipts, source SHA-256 and restart-safe outputs.

**The 40-task run was stopped before producing scored cases.** Competing local inference workloads drove free GPU memory below 1 GiB. The loaded model and other agents were not replaced or killed. There is **no extended baseline result** and no valid claim of improvement from this interrupted run. The remaining adult-erotica family requires explicit, independent age/consent and literary review. Development-authored holdouts must not be recycled into Bunon training material.

## Release acceptance milestones

- Run baseline and candidate with exactly matched model runtime, precision, prompts, token budget, thinking setting and pinned external task dataset; attach runtime settings and executable-code test receipts.
- Add source-disjoint independently authored holdouts and blind, semantic editorial scoring. Record directness, usefulness, factual errors, code execution, writing quality and safety-control retention separately.
- Compare unedited model, community model and Bunon-edited model. Record KL/perplexity, context/vision, tokens per second and peak memory. Preserve and verify MTP/draft components independently in Ojon.
- Publish weights only after accuracy/quality retention, useful-answer improvements and packaging verification pass. A creator's claimed refusal count is not itself a release gate.
