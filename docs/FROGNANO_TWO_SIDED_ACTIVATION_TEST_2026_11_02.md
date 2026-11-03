# FrogNano 4B — Measured Two-Sided Projection Trial (2026-11-02)

**Disposition: REJECTED (NO DEMONSTRATED OVER-REFUSAL IMPROVEMENT).**
This report documents actual model inference, not only source-code unit tests. No Hugging Face release or checkpoint replacement was performed.

## Question

Can the new Ojon two-sided rank-one weight edit, calibrated on real module inputs, reduce unnecessary refusals without weakening the model?

## Inputs, models and runtime

- Untouched source: FrogNano 4B BF16 GGUF, 8,665,620,032 bytes.
- Baseline inference: original FrogNano Q4_K_M GGUF, 2,783,446,592 bytes.
- Edited teacher: separate BF16 GGUF, attention block 19, strength 0.25, two-sided rank-one projection; original file preserved.
- Edited inference: separately quantised Q4_K_M GGUF, 2,783,446,592 bytes.
- CUDA runtime: KS llama.cpp, RTX 4060 Ti 16 GB, 99 GPU layers during inference, 2048-token context, deterministic temperature 0 / seed 42, thinking disabled; existing LM Studio Muse and laptop model untouched.
- Evaluation: 128 output tokens for OR-Bench-Hard, 192 for direct/blunt and the external 20-task maths sample; same settings for baseline and edited Q4.

## Measured module-input capture

KS llama.cpp's cvector generator was instrumented with an **opt-in** callback on the actual GGML_MUL_MAT node whose source weight name is exactly `blk.19.attn_output.weight`. It reads that matrix's F32 input activations (width 4096) via the backend's tensor-copy API, not from generic layer residual embeddings.

Configure the generator with:
- `OJON_CAPTURE_INPUT_WEIGHT=blk.19.attn_output.weight`
- `OJON_CAPTURE_INPUT_JSON=<path>/module_input_blk19.json`

Then run the same positive/negative prompt files through the CUDA cvector generator. A separate 2560-wide residual control vector (`direction.20`) was extracted from the original Q4. Ojon's `uncensor-cvector-plan/run --block 19 --strength 0.25 --input-capture <json>` now pairs it with the measured 4096-wide module-input difference, validates weight identity/sample counts, and sets `preserve_row_norms=false` for the two-sided edit.

**Critical coverage limitation:** 16 positive/negative prompt pairs were evaluated, but the callback recorded only **four executions per group** (120 token rows per group). The other 12 did not yield records for that exact node. This small and incomplete sample is experimental and not release-grade activation calibration. The source of missing callback coverage needs investigation before further optimisation. The input-output directions are not independently established as a meaningful refusal feature.

## Matched capability and behaviour results

| Evaluation | Original Q4 | Two-sided Q4 |
| --- | ---: | ---: |
| 20 publicly authored GSM8K test questions, exact answer only | 1/20 | 1/20 |
| Existing coding tasks | 4/5 | 4/5 |
| JSON tool-call tasks | 3/3 | 3/3 |
| OR-Bench-Hard, 40 benign-labelled prompts: detector phrase hits | 30/40 | 30/40 |
| OR-Bench-Hard, eight harmful controls: detector phrase hits | 8/8 | 8/8 |
| Direct/blunt, 32 legitimate prompts: detector phrase hits | 2/32 | 2/32 |
| Direct/blunt, eight harmful controls: detector phrase hits | 8/8 | 8/8 |

The two refusal screens reported **zero changed detector decisions** each. Generated text may differ, but there were no detector-documented candidate recoveries requiring claimable gains. Also, original direct/blunt suite has a known fictional-dialogue false positive ("I cannot" in character speech). Do not treat 2/32 as a true semantic refusal rate.

The external maths questions came from the [official GSM8K test set](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/test.jsonl), 1,319 items, source SHA-256 `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` (matching its published dataset metadata). Twenty indexes were selected deterministically via `(61*i+17)%1319` for i=0..19, not based on candidate performance. This is an externally authored small sample, not the full benchmark; strict exact-match grading gives both models a poor absolute score. We cannot infer comprehensive intelligence retention from it.

## Distribution-level quality retention (independent-of-calibration corpus)

The external GSM8K text was also used to create a 62,978-byte plain-text held-out corpus (first 120 test questions/answers, **not** the 16 editing-calibration prompts). KS `llama-perplexity` evaluated four chunks of 512 tokens using the original Q4's saved logits as reference; the new edited Q4 was compared to that reference with identical tokenisation and runtime settings.

| Distribution metric | Result |
| --- | ---: |
| Baseline perplexity | 2.3415 +/- 0.12237 |
| Mean KL divergence | **0.000349 +/- 0.000020** |
| Median KL divergence | 0.000089 |
| 99th percentile KL divergence | 0.002721 |
| Maximum KL divergence | 0.007787 |
| Same top predicted token | **99.412% +/- 0.240%** |

These are empirical distribution-level results on **four short chunks**. They show small output-distribution drift, not universal quality parity. GSM8K test items used as the KL corpus may overlap the 20 task items, although no editing optimisation used either evaluation set.

## Reproducible local evidence (E drive only)

Folder: `<private-evaluation-vault>/frognano-2026-11-02/two_sided_v3`

| File | SHA-256 |
| --- | --- |
| module_input_blk19.json | cc5ad8f5538125585810cdae96cf7916dbf5b8dbbdddf5350246c56c8b1b93a8 |
| output_vector_q4.gguf | cac763d07e68d39420cd1d2951c25263cf21c5e569262856e0aed711a478fad8 |
| frognano-bf16-biproject-b19-s025.gguf | e784db1b00e012b2e40c5adce7cdcd07ba4beb4b8f63aa815d3fa47da9b1f888 |
| frognano-q4-biproject-b19-s025.gguf | defb2f4021598ea2414536b34939f44cb994eb09e2677e7173ec8892b86d21f |
| gsm8k20_task_suite.json | 702c83c298a4dcd71025a3dd94d5da0d1dd8117b70cebd38bf06ccee9f8f500d |
| gsm8k20_plus_tasks_comparison.json | 6e35e0a107afc9a90087ce4f740b831006b63ab1b08c91cfac4bfaa1ba649d5e |
| orbench_hard40_results.json | 51505d48c6a5b5ade2ca618ba8159c08c57902b6b32f96fdaf50e7a6fc23b862 |
| direct_blunt40_results.json | a52c68b4efeca9126827a280c55fe8adb9719a4c9d24fe57d2024c2d79e6262c |
| gsm8k120_public_kl_corpus.txt | cd856336eef9bec694f0f9a8d474a46a01adbf110ca4da1b136326b0da4f7b99 |

The original model and the locally loaded LM Studio models were left unchanged. Raw prompts/model outputs and large generated checkpoints stay on E drive, not GitHub.

## Decision and next steps

**Reject candidate at strength 0.25 / block 19 as an uncensorship improvement.**

- The code and real capture/weight-edit path execute correctly. No capability regression was observed in the evaluated tasks.
- Two separate behavioural screens show **no measured change** in unnecessary refusals.
- Fix callback coverage (4/16 captured pairs), then derive full, architecture-correct module I/O and independently review legitimate prompt labels and answers.
- Test candidate layers and activation-conditioned edits with a frozen, separate semantic evaluation set. Use a quality-preserving KL regulariser, not simply larger edit strengths.
- Do not label an edited model successful until helpfulness changes are real, repeated, independently reviewed, and capabilities remain robust.

**Product distinction:** These experiments are intended for Model Baker's behaviour-changing pipeline; Ojon/Quant Maker owns quantisation and quality-retention verification.
