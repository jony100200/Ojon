# Measured results ledger

One entry per real run, with the receipt path. Numbers here are copied from
receipts, never typed from memory. A row proves what it says and nothing more:
"within budget on this corpus slice" is not a capability claim; the task suite
decides that.

| Date | Model | Command | Result | Receipt |
|---|---|---|---|---|
| 2026-10-30 | Qwen3-0.6B F16 (1,509,347,424 B) | `auto-quant --preset balanced --chunks 8` (held-out second half of a 633,601 B own-docs corpus, imatrix from the first half, RTX 4060 Ti) | base q5_k 522,916,384 B, mean KLD 0.024363 ± 0.000673, p99 0.13875. 17 probes measured. Selected recipe: token_embd q4_k, everything else q5_k. Final 503,468,576 B = 33.36% of F16, 96.3% of q5_k; mean KLD 0.025943 (delta +0.00158, budget +0.01), p99 0.15225 (budget +0.05). Attention Q/K, V/O, FFN gate/up and the output head all exceeded +0.01 below q5_k on this model; FFN down at q4_k measured +0.00995 but did not fit the remaining budget with the embedding step. | `D:\KSAppDev\_local\ojon-validation-20261030\q06b-balanced\recipe-receipt.json` |
| 2026-10-30 | Qwen3-0.6B F16 + zero-delta rank-8 LoRA (2 tensors) | `lora-merge-run` through `llama-export-lora` | 2 tensors merged, 311 written, every tensor numerically equal to the base (one -0.0 vs +0.0 byte difference). Plumbing check only. | `D:\KSAppDev\_local\ojon-validation-20261030\q06b-balanced\qwen3-0.6b-merged-f16.lora-merge-receipt.json` |
| 2026-10-30 | Qwen3-0.6B F16 | same `auto-quant`, first attempt | All 17 probes refused by the resource preflight (`budget_below_policy`) with 1.9 GiB host RAM free while a 35B server was loaded; `--resume` after freeing RAM reused imatrix, reference logits and base and completed. Recorded because the refusal is the intended behaviour. | `...\q06b-balanced\recipe-progress.jsonl` (first 17 entries) |

## Reading the Qwen3-0.6B row

A 0.6B model has almost no redundancy to spend: every attention class crossed
the budget at q4_k, so the measured "smallest within +0.01" is 3.7% below plain
q5_k. The same search on a larger model finds more (see the Flint partial run in
`RECIPE_SEARCH.md`), which is the point of measuring per model instead of
publishing one recipe. The `smallest` preset (budget +0.02, floor iq2_s) would
admit FFN down at q4_k and probably FFN gate/up at q4_k; that is a different
loss contract and was not run here.
