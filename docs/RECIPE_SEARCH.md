# Measured mixed-precision recipe search

`recipe-plan` / `recipe-run` / `recipe-apply` answer one question per model:
**how low can each tensor class go before the output distribution moves more
than you allow?** This note explains what the code does, why it is built this
way, and how to read its receipts. It is written so you can reproduce the
method with any llama.cpp build.

## CPU-only first experiment

GGUF conversion itself does not need a GPU. Ojon's `recipe-plan` and
`recipe-run` now default to **zero GPU layers**, including optional imatrix
calibration and KL-divergence evaluation. Opt in to `--ngl 99` or
`--ngl auto` only when a GPU has been explicitly made available.

For an initial low-cost reproducibility check, use an original 0.6B–1B F16
GGUF, a separate calibration text file and a *different* held-out evaluation
text file. Ensure enough **free host RAM** (the shared-workstation CPU-job
admission policy now distinguishes light, moderate and heavy jobs; do not
assume every quantisation needs 8 GiB free) and disk capacity before starting.
Never evict a model another agent is using.

```text
ojon recipe-plan <original-f16.gguf> <existing-empty-output-directory> --corpus <calibration.txt> --eval-corpus <heldout.txt> --quantize-tool <llama-quantize> --perplexity-tool <llama-perplexity> --imatrix auto --imatrix-tool <llama-imatrix> --ngl 0 --threads 4 --ctx 512 --chunks 8 --base q5_k --floor q4_k --budget 0.01 --budget-p99 0.05
```

The plan records the prospective number of conversions. If acceptable,
repeat the same arguments with `recipe-run` instead of `recipe-plan`, plus
`--resume --confirm`; **`--resume` is required** because planning has
already written `recipe-plan.json`. No inference or conversion is run by
the planning command. For a fair comparison, quantise the same original with
unmodified upstream `llama-quantize` using the same imatrix and evaluate
the original, stock and Ojon candidates on exactly the same held-out
corpus. A KL-budget pass is not proof of general capability retention.

## 1. Why a single format is the wrong unit

A GGUF format such as `Q4_K_M` is already a fixed mixed-precision recipe:
attention value/output projections and dense FFN down-projections get one notch
more bits, everything else gets the nominal level. That rule was tuned on
2023-era dense models. Mixture-of-experts models invert the byte budget (in the
Flint example below 38 of 39 billion parameters live in routed experts), and
small models concentrate sensitivity in embeddings and the output head. Any
fixed rule is wrong for some of them. Measuring replaces the rule.

## 2. The measurement: KL divergence of output distributions

For each scored token position `t`, the original model gives a distribution
`P_t` over the vocabulary and the candidate gives `Q_t`. The per-token loss is

```
KLD_t = sum_v P_t(v) * ( log P_t(v) - log Q_t(v) )
```

`llama-perplexity --save-all-logits` stores `log P_t` for a corpus slice;
`--kl-divergence --kl-divergence-base` recomputes `log Q_t` and prints the
distribution of `KLD_t`: mean, median, 99th percentile, maximum.

Why KLD instead of perplexity: perplexity only looks at the probability of the
one token that actually followed, so two models can share a perplexity while
disagreeing everywhere else. KLD compares the whole distribution at every
position. Mean KLD is the budgeted quantity; the 99th percentile is the tail
gate (`--budget-p99`), because a mean of 0.01 can hide positions where the
candidate puts almost no mass on what the original was certain about.

KLD is still a distance, not a task score. A recipe within budget goes to the
task suite before anyone claims capability was kept.

## 3. The protocol

1. **Reference**: original F16/BF16 GGUF over a fixed corpus slice
   (`--chunks`, default 40 chunks of `--ctx` tokens). Same slice for every
   candidate, so numbers are comparable.
2. **Importance matrix** (`--imatrix auto`): `llama-imatrix` on the calibration
   half of the corpus (`corpus-build` assembles one from your own chats, docs and
   code; aim for 1 MiB or more of task-representative text). Low-bit levels (iq2/iq3/iq4) need it; without one the search
   understates how low a class can go. On Qwen3-0.6B the all-q5_k base dropped
   from mean KLD 0.0438 to 0.0234 just by adding it.
3. **Base**: every searched class pinned to `--base` through a complete
   `--tensor-type-file`, so the converter's own mixture rules cannot blur the
   measurement. Its mean KLD is the floor.
4. **Probes**: for each class and each ladder level below the base (down to the
   class floor), one quantization with only that class lowered. The probe's
   mean KLD minus the base mean KLD is that class's measured cost at that level.
   One probe file is kept on disk at a time.
5. **Selection**: greedy. At each round every class may step one level down;
   the step with the most bytes saved per unit of KLD added wins, until the
   running sum of deltas reaches `--budget`. Steps whose probe measured no
   increase are free and always taken. Greedy on a ratio is the standard
   approximation for this knapsack; the exact optimum would need all 2^n
   combinations and the deltas are only approximately additive anyway.
6. **Final**: the selected recipe is quantized and measured again.
7. **Boundary refinement** (default on): one more file keeps blocks 0-1 and the
   last two blocks of every lowered class at the base level, because boundary
   blocks carry more representation sensitivity (the observation behind
   per-layer dynamic quants). It replaces the recipe only if it measures better
   by more than both reported uncertainties and is at most 2% larger; the plain
   recipe is kept beside it as `recipe-plain.gguf`. The tensor-type file lists
   the block-specific lines first because `llama-quantize` applies the first
   matching pattern. The receipt's
   `measured_within_budget` comes from this measurement only. The additive
   prediction is reported beside it so you can see how far the approximation
   was off for this model.

## 4. Tensor classes, patterns and floors

| class | pattern (lower-cased tensor name) | floor | why |
|---|---|---|---|
| token_embd | `^token_embd\.weight$` | q4_k | carries the whole vocabulary |
| output | `^output\.weight$` | q4_k | every logit passes through it |
| attn_qk | `\.attn_(q\|k\|qkv)\.weight$` | iq2_s | |
| attn_vo | `\.attn_(v\|output)\.weight$` | q3_k | value/output errors land directly on the residual |
| ffn_gate_up | `^blk\.\d+\.ffn_(gate\|up)\.weight$` | iq2_s | |
| ffn_down | `^blk\.\d+\.ffn_down\.weight$` | iq2_s | |
| expert_gate_up | `\.ffn_(gate\|up\|gate_up)_exps\.weight$` | iq2_s | most bytes in an MoE |
| expert_down | `\.ffn_down_exps\.weight$` | iq2_s | |
| shared_expert | `_shexp\.weight$` | q4_k | runs on every token |
| router | never changed | | llama.cpp keeps routers high precision on purpose |

Floors are conservative starting rules, recorded in every receipt. They can be
relaxed once a measured probe shows a class tolerating less; that is the
measurement doing the arguing, not the table.

The patterns are ECMAScript regexes because `llama-quantize` matches
`--tensor-type-file` entries with `std::regex_search` on the lower-cased name.
The Rust classifier (`TensorClass::of_name`) must agree with them; the unit
tests pin that on representative names.

## 5. The ladder and the byte model

```
iq2_s 2.50  < iq3_s 3.44 = q3_k 3.44 < iq4_xs 4.25 < q4_k 4.50 < q5_k 5.50 < q6_k 6.56 < q8_0 8.50  bits/weight
```

Bits per weight are nominal (block scales included) and are used only to rank
savings during selection. Real file sizes are read back from the produced GGUF
and reported.

## 6. Reading a receipt

```
recipe-receipt.json
  base.kld.mean               floor for every delta
  probes[].kld_delta_vs_base  measured cost per class and level
  recipe.steps[]              the greedy walk, in order, with bytes saved and KLD added
  recipe.predicted_kld_delta  additive prediction
  final.kld.mean              what was actually measured on the final file
  final.measured_within_budget  the only pass/fail that counts
  final.p99_within_budget     tail gate, when --budget-p99 was given
```

`recipe-apply` writes `recipe-applied-receipt.json`. Without `--verify-budget`
it says `"verification": "none"`; it never implies the transferred recipe was
measured on the new model.

## 7. Costs

A run performs `probes + 2` quantizations plus `probes + 2` KLD measurements on
the corpus slice. For a 0.6B model that is a few minutes; for a 39B-parameter
MoE with 31 probes it is hours and tens of GB of transient disk. Narrow it with
`--floor`, a higher `--base`, or fewer `--chunks`, and reuse the result on
sibling models with `recipe-apply`.

Measured on one test machine (RTX 4060 Ti 16 GB + Arc Pro B60 24 GB, 31 GiB
RAM, model on a ~400 MB/s drive): Flint-BF16 (74 GB, 39B MoE) loads in fit
mode in about 2.5 minutes with its experts mapped on the CPU, and the imatrix
pass over 21 chunks of 512 tokens reports an estimate of about 53 minutes. At
that rate a 20-quantization search is an overnight job; the limiting factor is
reading 74 GB from disk per quantization, not the GPUs.

## 8. Worked example (measured 2026-10-30)

Qwen3-0.6B BF16 (1.2 GB), 50 KB held-out corpus, 12 chunks of 512 tokens,
`--imatrix auto --base q5_k --floor iq2_s --budget 0.01 --budget-p99 0.05`,
public llama.cpp Vulkan tools, 21 quantizations in about 25 minutes.

| step | result |
|---|---|
| all-q5_k base, no imatrix (earlier run) | mean KLD 0.0438 |
| all-q5_k base, with imatrix | mean KLD 0.0234, 34.7% of source |
| cheapest probe: attn_qk at q4_k | +0.0092 |
| typical probes: token_embd q4_k +0.0127, ffn_down iq4_xs +0.0128, attn_vo iq4_xs +0.0177 |
| expensive: attn_qk iq2_s +0.188, ffn_gate_up iq2_s +0.391, attn_vo q3_k +0.115 |
| selected recipe | attn_qk -> q4_k, everything else q5_k |
| final, measured again | mean KLD 0.0326 (budget 0.0334), p99 0.194 (budget 0.0234+0.05), 33.8% of source, **within budget** |

What this teaches: a 0.6B model has almost no slack below q5_k at a 0.01
budget; the importance matrix alone was worth more than any single precision
step; and attention value/output and dense FFN down-projections are the classes
that punish low bits hardest here, which is the same shape the old `_M` rules
assumed but now with numbers attached. On a 39B MoE the expert classes carry
97% of the bytes and a different answer is expected; that is exactly why the
search measures instead of assuming.

## 9. What the public state of the art does, and what Ojon took from it (R&D 2026-10-30)

- **Unsloth Dynamic 2.0** picks a precision per *layer* (not per class), scores with
  mean KL divergence, and calibrates on a hand-curated 300K to 1.5M token
  chat-style set because wiki-style calibration overfits instruct models
  ([docs](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs),
  [blog](https://unsloth.ai/blog/dynamic-v2)). Taken: calibration and evaluation
  text are now split so no probe is measured on the text that built its imatrix,
  and the plan warns when the calibration text is under 1 MiB. Open: per-layer
  granularity multiplies probes by the layer count; a block-group split (first
  two, middle, last two) is the planned middle step.
- **Community KLD practice** (bartowski's
  [Sailor2 thread](https://huggingface.co/bartowski/Sailor2-20B-Chat-GGUF/discussions/1)):
  hold calibration samples out of the KLD set; llama.cpp keeps some KLD internals
  in fp16, so very small differences are noise; llama-gguf-optimize tunes recipes
  against measured KLD. Taken: the hold-out rule above, and the receipt keeps the
  95th/99th percentile so a tiny mean is not over-read.
- **"Accuracy is Not All You Need"** (Dutta et al., NeurIPS 2024,
  [arXiv](https://arxiv.org/html/2407.09141v1)): accuracy deltas of 0 to 2% hide
  answer *flips* of up to 13.6%; flips correlate with KL divergence at Spearman
  0.981. Taken: this is why the budget is on KLD and why no KLD cutoff is called
  "negligible loss" in a receipt; the task suite measures flips.
- **Diffusion quants** have no KLD equivalent in the wild; a self-published
  FLUX-family card measured Q8_0 at about 34.6 dB PSNR / 0.011 LPIPS and Q4_K at
  about 20.7 dB / 0.154 LPIPS against BF16
  ([card](https://huggingface.co/Harsha001/flux-3-image-klein-ROCmFPX)). Taken:
  `diffusion-parity` gates on PSNR with a 30 dB default and reports per-pixel
  tails; LPIPS needs a network and is out of the stdlib-only script by design.

## 10. Flint-BF16 (39B MoE) partial run, 2026-10-30

Stopped by the host after 8 of 15 probes (system memory pressure, not the
tool); the ledger in the output directory holds the finished probes and
`recipe-run ... --resume` continues. Settings: base q5_k, floor iq4_xs, 6
chunks, `--ngl auto`, imatrix from the calibration half of a 50 KB corpus.

| step | mean KLD |
|---|---|
| base, all q5_k (34.5% of source) | 0.1207 |
| token_embd q4_k | 0.1116 (below base: noise) |
| output q4_k | 0.1991 (+0.078, real) |
| attn_qk q4_k / iq4_xs | 0.1277 / 0.1219 |
| attn_vo q4_k / iq4_xs | 0.1432 / 0.1558 |
| ffn_gate_up q4_k / iq4_xs | 0.1150 / 0.1157 (within noise) |

Reading: with a 6-chunk slice and a 50 KB corpus the noise floor is about
0.01, the same size as the budget, which is exactly what the noise advisory
now reports. The two clear signals are the output head (do not lower it) and
the dense FFN gate/up (free to lower). The expert classes, where 97% of the
bytes are, had not been probed yet when the run stopped. Next run: 24 chunks
and a 1 MiB+ corpus from `corpus-build`, then resume.

## 11. Competition audit and next experiment — 2026-11-03

**Current verdict: experimental measured-search capability, NOT a benchmarked
leader in quality per byte.** The 0.6B worked example above is an actual
reported mean-KL result, but only 0.9 percentage points of BF16-source size
were saved relative to its all-Q5 imatrix base, while KL increased. It is not
a same-byte-budget quality victory. The Flint MoE search is incomplete.

| Reference | Demonstrated public method | Where Ojon currently stands |
| --- | --- | --- |
| [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md) | Q/IQ GGUF types, importance matrices and tensor overrides | Ojon uses its installed quantizer; compare against the *same* version and imatrix |
| [Unsloth Dynamic 2](https://www.unsloth.ai/blog/dynamic-v2) | Model-specific *per-layer* precision and large curated calibration sets; published vendor MMLU/KL comparisons | Ojon searches broad tensor classes and offers boundary-block refinement; per-block calibration and matched external scores remain open |
| [Intel AutoRound](https://github.com/intel/auto-round) | Learned post-training rounding, AutoScheme mixed bits and GGUF export | Ojon has a local adapter, not a native competing rounding solver |
| [ExLlamaV3 EXL3](https://github.com/turboderp-org/exllamav3) | Hessian-aware QTIP-derived quantisation, flexible BPW, separate low-bit inference runtime | Ojon has a local adapter; avoid confusing different engines with quantisation-only quality |
| [Other quantisation methods](https://huggingface.co/docs/transformers/v5.12.0/quantization/selecting) | GPTQ, AWQ, HQQ and sub-4-bit formats have distinct trade-offs | External published accuracy is context, not an Ojon result or an apples-to-apples victory |

### Required acceptance experiment

1. Validate the Ojon quantisation command and chosen CUDA/Vulkan backend
   against the selected `llama-quantize` executable before launching a costly
   model. The FrogNano guided plan previously discovered Vulkan despite a
   requested CUDA device (see `PRODUCT_DIFFERENTIATION.md`).
2. Freeze one untouched BF16 source, exact upstream quantizer, calibration
   corpus, *different* held-out corpus and task suite, and inference settings.
   Start with 0.6B-1B for pipeline proof; scale to one 4B-8B only when
   resources have been checked.
3. Run **same model, same byte budget**: plain llama.cpp, llama.cpp + common
   imatrix, Ojon class-search recipe, and Ojon adaptive output. Only then add
   AutoRound GGUF/Unsloth when conversion is actually supported.
4. Compare measured bytes/BPW, mean and 99th-percentile KL, held-out
   perplexity, coding/reasoning/tool/multilingual accuracy, output flips,
   real runtime load, tokens per second and peak RAM/VRAM. Record failures,
   hashes, backend revisions, elapsed quantisation time and uncertainty.
5. A result that fails its KL tail or capability tests remains experimental,
   even if a lower-mean-KL boundary-refined file appears attractive. Public
   claims need repeated independently reproducible measurements.

**Engineering audit fix:** the selected boundary-refined file's receipt
`mean_within_budget` and `p99_within_budget` must always describe that final
file, and a passing plain recipe must never be silently replaced with a
refined candidate that fails those same gates. This is tested in the
2026-11-03 implementation update.

## 12. CPU-only baseline and Ojon recipe comparison — 2026-11-03

This is an **executed, preliminary matched-inference comparison**, not a
benchmark leaderboard. We used the same Qwen3-0.6B F16 reference, the same
held-out text, the same installed llama-perplexity (llama.cpp b11575), and
zero GPU layers for all four inference loads. The evaluation was 8 chunks of
512 tokens with batch 512 and two CPU threads. Outputs were loaded and
evaluated for real; model quality beyond this corpus remains untested.

| Artifact | Bytes | Mean KL vs F16 (standard error) | 99th-pct KL | PPL(Q) | Same top-token % |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock llama.cpp Q4_K_M (new bounded CPU build) | 484,220,000 | 0.108343 ± 0.002959 | 0.588381 | 34.9166 | 82.941 |
| Stock Q5_K_M (previously converted) | 551,378,016 | 0.035708 ± 0.001185 | 0.206032 | 33.0673 | 91.029 |
| Ojon KLD-guided mixed recipe (previously converted) | 503,468,576 | 0.034154 ± 0.001179 | 0.196654 | 32.2711 | 90.784 |

The original F16 is 1,509,347,424 bytes. The evaluation tool's reference PPL
was 31.7700 for all candidates. The separately saved original reference
reported PPL 31.8639; use the **same candidate-side estimator** to compare
rows. Ojon's recipe is **8.69% smaller than stock Q5** with marginally lower
measured mean KL, but its same-top-token rate is slightly lower. The mean-KL
difference (0.001554) is smaller than twice the combined standard error:
**do not call this statistically established superiority**.

Important confound: the Ojon recipe used a calibration importance matrix;
the stock Q5 comparison was not built with that same matrix. Conversion
builds/provenance also differ. We have therefore demonstrated the *existing
Ojon workflow* can provide an attractive size/quality point, **not** that
Ojon's search method outperforms a stock quantizer with equal calibration
and target bytes. Final acceptance still requires held-out coding, reasoning
and tool-call tests, identical calibration and larger samples.

The fresh Q4 conversion ran CPU-side with `llama-quantize --max-buffer-size 32`,
processed all 311 tensors in **22.392 seconds**, and was loaded by the KL
evaluator. Measured peak process working set was approximately **1.6 GiB**;
the lowest observed free system RAM was approximately **2.0 GiB**, without
having to reserve 8 GiB. The *external test harness* stopped an earlier
attempt when other workloads reduced free RAM below its guard; it did not
evict or interrupt those workloads. This is **not** a claim that Ojon's
in-process memory monitoring has been validated.

SHA-256 (lowercase hex; use for verifying the exact test inputs/outputs):

```text
f16_source  ab9004daf660cd6a6ba1c07556e74fcceb2b756063ccce3f9c69d3a637b361cc
stock_q4    28b1c721c1e476d3e41c629d5993c0aaeca0f3b747afdbab667f5a4bc8e37665
stock_q5    35c54534401aff1fd9f12ff8ef804263780474d2d9a0ed5f3bf1253e411f2c5c
ojon_recipe 6746856986b19a7624a1a7de63bdbe2b047758a0fbbd9a7cb5569a28f31d7fbd
heldout     86e2e132492715d6bfb79fda94e9938a55445fd2e82eb36a397d95378b0f4f9d
```

The updated recipe quantizer asks the installed llama.cpp tool for an adaptive
32/64/128 MiB tensor slab according to live free RAM. For small original
GGUFs (up to 2 GiB), its separate RAM-admission calculation uses the source
size plus bounded slab and headroom rather than a blanket 4/8 GiB reserve.
It remains a conservative admission heuristic, **not a verified process
peak or an automatic operating-system memory reclamation mechanism**.

**Next controlled experiment:** regenerate both a stock quant and an Ojon
mixed recipe using the exact same original, tokenizer, imatrix, quantizer
build and calibration/held-out split; compare several equal-byte-budget
variants and a coding/reasoning/tool suite before a product claim.
