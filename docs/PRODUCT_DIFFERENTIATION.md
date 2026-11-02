# Ojon product differentiation and proof-driven release plan

Updated: 2026-11-02. This is a product and engineering plan, not a claim of benchmark superiority.

## Positioning

**Ojon — local model optimisation with measured, reversible results.**

Do not position Ojon as a replacement for Heretic 2.0 on refusal editing today. Ojon's current FrogNano Q4 screen found zero improved refusal decisions in a matched 26-case run; the original was already helpful on all 18 legitimate requests. A file named uncensored is not a measured improvement.

Build a local-first integrated workflow instead:

1. **One-click model intake**: HF folder, GGUF, architecture-aware compatibility preflight, hardware check, safe original preservation.
2. **One-click useful-answer evaluation**: direct and profane legitimate requests, controversial-but-permissible writing, cybersecurity education, coding and research; report both outright refusals and soft non-answers. Keep separate controls for requests where a refusal is appropriate.
3. **Model quality comparison**: before/after paired tasks, held-out over-refusal scoring, confidence bounds, actual compilation and tool-use tests, detailed changed-response inspection.
4. **Reversible variants**: retain original, stage candidate changes, expose attach/detach adapters where available and independently verified.
5. **Hardware-aware compression**: measure BF16 / Q4_K_M / Q5_K_M / Q6_K candidates on each device rather than promise a universal best precision. Display size, memory and throughput.
6. **Evidence-first shareable report**: model family, original/candidate digests, runtime settings, measured improvements and regressions, supported architecture, known limitations. A report is more credible than an unverified adjective.

A marketable feature is **model freedom without guesswork**: users choose trade-offs and can roll them back. Support maximally useful answers to legitimate requests rather than advertising universal removal of safeguards. A reduction in unnecessary refusals must be demonstrated on challenging legitimate prompts, not inferred from a phrase list.

## Research and competition

- Heretic 2.0 (2026-11-01) adds headless operation, plugins, modern editing methods and better thinking-model handling. https://github.com/p-e-w/heretic
- OR-Bench (ICML 2025) contains 80,000 over-refusal probes, a difficult subset and distinct toxic controls. https://proceedings.mlr.press/v267/cui25a.html
- Ojon's current module is explicitly a local proxy and is not full activation-trained ARA. https://github.com/jony100200/Ojon/blob/main/docs/HERETIC_2_RESEARCH.md

**Research gap:** integrate a defensible external over-refusal benchmark and human-reviewed cases, not more generic refusal/opening templates.

## Minimum release acceptance

- Genuine original and candidate runtime loads; same chat template and generation settings.
- Source family coverage and any unsupported paths disclosed.
- Matched before/after tasks and held-out legitimate-request prompts.
- Semantic refusal and answer-adequacy judgement, reviewed independently.
- Validated original/candidate/suite SHA-256 and nonempty source receipts.
- No claim of benefit when the original has no unnecessary refusals in the tested set.
- Explicit category-level confidence intervals and controls.
- Recorded memory, token throughput and artifact sizes.
- Atomic promotion only after evidence gates; allow rollback.

Ojon now exposes `paired-screen-audit` as a **read-only screening step**. It validates case IDs and aggregate totals in Ojon refusal-evaluation receipts, detects wording-only changes and always emits `publish_ready: false`. It does NOT establish semantic correctness or independently attest inference.

Example:

```powershell
ojon paired-screen-audit original-receipt.json edited-receipt.json --json
```

If output says `NO_BASELINE_OVERREFUSAL_TO_IMPROVE`, select a harder *legitimate* held-out set. Do not report an improvement where none was measured.

## Public repository launch ladder

**Milestone A: 1 compelling reproducible demonstration.** A 4–8B open model, raw legitimate requests it unnecessarily refuses before, useful answers after, no measurable loss on coding/tool tests, original and edited side-by-side, exact command and report. Do not post a speculative headline.

**Milestone B: usable first run.** Windows/Linux binaries, straightforward GUI, model-family preflight and a sample 5-minute workflow for a typical consumer GPU. If unsupported, state why rather than silently produce an unverified GGUF.

**Milestone C: credible launch assets.** README GIF and short video with real numbers; concise supported-family matrix; releases with checksums; CHANGELOG; test badges; issue templates; pinned demonstrator; end-to-end tutorial. Never claim a feature is complete on the basis of unit tests alone.

**Milestone D: distribution.** Announce a reproducible result to llama.cpp, local-model and developer communities using their contribution rules, publish an honest benchmark thread, respond to bug reports, and ship measurable improvements. GitHub Trending is an outcome, not a guarantee.

## Keep/refine/defer

- **Keep** local GGUF optimisation, quantisation, model inspection, quality reports and reversibility.
- **Refine** refusal editing into helpfulness-preserving model behaviour tuning with matched baselines and independent tests.
- **Defer marketing** any near-zero refusal percentage, universal architecture support or Heretic ARA parity until actual results support it.

## Hugging Face model factory — release lanes

**Two separately verifiable lanes; combine only after each passes.** The open-source tool and free models are the distribution engine; reproducible measurements are the product.

### Lane Q — smallest artifact with measured quality retention

1. Start from an untouched, high-precision source with pinned origin, tokenizer and checksum. Do not re-quantize a lossy GGUF as the primary source.
2. Partition representative calibration data for an importance matrix and a **distinct frozen holdout** for evaluation. Never choose the winning variant on the held-out set.
3. Produce Q6_K, Q5_K_M, Q4_K_M, IQ4_XS and Q3_K_M variants only where the current runtime actually supports them. Defer 2-bit extremes until demonstrated useful.
4. Compare all variants against original BF16 with measured byte count, runtime load, RAM/VRAM peak, prompt/gen tokens per second, perplexity delta, token/logit KL where supported, and task-level coding, reasoning, tools and multilingual accuracy. Verify vision and MTP separately for compatible models.
5. Select the **smallest point on the measured quality/size/throughput Pareto frontier**. Pre-register a quality tolerance (initial research target: <=1% relative perplexity regression and <=1 percentage point absolute accuracy regression on appropriately large held-out suites); never present a target as an observed result. If nothing passes, publish a larger variant or label the release experimental.
6. Compare any Ojon custom recipe with unmodified llama.cpp quantization at the same byte budget. An Ojon quant win requires an independently rerun result, not a smaller filename or a different evaluation dataset.

**Existing FrogNano starting point (not an acceptance result):** untouched BF16 8.07 GiB and original Q4_K_M 2.592 GiB; edited Q4_K_M also 2.592 GiB. The twelve-task development screen was 11/12 on both original and edited Q4; this is too small to prove BF16-to-Q4 parity. Existing CUDA build contains llama-quantize, llama-imatrix, llama-perplexity and llama-bench.

**First pilot preflight (2026-11-02):** Ojon's read-only quant-parity-plan accepted IQ4_XS,Q3_K_M against the existing Q4 reference, with explicitly selected D:\KSAppDev\tools\llama_cuda_b11575 binaries and CUDA0; no candidate was generated. The higher-level guided plan had auto-discovered **Vulkan** tools despite the requested CUDA device, so its suggested command must not be executed unmodified. The real runner probes runtime device IDs, but planning should expose backend mismatches early. Host free RAM was approximately 7.3 GB versus an 8.67 GB BF16 source, before conversion overhead; defer the heavy quantization run until enough RAM is free or an appropriately provisioned machine is selected. A passing minimum resource-policy flag is not a successful memory-fit test. Local plan output folder: E:\LMStudio\eval-reports\frognano-2026-11-02\hf_release_pilot.

References: https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md and https://github.com/ggml-org/llama.cpp/blob/master/tools/imatrix/README.md

### Lane B — model behaviour that is better, not merely different

1. Freeze representative legitimate-use and independently reviewed challenging-usage datasets. Include profanity, adult-fiction requests where permitted, direct contentious questions, research, creative writing and professional defensive topics; these are designed to reveal *unnecessary* refusals.
2. Record real baseline refusal/non-answer decisions and useful-answer accuracy. Sample several model/template settings as appropriate, noting when thinking is disabled.
3. Test Ojon, Huihui and Heretic as **distinct reference approaches**, with explicit model lineage. Huihui's Qwen3.5-4B card declares TRL fine-tuning and text-only changes; the model has modified/missing MTP weights. Heretic documents ARA/directional modifiers and hyperparameter search. Neither declaration proves equivalence to editing FrogNano's different source checkpoint.
4. Independently score **direct refusals, soft refusals, helpful answers, misinformation/errors, and capability regressions**. The current OR-Bench-Hard 40-case original/edited run was 30/40 versus 29/40 phrase hits, but the sole changed response was a non-answer; zero helpfulness recoveries verified.
5. Keep behaviour editing and quantization distinct: first show benefit at reference precision, **then** quantize the edited checkpoint and verify that the behaviour and capability results survive.
6. No model is labelled a successful edited release until human/independent evaluation finds a meaningful, repeated useful-answer improvement and the capability gates pass. Preserve original checkpoints and failure receipts.

References: https://huggingface.co/huihui-ai/Huihui-Qwen3.5-4B-abliterated and https://github.com/p-e-w/heretic and local FROGNANO_ORBENCH_HARD_2026_11_02.md.

### Public release structure

- **First:** verified quantized source model, separate from behaviour-edited variants. Launch only after larger held-out comparisons.
- **Second:** evidence-backed behaviour-edited source model, once it actually meets its gates.
- **Third:** quantized editions of the accepted edited model, with their own inference and quality verification.
- One Hugging Face repository per meaningfully different variant; use a Hugging Face Collection to group the related models, tool, reproducibility dataset and demo. Publish GGUF(s), verified SHA-256, runtime commands, the quantization settings, benchmark methodology, hardware, failures and original base_model linkage. For multimodal models include compatible projectors and tested vision configuration.
- Make a discoverable model card with a baseline-versus-release score table, real downloads, exact memory and processing figures. No invented rankings, no unverified "lossless" or "zero refusals" headline.

Hugging Face release guidance: https://huggingface.co/docs/hub/model-release-checklist

## Video-to-product series — ten linked episodes

Each episode should have a real command/demo, a raw measurable result or explicitly labelled failure, and links to the matching GitHub commit and Hugging Face artifact when published.

| Ep. | Working video topic | Proof shown / viewer payoff |
| --- | --- | --- |
| 01 | Why local models are too big, slow, or unhelpful | Actual hardware, GGUF source, bottleneck and baseline |
| 02 | Q4, Q5, Q6 and IQ: what changes? | Exact sizes, VRAM and token throughput on the same model |
| 03 | Importance matrices without guesswork | Calibration versus holdout; before/after quantization |
| 04 | How small before quality breaks? | PPL, KL and coding/tool regression versus size chart |
| 05 | Why an "uncensored" name proves nothing | FrogNano 26-case and hard-case failure, original versus edit |
| 06 | Huihui vs Heretic vs Ojon: method differences | Original cards/code, architecture and measured comparisons |
| 07 | Preventing fake benchmark wins | Soft refusal detection, reproducible hash/receipt audit |
| 08 | Quantizing on a normal 16 GB NVIDIA GPU | End-to-end local workflow and honest fit thresholds |
| 09 | Release an excellent Hugging Face model | Model card, quant variants, checksums, download and reproducibility |
| 10 | Community requests: your model on limited compute | Transparent model-intake, pricing, queue and what results customers receive |

Use a consistent series playlist and short clips made from the real graphs and examples. Video 05 can be produced **now as a transparent engineering failure analysis**; a success showcase must await a successful experiment.

## Sustainable free-and-supported business

- **Always free:** Ojon source, standard workflows, public evaluation scripts, model cards and verified public model builds.
- **Voluntary support:** GitHub Sponsors and Ko-fi for sustaining testing hardware, future models and documentation. Huihui's model card demonstrates a straightforward donation section; it does not prove donation revenue.
- **Paid custom-build service:** applicant supplies the exact base model, intended deployment hardware, constraints and a small clean calibration set. Deliver a quantized GGUF (or a supported variant), a reproducibility manifest, measured quality report and one correction window. Agree prices only after a real timing/cost pilot; do not promise impossible parity or universal hardware coverage.
- **Education:** practical courses, optional paid consulting and YouTube. Do not rely on ad revenue as the sole business. Validate demand with an enquiry form and a small number of finished pilot builds before scaling.
- **Feedback loop:** model downloads -> GitHub issues and community requests -> benchmark-driven improvements -> more useful releases and tutorials -> donations and custom commissions.

### Immediate release order

1. Run a **quant-only baseline competition** on FrogNano or another verified small open model using Ojon recipe and plain llama.cpp, at matched quality thresholds.
2. Build one honest model card + performance table for the winning quantization (only after validation).
3. Run the next controlled behaviour-editing experiment on reviewed legitimate requests, including a separately frozen holdout. Until a real gain appears, do not release FrogNano as proven modified.
4. Record Episodes 01–04 from actual quant measurements; Episodes 05–07 from the diagnostic experiment, not from unsupported promises.
5. Publish the initial free model, video and GitHub release together; invite specific custom-build requests with a clear deliverable.
