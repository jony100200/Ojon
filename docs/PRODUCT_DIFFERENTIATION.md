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
