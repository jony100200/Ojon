# Ojon documentation

Ojon is a **pre-alpha, local-first Rust workstation** for model inspection, quantisation, post-training optimisation experiments and evidence-based comparisons. There is no stable downloadable release. A working CLI command or a green source-code CI run is not proof of quality preservation or GPU compatibility.

This page separates **instructions**, **quality rules**, **experiments** and **plans** so older research is not mistaken for a product capability.

## Start here

| Document | Purpose |
| --- | --- |
| [Project README](../README.md) | Mission, build/first-run commands, current high-level feature status |
| [User guide](../USER_GUIDE.md) | CLI commands, backend requirements and workflow examples; advanced and optional routes are included |
| [Quality strategy](../QUALITY_STRATEGY.md) | What counts as a valid comparison, rejection and promotion |
| [Measured recipe search](RECIPE_SEARCH.md) | Mixed-precision search protocol, reproducibility requirements and measured limitations |

## Model and behaviour evaluation

| Document | Scope |
| --- | --- |
| [Behaviour quality audit](BEHAVIOUR_QA.md) | Read-only coverage checks, evaluation receipt checks and remaining acceptance work |
| [Community-model challenge](COMMUNITY_MODEL_CHALLENGE.md) | Comparable screening receipts; explicitly **not** an independent release certificate |
| [Two-sided editing algorithm](EDITING_ALGORITHM_TWO_SIDED.md) | Experimental, opt-in projection method; successful edit math is not proof of useful behaviour change |
| [Heretic 2 research](HERETIC_2_RESEARCH.md) | Upstream research and follow-up gates; **not** a parity claim |

### Historical FrogNano editing experiments — November 2, 2026

These reports are preserved as dated evidence, **not current instructions or accepted releases**. No listed editing trial demonstrated a validated over-refusal improvement.

- [Initial original/edited refusal comparison](FROGNANO_REFUSAL_EVAL_2026_11_02.md)
- [Direct/blunt legitimate-request stress test](FROGNANO_DIRECT_BLUNT_2026_11_02.md)
- [OR-Bench-Hard matched screening](FROGNANO_ORBENCH_HARD_2026_11_02.md)
- [Quality-preserving edit pilot, rejected](FROGNANO_QUALITY_EDIT_V2_2026_11_02.md)
- [Two-sided activation trial, rejected](FROGNANO_TWO_SIDED_ACTIVATION_TEST_2026_11_02.md)

## Other model families and product planning

- [MiniMax H3 W4A4 / NVFP4](MINIMAX_H3.md) — optional experimental conversion adapters; no general video-generation or parity guarantee.
- [Product differentiation and release plan](PRODUCT_DIFFERENTIATION.md) — planning and historical positioning, **not** a list of shipped features.

## Project boundaries

- **Ojon:** quantise, optimise, preserve source identity, verify, package and compare model variants. Experimental behaviour edits remain explicitly gated.
- **Bunon:** model training, fine-tuning, distillation, training datasets and training-job lifecycle.
- **Twarok / compatible inference backends:** run and benchmark models.
- **Porikkha:** independent acceptance.

Ojon's own quality screening is useful but is not independent Porikkha sign-off. Source-code CI can establish compilation/tests; final GPU load, task retention, runtime performance and downloadable release status require separate reproducible receipts.

## Maintaining these documents

1. Keep **current usage commands** in the root README or user guide, not duplicated across experiment reports.
2. Keep **acceptance rules** in the quality strategy; cite them from experiments rather than rewriting conflicting criteria.
3. Keep negative and incomplete experiments as dated, immutable evidence. Correct a result with a labelled addendum rather than erasing its provenance.
4. Mark new capabilities as **implemented**, **experimentally tested**, **hardware-verified** or **release-accepted** only when the corresponding evidence exists.
5. Avoid private machine paths, keys, unpublished checkpoints or internal Google Drive documents in this public repository.

For development conventions see [AGENTS.md](../AGENTS.md), [Contributing](../CONTRIBUTING.md) and [Changelog](../CHANGELOG.md).
