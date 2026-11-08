# Measured mixed-precision recipe search

Ojon's recipe search asks: **How much can each group of tensors be compressed while staying within a measured divergence budget?**

This is a reproducible optimization workflow, not a new tensor encoder. It uses compatible quantization and inference backends.

## Workflow

1. **Source:** Start from an original F16/BF16 GGUF model.
2. **Reference:** Evaluate source-model token distributions on a fixed held-out corpus.
3. **Calibration:** Build an importance matrix using separate calibration text.
4. **Probe:** Quantize candidate tensor classes at progressively lower precisions.
5. **Selection:** Rank size savings against incremental KL divergence.
6. **Refinement:** Compare the selected recipe with boundary-block refinements.
7. **Validation:** Measure the final saved file again. Run task-level and hardware tests before promoting the result.

Tensor-class sensitivity differs by architecture; MoE experts, embeddings, attention and output heads do not necessarily tolerate the same precision.

## Quick example

```bash
# Plan only.
cargo run --release --locked --bin ojon -- recipe-plan model-f16.gguf output/ --corpus calibration.txt --eval-corpus heldout.txt --base q5_k --floor q4_k --budget 0.01
```

After reviewing the plan, run the corresponding `recipe-run` command with the same settings, `--resume`, and `--confirm`. The `auto-quant` command offers presets and recipe-library reuse.

## Metrics

**KL divergence (KLD)** compares the reference and candidate next-token probability distributions. The search records mean and tail divergence rather than judging quality from file size or perplexity alone.

Important receipt fields include:

| Field | Meaning |
| --- | --- |
| `base.kld.mean` | Reference divergence for the higher-precision starting recipe |
| `probes[].kld_delta_vs_base` | Measured perturbation cost per probe |
| `recipe.steps[]` | Selected precision reductions and estimated size savings |
| `final.kld.mean` | Measured divergence for the generated output |
| `final.measured_within_budget` | Whether the final measurement meets the mean budget |
| `final.p99_within_budget` | Tail-divergence gate, when configured |

The selected recipe is not guaranteed to be globally optimal. Interactions between tensor changes mean probe deltas are not perfectly additive; final remeasurement is essential.

## Fair comparison

When comparing against stock or community GGUF files, keep the original checkpoint, calibration/imatrix, evaluation prompts, backend settings and execution device comparable. Compare at a common byte budget or a common quality target.

Measure actual file bytes and task results. A smaller file with acceptable KL can still fail coding, tool use or other real tasks.

See [Quality strategy](../QUALITY_STRATEGY.md) for acceptance criteria and [User guide](../USER_GUIDE.md) for setup.
