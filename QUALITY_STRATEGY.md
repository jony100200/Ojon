# Quality strategy

Ojon optimizes for **usable model quality per byte**. File size alone does not determine the best candidate.

## Comparison protocol

1. **Same source:** Record the original checkpoint and its hash. Compare candidates derived from the same model revision.
2. **Fair quantization:** Use comparable converter versions and calibration settings, including the same importance matrix when applicable.
3. **Independent evaluation:** Keep calibration and held-out evaluation datasets separate.
4. **Quality evidence:** Measure KL divergence/perplexity plus task-level results for the intended uses (for example, coding, reasoning, tool use, Bengali and English).
5. **Hardware evidence:** Confirm actual loading, memory demand and throughput on the intended runtime/device.
6. **Decision:** Retain the smallest candidate that meets declared quality and runtime requirements. Otherwise report no passing candidate.

A GGUF header check is not an inference test. A KL threshold alone is not proof of task accuracy.

## Receipts

A reproducible experiment should record:

- Source model identity and relevant file hashes
- Model architecture and tensor/quantization recipe
- Quantizer and runtime versions
- Calibration and evaluation dataset identities
- Output file bytes and measured distribution metrics
- Task-level scores and any known regressions
- Hardware, execution settings, memory and throughput when measured
- Decision: accepted, rejected or incomplete

## Model editing

Behaviour-editing experiments require a separate before/after comparison to the untouched original model. Quantizing an edited checkpoint is a second transformation and must be evaluated separately.

Preserve both references in the resulting report. Avoid claiming behavioral improvement unless the change is demonstrated on appropriate held-out requests without material capability loss.

## Current scope

The development experiments show working quantitative tools but have not demonstrated a general head-to-head victory against leading community quantizations. Acceptance evidence is model-specific and backend-specific.

See [the user guide](USER_GUIDE.md) for commands and [recipe search](docs/RECIPE_SEARCH.md) for the quantization method.
