# FrogNano 4B — refusal and over-refusal comparison (2026-11-02)

**Decision: NO OBSERVED REFUSAL REDUCTION — exploratory, not an acceptance benchmark.**

Ojon compared a new Q4_K_M quantisation of untouched FrogNano BF16 with the existing edited Q4_K_M checkpoint. Both were run on RTX 4060 Ti using KS llama.cpp CUDA b11575 (48e985756), CUDA0, 99 GPU layers, 2048 context, 256 output tokens, temperature 0, seed 42, and enable_thinking=false. Each used the same persistent loopback server and suite.

| Test | Original Q4 | Edited Q4 |
| --- | ---: | ---: |
| Bundled suite: four prompts labelled harmful | 0/4 refused | 0/4 refused |
| Bundled suite: four benign controls | 0/4 refused | 0/4 refused |
| Better-labelled screen: 18 legitimate requests | **0/18 refused** | **0/18 refused** |
| Better-labelled screen: eight explicitly malicious requests | **8/8 refused** | **8/8 refused** |
| Changed refusal decisions in 26 paired cases | — | **0/26** |
| Exactly identical generated text | — | **3/26** |

The other 23 responses differed in text, but this does not show a refusal-behaviour improvement. The bundled eight-question suite incorrectly labels ordinary educational cybersecurity questions as harmful, so its harmful-refusal metric is not meaningful.

All scores use Ojon's bounded phrase matcher, NOT an independent semantic refusal judge. It can miss indirect refusals and count quoted phrases. The 26-prompt suite is a small investigator-authored sample, not independently annotated. These percentages are not proof of a general 0% over-refusal rate or verified safety behaviour. Porikkha independent verification: NOT_RUN.

## Local evidence (raw output is not committed)

Folder: <private-evaluation-vault>/frognano-2026-11-02/

- Original model: FrogNano-4B-Q4_K_M-original.gguf, SHA-256 0cb5b755f5bd982ad09410d3544df094d0c89d13770476dd229053c21b3a2bb8.
- Edited model: FrogNano-4B-Q4_K_M-uncensored.gguf, SHA-256 5e75e8bb5037fc957baba59362ad2022dbe511cb8962269ff36f4ab667bb9439.
- Suite: refusal_screen_26.json, SHA-256 bf3b3d721efb6d7f78aba8b7764db57ce1c1f4d2a60b831e91f45bf5b1e27ea7.
- Original receipt: refusal_q4_cuda_original_26.json, SHA-256 8b53988293c64a89fed7c0c4ecb7796661300f864e47edb3cd0b56fc295305a6.
- Edited receipt: refusal_q4_cuda_edited_26.json, SHA-256 3dd8f6fb54178bf07c26d352f983b3db5f4c5f115b9f3bc6c0ab73a92f4da96b.
- Paired summary: refusal_comparison_summary.json.

## Next release gate

Require independently reviewed labels, hundreds of held-out legitimate prompts across categories, separate behaviour controls, semantic refusal/non-answer judging, matched runtime settings, uncertainty estimates, provenance validation, and separate capability-retention tests. Keep the original checkpoint intact. Until these gates pass, the edited FrogNano is a test candidate and **must not be advertised as having a demonstrably improved refusal rate**.

Mayamona should learn the distinction between weight changes, changed text, changed refusal decisions, and a statistically supported quality improvement.
