# Quality strategy

Ojon chooses the smallest **runnable** model that meets explicit quality
thresholds on the intended local hardware. It never treats a successful
conversion, a valid GGUF header, or a zero-score evaluation reference as
proof of retained capability.

## Model source and reference

- Use an original F16/BF16/F32 GGUF as the quantization source. Never
  requantize an existing Q4/Q5 file to make another candidate.
- The original GGUF can be its own quality reference. A matching, already
  accepted Q4 GGUF can instead be the reference when searching below Q4.
- Compare already available GGUFs without an original through
  `quant-compare-*`. Tensor agreement checks format compatibility, not
  cryptographic checkpoint lineage.
- Start with Q4_K_M and Q5_K_M. Search lower precision only when task,
  perplexity, size, load and hardware measurements justify it.

## Acceptance

1. Use the same prompts and inference settings for teacher and candidates.
   The teacher must score above zero for coding, reasoning and tool-call
   categories; otherwise the suite is uninformative and no model is chosen.
2. Set maximum tolerated loss and minimum size saving explicitly. Include
   vision tests for VLMs and a held-out perplexity corpus when required.
3. Confirm model load, tokens, speed and memory with the target inference
   backend and hardware. A GGUF container inspection is a preflight only.
4. Select the smallest candidate that passes every required gate. Report the
   exact evidence scope. “No detected regression on this suite” never means
   mathematically lossless.

## Directional editing and automatic uncensoring

An original single-file little-endian F16/BF16/F32 GGUF can be edited
directly with a measured activation direction. Ojon writes a separate GGUF
and recipe receipt while preserving the original. The supported projection targets are exact
`blk.N.attn_output.weight` and `blk.N.ffn_down.weight` float matrices.
Quantized, router, expert and vision tensors are excluded from this route.

Through `uncensor-auto-plan` / `uncensor-auto-run` (and the Kathamo GUI **Uncensor Model** workflow), Ojon automatically:
1. Captures paired refusal-vs-compliance directions across transformer blocks using `llama-cvector-generator` and ranks candidate layers by measured direction L2 norm.
2. Evaluates multiple editing strategies independently (`norm_preserving_ablation`, `standard_ablation`, and Heretic-style `projected_multi_layer_heretic` combining attention output and FFN down-projections across top-ranked middle blocks).
3. Measures both harmful-probe refusal reduction (`refusal_rate`) and benign-prompt false-refusal rate (`benign_overrefusal_rate`), alongside held-out coding/reasoning/tool-use capability retention.
4. Exports a detachable `<stem>-control-vector.gguf` alongside the winning edited high-precision checkpoint and reloads the published checkpoint to verify container integrity and real runtime inference.
5. When post-edit quantisation is enabled (`--quantize Q4_K_M|Q5_K_M|Q6_K`), runs the combined workflow (`Original -> Edited Teacher -> Quantised Candidate`) and verifies the final quantised artifact against **both** the original model (`vs_original`) and the accepted edited teacher (`vs_edited_teacher`).

The automatic selector requires nonzero baseline coding, reasoning and tool
scores and a measured reduction in harmful refusals or benign over-refusals.
It publishes no edit when the baseline cannot support those comparisons. A
quantised file is reported as the final output only when its dual-reference
gate passes; a rejected file remains available for inspection with a failed
status in the receipt.

The edit sweep evaluates candidates sequentially and retains at most the
current best plus the candidate under test. This bounds extra full-size GGUF
storage to two copies during the sweep.
