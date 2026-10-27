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

## Directional editing

An original single-file little-endian F16/BF16/F32 GGUF can be edited
directly with a measured activation direction. Ojon writes a separate GGUF
and recipe receipt while preserving the original. The supported projection targets are exact
`blk.N.attn_output.weight` and `blk.N.ffn_down.weight` float matrices.
Quantized, router, expert and vision tensors are excluded from this route.

The editor reports unmeasured refusal and capability results as
`NOT_EVALUATED`. A model is ready to quantize for use only after real load,
refusal, and capability tests against the original. Activation capture and
automatic strength selection are not provided by this route. Ojon can also
import a llama.cpp control-vector GGUF made from paired prompts; projecting
its residual-stream direction from one attention output matrix is an
experimental choice that requires the same validation.
