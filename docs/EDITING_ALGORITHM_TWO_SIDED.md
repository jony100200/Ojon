# Ojon model editing — measured two-sided projection (experimental)

**Status:** Implemented as an opt-in Rust GGUF/safetensors matrix edit; algebra and container tests pass. **No validated FrogNano improvement or Hugging Face release claim.**

## Problem addressed

The legacy edit projects a measured output direction **d** out of *every* input column:

`W' = W - alpha * d * (d^T W)`

Even when d is correct, this changes the behaviour of the output matrix for all inputs that activate that direction. The previous FrogNano strength-0.10, single-attention-block experiment preserved 11/12 development tasks but changed no OR-Bench-Hard refusal decisions. That experiment was rejected.

## New opt-in algorithm

Capture a **module-output residual direction d** and, independently, a **module-input activation direction v** from the same compatible transformer module. Both must be nonzero, finite and the correct dimensions. The editor normalises each.

Optionally supply **g**, a measured normal-behaviour output activation direction. Ojon removes its component from d:

`d_perp = normalize(d - (d dot normalize(g)) * normalize(g))`

The new two-sided edit is:

`s = d_perp^T W v`

`W' = W - alpha * s * d_perp * v^T`

If g is omitted, use the normalised original d. This rank-one operation has useful, mathematically testable local properties:

1. **Input preservation:** For any module input x satisfying `v dot x = 0`, `W' x = W x`.
2. **Good-output preservation:** When g is supplied, the edit delta is orthogonal to g: `g^T (W'-W) = 0`.
3. The stored GGUF and safetensors tensors are modified copy-on-write. Original files, other weights, tokenizer and MTP-related tensors remain unchanged.
4. The new method requires `preserve_row_norms=false`; row re-scaling after projection would break these exact invariants.
5. Data in d, v and g must be captured from real compatible module activations. Using unrelated invented vectors is a meaningless benchmark.

These are **single-matrix linear guarantees**, NOT whole-transformer intelligence-retention guarantees. Attention nonlinearities, residual accumulation, changing activation distributions and quantisation can still create unexpected differences. A KL-divergence evaluation and task benchmark are required.

## Recipe interface

Ojon's existing `ojon.abliteration.v1` recipe supports optional fields per edit:

- `direction`: an output-space control vector (or the existing `good_activations` and `bad_activations`).
- `protected_good_direction`: a separate measured ordinary-behaviour output direction to exclude from the edit.
- `input_direction`: an explicitly captured module-input direction, **or** both `input_good_activations` and `input_bad_activations` (matched input-space activation samples). Input and output widths must match the selected matrix.
- `strength`: alpha in (0,1].

Minimal **toy 2x2 test** example (the real model uses its actual tensor dimensions):

```json
{
  "schema": "ojon.abliteration.v1",
  "preserve_row_norms": false,
  "edits": [{
    "tensor": "blk.0.attn_output.weight",
    "direction": [1, 1],
    "protected_good_direction": [1, 0],
    "input_good_activations": [[0, 0], [0, 0]],
    "input_bad_activations": [[0, 1], [0, 1]],
    "strength": 0.25
  }]
}
```

Use `ojon abliterate-gguf-plan ...` or `ojon abliterate-gguf-run ... --confirm` with this recipe only when using a compatible *original floating-point GGUF* and real measured activations. Existing one-sided recipes continue to work unchanged.

## Validation completed

Rust tests now cover:

- GGUF two-sided edit with actual modified weights and preserved uncorrelated inputs;
- independently protected good-output direction;
- derivation of input direction from activation samples;
- zero/mismatched/nonnumeric vectors and conflicting input settings;
- rejection of row-norm preservation when it would invalidate guarantees;
- source container preservation; existing one-sided edits and safetensors functionality.

**What has NOT happened:** We have not yet captured matched FrogNano *module-input* activations for this feature; the previous cvector generator gives output residual directions, not matching module-input I/O. We did not manufacture an input direction or promote a new edited checkpoint. The necessary next milestone is a verified module-I/O extraction path, then a held-out experiment comparing legacy projection, two-sided projection and the untouched model on semantic answer usefulness, KL and coding/reasoning/tool benchmarks.

## Research references

- [Heretic projected direction and row normalisation implementation](https://github.com/p-e-w/heretic/blob/master/src/heretic/modifiers/abliteration.py)
- [Heretic configuration](https://heretic-project.org/configuration)
- [ICML 2025: concept cones and representational independence](https://proceedings.mlr.press/v267/wollschlager25a.html)

**Product ownership:** This is a prototype in Ojon; behaviour-changing algorithms belong in Model Baker as the product architecture matures. Quant Maker should only quantify quality/size and package verified variants.
