# EchoMimicV3-Flash — Inspection and Weight Packing

**Status:** Experimental weight-packing tool. The output is a structurally verified weight archive (`echo-manifest.json` + `.safetensors`), **not yet a proven end-to-end inference format**.

## Prerequisites

Point Ojon to a local EchoMimicV3-Flash weights directory containing the expected component files:

* `Wan2.1-Fun-V1.1-1.3B-InP/models_t5_umt5-xxl-enc-bf16.pth`
* `Wan2.1-Fun-V1.1-1.3B-InP/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth`
* `Wan2.1-Fun-V1.1-1.3B-InP/diffusion_pytorch_model.safetensors`
* `echomimicv3-flash-pro/diffusion_pytorch_model.safetensors`
* `Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth` (preserved reference)
* `chinese-wav2vec2-base/pytorch_model.bin` (preserved reference)

## Usage

Inspect eligible tensors and projected sizes without writing output:

```sh
ojon echo-inspect ./models/echo-flash --bits 8
ojon echo-inspect ./models/echo-flash --bits 4
```

Generate a dry-run conversion plan, then run conversion into a **new** output directory:

```sh
ojon echo-quant-plan ./models/echo-flash ./outputs/echo-int8 --bits 8
ojon echo-quant-run  ./models/echo-flash ./outputs/echo-int8 --bits 8 --confirm
```

Use `--bits 4` with a separate output directory for INT4 packing. `echo-quant-run` never modifies source files and requires an absent destination directory plus explicit `--confirm`.

## Output Format and Behavior

* **Rust-only reader & packer:** Reads `.pth`/`.bin` PyTorch ZIP checkpoints and `.safetensors` files directly without Python or GPU execution.
* **Layer selection:** Quantizes eligible 2D `F16`/`BF16`/`F32` linear matrices inside `blocks.*`, `encoder.block.*`, and `decoder.block.*` using symmetric per-row INT8 scaling or signed INT4 nibble packing with FP32 row scales.
* **Preserved precision:** Embeddings, norms, position biases, heads, time embeddings, audio projections, VAE, and wav2vec2 components remain untouched at original precision.
* **Outputs:** Writes `t5.safetensors`, `clip.safetensors`, `wan.safetensors`, `flash.safetensors`, and `echo-manifest.json` atomically via `.part` staging and post-write shape/dtype verification.

## Wan / Echo W4A8 Safetensors Alternative (`w4a8-plan` / `w4a8-run`)

To convert an individual `.safetensors` Wan or Echo Flash transformer checkpoint into ComfyUI's `asym_w4a8_int8` layout instead of the archive format above, use `--profile wan`:

```sh
ojon w4a8-plan ./models/wan-diffusion.safetensors ./outputs/wan-w4a8.safetensors --bits 4 --profile wan
ojon w4a8-run  ./models/wan-diffusion.safetensors ./outputs/wan-w4a8.safetensors --bits 4 --profile wan --confirm
```

`--profile wan` targets baseline `self_attn`, `cross_attn`, and `ffn` linear layers in `blocks.*` / `transformer_blocks.*` while keeping audio cross-attention projections at original precision.

## Limitations

* Outputs of `echo-quant-run` use `.ojon_scale` tensors and require a compatible quantized loader; standard `from_pretrained` / `load_state_dict` pipelines will not execute them directly.
* Valid container output does not imply runtime speed, VRAM reduction, or visual/audio quality parity without an independent inference evaluation.
