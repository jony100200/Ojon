# The four proofs (owner's restatement, 2026-10-30)

Goal in the owner's words: Ojon makes the best quantized models for local use.
No censored models ("I cannot" is a defect); quants that beat the online ones at
the same size or match them smaller, measured; then fine-tune and quant again.

| # | Family | Model | Original | Online quants to beat | Ojon route | Gate | Status |
|---|---|---|---|---|---|---|---|
| 1 | LLM / VLM agent | Qwen3.8-27B (dense, vision, native tool calls; B60 24 GB home) | unsloth BF16 split 54.7 GB; mmproj 0.93 GB | unsloth UD-Q5_K_M 19.77 GB, UD-Q4_K_XL 17.56 GB | `recipe-run` from BF16 with imatrix, held-out KLD; then `uncensor-auto` (refusal suite + task suite) and re-quant; later `lora-merge` on harvested episodes | mean + p99 KLD at equal bytes; task suite; refusal rate | Q5_K_M downloading for the agent bake-off; BF16 queued |
| 2 | Image | FIBO 2.0 Dev (Bria, 8B flow-matching; Qwen3-VL-4B encoder, FLUX.2 VAE) | Comfy-Org bf16 16.45 GB | vantagewithai Q4_K_M 6.23 / Q5_K_M 7.03 / Q6_K 7.52 GB; molbal Q4_K_M 5.54 / Q6_K 6.96 / Q8_CR 8.41 GB | sd.cpp has no FIBO arch; route is ComfyUI-GGUF `tools/convert.py` + patched quantizer with a per-tensor recipe (new Ojon command), gated by `diffusion-parity` | fixed-seed PSNR vs bf16 at equal bytes | not started |
| 3 | Video | MiniMax H3 fl2va + ref2va (+ Music-3) | ~40 GB (owner) | | `minimax-*` W4A8 mixed, pruned | video parity (PSNR + temporal delta) | DONE today: 12.54 GB + 11.77 GB on disk, see ledger |
| 4 | Utility | On disk now (M:\LMStudio, D:\KSAppDev\Models): EmbeddingGemma-2 Q6_K 0.25 GB (+mmproj), nomic-embed-text-v1.5 Q4_K_M (Dolil's), deepseek-ocr2 q4_k 2.31 GB, GUI-Owl-1.5-2B Q5_K_M 1.26 GB, KV-Ground-4B Q5_K_M 2.89 GB / F16 7.5 GB, LiquidAI d1-omni-600M Q8_0, Julia-1 Q8_0 0.17 GB, Decision-2.0-Nox-4B Q4_K_M 2.72 GB, laya.onnx 0.43 GB, wd-tagger.onnx 0.47 GB, yolov10m / vitpose / deeplabv3p ONNX, stable-audio-3-small q8_0 x2 | mostly only quants on disk; originals needed per model | the community quant on disk | embedding: `embed-parity` (exists); grounding / OCR / decision / audio: gates to build (click accuracy, CER, Dolil decision benchmark, audio parity) | quality first: a utility quant ships only with its task metric; q8 files violate the q5/q6 ceiling and are re-quant candidates once a gate exists | inventory done, gates not built |

Agent candidates already on disk to include in the Dolil bake-off besides Qwen3.8-27B: gemma-4-26B-A4B (MoE, 4B active, vision, Q4_K_M 16.8 GB: likely 2-3x the decode speed), Gemma4-12B QAT, gpt-oss-20b MXFP4, Muse-Glimmer-30B (Terminal-Bench 51.7), Ornith-1.5-9B.

Measured so far today: Qwen3.8-27B Q4_K_M on the B60 (Vulkan2): pp512 562.6 t/s,
tg128 19.1 t/s. Every row closes only with a receipt in `MEASURED_RESULTS.md`.
