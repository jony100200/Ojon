# The four proofs (owner's restatement, 2026-10-30)

Goal in the owner's words: Ojon makes the best quantized models for local use.
No censored models ("I cannot" is a defect); quants that beat the online ones at
the same size or match them smaller, measured; then fine-tune and quant again.

| # | Family | Model | Original | Online quants to beat | Ojon route | Gate | Status |
|---|---|---|---|---|---|---|---|
| 1 | LLM / VLM agent | Qwen3.8-27B (dense, vision, native tool calls; B60 24 GB home) | unsloth BF16 split 54.7 GB; mmproj 0.93 GB | unsloth UD-Q5_K_M 19.77 GB, UD-Q4_K_XL 17.56 GB | `recipe-run` from BF16 with imatrix, held-out KLD; then `uncensor-auto` (refusal suite + task suite) and re-quant; later `lora-merge` on harvested episodes | mean + p99 KLD at equal bytes; task suite; refusal rate | Q5_K_M downloading for the agent bake-off; BF16 queued |
| 2 | Image | FIBO 2.0 Dev (Bria, 8B flow-matching; Qwen3-VL-4B encoder, FLUX.2 VAE) | Comfy-Org bf16 16.45 GB | vantagewithai Q4_K_M 6.23 / Q5_K_M 7.03 / Q6_K 7.52 GB; molbal Q4_K_M 5.54 / Q6_K 6.96 / Q8_CR 8.41 GB | sd.cpp has no FIBO arch; route is ComfyUI-GGUF `tools/convert.py` + patched quantizer with a per-tensor recipe (new Ojon command), gated by `diffusion-parity` | fixed-seed PSNR vs bf16 at equal bytes | not started |
| 3 | Video | MiniMax variant | done earlier (`minimax-*`, video parity) | | | temporal delta + PSNR | done |
| 4 | Utility | KV-Ground 4B (GUI grounding VLM, on disk F16 + Q5_K_M), Laya typed-decisions 421M, Verdict 151M GLiClass | | | KV-Ground through the mmproj path; Laya/Verdict through the ONNX/OpenVINO utility route (roadmap item) | task accuracy on Dolil's decision benchmark (`scripts/benchmark_real_engines.py`) | not started |

Measured so far today: Qwen3.8-27B Q4_K_M on the B60 (Vulkan2): pp512 562.6 t/s,
tg128 19.1 t/s. Every row closes only with a receipt in `MEASURED_RESULTS.md`.
