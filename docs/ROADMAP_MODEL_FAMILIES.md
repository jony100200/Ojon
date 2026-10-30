# Model-family coverage: what exists, what is missing (audit 2026-10-30)

Read-only audit of this repository at commit 76310d6. Every claim below is
anchored to a file; the ranked task list at the end is the order of work.
"Works" means a command exists and measures something; "detected only" means
the probe names the family and nothing downstream consumes it.

## Shared entry points

- `probe` (`src/cli/inspect.rs`, `src/core/probe.rs::probe_model`). `ModelCategory`
  already has `VisionLanguage`, `Diffusion`, `Speech`, `EmbeddingOrReranker`, but
  only LLM, VLM and Diffusion get a conversion route; Speech and Embedding
  always resolve to `no_wired_converter`. Accepted inputs: `.gguf`, `.safetensors`,
  `.ojonq`, HF folders, Diffusers folders. ONNX, OpenVINO IR, whisper.cpp GGML
  `.bin` and `.pt` are rejected at the door.
- `optimize-plan` (`src/workflows/options.rs`) advertises routes for text models
  only, on purpose. `optimize-workflow-*` is text-only.

## LLM (strong)

GGUF recipe search with measured KL divergence (`src/quant/recipe*.rs`,
`kld.rs`), hold-out corpus split, importance-matrix calibration (`calibration.rs`,
`corpus.rs`), boundary refinement, resumable runs, `recipe-apply`. Task suites
and benches exist beside it. See `RECIPE_SEARCH.md`.

## VLM

Works: detection by `vision_config` / tower keys; `quantize-hf` converts and
quantizes the language part (no mmproj); mmproj creation only inside
`model-build-run` at fixed Q8_0; a real vision suite gate in
`quant-parity-*` / `quant-compare-*` (`src/quant/quant_parity.rs`, suite
`ojon.vision_suite.v1`, accuracy drop bounded by `max_loss_points`); smoke via
`adapter-smoke-run --image --mmproj`.

Gap: the projector is never a quantization variable. The same `--mmproj` feeds
reference and candidate, and `quantize-hf` drops it. A GGUF mmproj probes as
`Language weak_guess` because `clip` is not in the VLM list.

## Image (diffusion)

Works: Diffusers/safetensors detection; `diffusion-gguf-plan|run` via `sd-cli -M
convert` (q8_0, q5_0, q4_0, q4_K, q3_K), receipt `FORMAT_VALID_NOT_RUNTIME_TESTED`;
real smoke generations (`sd-cli` PNG, ComfyUI); `diffusion-parity` measured gate
(PSNR, mean abs diff, pixels over tolerance, default 30 dB on the worst image);
SVDQuant INT4 and native `.ojonq` 2/3/4-bit packing (storage only, no loader).

Gap: no sweep. Conversion and parity are separate manual steps; the gate is one
prompt and one seed.

## Video

Works: `diffusion-gguf-*` with `--kind video` on standalone denoisers (WebM out,
EBML magic checked); MiniMax-H3 W4A4 / NVFP4 routes; `minimax-video-test-*`
(frames decodable, `visual_quality_verified: False`); parity only when the
workflow saves PNG frames.

Gap: no video-native parity (no frame extraction from MP4/WebM, no temporal
metric).

## Audio

Detected only, and the detection is wrong: the diffusion substring check runs
first and matches `"dit"` inside `ForConditionalGeneration`, so Whisper, T5 and
BART HF folders classify as Diffusion. `whisper-cli` is discovered but
`DiscoveryOnly`. No convert, quantize, quality or smoke command.

## Utility (OCR, embeddings, rerankers, detection, segmentation)

Detected only, by substring; BERT, XLM-RoBERTa (bge-m3) and Qwen3-Embedding
fall through to `Language weak_guess`. OpenVINO research route hardcodes
`text-generation-with-past`. `learn-utility-models` prints a static catalogue.
No embedding, reranker, ONNX, detection or segmentation command.

## Cross-cutting

- Commands dispatched but absent from `usage()` (`src/cli/common.rs`):
  `diffusion-parity`, `corpus-build`, `recipe-plan`, `recipe-run`,
  `recipe-apply`, `flint-compare-logits`, `flint-rco-assemble`, `learn-harvest`,
  `learn-utility-models`, `learn-package-list`.
- `README.md` links `docs/TECHNICAL_REFERENCE.md`, which does not exist.
- No TODO/FIXME markers in `src`; gaps live in receipt fields (`NOT_EVALUATED`,
  `NOT_RUN`), which is the honest convention and should stay.

## Ranked next tasks (user value / effort)

1. **Probe classification + usage hygiene.** Check Speech/Embedding before
   Diffusion or match `dit` as a token; add `clip` to the VLM list; add `bert`,
   `xlm-roberta`, `*Embedding*`, `*Reranker*` hints; probe GGML `.bin` headers;
   list the ten missing commands in usage; fix the README link.
   Done when unit tests classify `WhisperForConditionalGeneration` as Speech,
   `T5ForConditionalGeneration` as not Diffusion, an mmproj (`clip`) as VLM,
   bge-m3 and Qwen3-Embedding as Embedding, and `ojon --help` lists the commands.
2. **Diffusion precision sweep + multi-seed parity.** `diffusion-sweep-plan|run`:
   convert to q8_0, q5_0, q4_K, q4_0, q3_K; run the parity script over N prompts
   x M seeds; one receipt names the smallest format that passes `--min-psnr`, or
   none. Optional `sd-cli --tensor-type-rules` for sensitive blocks.
3. **Embedding / reranker gate.** `embed-parity-plan|run`: `quantize-hf` then
   `llama-embedding` on reference and candidate over a held-out set; min and
   mean cosine, top-k overlap, Recall@k; reranker mode on NDCG@10. Needs
   `BackendId::LlamaEmbedding` and better embedding classification.
4. **mmproj as a quantization variable.** `quantize-hf --mmproj-out
   --mmproj-precision f16|bf16|q8_0`; `quant-parity --mmproj-candidate`; discover
   `llama-mtmd-cli`. Done when (Q4_K_M + mmproj f16) vs (Q4_K_M + mmproj q8_0) are
   scored against the F16 reference on one vision suite.
5. **whisper.cpp quantize + WER gate.** `whisper-quant-run` (q8_0, q5_0, q4_0)
   and `whisper-parity-run` (held-out clips, WER vs reference transcripts and vs
   the original model, `--max-wer-delta`).
6. **Video temporal parity.** ffmpeg frame extraction in the parity script,
   per-frame PSNR (min, mean) plus a temporal-consistency delta;
   `diffusion-parity --kind video`.
7. (deferred) ONNX/OpenVINO utility routes beyond text generation, with cosine
   / IoU / mIoU gates. Highest effort: needs a new loader and eval harness.
