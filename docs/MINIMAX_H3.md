# MiniMax H3 — W4A4 and NVFP4 conversion guide

This guide documents the **public, optional** MiniMax H3 adapters in Ojon. They require open-source tools installed separately. Ojon does not download model weights, install GPU kernels, or provide private machine configuration.

## Choose the correct format

| Path | Input | Converter | Output | ComfyUI loader |
| --- | --- | --- | --- | --- |
| **True W4A4** | A complete original MiniMax H3 Diffusers model directory | [OrbitQuant](https://github.com/iamwavecut/OrbitQuant) | Packed **component directory** | [ComfyUI-OrbitQuant](https://github.com/iamwavecut/ComfyUI-OrbitQuant), after assembling a **complete release** |
| **NVFP4 weight storage** | Original unquantised MiniMax H3 single-file safetensors denoiser | [convert-to-quant](https://github.com/silveroxides/convert_to_quant) | One compressed safetensors checkpoint | Standard ComfyUI UNETLoader, if supported by the local runtime |

**NVFP4 is not equivalent to W4A4.** NVFP4 describes four-bit weight storage; its activation precision and native GPU support are runtime-dependent. OrbitQuant W4A4 uses a dedicated packed weight-and-activation runtime. A W4A4 component directory is not a stock ComfyUI safetensors file.

The current CLI is a source-first, pre-alpha interface. Structural conversion checks, mock ComfyUI protocol tests and compilation do **not** prove real GPU inference or retained image/video quality.

## Prerequisites

1. Build the Ojon CLI as described in the [README](../README.md).
2. Download the original supported checkpoint from its legitimate publisher. Check the license and expected model family.
3. Install the selected converter in a compatible Python environment. Consult upstream installation instructions for operating system, PyTorch, CUDA/XPU and kernel versions.
4. For inference, install ComfyUI and the required model components, including text encoder, visual VAE and audio VAE. OrbitQuant also needs its own ComfyUI extension and an appropriate *complete release*.
5. Install FFmpeg/ffprobe for independent video decoding verification. Without ffprobe, Ojon only confirms a saved video-like container, not decoded frames.

The public instructions use relative paths. Change them to match **your own** model installation; there is no required drive letter, username, workstation directory or private test bridge.

## W4A4: convert an H3 Diffusers component

Point Ojon to an existing local full Diffusers model directory with a model index, matching component configs and safetensors. The destination must not exist.

~~~sh
ojon minimax-w4a4-plan "./models/minimax-h3" "./outputs/w4a4-transformer" --component fl2va --device cuda
ojon minimax-w4a4-run "./models/minimax-h3" "./outputs/w4a4-transformer" --component fl2va --device cuda --confirm
~~~

Other supported selections: --component ref2va for the reference denoiser, and --component text_encoder for the Qwen-based encoder when supported by your installed upstream version. Requested devices: auto, cpu, cuda, cuda:N, xpu, xpu:N. Actual inference kernel support must be checked separately.

Ojon invokes the upstream OrbitQuant CLI with 4-bit weight and 4-bit activation settings, streaming staging and target-policy auto, then runs its artifact validator. If conversion fails, Ojon removes the temporary output directory and does not publish a successful artifact.

**Important:** this command produces an individually packed component, not a runnable full video release. To generate a video, assemble/obtain an OrbitQuant MiniMax H3 *complete release*, with matching transformer, text encoder, VAEs, manifests and the ComfyUI release configuration. The upstream [published H3 W4A4 release and evidence](https://huggingface.co/WaveCut/MiniMax-H3-OrbitQuant-W4A4) documents a working reference release; Ojon currently does **not** automate full release assembly.

## NVFP4: convert an original single-file denoiser

~~~sh
ojon minimax-checkpoint-plan "./models/h3-original.safetensors" "./outputs/h3-nvfp4.safetensors" --device cuda
ojon minimax-checkpoint-run "./models/h3-original.safetensors" "./outputs/h3-nvfp4.safetensors" --device cuda --confirm
~~~

The installed convert-to-quant CLI is invoked using its official MiniMax H3 layer policy, extra protected boundary layers, ComfyUI metadata and low-memory mode. The default uses upstream learned rounding **without** the faster --simple option; expect more compute and potentially more scratch memory. Ojon checks that the final output is compressed, has quantisation metadata and retains identifiable H3 tensors.

Upstream lists Blackwell support for native NVFP4 inference. A file that has been successfully packed may **not** run on an older NVIDIA card or Intel GPU without an alternate compatible kernel. Do not claim W4A4 arithmetic from this output.

## Run an actual video test

Export a **ComfyUI API-format** workflow. All required model components and custom nodes must already be installed in that local ComfyUI instance. The test calls ComfyUI through loopback HTTP only; it does not start ComfyUI or download any weights.

### Standard ComfyUI safetensors/NVFP4

Place the checkpoint in your configured ComfyUI diffusion-model search folder (often models/diffusion_models). The model must appear in UNETLoader's list. Use an API workflow with a UNETLoader and SaveVideo or VHS_VideoCombine.

~~~sh
ojon minimax-video-test-plan "./models/h3-nvfp4.safetensors" "./workflows/h3-api.json" "./results/nvfp4.mp4" "./results/nvfp4.json"
ojon minimax-video-test-run "./models/h3-nvfp4.safetensors" "./workflows/h3-api.json" "./results/nvfp4.mp4" "./results/nvfp4.json" --confirm
~~~

Add --model-name only when the loader's registered filename differs from the local filename. It is not a filesystem path.

### Native OrbitQuant W4A4 video release

Use an OrbitQuant **complete release directory** containing comfyui_orbitquant.json and all required components, plus an exported workflow using OrbitQuantReleaseLoader and OrbitQuantGenerateVideo. A bare OrbitQuant component from the W4A4 converter is insufficient.

~~~sh
ojon minimax-video-test-plan "./models/h3-w4a4-release" "./workflows/h3-orbitquant-api.json" "./results/w4a4.mp4" "./results/w4a4.json" --runtime orbitquant
ojon minimax-video-test-run "./models/h3-w4a4-release" "./workflows/h3-orbitquant-api.json" "./results/w4a4.mp4" "./results/w4a4.json" --runtime orbitquant --confirm
~~~

Public example workflow files live in [ComfyUI-OrbitQuant evidence](https://github.com/iamwavecut/ComfyUI-OrbitQuant/tree/main/evidence/minimax-h3/workflows). Use the exact frame counts, steps and input constraints supported by your installed upstream node. The upstream OrbitQuant H3 generator has different constraints from simple 8-step turbo variants; do not blindly apply one recipe to another.

Both test commands accept --server http://127.0.0.1:8188, --python, --node-id, --timeout and --ffprobe. Existing output/receipt files are never overwritten. Their parent directories must already exist.

### Evidence and limitations

- **VIDEO_DECODED_NOT_QUALITY_VERIFIED:** ComfyUI returned a saved video and ffprobe successfully decoded at least two video frames.
- **VIDEO_SAVED_DECODE_UNVERIFIED:** ComfyUI returned a saved video container, but ffprobe was unavailable.
- **Failure:** missing loader, invalid workflow, execution error, no matching video output, invalid container, decoding failure or timeout. No successful receipt is written.

Neither success status means the quantisation preserves video quality. For a fair comparison record the original checkpoint identifier/hash, exact component versions, runtime and converter versions, prompt, seed, guidance, sampler, steps, frame count, resolution, VAE versions and hardware. Compare motion, temporal coherence and details on matched prompts. If the original model cannot run, report **quality not compared**, not estimated parity.

## Contributor verification

~~~sh
cargo check --locked --bin ojon
cargo test --locked --lib --bin ojon
python -m unittest discover -s scripts -p "test_comfy_safetensors_video_smoke.py" -v
~~~

The Python tests use a fake ComfyUI HTTP service and synthetic files. They validate protocol and error handling **only**, never real GPU video generation.

References: [OrbitQuant](https://github.com/iamwavecut/OrbitQuant), [ComfyUI-OrbitQuant](https://github.com/iamwavecut/ComfyUI-OrbitQuant), [convert-to-quant](https://github.com/silveroxides/convert_to_quant), [ComfyUI](https://github.com/Comfy-Org/ComfyUI).
