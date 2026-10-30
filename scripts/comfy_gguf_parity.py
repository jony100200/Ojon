#!/usr/bin/env python3
"""Measured image parity gate for diffusion GGUF quants, through a local ComfyUI.

The language-model side of Ojon gates a quant on KL divergence against the
original. Image models need the equivalent: run the SAME API-format workflow
twice, once with the reference weights and once with the candidate GGUF, same
seed, same prompt, same steps, and compare the pixels that come out. This
script does that and writes a receipt. It never quantizes, never downloads
weights, and only talks to a loopback/private ComfyUI.

What is measured (per output image, then averaged):
  * PSNR in dB over 8-bit RGB. Identical images are reported as "inf".
  * mean absolute pixel difference (0..255).
  * fraction of pixels whose max channel difference exceeds --pixel-tolerance.

Gate: --min-psnr (default 30 dB) on the worst image or frame. With --kind video,
MP4/WebM/GIF outputs are decoded to frames with ffmpeg (every decoded frame once,
so frame N lines up across the two runs) and a temporal-consistency delta is
reported: the mean absolute difference between the reference's and the
candidate's frame-to-frame motion. Flicker and blur show up there even when
single-frame PSNR looks fine; --max-temporal-delta gates it.

PNG decoding is done here with the standard library only (zlib + the five PNG
filters), for 8-bit RGB/RGBA/grey non-interlaced images, which is what ComfyUI's
SaveImage writes.
"""
import argparse
import copy
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import uuid
import zlib
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

LOADER_CLASSES = ("UnetLoaderGGUF", "UnetLoaderGGUFAdvanced", "UNETLoader")
SAMPLER_SEED_KEYS = ("seed", "noise_seed")


def fail(message):
    print(json.dumps({"ok": False, "error": message}))
    sys.exit(1)


def local_url(raw):
    parsed = urlparse(raw)
    host = parsed.hostname or ""
    if parsed.scheme not in ("http", "https") or host not in ("127.0.0.1", "localhost", "::1") and not (
        host.startswith("192.168.") or host.startswith("10.") or host.startswith("172.")
    ):
        fail("ComfyUI server must be a loopback or private-network URL")
    return raw.rstrip("/")


def request_json(server, path, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = Request(server + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urlopen(req, timeout=60) as response:
            return json.loads(response.read().decode("utf-8"))
    except URLError as e:
        fail(f"ComfyUI request {path} failed: {e}")


# ----------------------------------------------------------------- PNG decode

def _paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def decode_png(data):
    """Return (width, height, channels, bytes) for an 8-bit non-interlaced PNG.

    Channels is 1 (grey), 2 (grey+alpha), 3 (RGB) or 4 (RGBA); the pixel bytes
    are row-major without filter bytes.
    """
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos = 8
    width = height = None
    channels = None
    idat = []
    while pos + 8 <= len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if kind == b"IHDR":
            width, height, depth, color, _comp, _filt, interlace = struct.unpack(">IIBBBBB", body)
            if depth != 8:
                raise ValueError(f"unsupported PNG bit depth {depth}")
            if interlace != 0:
                raise ValueError("interlaced PNG unsupported")
            channels = {0: 1, 2: 3, 4: 2, 6: 4}.get(color)
            if channels is None:
                raise ValueError(f"unsupported PNG colour type {color}")
        elif kind == b"IDAT":
            idat.append(body)
        elif kind == b"IEND":
            break
    if width is None or not idat:
        raise ValueError("PNG missing IHDR or IDAT")
    raw = zlib.decompress(b"".join(idat))
    stride = width * channels
    out = bytearray(stride * height)
    prev = bytearray(stride)
    src = 0
    for y in range(height):
        f = raw[src]
        src += 1
        line = bytearray(raw[src:src + stride])
        src += stride
        if f == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 255
        elif f == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 255
        elif f == 3:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 255
        elif f == 4:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                up_left = prev[i - channels] if i >= channels else 0
                line[i] = (line[i] + _paeth(left, prev[i], up_left)) & 255
        elif f != 0:
            raise ValueError(f"bad PNG filter {f}")
        out[y * stride:(y + 1) * stride] = line
        prev = line
    return width, height, channels, bytes(out)


def to_rgb(width, height, channels, pixels):
    """Drop alpha, expand grey, so every comparison is on 3 channels."""
    if channels == 3:
        return pixels
    out = bytearray(width * height * 3)
    n = width * height
    for i in range(n):
        if channels == 4:
            out[i * 3:i * 3 + 3] = pixels[i * 4:i * 4 + 3]
        elif channels == 1:
            out[i * 3:i * 3 + 3] = bytes((pixels[i],)) * 3
        else:  # grey + alpha
            out[i * 3:i * 3 + 3] = bytes((pixels[i * 2],)) * 3
    return bytes(out)


# ------------------------------------------------------------------- metrics

def compare_rgb(a, b, width, height, pixel_tolerance):
    if len(a) != len(b):
        raise ValueError("images differ in size")
    n = width * height
    sq = 0
    abs_sum = 0
    bad = 0
    for i in range(n):
        worst = 0
        for c in range(3):
            d = a[i * 3 + c] - b[i * 3 + c]
            if d < 0:
                d = -d
            sq += d * d
            abs_sum += d
            if d > worst:
                worst = d
        if worst > pixel_tolerance:
            bad += 1
    mse = sq / (3 * n)
    psnr = math.inf if mse == 0 else 10 * math.log10(255 * 255 / mse)
    return {
        "psnr_db": psnr,
        "mean_abs_diff": abs_sum / (3 * n),
        "pixels_over_tolerance_frac": bad / n,
        "width": width,
        "height": height,
    }


# ------------------------------------------------------------------ ComfyUI

def loader_nodes(workflow):
    return [k for k, v in workflow.items() if isinstance(v, dict) and v.get("class_type") in LOADER_CLASSES]


def fixed_seed(workflow):
    """Every sampler must carry an explicit integer seed, or the two runs are not comparable."""
    seeds = []
    for key, node in workflow.items():
        inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
        for sk in SAMPLER_SEED_KEYS:
            if sk in inputs:
                value = inputs[sk]
                if not isinstance(value, int):
                    raise ValueError(f"node {key} has a non-integer {sk}; parity needs a fixed seed")
                seeds.append((key, value))
    if not seeds:
        raise ValueError("workflow has no sampler seed; parity needs a fixed seed")
    return seeds


def patch_model(workflow, node_id, model_name):
    patched = copy.deepcopy(workflow)
    node = patched[node_id]
    node.setdefault("inputs", {})
    node["inputs"]["unet_name"] = model_name
    return patched


def run_once(server, workflow, timeout_s):
    queued = request_json(server, "/prompt", {"prompt": workflow, "client_id": str(uuid.uuid4())})
    identifier = queued.get("prompt_id") if isinstance(queued, dict) else None
    if not identifier:
        fail("ComfyUI rejected the workflow (missing components or invalid inputs)")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        history = request_json(server, "/history/" + identifier)
        if isinstance(history, dict) and identifier in history:
            record = history[identifier]
            status = record.get("status", {}) if isinstance(record, dict) else {}
            if status.get("status_str") == "error":
                fail("ComfyUI history reports execution error")
            outputs = record.get("outputs") if isinstance(record, dict) else None
            if outputs:
                return outputs
        time.sleep(1.0)
    fail("ComfyUI did not finish within the timeout")


VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov", ".gif")


def extract_frames(media_bytes, suffix, ffmpeg):
    """Decode a video container to PNG frames with ffmpeg. Returns [(name, png_bytes)].

    `-vsync 0` keeps every decoded frame once (no duplication to a constant
    rate), so frame N of the reference lines up with frame N of the candidate
    when both came from the same workflow.
    """
    if not ffmpeg:
        raise ValueError("video output needs ffmpeg; pass --ffmpeg or put it on PATH")
    with tempfile.TemporaryDirectory(prefix="ojon-frames-") as tmp:
        src = Path(tmp) / ("clip" + suffix)
        src.write_bytes(media_bytes)
        pattern = str(Path(tmp) / "frame_%05d.png")
        proc = subprocess.run([ffmpeg, "-v", "error", "-nostdin", "-i", str(src), "-vsync", "0", "-f", "image2", pattern],
                              capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise ValueError(f"ffmpeg failed to decode {suffix}: {proc.stderr.strip()[:400]}")
        frames = sorted(Path(tmp).glob("frame_*.png"))
        if not frames:
            raise ValueError("ffmpeg produced no frames")
        return [(f.name, f.read_bytes()) for f in frames]


def fetch_media(server, outputs, kind, ffmpeg=None):
    """PNG outputs for images; for video, PNG frames (SaveImage) or decoded containers."""
    images = []
    categories = ("images",) if kind == "image" else ("images", "gifs", "videos")
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for category in categories:
            for item in node_output.get(category, []):
                name = item.get("filename", "")
                lower = name.lower()
                is_png = lower.endswith(".png")
                is_video = lower.endswith(VIDEO_EXTENSIONS)
                if not (is_png or (kind == "video" and is_video)):
                    continue
                params = urlencode({"filename": name, "subfolder": item.get("subfolder", ""), "type": item.get("type", "output")})
                with urlopen(server + "/view?" + params, timeout=120) as response:
                    data = response.read()
                if is_png:
                    images.append((name, data))
                else:
                    suffix = Path(name).suffix.lower()
                    images.extend((f"{name}#{fname}", fb) for fname, fb in extract_frames(data, suffix, ffmpeg))
    if not images:
        if kind == "video":
            fail("no video or PNG-frame outputs found in the workflow result")
        fail("no PNG outputs; a video workflow must use --kind video or export PNG frames")
    images.sort(key=lambda x: x[0])
    return images


def fetch_pngs(server, outputs):
    return fetch_media(server, outputs, "image")


def temporal_profile(frames_rgb, width, height):
    """Mean absolute pixel change between consecutive frames: the motion signature."""
    n = width * height * 3
    out = []
    for a, b in zip(frames_rgb, frames_rgb[1:]):
        out.append(sum(abs(x - y) for x, y in zip(a, b)) / n)
    return out


def temporal_delta(ref_profile, cand_profile):
    """How differently the candidate moves: mean |ref change - cand change| per step.

    A quant that blurs or flickers shows up here even when single-frame PSNR
    looks acceptable: flicker raises the candidate's change where the reference
    is still, blur lowers it where the reference moves.
    """
    if len(ref_profile) != len(cand_profile) or not ref_profile:
        return None
    return sum(abs(r - c) for r, c in zip(ref_profile, cand_profile)) / len(ref_profile)


def parity(reference_pngs, candidate_pngs, pixel_tolerance, temporal=False):
    if len(reference_pngs) != len(candidate_pngs):
        raise ValueError(f"reference produced {len(reference_pngs)} PNGs, candidate {len(candidate_pngs)}")
    per_image = []
    ref_frames, cand_frames = [], []
    size = None
    for (rn, rb), (cn, cb) in zip(reference_pngs, candidate_pngs):
        w, h, ch, px = decode_png(rb)
        w2, h2, ch2, px2 = decode_png(cb)
        if (w, h) != (w2, h2):
            raise ValueError(f"{rn} vs {cn}: size mismatch")
        r_rgb, c_rgb = to_rgb(w, h, ch, px), to_rgb(w2, h2, ch2, px2)
        m = compare_rgb(r_rgb, c_rgb, w, h, pixel_tolerance)
        m["reference"] = rn
        m["candidate"] = cn
        per_image.append(m)
        if temporal:
            if size is None:
                size = (w, h)
            elif size != (w, h):
                raise ValueError("frames change size within the clip")
            ref_frames.append(r_rgb)
            cand_frames.append(c_rgb)
    finite = [m["psnr_db"] for m in per_image if math.isfinite(m["psnr_db"])]
    summary = {
        "images": len(per_image),
        "min_psnr_db": min((m["psnr_db"] for m in per_image), default=math.inf),
        "mean_psnr_db": (sum(finite) / len(finite)) if finite else math.inf,
        "mean_abs_diff": sum(m["mean_abs_diff"] for m in per_image) / len(per_image),
        "max_pixels_over_tolerance_frac": max(m["pixels_over_tolerance_frac"] for m in per_image),
    }
    if temporal and size is not None:
        rp = temporal_profile(ref_frames, *size)
        cp = temporal_profile(cand_frames, *size)
        summary["frames"] = len(ref_frames)
        summary["temporal_delta"] = temporal_delta(rp, cp)
        summary["reference_mean_motion"] = (sum(rp) / len(rp)) if rp else 0.0
        summary["candidate_mean_motion"] = (sum(cp) / len(cp)) if cp else 0.0
    return summary, per_image


def run(args):
    server = local_url(args.server)
    workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
    if not isinstance(workflow, dict) or not workflow:
        fail("workflow must be an API-format JSON object keyed by node ID")
    loaders = loader_nodes(workflow)
    node_id = args.node_id or (loaders[0] if len(loaders) == 1 else None)
    if node_id not in workflow:
        fail("could not identify the single model loader node; pass --node-id")
    try:
        seeds = fixed_seed(workflow)
    except ValueError as e:
        fail(str(e))
    reference_outputs = run_once(server, patch_model(workflow, node_id, args.reference_name), args.timeout)
    candidate_outputs = run_once(server, patch_model(workflow, node_id, args.candidate_name), args.timeout)
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    try:
        ref_media = fetch_media(server, reference_outputs, args.kind, ffmpeg)
        cand_media = fetch_media(server, candidate_outputs, args.kind, ffmpeg)
        summary, per_image = parity(ref_media, cand_media, args.pixel_tolerance, temporal=(args.kind == "video"))
    except ValueError as e:
        fail(str(e))
    passed = summary["min_psnr_db"] >= args.min_psnr
    temporal_ok = None
    if args.kind == "video" and args.max_temporal_delta is not None:
        td = summary.get("temporal_delta")
        temporal_ok = td is not None and td <= args.max_temporal_delta
        passed = passed and temporal_ok
    receipt = {
        "schema": "ojon.diffusion_parity.v1",
        "ok": True,
        "passed": passed,
        "server": server,
        "workflow": str(args.workflow),
        "loader_node_id": node_id,
        "reference_name": args.reference_name,
        "candidate_name": args.candidate_name,
        "seeds": [{"node": k, "seed": v} for k, v in seeds],
        "kind": args.kind,
        "min_psnr_required_db": args.min_psnr,
        "max_temporal_delta": args.max_temporal_delta,
        "temporal_within_gate": temporal_ok,
        "ffmpeg": ffmpeg if args.kind == "video" else None,
        "pixel_tolerance": args.pixel_tolerance,
        "summary": {k: ("inf" if isinstance(v, float) and math.isinf(v) else v) for k, v in summary.items()},
        "per_image": [{k: ("inf" if isinstance(v, float) and math.isinf(v) else v) for k, v in m.items()} for m in per_image],
        "claim": (("frame parity within the PSNR gate" + (" and the temporal-delta gate" if temporal_ok is not None else "")
                   + " on this workflow and seed; not a statement about other prompts or seeds") if args.kind == "video" and passed
                  else "pixel parity within the PSNR gate on this workflow and seed; not a statement about other prompts or seeds" if passed
                  else "rejected: candidate output diverges from the reference beyond the gate"),
    }
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workflow", type=Path, required=True, help="ComfyUI API-format workflow JSON with a fixed seed")
    parser.add_argument("--reference-name", required=True, help="reference weights filename as listed by the loader node")
    parser.add_argument("--candidate-name", required=True, help="candidate quantized GGUF filename as listed by the loader node")
    parser.add_argument("--node-id", help="model loader node ID if the workflow has several")
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--kind", choices=("image", "video"), default="image",
                        help="video: decode MP4/WebM/GIF outputs to frames with ffmpeg and add a temporal-consistency delta")
    parser.add_argument("--ffmpeg", help="ffmpeg executable for --kind video (default: PATH)")
    parser.add_argument("--max-temporal-delta", type=float,
                        help="video gate: mean |reference motion - candidate motion| per frame step, 0..255 scale")
    parser.add_argument("--min-psnr", type=float, default=30.0, help="gate on the worst image or frame, dB")
    parser.add_argument("--pixel-tolerance", type=int, default=8, help="per-pixel max channel difference counted as 'over'")
    parser.add_argument("--timeout", type=float, default=1800.0, help="seconds per generation")
    parser.add_argument("--receipt", type=Path, help="write the JSON receipt here as well")
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
