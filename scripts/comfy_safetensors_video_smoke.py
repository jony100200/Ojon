#!/usr/bin/env python3
"""Run a local ComfyUI video workflow using a packed MiniMax-H3 safetensors model.

Requires a running loopback ComfyUI instance, an API-format video workflow with
a UNETLoader, and all companion encoder/VAE weights already installed.
Never downloads weights or claims W4A4 numerics / source-model quality parity.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
import uuid

MAX_HEADER = 32 * 1024 * 1024
MAX_JSON = 8 * 1024 * 1024
MAX_VIDEO = 512 * 1024 * 1024
VIDEO_OUTPUT_NODES = {"SaveVideo", "VHS_VideoCombine"}


def fail(text):
    raise ValueError(text)


def loopback(raw):
    url = urlsplit(raw)
    if (url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or not url.port or url.username or url.password
            or url.path not in {"", "/"} or url.query or url.fragment):
        fail("ComfyUI must be http://127.0.0.1:<port> on the local machine")
    return raw.rstrip("/")


def request_json(server, endpoint, data=None):
    payload = None if data is None else json.dumps(data).encode("utf-8")
    req = Request(server + endpoint, data=payload, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=30) as response:
        body = response.read(MAX_JSON + 1)
    if len(body) > MAX_JSON:
        fail("ComfyUI response exceeds 8 MiB")
    return json.loads(body)


def inspect_checkpoint(path):
    path = path.expanduser().resolve(strict=True)
    if not path.is_file() or path.suffix.lower() != ".safetensors":
        fail("checkpoint must be an existing single .safetensors file")
    size = path.stat().st_size
    if size < 16:
        fail("checkpoint too small")
    with path.open("rb") as handle:
        header_len = int.from_bytes(handle.read(8), "little")
        if not 0 < header_len <= min(MAX_HEADER, size - 8):
            fail("invalid safetensors header size")
        header = json.loads(handle.read(header_len))
    if not isinstance(header, dict):
        fail("safetensors header is not an object")
    keys = set(header) - {"__metadata__"}
    if not keys:
        fail("checkpoint contains no tensors")
    quant_marker_count = sum(key.endswith(".comfy_quant") for key in keys)
    return {
        "path": str(path), "bytes": size, "tensor_entries": len(keys),
        "comfy_quant_markers": quant_marker_count,
        "format_inspection": "COMFY_QUANT_METADATA_PRESENT" if quant_marker_count else "UNIDENTIFIED_SAFETENSORS",
    }


def prepare_workflow(workflow_file, model_name, node_id):
    workflow = json.loads(workflow_file.read_text(encoding="utf-8"))
    if not isinstance(workflow, dict) or not workflow:
        fail("workflow must be ComfyUI API-format JSON keyed by node ID")
    loaders = [key for key, node in workflow.items()
               if isinstance(node, dict) and node.get("class_type") == "UNETLoader"]
    if node_id is None and len(loaders) != 1:
        fail("workflow needs exactly one UNETLoader or explicit --node-id")
    node_id = node_id or loaders[0]
    if node_id not in loaders:
        fail("--node-id must identify a UNETLoader node")
    if not any(isinstance(node, dict) and node.get("class_type") in VIDEO_OUTPUT_NODES
               for node in workflow.values()):
        fail("workflow has no SaveVideo or VHS_VideoCombine output node")
    if not model_name.endswith(".safetensors") or "/" in model_name or "\\" in model_name:
        fail("model name must be a .safetensors filename from the ComfyUI loader list")
    workflow = copy.deepcopy(workflow)
    workflow[node_id].setdefault("inputs", {})["unet_name"] = model_name
    return workflow, node_id


def check_loader(server, name):
    meta = request_json(server, "/object_info/UNETLoader")
    try:
        entries = meta["UNETLoader"]["input"]["required"]["unet_name"][0]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("ComfyUI UNETLoader unavailable; update ComfyUI") from exc
    if name not in entries:
        fail("ComfyUI does not list this checkpoint. Put it in models/diffusion_models and refresh/restart ComfyUI")


def video_prefix(data):
    return (data[4:8] == b"ftyp" or data.startswith(b"\x1a\x45\xdf\xa3")
            or (data.startswith(b"RIFF") and data[8:12] == b"AVI "))


def media_items(record):
    if record.get("status", {}).get("status_str") == "error":
        fail("ComfyUI history reports an execution error")
    if record.get("status", {}).get("status_str") not in {"success", None}:
        fail("ComfyUI execution did not complete successfully")
    entries = []
    for node in record.get("outputs", {}).values():
        if not isinstance(node, dict):
            continue
        for category in ("videos", "gifs"):
            for item in node.get(category, []):
                if isinstance(item, dict) and item.get("filename"):
                    entries.append(item)
    if not entries:
        fail("ComfyUI completed but did not produce a saved video")
    return entries


def fetch_video(server, item, output_path):
    query = urlencode({
        "filename": item["filename"], "subfolder": item.get("subfolder", ""),
        "type": item.get("type", "output"),
    })
    temporary = output_path.with_name(output_path.name + ".partial")
    if output_path.exists() or temporary.exists():
        fail("refusing to overwrite existing video or partial download")
    try:
        with urlopen(server + "/view?" + query, timeout=90) as response:
            with temporary.open("xb") as handle:
                first = response.read(32)
                if not video_prefix(first):
                    fail("ComfyUI output is not an MP4, WebM, MKV or AVI video")
                handle.write(first)
                total = len(first)
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    total += len(block)
                    if total > MAX_VIDEO:
                        fail("video exceeds 512 MiB verification limit")
                    handle.write(block)
        if total < 1024:
            fail("ComfyUI video output is suspiciously small")
        temporary.rename(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return total


def probe_video(video_path, ffprobe):
    executable = ffprobe or shutil.which("ffprobe")
    if not executable:
        return {"verified": False, "reason": "ffprobe unavailable; video header only"}
    run = subprocess.run([
        str(executable), "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,nb_read_frames",
        "-of", "json", str(video_path),
    ], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120, check=False)
    if run.returncode:
        fail("ffprobe could not decode generated video: " + run.stderr[:500])
    data = json.loads(run.stdout)
    streams = data.get("streams", [])
    if not streams:
        fail("video has no decodable video stream")
    frames = streams[0].get("nb_read_frames", "N/A")
    if frames == "N/A" or not str(frames).isdigit() or int(frames) < 2:
        fail("video does not contain at least two decoded frames")
    return {"verified": True, "decoded_frames": int(frames),
            "width": streams[0].get("width"), "height": streams[0].get("height"),
            "codec": streams[0].get("codec_name")}


def run(args):
    server = loopback(args.server)
    model = inspect_checkpoint(args.model)
    model_name = args.model_name or Path(model["path"]).name
    workflow, loader_id = prepare_workflow(args.workflow, model_name, args.node_id)
    receipt = args.receipt.expanduser().resolve()
    output = args.video_out.expanduser().resolve()
    if not receipt.parent.is_dir() or not output.parent.is_dir():
        fail("receipt and video output parents must exist")
    if receipt.exists() or output.exists():
        fail("receipt or video output already exists")
    plan = {
        "schema": "ojon.comfy_safetensors_video.v1",
        "status": "PLAN_ONLY", "checkpoint": model, "workflow": str(args.workflow),
        "loader_id": loader_id, "model_name": model_name, "server": server,
        "video_out": str(output), "receipt": str(receipt),
        "generation_verified": False, "visual_quality_verified": False,
        "weight_quantization": "UNKNOWN_UNLESS_VALIDATED_BY_RUNTIME",
    }
    if not args.confirm:
        return plan
    check_loader(server, model_name)
    queued = request_json(server, "/prompt", {"prompt": workflow, "client_id": str(uuid.uuid4())})
    prompt_id = queued.get("prompt_id") if isinstance(queued, dict) else None
    if not prompt_id or queued.get("node_errors"):
        fail("ComfyUI rejected workflow; verify model dependencies and node settings")
    started = time.monotonic()
    while time.monotonic() - started < args.timeout:
        history = request_json(server, "/history/" + prompt_id)
        if prompt_id in history:
            items = media_items(history[prompt_id])
            count = fetch_video(server, items[0], output)
            check = probe_video(output, args.ffprobe)
            result = dict(plan)
            result.update({
                "status": "VIDEO_DECODED_NOT_QUALITY_VERIFIED" if check["verified"]
                          else "VIDEO_SAVED_DECODE_UNVERIFIED",
                "generation_verified": check["verified"], "video_bytes": count,
                "video_probe": check, "prompt_id": prompt_id,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            })
            with receipt.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
                handle.write("\n")
            return result
        time.sleep(2)
    fail("video generation timed out; no success receipt was written")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--workflow", type=Path, required=True)
    parser.add_argument("--video-out", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--node-id", help="UNETLoader ID if more than one")
    parser.add_argument("--model-name", help="filename shown in ComfyUI UNETLoader")
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--ffprobe", type=Path)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not 30 <= args.timeout <= 14400:
        parser.error("timeout must be between 30 and 14400 seconds")
    try:
        print(json.dumps(run(args), indent=2))
    except (OSError, ValueError, KeyError, json.JSONDecodeError, subprocess.TimeoutExpired) as error:
        print(json.dumps({"status": "FAILED", "error": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
