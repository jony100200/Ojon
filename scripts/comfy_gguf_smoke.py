#!/usr/bin/env python3
"""Local ComfyUI-GGUF generation smoke gate for Ojon.

Requires an already running LOCAL ComfyUI with a GGUF loader and an API-format
workflow containing UnetLoaderGGUF (or UnetLoaderGGUFAdvanced). This does not
quantize or download weights. A successful run proves actual workflow execution
and a retrievable media artifact, NOT visual parity or W4A4 arithmetic.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time
from urllib.parse import urlencode, urlsplit
from urllib.request import Request, urlopen
import uuid

LOADERS = {"UnetLoaderGGUF", "UnetLoaderGGUFAdvanced"}
OUTPUTS = {"image": ("images",), "video": ("gifs", "videos", "images")}
MAX_JSON = 8 * 1024 * 1024


def fail(message):
    raise ValueError(message)


def local_url(raw):
    url = urlsplit(raw)
    if (url.scheme != "http" or url.hostname not in {"localhost", "127.0.0.1", "::1"}
            or url.username or url.password or url.path not in {"", "/"}
            or url.query or url.fragment or url.port is None):
        fail("ComfyUI server must be a loopback HTTP URL with an explicit port")
    return raw.rstrip("/")


def request_json(server, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = Request(server + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=20) as response:
        body = response.read(MAX_JSON + 1)
    if len(body) > MAX_JSON:
        fail("ComfyUI API response exceeds bounded JSON limit")
    return json.loads(body)


def load_workflow(path, model_name, node_id=None):
    workflow = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(workflow, dict) or not workflow:
        fail("workflow must be an API-format JSON object keyed by node ID")
    loader_ids = [
        key for key, value in workflow.items()
        if isinstance(value, dict) and value.get("class_type") in LOADERS
    ]
    if node_id is None:
        if len(loader_ids) != 1:
            fail("expected exactly one GGUF diffusion loader; specify --node-id")
        node_id = loader_ids[0]
    if node_id not in loader_ids:
        fail("--node-id must identify an existing UnetLoaderGGUF node")
    patched = copy.deepcopy(workflow)
    if not isinstance(patched[node_id].get("inputs"), dict):
        fail("GGUF loader lacks inputs")
    patched[node_id]["inputs"]["unet_name"] = model_name
    return patched, node_id, patched[node_id]["class_type"]


def check_server_model(server, cls, model_name):
    info = request_json(server, "/object_info/" + cls)
    try:
        allowed = info[cls]["input"]["required"]["unet_name"][0]
    except (KeyError, TypeError, IndexError) as exc:
        raise ValueError("GGUF loader node is missing from this ComfyUI instance") from exc
    if model_name not in allowed:
        fail("ComfyUI GGUF loader does not list this model; install it in "
             "ComfyUI/models/diffusion_models (or models/unet) and restart/refresh")


def valid_media(prefix, kind):
    if kind == "image":
        return (prefix.startswith(bytes((137,80,78,71,13,10,26,10)))
                or prefix.startswith(bytes((255,216,255)))
                or prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP")
    return (prefix.startswith(bytes((26,69,223,163)))
            or prefix.startswith(b"RIFF") and prefix[8:12] == b"AVI "
            or prefix[4:8] == b"ftyp"
            or prefix.startswith(b"GIF8")
            or prefix.startswith(b"RIFF") and prefix[8:12] == b"WEBP")


def check_generated_outputs(server, record, kind):
    if record.get("status", {}).get("status_str") == "error":
        fail("ComfyUI history reports execution error")
    outputs = record.get("outputs", {})
    if not isinstance(outputs, dict):
        fail("ComfyUI history lacks outputs")
    media = []
    for node_id, result in outputs.items():
        if not isinstance(result, dict):
            continue
        for category in OUTPUTS[kind]:
            for item in result.get(category, []):
                if not isinstance(item, dict) or not item.get("filename"):
                    continue
                parameters = urlencode({
                    "filename": item["filename"],
                    "subfolder": item.get("subfolder", ""),
                    "type": item.get("type", "output"),
                })
                with urlopen(server + "/view?" + parameters, timeout=30) as response:
                    prefix = response.read(32)
                if valid_media(prefix, kind):
                    media.append({"node_id": node_id, "category": category,
                                  "filename": item["filename"], "type": item.get("type", "output")})
    if not media:
        fail("ComfyUI completed but no retrievable, recognized output media was produced")
    return media


def run(args):
    server = local_url(args.server)
    model = args.model.expanduser().resolve(strict=True)
    if model.suffix.lower() != ".gguf" or not model.is_file() or model.stat().st_size < 32:
        fail("model must be a nonempty local GGUF file")
    with model.open("rb") as handle:
        if handle.read(4) != b"GGUF":
            fail("input is not a GGUF container")
    receipt = args.receipt.expanduser().resolve()
    if receipt.exists():
        fail("refusing to replace existing receipt")
    if not receipt.parent.is_dir():
        fail("receipt parent folder does not exist")
    name = args.model_name or model.name
    if "/" in name or "\\" in name or not name.endswith(".gguf"):
        fail("--model-name must be a ComfyUI-listed GGUF filename (not a filesystem path)")
    patched, node_id, cls = load_workflow(args.workflow, name, args.node_id)
    result = {
        "schema": "ojon.comfy_gguf_smoke.v1",
        "status": "PLAN_ONLY", "model": str(model), "comfyui_model_name": name,
        "workflow": str(args.workflow), "loader_node_id": node_id,
        "loader_class": cls, "kind": args.kind, "server": server,
        "weight_quantization": "GGUF; W4A4 NOT VERIFIED",
        "visual_quality_verified": False, "sd_cpp_runtime_verified": False,
    }
    if not args.confirm:
        return result
    check_server_model(server, cls, name)
    queued = request_json(server, "/prompt", {
        "prompt": patched, "client_id": str(uuid.uuid4())
    })
    identifier = queued.get("prompt_id") if isinstance(queued, dict) else None
    if not identifier or queued.get("node_errors"):
        fail("ComfyUI rejected the workflow (missing components or invalid inputs)")
    started = time.monotonic()
    while time.monotonic() - started < args.timeout:
        history = request_json(server, "/history/" + identifier)
        if isinstance(history, dict) and identifier in history:
            record = history[identifier]
            if not isinstance(record, dict):
                fail("malformed ComfyUI history record")
            result["outputs"] = check_generated_outputs(server, record, args.kind)
            result["status"] = "RUNTIME_GENERATION_COMPLETED_NOT_QUALITY_VERIFIED"
            result["prompt_id"] = identifier
            result["elapsed_seconds"] = round(time.monotonic() - started, 3)
            result["comfyui_gguf_runtime_verified"] = True
            result["warning"] = "Only this workflow's model load and generated output passed; no visual parity or SD.cpp compatibility was demonstrated."
            # Exclusive create means a concurrent caller cannot destroy evidence.
            with receipt.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2)
                handle.write("\n")
            return result
        time.sleep(1)
    fail("ComfyUI generation timed out; no success receipt was created")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", type=Path, required=True, help="ComfyUI API-format workflow JSON")
    parser.add_argument("--model", type=Path, required=True, help="local quantized GGUF file")
    parser.add_argument("--model-name", help="filename as listed by ComfyUI loader")
    parser.add_argument("--node-id", help="GGUF loader node ID if the workflow has multiple")
    parser.add_argument("--kind", choices=("image", "video"), default="image")
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--confirm", action="store_true", help="submit a real generation job")
    args = parser.parse_args()
    if not 10 <= args.timeout <= 7200:
        parser.error("--timeout must be between 10 and 7200 seconds")
    try:
        print(json.dumps(run(args), indent=2))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "FAILED", "error": str(exc)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
