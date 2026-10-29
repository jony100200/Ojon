"""Offline protocol tests only; no claim of video generation or W4A4 parity."""
from http.server import BaseHTTPRequestHandler, HTTPServer
import argparse
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import comfy_safetensors_video_smoke as smoke


def checkpoint(path):
    payload = {"blocks.0.mlp.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
               "blocks.0.mlp.comfy_quant": {"dtype": "U8", "shape": [1], "data_offsets": [2, 3]}}
    header = json.dumps(payload).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0\0\0")


VIDEO = b"\0\0\0\x18ftyp" + b"\0" * 2048


class FakeComfy(BaseHTTPRequestHandler):
    scenario = "good"
    observed = None

    def log_message(self, *_):
        pass

    def respond(self, data):
        raw = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/object_info/UNETLoader":
            self.respond({"UNETLoader": {"input": {"required": {"unet_name": [["packed.safetensors"]]}}}})
        elif self.path == "/object_info/OrbitQuantReleaseLoader":
            self.respond({"OrbitQuantReleaseLoader": {"input": {"required": {"model_path": ["STRING"]}}}})
        elif self.path.startswith("/history/"):
            outputs = {"7": {"videos": [{"filename": "sample.mp4", "type": "output"}]}}
            if type(self).scenario == "missing":
                outputs = {"7": {"images": [{"filename": "still.png"}]}}
            elif type(self).scenario == "orbitquant":
                outputs = {"7": {"images": [{"filename": "orbitquant.mp4", "type": "output"}]}}
            self.respond({"smoke-test": {"status": {"status_str": "success"}, "outputs": outputs}})
        elif self.path.startswith("/view?"):
            self.send_response(200)
            self.end_headers()
            data = b"not-video" if type(self).scenario == "corrupt" else VIDEO
            self.wfile.write(data)
        else:
            self.send_error(404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).observed = body["prompt"]
        self.respond({"prompt_id": "smoke-test", "node_errors": {}})


class Tests(unittest.TestCase):
    def make(self, root, server):
        model = root / "packed.safetensors"
        checkpoint(model)
        workflow = root / "api-workflow.json"
        workflow.write_text(json.dumps({
            "4": {"class_type": "UNETLoader", "inputs": {"unet_name": "base.safetensors"}},
            "7": {"class_type": "SaveVideo", "inputs": {"video": ["5", 0]}}
        }), encoding="utf-8")
        return argparse.Namespace(
            model=model, workflow=workflow, model_name=None, node_id=None,
            server=server, receipt=root / "receipt.json", video_out=root / "result.mp4",
            ffprobe=None, timeout=30, confirm=False)

    def test_refuses_remote_and_nonvideo_workflow(self):
        for address in ["https://127.0.0.1:8188", "http://example.com:8188",
                        "http://127.0.0.1:8188/path", "http://127.0.0.1"]:
            with self.assertRaises(ValueError):
                smoke.loopback(address)
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "f.json"
            p.write_text(json.dumps({"4": {"class_type": "UNETLoader", "inputs": {}}}))
            with self.assertRaises(ValueError):
                smoke.prepare_workflow(p, "packed.safetensors", None)

    def test_no_receipt_for_failures(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            http = HTTPServer(("127.0.0.1", 0), FakeComfy)
            thread = threading.Thread(target=http.serve_forever, daemon=True)
            thread.start()
            try:
                args = self.make(root, f"http://127.0.0.1:{http.server_port}")
                self.assertEqual(smoke.run(args)["status"], "PLAN_ONLY")
                self.assertFalse(args.receipt.exists())
                args.confirm = True
                FakeComfy.scenario = "good"
                with patch.object(smoke, "probe_video", return_value={
                        "verified": True, "decoded_frames": 23, "width": 864, "height": 480}):
                    result = smoke.run(args)
                self.assertEqual(result["status"], "VIDEO_DECODED_NOT_QUALITY_VERIFIED")
                self.assertEqual(FakeComfy.observed["4"]["inputs"]["unet_name"], "packed.safetensors")
                self.assertTrue(args.receipt.is_file())
                self.assertTrue(args.video_out.is_file())

                args.receipt.unlink()
                args.video_out.unlink()
                FakeComfy.scenario = "missing"
                with self.assertRaisesRegex(ValueError, "did not produce a saved video"):
                    smoke.run(args)
                self.assertFalse(args.receipt.exists())

                FakeComfy.scenario = "corrupt"
                with self.assertRaisesRegex(ValueError, "not an MP4"):
                    smoke.run(args)
                self.assertFalse(args.receipt.exists())
                self.assertFalse(args.video_out.exists())
            finally:
                http.shutdown()
                http.server_close()
                thread.join(timeout=3)

    def test_w4a4_orbitquant_release_requires_real_video_and_correct_loader(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            release = root / "release"
            release.mkdir()
            (release / "comfyui_orbitquant.json").write_text(json.dumps({
                "schema_version": 1, "adapter": "minimax_h3", "media_type": "video"
            }), encoding="utf-8")
            flow = root / "w4a4-workflow.json"
            flow.write_text(json.dumps({
                "1": {"class_type": "OrbitQuantReleaseLoader",
                      "inputs": {"model_path": "/not/the/actual/model"}},
                "7": {"class_type": "OrbitQuantGenerateVideo",
                      "inputs": {"release": ["1", 0], "prompt": "A moving scene"}}
            }), encoding="utf-8")
            server = HTTPServer(("127.0.0.1", 0), FakeComfy)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                args = argparse.Namespace(
                    model=release, workflow=flow, model_name=None, node_id=None,
                    runtime="orbitquant", server=f"http://127.0.0.1:{server.server_port}",
                    receipt=root / "w4a4-receipt.json", video_out=root / "clip.mp4",
                    ffprobe=None, timeout=30, confirm=False)
                self.assertEqual(smoke.run(args)["status"], "PLAN_ONLY")
                self.assertFalse(args.receipt.exists())
                args.confirm = True
                FakeComfy.scenario = "orbitquant"
                with patch.object(smoke, "probe_video", return_value={
                        "verified": True, "decoded_frames": 24, "width": 608, "height": 480}):
                    result = smoke.run(args)
                self.assertEqual(result["runtime"], "orbitquant")
                self.assertEqual(FakeComfy.observed["1"]["inputs"]["model_path"], str(release.resolve()))
                self.assertTrue(args.video_out.is_file())
                self.assertTrue(args.receipt.is_file())
                args.receipt.unlink()
                args.video_out.unlink()
                FakeComfy.scenario = "missing"
                with self.assertRaisesRegex(ValueError, "did not produce a saved video"):
                    smoke.run(args)
                self.assertFalse(args.receipt.exists())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)

    def test_orbitquant_rejects_single_component_and_wrong_adapter(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            with self.assertRaisesRegex(ValueError, "comfyui_orbitquant.json"):
                smoke.inspect_orbitquant_release(root)
            (root / "comfyui_orbitquant.json").write_text(json.dumps({
                "adapter": "other", "media_type": "video"
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not a MiniMax"):
                smoke.inspect_orbitquant_release(root)

    def test_invalid_checkpoint_rejected_before_submission(self):
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder) / "not-model.safetensors"
            p.write_bytes(b"wrong")
            with self.assertRaises(ValueError):
                smoke.inspect_checkpoint(p)


if __name__ == "__main__":
    unittest.main()
