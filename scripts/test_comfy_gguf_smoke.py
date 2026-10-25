"""Offline tests for Ojon's ComfyUI HTTP integration. No model is loaded."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

import comfy_gguf_smoke as smoke

PNG = bytes((137, 80, 78, 71, 13, 10, 26, 10)) + b"\0" * 64


class FakeComfy(BaseHTTPRequestHandler):
    mode = "good"
    last_workflow = None

    def log_message(self, *_):
        pass

    def reply(self, body):
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.startswith("/object_info/UnetLoaderGGUF"):
            self.reply({"UnetLoaderGGUF": {"input": {
                "required": {"unet_name": [["quantized.gguf"]]}}}})
        elif self.path.startswith("/history/"):
            outputs = {"8": {"images": [{"filename": "proof.png", "type": "output"}]}}
            if type(self).mode == "empty":
                outputs = {}
            self.reply({"test-id": {"status": {"status_str": "success"}, "outputs": outputs}})
        elif self.path.startswith("/view?"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(PNG)
        else:
            self.send_error(404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        message = json.loads(self.rfile.read(length))
        type(self).last_workflow = message["prompt"]
        self.reply({"prompt_id": "test-id", "node_errors": {}})


class ComfyTests(unittest.TestCase):
    def test_reject_nonlocal_server(self):
        for address in ("https://127.0.0.1:8188", "http://example.com:8188",
                        "http://127.0.0.1:8188/x", "http://127.0.0.1",
                        "http://user:pass@localhost:8188"):
            with self.assertRaises(ValueError):
                smoke.local_url(address)
        self.assertEqual(smoke.local_url("http://localhost:8188"),
                         "http://localhost:8188")

    def test_workflow_requires_gguf_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "flow.json"
            path.write_text(json.dumps({"4": {"class_type": "CheckpointLoaderSimple",
                                              "inputs": {"ckpt_name": "base.safetensors"}}}))
            with self.assertRaises(ValueError):
                smoke.load_workflow(path, "quantized.gguf")

    def test_generation_receipt_only_after_real_api_output(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "quantized.gguf"
            model.write_bytes(b"GGUF" + bytes(64))
            flow = root / "flow.json"
            flow.write_text(json.dumps({
                "4": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "old.gguf"}},
                "8": {"class_type": "SaveImage", "inputs": {"images": ["4", 0]}}
            }))
            receipt = root / "receipt.json"
            server = HTTPServer(("127.0.0.1", 0), FakeComfy)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                args = argparse.Namespace(
                    model=model, model_name=None, receipt=receipt,
                    workflow=flow, node_id=None, kind="image", timeout=10,
                    server=f"http://127.0.0.1:{server.server_port}", confirm=False
                )
                self.assertEqual(smoke.run(args)["status"], "PLAN_ONLY")
                self.assertFalse(receipt.exists())
                args.confirm = True
                FakeComfy.mode = "good"
                result = smoke.run(args)
                self.assertTrue(receipt.is_file())
                self.assertTrue(result["comfyui_gguf_runtime_verified"])
                self.assertEqual(FakeComfy.last_workflow["4"]["inputs"]["unet_name"],
                                 "quantized.gguf")
                self.assertFalse(result["sd_cpp_runtime_verified"])
                receipt.unlink()
                FakeComfy.mode = "empty"
                with self.assertRaises(ValueError):
                    smoke.run(args)
                self.assertFalse(receipt.exists())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
