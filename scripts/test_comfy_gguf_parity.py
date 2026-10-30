import math
import struct
import unittest
import zlib

import comfy_gguf_parity as parity


def encode_png(width, height, rgb, filter_type=0, channels=3):
    """Minimal PNG writer used only to feed the decoder under test."""
    color = {1: 0, 3: 2, 4: 6, 2: 4}[channels]
    stride = width * channels
    rows = bytearray()
    prev = bytes(stride)
    for y in range(height):
        line = rgb[y * stride:(y + 1) * stride]
        if filter_type == 0:
            enc = bytes(line)
        elif filter_type == 1:
            enc = bytes((line[i] - (line[i - channels] if i >= channels else 0)) & 255 for i in range(stride))
        elif filter_type == 2:
            enc = bytes((line[i] - prev[i]) & 255 for i in range(stride))
        elif filter_type == 3:
            enc = bytes((line[i] - (((line[i - channels] if i >= channels else 0) + prev[i]) >> 1)) & 255 for i in range(stride))
        else:
            enc = bytes((line[i] - parity._paeth(line[i - channels] if i >= channels else 0, prev[i],
                                                 prev[i - channels] if i >= channels else 0)) & 255 for i in range(stride))
        rows.append(filter_type)
        rows += enc
        prev = line

    def chunk(kind, body):
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, color, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(bytes(rows))) + chunk(b"IEND", b"")


def gradient(width, height):
    return bytes(((x * 7 + y * 3 + c * 11) & 255) for y in range(height) for x in range(width) for c in range(3))


class PngDecode(unittest.TestCase):
    def test_all_filters_roundtrip(self):
        w, h = 9, 5
        px = gradient(w, h)
        for f in range(5):
            dw, dh, ch, out = parity.decode_png(encode_png(w, h, px, filter_type=f))
            self.assertEqual((dw, dh, ch), (w, h, 3), f"filter {f}")
            self.assertEqual(out, px, f"filter {f}")

    def test_grey_and_rgba_expand_to_rgb(self):
        grey = bytes(range(0, 16))
        w, h, ch, out = parity.decode_png(encode_png(4, 4, grey, channels=1))
        rgb = parity.to_rgb(w, h, ch, out)
        self.assertEqual(rgb[:6], bytes((0, 0, 0, 1, 1, 1)))
        rgba = bytes(v for i in range(16) for v in (i, i + 1, i + 2, 255))
        w, h, ch, out = parity.decode_png(encode_png(4, 4, rgba, channels=4))
        self.assertEqual(parity.to_rgb(w, h, ch, out)[:3], bytes((0, 1, 2)))

    def test_rejects_non_png(self):
        with self.assertRaises(ValueError):
            parity.decode_png(b"GIF89a....")


class Metrics(unittest.TestCase):
    def test_identical_is_infinite_psnr(self):
        px = gradient(8, 8)
        m = parity.compare_rgb(px, px, 8, 8, 8)
        self.assertTrue(math.isinf(m["psnr_db"]))
        self.assertEqual(m["mean_abs_diff"], 0)
        self.assertEqual(m["pixels_over_tolerance_frac"], 0)

    def test_known_difference(self):
        a = bytes([100] * (4 * 4 * 3))
        b = bytes([110] * (4 * 4 * 3))  # every channel off by 10 -> MSE 100 -> PSNR 28.13 dB
        m = parity.compare_rgb(a, b, 4, 4, 8)
        self.assertAlmostEqual(m["psnr_db"], 10 * math.log10(255 * 255 / 100), places=6)
        self.assertEqual(m["mean_abs_diff"], 10)
        self.assertEqual(m["pixels_over_tolerance_frac"], 1.0)

    def test_parity_summary_and_mismatch(self):
        px = gradient(6, 6)
        ref = [("a.png", encode_png(6, 6, px))]
        same = [("a.png", encode_png(6, 6, px, filter_type=2))]
        summary, per = parity.parity(ref, same, 8)
        self.assertEqual(summary["images"], 1)
        self.assertTrue(math.isinf(summary["min_psnr_db"]))
        with self.assertRaises(ValueError):
            parity.parity(ref, ref + ref, 8)
        with self.assertRaises(ValueError):
            parity.parity(ref, [("b.png", encode_png(5, 6, gradient(5, 6)))], 8)


class Temporal(unittest.TestCase):
    def test_profile_and_delta_detect_flicker_and_blur(self):
        w, h = 4, 4
        still = bytes([100] * (w * h * 3))
        moved = bytes([110] * (w * h * 3))
        ref = [still, moved, still]           # reference moves then returns
        cand_same = [still, moved, still]
        cand_flicker = [still, still, moved]  # same frames, different timing
        rp = parity.temporal_profile(ref, w, h)
        self.assertEqual(rp, [10.0, 10.0])
        self.assertEqual(parity.temporal_delta(rp, parity.temporal_profile(cand_same, w, h)), 0.0)
        self.assertGreater(parity.temporal_delta(rp, parity.temporal_profile(cand_flicker, w, h)), 0.0)
        self.assertIsNone(parity.temporal_delta(rp, [1.0]))

    def test_parity_video_mode_adds_temporal_summary(self):
        px_a, px_b = gradient(6, 6), bytes((v + 3) & 255 for v in gradient(6, 6))
        ref = [("f1.png", encode_png(6, 6, px_a)), ("f2.png", encode_png(6, 6, px_b))]
        summary, per = parity.parity(ref, ref, 8, temporal=True)
        self.assertEqual(summary["frames"], 2)
        self.assertEqual(summary["temporal_delta"], 0.0)
        self.assertGreater(summary["reference_mean_motion"], 0.0)
        summary_img, _ = parity.parity(ref, ref, 8)
        self.assertNotIn("temporal_delta", summary_img)

    def test_extract_frames_requires_ffmpeg_and_rejects_garbage(self):
        with self.assertRaises(ValueError):
            parity.extract_frames(b"not a video", ".mp4", None)
        import shutil as _sh
        ff = _sh.which("ffmpeg")
        if ff:
            with self.assertRaises(ValueError):
                parity.extract_frames(b"not a video", ".mp4", ff)


class WorkflowChecks(unittest.TestCase):
    def test_fixed_seed_required(self):
        wf = {"3": {"class_type": "KSampler", "inputs": {"seed": 42, "steps": 20}},
              "9": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "x.gguf"}}}
        self.assertEqual(parity.fixed_seed(wf), [("3", 42)])
        bad = {"3": {"class_type": "KSampler", "inputs": {"seed": "random"}}}
        with self.assertRaises(ValueError):
            parity.fixed_seed(bad)
        with self.assertRaises(ValueError):
            parity.fixed_seed({"9": {"class_type": "UnetLoaderGGUF", "inputs": {}}})

    def test_patch_model_and_loader_detection(self):
        wf = {"9": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "old.gguf"}},
              "3": {"class_type": "KSampler", "inputs": {"seed": 1}}}
        self.assertEqual(parity.loader_nodes(wf), ["9"])
        patched = parity.patch_model(wf, "9", "new.gguf")
        self.assertEqual(patched["9"]["inputs"]["unet_name"], "new.gguf")
        self.assertEqual(wf["9"]["inputs"]["unet_name"], "old.gguf")  # original untouched

    def test_server_must_be_local(self):
        self.assertEqual(parity.local_url("http://127.0.0.1:8188/"), "http://127.0.0.1:8188")
        with self.assertRaises(SystemExit):
            parity.local_url("http://example.com:8188")


if __name__ == "__main__":
    unittest.main()
