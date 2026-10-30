"""Render known text onto plain page images for the OCR parity gate.

Usage: python make_ocr_pages.py <out_dir> [--count N] [--seed S] [--font PATH] [--size PX]
Writes <out_dir>/page-NNNN.png and <out_dir>/pages.jsonl ({"image": "page-NNNN.png", "text": "..."}).
Needs Pillow. The texts are fixed sentences mixed deterministically from the seed so the
same seed gives the same pages; they exercise digits, punctuation and mixed case.
"""
import argparse, json, random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SENTENCES = [
    "Invoice 2026-10-30: total due 1,284.50 CAD by Friday.",
    "The quick brown fox jumps over the lazy dog.",
    "Meeting room B7, 14:30. Bring the Q3 report and the signed NDA.",
    "Serial number KS-4060-TI-16GB; firmware 2.9.1; status OK.",
    "Error 0xC0000139: entry point not found in mtmd.dll.",
    "Temperature 21.5 C, humidity 48%, wind 12 km/h from the north-west.",
    "Pay to the order of Kalponic Studio: seven hundred and twelve dollars.",
    "Chapter 4. Mixtures of experts route each token to a few experts.",
    "Order #A1B2C3 shipped: 2 x RTL8125 2.5GbE cards, 1 x 10 m Cat6 cable.",
    "Remember what matters. Guide what comes next. Keep improving.",
    "Mean KLD 0.024363 +/- 0.000673; p99 0.138750; within budget.",
    "Fibo 2.0 Dev uses 50 steps and guidance 5.0; Edit Turbo uses 4 steps.",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--count", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261030)
    ap.add_argument("--font", default=None, help="TTF path; default tries Arial/DejaVuSans, else Pillow's bitmap font")
    ap.add_argument("--size", type=int, default=28)
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(a.seed)
    font = None
    for cand in ([a.font] if a.font else []) + ["arial.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        try:
            font = ImageFont.truetype(cand, a.size); break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    rows = []
    for i in range(a.count):
        lines = rng.sample(SENTENCES, k=3)
        text = "\n".join(lines)
        img = Image.new("RGB", (1024, 320), "white")
        d = ImageDraw.Draw(img)
        y = 40
        for line in lines:
            d.text((40, y), line, fill="black", font=font)
            y += a.size + 22
        name = f"page-{i:04d}.png"
        img.save(out / name)
        rows.append({"image": name, "text": text})
    with open(out / "pages.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} pages -> {out / 'pages.jsonl'}")

if __name__ == "__main__":
    main()
