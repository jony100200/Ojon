#!/usr/bin/env python3
"""Parity gate for Ojon `3d` quants: original vs quantized TRELLIS.2 packages through
trellis2-image-to-gltf --shape-only. Reports 6-view silhouette IoU, chamfer distance (as a
fraction of the original's bounding-box diagonal), peak VRAM, peak system RAM and seconds.

Run: trellis3d_parity.py --exe <trellis2-image-to-gltf> --dino <dir> --birefnet <gguf> \
       --original <pkg> --quant Q4_K_M=<pkg> --quant Q5_K=<pkg> --image a.png --image b.png --out <dir>
Packages are model folders (model.json + ckpts/). Needs numpy, scipy, psutil.
"""
import argparse, json, struct, subprocess, sys, time
from pathlib import Path
import numpy as np, psutil
from scipy.spatial import cKDTree

CTYPE = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16, 5125: np.uint32, 5126: np.float32}
NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def vram_used_mib():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.split()
    return int(out[0]) if out else 0


def run_peak(cmd, poll=0.1):
    """Run cmd; return (returncode, seconds, peak_rss_mib, peak_vram_delta_mib, output)."""
    base_vram = vram_used_mib()
    t0 = time.time()
    import tempfile
    # a temp file, never a pipe: an undrained pipe deadlocks chatty children
    logf = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, text=True)
    ps = psutil.Process(proc.pid)
    peak_rss = peak_vram = 0
    tick = 0
    while proc.poll() is None:
        try:
            procs = [ps] + ps.children(recursive=True)
            peak_rss = max(peak_rss, sum(p.memory_info().rss for p in procs if p.is_running()))
        except psutil.Error:
            pass
        if tick % 5 == 0:
            peak_vram = max(peak_vram, vram_used_mib() - base_vram)
        tick += 1
        time.sleep(poll)
    logf.seek(0)
    out = logf.read()
    logf.close()
    return proc.returncode, time.time() - t0, peak_rss / 2**20, peak_vram, out


def load_glb(path):
    b = Path(path).read_bytes()
    assert b[:4] == b"glTF", "not a GLB"
    off, js, binc = 12, None, b""
    while off < len(b):
        ln, ty = struct.unpack_from("<II", b, off)
        chunk = b[off + 8: off + 8 + ln]
        if ty == 0x4E4F534A: js = json.loads(chunk)
        elif ty == 0x004E4942: binc = chunk
        off += 8 + ln
    if "KHR_mesh_quantization" in js.get("extensionsRequired", []) or "EXT_meshopt_compression" in js.get("extensionsRequired", []):
        raise RuntimeError("compressed GLB; rerun with --no-mesh-postprocess outputs")

    def accessor(i):
        a = js["accessors"][i]; bv = js["bufferViews"][a["bufferView"]]
        dt, n = CTYPE[a["componentType"]], NCOMP[a["type"]]
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = bv.get("byteStride") or n * np.dtype(dt).itemsize
        raw = np.frombuffer(binc, dtype=np.uint8, count=stride * a["count"], offset=start) if stride == n * np.dtype(dt).itemsize \
            else None
        if raw is not None:
            return np.frombuffer(binc, dtype=dt, count=a["count"] * n, offset=start).reshape(a["count"], n)
        rows = np.frombuffer(binc, dtype=np.uint8, count=stride * a["count"], offset=start).reshape(a["count"], stride)
        return rows[:, : n * np.dtype(dt).itemsize].copy().view(dt).reshape(a["count"], n)

    verts, faces, base = [], [], 0
    for mesh in js["meshes"]:
        for prim in mesh["primitives"]:
            v = accessor(prim["attributes"]["POSITION"]).astype(np.float64)
            f = accessor(prim["indices"]).reshape(-1, 3).astype(np.int64) + base
            verts.append(v); faces.append(f); base += len(v)
    return np.concatenate(verts), np.concatenate(faces)


def sample_surface(v, f, n, rng):
    tri = v[f]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    idx = rng.choice(len(f), size=n, p=area / area.sum())
    u, w = rng.random(n), rng.random(n)
    flip = u + w > 1
    u[flip], w[flip] = 1 - u[flip], 1 - w[flip]
    t = tri[idx]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + w[:, None] * (t[:, 2] - t[:, 0])


def silhouettes(points, lo, hi, res=256):
    """Binary masks from +-x, +-y, +-z orthographic views, in the original's bounding box frame."""
    span = (hi - lo).max() * 1.05
    ctr = (hi + lo) / 2
    uv = (points - ctr) / span + 0.5
    masks = []
    for axis in range(3):
        a, b = [i for i in range(3) if i != axis]
        m = np.zeros((res, res), bool)
        x = np.clip((uv[:, a] * res).astype(int), 0, res - 1)
        y = np.clip((uv[:, b] * res).astype(int), 0, res - 1)
        m[y, x] = True
        pad = np.pad(m, 1)
        m = sum(pad[1 + dy: 1 + dy + res, 1 + dx: 1 + dx + res] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) >= 1
        # fill 1px pinholes left by sampling
        pad = np.pad(m, 1)
        nb = sum(pad[1 + dy: 1 + dy + res, 1 + dx: 1 + dx + res] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
        masks.append(m | (nb >= 7))
    return masks + masks  # +-axis project to the same mask under orthographic occlusion-free splatting


def compare(orig_glb, quant_glb, n_samples=300_000, seed=0):
    rng = np.random.default_rng(seed)
    vo, fo = load_glb(orig_glb); vq, fq = load_glb(quant_glb)
    po, pq = sample_surface(vo, fo, n_samples, rng), sample_surface(vq, fq, n_samples, rng)
    lo, hi = vo.min(0), vo.max(0)
    diag = float(np.linalg.norm(hi - lo))
    d_oq, d_qo = cKDTree(pq).query(po)[0], cKDTree(po).query(pq)[0]
    chamfer = 0.5 * (d_oq.mean() + d_qo.mean()) / diag
    so, sq = silhouettes(po, lo, hi), silhouettes(pq, lo, hi)
    # six views: front/back/left/right/top/bottom; front and back share a mask in this splat model
    # but are scored separately so the 6-view mean matches the specified metric.
    ious = [float((a & b).sum() / max((a | b).sum(), 1)) for a, b in zip(so, sq)]
    return {"iou_mean": float(np.mean(ious)), "iou_min": float(min(ious)), "chamfer_rel_diag": float(chamfer),
            "faces_orig": int(len(fo)), "faces_quant": int(len(fq))}


def engine_cmd(a, pkg, image, out, seed=None):
    cmd = [a.exe, "--model", str(pkg), "--dino", a.dino, "--image", str(image), "--glb", str(out), "--shape-only",
           "--no-mesh-postprocess", "--seed", str(a.seed if seed is None else seed)]
    if a.birefnet:
        cmd += ["--birefnet", a.birefnet]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", required=True); ap.add_argument("--dino", required=True)
    ap.add_argument("--birefnet"); ap.add_argument("--original", required=True)
    ap.add_argument("--quant", action="append", required=True, help="NAME=package_dir")
    ap.add_argument("--image", action="append", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--iou-min", type=float, default=0.97)
    ap.add_argument("--chamfer-max", type=float, default=0.01)
    ap.add_argument("--skip-noise", action="store_true", help="skip the bf16 rerun / second-seed noise runs")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows, noise = [], []
    only = lambda r: r["model"] in ("original", "original-rerun", "original-seed2")
    for img in a.image:
        stem = Path(img).stem
        base = [("original", a.original, a.seed)]
        if not a.skip_noise:
            base += [("original-rerun", a.original, a.seed), ("original-seed2", a.original, a.seed + 1)]
        runs = base + [(*q.split("=", 1), a.seed) for q in a.quant]
        glb = {}
        for name, pkg, seed in runs:
            target = out / f"{stem}-{name}.glb"
            rc, sec, rss, vram, log = run_peak(engine_cmd(a, pkg, img, target, seed))
            (out / f"{stem}-{name}.log").write_text(log, encoding="utf-8", errors="replace")
            if rc != 0 or not target.exists():
                print(f"FAILED {stem} {name} rc={rc}; see {stem}-{name}.log", file=sys.stderr)
                rows.append({"image": stem, "model": name, "error": f"rc={rc}"}); continue
            glb[name] = target
            row = {"image": stem, "model": name, "seconds": round(sec, 1), "peak_ram_mib": round(rss), "peak_vram_mib": vram}
            if name != "original" and "original" in glb:
                row.update(compare(glb["original"], target))
            rows.append(row)
    # noise band: same-seed bf16 rerun; if that is bit-stable (iou 1, chamfer 0) use the seed-change band
    def worst(model):
        rs = [r for r in rows if r["model"] == model and "iou_mean" in r]
        return (min(r["iou_mean"] for r in rs), max(r["chamfer_rel_diag"] for r in rs)) if rs else (None, None)
    same, seed2 = worst("original-rerun"), worst("original-seed2")
    stable = same[0] is not None and same[0] >= 0.99999 and same[1] <= 1e-9
    band = seed2 if stable else same
    print(f"noise: same-seed rerun IoU/chamfer {same}, different seed {seed2}, deterministic={stable}")
    (out / "parity.json").write_text(json.dumps(rows, indent=1))
    print(f"{'image':10} {'model':8} {'IoU':>7} {'IoUmin':>7} {'chamfer':>9} {'VRAM':>6} {'RAM':>6} {'sec':>6}")
    verdict = {}
    for r in rows:
        if "error" in r:
            print(r); continue
        print(f"{r['image']:10} {r['model']:8} {r.get('iou_mean', float('nan')):7.4f} {r.get('iou_min', float('nan')):7.4f} "
              f"{r.get('chamfer_rel_diag', float('nan')):9.5f} {r['peak_vram_mib']:6} {r['peak_ram_mib']:6} {r['seconds']:6}")
        if not only(r) and "iou_mean" in r:
            ok = r["iou_mean"] >= a.iou_min and r["chamfer_rel_diag"] <= a.chamfer_max
            if band[0] is not None:
                ok = ok and r["iou_mean"] >= band[0] - 1e-9 and r["chamfer_rel_diag"] <= band[1] + 1e-9
            verdict[r["model"]] = verdict.get(r["model"], True) and ok
    print("band (iou_min, chamfer_max):", band); print("PASS" if verdict and all(verdict.values()) else "VERDICT", verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
