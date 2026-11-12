"""Add ComfyUI-compatible GGUF architecture metadata to the P-H3 Image Ojon GGUF.

The stable-diffusion.cpp converter emits no general.architecture metadata,
which causes ComfyUI-GGUF's legacy shape detector to reject P-H3.
A tested P-H3-Image-GGUF distribution uses the recognized 'wan' routing tag.
This is a loader compatibility tag, NOT a claim that P-H3 is a Wan model.

The source is never modified. This tool writes a distinct file and preserves
every tensor payload byte-for-byte by copying the raw GGUF data section.
No model weights are loaded into RAM and no re-quantization is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import struct
from pathlib import Path


def u32(f):
    b = f.read(4)
    if len(b) != 4:
        raise ValueError("Truncated GGUF header (u32)")
    return struct.unpack("<I", b)[0]


def u64(f):
    b = f.read(8)
    if len(b) != 8:
        raise ValueError("Truncated GGUF header (u64)")
    return struct.unpack("<Q", b)[0]


def align32(n):
    return (n + 31) & ~31


def inspect(path: Path):
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("Not a GGUF file")
        version = u32(f)
        count = u64(f)
        kv_count = u64(f)
        if version != 3 or not 1 <= count <= 100_000:
            raise ValueError(f"Unsupported GGUF version/count: {version}/{count}")
        # Specialized repair: refuse unfamiliar GGUF header layouts instead
        # of guessing where existing arbitrary metadata ends.
        if kv_count != 0:
            raise ValueError(
                f"GGUF has {kv_count} metadata items; this tool only repairs "
                "the known Ojon P-H3 output without metadata"
            )
        seen = set()
        for _ in range(count):
            name_len = u64(f)
            if not 1 <= name_len <= 4096:
                raise ValueError("Suspicious tensor name length")
            raw = f.read(name_len)
            if len(raw) != name_len:
                raise ValueError("Truncated tensor name")
            name = raw.decode("utf-8")
            seen.add(name)
            dims = u32(f)
            if not 1 <= dims <= 8:
                raise ValueError(f"Bad tensor rank: {dims}")
            for _ in range(dims):
                _ = u64(f)
            _ = u32(f)  # ggml tensor type
            _ = u64(f)  # relative offset within payload
        info_end = f.tell()
    expected = {
        "model.diffusion_model.adaln_t_table",
        "model.diffusion_model.blocks.0.attn.qkv_proj.weight",
    }
    if not expected.issubset(seen):
        raise ValueError(
            "This is not the expected P-H3 Image converter output; refusing to tag it"
        )
    src_data = align32(info_end)
    size = path.stat().st_size
    if src_data >= size:
        raise ValueError("Missing tensor payload")
    return {
        "version": version,
        "tensor_count": count,
        "metadata_count": kv_count,
        "tensor_infos_end": info_end,
        "data_start": src_data,
        "payload_bytes": size - src_data,
    }


def sample_digest(path, start, size):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for pos, amount in ((start, min(size, 1_048_576)),
                            (start + max(0, size - 1_048_576),
                             min(size, 1_048_576))):
            f.seek(pos)
            h.update(f.read(amount))
    return h.hexdigest()


def repair(source: Path, dest: Path):
    info = inspect(source)
    if source.resolve() == dest.resolve():
        raise ValueError("Refusing to overwrite the source")
    if dest.exists():
        raise FileExistsError(f"Output already exists: {dest}")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    if tmp.exists():
        raise FileExistsError(f"Temporary output already exists: {tmp}")
    kv_key = b"general.architecture"
    kv_value = b"wan"
    new_meta = (struct.pack("<Q", len(kv_key)) + kv_key
                + struct.pack("<I", 8)  # GGUFValueType.STRING
                + struct.pack("<Q", len(kv_value)) + kv_value)
    new_head_end = info["tensor_infos_end"] + len(new_meta)
    new_data_start = align32(new_head_end)
    expected_size = new_data_start + info["payload_bytes"]
    print(f"source={source}", flush=True)
    print(f"output={dest}", flush=True)
    print(f"tensors={info['tensor_count']} payload_bytes={info['payload_bytes']}", flush=True)
    print(f"new_architecture_tag=wan expected_bytes={expected_size}", flush=True)
    try:
        with source.open("rb") as src, tmp.open("xb") as dst:
            # The fixed GGUF header, same version and tensor descriptors.
            original = src.read(24)
            dst.write(original[:16] + struct.pack("<Q", 1))
            dst.write(new_meta)
            remaining_info_bytes = info["tensor_infos_end"] - 24
            while remaining_info_bytes:
                chunk = src.read(min(4 * 1024 * 1024, remaining_info_bytes))
                if not chunk:
                    raise IOError("Unexpected EOF in tensor descriptors")
                dst.write(chunk)
                remaining_info_bytes -= len(chunk)
            if dst.tell() != new_head_end:
                raise AssertionError("Header boundary mismatch")
            dst.write(b"\x00" * (new_data_start - dst.tell()))
            src.seek(info["data_start"])
            written = 0
            remaining = info["payload_bytes"]
            while remaining:
                chunk = src.read(min(4 * 1024 * 1024, remaining))
                if not chunk:
                    raise IOError("Unexpected EOF in tensor data")
                dst.write(chunk)
                remaining -= len(chunk)
                written += len(chunk)
                if written % (1024 * 1024 * 1024) < len(chunk):
                    print(f"copied={written // (1024 * 1024 * 1024)} GiB", flush=True)
            dst.flush()
            os.fsync(dst.fileno())
        if tmp.stat().st_size != expected_size:
            raise IOError("Incorrect output size")
        before = sample_digest(source, info["data_start"], info["payload_bytes"])
        after = sample_digest(tmp, new_data_start, info["payload_bytes"])
        if before != after:
            raise IOError("Tensor payload spot-check mismatch")
        os.replace(tmp, dest)
        print(f"VERIFIED: {dest} bytes={expected_size} sample_sha256={after}", flush=True)
    except Exception:
        # Leave any partially produced artifact clearly named as .partial,
        # rather than publishing it as a usable GGUF.
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("dest", type=Path)
    parser.add_argument("--confirm", action="store_true",
                        help="Actually create the separate repaired copy")
    args = parser.parse_args()
    meta = inspect(args.source)
    print(f"INSPECT_OK version={meta['version']} tensors={meta['tensor_count']} "
          f"kv={meta['metadata_count']} payload_start={meta['data_start']} "
          f"bytes={meta['payload_bytes']}", flush=True)
    if args.confirm:
        repair(args.source, args.dest)
    else:
        print("DRY RUN ONLY; add --confirm to write a repaired copy", flush=True)
