#!/usr/bin/env python3
"""Convert the opt-in Ojon CPU-render test PPM to PNG without dependencies."""
import pathlib
import struct
import sys
import zlib


def png_chunk(kind: bytes, body: bytes) -> bytes:
    return (struct.pack(">I", len(body)) + kind + body
            + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF))


def convert(source: pathlib.Path, output: pathlib.Path) -> None:
    data = source.read_bytes()
    header = data.split(b"\n", 3)
    if len(header) != 4 or header[0] != b"P6" or header[2] != b"255":
        raise ValueError("expected Ojon's binary P6 screenshot")
    dims = header[1].split()
    if len(dims) != 2:
        raise ValueError("missing PPM dimensions")
    width, height = map(int, dims)
    if width <= 0 or height <= 0 or width * height > 16_000_000:
        raise ValueError("invalid or excessive screenshot dimensions")
    pixels = header[3]
    stride = width * 3
    if len(pixels) != stride * height:
        raise ValueError("screenshot pixel count does not match header")
    scanlines = b"".join(
        b"\0" + pixels[y * stride:(y + 1) * stride]
        for y in range(height)
    )
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    output.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + png_chunk(b"IHDR", ihdr)
        + png_chunk(b"IDAT", zlib.compress(scanlines, 7))
        + png_chunk(b"IEND", b"")
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: ppm_to_png.py INPUT.ppm OUTPUT.png")
    convert(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]))
