"""nuka.mcp.png -- the editor's PPM screenshots as PNG bytes, cropped and scaled, with the standard library."""

from __future__ import annotations

import struct
import zlib
from typing import Optional, Tuple


def read_ppm(path: str) -> Tuple[int, int, bytes]:
    """Width, height and RGB bytes of a binary (P6) PPM."""
    with open(path, "rb") as f:
        data = f.read()
    fields, at = [], 0
    while len(fields) < 4:
        while data[at:at + 1].isspace():
            at += 1
        if data[at:at + 1] == b"#":
            at = data.index(b"\n", at)
            continue
        end = at
        while not data[end:end + 1].isspace():
            end += 1
        fields.append(data[at:end])
        at = end
    if fields[0] != b"P6" or int(fields[3]) != 255:
        raise ValueError(f"{path} is not an 8-bit binary PPM")
    width, height = int(fields[1]), int(fields[2])
    return width, height, data[at + 1:at + 1 + width * height * 3]


def crop(width: int, height: int, rgb: bytes, rect: dict) -> Tuple[int, int, bytes]:
    """The ``{x, y, width, height}`` part of an image, clamped to it."""
    x0, y0 = max(0, int(rect["x"])), max(0, int(rect["y"]))
    x1, y1 = min(width, x0 + int(rect["width"])), min(height, y0 + int(rect["height"]))
    rows = [rgb[(y * width + x0) * 3:(y * width + x1) * 3] for y in range(y0, y1)]
    return x1 - x0, y1 - y0, b"".join(rows)


def shrink(width: int, height: int, rgb: bytes, max_width: int) -> Tuple[int, int, bytes]:
    """Every n-th pixel and row, so the width fits ``max_width``."""
    step = -(-width // max_width)
    if step <= 1:
        return width, height, rgb
    out_w, rows = len(range(0, width, step)), []
    for y in range(0, height, step):
        row = rgb[y * width * 3:(y + 1) * width * 3]
        picked = bytearray(out_w * 3)
        for c in range(3):
            picked[c::3] = row[c::3][::step]
        rows.append(bytes(picked))
    return out_w, len(rows), b"".join(rows)


def encode(width: int, height: int, rgb: bytes) -> bytes:
    """An RGB image as PNG bytes."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    stride = width * 3
    raw = b"".join(b"\x00" + rgb[y * stride:(y + 1) * stride] for y in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def ppm_to_png(path: str, rect: Optional[dict] = None, max_width: int = 0) -> Tuple[int, int, bytes]:
    """A screenshot as PNG: cropped to ``rect`` when given, then shrunk to ``max_width`` when set."""
    width, height, rgb = read_ppm(path)
    if rect is not None:
        width, height, rgb = crop(width, height, rgb, rect)
    if max_width > 0:
        width, height, rgb = shrink(width, height, rgb, max_width)
    return width, height, encode(width, height, rgb)
