#!/usr/bin/env python3
"""Render invariant CSV traces to compact SVG plots."""

from __future__ import annotations

import argparse
import csv
import html
import struct
import zlib
from collections import defaultdict
from pathlib import Path


def _safe_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _scale(value: float, in_min: float, in_max: float, out_min: float, out_max: float) -> float:
    if in_max <= in_min:
        return (out_min + out_max) * 0.5
    t = (value - in_min) / (in_max - in_min)
    return out_min + t * (out_max - out_min)


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _build_series(rows: list[dict[str, str]]) -> dict[str, list[tuple[int, float, float]]]:
    grouped: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for row in rows:
        env_id = row.get("env_id", "0")
        step = int(_safe_float(row.get("step", "0")))
        value = _safe_float(row.get("value", "0"))
        threshold = _safe_float(row.get("threshold", "0"))
        grouped[env_id].append((step, value, threshold))
    return grouped


def _bounds(grouped: dict[str, list[tuple[int, float, float]]]) -> tuple[int, int, float, float]:
    all_steps = [step for series in grouped.values() for step, _, _ in series]
    all_values = [
        value
        for series in grouped.values()
        for _, value, threshold in series
        for value in (value, threshold)
    ]
    x_min = min(all_steps) if all_steps else 0
    x_max = max(all_steps) if all_steps else 1
    y_min = min(0.0, min(all_values) if all_values else 0.0)
    y_max = max(1.0, max(all_values) if all_values else 1.0)
    y_pad = max((y_max - y_min) * 0.08, 1.0e-6)
    return x_min, x_max, y_min - y_pad, y_max + y_pad


def _write_svg(path: Path, invariant: str, rows: list[dict[str, str]]) -> None:
    width = 960
    height = 420
    left = 64
    right = 24
    top = 36
    bottom = 52

    grouped = _build_series(rows)
    x_min, x_max, y_min, y_max = _bounds(grouped)

    colors = ["#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2"]
    plot_w = width - left - right
    plot_h = height - top - bottom

    lines: list[str] = []
    for index, (env_id, series) in enumerate(sorted(grouped.items())):
        series.sort()
        points = [
            (
                _scale(step, x_min, x_max, left, left + plot_w),
                _scale(value, y_min, y_max, top + plot_h, top),
            )
            for step, value, _ in series
        ]
        color = colors[index % len(colors)]
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" '
            f'points="{_polyline(points)}"><title>env {html.escape(env_id)}</title></polyline>'
        )

    first_threshold = None
    for series in grouped.values():
        for _, _, threshold in series:
            first_threshold = threshold
            break
        if first_threshold is not None:
            break
    threshold_line = ""
    if first_threshold is not None:
        y = _scale(first_threshold, y_min, y_max, top + plot_h, top)
        threshold_line = (
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" '
            'stroke="#64748b" stroke-width="1.5" stroke-dasharray="6 4"/>'
        )

    title = html.escape(invariant.replace("_", " "))
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{left}" y="24" font-family="sans-serif" font-size="18" fill="#0f172a">{title}</text>
  <line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#334155"/>
  <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#334155"/>
  <text x="{left}" y="{height - 14}" font-family="sans-serif" font-size="12" fill="#475569">step {x_min} to {x_max}</text>
  <text x="8" y="{top + 12}" font-family="sans-serif" font-size="12" fill="#475569">{y_max:.4g}</text>
  <text x="8" y="{top + plot_h}" font-family="sans-serif" font-size="12" fill="#475569">{y_min:.4g}</text>
  {threshold_line}
  {''.join(lines)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def _set_pixel(image: list[bytearray], x: int, y: int, color: tuple[int, int, int]) -> None:
    if y < 0 or y >= len(image):
        return
    row = image[y]
    offset = x * 3
    if offset < 0 or offset + 2 >= len(row):
        return
    row[offset] = color[0]
    row[offset + 1] = color[1]
    row[offset + 2] = color[2]


def _draw_line(
    image: list[bytearray],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    dx = abs(x1 - x0)
    sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0)
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        _set_pixel(image, x0, y0, color)
        _set_pixel(image, x0 + 1, y0, color)
        _set_pixel(image, x0, y0 + 1, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _write_png(path: Path, rows: list[dict[str, str]]) -> None:
    width = 960
    height = 420
    left = 64
    right = 24
    top = 36
    bottom = 52
    plot_w = width - left - right
    plot_h = height - top - bottom
    grouped = _build_series(rows)
    x_min, x_max, y_min, y_max = _bounds(grouped)
    image = [bytearray([255, 255, 255] * width) for _ in range(height)]

    axis_color = (51, 65, 85)
    grid_color = (226, 232, 240)
    threshold_color = (100, 116, 139)
    colors = [
        (37, 99, 235),
        (22, 163, 74),
        (220, 38, 38),
        (147, 51, 234),
        (234, 88, 12),
        (8, 145, 178),
    ]

    for index in range(5):
        y = top + int(index * plot_h / 4)
        _draw_line(image, left, y, left + plot_w, y, grid_color)
    _draw_line(image, left, top + plot_h, left + plot_w, top + plot_h, axis_color)
    _draw_line(image, left, top, left, top + plot_h, axis_color)

    first_threshold = None
    for series in grouped.values():
        for _, _, threshold in series:
            first_threshold = threshold
            break
        if first_threshold is not None:
            break
    if first_threshold is not None:
        y = int(_scale(first_threshold, y_min, y_max, top + plot_h, top))
        for x in range(left, left + plot_w, 12):
            _draw_line(image, x, y, min(x + 6, left + plot_w), y, threshold_color)

    for index, (_, series) in enumerate(sorted(grouped.items())):
        series.sort()
        color = colors[index % len(colors)]
        points = [
            (
                int(_scale(step, x_min, x_max, left, left + plot_w)),
                int(_scale(value, y_min, y_max, top + plot_h, top)),
            )
            for step, value, _ in series
        ]
        for first, second in zip(points, points[1:]):
            _draw_line(image, first[0], first[1], second[0], second[1], color)
        if len(points) == 1:
            _set_pixel(image, points[0][0], points[0][1], color)

    raw = b"".join(b"\x00" + bytes(row) for row in image)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + _png_chunk(b"IDAT", zlib.compress(raw, level=6))
        + _png_chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    rows = _read_rows(args.csv_path)
    if not rows:
        raise SystemExit(f"{args.csv_path} has no invariant rows")

    out_dir = args.out_dir or args.csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    by_invariant: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        invariant = row.get("invariant", "unknown")
        by_invariant[invariant].append(row)

    for invariant, invariant_rows in sorted(by_invariant.items()):
        png_output = out_dir / f"invariants_{invariant}.png"
        svg_output = out_dir / f"invariants_{invariant}.svg"
        _write_png(png_output, invariant_rows)
        _write_svg(svg_output, invariant, invariant_rows)
        print(png_output)
        print(svg_output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
