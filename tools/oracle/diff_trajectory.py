#!/usr/bin/env python3
"""Diff two NUKAGOLD float32 binary trajectories."""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import numpy as np


def _load(path: Path) -> tuple[dict[str, int | str], np.ndarray]:
    data = path.read_bytes()
    if len(data) < 36 or data[:8] != b"NUKAGOLD":
        raise SystemExit(f"{path}: invalid NUKAGOLD header")
    version, kind, samples, qpos, qvel, qacc, name_size = struct.unpack_from("<IIIIIII",
                                                                            data,
                                                                            8)
    if version != 1:
        raise SystemExit(f"{path}: unsupported version {version}")
    name_start = 36
    payload_start = name_start + name_size
    if payload_start > len(data):
        raise SystemExit(f"{path}: truncated model name")
    name = data[name_start:payload_start].decode("utf-8")
    record_floats = qpos + qvel + (qvel if kind == 1 else qacc)
    expected_bytes = payload_start + samples * record_floats * 4
    if expected_bytes != len(data):
        raise SystemExit(
            f"{path}: size mismatch expected {expected_bytes} bytes, got {len(data)}"
        )
    payload = np.frombuffer(data[payload_start:], dtype="<f4").copy()
    shape = {
        "version": version,
        "kind": kind,
        "samples": samples,
        "qpos": qpos,
        "qvel": qvel,
        "qacc": qacc,
        "name": name,
    }
    return shape, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--tolerance", default=1.0e-4, type=float)
    args = parser.parse_args()

    actual_shape, actual = _load(args.actual)
    golden_shape, golden = _load(args.golden)
    comparable_keys = ("kind", "samples", "qpos", "qvel", "qacc")
    for key in comparable_keys:
        if actual_shape[key] != golden_shape[key]:
            raise SystemExit(
                f"shape mismatch for {key}: {actual_shape[key]} != {golden_shape[key]}"
            )
    if actual.shape != golden.shape:
        raise SystemExit(f"shape mismatch: {actual.shape} != {golden.shape}")
    max_abs = float(np.max(np.abs(actual - golden))) if actual.size else 0.0
    print(
        f"max_abs={max_abs:.9g} count={actual.size} "
        f"samples={actual_shape['samples']} qpos={actual_shape['qpos']} "
        f"qvel={actual_shape['qvel']} qacc={actual_shape['qacc']}"
    )
    if max_abs > args.tolerance:
        raise SystemExit(f"trajectory mismatch: {max_abs:.9g} > {args.tolerance:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
