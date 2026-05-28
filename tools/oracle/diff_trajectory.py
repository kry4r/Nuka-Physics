#!/usr/bin/env python3
"""Diff two NUKAGOLD float32 binary trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from golden_format import load_golden


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual", required=True, type=Path)
    parser.add_argument("--golden", required=True, type=Path)
    parser.add_argument("--tolerance", default=1.0e-4, type=float)
    args = parser.parse_args()

    try:
        actual_shape, actual = load_golden(args.actual)
        golden_shape, golden = load_golden(args.golden)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    comparable = (
        ("kind", actual_shape.kind, golden_shape.kind),
        ("samples", actual_shape.samples, golden_shape.samples),
        ("qpos", actual_shape.qpos, golden_shape.qpos),
        ("qvel", actual_shape.qvel, golden_shape.qvel),
        ("qacc", actual_shape.qacc, golden_shape.qacc),
    )
    for key, actual_value, golden_value in comparable:
        if actual_value != golden_value:
            raise SystemExit(
                f"shape mismatch for {key}: {actual_value} != {golden_value}"
            )
    if actual.shape != golden.shape:
        raise SystemExit(f"shape mismatch: {actual.shape} != {golden.shape}")
    max_abs = float(np.max(np.abs(actual - golden))) if actual.size else 0.0
    print(
        f"max_abs={max_abs:.9g} count={actual.size} "
        f"samples={actual_shape.samples} qpos={actual_shape.qpos} "
        f"qvel={actual_shape.qvel} qacc={actual_shape.qacc}"
    )
    if max_abs > args.tolerance:
        raise SystemExit(f"trajectory mismatch: {max_abs:.9g} > {args.tolerance:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
