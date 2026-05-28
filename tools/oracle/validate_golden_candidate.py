#!/usr/bin/env python3
"""Validate owner-generated NUKAGOLD candidates outside protected storage."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from golden_format import (
    KIND_JOINT_TRAJECTORY,
    KIND_RANDOM_QACC,
    GoldenHeader,
    load_golden,
)


KIND_BY_NAME = {
    "random-qacc": KIND_RANDOM_QACC,
    "joint-trajectory": KIND_JOINT_TRAJECTORY,
}


def _require_equal(label: str, actual: int | str, expected: int | str | None) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"{label} mismatch: got {actual!r}, expected {expected!r}")


def _root_zero_offsets(header: GoldenHeader) -> list[int]:
    if header.kind == KIND_RANDOM_QACC:
        qpos = 0
        qvel = header.qpos
        tau = qvel + header.qvel
        qacc = tau + header.qvel
        return [qpos, qvel, tau, qacc]
    if header.kind == KIND_JOINT_TRAJECTORY:
        return [0]
    return []


def _validate_root_zero(header: GoldenHeader, records: np.ndarray) -> float:
    offsets = _root_zero_offsets(header)
    if not offsets:
        return 0.0
    values = records[:, offsets]
    return float(np.max(np.abs(values))) if values.size else 0.0


def validate(args: argparse.Namespace) -> None:
    header, payload = load_golden(args.path)
    expected_kind = KIND_BY_NAME.get(args.kind) if args.kind else None
    _require_equal("kind", header.kind, expected_kind)
    _require_equal("sample_count", header.samples, args.samples)
    _require_equal("qpos_count", header.qpos, args.qpos_count)
    _require_equal("qvel_count", header.qvel, args.qvel_count)
    _require_equal("qacc_count", header.qacc, args.qacc_count)
    _require_equal("model_name", header.name, args.model_name)

    if not np.all(np.isfinite(payload)):
        raise ValueError("payload contains NaN or Inf")

    records = payload.reshape(header.samples, header.record_floats)
    root_zero_max = 0.0
    if not args.no_root_zero_check:
        root_zero_max = _validate_root_zero(header, records)
        if root_zero_max > args.root_zero_tolerance:
            raise ValueError(
                "fixed-root slot is nonzero: "
                f"max_abs={root_zero_max:.9g} > {args.root_zero_tolerance:.9g}"
            )

    payload_min = float(np.min(payload)) if payload.size else 0.0
    payload_max = float(np.max(payload)) if payload.size else 0.0
    print(
        f"PASS: {args.path} kind={header.kind_name} samples={header.samples} "
        f"qpos={header.qpos} qvel={header.qvel} qacc={header.qacc} "
        f"record_floats={header.record_floats} model={header.name!r}"
    )
    print(
        f"payload_count={payload.size} min={payload_min:.9g} "
        f"max={payload_max:.9g} root_zero_max_abs={root_zero_max:.9g}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, type=Path)
    parser.add_argument("--kind", choices=sorted(KIND_BY_NAME))
    parser.add_argument("--samples", type=int)
    parser.add_argument("--qpos-count", type=int)
    parser.add_argument("--qvel-count", type=int)
    parser.add_argument("--qacc-count", type=int)
    parser.add_argument("--model-name")
    parser.add_argument("--root-zero-tolerance", default=0.0, type=float)
    parser.add_argument(
        "--no-root-zero-check",
        action="store_true",
        help="skip the v0.1 fixed-root zero-slot invariant check",
    )
    args = parser.parse_args()
    try:
        validate(args)
    except ValueError as error:
        raise SystemExit(f"FAIL: {error}") from error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
