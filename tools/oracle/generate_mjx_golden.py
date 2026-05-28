#!/usr/bin/env python3
"""Generate v0.1 MuJoCo/MJX-compatible golden trajectory files.

This script intentionally writes only to paths requested by the caller. The
repository's golden directory is owner-protected; humans should run this script
and add the resulting files with Git LFS.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path

import mujoco
import numpy as np


KIND_RANDOM_QACC = 1
KIND_JOINT_TRAJECTORY = 2


def _write_header(out,
                  model_name: str,
                  kind: int,
                  sample_count: int,
                  qpos_count: int,
                  qvel_count: int,
                  qacc_count: int) -> None:
    encoded = model_name.encode("utf-8")
    out.write(b"NUKAGOLD")
    out.write(struct.pack("<IIIIII",
                          1,
                          kind,
                          sample_count,
                          qpos_count,
                          qvel_count,
                          qacc_count))
    out.write(struct.pack("<I", len(encoded)))
    out.write(encoded)


def generate_random_samples(model_path: Path, out_path: Path, sample_count: int) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    rng = np.random.default_rng(seed=42)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as out:
        _write_header(out,
                      model_path.name,
                      KIND_RANDOM_QACC,
                      sample_count,
                      model.nq,
                      model.nv,
                      model.nv)
        for _ in range(sample_count):
            data = mujoco.MjData(model)
            qpos = rng.uniform(-0.2, 0.2, model.nq).astype(np.float32)
            qvel = rng.uniform(-0.5, 0.5, model.nv).astype(np.float32)
            tau = rng.uniform(-2.0, 2.0, model.nv).astype(np.float32)
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            data.qfrc_applied[:] = tau
            mujoco.mj_forward(model, data)
            qacc = np.asarray(data.qacc, dtype=np.float32)
            out.write(qpos.tobytes())
            out.write(qvel.tobytes())
            out.write(tau.tobytes())
            out.write(qacc.tobytes())


def generate_stand_trajectory(model_path: Path,
                              out_path: Path,
                              step_count: int,
                              dt: float) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = dt
    data = mujoco.MjData(model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as out:
        _write_header(out,
                      model_path.name,
                      KIND_JOINT_TRAJECTORY,
                      step_count,
                      model.nq,
                      model.nv,
                      model.nv)
        for _ in range(step_count):
            mujoco.mj_step(model, data)
            qpos = np.asarray(data.qpos, dtype=np.float32)
            qvel = np.asarray(data.qvel, dtype=np.float32)
            qacc = np.asarray(data.qacc, dtype=np.float32)
            out.write(qpos.tobytes())
            out.write(qvel.tobytes())
            out.write(qacc.tobytes())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode",
                        choices=("random-qacc", "stand-trajectory"),
                        default="random-qacc")
    parser.add_argument("--samples", default=1000, type=int)
    parser.add_argument("--steps", default=1200, type=int)
    parser.add_argument("--dt", default=1.0 / 240.0, type=float)
    args = parser.parse_args()
    if args.mode == "random-qacc":
        generate_random_samples(args.model, args.out, args.samples)
    else:
        generate_stand_trajectory(args.model, args.out, args.steps, args.dt)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
