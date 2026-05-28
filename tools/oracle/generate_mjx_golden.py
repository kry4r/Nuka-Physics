#!/usr/bin/env python3
"""Generate v0.1 MuJoCo/MJX-compatible golden trajectory files.

This script intentionally writes only to paths requested by the caller. The
repository's golden directory is owner-protected; humans should run this script
and add the resulting files with Git LFS.
"""

from __future__ import annotations

import argparse
import struct
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np


KIND_RANDOM_QACC = 1
KIND_JOINT_TRAJECTORY = 2


def _load_mjx_deps():
    try:
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx
    except ImportError as error:
        raise SystemExit(
            "generate_mjx_golden.py requires the owner-provided MJX environment "
            "(jax, mujoco, and mujoco.mjx)"
        ) from error
    return jax, jnp, mujoco, mjx


def _fixed_base_model_xml(source: Path, work: Path) -> Path:
    """Create a temporary fixed-base MJCF matching Nuka's v0.1 ABA importer."""
    if (source.parent / "assets").exists():
        shutil.copytree(source.parent / "assets", work / "assets")
    text = source.read_text()
    text = text.replace("<freejoint/>", "")
    text = re.sub(r"\n\s*<keyframe>.*?</keyframe>\s*", "\n", text, flags=re.S)
    text = re.sub(r"\n\s*<actuator>.*?</actuator>\s*", "\n", text, flags=re.S)
    out = work / source.name
    out.write_text(text)
    return out


def _nuka_qpos(qpos: np.ndarray) -> np.ndarray:
    """Prepend Nuka's fixed root-link slot to MJX fixed-base joint vectors."""
    return np.concatenate((np.zeros(1, dtype=np.float32),
                           np.asarray(qpos, dtype=np.float32)))


def _nuka_qvel(qvel: np.ndarray) -> np.ndarray:
    return _nuka_qpos(qvel)


def _nuka_tau(tau: np.ndarray) -> np.ndarray:
    return _nuka_qpos(tau)


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
    jax, jnp, mujoco, mjx = _load_mjx_deps()
    with tempfile.TemporaryDirectory(prefix="nuka_mjx_fixed_base_") as tmp:
        fixed_model_path = _fixed_base_model_xml(model_path, Path(tmp))
        model = mujoco.MjModel.from_xml_path(str(fixed_model_path))
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        mjx_model = mjx.put_model(model)
        rng = np.random.default_rng(seed=42)
        nuka_dofs = model.nv + 1
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as out:
            _write_header(out,
                          model_path.name,
                          KIND_RANDOM_QACC,
                          sample_count,
                          nuka_dofs,
                          nuka_dofs,
                          nuka_dofs)
            for _ in range(sample_count):
                qpos = rng.uniform(-0.2, 0.2, model.nq).astype(np.float32)
                qvel = rng.uniform(-0.5, 0.5, model.nv).astype(np.float32)
                tau = rng.uniform(-2.0, 2.0, model.nv).astype(np.float32)
                data = mjx.make_data(mjx_model).replace(
                    qpos=jnp.asarray(qpos),
                    qvel=jnp.asarray(qvel),
                    qfrc_applied=jnp.asarray(tau),
                )
                data = mjx.forward(mjx_model, data)
                qacc = np.asarray(jax.device_get(data.qacc), dtype=np.float32)
                out.write(_nuka_qpos(qpos).tobytes())
                out.write(_nuka_qvel(qvel).tobytes())
                out.write(_nuka_tau(tau).tobytes())
                out.write(_nuka_qvel(qacc).tobytes())


def generate_stand_trajectory(model_path: Path,
                              out_path: Path,
                              step_count: int,
                              dt: float) -> None:
    jax, _, mujoco, mjx = _load_mjx_deps()
    with tempfile.TemporaryDirectory(prefix="nuka_mjx_fixed_base_") as tmp:
        fixed_model_path = _fixed_base_model_xml(model_path, Path(tmp))
        model = mujoco.MjModel.from_xml_path(str(fixed_model_path))
        model.opt.timestep = dt
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        model.geom_contype[:] = 0
        model.geom_conaffinity[:] = 0
        mjx_model = mjx.put_model(model)
        data = mjx.make_data(mjx_model)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as out:
            _write_header(out,
                          model_path.name,
                          KIND_JOINT_TRAJECTORY,
                          step_count,
                          model.nq + 1,
                          0,
                          0)
            for _ in range(step_count):
                data = mjx.step(mjx_model, data)
                qpos = np.asarray(jax.device_get(data.qpos), dtype=np.float32)
                out.write(_nuka_qpos(qpos).tobytes())


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
