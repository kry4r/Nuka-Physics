#!/usr/bin/env python3
"""Validate CUDA Featherstone ABA against MuJoCo fixed-base robot oracles."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import mujoco


def _fixed_base_xml(source: Path, work: Path) -> Path:
    if (source.parent / "assets").exists():
        shutil.copytree(source.parent / "assets", work / "assets")
    text = source.read_text()
    text = text.replace("<freejoint/>", "")
    text = re.sub(r"\n\s*<keyframe>.*?</keyframe>\s*", "\n", text, flags=re.S)
    text = re.sub(r"\n\s*<actuator>.*?</actuator>\s*", "\n", text, flags=re.S)
    out = work / source.name
    out.write_text(text)
    return out


def _sample_for_model(model: mujoco.MjModel) -> list[tuple[str, float, float, float]]:
    samples: list[tuple[str, float, float, float]] = []
    for joint_index in range(model.njnt):
        joint = model.joint(joint_index)
        name = joint.name
        qpos_adr = int(model.jnt_qposadr[joint_index])
        dof_adr = int(model.jnt_dofadr[joint_index])
        lower, upper = model.jnt_range[joint_index]
        if lower < upper:
            q = float(0.35 * lower + 0.65 * upper)
        else:
            q = 0.03 * float(joint_index + 1)
        qdot = 0.07 * float((joint_index % 5) - 2)
        tau = 0.4 * float((joint_index % 7) - 3)
        samples.append((name, q, qdot, tau))
        # Keep MuJoCo addresses visible through the deterministic sample formula.
        assert qpos_adr >= 0 and dof_adr >= 0
    return samples


def _write_probe_input(path: Path, samples: list[tuple[str, float, float, float]]) -> None:
    lines = ["gravity_z -9.81\n"]
    for name, q, qdot, tau in samples:
        lines.append(f"joint {name} {q:.9g} {qdot:.9g} {tau:.9g}\n")
    path.write_text("".join(lines))


def _run_nuka_probe(probe: Path,
                    xml: Path,
                    sample_path: Path) -> dict[str, float]:
    output = subprocess.check_output(
        [str(probe), str(xml), str(sample_path)],
        text=True,
    )
    result: dict[str, float] = {}
    for line in output.splitlines():
        name, value = line.split()
        result[name] = float(value)
    return result


def _run_mujoco(xml: Path,
                samples: list[tuple[str, float, float, float]]) -> dict[str, float]:
    model = mujoco.MjModel.from_xml_path(str(xml))
    data = mujoco.MjData(model)
    for name, q, qdot, tau in samples:
        joint_id = model.joint(name).id
        data.qpos[model.jnt_qposadr[joint_id]] = q
        data.qvel[model.jnt_dofadr[joint_id]] = qdot
        data.qfrc_applied[model.jnt_dofadr[joint_id]] = tau
    mujoco.mj_forward(model, data)
    return {
        name: float(data.qacc[model.jnt_dofadr[model.joint(name).id]])
        for name, *_ in samples
    }


def validate_model(model_xml: Path,
                   probe: Path,
                   tolerance: float) -> tuple[int, float]:
    with tempfile.TemporaryDirectory(prefix="nuka_aba_oracle_") as tmp:
        work = Path(tmp)
        fixed_xml = _fixed_base_xml(model_xml, work)
        model = mujoco.MjModel.from_xml_path(str(fixed_xml))
        samples = _sample_for_model(model)
        sample_path = work / "sample.txt"
        _write_probe_input(sample_path, samples)

        nuka = _run_nuka_probe(probe, fixed_xml, sample_path)
        oracle = _run_mujoco(fixed_xml, samples)

    max_abs = 0.0
    worst = ""
    for name, *_ in samples:
        diff = abs(nuka[name] - oracle[name])
        if diff > max_abs:
            max_abs = diff
            worst = name
    print(f"{model_xml}: joints={len(samples)} max_abs_qddot={max_abs:.9g} worst={worst}")
    if max_abs > tolerance:
        raise SystemExit(
            f"ABA oracle mismatch for {model_xml}: {max_abs:.9g} > {tolerance:.9g}"
        )
    return len(samples), max_abs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe", required=True, type=Path)
    parser.add_argument("--model", required=True, action="append", type=Path)
    parser.add_argument("--tolerance", default=1.0e-3, type=float)
    args = parser.parse_args()

    total_joints = 0
    max_seen = 0.0
    for model in args.model:
        joints, max_abs = validate_model(model, args.probe, args.tolerance)
        total_joints += joints
        max_seen = max(max_seen, max_abs)
    print(f"ABA oracle validation passed: models={len(args.model)} "
          f"joints={total_joints} max_abs_qddot={max_seen:.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
