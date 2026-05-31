#!/usr/bin/env python3
"""Generate v0.1 MuJoCo/MJX-compatible golden trajectory files.

This script intentionally writes only to paths requested by the caller. The
repository's golden directory is owner-protected; humans should run this script
and add the resulting files with Git LFS.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import struct
import tempfile
import xml.sax.saxutils
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


KIND_RANDOM_QACC = 1
KIND_JOINT_TRAJECTORY = 2


@dataclass
class _UsdPrim:
    type: str
    name: str
    path: str
    parent_path: str
    order: int
    has_rigid_body_api: bool = False
    rigid_body_enabled: bool = False
    kinematic_enabled: bool = False
    mass: float = 1.0
    diagonal_inertia: tuple[float, float, float] = (1.0, 1.0, 1.0)
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    body0_path: str = ""
    body1_path: str = ""
    joint_path: str = ""
    axis_token: str = "Z"
    nuka_type: str = ""
    has_initial_position: bool = False
    initial_position: float = 0.0
    lower_limit: float = -3.14159
    upper_limit: float = 3.14159
    gain: float = 1.0
    force_limit: float = 0.0
    children: list["_UsdPrim"] = field(default_factory=list)


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


def _trim(text: str) -> str:
    return text.strip()


def _join_path(parent: str, name: str) -> str:
    return f"/{name}" if not parent else f"{parent}/{name}"


def _parse_def_line(line: str) -> tuple[str, str] | None:
    if not line.startswith("def "):
        return None
    parts = line.split(maxsplit=2)
    if len(parts) < 3:
        return None
    first_quote = line.find('"')
    second_quote = line.find('"', first_quote + 1)
    if first_quote < 0 or second_quote < 0:
        return None
    return parts[1], line[first_quote + 1:second_quote]


def _parse_bool(line: str, key: str) -> bool | None:
    if key not in line or "=" not in line:
        return None
    rhs = line.split("=", 1)[1].strip().lower()
    if rhs.startswith("true"):
        return True
    if rhs.startswith("false"):
        return False
    return None


def _parse_float(line: str, key: str) -> float | None:
    if key not in line or "=" not in line:
        return None
    rhs = line.split("=", 1)[1].strip()
    match = re.match(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", rhs)
    return float(match.group(0)) if match else None


def _parse_vec3(line: str, key: str) -> tuple[float, float, float] | None:
    if key not in line:
        return None
    match = re.search(r"\(([^)]*)\)", line)
    if not match:
        return None
    values = [float(item) for item in re.split(r"[,\s]+", match.group(1).strip()) if item]
    if len(values) != 3:
        return None
    return values[0], values[1], values[2]


def _parse_relationship(line: str, key: str) -> str | None:
    if key not in line:
        return None
    match = re.search(r"<([^>]*)>", line)
    return match.group(1) if match else None


def _parse_token(line: str, key: str) -> str | None:
    if key not in line:
        return None
    match = re.search(r'"([^"]*)"', line)
    return match.group(1) if match else None


def _apply_usd_property(line: str, prim: _UsdPrim) -> None:
    if "apiSchemas" in line:
        prim.has_rigid_body_api = (
            prim.has_rigid_body_api or "PhysicsRigidBodyAPI" in line
        )
    for key, attr in (
        ("physics:rigidBodyEnabled", "rigid_body_enabled"),
        ("physics:kinematicEnabled", "kinematic_enabled"),
    ):
        value = _parse_bool(line, key)
        if value is not None:
            setattr(prim, attr, value)
    for key, attr in (
        ("physics:mass", "mass"),
        ("physics:lowerLimit", "lower_limit"),
        ("physics:upperLimit", "upper_limit"),
    ):
        value = _parse_float(line, key)
        if value is not None:
            setattr(prim, attr, value)
    for key, attr in (
        ("physics:diagonalInertia", "diagonal_inertia"),
        ("xformOp:translate", "translate"),
    ):
        value = _parse_vec3(line, key)
        if value is not None:
            setattr(prim, attr, value)
    for key, attr in (
        ("physics:body0", "body0_path"),
        ("physics:body1", "body1_path"),
        ("nuka:joint", "joint_path"),
    ):
        value = _parse_relationship(line, key)
        if value is not None:
            setattr(prim, attr, value)
    for key, attr in (
        ("physics:axis", "axis_token"),
        ("nuka:type", "nuka_type"),
    ):
        value = _parse_token(line, key)
        if value is not None:
            setattr(prim, attr, value)
    for key, attr in (
        ("nuka:gain", "gain"),
        ("nuka:forceLimit", "force_limit"),
    ):
        value = _parse_float(line, key)
        if value is not None:
            setattr(prim, attr, value)
    value = _parse_float(line, "nuka:initialPosition")
    if value is not None:
        prim.initial_position = value
        prim.has_initial_position = True


def _parse_usda_prims(source: Path) -> list[_UsdPrim]:
    prims: list[_UsdPrim] = []
    stack: list[int] = []
    pending: int | None = None
    for raw_line in source.read_text().splitlines():
        line = _trim(raw_line)
        if not line or line.startswith("#"):
            continue
        definition = _parse_def_line(line)
        if definition is not None:
            prim_type, name = definition
            parent_path = prims[stack[-1]].path if stack else ""
            prim = _UsdPrim(
                type=prim_type,
                name=name,
                path=_join_path(parent_path, name),
                parent_path=parent_path,
                order=len(prims),
            )
            _apply_usd_property(line, prim)
            prims.append(prim)
            pending = len(prims) - 1
            if "{" in line:
                stack.append(pending)
                pending = None
            continue
        if pending is not None:
            _apply_usd_property(line, prims[pending])
        elif stack:
            _apply_usd_property(line, prims[stack[-1]])
        if "{" in line and pending is not None:
            stack.append(pending)
            pending = None
        if "}" in line and stack:
            stack.pop()
    return prims


def _axis_tuple(axis: str) -> tuple[float, float, float]:
    if axis.lower() == "x":
        return 1.0, 0.0, 0.0
    if axis.lower() == "y":
        return 0.0, 1.0, 0.0
    return 0.0, 0.0, 1.0


def _fmt(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _xml_name(name: str) -> str:
    return xml.sax.saxutils.escape(name, {'"': "&quot;"})


def _is_rigid_body(prim: _UsdPrim) -> bool:
    return prim.has_rigid_body_api and prim.rigid_body_enabled


def _is_joint(prim: _UsdPrim) -> bool:
    return prim.type.startswith("Physics") and "Joint" in prim.type


def _usda_to_fixed_base_mjcf(source: Path, work: Path) -> Path:
    """Create temporary MJCF matching the v0.1 ASCII USD articulation contract."""
    prims = _parse_usda_prims(source)
    bodies = {prim.path: prim for prim in prims if _is_rigid_body(prim)}
    joints = [prim for prim in prims if _is_joint(prim)]
    children_by_parent: dict[str, list[_UsdPrim]] = {}
    child_paths: set[str] = set()
    joint_for_child: dict[str, _UsdPrim] = {}
    for joint in joints:
        if joint.type != "PhysicsRevoluteJoint":
            raise SystemExit(
                f"{source}: MJX stand oracle supports only revolute USD joints; "
                f"got {joint.type} for {joint.name}"
            )
        if joint.body0_path not in bodies or joint.body1_path not in bodies:
            raise SystemExit(f"{source}: joint {joint.name} references an unknown body")
        children_by_parent.setdefault(joint.body0_path, []).append(joint)
        child_paths.add(joint.body1_path)
        joint_for_child[joint.body1_path] = joint
    for siblings in children_by_parent.values():
        siblings.sort(key=lambda item: item.order)

    roots = [prim for prim in prims if _is_rigid_body(prim) and prim.path not in child_paths]
    if not roots:
        raise SystemExit(f"{source}: no fixed-base USD articulation root found")

    lines = [
        '<mujoco model="nuka_go2_stand_usd">',
        '  <compiler angle="radian" coordinate="local"/>',
        '  <option integrator="Euler" gravity="0 0 -9.81"/>',
        '  <worldbody>',
    ]

    def emit_body(body: _UsdPrim, indent: int) -> None:
        prefix = " " * indent
        lines.append(
            f'{prefix}<body name="{_xml_name(body.name)}" pos="{_fmt(body.translate)}">'
        )
        joint = joint_for_child.get(body.path)
        if joint is not None:
            axis = _axis_tuple(joint.axis_token)
            lines.append(
                f'{prefix}  <joint name="{_xml_name(joint.name)}" type="hinge" '
                f'axis="{_fmt(axis)}" range="{joint.lower_limit:.9g} '
                f'{joint.upper_limit:.9g}" limited="true" damping="0" armature="0"/>'
            )
        if body.mass > 0.0 and not body.kinematic_enabled:
            lines.append(
                f'{prefix}  <inertial pos="0 0 0" mass="{body.mass:.9g}" '
                f'diaginertia="{_fmt(body.diagonal_inertia)}"/>'
            )
        for child_joint in children_by_parent.get(body.path, []):
            emit_body(bodies[child_joint.body1_path], indent + 2)
        lines.append(f"{prefix}</body>")

    for root in roots:
        emit_body(root, 4)
    lines.append("  </worldbody>")
    lines.append("</mujoco>")
    out = work / f"{source.stem}_mjx.xml"
    out.write_text("\n".join(lines) + "\n")
    return out


def _usd_joint_order(source: Path) -> list[str] | None:
    if source.suffix.lower() not in (".usd", ".usda"):
        return None
    prims = _parse_usda_prims(source)
    body_order = {prim.path: index for index, prim in enumerate(prims) if _is_rigid_body(prim)}
    children_by_parent: dict[str, list[_UsdPrim]] = {}
    child_paths: set[str] = set()
    for prim in prims:
        if not _is_joint(prim):
            continue
        children_by_parent.setdefault(prim.body0_path, []).append(prim)
        child_paths.add(prim.body1_path)
    for siblings in children_by_parent.values():
        siblings.sort(key=lambda item: item.order, reverse=True)
    roots = [
        prim for prim in prims
        if _is_rigid_body(prim) and prim.path not in child_paths
    ]
    roots.sort(key=lambda item: item.order)

    order: list[str] = []
    for root in roots:
        stack = [root.path]
        while stack:
            body_path = stack.pop()
            incoming = [
                prim for prim in prims
                if _is_joint(prim) and prim.body1_path == body_path
            ]
            if incoming:
                order.append(incoming[0].name)
            for joint in children_by_parent.get(body_path, []):
                if joint.body1_path in body_order:
                    stack.append(joint.body1_path)
    return order


def _usd_initial_positions(source: Path, joint_order: list[str] | None) -> np.ndarray | None:
    if source.suffix.lower() not in (".usd", ".usda") or joint_order is None:
        return None
    prims = _parse_usda_prims(source)
    initial_by_name = {
        prim.name: prim.initial_position
        for prim in prims
        if _is_joint(prim) and prim.has_initial_position
    }
    return np.asarray(
        [initial_by_name.get(name, 0.0) for name in joint_order],
        dtype=np.float32,
    )


def _usd_drive_arrays(
    source: Path,
    joint_order: list[str] | None,
) -> tuple[np.ndarray, np.ndarray] | None:
    if source.suffix.lower() not in (".usd", ".usda") or joint_order is None:
        return None
    prims = _parse_usda_prims(source)
    joint_name_by_path = {
        prim.path: prim.name
        for prim in prims
        if _is_joint(prim)
    }
    drive_by_joint: dict[str, tuple[float, float]] = {}
    for prim in prims:
        if prim.type != "NukaActuator" or prim.nuka_type.lower() != "position":
            continue
        joint_name = joint_name_by_path.get(prim.joint_path)
        if joint_name is not None:
            drive_by_joint[joint_name] = (prim.gain, prim.force_limit)
    stiffness = np.asarray(
        [max(drive_by_joint.get(name, (0.0, 0.0))[0], 0.0) for name in joint_order],
        dtype=np.float32,
    )
    force_limit = np.asarray(
        [max(drive_by_joint.get(name, (0.0, 0.0))[1], 0.0) for name in joint_order],
        dtype=np.float32,
    )
    return stiffness, force_limit


def _temporary_fixed_base_model(source: Path, work: Path) -> Path:
    suffix = source.suffix.lower()
    if suffix in (".xml", ".mjcf"):
        return _fixed_base_model_xml(source, work)
    if suffix in (".usd", ".usda"):
        return _usda_to_fixed_base_mjcf(source, work)
    raise SystemExit(f"unsupported MJX oracle source format: {source}")


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


def _disable_unconstrained_dynamics_terms(mujoco, model) -> None:
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_LIMIT)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_EQUALITY)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_FRICTIONLOSS)
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0


def generate_random_samples(model_path: Path,
                            out_path: Path,
                            sample_count: int,
                            batch_size: int) -> None:
    jax, jnp, mujoco, mjx = _load_mjx_deps()
    with tempfile.TemporaryDirectory(prefix="nuka_mjx_fixed_base_") as tmp:
        fixed_model_path = _temporary_fixed_base_model(model_path, Path(tmp))
        model = mujoco.MjModel.from_xml_path(str(fixed_model_path))
        _disable_unconstrained_dynamics_terms(mujoco, model)
        mjx_model = mjx.put_model(model)
        rng = np.random.default_rng(seed=42)
        nuka_dofs = model.nv + 1
        batch_size = sample_count if batch_size <= 0 else batch_size

        def forward_one(qpos, qvel, tau):
            data = mjx.make_data(mjx_model).replace(
                qpos=qpos,
                qvel=qvel,
                qfrc_applied=tau,
            )
            return mjx.forward(mjx_model, data).qacc

        forward_batch = jax.jit(jax.vmap(forward_one))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(f".{out_path.name}.tmp")
        with tmp_path.open("wb") as out:
            _write_header(out,
                          model_path.name,
                          KIND_RANDOM_QACC,
                          sample_count,
                          nuka_dofs,
                          nuka_dofs,
                          nuka_dofs)
            for start in range(0, sample_count, batch_size):
                count = min(batch_size, sample_count - start)
                qpos = rng.uniform(-0.2, 0.2, (count, model.nq)).astype(np.float32)
                qvel = rng.uniform(-0.5, 0.5, (count, model.nv)).astype(np.float32)
                tau = rng.uniform(-2.0, 2.0, (count, model.nv)).astype(np.float32)
                qacc = np.asarray(
                    jax.device_get(
                        forward_batch(jnp.asarray(qpos),
                                      jnp.asarray(qvel),
                                      jnp.asarray(tau))),
                    dtype=np.float32)

                root = np.zeros((count, 1), dtype=np.float32)
                records = np.concatenate((
                    root,
                    qpos,
                    root,
                    qvel,
                    root,
                    tau,
                    root,
                    qacc,
                ), axis=1)
                out.write(records.astype("<f4", copy=False).tobytes())
        os.replace(tmp_path, out_path)


def generate_stand_trajectory(model_path: Path,
                              out_path: Path,
                              step_count: int,
                              dt: float) -> None:
    jax, jnp, mujoco, mjx = _load_mjx_deps()
    with tempfile.TemporaryDirectory(prefix="nuka_mjx_fixed_base_") as tmp:
        fixed_model_path = _temporary_fixed_base_model(model_path, Path(tmp))
        model = mujoco.MjModel.from_xml_path(str(fixed_model_path))
        model.opt.timestep = dt
        _disable_unconstrained_dynamics_terms(mujoco, model)
        joint_order = _usd_joint_order(model_path)
        if joint_order is not None:
            q_indices = np.array(
                [model.jnt_qposadr[model.joint(name).id] for name in joint_order],
                dtype=np.int32,
            )
        else:
            q_indices = np.arange(model.nq, dtype=np.int32)
        mjx_model = mjx.put_model(model)
        initial_positions = _usd_initial_positions(model_path, joint_order)
        if initial_positions is not None:
            q0 = jnp.asarray(initial_positions)
        else:
            q0 = jnp.asarray(model.qpos0.astype(np.float32))
        v0 = jnp.zeros(model.nv, dtype=jnp.float32)
        drive_targets = q0
        drive_arrays = _usd_drive_arrays(model_path, joint_order)
        if drive_arrays is not None:
            stiffness_array, force_limit_array = drive_arrays
            drive_stiffness = jnp.asarray(stiffness_array)
            drive_force_limit = jnp.asarray(force_limit_array)
        else:
            drive_stiffness = jnp.full((model.nv,), 50.0, dtype=jnp.float32)
            drive_force_limit = jnp.full((model.nv,), 24.0, dtype=jnp.float32)
        drive_damping = 2.0 * jnp.sqrt(jnp.maximum(drive_stiffness, 0.0))

        def step_once(current, _):
            # Implicit joint-damping integration that mirrors Nuka's single-env
            # C-ABI step (src/c_abi/world.cpp, defer_velocity_damping=true) and the
            # batched contacts path. The #43 fix moved Nuka from explicit -Kd*qvel
            # (semi-implicit Euler) to GENERAL IMPLICIT (backward-Euler) joint
            # damping, so the explicit golden no longer matched; this rebuilds the
            # MuJoCo/MJX oracle with the SAME scheme. The scheme:
            #
            #   (1) drive emits ONLY the Kp stiffness torque, force-limit-clamped
            #       (the -Kd*qvel damping is DEFERRED, applied implicitly below).
            #   (2) one mjx.forward with qfrc_applied=tau_stiff yields BOTH the
            #       stiffness-only acceleration qacc AND the dense joint-space mass
            #       matrix M (= qM incl. armature, MuJoCo dof order).
            #   (3) explicit half-step velocity qvel_half = qvel + qacc*dt.
            #   (4) implicit damping (THE #43 change): solve the backward-Euler
            #       system (M + dt*C) qvel_next = M*qvel_half, written in the
            #       equivalent residual form Nuka applies in place,
            #           qvel_next = qvel_half - dt*(M + dt*C)^-1 * (C*qvel_half),
            #       with C = diag(drive_damping). This is algebraically identical
            #       to (M+dt*C)^-1 M qvel_half and matches Nuka's constrained-
            #       velocity (implicit) joint damping bit-for-bit in scheme.
            #   (5) position integrate qpos_next = qpos + qvel_next*dt.
            #
            # ORDERING: drive_damping (C), qvel, qacc and M (mjx.full_m) are all in
            # MuJoCo dof order -- the same order the OLD explicit code used C
            # elementwise against qvel -- so the linear solve is internally
            # consistent. Only the qpos OUTPUT is remapped to USD joint order via
            # q_indices (after the scan, line ~591); that remap is unchanged.
            qpos, qvel = current
            # (1) stiffness-only torque, clamped (damping deferred -- matches Nuka's
            #     defer_velocity_damping=true clamp on the stiffness-only tau).
            raw_tau = drive_stiffness * (drive_targets - qpos)
            tau_stiff = jnp.where(
                drive_force_limit > 0.0,
                jnp.clip(raw_tau, -drive_force_limit, drive_force_limit),
                raw_tau,
            )
            # (2) single forward: qacc + dense mass matrix M (armature on diagonal).
            data = mjx.make_data(mjx_model).replace(
                qpos=qpos,
                qvel=qvel,
                qfrc_applied=tau_stiff,
            )
            data = mjx.forward(mjx_model, data)
            qacc = data.qacc
            M = mjx.full_m(mjx_model, data)
            # (3) explicit half-step velocity.
            qvel_half = qvel + qacc * dt
            # (4) implicit (backward-Euler) joint damping.
            C = drive_damping
            M_damped = M + dt * jnp.diag(C)
            qvel_next = qvel_half - dt * jnp.linalg.solve(M_damped, C * qvel_half)
            # (5) position integrate.
            qpos_next = qpos + qvel_next * dt
            return (qpos_next, qvel_next), qpos_next

        @jax.jit
        def rollout(initial_q, initial_v):
            _, qpos = jax.lax.scan(step_once,
                                   (initial_q, initial_v),
                                   None,
                                   length=step_count)
            return qpos

        qpos = np.asarray(jax.device_get(rollout(q0, v0)), dtype=np.float32)
        if joint_order is not None:
            qpos = qpos[:, q_indices]
        root = np.zeros((step_count, 1), dtype=np.float32)
        records = np.concatenate((root, qpos), axis=1)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(f".{out_path.name}.tmp")
        with tmp_path.open("wb") as out:
            _write_header(out,
                          model_path.name,
                          KIND_JOINT_TRAJECTORY,
                          step_count,
                          model.nq + 1,
                          0,
                          0)
            out.write(records.astype("<f4", copy=False).tobytes())
        os.replace(tmp_path, out_path)


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
    parser.add_argument(
        "--batch-size",
        default=0,
        type=int,
        help="random-qacc JIT batch size; <=0 uses one batch for all samples",
    )
    args = parser.parse_args()
    if args.mode == "random-qacc":
        generate_random_samples(args.model, args.out, args.samples, args.batch_size)
    else:
        generate_stand_trajectory(args.model, args.out, args.steps, args.dt)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
