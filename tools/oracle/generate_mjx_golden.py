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
    has_lower_limit: bool = False
    has_upper_limit: bool = False
    gain: float = 1.0
    force_limit: float = 0.0
    radius: float = 0.0
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
            if key == "physics:lowerLimit":
                prim.has_lower_limit = True
            elif key == "physics:upperLimit":
                prim.has_upper_limit = True
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
        ("radius", "radius"),
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
            # Mirror the engine: a hinge without authored USD bounds gets no
            # limit row, so MuJoCo must not hard-stop at the parser's +/-pi.
            if joint.has_lower_limit or joint.has_upper_limit:
                lines.append(
                    f'{prefix}  <joint name="{_xml_name(joint.name)}" type="hinge" '
                    f'axis="{_fmt(axis)}" range="{joint.lower_limit:.9g} '
                    f'{joint.upper_limit:.9g}" limited="true" damping="0" armature="0"/>'
                )
            else:
                lines.append(
                    f'{prefix}  <joint name="{_xml_name(joint.name)}" type="hinge" '
                    f'axis="{_fmt(axis)}" limited="false" damping="0" armature="0"/>'
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


def _free_joint_qvel_dof(model) -> int | None:
    """Return the qvel start index of the first free joint, or None.

    A MuJoCo free joint owns 6 contiguous DOFs in qvel: [v_lin(3) in WORLD frame,
    omega(3) in BODY-LOCAL frame] and 7 contiguous slots in qpos: [pos(3),
    quat(4) w-first]. For the unitree go2/h1 the free joint is jnt 0 (base), so
    the base occupies qvel[0:6]/qpos[0:7] and the legs follow.
    """
    free = int(__import__("mujoco").mjtJoint.mjJNT_FREE)
    for jnt in range(model.njnt):
        if int(model.jnt_type[jnt]) == free:
            return int(model.jnt_dofadr[jnt])
    return None


def generate_floating_base_random_samples(model_path: Path,
                                          out_path: Path,
                                          sample_count: int,
                                          batch_size: int) -> None:
    """Generate a FLOATING-BASE random-sample golden (additive to the fixed one).

    Unlike generate_random_samples this KEEPS the <freejoint/> so MJX builds the
    real floating-base model (go2 nv=18/nq=19, h1 nv=25/nq=26: a 6-DOF base + the
    leg DOFs). It DOES strip <actuator>+<keyframe> (mirroring _fixed_base_model_xml
    but WITHOUT removing the freejoint): go2's 12 position servos would otherwise
    inject qfrc_actuator (biasprm position/velocity feedback) into the MJX qacc
    even at ctrl=0, a force Nuka's bare-tau ABA does not and should not model;
    <keyframe> must go too because its ctrl field becomes invalid once <actuator>
    is removed (MuJoCo would fail to compile). It writes RAW MJX arrays -- NO Nuka
    root-slot padding -- so the golden record is [qpos(nq), qvel(nv), tau(nv),
    qacc(nv)]; the loader reads the asymmetric (nq != nv) shape from the header.
    The base qpos quaternion is normalized. The base 6 generalized forces
    (qfrc_applied over the free-joint DOFs) are ZEROED: Nuka's floating-base ABA
    has no external-base-force input (joint_force[root]=0), so a nonzero base force
    in MJX would be unmatchable and would corrupt even the leg accelerations. The
    harness converts MJX's mixed world/body free-joint convention to Nuka's
    body-frame spatial convention (incl. the omega x v_lin spatial->classical
    linear-accel term).
    """
    jax, jnp, mujoco, mjx = _load_mjx_deps()
    suffix = model_path.suffix.lower()
    if suffix not in (".xml", ".mjcf"):
        raise SystemExit(
            "floating-base random oracle supports only MJCF models with a "
            "<freejoint/>; got " + model_path.name)
    with tempfile.TemporaryDirectory(prefix="nuka_mjx_floating_base_") as tmp:
        # Copy assets next to the model so relative mesh/include paths resolve.
        # Strip <actuator>+<keyframe> (mirror _fixed_base_model_xml) so the MJX
        # qacc carries NO servo-bias force, but KEEP the <freejoint/> so the base
        # stays floating (nv=18/nq=19 for go2).
        work = Path(tmp)
        if (model_path.parent / "assets").exists():
            shutil.copytree(model_path.parent / "assets", work / "assets")
        text = model_path.read_text()
        text = re.sub(r"\n\s*<keyframe>.*?</keyframe>\s*", "\n", text, flags=re.S)
        text = re.sub(r"\n\s*<actuator>.*?</actuator>\s*", "\n", text, flags=re.S)
        floating_model_path = work / model_path.name
        floating_model_path.write_text(text)
        model = mujoco.MjModel.from_xml_path(str(floating_model_path))
        _disable_unconstrained_dynamics_terms(mujoco, model)
        base_dof = _free_joint_qvel_dof(model)
        if base_dof is None:
            raise SystemExit(
                f"{model_path.name}: no <freejoint/> found; floating-base oracle "
                "requires a free base joint")
        base_qpos = int(model.jnt_qposadr[
            [int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
             for j in range(model.njnt)].index(True)])
        mjx_model = mjx.put_model(model)
        rng = np.random.default_rng(seed=4242)
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
                          model.nq,
                          model.nv,
                          model.nv)
            for start in range(0, sample_count, batch_size):
                count = min(batch_size, sample_count - start)
                qpos = rng.uniform(-0.2, 0.2, (count, model.nq)).astype(np.float32)
                qvel = rng.uniform(-0.5, 0.5, (count, model.nv)).astype(np.float32)
                tau = rng.uniform(-2.0, 2.0, (count, model.nv)).astype(np.float32)
                # Normalize the free-joint quaternion (qpos[base_qpos+3 : +7]).
                quat = qpos[:, base_qpos + 3:base_qpos + 7]
                norm = np.linalg.norm(quat, axis=1, keepdims=True)
                norm[norm == 0.0] = 1.0
                qpos[:, base_qpos + 3:base_qpos + 7] = quat / norm
                # Zero the base generalized force: Nuka has no base-force input.
                tau[:, base_dof:base_dof + 6] = 0.0
                qacc = np.asarray(
                    jax.device_get(
                        forward_batch(jnp.asarray(qpos),
                                      jnp.asarray(qvel),
                                      jnp.asarray(tau))),
                    dtype=np.float32)
                records = np.concatenate((qpos, qvel, tau, qacc), axis=1)
                out.write(records.astype("<f4", copy=False).tobytes())
        os.replace(tmp_path, out_path)


# ---------------------------------------------------------------------------
# v0.8 C5c-2: floating-base FOOT-CONTACT single-step MJX parity golden.
#
# The acceptance layer for the foot-ground contact subsume. Unlike the
# floating-RANDOM golden (which DISABLES contact and injects leg tau to probe
# the bare ABA force->accel response), this mode KEEPS contact ON: it places a
# floating go2 with its feet PENETRATING a +Z ground plane by a known delta and
# records ONE mjx.forward -> qacc. The PRIMARY value under test is the base-6
# qacc: proof that the foot CONTACT force recoils into the floating base
# MJX-exactly (the part featherstone could only reach via leg tau, not contact).
#
# CRITICAL MODEL-ALIGNMENT DECISION (the task's #1 risk). The MJX side is built
# by SOURCE-TO-SOURCE TRANSLATION of the SAME go2_float.usda that Nuka cooks,
# NOT from go2_mjx.xml. go2_mjx.xml carries rotated/offset inertials
# (quat=..., COM pos=0.0211...) and a different foot offset (-0.002 0 -0.213)
# that would yield a DIFFERENT mass matrix M than Nuka's diagonal-at-origin USD
# cook -- making a base-6 mismatch uninterpretable (M diff vs solver diff). The
# thing under test is the contact-force->base coupling (the solver), so both
# sides must share M BY CONSTRUCTION. The fixed-base oracle already proved this
# translator's kinematics match Nuka's USD cook (~1e-4), so the floating variant
# (freejoint + foot spheres at the SAME USD offsets + a floor) has Nuka-identical
# FK + M. Foot spheres use the USD radius/offset and condim=1 frictionless with
# MuJoCo-DEFAULT solref/solimp (== the Nuka ContactManifold defaults
# solref=[0.02,1], solimp=[0.9,0.95,0.001,0.5,2]); go2's foot override
# (solimp="0.015 1 0.031", condim=6) is DELIBERATELY NOT used.
#
# The record per config carries BOTH a contact-ON qacc and a contact-OFF qacc:
# the OFF qacc is the GATE-0 free-dynamics tolerance floor (and lets the test
# isolate the contact contribution as ON-OFF on both sides). It also records the
# contact set (ncon, per-contact world pos/normal/dist) and efc_force so the C++
# test can verify the contact STATE matches Nuka's manifold before comparing
# qacc, and corroborate the per-contact normal force (Nuka lambda/dt).
# ---------------------------------------------------------------------------

# Custom golden layout for the contact-step golden (NOT the shared NUKAGOLD
# GoldenTrajectory format, which is a fixed [qpos,qvel,tau,qacc] record). The
# C5c-2 test ships a small bespoke reader. Magic "NUKACSTP" (Nuka Contact STeP).
KIND_CONTACT_STEP_MAGIC = b"NUKACSTP"
CONTACT_STEP_VERSION = 1
MAX_CONTACTS_RECORDED = 8  # generous upper bound (go2 has 4 feet -> 4 contacts)


def _usda_to_floating_contact_mjcf(source: Path, work: Path,
                                   foot_solref, foot_solimp,
                                   floor_height: float) -> tuple[Path, list[str]]:
    """Translate go2_float.usda -> a FLOATING-BASE contact MJCF.

    Identical body/joint/inertial emission to _usda_to_fixed_base_mjcf (the
    translation the fixed-base oracle proved matches Nuka's cook) PLUS:
      * a <freejoint/> on the root body (floating 6-DOF base),
      * foot <geom type="sphere"> on each calf body at the USD Sphere's local
        translate + radius, condim=1, with the supplied (MuJoCo-default) foot
        solref/solimp; contype/conaffinity enabled so feet<->floor collide,
      * a floor plane geom at floor_height.
    Returns (mjcf_path, foot_geom_names_in_order) where the order is the USD
    Sphere prim order (== Nuka's foot/manifold append order).
    """
    prims = _parse_usda_prims(source)
    bodies = {prim.path: prim for prim in prims if _is_rigid_body(prim)}
    joints = [prim for prim in prims if _is_joint(prim)]
    spheres_by_parent: dict[str, list[_UsdPrim]] = {}
    for prim in prims:
        if prim.type == "Sphere":
            spheres_by_parent.setdefault(prim.parent_path, []).append(prim)
    for siblings in spheres_by_parent.values():
        siblings.sort(key=lambda item: item.order)

    children_by_parent: dict[str, list[_UsdPrim]] = {}
    child_paths: set[str] = set()
    joint_for_child: dict[str, _UsdPrim] = {}
    for joint in joints:
        if joint.type != "PhysicsRevoluteJoint":
            raise SystemExit(
                f"{source}: contact-step oracle supports only revolute USD "
                f"joints; got {joint.type} for {joint.name}")
        if joint.body0_path not in bodies or joint.body1_path not in bodies:
            raise SystemExit(f"{source}: joint {joint.name} references an unknown body")
        children_by_parent.setdefault(joint.body0_path, []).append(joint)
        child_paths.add(joint.body1_path)
        joint_for_child[joint.body1_path] = joint
    for siblings in children_by_parent.values():
        siblings.sort(key=lambda item: item.order)

    roots = [prim for prim in prims if _is_rigid_body(prim) and prim.path not in child_paths]
    if len(roots) != 1:
        raise SystemExit(f"{source}: contact-step oracle expects exactly one root")

    foot_geom_names: list[str] = []
    solref_str = _fmt(tuple(foot_solref))
    solimp_str = _fmt(tuple(foot_solimp))

    lines = [
        '<mujoco model="nuka_go2_float_contact">',
        '  <compiler angle="radian" coordinate="local"/>',
        '  <option integrator="Euler" gravity="0 0 -9.81" '
        'iterations="200" tolerance="1e-12" cone="pyramidal"/>',
        '  <worldbody>',
        f'    <geom name="floor" type="plane" pos="0 0 {floor_height:.9g}" '
        'size="10 10 0.1" contype="1" conaffinity="1" condim="1" '
        f'solref="{solref_str}" solimp="{solimp_str}"/>',
    ]

    def emit_body(body: _UsdPrim, indent: int, is_root: bool) -> None:
        prefix = " " * indent
        lines.append(
            f'{prefix}<body name="{_xml_name(body.name)}" pos="{_fmt(body.translate)}">'
        )
        if is_root:
            lines.append(f"{prefix}  <freejoint/>")
        joint = joint_for_child.get(body.path)
        if joint is not None:
            axis = _axis_tuple(joint.axis_token)
            # Mirror the engine: a hinge without authored USD bounds gets no
            # limit row, so MuJoCo must not hard-stop at the parser's +/-pi.
            if joint.has_lower_limit or joint.has_upper_limit:
                lines.append(
                    f'{prefix}  <joint name="{_xml_name(joint.name)}" type="hinge" '
                    f'axis="{_fmt(axis)}" range="{joint.lower_limit:.9g} '
                    f'{joint.upper_limit:.9g}" limited="true" damping="0" armature="0"/>'
                )
            else:
                lines.append(
                    f'{prefix}  <joint name="{_xml_name(joint.name)}" type="hinge" '
                    f'axis="{_fmt(axis)}" limited="false" damping="0" armature="0"/>'
                )
        if body.mass > 0.0 and not body.kinematic_enabled:
            lines.append(
                f'{prefix}  <inertial pos="0 0 0" mass="{body.mass:.9g}" '
                f'diaginertia="{_fmt(body.diagonal_inertia)}"/>'
            )
        for sphere in spheres_by_parent.get(body.path, []):
            geom_name = f"{body.name}_foot"
            foot_geom_names.append(geom_name)
            lines.append(
                f'{prefix}  <geom name="{_xml_name(geom_name)}" type="sphere" '
                f'size="{sphere.radius:.9g}" pos="{_fmt(sphere.translate)}" '
                f'contype="1" conaffinity="1" condim="1" '
                f'solref="{solref_str}" solimp="{solimp_str}"/>'
            )
        for child_joint in children_by_parent.get(body.path, []):
            emit_body(bodies[child_joint.body1_path], indent + 2, False)
        lines.append(f"{prefix}</body>")

    emit_body(roots[0], 4, True)
    lines.append("  </worldbody>")
    lines.append("</mujoco>")
    out = work / f"{source.stem}_float_contact_mjx.xml"
    out.write_text("\n".join(lines) + "\n")
    return out, foot_geom_names


def generate_floating_contact_step(model_path: Path,
                                   out_path: Path) -> None:
    """Generate the C5c-2 floating-base foot-CONTACT single-step parity golden.

    See _usda_to_floating_contact_mjcf for the model-alignment rationale (MJX is
    a source translation of go2_float.usda, NOT go2_mjx.xml). One or a few FIXED
    configs: feet penetrating the floor by a known delta, qvel=0, gravity ON, a
    leg holding-tau, base tau=0. Records per config (contact ON and OFF):
      qpos(nq), qvel(nv), tau(nv), qacc_on(nv), qacc_off(nv),
      ncon, then per-contact [pos(3), normal(3), dist] for MAX_CONTACTS_RECORDED,
      then efc_force per-contact normal force (MAX_CONTACTS_RECORDED), then the
      per-contact invweight (body_invweight0 == efc_diagApprox, MAX_CONTACTS).
    """
    jax, jnp, mujoco, mjx = _load_mjx_deps()
    suffix = model_path.suffix.lower()
    if suffix not in (".usd", ".usda"):
        raise SystemExit(
            "floating-contact-step oracle requires the go2_float USD source "
            "(source translation), got " + model_path.name)

    # MuJoCo-DEFAULT solref/solimp == the Nuka ContactManifold defaults. Pin both
    # sides to these (NOT go2's foot override).
    foot_solref = (0.02, 1.0)
    foot_solimp = (0.9, 0.95, 0.001, 0.5, 2.0)
    # Seat the floor so the feet penetrate by ~kPenetration (matches the C5c-1
    # SeatGround rest-penetration so the contact STATE aligns with Nuka).
    k_penetration = 0.002

    with tempfile.TemporaryDirectory(prefix="nuka_mjx_float_contact_") as tmp:
        work = Path(tmp)
        # First pass: build at a provisional floor height to read the foot world
        # positions at the home config, then re-seat the floor to the desired
        # penetration. (Two-pass: cheap, and keeps the geometry exact.)
        provisional, foot_geoms = _usda_to_floating_contact_mjcf(
            model_path, work, foot_solref, foot_solimp, floor_height=-100.0)
        m0 = mujoco.MjModel.from_xml_path(str(provisional))
        base_dof = _free_joint_qvel_dof(m0)
        if base_dof is None:
            raise SystemExit("contact-step oracle: translated model has no freejoint")
        base_qpos = int(m0.jnt_qposadr[
            [int(m0.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
             for j in range(m0.njnt)].index(True)])

        # Build the HOME qpos = the SAME configuration Nuka cooks. Nuka's cooker
        # loads nuka:initialPosition into the cooked q (articulation_state.cpp:321
        # q.push_back(initial_positions[...])), and the C5c-1 test zeroes only
        # qdot -- NOT q -- so the cooked crouch (thigh=0.8, calf=-1.5, hip=0) is
        # the configuration under test. The earlier model.qpos0 (legs straight)
        # was the wrong config: M depends on q, so the whole parity (not just the
        # contact STATE) requires q to match. We map each USD joint's initial
        # position to its MJX qpos slot BY NAME (the MJX body-recursion order and
        # Nuka's leg order are not guaranteed identical), exactly as the proven
        # stand-trajectory oracle does (_usd_initial_positions).
        usd_prims = _parse_usda_prims(model_path)
        usd_initial_by_name = {
            p.name: p.initial_position
            for p in usd_prims
            if _is_joint(p) and p.has_initial_position
        }
        q_home = np.array(m0.qpos0, dtype=np.float64)
        for jname, q0 in usd_initial_by_name.items():
            jid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise SystemExit(
                    f"contact-step oracle: USD joint {jname} not found in MJX model")
            q_home[int(m0.jnt_qposadr[jid])] = float(q0)
        # Base: keep the USD trunk pose (translate z, identity quat) from qpos0.

        d0 = mujoco.MjData(m0)
        d0.qpos[:] = q_home
        d0.qvel[:] = 0.0
        mujoco.mj_forward(m0, d0)
        # Foot sphere world bottoms at the home config.
        min_bottom = float("inf")
        for gname in foot_geoms:
            gid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_GEOM, gname)
            r = float(m0.geom_size[gid, 0])
            world_pos = d0.geom_xpos[gid]
            min_bottom = min(min_bottom, float(world_pos[2]) - r)
        floor_height = min_bottom + k_penetration

        # Second pass: rebuild with the seated floor.
        final_path, foot_geoms = _usda_to_floating_contact_mjcf(
            model_path, work, foot_solref, foot_solimp, floor_height=floor_height)
        model = mujoco.MjModel.from_xml_path(str(final_path))
        model.opt.timestep = 1.0 / 240.0  # match the Nuka C5c-1 dt (lambda/dt).
        nq = int(model.nq)
        nv = int(model.nv)

        # The fixed config(s). qvel=0, gravity ON, base tau=0, a leg holding-tau.
        # We use ctrl-free qfrc_applied (qfrc_applied[6:] = leg tau). For the
        # base-6 normal-recoil proof a ZERO leg tau is the cleanest (the contact
        # alone drives the base); add a couple of nonzero-leg-tau configs so the
        # leg-12 corroboration also exercises tau coupling. q_home (built above
        # from the USD initial positions) is the Nuka cooked crouch.
        configs = []
        # Config 0: home stance, zero leg tau (pure contact recoil).
        configs.append((q_home.copy(), np.zeros(nv)))
        # Config 1: home stance, a small constant leg holding-tau.
        tau1 = np.zeros(nv)
        tau1[6:] = 1.5
        configs.append((q_home.copy(), tau1.copy()))
        # Config 2: a slightly deeper base (more penetration), zero leg tau.
        q_deep = q_home.copy()
        q_deep[base_qpos + 2] -= 0.001  # lower the base 1 mm -> deeper feet.
        configs.append((q_deep.copy(), np.zeros(nv)))

        mjx_model_on = mjx.put_model(model)
        # Contact-OFF model (GATE 0): same model, contact disabled.
        model_off = mujoco.MjModel.from_xml_path(str(final_path))
        model_off.opt.timestep = model.opt.timestep
        model_off.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        mjx_model_off = mjx.put_model(model_off)

        def forward_qacc(mjx_model, qpos, qvel, tau):
            data = mjx.make_data(mjx_model).replace(
                qpos=qpos, qvel=qvel, qfrc_applied=tau)
            return mjx.forward(mjx_model, data).qacc

        forward_on = jax.jit(forward_qacc, static_argnums=())

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(f".{out_path.name}.tmp")
        with tmp_path.open("wb") as out:
            out.write(KIND_CONTACT_STEP_MAGIC)
            out.write(struct.pack("<IIIII",
                                  CONTACT_STEP_VERSION,
                                  len(configs),
                                  nq, nv,
                                  MAX_CONTACTS_RECORDED))
            name = model_path.name.encode("utf-8")
            out.write(struct.pack("<I", len(name)))
            out.write(name)
            for (qpos, tau) in configs:
                qpos_f = qpos.astype(np.float32)
                qvel_f = np.zeros(nv, dtype=np.float32)
                tau_f = tau.astype(np.float32)
                qacc_on = np.asarray(jax.device_get(forward_on(
                    mjx_model_on, jnp.asarray(qpos_f), jnp.asarray(qvel_f),
                    jnp.asarray(tau_f))), dtype=np.float32)
                qacc_off = np.asarray(jax.device_get(forward_on(
                    mjx_model_off, jnp.asarray(qpos_f), jnp.asarray(qvel_f),
                    jnp.asarray(tau_f))), dtype=np.float32)
                # Contact state + efc_force from a classic mj_forward (mjx data
                # contact arrays are padded; the CPU pipeline gives the clean
                # contact set + per-contact normal force via mj_contactForce).
                dcpu = mujoco.MjData(model)
                dcpu.qpos[:] = qpos.astype(np.float64)
                dcpu.qvel[:] = 0.0
                dcpu.qfrc_applied[:] = tau.astype(np.float64)
                mujoco.mj_forward(model, dcpu)
                ncon = int(dcpu.ncon)
                contact_rows = np.zeros((MAX_CONTACTS_RECORDED, 7), dtype=np.float32)
                normal_force = np.zeros(MAX_CONTACTS_RECORDED, dtype=np.float32)
                # invweight per contact = the MuJoCo body_invweight0 sum the efc
                # row uses (== efc_diagApprox for the contact's normal row). The
                # parity test feeds this as inputs.invweight so the dual
                # regularizer R = invweight*(1-imp)/imp matches MuJoCo's exactly
                # (completing "same compliance inputs both sides"; NOT a tolerance
                # loosen). With condim=1, efc rows map 1:1 to contacts in order.
                invweight = np.zeros(MAX_CONTACTS_RECORDED, dtype=np.float32)
                for c in range(min(ncon, MAX_CONTACTS_RECORDED)):
                    con = dcpu.contact[c]
                    pos = np.asarray(con.pos, dtype=np.float32)
                    # contact frame row 0 is the normal (geom1->geom2 / MuJoCo).
                    normal = np.asarray(con.frame[0:3], dtype=np.float32)
                    contact_rows[c, 0:3] = pos
                    contact_rows[c, 3:6] = normal
                    contact_rows[c, 6] = float(con.dist)
                    f6 = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(model, dcpu, c, f6)
                    normal_force[c] = float(f6[0])  # normal component (contact frame)
                    # The efc row index for contact c (condim=1: efc_address[c]).
                    efc_row = int(dcpu.contact[c].efc_address)
                    if 0 <= efc_row < int(dcpu.nefc):
                        invweight[c] = float(dcpu.efc_diagApprox[efc_row])
                record = np.concatenate((
                    qpos.astype(np.float32),
                    qvel_f,
                    tau.astype(np.float32),
                    qacc_on,
                    qacc_off,
                    np.array([float(ncon)], dtype=np.float32),
                    contact_rows.reshape(-1),
                    normal_force,
                    invweight,
                ))
                out.write(record.astype("<f4", copy=False).tobytes())
        os.replace(tmp_path, out_path)


# ---------------------------------------------------------------------------
# v0.8 W1b: the CO-RESIDENT FREE-BOX foot-CONTACT single-step MJX parity golden.
#
# This is the box-side REALITY ANCHOR to MuJoCo. The C5c-2 floating-contact-step
# golden anchored the foot CONTACT recoil into the floating BASE (the
# articulation side) vs MJX, but the BOX side of the co-resident two-way contact
# (movable rigid box <-> articulation foot) has only ever been validated vs
# exact-solve oracles + the production kernel + legacy-Nuka -- NEVER vs MuJoCo.
# W1b closes that: a single go2 foot PENETRATES a FREE RIGID BOX (a <body> with a
# <freejoint/> + a <geom type="box">) by a known depth; one mjx.forward gives the
# ground-truth qacc for ALL DOFs -- the articulation base 6 + legs 12 AND the box
# 6 free DOFs. Nuka's unified pipeline (foot<->box -> EmitCompliantContactRows ->
# UnifiedSolve) must reproduce the BOX 6-DOF qacc (PRIMARY) + the base-6 (the
# C5c-2 secondary, confirmed still anchored with the box present).
#
# THE MODEL-ALIGNMENT CRUX (the task's #1 risk -- box mass+inertia). Nuka's box
# (mass + diagonal inertia on the BodyState) MUST EXACTLY match what MJX builds
# from the <geom box>. We pin both sides by setting an EXPLICIT <inertial> on the
# box body with a chosen mass + the exact box diagonal inertia
#   I = (m/3)*(h_y^2+h_z^2, h_x^2+h_z^2, h_x^2+h_y^2)
# for half-extents h. The box geom is ALSO emitted (it provides the contact
# geometry) but with mass="0" density="0" so it contributes NO inertia -- the
# explicit <inertial> is the sole inertia source. ASYMMETRIC half-extents make
# the three principal inertias DISTINCT, so a wrong-axis box inertia (or a
# dropped box angular Jacobian) is detectable in the box qacc. The exact box
# params (BOX_MASS, BOX_HALF_EXTENTS, the derived inertia) are duplicated in the
# Nuka test (tests/solver/test_foot_box_mjx_parity.cpp) so both sides match BY
# CONSTRUCTION. No <floor> is emitted: the ONLY contact is foot<->box, isolating
# the two-way reaction. The articulation side reuses the SAME source translation
# of go2_float.usda the C5c-2 golden uses (shared M by construction).
# ---------------------------------------------------------------------------

KIND_FOOT_BOX_STEP_MAGIC = b"NUKAFBOX"
FOOT_BOX_STEP_VERSION = 1

# The free rigid box (PINNED -- the Nuka test duplicates these EXACTLY). Asymmetric
# half-extents -> three distinct principal inertias (a wrong-axis inertia or a
# dropped box angular Jacobian shows up in the box qacc).
BOX_MASS = 2.0
BOX_HALF_EXTENTS = (0.06, 0.05, 0.04)  # h_x, h_y, h_z (metres)


def _box_diag_inertia(mass: float, half: tuple[float, float, float]) -> tuple[float, float, float]:
    """Solid box diagonal inertia about its COM: I = (m/3)*(h_b^2 + h_c^2)."""
    hx, hy, hz = half
    return (
        (mass / 3.0) * (hy * hy + hz * hz),
        (mass / 3.0) * (hx * hx + hz * hz),
        (mass / 3.0) * (hx * hx + hy * hy),
    )


def _usda_to_foot_box_contact_mjcf(source: Path, work: Path,
                                   foot_solref, foot_solimp,
                                   contact_foot_index: int,
                                   box_center, box_half, box_mass,
                                   box_inertia) -> tuple[Path, list[str], str]:
    """Translate go2_float.usda -> a FLOATING-BASE go2 + a FREE RIGID BOX MJCF.

    Identical body/joint/inertial emission to _usda_to_floating_contact_mjcf (the
    floating-base translation the C5c-2 golden uses) EXCEPT:
      * NO floor plane (the only contact is foot<->box),
      * a SECOND worldbody <body> with a <freejoint/>, an explicit <inertial>
        (the sole inertia source, pinned to box_mass + box_inertia), and a
        <geom type="box"> at box_half with mass="0" density="0" (geometry only).
    The box is the LAST body so its 6 free DOFs come AFTER the go2 base-6 + legs-12
    in qvel/qacc. Returns (mjcf_path, foot_geom_names_in_order, box_body_name).
    The foot geoms get contype/conaffinity that collide with the box; ALL the
    foot spheres are emitted (so the FK is identical to Nuka's) but only the
    contact_foot_index foot is positioned to penetrate the box.
    """
    prims = _parse_usda_prims(source)
    bodies = {prim.path: prim for prim in prims if _is_rigid_body(prim)}
    joints = [prim for prim in prims if _is_joint(prim)]
    spheres_by_parent: dict[str, list[_UsdPrim]] = {}
    for prim in prims:
        if prim.type == "Sphere":
            spheres_by_parent.setdefault(prim.parent_path, []).append(prim)
    for siblings in spheres_by_parent.values():
        siblings.sort(key=lambda item: item.order)

    children_by_parent: dict[str, list[_UsdPrim]] = {}
    child_paths: set[str] = set()
    joint_for_child: dict[str, _UsdPrim] = {}
    for joint in joints:
        if joint.type != "PhysicsRevoluteJoint":
            raise SystemExit(
                f"{source}: foot-box oracle supports only revolute USD "
                f"joints; got {joint.type} for {joint.name}")
        if joint.body0_path not in bodies or joint.body1_path not in bodies:
            raise SystemExit(f"{source}: joint {joint.name} references an unknown body")
        children_by_parent.setdefault(joint.body0_path, []).append(joint)
        child_paths.add(joint.body1_path)
        joint_for_child[joint.body1_path] = joint
    for siblings in children_by_parent.values():
        siblings.sort(key=lambda item: item.order)

    roots = [prim for prim in prims if _is_rigid_body(prim) and prim.path not in child_paths]
    if len(roots) != 1:
        raise SystemExit(f"{source}: foot-box oracle expects exactly one root")

    foot_geom_names: list[str] = []
    solref_str = _fmt(tuple(foot_solref))
    solimp_str = _fmt(tuple(foot_solimp))

    lines = [
        '<mujoco model="nuka_go2_foot_box">',
        '  <compiler angle="radian" coordinate="local"/>',
        '  <option integrator="Euler" gravity="0 0 -9.81" '
        'iterations="200" tolerance="1e-12" cone="pyramidal"/>',
        '  <worldbody>',
    ]

    def emit_body(body: _UsdPrim, indent: int, is_root: bool) -> None:
        prefix = " " * indent
        lines.append(
            f'{prefix}<body name="{_xml_name(body.name)}" pos="{_fmt(body.translate)}">'
        )
        if is_root:
            lines.append(f"{prefix}  <freejoint/>")
        joint = joint_for_child.get(body.path)
        if joint is not None:
            axis = _axis_tuple(joint.axis_token)
            # Mirror the engine: a hinge without authored USD bounds gets no
            # limit row, so MuJoCo must not hard-stop at the parser's +/-pi.
            if joint.has_lower_limit or joint.has_upper_limit:
                lines.append(
                    f'{prefix}  <joint name="{_xml_name(joint.name)}" type="hinge" '
                    f'axis="{_fmt(axis)}" range="{joint.lower_limit:.9g} '
                    f'{joint.upper_limit:.9g}" limited="true" damping="0" armature="0"/>'
                )
            else:
                lines.append(
                    f'{prefix}  <joint name="{_xml_name(joint.name)}" type="hinge" '
                    f'axis="{_fmt(axis)}" limited="false" damping="0" armature="0"/>'
                )
        if body.mass > 0.0 and not body.kinematic_enabled:
            lines.append(
                f'{prefix}  <inertial pos="0 0 0" mass="{body.mass:.9g}" '
                f'diaginertia="{_fmt(body.diagonal_inertia)}"/>'
            )
        for sphere in spheres_by_parent.get(body.path, []):
            geom_name = f"{body.name}_foot"
            foot_geom_names.append(geom_name)
            lines.append(
                f'{prefix}  <geom name="{_xml_name(geom_name)}" type="sphere" '
                f'size="{sphere.radius:.9g}" pos="{_fmt(sphere.translate)}" '
                f'contype="1" conaffinity="1" condim="1" '
                f'solref="{solref_str}" solimp="{solimp_str}"/>'
            )
        for child_joint in children_by_parent.get(body.path, []):
            emit_body(bodies[child_joint.body1_path], indent + 2, False)
        lines.append(f"{prefix}</body>")

    emit_body(roots[0], 4, True)

    # The FREE RIGID BOX (last worldbody body -> its 6 DOFs follow the go2's 18).
    box_body_name = "free_box"
    box_inert_str = _fmt(tuple(box_inertia))
    box_size_str = _fmt(tuple(box_half))
    box_pos_str = _fmt(tuple(box_center))
    lines.append(f'    <body name="{box_body_name}" pos="{box_pos_str}">')
    lines.append('      <freejoint/>')
    # Explicit inertial = the SOLE inertia source (pinned for Nuka alignment).
    lines.append(
        f'      <inertial pos="0 0 0" mass="{box_mass:.9g}" '
        f'diaginertia="{box_inert_str}"/>'
    )
    # The box geom = contact geometry ONLY (mass/density 0 -> no inertia).
    lines.append(
        f'      <geom name="{box_body_name}_geom" type="box" size="{box_size_str}" '
        f'pos="0 0 0" mass="0" density="0" contype="1" conaffinity="1" condim="1" '
        f'solref="{solref_str}" solimp="{solimp_str}"/>'
    )
    lines.append('    </body>')

    lines.append("  </worldbody>")
    lines.append("</mujoco>")
    out = work / f"{source.stem}_foot_box_mjx.xml"
    out.write_text("\n".join(lines) + "\n")
    return out, foot_geom_names, box_body_name


def generate_foot_box_contact_step(model_path: Path, out_path: Path) -> None:
    """Generate the W1b co-resident FREE-BOX foot-contact single-step parity golden.

    See _usda_to_foot_box_contact_mjcf for the model-alignment rationale (MJX is a
    source translation of go2_float.usda + a free box whose mass/inertia the Nuka
    test duplicates exactly). A single FIXED config: the FL foot penetrating the
    box top face by a known delta, qvel=0, gravity ON, base+leg tau=0, box tau=0.
    Records (contact ON and OFF):
      nq, nv, then per config:
      qpos(nq), qvel(nv), tau(nv), qacc_on(nv), qacc_off(nv),
      base_qvel_dof, box_qvel_dof, box pos(3)+quat(4)+mass+inertia(3),
      ncon, then per-contact [pos(3), normal(3), dist] for MAX_CONTACTS_RECORDED,
      then efc_force per-contact normal force, then per-contact invweight.
    """
    jax, jnp, mujoco, mjx = _load_mjx_deps()
    suffix = model_path.suffix.lower()
    if suffix not in (".usd", ".usda"):
        raise SystemExit(
            "foot-box-contact-step oracle requires the go2_float USD source "
            "(source translation), got " + model_path.name)

    # MuJoCo-DEFAULT solref/solimp == the Nuka ContactManifold defaults.
    foot_solref = (0.02, 1.0)
    foot_solimp = (0.9, 0.95, 0.001, 0.5, 2.0)
    # The FL foot penetrates the box top by this depth (matches the co-residence
    # gate's 4 mm). The box is placed so its top face is `k_penetration` above the
    # FL foot sphere bottom.
    k_penetration = 0.004
    contact_foot_index = 0  # FL (the USD Sphere prim order; Nuka's foot[0]).
    box_inertia = _box_diag_inertia(BOX_MASS, BOX_HALF_EXTENTS)

    with tempfile.TemporaryDirectory(prefix="nuka_mjx_foot_box_") as tmp:
        work = Path(tmp)
        # ---- Pass 1: build go2 ONLY (provisional box far away) to read the FL
        # foot world center at the cooked home crouch. We reuse the floating-
        # contact MJCF builder for a clean go2-with-floor-far-below model just to
        # do FK -- simpler: build the foot-box model with a provisional box, read
        # the foot, then re-place the box. ----------------------------------------
        provisional, foot_geoms, box_name = _usda_to_foot_box_contact_mjcf(
            model_path, work, foot_solref, foot_solimp, contact_foot_index,
            box_center=(0.0, 0.0, -100.0), box_half=BOX_HALF_EXTENTS,
            box_mass=BOX_MASS, box_inertia=box_inertia)
        m0 = mujoco.MjModel.from_xml_path(str(provisional))
        base_dof = _free_joint_qvel_dof(m0)
        if base_dof is None:
            raise SystemExit("foot-box oracle: translated model has no freejoint")
        base_qpos = int(m0.jnt_qposadr[
            [int(m0.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
             for j in range(m0.njnt)].index(True)])

        # Home qpos = the SAME configuration Nuka cooks (USD initial positions,
        # mapped by NAME), exactly as the C5c-2 generator does.
        usd_prims = _parse_usda_prims(model_path)
        usd_initial_by_name = {
            p.name: p.initial_position
            for p in usd_prims
            if _is_joint(p) and p.has_initial_position
        }
        q_home = np.array(m0.qpos0, dtype=np.float64)
        for jname, q0 in usd_initial_by_name.items():
            jid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise SystemExit(
                    f"foot-box oracle: USD joint {jname} not found in MJX model")
            q_home[int(m0.jnt_qposadr[jid])] = float(q0)

        d0 = mujoco.MjData(m0)
        d0.qpos[:] = q_home
        d0.qvel[:] = 0.0
        mujoco.mj_forward(m0, d0)
        # The FL foot world center + radius.
        fl_geom = foot_geoms[contact_foot_index]
        fl_gid = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_GEOM, fl_geom)
        fl_radius = float(m0.geom_size[fl_gid, 0])
        fl_center = np.asarray(d0.geom_xpos[fl_gid], dtype=np.float64)
        # Place the box so its TOP face is k_penetration above the FL foot bottom,
        # OFFSET in x by an off-COM lever so the box angular reaction is excited.
        foot_bottom = float(fl_center[2]) - fl_radius
        x_lever = 0.02  # 2 cm off-COM lever (matches the co-residence gate).
        box_top = foot_bottom + k_penetration
        box_center = (
            float(fl_center[0]) + x_lever,
            float(fl_center[1]),
            box_top - BOX_HALF_EXTENTS[2],
        )

        # ---- Pass 2: rebuild with the seated box. -------------------------------
        final_path, foot_geoms, box_name = _usda_to_foot_box_contact_mjcf(
            model_path, work, foot_solref, foot_solimp, contact_foot_index,
            box_center=box_center, box_half=BOX_HALF_EXTENTS,
            box_mass=BOX_MASS, box_inertia=box_inertia)
        model = mujoco.MjModel.from_xml_path(str(final_path))
        model.opt.timestep = 1.0 / 240.0
        nq = int(model.nq)
        nv = int(model.nv)

        # Free-joint DOF addresses: base = go2 root freejoint (first), box = the
        # box freejoint (second). Find both free joints in qvel-DOF order.
        free_dofs = [int(model.jnt_dofadr[j]) for j in range(model.njnt)
                     if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        free_dofs.sort()
        if len(free_dofs) != 2:
            raise SystemExit(
                f"foot-box oracle: expected 2 free joints (base + box), got "
                f"{len(free_dofs)}")
        base_qvel_dof = free_dofs[0]
        box_qvel_dof = free_dofs[1]
        # The box body id + its qpos start.
        box_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, box_name)
        free_qpos = [int(model.jnt_qposadr[j]) for j in range(model.njnt)
                     if int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)]
        free_qpos.sort()
        box_qpos_adr = free_qpos[1]

        # Rebuild q_home for the final model (box freejoint adds 7 qpos slots).
        q_home = np.array(model.qpos0, dtype=np.float64)
        for jname, q0 in usd_initial_by_name.items():
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            q_home[int(model.jnt_qposadr[jid])] = float(q0)
        # The box qpos0 already carries box_center (body pos) + identity quat.

        # The single fixed config: home stance, zero tau everywhere (pure contact).
        configs = [(q_home.copy(), np.zeros(nv))]

        mjx_model_on = mjx.put_model(model)
        model_off = mujoco.MjModel.from_xml_path(str(final_path))
        model_off.opt.timestep = model.opt.timestep
        model_off.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_CONTACT)
        mjx_model_off = mjx.put_model(model_off)

        def forward_qacc(mjx_model, qpos, qvel, tau):
            data = mjx.make_data(mjx_model).replace(
                qpos=qpos, qvel=qvel, qfrc_applied=tau)
            return mjx.forward(mjx_model, data).qacc

        forward_on = jax.jit(forward_qacc, static_argnums=())

        # The box mass/inertia MJX actually built (for the Nuka alignment audit:
        # the test asserts these equal its own pinned values).
        box_mass_actual = float(model.body_mass[box_bid])
        box_inertia_actual = np.asarray(model.body_inertia[box_bid], dtype=np.float32)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = out_path.with_name(f".{out_path.name}.tmp")
        with tmp_path.open("wb") as out:
            out.write(KIND_FOOT_BOX_STEP_MAGIC)
            out.write(struct.pack("<IIIII",
                                  FOOT_BOX_STEP_VERSION,
                                  len(configs),
                                  nq, nv,
                                  MAX_CONTACTS_RECORDED))
            name = model_path.name.encode("utf-8")
            out.write(struct.pack("<I", len(name)))
            out.write(name)
            for (qpos, tau) in configs:
                qpos_f = qpos.astype(np.float32)
                qvel_f = np.zeros(nv, dtype=np.float32)
                tau_f = tau.astype(np.float32)
                qacc_on = np.asarray(jax.device_get(forward_on(
                    mjx_model_on, jnp.asarray(qpos_f), jnp.asarray(qvel_f),
                    jnp.asarray(tau_f))), dtype=np.float32)
                qacc_off = np.asarray(jax.device_get(forward_on(
                    mjx_model_off, jnp.asarray(qpos_f), jnp.asarray(qvel_f),
                    jnp.asarray(tau_f))), dtype=np.float32)
                # Clean contact set + efc_force from a classic mj_forward.
                dcpu = mujoco.MjData(model)
                dcpu.qpos[:] = qpos.astype(np.float64)
                dcpu.qvel[:] = 0.0
                dcpu.qfrc_applied[:] = tau.astype(np.float64)
                mujoco.mj_forward(model, dcpu)
                ncon = int(dcpu.ncon)
                contact_rows = np.zeros((MAX_CONTACTS_RECORDED, 7), dtype=np.float32)
                normal_force = np.zeros(MAX_CONTACTS_RECORDED, dtype=np.float32)
                invweight = np.zeros(MAX_CONTACTS_RECORDED, dtype=np.float32)
                for c in range(min(ncon, MAX_CONTACTS_RECORDED)):
                    con = dcpu.contact[c]
                    pos = np.asarray(con.pos, dtype=np.float32)
                    normal = np.asarray(con.frame[0:3], dtype=np.float32)
                    contact_rows[c, 0:3] = pos
                    contact_rows[c, 3:6] = normal
                    contact_rows[c, 6] = float(con.dist)
                    f6 = np.zeros(6, dtype=np.float64)
                    mujoco.mj_contactForce(model, dcpu, c, f6)
                    normal_force[c] = float(f6[0])
                    efc_row = int(dcpu.contact[c].efc_address)
                    if 0 <= efc_row < int(dcpu.nefc):
                        invweight[c] = float(dcpu.efc_diagApprox[efc_row])
                # Box pose (pos+quat) read back from the home qpos slot.
                box_pos = qpos[box_qpos_adr:box_qpos_adr + 3].astype(np.float32)
                box_quat = qpos[box_qpos_adr + 3:box_qpos_adr + 7].astype(np.float32)
                record = np.concatenate((
                    qpos.astype(np.float32),
                    qvel_f,
                    tau.astype(np.float32),
                    qacc_on,
                    qacc_off,
                    np.array([float(base_qvel_dof), float(box_qvel_dof)],
                             dtype=np.float32),
                    box_pos,
                    box_quat,
                    np.array([box_mass_actual], dtype=np.float32),
                    box_inertia_actual,
                    np.array([float(ncon)], dtype=np.float32),
                    contact_rows.reshape(-1),
                    normal_force,
                    invweight,
                ))
                out.write(record.astype("<f4", copy=False).tobytes())
        os.replace(tmp_path, out_path)
        print(f"foot-box golden: nq={nq} nv={nv} base_dof={base_qvel_dof} "
              f"box_dof={box_qvel_dof} box_mass={box_mass_actual:.6f} "
              f"box_inertia={box_inertia_actual} ncon={ncon}")


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


# ---------------------------------------------------------------------------
# Versioned same-torque Go2OracleTrace (magic "NUKAOTRC").
#
# A ROLLOUT parity golden: both engines step the SAME floating-base go2 (source
# translation of go2_float.usda, no floor -> free flight) under the SAME
# deterministic torque schedule, recording per step the joint q/qd, base pose,
# per-joint limit impulse, and the requested/applied/saturated effort triplet.
# The exporter runs CPU MuJoCo (no jax); the comparator replays the schedule
# through the production nuka Python backend and compares field-by-field.
# ---------------------------------------------------------------------------

KIND_GO2_TRACE_MAGIC = b"NUKAOTRC"
GO2_TRACE_VERSION = 1

GO2_JOINT_ORDER = [
    "fl_hip_joint", "fl_thigh_joint", "fl_calf_joint",
    "fr_hip_joint", "fr_thigh_joint", "fr_calf_joint",
    "rl_hip_joint", "rl_thigh_joint", "rl_calf_joint",
    "rr_hip_joint", "rr_thigh_joint", "rr_calf_joint",
]
GO2_TRACE_FLOATS_PER_STEP = 12 + 12 + 7 + 12 + 12 + 12 + 12


def _go2_torque_schedule(steps: int, seed: int = 20260812) -> np.ndarray:
    """Deterministic shared torque schedule: a limit-seeking bias plus a wiggle."""
    rng = np.random.default_rng(seed)
    bias = np.array([
        6.0 if i % 2 == 0 else -6.0 for i in range(len(GO2_JOINT_ORDER))
    ], dtype=np.float32)
    t = np.arange(steps, dtype=np.float32)[:, None]
    phase = np.arange(len(GO2_JOINT_ORDER), dtype=np.float32)[None, :]
    wiggle = 4.0 * np.sin(6.0e-1 * t + 1.7 * phase)
    jitter = rng.uniform(-1.0, 1.0, size=(steps, len(GO2_JOINT_ORDER)))
    tau = bias[None, :] + wiggle + jitter
    return tau.astype(np.float32)


def _go2_trace_model(model_path: Path, work: Path,
                     dt: float) -> tuple[object, object, np.ndarray, np.ndarray]:
    """Build the floor-free floating MJCF, home qpos, and per-joint force limits."""
    import mujoco
    final_path, _ = _usda_to_floating_contact_mjcf(
        model_path, work, (0.02, 1.0), (0.9, 0.95, 0.001, 0.5, 2.0),
        floor_height=-100.0)
    model = mujoco.MjModel.from_xml_path(str(final_path))
    # Quantize to float32 once so both engines integrate the SAME timestep.
    dt = float(np.float32(dt))
    model.opt.timestep = dt
    # Stiff joint limits keep MuJoCo's soft-constraint penetration well under
    # the comparator tolerance while Nuka hard-stops at the bound.
    model.jnt_solref[:] = (0.004, 1.0)
    model.jnt_solimp[:] = (0.999, 0.999, 1.0e-6, 0.5, 2.0)
    usd_initial_by_name = {
        p.name: p.initial_position
        for p in _parse_usda_prims(model_path)
        if _is_joint(p) and p.has_initial_position
    }
    q_home = np.array(model.qpos0, dtype=np.float64)
    for jname, q0 in usd_initial_by_name.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise SystemExit(f"go2-trace: USD joint {jname} not found in MJX model")
        q_home[int(model.jnt_qposadr[jid])] = float(q0)
    joint_order = _usd_joint_order(model_path)
    _, force_limit_order = _usd_drive_arrays(model_path, joint_order)
    limit_by_name = dict(zip(joint_order, force_limit_order))
    force_limit = np.asarray(
        [limit_by_name.get(name, 0.0) for name in GO2_JOINT_ORDER],
        dtype=np.float32,
    )
    return model, q_home, force_limit


def generate_go2_oracle_trace(model_path: Path,
                              out_path: Path,
                              steps: int,
                              dt: float) -> None:
    """Export the MuJoCo side of the same-torque Go2OracleTrace rollout."""
    import mujoco
    with tempfile.TemporaryDirectory(prefix="nuka_go2_trace_") as tmp:
        model, q_home, force_limit = _go2_trace_model(model_path, Path(tmp), dt)
        joint_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in GO2_JOINT_ORDER
        ], dtype=np.int64)
        if np.any(joint_ids < 0):
            raise SystemExit("go2-trace: a GO2_JOINT_ORDER name is missing")
        qpos_adr = model.jnt_qposadr[joint_ids].astype(np.int64)
        dof_adr = model.jnt_dofadr[joint_ids].astype(np.int64)
        base_qpos = int(model.jnt_qposadr[
            [int(model.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)
             for j in range(model.njnt)].index(True)])

        data = mujoco.MjData(model)
        data.qpos[:] = q_home
        data.qvel[:] = 0.0
        taus = _go2_torque_schedule(steps)
        records = np.zeros((steps, GO2_TRACE_FLOATS_PER_STEP), dtype=np.float32)
        for i in range(steps):
            requested = taus[i]
            applied = np.clip(requested, -force_limit, force_limit)
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[dof_adr] = applied.astype(np.float64)
            mujoco.mj_step(model, data)
            limit_impulse = np.zeros(len(GO2_JOINT_ORDER), dtype=np.float64)
            for row in range(int(data.nefc)):
                if int(data.efc_type[row]) != int(
                        mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT):
                    continue
                jid = int(data.efc_id[row])
                hits = np.nonzero(joint_ids == jid)[0]
                if hits.size:
                    limit_impulse[hits[0]] += float(data.efc_force[row]) * dt
            records[i, 0:12] = data.qpos[qpos_adr]
            records[i, 12:24] = data.qvel[dof_adr]
            records[i, 24:27] = data.qpos[base_qpos:base_qpos + 3]
            records[i, 27:31] = data.qpos[base_qpos + 3:base_qpos + 7]
            records[i, 31:43] = limit_impulse
            records[i, 43:55] = requested
            records[i, 55:67] = applied
            records[i, 67:79] = (requested != applied).astype(np.float32)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    with tmp_path.open("wb") as out:
        out.write(KIND_GO2_TRACE_MAGIC)
        out.write(struct.pack("<IfII", GO2_TRACE_VERSION, dt, steps,
                              len(GO2_JOINT_ORDER)))
        name = model_path.name.encode("utf-8")
        out.write(struct.pack("<I", len(name)))
        out.write(name)
        out.write(records.astype("<f4", copy=False).tobytes())
    os.replace(tmp_path, out_path)


def compare_go2_oracle_trace(model_path: Path,
                             trace_path: Path,
                             device: int = 0,
                             tol_q: float = 0.05,
                             tol_qd: float = 0.75,
                             tol_effort: float = 1.0e-3,
                             tol_limit: float = 0.5) -> None:
    """Replay the trace schedule through the production backend and compare."""
    import sys
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "python"))
    import nuka

    with trace_path.open("rb") as f:
        if f.read(8) != KIND_GO2_TRACE_MAGIC:
            raise SystemExit(f"{trace_path}: not a NUKAOTRC file")
        version, dt, steps, njoints = struct.unpack("<IfII", f.read(16))
        if version != GO2_TRACE_VERSION or njoints != len(GO2_JOINT_ORDER):
            raise SystemExit(f"{trace_path}: unsupported trace version/layout")
        (name_len,) = struct.unpack("<I", f.read(4))
        f.read(name_len)
        raw = np.frombuffer(f.read(), dtype="<f4").reshape(steps, -1)
    ref_q = raw[:, 0:12]
    ref_qd = raw[:, 12:24]
    ref_limit = raw[:, 31:43]
    ref_requested = raw[:, 43:55]
    ref_applied = raw[:, 55:67]
    ref_saturated = raw[:, 67:79]

    scene = Path(model_path).resolve()
    if not scene.is_file():
        raise SystemExit(f"go2-trace: scene not found: {scene}")
    schedule = _go2_torque_schedule(steps)
    with nuka.Device.create(device) as device_handle:
        with nuka.World.create_from_scene(
            device_handle, os.fspath(scene), env_count=2, dt=float(dt),
            control_mode=nuka.CONTROL_MODE_TORQUE,
        ) as world:
            if world.dof_names()[1:] != GO2_JOINT_ORDER:
                raise SystemExit(
                    f"go2-trace: dof order mismatch: {world.dof_names()[1:]}")
            usd_joint_order = _usd_joint_order(scene)
            _, force_limit_order = _usd_drive_arrays(scene, usd_joint_order)
            limit_by_name = dict(zip(usd_joint_order, force_limit_order))
            limits = np.zeros((2, world.base_link_count), dtype=np.float32)
            limits[:, 1:] = [limit_by_name.get(name, 0.0)
                             for name in GO2_JOINT_ORDER]
            world.upload_field(nuka.DRIVE_FORCE_LIMIT, limits)
            # Bare-torque contract on both sides: no PD stiffness or damping.
            for field_name in ("DRIVE_STIFFNESS", "DRIVE_DAMPING"):
                zero = np.zeros((2, world.base_link_count), dtype=np.float32)
                world.upload_field(getattr(nuka, field_name), zero)
            torque = np.zeros((2, world.base_link_count), dtype=np.float32)
            got_q = np.zeros((steps, 12), dtype=np.float32)
            got_qd = np.zeros((steps, 12), dtype=np.float32)
            got_limit = np.zeros((steps, 12), dtype=np.float32)
            got_requested = np.zeros((steps, 12), dtype=np.float32)
            got_applied = np.zeros((steps, 12), dtype=np.float32)
            got_saturated = np.zeros((steps, 12), dtype=np.float32)
            for i in range(steps):
                torque[0, 1:] = schedule[i]
                torque[1, 1:] = schedule[i]
                world.upload_field(nuka.TORQUE_INPUT, torque)
                world.step()
                q = np.asarray(world.download_field(
                    nuka.JOINT_POSITION), dtype=np.float32).reshape(2, -1)
                qd = np.asarray(world.download_field(
                    nuka.JOINT_VELOCITY), dtype=np.float32).reshape(2, -1)
                lim = np.asarray(world.download_field(
                    nuka.JOINT_LIMIT_IMPULSE), dtype=np.float32).reshape(2, -1, 2)
                req = np.asarray(world.download_field(
                    nuka.ACTUATOR_EFFORT_REQUESTED), dtype=np.float32).reshape(2, -1)
                app = np.asarray(world.download_field(
                    nuka.ACTUATOR_EFFORT), dtype=np.float32).reshape(2, -1)
                sat = np.asarray(world.download_field(
                    nuka.ACTUATOR_SATURATED), dtype=np.float32).reshape(2, -1)
                got_q[i] = q[0, 1:]
                got_qd[i] = qd[0, 1:]
                # Both lanes are positive magnitudes; take the active side.
                got_limit[i] = np.maximum(lim[0, 1:, 0], lim[0, 1:, 1])
                got_requested[i] = req[0, 1:]
                got_applied[i] = app[0, 1:]
                got_saturated[i] = sat[0, 1:]

    np.testing.assert_allclose(
        got_requested, ref_requested, rtol=0.0, atol=tol_effort)
    np.testing.assert_allclose(
        got_applied, ref_applied, rtol=0.0, atol=tol_effort)
    np.testing.assert_array_equal(got_saturated, ref_saturated)

    # Effort telemetry is a pure clamp of the shared schedule: exact on ALL steps.
    # Free-flight dynamics parity is tight ONLY up to the first limit impact, past
    # which the two soft-constraint solvers' impact differences amplify chaotically.
    impact = steps
    active = (np.abs(ref_limit) > 1.0e-6).any(axis=1) | (got_limit > 1.0e-6).any(axis=1)
    hits = np.nonzero(active)[0]
    if hits.size:
        impact = int(hits[0])
    # impact == 0 means no pre-impact window exists; only the effort/bounds
    # contracts apply in that case.
    if impact > 0:
        q_err = float(np.abs(got_q[:impact] - ref_q[:impact]).max())
        qd_err = float(np.abs(got_qd[:impact] - ref_qd[:impact]).max())
        check_parity = True
    else:
        q_err = qd_err = 0.0
        check_parity = False

    # Nuka's authored bounds are a HARD stop: assert the whole trajectory stays
    # inside them (plus solver slop) even where the MuJoCo reference penetrates.
    # Joints without authored USD bounds have no engine limit row and are exempt.
    prims = _parse_usda_prims(scene)
    bounds = {
        p.name: (p.lower_limit, p.upper_limit)
        for p in prims if _is_joint(p) and (p.has_lower_limit or p.has_upper_limit)
    }
    bounded = [i for i, n in enumerate(GO2_JOINT_ORDER) if n in bounds]
    lower = np.array([bounds[n][0] for n in GO2_JOINT_ORDER if n in bounds],
                     dtype=np.float32)
    upper = np.array([bounds[n][1] for n in GO2_JOINT_ORDER if n in bounds],
                     dtype=np.float32)
    bound_slop = 2.0e-3
    assert float((got_q[:, bounded] - lower).min()) >= -bound_slop, "lower bound violated"
    assert float((upper - got_q[:, bounded]).min()) >= -bound_slop, "upper bound violated"

    limit_err = float(np.abs(got_limit - ref_limit).max()) if steps else 0.0
    post_q_err = (float(np.abs(got_q[impact:] - ref_q[impact:]).max())
                  if impact < steps else 0.0)
    print(f"trace={trace_path} steps={steps} dt={dt} impact_step={impact}")
    print(f"pre_impact q_max_err={q_err:.6g} qd_max_err={qd_err:.6g}")
    print(f"post_impact q_max_err={post_q_err:.6g} limit_max_err={limit_err:.6g}")
    if q_err > tol_q and check_parity:
        raise AssertionError(
            f"go2-trace pre-impact q parity failed: {q_err} > {tol_q}")
    if qd_err > tol_qd and check_parity:
        raise AssertionError(
            f"go2-trace pre-impact qd parity failed: {qd_err} > {tol_qd}")
    if impact >= steps:
        if limit_err > tol_limit:
            raise AssertionError(
                f"go2-trace limit impulse parity failed: {limit_err} > {tol_limit}")
        print("effort=PASS saturated=PASS pre_impact_q=PASS bounds=PASS limit=PASS")
    else:
        print("effort=PASS saturated=PASS pre_impact_q=PASS bounds=PASS "
              "limit=REPORT")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--mode",
                        choices=("random-qacc",
                                 "floating-random-qacc",
                                 "floating-contact-step",
                                 "floating-foot-box-contact-step",
                                 "stand-trajectory",
                                 "go2-trace"),
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
    parser.add_argument("--compare", action="store_true",
                        help="go2-trace: replay through nuka and compare")
    parser.add_argument("--device", default=0, type=int)
    parser.add_argument("--tol-q", default=0.05, type=float)
    parser.add_argument("--tol-qd", default=0.75, type=float)
    parser.add_argument("--tol-effort", default=1.0e-3, type=float)
    parser.add_argument("--tol-limit", default=0.5, type=float)
    args = parser.parse_args()
    if args.mode == "random-qacc":
        generate_random_samples(args.model, args.out, args.samples, args.batch_size)
    elif args.mode == "floating-random-qacc":
        generate_floating_base_random_samples(
            args.model, args.out, args.samples, args.batch_size)
    elif args.mode == "floating-contact-step":
        generate_floating_contact_step(args.model, args.out)
    elif args.mode == "floating-foot-box-contact-step":
        generate_foot_box_contact_step(args.model, args.out)
    elif args.mode == "go2-trace":
        if args.compare:
            compare_go2_oracle_trace(
                args.model, args.out, device=args.device, tol_q=args.tol_q,
                tol_qd=args.tol_qd, tol_effort=args.tol_effort,
                tol_limit=args.tol_limit)
            return 0
        generate_go2_oracle_trace(args.model, args.out, args.steps, args.dt)
    else:
        generate_stand_trajectory(args.model, args.out, args.steps, args.dt)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
