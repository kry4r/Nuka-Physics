"""Assemble an authored robot, cloth and table through the common world pipeline."""

import json
from pathlib import Path

import numpy as np

import nuka
from nuka.author import Scene, SimOptions, materials, morphs, surfaces


def build_scene(device, description, *, dt=1.0 / 600.0, sweeps=512, contact_capacity=524288):
    path = Path(description)
    config = json.loads(path.read_text())
    cloth, table = config["cloth"], config["table"]
    scene = Scene(SimOptions(dt=dt, gravity=(0.0, 0.0, -9.81), solver_vel_iters=sweeps,
                             solver_pos_iters=4, ogc_contact_capacity=contact_capacity))
    scene.add_entity(morphs.NKS(str(path.parent / config["robot"])))
    scene.add_entity(morphs.Grid(cloth["nx"], cloth["ny"], cloth["spacing"],
                                origin=tuple(cloth["origin"])),
                     materials.Cloth.VBD(areal_density=cloth["density"], friction=cloth["friction"],
                                         stretch_stiffness=cloth["stretch"], poisson=cloth["poisson"],
                                         bend_stiffness=cloth["bend"], thickness=cloth["thickness"]),
                     surfaces.Cloth(free=True))
    half_extents = tuple(0.5 * value for value in table["size"])
    scene.add_entity(morphs.Box(half_extents, pos=tuple(table["position"]),
                                quat=tuple(table.get("quat", (1.0, 0.0, 0.0, 0.0)))),
                     materials.Rigid(static=True, friction=table["friction"]))
    world = scene.build(device)
    try:
        names = world.dof_names()
        indices = {name: names.index(name) for name in config["active"]}
        stiffness = np.asarray(world.download_field(nuka.Field.DRIVE_STIFFNESS)).copy()
        damping = np.asarray(world.download_field(nuka.Field.DRIVE_DAMPING)).copy()
        limits = np.asarray(world.download_field(nuka.Field.DRIVE_FORCE_LIMIT)).copy()
        for name, values in config["active"].items():
            index = indices[name]
            stiffness.reshape(-1)[index] = values["kp"]
            damping.reshape(-1)[index] = values["kd"]
            limits.reshape(-1)[index] = values["force_limit"]
        world.upload_field(nuka.Field.DRIVE_STIFFNESS, np.ascontiguousarray(stiffness))
        world.upload_field(nuka.Field.DRIVE_DAMPING, np.ascontiguousarray(damping))
        world.upload_field(nuka.Field.DRIVE_FORCE_LIMIT, np.ascontiguousarray(limits))
        config["active_indices"] = indices
        config["dof_names"] = names
        return world, config
    except BaseException:
        world.destroy()
        raise


def folded_cloth(config):
    """Initial cloth positions (nx*ny, 3), lattice order j*nx+i, from the scene's one-fold layout.

    In the towel frame (x along the lattice rows, centred), the part beyond the fold line n.u = d wraps over a
    half cylinder, runs down a straight ramp and lies flat one layer gap above the base layer. The cross-section
    is a unit-speed planar curve swept along the line, so lattice lengths along and across the line are kept and
    the flat cooked rest shape stays the reference; chords over the curved hinge shorten by under one percent
    when the radius is at least two lattice spacings.
    """
    cloth = config["cloth"]
    layout = cloth["layout"]
    nx, ny, spacing = cloth["nx"], cloth["ny"], cloth["spacing"]
    radius, ramp, gap = layout["hinge_radius_m"], layout["ramp_length_m"], layout["layer_gap_m"]
    drop = 2.0 * radius - gap
    if not 0.0 < drop < ramp or radius < 2.0 * spacing:
        raise ValueError("the ramp must descend from the hinge top to the layer gap, over a hinge of two spacings")
    slope = np.arcsin(drop / ramp)
    i, j = np.meshgrid(np.arange(nx), np.arange(ny))
    u = (i.ravel() - 0.5 * (nx - 1)) * spacing
    v = (j.ravel() - 0.5 * (ny - 1)) * spacing
    normal = np.asarray(layout["fold_normal"], dtype=np.float64)
    normal /= np.linalg.norm(normal)
    s = u * normal[0] + v * normal[1] - layout["fold_offset"]
    arc = np.pi * radius
    theta = np.clip(s, 0.0, arc) / radius
    along = np.clip(s - arc, 0.0, ramp)
    beyond = np.maximum(s - arc - ramp, 0.0)
    out = np.where(s <= 0.0, s, np.where(s <= arc, radius * np.sin(theta),
                                         -along * np.cos(slope) - beyond))
    height = np.where(s <= 0.0, 0.0, np.where(s <= arc, radius * (1.0 - np.cos(theta)),
                                              2.0 * radius - along * np.sin(slope)))
    planar = np.stack([u, v], axis=1) + ((out - s)[:, None] * normal[None, :])
    c, s_yaw = np.cos(layout["yaw"]), np.sin(layout["yaw"])
    x = layout["center_xy"][0] + c * planar[:, 0] - s_yaw * planar[:, 1]
    y = layout["center_xy"][1] + s_yaw * planar[:, 0] + c * planar[:, 1]
    return np.stack([x, y, layout["base_height_m"] + height], axis=1)


def hand_targets(config, closure):
    """Map left and right normalized closure to the authored six-motor hand targets."""
    closure = np.asarray(closure, dtype=np.float64).reshape(-1)
    if closure.shape != (2,) or not np.isfinite(closure).all() or np.any((closure < 0) | (closure > 1)):
        raise ValueError("Expected two finite hand closures in [0, 1]")
    targets = {}
    for side, amount in zip(("left", "right"), closure):
        for name, values in config["active"].items():
            if values["kind"] == "hand" and name.startswith(side + "_"):
                targets[name] = values["open"] + amount * (values["closed"] - values["open"])
    return targets


def set_targets(world, config, joint_targets):
    values = np.asarray(world.download_field(nuka.Field.DRIVE_TARGET)).copy()
    for name, position in joint_targets.items():
        if name not in config["active_indices"] or not np.isfinite(position):
            raise ValueError(f"Invalid active joint target: {name}")
        values.reshape(-1)[config["active_indices"][name]] = position
    world.set_drive_targets(np.ascontiguousarray(values.reshape(-1), dtype=np.float32))
