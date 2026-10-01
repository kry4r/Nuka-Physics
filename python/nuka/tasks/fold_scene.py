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
    scene.add_entity(morphs.Box(half_extents, pos=tuple(table["position"])),
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
