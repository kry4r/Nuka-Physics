"""Robot-only fixture: an NKS robot without cloth or table, either following a towel-fold replay's recorded drive
targets and feedforward or swinging passively from its initial pose, against MuJoCo stepping the same NKS robot."""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import nuka
from nuka.author import Scene, SimOptions, morphs
from nuka.diagnostics import DiagnosticSession, DiagnosticThresholds
from nuka.diagnostics.capture import sha256, write_json
from nuka.tasks.fold_scene import build_scene


def production_tolerance():
    tolerance = os.environ.get("NUKA_SOLVER_VEL_TOLERANCE")
    if tolerance is None:
        raise ValueError("set the production NUKA_SOLVER_VEL_TOLERANCE explicitly for reproducibility")
    return float(tolerance)


def recorded_inputs(run, steps):
    """Per-step drive targets and feedforward of a replay, with the replay's own frame-0 settle and blends."""
    meta = json.loads((run / "session/manifest.json").read_text())["metadata"]
    chunks = [np.load(c) for c in sorted((run / "session").glob("chunk_*.npz"))]
    controls = np.concatenate([c["controls"] for c in chunks])
    recorded = np.concatenate([c["state_JOINT_POSITION"] for c in chunks])
    plan = np.load(run / "plan.npz")
    ff, settle, per_frame = plan["feedforward"], meta["settle_steps"], meta["substeps_per_frame"]
    steps = len(controls) if steps is None else min(steps, len(controls))
    feed = np.empty((steps, ff.shape[1]))
    for step in range(1, steps + 1):
        if step <= settle:
            feed[step - 1] = ff[0]
            continue
        k, m = divmod(step - settle - 1, per_frame)
        tau = (m + 1) / per_frame
        feed[step - 1] = (1 - tau) * ff[k] + tau * ff[k + 1]
    return meta, controls[:steps], feed, [str(n) for n in plan["feedforward_names"]], recorded[:steps]


def nks_couplings(nks):
    """Mimic couplings of the NKS joints by name; a coupling's source is a joint ordinal in tree order."""
    joints = [node["joint"] for node in nks["tree"] if node.get("joint") is not None]
    return {joint["name"]: {"source": joints[joint["mimic"]["source_joint"]]["name"],
                            "multiplier": float(joint["mimic"]["multiplier"]), "offset": float(joint["mimic"]["offset"])}
            for joint in joints if "mimic" in joint}


def mujoco_xml(nks, drives, dt, coupling_time):
    """MJCF of the NKS tree with PD ``drives`` {joint: (kp, kd, force limit)}; the mimic couplings Nuka reduces
    exactly become near-hard joint equalities with time constant ``coupling_time``."""
    bodies = nks["tree"]
    children = defaultdict(list)
    for index, node in enumerate(bodies):
        joint = node.get("joint")
        if joint is not None:
            if joint["child_body"] != index or joint["type"] not in ("fixed", "revolute"):
                raise ValueError(f"unsupported NKS joint {joint['name']}")
            frame = joint["child_frame"]
            if any(frame["pos"]) or list(frame["quat"]) != [1.0, 0.0, 0.0, 0.0]:
                raise ValueError(f"joint {joint['name']} has a child frame offset")
            children[joint["parent_body"]].append(index)
    roots = [i for i, node in enumerate(bodies) if node.get("joint") is None]
    if len(roots) != 1 or not bodies[roots[0]]["rigid_body"]["is_static"]:
        raise ValueError("the fixture expects one static root body")
    text = lambda values: " ".join(repr(float(v)) for v in values)
    root = ET.Element("mujoco", model="robot_only")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    option = ET.SubElement(root, "option", timestep=repr(dt), integrator="implicitfast", gravity="0 0 -9.81")
    ET.SubElement(option, "flag", contact="disable")
    world = ET.SubElement(root, "worldbody")

    def add(parent, index):
        node = bodies[index]
        joint = node.get("joint")
        frame = joint["parent_frame"] if joint is not None else node["transform"]
        body = ET.SubElement(parent, "body", name=node["name"], pos=text(frame["pos"]), quat=text(frame["quat"]))
        rigid = node["rigid_body"]
        ET.SubElement(body, "inertial", pos=text(rigid["inertial"]["pos"]), quat=text(rigid["inertial"]["quat"]),
                      mass=repr(float(rigid["mass"])), diaginertia=text(rigid["inertia"]))
        if joint is not None and joint["type"] == "revolute":
            attributes = {"name": joint["name"], "type": "hinge", "axis": text(joint["axis"]),
                          "damping": repr(float(joint["damping"])), "armature": repr(float(joint["armature"])),
                          "frictionloss": repr(float(joint["frictionloss"])),
                          "stiffness": repr(float(joint["stiffness"]))}
            if joint["has_lower_limit"] != joint["has_upper_limit"]:
                raise ValueError(f"joint {joint['name']} has a one-sided limit")
            if joint["has_lower_limit"]:
                attributes["range"] = text((joint["lower_limit"], joint["upper_limit"]))
            ET.SubElement(body, "joint", attributes)
        for child in children[index]:
            add(body, child)

    add(world, roots[0])
    actuators = ET.SubElement(root, "actuator")
    for name, (kp, kd, limit) in drives.items():
        ET.SubElement(actuators, "general", name=name, joint=name, gainprm=repr(kp), biastype="affine",
                      biasprm=text((0.0, -kp, -kd)), forcelimited="true", forcerange=text((-limit, limit)))
    equality = ET.SubElement(root, "equality")
    for name, mimic in nks_couplings(nks).items():
        ET.SubElement(equality, "joint", joint1=name, joint2=mimic["source"],
                      polycoef=text((mimic["offset"], mimic["multiplier"], 0.0, 0.0, 0.0)),
                      solref=text((coupling_time, 1.0)), solimp="0.9999 0.9999 0.001")
    return ET.tostring(root, encoding="unicode")


def run_mujoco(xml, names, initial, controls, drive_names, feed, feed_names, substeps):
    """MuJoCo trajectory at the fixture steps, stepping ``substeps`` times per fixture step with zero-order hold."""
    import mujoco
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    joint = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names}
    joint = {name: index for name, index in joint.items() if index >= 0}
    for name, index in joint.items():
        data.qpos[model.jnt_qposadr[index]] = initial[names.index(name)]
    mujoco.mj_forward(model, data)
    control_columns = [names.index(name) for name in drive_names]
    feed_dofs = [model.jnt_dofadr[joint[name]] for name in feed_names]
    q = np.full((len(controls), len(names)), np.nan)
    qd = np.full((len(controls), len(names)), np.nan)
    kinetic = np.zeros(len(controls))
    momentum = np.zeros(model.nv)
    columns = [(names.index(name), model.jnt_qposadr[index], model.jnt_dofadr[index]) for name, index in joint.items()]
    for step in range(len(controls)):
        data.ctrl[:] = controls[step, control_columns]
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[feed_dofs] = feed[step]
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        for column, qpos, dof in columns:
            q[step, column] = data.qpos[qpos]
            qd[step, column] = data.qvel[dof]
        mujoco.mj_mulM(model, data, momentum, data.qvel)
        kinetic[step] = 0.5 * float(data.qvel @ momentum)
        if not np.isfinite(data.qpos).all():
            break
    return q, qd, kinetic


def link_pose_error(xml, nks, names, q, poses):
    """Largest position and rotation gap per step between Nuka's link poses and MuJoCo kinematics at Nuka's q."""
    import mujoco
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    joint = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names}
    columns = [(names.index(name), model.jnt_qposadr[index]) for name, index in joint.items() if index >= 0]
    # Nuka orders links its own way and names each by its joint, the root by its body.
    body_of = {node["name"]: node["name"] for node in nks["tree"]}
    body_of.update({node["joint"]["name"]: node["name"] for node in nks["tree"] if node.get("joint") is not None})
    bodies = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_of[name]) for name in names]
    poses = poses.reshape(len(q), len(bodies), 7)
    position = np.zeros(len(q))
    rotation = np.zeros(len(q))
    for step in range(len(q)):
        for column, qpos in columns:
            data.qpos[qpos] = q[step, column]
        mujoco.mj_kinematics(model, data)
        position[step] = np.linalg.norm(poses[step, :, :3] - data.xpos[bodies], axis=1).max()
        a, b = poses[step, :, 3:], data.xquat[bodies]
        w = np.sum(a * b, axis=1)
        v = a[:, :1] * b[:, 1:] - b[:, :1] * a[:, 1:] - np.cross(a[:, 1:], b[:, 1:])
        rotation[step] = (2.0 * np.arctan2(np.linalg.norm(v, axis=1), np.abs(w))).max()
    return position, rotation


def dat_witnesses(witness):
    """Steps per DAT truncation witness kind and collidable body pair; a key packs beta, body and counterpart."""
    kinds = ("motion", "particle_pair", "body_pair", "chain")
    count = defaultdict(int)
    for row in witness:
        for kind, key in zip(kinds, (int(value) for value in row)):
            if key != (1 << 64) - 1:
                body, other = (key >> 16) & 0xFFFF, key & 0xFFFF
                count[f"{kind} body {body} counterpart {other}"] += 1
    return dict(sorted(count.items(), key=lambda item: -item[1]))


def groups(names, kinds, couplings):
    """Joint groups by side and kind, with each side's mimic joints apart."""
    out = defaultdict(list)
    for name, kind in list(kinds.items()) + [(name, "mimic") for name in couplings]:
        side = "left_" if name.startswith("left_") else "right_" if name.startswith("right_") else ""
        out[side + kind].append(names.index(name))
    return dict(out)


def stats(values):
    values = np.abs(np.asarray(values))
    if not np.isfinite(values).all():
        return {"max": None, "p95": None, "finite": False}
    return {"max": float(values.max()), "p95": float(np.percentile(values, 95)), "finite": True}


def passive_world(device, robot, dt, sweeps, max_pairs, contact_capacity):
    """The NKS robot alone under gravity with every drive gain cleared, so it swings from its initial pose."""
    scene = Scene(SimOptions(dt=dt, gravity=(0.0, 0.0, -9.81), solver_vel_iters=sweeps, solver_pos_iters=4,
                             solver_max_pairs=max_pairs, ogc_contact_capacity=contact_capacity))
    scene.add_entity(morphs.NKS(str(robot)))
    world = scene.build(device)
    try:
        for field in (nuka.Field.DRIVE_STIFFNESS, nuka.Field.DRIVE_DAMPING, nuka.Field.JOINT_FEEDFORWARD):
            values = np.zeros_like(np.asarray(world.download_field(field), np.float32))
            world.upload_field(field, np.ascontiguousarray(values))
        return world
    except BaseException:
        world.destroy()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, help="replay: scene directory with scene.json and the robot NKS")
    parser.add_argument("--run", type=Path, help="replay: towel-fold replay whose controls are followed")
    parser.add_argument("--robot", type=Path, help="passive swing: robot NKS with one static root body")
    parser.add_argument("--dt", type=float, default=1.0 / 600.0, help="passive swing step")
    parser.add_argument("--max-pairs", type=int, default=0, help="passive swing: link pair capacity, 0 for the default")
    parser.add_argument("--contact-capacity", type=int, default=0,
                        help="passive swing: mesh contact capacity, 0 for the default")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-record", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--sweeps", type=int, default=512)
    parser.add_argument("--chunk-steps", type=int, default=600)
    parser.add_argument("--reference-substeps", type=int, default=10,
                        help="MuJoCo steps per fixture step; the reference runs at this and twice this rate")
    parser.add_argument("--horizons", type=int, nargs="*", default=(60, 300),
                        help="leading step counts over which the reference difference is also reported")
    args = parser.parse_args()
    if (args.run is None) == (args.robot is None) or (args.run is not None and args.scene is None):
        parser.error("give --scene with --run for a replay, or --robot with --steps for a passive swing")
    if args.robot is not None and args.steps is None:
        parser.error("a passive swing needs --steps")
    tolerance = production_tolerance()
    args.output.mkdir(parents=True)
    device = nuka.Device.create(0)
    if args.run is not None:
        meta, controls, feed, feed_names, recorded = recorded_inputs(args.run, args.steps)
        dt = float(meta["dt"])
        robot = args.scene / json.loads((args.scene / "scene.json").read_text())["robot"]
        world, config = build_scene(device, args.scene / "scene.json", dt=dt, sweeps=args.sweeps, robot_only=True)
        drives = {name: (float(d["kp"]), float(d["kd"]), float(d["force_limit"])) for name, d in config["active"].items()}
        kinds = {name: drive["kind"] for name, drive in config["active"].items()}
        source = {"scene": str(args.scene / "scene.json"), "scene_sha256": sha256(args.scene / "scene.json"),
                  "run": str(args.run)}
    else:
        dt, robot, recorded = args.dt, args.robot, None
        world = passive_world(device, robot, dt, args.sweeps, args.max_pairs, args.contact_capacity)
        drives, feed_names, source = {}, [], {"passive_swing": True}
    nks = json.loads(robot.read_text())
    couplings = nks_couplings(nks)
    session = None
    try:
        names = list(world.dof_names())
        initial = np.asarray(world.download_field(nuka.Field.JOINT_POSITION), np.float32).reshape(-1).copy()
        if args.run is None:
            controls = np.tile(initial, (args.steps, 1))
            feed = np.zeros((args.steps, 0))
            kinds = {node["joint"]["name"]: "joint" for node in nks["tree"] if node.get("joint") is not None
                     and node["joint"]["type"] == "revolute" and node["joint"]["name"] not in couplings}
        else:
            feed_slots = [config["active_indices"][name] for name in feed_names]
        base_feed = np.asarray(world.download_field(nuka.Field.JOINT_FEEDFORWARD), np.float32).reshape(-1).copy()
        metadata = {"fixture": "robot only", **source, "max_pairs": args.max_pairs, "contact_capacity": args.contact_capacity, "robot": str(robot), "robot_sha256": sha256(robot), "dt": dt,
                    "sweeps": args.sweeps, "steps": len(controls), "solver_velocity_tolerance_mps": tolerance,
                    "build_record": str(args.build_record),
                    "binary_hashes": (args.build_record / "binaries.sha256").read_text(),
                    "fixture_sha256": sha256(__file__), "owner_names": {"LINK": names},
                    "render_acceptance": "unmeasured", "claims_full_physics_acceptance": False}
        session = DiagnosticSession(world, args.output / "session", metadata, chunk_steps=args.chunk_steps,
                                    state_fields=(nuka.Field.ARTICULATION_LINK_POSE, nuka.Field.JOINT_POSITION,
                                                  nuka.Field.JOINT_VELOCITY, nuka.Field.JOINT_LIMIT_IMPULSE,
                                                  nuka.Field.LINK_CONTACT_WRENCH, nuka.Field.DAT_ARTICULATION_FRACTION,
                                                  nuka.Field.DAT_ARTICULATION_WITNESS),
                                    thresholds=DiagnosticThresholds(velocity_tolerance_mps=tolerance))
        reason = None
        for step in range(len(controls)):
            targets = np.ascontiguousarray(controls[step].astype(np.float32))
            if args.run is not None:
                values = base_feed.copy()
                values[feed_slots] = feed[step]
                world.set_drive_targets(targets)
                world.upload_field(nuka.Field.JOINT_FEEDFORWARD, np.ascontiguousarray(values))
            sample = session.step(controls=targets.copy())
            if np.any(sample["env_status"]):
                reason = "physics_failure"
                break
        reason = reason or "completed"
        session.close(reason)
        chunks = [np.load(c) for c in sorted((args.output / "session").glob("chunk_*.npz"))]
        q = np.concatenate([c["state_JOINT_POSITION"] for c in chunks]).astype(np.float64)
        qd = np.concatenate([c["state_JOINT_VELOCITY"] for c in chunks]).astype(np.float64)
        poses = np.concatenate([c["state_ARTICULATION_LINK_POSE"] for c in chunks]).astype(np.float64)
        wrench = np.concatenate([c["state_LINK_CONTACT_WRENCH"] for c in chunks])
        fraction = np.concatenate([c["state_DAT_ARTICULATION_FRACTION"] for c in chunks]).reshape(len(q), -1)
        witness = np.concatenate([c["state_DAT_ARTICULATION_WITNESS"] for c in chunks]).reshape(len(q), -1)
        energy = np.concatenate([c["energy"] for c in chunks]).reshape(len(q), -1, len(session.manifest["energy_columns"]))
        steps = len(q)
        reference_xml = mujoco_xml(nks, drives, dt, 2.0 * dt)
        (args.output / "reference.xml").write_text(reference_xml)
        # The stiff couplings on the light finger links need a finer MuJoCo step; twice as fine bounds its error.
        coarse, fine = args.reference_substeps, 2 * args.reference_substeps
        references = {}
        for substeps in (coarse, fine):
            mq, mqd, kinetic = run_mujoco(mujoco_xml(nks, drives, dt / substeps, 2.0 * dt), names, initial,
                                          controls[:steps], list(drives), feed[:steps], feed_names, substeps)
            references[substeps] = (mq, mqd, kinetic)
            np.savez_compressed(args.output / f"mujoco_substeps_{substeps}.npz", q=mq, qd=mqd, kinetic=kinetic)
        reference_q, reference_qd, _ = references[fine]
        position, rotation = link_pose_error(reference_xml, nks, names, q, poses)
        # The reference has no contact, so self-contact steps are compared apart.
        touching = np.abs(wrench.reshape(len(wrench), len(names), -1)).max(axis=2) > 0.0
        free = ~touching.any(axis=1)
        metrics = {"stop_reason": reason, "steps": steps, "dt": dt, "sweeps": args.sweeps,
                   "contact_steps": int(np.count_nonzero(~free)),
                   "contact_links": [names[link] for link in np.flatnonzero(touching.any(axis=0))],
                   "link_pose_position_error_m": stats(position), "link_pose_rotation_error_rad": stats(rotation),
                   "dat_truncated_steps": int(np.count_nonzero(fraction.min(axis=1) < 1.0)),
                   "dat_kinetic_loss_j": float(energy[:, :, session.manifest["energy_columns"].index("DAT_KINETIC_LOSS")].sum()),
                   "dat_witnesses": dat_witnesses(witness), "groups": {}}
        horizons = sorted({min(h, steps) for h in args.horizons} | {steps})
        for group, columns in sorted(groups(names, kinds, couplings).items()):
            entry = {"joints": [names[c] for c in columns]}
            if drives and not group.endswith("mimic"):
                target = controls[:steps, columns].astype(np.float64)
                entry["nuka_tracking_rad"] = stats(q[:, columns] - target)
                for substeps, (mq, _, _) in references.items():
                    entry[f"mujoco_{substeps}_tracking_rad"] = stats(mq[:, columns] - target)
            entry["nuka_minus_reference_q_rad"] = {str(h): stats(q[:h, columns] - reference_q[:h, columns])
                                                   for h in horizons}
            entry["nuka_minus_reference_qd_radps"] = stats(qd[:, columns] - reference_qd[:, columns])
            if free.any():
                entry["contact_free_nuka_minus_reference_q_rad"] = stats(q[free][:, columns] - reference_q[free][:, columns])
            entry["reference_step_error_q_rad"] = stats(references[coarse][0][:, columns] - reference_q[:, columns])
            if recorded is not None:
                entry["recorded_run_q_difference_rad"] = stats(q[:, columns] - recorded[:steps, columns])
            metrics["groups"][group] = entry
        for label, positions in (("nuka", q), ("mujoco", reference_q)):
            residual = [stats(positions[:, names.index(name)] - coupling["offset"] -
                              coupling["multiplier"] * positions[:, names.index(coupling["source"])])
                        for name, coupling in couplings.items()]
            metrics[f"{label}_mimic_residual_rad"] = max((math.inf if r["max"] is None else r["max"] for r in residual), default=0.0)
        metrics["mujoco_kinetic_max_j"] = {str(k): float(np.nanmax(v[2])) for k, v in references.items()}
        write_json(args.output / "fixture_metrics.json", metrics)
        print(json.dumps(metrics), flush=True)
    except BaseException as error:
        if session is not None and not session.closed:
            session.close("exception", error=repr(error))
        raise
    finally:
        world.destroy()
        device.close()


if __name__ == "__main__":
    main()
