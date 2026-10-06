"""Robot-only fixture: the towel-fold robot without cloth or table follows a replay's recorded drive targets and
feedforward through the diagnostic session, against MuJoCo stepping the same NKS robot as the reference."""

import argparse
import json
import math
import os
from collections import defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import nuka
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


def mujoco_xml(nks, config, dt, coupling_time):
    """MJCF of the NKS tree with the scene's PD gains and force limits; the mimic couplings Nuka reduces exactly
    become near-hard joint equalities with time constant ``coupling_time``."""
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
    root = ET.Element("mujoco", model="m3_upper_body")
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
    for name, drive in config["active"].items():
        kp, kd, limit = float(drive["kp"]), float(drive["kd"]), float(drive["force_limit"])
        ET.SubElement(actuators, "general", name=name, joint=name, gainprm=repr(kp), biastype="affine",
                      biasprm=text((0.0, -kp, -kd)), forcelimited="true", forcerange=text((-limit, limit)))
    equality = ET.SubElement(root, "equality")
    for name, mimic in config["mimic"].items():
        ET.SubElement(equality, "joint", joint1=name, joint2=mimic["source"],
                      polycoef=text((mimic["offset"], mimic["multiplier"], 0.0, 0.0, 0.0)),
                      solref=text((coupling_time, 1.0)), solimp="0.9999 0.9999 0.001")
    return ET.tostring(root, encoding="unicode")


def run_mujoco(xml, names, initial, controls, feed, feed_names, config, substeps):
    """MuJoCo trajectory at the fixture steps, stepping ``substeps`` times per fixture step with zero-order hold."""
    import mujoco
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    joint = {name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names}
    joint = {name: index for name, index in joint.items() if index >= 0}
    for name, index in joint.items():
        data.qpos[model.jnt_qposadr[index]] = initial[names.index(name)]
    mujoco.mj_forward(model, data)
    active = list(config["active"])
    control_columns = [names.index(name) for name in active]
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


def groups(config, names):
    """Fixture joint groups: waist, each arm, each hand's driven joints and each hand's mimic joints."""
    out = defaultdict(list)
    for name, drive in config["active"].items():
        side = "left" if name.startswith("left_") else "right" if name.startswith("right_") else ""
        kind = drive["kind"]
        out[f"{side}_{kind}" if side else kind].append(names.index(name))
    for name in config["mimic"]:
        out[f"{'left' if name.startswith('left_') else 'right'}_mimic"].append(names.index(name))
    return dict(out)


def stats(values):
    values = np.abs(np.asarray(values))
    if not np.isfinite(values).all():
        return {"max": None, "p95": None, "finite": False}
    return {"max": float(values.max()), "p95": float(np.percentile(values, 95)), "finite": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True, help="scene directory with scene.json and the robot NKS")
    parser.add_argument("--run", type=Path, required=True, help="towel-fold replay whose controls are followed")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-record", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--sweeps", type=int, default=512)
    parser.add_argument("--chunk-steps", type=int, default=600)
    parser.add_argument("--reference-substeps", type=int, default=10,
                        help="MuJoCo steps per fixture step; the reference runs at this and twice this rate")
    args = parser.parse_args()
    tolerance = production_tolerance()
    meta, controls, feed, feed_names, recorded = recorded_inputs(args.run, args.steps)
    dt = float(meta["dt"])
    config = json.loads((args.scene / "scene.json").read_text())
    nks = json.loads((args.scene / config["robot"]).read_text())
    args.output.mkdir(parents=True)
    device = nuka.Device.create(0)
    world, config = build_scene(device, args.scene / "scene.json", dt=dt, sweeps=args.sweeps, robot_only=True)
    session = None
    try:
        names = list(world.dof_names())
        slots = config["active_indices"]
        feed_slots = [slots[name] for name in feed_names]
        initial = np.asarray(world.download_field(nuka.Field.JOINT_POSITION), np.float32).reshape(-1).copy()
        base_feed = np.asarray(world.download_field(nuka.Field.JOINT_FEEDFORWARD), np.float32).reshape(-1).copy()
        metadata = {"fixture": "robot only", "scene": str(args.scene / "scene.json"),
                    "scene_sha256": sha256(args.scene / "scene.json"), "robot_sha256": sha256(args.scene / config["robot"]),
                    "run": str(args.run), "dt": dt, "sweeps": args.sweeps, "steps": len(controls),
                    "solver_velocity_tolerance_mps": tolerance, "build_record": str(args.build_record),
                    "binary_hashes": (args.build_record / "binaries.sha256").read_text(),
                    "fixture_sha256": sha256(__file__), "owner_names": {"LINK": names},
                    "render_acceptance": "unmeasured", "claims_full_physics_acceptance": False}
        session = DiagnosticSession(world, args.output / "session", metadata, chunk_steps=args.chunk_steps,
                                    state_fields=(nuka.Field.JOINT_POSITION, nuka.Field.JOINT_VELOCITY,
                                                  nuka.Field.JOINT_LIMIT_IMPULSE, nuka.Field.LINK_CONTACT_WRENCH),
                                    thresholds=DiagnosticThresholds(velocity_tolerance_mps=tolerance))
        reason = None
        for step in range(len(controls)):
            targets = np.ascontiguousarray(controls[step].astype(np.float32))
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
        wrench = np.concatenate([c["state_LINK_CONTACT_WRENCH"] for c in chunks])
        (args.output / "reference.xml").write_text(mujoco_xml(nks, config, dt, 2.0 * dt))
        # The stiff couplings on the light finger links need a finer MuJoCo step; twice as fine bounds its error.
        coarse, fine = args.reference_substeps, 2 * args.reference_substeps
        references = {}
        for substeps in (coarse, fine):
            mq, mqd, kinetic = run_mujoco(mujoco_xml(nks, config, dt / substeps, 2.0 * dt), names, initial,
                                          controls[:len(q)], feed[:len(q)], feed_names, config, substeps)
            references[substeps] = (mq, mqd, kinetic)
            np.savez_compressed(args.output / f"mujoco_substeps_{substeps}.npz", q=mq, qd=mqd, kinetic=kinetic)
        joint_groups = groups(config, names)
        steps = len(q)
        reference_q, reference_qd, _ = references[fine]
        # The reference has no contact, so self-contact steps are compared apart.
        touching = np.abs(wrench.reshape(len(wrench), len(names), -1)).max(axis=2) > 0.0
        free = ~touching.any(axis=1)
        metrics = {"stop_reason": reason, "steps": steps, "dt": dt, "sweeps": args.sweeps,
                   "contact_steps": int(np.count_nonzero(~free)),
                   "contact_links": [names[link] for link in np.flatnonzero(touching.any(axis=0))],
                   "recorded_cloth_run_q_difference": {}, "groups": {}}
        for group, columns in sorted(joint_groups.items()):
            entry = {"joints": [names[c] for c in columns]}
            if not group.endswith("_mimic"):
                target = controls[:steps, columns].astype(np.float64)
                entry["nuka_tracking_rad"] = stats(q[:, columns] - target)
                for substeps, (mq, _, _) in references.items():
                    entry[f"mujoco_{substeps}_tracking_rad"] = stats(mq[:, columns] - target)
            entry["nuka_minus_reference_q_rad"] = stats(q[:, columns] - reference_q[:, columns])
            entry["nuka_minus_reference_qd_radps"] = stats(qd[:, columns] - reference_qd[:, columns])
            if free.any():
                entry["contact_free_nuka_minus_reference_q_rad"] = stats(q[free][:, columns] - reference_q[free][:, columns])
            entry["reference_step_error_q_rad"] = stats(references[coarse][0][:, columns] - reference_q[:, columns])
            metrics["groups"][group] = entry
            metrics["recorded_cloth_run_q_difference"][group] = stats(q[:, columns] - recorded[:steps, columns])
        for label, positions in (("nuka", q), ("mujoco", reference_q)):
            residual = [stats(positions[:, names.index(name)] - coupling["offset"] -
                              coupling["multiplier"] * positions[:, names.index(coupling["source"])])
                        for name, coupling in config["mimic"].items()]
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
