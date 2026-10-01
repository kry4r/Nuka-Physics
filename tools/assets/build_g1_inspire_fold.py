"""Build a fixed G1 upper body and a reproducible towel scene description."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pyarrow.parquet as parquet
from scipy.spatial.transform import Rotation


ARM_JOINTS = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
              "wrist_roll", "wrist_pitch", "wrist_yaw"]
HAND_JOINTS = ["thumb_1", "thumb_2", "index_1", "middle_1", "ring_1", "little_1"]


def source_gains(path):
    tree = ast.parse(path.read_text())
    controller = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "G1_29_ArmController")
    constructor = next(node for node in controller.body if isinstance(node, ast.FunctionDef)
                       and node.name == "__init__")
    gains = {}
    for node in constructor.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Attribute) and target.attr in (
                    "kp_low", "kd_low", "kp_wrist", "kd_wrist"):
                gains[target.attr] = float(ast.literal_eval(node.value))
    if len(gains) != 4 or not all(np.isfinite(value) and value > 0 for value in gains.values()):
        raise ValueError("Controller source does not contain the expected arm gains")
    return gains


def remove_legs(robot):
    children = {}
    for joint in robot.findall("joint"):
        children.setdefault(joint.find("parent").get("link"), []).append(
            joint.find("child").get("link"))
    removed = set()
    pending = ["left_hip_pitch_link", "right_hip_pitch_link"]
    while pending:
        name = pending.pop()
        removed.add(name)
        pending.extend(children.get(name, []))
    for link in list(robot.findall("link")):
        if link.get("name") in removed:
            robot.remove(link)
    for joint in list(robot.findall("joint")):
        if joint.find("child").get("link") in removed:
            robot.remove(joint)
    return sorted(removed)


def build(args):
    import nuka

    first = parquet.read_table(args.episode).slice(0, 1)
    body = np.asarray(first["observation.body"][0].as_py(), dtype=np.float64)
    if body.shape != (29,) or not np.isfinite(body).all():
        raise ValueError("Expected the episode's finite 29-joint body observation")
    gains = source_gains(args.controller_source)
    args.out.mkdir(parents=True, exist_ok=False)
    tree = ET.parse(args.source)
    robot = tree.getroot()
    robot.set("xmlns:nuka", "https://nuka-physics.org/urdf")
    removed = remove_legs(robot)
    joints = {joint.get("name"): joint for joint in robot.findall("joint")}
    locked = {}
    for index, axis in enumerate(("yaw", "roll", "pitch")):
        name = f"waist_{axis}_joint"
        joint = joints[name]
        origin = joint.find("origin")
        frame = Rotation.from_euler("xyz", np.fromstring(origin.get("rpy", "0 0 0"), sep=" "))
        direction = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
        angle = float(body[12 + index])
        frame = frame * Rotation.from_rotvec(direction * angle)
        origin.set("rpy", " ".join(f"{value:.17g}" for value in frame.as_euler("xyz")))
        joint.set("type", "fixed")
        for tag in ("limit", "dynamics"):
            child = joint.find(tag)
            if child is not None:
                joint.remove(child)
        locked[name] = angle
    meshes = []
    for link in robot.findall("link"):
        name = link.get("name")
        hand = name.startswith(("left_", "right_")) and any(
            token in name for token in ("base_link", "palm", "thumb", "index", "middle", "ring", "little"))
        for geometry in link.findall(".//geometry/mesh"):
            filename = geometry.get("filename")
            if "://" in filename:
                raise ValueError(f"Expected source-relative Unitree mesh: {filename}")
            path = (args.source.parent / filename).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            geometry.set("filename", str(path))
            meshes.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        if hand:
            for collision in link.findall("collision"):
                if collision.find("geometry/mesh") is not None:
                    ET.SubElement(collision, "nuka:mesh_triangle_limit", value="400")
    source_out = args.out / "upper_body.urdf"
    ET.indent(tree, space="  ")
    tree.write(source_out, encoding="utf-8", xml_declaration=True)
    scene_path = args.out / "robot.nks"
    nuka.Scene.load(str(source_out)).save(str(scene_path))
    document = json.loads(scene_path.read_text())
    nodes = {node["name"]: node for node in document["tree"] if "rigid_body" in node}
    nodes["pelvis"]["rigid_body"]["is_static"] = True
    nodes["pelvis"]["transform"]["pos"] = [0.0, 0.0, args.base_height]
    by_joint = {node["joint"]["name"]: node for node in nodes.values() if "joint" in node}
    scalar = {name: node for name, node in by_joint.items()
              if node["joint"]["type"] in ("hinge", "revolute", "slide", "prismatic")}
    if len(scalar) != 38:
        raise ValueError(f"Expected 38 scalar coordinates, got {len(scalar)}")
    active, mimic = {}, {}
    for side, offset in (("left", 15), ("right", 22)):
        for index, suffix in enumerate(ARM_JOINTS):
            name = f"{side}_{suffix}_joint"
            wrist = suffix.startswith("wrist")
            active[name] = {"kp": gains["kp_wrist" if wrist else "kp_low"],
                            "kd": gains["kd_wrist" if wrist else "kd_low"],
                            "initial_position": float(body[offset + index]), "kind": "arm"}
        for suffix in HAND_JOINTS:
            name = f"{side}_{suffix}_joint"
            joint = by_joint[name]["joint"]
            active[name] = {"kp": args.hand_kp, "kd": args.hand_kd, "kind": "hand",
                            "open": 0.0, "closed": float(joint["upper_limit"])}
            gripper_value = np.asarray(first[f"action.{side}_gripper"][0].as_py()).reshape(-1)
            if gripper_value.size != 1 or not np.isfinite(gripper_value).all():
                raise ValueError("Expected one finite gripper motor command")
            gripper = float(gripper_value[0])
            closure = float(np.clip(1.0 - gripper / args.gripper_open_motor, 0.0, 1.0))
            active[name]["initial_position"] = (active[name]["closed"] if suffix == "thumb_1"
                                                 else closure * active[name]["closed"])
            if suffix == "thumb_1":
                active[name]["open"] = active[name]["closed"]
    for name, node in scalar.items():
        source_joint = joints[name]
        relation = source_joint.find("mimic")
        if relation is not None:
            mimic[name] = {"source": relation.get("joint"),
                           "multiplier": float(relation.get("multiplier", "1")),
                           "offset": float(relation.get("offset", "0"))}
            node.pop("actuator", None)
            continue
        config = active[name]
        node["joint"]["initial_position"] = config["initial_position"]
        effort = float(source_joint.find("limit").get("effort"))
        config["force_limit"] = effort
        node["actuator"] = {"name": name + "_drive", "type": "position",
                            "joint_id": node["actuator"]["joint_id"],
                            "gain": config["kp"], "force_limit": effort}
    pending = dict(mimic)
    while pending:
        ready = [name for name, rule in pending.items() if rule["source"] not in pending]
        if not ready:
            raise ValueError("Cyclic mimic relation")
        for name in ready:
            rule = pending.pop(name)
            position = by_joint[rule["source"]]["joint"]["initial_position"]
            by_joint[name]["joint"]["initial_position"] = position * rule["multiplier"] + rule["offset"]
    if len(active) != 26 or len(mimic) != 12:
        raise ValueError("Expected 14 arm and 12 active hand coordinates, with 12 mimic coordinates")
    scene_path.write_text(json.dumps(document, indent=2) + "\n")
    description = {"robot": "robot.nks", "coordinates": 38, "active": active, "mimic": mimic,
                   "locked_waist": locked, "removed_links": removed, "base_height": args.base_height,
                   "gripper": {"motor_closed": 0.0, "motor_open": args.gripper_open_motor,
                               "normalization": "recorded motor range, not jaw distance calibration"},
                   "hand_gains_source": "explicit simulation parameters; hardware calibration unavailable",
                   "arm_gains_source": {"path": str(args.controller_source),
                        "sha256": hashlib.sha256(args.controller_source.read_bytes()).hexdigest()},
                   "table": {"size": [1.2, 0.8, 0.05], "position": [0.6, 0.0, 0.725], "friction": 0.4},
                   "cloth": {"nx": 101, "ny": 61, "spacing": 0.005, "origin": [0.6, 0.0, 0.754],
                             "density": 0.4, "stretch": 5000.0, "poisson": 0.3,
                             "bend": 5e-5, "thickness": 0.003, "friction": 0.5},
                   "source_urdf": {"path": str(args.source),
                        "sha256": hashlib.sha256(args.source.read_bytes()).hexdigest()},
                   "episode": {"path": str(args.episode),
                        "sha256": hashlib.sha256(args.episode.read_bytes()).hexdigest()}, "meshes": meshes,
                   "status": "requires_cook_report_initial_overlap_and_pinch_frame_validation"}
    (args.out / "scene.json").write_text(json.dumps(description, indent=2) + "\n")
    return description


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / ".nuka-assets/src/unitree_ros/robots/g1_description/g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--controller-source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-height", type=float, default=0.8)
    parser.add_argument("--hand-kp", type=float, default=20.0)
    parser.add_argument("--hand-kd", type=float, default=0.2)
    parser.add_argument("--gripper-open-motor", type=float, default=4.5)
    args = parser.parse_args()
    if not all(np.isfinite(value) and value > 0 for value in (
            args.hand_kp, args.hand_kd, args.gripper_open_motor, args.base_height)):
        raise ValueError("Scene and control parameters must be finite and positive")
    result = build(args)
    print(json.dumps({"coordinates": result["coordinates"], "active": len(result["active"]),
                      "mimic": len(result["mimic"]), "output": str(args.out), "status": result["status"]}))


if __name__ == "__main__":
    main()
