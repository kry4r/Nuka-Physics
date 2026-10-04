"""Build the fixed G1 upper body with Inspire hands and the towel scene description for one recorded episode."""

import argparse
import ast
import hashlib
import json
import math
import re
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from nuka.tasks import fold_kinematics as fk  # noqa: E402


GAINS = ("kp_high", "kd_high", "kp_low", "kd_low", "kp_wrist", "kd_wrist")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
            if isinstance(target, ast.Attribute) and target.attr in GAINS:
                gains[target.attr] = float(ast.literal_eval(node.value))
    if len(gains) != len(GAINS) or not all(np.isfinite(value) and value > 0 for value in gains.values()):
        raise ValueError("Controller source does not contain the expected joint gains")
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


def lattice_size(length, spacing):
    count = round(length / spacing) + 1
    if count < 2 or not math.isclose((count - 1) * spacing, length, abs_tol=0.5 * spacing):
        raise ValueError("towel size must be resolvable by the lattice spacing")
    return count


def build(args):
    import nuka

    match = re.search(r"episode_(\d+)\.parquet$", args.episode.name)
    if match is None:
        raise ValueError("Expected an episode_<index>.parquet file")
    episode = str(int(match.group(1)))
    columns = fk.episode_columns(args.episode)
    body = columns["observation.body"]
    if body.shape[1] != 29 or not np.isfinite(body).all():
        raise ValueError("Expected the episode's finite 29-joint body observation")
    measurements = json.loads(args.measurements.read_text())
    plane = measurements["table_plane_lowest_pad_w_m"]
    slope_y, table_top = float(plane["slope_y"]), float(plane["offsets_m"][episode])
    edge = measurements["episodes"][episode]["table_edge"]["z_fixed_at_pads"]
    layout = json.loads(args.layout.read_text())
    if int(layout["episode"]) != int(episode) or layout["frame"] != 0:
        raise ValueError("The towel layout must come from the episode's first frame")
    if not np.allclose(layout["towel"]["size_wh"], args.towel_size):
        raise ValueError("The towel layout was fitted with a different towel size")
    gains = source_gains(args.controller_source)
    dex1 = fk.Kinematics(args.dex1_urdf)
    inspire = fk.Kinematics(args.source)
    R_wp, p_wp = fk.leg_world(dex1, body, slope_y)
    base_quat = Rotation.from_matrix(R_wp[0]).as_quat()
    hands = {}
    for side in fk.SIDES:
        pinch = fk.inspire_pinch(inspire, side, args.thickness, args.hand_kp, args.pinch_force,
                                 args.thumb_rotation, args.thumb_bend)
        dex1_R, dex1_p = fk.dex1_pinch_in_wrist(dex1, side)
        rotation = fk.match_closing_sign(dex1_R, np.array(pinch["pinch_rotation"]))
        pinch["pinch_rotation"] = rotation.tolist()
        pinch["dex1_pinch_rotation"] = dex1_R.tolist()
        pinch["dex1_pinch_position_m"] = dex1_p.tolist()
        hands[side] = pinch
    pads = {link for side in fk.SIDES for link in hands[side]["pad_links"]}
    args.out.mkdir(parents=True, exist_ok=False)
    tree = ET.parse(args.source)
    robot = tree.getroot()
    robot.set("xmlns:nuka", "https://nuka-physics.org/urdf")
    removed = remove_legs(robot)
    joints = {joint.get("name"): joint for joint in robot.findall("joint")}
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
            meshes.append({"path": str(path), "sha256": sha256(path)})
        for collision in link.findall("collision"):
            if collision.find("geometry/mesh") is not None:
                limit = args.hand_error_limit if hand else args.body_error_limit
                ET.SubElement(collision, "nuka:mesh_error_limit", value=repr(limit))
    source_out = args.out / "upper_body.urdf"
    ET.indent(tree, space="  ")
    tree.write(source_out, encoding="utf-8", xml_declaration=True)
    scene_path = args.out / "robot.nks"
    nuka.Scene.load(str(source_out)).save(str(scene_path))
    document = json.loads(scene_path.read_text())
    nodes = {node["name"]: node for node in document["tree"] if "rigid_body" in node}
    nodes["pelvis"]["rigid_body"]["is_static"] = True
    nodes["pelvis"]["transform"] = {**nodes["pelvis"].get("transform", {}), "pos": p_wp[0].tolist(),
                                    "quat": [float(base_quat[3]), *map(float, base_quat[:3])]}
    by_joint = {node["joint"]["name"]: node for node in nodes.values() if "joint" in node}
    scalar = {name: node for name, node in by_joint.items()
              if node["joint"]["type"] in ("hinge", "revolute", "slide", "prismatic")}
    if len(scalar) != 41:
        raise ValueError(f"Expected 41 scalar coordinates, got {len(scalar)}")
    first = {side: float(fk.closure(columns, side)[0]) for side in fk.SIDES}
    active, mimic = {}, {}
    for index, name in enumerate(fk.waist_joints()):
        active[name] = {"kp": gains["kp_high"], "kd": gains["kd_high"], "kind": "waist",
                        "initial_position": float(body[0, 12 + index])}
    for side, offset in (("left", 15), ("right", 22)):
        for index, name in enumerate(fk.arm_joints(side)):
            wrist = "wrist" in name
            active[name] = {"kp": gains["kp_wrist" if wrist else "kp_low"],
                            "kd": gains["kd_wrist" if wrist else "kd_low"], "kind": "arm",
                            "initial_position": float(body[0, offset + index])}
        for name, (opened, closed) in hands[side]["targets_open_closed"].items():
            active[name] = {"kp": args.hand_kp, "kd": args.hand_kd, "kind": "hand", "open": opened, "closed": closed,
                            "initial_position": opened + first[side] * (closed - opened)}
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
        side = name.split("_")[0]
        config["force_limit"] = (hands[side]["thumb_torque_limit_nm"] if name.endswith("thumb_2_joint")
                                 else effort)
        node["actuator"] = {"name": name + "_drive", "type": "position",
                            "joint_id": node["actuator"]["joint_id"],
                            "gain": config["kp"], "force_limit": config["force_limit"]}
    pending = dict(mimic)
    while pending:
        ready = [name for name, rule in pending.items() if rule["source"] not in pending]
        if not ready:
            raise ValueError("Cyclic mimic relation")
        for name in ready:
            rule = pending.pop(name)
            position = by_joint[rule["source"]]["joint"]["initial_position"]
            by_joint[name]["joint"]["initial_position"] = position * rule["multiplier"] + rule["offset"]
    if len(active) != 29 or len(mimic) != 12:
        raise ValueError("Expected 3 waist, 14 arm and 12 hand active coordinates, with 12 mimic coordinates")
    scene_path.write_text(json.dumps(document, indent=2) + "\n")
    # The table's near edge is x = x_e + s y on the measured plane; its depth runs away from the robot.
    yaw = -math.atan(edge["slope"])
    depth, width, thickness = args.table_size
    centre = np.array([edge["x_m"] + 0.5 * depth * math.cos(yaw), 0.5 * depth * math.sin(yaw),
                       table_top - 0.5 * thickness])
    towel = layout["towel"]
    nx, ny = lattice_size(args.towel_size[0], args.spacing), lattice_size(args.towel_size[1], args.spacing)
    description = {
        "robot": "robot.nks", "coordinates": 41, "active": active, "mimic": mimic, "removed_links": removed,
        "world": {"definition": "left sole flat on the floor, turned about x through the initial pelvis so the "
                                "measured table plane is level; x along the initial pelvis heading",
                  "slope_y": slope_y, "pelvis_position_m": p_wp[0].tolist(),
                  "pelvis_quat_wxyz": nodes["pelvis"]["transform"]["quat"],
                  "dex1_urdf": {"path": str(args.dex1_urdf), "sha256": sha256(args.dex1_urdf)},
                  "measurements": {"path": str(args.measurements), "sha256": sha256(args.measurements)}},
        "gains": {"values": gains, "waist": "kp_high/kd_high: non-arm joints of the source controller",
                  "source": {"path": str(args.controller_source), "sha256": sha256(args.controller_source)},
                  "feedforward": "gravity holding torques on the waist and arm joints, as the source controller "
                                 "sends its inverse-dynamics gravity term with each arm command"},
        "hand": {"kp": args.hand_kp, "kd": args.hand_kd, "pinch_force_n": args.pinch_force,
                 "thickness_m": args.thickness,
                 "posture": {"thumb_rotation": args.thumb_rotation, "thumb_bend": args.thumb_bend,
                             "selection": [{"path": str(path), "sha256": sha256(path)}
                                           for path in args.posture_evidence]},
                 "mapping": "closure g in [0, 1]: target = open + g (closed - open) per joint; the thumb holds its "
                            "rotation, thumb and index bends follow g, middle, ring and little stay half closed",
                 "left": hands["left"], "right": hands["right"]},
        "gripper": {"motor_closed": 0.0, "motor_open": fk.GRIPPER_OPEN,
                    "normalization": "recorded motor range, not jaw distance calibration"},
        "table": {"size": [depth, width, thickness], "position": centre.tolist(),
                  "quat": [math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], "friction": args.table_friction,
                  "top_m": table_top, "near_edge": edge},
        "cloth": {"nx": nx, "ny": ny, "spacing": args.spacing,
                  "origin": [towel["center_xy"][0], towel["center_xy"][1], table_top + args.clearance],
                  "density": 0.4, "stretch": 5000.0, "poisson": 0.3, "bend": 5e-5, "thickness": args.thickness,
                  "friction": 0.5, "measured_size_m": list(args.towel_size),
                  "layout": {"center_xy": towel["center_xy"], "yaw": towel["yaw"],
                             "fold_normal": towel["fold"]["normal_towel_frame"], "fold_offset": towel["fold"]["offset"],
                             "base_height_m": table_top + args.clearance, "hinge_radius_m": args.hinge_radius,
                             "ramp_length_m": args.ramp_length, "layer_gap_m": args.layer_gap,
                             "source": {"path": str(args.layout), "sha256": sha256(args.layout)}}},
        "source_urdf": {"path": str(args.source), "sha256": sha256(args.source)},
        "episode": {"index": int(episode), "path": str(args.episode), "sha256": sha256(args.episode)},
        "meshes": meshes, "pad_links": sorted(pads),
        "collision_mesh_error_limit_m": {"hand": args.hand_error_limit, "body": args.body_error_limit,
                                         "measure": "sampled Hausdorff distance between the source and cooked "
                                                    "surfaces, both directions; fewest triangles within it"},
        "status": "requires_cook_report_initial_overlap_and_pinch_frame_validation"}
    (args.out / "scene.json").write_text(json.dumps(description, indent=2) + "\n")
    return description


def main():
    root = Path(__file__).resolve().parents[2]
    description = root / ".nuka-assets/src/unitree_ros/robots/g1_description"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=description / "g1_29dof_rev_1_0_with_inspire_hand_FTP.urdf")
    parser.add_argument("--dex1-urdf", type=Path, default=description / "g1_29dof_mode_15_with_dex1_1.urdf")
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--controller-source", type=Path, required=True)
    parser.add_argument("--measurements", type=Path, required=True,
                        help="world-frame analysis with the table plane and near edge per episode")
    parser.add_argument("--layout", type=Path, required=True, help="first-frame towel layout fit")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--towel-size", type=float, nargs=2, required=True, metavar=("LENGTH", "WIDTH"))
    parser.add_argument("--spacing", type=float, default=0.005)
    parser.add_argument("--thickness", type=float, default=0.003)
    parser.add_argument("--clearance", type=float, default=0.005, help="base layer height above the table top")
    parser.add_argument("--hinge-radius", type=float, default=0.01)
    parser.add_argument("--ramp-length", type=float, default=0.05)
    parser.add_argument("--layer-gap", type=float, default=0.0033)
    parser.add_argument("--table-size", type=float, nargs=3, default=(0.8, 1.2, 0.05),
                        metavar=("DEPTH", "WIDTH", "THICKNESS"))
    parser.add_argument("--table-friction", type=float, default=0.4)
    parser.add_argument("--hand-kp", type=float, default=20.0)
    parser.add_argument("--hand-kd", type=float, default=0.2)
    parser.add_argument("--pinch-force", type=float, default=2.0)
    parser.add_argument("--thumb-rotation", type=float, required=True, help="thumb_1 of the pinch posture, rad")
    parser.add_argument("--thumb-bend", type=float, required=True, help="thumb_2 of the pinch posture, rad")
    parser.add_argument("--posture-evidence", type=Path, nargs="+", required=True,
                        help="arm reachability evaluations the pinch posture was selected from")
    parser.add_argument("--hand-error-limit", type=float, default=1e-4,
                        help="largest collision surface deviation on the hand links, m")
    parser.add_argument("--body-error-limit", type=float, default=1e-3,
                        help="largest collision surface deviation on the other links, m")
    args = parser.parse_args()
    if not all(np.isfinite(value) and value > 0 for value in (
            args.hand_kp, args.hand_kd, args.pinch_force, args.spacing, args.thickness, args.clearance,
            args.hinge_radius, args.ramp_length, args.layer_gap, *args.towel_size, *args.table_size)):
        raise ValueError("Scene and control parameters must be finite and positive")
    if not all(0.0 < limit <= 1e-3 for limit in (args.hand_error_limit, args.body_error_limit)):
        raise ValueError("Collision mesh error limits must lie in (0, 1 mm]")
    result = build(args)
    print(json.dumps({"coordinates": result["coordinates"], "active": len(result["active"]),
                      "mimic": len(result["mimic"]), "output": str(args.out), "status": result["status"],
                      "pinch": {side: {key: result["hand"][side][key] for key in
                                       ("index_bend", "pad_gap_m", "open_pad_gap_m", "thumb_lever_m",
                                        "thumb_torque_limit_nm")}
                                for side in fk.SIDES}}))


if __name__ == "__main__":
    main()
