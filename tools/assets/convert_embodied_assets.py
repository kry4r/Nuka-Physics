#!/usr/bin/env python3
"""Generate Nuka-compatible G1 dance and Panda pick/place MJCF scenes.

Downloaded source assets and generated files remain outside git under
``.nuka-assets``. The generated XML references the pinned source mesh trees.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET


REPO = Path(__file__).resolve().parents[2]
ASSETS = REPO / ".nuka-assets"
G1_SOURCE = ASSETS / "src/g1-moves/mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
PANDA_SOURCE = ASSETS / "src/mujoco_menagerie/franka_emika_panda/panda.xml"
G1_OUTPUT = ASSETS / "generated/g1/g1_dance_stage.xml"
PANDA_OUTPUT = ASSETS / "generated/panda/panda_pick_place.xml"


def _floats(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(float(value) for value in text.split()) if text else default


def _fmt(values: tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{value:.9g}" for value in values)


def _quat_mul(a: tuple[float, ...], b: tuple[float, ...]) -> tuple[float, ...]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_rotate(q: tuple[float, ...], v: tuple[float, ...]) -> tuple[float, ...]:
    pure = (0.0, *v)
    conj = (q[0], -q[1], -q[2], -q[3])
    return _quat_mul(_quat_mul(q, pure), conj)[1:]


def _compose_pose(
    parent_pos: tuple[float, ...],
    parent_quat: tuple[float, ...],
    child_pos: tuple[float, ...],
    child_quat: tuple[float, ...],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    rotated = _quat_rotate(parent_quat, child_pos)
    return (
        tuple(parent_pos[i] + rotated[i] for i in range(3)),
        _quat_mul(parent_quat, child_quat),
    )


def _rotation_matrix(q: tuple[float, ...]) -> list[list[float]]:
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def _transpose(a: list[list[float]]) -> list[list[float]]:
    return [[a[j][i] for j in range(3)] for i in range(3)]


def _add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[a[i][j] + b[i][j] for j in range(3)] for i in range(3)]


def _parallel_axis(mass: float, d: tuple[float, ...]) -> list[list[float]]:
    x, y, z = d
    r2 = x * x + y * y + z * z
    return [
        [mass * (r2 - x * x), -mass * x * y, -mass * x * z],
        [-mass * y * x, mass * (r2 - y * y), -mass * y * z],
        [-mass * z * x, -mass * z * y, mass * (r2 - z * z)],
    ]


def _merge_hand_inertia(link7: ET.Element, hand: ET.Element) -> None:
    """Combine link7 and hand physical inertia in the link7 body frame."""
    link_inertial = link7.find("inertial")
    hand_inertial = hand.find("inertial")
    if link_inertial is None or hand_inertial is None:
        raise ValueError("Panda link7/hand inertial records are required")

    m1 = float(link_inertial.attrib["mass"])
    m2 = float(hand_inertial.attrib["mass"])
    p1 = _floats(link_inertial.get("pos"), (0.0, 0.0, 0.0))
    hand_pos = _floats(hand.get("pos"), (0.0, 0.0, 0.0))
    hand_quat = _floats(hand.get("quat"), (1.0, 0.0, 0.0, 0.0))
    p2_local = _floats(hand_inertial.get("pos"), (0.0, 0.0, 0.0))
    p2, _ = _compose_pose(hand_pos, hand_quat, p2_local, (1.0, 0.0, 0.0, 0.0))
    total = m1 + m2
    center = tuple((m1 * p1[i] + m2 * p2[i]) / total for i in range(3))

    full1 = _floats(link_inertial.get("fullinertia"), ())
    diag2 = _floats(hand_inertial.get("diaginertia"), ())
    if len(full1) != 6 or len(diag2) != 3:
        raise ValueError("unexpected Panda link7/hand inertia encoding")
    inertia1 = [
        [full1[0], full1[3], full1[4]],
        [full1[3], full1[1], full1[5]],
        [full1[4], full1[5], full1[2]],
    ]
    rotation = _rotation_matrix(hand_quat)
    inertia2 = _matmul(_matmul(rotation, [
        [diag2[0], 0.0, 0.0],
        [0.0, diag2[1], 0.0],
        [0.0, 0.0, diag2[2]],
    ]), _transpose(rotation))
    d1 = tuple(p1[i] - center[i] for i in range(3))
    d2 = tuple(p2[i] - center[i] for i in range(3))
    combined = _add(_add(inertia1, _parallel_axis(m1, d1)),
                    _add(inertia2, _parallel_axis(m2, d2)))

    link_inertial.set("mass", f"{total:.9g}")
    link_inertial.set("pos", _fmt(center))
    link_inertial.set("fullinertia", _fmt((
        combined[0][0], combined[1][1], combined[2][2],
        combined[0][1], combined[0][2], combined[1][2],
    )))
    link_inertial.attrib.pop("quat", None)
    link_inertial.attrib.pop("diaginertia", None)


def _body(name: str, pos: str = "0 0 0") -> ET.Element:
    return ET.Element("body", {"name": name, "pos": pos})


def _geom(parent: ET.Element, **attrs: str) -> ET.Element:
    element = ET.SubElement(parent, "geom", attrs)
    return element


def _add_studio(worldbody: ET.Element, task: str) -> None:
    studio = _body(f"{task}_studio")
    worldbody.append(studio)
    if task == "g1":
        _geom(studio, name="dance_floor", type="box", size="3.2 2.5 0.04",
              pos="0 0 -0.045", rgba="0.12 0.15 0.20 1", friction="1.2")
        _geom(studio, name="stage_inlay", type="box", size="1.15 1.15 0.006",
              pos="0 0 0.002", rgba="0.22 0.27 0.34 1", contype="0", conaffinity="0")
        _geom(studio, name="back_wall", type="box", size="0.08 2.6 1.7",
              pos="-1.45 0 1.65", rgba="0.10 0.13 0.18 1", contype="0", conaffinity="0")
        for y, color in ((-1.55, "0.05 0.55 0.95 1"), (0.0, "0.95 0.2 0.25 1"),
                         (1.55, "0.1 0.8 0.55 1")):
            _geom(studio, type="box", size="0.012 0.42 1.35", pos=f"-1.355 {y} 1.45",
                  rgba=color, contype="0", conaffinity="0")
    else:
        _geom(studio, name="floor", type="box", size="2.4 2.2 0.05",
              pos="0.35 0 -0.055", rgba="0.06 0.075 0.085 1", friction="1.1")
        workbench = _body("workbench_collision", "0.45 0 0.40")
        worldbody.append(workbench)
        _geom(workbench, name="workbench", type="box", size="0.75 0.62 0.04",
              rgba="0.38 0.26 0.16 1", friction="0.9", solref="0.04 1")
        for x in (-0.20, 1.10):
            for y in (-0.48, 0.48):
                _geom(studio, type="box", size="0.045 0.045 0.20",
                      pos=f"{x} {y} 0.2", rgba="0.10 0.12 0.14 1")
        _geom(studio, name="backdrop", type="box", size="0.06 1.7 1.15",
              pos="1.45 0 1.1", rgba="0.10 0.14 0.17 1", contype="0", conaffinity="0")
        target_bin = _body("blue_bin", "0.62 -0.28 0.455")
        worldbody.append(target_bin)
        _geom(target_bin, name="bin_base", type="box", size="0.12 0.15 0.015",
              rgba="0.04 0.30 0.72 1", friction="1.0")
        for dx, dy, sx, sy in ((-0.11, 0.0, 0.012, 0.15), (0.11, 0.0, 0.012, 0.15),
                               (0.0, -0.14, 0.12, 0.012), (0.0, 0.14, 0.12, 0.012)):
            _geom(target_bin, type="box", size=f"{sx} {sy} 0.055",
                  pos=f"{dx} {dy} 0.06", rgba="0.04 0.30 0.72 1")
        for y, color in ((-1.25, "0.12 0.65 0.85 1"), (0.0, "0.92 0.35 0.12 1"),
                         (1.25, "0.30 0.75 0.38 1")):
            _geom(studio, type="box", size="0.015 0.3 0.65", pos=f"1.38 {y} 1.12",
                  rgba=color, contype="0", conaffinity="0")


def _flatten_g1_defaults(root: ET.Element) -> None:
    defaults = root.find("default")
    if defaults is not None:
        root.remove(defaults)

    # Nuka uses the two thin sole boxes below for contact. The source proxy
    # capsules were already disabled for physics; keeping them as contype=0 would
    # make the render facade interpret them as visual primitives and cover the STL.
    proxy_geoms = {
        geom for geom in root.findall(".//geom")
        if geom.get("class") in ("collision", "foot_capsule")
    }
    for parent in root.iter():
        for child in list(parent):
            if child in proxy_geoms:
                parent.remove(child)

    geom_defaults = {
        "visual": {
            "type": "mesh", "density": "0", "material": "silver",
            "contype": "0", "conaffinity": "0", "group": "2",
        },
    }
    for geom in root.findall(".//geom"):
        class_name = geom.attrib.pop("class", None)
        if class_name in geom_defaults:
            for key, value in geom_defaults[class_name].items():
                geom.attrib.setdefault(key, value)
    for side in ("left", "right"):
        foot = root.find(f".//body[@name='{side}_ankle_roll_link']")
        if foot is None:
            raise ValueError(f"missing {side} ankle roll body")
        ET.SubElement(foot, "geom", {
            "name": f"{side}_foot_nuka_collision",
            "type": "box",
            "size": "0.095 0.04 0.012",
            "pos": "0.04 0 -0.025",
            "contype": "1",
            "conaffinity": "1",
            "friction": "1.2",
            "rgba": "0.2 0.6 0.2 0.18",
        })
    for body in root.findall(".//body"):
        body.attrib.pop("childclass", None)
        for freejoint in list(body.findall("freejoint")):
            index = list(body).index(freejoint)
            replacement = ET.Element("joint", {
                "name": freejoint.get("name", "floating_base_joint"),
                "type": "free", "limited": "false",
            })
            body.remove(freejoint)
            body.insert(index, replacement)
    for camera in root.findall(".//camera"):
        for parent in root.iter():
            if camera in list(parent):
                parent.remove(camera)
                break
    for tag in ("sensor", "contact"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)


def convert_g1(source: Path = G1_SOURCE, output: Path = G1_OUTPUT) -> dict:
    tree = ET.parse(source)
    root = tree.getroot()
    _flatten_g1_defaults(root)
    for material in root.findall("./asset/material"):
        if material.get("name") == "silver":
            material.set("rgba", "0.72 0.78 0.86 1")
            material.set("metallic", "0.55")
            material.set("roughness", "0.28")
        elif material.get("name") == "black":
            material.set("rgba", "0.055 0.075 0.11 1")
            material.set("metallic", "0.2")
            material.set("roughness", "0.42")
    old_environment = root.find("nuka_environment")
    if old_environment is not None:
        root.remove(old_environment)
    root.append(ET.Element("nuka_environment", {
        "use_scene_materials": "true",
        "exposure_ev": "1.0",
        "grade": "0.12",
    }))
    compiler = root.find("compiler")
    worldbody = root.find("worldbody")
    if compiler is None or worldbody is None:
        raise ValueError("G1 MJCF compiler/worldbody is required")
    compiler.set("meshdir", "../../src/g1-moves/mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/assets")
    for joint in worldbody.findall(".//joint"):
        name = joint.get("name", "")
        if any(part in name for part in ("elbow", "shoulder", "wrist_roll")):
            armature = 0.003609725
        elif any(part in name for part in ("hip_pitch", "hip_yaw")) or name == "waist_yaw_joint":
            armature = 0.01017752004132231
        elif "hip_roll" in name or "knee" in name:
            armature = 0.025101925
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            armature = 0.00425
        else:
            armature = 0.00721945
        joint.set("armature", f"{armature:.12g}")
    _add_studio(worldbody, "g1")
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return {
        "output": str(output.relative_to(REPO)),
        "source": str(source.relative_to(REPO)),
        "removed": [
            "default classes (expanded)", "source proxy collision geoms",
            "sensor", "camera", "contact exclusions",
        ],
        "collision_mode": "one thin box per foot for Nuka; source proxy capsules removed from physics and rendering",
    }


def convert_panda(source: Path = PANDA_SOURCE, output: Path = PANDA_OUTPUT) -> dict:
    tree = ET.parse(source)
    root = tree.getroot()
    old_environment = root.find("nuka_environment")
    if old_environment is not None:
        root.remove(old_environment)
    root.append(ET.Element("nuka_environment", {
        "use_scene_materials": "true",
        "exposure_ev": "0.75",
        "grade": "0.10",
    }))
    compiler = root.find("compiler")
    worldbody = root.find("worldbody")
    if compiler is None or worldbody is None:
        raise ValueError("Panda MJCF compiler/worldbody is required")
    compiler.set("meshdir", "../../src/mujoco_menagerie/franka_emika_panda/assets")
    link0 = worldbody.find("body[@name='link0']")
    link7 = worldbody.find(".//body[@name='link7']")
    hand = worldbody.find(".//body[@name='hand']")
    if link0 is None or link7 is None or hand is None:
        raise ValueError("Panda link0/link7/hand hierarchy changed")

    link0.set("pos", "0 0 0.44")
    panda_home = (
        -0.04080849, 0.34862798, -0.04117866, -2.36771464,
        0.03044342, 2.70693517, 0.67838210, 0.04, 0.04,
    )
    for joint_name, initial_position in zip(
        (f"joint{i}" for i in range(1, 8)), panda_home[:7], strict=True
    ):
        joint = worldbody.find(f".//joint[@name='{joint_name}']")
        if joint is None:
            raise ValueError(f"Panda joint missing: {joint_name}")
        joint.set("nuka:initial_position", f"{initial_position:.9g}")
    for joint_name, initial_position in zip(
        ("finger_joint1", "finger_joint2"), panda_home[7:], strict=True
    ):
        joint = worldbody.find(f".//joint[@name='{joint_name}']")
        if joint is None:
            raise ValueError(f"Panda finger joint missing: {joint_name}")
        joint.set("nuka:initial_position", f"{initial_position:.9g}")
    for body in worldbody.findall(".//body"):
        for geom in list(body.findall("geom")):
            geom_class = geom.get("class", "")
            if geom_class == "collision" or geom_class.startswith("fingertip_pad_collision"):
                body.remove(geom)
    root_inertial = link0.find("inertial")
    if root_inertial is None:
        raise ValueError("Panda link0 inertial is required")
    root_inertial.set("mass", "0")
    _merge_hand_inertia(link7, hand)

    hand_pos = _floats(hand.get("pos"), (0.0, 0.0, 0.0))
    hand_quat = _floats(hand.get("quat"), (1.0, 0.0, 0.0, 0.0))
    for child in list(hand):
        if child.tag not in ("geom", "body"):
            continue
        moved = copy.deepcopy(child)
        child_pos = _floats(moved.get("pos"), (0.0, 0.0, 0.0))
        child_quat = _floats(moved.get("quat"), (1.0, 0.0, 0.0, 0.0))
        pos, quat = _compose_pose(hand_pos, hand_quat, child_pos, child_quat)
        moved.set("pos", _fmt(pos))
        moved.set("quat", _fmt(quat))
        link7.append(moved)
    link7.remove(hand)

    for finger_name in ("left_finger", "right_finger"):
        finger = link7.find(f"body[@name='{finger_name}']")
        if finger is None:
            raise ValueError(f"Panda finger body missing after reparent: {finger_name}")
        _geom(
            finger, name=f"{finger_name}_pad", type="box",
            size="0.0085 0.004 0.012", pos="0 0.0055 0.0445",
            friction="0.9", solref="0.04 1",
        )

    for tag in ("tendon", "equality", "actuator", "keyframe"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    for light in list(worldbody.findall("light")):
        worldbody.remove(light)
    _add_studio(worldbody, "panda")
    cube = _body("red_cube", "0.50 0.0 0.465")
    ET.SubElement(cube, "inertial", {"mass": "0.12", "pos": "0 0 0",
                                      "diaginertia": "0.00005 0.00005 0.00005"})
    _geom(cube, name="red_cube_geom", type="box", size="0.025 0.025 0.025",
          rgba="0.90 0.06 0.04 1", friction="0.9", solref="0.04 1")
    worldbody.append(cube)

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return {
        "output": str(output.relative_to(REPO)),
        "source": str(source.relative_to(REPO)),
        "removed": [
            "arm/hand collision mesh proxies", "tendon", "equality",
            "actuator", "keyframe",
        ],
        "action_mapping": {"arm": list(range(1, 8)), "finger_joint1": 8, "finger_joint2": 9},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", choices=("all", "g1", "panda"), default="all")
    args = parser.parse_args()
    result = {}
    if args.asset in ("all", "g1"):
        result["g1"] = convert_g1()
    if args.asset in ("all", "panda"):
        result["panda"] = convert_panda()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
