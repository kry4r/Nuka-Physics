#!/usr/bin/env python3
"""Generate a Nuka-native LIBERO object-suite orange-juice scene."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from convert_embodied_assets import PANDA_SOURCE, _body, _compose_pose, _floats, _fmt, _geom, _merge_hand_inertia

OUTPUT = REPO / ".nuka-assets/generated/libero/libero_object_orange_juice.xml"
COOKED = OUTPUT.with_suffix(".nks")
ASSETS = REPO / ".nuka_cache/LIBERO/libero/libero/assets"
ORANGE_XML = ASSETS / "stable_hope_objects/orange_juice/orange_juice.xml"
BASKET_XML = ASSETS / "stable_scanned_objects/basket/basket.xml"
TASK_BDDL = REPO / ".nuka_cache/LIBERO/libero/libero/bddl_files/libero_object/pick_up_the_orange_juice_and_place_it_in_the_basket.bddl"

# First official LIBERO object-suite episode state, converted to Nuka's 9-DOF
# convention. The source state uses a negative right-finger coordinate.
PANDA_HOME = (0.0, -0.161037389, 0.0, -2.44459747, 0.0, 2.22675220,
              0.7853981634, 0.02, 0.02)
ORANGE_POS = (0.05, -0.1, 0.035)
ORANGE_QUAT = "0.5 0.5 0.5 0.5"
BASKET_POS = (-0.00409625, 0.25071288, 0.035)
BASKET_QUAT = "0.70710678 0 0 0.70710678"


def _add_material(asset: ET.Element, name: str, rgba: str, roughness: str, texture: str | None = None) -> None:
    attrs = {"name": name, "rgba": rgba, "roughness": roughness, "metallic": "0"}
    if texture:
        attrs["texture"] = texture
    ET.SubElement(asset, "material", attrs)


def _add_mesh(asset: ET.Element, name: str, path: Path, scale: str) -> None:
    ET.SubElement(asset, "mesh", {"name": name, "file": f"../../../{path.relative_to(REPO).as_posix()}", "scale": scale})


def _add_visual(body: ET.Element, name: str, mesh: str, material: str) -> None:
    _geom(body, name=name, type="mesh", mesh=mesh, material=material, contype="0", conaffinity="0", group="2", **{"nuka:decompose": "skip"})


def _copy_collision_boxes(body: ET.Element, source_xml: Path, prefix: str) -> None:
    source = ET.parse(source_xml).find("./worldbody/body/body[@name='object']")
    if source is None:
        raise ValueError(f"missing object body in {source_xml}")
    for index, geom in enumerate(source.findall("geom")):
        if geom.get("type") != "box":
            continue
        attrs = dict(geom.attrib)
        attrs["name"] = f"{prefix}_collision_{index}"
        attrs["rgba"] = "0.8 0.8 0.8 0.03"
        attrs["solref"] = "0.025 1"
        attrs["solimp"] = "0.95 0.99 0.001"
        _geom(body, **attrs)


def _rigid_body(name: str, pos: tuple[float, float, float], quat: str, mass: str | None) -> ET.Element:
    body = _body(name, _fmt(pos))
    body.set("quat", quat)
    if mass is not None:
        ET.SubElement(body, "inertial", {"mass": mass, "pos": "0 0 0", "diaginertia": "0.00008 0.00008 0.00012"})
    return body


def generate(output: Path = OUTPUT) -> dict:
    if not PANDA_SOURCE.is_file() or not ORANGE_XML.is_file() or not BASKET_XML.is_file() or not TASK_BDDL.is_file():
        raise FileNotFoundError("LIBERO object-suite source asset is missing")
    tree = ET.parse(PANDA_SOURCE)
    root = tree.getroot()
    root.set("model", "nuka_libero_object_orange_juice")
    compiler = root.find("compiler")
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if compiler is None or asset is None or worldbody is None:
        raise ValueError("Panda MJCF structure changed")
    compiler.set("meshdir", ".")
    for mesh in asset.findall("mesh"):
        if mesh.get("file"):
            mesh.set("file", "../../src/mujoco_menagerie/franka_emika_panda/assets/" + mesh.get("file"))

    for name, path, material, rgba, scale, texture_file in (
        ("orange_juice_visual", ASSETS / "stable_hope_objects/orange_juice/textured.obj", "orange_juice", "1 1 1 1", "0.0075 0.0075 0.0075", ASSETS / "stable_hope_objects/orange_juice/texture_map.png"),
        ("basket_visual", ASSETS / "stable_scanned_objects/basket/basket.obj", "basket", "0.65 0.31 0.08 1", "1 1 1", ASSETS / "stable_scanned_objects/basket/texture.png"),
    ):
        texture_name = f"{material}_texture"
        ET.SubElement(asset, "texture", {"name": texture_name, "type": "2d", "file": f"../../../{texture_file.relative_to(REPO).as_posix()}"})
        _add_material(asset, material, rgba, "0.42", texture_name)
        _add_mesh(asset, name, path, scale)
    _add_material(asset, "floor_material", "0.63 0.64 0.61 1", "0.55")
    _add_material(asset, "distractor_material", "0.45 0.48 0.50 1", "0.5")

    for tag in ("tendon", "equality", "actuator", "keyframe", "contact"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    for light in list(worldbody.findall("light")):
        worldbody.remove(light)
    link0 = worldbody.find("body[@name='link0']")
    if link0 is None:
        raise ValueError("Panda link0 missing")
    camera_body = ET.Element("body", {"name": "libero_agent_camera", "pos": "0.89657737 0 0.65", "quat": "0.61821669 0.34323075 0.34323144 0.61821771"})
    worldbody.insert(list(worldbody).index(link0), camera_body)

    hand = worldbody.find(".//body[@name='hand']")
    link7 = worldbody.find(".//body[@name='link7']")
    link0.set("pos", "-0.6 0 0")
    for name, value in zip(tuple(f"joint{i}" for i in range(1, 8)) + ("finger_joint1", "finger_joint2"), PANDA_HOME, strict=True):
        joint = worldbody.find(f".//joint[@name='{name}']")
        if joint is None:
            raise ValueError(f"missing {name}")
        joint.set("nuka:initial_position", f"{value:.10g}")
    worldbody.find(".//joint[@name='finger_joint2']").set("range", "0 0.04")
    for body in worldbody.findall(".//body"):
        for geom in list(body.findall("geom")):
            if geom.get("class", "") == "collision" or geom.get("class", "").startswith("fingertip_pad_collision"):
                body.remove(geom)
    link0.find("inertial").set("mass", "0")
    _merge_hand_inertia(link7, hand)
    hand_pos = _floats(hand.get("pos"), (0, 0, 0))
    hand_quat = _floats(hand.get("quat"), (1, 0, 0, 0))
    for child in list(hand):
        if child.tag not in ("geom", "body"):
            continue
        moved = copy.deepcopy(child)
        pos, quat = _compose_pose(hand_pos, hand_quat, _floats(moved.get("pos"), (0, 0, 0)), _floats(moved.get("quat"), (1, 0, 0, 0)))
        moved.set("pos", _fmt(pos)); moved.set("quat", _fmt(quat)); link7.append(moved)
    link7.remove(hand)
    for finger_name in ("left_finger", "right_finger"):
        finger = link7.find(f"body[@name='{finger_name}']")
        for index, (size, pos) in enumerate((("0.0085 0.004 0.0085", "0 0.0055 0.0445"), ("0.003 0.002 0.003", "0.0055 0.002 0.05"), ("0.003 0.002 0.003", "-0.0055 0.002 0.05"), ("0.003 0.002 0.0035", "0.0055 0.002 0.0395"), ("0.003 0.002 0.0035", "-0.0055 0.002 0.0395"))):
            _geom(finger, name=f"{finger_name}_libero_pad{index}", type="box", size=size, pos=pos, friction="0.95 0.3 0.1", solref="0.001 1", solimp="0.998 0.998 0.001", priority="1")

    room = _body("libero_floor_room")
    worldbody.append(room)
    _geom(room, name="libero_floor", type="box", size="1.8 1.8 0.04", pos="0 0 -0.04", rgba="0.63 0.64 0.61 1", friction="1.0", material="floor_material")
    _geom(room, name="libero_back_wall", type="box", size="0.04 1.8 0.8", pos="0.95 0 0.8", rgba="0.66 0.65 0.59 1", contype="0", conaffinity="0")

    orange = _rigid_body("orange_juice_1_main", ORANGE_POS, ORANGE_QUAT, "0.08")
    _add_visual(orange, "orange_juice_1_visual", "orange_juice_visual", "orange_juice")
    _geom(
        orange,
        name="orange_juice_1_fallback_visual",
        type="box",
        size="0.026 0.026 0.055",
        rgba="0.95 0.35 0.04 1",
        contype="0",
        conaffinity="0",
        group="2",
    )
    _copy_collision_boxes(orange, ORANGE_XML, "orange_juice_1")
    worldbody.append(orange)
    basket = _rigid_body("basket_1_main", BASKET_POS, BASKET_QUAT, None)
    _add_visual(basket, "basket_1_visual", "basket_visual", "basket")
    _copy_collision_boxes(basket, BASKET_XML, "basket_1")
    worldbody.append(basket)

    # Preserve the clutter distribution from the official episode without making
    # unrelated assets a second conversion project for this diagnostic rollout.
    distractors = (("butter_1_main", -0.11817, -0.24280, 0.030, 0.035, 0.030, "0.86 0.68 0.18 1"), ("chocolate_pudding_1_main", -0.15, 0.06, 0.035, 0.035, 0.045, "0.27 0.10 0.06 1"), ("bbq_sauce_1_main", 0.09124, -0.19568, 0.035, 0.025, 0.025, "0.76 0.10 0.06 1"), ("ketchup_1_main", 0.15, 0.03, 0.035, 0.025, 0.025, "0.88 0.05 0.04 1"), ("salad_dressing_1_main", -0.2, -0.08, 0.035, 0.03, 0.03, "0.75 0.52 0.10 1"))
    for name, x, y, z, sx, sy, rgba in distractors:
        body = _rigid_body(name, (x, y, z), "1 0 0 0", "0.08")
        _geom(body, name=f"{name}_collision", type="box", size=f"{sx} {sy} 0.03", rgba=rgba, friction="0.8")
        _geom(body, name=f"{name}_visual", type="box", size=f"{sx} {sy} 0.03", rgba=rgba, contype="0", conaffinity="0", group="2")
        worldbody.append(body)
    for attrs in (("libero_key", "0.8 -0.7 2.0", "-0.2 0.2 -1"), ("libero_fill", "-0.4 0.5 1.6", "0.2 -0.1 -1")):
        ET.SubElement(worldbody, "light", {"name": attrs[0], "pos": attrs[1], "dir": attrs[2]})
    root.append(ET.Element("nuka_environment", {"use_scene_materials": "true", "exposure_ev": "0.25", "grade": "0.05", "specular_env": "true"}))
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return {"output": str(output.relative_to(REPO)), "task": "pick up the orange juice and place it in the basket", "suite": "libero_object", "panda_home": list(PANDA_HOME), "orange_position": list(ORANGE_POS), "basket_position": list(BASKET_POS)}


def cook(xml_path: Path, output_path: Path = COOKED, cooker: str | None = None) -> None:
    if sys.platform == "win32":
        cooker_path = cooker or "/root/nuka-build/src/nuka_cook_scene"
        command = [
            "wsl",
            "-d",
            "Ubuntu-24.04",
            "--",
            cooker_path,
            "/mnt/c/Softwares/code/Nuka-Physics/" + xml_path.relative_to(REPO).as_posix(),
            "/mnt/c/Softwares/code/Nuka-Physics/" + output_path.relative_to(REPO).as_posix(),
        ]
    else:
        cooker_path = cooker or "/root/nuka-build/src/nuka_cook_scene"
        command = [cooker_path, str(xml_path), str(output_path)]
    subprocess.run(command, check=True, cwd=REPO)


if __name__ == "__main__":
    result = generate()
    if "--cook" in sys.argv:
        cook(OUTPUT)
        result["cooked"] = str(COOKED.relative_to(REPO))
    print(json.dumps(result, indent=2))
