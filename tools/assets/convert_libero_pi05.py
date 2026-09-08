#!/usr/bin/env python3
"""Generate the Nuka-native LIBERO Spatial black-bowl scene."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))

from convert_embodied_assets import (  # noqa: E402
    PANDA_SOURCE,
    _body,
    _compose_pose,
    _floats,
    _fmt,
    _geom,
    _merge_hand_inertia,
)

OUTPUT = REPO / ".nuka-assets/generated/libero/libero_spatial_black_bowl.xml"
COOKED = OUTPUT.with_suffix(".nks")
LIBERO_ASSETS = REPO / ".nuka_cache/LIBERO/libero/libero/assets"
LIBERO_BOWL_XML = LIBERO_ASSETS / "stable_scanned_objects/akita_black_bowl/akita_black_bowl.xml"
LIBERO_PLATE_XML = LIBERO_ASSETS / "stable_scanned_objects/plate/plate.xml"
TASK_BDDL = (
    REPO
    / ".nuka_cache/LIBERO/libero/libero/bddl_files/libero_spatial"
    / "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate.bddl"
)

# Official LIBERO task-2 episode-0 initialization state.
PANDA_HOME = (
    0.0067131381,
    -0.191884762,
    -0.0099483010,
    -2.43256078,
    -0.0399100748,
    2.19352400,
    0.801371176,
    0.020833,
    0.020833,
)

OBJECT_POSES = {
    "target_bowl": (-0.07500000, 0.01499801, 0.89821996),
    "distractor_bowl": (-0.00235172, 0.30688924, 0.89821996),
    "cookies": (0.07668304, 0.03753320, 0.90920953),
    "ramekin": (-0.20139241, 0.18698477, 0.89917628),
    "plate": (0.07159546, 0.20039269, 0.90233128),
    "cabinet": (0.024782, -0.27050885, 0.905),
    "stove": (-0.404893, -0.14409259, 0.905),
}
OBJECT_QUAT = "0.70710678 -0.00002706 0.00000441 0.70710678"

# Nuka's pair-driven collision path resolves per-shape local transforms: a body
# with several colliding shapes (or one carrying an offset) cooks them into
# appended collidable proxy rows posed as owner_pose o local. Bodies are therefore
# authored at their real object pose with shapes offset in the body frame, matching
# the source assets.
# LIBERO reset places free scanned objects at this body height; gravity settles
# them onto the tabletop during subsequent simulation steps.
LIBERO_OBJECT_INITIAL_Z = 0.970
TABLE_TOP_Z = 0.900

# akita_black_bowl.obj at scale 0.7 measured in its own frame: base plane at
# z = 0, rim at z = 0.05265, foot radius 0.025 flaring to a rim outer radius of
# 0.0562 with a ~4mm wall. The Panda gripper (max opening 0.08m) cannot span the
# 0.112m bowl, so it must pinch the rim WALL with one finger inside the bowl --
# which requires the shell to be hollow, not a filled box.
BOWL_MESH_HEIGHT = 0.0526456


def _add_bowl_shell(body: ET.Element, prefix: str) -> None:
    """Copy LIBERO's compiled collision-box layout from the official MJCF."""
    tree = ET.parse(LIBERO_BOWL_XML)
    source_body = tree.find("./worldbody/body/body[@name='object']")
    if source_body is None:
        raise ValueError(f"Official bowl body missing in {LIBERO_BOWL_XML}")
    for index, source in enumerate(source_body.findall("geom")):
        if source.get("type") != "box":
            continue
        attrs = {key: value for key, value in source.attrib.items()}
        attrs["name"] = f"{prefix}_official_collision_{index}"
        attrs["material"] = "libero_black"
        attrs["rgba"] = "0.02 0.025 0.03 0.04"
        # The official MJCF uses MuJoCo's very stiff 1ms contacts. Nuka's
        # projected row solver needs the same geometry with a slightly softer
        # reference time to avoid injecting tangential energy across 40 boxes.
        attrs["solref"] = "0.025 1"
        attrs["solimp"] = "0.95 0.99 0.001"
        _geom(body, **attrs)


def _add_official_collision_boxes(
    body: ET.Element, source_xml: Path, prefix: str
) -> None:
    """Copy the scanned object's authored box collision layout."""
    tree = ET.parse(source_xml)
    source_body = tree.find("./worldbody/body/body[@name='object']")
    if source_body is None:
        raise ValueError(f"Official object body missing in {source_xml}")
    for index, source in enumerate(source_body.findall("geom")):
        if source.get("type") != "box":
            continue
        attrs = {key: value for key, value in source.attrib.items()}
        attrs["name"] = f"{prefix}_official_collision_{index}"
        attrs["rgba"] = "0.8 0.8 0.8 0.04"
        attrs["solref"] = "0.025 1"
        attrs["solimp"] = "0.95 0.99 0.001"
        _geom(body, **attrs)


def _resting_body_z(half_z: float) -> float:
    """Body origin height that rests a body-centred proxy on the tabletop."""
    return TABLE_TOP_Z + half_z


def _visual_offset_z(name: str, body_z: float) -> float:
    """Local z that keeps a visual asset at its authored world height."""
    return OBJECT_POSES[name][2] - body_z


def _resting_pose_text(name: str, half_z: float) -> tuple[str, float]:
    x, y, _ = OBJECT_POSES[name]
    body_z = _resting_body_z(half_z)
    return _fmt((x, y, body_z)), body_z


def _add_material(
    asset: ET.Element,
    name: str,
    rgba: str,
    roughness: str,
    *,
    texture: str | None = None,
) -> None:
    attrs = {"name": name, "rgba": rgba, "roughness": roughness, "metallic": "0.0"}
    if texture is not None:
        attrs["texture"] = texture
    ET.SubElement(asset, "material", attrs)


def _add_mesh(asset: ET.Element, name: str, path: Path, scale: str) -> None:
    relative = path.relative_to(REPO).as_posix()
    ET.SubElement(
        asset,
        "mesh",
        {"name": name, "file": f"../../../{relative}", "scale": scale},
    )


def _add_visual_mesh(
    body: ET.Element, name: str, mesh: str, material: str, *, pos: str = "0 0 0"
) -> None:
    _geom(
        body,
        name=name,
        type="mesh",
        mesh=mesh,
        material=material,
        pos=pos,
        contype="0",
        conaffinity="0",
        group="2",
        **{"nuka:decompose": "skip"},
    )


def _add_visual_twin(parent: ET.Element, name: str, **attrs: str) -> None:
    """Mirror a colliding primitive as a visual-only geom.

    Nuka renders MJCF geoms only when they are visual-only (contype and
    conaffinity 0), matching robosuite's table_collision/table_visual split.
    Colliding primitives stay untouched so physics is unchanged.
    """
    _geom(parent, name=name, contype="0", conaffinity="0", group="2", **attrs)


def _pose_text(name: str) -> str:
    return _fmt(OBJECT_POSES[name])


def generate(output: Path = OUTPUT) -> dict:
    if not PANDA_SOURCE.is_file():
        raise FileNotFoundError(PANDA_SOURCE)
    if not TASK_BDDL.is_file():
        raise FileNotFoundError(TASK_BDDL)

    tree = ET.parse(PANDA_SOURCE)
    root = tree.getroot()
    root.set("model", "nuka_libero_spatial_black_bowl")
    compiler = root.find("compiler")
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if compiler is None or asset is None or worldbody is None:
        raise ValueError("Panda MJCF compiler, asset, and worldbody are required")

    # Resolve every asset from the generated scene directory so Panda and LIBERO
    # meshes can coexist without copying third-party data into git.
    compiler.set("meshdir", ".")
    for mesh in asset.findall("mesh"):
        file_name = mesh.get("file")
        if file_name:
            mesh.set(
                "file",
                "../../src/mujoco_menagerie/franka_emika_panda/assets/" + file_name,
            )

    bowl_texture = "libero_black_bowl_texture"
    texture_path = LIBERO_ASSETS / "stable_scanned_objects/akita_black_bowl/texture.png"
    if not texture_path.is_file():
        raise FileNotFoundError(texture_path)
    ET.SubElement(
        asset,
        "texture",
        {
            "name": bowl_texture,
            "type": "2d",
            "file": f"../../../{texture_path.relative_to(REPO).as_posix()}",
        },
    )
    _add_material(
        asset,
        "libero_black",
        # Preserve the authored texture colors with a white material multiplier.
        "1 1 1 1",
        "0.24",
        texture=bowl_texture,
    )
    table_texture = "libero_table_texture"
    table_texture_path = LIBERO_ASSETS / "textures/martin_novak_wood_table.png"
    if not table_texture_path.is_file():
        raise FileNotFoundError(table_texture_path)
    ET.SubElement(
        asset,
        "texture",
        {
            "name": table_texture,
            "type": "cube",
            "file": f"../../../{table_texture_path.relative_to(REPO).as_posix()}",
        },
    )
    _add_material(
        asset,
        "libero_table",
        "1 1 1 1",
        "0.48",
        texture=table_texture,
    )
    plate_texture = "libero_plate_texture"
    plate_texture_path = LIBERO_ASSETS / "stable_scanned_objects/plate/texture.png"
    ramekin_texture = "libero_ramekin_texture"
    ramekin_texture_path = (
        LIBERO_ASSETS
        / "stable_scanned_objects/glazed_rim_porcelain_ramekin/texture.png"
    )
    cookies_texture = "libero_cookies_texture"
    cookies_texture_path = LIBERO_ASSETS / "stable_hope_objects/cookies/texture_map.png"
    for texture_name, texture_path in (
        (plate_texture, plate_texture_path),
        (ramekin_texture, ramekin_texture_path),
        (cookies_texture, cookies_texture_path),
    ):
        if not texture_path.is_file():
            raise FileNotFoundError(texture_path)
        ET.SubElement(
            asset,
            "texture",
            {
                "name": texture_name,
                "type": "2d",
                "file": f"../../../{texture_path.relative_to(REPO).as_posix()}",
            },
        )
    _add_material(
        asset,
        "libero_plate",
        "1 1 1 1",
        "0.32",
        texture=plate_texture,
    )
    _add_material(
        asset,
        "libero_ramekin",
        "1 1 1 1",
        "0.28",
        texture=ramekin_texture,
    )
    _add_material(
        asset,
        "libero_cookies",
        "1 1 1 1",
        "0.48",
        texture=cookies_texture,
    )
    _add_material(asset, "libero_metal", "0.16 0.17 0.18 1", "0.31")
    _add_mesh(
        asset,
        "libero_bowl_visual",
        LIBERO_ASSETS / "stable_scanned_objects/akita_black_bowl/akita_black_bowl.obj",
        "0.7 0.7 0.7",
    )
    _add_mesh(
        asset,
        "libero_plate_visual",
        LIBERO_ASSETS / "stable_scanned_objects/plate/model.obj",
        "0.5 0.5 0.5",
    )
    _add_mesh(
        asset,
        "libero_ramekin_visual",
        LIBERO_ASSETS
        / "stable_scanned_objects/glazed_rim_porcelain_ramekin/glazed_rim_porcelain_ramekin.obj",
        "1 1 1",
    )

    environment = root.find("nuka_environment")
    if environment is not None:
        root.remove(environment)
    root.append(
        ET.Element(
            "nuka_environment",
            {
                "use_scene_materials": "true",
                "exposure_ev": "0.45",
                "grade": "0.06",
                "sun_disc": "0.22",
                "specular_env": "true",
            },
        )
    )

    link0 = worldbody.find("body[@name='link0']")
    link7 = worldbody.find(".//body[@name='link7']")
    hand = worldbody.find(".//body[@name='hand']")
    if link0 is None or link7 is None or hand is None:
        raise ValueError("Panda link0/link7/hand hierarchy changed")
    link0.set("pos", "-0.66 0 0.912")

    for name, value in zip(
        tuple(f"joint{i}" for i in range(1, 8))
        + ("finger_joint1", "finger_joint2"),
        PANDA_HOME,
        strict=True,
    ):
        joint = worldbody.find(f".//joint[@name='{name}']")
        if joint is None:
            raise ValueError(f"Panda joint missing: {name}")
        joint.set("nuka:initial_position", f"{value:.10g}")

    right_finger_joint = worldbody.find(".//joint[@name='finger_joint2']")
    if right_finger_joint is None:
        raise ValueError("Panda right finger joint missing")
    # The menagerie Panda keeps both finger joints in the positive [0, 0.04]
    # coordinate convention. LIBERO's robosuite XML uses a negative right
    # joint because its right finger body has a different local frame; applying
    # that range to the menagerie body makes both fingers translate together.
    right_finger_joint.set("range", "0 0.04")

    # Robosuite finger joints use armature 1, damping 100, and dry friction 1 N.
    for finger_joint_name in ("finger_joint1", "finger_joint2"):
        joint = worldbody.find(f".//joint[@name='{finger_joint_name}']")
        if joint is None:
            raise ValueError(f"Panda finger joint missing: {finger_joint_name}")
        joint.set("armature", "1")
        joint.set("damping", "100")
        joint.set("frictionloss", "1")

    # Panda visual meshes remain unchanged; LIBERO's five collision pads are
    # added below with their source finger-local transforms.
    for body in worldbody.findall(".//body"):
        for geom in list(body.findall("geom")):
            geom_class = geom.get("class", "")
            if geom_class == "collision" or geom_class.startswith(
                "fingertip_pad_collision"
            ):
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
    pad_specs = (
        ("pad1", "0.0085 0.004 0.0085", "0 0.0055 0.0445"),
        ("pad2", "0.003 0.002 0.003", "0.0055 0.002 0.05"),
        ("pad3", "0.003 0.002 0.003", "-0.0055 0.002 0.05"),
        ("pad4", "0.003 0.002 0.0035", "0.0055 0.002 0.0395"),
        ("pad5", "0.003 0.002 0.0035", "-0.0055 0.002 0.0395"),
    )
    # Five collision pads per finger provide the gripper's contact surfaces.
    # Hand and finger meshes remain visual geometry.
    for finger_name in ("left_finger", "right_finger"):
        finger = link7.find(f"body[@name='{finger_name}']")
        if finger is None:
            raise ValueError(f"Panda finger missing after hand merge: {finger_name}")
        # Use the robosuite finger mass, center of mass, and principal inertia.
        finger_inertial = finger.find("inertial")
        if finger_inertial is None:
            raise ValueError(f"Panda finger inertial missing: {finger_name}")
        finger_inertial.set("mass", "0.1")
        finger_inertial.set("pos", "0 0 0.05")
        finger_inertial.set("diaginertia", "0.01 0.01 0.005")
        for pad_name, size, pos in pad_specs:
            _geom(
                finger,
                name=f"{finger_name}_libero_{pad_name}",
                type="box",
                size=size,
                pos=pos,
                # Robosuite pads use friction 2 and the (0.01, 0.5) contact reference.
                friction="2 0.05 0.0001",
                solref="0.01 0.5",
            )

    for tag in ("tendon", "equality", "actuator", "keyframe", "contact"):
        element = root.find(tag)
        if element is not None:
            root.remove(element)
    for light in list(worldbody.findall("light")):
        worldbody.remove(light)

    room = _body("libero_room")
    worldbody.append(room)
    _geom(
        room,
        name="libero_floor",
        type="box",
        size="1.35 1.45 0.04",
        pos="0 -0.05 -0.04",
        rgba="0.34 0.34 0.31 1",
        friction="1.0",
    )
    _geom(
        room,
        name="libero_back_wall",
        type="box",
        size="0.04 1.4 0.8",
        pos="0.72 0 0.8",
        rgba="0.66 0.65 0.59 1",
        contype="0",
        conaffinity="0",
    )
    _geom(
        room,
        name="libero_side_wall",
        type="box",
        size="1.35 0.04 0.8",
        pos="0 -0.82 0.8",
        rgba="0.59 0.61 0.59 1",
        contype="0",
        conaffinity="0",
    )

    # robosuite mounts the Panda on a rethink pedestal. Without it the arm base
    # floats above the floor, so mirror the official mount0 geometry relative to
    # robot0_base at (-0.66, 0, 0.912) plus the official 0.01 mount offset.
    mount = _body("mount0_base", "-0.66 0 0.922")
    worldbody.append(mount)
    for geom_name, geom_type, size, pos in (
        ("mount0_controller_box", "box", "0.11 0.2 0.265", "-0.325 0 -0.38"),
        ("mount0_pedestal_feet", "box", "0.385 0.35 0.155", "-0.1225 0 -0.758"),
        ("mount0_pedestal", "cylinder", "0.18 0.31", "-0.02 0 -0.29"),
    ):
        _geom(mount, name=f"{geom_name}_col", type=geom_type, size=size, pos=pos)
        _add_visual_twin(
            mount,
            f"{geom_name}_visual",
            type=geom_type,
            size=size,
            pos=pos,
            material="libero_metal",
        )
    _add_visual_twin(
        mount,
        "mount0_torso_visual",
        type="box",
        size="0.05 0.05 0.05",
        pos="0 0 -0.05",
        rgba="0.2 0.2 0.2 1",
    )

    table = _body("main_table", "0 0 0.875")
    worldbody.append(table)
    _geom(
        table,
        name="main_table_top",
        type="box",
        size="0.5 0.6 0.025",
        friction="0.95",
        solref="0.025 1",
    )
    _add_visual_twin(
        table,
        "main_table_top_visual",
        type="box",
        size="0.5 0.6 0.025",
        material="libero_table",
    )
    # Official table_arena legs: visual-only cylinders from floor to tabletop.
    for index, (x, y) in enumerate(
        ((0.4, 0.5), (-0.4, 0.5), (-0.4, -0.5), (0.4, -0.5)), start=1
    ):
        _add_visual_twin(
            table,
            f"main_table_leg{index}_visual",
            type="cylinder",
            size="0.025 0.4375",
            pos=f"{x} {y} -0.4375",
            rgba="0.24 0.13 0.07 1",
        )

    # Both bowls are authored at their real pose with the mesh base plane on the
    # tabletop; the hollow shell's shapes carry their own body-frame offsets.
    bowl_pose_x, bowl_pose_y, _ = OBJECT_POSES["target_bowl"]
    target_pose = _fmt((bowl_pose_x, bowl_pose_y, LIBERO_OBJECT_INITIAL_Z))
    target = _body("akita_black_bowl_1_main", target_pose)
    target.set("quat", OBJECT_QUAT)
    worldbody.append(target)
    ET.SubElement(
        target,
        "inertial",
        {
            "mass": "0.115",
            "pos": f"0 0 {0.5 * BOWL_MESH_HEIGHT:.9g}",
            "diaginertia": "0.00011 0.00011 0.00018",
        },
    )
    _add_visual_mesh(
        target,
        "akita_black_bowl_1_visual",
        "libero_bowl_visual",
        "libero_black",
    )
    _add_bowl_shell(target, "akita_black_bowl_1")

    distractor_x, distractor_y, _ = OBJECT_POSES["distractor_bowl"]
    distractor_pose = _fmt((distractor_x, distractor_y, LIBERO_OBJECT_INITIAL_Z))
    distractor = _body("akita_black_bowl_2_main", distractor_pose)
    distractor.set("quat", OBJECT_QUAT)
    worldbody.append(distractor)
    ET.SubElement(
        distractor,
        "inertial",
        {
            "mass": "0.115",
            "pos": f"0 0 {0.5 * BOWL_MESH_HEIGHT:.9g}",
            "diaginertia": "0.00011 0.00011 0.00018",
        },
    )
    _add_visual_mesh(
        distractor,
        "akita_black_bowl_2_visual",
        "libero_bowl_visual",
        "libero_black",
    )
    _add_bowl_shell(distractor, "akita_black_bowl_2")

    plate_pose = _fmt((OBJECT_POSES["plate"][0], OBJECT_POSES["plate"][1], LIBERO_OBJECT_INITIAL_Z))
    plate = _body("plate_1_main", plate_pose)
    plate.set("quat", OBJECT_QUAT)
    worldbody.append(plate)
    ET.SubElement(
        plate,
        "inertial",
        {
            "mass": "0.2",
            "pos": "0 0 0",
            "diaginertia": "0.00021 0.00021 0.00040",
        },
    )
    _add_visual_mesh(
        plate,
        "plate_1_visual",
        "libero_plate_visual",
        "libero_plate",
        pos="0 0 0",
    )
    _add_official_collision_boxes(plate, LIBERO_PLATE_XML, "plate_1")

    ramekin_half_z = 0.022
    ramekin_pose = _fmt((OBJECT_POSES["ramekin"][0], OBJECT_POSES["ramekin"][1], LIBERO_OBJECT_INITIAL_Z))
    ramekin_body_z = LIBERO_OBJECT_INITIAL_Z
    ramekin = _body("glazed_rim_porcelain_ramekin_1_main", ramekin_pose)
    ramekin.set("quat", OBJECT_QUAT)
    worldbody.append(ramekin)
    ET.SubElement(
        ramekin,
        "inertial",
        {
            "mass": "0.16",
            "pos": "0 0 0",
            "diaginertia": "0.00011 0.00011 0.00016",
        },
    )
    _add_visual_mesh(
        ramekin,
        "ramekin_1_visual",
        "libero_ramekin_visual",
        "libero_ramekin",
        pos="0 0 0",
    )
    _geom(
        ramekin,
        name="ramekin_1_collision",
        type="box",
        size=f"0.044 0.044 {ramekin_half_z:.9g}",
        rgba="0.84 0.86 0.89 0.03",
    )

    cookies_half_z = 0.012
    cookies_pose = _fmt((OBJECT_POSES["cookies"][0], OBJECT_POSES["cookies"][1], LIBERO_OBJECT_INITIAL_Z))
    cookies_body_z = LIBERO_OBJECT_INITIAL_Z
    cookies = _body("cookies_1_main", cookies_pose)
    cookies.set("quat", OBJECT_QUAT)
    worldbody.append(cookies)
    ET.SubElement(
        cookies,
        "inertial",
        {
            "mass": "0.25",
            "pos": "0 0 0",
            "diaginertia": "0.00020 0.00014 0.00028",
        },
    )
    _geom(
        cookies,
        name="cookies_box",
        type="box",
        size=f"0.031 0.042 {cookies_half_z:.9g}",
        friction="0.8",
    )
    _add_visual_twin(
        cookies,
        "cookies_box_visual",
        type="box",
        size=f"0.031 0.042 {cookies_half_z:.9g}",
        rgba="0.78 0.10 0.08 1",
        material="libero_cookies",
    )
    _add_visual_twin(
        cookies,
        "cookies_label",
        type="box",
        size="0.024 0.043 0.007",
        pos=f"0 0 {cookies_half_z + 0.001:.9g}",
        rgba="0.94 0.76 0.24 1",
    )

    cabinet_half_z = 0.145
    cabinet_pose, cabinet_body_z = _resting_pose_text("cabinet", cabinet_half_z)
    cabinet = _body("wooden_cabinet_1_main", cabinet_pose)
    cabinet.set("quat", "0.2231372 0 0 0.97478705")
    worldbody.append(cabinet)
    _geom(
        cabinet,
        name="cabinet_body",
        type="box",
        size=f"0.105 0.07 {cabinet_half_z:.9g}",
    )
    _add_visual_twin(
        cabinet,
        "cabinet_body_visual",
        type="box",
        size=f"0.105 0.07 {cabinet_half_z:.9g}",
        material="libero_wood",
    )
    # Drawer fronts and handles keep their authored world heights; the body
    # origin moved to the proxy centre so re-express them as local offsets.
    cabinet_authored_z = OBJECT_POSES["cabinet"][2]
    for z in (0.075, 0.15, 0.225):
        local_z = cabinet_authored_z + z - cabinet_body_z
        _add_visual_twin(
            cabinet,
            f"cabinet_drawer_{z}",
            type="box",
            size="0.098 0.071 0.035",
            pos=f"0 0 {local_z:.9g}",
            rgba="0.38 0.21 0.11 1",
        )
        _add_visual_twin(
            cabinet,
            f"cabinet_handle_{z}",
            type="box",
            size="0.035 0.008 0.006",
            pos=f"0 -0.077 {local_z:.9g}",
            rgba="0.12 0.12 0.11 1",
        )

    stove_half_z = 0.018
    stove_pose, stove_body_z = _resting_pose_text("stove", stove_half_z)
    stove = _body("flat_stove_1_main", stove_pose)
    worldbody.append(stove)
    _geom(
        stove,
        name="stove_base",
        type="box",
        size=f"0.095 0.105 {stove_half_z:.9g}",
    )
    _add_visual_twin(
        stove,
        "stove_base_visual",
        type="box",
        size=f"0.095 0.105 {stove_half_z:.9g}",
        material="libero_metal",
    )
    burner_local_z = OBJECT_POSES["stove"][2] + 0.040 - stove_body_z
    for index, (x, y) in enumerate(
        ((-0.045, -0.045), (-0.045, 0.045), (0.045, -0.045), (0.045, 0.045)), start=1
    ):
        _add_visual_twin(
            stove,
            f"stove_burner{index}",
            type="cylinder",
            size="0.029 0.003",
            pos=f"{x} {y} {burner_local_z:.9g}",
            rgba="0.055 0.06 0.065 1",
        )

    for attrs in (
        # Scale light intensity by pi to match fixed-function diffuse brightness.
        # The engine's Lambert term divides intensity by pi.
        {"name": "light1", "pos": "1 1 4", "diffuse": "0.8 0.8 0.8",
         "intensity": "3.14159265"},
        {"name": "light2", "pos": "-3 -3 4", "diffuse": "0.8 0.8 0.8",
         "intensity": "3.14159265"},
    ):
        ET.SubElement(worldbody, "light", attrs)

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return {
        "output": str(output.relative_to(REPO)),
        "source_robot": str(PANDA_SOURCE.relative_to(REPO)),
        "source_task": str(TASK_BDDL.relative_to(REPO)),
        "task_index": 2,
        "task": "pick up the black bowl from table center and place it on the plate",
        "panda_home": list(PANDA_HOME),
        "object_poses": {name: list(pos) for name, pos in OBJECT_POSES.items()},
        "camera_contract": ["agentview", "robot0_eye_in_hand"],
        "action_contract": "7D relative OSC: xyz +/-0.05m, axis-angle +/-0.5rad, gripper [-1,1]",
    }


def cook(xml_path: Path, output_path: Path = COOKED, cooker: str | None = None) -> None:
    if cooker is None:
        in_tree = REPO / "build-linux/src/nuka_cook_scene"
        cooker_path = str(in_tree) if in_tree.is_file() else "/root/nuka-build/src/nuka_cook_scene"
    else:
        cooker_path = cooker
    command = [cooker_path, str(xml_path), str(output_path)]
    if sys.platform == "win32" and cooker_path.startswith("/"):
        repo_wsl = "/mnt/c/Softwares/code/Nuka-Physics"
        xml_wsl = repo_wsl + "/" + xml_path.relative_to(REPO).as_posix()
        output_wsl = repo_wsl + "/" + output_path.relative_to(REPO).as_posix()
        command = [
            "wsl",
            "-d",
            "Ubuntu-24.04",
            "--",
            cooker_path,
            xml_wsl,
            output_wsl,
        ]
    subprocess.run(command, check=True, cwd=REPO)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cook", action="store_true")
    parser.add_argument("--cooker", default=None)
    args = parser.parse_args()
    result = generate(args.output)
    if args.cook:
        cook(args.output, args.output.with_suffix(".nks"), args.cooker)
        result["cooked"] = str(args.output.with_suffix(".nks").relative_to(REPO))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
