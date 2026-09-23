"""Build the G1, Newton jacket, shallow-water crossing and layered granular course."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from pxr import Usd, UsdGeom

from garment_retarget import body_envelopes, body_landmarks, fit_clearance, retarget_surface


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/media"))
from build_nuka_lab import look_rotation


def save(path, document):
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def mesh_payload(vertices, triangles):
    vertices = np.asarray(vertices, dtype="<f4")
    triangles = np.asarray(triangles, dtype="<u4").reshape(-1, 3)
    normals = np.zeros_like(vertices)
    face = np.cross(vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
                    vertices[triangles[:, 2]] - vertices[triangles[:, 0]])
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, np.finfo(np.float32).tiny)
    return (struct.pack("<II", len(vertices), triangles.size) + vertices.tobytes()
            + normals.tobytes() + np.zeros((len(vertices), 2), dtype="<f4").tobytes()
            + triangles.tobytes())


def write_meshes(path, meshes):
    payloads = [mesh_payload(*mesh) for mesh in meshes]
    offset = 12 + 28 * len(payloads)
    toc = []
    for data in payloads:
        digest = int.from_bytes(hashlib.sha256(data).digest()[:8], "little")
        toc.append(struct.pack("<IQQQ", int.from_bytes(b"MESH", "big"), offset, len(data), digest))
        offset += len(data)
    path.write_bytes(b"NKA1" + struct.pack("<II", 1, len(payloads)) + b"".join(toc) + b"".join(payloads))


def robot(output, cooker, base_height, sensor_exclusions, collision_geometry, mesh_orientation,
          collision_profile, camera_profile):
    source = ROOT / ".nuka-assets/src/g1-moves/mjlab/src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1.xml"
    tree = ET.parse(source)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(source.parent / "assets"))
    root.find("./worldbody/body").set("pos", f"0 0 {base_height}")
    deploy = yaml.safe_load((ROOT / ".nuka-assets/policies/g1_velocity_v0/deploy.yaml").read_text())
    actuators = json.loads((ROOT / ".nuka-assets/policies/g1_velocity_v0/actuators.json").read_text())
    names = actuators["joint_sdk_names"]
    default = {names[sdk]: q for sdk, q in zip(deploy["joint_ids_map"], deploy["default_joint_pos"])}
    for joint in root.findall(".//body/joint"):
        name = joint.get("name")
        joint.set("nuka:initial_position", str(default[name]))
        joint.set("armature", str(actuators["joints"][name]["armature"]))
    if collision_geometry == "visual_surface":
        for body in root.findall(".//body"):
            geometry = list(body.findall("geom"))
            visuals = [geom for geom in geometry if geom.get("class") == "visual"]
            for geom in geometry:
                if geom not in visuals:
                    body.remove(geom)
            for index, visual in enumerate(visuals):
                attributes = dict(visual.attrib)
                attributes.update(name=f"{body.get('name')}_surface_{index}",
                    type="mesh", contype="1", conaffinity="1", group="3", density="0")
                attributes["class"] = "collision"
                attributes["nuka:decompose"] = "skip"
                attributes["nuka:mesh_orientation"] = mesh_orientation
                ET.SubElement(body, "geom", **attributes)
    profile = json.loads(collision_profile.read_text(encoding="utf-8"))
    additional_exclusions = []
    if collision_geometry == profile["collision_geometry"]:
        contact = root.find("contact")
        if contact is None:
            contact = ET.SubElement(root, "contact")
        bodies = {body.get("name") for body in root.findall(".//body")}
        existing = {frozenset((entry.get("body1"), entry.get("body2")))
                    for entry in contact.findall("exclude")}
        for pair in profile["exclude_body_pairs"]:
            if len(pair) != 2 or not set(pair) <= bodies or pair[0] == pair[1]:
                raise ValueError(f"Invalid robot collision exclusion: {pair}")
            if frozenset(pair) not in existing:
                ET.SubElement(contact, "exclude", body1=pair[0], body2=pair[1])
                existing.add(frozenset(pair))
                additional_exclusions.append(pair)
    drives = ET.SubElement(root, "actuator")
    for name in names:
        limit = actuators["joints"][name]["effort_limit_sim"]
        ET.SubElement(drives, "position", name=name + "_drive", joint=name,
                      forcerange=f"{-limit} {limit}")
    omitted_sensors = []
    sensors = root.find("sensor")
    if sensors is not None:
        for sensor in list(sensors):
            if sensor.get("name") in sensor_exclusions:
                omitted_sensors.append({"type": sensor.tag, **sensor.attrib})
                sensors.remove(sensor)
    source_out = output / "robot.xml"
    ET.indent(tree, space="  ")
    tree.write(source_out, encoding="utf-8", xml_declaration=True)
    subprocess.run([str(cooker), str(source_out), str(output / "robot.nks")], check=True)
    authored = json.loads((output / "robot.nks").read_text(encoding="utf-8"))
    camera_document = json.loads(camera_profile.read_text(encoding="utf-8"))
    nodes = {}

    def collect(children):
        for node in children:
            if "rigid_body" in node:
                nodes[node["name"]] = node
            collect(node.get("children", []))

    collect(authored["tree"])
    for camera in camera_document["cameras"]:
        body = nodes.get(camera["attached_body"])
        if body is None:
            raise ValueError(f"Unknown camera body: {camera['attached_body']}")
        eye, look = np.asarray(camera["eye"]), np.asarray(camera["look"])
        optics = {key: camera[key] for key in
                  ("vertical_fov_degrees", "near_clip", "far_clip", "shadow_radius")}
        body.setdefault("children", []).append({"name": camera["name"], "camera": {
            "local": {"pos": eye.tolist(), "quat": look_rotation(eye, look)},
            "focus_distance": float(np.linalg.norm(eye - look)), **optics}})
    save(output / "robot.nks", authored)
    return {"source": str(source.relative_to(ROOT)), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "default_joint_position": default, "base_height": base_height,
            "collision_geometry": collision_geometry,
            "mesh_orientation": mesh_orientation,
            "collision_profile": {"file": str(collision_profile.relative_to(ROOT)
                if collision_profile.is_relative_to(ROOT) else collision_profile),
                "sha256": hashlib.sha256(collision_profile.read_bytes()).hexdigest(),
                "additional_exclusions": additional_exclusions, "description": profile["description"]},
            "camera_profile": {"file": str(camera_profile.relative_to(ROOT)
                if camera_profile.is_relative_to(ROOT) else camera_profile),
                "sha256": hashlib.sha256(camera_profile.read_bytes()).hexdigest(),
                "cameras": camera_document["cameras"]},
            "collisions": ("Complete source collision primitives and exclusions" if collision_geometry == "source"
                           else "Authored visual triangle surfaces and original body exclusions"),
            "omitted_sensors": omitted_sensors}


def jacket(output, scale, offset, robot_info, clearance):
    source = ROOT / ".nuka-assets/garments/newton_h1_jacket"
    stage = Usd.Stage.Open(str(source / "h1_jacket.usd"))
    prim = stage.GetPrimAtPath("/Root/h1_jacket/Root_Garment")
    mesh = UsdGeom.Mesh(prim)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    if not np.all(counts == 3):
        raise ValueError("The garment must be triangulated")
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)[:, [2, 0, 1]]
    triangles = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.uint32).reshape(-1, 3)
    source_names = [side + suffix for side in ("left", "right")
                    for suffix in ("_shoulder_roll_link", "_elbow_link", "_hand_link")]
    target_names = [side + suffix for side in ("left", "right")
                    for suffix in ("_shoulder_roll_link", "_elbow_link", "_wrist_yaw_link")]
    reference = body_landmarks(source / "h1_with_hand.xml", source_names)
    target = body_landmarks(ROOT / robot_info["source"], target_names,
                            robot_info["default_joint_position"], robot_info["base_height"])
    source_chains = [np.array([reference[name] for name in source_names[i:i+3]]) for i in (0, 3)]
    target_chains = [np.array([target[name] for name in target_names[i:i+3]]) for i in (0, 3)]
    vertices, fitting = retarget_surface(vertices, triangles, source_chains, target_chains, scale=scale,
        source_anchor=np.mean([chain[0] for chain in source_chains], axis=0),
        target_anchor=np.mean([chain[0] for chain in target_chains], axis=0))
    vertices += np.asarray(offset)
    envelopes = body_envelopes(ROOT / robot_info["source"], robot_info["default_joint_position"],
                              robot_info["base_height"])
    vertices, fitting["clearance_fit"] = fit_clearance(vertices, triangles, envelopes, clearance)
    uv = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
    pattern = np.asarray(uv.Get(), dtype=np.float64) * (0.001 * scale)
    pattern = np.column_stack((pattern, np.zeros(len(pattern))))
    pattern_triangles = np.asarray(uv.GetIndices(), dtype=np.uint32).reshape(-1, 3)
    write_meshes(output / "jacket.nka", [(vertices, triangles), (pattern, pattern_triangles)])
    save(output / "jacket.nks", {"nks_version": 1,
        "render_materials": {"jacket_avocado": {"base_color": [0.23, 0.34, 0.095, 1],
            "roughness": 0.88, "metallic": 0.0, "sheen": 0.28}},
        "media": [{"name": "jacket", "kind": "cloth", "method": "xpbd", "baked": "jacket.nka#MESH/0",
            "cloth_mesh": {"material_mesh": "jacket.nka#MESH/1", "pinned_vertices": []},
            "xpbd": {"surface_density": 0.5, "half_thickness": 0.003, "friction": 0.35,
                "distance_alpha": 1e-6, "bend_alpha": 25000.0, "iters": 8},
            "render_material_id": 0}]})
    shutil.copy2(source / "LICENSE", output / "JACKET_LICENSE")
    return {"source": json.loads((source / "manifest.json").read_text()), "scale": scale,
            "offset": offset, "fitting": fitting, "vertices": len(vertices), "triangles": len(triangles),
            "bounds": [vertices.min(0).tolist(), vertices.max(0).tolist()]}


def box(name, center, half, material, friction=0.8):
    return {"name": name, "transform": {"pos": center},
            "rigid_body": {"is_static": True, "mass": 0}, "children": [{"name": name + "_shape",
                "collision_shape": {"type": "box", "half_extents": half,
                    "material": material, "friction_mu": friction}}]}


def course(output, appearance):
    width, wall = 1.2, 0.08
    surface = 0.24
    segments = [(-1.5, 0.0, 0.30), (0.0, 0.3, 0.24), (0.3, 0.6, 0.18),
                (0.6, 0.9, 0.12), (0.9, 1.2, 0.06), (1.2, 4.2, 0.0),
                (4.2, 4.5, 0.06),
                (4.5, 4.8, 0.12), (4.8, 5.1, 0.18), (5.1, 5.4, 0.24),
                (5.4, 5.7, 0.30), (5.7, 7.8, 0.18), (7.8, 8.7, 0.30)]
    granular_start, granular_end = 5.7, 7.8
    course_center = (segments[0][0] + segments[-1][1]) / 2
    course_half_length = (segments[-1][1] - segments[0][0]) / 2
    nodes = []
    for i, (start, end, height) in enumerate(segments):
        nodes.append(box(f"tread_{i}", [(start+end)/2, 0, (height-0.12)/2],
                         [(end-start)/2, width/2, (height+0.12)/2], "work_deck"))
        if i > 0 and start != granular_start:
            nodes.append(box(f"tread_edge_{i}", [start+0.007, 0, height-0.005],
                             [0.007, width/2, 0.005], "avocado"))
    for side in (-1, 1):
        nodes.append(box(f"side_wall_{side}", [course_center, side*(width+wall)/2, 0.10],
                         [course_half_length, wall/2, 0.22], "warm_shell"))
    nodes.append(box("foundation", [course_center, 0, -0.22],
                     [course_half_length+0.15, 0.85, 0.10], "graphite"))
    eye, target = [10.4, -10.6, 6.3], [course_center, 0, 0.55]
    nodes.append({"name": "wide", "camera": {"local": {"pos": eye, "quat": look_rotation(eye, target)},
        "vertical_fov_degrees": 42, "near_clip": 0.02, "far_clip": 50,
        "focus_distance": float(np.linalg.norm(np.asarray(eye)-target))}})
    nodes.append({"name": "key_light", "light": {"type": "directional", "color": [3.4, 3.2, 2.9],
        "intensity": 1, "local": {"quat": look_rotation([-0.5, -0.7, 0.95], [0, 0, 0])}}})
    save(output / "course.nks", {"nks_version": 1, "imports": [{"file": str(appearance)}],
        "tree": nodes, "environment": {"shadow": {"center": [course_center, 0, 0.4],
                                                    "radius": course_half_length+0.5}}})
    grid = {"dx": 0.04, "substeps": 4, "floor_normal": [0, 0, 1], "floor_d": -0.12,
            "floor_friction": 0.0, "loft_headroom": 0.7}
    material = {"density": 1000, "model_kind": 3, "bulk_modulus": 50000, "tait_gamma": 7,
                **grid}
    fills = []
    spacing = 0.02
    pool_start, pool_end = 0.0, 5.4
    for start, end, height in segments:
        start, end = max(start, pool_start), min(end, pool_end)
        if height >= surface or start >= end:
            continue
        fills.append({"box": {"min": [start, -width/2, height],
                              "max": [end, width/2, surface],
                              "spacing": spacing}, "mpm": material})
    water = {"name": "pool_water", "kind": "fluid", "method": "mlsmpm",
             "fluid_box": fills[0]["box"], "mpm": material, "mpm_fills": fills[1:], "render_material_id": 0}
    save(output / "water.nks", {"nks_version": 1, "render_materials": {"water": {
        "base_color": [0.7, 0.88, 0.93, 1], "roughness": 0.025, "transmission": 1.0,
        "ior": 1.333, "absorption": [0.32, 0.08, 0.035]}}, "media": [water]})
    layers = [
        {"name": "sand", "bottom": 0.18, "top": 0.24, "spacing": 0.015,
         "density": 1600, "youngs": 100000, "friction_angle": 32, "round": True},
        {"name": "gravel", "bottom": 0.24, "top": 0.30, "spacing": 0.02,
         "density": 1850, "youngs": 250000, "friction_angle": 40, "round": False}]
    granular_media = []
    for index, layer in enumerate(layers):
        granular_media.append({"name": layer["name"], "kind": "granular", "method": "mlsmpm",
            "fluid_box": {"min": [granular_start, -width/2, layer["bottom"]],
                          "max": [granular_end, width/2, layer["top"]], "spacing": layer["spacing"]},
            "mpm": {**grid, "model_kind": 4, "density": layer["density"],
                    "youngs": layer["youngs"], "poisson": 0.2,
                    "dp_friction": layer["friction_angle"], "dp_cohesion": 0},
            "render_skin": {"grain_round": int(layer["round"]), "grain_radius_jitter": 0.15,
                            "grain_tint_jitter": 0.18}, "render_material_id": index})
    save(output / "granular.nks", {"nks_version": 1, "render_materials": {
        "sand": {"base_color": [0.56, 0.39, 0.21, 1], "roughness": 0.96},
        "gravel": {"base_color": [0.31, 0.29, 0.25, 1], "roughness": 0.9}}, "media": granular_media})
    save(output / "dry_cover.nks", {"nks_version": 1, "imports": [{"file": str(appearance)}],
        "tree": [box("granular_training_cover", [(granular_start+granular_end)/2, 0, 0.24],
                     [(granular_end-granular_start)/2, width/2, 0.06], "work_deck")]})
    pose = {"pos": [-0.85, 0, 0.30]}
    imports = [{"file": "course.nks"}, {"file": "robot.nks", "transform": pose}]
    save(output / "dry.nks", {"nks_version": 1, "imports": imports + [{"file": "dry_cover.nks"}]})
    imports = imports + [{"file": "jacket.nks", "transform": pose}]
    save(output / "clothed.nks", {"nks_version": 1, "imports": imports + [{"file": "dry_cover.nks"}]})
    save(output / "wading.nks", {"nks_version": 1,
        "imports": imports + [{"file": "water.nks"}, {"file": "granular.nks"}]})
    save(output / "flat.nks", {"nks_version": 1, "imports": [{"file": str(appearance)}, {"file": "robot.nks"}],
        "tree": [box("floor", [0, 0, -0.05], [10, 10, 0.05], "work_deck"), *nodes[-2:]]})
    return {"segments": segments, "water_surface": surface, "water_spacing": spacing, "width": width,
            "pool_region": [pool_start, pool_end], "pool_bottom_region": [1.2, 4.2],
            "start_position": pose["pos"], "finish_x": 8.25,
            "granular_region": [granular_start, granular_end], "granular_layers": layers,
            "granular_model": "Drucker-Prager MPM continuum; render grains represent material samples"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / ".nuka-assets/generated/g1_coupled_course")
    parser.add_argument("--cooker", type=Path, default=ROOT / "build-linux/src/nuka_cook_scene")
    parser.add_argument("--jacket-scale", type=float, default=0.75)
    parser.add_argument("--jacket-offset", nargs=3, type=float, default=[0.0, 0.0, 0.0])
    parser.add_argument("--base-height", type=float, default=0.8)
    parser.add_argument("--jacket-clearance", type=float, default=0.008)
    parser.add_argument("--collision-geometry", choices=("source", "visual_surface"), default="visual_surface")
    parser.add_argument("--mesh-orientation", choices=("automatic", "outward"), default="outward")
    parser.add_argument("--collision-profile", type=Path,
                        default=ROOT / "examples/assets/g1_wading/robot_collision.json")
    parser.add_argument("--camera-profile", type=Path,
                        default=ROOT / "examples/assets/g1_wading/cameras.json")
    parser.add_argument("--exclude-sensor", nargs="*",
                        default=["root_angmom", "imu_ang_vel", "imu_lin_vel", "imu_lin_acc"],
                        help="Omit named virtual diagnostics; the source subtree angular momentum is unsupported")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    robot_info = robot(args.out, args.cooker, args.base_height, args.exclude_sensor,
                       args.collision_geometry, args.mesh_orientation, args.collision_profile.resolve(),
                       args.camera_profile.resolve())
    manifest = {"robot": robot_info,
                "jacket": jacket(args.out, args.jacket_scale, args.jacket_offset, robot_info, args.jacket_clearance),
                "course": course(args.out, ROOT / "examples/assets/nuka_lab/appearance.nks")}
    save(args.out / "manifest.json", manifest)
    print(json.dumps({"output": str(args.out), "jacket": manifest["jacket"]["bounds"]}))


if __name__ == "__main__":
    main()
