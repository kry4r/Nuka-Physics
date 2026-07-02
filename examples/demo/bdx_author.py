#!/usr/bin/env python3
"""Author examples/scenes/bdx_stand.nks from the BDX MJCF -- pure Python.

Imports the open_duck_mini_v2 MJCF through the general C-ABI importer
(nuka.Scene.load -> save, meshes land in the sibling .nka), then edits the
declarative .nks JSON in place: pins the freejoint host body as the static
spawn anchor, bakes the home stance into the joint initial positions, swaps
the mesh sole colliders for sole CAPSULES (the engine's robust heightfield
foot primitive -- the convex-hull EPA shallow-penetration dead band makes a
mesh sole sink), and adds a flat heightfield floor.

The capsule local poses/size are the sole AABB of each foot at the home
stance (FK-derived once, checked by the stand gate); tree order = left, right.

Run from the repo root:
    python examples/demo/bdx_author.py
"""

import json

import nuka

MJCF = "examples/assets/bdx/open_duck_mini_v2.xml"
OUT = "examples/scenes/bdx_stand.nks"
SPAWN_Z = 0.21

HOME = {
    "left_hip_yaw": 0.002, "left_hip_roll": 0.053, "left_hip_pitch": -0.63,
    "left_knee": 1.368, "left_ankle": -0.784,
    "neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": -0.003, "right_hip_roll": -0.065, "right_hip_pitch": 0.635,
    "right_knee": 1.379, "right_ankle": -0.796,
}

# Sole capsules in the foot body frame (home-stance sole AABB; axis horizontal).
CAPSULE = {
    "type": "capsule", "half_extents": [0.5, 0.5, 0.5],
    "radius": 0.02064934, "half_height": 0.0309066065,
    "material_id": 4294967295, "contype": 1, "conaffinity": 1,
    "collision_group": 0, "solref": [0.02, 1.0],
    "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "friction_mu": 1.0,
    "priority": 0, "solmix": 1.0, "margin": 0.0, "gap": 0.0, "condim": 3,
    "decompose_mode": "auto", "decompose_max_pieces": 32,
}
FOOT_CAP_LOCAL = {  # foot body name -> capsule local pose.
    "foot_assembly": {"pos": [0.0134650022, -0.0181903541, 0.019052133],
                      "quat": [-0.50000006, 0.50000006, 0.5, 0.5]},
    "foot_assembly_2": {"pos": [0.0134650022, -0.0181903541, 0.0190521181],
                        "quat": [0.50000006, -0.50000006, -0.5, -0.5]},
}

FLAT_GROUND = {
    "name": "ground", "nrow": 21, "ncol": 21, "cell": 0.1,
    "origin": [-1.0, -1.0, 0.0], "base_z": 0.0, "grade_x": 0.0, "grade_y": 0.0,
    "ring_rise": 0.0, "ring_width": 0.0, "ring_platform": 0.0, "ring_count": 0,
    "bump_height": 0.0, "bump_cell": 0.0, "feature_cell": 0.0,
    "feature_margin": 0.0, "feature_seed": 0, "curric_levels": 0,
    "curric_types": 0, "image_path": "", "image_radius_x": 0.0,
    "image_radius_y": 0.0, "image_elevation_z": 0.0,
}


def edit(node):
    """Recursive declarative pass over one tree node (lists recurse)."""
    if isinstance(node, list):
        for child in node:
            edit(child)
        return
    name = node.get("name", "")
    if name == "base" and "rigid_body" in node:
        node["rigid_body"]["is_static"] = True
        node["transform"]["pos"] = [0.0, 0.0, SPAWN_Z]
    joint = node.get("joint")
    if joint and joint.get("name") in HOME:
        joint["initial_position"] = HOME[joint["name"]]
    shape = node.get("collision_shape")
    if shape and "foot_bottom_tpu" in name and shape.get("contype") == 1:
        shape["contype"] = 0   # mesh sole -> visual only (EPA dead band).
        shape["conaffinity"] = 0
    if name in FOOT_CAP_LOCAL:
        node.setdefault("children", []).append({
            "name": "foot_contact",
            "collision_shape": dict(CAPSULE, local=FOOT_CAP_LOCAL[name]),
            "children": [],
        })
    edit(node.get("children", []))


def main():
    scene = nuka.Scene.load(MJCF)   # general importer; meshes -> sibling .nka
    scene.save(OUT)
    scene.destroy()

    doc = json.load(open(OUT))
    edit(doc["tree"])
    doc["terrain"] = [FLAT_GROUND]
    with open(OUT, "w") as f:
        json.dump(doc, f)
    print(f"[author] {MJCF} -> {OUT} (home stance + sole capsules + flat floor)")


if __name__ == "__main__":
    main()
