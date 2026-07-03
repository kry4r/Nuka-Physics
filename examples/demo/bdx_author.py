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

# Sole contact = a 4-BALL support polygon per foot (toe/heel x left/right), so the
# foot has a real support area and the CoP can travel fore-aft for toe-off -- the
# single sole capsule was a LINE contact and could not push. The engine now cooks
# every collision geom on a link (extra ones become collidable proxy body rows), so
# all four balls are live. Foot-body frame (FK-calibrated at home): +X = forward
# (fore-aft), +Y = up (sole normal), +-Z = lateral. Ball bottoms sit at the sole
# plane (the old capsule's lowest point), preserving the stand height.
BALL_R = 0.013
L_FORE = 0.026   # fore-aft half-separation (foot-local +X).
L_LAT = 0.013    # lateral half-separation (foot-local +-Z).
SOLE_CENTER = [0.0134650022, -0.0181903541, 0.019052133]  # old capsule center.
OLD_CAP_R = 0.02064934
BALL_UP = SOLE_CENTER[1] - OLD_CAP_R + BALL_R  # ball center Y so its bottom == sole.


def _foot_balls():
    """Four sole balls at the foot footprint corners (tree order fixed)."""
    balls = []
    for i, (sf, sl) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        pos = [SOLE_CENTER[0] + sf * L_FORE, BALL_UP, SOLE_CENTER[2] + sl * L_LAT]
        balls.append({
            "name": "foot_contact" if i == 0 else f"foot_contact_{i}",
            "collision_shape": {
                "type": "sphere", "half_extents": [0.5, 0.5, 0.5],
                "radius": BALL_R, "half_height": 0.5,
                "material_id": 4294967295, "contype": 1, "conaffinity": 1,
                "collision_group": 0, "solref": [0.02, 1.0],
                "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "friction_mu": 1.0,
                "priority": 0, "solmix": 1.0, "margin": 0.0, "gap": 0.0, "condim": 3,
                "decompose_mode": "auto", "decompose_max_pieces": 32,
                "local": {"pos": pos, "quat": [1.0, 0.0, 0.0, 0.0]},
            },
            "children": [],
        })
    return balls


FOOT_BODIES = ("foot_assembly", "foot_assembly_2")

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
    if name in FOOT_BODIES:
        node.setdefault("children", []).extend(_foot_balls())
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
