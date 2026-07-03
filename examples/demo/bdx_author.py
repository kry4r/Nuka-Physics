#!/usr/bin/env python3
"""Author examples/scenes/bdx_stand.nks from the BDX MJCF -- pure Python.

Imports the open_duck_mini_v2 MJCF through the general C-ABI importer
(nuka.Scene.load -> save, meshes land in the sibling .nka), then edits the
declarative .nks JSON in place: pins the freejoint host body as the static
spawn anchor, bakes the home stance into the joint initial positions, drops the
mis-cooked mesh sole collider and gives each foot a flat elongated rigid FOOT
PLATE box (a rounded sole rolls and nets ~0 forward ground reaction; a flat plate
carries the fore-aft toe-off), and adds a flat heightfield floor.

The plate half-extents copy the MuJoCo foot_bottom_tpu sole; its local pose is
FK-derived at the home stance (foot-local +X = fore-aft, +Y = up, +-Z = lateral;
bottom face on the sole plane) and checked by the stand gate.

Run from the repo root:
    python examples/demo/bdx_author.py
"""

import json

import nuka

MJCF = "examples/assets/bdx/open_duck_mini_v2.xml"
OUT = "examples/scenes/bdx_stand.nks"
SPAWN_Z = 0.21
FRICTIONLOSS = 0.068   # sts3215 joint dry friction (MJCF class value).

HOME = {
    "left_hip_yaw": 0.002, "left_hip_roll": 0.053, "left_hip_pitch": -0.63,
    "left_knee": 1.368, "left_ankle": -0.784,
    "neck_pitch": 0.0, "head_pitch": 0.0, "head_yaw": 0.0, "head_roll": 0.0,
    "right_hip_yaw": -0.003, "right_hip_roll": -0.065, "right_hip_pitch": 0.635,
    "right_knee": 1.379, "right_ankle": -0.796,
}

# Flat rigid foot plate (foot-local frame, FK-calibrated at home): +X = fore-aft,
# +Y = up (thickness), +-Z = lateral. Half-extents copy MuJoCo foot_bottom_tpu
# (heel-to-toe ~10 cm). Centered fore-aft/lateral on the sole; bottom face on the
# sole plane (prior sole capsule's lowest point) so the stand height is preserved.
SOLE_CENTER = [0.0134650022, -0.0181903541, 0.019052133]  # sole geometric center.
OLD_CAP_R = 0.02064934                                     # prior sole capsule radius (sole depth).
PLATE_HALF = [0.052, 0.005, 0.021]                         # X fore-aft, Y thickness, Z lateral.
PLATE_UP = SOLE_CENTER[1] - OLD_CAP_R + PLATE_HALF[1]      # plate center Y: bottom on sole plane.
PLATE_POS = [SOLE_CENTER[0], PLATE_UP, SOLE_CENTER[2]]


def _foot_plate():
    """One flat rigid foot-plate box (both feet share this foot-local pose)."""
    return {
        "name": "foot_contact",
        "collision_shape": {
            "type": "box", "half_extents": list(PLATE_HALF),
            "radius": PLATE_HALF[1], "half_height": 0.5,
            "material_id": 4294967295, "contype": 1, "conaffinity": 1,
            "collision_group": 0, "solref": [0.02, 1.0],
            "solimp": [0.9, 0.95, 0.001, 0.5, 2.0], "friction_mu": 1.0,
            "priority": 0, "solmix": 1.0, "margin": 0.0, "gap": 0.0, "condim": 3,
            "decompose_mode": "auto", "decompose_max_pieces": 32,
            "local": {"pos": list(PLATE_POS), "quat": [1.0, 0.0, 0.0, 0.0]},
        },
        "children": [],
    }


FOOT_BODIES = ("foot_assembly", "foot_assembly_2")


def _is_mesh_sole(node):
    """The mis-cooked mesh sole collider (garbage AABB placeholder to drop)."""
    return bool(node.get("collision_shape")) and "foot_bottom_tpu" in node.get("name", "")

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
        joint["frictionloss"] = FRICTIONLOSS
    if name in FOOT_BODIES:
        # Drop the mis-cooked mesh sole collider; add the flat foot plate.
        kids = [c for c in node.get("children", []) if not _is_mesh_sole(c)]
        kids.append(_foot_plate())
        node["children"] = kids
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
    print(f"[author] {MJCF} -> {OUT} (home stance + flat foot plates + flat floor)")


if __name__ == "__main__":
    main()
