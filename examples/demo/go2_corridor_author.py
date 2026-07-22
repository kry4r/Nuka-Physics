#!/usr/bin/env python3
"""Author corridor_go2.nks/.nka: the canonical corridor with the Go2 robot subtree.

Geometry source is corridor_nomedia.nks (every static/dynamic corridor body kept
byte-for-byte); the BDX robot at tree[0] is replaced by the Go2 articulation from
go2.nks. Material tables are concatenated corridor-first so the preserved corridor
ids stay valid and the swapped Go2 subtree's ids shift by the corridor table length.
The Go2 mesh .nka is copied and its embedded basename retargeted; the BDX head-depth
sensors block is stripped (its links no longer exist).

Run from the repo root:
    python examples/demo/go2_corridor_author.py
"""

import copy
import json
import os
import shutil

CORRIDOR_SRC = "examples/scenes/corridor_nomedia.nks"
GO2_SRC = "examples/scenes/go2.nks"
GO2_NKA = "examples/scenes/go2.nka"
OUT_NKS = "examples/scenes/corridor_go2.nks"
OUT_NKA = "examples/scenes/corridor_go2.nka"

NO_MATERIAL = 4294967295          # sentinel: "use the default material".
SPAWN_X = -0.2                    # start platform centre (profile top 0.15 m).
PLATFORM_TOP = 0.15              # corridor start-platform top (scene ground truth).
GO2_STANCE = 0.30               # cooked-crouch trunk height above the feet.

# Cook the go2 subtree at the legged_gym default crouch (the env resets to it) so
# the feet spawn on the ground, not the straight-leg cook height. go2.nks joint
# order is URDF order (FL,FR,RL,RR x hip,thigh,calf).
CROUCH = {
    "FL_hip_joint": 0.1, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
    "FR_hip_joint": -0.1, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.1, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
    "RR_hip_joint": -0.1, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
}

# The go2.nks foot colliders (one per calf, named FL/FR/RL/RR) are 0.5 m
# half-extent boxes -- FusedFoot point markers that are unusable as general
# contact geometry. Replace each with a small sphere at the calf-end foot origin.
FOOT_NAMES = ("FL", "FR", "RL", "RR")
FOOT_RADIUS = 0.022
FOOT_LOCAL = [0.0, 0.0, -0.213]


def _remap_mesh_nka(node, old_base, new_base):
    """Retarget the embedded ``<base>.nka#MESH/i`` asset basename in place."""
    old, new = old_base + ".nka#", new_base + ".nka#"
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str) and old in v:
                node[k] = v.replace(old, new)
            else:
                _remap_mesh_nka(v, old_base, new_base)
    elif isinstance(node, list):
        for v in node:
            _remap_mesh_nka(v, old_base, new_base)


def _offset_material_ids(node, offset):
    """Shift every real material_id by ``offset`` (the sentinel stays default)."""
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "material_id" and isinstance(v, int) and v != NO_MATERIAL:
                node[k] = v + offset
            else:
                _offset_material_ids(v, offset)
    elif isinstance(node, list):
        for v in node:
            _offset_material_ids(v, offset)


def _set_crouch(node):
    """Bake the default crouch into the go2 joint initial positions."""
    n = 0
    if isinstance(node, dict):
        j = node.get("joint")
        if j and j.get("name") in CROUCH:
            j["initial_position"] = CROUCH[j["name"]]
            n += 1
        for c in node.get("children", []):
            n += _set_crouch(c)
    return n


def _fix_feet(node):
    """Rewrite the degenerate FL/FR/RL/RR foot boxes to calf-end foot spheres."""
    fixed = 0
    if isinstance(node, dict):
        cs = node.get("collision_shape")
        if node.get("name") in FOOT_NAMES and cs:
            cs["type"] = "sphere"
            cs["radius"] = FOOT_RADIUS
            cs["half_extents"] = [FOOT_RADIUS, FOOT_RADIUS, FOOT_RADIUS]
            cs["half_height"] = FOOT_RADIUS
            cs["local"] = {"pos": list(FOOT_LOCAL), "quat": [1.0, 0.0, 0.0, 0.0]}
            fixed += 1
        for c in node.get("children", []):
            fixed += _fix_feet(c)
    return fixed


def main():
    cor = json.load(open(CORRIDOR_SRC))
    go2 = json.load(open(GO2_SRC))

    offset = len(cor["render_materials"])
    assert len(cor["physics_materials"]) == offset, \
        "corridor render/physics material tables differ in length"
    assert len(go2["render_materials"]) == len(go2["physics_materials"]), \
        "go2 render/physics material tables differ in length"

    # Corridor materials first keep the preserved corridor ids valid; the Go2
    # subtree ids shift by the corridor table length. Ids resolve by insertion
    # index, so the Go2 keys are prefixed to stay unique (a shared name like
    # "metal" would otherwise collapse two distinct slots).
    def _prefixed(table):
        return {f"go2_{k}": v for k, v in table.items()}

    new_render = dict(cor["render_materials"])
    new_render.update(_prefixed(go2["render_materials"]))
    new_physics = dict(cor["physics_materials"])
    new_physics.update(_prefixed(go2["physics_materials"]))
    assert len(new_render) == offset + len(go2["render_materials"]), \
        "material name collision after prefixing the go2 tables"

    robot = copy.deepcopy(go2["tree"][0])
    assert robot.get("name") == "base", "go2 robot root is not 'base'"
    n_feet = _fix_feet(robot)
    assert n_feet == len(FOOT_NAMES), \
        f"expected {len(FOOT_NAMES)} foot colliders, rewrote {n_feet}"
    n_crouch = _set_crouch(robot)
    assert n_crouch == len(CROUCH), \
        f"expected {len(CROUCH)} joints, set {n_crouch}"
    _offset_material_ids(robot, offset)
    _remap_mesh_nka(robot, "go2", "corridor_go2")
    robot["transform"]["pos"] = [SPAWN_X, 0.0, PLATFORM_TOP + GO2_STANCE]

    out = dict(cor)
    out["render_materials"] = new_render
    out["physics_materials"] = new_physics
    out["tree"] = [robot] + cor["tree"][1:]
    out["sensors"] = []
    # exclude_pairs / contact_pairs carry no BDX link names in the source; keep them.
    assert not cor.get("exclude_pairs") and not cor.get("contact_pairs"), \
        "source corridor has non-empty pair lists referencing BDX links"

    with open(OUT_NKS, "w") as f:
        json.dump(out, f)
    shutil.copyfile(GO2_NKA, OUT_NKA)
    print(f"[author] {CORRIDOR_SRC} + {GO2_SRC} -> {OUT_NKS}")
    print(f"[author] robot base pos={robot['transform']['pos']} "
          f"materials={len(new_render)} (corridor {offset} + go2 "
          f"{len(go2['render_materials'])}) sensors stripped; "
          f".nka {GO2_NKA} -> {OUT_NKA} ({os.path.getsize(OUT_NKA)} bytes)")


if __name__ == "__main__":
    main()
