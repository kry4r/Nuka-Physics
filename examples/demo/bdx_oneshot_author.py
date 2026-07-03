#!/usr/bin/env python3
"""Author the BDX one-shot demo scene: a single continuous +x walkway that strings
together four zones onto the ONE general physics path, saved as a self-contained
``examples/scenes/bdx_oneshot.nks`` (cook it back with
``SceneBuilder.create(path).build(...)``).

Zones (world coords, +z up, +x = travel):
  A  wooden stairs (5 x rise 0.04, run 0.14) rising behind the spawn to a top
     platform, a wooden door frame at x=0.9 with a fabric curtain draped over the
     lintel (a free XPBD cloth);
  B  a shallow gravel trough x in [1.4,3.0] filled with an MLS-MPM Drucker-Prager
     granular bed (model_kind 4);
  C  ~40 small free rigid bodies (pebbles / bolts: sphere+capsule+box) scattered
     on the walkway x in [3.2,4.2];
  D  an overhead beam at x=5.0 with a rope (revolute capsule chain) hanging a
     ~0.3 kg block at head height.

The duck (open_duck_mini_v2, examples/scenes/bdx_stand.nks) spawns at the stair
foot; its head + trunk gain a colliding geom (they ship visual-only) so the D01
cloth and the D04 block can push on them.

Everything is authored through the general Python facade: SceneBuilder rigid
primitives + media + render materials + environment, add_collision_shape for the
duck geoms, and compose for the rope articulation fragment. The ground terrain (a
big flat heightfield spanning the whole walkway) is written as the declarative
.nks terrain block, matching how bdx_author authors terrain.

Run (repo root, with the scene-oneshot nuka on PYTHONPATH):
    python examples/demo/bdx_oneshot_author.py [--granular-spacing S] [--rocks N]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random

import nuka
from bdx_oneshot_rope import write_rope_usda

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEX = os.path.join(REPO, "examples", "assets", "textures")
BDX = os.path.join(REPO, "examples", "scenes", "bdx_stand.nks")
OUT = os.path.join(REPO, "examples", "scenes", "bdx_oneshot.nks")
ROPE_USDA = os.path.join(REPO, "examples", "scenes", "bdx_oneshot_rope.usda")

HEAD = ("base/trunk_assembly/neck_pitch_assembly/head_pitch_to_yaw/"
        "neck_yaw_assembly/head_assembly")
TRUNK = "base/trunk_assembly"

# A big flat heightfield spanning the walkway (physics ground); tuned to cover
# x in [-1.5, 8.5], y in [-2.0, 2.5] so every zone stands on it.
# The physics heightfield sits 3 cm BELOW the visual dirt slab (top +0.002) so it
# never z-fights the slab in the render; the duck's capsule feet rest on the slab.
TERRAIN = {
    "name": "ground", "nrow": 100, "ncol": 45, "cell": 0.1,
    "origin": [-1.5, -2.0, 0.0], "base_z": -0.03, "grade_x": 0.0, "grade_y": 0.0,
    "ring_rise": 0.0, "ring_width": 0.0, "ring_platform": 0.0, "ring_count": 0,
    "bump_height": 0.0, "bump_cell": 0.0, "feature_cell": 0.0,
    "feature_margin": 0.0, "feature_seed": 0, "curric_levels": 0,
    "curric_types": 0, "image_path": "", "image_radius_x": 0.0,
    "image_radius_y": 0.0, "image_elevation_z": 0.0,
}


# Shared zone coordinates (the rigid stage bakes the door / trough geometry; the
# build-time media helpers place the cloth / granular against the SAME frames).
DOOR_X = 0.9
TROUGH = (1.4, 3.0, 0.45)        # x0, x1, half-y
GRANULAR_SPACING = 0.013


def tex(name, **kw):
    return dict(albedo_map=f"{TEX}/{name}/albedo.jpg",
               roughness_map=f"{TEX}/{name}/roughness.jpg",
               normal_map=f"{TEX}/{name}/normal.jpg", **kw)


def add_cloth(b):
    """Add the D01 fabric curtain: a perimeter-pinned XPBD sheet stretched across
    the doorway at head height. The pinned rim holds it steady while the duck's
    head bulges it up from below as it passes (the coupling beat). Perimeter pins
    (not a free drape on the lintel) because the general built-scene path wires
    cloth<->heightfield + cloth<->articulation contact but NOT cloth<->free-rigid
    box, so a free curtain would fall through the wooden lintel. Cooks alone (an
    XPBD medium cannot co-reside with the Zone B MPM granular)."""
    fid = b.add_material("cloth_fabric", base_color=[0.72, 0.34, 0.30], sheen=0.5,
                         **tex("fabric", uv_scale=5.0))
    b.add_media(nuka.MEDIA_CLOTH, nuka.MEDIA_METHOD_XPBD,
                cloth_nx=18, cloth_ny=26, cloth_spacing=0.02,
                cloth_origin=[DOOR_X, 0.0, 0.30], cloth_free=False,
                xpbd_particle_mass=0.004, xpbd_friction=0.9,
                xpbd_bend_alpha=0.05, xpbd_iters=30, render_material_id=fid)


def add_granular(b, spacing=GRANULAR_SPACING):
    """Add the D02 MLS-MPM Drucker-Prager granular bed filling the Zone B trough
    (model_kind 4). Cooks alone (MPM is its own particle mode)."""
    tx0, tx1, ty = TROUGH
    gid = b.add_material("gravel_bed", **tex("gravel", uv_scale=8.0))
    b.add_media(nuka.MEDIA_GRANULAR, nuka.MEDIA_METHOD_MLSMPM,
                fluid_min=[tx0 + 0.02, -ty + 0.02, 0.004],
                fluid_max=[tx1 - 0.02, ty - 0.02, 0.037],
                fluid_spacing=spacing, mpm_model_kind=4.0, mpm_dp_friction=34.0,
                mpm_dp_cohesion=0.0, mpm_density=1600.0, mpm_youngs=3.0e5,
                mpm_poisson=0.3, mpm_dx=max(2.0 * spacing, 0.02), mpm_substeps=25,
                mpm_floor_normal=[0.0, 0.0, 1.0], mpm_floor_d=0.0,
                mpm_floor_friction=0.6, render_material_id=gid)


def build(args):
    b = nuka.SceneBuilder.create(BDX)

    # -- render materials (appearance only; friction rides the shapes) ---------
    wood = b.add_material("wood_planks", **tex("wood_planks", uv_scale=2.5))
    gravel = b.add_material("gravel", **tex("gravel", uv_scale=6.0))
    dirt = b.add_material("dirt", **tex("dirt", uv_scale=3.0))
    fabric = b.add_material("fabric", base_color=[0.72, 0.34, 0.30], sheen=0.5,
                            **tex("fabric", uv_scale=5.0))
    stone = b.add_material("stone", base_color=[0.42, 0.40, 0.38], roughness=0.85)
    metal = b.add_material("metal", base_color=[0.55, 0.55, 0.58], metallic=0.9,
                           roughness=0.4)
    crate = b.add_material("crate", **tex("wood_planks", uv_scale=4.0))
    b.add_material("rope_cord", base_color=[0.28, 0.20, 0.13], roughness=0.9)

    # -- duck head + trunk colliders (they ship visual-only) -------------------
    # Kept small + separated (head-sphere bottom well above trunk-box top) so the
    # two added colliders never touch each other -- otherwise their mutual push
    # extends the neck and lifts the head off the body.
    b.add_collision_shape(HEAD, nuka.PRIMITIVE_SPHERE, dims=[0.04],
                          pos=[0.02, 0.0, 0.0], friction=0.8)
    b.add_collision_shape(TRUNK, nuka.PRIMITIVE_BOX, dims=[0.04, 0.035, 0.028],
                          pos=[0.0, 0.0, -0.005], friction=0.8)

    # -- visual ground: a large dirt slab whose TOP sits just above the physics
    # heightfield, so it reads as the textured walkway and fills to the horizon
    # (a finite ground would let the HDRI's grey lower hemisphere show past its rim) -
    b.add_rigid_primitive(nuka.PRIMITIVE_BOX, dims=[16.0, 16.0, 0.06],
                          pos=[3.0, 0.0, -0.058], static=True, material="dirt")

    # -- Zone A: stairs rising behind the spawn to a top platform --------------
    rise, run, half_w = 0.04, 0.14, 0.35
    for j in range(1, 6):
        top = j * rise
        xc = -(j - 0.5) * run
        b.add_rigid_primitive(nuka.PRIMITIVE_BOX,
                              dims=[run * 0.5, half_w, top * 0.5],
                              pos=[xc, 0.0, top * 0.5], static=True, material="wood_planks")
    b.add_rigid_primitive(nuka.PRIMITIVE_BOX, dims=[0.45, half_w, 0.10],
                          pos=[-0.70 - 0.45, 0.0, 0.10], static=True, material="wood_planks")

    # -- Zone A: door frame at x=0.9 (the fabric curtain is a BUILD-TIME medium;
    # see add_cloth -- an XPBD particle medium cannot co-cook with the Zone B MPM
    # granular, so the two particle showcases are cooked one at a time) ---------
    dx = DOOR_X
    for sy in (-0.35, 0.35):
        b.add_rigid_primitive(nuka.PRIMITIVE_BOX, dims=[0.03, 0.03, 0.25],
                              pos=[dx, sy, 0.25], static=True, material="wood_planks")
    b.add_rigid_primitive(nuka.PRIMITIVE_BOX, dims=[0.04, 0.40, 0.03],
                          pos=[dx, 0.0, 0.50], static=True, material="wood_planks")

    # -- Zone B: gravel trough rim (the granular bed is a BUILD-TIME medium; see
    # add_granular) -----------------------------------------------------------
    tx0, tx1, ty = TROUGH
    for sy in (-ty - 0.03, ty + 0.03):
        b.add_rigid_primitive(nuka.PRIMITIVE_BOX, dims=[(tx1 - tx0) * 0.5 + 0.05, 0.03, 0.03],
                              pos=[(tx0 + tx1) * 0.5, sy, 0.03], static=True, material="wood_planks")
    for sx in (tx0 - 0.03, tx1 + 0.03):
        b.add_rigid_primitive(nuka.PRIMITIVE_BOX, dims=[0.03, ty + 0.05, 0.03],
                              pos=[sx, 0.0, 0.03], static=True, material="wood_planks")

    # -- Zone C: ~N small free rigid bodies scattered on the walkway -----------
    rng = random.Random(7)
    g0 = 0.004                                # ground top (dirt slab) + margin
    for k in range(int(args.rocks)):
        x = rng.uniform(3.45, 4.05)          # a dense micro-cluster on the lane
        y = rng.uniform(-0.22, 0.22)
        kind = rng.random()
        if kind < 0.5:                       # pebble
            r = rng.uniform(0.018, 0.036)
            b.add_rigid_primitive(nuka.PRIMITIVE_SPHERE, dims=[r],
                                  pos=[x, y, g0 + r], mass=0.04, friction=0.9,
                                  material="stone")
        elif kind < 0.8:                     # bolt / screw
            r = rng.uniform(0.009, 0.014)
            hh = rng.uniform(0.024, 0.040)
            b.add_rigid_primitive(nuka.PRIMITIVE_CAPSULE, dims=[r, hh],
                                  pos=[x, y, g0 + r],
                                  quat=[0.707, 0.707, 0.0, 0.0], mass=0.02,
                                  friction=0.8, material="metal")
        else:                                # washer / chip
            e = rng.uniform(0.014, 0.024)
            b.add_rigid_primitive(nuka.PRIMITIVE_BOX, dims=[e, e, e * 0.5],
                                  pos=[x, y, g0 + e], mass=0.025, friction=0.8,
                                  material="metal")

    # -- Zone D: overhead beam + hanging rope + block --------------------------
    beam_x, beam_z = 5.0, 0.75
    for sy in (-0.4, 0.4):                    # two posts
        b.add_rigid_primitive(nuka.PRIMITIVE_BOX, dims=[0.035, 0.035, beam_z * 0.5],
                              pos=[beam_x, sy, beam_z * 0.5], static=True, material="wood_planks")
    b.add_rigid_primitive(nuka.PRIMITIVE_BOX, dims=[0.05, 0.44, 0.035],
                          pos=[beam_x, 0.0, beam_z + 0.03], static=True, material="wood_planks")
    write_rope_usda(ROPE_USDA, n_links=7, seg=0.055, radius=0.012,
                    link_mass=0.025, block=(0.08, 0.05, 0.035, 0.30))
    rope = nuka.SceneBuilder.create(ROPE_USDA)
    b.compose(rope, pos=[beam_x, 0.0, beam_z], attach_at="rope")
    rope.destroy()

    # -- set dressing: background crates kept OUT of the travel lane (y beyond
    # ~1.3 m) so the four zones read clearly; a low curb hints the walkway edge --
    for (cx, cy, s) in [(1.7, -1.35, 0.16), (2.6, 1.45, 0.13), (4.7, -1.4, 0.15),
                        (0.2, 1.4, 0.14), (5.7, 1.3, 0.17), (3.6, -1.55, 0.12),
                        (5.2, -1.45, 0.13), (1.1, 1.5, 0.15)]:
        b.add_rigid_primitive(nuka.PRIMITIVE_BOX, dims=[s, s, s],
                              pos=[cx, cy, s + 0.002], static=True, material="crate")

    # -- environment ----------------------------------------------------------
    b.set_environment(hdri=f"{TEX}/hdri/sky_2k.hdr", yaw_deg=-35.0, intensity=1.05,
                      use_scene_materials=True)

    b.save(OUT)
    b.destroy()

    # -- declarative terrain (big flat heightfield) + a visible rope skin ------
    doc = json.load(open(OUT))
    doc["terrain"] = [TERRAIN]
    # The composed rope ships collision-only capsules (the USD importer binds no
    # material); give them a render_material_id so the collision-primitive fallback
    # renders a dark cord + a wood block instead of bare grey.
    mats = list(doc["render_materials"].keys())
    cord_id, wood_id = mats.index("rope_cord"), mats.index("wood_planks")

    def skin_rope(node):
        if isinstance(node, list):
            for c in node:
                skin_rope(c)
            return
        cs = node.get("collision_shape")
        if cs is not None and node.get("name") == "ropegeom":
            cs["material_id"] = wood_id if cs.get("type") == "box" else cord_id
        for c in node.get("children", []):
            skin_rope(c)

    skin_rope(doc["tree"])
    with open(OUT, "w") as f:
        json.dump(doc, f)
    print(f"[oneshot] wrote {OUT} (continuous rigid stage, {args.rocks} rocks; "
          "cloth/granular media are added at build time -- see add_cloth/add_granular)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rocks", type=int, default=40)
    build(ap.parse_args())


if __name__ == "__main__":
    main()
