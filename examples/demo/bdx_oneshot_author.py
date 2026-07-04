#!/usr/bin/env python3
"""Author the BDX one-shot workshop corridor: four coupled-media beats strung
along ONE compact indoor walkway (+x travel), saved as a self-contained
``examples/scenes/bdx_oneshot.nks`` (cook back with SceneBuilder.create().build()).

Layout (world metres, +z up):
  A  x in [-0.6, 0.9]: raised wooden departure platform (top 0.10), TWO descending
     steps, chunky stair stringers, and a wooden door frame at x=0.30 whose lintel
     carries a fabric curtain DRAPED over it (a free XPBD cloth held by contact +
     friction, nothing pinned);
  B  x in [0.9, 2.3]: a full-lane gravel bed -- an MLS-MPM Drucker-Prager granular
     medium filling a shallow flush tray dug into the dirt floor, dirt shoulder
     ramps all around, one half-buried probe pebble on the walk line;
  C  x in [2.3, 3.2]: ~45 tiny free rigid bodies (pebbles, bolts, washers, nuts)
     scattered dense on the lane for the macro kick beat;
  D  x in [3.2, 4.6]: a portal frame at x=3.75 hanging a stone slab at duck-head
     height from a revolute-Y rope chain + rigid V-yoke (two capsule legs on ONE
     link) -- reads as a two-rope suspension, topologically a tree.

Dressing: plank walls both sides + a back wall, wall studs, crates/barrel off the
walk line, a tool board, HDRI sky light. The duck (bdx_stand.nks) spawns on the
platform; its head + trunk gain colliding geoms via add_collision_shape.

Both media (cloth XPBD + gravel MLS-MPM) are DECLARED in the saved .nks and cook
together into one world (ParticleMode::MpmXpbd).

Run (repo root): python examples/demo/bdx_oneshot_author.py [--rocks 45]
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

# -- shared stage coordinates (gate / choreo / preview import these) -----------
SPAWN = [-0.20, 0.0, 0.31]          # duck base anchor on the platform (top 0.10).
BASE_OVER_FLOOR = 0.21              # duck base anchor height over the floor it stands on.
SCENE_LIFT = 0.05                   # sit the scene above the render's z=0 studio disc.
PLATFORM_TOP = 0.10
STEP_TOP = 0.05                     # the ONE intermediate step (two risers of 0.05).
STEP_X = (0.10, 0.26)               # step tread span; ground (z=0) beyond.
DOOR_X = 0.30                       # door frame plane (over the stair bottom).
LINTEL_Z = (0.48, 0.56)             # lintel bottom/top faces.
CLOTH_SPACING = 0.02
CLOTH_NX, CLOTH_NY = 17, 28         # 17 nodes along x (0.32 hang run), 28 across y.
CLOTH_ORIGIN = [0.175, 0.0, 0.572]  # +x edge line over the lintel top (the pin).
TROUGH = (0.95, 2.25, 0.35, -0.03)  # x0, x1, half-y, floor z (flush tray).
GRAVEL_SPACING = 0.009
PROBE_PEBBLE = (1.60, 0.06, 0.02)   # x, y, radius -- half-buried on the walk line.
ZONE_C = (2.45, 3.05, 0.20)         # x0, x1, half-y for the small-object cluster.
BEAM_X, BEAM_Z = 3.75, 0.79         # portal frame plane and beam centre height.
ROPE_SEG, ROPE_LINKS = 0.043, 7
SLAB_HALF = (0.025, 0.11, 0.065)    # x thickness, y width, z height (slab hangs ~0.27).
WALL_Y, WALL_TOP = 0.55, 0.46       # wall inner faces / height (low: let sky in).


def tex(name, **kw):
    return dict(albedo_map=f"{TEX}/{name}/albedo.jpg",
                roughness_map=f"{TEX}/{name}/roughness.jpg",
                normal_map=f"{TEX}/{name}/normal.jpg", **kw)


def yrot(deg):
    h = math.radians(deg) * 0.5
    return [math.cos(h), 0.0, math.sin(h), 0.0]


def xrot(deg):
    h = math.radians(deg) * 0.5
    return [math.cos(h), math.sin(h), 0.0, 0.0]


def add_materials(b):
    m = {}
    m["wood"] = b.add_material("wood_planks", base_color=[0.55, 0.40, 0.26],
                               **tex("wood_planks", uv_scale=2.5))
    m["dirt"] = b.add_material("dirt", base_color=[0.32, 0.25, 0.18],
                               roughness=1.0, **tex("dirt", uv_scale=3.0))
    m["gravel"] = b.add_material("gravel_bed", base_color=[0.55, 0.51, 0.45],
                                 **tex("gravel", uv_scale=8.0))
    m["fabric"] = b.add_material("fabric_curtain", base_color=[0.74, 0.32, 0.26],
                                 sheen=0.5, **tex("fabric", uv_scale=5.0))
    m["stone"] = b.add_material("stone", base_color=[0.20, 0.17, 0.14],
                                roughness=0.95)
    m["metal"] = b.add_material("metal", base_color=[0.55, 0.56, 0.60],
                                metallic=0.9, roughness=0.3)
    m["cord"] = b.add_material("rope_cord", base_color=[0.28, 0.20, 0.13],
                               roughness=0.9)
    m["crate"] = b.add_material("crate", base_color=[0.50, 0.36, 0.22],
                                **tex("wood_planks", uv_scale=4.0))
    m["board"] = b.add_material("board", base_color=[0.35, 0.26, 0.16],
                                **tex("wood_planks", uv_scale=3.0))
    return m


def add_cloth(b, fabric_id):
    """The D01 curtain: its +x node line is pinned over the lintel top (the
    general edge-pin mode); the sheet swings down and hangs to z~0.27, free to
    billow, be lifted by the passing head, and swing back. A pure friction drape
    provably slips off (the folding overhangs yank the sheet off the beam)."""
    b.add_media(nuka.MEDIA_CLOTH, nuka.MEDIA_METHOD_XPBD,
                cloth_nx=CLOTH_NX, cloth_ny=CLOTH_NY, cloth_spacing=CLOTH_SPACING,
                cloth_origin=list(CLOTH_ORIGIN), cloth_pin=nuka.CLOTH_PIN_EDGE_X1,
                xpbd_particle_mass=0.003, xpbd_friction=1.0,
                xpbd_bend_alpha=0.02, xpbd_iters=30,
                xpbd_aero_normal=0.8, xpbd_aero_tangent=0.2, xpbd_aero_max_dv=0.5,
                render_material_id=fabric_id)


def add_granular(b, gravel_id, spacing=GRAVEL_SPACING):
    """The D02 gravel: an MLS-MPM Drucker-Prager bed filling the flush tray."""
    tx0, tx1, ty, tz = TROUGH
    b.add_media(nuka.MEDIA_GRANULAR, nuka.MEDIA_METHOD_MLSMPM,
                fluid_min=[tx0 + 0.02, -ty + 0.02, tz + 0.002],
                fluid_max=[tx1 - 0.02, ty - 0.02, 0.004],
                fluid_spacing=spacing, mpm_model_kind=4.0, mpm_dp_friction=34.0,
                mpm_dp_cohesion=0.0, mpm_density=1600.0, mpm_youngs=3.0e5,
                mpm_poisson=0.3, mpm_dx=max(2.0 * spacing, 0.02), mpm_substeps=14,
                mpm_floor_normal=[0.0, 0.0, 1.0], mpm_floor_d=tz,
                mpm_floor_friction=0.7, render_material_id=gravel_id)


def sbox(b, half, pos, mat, quat=None, friction=0.9):
    b.add_rigid_primitive(nuka.PRIMITIVE_BOX, dims=list(half), pos=list(pos),
                          quat=list(quat) if quat else [], static=True,
                          friction=friction, material=mat)


def build(args):
    b = nuka.SceneBuilder.create(BDX)
    mat_ids = add_materials(b)

    # -- duck head + trunk colliders (shipped visual-only; keep the two geoms
    # separated or their mutual push extends the neck) -------------------------
    b.add_collision_shape(HEAD, nuka.PRIMITIVE_SPHERE, dims=[0.04],
                          pos=[0.02, 0.0, 0.0], friction=0.8)
    b.add_collision_shape(TRUNK, nuka.PRIMITIVE_BOX, dims=[0.04, 0.035, 0.028],
                          pos=[0.0, 0.0, -0.005], friction=0.8)

    # -- floor: one big base slab (top -0.03) + packed-earth plates (top 0.00)
    # everywhere EXCEPT the gravel tray span, which stays 3 cm deep ------------
    sbox(b, [3.75, 1.5, 0.06], [2.25, 0.0, -0.09], "dirt")
    # a wide apron below the walk plates (and tray floor) hides the studio disc.
    sbox(b, [11.0, 8.0, 0.015], [4.0, 0.0, -0.06], "dirt")
    tx0, tx1, ty, tz = TROUGH
    sbox(b, [(tx0 + 1.5) * 0.5, 1.5, 0.015], [(tx0 - 1.5) * 0.5, 0.0, -0.015], "dirt")
    sbox(b, [(6.0 - tx1) * 0.5, 1.5, 0.015], [(6.0 + tx1) * 0.5, 0.0, -0.015], "dirt")
    for sy in (1.0, -1.0):  # tray side strips keep the lane walls' footing level
        cy = sy * (ty + 1.5) * 0.5
        sbox(b, [(tx1 - tx0) * 0.5, (1.5 - ty) * 0.5, 0.015], [(tx0 + tx1) * 0.5, cy, -0.015], "dirt")
    # dirt shoulder ramps: the tray rim reads as ground, not a planter box.
    sbox(b, [0.05, ty, 0.006], [tx0 + 0.03, 0.0, -0.013], "dirt", quat=yrot(12))
    sbox(b, [0.05, ty, 0.006], [tx1 - 0.03, 0.0, -0.013], "dirt", quat=yrot(-12))
    sbox(b, [(tx1 - tx0) * 0.5, 0.05, 0.006], [(tx0 + tx1) * 0.5, ty + 0.01, -0.013], "dirt", quat=xrot(10))
    sbox(b, [(tx1 - tx0) * 0.5, 0.05, 0.006], [(tx0 + tx1) * 0.5, -ty - 0.01, -0.013], "dirt", quat=xrot(-10))

    # -- Zone A: platform + two descending risers + stringers + door -----------
    sbox(b, [0.35, 0.45, 0.05], [-0.25, 0.0, PLATFORM_TOP * 0.5], "wood_planks")
    sx0, sx1 = STEP_X
    sbox(b, [(sx1 - sx0) * 0.5, 0.35, STEP_TOP * 0.5],
         [(sx0 + sx1) * 0.5, 0.0, STEP_TOP * 0.5], "wood_planks")
    for sy in (1.0, -1.0):  # chunky slanted stair stringers
        sbox(b, [0.20, 0.025, 0.05], [0.20, sy * 0.44, 0.045], "wood_planks", quat=yrot(18))
    lz0, lz1 = LINTEL_Z
    for sy in (1.0, -1.0):  # door posts
        sbox(b, [0.04, 0.04, 0.275], [DOOR_X, sy * 0.38, 0.275], "wood_planks")
    sbox(b, [0.04, 0.45, (lz1 - lz0) * 0.5], [DOOR_X, 0.0, (lz0 + lz1) * 0.5],
         "wood_planks", friction=1.2)  # lintel: high friction holds the drape

    # -- Zone B: the probe pebble half-buried on the walk line -----------------
    px, py, pr = PROBE_PEBBLE
    if not args.no_pebble:
        b.add_rigid_primitive(nuka.PRIMITIVE_SPHERE, dims=[pr],
                              pos=[px, py, tz + pr], mass=0.05, friction=0.9,
                              material="stone")
    # rigid cobbles thinning toward Zone C plus the dense micro-object cluster.
    # ONE rejection sampler + ONE placed list spaces every free body so it beds
    # down isolated (a clustered pile of gram-scale parts never fully quiesces).
    cx0, cx1, cy = ZONE_C
    rng = random.Random(7)
    placed = [(px, py, pr)]

    def free_spot(extent, x0, x1, y0, y1, margin=0.013, tries=110):
        for _ in range(tries):
            x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
            if all((x - qx) ** 2 + (y - qy) ** 2 > (extent + qe + margin) ** 2
                   for qx, qy, qe in placed):
                placed.append((x, y, extent))
                return x, y
        return None, None

    for _ in range(10):  # sparse rounded stone cobbles on the D02->D03 transition
        r = rng.uniform(0.009, 0.014)
        x, y = free_spot(r, 2.32, 2.42, -0.25, 0.25)
        if x is None:
            continue
        b.add_rigid_primitive(nuka.PRIMITIVE_SPHERE, dims=[r], pos=[x, y, r],
                              mass=0.04, friction=0.9, material="stone")

    for _ in range(int(args.rocks)):
        kind = rng.random()
        if kind < 0.40:      # pebble: a rounded stone; the analytic sphere-vs-floor
                             # contact beds cleaner than a small box manifold
            r = rng.uniform(0.010, 0.016)
            x, y = free_spot(r, cx0, cx1, -cy, cy)
            if x is None:
                continue
            b.add_rigid_primitive(nuka.PRIMITIVE_SPHERE, dims=[r], pos=[x, y, r],
                                  mass=0.03, friction=0.9, material="stone")
        elif kind < 0.65:    # bolt/screw: a faceted metal prism (real hardware is
                             # faceted, not a smooth cylinder) resting on a face
            hl = rng.uniform(0.011, 0.015)
            hw = 0.005
            x, y = free_spot(math.hypot(hl, hw), cx0, cx1, -cy, cy)
            if x is None:
                continue
            # Spun about +z it stays flat on a face -> a clean face contact.
            a = rng.uniform(0.0, math.pi)
            b.add_rigid_primitive(nuka.PRIMITIVE_BOX, dims=[hl, hw, hw],
                                  pos=[x, y, hw],
                                  quat=[math.cos(a * 0.5), 0.0, 0.0, math.sin(a * 0.5)],
                                  mass=0.028, friction=0.8, material="metal")
        elif kind < 0.85:    # washer: a flat metal square (thick enough to bed flat)
            e = rng.uniform(0.008, 0.012)
            x, y = free_spot(e * 1.42, cx0, cx1, -cy, cy)
            if x is None:
                continue
            b.add_rigid_primitive(nuka.PRIMITIVE_BOX, dims=[e, e, 0.003],
                                  pos=[x, y, 0.003], mass=0.016, friction=0.8,
                                  material="metal")
        else:                # nut: a small squat metal block
            e = rng.uniform(0.006, 0.008)
            x, y = free_spot(e * 1.42, cx0, cx1, -cy, cy)
            if x is None:
                continue
            b.add_rigid_primitive(nuka.PRIMITIVE_BOX, dims=[e, e, e * 0.75],
                                  pos=[x, y, e * 0.75], mass=0.02, friction=0.8,
                                  material="metal")

    # -- Zone D: portal frame + rope chain + V-yoke + stone slab ----------------
    for sy in (1.0, -1.0):
        sbox(b, [0.04, 0.04, 0.415], [BEAM_X, sy * 0.50, 0.415], "wood_planks")
    sbox(b, [0.05, 0.55, 0.04], [BEAM_X, 0.0, BEAM_Z], "wood_planks")
    write_rope_usda(ROPE_USDA, n_links=ROPE_LINKS, seg=ROPE_SEG, radius=0.008,
                    link_mass=0.02, yoke=(SLAB_HALF[1], 0.046, 0.03),
                    slab=(*SLAB_HALF, 0.6))
    rope = nuka.SceneBuilder.create(ROPE_USDA)
    b.compose(rope, pos=[BEAM_X, 0.0, BEAM_Z], attach_at="rope")
    rope.destroy()

    # -- dressing: walls / studs / crates / barrel / tool board -----------------
    # The camera side (-y) is left open (the classic open fourth wall); the far
    # (+y) plank wall + studs + a low back wall carry the workshop backdrop.
    wz = [-0.03, WALL_TOP]
    wh, wc = (wz[1] - wz[0]) * 0.5, (wz[0] + wz[1]) * 0.5
    sbox(b, [(4.6 + 0.3) * 0.5, 0.03, wh], [(4.6 - 0.3) * 0.5, WALL_Y + 0.03, wc], "wood_planks")
    sbox(b, [0.03, 0.61, wh], [-0.66, 0.32, wc], "wood_planks")
    for sx in (-0.15, 0.75, 1.65, 2.55, 3.45, 4.35):
        sbox(b, [0.04, 0.04, 0.245], [sx, WALL_Y - 0.04, 0.215], "wood_planks")
    for (cx, cy2, s) in [(0.62, 0.44, 0.10), (1.30, 0.46, 0.08),
                         (2.40, 0.45, 0.12), (4.35, 0.42, 0.13)]:
        sbox(b, [s, s, s], [cx, cy2, s], "crate")
    b.add_rigid_primitive(nuka.PRIMITIVE_CAPSULE, dims=[0.06, 0.075],
                          pos=[4.05, -0.42, 0.135], static=True, material="crate")
    sbox(b, [0.25, 0.008, 0.13], [2.55, WALL_Y - 0.012, 0.30], "board")

    # -- media (both records live in the .nks; ONE MpmXpbd cook) ----------------
    if not args.no_cloth:
        add_cloth(b, mat_ids["fabric"])
    if not args.no_gravel:
        add_granular(b, mat_ids["gravel"], spacing=args.gravel_spacing)

    b.set_environment(hdri=f"{TEX}/hdri/sky_2k.hdr", yaw_deg=-35.0, intensity=1.3,
                      use_scene_materials=True)
    b.save(args.out)
    b.destroy()
    post_process(args.out)
    print(f"[oneshot] wrote {args.out} ({args.rocks} micro objects; media: "
          f"cloth={not args.no_cloth} gravel={not args.no_gravel})")


def post_process(out):
    """Declarative touches on the saved .nks: duck spawn on the platform, no
    heightfield (the dirt floor boxes ARE the ground), and a render skin for the
    composed rope (importer collision capsules ship no material)."""
    doc = json.load(open(out))
    doc["terrain"] = []
    mats = list(doc["render_materials"].keys())
    cord, stone = mats.index("rope_cord"), mats.index("stone")

    def walk(node):
        if isinstance(node, list):
            for c in node:
                walk(c)
            return
        name = node.get("name", "")
        if name == "base" and "rigid_body" in node:
            node["transform"]["pos"] = list(SPAWN)
        if name.startswith("rope"):
            cs = node.get("collision_shape")
            if cs is not None:
                cs["material_id"] = stone if cs.get("type") == "box" else cord
        for c in node.get("children", []):
            walk(c)

    walk(doc["tree"])
    # lift the whole scene above the render's z=0 studio-floor disc so the dirt
    # floor renders instead of Z-fighting it (a uniform, physics-invariant shift).
    for node in doc["tree"]:
        tr = node.get("transform")
        if tr and len(tr.get("pos", [])) == 3:
            tr["pos"][2] += SCENE_LIFT
    for m in doc["media"]:
        if m["kind"] == "cloth":
            m["cloth_grid"]["origin"][2] += SCENE_LIFT
        elif m["kind"] == "granular":
            m["fluid_box"]["min"][2] += SCENE_LIFT
            m["fluid_box"]["max"][2] += SCENE_LIFT
            m["mpm"]["floor_d"] += SCENE_LIFT
    with open(out, "w") as f:
        json.dump(doc, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rocks", type=int, default=45)
    ap.add_argument("--gravel-spacing", type=float, default=GRAVEL_SPACING)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-cloth", action="store_true")
    ap.add_argument("--no-gravel", action="store_true")
    ap.add_argument("--no-pebble", action="store_true")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
