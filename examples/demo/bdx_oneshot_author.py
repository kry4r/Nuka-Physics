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
     medium filling a shallow flush tray dug into the dirt floor, round-grain skin,
     dirt shoulder ramps all around, one half-buried probe pebble on the walk line;
  C  x in [2.3, 3.2]: a thin light foot-loftable MLS-MPM debris fill appended to the
     Zone B bed on the SAME grid -- coarse grains the passing foot sweeps and lofts;
  D  x in [3.2, 4.6]: a portal frame at x=3.75 hanging an XPBD cord pinned under the
     beam with a rigid stone slab welded to its loaded end, dangling at duck-head
     height so the head strikes it and it swings in the x-z plane.

Dressing: plank walls both sides, a raised back-drop wall + far-end wall, overhead
rafters, wall studs, crates/barrels/shelves off the walk line, a tool board, HDRI
sky light. The duck (bdx_stand.nks) spawns on the platform; its head + trunk gain
colliding geoms via add_collision_shape.

The media (cloth XPBD + gravel/debris MLS-MPM + cable XPBD) are DECLARED in the
saved .nks and cook together into one world (ParticleMode::MpmXpbd). Build with
``bake_link_sdf=True`` so the duck feet gain visual-mesh SDFs and loft the debris.

Run (repo root): python examples/demo/bdx_oneshot_author.py
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os

import nuka
from nuka import materials, morphs, surfaces

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEX = os.path.join(REPO, "examples", "assets", "textures")
BDX = os.path.join(REPO, "examples", "scenes", "bdx_stand.nks")
OUT = os.path.join(REPO, "examples", "scenes", "bdx_oneshot.nks")
TRAINING_COARSE_OUT = os.path.join(
    REPO, "examples", "scenes", "bdx_oneshot_training_coarse.nks")

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
LINTEL_Z = (0.48, 0.56)            # lintel bottom/top faces.
CLOTH_SPACING = 0.02
CLOTH_NX, CLOTH_NY = 26, 28         # 26 along x (0.50 span straddling the lintel), 28 across y.
CLOTH_ORIGIN = [0.24, 0.0, 0.572]  # grid CENTER over the lintel (the cook centers on origin).
TROUGH = (0.95, 2.25, 0.35, -0.03)  # gravel fill x0, x1, half-y, floor z.
RECESS_X1 = 3.05                    # flush dirt recess spans [x0, RECESS_X1]: the debris
                                    # shares the ONE MLS-MPM floor and reads on-ground.
GRAVEL_SPACING = 0.013
PROBE_PEBBLE = (1.60, 0.06, 0.02)   # x, y, radius -- half-buried on the walk line.
BED_JITTER = 0.5                    # de-lattice the gravel so it never reads as a waffle.
LOFT_HEADROOM = 1.0                 # +z grid ceiling (m) so kicked debris has room.
ZONE_C = (2.47, 3.03, 0.18)         # x0, x1, half-y for the MPM debris fill.
DEBRIS_DZ = (0.006, 0.062)          # ~2 grain layers so the surface stacks above the floor.
DEBRIS_SPACING = 0.018              # fine debris grains: small render radius, less half-sink.
DEBRIS_JITTER = 0.5
BEAM_X, BEAM_Z = 3.75, 0.79         # portal frame plane and beam centre height.
CABLE_ANCHOR_Z = 0.745             # pinned bead just under the beam bottom face.
CABLE_END_Z = 0.44                 # loaded end == slab top; slab bottom ~0.29 at head height.
CABLE_SEGMENTS, CABLE_RADIUS = 14, 0.011
SLAB_HALF = (0.065, 0.11, 0.075)    # hanging stone: thin across travel, wide y, tall z.
WALL_Y, WALL_TOP = 0.55, 0.46       # near-side wall inner face / height (low: let sky in).
BACKDROP_Y, BACKDROP_TOP = 0.62, 0.85  # raised +y back-drop that blocks the sky horizon.
FAR_WALL_X = 4.72                   # far-end wall closing the +x horizon.
SCRIM_Y = -1.7                      # distant back wall well behind the -y camera lane.
SCRIM_TOP = 1.1                     # distant enclosure height that caps the open-side sky.
SCRIM_X = (-1.5, 5.7)              # -y back wall x-span, past both corridor ends.


@dataclasses.dataclass(frozen=True)
class GranularProfile:
    gravel_spacing: float
    debris_spacing: float
    substeps: int
    loft_headroom: float

    @property
    def dx(self):
        return max(2.0 * self.gravel_spacing, 0.02)


_GRANULAR_PROFILES = {
    "fine": GranularProfile(
        gravel_spacing=GRAVEL_SPACING,
        debris_spacing=DEBRIS_SPACING,
        substeps=14,
        loft_headroom=LOFT_HEADROOM,
    ),
    "training-coarse": GranularProfile(
        gravel_spacing=0.026,
        debris_spacing=0.026,
        substeps=7,
        loft_headroom=0.15,
    ),
}


def granular_profile(name):
    try:
        return _GRANULAR_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"unknown granular profile: {name}") from exc


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
    # Stone / metal tuned mid-low roughness so the specular env reflection reads.
    m["stone"] = b.add_material("stone", base_color=[0.20, 0.17, 0.14],
                                roughness=0.5, metallic=0.05)
    m["metal"] = b.add_material("metal", base_color=[0.55, 0.56, 0.60],
                                metallic=0.92, roughness=0.26)
    m["slab_stone"] = b.add_material("slab_stone", base_color=[0.26, 0.24, 0.22],
                                     metallic=0.25, roughness=0.45)
    m["debris"] = b.add_material("debris", base_color=[0.22, 0.21, 0.19],
                                 metallic=0.8, roughness=0.6)
    m["cord"] = b.add_material("rope_cord", base_color=[0.24, 0.17, 0.11],
                               roughness=0.85)
    m["crate"] = b.add_material("crate", base_color=[0.50, 0.36, 0.22],
                                **tex("wood_planks", uv_scale=4.0))
    m["board"] = b.add_material("board", base_color=[0.35, 0.26, 0.16],
                                **tex("wood_planks", uv_scale=3.0))
    return m


def add_cloth(b, fabric_id):
    """The doorway curtain: a fabric sheet centered over the lintel so it straddles
    the top rail. Its +x (back) edge is pinned behind the rail so the sheet cannot
    slip off the thin beam, while the long free front flap drapes down through the
    opening -- lifted by the passing head and swinging back."""
    b.add_media(nuka.MEDIA_CLOTH, nuka.MEDIA_METHOD_XPBD,
                cloth_nx=CLOTH_NX, cloth_ny=CLOTH_NY, cloth_spacing=CLOTH_SPACING,
                cloth_origin=list(CLOTH_ORIGIN), cloth_pin=nuka.CLOTH_PIN_EDGE_X1,
                xpbd_particle_mass=0.003, xpbd_friction=1.0,
                xpbd_bend_alpha=0.02, xpbd_iters=30,
                xpbd_aero_normal=0.8, xpbd_aero_tangent=0.2, xpbd_aero_max_dv=0.5,
                render_material_id=fabric_id)


def add_granular(b, mats, profile):
    """The Zone B gravel bed (base MLS-MPM Drucker-Prager fill) plus the Zone C
    light debris as a heterogeneous sub-fill on the SAME grid. The base material
    owns the shared grid scalars (dx / substeps / floor / loft headroom); the fill
    carries only its own constitutive, and the round-grain skin propagates to it."""
    tx0, tx1, ty, tz = TROUGH
    bed = morphs.GranularBed(min=(tx0 + 0.02, -ty + 0.02, tz + 0.002),
                             max=(tx1 - 0.02, ty - 0.02, 0.004),
                             spacing=profile.gravel_spacing,
                             position_jitter=BED_JITTER)
    base_mat = materials.Granular.MPM(youngs=3.0e5, poisson=0.3, density=1600.0,
                                      friction_angle=34.0, cohesion=0.0,
                                      dx=profile.dx, substeps=profile.substeps,
                                      floor_normal=(0.0, 0.0, 1.0), floor_d=tz,
                                      floor_friction=0.7,
                                      loft_headroom=profile.loft_headroom)
    grains = surfaces.Grains(round=True, radius_jitter=0.25, tint_jitter=0.15)
    kw = bed.media_geometry_kwargs()
    kw.update(base_mat.media_material_kwargs())
    kw.update(grains.media_kwargs())
    kw["render_material_id"] = mats["gravel"]
    media_id = b.add_media(kind=nuka.MEDIA_GRANULAR,
                           method=nuka.MEDIA_METHOD_MLSMPM, **kw)

    cx0, cx1, cy = ZONE_C
    dz0, dz1 = DEBRIS_DZ
    debris_mat = materials.Granular.MPM(youngs=1.5e5, poisson=0.3, density=600.0,
                                        friction_angle=22.0, cohesion=0.0)
    fill = morphs.MpmFill(min=(cx0, -cy, tz + dz0), max=(cx1, cy, tz + dz1),
                          spacing=profile.debris_spacing, material=debris_mat,
                          position_jitter=DEBRIS_JITTER)
    fkw = fill.fill_geometry_kwargs()
    fkw.update(debris_mat.fill_kwargs())
    fkw["render_material_id"] = mats["debris"]
    b.add_mpm_fill(media_id, **fkw)


def add_cable(b, mats):
    """The D04 hanging stone: an inextensible XPBD cord pinned under the beam with a
    rigid stone slab welded to its loaded end. The slab's top face sits at the cord
    end and hangs downward, dangling at duck-head height for the head-strike swing."""
    cord = materials.Cable.XPBD(mass=0.04, friction=0.4, distance_alpha=0.0,
                                bend_alpha=1.0e-3, iters=32)
    slab = morphs.CableSlab(half_extents=SLAB_HALF, mass=0.075, stiffness=1.0)
    cable = morphs.Cable(start=(BEAM_X, 0.0, CABLE_ANCHOR_Z),
                         end=(BEAM_X, 0.0, CABLE_END_Z), segments=CABLE_SEGMENTS,
                         radius=CABLE_RADIUS, pin="start", bend=False, slab=slab)
    kw = cable.media_geometry_kwargs()
    kw.update(cord.media_material_kwargs())
    kw["render_material_id"] = mats["cord"]
    kw["cable_slab_render_material_id"] = mats["slab_stone"]
    b.add_media(kind=nuka.MEDIA_CABLE, method=nuka.MEDIA_METHOD_XPBD, **kw)


def sbox(b, half, pos, mat, quat=None, friction=0.9):
    b.add_rigid_primitive(nuka.PRIMITIVE_BOX, dims=list(half), pos=list(pos),
                          quat=list(quat) if quat else [], static=True,
                          friction=friction, material=mat)


def build(args):
    profile = granular_profile(getattr(args, "profile", "fine"))
    spacing_override = getattr(args, "gravel_spacing", None)
    if spacing_override is not None:
        profile = dataclasses.replace(profile, gravel_spacing=spacing_override)
    b = nuka.SceneBuilder.create(BDX)
    mat_ids = add_materials(b)

    # -- duck head + trunk colliders (shipped visual-only; keep the two geoms
    # separated or their mutual push extends the neck) -------------------------
    b.add_collision_shape(HEAD, nuka.PRIMITIVE_SPHERE, dims=[0.04],
                          pos=[0.02, 0.0, 0.0], friction=0.8)
    b.add_collision_shape(TRUNK, nuka.PRIMITIVE_BOX, dims=[0.04, 0.035, 0.028],
                          pos=[0.0, 0.0, -0.005], friction=0.8)

    # -- floor: one big base slab (top -0.03) + packed-earth plates (top 0.00)
    # everywhere EXCEPT the gravel+debris recess span, which stays 3 cm deep ----
    sbox(b, [3.75, 1.5, 0.06], [2.25, 0.0, -0.09], "dirt")
    # a wide apron below the walk plates (and recess floor) hides the studio disc.
    sbox(b, [11.0, 8.0, 0.015], [4.0, 0.0, -0.06], "dirt")
    tx0, tx1, ty, tz = TROUGH
    rx1 = RECESS_X1
    sbox(b, [(tx0 + 1.5) * 0.5, 1.5, 0.015], [(tx0 - 1.5) * 0.5, 0.0, -0.015], "dirt")
    sbox(b, [(6.0 - rx1) * 0.5, 1.5, 0.015], [(6.0 + rx1) * 0.5, 0.0, -0.015], "dirt")
    for sy in (1.0, -1.0):  # recess side strips keep the lane walls' footing level
        cy = sy * (ty + 1.5) * 0.5
        sbox(b, [(rx1 - tx0) * 0.5, (1.5 - ty) * 0.5, 0.015], [(tx0 + rx1) * 0.5, cy, -0.015], "dirt")
    # dirt shoulder ramps: the recess rim reads as ground, not a planter box.
    sbox(b, [0.05, ty, 0.006], [tx0 + 0.03, 0.0, -0.013], "dirt", quat=yrot(12))
    sbox(b, [0.05, ty, 0.006], [rx1 - 0.03, 0.0, -0.013], "dirt", quat=yrot(-12))
    sbox(b, [(rx1 - tx0) * 0.5, 0.05, 0.006], [(tx0 + rx1) * 0.5, ty + 0.01, -0.013], "dirt", quat=xrot(10))
    sbox(b, [(rx1 - tx0) * 0.5, 0.05, 0.006], [(tx0 + rx1) * 0.5, -ty - 0.01, -0.013], "dirt", quat=xrot(-10))

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
    for sy in (1.0, -1.0):  # metal post hinges (specular hardware, clear of the drape)
        sbox(b, [0.05, 0.015, 0.03], [DOOR_X, sy * 0.34, 0.30], "metal")

    # -- Zone B: probe pebble half-buried in the gravel + rounded cobbles on the
    # gravel->debris transition (rigid stones bedding into the MPM medium) ------
    px, py, pr = PROBE_PEBBLE
    if not args.no_pebble:
        b.add_rigid_primitive(nuka.PRIMITIVE_SPHERE, dims=[pr],
                              pos=[px, py, tz + pr], mass=0.05, friction=0.9,
                              material="stone")
    for i in range(8):  # sparse rounded cobbles thinning toward Zone C
        r = 0.010 + 0.0006 * i
        x = 2.30 + 0.016 * i
        yy = 0.16 * math.sin(2.3 * i)
        b.add_rigid_primitive(nuka.PRIMITIVE_SPHERE, dims=[r], pos=[x, yy, tz + r],
                              mass=0.04, friction=0.9, material="stone")

    # -- Zone D: portal frame (the cord + slab are added with the media below) --
    for sy in (1.0, -1.0):
        sbox(b, [0.04, 0.04, 0.415], [BEAM_X, sy * 0.50, 0.415], "wood_planks")
    sbox(b, [0.05, 0.55, 0.04], [BEAM_X, 0.0, BEAM_Z], "wood_planks")

    # -- dressing: enclose so no framing sees dirt-to-horizon. A distant -y scrim +
    # taller -x/+x ends sit behind the camera lane; +y keeps its low wall + back-drop.
    wz = [-0.03, WALL_TOP]
    wh, wc = (wz[1] - wz[0]) * 0.5, (wz[0] + wz[1]) * 0.5
    sbox(b, [(4.6 + 0.3) * 0.5, 0.03, wh], [(4.6 - 0.3) * 0.5, WALL_Y + 0.03, wc], "wood_planks")
    for sx in (-0.15, 0.75, 1.65, 2.55, 3.45, 4.35):
        sbox(b, [0.04, 0.04, 0.245], [sx, WALL_Y - 0.04, 0.215], "wood_planks")
    bh, bcz = (BACKDROP_TOP + 0.03) * 0.5, (BACKDROP_TOP - 0.03) * 0.5
    sbox(b, [(FAR_WALL_X + 0.3) * 0.5, 0.03, bh], [(FAR_WALL_X - 0.3) * 0.5, BACKDROP_Y, bcz], "wood_planks")
    # distant -y back wall + taller/wider -x and +x ends: cap the open-side sky from
    # behind the lane (SCRIM_Y is well past every camera, so nothing clips the near field).
    sh, scz = (SCRIM_TOP + 0.03) * 0.5, (SCRIM_TOP - 0.03) * 0.5
    ey = (0.9 - SCRIM_Y) * 0.5, (0.9 + SCRIM_Y) * 0.5
    sbox(b, [(SCRIM_X[1] - SCRIM_X[0]) * 0.5, 0.04, sh], [(SCRIM_X[0] + SCRIM_X[1]) * 0.5, SCRIM_Y, scz], "wood_planks")
    sbox(b, [0.03, ey[0], sh], [FAR_WALL_X, ey[1], scz], "wood_planks")   # +x end
    sbox(b, [0.03, ey[0], sh], [-0.68, ey[1], scz], "wood_planks")        # -x end
    for (cx, cs) in [(-0.4, 0.16), (2.1, 0.19), (4.3, 0.17)]:  # distant -y silhouettes for depth
        sbox(b, [cs, cs, cs], [cx, SCRIM_Y + 0.24, cs], "crate")
    for rx in (0.5, 1.7, 2.9, 4.1):  # overhead rafters break the open sky
        sbox(b, [0.04, 0.62, 0.04], [rx, 0.03, 0.80], "wood_planks")
    # midground clutter off the walk line (all on the +y side, behind the action).
    for (cx, cy2, s) in [(0.62, 0.44, 0.10), (1.30, 0.46, 0.08),
                         (2.40, 0.45, 0.12), (4.35, 0.42, 0.13),
                         (0.15, 0.45, 0.09), (3.30, 0.45, 0.10)]:
        sbox(b, [s, s, s], [cx, cy2, s], "crate")
    sbox(b, [0.08, 0.08, 0.05], [3.30, 0.45, 0.25], "crate")  # a stacked crate
    for (sx, sz) in [(4.30, 0.14), (4.30, 0.30)]:  # a far-corner shelf (two boards)
        sbox(b, [0.22, 0.02, 0.01], [sx, 0.47, sz], "board")
    for (bx, by) in [(4.05, -0.42), (1.90, 0.47)]:  # barrels
        b.add_rigid_primitive(nuka.PRIMITIVE_CAPSULE, dims=[0.06, 0.075],
                              pos=[bx, by, 0.135], static=True, material="crate")
    sbox(b, [0.25, 0.008, 0.13], [2.55, WALL_Y - 0.012, 0.30], "board")  # tool board

    # -- media (all records live in the .nks; ONE MpmXpbd cook) -----------------
    if not args.no_cloth:
        add_cloth(b, mat_ids["fabric"])
    if not args.no_gravel:
        add_granular(b, mat_ids, profile)
    add_cable(b, mat_ids)

    # Beauty look levers: env-miss fill at the HDRI intensity, tamed exposure +
    # filmic grade, a softer sky sun disc, and Cook-Torrance specular env reflection.
    b.set_environment(hdri=f"{TEX}/hdri/sky_2k.hdr", yaw_deg=-35.0, intensity=1.3,
                      use_scene_materials=True, ibl_full_fill=True,
                      exposure_ev=0.0, grade=0.25, sun_disc=0.6, specular_env=True)
    b.save(args.out)
    b.destroy()
    post_process(args.out)
    print(f"[oneshot] wrote {args.out} (profile={getattr(args, 'profile', 'fine')} "
          f"media: cloth={not args.no_cloth} gravel+debris={not args.no_gravel} "
          "cable=True)")


def post_process(out):
    """Declarative touches on the saved .nks: spawn the duck on the platform, drop
    the heightfield (the dirt boxes ARE the ground), and lift the whole scene above
    the render's z=0 studio disc (a uniform, physics-invariant z shift)."""
    doc = json.load(open(out))
    doc["terrain"] = []

    def walk(node):
        if isinstance(node, list):
            for c in node:
                walk(c)
            return
        if node.get("name") == "base" and "rigid_body" in node:
            node["transform"]["pos"] = list(SPAWN)
        for c in node.get("children", []):
            walk(c)

    walk(doc["tree"])
    for node in doc["tree"]:
        tr = node.get("transform")
        if tr and len(tr.get("pos", [])) == 3:
            tr["pos"][2] += SCENE_LIFT
    for m in doc.get("media", []):
        if m["kind"] == "cloth":
            m["cloth_grid"]["origin"][2] += SCENE_LIFT
        elif m["kind"] == "granular":
            m["fluid_box"]["min"][2] += SCENE_LIFT
            m["fluid_box"]["max"][2] += SCENE_LIFT
            m["mpm"]["floor_d"] += SCENE_LIFT
            for fl in m.get("mpm_fills", []):
                fl["box"]["min"][2] += SCENE_LIFT
                fl["box"]["max"][2] += SCENE_LIFT
        elif m["kind"] == "cable":
            m["cable_line"]["start"][2] += SCENE_LIFT
            m["cable_line"]["end"][2] += SCENE_LIFT
    with open(out, "w") as f:
        json.dump(doc, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=sorted(_GRANULAR_PROFILES), default="fine")
    ap.add_argument("--gravel-spacing", type=float)
    ap.add_argument("--out")
    ap.add_argument("--no-cloth", action="store_true")
    ap.add_argument("--no-gravel", action="store_true")
    ap.add_argument("--no-pebble", action="store_true")
    args = ap.parse_args()
    if args.out is None:
        args.out = TRAINING_COARSE_OUT if args.profile == "training-coarse" else OUT
    build(args)


if __name__ == "__main__":
    main()
