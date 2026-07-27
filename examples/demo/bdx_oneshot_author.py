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
import copy
import dataclasses
import json
import math
import os
import shutil

import nuka
from nuka import materials, morphs, surfaces

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEX = os.path.join(REPO, "examples", "assets", "textures")
BDX = os.path.join(REPO, "examples", "scenes", "bdx_stand.nks")
GO2 = os.path.join(REPO, "examples", "scenes", "go2.nks")
OUT = os.path.join(REPO, "examples", "scenes", "bdx_oneshot.nks")
TRAINING_COARSE_OUT = os.path.join(
    REPO, "examples", "scenes", "bdx_oneshot_training_coarse.nks")

HEAD = ("base/trunk_assembly/neck_pitch_assembly/head_pitch_to_yaw/"
        "neck_yaw_assembly/head_assembly")
TRUNK = "base/trunk_assembly"

# -- shared stage coordinates (gate / choreo / preview import these) -----------
SPAWN = [-0.20, 0.0, 0.31]          # duck base anchor on the platform (top 0.10).
GO2_SPAWN = [-0.45, 0.0, 0.425]     # 2-3 mm foot clearance in the baked crouch.
GO2_CROUCH = {
    "FL_hip_joint": 0.1, "FL_thigh_joint": 0.8, "FL_calf_joint": -1.5,
    "FR_hip_joint": -0.1, "FR_thigh_joint": 0.8, "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.1, "RL_thigh_joint": 1.0, "RL_calf_joint": -1.5,
    "RR_hip_joint": -0.1, "RR_thigh_joint": 1.0, "RR_calf_joint": -1.5,
}
GO2_FOOT_NAMES = ("FL", "FR", "RL", "RR")
GO2_FOOT_RADIUS = 0.022
GO2_FOOT_LOCAL = [0.0, 0.0, -0.213]
BASE_OVER_FLOOR = 0.21              # duck base anchor height over the floor it stands on.
SCENE_LIFT = 0.05                   # sit the scene above the render's z=0 studio disc.
PLATFORM_TOP = 0.10
PLATFORM_X = (-1.10, 0.10)          # long enough for the 0.73 m Go2 collision hull.
STEP_TOP = 0.05                     # the ONE intermediate step (two risers of 0.05).
STEP_X = (0.10, 0.26)               # step tread span; ground (z=0) beyond.
DOOR_X = 0.30                       # door frame plane (over the stair bottom).
LANE_HALF_Y = 0.40                  # preserve a 0.80 m collision-free centre lane.
DOOR_POST_Y = 0.44                  # 0.80 m clear opening for the 0.39 m-wide Go2.
LINTEL_Z = (0.59, 0.67)            # clears the 0.43 m-tall crouched Go2 hull.
CLOTH_SPACING = 0.02
CLOTH_NX, CLOTH_NY = 26, 28         # 26 along x (0.50 span straddling the lintel), 28 across y.
CLOTH_ORIGIN = [0.24, 0.0, 0.682]  # grid CENTER just above the raised lintel.
TROUGH = (0.95, 2.25, LANE_HALF_Y, -0.03)  # Go2-width gravel/debris lane.
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
GO2_CABLE_END_Z = 0.570            # slab top; bottom ~0.400 grazes the upper forehead shell.
CABLE_SEGMENTS, CABLE_RADIUS = 14, 0.011
SLAB_HALF = (0.065, 0.11, 0.075)    # hanging stone: thin across travel, wide y, tall z.
GO2_SLAB_HALF = (0.075, 0.27, 0.085)  # broad enough that the Go2 cannot sidestep it.
GO2_CABLE_PARTICLE_MASS = 0.008    # per bead: ~0.12 kg cord instead of 0.60 kg.
GO2_SLAB_CORNER_MASS = 0.0125      # per corner: 0.10 kg broad lightweight panel.
WALL_Y, WALL_TOP = 0.55, 0.46       # near-side wall inner face / height (low: let sky in).
DRESSING_INNER_Y = 0.52             # props do not narrow the 1.04 m main corridor.
BACKDROP_Y, BACKDROP_TOP = 0.62, 0.85  # raised +y back-drop that blocks the sky horizon.
FAR_WALL_X = 4.72                   # far-end wall closing the +x horizon.
SCRIM_Y = -1.7                      # distant back wall well behind the -y camera lane.
SCRIM_TOP = 1.1                     # distant enclosure height that caps the open-side sky.
SCRIM_X = (-1.5, 5.7)              # -y back wall x-span, past both corridor ends.
START_WALL_X = -1.18               # leave a useful run-up behind the Go2 spawn.

# Go2-specific transverse scale.  The original workshop dimensions were framed
# around BDX and made the larger quadruped read as wedged between walls/props.
# Keep the longitudinal obstacle sequence unchanged for the learned policy, but
# widen the walk surfaces, media tray, doorway, portal, and visual enclosure.
GO2_LANE_HALF_Y = 0.58
GO2_DOOR_POST_Y = 0.62
GO2_PLATFORM_HALF_Y = 0.68
GO2_STEP_HALF_Y = 0.56
GO2_STRINGER_Y = 0.64
GO2_DEBRIS_HALF_Y = 0.32
GO2_PORTAL_POST_Y = 0.70
GO2_PORTAL_BEAM_HALF_Y = 0.76
GO2_WALL_Y = 0.82
GO2_DRESSING_INNER_Y = 0.78
GO2_BACKDROP_Y = 0.92
GO2_SCRIM_Y = -2.40


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


def add_cloth(b, fabric_id, robot="bdx"):
    """The doorway curtain: a fabric sheet centered over the lintel so it straddles
    the top rail. Its +x (back) edge is pinned behind the rail so the sheet cannot
    slip off the thin beam, while the long free front flap drapes down through the
    opening -- lifted by the passing head and swinging back."""
    # The original BDX cinematic curtain is intentionally heavy.  At 728
    # particles x 3 g it weighs about 2.2 kg, which is not a plausible hanging
    # fabric obstacle for Go2.  Keep the full-length colliding sheet, but use a
    # canvas-like 0.36 kg total mass and moderate friction so the head must push
    # it aside instead of meeting a quasi-rigid wall.
    particle_mass = 0.0005 if robot == "go2" else 0.003
    friction = 0.6 if robot == "go2" else 1.0
    b.add_media(nuka.MEDIA_CLOTH, nuka.MEDIA_METHOD_XPBD,
                cloth_nx=CLOTH_NX, cloth_ny=CLOTH_NY, cloth_spacing=CLOTH_SPACING,
                cloth_origin=list(CLOTH_ORIGIN), cloth_pin=nuka.CLOTH_PIN_EDGE_X1,
                xpbd_particle_mass=particle_mass, xpbd_friction=friction,
                xpbd_bend_alpha=0.02, xpbd_iters=30,
                xpbd_aero_normal=0.8, xpbd_aero_tangent=0.2, xpbd_aero_max_dv=0.5,
                render_material_id=fabric_id)


def add_granular(b, mats, profile, *, trough=TROUGH, zone_c=ZONE_C):
    """The Zone B gravel bed (base MLS-MPM Drucker-Prager fill) plus the Zone C
    light debris as a heterogeneous sub-fill on the SAME grid. The base material
    owns the shared grid scalars (dx / substeps / floor / loft headroom); the fill
    carries only its own constitutive, and the round-grain skin propagates to it."""
    tx0, tx1, ty, tz = trough
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

    cx0, cx1, cy = zone_c
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


def add_cable(b, mats, robot="bdx"):
    """The D04 hanging stone: an inextensible XPBD cord pinned under the beam with a
    rigid stone slab welded to its loaded end. The slab's top face sits at the cord
    end and hangs downward, dangling at duck-head height for the head-strike swing."""
    go2 = robot == "go2"
    cord = materials.Cable.XPBD(
                                mass=(GO2_CABLE_PARTICLE_MASS if go2 else 0.04),
                                friction=(0.05 if go2 else 0.4), distance_alpha=0.0,
                                bend_alpha=1.0e-3, iters=32)
    slab_half = GO2_SLAB_HALF if go2 else SLAB_HALF
    slab = morphs.CableSlab(
        half_extents=slab_half,
        mass=(GO2_SLAB_CORNER_MASS if go2 else 0.075), stiffness=1.0)
    end_z = GO2_CABLE_END_Z if go2 else CABLE_END_Z
    cable = morphs.Cable(start=(BEAM_X, 0.0, CABLE_ANCHOR_Z),
                         end=(BEAM_X, 0.0, end_z), segments=CABLE_SEGMENTS,
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
    robot = getattr(args, "robot", "bdx")
    if robot not in ("bdx", "go2"):
        raise ValueError(f"unknown robot {robot!r}")
    spawn = SPAWN if robot == "bdx" else GO2_SPAWN
    go2 = robot == "go2"
    lane_half_y = GO2_LANE_HALF_Y if go2 else LANE_HALF_Y
    trough = (TROUGH[0], TROUGH[1], lane_half_y, TROUGH[3])
    zone_c = (ZONE_C[0], ZONE_C[1],
              GO2_DEBRIS_HALF_Y if go2 else ZONE_C[2])
    door_post_y = GO2_DOOR_POST_Y if go2 else DOOR_POST_Y
    platform_half_y = GO2_PLATFORM_HALF_Y if go2 else 0.45
    step_half_y = GO2_STEP_HALF_Y if go2 else 0.35
    stringer_y = GO2_STRINGER_Y if go2 else 0.44
    portal_post_y = GO2_PORTAL_POST_Y if go2 else 0.50
    portal_beam_half_y = GO2_PORTAL_BEAM_HALF_Y if go2 else 0.55
    wall_y = GO2_WALL_Y if go2 else WALL_Y
    dressing_inner_y = GO2_DRESSING_INNER_Y if go2 else DRESSING_INNER_Y
    backdrop_y = GO2_BACKDROP_Y if go2 else BACKDROP_Y
    scrim_y = GO2_SCRIM_Y if go2 else SCRIM_Y
    b = nuka.SceneBuilder.create(BDX if robot == "bdx" else GO2)
    mat_ids = add_materials(b)

    if robot == "bdx":
        b.add_collision_shape(HEAD, nuka.PRIMITIVE_SPHERE, dims=[0.04],
                              pos=[0.02, 0.0, 0.0], friction=0.8)
        b.add_collision_shape(TRUNK, nuka.PRIMITIVE_BOX, dims=[0.04, 0.035, 0.028],
                              pos=[0.0, 0.0, -0.005], friction=0.8)

    # -- floor: one big base slab (top -0.03) + packed-earth plates (top 0.00)
    # everywhere EXCEPT the gravel+debris recess span, which stays 3 cm deep ----
    sbox(b, [3.75, 1.5, 0.06], [2.25, 0.0, -0.09], "dirt")
    # a wide apron below the walk plates (and recess floor) hides the studio disc.
    sbox(b, [11.0, 8.0, 0.015], [4.0, 0.0, -0.06], "dirt")
    tx0, tx1, ty, tz = trough
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
    platform_half = (PLATFORM_X[1] - PLATFORM_X[0]) * 0.5
    platform_x = (PLATFORM_X[0] + PLATFORM_X[1]) * 0.5
    sbox(b, [platform_half, platform_half_y, 0.05],
         [platform_x, 0.0, PLATFORM_TOP * 0.5], "wood_planks")
    sx0, sx1 = STEP_X
    sbox(b, [(sx1 - sx0) * 0.5, step_half_y, STEP_TOP * 0.5],
         [(sx0 + sx1) * 0.5, 0.0, STEP_TOP * 0.5], "wood_planks")
    for sy in (1.0, -1.0):  # chunky slanted stair stringers
        sbox(b, [0.20, 0.025, 0.05], [0.20, sy * stringer_y, 0.045],
             "wood_planks", quat=yrot(18))
    lz0, lz1 = LINTEL_Z
    post_half = lz0 * 0.5
    for sy in (1.0, -1.0):  # door posts
        sbox(b, [0.04, 0.04, post_half],
             [DOOR_X, sy * door_post_y, post_half], "wood_planks")
    sbox(b, [0.04, door_post_y + 0.05, (lz1 - lz0) * 0.5],
         [DOOR_X, 0.0, (lz0 + lz1) * 0.5],
         "wood_planks", friction=1.2)  # lintel: high friction holds the drape
    for sy in (1.0, -1.0):  # metal post hinges (specular hardware, clear of the drape)
        sbox(b, [0.05, 0.015, 0.03],
             [DOOR_X, sy * (door_post_y - 0.04), 0.30], "metal")

    # -- Zone B: probe pebble half-buried in the gravel + rounded cobbles on the
    # gravel->debris transition (rigid stones bedding into the MPM medium) ------
    px, py, pr = PROBE_PEBBLE
    if not args.no_pebble:
        b.add_rigid_primitive(nuka.PRIMITIVE_SPHERE, dims=[pr],
                              pos=[px, py, tz + pr], mass=0.05, friction=0.9,
                              material="stone")
    # This scenic BDX close-up row spans only 11 cm longitudinally.  The Go2
    # scene keeps the isolated probe pebble and real gravel/debris media, but
    # omits this dense duplicate overlay so a whole stance cannot land on it at
    # once during the narrow-lane traversal.
    if robot == "bdx":
        for i in range(8):  # sparse rounded cobbles thinning toward Zone C
            r = 0.010 + 0.0006 * i
            x = 2.30 + 0.016 * i
            yy = 0.16 * math.sin(2.3 * i)
            b.add_rigid_primitive(
                nuka.PRIMITIVE_SPHERE, dims=[r], pos=[x, yy, tz + r],
                mass=0.04, friction=0.9, material="stone")

    # -- Zone D: portal frame (the cord + slab are added with the media below) --
    for sy in (1.0, -1.0):
        sbox(b, [0.04, 0.04, 0.415],
             [BEAM_X, sy * portal_post_y, 0.415], "wood_planks")
    sbox(b, [0.05, portal_beam_half_y, 0.04],
         [BEAM_X, 0.0, BEAM_Z], "wood_planks")

    # -- dressing: enclose so no framing sees dirt-to-horizon. A distant -y scrim +
    # taller -x/+x ends sit behind the camera lane; +y keeps its low wall + back-drop.
    wz = [-0.03, WALL_TOP]
    wh, wc = (wz[1] - wz[0]) * 0.5, (wz[0] + wz[1]) * 0.5
    sbox(b, [(4.6 + 0.3) * 0.5, 0.03, wh],
         [(4.6 - 0.3) * 0.5, wall_y + 0.03, wc], "wood_planks")
    for sx in (-0.15, 0.75, 1.65, 2.55, 3.45, 4.35):
        sbox(b, [0.04, 0.04, 0.245],
             [sx, wall_y - 0.04, 0.215], "wood_planks")
    bh, bcz = (BACKDROP_TOP + 0.03) * 0.5, (BACKDROP_TOP - 0.03) * 0.5
    sbox(b, [(FAR_WALL_X + 0.3) * 0.5, 0.03, bh],
         [(FAR_WALL_X - 0.3) * 0.5, backdrop_y, bcz], "wood_planks")
    # distant -y back wall + taller/wider -x and +x ends: cap the open-side sky from
    # behind the lane (SCRIM_Y is well past every camera, so nothing clips the near field).
    sh, scz = (SCRIM_TOP + 0.03) * 0.5, (SCRIM_TOP - 0.03) * 0.5
    outer_y = 1.20 if go2 else 0.90
    ey = (outer_y - scrim_y) * 0.5, (outer_y + scrim_y) * 0.5
    sbox(b, [(SCRIM_X[1] - SCRIM_X[0]) * 0.5, 0.04, sh],
         [(SCRIM_X[0] + SCRIM_X[1]) * 0.5, scrim_y, scz], "wood_planks")
    sbox(b, [0.03, ey[0], sh], [FAR_WALL_X, ey[1], scz], "wood_planks")   # +x end
    sbox(b, [0.03, ey[0], sh], [START_WALL_X, ey[1], scz], "wood_planks")  # -x end
    for (cx, cs) in [(-0.4, 0.16), (2.1, 0.19), (4.3, 0.17)]:  # distant -y silhouettes for depth
        sbox(b, [cs, cs, cs], [cx, scrim_y + 0.24, cs], "crate")
    for rx in (0.5, 1.7, 2.9, 4.1):  # overhead rafters break the open sky
        sbox(b, [0.04, wall_y + 0.12, 0.04], [rx, 0.03, 0.80], "wood_planks")
    # midground clutter off the walk line (all on the +y side, behind the action).
    for (cx, cy2, s) in [(0.62, 0.44, 0.10), (1.30, 0.46, 0.08),
                         (2.40, 0.45, 0.12), (4.35, 0.42, 0.13),
                         (0.15, 0.45, 0.09), (3.30, 0.45, 0.10)]:
        # The original BDX dressing intruded as far as y=0.29.  Keep the props
        # visually against the +y wall without narrowing the 1.04 m main lane;
        # the doorway/trough remain the deliberate 0.80 m bottlenecks.
        cy2 = max(cy2, dressing_inner_y + s)
        sbox(b, [s, s, s], [cx, cy2, s], "crate")
    sbox(b, [0.08, 0.08, 0.05],
         [3.30, dressing_inner_y + 0.10, 0.25],
         "crate")  # a stacked crate
    for (sx, sz) in [(4.30, 0.14), (4.30, 0.30)]:  # a far-corner shelf (two boards)
        sbox(b, [0.22, 0.02, 0.01],
             [sx, wall_y - 0.08, sz], "board")
    barrel_y = wall_y + 0.08
    for (bx, by) in [(4.05, -barrel_y), (1.90, barrel_y)]:
        b.add_rigid_primitive(nuka.PRIMITIVE_CAPSULE, dims=[0.06, 0.075],
                              pos=[bx, by, 0.135], static=True, material="crate")
    sbox(b, [0.25, 0.008, 0.13],
         [2.55, wall_y - 0.012, 0.30], "board")  # tool board

    # -- media (all records live in the .nks; ONE MpmXpbd cook) -----------------
    if not args.no_cloth:
        add_cloth(b, mat_ids["fabric"], robot)
    if not args.no_gravel:
        add_granular(b, mat_ids, profile, trough=trough, zone_c=zone_c)
    add_cable(b, mat_ids, robot)

    # Beauty look levers: env-miss fill at the HDRI intensity, tamed exposure +
    # filmic grade, a softer sky sun disc, and Cook-Torrance specular env reflection.
    b.set_environment(hdri=f"{TEX}/hdri/sky_2k.hdr", yaw_deg=-35.0, intensity=1.3,
                      use_scene_materials=True, ibl_full_fill=True,
                      exposure_ev=0.0, grade=0.25, sun_disc=0.6, specular_env=True)
    b.save(args.out)
    b.destroy()
    post_process(args.out, spawn, robot)
    print(f"[oneshot] wrote {args.out} robot={robot} "
          f"(profile={getattr(args, 'profile', 'fine')} "
          f"media: cloth={not args.no_cloth} gravel+debris={not args.no_gravel} "
          "cable=True)")


def _set_go2_crouch(node):
    count = 0
    joint = node.get("joint")
    if joint and joint.get("name") in GO2_CROUCH:
        joint["initial_position"] = GO2_CROUCH[joint["name"]]
        count += 1
    for child in node.get("children", []):
        count += _set_go2_crouch(child)
    return count


def _fix_go2_feet(node):
    count = 0
    shape = node.get("collision_shape")
    if node.get("name") in GO2_FOOT_NAMES and shape:
        shape["type"] = "sphere"
        shape["radius"] = GO2_FOOT_RADIUS
        shape["half_extents"] = [GO2_FOOT_RADIUS] * 3
        shape["half_height"] = GO2_FOOT_RADIUS
        shape["local"] = {
            "pos": list(GO2_FOOT_LOCAL),
            "quat": [1.0, 0.0, 0.0, 0.0],
        }
        count += 1
    for child in node.get("children", []):
        count += _fix_go2_feet(child)
    return count


def _retarget_go2_meshes(node, output_nks):
    """Keep the source Go2 mesh indices while changing only the pack basename.

    SceneBuilder.save() currently groups articulation children before visual
    children.  Reusing the original .nka after that reorder renumbers MESH/i and
    binds each visual (and baked link SDF) to the wrong link.  The source subtree
    already has the canonical mesh-index mapping, so preserve it byte-for-byte
    apart from the output pack basename.
    """
    count = 0
    new_pack = os.path.splitext(os.path.basename(output_nks))[0] + ".nka#"
    visual = node.get("visual_mesh")
    if visual and isinstance(visual.get("mesh"), str):
        old = visual["mesh"]
        if ".nka#" in old:
            visual["mesh"] = new_pack + old.split(".nka#", 1)[1]
            count += 1
    for child in node.get("children", []):
        count += _retarget_go2_meshes(child, output_nks)
    return count


def post_process(out, spawn=SPAWN, robot="bdx"):
    """Apply robot spawn details and the uniform scene lift to the saved NKS."""
    doc = json.load(open(out))
    doc["terrain"] = []

    if robot == "go2":
        source = json.load(open(GO2))
        source_robot = copy.deepcopy(source["tree"][0])
        assert source_robot.get("name") == "base", \
            "source Go2 root is not the base articulation"
        feet = _fix_go2_feet(source_robot)
        joints = _set_go2_crouch(source_robot)
        meshes = _retarget_go2_meshes(source_robot, out)
        assert feet == len(GO2_FOOT_NAMES), \
            f"expected {len(GO2_FOOT_NAMES)} Go2 feet, rewrote {feet}"
        assert joints == len(GO2_CROUCH), \
            f"expected {len(GO2_CROUCH)} Go2 joints, set {joints}"
        assert meshes == 33, f"expected 33 Go2 visual meshes, retargeted {meshes}"
        # Restore the canonical child traversal that the unchanged .nka expects.
        doc["tree"][0] = source_robot

    def walk(node):
        if isinstance(node, list):
            for c in node:
                walk(c)
            return
        if node.get("name") == "base" and "rigid_body" in node:
            node["transform"]["pos"] = list(spawn)
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
    if robot == "go2":
        # The restored source subtree indexes the canonical source mesh pack.
        # Overwrite SceneBuilder.save()'s traversal-repacked NKA so those indices
        # and the binary pack agree exactly.
        source_nka = os.path.splitext(GO2)[0] + ".nka"
        output_nka = os.path.splitext(out)[0] + ".nka"
        shutil.copyfile(source_nka, output_nka)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", choices=("bdx", "go2"), default="bdx")
    ap.add_argument("--profile", choices=sorted(_GRANULAR_PROFILES), default="fine")
    ap.add_argument("--gravel-spacing", type=float)
    ap.add_argument("--out")
    ap.add_argument("--no-cloth", action="store_true")
    ap.add_argument("--no-gravel", action="store_true")
    ap.add_argument("--no-pebble", action="store_true")
    args = ap.parse_args()
    if args.out is None:
        if args.robot == "go2":
            args.out = os.path.join(REPO, "examples", "scenes",
                                    "go2_oneshot_training_coarse.nks" if
                                    args.profile == "training-coarse" else
                                    "go2_oneshot.nks")
        else:
            args.out = TRAINING_COARSE_OUT if args.profile == "training-coarse" else OUT
    build(args)


if __name__ == "__main__":
    main()
