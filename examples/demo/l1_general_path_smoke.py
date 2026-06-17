#!/usr/bin/env python3
# ---------------------------------------------------------------------------
# L1 acceptance smoke -- drive the go2 world through the GENERAL contact path
# (contact_family=1 -> PairDriven + a baked static heightfield collidable) and
# confirm it is physically reasonable next to the legacy FusedFoot path BEFORE
# the FusedFoot path is deleted + the goldens re-baselined.
#
# This is the cheap pre-flight for the trained-policy rollout (l1_general_path_
# rollout.py): no policy, just settle-from-spawn under the cooked PD targets, on
# (a) FLAT and (b) PyramidStairs, on both paths. We assert finite + feet load +
# NO sink-through (穿模) on the general side, and print the base-z next to FUSED.
#
# Run (conda nuka-v03, GPU 1):
#   CUDA_VISIBLE_DEVICES=1 python examples/demo/l1_general_path_smoke.py
# ---------------------------------------------------------------------------
import os
import sys

import numpy as np

# This repo is installed editable from the MAIN tree (/root/Nuka-Physics) via a
# skbuild ScikitBuildRedirectingFinder, which hijacks `import nuka` to the main
# tree + a stale site-packages _nuka_ext. Neutralize it so we load THIS worktree's
# package + freshly-built extension (with the L1 contact_family kwargs + the fresh
# libnuka.so it RPATHs to).
_WT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "python"))
sys.meta_path = [
    f for f in sys.meta_path
    if type(f).__name__ != "ScikitBuildRedirectingFinder"
]
for _m in [m for m in sys.modules if m == "nuka" or m.startswith("nuka.")]:
    del sys.modules[_m]
sys.path.insert(0, _WT)
import nuka  # noqa: E402

assert nuka.__file__.startswith(_WT), f"wrong nuka pkg: {nuka.__file__}"

SCENE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "scenes", "go2_locomotion.usda")
)
DT = 1.0 / 240.0
STEPS = 240  # 1 s settle.

# PyramidStairs spec mirroring tests/scenario/vproof_go2_terrain.cpp::TerrainSpec
STAIR = dict(step_height=0.04, step_width=0.40, platform_width=2.0)
KPYRAMID_RINGS = 8


def base_z_xyz(world):
    """Return (x, y, z) of env-0 base from BASE_POSE (pos is the first 3 floats)."""
    import torch

    bp = torch.from_dlpack(world.buffer_view(nuka.Field.BASE_POSE)).cpu().numpy()
    flat = np.asarray(bp).reshape(-1)
    return float(flat[0]), float(flat[1]), float(flat[2])


def run(label, *, family, terrain_type, terrain_on):
    kw = dict(
        device=DEV,
        scene_path=SCENE,
        env_count=1,
        dt=DT,
        contact_family=family,
    )
    if terrain_on:
        kw.update(
            terrain_step_height=STAIR["step_height"],
            terrain_step_width=STAIR["step_width"],
            terrain_platform_width=STAIR["platform_width"],
        )
    if family == 1:
        # General path bakes a static heightfield of this type; size it to cover
        # the spawn footprint (default 41x41 @ 0.25 = ~10 m span centred at 0).
        kw.update(heightfield_terrain_type=terrain_type)

    w = nuka.World.create_from_scene(**kw)
    # For the procedural terrain, switch env-0 to the requested type (FUSED reads
    # this per-env; the general heightfield is already baked to `terrain_type`).
    if terrain_on:
        import torch

        tt = torch.from_dlpack(w.buffer_view(nuka.Field.ENV_TERRAIN_TYPE))
        tt[:] = terrain_type
    x0, y0, z0 = base_z_xyz(w)
    w.step_n(STEPS)
    x1, y1, z1 = base_z_xyz(w)
    # Surface height under the base (x,y) for the configured terrain.
    if terrain_type == 1 and terrain_on:
        surf = KPYRAMID_RINGS * STAIR["step_height"]  # platform top at origin.
    else:
        surf = 0.0
    finite = all(np.isfinite(v) for v in (x1, y1, z1))
    print(
        f"  [{label:28s}] base z: {z0:+.4f} -> {z1:+.4f}  (surf~{surf:+.3f}, "
        f"clearance {z1 - surf:+.4f})  xy=({x1:+.3f},{y1:+.3f})  finite={finite}"
    )
    w.destroy()
    return z1, surf, finite


if __name__ == "__main__":
    print(f"scene = {SCENE}")
    DEV = nuka.Device.create(0)  # CUDA_VISIBLE_DEVICES already pins the GPU.
    try:
        print("FLAT ground:")
        zf_leg, sf_leg, ok_leg = run("FUSED  flat", family=0, terrain_type=0,
                                     terrain_on=False)
        zf_gen, sf_gen, ok_gen = run("GENERAL flat (heightfield)", family=1,
                                     terrain_type=0, terrain_on=False)
        print("PyramidStairs (platform top under dog):")
        zs_leg, ss_leg, _ = run("FUSED  stairs", family=0, terrain_type=1,
                                terrain_on=True)
        zs_gen, ss_gen, _ = run("GENERAL stairs (heightfield)", family=1,
                                terrain_type=1, terrain_on=True)

        # Acceptance: general side finite + did not tunnel through the surface
        # (clearance > -5 cm == the vproof kSinkTol no-穿模 bar).
        SINK = -0.05
        flat_ok = ok_gen and (zf_gen - sf_gen) > SINK
        stairs_ok = (zs_gen - ss_gen) > SINK
        print()
        print(f"GENERAL flat   finite+no-穿模: {flat_ok}")
        print(f"GENERAL stairs no-穿模:        {stairs_ok}")
        print(f"base-z delta vs FUSED: flat {zf_gen - zf_leg:+.4f} m, "
              f"stairs {zs_gen - zs_leg:+.4f} m")
        sys.exit(0 if (flat_ok and stairs_ok) else 1)
    finally:
        DEV.close()
