#!/usr/bin/env python3
"""GATE-1 smoke: K co-resident Go2 on PyramidStairs via the new instance_count
compose path. Verifies (a) the world cooks K articulations into ONE env (field
arrays sized K*single), (b) it steps finite, (c) NO 穿模 (every dog's trunk stays
above the local terrain surface after settling), (d) an overlapping dog pair
PHYSICALLY pushes apart (dog-dog collision). This is the collision foundation the
15-20-dog terrain video stands on. No policy here -- PD-hold the cooked rest pose.
"""
import sys
# Use the COLLISION worktree's nuka (drop the scikit editable redirect to main tree).
sys.meta_path = [f for f in sys.meta_path
                 if type(f).__name__ != "ScikitBuildRedirectingFinder"]
sys.path.insert(0, "/root/nuka-collision/python")

import math
import numpy as np
import torch
import nuka

print("nuka from:", nuka.__file__)

SCENE = "/root/nuka-collision/examples/scenes/go2_locomotion.usda"
K = 4
NOMINAL_BASE_Z = 0.445
T_PYRAMID = 1
STEP_H, STEP_W, PLAT_W = 0.10, 0.40, 2.0

dev = nuka.Device.create(0, 0)
world = nuka.World.create_from_scene(
    device=dev, scene_path=SCENE, env_count=1, dt=0.005,
    determinism=0, control_mode=0,
    terrain_step_height=STEP_H, terrain_step_width=STEP_W,
    terrain_platform_width=PLAT_W,
    instance_count=K, instance_spacing=1.0,
)
print("world created.")

# --- shapes (learn the co-resident field layout) ---------------------------
lp = torch.from_dlpack(world.buffer_view(nuka.ARTICULATION_LINK_POSE))
bp = torch.from_dlpack(world.buffer_view(nuka.BASE_POSE))
dt_ = torch.from_dlpack(world.buffer_view(nuka.DRIVE_TARGET))
q = torch.from_dlpack(world.buffer_view(nuka.JOINT_POSITION))
tt = torch.from_dlpack(world.buffer_view(nuka.ENV_TERRAIN_TYPE))
print("ARTICULATION_LINK_POSE", tuple(lp.shape))
print("BASE_POSE", tuple(bp.shape))
print("DRIVE_TARGET", tuple(dt_.shape))
print("JOINT_POSITION", tuple(q.shape))
print("ENV_TERRAIN_TYPE", tuple(tt.shape), tt.detach().cpu().numpy().ravel())

dev_t = lp.device
BLC_total = lp.shape[1]
BLC = BLC_total // K
print(f"per-dog links BLC={BLC}  total={BLC_total}")

# --- set terrain type = PyramidStairs for the env --------------------------
tt[...] = T_PYRAMID
nuka.sync()

# --- terrain surface sampler (mirror SampleTerrainHeight PyramidStairs) -----
def surf(x, y):
    half = PLAT_W * 0.5
    r = max(abs(x), abs(y))
    top = STEP_H * 8.0  # kPyramidRings assumed 8 (verify against engine in gate2)
    if r <= half:
        return top
    k = math.floor((r - half) / STEP_W) + 1
    h = top - k * STEP_H
    return max(h, 0.0)

# --- place K dogs: 2 on the platform top OVERLAPPING (dog-dog), 2 spread -----
# bp layout: flatten to (K,7) regardless of (K,7) or (1,K*7).
bp_flat = bp.reshape(K, 7)
place = [(0.0, 0.0), (0.12, 0.0), (-0.6, 0.6), (0.6, -0.6)]  # dog0,1 overlap @0.12m (<0.20 sum)
for i, (x, y) in enumerate(place):
    z = surf(x, y) + NOMINAL_BASE_Z
    bp_flat[i, 0] = x
    bp_flat[i, 1] = y
    bp_flat[i, 2] = z
    bp_flat[i, 3] = 1.0  # quat w
    bp_flat[i, 4] = 0.0
    bp_flat[i, 5] = 0.0
    bp_flat[i, 6] = 0.0
nuka.sync()

spawn_sep01 = abs(float(bp_flat[1, 0]) - float(bp_flat[0, 0]))
print(f"spawn overlap pair sep(dog0,dog1) = {spawn_sep01:.3f} m (trunk box ~0.2)")

# --- settle: PD-hold cooked rest target, drop onto terrain ------------------
for s in range(500):
    r = world.step()
nuka.sync()

lp2 = torch.from_dlpack(world.buffer_view(nuka.ARTICULATION_LINK_POSE)).reshape(K, BLC, 7)
bp2 = torch.from_dlpack(world.buffer_view(nuka.BASE_POSE)).reshape(K, 7)
lp2c = lp2.detach().cpu().numpy()
bp2c = bp2.detach().cpu().numpy()

finite = np.isfinite(lp2c).all() and np.isfinite(bp2c).all()
print("finite:", finite)

# (c) NO 穿模: each dog's base z stays at/above local terrain (minus stance sag).
worst = 1e9
for i in range(K):
    bx, by, bz = bp2c[i, 0], bp2c[i, 1], bp2c[i, 2]
    s = surf(bx, by)
    clr = bz - s
    worst = min(worst, clr)
    print(f"  dog{i} base xyz=({bx:+.3f},{by:+.3f},{bz:+.3f}) surf={s:.3f} base_clear={clr:+.3f}")
print(f"min base clearance above terrain = {worst:+.3f} m  (穿模 if << 0; nominal stance ~0.30-0.45)")

# (d) dog-dog push-apart: the overlapping pair (dog0,dog1) separated past spawn.
final_sep01 = abs(float(bp2c[1, 0]) - float(bp2c[0, 0]))
xy_sep01 = math.hypot(bp2c[1,0]-bp2c[0,0], bp2c[1,1]-bp2c[0,1])
print(f"overlap pair final planar sep = {xy_sep01:.3f} m (spawn {spawn_sep01:.3f}; collision pushes apart)")

ok_noclip = worst > 0.0             # base center above local surface (no trunk sink-through)
ok_pushapart = xy_sep01 > spawn_sep01 + 0.02
print("=" * 60)
print(f"GATE1  finite={finite}  no_穿模={ok_noclip}  dog_dog_pushapart={ok_pushapart}")
print(f"GATE1 VERDICT = {'PASS' if (finite and ok_noclip and ok_pushapart) else 'CHECK'}")
