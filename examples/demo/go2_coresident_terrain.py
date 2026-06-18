#!/usr/bin/env python3
"""GATE-2: policy-driven K co-resident Go2 on a tall PyramidStairs, with REAL
collision (foot-ground + body-vs-terrain NO穿模 + DOG-DOG), dumped as GO2T v1 for
the C++ terrain renderer.

This is the owner-bottom-line deliverable: 15-20 dogs TOGETHER on a complex stairs
scene that physically collide and never clip through the terrain. ONE nk::World, K
co-resident articulations (instance_count compose path), driven by the SAME trained
height-scan policy (out/go2_terrain_hs) via the reshape (1,K*single)<->(K,single)
trick -- the policy runs through its NATIVE obs/action code (Go2ObsBuilder +
_height_scan), so there is ZERO policy-transfer error; only the physics topology
changed (K co-resident, collision ON) vs the K-separate-env training world.

Run:
  export LD_LIBRARY_PATH=.../lib64:/root/nuka-collision/build-collision/src
  CUDA_VISIBLE_DEVICES=1 python examples/demo/go2_coresident_terrain.py \
      --robots 18 --seconds 14 --command 0.4 --spawn-r 2.2 \
      --checkpoint /root/Nuka-Physics/out/go2_terrain_hs/nn/last_go2_terrain_hs_ep_1500_rew_10.463228.pth \
      --out out/m10/go2_terrain_collision.bin
"""
import sys
sys.meta_path = [f for f in sys.meta_path
                 if type(f).__name__ != "ScikitBuildRedirectingFinder"]
sys.path.insert(0, "/root/nuka-collision/python")  # collision nuka + tasks/go2_obs + rl_games

import argparse
import math
import os
import struct

import numpy as np
import torch
import yaml

import nuka
import nuka.tasks.go2_obs as G
from nuka.tasks.go2_obs import Go2ObsBuilder

SCENE = "/root/nuka-collision/examples/scenes/go2_locomotion.usda"
CFG = "/root/Nuka-Physics/examples/training/go2_ppo_cfg.yaml"
GO2T_MAGIC = 0x47324F54
NOMINAL_BASE_Z = 0.445
T_PYRAMID = 1
KPYRAMID_RINGS = 8


def pyramid_surf(x, y, step_h, step_w, plat_w, ground=0.0):
    """Host mirror of SampleTerrainHeight(PyramidStairs)."""
    half = plat_w * 0.5
    r = max(abs(x), abs(y))
    top = ground + KPYRAMID_RINGS * step_h
    if r <= half:
        return top
    if step_w <= 0.0:
        return top
    k = math.floor((r - half) / step_w) + 1
    h = top - k * step_h
    return max(h, ground)


def build_player(checkpoint, full_dim):
    import nuka.rl_games  # noqa: F401
    from gymnasium import spaces
    from rl_games.torch_runner import Runner
    with open(CFG) as fh:
        params = yaml.safe_load(fh)["params"]
    obs_space = spaces.Box(low=-G.OBS_CLIP, high=G.OBS_CLIP,
                           shape=(full_dim,), dtype=np.float32)
    act_space = spaces.Box(low=-G.ACTION_SPACE_LIMIT, high=G.ACTION_SPACE_LIMIT,
                           shape=(G.GO2_ACTION_DIM,), dtype=np.float32)
    params["config"]["env_info"] = {"observation_space": obs_space,
                                    "action_space": act_space, "agents": 1,
                                    "value_size": 1}
    runner = Runner()
    runner.load_config(params)
    player = runner.create_player()
    player.restore(checkpoint)
    player.has_batch_dimension = True
    player.is_deterministic = True
    player.is_tensor_obses = True
    return player


def build_scan_grid(dev):
    """Replicate Go2LocomotionEnv._scan_grid from the cfg height_scan block."""
    with open(CFG) as fh:
        tcfg = yaml.safe_load(fh)["params"]["config"]["env_config"]["terrain"]
    hs = dict(tcfg.get("height_scan", {}))
    x_min = float(hs.get("x_min", -0.8)); x_max = float(hs.get("x_max", 0.8))
    y_min = float(hs.get("y_min", -0.5)); y_max = float(hs.get("y_max", 0.5))
    res = float(hs.get("resolution", 0.1))
    scale = float(hs.get("scale", 5.0)); clip = float(hs.get("clip", 1.0))
    nx = int(round((x_max - x_min) / res)) + 1
    ny = int(round((y_max - y_min) / res)) + 1
    gx = torch.linspace(x_min, x_max, nx, device=dev)
    gy = torch.linspace(y_min, y_max, ny, device=dev)
    mx, my = torch.meshgrid(gx, gy, indexing="ij")
    grid = torch.stack((mx.reshape(-1), my.reshape(-1)), dim=1).contiguous()
    # terrain geometry (the same params the cook uses; difficulty 1.0 at render)
    geom = dict(step_h=float(tcfg.get("step_height", 0.15)),
                step_w=float(tcfg.get("step_width", 0.3)),
                plat_w=float(tcfg.get("platform_width", 1.0)),
                grid_w=float(tcfg.get("grid_width", 0.45)),
                grid_hmax=float(tcfg.get("grid_height_max", 0.15)),
                ground=float(tcfg.get("ground_height", 0.0)))
    return grid, scale, clip, geom


def height_scan(world, base_pose_K7, scan_grid, scan_scale, scan_clip, dev):
    """Standalone (K,n_scan) legged_gym height scan, sampling the world's cooked
    heightfield grid (the ONE source obs/physics share)."""
    bp = base_pose_K7
    K = bp.shape[0]
    base_x = bp[:, 0:1]; base_y = bp[:, 1:2]; base_z = bp[:, 2:3]
    qw, qx, qy, qz = bp[:, 3], bp[:, 4], bp[:, 5], bp[:, 6]
    yaw = torch.atan2(2.0 * (qw * qz + qx * qy),
                      1.0 - 2.0 * (qy * qy + qz * qz)).unsqueeze(1)
    cos_y = torch.cos(yaw); sin_y = torch.sin(yaw)
    px = scan_grid[:, 0].unsqueeze(0); py = scan_grid[:, 1].unsqueeze(0)
    wx = base_x + (px * cos_y - py * sin_y)          # (K, n_scan)
    wy = base_y + (px * sin_y + py * cos_y)
    n, m = wx.shape
    surf = world.sample_terrain_height(
        wx.reshape(-1).detach().cpu().contiguous(),
        wy.reshape(-1).detach().cpu().contiguous())
    heights = torch.as_tensor(surf, device=dev, dtype=torch.float32).view(n, m)
    value = (base_z - 0.5 - heights).clamp(-scan_clip, scan_clip) * scan_scale
    return value.to(torch.float32)


def spawn_ring(K, spawn_r, seed, geom):
    """K dogs on the pyramid steps in 2 concentric rings, headings INWARD (climb to
    top) with jitter. Returns (xs, ys, yaws)."""
    rng = np.random.RandomState(seed)
    xs, ys, yaws = [], [], []
    n_inner = K // 2
    radii = [spawn_r, spawn_r + 0.9]
    counts = [n_inner, K - n_inner]
    for ring, (rad, cnt) in enumerate(zip(radii, counts)):
        for i in range(cnt):
            ang = 2.0 * math.pi * (i + 0.5 * ring) / max(cnt, 1) + rng.uniform(-0.12, 0.12)
            x = rad * math.cos(ang)
            y = rad * math.sin(ang)
            # heading toward centre (climb inward/up) + small jitter
            yaw = math.atan2(-y, -x) + rng.uniform(-0.20, 0.20)
            xs.append(x); ys.append(y); yaws.append(yaw)
    return xs[:K], ys[:K], yaws[:K]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robots", type=int, default=18)
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--command", type=float, default=0.4)
    ap.add_argument("--spawn-r", type=float, default=2.2)
    ap.add_argument("--settle", type=int, default=40,
                    help="control steps holding default pose (zero cmd) before policy.")
    ap.add_argument("--spawn-clear", type=float, default=0.04,
                    help="extra spawn z clearance so dogs drop gently (no spawn 穿模 launch).")
    ap.add_argument("--step-h", type=float, default=-1.0, help="override step_height.")
    ap.add_argument("--step-w", type=float, default=-1.0, help="override step_width.")
    ap.add_argument("--no-gains", action="store_true", help="skip PD gain/IC writes (debug).")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="out/m10/go2_terrain_collision.bin")
    args = ap.parse_args()

    K = args.robots
    dev_t = torch.device("cuda")
    grid, scan_scale, scan_clip, geom = build_scan_grid(dev_t)
    if args.step_h > 0:
        geom["step_h"] = args.step_h
    if args.step_w > 0:
        geom["step_w"] = args.step_w
    print(f"[gate2] terrain PyramidStairs step_h={geom['step_h']} step_w={geom['step_w']} "
          f"plat_w={geom['plat_w']} -> top={KPYRAMID_RINGS*geom['step_h']:.2f}m, "
          f"base_r~{geom['plat_w']*0.5 + KPYRAMID_RINGS*geom['step_w']:.2f}m")

    dev = nuka.Device.create(0, 0)
    world = nuka.World.create_from_scene(
        device=dev, scene_path=SCENE, env_count=1, dt=0.005,
        determinism=0, control_mode=0,
        terrain_step_height=geom["step_h"], terrain_step_width=geom["step_w"],
        terrain_platform_width=geom["plat_w"],
        terrain_grid_width=geom["grid_w"], terrain_grid_height_max=geom["grid_hmax"],
        instance_count=K, instance_spacing=1.5,
    )
    # terrain type = PyramidStairs, difficulty 1.0 for the env.
    tt = torch.from_dlpack(world.buffer_view(nuka.ENV_TERRAIN_TYPE)); tt[...] = T_PYRAMID
    td = torch.from_dlpack(world.buffer_view(nuka.ENV_TERRAIN_DIFFICULTY)); td[...] = 1.0
    nuka.sync()

    # obs builder, re-pointed to per-dog (K,...) views.
    ob = Go2ObsBuilder(dev, world)
    BLC = G.GO2_BLC
    ob.num_envs = K
    ob._q = ob._q.reshape(K, BLC)
    ob._qd = ob._qd.reshape(K, BLC)
    ob._pose = ob._pose.reshape(K, BLC, 7)
    ob._vel = ob._vel.reshape(K, BLC, 6)
    ob._tgt = ob._tgt.reshape(K, BLC)
    ob._base_pose = ob._base_pose.reshape(K, 7)

    # PD gains per-dog (Kp=20, Kd=0.5, per-joint force limits) on all K blocks.
    if not args.no_gains:
        kp = torch.from_dlpack(world.buffer_view(nuka.DRIVE_STIFFNESS)).reshape(K, BLC)
        kd = torch.from_dlpack(world.buffer_view(nuka.DRIVE_DAMPING)).reshape(K, BLC)
        fl = torch.from_dlpack(world.buffer_view(nuka.DRIVE_FORCE_LIMIT)).reshape(K, BLC)
        kp[:, 1:BLC] = G.KP
        kd[:, 1:BLC] = G.KD
        fl[:, 1:BLC] = ob.force_limit_urdf[ob.nuka_slot_for_urdf].unsqueeze(0).expand(K, 12)
        nuka.sync()

    # spawn the dogs on the pyramid (z = local surface + nominal stance).
    xs, ys, yaws = spawn_ring(K, args.spawn_r, args.seed, geom)
    bp = ob._base_pose
    for i in range(K):
        z = pyramid_surf(xs[i], ys[i], geom["step_h"], geom["step_w"], geom["plat_w"],
                         geom["ground"]) + NOMINAL_BASE_Z + args.spawn_clear
        half = yaws[i] * 0.5
        bp[i, 0] = xs[i]; bp[i, 1] = ys[i]; bp[i, 2] = z
        bp[i, 3] = math.cos(half); bp[i, 4] = 0.0; bp[i, 5] = 0.0; bp[i, 6] = math.sin(half)
    nuka.sync()

    command = torch.zeros(K, 3, device=dev_t); command[:, 0] = args.command
    last_action = torch.zeros(K, 12, device=dev_t)

    player = build_player(args.checkpoint, 48 + grid.shape[0])

    decimation = 4
    ctrl_dt = 0.005 * decimation
    n_steps = int(round(args.seconds / ctrl_dt))
    pose_rec, drive_rec = [], []
    drive_view = torch.from_dlpack(world.buffer_view(nuka.DRIVE_TARGET)).reshape(K, BLC)
    pose_view = ob._pose

    # SETTLE: hold the cooked default pose (zero cmd) so each dog drops gently onto
    # the steps and finds footing before the policy + forward command engage. This
    # avoids a spawn-interpenetration contact launch (the explosive-recovery NaN).
    zero_act = torch.zeros(K, 12, device=dev_t)
    zero_cmd = torch.zeros(K, 3, device=dev_t)
    bz_chk = pose_view[:, 0, 2]
    print(f"[settle] spawn base_z finite={torch.isfinite(bz_chk).all().item()} "
          f"min/max={bz_chk.min().item():.3f}/{bz_chk.max().item():.3f}", flush=True)
    for ss in range(args.settle):
        if not args.no_gains:
            ob.write_action(zero_act)
        for _ in range(decimation):
            world.step()
        nuka.sync()
        nfin = int((~torch.isfinite(pose_view[:, 0, 2])).sum().item())
        if nfin > 0 or ss < 5 or ss % 10 == 0:
            bz = pose_view[:, 0, 2]
            fin = torch.isfinite(bz)
            mn = bz[fin].min().item() if fin.any() else float('nan')
            mx = bz[fin].max().item() if fin.any() else float('nan')
            print(f"[settle] step {ss}: non_finite_dogs={nfin}/{K} base_z[min/max]={mn:.3f}/{mx:.3f}",
                  flush=True)
            if nfin > 0:
                break
    bz0 = pose_view[:, 0, 2].detach().cpu().numpy()
    print(f"[settle] after {args.settle} steps base_z[min/mean/max]="
          f"{bz0.min():.3f}/{bz0.mean():.3f}/{bz0.max():.3f}", flush=True)

    falls = 0
    for k in range(n_steps):
        obs = ob.compute_obs(command, last_action)            # (K,48)
        hs = height_scan(world, ob._base_pose, grid, scan_scale, scan_clip, dev_t)
        full = torch.cat([obs, hs], dim=-1)                    # (K,235)
        full = torch.nan_to_num(full, nan=0.0, posinf=G.OBS_CLIP, neginf=-G.OBS_CLIP)
        with torch.no_grad():
            action = player.get_action(full, is_deterministic=True)
        action = torch.nan_to_num(action, nan=0.0).clamp(-G.ACTION_SPACE_LIMIT,
                                                         G.ACTION_SPACE_LIMIT)
        clipped = ob.write_action(action)
        last_action = clipped
        for _ in range(decimation):
            world.step()
        nuka.sync()
        drive_rec.append(drive_view.detach().cpu().numpy().copy())
        pose_rec.append(pose_view.detach().cpu().numpy().copy())
        if (k % 50) == 0 or k == n_steps - 1:
            bz = pose_view[:, 0, 2].detach().cpu().numpy()
            print(f"  t={k*ctrl_dt:5.2f}s base_z[min/mean/max]="
                  f"{bz.min():.3f}/{bz.mean():.3f}/{bz.max():.3f}", flush=True)

    pose_arr = np.stack(pose_rec).astype("<f4")    # (N,K,13,7)
    drive_arr = np.stack(drive_rec).astype("<f4")  # (N,K,13)
    N = pose_arr.shape[0]

    # ---- analysis: 穿模 + dog-dog + alive ----
    base_xy = pose_arr[:, :, 0, 0:2]
    base_z = pose_arr[:, :, 0, 2]
    # min pairwise dog-dog distance over all frames
    d = np.linalg.norm(base_xy[:, :, None, :] - base_xy[:, None, :, :], axis=-1)
    iu = np.triu_indices(K, 1)
    min_dd = float(d[:, iu[0], iu[1]].min())
    # 穿模: base center vs local terrain surface (worst over frames/dogs)
    worst_clear = 1e9
    for s in range(N):
        for i in range(K):
            sf = pyramid_surf(base_xy[s, i, 0], base_xy[s, i, 1],
                              geom["step_h"], geom["step_w"], geom["plat_w"], geom["ground"])
            worst_clear = min(worst_clear, base_z[s, i] - sf)
    # alive: dogs whose final base stays >0.15 above local surface
    alive = 0
    for i in range(K):
        sf = pyramid_surf(base_xy[-1, i, 0], base_xy[-1, i, 1],
                          geom["step_h"], geom["step_w"], geom["plat_w"], geom["ground"])
        if base_z[-1, i] - sf > 0.15:
            alive += 1
    print("=" * 64)
    print(f"[gate2] N={N} K={K} | min dog-dog dist={min_dd:.3f}m (collision keeps >~0.2) | "
          f"worst base clearance={worst_clear:+.3f}m (穿模 if <<0) | alive={alive}/{K}")

    # ---- write GO2T v1 ----
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(struct.pack("<8i2f", GO2T_MAGIC, 1, K, N, BLC, BLC, 7,
                            decimation, float(ctrl_dt), float(ctrl_dt)))
        for r in range(K):
            f.write(struct.pack("<2fif", 0.0, 0.0, int(T_PYRAMID), 1.0))
        for s in range(N):
            for r in range(K):
                f.write(drive_arr[s, r].tobytes(order="C"))
                f.write(pose_arr[s, r].reshape(-1).tobytes(order="C"))
    print(f"[gate2] WROTE {out} ({os.path.getsize(out)/1e6:.2f} MB, {N}x{K}, {N*ctrl_dt:.1f}s)")


if __name__ == "__main__":
    main()
