#!/usr/bin/env python3
"""[demo] Dump a DETERMINISTIC Go2 HANDSTAND (倒立) trajectory for C++ video replay.

The python PRE-PASS for the Go2 rear-leg handstand hero clip. Loads a TRAINED
Go2HandstandEnv actor (rl_games checkpoint -> obs-normalizer + MLP -> mu) and
drives the env DETERMINISTICALLY with NO autoreset (action = policy mean), exactly
like go2_handstand_evalmanual.py -- the SAME closed-loop path the deterministic
gate measures, so the rendered motion IS the gated handstand (no scripted assist,
no IC artifact). It records EVERY physics sub-step's ARTICULATION_LINK_POSE (env 0)
+ the held DRIVE_TARGET into the SAME flat GO2W v1 binary the walk video tool
reads, so examples/demo/go2_handstand_video.cpp can pose-replay + render it.

GROUND: implicit z=0 half-space (model.ground_height=0). The renderer draws a
studio ground at z=0. The dog stands inverted on its REAR feet (nose-down pitch,
proj_grav_x -> -1), front paws lifted.

Run (GPU1, offscreen later)::

  CUDA_VISIBLE_DEVICES=1 \
    LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:/root/Nuka-Physics/build-cuda128/src \
    PYTHONPATH=/root/Nuka-Physics/build-cuda128/src:/root/Nuka-Physics/python \
    python examples/demo/go2_dump_handstand_trajectory.py \
      --ckpt out/go2_handstand2_s2/nn/<best>.pth --stage S2 --theta-cmd 75 \
      --seconds 10 --out out/go2_handstand_trajectory.bin
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "training")))

import nuka  # noqa: E402
from nuka.tasks.go2_handstand import make_env, COLLAPSE_FLOOR_Z  # noqa: E402
from nuka.tasks import go2_obs as G  # noqa: E402
from go2_handstand_eval import load_actor  # noqa: E402

MAGIC = 0x474F3257  # 'GO2W' (same binary format as the walk dump)
VERSION = 1
DT = 0.005
DECIMATION = 4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stage", default="S2")
    ap.add_argument("--theta-cmd", type=float, default=75.0)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--settle", type=int, default=20,
                    help="control steps to let the policy reach the hold before "
                         "recording starts (so the clip opens already inverted).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-index", type=int, default=-1,
                    help="which env to record (-1 = auto-pick the best-held env).")
    ap.add_argument("--num-envs", type=int, default=64)
    ap.add_argument("--out", default="out/go2_handstand_trajectory.bin")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_env(args.num_envs, stage=args.stage, theta_cmd_obs=True, seed=args.seed)
    dev = env._torch_device
    n = env.num_envs
    forward, ep, rew = load_actor(args.ckpt, env._obs_dim, G.GO2_ACTION_DIM, dev)
    print(f"[hs-dump] ckpt epoch={ep} stage={args.stage} theta={args.theta_cmd} "
          f"n={n} settle={args.settle} seconds={args.seconds}", flush=True)

    env.reset(seed=args.seed)
    env._theta_cmd_deg[:] = float(args.theta_cmd)
    nuka.sync()
    env._world.step_n(env.decimation)
    nuka.sync()

    b = env._obs
    front_slot = env._front_foot_slot
    rear_slot = env._rear_foot_slot
    pose_buf = torch.from_dlpack(env._world.buffer_view(nuka.ARTICULATION_LINK_POSE))
    drive_buf = torch.from_dlpack(env._world.buffer_view(nuka.DRIVE_TARGET))

    ctrl_steps = int(round(args.seconds / (DT * DECIMATION)))
    last_action = torch.zeros(n, G.GO2_ACTION_DIM, device=dev)

    # Track per-env held inversion over the recorded window so we can auto-pick the
    # best-held env to render (the one that stays most cleanly inverted + alive).
    inv_sum = torch.zeros(n, device=dev)
    inv_cnt = torch.zeros(n, device=dev)
    alive = torch.ones(n, dtype=torch.bool, device=dev)

    drive_records = []   # list of (13,) f32
    pose_records = []    # list of (13,7) f32 -- for the chosen env, filled after pick

    # First pass: roll the policy, recording every env's pose each sub-step into a
    # big buffer (n,13,7) per sub-step, plus the drive target, so after we pick the
    # best env we slice its column. n*steps*13*7 f32 -- for n=64, 500 ctrl steps,
    # 2000 sub-steps that's ~46M floats ~ 186 MB; fine.
    all_pose = []        # list of (n,13,7)
    all_drive = []       # list of (n,13)

    for k in range(args.settle + ctrl_steps):
        obs48 = b.compute_obs(env.command, last_action)
        obs = env._append_theta(obs48)
        act = forward(obs)
        clipped = b.write_action(act)
        last_action = clipped
        nuka.sync()
        dt_now = drive_buf.detach().cpu().numpy().copy()       # (n,13)
        for _ in range(DECIMATION):
            env._world.step()
            nuka.sync()
            if k >= args.settle:
                all_pose.append(pose_buf.detach().cpu().numpy().copy())  # (n,13,7)
                all_drive.append(dt_now.copy())

        pg = b.projected_gravity()
        base_z = b.base_pos()[:, 2]
        q = b.q_urdf(); qd = b.qd_urdf()
        flipped = (pg[:, 2] > 0.7) & (pg[:, 0] > 0.0)
        collapsed = base_z < COLLAPSE_FLOOR_Z
        nonfinite = (~torch.isfinite(b.base_pos()).all(1) | ~torch.isfinite(q).all(1)
                     | ~torch.isfinite(qd).all(1))
        runaway = qd.abs().amax(1) > 50.0
        alive = alive & (~(flipped | collapsed | nonfinite | runaway))
        if k >= args.settle:
            m = alive.to(torch.float32)
            inv_sum += (-pg[:, 0]) * m
            inv_cnt += m
        if k % 50 == 0:
            a = alive
            mp = float((-pg[a, 0]).median()) if bool(a.any()) else float("nan")
            print(f"[hs-dump] step {k:4d} alive={int(a.sum())}/{n} "
                  f"inv(med)={mp:+.3f} bz={float(base_z[a].median()) if bool(a.any()) else float('nan'):.3f}",
                  flush=True)

    # Pick the env to render: the alive env with the highest mean held inversion.
    inv_mean = torch.where(inv_cnt > 0, inv_sum / inv_cnt.clamp_min(1),
                           torch.full_like(inv_sum, -2.0))
    # require it survived (inv_cnt == ctrl-step count means alive the whole window)
    full = inv_cnt >= float(int(0.95 * ctrl_steps))
    score = torch.where(full, inv_mean, inv_mean - 5.0)
    if args.env_index >= 0:
        pick = int(args.env_index)
    else:
        pick = int(torch.argmax(score).item())
    print(f"[hs-dump] picked env {pick}: held-inv mean={float(inv_mean[pick]):+.3f} "
          f"alive_frac={float(inv_cnt[pick]) / max(1, ctrl_steps):.2f} "
          f"survived_full={bool(full[pick])}", flush=True)

    # Slice the chosen env's records.
    for p, d in zip(all_pose, all_drive):
        pose_records.append(p[pick])      # (13,7)
        drive_records.append(d[pick])     # (13,)

    drive_arr = np.stack(drive_records).astype(np.float32)     # (N,13)
    pose_arr = np.stack(pose_records).astype(np.float32)       # (N,13,7)
    N = drive_arr.shape[0]

    # Gate: refuse to write a non-inverted / collapsed clip.
    base_z_traj = pose_arr[:, 0, 2]
    held_inv = float(inv_mean[pick])
    z_min = float(base_z_traj.min())
    inverted = held_inv >= 0.6 and z_min > 0.05 and bool(full[pick])
    print(f"[hs-dump] CLIP GATE: held_inv={held_inv:+.3f}(>=0.6) z_min={z_min:.3f}(>0.05) "
          f"survived_full={bool(full[pick])} -> {'PASS' if inverted else 'FAIL'}", flush=True)
    if not inverted:
        print("[hs-dump] ABORT: the picked rollout is not a clean sustained "
              "handstand; not writing. Train/eval more before rendering.", flush=True)
        env.close()
        return 1

    out_bin = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_bin), exist_ok=True)
    with open(out_bin, "wb") as f:
        f.write(struct.pack("<8i2f", MAGIC, VERSION, N, 13, 13, 7, DECIMATION,
                            int(round(1.0 / (DT * DECIMATION) * 1000)),
                            float(DT), float(DT * DECIMATION)))
        flat = np.concatenate([drive_arr.reshape(N, 13),
                               pose_arr.reshape(N, 13 * 7)], axis=1).astype("<f4")
        f.write(flat.tobytes(order="C"))
    sz = os.path.getsize(out_bin)

    meta_path = os.path.splitext(out_bin)[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "format": "go2_handstand_trajectory.bin (GO2W v1, same as walk)",
            "binary_file": os.path.basename(out_bin),
            "num_steps": int(N), "dof_count": 13, "link_count": 13, "pose_floats": 7,
            "dt": DT, "decimation": DECIMATION,
            "control_period_s": DT * DECIMATION, "control_hz": 1.0 / (DT * DECIMATION),
            "sim_time_s": N * DT,
            "ckpt": args.ckpt, "ckpt_epoch": int(ep) if ep is not None else -1,
            "stage": args.stage, "theta_cmd_deg": args.theta_cmd,
            "picked_env": pick, "held_inversion_mean": held_inv,
            "base_z_min": z_min, "base_z_max": float(base_z_traj.max()),
            "convention": ("NOSE-DOWN rear-leg handstand: proj_grav_x -> -1, rear "
                           "feet planted on z=0, front paws up. Render scene "
                           "go2.nks (same articulation/cook order as the walk)."),
            "render_scene": "examples/scenes/go2.nks",
            "ground_plane": "implicit z=0 half-space; draw studio ground at z=0",
        }, f, indent=2)

    print(f"\n[hs-dump] WROTE {out_bin} ({sz/1e6:.3f} MB, {N} sub-steps, "
          f"{N*DT:.1f}s) + {meta_path}", flush=True)
    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
