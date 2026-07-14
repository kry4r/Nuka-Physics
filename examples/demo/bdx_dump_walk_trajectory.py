#!/usr/bin/env python3
"""[demo] Dump a DETERMINISTIC BDX walking trajectory for render/replay.

Rolls a TRAINED BDX walk policy (rl_games checkpoint, action = mu) through the
unified ``nuka.World`` HEADLESS at a fixed forward command, recording per control
step the base pose, full joint angles (15 incl. root + head), all link poses, and
the drive targets. Saves a self-describing ``.npz`` + a sidecar ``.meta.json``.
Gated on a clean forward walk (net forward displacement, upright, no NaN).

Run::

  LD_LIBRARY_PATH=... CUDA_VISIBLE_DEVICES=0 python examples/demo/bdx_dump_walk_trajectory.py \
      --ckpt /data/xtzhang25/_work/activate/out/bdx_walk/nn/bdx_walk.pth \
      --seconds 20 --cmd-vx 0.2 \
      --out /data/xtzhang25/_work/activate/out/bdx_walk/traj/bdx_walk_traj.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

import nuka  # noqa: F401
from nuka.tasks.bdx_locomotion import make_env
from nuka.tasks import bdx_obs as B

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "training"))
from bdx_walk_eval import load_actor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--cmd-vx", type=float, default=0.2)
    ap.add_argument("--cmd-vy", type=float, default=0.0)
    ap.add_argument("--cmd-wyaw", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true",
                    help="Write even if the gate fails; replay force-glides the base straight.")
    ap.add_argument("--out",
                    default="/data/xtzhang25/_work/activate/out/bdx_walk/traj/bdx_walk_traj.npz")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = make_env(1, fixed_command=True,
                   command=(args.cmd_vx, args.cmd_vy, args.cmd_wyaw), seed=args.seed)
    dev = env._dev
    steps = int(round(args.seconds / (env.dt * env.decimation)))
    forward, ep, rew = load_actor(args.ckpt, B.BDX_OBS_DIM, B.BDX_ACTION_DIM, dev)
    print(f"[dump] ckpt epoch={ep} rew={rew} cmd=({args.cmd_vx},{args.cmd_vy},"
          f"{args.cmd_wyaw}) steps={steps}", flush=True)

    obs, _ = env.reset(seed=args.seed)
    env.command[:] = torch.tensor([args.cmd_vx, args.cmd_vy, args.cmd_wyaw], device=dev)

    base_pose = torch.from_dlpack(env._world.buffer_view(nuka.BASE_POSE))
    link_pose = torch.from_dlpack(env._world.buffer_view(nuka.ARTICULATION_LINK_POSE))
    joint_pos = torch.from_dlpack(env._world.buffer_view(nuka.JOINT_POSITION))
    drive_tgt = torch.from_dlpack(env._world.buffer_view(nuka.DRIVE_TARGET))

    bp, lp, jp, dt_, tilt = [], [], [], [], []
    xy0 = base_pose[0, 0:2].detach().cpu().numpy().copy()
    nan = False
    for k in range(steps):
        mu = forward(obs)
        obs, *_ = env.step(mu)
        if not bool(torch.isfinite(obs).all()):
            nan = True
            break
        bp.append(base_pose[0].detach().cpu().numpy().copy())
        lp.append(link_pose[0].detach().cpu().numpy().copy())
        jp.append(joint_pos[0].detach().cpu().numpy().copy())
        dt_.append(drive_tgt[0].detach().cpu().numpy().copy())
        tilt.append(float(env._obs.tilt_deg()[0]))

    xyf = base_pose[0, 0:2].detach().cpu().numpy().copy()
    bp = np.stack(bp); lp = np.stack(lp); jp = np.stack(jp); dt_ = np.stack(dt_)
    dx = float(xyf[0] - xy0[0]); dy = float(xyf[1] - xy0[1])
    sim_t = bp.shape[0] * env.dt * env.decimation
    z_min = float(bp[:, 2].min()); tilt_f = tilt[-1] if tilt else 999.0
    walked = (not nan) and abs(dx) > 0.3 and tilt_f < 35.0 and z_min > 0.12

    print(f"[dump] frames={bp.shape[0]} sim_t={sim_t:.1f}s dx={dx:+.3f} dy={dy:+.3f} "
          f"vx={dx/sim_t:+.3f} z_min={z_min:.3f} tilt_final={tilt_f:.1f} nan={nan}")
    print(f"[dump] WALK GATE: {'PASS' if walked else 'FAIL'}")
    if not walked and not args.force:
        print("[dump] not a clean walk; not writing.")
        env.close()
        return 3

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(
        args.out, base_pose=bp, link_pose=lp, joint_pos=jp, drive_target=dt_,
        dof_names=np.array(env._world.dof_names()),
        dt=env.dt, decimation=env.decimation,
        cmd=np.array([args.cmd_vx, args.cmd_vy, args.cmd_wyaw], np.float32),
    )
    meta = {
        "scene": "examples/scenes/bdx_stand.nks",
        "frames": int(bp.shape[0]),
        "control_dt_s": env.dt * env.decimation,
        "dt_s": env.dt, "decimation": env.decimation,
        "dof_names": env._world.dof_names(),
        "arrays": {
            "base_pose": "(N,7) [px,py,pz,qw,qx,qy,qz] world",
            "link_pose": "(N,15,7) per-link world pose (slot 0=trunk, 1-14 joints)",
            "joint_pos": "(N,15) JOINT_POSITION (slot 0=root)",
            "drive_target": "(N,15) DRIVE_TARGET",
        },
        "cmd": [args.cmd_vx, args.cmd_vy, args.cmd_wyaw],
        "walk_validation": {"dx_m": dx, "dy_m": dy, "vx_mps": dx / sim_t,
                            "z_min_m": z_min, "tilt_final_deg": tilt_f, "nan": nan},
    }
    meta_path = os.path.splitext(os.path.abspath(args.out))[0] + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[dump] wrote {args.out} + {meta_path}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
