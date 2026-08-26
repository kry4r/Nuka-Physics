#!/usr/bin/env python3
"""Capture + verify the STRICT front-handstand walk policy, and emit an npz
the existing stick-skeleton renderer can rasterize.

Strict-handstand gate: ONLY the two front paws may touch the ground. The
metrics that matter:
  * verticality: mean pgx, hold fraction (pgx>0.85 with both front paws loaded)
  * rear clearance: rear-paw contact duty (must be ~0) + mean height above ground
  * traversal: net world-x advance per env (walking on hands, not balancing)
  * front stepping: FL<->FR load alternation rate (a hand-walk gait)

npz schema matches examples/demo/go2_demo_render.py: frames (T,N,13,7),
cmds (N,3), dx_world (N,), plus pgx (T,N) for handstand-aware render checks.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "python")

import nuka  # noqa: F401
from nuka.tasks.go2_front_handstand import make_env


def load_policy(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    sd = ck["model"]
    rms_mean = sd["running_mean_std.running_mean"].float()
    rms_var = sd["running_mean_std.running_var"].clamp(min=1e-6).float()
    layers = []
    for i in (0, 2, 4):
        w = sd[f"a2c_network.actor_mlp.{i}.weight"]
        layers.append(torch.nn.Linear(w.shape[1], w.shape[0]))
        layers[-1].weight.data = w.float()
        layers[-1].bias.data = sd[f"a2c_network.actor_mlp.{i}.bias"].float()
        layers.append(torch.nn.ELU())
    mu = torch.nn.Linear(128, 12)
    mu.weight.data = sd["a2c_network.mu.weight"].float()
    mu.bias.data = sd["a2c_network.mu.bias"].float()
    return torch.nn.Sequential(*layers, mu).cuda().eval(), rms_mean, rms_var.sqrt()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=(
        "runs/go2_front_handstand_walk/nn/"
        "last_go2_front_handstand_ep_4200_rew_1879.667.pth"))
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--out", default="out/go2_handstand_walk/go2_hs_walk.npz")
    args = ap.parse_args()

    policy, rms_mean, rms_std = load_policy(args.checkpoint)
    n = args.envs
    env = make_env(n)
    obs, _ = env.reset()

    steps = int(args.seconds * 50)
    frames, pgxs = [], []
    front_contact = torch.zeros(steps, n, 2, device="cuda")
    rear_contact = torch.zeros(steps, n, 2, device="cuda")
    rear_height = torch.zeros(steps, n, device="cuda")
    hold_n = torch.zeros(n, device="cuda")

    with torch.no_grad():
        for k in range(steps):
            x = ((obs - rms_mean) / rms_std).float()
            a = policy(x).clamp(-1.0, 1.0)
            obs, r, term, trunc, _info = env.step(a)
            pose = env._pose[:, :13].detach()          # (N,13,7) world xyzwxyz
            frames.append(pose.cpu().numpy().copy())
            pg = env._obs.projected_gravity()
            pgxs.append(pg[:, 0].cpu().numpy().copy())
            hold_n += ((pg[:, 0] > 0.85)
                       & (env._front_fz().min(dim=-1).values > 5.0)).float()
            fz = env._wrench[:, env._paw_slot, 2].abs()
            front_contact[k] = (fz[:, 0:2] > 5.0).float()
            rear_contact[k] = (fz[:, 2:4] > 5.0).float()
            rear_height[k] = env._paw_world()[:, 2:4, 2].mean(dim=-1)
    frames_arr = np.stack(frames)                     # (T,N,13,7)
    pgx_arr = np.stack(pgxs)                          # (T,N)
    dx_world = (frames_arr[-1, :, 0, 0] - frames_arr[0, :, 0, 0]).astype(np.float64)

    fc = front_contact.cpu().numpy()
    rc = rear_contact.cpu().numpy()
    rh = rear_height.cpu().numpy()
    dur = steps * 0.02
    # front-paw gait: sign flips of (FL loaded - FR loaded) per second.
    diff = np.sign(fc[:, :, 0].astype(int) - fc[:, :, 1].astype(int))
    flips = (np.diff(diff, axis=0) != 0).sum(axis=0)
    rear_duty = rc.max(axis=2).mean(axis=0)           # any-rear-contact fraction

    print("\n===== STRICT HANDSTAND-WALK VERIFICATION =====")
    print(f"checkpoint: {args.checkpoint}")
    print(f"rollout: {steps} steps ({dur:.1f}s), {n} envs")
    print(f"mean pgx       per env: {np.round(pgx_arr.mean(axis=0), 3).tolist()}")
    print(f"hold fraction  per env: {np.round((hold_n / steps).cpu().numpy(), 2).tolist()}")
    print(f"REAR contact duty per env (want ~0): {np.round(rear_duty, 3).tolist()}")
    print(f"REAR mean height m per env: {np.round(rh.mean(axis=0), 2).tolist()}")
    print(f"net dx (m)     per env: {np.round(dx_world, 2).tolist()}")
    print(f"front-step rate Hz per env: {np.round(flips / dur, 2).tolist()}")
    ok = ((rear_duty < 0.05) & (dx_world > 0.15)).sum()
    print(f"STRICT-WALK ENVS (rear-duty<0.05 AND dx>0.15m): {int(ok)}/{n}")

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmds = np.zeros((n, 3), np.float32)   # renderer schema compat; unused here
    np.savez_compressed(out, frames=frames_arr.astype(np.float32), cmds=cmds,
                        dx_world=dx_world, pgx=pgx_arr.astype(np.float32))
    print(f"[capture] wrote {out} ({frames_arr.shape[0]} frames x {n} envs)")
    env.close()


if __name__ == "__main__":
    main()
