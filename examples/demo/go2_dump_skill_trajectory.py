#!/usr/bin/env python3
"""Dump a SINGLE-env skill rollout for the proven Go2 PBR visual replay.

The policy simulates on go2_locomotion.usda (13-link physics). This writes the
GO2W v1 stream consumed by nuka_go2_walk_video, which binds each recorded link
pose to the 33 real visual meshes in go2.nks/go2.nka. Thus validation shows the
actual Go2 body, not the training collision primitives.
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, "python")

import nuka  # noqa: F401

MAGIC = 0x474F3257
VERSION = 1


def load_policy(path):
    ck = torch.load(path, map_location="cuda", weights_only=False)
    sd = ck["model"]
    mean = sd["running_mean_std.running_mean"].float()
    std = sd["running_mean_std.running_var"].clamp(min=1e-6).sqrt().float()
    layers = []
    for i in (0, 2, 4):
        w = sd[f"a2c_network.actor_mlp.{i}.weight"]
        layer = torch.nn.Linear(w.shape[1], w.shape[0])
        layer.weight.data = w.float()
        layer.bias.data = sd[f"a2c_network.actor_mlp.{i}.bias"].float()
        layers += [layer, torch.nn.ELU()]
    mu = torch.nn.Linear(128, 12)
    mu.weight.data = sd["a2c_network.mu.weight"].float()
    mu.bias.data = sd["a2c_network.mu.bias"].float()
    return torch.nn.Sequential(*layers, mu).cuda().eval(), mean, std


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["hs", "bf"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--out", default="out/go2_skill_trajectory.bin")
    ap.add_argument("--vx", type=float, default=0.2)
    args = ap.parse_args()

    if args.task == "hs":
        from nuka.tasks.go2_front_handstand import make_env
        env = make_env(1, fixed_vx=args.vx)
        seconds = 8.0 if args.seconds is None else args.seconds
    else:
        from nuka.tasks.go2_backflip import make_env
        env = make_env(1, action_mode="pd")
        seconds = 3.0 if args.seconds is None else args.seconds

    net, mean, std = load_policy(args.checkpoint)
    pose = torch.from_dlpack(env._world.buffer_view(nuka.ARTICULATION_LINK_POSE))
    drive = torch.from_dlpack(env._world.buffer_view(nuka.DRIVE_TARGET))
    obs, _ = env.reset()
    n_steps = int(seconds * 50)
    poses, drives = [], []

    with torch.no_grad():
        for _ in range(n_steps):
            action = net(((obs - mean) / std).float()).clamp(-1.0, 1.0)
            obs, _r, _term, _trunc, _info = env.step(action)
            nuka.sync()
            poses.append(pose[0, :13].detach().cpu().numpy().copy())
            drives.append(drive[0, :13].detach().cpu().numpy().copy())
    env.close()

    pose_np = np.stack(poses).astype("<f4")
    drive_np = np.stack(drives).astype("<f4")
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(struct.pack("<8i2f", MAGIC, VERSION, n_steps, 13, 13, 7,
                            1, 50000, 0.02, 0.02))
        flat = np.concatenate((drive_np, pose_np.reshape(n_steps, 91)), axis=1)
        f.write(flat.astype("<f4").tobytes(order="C"))
    print(f"[dump-skill] {args.task} {n_steps} control frames -> {out}")


if __name__ == "__main__":
    main()
