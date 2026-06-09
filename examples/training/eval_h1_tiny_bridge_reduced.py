#!/usr/bin/env python
"""Evaluate the exported tiny bridge MLP in the reduced H1 torque env.

The tiny bridge outputs normalized actions. The Python reduced env applies the
same per-joint action-to-torque scaling as deploy, so this evaluator proves the
exported bridge policy before full-world co-resident deployment is blamed.
"""

from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = REPO / "python/spikes/out/h1_bridge_mlp.bin"
OBS_DIM = 32
ACT_DIM = 10
H = 64


class TinyMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(OBS_DIM, H)
        self.fc2 = torch.nn.Linear(H, H)
        self.fc3 = torch.nn.Linear(H, ACT_DIM)

    def forward(self, x):
        return self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(x)))))


def load_tiny(path: pathlib.Path) -> TinyMLP:
    flat = np.fromfile(path, dtype=np.float32)
    expect = (H * OBS_DIM + H) + (H * H + H) + (ACT_DIM * H + ACT_DIM)
    if flat.size != expect:
        raise ValueError(f"{path} has {flat.size} floats, expected {expect}")
    model = TinyMLP().cuda().float()
    off = 0
    with torch.no_grad():
        for layer, rows, cols in ((model.fc1, H, OBS_DIM), (model.fc2, H, H), (model.fc3, ACT_DIM, H)):
            n = rows * cols
            layer.weight.copy_(torch.from_numpy(flat[off:off+n].reshape(rows, cols)).cuda())
            off += n
            layer.bias.copy_(torch.from_numpy(flat[off:off+rows]).cuda())
            off += rows
    model.eval()
    return model


def _tilt_deg(q):
    cos_up = (1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos_up))


def main() -> int:
    p = argparse.ArgumentParser(description="Evaluate exported H1 tiny bridge in reduced env.")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        print("[eval_tiny] WARNING: CUDA_VISIBLE_DEVICES unset; use CUDA_VISIBLE_DEVICES=0.")

    import nuka
    from nuka.tasks.h1_stand import H1_BLC, make_env

    model = load_tiny(pathlib.Path(args.weights))
    env = make_env(args.num_envs, seed=args.seed)
    try:
        obs, _ = env.reset(seed=args.seed)
        max_tilt = torch.zeros(args.num_envs, device="cuda")
        min_contacts = torch.full((args.num_envs,), 4, device="cuda", dtype=torch.int32)
        max_tau_norm = torch.zeros(args.num_envs, device="cuda")
        done_step = torch.zeros(args.num_envs, device="cuda", dtype=torch.int32)
        alive = torch.ones(args.num_envs, device="cuda", dtype=torch.bool)
        for step in range(1, args.steps + 1):
            with torch.no_grad():
                action = model(obs).clamp(-1.0, 1.0)
            obs, reward, terminated, truncated, _ = env.step(action)
            bp = torch.from_dlpack(env._world.buffer_view(nuka.BASE_POSE)).view(args.num_envs, 7)
            cf = torch.from_dlpack(env._world.buffer_view(nuka.CONTACT_FORCE)).view(args.num_envs, -1, 3)
            tq = torch.from_dlpack(env._world.buffer_view(nuka.TORQUE_INPUT)).view(args.num_envs, H1_BLC)
            max_tilt = torch.maximum(max_tilt, _tilt_deg(bp[:, 3:7]))
            min_contacts = torch.minimum(min_contacts, (cf.norm(dim=-1) > 1e-4).sum(dim=1).to(torch.int32))
            max_tau_norm = torch.maximum(max_tau_norm, (tq[:, 1:11] / env.torque_limits.view(1, ACT_DIM)).abs().amax(dim=1))
            dones = terminated | truncated
            just_done = alive & dones
            done_step[just_done] = step
            alive &= ~just_done
            if not bool(alive.any()):
                break
        survived = done_step == 0
        effective_len = torch.where(survived, torch.full_like(done_step, args.steps), done_step)
        print(
            "[eval_tiny] "
            f"weights={args.weights} actors={args.num_envs} steps={args.steps} "
            f"survived={int(survived.sum())}/{args.num_envs} "
            f"mean_len={float(effective_len.float().mean()):.1f} "
            f"min_len={int(effective_len.min())} "
            f"max_tilt={float(max_tilt.max()):.2f}deg "
            f"min_contacts={int(min_contacts.min())}/4 "
            f"max_tau_norm={float(max_tau_norm.max()):.3f}",
            flush=True,
        )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
