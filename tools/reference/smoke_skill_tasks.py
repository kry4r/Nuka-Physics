"""Smoke test: random-torque rollouts on both skill tasks."""
from __future__ import annotations

import sys

import torch

import nuka  # noqa: F401
sys.path.insert(0, "python")

from nuka.tasks.go2_front_handstand import make_env as make_hs
from nuka.tasks.go2_backflip import make_env as make_bf


def rollout(make_env, name, steps=300):
    env = make_env(8)
    gen = torch.Generator(device="cuda").manual_seed(0)
    obs, _ = env.reset()
    print(f"[{name}] obs {tuple(obs.shape)} finite={bool(torch.isfinite(obs).all())}")
    rew_sum = torch.zeros(8, device="cuda")
    dones_total = 0
    for t in range(steps):
        a = torch.rand(8, 12, generator=gen, device="cuda") * 2 - 1
        obs, r, term, trunc, info = env.step(a)
        rew_sum += r
        done = (term | trunc)
        dones_total += int(done.sum().item())
        if not torch.isfinite(obs).all():
            print(f"  NONFINITE OBS at {t}")
            break
    pg = env._obs.projected_gravity()
    print(f"[{name}] mean_reward={rew_sum.mean().item():.2f} "
          f"dones={dones_total}/{steps * 8} "
          f"pgx_mean={pg[:, 0].mean().item():.3f} "
          f"pgz_mean={pg[:, 2].mean().item():.3f}")
    if name == "backflip":
        print(f"[{name}] info={{{', '.join(f'{k}: {v:.1f}' for k, v in info.items())}}}")
    env.close()


if __name__ == "__main__":
    rollout(make_hs, "handstand")
    rollout(make_bf, "backflip")
    print("SMOKE OK")
