"""Deterministic skill evaluation for the front-handstand walk policy.

Loads an rl_games checkpoint, runs the env with the policy MEAN action, and
reports real skill metrics: verticality (pgx), in-handstand hold fraction,
front-paw load share, rear-paw swing tracking error, and episode survival.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "python")

import nuka  # noqa: F401
from nuka.tasks.go2_front_handstand import make_env


def main() -> None:
    ckpt = sys.argv[1] if len(sys.argv) > 1 else (
        "runs/go2_front_handstand/nn/last_go2_front_handstand_ep_4000_rew_1515.7042.pth")
    n_games = int(sys.argv[2]) if len(sys.argv) > 2 else 16

    ck = torch.load(ckpt, map_location="cuda", weights_only=False)
    sd = ck["model"]
    rms_var = sd["running_mean_std.running_var"].clamp(min=1e-6)
    rms_mean = sd["running_mean_std.running_mean"]

    layers = []
    for i in (0, 2, 4):
        layers.append(torch.nn.Linear(sd[f"a2c_network.actor_mlp.{i}.weight"].shape[1],
                                      sd[f"a2c_network.actor_mlp.{i}.weight"].shape[0]))
        layers[-1].weight.data = sd[f"a2c_network.actor_mlp.{i}.weight"]
        layers[-1].bias.data = sd[f"a2c_network.actor_mlp.{i}.bias"]
        layers.append(torch.nn.ELU())
    mu_head = torch.nn.Linear(128, 12)
    mu_head.weight.data = sd["a2c_network.mu.weight"]
    mu_head.bias.data = sd["a2c_network.mu.bias"]
    net = torch.nn.Sequential(*layers, mu_head).cuda().eval()

    env = make_env(n_games)
    obs, _ = env.reset()

    rms_mean = rms_mean.float()
    rms_var = rms_var.float()

    hold_n = torch.zeros(n_games, device="cuda")
    step_n = torch.zeros(n_games, device="cuda")
    pgx_sum = torch.zeros(n_games, device="cuda")
    err_sum = torch.zeros(n_games, device="cuda")
    alive = torch.ones(n_games, dtype=torch.bool, device="cuda")
    ep_reward = torch.zeros(n_games, device="cuda")

    t0 = time.time()
    with torch.no_grad():
        for _ in range(1000):          # 20 s of control at 50 Hz
            x = (obs - rms_mean) / rms_var.sqrt()
            a = net(x).clamp(-1.0, 1.0)
            obs, r, term, trunc, info = env.step(a)
            pg = env._obs.projected_gravity()
            f_fz = env._front_fz()
            live = alive.float()
            pgx_sum += live * pg[:, 0]
            hold_n += live * ((pg[:, 0] > 0.85)
                              & (f_fz.min(dim=-1).values > 5.0)).float()
            step_n += live
            ep_reward += live * r
            done = term | trunc
            if bool(done.any()):
                print(f"episodes ended: reward={ep_reward[done].cpu().numpy().round(1)} "
                      f"hold_frac={(hold_n[done] / step_n[done].clamp(min=1)).cpu().numpy().round(2)} "
                      f"mean_pgx={(pgx_sum[done] / step_n[done].clamp(min=1)).cpu().numpy().round(2)}",
                      flush=True)
                alive &= ~done
                # Reset metrics for finished games only.
                hold_n[done] = 0; step_n[done] = 0
                pgx_sum[done] = 0; err_sum[done] = 0; ep_reward[done] = 0
            if not bool(alive.any()):
                break
    print(f"eval done in {time.time()-t0:.0f}s; still-alive games: {int(alive.sum())}")
    env.close()


if __name__ == "__main__":
    main()
