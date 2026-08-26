"""Deterministic skill evaluation for the backflip policy.

Reports per-episode peak backward rotation, take-off rate, landing quality,
and curriculum level reached -- the numbers the gate cares about.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

sys.path.insert(0, "python")

import nuka  # noqa: F401
from nuka.tasks.go2_backflip import make_env


def main() -> None:
    import glob
    cands = sorted(glob.glob("runs/go2_backflip_pd_pd/nn/*.pth"),
                   key=lambda p: len(p))  # stable order
    # prefer the frame-budget final save if present
    finals = [p for p in cands if "frame_40003584" in p]
    ckpt = finals[0] if finals else cands[-1]
    n_games = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    if len(sys.argv) > 2:
        cands = [sys.argv[2]]

    ck = torch.load(ckpt, map_location="cuda", weights_only=False)
    sd = ck["model"]
    rms_var = sd["running_mean_std.running_var"].clamp(min=1e-6).float()
    rms_mean = sd["running_mean_std.running_mean"].float()

    layers = []
    for i in (0, 2, 4):
        w = sd[f"a2c_network.actor_mlp.{i}.weight"]
        layers.append(torch.nn.Linear(w.shape[1], w.shape[0]))
        layers[-1].weight.data = w.float()
        layers[-1].bias.data = sd[f"a2c_network.actor_mlp.{i}.bias"].float()
        layers.append(torch.nn.ELU())
    mu_head = torch.nn.Linear(128, 12)
    mu_head.weight.data = sd["a2c_network.mu.weight"].float()
    mu_head.bias.data = sd["a2c_network.mu.bias"].float()
    net = torch.nn.Sequential(*layers, mu_head).cuda().eval()

    env = make_env(n_games, action_mode="pd")
    obs, _ = env.reset()

    peak_rot = torch.zeros(n_games, device="cuda")
    alive = torch.ones(n_games, dtype=torch.bool, device="cuda")
    took_off = torch.zeros(n_games, dtype=torch.bool, device="cuda")
    landed_upright = torch.zeros(n_games, dtype=torch.bool, device="cuda")

    with torch.no_grad():
        for _ in range(800):   # 16 s of control at 50 Hz
            x = ((obs - rms_mean) / rms_var.sqrt()).float()
            a = net(x).clamp(-1.0, 1.0)
            obs, r, term, trunc, _info = env.step(a)
            live = alive.float()
            rot = -env._bf.rot_acc
            peak_rot = torch.maximum(peak_rot, live * rot)
            took_off |= env._took_off & alive
            landed_upright |= ((env._settle_steps >= 25) & alive).float() > 0.5
            done = term | trunc
            if bool(done.any()):
                deg = (peak_rot[done] * 180 / np.pi).cpu().numpy().round(0)
                toff = took_off[done].cpu().numpy().tolist()
                land = landed_upright[done].cpu().numpy().tolist()
                print(f"episodes ended: peak_rotation_deg={deg} "
                      f"took_off={toff} settled_upright={land}", flush=True)
                alive &= ~done
                peak_rot[done] = 0; took_off[done] = False
                landed_upright[done] = False
            if not bool(alive.any()):
                break
    print(f"still-alive games: {int(alive.sum())}")
    env.close()


if __name__ == "__main__":
    main()
