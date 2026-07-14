#!/usr/bin/env python
"""DETERMINISTIC eval of a TRAINED BDX walk policy -- honest S1 gate verdict.

Loads an rl_games checkpoint's ACTOR (obs running_mean_std + MLP -> mu), drives
BdxWalkEnv deterministically (action = mu, no sigma), and over ``num_envs`` x
``seconds`` reports the S1 acceptance metrics:

  * fall rate      : fraction of envs that terminated (fell) at least once.
  * lin-vel track  : mean ||cmd_xy - base_lin_vel_xy|| over first-episode steps.
  * stand height   : mean base z over first-episode steps.
  * NaN            : any non-finite obs/state.

GATE: fall rate < 5%, lin-vel tracking error < 0.08 m/s, stand height > 0.17 m, no NaN.

Run::

  LD_LIBRARY_PATH=... CUDA_VISIBLE_DEVICES=0 python examples/training/bdx_walk_eval.py \
      --ckpt /data/xtzhang25/_work/activate/out/bdx_walk/nn/bdx_walk.pth \
      --num-envs 1000 --seconds 20
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import nuka  # noqa: F401
from nuka.tasks.bdx_locomotion import make_env
from nuka.tasks import bdx_obs as B


def load_actor(ckpt_path, obs_dim, act_dim, device):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = sd["model"]
    g = {k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k: v
         for k, v in m.items()}
    rmean = g["running_mean_std.running_mean"].to(device).float()
    rvar = g["running_mean_std.running_var"].to(device).float()
    assert rmean.numel() == obs_dim, f"ckpt obs dim {rmean.numel()} != {obs_dim}"
    eps = 1e-5
    w0 = g["a2c_network.actor_mlp.0.weight"].to(device).float()
    b0 = g["a2c_network.actor_mlp.0.bias"].to(device).float()
    w2 = g["a2c_network.actor_mlp.2.weight"].to(device).float()
    b2 = g["a2c_network.actor_mlp.2.bias"].to(device).float()
    w4 = g["a2c_network.actor_mlp.4.weight"].to(device).float()
    b4 = g["a2c_network.actor_mlp.4.bias"].to(device).float()
    wmu = g["a2c_network.mu.weight"].to(device).float()
    bmu = g["a2c_network.mu.bias"].to(device).float()
    assert wmu.shape[0] == act_dim, f"mu dim {wmu.shape[0]} != {act_dim}"
    elu = torch.nn.functional.elu

    @torch.no_grad()
    def forward(obs):
        x = ((obs - rmean) / torch.sqrt(rvar + eps)).clamp(-5.0, 5.0)
        x = elu(torch.nn.functional.linear(x, w0, b0))
        x = elu(torch.nn.functional.linear(x, w2, b2))
        x = elu(torch.nn.functional.linear(x, w4, b4))
        mu = torch.nn.functional.linear(x, wmu, bmu)
        # rl_games trains with clip_actions=True: the env only ever saw [-1,1].
        return mu.clamp_(-1.0, 1.0)

    return forward, sd.get("epoch"), sd.get("last_mean_rewards")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Deterministic BDX walk eval (S1 gate).")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--num-envs", type=int, default=1000)
    p.add_argument("--seconds", type=float, default=20.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--sweep", action="store_true",
                   help="Also run a fixed-vx command sweep diagnostic.")
    p.add_argument("--domain", choices=("demo", "train"), default="demo",
                   help="Command sampling domain: 'demo' = the historical gate "
                        "domain (D01 envelope, comparable across recipe pivots); "
                        "'train' = the env default (Playground joystick domain).")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env_kw = {}
    if args.domain == "demo":
        env_kw["command_ranges"] = {
            "lin_vel_x": (-0.10, 0.15),
            "lin_vel_y": (-0.08, 0.08),
            "ang_vel_yaw": (-0.4, 0.4),
        }
    env = make_env(args.num_envs, seed=args.seed, **env_kw)
    print(f"[eval] command domain = {args.domain}")
    dev = env._dev
    n = env.num_envs
    steps = int(round(args.seconds / (env.dt * env.decimation)))
    forward, ep, rew = load_actor(args.ckpt, B.BDX_OBS_DIM, B.BDX_ACTION_DIM, dev)
    print(f"[eval] ckpt={args.ckpt} epoch={ep} last_mean_rewards={rew}")
    print(f"[eval] num_envs={n} steps={steps} ({args.seconds}s @ {env.dt*env.decimation}s ctrl)")

    obs, _ = env.reset(seed=args.seed)
    alive = torch.ones(n, dtype=torch.bool, device=dev)   # still on first episode
    ever_fell = torch.zeros(n, dtype=torch.bool, device=dev)
    cnt = torch.zeros(n, device=dev)
    sum_track = torch.zeros(n, device=dev)
    sum_z = torch.zeros(n, device=dev)
    any_nan = False

    for k in range(steps):
        mu = forward(obs)
        obs, reward, terminated, truncated, _ = env.step(mu)
        any_nan |= not bool(torch.isfinite(obs).all())
        vel = env._obs.base_lin_vel()[:, :2]
        cmd = env.command[:, :2]
        track = (cmd - vel).norm(dim=1)
        z = env._obs.base_pos()[:, 2]
        m = alive
        cnt[m] += 1.0
        sum_track[m] += track[m]
        sum_z[m] += z[m]
        ever_fell |= terminated
        alive = alive & (~terminated)

    fall_rate = float(ever_fell.float().mean())
    valid = cnt > 0
    track_err = float((sum_track[valid] / cnt[valid]).mean())
    stand_z = float((sum_z[valid] / cnt[valid]).mean())
    surv_frac = float(alive.float().mean())

    print(f"[eval] fall_rate       = {fall_rate*100:.2f}%   (gate < 5%)")
    print(f"[eval] lin_vel_track   = {track_err:.4f} m/s (gate < 0.08)")
    print(f"[eval] stand_height    = {stand_z:.4f} m   (gate > 0.17)")
    print(f"[eval] survivors@{args.seconds}s = {surv_frac*100:.1f}%   nan={any_nan}")
    gates = {
        "fall<5%": fall_rate < 0.05,
        "track<0.08": track_err < 0.08,
        "height>0.17": stand_z > 0.17,
        "no-nan": not any_nan,
    }
    for k2, v in gates.items():
        print(f"[gate] {k2}: {'PASS' if v else 'FAIL'}")
    verdict = all(gates.values())
    print(f"[gate] S1 WALK {'PASS' if verdict else 'FAIL'}")

    if args.sweep:
        print("\n[sweep] fixed-command achieved velocity (mean over last 50% of 10s):")
        for cx in (-0.15, 0.0, 0.10, 0.20, 0.25):
            env.command[:] = torch.tensor([cx, 0.0, 0.0], device=dev)
            env._fixed_command = True
            obs, _ = env.reset(seed=args.seed)
            env.command[:] = torch.tensor([cx, 0.0, 0.0], device=dev)
            vs = []
            for k in range(500):
                mu = forward(obs)
                obs, *_ = env.step(mu)
                if k >= 250:
                    vs.append(env._obs.base_lin_vel()[:, 0].mean().item())
            print(f"  cmd vx={cx:+.2f} -> achieved vx={np.mean(vs):+.3f}")

    env.close()
    return 0 if verdict else 3


if __name__ == "__main__":
    raise SystemExit(main())
