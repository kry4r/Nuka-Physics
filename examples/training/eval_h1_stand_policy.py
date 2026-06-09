#!/usr/bin/env python
"""Evaluate an H1 standing checkpoint in the reduced torque training world.

This is a training-side deployment proxy, not the final co-resident full-world gate.
It runs deterministic policy means and prints the S3-style standing metrics:
max tilt, minimum active contacts, maximum normalized torque, and episode length.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_CFG = os.path.join(_HERE, "h1_stand_ppo_cfg.yaml")


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate H1 standing PPO checkpoint.")
    p.add_argument("--cfg", default=_DEFAULT_CFG)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_actors", type=int, default=64)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _tilt_deg(base_quat_wxyz: torch.Tensor) -> torch.Tensor:
    qx = base_quat_wxyz[:, 1]
    qy = base_quat_wxyz[:, 2]
    cos_up = (1.0 - 2.0 * (qx * qx + qy * qy)).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos_up))


def main() -> int:
    args = _parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        print("[eval_h1] WARNING: CUDA_VISIBLE_DEVICES unset; use CUDA_VISIBLE_DEVICES=0.", file=sys.stderr)

    import nuka
    import nuka.rl_games  # noqa: F401
    from rl_games.torch_runner import Runner
    from rl_games.common import vecenv as _vecenv
    from nuka.tasks.h1_stand import H1_BLC

    with open(args.cfg, "r") as fh:
        raw = yaml.safe_load(fh)
    params = raw["params"] if "params" in raw else raw
    cfg = params["config"]
    cfg["num_actors"] = args.num_actors
    cfg.setdefault("env_config", {})["seed"] = args.seed
    cfg["player"] = {
        "use_vecenv": True,
        "deterministic": True,
        "games_num": 1,
        "env_config": cfg.get("env_config", {}),
    }

    runner = Runner()
    runner.load_config(params)
    player = runner.create_player()
    player.restore(args.checkpoint)
    player.reset()
    player.has_batch_dimension = True

    vec_env = player.env
    env = vec_env.env
    try:
        obs = vec_env.reset()
        max_tilt = torch.zeros(args.num_actors, device="cuda")
        min_contacts = torch.full((args.num_actors,), 4, device="cuda", dtype=torch.int32)
        max_tau_norm = torch.zeros(args.num_actors, device="cuda")
        done_step = torch.zeros(args.num_actors, device="cuda", dtype=torch.int32)
        alive = torch.ones(args.num_actors, device="cuda", dtype=torch.bool)

        for step in range(1, args.steps + 1):
            action = player.get_action(obs, is_deterministic=True).to("cuda")
            obs, reward, dones, infos = vec_env.step(action)

            bp = torch.from_dlpack(env._world.buffer_view(nuka.BASE_POSE)).view(args.num_actors, 7)
            cf = torch.from_dlpack(env._world.buffer_view(nuka.CONTACT_FORCE)).view(args.num_actors, -1, 3)
            tq = torch.from_dlpack(env._world.buffer_view(nuka.TORQUE_INPUT)).view(args.num_actors, H1_BLC)
            tilt = _tilt_deg(bp[:, 3:7])
            contacts = (cf.norm(dim=-1) > 1e-4).sum(dim=1).to(torch.int32)
            tau_norm = (tq[:, 1:11] / env.torque_limits.view(1, 10)).abs().amax(dim=1)

            max_tilt = torch.maximum(max_tilt, tilt)
            min_contacts = torch.minimum(min_contacts, contacts)
            max_tau_norm = torch.maximum(max_tau_norm, tau_norm)

            just_done = alive & (dones > 0.5)
            done_step[just_done] = step
            alive &= ~just_done
            if not bool(alive.any()):
                break

        survived = done_step == 0
        effective_len = torch.where(survived, torch.full_like(done_step, args.steps), done_step)
        print(
            "[eval_h1] "
            f"checkpoint={args.checkpoint} actors={args.num_actors} steps={args.steps} "
            f"survived={int(survived.sum())}/{args.num_actors} "
            f"mean_len={float(effective_len.float().mean()):.1f} "
            f"min_len={int(effective_len.min())} "
            f"max_tilt={float(max_tilt.max()):.2f}deg "
            f"min_contacts={int(min_contacts.min())}/4 "
            f"max_tau_norm={float(max_tau_norm.max()):.3f}",
            flush=True,
        )
    finally:
        vec_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
