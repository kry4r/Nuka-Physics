#!/usr/bin/env python3
"""Reusable closed-loop trajectory gate for the BDX perception policy.

This is an end-to-end pipeline/scenario check, not a unit test.  It creates the
real sharp-edge rigid corridor, mounts the real head camera, restores a fused
rl_games checkpoint, and drives the free-base robot without pose overrides.

Besides traversal metrics, the gate evaluates the same observation with normal
and zeroed depth.  Their deterministic-action difference proves whether the
compact visual branch actually influences the policy instead of merely occupying
rollout memory.  ``--drive-depth zero`` provides a causal rollout ablation using
the identical seed and reset distribution.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from nuka.rl_games.bdx_perception_network import BdxDepthFusionNetwork
from nuka.tasks import bdx_obs as B
from nuka.tasks.bdx_perception import CORRIDOR_SCENE, make_env


def _plain_state(checkpoint: dict) -> dict:
    result = {}
    for key, value in checkpoint["model"].items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        result[key] = value
    return result


def _load_actor(path: str, env, latent_dim: int):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    network = BdxDepthFusionNetwork(
        {
            "depth_latent_dim": latent_dim,
            "proprio_units": [512, 256, 128],
        },
        actions_num=B.BDX_ACTION_DIM,
        input_shape={
            "proprio": (B.BDX_OBS_DIM,),
            "depth": (1, env.depth_height, env.depth_width),
            "depth_age": (1,),
        },
        value_size=1,
        num_seqs=env.num_envs,
    ).to(env._dev)
    source = _plain_state(checkpoint)
    prefix = "a2c_network."
    state = {
        key[len(prefix):]: value
        for key, value in source.items()
        if key.startswith(prefix)
    }
    network.load_state_dict(state, strict=True)
    network.eval()
    epoch = checkpoint.get("epoch")
    reward = checkpoint.get("last_mean_rewards")
    return (
        network,
        None if epoch is None else int(epoch),
        None if reward is None else float(reward),
    )


def run(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.perf_counter()
    env = make_env(
        args.envs,
        scene=args.scene,
        seed=args.seed,
        fixed_command=True,
        command=(args.command_vx, 0.0, 0.0),
        episode_length_s=args.seconds + 1.0,
        push_enable=False,
        spawn_x_range=(args.spawn_x_low, args.spawn_x_high),
        success_progress_m=args.success_progress_m,
        success_bonus=0.0,
        fall_penalty=0.0,
        sensor={
            "width": args.width,
            "height": args.height,
            "update_period": args.update_period,
        },
    )
    try:
        network, epoch, checkpoint_reward = _load_actor(
            args.checkpoint, env, args.latent_dim)
        obs, _ = env.reset(seed=args.seed)
        spawn_x = env._obs.base_pos()[:, 0].detach().clone()
        zero_depth = torch.zeros_like(obs["depth"])

        active = torch.ones(args.envs, dtype=torch.bool, device=env._dev)
        ever_fell = torch.zeros_like(active)
        ever_succeeded = torch.zeros_like(active)
        max_x = spawn_x.clone()
        last_x = spawn_x.clone()
        vx_sum = torch.zeros(args.envs, device=env._dev)
        sample_count = torch.zeros(args.envs, device=env._dev)
        action_zero_abs_sum = torch.zeros(args.envs, device=env._dev)
        action_zero_abs_max = torch.zeros(args.envs, device=env._dev)
        any_nonfinite = False
        control_dt = env.dt * env.decimation
        steps = int(round(args.seconds / control_dt))

        with torch.no_grad():
            for _ in range(steps):
                if not bool(active.any()):
                    break
                x = env._obs.base_pos()[:, 0]
                vx = env._obs.base_lin_vel()[:, 0]
                max_x[active] = torch.maximum(max_x[active], x[active])
                last_x[active] = x[active]
                vx_sum[active] += vx[active]
                sample_count[active] += 1.0

                normal_mu = network({"obs": obs})[0].clamp(-1.0, 1.0)
                zero_obs = {
                    "proprio": obs["proprio"],
                    "depth": zero_depth,
                    "depth_age": obs["depth_age"],
                }
                zero_mu = network({"obs": zero_obs})[0].clamp(-1.0, 1.0)
                delta = (normal_mu - zero_mu).abs()
                action_zero_abs_sum[active] += delta[active].mean(dim=1)
                action_zero_abs_max[active] = torch.maximum(
                    action_zero_abs_max[active], delta[active].amax(dim=1))

                actions = normal_mu if args.drive_depth == "normal" else zero_mu
                obs, _, terminated, truncated, info = env.step(actions)
                any_nonfinite |= not all(
                    bool(torch.isfinite(value).all()) for value in obs.values())
                success = active & info["success"]
                if bool(success.any()):
                    goal_x = spawn_x + args.success_progress_m
                    max_x[success] = torch.maximum(max_x[success], goal_x[success])
                    last_x[success] = goal_x[success]
                ever_succeeded |= success
                done = active & (terminated | truncated)
                ever_fell |= active & info["fall"]
                active &= ~done

        valid = sample_count > 0
        per_env_vx = torch.where(
            valid, vx_sum / sample_count.clamp_min(1.0), torch.zeros_like(vx_sum))
        per_env_action_delta = torch.where(
            valid,
            action_zero_abs_sum / sample_count.clamp_min(1.0),
            torch.zeros_like(action_zero_abs_sum),
        )
        first_clear = max_x >= args.first_clear_x
        two_step_clear = max_x >= args.two_step_clear_x
        reached = max_x >= args.reach_x
        result = {
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": epoch,
            "checkpoint_reward": checkpoint_reward,
            "drive_depth": args.drive_depth,
            "seed": args.seed,
            "envs": args.envs,
            "seconds": args.seconds,
            "steps_executed": int(sample_count.max().item()),
            "spawn_x_min": float(spawn_x.min()),
            "spawn_x_max": float(spawn_x.max()),
            "furthest_x": float(max_x.max()),
            "mean_max_x": float(max_x.mean()),
            "median_max_x": float(max_x.median()),
            "mean_progress": float((max_x - spawn_x).mean()),
            "mean_last_x": float(last_x.mean()),
            "first_clear_x": args.first_clear_x,
            "first_clear_rate": float(first_clear.float().mean()),
            "two_step_clear_x": args.two_step_clear_x,
            "two_step_clear_rate": float(two_step_clear.float().mean()),
            "reach_x": args.reach_x,
            "reach_rate": float(reached.float().mean()),
            "fall_rate": float(ever_fell.float().mean()),
            "success_progress_m": args.success_progress_m,
            "success_rate": float(ever_succeeded.float().mean()),
            "survival_rate": float(active.float().mean()),
            "mean_vx": float(per_env_vx.mean()),
            "mean_first_episode_s": float((sample_count * control_dt).mean()),
            "normal_vs_zero_action_mean_abs": float(per_env_action_delta.mean()),
            "normal_vs_zero_action_max_abs": float(action_zero_abs_max.max()),
            "vision_adapter_weight_l2": float(network.vision_adapter.weight.norm()),
            "nonfinite": any_nonfinite,
            "elapsed_s": time.perf_counter() - started,
        }
        result["gates"] = {
            "finite": not any_nonfinite,
            "vision_influences_action": (
                result["normal_vs_zero_action_mean_abs"]
                >= args.min_action_depth_delta),
            "fall_rate": result["fall_rate"] <= args.max_fall_rate,
            "success_rate": result["success_rate"] >= args.min_success_rate,
            "two_step_clear_rate": (
                result["two_step_clear_rate"] >= args.min_two_step_rate),
            "reach_rate": result["reach_rate"] >= args.min_reach_rate,
            "mean_vx": result["mean_vx"] >= args.min_mean_vx,
        }
        return result
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene", default=CORRIDOR_SCENE)
    parser.add_argument("--envs", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--command-vx", type=float, default=0.15)
    parser.add_argument("--spawn-x-low", type=float, default=-0.32)
    parser.add_argument("--spawn-x-high", type=float, default=-0.12)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--update-period", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--drive-depth", choices=("normal", "zero"), default="normal")
    # Base clearances include a small body/foot margin beyond the physical edges
    # at x=0.10 and x=0.26.  They are policy gates, not collision geometry.
    parser.add_argument("--first-clear-x", type=float, default=0.15)
    parser.add_argument("--two-step-clear-x", type=float, default=0.45)
    parser.add_argument("--reach-x", type=float, default=0.90)
    parser.add_argument("--success-progress-m", type=float, default=1.20)
    parser.add_argument("--min-action-depth-delta", type=float, default=1.0e-4)
    parser.add_argument("--max-fall-rate", type=float, default=0.10)
    parser.add_argument("--min-two-step-rate", type=float, default=0.90)
    parser.add_argument("--min-reach-rate", type=float, default=0.80)
    parser.add_argument("--min-success-rate", type=float, default=0.80)
    parser.add_argument("--min-mean-vx", type=float, default=0.12)
    parser.add_argument(
        "--enforce", action="store_true",
        help="return nonzero unless every reported gate passes")
    args = parser.parse_args()
    if min(args.envs, args.seconds, args.width, args.height,
           args.update_period, args.latent_dim) <= 0:
        parser.error("envs, seconds, image dimensions, cadence and latent must be positive")
    if not args.spawn_x_low < args.spawn_x_high:
        parser.error("--spawn-x-low must be smaller than --spawn-x-high")
    if not args.first_clear_x < args.two_step_clear_x < args.reach_x:
        parser.error("clearance thresholds must be strictly increasing")
    if args.success_progress_m < 0.0:
        parser.error("--success-progress-m must be non-negative")
    result = run(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if not args.enforce or all(result["gates"].values()) else 3


if __name__ == "__main__":
    raise SystemExit(main())
