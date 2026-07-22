#!/usr/bin/env python3
"""Reusable closed-loop trajectory gate for the Go2 privileged corridor teacher.

An end-to-end pipeline/scenario check, not a unit test.  It cooks the real
authored corridor with the Go2 robot, restores an rl_games teacher checkpoint, and
drives the free-base robot with the deterministic policy mean (no pose overrides),
reporting the canonical corridor traversal metrics + gates.

``--spawn-window`` drops the robot into a single [low,high] window that follows the
terrain height (feet land on the tread/floor) so a per-window probe is possible.
``--drive-depth zero`` re-runs with the privileged forward height-profile block
zeroed (an identical-seed causal ablation), and also reports the deterministic
action delta between the full and ablated observation.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from nuka.tasks import go2_obs as G
from nuka.tasks.go2_corridor import CORRIDOR_SCENE, make_env


def _load_teacher_actor(path: str):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = {}
    for key, value in checkpoint["model"].items():
        if key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]
        state[key] = value
    epoch = checkpoint.get("epoch")
    reward = checkpoint.get("last_mean_rewards")
    return (
        state,
        None if epoch is None else int(epoch),
        None if reward is None else float(reward),
    )


def _teacher_mu(state: dict, obs: torch.Tensor, dev) -> torch.Tensor:
    mean = state["running_mean_std.running_mean"].to(dev).float()
    var = state["running_mean_std.running_var"].to(dev).float()
    x = torch.clamp((obs.float() - mean) / torch.sqrt(var + 1.0e-5), -5.0, 5.0)
    for index in (0, 2, 4):
        x = F.elu(F.linear(
            x,
            state[f"a2c_network.actor_mlp.{index}.weight"].to(dev),
            state[f"a2c_network.actor_mlp.{index}.bias"].to(dev)))
    return F.linear(
        x,
        state["a2c_network.mu.weight"].to(dev),
        state["a2c_network.mu.bias"].to(dev))


def run(args) -> dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    started = time.perf_counter()
    # A single spawn window drops the articulation onto its window's ground height
    # (feet land, no free-fall); the default range path keeps the canonical spawn.
    spawn_kwargs = (
        {"spawn_x_windows": [[args.spawn_x_low, args.spawn_x_high, 1.0]]}
        if args.spawn_window
        else {"spawn_x_range": (args.spawn_x_low, args.spawn_x_high)})
    env = make_env(
        args.envs,
        scene=args.scene,
        seed=args.seed,
        fixed_command=True,
        command=(args.command_vx, 0.0, 0.0),
        termination_height=args.termination_height,
        termination_tilt_deg=args.termination_tilt_deg,
        episode_length_s=args.seconds + 1.0,
        success_progress_m=args.success_progress_m,
        success_bonus=0.0,
        fall_penalty=0.0,
        **spawn_kwargs,
    )
    try:
        state, epoch, checkpoint_reward = _load_teacher_actor(args.checkpoint)
        # Ablation replaces the forward height-profile block with its flat-ground
        # value (all step differences zero), keeping proprio + local trunk clearance.
        priv_start = G.GO2_OBS_DIM + 1
        obs, _ = env.reset(seed=args.seed)
        spawn_x = env._obs.base_pos()[:, 0].detach().clone()

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

                normal_mu = _teacher_mu(state, obs, env._dev).clamp(-1.0, 1.0)
                zero_obs = obs.clone()
                zero_obs[:, priv_start:] = 0.0
                zero_mu = _teacher_mu(state, zero_obs, env._dev).clamp(-1.0, 1.0)
                delta = (normal_mu - zero_mu).abs()
                action_zero_abs_sum[active] += delta[active].mean(dim=1)
                action_zero_abs_max[active] = torch.maximum(
                    action_zero_abs_max[active], delta[active].amax(dim=1))

                actions = normal_mu if args.drive_depth == "normal" else zero_mu
                obs, _, terminated, truncated, info = env.step(actions)
                any_nonfinite |= not bool(torch.isfinite(obs).all())
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
            "policy": "teacher",
            "checkpoint": args.checkpoint,
            "checkpoint_epoch": epoch,
            "checkpoint_reward": checkpoint_reward,
            "drive_depth": args.drive_depth,
            "spawn_window": bool(args.spawn_window),
            "seed": args.seed,
            "envs": args.envs,
            "seconds": args.seconds,
            "command_vx": args.command_vx,
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
            "privileged_dim": int(env.privileged_dim),
            "nonfinite": any_nonfinite,
            "elapsed_s": time.perf_counter() - started,
        }
        result["gates"] = {
            "finite": not any_nonfinite,
            "privileged_influences_action": (
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
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--command-vx", type=float, default=0.4)
    parser.add_argument("--spawn-x-low", type=float, default=-0.32)
    parser.add_argument("--spawn-x-high", type=float, default=-0.12)
    parser.add_argument("--termination-height", type=float, default=0.25)
    parser.add_argument("--termination-tilt-deg", type=float, default=60.0)
    parser.add_argument("--drive-depth", choices=("normal", "zero"), default="normal")
    parser.add_argument(
        "--spawn-window", action="store_true",
        help="spawn in a single [low,high] window that follows the terrain height "
             "(feet land on the tread/floor) instead of the raised platform pose.")
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
    if min(args.envs, args.seconds) <= 0:
        parser.error("envs and seconds must be positive")
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
