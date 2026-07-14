#!/usr/bin/env python3
"""Reusable structured-observation gate for the BDX depth policy.

This is a pipeline/scenario check, not a unit test.  It creates the real cooked
corridor, mounts the real head camera, exercises its lower-rate frame cadence,
builds the rl_games fusion network, and proves that the zero visual residual starts
from the supplied blind actor exactly.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
import torch.nn.functional as F

from nuka.rl_games.bdx_perception_network import BdxDepthFusionNetwork
from nuka.tasks import bdx_obs as B
from nuka.tasks.bdx_perception import CORRIDOR_SCENE, make_env


def _checkpoint_actor(checkpoint: str, obs: torch.Tensor) -> torch.Tensor:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["model"]
    plain = {
        (key[len("_orig_mod."):] if key.startswith("_orig_mod.") else key): value
        for key, value in state.items()
    }
    dev = obs.device
    x = torch.clamp(
        (obs.float() - plain["running_mean_std.running_mean"].to(dev).float())
        / torch.sqrt(
            plain["running_mean_std.running_var"].to(dev).float() + 1.0e-5),
        -5.0,
        5.0,
    )
    for index in (0, 2, 4):
        x = F.elu(F.linear(
            x,
            plain[f"a2c_network.actor_mlp.{index}.weight"].to(dev),
            plain[f"a2c_network.actor_mlp.{index}.bias"].to(dev),
        ))
    return F.linear(
        x,
        plain["a2c_network.mu.weight"].to(dev),
        plain["a2c_network.mu.bias"].to(dev),
    )


def run(args) -> dict:
    started = time.perf_counter()
    env_kwargs = {}
    if args.episode_steps:
        # BdxWalkEnv rounds seconds to control steps.  Make the scenario short
        # enough to exercise its real autoreset path without a long rollout.
        env_kwargs["episode_length_s"] = args.episode_steps * 0.005 * 4
    env = make_env(
        args.envs,
        scene=args.scene,
        seed=args.seed,
        fixed_command=True,
        command=(0.15, 0.0, 0.0),
        push_enable=False,
        spawn_x_range=(args.spawn_x_low, args.spawn_x_high),
        forward_progress_scale=args.forward_progress_scale,
        progress_milestones=args.progress_milestones,
        progress_milestone_bonus=args.progress_milestone_bonus,
        success_progress_m=0.5 if args.force_success_reset else 0.0,
        success_bonus=100.0,
        fall_penalty=20.0,
        sensor={
            "width": args.width,
            "height": args.height,
            "update_period": args.update_period,
        },
        **env_kwargs,
    )
    try:
        obs, _ = env.reset(seed=args.seed)
        spawn_x = env._obs.base_pos()[:, 0].detach().clone()
        if (float(spawn_x.min()) < args.spawn_x_low
                or float(spawn_x.max()) > args.spawn_x_high):
            raise RuntimeError(
                f"spawn x escaped [{args.spawn_x_low},{args.spawn_x_high}]")
        shapes = {key: list(value.shape) for key, value in obs.items()}
        dtypes = {key: str(value.dtype) for key, value in obs.items()}
        expected = {
            "proprio": [args.envs, B.BDX_OBS_DIM],
            "depth": [args.envs, 1, args.height, args.width],
            "depth_age": [args.envs, 1],
        }
        if shapes != expected:
            raise RuntimeError(f"observation shapes {shapes} != {expected}")
        if obs["depth"].dtype != torch.float16:
            raise RuntimeError(f"depth rollout dtype is {obs['depth'].dtype}, not fp16")
        if not bool(torch.isfinite(obs["depth"]).all()):
            raise RuntimeError("normalized depth contains non-finite values")
        if float(obs["depth"].min()) < 0.0 or float(obs["depth"].max()) > 1.0:
            raise RuntimeError("normalized depth escaped [0,1]")
        cross_env_depth_delta = 0.0
        if args.envs > 1:
            cross_env_depth_delta = float(
                (obs["depth"][0] - obs["depth"][1]).abs().max())
            if float((spawn_x[0] - spawn_x[1]).abs()) > 1.0e-5 \
                    and cross_env_depth_delta == 0.0:
                raise RuntimeError("different reset poses produced identical depth")

        input_shape = {
            "proprio": (B.BDX_OBS_DIM,),
            "depth": (1, args.height, args.width),
            "depth_age": (1,),
        }
        network = BdxDepthFusionNetwork(
            {
                "depth_latent_dim": args.latent_dim,
                "proprio_units": [512, 256, 128],
                "bootstrap_checkpoint": args.checkpoint,
            },
            actions_num=B.BDX_ACTION_DIM,
            input_shape=input_shape,
            value_size=1,
            num_seqs=args.envs,
        ).to(env._dev).eval()
        with torch.no_grad():
            mu, logstd, value, _ = network({"obs": obs})
            blind_mu = _checkpoint_actor(args.checkpoint, obs["proprio"])
        parity = float((mu - blind_mu).abs().max())
        if parity != 0.0:
            raise RuntimeError(
                f"zero visual residual changed the blind actor (max |delta|={parity})")

        ages = [float(obs["depth_age"][0, 0])]
        expected_ages = [0.0]
        rewards = []
        autoreset_count = 0
        success_autoreset_count = 0
        progress_reset_max_error = 0.0
        success_primed = False
        actions = mu.clamp(-1.0, 1.0)
        for _ in range(args.steps):
            # Prime the episodic accumulators immediately before the forced
            # truncation.  This makes the reset check meaningful even when the
            # short rollout has not naturally reached a distance milestone.
            about_to_truncate = bool(
                args.episode_steps
                and (env.episode_step + 1 >= env.max_episode_length).all())
            if about_to_truncate:
                env._episode_max_progress.fill_(123.0)
                env._milestones_reached.fill_(True)
            if args.force_success_reset and not success_primed:
                env._episode_start_x.copy_(
                    env._obs.base_pos()[:, 0] - env.success_progress_m - 1.0)
                success_primed = True
            obs, reward, terminated, truncated, info = env.step(actions)
            ages.append(float(obs["depth_age"][0, 0]))
            rewards.append(float(reward.mean()))
            with torch.no_grad():
                actions = network({"obs": obs})[0].clamp(-1.0, 1.0)
            done = terminated | truncated
            success = info["success"]
            if bool(success.any()):
                success_autoreset_count += int(success.sum())
                if bool((success & ~terminated).any()) \
                        or bool((success & truncated).any()):
                    raise RuntimeError("success was not a clean task termination")
                if float(reward[success].min()) < env.success_bonus:
                    raise RuntimeError("success bonus was not applied before autoreset")
            if bool(done.any()):
                autoreset_count += int(done.sum())
                # A reset forces a fresh frame; the env's stale-frame invariant is
                # checked by the zero age below.
                if float(obs["depth_age"].max()) != 0.0:
                    raise RuntimeError("an autoreset observation retained a stale frame")
                start_error = float((
                    env._episode_start_x[done]
                    - env._obs.base_pos()[done, 0]
                ).abs().max())
                max_progress = float(env._episode_max_progress[done].abs().max())
                reached = bool(env._milestones_reached[done].any())
                progress_reset_max_error = max(
                    progress_reset_max_error, start_error, max_progress)
                if start_error != 0.0 or max_progress != 0.0 or reached:
                    raise RuntimeError(
                        "an autoreset retained episodic progress state: "
                        f"start_error={start_error} max_progress={max_progress} "
                        f"milestone_reached={reached}")
            expected_ages.append(
                0.0 if bool(done.any()) else
                (env._sensor_tick % args.update_period) * (env.dt * env.decimation))

        if args.episode_steps and autoreset_count == 0:
            raise RuntimeError("short-episode gate did not exercise autoreset")
        if args.force_success_reset and success_autoreset_count == 0:
            raise RuntimeError("success gate did not exercise success autoreset")
        age_error = max(abs(a - b) for a, b in zip(ages, expected_ages))
        if age_error > 1.0e-6:
            raise RuntimeError(
                f"camera age cadence {ages} != expected {expected_ages}")

        depth_bytes = (
            args.horizon * args.envs * args.width * args.height * 2)
        result = {
            "scene": args.scene,
            "envs": args.envs,
            "observation_shapes": shapes,
            "observation_dtypes": dtypes,
            "policy_encoding": {
                "raw_depth_pixels": args.width * args.height,
                "depth_latent": args.latent_dim,
                "visual_adapter_input": args.latent_dim + 1,
                "raw_proprio": B.BDX_OBS_DIM,
                "proprio_latent": 128,
                "policy_fused_state": 128,
                "fusion": "additive_visual_residual",
            },
            "rollout_depth_mib": depth_bytes / (1024.0 * 1024.0),
            "blind_bootstrap_max_abs": parity,
            "frame_ages_s": ages,
            "frame_age_max_error": age_error,
            "autoreset_count": autoreset_count,
            "success_autoreset_count": success_autoreset_count,
            "progress_reset_max_error": progress_reset_max_error,
            "spawn_x_min": float(spawn_x.min()),
            "spawn_x_max": float(spawn_x.max()),
            "depth_min": float(obs["depth"].min()),
            "depth_max": float(obs["depth"].max()),
            "cross_env_depth_max_abs": cross_env_depth_delta,
            "mu_shape": list(mu.shape),
            "logstd_shape": list(logstd.shape),
            "value_shape": list(value.shape),
            "mean_rewards": rewards,
            "elapsed_s": time.perf_counter() - started,
        }
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default=CORRIDOR_SCENE)
    parser.add_argument(
        "--checkpoint",
        default=("/data/xtzhang25/_work/activate/out/bdx_walk_v9/nn/"
                 "last_bdx_walk_v9_ep_2000_rew_447.30893.pth"),
    )
    parser.add_argument("--envs", type=int, default=8)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--height", type=int, default=96)
    parser.add_argument("--update-period", type=int, default=2)
    parser.add_argument("--latent-dim", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--episode-steps", type=int, default=0,
        help="force this many control steps per episode and require autoreset")
    parser.add_argument(
        "--force-success-reset", action="store_true",
        help="force one goal completion and verify success autoreset semantics")
    parser.add_argument("--forward-progress-scale", type=float, default=15.0)
    parser.add_argument(
        "--progress-milestones", type=float, nargs="+",
        default=(0.35, 0.65, 1.2))
    parser.add_argument("--progress-milestone-bonus", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--spawn-x-low", type=float, default=-0.32)
    parser.add_argument("--spawn-x-high", type=float, default=-0.12)
    args = parser.parse_args()
    if min(args.envs, args.width, args.height, args.update_period,
           args.latent_dim, args.horizon, args.steps) <= 0:
        parser.error("all numeric arguments must be positive")
    if not args.spawn_x_low < args.spawn_x_high:
        parser.error("--spawn-x-low must be smaller than --spawn-x-high")
    if args.episode_steps < 0:
        parser.error("--episode-steps must be non-negative")
    if args.episode_steps and args.steps < args.episode_steps:
        parser.error("--steps must reach --episode-steps")
    if args.episode_steps and args.force_success_reset:
        parser.error("choose either a time-limit reset or a success reset gate")
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
