#!/usr/bin/env python3
"""Drive and render the Go2 through the single full-media one-shot world."""

from __future__ import annotations

import argparse
import glob
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import nuka
from nuka.tasks import go2_obs as G
from nuka.tasks.bdx_corridor import (
    CorridorHeightProfile,
    PRIV_HEIGHT_CLAMP,
    TEACHER_PROFILE_OFFSETS,
)

from bdx_oneshot_film import camera_at


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SCENE = os.path.join(REPO, "examples", "scenes", "go2_oneshot.nks")
RL_GAMES_DT = 0.005
RL_GAMES_DECIMATION = 4
TORCHSCRIPT_DT = 0.001
TORCHSCRIPT_DECIMATION = 20


@dataclass(frozen=True)
class Policy:
    actor: Any
    policy_format: str
    provenance: str
    obs_dim: int
    epoch: int | None
    reward: float | None
    physics_dt: float
    decimation: int
    action_postprocess: str

    @property
    def control_dt(self) -> float:
        return self.physics_dt * self.decimation


def load_rl_games_actor(path: str, device: torch.device) -> Policy:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("checkpoint is not an rl_games model dictionary")
    state = {
        key.removeprefix("_orig_mod."): (
            value.to(device) if torch.is_tensor(value) else value)
        for key, value in checkpoint["model"].items()
    }
    obs_dim = int(state["running_mean_std.running_mean"].numel())
    reward = float(checkpoint.get("last_mean_rewards", float("nan")))
    return Policy(
        actor=state,
        policy_format="rl_games",
        provenance="in-house rl_games checkpoint",
        obs_dim=obs_dim,
        epoch=int(checkpoint.get("epoch", -1)),
        reward=reward if np.isfinite(reward) else None,
        physics_dt=RL_GAMES_DT,
        decimation=RL_GAMES_DECIMATION,
        action_postprocess="deterministic mu clamped to [-1, 1]",
    )


def load_torchscript_actor(path: str, device: torch.device) -> Policy:
    actor = torch.jit.load(path, map_location=device).eval()
    matrices = [parameter for parameter in actor.parameters()
                if parameter.ndim == 2]
    if not matrices:
        raise ValueError("TorchScript policy has no matrix parameters")
    obs_dim = int(matrices[0].shape[1])
    action_dim = int(matrices[-1].shape[0])
    with torch.inference_mode():
        probe = actor(torch.zeros(1, obs_dim, device=device))
    if not torch.is_tensor(probe) or tuple(probe.shape) != (1, action_dim):
        raise ValueError(
            "TorchScript policy must return one rank-2 action tensor")
    if action_dim != G.GO2_ACTION_DIM:
        raise ValueError(
            f"TorchScript action dimension {action_dim}, expected "
            f"{G.GO2_ACTION_DIM}")
    return Policy(
        actor=actor,
        policy_format="torchscript",
        provenance="external TorchScript policy",
        obs_dim=obs_dim,
        epoch=None,
        reward=None,
        physics_dt=TORCHSCRIPT_DT,
        decimation=TORCHSCRIPT_DECIMATION,
        action_postprocess="raw policy output; Go2ObsBuilder safety clamp only",
    )


def load_policy(path: str, device: torch.device, policy_format: str) -> Policy:
    loaders = {
        "rl_games": load_rl_games_actor,
        "torchscript": load_torchscript_actor,
    }
    if policy_format != "auto":
        return loaders[policy_format](path, device)
    errors = []
    for name in ("torchscript", "rl_games"):
        try:
            return loaders[name](path, device)
        except (KeyError, RuntimeError, TypeError, ValueError) as exc:
            errors.append(f"{name}: {exc}")
    raise ValueError(
        "could not auto-detect checkpoint format; " + "; ".join(errors))


def actor_mu(state: dict[str, torch.Tensor], obs: torch.Tensor) -> torch.Tensor:
    mean = state["running_mean_std.running_mean"].float()
    var = state["running_mean_std.running_var"].float()
    x = ((obs.float() - mean) / torch.sqrt(var + 1.0e-5)).clamp(-5.0, 5.0)
    for index in (0, 2, 4):
        x = F.elu(F.linear(
            x,
            state[f"a2c_network.actor_mlp.{index}.weight"],
            state[f"a2c_network.actor_mlp.{index}.bias"],
        ))
    return F.linear(
        x,
        state["a2c_network.mu.weight"],
        state["a2c_network.mu.bias"],
    ).clamp(-1.0, 1.0)


def policy_action(policy: Policy, obs: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        if policy.policy_format == "torchscript":
            action = policy.actor(obs.float())
        else:
            action = actor_mu(policy.actor, obs)
    if not torch.is_tensor(action) or action.ndim != 2:
        raise ValueError("policy must return a rank-2 action tensor")
    return action


def set_default_pose(world, obs_builder: G.Go2ObsBuilder) -> None:
    q = torch.from_dlpack(world.buffer_view(nuka.JOINT_POSITION))
    qd = torch.from_dlpack(world.buffer_view(nuka.JOINT_VELOCITY))
    q[:, 1:G.GO2_BLC] = obs_builder.default_angles[obs_builder.nuka_slot_for_urdf]
    qd[:, 1:G.GO2_BLC] = 0.0
    nuka.sync()


def set_hold_gains(world, obs_builder: G.Go2ObsBuilder, kp: float, kd: float) -> None:
    kp_view = torch.from_dlpack(world.buffer_view(nuka.DRIVE_STIFFNESS))
    kd_view = torch.from_dlpack(world.buffer_view(nuka.DRIVE_DAMPING))
    fl_view = torch.from_dlpack(world.buffer_view(nuka.DRIVE_FORCE_LIMIT))
    kp_view[:, 1:G.GO2_BLC] = kp
    kd_view[:, 1:G.GO2_BLC] = kd
    fl_view[:, 1:G.GO2_BLC] = obs_builder.force_limit_urdf[
        obs_builder.nuka_slot_for_urdf]
    target = torch.from_dlpack(world.buffer_view(nuka.DRIVE_TARGET))
    target[:, 1:G.GO2_BLC] = obs_builder.default_angles[
        obs_builder.nuka_slot_for_urdf]
    nuka.sync()


def privileged_obs(proprio: torch.Tensor, world, profile: CorridorHeightProfile,
                   offsets: torch.Tensor) -> torch.Tensor:
    pose = torch.from_dlpack(world.buffer_view(nuka.ARTICULATION_LINK_POSE))
    base_x = pose[:, 0, 0:1]
    base_z = pose[:, 0, 2:3]
    h_base = profile.heights_at(base_x)
    h_offsets = profile.heights_at(base_x + offsets)
    priv = torch.cat((base_z - h_base, h_offsets - h_base), dim=1)
    return torch.cat((proprio, priv.clamp(-PRIV_HEIGHT_CLAMP, PRIV_HEIGHT_CLAMP)), dim=1)


def build_world(device, scene: str, physics_dt: float):
    builder = nuka.SceneBuilder.create(scene)
    try:
        world = builder.build(
            device, env_count=1, dt=physics_dt, control_mode=0,
            bake_link_sdf=True,
        )
    finally:
        builder.destroy()
    return world


def particle_snapshot(world) -> np.ndarray:
    try:
        return np.asarray(
            world.download_field(nuka.Field.PARTICLE_POSITION),
            dtype=np.float32,
        ).reshape(-1, 3).copy()
    except Exception:
        return np.empty((0, 3), dtype=np.float32)


def run(args: argparse.Namespace) -> dict:
    os.makedirs(args.outdir, exist_ok=True)
    old_frame = next(glob.iglob(os.path.join(args.outdir, "f_*.png")), None)
    if old_frame is not None:
        raise FileExistsError(
            f"output directory contains an existing frame: {old_frame}")
    dev_t = torch.device("cuda")
    policy = load_policy(args.checkpoint, dev_t, args.policy_format)
    obs_dim = policy.obs_dim
    if obs_dim not in (48, 65):
        raise ValueError(f"unsupported actor observation dimension {obs_dim}")
    policy_info = {
        "policy_format": policy.policy_format,
        "policy_provenance": policy.provenance,
        "policy_action_postprocess": policy.action_postprocess,
        "physics_dt": policy.physics_dt,
        "decimation": policy.decimation,
        "control_period_s": policy.control_dt,
    }
    print(json.dumps(policy_info, sort_keys=True), flush=True)
    started = time.perf_counter()
    with nuka.Device.create(0) as device:
        world = build_world(device, args.scene, policy.physics_dt)
        try:
            obs_builder = G.Go2ObsBuilder(device, world)
            obs_builder.apply_pd_gains()
            set_default_pose(world, obs_builder)
            profile = (CorridorHeightProfile(args.scene, dev_t)
                       if obs_dim == 65 else None)
            offsets = torch.as_tensor(
                TEACHER_PROFILE_OFFSETS, dtype=torch.float32, device=dev_t)
            command = torch.zeros(1, 3, device=dev_t)
            last_action = torch.zeros(1, G.GO2_ACTION_DIM, device=dev_t)

            # Seat the media with a physical high-gain hold, then return to policy gains.
            set_hold_gains(world, obs_builder, 60.0, 4.0)
            settle_steps = int(round(args.settle_seconds / policy.physics_dt))
            for _ in range(settle_steps):
                world.step()
            obs_builder.apply_pd_gains()
            nuka.sync()

            base0 = obs_builder.base_pos()[0].detach().clone()
            start_x = float(base0[0])
            frames = 0
            ctrl_steps = 0
            next_frame_t = 0.0
            sim_t = 0.0
            reached = {"stairs": False, "gravel": False,
                       "debris": False, "slab": False}
            trace = []
            finite = True
            fell = False
            stopped = False
            stop_step = None
            tail_steps = int(round(args.tail_seconds / policy.control_dt))
            total_steps = int(round(args.seconds / policy.control_dt)) + tail_steps
            for step in range(total_steps):
                if not stopped:
                    ramp = min(1.0, (ctrl_steps + 1) / max(1, args.command_ramp))
                    command[:, 0] = args.command_vx * ramp
                else:
                    command.zero_()
                proprio = obs_builder.compute_obs(command, last_action)
                policy_obs = (privileged_obs(proprio, world, profile, offsets)
                              if obs_dim == 65 else proprio)
                action = policy_action(policy, policy_obs)
                last_action = obs_builder.write_action(action)
                world.step_n(policy.decimation)
                nuka.sync()
                ctrl_steps += 1
                sim_t += policy.control_dt

                bp = obs_builder.base_pos()[0]
                tilt = float(obs_builder.tilt_deg()[0])
                base_vel = obs_builder.base_lin_vel()[0]
                x = float(bp[0])
                z = float(bp[2])
                finite &= bool(
                    torch.isfinite(bp).all() and
                    torch.isfinite(base_vel).all() and
                    torch.isfinite(action).all() and
                    np.isfinite(tilt))
                fell |= (not finite) or z < args.fall_height or tilt > args.fall_tilt
                reached["stairs"] |= x >= 0.45
                reached["gravel"] |= x >= 2.25
                reached["debris"] |= x >= 3.05
                reached["slab"] |= x >= 3.65
                trace.append({"t": sim_t, "x": x, "y": float(bp[1]),
                              "z": z, "tilt_deg": tilt,
                              "vx": float(base_vel[0])})
                if not stopped and x >= args.x_end:
                    stopped = True
                    stop_step = step
                if not stopped and fell:
                    break

                if not args.probe and sim_t + 1.0e-8 >= next_frame_t:
                    eye, look, fov = camera_at(x)
                    image = world.render_beauty(
                        eye=tuple(float(v) for v in eye),
                        look=tuple(float(v) for v in look),
                        fov_deg=float(fov), width=args.width, height=args.height,
                        spp=args.spp,
                    )
                    Image.fromarray(np.ascontiguousarray(image)).save(
                        os.path.join(args.outdir, f"f_{frames:05d}.png"))
                    frames += 1
                    next_frame_t += 1.0 / args.fps
                if stopped and stop_step is not None and step - stop_step >= tail_steps:
                    break

            particles = particle_snapshot(world)
            particle_finite = bool(
                np.isfinite(particles).all()) if particles.size else True
            finite &= particle_finite
            trace_path = os.path.join(args.outdir, "trajectory.json")
            max_x = max((row["x"] for row in trace), default=start_x)
            result = {
                "scene": args.scene,
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": policy.epoch,
                "checkpoint_reward": policy.reward,
                **policy_info,
                "obs_dim": obs_dim,
                "settle_seconds": args.settle_seconds,
                "settle_physics_steps": settle_steps,
                "start_x": start_x,
                "end_x": float(trace[-1]["x"]) if trace else start_x,
                "max_x": max_x,
                "target_x": args.x_end,
                "min_z": min((row["z"] for row in trace), default=float(base0[2])),
                "max_tilt_deg": max((row["tilt_deg"] for row in trace), default=0.0),
                "mean_vx": (float(np.mean([row["vx"] for row in trace]))
                            if trace else 0.0),
                "control_steps": len(trace),
                "frames": frames,
                "particle_count": int(particles.shape[0]),
                "particle_finite": particle_finite,
                "finite": finite,
                "fell": fell,
                "stages": reached,
                "elapsed_s": time.perf_counter() - started,
            }
            result["success"] = bool(
                finite and not fell and all(reached.values()) and
                max_x >= args.x_end)
            with open(trace_path, "w") as fh:
                json.dump({"summary": result, "trace": trace}, fh, indent=2)
            print(json.dumps(result, sort_keys=True), flush=True)
            return result
        finally:
            world.destroy()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a Go2 policy through the physical one-shot scene.")
    parser.add_argument(
        "--checkpoint", required=True,
        help="rl_games .pth checkpoint or external TorchScript motion.pt")
    parser.add_argument(
        "--policy-format", choices=("auto", "rl_games", "torchscript"),
        default="auto",
        help=("checkpoint format (default: auto); TorchScript uses physics "
              "dt=0.001/decimation=20, rl_games uses dt=0.005/decimation=4"))
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seconds", type=float, default=18.0)
    parser.add_argument("--tail-seconds", type=float, default=2.0)
    parser.add_argument("--command-vx", type=float, default=0.4)
    parser.add_argument("--command-ramp", type=int, default=30)
    parser.add_argument("--x-end", type=float, default=4.05)
    parser.add_argument(
        "--settle-seconds", type=float, default=0.8,
        help="physical high-gain settling duration in simulated seconds")
    parser.add_argument("--fall-height", type=float, default=0.18)
    parser.add_argument("--fall-tilt", type=float, default=65.0)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--spp", type=int, default=32)
    parser.add_argument("--probe", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0.0 or args.width <= 0 or args.height <= 0:
        parser.error("fps and image dimensions must be positive")
    if args.seconds <= 0.0 or args.tail_seconds < 0.0:
        parser.error("seconds must be positive and tail-seconds non-negative")
    if args.settle_seconds < 0.0:
        parser.error("settle-seconds must be non-negative")
    result = run(args)
    return 0 if result["success"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
