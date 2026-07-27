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
from PIL import Image

import nuka
import bdx_oneshot_author as A
from nuka.rl_games.go2_policy import (
    actor_mu,
    load_contract,
    plain_model_state,
    postprocess_action,
)
from nuka.tasks import go2_obs as G
from nuka.tasks.bdx_corridor import (
    CorridorHeightProfile,
    PRIV_HEIGHT_CLAMP,
    TEACHER_PROFILE_OFFSETS,
)
from nuka.tasks.go2_corridor import slab_gate

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SCENE = os.path.join(REPO, "examples", "scenes", "go2_oneshot.nks")
RL_GAMES_DT = 0.005
RL_GAMES_DECIMATION = 4
TORCHSCRIPT_DT = 0.001
TORCHSCRIPT_DECIMATION = 20


def camera_at(x: float):
    """Clear Go2-scale side tracking shot.

    Center the whole quadruped instead of inheriting the BDX macro camera, while
    retaining enough lead room to see cloth, MPM deformation, and the slab before
    contact.  The open -y side of the widened workshop provides an unobstructed
    dolly lane.
    """
    slab_focus = max(0.0, min(1.0, (float(x) - 3.15) / 0.55))
    slab_focus = slab_focus * slab_focus * (3.0 - 2.0 * slab_focus)
    # Pull the dolly backwards and farther out as Go2 enters the slab portal.
    # Without this offset the near portal post crosses the optical axis and
    # hides the head/slab contact in an otherwise physically valid rollout.
    eye = np.array((float(x) - 0.12 - 0.65 * slab_focus,
                    -1.58 - 0.25 * slab_focus,
                    0.70 + 0.14 * slab_focus), dtype=np.float32)
    look = np.array((float(x) + 0.18, 0.0,
                     0.31 + 0.10 * slab_focus), dtype=np.float32)
    return eye, look, 50.0 + 4.0 * slab_focus


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
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in plain_model_state(checkpoint).items()
    }
    contract = load_contract(checkpoint)
    obs_dim = int(state["a2c_network.actor_mlp.0.weight"].shape[1])
    reward = float(checkpoint.get("last_mean_rewards", float("nan")))
    return Policy(
        actor=state,
        policy_format="rl_games",
        provenance=contract.provenance,
        obs_dim=obs_dim,
        epoch=int(checkpoint.get("epoch", -1)),
        reward=reward if np.isfinite(reward) else None,
        physics_dt=RL_GAMES_DT,
        decimation=RL_GAMES_DECIMATION,
        action_postprocess=contract.action_postprocess,
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


def policy_action(policy: Policy, obs: torch.Tensor, *, sample: bool = False,
                  generator: torch.Generator | None = None) -> torch.Tensor:
    with torch.inference_mode():
        if policy.policy_format == "torchscript":
            if sample:
                raise ValueError("action sampling requires an rl_games checkpoint")
            action = policy.actor(obs.float())
        else:
            contract = load_contract({
                "nuka_go2_policy": {
                    "action_postprocess": policy.action_postprocess,
                    "normalize_input": (
                        "running_mean_std.running_mean" in policy.actor),
                    "provenance": policy.provenance,
                }
            })
            action = actor_mu(
                policy.actor, obs,
                normalize_input=contract.normalize_input)
            if sample:
                logstd = policy.actor["a2c_network.sigma"].float()
                noise = torch.randn(
                    action.shape, dtype=action.dtype, device=action.device,
                    generator=generator)
                action = action + noise * logstd.exp().unsqueeze(0)
            action = postprocess_action(action, contract)
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
                   offsets: torch.Tensor, *, include_slab_gate: bool = False
                   ) -> torch.Tensor:
    pose = torch.from_dlpack(world.buffer_view(nuka.ARTICULATION_LINK_POSE))
    base_x = pose[:, 0, 0:1]
    base_z = pose[:, 0, 2:3]
    h_base = profile.heights_at(base_x)
    h_offsets = profile.heights_at(base_x + offsets)
    priv = torch.cat((base_z - h_base, h_offsets - h_base), dim=1)
    priv = priv.clamp(-PRIV_HEIGHT_CLAMP, PRIV_HEIGHT_CLAMP)
    if include_slab_gate:
        priv = torch.cat((priv, slab_gate(base_x)), dim=1)
    return torch.cat((proprio, priv), dim=1)


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
    recovery_policy = (
        load_policy(args.recovery_checkpoint, dev_t, args.recovery_policy_format)
        if args.recovery_checkpoint else None)
    if recovery_policy is not None and (
            recovery_policy.physics_dt != policy.physics_dt
            or recovery_policy.decimation != policy.decimation):
        raise ValueError(
            "primary and recovery policies must use the same physics timing")
    action_generator = None
    if args.sample_actions:
        if policy.policy_format != "rl_games":
            raise ValueError("--sample-actions requires an rl_games checkpoint")
        action_generator = torch.Generator(device=dev_t).manual_seed(args.seed)
    obs_dim = policy.obs_dim
    if obs_dim not in (48, 65, 66):
        raise ValueError(f"unsupported actor observation dimension {obs_dim}")
    policy_info = {
        "policy_format": policy.policy_format,
        "policy_provenance": policy.provenance,
        "policy_action_postprocess": policy.action_postprocess,
        "physics_dt": policy.physics_dt,
        "decimation": policy.decimation,
        "control_period_s": policy.control_dt,
        "action_mode": "sample" if args.sample_actions else "mean",
        "seed": args.seed,
        "recovery_checkpoint": args.recovery_checkpoint,
        "recovery_x": args.recovery_x if recovery_policy is not None else None,
        "recovery_blend_m": (
            args.recovery_blend_m if recovery_policy is not None else None),
    }
    print(json.dumps(policy_info, sort_keys=True), flush=True)
    started = time.perf_counter()
    with nuka.Device.create(0) as device:
        world = build_world(device, args.scene, policy.physics_dt)
        try:
            obs_builder = G.Go2ObsBuilder(device, world)
            obs_builder.apply_pd_gains()
            set_default_pose(world, obs_builder)
            needs_profile = (
                obs_dim in (65, 66)
                or (recovery_policy is not None
                    and recovery_policy.obs_dim in (65, 66)))
            profile = (CorridorHeightProfile(args.scene, dev_t)
                       if needs_profile else None)
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
            particles0 = particle_snapshot(world)
            cloth_count = A.CLOTH_NX * A.CLOTH_NY
            # Particle layout is MPM, then cloth, then cable beads + 8 slab
            # corners.  Derive the cable size from the authored segment count.
            cable_count = A.CABLE_SEGMENTS + 1 + 8
            cloth_end = particles0.shape[0] - cable_count
            cloth_start = cloth_end - cloth_count
            cloth0 = (particles0[cloth_start:cloth_end].copy()
                      if cloth_start >= 0
                      else np.empty((0, 3), dtype=np.float32))
            slab0 = (particles0[-8:].mean(axis=0) if cable_count >= 8
                     else np.zeros(3, dtype=np.float32))
            cloth_max_displacement = 0.0
            slab_max_displacement = 0.0
            slab_max_dx = 0.0
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
            recovery_activated = False
            recovery_max_alpha = 0.0
            pause_started_step = None
            pause_completed = False
            pause_steps = int(round(args.pause_seconds / policy.control_dt))
            tail_steps = int(round(args.tail_seconds / policy.control_dt))
            total_steps = int(round(args.seconds / policy.control_dt)) + tail_steps
            for step in range(total_steps):
                if not stopped:
                    x_before_command = float(obs_builder.base_pos()[0, 0])
                    if (args.pause_x is not None and pause_started_step is None
                            and x_before_command >= args.pause_x):
                        pause_started_step = ctrl_steps
                    pausing = (
                        pause_started_step is not None and not pause_completed
                        and ctrl_steps - pause_started_step < pause_steps)
                    if pausing:
                        command.zero_()
                    else:
                        if pause_started_step is not None:
                            pause_completed = True
                        ramp = min(
                            1.0, (ctrl_steps + 1) / max(1, args.command_ramp))
                        command[:, 0] = args.command_vx * ramp
                else:
                    command.zero_()
                proprio = obs_builder.compute_obs(command, last_action)
                policy_obs = (privileged_obs(
                    proprio, world, profile, offsets,
                    include_slab_gate=(obs_dim == 66))
                              if obs_dim in (65, 66) else proprio)
                action = policy_action(
                    policy, policy_obs, sample=args.sample_actions,
                    generator=action_generator)
                if recovery_policy is not None:
                    recovery_obs = (privileged_obs(
                        proprio, world, profile, offsets,
                        include_slab_gate=(recovery_policy.obs_dim == 66))
                                    if recovery_policy.obs_dim in (65, 66)
                                    else proprio)
                    recovery_action = policy_action(
                        recovery_policy, recovery_obs, sample=False)
                    x_before = float(obs_builder.base_pos()[0, 0])
                    blend = max(args.recovery_blend_m, 1.0e-6)
                    alpha = max(0.0, min(
                        1.0, (x_before - args.recovery_x) / blend))
                    alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                    if alpha > 0.0:
                        recovery_activated = True
                        recovery_max_alpha = max(recovery_max_alpha, alpha)
                        action = action * (1.0 - alpha) + recovery_action * alpha
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
                # Sparse live interaction probes: track peak cloth deflection and
                # slab swing, not merely their final (possibly returned) poses.
                if step % 10 == 0:
                    live_particles = particle_snapshot(world)
                    if (cloth0.size and
                            live_particles.shape[0] == particles0.shape[0]):
                        live_cloth = live_particles[cloth_start:cloth_end]
                        cloth_max_displacement = max(
                            cloth_max_displacement,
                            float(np.linalg.norm(
                                live_cloth - cloth0, axis=1).max()))
                    if live_particles.shape[0] >= 8:
                        slab_delta = live_particles[-8:].mean(axis=0) - slab0
                        slab_max_displacement = max(
                            slab_max_displacement,
                            float(np.linalg.norm(slab_delta)))
                        slab_max_dx = max(slab_max_dx, abs(float(slab_delta[0])))
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
            mpm_moved_count = 0
            mpm_lofted_count = 0
            mpm_max_displacement = 0.0
            mpm_center_mean_displacement = 0.0
            if (particles0.shape == particles.shape and
                    cloth_start > 0):
                p0_mpm = particles0[:cloth_start]
                p1_mpm = particles[:cloth_start]
                lane = ((p0_mpm[:, 0] >= A.TROUGH[0] - 0.02) &
                        (p0_mpm[:, 0] <= A.RECESS_X1 + 0.02) &
                        (np.abs(p0_mpm[:, 1]) <= A.GO2_LANE_HALF_Y + 0.05))
                delta = p1_mpm - p0_mpm
                displacement = np.linalg.norm(delta, axis=1)
                if np.any(lane):
                    lane_disp = displacement[lane]
                    mpm_moved_count = int(np.sum(lane_disp > 0.005))
                    mpm_lofted_count = int(np.sum(delta[lane, 2] > 0.010))
                    mpm_max_displacement = float(lane_disp.max())
                center = lane & (np.abs(p0_mpm[:, 1]) < 0.28)
                if np.any(center):
                    mpm_center_mean_displacement = float(
                        displacement[center].mean())
            trace_path = os.path.join(args.outdir, "trajectory.json")
            max_x = max((row["x"] for row in trace), default=start_x)
            result = {
                "scene": args.scene,
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": policy.epoch,
                "checkpoint_reward": policy.reward,
                **policy_info,
                "obs_dim": obs_dim,
                "recovery_obs_dim": (
                    recovery_policy.obs_dim if recovery_policy is not None else None),
                "recovery_activated": recovery_activated,
                "recovery_max_alpha": recovery_max_alpha,
                "pause_x": args.pause_x,
                "pause_seconds": args.pause_seconds,
                "pause_triggered": pause_started_step is not None,
                "pause_completed": pause_completed,
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
                "cloth_max_displacement": cloth_max_displacement,
                "mpm_moved_count_gt_5mm": mpm_moved_count,
                "mpm_lofted_count_gt_10mm": mpm_lofted_count,
                "mpm_max_displacement": mpm_max_displacement,
                "mpm_center_mean_displacement": mpm_center_mean_displacement,
                "slab_max_displacement": slab_max_displacement,
                "slab_max_dx": slab_max_dx,
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
    parser.add_argument("--recovery-checkpoint")
    parser.add_argument(
        "--recovery-policy-format",
        choices=("auto", "rl_games", "torchscript"), default="auto")
    parser.add_argument("--recovery-x", type=float, default=2.90)
    parser.add_argument("--recovery-blend-m", type=float, default=0.30)
    parser.add_argument("--pause-x", type=float)
    parser.add_argument("--pause-seconds", type=float, default=0.0)
    parser.add_argument("--scene", default=DEFAULT_SCENE)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seconds", type=float, default=18.0)
    parser.add_argument("--tail-seconds", type=float, default=2.0)
    parser.add_argument("--command-vx", type=float, default=0.4)
    parser.add_argument("--command-ramp", type=int, default=30)
    parser.add_argument("--sample-actions", action="store_true",
                        help="sample the rl_games Gaussian policy with a fixed seed")
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--x-end", type=float, default=4.10)
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
    if args.pause_seconds < 0.0:
        parser.error("pause-seconds must be non-negative")
    result = run(args)
    return 0 if result["success"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
