#!/usr/bin/env python3
"""Run the Go2 skill TorchScript policies through Nuka.

This is an inference-only harness: it keeps the 50 Hz observation/action
contracts, PD gains, timing, staged handoff logic, and GO2W trajectory export
used by the two included demonstration cases.
"""
from __future__ import annotations

import argparse
import math
import struct
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "python")
import nuka  # noqa: E402
from nuka.tasks import go2_obs as G  # noqa: E402

MAGIC = 0x474F3257
VERSION = 1
CONTROL_DT = 0.02
DECIMATION = 4
N_LINKS = 13

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


# The policy contract uses FR, FL, RR, RL. Nuka's policy-facing order is
# FL, FR, RL, RR. The same permutation is its own inverse here.
GENESIS_FROM_URDF = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], device="cuda")

DEFAULT_URDF = torch.tensor(
    [0.1, 0.8, -1.5, -0.1, 0.8, -1.5,
     0.1, 1.0, -1.5, -0.1, 1.0, -1.5], device="cuda")
BACKFLIP_DEFAULT_URDF = torch.tensor(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5,
     0.0, 1.0, -1.5, 0.0, 1.0, -1.5], device="cuda")
HANDSTAND_DESIRED_URDF = torch.tensor(
    [0.0, 0.8, -1.5, 0.0, 0.8, -1.5,
     0.0, 2.25, -1.75, 0.0, 2.25, -1.75], device="cuda")


def load_policy(path: Path):
    return torch.jit.load(str(path), map_location="cuda").eval()


def set_gains(env, kp: float, kd: float, tau: float = 33.5):
    drive_kp = torch.from_dlpack(env._world.buffer_view(nuka.DRIVE_STIFFNESS))
    drive_kd = torch.from_dlpack(env._world.buffer_view(nuka.DRIVE_DAMPING))
    force = torch.from_dlpack(env._world.buffer_view(nuka.DRIVE_FORCE_LIMIT))
    drive_kp[:, 1:N_LINKS] = kp
    drive_kd[:, 1:N_LINKS] = kd
    force[:, 1:N_LINKS] = tau


def write_target_with_obs(env, target_urdf: torch.Tensor):
    # The task already discovered the structural permutation. Keep all writes in
    # URDF order so the policy contract remains independent of cooked slot order.
    target_slots = target_urdf.index_select(0, env._obs.nuka_slot_for_urdf)
    env._obs._tgt[:, 1:N_LINKS] = target_slots


def step_direct(env, target_urdf: torch.Tensor):
    write_target_with_obs(env, target_urdf)
    env._world.step_n(DECIMATION)
    env.episode_step += 1
    nuka.sync()


def step_torque(env, torque_urdf: torch.Tensor):
    torque_slots = torque_urdf.index_select(1, env._obs.nuka_slot_for_urdf)
    tau_view = getattr(env, "_reference_tau_view", None)
    if tau_view is None:
        tau_view = torch.from_dlpack(env._world.buffer_view(nuka.TORQUE_INPUT))
        env._reference_tau_view = tau_view
    tau_view[:, 1:N_LINKS] = torque_slots
    env._world.step_n(DECIMATION)
    env.episode_step += 1
    nuka.sync()


def base_obs(env, last_action: torch.Tensor, *, zero_lin_vel: bool = False):
    b = env._obs
    q = b.q_urdf()
    qd = b.qd_urdf()
    obs = torch.cat((
        torch.zeros_like(b.base_lin_vel()) if zero_lin_vel else b.base_lin_vel() * 2.0,
        b.base_ang_vel() * 0.25,
        b.projected_gravity(),
        torch.zeros((env.num_envs, 3), device="cuda"),
        q - DEFAULT_URDF,
        qd * 0.05,
        last_action,
    ), dim=-1)
    return obs.clamp(-100.0, 100.0)


def write_go2w(path: Path, poses: list[np.ndarray], drives: list[np.ndarray]):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(poses)
    with path.open("wb") as f:
        f.write(struct.pack("<8i2f", MAGIC, VERSION, n, N_LINKS, N_LINKS, 7,
                            1, 50000, CONTROL_DT, CONTROL_DT))
        flat = np.concatenate((np.stack(drives), np.stack(poses).reshape(n, N_LINKS * 7)), axis=1)
        f.write(flat.astype("<f4").tobytes(order="C"))


def collect_frame(env, drives, poses):
    nuka.sync()
    drives.append(torch.from_dlpack(env._world.buffer_view(nuka.DRIVE_TARGET))[0, :N_LINKS].detach().cpu().numpy().copy())
    poses.append(torch.from_dlpack(env._world.buffer_view(nuka.ARTICULATION_LINK_POSE))[0, :N_LINKS].detach().cpu().numpy().copy())


def replay_backflip(args):
    from nuka.tasks.go2_backflip import make_env

    env = make_env(1, action_mode="pd", episode_length_s=args.seconds + 0.5)
    set_gains(env, 70.0, 3.0)
    env.reset()
    # Match the skill infer reset, not the RL fall-reset task.
    q = torch.from_dlpack(env._world.buffer_view(nuka.JOINT_POSITION))
    q[:, 1:N_LINKS] = BACKFLIP_DEFAULT_URDF.index_select(
        0, env._obs.urdf_from_nuka_slot).unsqueeze(0)
    qd = torch.from_dlpack(env._world.buffer_view(nuka.JOINT_VELOCITY))
    qd[:, 1:N_LINKS] = 0.0
    # Match the skill infer reset, not the RL fall-reset task.
    bp = torch.from_dlpack(env._world.buffer_view(nuka.BASE_POSE))
    bp[:, 2] = 0.35
    bp[:, 3] = 1.0
    bp[:, 4:7] = 0.0
    env._pose[:, 0, 2] = 0.35
    env._pose[:, 0, 3] = 1.0
    env._pose[:, 0, 4:7] = 0.0
    env._link_velocity.zero_()
    env.episode_step.zero_()
    nuka.sync()

    policy = load_policy(Path(args.backflip_model))
    default_gen = BACKFLIP_DEFAULT_URDF.index_select(0, GENESIS_FROM_URDF).unsqueeze(0)
    action = torch.zeros((1, 12), device="cuda")
    last_action = torch.zeros_like(action)
    poses: list[np.ndarray] = []
    drives: list[np.ndarray] = []
    switched = False
    stable = 0
    peak = 0.0
    rot_integral = 0.0
    for k in range(int(args.seconds / CONTROL_DT)):
        # Infer's obs uses the action pair before updating the policy. The motor
        # target below likewise executes last_action (one control-step latency).
        obs = base_obs(env, action, zero_lin_vel=True)
        q = env._obs.q_urdf().index_select(1, GENESIS_FROM_URDF)
        qd = env._obs.qd_urdf().index_select(1, GENESIS_FROM_URDF)
        flip_obs = torch.cat((
            env._obs.base_ang_vel() * 0.25,
            env._obs.projected_gravity(),
            q - default_gen,
            qd * 0.05,
            action,
            last_action,
        ), dim=-1)
        phase = math.pi * min(k, 150) / 150.0
        phase_t = torch.tensor([math.sin(phase), math.cos(phase),
                                math.sin(phase / 2), math.cos(phase / 2),
                                math.sin(phase / 4), math.cos(phase / 4)], device="cuda")
        flip_obs = torch.cat((flip_obs, phase_t.expand(1, -1)), dim=-1)
        rot_integral += float(env._obs.base_ang_vel()[0, 1]) * CONTROL_DT
        rot = -rot_integral * 180.0 / math.pi
        peak = max(peak, rot)
        exec_action = last_action
        if not switched:
            new_action = policy(flip_obs).clamp(-100.0, 100.0)
            if rot >= 300.0:
                switched = True
                new_action = torch.zeros_like(new_action)
        else:
            new_action = torch.zeros_like(action)
        target_gen = default_gen + 0.5 * exec_action
        target_urdf = target_gen.index_select(1, GENESIS_FROM_URDF).squeeze(0)
        step_direct(env, target_urdf)
        action = new_action
        last_action = new_action.clone()
        fz = env._feet_fz()[0]
        pg = env._obs.projected_gravity()[0]
        z = float(env._obs.base_pos()[0, 2])
        good = bool((fz > 5.0).all() and (pg[2] < -0.85) and
                    (pg[:2].norm() < 0.3) and (0.20 < z < 0.45) and
                    (env._wrench[0, 0, 2].abs() < 5.0))
        stable = stable + 1 if good else 0
        collect_frame(env, drives, poses)
        if stable >= 60:
            # Keep a short visible hold after the acceptance window.
            for _ in range(15):
                step_direct(env, BACKFLIP_DEFAULT_URDF)
                collect_frame(env, drives, poses)
            break
    write_go2w(Path(args.out), poses, drives)
    print(f"[go2-infer] backflip peak_deg={peak:.1f} switched={switched} "
          f"stable_s={stable * CONTROL_DT:.2f} frames={len(poses)} out={args.out}")
    env.close()


def replay_handstand(args):
    from nuka.tasks.go2_front_handstand import make_env

    env = make_env(1, episode_length_s=args.seconds + 0.5)
    set_gains(env, 40.0, 1.6 * 1.2)
    bp = torch.from_dlpack(env._world.buffer_view(nuka.BASE_POSE))
    bp[:, 2] = 0.32
    bp[:, 3] = 1.0
    bp[:, 4:7] = 0.0
    env._pose[:, 0, 2] = 0.32
    env._pose[:, 0, 3] = 1.0
    env._pose[:, 0, 4:7] = 0.0
    link_velocity = torch.from_dlpack(env._world.buffer_view(nuka.LINK_VELOCITY))
    link_velocity.zero_()
    q = torch.from_dlpack(env._world.buffer_view(nuka.JOINT_POSITION))
    q[:, 1:N_LINKS] = DEFAULT_URDF.index_select(
        0, env._obs.urdf_from_nuka_slot).unsqueeze(0)
    qd = torch.from_dlpack(env._world.buffer_view(nuka.JOINT_VELOCITY))
    qd[:, 1:N_LINKS] = 0.0
    link_velocity.zero_()
    env.episode_step.zero_()
    nuka.sync()
    policy = load_policy(Path(args.handstand_model))
    default = DEFAULT_URDF.unsqueeze(0)
    desired_delta = (HANDSTAND_DESIRED_URDF - DEFAULT_URDF).unsqueeze(0)
    action = torch.zeros((1, 12), device="cuda")
    filtered = torch.zeros_like(action)
    poses: list[np.ndarray] = []
    drives: list[np.ndarray] = []
    startup_steps = int(1.5 / CONTROL_DT)
    ramp_steps = int(0.75 / CONTROL_DT)
    filtered = torch.zeros_like(action)
    stand_warmup_done = False
    for k in range(int(args.seconds / CONTROL_DT)):
        # handstand.pt has the Isaac-Gym 48D contract: the first three values
        # are reserved/zero, followed by angular velocity and gravity.
        obs = base_obs(env, action, zero_lin_vel=True)
        if k >= startup_steps:
            policy_target = policy(obs).clamp(-100.0, 100.0) * 0.25
            stand_prog = ((-env._obs.projected_gravity()[:, 0] - 0.45) / 0.30).clamp(0.0, 1.0)
            alpha = 0.05 * (1.0 - stand_prog) + 0.30 * stand_prog
            filtered = alpha * filtered + (1.0 - alpha) * policy_target
        ramp = 1.0 if stand_warmup_done else max(0.0, min(1.0, (k - startup_steps + 1) / max(1, ramp_steps)))
        if ramp >= 1.0:
            stand_warmup_done = True
        if ramp <= 0.0:
            delta = torch.zeros_like(action)
        else:
            stand_prog = ((-env._obs.projected_gravity()[:, 0] - 0.45) / 0.30).clamp(0.0, 1.0)
            hold = 0.08 * ramp * stand_prog
            stabilized = (1.0 - hold) * filtered + hold * desired_delta
            delta = stabilized * ramp
        action = delta / 0.25
        target_urdf = (default + delta).squeeze(0)
        step_direct(env, target_urdf)
        collect_frame(env, drives, poses)
    pg = env._obs.projected_gravity()[0]
    z = float(env._obs.base_pos()[0, 2])
    fz = env._wrench[0, env._paw_slot, 2].abs()
    print(f"[go2-infer] handstand pg={pg.cpu().numpy().round(3)} z={z:.3f} "
          f"front_fz={fz[:2].cpu().numpy().round(1)} rear_fz={fz[2:].cpu().numpy().round(1)} "
          f"frames={len(poses)} out={args.out}")
    write_go2w(Path(args.out), poses, drives)
    env.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("backflip", "handstand"), required=True)
    ap.add_argument("--backflip-model", default=str(
        DEFAULT_MODEL_DIR / "nuka_go2_backflip.pt"))
    ap.add_argument("--handstand-model", default=str(
        DEFAULT_MODEL_DIR / "nuka_go2_front_handstand.pt"))
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.task == "backflip":
        replay_backflip(args)
    else:
        replay_handstand(args)


if __name__ == "__main__":
    main()
