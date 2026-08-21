#!/usr/bin/env python3
"""Headless Go2 gate for authored limits, actuator telemetry, and D1 replay."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = ROOT / "python"
sys.path.insert(0, str(PYTHON_ROOT))

import nuka  # noqa: E402


EXPECTED_DOF_NAMES = [
    "base",
    "fl_hip_joint", "fl_thigh_joint", "fl_calf_joint",
    "fr_hip_joint", "fr_thigh_joint", "fr_calf_joint",
    "rl_hip_joint", "rl_thigh_joint", "rl_calf_joint",
    "rr_hip_joint", "rr_thigh_joint", "rr_calf_joint",
]
LOWER = np.array(
    [0.0] + [-1.0472, -1.5708, -2.7227] * 4, dtype=np.float32
)
UPPER = np.array(
    [0.0] + [1.0472, 3.4907, -0.83776] * 4, dtype=np.float32
)


def field_matrix(world: nuka.World, field: nuka.Field) -> np.ndarray:
    values = np.asarray(world.download_field(field), dtype=np.float32)
    return values.reshape(world.env_count, world.base_link_count)


def upload_matrix(world: nuka.World, field: nuka.Field, values: np.ndarray) -> None:
    world.upload_field(field, np.ascontiguousarray(values, dtype=np.float32))


def verify_effort_telemetry(world: nuka.World) -> None:
    q = field_matrix(world, nuka.JOINT_POSITION)
    env_count, link_count = q.shape
    delta = np.zeros_like(q)
    signs = np.where(
        (np.arange(env_count)[:, None] + np.arange(link_count)[None, :]) % 2 == 0,
        1.0,
        -1.0,
    ).astype(np.float32)
    split = env_count // 2
    delta[:split, 1:] = signs[:split, 1:]
    delta[split:, 1:] = 0.01 * signs[split:, 1:]

    stiffness = np.zeros_like(q)
    stiffness[:, 1:] = 100.0
    force_limit = np.zeros_like(q)
    force_limit[:, 1:] = 5.0
    feedforward = np.zeros_like(q)
    feedforward[:, 1:] = 7.0 * signs[:, 1:]

    upload_matrix(world, nuka.DRIVE_TARGET, q + delta)
    upload_matrix(world, nuka.DRIVE_STIFFNESS, stiffness)
    upload_matrix(world, nuka.DRIVE_DAMPING, np.zeros_like(q))
    upload_matrix(world, nuka.DRIVE_FORCE_LIMIT, force_limit)
    upload_matrix(world, nuka.JOINT_FEEDFORWARD, feedforward)
    world.step()

    requested = field_matrix(world, nuka.ACTUATOR_EFFORT_REQUESTED)
    applied = field_matrix(world, nuka.ACTUATOR_EFFORT)
    saturated = field_matrix(world, nuka.ACTUATOR_SATURATED)
    expected_requested = stiffness * delta
    expected_applied = np.clip(expected_requested, -force_limit, force_limit)
    expected_saturated = (expected_requested != expected_applied).astype(np.float32)

    np.testing.assert_allclose(
        requested, expected_requested, rtol=2.0e-6, atol=2.0e-4
    )
    np.testing.assert_allclose(applied, expected_applied, rtol=0.0, atol=2.0e-5)
    np.testing.assert_array_equal(saturated, expected_saturated)
    assert np.all(saturated[:split, 1:] == 1.0)
    assert np.all(saturated[split:, 1:] == 0.0)


def verify_torque_effort_telemetry(world: nuka.World) -> None:
    env_count = world.env_count
    command = np.zeros((env_count, world.base_link_count), dtype=np.float32)
    signs = np.where(
        (np.arange(env_count)[:, None] + np.arange(world.base_link_count)[None, :])
        % 2
        == 0,
        1.0,
        -1.0,
    ).astype(np.float32)
    command[:, 1:] = 100.0 * signs[:, 1:]
    force_limit = np.zeros_like(command)
    force_limit[:, 1:] = 6.0
    feedforward = np.zeros_like(command)
    feedforward[:, 1:] = 7.0 * signs[:, 1:]

    upload_matrix(world, nuka.TORQUE_INPUT, command)
    upload_matrix(world, nuka.DRIVE_FORCE_LIMIT, force_limit)
    upload_matrix(world, nuka.JOINT_FEEDFORWARD, feedforward)
    world.step()

    requested = field_matrix(world, nuka.ACTUATOR_EFFORT_REQUESTED)
    applied = field_matrix(world, nuka.ACTUATOR_EFFORT)
    saturated = field_matrix(world, nuka.ACTUATOR_SATURATED)
    expected_applied = np.clip(command, -force_limit, force_limit)
    expected_saturated = (command != expected_applied).astype(np.float32)
    np.testing.assert_array_equal(requested, command)
    np.testing.assert_array_equal(applied, expected_applied)
    np.testing.assert_array_equal(saturated, expected_saturated)


def run_limit_trace(world: nuka.World, steps: int) -> tuple[dict[str, np.ndarray], float]:
    env_count = world.env_count
    split = env_count // 2
    target = np.zeros((env_count, world.base_link_count), dtype=np.float32)
    target[:split, 1:] = LOWER[None, 1:] - 1.0
    target[split:, 1:] = UPPER[None, 1:] + 1.0
    stiffness = np.zeros_like(target)
    stiffness[:, 1:] = 100.0
    force_limit = np.zeros_like(target)
    force_limit[:, 1:] = 24.0

    upload_matrix(world, nuka.DRIVE_TARGET, target)
    upload_matrix(world, nuka.DRIVE_STIFFNESS, stiffness)
    upload_matrix(world, nuka.DRIVE_DAMPING, np.zeros_like(target))
    upload_matrix(world, nuka.DRIVE_FORCE_LIMIT, force_limit)
    upload_matrix(world, nuka.JOINT_FEEDFORWARD, np.zeros_like(target))

    nuka.sync()
    start = time.perf_counter()
    world.step_n(steps)
    nuka.sync()
    elapsed = time.perf_counter() - start
    trace = {
        "q": field_matrix(world, nuka.JOINT_POSITION).copy(),
        "qd": field_matrix(world, nuka.JOINT_VELOCITY).copy(),
        "limit": np.asarray(
            world.download_field(nuka.JOINT_LIMIT_IMPULSE), dtype=np.float32
        ).reshape(env_count, world.base_link_count, 2).copy(),
        "requested": field_matrix(world, nuka.ACTUATOR_EFFORT_REQUESTED).copy(),
        "applied": field_matrix(world, nuka.ACTUATOR_EFFORT).copy(),
        "saturated": field_matrix(world, nuka.ACTUATOR_SATURATED).copy(),
    }
    return trace, elapsed


def verify_limit_trace(trace: dict[str, np.ndarray]) -> None:
    q = trace["q"]
    impulse = trace["limit"]
    split = q.shape[0] // 2
    tolerance = 2.0e-4

    lower_gap = q[:split, 1:] - LOWER[None, 1:]
    upper_gap = UPPER[None, 1:] - q[split:, 1:]
    assert np.isfinite(q).all() and np.isfinite(impulse).all()
    assert float(lower_gap.min()) >= -tolerance
    assert float(upper_gap.min()) >= -tolerance
    assert float(np.abs(lower_gap).max()) <= tolerance
    assert float(np.abs(upper_gap).max()) <= tolerance
    assert np.all(impulse[:split, 1:, 0] > 0.0)
    assert np.all(impulse[split:, 1:, 1] > 0.0)
    assert np.count_nonzero(impulse[:split, :, 1]) == 0
    assert np.count_nonzero(impulse[split:, :, 0]) == 0
    assert np.count_nonzero(impulse[:, 0, :]) == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-count", type=int, default=256)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--scene", type=Path, default=ROOT / "examples/scenes/go2_float.usda"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.env_count < 4 or args.env_count % 2 != 0:
        raise ValueError("--env-count must be even and at least 4")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    scene = args.scene.resolve()
    if not scene.is_file():
        raise FileNotFoundError(scene)

    with nuka.Device.create(args.device) as device:
        with nuka.World.create_from_scene(
            device, os.fspath(scene), env_count=args.env_count, dt=1.0 / 240.0
        ) as world:
            assert world.dof_names() == EXPECTED_DOF_NAMES
            verify_effort_telemetry(world)
            world.reset()
            first, first_elapsed = run_limit_trace(world, args.steps)
            verify_limit_trace(first)
            world.reset()
            replay, replay_elapsed = run_limit_trace(world, args.steps)
            verify_limit_trace(replay)

        with nuka.World.create_from_scene(
            device,
            os.fspath(scene),
            env_count=args.env_count,
            dt=1.0 / 240.0,
            control_mode=nuka.CONTROL_MODE_TORQUE,
        ) as torque_world:
            assert torque_world.dof_names() == EXPECTED_DOF_NAMES
            verify_torque_effort_telemetry(torque_world)

    for name in first:
        if not np.array_equal(first[name], replay[name]):
            raise AssertionError(f"D1 replay mismatch in {name}")

    env_steps = args.env_count * args.steps
    split = args.env_count // 2
    print(f"scene={scene}")
    print(f"envs={args.env_count} steps={args.steps}")
    print(
        f"throughput={env_steps / first_elapsed:.1f} env-steps/s "
        f"replay={env_steps / replay_elapsed:.1f} env-steps/s"
    )
    print(
        "limit_impulse_max="
        f"lower:{first['limit'][:split, :, 0].max():.7g} "
        f"upper:{first['limit'][split:, :, 1].max():.7g}"
    )
    print("pd_effort=PASS torque_effort=PASS joint_limits=PASS d1_replay=PASS")


if __name__ == "__main__":
    main()
