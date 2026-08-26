#!/usr/bin/env python3
"""Headless Go2 gate for checkpoint capture/restore, D1 replay, and state hash."""

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


def field_matrix(world: nuka.World, field: nuka.Field) -> np.ndarray:
    values = np.asarray(world.download_field(field), dtype=np.float32)
    return values.reshape(world.env_count, world.base_link_count)


def make_actions(env_count: int, link_count: int, steps: int) -> np.ndarray:
    rng = np.random.default_rng(20260812)
    actions = rng.uniform(-8.0, 8.0, size=(steps, env_count, link_count))
    actions[:, :, 0] = 0.0
    return actions.astype(np.float32)


def run_segment(world: nuka.World, actions: np.ndarray, start: int, count: int,
                force_limit: np.ndarray) -> float:
    upload = np.zeros((world.env_count, world.base_link_count), dtype=np.float32)
    nuka.sync()
    start_time = time.perf_counter()
    for i in range(start, start + count):
        np.multiply(actions[i], 1.0, out=upload)
        world.upload_field(nuka.TORQUE_INPUT, upload)
        world.step()
    nuka.sync()
    return time.perf_counter() - start_time


def capture_trace(world: nuka.World) -> dict[str, np.ndarray]:
    env_count = world.env_count
    return {
        "q": field_matrix(world, nuka.JOINT_POSITION).copy(),
        "qd": field_matrix(world, nuka.JOINT_VELOCITY).copy(),
        "limit": np.asarray(
            world.download_field(nuka.JOINT_LIMIT_IMPULSE), dtype=np.float32
        ).reshape(env_count, world.base_link_count, 2).copy(),
        "requested": field_matrix(world, nuka.ACTUATOR_EFFORT_REQUESTED).copy(),
        "applied": field_matrix(world, nuka.ACTUATOR_EFFORT).copy(),
        "saturated": field_matrix(world, nuka.ACTUATOR_SATURATED).copy(),
    }


def assert_finite(trace: dict[str, np.ndarray]) -> None:
    for name, values in trace.items():
        assert np.isfinite(values).all(), f"non-finite {name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-count", type=int, default=256)
    parser.add_argument("--lead", type=int, default=300)
    parser.add_argument("--tail", type=int, default=300)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--scene", type=Path, default=ROOT / "examples/scenes/go2_float.usda"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.env_count < 1 or args.lead < 1 or args.tail < 1:
        raise ValueError("--env-count/--lead/--tail must be positive")
    scene = args.scene.resolve()
    if not scene.is_file():
        raise FileNotFoundError(scene)

    with nuka.Device.create(args.device) as device:
        with nuka.World.create_from_scene(
            device,
            os.fspath(scene),
            env_count=args.env_count,
            dt=1.0 / 240.0,
            control_mode=nuka.CONTROL_MODE_TORQUE,
        ) as world:
            env_count = world.env_count
            link_count = world.base_link_count
            actions = make_actions(env_count, link_count, args.lead + args.tail)

            world.reset()
            force_limit = np.zeros((env_count, link_count), dtype=np.float32)
            force_limit[:, 1:] = 24.0
            world.upload_field(nuka.DRIVE_FORCE_LIMIT, force_limit)
            run_segment(world, actions, 0, args.lead, force_limit)
            hash_mid = world.state_hash()
            checkpoint = world.capture_checkpoint()
            hash_after_capture = world.state_hash()
            assert hash_after_capture == hash_mid, "capture perturbed state"

            # Advancing past the checkpoint must move the hash off the captured one.
            run_segment(world, actions, args.lead, args.tail, force_limit)
            trace_a = capture_trace(world)
            hash_a = world.state_hash()
            assert hash_a != hash_mid, "hash did not advance across steps"
            assert_finite(trace_a)

            # Restore replays the same tail bit-exactly, warm-start cache included.
            world.restore_checkpoint(checkpoint)
            hash_restored = world.state_hash()
            assert hash_restored == hash_mid, (
                f"restore hash mismatch: {hash_restored:#x} != {hash_mid:#x}"
            )
            elapsed = run_segment(
                world, actions, args.lead, args.tail, force_limit
            )
            trace_b = capture_trace(world)
            hash_b = world.state_hash()

            # A second restore from the SAME handle stays valid and reproducible.
            world.restore_checkpoint(checkpoint)
            assert world.state_hash() == hash_mid, "checkpoint reuse diverged"
            checkpoint.close()

    for name in trace_a:
        if not np.array_equal(trace_a[name], trace_b[name]):
            raise AssertionError(f"D1 replay mismatch in {name}")
    assert hash_b == hash_a, f"end hash mismatch: {hash_b:#x} != {hash_a:#x}"

    env_steps = env_count * args.tail
    print(f"scene={scene}")
    print(f"envs={env_count} lead={args.lead} tail={args.tail}")
    print(f"replay_throughput={env_steps / elapsed:.1f} env-steps/s")
    print(f"hash_mid={hash_mid:#018x} hash_end={hash_a:#018x}")
    print("capture=PASS restore_hash=PASS replay=PASS reuse=PASS")


if __name__ == "__main__":
    main()
