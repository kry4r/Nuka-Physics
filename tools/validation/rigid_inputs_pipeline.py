"""Validate public rigid inputs, environment reset and headless rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

import nuka


ROOT = Path(__file__).resolve().parents[2]
GRAVITY = np.array([0.6, -0.4, -9.81], dtype=np.float32)
FORCE = np.array([6.0, -4.0, 8.0], dtype=np.float32)
TORQUE = np.array([0.03, -0.02, 0.01], dtype=np.float32)
POSITION = np.array([0.75, 0.35, 1.0], dtype=np.float32)
MASS = 2.0
RADIUS = 0.075
STEPS = 36


def read(world, field, components):
    return world.download_field(field).reshape(world.env_count, -1, components).copy()


def state(world):
    fields = (
        (nuka.RIGID_BODY_TRANSFORM, 7),
        (nuka.Field.BODY_LINEAR_VELOCITY, 3),
        (nuka.Field.BODY_ANGULAR_VELOCITY, 3),
        (nuka.ARTICULATION_LINK_POSE, 7),
        (nuka.JOINT_POSITION, 1),
        (nuka.JOINT_VELOCITY, 1),
        (nuka.LINK_VELOCITY, 6),
        (nuka.Field.PARTICLE_POSITION, 3),
        (nuka.Field.PARTICLE_VELOCITY, 3),
    )
    result = [read(world, field, width) for field, width in fields]
    for value in result:
        assert np.isfinite(value).all()
    return result


def render(world):
    return world.render_beauty(
        eye=[1.8, -2.2, 1.7], look=[0.15, 0.0, 0.45],
        width=640, height=480, spp=16,
    ).copy()


def run_pipeline(device, output, dt=1.0 / 240.0, steps=STEPS):
    output.mkdir(parents=True, exist_ok=True)
    builder = nuka.SceneBuilder.create(str(ROOT / "examples/scenes/go2_stand.usda"))
    try:
        builder.add_material("load_body", base_color=[0.9, 0.25, 0.06])
        builder.add_rigid_primitive(
            nuka.PRIMITIVE_SPHERE, dims=[RADIUS], pos=POSITION.tolist(),
            mass=MASS, material="load_body",
        )
        builder.add_media(
            nuka.MEDIA_CLOTH, nuka.MEDIA_METHOD_XPBD,
            cloth_nx=9, cloth_ny=9, cloth_spacing=0.025,
            cloth_origin=[0.1, -0.1, 0.07], xpbd_iters=24,
            xpbd_aero_normal=0.6, xpbd_aero_tangent=0.04, xpbd_aero_max_dv=0.5,
        )
        builder.add_media(
            nuka.MEDIA_FLUID, nuka.MEDIA_METHOD_PBF,
            fluid_min=[-0.3, -0.2, 0.025], fluid_max=[-0.2, -0.1, 0.1],
            fluid_spacing=0.025, pbf_floor_z=0.0, pbf_iters=4,
        )
        world = builder.build(
            device, env_count=2, dt=dt, gravity_x=float(GRAVITY[0]),
            gravity_y=float(GRAVITY[1]), gravity_z=float(GRAVITY[2]),
            solver_vel_iters=48,
        )
    finally:
        builder.destroy()

    assert world.__enter__() is world
    with world:
        initial = state(world)
        candidates = np.flatnonzero(np.all(np.isclose(initial[0][0, :, :3], POSITION), axis=1))
        assert candidates.size == 1, candidates
        body = int(candidates[0])
        force_view = torch.from_dlpack(world.buffer_view(nuka.BODY_FORCE)).reshape(2, -1, 3)
        torque_view = torch.from_dlpack(world.buffer_view(nuka.BODY_TORQUE)).reshape(2, -1, 3)
        assert force_view.data_ptr() == world.buffer_device_ptr(nuka.BODY_FORCE)
        assert torque_view.data_ptr() == world.buffer_device_ptr(nuka.BODY_TORQUE)
        frames = [render(world)]
        forces = np.zeros(force_view.shape, dtype=np.float32)
        torques = np.zeros(torque_view.shape, dtype=np.float32)
        dt = world.dt
        load_scale = float(np.float32(1.0 / 240.0)) / dt
        forces[0, body] = FORCE * load_scale
        torques[0, body] = TORQUE * load_scale
        force_view.copy_(torch.from_numpy(forces))
        torch.cuda.synchronize()
        world.upload_field(nuka.BODY_TORQUE, torques)
        world.step()
        velocity = read(world, nuka.Field.BODY_LINEAR_VELOCITY, 3)[:, body]
        expected_velocity = GRAVITY[None, :] * dt + forces[:, body] * (dt / MASS)
        np.testing.assert_allclose(velocity, expected_velocity, rtol=0, atol=1e-6)
        omega = read(world, nuka.Field.BODY_ANGULAR_VELOCITY, 3)[:, body]
        expected_omega = torques[:, body] * (dt / (0.4 * MASS * RADIUS**2))
        np.testing.assert_allclose(omega, expected_omega, rtol=0, atol=1e-6)
        assert not np.any(world.download_field(nuka.BODY_FORCE))
        assert not np.any(world.download_field(nuka.BODY_TORQUE))

        world.step_n(steps - 1)
        advanced = state(world)
        gyro_residual = read(world, nuka.BODY_GYRO_RESIDUAL, 1)
        gyro_iterations = read(world, nuka.BODY_GYRO_ITERATIONS, 1)
        gyro_status = read(world, nuka.BODY_GYRO_STATUS, 1)
        env_status = world.download_field(nuka.ENV_STATUS).copy()
        assert not np.any(gyro_status)
        assert not np.any(env_status), env_status
        neighbor_attempted = read(world, nuka.PARTICLE_NEIGHBOR_ATTEMPTED, 1)
        neighbor_count = read(world, nuka.PARTICLE_NEIGHBOR_COUNT, 1)
        np.testing.assert_array_equal(neighbor_count, neighbor_attempted)
        assert float(gyro_residual.max()) <= 1e-6
        gyro_view = torch.from_dlpack(world.buffer_view(nuka.BODY_GYRO_STATUS))
        assert gyro_view.dtype == torch.uint32
        assert gyro_view.data_ptr() == world.buffer_device_ptr(nuka.BODY_GYRO_STATUS)
        expected_position = (
            POSITION + GRAVITY * (dt**2 * steps * (steps + 1) / 2)
            + forces[:, body] * (dt**2 * steps / MASS)
        )
        position_error = float(np.max(np.abs(advanced[0][:, body, :3] - expected_position)))
        np.testing.assert_allclose(advanced[0][:, body, :3], expected_position, rtol=0, atol=3e-6)
        np.testing.assert_allclose(
            advanced[1][:, body], GRAVITY * (dt * steps) + forces[:, body] * (dt / MASS),
            rtol=0, atol=3e-6,
        )
        np.testing.assert_allclose(advanced[2][:, body], expected_omega, rtol=0, atol=1e-6)
        frames.append(render(world))
        assert np.any(frames[0] != frames[1])

        pending_force = np.repeat(forces[:1], 2, axis=0)
        pending_torque = np.repeat(torques[:1], 2, axis=0)
        world.upload_field(nuka.BODY_FORCE, pending_force)
        world.upload_field(nuka.BODY_TORQUE, pending_torque)
        world.reset_envs(np.array([0], dtype=np.uint32))
        reset_state = state(world)
        for field, before in ((nuka.BODY_GYRO_RESIDUAL, gyro_residual),
                              (nuka.BODY_GYRO_ITERATIONS, gyro_iterations),
                              (nuka.BODY_GYRO_STATUS, gyro_status)):
            restored = read(world, field, 1)
            assert not np.any(restored[0])
            np.testing.assert_array_equal(restored[1], before[1])
        for field, before in ((nuka.PARTICLE_NEIGHBOR_ATTEMPTED, neighbor_attempted),
                              (nuka.PARTICLE_NEIGHBOR_COUNT, neighbor_count)):
            restored = read(world, field, 1)
            assert not np.any(restored[0])
            np.testing.assert_array_equal(restored[1], before[1])
        for before, moved, restored in zip(initial, advanced, reset_state):
            np.testing.assert_array_equal(restored[0], before[0])
            np.testing.assert_array_equal(restored[1], moved[1])
        pending_force[0] = 0
        pending_torque[0] = 0
        np.testing.assert_array_equal(read(world, nuka.BODY_FORCE, 3), pending_force)
        np.testing.assert_array_equal(read(world, nuka.BODY_TORQUE, 3), pending_torque)
        assert force_view.data_ptr() == world.buffer_device_ptr(nuka.BODY_FORCE)
        assert torque_view.data_ptr() == world.buffer_device_ptr(nuka.BODY_TORQUE)
        frames.append(render(world))
        np.testing.assert_array_equal(frames[2], frames[0])

        world.reset()
        assert not np.any(world.download_field(nuka.BODY_FORCE))
        assert not np.any(world.download_field(nuka.BODY_TORQUE))
        world.upload_field(nuka.BODY_FORCE, forces)
        torque_view.copy_(torch.from_numpy(torques))
        torch.cuda.synchronize()
        world.step_n(steps)
        for expected, actual in zip(advanced, state(world)):
            np.testing.assert_array_equal(actual, expected)

        panel = Image.new("RGB", (640 * len(frames), 510), "white")
        draw = ImageDraw.Draw(panel)
        for i, (label, pixels) in enumerate(zip(("Created", "Stepped", "Reset"), frames)):
            Image.fromarray(pixels).save(output / f"{label.lower()}.png")
            panel.paste(Image.fromarray(pixels), (640 * i, 30))
            draw.text((640 * i + 12, 8), label, fill="black")
        panel.save(output / "render_roundtrip.png")
        duration = dt * steps
        continuous_position = (POSITION + 0.5 * GRAVITY * duration**2
                               + forces[:, body] * (dt * duration / MASS))
        continuous_error = float(np.max(np.abs(advanced[0][:, body, :3] - continuous_position)))
        return {
            "env_count": world.env_count, "body_count_per_env": int(force_view.shape[1]),
            "particles_per_env": int(initial[-1].shape[1]), "steps": steps, "dt": dt,
            "position_max_error_m": position_error, "dlpack_alias": True,
            "continuous_position_max_error_m": continuous_error,
            "loads_consumed_once": True, "masked_reset_isolated": True,
            "replay_bit_exact": True, "render_reset_bit_exact": True,
            "gyro_max_residual": float(gyro_residual.max()),
            "gyro_max_iterations": int(gyro_iterations.max()),
            "env_status": env_status.tolist(), "gyro_reset_isolated": True,
            "neighbor_max_count": int(neighbor_count.max()),
            "neighbor_count": int(neighbor_count.sum()), "neighbors_not_truncated": True,
        }


def check_file_gravity(device):
    snapshots = []
    for kwargs in ({}, {"gravity_z": -9.81}, {
        "gravity_x": float(GRAVITY[0]), "gravity_y": float(GRAVITY[1]),
        "gravity_z": float(GRAVITY[2]),
    }):
        with nuka.World.create_from_scene(
            device, str(ROOT / "examples/scenes/go2_float.usda"), env_count=2, **kwargs,
        ) as world:
            world.step()
            snapshots.append(read(world, nuka.BASE_POSE, 7))
            dt = world.dt
    np.testing.assert_array_equal(snapshots[0], snapshots[1])
    delta = snapshots[2][:, :, :3] - snapshots[0][:, :, :3]
    expected = np.broadcast_to(np.array([GRAVITY[0], GRAVITY[1], 0]) * dt**2, delta.shape)
    np.testing.assert_allclose(delta, expected, rtol=0, atol=1e-6)
    return {"defaults_unchanged": True, "translation_delta_m": delta.tolist()}


def check_creation_contracts(device):
    results = {}
    for addon in ("go2_float.usda", "h1_visual.nks"):
        with nuka.SceneBuilder.create(str(ROOT / "examples/scenes/go2_stand.usda")) as builder:
            with nuka.SceneBuilder.create(str(ROOT / "examples/scenes" / addon)) as other:
                builder.compose(other, pos=[2, 0, 0], attach_at="second_")
            try:
                world = builder.build(device, env_count=2)
            except RuntimeError as error:
                assert str(error).endswith("not supported (7)"), str(error)
                results[addon] = str(error)
            else:
                world.destroy()
                raise AssertionError("unsupported topology was accepted: " + addon)
    with nuka.SceneBuilder.create(str(ROOT / "examples/scenes/go2_stand.usda")) as builder:
        try:
            world = builder.build(device, dt=float("nan"))
        except RuntimeError as error:
            assert str(error).endswith("invalid argument (1)"), str(error)
            results["nonfinite_dt"] = str(error)
        else:
            world.destroy()
            raise AssertionError("nonfinite timestep was accepted")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contracts-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with nuka.Device.create(0) as device:
        if args.contracts_only:
            report = check_creation_contracts(device)
            (args.output / "public_creation_contracts.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
            return
        report = {"pipeline": run_pipeline(device, args.output),
                  "refined_pipeline": run_pipeline(device, args.output / "refined", 1.0 / 480.0, 2 * STEPS),
                  "file_gravity": check_file_gravity(device),
                  "creation_contracts": check_creation_contracts(device)}
    coarse = report["pipeline"]["continuous_position_max_error_m"]
    fine = report["refined_pipeline"]["continuous_position_max_error_m"]
    assert 1.9 < coarse / fine < 2.1, (coarse, fine)
    report["linear_refinement_error_ratio"] = coarse / fine
    (args.output / "public_api.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
