"""Observation isolation, replay and noise configuration through the public Python API."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import nuka

SCENE = str(Path(__file__).resolve().parents[2] / "examples/scenes/go2_float.usda")
GO2_BLC = 13  # base_link_count: root + 12 actuated leg joints


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def device():
    dev = nuka.Device.create(0)
    yield dev
    dev.close()


def make_world(device, env_count=64):
    return nuka.World.create_from_scene(device, SCENE, env_count)


# ---------------------------------------------------------------------------
# Module-level noise-kind ints are exposed (mirror nuka_noise_kind_t).
# ---------------------------------------------------------------------------
def test_noise_kind_ints_exposed():
    assert nuka.NOISE_NONE == 0
    assert nuka.NOISE_GAUSSIAN == 1
    assert nuka.NOISE_POISSON == 2
    # The pure-python config module imports without torch/jax.
    assert nuka.GaussianNoise().kind == nuka.NOISE_GAUSSIAN
    assert nuka.PoissonNoise().kind == nuka.NOISE_POISSON


# ---------------------------------------------------------------------------
# Ideal acquisition preserves the physical source.
# ---------------------------------------------------------------------------
def test_sensor_noise_none_is_no_op(device):
    with make_world(device, 64) as w:
        w.step()
        nuka.sync()
        qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
        before = qd.detach().cpu().clone()

        # Unregistered acquisition copies the physical source.
        w.apply_sensor_noise(nuka.JOINT_VELOCITY)
        nuka.sync()
        after_unreg = qd.detach().cpu().clone()
        assert torch.equal(before, after_unreg), "unregistered apply changed qd"

        # Explicit NONE also preserves physics.
        w.set_sensor_noise(nuka.JOINT_VELOCITY, nuka.NOISE_NONE, 0.0, 0.0, 0)
        w.apply_sensor_noise(nuka.JOINT_VELOCITY)
        nuka.sync()
        after_none = qd.detach().cpu().clone()
        assert torch.equal(before, after_none), "NONE apply changed qd"


# ---------------------------------------------------------------------------
# Gaussian acquisition perturbs only the observation.
# ---------------------------------------------------------------------------
def test_sensor_noise_gaussian_changes_only_observation(device):
    with make_world(device, 64) as w:
        w.step()
        nuka.sync()
        qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
        before = qd.detach().cpu().clone()

        w.set_sensor_noise(nuka.JOINT_VELOCITY, nuka.NOISE_GAUSSIAN, 0.0, 0.02, 123)
        w.apply_sensor_noise(nuka.JOINT_VELOCITY)
        nuka.sync()
        after = torch.from_dlpack(w.get_observation_view(nuka.JOINT_VELOCITY)).cpu().clone()
        assert torch.equal(before, qd.cpu())

        assert not torch.equal(before, after), "Gaussian apply did not perturb qd"
        max_abs = (after - before).abs().max().item()
        assert max_abs > 0.0
        assert max_abs < 1.0, f"noise unbounded ({max_abs}); should be ~few stddev"


# ---------------------------------------------------------------------------
# Sensor noise D1: two FRESH worlds, SAME seed -> post-apply BIT-IDENTICAL.
# (The upstream qd is D1-deterministic; the noise is counter-based pure, so the
#  post-apply buffer is bit-identical -- torch.equal on the CPU copies.)
# ---------------------------------------------------------------------------
def _gaussian_qd_after_apply(device, seed):
    with make_world(device, 64) as w:
        w.step()
        nuka.sync()
        qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
        w.set_sensor_noise(nuka.JOINT_VELOCITY, nuka.NOISE_GAUSSIAN, 0.0, 0.02, seed)
        w.apply_sensor_noise(nuka.JOINT_VELOCITY)
        nuka.sync()
        return torch.from_dlpack(w.get_observation_view(nuka.JOINT_VELOCITY)).cpu().clone()


def test_sensor_noise_two_world_bit_exact(device):
    a = _gaussian_qd_after_apply(device, 123)
    b = _gaussian_qd_after_apply(device, 123)
    assert torch.equal(a, b), (
        f"two-world same-seed noise not bit-identical "
        f"(max|a-b|={(a - b).abs().max().item():.3e})"
    )


def test_sensor_noise_different_seed_differs(device):
    a = _gaussian_qd_after_apply(device, 123)
    c = _gaussian_qd_after_apply(device, 999)
    assert not torch.equal(a, c), "different seed produced identical noise"


# ---------------------------------------------------------------------------
# Sensor noise: successive applies advance the per-field sequence (independent
# noise across steps).
# ---------------------------------------------------------------------------
def test_sensor_noise_sequence_advances(device):
    with make_world(device, 64) as w:
        w.step()
        nuka.sync()
        qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
        w.set_sensor_noise(nuka.JOINT_VELOCITY, nuka.NOISE_GAUSSIAN, 0.0, 0.02, 777)

        s0 = qd.detach().cpu().clone()
        w.apply_sensor_noise(nuka.JOINT_VELOCITY)
        nuka.sync()
        s1 = torch.from_dlpack(w.get_observation_view(nuka.JOINT_VELOCITY)).cpu().clone()
        w.apply_sensor_noise(nuka.JOINT_VELOCITY)
        nuka.sync()
        s2 = torch.from_dlpack(w.get_observation_view(nuka.JOINT_VELOCITY)).cpu().clone()

        d1 = s1 - s0  # seq 0 noise
        d2 = s2 - s0  # seq 1 noise
        assert not torch.equal(d1, d2), (
            "successive applies did not advance the sequence (same increment)"
        )


# ---------------------------------------------------------------------------
# Sensor noise: the ergonomic GaussianNoise.apply_to helper matches the raw
# binding path bit-for-bit (and is itself deterministic two-world).
# ---------------------------------------------------------------------------
def _gaussian_qd_via_helper(device, seed):
    with make_world(device, 64) as w:
        w.step()
        nuka.sync()
        qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
        nuka.GaussianNoise(mean=0.0, stddev=0.02, seed=seed).apply_to(
            w, nuka.JOINT_VELOCITY
        )
        nuka.sync()
        return torch.from_dlpack(w.get_observation_view(nuka.JOINT_VELOCITY)).cpu().clone()


def test_gaussian_noise_helper_matches_raw_and_deterministic(device):
    raw = _gaussian_qd_after_apply(device, 555)
    helper_a = _gaussian_qd_via_helper(device, 555)
    helper_b = _gaussian_qd_via_helper(device, 555)
    assert torch.equal(raw, helper_a), "GaussianNoise.apply_to != raw binding path"
    assert torch.equal(helper_a, helper_b), "GaussianNoise.apply_to not deterministic"


def test_poisson_noise_helper_changes_buffer(device):
    with make_world(device, 64) as w:
        w.step()
        nuka.sync()
        qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
        before = qd.detach().cpu().clone()
        nuka.PoissonNoise(lam=1.0, seed=321).apply_to(w, nuka.JOINT_VELOCITY)
        nuka.sync()
        after = torch.from_dlpack(w.get_observation_view(nuka.JOINT_VELOCITY)).cpu().clone()
        assert torch.equal(before, qd.cpu())
        assert not torch.equal(before, after), "Poisson apply did not perturb qd"


# ---------------------------------------------------------------------------
# Sensor noise: a non-float-stride field (ARTICULATION_LINK_POSE, 7-float
# Transform) raises on apply (NUKA_RESULT_NOT_SUPPORTED).
# ---------------------------------------------------------------------------
def test_sensor_noise_non_float_stride_field_rejected(device):
    with make_world(device, 64) as w:
        w.step()
        nuka.sync()
        with pytest.raises(Exception):
            w.set_sensor_noise(
                nuka.ARTICULATION_LINK_POSE, nuka.NOISE_GAUSSIAN, 0.0, 0.01, 5
            )
        with pytest.raises(Exception):
            w.apply_sensor_noise(nuka.ARTICULATION_LINK_POSE)


@pytest.mark.parametrize("kind", [nuka.StateSensorKind.IMU, nuka.StateSensorKind.CONTACT_WRENCH,
                                 nuka.StateSensorKind.FORCE_TORQUE])
def test_mounted_sensor_timing_replay_and_late_attachment(device, kind):
    with make_world(device, 4) as world:
        world.set_execution_mode("graph")
        world.step_n(3)
        with world.capture_checkpoint() as unregistered:
            initial_hash = world.state_hash()
            sensor = world.attach_state_sensor(kind, mount=nuka.SensorMount.LINK, mount_index=2, update_period=2,
                latency=world.dt * 3, latency_jitter=world.dt * 0.5,
                dropout_probability=0.25, seed=36)
            nuka.MeasurementError(noise_density=0.002, bias_random_walk=0.001,
                correlated_bias_stddev=0.01, correlation_time=0.1, seed=71).configure_sensor(world, sensor, 0)
            view = torch.from_dlpack(world.get_state_sensor_view(sensor))
            address = view.data_ptr()
            assert world.state_sensor_stamp(sensor)["valid"] is False
            world.step_n(8)
            with world.capture_checkpoint() as saved:
                world.step_n(8)
                expected_values = world.download_state_sensor(sensor).copy()
                expected_stamps = [world.state_sensor_stamp(sensor, env) for env in range(4)]
                expected_hash = world.state_hash()
                world.restore_checkpoint(saved)
                world.step_n(8)
                np.testing.assert_array_equal(world.download_state_sensor(sensor), expected_values)
                assert [world.state_sensor_stamp(sensor, env) for env in range(4)] == expected_stamps
                assert world.state_hash() == expected_hash
                assert view.data_ptr() == address
                for stamp in expected_stamps:
                    assert stamp["acquisitions"] == 8
                    assert stamp["delivery_time"] - stamp["sample_time"] >= world.dt * 2.5
            world.reset_envs([1])
            assert world.state_sensor_stamp(sensor, 1)["acquisitions"] == 0
            assert not world.state_sensor_stamp(sensor, 1)["valid"]
            np.testing.assert_array_equal(world.download_state_sensor(sensor)[1], 0.0)
            assert world.state_sensor_stamp(sensor, 0) == expected_stamps[0]
            world.restore_checkpoint(unregistered)
            assert world.state_hash() == initial_hash
            assert world.state_sensor_count() == 1
            assert not world.state_sensor_active(sensor)
            np.testing.assert_array_equal(view.cpu(), 0.0)
            with pytest.raises(RuntimeError):
                world.get_state_sensor_view(sensor)


def test_imported_mounted_sensor_physics(device):
    scene = Path(__file__).resolve().parents[2] / "tests/data/rotated_slider.xml"
    with nuka.World.create_from_scene(device, str(scene), 2, dt=0.0001,
                                      control_mode=nuka.CONTROL_MODE_TORQUE) as world:
        world.set_gravity_z(0.0)
        world.set_execution_mode("graph")
        names = world.dof_names()
        hinge, slide = names.index("hinge"), names.index("slide")
        q = np.array(world.download_field(nuka.JOINT_POSITION)).reshape(2, -1)
        q[:, slide] = 0.4
        world.upload_field(nuka.JOINT_POSITION, q.ravel())
        torque = np.zeros_like(q)
        torque[:, hinge] = 1.0
        world.set_drive_targets(torque.ravel())
        world.step()
        velocity = np.asarray(world.download_field(nuka.JOINT_VELOCITY)).reshape(2, -1)
        acceleration = 1.0 / (0.1 + 0.05 + 2 * 0.4**2)
        np.testing.assert_allclose(velocity[:, hinge] / world.dt, acceleration, rtol=5e-5)
        np.testing.assert_allclose(velocity[:, slide] / world.dt, -0.3 * acceleration, rtol=5e-5)
        assert world.state_sensor_count() == 5
        encoder = world.download_state_sensor(0)
        np.testing.assert_array_equal(encoder[:, 1], velocity[:, slide])
        imu = world.download_state_sensor(1)
        expected = np.tile([0.03 * acceleration, 0.42 * acceleration, 0.0,
                            0.0, 0.0, 0.5 * acceleration * world.dt], (2, 1))
        np.testing.assert_allclose(imu, expected, atol=3e-5, rtol=5e-5)
        measured_velocity = world.download_state_sensor(2)
        np.testing.assert_allclose(measured_velocity, expected[:, :3] * world.dt, atol=2e-8)
        pose = world.download_state_sensor(3)
        np.testing.assert_allclose(pose[:, :3], np.tile([0.3, 0.4, 3.0], (2, 1)), atol=2e-7)
        np.testing.assert_allclose(np.linalg.norm(pose[:, 3:], axis=1), 1.0, atol=2e-7)
        wrench = world.download_state_sensor(4)
        expected_wrench = np.tile([0.0, 0.8 * acceleration, 0.0, 0.032 * acceleration,
                                  0.0, 0.034 * acceleration], (2, 1))
        np.testing.assert_allclose(wrench, expected_wrench, atol=4e-5, rtol=5e-5)
        support = world.attach_state_sensor(nuka.StateSensorKind.FORCE_TORQUE,
                                           mount=nuka.SensorMount.LINK, mount_index=hinge)
        world.reset_envs([0, 1])
        world.upload_field(nuka.JOINT_POSITION, q.ravel())
        world.set_drive_targets(torque.ravel())
        world.step()
        expected_support = np.tile([-0.8 * acceleration, 0, 0, 0, 0, 1], (2, 1))
        np.testing.assert_allclose(world.download_state_sensor(support), expected_support, atol=4e-5, rtol=5e-5)
        with pytest.raises(RuntimeError):
            world.set_state_sensor_error(3, 3, scale_error=0.01)


@pytest.mark.parametrize("floating", [False, True])
def test_mounted_loads_include_fixed_tool_and_live_mass(device, tmp_path, floating):
    source = Path(__file__).resolve().parents[2] / "tests/data/mounted_loads.xml"
    text = source.read_text()
    if floating:
        text = text.replace('<body name="base" pos="0 0 3">', '<body name="base" pos="0 0 3"><freejoint/>')
    path = tmp_path / "mounted_loads.xml"
    path.write_text(text)
    with nuka.World.create_from_scene(device, str(path), 2, dt=0.001,
                                     control_mode=nuka.CONTROL_MODE_TORQUE) as world:
        world.set_execution_mode("graph")
        for mass in (0.5, 1.5):
            world.set_link_mass(2, mass)
            world.step()
            wrist = np.tile([0, 0, (2 + mass) * 9.81, mass * 0.08 * 9.81,
                             -(2 * 0.17 + mass * 0.31) * 9.81, 0], (2, 1))
            tool = np.tile([0, 0, mass * 9.81, -mass * 0.04 * 9.81, mass * 0.04 * 9.81, 0], (2, 1))
            if floating:
                wrist[:] = tool[:] = 0
            np.testing.assert_allclose(world.download_state_sensor(0), wrist, atol=8e-5, rtol=2e-5)
            np.testing.assert_allclose(world.download_state_sensor(1), tool, atol=8e-5, rtol=2e-5)
        position = world.download_state_sensor(2)[:, :3]
        np.testing.assert_allclose(position[:, :2], np.tile([0.3, 0.1], (2, 1)), atol=2e-6)
        if floating:
            assert np.all(position[:, 2] < 3.0)
        else:
            np.testing.assert_allclose(position[:, 2], 3.0, atol=2e-6)


def test_mounted_sensors_reject_incompatible_tape_replay(device):
    with make_world(device, 1) as world:
        with pytest.raises(RuntimeError):
            world.attach_state_sensor(nuka.StateSensorKind.FORCE_TORQUE, mount=nuka.SensorMount.BASE)
        with nuka.Tape.create(world, checkpoint_interval=2, max_tape_entries=8,
                              max_checkpoints=8, recompute_on_backward=1) as tape:
            tape.step_with_tape()
            sensor = world.attach_state_sensor(nuka.StateSensorKind.IMU)
            assert world.state_sensor_active(sensor)
            before = world.state_hash()
            with pytest.raises(RuntimeError):
                tape.step_with_tape()
            with pytest.raises(RuntimeError):
                tape.backward(np.zeros(8 * tape.link_count, dtype=np.float32))
            with pytest.raises(RuntimeError):
                nuka.Tape.create(world)
            assert world.state_hash() == before


def test_measurement_error_and_checkpoint_before_registration(device):
    with make_world(device, 4) as world:
        field = nuka.JOINT_VELOCITY
        checkpoint = world.capture_checkpoint()
        initial_hash = world.state_hash()
        model = nuka.MeasurementError(bias=0.1, temperature_coefficient=0.02,
                                      quantization=0.05, minimum=-0.25, maximum=0.25)
        model.configure(world, field)
        world.sample_observation(field, 0.01, temperature=35.0)
        observed = world.download_observation(field)
        np.testing.assert_allclose(observed, 0.25)
        assert np.count_nonzero(world.download_field(field)) == 0
        stamp = world.observation_stamp(field)
        assert stamp == {"sequence": 1, "elapsed_time": 0.01, "valid": True}
        pointer = torch.from_dlpack(world.get_observation_view(field)).data_ptr()
        world.restore_checkpoint(checkpoint)
        assert world.state_hash() == initial_hash
        with pytest.raises(Exception):
            world.get_observation_view(field)
        world.apply_sensor_noise(field)
        assert torch.from_dlpack(world.get_observation_view(field)).data_ptr() == pointer
        np.testing.assert_array_equal(world.download_observation(field), 0.0)
        checkpoint.close()
        nuka.MeasurementError(noise_density=0.001, initial_bias_stddev=0.02,
                              bias_random_walk=0.01, correlated_bias_stddev=0.03,
                              correlation_time=0.2, response_time=0.1, seed=17).configure(world, field)
        world.sample_observation(field, 0.01)
        with world.capture_checkpoint() as saved:
            world.step()
            world.sample_observation(field, 0.02)
            expected = world.download_observation(field).copy()
            world.restore_checkpoint(saved)
            world.step()
            world.sample_observation(field, 0.02)
            np.testing.assert_array_equal(world.download_observation(field), expected)


# ---------------------------------------------------------------------------
# Domain randomization. The C-ABI exposes no mass/gravity getter, so (mirroring
# the C++ gate-4) we observe DR through its effect on a single-env contact-free
# diff-sim Tape backward: DR (set + apply) BEFORE Tape.create, then a rollout +
# backward. Two fresh worlds same seed -> identical grads; DR-on != DR-off.
# ---------------------------------------------------------------------------
N_STEPS = 16


def _run_tape_with_dr(device, seed, dr_enabled):
    """Single-env go2_float + optional DR + contact-free tape rollout + backward.
    Returns the concatenated [grad_actions | grad_parameters] host numpy array."""
    with make_world(device, 1) as w:
        if dr_enabled:
            # Wide ranges + fixed seed (mirrors C++ MakeDrDesc). mass + gravity
            # drive the contact-free tape; friction/armature/restitution are
            # sampled but inert there.
            nuka.DomainRandomization(
                mass_range=(0.8, 1.2),
                friction_range=(0.5, 1.5),
                restitution_range=(-0.1, 0.1),
                armature_range=(0.0, 0.05),
                gravity_range=(-0.5, 0.5),
                seed=seed,
                enabled=True,
            ).apply(w)  # set + apply, BEFORE Tape.create

        tape = nuka.Tape.create(
            w,
            checkpoint_interval=5,
            max_tape_entries=128,
            max_checkpoints=64,
            recompute_on_backward=1,
        )
        try:
            n = tape.link_count
            assert n > 0
            for _ in range(N_STEPS):
                tape.step_with_tape()
            assert tape.step_count == N_STEPS

            # seed: dL/dqdot' = 0.3, everything else 0 (layout 8n; mirrors C++).
            seed_vec = np.zeros(8 * n, dtype=np.float32)
            seed_vec[n:2 * n] = 0.3
            ga, gp = tape.backward(seed_vec)
            return np.concatenate([np.asarray(ga).ravel(), np.asarray(gp).ravel()])
        finally:
            tape.destroy()


def test_dr_backward_two_world_bit_exact(device):
    on1 = _run_tape_with_dr(device, seed=0x5EED, dr_enabled=True)
    on2 = _run_tape_with_dr(device, seed=0x5EED, dr_enabled=True)
    assert on1.shape == on2.shape
    assert np.array_equal(on1, on2), (
        "DR-on backward not bit-identical across two same-seed worlds (D1 broken)"
    )


def test_dr_applies_changes_gradient(device):
    on = _run_tape_with_dr(device, seed=0x5EED, dr_enabled=True)
    off = _run_tape_with_dr(device, seed=0x5EED, dr_enabled=False)
    assert on.shape == off.shape
    assert not np.array_equal(on, off), (
        "DR-on grad == DR-off grad: DR did not actually apply to the rollout"
    )
    max_diff = float(np.max(np.abs(on - off)))
    assert max_diff > 0.0


def test_dr_disabled_is_no_op(device):
    off = _run_tape_with_dr(device, seed=0x5EED, dr_enabled=False)
    # DR-off must itself be D1 byte-exact across two runs (sanity).
    off2 = _run_tape_with_dr(device, seed=0x5EED, dr_enabled=False)
    assert np.array_equal(off, off2), "DR-off backward not bit-identical two runs"

    # An explicitly DISABLED descriptor must equal the never-set baseline.
    with make_world(device, 1) as w:
        nuka.DomainRandomization(seed=0x5EED, enabled=False).apply(w)
        tape = nuka.Tape.create(w, checkpoint_interval=5, max_tape_entries=128,
                                max_checkpoints=64, recompute_on_backward=1)
        try:
            n = tape.link_count
            for _ in range(N_STEPS):
                tape.step_with_tape()
            seed_vec = np.zeros(8 * n, dtype=np.float32)
            seed_vec[n:2 * n] = 0.3
            ga, gp = tape.backward(seed_vec)
            disabled = np.concatenate(
                [np.asarray(ga).ravel(), np.asarray(gp).ravel()]
            )
        finally:
            tape.destroy()
    assert np.array_equal(disabled, off), (
        "explicitly-disabled DR grad != never-set baseline (disabled not a no-op)"
    )
