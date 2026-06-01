"""pytest: nuka v0.5 sim-to-real noise wrapper (Task 5.4.9) through the binding.

Mirrors the C++ gtests (tests/c_abi/test_sensor_noise.cpp +
tests/sensor/test_n2_domain_randomization.cpp) but exercises the PYTHON binding:
``World.set_sensor_noise`` / ``apply_sensor_noise`` /
``set_domain_randomization`` / ``apply_domain_randomization`` and the ergonomic
``nuka.GaussianNoise`` / ``nuka.PoissonNoise`` / ``nuka.DomainRandomization``
config dataclasses.

Run (single GPU only):
    export CUDA_VISIBLE_DEVICES=0
    python -m pytest python/tests/test_noise.py -v

Gates:
  Sensor noise (64-env go2_float):
    * NONE / no apply -> JOINT_VELOCITY byte-unchanged.
    * GAUSSIAN apply CHANGES the buffer.
    * D1: two FRESH worlds, SAME seed -> post-apply buffer is BIT-IDENTICAL.
    * sequence advances: two successive applies give different increments.
    * non-float-stride field (ARTICULATION_LINK_POSE) -> apply raises.
  Domain randomization (single-env go2_float + Tape + backward, mirrors C++
  gate-4 since the C-ABI exposes no mass/gravity getter):
    * two FRESH worlds, SAME seed, DR ON -> backward grads BIT-IDENTICAL.
    * DR-on grads != DR-off grads (proves DR actually applied to the rollout).
    * disabled (enabled=False) -> grads identical to the never-set baseline.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import nuka

SCENE = "/root/Nuka-Physics/examples/scenes/go2_float.usda"
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
# Sensor noise: NONE / no apply is a byte no-op.
# ---------------------------------------------------------------------------
def test_sensor_noise_none_is_no_op(device):
    with make_world(device, 64) as w:
        w.step()
        nuka.sync()
        qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
        before = qd.detach().cpu().clone()

        # No noise registered -> apply is a byte no-op.
        w.apply_sensor_noise(nuka.JOINT_VELOCITY)
        nuka.sync()
        after_unreg = qd.detach().cpu().clone()
        assert torch.equal(before, after_unreg), "unregistered apply changed qd"

        # Explicit NONE kind -> still a no-op.
        w.set_sensor_noise(nuka.JOINT_VELOCITY, nuka.NOISE_NONE, 0.0, 0.0, 0)
        w.apply_sensor_noise(nuka.JOINT_VELOCITY)
        nuka.sync()
        after_none = qd.detach().cpu().clone()
        assert torch.equal(before, after_none), "NONE apply changed qd"


# ---------------------------------------------------------------------------
# Sensor noise: GAUSSIAN apply CHANGES the buffer (bounded perturbation).
# ---------------------------------------------------------------------------
def test_sensor_noise_gaussian_changes_buffer(device):
    with make_world(device, 64) as w:
        w.step()
        nuka.sync()
        qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
        before = qd.detach().cpu().clone()

        w.set_sensor_noise(nuka.JOINT_VELOCITY, nuka.NOISE_GAUSSIAN, 0.0, 0.02, 123)
        w.apply_sensor_noise(nuka.JOINT_VELOCITY)
        nuka.sync()
        after = qd.detach().cpu().clone()

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
        return qd.detach().cpu().clone()


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
        s1 = qd.detach().cpu().clone()
        w.apply_sensor_noise(nuka.JOINT_VELOCITY)
        nuka.sync()
        s2 = qd.detach().cpu().clone()

        d1 = s1 - s0  # seq 0 noise
        d2 = s2 - s1  # seq 1 noise
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
        return qd.detach().cpu().clone()


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
        after = qd.detach().cpu().clone()
        assert not torch.equal(before, after), "Poisson apply did not perturb qd"


# ---------------------------------------------------------------------------
# Sensor noise: a non-float-stride field (ARTICULATION_LINK_POSE, 7-float
# Transform) raises on apply (NUKA_RESULT_NOT_SUPPORTED).
# ---------------------------------------------------------------------------
def test_sensor_noise_non_float_stride_field_rejected(device):
    with make_world(device, 64) as w:
        w.step()
        nuka.sync()
        # set is OK (registers), apply raises NOT_SUPPORTED.
        w.set_sensor_noise(
            nuka.ARTICULATION_LINK_POSE, nuka.NOISE_GAUSSIAN, 0.0, 0.01, 5
        )
        with pytest.raises(Exception):
            w.apply_sensor_noise(nuka.ARTICULATION_LINK_POSE)


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
