"""pytest: nuka nanobind binding -- DLPack zero-copy + engine behavior.

Run (single GPU only):
    export CUDA_VISIBLE_DEVICES=0
    python -m pytest python/tests -v

Proves (the p02 exit criteria):
  * import nuka + torch in one process with no ABI crash;
  * create a go2_float world (64 envs; 4096 smoke);
  * DLPack zero-copy: torch.from_dlpack(view).data_ptr() == engine device_ptr,
    for JOINT_POSITION / JOINT_VELOCITY / DRIVE_TARGET / ARTICULATION_LINK_POSE;
  * step() advances q; writing DRIVE_TARGET (DLPack in-place AND set_drive_targets)
    moves the driven joint toward the target, sign-correct, after a step;
  * determinism: two identical runs -> bit-identical q;
  * floating base pose is LIVE (moves) on go2_float.
"""

from __future__ import annotations

import os

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
# import / ABI coexistence
# ---------------------------------------------------------------------------
def test_import_coexist():
    # If the ABI clashed (engine CXX11=1 vs libtorch=0) this import pair would
    # have crashed at module load. Reaching here proves they coexist.
    assert torch.__version__.startswith("2.")
    assert isinstance(nuka.__engine_version__, str)
    # DLPack is the only bridge: torch must NOT be a dep of the ext.
    import nuka._nuka_ext as ext
    assert "torch" not in repr(ext)


# ---------------------------------------------------------------------------
# world creation + metadata
# ---------------------------------------------------------------------------
def test_world_metadata(device):
    with make_world(device, 64) as w:
        assert w.env_count == 64
        assert w.base_link_count == GO2_BLC
        assert w.dt > 0.0


# T1 (unified actuator): the Torque control mode is wired onto the ONE generic
# nk::World. control_mode=Torque routes to model.drive_mode==1, selecting the
# direct-torque preset of OpApplyDrives (tau = clamp(u, +/-force_limit)). The
# control input `u` is the TORQUE_INPUT field, which ALIASES DriveTarget (the same
# persistent device buffer, preset-reinterpreted -- "禁止特化": one ctrl buffer).
# This smoke proves: (a) the world constructs in Torque mode; (b) TORQUE_INPUT is a
# writable, correctly-shaped, zero-copy view that aliases the DRIVE_TARGET buffer;
# (c) the world steps finite; (d) a non-zero torque on one joint moves that joint's
# own velocity sign-correct (it actually reaches the solver as a torque).
def test_torque_mode_world_steps(device):
    with nuka.World.create_from_scene(
        device, SCENE, 64, control_mode=nuka.CONTROL_MODE_TORQUE
    ) as w:
        # The torque control buffer is writable + correctly shaped (env x links).
        view = torch.from_dlpack(w.buffer_view(nuka.TORQUE_INPUT))
        assert view.is_cuda
        assert view.numel() == w.env_count * w.base_link_count
        # The alias: TORQUE_INPUT is the SAME device buffer as DRIVE_TARGET.
        assert w.buffer_device_ptr(nuka.TORQUE_INPUT) == \
            w.buffer_device_ptr(nuka.DRIVE_TARGET), \
            "TORQUE_INPUT must alias the DRIVE_TARGET device buffer"
        view.zero_()  # writable in place.
        w.step()  # the torque preset must step without throwing.
        nuka.sync()
        q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
        assert torch.isfinite(q).all()


def test_torque_mode_drives_joint_sign_correct(device):
    # A positive torque on one actuated joint must increase that joint's own
    # velocity after a step; a negative torque must decrease it (the torque reached
    # the solver with the right sign). N=64; we read env0, the perturbed joint only.
    GO2_TORQUE = 5.0  # well inside the cooked force limit; enough to move qd in 1 dt.

    def _own_qd(slot, tau):
        with nuka.World.create_from_scene(
            device, SCENE, 64, control_mode=nuka.CONTROL_MODE_TORQUE
        ) as w:
            tq = torch.from_dlpack(w.buffer_view(nuka.TORQUE_INPUT))
            qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
            tq.zero_()
            if tau != 0.0:
                tq[0, slot] = tau
            w.step()
            nuka.sync()
            return float(qd[0, slot].item())

    bad = []
    for slot in range(1, GO2_BLC):  # actuated joints 1..12
        base = _own_qd(slot, 0.0)
        d_pos = _own_qd(slot, +GO2_TORQUE) - base
        d_neg = _own_qd(slot, -GO2_TORQUE) - base
        if not (d_pos > 0 and d_neg < 0):
            bad.append((slot, d_pos, d_neg))
    assert not bad, f"torque wrong-sign joints (slot, d+, d-): {bad}"


# NOT YET WIRED on the unified nk::World: the ComputedTorque / Actuator / Osc
# presets have no op kernel on nk::World yet (T1 lights PDPosition + Torque only).
# create_from_scene returns NOT_SUPPORTED for these, so their binding smokes stay
# skipped until those presets are wired.
@pytest.mark.skip(reason="ComputedTorque/Actuator presets not yet wired on the "
                         "unified nk::World (T1 wires PDPosition + Torque)")
@pytest.mark.parametrize(
    "mode, field",
    [
        (nuka.CONTROL_MODE_COMPUTED_TORQUE, nuka.DRIVE_TARGET),
        (nuka.CONTROL_MODE_ACTUATOR, nuka.ACTUATOR_NOLOAD_SPEED),
    ],
)
def test_slice2_control_mode_world_steps(device, mode, field):
    with nuka.World.create_from_scene(device, SCENE, 64, control_mode=mode) as w:
        # The mode-specific control buffer is a writable, correctly-shaped view.
        view = torch.from_dlpack(w.buffer_view(field))
        assert view.is_cuda
        assert view.numel() == w.env_count * w.base_link_count
        view.zero_()  # writable in place.
        w.step()  # the slice-2 stage-1 law must step without throwing.
        nuka.sync()
        q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
        assert torch.isfinite(q).all()


@pytest.mark.skip(reason="M10 named gap: Osc control mode deferred to M10 "
                         "(M9 unified-world wires PDPosition only)")
def test_osc_mode_world_steps(device):
    # Osc (4, p03 R2): operational-space control, forward position task. The
    # binding smoke -- the world constructs in Osc mode (with a task link) and
    # steps without throwing, and the per-ENV TASK_TARGET view is writable and
    # correctly shaped (env_count x 3, NOT per-link). osc_task_link picks a
    # non-root link so the task Jacobian is non-trivial.
    with nuka.World.create_from_scene(
        device, SCENE, 64, control_mode=nuka.CONTROL_MODE_OSC, osc_task_link=3
    ) as w:
        view = torch.from_dlpack(w.buffer_view(nuka.TASK_TARGET))
        assert view.is_cuda
        # Per-ENV float3 {x,y,z}: env_count * 3 floats.
        assert view.numel() == w.env_count * 3
        view.zero_()  # writable in place.
        w.step()  # the Osc stage-1 law must step without throwing.
        nuka.sync()
        q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
        assert torch.isfinite(q).all()


def test_out_of_range_control_mode_rejected(device):
    # All six modes {0..5} are implemented; a value above Actuator (5) is rejected.
    with pytest.raises(Exception):
        nuka.World.create_from_scene(device, SCENE, 64, control_mode=6)


# ---------------------------------------------------------------------------
# T3 unified-actuator feed-forward joint force (JOINT_FEEDFORWARD): a per-link
# direct force added to the actuator output of EVERY control mode (tau += joint_f).
# Default 0 => no-op (the drive is byte-identical to a world without it -- proven
# at the C++ level by the go2_stand PD trajectory memcmp). These smokes prove the
# field is a writable zero-copy view AND that a non-zero feed-forward actually
# reaches the solver and moves the joint sign-correct.
# ---------------------------------------------------------------------------
def test_joint_feedforward_zero_copy_and_shaped(device):
    with make_world(device, 64) as w:
        view = w.buffer_view(nuka.JOINT_FEEDFORWARD)
        ff = torch.from_dlpack(view)
        assert ff.is_cuda and ff.dtype == torch.float32
        assert tuple(ff.shape) == (64, GO2_BLC)
        # writable + zero-copy: torch aliases the engine device buffer.
        assert ff.data_ptr() == w.buffer_device_ptr(nuka.JOINT_FEEDFORWARD), \
            "DLPack made a copy!"
        ff.zero_()  # writable in place.
        w.step()
        nuka.sync()
        q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
        assert torch.isfinite(q).all()


def test_joint_feedforward_reaches_solver_sign_correct(device):
    # In PD mode, a positive feed-forward torque on one actuated joint must push
    # that joint's own velocity MORE positive than with zero feed-forward (and a
    # negative feed-forward more negative) after one step -- i.e. tau += joint_f
    # reached the solver additively, on top of the PD hold. N=64; env0, the
    # perturbed joint only.
    FF = 4.0  # well inside the cooked force limit.

    def _own_qd(slot, ff_val):
        with make_world(device, 64) as w:
            ff = torch.from_dlpack(w.buffer_view(nuka.JOINT_FEEDFORWARD))
            qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
            ff.zero_()
            if ff_val != 0.0:
                ff[0, slot] = ff_val
            w.step()
            nuka.sync()
            return float(qd[0, slot].item())

    bad = []
    for slot in range(1, GO2_BLC):  # actuated joints 1..12
        base = _own_qd(slot, 0.0)
        d_pos = _own_qd(slot, +FF) - base
        d_neg = _own_qd(slot, -FF) - base
        if not (d_pos > 0 and d_neg < 0):
            bad.append((slot, d_pos, d_neg))
    assert not bad, f"feed-forward wrong-sign joints (slot, d+, d-): {bad}"


def test_4096_smoke(device):
    with make_world(device, 4096) as w:
        w.step()
        nuka.sync()
        q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
        assert q.shape == (4096, GO2_BLC)
        assert torch.isfinite(q).all()


# ---------------------------------------------------------------------------
# DLPack zero-copy: device-ptr match (the headline exit criterion)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field,expect_shape",
    [
        (nuka.JOINT_POSITION, (64, GO2_BLC)),
        (nuka.JOINT_VELOCITY, (64, GO2_BLC)),
        (nuka.DRIVE_TARGET, (64, GO2_BLC)),
        (nuka.ARTICULATION_LINK_POSE, (64, GO2_BLC, 7)),
        # infer-enable #40: base velocity read + writable PD gains.
        (nuka.LINK_VELOCITY, (64, GO2_BLC, 6)),
        (nuka.DRIVE_STIFFNESS, (64, GO2_BLC)),
        (nuka.DRIVE_DAMPING, (64, GO2_BLC)),
        (nuka.DRIVE_FORCE_LIMIT, (64, GO2_BLC)),
        # p14a (v0.7) contact-force readout. LINK_CONTACT_WRENCH is per-GLOBAL-link
        # [F(3),tau(3)] -> (env, base_link, 6). CONTACT_NORMAL / CONTACT_FORCE are
        # per-slot (kMaxFootContactsPerEnv == 4) -> (env, 4, 3).
        (nuka.LINK_CONTACT_WRENCH, (64, GO2_BLC, 6)),
        (nuka.CONTACT_NORMAL, (64, 4, 3)),
        (nuka.CONTACT_FORCE, (64, 4, 3)),
    ],
)
def test_dlpack_zero_copy(device, field, expect_shape):
    with make_world(device, 64) as w:
        view = w.buffer_view(field)
        assert hasattr(view, "__dlpack__")
        assert hasattr(view, "__dlpack_device__")
        t = torch.from_dlpack(view)
        assert t.is_cuda
        assert t.dtype == torch.float32
        assert tuple(t.shape) == expect_shape
        # THE zero-copy proof: torch aliases the engine device buffer.
        assert t.data_ptr() == w.buffer_device_ptr(field), "DLPack made a copy!"


def test_dlpack_zero_copy_contact_link_uint32(device):
    # p14a: CONTACT_LINK is the ONE non-float32 field (per-slot owning GLOBAL link
    # index, uint32). torch>=2.6 imports a uint32 CUDA DLPack capsule zero-copy, so
    # the only difference from the float fields is the dtype assertion.
    with make_world(device, 64) as w:
        view = w.buffer_view(nuka.CONTACT_LINK)
        assert hasattr(view, "__dlpack__")
        assert hasattr(view, "__dlpack_device__")
        t = torch.from_dlpack(view)
        assert t.is_cuda
        assert t.dtype == torch.uint32
        assert tuple(t.shape) == (64, 4)  # (env, kMaxFootContactsPerEnv)
        assert t.data_ptr() == w.buffer_device_ptr(nuka.CONTACT_LINK), \
            "DLPack made a copy!"


# ---------------------------------------------------------------------------
# step advances q
# ---------------------------------------------------------------------------
def test_step_changes_q(device):
    with make_world(device, 64) as w:
        q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
        before = q.clone()
        w.step_n(10)
        nuka.sync()
        # zero-copy view reflects the new state without re-fetching.
        assert not torch.equal(before, q), "q did not change after stepping"
        assert torch.isfinite(q).all()


# ---------------------------------------------------------------------------
# drive write -> q moves toward target (sign-correct), N=1, single joint.
# Mirrors the ctypes harness L5: perturb ONE actuated joint's own target by
# +/-delta on a fresh world and confirm its OWN velocity moves the matching sign
# after one step (PD reached the solver; gravity/contact are common-mode and
# cancel at N=1). Done both via DLPack in-place write and set_drive_targets.
# ---------------------------------------------------------------------------
def _own_qd_after_step(device, slot, delta, use_dlpack):
    with make_world(device, 64) as w:
        n = w.env_count * w.base_link_count
        tgt = torch.from_dlpack(w.buffer_view(nuka.DRIVE_TARGET))
        qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
        if delta != 0.0:
            if use_dlpack:
                tgt[0, slot] += delta  # env0, this joint only -- in place
            else:
                flat = tgt.flatten().clone()
                flat[slot] += delta
                w.set_drive_targets(flat.contiguous())
        w.step()
        nuka.sync()
        return float(qd[0, slot].item())


@pytest.mark.parametrize("use_dlpack", [True, False], ids=["dlpack", "set_drive_targets"])
def test_drive_write_moves_joint_sign_correct(device, use_dlpack):
    delta = 0.2
    bad = []
    for slot in range(1, GO2_BLC):  # actuated joints 1..12
        base = _own_qd_after_step(device, slot, 0.0, use_dlpack)
        d_pos = _own_qd_after_step(device, slot, +delta, use_dlpack) - base
        d_neg = _own_qd_after_step(device, slot, -delta, use_dlpack) - base
        if not (d_pos > 0 and d_neg < 0):
            bad.append((slot, d_pos, d_neg))
    assert not bad, f"wrong-sign joints (slot, d+, d-): {bad}"


def _own_qd_after_step_numpy(device, slot, delta):
    """set_drive_targets with a HOST numpy float32 array (exercises the
    cudaMemcpyHostToDevice branch -- the path a CPU array hits)."""
    with make_world(device, 64) as w:
        n = w.env_count * w.base_link_count
        tgt = torch.from_dlpack(w.buffer_view(nuka.DRIVE_TARGET))
        qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
        flat = tgt.flatten().detach().cpu().numpy().astype(np.float32).copy()  # HOST
        if delta != 0.0:
            flat[slot] += delta
        w.set_drive_targets(flat)  # numpy CPU array -> H2D copy
        w.step()
        nuka.sync()
        return float(qd[0, slot].item())


def test_set_drive_targets_host_numpy_sign_correct(device):
    delta = 0.2
    bad = []
    for slot in range(1, GO2_BLC):
        base = _own_qd_after_step_numpy(device, slot, 0.0)
        d_pos = _own_qd_after_step_numpy(device, slot, +delta) - base
        d_neg = _own_qd_after_step_numpy(device, slot, -delta) - base
        if not (d_pos > 0 and d_neg < 0):
            bad.append((slot, d_pos, d_neg))
    assert not bad, f"host-path wrong-sign joints (slot, d+, d-): {bad}"


# ---------------------------------------------------------------------------
# determinism: two identical runs -> bit-identical q
# ---------------------------------------------------------------------------
def _run(device, env_count, steps):
    with make_world(device, env_count) as w:
        # deterministic non-trivial drive: nudge every actuated joint a fixed amount
        tgt = torch.from_dlpack(w.buffer_view(nuka.DRIVE_TARGET))
        tgt[:, 1:GO2_BLC] += 0.05
        w.step_n(steps)
        nuka.sync()
        q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
        return q.detach().cpu().clone()


def test_determinism(device):
    q1 = _run(device, 64, 40)
    q2 = _run(device, 64, 40)
    assert torch.equal(q1, q2), (
        f"non-deterministic: max|q1-q2|={(q1 - q2).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# floating base is LIVE (moves) on go2_float
# ---------------------------------------------------------------------------
def test_base_live(device):
    with make_world(device, 64) as w:
        pose = torch.from_dlpack(w.buffer_view(nuka.ARTICULATION_LINK_POSE))
        base0 = pose[0, 0, :].clone()  # env0 root link world pose (7 floats)
        w.step_n(30)  # hold at rest targets; floating base sags under gravity
        nuka.sync()
        base1 = pose[0, 0, :]
        moved = (base1 - base0).abs().max().item()
        assert torch.isfinite(base1).all()
        assert moved > 1e-4, f"base pose did not move ({moved:.3e}); not live?"


# ---------------------------------------------------------------------------
# infer-enable #40: base LINK_VELOCITY read is zero-copy, shaped (env,13,6),
# finite, and the floating base's root velocity is non-zero & changing.
# ---------------------------------------------------------------------------
def test_link_velocity_zero_copy_and_live(device):
    with make_world(device, 64) as w:
        view = w.buffer_view(nuka.LINK_VELOCITY)
        vel = torch.from_dlpack(view)
        assert vel.is_cuda and vel.dtype == torch.float32
        assert tuple(vel.shape) == (64, GO2_BLC, 6)  # omega-first [wx,wy,wz,vx,vy,vz]
        # zero-copy: torch aliases the engine device buffer.
        assert vel.data_ptr() == w.buffer_device_ptr(nuka.LINK_VELOCITY)
        assert torch.isfinite(vel).all()

        root0 = vel[0, 0, :].clone()  # env0 root base spatial velocity (body frame)
        w.step_n(30)  # floating base settles under gravity -> base picks up velocity
        nuka.sync()
        root1 = vel[0, 0, :]          # zero-copy view reflects new state
        assert torch.isfinite(root1).all()
        assert root1.norm().item() > 1e-4, "root base velocity ~zero (not live?)"
        assert (root1 - root0).abs().max().item() > 1e-5, "base velocity did not change"


# ---------------------------------------------------------------------------
# infer-enable #40: writable PD gain fields (Kp/Kd/force-limit) are zero-copy
# (device-ptr match) and writes reach the solver -- same target, stiffer gains
# pull q measurably closer to the raised target.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "field", [nuka.DRIVE_STIFFNESS, nuka.DRIVE_DAMPING, nuka.DRIVE_FORCE_LIMIT]
)
def test_drive_gain_fields_zero_copy(device, field):
    with make_world(device, 64) as w:
        view = w.buffer_view(field)
        g = torch.from_dlpack(view)
        assert g.is_cuda and g.dtype == torch.float32
        assert tuple(g.shape) == (64, GO2_BLC)
        # writable + zero-copy: torch aliases the engine device buffer.
        assert g.data_ptr() == w.buffer_device_ptr(field), "DLPack made a copy!"


def _settled_q_with_gains(device, kp, kd, write_gains):
    """Run 64-env go2_float with the SAME +0.25 target perturbation; optionally
    overwrite the actuated-joint gains to (kp,kd). Returns settled q (CPU)."""
    with make_world(device, 64) as w:
        tgt = torch.from_dlpack(w.buffer_view(nuka.DRIVE_TARGET))
        tgt[:, 1:GO2_BLC] += 0.25  # same perturbation in both runs
        if write_gains:
            kp_v = torch.from_dlpack(w.buffer_view(nuka.DRIVE_STIFFNESS))
            kd_v = torch.from_dlpack(w.buffer_view(nuka.DRIVE_DAMPING))
            kp_v[:, 1:GO2_BLC] = kp  # write Go2 training gains in place (zero-copy)
            kd_v[:, 1:GO2_BLC] = kd
        w.step_n(25)
        nuka.sync()
        q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
        return q.detach().cpu().clone()


def test_writing_gains_reaches_solver(device):
    q_cooked = _settled_q_with_gains(device, 0.0, 0.0, write_gains=False)
    q_stiff = _settled_q_with_gains(device, 20.0, 0.5, write_gains=True)  # Go2 Kp/Kd
    diff = (q_stiff - q_cooked).abs()
    total = diff.sum().item()
    differing = int((diff > 1e-4).sum().item())
    assert total > 1e-3, (
        f"writing Kp=20/Kd=0.5 did not change PD response (sum|dq|={total:.3e})"
    )
    assert differing >= GO2_BLC - 1, "fewer than one env's actuated joints responded"


# ---------------------------------------------------------------------------
# p03 RL autoreset: World.reset() / World.reset_envs(ids) restore the engine's
# AUTHORITATIVE internal state (floating-base pose, base/joint velocities, joint
# positions). The headline gate is that the reset SURVIVES a step: a write
# through the ARTICULATION_LINK_POSE view alone would be overwritten by the
# integrator within one step (the bug this primitive fixes), so we assert the
# reset base pose is still near the initial pose after a step -- via the engine
# reset, not a view write. Plus per-env isolation, reset-all, and the int-array
# argument types (list / numpy / torch).
# ---------------------------------------------------------------------------
def _base_z(w, env):
    pose = torch.from_dlpack(w.buffer_view(nuka.ARTICULATION_LINK_POSE))
    return float(pose[env, 0, 2].item())  # env root link world z (slot 2 = pz)


def test_reset_envs_authority_and_isolation(device):
    with make_world(device, 8) as w:
        # Capture the creation-time initial base z (before any step) for env 2.
        nuka.sync()
        init_z = _base_z(w, 2)

        # Diverge: hold at rest targets and let the floating base settle under
        # gravity for many steps, well away from the un-stepped init.
        w.step_n(200)
        nuka.sync()
        q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
        qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
        diverged_q = q.detach().cpu().clone()
        diverged_qd = qd.detach().cpu().clone()
        diverged_z2 = _base_z(w, 2)
        assert abs(diverged_z2 - init_z) > 5e-3, "env did not diverge -- vacuous"

        # Reset env 2 only.
        w.reset_envs([2])
        nuka.sync()
        # Isolation: every OTHER env's q/qd is byte-unchanged from diverged.
        after_q = q.detach().cpu().clone()
        after_qd = qd.detach().cpu().clone()
        blc = w.base_link_count
        for env in range(w.env_count):
            if env == 2:
                continue
            s = slice(env * blc, (env + 1) * blc)
            assert torch.equal(after_q.flatten()[s], diverged_q.flatten()[s]), (
                f"reset_envs perturbed un-listed env {env} (q)"
            )
            assert torch.equal(after_qd.flatten()[s], diverged_qd.flatten()[s]), (
                f"reset_envs perturbed un-listed env {env} (qd)"
            )
        # env 2 qd restored to the init (zero).
        s2 = slice(2 * blc, 3 * blc)
        assert after_qd.flatten()[s2].abs().max().item() == 0.0, (
            "reset env 2 qd not restored to the initial zero velocity"
        )

        # AUTHORITY ACROSS A STEP: one more step; env 2's base z stays near init
        # (does NOT snap back to the diverged pose). This is the load-bearing gate.
        w.step()
        nuka.sync()
        stepped_z2 = _base_z(w, 2)
        assert abs(stepped_z2 - init_z) < 5e-3, (
            f"reset env base z snapped away after a step "
            f"(stepped={stepped_z2:.4f} init={init_z:.4f})"
        )


def test_reset_all_returns_every_env_to_init(device):
    with make_world(device, 8) as w:
        nuka.sync()
        q0 = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION)).detach().cpu().clone()
        w.step_n(200)
        nuka.sync()
        w.reset()
        nuka.sync()
        q1 = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION)).detach().cpu().clone()
        assert torch.equal(q0, q1), "reset() did not return q to the initial snapshot"
        qd1 = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
        assert qd1.abs().max().item() == 0.0, "reset() did not zero qd"


@pytest.mark.parametrize(
    "mk_ids",
    [
        lambda: [1, 4],                                  # python list
        lambda: np.array([1, 4], dtype=np.int64),        # numpy int array
        lambda: torch.tensor([1, 4], dtype=torch.int32), # torch (CPU) int tensor
    ],
    ids=["list", "numpy", "torch"],
)
def test_reset_envs_accepts_int_array_types(device, mk_ids):
    with make_world(device, 8) as w:
        w.step_n(50)
        nuka.sync()
        # Should not raise for any of the accepted 1-D int array types.
        w.reset_envs(mk_ids())
        nuka.sync()
        q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
        assert torch.isfinite(q).all()
