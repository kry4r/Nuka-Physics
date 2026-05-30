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
