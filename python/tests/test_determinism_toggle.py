"""pytest: p01-W4 D1/D2 determinism toggle (the C ABI + nanobind mechanism).

Run (single GPU only):
    export CUDA_VISIBLE_DEVICES=0
    python -m pytest python/tests/test_determinism_toggle.py -q

Proves the p01-W4 deliverable -- the wired determinism MECHANISM, NOT a phantom
D2 speedup:
  * The module exposes DETERMINISM_STRONG == 0 and DETERMINISM_WEAK == 1.
  * STRONG is the default: World.create_from_scene(..., determinism=0) and the
    no-arg form take the same (D1) path.
  * STRONG (D1) is the bit-exact contract: two STRONG worlds run with the SAME
    seed/drive give BYTE-IDENTICAL q (same-process two-run reproducibility -- the
    in-process witness of the D1 byte-exact property; the cross-run/golden bar is
    enforced by the C++/oracle ctests).
  * WEAK (D2) runs without error and is SAME-PROCESS reproducible (two WEAK worlds,
    same seed -> byte-identical within one process). D2 is the reserved escape
    hatch (today it shares the D1 kernels); we deliberately do NOT assert D2 is
    byte-exact across processes and do NOT hold D2 to the D1 golden.
  * An invalid determinism level (> 1) is rejected (does not silently run as D1).
"""

from __future__ import annotations

import pytest
import torch

import nuka

SCENE = "/root/Nuka-Physics/examples/scenes/go2_float.usda"
GO2_BLC = 13  # base_link_count: root + 12 actuated leg joints


@pytest.fixture(scope="module")
def device():
    dev = nuka.Device.create(0)
    yield dev
    dev.close()


def _run(device, determinism, env_count=64, steps=40):
    """Create a world at `determinism`, apply a fixed deterministic drive nudge,
    step, and return the host q tensor (a copy)."""
    with nuka.World.create_from_scene(
        device, SCENE, env_count, determinism=determinism
    ) as w:
        tgt = torch.from_dlpack(w.buffer_view(nuka.DRIVE_TARGET))
        tgt[:, 1:GO2_BLC] += 0.05  # same fixed seed for every run.
        w.step_n(steps)
        nuka.sync()
        q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
        return q.detach().cpu().clone()


# ---------------------------------------------------------------------------
# constants + default
# ---------------------------------------------------------------------------
def test_determinism_constants_present():
    assert nuka.DETERMINISM_STRONG == 0
    assert nuka.DETERMINISM_WEAK == 1


def test_strong_is_the_default(device):
    # The no-arg (zero-init desc) form must match an explicit STRONG world.
    with nuka.World.create_from_scene(device, SCENE, 64) as w:
        tgt = torch.from_dlpack(w.buffer_view(nuka.DRIVE_TARGET))
        tgt[:, 1:GO2_BLC] += 0.05
        w.step_n(40)
        nuka.sync()
        q_default = (
            torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
            .detach()
            .cpu()
            .clone()
        )
    q_strong = _run(device, nuka.DETERMINISM_STRONG)
    assert torch.equal(q_default, q_strong), (
        "default determinism is not the Strong/D1 path: "
        f"max|d-s|={(q_default - q_strong).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# (a) STRONG (D1) is two-run byte-exact -- the in-process witness of the
#     bit-exact contract.
# ---------------------------------------------------------------------------
def test_strong_two_run_byte_exact(device):
    q1 = _run(device, nuka.DETERMINISM_STRONG)
    q2 = _run(device, nuka.DETERMINISM_STRONG)
    assert torch.equal(q1, q2), (
        "STRONG/D1 is not byte-exact across two in-process runs: "
        f"max|q1-q2|={(q1 - q2).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# (b) WEAK (D2) runs error-free AND is same-process reproducible. We do NOT
#     assert D2 == D1, and we do NOT hold D2 to a cross-process / golden bar.
# ---------------------------------------------------------------------------
def test_weak_runs_and_is_same_process_reproducible(device):
    q1 = _run(device, nuka.DETERMINISM_WEAK)
    q2 = _run(device, nuka.DETERMINISM_WEAK)
    assert q1.numel() > 0, "WEAK/D2 produced an empty q buffer"
    assert torch.isfinite(q1).all(), "WEAK/D2 produced non-finite q"
    assert torch.equal(q1, q2), (
        "WEAK/D2 is not reproducible within one process (same seed): "
        f"max|q1-q2|={(q1 - q2).abs().max().item():.3e}"
    )


# ---------------------------------------------------------------------------
# invalid level is rejected (not silently treated as D1)
# ---------------------------------------------------------------------------
def test_invalid_determinism_rejected(device):
    with pytest.raises(Exception):
        nuka.World.create_from_scene(device, SCENE, 64, determinism=2)
