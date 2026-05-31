"""pytest: nuka.autograd -- v0.3 forward-only autograd.Function skeleton.

Proves the p02 autograd exit criterion:
  * nuka.autograd.step(world, actions) runs the forward (writes actions into
    DRIVE_TARGET, steps, returns a post-step JOINT_POSITION tensor);
  * the forward output is CUDA + finite;
  * out.sum().backward() does NOT raise;
  * actions.grad is not None and -- this is the v0.3 STUB contract -- is ALL ZERO
    (v0.5 fills the real adjoint; the signature is locked).

Reuses the same scene as test_nuka_dlpack.py (go2_float.usda).

Run (single GPU only):
    export CUDA_VISIBLE_DEVICES=0
    python -m pytest python/tests/test_autograd_forward_only.py -v
"""

from __future__ import annotations

import pytest
import torch

import nuka

SCENE = "/root/Nuka-Physics/examples/scenes/go2_float.usda"
GO2_BLC = 13       # base_link_count: root slot 0 + 12 actuated leg joints
GO2_ACTUATED = 12  # action_dim: the policy action width == base_link_count - 1


@pytest.fixture(scope="module")
def device():
    dev = nuka.Device.create(0)
    yield dev
    dev.close()


def test_action_dim_is_actuated_dof(device):
    """action_dim is the ACTUATED joint width a policy emits (12 for Go2), NOT the
    raw base_link_count buffer width (13). The DRIVE_TARGET / JOINT_POSITION
    buffers are base_link_count wide (slot 0 = root); the actuated joints occupy
    slots [1:], so an (env, action_dim) actions tensor maps onto tgt[:, 1:] with no
    size mismatch -- the round-trip invariant the autograd forward relies on. (A
    12-wide Go2 policy action would RAISE if action_dim wrongly reported 13.)"""
    with nuka.World.create_from_scene(device, SCENE, 4) as w:
        assert w.base_link_count == GO2_BLC
        assert w.action_dim == GO2_ACTUATED
        assert w.action_dim == w.base_link_count - 1
        tgt = torch.from_dlpack(w.buffer_view(nuka.DRIVE_TARGET))
        # buffer is base_link_count wide; the actuated slice [:, 1:] is action_dim.
        assert w.env_count * w.base_link_count == tgt.numel()
        assert tuple(tgt[:, 1:].shape) == (w.env_count, w.action_dim)


def test_autograd_forward_only(device):
    env_count = 4
    with nuka.World.create_from_scene(device, SCENE, env_count) as w:
        actions = torch.zeros(
            env_count, w.action_dim, device="cuda", dtype=torch.float32,
            requires_grad=True,
        )

        out = nuka.autograd.step(w, actions)
        nuka.sync()

        # forward output: CUDA, finite, post-step actuated JOINT_POSITION shape
        # (env_count, action_dim) -- the actuated slice [1:], matching action width.
        assert out.is_cuda
        assert out.dtype == torch.float32
        assert tuple(out.shape) == (env_count, GO2_ACTUATED)
        assert torch.isfinite(out).all()

        # backward must not raise (v0.3 stub).
        out.sum().backward()

        # v0.3 STUB contract: grad exists and is ALL ZERO.
        assert actions.grad is not None
        assert actions.grad.shape == actions.shape
        assert torch.count_nonzero(actions.grad).item() == 0, (
            "v0.3 skeleton must return zero grads (got non-zero -- did the "
            "backward stub change?)"
        )


def test_autograd_nonzero_actions_forward(device):
    """A non-zero requires_grad actions tensor flows through the DLPack copy_
    write path (which detaches internally) without the 'cannot dlpack-export a
    requires_grad tensor' error, and still produces finite state + zero grads."""
    env_count = 4
    with nuka.World.create_from_scene(device, SCENE, env_count) as w:
        actions = torch.full(
            (env_count, w.action_dim), 0.05, device="cuda", dtype=torch.float32,
            requires_grad=True,
        )
        out = nuka.autograd.step(w, actions)
        nuka.sync()
        assert torch.isfinite(out).all()
        (out * 2.0).sum().backward()
        assert actions.grad is not None
        assert torch.count_nonzero(actions.grad).item() == 0
