"""nuka.autograd -- PyTorch ``autograd.Function`` skeleton for a physics step.

This is the **v0.3 forward-only skeleton**. The **caller-facing surface is
LOCKED**; the autograd.Function *internals* are skeleton-stage and will be
fleshed out (not redesigned at the call site) when v0.5 adds the real adjoint.

What is FROZEN (the public contract callers and Phase-3 RL code bind to):
  * ``step(world, actions) -> Tensor`` -- the public entry point. Input order is
    ``(world, actions)``; ``actions`` is the ``(env_count, action_dim)`` actuated
    -joint tensor (``action_dim`` = the policy action width, 12 for Go2 -- NOT the
    raw ``base_link_count`` buffer width).
  * ``_NukaPhysicsStep.forward(ctx, world, actions)`` input signature -- first
    positional input is the (non-tensor) ``nuka.World``, second is the actions
    tensor. forward writes ``actions`` into DRIVE_TARGET slots [1:], steps once.

What is NOT yet frozen (skeleton-stage; v0.5 may change these -- callers do not
depend on them, so this is honest about the lock's limits):
  * ``backward``'s BODY: v0.3 returns ``torch.zeros_like(actions)``; v0.5 replaces
    it with the engine adjoint using ``ctx`` + ``grad_out``.
  * ``forward``'s BODY and ``ctx`` contents: v0.5's adjoint is tape-based
    (record a differentiable tape / checkpoints, stash a tape handle on ``ctx``),
    not a single ``world.step()`` -- so forward records more than today's
    ``{world, actions}``.
  * the OUTPUT contract: v0.3 returns the cloned post-step actuated JOINT_POSITION
    ``(env_count, action_dim)``. Locomotion gradients ultimately flow through a
    richer observation (base velocity / projected gravity, composed in Python),
    so v0.5 may make ``forward`` multi-output and ``backward`` take ``*grad_outs``.
    The single-tensor return here is a SKELETON choice, not a frozen contract.

torch is an OPTIONAL dependency of the ``nuka`` package: importing ``nuka`` must
NOT require torch. This module imports torch at *module import time*, but the
package only imports this submodule lazily (PEP 562 ``__getattr__`` in
``nuka/__init__.py``), so ``import nuka`` works without torch; torch is required
only the moment ``nuka.autograd`` is actually touched.
"""

from __future__ import annotations

import torch  # noqa: F401  (module-level: this submodule is the torch-gated surface)

from ._nuka_ext import Field


class _NukaPhysicsStep(torch.autograd.Function):
    """Differentiable physics step (v0.3: forward only; backward is a zero stub).

    LOCKED signature -- see the module docstring. v0.5 only replaces the
    ``backward`` body.
    """

    @staticmethod
    def forward(ctx, world, actions: "torch.Tensor") -> "torch.Tensor":  # type: ignore[override]
        # --- write actions into the live DRIVE_TARGET device buffer ----------
        # ``actions`` is (env_count, action_dim) -- the ACTUATED joint targets a
        # policy emits (12-wide for Go2). The engine DRIVE_TARGET buffer is
        # base_link_count wide (== action_dim + 1): slot 0 is the root link, which
        # is inert under the PD drive on a floating base, and the actuated joints
        # occupy slots [1:]. So we write into ``tgt[:, 1:]`` and leave the root slot
        # untouched -- exactly matching the proven go2_policy_drive harness.
        #
        # We use the ZERO-COPY DLPack view + an in-place ``copy_`` rather than
        # ``world.set_drive_targets(...)`` on purpose: ``set_drive_targets``
        # DLPack-exports its argument, and torch REFUSES to ``__dlpack__``-export
        # a tensor that ``requires_grad`` ("use tensor.detach()"). ``copy_`` into
        # the engine buffer slice sidesteps that and is a single strided D2D copy.
        # We ``.detach()`` the source so an autograd-tracked ``actions`` flows in
        # cleanly. The DRIVE_TARGET view is (env_count, base_link_count); the
        # slice ``tgt[:, 1:]`` is (env_count, action_dim), matching ``actions``.
        tgt = torch.from_dlpack(world.buffer_view(Field.DRIVE_TARGET))
        tgt[:, 1:].copy_(actions.detach())

        # Stash for the v0.5 adjoint. ``world`` is a non-tensor input -> it must
        # be carried on ``ctx`` directly (not via save_for_backward). (v0.5 will
        # additionally stash a differentiable-tape / checkpoint handle here -- see
        # the module docstring; the FORWARD BODY is not frozen, only the public
        # ``step(world, actions)`` signature is.)
        ctx.world = world
        ctx.save_for_backward(actions)

        # --- advance one fixed step -----------------------------------------
        world.step()

        # --- return the post-step ACTUATED joint positions ------------------
        # slots [1:] -> (env_count, action_dim), matching the action width. We
        # CLONE: without it the next world.step() would mutate the engine buffer in
        # place behind torch's back (the version counter would never bump and
        # downstream autograd would silently consume corrupted-but-undetected
        # data). The clone is a stable, contiguous snapshot and removes the
        # in-place / version-counter hazard for the graph. (v0.5 may return a
        # richer/multi-tensor observation; see the docstring -- the output contract
        # is NOT frozen, the caller-facing input signature is.)
        jp = torch.from_dlpack(world.buffer_view(Field.JOINT_POSITION))
        out = jp[:, 1:].clone()
        return out

    @staticmethod
    def backward(ctx, grad_out: "torch.Tensor"):  # type: ignore[override]
        # v0.3 STUB: no adjoint yet. Return one grad per forward input, matching
        # forward's (ctx, world, actions) -> two non-ctx inputs:
        #   * world   : non-differentiable -> None
        #   * actions : zero gradient (the v0.3 skeleton carries no adjoint)
        #
        # v0.5 fills the real adjoint HERE and ONLY here: replace
        # ``torch.zeros_like(actions)`` with the engine's backward pass driven by
        # ``ctx.world`` + ``grad_out`` (e.g. world.step_backward(...)). The
        # SIGNATURE IS LOCKED -- arity, order, and the None-for-world slot do not
        # change between v0.3 and v0.5.
        (actions,) = ctx.saved_tensors
        return None, torch.zeros_like(actions)


def step(world, actions: "torch.Tensor") -> "torch.Tensor":
    """Differentiable physics step (v0.3 forward only; v0.5 completes backward).

    Writes ``actions`` (shape ``(world.env_count, world.action_dim)``, CUDA
    float32 -- the actuated-joint width, 12 for Go2) into DRIVE_TARGET slots [1:],
    advances one fixed step, and returns the cloned post-step actuated
    JOINT_POSITION ``(env_count, action_dim)``. ``out.backward()`` runs without
    error; in v0.3 the gradient w.r.t. ``actions`` is all zeros (the skeleton
    stub). The caller-facing signature is LOCKED for v0.5.
    """
    return _NukaPhysicsStep.apply(world, actions)


__all__ = ["step", "_NukaPhysicsStep"]
