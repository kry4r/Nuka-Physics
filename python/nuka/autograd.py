"""nuka.autograd -- PyTorch ``autograd.Function`` skeleton for a physics step.

This is the **v0.3 forward-only skeleton**. The user-facing surface (the
``_NukaPhysicsStep.forward`` / ``backward`` signatures and the ``step(world,
actions)`` convenience) is **LOCKED**: v0.5 fills in the real adjoint by
replacing only the *body* of ``backward`` (and reading the handles already
stashed on ``ctx``). No caller code, no signature, and no return arity changes
between v0.3 and v0.5.

What is FROZEN (do not change in v0.5):
  * ``forward(ctx, world, actions) -> Tensor`` -- the first positional input is
    the (non-tensor) ``nuka.World``, the second is the actions tensor.
  * ``backward(ctx, grad_out) -> (None, grad_actions)`` -- the 2-tuple arity
    mirrors forward's two inputs: ``world`` gets ``None`` (non-differentiable),
    ``actions`` gets the gradient. v0.5 fills ``grad_actions``.
  * ``step(world, actions) -> Tensor`` -- the public entry point.
  * forward writes ``actions`` into the engine's DRIVE_TARGET buffer, steps, and
    returns a (cloned) post-step state tensor.

What v0.5 WILL change (and only this):
  * ``backward``'s body: replace ``torch.zeros_like(actions)`` with the engine
    adjoint (e.g. ``world.step_backward(grad_out, ...)``), using ``ctx.world``
    and ``ctx.saved_tensors``.

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
        # We use the ZERO-COPY DLPack view + an in-place ``copy_`` rather than
        # ``world.set_drive_targets(...)`` on purpose: ``set_drive_targets``
        # DLPack-exports its argument, and torch REFUSES to ``__dlpack__``-export
        # a tensor that ``requires_grad`` ("use tensor.detach()"). ``copy_`` into
        # the engine buffer sidesteps that and is a single D2D copy. We
        # ``.detach()`` the source so an autograd-tracked ``actions`` flows in
        # cleanly. The actions tensor's shape is (env_count, action_dim); the
        # DRIVE_TARGET view has the same (env_count, base_link_count) shape.
        tgt = torch.from_dlpack(world.buffer_view(Field.DRIVE_TARGET))
        tgt.copy_(actions.detach())

        # Stash for the v0.5 adjoint. ``world`` is a non-tensor input -> it must
        # be carried on ``ctx`` directly (not via save_for_backward).
        ctx.world = world
        ctx.save_for_backward(actions)

        # --- advance one fixed step -----------------------------------------
        world.step()

        # --- return a post-step state tensor --------------------------------
        # We CLONE the zero-copy JOINT_POSITION view. Without the clone, the next
        # world.step() would mutate the engine buffer in place behind torch's
        # back: torch's version counter would never bump, and downstream autograd
        # would silently consume corrupted-but-undetected data. The clone makes
        # the returned tensor a stable snapshot of the post-step state and
        # removes any in-place / version-counter hazard for the graph.
        out = torch.from_dlpack(world.buffer_view(Field.JOINT_POSITION)).clone()
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
    float32) into the world's DRIVE_TARGET buffer, advances one fixed step, and
    returns a cloned post-step JOINT_POSITION tensor. ``out.backward()`` runs
    without error; in v0.3 the gradient w.r.t. ``actions`` is all zeros (the
    skeleton stub). The signature is LOCKED for v0.5.
    """
    return _NukaPhysicsStep.apply(world, actions)


__all__ = ["step", "_NukaPhysicsStep"]
