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

import numpy as np
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


# ============================================================================
# v0.5 -- the REAL differentiable rollout (tape-based engine adjoint)
# ============================================================================
#
# This path is SEPARATE from the frozen ``step``/``_NukaPhysicsStep`` skeleton
# above. The skeleton keeps its locked single-step ``(world, actions)``
# signature and its documented zero-stub backward; nothing here touches it.
#
# Here we wrap an ENTIRE contact-free, single-env PD rollout in ONE
# ``torch.autograd.Function`` whose ``backward`` is a SINGLE ``tape.backward``
# call. The adjoint is the engine's own deterministic reverse pass over the
# recorded tape, so gradients are exact (engine-consistent) rather than
# autograd-traced through the Python state writes.
#
# Two gradient channels are exposed:
#   * grad w.r.t. ``actions_seq`` -- the per-step DRIVE_TARGET the policy emits,
#     read back from ``tape.backward``'s ``grad_actions``.
#   * grad w.r.t. ``params`` (link masses) -- read back from ``tape.backward``'s
#     ``grad_parameters``. The mass values are PUSHED into the world via
#     ``world.set_link_mass`` in forward (NOT autograd-traced); the gradient
#     comes purely from the engine's deterministic ``grad_parameters``, copied
#     host->device. That host->device copy is bit-deterministic, so the strong
#     determinism guarantee (D1) is preserved end to end.
#
# The observation is read from the LIVE tape state via ``tape.state_view`` --
# NOT ``world.buffer_view``, which is the stale host mirror that
# ``step_with_tape`` does not refresh.

_OBS = ("q", "qdot", "v_root")


def _seed_from_grad(obs: str, grad_out: "torch.Tensor", n: int) -> "np.ndarray":
    """Build the HOST ``float32[8n]`` adjoint seed for ``tape.backward``.

    Layout (per the C-ABI contract): ``[dL/dq'(n) | dL/dqd'(n) | dL/dv_root'(6n)]``.
    We place the upstream ``grad_out`` into the third matching ``obs``:

      * ``"q"``      -> ``seed[1 : 1+g.size]``       (q-third, skip root slot 0)
      * ``"qdot"``   -> ``seed[n+1 : n+1+g.size]``   (qd-third middle, skip root)
      * ``"v_root"`` -> ``seed[2*n : 2*n+6]``        (v_root-third, root == link 0)

    The ``[1:]`` root-slot skip here MUST mirror the ``[1:]`` slice used on the
    observation-return and grad-action-readback sides.
    """
    seed = np.zeros(8 * n, dtype=np.float32)
    g = grad_out.detach().reshape(-1).to("cpu", torch.float32).contiguous().numpy()
    if obs == "q":
        seed[1 : 1 + g.size] = g          # dL/dq' third, skip root slot 0
    elif obs == "qdot":
        seed[n + 1 : n + 1 + g.size] = g  # dL/dqd' third (middle), skip root
    elif obs == "v_root":
        seed[2 * n : 2 * n + 6] = g       # dL/dv_root' third, root link == link 0
    else:
        raise ValueError(f"obs must be one of {_OBS}, got {obs!r}")
    return seed


class _NukaDiffRollout(torch.autograd.Function):
    """Differentiable contact-free single-env PD rollout (real engine adjoint).

    forward inputs (6): ``(world, tape, actions_seq, params, param_link_indices,
    obs)``. ``world``/``tape`` are non-tensor; ``actions_seq`` is the
    ``(K, action_dim)`` per-step actuated-joint target sequence; ``params`` is
    the ``(P,)`` leaf tensor of link masses (or ``empty(0)`` for the no-param
    path); ``param_link_indices`` are the GLOBAL link indices those masses map
    to; ``obs`` selects the observation channel returned.

    backward is ONE ``tape.backward`` call. Returns one grad per forward input,
    in order: ``(None, None, grad_actions, grad_params, None, None)``.
    """

    @staticmethod
    def forward(ctx, world, tape, actions_seq, params, param_link_indices, obs):  # type: ignore[override]
        assert obs in _OBS, f"obs must be one of {_OBS}, got {obs!r}"
        assert world.env_count == 1, "the tape path is SINGLE-ENV (env_count must be 1)"
        K, ad = actions_seq.shape
        assert ad == world.base_link_count - 1, (
            f"action_dim {ad} must equal base_link_count-1 "
            f"({world.base_link_count - 1}) -- actuated joints occupy slots [1:]"
        )

        # --- push the forward mass values into the engine (NOT autograd-traced)
        # The gradient w.r.t. these masses comes from the engine's deterministic
        # grad_parameters in backward, not from tracing set_link_mass.
        if params.numel() > 0:
            pv = params.detach().to("cpu", torch.float32).contiguous().reshape(-1)
            for i, li in enumerate(param_link_indices):
                world.set_link_mass(int(li), float(pv[i]))

        # --- drive the rollout, recording each step onto the tape -------------
        # The DRIVE_TARGET view is (1, base_link_count); slot 0 is the inert root,
        # actuated joints are slots [1:]. step_with_tape advances ONE contact-free
        # PD step in place using whatever DRIVE_TARGET currently holds.
        tgt = torch.from_dlpack(world.buffer_view(Field.DRIVE_TARGET))
        for k in range(K):
            tgt[:, 1:].copy_(actions_seq[k].detach().reshape(1, ad))
            tape.step_with_tape()

        # --- read the observation from the LIVE tape state -------------------
        # state_view aliases the state step_with_tape advances; buffer_view is the
        # stale host mirror and would return pre-rollout values.
        if obs == "q":
            out = torch.from_dlpack(tape.state_view(Field.JOINT_POSITION))[:, 1:].clone()   # (1, ad)
        elif obs == "qdot":
            out = torch.from_dlpack(tape.state_view(Field.JOINT_VELOCITY))[:, 1:].clone()   # (1, ad)
        else:  # "v_root"
            out = torch.from_dlpack(tape.state_view(Field.LINK_VELOCITY))[0, 0, :].clone()  # (6,)

        ctx.save_for_backward(actions_seq, params)
        ctx.tape = tape
        ctx.obs = obs
        ctx.n = tape.link_count
        ctx.action_dim = ad
        ctx.param_link_indices = tuple(int(i) for i in param_link_indices)
        return out

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        actions_seq, params = ctx.saved_tensors

        seed = _seed_from_grad(ctx.obs, grad_out, ctx.n)
        ga_np, gp_np = ctx.tape.backward(seed)                       # (K, n), (n,)

        # grad_actions: take the actuated slots [1:1+ad], move to the actions
        # device/dtype, and reshape to the input shape. The [1:] skip mirrors the
        # DRIVE_TARGET slot-0 (root) skip used in forward.
        ga = (
            torch.from_numpy(np.ascontiguousarray(ga_np[:, 1 : 1 + ctx.action_dim]))
            .to(device=actions_seq.device, dtype=actions_seq.dtype)
            .reshape(actions_seq.shape)
        )

        # grad_params: gather the engine's grad_parameters at the parameter link
        # indices (host->device, deterministic). Empty-param path -> zeros.
        if params.numel() > 0:
            gp = (
                torch.from_numpy(np.ascontiguousarray(gp_np[list(ctx.param_link_indices)]))
                .to(params)
                .reshape(params.shape)
            )
        else:
            gp = torch.zeros_like(params)

        # one grad per forward input: (world, tape, actions_seq, params, indices, obs)
        return None, None, ga, gp, None, None


def differentiable_rollout(world, tape, actions_seq, *, params=None,
                           param_link_indices=None, obs="qdot"):
    """Differentiable contact-free single-env PD rollout over ``tape``.

    Drives ``actions_seq`` ``(K, action_dim)`` into DRIVE_TARGET slots [1:] one
    step at a time (recording each onto ``tape``) and returns the observation
    selected by ``obs`` read from the LIVE tape state:

      * ``"q"``      -> post-rollout actuated JOINT_POSITION ``(1, action_dim)``
      * ``"qdot"``   -> post-rollout actuated JOINT_VELOCITY ``(1, action_dim)``
      * ``"v_root"`` -> post-rollout root-link spatial velocity ``(6,)`` (omega
        first: ``[wx, wy, wz, vx, vy, vz]``)

    ``out.backward()`` runs the engine's deterministic adjoint as a SINGLE
    ``tape.backward`` and populates ``actions_seq.grad`` (and ``params.grad`` if
    ``params`` is given). The mass-parameter gradient comes from the engine's
    ``grad_parameters`` (host->device copy, D1-preserving), NOT from
    autograd-tracing ``set_link_mass``.

    ``params`` (optional): a 1-D leaf tensor of link masses. If given,
    ``param_link_indices`` (a sequence of GLOBAL link indices, same length) is
    REQUIRED. If ``params`` is ``None`` the rollout carries no mass gradient.

    SINGLE-ENV ONLY: ``world.env_count`` must be 1. The rollout mutates engine
    state IN PLACE, so each forward eval needs a FRESH world+tape.
    """
    if params is None:
        params = torch.empty(0, dtype=torch.float32)
        param_link_indices = ()
    elif param_link_indices is None:
        raise ValueError("param_link_indices is required when params is given")
    return _NukaDiffRollout.apply(
        world, tape, actions_seq, params, tuple(param_link_indices), obs
    )


def differentiable_step(world, tape, actions, *, params=None,
                        param_link_indices=None, obs="qdot"):
    """Single-step (K=1) wrapper around :func:`differentiable_rollout`.

    ``actions`` is ``(action_dim,)`` or ``(1, action_dim)``; it is normalized to
    a ``(1, action_dim)`` sequence and delegated. Advances exactly ONE
    contact-free PD step per tape, single-env. See
    :func:`differentiable_rollout` for the gradient/observation contract.
    """
    if actions.dim() == 1:
        actions_seq = actions.reshape(1, -1)
    else:
        actions_seq = actions.reshape(1, actions.shape[-1])
    return differentiable_rollout(
        world, tape, actions_seq, params=params,
        param_link_indices=param_link_indices, obs=obs,
    )


__all__ = [
    "step",
    "_NukaPhysicsStep",
    "differentiable_rollout",
    "differentiable_step",
    "_NukaDiffRollout",
    "_seed_from_grad",
]
