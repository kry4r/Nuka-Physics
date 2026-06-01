"""nuka.jax_frontend -- a ``jax.custom_vjp`` differentiable rollout (spec 5.4.3).

This is the v0.5 JAX-frontend headline (exit criterion #4). It is PURE PYTHON
over the EXISTING, proven nanobind binding -- no C++/CUDA, no rebuild, no
nanobind change. It MIRRORS the torch reference path
:class:`nuka.autograd._NukaDiffRollout` line-for-line on every engine-touching
call, reusing the SAME ``nuka.autograd._seed_from_grad`` seed builder and the
SAME ``tape.backward`` reverse pass. Because both frontends drive the same
deterministic engine adjoint, ``jax.grad`` and ``torch``'s ``loss.backward()``
gradients AGREE to engine round-off (rel-err ~1e-6) on an identical scene.

How it can work WITHOUT jit (and why jit is out of scope)
---------------------------------------------------------
The engine calls (writing DRIVE_TARGET, ``step_with_tape``, ``tape.backward``)
are SIDE-EFFECTING foreign mutations. Under ``jax.grad`` / ``jax.vjp`` *without*
``jax.jit``, the trace bottoms out on JAX's eager ``EvalTrace``: the primal
``actions``/``params`` arriving in ``_fwd`` are CONCRETE ``jax.Array`` (have
``__dlpack__``), and the cotangent ``g`` arriving in ``_bwd`` is concrete too.
So ``custom_vjp`` runs ``_fwd`` then ``_bwd`` eagerly, in order -- the foreign
mutations happen in the right sequence and the DLPack bridges succeed. This is
the supported mode here.

Under ``jax.jit`` everything becomes a *tracer* (no ``__dlpack__``), so the
engine step cannot run inline -- it would need ``jax.experimental.io_callback``
to host-call the foreign side effects. That is deliberately OUT OF SCOPE for
this frontend. To get ``jit``/``vmap``/``grad`` over pure-jax state arrays, use
:class:`nuka.jax_state_pytree.NukaWorldState` (deliverable 1), which carries no
engine step. We assert concreteness in ``_run`` so a jit misuse fails loudly
rather than silently producing wrong results.

SINGLE-ENV ONLY: ``world.env_count`` must be 1; the rollout mutates engine state
IN PLACE, so each forward eval needs a FRESH world + tape (same rule as the torch
path -- see :func:`nuka.autograd.differentiable_rollout`).
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import torch

from ._nuka_ext import Field
from .autograd import _OBS, _seed_from_grad  # REUSE the proven pure-numpy seed builder


def _field_for_obs(obs: str):
    """Map an obs channel name to the tape ``state_view`` Field (mirrors the
    torch path's obs->field selection)."""
    if obs == "q":
        return Field.JOINT_POSITION
    if obs == "qdot":
        return Field.JOINT_VELOCITY
    if obs == "v_root":
        return Field.LINK_VELOCITY
    raise ValueError(f"obs must be one of {_OBS}, got {obs!r}")


def _run(world, tape, param_link_indices, obs, action_dim, actions, params):
    """Forward primal: push masses, write actions, step the tape K times, read obs.

    This is a verbatim mirror of ``_NukaDiffRollout.forward``'s BODY; the only
    differences are that ``actions``/``params`` arrive as CONCRETE jax arrays
    (converted at the engine boundary via DLPack / numpy) and the returned obs is
    a jax array. Returns the obs jax array.
    """
    assert world.env_count == 1, "the tape path is SINGLE-ENV (env_count must be 1)"

    # --- trap-zero guard: under jax.grad (no jit) the primal is CONCRETE. If we
    # were traced (jit), `actions` would be a tracer with no __dlpack__ and the
    # bridges below would silently break -- fail loudly instead. (This is the one
    # assumption the whole eager-side-effect design rests on.)
    if not hasattr(actions, "__dlpack__"):
        raise RuntimeError(
            "nuka.jax_frontend.diff_rollout requires CONCRETE arrays (eager "
            "jax.grad/jax.vjp). It cannot be jax.jit-compiled: the engine step is "
            "a foreign side effect needing io_callback. Use NukaWorldState "
            "(jax_state_pytree) for jit/vmap over pure-jax state arrays."
        )

    K = int(actions.shape[0])
    ad = int(actions.shape[1])
    assert ad == action_dim, f"action_dim mismatch: actions has {ad}, expected {action_dim}"
    assert ad == world.base_link_count - 1, (
        f"action_dim {ad} must equal base_link_count-1 ({world.base_link_count - 1})"
    )

    # --- push the forward mass values into the engine (NOT autograd-traced).
    # The mass gradient comes from the engine's deterministic grad_parameters in
    # backward, not from tracing set_link_mass. (mirror of the torch path)
    pv = np.asarray(jax.device_get(params), dtype=np.float32).reshape(-1)
    if pv.size > 0:
        for i, li in enumerate(param_link_indices):
            world.set_link_mass(int(li), float(pv[i]))

    # --- drive the rollout, recording each step onto the tape.
    # DRIVE_TARGET view is (1, base_link_count); slot 0 is the inert root, the
    # actuated joints are slots [1:]. jax->torch via DLPack works for a concrete
    # jax array, so we copy_ the per-step jax action slice into the engine slot.
    tgt = torch.from_dlpack(world.buffer_view(Field.DRIVE_TARGET))
    for k in range(K):
        a_k = torch.from_dlpack(actions[k].reshape(1, ad))  # jax->torch (zero-copy)
        tgt[:, 1:].copy_(a_k)
        tape.step_with_tape()

    # --- read the observation from the LIVE tape state (NOT world.buffer_view,
    # which is the stale host mirror). Materialize (+ 0.0 / asarray) so the
    # returned primal does not alias engine device memory.
    field = _field_for_obs(obs)
    view = jnp.from_dlpack(tape.state_view(field))
    if obs == "v_root":
        out = jnp.asarray(view[0, 0, :]) + 0.0          # (6,)
    else:
        out = jnp.asarray(view[:, 1:]) + 0.0            # (1, ad)
    return out


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 4))
def diff_rollout(world, tape, param_link_indices, obs, action_dim, actions, params):
    """Differentiable contact-free single-env PD rollout (engine adjoint).

    Non-differentiable args (nondiff_argnums 0..4): ``world``, ``tape``,
    ``param_link_indices`` (tuple of GLOBAL link indices), ``obs`` (channel
    name), ``action_dim``. Differentiable args: ``actions`` ``(K, action_dim)``
    jax array, ``params`` ``(P,)`` jax array of link masses.

    Returns the obs jax array selected by ``obs`` (``"q"``/``"qdot"`` ->
    ``(1, action_dim)``; ``"v_root"`` -> ``(6,)``). The bare ``diff_rollout`` is
    also a valid forward primal on its own.
    """
    return _run(world, tape, param_link_indices, obs, action_dim, actions, params)


def _fwd(world, tape, param_link_indices, obs, action_dim, actions, params):
    """custom_vjp forward: run the primal, stash the residual.

    The residual carries only the input SHAPES (the ``tape`` handle lives in the
    nondiff closure passed to ``_bwd``). We do NOT need the obs values in
    backward -- ``_seed_from_grad`` keys only off the obs CHANNEL (a nondiff
    arg) and the cotangent ``g`` -- so the residual stays minimal, mirroring the
    torch path's ``ctx`` (which saves shapes/obs/n, not the obs array)."""
    out = _run(world, tape, param_link_indices, obs, action_dim, actions, params)
    # Residual carries the input shapes AND ``n`` (tape.link_count), captured here
    # while the tape is alive -- mirrors the torch path's ``ctx.n``. The ``tape``
    # handle itself reaches ``_bwd`` via the nondiff closure; ``tape.backward`` in
    # ``_bwd`` requires the tape to still be ALIVE at backward time (callers must
    # not destroy the tape before the grad call completes).
    residual = (tuple(actions.shape), tuple(params.shape), int(tape.link_count))
    return out, residual


def _bwd(world, tape, param_link_indices, obs, action_dim, residual, g):
    """custom_vjp backward: ONE ``tape.backward`` (verbatim mirror of
    ``_NukaDiffRollout.backward``). Returns cotangents for the 2 DIFFERENTIABLE
    args only: ``(grad_actions, grad_params)`` as jax arrays."""
    actions_shape, params_shape, n = residual

    # SEED from the cotangent g (concrete under eager jax.grad). _seed_from_grad
    # takes a torch tensor with .detach()/.reshape()/.to() -- wrap the host numpy
    # g in a torch tensor so we reuse the EXACT same seed builder as the torch
    # path (bit-identical seed -> bit-close gradient).
    # ``np.array(..., copy=True)`` yields a fresh WRITABLE contiguous buffer so
    # ``torch.from_numpy`` does not warn about a non-writable array (device_get
    # may hand back a read-only buffer). _seed_from_grad only READS g_t, but a
    # writable source keeps torch happy and avoids the UB warning.
    g_np = np.array(jax.device_get(g), dtype=np.float32, copy=True).reshape(-1)
    g_t = torch.from_numpy(np.ascontiguousarray(g_np))
    seed = _seed_from_grad(obs, g_t, n)                     # HOST float32[8n]
    ga_np, gp_np = tape.backward(seed)                      # (K, n), (n,)

    # grad_actions: actuated slots [1:1+ad], reshape to the actions input shape.
    # The [1:] skip mirrors the DRIVE_TARGET slot-0 (root) skip in forward.
    ga = jnp.asarray(
        np.ascontiguousarray(ga_np[:, 1 : 1 + action_dim]).reshape(actions_shape)
    )

    # grad_params: gather the engine's grad_parameters at the parameter link
    # indices (host->device, deterministic). Empty-param path -> zeros.
    P = int(np.prod(params_shape)) if len(params_shape) else 0
    if P > 0:
        gp = jnp.asarray(
            np.ascontiguousarray(gp_np[list(param_link_indices)]).reshape(params_shape)
        )
    else:
        gp = jnp.zeros(params_shape, dtype=jnp.float32)

    # one cotangent per DIFFERENTIABLE arg, in order: (actions, params).
    return ga, gp


diff_rollout.defvjp(_fwd, _bwd)


# ---------------------------------------------------------------------------
# jax-friendly wrappers (mirror nuka.autograd.differentiable_rollout / _step)
# ---------------------------------------------------------------------------
def differentiable_rollout(world, tape, actions, *, params=None,
                           param_link_indices=None, obs="qdot"):
    """JAX differentiable contact-free single-env PD rollout over ``tape``.

    ``actions`` is a jax (or numpy) ``(K, action_dim)`` per-step actuated-joint
    target sequence. Drives each step's actuated slots [1:] (recording onto
    ``tape``) and returns the obs selected by ``obs`` read from the LIVE tape
    state:

      * ``"q"``      -> post-rollout actuated JOINT_POSITION ``(1, action_dim)``
      * ``"qdot"``   -> post-rollout actuated JOINT_VELOCITY ``(1, action_dim)``
      * ``"v_root"`` -> post-rollout root-link spatial velocity ``(6,)``
        (omega-first ``[wx,wy,wz,vx,vy,vz]``)

    ``jax.grad`` through this runs the engine's deterministic adjoint as a SINGLE
    ``tape.backward`` and yields gradients w.r.t. ``actions`` (and ``params`` if
    given). The mass-parameter gradient comes from the engine's
    ``grad_parameters`` (host->device copy), NOT from tracing ``set_link_mass``.

    ``params`` (optional): a 1-D jax/numpy array of link masses. If given,
    ``param_link_indices`` (GLOBAL link indices, same length) is REQUIRED. If
    ``None`` the rollout carries no mass gradient.

    SINGLE-ENV ONLY (``world.env_count`` == 1). The rollout mutates engine state
    IN PLACE, so each evaluation needs a FRESH world + tape. NOT jax.jit-able
    (the engine step is a foreign side effect; see the module docstring).
    """
    actions = jnp.asarray(actions, dtype=jnp.float32)
    if actions.ndim != 2:
        raise ValueError(f"actions must be (K, action_dim) 2-D, got shape {actions.shape}")
    action_dim = int(actions.shape[1])

    if params is None:
        params_j = jnp.zeros((0,), dtype=jnp.float32)
        pli = ()
    else:
        if param_link_indices is None:
            raise ValueError("param_link_indices is required when params is given")
        params_j = jnp.asarray(params, dtype=jnp.float32).reshape(-1)
        pli = tuple(int(i) for i in param_link_indices)

    return diff_rollout(world, tape, pli, obs, action_dim, actions, params_j)


def differentiable_step(world, tape, action, *, params=None,
                        param_link_indices=None, obs="qdot"):
    """Single-step (K=1) wrapper around :func:`differentiable_rollout`.

    ``action`` is ``(action_dim,)`` or ``(1, action_dim)``; normalized to a
    ``(1, action_dim)`` sequence and delegated. See
    :func:`differentiable_rollout` for the gradient / observation contract.
    """
    action = jnp.asarray(action, dtype=jnp.float32)
    if action.ndim == 1:
        actions = action.reshape(1, -1)
    else:
        actions = action.reshape(1, action.shape[-1])
    return differentiable_rollout(
        world, tape, actions, params=params,
        param_link_indices=param_link_indices, obs=obs,
    )


__all__ = [
    "diff_rollout",
    "differentiable_rollout",
    "differentiable_step",
]
