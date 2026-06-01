"""JAX custom_vjp differentiable rollout + NukaWorldState pytree (spec 5.4.3/5.4.4).

Proves the v0.5 JAX frontend, exit criterion #4 (the engine-adjoint half):

  * ``jax.grad`` of a scalar loss through ``nuka.jax_frontend.differentiable_rollout``
    w.r.t. (actions, params=thigh/calf masses) returns FINITE, correctly-shaped,
    NONZERO gradients on a single-env go2_stand world (g=-9.81), obs="qdot".
  * ``NukaWorldState`` pytree round-trips (flatten/unflatten) and supports
    ``jax.jit`` + ``jax.vmap`` over a PURE-JAX function of its arrays (NOT the
    engine step -- the engine step is a foreign side effect, out of scope for jit).

Run (single GPU only):
    export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
    python -m pytest python/tests/test_jax_custom_vjp.py -q
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")
torch = pytest.importorskip("torch")
nuka = pytest.importorskip("nuka")

from nuka.jax_frontend import differentiable_rollout, differentiable_step  # noqa: E402
from nuka.jax_state_pytree import NukaWorldState  # noqa: E402

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA device required for the JAX tape frontend", allow_module_level=True)

# ---------------------------------------------------------------------------
# scene paths (repo-relative -> absolute)
# ---------------------------------------------------------------------------
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCENE_STAND = os.path.join(_REPO, "examples", "scenes", "go2_stand.usda")

# THIGH/CALF global link indices (hips {1,4,7,10} and root {0} are physics-zeros).
_THIGH_CALF = (2, 3, 5, 6, 8, 9, 11, 12)

_DEV = nuka.Device.create(0)


def teardown_module(module):  # noqa: ARG001
    _DEV.close()


def _make_world_tape(scene, gravity_z):
    """Fresh single-env world + tape (gravity set BEFORE Tape.create)."""
    world = nuka.World.create_from_scene(_DEV, scene, 1)
    world.set_gravity_z(gravity_z)
    tape = nuka.Tape.create(
        world,
        checkpoint_interval=3,
        max_tape_entries=1024,
        max_checkpoints=128,
        recompute_on_backward=1,
    )
    return world, tape


def _q0_actuated(world, tape):
    """Initial actuated joint positions ``(1, action_dim)`` (slots [1:])."""
    return torch.from_dlpack(tape.state_view(nuka.JOINT_POSITION))[:, 1:].clone()


# ===========================================================================
# TEST 1: jax.grad of a loss wrt (actions, params) -- finite, shaped, nonzero
# ===========================================================================
def test_jax_grad_mass_and_action_finite_nonzero():
    """go2_stand, g=-9.81, obs='qdot'. jax.grad wrt mass params (thigh/calf) and
    wrt the action sequence must be FINITE, correctly-shaped, and NONZERO."""
    K = 2
    param_idx = [2, 6]  # one FR thigh, one FR calf (genuine nonzero-grad links)

    # Capture a constant rest-hold target + obs width from a throwaway world.
    w0, t0 = _make_world_tape(_SCENE_STAND, -9.81)
    try:
        q0 = _q0_actuated(w0, t0).detach().cpu().numpy().astype(np.float32)  # (1, ad)
        ad = int(q0.shape[1])
    finally:
        t0.destroy()
        w0.destroy()

    actions0 = jnp.asarray(np.repeat(q0, K, axis=0))            # (K, ad)
    mass0 = jnp.asarray(np.array([1.0, 1.0], dtype=np.float32))  # (2,)
    weight = jnp.asarray(np.linspace(0.5, 1.5, ad, dtype=np.float32))

    # The tape must stay ALIVE through the WHOLE grad call: jax.grad runs the
    # forward (which steps the tape) AND the custom_vjp backward (tape.backward),
    # so teardown must wrap the entire grad call, NOT live inside the
    # differentiated closure (else _bwd hits a destroyed tape).
    world, tape = _make_world_tape(_SCENE_STAND, -9.81)
    try:
        def loss(actions, params):
            out = differentiable_rollout(
                world, tape, actions,
                params=params, param_link_indices=param_idx, obs="qdot",
            )  # (1, ad)
            return jnp.sum(out[0] * weight)

        (g_actions, g_params) = jax.grad(loss, argnums=(0, 1))(actions0, mass0)
    finally:
        tape.destroy()
        world.destroy()
    g_actions = np.asarray(jax.device_get(g_actions))
    g_params = np.asarray(jax.device_get(g_params))

    print(f"\n[jax.grad qdot] g_actions shape {g_actions.shape} "
          f"finite={np.isfinite(g_actions).all()} nonzero={int((np.abs(g_actions) > 0).sum())}")
    print(f"[jax.grad qdot] g_params (links {param_idx}) = {g_params} "
          f"finite={np.isfinite(g_params).all()}")

    assert g_actions.shape == (K, ad)
    assert g_params.shape == (2,)
    assert np.isfinite(g_actions).all(), "action grad not finite"
    assert np.isfinite(g_params).all(), "param grad not finite"
    # thigh/calf mass grads must be genuinely nonzero (vacuous-obs guard).
    assert np.all(np.abs(g_params) > 1e-6), f"thigh/calf grads must be nonzero, got {g_params}"
    # the action grad must be nonzero somewhere.
    assert np.any(np.abs(g_actions) > 1e-8), "action grad is all ~zero"


# ===========================================================================
# TEST 2: differentiable_step (K=1) smoke -- shape + jax.grad nonzero
# ===========================================================================
def test_jax_differentiable_step_smoke():
    """The K=1 differentiable_step wrapper: a 1-D action -> (1, ad) qdot obs, and
    jax.grad of a scalar loss populates a correctly-shaped nonzero action grad."""
    w0, t0 = _make_world_tape(_SCENE_STAND, -9.81)
    try:
        q0 = _q0_actuated(w0, t0).reshape(-1).detach().cpu().numpy().astype(np.float32)  # (ad,)
        ad = int(q0.shape[0])
    finally:
        t0.destroy()
        w0.destroy()

    action0 = jnp.asarray(q0 + 0.05)
    weight = jnp.asarray(np.linspace(0.5, 1.5, ad, dtype=np.float32))

    world, tape = _make_world_tape(_SCENE_STAND, -9.81)
    try:
        def loss(action):
            out = differentiable_step(world, tape, action, obs="qdot")  # (1, ad)
            assert out.shape == (1, ad)
            return jnp.sum(out[0] * weight)

        g = np.asarray(jax.device_get(jax.grad(loss)(action0)))
    finally:
        tape.destroy()
        world.destroy()
    print(f"\n[jax step smoke] grad shape {g.shape} nonzero {int((np.abs(g) > 0).sum())}/{ad}")
    assert g.shape == (ad,)
    assert np.isfinite(g).all()
    assert np.any(np.abs(g) > 1e-8), "K=1 action grad is all ~zero"


# ===========================================================================
# TEST 3: NukaWorldState pytree round-trip
# ===========================================================================
def test_pytree_roundtrip():
    """flatten -> (3 leaves, aux) and unflatten reconstructs identical arrays."""
    st = NukaWorldState(
        joint_pos=jnp.asarray(np.arange(4, dtype=np.float32)),
        joint_vel=jnp.asarray(np.arange(4, 8, dtype=np.float32)),
        root_vel=jnp.asarray(np.arange(8, 14, dtype=np.float32)),
    )
    leaves, treedef = jax.tree_util.tree_flatten(st)
    assert len(leaves) == 3, f"expected 3 pytree leaves, got {len(leaves)}"
    st2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(st2, NukaWorldState)
    assert np.array_equal(np.asarray(st.joint_pos), np.asarray(st2.joint_pos))
    assert np.array_equal(np.asarray(st.joint_vel), np.asarray(st2.joint_vel))
    assert np.array_equal(np.asarray(st.root_vel), np.asarray(st2.root_vel))

    # tree_map over the leaves (a pure structural transform) works.
    scaled = jax.tree_util.tree_map(lambda x: x * 2.0, st)
    assert np.array_equal(np.asarray(scaled.joint_pos), np.asarray(st.joint_pos) * 2.0)


# ===========================================================================
# TEST 4: jax.jit + jax.vmap over a PURE-JAX function of NukaWorldState
#         (NOT the engine step -- demonstrates the pytree threads through the
#          tracing transforms).
# ===========================================================================
def test_pytree_jit_vmap_grad_pure():
    """A pure-jax cost over NukaWorldState arrays: jittable, vmappable, grad-able.
    This is the jit/vmap surface (no engine step) the pytree unlocks."""

    def cost(state: NukaWorldState) -> jnp.ndarray:
        # purely-functional scalar cost over the three leaf arrays.
        return (jnp.sum(state.joint_pos ** 2)
                + jnp.sum(state.joint_vel ** 2)
                + jnp.sum(state.root_vel ** 2))

    # --- jit: compiles, runs, and the pytree is the (traced) argument ---------
    jit_cost = jax.jit(cost)
    single = NukaWorldState(
        joint_pos=jnp.ones((4,), jnp.float32),
        joint_vel=jnp.full((4,), 2.0, jnp.float32),
        root_vel=jnp.full((6,), 3.0, jnp.float32),
    )
    val = float(jit_cost(single))
    expected = 4 * 1.0 + 4 * 4.0 + 6 * 9.0  # 4 + 16 + 54 = 74
    assert abs(val - expected) < 1e-4, f"jit cost {val} != {expected}"

    # --- vmap: batch of states (leading axis on every leaf) -------------------
    B = 5
    batched = NukaWorldState(
        joint_pos=jnp.ones((B, 4), jnp.float32),
        joint_vel=jnp.full((B, 4), 2.0, jnp.float32),
        root_vel=jnp.full((B, 6), 3.0, jnp.float32),
    )
    batch_vals = np.asarray(jax.device_get(jax.vmap(cost)(batched)))
    assert batch_vals.shape == (B,)
    assert np.allclose(batch_vals, expected), f"vmap costs {batch_vals} != {expected}"

    # --- grad: differentiate the scalar cost wrt the pytree leaves ------------
    g = jax.grad(cost)(single)
    assert isinstance(g, NukaWorldState)
    assert np.allclose(np.asarray(g.joint_pos), 2.0 * np.asarray(single.joint_pos))
    assert np.allclose(np.asarray(g.joint_vel), 2.0 * np.asarray(single.joint_vel))
    assert np.allclose(np.asarray(g.root_vel), 2.0 * np.asarray(single.root_vel))

    # --- jit ∘ vmap composition (the real combinator the pytree enables) ------
    composed = float(jnp.sum(jax.jit(jax.vmap(cost))(batched)))
    assert abs(composed - B * expected) < 1e-3
    print(f"\n[pytree] jit cost={val} vmap shape={batch_vals.shape} "
          f"jit(vmap) sum={composed} (B*{expected})")
