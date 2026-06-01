"""HEADLINE (spec Task 5.4.5, exit criterion #4): JAX vs PyTorch gradient agreement.

Computes the mass-gradient AND the action-gradient on an IDENTICAL go2_stand
scene, with IDENTICAL action sequence, mass params, and loss weight, BOTH via:

  * ``nuka.autograd.differentiable_rollout`` + torch ``loss.backward()``  (the
    proven torch reference path), and
  * ``nuka.jax_frontend.differentiable_rollout`` + ``jax.grad``           (the
    new JAX custom_vjp frontend).

Both frontends drive the SAME deterministic engine adjoint (the SAME
``_seed_from_grad`` + ``tape.backward`` reverse pass), so the two gradients must
AGREE TIGHTLY -- expected near bit-close, gated at rel-err < 1e-4. The actual
worst rel-err is printed (the exit-#4 headline number).

CRITICAL harness rule (same as the FD gradcheck): the rollout mutates engine
state IN PLACE, so EVERY evaluation builds a FRESH single-env world + tape. The
shared inputs (actions/mass/weight) are built ONCE as numpy float32 and fed
identically to both paths, so the only difference in the cotangent path is the
frontend, not the input bits.

Run (single GPU only):
    export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false
    python -m pytest python/tests/test_jax_grad_agreement_pytorch.py -q
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

from nuka.autograd import differentiable_rollout as torch_rollout  # noqa: E402
from nuka.jax_frontend import differentiable_rollout as jax_rollout  # noqa: E402

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA device required for the JAX/torch agreement", allow_module_level=True)

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCENE_STAND = os.path.join(_REPO, "examples", "scenes", "go2_stand.usda")

_DEV = nuka.Device.create(0)


def teardown_module(module):  # noqa: ARG001
    _DEV.close()


def _make_world_tape(scene, gravity_z):
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


def _q0_actuated_np(world, tape):
    return torch.from_dlpack(tape.state_view(nuka.JOINT_POSITION))[:, 1:].detach().cpu().numpy().astype(np.float32)


def _rel_err(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.maximum(np.abs(b), 1e-8)
    return float(np.max(np.abs(a - b) / denom))


# ---------------------------------------------------------------------------
# shared inputs (built ONCE as numpy float32, fed identically to both paths)
# ---------------------------------------------------------------------------
_K = 3
_PARAM_IDX = [2, 6]  # FR thigh + FR calf (genuine nonzero-grad links)


def _build_shared():
    """Build the identical action sequence, mass params, and loss weight (numpy
    float32) used by BOTH the torch and jax evaluations."""
    w0, t0 = _make_world_tape(_SCENE_STAND, -9.81)
    try:
        q0 = _q0_actuated_np(w0, t0)  # (1, ad)
        ad = int(q0.shape[1])
    finally:
        t0.destroy()
        w0.destroy()
    # rest-hold + small offset so PD torque is sensitive to both actions & mass.
    actions = np.repeat(q0 + 0.03, _K, axis=0).astype(np.float32)  # (K, ad)
    mass = np.array([1.0, 1.1], dtype=np.float32)                   # (2,)
    weight = np.linspace(0.5, 1.5, ad, dtype=np.float32)           # (ad,)
    return actions, mass, weight, ad


def _torch_grads(actions_np, mass_np, weight_np):
    """grad via torch differentiable_rollout + loss.backward() on a fresh w/tape."""
    world, tape = _make_world_tape(_SCENE_STAND, -9.81)
    try:
        actions = torch.tensor(actions_np, dtype=torch.float32, device="cuda", requires_grad=True)
        params = torch.tensor(mass_np, dtype=torch.float32, device="cuda", requires_grad=True)
        weight = torch.tensor(weight_np, dtype=torch.float32, device="cuda")
        out = torch_rollout(world, tape, actions, params=params,
                            param_link_indices=_PARAM_IDX, obs="qdot")  # (1, ad)
        loss = (out[0] * weight).sum()
        loss.backward()
        return (actions.grad.detach().cpu().numpy().copy(),
                params.grad.detach().cpu().numpy().copy(),
                float(loss.item()))
    finally:
        tape.destroy()
        world.destroy()


def _jax_grads(actions_np, mass_np, weight_np):
    """grad via jax differentiable_rollout + jax.grad on a fresh w/tape."""
    actions0 = jnp.asarray(actions_np)
    mass0 = jnp.asarray(mass_np)
    weight = jnp.asarray(weight_np)

    # The tape must stay ALIVE through the whole grad call (fwd steps it, the
    # custom_vjp backward runs tape.backward) -- teardown wraps the grad call,
    # NOT the differentiated closure. (Same lifecycle rule the torch path gets
    # for free by running loss.backward() inside its try block.)
    world, tape = _make_world_tape(_SCENE_STAND, -9.81)
    try:
        def loss(actions, params):
            out = jax_rollout(world, tape, actions, params=params,
                             param_link_indices=_PARAM_IDX, obs="qdot")  # (1, ad)
            return jnp.sum(out[0] * weight)

        val, (ga, gp) = jax.value_and_grad(loss, argnums=(0, 1))(actions0, mass0)
        return (np.asarray(jax.device_get(ga)),
                np.asarray(jax.device_get(gp)),
                float(val))
    finally:
        tape.destroy()
        world.destroy()


def test_jax_vs_pytorch_gradient_agreement():
    """The exit-#4 headline: jax.grad and torch.backward mass+action gradients
    agree to engine round-off (gate rel-err < 1e-4; report the actual)."""
    actions_np, mass_np, weight_np = _build_shared()[:3]

    t_ga, t_gp, t_loss = _torch_grads(actions_np, mass_np, weight_np)
    j_ga, j_gp, j_loss = _jax_grads(actions_np, mass_np, weight_np)

    re_actions = _rel_err(j_ga, t_ga)
    re_params = _rel_err(j_gp, t_gp)
    re_loss = abs(j_loss - t_loss) / max(abs(t_loss), 1e-8)
    worst = max(re_actions, re_params)

    print("\n========== JAX vs PyTorch gradient agreement (exit #4) ==========")
    print(f"  loss:    torch={t_loss:.8e}  jax={j_loss:.8e}  rel_err={re_loss:.3e}")
    print(f"  g_params torch={t_gp}  jax={j_gp}  rel_err={re_params:.3e}")
    print(f"  g_actions max|torch|={np.abs(t_ga).max():.6e}  rel_err={re_actions:.3e}")
    print(f"  WORST grad rel_err = {worst:.3e}")
    print("=================================================================")

    # both paths must produce genuinely-nonzero mass gradients (not vacuous).
    assert np.all(np.abs(t_gp) > 1e-6), f"torch mass grad vacuous: {t_gp}"
    assert np.all(np.abs(j_gp) > 1e-6), f"jax mass grad vacuous: {j_gp}"
    # shapes match.
    assert j_ga.shape == t_ga.shape == actions_np.shape
    assert j_gp.shape == t_gp.shape == mass_np.shape
    # THE GATE: tight agreement (same engine adjoint -> near bit-close).
    assert worst < 1e-4, (
        f"JAX vs PyTorch gradient rel_err {worst:.3e} exceeds the 1e-4 gate "
        f"(actions={re_actions:.3e}, params={re_params:.3e})"
    )
