"""Regression: gradient-based system-ID recovers a Go2 link mass (<1%).

This is the runnable, CI-discoverable companion to
``examples/demos/go2_system_id.py``. It runs a cheaper, fixed-seed version of the
same contact-free fixed-base direct-dynamics mass-recovery loop and asserts the
HEADLINE v0.5 exit-criterion #7 claim empirically:

  (a) the recovered mass is within 1% of the ground-truth mass, where "recovered"
      is the BEST-LOSS iterate over a fixed budget (answer-blind: loss is the only
      observable; selecting by rel-error would peek at the true mass), AND
  (b) the loss dropped substantially (best < 0.1 * initial) -- NOT strict
      per-iteration monotonicity (Adam overshoots; that is expected), AND
  (c) the loss-vs-mass channel is non-flat (mass gradient at the start guess is
      genuinely nonzero -- the basin the descent rides), AND
  (d) determinism (D1): two identical-seed runs give a BIT-IDENTICAL recovered
      -mass sequence.

The mass gradient itself is proven separately (to ~1.8e-5 rel-err) by
``test_mass_gradcheck.py``; this test proves that the proven gradient actually
*converges* an optimizer to the true mass.

Spec deviation (documented): the v0.5 spec names this
``tests/regression/test_go2_system_id_convergence.py``, but that directory holds
C++ ctests the pytest gate does not discover, so the runnable Python regression
lives here under ``python/tests/`` where the existing suite finds it.

Engineering facts mirrored from the demo (do NOT relitigate -- see the demo
docstring): single-env fixed-base scene, g=-9.81, thigh link GLOBAL index 2,
K in the transient regime, the SAME fixed rest-hold action sequence drives both
the ground-truth and every optimization rollout, mass passed as a TENSOR via
``params=`` (``.item()`` would sever the graph), and a FRESH world+tape per
iteration kept alive until AFTER ``loss.backward()`` (``world.reset()`` is "not
supported" and ``tape.reset()`` alone is not bit-identical to a fresh rollout).
"""

from __future__ import annotations

import os

import torch
import pytest

torch = pytest.importorskip("torch")
nuka = pytest.importorskip("nuka")
from nuka.autograd import differentiable_rollout  # noqa: E402

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

if not torch.cuda.is_available():  # pragma: no cover
    pytest.skip("CUDA device required for the system-ID regression",
                allow_module_level=True)

# repo-relative scene path -> absolute
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCENE = os.path.join(_REPO, "examples", "scenes", "go2_system_id.usda")

# --- cheaper-than-demo but still reliably <1% configuration ---------------- #
_LINK_INDEX = 2        # thigh link (nonzero mass gradient)
_GRAVITY_Z = -9.81
_K = 30                # transient regime, max obs sensitivity (see demo probe)
_TRUE_MASS = 1.4
_START_MASS = 0.9
_LR = 0.03
_MAX_ITERS = 80        # fixed budget; best-loss iterate is the recovered value
_TOL = 0.01            # <1% recovery (best-loss rel-err is ~3e-4 -- huge margin)
_SEED = 0

_DEV = nuka.Device.create(0)


def teardown_module(module):  # noqa: ARG001
    _DEV.close()


def _make_world_tape():
    world = nuka.World.create_from_scene(_DEV, _SCENE, 1)
    world.set_gravity_z(_GRAVITY_Z)
    tape = nuka.Tape.create(
        world, checkpoint_interval=3, max_tape_entries=4096,
        max_checkpoints=512, recompute_on_backward=1,
    )
    return world, tape


def _build_actions_seq():
    world, tape = _make_world_tape()
    try:
        q0 = torch.from_dlpack(tape.state_view(nuka.JOINT_POSITION))[:, 1:].clone()
        ad = q0.shape[1]
        return q0.reshape(1, ad).repeat(_K, 1).detach().contiguous()
    finally:
        tape.destroy()
        world.destroy()


def _rollout(mass_param, actions_seq):
    """Returns (obs, world, tape); caller destroys world+tape AFTER backward()."""
    world, tape = _make_world_tape()
    obs = differentiable_rollout(
        world, tape, actions_seq,
        params=mass_param, param_link_indices=[_LINK_INDEX], obs="qdot",
    )
    return obs, world, tape


def _make_target(actions_seq):
    mass = torch.tensor([_TRUE_MASS], dtype=torch.float32, device="cuda")
    obs, world, tape = _rollout(mass, actions_seq)
    target = obs.detach().clone()
    tape.destroy()
    world.destroy()
    return target


def _run_loop(actions_seq, target):
    """Adam descent over a FIXED budget.

    Returns ``(history[(it,mass,loss)], recovered_mass, best)``.
    ``recovered_mass`` is the BEST-LOSS iterate's mass (answer-blind: loss is the
    only observable; selecting by rel-error would peek at the true mass we are
    recovering). Best-loss is monotone in iteration count -> robust, not phase-
    sensitive to where Adam's terminal oscillation lands at the cutoff. Mirrors
    ``examples/demos/go2_system_id.py``.
    """
    torch.manual_seed(_SEED)
    mass_param = torch.nn.Parameter(
        torch.tensor([_START_MASS], dtype=torch.float32, device="cuda"))
    optimizer = torch.optim.Adam([mass_param], lr=_LR)
    history = []
    best = None  # (it, mass, loss) lowest-loss iterate
    for it in range(_MAX_ITERS):
        optimizer.zero_grad()
        obs, world, tape = _rollout(mass_param, actions_seq)
        loss = (obs - target).pow(2).mean()
        loss.backward()
        cur_mass = float(mass_param.detach().cpu()[0])
        cur_loss = float(loss.detach().cpu())
        history.append((it, cur_mass, cur_loss))
        if best is None or cur_loss < best[2]:
            best = (it, cur_mass, cur_loss)
        optimizer.step()
        tape.destroy()
        world.destroy()
    return history, best[1], best


def test_go2_system_id_converges_within_one_percent():
    """End-to-end: recover the true thigh-link mass within 1% by Adam descent."""
    actions_seq = _build_actions_seq()

    # (c) channel non-flat: mass grad at the start guess must be nonzero.
    target = _make_target(actions_seq)
    sg = torch.nn.Parameter(
        torch.tensor([_START_MASS], dtype=torch.float32, device="cuda"))
    obs0, w0, t0 = _rollout(sg, actions_seq)
    (obs0 - target).pow(2).mean().backward()
    g0 = float(sg.grad.detach().cpu()[0])
    t0.destroy()
    w0.destroy()
    assert abs(g0) > 1e-8, f"mass gradient at start guess is ~0 ({g0}); channel is flat"

    # the descent -- recovered := best-LOSS iterate (answer-blind, robust)
    history, recovered, best = _run_loop(actions_seq, target)
    rel_err = abs(recovered - _TRUE_MASS) / _TRUE_MASS
    initial_loss, best_loss = history[0][2], best[2]

    print(f"\n[system-id] start_mass={_START_MASS} -> recovered={recovered:.6f} "
          f"true={_TRUE_MASS} rel_err={rel_err:.4e} (tol {_TOL}) "
          f"best@iter={best[0]} of {len(history)}")
    print(f"[system-id] loss {initial_loss:.6e} -> best {best_loss:.6e} "
          f"({initial_loss / max(best_loss, 1e-30):.1f}x), grad@start={g0:.4e}")

    # (a) ROBUST recovery assert (best-loss iterate, NOT the arbitrary-phase final)
    assert rel_err < _TOL, (
        f"recovered mass {recovered:.6f} is {rel_err:.4e} from true {_TRUE_MASS} "
        f"-- exceeds {_TOL}"
    )
    # (b) substantial loss reduction (NOT per-iter monotonicity -- Adam overshoots)
    assert best_loss < 0.1 * initial_loss, (
        f"loss did not drop substantially: {initial_loss:.3e} -> {best_loss:.3e}"
    )


def test_go2_system_id_deterministic():
    """(d) D1: two identical-seed runs give a bit-identical recovered-mass path."""
    actions_seq = _build_actions_seq()
    target = _make_target(actions_seq)

    hist1, rec1, _ = _run_loop(actions_seq, target)
    hist2, rec2, _ = _run_loop(actions_seq, target)

    m1 = torch.tensor([h[1] for h in hist1], dtype=torch.float64)
    m2 = torch.tensor([h[1] for h in hist2], dtype=torch.float64)
    print(f"\n[system-id D1] run1 recovered={rec1:.8f} run2 recovered={rec2:.8f} "
          f"len={len(m1)} bit-identical={bool(torch.equal(m1, m2))}")

    assert len(m1) == len(m2), "two runs produced different-length histories"
    assert torch.equal(m1, m2), "recovered-mass sequence not bit-identical (D1 violated)"
