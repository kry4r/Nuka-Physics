"""Gradient-based system identification on a fixed-base Go2 (Nuka v0.5).

HEADLINE DEMO -- recover an UNKNOWN link mass by gradient descent through a
differentiable physics rollout.

What this is (HONESTLY):
  A CONTACT-FREE, FIXED-BASE, DIRECT-DYNAMICS mass-recovery demo. The Go2's base
  is rigidly fixed (kinematic root) and there is NO ground contact. We hold a
  CONSTANT rest-pose PD target while the robot sags under gravity; the actuated
  joint velocities `qdot` after a short transient rollout are a function of the
  link masses through `qddot = M^-1 . (tau_PD - gravity_torque)`. By minimizing
  the MSE between the simulated `qdot` and a ground-truth `qdot` (generated with
  a known "true" mass) we recover that true mass.

What this is NOT:
  This is NOT locomotion / gait system-identification. There is no contact, no
  walking, no foot forces -- it is pure articulated-body direct dynamics on a
  fixed base. That is deliberate: fixed-base + obs="qdot" is the engine's CLEAN
  differentiable mass channel (the validated gradcheck measured a thigh/calf mass
  rel-err of ~1.8e-5 on exactly this configuration). Floating-base / contact mass
  channels are deferred to a later phase.

The differentiable rollout is the engine's own deterministic tape adjoint, wrapped
as a torch.autograd.Function in `nuka.autograd.differentiable_rollout`. The mass
gradient comes from the engine's `grad_parameters` (a single `tape.backward`),
host->device copied and D1-deterministic -- NOT from autograd-tracing
`set_link_mass`. We therefore pass the mass as a TENSOR via `params=...`; calling
`.item()` would sever the graph and `mass_param.grad` would stay None.

Key engineering facts baked in (see the header comments below for the rationale):
  * Single env, fixed-base scene examples/scenes/go2_system_id.usda, g=-9.81.
  * Tunable link = ONE thigh link, GLOBAL index 2 (thigh/calf links {2,3,5,6,8,9,
    11,12} carry a nonzero mass gradient; hips {1,4,7,10} are physics-zeros).
  * `world.set_link_mass(GLOBAL_INDEX, mass)` takes an INTEGER global index.
  * Rollout length K=30 keeps qdot in the TRANSIENT regime where the mass signal
    is largest (the static PD equilibrium is mass-independent). K was chosen by an
    empirical loss-vs-mass sweep: every candidate K gave a single clean basin with
    the minimum exactly at the true mass; K=30 had the steepest basin / largest
    obs sensitivity to mass.
  * The SAME fixed, deterministic `actions_seq` (rest-pose PD target repeated K
    times) drives BOTH the ground-truth rollout and every optimization rollout.
  * `world.reset()` is "not supported" and `tape.reset()` alone does NOT reproduce
    a fresh rollout bit-identically (measured diff-norm ~0.45), so we build a
    FRESH world+tape every iteration. This is correct and cheap (~6 ms/iter).
  * The world+tape MUST stay alive until AFTER `loss.backward()` -- the adjoint
    reads the live tape. Destroying the tape before backward raises
    "Tape.backward: empty tape". So we destroy only after `optimizer.step()`.

Outputs (written to examples/demos/go2_system_id_results/):
  * run_log.txt    -- the full console log (source of record for the report).
  * convergence.csv -- iter,mass,loss for every iteration.
  * loss_curve.png / mass_curve.png -- ONLY if matplotlib is importable;
    skipped gracefully otherwise (the CSV is the source of truth).

A cheaper, fixed-seed regression of this demo lives in
python/tests/test_go2_system_id_convergence.py (the v0.5 spec names
tests/regression/test_go2_system_id_convergence.py, but that directory holds C++
ctests that the pytest gate does not discover, so the runnable Python regression
goes under python/tests/ where the existing suite finds it).

Run with:  CUDA_VISIBLE_DEVICES=0 python examples/demos/go2_system_id.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

import nuka
from nuka.autograd import differentiable_rollout

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# --------------------------------------------------------------------------- #
# configuration (all tuned + verified empirically -- see the module docstring) #
# --------------------------------------------------------------------------- #
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCENE = os.path.join(_REPO, "examples", "scenes", "go2_system_id.usda")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "go2_system_id_results")

LINK_INDEX = 2          # GLOBAL link index of the tunable thigh-link mass
GRAVITY_Z = -9.81       # scene-default gravity (the clean fixed-base channel)
K = 30                  # rollout length (transient regime; max mass sensitivity)
TRUE_MASS = 1.4         # ground-truth mass of the tunable link (kg)
START_MASS = 0.9        # initial guess (~0.64x of true -- inside the convex basin)
LR = 0.03               # Adam step size (tuned so best-loss lands <1% well within budget)
MAX_ITERS = 80          # fixed iteration budget (best-loss iterate is the result)
TOL = 0.01              # success := |recovered - true| / true < TOL
SEED = 0                # determinism anchor (no RNG in our path, but pinned)


# --------------------------------------------------------------------------- #
# engine helpers                                                              #
# --------------------------------------------------------------------------- #
def make_world_tape(dev):
    """A FRESH single-env fixed-base world + recording tape.

    Gravity is set BEFORE Tape.create (the tape captures gravity at create time).
    """
    world = nuka.World.create_from_scene(dev, SCENE, 1)
    world.set_gravity_z(GRAVITY_Z)
    tape = nuka.Tape.create(
        world,
        checkpoint_interval=3,
        max_tape_entries=4096,
        max_checkpoints=512,
        recompute_on_backward=1,
    )
    return world, tape


def build_actions_seq(dev):
    """Build the SHARED, fixed action sequence ONCE: the robot's initial rest
    joint angles (actuated slots [1:]) repeated K times -> (K, action_dim).

    A constant hold-at-rest PD target makes the rollout the clean
    gravity-sag transient `qddot = M^-1 . gravity_torque` mass channel. No random
    excitation is added (it would only inject mass-independent noise)."""
    world, tape = make_world_tape(dev)
    try:
        q0 = torch.from_dlpack(tape.state_view(nuka.JOINT_POSITION))[:, 1:].clone()
        action_dim = q0.shape[1]
        actions_seq = q0.reshape(1, action_dim).repeat(K, 1).detach().contiguous()
        return actions_seq, action_dim
    finally:
        tape.destroy()
        world.destroy()


def rollout_qdot(dev, mass_param, actions_seq):
    """One differentiable rollout at the given (TENSOR) mass.

    Returns (obs, world, tape) -- the caller MUST keep world+tape alive until
    after backward(), then destroy them. The mass is passed as a tensor via
    `params=` so the engine's grad_parameters can populate `mass_param.grad`.
    """
    world, tape = make_world_tape(dev)
    obs = differentiable_rollout(
        world, tape, actions_seq,
        params=mass_param, param_link_indices=[LINK_INDEX], obs="qdot",
    )
    return obs, world, tape


def make_target(dev, actions_seq):
    """Ground-truth observation: rollout at TRUE_MASS, detached."""
    mass = torch.tensor([TRUE_MASS], dtype=torch.float32, device="cuda")
    obs, world, tape = rollout_qdot(dev, mass, actions_seq)
    target = obs.detach().clone()
    tape.destroy()
    world.destroy()
    return target


# --------------------------------------------------------------------------- #
# the optimization loop                                                       #
# --------------------------------------------------------------------------- #
def run_system_id(dev, actions_seq, target, *, max_iters=MAX_ITERS, lr=LR,
                  start_mass=START_MASS, log=print):
    """Adam descent on a single link mass over a FIXED budget.

    Returns ``(history, recovered_mass, best)`` where ``history`` is a list of
    ``(iter, mass, loss)`` tuples, ``best`` is the BEST-LOSS iterate
    ``(iter, mass, loss)`` and ``recovered_mass`` is that best-loss mass.

    Why best-LOSS (not the final iterate, and not best-rel-error):
      * Adam's terminal behavior OSCILLATES around the minimum, so the iterate at
        an arbitrary cutoff samples that oscillation at an arbitrary phase -- the
        final mass is not a robust estimate. The best-loss iterate is MONOTONE in
        iteration count (more iters can never make it worse), which is the
        property a reproducible result needs.
      * loss is the ONLY observable a real system-ID has -- it does NOT know the
        true mass. Selecting by loss is answer-BLIND (legitimate); selecting by
        rel-error would peek at the unknown we are recovering (circular).
      * Because the loss-vs-mass landscape here is a single monotone basin with
        its minimum exactly at the true mass (verified by sweep), the best-loss
        iterate also coincides with the best-accuracy iterate.

    Fresh world+tape per iter; destroyed only AFTER optimizer.step() (the adjoint
    reads the live tape during backward).
    """
    torch.manual_seed(SEED)
    mass_param = torch.nn.Parameter(
        torch.tensor([start_mass], dtype=torch.float32, device="cuda")
    )
    optimizer = torch.optim.Adam([mass_param], lr=lr)

    history = []
    best = None  # (iter, mass, loss) with the lowest loss seen
    for it in range(max_iters):
        optimizer.zero_grad()
        obs, world, tape = rollout_qdot(dev, mass_param, actions_seq)
        loss = (obs - target).pow(2).mean()
        loss.backward()
        cur_mass = float(mass_param.detach().cpu()[0])
        cur_loss = float(loss.detach().cpu())
        history.append((it, cur_mass, cur_loss))
        if best is None or cur_loss < best[2]:
            best = (it, cur_mass, cur_loss)
        optimizer.step()
        # destroy ONLY after step() -- backward already consumed the tape above.
        tape.destroy()
        world.destroy()

        rel = abs(cur_mass - TRUE_MASS) / TRUE_MASS
        if it < 5 or it % 20 == 0 or it == max_iters - 1:
            log(f"  iter {it:4d}  mass={cur_mass:.6f}  loss={cur_loss:.6e}  "
                f"rel_err={rel:.3e}")

    recovered_mass = best[1]  # best-LOSS iterate (answer-blind, monotone)
    return history, recovered_mass, best


# --------------------------------------------------------------------------- #
# output writers                                                              #
# --------------------------------------------------------------------------- #
def write_csv(path, history):
    lines = ["iter,mass,loss"]
    lines += [f"{it},{m:.10f},{l:.10e}" for (it, m, l) in history]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def write_plots(results_dir, history):
    """Write loss_curve.png + mass_curve.png IF matplotlib is importable.
    Returns True if plots were written, False if skipped (CSV is the truth)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    its = [h[0] for h in history]
    masses = [h[1] for h in history]
    losses = [h[2] for h in history]

    fig, ax = plt.subplots()
    ax.semilogy(its, losses, marker=".")
    ax.set_xlabel("iteration")
    ax.set_ylabel("qdot MSE loss (log)")
    ax.set_title("Go2 system-ID: loss convergence")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "loss_curve.png"), dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(its, masses, marker=".", label="recovered mass")
    ax.axhline(TRUE_MASS, color="r", ls="--", label=f"true mass = {TRUE_MASS}")
    ax.set_xlabel("iteration")
    ax.set_ylabel("link mass (kg)")
    ax.set_title("Go2 system-ID: mass recovery")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "mass_curve.png"), dpi=120)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    if not torch.cuda.is_available():
        print("ERROR: this demo requires a CUDA device.", file=sys.stderr)
        return 2

    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_path = os.path.join(RESULTS_DIR, "run_log.txt")
    log_lines = []

    def log(msg=""):
        print(msg)
        log_lines.append(str(msg))

    dev = nuka.Device.create(0)
    try:
        log("=" * 74)
        log("Nuka v0.5 -- gradient-based system identification (fixed-base Go2)")
        log("contact-free direct-dynamics mass recovery, obs='qdot'")
        log("=" * 74)
        log(f"scene      : {SCENE}")
        log(f"link index : {LINK_INDEX} (thigh)   gravity_z: {GRAVITY_Z}")
        log(f"K (rollout): {K}   true_mass: {TRUE_MASS}   start_mass: {START_MASS}")
        log(f"optimizer  : Adam(lr={LR})   max_iters: {MAX_ITERS}   tol: {TOL}")
        log("")

        # 1) the SHARED fixed action sequence (built once)
        actions_seq, action_dim = build_actions_seq(dev)
        log(f"action_dim : {action_dim}   actions_seq shape: "
            f"{tuple(actions_seq.shape)} (constant rest-hold PD target)")

        # 2) gradient sanity: at the start guess the mass gradient must be nonzero
        #    (the convex-basin descent depends on it). The gradcheck already
        #    proved this channel to ~1.8e-5; here we just confirm it transferred.
        sg_mass = torch.nn.Parameter(
            torch.tensor([START_MASS], dtype=torch.float32, device="cuda"))
        target_pre = make_target(dev, actions_seq)
        obs0, w0, t0 = rollout_qdot(dev, sg_mass, actions_seq)
        loss0 = (obs0 - target_pre).pow(2).mean()
        loss0.backward()
        g0 = float(sg_mass.grad.detach().cpu()[0])
        t0.destroy()
        w0.destroy()
        log(f"grad sanity: dLoss/dmass @ start_mass = {g0:.6e} "
            f"(must be nonzero -> {'OK' if abs(g0) > 1e-8 else 'FAIL'})")
        assert abs(g0) > 1e-8, (
            f"mass gradient at start guess is ~0 ({g0}); the obs channel is flat "
            f"-- system-ID cannot converge. Check K / link index."
        )
        log("")

        # 3) ground truth (same shared actions, detached)
        target = make_target(dev, actions_seq)
        log(f"target obs : qdot at true_mass, shape {tuple(target.shape)}, "
            f"norm {float(target.norm()):.6e}")
        log("")

        # 4) the optimization loop
        log("Adam descent:")
        history, recovered, best = run_system_id(dev, actions_seq, target, log=log)

        # 5) determinism (D1): run the loop AGAIN, assert bit-identical mass path
        log("")
        log("determinism check (D1): re-running the loop ...")
        hist2, recovered2, _ = run_system_id(
            dev, actions_seq, target, log=lambda *_: None)
        masses1 = torch.tensor([h[1] for h in history], dtype=torch.float64)
        masses2 = torch.tensor([h[1] for h in hist2], dtype=torch.float64)
        det_ok = (len(masses1) == len(masses2)) and bool(torch.equal(masses1, masses2))
        log(f"  run #1 length {len(masses1)}, run #2 length {len(masses2)}")
        log(f"  recovered-mass sequence bit-identical: {det_ok}")

        # 6) verdict -- recovered := the BEST-LOSS iterate (answer-blind; see
        #    run_system_id's docstring). We also report the final iterate for
        #    transparency (it is NOT cherry-picked -- best-loss is the principled
        #    estimate, the final is just the last oscillation sample).
        best_it, best_mass, best_loss = best
        rel_err = abs(recovered - TRUE_MASS) / TRUE_MASS
        initial_loss = history[0][2]
        final_mass = history[-1][1]
        final_loss = history[-1][2]
        log("")
        log("=" * 74)
        log(f"RESULT: recovered_mass = {recovered:.6f}  true_mass = {TRUE_MASS}  "
            f"(best-loss iterate @ iter {best_it})")
        log(f"        relative error = {rel_err:.4e}  (tol {TOL})  "
            f"iters run = {len(history)}")
        log(f"        loss: initial {initial_loss:.6e} -> best {best_loss:.6e} "
            f"({initial_loss / max(best_loss, 1e-30):.1f}x reduction)")
        log(f"        (final iterate: mass={final_mass:.6f}, "
            f"loss={final_loss:.6e} -- last oscillation sample, for reference)")
        log(f"        determinism (D1): {'PASS' if det_ok else 'FAIL'}")
        log("=" * 74)

        # 7) outputs (write log + csv unconditionally; plots only if mpl present)
        with open(log_path, "w") as f:
            f.write("\n".join(log_lines) + "\n")
        csv_path = os.path.join(RESULTS_DIR, "convergence.csv")
        write_csv(csv_path, history)
        plotted = write_plots(RESULTS_DIR, history)
        print(f"\nwrote: {log_path}")
        print(f"wrote: {csv_path}")
        if plotted:
            print(f"wrote: {os.path.join(RESULTS_DIR, 'loss_curve.png')}")
            print(f"wrote: {os.path.join(RESULTS_DIR, 'mass_curve.png')}")
        else:
            print("(matplotlib not available -- skipped PNGs; CSV is the source of truth)")

        # 8) hard asserts (do NOT fake -- if these fail the demo failed)
        assert rel_err < TOL, (
            f"recovered mass {recovered:.6f} is {rel_err:.4e} from true "
            f"{TRUE_MASS} -- exceeds {TOL} tolerance"
        )
        assert det_ok, "determinism (D1) violated: two runs gave different mass paths"
        print("\nSUCCESS: mass recovered within tolerance, deterministic.")
        return 0
    finally:
        dev.close()


if __name__ == "__main__":
    raise SystemExit(main())
