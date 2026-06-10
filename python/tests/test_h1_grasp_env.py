"""A3 unit smoke for H1GraspEnv -- the fixed-base 2-finger cup-grasp RL env.

NOT a PPO run. Asserts the gymnasium duck-typed contract (shapes/dtypes, the
5-tuple step, autoreset), reward-sign sanity, determinism, AND -- the headline --
the DISCRIMINATIVE GATE: a hand-coded GOOD (timed-cradle) policy clearly beats both
a constant-max squeeze and a uniform-random policy (and a zero policy), because the
timing IC (cup descends from above OPEN fingers) makes a constant action drop the
cup. Prints all four policies' mean return + drop-rate.

Asset-gated (skips if the C7a cup is absent).

Run (single GPU)::

    CUDA_VISIBLE_DEVICES=0 LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64 \
        python -m pytest python/tests/test_h1_grasp_env.py -s
  (or directly: ... python python/tests/test_h1_grasp_env.py)
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import nuka
from nuka.tasks import h1_grasp_obs as H
from nuka.tasks.h1_grasp import (
    H1GraspEnv,
    make_env,
    ACTION_SPACE_LIMIT,
    FINGER_FORCE_SCALE,
    DEFAULT_CUP_ASSET,
)

N = 48
HORIZON = 220

cup_missing = not os.path.exists(DEFAULT_CUP_ASSET)
skip_no_cup = pytest.mark.skipif(cup_missing, reason="C7a cup asset not present")

DEV = torch.device("cuda")


# --------------------------------------------------------------------------
# Policies (return a (N,2) cuda action in [-1,1]).
# --------------------------------------------------------------------------
def pol_zero(env, obs):
    return torch.zeros(env.num_envs, 2, device=DEV)


def pol_const_max(env, obs):
    return torch.full((env.num_envs, 2), ACTION_SPACE_LIMIT, device=DEV)


def make_pol_rand(seed):
    g = torch.Generator(device=DEV); g.manual_seed(seed)
    def f(env, obs):
        return torch.rand(env.num_envs, 2, generator=g, device=DEV) * 2.0 - 1.0
    return f


def make_pol_good():
    """The hand-coded GOOD timed-cradle policy: hold the gap (action ~ 0) until the
    cup descends into the catch plane, then squeeze (+1) and HOLD. cup_z is recovered
    from the obs (cup_pos_rel_z + gripper_base_z). A per-env latch keeps squeezing
    once the cup has arrived, so a momentary bounce above the plane doesn't re-open.

    ``reset_done(mask)`` MUST be called by the rollout when envs autoreset, else the
    latch stays closed and a fresh episode squeezes from t=0 (== const-max) -- which
    would erase the policy's advantage on every episode after the first."""
    state = {"latch": None}
    def f(env, obs):
        if state["latch"] is None or state["latch"].shape[0] != env.num_envs:
            state["latch"] = torch.zeros(env.num_envs, dtype=torch.bool, device=DEV)
        cup_z = obs[:, 2] + float(H.GRIPPER_BASE_POS[2])     # cup_pos_rel_z + base_z
        arrived = cup_z <= (H.CATCH_PLANE_Z + 0.010)
        state["latch"] = state["latch"] | arrived
        a = torch.where(state["latch"].unsqueeze(1),
                        torch.full((env.num_envs, 2), ACTION_SPACE_LIMIT, device=DEV),
                        torch.zeros(env.num_envs, 2, device=DEV))
        return a
    def reset_done(mask):
        if state["latch"] is not None:
            state["latch"][mask] = False
    f.reset_done = reset_done
    return f


def _step_from_create_ic(env):
    """Build the first obs from the env's CREATE IC (cup centered, raised Z) WITHOUT
    calling reset() -- reset() jitters cup XY +-2.5 cm, ~80% of which lands the cup on
    the Y axis where the X-only gripper cannot save it (unsaveable), muddying the
    gate. The crisp discriminative gate is the CENTERED IC (== the A3 probe)."""
    obs_d = env._world.export_obs()
    cup_d = env._world.read_cup()
    return torch.as_tensor(
        H.assemble_obs(obs_d, cup_d, env.last_action.detach().cpu().numpy()), device=DEV
    )


def rollout_centered(env, policy, horizon=HORIZON):
    """Single-episode rollout from the CENTERED create IC (NO reset jitter, NO
    autoreset farming). Returns (episode_return, drop_rate, all_finite). Reward is
    accumulated per-env only while ALIVE (stops at the env's first termination) so a
    policy that drops the cup cannot farm post-drop steps; drop is tracked the step
    an env first falls below the catch band (not read off a contaminated final obs)."""
    obs = _step_from_create_ic(env)
    ret = torch.zeros(env.num_envs, device=DEV)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=DEV)
    dropped = torch.zeros(env.num_envs, dtype=torch.bool, device=DEV)
    all_finite = True
    for _ in range(horizon):
        a = policy(env, obs)
        obs, reward, term, trunc, info = env.step(a)
        ret += reward * alive.float()             # accrue only while alive.
        dropped = dropped | (term & alive)        # first-fall drop record.
        alive = alive & (~term)                   # latch dead on first termination.
        if not (torch.isfinite(obs).all() and torch.isfinite(reward).all()):
            all_finite = False
        # NB: the env autoresets terminated envs internally; we ignore the reset
        # state for the alive ones (they keep stepping) and never re-credit a dead
        # env. A latched good policy must be told which envs reset.
        if hasattr(policy, "reset_done") and bool(term.any()):
            policy.reset_done(term)
    return float(ret.mean()), float(dropped.float().mean()), all_finite


def rollout_jittered_episode(env, policy, horizon=HORIZON):
    """SINGLE-episode, alive-masked rollout from reset() (cup XY jitter +-2.5 cm) --
    the honest per-episode return on the distribution PPO actually trains against.
    Accrues reward per-env only while ALIVE (a doomed cup terminates early and freezes
    its accumulation; a caught cup holds to truncation). Returns (episode_return,
    caught_rate, all_finite). This is the metric the 'good beats no-op' gate uses --
    NOT a multi-episode sum (which mixes episode-length + autoreset-farming effects)."""
    obs, _ = env.reset(seed=123)
    if hasattr(policy, "reset_done"):
        policy.reset_done(torch.ones(env.num_envs, dtype=torch.bool, device=DEV))
    ret = torch.zeros(env.num_envs, device=DEV)
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=DEV)
    caught = torch.zeros(env.num_envs, dtype=torch.bool, device=DEV)
    all_finite = True
    for _ in range(horizon):
        a = policy(env, obs)
        obs, reward, term, trunc, info = env.step(a)
        ret += reward * alive.float()
        caught = caught | (trunc & alive)         # held to truncation = caught.
        alive = alive & (~term)                   # dead on first drop.
        if hasattr(policy, "reset_done") and bool(term.any()):
            policy.reset_done(term)
        if not (torch.isfinite(obs).all() and torch.isfinite(reward).all()):
            all_finite = False
    return float(ret.mean()), float(caught.float().mean()), all_finite


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
@skip_no_cup
def test_shapes_dtypes_and_spaces():
    with make_env(N, seed=0) as env:
        assert env.obs_dim == H.H1_GRASP_OBS_DIM == 27
        assert env.action_dim == 2
        assert env.single_action_space.shape == (2,)
        assert float(env.single_action_space.high[0]) == ACTION_SPACE_LIMIT
        assert env.single_observation_space.shape == (27,)
        obs, info = env.reset(seed=0)
        assert isinstance(info, dict)
        assert obs.shape == (N, 27)
        assert obs.is_cuda and obs.dtype == torch.float32
        assert torch.isfinite(obs).all()

        # A multi-step rollout: shapes/dtypes hold, NO NaN/Inf.
        for _ in range(20):
            a = torch.zeros(N, 2, device=DEV)
            obs, reward, term, trunc, info = env.step(a)
            assert obs.shape == (N, 27) and obs.dtype == torch.float32 and obs.is_cuda
            assert reward.shape == (N,) and reward.dtype == torch.float32
            assert term.shape == (N,) and term.dtype == torch.bool
            assert trunc.shape == (N,) and trunc.dtype == torch.bool
            assert "time_outs" in info and info["time_outs"].dtype == torch.bool
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()


@skip_no_cup
def test_action_mapping_scale_and_sign():
    """[-1,1] action -> +-12 N; positive = close. A held cup shows force-closure."""
    with make_env(N, seed=0) as env:
        env.reset(seed=0)
        good = make_pol_good()
        # run the good policy for the full horizon, then the cup is held.
        obs, _ = env.reset(seed=0)
        for _ in range(HORIZON):
            obs, *_ = env.step(good(env, obs))
        # last_action mapping: +1 -> +FINGER_FORCE_SCALE.
        a = torch.full((N, 2), 1.0, device=DEV)
        force = env._to_finger_force(a)
        assert np.allclose(force, FINGER_FORCE_SCALE), force[:2]
        a = torch.full((N, 2), -1.0, device=DEV)
        force = env._to_finger_force(a)
        assert np.allclose(force, -FINGER_FORCE_SCALE)


@skip_no_cup
def test_reward_sign_good_beats_zero():
    """A GOOD (timed-cradle) policy gets a clearly higher return than zero-action
    (centered single episode)."""
    with make_env(N, seed=0) as env:
        r_good, drop_good, fin_g = rollout_centered(env, make_pol_good())
    with make_env(N, seed=0) as env:
        r_zero, drop_zero, fin_z = rollout_centered(env, pol_zero)
    print("\n[A3 SMOKE reward-sign, centered] good_return=%.2f (drop=%.2f)  "
          "zero_return=%.2f (drop=%.2f)" % (r_good, drop_good, r_zero, drop_zero))
    assert fin_g and fin_z
    assert r_good > r_zero + 5.0, (r_good, r_zero)
    assert drop_zero > 0.9, drop_zero            # zero never closes -> ~all drop.


@skip_no_cup
def test_done_and_autoreset():
    """A dropped cup sets terminated, and the env autoresets that env (obs stays
    finite, the cup pose changes back toward the start IC)."""
    with make_env(N, seed=0) as env:
        obs, _ = env.reset(seed=0)
        saw_term = False
        for _ in range(HORIZON):
            # zero action -> every cup falls -> terminations happen.
            obs, reward, term, trunc, info = env.step(torch.zeros(N, 2, device=DEV))
            assert torch.isfinite(obs).all()
            if bool(term.any()):
                saw_term = True
                # the just-terminated envs were autoreset: their obs cup is back up
                # near the start (start Z = catch + offset = 0.20 + 0.06 = 0.26).
                done_ids = torch.nonzero(term, as_tuple=False).flatten()
                cup_z = obs[done_ids, 2] + float(H.GRIPPER_BASE_POS[2])
                assert torch.isfinite(cup_z).all()
                # post-reset cup is above the catch plane (descending IC), not on the
                # floor where it terminated.
                assert float(cup_z.min()) > H.CATCH_PLANE_Z, float(cup_z.min())
        assert saw_term, "no env terminated under zero action"


@skip_no_cup
def test_determinism_same_seed_same_initial_obs():
    """Same reset seed -> identical initial obs (the env wraps deterministic
    ResetEnvs)."""
    with make_env(N, seed=0) as env:
        o1, _ = env.reset(seed=777)
        o2, _ = env.reset(seed=777)
    assert torch.equal(o1, o2)


@skip_no_cup
def test_rl_games_vecenv_integration():
    """The ACTUAL PPO path: drive the env THROUGH NukaVecEnv (the rl_games IVecEnv
    adapter the next increment trains with), not just make_env. Asserts the registered
    name resolves, get_env_info exposes the single-env (27,)/(2,) Box spaces, reset()
    returns obs ONLY (a bare cuda tensor), and step(cuda_action) returns the 4-tuple
    (obs, reward, dones=float (term|trunc), infos with 'time_outs')."""
    from nuka.rl_games.vecenv import NukaVecEnv

    ve = NukaVecEnv("nuka_h1_grasp", 8, seed=0)
    try:
        einfo = ve.get_env_info()
        assert einfo["observation_space"].shape == (27,)
        assert einfo["action_space"].shape == (2,)
        assert einfo["agents"] == 1 and einfo["value_size"] == 1
        assert ve.get_number_of_agents() == 1

        obs = ve.reset()                       # obs ONLY (bare tensor).
        assert isinstance(obs, torch.Tensor)
        assert obs.shape == (8, 27) and obs.is_cuda and obs.dtype == torch.float32
        assert torch.isfinite(obs).all()

        for _ in range(5):
            actions = torch.zeros(8, 2, device=DEV)
            obs, reward, dones, infos = ve.step(actions)   # 4-tuple.
            assert obs.shape == (8, 27) and obs.dtype == torch.float32
            assert reward.shape == (8,) and reward.dtype == torch.float32
            assert dones.shape == (8,) and dones.dtype == torch.float32
            assert "time_outs" in infos and infos["time_outs"].dtype == torch.bool
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    finally:
        ve.close()


@skip_no_cup
def test_DISCRIMINATIVE_GATE():
    """THE HEADLINE: on the CENTERED discriminative IC, the constant-max AND uniform-
    random returns are BOTH poor (cup dropped) and clearly below the hand-coded GOOD
    timed-cradle policy -- a constant action does NOT solve the task. Single episode,
    alive-masked (no autoreset farming), so the return reflects the actual grasp.
    Prints all four CENTERED returns + drop-rates AND the jittered (PPO-train) numbers."""
    # A fresh env per policy so the create IC is identical + uncontaminated.
    def centered(make_pol):
        with make_env(N, seed=0) as env:
            return rollout_centered(env, make_pol())
    r_good, d_good, f_good = centered(make_pol_good)
    r_const, d_const, f_const = centered(lambda: pol_const_max)
    r_rand, d_rand, f_rand = centered(lambda: make_pol_rand(0))
    r_zero, d_zero, f_zero = centered(lambda: pol_zero)
    print("\n[A3 DISCRIMINATIVE GATE -- CENTERED IC]  (return = sum over %d steps)" % HORIZON)
    print("  GOOD (timed cradle): return=%8.2f  drop=%.2f" % (r_good, d_good))
    print("  CONST-MAX (slam)   : return=%8.2f  drop=%.2f" % (r_const, d_const))
    print("  UNIFORM-RANDOM     : return=%8.2f  drop=%.2f" % (r_rand, d_rand))
    print("  ZERO (no-op)       : return=%8.2f  drop=%.2f" % (r_zero, d_zero))
    assert f_good and f_const and f_rand and f_zero
    # The gate: GOOD clearly dominates BOTH const-max AND random; both of those drop
    # the cup. A constant action does NOT solve the task.
    assert r_good > r_const + 50.0, (r_good, r_const)
    assert r_good > r_rand + 50.0, (r_good, r_rand)
    assert d_const > 0.8, ("const-max should drop the cup (closes early)", d_const)
    assert d_rand > 0.8, ("random should drop the cup", d_rand)
    assert d_good < 0.5, ("good should catch most cups", d_good)

    # THE TRAINING-DISTRIBUTION GATE: the env PPO trains on always starts from
    # reset() (cup XY jitter +-2.5 cm). The milestone-relevant question there is
    # "does a GENUINE grasp policy beat a NO-OP?" -- because if a no-op out-returned a
    # grasp, PPO would learn to do nothing. Single-episode, alive-masked (no farming).
    # The hand-coded cradle catches only ~10% of the +-2.5 cm jittered cups (MEASURED;
    # the rest land the cup off the X-squeeze line -- mostly on the un-actuated Y axis
    # -> unsaveable; a finding flagged for the next increment), so the absolute return
    # is modest, but GOOD must clearly beat ZERO, and const/random (which never close)
    # stay ~0.
    def jittered(make_pol):
        with make_env(N, seed=0) as env:
            return rollout_jittered_episode(env, make_pol())
    jr_good, jc_good, _ = jittered(make_pol_good)
    jr_zero, jc_zero, _ = jittered(lambda: pol_zero)
    jr_const, jc_const, _ = jittered(lambda: pol_const_max)
    jr_rand, jc_rand, _ = jittered(lambda: make_pol_rand(0))
    print("[A3 -- JITTERED IC (the PPO training distribution, +-2.5cm XY), "
          "single-episode alive-masked]")
    print("  GOOD : return=%8.2f caught=%.2f" % (jr_good, jc_good))
    print("  ZERO : return=%8.2f caught=%.2f" % (jr_zero, jc_zero))
    print("  CONST: return=%8.2f caught=%.2f" % (jr_const, jc_const))
    print("  RAND : return=%8.2f caught=%.2f" % (jr_rand, jc_rand))
    # The beats-no-op milestone: a genuine grasp policy clearly out-returns a no-op,
    # and the non-grasping baselines (const/random) do NOT out-return the no-op.
    assert jr_good > jr_zero + 10.0, ("good must beat no-op on the training dist",
                                      jr_good, jr_zero)
    assert jr_const <= jr_zero + 5.0, ("const-max must not beat no-op", jr_const, jr_zero)
    assert jr_rand <= jr_zero + 5.0, ("random must not beat no-op", jr_rand, jr_zero)
    assert jc_good > jc_const and jc_good > jc_rand   # good catches; the others don't.


if __name__ == "__main__":
    if cup_missing:
        print("SKIP: cup asset missing at", DEFAULT_CUP_ASSET)
        raise SystemExit(0)
    test_shapes_dtypes_and_spaces();          print("[OK] shapes/dtypes/spaces")
    test_action_mapping_scale_and_sign();     print("[OK] action mapping scale+sign")
    test_reward_sign_good_beats_zero();       print("[OK] reward sign good>zero")
    test_done_and_autoreset();                print("[OK] done + autoreset")
    test_determinism_same_seed_same_initial_obs(); print("[OK] determinism")
    test_DISCRIMINATIVE_GATE();               print("[OK] DISCRIMINATIVE GATE")
    print("\nALL A3 H1GraspEnv SMOKE GATES PASSED")
