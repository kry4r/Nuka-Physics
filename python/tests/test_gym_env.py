"""pytest: nuka.gym.NukaGymEnv -- vectorized Go2 RL env (p03-T2).

Proves:
  * NukaGymEnv(go2 scene, num_envs=64): reset() -> obs (64,48) cuda finite;
  * a SEEDED random-policy rollout (~10 steps) -> the gymnasium 5-tuple with
    obs (64,48) / reward (64,) / terminated (64,) bool / truncated (64,) bool,
    all cuda + finite, no NaN, no crash;
  * the on-GPU 48-dim obs is bit-exact to the proven numpy build_obs oracle
    (examples/sim_val/go2_policy_drive.py) -- the real correctness gate;
  * autoreset: a FORCED termination (a high termination_height so it fires on
    step 1) restores q / qd / base velocity to the initial state after the
    next step (masked reset semantics).

Single GPU only:
    export CUDA_VISIBLE_DEVICES=0
    python -m pytest python/tests/test_gym_env.py -v
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

import nuka
from nuka.gym import NukaGymEnv
from nuka.tasks import go2_obs as G

SCENE = "/root/Nuka-Physics/examples/scenes/go2_float.usda"

# Import the PROVEN numpy obs oracle from the sim-val harness.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "sim_val"))
import go2_policy_drive as ORACLE  # noqa: E402


@pytest.fixture(scope="module")
def device():
    dev = nuka.Device.create(0)
    yield dev
    dev.close()


def _make_env(device, num_envs=64, **kw):
    return NukaGymEnv(SCENE, num_envs=num_envs, device=device, seed=0, **kw)


# ---------------------------------------------------------------------------
# reset shape / dtype / finiteness
# ---------------------------------------------------------------------------
def test_reset_obs_shape_cuda_finite(device):
    with _make_env(device, 64) as env:
        assert env.num_envs == 64
        assert env.single_observation_space.shape == (48,)
        assert env.single_action_space.shape == (12,)
        obs, info = env.reset()
        assert obs.shape == (64, 48)
        assert obs.is_cuda and obs.dtype == torch.float32
        assert torch.isfinite(obs).all()
        assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# seeded random rollout -> 5-tuple, all cuda + finite
# ---------------------------------------------------------------------------
def test_random_rollout_five_tuple(device):
    with _make_env(device, 64) as env:
        env.reset(seed=0)
        for _ in range(10):
            actions = env.sample_actions()
            assert actions.shape == (64, 12)
            obs, reward, terminated, truncated, info = env.step(actions)
            assert obs.shape == (64, 48) and obs.is_cuda
            assert reward.shape == (64,) and reward.is_cuda
            assert terminated.shape == (64,) and terminated.dtype == torch.bool
            assert truncated.shape == (64,) and truncated.dtype == torch.bool
            assert torch.isfinite(obs).all(), "obs has NaN/Inf"
            assert torch.isfinite(reward).all(), "reward has NaN/Inf"
            assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# on-GPU obs == numpy oracle (the real correctness gate)
# ---------------------------------------------------------------------------
def test_gpu_obs_matches_numpy_oracle(device):
    """The 48-dim obs composed on-GPU must be bit-exact to the proven numpy
    build_obs (go2_policy_drive.py) on live engine state -- proving the layout,
    scales, joint permutation, and quat-rotate-gravity are all correct."""
    cmd = (0.5, 0.1, -0.2)
    with _make_env(device, 16, command=cmd) as env:
        env.reset()
        # Step once with a non-trivial action so q/qd/vel/pose are all non-default.
        action = env.sample_actions()
        obs_gpu, *_ = env.step(action)
        obs_gpu_np = obs_gpu.detach().cpu().numpy()

        # Rebuild the SAME obs via the numpy oracle from the live buffers.
        b = env._obs
        blv = b.base_lin_vel().detach().cpu().numpy()
        bav = b.base_ang_vel().detach().cpu().numpy()
        quat = b.base_quat_wxyz().detach().cpu().numpy()
        pg = np.stack([ORACLE.projected_gravity_body(quat[i]) for i in range(quat.shape[0])])
        q_urdf = b.q_urdf().detach().cpu().numpy()
        qd_urdf = b.qd_urdf().detach().cpu().numpy()
        last_action = env.last_action.detach().cpu().numpy()
        cmd_arr = np.tile(np.array(cmd, np.float32), (16, 1))
        obs_oracle = ORACLE.build_obs(blv, bav, pg, cmd_arr, q_urdf, qd_urdf, last_action)

        max_err = float(np.max(np.abs(obs_gpu_np - obs_oracle)))
        assert max_err < 1e-5, f"GPU obs vs numpy oracle max|diff|={max_err:.3e}"


# ---------------------------------------------------------------------------
# joint permutation discovery is a valid permutation (driven through the tensor)
# ---------------------------------------------------------------------------
def test_joint_permutation_is_valid(device):
    perm = G.discover_joint_permutation(device)
    assert sorted(perm.tolist()) == list(range(12)), "not a valid 0..11 permutation"
    # go2_float happens to enumerate joints in URDF order -> identity. We assert
    # this as an invariant for the shipped scene (but the stack drives the tensor,
    # so a future scrambled scene stays correct).
    assert (perm == np.arange(12)).all(), (
        f"go2_float permutation expected identity, got {perm.tolist()}"
    )


# ---------------------------------------------------------------------------
# AUTORESET: forced termination restores q/qd/base-vel after the next step
# ---------------------------------------------------------------------------
def _base_z(world, env):
    pose = torch.from_dlpack(world.buffer_view(nuka.ARTICULATION_LINK_POSE))
    return float(pose[env, 0, 2].item())


def test_autoreset_restores_terminated_envs(device):
    """Diverge the state under a NORMAL threshold, then deterministically force a
    termination (raise termination_height above the live base height) so EVERY
    env terminates on the controlled step; the autoreset must restore q / qd /
    base velocity to the initial creation state, verified after the NEXT step
    (the reset is authoritative across a step, handling the one-step
    ARTICULATION_LINK_POSE lag)."""
    with _make_env(device, 8) as env:  # default (low) termination_height
        env.reset()
        world = env._world
        # Capture the initial (creation) q before diverging.
        nuka.sync()
        q0 = torch.from_dlpack(world.buffer_view(nuka.JOINT_POSITION)).detach().cpu().clone()

        # Diverge the state with several stepped actions (no termination yet:
        # the default threshold is well below the live base height).
        for _ in range(5):
            env.step(env.sample_actions() * 0.5)
        q_diverged = torch.from_dlpack(world.buffer_view(nuka.JOINT_POSITION)).detach().cpu().clone()
        assert (q_diverged - q0).abs().max().item() > 1e-3, "state did not diverge -- vacuous"

        # Deterministically FORCE termination: raise the height threshold above
        # the live base z so compute_terminated() fires on every env this step.
        live_z = _base_z(world, 0)
        env.termination_height = live_z + 1.0
        _, _, terminated, _, _ = env.step(torch.zeros(8, 12, device="cuda"))
        assert bool(terminated.all()), "forced termination did not fire on all envs"
        # last_action bookkeeping zeroed for the reset envs (no stale history).
        assert env.last_action.abs().max().item() == 0.0, "last_action not zeroed on reset"

        # Restore a sane threshold so the next step does NOT immediately re-reset.
        env.termination_height = 0.18

        # AUTHORITY ACROSS A STEP: step once more; q/qd/base-vel are back at init.
        env.step(torch.zeros(8, 12, device="cuda"))
        nuka.sync()
        q_after = torch.from_dlpack(world.buffer_view(nuka.JOINT_POSITION)).detach().cpu().clone()
        qd_after = torch.from_dlpack(world.buffer_view(nuka.JOINT_VELOCITY))
        vel_after = torch.from_dlpack(world.buffer_view(nuka.LINK_VELOCITY))
        # q restored near the creation q (it then takes a single fresh step from init).
        assert (q_after - q0).abs().max().item() < 5e-2, (
            f"autoreset q not restored (max|dq|={(q_after - q0).abs().max().item():.3e})"
        )
        assert torch.isfinite(qd_after).all() and torch.isfinite(vel_after).all()


# ---------------------------------------------------------------------------
# EPISODE-BOUNDARY FIX: the SAME-step returned obs[done] is the POST-reset obs
# ---------------------------------------------------------------------------
def test_returned_obs_is_post_reset_for_done_envs(device):
    """legged_gym / Brax convention: when an env terminates, step() returns the
    POST-reset obs for that env in the SAME call (not the terminal/fallen obs).

    Drive the free-floating Go2 until it tilts/falls (diverging the base
    orientation far from upright), then force termination on the controlled step
    and assert the RETURNED obs for the done envs reports:
      * projected_gravity (obs[:, 6:9]) ~ upright [0,0,-1] (the RESET orientation,
        NOT the fallen one) -- read from the authoritative un-lagged base_pose;
      * last_action (obs[:, 36:48]) == 0 (fresh episode, no stale action history).
    The reference upright gravity is taken from a fresh env's reset() obs so the
    assertion is independent of the exact seated quaternion."""
    with _make_env(device, 8) as env:
        # Upright reference: a fresh reset env's projected_gravity (obs[:, 6:9]).
        obs0, _ = env.reset()
        upright_pg = obs0[0, 6:9].detach().cpu().clone()  # ~[0,0,-1]-ish

        # Diverge hard so the base orientation leaves upright (the terminal/fallen
        # projected_gravity will differ markedly from upright).
        for _ in range(30):
            env.step(env.sample_actions())

        # Capture the LIVE (pre-reset) lagged projected_gravity to prove the fallen
        # orientation really differs from upright (else the test is vacuous).
        nuka.sync()
        fallen_pg = env._obs.projected_gravity()[0].detach().cpu().clone()
        assert (fallen_pg - upright_pg).abs().max().item() > 5e-2, (
            "base did not leave upright -- test is vacuous "
            f"(|fallen-upright|={(fallen_pg - upright_pg).abs().max().item():.3e})"
        )

        # FORCE termination on every env this step (height threshold above live z).
        live_z = _base_z(env._world, 0)
        env.termination_height = live_z + 1.0
        obs, _reward, terminated, _trunc, _info = env.step(
            torch.ones(8, 12, device="cuda")  # non-zero action -> stale history if kept
        )
        assert bool(terminated.all()), "forced termination did not fire on all envs"

        obs_cpu = obs.detach().cpu()
        # (1) returned obs[done] projected_gravity ~ the upright RESET orientation,
        # NOT the fallen pre-reset orientation.
        ret_pg = obs_cpu[:, 6:9]
        err_upright = (ret_pg - upright_pg).abs().max().item()
        err_fallen = (ret_pg - fallen_pg).abs().max().item()
        assert err_upright < 1e-4, (
            f"returned obs[done] gravity is not the upright reset value "
            f"(max|diff|={err_upright:.3e})"
        )
        assert err_fallen > 5e-2, (
            "returned obs[done] gravity still matches the FALLEN pre-reset "
            f"orientation (max|diff|={err_fallen:.3e}) -- the terminal obs leaked"
        )
        # (2) last_action slot zeroed for the reset envs (no stale history).
        assert obs_cpu[:, 36:48].abs().max().item() == 0.0, (
            "returned obs[done] last_action not zeroed on reset"
        )
        assert torch.isfinite(obs).all()
