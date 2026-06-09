"""pytest: H1 reduced torque standing env contract for S4 phase 1.

Locks the train/deploy bridge contract before adding the env:
  * obs is the exact 32-wide H1 bridge layout from bridge_export.py;
  * action is 10-wide normalized torque, scaled per joint to the H1 MJCF limits;
  * reset seats the reduced H1 in the bent stance and returns finite CUDA obs;
  * rl_games registration exposes the single-env (32,) / (10,) spaces.

Single GPU only::

    export CUDA_VISIBLE_DEVICES=0
    python -m pytest python/tests/test_h1_stand_env.py -v
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import torch
from gymnasium import spaces

import nuka
from nuka.rl_games.vecenv import NukaVecEnv
from nuka.tasks.h1_stand import (
    H1_ACTION_DIM,
    H1_BLC,
    H1_OBS_DIM,
    LEG_TORQUE_LIMITS,
    STANCE_Q,
    make_env,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
REDUCED_MJCF = REPO / ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_reduced_train.xml"
N = 8


@pytest.fixture(scope="module")
def device():
    dev = nuka.Device.create(0)
    yield dev
    dev.close()


def test_reset_obs_matches_bridge_layout(device):
    with make_env(N, device=device, episode_length_s=0.25, seed=0) as env:
        assert env.obs_dim == H1_OBS_DIM == 32
        assert env.action_dim == H1_ACTION_DIM == 10
        assert env._world.base_link_count == H1_BLC == 11
        assert env._world.action_dim == H1_ACTION_DIM

        obs, info = env.reset()
        assert isinstance(info, dict)
        assert obs.shape == (N, H1_OBS_DIM)
        assert obs.is_cuda and obs.dtype == torch.float32
        assert torch.isfinite(obs).all()

        bp = torch.from_dlpack(env._world.buffer_view(nuka.BASE_POSE)).view(N, 7)
        lv = torch.from_dlpack(env._world.buffer_view(nuka.LINK_VELOCITY)).view(N, H1_BLC, 6)
        q = torch.from_dlpack(env._world.buffer_view(nuka.JOINT_POSITION)).view(N, H1_BLC)
        qd = torch.from_dlpack(env._world.buffer_view(nuka.JOINT_VELOCITY)).view(N, H1_BLC)

        expected = torch.zeros_like(obs)
        expected[:, 0:4] = bp[:, 3:7]
        expected[:, 4:10] = lv[:, 0, :]
        expected[:, 10] = bp[:, 0] - env.poly_center[0]
        expected[:, 11] = bp[:, 1] - env.poly_center[1]
        expected[:, 12:22] = q[:, 1:11]
        expected[:, 22:32] = qd[:, 1:11]

        max_err = float((obs - expected).abs().max())
        assert max_err < 1e-6, f"H1 obs drifted from bridge layout: max|diff|={max_err:.3e}"
        assert torch.allclose(q[:, 1:11], STANCE_Q.view(1, H1_ACTION_DIM), atol=1e-6)
        assert torch.allclose(qd[:, 1:11], torch.zeros_like(qd[:, 1:11]), atol=1e-6)


def test_action_space_and_torque_scaling_are_per_joint(device):
    with make_env(N, device=device, episode_length_s=0.25, seed=0) as env:
        assert isinstance(env.single_action_space, spaces.Box)
        assert env.single_action_space.shape == (H1_ACTION_DIM,)
        assert np.allclose(env.single_action_space.low, -1.0)
        assert np.allclose(env.single_action_space.high, 1.0)
        assert torch.equal(env.torque_limits, LEG_TORQUE_LIMITS)
        assert torch.equal(
            env.torque_limits,
            torch.tensor([200, 200, 200, 300, 40, 200, 200, 200, 300, 40], device="cuda", dtype=torch.float32),
        )

        env.reset()
        action = torch.tensor(
            [[-1.0, -0.5, 0.25, 1.0, -1.0, 1.0, 0.5, -0.25, -1.0, 1.0]],
            device="cuda",
            dtype=torch.float32,
        ).repeat(N, 1)
        env.step(action)
        tq = torch.from_dlpack(env._world.buffer_view(nuka.TORQUE_INPUT)).view(N, H1_BLC)
        expected = action * env.torque_limits.view(1, H1_ACTION_DIM)
        assert torch.allclose(tq[:, 0], torch.zeros(N, device="cuda"), atol=1e-6)
        assert torch.allclose(tq[:, 1:11], expected, atol=1e-5)


def test_random_rollout_autoresets_and_rewards_are_finite(device):
    with make_env(N, device=device, episode_length_s=0.08, seed=0) as env:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (N, H1_OBS_DIM)
        any_done = False
        for _ in range(8):
            obs, reward, terminated, truncated, info = env.step(env.sample_actions() * 0.1)
            any_done = any_done or bool((terminated | truncated).any())
            assert obs.shape == (N, H1_OBS_DIM) and obs.is_cuda
            assert reward.shape == (N,) and reward.is_cuda and reward.dtype == torch.float32
            assert terminated.shape == (N,) and terminated.dtype == torch.bool
            assert truncated.shape == (N,) and truncated.dtype == torch.bool
            assert torch.isfinite(obs).all()
            assert torch.isfinite(reward).all()
            assert isinstance(info, dict)
        assert any_done, "short episode did not exercise H1 autoreset path"


def test_reward_prefers_flat_sagittal_feet(device):
    with make_env(N, device=device, episode_length_s=0.25, seed=0) as env:
        env.reset()
        flat_reward = env.compute_reward().mean()
        q = torch.from_dlpack(env._world.buffer_view(nuka.JOINT_POSITION)).view(N, H1_BLC)
        q[:, 10] -= 0.10  # right foot pitch sum becomes toe-up instead of flat.
        toe_up_reward = env.compute_reward().mean()
        assert float(flat_reward - toe_up_reward) > 0.05


def test_default_stage_matches_explicit_no_disturbance(device):
    action = torch.zeros(N, H1_ACTION_DIM, device="cuda")
    with make_env(N, device=device, episode_length_s=0.25, seed=0) as env:
        env.reset(seed=0)
        default_obs, *_ = env.step(action)
    with make_env(
        N,
        device=device,
        episode_length_s=0.25,
        seed=0,
        disturbance_interval_steps=0,
        disturbance_linear_velocity=(0.25, -0.10, 0.0),
    ) as env:
        env.reset(seed=0)
        no_disturbance_obs, *_ = env.step(action)
    assert torch.allclose(default_obs, no_disturbance_obs, atol=1e-6)


def test_configured_push_disturbance_changes_root_linear_velocity(device):
    action = torch.zeros(N, H1_ACTION_DIM, device="cuda")
    with make_env(N, device=device, episode_length_s=0.25, seed=0) as env:
        env.reset(seed=0)
        baseline_obs, *_ = env.step(action)
    with make_env(
        N,
        device=device,
        episode_length_s=0.25,
        seed=0,
        disturbance_interval_steps=1,
        disturbance_linear_velocity=(0.25, -0.10, 0.0),
    ) as env:
        env.reset(seed=0)
        disturbed_obs, *_ = env.step(action)
    delta = disturbed_obs[:, 7:10] - baseline_obs[:, 7:10]
    expected = torch.tensor([0.25, -0.10, 0.0], device="cuda", dtype=torch.float32).view(1, 3)
    assert torch.allclose(delta, expected.repeat(N, 1), atol=1e-6)


def test_h1_rl_games_registration_and_vecenv_contract():
    import nuka.rl_games  # noqa: F401
    from rl_games.common import env_configurations as e

    assert "nuka_h1_stand" in e.configurations
    assert e.configurations["nuka_h1_stand"]["vecenv_type"] == "NUKA"

    ve = NukaVecEnv("nuka_h1_stand", N, episode_length_s=0.08, seed=0)
    try:
        info = ve.get_env_info()
        assert info["observation_space"].shape == (H1_OBS_DIM,)
        assert info["action_space"].shape == (H1_ACTION_DIM,)
        obs = ve.reset()
        assert obs.shape == (N, H1_OBS_DIM) and obs.is_cuda
        action = torch.zeros(N, H1_ACTION_DIM, device="cuda")
        out = ve.step(action)
        assert len(out) == 4
        obs, reward, dones, infos = out
        assert obs.shape == (N, H1_OBS_DIM)
        assert reward.shape == (N,)
        assert dones.shape == (N,) and dones.dtype == torch.float32
        assert "time_outs" in infos and infos["time_outs"].shape == (N,)
    finally:
        ve.close()


def test_bridge_obs_is_not_clipped(device):
    """The bridge obs is raw state, not Go2's clipped observation convention."""
    with make_env(N, device=device, episode_length_s=0.25, seed=0) as env:
        env.reset()
        qd = torch.from_dlpack(env._world.buffer_view(nuka.JOINT_VELOCITY)).view(N, H1_BLC)
        qd[:, 1] = 123.0
        obs = env.compute_obs()
        assert torch.equal(obs[:, 22], torch.full((N,), 123.0, device="cuda"))


def test_nonfinite_state_terminates_before_ppo_storage(device):
    """A numerically bad env must terminate instead of leaking NaN obs to PPO."""
    with make_env(N, device=device, episode_length_s=0.25, seed=0) as env:
        env.reset()
        qd = torch.from_dlpack(env._world.buffer_view(nuka.JOINT_VELOCITY)).view(N, H1_BLC)
        qd[0, 1] = float("nan")
        terminated = env.compute_terminated()
        assert bool(terminated[0]), "nonfinite qd must be treated as terminal"


def test_extreme_joint_velocity_terminates_before_overflow(device):
    """Runaway qvel is a terminal numerical-safety condition for PPO stability."""
    with make_env(N, device=device, episode_length_s=0.25, seed=0) as env:
        env.reset()
        qd = torch.from_dlpack(env._world.buffer_view(nuka.JOINT_VELOCITY)).view(N, H1_BLC)
        qd[0, 1] = 2500.0
        terminated = env.compute_terminated()
        assert bool(terminated[0]), "runaway qvel must reset before PPO sees overflow-scale obs"


def test_reduced_asset_exists_or_generator_is_available():
    assert REDUCED_MJCF.exists() or (REPO / "python/spikes/gen_h1_reduced_train.py").exists()
