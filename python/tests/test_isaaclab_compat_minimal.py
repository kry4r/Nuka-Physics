"""pytest: nuka.isaaclab_compat -- minimal ManagerBasedRLEnv (p03-T2).

Builds a minimal manager-based env with a couple of obs terms + reward terms +
a termination term + a PD action term (driving the Go2 via the on-GPU obs
builder), then reset() + 10 step()s, asserting obs / reward / terminated /
truncated shapes are cuda + finite. This exercises the manager dataclass term
pattern end-to-end (obs concat, weighted reward sum, terminated/truncated split,
action -> DRIVE_TARGET) on the real engine.

Single GPU only:
    export CUDA_VISIBLE_DEVICES=0
    python -m pytest python/tests/test_isaaclab_compat_minimal.py -v
"""

from __future__ import annotations

import pytest
import torch

import nuka
from nuka.isaaclab_compat import (
    ManagerBasedRLEnv,
    ManagerBasedRLEnvCfg,
    ObservationTerm,
    RewardTerm,
    TerminationTerm,
    ActionTerm,
)
from nuka.tasks.go2_obs import Go2ObsBuilder

SCENE = "/root/Nuka-Physics/examples/scenes/go2_float.usda"


@pytest.fixture(scope="module")
def device():
    dev = nuka.Device.create(0)
    yield dev
    dev.close()


# ---- term functions (Isaac-Lab-style signatures: func(env) / func(env, act)) --
def _builder(env):
    # Lazily attach a Go2ObsBuilder to the env (constructed once).
    b = getattr(env, "_go2_builder", None)
    if b is None:
        b = Go2ObsBuilder(env.sim.device, env.world)
        b.apply_pd_gains()
        env._go2_builder = b
    return b


def obs_base_lin_vel(env):
    return _builder(env).base_lin_vel()              # (N,3)


def obs_projected_gravity(env):
    return _builder(env).projected_gravity()         # (N,3)


def obs_joint_pos_rel(env):
    b = _builder(env)
    return b.q_urdf() - b.default_angles             # (N,12)


def reward_alive(env):
    return torch.ones(env.num_envs, device="cuda")


def reward_track_lin_vel(env):
    b = _builder(env)
    cmd = env.command_manager.command_velocity[:, :2]
    actual = b.base_lin_vel()[:, :2]
    err = (cmd - actual).pow(2).sum(dim=-1)
    return torch.exp(-err / 0.25)


def terminated_fell(env):
    return _builder(env).base_pos()[:, 2] < 0.18     # (N,) bool-ish


def action_pd(env, action):
    return _builder(env).write_action(action)        # writes DRIVE_TARGET, returns clipped


GO2_MINIMAL_CFG = ManagerBasedRLEnvCfg(
    scene_path=SCENE,
    num_envs=32,
    sim_dt=0.005,
    decimation=4,
    episode_length_s=2.0,
    command=(0.5, 0.0, 0.0),
    obs_terms=[
        ObservationTerm("base_lin_vel", obs_base_lin_vel),
        ObservationTerm("projected_gravity", obs_projected_gravity),
        ObservationTerm("joint_pos_rel", obs_joint_pos_rel),
    ],
    reward_terms=[
        RewardTerm("track_lin_vel", reward_track_lin_vel, weight=1.0),
        RewardTerm("alive", reward_alive, weight=0.1),
    ],
    termination_terms=[
        TerminationTerm("fell", terminated_fell),
    ],
    action_term=ActionTerm("pd", action_pd),
)

# obs dim = 3 (lin vel) + 3 (gravity) + 12 (joint pos) = 18.
EXPECTED_OBS_DIM = 18


def test_minimal_isaaclab_env_runs(device):
    env = ManagerBasedRLEnv(GO2_MINIMAL_CFG, device=device)
    try:
        obs, info = env.reset()
        assert obs.shape == (32, EXPECTED_OBS_DIM)
        assert obs.is_cuda and torch.isfinite(obs).all()
        assert isinstance(info, dict)

        for _ in range(10):
            actions = torch.zeros(env.num_envs, env.world.action_dim, device="cuda")
            obs, reward, terminated, truncated, info = env.step(actions)
            assert obs.shape == (32, EXPECTED_OBS_DIM) and obs.is_cuda
            assert reward.shape == (32,) and reward.is_cuda
            assert terminated.shape == (32,) and terminated.dtype == torch.bool
            assert truncated.shape == (32,) and truncated.dtype == torch.bool
            assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
    finally:
        env.close()


def test_manager_compose_shapes(device):
    """Direct manager checks: obs concat width, weighted reward sum, and the
    terminated/truncated split from the TerminationManager."""
    env = ManagerBasedRLEnv(GO2_MINIMAL_CFG, device=device)
    try:
        env.reset()
        env.sim.step(4)
        # ObservationManager concatenation width.
        obs = env.observation_manager.compute()
        assert obs.shape == (32, EXPECTED_OBS_DIM)
        # RewardManager weighted sum is finite (num_envs,).
        rew = env.reward_manager.compute()
        assert rew.shape == (32,) and torch.isfinite(rew).all()
        # TerminationManager returns a (terminated, truncated) bool split.
        term, trunc = env.termination_manager.compute()
        assert term.shape == (32,) and term.dtype == torch.bool
        assert trunc.shape == (32,) and trunc.dtype == torch.bool
    finally:
        env.close()
