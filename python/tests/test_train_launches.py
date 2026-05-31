"""pytest: the rl_games (1.6.5) integration CONTRACT for the Nuka Go2 env (p03).

This is the LIGHT, fast, non-flaky half of the p03 RL-integration gate: it builds
the env THROUGH the rl_games IVecEnv adapter and asserts the exact contract
``rl_games.common.a2c_common`` relies on -- get_env_info (single-env spaces),
reset (bare cuda tensor), step (4-tuple obs/reward/dones/infos with the right
shapes + dtypes), and ``infos['time_outs']``. It does NOT launch a full
``Runner`` (that's heavy/flaky for pytest -- it lives in the standalone
``examples/training/train_go2_ppo.py --smoke`` which the controller runs for the
verbatim "training launches + >=2 epochs reward/loss" proof).

Single GPU only::

    export CUDA_VISIBLE_DEVICES=0
    python -m pytest python/tests/test_train_launches.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import spaces

from nuka.rl_games.vecenv import NukaVecEnv
from nuka.tasks import go2_obs as G


N = 64  # small, single-GPU-friendly


@pytest.fixture(scope="module")
def vecenv():
    """One NukaVecEnv (owns its nuka.Device) for the module."""
    ve = NukaVecEnv("nuka_go2", N, command=[0.5, 0.0, 0.0],
                    episode_length_s=1.0, seed=0)
    yield ve
    ve.close()


# ---------------------------------------------------------------------------
# registration: importing nuka.rl_games makes 'nuka_go2' resolvable
# ---------------------------------------------------------------------------
def test_registration():
    import nuka.rl_games  # noqa: F401
    from rl_games.common import vecenv as v
    from rl_games.common import env_configurations as e
    assert "NUKA" in v.vecenv_config
    assert "nuka_go2" in e.configurations
    assert e.configurations["nuka_go2"]["vecenv_type"] == "NUKA"


# ---------------------------------------------------------------------------
# get_env_info: SINGLE-env spaces (rl_games prepends num_actors itself)
# ---------------------------------------------------------------------------
def test_get_env_info_single_env_spaces(vecenv):
    info = vecenv.get_env_info()
    assert info["agents"] == 1
    assert info["value_size"] == 1
    assert isinstance(info["observation_space"], spaces.Box)
    assert isinstance(info["action_space"], spaces.Box)
    # MUST be the single-env shapes, not (num_envs, ...).
    assert info["observation_space"].shape == (G.GO2_OBS_DIM,)
    assert info["action_space"].shape == (G.GO2_ACTION_DIM,)
    assert vecenv.get_number_of_agents() == 1


# ---------------------------------------------------------------------------
# reset: bare cuda float32 obs tensor (N, 48) (rl_games wraps to {'obs': ...})
# ---------------------------------------------------------------------------
def test_reset_returns_bare_cuda_tensor(vecenv):
    obs = vecenv.reset()
    assert isinstance(obs, torch.Tensor)
    assert obs.shape == (N, G.GO2_OBS_DIM)
    assert obs.is_cuda and obs.dtype == torch.float32
    assert torch.isfinite(obs).all()


# ---------------------------------------------------------------------------
# step: 4-tuple, merged single-dones, time_outs -- the load-bearing contract
# ---------------------------------------------------------------------------
def test_step_four_tuple_contract(vecenv):
    vecenv.reset()
    for _ in range(20):
        # rl_games hands a cuda action tensor -> pass straight through.
        actions = torch.randn(N, G.GO2_ACTION_DIM, device="cuda")
        out = vecenv.step(actions)
        assert len(out) == 4, "IVecEnv.step must return a 4-tuple"
        obs, reward, dones, infos = out

        assert obs.shape == (N, G.GO2_OBS_DIM) and obs.is_cuda
        assert obs.dtype == torch.float32
        assert torch.isfinite(obs).all(), "obs has NaN/Inf"

        # rewards: (N,) cuda float32 (rl_games does rewards.unsqueeze(1)).
        assert reward.shape == (N,) and reward.is_cuda
        assert reward.dtype == torch.float32
        assert torch.isfinite(reward).all(), "reward has NaN/Inf"

        # dones: SINGLE merged done, (N,) cuda float32 (terminated|truncated).
        assert dones.shape == (N,) and dones.is_cuda
        assert dones.dtype == torch.float32
        assert torch.isfinite(dones).all()
        assert ((dones == 0) | (dones == 1)).all(), "dones not 0/1"

        # infos['time_outs']: (N,) bool, for value-bootstrap.
        assert "time_outs" in infos
        to = infos["time_outs"]
        assert to.shape == (N,) and to.dtype == torch.bool


# ---------------------------------------------------------------------------
# autoreset exercised through the adapter: force a done, env keeps stepping
# ---------------------------------------------------------------------------
def test_dones_fire_and_env_survives(vecenv):
    """Short episodes (1.0s) guarantee truncations fire within a rollout; the
    adapter must keep returning valid obs across the autoreset (no crash, finite,
    correct shape) -- exactly what rl_games needs for a stable rollout."""
    vecenv.reset()
    any_done = False
    for _ in range(60):  # 60 control steps > 50-step episode -> at least one done
        actions = torch.zeros(N, G.GO2_ACTION_DIM, device="cuda")
        obs, reward, dones, infos = vecenv.step(actions)
        any_done = any_done or bool(dones.any())
        assert torch.isfinite(obs).all() and obs.shape == (N, G.GO2_OBS_DIM)
    assert any_done, "no done fired in 60 steps -- autoreset path not exercised"
