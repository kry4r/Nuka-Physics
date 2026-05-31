"""pytest: p04 regression guards locking the exit-#3 convergence FIX.

These guard the full Go2 PPO action pipeline against the SPECIFIC regression that
collapsed the first 1500-epoch run (reward==0, ep_len~22, value-head collapse).

THE BUG (fixed in commit ``p04 #3``): the gym action space was declared at +-100
(``go2_obs.ACTION_CLIP``). rl_games runs with ``clip_actions=True`` (the default)
and RESCALES the policy's [-1,1] output onto the *declared action-space bounds*
(``rl_games.common.a2c_common.rescale_actions``: ``a*(high-low)/2 + (high+low)/2``).
So a +-100 space was a **100x amplification** of every applied action (measured
``mean|a|=64``) -> ``target_q = default + 0.25*64 = default + 16 rad`` -> the joints
slam to the force limit -> the robot tips in ~21 steps -> every reward step is
all-negative -> ``only_positive_rewards`` zeros it -> the value head collapses ->
PPO dies. FIX: declare the action space at +-1 (``ACTION_SPACE_LIMIT=1.0``) so the
rl_games rescale is the IDENTITY and the env receives the policy's true O(1) action.

The three guards below pin (1) the declared bound, (2) the rl_games rescale being
the identity for that bound -- using rl_games' OWN function, with a +-100 positive
control proving the guard bites -- and (3) the end-to-end applied PD target being at
the legged_gym ACTION_SCALE (0.25 rad/unit), not amplified.

Single GPU only::

    export CUDA_VISIBLE_DEVICES=0
    python -m pytest python/tests/test_ppo_loop_regression.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from rl_games.common.a2c_common import rescale_actions

import nuka
from nuka.tasks.go2_locomotion import make_env
from nuka.tasks import go2_obs as G


N = 8  # tiny, single-GPU-friendly


@pytest.fixture(scope="module")
def env():
    e = make_env(N, command=[0.5, 0.0, 0.0], episode_length_s=1.0, seed=0)
    yield e
    e.close()


def test_action_space_is_unit_box(env):
    """REGRESSION GUARD for the +-100 action-space bug. rl_games rescales the
    policy's [-1,1] output onto these bounds, so they MUST be +-1 (-> identity
    rescale), NOT +-ACTION_CLIP (100, a 100x amplification that collapsed PPO)."""
    for name, sp in (("single", env.single_action_space), ("batched", env.action_space)):
        assert np.allclose(sp.high, 1.0), f"{name} action_space.high must be +1, got max {sp.high.max()}"
        assert np.allclose(sp.low, -1.0), f"{name} action_space.low must be -1, got min {sp.low.min()}"
    # The declared limit is the unit constant; ACTION_CLIP stays a looser, separate
    # internal safety clamp (it must NOT be what defines the gym space again).
    assert G.ACTION_SPACE_LIMIT == 1.0
    assert G.ACTION_CLIP != G.ACTION_SPACE_LIMIT


def test_rl_games_rescale_is_identity_for_declared_space(env):
    """Using rl_games' OWN ``rescale_actions`` with the bounds it reads from the
    env's action_space, a policy sample in [-1,1] must rescale to the IDENTITY.
    The +-100 positive control proves the guard discriminates (would be 100x)."""
    a = torch.tensor([-1.0, -0.5, 0.0, 0.6, 1.0])
    lo = torch.as_tensor(np.asarray(env.single_action_space.low).reshape(-1)[:5]).float()   # all -1
    hi = torch.as_tensor(np.asarray(env.single_action_space.high).reshape(-1)[:5]).float()  # all +1
    out = rescale_actions(lo, hi, a)
    assert torch.allclose(out, a, atol=1e-6), f"declared space must rescale to identity, got {out.tolist()}"
    # Positive control: the OLD +-100 space would have 100x-amplified -> proves the
    # assertion above is meaningful (not vacuously true for any bound).
    bad = rescale_actions(torch.full_like(a, -100.0), torch.full_like(a, 100.0), a)
    assert torch.allclose(bad, a * 100.0), "control: a +-100 space MUST 100x (the guard bites)"


def test_end_to_end_applied_target_not_amplified(env):
    """Full pipeline guard: a unit policy action, pushed through rl_games' rescale
    with the env's DECLARED bounds, then ``env.step``, must land a PD target at the
    legged_gym ACTION_SCALE (``|target - default| == 0.25`` rad for action=+1), NOT
    the ~25 rad the +-100 bug produced. Reads DRIVE_TARGET directly (the commanded
    target persists through the decimation sub-steps)."""
    env.command[:] = torch.tensor([0.5, 0.0, 0.0], device="cuda")
    env.reset()
    policy_out = torch.ones(N, G.GO2_ACTION_DIM)  # the policy's max [-1,1] output
    lo = torch.as_tensor(np.asarray(env.single_action_space.low)).float()
    hi = torch.as_tensor(np.asarray(env.single_action_space.high)).float()
    applied = rescale_actions(lo, hi, policy_out).to("cuda")  # identity for +-1 space
    env.step(applied)

    tgt = torch.from_dlpack(env._world.buffer_view(nuka.DRIVE_TARGET))  # (N, 13)
    default_nuka = env._obs.default_angles.index_select(0, env._obs.nuka_slot_for_urdf)
    delta = (tgt[:, 1:G.GO2_BLC] - default_nuka).abs()
    # action=+1 -> target = default + ACTION_SCALE*1 = default + 0.25 (every joint).
    assert torch.allclose(delta, torch.full_like(delta, G.ACTION_SCALE), atol=1e-4), (
        f"applied PD target delta must be ACTION_SCALE={G.ACTION_SCALE} rad, got "
        f"max {float(delta.max()):.4f} (a value ~25 would mean the +-100 rescale bug regressed)"
    )
