"""Go2 flat-ground locomotion task (p03 RL-integration target).

A thin :class:`~nuka.gym.env.NukaGymEnv` subclass that supplies the locomotion
*reward* on top of the on-GPU 48-dim obs + 12-DOF PD action contract already
shipped in :mod:`nuka.tasks.go2_obs` / :mod:`nuka.gym.env`. The obs math, the
joint-order permutation, and the height/tilt termination all live in the base
env (and the builder); this layer ONLY adds reward shaping + a ``make_env``
factory so the RL stack (``nuka.rl_games`` adapter + ``train_go2_ppo.py``) has a
single entry point.

Reward (per env, summed -- all on GPU, no host round-trip):

    + 1.0 * exp(-||cmd_xy - base_lin_vel_xy||^2 / 0.25)   forward-vel tracking
    + 0.5 * exp(-(cmd_wyaw - base_ang_vel_z)^2 / 0.25)    yaw-rate tracking
    + 0.5 * alive                                          stay-up bonus
    - 0.01 * ||action - last_action||^2                    action-rate penalty
    - 2.0  * base_lin_vel_z^2                               vertical-bounce pen.

This is a deliberately MILD shaping sufficient for the p03 "training launches and
begins" gate. The full legged_gym reward set (foot-air-time, torque/accel
penalties, contact terms, command resampling, domain randomization) and the
convergence run land in p04 -- see
``docs/plans/2026-05-28-v03-p04-go2-ppo-4096-demo.md``.

p04 SHAPING TODO (deferred -- do NOT fix in p03, the launch gate is met):
  * The action-rate penalty is unbounded (actions clip at +-100), so large
    exploratory actions early in training produce very negative episode returns
    -- this is calibration, not a bug. Add action clipping / a bounded penalty
    with the full reward set.
  * ``_prev_action`` is reset only in :meth:`reset` (full reset), NOT on the
    base env's per-step masked AUTORESET. So on the first step after an episode
    boundary the action-rate term compares the fresh action against the terminal
    action -> a spurious one-step penalty spike per episode. Zero ``_prev_action``
    for the autoreset envs when wiring the real reward in p04.
"""

from __future__ import annotations

import torch

from ..gym.env import NukaGymEnv
from . import go2_obs as G


# Reward weights (kept here, not in go2_obs, so the obs contract stays pure).
W_LIN_VEL_TRACK = 1.0
W_ANG_VEL_TRACK = 0.5
W_ALIVE = 0.5
W_ACTION_RATE = 0.01
W_LIN_VEL_Z = 2.0
TRACK_SIGMA = 0.25  # exp(-err^2 / sigma); legged_gym tracking_sigma


class Go2LocomotionEnv(NukaGymEnv):
    """Go2 flat-ground velocity-tracking locomotion env.

    Identical construction signature to :class:`NukaGymEnv` (it just overrides
    :meth:`compute_reward`). Holds the previous control action so the action-rate
    penalty has a delta to penalize.
    """

    def __init__(self, scene: str, num_envs: int, **kw) -> None:
        super().__init__(scene, num_envs, **kw)
        # Previous *control* action (distinct from base.last_action, which is the
        # clipped action stored for the obs): used for the action-rate penalty.
        self._prev_action = torch.zeros(
            self.num_envs, G.GO2_ACTION_DIM, device=self._torch_device
        )

    def reset(self, *, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        self._prev_action.zero_()
        return out

    def compute_reward(self) -> torch.Tensor:
        """Locomotion reward (num_envs,) cuda float32. Reads the live, post-step
        engine state via the base obs builder; ``self.last_action`` is THIS
        step's clipped action (the base sets it before calling us)."""
        b = self._obs
        lin_vel = b.base_lin_vel()          # (N,3) body frame
        ang_vel = b.base_ang_vel()          # (N,3) body frame
        cmd = self.command                  # (N,3) [vx, vy, wyaw]

        # Linear-velocity (xy) command tracking.
        lin_err = (cmd[:, :2] - lin_vel[:, :2]).pow(2).sum(dim=-1)
        r_lin = W_LIN_VEL_TRACK * torch.exp(-lin_err / TRACK_SIGMA)

        # Yaw-rate command tracking.
        ang_err = (cmd[:, 2] - ang_vel[:, 2]).pow(2)
        r_ang = W_ANG_VEL_TRACK * torch.exp(-ang_err / TRACK_SIGMA)

        # Alive bonus.
        r_alive = torch.full(
            (self.num_envs,), W_ALIVE, device=self._torch_device
        )

        # Action-rate penalty (smoothness): penalize change vs the previous step.
        action = self.last_action
        r_action_rate = -W_ACTION_RATE * (action - self._prev_action).pow(2).sum(dim=-1)
        self._prev_action = action.clone()

        # Vertical-velocity penalty (discourage bouncing).
        r_vz = -W_LIN_VEL_Z * lin_vel[:, 2].pow(2)

        return r_lin + r_ang + r_alive + r_action_rate + r_vz


def make_env(num_envs: int, *, device=None, **kw) -> Go2LocomotionEnv:
    """Construct the Go2 locomotion env on the shipped go2_float scene.

    Parameters mirror :class:`NukaGymEnv` (``dt``, ``decimation``, ``command``,
    ``termination_height``, ``termination_tilt_deg``, ``episode_length_s``,
    ``seed``). ``device`` is an open ``nuka.Device`` (created+owned by the env if
    ``None``). The scene is fixed to ``examples/scenes/go2_float.usda`` (p04 swaps
    in ``go2_locomotion.usda``).
    """
    scene = G._go2_scene_path()
    return Go2LocomotionEnv(scene, num_envs, device=device, **kw)
