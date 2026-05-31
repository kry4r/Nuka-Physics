"""Go2 flat-ground locomotion task -- legged_gym reward + command curriculum.

A :class:`~nuka.gym.env.NukaGymEnv` subclass that supplies (a) the legged_gym Go2
locomotion REWARD ported verbatim (see :mod:`nuka.tasks.go2_rewards`) and (b) the
per-env velocity COMMAND curriculum (resampled from ``commands.ranges`` on a timer
and on reset, with ``|lin_vel|<0.2`` zeroed -- legged_gym behaviour). The obs math,
joint-order permutation, and height/tilt termination all live in the base env (and
the builder); this layer adds reward shaping + command randomization + a
``make_env`` factory so the RL stack has a single entry point.

This REPLACES the earlier mild hand-rolled shaping (forward-vel tracking + alive
bonus + unbounded action-rate). Under that placeholder the per-episode return was
~-14000..-20000 (~-1000/step) -- the unbounded action-rate penalty dominated and
the gradient did NOT point toward walking. The legged_gym port fixes this by
construction: ``only_positive_rewards`` floors the per-step total at 0 (so no
regularizer can drive it deeply negative) and the term BALANCE is the proven-to-
converge one.

----------------------------------------------------------------------------
COMMAND CURRICULUM (legged_gym ``commands`` + ``_resample_commands``):
  ranges (go2 inherits the base): lin_vel_x [-1,1], lin_vel_y [-1,1],
          ang_vel_yaw [-1,1] m/s | rad/s.
  resample period: ``resampling_time = 10.0`` s => 10.0 / control_dt(0.02) = 500
          control steps (and on full reset + on every autoreset/done env).
  zeroing: after sampling, ``commands[:, :2] *= (||commands_xy|| > 0.2)`` -- a
          near-zero xy command is snapped to exactly 0 (legged_gym line 306).
  DEVIATION (logged): legged_gym go2 has ``heading_command = True`` (the yaw
          command is derived from a heading error). We sample ``ang_vel_yaw``
          DIRECTLY from its range instead (the ``else`` branch). Heading mode
          needs a 4th command dim + a per-step yaw recompute from the base
          quaternion + an obs-layout change; the 3-dim obs command is golden-
          pinned, so we keep the direct-yaw form. The ±velocity range the p04
          exit needs is fully covered.
  Seeded (deterministic) via the base env's ``torch.Generator``.

AUTORESET FIXES (the two bugs the p03 review flagged):
  * ``_prev_action`` (the action-rate delta source) is zeroed for done envs on
    the per-step MASKED autoreset (legged_gym ``last_actions[env_ids]=0`` in
    ``reset_idx``), not only on full ``reset()``. Without this, the first step of
    a fresh episode penalizes the fresh action against the terminal action.
  * ``reward.last_dof_vel`` (the dof_acc finite-difference source) is likewise
    zeroed for done envs (legged_gym ``last_dof_vel[env_ids]=0``), else the first
    fresh step sees a spurious huge acceleration.
----------------------------------------------------------------------------
"""

from __future__ import annotations

import torch

from ..gym.env import NukaGymEnv
from . import go2_obs as G
from .go2_rewards import Go2Reward, DROPPED, REWARD_DT


# legged_gym commands.ranges (go2 inherits the LeggedRobotCfg base defaults).
CMD_RANGE_LIN_VEL_X = (-1.0, 1.0)
CMD_RANGE_LIN_VEL_Y = (-1.0, 1.0)
CMD_RANGE_ANG_VEL_YAW = (-1.0, 1.0)
# commands.resampling_time = 10.0 s ; min |xy| command (else zeroed).
RESAMPLING_TIME_S = 10.0
CMD_ZERO_THRESHOLD = 0.2


class Go2LocomotionEnv(NukaGymEnv):
    """Go2 flat-ground velocity-tracking locomotion env (legged_gym reward).

    Same construction signature as :class:`NukaGymEnv`; overrides
    :meth:`compute_reward` (the ported reward) and :meth:`step` (command
    resampling + the autoreset bookkeeping fixes). The ``command`` ctor arg is the
    INITIAL command (immediately overwritten by the first resample on reset).
    """

    def __init__(self, scene: str, num_envs: int, **kw) -> None:
        super().__init__(scene, num_envs, **kw)
        # Ported legged_gym reward (owns last_dof_vel for dof_acc).
        self._reward = Go2Reward(self.num_envs, self._torch_device)
        # Prior step's clipped action (the action-rate delta source). Distinct
        # from base.last_action, which is THIS step's clipped action.
        self._prev_action = torch.zeros(
            self.num_envs, G.GO2_ACTION_DIM, device=self._torch_device
        )
        # PD target this step (default + 0.25*action), in URDF order -- the torque
        # approximation needs it; recomputed each step from the clipped action.
        self._target_q_urdf = self._reward.default_angles.expand(
            self.num_envs, G.GO2_ACTION_DIM
        ).contiguous()
        # Resample period in CONTROL steps.
        self._resample_every = max(1, int(round(RESAMPLING_TIME_S / REWARD_DT)))
        # Log the dropped terms once (term + reason) -- honest record in stdout.
        for name, reason in DROPPED.items():
            print(f"[go2_locomotion] DROPPED reward term '{name}': {reason}",
                  flush=True)

    # -- command sampling (legged_gym _resample_commands, seeded) -----------
    def _resample_commands(self, mask: torch.Tensor) -> None:
        """Resample the velocity command for the envs in the boolean ``mask``.

        ``commands[i] = U[range]`` for lin_vel_x / lin_vel_y / ang_vel_yaw, then
        ``commands[:, :2] *= (||xy|| > 0.2)`` (zero near-still commands). Uses the
        base env's seeded generator for determinism. All on GPU."""
        n = int(mask.sum().item())
        if n == 0:
            return
        gen = self._generator

        def _u(lo, hi):
            return torch.rand(n, generator=gen, device=self._torch_device) * (hi - lo) + lo

        new = torch.empty(n, 3, device=self._torch_device)
        new[:, 0] = _u(*CMD_RANGE_LIN_VEL_X)
        new[:, 1] = _u(*CMD_RANGE_LIN_VEL_Y)
        new[:, 2] = _u(*CMD_RANGE_ANG_VEL_YAW)
        # Zero the xy command if its magnitude is below the threshold.
        keep = (new[:, :2].norm(dim=1) > CMD_ZERO_THRESHOLD).unsqueeze(1)
        new[:, :2] = new[:, :2] * keep
        self.command[mask] = new

    def reset(self, *, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        # Fresh-episode bookkeeping.
        self._prev_action.zero_()
        self._reward.last_dof_vel.zero_()
        # Resample EVERY env's command on a full reset, then recompose the obs
        # command slot so the returned reset obs carries the new command.
        all_envs = torch.ones(
            self.num_envs, dtype=torch.bool, device=self._torch_device
        )
        self._resample_commands(all_envs)
        obs, info = out
        obs[:, 9:12] = self.command * self._obs.cmd_scale
        return obs, info

    def compute_reward(self) -> torch.Tensor:
        """Ported legged_gym Go2 reward (num_envs,) cuda float32. Reads the live,
        post-step engine state via the base obs builder; ``self.last_action`` is
        THIS step's clipped action (the base sets it before calling us);
        ``self._prev_action`` is the prior step's clipped action."""
        b = self._obs
        return self._reward.compute(
            cmd=self.command,
            lin_vel=b.base_lin_vel(),
            ang_vel=b.base_ang_vel(),
            q_urdf=b.q_urdf(),
            qd_urdf=b.qd_urdf(),
            target_q_urdf=self._target_q_urdf,
            action=self.last_action,
            last_action=self._prev_action,
        )

    def step(self, actions: torch.Tensor):
        # Record the PD target this step (default + 0.25*action, URDF order) for
        # the torque approximation -- mirror the clip the base write_action does.
        clipped = actions.to(self._torch_device, dtype=torch.float32).clamp(
            -G.ACTION_CLIP, G.ACTION_CLIP
        )
        self._target_q_urdf = self._reward.default_angles + G.ACTION_SCALE * clipped

        # Base step: writes PD targets, steps physics, composes obs, calls our
        # compute_reward (which reads self.last_action == this step's clipped
        # action and self._prev_action == prior), and runs the masked autoreset.
        obs, reward, terminated, truncated, info = super().step(actions)

        # Roll the action-rate history AFTER the reward used the (prev, this) pair.
        self._prev_action = self.last_action.clone()

        # Periodic command resample (legged_gym _post_physics_step_callback:
        # episode_length_buf % (resampling_time/dt) == 0). episode_step was just
        # incremented in the base step.
        periodic = (self.episode_step % self._resample_every == 0)

        done = terminated | truncated
        if bool(done.any()):
            # Autoreset bookkeeping fixes: zero the action-rate + dof_acc history
            # for the done envs (legged_gym reset_idx zeros last_actions +
            # last_dof_vel). The base already zeroed last_action[done].
            self._prev_action[done] = 0.0
            self._reward.last_dof_vel[done] = 0.0

        # Resample command for (periodic | done) envs; recompose the obs command
        # slot so the returned obs carries the fresh command for those envs.
        to_resample = periodic | done
        if bool(to_resample.any()):
            self._resample_commands(to_resample)
            # compute_obs returns a fresh tensor (the done-path already cloned),
            # so patching the command slot in place is always safe here.
            obs[:, 9:12] = self.command * self._obs.cmd_scale

        return obs, reward, terminated, truncated, info


def make_env(num_envs: int, *, device=None, **kw) -> Go2LocomotionEnv:
    """Construct the Go2 locomotion env on the shipped go2_locomotion scene.

    Parameters mirror :class:`NukaGymEnv` (``dt``, ``decimation``, ``command``,
    ``termination_height``, ``termination_tilt_deg``, ``episode_length_s``,
    ``seed``). ``device`` is an open ``nuka.Device`` (created+owned by the env if
    ``None``). The scene is fixed to ``examples/scenes/go2_locomotion.usda`` (a
    byte-identical copy of the proven go2_float flat-ground floating-base scene)."""
    scene = G._go2_scene_path()
    return Go2LocomotionEnv(scene, num_envs, device=device, **kw)
