"""BDX biped flat-ground velocity-tracking locomotion env (rl_games backend).

A standalone vectorized env on the ONE general ``nuka.World`` path (Scene ->
CookToModel -> nk::World), mirroring the Go2 locomotion env's autoreset / command
curriculum but for the 14-DOF duck: the policy drives the 10 leg joints (head PD-held
at home), obs is the 44-dim proprioceptive vector (incl. the gait phase clock),
reward is the Open Duck Playground imitation recipe in :mod:`nuka.tasks.bdx_rewards`
tracking the polynomial reference gait. Kept separate from the Go2 stack so the
Go2 golden path is byte-untouched; the physics path is identical (no special-casing).

Contact matching uses the kinematic foot-contact proxy (sole bottom near the
ground) -- flat-ground robust and independent of the heightfield contact-wrench readout.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

import nuka
from . import bdx_obs as B
from .bdx_rewards import BdxReward, REWARD_DT
from .bdx_reference_motion import BdxReferenceMotion


SCENE = "examples/scenes/bdx_stand.nks"
REFERENCE_PKL = "examples/assets/bdx/polynomial_coefficients.pkl"

# Velocity command ranges (the Playground joystick training domain; commands
# outside the reference-motion grid snap to its nearest bin for imitation).
CMD_RANGE_VX = (-0.15, 0.15)
CMD_RANGE_VY = (-0.2, 0.2)
CMD_RANGE_WYAW = (-1.0, 1.0)
# Hold each sampled command for the WHOLE episode: sustained walking is the
# task (a 10 s resample never demanded one continuous 20 s walk -> late falls).
RESAMPLING_TIME_S = 20.0
ZERO_COMMAND_PROB = 0.1

# Reset leg stance = home x U(lo,hi) (the Playground scheme): wild-but-legal
# stances make every episode start a step-recovery, breaking the stand basin.
RESET_JOINT_SCALE = (0.5, 1.5)


def _foot_capsules(nks_path):
    """The two sole-capsule (local_pos, radius) pairs, tree order = left, right."""
    caps = []

    def walk(node):
        if isinstance(node, list):
            for child in node:
                walk(child)
            return
        shape = node.get("collision_shape")
        if node.get("name") == "foot_contact" and shape:
            caps.append((np.array(shape["local"]["pos"], np.float32),
                         float(shape["radius"])))
        walk(node.get("children", []))

    walk(json.load(open(nks_path))["tree"])
    return caps


class BdxWalkEnv:
    """Vectorized BDX walk env (gymnasium 5-tuple step, all tensors on cuda)."""

    metadata = {"render_modes": []}

    def __init__(self, scene: str, num_envs: int, *, device=None,
                 dt: float = 0.005, decimation: int = 4,
                 command=(0.15, 0.0, 0.0), fixed_command: bool = False,
                 termination_height: float = 0.10, termination_tilt_deg: float = 60.0,
                 episode_length_s: float = 20.0, seed: int | None = None,
                 reward_scales=None, sigma_lin=None, sigma_ang=None,
                 command_ranges=None, resample_time_s: float = RESAMPLING_TIME_S,
                 reference_pkl: str = REFERENCE_PKL,
                 push_enable: bool = False,
                 push_interval_range=(5.0, 10.0), push_magnitude_range=(0.1, 1.0),
                 **_ignored) -> None:
        self.num_envs = int(num_envs)
        self.decimation = int(decimation)
        self.dt = float(dt)
        self.termination_height = float(termination_height)
        self.termination_tilt_deg = float(termination_tilt_deg)
        self.max_episode_length = max(1, int(round(episode_length_s / (dt * decimation))))
        self._dev = torch.device("cuda")

        self._owns_device = device is None
        self._device = device if device is not None else nuka.Device.create(0)
        self._world = nuka.World.create_from_scene(self._device, scene, self.num_envs, self.dt)
        if int(self._world.action_dim) != len(B.LEG_JOINTS) + len(B.HEAD_JOINTS):
            raise RuntimeError(
                f"BdxWalkEnv expects a 14-DOF duck; got action_dim={self._world.action_dim}")

        # Obs builder (name-resolved leg/head slots + foot-contact geometry).
        ankle_slots = [int(self._world.dof_index("^left_ankle$")[0]),
                       int(self._world.dof_index("^right_ankle$")[0])]
        caps = _foot_capsules(scene)
        self._obs = B.BdxObsBuilder(self._world, ankle_slots, caps)
        self._obs.set_target_rate_cap(dt * decimation)
        self._obs.apply_pd_gains()

        # gym spaces (single-env; rl_games prepends num_actors).
        self.single_observation_space = spaces.Box(
            low=-B.OBS_CLIP, high=B.OBS_CLIP, shape=(B.BDX_OBS_DIM,), dtype=np.float32)
        self.single_action_space = spaces.Box(
            low=-B.ACTION_SPACE_LIMIT, high=B.ACTION_SPACE_LIMIT,
            shape=(B.BDX_ACTION_DIM,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-B.OBS_CLIP, high=B.OBS_CLIP,
            shape=(self.num_envs, B.BDX_OBS_DIM), dtype=np.float32)
        self.action_space = spaces.Box(
            low=-B.ACTION_SPACE_LIMIT, high=B.ACTION_SPACE_LIMIT,
            shape=(self.num_envs, B.BDX_ACTION_DIM), dtype=np.float32)

        cr = command_ranges or {}
        self._cmd_range_x = tuple(cr.get("lin_vel_x", CMD_RANGE_VX))
        self._cmd_range_y = tuple(cr.get("lin_vel_y", CMD_RANGE_VY))
        self._cmd_range_yaw = tuple(cr.get("ang_vel_yaw", CMD_RANGE_WYAW))
        self._fixed_command = bool(fixed_command)

        # Per-env state (all on GPU).
        self.command = torch.tensor(command, dtype=torch.float32, device=self._dev
                                    ).expand(self.num_envs, 3).contiguous()
        self.last_action = torch.zeros(self.num_envs, B.BDX_ACTION_DIM, device=self._dev)
        self._prev_action = torch.zeros(self.num_envs, B.BDX_ACTION_DIM, device=self._dev)
        self.episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self._dev)
        self._resample_every = max(1, int(round(float(resample_time_s) / (dt * decimation))))

        self._generator = torch.Generator(device=self._dev)
        if seed is not None:
            self._generator.manual_seed(int(seed))

        rw_kw = {}
        if sigma_lin is not None:
            rw_kw["sigma_lin"] = sigma_lin
        if sigma_ang is not None:
            rw_kw["sigma_ang"] = sigma_ang
        self._reward = BdxReward(
            self.num_envs, self._dev, scales=reward_scales, **rw_kw)
        self._reward.default_leg = self._obs.default_leg.clone()

        # Reference gait (per-command periodic joint/vel/contact lookup) + the
        # per-env phase clock it is indexed by (advances every control step).
        self._ref = BdxReferenceMotion(reference_pkl, self._dev)
        self.phase_i = torch.zeros(self.num_envs, dtype=torch.long, device=self._dev)

        # Hard joint limits (small margin) bounding the randomized reset stance.
        self._leg_lo = torch.as_tensor(B.DOF_POS_LOWER + 0.01, device=self._dev)
        self._leg_hi = torch.as_tensor(B.DOF_POS_UPPER - 0.01, device=self._dev)

        # Push perturbations (Playground push_config): every U(interval) seconds
        # add a random horizontal velocity kick of U(magnitude) m/s to the base.
        self._push_enable = bool(push_enable)
        self._push_interval_range = tuple(float(v) for v in push_interval_range)
        self._push_magnitude_range = tuple(float(v) for v in push_magnitude_range)
        self.push_step = torch.zeros(self.num_envs, dtype=torch.long, device=self._dev)
        self.push_interval_steps = torch.full(
            (self.num_envs,), 10 ** 9, dtype=torch.long, device=self._dev)
        self._vel_view = torch.from_dlpack(self._world.buffer_view(nuka.LINK_VELOCITY))

        # Writable joint views for the reset teleport (zero-copy, in place).
        self._q = torch.from_dlpack(self._world.buffer_view(nuka.JOINT_POSITION))
        self._qd = torch.from_dlpack(self._world.buffer_view(nuka.JOINT_VELOCITY))

        self._diag = (dict(n=0, sum_r=0.0, n_term=0)
                      if os.environ.get("NUKA_BDX_DIAG") else None)
        print(f"[bdx_locomotion] envs={self.num_envs} action_dim={B.BDX_ACTION_DIM} "
              f"obs_dim={B.BDX_OBS_DIM} leg_slots={self._obs.leg_slots.tolist()} "
              f"head_slots={self._obs.head_slots.tolist()} ctrl_dt={dt*decimation:.3f} "
              f"max_ep={self.max_episode_length}", flush=True)

    @property
    def action_dim(self):
        return B.BDX_ACTION_DIM

    @property
    def obs_dim(self):
        return B.BDX_OBS_DIM

    # -- reset teleport (home stance x random scale, Playground-style) -------
    def _teleport(self, mask) -> None:
        n = self.num_envs
        lo, hi = RESET_JOINT_SCALE
        scale = torch.rand(n, B.BDX_ACTION_DIM, generator=self._generator,
                           device=self._dev) * (hi - lo) + lo
        leg = torch.clamp(self._obs.default_leg * scale,
                          self._leg_lo, self._leg_hi)
        leg_cols = self._obs.leg_slots.unsqueeze(0)                 # (1,10)
        head_cols = self._obs.head_slots.unsqueeze(0)              # (1,4)
        if mask is None:
            self._q[:, self._obs.leg_slots] = leg
            self._q[:, self._obs.head_slots] = self._obs.default_head
            self._qd[:, 1:self._obs.blc] = 0.0
        else:
            idx = torch.nonzero(mask).flatten().unsqueeze(1)       # (K,1)
            self._q[idx, leg_cols] = leg[mask]
            self._q[idx, head_cols] = self._obs.default_head
            self._qd[mask, 1:self._obs.blc] = 0.0
        nuka.sync()

    def _resample_push_intervals(self, mask) -> None:
        if not self._push_enable:
            return
        lo, hi = self._push_interval_range
        secs = torch.rand(self.num_envs, generator=self._generator,
                          device=self._dev) * (hi - lo) + lo
        steps = (secs / (self.dt * self.decimation)).round().long().clamp(min=1)
        self.push_interval_steps[mask] = steps[mask]
        self.push_step[mask] = 0

    def _apply_pushes(self) -> None:
        self.push_step += 1
        due = (self.push_step % self.push_interval_steps) == 0
        if not bool(due.any()):
            return
        gen = self._generator
        theta = torch.rand(self.num_envs, generator=gen, device=self._dev) * (2 * np.pi)
        lo, hi = self._push_magnitude_range
        mag = torch.rand(self.num_envs, generator=gen, device=self._dev) * (hi - lo) + lo
        kick = torch.stack((mag * torch.cos(theta), mag * torch.sin(theta)), dim=1)
        self._vel_view[due, 0, 3:5] += kick[due]
        nuka.sync()

    def _resample_commands(self, mask) -> None:
        if self._fixed_command:
            return
        n = int(mask.sum().item())
        if n == 0:
            return
        gen = self._generator

        def _u(lo, hi):
            return torch.rand(n, generator=gen, device=self._dev) * (hi - lo) + lo

        new = torch.empty(n, 3, device=self._dev)
        new[:, 0] = _u(*self._cmd_range_x)
        new[:, 1] = _u(*self._cmd_range_y)
        new[:, 2] = _u(*self._cmd_range_yaw)
        # With ZERO_COMMAND_PROB, snap the whole command to zero (learn to stand).
        zero = torch.rand(n, generator=gen, device=self._dev) < ZERO_COMMAND_PROB
        new[zero] = 0.0
        self.command[mask] = new

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._generator.manual_seed(int(seed))
        self._world.reset()
        nuka.sync()
        self._teleport(None)
        self._obs.reset_prev_target(None)
        self.last_action.zero_()
        self._prev_action.zero_()
        self.episode_step.zero_()
        self.phase_i.zero_()
        all_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self._dev)
        self._resample_commands(all_envs)
        self._resample_push_intervals(all_envs)
        obs = self._obs.compute_obs(self.command, self.last_action,
                                    self._ref.phase_features(self.phase_i))
        return obs, {}

    def compute_terminated(self) -> torch.Tensor:
        base_z = self._obs.base_pos()[:, 2]
        tilt = self._obs.tilt_deg()
        q = self._obs.q_leg()
        bad = ~torch.isfinite(q).all(dim=1) | ~torch.isfinite(base_z)
        return (base_z < self.termination_height) | (tilt > self.termination_tilt_deg) | bad

    def step(self, actions: torch.Tensor):
        actions = actions.to(self._dev, dtype=torch.float32)
        clipped = self._obs.write_action(actions)
        self.last_action = clipped

        if self._push_enable:
            self._apply_pushes()
        self._world.step_n(self.decimation)
        self.episode_step += 1
        self.phase_i = (self.phase_i + 1) % self._ref.nb_steps

        # Reference frame at each env's live command bin + phase (bin recomputed
        # from the live command so external pinning/eval never goes stale).
        frame = self._ref.gather(self._ref.command_to_bin(self.command), self.phase_i)
        phase = self._ref.phase_features(self.phase_i)

        obs = self._obs.compute_obs(self.command, self.last_action, phase)
        reward = self._reward.compute(
            cmd=self.command,
            lin_vel=self._obs.base_lin_vel(),
            ang_vel=self._obs.base_ang_vel(),
            q_leg=self._obs.q_leg(),
            qd_leg=self._obs.qd_leg(),
            target_leg=self._obs.prev_leg_target,
            action=self.last_action,
            last_action=self._prev_action,
            contact=self._obs.foot_contact(),
            ref_leg_pos=self._ref.ref_leg_pos(frame),
            ref_leg_vel=self._ref.ref_leg_vel(frame),
            ref_contacts=self._ref.ref_foot_contacts(frame),
            ref_lin_vel=self._ref.ref_base_lin_vel(frame),
            ref_ang_vel=self._ref.ref_base_ang_vel(frame),
        )
        terminated = self.compute_terminated()
        truncated = self.episode_step >= self.max_episode_length
        info: dict = {}

        self._prev_action = self.last_action.clone()

        done = terminated | truncated
        if bool(done.any()):
            done_ids = torch.nonzero(done, as_tuple=False).flatten().to(torch.int32)
            self._world.reset_envs(done_ids.cpu())
            self._teleport(done)
            self._obs.reset_prev_target(done)
            self.last_action[done] = 0.0
            self._prev_action[done] = 0.0
            self.episode_step[done] = 0
            self.phase_i[done] = 0
            self._resample_push_intervals(done)
            obs_reset = self._obs.compute_obs(
                self.command, self.last_action, self._ref.phase_features(self.phase_i))
            obs = obs.clone()
            obs[done] = obs_reset[done]

        periodic = (self.episode_step % self._resample_every == 0)
        to_resample = periodic | done
        if bool(to_resample.any()):
            self._resample_commands(to_resample)
            obs[:, 9:12] = self.command * self._obs.cmd_scale

        if self._diag is not None:
            self._diag_step(reward, terminated)

        return obs, reward, terminated, truncated, info

    def _diag_step(self, reward, terminated):
        d = self._diag
        d["sum_r"] += float(reward.mean())
        d["n_term"] += int(terminated.sum())
        d["n"] += 1
        if d["n"] % 100 == 0:
            z = self._obs.base_pos()[:, 2]
            tilt = self._obs.tilt_deg()
            v = self._obs.base_lin_vel()
            print(f"[bdx_diag] step{d['n']}: mean_r={d['sum_r']/d['n']:.4f} "
                  f"term/100={d['n_term']} z={float(z.mean()):.3f} "
                  f"tilt={float(tilt.mean()):.1f} vx={float(v[:,0].mean()):.3f}",
                  flush=True)
            d["n_term"] = 0

    def close(self):
        if getattr(self, "_world", None) is not None:
            self._world.destroy()
            self._world = None
        if self._owns_device and getattr(self, "_device", None) is not None:
            self._device.close()
            self._device = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_env(num_envs: int, *, device=None, **kw) -> BdxWalkEnv:
    """Construct the BDX walk env on the shipped bdx_stand scene."""
    return BdxWalkEnv(SCENE, num_envs, device=device, **kw)
