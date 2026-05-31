"""``NukaGymEnv`` -- vectorized gymnasium env backed by ``nuka.World``.

See the package docstring for the contract. The step loop is:

    write actions -> DRIVE_TARGET[:,1:] (URDF order via the permutation)
    -> step_n(decimation) physics steps (hold the PD target)
    -> compose the 48-dim obs ON-GPU (pose is now fresh, post-step)
    -> reward + termination/truncation (the TERMINAL-transition values)
    -> AUTORESET the terminated|truncated envs via world.reset_envs, then return
       the POST-reset obs for those done envs (legged_gym / rl_games Brax
       convention) -- so the experience buffer pairs a fresh-episode obs with the
       fresh-episode reward, not a fallen-robot obs.

The post-reset obs of a done env is recomposed from the engine's restored
buffers (q / qd / base velocity all back at the creation snapshot). The one term
that would still be wrong is projected_gravity: it reads the FK-lagged
ARTICULATION_LINK_POSE root quat, which reset does NOT refresh (it still shows
the pre-reset/fallen orientation until the next Step's FK). So the done envs'
gravity slot is taken from the AUTHORITATIVE, un-lagged base_pose view
(NUKA_FIELD_BASE_POSE, restored directly by reset). Non-done envs keep their
normal post-step obs (which the golden bit-exact test pins).
"""

from __future__ import annotations

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

import nuka
from ..tasks import go2_obs as G


class NukaGymEnv:
    """Vectorized Go2 RL env on top of ``nuka.World`` (gymnasium 5-tuple step).

    Parameters
    ----------
    scene : str
        USDA scene path (the Go2 floating-base scene).
    num_envs : int
        Number of parallel envs.
    device : nuka.Device | None
        An open ``nuka.Device``; if ``None`` one is created (ordinal 0) and
        owned by this env (closed in :meth:`close`).
    dt : float
        Physics timestep (default 0.005, the Go2 native training dt).
    decimation : int
        Physics sub-steps per control step (default 4 -> 50 Hz control).
    command : sequence[float] | None
        Constant per-env velocity command ``[vx, vy, wyaw]`` (default
        ``[0.5, 0.0, 0.0]``). v0.3 uses a fixed command (no per-env resampling /
        domain randomization).
    termination_height : float
        Base height (m) below which an env terminates (fell). Default 0.18.
    termination_tilt_deg : float
        Base tilt (deg) above which an env terminates. Default 75.0.
    episode_length_s : float
        Max episode length (s) before truncation. Default 20.0.
    seed : int | None
        Seed for the action/space RNG (determinism).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        scene: str,
        num_envs: int,
        *,
        device: "nuka.Device | None" = None,
        dt: float = 0.005,
        decimation: int = 4,
        command=(0.5, 0.0, 0.0),
        termination_height: float = 0.18,
        termination_tilt_deg: float = 75.0,
        episode_length_s: float = 20.0,
        seed: int | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.decimation = int(decimation)
        self.dt = float(dt)
        self.termination_height = float(termination_height)
        self.termination_tilt_deg = float(termination_tilt_deg)
        self.max_episode_length = max(1, int(round(episode_length_s / (dt * decimation))))
        self._torch_device = torch.device("cuda")

        # Device ownership: create one iff the caller didn't pass one.
        self._owns_device = device is None
        self._device = device if device is not None else nuka.Device.create(0)

        self._world = nuka.World.create_from_scene(
            self._device, scene, self.num_envs, self.dt
        )
        assert self._world.action_dim == G.GO2_ACTION_DIM, (
            f"NukaGymEnv expects a 12-DOF Go2 (action_dim=12), got "
            f"{self._world.action_dim}"
        )

        # On-GPU obs builder (discovers the joint permutation on a throwaway
        # world; does NOT step our persistent world).
        self._obs = G.Go2ObsBuilder(self._device, self._world)
        self._obs.apply_pd_gains()

        # gym spaces.
        self.single_observation_space = spaces.Box(
            low=-G.OBS_CLIP, high=G.OBS_CLIP, shape=(G.GO2_OBS_DIM,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=-G.ACTION_CLIP, high=G.ACTION_CLIP, shape=(G.GO2_ACTION_DIM,),
            dtype=np.float32,
        )
        # Batched spaces (cheap insurance for rl_games / SB3 consumers).
        self.observation_space = spaces.Box(
            low=-G.OBS_CLIP, high=G.OBS_CLIP,
            shape=(self.num_envs, G.GO2_OBS_DIM), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-G.ACTION_CLIP, high=G.ACTION_CLIP,
            shape=(self.num_envs, G.GO2_ACTION_DIM), dtype=np.float32,
        )

        # Per-env state carried across steps (all on GPU).
        self.command = torch.tensor(
            command, dtype=torch.float32, device=self._torch_device
        ).expand(self.num_envs, 3).contiguous()
        self.last_action = torch.zeros(
            self.num_envs, G.GO2_ACTION_DIM, device=self._torch_device
        )
        self.episode_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self._torch_device
        )

        # Seeded RNG for any sampling (e.g. random-policy rollouts use this).
        self._generator = torch.Generator(device=self._torch_device)
        if seed is not None:
            self._generator.manual_seed(int(seed))
            self.single_action_space.seed(int(seed))

    # -- gym API ------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return G.GO2_ACTION_DIM

    @property
    def obs_dim(self) -> int:
        return G.GO2_OBS_DIM

    def reset(self, *, seed: int | None = None, options=None):
        """Reset ALL envs to the creation-time pose and return ``(obs, info)``.

        ``world.reset()`` restores the authoritative base_pose / q / qd / base
        velocity for every env. The base-orientation obs term (projected_gravity)
        is read from the AUTHORITATIVE, un-lagged base_pose view: the FK-lagged
        ARTICULATION_LINK_POSE root quat is NOT refreshed by reset, so on an
        already-stepped world its root slot still carries the pre-reset (fallen)
        orientation until the next Step. Right after create both sources agree
        (link_pose holds the creation pose), so this is a no-op there; on a stepped
        world it is what makes the returned reset obs correctly upright.
        """
        if seed is not None:
            self._generator.manual_seed(int(seed))
            self.single_action_space.seed(int(seed))
        self._world.reset()
        nuka.sync()
        self.last_action.zero_()
        self.episode_step.zero_()
        obs = self._obs.compute_obs(self.command, self.last_action)
        # All base_poses are authoritative-upright after a full reset; take the
        # gravity term from the un-lagged base_pose view (no-op post-create, fixes
        # the stale-lag obs on an already-stepped world).
        obs[:, 6:9] = self._obs.projected_gravity_auth()
        return obs, {}

    def step(self, actions: torch.Tensor):
        """One control step. ``actions``: (num_envs, 12) on cuda.

        Returns the gymnasium 5-tuple ``(obs, reward, terminated, truncated,
        info)``, all tensors on cuda.
        """
        actions = actions.to(self._torch_device, dtype=torch.float32)
        # Write PD targets (URDF -> Nuka slots, on GPU) and store the clipped
        # action as the next obs's last_action.
        clipped = self._obs.write_action(actions)
        self.last_action = clipped

        # Hold the PD target for `decimation` physics steps.
        self._world.step_n(self.decimation)

        self.episode_step += 1

        # Compose obs ON-GPU from the now-fresh post-step state.
        obs = self._obs.compute_obs(self.command, self.last_action)

        reward = self.compute_reward()
        terminated = self.compute_terminated()
        truncated = self.episode_step >= self.max_episode_length
        info: dict = {}

        # AUTORESET (legged_gym / Brax convention): a vectorized env that
        # terminates an env must RESET it and return the POST-reset obs for that
        # env in the SAME step, so the experience buffer pairs the fresh-episode
        # obs with the fresh-episode reward (not a fallen-robot obs). reward /
        # terminated / truncated above are the TERMINAL-transition values (correct
        # -- computed pre-reset); only the returned obs[done] is swapped to the
        # reset state.
        done = terminated | truncated
        if bool(done.any()):
            done_ids = torch.nonzero(done, as_tuple=False).flatten().to(torch.int32)
            # Reset the engine's authoritative state for the done envs (q/qd/base
            # velocity/base_pose restored to the creation snapshot; isolation:
            # non-done envs are byte-for-byte untouched). reset_envs syncs the
            # stream, so the torch reads below see the settled post-reset buffers.
            self._world.reset_envs(done_ids.cpu())
            # Zero the per-env bookkeeping for the reset envs (a fresh env must NOT
            # carry stale action history into its obs).
            self.last_action[done] = 0.0
            self.episode_step[done] = 0

            # POST-reset obs for the done envs. After reset_envs the engine buffers
            # for these envs hold the upright init state, so recomposing the obs
            # gives the correct fresh-episode obs EXCEPT projected_gravity [6:9]:
            # that term reads the FK-lagged ARTICULATION_LINK_POSE root quat, which
            # reset does NOT refresh (it still shows the pre-reset/fallen pose until
            # the next Step's FK). So we recompose with last_action already zeroed,
            # then overwrite the gravity slot from the AUTHORITATIVE (un-lagged)
            # base_pose view. Only the done rows are written back into `obs`; the
            # non-done rows keep their normal post-step obs (golden-pinned).
            obs_reset = self._obs.compute_obs(self.command, self.last_action)
            obs_reset[:, 6:9] = self._obs.projected_gravity_auth()
            obs = obs.clone()
            obs[done] = obs_reset[done]

        return obs, reward, terminated, truncated, info

    # -- overridable reward / termination hooks -----------------------------
    def compute_reward(self) -> torch.Tensor:
        """Minimal reward: forward-velocity command tracking + alive bonus.

        ``exp(-||cmd_xy - base_lin_vel_xy||^2 / 0.25) + 0.1``. The task layer
        (p04) overrides this with the full locomotion shaping. Returns
        (num_envs,) cuda float32.
        """
        lin_vel_xy = self._obs.base_lin_vel()[:, :2]
        cmd_xy = self.command[:, :2]
        err = (cmd_xy - lin_vel_xy).pow(2).sum(dim=-1)
        track = torch.exp(-err / 0.25)
        alive = torch.full(
            (self.num_envs,), 0.1, device=self._torch_device
        )
        return track + alive

    def compute_terminated(self) -> torch.Tensor:
        """Terminate envs whose base fell below ``termination_height`` or tilted
        past ``termination_tilt_deg``. Returns (num_envs,) cuda bool."""
        base_z = self._obs.base_pos()[:, 2]
        tilt = self._obs.tilt_deg()
        fell = base_z < self.termination_height
        tipped = tilt > self.termination_tilt_deg
        return fell | tipped

    # -- random policy helper (seeded) --------------------------------------
    def sample_actions(self) -> torch.Tensor:
        """A seeded random action batch in [-1, 1] (num_envs, 12) on cuda --
        convenience for smoke rollouts."""
        return (
            torch.rand(
                self.num_envs, G.GO2_ACTION_DIM,
                generator=self._generator, device=self._torch_device,
            ) * 2.0 - 1.0
        )

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        """Destroy the world and (if owned) close the device."""
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
