"""H1GraspEnv (A3) -- a fixed-base 2-finger cup-grasp RL env over nuka.GraspWorld.

A FRESH env class (NOT a NukaGymEnv subclass -- that base is married to the C-ABI
``nuka.World`` / Go2 obs builder) with the SAME duck-typed surface the rl_games
adapter + a gym consumer expect:

    single_observation_space / single_action_space   (single-env Box spaces)
    observation_space / action_space                  (batched Box spaces)
    num_envs / action_dim / obs_dim                   (scalars)
    reset(seed=...) -> (obs, info)
    step(actions)   -> (obs, reward, terminated, truncated, info)  (5-tuple)
    sample_actions() -> a seeded random action batch (smoke convenience)
    close()

All step tensors are CUDA float32 (obs/reward/done) -- assembled on the HOST from
the GraspWorld export dicts (one bulk device->host copy per buffer) then moved to
CUDA in one round trip (the engine's export is host-resident; the 2-DOF obs is
tiny). See h1_grasp_obs for the obs layout and h1_grasp_rewards for the terms.

THE DISCRIMINATIVE IC (A3, proven empirically -- see the increment report):
  The synthetic gripper actuates its two fingertips along X ONLY (prismatic joints,
  axes +-X). At the legacy IC (cup centered between fingers, fingertips pre-posed in
  1.5 mm contact) a constant-max squeeze trivially catches the cup -> NON-
  discriminative; and the +-2.5 cm reset jitter does NOT fix it (the dropped cups
  are jittered along Y, which has NO actuation -> unsaveable by ANY policy, so a
  state-using policy cannot beat const-max).

  The fix is the TIMING IC, realized by the minimal C-ABI knob ``cup_start_z_offset``
  (default 0 == byte-identical validated scene; the 21 C++ gates re-pass green): the
  cup starts ``cup_start_z_offset`` metres ABOVE the fingertip catch plane (z=0.20)
  with the fingers OPEN and NO contact at t=0, and DESCENDS under gravity. A constant-
  max slam shuts the fingers on empty space while the cup is still high -> the cup
  deflects/falls past -> DROP (const-max hold rate ~0). A policy that holds the gap
  (action ~ 0) and squeezes when the cup reaches the plane CATCHES it (hold ~1). So a
  constant action does NOT solve the task -- the gate the task demands.

ACTION MAPPING: single_action_space = Box(-1, 1, (2,)). rl_games rescales the
policy's [-1,1] output onto these bounds (ACTION_SPACE_LIMIT == 1.0). In step() the
[-1,1] action is mapped to a physical per-finger drive force by x FINGER_FORCE_SCALE
(== 12 N), so the proven ~8 N holding force is reachable at action ~ +0.67 (mid
range). SIGN: POSITIVE action = CLOSE / squeeze (both prismatic joints drive their
fingertip toward the cup centre under positive torque); NEGATIVE action = open.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from gymnasium import spaces

import nuka
from . import h1_grasp_obs as H
from .h1_grasp_rewards import H1GraspReward


# The default validated cup asset (the SAME path the C7a factory + the A2 binding
# test default to), repo-relative -> absolute.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_CUP_ASSET = os.path.join(
    _REPO_ROOT, ".nuka-assets", "newton_assets", "manipulation_objects", "cup",
    "model.usda",
)

# [-1,1] action -> physical per-finger drive force (N). 12 N so the proven ~8 N hold
# is reachable mid-range (action ~ +0.67). rl_games' ACTION_SPACE_LIMIT == 1.0.
ACTION_SPACE_LIMIT = 1.0
FINGER_FORCE_SCALE = 12.0

# The A3 discriminative cup-start offset (m) above the fingertip catch plane. 0.06
# is the gentlest robust point: the cup clears the fingertip band (t0 contacts == 0)
# so const-max fails, while the arrival velocity (~ -0.95 m/s) is low enough that a
# timed close catches it (proven in the A3 clean centered scan).
DEFAULT_CUP_START_Z_OFFSET = 0.06

# Drop-termination floor (m): BELOW the catch plane so a cup that legitimately
# descends from the start height THROUGH the catch plane before being caught is NOT
# killed mid-flight; a true drop runs well past this to terminate.
DEFAULT_TERMINATION_Z = 0.12


class H1GraspEnv:
    """Fixed-base 2-finger cup-grasp RL env (gymnasium 5-tuple step) on GraspWorld."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        num_envs: int,
        *,
        device: "nuka.Device | None" = None,
        cup_asset_path: str | None = None,
        dt: float = 1.0 / 240.0,
        decimation: int = 1,
        grip_force: float = 0.0,
        friction_mu: float = 0.8,
        gravity_z: float = -9.81,
        cup_start_z_offset: float = DEFAULT_CUP_START_Z_OFFSET,
        termination_z: float = DEFAULT_TERMINATION_Z,
        episode_length: int = 220,
        seed: int | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.decimation = int(decimation)
        self.dt = float(dt)
        self.termination_z = float(termination_z)
        self.max_episode_length = int(episode_length)
        self._torch_device = torch.device("cuda")
        cup_asset_path = cup_asset_path or DEFAULT_CUP_ASSET
        if not os.path.exists(cup_asset_path):
            raise FileNotFoundError(f"H1GraspEnv cup asset not found: {cup_asset_path}")

        # Device ownership: create one iff the caller didn't pass one.
        self._owns_device = device is None
        self._device = device if device is not None else nuka.Device.create(0)

        # grip_force=0 so the DEFAULT (no set_actions) drives ZERO torque -> the
        # discriminative IC is honest (a no-op env is a clean drop, not a template
        # hold). The policy's action is the ONLY squeeze source.
        self._world = nuka.GraspWorld.create(
            self._device, cup_asset_path, env_count=self.num_envs,
            grip_force=float(grip_force), friction_mu=float(friction_mu),
            gravity_z=float(gravity_z), dt=self.dt,
            cup_start_z_offset=float(cup_start_z_offset),
        )
        assert self._world.action_dim == H.H1_GRASP_ACTION_DIM, (
            f"H1GraspEnv expects a 2-DOF gripper, got action_dim="
            f"{self._world.action_dim}"
        )

        # gym spaces.
        self.single_observation_space = spaces.Box(
            low=-H.OBS_CLIP, high=H.OBS_CLIP, shape=(H.H1_GRASP_OBS_DIM,),
            dtype=np.float32,
        )
        self.single_action_space = spaces.Box(
            low=-ACTION_SPACE_LIMIT, high=ACTION_SPACE_LIMIT,
            shape=(H.H1_GRASP_ACTION_DIM,), dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-H.OBS_CLIP, high=H.OBS_CLIP,
            shape=(self.num_envs, H.H1_GRASP_OBS_DIM), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-ACTION_SPACE_LIMIT, high=ACTION_SPACE_LIMIT,
            shape=(self.num_envs, H.H1_GRASP_ACTION_DIM), dtype=np.float32,
        )

        self._reward = H1GraspReward(H.CATCH_PLANE_Z, self._torch_device)

        # Per-env state carried across steps.
        self.last_action = torch.zeros(
            self.num_envs, H.H1_GRASP_ACTION_DIM, device=self._torch_device
        )
        self.episode_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self._torch_device
        )
        # A monotonically advancing reset-seed counter so each autoreset draws a
        # fresh (but deterministic given the base seed) cup IC.
        self._base_seed = 0 if seed is None else int(seed)
        self._reset_counter = 0

        # Seeded RNG for sample_actions() (random-policy smoke rollouts).
        self._generator = torch.Generator(device=self._torch_device)
        if seed is not None:
            self._generator.manual_seed(int(seed))
            self.single_action_space.seed(int(seed))

    # -- properties ---------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return H.H1_GRASP_ACTION_DIM

    @property
    def obs_dim(self) -> int:
        return H.H1_GRASP_OBS_DIM

    # -- obs assembly -------------------------------------------------------
    def _export_dicts(self):
        """ONE bulk device->host export per buffer (export_obs + read_cup). Both the
        obs assembly and the reward read from these SAME two dicts -- per the A2
        throughput finding the per-step cost is host-bound, so exporting once (not
        twice) keeps the PPO wall-time at the measured feasibility headline."""
        return self._world.export_obs(), self._world.read_cup()

    def _compute_obs(self, obs_dict=None, cup_dict=None) -> torch.Tensor:
        """Assemble the (N,27) obs on host from the export dicts, move to cuda. The
        dicts are fetched once per step and passed in; if omitted (reset path) they
        are fetched here. ``last_action`` (cuda) round-trips to host for the obs tail."""
        if obs_dict is None or cup_dict is None:
            obs_dict, cup_dict = self._export_dicts()
        last_action_np = self.last_action.detach().cpu().numpy()
        obs_np = H.assemble_obs(obs_dict, cup_dict, last_action_np)
        return torch.as_tensor(obs_np, device=self._torch_device)

    def _reward_tensors(self, obs_dict, cup_dict):
        """From the already-fetched export dicts -> the cuda tensors the reward +
        termination need (cup_z, cup lin/ang vel, per-finger impulse). NO extra
        device->host export."""
        cz = torch.as_tensor(cup_dict["cup_z"], device=self._torch_device)
        cup_lin = torch.as_tensor(cup_dict["cup_vel"][:, 0:3].copy(),
                                  device=self._torch_device)
        cup_ang = torch.as_tensor(cup_dict["cup_vel"][:, 3:6].copy(),
                                  device=self._torch_device)
        imp = torch.as_tensor(obs_dict["finger_normal_impulse"],
                              device=self._torch_device)
        return cz, cup_lin, cup_ang, imp

    # -- action mapping -----------------------------------------------------
    def _to_finger_force(self, actions: torch.Tensor) -> np.ndarray:
        """Clip the [-1,1] action and map to a HOST (N,2) float32 per-finger drive
        force. POSITIVE = close/squeeze."""
        clipped = actions.to(self._torch_device, dtype=torch.float32).clamp(
            -ACTION_SPACE_LIMIT, ACTION_SPACE_LIMIT
        )
        self.last_action = clipped
        force = (clipped * FINGER_FORCE_SCALE).detach().cpu().numpy()
        return np.ascontiguousarray(force, dtype=np.float32)

    # -- gym API ------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        """Reset ALL envs to a fresh seed-randomized IC and return ``(obs, info)``."""
        if seed is not None:
            self._base_seed = int(seed)
            self._generator.manual_seed(int(seed))
            self.single_action_space.seed(int(seed))
            self._reset_counter = 0
        ids = np.arange(self.num_envs, dtype=np.uint32)
        self._world.reset_envs(ids, seed=self._next_seed())
        nuka.sync()
        self.last_action.zero_()
        self.episode_step.zero_()
        obs = self._compute_obs()
        return obs, {}

    def step(self, actions: torch.Tensor):
        """One control step. ``actions``: (N,2) on cuda. Returns the gymnasium
        5-tuple (obs, reward, terminated, truncated, info), all tensors on cuda."""
        force_np = self._to_finger_force(actions)            # sets self.last_action
        self._world.set_actions(force_np.reshape(-1))
        for _ in range(self.decimation):
            self._world.step()
        self.episode_step += 1

        # ONE export of the post-step state; reward AND obs both read these dicts.
        obs_dict, cup_dict = self._export_dicts()
        cz, cup_lin, cup_ang, imp = self._reward_tensors(obs_dict, cup_dict)
        reward = self._reward.compute(
            cup_z=cz, cup_lin_vel=cup_lin, cup_ang_vel=cup_ang,
            finger_impulse=imp, action=self.last_action,
        ).to(dtype=torch.float32)
        terminated = cz < self.termination_z
        truncated = self.episode_step >= self.max_episode_length
        info = {"time_outs": truncated}

        obs = self._compute_obs(obs_dict, cup_dict)

        # AUTORESET (legged_gym / Brax convention): a done env is reset in-place and
        # its POST-reset obs returned, so the experience buffer pairs a fresh-episode
        # obs with the (terminal) reward computed above.
        done = terminated | truncated
        if bool(done.any()):
            done_ids = torch.nonzero(done, as_tuple=False).flatten().to(torch.int32)
            self._world.reset_envs(done_ids.cpu().numpy().astype(np.uint32),
                                   seed=self._next_seed())
            nuka.sync()
            self.last_action[done] = 0.0
            self.episode_step[done] = 0
            obs_reset = self._compute_obs()
            obs = obs.clone()
            obs[done] = obs_reset[done]

        return obs, reward, terminated, truncated, info

    # -- seed bookkeeping ---------------------------------------------------
    def _next_seed(self) -> int:
        """A fresh, deterministic-given-base reset seed per call (so autoresets draw
        varied cup ICs but the whole run is reproducible)."""
        s = (self._base_seed * 1_000_003 + self._reset_counter) & 0xFFFFFFFFFFFFFFFF
        self._reset_counter += 1
        return s

    # -- random policy helper (seeded) --------------------------------------
    def sample_actions(self) -> torch.Tensor:
        """A seeded random action batch in [-1,1] (N,2) on cuda."""
        return (
            torch.rand(self.num_envs, H.H1_GRASP_ACTION_DIM,
                       generator=self._generator, device=self._torch_device) * 2.0 - 1.0
        )

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
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


def make_env(num_envs: int, *, device=None, **kw) -> H1GraspEnv:
    """Construct the H1 cup-grasp env (the rl_games _ENV_FACTORIES entry point).

    Mirrors the go2/h1_stand make_env signature: ``num_envs`` positional, ``device``
    an open nuka.Device (created+owned by the env if None), the rest forwarded to
    :class:`H1GraspEnv` (dt, friction_mu, gravity_z, cup_start_z_offset,
    termination_z, episode_length, seed, cup_asset_path)."""
    return H1GraspEnv(num_envs, device=device, **kw)
