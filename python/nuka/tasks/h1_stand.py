"""H1 reduced-DOF torque standing task for S4 phase 1.

This env deliberately does NOT subclass :class:`nuka.gym.NukaGymEnv`: that base is
Go2-specific (12 actions / 48 obs / PD targets). The H1 standing task drives the
reduced 16-DOF training MJCF directly in torque mode and preserves the locked
train/deploy bridge contract:

  obs[32] = base quat(4) + base spatial vel(6, omega-first) + base xy relative to
            the fixed support-polygon centre + leg q[10] + leg qd[10]
  action[10] = normalized leg torques in q[1:11] order, scaled per joint to the
               H1 MJCF torque limits [200,200,200,300,40] per leg.

The reward is a phase-1 static-standing signal only. Later S4 phases should keep
this reward structure stable and add disturbances/curriculum outside this file's
core obs/action contract.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

import nuka


REPO = pathlib.Path(__file__).resolve().parents[3]
REDUCED_MJCF = REPO / ".nuka-assets/newton_assets/unitree_h1/mjcf/h1_reduced_train.xml"
GEN_REDUCED = REPO / "python/spikes/gen_h1_reduced_train.py"

H1_BLC = 11
H1_ACTION_DIM = 10
H1_OBS_DIM = 32
OBS_CLIP = 100.0
ACTION_SPACE_LIMIT = 1.0
ACTION_CLIP = 1.0

POLY_CX = 0.0770
POLY_CY = 0.0

STANCE = (-0.40, 0.70, -0.30)  # hip_pitch, knee, ankle.
STANCE_Q = torch.tensor(
    [0.0, 0.0, STANCE[0], STANCE[1], STANCE[2],
     0.0, 0.0, STANCE[0], STANCE[1], STANCE[2]],
    dtype=torch.float32,
    device="cuda",
)
LEG_TORQUE_LIMITS = torch.tensor(
    [200.0, 200.0, 200.0, 300.0, 40.0,
     200.0, 200.0, 200.0, 300.0, 40.0],
    dtype=torch.float32,
    device="cuda",
)

# Seat helpers mirror python/spikes/h1_standing_pd_gate.py / h1_balance_gate.py.
PELVIS_Z0 = 1.1
HIP_YAW_OFF_Z = -0.1742
HIP_ROLL_OFF_X = 0.039468
KNEE_OFF_XZ = (0.0, -0.4)
ANKLE_OFF_XZ = (0.0, -0.4)
FOOT_BOTTOM_Z = -0.055
FOOT_SPHERE_R = 0.025
FOOT_TOE_X = 0.10
FOOT_HEEL_X = -0.10


# ---------------------------------------------------------------------------
# Asset + bridge obs helpers
# ---------------------------------------------------------------------------
def _ensure_reduced_asset() -> pathlib.Path:
    """Return the generated reduced H1 MJCF, generating it if absent."""
    if REDUCED_MJCF.exists():
        return REDUCED_MJCF
    if not GEN_REDUCED.exists():
        raise FileNotFoundError(
            f"missing H1 reduced asset {REDUCED_MJCF} and generator {GEN_REDUCED}"
        )
    spikes_dir = str(GEN_REDUCED.parent)
    if spikes_dir not in sys.path:
        sys.path.insert(0, spikes_dir)
    runpy.run_path(str(GEN_REDUCED), run_name="__main__")
    if not REDUCED_MJCF.exists():
        raise FileNotFoundError(f"generator did not create {REDUCED_MJCF}")
    return REDUCED_MJCF


def _read(world, field, env_count: int, width: int) -> torch.Tensor:
    return torch.from_dlpack(world.buffer_view(field)).view(env_count, width)


def assemble_h1_obs(
    world,
    env_count: int,
    *,
    poly_center: tuple[float, float] = (POLY_CX, POLY_CY),
) -> torch.Tensor:
    """Build the locked 32-wide H1 bridge observation from live world buffers."""
    bp = _read(world, nuka.BASE_POSE, env_count, 7)
    lv = torch.from_dlpack(world.buffer_view(nuka.LINK_VELOCITY)).view(
        env_count, H1_BLC, 6
    )
    q = _read(world, nuka.JOINT_POSITION, env_count, H1_BLC)
    qd = _read(world, nuka.JOINT_VELOCITY, env_count, H1_BLC)
    obs = torch.zeros(env_count, H1_OBS_DIM, device="cuda", dtype=torch.float32)
    obs[:, 0:4] = bp[:, 3:7]
    obs[:, 4:10] = lv[:, 0, :]
    obs[:, 10] = bp[:, 0] - float(poly_center[0])
    obs[:, 11] = bp[:, 1] - float(poly_center[1])
    obs[:, 12:22] = q[:, 1:11]
    obs[:, 22:32] = qd[:, 1:11]
    return obs


def _foot_z_under_pose(
    hip_pitch: float,
    knee: float,
    ankle: float,
    toe_x: float = FOOT_TOE_X,
    heel_x: float = FOOT_HEEL_X,
) -> float:
    """Foot-bottom z relative to base z for the sagittal H1 stance."""
    th_hip = hip_pitch
    th_knee = hip_pitch + knee
    th_ankle = hip_pitch + knee + ankle

    def rot_y(point, theta):
        x, z = point
        c, s = np.cos(theta), np.sin(theta)
        return c * x + s * z, -s * x + c * z

    pos_x = HIP_ROLL_OFF_X
    pos_z = HIP_YAW_OFF_Z
    seg_x, seg_z = rot_y(KNEE_OFF_XZ, th_hip)
    pos_x += seg_x
    pos_z += seg_z
    seg_x, seg_z = rot_y(ANKLE_OFF_XZ, th_knee)
    pos_x += seg_x
    pos_z += seg_z
    toe = rot_y((toe_x, FOOT_BOTTOM_Z), th_ankle)
    heel = rot_y((heel_x, FOOT_BOTTOM_Z), th_ankle)
    return min(pos_z + toe[1], pos_z + heel[1]) - FOOT_SPHERE_R


def nominal_base_height() -> float:
    ground = PELVIS_Z0 + _foot_z_under_pose(0.0, 0.0, 0.0)
    return ground - _foot_z_under_pose(*STANCE, FOOT_TOE_X, FOOT_HEEL_X)


def _projected_up_cos(base_quat_wxyz: torch.Tensor) -> torch.Tensor:
    qx = base_quat_wxyz[:, 1]
    qy = base_quat_wxyz[:, 2]
    return (1.0 - 2.0 * (qx * qx + qy * qy)).clamp(-1.0, 1.0)


def _tilt_deg(base_quat_wxyz: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(torch.arccos(_projected_up_cos(base_quat_wxyz)))


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------
class H1StandEnv:
    """Vectorized gymnasium-style H1 torque standing env, all tensors on CUDA."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        scene: str,
        num_envs: int,
        *,
        device: "nuka.Device | None" = None,
        dt: float = 0.005,
        decimation: int = 4,
        determinism: int = nuka.DETERMINISM_STRONG,
        termination_height: float = 0.55,
        termination_tilt_deg: float = 60.0,
        termination_qvel: float = 500.0,
        episode_length_s: float = 20.0,
        disturbance_interval_steps: int = 0,
        disturbance_linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        seed: int | None = None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.decimation = int(decimation)
        self.dt = float(dt)
        self.termination_height = float(termination_height)
        self.termination_tilt_deg = float(termination_tilt_deg)
        self.termination_qvel = float(termination_qvel)
        self.max_episode_length = max(1, int(round(episode_length_s / (dt * decimation))))
        self._torch_device = torch.device("cuda")
        self.disturbance_interval_steps = int(disturbance_interval_steps)
        self.disturbance_linear_velocity = torch.tensor(
            disturbance_linear_velocity,
            device=self._torch_device,
            dtype=torch.float32,
        )
        if self.disturbance_linear_velocity.shape != (3,):
            raise ValueError("disturbance_linear_velocity must have exactly 3 components")
        self.poly_center = (POLY_CX, POLY_CY)
        self.nominal_base_z = nominal_base_height()

        self._owns_device = device is None
        self._device = device if device is not None else nuka.Device.create(0)
        self._world = nuka.World.create_from_scene(
            self._device,
            scene,
            self.num_envs,
            self.dt,
            determinism=determinism,
            control_mode=nuka.CONTROL_MODE_TORQUE,
        )
        assert self._world.base_link_count == H1_BLC, (
            f"H1StandEnv expects reduced H1 base_link_count={H1_BLC}, got "
            f"{self._world.base_link_count}"
        )
        assert self._world.action_dim == H1_ACTION_DIM, (
            f"H1StandEnv expects action_dim={H1_ACTION_DIM}, got {self._world.action_dim}"
        )

        self.single_observation_space = spaces.Box(
            low=-OBS_CLIP, high=OBS_CLIP, shape=(H1_OBS_DIM,), dtype=np.float32
        )
        self.single_action_space = spaces.Box(
            low=-ACTION_SPACE_LIMIT,
            high=ACTION_SPACE_LIMIT,
            shape=(H1_ACTION_DIM,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-OBS_CLIP,
            high=OBS_CLIP,
            shape=(self.num_envs, H1_OBS_DIM),
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-ACTION_SPACE_LIMIT,
            high=ACTION_SPACE_LIMIT,
            shape=(self.num_envs, H1_ACTION_DIM),
            dtype=np.float32,
        )

        self.torque_limits = LEG_TORQUE_LIMITS.clone()
        self.stance_q = STANCE_Q.clone()
        self.last_action = torch.zeros(
            self.num_envs, H1_ACTION_DIM, device=self._torch_device, dtype=torch.float32
        )
        self._prev_action = torch.zeros_like(self.last_action)
        self.episode_step = torch.zeros(
            self.num_envs, dtype=torch.long, device=self._torch_device
        )
        self._generator = torch.Generator(device=self._torch_device)
        if seed is not None:
            self._generator.manual_seed(int(seed))
            self.single_action_space.seed(int(seed))

    @property
    def action_dim(self) -> int:
        return H1_ACTION_DIM

    @property
    def obs_dim(self) -> int:
        return H1_OBS_DIM

    def _seat_pose(self, mask: "torch.Tensor | None") -> None:
        q = _read(self._world, nuka.JOINT_POSITION, self.num_envs, H1_BLC)
        qd = _read(self._world, nuka.JOINT_VELOCITY, self.num_envs, H1_BLC)
        bp = _read(self._world, nuka.BASE_POSE, self.num_envs, 7)
        tq = _read(self._world, nuka.TORQUE_INPUT, self.num_envs, H1_BLC)
        if mask is None:
            q[:, 0] = 0.0
            q[:, 1:11] = self.stance_q.view(1, H1_ACTION_DIM)
            qd[:] = 0.0
            bp[:, 0:2] = 0.0
            bp[:, 2] = self.nominal_base_z
            bp[:, 3] = 1.0
            bp[:, 4:7] = 0.0
            tq[:] = 0.0
        else:
            q[mask, 0] = 0.0
            q[mask, 1:11] = self.stance_q.view(1, H1_ACTION_DIM)
            qd[mask] = 0.0
            bp[mask, 0:2] = 0.0
            bp[mask, 2] = self.nominal_base_z
            bp[mask, 3] = 1.0
            bp[mask, 4:7] = 0.0
            tq[mask] = 0.0
        nuka.sync()

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._generator.manual_seed(int(seed))
            self.single_action_space.seed(int(seed))
        self._world.reset()
        nuka.sync()
        self._seat_pose(None)
        self.last_action.zero_()
        self._prev_action.zero_()
        self.episode_step.zero_()
        return self.compute_obs(), {}

    def _apply_disturbance(self) -> None:
        if self.disturbance_interval_steps <= 0:
            return
        if bool((self.episode_step % self.disturbance_interval_steps == 0).any()):
            lv = torch.from_dlpack(self._world.buffer_view(nuka.LINK_VELOCITY)).view(
                self.num_envs, H1_BLC, 6
            )
            mask = (self.episode_step % self.disturbance_interval_steps == 0).view(
                self.num_envs, 1
            )
            lv[:, 0, 3:6] += mask.to(torch.float32) * self.disturbance_linear_velocity.view(1, 3)

    def _write_torque_action(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.to(self._torch_device, dtype=torch.float32).clamp(
            -ACTION_CLIP, ACTION_CLIP
        )
        tq = _read(self._world, nuka.TORQUE_INPUT, self.num_envs, H1_BLC)
        tq[:, 0] = 0.0
        tq[:, 1:11] = actions * self.torque_limits.view(1, H1_ACTION_DIM)
        return actions

    def step(self, actions: torch.Tensor):
        clipped = self._write_torque_action(actions)
        self.last_action = clipped
        self._world.step_n(self.decimation)
        self.episode_step += 1
        self._apply_disturbance()

        obs = self.compute_obs()
        reward = self.compute_reward()
        terminated = self.compute_terminated()
        truncated = self.episode_step >= self.max_episode_length
        info: dict = {}

        done = terminated | truncated
        if bool(done.any()):
            done_ids = torch.nonzero(done, as_tuple=False).flatten().to(torch.int32)
            self._world.reset_envs(done_ids.cpu())
            self._seat_pose(done)
            self.last_action[done] = 0.0
            self._prev_action[done] = 0.0
            self.episode_step[done] = 0
            obs_reset = self.compute_obs()
            obs = obs.clone()
            obs[done] = obs_reset[done]

        self._prev_action = self.last_action.clone()
        return obs, reward, terminated, truncated, info

    def compute_obs(self) -> torch.Tensor:
        return assemble_h1_obs(self._world, self.num_envs, poly_center=self.poly_center)

    def compute_reward(self) -> torch.Tensor:
        bp = _read(self._world, nuka.BASE_POSE, self.num_envs, 7)
        q = _read(self._world, nuka.JOINT_POSITION, self.num_envs, H1_BLC)
        qd = _read(self._world, nuka.JOINT_VELOCITY, self.num_envs, H1_BLC)
        tq = _read(self._world, nuka.TORQUE_INPUT, self.num_envs, H1_BLC)
        cf = torch.from_dlpack(self._world.buffer_view(nuka.CONTACT_FORCE)).view(
            self.num_envs, -1, 3
        )

        up_cos = _projected_up_cos(bp[:, 3:7])
        upright = torch.exp(-6.0 * (1.0 - up_cos).clamp_min(0.0))
        height = torch.exp(-((bp[:, 2] - self.nominal_base_z) ** 2) / 0.02)
        posture_err = (q[:, 1:11] - self.stance_q.view(1, H1_ACTION_DIM)).pow(2).mean(dim=1)
        posture = torch.exp(-posture_err / 0.20)
        foot_pitch = q[:, [3, 8]] + q[:, [4, 9]] + q[:, [5, 10]]
        foot_flat = torch.exp(-foot_pitch.pow(2).mean(dim=1) / 0.001)
        contacts = (cf.norm(dim=-1) > 1e-4).to(torch.float32).sum(dim=1).clamp(max=4.0) / 4.0

        qvel_pen = 0.02 * qd[:, 1:11].pow(2).mean(dim=1)
        tau_norm = tq[:, 1:11] / self.torque_limits.view(1, H1_ACTION_DIM)
        torque_pen = 0.01 * tau_norm.pow(2).mean(dim=1)
        action_rate_pen = 0.02 * (self.last_action - self._prev_action).pow(2).mean(dim=1)

        reward = (
            0.25
            + 1.00 * upright
            + 0.50 * height
            + 0.30 * posture
            + 0.20 * contacts
            + 0.50 * foot_flat
            - qvel_pen
            - torque_pen
            - action_rate_pen
        )
        return reward.clamp_min(0.0).to(torch.float32)

    def compute_terminated(self) -> torch.Tensor:
        bp = _read(self._world, nuka.BASE_POSE, self.num_envs, 7)
        q = _read(self._world, nuka.JOINT_POSITION, self.num_envs, H1_BLC)
        qd = _read(self._world, nuka.JOINT_VELOCITY, self.num_envs, H1_BLC)
        lv = torch.from_dlpack(self._world.buffer_view(nuka.LINK_VELOCITY)).view(
            self.num_envs, H1_BLC, 6
        )
        fell = bp[:, 2] < self.termination_height
        tipped = _tilt_deg(bp[:, 3:7]) > self.termination_tilt_deg
        nonfinite = (
            ~torch.isfinite(bp).all(dim=1)
            | ~torch.isfinite(q).all(dim=1)
            | ~torch.isfinite(qd).all(dim=1)
            | ~torch.isfinite(lv).flatten(1).all(dim=1)
        )
        runaway_qvel = qd[:, 1:11].abs().amax(dim=1) > self.termination_qvel
        return fell | tipped | nonfinite | runaway_qvel

    def sample_actions(self) -> torch.Tensor:
        return (
            torch.rand(
                self.num_envs,
                H1_ACTION_DIM,
                generator=self._generator,
                device=self._torch_device,
                dtype=torch.float32,
            )
            * 2.0
            - 1.0
        )

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


def make_env(num_envs: int, *, device=None, scene: str | None = None, **kw) -> H1StandEnv:
    path = str(_ensure_reduced_asset() if scene is None else scene)
    return H1StandEnv(path, num_envs, device=device, **kw)
