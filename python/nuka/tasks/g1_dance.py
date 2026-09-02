"""G1 pretrained motion-policy playback over Nuka articulation controls."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

import nuka
from .g1_motion_contract import G1_JOINT_NAMES, G1_OBS_DIM, G1Motion, load_g1_motion, validate_g1_world_names


G1_DEFAULT_Q = np.zeros(29, dtype=np.float32)
for _i, _name in enumerate(G1_JOINT_NAMES):
    if _name.endswith("hip_pitch_joint"):
        G1_DEFAULT_Q[_i] = -0.312
    elif _name.endswith("knee_joint"):
        G1_DEFAULT_Q[_i] = 0.669
    elif _name.endswith("ankle_pitch_joint"):
        G1_DEFAULT_Q[_i] = -0.363
    elif _name.endswith("elbow_joint"):
        G1_DEFAULT_Q[_i] = 0.6
    elif _name == "left_shoulder_roll_joint":
        G1_DEFAULT_Q[_i] = 0.2
    elif _name == "left_shoulder_pitch_joint":
        G1_DEFAULT_Q[_i] = 0.2
    elif _name == "right_shoulder_roll_joint":
        G1_DEFAULT_Q[_i] = -0.2
    elif _name == "right_shoulder_pitch_joint":
        G1_DEFAULT_Q[_i] = 0.2

G1_ACTION_SCALE = np.array([
    0.5475464629911068 if ("hip_pitch" in n or "hip_yaw" in n or n == "waist_yaw_joint") else
    0.35066146637882434 if ("hip_roll" in n or "knee" in n) else
    0.07450087032950714 if ("wrist_pitch" in n or "wrist_yaw" in n) else
    0.43857731392336724 for n in G1_JOINT_NAMES
], dtype=np.float32)
G1_KP = np.array([
    14.25062309787429 if any(x in n for x in ("elbow", "shoulder", "wrist_roll")) else
    40.17923863450712 if ("hip_pitch" in n or "hip_yaw" in n or n == "waist_yaw_joint") else
    99.09842777666111 if ("hip_roll" in n or "knee" in n) else
    16.77832748089279 if ("wrist_pitch" in n or "wrist_yaw" in n) else
    28.50124619574858 for n in G1_JOINT_NAMES
], dtype=np.float32)
G1_KD = np.array([
    0.907222843292423 if any(x in n for x in ("elbow", "shoulder", "wrist_roll")) else
    2.557889775413375 if ("hip_pitch" in n or "hip_yaw" in n or n == "waist_yaw_joint") else
    6.308801853496639 if ("hip_roll" in n or "knee" in n) else
    1.06814150219 if ("wrist_pitch" in n or "wrist_yaw" in n) else
    1.814445686584846 for n in G1_JOINT_NAMES
], dtype=np.float32)
G1_TORQUE_LIMIT = np.array([
    25.0 if any(x in n for x in ("elbow", "shoulder", "wrist_roll")) else
    88.0 if ("hip_pitch" in n or "hip_yaw" in n or n == "waist_yaw_joint") else
    139.0 if ("hip_roll" in n or "knee" in n) else
    5.0 if ("wrist_pitch" in n or "wrist_yaw" in n) else
    50.0 for n in G1_JOINT_NAMES
], dtype=np.float32)


def _rotation_matrix_from_wxyz(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), -1).reshape(q.shape[:-1] + (3, 3))


def _relative_position(position: torch.Tensor, base_pos: torch.Tensor, base_quat: torch.Tensor) -> torch.Tensor:
    rotation = _rotation_matrix_from_wxyz(base_quat)
    return torch.einsum("bij,bj->bi", rotation.transpose(1, 2), position - base_pos)


def _relative_rotation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return a^T b without routing tiny 3x3 batches through cuBLAS."""
    columns = []
    for row in range(3):
        for column in range(3):
            columns.append((a[:, :, row] * b[:, :, column]).sum(dim=1))
    return torch.stack(columns, dim=1).reshape(-1, 3, 3)


def _sixd(rotation: torch.Tensor) -> torch.Tensor:
    return rotation[..., :, :2].reshape(rotation.shape[:-2] + (6,))


class G1OnnxActor:
    """Official G1 Moves actor executed by ONNX Runtime.

    Observation normalization is embedded in the ONNX graph. The raw output is
    intentionally not clipped; mjlab applies the serialized per-joint scale and
    default-pose offset after inference.
    """

    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name

    @classmethod
    def load(cls, checkpoint: str | Path, device: torch.device | None = None) -> "G1OnnxActor":
        del device
        session = ort.InferenceSession(str(checkpoint), providers=["CPUExecutionProvider"])
        input_info = session.get_inputs()[0]
        output_info = session.get_outputs()[0]
        if input_info.name != "obs" or input_info.shape[-1] != G1_OBS_DIM:
            raise ValueError(f"unexpected G1 ONNX input: {input_info.name} {input_info.shape}")
        if output_info.name != "actions" or output_info.shape[-1] != 29:
            raise ValueError(f"unexpected G1 ONNX output: {output_info.name} {output_info.shape}")
        return cls(session)

    @torch.no_grad()
    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.shape[-1] != G1_OBS_DIM:
            raise ValueError(f"G1 actor expects {G1_OBS_DIM} observations, got {observation.shape[-1]}")
        host_obs = observation.detach().to(device="cpu", dtype=torch.float32).numpy()
        actions = self.session.run([self.output_name], {self.input_name: host_obs})[0]
        return torch.as_tensor(actions, device=observation.device, dtype=torch.float32)


# Compatibility name used by the original demo entry point.
G1Actor = G1OnnxActor


class G1DanceController:
    """Single-env G1 controller using engine PD by default and torque for diagnostics."""

    def __init__(
        self,
        scene: str,
        motion: G1Motion,
        device: nuka.Device,
        actor: G1OnnxActor | None = None,
        *,
        control_backend: str = "external_torque",
        dt: float = 0.005,
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
    ):
        if control_backend not in ("external_torque", "engine_pd"):
            raise ValueError("control_backend must be external_torque or engine_pd")
        self.device = device
        self.motion = motion
        self.actor = actor
        self.control_backend = control_backend
        self.dt = float(dt)
        self.kp_scale = float(kp_scale)
        self.kd_scale = float(kd_scale)
        self.policy_stride = max(1, int(round(0.02 / self.dt)))
        mode = nuka.CONTROL_MODE_TORQUE if control_backend == "external_torque" else nuka.CONTROL_MODE_PD_POSITION
        self.world = nuka.World.create_from_scene(
            device, scene, env_count=1, dt=self.dt,
            determinism=nuka.DETERMINISM_STRONG,
            control_mode=mode,
            contact_family=1,
            heightfield_terrain_type=0,
        )
        validate_g1_world_names(self.world.dof_names())
        if self.world.base_link_count != 30 or self.world.action_dim != 29:
            raise RuntimeError(f"G1 topology mismatch: {self.world.base_link_count}/{self.world.action_dim}")
        self.q = torch.from_dlpack(self.world.buffer_view(nuka.JOINT_POSITION)).view(1, 30)
        self.qd = torch.from_dlpack(self.world.buffer_view(nuka.JOINT_VELOCITY)).view(1, 30)
        self.base = torch.from_dlpack(self.world.buffer_view(nuka.BASE_POSE)).view(1, 7)
        self.vel = torch.from_dlpack(self.world.buffer_view(nuka.LINK_VELOCITY)).view(1, 30, 6)
        if control_backend == "external_torque":
            self.torque = torch.from_dlpack(self.world.buffer_view(nuka.TORQUE_INPUT)).view(1, 30)
            self.drive_target = self.drive_kp = self.drive_kd = self.drive_limit = None
        else:
            self.torque = None
            self.drive_target = torch.from_dlpack(self.world.buffer_view(nuka.DRIVE_TARGET)).view(1, 30)
            self.drive_kp = torch.from_dlpack(self.world.buffer_view(nuka.DRIVE_STIFFNESS)).view(1, 30)
            self.drive_kd = torch.from_dlpack(self.world.buffer_view(nuka.DRIVE_DAMPING)).view(1, 30)
            self.drive_limit = torch.from_dlpack(self.world.buffer_view(nuka.DRIVE_FORCE_LIMIT)).view(1, 30)
        self.prev_action = torch.zeros((1, 29), device=self.q.device)
        self._policy_action = torch.zeros((1, 29), device=self.q.device)
        self._target = torch.as_tensor(G1_DEFAULT_Q, device=self.q.device).view(1, 29).clone()
        self._obs_abs_max = 0.0
        self._physics_step = 0
        self.torso_slot = int(self.world.dof_index("waist_pitch_joint")[0])
        self._reset()

    def _reset(self) -> None:
        frame = self.motion.sample(0.0)
        self.q[:, 0] = 0.0
        self.q[:, 1:] = torch.as_tensor(frame["joint_position"], device=self.q.device)
        self.qd[:, 0] = 0.0
        self.qd[:, 1:] = torch.as_tensor(frame["joint_velocity"], device=self.q.device)
        self.base[:, :3] = torch.as_tensor(frame["body_position_world"][0], device=self.base.device)
        self.base[:, 3:] = torch.as_tensor(
            frame["body_quaternion_wxyz"][0], device=self.base.device)
        self.vel.zero_()
        base_rotation = _rotation_matrix_from_wxyz(self.base[:, 3:])
        angular_world = torch.as_tensor(
            frame["body_angular_velocity_world"][0], device=self.vel.device).view(1, 3)
        linear_world = torch.as_tensor(
            frame["body_linear_velocity_world"][0], device=self.vel.device).view(1, 3)
        self.vel[:, 0, 0:3] = torch.einsum("bij,bj->bi", base_rotation.transpose(1, 2), angular_world)
        self.vel[:, 0, 3:6] = torch.einsum("bij,bj->bi", base_rotation.transpose(1, 2), linear_world)
        if self.torque is not None:
            self.torque.zero_()
        else:
            self.drive_target.copy_(self.q)
            self.drive_kp.zero_()
            self.drive_kd.zero_()
            self.drive_limit.zero_()
            self.drive_kp[:, 1:] = torch.as_tensor(G1_KP * self.kp_scale, device=self.q.device)
            self.drive_kd[:, 1:] = torch.as_tensor(G1_KD * self.kd_scale, device=self.q.device)
            self.drive_limit[:, 1:] = torch.as_tensor(G1_TORQUE_LIMIT, device=self.q.device)
        nuka.sync()

    def observation(self, time_s: float) -> torch.Tensor:
        ref = self.motion.sample(time_s)
        ref_q = torch.as_tensor(ref["joint_position"], device=self.q.device).view(1, 29)
        ref_qd = torch.as_tensor(ref["joint_velocity"], device=self.q.device).view(1, 29)
        ref_torso_pos = torch.as_tensor(ref["body_position_world"][self.torso_slot], device=self.q.device).view(1, 3)
        ref_torso_quat = torch.as_tensor(
            ref["body_quaternion_wxyz"][self.torso_slot], device=self.q.device).view(1, 4)
        current_torso = self._link_pose()[self.torso_slot].view(1, 7)
        anchor_pos = _relative_position(ref_torso_pos, current_torso[:, :3], current_torso[:, 3:])
        anchor_rotation = _rotation_matrix_from_wxyz(ref_torso_quat)
        torso_rotation = _rotation_matrix_from_wxyz(current_torso[:, 3:])
        anchor_ori = _sixd(_relative_rotation(torso_rotation, anchor_rotation))
        obs = torch.cat((
            ref_q, ref_qd, anchor_pos, anchor_ori,
            self.vel[:, 0, 3:6], self.vel[:, 0, 0:3],
            self.q[:, 1:] - torch.as_tensor(G1_DEFAULT_Q, device=self.q.device).view(1, 29),
            self.qd[:, 1:], self.prev_action,
        ), dim=-1)
        return obs

    def _link_pose(self) -> torch.Tensor:
        return torch.from_dlpack(self.world.buffer_view(nuka.ARTICULATION_LINK_POSE)).view(1, 30, 7)[0]

    def step(self, time_s: float, reference_only: bool = False) -> dict[str, float]:
        if self._physics_step % self.policy_stride == 0:
            if reference_only or self.actor is None:
                self._target.copy_(torch.as_tensor(
                    self.motion.sample(time_s)["joint_position"], device=self.q.device).view(1, 29))
                self._policy_action.zero_()
            else:
                observation = self.observation(time_s)
                self._obs_abs_max = float(observation.abs().max().item())
                self._policy_action.copy_(self.actor(observation))
                self._target.copy_(
                    torch.as_tensor(G1_DEFAULT_Q, device=self.q.device).view(1, 29)
                    + self._policy_action * torch.as_tensor(G1_ACTION_SCALE, device=self.q.device)
                )
            self.prev_action.copy_(self._policy_action)
        target = self._target
        kp = torch.as_tensor(G1_KP * self.kp_scale, device=self.q.device).view(1, 29)
        kd = torch.as_tensor(G1_KD * self.kd_scale, device=self.q.device).view(1, 29)
        limit = torch.as_tensor(G1_TORQUE_LIMIT, device=self.q.device).view(1, 29)
        tau = (kp * (target - self.q[:, 1:]) - kd * self.qd[:, 1:]).clamp(-limit, limit)
        if self.torque is not None:
            self.torque[:, 0] = 0.0
            self.torque[:, 1:] = tau
        else:
            self.drive_target[:, 0] = 0.0
            self.drive_target[:, 1:] = target
        self.world.step()
        self._physics_step += 1
        return {
            "joint_rmse_rad": float(torch.sqrt(torch.mean((self.q[:, 1:] - target) ** 2)).item()),
            "root_z_m": float(self.base[0, 2].item()),
            "root_speed_mps": float(torch.linalg.vector_norm(self.vel[0, 0, 3:6]).item()),
            "observation_abs_max": self._obs_abs_max,
            "action_abs_max": float(self._policy_action.abs().max().item()),
            "target_abs_max_rad": float(target.abs().max().item()),
            "finite": float(torch.isfinite(self.q).all() and torch.isfinite(self.base).all()),
        }

    def close(self) -> None:
        if getattr(self, "world", None) is not None:
            self.world.destroy()
            self.world = None

 # wsl -d Ubuntu-24.04 -- bash -lc 'source /root/nuka-vla/bin/activate && cd /mnt/c/Softwares/code/Nuka-Physics && PYTHONUNBUFFERED=1 python examples/demo/libero_pi05_play.py --control-backend osc --seconds 20 --execute-steps 10 --max-queries 2 --render-quality preview --skip-video --out out/libero/pi05_black_bowl_osc_diag'

# wsl -d Ubuntu-24.04 -- bash -lc 'tail -f /mnt/c/Softwares/code/Nuka-Physics/out/libero/pi05_black_bowl_osc_diag/rollout_progress.jsonl'
