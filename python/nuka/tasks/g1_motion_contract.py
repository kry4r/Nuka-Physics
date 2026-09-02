"""Locked Unitree G1 29DoF motion and 160-observation contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


G1_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
G1_ACTION_DIM = 29
G1_OBS_DIM = 160
G1_BODY_COUNT = 30

OBS_SLICES = {
    "reference_joint_position": slice(0, 29),
    "reference_joint_velocity": slice(29, 58),
    "anchor_position_body": slice(58, 61),
    "anchor_orientation_6d_body": slice(61, 67),
    "base_linear_velocity": slice(67, 70),
    "base_angular_velocity": slice(70, 73),
    "joint_position_relative": slice(73, 102),
    "joint_velocity": slice(102, 131),
    "previous_action": slice(131, 160),
}


@dataclass(frozen=True)
class G1Motion:
    fps: float
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    body_position_world: np.ndarray
    body_quaternion_wxyz: np.ndarray
    body_linear_velocity_world: np.ndarray
    body_angular_velocity_world: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.joint_position.shape[0])

    @property
    def duration(self) -> float:
        return (self.frame_count - 1) / self.fps

    def sample(self, time_s: float) -> dict[str, np.ndarray]:
        phase = np.clip(float(time_s) * self.fps, 0.0, self.frame_count - 1.0)
        lo = int(np.floor(phase))
        hi = min(lo + 1, self.frame_count - 1)
        alpha = np.float32(phase - lo)

        def lerp(values: np.ndarray) -> np.ndarray:
            return (values[lo] * (1.0 - alpha) + values[hi] * alpha).astype(np.float32)

        quat = lerp(self.body_quaternion_wxyz)
        quat /= np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8)
        return {
            "joint_position": lerp(self.joint_position),
            "joint_velocity": lerp(self.joint_velocity),
            "body_position_world": lerp(self.body_position_world),
            "body_quaternion_wxyz": quat,
            "body_linear_velocity_world": lerp(self.body_linear_velocity_world),
            "body_angular_velocity_world": lerp(self.body_angular_velocity_world),
        }


def load_g1_motion(path: str | Path) -> G1Motion:
    source = Path(path)
    with np.load(source) as data:
        required = {
            "fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
            "body_lin_vel_w", "body_ang_vel_w",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"{source}: missing arrays {missing}")
        fps_values = np.asarray(data["fps"], dtype=np.float64).reshape(-1)
        if fps_values.size != 1 or not np.isfinite(fps_values[0]) or fps_values[0] <= 0:
            raise ValueError(f"{source}: fps must contain one finite positive value")
        arrays = {
            "joint_position": np.asarray(data["joint_pos"], dtype=np.float32),
            "joint_velocity": np.asarray(data["joint_vel"], dtype=np.float32),
            "body_position_world": np.asarray(data["body_pos_w"], dtype=np.float32),
            "body_quaternion_wxyz": np.asarray(data["body_quat_w"], dtype=np.float32),
            "body_linear_velocity_world": np.asarray(data["body_lin_vel_w"], dtype=np.float32),
            "body_angular_velocity_world": np.asarray(data["body_ang_vel_w"], dtype=np.float32),
        }
    frames = arrays["joint_position"].shape[0]
    expected = {
        "joint_position": (frames, G1_ACTION_DIM),
        "joint_velocity": (frames, G1_ACTION_DIM),
        "body_position_world": (frames, G1_BODY_COUNT, 3),
        "body_quaternion_wxyz": (frames, G1_BODY_COUNT, 4),
        "body_linear_velocity_world": (frames, G1_BODY_COUNT, 3),
        "body_angular_velocity_world": (frames, G1_BODY_COUNT, 3),
    }
    for name, values in arrays.items():
        if values.shape != expected[name]:
            raise ValueError(f"{source}: {name} shape {values.shape}, expected {expected[name]}")
        if not np.isfinite(values).all():
            raise ValueError(f"{source}: {name} contains non-finite values")
    return G1Motion(float(fps_values[0]), **arrays)


def xyzw_to_wxyz(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion)
    return quaternion[..., (3, 0, 1, 2)]


def validate_g1_world_names(cooked_names: list[str] | tuple[str, ...]) -> None:
    expected = ("floating_base_joint",) + G1_JOINT_NAMES
    if tuple(cooked_names) != expected:
        raise ValueError(f"G1 cooked DOF order mismatch\nexpected={expected}\nactual={tuple(cooked_names)}")
