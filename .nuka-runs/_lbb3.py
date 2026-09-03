"""Nuka-native LIBERO Spatial black-bowl task and OSC-compatible control."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import nuka

LIBERO_TASK = "pick up the black bowl from table center and place it on the plate"
# Official LIBERO task-2 episode-0 initialization state.
LIBERO_HOME = np.array(
    [
        0.0067131381,
        -0.191884762,
        -0.0099483010,
        -2.43256078,
        -0.0399100748,
        2.19352400,
        0.801371176,
        0.020833,
        0.020833,
    ],
    dtype=np.float32,
)
# Official LIBERO task-2 episode-0 grip_site pose, measured after reset settles.
LIBERO_EEF_REFERENCE = np.array(
    [-0.20991100, -0.00928400, 1.18219700], dtype=np.float32
)
LIBERO_EEF_QUATERNION_WXYZ = np.array(
    [0.01512727, 0.99960626, 0.00277676, -0.02346881], dtype=np.float32
)
LIBERO_TARGET_BOWL = np.array(
    [-0.07500000, 0.01499801, 0.89821996], dtype=np.float32
)
LIBERO_PLATE = np.array([0.07159546, 0.20039269, 0.90233128], dtype=np.float32)

PANDA_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
    "finger_joint1",
    "finger_joint2",
)
PANDA_ARM_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float32,
)
PANDA_ARM_UPPER = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float32,
)
PANDA_KP = np.array([600, 600, 500, 500, 360, 240, 160, 80, 80], dtype=np.float32)
PANDA_KD = np.array([46, 46, 38, 38, 28, 20, 14, 7, 7], dtype=np.float32)
PANDA_FORCE_LIMIT = np.array([87, 87, 87, 87, 12, 12, 12, 20, 20], dtype=np.float32)
PANDA_PD_GAIN_SCALE = 10.0
LIBERO_OSC_KP = 150.0
LIBERO_OSC_KD = 2.0 * np.sqrt(LIBERO_OSC_KP)
CONTROL_BACKENDS = ("joint_pd", "osc")
# robosuite advances PandaGripper.format_action once per MuJoCo substep, so its
# 0.01 speed accumulates in normalized gripper space at 0.01 per 0.002 s.
LIBERO_GRIPPER_RATE_PER_SECOND = 0.01 / 0.002

# link7 -> robosuite `grip_site`, measured from the official MuJoCo FK by
# expressing (grip_site - link7) in link7's frame at LIBERO task-2 episode-0.
ROBOSUITE_EEF_LOCAL = (
    0.00075822,
    0.00609498,
    0.21117520,
    0.92378998,
    0.0,
    0.0,
    -0.38290998,
)
LIBERO_AGENT_CAMERA_LOCAL = (
    1.318613,
    0.0,
    0.698350,
    0.638018,
    0.30485,
    0.30485,
    0.638018,
)
# link7 -> robosuite `robot0_eye_in_hand`, taken from the official model: local
# position [0.03533792, -0.03537275, 0.1065] and a 180-degree rotation about
# (0.923974, 0.382456, 0). Camera-local -Z is forward.
LIBERO_WRIST_CAMERA_LOCAL = (
    0.03533792,
    -0.03537275,
    0.10650000,
    0.0,
    0.92397376,
    0.38245586,
    0.0,
)
LIBERO_AGENT_FOV = 45.0
LIBERO_WRIST_FOV = 75.0
# Replay-only camera on the wall-free +Y side, close enough to show most of the
# arm and the whole tabletop. Never part of the policy observation.
LIBERO_THIRD_PERSON_EYE = np.array(
    [-0.75, 1.55, 1.55], dtype=np.float32
)
LIBERO_THIRD_PERSON_LOOK_AT = np.array(
    [-0.22, 0.02, 0.99], dtype=np.float32
)
LIBERO_THIRD_PERSON_FOV = 58.0

LIBERO_CAMERA_SIZE = 360


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        dim=-1,
    )


def _quat_rotate(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    q_xyz = q[..., 1:]
    uv = torch.cross(q_xyz, vector, dim=-1)
    uuv = torch.cross(q_xyz, uv, dim=-1)
    return vector + 2.0 * (q[..., :1] * uv + uuv)


def _quat_to_axis_angle(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(1e-8)
    q = torch.where(q[..., :1] < 0.0, -q, q)
    vector = q[..., 1:]
    norm = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(norm, q[..., :1].clamp_min(1e-8))
    regular = vector * (angle / norm.clamp_min(1e-8))
    return torch.where(norm < 1e-6, 2.0 * vector, regular)


def _axis_angle_to_quat(vector: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.vector_norm(vector)
    half = 0.5 * angle
    scale = torch.where(
        angle > 1e-8,
        torch.sin(half) / angle.clamp_min(1e-8),
        torch.full_like(angle, 0.5),
    )
    return torch.cat((torch.cos(half).view(1), vector * scale))


def _quat_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


class LiberoBlackBowlController:
    """One Nuka Panda with the official LIBERO state and 7D action contracts."""

    def __init__(
        self,
        scene: str | Path,
        device: nuka.Device,
        *,
        dt: float = 0.005,
        control_backend: str = "joint_pd",
        render_quality: str = "high",
    ):
        if control_backend not in CONTROL_BACKENDS:
            raise ValueError(
                f"control_backend must be one of {CONTROL_BACKENDS}, got {control_backend!r}"
            )
        if render_quality not in ("preview", "high", "ultra"):
            raise ValueError(
                "render_quality must be one of ('preview', 'high', 'ultra'), "
                f"got {render_quality!r}"
            )
        self.dt = float(dt)
        self.control_backend = control_backend
        self.render_quality = render_quality
        control_mode = (
            nuka.CONTROL_MODE_OSC
            if control_backend == "osc"
            else nuka.CONTROL_MODE_PD_POSITION
        )
        self.world = nuka.World.create_from_scene(
            device,
            str(scene),
            env_count=2,
            dt=self.dt,
            determinism=nuka.DETERMINISM_STRONG,
            control_mode=control_mode,
            osc_task_link=7,
            contact_family=1,
            heightfield_terrain_type=0,
            solver_contact_margin=float(__import__('os').environ.get('NK_MARGIN', 0.0)),
            solver_max_pairs=int(__import__('os').environ.get('NK_PAIRS', 0)),
        )
        names = tuple(self.world.dof_names())
        ordered = tuple(name for name in names if name in PANDA_JOINT_NAMES)
        if self.world.base_link_count != 10 or self.world.action_dim != 9:
            raise RuntimeError(
                "LIBERO Panda topology mismatch: "
                f"links={self.world.base_link_count} actions={self.world.action_dim}"
            )
        if ordered != PANDA_JOINT_NAMES:
            raise RuntimeError(f"LIBERO Panda joint order mismatch: {ordered}")

        self.q = torch.from_dlpack(
            self.world.buffer_view(nuka.JOINT_POSITION)
        ).view(2, 10)[:1]
        self.qd = torch.from_dlpack(
            self.world.buffer_view(nuka.JOINT_VELOCITY)
        ).view(2, 10)[:1]
        self.base = torch.from_dlpack(self.world.buffer_view(nuka.BASE_POSE)).view(2, 7)[:1]
        self.link_pose = torch.from_dlpack(
            self.world.buffer_view(nuka.ARTICULATION_LINK_POSE)
        ).view(2, 10, 7)[:1][0].unsqueeze(0)
        self.rigid_pose = torch.from_dlpack(
            self.world.buffer_view(nuka.RIGID_BODY_TRANSFORM)
        ).view(-1, 7)
        self.drive_target = torch.from_dlpack(
            self.world.buffer_view(nuka.DRIVE_TARGET)
        ).view(2, 10)[:1]
        self.drive_kp = torch.from_dlpack(
            self.world.buffer_view(nuka.DRIVE_STIFFNESS)
        ).view(2, 10)[:1]
        self.drive_kd = torch.from_dlpack(
            self.world.buffer_view(nuka.DRIVE_DAMPING)
        ).view(2, 10)[:1]
        self.drive_limit = torch.from_dlpack(
            self.world.buffer_view(nuka.DRIVE_FORCE_LIMIT)
        ).view(2, 10)[:1]
        if self.control_backend == "osc":
            self.task_target = torch.from_dlpack(
                self.world.buffer_view(nuka.TASK_TARGET)
            ).view(2, 3)[:1]
            self.task_rotation_target = torch.from_dlpack(
                self.world.buffer_view(nuka.TASK_ROTATION_TARGET)
            ).view(2, 4)[:1]
            self.task_local_pose = torch.from_dlpack(
                self.world.buffer_view(nuka.TASK_LOCAL_POSE)
            ).view(2, 7)[:1]
        else:
            self.task_target = None
            self.task_rotation_target = None
            self.task_local_pose = None

        self.target_body_index = self._nearest_rigid_body(LIBERO_TARGET_BOWL)
        self.plate_body_index = self._nearest_rigid_body(LIBERO_PLATE)
        if self.target_body_index == self.plate_body_index:
            raise RuntimeError("target bowl and plate resolved to the same rigid body")
        self._gripper_action = torch.zeros(1, device=self.q.device)
        self._gripper_sign = torch.zeros(1, device=self.q.device)
        self._osc_bootstrap = False
        self._osc_task_kp = LIBERO_OSC_KP
        self._osc_task_kd = LIBERO_OSC_KD
        self._camera_tensor: torch.Tensor | None = None
        self._cameras_attached = False
        self.reset()

    def _nearest_rigid_body(self, position: np.ndarray) -> int:
        expected = torch.as_tensor(position, device=self.rigid_pose.device)
        distance = torch.linalg.vector_norm(self.rigid_pose[:, :3] - expected, dim=-1)
        index = int(torch.argmin(distance).item())
        error = float(distance[index].item())
        if error > 0.08:
            raise RuntimeError(
                f"LIBERO rigid body discovery failed for {position.tolist()}: {error:.3f}m"
            )
        return index

    def reset(self, state: Sequence[float] | torch.Tensor = LIBERO_HOME) -> None:
        values = np.asarray(state, dtype=np.float32)
        if values.shape != (9,) or not np.isfinite(values).all():
            raise ValueError(f"expected finite LIBERO joint state (9,), got {values.shape}")
        home = torch.as_tensor(values, device=self.q.device)
        self.q.zero_()
        self.q[:, 1:] = home
        self.qd.zero_()
        self._gripper_action.zero_()
        self._gripper_sign.zero_()
        self.drive_target.zero_()
        self.drive_target[:, 1:] = home
        self.drive_kp.zero_()
        self.drive_kd.zero_()
        self.drive_limit.zero_()
        if self.control_backend == "osc":
            # The task-link slots carry the scalar 6D task gains. Arm actuator
            # limits stay physical; the two fingers retain independent joint PD.
            self.drive_kp[0, 7] = LIBERO_OSC_KP
            self.drive_kd[0, 7] = LIBERO_OSC_KD
            self.drive_kp[0, 8:10] = torch.as_tensor(
                PANDA_KP[7:] * PANDA_PD_GAIN_SCALE, device=self.q.device
            )
            self.drive_kd[0, 8:10] = torch.as_tensor(
                PANDA_KD[7:] * np.sqrt(PANDA_PD_GAIN_SCALE), device=self.q.device
            )
            self.drive_limit[:, 1:] = torch.as_tensor(
                PANDA_FORCE_LIMIT, device=self.q.device
            )
        if self.control_backend == "osc":
            self.task_local_pose[0] = torch.as_tensor(
                ROBOSUITE_EEF_LOCAL, device=self.q.device
            )
            # Link pose is refreshed on the first world step after qpos writes.
            # Bootstrap against the pre-refresh pose, then lock the task target
            # to Nuka's own FK pose so settling does not alter the init state.
            self._osc_task_kp = LIBERO_OSC_KP
            self._osc_task_kd = LIBERO_OSC_KD
            # q is written directly, while ARTICULATION_LINK_POSE is refreshed
            # by the next world step. Disable the task term for that one refresh
            # step so OSC cannot act on the previous FK pose.
            self.drive_kp[0, 7] = 0.0
            self.drive_kd[0, 7] = 0.0
            self._osc_bootstrap = True
        else:
            self.drive_kp[:, 1:] = torch.as_tensor(
                PANDA_KP * PANDA_PD_GAIN_SCALE, device=self.q.device
            )
            self.drive_kd[:, 1:] = torch.as_tensor(
                PANDA_KD * np.sqrt(PANDA_PD_GAIN_SCALE), device=self.q.device
            )
            self.drive_limit[:, 1:] = torch.as_tensor(
                PANDA_FORCE_LIMIT * PANDA_PD_GAIN_SCALE, device=self.q.device
            )
        nuka.sync()

    def eef_pose(self) -> torch.Tensor:
        link = self.link_pose[0, 7]
        if self.control_backend == "osc":
            local = self.task_local_pose[0]
        else:
            local = torch.as_tensor(
                ROBOSUITE_EEF_LOCAL, device=link.device, dtype=link.dtype
            )
        position = link[:3] + _quat_rotate(link[3:], local[:3])
        rotation = _quat_mul(link[3:], local[3:])
        rotation = rotation / torch.linalg.vector_norm(rotation).clamp_min(1e-8)
        return torch.cat((position, rotation))

    def state8(self) -> torch.Tensor:
        """Return [eef xyz, eef axis-angle, left qpos, robosuite-right qpos]."""
        pose = self.eef_pose()
        axis_angle = _quat_to_axis_angle(pose[3:])
        gripper = torch.stack((self.q[0, 8], -self.q[0, 9]))

        return torch.cat((pose[:3], axis_angle, gripper)).view(1, 8)

    def target_position(self) -> torch.Tensor:
        return self.rigid_pose[self.target_body_index, :3]

    def plate_position(self) -> torch.Tensor:
        return self.rigid_pose[self.plate_body_index, :3]

    def attach_policy_cameras(self) -> None:
        if self._cameras_attached:
            return
        self.world.attach_camera_sensor(
            nuka.SensorMount.BASE.value,
            0,
            LIBERO_AGENT_CAMERA_LOCAL,
            LIBERO_AGENT_FOV,
            LIBERO_CAMERA_SIZE,
            LIBERO_CAMERA_SIZE,
        )
        self.world.attach_camera_sensor(
            nuka.SensorMount.LINK.value,
            7,
            LIBERO_WRIST_CAMERA_LOCAL,
            LIBERO_WRIST_FOV,
            LIBERO_CAMERA_SIZE,
            LIBERO_CAMERA_SIZE,
        )
        self.world.set_sensor_aov_mask(int(nuka.SensorAov.COLOR))
        if self.render_quality == "ultra":
            fidelity = dict(
                spp=32,
                shadow_samples=24,
                ao_enabled=True,
                ao_samples=16,
                gi_enabled=True,
            )
        elif self.render_quality == "high":
            fidelity = dict(
                spp=16,
                shadow_samples=12,
                ao_enabled=True,
                ao_samples=8,
                gi_enabled=True,
            )
        else:
            fidelity = dict(
                spp=4,
                shadow_samples=4,
                ao_enabled=True,
                ao_samples=3,
                gi_enabled=True,
            )
        self.world.set_sensor_fidelity(
            **fidelity,
            tonemap_enabled=True,
            sky_intensity=0.01,
            seed=0x4C494245,
        )
        self._cameras_attached = True

    def camera_images(self) -> torch.Tensor:
        """Return only the two policy cameras in checkpoint order."""
        self.attach_policy_cameras()
        self.world.render_sensors()
        if self._camera_tensor is None:
            self._camera_tensor = torch.from_dlpack(
                self.world.get_sensor_view(nuka.SensorChannel.COLOR)
            )
        expected = (1, 2, LIBERO_CAMERA_SIZE, LIBERO_CAMERA_SIZE, 3)
        if tuple(self._camera_tensor.shape) != expected:
            raise RuntimeError(
                f"unexpected LIBERO camera shape: {tuple(self._camera_tensor.shape)}"
            )
        # Both streams pass through raw: the checkpoint's LIBERO processor applies
        # its own camera-orientation rotation, so flipping here double-corrects.
        return self._camera_tensor[0]

    def third_person_image(
        self, *, width: int = 820, height: int = 615, spp: int = 16
    ) -> np.ndarray:
        """Render a presentation-only world camera; never used by policy input."""
        nuka.sync()
        eye = LIBERO_THIRD_PERSON_EYE
        look = LIBERO_THIRD_PERSON_LOOK_AT
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        return self.world.render_beauty(
            eye=tuple(eye.tolist()),
            look=tuple(look.tolist()),
            up=tuple(up.tolist()),
            fov_deg=LIBERO_THIRD_PERSON_FOV,
            width=width,
            height=height,
            spp=spp,
        )

    def _geometric_jacobian(self, eef_position: torch.Tensor) -> torch.Tensor:
        joints = self.link_pose[0, 1:8]
        local_z = torch.zeros((7, 3), device=joints.device, dtype=joints.dtype)
        local_z[:, 2] = 1.0
        axes = _quat_rotate(joints[:, 3:], local_z)
        linear = torch.cross(axes, eef_position.view(2, 3)[:1] - joints[:, :3], dim=-1)
        return torch.cat((linear.T, axes.T), dim=0)

    def set_policy_action(self, action: torch.Tensor | Sequence[float]) -> None:
        command = torch.as_tensor(
            action, device=self.q.device, dtype=torch.float32
        ).reshape(7)
        if not torch.isfinite(command).all():
            raise ValueError("non-finite LIBERO policy action")
        command = command.clamp(-1.0, 1.0)
        if self.control_backend == "osc":
            eef = self.eef_pose()
            self.task_target[0] = eef[:3] + command[:3] * 0.05
            rotation_delta = command[3:6] * 0.5
            self.task_rotation_target[0] = _quat_mul(
                _axis_angle_to_quat(rotation_delta), eef[3:]
            )
            self.task_rotation_target[0] = self.task_rotation_target[0] / torch.linalg.vector_norm(
                self.task_rotation_target[0]
            ).clamp_min(1e-8)
            # LIBERO/robosuite: -1 opens the gripper, +1 closes it. sign(0) holds
            # the aperture, which then advances per physics tick.
            self._gripper_sign.copy_(torch.sign(command[6]).view(1))
            return

        twist = torch.cat((command[:3] * 0.05, command[3:6] * 0.5))
        eef = self.eef_pose()
        jacobian = self._geometric_jacobian(eef[:3])
        damping = 0.045
        lhs = jacobian @ jacobian.T
        lhs.diagonal().add_(damping * damping)
        delta = jacobian.T @ torch.linalg.solve(lhs, twist)
        delta = delta.clamp(-0.18, 0.18)

        lower = torch.as_tensor(PANDA_ARM_LOWER, device=self.q.device)
        upper = torch.as_tensor(PANDA_ARM_UPPER, device=self.q.device)
        arm_target = (self.q[0, 1:8] + delta).clamp(lower, upper)
        self.drive_target[0, 1:8] = arm_target
        self._gripper_sign.copy_(torch.sign(command[6]).view(1))

    def _advance_gripper(self) -> None:
        """Accumulate the normalized gripper command at robosuite's substep rate."""
        rate = LIBERO_GRIPPER_RATE_PER_SECOND * self.dt
        self._gripper_action.add_(-rate * self._gripper_sign).clamp_(-1.0, 1.0)
        aperture = 0.02 * (1.0 + self._gripper_action[0])
        self.drive_target[0, 8] = aperture
        self.drive_target[0, 9] = aperture

    def step(
        self,
        action: torch.Tensor | Sequence[float] | None = None,
        *,
        advance_gripper: bool = True,
    ) -> dict[str, float]:
        if action is not None:
            self.set_policy_action(action)
        # Replay supplies recorded finger drive targets and owns them directly.
        if advance_gripper:
            self._advance_gripper()
        self.world.step()
        if self.control_backend == "osc" and self._osc_bootstrap:
            pose = self.eef_pose()
            self.task_target[0] = pose[:3]
            self.task_rotation_target[0] = pose[3:]
            self.drive_kp[0, 7] = self._osc_task_kp
            self.drive_kd[0, 7] = self._osc_task_kd
            self._osc_bootstrap = False
        target_error = self.q[:, 1:] - self.drive_target[:, 1:]
        if self.control_backend == "osc":
            eef_error = torch.linalg.vector_norm(
                self.eef_pose()[:3] - self.task_target[0]
            )
            joint_rmse = eef_error
        else:
            joint_rmse = torch.sqrt(torch.mean(target_error[:, :7] ** 2))
        return {
            "finite": float(
                torch.isfinite(self.q).all()
                and torch.isfinite(self.base).all()
                and torch.isfinite(self.rigid_pose).all()
            ),
            "joint_rmse_rad": float(joint_rmse.item()),
            "finger_error_m": float(target_error[:, 7:].abs().max().item()),
            "max_joint_speed": float(self.qd[:, 1:8].abs().max().item()),
        }

    def close(self) -> None:
        world = getattr(self, "world", None)
        if world is None:
            return
        for name in (
            "q",
            "qd",
            "base",
            "link_pose",
            "rigid_pose",
            "drive_target",
            "drive_kp",
            "drive_kd",
            "drive_limit",
            "task_target",
            "task_rotation_target",
            "task_local_pose",
            "_camera_tensor",
            "_gripper_action",
            "_osc_bootstrap",
        ):
            setattr(self, name, None)
        self.world = None
        world.destroy()


__all__ = [
    "LIBERO_AGENT_CAMERA_LOCAL",
    "LIBERO_CAMERA_SIZE",
    "LIBERO_EEF_REFERENCE",
    "LIBERO_HOME",
    "LIBERO_PLATE",
    "LIBERO_TARGET_BOWL",
    "LIBERO_TASK",
    "LIBERO_WRIST_CAMERA_LOCAL",
    "LIBERO_THIRD_PERSON_EYE",
    "LIBERO_THIRD_PERSON_LOOK_AT",
    "LIBERO_THIRD_PERSON_FOV",
    "LiberoBlackBowlController",
]
