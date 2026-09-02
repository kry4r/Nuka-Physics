"""Single-environment Franka Panda pick/place control for pi0.5 inference."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import nuka

PANDA_JOINT_NAMES = (
    "joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7",
    "finger_joint1", "finger_joint2",
)

# Median state from the pinned C1 Panda pick-cube checkpoint dataset.
PANDA_C1_HOME = np.array([
    -0.04080849, 0.34862798, -0.04117866, -2.36771464,
    0.03044342, 2.70693517, 0.67838210, 0.04, 0.04,
], dtype=np.float32)
# Verified first frame of dataset episode 0. The Isaac demonstrations do not
# reset every episode to the global median; the policy expects the arm already
# staged near the cube at the beginning of a recorded rollout.
PANDA_C1_EPISODE0_START = np.array([
    0.06096641, 0.09969219, -0.08263995, -2.51266980,
    0.01657944, 2.61094427, 0.75345570, 0.04,
], dtype=np.float32)
PANDA_ARM_LOWER = np.array([
    -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973,
], dtype=np.float32)
PANDA_ARM_UPPER = np.array([
    2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973,
], dtype=np.float32)
PANDA_KP = np.array([600, 600, 500, 500, 360, 240, 160, 80, 80], dtype=np.float32)
PANDA_KD = np.array([46, 46, 38, 38, 28, 20, 14, 7, 7], dtype=np.float32)
PANDA_FORCE_LIMIT = np.array([87, 87, 87, 87, 12, 12, 12, 20, 20], dtype=np.float32)
# The source Isaac/ROS controller holds absolute joint targets at 30 Hz. Nuka's
# legacy PD law needs a higher servo stiffness to keep a loaded Panda near that
# target under Earth gravity; the scale is applied to Kp, Kd=sqrt(scale), and
# force limits together so the position-servo behavior remains well damped.
PANDA_PD_GAIN_SCALE = 10.0

# A link7-mounted camera. Camera-local axes are -Z forward and +Y up. This
# calibrated mount sees both fingertips and the table workspace at C1 home.
PANDA_WRIST_CAMERA_LOCAL = (
    0.11230725, -0.00124166, 0.03945775,
    -0.48216370, 0.71988928, -0.46311759, 0.18654624,
)
PANDA_CUBE_INITIAL = np.array([0.50, 0.0, 0.465], dtype=np.float32)
PANDA_POLICY_CAMERA_EYE = (-0.35, -1.25, 1.25)
PANDA_POLICY_CAMERA_LOOK = (0.40, 0.06, 0.63)
PANDA_POLICY_CAMERA_FOV = 40.0
PANDA_POLICY_CAMERA_LOCAL = (
    -0.35, -1.25, 0.81,
    0.80272752, 0.53809489, -0.14313556, -0.21352898,
)


def _quat_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


class PandaPickPlaceController:
    """Panda engine-PD controller with an 8D pi0.5 action contract."""

    def __init__(
        self,
        scene: str | Path,
        device: nuka.Device,
        *,
        dt: float = 0.005,
        cube_position: Sequence[float] | None = None,
    ):
        self.dt = float(dt)
        self.world = nuka.World.create_from_scene(
            device, str(scene), env_count=1, dt=self.dt,
            determinism=nuka.DETERMINISM_STRONG,
            control_mode=nuka.CONTROL_MODE_PD_POSITION,
            contact_family=1,
            heightfield_terrain_type=0,
        )
        names = tuple(self.world.dof_names())
        missing = [name for name in PANDA_JOINT_NAMES if name not in names]
        if missing or self.world.base_link_count != 10 or self.world.action_dim != 9:
            raise RuntimeError(
                f"Panda topology mismatch: links={self.world.base_link_count} "
                f"actions={self.world.action_dim} missing={missing} names={names}")
        ordered = tuple(name for name in names if name in PANDA_JOINT_NAMES)
        if ordered != PANDA_JOINT_NAMES:
            raise RuntimeError(f"Panda joint order mismatch: {ordered}")

        self.q = torch.from_dlpack(
            self.world.buffer_view(nuka.JOINT_POSITION)).view(1, 10)
        self.qd = torch.from_dlpack(
            self.world.buffer_view(nuka.JOINT_VELOCITY)).view(1, 10)
        self.base = torch.from_dlpack(
            self.world.buffer_view(nuka.BASE_POSE)).view(1, 7)
        self.link_pose = torch.from_dlpack(
            self.world.buffer_view(nuka.ARTICULATION_LINK_POSE)).view(1, 10, 7)
        self.rigid_pose = torch.from_dlpack(
            self.world.buffer_view(nuka.RIGID_BODY_TRANSFORM)).view(-1, 7)
        self.drive_target = torch.from_dlpack(
            self.world.buffer_view(nuka.DRIVE_TARGET)).view(1, 10)
        self.drive_kp = torch.from_dlpack(
            self.world.buffer_view(nuka.DRIVE_STIFFNESS)).view(1, 10)
        self.drive_kd = torch.from_dlpack(
            self.world.buffer_view(nuka.DRIVE_DAMPING)).view(1, 10)
        self.drive_limit = torch.from_dlpack(
            self.world.buffer_view(nuka.DRIVE_FORCE_LIMIT)).view(1, 10)
        cube_guess = torch.as_tensor(
            PANDA_CUBE_INITIAL if cube_position is None else cube_position,
            device=self.rigid_pose.device,
        )
        self.cube_body_index = int(torch.argmin(
            torch.linalg.vector_norm(self.rigid_pose[:, :3] - cube_guess, dim=-1)
        ).item())
        cube_error = torch.linalg.vector_norm(
            self.rigid_pose[self.cube_body_index, :3] - cube_guess).item()
        if cube_error > 0.20:
            raise RuntimeError(f"red cube body discovery failed: nearest error={cube_error:.3f}m")
        self._policy_camera_attached = False
        self._camera_tensor: torch.Tensor | None = None
        self.reset()

    def reset(self, state: Sequence[float] | torch.Tensor | None = None) -> None:
        values = np.asarray(PANDA_C1_HOME if state is None else state, dtype=np.float32)
        if values.shape == (8,):
            home = np.concatenate((values[:7], values[7:8], values[7:8]))
        elif values.shape == (9,):
            home = values
        else:
            raise ValueError(f"expected finite C1 reset state with shape (8,) or (9,), got {values.shape}")
        if not np.isfinite(home).all():
            raise ValueError("C1 reset state must be finite")
        home = torch.as_tensor(home, device=self.q.device)
        self.q.zero_()
        self.q[:, 1:] = home
        self.qd.zero_()
        self.drive_target.zero_()
        self.drive_target[:, 1:] = home
        self.drive_kp.zero_()
        self.drive_kd.zero_()
        self.drive_limit.zero_()
        gain_scale = PANDA_PD_GAIN_SCALE
        self.drive_kp[:, 1:] = torch.as_tensor(
            PANDA_KP * gain_scale, device=self.q.device)
        self.drive_kd[:, 1:] = torch.as_tensor(
            PANDA_KD * np.sqrt(gain_scale), device=self.q.device)
        self.drive_limit[:, 1:] = torch.as_tensor(
            PANDA_FORCE_LIMIT * gain_scale, device=self.q.device)
        nuka.sync()

    def state8(self) -> torch.Tensor:
        """Return the checkpoint state: seven joints and left finger opening."""
        return torch.cat((self.q[:, 1:8], self.q[:, 8:9]), dim=-1)

    def cube_position(self) -> torch.Tensor:
        """Return the live red-cube center as a CUDA tensor view."""
        return self.rigid_pose[self.cube_body_index, :3]

    def finger_center(self) -> torch.Tensor:
        return self.link_pose[0, 8:10, :3].mean(dim=0)

    def attach_policy_camera(self) -> None:
        """Attach external policy camera 0 and link7 wrist display camera 1."""
        if self._policy_camera_attached:
            return
        self.world.attach_camera_sensor(
            nuka.SensorMount.BASE.value, 0, PANDA_POLICY_CAMERA_LOCAL,
            PANDA_POLICY_CAMERA_FOV, 640, 480,
        )
        self.world.attach_camera_sensor(
            nuka.SensorMount.LINK.value, 7, PANDA_WRIST_CAMERA_LOCAL,
            55.0, 640, 480,
        )
        self.world.set_sensor_aov_mask(int(nuka.SensorAov.COLOR))
        self.world.set_sensor_fidelity(
            spp=4, shadow_samples=4, ao_enabled=True, ao_samples=3,
            gi_enabled=True, tonemap_enabled=True, sky_intensity=0.35,
            seed=0x50493035,
        )
        self._policy_camera_attached = True

    def camera_images(self) -> torch.Tensor:
        """Render and return external/wrist CUDA HWC RGB images in [0,1]."""
        self.attach_policy_camera()
        self.world.render_sensors()
        if self._camera_tensor is None:
            self._camera_tensor = torch.from_dlpack(
                self.world.get_sensor_view(nuka.SensorChannel.COLOR))
        image = self._camera_tensor
        if image.shape != (1, 2, 480, 640, 3):
            raise RuntimeError(f"unexpected Panda camera shape: {tuple(image.shape)}")
        return image[0]

    def policy_image(self) -> torch.Tensor:
        """Return the C1 external camera directly from the engine CUDA buffer."""
        return self.camera_images()[0]

    def wrist_image(self) -> torch.Tensor:
        """Return the link7 camera from the same persistent CUDA sensor block."""
        return self.camera_images()[1]

    def policy_image_host(self, *, spp: int = 4) -> np.ndarray:
        """Render C1's external camera as uint8 HWC RGB for HOST_COMPAT."""
        return self.world.render_beauty(
            eye=PANDA_POLICY_CAMERA_EYE, look=PANDA_POLICY_CAMERA_LOOK,
            fov_deg=PANDA_POLICY_CAMERA_FOV, width=640, height=480, spp=spp,
        )

    def wrist_camera_extrinsics(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        """Convert the fixed link7 camera mount into live world-space vectors."""
        pose = self.link_pose[0, 7].detach().cpu().numpy()
        link_rotation = _quat_matrix_wxyz(pose[3:7])
        mount_position = np.asarray(PANDA_WRIST_CAMERA_LOCAL[:3], dtype=np.float32)
        mount_rotation = _quat_matrix_wxyz(
            np.asarray(PANDA_WRIST_CAMERA_LOCAL[3:7], dtype=np.float32))
        camera_rotation = link_rotation @ mount_rotation
        eye = pose[:3] + link_rotation @ mount_position
        look = eye + camera_rotation @ np.array([0.0, 0.0, -1.0], dtype=np.float32)
        up = camera_rotation @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
        return tuple(eye.tolist()), tuple(look.tolist()), tuple(up.tolist())

    def wrist_image_host(
        self, *, width: int = 640, height: int = 480, spp: int = 2,
    ) -> np.ndarray:
        eye, look, up = self.wrist_camera_extrinsics()
        return self.world.render_beauty(
            eye=eye, look=look, up=up, fov_deg=55.0,
            width=width, height=height, spp=spp,
        )

    def set_policy_action(self, action: torch.Tensor | Sequence[float]) -> None:
        command = torch.as_tensor(action, device=self.q.device, dtype=torch.float32).view(1, 8)
        lower = torch.as_tensor(PANDA_ARM_LOWER, device=self.q.device).view(1, 7)
        upper = torch.as_tensor(PANDA_ARM_UPPER, device=self.q.device).view(1, 7)
        arm = torch.minimum(torch.maximum(command[:, :7], lower), upper)
        finger = command[:, 7:8].clamp(0.0, 1.0) * 0.04
        self.drive_target[:, 1:8] = arm
        self.drive_target[:, 8:9] = finger
        self.drive_target[:, 9:10] = finger

    def step(self, action: torch.Tensor | Sequence[float] | None = None) -> dict[str, float]:
        if action is not None:
            self.set_policy_action(action)
        self.world.step()
        target_error = self.q[:, 1:] - self.drive_target[:, 1:]
        finger_symmetry = torch.abs(self.q[:, 8] - self.q[:, 9])
        return {
            "finite": float(torch.isfinite(self.q).all() and torch.isfinite(self.base).all()),
            "joint_rmse_rad": float(torch.sqrt(torch.mean(target_error[:, :7] ** 2)).item()),
            "finger_error_m": float(torch.max(torch.abs(target_error[:, 7:])).item()),
            "finger_symmetry_error_m": float(finger_symmetry.item()),
            "max_joint_speed": float(self.qd[:, 1:8].abs().max().item()),
        }

    def close(self) -> None:
        world = getattr(self, "world", None)
        if world is None:
            return
        for name in (
            "q", "qd", "base", "link_pose", "rigid_pose", "drive_target",
            "drive_kp", "drive_kd", "drive_limit", "_camera_tensor",
        ):
            setattr(self, name, None)
        self.world = None
        world.destroy()


__all__ = [
    "PANDA_C1_HOME", "PANDA_C1_EPISODE0_START", "PANDA_PD_GAIN_SCALE", "PANDA_CUBE_INITIAL", "PANDA_JOINT_NAMES",
    "PANDA_POLICY_CAMERA_EYE", "PANDA_POLICY_CAMERA_FOV",
    "PANDA_POLICY_CAMERA_LOCAL", "PANDA_POLICY_CAMERA_LOOK",
    "PANDA_WRIST_CAMERA_LOCAL", "PandaPickPlaceController",
]
