"""Head-depth BDX locomotion on a real cooked scene.

This module adds perception to :class:`BdxWalkEnv` without adding a second
physics or rendering path.  The camera is mounted on a name-resolved cooked link,
renders the world's real visual instances, and returns a structured observation:

``proprio``
    The existing 44 float32 locomotion features.
``depth``
    A bounded ``(1,H,W)`` float16 image.  It stays an image for a CNN; it is never
    flattened into the policy MLP.  Zero means miss/far and one means near.
``depth_age``
    Seconds since the last camera update, so a lower-rate industrial camera can be
    fused with the high-rate joint/IMU path without pretending both are synchronous.

The physical sensor and the policy representation are deliberately separate.  A
scene can still be rendered at 640x480 (or higher) for sensor validation/video,
while training uses a configurable smaller image and an explicit encoder.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from gymnasium import spaces

import nuka
from . import bdx_obs as B
from .bdx_locomotion import BdxWalkEnv


# Match the rest of the training stack: launch from the repository root.  Deriving
# the repo from __file__ is wrong for a staged/installed Python package.
CORRIDOR_SCENE = "examples/scenes/corridor_nomedia.nks"

# The shipped duck's head-link axes are not camera axes.  This mount first rolls
# camera +Y onto world-up, then pitches the optical -Z axis 20 degrees downward.
# Translation puts a small external camera above/ahead of the visual head shell.
DEFAULT_MOUNT_JOINT = r"^head_roll$"
DEFAULT_MOUNT_OFFSET = (
    0.12,
    0.0,
    -0.14,
    0.696364240320019,
    -0.12278780396897285,
    0.12278780396897285,
    -0.696364240320019,
)


def _enum_value(value) -> int:
    return int(value.value) if hasattr(value, "value") else int(value)


class BdxPerceptionEnv(BdxWalkEnv):
    """BDX locomotion with a lower-rate head depth camera.

    ``sensor`` is a plain mapping so it can be authored directly in an rl_games
    YAML.  All dimensions and mount parameters are configurable; the default is
    the measured BDX head mount used by the one-shot corridor.
    """

    def __init__(self, scene: str = CORRIDOR_SCENE, num_envs: int = 256, *,
                 sensor: "dict | None" = None, **kwargs) -> None:
        cfg = dict(sensor or {})
        spawn_x_range = kwargs.pop("spawn_x_range", None)
        self.depth_width = int(cfg.get("width", 128))
        self.depth_height = int(cfg.get("height", 96))
        self.depth_update_period = int(cfg.get("update_period", 2))
        self.depth_near = float(cfg.get("near_clip", 0.05))
        self.depth_far = float(cfg.get("far_clip", 3.0))
        self.depth_vfov = float(cfg.get("vfov", 70.0))
        self.mount_joint_pattern = str(
            cfg.get("mount_joint_pattern", DEFAULT_MOUNT_JOINT))
        self.mount_offset = tuple(
            float(v) for v in cfg.get("mount_offset", DEFAULT_MOUNT_OFFSET))

        if self.depth_width <= 0 or self.depth_height <= 0:
            raise ValueError("sensor width/height must be positive")
        if self.depth_update_period <= 0:
            raise ValueError("sensor update_period must be positive")
        if not (0.0 < self.depth_near < self.depth_far):
            raise ValueError("sensor clips must satisfy 0 < near_clip < far_clip")
        if not (0.0 < self.depth_vfov < 180.0):
            raise ValueError("sensor vfov must be in (0,180) degrees")
        if len(self.mount_offset) != 7:
            raise ValueError("sensor mount_offset must be pos3 + quaternion wxyz")
        qnorm = math.sqrt(sum(v * v for v in self.mount_offset[3:]))
        if not math.isfinite(qnorm) or abs(qnorm - 1.0) > 1.0e-4:
            raise ValueError("sensor mount quaternion must be unit length")

        # Full corridor/media scenes use the already-established SceneBuilder cook;
        # BdxWalkEnv keeps this opt-in so its blind training default is untouched.
        kwargs.setdefault("use_scene_builder", True)
        kwargs.setdefault("scene_build_kwargs", {"bake_link_sdf": True})
        super().__init__(scene, num_envs, **kwargs)

        self.spawn_x_range = (
            None if spawn_x_range is None
            else tuple(float(v) for v in spawn_x_range))
        if self.spawn_x_range is not None:
            if (len(self.spawn_x_range) != 2
                    or not self.spawn_x_range[0] < self.spawn_x_range[1]):
                raise ValueError("spawn_x_range must be [low, high] with low < high")

        matches = self._world.dof_index(self.mount_joint_pattern)
        if len(matches) != 1:
            raise RuntimeError(
                f"head camera mount {self.mount_joint_pattern!r} resolved to "
                f"{matches.tolist()}, expected exactly one cooked link")
        self.sensor_link_index = int(matches[0])
        self._world.attach_camera_sensor(
            _enum_value(nuka.SensorMount.LINK),
            self.sensor_link_index,
            self.mount_offset,
            self.depth_vfov,
            self.depth_width,
            self.depth_height,
        )
        self._world.set_camera_intrinsics(
            False, 0.0, 0.0, self.depth_near, self.depth_far)
        # Depth alone selects the same dedicated primary-hit kernel as DEPTH|PRIM,
        # while avoiding an unused uint32 rollout plane.
        self._world.set_sensor_aov_mask(_enum_value(nuka.SensorAov.DEPTH))

        self._depth = torch.zeros(
            self.num_envs, 1, self.depth_height, self.depth_width,
            dtype=torch.float16, device=self._dev)
        self._depth_age = torch.zeros(
            self.num_envs, 1, dtype=torch.float32, device=self._dev)
        self._sensor_tick = 0
        self._control_dt = self.dt * self.decimation

        proprio_one = spaces.Box(
            low=-B.OBS_CLIP, high=B.OBS_CLIP,
            shape=(B.BDX_OBS_DIM,), dtype=np.float32)
        depth_one = spaces.Box(
            low=np.float16(0.0), high=np.float16(1.0),
            shape=(1, self.depth_height, self.depth_width), dtype=np.float16)
        age_one = spaces.Box(
            low=0.0, high=max(1.0, self.depth_update_period * self._control_dt),
            shape=(1,), dtype=np.float32)
        self.single_observation_space = spaces.Dict({
            "proprio": proprio_one,
            "depth": depth_one,
            "depth_age": age_one,
        })
        self.observation_space = spaces.Dict({
            "proprio": spaces.Box(
                low=-B.OBS_CLIP, high=B.OBS_CLIP,
                shape=(self.num_envs, B.BDX_OBS_DIM), dtype=np.float32),
            "depth": spaces.Box(
                low=np.float16(0.0), high=np.float16(1.0),
                shape=(self.num_envs, 1, self.depth_height, self.depth_width),
                dtype=np.float16),
            "depth_age": spaces.Box(
                low=0.0,
                high=max(1.0, self.depth_update_period * self._control_dt),
                shape=(self.num_envs, 1), dtype=np.float32),
        })
        rollout_mib = (
            self.num_envs * self.depth_height * self.depth_width * 2.0
            / (1024.0 * 1024.0))
        print(
            f"[bdx_perception] scene={scene} head_link={self.sensor_link_index}:"
            f"{self._world.dof_names()[self.sensor_link_index]} "
            f"depth={self.depth_width}x{self.depth_height}@"
            f"{1.0 / (self.depth_update_period * self._control_dt):.1f}Hz "
            f"frame={rollout_mib:.1f}MiB fp16",
            flush=True,
        )

    def _randomize_spawn_x(self, mask: "torch.Tensor | None") -> None:
        """Translate the articulation on the start platform, environment unchanged.

        Moving only the reset pose varies time-to-edge without authoring a second
        corridor or changing collision geometry.  LinkPose is translated alongside
        BasePose so the head camera's first post-reset frame is immediately coherent;
        the next physics step recomputes the same FK from the new base state.
        """
        if self.spawn_x_range is None:
            return
        if mask is None:
            mask = torch.ones(
                self.num_envs, dtype=torch.bool, device=self._dev)
        count = int(mask.sum().item())
        if count == 0:
            return
        lo, hi = self.spawn_x_range
        target = (
            torch.rand(count, generator=self._generator, device=self._dev)
            * (hi - lo) + lo)
        base_pose = self._obs._base_pose
        link_pose = self._obs._pose
        delta = target - base_pose[mask, 0]
        base_pose[mask, 0] = target
        link_pose[mask, :, 0] += delta.unsqueeze(1)
        nuka.sync()

    def _structured_obs(self, proprio: torch.Tensor, *, force: bool = False):
        update = force or (self._sensor_tick % self.depth_update_period == 0)
        if update:
            self._world.render_sensors()
            # The renderer and PyTorch own different stream abstractions.  Keep the
            # hand-off explicit until the public DLPack surface carries an event.
            nuka.sync()
            raw = torch.from_dlpack(
                self._world.get_sensor_view(nuka.SensorChannel.DEPTH))
            if raw.ndim != 4 or tuple(raw.shape) != (
                    self.num_envs, self.depth_height, self.depth_width, 1):
                raise RuntimeError(
                    f"unexpected depth tensor shape {tuple(raw.shape)}; expected "
                    f"({self.num_envs},{self.depth_height},{self.depth_width},1)")
            z = raw[..., 0]
            valid = torch.isfinite(z) & (z >= self.depth_near) & (z <= self.depth_far)
            # Near -> 1, far/miss -> 0.  Bounded input avoids a per-pixel running
            # normalizer and stores compactly in the PPO experience buffer.
            normalized = torch.where(
                valid,
                (self.depth_far - z.clamp(self.depth_near, self.depth_far))
                / (self.depth_far - self.depth_near),
                torch.zeros((), dtype=z.dtype, device=z.device),
            )
            self._depth.copy_(normalized.unsqueeze(1).to(torch.float16))
            self._depth_age.zero_()
        else:
            age_s = min(
                (self._sensor_tick % self.depth_update_period) * self._control_dt,
                self.depth_update_period * self._control_dt,
            )
            self._depth_age.fill_(age_s)
        return {
            "proprio": proprio,
            "depth": self._depth,
            "depth_age": self._depth_age,
        }

    def reset(self, *, seed=None, options=None):
        proprio, info = super().reset(seed=seed, options=options)
        self._randomize_spawn_x(None)
        if self.spawn_x_range is not None:
            proprio = self._obs.compute_obs(
                self.command, self.last_action,
                self._ref.phase_features(self.phase_i))
        self._sensor_tick = 0
        return self._structured_obs(proprio, force=True), info

    def step(self, actions: torch.Tensor):
        proprio, reward, terminated, truncated, info = super().step(actions)
        self._sensor_tick += 1
        # A reset env must not inherit the previous episode's last camera frame.
        done = terminated | truncated
        force = bool(done.any())
        if force and self.spawn_x_range is not None:
            self._randomize_spawn_x(done)
            fresh = self._obs.compute_obs(
                self.command, self.last_action,
                self._ref.phase_features(self.phase_i))
            proprio = proprio.clone()
            proprio[done] = fresh[done]
        obs = self._structured_obs(proprio, force=force)
        return obs, reward, terminated, truncated, info


def make_env(num_envs: int, *, device=None, scene: str = CORRIDOR_SCENE,
             **kwargs) -> BdxPerceptionEnv:
    """Construct the structured-observation BDX scene environment."""
    return BdxPerceptionEnv(
        scene=scene, num_envs=num_envs, device=device, **kwargs)
