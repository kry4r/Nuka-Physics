"""Shared corridor traversal task for the BDX depth and privileged-teacher envs.

The corridor progress/success/spawn-randomization logic lives here so the depth
policy and the flat privileged teacher run the identical task on the identical
cooked scene, differing only in the observation their policy sees.  The teacher
adds a parsed centre-line height profile in place of a camera.
"""

from __future__ import annotations

import json
import math

import numpy as np
import torch
from gymnasium import spaces

import nuka
from . import bdx_obs as B
from .bdx_locomotion import BdxWalkEnv


# Launch from the repository root; deriving the repo from __file__ is wrong for a
# staged/installed package.
CORRIDOR_SCENE = "examples/scenes/corridor_nomedia.nks"

# Forward height-profile sample offsets along +x (m): -0.10 .. +0.50 step 0.04.
TEACHER_PROFILE_OFFSETS = tuple(round(-0.10 + 0.04 * i, 2) for i in range(16))
PRIV_HEIGHT_CLAMP = 0.5


class CorridorHeightProfile:
    """1 cm-resolution walkable height h(x) along the corridor centre line (y=0).

    A static box with contype!=0 whose footprint covers (x, y=0) and whose bottom
    sits at/near the floor contributes its top face; h(x) is the max such top.
    Overhead beams (bottom well above the floor) are filtered so the profile is the
    ground the feet actually rest on.
    """

    X_MIN = -1.5
    X_MAX = 6.0
    CELL = 0.01

    def __init__(self, scene: str, device) -> None:
        self._dev = device
        boxes = self._static_ground_boxes(scene)
        self._n = int(round((self.X_MAX - self.X_MIN) / self.CELL)) + 1
        xs = self.X_MIN + self.CELL * np.arange(self._n, dtype=np.float64)
        h = np.full(self._n, -1.0e9, dtype=np.float64)
        # Half-cell margin so two flush boxes both cover their shared boundary cell
        # (float edges otherwise leave a sub-cell gap that notches the profile).
        eps = 0.5 * self.CELL
        for xlo, xhi, ztop in boxes:
            cover = (xs >= xlo - eps) & (xs <= xhi + eps)
            h = np.where(cover & (ztop > h), ztop, h)
        floor = min((b[2] for b in boxes), default=0.0)
        h = np.where(h < -1.0e8, floor, h)
        self._table = torch.as_tensor(h, dtype=torch.float32, device=device)
        self._assert_ground_truth()

    @staticmethod
    def _static_ground_boxes(scene: str):
        doc = json.load(open(scene))
        out = []
        for node in doc["tree"]:
            rb = node.get("rigid_body") or {}
            if not bool(rb.get("is_static", False)):
                continue
            pos = (node.get("transform") or {}).get("pos")
            if pos is None:
                continue
            for child in node.get("children", []):
                cs = child.get("collision_shape")
                if not cs or cs.get("type") != "box" or not cs.get("contype"):
                    continue
                he = cs["half_extents"]
                loc = (cs.get("local") or {}).get("pos", [0.0, 0.0, 0.0])
                cx, cy, cz = pos[0] + loc[0], pos[1] + loc[1], pos[2] + loc[2]
                if not (cy - he[1] <= 0.0 <= cy + he[1]):
                    continue
                if cz - he[2] > 0.1:
                    continue
                out.append((cx - he[0], cx + he[0], cz + he[2]))
        if not out:
            raise RuntimeError(f"no walkable static boxes parsed from {scene}")
        return out

    def _assert_ground_truth(self) -> None:
        checks = {-0.2: 0.15, 0.18: 0.10, 0.5: 0.05}
        xq = torch.tensor(list(checks), dtype=torch.float32, device=self._dev)
        got = self.heights_at(xq.unsqueeze(0)).squeeze(0)
        for (x, expect), value in zip(checks.items(), got.tolist()):
            if abs(value - expect) > 1.0e-3:
                raise RuntimeError(
                    f"corridor profile h({x})={value:.4f} != {expect} "
                    "(scene parse broken)")

    def heights_at(self, x: torch.Tensor) -> torch.Tensor:
        """Nearest-cell h(x) for x of any shape, clamped to the table domain."""
        idx = torch.round((x - self.X_MIN) / self.CELL).long().clamp_(0, self._n - 1)
        return self._table[idx]


class CorridorTaskMixin:
    """Corridor progress/success/fall shaping + start-platform spawn randomization.

    Mixed into a :class:`BdxWalkEnv` subclass that supplies ``_structured_obs`` to
    turn the base proprio vector into its policy observation.  It defines no
    ``__init__`` so the subclass ``super().__init__`` reaches ``BdxWalkEnv`` directly.
    """

    def _pop_corridor_kwargs(self, kwargs: dict) -> None:
        self._spawn_x_range_raw = kwargs.pop("spawn_x_range", None)
        self._spawn_x_windows_raw = kwargs.pop("spawn_x_windows", None)
        self.forward_progress_scale = float(
            kwargs.pop("forward_progress_scale", 0.0))
        self.progress_milestones = tuple(
            float(v) for v in kwargs.pop("progress_milestones", ()))
        self.progress_milestone_bonus = float(
            kwargs.pop("progress_milestone_bonus", 0.0))
        self.success_progress_m = float(kwargs.pop("success_progress_m", 0.0))
        self.success_bonus = float(kwargs.pop("success_bonus", 0.0))
        self.fall_penalty = float(kwargs.pop("fall_penalty", 0.0))
        if not math.isfinite(self.forward_progress_scale) \
                or self.forward_progress_scale < 0.0:
            raise ValueError("forward_progress_scale must be non-negative")
        if not math.isfinite(self.progress_milestone_bonus) \
                or self.progress_milestone_bonus < 0.0:
            raise ValueError("progress_milestone_bonus must be non-negative")
        if (any(not math.isfinite(v) or v <= 0.0
                for v in self.progress_milestones)
                or any(b <= a for a, b in zip(
                    self.progress_milestones, self.progress_milestones[1:]))):
            raise ValueError("progress_milestones must be positive and increasing")
        if not math.isfinite(self.success_progress_m) \
                or self.success_progress_m < 0.0:
            raise ValueError("success_progress_m must be non-negative")
        if not math.isfinite(self.success_bonus) or self.success_bonus < 0.0:
            raise ValueError("success_bonus must be non-negative")
        if not math.isfinite(self.fall_penalty) or self.fall_penalty < 0.0:
            raise ValueError("fall_penalty must be non-negative")

    def _finalize_spawn_range(self) -> None:
        self.spawn_x_range = (
            None if self._spawn_x_range_raw is None
            else tuple(float(v) for v in self._spawn_x_range_raw))
        if self.spawn_x_range is not None:
            if (len(self.spawn_x_range) != 2
                    or not self.spawn_x_range[0] < self.spawn_x_range[1]):
                raise ValueError("spawn_x_range must be [low, high] with low < high")
        self.spawn_x_windows = None
        self._window_bounds = None
        self._window_cdf = None
        if self._spawn_x_windows_raw is not None:
            if self.spawn_x_range is not None:
                raise ValueError(
                    "spawn_x_range and spawn_x_windows are mutually exclusive")
            windows, weights = [], []
            for entry in self._spawn_x_windows_raw:
                vals = tuple(float(v) for v in entry)
                if len(vals) != 3 or not vals[0] < vals[1] or vals[2] <= 0.0:
                    raise ValueError(
                        "spawn_x_windows entries must be [low, high, weight] "
                        "with low < high and weight > 0")
                windows.append((vals[0], vals[1]))
                weights.append(vals[2])
            self.spawn_x_windows = tuple(windows)
            self._window_bounds = torch.tensor(
                windows, dtype=torch.float32, device=self._dev)
            w = torch.tensor(weights, dtype=torch.float32, device=self._dev)
            self._window_cdf = torch.cumsum(w / w.sum(), dim=0)

    @property
    def _spawn_active(self) -> bool:
        return self.spawn_x_range is not None or self.spawn_x_windows is not None

    def _init_progress_state(self) -> None:
        self._episode_start_x = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self._dev)
        self._episode_max_progress = torch.zeros_like(self._episode_start_x)
        self._milestones_reached = torch.zeros(
            self.num_envs, len(self.progress_milestones),
            dtype=torch.bool, device=self._dev)
        self._milestone_values = torch.as_tensor(
            self.progress_milestones, dtype=torch.float32,
            device=self._dev).unsqueeze(0)
        self._last_success = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self._dev)

    # -- subclass hooks -------------------------------------------------------
    def _structured_obs(self, proprio: torch.Tensor, *, force: bool = False):
        raise NotImplementedError

    def _reset_obs_clock(self) -> None:
        pass

    def _advance_obs_clock(self) -> None:
        pass

    def _recompute_proprio(self) -> "torch.Tensor":
        """Recompose the base proprio vector after a spawn teleport / autoreset.

        The default is the BDX proprio (gait-phase clocked); a host env whose obs
        differs (e.g. the Go2 48-d vector) overrides this."""
        return self._obs.compute_obs(
            self.command, self.last_action,
            self._ref.phase_features(self.phase_i))

    # -- corridor task logic --------------------------------------------------
    def _reset_progress_state(self, mask: "torch.Tensor | None") -> None:
        x = self._obs.base_pos()[:, 0]
        if mask is None:
            self._episode_start_x.copy_(x)
            self._episode_max_progress.zero_()
            self._milestones_reached.zero_()
        else:
            self._episode_start_x[mask] = x[mask]
            self._episode_max_progress[mask] = 0.0
            self._milestones_reached[mask] = False

    def compute_terminated(self) -> "torch.Tensor":
        physical_fall = super().compute_terminated()
        self._last_success.zero_()
        if self.success_progress_m > 0.0:
            progress = self._obs.base_pos()[:, 0] - self._episode_start_x
            self._last_success.copy_(
                (progress >= self.success_progress_m) & ~physical_fall)
        return physical_fall | self._last_success

    def _randomize_spawn_x(self, mask: "torch.Tensor | None") -> None:
        """Translate the articulation on the start platform, environment unchanged.

        Moving only the reset pose varies time-to-edge without authoring a second
        corridor or changing collision geometry.  LinkPose is translated alongside
        BasePose so the next physics step recomputes the same FK from the new base.
        """
        if not self._spawn_active:
            return
        if mask is None:
            mask = torch.ones(
                self.num_envs, dtype=torch.bool, device=self._dev)
        count = int(mask.sum().item())
        if count == 0:
            return
        base_pose = self._obs._base_pose
        link_pose = self._obs._pose
        if self.spawn_x_windows is None:
            lo, hi = self.spawn_x_range
            target = (
                torch.rand(count, generator=self._generator, device=self._dev)
                * (hi - lo) + lo)
            delta = target - base_pose[mask, 0]
            base_pose[mask, 0] = target
            link_pose[mask, :, 0] += delta.unsqueeze(1)
            nuka.sync()
            return
        # Multi-window start curriculum: pick a window by weight, sample within it,
        # and drop the whole articulation by the terrain-height change so feet land.
        pick = torch.rand(count, generator=self._generator, device=self._dev)
        win = torch.searchsorted(self._window_cdf, pick).clamp_(
            0, self._window_cdf.numel() - 1)
        lo = self._window_bounds[win, 0]
        hi = self._window_bounds[win, 1]
        u = torch.rand(count, generator=self._generator, device=self._dev)
        target = lo + u * (hi - lo)
        old_x = base_pose[mask, 0]
        dz = (self._profile.heights_at(target.unsqueeze(1))
              - self._profile.heights_at(old_x.unsqueeze(1))).squeeze(1)
        delta = target - old_x
        base_pose[mask, 0] = target
        base_pose[mask, 2] += dz
        link_pose[mask, :, 0] += delta.unsqueeze(1)
        link_pose[mask, :, 2] += dz.unsqueeze(1)
        nuka.sync()

    def reset(self, *, seed=None, options=None):
        proprio, info = super().reset(seed=seed, options=options)
        self._randomize_spawn_x(None)
        if self._spawn_active:
            proprio = self._recompute_proprio()
        self._reset_progress_state(None)
        self._last_success.zero_()
        self._reset_obs_clock()
        return self._structured_obs(proprio, force=True), info

    def step(self, actions: torch.Tensor):
        previous_x = self._obs.base_pos()[:, 0].clone()
        proprio, reward, terminated, truncated, info = super().step(actions)
        success = self._last_success.clone()
        if self.success_bonus > 0.0:
            reward = reward + success.to(reward.dtype) * self.success_bonus
        fall = terminated & ~success
        if self.fall_penalty > 0.0:
            reward = reward - fall.to(reward.dtype) * self.fall_penalty
        info = dict(info)
        info["success"] = success
        info["fall"] = fall
        self._advance_obs_clock()
        done = terminated | truncated
        force = bool(done.any())
        active = ~done
        current_x = self._obs.base_pos()[:, 0]
        if self.forward_progress_scale > 0.0:
            # Bound one-step credit so a collision impulse cannot manufacture a
            # large reward by launching the base forward.
            dx = torch.clamp(current_x - previous_x, -0.02, 0.02)
            reward = reward + (
                self.forward_progress_scale * dx * active.to(dx.dtype))
        if self.progress_milestones and self.progress_milestone_bonus > 0.0:
            progress = torch.clamp_min(current_x - self._episode_start_x, 0.0)
            self._episode_max_progress[active] = torch.maximum(
                self._episode_max_progress[active], progress[active])
            crossed = self._episode_max_progress.unsqueeze(1) >= self._milestone_values
            new_crossings = crossed & ~self._milestones_reached
            new_crossings &= active.unsqueeze(1)
            reward = reward + (
                new_crossings.sum(dim=1).to(reward.dtype)
                * self.progress_milestone_bonus)
            self._milestones_reached |= new_crossings
        if force and self._spawn_active:
            self._randomize_spawn_x(done)
            fresh = self._recompute_proprio()
            proprio = proprio.clone()
            proprio[done] = fresh[done]
        if force:
            self._reset_progress_state(done)
            self._last_success[done] = False
        obs = self._structured_obs(proprio, force=force)
        return obs, reward, terminated, truncated, info


class BdxCorridorTeacherEnv(CorridorTaskMixin, BdxWalkEnv):
    """Flat privileged teacher: proprio + a parsed height profile, no camera.

    The observation is ``[proprio(44) | base_z - h(base_x) | h(base_x+off) - h(base_x)]``
    with the 16 forward offsets, so PPO can anticipate the sharp down-steps the depth
    policy must later infer from pixels.
    """

    def __init__(self, scene: str = CORRIDOR_SCENE, num_envs: int = 256,
                 **kwargs) -> None:
        self._pop_corridor_kwargs(kwargs)
        # Same cooked scene + baked link SDF as the perception env.
        kwargs.setdefault("use_scene_builder", True)
        kwargs.setdefault("scene_build_kwargs", {"bake_link_sdf": True})
        super().__init__(scene, num_envs, **kwargs)
        self._finalize_spawn_range()
        self._init_progress_state()

        self._profile = CorridorHeightProfile(scene, self._dev)
        # Sample along world +x: the corridor command is a fixed yaw-0 forward walk.
        self._priv_offsets = torch.as_tensor(
            TEACHER_PROFILE_OFFSETS, dtype=torch.float32, device=self._dev)
        self.privileged_dim = 1 + int(self._priv_offsets.numel())
        obs_dim = B.BDX_OBS_DIM + self.privileged_dim
        self.teacher_obs_dim = obs_dim

        self.single_observation_space = spaces.Box(
            low=-B.OBS_CLIP, high=B.OBS_CLIP, shape=(obs_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-B.OBS_CLIP, high=B.OBS_CLIP,
            shape=(self.num_envs, obs_dim), dtype=np.float32)
        print(
            f"[bdx_corridor_teacher] scene={scene} obs={obs_dim} "
            f"(proprio {B.BDX_OBS_DIM} + privileged {self.privileged_dim}) "
            f"offsets={list(TEACHER_PROFILE_OFFSETS)}",
            flush=True,
        )

    def _structured_obs(self, proprio: torch.Tensor, *, force: bool = False):
        base = self._obs.base_pos()
        base_x = base[:, 0:1]
        base_z = base[:, 2:3]
        h_base = self._profile.heights_at(base_x)                      # (N,1)
        h_off = self._profile.heights_at(base_x + self._priv_offsets)  # (N,16)
        priv = torch.cat((base_z - h_base, h_off - h_base), dim=1)
        priv = priv.clamp(-PRIV_HEIGHT_CLAMP, PRIV_HEIGHT_CLAMP)
        return torch.cat((proprio, priv), dim=1)


def make_env(num_envs: int, *, device=None, scene: str = CORRIDOR_SCENE,
             **kwargs) -> BdxCorridorTeacherEnv:
    """Construct the flat privileged corridor teacher environment."""
    return BdxCorridorTeacherEnv(
        scene=scene, num_envs=num_envs, device=device, **kwargs)
