"""Go2 privileged corridor teacher on the canonical one-shot corridor.

Reuses the shared :class:`~nuka.tasks.bdx_corridor.CorridorTaskMixin` (progress /
success shaping + start-window spawn randomization) and its parsed centre-line
:class:`~nuka.tasks.bdx_corridor.CorridorHeightProfile`, mixed into the Go2
locomotion host env instead of the duck.  The observation is the 48-d Go2 proprio
vector plus a privileged height profile (trunk clearance + 16 forward step-height
deltas), so PPO can anticipate the corridor's down-steps without a camera.

The corridor scene is authored boxes (no procedural terrain heightfield); the Go2
foot spheres contact them on the same general primitive-contact path the flat
walker uses, so no per-scene solver branch is introduced.
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from . import go2_obs as G
from .go2_locomotion import Go2LocomotionEnv
from .bdx_corridor import (
    CORRIDOR_SCENE as _BDX_CORRIDOR_SCENE,  # noqa: F401  (kept for parity)
    CorridorHeightProfile,
    CorridorTaskMixin,
    PRIV_HEIGHT_CLAMP,
    TEACHER_PROFILE_OFFSETS,
)


CORRIDOR_SCENE = "examples/scenes/corridor_go2.nks"
SLAB_GATE_X0 = 3.00
SLAB_GATE_X1 = 3.40


def slab_gate(base_x: torch.Tensor) -> torch.Tensor:
    """Smooth stage feature that is exactly zero before the hanging-panel zone."""
    u = ((base_x - SLAB_GATE_X0) / (SLAB_GATE_X1 - SLAB_GATE_X0)).clamp(0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


class Go2CorridorTeacherEnv(CorridorTaskMixin, Go2LocomotionEnv):
    """Flat privileged Go2 teacher: 48-d proprio + a parsed height profile.

    The observation is ``[proprio(48) | base_z - h(base_x) | h(base_x+off) -
    h(base_x)]`` over the 16 forward offsets = 65-d.  Termination / reward stay the
    Go2 recipe; the mixin adds the corridor progress / success / start-window logic.
    """

    def __init__(self, scene: str = CORRIDOR_SCENE, num_envs: int = 1024,
                 **kwargs) -> None:
        self.use_slab_gate_obs = bool(kwargs.pop("slab_gate_obs", False))
        self._pop_corridor_kwargs(kwargs)
        super().__init__(scene, num_envs, **kwargs)
        # The mixin + profile use ``self._dev``; the Go2 host names it differently.
        self._dev = self._torch_device
        self._finalize_spawn_range()
        self._init_progress_state()

        self._profile = CorridorHeightProfile(scene, self._dev)
        # Sample along world +x: the corridor command is a fixed yaw-0 forward walk.
        self._priv_offsets = torch.as_tensor(
            TEACHER_PROFILE_OFFSETS, dtype=torch.float32, device=self._dev)
        self.privileged_dim = (
            1 + int(self._priv_offsets.numel()) + int(self.use_slab_gate_obs))
        obs_dim = G.GO2_OBS_DIM + self.privileged_dim
        self.teacher_obs_dim = obs_dim

        self.single_observation_space = spaces.Box(
            low=-G.OBS_CLIP, high=G.OBS_CLIP, shape=(obs_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-G.OBS_CLIP, high=G.OBS_CLIP,
            shape=(self.num_envs, obs_dim), dtype=np.float32)
        print(
            f"[go2_corridor_teacher] scene={scene} obs={obs_dim} "
            f"(proprio {G.GO2_OBS_DIM} + privileged {self.privileged_dim}) "
            f"offsets={list(TEACHER_PROFILE_OFFSETS)}",
            flush=True,
        )

    def _recompute_proprio(self) -> torch.Tensor:
        """Go2 48-d proprio after a spawn teleport; the gravity slot is taken from
        the authoritative (un-lagged) base pose since FK trails a teleport."""
        obs = self._obs.compute_obs(self.command, self.last_action)
        obs[:, 6:9] = self._obs.projected_gravity_auth()
        return obs

    def _structured_obs(self, proprio: torch.Tensor, *, force: bool = False):
        base = self._obs.base_pos()
        base_x = base[:, 0:1]
        base_z = base[:, 2:3]
        h_base = self._profile.heights_at(base_x)                      # (N,1)
        h_off = self._profile.heights_at(base_x + self._priv_offsets)  # (N,16)
        priv = torch.cat((base_z - h_base, h_off - h_base), dim=1)
        priv = priv.clamp(-PRIV_HEIGHT_CLAMP, PRIV_HEIGHT_CLAMP)
        if self.use_slab_gate_obs:
            priv = torch.cat((priv, slab_gate(base_x)), dim=1)
        return torch.cat((proprio, priv), dim=1)


def make_env(num_envs: int, *, device=None, scene: str = CORRIDOR_SCENE,
             **kwargs) -> Go2CorridorTeacherEnv:
    """Construct the flat privileged Go2 corridor teacher environment."""
    return Go2CorridorTeacherEnv(
        scene=scene, num_envs=num_envs, device=device, **kwargs)
