"""nuka.tasks -- RL task definitions + on-GPU observation builders.

This package holds the reusable, engine-facing pieces that both the pure-gym
layer (:mod:`nuka.gym`) and the Isaac Lab compat task configs build on:

* :mod:`nuka.tasks.go2_obs` -- the on-GPU 48-dim legged_gym Go2 observation
  builder, the Nuka-slot <-> URDF-index joint permutation discovery, and the
  Go2 obs/action constants (DEFAULT_ANGLES, Kp/Kd, scales, force limits).

The full ``go2_locomotion`` ManagerBasedRLEnv task config + the rl_games train
script are p03-T3 (deferred); this package ships only the obs/permutation
machinery that T3 will compose into a task.
"""

from __future__ import annotations

from . import go2_obs  # noqa: F401

__all__ = ["go2_obs"]
