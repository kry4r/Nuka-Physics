"""nuka.gym -- a vectorized gymnasium env backed by ``nuka.World``.

``NukaGymEnv`` exposes the standard vectorized-gym surface (``num_envs``,
``single_observation_space`` / ``single_action_space``, ``reset`` -> (obs, info),
``step`` -> the gymnasium 5-tuple) with **all tensors on cuda**. It composes the
48-dim legged_gym Go2 observation on-GPU (via :mod:`nuka.tasks.go2_obs`),
runs ``decimation`` physics sub-steps per control step holding the PD target,
computes a minimal forward-velocity-tracking + alive reward, and AUTORESETS
terminated envs via ``world.reset_envs``.

The reward / termination here are deliberately minimal -- the real reward
shaping is the task layer (p04). They are exposed as overridable hooks
(``compute_reward`` / ``compute_terminated``) so the task layer / a subclass can
replace them without touching the step loop.

This module imports torch at top level (torch is a hard dep of the RL layer),
which is fine: ``nuka.gym`` is only imported when you actually do RL. ``import
nuka`` itself stays torch-free (this submodule is NOT eagerly imported by
``nuka/__init__.py``).
"""

from __future__ import annotations

from .env import NukaGymEnv

__all__ = ["NukaGymEnv"]
