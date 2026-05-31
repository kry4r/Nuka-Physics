"""Minimal manager-based term machinery (Isaac Lab RL-training subset).

The dataclass term + manager pattern matches the p03 spec's API shape so a user
can migrate Isaac Lab reward / observation / termination / action functions by
signature. Each term's ``func`` takes the env and returns a torch tensor on
cuda; managers compose them. This is intentionally tiny -- no noise models, no
curriculum, no group/concatenation config beyond a flat concat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
@dataclass
class ObservationTerm:
    """One observation component. ``func(env) -> (num_envs, k) cuda tensor``."""

    name: str
    func: Callable[["object"], torch.Tensor]


class ObservationManager:
    """Concatenates the obs terms along the feature axis (flat policy group)."""

    def __init__(self, env, terms):
        self._env = env
        self._terms = list(terms or [])

    @property
    def terms(self):
        return self._terms

    def compute(self) -> torch.Tensor:
        if not self._terms:
            return torch.zeros(self._env.num_envs, 0, device="cuda")
        return torch.cat([t.func(self._env) for t in self._terms], dim=-1)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
@dataclass
class RewardTerm:
    """One reward component. ``func(env) -> (num_envs,) cuda tensor``; the
    weighted sum over terms is the scalar per-env reward."""

    name: str
    func: Callable[["object"], torch.Tensor]
    weight: float = 1.0


class RewardManager:
    def __init__(self, env, terms):
        self._env = env
        self._terms = list(terms or [])

    @property
    def terms(self):
        return self._terms

    def compute(self) -> torch.Tensor:
        total = torch.zeros(self._env.num_envs, device="cuda")
        for term in self._terms:
            total = total + term.weight * term.func(self._env)
        return total


# ---------------------------------------------------------------------------
# Termination
# ---------------------------------------------------------------------------
@dataclass
class TerminationTerm:
    """One termination condition. ``func(env) -> (num_envs,) cuda bool tensor``.

    ``time_out=True`` marks the term as a truncation (episode timeout) rather
    than an MDP termination -- the manager tracks the two separately so the env
    can return the gymnasium ``terminated`` / ``truncated`` split."""

    name: str
    func: Callable[["object"], torch.Tensor]
    time_out: bool = False


class TerminationManager:
    def __init__(self, env, terms):
        self._env = env
        self._terms = list(terms or [])
        self._terminated = None
        self._truncated = None

    @property
    def terms(self):
        return self._terms

    def compute(self):
        """Return ``(terminated, truncated)`` -- each (num_envs,) cuda bool."""
        n = self._env.num_envs
        terminated = torch.zeros(n, dtype=torch.bool, device="cuda")
        truncated = torch.zeros(n, dtype=torch.bool, device="cuda")
        for term in self._terms:
            mask = term.func(self._env).to(torch.bool)
            if term.time_out:
                truncated = truncated | mask
            else:
                terminated = terminated | mask
        self._terminated, self._truncated = terminated, truncated
        return terminated, truncated

    @property
    def dones(self) -> torch.Tensor:
        return self._terminated | self._truncated


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------
@dataclass
class ActionTerm:
    """How to apply a raw policy action to the world.

    ``func(env, action) -> processed_action`` writes the action into the engine
    (e.g. PD targets) and returns the stored (clipped) action. If ``func`` is
    ``None`` the :class:`ActionManager` no-ops (useful for a passive env)."""

    name: str
    func: Callable[["object", torch.Tensor], torch.Tensor] | None = None


class ActionManager:
    def __init__(self, env, action_term=None):
        self._env = env
        self._term = action_term
        self.last_action = None

    def process_action(self, action: torch.Tensor) -> torch.Tensor:
        action = action.to("cuda", dtype=torch.float32)
        if self._term is not None and self._term.func is not None:
            action = self._term.func(self._env, action)
        self.last_action = action
        return action


# ---------------------------------------------------------------------------
# Command
# ---------------------------------------------------------------------------
class CommandManager:
    """Holds the per-env velocity command. v0.3 is a fixed broadcast command
    (no resampling / curriculum). ``command``: (num_envs, 3) cuda tensor."""

    def __init__(self, env, command=(0.5, 0.0, 0.0)):
        self._env = env
        self.command = torch.tensor(
            command, dtype=torch.float32, device="cuda"
        ).expand(env.num_envs, 3).contiguous()

    @property
    def command_velocity(self) -> torch.Tensor:
        return self.command

    def reset(self, env_ids=None) -> None:
        # Fixed command in v0.3 -- nothing to resample.
        return None
