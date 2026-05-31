"""``ManagerBasedRLEnv`` + ``ManagerBasedRLEnvCfg`` (Isaac Lab RL-training subset).

A manager-based vectorized RL env: an :class:`SimulationContext` drives the
physics, and the obs / reward / termination / action / command managers compose
the MDP from user-supplied term functions. The step loop mirrors Isaac Lab:

    process action -> step_n(decimation) -> compute obs / reward / terminations
    -> AUTORESET the done envs (masked) -> return the gymnasium 5-tuple.

Autoreset is LAST so the returned obs is the terminal (post-step, pre-reset)
obs and the one-step ARTICULATION_LINK_POSE lag never bites a base-pose obs term
(reset shows up next step). Reward / obs / termination are pure torch on the
zero-copy DLPack views (the term funcs decide; nothing here round-trips numpy).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .sim import SimulationContext
from .managers import (
    ObservationManager,
    RewardManager,
    TerminationManager,
    ActionManager,
    CommandManager,
)


@dataclass
class ManagerBasedRLEnvCfg:
    """Config for :class:`ManagerBasedRLEnv` (term lists + sim params)."""

    scene_path: str
    num_envs: int = 1
    sim_dt: float = 0.005
    decimation: int = 4  # control freq = sim freq / decimation
    episode_length_s: float = 20.0
    command: tuple = (0.5, 0.0, 0.0)
    reward_terms: list = field(default_factory=list)
    obs_terms: list = field(default_factory=list)
    termination_terms: list = field(default_factory=list)
    action_term: object = None


class ManagerBasedRLEnv:
    """Isaac Lab ``ManagerBasedRLEnv`` drop-in (RL training path only)."""

    def __init__(self, cfg: ManagerBasedRLEnvCfg, device=None):
        self.cfg = cfg
        self.num_envs = int(cfg.num_envs)
        self.decimation = int(cfg.decimation)
        self.max_episode_length = max(
            1, int(round(cfg.episode_length_s / (cfg.sim_dt * cfg.decimation)))
        )

        self.sim = SimulationContext(
            cfg.scene_path, self.num_envs, cfg.sim_dt, device=device
        )
        # Convenience: managers + term funcs read state off the world.
        self.world = self.sim.world

        self.command_manager = CommandManager(self, cfg.command)
        self.observation_manager = ObservationManager(self, cfg.obs_terms)
        self.reward_manager = RewardManager(self, cfg.reward_terms)
        self.termination_manager = TerminationManager(self, cfg.termination_terms)
        self.action_manager = ActionManager(self, cfg.action_term)

        self.episode_step = torch.zeros(
            self.num_envs, dtype=torch.long, device="cuda"
        )

    # -- gym-style API ------------------------------------------------------
    def reset(self, env_ids=None):
        """Reset (all envs in v0.3) and return ``(obs, info)``.

        Composes obs from the un-stepped creation buffers (valid right after
        reset-all -- the one-step pose lag only follows a *running* world's
        reset_envs)."""
        self.sim.reset()
        self.sim.sync()
        self.episode_step.zero_()
        self.action_manager.last_action = torch.zeros(
            self.num_envs, self.world.action_dim, device="cuda"
        )
        self.command_manager.reset()
        obs = self.observation_manager.compute()
        return obs, {}

    def step(self, actions: torch.Tensor):
        """One control step -> gymnasium 5-tuple (all tensors cuda)."""
        self.action_manager.process_action(actions)
        self.sim.step(self.decimation)
        self.episode_step += 1

        obs = self.observation_manager.compute()
        reward = self.reward_manager.compute()
        terminated, truncated = self.termination_manager.compute()
        # Episode-length timeout folds into truncation.
        truncated = truncated | (self.episode_step >= self.max_episode_length)
        info: dict = {}

        done = terminated | truncated
        if bool(done.any()):
            done_ids = torch.nonzero(done, as_tuple=False).flatten().to(torch.int32)
            self.sim.reset_envs(done_ids.cpu())
            self.episode_step[done] = 0
            if self.action_manager.last_action is not None:
                self.action_manager.last_action[done] = 0.0

        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        self.sim.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
