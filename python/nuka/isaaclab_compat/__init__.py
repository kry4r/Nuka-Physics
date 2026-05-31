"""nuka.isaaclab_compat -- minimal Isaac Lab ``ManagerBasedRLEnv`` drop-in.

Implements ONLY the RL-training subset of Isaac Lab's manager-based API so a
user can write reward / observation / termination functions that match by
signature and migrate trivially. **No UI, no editor, no Omniverse Kit.**

Public surface:

* :class:`SimulationContext` -- thin wrapper over ``nuka.World``.
* term dataclasses + managers: :class:`ObservationTerm` /
  :class:`ObservationManager`, :class:`RewardTerm` / :class:`RewardManager`,
  :class:`TerminationTerm` / :class:`TerminationManager`, :class:`ActionTerm` /
  :class:`ActionManager`, :class:`CommandManager`.
* :class:`ManagerBasedRLEnvCfg` + :class:`ManagerBasedRLEnv`.

Like :mod:`nuka.gym`, this imports torch at module top (it is a hard dep of the
RL layer); it is NOT eagerly imported by ``nuka/__init__.py``, so bare
``import nuka`` stays torch-free.
"""

from __future__ import annotations

from .sim import SimulationContext
from .managers import (
    ObservationTerm,
    ObservationManager,
    RewardTerm,
    RewardManager,
    TerminationTerm,
    TerminationManager,
    ActionTerm,
    ActionManager,
    CommandManager,
)
from .envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg

__all__ = [
    "SimulationContext",
    "ObservationTerm",
    "ObservationManager",
    "RewardTerm",
    "RewardManager",
    "TerminationTerm",
    "TerminationManager",
    "ActionTerm",
    "ActionManager",
    "CommandManager",
    "ManagerBasedRLEnv",
    "ManagerBasedRLEnvCfg",
]
