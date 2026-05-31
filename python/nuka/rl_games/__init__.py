"""Nuka <-> rl_games (1.6.5) integration.

Importing this package registers the 'NUKA' vecenv type and the 'nuka_go2' env
name with rl_games, so an rl_games yaml ``config.env_name: nuka_go2`` resolves to
the on-GPU Go2 locomotion env. See :mod:`nuka.rl_games.vecenv` for the exact
IVecEnv contract matched.

Usage::

    import nuka.rl_games                      # noqa: F401  (side-effect: register)
    from rl_games.torch_runner import Runner
    Runner().run({'train': True, ...})
"""

from __future__ import annotations

from .vecenv import NukaVecEnv, register

__all__ = ["NukaVecEnv", "register"]
