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

from rl_games.algos_torch import model_builder

from .bdx_perception_network import BdxDepthFusionBuilder
from .vecenv import NukaVecEnv, register

# rl_games resolves custom networks through this process-global registry.  Keep
# registration beside the env registration so one ``import nuka.rl_games`` wires
# the complete structured-observation pipeline.
model_builder.register_network("bdx_depth_fusion", BdxDepthFusionBuilder)

__all__ = ["NukaVecEnv", "BdxDepthFusionBuilder", "register"]
