"""Shared rl_games Go2 checkpoint contract and deterministic actor forward."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F


CONTRACT_KEY = "nuka_go2_policy"
ACTION_RAW = "raw"
ACTION_CLAMP = "clamp_-1_1"
_SUPPORTED_ACTION_POSTPROCESS = {ACTION_RAW, ACTION_CLAMP}


@dataclass(frozen=True)
class Go2PolicyContract:
    """Deployment behavior carried beside an rl_games checkpoint."""

    action_postprocess: str = ACTION_CLAMP
    normalize_input: bool = True
    provenance: str = "legacy rl_games checkpoint"


def plain_model_state(checkpoint: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    """Return checkpoint model tensors without the torch.compile key prefix."""
    model = checkpoint.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("checkpoint does not contain an rl_games model mapping")
    return {
        key.removeprefix("_orig_mod."): value
        for key, value in model.items()
    }


def load_contract(checkpoint: Mapping[str, Any]) -> Go2PolicyContract:
    """Read explicit metadata, preserving historical rl_games behavior by default."""
    raw = checkpoint.get(CONTRACT_KEY)
    if raw is None:
        env_state = checkpoint.get("env_state")
        if isinstance(env_state, Mapping):
            raw = env_state.get(CONTRACT_KEY)
    if raw is None:
        return Go2PolicyContract()
    if not isinstance(raw, Mapping):
        raise ValueError(f"{CONTRACT_KEY} must be a mapping")
    action_postprocess = str(raw.get("action_postprocess", ACTION_CLAMP))
    if action_postprocess not in _SUPPORTED_ACTION_POSTPROCESS:
        raise ValueError(
            f"unsupported action_postprocess {action_postprocess!r}; expected "
            f"one of {sorted(_SUPPORTED_ACTION_POSTPROCESS)}")
    return Go2PolicyContract(
        action_postprocess=action_postprocess,
        normalize_input=bool(raw.get("normalize_input", True)),
        provenance=str(raw.get("provenance", "rl_games checkpoint")),
    )


def actor_mu(
    state: Mapping[str, torch.Tensor],
    obs: torch.Tensor,
    *,
    normalize_input: bool = True,
) -> torch.Tensor:
    """Run the deterministic 3-layer ELU actor encoded in a Go2 checkpoint."""
    x = obs.float()
    if normalize_input:
        mean = state["running_mean_std.running_mean"].to(x.device).float()
        var = state["running_mean_std.running_var"].to(x.device).float()
        x = ((x - mean) / torch.sqrt(var + 1.0e-5)).clamp(-5.0, 5.0)
    for index in (0, 2, 4):
        x = F.elu(F.linear(
            x,
            state[f"a2c_network.actor_mlp.{index}.weight"].to(x.device),
            state[f"a2c_network.actor_mlp.{index}.bias"].to(x.device),
        ))
    return F.linear(
        x,
        state["a2c_network.mu.weight"].to(x.device),
        state["a2c_network.mu.bias"].to(x.device),
    )


def postprocess_action(action: torch.Tensor, contract: Go2PolicyContract) -> torch.Tensor:
    """Apply the checkpoint's deployment action contract."""
    if contract.action_postprocess == ACTION_RAW:
        return action
    return action.clamp(-1.0, 1.0)
