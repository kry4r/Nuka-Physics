#!/usr/bin/env python3
"""Import the external 48-d Go2 TorchScript actor into an rl_games checkpoint.

The generated checkpoint owns a newly initialized critic and optimizer. Actor
columns 0:48 are copied exactly; optional privileged columns are zero, so a 65-d
actor is bit-exact to the source whenever those columns are zero.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
import yaml
from rl_games.algos_torch import model_builder

from nuka.rl_games.go2_policy import ACTION_RAW, CONTRACT_KEY, actor_mu, plain_model_state
from nuka.tasks import go2_obs as G


_ACTOR_KEYS = {
    "0.weight": "a2c_network.actor_mlp.0.weight",
    "0.bias": "a2c_network.actor_mlp.0.bias",
    "2.weight": "a2c_network.actor_mlp.2.weight",
    "2.bias": "a2c_network.actor_mlp.2.bias",
    "4.weight": "a2c_network.actor_mlp.4.weight",
    "4.bias": "a2c_network.actor_mlp.4.bias",
    "6.weight": "a2c_network.mu.weight",
    "6.bias": "a2c_network.mu.bias",
}


def _load_params(path: Path) -> dict:
    with path.open() as stream:
        raw = yaml.safe_load(stream)
    return copy.deepcopy(raw["params"] if "params" in raw else raw)


def _build_model(params: dict, obs_dim: int):
    config = params["config"]
    if bool(config.get("normalize_input", False)):
        raise ValueError("import config must set normalize_input: false")
    network = params["network"]
    if not bool(network.get("separate", False)):
        raise ValueError("import config must set network.separate: true")
    builder = model_builder.ModelBuilder().load(params)
    return builder.build({
        "actions_num": G.GO2_ACTION_DIM,
        "input_shape": (obs_dim,),
        "num_seqs": 1,
        "value_size": 1,
        "normalize_value": bool(config.get("normalize_value", True)),
        "normalize_input": False,
    })


def _copy_actor(model, source, obs_dim: int) -> None:
    source_state = source.state_dict()
    target_state = model.state_dict()
    for source_key, target_key in _ACTOR_KEYS.items():
        value = source_state[source_key].detach().clone()
        target = target_state[target_key]
        if target_key.endswith("actor_mlp.0.weight"):
            if tuple(value.shape) != (512, G.GO2_OBS_DIM):
                raise ValueError(
                    f"source first layer must be (512, {G.GO2_OBS_DIM}), "
                    f"got {tuple(value.shape)}")
            if tuple(target.shape) != (512, obs_dim):
                raise ValueError(
                    f"target first layer must be (512, {obs_dim}), "
                    f"got {tuple(target.shape)}")
            target.zero_()
            target[:, :G.GO2_OBS_DIM].copy_(value)
        else:
            if target.shape != value.shape:
                raise ValueError(
                    f"shape mismatch for {target_key}: {tuple(target.shape)} "
                    f"!= {tuple(value.shape)}")
            target.copy_(value)
    model.load_state_dict(target_state)


def import_prior(source_path: Path, config_path: Path, output_path: Path,
                 obs_dim: int) -> dict:
    if obs_dim < G.GO2_OBS_DIM:
        raise ValueError(f"obs_dim must be at least {G.GO2_OBS_DIM}")
    params = _load_params(config_path)
    source = torch.jit.load(str(source_path), map_location="cpu").eval()
    model = _build_model(params, obs_dim)
    _copy_actor(model, source, obs_dim)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(params["config"]["learning_rate"]),
        eps=1.0e-8, fused=False)
    checkpoint = {
        "model": model.state_dict(),
        "epoch": 0,
        "frame": 0,
        "optimizer": optimizer.state_dict(),
        "last_mean_rewards": -1.0e9,
        "env_state": None,
        CONTRACT_KEY: {
            "version": 1,
            "provenance": f"imported TorchScript actor: {source_path}",
            "source_format": "torchscript",
            "source_obs_dim": G.GO2_OBS_DIM,
            "obs_dim": obs_dim,
            "privileged_zero_columns": [G.GO2_OBS_DIM, obs_dim],
            "normalize_input": False,
            "action_postprocess": ACTION_RAW,
        },
    }

    generator = torch.Generator().manual_seed(20260723)
    proprio = torch.randn(1024, G.GO2_OBS_DIM, generator=generator)
    expanded = torch.zeros(1024, obs_dim)
    expanded[:, :G.GO2_OBS_DIM] = proprio
    with torch.inference_mode():
        source_action = source(proprio)
        imported_action = actor_mu(
            plain_model_state(checkpoint), expanded, normalize_input=False)
    max_actor_delta = float((source_action - imported_action).abs().max())
    if max_actor_delta != 0.0:
        raise ValueError(f"actor equivalence failed: max delta {max_actor_delta}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    return {
        "source": str(source_path),
        "config": str(config_path),
        "output": str(output_path),
        "obs_dim": obs_dim,
        "action_dim": G.GO2_ACTION_DIM,
        "max_actor_delta": max_actor_delta,
        "optimizer_state_count": len(checkpoint["optimizer"]["state"]),
        "action_postprocess": ACTION_RAW,
        "normalize_input": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--obs-dim", type=int, choices=(48, 65), default=65)
    args = parser.parse_args()
    print(json.dumps(import_prior(
        args.source, args.config, args.output, args.obs_dim), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
