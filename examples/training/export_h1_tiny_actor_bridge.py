#!/usr/bin/env python
"""Export a deploy-shaped H1 PPO actor directly to the C++ tiny bridge format.

This is for checkpoints trained with the deploy-matched actor architecture:
32 -> 64 -> 64 -> 10, tanh hidden activations, linear mu output. The C++ bridge
receives raw 32-dim observations and outputs normalized actions. The C++ deploy
side clamps those actions to [-1, 1] and applies the same per-joint H1 torque-limit
scaling as training. Optional action biases are explicit deployment calibrations:
they are folded into the exported final-layer bias and covered by Gate-1 goldens.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
REPO = _HERE.parents[1]
OUT_DIR = REPO / "python/spikes/out"
WEIGHTS_BIN = OUT_DIR / "h1_bridge_mlp.bin"
GOLDENS_TXT = OUT_DIR / "h1_bridge_goldens.txt"
OBS_DIM = 32
ACT_DIM = 10
H = 64
EPS = 1.0e-5


def _parse_args():
    p = argparse.ArgumentParser(description="Export deploy-shaped H1 PPO actor to tiny bridge.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--num_envs", type=int, default=64)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--weights-out", default=str(WEIGHTS_BIN))
    p.add_argument("--goldens-out", default=str(GOLDENS_TXT))
    p.add_argument(
        "--action-bias",
        nargs=2,
        action="append",
        default=[],
        metavar=("SLOT", "VALUE"),
        help="Fold an explicit normalized-action calibration into output slot SLOT.",
    )
    return p.parse_args()


def _key(state: dict[str, torch.Tensor], suffix: str) -> str:
    matches = [k for k in state if k.endswith(suffix)]
    if len(matches) != 1:
        raise KeyError(f"expected one checkpoint key ending {suffix!r}, found {matches}")
    return matches[0]


def _parse_action_bias(pairs) -> torch.Tensor:
    bias = torch.zeros(ACT_DIM, dtype=torch.float32)
    for slot_text, value_text in pairs:
        slot = int(slot_text)
        if slot < 0 or slot >= ACT_DIM:
            raise ValueError(f"action-bias slot {slot} outside [0,{ACT_DIM})")
        bias[slot] += float(value_text)
    return bias


def _load_folded_actor(checkpoint: pathlib.Path, action_bias: torch.Tensor):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt["model"]
    w1 = state[_key(state, "a2c_network.actor_mlp.0.weight")].float().clone()
    b1 = state[_key(state, "a2c_network.actor_mlp.0.bias")].float().clone()
    w2 = state[_key(state, "a2c_network.actor_mlp.2.weight")].float().clone()
    b2 = state[_key(state, "a2c_network.actor_mlp.2.bias")].float().clone()
    w3 = state[_key(state, "a2c_network.mu.weight")].float().clone()
    b3 = state[_key(state, "a2c_network.mu.bias")].float().clone()
    if w1.shape != (H, OBS_DIM) or b1.shape != (H,) or w2.shape != (H, H) or b2.shape != (H,) or w3.shape != (ACT_DIM, H) or b3.shape != (ACT_DIM,):
        raise ValueError(
            "checkpoint is not deploy-shaped 32->64->64->10 actor: "
            f"w1={tuple(w1.shape)} w2={tuple(w2.shape)} w3={tuple(w3.shape)}"
        )

    mean_key = _key(state, "running_mean_std.running_mean")
    var_key = _key(state, "running_mean_std.running_var")
    mean = state[mean_key].float().clone()
    var = state[var_key].float().clone()
    scale = torch.rsqrt(var + EPS)

    # rl_games RunningMeanStd normalizes as clamp((obs - mean) / sqrt(var+eps), -5, 5).
    # For standing deployment, observed goldens remain far inside the clamp, so folding
    # the affine normalization into W1 is exact on the policy manifold and preserves the
    # C++ bridge shape without adding another runtime op.
    w1_fold = w1 * scale.view(1, OBS_DIM)
    b1_fold = b1 - (w1 * (mean * scale).view(1, OBS_DIM)).sum(dim=1)

    # PPO outputs normalized actions. Keep this output contract and let deploy apply
    # the same per-joint action-to-torque scaling used by H1StandEnv. Explicit deploy
    # calibrations are folded into the output bias so C++ Gate 1 validates them.
    b3 = b3 + action_bias.view(ACT_DIM)
    return w1_fold, b1_fold, w2, b2, w3, b3, mean, var, ckpt


def _forward(raw_obs: torch.Tensor, weights) -> torch.Tensor:
    w1, b1, w2, b2, w3, b3, _mean, _var, _ckpt = weights
    h1 = torch.tanh(torch.nn.functional.linear(raw_obs.float(), w1, b1))
    h2 = torch.tanh(torch.nn.functional.linear(h1, w2, b2))
    return torch.nn.functional.linear(h2, w3, b3)


def _forward_cpp_style(raw_obs: torch.Tensor, weights) -> torch.Tensor:
    """Match tests/coresident/tiny_mlp.hpp's f32 storage + f64 accumulation."""
    w1, b1, w2, b2, w3, b3, _mean, _var, _ckpt = weights
    x = raw_obs.detach().cpu().numpy().astype(np.float32)
    w1n, b1n = w1.numpy().astype(np.float32), b1.numpy().astype(np.float32)
    w2n, b2n = w2.numpy().astype(np.float32), b2.numpy().astype(np.float32)
    w3n, b3n = w3.numpy().astype(np.float32), b3.numpy().astype(np.float32)
    out = np.empty((x.shape[0], ACT_DIM), dtype=np.float32)
    for r in range(x.shape[0]):
        h1 = np.empty(H, dtype=np.float32)
        for o in range(H):
            acc = np.float64(b1n[o])
            for i in range(OBS_DIM):
                acc += np.float64(w1n[o, i]) * np.float64(x[r, i])
            h1[o] = np.tanh(np.float32(acc)).astype(np.float32)
        h2 = np.empty(H, dtype=np.float32)
        for o in range(H):
            acc = np.float64(b2n[o])
            for i in range(H):
                acc += np.float64(w2n[o, i]) * np.float64(h1[i])
            h2[o] = np.tanh(np.float32(acc)).astype(np.float32)
        for o in range(ACT_DIM):
            acc = np.float64(b3n[o])
            for i in range(H):
                acc += np.float64(w3n[o, i]) * np.float64(h2[i])
            out[r, o] = np.float32(acc)
    return torch.from_numpy(out)


def _export_weights(weights, path: pathlib.Path) -> None:
    w1, b1, w2, b2, w3, b3, *_ = weights
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = torch.cat([x.detach().reshape(-1).cpu().float() for x in (w1, b1, w2, b2, w3, b3)]).numpy().astype(np.float32)
    expect = (H * OBS_DIM + H) + (H * H + H) + (ACT_DIM * H + ACT_DIM)
    if flat.size != expect:
        raise ValueError(f"flat size {flat.size}, expected {expect}")
    path.write_bytes(flat.tobytes())
    print(f"[export_actor] wrote {flat.size} f32 -> {path}", flush=True)


def _collect_goldens(num_envs: int, steps: int, seed: int, weights):
    import nuka.rl_games  # noqa: F401
    from nuka.tasks.h1_stand import make_env

    env = make_env(num_envs, seed=seed)
    obs_chunks = []
    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(steps):
            with torch.no_grad():
                action = _forward(obs.detach().cpu(), weights).cuda().clamp(-1.0, 1.0)
            ok = torch.isfinite(obs).all(dim=1) & torch.isfinite(action).all(dim=1)
            if bool(ok.any()):
                obs_chunks.append(obs[ok].detach().cpu().float())
            obs, _reward, _terminated, _truncated, _info = env.step(action)
    finally:
        env.close()
    if not obs_chunks:
        raise RuntimeError("failed to collect any finite observations for goldens")
    return torch.cat(obs_chunks, dim=0)


def _export_goldens(obs: torch.Tensor, weights, path: pathlib.Path, k: int = 16) -> None:
    n = obs.shape[0]
    idx = np.unique(np.linspace(0, n - 1, k, dtype=int))[:k]
    sel = obs[idx].float()
    with torch.no_grad():
        act = _forward_cpp_style(sel, weights).float()
    sel_np = sel.numpy().astype(np.float32)
    act_np = act.numpy().astype(np.float32)
    lines = [f"{len(idx)} {OBS_DIM} {ACT_DIM}"]
    for row in range(len(idx)):
        raw = " ".join(f"{v:.9e}" for v in sel_np[row])
        ob = " ".join(f"{v:.9e}" for v in sel_np[row])
        ac = " ".join(f"{v:.9e}" for v in act_np[row])
        lines.append(f"{raw} | {ob} | {ac}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"[export_actor] wrote {len(idx)} goldens -> {path}", flush=True)


def main() -> int:
    args = _parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        print("[export_actor] WARNING: CUDA_VISIBLE_DEVICES unset; use CUDA_VISIBLE_DEVICES=0.", file=sys.stderr)
    action_bias = _parse_action_bias(args.action_bias)
    nonzero_bias = [(i, float(v)) for i, v in enumerate(action_bias) if float(v) != 0.0]
    if nonzero_bias:
        print(f"[export_actor] folding action_bias={nonzero_bias}", flush=True)
    weights = _load_folded_actor(pathlib.Path(args.checkpoint), action_bias)
    obs = _collect_goldens(args.num_envs, args.steps, args.seed, weights)
    mean, var = weights[6], weights[7]
    norm = ((obs - mean.view(1, OBS_DIM)) * torch.rsqrt(var.view(1, OBS_DIM) + EPS)).abs()
    print(
        f"[export_actor] collected obs={tuple(obs.shape)} max_norm={float(norm.max()):.3f} "
        f"clamp_rows={(norm > 5.0).any(dim=1).sum().item()}",
        flush=True,
    )
    with torch.no_grad():
        action = _forward(obs[: min(obs.shape[0], 65536)], weights)
        print(
            f"[export_actor] action range=[{float(action.min()):.3f}, {float(action.max()):.3f}] "
            f"max_abs={float(action.abs().max()):.3f}",
            flush=True,
        )
    _export_weights(weights, pathlib.Path(args.weights_out))
    _export_goldens(obs, weights, pathlib.Path(args.goldens_out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
