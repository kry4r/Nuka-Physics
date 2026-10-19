"""Export an rl_games actor and its input normalizer to NKML, the file the C++ MlpPolicy runs.

The .pth is an rl_games A2C checkpoint: running_mean_std (input normalize) -> actor_mlp
(Linear/ELU pairs) -> mu (Linear). The value head and sigma are dropped; the state_dict tensors
are read with torch alone.
"""

from __future__ import annotations

import struct

import numpy as np
import torch

MAGIC = 0x4E4B4D4C  # "NKML"
NORM_EPS = 1.0e-5    # rl_games RunningMeanStd epsilon
NORM_CLIP = 5.0      # rl_games RunningMeanStd clamp
ELU_ALPHA = 1.0


def load_state_dict(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    # Strip the torch.compile "_orig_mod." prefix if present.
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


def actor_layers(sd):
    """The Linear layers of actor_mlp (indices 0,2,4 -> ELU) + the mu head."""
    layers = []
    idx = 0
    while f"a2c_network.actor_mlp.{idx}.weight" in sd:
        w = sd[f"a2c_network.actor_mlp.{idx}.weight"].float().numpy()
        b = sd[f"a2c_network.actor_mlp.{idx}.bias"].float().numpy()
        layers.append((w, b, 1))  # 1 == ELU activation
        idx += 2                   # skip the activation module slot
    mw = sd["a2c_network.mu.weight"].float().numpy()
    mb = sd["a2c_network.mu.bias"].float().numpy()
    layers.append((mw, mb, 0))     # 0 == no activation (linear mu head)
    return layers


def torch_forward(sd, obs):
    """The reference rl_games actor forward (the validation ground truth).

    rl_games RunningMeanStd normalizes in float32: (x - mean.float()) /
    sqrt(var.float() + eps), then clamps +/-5 -- replicated exactly here.
    """
    mean = sd["running_mean_std.running_mean"].float()
    var = sd["running_mean_std.running_var"].float()
    x = torch.from_numpy(obs).float()
    x = (x - mean) / torch.sqrt(var + NORM_EPS)
    x = torch.clamp(x, -NORM_CLIP, NORM_CLIP)
    idx = 0
    while f"a2c_network.actor_mlp.{idx}.weight" in sd:
        w = sd[f"a2c_network.actor_mlp.{idx}.weight"].float()
        b = sd[f"a2c_network.actor_mlp.{idx}.bias"].float()
        x = torch.nn.functional.elu(torch.nn.functional.linear(x, w, b), ELU_ALPHA)
        idx += 2
    mw = sd["a2c_network.mu.weight"].float()
    mb = sd["a2c_network.mu.bias"].float()
    return torch.nn.functional.linear(x, mw, mb).numpy()


def write_nkml(path, sd):
    """Writes the actor of a state_dict; returns (obs_dim, act_dim)."""
    # f32 mean + f32 std (== rl_games' torch.sqrt(var.float()+eps)); the C++ runner
    # divides (obs-mean)/std in f32 -- bit-faithful to the reference normalize.
    mean_f = sd["running_mean_std.running_mean"].float().numpy()
    var_f = sd["running_mean_std.running_var"].float().numpy()
    std_f = np.sqrt(var_f + np.float32(NORM_EPS)).astype(np.float32)
    obs_dim = int(mean_f.shape[0])
    layers = actor_layers(sd)
    with open(path, "wb") as f:
        f.write(struct.pack("<IIIf f", MAGIC, 1, obs_dim, NORM_CLIP, ELU_ALPHA))
        f.write(mean_f.astype(np.float32).tobytes())
        f.write(std_f.tobytes())
        f.write(struct.pack("<I", len(layers)))
        for w, b, act in layers:
            out_dim, in_dim = w.shape
            f.write(struct.pack("<III", int(in_dim), int(out_dim), int(act)))
            f.write(w.astype(np.float32).tobytes())   # row-major [out, in]
            f.write(b.astype(np.float32).tobytes())
    return obs_dim, layers[-1][0].shape[0]


def write_golden(path, sd, obs_dim, act_dim, n):
    rng = np.random.default_rng(20260627)
    mean = sd["running_mean_std.running_mean"].double().numpy()
    std = np.sqrt(sd["running_mean_std.running_var"].double().numpy())
    # Mostly in-distribution (mean +/- a few std); a tail of extremes to exercise
    # the normalize clamp + ELU negative branch.
    obs = mean + std * rng.standard_normal((n, obs_dim))
    obs[: n // 5] = mean + std * (8.0 * rng.standard_normal((n // 5, obs_dim)))
    obs = obs.astype(np.float32)
    mu = torch_forward(sd, obs).astype(np.float32)
    with open(path, "wb") as f:
        f.write(struct.pack("<III", n, obs_dim, act_dim))
        f.write(obs.tobytes())
        f.write(mu.tobytes())


def export(checkpoint, out):
    """Writes the NKML of an rl_games checkpoint; returns (obs_dim, act_dim)."""
    return write_nkml(out, load_state_dict(checkpoint))
