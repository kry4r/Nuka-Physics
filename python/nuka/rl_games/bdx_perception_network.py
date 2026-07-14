"""Compact depth/proprio fusion network for rl_games.

The image branch preserves spatial structure until a 96-dimensional latent.  The
44-dimensional proprio branch is the proven blind BDX actor MLP.  A zero-initialized
vision residual makes the initial policy exactly the blind checkpoint, then lets PPO
learn only the visual correction needed for sharp edges and media interaction.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from rl_games.algos_torch.network_builder import NetworkBuilder


class _DepthwiseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = 4 if out_channels % 4 == 0 else 1
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=1,
                groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DepthEncoder(nn.Module):
    """Small spatial encoder: 128x96 depth -> configurable compact latent."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(4, 16),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            _DepthwiseBlock(16, 24),
            _DepthwiseBlock(24, 32),
            _DepthwiseBlock(32, 48),
        )
        self.pool = nn.AdaptiveAvgPool2d((3, 4))
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 3 * 4, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(inplace=True),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2.0 ** 0.5)
                nn.init.zeros_(module.bias)

    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        # ExperienceBuffer stores the image as fp16; convolutions and optimizer
        # remain fp32 unless the trainer explicitly opts into mixed precision.
        x = depth.float()
        return self.projection(self.pool(self.blocks(self.stem(x))))


class BdxDepthFusionNetwork(NetworkBuilder.BaseNetwork):
    """Shared actor/critic with a compact visual residual."""

    def __init__(self, params: dict, **kwargs) -> None:
        super().__init__()
        actions_num = int(kwargs.pop("actions_num"))
        input_shape = kwargs.pop("input_shape")
        self.value_size = int(kwargs.pop("value_size", 1))
        self.num_seqs = int(kwargs.pop("num_seqs", 1))
        if not isinstance(input_shape, dict):
            raise TypeError("bdx_depth_fusion requires a Dict observation space")
        required = {"proprio", "depth", "depth_age"}
        if set(input_shape) != required:
            raise ValueError(
                f"bdx_depth_fusion observation keys={sorted(input_shape)}, "
                f"expected={sorted(required)}")
        proprio_dim = int(input_shape["proprio"][0])
        depth_shape = tuple(int(v) for v in input_shape["depth"])
        if len(depth_shape) != 3 or depth_shape[0] != 1:
            raise ValueError(f"depth must be CHW with C=1, got {depth_shape}")
        if tuple(input_shape["depth_age"]) != (1,):
            raise ValueError("depth_age must have shape (1,)")

        latent_dim = int(params.get("depth_latent_dim", 96))
        units = tuple(int(v) for v in params.get("proprio_units", (512, 256, 128)))
        if len(units) != 3:
            raise ValueError("proprio_units must contain exactly three widths")

        # Keep these names and layer indices compatible with the established
        # rl_games actor_critic checkpoint.  This is a behavioral warm start, not
        # a second policy path: all later checkpoints contain one fused network.
        self.actor_mlp = nn.Sequential(
            nn.Linear(proprio_dim, units[0]),
            nn.ELU(),
            nn.Linear(units[0], units[1]),
            nn.ELU(),
            nn.Linear(units[1], units[2]),
            nn.ELU(),
        )
        self.depth_encoder = _DepthEncoder(latent_dim)
        self.vision_adapter = nn.Linear(latent_dim + 1, units[2])
        nn.init.zeros_(self.vision_adapter.weight)
        nn.init.zeros_(self.vision_adapter.bias)

        self.mu = nn.Linear(units[2], actions_num)
        self.value = nn.Linear(units[2], self.value_size)
        self.sigma = nn.Parameter(torch.zeros(actions_num, dtype=torch.float32))

        # Fixed normalization from the blind policy preserves its exact initial
        # behavior while keeping the already-bounded depth branch independent.
        self.register_buffer("proprio_mean", torch.zeros(proprio_dim))
        self.register_buffer("proprio_var", torch.ones(proprio_dim))
        self.proprio_eps = float(params.get("proprio_epsilon", 1.0e-5))
        self.bootstrap_checkpoint = str(params.get("bootstrap_checkpoint", ""))
        if self.bootstrap_checkpoint:
            self._load_blind_bootstrap(self.bootstrap_checkpoint)

        parameter_count = sum(p.numel() for p in self.parameters())
        print(
            f"[bdx_depth_fusion] depth={depth_shape} -> {latent_dim}d; "
            f"proprio={proprio_dim}d; fused={units[-1]}d; "
            f"params={parameter_count:,}",
            flush=True,
        )

    @staticmethod
    def _plain_state(checkpoint: dict) -> dict:
        result = {}
        for key, value in checkpoint["model"].items():
            if key.startswith("_orig_mod."):
                key = key[len("_orig_mod."):]
            result[key] = value
        return result

    def _load_blind_bootstrap(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"blind bootstrap checkpoint not found: {path}")
        source = self._plain_state(
            torch.load(path, map_location="cpu", weights_only=False))
        pairs = {
            "actor_mlp.0.weight": "a2c_network.actor_mlp.0.weight",
            "actor_mlp.0.bias": "a2c_network.actor_mlp.0.bias",
            "actor_mlp.2.weight": "a2c_network.actor_mlp.2.weight",
            "actor_mlp.2.bias": "a2c_network.actor_mlp.2.bias",
            "actor_mlp.4.weight": "a2c_network.actor_mlp.4.weight",
            "actor_mlp.4.bias": "a2c_network.actor_mlp.4.bias",
            "mu.weight": "a2c_network.mu.weight",
            "mu.bias": "a2c_network.mu.bias",
            "value.weight": "a2c_network.value.weight",
            "value.bias": "a2c_network.value.bias",
            "sigma": "a2c_network.sigma",
            "proprio_mean": "running_mean_std.running_mean",
            "proprio_var": "running_mean_std.running_var",
        }
        target = dict(self.named_parameters())
        target.update(dict(self.named_buffers()))
        with torch.no_grad():
            for dst, src in pairs.items():
                if src not in source:
                    raise KeyError(f"bootstrap checkpoint is missing {src}")
                if target[dst].shape != source[src].shape:
                    raise ValueError(
                        f"bootstrap shape mismatch {src}={tuple(source[src].shape)} "
                        f"vs {dst}={tuple(target[dst].shape)}")
                target[dst].copy_(source[src])
        print(f"[bdx_depth_fusion] bootstrapped blind actor from {path}", flush=True)

    def is_rnn(self) -> bool:
        return False

    def get_default_rnn_state(self):
        return None

    def get_value_layer(self):
        return self.value

    def forward(self, input_dict: dict):
        obs = input_dict["obs"]
        proprio = obs["proprio"].float()
        normalized = torch.clamp(
            (proprio - self.proprio_mean)
            / torch.sqrt(self.proprio_var + self.proprio_eps),
            -5.0,
            5.0,
        )
        body = self.actor_mlp(normalized)
        visual = self.depth_encoder(obs["depth"])
        age = obs["depth_age"].float()
        fused = body + self.vision_adapter(torch.cat((visual, age), dim=-1))
        mu = self.mu(fused)
        value = self.value(fused)
        logstd = self.sigma.unsqueeze(0).expand_as(mu)
        return mu, logstd, value, None


class BdxDepthFusionBuilder(NetworkBuilder):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.params = {}

    def load(self, params: dict) -> None:
        self.params = params

    def build(self, name: str, **kwargs) -> BdxDepthFusionNetwork:
        return BdxDepthFusionNetwork(self.params, **kwargs)
