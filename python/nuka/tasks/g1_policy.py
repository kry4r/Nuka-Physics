"""A bounded proprioceptive residual with optional RGB-D over the G1 velocity actor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Normal

from nuka.tasks.g1_locomotion import G1VelocityActor


@dataclass
class G1FusionConfig:
    use_camera: bool = True
    use_terrain: bool = False
    history_length: int = 3
    proprio_dim: int = 480
    tactile_dim: int = 16
    critic_dim: int = 30
    latent_dim: int = 96
    terrain_latent_dim: int = 64
    residual_scale: float = 1.0
    initial_std: float = 0.08


class G1FusionPolicy(nn.Module):
    """Only measured observations enter the actor; the critic has its own privileged input."""

    def __init__(self, prior: G1VelocityActor, config: G1FusionConfig | None = None):
        super().__init__()
        self.config = config or G1FusionConfig()
        cfg = self.config
        if min(cfg.history_length, cfg.latent_dim, cfg.terrain_latent_dim, cfg.initial_std, cfg.residual_scale) <= 0:
            raise ValueError("History, latent size, exploration and residual scale must be positive")
        self.prior = prior.requires_grad_(False)
        self.encoder = nn.Sequential(
            nn.Conv2d(5, 16, 5, stride=2, padding=2), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.SiLU(),
            nn.AdaptiveAvgPool2d((3, 4)), nn.Flatten(),
            nn.Linear(48 * 3 * 4, cfg.latent_dim), nn.LayerNorm(cfg.latent_dim), nn.SiLU()) if cfg.use_camera else None
        self.terrain_encoder = nn.Sequential(
            nn.Conv2d(5, 16, 3, padding=1), nn.ELU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ELU(),
            nn.AdaptiveAvgPool2d((4, 3)), nn.Flatten(),
            nn.Linear(32 * 4 * 3, cfg.terrain_latent_dim), nn.LayerNorm(cfg.terrain_latent_dim), nn.ELU()
        ) if cfg.use_terrain else None
        feature_dim = cfg.proprio_dim + cfg.tactile_dim
        if cfg.use_camera:
            feature_dim += cfg.history_length * cfg.latent_dim
        if cfg.use_terrain:
            feature_dim += cfg.history_length * cfg.terrain_latent_dim
        if cfg.use_camera or cfg.use_terrain:
            feature_dim += cfg.history_length * 2
        self.adapter = nn.Sequential(nn.Linear(feature_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 29))
        nn.init.zeros_(self.adapter[-1].weight)
        nn.init.zeros_(self.adapter[-1].bias)
        self.log_std = nn.Parameter(torch.full((29,), float(torch.tensor(cfg.initial_std).log())))
        self.value = nn.Sequential(nn.Linear(cfg.proprio_dim + cfg.tactile_dim + cfg.critic_dim, 256),
            nn.ELU(), nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 1))

    def residual_mean(self, observation: dict[str, torch.Tensor]):
        proprioception = torch.cat((observation["proprio"], observation["tactile"]), dim=-1)
        if not self.config.use_camera and not self.config.use_terrain:
            return self.adapter(proprioception)
        frame_valid = observation["frame_valid"].float()
        features = []
        if self.config.use_camera:
            rgb = observation["rgb"].float()
            images = torch.cat((rgb, observation["depth"].float(), observation["depth_valid"].float()), dim=2)
            encoded = self.encoder(images.flatten(0, 1)).reshape(
                rgb.shape[0], self.config.history_length, self.config.latent_dim)
            features.append(encoded * frame_valid)
        if self.config.use_terrain:
            terrain = observation["terrain"].float()
            encoded = self.terrain_encoder(terrain.flatten(0, 1)).reshape(
                terrain.shape[0], self.config.history_length, self.config.terrain_latent_dim)
            features.append(encoded * frame_valid)
        temporal = torch.cat((*features, frame_valid, observation["age"].clamp(0, 2)), dim=-1)
        return self.adapter(torch.cat((proprioception, temporal.flatten(1)), dim=-1))

    @torch.no_grad()
    def initialize_from(self, source, *, reset_load_inputs=False):
        if source.config.use_camera or source.config.use_terrain:
            raise ValueError("Adding perception inputs requires a proprioceptive source checkpoint")
        for name in ("proprio_dim", "tactile_dim", "critic_dim", "residual_scale"):
            if getattr(source.config, name) != getattr(self.config, name):
                raise ValueError(f"Initialization requires the same {name}")
        destination = self.state_dict()
        for name, value in source.state_dict().items():
            if name == "adapter.0.weight":
                destination[name].zero_()
                destination[name][:, :value.shape[1]].copy_(value)
            else:
                destination[name].copy_(value)
        if reset_load_inputs:
            begin = self.config.proprio_dim
            self.adapter[0].weight[:, begin:begin + self.config.tactile_dim].zero_()
            self.value[0].weight[:, begin:begin + self.config.tactile_dim].zero_()

    def distribution(self, observation):
        return Normal(self.residual_mean(observation), self.log_std.clamp(-5, 0).exp())

    def action(self, observation, latent):
        return self.prior(observation["proprio"]) + self.config.residual_scale * torch.tanh(latent)

    def forward(self, observation: dict[str, torch.Tensor]):
        return self.action(observation, self.residual_mean(observation))

    def sample(self, observation):
        distribution = self.distribution(observation)
        latent = distribution.sample()
        return self.action(observation, latent), latent, distribution.log_prob(latent).sum(-1)

    def critic(self, observation, privileged):
        return self.value(torch.cat((observation["proprio"], observation["tactile"], privileged), dim=-1)).squeeze(-1)

    def specification(self):
        layers = []
        for layer in self.prior.layers:
            if isinstance(layer, nn.Linear):
                layers.append({"kind": "linear", "input": layer.in_features, "output": layer.out_features})
            elif isinstance(layer, nn.ELU):
                layers.append({"kind": "elu", "alpha": layer.alpha})
            else:
                raise ValueError(f"Unsupported prior layer: {type(layer).__name__}")
        return {"version": 1, "fusion": asdict(self.config), "prior": layers}

    def save(self, path: str | Path, **training):
        path = Path(path)
        state = {"specification": self.specification(), "model": self.state_dict(), "training": training}
        pending = path.with_suffix(path.suffix + ".partial")
        torch.save(state, pending)
        pending.replace(path)

    @classmethod
    def load(cls, path: str | Path, *, device="cuda"):
        state = torch.load(path, map_location=device, weights_only=True)
        specification = state["specification"]
        if specification["version"] != 1:
            raise ValueError("Unsupported G1 policy checkpoint version")
        layers = []
        for layer in specification["prior"]:
            if layer["kind"] == "linear":
                layers.append(nn.Linear(layer["input"], layer["output"]))
            elif layer["kind"] == "elu":
                layers.append(nn.ELU(layer["alpha"]))
            else:
                raise ValueError("Unsupported G1 checkpoint prior layer")
        model = cls(G1VelocityActor(nn.Sequential(*layers)), G1FusionConfig(**specification["fusion"]))
        model.to(device)
        model.load_state_dict(state["model"], strict=True)
        return model, state["training"]
