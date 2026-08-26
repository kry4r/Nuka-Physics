"""Wall-clock gait/command module for the Go2 front-paw handstand walk.

Mirrors Sang-SC/go2_quad2hand: 0.4 s gait period, front paws anti-phase,
clock sin/cos in observations, and velocity commands resampled per episode.
"""
from __future__ import annotations

import torch

CMD_DIM = 8
GAIT_PERIOD = 0.4
STANCE_RATIO = 0.6


class HandstandWalkCommander:
    def __init__(self, num_envs: int, device: torch.device,
                 fixed_vx: float | None = None) -> None:
        self.n = int(num_envs)
        self.dev = device
        self.fixed_vx = fixed_vx
        self.phase = torch.zeros(self.n, device=device)
        self.vx = torch.zeros(self.n, device=device)
        self.yaw = torch.zeros(self.n, device=device)
        self._offsets = torch.tensor([0.0, 0.5, 0.0, 0.0], device=device)

    @property
    def leg_phase(self) -> torch.Tensor:
        return (self.phase.unsqueeze(1) + self._offsets) % 1.0

    @property
    def command(self) -> torch.Tensor:
        return torch.stack((self.vx, torch.zeros_like(self.vx), self.yaw), dim=-1)

    def advance(self, dt: float) -> None:
        self.phase = (self.phase + dt / GAIT_PERIOD) % 1.0

    def resample(self, mask: torch.Tensor, generator: torch.Generator) -> None:
        if not bool(mask.any()):
            return
        u = torch.rand(self.n, generator=generator, device=self.dev)
        vx = -0.4 + 0.8 * u if self.fixed_vx is None else torch.full_like(u, self.fixed_vx)
        u = torch.rand(self.n, generator=generator, device=self.dev)
        yaw = -0.5 + u
        self.vx = torch.where(mask, vx, self.vx)
        self.yaw = torch.where(mask, yaw, self.yaw)
        self.phase = torch.where(mask, torch.zeros_like(self.phase), self.phase)

    def is_stance(self) -> torch.Tensor:
        return self.leg_phase < STANCE_RATIO

    def cmd_tensor(self) -> torch.Tensor:
        phase = 2.0 * torch.pi * self.leg_phase
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=-1)
