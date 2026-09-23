"""Measured G1 observations, episodic course rewards and joint physical/sensor replay."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

import torch

from nuka.tasks.g1_locomotion import G1VelocityController, locomotion_outcome, projected_gravity, rotate_vector
from nuka.tasks.g1_perception import G1RgbdHistory, ProprioceptionConfig, RgbdConfig
from nuka.tasks.g1_terrain import TerrainConfig


DEFAULT_REWARD_WEIGHTS = {
    "progress": 2.0, "tracking": 2.0, "speed_error": 1.0, "turning": 0.5, "upright": 2.0,
    "heading": 1.0, "lateral": 2.0, "effort": 0.02, "action_rate": 0.1,
    "angular_velocity": 0.1, "slip": 0.2, "reached": 20.0, "fallen": 10.0,
}


@dataclass
class WadingConfig:
    scene: str
    deployment: str
    manifest: str
    num_envs: int = 8
    dt: float = 0.002
    execution: str = "graph"
    episode_seconds: float = 35.0
    commands: tuple[float, float, float] = (0.4, 0.0, 0.0)
    finish_x: float | None = None
    finish_speed_tolerance: float = 0.25
    finish_hold_seconds: float = 0.6
    seed: int = 20260915
    camera: dict | None = None
    terrain: dict | None = None
    proprioception: dict = field(default_factory=dict)
    reward_weights: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.terrain is not None and self.camera is None:
            raise ValueError("Measured terrain observations require an onboard depth camera")
        unknown = self.reward_weights.keys() - DEFAULT_REWARD_WEIGHTS.keys()
        if unknown:
            raise ValueError(f"Unknown reward weights: {sorted(unknown)}")
        self.reward_weights = {**DEFAULT_REWARD_WEIGHTS, **self.reward_weights}
        if any(not math.isfinite(value) or value < 0 for value in self.reward_weights.values()):
            raise ValueError("Reward weights must be finite and nonnegative")
        if not math.isfinite(self.finish_speed_tolerance) or self.finish_speed_tolerance <= 0:
            raise ValueError("Finish speed tolerance must be finite and positive")
        if not math.isfinite(self.finish_hold_seconds) or self.finish_hold_seconds <= 0:
            raise ValueError("Finish hold duration must be finite and positive")


class G1WadingTask:
    """The actor receives measurements; course geometry is restricted to reward and critic."""

    def __init__(self, config: WadingConfig):
        self.config = config
        if not math.isfinite(config.episode_seconds) or config.episode_seconds <= 0:
            raise ValueError("Episode duration must be finite and positive")
        manifest = json.loads(Path(config.manifest).read_text(encoding="utf-8"))
        course = manifest["course"]
        self.finish_x = course["finish_x"] if config.finish_x is None else config.finish_x
        self.course_width = float(course["width"])
        self.controller = G1VelocityController(config.scene, config.deployment,
            num_envs=config.num_envs, dt=config.dt, commands=config.commands,
            execution=config.execution, observation_source="sensors",
            proprioception_config=ProprioceptionConfig(**{"seed": config.seed, **config.proprioception}))
        self.device = self.controller.device
        self.num_envs = self.controller.num_envs
        self.dt = self.controller.contract.step_dt
        try:
            self.camera = (G1RgbdHistory(self.controller,
                RgbdConfig(**{"seed": config.seed + 1, **config.camera}),
                TerrainConfig(**config.terrain) if config.terrain is not None else None)
                if config.camera is not None else None)
        except BaseException:
            self.controller.close()
            raise
        self.segments = torch.as_tensor(course["segments"], device=self.device)
        self.terrain_offsets = torch.linspace(-0.2, 1.2, 12, device=self.device)
        self.start = self.controller.base[:, :3].clone()
        self.previous_position = self.start.clone()
        self.previous_action = torch.zeros_like(self.controller.last_action)
        self.episode_return = torch.zeros(self.num_envs, device=self.device)
        self.finish_streak = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        self.finish_hold_steps = math.ceil(config.finish_hold_seconds / self.dt)
        self.episode_limit = math.ceil(config.episode_seconds / self.dt)
        self.closed = False

    def observation(self):
        return {"proprio": self.controller.observation(),
                "tactile": self.controller.proprioception.extra_observation(),
                **(self.camera.observation() if self.camera is not None else {})}

    def policy_observation_contract(self):
        return {"use_camera": self.camera is not None and self.camera.config.image_observation,
                "use_terrain": self.config.terrain is not None,
                "history_length": self.camera.config.history_length if self.camera is not None else 3,
                "proprio_dim": self.controller.contract.observation_dim,
                "tactile_dim": self.controller.proprioception.extra_observation().shape[-1],
                "critic_dim": self.critic_observation().shape[-1]}

    def validate_policy(self, policy):
        for name, value in self.policy_observation_contract().items():
            if getattr(policy.config, name) != value:
                raise ValueError(f"Policy and task observation contracts differ for {name}")

    def critic_observation(self):
        controller = self.controller
        base = controller.base
        x = base[:, 0, None] + self.terrain_offsets
        contains = ((x[:, :, None] >= self.segments[:, 0]) &
                    (x[:, :, None] < self.segments[:, 1]))
        height = torch.where(contains, self.segments[:, 2], -0.12).max(-1).values
        relative_height = height - base[:, 2, None]
        feet = controller.link_pose[:, controller.foot_slots, :3] - base[:, None, :3]
        return torch.cat((base[:, :3] - self.start, controller.velocity[:, 0],
            projected_gravity(base[:, 3:]), relative_height, feet.flatten(1)), dim=-1)

    @torch.no_grad()
    def reset(self, env_ids=None):
        self.controller.reset(env_ids)
        if self.camera is not None:
            self.camera.reset(env_ids)
        selected = slice(None) if env_ids is None else env_ids
        self.previous_position[selected] = self.controller.base[selected, :3]
        self.start[selected] = self.controller.base[selected, :3]
        self.previous_action[selected] = 0
        self.episode_return[selected] = 0
        self.finish_streak[selected] = 0
        return self.observation()

    @torch.no_grad()
    def step(self, action):
        controller = self.controller
        controller.step(action)
        if self.camera is not None:
            self.camera.advance()
        base, velocity = controller.base, controller.velocity[:, 0]
        if (not bool(torch.isfinite(base).all() & torch.isfinite(controller.qd).all())
                or bool((controller.status != 0).any())):
            raise RuntimeError("The physical pipeline reported invalid state during the G1 rollout")
        up = -projected_gravity(base[:, 3:])[:, 2]
        fallen, off_course, reached = locomotion_outcome(controller, self.start,
            finish_x=self.finish_x, course_width=self.course_width,
            finish_speed_tolerance=self.config.finish_speed_tolerance,
            finish_streak=self.finish_streak, finish_hold_steps=self.finish_hold_steps)
        time_out = controller.episode_steps >= self.episode_limit
        progress_limit = controller.commands[:, :2].norm(dim=-1) * self.dt
        progress = (base[:, 0] - self.previous_position[:, 0]).clamp(min=-progress_limit, max=progress_limit)
        speed_error = (velocity[:, 3:5] - controller.commands[:, :2]).square().sum(-1)
        tracking = torch.exp(-speed_error / 0.25)
        turning = torch.exp(-(velocity[:, 2] - controller.commands[:, 2]).square() / 0.25)
        effort_fraction = controller.effort[:, controller.joint_slots] / controller.drive_limit[:, controller.joint_slots]
        supported = controller.contact_wrench[:, controller.foot_slots, :3].norm(dim=-1) > 20
        foot_velocity = rotate_vector(controller.link_pose[:, controller.foot_slots, 3:],
                                      controller.velocity[:, controller.foot_slots, 3:])
        slip = (foot_velocity[:, :, :2].square().sum(-1) * supported).sum(-1)
        weights = self.config.reward_weights
        heading = 1 - 2 * (base[:, 5].square() + base[:, 6].square())
        terms = {
            "progress": weights["progress"] * progress,
            "tracking": self.dt * (weights["tracking"] * tracking + weights["turning"] * turning),
            "speed_error": -self.dt * weights["speed_error"] * speed_error,
            "upright": -self.dt * weights["upright"] * (1 - up.square()).clamp_min(0),
            "angular_velocity": -self.dt * weights["angular_velocity"] * velocity[:, :2].square().sum(-1),
            "heading": -self.dt * weights["heading"] * (1 - heading).square(),
            "lateral": -self.dt * weights["lateral"] * (base[:, 1] - self.start[:, 1]).square(),
            "effort": -self.dt * weights["effort"] * effort_fraction.square().mean(-1),
            "action_rate": -self.dt * weights["action_rate"] * (action - self.previous_action).square().mean(-1),
            "slip": -self.dt * weights["slip"] * slip,
            "terminal": weights["reached"] * reached.float() - weights["fallen"] * (fallen | off_course).float(),
        }
        reward = sum(terms.values())
        self.episode_return += reward
        self.previous_position.copy_(base[:, :3])
        self.previous_action.copy_(action)
        info = {"reward_terms": terms, "fallen": fallen, "off_course": off_course, "reached": reached,
                "return": self.episode_return.clone(), "steps": controller.episode_steps.clone(),
                "progress": base[:, 0] - self.start[:, 0],
                "finish_hold_seconds": self.finish_streak.clone() * self.dt}
        terminated = fallen | off_course | reached
        return self.observation(), reward, terminated, time_out & ~terminated, info

    @contextmanager
    def checkpoint(self):
        with self.controller.checkpoint() as physical:
            yield {"controller": physical, "camera": self.camera.state_dict() if self.camera is not None else None,
                   **{name: getattr(self, name).clone() for name in
                      ("start", "previous_position", "previous_action", "episode_return", "finish_streak")}}

    def restore_checkpoint(self, state):
        if (state["camera"] is None) != (self.camera is None):
            raise ValueError("Checkpoint and task camera configurations differ")
        self.controller.restore_checkpoint(state["controller"])
        if self.camera is not None:
            self.camera.load_state_dict(state["camera"])
        for name in ("start", "previous_position", "previous_action", "episode_return", "finish_streak"):
            getattr(self, name).copy_(state[name])
        return self.observation()

    def metadata(self):
        return {"config": asdict(self.config), "camera": self.camera.metadata() if self.camera is not None else None,
                "proprioception": self.controller.proprioception.metadata(),
                "actor_inputs": list(self.observation()),
                "critic_inputs": "relative base position, body velocity/gravity, terrain heights, feet"}

    def close(self):
        if self.closed:
            return
        if self.camera is not None:
            self.camera.pending.clear()
            self.camera.latest_capture = None
        self.controller.close()
        self.closed = True
