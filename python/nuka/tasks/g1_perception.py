"""Mounted RGB-D acquisition, measurement errors and delayed observation history."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import copy
import math

import numpy as np
import torch

import nuka
from nuka.tasks.g1_locomotion import rotate_vector
from nuka.tasks.g1_terrain import DepthTerrainProjector, TerrainConfig


@dataclass
class ProprioceptionConfig:
    acceleration_noise_density: float = 0.005
    gyro_noise_density: float = 0.0001
    position_noise_density: float = 0.000002
    velocity_noise_density: float = 0.0001
    force_noise_density: float = 0.01
    effort_noise_density: float = 0.02
    load_source: str = "foot_wrench"
    attitude_time_constant: float = 0.75
    attitude_acceleration_tolerance: float = 1.0
    seed: int = 20260915


class G1Proprioception:
    """Measured encoders, foot loads and an IMU complementary attitude estimate."""

    def __init__(self, controller, config=None):
        self.controller = controller
        self.config = config or ProprioceptionConfig()
        cfg = self.config
        values = [cfg.acceleration_noise_density, cfg.gyro_noise_density,
                  cfg.position_noise_density, cfg.velocity_noise_density, cfg.force_noise_density,
                  cfg.effort_noise_density]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("Measurement noise densities must be finite and nonnegative")
        if not math.isfinite(cfg.attitude_time_constant) or cfg.attitude_time_constant <= 0:
            raise ValueError("The attitude filter time constant must be finite and positive")
        if not math.isfinite(cfg.attitude_acceleration_tolerance) or cfg.attitude_acceleration_tolerance <= 0:
            raise ValueError("Attitude acceleration tolerance must be finite and positive")
        if cfg.load_source not in ("foot_wrench", "motor_effort"):
            raise ValueError("Load sensing must use installed foot wrench sensors or motor effort feedback")
        world = controller.world
        self.sensor_ids = []

        def attach(kind, mount, index, densities):
            sensor = world.attach_state_sensor(kind, mount=mount, mount_index=index, seed=cfg.seed,
                                               sample_rate_hz=1 / controller.contract.step_dt)
            self.sensor_ids.append(sensor)
            for component, density in enumerate(densities):
                world.set_state_sensor_error(sensor, component, noise_density=density,
                                             seed=cfg.seed + 37 * sensor + component)
            return torch.from_dlpack(world.get_state_sensor_view(sensor))

        self.imu = attach(nuka.StateSensorKind.IMU, nuka.SensorMount.BASE, 0,
                          [cfg.acceleration_noise_density] * 3 + [cfg.gyro_noise_density] * 3)
        self.encoders = [attach(nuka.StateSensorKind.JOINT_STATE, nuka.SensorMount.LINK, slot,
                               [cfg.position_noise_density, cfg.velocity_noise_density])
                         for slot in controller.joint_slots.cpu().tolist()]
        self.feet = [attach(nuka.StateSensorKind.CONTACT_WRENCH, nuka.SensorMount.LINK, slot,
                           [cfg.force_noise_density] * 3 + [cfg.force_noise_density * 0.05] * 3)
                     for slot in controller.foot_slots.cpu().tolist()] if cfg.load_source == "foot_wrench" else []
        leg_joints = [index for index, name in enumerate(controller.contract.joint_names)
                      if any(part in name for part in ("hip_", "knee_", "ankle_"))]
        self.motor_limits = controller._tensor(controller.contract.effort_limit[leg_joints])
        self.motors = [attach(nuka.StateSensorKind.JOINT_EFFORT, nuka.SensorMount.LINK,
                             int(controller.joint_slots[index]), [cfg.effort_noise_density])
                       for index in leg_joints] if cfg.load_source == "motor_effort" else []
        self.gravity = torch.zeros((controller.num_envs, 3), device=controller.device)
        self.joint_state = torch.zeros((controller.num_envs, len(self.encoders), 2), device=controller.device)
        self.specific_force = torch.zeros_like(self.gravity)
        self.angular_velocity = torch.zeros_like(self.gravity)
        self.foot_wrench = torch.zeros((controller.num_envs, 2, 6), device=controller.device)
        self.motor_effort = torch.zeros((controller.num_envs, len(leg_joints)), device=controller.device)
        self.valid = torch.zeros((controller.num_envs, 1), device=controller.device, dtype=torch.bool)
        self.reset()

    @torch.no_grad()
    def reset(self, env_ids=None):
        selected = slice(None) if env_ids is None else env_ids
        self.gravity[selected] = self.gravity.new_tensor([0, 0, -1])
        self.joint_state[selected, :, 0] = self.controller.nominal_q
        self.joint_state[selected, :, 1] = 0
        for name in ("specific_force", "angular_velocity", "foot_wrench", "motor_effort", "valid"):
            getattr(self, name)[selected] = 0

    @torch.no_grad()
    def advance(self):
        self.specific_force.copy_(self.imu[:, :3])
        self.angular_velocity.copy_(self.imu[:, 3:])
        self.joint_state.copy_(torch.stack(self.encoders, dim=1))
        if self.feet:
            self.foot_wrench.copy_(torch.stack(self.feet, dim=1))
        if self.motors:
            self.motor_effort.copy_(torch.cat(self.motors, dim=-1))
        self.valid.copy_((self.controller.episode_steps > 0)[:, None])
        dt = self.controller.contract.step_dt
        half_angle = 0.5 * dt * self.angular_velocity.norm(dim=-1, keepdim=True)
        increment = torch.cat((half_angle.cos(),
            -0.5 * dt * torch.sinc(half_angle / math.pi) * self.angular_velocity), dim=-1)
        predicted = rotate_vector(increment, self.gravity)
        magnitude = self.specific_force.norm(dim=-1, keepdim=True)
        measured = -self.specific_force / magnitude.clamp_min(1e-6)
        confidence = (1 - (magnitude - 9.81).abs() / (0.25 * 9.81)).clamp(0, 1)
        acceleration = self.specific_force + 9.81 * predicted
        confidence *= torch.exp(-0.5 * acceleration.square().sum(-1, keepdim=True)
                                / self.config.attitude_acceleration_tolerance ** 2)
        support = ((self.foot_wrench[:, :, :3].norm(dim=-1).sum(-1, keepdim=True) > 20)
                   if self.feet else magnitude > 0.5 * 9.81)
        blend = (1 - math.exp(-dt / self.config.attitude_time_constant)) * confidence * support
        estimated = predicted + blend * (measured - predicted)
        self.gravity.copy_(estimated / estimated.norm(dim=-1, keepdim=True).clamp_min(1e-6))

    def measurements(self):
        return {"base_ang_vel": self.angular_velocity, "projected_gravity": self.gravity,
                "joint_pos_rel": self.joint_state[:, :, 0] - self.controller.nominal_q,
                "joint_vel_rel": self.joint_state[:, :, 1]}

    def extra_observation(self):
        scale = self.foot_wrench.new_tensor([300, 300, 300, 20, 20, 20])
        loads = self.motor_effort / self.motor_limits if self.motors else (self.foot_wrench / scale).flatten(1)
        return torch.cat((self.specific_force / 9.81, loads,
                          self.valid.to(torch.float32)), dim=-1)

    def state_dict(self):
        return {name: getattr(self, name).clone() for name in
                ("gravity", "joint_state", "specific_force", "angular_velocity", "foot_wrench", "motor_effort", "valid")}

    def load_state_dict(self, state):
        for name, value in state.items():
            getattr(self, name).copy_(value)

    def metadata(self):
        return {**asdict(self.config), "sensor_ids": self.sensor_ids,
                "sample_rate_hz": 1 / self.controller.contract.step_dt,
                "attitude": "rotation-exponential gyro integration with acceleration-gated correction",
                "load_contract": ("measured leg motor effort after actuator limits; no external contact oracle"
                    if self.motors else "requires instrumented six-axis foot sensors; not assumed on stock G1"),
                "reset_prior": "upright and commanded home joints until the first sensor sample",
                "calibration": "illustrative SI noise densities; no world-pose input"}

    def close(self):
        self.imu = None
        self.encoders.clear()
        self.feet.clear()
        self.motors.clear()
        self.controller = None


@dataclass
class RgbdConfig:
    camera_name: str = "policy_rgbd"
    width: int = 192
    height: int = 144
    vfov: float | None = None
    near_clip: float | None = None
    far_clip: float | None = None
    update_period: int = 5
    history_length: int = 3
    image_observation: bool = True
    latency: float = 0.04
    latency_jitter: float = 0.0
    dropout_probability: float = 0.0
    seed: int = 20260914
    fidelity: dict = field(default_factory=lambda: {
        "spp": 4, "shadow_samples": 4, "ao_samples": 2, "gi_enabled": False})
    color_response: dict = field(default_factory=lambda: {
        "read_noise_electrons": 2.0, "pixel_gain_stddev": 0.005, "adc_bits": 12})
    depth_response: dict = field(default_factory=lambda: {
        "distance_stddev": 0.001, "quadratic_stddev": 0.001,
        "quantization": 0.001, "return_photons": 150.0, "dropout_probability": 0.01})

    def validate(self):
        if min(self.width, self.height, self.update_period, self.history_length) < 1:
            raise ValueError("Image dimensions, acquisition period and history must be positive")
        if not 0 < self.vfov < 180 or not 0 < self.near_clip < self.far_clip:
            raise ValueError("Camera field of view and clip interval must be ordered and positive")
        if (not math.isfinite(self.latency) or not math.isfinite(self.latency_jitter)
                or min(self.latency, self.latency_jitter) < 0 or not 0 <= self.dropout_probability <= 1):
            raise ValueError("Camera delay and dropout parameters are invalid")
        if not self.camera_name:
            raise ValueError("The observation camera requires an authored scene camera name")


class G1RgbdHistory:
    """Keep first-surface depth, missing returns and acquisition ages separate."""

    def __init__(self, controller, config: RgbdConfig | None = None, terrain: TerrainConfig | None = None):
        self.controller = controller
        world = controller.world
        requested = config or RgbdConfig()
        self.asset_camera = world.scene_camera(requested.camera_name)
        self.config = replace(requested, **{name: self.asset_camera[name]
            for name in ("vfov", "near_clip", "far_clip") if getattr(requested, name) is None})
        cfg = self.config
        cfg.validate()
        if terrain is None and not cfg.image_observation:
            raise ValueError("A depth-only camera needs a terrain observation consumer")
        self.projector = DepthTerrainProjector(controller, self.asset_camera, cfg, terrain) if terrain else None
        if world.sensor_width or world.sensor_height:
            raise ValueError("The observation history requires a world without an existing camera")
        with torch.cuda.stream(controller.stream):
            world.attach_camera_sensor(self.asset_camera["mount"], self.asset_camera["mount_index"],
                                       self.asset_camera["local_offset"],
                                       cfg.vfov, cfg.width, cfg.height)
            world.set_camera_intrinsics(near_clip=cfg.near_clip, far_clip=cfg.far_clip)
            mask = nuka.SensorAov.DEPTH.value | (nuka.SensorAov.COLOR.value if cfg.image_observation else 0)
            world.set_sensor_aov_mask(mask)
            world.set_sensor_fidelity(**cfg.fidelity)
            if cfg.image_observation:
                nuka.CameraResponse(seed=cfg.seed, **cfg.color_response).configure(world)
            nuka.RangeResponse(seed=cfg.seed + 1, **cfg.depth_response).configure(world, nuka.SensorChannel.DEPTH)
        self.mount_index = self.asset_camera["mount_index"]
        self.rng = np.random.default_rng(cfg.seed)
        shape = (controller.num_envs, cfg.history_length)
        self.rgb = (torch.zeros((*shape, 3, cfg.height, cfg.width), device=controller.device, dtype=torch.float16)
                    if cfg.image_observation else None)
        self.depth = torch.zeros((*shape, 1, cfg.height, cfg.width), device=controller.device, dtype=torch.float16)
        self.depth_valid = torch.zeros_like(self.depth, dtype=torch.bool)
        self.frame_valid = torch.zeros((*shape, 1), device=controller.device, dtype=torch.bool)
        self.sample_time = torch.zeros((*shape, 1), device=controller.device, dtype=torch.float64)
        self.sequence = torch.zeros((*shape, 1), device=controller.device, dtype=torch.int64)
        self.terrain = (torch.zeros((*shape, len(self.projector.channels), *self.projector.shape),
            device=controller.device, dtype=torch.float16) if self.projector is not None else None)
        self.history_fields = ["depth", "depth_valid", "frame_valid", "sample_time", "sequence"]
        self.history_fields += [name for name in ("rgb", "terrain") if getattr(self, name) is not None]
        self.pending = []
        self.last_available_time = np.zeros(controller.num_envs)
        self.tick = 0
        self.latest_capture = None
        self.reset()

    def _time(self):
        return self.controller.episode_steps.cpu().numpy() * self.controller.contract.step_dt

    @torch.no_grad()
    def _acquire(self):
        world, cfg = self.controller.world, self.config
        world.render_sensors()
        world.synchronize()
        rgb = (torch.from_dlpack(world.get_sensor_view(nuka.SensorChannel.COLOR)).permute(0, 3, 1, 2)
               if cfg.image_observation else None)
        depth = torch.from_dlpack(world.get_sensor_view(nuka.SensorChannel.DEPTH)).permute(0, 3, 1, 2)
        valid = torch.isfinite(depth) & (depth >= cfg.near_clip) & (depth <= cfg.far_clip)
        normalized = torch.where(valid, depth / cfg.far_clip, 0).clamp(0, 1)
        stamps = [world.imaging_stamp(nuka.SensorChannel.DEPTH.value, 0, env)
                  for env in range(self.controller.num_envs)]
        times = np.asarray([stamp["sample_time"] for stamp in stamps])
        delays = np.maximum(0, cfg.latency + self.rng.uniform(-cfg.latency_jitter, cfg.latency_jitter, len(times)))
        active = np.asarray([stamp["valid"] for stamp in stamps], dtype=bool)
        active &= self.rng.random(len(times)) >= cfg.dropout_probability
        available = np.maximum(times + delays, self.last_available_time)
        self.last_available_time = available
        packet = {"depth": normalized.to(torch.float16),
                  "metric_depth": depth.clone(),
                  "depth_valid": valid.clone(), "sample_time": times,
                  "sequence": np.asarray([stamp["acquisitions"] for stamp in stamps]),
                  "available_time": available, "active": active}
        if rgb is not None:
            packet["rgb"] = rgb.to(torch.float16).clone()
        if self.projector is not None:
            grid, pose, up = self.projector.project(depth, valid)
            packet.update(terrain=grid.to(torch.float16), camera_pose_in_base=pose, estimated_up=up)
        self.pending.append(packet)
        self.latest_capture = packet

    @torch.no_grad()
    def _deliver(self):
        now = self._time()
        for packet in self.pending:
            ready = packet["active"] & (packet["available_time"] <= now + 1e-7)
            if not ready.any():
                continue
            ids = torch.as_tensor(np.flatnonzero(ready), device=self.controller.device)
            for name in self.history_fields:
                history = getattr(self, name)
                history[ids, :-1] = history[ids, 1:].clone()
            self.depth[ids, -1] = packet["depth"][ids]
            self.depth_valid[ids, -1] = packet["depth_valid"][ids]
            for name in ("rgb", "terrain"):
                if getattr(self, name) is not None:
                    getattr(self, name)[ids, -1] = packet[name][ids]
            self.frame_valid[ids, -1] = True
            self.sample_time[ids, -1, 0] = torch.as_tensor(packet["sample_time"][ready], device=ids.device)
            self.sequence[ids, -1, 0] = torch.as_tensor(packet["sequence"][ready], device=ids.device)
            packet["active"][ready] = False
        self.pending = [packet for packet in self.pending if packet["active"].any()]

    @torch.no_grad()
    def reset(self, env_ids=None):
        with torch.cuda.stream(self.controller.stream):
            selected = slice(None) if env_ids is None else torch.as_tensor(env_ids, device=self.controller.device)
            for name in self.history_fields:
                getattr(self, name)[selected] = 0
            if env_ids is None:
                self.pending.clear()
                self.tick = 0
                self.rng = np.random.default_rng(self.config.seed)
                self.last_available_time.fill(0)
                self._acquire()
                self._deliver()
            else:
                selected_cpu = selected.cpu().numpy()
                self.last_available_time[selected_cpu] = 0
                for packet in self.pending:
                    packet["active"][selected_cpu] = False

    @torch.no_grad()
    def advance(self):
        with torch.cuda.stream(self.controller.stream):
            self.tick += 1
            if self.tick % self.config.update_period == 0:
                self._acquire()
            self._deliver()
        return self.observation()

    def observation(self):
        now = self.controller.episode_steps.to(torch.float64)[:, None, None] * self.controller.contract.step_dt
        age = (now - self.sample_time).clamp_min(0).to(torch.float32)
        result = {"frame_valid": self.frame_valid, "age": age}
        if self.config.image_observation:
            result.update(rgb=self.rgb, depth=self.depth, depth_valid=self.depth_valid)
        if self.terrain is not None:
            result["terrain"] = self.terrain
        return result

    def state_dict(self):
        state = {name: getattr(self, name).clone() for name in self.history_fields}
        state.update(pending=copy.deepcopy(self.pending), tick=self.tick,
                     last_available_time=self.last_available_time.copy(),
                     rng=copy.deepcopy(self.rng.bit_generator.state),
                     latest_capture=copy.deepcopy(self.latest_capture))
        return state

    def load_state_dict(self, state):
        for name in self.history_fields:
            getattr(self, name).copy_(state[name])
        self.pending = copy.deepcopy(state["pending"])
        self.tick = state["tick"]
        self.last_available_time = state["last_available_time"].copy()
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.latest_capture = copy.deepcopy(state["latest_capture"])

    def metadata(self):
        return {**asdict(self.config), "mount_index": self.mount_index, "scene_camera": self.asset_camera,
                "terrain": self.projector.metadata() if self.projector is not None else None,
                "depth_contract": "first-surface metric range divided by far_clip; invalid pixels carry a separate mask",
                "noise_calibration": "illustrative parameters, not a calibrated hardware model"}
