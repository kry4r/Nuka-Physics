"""G1 velocity policy deployment on the shared rigid and deformable world pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
import yaml

import nuka
from nuka.kinematics import rotate_vector


@dataclass(frozen=True)
class VelocityPolicyContract:
    joint_names: tuple[str, ...]
    step_dt: float
    default_position: np.ndarray
    action_scale: np.ndarray
    action_offset: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    effort_limit: np.ndarray
    observations: dict
    command_bounds: np.ndarray
    action_clip: tuple[float, float] | None
    gains_source: str

    @classmethod
    def load(cls, deployment: str | Path, *, gains_source: str = "deployment"):
        path = Path(deployment)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        actuator = json.loads(path.with_name("actuators.json").read_text(encoding="utf-8"))
        sdk_names = actuator["joint_sdk_names"]
        order = data["joint_ids_map"]
        if sorted(order) != list(range(len(sdk_names))):
            raise ValueError("joint_ids_map must be a permutation of the SDK joints")
        names = tuple(sdk_names[index] for index in order)
        if len(names) != 29 or len(set(names)) != len(names):
            raise ValueError("The velocity policy requires the complete G1 29-joint contract")
        if gains_source == "deployment":
            kp = [data["stiffness"][index] for index in order]
            kd = [data["damping"][index] for index in order]
        elif gains_source == "training":
            kp = [actuator["joints"][name]["stiffness"] for name in names]
            kd = [actuator["joints"][name]["damping"] for name in names]
        else:
            raise ValueError("gains_source must be deployment or training")
        action = data["actions"]["JointPositionAction"]
        if action.get("joint_ids") is not None:
            raise ValueError("An additional action joint selection is unsupported")
        bounds = data["commands"]["base_velocity"]["ranges"]
        arrays = [data["default_joint_pos"], action["scale"], action["offset"], kp, kd,
                  [actuator["joints"][name]["effort_limit_sim"] for name in names]]
        arrays = [np.asarray(value, dtype=np.float32) for value in arrays]
        if any(value.shape != (len(names),) or not np.isfinite(value).all() for value in arrays):
            raise ValueError("Every joint parameter must contain one finite value per policy joint")
        if not (data["step_dt"] > 0 and np.isfinite(data["step_dt"])) or np.any(arrays[-1] <= 0):
            raise ValueError("The policy interval and motor effort limits must be positive")
        return cls(names, float(data["step_dt"]), *arrays, data["observations"],
                   np.asarray([bounds[key] for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")],
                              dtype=np.float32),
                   tuple(action["clip"]) if action.get("clip") is not None else None, gains_source)

    @property
    def observation_dim(self):
        return sum(len(term["scale"]) * int(term["history_length"])
                   for term in self.observations.values())


class G1VelocityActor(nn.Module):
    """Trainable Torch copy of the pinned ONNX actor, with no added normalization."""

    def __init__(self, layers: nn.Sequential):
        super().__init__()
        self.layers = layers

    @classmethod
    def from_onnx(cls, path: str | Path, *, device="cpu"):
        import onnx
        from onnx import numpy_helper

        model = onnx.load(str(path))
        onnx.checker.check_model(model)
        values = {value.name: numpy_helper.to_array(value) for value in model.graph.initializer}
        cursor = model.graph.input[0].name
        layers = []
        for node in model.graph.node:
            if node.input[0] != cursor or len(node.output) != 1:
                raise ValueError("Expected a single sequential actor in the ONNX graph")
            attrs = {entry.name: onnx.helper.get_attribute_value(entry) for entry in node.attribute}
            if node.op_type == "Gemm":
                if (attrs.get("transA", 0) != 0 or attrs.get("alpha", 1.0) != 1.0
                        or attrs.get("beta", 1.0) != 1.0 or len(node.input) != 3):
                    raise ValueError("Unsupported linear transform in the ONNX actor")
                weight = values[node.input[1]]
                if not attrs.get("transB", 0):
                    weight = weight.T
                bias = values[node.input[2]]
                layer = nn.Linear(weight.shape[1], weight.shape[0])
                with torch.no_grad():
                    layer.weight.copy_(torch.from_numpy(np.array(weight, copy=True)))
                    layer.bias.copy_(torch.from_numpy(np.array(bias, copy=True)))
                layers.append(layer)
            elif node.op_type == "Elu":
                layers.append(nn.ELU(alpha=float(attrs.get("alpha", 1.0))))
            else:
                raise ValueError(f"Unsupported actor operator: {node.op_type}")
            cursor = node.output[0]
        if cursor != model.graph.output[0].name:
            raise ValueError("The actor output is disconnected")
        return cls(nn.Sequential(*layers)).to(device=device, dtype=torch.float32)

    def forward(self, observation):
        return self.layers(observation)


class PolicyObservationHistory:
    """Term-major history, oldest sample first, matching Unitree's deployment manager."""

    def __init__(self, specification, envs, device):
        self.specification = specification
        self.values = {}
        self.scales = {}
        for name, config in specification.items():
            scale = torch.as_tensor(config["scale"], device=device, dtype=torch.float32)
            history = int(config["history_length"])
            if history < 1:
                raise ValueError("Observation history length must be positive")
            self.scales[name] = scale
            self.values[name] = torch.zeros((envs, history, len(scale)), device=device)

    def scaled(self, name, value):
        limits = self.specification[name].get("clip")
        if limits is not None:
            value = value.clamp(float(limits[0]), float(limits[1]))
        return value * self.scales[name]

    def reset(self, measurements, env_ids=slice(None)):
        for name, history in self.values.items():
            history[env_ids] = self.scaled(name, measurements[name])[env_ids, None, :]

    def append(self, measurements):
        for name, history in self.values.items():
            history[:, :-1] = history[:, 1:].clone()
            history[:, -1] = self.scaled(name, measurements[name])

    def observation(self):
        return torch.cat([value.flatten(1) for value in self.values.values()], dim=-1)


def projected_gravity(quaternion):
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack((2 * (w*y - x*z), -2 * (w*x + y*z), 2 * (x*x + y*y) - 1), dim=-1)


def locomotion_outcome(controller, start, *, finish_x=None, course_width=None,
                       finish_speed_tolerance=0.25, finish_streak=None, finish_hold_steps=1):
    base = controller.base
    feet = controller.link_pose[:, controller.foot_slots, :3]
    up = -projected_gravity(base[:, 3:])[:, 2]
    fallen = (up < 0.5) | ((base[:, 2] - feet[:, :, 2].mean(-1)) < 0.35)
    off_course = torch.zeros_like(fallen)
    if course_width is not None:
        off_course = (base[:, 1] - start[:, 1]).abs() > course_width / 2
    reached = torch.zeros_like(fallen)
    if finish_x is not None:
        reached = (base[:, 0] >= finish_x) & (feet[:, :, 0].min(-1).values >= finish_x - 0.2)
        speed_error = (controller.velocity[:, 0, 3:5] - controller.commands[:, :2]).norm(dim=-1)
        reached &= (up >= 0.95) & (speed_error <= finish_speed_tolerance) & ~fallen & ~off_course
    if finish_hold_steps < 1 or (finish_hold_steps > 1 and finish_streak is None):
        raise ValueError("Sustained completion requires a positive duration and a per-environment counter")
    if finish_streak is not None:
        finish_streak.copy_(torch.where(reached, finish_streak + 1, 0))
        reached &= finish_streak >= finish_hold_steps
    return fallen, off_course, reached


class G1VelocityController:
    """Finite-effort engine PD and policy history over an unfixed G1 articulation."""

    def __init__(self, scene: str | Path, deployment: str | Path, *, num_envs=1,
                 dt=0.002, ordinal=0, gains_source="deployment", execution="graph",
                 commands=(0.4, 0.0, 0.0), solver_options=None,
                 observation_source="ideal", proprioception_config=None):
        self.contract = VelocityPolicyContract.load(deployment, gains_source=gains_source)
        self.num_envs = int(num_envs)
        self.dt = float(dt)
        self.decimation = round(self.contract.step_dt / self.dt)
        if self.decimation < 1 or abs(self.decimation*self.dt - self.contract.step_dt) > 1e-8:
            raise ValueError("The policy interval must be an integer number of physics steps")
        self.device = torch.device("cuda", ordinal)
        self.stream = torch.cuda.current_stream(self.device)
        self.backend = nuka.Device.create(ordinal, stream_ptr=self.stream.cuda_stream)
        builder = nuka.SceneBuilder.create(str(scene))
        try:
            self.world = builder.build(self.backend, env_count=self.num_envs, dt=self.dt,
                control_mode=nuka.CONTROL_MODE_PD_POSITION, **(solver_options or {}))
        except BaseException:
            self.backend.close()
            raise
        finally:
            builder.destroy()
        names = self.world.dof_names()
        mapping = {}
        for index, name in enumerate(names):
            tail = name.rsplit("/", 1)[-1]
            if tail in mapping:
                raise ValueError(f"Ambiguous joint name: {tail}")
            mapping[tail] = index
        if self.world.action_dim != len(self.contract.joint_names):
            raise ValueError("The scene must contain one complete G1 articulation")
        self.joint_slots = torch.tensor([mapping[name] for name in self.contract.joint_names],
                                       device=self.device, dtype=torch.int64)
        links = self.world.base_link_count
        self.q = self._view(nuka.JOINT_POSITION, self.num_envs, links)
        self.qd = self._view(nuka.JOINT_VELOCITY, self.num_envs, links)
        self.base = self._view(nuka.BASE_POSE, self.num_envs, 7)
        self.velocity = self._view(nuka.LINK_VELOCITY, self.num_envs, links, 6)
        self.link_pose = self._view(nuka.ARTICULATION_LINK_POSE, self.num_envs, links, 7)
        self.drive_target = self._view(nuka.DRIVE_TARGET, self.num_envs, links)
        self.drive_kp = self._view(nuka.DRIVE_STIFFNESS, self.num_envs, links)
        self.drive_kd = self._view(nuka.DRIVE_DAMPING, self.num_envs, links)
        self.drive_limit = self._view(nuka.DRIVE_FORCE_LIMIT, self.num_envs, links)
        self.effort = self._view(nuka.Field.ACTUATOR_EFFORT, self.num_envs, links)
        self.effort_requested = self._view(nuka.Field.ACTUATOR_EFFORT_REQUESTED, self.num_envs, links)
        self.saturated = self._view(nuka.Field.ACTUATOR_SATURATED, self.num_envs, links)
        self.contact_wrench = self._view(nuka.LINK_CONTACT_WRENCH, self.num_envs, links, 6)
        self.foot_slots = torch.tensor([mapping[side + "_ankle_roll_joint"] for side in ("left", "right")],
                                      device=self.device, dtype=torch.int64)
        self.status = self._view(nuka.Field.ENV_STATUS, self.num_envs)
        self.nominal_q = self._tensor(self.contract.default_position)
        self.scale = self._tensor(self.contract.action_scale)
        self.offset = self._tensor(self.contract.action_offset)
        self.last_action = torch.zeros((self.num_envs, len(self.contract.joint_names)), device=self.device)
        self.commands = self._tensor(commands).expand(self.num_envs, 3).clone()
        self.history = PolicyObservationHistory(self.contract.observations, self.num_envs, self.device)
        self.episode_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int64)
        if observation_source not in ("ideal", "sensors"):
            raise ValueError("The observation source must be ideal or sensors")
        self.proprioception = None
        if observation_source == "sensors":
            from nuka.tasks.g1_perception import G1Proprioception
            self.proprioception = G1Proprioception(self, proprioception_config)
        self.reset()
        self.world.set_execution_mode(execution)

    def _tensor(self, values):
        return torch.as_tensor(values, device=self.device, dtype=torch.float32)

    def _view(self, field, *shape):
        return torch.from_dlpack(self.world.buffer_view(field)).reshape(shape)

    def measurements(self):
        if self.proprioception is not None:
            return {**self.proprioception.measurements(), "velocity_commands": self.commands,
                    "last_action": self.last_action}
        return {"base_ang_vel": self.velocity[:, 0, :3],
                "projected_gravity": projected_gravity(self.base[:, 3:7]),
                "velocity_commands": self.commands,
                "joint_pos_rel": self.q[:, self.joint_slots] - self.nominal_q,
                "joint_vel_rel": self.qd[:, self.joint_slots],
                "last_action": self.last_action}

    @torch.no_grad()
    def reset(self, env_ids=None):
        with torch.cuda.stream(self.stream):
            if env_ids is None:
                self.world.reset()
                selected = slice(None)
            else:
                selected = torch.as_tensor(env_ids, device=self.device, dtype=torch.int64)
                self.world.reset_envs(selected.cpu())
            self.world.synchronize()
            self.drive_kp[:, self.joint_slots] = self._tensor(self.contract.stiffness)
            self.drive_kd[:, self.joint_slots] = self._tensor(self.contract.damping)
            self.drive_limit[:, self.joint_slots] = self._tensor(self.contract.effort_limit)
            self.drive_target[selected] = self.q[selected]
            self.last_action[selected] = 0.0
            self.episode_steps[selected] = 0
            if self.proprioception is not None:
                self.proprioception.reset(selected)
            self.history.reset(self.measurements(), selected)
            return self.history.observation()

    @torch.no_grad()
    def step(self, action):
        with torch.cuda.stream(self.stream):
            action = action.detach().to(device=self.device, dtype=torch.float32)
            if action.shape != self.last_action.shape:
                raise ValueError(f"Expected actions shaped {tuple(self.last_action.shape)}")
            if self.contract.action_clip is not None:
                action = action.clamp(*self.contract.action_clip)
            self.last_action.copy_(action)
            self.drive_target[:, self.joint_slots] = self.offset + self.scale * action
            self.world.step_n(self.decimation)
            self.world.synchronize()
            self.episode_steps.add_(1)
            if self.proprioception is not None:
                self.proprioception.advance()
            self.history.append(self.measurements())
            return self.history.observation()

    def observation(self):
        return self.history.observation()

    @contextmanager
    def checkpoint(self):
        with torch.cuda.stream(self.stream), self.world.capture_checkpoint() as physical:
            yield {"physical": physical, "commands": self.commands.clone(),
                   "last_action": self.last_action.clone(), "episode_steps": self.episode_steps.clone(),
                   "history": {name: value.clone() for name, value in self.history.values.items()},
                   "proprioception": None if self.proprioception is None else self.proprioception.state_dict()}

    def restore_checkpoint(self, state):
        with torch.cuda.stream(self.stream):
            self.world.restore_checkpoint(state["physical"])
            self.world.synchronize()
            for name in ("commands", "last_action", "episode_steps"):
                getattr(self, name).copy_(state[name])
            for name, value in state["history"].items():
                self.history.values[name].copy_(value)
            if self.proprioception is not None:
                self.proprioception.load_state_dict(state["proprioception"])
        return self.observation()

    def close(self):
        if self.world is None:
            return
        if self.proprioception is not None:
            self.proprioception.close()
        for name, value in list(vars(self).items()):
            if isinstance(value, torch.Tensor):
                setattr(self, name, None)
        self.world.destroy()
        self.backend.close()
        self.world = None
