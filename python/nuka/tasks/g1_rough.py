"""G1 humanoid velocity tracking on a bump heightfield (rl_games backend).

The robot's revolute joints are found by name, so any G1 scene works; the 23-joint G1 (31 links) stays on the fast
articulation path, while the 29-joint one (39 links) does not. Observations follow unitree_rl_lab's velocity task:
base angular velocity, projected gravity, the command, joint positions and velocities and the last action, five
frames of history, term by term and oldest first. As there, the critic sees `states`: the base's linear velocity
followed by the same terms without noise, five frames of history. Rewards and terminations follow its
velocity_env_cfg (Apache-2.0). Foot contact is the sole spheres' distance to the heightfield, and other links
touching down are found the same way from their origins.

Training runs on a copy of the scene with its mesh collision shapes dropped: the sole spheres carry the contacts,
and dense meshes on thousands of robots overflow the contact budget. The copy also drops the scene's terrain, since
the task bakes the field its `terrain` option describes.
"""

from __future__ import annotations

import json
import math
import re
import tempfile
from pathlib import Path

import numpy as np
import torch
from gymnasium import spaces

import nuka
from .bdx_obs import quat_rotate_inverse_wxyz, quat_rotate_wxyz

# The default stance by joint name; joints not listed stand at zero.
DEFAULT_POSE = [
    (r".*_hip_pitch_joint", -0.1), (r".*_knee_joint", 0.3), (r".*_ankle_pitch_joint", -0.2),
    (r".*_shoulder_pitch_joint", 0.3), (r"left_shoulder_roll_joint", 0.25), (r"right_shoulder_roll_joint", -0.25),
    (r".*_elbow_joint", 0.97), (r"left_wrist_roll_joint", 0.15), (r"right_wrist_roll_joint", -0.15),
]
# (joint pattern, stiffness N m/rad, damping N m s/rad, effort limit N m) of unitree_rl_lab's G1; first match wins.
ACTUATORS = [
    (r".*_hip_(pitch|yaw)_joint", 100.0, 2.0, 88.0), (r".*_hip_roll_joint", 100.0, 2.0, 139.0),
    (r".*_knee_joint", 150.0, 4.0, 139.0), (r".*_ankle_.*_joint", 40.0, 2.0, 25.0),
    (r"waist_yaw_joint", 200.0, 5.0, 88.0), (r"waist_.*_joint", 40.0, 5.0, 25.0),
    (r".*_wrist_(pitch|yaw)_joint", 40.0, 1.0, 5.0), (r".*", 40.0, 1.0, 25.0),
]
ARMATURE = 0.01
FOOT_LINK = r".*_ankle_roll_link"
# Links whose origin near the ground counts as an unwanted contact (unitree_rl_lab penalizes every non-ankle link).
TOUCH_LINKS = r".*_(knee|elbow|wrist_roll)_link"
ARM_JOINTS = r".*_(shoulder_.*|elbow|wrist_.*)_joint"
WAIST_JOINTS = r"waist_.*_joint"
LEG_DEVIATION_JOINTS = r".*_hip_(roll|yaw)_joint"

ACTION_SCALE = 0.25
HISTORY = 5
OBS_SCALES = {"base_ang_vel": 0.2, "projected_gravity": 1.0, "velocity_commands": 1.0, "joint_pos_rel": 1.0,
              "joint_vel_rel": 0.05, "last_action": 1.0}
# Uniform noise added before scaling, as in the actor's observation group.
OBS_NOISE = {"base_ang_vel": 0.2, "projected_gravity": 0.05, "joint_pos_rel": 0.01, "joint_vel_rel": 1.5}

REWARD_SCALES = {
    "track_lin_vel_xy": 1.0, "track_ang_vel_z": 0.5, "alive": 0.15, "lin_vel_z": -2.0, "ang_vel_xy": -0.05,
    "joint_vel": -0.001, "joint_acc": -2.5e-7, "action_rate": -0.05, "dof_pos_limits": -5.0, "energy": -2e-5,
    "joint_deviation_arms": -0.1, "joint_deviation_waist": -1.0, "joint_deviation_legs": -1.0,
    "flat_orientation": -5.0, "base_height": -10.0, "gait": 0.5, "feet_slide": -0.2, "feet_clearance": 1.0,
    "undesired_contacts": -1.0,
}
BASE_HEIGHT_TARGET = 0.78
GAIT_PERIOD, GAIT_OFFSETS, GAIT_STANCE = 0.8, (0.0, 0.5), 0.55
CLEARANCE_TARGET, CLEARANCE_STD = 0.1, 0.05
CONTACT_MARGIN = 0.02
TOUCH_MARGIN = 0.08

# Command ranges grow from +-START toward LIMIT while the robots track them (unitree_rl_lab's level curriculum).
COMMAND_LIMIT = {"lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.3, 0.3), "ang_vel_yaw": (-0.5, 0.5)}
COMMAND_START = 0.1
STANDING_ENVS = 0.02


def _match(table, name, default):
    for pattern, *values in table:
        if re.fullmatch(pattern, name):
            return values[0] if len(values) == 1 else values
    return default


def training_scene(scene: str) -> tuple[Path, dict]:
    """The scene flattened into a temporary .nks without mesh collision shapes, and its parsed tree."""
    folder = Path(tempfile.mkdtemp(prefix="nuka_g1_"))
    out = folder / "scene.nks"
    loaded = nuka.Scene.load(scene)
    loaded.save(str(out))
    loaded.destroy()
    doc = json.loads(out.read_text(encoding="utf-8"))

    def strip(nodes):
        for n in nodes:
            if n.get("collision_shape", {}).get("type") == "trimesh":
                del n["collision_shape"]
            strip(n.get("children", []))

    strip(doc["tree"])
    # The task bakes its own field; the scene's terrain would lie under it as a second one.
    doc.pop("terrain", None)
    out.write_text(json.dumps(doc), encoding="utf-8")
    return out, doc


def _nodes(nodes):
    for n in nodes:
        yield n
        yield from _nodes(n.get("children", []))


class TerrainSampler:
    """Bilinear heights of the world's cooked heightfield, on the GPU."""

    def __init__(self, world, device):
        placement, values = world.terrain()
        self.ox, self.oy = placement["origin"][0], placement["origin"][1]
        self.cx, self.cy = placement["cell_x"], placement["cell_y"]
        self.heights = (placement["base_z"] + torch.as_tensor(values, device=device) * placement["scale_z"])
        self.nrow, self.ncol = self.heights.shape
        self.extent = ((self.ncol - 1) * self.cx, (self.nrow - 1) * self.cy)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        u = ((x - self.ox) / self.cx).clamp(0.0, self.ncol - 1.0001)
        v = ((y - self.oy) / self.cy).clamp(0.0, self.nrow - 1.0001)
        c, r = u.floor().long(), v.floor().long()
        fu, fv = u - c, v - r
        h = self.heights
        top = h[r, c] * (1 - fu) + h[r, c + 1] * fu
        bottom = h[r + 1, c] * (1 - fu) + h[r + 1, c + 1] * fu
        return top * (1 - fv) + bottom * fv


class G1RoughEnv:
    """Vectorized G1 velocity-tracking env (gymnasium 5-tuple step, all tensors on cuda)."""

    metadata = {"render_modes": []}

    # 2 ms physics steps: at 5 ms Unitree's pretrained G1 gait covers a third of its distance in Nuka.
    def __init__(self, scene: str, num_envs: int, *, device=None, dt: float = 0.002, decimation: int = 10,
                 terrain: dict | None = None, command_ranges: dict | None = None, reward_scales: dict | None = None,
                 push: bool = True, noise: bool = True, termination_height: float = 0.3,
                 termination_tilt_deg: float = 45.0, episode_length_s: float = 20.0, env_spacing: float = 2.5,
                 fixed_command: bool = False, command=(0.5, 0.0, 0.0), curriculum: bool = True,
                 seed: int | None = None, **_ignored):
        self.num_envs = int(num_envs)
        self.dt, self.decimation = float(dt), int(decimation)
        self.step_dt = self.dt * self.decimation
        self.max_episode_length = max(1, int(round(episode_length_s / self.step_dt)))
        self.termination_height = float(termination_height)
        self.termination_cos = math.cos(math.radians(termination_tilt_deg))
        self._dev = torch.device("cuda")
        self._gen = torch.Generator(device=self._dev)
        self._gen.manual_seed(0 if seed is None else int(seed))
        self.push, self.noise = bool(push), bool(noise)
        self.scales = dict(REWARD_SCALES)
        self.scales.update(reward_scales or {})
        self.scene = scene

        path, doc = training_scene(scene)
        nodes = list(_nodes(doc["tree"]))
        joints = {n["joint"]["name"]: n["joint"] for n in nodes if n.get("joint", {}).get("type") == "revolute"}
        feet = [n for n in nodes if re.fullmatch(FOOT_LINK, n.get("name", ""))]
        if len(feet) != 2:
            raise RuntimeError(f"g1_rough: the scene needs two links matching {FOOT_LINK}, found {len(feet)}")
        self._sole = [[(c["collision_shape"]["local"]["pos"], c["collision_shape"]["radius"])
                       for c in f.get("children", []) if c.get("collision_shape", {}).get("type") == "sphere"]
                      for f in feet]
        if not all(self._sole):
            raise RuntimeError("g1_rough: each foot link needs sole collision spheres")

        # The cooked heightfield: flat, hashed bumps, or the tiled field of stairs, pits and bumps.
        t = {"kind": "bumps", "bump_height": 0.05, "bump_cell": 0.4, "step_height": 0.08, "cell": 0.1}
        t.update(terrain or {})
        origins = min(self.num_envs, 1024)
        side = int(math.ceil(math.sqrt(origins)))
        size = side * float(env_spacing) + 4.0
        n = int(round(size / t["cell"])) + 1
        kind = {"flat": 0, "bumps": 3, "mixed": 4}[t["kind"]]
        self._device = device if device is not None else nuka.Device.create(0)
        self._owns_device = device is None
        self.world = nuka.World.create_from_scene(
            self._device, str(path), self.num_envs, self.dt, contact_family=1, heightfield_terrain_type=kind,
            heightfield_nrow=n, heightfield_ncol=n, heightfield_cell=float(t["cell"]),
            terrain_grid_height_max=float(t["bump_height"]), terrain_grid_width=float(t["bump_cell"]),
            terrain_step_height=float(t["step_height"]), terrain_step_width=0.3, terrain_platform_width=1.0)
        self.terrain = TerrainSampler(self.world, self._dev)
        grid = torch.arange(origins, device=self._dev)
        self._origins = torch.stack(((grid % side).float(), (grid // side).float()), 1) * env_spacing
        self._origins -= self._origins.mean(0)
        self._origins = self._origins[torch.arange(self.num_envs, device=self._dev) % origins]

        w = self.world
        names = sorted(joints, key=lambda j: int(w.dof_index("^" + j + "$")[0]))
        self.joint_names = names
        self.slots = torch.as_tensor([int(w.dof_index("^" + j + "$")[0]) for j in names], device=self._dev)
        self.num_joints = len(names)
        self.default = torch.as_tensor([_match(DEFAULT_POSE, j, 0.0) for j in names], device=self._dev)
        gains = [_match(ACTUATORS, j, None) for j in names]
        self.kp = torch.as_tensor([g[0] for g in gains], device=self._dev)
        self.kd = torch.as_tensor([g[1] for g in gains], device=self._dev)
        self.effort = torch.as_tensor([g[2] for g in gains], device=self._dev)
        lo = torch.as_tensor([joints[j]["lower_limit"] for j in names], device=self._dev)
        hi = torch.as_tensor([joints[j]["upper_limit"] for j in names], device=self._dev)
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * 0.9
        self.soft_lo, self.soft_hi = mid - half, mid + half

        def mask(pattern):
            return torch.as_tensor([bool(re.fullmatch(pattern, j)) for j in names], device=self._dev)

        self._arms, self._waist, self._legs = mask(ARM_JOINTS), mask(WAIST_JOINTS), mask(LEG_DEVIATION_JOINTS)
        link_slot = {n["name"]: int(w.dof_index("^" + n["joint"]["name"] + "$")[0])
                     for n in nodes if n.get("joint", {}).get("type") == "revolute"}
        self._feet = torch.as_tensor([link_slot[f["name"]] for f in feet], device=self._dev)
        self._touch = torch.as_tensor([s for k, s in link_slot.items() if re.fullmatch(TOUCH_LINKS, k)],
                                      device=self._dev)
        self._sole_local = [torch.as_tensor([p for p, _ in s], device=self._dev) for s in self._sole]
        self._sole_radius = [torch.as_tensor([r for _, r in s], device=self._dev) for s in self._sole]

        self._q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
        self._qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
        self._tgt = torch.from_dlpack(w.buffer_view(nuka.DRIVE_TARGET))
        self._pose = torch.from_dlpack(w.buffer_view(nuka.ARTICULATION_LINK_POSE))
        self._vel = torch.from_dlpack(w.buffer_view(nuka.LINK_VELOCITY))
        self._base = torch.from_dlpack(w.buffer_view(nuka.BASE_POSE)).reshape(self.num_envs, -1)
        kp = torch.from_dlpack(w.buffer_view(nuka.DRIVE_STIFFNESS))
        kd = torch.from_dlpack(w.buffer_view(nuka.DRIVE_DAMPING))
        fl = torch.from_dlpack(w.buffer_view(nuka.DRIVE_FORCE_LIMIT))
        kp[:, self.slots], kd[:, self.slots], fl[:, self.slots] = self.kp, self.kd, self.effort
        armature = next(i for i, p in enumerate(w.param_infos()) if p["key"] == "joint_armature")
        for s in self.slots.tolist():
            w.set_param(armature, s, ARMATURE)

        # Command ranges and the per-env command; the curriculum widens the ranges toward their limits, and
        # without it they start there.
        self._limit = {k: tuple(v) for k, v in {**COMMAND_LIMIT, **(command_ranges or {})}.items()}
        start = COMMAND_START if curriculum else math.inf
        self._ranges = {k: (max(lo, -start), min(hi, start)) for k, (lo, hi) in self._limit.items()}
        self._fixed_command = bool(fixed_command)
        self.command = torch.tensor(command, dtype=torch.float32, device=self._dev).repeat(self.num_envs, 1)
        self._track_sum = torch.zeros(self.num_envs, 2, device=self._dev)
        self._tracked = torch.zeros(3, device=self._dev)
        self._common_step = 0

        J = self.num_joints
        self._terms = [("base_ang_vel", 3), ("projected_gravity", 3), ("velocity_commands", 3),
                       ("joint_pos_rel", J), ("joint_vel_rel", J), ("last_action", J)]
        self._frame = sum(d for _, d in self._terms)
        self.obs_dim = self._frame * HISTORY
        self.state_dim = (self._frame + 3) * HISTORY
        self.action_dim = J
        self._history = torch.zeros(self.num_envs, HISTORY, self._frame, device=self._dev)
        self._state_history = torch.zeros(self.num_envs, HISTORY, self._frame + 3, device=self._dev)
        self.states = torch.zeros(self.num_envs, self.state_dim, device=self._dev)
        self.action = torch.zeros(self.num_envs, J, device=self._dev)
        self._last_action = torch.zeros_like(self.action)
        self._last_qd = torch.zeros(self.num_envs, J, device=self._dev)
        self._last_feet = torch.zeros(self.num_envs, 2, 3, device=self._dev)
        self._fresh = torch.ones(self.num_envs, dtype=torch.bool, device=self._dev)
        self.episode_step = torch.zeros(self.num_envs, dtype=torch.long, device=self._dev)
        self._command_step = torch.zeros_like(self.episode_step)
        self._push_step = torch.zeros_like(self.episode_step)

        high = np.full(self.obs_dim, np.inf, np.float32)
        self.single_observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.single_action_space = spaces.Box(-np.inf, np.inf, (J,), dtype=np.float32)
        self.single_state_space = spaces.Box(-np.inf, np.inf, (self.state_dim,), dtype=np.float32)
        self.observation_space = self.single_observation_space
        self.action_space = self.single_action_space
        print(f"[g1_rough] envs={self.num_envs} joints={J} obs={self.obs_dim} terrain={t} "
              f"field={n}x{n} feet={self._feet.tolist()} touch={self._touch.tolist()}", flush=True)

    # -- state ---------------------------------------------------------------
    def _base_quat(self):
        return self._base[:, 3:7]

    def _gravity(self):
        g = torch.tensor([0.0, 0.0, -1.0], device=self._dev).expand(self.num_envs, 3)
        return quat_rotate_inverse_wxyz(self._base_quat(), g)

    def _soles(self):
        """(N, 2) lowest sole-sphere height above the terrain under it, and (N, 2, 3) foot link positions."""
        pose = self._pose[:, self._feet]
        heights = []
        for k in range(2):
            pos, quat = pose[:, k, 0:3], pose[:, k, 3:7]
            local = self._sole_local[k]
            centers = pos[:, None] + quat_rotate_wxyz(quat[:, None].expand(-1, len(local), 4),
                                                      local[None].expand(self.num_envs, -1, 3))
            ground = self.terrain(centers[..., 0], centers[..., 1])
            heights.append((centers[..., 2] - self._sole_radius[k] - ground).min(1).values)
        return torch.stack(heights, 1), pose[..., 0:3]

    def contract(self) -> dict:
        """What the editor's policy runner needs to run this env's actor."""
        return {"version": 1, "joints": self.joint_names, "default": self.default.tolist(),
                "stiffness": self.kp.tolist(), "damping": self.kd.tolist(), "effort": self.effort.tolist(),
                "armature": ARMATURE, "action_scale": ACTION_SCALE, "dt": self.dt, "decimation": self.decimation,
                "history": HISTORY, "terms": [{"name": k, "dims": d, "scale": OBS_SCALES[k]} for k, d in self._terms],
                "command_limit": {k: list(v) for k, v in self._limit.items()}}

    # -- observation ---------------------------------------------------------
    def _frame_now(self):
        """The actor's frame and the critic's: the base's linear velocity, then the terms without noise."""
        q = self._q[:, self.slots]
        qd = self._qd[:, self.slots]
        values = {"base_ang_vel": self._vel[:, 0, 0:3], "projected_gravity": self._gravity(),
                  "velocity_commands": self.command, "joint_pos_rel": q - self.default, "joint_vel_rel": qd,
                  "last_action": self.action}
        parts, clean = [], [self._vel[:, 0, 3:6]]
        for k, _ in self._terms:
            v = values[k]
            clean.append(v * OBS_SCALES[k])
            if self.noise and k in OBS_NOISE:
                v = v + (torch.rand(v.shape, generator=self._gen, device=self._dev) * 2 - 1) * OBS_NOISE[k]
            parts.append(v * OBS_SCALES[k])
        return torch.cat(parts, 1), torch.cat(clean, 1)

    def _observe(self, fresh=None):
        frame, state = self._frame_now()
        self._history = torch.roll(self._history, -1, 1)
        self._history[:, -1] = frame
        self._state_history = torch.roll(self._state_history, -1, 1)
        self._state_history[:, -1] = state
        if fresh is not None:
            self._history[fresh] = frame[fresh, None].expand(-1, HISTORY, -1)
            self._state_history[fresh] = state[fresh, None].expand(-1, HISTORY, -1)
        self.states = self._state_history.reshape(self.num_envs, -1).nan_to_num(0.0, 100.0, -100.0)
        out, start = [], 0
        for _, d in self._terms:
            out.append(self._history[:, :, start:start + d].reshape(self.num_envs, -1))
            start += d
        return torch.cat(out, 1).nan_to_num_(0.0, 100.0, -100.0)

    # -- reset ---------------------------------------------------------------
    def _place(self, ids):
        k = len(ids)
        if k == 0:
            return
        xy = self._origins[ids] + (torch.rand(k, 2, generator=self._gen, device=self._dev) * 2 - 1) * 0.5
        yaw = (torch.rand(k, generator=self._gen, device=self._dev) * 2 - 1) * math.pi
        ground = self.terrain(xy[:, 0], xy[:, 1])
        self._base[ids, 0:2] = xy
        self._base[ids, 2] = ground + 0.8
        self._base[ids, 3] = torch.cos(yaw / 2)
        self._base[ids, 4:6] = 0.0
        self._base[ids, 6] = torch.sin(yaw / 2)
        rows = ids[:, None]
        self._q[rows, self.slots[None]] = self.default
        self._qd[rows, self.slots[None]] = 0.0
        self._tgt[rows, self.slots[None]] = self.default
        self._vel[ids] = 0.0
        nuka.sync()

    def _resample(self, ids):
        k = len(ids)
        if k == 0 or self._fixed_command:
            return
        draw = [(torch.rand(k, generator=self._gen, device=self._dev) * (hi - lo) + lo)
                for lo, hi in (self._ranges["lin_vel_x"], self._ranges["lin_vel_y"], self._ranges["ang_vel_yaw"])]
        cmd = torch.stack(draw, 1)
        cmd[torch.rand(k, generator=self._gen, device=self._dev) < STANDING_ENVS] = 0.0
        self.command[ids] = cmd
        self._command_step[ids] = 0

    def _reset_ids(self, ids):
        self._place(ids)
        self.action[ids] = 0.0
        self._last_action[ids] = 0.0
        self._last_qd[ids] = 0.0
        self.episode_step[ids] = 0
        self._push_step[ids] = 0
        self._track_sum[ids] = 0.0
        self._resample(ids)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._gen.manual_seed(int(seed))
        self.world.reset()
        nuka.sync()
        self._reset_ids(torch.arange(self.num_envs, device=self._dev))
        # Episodes start part of the way in so that the robots time out at different steps, as in rsl_rl.
        if not self._fixed_command:
            self.episode_step = torch.randint(0, self.max_episode_length, (self.num_envs,), generator=self._gen,
                                              device=self._dev)
        self._fresh[:] = True
        return self._observe(self._fresh), {}

    # -- curriculum ------------------------------------------------------------
    def _grow_commands(self):
        """Once per episode length, widen the linear and the yaw ranges by 0.1 toward their limits when the episodes
        that timed out since the last check tracked them well. Falls are left to the termination terms."""
        if self._fixed_command or self._common_step % self.max_episode_length != 0:
            return
        count = float(self._tracked[2])
        tracked = self._tracked[0:2] / (max(count, 1.0) * self.max_episode_length)
        self._tracked.zero_()
        for keys, score in (("lin_vel_x", "lin_vel_y"), tracked[0]), (("ang_vel_yaw",), tracked[1]):
            if count == 0 or float(score) <= 0.8:
                continue
            for key in keys:
                (lo, hi), (llo, lhi) = self._ranges[key], self._limit[key]
                self._ranges[key] = (round(max(llo, lo - 0.1), 3), round(min(lhi, hi + 0.1), 3))
        print(f"[g1_rough] step {self._common_step}: {int(count)} timed out, tracking {tracked[0]:.3f} linear "
              f"{tracked[1]:.3f} yaw; command ranges {self._ranges}", flush=True)

    # -- step ------------------------------------------------------------------
    def step(self, actions: torch.Tensor):
        self._last_action = self.action
        self.action = actions.to(self._dev, torch.float32).nan_to_num(0.0).clamp(-100.0, 100.0)
        self._tgt[:, self.slots] = self.default + ACTION_SCALE * self.action
        if self.push:
            self._push_step += 1
            due = self._push_step >= int(round(5.0 / self.step_dt))
            if bool(due.any()):
                kick = (torch.rand(self.num_envs, 2, generator=self._gen, device=self._dev) * 2 - 1) * 0.5
                self._vel[due, 0, 3:5] += kick[due]
                self._push_step[due] = 0
        nuka.sync()
        self.world.step_n(self.decimation)
        self.episode_step += 1
        self._command_step += 1
        self._common_step += 1

        reward, track = self._reward()
        self._track_sum += track
        terminated = self._terminated()
        truncated = self.episode_step >= self.max_episode_length
        done = terminated | truncated
        timed_out = (truncated & ~terminated).float()
        self._tracked[0:2] += (self._track_sum * timed_out[:, None]).sum(0)
        self._tracked[2] += timed_out.sum()
        self._grow_commands()
        ids = torch.nonzero(done).flatten()
        if len(ids):
            self.world.reset_envs(ids.to(torch.int32).cpu())
            self._reset_ids(ids)
        due = torch.nonzero(self._command_step >= int(round(10.0 / self.step_dt))).flatten()
        self._resample(due)
        self._last_qd = self._qd[:, self.slots].clone()
        self._last_feet = self._pose[:, self._feet, 0:3].clone()
        self._fresh = done
        return self._observe(done), reward, terminated, truncated, {}

    def _terminated(self):
        base = self._base[:, 0:3]
        height = base[:, 2] - self.terrain(base[:, 0], base[:, 1])
        upright = -self._gravity()[:, 2]
        bad = ~torch.isfinite(self._base).all(1) | ~torch.isfinite(self._q).all(1)
        return (height < self.termination_height) | (upright < self.termination_cos) | bad

    def _reward(self):
        s = self.scales
        q, qd = self._q[:, self.slots], self._qd[:, self.slots]
        quat = self._base_quat()
        w_body, v_body = self._vel[:, 0, 0:3], self._vel[:, 0, 3:6]
        v_world = quat_rotate_wxyz(quat, v_body)
        w_world = quat_rotate_wxyz(quat, w_body)
        yaw = torch.atan2(2 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
                          1 - 2 * (quat[:, 2] ** 2 + quat[:, 3] ** 2))
        c, sn = torch.cos(yaw), torch.sin(yaw)
        v_yaw = torch.stack((c * v_world[:, 0] + sn * v_world[:, 1], -sn * v_world[:, 0] + c * v_world[:, 1]), 1)
        cmd = self.command
        moving = cmd.norm(dim=1) > 0.1
        sole, feet = self._soles()
        contact = sole < CONTACT_MARGIN
        foot_vel = (feet - self._last_feet)[..., 0:2].norm(dim=-1) / self.step_dt
        ground = self.terrain(self._base[:, 0], self._base[:, 1])
        torque = (self.kp * (self._tgt[:, self.slots] - q) - self.kd * qd).clamp(-self.effort, self.effort)
        phase = (self.episode_step.float() * self.step_dt % GAIT_PERIOD) / GAIT_PERIOD
        stance = torch.stack([((phase + o) % 1.0) < GAIT_STANCE for o in GAIT_OFFSETS], 1)
        touch = self._pose[:, self._touch, 0:3]
        touch_h = touch[..., 2] - self.terrain(touch[..., 0], touch[..., 1])
        # A robot placed by a reset has no previous foot position yet.
        foot_vel[self._fresh] = 0.0
        track = torch.exp(-((cmd[:, 0:2] - v_yaw) ** 2).sum(1) / 0.25)
        terms = {
            "track_lin_vel_xy": track,
            "track_ang_vel_z": torch.exp(-(cmd[:, 2] - w_world[:, 2]) ** 2 / 0.25),
            "alive": torch.ones(self.num_envs, device=self._dev),
            "lin_vel_z": v_body[:, 2] ** 2,
            "ang_vel_xy": (w_body[:, 0:2] ** 2).sum(1),
            "joint_vel": (qd ** 2).sum(1),
            "joint_acc": (((qd - self._last_qd) / self.step_dt) ** 2).sum(1),
            "action_rate": ((self.action - self._last_action) ** 2).sum(1),
            "dof_pos_limits": ((q - self.soft_hi).clamp(min=0) + (self.soft_lo - q).clamp(min=0)).sum(1),
            "energy": (torque * qd).abs().sum(1),
            "joint_deviation_arms": ((q - self.default).abs() * self._arms).sum(1),
            "joint_deviation_waist": ((q - self.default).abs() * self._waist).sum(1),
            "joint_deviation_legs": ((q - self.default).abs() * self._legs).sum(1),
            "flat_orientation": (self._gravity()[:, 0:2] ** 2).sum(1),
            "base_height": (self._base[:, 2] - ground - BASE_HEIGHT_TARGET) ** 2,
            "gait": (contact == stance).float().sum(1) * moving,
            "feet_slide": (foot_vel * contact).sum(1),
            "feet_clearance": torch.exp(-(((sole - CLEARANCE_TARGET) ** 2) * torch.tanh(2.0 * foot_vel)).sum(1)
                                        / CLEARANCE_STD),
            "undesired_contacts": (touch_h < TOUCH_MARGIN).float().sum(1),
        }
        reward = sum(terms[k] * s[k] for k in terms) * self.step_dt
        return reward.nan_to_num_(0.0), torch.stack((track, terms["track_ang_vel_z"]), 1)

    def close(self):
        if getattr(self, "world", None) is not None:
            self.world.destroy()
            self.world = None
        if getattr(self, "_owns_device", False) and getattr(self, "_device", None) is not None:
            self._device.close()
            self._device = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def make_env(num_envs: int, *, scene: str, device=None, **kw) -> G1RoughEnv:
    return G1RoughEnv(scene, num_envs, device=device, **kw)
