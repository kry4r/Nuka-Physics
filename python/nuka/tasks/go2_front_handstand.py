"""Go2 FRONT-handstand walk task -- direct-torque, command-conditioned policy.

Skill: the robot balances nose-down-vertical on its two FRONT paws while the
FREE rear paws swing/walk fore-aft under motion-module commands (S0 hold -> S1
small swings -> S2 anti-phase walk). Geometry verified by the FK/dynamic probes
(tools/reference/probe_front_handstand.py): at pgx -> +1 the front paws plant
under the lowered nose end and the rear rises clear.

CONTRACT (spec): the ONLY control source is policy torque -- actions map
linearly to joint torques saturated at the URDF effort limits. The motion
commander's targets enter via obs + reward tracking terms, never torques.
"""
from __future__ import annotations

import numpy as np
import torch

import nuka

from ..gym.env import NukaGymEnv
from . import go2_obs as G
from .go2_motion_cmd import HandstandWalkCommander, CMD_DIM

# Settle-verified vertical posture (probe champions): front legs folded under
# with paws planted, rear legs shallow-folded and lifted.
NOMINAL_URDF = np.array(
    [0.0, 0.85, -1.8,   0.0, 0.85, -1.8,
     0.0, -1.2, -1.2,   0.0, -1.2, -1.2], dtype=np.float32)
IC_BASE_Z = 0.445          # standard Go2 standing root height
PITCH_DEG = 90.0          # target transition angle, not the reset pose
TRANSITION_RAMP_S = 3.0
HANDSTAND_GATE_S = 3.0

# Reward weights.
W_ORIENT = 2.5            # pitch progress toward vertical (front-load gated)
W_ORIENT_Y = 1.0          # roll/yaw suppression (stay planar)
W_TRUNK_CLEAR = 1.0       # trunk off the ground
W_PROGRESS = 3.0          # forward base advance while holding vertical
V_WALK_TARGET = 0.12      # m/s progress that earns full W_PROGRESS
# Strict handstand: rear legs fully airborne, only front paws touch ground.
REAR_TUCK_M = 0.10        # desired rear-paw height above ground
REAR_TUCK_SIGMA = 0.02
REAR_FZ_SCALE = 8.0       # N; exp(-sum(Fz)/scale) -> any real contact hurts
W_REAR_TUCK = 2.0
W_REAR_UNLOADED = 2.5
W_LIMB_CONTACT = 1.0      # thigh/calf links must NOT touch (paws-only support)
LIMB_CONTACT_SCALE = 20.0  # N; sum(Fz)/scale -> a loaded thigh is heavily taxed
W_ALIVE = 0.15
W_HOLD = 3.0              # sustained in-handstand bonus (anti-harvest)
K_ORIENT = 4.0
FRONT_LOAD_MIN_N = 5.0    # a front paw counts as LOADED above this Fz


class Go2HandstandWalkObs(G.Go2ObsBuilder):
    """Base PD-target observations plus the quad2hand 8D gait clock."""

    def __init__(self, device, world) -> None:
        super().__init__(device, world)
        self.default_angles = torch.as_tensor(
            G.DEFAULT_ANGLES, dtype=torch.float32, device=self.q_urdf().device)
        self.nominal_angles = torch.as_tensor(
            NOMINAL_URDF, dtype=torch.float32, device=self.q_urdf().device)
        self.cmd: "HandstandWalkCommander | None" = None

    def wire_commander(self, cmd: HandstandWalkCommander) -> None:
        self.cmd = cmd

    def state_finite(self) -> torch.Tensor:
        return (torch.isfinite(self.q_urdf()).all(-1)
                & torch.isfinite(self.qd_urdf()).all(-1)
                & torch.isfinite(self.base_lin_vel()).all(-1)
                & torch.isfinite(self.base_ang_vel()).all(-1))

    def write_action(self, action: torch.Tensor) -> torch.Tensor:
        """Zero action holds the normal stance; positive action moves toward
        the validated front-paw handstand posture."""
        action = action.clamp(-1.0, 1.0)
        target_urdf = self.default_angles + action * (
            self.nominal_angles - self.default_angles)
        self._tgt[:, 1:G.GO2_BLC] = target_urdf.index_select(
            1, self.nuka_slot_for_urdf)
        return action

    def compute_obs(self, command: torch.Tensor,
                    last_action: torch.Tensor) -> torch.Tensor:
        base = super().compute_obs(command, last_action)
        return torch.cat((base, self.cmd.cmd_tensor()), dim=-1).clamp(
            -G.OBS_CLIP, G.OBS_CLIP)


class Go2FrontHandstandEnv(NukaGymEnv):
    """Vertical front-support balance + rear-feet walking under commands."""

    def __init__(self, scene: str, num_envs: int, *,
                 episode_length_s: float = 8.0,
                 seed: int | None = None,
                 fixed_vx: float | None = None,
                 reference_torque: bool = False,
                 **kw) -> None:
        self._reference_torque = bool(reference_torque)
        if not self._reference_torque:
            kw.pop("control_mode", None)
        kw.pop("command", None)
        tc = dict(kw.pop("terrain_create", {}) or {})
        if not tc:
            tc = dict(contact_family=1, heightfield_terrain_type=1,
                      heightfield_nrow=161, heightfield_ncol=161,
                      heightfield_cell=0.25)
        kw["terrain_create"] = tc
        if self._reference_torque:
            kw["control_mode"] = 1
        super().__init__(
            scene, num_envs,
            command=(0.0, 0.0, 0.0),
            episode_length_s=episode_length_s,
            seed=seed,
            obs_builder_factory=lambda dev, world: Go2HandstandWalkObs(dev, world),
            **kw)

        dev = self._torch_device
        self._obs_builder: Go2HandstandWalkObs = self._obs
        self._cmd = HandstandWalkCommander(num_envs, dev, fixed_vx=fixed_vx)
        self._obs_builder.wire_commander(self._cmd)
        kp = torch.from_dlpack(self._world.buffer_view(nuka.DRIVE_STIFFNESS))
        kd = torch.from_dlpack(self._world.buffer_view(nuka.DRIVE_DAMPING))
        kp[:, 1:G.GO2_BLC] = 30.0
        kd[:, 1:G.GO2_BLC] = 0.75

        # Widen the gym spaces to the command-augmented obs (Option A pattern).
        obs_dim = G.GO2_OBS_DIM + CMD_DIM
        from gymnasium import spaces as _spaces
        self.single_observation_space = _spaces.Box(
            low=-G.OBS_CLIP, high=G.OBS_CLIP, shape=(obs_dim,), dtype=np.float32)
        self.observation_space = _spaces.Box(
            low=-G.OBS_CLIP, high=G.OBS_CLIP,
            shape=(self.num_envs, obs_dim), dtype=np.float32)

        # Zero-copy engine views + derived slots.
        self._wrench = torch.from_dlpack(
            self._world.buffer_view(nuka.LINK_CONTACT_WRENCH))
        self._pose = torch.from_dlpack(
            self._world.buffer_view(nuka.ARTICULATION_LINK_POSE))
        u_calf = [G.URDF_JOINT_NAMES.index(nm)
                  for nm in ("FL_calf", "FR_calf", "RL_calf", "RR_calf")]
        self._paw_slot = torch.tensor(
            [int(self._obs.nuka_slot_for_urdf_np[u]) + 1 for u in u_calf],
            dtype=torch.long, device=dev)
        u_thigh = [G.URDF_JOINT_NAMES.index(nm)
                   for nm in ("FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh")]
        self._limb_slot = torch.tensor(
            [int(self._obs.nuka_slot_for_urdf_np[u]) + 1 for u in u_thigh],
            dtype=torch.long, device=dev)
        self._foot_off = torch.tensor([[0.0, 0.0, -0.213]], device=dev)
        self._quat_conj = torch.tensor([[1.0, -1.0, -1.0, -1.0]], device=dev)

        # IC tensors.
        self._nominal = torch.as_tensor(NOMINAL_URDF, device=dev)
        self._prev_paw = torch.zeros(num_envs, 4, 3, device=dev)
        self._prev_action = torch.zeros(num_envs, G.GO2_ACTION_DIM, device=dev)
        self.dt_control = self.dt * self.decimation

        print(f"[go2_front_handstand] n={num_envs} action_mode=pd "
              f"obs_dim={obs_dim} gait_period=0.4s", flush=True)

    # -- ICs -----------------------------------------------------------------
    def _reset_joint_state(self, mask) -> None:
        m = torch.ones(self.num_envs, dtype=torch.bool,
                       device=self._torch_device) if mask is None else mask
        if not bool(m.any()):
            return
        # Keep the physical reset in the ordinary four-foot stance. The skill
        # must learn the pitch-up transition instead of receiving it for free.
        q = torch.from_dlpack(self._world.buffer_view(nuka.JOINT_POSITION))
        qd = torch.from_dlpack(self._world.buffer_view(nuka.JOINT_VELOCITY))
        normal_urdf = torch.as_tensor(G.DEFAULT_ANGLES, device=self._torch_device)
        normal_slots = normal_urdf.index_select(0, self._obs.urdf_from_nuka_slot)
        q[m, 1:G.GO2_BLC] = normal_slots.unsqueeze(0).expand(int(m.sum()), -1)
        qd[m, 1:G.GO2_BLC] = 0.0
        tgt = torch.from_dlpack(self._world.buffer_view(nuka.DRIVE_TARGET))
        normal_targets = normal_urdf.index_select(0, self._obs.nuka_slot_for_urdf)
        tgt[m, 1:G.GO2_BLC] = normal_targets.unsqueeze(0).expand(int(m.sum()), -1)
        bp = torch.from_dlpack(self._world.buffer_view(nuka.BASE_POSE))
        bp[m, 2] = IC_BASE_Z
        bp[m, 3] = 1.0           # BASE_POSE quaternion is [qw,qx,qy,qz]
        bp[m, 4:7] = 0.0
        self._cmd.resample(m, self._generator)
        self.command[m] = self._cmd.command[m]
        self._prev_action[m] = 0.0

    # -- helpers -------------------------------------------------------------
    def _paw_world(self) -> torch.Tensor:
        """(N,4,3) paw world positions (calf origin + shank offset)."""
        origin = self._pose[:, self._paw_slot, 0:3]
        quat = self._pose[:, self._paw_slot, 3:7] * self._quat_conj
        return origin + G.quat_rotate_inverse_wxyz(quat, self._foot_off.expand(
            self.num_envs, 4, 3))

    def _front_fz(self) -> torch.Tensor:
        return self._wrench[:, self._paw_slot[0:2], 2].abs()   # (N,2)

    # -- reward / termination ------------------------------------------------
    def compute_reward(self) -> torch.Tensor:
        """Validated quad2hand handstand-walk reward, scaled by control dt."""
        b = self._obs
        pg = b.projected_gravity()
        vel = b.base_lin_vel()
        omega = b.base_ang_vel()
        paw = self._paw_world()
        paw_vel = (paw - self._prev_paw) / self.dt_control
        paw_vel = torch.where((self.episode_step > 1).view(-1, 1, 1),
                              paw_vel, torch.zeros_like(paw_vel))
        self._prev_paw = paw.clone()

        fz = self._wrench[:, self._paw_slot, 2].abs()
        contact = fz > FRONT_LOAD_MIN_N
        gait = (contact[:, 0:2] == self._cmd.is_stance()[:, 0:2]).float().mean(dim=-1)

        vx_err = (self.command[:, 0] - vel[:, 0]).pow(2)
        yaw_err = (self.command[:, 2] - omega[:, 2]).pow(2)
        r_vx = torch.exp(-vx_err / 0.25)
        r_yaw = torch.exp(-yaw_err / 0.25)
        t = self.episode_step.float() * self.dt_control
        ramp = (t / TRANSITION_RAMP_S).clamp(0.0, 1.0)
        target_pitch = torch.as_tensor(np.radians(84.0), device=pg.device) * ramp
        target_pg = torch.stack((target_pitch.sin(),
                                 torch.zeros_like(target_pitch),
                                 -target_pitch.cos()), dim=-1)
        orient_err = (pg - target_pg).pow(2).sum(dim=-1)
        r_orient = torch.exp(-orient_err / 0.25)
        height_target = IC_BASE_Z + (0.30 - IC_BASE_Z) * ramp
        r_height = torch.exp(-(b.base_pos()[:, 2] - height_target).pow(2) / 0.04)

        front_speed = paw_vel[:, 0:2, 0:2].norm(dim=-1)
        clear_err = (front_speed * (paw[:, 0:2, 2] - 0.072).pow(2)).sum(dim=-1)
        r_clearance = torch.exp(-clear_err / 0.01)
        rear_contact = contact[:, 2:4].float().sum(dim=-1) * ramp

        q = b.q_urdf()
        posture_target = (self._obs_builder.default_angles.unsqueeze(0)
                          + ramp.unsqueeze(-1) *
                          (self._nominal.unsqueeze(0) -
                           self._obs_builder.default_angles.unsqueeze(0)))
        rear_posture = (q[:, 6:12] - posture_target[:, 6:12]).pow(2).sum(dim=-1) * ramp
        hip_posture = (q[:, [0, 3, 6, 9]]
                       - posture_target[:, [0, 3, 6, 9]]).pow(2).sum(dim=-1) * ramp
        contact_motion = (paw_vel.pow(2)
                          * contact.unsqueeze(-1).float()).sum(dim=(1, 2))
        action_rate = (self.last_action - self._prev_action).pow(2).sum(dim=-1)
        self._prev_action = self.last_action.clone()

        thigh_contact = (self._wrench[:, self._limb_slot, 2].abs()
                         > 1.0).float().sum(dim=-1)
        base_contact = (self._wrench[:, 0, 2].abs() > 1.0).float()
        collision = thigh_contact + base_contact

        r = (1.5 * r_vx
             + 1.0 * r_yaw
             + 0.5 * gait
             + 1.0 * r_height
             + 1.0 * r_orient
             + 0.2 * r_clearance
             - 2.0 * rear_contact
             - 0.1 * rear_posture
             - 1.0 * hip_posture
             - 0.1 * contact_motion
             - 0.001 * action_rate
             - 1.5 * collision) * self.dt_control
        return torch.nan_to_num(r, nan=-1.0).clamp(-10.0, 10.0)

    def compute_terminated(self) -> torch.Tensor:
        b = self._obs
        t = self.episode_step.float() * self.dt_control
        base_hit = self._wrench[:, 0, 2].abs() > 40.0
        fallen = b.base_pos()[:, 2] < 0.12
        lost_pose = (t >= HANDSTAND_GATE_S) & (b.projected_gravity()[:, 0] < 0.25)
        return base_hit | fallen | lost_pose | ~b.state_finite()

    def step(self, actions):
        self._cmd.advance(self.dt_control)
        self.command = self._cmd.command
        return super().step(actions)


def make_env(num_envs: int, *, device=None, **kw) -> Go2FrontHandstandEnv:
    scene = G._go2_scene_path()
    return Go2FrontHandstandEnv(scene, num_envs, device=device, **kw)
