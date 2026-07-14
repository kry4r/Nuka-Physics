"""On-GPU proprioceptive observation builder for the BDX biped (open_duck_mini_v2).

Name-based (not quadrant-structural like Go2): the actuated joints are resolved
from the cooked ``dof_names`` via regex, so the same builder serves any robot whose
joints follow the naming convention. The policy controls the 10 LEG joints; the 4
head/neck joints are PD-held at the home stance (not in the action) -- they are
cosmetic for flat locomotion and adding them only enlarges the search space.

The 44-dim obs (per env), all torch ops on the zero-copy DLPack CUDA views:

    [ 0: 3] base_lin_vel (body frame) * 2.0
    [ 3: 6] base_ang_vel (body frame) * 0.25
    [ 6: 9] projected_gravity (world [0,0,-1] into body frame; unit)
    [ 9:12] command[:3] * [2.0, 2.0, 0.25]
    [12:22] (q_leg - default_leg) * 1.0        (10 leg joints, policy order)
    [22:32] qd_leg * 0.05                       (10 leg joints, policy order)
    [32:42] last_action (previous raw clipped policy output)
    [42:44] gait phase (cos, sin) of the reference-motion period clock

Action (10): target_q = default_leg + ACTION_SCALE * action, then rate-limited to the
sts3215 max motor speed; head slots hold the home stance. Kp=13.37, Kd=0,
force limit 3.23 N.m (the real servo gains applied by the general drive path).
"""

from __future__ import annotations

import os

import numpy as np
import torch

import nuka

# Policy joint order (the 10 actuated LEG joints): left leg then right leg.
LEG_JOINTS = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
# Head/neck joints -- PD-held at home, NOT part of the action.
HEAD_JOINTS = ["neck_pitch", "head_pitch", "head_yaw", "head_roll"]

BDX_ACTION_DIM = 10
BDX_OBS_DIM = 44

# obs scales (legged_gym convention).
S_LIN_VEL = 2.0
S_ANG_VEL = 0.25
S_DOF_VEL = 0.05
CMD_SCALE = np.array([2.0, 2.0, 0.25], dtype=np.float32)
ACTION_SCALE = float(os.environ.get("BDX_ACTION_SCALE", "0.5"))
OBS_CLIP = 100.0
ACTION_CLIP = 100.0
ACTION_SPACE_LIMIT = 1.0

# sts3215 position servo (kp=13.37, kv=0, forcerange +-3.23; joint damping 0.56 is
# applied by the engine ABA, not here). Max motor speed 5.24 rad/s (target rate cap).
KP, KD, FORCE_LIMIT = 13.37, 0.0, 3.23
MAX_MOTOR_VEL = 5.24

# Leg joint position limits (rad), policy order, parsed from the MJCF <joint range>.
DOF_POS_LOWER = np.array(
    [-0.5236, -0.4363, -1.2217, -1.5708, -1.5708,
     -0.5236, -0.4363, -0.5236, -1.5708, -1.5708], dtype=np.float32,
)
DOF_POS_UPPER = np.array(
    [0.5236, 0.4363, 0.5236, 1.5708, 1.5708,
     0.5236, 0.4363, 1.2217, 1.5708, 1.5708], dtype=np.float32,
)


def quat_rotate_inverse_wxyz(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate world vector(s) ``v`` into the BODY frame (R(q)^T @ v). ``q_wxyz`` is
    (...,4) [w,x,y,z]; ``v`` is (...,3) and broadcasts. All-torch, on-device."""
    w = q_wxyz[..., 0:1]
    qvec = q_wxyz[..., 1:4]
    a = (w * w - (qvec * qvec).sum(-1, keepdim=True)) * v
    b = 2.0 * w * torch.cross(qvec, v.expand_as(qvec), dim=-1)
    c = 2.0 * qvec * (qvec * v).sum(-1, keepdim=True)
    return a - b + c


def quat_rotate_wxyz(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate body vector(s) ``v`` into the WORLD frame (R(q) @ v)."""
    w = q_wxyz[..., 0:1]
    qvec = q_wxyz[..., 1:4]
    a = (w * w - (qvec * qvec).sum(-1, keepdim=True)) * v
    b = 2.0 * w * torch.cross(qvec, v.expand_as(qvec), dim=-1)
    c = 2.0 * qvec * (qvec * v).sum(-1, keepdim=True)
    return a + b + c


class BdxObsBuilder:
    """On-GPU composer of the 42-dim BDX obs + the name-resolved action map.

    Construct once per world. Resolves the leg/head joint slots from the cooked DOF
    names, caches the home stance as the PD default, applies the servo gains, and
    holds the foot-contact geometry (ankle-link slots + sole capsules) for the
    kinematic foot-contact signal the reward uses.
    """

    def __init__(self, world, foot_ankle_slots, foot_caps, *, dtype=torch.float32):
        self.world = world
        self.num_envs = world.env_count
        self.blc = int(world.base_link_count)
        self._dev = torch.device("cuda")
        self.dtype = dtype

        # Name -> cooked slot (per-env buffer column) for legs + head.
        def _slot(name):
            idx = world.dof_index("^" + name + "$")
            if len(idx) != 1:
                raise RuntimeError(f"BdxObsBuilder: joint {name!r} -> {idx.tolist()}")
            return int(idx[0])
        self.leg_slots = torch.as_tensor(
            [_slot(n) for n in LEG_JOINTS], dtype=torch.long, device=self._dev
        )
        self.head_slots = torch.as_tensor(
            [_slot(n) for n in HEAD_JOINTS], dtype=torch.long, device=self._dev
        )

        # Zero-copy engine views (alias the live buffers; re-read each step).
        self._q = torch.from_dlpack(world.buffer_view(nuka.JOINT_POSITION))
        self._qd = torch.from_dlpack(world.buffer_view(nuka.JOINT_VELOCITY))
        self._pose = torch.from_dlpack(world.buffer_view(nuka.ARTICULATION_LINK_POSE))
        self._vel = torch.from_dlpack(world.buffer_view(nuka.LINK_VELOCITY))
        self._tgt = torch.from_dlpack(world.buffer_view(nuka.DRIVE_TARGET))
        self._base_pose = torch.from_dlpack(world.buffer_view(nuka.BASE_POSE))

        # Home stance (cooked JOINT_POSITION) is the PD default target.
        q0 = self._q.detach().clone()
        self.default_leg = q0[0, self.leg_slots].to(dtype).contiguous()   # (10,)
        self.default_head = q0[0, self.head_slots].to(dtype).contiguous()  # (4,)

        self.cmd_scale = torch.as_tensor(CMD_SCALE, dtype=dtype, device=self._dev)
        self.soft_lo, self.soft_hi = self._soft_limits(dtype)

        # Foot contact geometry (ankle-link slots + sole capsule local pose/radius).
        self.foot_ankle_slots = torch.as_tensor(
            foot_ankle_slots, dtype=torch.long, device=self._dev
        )                                                                # (2,)
        self.foot_cap_local = torch.as_tensor(
            np.stack([c[0] for c in foot_caps]), dtype=dtype, device=self._dev
        )                                                                # (2,3)
        self.foot_cap_radius = torch.as_tensor(
            [c[1] for c in foot_caps], dtype=dtype, device=self._dev
        )                                                                # (2,)

        # Rate-limited PD target (prev control step's leg target); seeded at home.
        self.prev_leg_target = self.default_leg.expand(
            self.num_envs, BDX_ACTION_DIM
        ).contiguous()
        # ctrl-step target rate cap (set by the env from its control dt).
        self.target_rate_cap = MAX_MOTOR_VEL * 0.02

    @staticmethod
    def _soft_limits_static(lo, hi, factor=0.9):
        m = (lo + hi) * 0.5
        r = hi - lo
        return m - 0.5 * r * factor, m + 0.5 * r * factor

    def _soft_limits(self, dtype):
        lo = torch.as_tensor(DOF_POS_LOWER, dtype=dtype, device=self._dev)
        hi = torch.as_tensor(DOF_POS_UPPER, dtype=dtype, device=self._dev)
        return self._soft_limits_static(lo, hi)

    # -- gain setup (once at construction) -----------------------------------
    def apply_pd_gains(self) -> None:
        """Write the sts3215 servo gains (Kp/Kd/force limit) on every actuated slot
        (legs + head); the head slots also PD-hold the home target set below."""
        kp = torch.from_dlpack(self.world.buffer_view(nuka.DRIVE_STIFFNESS))
        kd = torch.from_dlpack(self.world.buffer_view(nuka.DRIVE_DAMPING))
        fl = torch.from_dlpack(self.world.buffer_view(nuka.DRIVE_FORCE_LIMIT))
        kp[:, 1:self.blc] = KP
        kd[:, 1:self.blc] = KD
        fl[:, 1:self.blc] = FORCE_LIMIT
        # Head holds the home stance for the whole run (never touched by the policy).
        self._tgt[:, self.head_slots] = self.default_head
        self._tgt[:, self.leg_slots] = self.default_leg

    def set_target_rate_cap(self, ctrl_dt: float) -> None:
        self.target_rate_cap = float(MAX_MOTOR_VEL * ctrl_dt)

    # -- per-step reads (all on GPU) -----------------------------------------
    def q_leg(self) -> torch.Tensor:
        return self._q[:, self.leg_slots]

    def qd_leg(self) -> torch.Tensor:
        return self._qd[:, self.leg_slots]

    def base_pos(self) -> torch.Tensor:
        return self._base_pose[:, 0:3]

    def base_quat_wxyz(self) -> torch.Tensor:
        return self._base_pose[:, 3:7]

    def base_ang_vel(self) -> torch.Tensor:
        return self._vel[:, 0, 0:3]

    def base_lin_vel(self) -> torch.Tensor:
        return self._vel[:, 0, 3:6]

    def projected_gravity(self) -> torch.Tensor:
        g = torch.tensor([0.0, 0.0, -1.0], dtype=self.dtype, device=self._dev).expand(
            self.num_envs, 3
        )
        return quat_rotate_inverse_wxyz(self.base_quat_wxyz(), g)

    def tilt_deg(self) -> torch.Tensor:
        pg = self.projected_gravity()
        cos_up = (-pg[:, 2]).clamp(-1.0, 1.0)
        return torch.rad2deg(torch.arccos(cos_up))

    def foot_bottom_xyz(self) -> torch.Tensor:
        """(N,2,3) each sole capsule's lowest world point [x, y, z] (ankle link pose
        composed with the capsule local pose; z lowered by the capsule radius)."""
        pose = self._pose[:, self.foot_ankle_slots, :]            # (N,2,7)
        pos = pose[:, :, 0:3]                                     # (N,2,3)
        quat = pose[:, :, 3:7]                                    # (N,2,4)
        local = self.foot_cap_local.expand(self.num_envs, 2, 3)   # (N,2,3)
        world = pos + quat_rotate_wxyz(quat, local)               # (N,2,3)
        bottom = world.clone()
        bottom[:, :, 2] = world[:, :, 2] - self.foot_cap_radius
        return bottom

    def foot_bottom_z(self) -> torch.Tensor:
        """(N,2) each sole capsule's lowest world z."""
        return self.foot_bottom_xyz()[:, :, 2]

    def foot_contact(self, ground_z=0.0, thresh: float = 0.02) -> torch.Tensor:
        """(N,2) bool: a foot is loaded iff its sole bottom is within ``thresh`` of
        the ground. ``ground_z`` is a scalar (flat, default 0) or an (N,2) local
        terrain surface under each sole -- the kinematic contact proxy (sidesteps
        the heightfield contact-wrench zero-fill)."""
        return self.foot_bottom_z() <= (ground_z + thresh)

    # -- obs composition ------------------------------------------------------
    def compute_obs(self, command: torch.Tensor, last_action: torch.Tensor,
                    phase: torch.Tensor) -> torch.Tensor:
        """``phase``: (N,2) [cos, sin] of the reference-motion period clock."""
        obs = torch.cat(
            (
                self.base_lin_vel() * S_LIN_VEL,
                self.base_ang_vel() * S_ANG_VEL,
                self.projected_gravity(),
                command * self.cmd_scale,
                (self.q_leg() - self.default_leg) * 1.0,
                self.qd_leg() * S_DOF_VEL,
                last_action,
                phase,
            ),
            dim=-1,
        )
        return obs.clamp_(-OBS_CLIP, OBS_CLIP)

    # -- action -> DRIVE_TARGET (leg slots; rate-limited) --------------------
    def write_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map a (N,10) raw action to rate-limited PD leg targets and write them into
        the leg slots. Returns the CLIPPED raw action (for next obs's last_action)."""
        action = action.clamp(-ACTION_CLIP, ACTION_CLIP)
        target = self.default_leg + ACTION_SCALE * action           # (N,10)
        cap = self.target_rate_cap
        target = torch.clamp(target, self.prev_leg_target - cap,
                             self.prev_leg_target + cap)
        self.prev_leg_target = target
        self._tgt[:, self.leg_slots] = target
        return action

    def reset_prev_target(self, mask=None) -> None:
        """Reset the rate-limit memory to the home target (all envs or the masked)."""
        if mask is None:
            self.prev_leg_target[:] = self.default_leg
        else:
            self.prev_leg_target[mask] = self.default_leg
