"""BDX biped locomotion reward -- the Open Duck Playground imitation recipe.

The full Playground joystick weight system, ported to torch: velocity tracking
(shared sigma), alive, torque/action-rate/stand-still regularizers, and the
IMITATION term that tracks the polynomial reference gait (joint positions
dominate at w=15, plus base-velocity and foot-contact matching, gated off at
zero command). Per-step total is scaled by the control dt and floored at 0.

Velocities are compared in the BODY frame: the reference was recorded at a +x
heading (body == world there), so the body-frame comparison keeps the term
meaningful at every heading. Foot contact is the kinematic sole proxy supplied
by the env (robust on flat ground; heightfield contact wrench is zero-filled).
"""

from __future__ import annotations

import torch

from . import bdx_obs as B

# Control dt = decimation(4) * sim_dt(0.005) = 0.02 (the reward dt).
REWARD_DT = 0.02

# Playground joystick reward scales (raw config values; REWARD_DT applied in
# compute()). The weights are one tuned system -- retune via yaml only as a set.
SCALES = {
    "tracking_lin_vel": 2.5,
    "tracking_ang_vel": 6.0,
    "torques": -1.0e-3,
    "action_rate": -0.5,
    "stand_still": -0.2,
    "alive": 20.0,
    "imitation": 1.0,
}

TRACKING_SIGMA = 0.01
LIN_VEL_Y_TOL = 0.1
CMD_ACTIVE_THRESHOLD = 0.01

# Imitation internal weights (Playground custom_rewards.reward_imitation).
W_LIN_VEL_XY = 1.0
W_LIN_VEL_Z = 1.0
W_ANG_VEL_XY = 0.5
W_ANG_VEL_Z = 0.5
W_JOINT_POS = 15.0
W_JOINT_VEL = 1.0e-3
W_CONTACT = 1.0
# Per-joint anchor over the leg order [hy,hr,hp,kn,an]x2. Uniform 1.0: the full
# anchor IS the stepping pressure (freeing sagittal joints re-opens the stand basin).
IMITATION_JOINT_WEIGHTS = [1.0] * 10


class BdxReward:
    """On-GPU evaluator of the Playground-style BDX reward (10 leg joints)."""

    def __init__(self, num_envs, device, *, dtype=torch.float32,
                 scales=None, sigma_lin=TRACKING_SIGMA, sigma_ang=TRACKING_SIGMA):
        self.num_envs = int(num_envs)
        self.device = device
        self.dtype = dtype
        self.scales = dict(SCALES)
        if scales:
            self.scales.update({k: float(v) for k, v in scales.items()})
        self.sigma_lin = float(sigma_lin)
        self.sigma_ang = float(sigma_ang)

        # Overwritten by the env with the cooked home stance.
        self.default_leg = torch.zeros(B.BDX_ACTION_DIM, dtype=dtype, device=device)
        self.kp = float(B.KP)
        self.force_limit = float(B.FORCE_LIMIT)
        self.imitation_joint_w = torch.as_tensor(
            IMITATION_JOINT_WEIGHTS, dtype=dtype, device=device)

    # -- term helpers ---------------------------------------------------------
    def _tracking_lin_vel(self, cmd, lin_vel):
        err_x = (cmd[:, 0] - lin_vel[:, 0]).pow(2)
        err_y = (lin_vel[:, 1] - cmd[:, 1]).abs().sub(LIN_VEL_Y_TOL).clamp(min=0.0).pow(2)
        return torch.exp(-(err_x + err_y) / self.sigma_lin)

    def _tracking_ang_vel(self, cmd, ang_vel):
        err = (cmd[:, 2] - ang_vel[:, 2]).pow(2)
        return torch.exp(-err / self.sigma_ang)

    def _torques(self, q_leg, target_leg):
        # Position servo (kv=0) with the sts3215 force limit, as the plant applies it.
        tau = (self.kp * (target_leg - q_leg)).clamp(-self.force_limit, self.force_limit)
        return tau.pow(2).sum(dim=1)

    def _stand_still(self, q_leg, qd_leg, cmd):
        cmd_norm = cmd[:, :3].norm(dim=1)
        pose_cost = (q_leg - self.default_leg).abs().sum(dim=1)
        vel_cost = qd_leg.abs().sum(dim=1)
        return (pose_cost + vel_cost) * (cmd_norm < CMD_ACTIVE_THRESHOLD).to(self.dtype)

    def _imitation(self, cmd, lin_vel, ang_vel, q_leg, qd_leg, contact,
                   ref_leg_pos, ref_leg_vel, ref_contacts, ref_lin_vel, ref_ang_vel):
        lin_xy = torch.exp(
            -8.0 * (lin_vel[:, :2] - ref_lin_vel[:, :2]).pow(2).sum(dim=1)) * W_LIN_VEL_XY
        lin_z = torch.exp(
            -8.0 * (lin_vel[:, 2] - ref_lin_vel[:, 2]).pow(2)) * W_LIN_VEL_Z
        ang_xy = torch.exp(
            -2.0 * (ang_vel[:, :2] - ref_ang_vel[:, :2]).pow(2).sum(dim=1)) * W_ANG_VEL_XY
        ang_z = torch.exp(
            -2.0 * (ang_vel[:, 2] - ref_ang_vel[:, 2]).pow(2)) * W_ANG_VEL_Z
        joint_pos = -((q_leg - ref_leg_pos).pow(2)
                      * self.imitation_joint_w).sum(dim=1) * W_JOINT_POS
        joint_vel = -(qd_leg - ref_leg_vel).pow(2).sum(dim=1) * W_JOINT_VEL
        contact_match = (contact.to(self.dtype) == ref_contacts).to(self.dtype).sum(
            dim=1) * W_CONTACT
        rew = lin_xy + lin_z + ang_xy + ang_z + joint_pos + joint_vel + contact_match
        active = (cmd[:, :3].norm(dim=1) > CMD_ACTIVE_THRESHOLD).to(self.dtype)
        return torch.nan_to_num(rew * active)

    # -- full reward ----------------------------------------------------------
    def compute(self, *, cmd, lin_vel, ang_vel, q_leg, qd_leg, target_leg,
                action, last_action, contact,
                ref_leg_pos, ref_leg_vel, ref_contacts, ref_lin_vel, ref_ang_vel):
        terms = {
            "tracking_lin_vel": self._tracking_lin_vel(cmd, lin_vel),
            "tracking_ang_vel": self._tracking_ang_vel(cmd, ang_vel),
            "torques": self._torques(q_leg, target_leg),
            "action_rate": (action - last_action).pow(2).sum(dim=1),
            "stand_still": self._stand_still(q_leg, qd_leg, cmd),
            "alive": torch.ones(self.num_envs, dtype=self.dtype, device=self.device),
            "imitation": self._imitation(
                cmd, lin_vel, ang_vel, q_leg, qd_leg, contact,
                ref_leg_pos, ref_leg_vel, ref_contacts, ref_lin_vel, ref_ang_vel),
        }
        total = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        for name, term in terms.items():
            scale = self.scales.get(name, 0.0)
            if scale != 0.0:
                total = total + term * (scale * REWARD_DT)
        return total.clamp(min=0.0)
