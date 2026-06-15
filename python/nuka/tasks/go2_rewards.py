"""legged_gym Go2 reward terms, ported VERBATIM (term math + exact scales).

This is a faithful replication -- NOT a hand-rolled shaping -- of the Go2 reward
set that ``unitree_rl_gym`` (legged_gym) is proven-to-converge under. The term
balance (tracking-dominated positive, small regularizing negatives, the whole
total floored at 0 by ``only_positive_rewards``) is exactly what lets PPO learn to
walk; we copy it rather than invent.

Sources of truth (read + ported line-for-line):
  * ``legged_gym/envs/go2/go2_config.py``            -- go2 scale OVERRIDES
    (``torques=-0.0002``, ``dof_pos_limits=-10.0``), ``base_height_target=0.25``,
    ``soft_dof_pos_limit=0.9``.
  * ``legged_gym/envs/base/legged_robot_config.py``  -- INHERITED defaults: the
    rest of ``rewards.scales``, ``tracking_sigma=0.25``, ``only_positive_rewards``.
  * ``legged_gym/envs/base/legged_robot.py``         -- the ``_reward_*`` impls,
    ``compute_reward`` (``Σ scale·term``; clip total ≥0; termination added AFTER),
    and ``_prepare_reward_function`` (every scale is multiplied by the CONTROL dt
    ``self.dt = decimation·sim_dt = 4·0.005 = 0.02``).

THE CONTROL-DT MULTIPLY (the easiest 4x-off bug): legged_gym pre-multiplies every
reward scale by ``self.dt``, and ``self.dt`` is the *control* dt (0.02 s, one
policy step = ``decimation`` physics steps), NOT the physics dt (0.005). The same
0.02 also appears INSIDE ``_reward_dof_acc`` (``Δqd/dt``). We bake the control dt
into :data:`REWARD_DT` and use it both places.

----------------------------------------------------------------------------
PORTED reward terms (term | legged_gym scale | status):
  tracking_lin_vel  |  1.0      | yes   exp(-||cmd_xy - v_xy||^2 / 0.25)
  tracking_ang_vel  |  0.5      | yes   exp(-(cmd_wz - w_z)^2 / 0.25)
  lin_vel_z         | -2.0      | yes   v_z^2
  ang_vel_xy        | -0.05     | yes   Σ w_xy^2
  torques           | -0.0002   | APPROX  Σ τ^2 with τ = Kp(tgt-q) - Kd·qd
  dof_acc           | -2.5e-7   | yes   Σ ((qd_prev - qd)/dt)^2
  action_rate       | -0.01     | yes   Σ (last_action - action)^2
  dof_pos_limits    | -10.0     | yes   Σ out-of-(soft 0.9 URDF limit)
  feet_air_time     |  1.0      | yes   Σ (air_time - 0.5)·first_contact, gated
                                       by ||cmd_xy||>0.1 -- the TRUE per-foot
                                       contact comes from LINK_CONTACT_WRENCH (the
                                       foot sphere's contact force/torque is summed
                                       onto its owning CALF link, field 16; a foot
                                       is "in contact" iff |wrench.Fz| > 1 N), NOT
                                       from CONTACT_POINTS (which is positions).
  collision         | -1.0      | PROXY Σ 1[ thigh/calf/base link clips BELOW the
                                       local terrain surface ]. FusedFoot has no
                                       thigh/calf collision geometry (only the 4
                                       foot spheres are collidable), so there is no
                                       true per-body contact force for these links.
                                       We use a faithful KINEMATIC terrain-clipping
                                       proxy: ARTICULATION_LINK_POSE link world z
                                       vs terrain_sample_height_batch at the link
                                       (x,y) (the SINGLE-source heightfield). See
                                       Go2Reward._collision + the env wiring + the
                                       HONESTY caveat there.

PREVIOUSLY DROPPED (now re-enabled) -- the missing-tensor diagnosis was wrong:
  Nuka DOES populate a per-link contact wrench (LINK_CONTACT_WRENCH, field 16;
  ReadoutContactWrench runs for the !is_union FusedFoot Go2 world). The earlier
  code looked at CONTACT_POINTS (foot positions) instead of LINK_CONTACT_WRENCH
  (the per-link summed contact force) -- so feet_air_time is now sourced correctly.
  collision still has NO true per-body force (no thigh/calf geometry in FusedFoot)
  and is the documented kinematic terrain-clip PROXY.

NOT PORTED -- ZERO scale in legged_gym (so legged_gym itself pops these in
``_prepare_reward_function``; they are NOT "drops", they are simply inactive):
  termination(-0.0), orientation(-0.), dof_vel(-0.), base_height(-0.),
  feet_stumble(-0.0), stand_still(-0.).
  NB ``base_height_target=0.25`` IS overridden by go2, but the ``base_height``
  SCALE stays at the inherited ``-0.`` => the term is inactive. (Easy trap.)

The old hand-rolled ``W_ALIVE=0.5`` bonus is REMOVED -- legged_gym go2 has no
alive term.
----------------------------------------------------------------------------

Everything below runs as torch ops on the builder's zero-copy CUDA views (URDF
joint order via ``q_urdf()`` / ``qd_urdf()``); no per-step numpy / host copy.
"""

from __future__ import annotations

import numpy as np
import torch

from . import go2_obs as G

# ---------------------------------------------------------------------------
# Control dt (the legged_gym reward dt): decimation(4) * sim_dt(0.005) = 0.02.
# ---------------------------------------------------------------------------
REWARD_DT = 0.02

# legged_gym go2 reward scales (go2 overrides applied over the inherited base).
# Only the ACTIVE (non-zero, portable-here) terms are listed -- this IS the set
# legged_gym's _prepare_reward_function keeps for go2 minus the two dropped
# contact-force terms. Scales are the RAW config values; REWARD_DT is applied in
# compute_reward (mirroring _prepare_reward_function multiplying every scale by
# self.dt).
SCALES = {
    "tracking_lin_vel": 1.0,
    "tracking_ang_vel": 0.5,
    "lin_vel_z": -2.0,
    "ang_vel_xy": -0.05,
    "torques": -0.0002,
    "dof_acc": -2.5e-7,
    "action_rate": -0.01,
    "dof_pos_limits": -10.0,
    # GAIT-CRITICAL terms (owner: "feet_air_time, collision 一定要加上, 步态很重要").
    # Both legged_gym defaults; both now ACTIVE (no longer dropped) -- feet_air_time
    # is sourced from the TRUE per-foot contact (LINK_CONTACT_WRENCH onto the calf
    # link), collision is the kinematic terrain-clipping proxy (see the module
    # docstring + Go2Reward._feet_air_time / _collision).
    "feet_air_time": 1.0,
    "collision": -1.0,
}

# No reward terms are dropped any more. Kept (empty) so the env's once-at-init
# "DROPPED reward term" logging loop is a no-op and downstream imports stay valid.
# (Historical: feet_air_time + collision were dropped on the WRONG premise that no
# per-link contact tensor existed; LINK_CONTACT_WRENCH provides it for FusedFoot.)
DROPPED: dict[str, str] = {}

# legged_gym _reward_feet_air_time params:
#   contact threshold (foot "loaded" iff |contact_force.z| > 1 N),
#   air_time bias 0.5 s (reward = air_time - 0.5 at the touchdown step),
#   command-gate ||cmd_xy|| > 0.1 (no air-time reward when commanded to stand).
FEET_CONTACT_FORCE_THRESHOLD = 1.0
FEET_AIR_TIME_BIAS = 0.5
FEET_AIR_TIME_CMD_THRESHOLD = 0.1
# collision proxy: a link counts as "clipping" when its world z is more than this
# many metres BELOW the local terrain surface (a small tolerance absorbs FK lag /
# numerical jitter so a grazing link near the surface is not falsely penalized).
COLLISION_PENETRATION_TOL = 0.02

# legged_gym rewards params (go2 overrides: tracking_sigma inherited 0.25;
# soft_dof_pos_limit overridden to 0.9).
TRACKING_SIGMA = 0.25
SOFT_DOF_POS_LIMIT = 0.9

# Go2 URDF joint position limits (rad), URDF order
# [FL,FR,RL,RR x hip,thigh,calf] -- parsed from
# unitree_rl_gym/resources/robots/go2/urdf/go2.urdf <limit lower/upper>.
# Front (FL/FR) thigh and rear (RL/RR) thigh limits differ (matches the
# asymmetric default angles); hips/calves are uniform per type.
DOF_POS_LOWER_URDF = np.array(
    [-1.0472, -1.5708, -2.7227,   # FL hip, thigh, calf
     -1.0472, -1.5708, -2.7227,   # FR
     -1.0472, -0.5236, -2.7227,   # RL
     -1.0472, -0.5236, -2.7227],  # RR
    dtype=np.float32,
)
DOF_POS_UPPER_URDF = np.array(
    [1.0472, 3.4907, -0.8378,   # FL
     1.0472, 3.4907, -0.8378,   # FR
     1.0472, 4.5379, -0.8378,   # RL
     1.0472, 4.5379, -0.8378],  # RR
    dtype=np.float32,
)


def _soft_limits(device, dtype):
    """legged_gym _process_dof_props soft-limit shrink: per joint,
    m=(lo+hi)/2, r=hi-lo, soft=[m - 0.5 r s, m + 0.5 r s] with s=0.9.
    Returns (soft_lower, soft_upper) device tensors (12,) in URDF order."""
    lo = torch.as_tensor(DOF_POS_LOWER_URDF, dtype=dtype, device=device)
    hi = torch.as_tensor(DOF_POS_UPPER_URDF, dtype=dtype, device=device)
    m = (lo + hi) * 0.5
    r = hi - lo
    soft_lo = m - 0.5 * r * SOFT_DOF_POS_LIMIT
    soft_hi = m + 0.5 * r * SOFT_DOF_POS_LIMIT
    return soft_lo, soft_hi


class Go2Reward:
    """On-GPU evaluator of the ported legged_gym Go2 reward (URDF joint order).

    Holds the constant device tensors (PD gains, default angles, soft DOF limits)
    and the per-step bookkeeping the legged_gym terms need (``last_action`` is
    supplied by the env; ``last_dof_vel`` is owned here, updated each call, and
    zeroed externally for autoreset/done envs -- see the env's step override).
    """

    def __init__(self, num_envs: int, device, *, dtype=torch.float32):
        self.num_envs = int(num_envs)
        self.device = device
        self.dtype = dtype
        self.scales = dict(SCALES)
        # Constant device tensors (URDF order).
        self.default_angles = torch.as_tensor(
            G.DEFAULT_ANGLES, dtype=dtype, device=device
        )
        self.soft_lo, self.soft_hi = _soft_limits(device, dtype)
        self.kp = float(G.KP)
        self.kd = float(G.KD)
        # last_dof_vel for the dof_acc finite difference (URDF order).
        self.last_dof_vel = torch.zeros(
            self.num_envs, G.GO2_ACTION_DIM, dtype=dtype, device=device
        )
        # feet_air_time bookkeeping (legged_gym): per-FOOT (4) cumulative time off
        # the ground (s) + the prior step's per-foot contact bool (the
        # contact_filt OR-with-last-step debounce). 4 = FL/FR/RL/RR feet. Both are
        # zeroed externally for done envs in the env's autoreset (mirroring
        # legged_gym reset_idx zeroing feet_air_time / last_contacts), exactly like
        # last_dof_vel.
        self.feet_air_time = torch.zeros(
            self.num_envs, 4, dtype=dtype, device=device
        )
        self.last_contacts = torch.zeros(
            self.num_envs, 4, dtype=torch.bool, device=device
        )

    # -- term helpers (each mirrors a legged_gym _reward_*) ------------------
    @staticmethod
    def _tracking_lin_vel(cmd, lin_vel):
        err = (cmd[:, :2] - lin_vel[:, :2]).pow(2).sum(dim=1)
        return torch.exp(-err / TRACKING_SIGMA)

    @staticmethod
    def _tracking_ang_vel(cmd, ang_vel):
        err = (cmd[:, 2] - ang_vel[:, 2]).pow(2)
        return torch.exp(-err / TRACKING_SIGMA)

    @staticmethod
    def _lin_vel_z(lin_vel):
        return lin_vel[:, 2].pow(2)

    @staticmethod
    def _ang_vel_xy(ang_vel):
        return ang_vel[:, :2].pow(2).sum(dim=1)

    def _torques(self, q_urdf, qd_urdf, target_q_urdf):
        # APPROX: legged_gym penalizes the engine's applied torque; Nuka does not
        # expose it, so reconstruct the PD law the engine itself applies
        # (tau = Kp(target-q) - Kd*qd; ignores the force-limit clamp / per-substep
        # resolution -- a small approximation on a tiny-weight regularizer).
        tau = self.kp * (target_q_urdf - q_urdf) - self.kd * qd_urdf
        return tau.pow(2).sum(dim=1)

    def _dof_acc(self, qd_urdf):
        return ((self.last_dof_vel - qd_urdf) / REWARD_DT).pow(2).sum(dim=1)

    @staticmethod
    def _action_rate(action, last_action):
        return (last_action - action).pow(2).sum(dim=1)

    def _dof_pos_limits(self, q_urdf):
        out = -(q_urdf - self.soft_lo).clamp(max=0.0)        # below lower
        out = out + (q_urdf - self.soft_hi).clamp(min=0.0)   # above upper
        return out.sum(dim=1)

    def _feet_air_time(self, foot_contact_fz, cmd):
        """legged_gym _reward_feet_air_time, ported VERBATIM.

        ``foot_contact_fz`` : (N,4) the four feet's CONTACT FORCE z (world) read
        from LINK_CONTACT_WRENCH at each foot's owning CALF link (field 16). A
        foot is "in contact" iff ``|Fz| > 1 N`` (legged_gym uses contact_forces.z
        > 1). ``cmd`` : (N,3) velocity command (the air-time reward is gated off
        when ||cmd_xy|| <= 0.1, i.e. when commanded to stand still).

        Reward (per env): Σ_feet (air_time - 0.5) at the FIRST contact step after a
        flight phase, so a clean long swing -> touchdown is rewarded and a foot that
        never lifts (drag) earns nothing. Mutates ``self.feet_air_time`` /
        ``self.last_contacts`` in place (the legged_gym per-step state machine);
        the env zeroes both for done envs in autoreset.
        """
        contact = foot_contact_fz.abs() > FEET_CONTACT_FORCE_THRESHOLD  # (N,4) bool
        # contact_filt: OR with last step (debounce a 1-frame contact dropout --
        # legged_gym does this because PhysX can miss a contact for one substep).
        contact_filt = contact | self.last_contacts
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.0) & contact_filt
        self.feet_air_time = self.feet_air_time + REWARD_DT
        rew = ((self.feet_air_time - FEET_AIR_TIME_BIAS) * first_contact.to(self.dtype)
               ).sum(dim=1)
        # No air-time reward when commanded to stand (||cmd_xy|| <= 0.1).
        cmd_moving = (cmd[:, :2].norm(dim=1) > FEET_AIR_TIME_CMD_THRESHOLD).to(self.dtype)
        rew = rew * cmd_moving
        # Reset the air-time accumulator for feet currently in (filtered) contact.
        self.feet_air_time = self.feet_air_time * (~contact_filt).to(self.dtype)
        return rew

    @staticmethod
    def _collision(link_world_z, local_surface_z):
        """Kinematic terrain-CLIPPING proxy for legged_gym _reward_collision.

        legged_gym penalizes ``Σ 1[ ||contact_force[penalised_body]|| > 0.1 ]`` --
        a COUNT of penalised bodies (thigh/calf [+ base]) in hard contact. The
        FusedFoot Go2 model has collision geometry ONLY on the 4 foot spheres, so
        there is NO true per-body contact force for the thighs/calves/base. We
        substitute a faithful KINEMATIC proxy that matches the term's INTENT
        (discourage limbs hitting / clipping the terrain): a penalised link counts
        as "in collision" when its world z is below the LOCAL terrain surface at
        its (x,y) -- i.e. it is clipping INTO a step.

        ``link_world_z`` / ``local_surface_z`` : (N, K) world z and local terrain
        surface z for the K penalised links. Returns (N,) the count of links that
        penetrate the surface by more than COLLISION_PENETRATION_TOL.

        >>> HONESTY CAVEAT <<<  This is a TERRAIN-CLIP proxy, not a contact-force
        readout. It cannot see a limb hitting ANOTHER limb (self-collision) or a
        limb resting exactly on a surface; it only fires on geometric penetration
        of the heightfield. A TRUE collision penalty would require thigh/calf
        collision geometry + per-body contact force in the engine (a future engine
        change -- the FusedFoot pipeline is foot-spheres-only by construction).
        """
        penetration = local_surface_z - link_world_z  # >0 when link is BELOW surface
        clipping = penetration > COLLISION_PENETRATION_TOL
        return clipping.to(torch.float32).sum(dim=1)

    # -- the full reward (legged_gym compute_reward semantics) --------------
    def compute(
        self,
        *,
        cmd,
        lin_vel,
        ang_vel,
        q_urdf,
        qd_urdf,
        target_q_urdf,
        action,
        last_action,
        foot_contact_fz=None,
        collision_count=None,
        return_terms: bool = False,
    ):
        """Total reward (num_envs,) = clip(Σ scale·term·REWARD_DT, min=0).

        All inputs are device tensors in URDF joint order. ``action`` is THIS
        step's clipped action (== env.last_action); ``last_action`` is the prior
        step's clipped action (the action-rate delta). ``target_q_urdf`` is the
        PD target the env wrote this step (default + 0.25*action). Updates
        ``last_dof_vel`` to the current ``qd_urdf`` at the end (legged_gym does
        ``last_dof_vel[:] = dof_vel`` at the tail of post_physics_step).

        ``foot_contact_fz`` : (N,4) the four feet's contact-force z (world) read by
        the env from LINK_CONTACT_WRENCH at each foot's owning calf link -- drives
        feet_air_time. ``collision_count`` : (N,) the env's precomputed count of
        penalised links clipping the terrain surface -- drives the collision proxy.
        Both default None (e.g. the obs/oracle sanity probes that call compute()
        without an engine world); the matching term then contributes 0 and its
        per-step state machine is not advanced.

        With ``return_terms`` also returns a dict of the SIGNED, dt-scaled
        per-term contributions (for the sanity probe) -- pre-clamp.
        """
        terms = {
            "tracking_lin_vel": self._tracking_lin_vel(cmd, lin_vel),
            "tracking_ang_vel": self._tracking_ang_vel(cmd, ang_vel),
            "lin_vel_z": self._lin_vel_z(lin_vel),
            "ang_vel_xy": self._ang_vel_xy(ang_vel),
            "torques": self._torques(q_urdf, qd_urdf, target_q_urdf),
            "dof_acc": self._dof_acc(qd_urdf),
            "action_rate": self._action_rate(action, last_action),
            "dof_pos_limits": self._dof_pos_limits(q_urdf),
        }
        # GAIT-CRITICAL terms (sourced by the env from the engine contact/pose
        # fields). feet_air_time advances a per-foot state machine, so only run it
        # when the env supplies the true per-foot contact (else its state must NOT
        # tick). collision is a pure per-step proxy count.
        if foot_contact_fz is not None:
            terms["feet_air_time"] = self._feet_air_time(foot_contact_fz, cmd)
        if collision_count is not None:
            terms["collision"] = collision_count.to(self.dtype)
        total = torch.zeros(self.num_envs, dtype=self.dtype, device=self.device)
        contrib = {} if return_terms else None
        for name, term in terms.items():
            c = term * (self.scales[name] * REWARD_DT)
            total = total + c
            if return_terms:
                contrib[name] = c
        # only_positive_rewards = True (inherited): floor the total at 0.
        total = total.clamp(min=0.0)
        # legged_gym updates last_dof_vel at the tail of every post_physics_step.
        self.last_dof_vel = qd_urdf.clone()
        if return_terms:
            return total, contrib
        return total
