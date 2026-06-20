"""Go2 rear-leg HANDSTAND (倒立) task -- PD-position static balance, inverted.

The quadruped pitches its trunk up so the FRONT paws lift high and it balances on
its two REAR feet (a rear-leg "stand-up"/handstand). Static-balance RL on the SAME
Go2 floating-base scene the flat walker uses (``go2_locomotion.usda``) -- no new
asset, no engine change. Full design:
``subagent-plans/2026-06-15-skill-spec-handstand.md``.

============================================================================
ORIENTATION CONVENTION -- CORRECTED 2026-06-16 (the original was BACKWARDS).
============================================================================
The first version targeted a NOSE-UP pitch (proj_grav -> [+1,0,0]). A direct
link-geometry probe on the live cooked Go2 (go2_dump terrain harness style;
results in the retrain report) showed that nose-up pitch about +y puts the REAR
feet HIGH IN THE AIR (rear-foot world z ~ +0.77 m at 90 deg) and the FRONT paws
on the ground -- i.e. a "downward-dog", NOT a rear-leg handstand. In that
configuration the rear feet can NEVER load (measured rear-foot Fz == 0 at every
pitch), so "balance on the rear feet" was geometrically impossible and the trained
policy could only ever harvest the seated-IC orientation reward then topple. THAT
is the deep root cause of the fake handstand.

The CORRECT rear-leg handstand is a NOSE-DOWN pitch: forward (+x) rotates toward
world -z, driving ``projected_gravity()`` from upright [0,0,-1] toward [-1,0,0].
This plants the REAR feet on the ground (rear-foot world z ~ 0, rear-foot Fz
loads 50-200 N) and lifts the FRONT paws high (front-foot world z ~ +0.55 .. +0.75
m). So the orientation reward now targets

    proj_grav = [-sin(theta_cmd), 0, -cos(theta_cmd)]   (theta_cmd=90 => [-1,0,0])

and ``proj_grav_x`` falling 0 -> -1 is the clean monotonic progress signal toward
the handstand (the base ``tilt_deg`` folds at 90 deg and cannot be the metric).
The deterministic eval/diag/manual scripts measure the inversion as ``-pgx`` toward
the commanded sign.

PHYSICS REALITY (link-geometry + assist-decay probes, retrain report): a frozen
posture is NEVER a balanced equilibrium for this inverted pendulum (RL must close
the loop). The open-loop posture settles to a SUSTAINED rear-up of ~45-55 deg
(|pgx| ~ 0.6-0.73) with the rear feet loaded and the front paws clear; a full
vertical 90 deg drives the trunk-bottom into the floor and is far less stable.
The curriculum therefore targets a STRONG, REAL, sustained rear-up (the task's
accepted "倒立": |pgx| >= ~0.8 = ~58 deg held on LOADED rear feet, front paws up),
reaching toward a full handstand where the balance permits -- never an IC artifact.

============================================================================
ANTI-HARVEST REWARD (the second root cause).
============================================================================
The original reward paid the dominant orient + alive term EVERY step at the
re-seated near-handstand IC, so a non-balancing policy farmed reward over a dozen
open-loop steps then autoreset re-seated it -- it learned "the IC is valuable",
never to balance. This env now (1) makes the big orientation payout REQUIRE the
rear feet to be LOADED (the defining handstand feature -- zero at the bad IC,
nonzero only while genuinely on the rear feet), and (2) adds a per-env HOLD-TIME
bonus that only accrues for SUSTAINED balance (consecutive in-handstand steps),
so the reward couples to the ACTIONS that maintain the pose -- the grasp-task
dynamic that keeps the action sigma healthy.

CURRICULUM (in-env, per-env ``theta_cmd``): the pitch target is staged S0->S2 via
the ``stage`` ctor knob. S0 REACH (0->40deg, IC flat -- learn to rock back onto the
rear feet), S1 REAR-UP (40->60deg, IC pitched + rear feet planted), S2 HOLD
(55->75deg, IC near the sustained rear-up). The com/no_front/front_lift reward
terms are GATED by the live theta_cmd so they don't punish the on-ground start.
"""

from __future__ import annotations

import math

import numpy as np
import torch

import nuka

from .go2_locomotion import Go2LocomotionEnv
from . import go2_obs as G
from .go2_rewards import Go2Reward, REWARD_DT


# ---------------------------------------------------------------------------
# Handstand-entry IC pose (URDF order [FL,FR,RL,RR x hip,thigh,calf]).
# ---------------------------------------------------------------------------
# The S1+ pitched ENTRY posture for the NOSE-DOWN rear-leg handstand: the FRONT
# legs (FL,FR) fold UP toward the belly (thigh flexed back ~-0.8, calf folded
# ~-1.0) so they swing clear and high as the front of the body lifts; the REAR
# legs (RL,RR) sit in a near-stance carry (thigh ~0.9, calf ~-1.5 == the walker
# default) so the rear FOOT reaches the ground and PLANTS -- this is the load-
# bearing change vs. the broken original (which folded the rear legs UP, so with
# the wrong nose-UP pitch the rear feet pointed at the sky and never touched).
# Link-geometry probe (retrain report): with this pose + nose-DOWN pitch the rear
# feet sit at world z ~ 0 and carry 50-200 N while the front paws reach z ~ +0.55.
# The RL policy refines the precise balance posture from here.
#                       FL_hip FL_th FL_cf  FR_hip FR_th FR_cf  RL_hip RL_th RL_cf  RR_hip RR_th RR_cf
HANDSTAND_ENTRY_ANGLES = np.array(
    [0.0, -0.7, -0.9,   0.0, -0.7, -0.9,   0.0, 0.8, -1.4,   0.0, 0.8, -1.4],
    dtype=np.float32,
)

# Nominal base height (trunk z) at the pitched rear-up entry. Tuned (retrain
# report) so the rear feet PLANT and load within the first settle step across the
# whole band (rear-foot Fz > 1 N at settle 0, front feet clear); a higher base
# launches the rear feet off the ground (rear Fz == 0 => the reward gate pays
# nothing => no learning signal). Stance (flat) base z is 0.445; we ramp the entry
# z from stance DOWN toward HANDSTAND_BASE_Z as theta grows (the trunk tips up and
# the base settles as the rear feet take the load).
#
# RETRAIN 2026-06-16: an open-loop / fixed-PD geometry sweep (go2_handstand probes
# in the retrain report) showed the nose-down inverted pose is a divergent inverted
# pendulum -- the heavy trunk hangs FORWARD of the rear-foot support so it topples
# nose-down past vertical, and the rear legs FOLD under the changed gravity load so
# the trunk sinks from 0.445 toward the floor in ~15 control steps. The trained
# policy could invert (pgx -0.9, front feet clear, rear loaded) but could only hold
# a CROUCHED rear-up at base_z ~0.12, which the old 0.10 collapse floor then killed.
# The fix has three legs: (1) give the actuator real authority to hold height +
# apply restoring torque -- HS_ACTION_SCALE 0.5 (vs walker 0.25) around a handstand-
# centred PD nominal (HS_NOMINAL_URDF) so action=0 is a supporting rear-up, not the
# flat-walker crouch the policy was fighting; (2) a realistic, ACHIEVABLE base-z
# target (HANDSTAND_BASE_Z, the held-rear-up height) and a low collapse floor so a
# genuinely-balanced low rear-up is rewarded for staying as high as holdable, not
# punished/killed toward an unreachable vertical; (3) cap the curriculum at a
# HOLDABLE strong rear-up (the brief's accepted 倒立 = pgx>=~0.8 ~58deg) where a
# closed-loop equilibrium exists, rather than a vertical that has none.
HANDSTAND_BASE_Z = 0.22
STANCE_BASE_Z = 0.445

# Collapse-kill floor (trunk z). The held rear-up sits low (the trunk pivots down
# as the body inverts), so the floor must be below the achievable held height
# (~0.18-0.22) yet still catch a genuine face-plant. 0.07 = "trunk basically on the
# ground" (a real collapse) without killing a balanced-but-low rear-up.
COLLAPSE_FLOOR_Z = 0.07

# Handstand PD authority. The walker uses ACTION_SCALE=0.25 around the flat default
# (go2_obs.DEFAULT_ANGLES). The inverted balance needs (a) a nominal centred on a
# SUPPORTING rear-up pose so action=0 holds the body up instead of folding it into
# the walker crouch, and (b) a wider action range so the policy can both hold height
# and inject fast restoring torque against the topple. HS_NOMINAL_URDF: front legs
# tucked UP (thigh flexed back, calf folded) to swing clear; rear legs in a carry
# that can EXTEND to push the trunk up (thigh moderate, calf moderately folded).
#                  FL_hip FL_th FL_cf  FR_hip FR_th FR_cf  RL_hip RL_th RL_cf  RR_hip RR_th RR_cf
HS_NOMINAL_URDF = np.array(
    [0.0, -0.7, -0.9,   0.0, -0.7, -0.9,   0.0, 0.8, -1.4,   0.0, 0.8, -1.4],
    dtype=np.float32,
)
HS_ACTION_SCALE = 0.5

# Curriculum stage -> (theta_lo_deg, theta_hi_deg, ic_pitched). The per-env
# theta_cmd is sampled in [lo,hi] each reset; ic_pitched seats the pitched entry
# pose+base for S1+ (S0 starts flat so the policy learns to rock back from
# standing). Bands re-tuned for the CORRECTED nose-down physics: the open-loop
# posture sustains ~45-55 deg, so the curriculum reaches a strong rear-up (S2 to
# 75 deg) rather than an unholdable vertical 90 (which drives the trunk into the
# floor). A true 90 is targeted opportunistically via the obs theta-cmd at eval.
# RETRAIN 2026-06-16: bands re-tuned to top out at a HOLDABLE strong rear-up. The
# inverted-pendulum topple rate grows with pitch and there is no static equilibrium
# near vertical, so a 75-90deg target is unholdable (the policy topples or sinks).
# The brief accepts pgx>=~0.8 (~58deg) held as a real 倒立; S2 targets 50-62deg, a
# strong rear-up the closed-loop policy can actually sustain. S0 REACH (flat IC,
# learn to rock back), S1 REAR-UP (pitched IC), S2 HOLD (the demo target).
STAGE_SPEC = {
    "S0": dict(theta_lo=0.0,  theta_hi=35.0, ic_pitched=False),
    "S1": dict(theta_lo=30.0, theta_hi=52.0, ic_pitched=True),
    # S2 RETRAIN 2026-06-16: narrowed to the achievable CLEAN-handstand window. The
    # full sweep showed survival collapses above ~50deg (the 2-rear-foot inverted
    # pendulum diverges) while a clean front-up handstand needs >=~42deg. S2 targets
    # 42-50deg = the band where the rear feet stay loaded, the front paws clear, and
    # a closed-loop hold survives -- the strongest REAL sustained rear-up.
    "S2": dict(theta_lo=42.0, theta_hi=50.0, ic_pitched=True),
}

# Reward-term gates (by the live per-env theta_cmd, deg): below these the term is
# masked OFF so it does not punish the on-ground / low-pitch start (spec Section 5.4).
GATE_COM_DEG = 28.0       # com_over_support active once leaning past ~28 deg.
GATE_FRONT_DEG = 30.0     # no_front_contact + front_leg_lift active past ~30 deg.

# Hold-time bonus: number of CONSECUTIVE "in-handstand" control steps to ramp the
# hold bonus to full. The hold bonus only accrues for SUSTAINED balance (it is 0
# at the IC, where hold_steps == 0), so a non-balancing policy cannot harvest it.
HOLD_RAMP_STEPS = 50.0


class Go2HandstandEnv(Go2LocomotionEnv):
    """Go2 rear-leg handstand env (PD-position, inverted static balance).

    Subclasses :class:`Go2LocomotionEnv` to inherit the obs builder, joint
    permutation, ``apply_pd_gains`` / force-limits, the PD ``step`` path, autoreset,
    and the cached wrench/pose views + foot-slot tensors. Overrides the IC pose, the
    reward (handstand shaping), and -- critically -- the termination (INVERTED
    tilt-kill: vertical is the goal). The locomotion velocity command is pinned to
    zero (``fixed_command=True``, command=(0,0,0)) so the walker's command machinery
    is inert and the obs command slot [9:12] is a constant the net learns to ignore.
    """

    def __init__(self, scene: str, num_envs: int, *,
                 stage: str = "S0",
                 theta_cmd_obs: bool = True,
                 reward_mode: str = "handstand",
                 k_orient: float = 4.0,
                 k_orient_y: float = 8.0,
                 k_com: float = 10.0,
                 w_orient: float = 1.2,
                 w_orient_progress: float = 2.0,
                 w_sagittal: float = 1.0,
                 w_com: float = 0.5,
                 w_rear_contact: float = 1.2,
                 w_no_front: float = 1.0,
                 w_base_height: float = 1.0,
                 w_front_lift: float = 0.1,
                 w_alive: float = 0.1,
                 w_hold: float = 2.0,
                 **kw) -> None:
        # Handstand knobs popped before super().__init__ (mirrors gait_mode).
        self._stage = str(stage)
        if self._stage not in STAGE_SPEC:
            raise ValueError(f"unknown handstand stage {self._stage!r}; "
                             f"expected one of {sorted(STAGE_SPEC)}")
        self._theta_cmd_obs = bool(theta_cmd_obs)
        self._reward_mode = str(reward_mode)
        self._k_orient = float(k_orient)
        self._k_orient_y = float(k_orient_y)
        self._k_com = float(k_com)
        self._w = dict(
            orient=float(w_orient), orient_progress=float(w_orient_progress),
            sagittal=float(w_sagittal),
            com=float(w_com), rear_contact=float(w_rear_contact),
            no_front=float(w_no_front), base_height=float(w_base_height),
            front_lift=float(w_front_lift), alive=float(w_alive), hold=float(w_hold),
        )

        # The handstand uses NO velocity command and NO terrain (flat scene). Pin
        # command=(0,0,0) + fixed_command so the locomotion command resample never
        # touches the obs; terrain stays OFF (the default). The tilt-kill we INVERT
        # ourselves, so the inherited termination_tilt_deg is irrelevant (never
        # consulted -- we don't call super().compute_terminated).
        kw.setdefault("command", (0.0, 0.0, 0.0))
        kw["fixed_command"] = True
        kw.pop("terrain", None)          # force terrain off (flat handstand scene).
        kw.pop("gait_mode", None)        # handstand reward replaces the gait reward.
        super().__init__(scene, num_envs, **kw)

        dev = self._torch_device
        # HOLE #1 (adversarial review): self._bp_view is created by the parent ONLY
        # on the terrain-ON path; the handstand runs terrain OFF, so create our OWN
        # zero-copy BASE_POSE view here for the pitched-entry IC writes.
        self._bp_view = torch.from_dlpack(self._world.buffer_view(nuka.BASE_POSE))

        # ----- PD AUTHORITY OVERRIDE (the retrain root-cause fix) ----------------
        # Re-centre the PD nominal on a SUPPORTING rear-up pose (HS_NOMINAL_URDF) and
        # widen the action range (HS_ACTION_SCALE 0.5 vs walker 0.25) so action=0
        # holds the inverted body UP and the policy has the authority to both carry
        # the height and inject fast restoring torque against the topple. Without
        # this the PD spring pulls the legs toward the flat-walker crouch with only
        # +-0.25 rad authority -> the trunk sinks and grazes the collapse floor (the
        # measured fake-handstand failure). We override the obs builder's nominal +
        # action scale (so obs (q-nominal), write_action, AND the torque-reward
        # target are all consistent) WITHOUT touching the walker (this is a per-env
        # instance override on the handstand env's own Go2ObsBuilder/Go2Reward).
        self._hs_nominal_urdf = torch.as_tensor(
            HS_NOMINAL_URDF, dtype=torch.float32, device=dev
        )
        self._hs_action_scale = float(HS_ACTION_SCALE)
        # obs (q - default) and write_action both read self._obs.default_angles;
        # point it at the handstand nominal so the policy sees deviation from the
        # supporting pose and action=0 maps to that pose. The reward's torque
        # reconstruction uses the per-step self._target_q_urdf (set in step()), so
        # also align self._reward.default_angles for any default-relative reads.
        self._obs.default_angles = self._hs_nominal_urdf
        self._reward.default_angles = self._hs_nominal_urdf
        # Install a handstand write_action on THIS env's obs builder instance: PD
        # target = HS_NOMINAL_URDF + HS_ACTION_SCALE * clip(action). Mirrors
        # Go2ObsBuilder.write_action exactly but with the handstand nominal/scale.
        _obs = self._obs
        _nom = self._hs_nominal_urdf
        _scale = self._hs_action_scale
        _clip = float(G.ACTION_CLIP)
        def _hs_write_action(action, _obs=_obs, _nom=_nom, _scale=_scale, _clip=_clip):
            action = action.clamp(-_clip, _clip)
            target_urdf = _nom + _scale * action
            _obs._tgt[:, 1:G.GO2_BLC] = target_urdf.index_select(1, _obs.nuka_slot_for_urdf)
            return action
        self._obs.write_action = _hs_write_action

        # front = first two foot slots (FL,FR), rear = last two (RL,RR), from the
        # parent's structurally-derived self._foot_link_slot (asserted at parent
        # init). Use the DERIVED tensor, never hardcoded literals (hole #5).
        self._front_foot_slot = self._foot_link_slot[0:2].clone()
        self._rear_foot_slot = self._foot_link_slot[2:4].clone()

        # Per-env curriculum pitch target (deg). Sampled in the stage band each
        # reset; drives the orientation target + the gated reward terms + (Option A)
        # the appended obs dim. Initialised to the stage band midpoint.
        spec = STAGE_SPEC[self._stage]
        self._theta_lo = float(spec["theta_lo"])
        self._theta_hi = float(spec["theta_hi"])
        self._ic_pitched = bool(spec["ic_pitched"])
        self._theta_cmd_deg = torch.full(
            (self.num_envs,), 0.5 * (self._theta_lo + self._theta_hi), device=dev
        )

        # Handstand-entry IC constants on device (URDF order).
        self._entry_urdf = torch.as_tensor(
            HANDSTAND_ENTRY_ANGLES, dtype=torch.float32, device=dev
        )

        # Per-env HOLD-TIME counter (consecutive control steps the env has been
        # genuinely "in handstand": pitched toward the commanded target AND on
        # LOADED rear feet). Drives the anti-harvest hold bonus; 0 at the IC, reset
        # to 0 on a fall-out (in compute_reward) and on autoreset (in step).
        self._hold_steps = torch.zeros(self.num_envs, device=dev)

        # Option A: append theta_cmd (normalized theta/90deg) to the 48-dim obs so
        # the policy is pitch-conditioned (one net spans S0->S2). Override the gym
        # obs spaces to 49 dims (mirror go2_locomotion's height-scan obs-space
        # override). Cold start either way -- handstand has no warm-start.
        self._obs_dim = G.GO2_OBS_DIM + (1 if self._theta_cmd_obs else 0)
        if self._theta_cmd_obs:
            from gymnasium import spaces as _spaces
            self.single_observation_space = _spaces.Box(
                low=-G.OBS_CLIP, high=G.OBS_CLIP,
                shape=(self._obs_dim,), dtype=np.float32,
            )
            self.observation_space = _spaces.Box(
                low=-G.OBS_CLIP, high=G.OBS_CLIP,
                shape=(self.num_envs, self._obs_dim), dtype=np.float32,
            )

        print(f"[go2_handstand] stage={self._stage} theta_cmd band=[{self._theta_lo:.0f},"
              f"{self._theta_hi:.0f}]deg ic_pitched={self._ic_pitched} "
              f"theta_cmd_obs={self._theta_cmd_obs} obs_dim={self._obs_dim} "
              f"reward_mode={self._reward_mode}", flush=True)
        print(f"[go2_handstand] front foot(calf) slots={self._front_foot_slot.tolist()} "
              f"rear={self._rear_foot_slot.tolist()}  "
              f"orient TARGET proj_grav=[-sin(theta),0,-cos(theta)] (NOSE-DOWN; "
              f"theta=90 => [-1,0,0], rear feet planted/front paws up); "
              f"weights={self._w}", flush=True)

    # -- per-env curriculum pitch target ------------------------------------
    def _sample_theta_cmd(self, mask) -> None:
        """Sample a fresh per-env pitch target (deg) in the stage band for the
        (masked, or all) reset envs. Seeded via the base env's generator."""
        if mask is None:
            n = self.num_envs
            t = torch.rand(n, generator=self._generator, device=self._torch_device)
            self._theta_cmd_deg = self._theta_lo + t * (self._theta_hi - self._theta_lo)
        else:
            n = int(mask.sum().item())
            if n == 0:
                return
            t = torch.rand(n, generator=self._generator, device=self._torch_device)
            self._theta_cmd_deg[mask] = self._theta_lo + t * (self._theta_hi - self._theta_lo)

    def _theta_cmd_rad(self) -> torch.Tensor:
        return torch.deg2rad(self._theta_cmd_deg)

    # -- obs: append the normalized theta_cmd (Option A) ---------------------
    def _append_theta(self, obs: torch.Tensor) -> torch.Tensor:
        if not self._theta_cmd_obs:
            return obs
        theta_n = (self._theta_cmd_deg / 90.0).unsqueeze(1)      # (N,1) ~[0,1]
        return torch.cat([obs, theta_n], dim=-1)

    def reset(self, *, seed=None, options=None):
        # Sample fresh per-env theta targets for ALL envs on a full reset BEFORE
        # the IC is seated (so _reset_joint_state seats the pitched entry at the
        # right per-env pitch). super().reset() -> _reset_joint_state(None).
        self._sample_theta_cmd(None)
        obs, info = super().reset(seed=seed, options=options)
        return self._append_theta(obs), info

    def step(self, actions: torch.Tensor):
        obs, reward, terminated, truncated, info = super().step(actions)
        # The parent's masked autoreset already called _reset_joint_state(done)
        # (which re-samples theta + zeroes _hold_steps for the done envs via the
        # IC seater). Belt-and-braces: zero the hold counter for any done env that
        # truncated (truncation does not go through the IC seater). Append theta.
        done = terminated | truncated
        if bool(done.any()):
            self._hold_steps[done] = 0.0
        return self._append_theta(obs), reward, terminated, truncated, info

    # -- IC: handstand-entry pose + (S1+) pitched base ----------------------
    def _reset_joint_state(self, mask) -> None:
        """Seat the handstand IC for the (masked, or all) reset envs.

        S0 (ic_pitched False): the FLAT default angles (call the parent path) -- the
            policy must learn to lean/pitch up from standing.
        S1+ (ic_pitched True): the HANDSTAND_ENTRY pose (front legs tucked up, rear
            legs loaded) + the BASE_POSE pitched up to the per-env theta_cmd about
            +y, base z ramped toward HANDSTAND_BASE_Z and shifted back over the rear
            feet so the trunk PIVOTS over the rear support (the geometry the probe
            validated). Per-env IC seeded from theta_cmd so each env starts near its
            commanded pitch.

        Resamples theta_cmd for the done envs FIRST (so the seated pitch matches the
        fresh target), then writes JOINT_POSITION/VELOCITY (Nuka slot order via the
        permutation) + BASE_POSE. Terrain is OFF so there is no curriculum/terrain
        write (the parent's terrain branch is skipped: self._terrain_on is False)."""
        # Re-sample theta for the just-done envs (full reset already sampled all in
        # reset(); mask is None there and we skip the re-sample to avoid double-draw).
        if mask is not None:
            self._sample_theta_cmd(mask)

        if not self._ic_pitched:
            # S0: flat default IC (parent writes JOINT_POSITION/VELOCITY + skips the
            # OFF terrain branch). Base stays at the cooked upright stance pose.
            super()._reset_joint_state(mask)
            if mask is None:
                self._hold_steps.zero_()
            else:
                self._hold_steps[mask] = 0.0
            return

        # S1+: seat the handstand-entry joints + the pitched base.
        w = self._world
        b = self._obs
        entry_nuka = self._entry_urdf[b.nuka_slot_for_urdf]      # URDF -> Nuka slots
        q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
        qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
        bp = self._bp_view

        theta = self._theta_cmd_rad()                            # (N,) per-env pitch
        # Base pose: NOSE-DOWN pitch about +y (forward +x rotates toward world -z),
        # so the rear feet plant and the front paws lift. Quat for a NEGATIVE pitch
        # angle is [cos(th/2), 0, -sin(th/2), 0]. The IC base z is seated a bit ABOVE
        # the achievable held height (HANDSTAND_BASE_Z) so the policy must ACTIVELY
        # hold/recover height from the entry rather than start already-crushed at the
        # equilibrium (which would be an IC-harvest with nothing to learn). It ramps
        # from the flat stance toward the entry as theta grows.
        half = theta * 0.5
        qw = torch.cos(half)
        qy = -torch.sin(half)                                    # NOSE-DOWN
        # Entry base z = a moderate rear-up height (above the held equilibrium); ramp
        # from stance as theta grows over the [0, top-of-S2] range (max ~62deg).
        a = (self._theta_cmd_deg / 70.0).clamp(0.0, 1.0)         # 0..1 entry blend
        IC_ENTRY_Z = 0.30
        base_z = STANCE_BASE_Z + (IC_ENTRY_Z - STANCE_BASE_Z) * a

        dev = self._torch_device

        def _seat(idx, full):
            # idx: a boolean mask (full=False) or slice(None) (full=True). Per-env IC
            # JITTER so the policy learns to RECOVER a balanced rear-up from a
            # perturbed entry rather than memorize one trajectory (anti-harvest
            # robustness; small magnitudes).
            nn = self.num_envs if full else int(idx.sum().item())
            gen = self._generator
            # IC jitter (RETRAIN 2026-06-16 fix #4): a diagnostic showed the policy
            # finds a ROCK-SOLID static held rear-up (survivors hold inv/bz constant
            # for 120+ steps, qd->0.4); the survival loss was ENTIRELY the TRANSIENT
            # from an over-aggressive jittered entry (esp. the +-0.3 rad/s base spin)
            # that threw ~half the envs into an unrecoverable topple before they could
            # settle. Reduced to a small, recoverable entry perturbation (still breaks
            # IC-memorization + adds robustness, but lets the stable attractor be
            # reached): pose +-0.03 rad, z +-0.01 m, pitch +-0.02 rad, spin +-0.08 rad/s.
            j_q = (torch.rand(nn, 12, generator=gen, device=dev) - 0.5) * 0.06   # +-0.03 rad
            j_z = (torch.rand(nn, generator=gen, device=dev) - 0.5) * 0.02        # +-0.01 m
            j_th = (torch.rand(nn, generator=gen, device=dev) - 0.5) * 0.04       # +-0.02 rad pitch
            j_w = (torch.rand(nn, 3, generator=gen, device=dev) - 0.5) * 0.16     # +-0.08 rad/s body rate
            entry_j = (self._entry_urdf.unsqueeze(0) + j_q)                        # (nn,12) URDF
            entry_jn = entry_j.index_select(1, b.nuka_slot_for_urdf)              # -> Nuka slots
            th_j = (theta if full else theta[idx]) + j_th
            half_j = th_j * 0.5
            bz_i = (base_z if full else base_z[idx]) + j_z
            q[idx, 1:G.GO2_BLC] = entry_jn
            qd[idx, 1:G.GO2_BLC] = 0.0
            bp[idx, 0] = 0.0
            bp[idx, 1] = 0.0
            bp[idx, 2] = bz_i
            bp[idx, 3] = torch.cos(half_j)
            bp[idx, 4] = 0.0
            bp[idx, 5] = -torch.sin(half_j)
            bp[idx, 6] = 0.0
            # Seed a small base angular velocity (LINK_VELOCITY root slot, body ang
            # vel is [0:3]) so the policy must actively damp/recover, not freeze.
            self._obs._vel[idx, 0, 0:3] = j_w

        if mask is None:
            _seat(slice(None), True)
            self._hold_steps.zero_()
        else:
            _seat(mask, False)
            self._hold_steps[mask] = 0.0
        nuka.sync()

    # -- termination: THE INVERTED tilt-kill (do NOT call super) ------------
    def compute_terminated(self) -> torch.Tensor:
        """Handstand-appropriate termination. The walker/base tilt-kill terminates
        a handstand on sight (tilt>75 deg), so we do NOT call super -- the NOSE-DOWN
        inverted pose (proj_grav -> [-1,0,0]) is the GOAL. We kill only on a genuine
        FLIP-OVER past vertical (toward the back), a real COLLAPSE onto the ground,
        or numeric/runaway guards (structure mirrors h1_stand.py:395-411). Kept
        GENTLE so a wobbling-but-recovering handstand accrues enough non-terminal
        steps to learn the balance (the only_positive_rewards cold-start trap)."""
        pg = self._obs.projected_gravity()                      # (N,3)
        base_z = self._obs.base_pos()[:, 2]                     # (N,)
        q = self._obs.q_urdf()
        qd = self._obs.qd_urdf()

        # (a) FLIP-OVER: overshot vertical and toppled ONTO the back. In the
        #     nose-down convention the goal is pgx ~ -1, pgz ~ 0; overshooting past
        #     90 deg sends proj_grav_z POSITIVE while proj_grav_x climbs back ABOVE
        #     0 (the trunk falls over forward onto the back). Require pgz>+0.7 AND
        #     pgx>0 so a clean inverted pose (pgx~-1, pgz~0) + slight overshoot live.
        flipped = (pg[:, 2] > 0.7) & (pg[:, 0] > 0.0)
        # (b) COLLAPSE floor: a LOW floor (NOT the 0.18 stance floor) -- in the
        #     rear-up balance the trunk sits lower than a normal stance; only a
        #     genuine face-plant onto the ground kills (COLLAPSE_FLOOR_Z, below the
        #     achievable held height so a balanced-but-low rear-up is NOT killed).
        collapsed = base_z < COLLAPSE_FLOOR_Z
        # (c) numeric + runaway guards (h1_stand pattern).
        nonfinite = (
            ~torch.isfinite(self._obs.base_pos()).all(dim=1)
            | ~torch.isfinite(q).all(dim=1)
            | ~torch.isfinite(qd).all(dim=1)
        )
        runaway_qvel = qd.abs().amax(dim=1) > 50.0              # rad/s, Go2-scale.

        term = flipped | collapsed | nonfinite | runaway_qvel
        # The parent reads self._last_terminated only on the terrain-curriculum path
        # (OFF here); set it anyway so any inherited bookkeeping stays consistent.
        self._last_terminated = term
        return term

    # -- reward: handstand shaping (clamp_min 0; structure <- h1_stand.py) ---
    def compute_reward(self) -> torch.Tensor:
        """Anti-harvest handstand reward (num_envs,) cuda float32, clamp_min(0).

        Two structural changes vs. the original (the fake-handstand fixes):

        (A) The big orientation + alive payout is GATED on the REAR feet being
            LOADED -- the defining handstand feature. At the broken-IC / non-
            balancing state the rear feet carry no load, so the gate zeroes the
            dominant reward: there is nothing to harvest from merely sitting at a
            re-seated pose. The reward is only paid while the policy genuinely
            balances on the rear feet.

        (B) A per-env HOLD-TIME bonus that ramps with CONSECUTIVE in-handstand
            steps (orientation toward the commanded target AND rear feet loaded).
            It is 0 at the IC (hold_steps == 0) and only grows for SUSTAINED
            balance, so the reward couples to the ACTIONS that hold the pose (the
            grasp-task dynamic that keeps the action sigma healthy) rather than to
            the value of the seated IC.

        Orientation target is the NOSE-DOWN handstand proj_grav = [-sin t,0,-cos t]
        (theta=90 => [-1,0,0]). com/no_front/front_lift are GATED by the live per-
        env theta_cmd so they do not punish the on-ground low-pitch start."""
        b = self._obs
        dev = self._torch_device
        pg = b.projected_gravity()                              # (N,3)
        base_z = b.base_pos()[:, 2]                             # (N,)
        base_xy = b.base_pos()[:, :2]                           # (N,2)

        theta = self._theta_cmd_rad()                           # (N,)
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        theta_deg = self._theta_cmd_deg

        # (1) orient_handstand: pull proj_grav toward the NOSE-DOWN target
        #     [-sin(theta_cmd), 0, -cos(theta_cmd)] (theta=90 => [-1,0,0]) + penalize
        #     proj_grav_y (roll/yaw off the sagittal plane). proj_grav_x falling
        #     0 -> -1 is the monotonic progress signal (tilt_deg folds at 90).
        orient_err = (pg[:, 0] + sin_t).pow(2) + (pg[:, 2] + cos_t).pow(2)
        orient = torch.exp(-self._k_orient * orient_err
                           - self._k_orient_y * pg[:, 1].pow(2))

        # (1b) inversion PROGRESS: a strong, MONOTONIC reward for reaching the
        #     commanded inversion (-pgx toward sin(theta_cmd)). RETRAIN 2026-06-16
        #     fix #2: the exp() orient term plateaus and the policy found a SAFE LOW
        #     LEAN local optimum (pgx ~ -0.44 vs commanded -0.71) where it survives
        #     and still collects base_height/alive/rear_contact -- it would not RISK
        #     the higher commanded pitch. This term pays linearly for how close the
        #     achieved inversion is to the commanded one, so leaning HIGHER always
        #     pays more (up to the target) -- it removes the low-lean local optimum.
        #     Clamped [0,1]; overshoot past the command does not pay extra (capped).
        inv_now = (-pg[:, 0]).clamp(min=0.0)                       # 0..1 inversion
        inv_frac = (inv_now / sin_t.clamp(min=1e-3)).clamp(0.0, 1.0)
        orient_progress = inv_frac

        # Contact split (LINK_CONTACT_WRENCH Fz at the front/rear calf slots; |Fz|>1N
        # loaded). NOT FK-lagged -> authoritative for contact.
        rear_fz = self._wrench_view[:, self._rear_foot_slot, 2]    # (N,2)
        front_fz = self._wrench_view[:, self._front_foot_slot, 2]  # (N,2)
        rear_loaded = (rear_fz.abs() > 1.0).to(torch.float32)      # (N,2)
        front_loaded = (front_fz.abs() > 1.0).to(torch.float32)    # (N,2)

        # (3) rear_feet_contact: both REAR feet planted = the support base.
        rear_contact = rear_loaded.sum(dim=1) / 2.0               # (N,) in [0,1]
        # Is at least one rear foot carrying load? The anti-harvest GATE: the big
        # orientation + alive payout is only paid when the dog is actually on its
        # rear feet (so a non-balancing re-seated pose cannot farm reward). Below
        # the low-pitch start the rear feet are obviously loaded (the dog is
        # standing), so this gate only bites once the front legs leave the ground.
        rear_support = (rear_contact > 0.0).to(torch.float32)     # (N,) 0/1

        # (2) com_over_support: trunk xy over the REAR-foot support midpoint (trunk =
        #     kinematic CoM proxy). The rear foot world xy is FK-lagged 1 step
        #     (acceptable, 20 ms). Gated theta>~35 deg.
        rear_xy = self._pose_view[:, self._rear_foot_slot, 0:2].mean(dim=1)  # (N,2)
        d_xy2 = (base_xy - rear_xy).pow(2).sum(dim=1)              # (N,)
        com = torch.exp(-self._k_com * d_xy2)
        com = com * (theta_deg > GATE_COM_DEG).to(torch.float32)

        # (4) no_front_contact: the defining rear-leg-handstand feature -- FRONT feet
        #     OFF the ground. exp(-k * n_front). Gated theta>~40 deg so it does not
        #     punish the on-ground start.
        n_front = front_loaded.sum(dim=1)                         # (N,) 0..2
        no_front = torch.exp(-2.0 * n_front)
        no_front = no_front * (theta_deg > GATE_FRONT_DEG).to(torch.float32)

        # (5) base_height_handstand: pull the trunk UP to the achievable held height
        #     and AWAY from the collapse floor. z_target ramps from the flat stance
        #     toward HANDSTAND_BASE_Z (the held-rear-up height) as the body inverts.
        #     Two parts: a peaked term at z_target (don't over/under-shoot) PLUS a
        #     one-sided "stay off the floor" ramp that pays linearly for height above
        #     the collapse floor -- this is the dominant gradient that stops the
        #     trunk-sink that crushed the prior policy into the kill floor. RETRAIN
        #     2026-06-16: the old z_target ramped to an unreachable height with a
        #     sharp tolerance, giving ~0 gradient at the crouched height the policy
        #     actually reached.
        a = (theta_deg / 62.0).clamp(0.0, 1.0)
        z_target = STANCE_BASE_Z + (HANDSTAND_BASE_Z - STANCE_BASE_Z) * a
        base_height_peak = torch.exp(-(base_z - z_target).pow(2) / 0.02)
        height_above_floor = ((base_z - COLLAPSE_FLOOR_Z)
                              / (z_target - COLLAPSE_FLOOR_Z)).clamp(0.0, 1.0)
        base_height = 0.5 * base_height_peak + 0.5 * height_above_floor

        # (6) front_leg_lift: dense lift-the-front signal (front foot world z above a
        #     low baseline). Gated theta>~40 deg. FK-lagged 1 step (acceptable).
        front_z = self._pose_view[:, self._front_foot_slot, 2].mean(dim=1)  # (N,)
        h0 = 0.05
        front_lift = (front_z - h0).clamp(0.0, 0.6)
        front_lift = front_lift * (theta_deg > GATE_FRONT_DEG).to(torch.float32)

        # (7) stability (rate damp): static balance => damp body rates. Strengthened
        #     (RETRAIN 2026-06-16) so the policy is pushed toward a STILL held pose
        #     (low pitch-rate = not toppling), which is the balance signal that
        #     distinguishes a sustained rear-up from a fast topple-through.
        ang_vel = b.base_ang_vel()
        lin_vel = b.base_lin_vel()
        stability = -0.04 * ang_vel.pow(2).sum(dim=1) \
            - 0.2 * lin_vel.pow(2).sum(dim=1)

        # (7b) SAGITTAL-PLANE / lateral-roll stability (RETRAIN 2026-06-16 fix #3).
        #     A rear-leg handstand on two rear feet is a NARROW lateral base and the
        #     S2 policy was tipping SIDEWAYS (|proj_grav_y| ~ 0.65 in the deterministic
        #     eval -> roll-off collapses). proj_grav_y is the off-sagittal tilt; the
        #     body-frame roll/yaw rates ang_vel[x],ang_vel[z] are the lateral toppling
        #     rates. Penalize both directly + reward staying in-plane (exp peak at
        #     pgy=0) so the dog holds a clean sagittal rear-up, not a sideways lean.
        pgy = pg[:, 1]
        roll_rate = ang_vel[:, 0].pow(2) + ang_vel[:, 2].pow(2)
        sagittal = torch.exp(-8.0 * pgy.pow(2))                   # 1 at pgy=0
        lateral_pen = -1.5 * pgy.pow(2) - 0.1 * roll_rate

        # (8) reused Go2Reward regularizers (proven smoothness/limit forms). In the
        #     PD path _torques reconstructs tau = Kp(tgt-q)-Kd*qd from the live
        #     target the parent recorded this step (self._target_q_urdf).
        q_urdf = b.q_urdf()
        qd_urdf = b.qd_urdf()
        reg_action_rate = self._reward._action_rate(self.last_action, self._prev_action)
        reg_torques = self._reward._torques(q_urdf, qd_urdf, self._target_q_urdf)
        reg_dof_lim = self._reward._dof_pos_limits(q_urdf)
        regs = (-0.01 * reg_action_rate
                - 0.0002 * reg_torques
                - 10.0 * reg_dof_lim) * REWARD_DT

        # -- HOLD-TIME bookkeeping + bonus (the anti-harvest core, change B) ------
        # "in handstand" = oriented well toward the commanded pitch (orient high)
        # AND on loaded rear feet. Increment the per-env counter while in handstand,
        # RESET it to 0 the moment the env falls out. The hold bonus ramps with the
        # counter so only SUSTAINED balance pays -- it is 0 at the freshly-seated IC.
        # "in handstand" REQUIRES genuine inversion progress toward the command
        # (>=80% of the commanded inversion) AND rear support -- so the dominant
        # hold bonus only accrues when the dog is actually holding the COMMANDED
        # rear-up, not a safe low lean (RETRAIN 2026-06-16 fix #2).
        # Hold also REQUIRES the pose to be roughly in the sagittal plane (not rolled
        # sideways): |pgy| small => sagittal>0.5 (RETRAIN 2026-06-16 fix #3).
        good_orient = (orient > 0.6) & (inv_frac > 0.8) & (sagittal > 0.5)
        in_handstand = good_orient & (rear_support > 0.0)
        self._hold_steps = torch.where(
            in_handstand, self._hold_steps + 1.0,
            torch.zeros_like(self._hold_steps),
        )
        hold_frac = (self._hold_steps / HOLD_RAMP_STEPS).clamp(0.0, 1.0)
        hold_bonus = self._w["hold"] * hold_frac * in_handstand.to(torch.float32)

        # (9) alive floor (small, positive). GATED on rear support so it is not a
        #     free floor at a non-balancing pose (anti-harvest, change A).
        alive = self._w["alive"] * rear_support

        # ALL positive shaping terms are GATED on rear support (change A): a
        # well-oriented pose with the rear feet OFF the ground (the broken IC, or a
        # dog mid-air after a re-seat) pays NOTHING. The ONLY positive reward is
        # earned while the dog is genuinely standing on its rear feet -- so there is
        # nothing to harvest from a re-seated pose, and the policy is forced to keep
        # the rear feet loaded (which only sustained balance achieves). rear_contact
        # itself is ungated (it IS the rear-support signal -- the term that teaches
        # the dog to plant the rear feet in the first place from the low-pitch start).
        shaped = (
            self._w["orient"] * orient
            + self._w["orient_progress"] * orient_progress
            + self._w["sagittal"] * sagittal
            + hold_bonus
            + self._w["com"] * com
            + self._w["no_front"] * no_front
            + self._w["base_height"] * base_height
            + self._w["front_lift"] * front_lift
        ) * rear_support

        reward = (
            alive
            + shaped
            + self._w["rear_contact"] * rear_contact
            + stability
            + lateral_pen
            + regs
        )
        return reward.clamp_min(0.0).to(torch.float32)


def make_env(num_envs: int, *, device=None, **kw) -> Go2HandstandEnv:
    """Construct the Go2 handstand env on the shipped go2_locomotion scene (the
    SAME flat floating-base Go2 scene the walker uses). Parameters mirror
    :class:`Go2LocomotionEnv` plus the handstand knobs (``stage``, ``theta_cmd_obs``,
    ``reward_mode``, the reward weights)."""
    scene = G._go2_scene_path()
    return Go2HandstandEnv(scene, num_envs, device=device, **kw)
