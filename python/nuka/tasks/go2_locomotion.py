"""Go2 flat-ground locomotion task -- legged_gym reward + command curriculum.

A :class:`~nuka.gym.env.NukaGymEnv` subclass that supplies (a) the legged_gym Go2
locomotion REWARD ported verbatim (see :mod:`nuka.tasks.go2_rewards`) and (b) the
per-env velocity COMMAND curriculum (resampled from ``commands.ranges`` on a timer
and on reset, with ``|lin_vel|<0.2`` zeroed -- legged_gym behaviour). The obs math,
joint-order permutation, and height/tilt termination all live in the base env (and
the builder); this layer adds reward shaping + command randomization + a
``make_env`` factory so the RL stack has a single entry point.

This REPLACES the earlier mild hand-rolled shaping (forward-vel tracking + alive
bonus + unbounded action-rate). Under that placeholder the per-episode return was
~-14000..-20000 (~-1000/step) -- the unbounded action-rate penalty dominated and
the gradient did NOT point toward walking. The legged_gym port fixes this by
construction: ``only_positive_rewards`` floors the per-step total at 0 (so no
regularizer can drive it deeply negative) and the term BALANCE is the proven-to-
converge one.

----------------------------------------------------------------------------
COMMAND CURRICULUM (legged_gym ``commands`` + ``_resample_commands``):
  ranges (go2 inherits the base): lin_vel_x [-1,1], lin_vel_y [-1,1],
          ang_vel_yaw [-1,1] m/s | rad/s.
  resample period: ``resampling_time = 10.0`` s => 10.0 / control_dt(0.02) = 500
          control steps (and on full reset + on every autoreset/done env).
  zeroing: after sampling, ``commands[:, :2] *= (||commands_xy|| > 0.2)`` -- a
          near-zero xy command is snapped to exactly 0 (legged_gym line 306).
  DEVIATION (logged): legged_gym go2 has ``heading_command = True`` (the yaw
          command is derived from a heading error). We sample ``ang_vel_yaw``
          DIRECTLY from its range instead (the ``else`` branch). Heading mode
          needs a 4th command dim + a per-step yaw recompute from the base
          quaternion + an obs-layout change; the 3-dim obs command is golden-
          pinned, so we keep the direct-yaw form. The ±velocity range the p04
          exit needs is fully covered.
  Seeded (deterministic) via the base env's ``torch.Generator``.

AUTORESET FIXES (the two bugs the p03 review flagged):
  * ``_prev_action`` (the action-rate delta source) is zeroed for done envs on
    the per-step MASKED autoreset (legged_gym ``last_actions[env_ids]=0`` in
    ``reset_idx``), not only on full ``reset()``. Without this, the first step of
    a fresh episode penalizes the fresh action against the terminal action.
  * ``reward.last_dof_vel`` (the dof_acc finite-difference source) is likewise
    zeroed for done envs (legged_gym ``last_dof_vel[env_ids]=0``), else the first
    fresh step sees a spurious huge acceleration.
----------------------------------------------------------------------------
"""

from __future__ import annotations

import os

import torch

import nuka

from ..gym.env import NukaGymEnv
from . import go2_obs as G
from . import terrain as T
from .go2_rewards import Go2Reward, DROPPED, REWARD_DT


# Nominal base spawn height (m): the go2_locomotion scene cooks the trunk at
# z=0.445 (verified). Used as the offset added to the local terrain surface height
# so the base spawns ABOVE the platform/pit/box, not inside it.
NOMINAL_BASE_Z = 0.445


# legged_gym commands.ranges (go2 inherits the LeggedRobotCfg base defaults).
CMD_RANGE_LIN_VEL_X = (-1.0, 1.0)
CMD_RANGE_LIN_VEL_Y = (-1.0, 1.0)
CMD_RANGE_ANG_VEL_YAW = (-1.0, 1.0)
# commands.resampling_time = 10.0 s ; min |xy| command (else zeroed).
RESAMPLING_TIME_S = 10.0
CMD_ZERO_THRESHOLD = 0.2

# Flight-gait LANDING GRACE (pronk/bound): terminate on the base-height floor only
# after the floor has been breached for this many CONSECUTIVE control steps, so a
# hard 1-step touchdown dip does not false-terminate. Active ONLY for non-walk
# gait modes (walk is byte-identical to before -- the grace counter is never read).
LANDING_GRACE_STEPS = 2


class Go2LocomotionEnv(NukaGymEnv):
    """Go2 flat-ground velocity-tracking locomotion env (legged_gym reward).

    Same construction signature as :class:`NukaGymEnv`; overrides
    :meth:`compute_reward` (the ported reward) and :meth:`step` (command
    resampling + the autoreset bookkeeping fixes). The ``command`` ctor arg is the
    INITIAL command (immediately overwritten by the first resample on reset).
    """

    def __init__(self, scene: str, num_envs: int, *,
                 fixed_command: bool = False, terrain: "dict | None" = None,
                 gait_mode: str = "walk",
                 contact_family: int = 0,
                 heightfield_terrain_type: int = 0,
                 heightfield_nrow: int = 0,
                 heightfield_ncol: int = 0,
                 heightfield_cell: float = 0.0,
                 **kw) -> None:
        # GAIT MODE (Go2 pronk/bound skill). Re-weights the reward dict only (see
        # go2_rewards.GAIT_SCALE_OVERRIDES / GAIT_AIR_TIME_BIAS) -- no engine, obs or
        # action change. ``walk`` (default) => byte-identical to the proven flat
        # walker. Stored before super().__init__ (which does NOT take it) so it is
        # available when the Go2Reward is constructed below.
        self._gait_mode = str(gait_mode)
        # ----------------------------------------------------------------------
        # TERRAIN CURRICULUM (Go2-on-stairs Phase 2b). ``terrain`` is the yaml
        # ``config.env_config.terrain`` block (or None => terrain DISABLED, the
        # flat path -- byte-identical to before). When enabled it (1) cooks the
        # world with the procedural-terrain geometry at the MAX step height
        # (terrain.STEP_HEIGHT_MAX) and (2) drives a per-env type+difficulty
        # curriculum, writing ENV_TERRAIN_TYPE / ENV_TERRAIN_DIFFICULTY +
        # BASE_POSE.z on every reset. Default OFF: with terrain None, NO terrain
        # kwargs reach create_from_scene and NO per-env terrain field is written,
        # so the existing flat training is reproduced exactly.
        # ----------------------------------------------------------------------
        tcfg = dict(terrain) if terrain else {}
        self._terrain_on = bool(tcfg.get("enable", False))
        self._tcfg = tcfg
        if self._terrain_on:
            # Cook the world with the procedural-terrain geometry (step height set
            # to the MAX; per-env difficulty scales it down toward 0.08 m).
            kw["terrain_create"] = {
                "terrain_step_height": float(tcfg.get("step_height", T.STEP_HEIGHT_MAX)),
                "terrain_step_width": float(tcfg.get("step_width", 0.3)),
                "terrain_platform_width": float(tcfg.get("platform_width", 1.0)),
                "terrain_grid_width": float(tcfg.get("grid_width", 0.45)),
                "terrain_grid_height_max": float(tcfg.get("grid_height_max", 0.15)),
            }

        # ONE-GENERAL-SOLVER landing (L1): route the world through the GENERAL
        # contact path (contact_family=1 -> PairDriven + a baked static heightfield
        # collidable) instead of the legacy analytic FusedFoot kernel. 0 (default) ==
        # byte-identical FusedFoot. terrain_create is splatted into create_from_scene
        # by NukaGymEnv, so we inject the general-path kwargs there (building the dict
        # even when the procedural terrain is OFF -- a flat heightfield is baked from
        # the all-zero model.terrain). NOTE: the baked heightfield is MODEL-LEVEL (one
        # terrain type for the whole batch), so the general path uses a single
        # heightfield_terrain_type rather than the per-env curriculum type; used by
        # the L1 acceptance rollout (single homogeneous terrain) + the demo.
        self._contact_family = int(contact_family)
        if self._contact_family != 0:
            tc = dict(kw.get("terrain_create", {}))
            tc["contact_family"] = int(contact_family)
            tc["heightfield_terrain_type"] = int(heightfield_terrain_type)
            if heightfield_nrow:
                tc["heightfield_nrow"] = int(heightfield_nrow)
            if heightfield_ncol:
                tc["heightfield_ncol"] = int(heightfield_ncol)
            if heightfield_cell > 0.0:
                tc["heightfield_cell"] = float(heightfield_cell)
            kw["terrain_create"] = tc

        super().__init__(scene, num_envs, **kw)

        # Build the curriculum + cache the engine terrain field views (zero-copy).
        if self._terrain_on:
            type_dist = tcfg.get("type_distribution", None)
            if isinstance(type_dist, dict):
                # yaml gives str keys -> int.
                type_dist = {int(k): float(v) for k, v in type_dist.items()}
            self._curriculum = T.TerrainCurriculum(
                self.num_envs, self._torch_device,
                # COLD-START: default near-FLAT (difficulty 0 => flat geometry for
                # every terrain type) so a from-scratch policy can bootstrap walking
                # before any step height; the survival+displacement curriculum then
                # raises difficulty toward 1.0 (= 0.15 m). Config knob:
                # env_config.terrain.start_difficulty (default DIFFICULTY_COLD_START).
                start_difficulty=float(tcfg.get("start_difficulty", T.DIFFICULTY_COLD_START)),
                demote_on_fall=bool(tcfg.get("demote_on_fall", True)),
                difficulty_step=float(tcfg.get("difficulty_step", 0.1)),
                ema_beta=float(tcfg.get("ema_beta", 0.99)),
                type_distribution=type_dist,
                # SCALE-INDEPENDENT promotion gates (replace the broken reward-
                # threshold gate): survive most of the episode AND actually WALK A
                # LOT. The PRIMARY "walked a lot" signal is the cumulative PATH
                # length (promote_path_m), which -- unlike NET displacement -- is
                # FREE-ROAM robust (a wandering dog has small net disp but large
                # path). NET displacement (promote_dist_m) is kept as a diagnostic.
                max_episode_length=int(self.max_episode_length),
                survive_frac=float(tcfg.get("survive_frac", 0.8)),
                promote_path_m=float(tcfg.get("promote_path_m", 5.0)),
                promote_dist_m=float(tcfg.get("promote_dist_m", 3.0)),  # diag only.
                promote_thr=float(tcfg.get("promote_thr", 8.0)),  # diagnostic only.
            )
            self._ground_height = float(tcfg.get("ground_height", 0.0))
            self._terrain_step_height = float(tcfg.get("step_height", T.STEP_HEIGHT_MAX))
            self._terrain_grid_height_max = float(tcfg.get("grid_height_max", 0.15))
            # FULL terrain geometry (the create-time TerrainParams) the BATCHED
            # arbitrary-(x,y) sampler needs -- mirror the create_from_scene kwargs
            # so the spawn height at a RANDOM column matches the foot kernel exactly.
            self._terrain_step_width = float(tcfg.get("step_width", 0.3))
            self._terrain_platform_width = float(tcfg.get("platform_width", 1.0))
            self._terrain_grid_width = float(tcfg.get("grid_width", 0.45))
            # RANDOM-XY SPAWN + FREE-ROAM (随地走) config. spawn_xy_range = the half-
            # edge (m) of the square the per-reset (x,y) is sampled uniformly from,
            # centered on the tile origin. Default 2.0 m keeps the dog inside the
            # real stepped region (half_platform + kPyramidRings*step_width ~=
            # 0.5 + 8*0.3 = 2.9 m), so it starts on different steps/rings/boxes.
            # randomize_yaw randomizes the base HEADING so the same body-frame
            # velocity command sends the dog in a random WORLD direction (free roam).
            self._spawn_xy_range = float(tcfg.get("spawn_xy_range", 2.0))
            self._randomize_yaw = bool(tcfg.get("randomize_yaw", True))
            # Zero-copy WRITABLE engine views (terrain type uint32 (N,1); difficulty
            # float32 (N,); base_pose float32 (N,7)).
            w = self._world
            self._tt_view = torch.from_dlpack(w.buffer_view(nuka.ENV_TERRAIN_TYPE))
            self._td_view = torch.from_dlpack(w.buffer_view(nuka.ENV_TERRAIN_DIFFICULTY))
            self._bp_view = torch.from_dlpack(w.buffer_view(nuka.BASE_POSE))
            # Per-step bookkeeping the curriculum-on-done hook reads (set in step()).
            self._last_reward = torch.zeros(self.num_envs, device=self._torch_device)
            self._last_terminated = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self._torch_device
            )
            # LIVE base (x,y) captured each step in compute_reward (the displacement
            # gate's end point at the done step -- before autoreset clobbers it).
            self._last_base_xy = torch.zeros(
                self.num_envs, 2, device=self._torch_device
            )
            # Optional terrain diagnostic (NUKA_GO2_TERRAIN_DIAG=1): base-z min/max
            # + periodic type/difficulty histogram. Always-on float guards start at
            # +-inf so the first step seeds them.
            self._tdiag = (dict(n=0, z_min=float("inf"), z_max=float("-inf"))
                           if os.environ.get("NUKA_GO2_TERRAIN_DIAG") else None)
            print(f"[go2_locomotion] TERRAIN curriculum ON: "
                  f"{self._curriculum.histogram()}", flush=True)
        else:
            self._curriculum = None
            self._tdiag = None

        # ----------------------------------------------------------------------
        # HEIGHT SCAN (legged_gym terrain perception). DEFAULT OFF. Only active on
        # the terrain-ON path (a height scan needs the curriculum terrain field to
        # sample). When OFF -- height_scan.enable false OR terrain off -- NOTHING
        # below runs and the env (obs 48, observation_space, every behaviour) is
        # byte-identical to the blind env. When ON, the env APPENDS an n_scan-dim
        # heightfield measurement to the 48-dim obs (see _height_scan + step/reset)
        # and OVERRIDES the gym observation_space to (48 + n_scan,) so rl_games
        # builds the right network input.
        # ----------------------------------------------------------------------
        hscfg = dict(tcfg.get("height_scan", {})) if self._terrain_on else {}
        self._height_scan_on = bool(hscfg.get("enable", False)) and self._terrain_on
        self._scan_n = 0
        if self._height_scan_on:
            x_min = float(hscfg.get("x_min", -0.8))
            x_max = float(hscfg.get("x_max", 0.8))
            y_min = float(hscfg.get("y_min", -0.5))
            y_max = float(hscfg.get("y_max", 0.5))
            res = float(hscfg.get("resolution", 0.1))
            self._scan_scale = float(hscfg.get("scale", 5.0))
            self._scan_clip = float(hscfg.get("clip", 1.0))
            nx = int(round((x_max - x_min) / res)) + 1
            ny = int(round((y_max - y_min) / res)) + 1
            self._scan_n = nx * ny
            # Fixed (n_scan, 2) base-frame grid offsets (meshgrid of x in
            # [x_min,x_max], y in [y_min,y_max], step res). Kept on device; rotated
            # by each env's yaw + translated to its base xy in _height_scan.
            gx = torch.linspace(x_min, x_max, nx, device=self._torch_device)
            gy = torch.linspace(y_min, y_max, ny, device=self._torch_device)
            mesh_x, mesh_y = torch.meshgrid(gx, gy, indexing="ij")   # (nx, ny)
            self._scan_grid = torch.stack(
                (mesh_x.reshape(-1), mesh_y.reshape(-1)), dim=1
            ).contiguous()                                           # (n_scan, 2)
            # OVERRIDE the gym obs spaces to (48 + n_scan,) / (N, 48 + n_scan) so
            # rl_games sizes the network input to the scan-augmented obs. Same
            # Box low/high (-OBS_CLIP/+OBS_CLIP) + dtype as the base 48-dim space.
            import numpy as _np
            from gymnasium import spaces as _spaces
            full_dim = G.GO2_OBS_DIM + self._scan_n
            self.single_observation_space = _spaces.Box(
                low=-G.OBS_CLIP, high=G.OBS_CLIP, shape=(full_dim,), dtype=_np.float32
            )
            self.observation_space = _spaces.Box(
                low=-G.OBS_CLIP, high=G.OBS_CLIP,
                shape=(self.num_envs, full_dim), dtype=_np.float32,
            )
            print(f"[go2_locomotion] HEIGHT SCAN ON: grid {nx}x{ny} = "
                  f"{self._scan_n} pts (x[{x_min},{x_max}] y[{y_min},{y_max}] "
                  f"res {res}); scale {self._scan_scale} clip {self._scan_clip}; "
                  f"obs dim {G.GO2_OBS_DIM} -> {full_dim}", flush=True)

        # CURRICULUM SWITCH (researcher's reduce-the-problem lever). When
        # ``fixed_command=True`` the per-env velocity command is PINNED to the ctor
        # ``command`` for the whole run (no resample on reset / timer / autoreset).
        # WHY: legged_gym trains under a random command in [-1,1]; but with the
        # ``only_positive_rewards`` clamp, a cold-start (init_noise_std=1.0) policy
        # under random commands tracks neither lin nor ang vel, so the small
        # tracking positive (~0.004/step) is swamped by the regularizing negatives
        # (~-0.007/step) -> the clamp zeros ~60% of steps -> the episodic return is
        # 0 -> the value head collapses -> PPO collapse (the observed 1500-epoch
        # run). Pinning the command to [0.5,0,0] hands the policy a POSITIVE,
        # CLIMBABLE reward floor (~+0.005/step, only ~10% clamped -- measured): a
        # still robot already tracks the 0 yaw / 0 lat command, and forward motion
        # then RAISES tracking_lin_vel toward +0.02/step (a clean gradient toward the
        # commanded 0.5 m/s). This is the task-sanctioned "demonstrate a learning
        # signal" reduction; broaden the command range as a later single change once
        # the fixed-command signal is established.
        self._fixed_command = bool(fixed_command)
        # Optional in-env training instrumentation (NUKA_GO2_DIAG=1; see _diag_step).
        self._diag = (dict(n=0, sum_a_mean=0.0, max_a=0.0, sum_r=0.0,
                           n_term=0, n_trunc=0)
                      if os.environ.get("NUKA_GO2_DIAG") else None)
        # Ported legged_gym reward (owns last_dof_vel for dof_acc + feet_air_time).
        # gait_mode re-weights its scale dict (walk == byte-identical to before).
        self._reward = Go2Reward(
            self.num_envs, self._torch_device, gait_mode=self._gait_mode
        )

        # LANDING GRACE (flight gaits only). A hard pronk/bound touchdown can dip the
        # base below termination_height for a SINGLE control step before the legs
        # re-extend; the absolute height-kill would false-terminate at the first hop,
        # starving the learning signal (risk #3). For gait_mode != 'walk' we require
        # the height floor to be breached for >= LANDING_GRACE_STEPS CONSECUTIVE
        # control steps before terminating on height (the tilt-kill is unchanged --
        # a genuine flip still dies immediately). DEFAULT-OFF for walk: when
        # gait_mode == 'walk' compute_terminated never consults this counter, so the
        # flat reward/termination/curriculum/goldens are byte-identical.
        self._landing_grace_on = (self._gait_mode != "walk")
        self._below_floor_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self._torch_device
        )

        # ----------------------------------------------------------------------
        # GAIT-CRITICAL term wiring (feet_air_time + collision). Both read engine
        # per-link fields that the FusedFoot Go2 world DOES populate:
        #   * LINK_CONTACT_WRENCH (field 16): net per-link contact force/torque
        #     (world). ReadoutContactWrench runs for this !is_union world, so each
        #     foot sphere's contact force is summed onto its owning CALF link. A
        #     foot is "in contact" iff |wrench.Fz| > 1 N -> feets_air_time signal.
        #   * ARTICULATION_LINK_POSE (field 1): per-link world pose -> the thigh/
        #     calf/base link world z for the collision terrain-clip proxy.
        # LINK index map: ARTICULATION_LINK_POSE / LINK_CONTACT_WRENCH are indexed
        # by LOCAL link slot (0 = trunk/base, 1..12 = the leg-joint child links).
        # Each foot sphere is cooked onto its CALF link; the calf link slot for a
        # URDF calf joint j is (nuka_slot_for_urdf[j] + 1) (the +1 skips the root
        # slot). We derive foot/thigh slots from the obs builder's structurally-
        # discovered permutation so a future scrambled joint order stays correct.
        b = self._obs
        _u_calf = [G.URDF_JOINT_NAMES.index(n) for n in
                   ("FL_calf", "FR_calf", "RL_calf", "RR_calf")]
        _u_thigh = [G.URDF_JOINT_NAMES.index(n) for n in
                    ("FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh")]
        # foot CONTACT links == the 4 calf links (the foot sphere's owning link).
        self._foot_link_slot = torch.as_tensor(
            [int(b.nuka_slot_for_urdf_np[u]) + 1 for u in _u_calf],
            dtype=torch.long, device=self._torch_device,
        )
        # collision penalised links: thigh + calf + base/trunk (legged_gym go2
        # penalised_contacts = thigh + calf; we also include the base/trunk slot 0
        # so a base belly-flop onto a step is penalised before the height-kill).
        _calf_slots = [int(b.nuka_slot_for_urdf_np[u]) + 1 for u in _u_calf]
        _thigh_slots = [int(b.nuka_slot_for_urdf_np[u]) + 1 for u in _u_thigh]
        self._collision_link_slot = torch.as_tensor(
            _thigh_slots + _calf_slots + [0],   # +base/trunk (root) slot
            dtype=torch.long, device=self._torch_device,
        )
        # Zero-copy engine field views (alias the live buffers; re-read each step).
        self._wrench_view = torch.from_dlpack(
            self._world.buffer_view(nuka.LINK_CONTACT_WRENCH)  # (N, BLC, 6)
        )
        self._pose_view = torch.from_dlpack(
            self._world.buffer_view(nuka.ARTICULATION_LINK_POSE)  # (N, BLC, 7)
        )
        print(f"[go2_locomotion] feet_air_time foot(calf)-link slots = "
              f"{self._foot_link_slot.tolist()}; collision penalised-link slots = "
              f"{self._collision_link_slot.tolist()} (thigh+calf+base)", flush=True)
        # Prior step's clipped action (the action-rate delta source). Distinct
        # from base.last_action, which is THIS step's clipped action.
        self._prev_action = torch.zeros(
            self.num_envs, G.GO2_ACTION_DIM, device=self._torch_device
        )
        # PD target this step (default + 0.25*action), in URDF order -- the torque
        # approximation needs it; recomputed each step from the clipped action.
        self._target_q_urdf = self._reward.default_angles.expand(
            self.num_envs, G.GO2_ACTION_DIM
        ).contiguous()
        # Resample period in CONTROL steps.
        self._resample_every = max(1, int(round(RESAMPLING_TIME_S / REWARD_DT)))
        # Log the ACTIVE reward terms once (no DROPs now: feet_air_time + collision
        # are re-enabled -- see go2_rewards). DROPPED is empty so this loop is a
        # no-op; kept so a future genuine drop is still surfaced honestly.
        print(f"[go2_locomotion] ACTIVE reward terms = "
              f"{sorted(self._reward.scales.keys())}", flush=True)
        for name, reason in DROPPED.items():
            print(f"[go2_locomotion] DROPPED reward term '{name}': {reason}",
                  flush=True)

    # -- init-pose teleport (legged_gym resets dof_pos to the exact default) --
    def _reset_joint_state(self, mask) -> None:
        """Teleport the (masked, or all) envs' 12 leg joints to the legged_gym
        ASYMMETRIC default (hip +-0.1, front thigh 0.8 / rear thigh 1.0, calf -1.5)
        and zero their joint velocity, in Nuka slot order via the permutation.

        WHY (the PPO-collapse root cause): the cooked scene snapshot the engine
        restores on reset/autoreset is a near-symmetric crouch (hip=0), which for
        Go2 is metastable in roll -- a cold-start (init_noise_std=1.0) policy tips it
        into clamped-negative flailing within ~22 control steps before any positive
        tracking return accrues, so the value head collapses to 0 and PPO dies
        (observed: reward==0, ep_len~22, entropy/bounds_loss runaway). legged_gym
        resets dof_pos to the exact default on EVERY episode start; the hip +-0.1
        splay gives the roll margin that lets the robot survive long enough at cold
        start to clear the only_positive_rewards clamp (+~0.005/step upright). The
        base writes the engine's JOINT_POSITION / JOINT_VELOCITY buffers in place
        (DLPack views), which persists through the subsequent step (verified)."""
        w = self._world
        b = self._obs
        default_nuka = b.default_angles[b.nuka_slot_for_urdf]  # URDF -> Nuka slots
        q = torch.from_dlpack(w.buffer_view(nuka.JOINT_POSITION))
        qd = torch.from_dlpack(w.buffer_view(nuka.JOINT_VELOCITY))
        if mask is None:
            q[:, 1:G.GO2_BLC] = default_nuka
            qd[:, 1:G.GO2_BLC] = 0.0
        else:
            q[mask, 1:G.GO2_BLC] = default_nuka
            qd[mask, 1:G.GO2_BLC] = 0.0
        nuka.sync()
        # Spawn-on-terrain placement (Phase 2b). For the DONE envs (mask is the
        # done mask in autoreset), FIRST run the curriculum promote/demote off this
        # step's reward + terminated (mirrored onto self by compute_reward /
        # compute_terminated), THEN write the (possibly updated) per-env terrain
        # TYPE/DIFFICULTY + lift the base z to the LOCAL surface + nominal stance
        # height -- so a reset onto a raised platform/pit/box does NOT spawn the
        # robot inside the surface (the Phase-2a Baumgarte-launch bug). On a full
        # reset (mask is None) the curriculum is left at its current assignment and
        # only the engine fields are (re)written. No-op when terrain is disabled.
        if self._terrain_on:
            if mask is not None:
                # mask is the (num_envs,) done bool. Promote/demote off the SCALE-
                # INDEPENDENT survival+displacement gate: an env that survived most
                # of the episode (curriculum.ep_len) AND walked far enough
                # (||terminal base xy - start xy||, terminal captured pre-autoreset
                # in compute_reward) is promoted; a fall (terminated, not just
                # time-limit truncation) demotes.
                self._curriculum.update_on_done(
                    mask, self._last_terminated & mask, base_xy=self._last_base_xy
                )
            self._write_terrain_for_reset(mask)

    # -- spawn-on-terrain: random (x,y) + yaw + lift base z to local surface --
    def _write_terrain_for_reset(self, mask) -> None:
        """Write ENV_TERRAIN_TYPE / ENV_TERRAIN_DIFFICULTY (the curriculum's
        current per-env assignment) and place the base at a RANDOM (x,y) within the
        terrain tile, at a RANDOM heading, lifted to the LOCAL surface height +
        NOMINAL_BASE_Z, for the (masked, or all) reset envs.

        RANDOM-XY SPAWN + FREE-ROAM (随地走). Per reset env we sample (x,y)
        uniformly in [-spawn_xy_range, +spawn_xy_range]^2 around the tile origin and
        (when randomize_yaw) a heading yaw in [-pi, pi]. The LOCAL surface height at
        each sampled column comes from the BATCHED C-ABI sampler
        ``nuka.terrain_sample_height_batch`` -- the host-side projection of the
        SINGLE source of truth ``SampleTerrainHeight`` (+ difficulty scale), so the
        spawn z is read from the EXACT heightfield the foot-contact kernel rests the
        feet on (no python center-only mirror drift for arbitrary columns). This
        lifts the base ABOVE the local platform/pit/box step instead of inside it
        (the Phase-2a "spawn inside a raised step -> Baumgarte launch to z~=18" bug),
        AND, combined with the body-frame velocity command, sends the dog walking in
        a random world direction from a random start = 随地走.

        Called from :meth:`_reset_joint_state` AFTER world.reset/reset_envs settled
        the buffers and BEFORE the obs is recomposed. The curriculum's promote/
        demote for the done envs has already run this step. ``terrain.center_height``
        is kept only for reference/diagnostic now; the live reset uses the batched
        arbitrary-(x,y) sampler."""
        cur = self._curriculum
        gen = self._generator
        dev = self._torch_device
        n = self.num_envs

        # Per-env random spawn column (x,y) in the configurable square + heading.
        rng = self._spawn_xy_range
        spawn_x = (torch.rand(n, generator=gen, device=dev) * 2.0 - 1.0) * rng
        spawn_y = (torch.rand(n, generator=gen, device=dev) * 2.0 - 1.0) * rng
        if self._randomize_yaw:
            yaw = (torch.rand(n, generator=gen, device=dev) * 2.0 - 1.0) * torch.pi
        else:
            yaw = torch.zeros(n, device=dev)

        # LOCAL surface height at each sampled column via the SINGLE-source C-ABI
        # sampler (host-side; (x,y,type,diff) -> z). The sampler takes CPU arrays;
        # the per-reset cost is tiny (N control-plane samples) so the host round-trip
        # is fine. Difficulty is the curriculum's live per-env scale.
        types_cpu = cur.terrain_type.to(torch.int64).cpu().to(torch.uint32).contiguous()
        xs_cpu = spawn_x.detach().cpu().to(torch.float32).contiguous()
        ys_cpu = spawn_y.detach().cpu().to(torch.float32).contiguous()
        diff_cpu = cur.difficulty.detach().cpu().to(torch.float32).contiguous()
        surf = nuka.terrain_sample_height_batch(
            types_cpu, xs_cpu, ys_cpu, diff_cpu,
            self._ground_height, self._terrain_step_height,
            self._terrain_step_width, self._terrain_platform_width,
            self._terrain_grid_width, self._terrain_grid_height_max,
        )
        base_z = torch.as_tensor(surf, device=dev, dtype=torch.float32) + NOMINAL_BASE_Z

        # Yaw -> quaternion (w-first, rotation about world +Z): [cos(y/2),0,0,sin(y/2)].
        half = yaw * 0.5
        qw = torch.cos(half)
        qz = torch.sin(half)

        if mask is None:
            self._tt_view[:, 0] = cur.terrain_type.to(self._tt_view.dtype)
            self._td_view[:] = cur.difficulty
            self._bp_view[:, 0] = spawn_x
            self._bp_view[:, 1] = spawn_y
            self._bp_view[:, 2] = base_z
            # BASE_POSE quat is [qw,qx,qy,qz] at cols 3..6; a yaw rotation sets only
            # qw (col 3) and qz (col 6), qx=qy=0 (upright heading).
            self._bp_view[:, 3] = qw
            self._bp_view[:, 4] = 0.0
            self._bp_view[:, 5] = 0.0
            self._bp_view[:, 6] = qz
        else:
            # NB torch has no masked index_put for UInt32, so write the ENV_TERRAIN
            # TYPE column via torch.where (the curriculum type is unchanged for
            # non-done envs, so a full-column write is a no-op for them). The float
            # BASE_POSE / DIFFICULTY writes are masked to the done envs so a non-done
            # env's LIVE mid-episode pose is NOT clobbered.
            self._tt_view[:, 0] = torch.where(
                mask, cur.terrain_type, self._tt_view[:, 0].to(torch.long)
            ).to(self._tt_view.dtype)
            self._td_view[mask] = cur.difficulty[mask]
            self._bp_view[mask, 0] = spawn_x[mask]
            self._bp_view[mask, 1] = spawn_y[mask]
            self._bp_view[mask, 2] = base_z[mask]
            self._bp_view[mask, 3] = qw[mask]
            self._bp_view[mask, 4] = 0.0
            self._bp_view[mask, 5] = 0.0
            self._bp_view[mask, 6] = qz[mask]
        nuka.sync()

        # Record the spawn (x,y) as the episode START for the displacement-gate
        # curriculum (||terminal - start||). Done here, AFTER the spawn pose is
        # written, for the same (masked or all) reset envs.
        spawn_xy = torch.stack((spawn_x, spawn_y), dim=1)
        self._curriculum.set_start_xy(mask, spawn_xy)

    # -- command sampling (legged_gym _resample_commands, seeded) -----------
    def _resample_commands(self, mask: torch.Tensor) -> None:
        """Resample the velocity command for the envs in the boolean ``mask``.

        ``commands[i] = U[range]`` for lin_vel_x / lin_vel_y / ang_vel_yaw, then
        ``commands[:, :2] *= (||xy|| > 0.2)`` (zero near-still commands). Uses the
        base env's seeded generator for determinism. All on GPU.

        No-op when ``fixed_command=True`` -- the command stays pinned to the ctor
        ``command`` for the whole run (see __init__)."""
        if self._fixed_command:
            return
        n = int(mask.sum().item())
        if n == 0:
            return
        gen = self._generator

        def _u(lo, hi):
            return torch.rand(n, generator=gen, device=self._torch_device) * (hi - lo) + lo

        new = torch.empty(n, 3, device=self._torch_device)
        new[:, 0] = _u(*CMD_RANGE_LIN_VEL_X)
        new[:, 1] = _u(*CMD_RANGE_LIN_VEL_Y)
        new[:, 2] = _u(*CMD_RANGE_ANG_VEL_YAW)
        # Zero the xy command if its magnitude is below the threshold.
        keep = (new[:, :2].norm(dim=1) > CMD_ZERO_THRESHOLD).unsqueeze(1)
        new[:, :2] = new[:, :2] * keep
        self.command[mask] = new

    def reset(self, *, seed=None, options=None):
        # Reset the curriculum's per-episode accumulators on a FULL reset (the
        # type/difficulty ASSIGNMENT is preserved across the reset -- a full reset
        # is a fresh rollout, not a demotion). super().reset() -> _reset_joint_state
        # (None) then writes the engine terrain fields + base z for all envs.
        if self._curriculum is not None:
            self._curriculum.ep_reward.zero_()
            self._curriculum.ep_len.zero_()
        out = super().reset(seed=seed, options=options)
        # Fresh-episode bookkeeping.
        self._prev_action.zero_()
        self._reward.last_dof_vel.zero_()
        self._reward.feet_air_time.zero_()
        self._reward.last_contacts.zero_()
        if self._landing_grace_on:
            self._below_floor_steps.zero_()
        # Resample EVERY env's command on a full reset, then recompose the obs
        # command slot so the returned reset obs carries the new command.
        all_envs = torch.ones(
            self.num_envs, dtype=torch.bool, device=self._torch_device
        )
        self._resample_commands(all_envs)
        obs, info = out
        obs[:, 9:12] = self.command * self._obs.cmd_scale
        # APPEND the legged_gym height scan (the base 48 cols are unchanged; we
        # only concatenate). No-op when height_scan is disabled (obs stays 48-dim).
        if self._height_scan_on:
            obs = torch.cat([obs, self._height_scan()], dim=-1)
        return obs, info

    def compute_reward(self) -> torch.Tensor:
        """Ported legged_gym Go2 reward (num_envs,) cuda float32. Reads the live,
        post-step engine state via the base obs builder; ``self.last_action`` is
        THIS step's clipped action (the base sets it before calling us);
        ``self._prev_action`` is the prior step's clipped action."""
        b = self._obs
        # GAIT-CRITICAL signals (read live, zero-copy, post-step):
        #   foot_contact_fz: (N,4) the 4 feet's world contact-force z, taken from
        #     LINK_CONTACT_WRENCH at each foot's owning CALF link. |Fz|>1N == loaded.
        #   collision_count: (N,) # of penalised links (thigh/calf/base) clipping
        #     BELOW the local terrain surface (the kinematic terrain-clip proxy).
        foot_contact_fz = self._wrench_view[:, self._foot_link_slot, 2]  # (N,4)
        collision_count = self._collision_count()                        # (N,)
        r = self._reward.compute(
            cmd=self.command,
            lin_vel=b.base_lin_vel(),
            ang_vel=b.base_ang_vel(),
            q_urdf=b.q_urdf(),
            qd_urdf=b.qd_urdf(),
            target_q_urdf=self._target_q_urdf,
            action=self.last_action,
            last_action=self._prev_action,
            # base_lin_vel (body frame) drives the gait-mode vertical_oscillation
            # term (v_z at lift-off). 0-weighted for walk/trot -> no effect there.
            base_lin_vel=b.base_lin_vel(),
            foot_contact_fz=foot_contact_fz,
            collision_count=collision_count,
        )
        # Feed the per-env reward into the terrain curriculum's running episode
        # return (the promotion gate). Done at reward time so it runs BEFORE the
        # base step's autoreset (which calls _reset_joint_state -> the curriculum
        # update). No-op when terrain is disabled.
        if self._curriculum is not None:
            self._curriculum.accumulate(r)
            # Capture the LIVE base (x,y) this step. compute_reward runs BEFORE the
            # base step's autoreset clobbers the done envs' pose, so the value held
            # here at the done step IS the episode's TERMINAL position -- the
            # net-displacement diagnostic's end point (||terminal - start||).
            self._last_base_xy = b.base_pos()[:, :2].detach().clone()
            # Accumulate this step's PATH increment (||base_xy - prev_xy||) into the
            # cumulative path length -- the PRIMARY (free-roam-robust) promotion
            # gate. Done here (pre-autoreset) so the terminal step's true motion is
            # counted; the subsequent set_start_xy re-seeds prev_xy to the new spawn
            # so the terminal->spawn teleport is NOT counted as path.
            self._curriculum.accumulate_path(self._last_base_xy)
        return r

    # -- local terrain surface sampler (shared by collision proxy + height-kill) --
    def _terrain_surface(self, xs: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        """(N, M) local terrain surface z at world (xs, ys) per env, using the env's
        LIVE curriculum terrain TYPE + DIFFICULTY (broadcast across the trailing M
        dim), via the SINGLE-source batched C-ABI sampler
        ``nuka.terrain_sample_height_batch`` -- the exact heightfield the foot kernel
        rests the feet on. ``xs``/``ys`` are (N, M); the sampler takes flat CPU
        arrays so we flatten (N*M,) and reshape back. Only called on the terrain-ON
        paths (the collision proxy and the relative height-kill)."""
        n, m = xs.shape
        cur = self._curriculum
        types = cur.terrain_type.view(n, 1).expand(n, m)         # (N, M) long
        diff = cur.difficulty.view(n, 1).expand(n, m)            # (N, M) float
        types_cpu = types.reshape(-1).to(torch.int64).cpu().to(torch.uint32).contiguous()
        xs_cpu = xs.reshape(-1).detach().cpu().to(torch.float32).contiguous()
        ys_cpu = ys.reshape(-1).detach().cpu().to(torch.float32).contiguous()
        diff_cpu = diff.reshape(-1).detach().cpu().to(torch.float32).contiguous()
        surf = nuka.terrain_sample_height_batch(
            types_cpu, xs_cpu, ys_cpu, diff_cpu,
            self._ground_height, self._terrain_step_height,
            self._terrain_step_width, self._terrain_platform_width,
            self._terrain_grid_width, self._terrain_grid_height_max,
        )
        return torch.as_tensor(
            surf, device=self._torch_device, dtype=torch.float32
        ).view(n, m)

    # -- legged_gym height scan: forward-looking terrain measurement -----------
    def _height_scan(self) -> torch.Tensor:
        """(N, n_scan) terrain height measurement around the base, legged_gym
        convention. The fixed base-frame grid (``self._scan_grid``, (n_scan, 2)) is
        rotated by each env's heading (YAW only) and translated to the base (x,y),
        sampled against the env's LIVE curriculum terrain (type+difficulty) via the
        shared C-ABI heightfield sampler (:meth:`_terrain_surface` -- the exact
        field the foot kernel rests the feet on), then reduced to the legged_gym
        value ``clip(base_z - 0.5 - surface, -clip, +clip) * scale``.

        Base pose is read from the AUTHORITATIVE BASE_POSE view (``self._bp_view``):
        xy at cols 0,1; z at col 2; quat (w-first) at cols 3..6. Only called on the
        height-scan-ON path. Returns float32 on device."""
        bp = self._bp_view                                       # (N, 7)
        base_x = bp[:, 0:1]                                      # (N, 1)
        base_y = bp[:, 1:2]                                      # (N, 1)
        base_z = bp[:, 2:3]                                      # (N, 1)
        qw, qx, qy, qz = bp[:, 3], bp[:, 4], bp[:, 5], bp[:, 6]  # each (N,)
        # Heading yaw from the (w-first) quaternion (rotation about world +Z).
        yaw = torch.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        ).unsqueeze(1)                                           # (N, 1)
        cos_y = torch.cos(yaw)                                   # (N, 1)
        sin_y = torch.sin(yaw)                                   # (N, 1)
        px = self._scan_grid[:, 0].unsqueeze(0)                  # (1, n_scan)
        py = self._scan_grid[:, 1].unsqueeze(0)                  # (1, n_scan)
        # Rotate the base-frame grid by yaw (heading frame), translate to base xy.
        wx = base_x + (px * cos_y - py * sin_y)                  # (N, n_scan)
        wy = base_y + (px * sin_y + py * cos_y)                  # (N, n_scan)
        heights = self._terrain_surface(wx, wy)                  # (N, n_scan)
        value = (base_z - 0.5 - heights).clamp(
            -self._scan_clip, self._scan_clip
        ) * self._scan_scale
        return value.to(torch.float32)

    # -- collision proxy: penalised links clipping the local terrain surface --
    def _collision_count(self) -> torch.Tensor:
        """(N,) count of penalised links (thigh/calf/base) whose world z is BELOW
        the local terrain surface at their (x,y) -- the kinematic terrain-clipping
        proxy for the legged_gym collision penalty (see Go2Reward._collision; there
        is no true per-body contact force in the foot-only FusedFoot model).

        TERRAIN ON  -> the local surface z per (link x,y) comes from the SINGLE-
            source batched C-ABI sampler ``nuka.terrain_sample_height_batch`` (the
            exact heightfield the foot kernel rests the feet on), using each env's
            live curriculum terrain TYPE + DIFFICULTY.
        TERRAIN OFF -> the surface is the flat ground at z=0 (``ground_height``),
            so the proxy is still ACTIVE (and faithfully ~0: a thigh/calf/base only
            clips z<0 in a hard fall, which the height-kill already terminates) --
            no DROP, byte-compatible with the flat reward semantics.
        """
        pose = self._pose_view                                   # (N, BLC, 7)
        links = self._collision_link_slot                        # (K,)
        link_z = pose[:, links, 2]                               # (N, K) world z
        n, k = link_z.shape

        if not self._terrain_on:
            # Flat ground at ground_height 0 for every link.
            surface = torch.zeros_like(link_z)
            return self._reward._collision(link_z, surface)

        # Per-link (x,y) for the batched terrain sampler. The terrain TYPE +
        # DIFFICULTY are PER-ENV (broadcast across that env's K links inside the
        # shared sampler helper).
        link_x = pose[:, links, 0]                               # (N, K)
        link_y = pose[:, links, 1]                               # (N, K)
        surface = self._terrain_surface(link_x, link_y)          # (N, K)
        return self._reward._collision(link_z, surface)

    def compute_terminated(self) -> torch.Tensor:
        """Termination predicate, with the terminal (fell) mask captured for the
        terrain curriculum's demote-on-fall rule.

        TERRAIN OFF -> the base predicate verbatim (absolute base_z <
        termination_height OR tilt > termination_tilt_deg); byte-identical, no
        capture overhead when there is no curriculum.

        TERRAIN ON -> the height-kill is made RELATIVE to the LOCAL terrain surface
        at the base (x,y): ``(base_z - local_surface) < termination_height``. The
        ABSOLUTE base-z kill is WRONG on terrain -- a dog walking DOWN into the
        inverted-pyramid pit (surface ~ -1.2 m) or down stairs has base_z far below
        the 0.18 m absolute floor even while walking PERFECTLY, so it was being
        terminated on contact (eval: Pit d>=0.5 ep_len~1, instant death) and the
        curriculum could never train descents. On FLAT (terrain_type 0 -> surface 0)
        this reduces EXACTLY to the absolute kill, so flat behaviour is preserved.
        The tilt kill is unchanged.

        LANDING GRACE (gait_mode != 'walk' only): the base-height breach (``fell``)
        must persist for LANDING_GRACE_STEPS consecutive control steps before it
        terminates, so a single-step hard-landing dip does not false-terminate a
        pronk/bound (risk #3). The tilt-kill is NEVER graced. For gait_mode == 'walk'
        this whole branch is skipped (``_landing_grace_on`` False) and the original
        two paths below run UNCHANGED -- the flat termination is byte-identical."""
        # WALK (and any non-flight default): the original byte-identical paths.
        if not self._landing_grace_on:
            if self._curriculum is None or not self._terrain_on:
                term = super().compute_terminated()
                if self._curriculum is not None:
                    self._last_terminated = term
                return term
            base = self._obs.base_pos()                          # (N, 7) -> xyz
            base_z = base[:, 2]
            surf = self._terrain_surface(base[:, 0:1], base[:, 1:2])[:, 0]  # (N,)
            fell = (base_z - surf) < self.termination_height
            tipped = self._obs.tilt_deg() > self.termination_tilt_deg
            term = fell | tipped
            self._last_terminated = term
            return term

        # FLIGHT GAIT (pronk/bound): same height/tilt predicates, but the height
        # breach is debounced over LANDING_GRACE_STEPS consecutive steps.
        base = self._obs.base_pos()                              # (N, 7) -> xyz
        base_z = base[:, 2]
        if self._curriculum is not None and self._terrain_on:
            surf = self._terrain_surface(base[:, 0:1], base[:, 1:2])[:, 0]  # (N,)
            below = (base_z - surf) < self.termination_height
        else:
            below = base_z < self.termination_height
        tipped = self._obs.tilt_deg() > self.termination_tilt_deg
        # Advance the consecutive-below-floor counter (reset to 0 the moment the base
        # recovers above the floor); terminate on height only once the counter has
        # reached the grace window.
        self._below_floor_steps = torch.where(
            below,
            self._below_floor_steps + 1,
            torch.zeros_like(self._below_floor_steps),
        )
        fell = self._below_floor_steps >= LANDING_GRACE_STEPS
        term = fell | tipped
        if self._curriculum is not None:
            self._last_terminated = term
        return term

    def step(self, actions: torch.Tensor):
        # Record the PD target this step (default + 0.25*action, URDF order) for
        # the torque approximation -- mirror the clip the base write_action does.
        clipped = actions.to(self._torch_device, dtype=torch.float32).clamp(
            -G.ACTION_CLIP, G.ACTION_CLIP
        )
        self._target_q_urdf = self._reward.default_angles + G.ACTION_SCALE * clipped

        # Base step: writes PD targets, steps physics, composes obs, calls our
        # compute_reward (which reads self.last_action == this step's clipped
        # action and self._prev_action == prior), and runs the masked autoreset.
        obs, reward, terminated, truncated, info = super().step(actions)

        # Roll the action-rate history AFTER the reward used the (prev, this) pair.
        self._prev_action = self.last_action.clone()

        # Periodic command resample (legged_gym _post_physics_step_callback:
        # episode_length_buf % (resampling_time/dt) == 0). episode_step was just
        # incremented in the base step.
        periodic = (self.episode_step % self._resample_every == 0)

        done = terminated | truncated
        if bool(done.any()):
            # Autoreset bookkeeping fixes: zero the action-rate + dof_acc history
            # for the done envs (legged_gym reset_idx zeros last_actions +
            # last_dof_vel). The base already zeroed last_action[done].
            self._prev_action[done] = 0.0
            self._reward.last_dof_vel[done] = 0.0
            # feet_air_time state machine: zero the per-foot air-time accumulator +
            # last-contact debounce for the done envs (legged_gym reset_idx zeros
            # feet_air_time + last_contacts) so a fresh episode does not inherit the
            # terminal swing phase / contact history.
            self._reward.feet_air_time[done] = 0.0
            self._reward.last_contacts[done] = False
            # Landing-grace counter: a fresh episode must not inherit the terminal
            # below-floor streak (no-op for walk -- the counter is never read there).
            if self._landing_grace_on:
                self._below_floor_steps[done] = 0

        # Resample command for (periodic | done) envs; recompose the obs command
        # slot so the returned obs carries the fresh command for those envs.
        to_resample = periodic | done
        if bool(to_resample.any()):
            self._resample_commands(to_resample)
            # compute_obs returns a fresh tensor (the done-path already cloned),
            # so patching the command slot in place is always safe here.
            obs[:, 9:12] = self.command * self._obs.cmd_scale

        if self._diag is not None:
            self._diag_step(actions, reward, terminated, truncated)

        # Terrain curriculum diagnostic (guarded by NUKA_GO2_TERRAIN_DIAG): tracks
        # the base-z min/max over the run (catches a spawn-on-terrain regression --
        # a z~=18 launch) + prints the live type/difficulty histogram periodically.
        if self._terrain_on and self._tdiag is not None:
            self._terrain_diag_step()

        # APPEND the legged_gym height scan (the base 48 cols are byte-unchanged;
        # the autoreset already patched obs[done,6:9] within those 48 dims). No-op
        # when height_scan is disabled (obs stays the 48-dim blind vector).
        if self._height_scan_on:
            obs = torch.cat([obs, self._height_scan()], dim=-1)

        return obs, reward, terminated, truncated, info

    # -- terrain curriculum diagnostic (NUKA_GO2_TERRAIN_DIAG=1) --------------
    def _terrain_diag_step(self) -> None:
        d = self._tdiag
        z = self._bp_view[:, 2]
        d["z_min"] = min(d["z_min"], float(z.min()))
        d["z_max"] = max(d["z_max"], float(z.max()))
        d["n"] += 1
        if d["n"] % 25 == 0:
            print(f"[go2_terrain] step{d['n']}: base_z[min={d['z_min']:.3f} "
                  f"max={d['z_max']:.3f}]  {self._curriculum.histogram()}  "
                  f"{self._curriculum.gate_diag()}",
                  flush=True)

    # -- optional in-env training instrumentation (NUKA_GO2_DIAG=1) ----------
    def _diag_step(self, actions, reward, terminated, truncated) -> None:
        """Aggregate per-step diagnostics on the REAL training loop (guarded by
        the NUKA_GO2_DIAG env var). Prints every ~50 control steps: mean/max |raw
        action| received from the policy (catches the action-saturation the IID
        sim can't reproduce), post-clamp reward mean, termination count + cause
        (z<height vs tilt>cutoff), and the reset pose -- so we can attribute the
        ep_len=~22 collapse to real dynamics vs a metric artifact."""
        d = self._diag
        a_abs = actions.abs()
        d["sum_a_mean"] += float(a_abs.mean()); d["max_a"] = max(d["max_a"], float(a_abs.max()))
        d["sum_r"] += float(reward.mean())
        b = self._obs
        z = b.base_pos()[:, 2]; td = b.tilt_deg()
        # terminated is the PRE-autoreset terminal mask; classify its cause from the
        # terminal state we still hold here (obs was swapped, but base_pos reads live
        # buffers which for done envs are now reset -- so recompute cause from the
        # termination predicate evaluated against the recorded terminal would need a
        # pre-reset capture; instead count terminations + truncations only).
        d["n_term"] += int(terminated.sum()); d["n_trunc"] += int(truncated.sum())
        d["n"] += 1
        if d["n"] % 50 == 0:
            print(f"[go2_diag] step{d['n']}: mean|a|={d['sum_a_mean']/d['n']:.3f} "
                  f"max|a|={d['max_a']:.2f}  mean_r={d['sum_r']/d['n']:.5f}  "
                  f"term/50={d['n_term']}  trunc/50={d['n_trunc']}  "
                  f"z_mean={float(z.mean()):.3f} tilt_mean={float(td.mean()):.1f}",
                  flush=True)
            d.update(n_term=0, n_trunc=0)


def make_env(num_envs: int, *, device=None, **kw) -> Go2LocomotionEnv:
    """Construct the Go2 locomotion env on the shipped go2_locomotion scene.

    Parameters mirror :class:`NukaGymEnv` (``dt``, ``decimation``, ``command``,
    ``termination_height``, ``termination_tilt_deg``, ``episode_length_s``,
    ``seed``). ``device`` is an open ``nuka.Device`` (created+owned by the env if
    ``None``). The scene is fixed to ``examples/scenes/go2_locomotion.usda`` (a
    byte-identical copy of the proven go2_float flat-ground floating-base scene)."""
    scene = G._go2_scene_path()
    return Go2LocomotionEnv(scene, num_envs, device=device, **kw)
