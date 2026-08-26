"""Go2 standing BACKFLIP task -- direct-torque policy, event-physics reward.

Skill: from a standing start the robot crouches, launches, completes a backward
somersault, and lands stable on four feet. The target rotation is a per-env
COMMAND (see docs/roadmap/2026-08-25-spec-03-backflip-rotation-state-zh.md):
levels run 90 -> 180 -> 270 -> 360 deg, promoted by measured gyro-integrated
rotation, mirroring published Go2 backflip pipelines.

REWARD ORGANISATION (spec): shaped purely by PHYSICAL EVENTS derived from sim
state -- ground contact loss/gain (takeoff/landing), vertical launch speed,
CONTINUOUS unfolded pitch-angle integral (no wall-clock phase anywhere; the
accumulated angle IS the progress signal fed back through the observation),
and an upright four-foot stability window after touchdown.
"""
from __future__ import annotations

import numpy as np
import torch

import nuka

from ..gym.env import NukaGymEnv
from . import go2_obs as G

BACKFLIP_DEFAULT_URDF = np.array(
    [0.0, 0.8, -1.5,   0.0, 0.8, -1.5,
     0.0, 1.0, -1.5,   0.0, 1.0, -1.5], dtype=np.float32)
# Kept as a named alias for callers that refer to the generic standing pose.
STAND_URDF = BACKFLIP_DEFAULT_URDF
IC_BASE_Z = 0.32          # standing height (probe: settles ~0.28-0.33)

TARGET_LEVELS = tuple(t * np.pi for t in (0.0, 1.0, 2.0, 2.0))
FLIP_RATE_LEVELS = (0.0, 3.6, 7.2, 7.2)
_LEVELS_T = None          # cached cuda tensor of TARGET_LEVELS (built lazily)
PROMOTE_STREAK = 3        # consecutive credited episodes to advance a level

PREP_END_S = 1.10
TAKEOFF_START_S, TAKEOFF_END_S = 1.10, 1.45
FLIP_START_S, FLIP_END_S = 1.10, 1.90
LAND_START_S = 1.90
LANDING_HOLD_S = 0.75
FALL_START_Z = 0.72
FALL_DOWN_SPEED = -0.45
UP_VEL_CLAMP = 3.0
TARGET_HEIGHT = 0.30
FEET_DISTANCE_TARGET = 0.30

W_UP = 20.0
W_FLIP = 5.0
W_ORI = -1.0
W_HEIGHT = -10.0
W_FLAT_STAGE0 = -5.0
W_ROLL = -10.0
W_YAW = -1.0
W_SYM = -0.1
W_FEET_PRE = -30.0
W_FEET_DIST = -1.0
W_RATE = -0.001
W_CONTACT = -1.0
W_LANDING = 10.0
W_DRIFT = -5.0
OMEGA_KILL = 60.0
REWARD_MIN, REWARD_MAX = -20.0, 20.0

LOADED_FZ_N = 5.0
UPRIGHT_PGZ = -0.85
LAND_ANGLE_TOL = 0.3


class BackflipState:
    """Per-env command/state shared between the obs builder and the env."""

    def __init__(self, num_envs: int, device: torch.device) -> None:
        self.n = num_envs
        self.dev = device
        self.target_rot = torch.full((num_envs,), TARGET_LEVELS[0], device=device)
        self.rot_acc = torch.zeros(num_envs, device=device)
        self.phase = torch.zeros(num_envs, device=device)
        self.stage_idx = torch.zeros(num_envs, dtype=torch.long, device=device)
        self.max_up_vel = torch.zeros(num_envs, device=device)
        self.consec_ok = torch.zeros(num_envs, device=device)
        self.ep_success = torch.zeros(num_envs, dtype=torch.bool, device=device)

    def cmd_tensor(self) -> torch.Tensor:
        """(N,3): target rotation, measured rotation, wall-clock phase."""
        return torch.stack((self.target_rot / (2.0 * torch.pi),
                            self.rot_acc / (2.0 * torch.pi), self.phase), dim=-1)


class Go2BackflipObs(G.Go2TorqueObs):
    """Base 48-dim torque-mode obs + the rotation command/progress block."""

    def __init__(self, device, world) -> None:
        super().__init__(device, world)
        self.default_angles = torch.as_tensor(
            BACKFLIP_DEFAULT_URDF, dtype=torch.float32, device=self.q_urdf().device)
        self.state: "BackflipState | None" = None

    def wire_state(self, st: BackflipState) -> None:
        self.state = st

    def compose_extra(self) -> torch.Tensor:
        return self.state.cmd_tensor()


class Go2BackflipPDObs(G.Go2ObsBuilder):
    """PD-target-offset action variant: actions are joint-target offsets
    (nominal + 0.5*action, rad) executed by the engine's PD-position drive --
    the parameterization published Go2 flip pipelines use. Same obs layout as
    the torque variant (48 + rotation command/progress block)."""

    # Training policies remain bounded by the declared +/-1 Gym action space,
    # while imported Genesis/TorchScript skills legitimately emit larger PD
    # offsets (their deployment contract clips at +/-100).
    ACTION_CLIP_PD = 100.0
    ACTION_SCALE_PD = 0.5

    def __init__(self, device, world) -> None:
        super().__init__(device, world)
        self.state = None   # BackflipState, wired by the env

    def wire_state(self, st) -> None:
        self.state = st

    def state_finite(self) -> torch.Tensor:
        return (torch.isfinite(self.q_urdf()).all(-1)
                & torch.isfinite(self.qd_urdf()).all(-1)
                & torch.isfinite(self.base_lin_vel()).all(-1)
                & torch.isfinite(self.base_ang_vel()).all(-1))

    def compose_extra(self) -> torch.Tensor:
        return self.state.cmd_tensor()

    def compute_obs(self, command: torch.Tensor,
                    last_action: torch.Tensor) -> torch.Tensor:
        q_urdf = self.q_urdf()
        obs = torch.cat((
            self.base_lin_vel() * G.S_LIN_VEL,
            self.base_ang_vel() * G.S_ANG_VEL,
            self.projected_gravity(),
            torch.zeros_like(command),
            (q_urdf - self.default_angles) * G.S_DOF_POS,
            self.qd_urdf() * G.S_DOF_VEL,
            last_action,
            self.state.cmd_tensor(),
        ), dim=-1)
        return obs.clamp_(-G.OBS_CLIP, G.OBS_CLIP)

    def write_action(self, action: torch.Tensor) -> torch.Tensor:
        action = action.clamp(-self.ACTION_CLIP_PD, self.ACTION_CLIP_PD)
        target_urdf = self.default_angles + self.ACTION_SCALE_PD * action
        self._tgt[:, 1:G.GO2_BLC] = target_urdf.index_select(
            1, self.nuka_slot_for_urdf)
        return action


class Go2BackflipEnv(NukaGymEnv):
    """Standing backflip; action space is direct torque (default) or
    PD-target offsets (``action_mode="pd"``), selected at construction."""

    def __init__(self, scene: str, num_envs: int, *,
                 episode_length_s: float = 3.0,
                 action_mode: str = "torque",
                 seed: int | None = None,
                 **kw) -> None:
        # Flat horizontal ground: bake a static heightfield on the general
        # PairDriven contact path (same recipe as go2_locomotion).
        tc = dict(kw.pop("terrain_create", {}) or {})
        if not tc:
            tc = dict(contact_family=1, heightfield_terrain_type=1,
                      heightfield_nrow=161, heightfield_ncol=161,
                      heightfield_cell=0.25)
        kw["terrain_create"] = tc
        self._pd_mode = action_mode == "pd"
        if not self._pd_mode:
            kw.setdefault("control_mode", 1)
            obs_factory = lambda dev, world: Go2BackflipObs(dev, world)
        else:
            kw.pop("control_mode", None)   # engine default = PD-position drive
            obs_factory = lambda dev, world: Go2BackflipPDObs(dev, world)
        kw.pop("command", None)
        super().__init__(
            scene, num_envs,
            command=(0.0, 0.0, 0.0),
            episode_length_s=episode_length_s,
            seed=seed,
            obs_builder_factory=obs_factory,
            **kw)
        dev = self._torch_device
        self._obs_builder: Go2BackflipObs = self._obs
        self._bf = BackflipState(num_envs, dev)
        self._obs_builder.wire_state(self._bf)

        self._obs_builder.apply_pd_gains()
        if self._pd_mode:
            kp = torch.from_dlpack(self._world.buffer_view(nuka.DRIVE_STIFFNESS))
            kd = torch.from_dlpack(self._world.buffer_view(nuka.DRIVE_DAMPING))
            kp[:, 1:G.GO2_BLC] = 25.0
            kd[:, 1:G.GO2_BLC] = 0.5

        # Widen the gym spaces for the command/progress/phase block.
        from gymnasium import spaces as _spaces
        obs_dim = G.GO2_OBS_DIM + 3
        self.single_observation_space = _spaces.Box(
            low=-G.OBS_CLIP, high=G.OBS_CLIP, shape=(obs_dim,), dtype=np.float32)
        self.observation_space = _spaces.Box(
            low=-G.OBS_CLIP, high=G.OBS_CLIP,
            shape=(self.num_envs, obs_dim), dtype=np.float32)

        self._wrench = torch.from_dlpack(
            self._world.buffer_view(nuka.LINK_CONTACT_WRENCH))
        self._link_velocity = torch.from_dlpack(
            self._world.buffer_view(nuka.LINK_VELOCITY))
        self._pose = torch.from_dlpack(
            self._world.buffer_view(nuka.ARTICULATION_LINK_POSE))
        u_calf = [G.URDF_JOINT_NAMES.index(nm)
                  for nm in ("FL_calf", "FR_calf", "RL_calf", "RR_calf")]
        u_thigh = [G.URDF_JOINT_NAMES.index(nm)
                   for nm in ("FL_thigh", "FR_thigh", "RL_thigh", "RR_thigh")]
        slot = self._obs.nuka_slot_for_urdf_np
        self._paw_slot = torch.tensor(
            [int(slot[u]) + 1 for u in u_calf], dtype=torch.long, device=dev)
        self._thigh_slot = torch.tensor(
            [int(slot[u]) + 1 for u in u_thigh], dtype=torch.long, device=dev)
        self._foot_off = torch.tensor([[0.0, 0.0, -0.213]], device=dev)
        self._quat_conj = torch.tensor([[1.0, -1.0, -1.0, -1.0]], device=dev)

        self._stand = torch.as_tensor(STAND_URDF, device=dev)
        self._prev_action = torch.zeros(num_envs, G.GO2_ACTION_DIM, device=dev)
        self._home_xy = torch.zeros(num_envs, 2, device=dev)
        self._airborne = torch.zeros(num_envs, dtype=torch.bool, device=dev)
        self._took_off = torch.zeros(num_envs, dtype=torch.bool, device=dev)
        self._settle_steps = torch.zeros(num_envs, device=dev)
        self._landing_streak = torch.zeros(num_envs, dtype=torch.long, device=dev)
        self._landed = torch.zeros(num_envs, dtype=torch.bool, device=dev)

        # Failure-class counters (episode totals, reported via info).
        self._fail_under = torch.zeros((), device=dev)
        self._fail_over = torch.zeros((), device=dev)
        self._fail_crash = torch.zeros((), device=dev)
        self._fail_notakeoff = torch.zeros((), device=dev)
        self._success_count = torch.zeros((), device=dev)

        print(f"[go2_backflip] n={num_envs} action_mode={action_mode} "
              f"obs_dim={obs_dim} levels={[round(t/np.pi,1) for t in TARGET_LEVELS]}pi",
              flush=True)

    # -- ICs -----------------------------------------------------------------
    def _reset_joint_state(self, mask) -> None:
        m = torch.ones(self.num_envs, dtype=torch.bool,
                       device=self._torch_device) if mask is None else mask
        if not bool(m.any()):
            return
        q = torch.from_dlpack(self._world.buffer_view(nuka.JOINT_POSITION))
        qd = torch.from_dlpack(self._world.buffer_view(nuka.JOINT_VELOCITY))
        stand_slots = self._stand.index_select(0, self._obs.urdf_from_nuka_slot)
        q[m, 1:G.GO2_BLC] = stand_slots.unsqueeze(0).expand(int(m.sum()), -1)
        qd[m, 1:G.GO2_BLC] = 0.0
        tgt = torch.from_dlpack(self._world.buffer_view(nuka.DRIVE_TARGET))
        stand_slots = self._stand.index_select(0, self._obs.nuka_slot_for_urdf)
        if self._pd_mode:
            tgt[m, 1:G.GO2_BLC] = stand_slots.unsqueeze(0).expand(int(m.sum()), -1)
        else:
            tgt[m, 1:G.GO2_BLC] = 0.0   # torque mode: zero the aliasing buffer
        bp = torch.from_dlpack(self._world.buffer_view(nuka.BASE_POSE))
        bp[m, 2] = FALL_START_Z
        bp[m, 3] = 1.0             # BASE_POSE quaternion is [qw,qx,qy,qz]
        bp[m, 4:7] = 0.0
        self._link_velocity[m] = 0.0
        self._link_velocity[m, 0, 5] = FALL_DOWN_SPEED
        # Validated curriculum: vertical jump -> 180 deg -> 360 deg -> robust.
        promote = self._bf.ep_success[m] & (self._bf.consec_ok[m] >= PROMOTE_STREAK - 1)
        self._bf.stage_idx[m] = torch.where(
            promote,
            (self._bf.stage_idx[m] + 1).clamp(max=len(TARGET_LEVELS) - 1),
            self._bf.stage_idx[m])
        levels = torch.tensor(TARGET_LEVELS, dtype=torch.float32,
                              device=self._torch_device)
        self._bf.target_rot[m] = levels[self._bf.stage_idx[m]]
        self._bf.consec_ok[m] = torch.where(
            self._bf.ep_success[m],
            (self._bf.consec_ok[m] + 1).clamp(max=PROMOTE_STREAK),
            torch.zeros_like(self._bf.consec_ok[m]))
        self._bf.rot_acc[m] = 0.0
        self._bf.phase[m] = 0.0
        self._bf.max_up_vel[m] = 0.0
        self._bf.ep_success[m] = False
        self._airborne[m] = False
        self._took_off[m] = False
        self._settle_steps[m] = 0.0
        self._landing_streak[m] = 0
        self._landed[m] = False
        self._prev_action[m] = 0.0
        bpv = torch.from_dlpack(self._world.buffer_view(nuka.BASE_POSE))
        nuka.sync()
        self._home_xy[m] = bpv[m, 0:2]

    # -- helpers -------------------------------------------------------------
    def _feet_fz(self) -> torch.Tensor:
        return self._wrench[:, self._paw_slot, 2].abs()        # (N,4)

    def _paw_world(self) -> torch.Tensor:
        origin = self._pose[:, self._paw_slot, 0:3]
        quat = self._pose[:, self._paw_slot, 3:7] * self._quat_conj
        return origin + G.quat_rotate_inverse_wxyz(
            quat.reshape(-1, 4), self._foot_off.expand(
                self.num_envs, 4, 3).reshape(-1, 3)).reshape(self.num_envs, 4, 3)

    def compute_reward(self) -> torch.Tensor:
        """Validated go2-backflip-rl time-window reward, scaled by dt."""
        b, bf = self._obs, self._bf
        dt = self.dt * self.decimation
        t = self.episode_step.float() * dt
        in_prep = t < PREP_END_S
        in_takeoff = (t >= TAKEOFF_START_S) & (t < TAKEOFF_END_S)
        in_flip = (t >= FLIP_START_S) & (t < FLIP_END_S)
        in_land = t >= LAND_START_S

        omega = b.base_ang_vel()
        pg = b.projected_gravity()
        vz = b.base_lin_vel()[:, 2]
        z = b.base_pos()[:, 2]
        d_rot = (omega[:, 1] * dt).clamp(-0.6, 0.6)
        bf.rot_acc += d_rot
        bf.max_up_vel = torch.maximum(bf.max_up_vel, vz)

        stage = bf.stage_idx
        max_rates = torch.tensor(FLIP_RATE_LEVELS, device=stage.device)
        flip_max = max_rates[stage]
        up_vel = vz.clamp(0.0, UP_VEL_CLAMP) * in_takeoff
        flip_vel = (-omega[:, 1]).clamp(min=0.0)
        flip_vel = torch.minimum(flip_vel, flip_max) * in_flip

        ramp = ((t - FLIP_START_S) / (FLIP_END_S - FLIP_START_S)).clamp(0.0, 1.0)
        theta_ref = -bf.target_rot * ramp
        pitch_err = (bf.rot_acc - theta_ref).pow(2)
        height_err = (z - TARGET_HEIGHT).abs() * (~in_flip)
        flat = pg[:, 0:2].pow(2).sum(dim=-1)
        roll = pg[:, 1].abs()
        yaw = omega[:, 2].pow(2)

        paw = self._paw_world()
        # Only tax paw height AFTER the drop landing; the fall itself is free.
        feet_height_pre = ((paw[:, :, 2] - 0.03).clamp(min=0.0).sum(dim=-1)
                           * in_prep * self._landed.float())
        d_front = (paw[:, 0] - paw[:, 1]).norm(dim=-1)
        d_rear = (paw[:, 2] - paw[:, 3]).norm(dim=-1)
        feet_dist = (d_front - FEET_DISTANCE_TARGET).abs() + (d_rear - FEET_DISTANCE_TARGET).abs()

        act = self.last_action
        mirrored = torch.cat((-act[:, 3:4], act[:, 4:6], -act[:, 0:1],
                              act[:, 1:3], -act[:, 9:10], act[:, 10:12],
                              -act[:, 6:7], act[:, 7:9]), dim=-1)
        sym = (act - mirrored).pow(2).sum(dim=-1)
        rate = (act - self._prev_action).pow(2).sum(dim=-1)
        self._prev_action = act.clone()

        thigh_fz = self._wrench[:, self._thigh_slot, 2].abs()
        base_hit = self._wrench[:, 0, 2].abs() > 1.0
        undesired = base_hit.float() + (thigh_fz > 1.0).sum(dim=-1).float()
        fz = self._feet_fz()
        all_feet = (fz > LOADED_FZ_N).all(dim=-1)
        self._landing_streak = torch.where(
            all_feet, self._landing_streak + 1, torch.zeros_like(self._landing_streak))
        self._landed |= self._landing_streak >= 3
        pitch_done = (bf.rot_acc + bf.target_rot).abs() < LAND_ANGLE_TOL
        level = flat < 0.04
        landing = (all_feet & level & pitch_done & in_land).float()
        drift = (b.base_pos()[:, 0:2] - self._home_xy).pow(2).sum(dim=-1) * in_land

        stage0 = stage == 0
        r = (W_UP * up_vel
             + W_FLIP * flip_vel
             + W_ORI * pitch_err * (~stage0)
             + W_HEIGHT * height_err
             + W_FLAT_STAGE0 * flat * stage0
             + W_ROLL * roll
             + W_YAW * yaw
             + W_SYM * sym
             + W_FEET_PRE * feet_height_pre
             + W_FEET_DIST * feet_dist
             + W_RATE * rate
             + W_CONTACT * undesired
             + W_LANDING * landing
             + W_DRIFT * drift) * dt

        planted = (fz > LOADED_FZ_N).sum(dim=-1) >= 3
        self._took_off |= (~planted) & (self.episode_step > 5)
        near_end = self.episode_step >= self.max_episode_length - 2
        stage_ok = torch.where(stage0, bf.max_up_vel >= 1.0,
                    torch.where(stage == 1, bf.rot_acc <= -2.8,
                                             bf.rot_acc <= -5.9))
        with torch.no_grad():
            bf.ep_success |= near_end & stage_ok
            self._success_count += (near_end & stage_ok).float().sum()
            self._fail_notakeoff += (near_end & ~self._took_off).float().sum()

        return torch.clamp(torch.nan_to_num(r, nan=0.0), REWARD_MIN, REWARD_MAX)

    def compute_terminated(self) -> torch.Tensor:
        b = self._obs
        head_plant = self._wrench[:, 0, 2].abs() > 40.0
        bad_state = ~b.state_finite()   # NaN anywhere in state
        # Finite-but-absurd rates are solver blowups too: kill them before the
        # (clamped) reward lets a glitch look like progress.
        blowup = b.base_ang_vel().norm(dim=-1) > OMEGA_KILL
        return bad_state | head_plant | blowup

    def step(self, actions):
        # The episode starts with a real fall. Hold the standing PD target until
        # the paws have actually loaded AND the hold window elapsed, then expose
        # the flip controls.
        t = self.episode_step.float() * self.dt * self.decimation
        startup = ((t < LANDING_HOLD_S) | ~self._landed).view(-1, 1)
        actions = torch.where(startup, torch.zeros_like(actions), actions)
        self._bf.phase = ((self.episode_step.float() + 1.0)
                          / float(self.max_episode_length)).clamp(0.0, 1.0)
        obs, reward, terminated, truncated, info = super().step(actions)
        info["fail_under"] = float(self._fail_under.item())
        info["fail_over"] = float(self._fail_over.item())
        info["fail_crash"] = float(self._fail_crash.item())
        info["fail_no_takeoff"] = float(self._fail_notakeoff.item())
        info["success"] = float(self._success_count.item())
        info["stage_rot_deg"] = float(
            (-self._bf.rot_acc * 180.0 / torch.pi).mean().item())
        info["target_level"] = float(
            (self._bf.target_rot / (np.pi / 2)).mean().item())
        return obs, reward, terminated, truncated, info


def make_env(num_envs: int, *, device=None, action_mode: str = "torque",
             **kw) -> Go2BackflipEnv:
    scene = G._go2_scene_path()
    return Go2BackflipEnv(scene, num_envs, device=device,
                          action_mode=action_mode, **kw)
