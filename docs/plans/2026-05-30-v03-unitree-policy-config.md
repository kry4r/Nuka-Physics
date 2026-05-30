# Unitree Go2 Policy Conventions — Contract for the Nuka Validation Harness

Date: 2026-05-30
Author: research/download subagent (v0.3 sprint)
Purpose: Capture the EXACT obs / action / PD conventions a Unitree Go2 locomotion
policy expects, so the Python harness (next task) drives Nuka's 4096-env Go2
without a convention mismatch (wrong obs dim / scale / joint order) that would
look like a physics bug.

---

## 0. HEADLINE FINDING — NO pretrained Go2 policy ships in either repo

**Neither Unitree repo ships a pretrained Go2 walking policy.** This is the single
most important takeaway and it changes the next task.

- `unitree_rl_gym` (legged_gym / Isaac Gym based, the PRIMARY repo) ships
  `deploy/pre_train/{g1,h1,h1_2}/motion.pt` — **but NOT go2.** Go2 is supported
  for *training only* (`--task go2`). The README even states "The default model is
  located at `deploy/pre_train/{robot}/motion.pt`", yet the `go2` subdirectory does
  not exist. There is also **no go2 deploy YAML** (only g1/h1/h1_2 under
  `deploy/deploy_mujoco/configs/` and `deploy/deploy_real/configs/`).
- `unitree_rl_lab` (IsaacLab based, SECONDARY repo) ships exported policies only for
  `g1_29dof` (`deploy/robots/g1_29dof/config/policy/**/exported/policy.onnx`).
  Its go2 deploy config points at a `policy_dir`
  (`../../../logs/rsl_rl/unitree_go2_velocity`) that the user is expected to train
  themselves — **no go2 policy is shipped.**
- **No GitHub releases and no tags** exist on either repo (verified via GitHub API:
  `releases` count = 0, `tags` = []), so there is no release-asset policy and no
  download script anywhere. The README does not reference any go2 policy download.

**Consequence for the next task:** to "drive Nuka with a *pretrained* Go2 policy",
the controller must either (a) train a Go2 policy from `unitree_rl_gym`'s shipped
config (`legged_gym/envs/go2/go2_config.py`, `--task go2`) and export it, or
(b) source a Go2 `.pt`/`.onnx` from elsewhere. This doc specifies the conventions
that policy WILL have if trained from this repo's default config (the most likely
path). If a 3rd-party / deploy-style 45-dim policy is substituted instead, see §3
note on dropping base_lin_vel.

---

## 1. Sources (cloned OUTSIDE the Nuka tree, no commits)

| Repo | Path | Commit | Date |
|---|---|---|---|
| unitree_rl_gym (PRIMARY) | `/root/third_party/unitree_rl_gym` | `276801e46c5d433564f24658bac64f254b7d2d4b` | 2025-07-25 |
| unitree_rl_lab (SECONDARY) | `/root/third_party/unitree_rl_lab` | `4960b84732b0c2ec593dccbfe963fda1bcd7b1e3` | (HEAD) |

Both shallow clones (`--depth 1`, `GIT_LFS_SKIP_SMUDGE=1`). The go2 `.pt` policies
that DO exist (g1/h1/h1_2) are plain git blobs (~140 KB each, repo has no
`.gitattributes`/LFS), so they are real, not LFS pointers — but again, no go2 one.

Pretrained `.pt` files present (for reference; NONE are go2):
- `/root/third_party/unitree_rl_gym/deploy/pre_train/g1/motion.pt`   (145745 bytes)
- `/root/third_party/unitree_rl_gym/deploy/pre_train/h1/motion.pt`   (139345 bytes)
- `/root/third_party/unitree_rl_gym/deploy/pre_train/h1_2/motion.pt` (145745 bytes)

---

## 2. CONTRACT — conventions for a Go2 policy trained from `unitree_rl_gym`

All values below are read VERBATIM from:
- `legged_gym/envs/go2/go2_config.py` (go2-specific overrides)
- `legged_gym/envs/base/legged_robot_config.py` (inherited defaults)
- `legged_gym/envs/base/legged_robot.py` (the actual obs/torque code)
- `resources/robots/go2/urdf/go2.urdf` (the DOF order)

Nothing here is guessed.

### 2.1 Dimensions

| Field | Value | Source |
|---|---|---|
| `num_observations` | **48** | base cfg `env.num_observations = 48` |
| `num_privileged_obs` | `None` (go2 does NOT use asymmetric critic obs) | base cfg |
| `num_actions` | **12** | base cfg `env.num_actions = 12` |

**The Go2 training obs IS 48-dim and INCLUDES base linear velocity.** (Contrast the
deploy/sim2sim policies for g1, whose deploy YAML uses `num_obs: 47` — a
g1-specific humanoid layout that drops lin_vel and adds phase terms. That 47 is
irrelevant to go2. There is NO go2 deploy config, hence no 45-dim go2 variant.)

### 2.2 Exact obs vector composition (order matters — from `compute_observations()`)

`legged_robot.py:185` builds `obs_buf = torch.cat((...))` in THIS order:

| Idx range | Component | Scale | Dim |
|---|---|---|---|
| `[0:3]`   | `base_lin_vel` (body frame) `* obs_scales.lin_vel` | **2.0** | 3 |
| `[3:6]`   | `base_ang_vel` (body frame) `* obs_scales.ang_vel` | **0.25** | 3 |
| `[6:9]`   | `projected_gravity` (gravity unit vec in body frame) | 1.0 (no scale) | 3 |
| `[9:12]`  | `commands[:, :3] * commands_scale` | `[2.0, 2.0, 0.25]` | 3 |
| `[12:24]` | `(dof_pos - default_dof_pos) * obs_scales.dof_pos` | **1.0** | 12 |
| `[24:36]` | `dof_vel * obs_scales.dof_vel` | **0.05** | 12 |
| `[36:48]` | `actions` (the PREVIOUS step's raw policy output, unscaled) | 1.0 | 12 |

Total = 3+3+3+3+12+12+12 = **48**. Confirmed by the noise-vector code
(`legged_robot.py:416`: `noise_vec[12:12+num_actions] = ... dof_pos` → dof_pos
block starts at index 12, i.e. 3+3+3+3 precede it).

Notes:
- `base_lin_vel`/`base_ang_vel` are in the **base/body frame**
  (`quat_rotate_inverse(base_quat, root_states[..])`), not world frame.
- `commands_scale = [lin_vel(2.0), lin_vel(2.0), ang_vel(0.25)]` — note the 3rd
  command is the yaw rate scaled by **ang_vel scale 0.25**, NOT a separate scale.
- `commands[:, :3]` = [lin_vel_x_cmd, lin_vel_y_cmd, ang_vel_yaw_cmd]. The 4th
  command (heading) is NOT in the obs.
- The `actions` term is the raw (pre-scale) action from the previous policy step.
  At t=0 it is zeros.

### 2.3 obs scales / clipping (`normalization` in base cfg)

```
obs_scales.lin_vel = 2.0
obs_scales.ang_vel = 0.25
obs_scales.dof_pos = 1.0
obs_scales.dof_vel = 0.05
obs_scales.height_measurements = 5.0   # only used if measure_heights/perceptive obs; NOT in the 48-dim blind obs
clip_observations = 100.0   # final obs_buf clamped to [-100, 100]
clip_actions      = 100.0   # incoming actions clamped to [-100, 100] before use
```

### 2.4 DOF / joint ORDER — THE SILENT KILLER

The DOF order is **the URDF joint declaration order**, obtained at runtime via
`self.gym.get_asset_dof_names(robot_asset)` (`legged_robot.py:556`). It is NOT the
order of the python dict in the config. The config dict in `go2_config.py` lists
joints as FL, RL, FR, RR — **that ordering is a decoy; ignore it.**

Parsed from `resources/robots/go2/urdf/go2.urdf` (declaration order = DOF index):

```
index : joint name
  0   : FL_hip_joint
  1   : FL_thigh_joint
  2   : FL_calf_joint
  3   : FR_hip_joint
  4   : FR_thigh_joint
  5   : FR_calf_joint
  6   : RL_hip_joint
  7   : RL_thigh_joint
  8   : RL_calf_joint
  9   : RR_hip_joint
 10   : RR_thigh_joint
 11   : RR_calf_joint
```

Leg order: **FL, FR, RL, RR**; within each leg: **hip, thigh, calf**.
**Nuka MUST present `dof_pos`, `dof_vel`, `default_dof_pos`, Kp/Kd, and consume the
policy's `actions` in exactly this 12-element order, or remap.** If Nuka's internal
joint order differs, build a permutation once and apply it on both the obs in and
the action out.

### 2.5 default_dof_pos (nominal standing angles, in DOF order above)

From `go2_config.py:default_joint_angles` (mapped into the URDF DOF order):

| Idx | Joint | Angle [rad] |
|---|---|---|
| 0 | FL_hip   |  0.1 |
| 1 | FL_thigh |  0.8 |
| 2 | FL_calf  | -1.5 |
| 3 | FR_hip   | -0.1 |
| 4 | FR_thigh |  0.8 |
| 5 | FR_calf  | -1.5 |
| 6 | RL_hip   |  0.1 |
| 7 | RL_thigh |  1.0 |
| 8 | RL_calf  | -1.5 |
| 9 | RR_hip   | -0.1 |
| 10| RR_thigh |  1.0 |
| 11| RR_calf  | -1.5 |

As a flat vector in DOF order:
```
default_dof_pos = [ 0.1, 0.8, -1.5,  -0.1, 0.8, -1.5,   0.1, 1.0, -1.5,  -0.1, 1.0, -1.5 ]
                    # FL hip/thigh/calf  FR ...           RL ...           RR ...
```
(Note hip sign: +0.1 on LEFT legs, -0.1 on RIGHT legs; thigh is 0.8 front / 1.0 rear.)
Initial base height `init_state.pos = [0, 0, 0.42] m`.

### 2.6 Action / control law (control_type = "P")

From `go2_config.py:control` and `legged_robot.py:_compute_torques`:

```
control_type  = 'P'           # position PD
action_scale  = 0.25
decimation    = 4
stiffness     = {'joint': 20.0}   # matches ALL 12 joints (substring 'joint' in every name)
damping       = {'joint': 0.5}    # matches ALL 12 joints
```

Gain assignment (`legged_robot.py:473`): for each joint, if a stiffness key is a
substring of the joint name, use it. `'joint'` is a substring of every Go2 joint
name → **Kp = 20.0 and Kd = 0.5 for ALL 12 joints (uniform).**

Torque law actually executed (`legged_robot.py:323`):
```
actions_scaled = action * action_scale          # 0.25 * action
target_q       = actions_scaled + default_dof_pos
torque = Kp * (target_q - dof_pos) - Kd * dof_vel
       = 20.0 * (0.25*action + default_dof_pos - dof_pos) - 0.5 * dof_vel
torque = clip(torque, -torque_limits, +torque_limits)
```
**Target joint VELOCITY is 0** (the Kd term multiplies `dof_vel` directly; there is
no `target_vel`). This matches Nuka's `tau = Kp*(target - q) + Kd*(target_vel - qd)`
with `target_vel = 0`.

Per-joint torque (effort) limits from the URDF (used by the final clip):
- hip:   effort 23.7 N·m, vel 30.1 rad/s, range [-1.0472, 1.0472]
- thigh: effort 23.7 N·m, vel 30.1 rad/s, range [-1.5708, 3.4907]
- calf:  effort 35.55 N·m, vel 20.07 rad/s, range [-2.7227, -0.83776]

### 2.7 Timestep / decimation / control frequency

```
sim.dt        = 0.005 s          # 200 Hz physics
decimation    = 4                # policy acts every 4 physics steps
=> policy dt  = 0.020 s          # 50 Hz control frequency
```
The policy is queried at **50 Hz**; each policy action is held for 4 physics
sub-steps of 5 ms (the PD torque is recomputed every physics step with the same
target, using the current dof_pos/dof_vel — see `step()` loop).

### 2.8 Command ranges (training, base cfg `commands.ranges`)

```
num_commands = 4   # [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]
lin_vel_x   = [-1.0, 1.0]  m/s
lin_vel_y   = [-1.0, 1.0]  m/s
ang_vel_yaw = [-1.0, 1.0]  rad/s
heading     = [-3.14, 3.14] rad   # heading_command=True; yaw-rate cmd is recomputed from heading error, NOT in obs
```
Only the first 3 commands enter the obs (see §2.2 idx [9:12]). For a simple
"walk forward" validation, set commands ≈ `[vx, 0, 0]` (e.g. vx=0.5 m/s). Note the
engine zeroes lin-vel commands whose norm < 0.2 during training resampling
(`legged_robot.py`: `commands[:, :2] *= (norm > 0.2)`), so use vx ≥ 0.2 to move.

---

## 3. How to map this to Nuka (the contract the harness follows)

For each policy step (50 Hz), per env:

1. **Build the 48-dim obs** in the order of §2.2, using Nuka ground-truth state:
   - base_lin_vel (body frame) * 2.0
   - base_ang_vel (body frame) * 0.25
   - projected_gravity (gravity dir in body frame, unit) — no scale
   - command[:3] * [2.0, 2.0, 0.25]   (e.g. [0.5,0,0] for forward walk → [1.0,0,0])
   - (dof_pos - default_dof_pos) * 1.0     # joints in URDF order, §2.4
   - dof_vel * 0.05
   - last_action (previous raw policy output; zeros on first step)
   - clip the whole vector to [-100, 100].
2. **Run the policy** → `action` (12,), clip to [-100, 100], keep as `last_action`.
3. **Compute the PD target**:
   ```
   target_position[j] = default_dof_pos[j] + action_scale * action[j]
                      = default_dof_pos[j] + 0.25 * action[j]
   ```
4. **Feed Nuka's PD drive** with `target_position`, `target_velocity = 0`,
   `Kp = 20.0`, `Kd = 0.5` (uniform across all 12 joints), and let Nuka clamp by the
   URDF effort limits (§2.6). Hold this target for 4 physics steps (dt=0.005) before
   the next policy query.

**Joint-order guard:** the obs you feed in (steps for dof_pos/dof_vel) and the
action you read out MUST be in the §2.4 URDF order [FL,FR,RL,RR × hip,thigh,calf].
If Nuka's Go2 articulation enumerates joints differently, compute the permutation
once and apply it symmetrically (obs in, action out, default_dof_pos, Kp/Kd).

**Frame guard:** lin/ang velocity in the obs are in the BASE/BODY frame
(`quat_rotate_inverse(base_quat, world_vel)`), and `projected_gravity` is the world
gravity unit vector rotated into the body frame. Provide them in body frame.

### Contingency — if a 45-dim deploy/community Go2 policy is substituted later
This repo's go2 training policy is 48-dim WITH base_lin_vel, which is FINE for Nuka
because Nuka is a sim and can supply ground-truth base_lin_vel. If instead a
deploy-style policy (real-robot, base_lin_vel unobservable) is used, it will be
**45-dim**: simply DROP the first 3 elements (`base_lin_vel`) and keep the remaining
order/scales identical (ang_vel, gravity, commands, dof_pos, dof_vel, actions). Then
re-confirm the new policy's scales/joint-order against ITS own config — do not assume
they match this repo's.

---

## 4. Cross-reference: unitree_rl_lab (IsaacLab) Go2 — for awareness only

`unitree_rl_lab` is a DIFFERENT framework with DIFFERENT conventions. It ships no go2
policy, so it is not our contract, but noting the deltas prevents accidental mixing:
- Policy obs group (`velocity_env_cfg.py`, order): base_ang_vel(scale 0.2),
  projected_gravity, velocity_commands, joint_pos_rel, joint_vel_rel(scale 0.05),
  last_action → **45-dim, NO base_lin_vel** (it lives only in the critic group).
- action: `JointPositionActionCfg(scale=0.25, use_default_offset=True)` →
  same target law `default + 0.25*action`.
- ang_vel scale **0.2** (vs 0.25 in legged_gym) — DIFFERENT. dof_vel scale 0.05 (same).
- decimation=4, sim.dt=0.005 → 50 Hz (same).
- IsaacLab joint order is its own (alphabetical-ish / asset-defined), generally
  **NOT** the URDF order above — do not reuse §2.4 for an IsaacLab policy.
Treat rl_lab values as a separate contract; they are only relevant if the controller
chooses the IsaacLab path.

---

## 5. Open items / things NOT found (so nothing is silently guessed)

- **No pretrained Go2 policy** in either repo, no release, no tag, no download script
  (§0) — the harness needs a policy sourced/trained separately.
- **torch I/O verification skipped.** A `torch` import was momentarily available early
  (reported 2.12.0+cu130, likely the `nuka-v03` conda env mid-install) but was not
  importable by the time of the I/O check (`ModuleNotFoundError: No module named
  'torch'` from `/root/miniconda3/envs/nuka-v03/bin/python` and all other
  interpreters). Per task guidance the config is the source of truth, and a go2
  policy would be exported separately with its own signature anyway, so loading
  g1/h1 `.pt` would not validate go2. Re-run the I/O check once the policy exists:
  `m = torch.jit.load(PATH); m(torch.zeros(1,48))` should return shape `(1,12)`.
- **torque_limits exact source:** the final clip uses per-joint `torque_limits`
  derived from the URDF `effort` attributes (§2.6 values). legged_gym multiplies by
  `soft_torque_limit` (go2 cfg leaves it at base default 1.0 — i.e. full URDF effort).
