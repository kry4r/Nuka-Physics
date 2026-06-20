#!/usr/bin/env python3
"""[demo] Dump a DETERMINISTIC GO2T-v1 trajectory for 20 Go2 dogs co-resident as
20 ARTICULATIONS in ONE nk::World env on ONE shared baked composite heightfield,
with REAL general-path dog-dog + foot-terrain collision (no penetration).

WHY THIS, NOT "R SEPARATE ENVS"
-------------------------------
go2_dump_terrain_trajectory.py rolls R dogs in R SEPARATE envs (each dog "各跑各的"
-- no dog can ever touch another, the collision engine never sees a dog-dog pair).
The owner mandate is the OPPOSITE: ALL 20 dogs live in ONE world env as 20 separate
articulations, so the ONE general contact pipeline (LBVH broadphase -> cvx GJK/EPA
narrowphase -> mixed-island solve) resolves both foot<->terrain AND dog<->dog
contacts on the SAME unified per-env contact buffer -- a robot body is just a
physics body, there is no "dog-dog" special case. Dogs placed close enough to come
within range PHYSICALLY push apart (verified, not assumed).

THE WORLD (ONE env, 20 articulations)
-------------------------------------
``nuka.World.create_from_scene(dev, go2_locomotion.usda, env_count=1,
instance_count=20, ...)`` composes the floating-base go2 scene with itself 20x
(``nuka::scene::Compose``) -> the cook emits 20 SEPARATE articulations co-resident
in one env (articulation_count==20). The buffers are then [1 env, 20*13 links]:
ARTICULATION_LINK_POSE/JOINT_POSITION/JOINT_VELOCITY/LINK_VELOCITY/DRIVE_TARGET are
(1, 260, ...) and BASE_POSE is (1, 20, 7) -- ONE root pose per dog. Dog d owns the
link block [d*13, (d+1)*13). A static composite heightfield collidable
(contact_family=1, type 4) is baked over the whole 20-dog spread so the feet
physically collide with the terrain on the general path.

CONTACT BUDGET (the general fix that made 20-dog ONE-env construct)
------------------------------------------------------------------
The C-ABI heightfield branch (src/c_abi/world.cpp) used to HARDCODE
cap.max_contacts_per_env=32 -- correct for ONE dog, FAR too small for 20 (20 dogs
* per-link collidables + dog-dog candidate pairs needs thousands of rows). That 32
was itself a single-dog shortcut; it is now derived GENERALLY from the cooked
collidable count: max_contacts_per_env = bodies_per_env * 4 (candidate pairs per
collidable), max_rows_per_env = that * kPairDrivenRowsPerSlot -- the SAME rule the
cook + broadphase use (tests/scenario/multi_dog_terrain_scale.cpp). Rebuild
``cmake --build build-cuda128 --target nuka`` after editing the C-ABI.

CONTROL (closed-loop, per dog)
------------------------------
Each control step builds the SAME 235-dim height-scan obs (48 proprio + 187 scan)
PER DOG by slicing the (1,260,..) buffers into 20 13-link blocks, batches all 20
obs through the trained height-scan policy (deterministic mean) -> 20x12 actions ->
writes each dog's DRIVE_TARGET[1:13] block via the structural joint permutation.
Step (decimation 4). Records all 20 dogs' ARTICULATION_LINK_POSE + DRIVE_TARGET.

OUTPUT: the EXACT GO2T-v1 binary format go2_dump_terrain_trajectory.py::_write_go2t
writes (R=20), so examples/demo/go2_terrain_demo.cpp --bin <out> replays it.

Run:
  export CUDA_VISIBLE_DEVICES=0
  export LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64
  /root/miniconda3/envs/nuka-v03/bin/python examples/demo/go2_dump_terrain_multidog.py \
      --checkpoint out/go2_terrain_hs/nn/last_go2_terrain_hs_ep_1500_rew_10.463228.pth \
      --seconds 14 --out out/m10/go2_terrain_20dog.bin
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np
import torch
import yaml

import go2_dump_terrain_trajectory as ST  # reuse spawn/dump/analyze/build_player

# Mirror the single-dog dump's constants (anti-drift: keep equal).
T_FLAT, T_STAIRS, T_PIT, T_BOXES, T_COMPOSITE = 0, 1, 2, 3, 4
NOMINAL_BASE_Z = ST.NOMINAL_BASE_Z
GO2_BLC = 13          # links/dofs per dog (root + 12 leg joints)
DECIMATION = 4
SIM_DT = 0.005


def _compact_spread_spawn(R: int, seed: int):
    """Place R dogs ONE-per-cell on a COMPACT cluster of mixed stairs/pit/box cells
    so the pack frames prominently while every dog traverses a real feature: stairs
    dogs climb, pit dogs descend into the recess, box dogs cross the bumps.

    Cells come from a tight window picked to contain pit + stairs + box; dogs are
    placed nearest-origin-first so the cluster is the smallest that holds R distinct
    feature cells, then seeded headings carry each dog into its feature. No two dogs
    share a cell -> spawn separation is the 5 m cell pitch (no clip), upright gait.
    Returns parallel lists (xs, ys, yaws, kinds, labels) length R."""
    rng = np.random.RandomState(seed)
    # Centre on the densest mixed-feature patch (this cell holds stairs/box with a
    # pit one cell away) and rank feature cells by distance to it, so the chosen R
    # form the SMALLEST bounding cluster -> a tight pack that frames prominently.
    cx0, cy0 = 0.0, -1.0
    cells = []
    for cx in range(-3, 3):
        for cy in range(-3, 2):
            st = ST._composite_subtype(cx, cy)
            if st in (T_STAIRS, T_PIT, T_BOXES):
                wx = (cx + 0.5) * ST.COMPOSITE_CELL
                wy = (cy + 0.5) * ST.COMPOSITE_CELL
                d2 = (cx - cx0) ** 2 + (cy - cy0) ** 2
                cells.append((cx, cy, wx, wy, st, d2))
    cells.sort(key=lambda c: c[5])
    # Take the R nearest cells, then guarantee >=2 pit + >=1 stairs by swapping the
    # farthest over-represented cell for the NEAREST missing-type cell, so the clip
    # always shows descent and climb without inflating the cluster.
    chosen = cells[:R]

    def count(t):
        return sum(1 for c in chosen if c[4] == t)

    for need, want_n in ((T_PIT, 2), (T_STAIRS, 1)):
        while count(need) < want_n:
            cand = next((c for c in cells if c[4] == need and c not in chosen), None)
            if cand is None:
                break
            drop = None
            for c in sorted(chosen, key=lambda c: -c[5]):
                if count(c[4]) > 1 and c[4] != need:
                    drop = c
                    break
            if drop is None:
                break
            chosen[chosen.index(drop)] = cand

    xs, ys, yaws, kinds, labels = [], [], [], [], []
    kind_name = {T_STAIRS: "stairs", T_PIT: "pit", T_BOXES: "boxes"}
    for (cx, cy, wx, wy, st, _r2) in chosen:
        ang = 2.0 * math.pi * rng.uniform(0.0, 1.0)
        # Spawn every dog on the seamless FLAT border just OUTSIDE its feature ring,
        # heading inward, so it settles a confident gait before climbing/descending
        # (a feature-slope spawn tips a cold-started dog).
        if st == T_STAIRS:
            ar = 2.3
            jit = 0.22
        elif st == T_PIT:
            ar = ST.FEATURE_HALF + 0.30
            jit = 0.18
        else:
            ar = ST.FEATURE_HALF + 0.40
            jit = 0.18
        sx = wx + ar * math.cos(ang)
        sy = wy + ar * math.sin(ang)
        yaw = math.atan2(wy - sy, wx - sx) + rng.uniform(-jit, jit)
        xs.append(sx); ys.append(sy); yaws.append(yaw); kinds.append(st)
        labels.append(f"{kind_name[st]} cell({cx:+d},{cy:+d})")
    return xs[:R], ys[:R], yaws[:R], kinds[:R], labels[:R]


def _spread_spawn(R: int, seed: int):
    """Per-dog spawn on the ONE composite field: dogs are PAIRED in clusters that
    converge so several pairs come within dog-dog collision range DURING the walk
    (the owner WANTS visible real dog-dog collision), each on a MIX of
    stairs/pit/box cells.

    Strategy: take the single-dog dump's deterministic composite placement
    (_composite_spawn -- one dog per 5 m cell, mixed sub-types, headings aimed into
    each feature) and place a PAIR of dogs per cell, each ~1.5 m from the cell centre
    on OPPOSITE sides, both HEADING toward the centre. A go2 is ~0.6 m long, so at
    1.5 m the dogs spawn cleanly separated and can establish a confident gait FIRST;
    walking inward (steady +x command) then drives them TOWARD each other across the
    feature, so they MEET and collide near the cell centre a few seconds in -- the
    push-apart is the general-path two-way contact, not a spawn-time pile-up that
    tangles legs and collapses the policy. Different cells stay 5 m apart so the pack
    reads as clusters on the terrain. Returns (xs, ys, yaws, kinds, labels)."""
    rng = np.random.RandomState(seed + 101)
    # One CELL per pair of dogs; ask for ceil(R/2) mixed-type cells.
    per_cell = 2
    n_cells = max(2, math.ceil(R / per_cell))
    bxs, bys, byaws, bkinds, blabels = ST._composite_spawn(n_cells, seed)

    xs, ys, yaws, kinds, labels = [], [], [], [], []
    ci = 0
    while len(xs) < R:
        cx, cy = bxs[ci % n_cells], bys[ci % n_cells]
        kind = bkinds[ci % n_cells]
        base_yaw = byaws[ci % n_cells]
        lab = blabels[ci % n_cells]
        # The cell centre this pair walks toward (the base spawn sits at the feature
        # ring; the centre is base_spawn + approach along base_yaw). Recover the
        # cell-centre direction from base_yaw and place the pair symmetrically across
        # it so they converge head-on-ish (yet offset, so they graze + bounce, not
        # perfectly collide and stall).
        n_here = min(per_cell, R - len(xs))
        approach_r = 1.55                 # >2 body lengths apart at spawn -> clean gait.
        for j in range(n_here):
            # opposite sides of the cell centre for the pair (j=0 base side, j=1 across).
            side = 0.0 if j == 0 else math.pi
            ang = base_yaw + math.pi + side + rng.uniform(-0.25, 0.25)
            px = cx + approach_r * math.cos(ang)
            py = cy + approach_r * math.sin(ang)
            # heading aimed at the cell centre (so the dog walks INWARD and the pair
            # converges) with a per-dog jitter + a slight lateral offset so they graze.
            yaw = math.atan2(cy - py, cx - px) + rng.uniform(-0.20, 0.20)
            xs.append(px); ys.append(py); yaws.append(yaw); kinds.append(kind)
            labels.append(f"{lab} pair{j}")
        ci += 1
    return xs[:R], ys[:R], yaws[:R], kinds[:R], labels[:R]


def _make_world(dev, R, nrow, ncol, cell, difficulty=1.0):
    """Create ONE env holding R floating-base go2 articulations on a baked composite
    heightfield (general contact path)."""
    import nuka
    go2 = "/root/Nuka-Physics/examples/scenes/go2_locomotion.usda"
    w = nuka.World.create_from_scene(
        dev, go2, 1, SIM_DT,
        instance_count=int(R), instance_spacing=2.5,
        contact_family=1, heightfield_terrain_type=T_COMPOSITE,
        heightfield_nrow=int(nrow), heightfield_ncol=int(ncol),
        heightfield_cell=float(cell),
        # Cook the SAME composite amplitudes the renderer draws (go2_terrain_demo
        # TilePreset * difficulty) so feet rest exactly on the drawn features; the
        # difficulty is mirrored into the GO2T meta the renderer reads.
        terrain_step_height=0.15 * difficulty, terrain_step_width=0.30,
        terrain_platform_width=1.0, terrain_grid_width=0.45,
        terrain_grid_height_max=0.15 * difficulty,
    )
    return w


def _sample_surface(world, xs, ys):
    """Surface z at each (x,y) via the SINGLE-source grid sampler (the EXACT cooked
    heightfield the foot kernel rests the feet on)."""
    xs_t = torch.as_tensor(xs, dtype=torch.float32).contiguous()
    ys_t = torch.as_tensor(ys, dtype=torch.float32).contiguous()
    surf = world.sample_terrain_height(xs_t, ys_t)
    return torch.as_tensor(surf, dtype=torch.float32)


def _terrain_surface_grid(world, wx, wy, dev):
    """(R, n_scan) surface z at world (wx,wy) [each (R,n_scan)] for the height scan
    -- batched through the grid sampler (the cooked heightfield)."""
    r, m = wx.shape
    xs_cpu = wx.reshape(-1).detach().cpu().to(torch.float32).contiguous()
    ys_cpu = wy.reshape(-1).detach().cpu().to(torch.float32).contiguous()
    surf = world.sample_terrain_height(xs_cpu, ys_cpu)
    return torch.as_tensor(surf, device=dev, dtype=torch.float32).view(r, m)


class MultiDogObs:
    """235-dim height-scan obs builder for R dogs co-resident in ONE env.

    Slices the (1, R*13, ..) engine buffers into R 13-link blocks and computes the
    SAME 48 proprio + 187 height-scan obs the training env builds, per dog. Reuses
    the structural joint permutation + obs math/scales from nuka.tasks.go2_obs."""

    def __init__(self, dev, world, R, scan_cfg):
        import nuka
        from nuka.tasks import go2_obs as G
        self.G = G
        self.world = world
        self.R = R
        self.dev = torch.device("cuda")
        # Joint permutation (discovered on a throwaway single-dog world; all R dogs
        # are byte-identical composed copies -> the same per-dog slot order).
        self.urdf_from_nuka_slot = torch.as_tensor(
            G.discover_joint_permutation(dev), dtype=torch.long, device=self.dev
        )
        nuka_slot_for_urdf = np.empty(12, dtype=np.int64)
        ufn = self.urdf_from_nuka_slot.cpu().numpy()
        for s in range(12):
            nuka_slot_for_urdf[ufn[s]] = s
        self.nuka_slot_for_urdf = torch.as_tensor(
            nuka_slot_for_urdf, dtype=torch.long, device=self.dev
        )
        self.default_angles = torch.as_tensor(
            G.DEFAULT_ANGLES, dtype=torch.float32, device=self.dev
        )
        self.cmd_scale = torch.as_tensor(
            G.CMD_SCALE, dtype=torch.float32, device=self.dev
        )
        self.force_limit_urdf = torch.as_tensor(
            G.FORCE_LIMIT_URDF, dtype=torch.float32, device=self.dev
        )
        # Zero-copy live views (1, R*13, ..).
        self._q = torch.from_dlpack(world.buffer_view(nuka.JOINT_POSITION))[0]      # (R*13,)
        self._qd = torch.from_dlpack(world.buffer_view(nuka.JOINT_VELOCITY))[0]
        self._pose = torch.from_dlpack(world.buffer_view(nuka.ARTICULATION_LINK_POSE))[0]  # (R*13,7)
        self._vel = torch.from_dlpack(world.buffer_view(nuka.LINK_VELOCITY))[0]     # (R*13,6)
        self._tgt = torch.from_dlpack(world.buffer_view(nuka.DRIVE_TARGET))[0]      # (R*13,)
        self._bp = torch.from_dlpack(world.buffer_view(nuka.BASE_POSE))[0]          # (R,7)
        # Per-dog link block start indices (root slot of each dog).
        self._root = torch.arange(R, device=self.dev) * GO2_BLC                     # (R,)
        # Per-dog actuated-joint slot indices (R,12) = root+1 .. root+12.
        leg = torch.arange(1, GO2_BLC, device=self.dev)                             # (12,)
        self._leg_idx = self._root[:, None] + leg[None, :]                          # (R,12)
        # Height-scan grid (n_scan,2) in base frame.
        self._scan_n = 0
        if scan_cfg is not None:
            x_min, x_max = scan_cfg["x_min"], scan_cfg["x_max"]
            y_min, y_max = scan_cfg["y_min"], scan_cfg["y_max"]
            res = scan_cfg["resolution"]
            self._scan_scale = scan_cfg["scale"]
            self._scan_clip = scan_cfg["clip"]
            nx = int(round((x_max - x_min) / res)) + 1
            ny = int(round((y_max - y_min) / res)) + 1
            self._scan_n = nx * ny
            gx = torch.linspace(x_min, x_max, nx, device=self.dev)
            gy = torch.linspace(y_min, y_max, ny, device=self.dev)
            mx, my = torch.meshgrid(gx, gy, indexing="ij")
            self._scan_grid = torch.stack((mx.reshape(-1), my.reshape(-1)), dim=1).contiguous()

    def apply_pd_gains(self):
        import nuka
        kp = torch.from_dlpack(self.world.buffer_view(nuka.DRIVE_STIFFNESS))[0]
        kd = torch.from_dlpack(self.world.buffer_view(nuka.DRIVE_DAMPING))[0]
        fl = torch.from_dlpack(self.world.buffer_view(nuka.DRIVE_FORCE_LIMIT))[0]
        # Set Kp/Kd/force-limit on every dog's 12 actuated slots.
        flat = self._leg_idx.reshape(-1)
        kp[flat] = self.G.KP
        kd[flat] = self.G.KD
        fl_urdf = self.force_limit_urdf[self.nuka_slot_for_urdf]                    # (12,)
        fl[flat] = fl_urdf.repeat(self.R)

    # -- per-dog reads (R, ..) in URDF order --
    def q_urdf(self):
        q = self._q[self._leg_idx]                                                  # (R,12) Nuka order
        return q.index_select(1, self.urdf_from_nuka_slot)

    def qd_urdf(self):
        qd = self._qd[self._leg_idx]
        return qd.index_select(1, self.urdf_from_nuka_slot)

    def base_quat_wxyz(self):
        return self._pose[self._root, 3:7]                                          # (R,4)

    def base_ang_vel(self):
        return self._vel[self._root, 0:3]                                           # (R,3)

    def base_lin_vel(self):
        return self._vel[self._root, 3:6]                                           # (R,3)

    def projected_gravity(self):
        return self.G.projected_gravity_body(self.base_quat_wxyz())

    def compute_obs48(self, command, last_action):
        obs = torch.cat((
            self.base_lin_vel() * self.G.S_LIN_VEL,
            self.base_ang_vel() * self.G.S_ANG_VEL,
            self.projected_gravity(),
            command * self.cmd_scale,
            (self.q_urdf() - self.default_angles) * self.G.S_DOF_POS,
            self.qd_urdf() * self.G.S_DOF_VEL,
            last_action,
        ), dim=-1)
        return obs.clamp_(-self.G.OBS_CLIP, self.G.OBS_CLIP)

    def height_scan(self):
        """(R, n_scan) legged_gym height scan around each dog's base, from the
        AUTHORITATIVE BASE_POSE view (xy/z/quat), composite surface sampled."""
        bp = self._bp                                                              # (R,7)
        base_x = bp[:, 0:1]; base_y = bp[:, 1:2]; base_z = bp[:, 2:3]
        qw, qx, qy, qz = bp[:, 3], bp[:, 4], bp[:, 5], bp[:, 6]
        yaw = torch.atan2(2.0 * (qw * qz + qx * qy),
                          1.0 - 2.0 * (qy * qy + qz * qz)).unsqueeze(1)            # (R,1)
        cy = torch.cos(yaw); sy = torch.sin(yaw)
        px = self._scan_grid[:, 0].unsqueeze(0)                                    # (1,n)
        py = self._scan_grid[:, 1].unsqueeze(0)
        wx = base_x + (px * cy - py * sy)                                          # (R,n)
        wy = base_y + (px * sy + py * cy)
        heights = _terrain_surface_grid(self.world, wx, wy, self.dev)              # (R,n)
        return ((base_z - 0.5 - heights).clamp(-self._scan_clip, self._scan_clip)
                * self._scan_scale).to(torch.float32)

    def compute_obs(self, command, last_action):
        obs = self.compute_obs48(command, last_action)
        if self._scan_n:
            obs = torch.cat([obs, self.height_scan()], dim=-1)
        return obs

    def write_action(self, action):
        """Map (R,12) URDF-order raw action -> PD target -> DRIVE_TARGET[leg slots]."""
        action = action.clamp(-self.G.ACTION_CLIP, self.G.ACTION_CLIP)
        target_urdf = self.default_angles + self.G.ACTION_SCALE * action           # (R,12)
        target_nuka = target_urdf.index_select(1, self.nuka_slot_for_urdf)         # (R,12)
        self._tgt[self._leg_idx] = target_nuka
        return action

    def set_spawn(self, xs, ys, yaws):
        """Write each dog's base root pose into BASE_POSE (the engine integrates from
        exactly this), z lifted to the composite surface + nominal stance, and reset
        each dog's joints to the legged_gym default + zero velocity."""
        import nuka
        surf = _sample_surface(self.world, xs, ys).to(self.dev)                    # (R,)
        base_z = surf + NOMINAL_BASE_Z
        xs_t = torch.as_tensor(xs, device=self.dev, dtype=torch.float32)
        ys_t = torch.as_tensor(ys, device=self.dev, dtype=torch.float32)
        yaw_t = torch.as_tensor(yaws, device=self.dev, dtype=torch.float32)
        half = yaw_t * 0.5
        self._bp[:, 0] = xs_t
        self._bp[:, 1] = ys_t
        self._bp[:, 2] = base_z
        self._bp[:, 3] = torch.cos(half)
        self._bp[:, 4] = 0.0
        self._bp[:, 5] = 0.0
        self._bp[:, 6] = torch.sin(half)
        # Reset joints to the legged_gym asymmetric default (the metastable cooked
        # crouch tips a cold dog; the default's hip splay gives roll margin), zero qd.
        default_nuka = self.default_angles[self.nuka_slot_for_urdf]                 # (12,)
        self._q[self._leg_idx] = default_nuka.repeat(self.R, 1)
        self._qd[self._leg_idx] = 0.0
        nuka.sync()


def _rollout(checkpoint, R, seed, seconds, cmd_x, nrow, ncol, cell, layout="pairs",
             difficulty=1.0):
    """Roll the height-scan policy on ONE env of R co-resident articulations and
    return the recorded trajectory + spawn meta + diagnostics."""
    import nuka
    from nuka.tasks import go2_obs as G
    torch.manual_seed(seed); np.random.seed(seed)

    spawn_fn = _compact_spread_spawn if layout == "spread" else _spread_spawn
    xs, ys, yaws, kinds, labels = spawn_fn(R, seed)
    print("-" * 78)
    print(f"[multidog] seed {seed}: {R} dogs as {R} ARTICULATIONS in ONE env on "
          f"ONE composite field; placement:")
    for r in range(R):
        print(f"   dog[{r:2d}] @world({xs[r]:+.2f},{ys[r]:+.2f}) "
              f"yaw={math.degrees(yaws[r]):+6.1f} deg -> {labels[r]}")
    print("-" * 78, flush=True)

    dev = nuka.Device.create(0)
    world = _make_world(dev, R, nrow, ncol, cell, difficulty)
    assert world.env_count == 1, f"expected ONE env, got {world.env_count}"
    blc = world.base_link_count
    assert blc == R * GO2_BLC, f"base_link_count {blc} != {R}*{GO2_BLC}"
    print(f"[multidog] ONE env, base_link_count={blc} = {R} dogs x {GO2_BLC} links "
          f"(20 ARTICULATIONS co-resident, NOT {R} envs)", flush=True)

    # Height-scan config -- match the training env defaults (235 = 48 + 187).
    scan_cfg = dict(x_min=-0.8, x_max=0.8, y_min=-0.5, y_max=0.5,
                    resolution=0.1, scale=5.0, clip=1.0)
    obs_b = MultiDogObs(dev, world, R, scan_cfg)
    obs_b.apply_pd_gains()
    assert obs_b._scan_n == 187, f"scan_n {obs_b._scan_n} != 187"

    # Seat the dogs at their composite spawn, then settle a beat so the feet load.
    obs_b.set_spawn(xs, ys, yaws)
    world.step_n(2); nuka.sync()
    obs_b.set_spawn(xs, ys, yaws)   # re-seat after the FK lag populates link poses.
    nuka.sync()

    # Steady forward command per dog (body +x); each dog's heading makes the
    # world-frame directions varied (clusters converge into their feature + jostle).
    command = torch.tensor([cmd_x, 0.0, 0.0], device=obs_b.dev).expand(R, 3).contiguous()
    last_action = torch.zeros(R, G.GO2_ACTION_DIM, device=obs_b.dev)

    # Build the policy player on a 235-obs / 12-action space (reuse the proven
    # builder via a tiny shim exposing single_observation_space).
    player = _build_player_235(checkpoint, R)

    pose_view = torch.from_dlpack(world.buffer_view(nuka.ARTICULATION_LINK_POSE))   # (1,R*13,7)
    drive_view = torch.from_dlpack(world.buffer_view(nuka.DRIVE_TARGET))            # (1,R*13)
    bp_view = torch.from_dlpack(world.buffer_view(nuka.BASE_POSE))                  # (1,R,7)

    ctrl_dt = SIM_DT * DECIMATION
    n_steps = int(round(seconds / ctrl_dt))

    obs = obs_b.compute_obs(command, last_action)
    assert obs.shape == (R, 235), f"obs shape {tuple(obs.shape)} != (R,235)"

    pose_records, drive_records = [], []
    falls = 0
    min_pair_global = float("inf")
    for k in range(n_steps):
        with torch.no_grad():
            action = player.get_action(obs, is_deterministic=True)
        last_action = obs_b.write_action(action)
        world.step_n(DECIMATION)
        nuka.sync()
        obs = obs_b.compute_obs(command, last_action)

        # Record this control step (slice the ONE-env buffers into R dogs).
        pose_np = pose_view[0].detach().cpu().numpy().reshape(R, GO2_BLC, 7).astype("<f4")
        drive_np = drive_view[0].detach().cpu().numpy().reshape(R, GO2_BLC).astype("<f4")
        pose_records.append(pose_np.copy())
        drive_records.append(drive_np.copy())

        # Live pairwise dog-dog min base distance (from BASE_POSE -- authoritative).
        bxy = bp_view[0, :, 0:2]                                                    # (R,2)
        d = torch.cdist(bxy, bxy)
        d = d + torch.eye(R, device=d.device) * 1e9
        cur_min = float(d.min().item())
        min_pair_global = min(min_pair_global, cur_min)

        # Diagnostics every ~75 control steps.
        if (k % 75) == 0 or k == n_steps - 1:
            bz = bp_view[0, :, 2].detach().cpu().numpy()
            falls = int((bz < -1.0).sum())  # a dog that free-fell to large negative.
            print(f"  seed {seed} t={k*ctrl_dt:5.2f}s ctrl{k:4d} "
                  f"base_z[min/mean/max]={bz.min():.3f}/{bz.mean():.3f}/{bz.max():.3f} "
                  f"min_pair_dist(live)={cur_min:.3f}m falls(z<-1)={falls}", flush=True)

    pose_arr = np.stack(pose_records)   # (N, R, 13, 7)
    drive_arr = np.stack(drive_records) # (N, R, 13)
    world.destroy()
    return {
        "pose_arr": pose_arr, "drive_arr": drive_arr, "N": pose_arr.shape[0],
        "BLC": GO2_BLC, "POSE_F": 7, "decimation": DECIMATION, "ctrl_dt": ctrl_dt,
        "kinds": kinds, "labels": labels, "xs": xs, "ys": ys, "yaws": yaws,
        "falls": falls, "seed": seed, "min_pair_global": min_pair_global,
        "difficulty": difficulty,
    }


def _build_player_235(checkpoint, R):
    """Build the rl_games player on a (235,) obs / (12,) action space from the
    height-scan training config -- the SAME logic as the single-dog dump's
    build_player, but without an env handle (we know the dims)."""
    import nuka.rl_games  # noqa: F401
    from nuka.tasks import go2_obs as G
    from gymnasium import spaces
    from rl_games.torch_runner import Runner
    with open(ST._CFG) as fh:
        raw = yaml.safe_load(fh)
    params = raw["params"]
    obs_space = spaces.Box(low=-G.OBS_CLIP, high=G.OBS_CLIP, shape=(235,), dtype=np.float32)
    act_space = spaces.Box(low=-G.ACTION_SPACE_LIMIT, high=G.ACTION_SPACE_LIMIT,
                           shape=(G.GO2_ACTION_DIM,), dtype=np.float32)
    params["config"]["env_info"] = {"observation_space": obs_space,
                                    "action_space": act_space, "agents": 1,
                                    "value_size": 1}
    runner = Runner()
    runner.load_config(params)
    player = runner.create_player()
    player.restore(checkpoint)
    player.has_batch_dimension = True
    player.is_deterministic = True
    player.is_tensor_obses = True
    return player


def _analyze_pushapart(pose_arr):
    """Detect dog-dog push-apart on the general path: for every frame find the
    closest pair; report the global min pairwise base distance, and confirm that
    when a pair gets within trunk-box contact range its distance subsequently GROWS
    (separation increases after the closest approach) -- i.e. they push apart, they
    do not interpenetrate. Returns a dict of stats."""
    base_xy = pose_arr[:, :, 0, 0:2]              # (N,R,2)
    base_z = pose_arr[:, :, 0, 2]                 # (N,R)
    N, R, _ = base_xy.shape
    d = np.linalg.norm(base_xy[:, :, None, :] - base_xy[:, None, :, :], axis=-1)  # (N,R,R)
    iu = np.triu_indices(R, k=1)
    pair = d[:, iu[0], iu[1]]                      # (N, n_pairs)
    per_frame_min = pair.min(axis=1)              # (N,)
    global_min = float(per_frame_min.min())
    argf = int(per_frame_min.argmin())
    # Trunk box half-extent ~0.10 m each -> centres < ~0.20 m == interpenetration.
    # Count frames where any pair is below the no-interpenetration distance.
    INTERPEN = 0.20
    n_interpen_frames = int((per_frame_min < INTERPEN).sum())
    # Push-apart evidence: track the single closest pair over the whole clip; its
    # distance at the closest-approach frame, and whether it grows afterward.
    pair_min_over_time = pair.min(axis=0)         # (n_pairs,) each pair's closest ever
    closest_pair = int(pair_min_over_time.argmin())
    cp_series = pair[:, closest_pair]             # (N,)
    cp_argmin = int(cp_series.argmin())
    cp_min = float(cp_series[cp_argmin])
    # separation a few frames after the closest approach (push-apart -> grows).
    look = min(N - 1, cp_argmin + 12)
    cp_after = float(cp_series[look])
    # Collision honesty: every pair that EVER closes within trunk-box contact range
    # (< 0.20 m) must (a) not stay deeply overlapped (sustained interpenetration) and
    # (b) PUSH APART afterward. A real hard head-on collision lets the two trunk
    # boxes briefly deep-overlap for a couple frames before the general two-way
    # contact forcefully separates them -- that transient is physical, sustained
    # interpenetration (passing through each other) is not. Count both.
    DEEP = 0.06                                   # < this ~= trunk boxes deeply overlap.
    n_pairs_in_range = 0
    all_recover = True
    max_deep_frames = 0
    for p in range(pair.shape[1]):
        s = pair[:, p]
        if float(s.min()) < INTERPEN:
            n_pairs_in_range += 1
            mn = int(s.argmin())
            max_deep_frames = max(max_deep_frames, int((s < DEEP).sum()))
            lk = min(N - 1, mn + 12)
            if not (float(s[lk]) > float(s[mn])):  # did NOT push apart.
                all_recover = False
    return {
        "global_min": global_min, "argf": argf, "n_interpen_frames": n_interpen_frames,
        "closest_pair": closest_pair, "cp_min": cp_min, "cp_after": cp_after,
        "n_pairs_in_range": n_pairs_in_range, "all_recover": all_recover,
        "max_deep_frames": max_deep_frames,
        "z_min": float(base_z.min()), "z_max": float(base_z.max()),
        "z_mean_final": float(base_z[-1].mean()),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seconds", type=float, default=14.0)
    ap.add_argument("--out", default="out/m10/go2_terrain_20dog.bin")
    ap.add_argument("--robots", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--command", type=float, default=0.45)
    # "spread" = one dog per compact mixed-feature cell (upright traversal, prominent
    # framing); "pairs" = converging pairs (visible dog-dog collision).
    ap.add_argument("--layout", choices=["spread", "pairs"], default="pairs")
    # Composite amplitude scale, cooked into physics AND written to the GO2T meta the
    # renderer reads, so the drawn features == the feet's surface at any difficulty.
    ap.add_argument("--difficulty", type=float, default=1.0)
    # The baked heightfield must Nyquist-sample the composite terrain it cooks:
    # the box sub-grid (0.45 m) and stair tread (0.40 m) feature widths alias
    # badly at 0.25 m, leaving the cooked physics surface up to ~0.1 m off the
    # analytic surface the renderer/obs use -> feet rest on the wrong height
    # (apparent 穿模). cell 0.05 divides both feature widths and resolves them;
    # 801x801 covers the +-20 m dog spread.
    ap.add_argument("--nrow", type=int, default=801)
    ap.add_argument("--ncol", type=int, default=801)
    ap.add_argument("--cell", type=float, default=0.05)
    args = ap.parse_args(argv)

    import nuka  # noqa: F401
    R = int(args.robots)
    print("=" * 78)
    print(f"[multidog] {R} dogs = {R} ARTICULATIONS in ONE nk::World env, "
          f"REAL dog-dog + foot-terrain collision (general path)")
    print("=" * 78, flush=True)

    roll = _rollout(args.checkpoint, R, args.seed, args.seconds, args.command,
                    args.nrow, args.ncol, args.cell, args.layout, args.difficulty)
    st = _analyze_pushapart(roll["pose_arr"])
    print("\n" + "=" * 78)
    print(f"[multidog] DIAGNOSTICS  N={roll['N']} ({roll['N']*roll['ctrl_dt']:.2f}s) R={R}")
    print(f"[multidog]   base_z over clip: min={st['z_min']:.3f} max={st['z_max']:.3f} "
          f"final_mean={st['z_mean_final']:.3f}  (SANE: no free-fall to large neg)")
    print(f"[multidog]   falls (z<-1, free-fell) = {roll['falls']}")
    print(f"[multidog]   GLOBAL min pairwise dog-dog base dist = {st['global_min']:.3f} m "
          f"(@frame {st['argf']})")
    print(f"[multidog]   frames with a pair < 0.20 m (interpenetration risk) = "
          f"{st['n_interpen_frames']}")
    print(f"[multidog]   pairs that closed within contact range (<0.20m) = "
          f"{st['n_pairs_in_range']}; closest pair {st['closest_pair']}: "
          f"{st['cp_min']:.3f} m -> {st['cp_after']:.3f} m ~12 frames later "
          f"({'PUSHED APART' if st['cp_after'] > st['cp_min'] else 'still close'})")
    print(f"[multidog]   max frames any pair deeply overlapped (<0.06m) = "
          f"{st['max_deep_frames']} (transient collision OK; sustained = 穿模); "
          f"all colliding pairs pushed apart = {st['all_recover']}")
    # PASS criteria: dogs stayed sane (no free-fall); pairs came within real contact
    # range (visible dog-dog collision); the contact is two-way (every colliding pair
    # PUSHES APART, none pass through); and no SUSTAINED interpenetration (a hard
    # head-on collision may deep-overlap a couple frames before the general two-way
    # solver separates them, but it must recover -- not pass through).
    sane = (st["z_min"] > -1.0) and (roll["falls"] == 0)
    came_close = st["global_min"] < 1.2
    real_collision = st["n_pairs_in_range"] >= 1 and st["all_recover"]
    no_sustained_interpen = st["max_deep_frames"] <= 8   # ~couple control steps only.
    print(f"[multidog]   VERDICT: sane(no free-fall)={sane}  visible-collision="
          f"{came_close}  two-way-push-apart={real_collision}  "
          f"no-sustained-interpenetration={no_sustained_interpen}")
    print("=" * 78, flush=True)

    ST._write_go2t(roll, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
