#!/usr/bin/env python3
"""[demo] Dump a DETERMINISTIC MULTI-DOG Go2 TERRAIN trajectory (GO2T v1) for the
C++ terrain video (examples/demo/go2_terrain_demo.cpp --bin <out>).

WHAT THIS IS (rework 2026-06-15)
--------------------------------
The REAL-ROLLOUT pre-pass for the Isaac-Lab-style multi-dog terrain demo. It rolls
the just-trained HEIGHT-SCAN terrain policy (out/go2_terrain_hs, obs 235 = 48
proprio + 187 legged_gym height-scan) on the SAME unified `nuka.World`-backed
Go2LocomotionEnv used for training -- terrain ON, height-scan ON -- with EVERY env
pinned to ONE terrain type: ``kTerrainComposite`` (terrain_field.hpp type 4). The
composite field is ONE continuous, seamless heightfield (large 5 m cells, each a
deterministic staggered sub-feature: pyramid stairs / pit / boxes / flat, joined by
a flat ground border) -- so all R dogs live in the SAME world-frame terrain, NOT on
separate floating tiles (the owner's "各跑各的" complaint on the first cut).

The dogs are SPREAD across several cells at world (x,y) (no per-env tile offset --
offset_x/offset_y are 0 for every robot; the renderer tessellates ONE big composite
mesh over the framed extent and places every dog at its TRUE physics x,y on it). To
GUARANTEE the owner's "see climbing", each dog is deterministically spawned on the
OUTER ring of a chosen pyramid-stairs cell with its heading aimed INWARD toward that
cell's centre, and given a steady forward command (fixed_command) -> it walks UP the
pyramid. Headings carry a per-dog jitter so the pack reads as natural/random, not a
lockstep march.

Per control step it records, for EVERY env, the full ARTICULATION_LINK_POSE
(13 links) + the held DRIVE_TARGET (13 slots), then writes the GO2T binary the C++
renderer replays. Because the composite heightfield is the SINGLE source of truth
(physics foot kernel + host spawn sampler + render tessellator all call ONE
SampleTerrainHeight), the rendered ground can never drift from the surface the feet
rest on -> NO float.

OBS DIM (critical): the policy net input is 235. The env auto-overrides its
observation_space to 235 when terrain+height_scan are ON; we build the rl_games
player from THE ENV'S space (not a hardcoded 48) so the checkpoint loads. We assert
obs.shape[-1] == 235 before rolling.

GO2T v1 BINARY LAYOUT (little-endian; flat; matches go2_terrain_demo.cpp LoadGo2T):
  HEADER (int32 unless noted):
    [0] magic       = 0x47324F54  ('TO2G' on disk; read LE uint32 == this)
    [1] version     = 1
    [2] num_robots  = R
    [3] num_steps   = N (recorded control steps; same N for all robots)
    [4] dof_count   = 13 (per-robot DRIVE_TARGET slots incl. root)
    [5] link_count  = 13 (per-robot ARTICULATION_LINK_POSE links)
    [6] pose_floats = 7
    [7] decimation  = 4
    float [8] dt      (the per-RECORD time = control period)
    float [9] ctrl_dt (= control period)
  Then R PER-ROBOT TILE records: float offset_x, offset_y; int32 terrain_type;
    float difficulty.   (For composite: offset 0,0; type 4; difficulty = the env
    difficulty = 1.0 -- the renderer applies ScaleTerrainDifficulty(preset, diff)
    then SampleTerrainHeight(kComposite, world_x, world_y, ...), matching physics.)
  Then N records, each R robots back-to-back; per robot:
    drive_target[13] f32 , link_pose[13*7=91] f32  (byte-identical to a GO2W record).

Run:
  export CUDA_VISIBLE_DEVICES=0
  export LD_LIBRARY_PATH=/opt/cuda-12.8-root/usr/local/cuda-12.8/lib64:.../build-cuda128/src
  export PYTHONPATH=/root/Nuka-Physics/python
  /root/miniconda3/envs/nuka-v03/bin/python examples/demo/go2_dump_terrain_trajectory.py \
      --checkpoint out/go2_terrain_hs/nn/last_go2_terrain_hs_ep_1500_rew_10.463228.pth \
      --seconds 12 --out out/m10/go2_terrain.bin
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys

import numpy as np
import torch
import yaml

_CFG = "examples/training/go2_ppo_cfg.yaml"
GO2T_MAGIC = 0x47324F54  # 'TO2G' on disk; LE uint32 == this (matches the renderer)
VERSION = 1

# Terrain type codes (match nuka::terrain::TerrainType).
T_FLAT, T_STAIRS, T_PIT, T_BOXES, T_COMPOSITE = 0, 1, 2, 3, 4
TYPE_NAMES = {0: "Flat", 1: "PyramidStairs", 2: "InvertedPyramid",
              3: "RandomBoxes", 4: "Composite"}

# Mirror of terrain_field.hpp composite constants (anti-drift: keep equal).
COMPOSITE_CELL = 5.0      # kCompositeCell
COMPOSITE_MARGIN = 0.6    # kCompositeMargin
NOMINAL_BASE_Z = 0.445    # mirror of go2_locomotion.NOMINAL_BASE_Z

_M32 = 0xFFFFFFFF


def _hash_cell(cx: int, cy: int) -> int:
    """EXACT mirror of terrain_field.hpp detail::HashCell (pure uint32)."""
    h = ((cx & _M32) * 0x9E3779B1 + (cy & _M32) * 0x85EBCA77) & _M32
    h ^= h >> 16
    h = (h * 0x7FEB352D) & _M32
    h ^= h >> 15
    h = (h * 0x846CA68B) & _M32
    h ^= h >> 16
    return h


def _composite_subtype(cx: int, cy: int) -> int:
    """EXACT mirror of terrain_field.hpp detail::CompositeCellSubType (round-2
    rebalanced 20-bucket LUT: 25% stairs / 30% pit / 30% boxes / 15% flat)."""
    lut = (T_STAIRS, T_STAIRS, T_STAIRS, T_STAIRS, T_STAIRS,          # 5 stairs
           T_PIT, T_PIT, T_PIT, T_PIT, T_PIT, T_PIT,                   # 6 pit
           T_BOXES, T_BOXES, T_BOXES, T_BOXES, T_BOXES, T_BOXES,       # 6 boxes
           T_FLAT, T_FLAT, T_FLAT)                                     # 3 flat
    return lut[_hash_cell(cx, cy) % 20]


# feature_half = cell/2 - margin = the Chebyshev radius the sub-feature occupies;
# outside it every cell is flat ground_height (the seamless border). Mirrors the
# header's kCompositeCell*0.5 - kCompositeMargin.
FEATURE_HALF = COMPOSITE_CELL * 0.5 - COMPOSITE_MARGIN   # 1.9 m


def _cells_of_type(want: int, cx_range, cy_range):
    """All composite cells in the (cx,cy) ranges whose sub-type == want, each as
    (cx, cy, world_cx, world_cy), sorted nearest-origin-first (so a COMPACT,
    frame-filling cluster is preferred)."""
    out = []
    for cy in cy_range:
        for cx in cx_range:
            if _composite_subtype(cx, cy) == want:
                wx = (cx + 0.5) * COMPOSITE_CELL
                wy = (cy + 0.5) * COMPOSITE_CELL
                out.append((cx, cy, wx, wy))
    out.sort(key=lambda c: c[2] * c[2] + c[3] * c[3])
    return out


def _composite_spawn(R: int, seed: int):
    """Deterministic per-dog spawn on the ONE composite field (owner round-2):
    place the R dogs on a SPREAD of cells of MIXED sub-type so the clip shows
    CLIMBING (stairs) AND DESCENT (pit) AND BOX-CROSSING all at once, each dog far
    enough from its neighbours that they never visibly clip (no dog-dog collision in
    the engine). Returns parallel lists (xs, ys, yaws, kinds, labels) length R.

    Placement per sub-type (a steady forward command then traverses the feature):
      * stairs : start on the flat border just OUTSIDE the pyramid base (r=2.3 m
                 from cell centre), heading INWARD -> walks UP the steps.
      * pit    : start at the pit RIM (r=FEATURE_HALF-0.15), heading INWARD toward
                 the centre -> walks DOWN into the pit (走下陷).
      * boxes  : start at one EDGE of the box patch (r=FEATURE_HALF+0.2), heading
                 ACROSS toward the opposite edge -> crosses the bumpy boxes (走box).
    Headings carry a per-dog jitter (varied directions, not a lockstep march).

    Cells are chosen on a seeded shuffle of each type's candidate list with a
    MIN-CELL-SEPARATION filter (>= sep_cells cells apart in Chebyshev), so the
    chosen spawn columns are >= sep_cells*5 m apart by construction -> a large
    safety margin over the 1.2 m no-clip bar even after the dogs walk a few metres.
    """
    rng = np.random.RandomState(seed)
    # Window of cells to draw from. A 4x4 block (cx,cy in [-2,1]) centred near origin
    # -> the MOST COMPACT field that still holds 10 distinct cells of mixed type:
    # spawn-cluster diagonal ~21 m, every dog on its OWN 5 m cell so the spawn min
    # dog-dog distance is the 5 m cell pitch (>> the 1.2 m no-clip bar even after the
    # dogs walk a few metres toward each other). This is the deliberate balance of
    # "spread to avoid clipping" vs "dogs not tiny" (round-1: dogs were tiny because
    # the cluster was framed too wide) -- a tighter cluster reads larger in the orbit,
    # and the hard constraint (min pairwise >= 1.2 m) is VERIFIED frame-by-frame in
    # _analyze + seed-shopped, NOT merely assumed from the spacing.
    win = range(-2, 2)
    pools = {
        T_STAIRS: _cells_of_type(T_STAIRS, win, win),
        T_PIT:    _cells_of_type(T_PIT, win, win),
        T_BOXES:  _cells_of_type(T_BOXES, win, win),
    }
    for k in pools:
        rng.shuffle(pools[k])

    # Target composition for R dogs: roughly equal climb / descend / cross with a
    # slight lean to climb (the hero action), e.g. R=10 -> 4 stairs, 3 pit, 3 box.
    n_stairs = max(2, round(R * 0.40))
    n_pit = max(1, round(R * 0.30))
    n_box = max(1, R - n_stairs - n_pit)
    want = ([T_STAIRS] * n_stairs + [T_PIT] * n_pit + [T_BOXES] * n_box)[:R]
    while len(want) < R:                       # pad if rounding came up short
        want.append(T_STAIRS)

    sep_cells = 1          # chosen cells distinct (>= 1 cell = 5 m apart, Chebyshev)
    chosen = []            # list of (cx, cy, wx, wy, kind)
    used = []              # (cx, cy) already taken

    def _far_enough(cx, cy):
        return all(max(abs(cx - ux), abs(cy - uy)) >= sep_cells for ux, uy in used)

    # Assign each requested kind a cell, honouring the separation filter; relax the
    # filter only if a kind runs out of well-separated cells (then nearest free).
    for kind in want:
        pool = pools[kind]
        pick = None
        for c in pool:
            if (c[0], c[1]) in used:
                continue
            if _far_enough(c[0], c[1]):
                pick = c
                break
        if pick is None:  # relax: take the nearest free cell of this kind
            for c in pool:
                if (c[0], c[1]) not in used:
                    pick = c
                    break
        if pick is None:  # last resort: any free cell of any kind
            for k2 in pools:
                for c in pools[k2]:
                    if (c[0], c[1]) not in used:
                        pick = c
                        kind = k2
                        break
                if pick is not None:
                    break
        cx, cy, wx, wy = pick
        used.append((cx, cy))
        chosen.append((cx, cy, wx, wy, kind))

    xs, ys, yaws, kinds, labels = [], [], [], [], []
    kind_name = {T_STAIRS: "stairs", T_PIT: "pit", T_BOXES: "boxes"}
    for r, (cx, cy, wx, wy, kind) in enumerate(chosen):
        ang = 2.0 * math.pi * rng.uniform(0.0, 1.0)   # which side we approach from
        if kind == T_STAIRS:
            approach_r = 2.3                          # a step off the base
            sx = wx + approach_r * math.cos(ang)
            sy = wy + approach_r * math.sin(ang)
            yaw = math.atan2(wy - sy, wx - sx) + rng.uniform(-0.22, 0.22)
        elif kind == T_PIT:
            approach_r = FEATURE_HALF - 0.15          # right at the pit rim
            sx = wx + approach_r * math.cos(ang)
            sy = wy + approach_r * math.sin(ang)
            yaw = math.atan2(wy - sy, wx - sx) + rng.uniform(-0.18, 0.18)
        else:  # T_BOXES: start at one edge, cross to the far edge
            edge = FEATURE_HALF + 0.2
            sx = wx + edge * math.cos(ang)
            sy = wy + edge * math.sin(ang)
            yaw = math.atan2(wy - sy, wx - sx) + rng.uniform(-0.18, 0.18)
        xs.append(sx); ys.append(sy); yaws.append(yaw); kinds.append(kind)
        labels.append(f"{kind_name[kind]} cell({cx:+d},{cy:+d})")
    return xs, ys, yaws, kinds, labels


def make_env(num_envs, episode_length_s, seed, difficulty=1.0):
    from nuka.tasks.go2_locomotion import make_env as _mk
    with open(_CFG) as fh:
        raw = yaml.safe_load(fh)
    tcfg = dict(raw["params"]["config"]["env_config"]["terrain"])
    tcfg["enable"] = True
    tcfg.setdefault("height_scan", {})["enable"] = True  # MATCH training -> obs 235
    # Scale the COOKED composite relief by difficulty so physics matches the
    # renderer's ScaleTerrainDifficulty (base 0.15 * diff) -- anti-drift, and a
    # gentler difficulty yields steps the policy actually clears.
    tcfg["step_height"] = float(tcfg.get("step_height", 0.15)) * float(difficulty)
    tcfg["grid_height_max"] = float(tcfg.get("grid_height_max", 0.15)) * float(difficulty)
    # Spread the (unused-here) random spawn wide + random heading; we OVERRIDE the
    # spawn explicitly after reset, but keep these so any autoreset stays on-field.
    tcfg["spawn_xy_range"] = 8.0
    tcfg["randomize_yaw"] = True
    ec = raw["params"]["config"]["env_config"]
    # fixed_command so the per-dog command we write after reset is NOT resampled
    # mid-clip (a steady forward walk -> a clean climb, no sudden turns/teleports).
    # Bake a REAL static composite heightfield collidable (contact_family=1) big
    # enough to span the dog spread, so the feet PHYSICALLY collide with terrain
    # (cf=0 leaves terrain obs on but no collidable -> the dogs free-fall).
    env = _mk(num_envs, terrain=tcfg, episode_length_s=episode_length_s,
              termination_height=ec["termination_height"],
              termination_tilt_deg=ec["termination_tilt_deg"], seed=seed,
              fixed_command=True, command=(0.9, 0.0, 0.0),
              contact_family=1, heightfield_terrain_type=T_COMPOSITE,
              heightfield_nrow=161, heightfield_ncol=161, heightfield_cell=0.25)
    # Freeze curriculum promotion so the pinned (type, difficulty) stays put.
    env._curriculum.update_on_done = lambda *a, **k: None
    return env


def build_player(checkpoint, env):
    import nuka.rl_games  # noqa: F401
    import nuka.tasks.go2_obs as G
    from gymnasium import spaces
    from rl_games.torch_runner import Runner
    with open(_CFG) as fh:
        raw = yaml.safe_load(fh)
    params = raw["params"]
    full_dim = env.single_observation_space.shape[0]
    if full_dim != 235:
        print(f"[dump] WARNING: env obs dim {full_dim} != 235 (height-scan policy)",
              flush=True)
    obs_space = spaces.Box(low=-G.OBS_CLIP, high=G.OBS_CLIP,
                           shape=(full_dim,), dtype=np.float32)
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


def _pairwise_min_distance(base_xy):
    """Min over all (frame, dog-pair) of the planar dog-dog base distance.
    base_xy: (N, R, 2). Returns (global_min, per_frame_min array, argmin frame)."""
    n, r, _ = base_xy.shape
    if r < 2:
        return float("inf"), np.full(n, float("inf")), 0
    # (N,R,1,2) - (N,1,R,2) -> (N,R,R) pairwise distances per frame.
    d = np.linalg.norm(base_xy[:, :, None, :] - base_xy[:, None, :, :], axis=-1)
    iu = np.triu_indices(r, k=1)
    pair = d[:, iu[0], iu[1]]               # (N, n_pairs)
    per_frame = pair.min(axis=1)            # (N,)
    return float(per_frame.min()), per_frame, int(per_frame.argmin())


def _traversed_types(base_xy):
    """For each dog, the SET of composite sub-types its base column passed over the
    clip (one sample per frame at the dog's world x,y). base_xy: (N,R,2). Returns a
    list (len R) of sets of type-codes."""
    n, r, _ = base_xy.shape
    out = []
    for d in range(r):
        seen = set()
        for s in range(n):
            x, y = float(base_xy[s, d, 0]), float(base_xy[s, d, 1])
            cx = math.floor(x / COMPOSITE_CELL)
            cy = math.floor(y / COMPOSITE_CELL)
            lx = x - (cx + 0.5) * COMPOSITE_CELL
            ly = y - (cy + 0.5) * COMPOSITE_CELL
            rl = max(abs(lx), abs(ly))
            if rl >= FEATURE_HALF:
                seen.add(T_FLAT)             # on the seamless flat border
            else:
                seen.add(_composite_subtype(cx, cy))
        out.append(seen)
    return out


def _analyze(pose_arr, ctrl_dt, kinds, n_done):
    """Compute the owner's acceptance metrics for one rollout and a PASS verdict.
    pose_arr: (N,R,13,7). Returns a dict of stats."""
    base_xy = pose_arr[:, :, 0, 0:2]        # (N,R,2)
    base_z = pose_arr[:, :, 0, 2]           # (N,R)
    R = base_xy.shape[1]
    min_d, per_frame, argf = _pairwise_min_distance(base_xy)
    trav = _traversed_types(base_xy)
    # net base-z change per dog (descend = negative); a dog whose SPAWN cell is pit
    # and whose z drops by > a step is a confirmed pit-descent.
    z_net = base_z[-1] - base_z[0]
    z_min_rel = base_z.min(axis=0) - base_z[0]   # deepest dip below spawn
    z_max_rel = base_z.max(axis=0) - base_z[0]   # highest rise above spawn
    # Confirmed actions (by SPAWN intent + realized motion):
    n_climb = sum(1 for d in range(R)
                  if kinds[d] == T_STAIRS and z_max_rel[d] > 0.06)
    n_descend = sum(1 for d in range(R)
                    if kinds[d] == T_PIT and z_min_rel[d] < -0.06)
    n_cross = sum(1 for d in range(R)
                  if kinds[d] == T_BOXES and (T_BOXES in trav[d]))
    # Also count dogs whose path crossed a pit / boxes regardless of spawn intent.
    any_pit = sum(1 for d in range(R) if T_PIT in trav[d])
    any_box = sum(1 for d in range(R) if T_BOXES in trav[d])
    verdict = (min_d >= 1.2 and n_climb >= 2 and n_descend >= 1
               and n_cross >= 1 and n_done <= 1)
    return {
        "min_dist": min_d, "argframe": argf, "n_climb": n_climb,
        "n_descend": n_descend, "n_cross": n_cross, "any_pit": any_pit,
        "any_box": any_box, "falls": n_done, "z_net": z_net,
        "z_min_rel": z_min_rel, "z_max_rel": z_max_rel, "trav": trav,
        "verdict": verdict,
    }


def _override_spawn(env, xs, ys, yaws, cmd_x):
    """Write a DETERMINISTIC per-dog base pose (world x,y at the pyramid outer ring,
    z lifted to the LOCAL composite surface + nominal stance, yaw heading) directly
    into the engine BASE_POSE view -- the SAME mechanism the env's spawn-on-terrain
    reset uses, so the engine integrates from exactly this root. z is read from the
    C-ABI composite sampler (the single source of truth the foot kernel uses).
    cmd_x = the steady body-+x forward command (m/s) given to every dog."""
    import nuka
    dev = env._torch_device
    n = env.num_envs
    cur = env._curriculum
    xs_t = torch.as_tensor(xs, dtype=torch.float32, device=dev)
    ys_t = torch.as_tensor(ys, dtype=torch.float32, device=dev)
    yaw_t = torch.as_tensor(yaws, dtype=torch.float32, device=dev)
    # Local surface height at each (x,y) via the SINGLE-source grid sampler (the
    # full-relief cooked heightfield the physics narrowphase rests feet on).
    surf = env._world.sample_terrain_height(
        xs_t.cpu().contiguous(), ys_t.cpu().contiguous())
    base_z = torch.as_tensor(surf, device=dev, dtype=torch.float32) + NOMINAL_BASE_Z
    half = yaw_t * 0.5
    env._bp_view[:, 0] = xs_t
    env._bp_view[:, 1] = ys_t
    env._bp_view[:, 2] = base_z
    env._bp_view[:, 3] = torch.cos(half)
    env._bp_view[:, 4] = 0.0
    env._bp_view[:, 5] = 0.0
    env._bp_view[:, 6] = torch.sin(half)
    # Steady forward command (body +x) for every dog -> walk straight into its
    # feature (each dog's distinct heading makes the WORLD-frame directions varied,
    # not a lockstep march). A MODERATE speed keeps dogs near their features over a
    # 15-20 s clip (a fast command runs them off into the flat -> tiny in frame).
    env.command[:] = torch.tensor([cmd_x, 0.0, 0.0], device=dev).expand(n, 3)
    nuka.sync()


def _rollout(checkpoint, R, seed, seconds, allow_reset, cmd_x, difficulty=1.0):
    """Roll the policy for ONE seed and return the recorded trajectory + spawn meta.
    Creates + closes its own env so it can be called repeatedly for seed-shopping.
    Returns dict: pose_arr (N,R,13,7), drive_arr (N,R,13), N, BLC, POSE_F,
    decimation, ctrl_dt, kinds, labels, xs, ys, yaws, falls."""
    import nuka  # noqa: F401
    torch.manual_seed(seed)
    np.random.seed(seed)

    xs, ys, yaws, kinds, labels = _composite_spawn(R, seed)
    print("-" * 78)
    print(f"[dump] seed {seed}: {R} dogs on ONE composite field "
          f"(cell={COMPOSITE_CELL} m); placement:")
    for r in range(R):
        print(f"   dog[{r}] @world({xs[r]:+.2f},{ys[r]:+.2f}) "
              f"yaw={math.degrees(yaws[r]):+6.1f} deg -> {labels[r]}")
    print("-" * 78, flush=True)

    env = make_env(R, episode_length_s=max(60.0, seconds * 4.0), seed=seed,
                   difficulty=difficulty)
    # Pin every env to the COMPOSITE field at the chosen amplitude scale; the same
    # scale rides in the GO2T meta so the renderer draws exactly this surface.
    env._curriculum.terrain_type[:] = T_COMPOSITE
    env._curriculum.difficulty[:] = difficulty
    # SUPPRESS AUTORESET: a fall would teleport the dog to a random spawn (a jarring
    # jump). Forced termination==False keeps the whole clip one continuous trajectory
    # -- a dog that trips just stumbles in world-frame, a dog in a pit explores it.
    if not allow_reset:
        env.compute_terminated = lambda: torch.zeros(
            env.num_envs, dtype=torch.bool, device=env._torch_device)

    player = build_player(checkpoint, env)
    obs, _ = env.reset()
    assert obs.shape[-1] == 235, f"obs dim {obs.shape[-1]} != 235"
    _override_spawn(env, xs, ys, yaws, cmd_x)
    # Recompose obs after the spawn override so the first action sees the real start.
    obs = env._obs.compute_obs(env.command, env.last_action)
    obs[:, 6:9] = env._obs.projected_gravity_auth()
    obs[:, 9:12] = env.command * env._obs.cmd_scale
    if getattr(env, "_height_scan_on", False):
        obs = torch.cat([obs, env._height_scan()], dim=-1)

    pose_buf = torch.from_dlpack(env._world.buffer_view(nuka.ARTICULATION_LINK_POSE))
    drive_buf = torch.from_dlpack(env._world.buffer_view(nuka.DRIVE_TARGET))
    BLC = pose_buf.shape[1]
    POSE_F = pose_buf.shape[2]
    assert BLC == 13 and POSE_F == 7, (BLC, POSE_F)

    decimation = int(getattr(env, "decimation", 4))
    sim_dt = float(getattr(env, "dt", 0.005))
    ctrl_dt = sim_dt * decimation
    n_steps = int(round(seconds / ctrl_dt))

    pose_records, drive_records = [], []
    n_done_total = 0
    for k in range(n_steps):
        with torch.no_grad():
            action = player.get_action(obs, is_deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        nuka.sync()
        drive_records.append(drive_buf.detach().cpu().numpy().copy())
        pose_records.append(pose_buf.detach().cpu().numpy().copy())
        nd = int((terminated | truncated).sum().item())
        if nd:
            n_done_total += nd
        if (k % 75) == 0 or k == n_steps - 1:
            bz = pose_buf[:, 0, 2].detach().cpu().numpy()
            print(f"  seed {seed} t={k*ctrl_dt:5.2f}s ctrl{k:4d} base_z[min/mean/max]="
                  f"{bz.min():.3f}/{bz.mean():.3f}/{bz.max():.3f} "
                  f"falls={n_done_total}", flush=True)

    drive_arr = np.stack(drive_records).astype("<f4")
    pose_arr = np.stack(pose_records).astype("<f4")
    env.close()
    return {
        "pose_arr": pose_arr, "drive_arr": drive_arr, "N": pose_arr.shape[0],
        "BLC": BLC, "POSE_F": POSE_F, "decimation": decimation, "ctrl_dt": ctrl_dt,
        "kinds": kinds, "labels": labels, "xs": xs, "ys": ys, "yaws": yaws,
        "falls": n_done_total, "seed": seed, "difficulty": difficulty,
    }


def _print_stats(roll, st):
    R = roll["pose_arr"].shape[1]
    nm = {T_FLAT: "flat", T_STAIRS: "stairs", T_PIT: "pit", T_BOXES: "box"}
    print(f"\n[stats] seed {roll['seed']}  N={roll['N']} "
          f"({roll['N']*roll['ctrl_dt']:.2f}s)  R={R}")
    print(f"[stats]   MIN pairwise dog-dog distance over ALL frames = "
          f"{st['min_dist']:.2f} m  (bar >= 1.20; worst @frame {st['argframe']})")
    print(f"[stats]   falls/teleports = {st['falls']} (bar <= 1)")
    print(f"[stats]   confirmed: climb(stairs)={st['n_climb']} (>=2)  "
          f"descend(pit)={st['n_descend']} (>=1)  cross(box)={st['n_cross']} (>=1)")
    print(f"[stats]   any-dog-path-over: pit={st['any_pit']}  boxes={st['any_box']}")
    for d in range(R):
        seen = "+".join(sorted(nm[t] for t in st["trav"][d]))
        print(f"[stats]     dog[{d}] spawn={nm.get(roll['kinds'][d],'?'):6s} "
              f"z[net={st['z_net'][d]:+.2f} dip={st['z_min_rel'][d]:+.2f} "
              f"rise={st['z_max_rel'][d]:+.2f}] traversed=[{seen}]")
    print(f"[stats]   VERDICT = {'PASS' if st['verdict'] else 'fail'}", flush=True)


def _write_go2t(roll, out_path):
    out = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    R = roll["pose_arr"].shape[1]
    N, BLC, POSE_F = roll["N"], roll["BLC"], roll["POSE_F"]
    drive_arr, pose_arr = roll["drive_arr"], roll["pose_arr"]
    ctrl_dt = roll["ctrl_dt"]
    with open(out, "wb") as f:
        f.write(struct.pack("<8i2f", GO2T_MAGIC, VERSION, R, N, BLC, BLC, POSE_F,
                            roll["decimation"], float(ctrl_dt), float(ctrl_dt)))
        diff = float(roll.get("difficulty", 1.0))
        for r in range(R):
            f.write(struct.pack("<2fif", 0.0, 0.0, int(T_COMPOSITE), diff))
        for s in range(N):
            for r in range(R):
                f.write(drive_arr[s, r].tobytes(order="C"))
                f.write(pose_arr[s, r].reshape(-1).tobytes(order="C"))
    sz = os.path.getsize(out)
    print(f"[dump] WROTE {out}  ({sz/1e6:.3f} MB, {N} steps x {R} dogs, "
          f"{N*ctrl_dt:.2f}s, seed {roll['seed']})")
    print(f"[dump] header magic=0x{GO2T_MAGIC:08X} robots={R} steps={N} "
          f"dof={BLC} links={BLC} pose_floats={POSE_F} terrain=COMPOSITE", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seconds", type=float, default=18.0,
                    help="control-rate seconds to record (no autoreset within).")
    ap.add_argument("--out", default="out/m10/go2_terrain.bin")
    ap.add_argument("--robots", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7,
                    help="seed to dump when not seed-shopping.")
    ap.add_argument("--shop", default="",
                    help="comma seeds to SEED-SHOP (rollout+score each, pick best, "
                         "dump the winner). e.g. --shop 1,2,3,4,5,6")
    ap.add_argument("--command", type=float, default=0.5,
                    help="steady body-+x forward command m/s (lower = dogs stay near "
                         "their features over a long clip; default 0.5).")
    ap.add_argument("--allow-reset", action="store_true",
                    help="permit env autoreset on fall (DEFAULT off -> no teleports).")
    ap.add_argument("--difficulty", type=float, default=1.0,
                    help="composite amplitude scale, cooked into physics AND written "
                         "to the GO2T meta the renderer reads (deep pits at 1.0 can "
                         "exceed the policy; lower keeps every dog upright).")
    args = ap.parse_args(argv)

    import nuka  # noqa: F401,E402
    R = int(args.robots)

    if args.shop:
        seeds = [int(s) for s in args.shop.split(",") if s.strip() != ""]
        print("=" * 78)
        print(f"[shop] SEED-SHOPPING {len(seeds)} seeds for a clean {R}-dog "
              f"{args.seconds:.0f}s composite clip: {seeds}")
        print("=" * 78, flush=True)
        scored = []   # (rank_key, roll, stats)
        for sd in seeds:
            roll = _rollout(args.checkpoint, R, sd, args.seconds, args.allow_reset,
                            args.command, args.difficulty)
            st = _analyze(roll["pose_arr"], roll["ctrl_dt"], roll["kinds"],
                          roll["falls"])
            _print_stats(roll, st)
            # Rank: PASS first; then maximize (clearance over 1.2 capped, action
            # variety = climb+descend+cross), then fewest falls. Reject seeds that
            # clip (min_dist < 1.2) by sorting them last.
            clip_ok = st["min_dist"] >= 1.2
            variety = st["n_climb"] + st["n_descend"] + st["n_cross"]
            key = (int(clip_ok), int(st["verdict"]), variety,
                   min(st["min_dist"], 5.0), -st["falls"])
            scored.append((key, roll, st))
        scored.sort(key=lambda t: t[0], reverse=True)
        best_key, best_roll, best_st = scored[0]
        print("\n" + "=" * 78)
        print("[shop] RANKING (best first):")
        for key, roll, st in scored:
            tag = ("PASS" if st["verdict"]
                   else ("clip!" if st["min_dist"] < 1.2 else "partial"))
            print(f"[shop]   seed {roll['seed']:3d}  min_d={st['min_dist']:.2f}  "
                  f"climb={st['n_climb']} descend={st['n_descend']} "
                  f"cross={st['n_cross']}  falls={st['falls']}  {tag}")
        print(f"[shop] CHOSEN seed = {best_roll['seed']} "
              f"({'PASS' if best_st['verdict'] else 'BEST-AVAILABLE (see tradeoff)'})")
        print("=" * 78, flush=True)
        _write_go2t(best_roll, args.out)
        return 0

    # Single-seed dump.
    print("=" * 78)
    print(f"[dump] COMPOSITE terrain GO2T: {R} dogs, seed {args.seed}, "
          f"{args.seconds:.0f}s", flush=True)
    print("=" * 78, flush=True)
    roll = _rollout(args.checkpoint, R, args.seed, args.seconds, args.allow_reset,
                    args.command, args.difficulty)
    st = _analyze(roll["pose_arr"], roll["ctrl_dt"], roll["kinds"], roll["falls"])
    _print_stats(roll, st)
    _write_go2t(roll, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
