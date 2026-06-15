"""Python MIRROR of ``src/sensor/terrain/terrain_field.hpp`` -- the per-type
terrain CENTER height + a Go2 terrain CURRICULUM scheduler.

WHY THIS FILE EXISTS
--------------------
The engine procedural terrain (the FusedFoot foot kernel + the future render
tessellator) computes the surface height from the SINGLE source of truth
``src/sensor/terrain/terrain_field.hpp`` (``SampleTerrainHeight(type,x,y,p)`` +
``ScaleTerrainDifficulty(p,diff)``). The RL spawn-on-terrain reset needs to know
the LOCAL surface height at the Go2 spawn column so it can place the base ABOVE
the platform/pit/box instead of inside it (the Phase-2a "spawn below a raised
platform -> first-step penetration -> Baumgarte launch to z~=18" bug).

The Go2 spawns at the tile CENTER (world x,y ~= 0,0), so we only need the
CENTER-column height per terrain type -- a tiny closed-form subset of the full
``SampleTerrainHeight``. This module reproduces EXACTLY that subset (plus the
difficulty scaling) in pure Python/torch so the reset can run on-GPU per env.

>>> ANTI-DRIFT CONTRACT <<<
This module is a HAND-MAINTAINED MIRROR of the C++ header. The two MUST stay in
sync. If you ever change the terrain math in ``terrain_field.hpp`` -- the
``kPyramidRings`` constant, the platform-top / pit-bottom formula, the
RandomBoxes center cell-hash, or the difficulty scaling semantics -- you MUST
update THIS file in the same change. There is no automated cross-language check
(the header is host/device C++; this is torch); the only guard is this notice +
the ``KPYRAMID_RINGS`` assert below mirroring ``kPyramidRings``.

MIRRORED CONSTANTS / FORMULAS (verbatim from terrain_field.hpp):
  * ``kPyramidRings = 8``  (lines ~179).
  * Flat (type 0):             h = ground_height.
  * PyramidStairs (type 1):    platform_top = ground + kPyramidRings*step_height;
                               at center (r<=half_plat) h = platform_top.
  * InvertedPyramid (type 2):  pit_bottom = ground - kPyramidRings*step_height;
                               at center h = pit_bottom.
  * RandomBoxes (type 3):      cell = floor(0/grid_width)=0 -> hash(0,0);
                               u = HashCellUniform01(0,0);
                               h = ground + u*grid_height_max.
  * ScaleTerrainDifficulty:    step_height *= diff; grid_height_max *= diff
                               (vertical magnitude ONLY; ground/horizontal
                               geometry unchanged). diff==1 -> unchanged;
                               diff==0 -> flat.

RESOLVED (random-XY spawn + free-roam, 随地走): the Go2 now spawns at a RANDOM
(x,y) within the terrain tile (legged_gym terrain_origins jitter), so the reset
needs the FULL arbitrary-(x,y) surface height -- the ring index / per-cell hash
for any column, not just the center. Rather than port that into torch here (and
risk drift from the header), the FULL height is now served by the BATCHED C-ABI
sampler ``nuka.terrain_sample_height_batch`` (src/c_abi/terrain.cpp), which calls
the C++ header's ``SampleTerrainHeight`` + ``ScaleTerrainDifficulty`` DIRECTLY --
so the header itself IS the single source for arbitrary columns and there is no
parallel python full-sampler to drift. ``center_height`` below is kept only as a
LEGACY / DIAGNOSTIC convenience (the center-column height per type); the live
spawn-on-terrain reset uses the C-ABI sampler at the random column. The
ANTI-DRIFT CONTRACT above still binds ``center_height`` to the header's center
formula.
"""

from __future__ import annotations

import torch

# Mirror of terrain_field.hpp ``enum TerrainType`` (the type CODES).
TERRAIN_FLAT = 0
TERRAIN_PYRAMID_STAIRS = 1
TERRAIN_INVERTED_PYRAMID = 2
TERRAIN_RANDOM_BOXES = 3

# Mirror of terrain_field.hpp ``inline constexpr int kPyramidRings = 8``.
# >>> ANTI-DRIFT: keep equal to terrain_field.hpp::kPyramidRings <<<
KPYRAMID_RINGS = 8


# ---------------------------------------------------------------------------
# RandomBoxes center cell hash -- the EXACT mirror of terrain_field.hpp
# detail::HashCell(0,0) -> HashCellUniform01(0,0). cx=cy=floor(0/grid_width)=0,
# so HashCell(0,0) = 0 ^>>... = 0 (every term is 0*const). The MurmurHash3
# finalizer of 0 is 0, so u(0,0) == 0.0 -> the center box height is EXACTLY
# ground_height for ALL difficulties. We still compute it through the mirrored
# finalizer (not just hardcode 0) so that if the header's hash seed/center cell
# ever changes, the fix lands here too.
# ---------------------------------------------------------------------------
def _hash_cell_uniform01(cx: int, cy: int) -> float:
    """Mirror of terrain_field.hpp detail::HashCellUniform01 (pure uint32)."""
    M32 = 0xFFFFFFFF
    h = (cx * 0x9E3779B1 + cy * 0x85EBCA77) & M32
    h ^= h >> 16
    h = (h * 0x7FEB352D) & M32
    h ^= h >> 15
    h = (h * 0x846CA68B) & M32
    h ^= h >> 16
    return float(h >> 8) * (1.0 / 16777216.0)


# The center cell is always (0,0) for an x,y~=0 spawn -> a single constant.
_CENTER_BOX_U = _hash_cell_uniform01(0, 0)  # == 0.0 (see note above)


def center_height(
    terrain_type: torch.Tensor,
    difficulty: torch.Tensor,
    *,
    ground_height: float,
    step_height: float,
    grid_height_max: float,
) -> torch.Tensor:
    """Per-env terrain surface height at the tile CENTER (x,y ~= 0,0).

    EXACT mirror of ``SampleTerrainHeight(type, 0, 0, ScaleTerrainDifficulty(p,
    diff))`` for the center column. ``step_height`` / ``grid_height_max`` are the
    MODEL-level (un-scaled, the create-time) values; ``difficulty`` scales the
    vertical magnitude per env (terrain_field.hpp ScaleTerrainDifficulty).

    Parameters
    ----------
    terrain_type : (N,) or (N,1) int/uint tensor -- per-env terrain type code.
    difficulty   : (N,) float tensor             -- per-env difficulty in [0,1].
    ground_height, step_height, grid_height_max  -- the create-time TerrainParams
        (horizontal geometry is irrelevant to the center height).

    Returns
    -------
    (N,) float32 tensor on ``difficulty.device`` -- the absolute surface z at the
    center column for each env.
    """
    tt = terrain_type.reshape(-1).to(torch.long)
    diff = difficulty.reshape(-1).to(torch.float32)
    dev = diff.device

    # ScaleTerrainDifficulty: scale the vertical magnitudes only.
    sh = step_height * diff               # (N,) scaled step height
    gh = grid_height_max * diff           # (N,) scaled grid max

    h = torch.full_like(diff, float(ground_height))  # default = Flat.

    # PyramidStairs center: platform_top = ground + kPyramidRings*step_height.
    is_stairs = tt == TERRAIN_PYRAMID_STAIRS
    h = torch.where(is_stairs, ground_height + KPYRAMID_RINGS * sh, h)

    # InvertedPyramid center: pit_bottom = ground - kPyramidRings*step_height.
    is_pit = tt == TERRAIN_INVERTED_PYRAMID
    h = torch.where(is_pit, ground_height - KPYRAMID_RINGS * sh, h)

    # RandomBoxes center: h = ground + u(0,0)*grid_height_max (u(0,0)==0 -> ground).
    is_boxes = tt == TERRAIN_RANDOM_BOXES
    h = torch.where(is_boxes, ground_height + _CENTER_BOX_U * gh, h)

    return h.to(device=dev, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Difficulty <-> step_height mapping (owner top step 0.15 m).
# The world is cooked with step_height = STEP_HEIGHT_MAX (0.15); per-env
# DIFFICULTY in [0,1] scales the vertical magnitude (difficulty==0 -> FLAT for
# every terrain TYPE, difficulty==1 -> the full 0.15 m step). For a COLD-START
# (from-scratch) policy the curriculum starts near-flat (start_difficulty ~= 0.0)
# so the dog can bootstrap walking on effectively flat ground, then the
# survival/displacement curriculum raises difficulty toward 1.0 (= 0.15 m) as the
# policy improves.
#
# DIFFICULTY range is [0.0, 1.0] -- the easiest tier is TRULY FLAT (not the old
# 0.08 m floor), which is what cold-start bootstrapping needs. The legacy 0.08 m
# "easiest real step" constant is kept only for reference / difficulty_for_step.
# ---------------------------------------------------------------------------
STEP_HEIGHT_MAX = 0.15        # the create-time terrain_step_height (the MAX).
STEP_HEIGHT_MIN = 0.08        # legacy reference: a "small real step" (= diff 0.533).
DIFFICULTY_MIN = 0.0          # truly FLAT easiest tier (cold-start bootstrap).
DIFFICULTY_MAX = 1.0
# Default cold-start difficulty: ~flat so a from-scratch policy can learn to walk
# before any step height is introduced.
DIFFICULTY_COLD_START = 0.0


def difficulty_for_step(step_height_m: float) -> float:
    """Map a desired physical step height (m) to the per-env DIFFICULTY scalar."""
    return float(step_height_m) / STEP_HEIGHT_MAX


# ---------------------------------------------------------------------------
# Per-env terrain CURRICULUM scheduler (legged_gym terrain-level style).
# ---------------------------------------------------------------------------
class TerrainCurriculum:
    """Per-env terrain TYPE + DIFFICULTY curriculum, legged_gym-style SURVIVAL +
    DISPLACEMENT promotion (SCALE-INDEPENDENT).

    legged_gym's terrain curriculum promotes a per-env terrain LEVEL on reset by
    how FAR the robot walked from its origin (and demotes envs that walked too
    little / fell). We mirror that with two scale-independent gates that do NOT
    depend on the (tiny, ~0.01-0.07/step) reward magnitude:

      * SURVIVAL: the env survived most of the episode
        (``ep_len >= survive_frac * max_episode_length``), AND
      * PATH LENGTH: the base actually WALKED A LOT over the episode --
        the CUMULATIVE path length (``sum over steps of ||delta base_xy||``)
        exceeds ``promote_path_m``.

    >>> WHY PATH LENGTH, NOT NET DISPLACEMENT (the free-roam fix) <<<
    The Go2 trains under FREE-ROAM commands (random vx/vy/wyaw resampled every
    10 s, "随地走") AND a randomized spawn HEADING, so a perfectly-walking dog
    WANDERS -- it turns, sidesteps, and circles, and its NET displacement from
    spawn over an episode is typically << the distance it actually travelled.
    Measured on a trained flat-walker (go2_terrain_eval.py): even a 100%-survival,
    0%-fall flat episode has NET disp ~2.7 m but PATH length ~6.6 m -- net/path
    ~0.32-0.41 across all survivable cells. The OLD net-displacement gate
    (``||base_xy_done - start_xy|| > 3.0 m``) therefore NEVER fired -- 0% of
    eval cells reached net>3 m even for the perfect walker -- so the curriculum
    stayed pinned at flat (difficulty 0) forever (the symptom: reward plateaued
    ~11 by epoch ~250, ep_len ~750-900 = flat mastery, no difficulty-induced
    dips). The CUMULATIVE path-length gate fires for a wandering-but-walking dog
    (42% of eval cells reach path>5 m), so the curriculum can actually climb.
    NET displacement is RETAINED as a logged DIAGNOSTIC only.

    An env meeting BOTH is PROMOTED (difficulty += step; once it tops out at 1.0
    it advances to the next terrain TYPE and resets difficulty to the floor). An
    env that FELL (terminated, not just time-limit truncated) is DEMOTED
    (difficulty -= step, back toward flat). This climbs difficulty easy->1.0
    (= 0.15 m steps) as the policy learns to walk -- the OLD reward-threshold gate
    (``ep_mean > 8.0``) NEVER fired because the real reward scale is ~0.01-0.07,
    so every env stayed pinned at the easiest tier.

    State (all (num_envs,) device tensors): ``terrain_type`` (long), ``difficulty``
    (float in [DIFFICULTY_MIN, 1]), ``ep_len`` (running step count), ``path_len``
    (running cumulative path length, the promotion gate), ``prev_xy`` ((N,2) last
    step's base xy, the path-increment source), ``start_xy`` ((N,2) base xy at each
    reset, the NET-disp diagnostic origin), plus diagnostics ``ep_reward`` /
    ``ep_reward_ema`` (kept for logging only -- no longer the promotion gate).

    The class never touches the engine -- it only updates Python tensors. The env
    is responsible for (a) WRITING the per-env ``terrain_type`` / ``difficulty``
    into the engine fields + placing the base z, (b) calling ``set_start_xy`` with
    the spawn (x,y) at each reset, and (c) passing the live ``base_xy`` into
    ``update_on_done`` (the displacement source).
    """

    # The terrain TYPES the curriculum cycles an env through as it tops out a
    # type's difficulty. flat is implicit (difficulty 0 is flat for every type).
    # Order = ascending challenge.
    TYPE_LADDER = (
        TERRAIN_PYRAMID_STAIRS,
        TERRAIN_INVERTED_PYRAMID,
        TERRAIN_RANDOM_BOXES,
    )

    def __init__(
        self,
        num_envs: int,
        device,
        *,
        start_difficulty: float = DIFFICULTY_COLD_START,
        demote_on_fall: bool = True,
        difficulty_step: float = 0.1,
        ema_beta: float = 0.99,
        type_distribution=None,
        # SCALE-INDEPENDENT promotion gates (legged_gym terrain-level style).
        max_episode_length: int = 1,
        survive_frac: float = 0.8,
        # PRIMARY promotion gate (free-roam robust): cumulative PATH length.
        promote_path_m: float = 5.0,
        # NET-displacement (kept as a logged DIAGNOSTIC only -- the OLD gate).
        promote_dist_m: float = 3.0,
        # DEPRECATED reward-threshold gate (kept for back-compat / diagnostics;
        # NOT used by the new survival+path-length promotion).
        promote_thr: float = 8.0,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = device
        self.demote_on_fall = bool(demote_on_fall)
        self.difficulty_step = float(difficulty_step)
        self.ema_beta = float(ema_beta)
        self.max_episode_length = int(max_episode_length)
        self.survive_frac = float(survive_frac)
        self.promote_path_m = float(promote_path_m)
        self.promote_dist_m = float(promote_dist_m)  # diagnostic only now.
        self.promote_thr = float(promote_thr)  # diagnostic only now.
        # Promotion/demotion counters since the last diagnostic print (always-on).
        self.n_promote = 0
        self.n_demote = 0
        # The survival step threshold (an env must run at least this many steps).
        self.survive_steps = max(1, int(round(self.survive_frac * self.max_episode_length)))

        # Distribute the START terrain type across envs (round-robin over the
        # ladder by default, so all 3 types train simultaneously). Override with a
        # per-type weight dict {type_code: weight}.
        idx = torch.arange(self.num_envs, device=device)
        if type_distribution:
            codes, weights = zip(*sorted(type_distribution.items()))
            w = torch.tensor(weights, dtype=torch.float32, device=device)
            w = w / w.sum()
            # Deterministic interleave proportional to weights (stable, no RNG):
            # bucket env i by the cumulative-weight band it lands in.
            cum = torch.cumsum(w, 0)
            frac = (idx.float() + 0.5) / self.num_envs
            bucket = torch.searchsorted(cum, frac).clamp(max=len(codes) - 1)
            code_t = torch.tensor(codes, dtype=torch.long, device=device)
            self.terrain_type = code_t[bucket]
        else:
            ladder = torch.tensor(self.TYPE_LADDER, dtype=torch.long, device=device)
            self.terrain_type = ladder[idx % len(self.TYPE_LADDER)]

        self.difficulty = torch.full(
            (self.num_envs,), float(start_difficulty),
            dtype=torch.float32, device=device,
        )
        self.ep_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.ep_len = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.ep_reward_ema = torch.zeros(
            self.num_envs, dtype=torch.float32, device=device
        )
        # CUMULATIVE PATH LENGTH over the episode (the PRIMARY promotion gate) +
        # the previous step's base (x,y) it is differenced against. ``prev_xy`` is
        # seeded with the spawn xy on each reset (set_start_xy) so the first step's
        # increment is measured from the true episode origin.
        self.path_len = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.prev_xy = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=device)
        # Base (x,y) at the episode START (set by the env via set_start_xy on every
        # reset). NET displacement ||base_xy_done - start_xy|| is a DIAGNOSTIC now.
        self.start_xy = torch.zeros(self.num_envs, 2, dtype=torch.float32, device=device)
        # Last-computed per-env promotion-gate values (for the diagnostic prints).
        self.last_disp = torch.zeros(self.num_envs, dtype=torch.float32, device=device)
        self.last_path = torch.zeros(self.num_envs, dtype=torch.float32, device=device)

    def set_start_xy(self, mask, xy: torch.Tensor) -> None:
        """Record the base (x,y) at episode start for the reset envs, seed the
        path-length increment origin (``prev_xy``), and zero the cumulative
        ``path_len`` for the fresh episode.

        ``mask`` is the (N,) done bool (autoreset) or None (full reset); ``xy`` is
        the (N,2) spawn position. Called by the env right after it writes the spawn
        pose so BOTH the net-disp diagnostic and the path-length gate measure from
        the true episode origin."""
        xy = xy.to(self.start_xy.dtype)
        if mask is None:
            self.start_xy[:] = xy
            self.prev_xy[:] = xy
            self.path_len.zero_()
        else:
            self.start_xy[mask] = xy[mask]
            self.prev_xy[mask] = xy[mask]
            self.path_len[mask] = 0.0

    def accumulate_path(self, base_xy: torch.Tensor) -> None:
        """Add this control step's per-env path increment ``||base_xy - prev_xy||``
        into the running cumulative ``path_len`` (the PRIMARY promotion gate), then
        roll ``prev_xy`` forward. Called every step (in compute_reward, alongside
        ``accumulate``) BEFORE the autoreset clobbers the done envs' pose -- so the
        increment at the terminal step uses the true terminal xy. The subsequent
        ``set_start_xy`` (on the autoreset) re-seeds ``prev_xy`` to the new spawn,
        so a teleport from terminal->spawn is NOT counted as path."""
        xy = base_xy.to(self.prev_xy.dtype)
        self.path_len += (xy - self.prev_xy).norm(dim=1)
        self.prev_xy = xy

    def accumulate(self, reward: torch.Tensor) -> None:
        """Add this control step's per-env reward into the running episode return
        (diagnostic only now) + tick the per-env episode length (the survival
        gate). Called every step BEFORE update_on_done."""
        self.ep_reward += reward.detach().to(self.ep_reward.dtype)
        self.ep_len += 1

    def update_on_done(
        self, done: torch.Tensor, terminated: torch.Tensor,
        base_xy: "torch.Tensor | None" = None,
    ) -> None:
        """Promote/demote the DONE envs (survival + cumulative PATH length), then
        reset their episode accumulators.

        ``done`` = terminated | truncated; ``terminated`` = fell (base contact /
        flipped). PROMOTE a done env iff it SURVIVED most of the episode
        (``ep_len >= survive_steps``) AND actually WALKED A LOT -- its cumulative
        PATH length (``path_len``, accumulated per step via ``accumulate_path``)
        exceeds ``promote_path_m``. DEMOTE on a fall (terminated). Scale-independent
        -- no dependence on the reward magnitude, and FREE-ROAM robust (a wandering
        dog's path length, unlike its net displacement, reflects the distance it
        truly walked). ``base_xy`` is the (N,2) live base position at the done step,
        used ONLY for the NET-displacement DIAGNOSTIC now (``last_disp``).
        """
        if not bool(done.any()):
            return
        # Per-env mean reward over the finished episode (DIAGNOSTIC EMA only).
        denom = self.ep_len.clamp(min=1).to(torch.float32)
        ep_mean = self.ep_reward / denom
        self.ep_reward_ema = torch.where(
            done,
            self.ema_beta * self.ep_reward_ema + (1.0 - self.ema_beta) * ep_mean,
            self.ep_reward_ema,
        )

        # SCALE-INDEPENDENT gates: survived most of the episode AND walked a lot.
        survived = self.ep_len >= self.survive_steps
        walked_far = self.path_len > self.promote_path_m
        # NET-displacement DIAGNOSTIC (the OLD gate; logged, not enforced).
        if base_xy is not None:
            disp = (base_xy.to(self.start_xy.dtype) - self.start_xy).norm(dim=1)
        else:
            disp = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.last_disp = torch.where(done, disp, self.last_disp)
        self.last_path = torch.where(done, self.path_len, self.last_path)

        promote = done & survived & walked_far
        demote = done & terminated if self.demote_on_fall else torch.zeros_like(done)
        # An env can't both promote and demote; a fall (terminated) wins.
        promote = promote & ~demote
        # Always-on diagnostic counters (cheap; reset on each diagnostic print).
        self.n_promote += int(promote.sum())
        self.n_demote += int(demote.sum())

        # Raise / lower difficulty within [DIFFICULTY_MIN, 1].
        self.difficulty = torch.where(
            promote, self.difficulty + self.difficulty_step, self.difficulty
        )
        self.difficulty = torch.where(
            demote, self.difficulty - self.difficulty_step, self.difficulty
        )

        # When a PROMOTED env tops out (difficulty would exceed 1), advance it to
        # the next terrain TYPE in the ladder and reset difficulty to the floor.
        topped = promote & (self.difficulty > DIFFICULTY_MAX + 1e-6)
        if bool(topped.any()):
            cur = self.terrain_type[topped]
            ladder = torch.tensor(self.TYPE_LADDER, dtype=torch.long, device=self.device)
            # position of cur in the ladder (default 0 if not found), advance + wrap.
            pos = (cur.unsqueeze(1) == ladder.unsqueeze(0)).float().argmax(dim=1)
            nxt = ladder[(pos + 1) % len(self.TYPE_LADDER)]
            self.terrain_type[topped] = nxt
            # Start the NEW (harder) type at one difficulty step above flat, not at
            # 0 -- a topped-out env already walks well, so a pointless flat re-climb
            # on the new type is wasted; one step in keeps a gentle re-introduction.
            self.difficulty[topped] = min(self.difficulty_step, DIFFICULTY_MAX)

        self.difficulty = self.difficulty.clamp(DIFFICULTY_MIN, DIFFICULTY_MAX)

        # Reset the episode accumulators for the done envs. ``path_len`` is also
        # re-zeroed by the subsequent ``set_start_xy`` (which re-seeds ``prev_xy``
        # to the fresh spawn), but we zero it here too so the accumulator is in a
        # clean state regardless of call ordering.
        self.ep_reward = torch.where(done, torch.zeros_like(self.ep_reward), self.ep_reward)
        self.ep_len = torch.where(done, torch.zeros_like(self.ep_len), self.ep_len)
        self.path_len = torch.where(done, torch.zeros_like(self.path_len), self.path_len)

    # -- always-on training diagnostic (per-type difficulty + promote/demote) --
    def difficulty_report(self, reset_counters: bool = True) -> str:
        """A compact PER-TYPE difficulty report (min/mean/max) + the #promotions /
        #demotions and mean path-length SINCE the last report -- the always-on
        training-metrics line that SHOWS the curriculum climbing flat -> 0.15 m.

        Cheap: a handful of reductions over the (num_envs,) tensors. When
        ``reset_counters`` the promote/demote counters are zeroed after reading so
        the next report covers only the next interval."""
        names = {0: "Flat", 1: "Stairs", 2: "Pit", 3: "Boxes"}
        tt = self.terrain_type.detach().cpu()
        td = self.difficulty.detach().cpu()
        parts = []
        for code in (0, 1, 2, 3):
            m = tt == code
            n = int(m.sum())
            if n:
                d = td[m]
                parts.append(f"{names[code]}={n}"
                             f"[d {float(d.min()):.2f}/{float(d.mean()):.2f}/"
                             f"{float(d.max()):.2f}]")
        mean_path = float(self.last_path.detach().mean())
        line = (f"difficulty(min/mean/max): {'  '.join(parts)}  |  "
                f"+{self.n_promote}/-{self.n_demote} (promote/demote since last)  "
                f"|  mean_done_path={mean_path:.2f}m")
        if reset_counters:
            self.n_promote = 0
            self.n_demote = 0
        return line

    def histogram(self) -> str:
        """A one-line type+difficulty histogram (smoke-run diagnostic)."""
        names = {0: "Flat", 1: "Stairs", 2: "Pit", 3: "Boxes"}
        tt = self.terrain_type.detach().cpu()
        td = self.difficulty.detach().cpu()
        parts = []
        for code in (0, 1, 2, 3):
            m = tt == code
            n = int(m.sum())
            if n:
                dmin, dmax = float(td[m].min()), float(td[m].max())
                parts.append(f"{names[code]}={n}(d{dmin:.2f}-{dmax:.2f})")
        return "  ".join(parts)

    def gate_diag(self) -> str:
        """One-line promotion-gate reachability diagnostic (smoke run): the
        survival-step threshold + the live PATH length (primary gate) and net
        displacement (diagnostic) seen at done, so a smoke can show whether the
        SCALE-INDEPENDENT criterion is REACHABLE."""
        disp = self.last_disp.detach().cpu()
        path = self.last_path.detach().cpu()
        return (f"promote_gate[survive>={self.survive_steps} steps & "
                f"path>{self.promote_path_m:.1f}m]  "
                f"max_done_path={float(path.max()):.3f}m mean={float(path.mean()):.3f}m  "
                f"(diag net_disp max={float(disp.max()):.3f}m mean={float(disp.mean()):.3f}m)")
