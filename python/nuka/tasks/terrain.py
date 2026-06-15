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

NOTE (future enhancement): the Go2 always spawns at the tile CENTER (x,y~=0,0),
which is sufficient for v1 -- a single closed-form center height per type. A
random-XY spawn (legged_gym terrain_origins jitter) would need the FULL
``SampleTerrainHeight`` (the ring index / per-cell hash for arbitrary x,y) ported
here; that is explicitly deferred.
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
# Difficulty <-> step_height mapping (owner range 0.08-0.15 m).
# The world is cooked with step_height = STEP_HEIGHT_MAX (0.15); per-env
# DIFFICULTY scales the vertical magnitude. To keep the EASIEST tier at a real
# 0.08 m step (not a flat 0), difficulty is clamped to [DIFFICULTY_MIN, 1.0]
# with DIFFICULTY_MIN = 0.08/0.15 ~= 0.5333, so the realized step spans the full
# owner band [0.08, 0.15].
# ---------------------------------------------------------------------------
STEP_HEIGHT_MAX = 0.15        # the create-time terrain_step_height (the MAX).
STEP_HEIGHT_MIN = 0.08        # owner's easiest real step.
DIFFICULTY_MIN = STEP_HEIGHT_MIN / STEP_HEIGHT_MAX  # ~= 0.5333
DIFFICULTY_MAX = 1.0


def difficulty_for_step(step_height_m: float) -> float:
    """Map a desired physical step height (m) to the per-env DIFFICULTY scalar."""
    return float(step_height_m) / STEP_HEIGHT_MAX


# ---------------------------------------------------------------------------
# Per-env terrain CURRICULUM scheduler (legged_gym terrain-level style).
# ---------------------------------------------------------------------------
class TerrainCurriculum:
    """Per-env terrain TYPE + DIFFICULTY curriculum with reward-EMA promotion.

    legged_gym uses a per-env terrain LEVEL that promotes/demotes on reset by how
    far the robot walked. We use a per-env reward-EMA (a blind-policy proxy for
    "did it walk well") because Nuka's blind obs has no terrain-origin distance
    signal -- an env whose EMA clears ``promote_thr`` is promoted (taller steps,
    and once it tops out, advanced to the next terrain TYPE); an env that FELL
    (terminated, not just truncated) is demoted (shorter steps / back toward flat).

    State (all (num_envs,) device tensors): ``terrain_type`` (long), ``difficulty``
    (float in [DIFFICULTY_MIN, 1]), ``ep_reward_ema`` (the per-env reward-EMA
    accumulated over the live episode and frozen at the done step), ``ep_reward``
    (running episode return), ``ep_len`` (running step count).

    The class never touches the engine -- it only updates Python tensors. The env
    is responsible for WRITING the per-env ``terrain_type`` / ``difficulty`` into
    the engine ``ENV_TERRAIN_TYPE`` / ``ENV_TERRAIN_DIFFICULTY`` buffer_view and
    placing the base z via ``center_height``.
    """

    # The terrain TYPES the curriculum cycles an env through as it tops out a
    # type's difficulty. flat is implicit (low-difficulty stairs already ~ flat at
    # diff_min the step is 0.08 m). Order = ascending challenge.
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
        start_difficulty: float = DIFFICULTY_MIN,
        promote_thr: float = 8.0,
        demote_on_fall: bool = True,
        difficulty_step: float = 0.1,
        ema_beta: float = 0.99,
        type_distribution=None,
    ) -> None:
        self.num_envs = int(num_envs)
        self.device = device
        self.promote_thr = float(promote_thr)
        self.demote_on_fall = bool(demote_on_fall)
        self.difficulty_step = float(difficulty_step)
        self.ema_beta = float(ema_beta)

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

    def accumulate(self, reward: torch.Tensor) -> None:
        """Add this control step's per-env reward into the running episode return
        (called every step BEFORE update_on_done)."""
        self.ep_reward += reward.detach().to(self.ep_reward.dtype)
        self.ep_len += 1

    def update_on_done(
        self, done: torch.Tensor, terminated: torch.Tensor
    ) -> None:
        """Promote/demote the DONE envs, then reset their episode accumulators.

        ``done`` = terminated | truncated; ``terminated`` = fell (base contact /
        flipped). Promotion uses the per-env mean reward over the just-finished
        episode (return / length) as the walk-quality proxy. A fall demotes.
        """
        if not bool(done.any()):
            return
        # Per-env mean reward over the finished episode (avoid div-by-zero).
        denom = self.ep_len.clamp(min=1).to(torch.float32)
        ep_mean = self.ep_reward / denom
        # EMA of the per-episode mean reward (diagnostic + smoother gate).
        self.ep_reward_ema = torch.where(
            done,
            self.ema_beta * self.ep_reward_ema + (1.0 - self.ema_beta) * ep_mean,
            self.ep_reward_ema,
        )

        promote = done & (ep_mean > self.promote_thr)
        demote = done & terminated if self.demote_on_fall else torch.zeros_like(done)
        # An env can't both promote and demote; a fall (terminated) wins.
        promote = promote & ~demote

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
            self.difficulty[topped] = DIFFICULTY_MIN

        self.difficulty = self.difficulty.clamp(DIFFICULTY_MIN, DIFFICULTY_MAX)

        # Reset the episode accumulators for the done envs.
        self.ep_reward = torch.where(done, torch.zeros_like(self.ep_reward), self.ep_reward)
        self.ep_len = torch.where(done, torch.zeros_like(self.ep_len), self.ep_len)

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
