#pragma once
// ---------------------------------------------------------------------------
// terrain_field.hpp -- the SHARED procedural-terrain heightfield h(x,y).
// (Go2-on-stairs demo, Phase 1: procedural terrain physics.)
// ---------------------------------------------------------------------------
//
// THE SINGLE SOURCE OF TRUTH for the terrain surface height. Both the PHYSICS
// foot-contact kernel (src/phi/backend_cuda/ops/contacts_foot.cu, compiled by
// nvcc) AND the future render-mesh tessellator (a plain-C++ CPU pass, compiled
// by g++) include THIS header and call SampleTerrainHeight() -- so the rendered
// ground mesh can NEVER drift from the surface the feet actually rest on. The
// #1 correctness risk of "render h(x,y) != physics h(x,y)" is eliminated by
// having ONE closed-form definition.
//
// HOST/DEVICE portability follows the src/sensor/noise/philox.cuh precedent:
// the helpers are __host__ __device__ so the SAME pure function compiles under
// nvcc (the .cu kernel) AND the plain C++ toolchain (the .cpp tessellator).
// When this header is included from a non-NVCC translation unit, the CUDA
// qualifier macros are not defined, so we map them to empty / inline.
//
// MODEL: heightfield-as-VERTICAL-SUPPORT. SampleTerrainHeight(type,x,y,p)
// returns the ABSOLUTE world-space surface height z at the column (x,y). The
// foot rests when its sphere center.z drops to (h(x,y) + radius); the contact
// normal stays {0,0,1} (gradient/slope normals are an EXPLICIT non-goal for
// v1 -- this is the standard heightfield support model the locomotion-RL
// terrain curricula (Isaac/legged_gym) use for blind walking).
//
// DETERMINISM: every formula is closed-form (no loops over steps -- the step
// index is computed directly) and the RandomBoxes height uses a PURE integer
// hash of the cell (no Math.random / no RNG state), so the result is bit-exact
// reproducible on host and device and across replay passes.
// ---------------------------------------------------------------------------

#include <cstdint>

#if defined(__CUDACC__)
#define NK_TERRAIN_HD __host__ __device__
#else
#define NK_TERRAIN_HD
#endif

namespace nuka::terrain {

// -- terrain type codes ------------------------------------------------------
// type==0 (Flat) MUST return exactly p.ground_height so the foot kernel is
// byte-identical to today's scalar-plane path (the D1 default).
enum TerrainType : uint32_t {
    kTerrainFlat            = 0u,  // flat plane at ground_height (D1 default).
    kTerrainPyramidStairs   = 1u,  // concentric square steps climbing UP to a
                                   //   central platform.
    kTerrainInvertedPyramid = 2u,  // concentric square steps descending DOWN
                                   //   to a central pit (the mirror of stairs).
    kTerrainRandomBoxes     = 3u,  // per-cell deterministic-hash random height.
};

// -- terrain parameters (POD) ------------------------------------------------
// Model-level cook config (default all-zero except ground_height => flat). The
// SAME params drive the physics sample and the render tessellator.
struct TerrainParams {
    float ground_height;    // base plane height (the absolute floor / step-0).
    float step_height;      // vertical rise per step ring (PyramidStairs/Pit).
    float step_width;       // horizontal width of each step ring.
    float platform_width;   // full edge length of the central top platform.
    float grid_width;       // cell edge length (RandomBoxes).
    float grid_height_max;  // max per-cell random rise above ground (RandomBoxes).
};

// A reasonable default: flat at z=0 with no terrain features.
NK_TERRAIN_HD inline TerrainParams DefaultTerrainParams() {
    return TerrainParams{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
}

// ---------------------------------------------------------------------------
// ScaleTerrainDifficulty: scale a TerrainParams by a per-env curriculum
// DIFFICULTY factor (Go2-on-stairs demo, Phase 2a). The SINGLE source of truth
// for the difficulty scaling, called by BOTH the PHYSICS foot-contact kernel
// (contacts_foot.cu) AND the future RENDER tessellator -- so the rendered ground
// mesh can never disagree with the surface the feet rest on (the #1 drift risk,
// mirroring why SampleTerrainHeight itself is shared).
//
// SEMANTICS: difficulty scales only the VERTICAL feature magnitude --
// step_height (PyramidStairs / InvertedPyramid rise per ring) and grid_height_max
// (RandomBoxes max per-cell rise). The HORIZONTAL geometry (step_width,
// platform_width, grid_width) and ground_height are LEFT UNCHANGED so a curriculum
// raises/lowers the same step/box layout in place (the legged_gym terrain-level
// convention: higher level == taller steps over the SAME footprint). diff == 1.0
// returns the params unchanged; diff == 0.0 collapses the terrain to a flat plane
// at ground_height (every step/box rise becomes zero -> SampleTerrainHeight
// returns ground_height for every column).
// ---------------------------------------------------------------------------
NK_TERRAIN_HD inline TerrainParams ScaleTerrainDifficulty(TerrainParams p,
                                                          float diff) {
    p.step_height     *= diff;  // vertical rise per step ring.
    p.grid_height_max *= diff;  // max per-cell random rise (RandomBoxes).
    return p;                   // horizontal geometry + ground_height unchanged.
}

namespace detail {

NK_TERRAIN_HD inline float AbsF(float v) { return v < 0.0f ? -v : v; }
NK_TERRAIN_HD inline float MaxF(float a, float b) { return a > b ? a : b; }

// floor() without pulling <cmath> into the device path. For the cell index we
// only need the integer floor of x/grid_width; a truncate-toward-zero cast plus
// a correction for negatives is exact for the finite ranges terrain spans.
NK_TERRAIN_HD inline int FloorToInt(float v) {
    const int t = static_cast<int>(v);  // truncates toward zero.
    return (static_cast<float>(t) > v) ? (t - 1) : t;
}

// Deterministic integer hash of a 2D cell -> uniform [0,1). A pure function of
// the cell coordinates (no RNG state), so RandomBoxes is bit-exact reproducible
// on host + device. Mixes the two signed cell ids into one uint32 via a
// Wang/MurmurHash-style avalanche, then maps the top bits to [0,1).
NK_TERRAIN_HD inline uint32_t HashCell(int cx, int cy) {
    uint32_t h = static_cast<uint32_t>(cx) * 0x9E3779B1u  // golden-ratio mix
               + static_cast<uint32_t>(cy) * 0x85EBCA77u;
    // MurmurHash3 finalizer (avalanche).
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}

NK_TERRAIN_HD inline float HashCellUniform01(int cx, int cy) {
    // Top 24 bits -> [0,1): exact float division, deterministic.
    return static_cast<float>(HashCell(cx, cy) >> 8) * (1.0f / 16777216.0f);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// SampleTerrainHeight: the absolute surface height z at world column (x,y).
//
// Square-ring geometry uses the Chebyshev distance r = max(|x|,|y|) from the
// origin (so each "ring" is an axis-aligned square, matching the legged_gym /
// Isaac pyramid-stairs convention). Let half_plat = platform_width * 0.5.
//
// FORMULAS (all closed-form; the step index k is computed directly, no loop):
//
//   Flat (type 0):
//       h = ground_height                                  (EXACTLY -- D1)
//
//   PyramidStairs (type 1): a central square platform of half-width half_plat
//   sits at the TOP (platform_top = ground_height + kPyramidRings*step_height);
//   outside the platform the surface steps DOWN by step_height per step_width of
//   outward Chebyshev distance, clamped BELOW at ground_height:
//       r = max(|x|,|y|)
//       if r <= half_plat:  h = platform_top
//       else:               k = floor((r - half_plat)/step_width) + 1
//                           h = max(ground_height, platform_top - k*step_height)
//   Ring 1 is the first step OUTSIDE the platform; the clamp flattens the
//   surface to the ground floor beyond kPyramidRings rings.
//
//   InvertedPyramid (type 2): the mirror of stairs -- a central PIT
//   (pit_bottom = ground_height - kPyramidRings*step_height) rising UP to the
//   ground floor as you move outward:
//       if r <= half_plat:  h = pit_bottom
//       else:               k = floor((r - half_plat)/step_width) + 1
//                           h = min(ground_height, pit_bottom + k*step_height)
//
//   RandomBoxes (type 3): per-cell deterministic-hash random height.
//       cx = floor(x/grid_width), cy = floor(y/grid_width)
//       u  = hash(cx,cy) in [0,1)          (PURE integer hash -- no RNG state)
//       h  = ground_height + u*grid_height_max   in [ground, ground+max].
//
// platform_top / pit_bottom are computed from the given params + the fixed
// kPyramidRings constant, so SampleTerrainHeight is a pure function of
// (type,x,y,params) and the render tessellator reproduces the surface 1:1.
// ---------------------------------------------------------------------------

// Number of step rings from the ground floor up to the central platform top.
// The platform sits kPyramidRings*step_height above ground; columns farther out
// than kPyramidRings rings clamp to the ground floor. A fixed, documented
// constant keeps SampleTerrainHeight a pure function of (type,x,y,params) and
// lets the renderer reproduce the exact platform elevation.
inline constexpr int kPyramidRings = 8;

NK_TERRAIN_HD inline float SampleTerrainHeight(uint32_t type, float x, float y,
                                               const TerrainParams& p) {
    // type==0 (Flat): EXACTLY ground_height -> the foot kernel is byte-identical
    // to the legacy scalar-plane path. This MUST stay an unconditional return so
    // the default terrain reproduces today's goldens to the bit.
    if (type == kTerrainFlat) {
        return p.ground_height;
    }

    const float half_plat = p.platform_width * 0.5f;
    const float r = detail::MaxF(detail::AbsF(x), detail::AbsF(y));  // Chebyshev

    if (type == kTerrainPyramidStairs) {
        // Platform top is kPyramidRings steps above ground.
        const float platform_top =
            p.ground_height + static_cast<float>(kPyramidRings) * p.step_height;
        if (r <= half_plat) {
            return platform_top;  // on the central top platform.
        }
        // Ring index 1.. outside the platform (closed-form, no loop). Guard a
        // zero/negative step_width (degenerate params) by treating it as flat.
        if (!(p.step_width > 0.0f)) {
            return platform_top;
        }
        const int k = detail::FloorToInt((r - half_plat) / p.step_width) + 1;
        float h = platform_top - static_cast<float>(k) * p.step_height;
        if (h < p.ground_height) {
            h = p.ground_height;  // flatten to the floor far from center.
        }
        return h;
    }

    if (type == kTerrainInvertedPyramid) {
        // Mirror of the stairs: a central PIT kPyramidRings steps BELOW ground,
        // rising back UP to ground_height as you move outward. pit_bottom is the
        // deepest point (center); columns far out clamp to ground_height.
        const float pit_bottom =
            p.ground_height - static_cast<float>(kPyramidRings) * p.step_height;
        if (r <= half_plat) {
            return pit_bottom;  // on the central pit floor.
        }
        if (!(p.step_width > 0.0f)) {
            return pit_bottom;
        }
        const int k = detail::FloorToInt((r - half_plat) / p.step_width) + 1;
        float h = pit_bottom + static_cast<float>(k) * p.step_height;
        if (h > p.ground_height) {
            h = p.ground_height;  // level out to the floor far from center.
        }
        return h;
    }

    if (type == kTerrainRandomBoxes) {
        // Per-cell deterministic-hash random height: cell = (floor(x/gw),
        // floor(y/gw)); u = hash(cell) in [0,1); h = ground + u*grid_height_max.
        // Pure hash -> bit-exact reproducible (no RNG state). Guard a zero
        // grid_width (degenerate) by returning the flat floor.
        if (!(p.grid_width > 0.0f)) {
            return p.ground_height;
        }
        const int cx = detail::FloorToInt(x / p.grid_width);
        const int cy = detail::FloorToInt(y / p.grid_width);
        const float u = detail::HashCellUniform01(cx, cy);  // [0,1)
        return p.ground_height + u * p.grid_height_max;
    }

    // Unknown type: fall back to the flat floor (defensive; never reached for
    // the four defined codes).
    return p.ground_height;
}

}  // namespace nuka::terrain
