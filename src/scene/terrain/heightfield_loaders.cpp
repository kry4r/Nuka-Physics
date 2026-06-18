// ---------------------------------------------------------------------------
// heightfield_loaders.cpp -- the ONE parametric field composer + image loader.
//
// GenerateHeightField composes analytic feature terms (slope + Chebyshev-ring
// steps + hashed bumps) by amplitude -- no terrain-type branch. feature_cell > 0
// tiles the same parametric features per cell with a flat seam rim. Absolute
// per-cell heights are normalized to [0,1] against the grid's own z-range, so the
// cooked HeightfieldData (base_z=min, scale_z=range) matches a flat field bit-for-
// bit (all-equal heights -> the flat-grid path).
// ---------------------------------------------------------------------------

#include "scene/terrain/heightfield_loaders.hpp"

#include <cstdint>
#include <vector>

#include "scene/terrain/image_grayscale.hpp"

namespace nuka::terrain {
namespace {

float AbsF(float v) { return v < 0.0f ? -v : v; }
float MaxF(float a, float b) { return a > b ? a : b; }

// floor() without <cmath> on the host generator path: truncate then correct.
int FloorToInt(float v) {
    const int t = static_cast<int>(v);
    return (static_cast<float>(t) > v) ? (t - 1) : t;
}

// Pure 2D cell hash -> [0,1); MurmurHash3-style avalanche, no RNG state.
uint32_t HashCell(int cx, int cy) {
    uint32_t h = static_cast<uint32_t>(cx) * 0x9E3779B1u +
                 static_cast<uint32_t>(cy) * 0x85EBCA77u;
    h ^= h >> 16;
    h *= 0x7FEB352Du;
    h ^= h >> 15;
    h *= 0x846CA68Bu;
    h ^= h >> 16;
    return h;
}
float HashCellUniform01(int cx, int cy) {
    return static_cast<float>(HashCell(cx, cy) >> 8) * (1.0f / 16777216.0f);
}
float HashCellUniform01B(int cx, int cy) {
    return static_cast<float>(HashCell(cx + 0x1234, cy - 0x5678) >> 8) *
           (1.0f / 16777216.0f);
}

// Chebyshev-ring staircase rise (signed) at radius r from the field centre.
// rise > 0: a plateau ring_count*rise above the floor, stepping DOWN outward.
// rise < 0: a pit |ring_count*rise| below the floor, stepping UP outward.
// Returns the rise RELATIVE to the floor (0 outside the staircase footprint).
float RingStepRise(float r, float rise, float width, float platform, int count) {
    if (rise == 0.0f || !(width > 0.0f) || count < 1) return 0.0f;
    const float half_plat = platform * 0.5f;
    const float peak = static_cast<float>(count) * rise;  // signed plateau/pit.
    if (r <= half_plat) return peak;
    const int k = FloorToInt((r - half_plat) / width) + 1;
    float h = peak - static_cast<float>(k) * rise;
    // Clamp back to the floor once the rings run out (toward 0 from the peak side).
    if (rise > 0.0f) { if (h < 0.0f) h = 0.0f; }
    else { if (h > 0.0f) h = 0.0f; }
    return h;
}

// Per-cell hashed bump rise at column (x,y) over `cell` footprints, in [0,max].
float BumpRise(float x, float y, float cell, float max_rise) {
    if (!(cell > 0.0f) || !(max_rise > 0.0f)) return 0.0f;
    const int cx = FloorToInt(x / cell);
    const int cy = FloorToInt(y / cell);
    return HashCellUniform01(cx, cy) * max_rise;
}

// The composed feature rise (relative to the floor) at a column, for the GLOBAL
// (non-tiled) field. Sum of the slope + ring + bump terms by amplitude.
float GlobalRise(const TerrainGenConfig& cfg, float lx, float ly) {
    const float r = MaxF(AbsF(lx), AbsF(ly));  // Chebyshev radius from centre.
    float h = cfg.grade_x * lx + cfg.grade_y * ly;
    h += RingStepRise(r, cfg.ring_rise, cfg.ring_width, cfg.ring_platform,
                      static_cast<int>(cfg.ring_count));
    h += BumpRise(lx, ly, cfg.bump_cell, cfg.bump_height);
    return h;
}

// The TILED feature rise: pave the world with feature_cell squares; each tile holds
// the parametric features in its inner region, with a flat feature_margin rim so
// neighbours meet seamlessly. A per-tile deterministic vertical scale + a per-tile
// mix (which features that tile shows) keep the field varied -- still pure params.
float TiledRise(const TerrainGenConfig& cfg, float x, float y) {
    const float cell = cfg.feature_cell;
    const int cx = FloorToInt(x / cell);
    const int cy = FloorToInt(y / cell);
    const float lx = x - (static_cast<float>(cx) + 0.5f) * cell;  // tile-local.
    const float ly = y - (static_cast<float>(cy) + 0.5f) * cell;
    const float feature_half = cell * 0.5f - cfg.feature_margin;
    if (!(feature_half > 0.0f)) return 0.0f;
    const float rl = MaxF(AbsF(lx), AbsF(ly));
    if (rl >= feature_half) return 0.0f;  // flat seam rim.

    const int sx = cx + static_cast<int>(cfg.feature_seed);
    const float vscale = 0.55f + 0.40f * HashCellUniform01B(sx, cy);
    // Per-tile mix over the available features (deterministic, no RNG): which
    // amplitude this tile expresses, scaled by vscale. Tiles with neither ring nor
    // bump amplitude fall back to flat (a walkway).
    const uint32_t pick = HashCell(sx, cy) % 20u;
    float h = 0.0f;
    if (pick < 5u) {  // climbing staircase tile.
        const int rc = FloorToInt((feature_half - cfg.ring_platform * 0.5f) /
                                  MaxF(cfg.ring_width, 1.0e-6f));
        h = RingStepRise(rl, cfg.ring_rise * vscale, cfg.ring_width,
                         cfg.ring_platform, rc);
    } else if (pick < 11u) {  // descending pit tile.
        const int rc = FloorToInt((feature_half - cfg.ring_platform * 0.5f) /
                                  MaxF(cfg.ring_width, 1.0e-6f));
        h = RingStepRise(rl, -cfg.ring_rise * vscale, cfg.ring_width,
                         cfg.ring_platform, rc);
    } else if (pick < 17u) {  // bumpy tile.
        const int bx = FloorToInt(lx / MaxF(cfg.bump_cell, 1.0e-6f));
        const int by = FloorToInt(ly / MaxF(cfg.bump_cell, 1.0e-6f));
        h = HashCellUniform01(bx + cx * 131, by + cy * 137) *
            (cfg.bump_height * vscale);
    }  // else: flat walkway tile.
    return h;
}

float FieldRise(const TerrainGenConfig& cfg, float lx, float ly) {
    if (cfg.feature_cell > 0.0f) return TiledRise(cfg, lx, ly);
    return GlobalRise(cfg, lx, ly);
}

// Normalize absolute heights into `out`: range from the grid's own min/max; base_z
// = min, scale_z = range; a flat grid gets a tiny finite range so the stored values
// are all 0 (the flat byte-identity contract).
void NormalizeIntoField(const TerrainGenConfig& cfg,
                        const std::vector<float>& abs_z, HeightField& out) {
    float min_z = 3.402823466e+38f, max_z = -3.402823466e+38f;
    for (float z : abs_z) {
        if (z < min_z) min_z = z;
        if (z > max_z) max_z = z;
    }
    if (!(max_z > min_z)) max_z = min_z + 1.0e-4f;  // flat: tiny finite range.
    const float range = max_z - min_z;
    out.values.resize(abs_z.size());
    for (size_t i = 0u; i < abs_z.size(); ++i) {
        out.values[i] = (range > 0.0f) ? (abs_z[i] - min_z) / range : 0.0f;
    }
    out.nrow = cfg.nrow;
    out.ncol = cfg.ncol;
    out.cell_x = cfg.cell_x;
    out.cell_y = cfg.cell_y;
    out.origin = cfg.origin;
    out.base_z = min_z;
    out.scale_z = range;
}

}  // namespace

bool GenerateHeightField(const TerrainGenConfig& cfg, HeightField& out) {
    out = HeightField{};
    if (cfg.nrow < 2u || cfg.ncol < 2u || !(cfg.cell_x > 0.0f) ||
        !(cfg.cell_y > 0.0f)) {
        return false;
    }
    const size_t cells =
        static_cast<size_t>(cfg.nrow) * static_cast<size_t>(cfg.ncol);
    std::vector<float> abs_z(cells, 0.0f);
    for (uint32_t r = 0u; r < cfg.nrow; ++r) {
        const float y = cfg.origin.y + static_cast<float>(r) * cfg.cell_y;
        for (uint32_t c = 0u; c < cfg.ncol; ++c) {
            const float x = cfg.origin.x + static_cast<float>(c) * cfg.cell_x;
            abs_z[static_cast<size_t>(r) * cfg.ncol + c] =
                cfg.base_z + FieldRise(cfg, x, y);
        }
    }
    NormalizeIntoField(cfg, abs_z, out);
    return true;
}

bool LoadHeightFieldFromImage(const std::string& path, float radius_x,
                              float radius_y, float elevation_z, float base_z,
                              const math::Vec3& origin, HeightField& out,
                              std::string& err) {
    out = HeightField{};
    if (!(radius_x > 0.0f) || !(radius_y > 0.0f)) {
        err = "image heightfield needs positive radius_x/radius_y";
        return false;
    }
    GrayscaleImage img;
    if (!LoadGrayscaleImage(path, img, err)) return false;
    if (img.width < 2u || img.height < 2u) {
        err = "image heightfield needs >=2x2 pixels: " + path;
        return false;
    }
    out.values = std::move(img.values);  // already normalized [0,1].
    out.nrow = img.height;
    out.ncol = img.width;
    out.cell_x = 2.0f * radius_x / static_cast<float>(out.ncol - 1u);
    out.cell_y = 2.0f * radius_y / static_cast<float>(out.nrow - 1u);
    out.origin = origin;
    out.base_z = base_z;
    out.scale_z = elevation_z;
    return true;
}

}  // namespace nuka::terrain
