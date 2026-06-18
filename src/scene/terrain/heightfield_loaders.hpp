#pragma once
// ---------------------------------------------------------------------------
// heightfield_loaders.hpp -- the two sources that fill a HeightField.
//
// (a) GenerateHeightField(TerrainGenConfig): ONE general parametric field -- a
//     COMPOSITION of analytic feature terms (slope + Chebyshev-ring steps + hashed
//     bumps), each gated by an amplitude parameter. There is NO terrain-type enum:
//     a flat field is every amplitude 0; a staircase is ring_rise > 0; a pit is
//     ring_rise < 0; a bumpy field is bump_height > 0; a ramp is grade != 0; the
//     "complex" multi-feature field (the 20-dog scene) is feature_cell > 0, which
//     TILES the same parametric features per cell. Every preset is a parameter set.
// (b) LoadHeightFieldFromImage: a grayscale height map (PGM/raw-16) -> normalized
//     grid, placed by the given world extent. PNG is a loud unsupported error.
//
// Both produce the ONE HeightField the cook + the bilinear sampler consume.
// ---------------------------------------------------------------------------

#include <cstdint>
#include <string>

#include "math/vec3.hpp"
#include "scene/terrain/heightfield.hpp"

namespace nuka::terrain {

// A parametric terrain field: the grid resolution + world placement + the analytic
// feature amplitudes. The field z at a column is a COMPOSITION of the features whose
// amplitude is non-zero -- no discrete type selector. base_z is the ground floor.
//
// Features (each independent; zero amplitude == absent):
//   - SLOPE: grade_x*(x-cx) + grade_y*(y-cy) (a planar ramp).
//   - RING STEPS: a square (Chebyshev) staircase of |ring_rise| per ring_width
//     outside a central platform of ring_platform; ring_rise > 0 climbs to a
//     central plateau, < 0 descends to a central pit. ring_count bounds the rings.
//   - BUMPS: per-cell deterministic-hash rise in [0, bump_height] over bump_cell
//     footprints (a rough/boxy field).
//
// feature_cell > 0 TILES the field: the world is paved with feature_cell squares,
// each cell carrying the SAME parametric features inside an inner region (a flat
// feature_margin rim keeps cells seamless), with a per-cell deterministic vertical
// scale + a per-cell feature mix -- the "complex" multi-feature field, still pure
// parameters. feature_cell == 0 is a single global field over the whole grid.
struct TerrainGenConfig {
    uint32_t nrow = 0u, ncol = 0u;
    float cell_x = 0.0f, cell_y = 0.0f;
    math::Vec3 origin{};          // world corner cell (0,0).
    float base_z = 0.0f;          // absolute z of the ground floor.

    // Planar ramp.
    float grade_x = 0.0f;         // rise per metre along +X.
    float grade_y = 0.0f;         // rise per metre along +Y.

    // Chebyshev-ring staircase (signed: + climbs, - descends to a pit).
    float ring_rise = 0.0f;       // vertical rise per ring (signed amplitude).
    float ring_width = 0.0f;      // horizontal width of each ring.
    float ring_platform = 0.0f;   // central platform edge length.
    uint32_t ring_count = 0u;     // number of rings from floor to the plateau/pit.

    // Per-cell hashed bumps (a rough/boxy field).
    float bump_height = 0.0f;     // max per-cell rise above the floor.
    float bump_cell = 0.0f;       // bump footprint edge.

    // Feature TILING (the multi-feature "complex" field). 0 == one global field.
    float feature_cell = 0.0f;    // tile edge length.
    float feature_margin = 0.0f;  // flat seam rim per tile side.
    uint32_t feature_seed = 0u;   // hash salt for the per-tile mix/scale.
};

// Fill `out` from a parametric config. The feature composition runs ONCE per cell
// here (host) and writes the normalized [0,1] grid; placement (cell/origin/base/
// scale) is set from the config. Determinism: the bump + tile mix use a pure cell
// hash. Returns false on a degenerate config (nrow/ncol<2 or non-positive spacing).
bool GenerateHeightField(const TerrainGenConfig& cfg, HeightField& out);

// Load a grayscale height map into a HeightField. pixel -> [0,1]; nrow=image
// height, ncol=image width, cell from the world extent (2*radius/(n-1)). Returns
// false (and leaves out empty) on a malformed/undecodable/unsupported file -- a
// LOUD failure (err set), never a silent flat fallback. PNG is unsupported here.
bool LoadHeightFieldFromImage(const std::string& path, float radius_x,
                              float radius_y, float elevation_z, float base_z,
                              const math::Vec3& origin, HeightField& out,
                              std::string& err);

}  // namespace nuka::terrain
