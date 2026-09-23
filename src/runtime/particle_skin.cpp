#include "runtime/particle_skin.hpp"

#include <cstddef>

namespace nuka::runtime {
namespace {

using math::Vec3;
using render::MeshGeometry;

// Deterministic 32-bit integer hash (index-keyed, stable across frames) -> [0,1).
// `salt` decorrelates the radius / value / hue channels of one grain.
float GrainHash01(uint32_t key, uint32_t salt) {
    uint32_t x = key ^ (salt * 0x9e3779b9u);
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return static_cast<float>(x >> 8) * (1.0f / 16777216.0f);
}

// Per-grain radius scale in [1-amp, 1+amp) (clamped positive) from the global index.
float GrainRadiusScale(uint32_t g, float amp) {
    if (amp <= 0.0f) return 1.0f;
    const float s = 1.0f + amp * (2.0f * GrainHash01(g, 1u) - 1.0f);
    return s < 0.05f ? 0.05f : s;
}

// Per-grain albedo tint multiplier: a shared value (brightness) shift plus small
// per-channel hue offsets, both scaled by `amp`. amp 0 => neutral white.
Vec3 GrainTint(uint32_t g, float amp) {
    if (amp <= 0.0f) return Vec3{1.0f, 1.0f, 1.0f};
    const float dv = amp * (2.0f * GrainHash01(g, 2u) - 1.0f);
    const float hr = 0.5f * amp * (2.0f * GrainHash01(g, 3u) - 1.0f);
    const float hg = 0.5f * amp * (2.0f * GrainHash01(g, 4u) - 1.0f);
    const float hb = 0.5f * amp * (2.0f * GrainHash01(g, 5u) - 1.0f);
    auto pos = [](float x) { return x < 0.0f ? 0.0f : x; };
    return Vec3{pos(1.0f + dv + hr), pos(1.0f + dv + hg), pos(1.0f + dv + hb)};
}

}  // namespace

// Analytic spheres or octahedra share deterministic particle-indexed radius and tint.
MeshGeometry BakeParticleSpheres(const std::vector<Vec3>& pos, uint32_t first,
                                 uint32_t count, float r, bool round,
                                 float radius_jitter, float tint_jitter, uint32_t index_base) {
    MeshGeometry g;
    const uint32_t total = static_cast<uint32_t>(pos.size());
    const uint32_t lo = first < total ? first : total;
    const uint32_t n = (count == 0u) ? (total - lo)
                                     : (count < total - lo ? count : total - lo);
    if (round) {
        g.sphere_centers.reserve(static_cast<size_t>(n) * 3u);
        g.sphere_radii.reserve(n);
        if (tint_jitter > 0.0f) g.sphere_colors.reserve(static_cast<size_t>(n) * 3u);
        for (uint32_t i = 0; i < n; ++i) {
            const Vec3& p = pos[lo + i];
            const uint32_t gi = index_base + lo + i;
            g.sphere_centers.insert(g.sphere_centers.end(), {p.x, p.y, p.z});
            g.sphere_radii.push_back(r * GrainRadiusScale(gi, radius_jitter));
            if (tint_jitter > 0.0f) {
                const Vec3 t = GrainTint(gi, tint_jitter);
                g.sphere_colors.insert(g.sphere_colors.end(), {t.x, t.y, t.z});
            }
        }
        return g;
    }
    static const float kV[6][3] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0},
                                   {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    static const uint32_t kF[8][3] = {{0, 2, 4}, {2, 1, 4}, {1, 3, 4}, {3, 0, 4},
                                      {2, 0, 5}, {1, 2, 5}, {3, 1, 5}, {0, 3, 5}};
    g.positions.reserve(static_cast<size_t>(n) * 18u);
    g.indices.reserve(static_cast<size_t>(n) * 24u);
    for (uint32_t i = 0; i < n; ++i) {
        const Vec3& p = pos[lo + i];
        const uint32_t base = i * 6u;
        const float rp = r * GrainRadiusScale(index_base + lo + i, radius_jitter);
        for (int v = 0; v < 6; ++v) {
            g.positions.insert(g.positions.end(),
                               {p.x + rp * kV[v][0], p.y + rp * kV[v][1],
                                p.z + rp * kV[v][2]});
        }
        for (const auto& f : kF)
            g.indices.insert(g.indices.end(), {base + f[0], base + f[1], base + f[2]});
    }
    return g;
}

}  // namespace nuka::runtime
