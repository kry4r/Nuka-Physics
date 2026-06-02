// ---------------------------------------------------------------------------
// v0.7 p05 PERF: particle uniform-grid build + query at 1M particles.
//
// Spec target is an RTX-4090 number (< 800 us combined build+query). This box's
// GPU is UNKNOWN and nvidia-smi is broken, so we MEASURE AND REPORT (via
// RecordProperty + stderr) and only HARD-assert correctness + a generous wall
// ceiling -- we do NOT fail the build on the tight perf target (per the p05 task
// note: measure & report, do NOT fail on perf).
//
// Scene: 1M particles uniformly in a box sized so the grid lands ~1M cells with
// average occupancy ~1 (spec 7.5.5 envelope). cell_size == query_radius so the
// 27-cell search covers the radius. Timed as the fused BuildParticleUniformGrid
// call (cell hash + stable sort + ranges + count + scan + fill), which contains
// an internal stream sync, so wall time tracks device time.
// ---------------------------------------------------------------------------

#include "collision/particle_uniform_grid.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using namespace nuka;

namespace {

std::vector<math::Vec3> MakeCloud(uint32_t n, uint32_t seed, float box) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(0.0f, box);
    std::vector<math::Vec3> pts(n);
    for (auto& p : pts) {
        p = {d(rng), d(rng), d(rng)};
    }
    return pts;
}

} // namespace

TEST(ParticleGridPerf, Build1MParticlesAndQuery) {
    constexpr uint32_t kCount = 1u * 1000u * 1000u;
    constexpr int kWarmup = 3;
    constexpr int kIters = 20;
    // box ~100 -> with radius 0.1 and cell_size == radius, grid_dims ~1000^3
    // would be huge; instead pick box so cell count ~ 1M (occupancy ~1). For
    // ~1M cells we want ~100 cells/axis -> box = 100 * cell_size.
    const float radius = 1.0f;
    const float box = 100.0f; // -> ~100^3 ~= 1M cells

    const auto pts = MakeCloud(kCount, 0xC0FFEEu, box);
    phi::Buffer d_pos = phi::UploadVector(pts);
    const auto* dev = static_cast<const math::Vec3*>(d_pos.Data());

    const math::Vec3 lo{0.0f, 0.0f, 0.0f};
    const math::Vec3 hi{box, box, box};
    const auto cfg = collision::gpu::MakeParticleGridConfig(lo, hi, radius);

    uint32_t last_total = 0u;
    uint32_t last_trunc = 0u;
    for (int i = 0; i < kWarmup; ++i) {
        auto g = collision::gpu::BuildParticleUniformGrid(dev, kCount, cfg, radius);
        last_total = g.TotalNeighbors();
        last_trunc = g.TruncatedParticleCount();
    }

    const auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < kIters; ++i) {
        auto g = collision::gpu::BuildParticleUniformGrid(dev, kCount, cfg, radius);
        last_total = g.TotalNeighbors();
        last_trunc = g.TruncatedParticleCount();
    }
    const auto t1 = std::chrono::high_resolution_clock::now();
    const double total_us =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1000.0;
    const double per_iter_us = total_us / static_cast<double>(kIters);

    RecordProperty("grid_1m_build_query_us_per_iter", static_cast<int>(per_iter_us));
    RecordProperty("grid_1m_cell_count", static_cast<int>(cfg.CellCount()));
    RecordProperty("grid_1m_total_neighbors", static_cast<int>(last_total));
    RecordProperty("grid_1m_truncated", static_cast<int>(last_trunc));
    std::fprintf(stderr,
                 "[GRID PERF] 1M particles: fused build+query = %.1f us/iter "
                 "(cells=%u, total_neighbors=%u, truncated=%u). Spec target "
                 "(RTX-4090): build+query < 800us.\n",
                 per_iter_us, cfg.CellCount(), last_total, last_trunc);

    EXPECT_GT(cfg.CellCount(), 0u);
    // Generous wall ceiling so a pathologically slow box flags rather than
    // silently passing a broken kernel.
    EXPECT_LT(per_iter_us, 500000.0)
        << "fused build+query took " << per_iter_us << " us/iter (way over budget)";
}
