// ---------------------------------------------------------------------------
// v0.7 p05 CORRECTNESS GATE: the GPU particle uniform-grid neighbor lists must
// match a CPU brute-force O(N^2) neighbor oracle on small scenes.
//
// Oracle: for every particle, the set of OTHER particles within query_radius
// (squared-distance test, self excluded). The grid output is CSR-like; we
// reconstruct each particle's neighbor set and compare to the oracle SET.
//
// To stay inside the 16-candidate cap (spec 7.5.5) -- below which the grid is an
// EXACT match to brute force -- the correctness scenes keep every particle's
// neighbor count <= 16. The cap / truncation behavior is exercised separately in
// the determinism test (a deliberately dense scene). Distance margins are kept
// clear (no neighbor sits exactly on the radius) so GPU FMA vs CPU non-FMA float
// rounding cannot flip an in/out decision.
// ---------------------------------------------------------------------------

#include "collision/particle_uniform_grid.hpp"
#include "math/vec3.hpp"
#include "phi/buffer_legacy.hpp"
#include "phi/buffer_transfer.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <set>
#include <vector>

using namespace nuka;

namespace {

// CPU brute-force neighbor oracle: per-particle set of in-radius OTHER indices.
std::vector<std::set<uint32_t>> CpuNeighborOracle(
    const std::vector<math::Vec3>& pts, float radius) {
    const uint32_t n = static_cast<uint32_t>(pts.size());
    const float r2 = radius * radius;
    std::vector<std::set<uint32_t>> out(n);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            const float dx = pts[j].x - pts[i].x;
            const float dy = pts[j].y - pts[i].y;
            const float dz = pts[j].z - pts[i].z;
            if (dx * dx + dy * dy + dz * dz <= r2) {
                out[i].insert(j);
            }
        }
    }
    return out;
}

// Reconstruct per-particle neighbor sets from the CSR grid result.
std::vector<std::set<uint32_t>> GridNeighborSets(
    const collision::gpu::ParticleGridResult& grid) {
    const auto offsets = grid.DownloadNeighborOffsets();
    const auto counts = grid.DownloadNeighborCounts();
    const auto indices = grid.DownloadNeighborIndices();
    const uint32_t n = grid.ParticleCount();
    std::vector<std::set<uint32_t>> out(n);
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t k = 0; k < counts[i]; ++k) {
            out[i].insert(indices[offsets[i] + k]);
        }
    }
    return out;
}

// Build the grid over a host point set and a query radius. cell_size = radius
// (so a 27-cell search covers all neighbors within the radius).
collision::gpu::ParticleGridResult RunGrid(const std::vector<math::Vec3>& pts,
                                           float radius) {
    math::Vec3 lo = pts[0], hi = pts[0];
    for (const auto& p : pts) {
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
    }
    const auto cfg = collision::gpu::MakeParticleGridConfig(lo, hi, radius);
    phi::Buffer d_pos = phi::UploadVector(pts);
    const auto* dev = static_cast<const math::Vec3*>(d_pos.Data());
    return collision::gpu::BuildParticleUniformGrid(
        dev, static_cast<uint32_t>(pts.size()), cfg, radius);
}

} // namespace

// 10x10 grid of particles spaced 1.0 apart; radius 1.5 -> 4-connected + diagonal
// neighbors. Every particle has <= 8 neighbors (under the cap).
TEST(ParticleGridCorrectness, LatticeMatchesBruteForce) {
    std::vector<math::Vec3> pts;
    for (int x = 0; x < 10; ++x) {
        for (int y = 0; y < 10; ++y) {
            pts.push_back({static_cast<float>(x), static_cast<float>(y), 0.0f});
        }
    }
    const float radius = 1.5f; // includes axis (1.0) + diagonal (1.414) neighbors

    auto grid = RunGrid(pts, radius);
    const auto oracle = CpuNeighborOracle(pts, radius);
    const auto got = GridNeighborSets(grid);

    EXPECT_EQ(grid.TruncatedParticleCount(), 0u) << "scene unexpectedly truncated";
    ASSERT_EQ(oracle.size(), got.size());
    for (size_t i = 0; i < oracle.size(); ++i) {
        EXPECT_EQ(oracle[i], got[i]) << "particle " << i << " neighbor set mismatch";
    }
}

// 100 random particles in a 5^3 box; radius 0.8. Sparse enough that no particle
// exceeds kParticleGridMaxNeighbors (32). Margins are statistically clear (uniform
// in a continuum; exact-radius hits have measure zero).
TEST(ParticleGridCorrectness, RandomSparseMatchesBruteForce) {
    std::mt19937 rng(0xBEEFu);
    std::uniform_real_distribution<float> d(0.0f, 5.0f);
    std::vector<math::Vec3> pts(100);
    for (auto& p : pts) {
        p = {d(rng), d(rng), d(rng)};
    }
    const float radius = 0.8f;

    auto grid = RunGrid(pts, radius);
    const auto oracle = CpuNeighborOracle(pts, radius);
    const auto got = GridNeighborSets(grid);

    ASSERT_EQ(grid.TruncatedParticleCount(), 0u)
        << "scene exceeded the neighbor cap (32); lower density for an exact-match oracle";
    ASSERT_EQ(oracle.size(), got.size());
    for (size_t i = 0; i < oracle.size(); ++i) {
        EXPECT_EQ(oracle[i], got[i]) << "particle " << i << " neighbor set mismatch";
    }
}

// A single particle has no neighbors (CSR well-formed, count 0).
TEST(ParticleGridCorrectness, SingleParticleNoNeighbors) {
    std::vector<math::Vec3> pts = {{1.0f, 2.0f, 3.0f}};
    auto grid = RunGrid(pts, 1.0f);
    const auto counts = grid.DownloadNeighborCounts();
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts[0], 0u);
    EXPECT_EQ(grid.TotalNeighbors(), 0u);
}

// Neighbor lists are sorted ascending by neighbor index within each particle.
TEST(ParticleGridCorrectness, NeighborListsSortedAscending) {
    std::vector<math::Vec3> pts;
    for (int x = 0; x < 6; ++x) {
        for (int y = 0; y < 6; ++y) {
            pts.push_back({static_cast<float>(x), static_cast<float>(y), 0.0f});
        }
    }
    auto grid = RunGrid(pts, 1.5f);
    const auto offsets = grid.DownloadNeighborOffsets();
    const auto counts = grid.DownloadNeighborCounts();
    const auto indices = grid.DownloadNeighborIndices();
    for (uint32_t i = 0; i < grid.ParticleCount(); ++i) {
        for (uint32_t k = 1; k < counts[i]; ++k) {
            EXPECT_LT(indices[offsets[i] + k - 1u], indices[offsets[i] + k])
                << "particle " << i << " neighbor list not strictly ascending";
        }
    }
}
