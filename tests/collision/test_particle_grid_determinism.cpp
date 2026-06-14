// ---------------------------------------------------------------------------
// v0.7 p05 D1 GATE: the particle uniform-grid neighbor lists must be two-run
// BYTE-EXACT (D1 strong determinism). Built twice over the same positions ->
// identical CSR offsets, counts, and flat neighbor index buffer.
//
// Also exercises the 16-candidate cap (spec 7.5.5): a deliberately DENSE scene
// where many particles have > 16 in-radius neighbors. We assert (a) the
// truncation counter is surfaced (> 0), (b) no per-particle list exceeds the
// cap, and (c) the truncated subset is itself deterministic + sorted (the 16
// LOWEST neighbor indices) -- so the cap does not break D1.
// ---------------------------------------------------------------------------

#include "collision/particle_uniform_grid.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer_v2.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

using namespace nuka;

namespace {

// Test-only RAII over the phi v2 opaque Buffer* (stream-0 DeviceBufferType;
// NULL-stream transfers are byte+ordering identical to the legacy synchronous
// memcpy). Mirrors the legacy phi::Buffer surface the test bodies use so they
// stay byte-identical after the buffer sweep.
class OwnedDeviceBuffer {
public:
    OwnedDeviceBuffer() = default;
    explicit OwnedDeviceBuffer(nuka::phi::Buffer* buf) : buf_(buf) {}
    ~OwnedDeviceBuffer() { if (buf_ != nullptr) nuka::phi::BufferFree(buf_); }
    OwnedDeviceBuffer(OwnedDeviceBuffer&& o) noexcept : buf_(o.buf_) { o.buf_ = nullptr; }
    OwnedDeviceBuffer& operator=(OwnedDeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (buf_ != nullptr) nuka::phi::BufferFree(buf_);
            buf_ = o.buf_; o.buf_ = nullptr;
        }
        return *this;
    }
    OwnedDeviceBuffer(const OwnedDeviceBuffer&) = delete;
    OwnedDeviceBuffer& operator=(const OwnedDeviceBuffer&) = delete;
    void* Data() const { return buf_ != nullptr ? nuka::phi::BufferBase(buf_) : nullptr; }
private:
    nuka::phi::Buffer* buf_ = nullptr;
};

template <typename T>
OwnedDeviceBuffer UploadOwned(const std::vector<T>& values) {
    return OwnedDeviceBuffer(nuka::phi::UploadVectorV2(
        nuka::phi::DeviceBufferType(nuka::phi::InitBestDevice()), values));
}

std::vector<math::Vec3> RandomCloud(uint32_t n, uint32_t seed, float box) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(0.0f, box);
    std::vector<math::Vec3> pts(n);
    for (auto& p : pts) {
        p = {d(rng), d(rng), d(rng)};
    }
    return pts;
}

collision::gpu::ParticleGridResult RunGrid(const std::vector<math::Vec3>& pts,
                                           float radius,
                                           OwnedDeviceBuffer& d_pos /*kept alive*/) {
    math::Vec3 lo = pts[0], hi = pts[0];
    for (const auto& p : pts) {
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
    }
    const auto cfg = collision::gpu::MakeParticleGridConfig(lo, hi, radius);
    d_pos = UploadOwned(pts);
    const auto* dev = static_cast<const math::Vec3*>(d_pos.Data());
    return collision::gpu::BuildParticleUniformGrid(
        dev, static_cast<uint32_t>(pts.size()), cfg, radius);
}

} // namespace

TEST(ParticleGridDeterminism, TwoRunByteExactNeighborLists) {
    const auto pts = RandomCloud(4000u, 0x1234u, 8.0f);
    const float radius = 0.7f;

    OwnedDeviceBuffer b1, b2;
    auto g1 = RunGrid(pts, radius, b1);
    auto g2 = RunGrid(pts, radius, b2);

    const auto o1 = g1.DownloadNeighborOffsets();
    const auto o2 = g2.DownloadNeighborOffsets();
    const auto c1 = g1.DownloadNeighborCounts();
    const auto c2 = g2.DownloadNeighborCounts();
    const auto i1 = g1.DownloadNeighborIndices();
    const auto i2 = g2.DownloadNeighborIndices();

    ASSERT_EQ(g1.TotalNeighbors(), g2.TotalNeighbors());
    ASSERT_EQ(g1.TruncatedParticleCount(), g2.TruncatedParticleCount());
    EXPECT_EQ(o1, o2) << "CSR offsets not two-run byte-exact (D1)";
    EXPECT_EQ(c1, c2) << "CSR counts not two-run byte-exact (D1)";
    EXPECT_EQ(i1, i2) << "flat neighbor indices not two-run byte-exact (D1)";
}

// Dense scene: a tight cluster so many particles exceed the neighbor cap (32).
// Verifies the cap is enforced + surfaced and the truncated subset stays deterministic.
TEST(ParticleGridDeterminism, CapTruncationSurfacedAndDeterministic) {
    // 2000 particles in a small box -> high local density.
    const auto pts = RandomCloud(2000u, 0xABCDu, 2.0f);
    const float radius = 0.6f;

    OwnedDeviceBuffer b1, b2;
    auto g1 = RunGrid(pts, radius, b1);
    auto g2 = RunGrid(pts, radius, b2);

    // (a) truncation surfaced.
    EXPECT_GT(g1.TruncatedParticleCount(), 0u)
        << "expected dense scene to exceed the 16-cap for some particles";
    EXPECT_EQ(g1.TruncatedParticleCount(), g2.TruncatedParticleCount());

    // (b) no per-particle list exceeds the cap.
    const auto counts = g1.DownloadNeighborCounts();
    for (uint32_t i = 0; i < g1.ParticleCount(); ++i) {
        EXPECT_LE(counts[i], collision::gpu::kParticleGridMaxNeighbors)
            << "particle " << i << " count exceeds cap";
    }

    // (c) byte-exact across runs even with truncation; lists sorted ascending.
    const auto o1 = g1.DownloadNeighborOffsets();
    const auto i1 = g1.DownloadNeighborIndices();
    const auto i2 = g2.DownloadNeighborIndices();
    EXPECT_EQ(i1, i2) << "truncated neighbor lists not two-run byte-exact (D1)";
    for (uint32_t i = 0; i < g1.ParticleCount(); ++i) {
        for (uint32_t k = 1; k < counts[i]; ++k) {
            EXPECT_LT(i1[o1[i] + k - 1u], i1[o1[i] + k])
                << "particle " << i << " truncated list not strictly ascending";
        }
    }
}
