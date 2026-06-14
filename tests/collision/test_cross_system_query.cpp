// ---------------------------------------------------------------------------
// v0.7 p05 CORRECTNESS + D1 GATE: the cross-system particle->rigid query must
// match a CPU brute-force AABB-overlap oracle, and be two-run byte-exact.
//
// Setup: N rigid AABBs (the LEAF bounds) + M particles. Build the rigid LBVH via
// BuildLbvhForQuery (retains the node tree), then for each particle query its
// candidate rigid bodies. Oracle: for each particle, the SET of rigid bodies
// whose AABB overlaps the particle's query AABB (pos +/- query_radius). We feed
// the SAME device AABB buffer into the LBVH and the SAME AABBs into the oracle
// (apples-to-apples; the AABB overlap test is plain </> on identical inputs, no
// FMA, so it is exact on both sides).
//
// Scenes keep candidate counts <= 16 (the cap, spec 7.5.5) for the exact-match
// oracle; cap/truncation is exercised separately on a dense scene.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "collision/broadphase_lbvh.hpp"
#include "collision/cross_system_query.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer_v2.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <set>
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

collision::AABB ParticleAabb(math::Vec3 p, float r) {
    collision::AABB b;
    b.min = {p.x - r, p.y - r, p.z - r};
    b.max = {p.x + r, p.y + r, p.z + r};
    return b;
}

// CPU oracle: per-particle set of rigid body indices whose AABB overlaps the
// particle's query AABB.
std::vector<std::set<uint32_t>> CpuCrossOracle(
    const std::vector<math::Vec3>& particles, float radius,
    const std::vector<collision::AABB>& rigid) {
    std::vector<std::set<uint32_t>> out(particles.size());
    for (size_t i = 0; i < particles.size(); ++i) {
        const collision::AABB q = ParticleAabb(particles[i], radius);
        for (uint32_t j = 0; j < rigid.size(); ++j) {
            if (q.Overlaps(rigid[j])) {
                out[i].insert(j);
            }
        }
    }
    return out;
}

std::vector<std::set<uint32_t>> CandidateSets(
    const collision::gpu::CrossSystemQueryResult& r) {
    const auto offsets = r.DownloadCandidateOffsets();
    const auto counts = r.DownloadCandidateCounts();
    const auto idx = r.DownloadCandidateIndices();
    std::vector<std::set<uint32_t>> out(r.ParticleCount());
    for (uint32_t i = 0; i < r.ParticleCount(); ++i) {
        for (uint32_t k = 0; k < counts[i]; ++k) {
            out[i].insert(idx[offsets[i] + k]);
        }
    }
    return out;
}

} // namespace

// 10 rigid boxes in a row + 1000 particles scattered around them; query radius
// keeps each particle's candidate set small (<= 16).
TEST(CrossSystemQuery, MatchesBruteForceAabbOracle) {
    std::vector<collision::AABB> rigid;
    for (int i = 0; i < 10; ++i) {
        const float cx = static_cast<float>(i) * 2.0f;
        collision::AABB b;
        b.min = {cx - 0.5f, -0.5f, -0.5f};
        b.max = {cx + 0.5f, 0.5f, 0.5f};
        rigid.push_back(b);
    }

    std::mt19937 rng(0x5151u);
    std::uniform_real_distribution<float> dx(-2.0f, 20.0f);
    std::uniform_real_distribution<float> dyz(-2.0f, 2.0f);
    std::vector<math::Vec3> particles(1000);
    for (auto& p : particles) {
        p = {dx(rng), dyz(rng), dyz(rng)};
    }
    const float radius = 0.4f;

    OwnedDeviceBuffer d_rigid = UploadOwned(rigid);
    const auto* dev_rigid = static_cast<const collision::AABB*>(d_rigid.Data());
    auto tree = collision::gpu::BuildLbvhForQuery(
        dev_rigid, static_cast<uint32_t>(rigid.size()));
    ASSERT_TRUE(tree.HasNodes());

    OwnedDeviceBuffer d_part = UploadOwned(particles);
    const auto* dev_part = static_cast<const math::Vec3*>(d_part.Data());
    auto res = collision::gpu::QueryParticlesAgainstRigidLbvh(
        dev_part, static_cast<uint32_t>(particles.size()), radius, tree);

    EXPECT_EQ(res.TruncatedParticleCount(), 0u);
    const auto oracle = CpuCrossOracle(particles, radius, rigid);
    const auto got = CandidateSets(res);
    ASSERT_EQ(oracle.size(), got.size());
    for (size_t i = 0; i < oracle.size(); ++i) {
        EXPECT_EQ(oracle[i], got[i]) << "particle " << i << " candidate set mismatch";
    }
}

// Larger random rigid set; sparse particles. Still under the cap.
TEST(CrossSystemQuery, RandomRigidMatchesBruteForce) {
    std::mt19937 rng(0x2024u);
    std::uniform_real_distribution<float> pos(-20.0f, 20.0f);
    std::uniform_real_distribution<float> ext(0.2f, 0.6f);
    std::vector<collision::AABB> rigid(64);
    for (auto& b : rigid) {
        const float cx = pos(rng), cy = pos(rng), cz = pos(rng);
        const float ex = ext(rng), ey = ext(rng), ez = ext(rng);
        b.min = {cx - ex, cy - ey, cz - ez};
        b.max = {cx + ex, cy + ey, cz + ez};
    }
    std::vector<math::Vec3> particles(500);
    for (auto& p : particles) {
        p = {pos(rng), pos(rng), pos(rng)};
    }
    const float radius = 0.5f;

    OwnedDeviceBuffer d_rigid = UploadOwned(rigid);
    auto tree = collision::gpu::BuildLbvhForQuery(
        static_cast<const collision::AABB*>(d_rigid.Data()),
        static_cast<uint32_t>(rigid.size()));
    OwnedDeviceBuffer d_part = UploadOwned(particles);
    auto res = collision::gpu::QueryParticlesAgainstRigidLbvh(
        static_cast<const math::Vec3*>(d_part.Data()),
        static_cast<uint32_t>(particles.size()), radius, tree);

    ASSERT_EQ(res.TruncatedParticleCount(), 0u)
        << "scene exceeded the 16-cap; thin out rigid set for exact-match oracle";
    const auto oracle = CpuCrossOracle(particles, radius, rigid);
    const auto got = CandidateSets(res);
    for (size_t i = 0; i < oracle.size(); ++i) {
        EXPECT_EQ(oracle[i], got[i]) << "particle " << i << " candidate set mismatch";
    }
}

// D1: two-run byte-exact candidate lists.
TEST(CrossSystemQuery, D1TwoRunByteExact) {
    std::mt19937 rng(0x9090u);
    std::uniform_real_distribution<float> pos(-15.0f, 15.0f);
    std::uniform_real_distribution<float> ext(0.2f, 0.5f);
    std::vector<collision::AABB> rigid(128);
    for (auto& b : rigid) {
        const float cx = pos(rng), cy = pos(rng), cz = pos(rng);
        const float ex = ext(rng), ey = ext(rng), ez = ext(rng);
        b.min = {cx - ex, cy - ey, cz - ez};
        b.max = {cx + ex, cy + ey, cz + ez};
    }
    std::vector<math::Vec3> particles(2000);
    for (auto& p : particles) {
        p = {pos(rng), pos(rng), pos(rng)};
    }
    const float radius = 1.0f;

    OwnedDeviceBuffer d_rigid = UploadOwned(rigid);
    auto tree = collision::gpu::BuildLbvhForQuery(
        static_cast<const collision::AABB*>(d_rigid.Data()),
        static_cast<uint32_t>(rigid.size()));
    OwnedDeviceBuffer d_part = UploadOwned(particles);
    const auto* dev_part = static_cast<const math::Vec3*>(d_part.Data());

    auto r1 = collision::gpu::QueryParticlesAgainstRigidLbvh(
        dev_part, static_cast<uint32_t>(particles.size()), radius, tree);
    auto r2 = collision::gpu::QueryParticlesAgainstRigidLbvh(
        dev_part, static_cast<uint32_t>(particles.size()), radius, tree);

    ASSERT_EQ(r1.TotalCandidates(), r2.TotalCandidates());
    EXPECT_EQ(r1.DownloadCandidateOffsets(), r2.DownloadCandidateOffsets());
    EXPECT_EQ(r1.DownloadCandidateCounts(), r2.DownloadCandidateCounts());
    EXPECT_EQ(r1.DownloadCandidateIndices(), r2.DownloadCandidateIndices())
        << "cross-system candidate list not two-run byte-exact (D1)";
}

// 16-candidate cap (spec 7.5.5): a particle whose query AABB overlaps MANY rigid
// bodies is truncated to the 16 LOWEST body ids (deterministic subset). Asserts
// the cap is enforced + surfaced + two-run byte-exact, and that the kept subset
// equals the 16 smallest of the full brute-force overlap set.
TEST(CrossSystemQuery, CapTruncationSurfacedAndDeterministic) {
    // 50 small rigid AABBs packed tightly around the origin.
    std::vector<collision::AABB> rigid;
    std::mt19937 rng(0x7777u);
    std::uniform_real_distribution<float> jitter(-0.3f, 0.3f);
    for (int i = 0; i < 50; ++i) {
        collision::AABB b;
        const float cx = jitter(rng), cy = jitter(rng), cz = jitter(rng);
        b.min = {cx - 0.1f, cy - 0.1f, cz - 0.1f};
        b.max = {cx + 0.1f, cy + 0.1f, cz + 0.1f};
        rigid.push_back(b);
    }
    // One particle at the origin whose query AABB covers all 50 boxes.
    std::vector<math::Vec3> particles = {{0.0f, 0.0f, 0.0f}};
    const float radius = 2.0f;

    OwnedDeviceBuffer d_rigid = UploadOwned(rigid);
    auto tree = collision::gpu::BuildLbvhForQuery(
        static_cast<const collision::AABB*>(d_rigid.Data()),
        static_cast<uint32_t>(rigid.size()));
    OwnedDeviceBuffer d_part = UploadOwned(particles);
    const auto* dev_part = static_cast<const math::Vec3*>(d_part.Data());

    auto r1 = collision::gpu::QueryParticlesAgainstRigidLbvh(dev_part, 1u, radius, tree);
    auto r2 = collision::gpu::QueryParticlesAgainstRigidLbvh(dev_part, 1u, radius, tree);

    // (a) truncation surfaced; (b) capped.
    EXPECT_GT(r1.TruncatedParticleCount(), 0u)
        << "expected a particle overlapping 50 bodies to exceed the 16-cap";
    EXPECT_EQ(r1.TruncatedParticleCount(), r2.TruncatedParticleCount());
    const auto counts = r1.DownloadCandidateCounts();
    ASSERT_EQ(counts.size(), 1u);
    EXPECT_EQ(counts[0], collision::gpu::kCrossSystemMaxCandidates);

    // (c) two-run byte-exact under truncation.
    EXPECT_EQ(r1.DownloadCandidateIndices(), r2.DownloadCandidateIndices())
        << "truncated candidate list not two-run byte-exact (D1)";

    // (d) kept subset == 16 smallest of the full brute-force overlap set, sorted.
    const auto full = CpuCrossOracle(particles, radius, rigid)[0];
    ASSERT_GT(full.size(), collision::gpu::kCrossSystemMaxCandidates);
    std::vector<uint32_t> expected(full.begin(), full.end()); // std::set -> ascending
    expected.resize(collision::gpu::kCrossSystemMaxCandidates);
    const auto idx = r1.DownloadCandidateIndices();
    const auto offsets = r1.DownloadCandidateOffsets();
    std::vector<uint32_t> got(idx.begin() + offsets[0],
                              idx.begin() + offsets[0] + counts[0]);
    EXPECT_EQ(expected, got)
        << "truncated subset is not the 16 lowest-index candidates";
}

// Single rigid body (leaf_count==1) path: candidates resolve against the lone
// leaf node retained by BuildLbvhForQuery.
TEST(CrossSystemQuery, SingleRigidBody) {
    std::vector<collision::AABB> rigid(1);
    rigid[0].min = {-1.0f, -1.0f, -1.0f};
    rigid[0].max = {1.0f, 1.0f, 1.0f};
    std::vector<math::Vec3> particles = {
        {0.0f, 0.0f, 0.0f},   // inside -> candidate
        {5.0f, 5.0f, 5.0f}};  // far -> no candidate
    const float radius = 0.1f;

    OwnedDeviceBuffer d_rigid = UploadOwned(rigid);
    auto tree = collision::gpu::BuildLbvhForQuery(
        static_cast<const collision::AABB*>(d_rigid.Data()), 1u);
    ASSERT_TRUE(tree.HasNodes());
    OwnedDeviceBuffer d_part = UploadOwned(particles);
    auto res = collision::gpu::QueryParticlesAgainstRigidLbvh(
        static_cast<const math::Vec3*>(d_part.Data()), 2u, radius, tree);

    const auto counts = res.DownloadCandidateCounts();
    const auto offsets = res.DownloadCandidateOffsets();
    const auto idx = res.DownloadCandidateIndices();
    ASSERT_EQ(counts.size(), 2u);
    EXPECT_EQ(counts[0], 1u);
    EXPECT_EQ(idx[offsets[0]], 0u);
    EXPECT_EQ(counts[1], 0u);
}
