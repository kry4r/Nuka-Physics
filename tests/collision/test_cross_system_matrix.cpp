// ---------------------------------------------------------------------------
// nuka v0.8 C2c VALIDATION GATES: the particle<->rigid + particle<->particle
// candidate-pair builders (particle_candidate_pairs.{hpp,cu}) must:
//   1. p<->r equivalence: the builder's (particle, rigid-BODY) candidate SET ==
//      the raw QueryParticlesAgainstRigidLbvh CSR reframed BY HAND (particle p
//      paired with body_ids[each candidate SHAPE s]) on the same scene. Uses a
//      NON-IDENTITY shape->body map incl. a 2-shapes->1-body case to exercise the
//      reframe + the adjacent dedup (not the query itself -- that is
//      test_cross_system_query's job).
//   2. p<->p equivalence: the builder's set == the raw grid neighbor CSR reframed
//      (i<j) AND == an independent brute-force radius oracle (i<j set).
//   3. SystemPairMatrix gate: same scene, toggle one bit -> disabled == EMPTY,
//      enabled == non-empty (non-vacuous).
//   4. D1 + scale: >=32 particles -> 2-run byte-exact; Count() > 0.
//   5. sanity: 0 particles -> empty, no crash.
//
// Set comparison is by the canonical (type, handle) pair carried in stable_key
// (react rides along deterministically; we compare on stable_key which packs
// type+handle only).
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "collision/broadphase_lbvh.hpp"
#include "collision/candidate_pair.hpp"
#include "collision/cross_system_query.hpp"
#include "collision/particle_candidate_pairs.hpp"
#include "collision/particle_uniform_grid.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer_v2.hpp"
#include "scene/cooked_blob.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <set>
#include <utility>
#include <vector>

using namespace nuka;
using nuka::collision::BuildParticleParticleCandidatePairs;
using nuka::collision::BuildParticleRigidCandidatePairs;
using nuka::collision::CandidatePair;
using nuka::collision::CandidatePairStream;
using nuka::constraint::CollidableType;
using nuka::scene::SetSystemPairEnabled;
using nuka::scene::SystemKind;
using nuka::scene::SystemPairMatrix;

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

// A canonical (type,handle)-pair set extracted from a stream by its stable_key.
// stable_key packs ONLY type+handle (canonical min/max), so it uniquely names the
// unordered pair regardless of react -- the right key to compare sets on.
std::set<uint64_t> StreamKeySet(const CandidatePairStream& s) {
    std::set<uint64_t> out;
    for (const auto& p : s.DownloadPairs()) {
        out.insert(p.stable_key);
    }
    return out;
}

// Build an LBVH (retain nodes) over a host AABB set.
struct RigidScene {
    OwnedDeviceBuffer d_aabbs;           // keep alive for the tree's lifetime
    collision::gpu::LbvhBroadphaseResult tree;
};

collision::AABB BoxAabb(math::Vec3 c, float h) {
    collision::AABB b;
    b.min = {c.x - h, c.y - h, c.z - h};
    b.max = {c.x + h, c.y + h, c.z + h};
    return b;
}

// Reframe the raw cross-system CSR BY HAND into the canonical (Particle p,
// RigidBody body_ids[s]) key set -- the gate-1 oracle (same query, hand reframe).
std::set<uint64_t> RawCrossReframeKeySet(
    const collision::gpu::CrossSystemQueryResult& q,
    const std::vector<uint32_t>& body_ids) {
    const auto offsets = q.DownloadCandidateOffsets();
    const auto counts = q.DownloadCandidateCounts();
    const auto idx = q.DownloadCandidateIndices();
    std::set<uint64_t> out;
    for (uint32_t p = 0; p < q.ParticleCount(); ++p) {
        for (uint32_t k = 0; k < counts[p]; ++k) {
            const uint32_t shape = idx[offsets[p] + k];
            const uint32_t body = body_ids[shape];
            constraint::CollidableRef a;
            a.type = CollidableType::Particle;
            a.handle = p;
            constraint::CollidableRef b;
            b.type = CollidableType::RigidBody;
            b.handle = body;
            out.insert(collision::MakeStableKey(a, b));
        }
    }
    return out;
}

// Brute-force i<j radius oracle -> canonical (Particle,Particle) key set.
std::set<uint64_t> BruteParticleKeySet(const std::vector<math::Vec3>& pts,
                                       float radius) {
    const float r2 = radius * radius;
    std::set<uint64_t> out;
    for (uint32_t i = 0; i < pts.size(); ++i) {
        for (uint32_t j = i + 1; j < pts.size(); ++j) {
            const float dx = pts[j].x - pts[i].x;
            const float dy = pts[j].y - pts[i].y;
            const float dz = pts[j].z - pts[i].z;
            if (dx * dx + dy * dy + dz * dz <= r2) {
                constraint::CollidableRef a;
                a.type = CollidableType::Particle;
                a.handle = i;
                constraint::CollidableRef b;
                b.type = CollidableType::Particle;
                b.handle = j;
                out.insert(collision::MakeStableKey(a, b));
            }
        }
    }
    return out;
}

collision::gpu::ParticleGridConfig MakeCfg(const std::vector<math::Vec3>& pts,
                                           float radius) {
    math::Vec3 lo = pts[0], hi = pts[0];
    for (const auto& p : pts) {
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
    }
    return collision::gpu::MakeParticleGridConfig(lo, hi, radius);
}

} // namespace

// ===========================================================================
// GATE 1: particle<->rigid equivalence (builder set == hand-reframed raw CSR).
// Non-identity shape->body map; 2 shapes (4,5) share body 99 -> exercises dedup.
// ===========================================================================
TEST(CrossSystemMatrix, ParticleRigidEquivalence) {
    // 6 rigid shape AABBs. Shapes 0..3 are spread along x = 0,2,4,6; shapes 4 and
    // 5 are CO-LOCATED at (8,0,0) and both map to the SAME body (99). A particle
    // near (8,0,0) is then a candidate of BOTH shapes -> two CSR entries collapse
    // to one body-pair, which is the multi-shape-body dedup this gate must prove.
    std::vector<collision::AABB> rigid;
    for (int i = 0; i < 4; ++i) {
        rigid.push_back(BoxAabb({static_cast<float>(i) * 2.0f, 0.0f, 0.0f}, 0.5f));
    }
    rigid.push_back(BoxAabb({8.0f, 0.0f, 0.0f}, 0.5f));  // shape 4 -> body 99
    rigid.push_back(BoxAabb({8.0f, 0.0f, 0.0f}, 0.5f));  // shape 5 -> body 99 (co-located)
    // NON-identity shape->body map; shapes 4 and 5 map to the SAME body (99).
    const std::vector<uint32_t> body_ids = {10u, 20u, 30u, 40u, 99u, 99u};

    // Particles scattered along the row so candidate counts stay small (<16), PLUS
    // one explicit particle at (8,0,0) guaranteed to see both shape 4 and shape 5.
    std::mt19937 rng(0xC2C0u);
    std::uniform_real_distribution<float> dx(-1.0f, 9.0f);
    std::uniform_real_distribution<float> dyz(-1.0f, 1.0f);
    std::vector<math::Vec3> particles(200);
    for (auto& p : particles) p = {dx(rng), dyz(rng), dyz(rng)};
    particles.push_back({8.0f, 0.0f, 0.0f});  // sees both co-located shapes -> body 99
    const float radius = 0.4f;

    OwnedDeviceBuffer d_rigid = UploadOwned(rigid);
    auto tree = collision::gpu::BuildLbvhForQuery(
        static_cast<const collision::AABB*>(d_rigid.Data()),
        static_cast<uint32_t>(rigid.size()));
    ASSERT_TRUE(tree.HasNodes());
    ASSERT_EQ(tree.LeafCount(), rigid.size());

    OwnedDeviceBuffer d_part = UploadOwned(particles);
    const auto* dev_part = static_cast<const math::Vec3*>(d_part.Data());

    // Raw query (oracle source) + the builder, on the SAME scene.
    auto raw = collision::gpu::QueryParticlesAgainstRigidLbvh(
        dev_part, static_cast<uint32_t>(particles.size()), radius, tree);
    ASSERT_EQ(raw.TruncatedParticleCount(), 0u);

    SystemPairMatrix matrix;  // default all-enabled
    auto stream = BuildParticleRigidCandidatePairs(
        dev_part, static_cast<uint32_t>(particles.size()), SystemKind::Fluid,
        tree, body_ids.data(), radius, matrix);

    const auto oracle = RawCrossReframeKeySet(raw, body_ids);
    const auto got = StreamKeySet(stream);
    EXPECT_EQ(oracle, got) << "builder p<->r set != hand-reframed raw CSR set";
    EXPECT_GT(stream.Count(), 0u);
    // NON-VACUITY: the raw CSR has MORE flat entries than unique body-pairs because
    // shapes 4 and 5 (co-located, both -> body 99) generate two shape candidates
    // that map to the SAME body-pair. The dedup must collapse them, so the deduped
    // stream Count equals the unique-key count, strictly below the raw flat total.
    const auto raw_counts = raw.DownloadCandidateCounts();
    uint32_t raw_flat = 0u;
    for (uint32_t c : raw_counts) raw_flat += c;
    EXPECT_LT(oracle.size(), raw_flat)
        << "scene did not actually produce a multi-shape->body duplicate";
    EXPECT_EQ(stream.Count(), oracle.size())
        << "builder did not collapse the multi-shape->body duplicate";
}

// ===========================================================================
// GATE 2: particle<->particle equivalence (builder set == grid CSR i<j reframe
// AND == brute-force radius oracle).
// ===========================================================================
TEST(CrossSystemMatrix, ParticleParticleEquivalence) {
    // 8x8 lattice spaced 1.0; radius 1.5 -> axis + diagonal neighbors (<32 cap).
    std::vector<math::Vec3> pts;
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y)
            pts.push_back({static_cast<float>(x), static_cast<float>(y), 0.0f});
    const float radius = 1.5f;

    OwnedDeviceBuffer d_pos = UploadOwned(pts);
    const auto* dev = static_cast<const math::Vec3*>(d_pos.Data());
    const auto cfg = MakeCfg(pts, radius);

    // Sanity: the raw grid is not truncated (else the oracle wouldn't match).
    auto grid = collision::gpu::BuildParticleUniformGrid(
        dev, static_cast<uint32_t>(pts.size()), cfg, radius);
    ASSERT_EQ(grid.TruncatedParticleCount(), 0u);

    SystemPairMatrix matrix;
    auto stream = BuildParticleParticleCandidatePairs(
        dev, static_cast<uint32_t>(pts.size()), SystemKind::Fluid, cfg, radius,
        matrix);

    const auto brute = BruteParticleKeySet(pts, radius);
    const auto got = StreamKeySet(stream);
    EXPECT_EQ(brute, got) << "builder p<->p set != brute-force i<j radius oracle";
    EXPECT_GT(stream.Count(), 0u);
    // Canonical (i<j) emit + adjacent dedup -> each unordered pair appears once.
    EXPECT_EQ(stream.Count(), got.size());
}

// ===========================================================================
// GATE 3: SystemPairMatrix gate (non-vacuous -- same scene, toggle one bit).
// ===========================================================================
TEST(CrossSystemMatrix, MatrixGateParticleRigid) {
    std::vector<collision::AABB> rigid;
    for (int i = 0; i < 4; ++i)
        rigid.push_back(BoxAabb({static_cast<float>(i) * 2.0f, 0.0f, 0.0f}, 0.5f));
    const std::vector<uint32_t> body_ids = {0u, 1u, 2u, 3u};

    std::vector<math::Vec3> particles;
    for (int i = 0; i < 40; ++i)
        particles.push_back({static_cast<float>(i % 4) * 2.0f, 0.0f, 0.0f});
    const float radius = 0.4f;

    OwnedDeviceBuffer d_rigid = UploadOwned(rigid);
    auto tree = collision::gpu::BuildLbvhForQuery(
        static_cast<const collision::AABB*>(d_rigid.Data()),
        static_cast<uint32_t>(rigid.size()));
    OwnedDeviceBuffer d_part = UploadOwned(particles);
    const auto* dev_part = static_cast<const math::Vec3*>(d_part.Data());

    SystemPairMatrix enabled;  // default all-true
    auto on = BuildParticleRigidCandidatePairs(
        dev_part, static_cast<uint32_t>(particles.size()), SystemKind::Fluid,
        tree, body_ids.data(), radius, enabled);
    EXPECT_GT(on.Count(), 0u) << "enabled Fluid<->Rigid should be non-empty";

    SystemPairMatrix disabled;
    SetSystemPairEnabled(disabled, SystemKind::Fluid, SystemKind::Rigid, false);
    auto off = BuildParticleRigidCandidatePairs(
        dev_part, static_cast<uint32_t>(particles.size()), SystemKind::Fluid,
        tree, body_ids.data(), radius, disabled);
    EXPECT_EQ(off.Count(), 0u) << "disabled Fluid<->Rigid must be EMPTY";
}

TEST(CrossSystemMatrix, MatrixGateParticleParticle) {
    std::vector<math::Vec3> pts;
    for (int x = 0; x < 8; ++x)
        for (int y = 0; y < 8; ++y)
            pts.push_back({static_cast<float>(x), static_cast<float>(y), 0.0f});
    const float radius = 1.5f;

    OwnedDeviceBuffer d_pos = UploadOwned(pts);
    const auto* dev = static_cast<const math::Vec3*>(d_pos.Data());
    const auto cfg = MakeCfg(pts, radius);

    SystemPairMatrix enabled;
    auto on = BuildParticleParticleCandidatePairs(
        dev, static_cast<uint32_t>(pts.size()), SystemKind::Fluid, cfg, radius,
        enabled);
    EXPECT_GT(on.Count(), 0u) << "enabled Fluid<->Fluid should be non-empty";

    SystemPairMatrix disabled;
    SetSystemPairEnabled(disabled, SystemKind::Fluid, SystemKind::Fluid, false);
    auto off = BuildParticleParticleCandidatePairs(
        dev, static_cast<uint32_t>(pts.size()), SystemKind::Fluid, cfg, radius,
        disabled);
    EXPECT_EQ(off.Count(), 0u) << "disabled Fluid<->Fluid must be EMPTY";
}

// ===========================================================================
// GATE 4: D1 (two-run byte-exact) + scale (>=32 particles), both builders.
// ===========================================================================
TEST(CrossSystemMatrix, D1TwoRunByteExactParticleRigid) {
    std::vector<collision::AABB> rigid;
    for (int i = 0; i < 5; ++i)
        rigid.push_back(BoxAabb({static_cast<float>(i) * 2.0f, 0.0f, 0.0f}, 0.5f));
    const std::vector<uint32_t> body_ids = {7u, 7u, 13u, 21u, 34u};  // 0,1 share body 7

    std::mt19937 rng(0xD1A0u);
    std::uniform_real_distribution<float> dx(-1.0f, 9.0f);
    std::uniform_real_distribution<float> dyz(-1.0f, 1.0f);
    std::vector<math::Vec3> particles(64);  // >= 32
    for (auto& p : particles) p = {dx(rng), dyz(rng), dyz(rng)};
    const float radius = 0.4f;

    OwnedDeviceBuffer d_rigid = UploadOwned(rigid);
    auto tree = collision::gpu::BuildLbvhForQuery(
        static_cast<const collision::AABB*>(d_rigid.Data()),
        static_cast<uint32_t>(rigid.size()));
    OwnedDeviceBuffer d_part = UploadOwned(particles);
    const auto* dev_part = static_cast<const math::Vec3*>(d_part.Data());
    SystemPairMatrix matrix;

    auto s1 = BuildParticleRigidCandidatePairs(
        dev_part, 64u, SystemKind::Fluid, tree, body_ids.data(), radius, matrix);
    auto s2 = BuildParticleRigidCandidatePairs(
        dev_part, 64u, SystemKind::Fluid, tree, body_ids.data(), radius, matrix);
    ASSERT_EQ(s1.Count(), s2.Count());
    EXPECT_GT(s1.Count(), 0u);
    const auto o1 = s1.DownloadPairs();
    const auto o2 = s2.DownloadPairs();
    ASSERT_EQ(o1.size(), o2.size());
    EXPECT_EQ(std::memcmp(o1.data(), o2.data(), o1.size() * sizeof(CandidatePair)), 0)
        << "p<->r stream not two-run byte-exact (D1)";
}

TEST(CrossSystemMatrix, D1TwoRunByteExactParticleParticle) {
    std::mt19937 rng(0xD1B0u);
    std::uniform_real_distribution<float> d(0.0f, 5.0f);
    std::vector<math::Vec3> pts(64);  // >= 32
    for (auto& p : pts) p = {d(rng), d(rng), d(rng)};
    const float radius = 0.8f;

    OwnedDeviceBuffer d_pos = UploadOwned(pts);
    const auto* dev = static_cast<const math::Vec3*>(d_pos.Data());
    const auto cfg = MakeCfg(pts, radius);
    SystemPairMatrix matrix;

    auto s1 = BuildParticleParticleCandidatePairs(
        dev, 64u, SystemKind::Fluid, cfg, radius, matrix);
    auto s2 = BuildParticleParticleCandidatePairs(
        dev, 64u, SystemKind::Fluid, cfg, radius, matrix);
    ASSERT_EQ(s1.Count(), s2.Count());
    EXPECT_GT(s1.Count(), 0u);
    const auto o1 = s1.DownloadPairs();
    const auto o2 = s2.DownloadPairs();
    ASSERT_EQ(o1.size(), o2.size());
    EXPECT_EQ(std::memcmp(o1.data(), o2.data(), o1.size() * sizeof(CandidatePair)), 0)
        << "p<->p stream not two-run byte-exact (D1)";
}

// ===========================================================================
// GATE 5: sanity -- 0 particles -> empty, no crash (both builders).
// ===========================================================================
TEST(CrossSystemMatrix, ZeroParticlesEmpty) {
    std::vector<collision::AABB> rigid = {BoxAabb({0.0f, 0.0f, 0.0f}, 0.5f)};
    const std::vector<uint32_t> body_ids = {0u};
    OwnedDeviceBuffer d_rigid = UploadOwned(rigid);
    auto tree = collision::gpu::BuildLbvhForQuery(
        static_cast<const collision::AABB*>(d_rigid.Data()), 1u);
    SystemPairMatrix matrix;

    auto pr = BuildParticleRigidCandidatePairs(
        nullptr, 0u, SystemKind::Fluid, tree, body_ids.data(), 0.4f, matrix);
    EXPECT_EQ(pr.Count(), 0u);

    collision::gpu::ParticleGridConfig cfg;  // 1x1x1 default is fine for 0 particles
    auto pp = BuildParticleParticleCandidatePairs(
        nullptr, 0u, SystemKind::Fluid, cfg, 1.0f, matrix);
    EXPECT_EQ(pp.Count(), 0u);
}
