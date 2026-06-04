// ---------------------------------------------------------------------------
// nuka v0.8 C2a: CandidatePair / CandidatePairStream tests.
//   1. MakeStableKey is canonical + collision-free (type in high bits).
//   2. BuildCandidatePairStream sorts a fixed unsorted set ascending by key.
//   3. D1: two-run byte-exact (memcmp) on the built stream.
//   4. N>=32 cross-replica identity (the cross_system_query D1 idiom).
// ---------------------------------------------------------------------------

#include "collision/candidate_pair.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <random>
#include <vector>

namespace {

using nuka::collision::BuildCandidatePairStream;
using nuka::collision::CandidatePair;
using nuka::collision::MakeStableKey;
using nuka::collision::PackSideKey;
using nuka::constraint::CollidableRef;
using nuka::constraint::CollidableType;
using nuka::constraint::ReactionProviderKind;

CollidableRef Ref(CollidableType type, uint32_t handle) {
    CollidableRef r;
    r.type = type;
    // react does NOT enter the stable_key (handle+type only); keep a fixed value.
    r.react = ReactionProviderKind::RigidInvMass;
    r.handle = handle;
    return r;
}

CandidatePair Pair(CollidableRef a, CollidableRef b) {
    CandidatePair p;
    p.a = a;
    p.b = b;
    p.stable_key = 0ull;  // the builder recomputes it
    return p;
}

} // namespace

// (1a) Canonical: key(a,b) == key(b,a).
TEST(CandidateStream, MakeStableKeyIsCanonical) {
    const CollidableRef a = Ref(CollidableType::RigidBody, 7u);
    const CollidableRef b = Ref(CollidableType::Particle, 3u);
    EXPECT_EQ(MakeStableKey(a, b), MakeStableKey(b, a));

    // react must NOT affect the key (only type+handle pack into a side key).
    CollidableRef a2 = a;
    a2.react = ReactionProviderKind::ParticleInvMass;
    EXPECT_EQ(MakeStableKey(a, b), MakeStableKey(a2, b));
}

// (1b) Distinct (type,handle) pairs get distinct keys; same type different handle
// differ; different type same handle differ.
TEST(CandidateStream, MakeStableKeyDistinguishesPairs) {
    const auto k_r0_r1 = MakeStableKey(Ref(CollidableType::RigidBody, 0u),
                                       Ref(CollidableType::RigidBody, 1u));
    const auto k_r0_r2 = MakeStableKey(Ref(CollidableType::RigidBody, 0u),
                                       Ref(CollidableType::RigidBody, 2u));
    EXPECT_NE(k_r0_r1, k_r0_r2);  // same types, one handle differs

    const auto k_r5_p5 = MakeStableKey(Ref(CollidableType::RigidBody, 5u),
                                       Ref(CollidableType::Particle, 5u));
    const auto k_r5_r5 = MakeStableKey(Ref(CollidableType::RigidBody, 5u),
                                       Ref(CollidableType::RigidBody, 5u));
    EXPECT_NE(k_r5_p5, k_r5_r5);  // same handles, one type differs
}

// (1c) Reserve-high-bits: a (Particle,0) side key never collides with a
// (RigidBody, large) side key, even at the 28-bit handle ceiling. This is the
// property that makes type-blind keys safe across all collidable types.
TEST(CandidateStream, TypeBitsPreventCrossTypeCollision) {
    const uint32_t kMaxHandle = (1u << 28) - 1u;  // 0x0FFFFFFF
    const uint32_t particle0 = PackSideKey(Ref(CollidableType::Particle, 0u));
    const uint32_t rigidMax = PackSideKey(Ref(CollidableType::RigidBody, kMaxHandle));
    EXPECT_NE(particle0, rigidMax);
    // RigidBody (type 0) occupies side keys [0, 2^28); Particle (type 2) starts
    // at 2u<<28. So the LOWEST Particle side key is strictly above the HIGHEST
    // RigidBody side key.
    EXPECT_GT(particle0, rigidMax);
    EXPECT_EQ(particle0, 2u << 28);
    EXPECT_EQ(rigidMax, kMaxHandle);
}

// (2) Build a stream from a FIXED unsorted set -> ascending-sorted by stable_key.
TEST(CandidateStream, BuildSortsAscendingByStableKey) {
    std::vector<CandidatePair> unsorted = {
        Pair(Ref(CollidableType::Particle, 9u), Ref(CollidableType::RigidBody, 2u)),
        Pair(Ref(CollidableType::RigidBody, 0u), Ref(CollidableType::RigidBody, 5u)),
        Pair(Ref(CollidableType::ArticulationLink, 4u), Ref(CollidableType::Particle, 1u)),
        Pair(Ref(CollidableType::RigidBody, 0u), Ref(CollidableType::RigidBody, 1u)),
        Pair(Ref(CollidableType::StaticWorld, 3u), Ref(CollidableType::Particle, 8u)),
    };

    auto stream = BuildCandidatePairStream(unsorted);
    ASSERT_EQ(stream.Count(), unsorted.size());
    const auto out = stream.DownloadPairs();
    ASSERT_EQ(out.size(), unsorted.size());

    for (size_t i = 0; i < out.size(); ++i) {
        // The builder stamps the canonical key on each pair.
        EXPECT_EQ(out[i].stable_key, MakeStableKey(out[i].a, out[i].b));
        if (i > 0) {
            EXPECT_LE(out[i - 1].stable_key, out[i].stable_key)
                << "candidate stream not ascending by stable_key at index " << i;
        }
    }
}

// (3) D1: two-run byte-exact (memcmp) on the built stream.
TEST(CandidateStream, D1TwoRunByteExact) {
    std::mt19937 rng(0xC2A0u);
    std::uniform_int_distribution<uint32_t> type_dist(0u, 3u);
    std::uniform_int_distribution<uint32_t> handle_dist(0u, 4096u);
    const CollidableType types[4] = {
        CollidableType::RigidBody, CollidableType::ArticulationLink,
        CollidableType::Particle, CollidableType::StaticWorld};

    std::vector<CandidatePair> unsorted;
    unsorted.reserve(512);
    for (int i = 0; i < 512; ++i) {
        unsorted.push_back(Pair(Ref(types[type_dist(rng)], handle_dist(rng)),
                                Ref(types[type_dist(rng)], handle_dist(rng))));
    }

    auto s1 = BuildCandidatePairStream(unsorted);
    auto s2 = BuildCandidatePairStream(unsorted);
    ASSERT_EQ(s1.Count(), s2.Count());
    const auto o1 = s1.DownloadPairs();
    const auto o2 = s2.DownloadPairs();
    ASSERT_EQ(o1.size(), o2.size());
    EXPECT_EQ(std::memcmp(o1.data(), o2.data(), o1.size() * sizeof(CandidatePair)), 0)
        << "candidate stream not two-run byte-exact (D1)";
}

// (4) N>=32 cross-replica identity: build the same logical pair set in N
// independent streams; every replica must be byte-identical (the
// cross_system_query D1 idiom -- replicas are not allowed to diverge).
TEST(CandidateStream, NReplicaCrossIdentity) {
    constexpr int kReplicas = 32;
    std::mt19937 rng(0x5EEDu);
    std::uniform_int_distribution<uint32_t> handle_dist(0u, 8192u);
    const CollidableType types[4] = {
        CollidableType::RigidBody, CollidableType::ArticulationLink,
        CollidableType::Particle, CollidableType::StaticWorld};
    std::uniform_int_distribution<uint32_t> type_dist(0u, 3u);

    std::vector<CandidatePair> unsorted;
    unsorted.reserve(256);
    for (int i = 0; i < 256; ++i) {
        unsorted.push_back(Pair(Ref(types[type_dist(rng)], handle_dist(rng)),
                                Ref(types[type_dist(rng)], handle_dist(rng))));
    }

    std::vector<CandidatePair> golden;
    for (int r = 0; r < kReplicas; ++r) {
        auto stream = BuildCandidatePairStream(unsorted);
        auto out = stream.DownloadPairs();
        if (r == 0) {
            golden = std::move(out);
            continue;
        }
        ASSERT_EQ(out.size(), golden.size());
        EXPECT_EQ(std::memcmp(out.data(), golden.data(),
                              golden.size() * sizeof(CandidatePair)), 0)
            << "replica " << r << " diverged from replica 0 (D1)";
    }
}
