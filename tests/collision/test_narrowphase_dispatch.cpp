// ---------------------------------------------------------------------------
// v0.8 C3a -- narrowphase dispatch table + extended ContactManifold tests.
// ---------------------------------------------------------------------------
// Validates the REAL, shipped-now parts of C3a:
//   (1) SelectTier routes (ShapeType,ShapeType,has_sdf) to the correct tier.
//   (2) the constexpr dispatch table routes (typeA,typeB,tier) to the EXPECTED
//       tier handler -- proven two ways: looked-up fn-ptr == expected stub, AND
//       calling through the table stamps the tier's stub marker into the manifold.
//   (3) the extended ContactManifold round-trips: AddPoint up to 4 (5th dropped),
//       Clear resets, CollidableRef a/b carry type+react+handle, and the struct
//       is byte-stable (trivially copyable + memcpy round-trip + field-wise equal
//       across two identical constructions).
// The geometry MATH is deferred to C3b/c/d; this proves ROUTING, not contacts.
// ---------------------------------------------------------------------------

#include "collision/narrowphase_dispatch.hpp"
#include "constraint/collidable.hpp"
#include "constraint/contact_manifold.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <type_traits>

using nuka::collision::CandidatePair;
using nuka::collision::kNarrowphaseTable;
using nuka::collision::NarrowphaseSdf;  // C3d: real Sdf-tier handler (stub removed)
using nuka::collision::NarrowphaseStubAnalytical;
using nuka::collision::NarrowphaseStubConvex;
using nuka::collision::NarrowphaseTier;
using nuka::collision::ResolveNarrowphase;
using nuka::collision::SelectTier;
using nuka::collision::ShapeProxyView;
using nuka::collision::StubMarkerForTier;
using nuka::constraint::CollidableRef;
using nuka::constraint::CollidableType;
using nuka::constraint::ContactManifold;
using nuka::constraint::ContactPoint;
using nuka::constraint::ReactionProviderKind;
using nuka::scene::ShapeType;

// ---------------------------------------------------------------------------
// SelectTier
// ---------------------------------------------------------------------------

TEST(NarrowphaseDispatch, SelectTierPrimitivePairsAnalytical) {
    EXPECT_EQ(SelectTier(ShapeType::Sphere, ShapeType::Sphere, false),
              NarrowphaseTier::Analytical);
    EXPECT_EQ(SelectTier(ShapeType::Box, ShapeType::Plane, false),
              NarrowphaseTier::Analytical);
    EXPECT_EQ(SelectTier(ShapeType::Capsule, ShapeType::Box, false),
              NarrowphaseTier::Analytical);
    EXPECT_EQ(SelectTier(ShapeType::Sphere, ShapeType::Capsule, false),
              NarrowphaseTier::Analytical);
}

TEST(NarrowphaseDispatch, SelectTierMeshConvexHeightfieldConvex) {
    EXPECT_EQ(SelectTier(ShapeType::ConvexHull, ShapeType::Box, false),
              NarrowphaseTier::Convex);
    EXPECT_EQ(SelectTier(ShapeType::TriMesh, ShapeType::Sphere, false),
              NarrowphaseTier::Convex);
    EXPECT_EQ(SelectTier(ShapeType::HeightField, ShapeType::Box, false),
              NarrowphaseTier::Convex);
    EXPECT_EQ(SelectTier(ShapeType::ConvexHull, ShapeType::TriMesh, false),
              NarrowphaseTier::Convex);
}

TEST(NarrowphaseDispatch, SelectTierSdfOverridesEverything) {
    // SDF available -> Sdf tier regardless of shape types.
    EXPECT_EQ(SelectTier(ShapeType::Sphere, ShapeType::Sphere, true),
              NarrowphaseTier::Sdf);
    EXPECT_EQ(SelectTier(ShapeType::Box, ShapeType::Plane, true),
              NarrowphaseTier::Sdf);
    EXPECT_EQ(SelectTier(ShapeType::TriMesh, ShapeType::ConvexHull, true),
              NarrowphaseTier::Sdf);
}

// SelectTier is constexpr -> usable in a constant expression (compile-time D1).
static_assert(SelectTier(ShapeType::Sphere, ShapeType::Sphere, false) ==
                  NarrowphaseTier::Analytical,
              "primitive pair must be Analytical at compile time");
static_assert(SelectTier(ShapeType::TriMesh, ShapeType::Box, false) ==
                  NarrowphaseTier::Convex,
              "mesh pair must be Convex at compile time");
static_assert(SelectTier(ShapeType::Sphere, ShapeType::Sphere, true) ==
                  NarrowphaseTier::Sdf,
              "SDF-equipped pair must be Sdf at compile time");

// ---------------------------------------------------------------------------
// Table routing -- looked-up fn pointer matches the expected tier handler.
// ---------------------------------------------------------------------------

namespace {
// True iff (a,b) got a REAL Analytical handler in C3b (else it keeps the stub).
// Mirrors MakeNarrowphaseTable's C3b registration block.
bool HasRealAnalyticalHandler(ShapeType a, ShapeType b) {
    using nuka::scene::ShapeType;
    auto is = [](ShapeType x, ShapeType y, ShapeType s, ShapeType t) {
        return x == s && y == t;
    };
    const ShapeType S = ShapeType::Sphere, C = ShapeType::Capsule,
                    B = ShapeType::Box, P = ShapeType::Plane;
    return is(a, b, S, S) ||
           is(a, b, S, B) || is(a, b, B, S) ||
           is(a, b, S, P) || is(a, b, P, S) ||
           is(a, b, B, P) || is(a, b, P, B) ||
           is(a, b, B, B) ||
           is(a, b, C, P) || is(a, b, P, C) ||
           is(a, b, C, S) || is(a, b, S, C);
}

// Mirrors MakeNarrowphaseTable's C3c Convex registration block: every
// ConvexHull-involving pair (convex x convex + convex x {box,sphere,capsule,
// plane}, both orderings) gets the real NarrowphaseConvex; TriMesh/HeightField
// slots stay the Convex stub (the v0.9 named deferral).
bool HasRealConvexHandler(ShapeType a, ShapeType b) {
    using nuka::scene::ShapeType;
    auto is = [](ShapeType x, ShapeType y, ShapeType s, ShapeType t) {
        return x == s && y == t;
    };
    const ShapeType S = ShapeType::Sphere, C = ShapeType::Capsule,
                    B = ShapeType::Box, P = ShapeType::Plane,
                    H = ShapeType::ConvexHull;
    return is(a, b, H, H) ||
           is(a, b, H, B) || is(a, b, B, H) ||
           is(a, b, H, S) || is(a, b, S, H) ||
           is(a, b, H, C) || is(a, b, C, H) ||
           is(a, b, H, P) || is(a, b, P, H);
}
}  // namespace

TEST(NarrowphaseDispatch, LookupReturnsExpectedTierHandler) {
    // C3b registered REAL Analytical handlers for the primitive pairs listed in
    // HasRealAnalyticalHandler; C3c registered REAL Convex handlers for the
    // ConvexHull-involving pairs in HasRealConvexHandler. Every OTHER slot in each
    // tier (and ALL Sdf slots) still defaults to its tier stub.
    for (uint32_t ia = 0; ia < nuka::collision::kShapeTypeCount; ++ia) {
        for (uint32_t ib = 0; ib < nuka::collision::kShapeTypeCount; ++ib) {
            const auto a = static_cast<ShapeType>(ia);
            const auto b = static_cast<ShapeType>(ib);
            const auto ana = kNarrowphaseTable.Lookup(a, b, NarrowphaseTier::Analytical);
            if (HasRealAnalyticalHandler(a, b)) {
                EXPECT_NE(ana, &NarrowphaseStubAnalytical)
                    << "expected a REAL C3b Analytical handler for ("
                    << ia << "," << ib << ")";
            } else {
                EXPECT_EQ(ana, &NarrowphaseStubAnalytical)
                    << "expected the Analytical STUB for unregistered ("
                    << ia << "," << ib << ")";
            }
            // Convex tier: real handler for ConvexHull pairs (C3c), stub otherwise.
            const auto cvx = kNarrowphaseTable.Lookup(a, b, NarrowphaseTier::Convex);
            if (HasRealConvexHandler(a, b)) {
                EXPECT_NE(cvx, &NarrowphaseStubConvex)
                    << "expected a REAL C3c Convex handler for ("
                    << ia << "," << ib << ")";
            } else {
                EXPECT_EQ(cvx, &NarrowphaseStubConvex)
                    << "expected the Convex STUB for unregistered ("
                    << ia << "," << ib << ")";
            }
            // C3d LANDED: the WHOLE Sdf plane now routes to the REAL
            // NarrowphaseSdf handler (has_sdf overrides tier for ANY (a,b), so the
            // SDF math -- find_sdf_contact_newton on the SDF views -- covers every
            // slot; the Sdf stub is removed). No slot still points at a stub.
            EXPECT_EQ(kNarrowphaseTable.Lookup(a, b, NarrowphaseTier::Sdf),
                      &NarrowphaseSdf)
                << "every Sdf-plane slot must be the real C3d handler (" << ia
                << "," << ib << ")";
        }
    }
}

TEST(NarrowphaseDispatch, ResolveNarrowphasePicksTierThenHandler) {
    // Primitive pair, no SDF -> Analytical tier; C3b makes Sphere x Box a REAL
    // handler (no longer the stub).
    EXPECT_NE(ResolveNarrowphase(ShapeType::Sphere, ShapeType::Box, false),
              &NarrowphaseStubAnalytical);
    // An UNregistered Analytical pair still routes to the Analytical stub
    // (Capsule x Box deferred to C3c).
    EXPECT_EQ(ResolveNarrowphase(ShapeType::Capsule, ShapeType::Box, false),
              &NarrowphaseStubAnalytical);
    // Mesh pair, no SDF -> Convex handler.
    EXPECT_EQ(ResolveNarrowphase(ShapeType::TriMesh, ShapeType::Box, false),
              &NarrowphaseStubConvex);
    // SDF-equipped -> the REAL C3d Sdf handler (tier override). No longer a stub.
    EXPECT_EQ(ResolveNarrowphase(ShapeType::Sphere, ShapeType::Box, true),
              &NarrowphaseSdf);
}

// ---------------------------------------------------------------------------
// Table routing -- calling THROUGH the table reaches the right tier handler
// (the stub stamps a per-tier marker we read back).
// ---------------------------------------------------------------------------

namespace {
// Drive a pair through the resolved handler and return which tier marker it left.
uint64_t RouteAndReadMarker(ShapeType a, ShapeType b, bool has_sdf) {
    CandidatePair pair;
    pair.a = CollidableRef{CollidableType::RigidBody,
                           ReactionProviderKind::RigidInvMass, 7u};
    pair.b = CollidableRef{CollidableType::StaticWorld,
                           ReactionProviderKind::StaticNull, 3u};
    ShapeProxyView geom;
    geom.type_a = a;
    geom.type_b = b;
    ContactManifold out;
    const auto fn = ResolveNarrowphase(a, b, has_sdf);
    fn(pair, geom, &out);
    return out.manifold_key;
}
}  // namespace

TEST(NarrowphaseDispatch, CallThroughTableReachesExpectedTier) {
    // Analytical tier: use an UNregistered pair (Capsule x Box, deferred to C3c)
    // so the stub still runs and stamps its marker. (Registered Analytical pairs
    // now run real geometry math -- covered by test_analytical_manifold.)
    EXPECT_EQ(RouteAndReadMarker(ShapeType::Capsule, ShapeType::Box, false),
              StubMarkerForTier(NarrowphaseTier::Analytical));
    EXPECT_EQ(RouteAndReadMarker(ShapeType::TriMesh, ShapeType::Box, false),
              StubMarkerForTier(NarrowphaseTier::Convex));
    // Sdf tier: C3d landed -- the real NarrowphaseSdf runs (no stub marker). With
    // the default ShapeProxyView (null SDF seam) it null-guards to an EMPTY
    // manifold (manifold_key untouched == 0), proving the real handler ran without
    // crashing. The SDF-driven manifold is exercised in test_sdf_tier_wired.
    CandidatePair sdf_pair;
    sdf_pair.a = CollidableRef{CollidableType::RigidBody,
                               ReactionProviderKind::RigidInvMass, 7u};
    sdf_pair.b = CollidableRef{CollidableType::StaticWorld,
                               ReactionProviderKind::StaticNull, 3u};
    ShapeProxyView sdf_geom;  // geom_a/geom_b default null -> handler null-guards
    sdf_geom.type_a = ShapeType::Sphere;
    sdf_geom.type_b = ShapeType::Box;
    ContactManifold sdf_out;
    sdf_out.AddPoint(ContactPoint{});  // dirty it; the handler must Clear
    ASSERT_EQ(ResolveNarrowphase(ShapeType::Sphere, ShapeType::Box, true),
              &NarrowphaseSdf);
    ResolveNarrowphase(ShapeType::Sphere, ShapeType::Box, true)(sdf_pair, sdf_geom,
                                                                &sdf_out);
    EXPECT_EQ(sdf_out.point_count, 0u) << "null SDF seam -> empty manifold";
    EXPECT_EQ(sdf_out.a.handle, 7u);   // sides preserved
    EXPECT_EQ(sdf_out.b.handle, 3u);
}

TEST(NarrowphaseDispatch, StubPreservesCollidableSidesAndEmptyManifold) {
    CandidatePair pair;
    pair.a = CollidableRef{CollidableType::ArticulationLink,
                           ReactionProviderKind::ArticulationChainJ, 11u};
    pair.b = CollidableRef{CollidableType::Particle,
                           ReactionProviderKind::ParticleInvMass, 22u};
    ShapeProxyView geom;
    ContactManifold out;
    out.AddPoint(ContactPoint{});  // dirty it first to prove Clear ran

    // Use an UNregistered Analytical pair so the STUB (not a C3b real handler)
    // runs (Capsule x Box deferred to C3c). The stub yields an empty manifold.
    ResolveNarrowphase(ShapeType::Capsule, ShapeType::Box, false)(pair, geom, &out);

    EXPECT_EQ(out.point_count, 0u);  // stub yields an EMPTY manifold
    EXPECT_EQ(out.a.type, CollidableType::ArticulationLink);
    EXPECT_EQ(out.a.handle, 11u);
    EXPECT_EQ(out.b.type, CollidableType::Particle);
    EXPECT_EQ(out.b.handle, 22u);
}

// ---------------------------------------------------------------------------
// Extended ContactManifold round-trip.
// ---------------------------------------------------------------------------

TEST(ContactManifoldExtended, AddPointClampsAtFourAndClearResets) {
    ContactManifold m;
    ContactPoint pt;
    pt.penetration = 0.01f;

    m.AddPoint(pt);
    m.AddPoint(pt);
    m.AddPoint(pt);
    m.AddPoint(pt);
    EXPECT_EQ(m.point_count, ContactManifold::kMaxPoints);
    EXPECT_EQ(m.point_count, 4u);

    m.AddPoint(pt);  // 5th must be dropped
    EXPECT_EQ(m.point_count, 4u);

    m.Clear();
    EXPECT_EQ(m.point_count, 0u);
}

TEST(ContactManifoldExtended, CollidableRefSidesCarryTypeReactHandle) {
    ContactManifold m;
    m.a = CollidableRef{CollidableType::RigidBody,
                        ReactionProviderKind::RigidInvMass, 5u};
    m.b = CollidableRef{CollidableType::StaticWorld,
                        ReactionProviderKind::StaticNull, 0u};

    EXPECT_EQ(m.a.type, CollidableType::RigidBody);
    EXPECT_EQ(m.a.react, ReactionProviderKind::RigidInvMass);
    EXPECT_EQ(m.a.handle, 5u);
    EXPECT_EQ(m.b.type, CollidableType::StaticWorld);
    EXPECT_EQ(m.b.react, ReactionProviderKind::StaticNull);
    EXPECT_EQ(m.b.handle, 0u);
}

TEST(ContactManifoldExtended, DefaultsMatchRigidMaximalAndMuJoCo) {
    ContactManifold m;
    // The rigid-maximal-preserving defaults (OPEN-A): a/b default to a benign
    // RigidBody/RigidInvMass ref so a former `body_a = X` -> `a.handle = X`.
    EXPECT_EQ(m.a.type, CollidableType::RigidBody);
    EXPECT_EQ(m.a.react, ReactionProviderKind::RigidInvMass);
    EXPECT_EQ(m.a.handle, ~0u);
    EXPECT_EQ(m.b.type, CollidableType::RigidBody);
    EXPECT_EQ(m.b.handle, ~0u);
    EXPECT_FLOAT_EQ(m.friction, 0.5f);
    EXPECT_FLOAT_EQ(m.restitution, 0.0f);
    // MuJoCo solimp defaults.
    EXPECT_FLOAT_EQ(m.solimp[0], 0.9f);
    EXPECT_FLOAT_EQ(m.solimp[1], 0.95f);
    EXPECT_FLOAT_EQ(m.solimp[2], 0.001f);
    EXPECT_FLOAT_EQ(m.solimp[3], 0.5f);
    EXPECT_FLOAT_EQ(m.solimp[4], 2.0f);
    EXPECT_EQ(m.manifold_key, 0u);
    // Per-point compliant defaults.
    ContactPoint p;
    EXPECT_FLOAT_EQ(p.solref_timeconst, 0.02f);
    EXPECT_FLOAT_EQ(p.solref_dampratio, 1.0f);
    EXPECT_EQ(p.stable_key, 0u);
}

// Byte-stability: the manifold is a trivially-copyable POD aggregate, so a
// memcpy round-trip preserves every byte and two identical constructions are
// field-wise equal. (We avoid memcmp of two SEPARATELY-constructed objects: the
// CollidableRef enum/handle layout and the float arrays leave padding bytes that
// are not guaranteed equal between stack objects -- memcpy of ONE object is the
// well-defined byte-stability check.)
static_assert(std::is_trivially_copyable_v<ContactManifold>,
              "ContactManifold must be trivially copyable (D1 memcpy + device upload)");
static_assert(std::is_trivially_copyable_v<ContactPoint>,
              "ContactPoint must be trivially copyable");
static_assert(std::is_trivially_copyable_v<CollidableRef>,
              "CollidableRef must be trivially copyable");

TEST(ContactManifoldExtended, MemcpyRoundTripPreservesBytes) {
    ContactManifold m;
    m.a = CollidableRef{CollidableType::RigidBody,
                        ReactionProviderKind::RigidInvMass, 42u};
    m.b = CollidableRef{CollidableType::Particle,
                        ReactionProviderKind::ParticleInvMass, 99u};
    m.friction = 0.7f;
    m.restitution = 0.3f;
    m.solimp[2] = 0.002f;
    m.manifold_key = 0xABCD'1234'5678'9ABCull;
    ContactPoint pt;
    pt.position = {1.0f, 2.0f, 3.0f};
    pt.normal = {0.0f, 1.0f, 0.0f};
    pt.penetration = 0.05f;
    pt.stable_key = 0xDEAD'BEEFull;
    pt.solref_timeconst = 0.03f;
    pt.solref_dampratio = 0.9f;
    m.AddPoint(pt);

    unsigned char buffer[sizeof(ContactManifold)];
    std::memcpy(buffer, &m, sizeof(ContactManifold));
    ContactManifold restored;
    std::memcpy(&restored, buffer, sizeof(ContactManifold));

    // Round-tripped bytes are identical (the trivially-copyable guarantee).
    EXPECT_EQ(std::memcmp(&m, &restored, sizeof(ContactManifold)), 0);

    // ...and the fields survived (field-wise read-back).
    EXPECT_EQ(restored.a.type, CollidableType::RigidBody);
    EXPECT_EQ(restored.a.handle, 42u);
    EXPECT_EQ(restored.b.type, CollidableType::Particle);
    EXPECT_EQ(restored.b.handle, 99u);
    EXPECT_FLOAT_EQ(restored.friction, 0.7f);
    EXPECT_FLOAT_EQ(restored.restitution, 0.3f);
    EXPECT_FLOAT_EQ(restored.solimp[2], 0.002f);
    EXPECT_EQ(restored.manifold_key, 0xABCD'1234'5678'9ABCull);
    ASSERT_EQ(restored.point_count, 1u);
    EXPECT_EQ(restored.points[0].position, (nuka::math::Vec3{1.0f, 2.0f, 3.0f}));
    EXPECT_FLOAT_EQ(restored.points[0].penetration, 0.05f);
    EXPECT_EQ(restored.points[0].stable_key, 0xDEAD'BEEFull);
    EXPECT_FLOAT_EQ(restored.points[0].solref_timeconst, 0.03f);
    EXPECT_FLOAT_EQ(restored.points[0].solref_dampratio, 0.9f);
}

TEST(ContactManifoldExtended, TwoIdenticalConstructionsAreFieldwiseEqual) {
    auto build = []() {
        ContactManifold m;
        m.a = CollidableRef{CollidableType::RigidBody,
                            ReactionProviderKind::RigidInvMass, 3u};
        m.b = CollidableRef{CollidableType::StaticWorld,
                            ReactionProviderKind::StaticNull, 1u};
        m.friction = 0.42f;
        m.manifold_key = 0x1122'3344'5566'7788ull;
        ContactPoint pt;
        pt.position = {4.0f, 5.0f, 6.0f};
        pt.penetration = 0.02f;
        m.AddPoint(pt);
        return m;
    };
    const ContactManifold x = build();
    const ContactManifold y = build();

    // Field-wise (NOT memcmp of two separate objects -- padding may differ).
    EXPECT_EQ(x.a.type, y.a.type);
    EXPECT_EQ(x.a.react, y.a.react);
    EXPECT_EQ(x.a.handle, y.a.handle);
    EXPECT_EQ(x.b.type, y.b.type);
    EXPECT_EQ(x.b.handle, y.b.handle);
    EXPECT_EQ(x.point_count, y.point_count);
    EXPECT_FLOAT_EQ(x.friction, y.friction);
    EXPECT_FLOAT_EQ(x.restitution, y.restitution);
    EXPECT_EQ(x.manifold_key, y.manifold_key);
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(x.solimp[i], y.solimp[i]);
    }
    ASSERT_EQ(x.point_count, 1u);
    EXPECT_EQ(x.points[0].position, y.points[0].position);
    EXPECT_FLOAT_EQ(x.points[0].penetration, y.points[0].penetration);
}
