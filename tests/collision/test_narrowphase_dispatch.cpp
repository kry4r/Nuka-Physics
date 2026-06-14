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

// M11 COL-1: the SDF-tier FORWARD gate, salvaged VERBATIM from the M9-deleted
// tests/collision/test_sdf_tier_wired.cpp (git show fcae2a6:...) and folded into
// this dispatch suite. The SDF forward infra (find_sdf_contact_newton + the
// NarrowphaseSdf dispatch handler) survived M9; this re-asserts the analytical
// box-on-box truth + D1 two-run byte-identity gates that proved the SDF tier
// wired. These extra includes drive the cooked-SDF host path.
#include "import/cooker/sparse_sdf_cooker.hpp"
#include "tests/import/sdf_host_sampler.hpp"
#include "tests/import/sdf_test_meshes.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <type_traits>

using nuka::collision::CandidatePair;
using nuka::collision::kNarrowphaseTable;
using nuka::collision::MakeStableKey;
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
    // Capsule pairs WITH a closed-form handler stay Analytical (C3b):
    EXPECT_EQ(SelectTier(ShapeType::Sphere, ShapeType::Capsule, false),
              NarrowphaseTier::Analytical);
    EXPECT_EQ(SelectTier(ShapeType::Capsule, ShapeType::Plane, false),
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

TEST(NarrowphaseDispatch, SelectTierCapsuleNoClosedFormGoesConvex) {
    // C3-batch review fix: capsule×box and capsule×capsule have NO closed-form
    // analytical contact -> Convex tier (C3c GJK/EPA via SupportCapsule). Both
    // orderings.
    EXPECT_EQ(SelectTier(ShapeType::Capsule, ShapeType::Box, false),
              NarrowphaseTier::Convex);
    EXPECT_EQ(SelectTier(ShapeType::Box, ShapeType::Capsule, false),
              NarrowphaseTier::Convex);
    EXPECT_EQ(SelectTier(ShapeType::Capsule, ShapeType::Capsule, false),
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
// slots stay the Convex stub (the v0.9 named deferral). PLUS the C3-batch review
// fix: capsule×box / capsule×capsule (no closed-form) also route here.
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
           is(a, b, H, P) || is(a, b, P, H) ||
           is(a, b, C, B) || is(a, b, B, C) ||   // capsule×box (review fix)
           is(a, b, C, C);                        // capsule×capsule (review fix)
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
    // An UNregistered Analytical pair still routes to the Analytical stub.
    // (Plane x Plane is degenerate -> stub by design; capsule×box now routes to
    // the Convex tier, see SelectTierCapsuleNoClosedFormGoesConvex.)
    EXPECT_EQ(ResolveNarrowphase(ShapeType::Plane, ShapeType::Plane, false),
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
    // Analytical tier: use an UNregistered pair (Plane x Plane, degenerate -> stub
    // by design) so the stub still runs and stamps its marker. (Registered
    // Analytical pairs now run real geometry math -- covered by
    // test_analytical_manifold; capsule×box now routes to the Convex tier.)
    EXPECT_EQ(RouteAndReadMarker(ShapeType::Plane, ShapeType::Plane, false),
              StubMarkerForTier(NarrowphaseTier::Analytical));
    EXPECT_EQ(RouteAndReadMarker(ShapeType::TriMesh, ShapeType::Box, false),
              StubMarkerForTier(NarrowphaseTier::Convex));
    // Sdf tier: C3d landed -- the real NarrowphaseSdf runs (no stub marker). With
    // the default ShapeProxyView (null SDF seam) it null-guards to an EMPTY
    // manifold (point_count 0; StampSides still stamps the real manifold_key from
    // the sides, NOT a stub marker), proving the real handler ran without crashing.
    // The SDF-driven manifold's forward-accuracy gate is deferred to M11 (the SDF
    // tier + find_sdf_contact_newton are KEPT per ruling #4; the gate re-asserts there).
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

TEST(NarrowphaseDispatch, RealHandlerStampsManifoldKeyForD1Sort) {
    // §0.1: the ContactManifold stream (C3->C4 handoff) is D1-sorted by
    // `manifold_key`. Every REAL handler (via StampSides) must stamp it =
    // MakeStableKey(a,b) -- NOT leave it 0 and NOT a stub marker. (Box x Box is a
    // real C3b Analytical handler; default prims overlap at the origin so it runs.)
    CandidatePair pair;
    pair.a = CollidableRef{CollidableType::RigidBody,
                           ReactionProviderKind::RigidInvMass, 7u};
    pair.b = CollidableRef{CollidableType::StaticWorld,
                           ReactionProviderKind::StaticNull, 3u};
    ShapeProxyView geom;
    geom.type_a = ShapeType::Box;
    geom.type_b = ShapeType::Box;
    ContactManifold out;
    ResolveNarrowphase(ShapeType::Box, ShapeType::Box, false)(pair, geom, &out);

    const uint64_t expected = MakeStableKey(pair.a, pair.b);
    EXPECT_EQ(out.manifold_key, expected) << "real handler must stamp the D1 sort key";
    EXPECT_NE(out.manifold_key, 0u);
    EXPECT_NE(out.manifold_key, StubMarkerForTier(NarrowphaseTier::Analytical))
        << "a real manifold_key must be distinguishable from a stub marker";
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
    // runs (Plane x Plane is degenerate -> stub by design). The stub yields an
    // empty manifold but preserves the type-tagged sides.
    ResolveNarrowphase(ShapeType::Plane, ShapeType::Plane, false)(pair, geom, &out);

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

// ===========================================================================
// M11 COL-1 -- SDF high-precision tier WIRED into the narrowphase dispatch.
// ---------------------------------------------------------------------------
// Salvaged VERBATIM from fcae2a6:tests/collision/test_sdf_tier_wired.cpp (the
// v0.8 C3d gate M9 deleted while KEEPING the SDF forward infra). Folded here so
// the dispatch test target re-asserts:
//
//   (1) ROUTING: ResolveNarrowphase(typeA, typeB, has_sdf=true) returns the REAL
//       NarrowphaseSdf handler (NOT the removed C3a stub) -- the >=1-callsite gate.
//   (2) DRIVES THE SDF QUERY: feeding the handler two SDF-equipped (cooked) boxes
//       in contact yields an SDF-based 1-point ContactManifold whose normal +
//       penetration match the analytical box-on-box contact within voxel tol.
//   (3) NORMAL SIGN: top box A on ground box B -> normal == separation dir for
//       side A == +Y (the SDF a<-b sense, mapped 1:1 to the manifold's "sep dir
//       for side A"; a flipped geom_a/geom_b binding would give -Y).
//   (4) SEPARATED pair -> EMPTY manifold (point_count 0), no crash.
//   (5) D1: two runs of the handler -> BYTE-IDENTICAL manifold (memcmp on zeroed
//       buffers, the C3 D1 gate posture).
//
// BAND-AWARE GEOMETRY (the load-bearing test-design choice). find_sdf_contact_
// newton needs the INITIAL guess in BOTH narrow bands (sdf_contact.hpp), and the
// handler seeds the guess at the MIDPOINT of the two body centers. So the test
// uses TWO EQUAL unit cubes whose centers are SYMMETRIC about y=0 and which
// overlap by ~2 voxels: the contact plane (and the midpoint) is y=0, squarely in
// both bands.
//
// The handler calls the header-inline NUKA_SDF_HD find_sdf_contact_newton -- the
// SAME function the device kernel wraps -- so this host test validates the
// SHIPPING SDF path through the dispatch seam.
// ===========================================================================

using nuka::collision::SdfBodyFrame;
using nuka::collision::SdfProxyView;
using nuka::import::cooker::CookSparseSdf;
using nuka::import::cooker::SparseSdfData;
using nuka::import::cooker::SparseSdfParams;
using nuka::math::Vec3;

namespace {

// Cook a unit cube SDF ([-0.5,0.5]^3) at the given voxel/band -- the same
// construction the existing SDF contact test uses (CookSparseSdf + the
// MakeHostView SparseSdfDevice view).
SparseSdfData CookUnitCube(float voxel, uint32_t band_voxels) {
    const auto mesh = nuka::test::UnitCubeMesh();
    SparseSdfParams p;
    p.voxel_size = voxel;
    p.band_voxels = band_voxels;
    return CookSparseSdf(mesh.vertices.data(),
                         static_cast<uint32_t>(mesh.vertices.size() / 3),
                         mesh.indices.data(),
                         static_cast<uint32_t>(mesh.indices.size() / 3), p);
}

// Identity-rotation frame at a world translation.
SdfBodyFrame FrameAt(Vec3 t) {
    SdfBodyFrame f;
    f.rot_col0 = Vec3{1.0f, 0.0f, 0.0f};
    f.rot_col1 = Vec3{0.0f, 1.0f, 0.0f};
    f.rot_col2 = Vec3{0.0f, 0.0f, 1.0f};
    f.translation = t;
    return f;
}

// A pair of SDF-equipped collidables: side A = top RigidBody box, side B = ground
// StaticWorld box, both unit cubes, centers symmetric about y=0, overlapping by
// `overlap` along Y (so the contact plane + the handler's center-midpoint guess
// are both y=0, inside both narrow bands).
struct StackedBoxes {
    nuka::test::HostSdfView view_a;  // owns the gradient buffer for sdf A
    nuka::test::HostSdfView view_b;  // owns the gradient buffer for sdf B
    SdfProxyView proxy_a;
    SdfProxyView proxy_b;
    CandidatePair pair;
};

StackedBoxes MakeStackedBoxes(const SparseSdfData& cube, float overlap) {
    StackedBoxes s;
    s.view_a = nuka::test::MakeHostView(cube);
    s.view_b = nuka::test::MakeHostView(cube);

    // A bottom face at (cy_a - 0.5) = -overlap/2 ; B top face at (cy_b + 0.5) =
    // +overlap/2 -> they overlap by `overlap`, midpoint of centers = y=0.
    const float cy = 0.5f - 0.5f * overlap;
    s.proxy_a.sdf = s.view_a.device;
    s.proxy_a.frame = FrameAt(Vec3{0.0f, cy, 0.0f});   // top box A
    s.proxy_b.sdf = s.view_b.device;
    s.proxy_b.frame = FrameAt(Vec3{0.0f, -cy, 0.0f});  // ground box B

    s.pair.a = CollidableRef{CollidableType::RigidBody,
                             ReactionProviderKind::RigidInvMass, 4u};
    s.pair.b = CollidableRef{CollidableType::StaticWorld,
                             ReactionProviderKind::StaticNull, 9u};
    return s;
}

// Build the ShapeProxyView pointing geom_a/geom_b at the SDF proxies (the C3d
// void* seam). type_a/type_b are Box -- but for the Sdf tier the shape tags are
// not read (the SDF view is the geometry); they document the cooked shapes.
ShapeProxyView MakeGeom(const StackedBoxes& s) {
    ShapeProxyView g;
    g.type_a = ShapeType::Box;
    g.type_b = ShapeType::Box;
    g.geom_a = &s.proxy_a;
    g.geom_b = &s.proxy_b;
    return g;
}

}  // namespace

// (1) + (the >=1-callsite gate): the resolved Sdf-tier handler IS the real
// NarrowphaseSdf, NOT a stub. has_sdf overrides tier for ANY (typeA,typeB).
TEST(SdfTierWired, ResolveReturnsRealSdfHandlerNotStub) {
    EXPECT_EQ(ResolveNarrowphase(ShapeType::Box, ShapeType::Box, true),
              &NarrowphaseSdf);
    // The override holds regardless of the shape tags.
    EXPECT_EQ(ResolveNarrowphase(ShapeType::Sphere, ShapeType::Capsule, true),
              &NarrowphaseSdf);
    EXPECT_EQ(ResolveNarrowphase(ShapeType::ConvexHull, ShapeType::Plane, true),
              &NarrowphaseSdf);
    // NOT the same address as the analytical/convex stubs (sanity: it is a
    // distinct, real handler).
    EXPECT_NE(reinterpret_cast<const void*>(&NarrowphaseSdf),
              reinterpret_cast<const void*>(
                  ResolveNarrowphase(ShapeType::Box, ShapeType::Box, false)));
}

// (2)+(3): box-on-box via SDF, through the dispatch handler, matches the
// analytical contact: normal == +Y (separation dir for the top box A) and
// penetration ~= the overlap, within voxel tol.
TEST(SdfTierWired, BoxOnBoxMatchesAnalyticalNormalAndPenetration) {
    const float voxel = 0.02f;
    const uint32_t band = 4u;
    const float overlap = 2.0f * voxel;  // 0.04 -- ~2 voxels of mutual penetration

    const auto cube = CookUnitCube(voxel, band);
    ASSERT_GT(cube.CellCount(), 0u);
    const auto s = MakeStackedBoxes(cube, overlap);
    const auto geom = MakeGeom(s);

    ContactManifold out;
    std::memset(&out, 0, sizeof(out));  // zeroed -> D1-clean buffer
    ResolveNarrowphase(geom.type_a, geom.type_b, /*has_sdf=*/true)(s.pair, geom,
                                                                   &out);

    // 1-point high-precision manifold.
    ASSERT_EQ(out.point_count, 1u) << "SDF tier yields a single witness point";
    const auto& pt = out.points[0];

    // Sides preserved + reaction providers tagged from the type metadata.
    EXPECT_EQ(out.a.handle, 4u);
    EXPECT_EQ(out.b.handle, 9u);
    EXPECT_EQ(out.a.type, CollidableType::RigidBody);
    EXPECT_EQ(out.b.type, CollidableType::StaticWorld);

    // Normal = separation dir for side A (the top box) = +Y. A flipped geom
    // binding (sdf_a/sdf_b swapped vs pair A/B) would give -Y -- the sign trap.
    EXPECT_GT(pt.normal.y, 0.9f) << "top box A separates +Y (a<-b SDF sense)";
    EXPECT_LT(std::fabs(pt.normal.x), 0.15f);
    EXPECT_LT(std::fabs(pt.normal.z), 0.15f);

    // Penetration ~= the overlap (voxel-limited: trilinear + cook).
    const float pen_err = std::fabs(pt.penetration - overlap);
    EXPECT_LT(pen_err, 1.5f * voxel)
        << "penetration " << pt.penetration << " vs overlap " << overlap;

    // Witness near the contact plane y~=0 (the valley midpoint), on-axis.
    EXPECT_LT(std::fabs(pt.position.y), 2.0f * voxel);

    std::printf("[sdf-tier] pts=%u normal=(%.3f,%.3f,%.3f) pen=%.4f (overlap %.4f, "
                "err %.2f vox) point=(%.3f,%.3f,%.3f)\n",
                out.point_count, pt.normal.x, pt.normal.y, pt.normal.z,
                pt.penetration, overlap, pen_err / voxel, pt.position.x,
                pt.position.y, pt.position.z);
}

// (4): SEPARATED SDF pair -> EMPTY manifold (no contact), no crash. Two unit
// cubes pulled far apart along Y: the center-midpoint guess is out of both narrow
// bands -> find_sdf_contact_newton returns invalid -> point_count 0.
TEST(SdfTierWired, SeparatedPairYieldsEmptyManifold) {
    const float voxel = 0.02f;
    const uint32_t band = 4u;
    const auto cube = CookUnitCube(voxel, band);
    ASSERT_GT(cube.CellCount(), 0u);

    auto s = MakeStackedBoxes(cube, 0.0f);
    // Pull them widely apart (gap of 2.0 along Y): midpoint is in empty space,
    // far outside both narrow bands.
    s.proxy_a.frame = FrameAt(Vec3{0.0f, 2.0f, 0.0f});
    s.proxy_b.frame = FrameAt(Vec3{0.0f, -2.0f, 0.0f});
    const auto geom = MakeGeom(s);

    ContactManifold out;
    out.AddPoint(nuka::constraint::ContactPoint{});  // dirty it; handler must Clear
    ResolveNarrowphase(geom.type_a, geom.type_b, /*has_sdf=*/true)(s.pair, geom,
                                                                   &out);
    EXPECT_EQ(out.point_count, 0u) << "separated SDF pair -> empty manifold";
    // Sides still stamped (the empty manifold is well-defined).
    EXPECT_EQ(out.a.handle, 4u);
    EXPECT_EQ(out.b.handle, 9u);
}

// (4b): null SDF seam (unpopulated geom) -> empty manifold, no crash.
TEST(SdfTierWired, NullSeamYieldsEmptyManifoldNoCrash) {
    CandidatePair pair;
    pair.a = CollidableRef{CollidableType::RigidBody,
                           ReactionProviderKind::RigidInvMass, 1u};
    pair.b = CollidableRef{CollidableType::RigidBody,
                           ReactionProviderKind::RigidInvMass, 2u};
    ShapeProxyView geom;  // geom_a/geom_b default null
    geom.type_a = ShapeType::Box;
    geom.type_b = ShapeType::Box;
    ContactManifold out;
    out.AddPoint(nuka::constraint::ContactPoint{});
    ResolveNarrowphase(ShapeType::Box, ShapeType::Box, true)(pair, geom, &out);
    EXPECT_EQ(out.point_count, 0u);
}

// (5): D1 -- two runs of the handler produce a BYTE-IDENTICAL manifold. memcmp on
// ZEROED buffers (two separately-constructed manifolds have nondeterministic
// padding; zero them first so only the handler-written bytes differ).
TEST(SdfTierWired, TwoRunByteIdenticalManifold) {
    const float voxel = 0.02f;
    const uint32_t band = 4u;
    const auto cube = CookUnitCube(voxel, band);
    ASSERT_GT(cube.CellCount(), 0u);
    const auto s = MakeStackedBoxes(cube, 2.0f * voxel);
    const auto geom = MakeGeom(s);
    const auto fn = ResolveNarrowphase(geom.type_a, geom.type_b, true);

    ContactManifold m1, m2;
    std::memset(&m1, 0, sizeof(m1));
    std::memset(&m2, 0, sizeof(m2));
    fn(s.pair, geom, &m1);
    fn(s.pair, geom, &m2);

    ASSERT_EQ(m1.point_count, 1u);
    EXPECT_EQ(0, std::memcmp(&m1, &m2, sizeof(ContactManifold)))
        << "SDF-tier manifold must be byte-identical run-to-run (D1)";
    std::printf("[sdf-tier-D1] two-run byte-exact manifold (sizeof %zu)\n",
                sizeof(ContactManifold));
}
