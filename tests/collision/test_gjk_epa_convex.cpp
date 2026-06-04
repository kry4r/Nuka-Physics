// ---------------------------------------------------------------------------
// v0.8 C3c -- convex narrowphase (GJK / EPA / face-clip) tests.
// ---------------------------------------------------------------------------
// Drives the REAL Convex-tier handler (NarrowphaseConvex) registered in the C3a
// dispatch table (narrowphase_dispatch.hpp + convex_narrowphase.hpp):
//   - two penetrating tetrahedra (as convex hulls) -> EPA depth + normal vs a
//     hand oracle.
//   - two penetrating boxes (as convex hulls) -> depth/normal vs analytic, AND a
//     TIGHT cross-check of depth + normal against the C3b box-box SAT result
//     (point sets/keys intentionally NOT compared -- different feature schemes).
//   - a clearly-separated pair -> GJK reports no overlap (empty manifold).
//   - convex (hull) x box / convex (hull) x sphere -> correctness.
//   - convex (hull) x plane (both orderings) -> the plane special path + sign.
//   - 2-run bit-identity (memset+memcmp on the whole manifold) + N>=32 cross-
//     replica (>=32 identical pairs -> byte-identical manifolds) on a SYMMETRIC,
//     tie-prone box-box-as-hull config (the EPA-determinism gate).
//   - routing: ResolveNarrowphase(ConvexHull, ConvexHull, false) == real handler.
// ---------------------------------------------------------------------------

#include "collision/narrowphase_dispatch.hpp"
#include "collision/convex_narrowphase.hpp"
#include "collision/analytical_manifold.hpp"
#include "constraint/collidable.hpp"
#include "constraint/contact_manifold.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

using nuka::collision::CandidatePair;
using nuka::collision::NarrowphaseConvex;
using nuka::collision::NarrowphaseStubConvex;
using nuka::collision::ResolveNarrowphase;
using nuka::collision::ShapeProxyView;
using nuka::collision::amf::BuildPrimFrame;
using nuka::collision::amf::PrimParams;
using nuka::collision::cvx::ConvexHullView;
using nuka::constraint::CollidableRef;
using nuka::constraint::CollidableType;
using nuka::constraint::ContactManifold;
using nuka::constraint::ReactionProviderKind;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::scene::ShapeType;

namespace {

constexpr float kTol = 2.0e-3f;  // EPA converges to kEpaTol=1e-4; slack for accum.

// --- Hull backing storage (must outlive the ConvexHullView pointing at it). ---
// A unit AXIS-ALIGNED box's 8 corners in mesh-local space (half-extent `he`),
// in the C3b corner order (bit0=x,bit1=y,bit2=z) -- order does not matter to the
// support scan, but keep it fixed for determinism clarity.
std::vector<float> BoxVerts(Vec3 he) {
    std::vector<float> v;
    for (uint32_t c = 0; c < 8u; ++c) {
        const float sx = (c & 1u) ? he.x : -he.x;
        const float sy = (c & 2u) ? he.y : -he.y;
        const float sz = (c & 4u) ? he.z : -he.z;
        v.push_back(sx); v.push_back(sy); v.push_back(sz);
    }
    return v;
}

// A regular-ish tetrahedron's 4 verts in mesh-local space (around the origin).
std::vector<float> TetraVerts(float s) {
    // 4 verts of a tetra inscribed in a cube (the standard alternating corners).
    return {
         s,  s,  s,
         s, -s, -s,
        -s,  s, -s,
        -s, -s,  s,
    };
}

ConvexHullView MakeHull(const std::vector<float>& verts, Vec3 pos,
                        Quat rot = Quat::Identity()) {
    ConvexHullView h;
    h.verts = verts.data();
    h.vcount = static_cast<uint32_t>(verts.size() / 3u);
    h.frame = BuildPrimFrame(Transform{pos, rot});
    return h;
}

PrimParams MakeBoxPrim(Vec3 he, Vec3 pos, Quat rot = Quat::Identity()) {
    PrimParams p;
    p.half_extents = he;
    p.frame = BuildPrimFrame(Transform{pos, rot});
    return p;
}
PrimParams MakeSpherePrim(float r, Vec3 pos) {
    PrimParams p;
    p.radius = r;
    p.frame = BuildPrimFrame(Transform{pos, Quat::Identity()});
    return p;
}
PrimParams MakePlanePrim(Vec3 pos, Quat rot = Quat::Identity()) {
    PrimParams p;
    p.frame = BuildPrimFrame(Transform{pos, rot});
    return p;
}
PrimParams MakeCapsulePrim(float r, float hh, Vec3 pos, Quat rot = Quat::Identity()) {
    PrimParams p;
    p.radius = r;
    p.half_height = hh;
    p.frame = BuildPrimFrame(Transform{pos, rot});
    return p;
}

// --- Independent penetration-depth oracle (the GENERAL definition, no closed
// form needed). For two convex sets and a unit direction `n`, the overlap along
// n is support_A(-n) + support_B(n) where support_X(d) = max over X's WORLD verts
// of v.Dot(d). The TRUE penetration depth is the overlap along the MTV axis (the
// emitted normal), and for ANY axis the overlap is an UPPER bound on the true
// depth (no axis separates tighter than the MTV) -> so depth_along(any u) >= true
// depth. This is the brute-force oracle EPA's face-plane distance must match. */
float SupportMaxWorld(const std::vector<float>& verts, Vec3 pos, Vec3 dir) {
    float best = -3.4e38f;
    const uint32_t n = static_cast<uint32_t>(verts.size() / 3u);
    for (uint32_t i = 0; i < n; ++i) {
        const Vec3 w{verts[i * 3 + 0] + pos.x, verts[i * 3 + 1] + pos.y,
                     verts[i * 3 + 2] + pos.z};
        const float d = w.Dot(dir);
        if (d > best) best = d;
    }
    return best;
}
// Overlap of two AXIS-ALIGNED-placed hulls (verts in local + translation pos)
// along direction `n`: support_A(-n) + support_B(+n).
float OverlapAlong(const std::vector<float>& va, Vec3 pa,
                   const std::vector<float>& vb, Vec3 pb, Vec3 n) {
    return SupportMaxWorld(va, pa, -n) + SupportMaxWorld(vb, pb, n);
}

// Build a candidate pair with ZEROED padding. CollidableRef has 2 pad bytes after
// its two uint8 enums (type@0, react@1, handle@4 -> bytes 2-3 are padding); a
// stack temporary leaves those garbage, and StampSides' `out->a = pair.a` would
// copy that garbage into the manifold -> a memcmp-visible 2-run/replica DIVERGENCE
// that is a TEST ARTIFACT (production pairs come from zeroed device buffers via the
// thrust-built CandidatePairStream). memset + FIELD assignment (not `pair.a =
// CollidableRef{...}`, which re-imports a temporary's padding) keeps the pair, and
// hence the stamped manifold, byte-stable -- the proper way to exercise the EPA D1
// gate without a false padding failure.
CandidatePair MakePair() {
    CandidatePair pair;
    std::memset(&pair, 0, sizeof(pair));
    pair.a.type = CollidableType::RigidBody;
    pair.a.react = ReactionProviderKind::RigidInvMass;
    pair.a.handle = 1u;
    pair.b.type = CollidableType::RigidBody;
    pair.b.react = ReactionProviderKind::RigidInvMass;
    pair.b.handle = 2u;
    return pair;
}

// Route a hull-vs-hull pair through the resolved dispatch handler. The manifold
// is memset to 0 first so a 2-run memcmp covers padding + unused point slots.
ContactManifold RouteHullHull(const ConvexHullView& a, const ConvexHullView& b) {
    ShapeProxyView g;
    g.type_a = ShapeType::ConvexHull;
    g.type_b = ShapeType::ConvexHull;
    g.geom_a = &a;
    g.geom_b = &b;
    ContactManifold out;
    std::memset(&out, 0, sizeof(out));
    ResolveNarrowphase(g.type_a, g.type_b, false)(MakePair(), g, &out);
    return out;
}

bool Vec3Near(Vec3 a, Vec3 b, float tol = kTol) {
    return std::fabs(a.x - b.x) < tol && std::fabs(a.y - b.y) < tol &&
           std::fabs(a.z - b.z) < tol;
}

}  // namespace

// ===========================================================================
// Routing sanity.
// ===========================================================================
TEST(GjkEpaConvex, RoutesToRealConvexHandler) {
    auto fn = ResolveNarrowphase(ShapeType::ConvexHull, ShapeType::ConvexHull, false);
    EXPECT_EQ(fn, &NarrowphaseConvex);
    EXPECT_NE(fn, &NarrowphaseStubConvex);
    // convex x box, both orders, also resolve to the real handler.
    EXPECT_EQ(ResolveNarrowphase(ShapeType::ConvexHull, ShapeType::Box, false),
              &NarrowphaseConvex);
    EXPECT_EQ(ResolveNarrowphase(ShapeType::Box, ShapeType::ConvexHull, false),
              &NarrowphaseConvex);
    // convex x plane resolves to the real handler (plane special path).
    EXPECT_EQ(ResolveNarrowphase(ShapeType::ConvexHull, ShapeType::Plane, false),
              &NarrowphaseConvex);
    // TriMesh stays the stub (named deferral to v0.9).
    EXPECT_EQ(ResolveNarrowphase(ShapeType::TriMesh, ShapeType::TriMesh, false),
              &NarrowphaseStubConvex);
}

// ===========================================================================
// Two penetrating tetrahedra -> EPA depth/normal vs hand oracle.
// ===========================================================================
// Two tetra (vert reach 0.5 along each axis from BoxVerts-style corners). Place
// them overlapping along +X: A at origin, B shifted -dx so they interpenetrate.
TEST(GjkEpaConvex, TetraTetraPenetration) {
    const Vec3 pa{0.0f, 0.0f, 0.0f};
    const Vec3 pb{0.6f, 0.0f, 0.0f};
    const auto va = TetraVerts(0.5f);
    const auto vb = TetraVerts(0.5f);
    // A centered at origin, B centered at (0.6,0,0): the tetras' x-extent is
    // [-0.5,0.5]; B's leftmost vert reaches 0.6-0.5=0.1 < A's 0.5 -> overlap.
    const ConvexHullView A = MakeHull(va, pa);
    const ConvexHullView B = MakeHull(vb, pb);
    const ContactManifold m = RouteHullHull(A, B);
    ASSERT_GT(m.point_count, 0u) << "expected a contact";
    const Vec3 n = m.points[0].normal;
    // (1) normal is ~unit.
    EXPECT_NEAR(std::sqrt(n.LengthSq()), 1.0f, 1.0e-2f);
    // (2) DEPTH ORACLE: the emitted penetration must equal the brute-force overlap
    //     of the two vertex sets along the emitted normal (the general definition;
    //     this proves EPA actually CONVERGED on a triangular face, not just that
    //     depth>0). OverlapAlong(.,n) == support_A(-n)+support_B(n).
    const float d_oracle = OverlapAlong(va, pa, vb, pb, n);
    EXPECT_NEAR(m.points[0].penetration, d_oracle, kTol)
        << "EPA depth must match the support-difference oracle along the normal";
    // (3) NORMAL is the MINIMUM-translation axis: no sampled unit axis separates
    //     tighter than the emitted depth (overlap along ANY axis >= true depth).
    const Vec3 axes[7] = {Vec3{1,0,0}, Vec3{0,1,0}, Vec3{0,0,1},
                          Vec3{0.577f,0.577f,0.577f}, Vec3{-0.577f,0.577f,0.577f},
                          Vec3{0.707f,0.707f,0.0f}, Vec3{0.0f,0.707f,0.707f}};
    for (const Vec3& u : axes) {
        const float ov = OverlapAlong(va, pa, vb, pb, u);
        EXPECT_GE(ov, m.points[0].penetration - kTol)
            << "found an axis separating tighter than the EPA normal (n not MTV)";
    }
    EXPECT_GT(m.points[0].penetration, 0.0f);
    // Sides preserved from the pair.
    EXPECT_EQ(m.a.handle, 1u);
    EXPECT_EQ(m.b.handle, 2u);
}

// ===========================================================================
// Two penetrating boxes (as hulls) -> depth/normal vs analytic + C3b SAT.
// ===========================================================================
TEST(GjkEpaConvex, BoxBoxAsHull_DepthNormalVsAnalytic) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const auto vb = BoxVerts(he);
    // A at origin, B at (0.8,0,0): overlap on X = (0.5+0.5) - 0.8 = 0.2. Faces are
    // axis-aligned -> reference/incident on the +/-X faces, normal ~ +/-X.
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    const ConvexHullView B = MakeHull(vb, Vec3{0.8f, 0.0f, 0.0f});
    const ContactManifold m = RouteHullHull(A, B);
    ASSERT_GT(m.point_count, 0u);
    // Analytic: depth = 0.2, normal = sep dir for A = -X (A must move -X off B).
    const float kExpectDepth = 0.2f;
    const Vec3  kExpectNormalA{-1.0f, 0.0f, 0.0f};
    for (uint32_t i = 0; i < m.point_count; ++i) {
        EXPECT_NEAR(m.points[i].penetration, kExpectDepth, kTol);
        EXPECT_TRUE(Vec3Near(m.points[i].normal, kExpectNormalA, 1.0e-2f))
            << "n=(" << m.points[i].normal.x << "," << m.points[i].normal.y
            << "," << m.points[i].normal.z << ")";
    }
    EXPECT_EQ(m.point_count, 4u) << "axis-aligned face-face -> 4-pt patch";

    // --- TIGHT cross-check vs C3b box-box SAT (depth + normal only). ---
    using nuka::collision::amf::BoxBox;
    PrimParams pa = MakeBoxPrim(he, Vec3{0.0f, 0.0f, 0.0f});
    PrimParams pb = MakeBoxPrim(he, Vec3{0.8f, 0.0f, 0.0f});
    ContactManifold sat;
    BoxBox(pa, pb, &sat);
    ASSERT_GT(sat.point_count, 0u);
    EXPECT_NEAR(sat.points[0].penetration, kExpectDepth, kTol);
    // Both paths agree on depth + normal direction (NOT point ids/sets).
    EXPECT_NEAR(m.points[0].penetration, sat.points[0].penetration, kTol);
    EXPECT_TRUE(Vec3Near(m.points[0].normal, sat.points[0].normal, 1.0e-2f));
}

// ===========================================================================
// Separated pair -> GJK reports no overlap (empty manifold).
// ===========================================================================
TEST(GjkEpaConvex, SeparatedPairNoContact) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const auto vb = BoxVerts(he);
    // A at origin, B at (3,0,0): gap = 3 - 1 = 2.0 (clearly separated, far from
    // the iteration cap's near-touch false-positive zone).
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    const ConvexHullView B = MakeHull(vb, Vec3{3.0f, 0.0f, 0.0f});
    const ContactManifold m = RouteHullHull(A, B);
    EXPECT_EQ(m.point_count, 0u) << "separated -> no contact";
}

// ===========================================================================
// Convex (hull) x box (primitive) -> correctness.
// ===========================================================================
TEST(GjkEpaConvex, HullVsBoxPrimitive) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    const PrimParams B = MakeBoxPrim(he, Vec3{0.0f, 0.8f, 0.0f});  // overlap on Y=0.2
    ShapeProxyView g;
    g.type_a = ShapeType::ConvexHull;
    g.type_b = ShapeType::Box;
    g.geom_a = &A;
    g.prim_b = B;
    ContactManifold m;
    std::memset(&m, 0, sizeof(m));
    ResolveNarrowphase(g.type_a, g.type_b, false)(MakePair(), g, &m);
    ASSERT_GT(m.point_count, 0u);
    for (uint32_t i = 0; i < m.point_count; ++i) {
        EXPECT_NEAR(m.points[i].penetration, 0.2f, kTol);
        // sep dir for A (hull) = -Y (hull pushes down off the box above it).
        EXPECT_TRUE(Vec3Near(m.points[i].normal, Vec3{0.0f, -1.0f, 0.0f}, 1.0e-2f));
    }
}

// ===========================================================================
// Convex (hull) x sphere (primitive) -> 1-pt witness (sphere has no face).
// ===========================================================================
TEST(GjkEpaConvex, HullVsSpherePrimitive) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    // Sphere radius 0.5 centered at (0.8,0,0): nearest box face is +X at x=0.5,
    // sphere left edge at 0.3 -> overlap 0.2 along +X.
    const PrimParams B = MakeSpherePrim(0.5f, Vec3{0.8f, 0.0f, 0.0f});
    ShapeProxyView g;
    g.type_a = ShapeType::ConvexHull;
    g.type_b = ShapeType::Sphere;
    g.geom_a = &A;
    g.prim_b = B;
    ContactManifold m;
    std::memset(&m, 0, sizeof(m));
    ResolveNarrowphase(g.type_a, g.type_b, false)(MakePair(), g, &m);
    ASSERT_EQ(m.point_count, 1u) << "sphere side has no face -> 1-pt witness";
    EXPECT_NEAR(m.points[0].penetration, 0.2f, kTol);
    // sep dir for A (hull) = -X (hull pushes away from sphere on +X side).
    EXPECT_TRUE(Vec3Near(m.points[0].normal, Vec3{-1.0f, 0.0f, 0.0f}, 1.0e-2f));
}

// ===========================================================================
// Convex (hull) x capsule (primitive) -> exercises SupportCapsule + 1-pt witness.
// ===========================================================================
TEST(GjkEpaConvex, HullVsCapsulePrimitive) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    // Capsule radius 0.25, half-height 0.5 (axis = local Y), centered at
    // (0.65,0,0): its nearest surface point is the radius-0.25 sphere swept along
    // Y; closest approach to the box +X face (x=0.5) is at x = 0.65-0.25 = 0.40 <
    // 0.5 -> overlap 0.10 along +X. (Capsule has no flat face -> 1-pt witness.)
    const PrimParams B = MakeCapsulePrim(0.25f, 0.5f, Vec3{0.65f, 0.0f, 0.0f});
    ShapeProxyView g;
    g.type_a = ShapeType::ConvexHull;
    g.type_b = ShapeType::Capsule;
    g.geom_a = &A;
    g.prim_b = B;
    ContactManifold m;
    std::memset(&m, 0, sizeof(m));
    ResolveNarrowphase(g.type_a, g.type_b, false)(MakePair(), g, &m);
    ASSERT_EQ(m.point_count, 1u) << "capsule side has no face -> 1-pt witness";
    EXPECT_NEAR(m.points[0].penetration, 0.10f, kTol);
    // sep dir for A (hull) = -X (hull pushes away from the capsule on +X).
    EXPECT_TRUE(Vec3Near(m.points[0].normal, Vec3{-1.0f, 0.0f, 0.0f}, 1.0e-2f));
}

// ===========================================================================
// C3-batch review fix: capsule x box / capsule x capsule have NO closed-form, so
// SelectTier routes them to the CONVEX tier (two primitive SupportProxies through
// the SAME GJK/EPA). These pairs USED to fall to the Analytical stub (silent empty
// manifold) -- prove they now produce a real contact.
// ===========================================================================
TEST(GjkEpaConvex, CapsuleBoxPenetration) {
    // Both sides are PRIMITIVES (no hull): capsule(A) at origin, box(B) at +X.
    // capsule radius 0.25 (right surface x=0.25); box he 0.5 at x=0.6 (left face
    // x=0.10) -> overlap 0.15 along X. sep dir for A(capsule) = -X.
    ASSERT_EQ(ResolveNarrowphase(ShapeType::Capsule, ShapeType::Box, false),
              &NarrowphaseConvex)
        << "capsule x box must route to the real Convex handler (review fix)";
    const PrimParams A = MakeCapsulePrim(0.25f, 0.5f, Vec3{0.0f, 0.0f, 0.0f});
    const PrimParams B = MakeBoxPrim(Vec3{0.5f, 0.5f, 0.5f}, Vec3{0.6f, 0.0f, 0.0f});
    ShapeProxyView g;
    g.type_a = ShapeType::Capsule;
    g.type_b = ShapeType::Box;
    g.prim_a = A;
    g.prim_b = B;
    ContactManifold m;
    std::memset(&m, 0, sizeof(m));
    ResolveNarrowphase(g.type_a, g.type_b, false)(MakePair(), g, &m);
    ASSERT_GT(m.point_count, 0u) << "capsule x box must produce a contact, not the stub";
    EXPECT_NEAR(m.points[0].penetration, 0.15f, 1.0e-2f);
    EXPECT_TRUE(Vec3Near(m.points[0].normal, Vec3{-1.0f, 0.0f, 0.0f}, 1.0e-2f));
}

TEST(GjkEpaConvex, CapsuleCapsulePenetration) {
    // Two parallel (axis-Y) capsules, radius 0.3, centers 0.5 apart in X ->
    // combined radius 0.6 > 0.5 -> overlap 0.1 along X. sep dir for A = -X.
    ASSERT_EQ(ResolveNarrowphase(ShapeType::Capsule, ShapeType::Capsule, false),
              &NarrowphaseConvex)
        << "capsule x capsule must route to the real Convex handler (review fix)";
    const PrimParams A = MakeCapsulePrim(0.3f, 0.5f, Vec3{0.0f, 0.0f, 0.0f});
    const PrimParams B = MakeCapsulePrim(0.3f, 0.5f, Vec3{0.5f, 0.0f, 0.0f});
    ShapeProxyView g;
    g.type_a = ShapeType::Capsule;
    g.type_b = ShapeType::Capsule;
    g.prim_a = A;
    g.prim_b = B;
    ContactManifold m;
    std::memset(&m, 0, sizeof(m));
    ResolveNarrowphase(g.type_a, g.type_b, false)(MakePair(), g, &m);
    ASSERT_GT(m.point_count, 0u) << "capsule x capsule must produce a contact, not the stub";
    EXPECT_NEAR(m.points[0].penetration, 0.10f, 1.0e-2f);
    // Parallel-axis capsules => the CSO is degenerate (a line of equally-close
    // points along the shared Y axis), the EPA-hard case. The depth is exact and
    // the normal is -X to ~1% (a ~0.01 off-axis wobble on the degenerate CSO),
    // so widen the normal tol slightly -- this is EVIDENCE the EPA stays robust
    // on the parallel-degenerate config, not a sign error.
    EXPECT_TRUE(Vec3Near(m.points[0].normal, Vec3{-1.0f, 0.0f, 0.0f}, 2.0e-2f));
}

// ===========================================================================
// Convex (hull) x plane -> the plane special path, BOTH orderings (sign bug).
// ===========================================================================
TEST(GjkEpaConvex, HullVsPlane_BothOrders) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto vh = BoxVerts(he);
    // Box resting so its bottom corners dip below the plane at y=0 by 0.1:
    // box center at y = 0.4 -> bottom corners at y = -0.1 (penetration 0.1).
    const ConvexHullView H = MakeHull(vh, Vec3{0.0f, 0.4f, 0.0f});
    const PrimParams P = MakePlanePrim(Vec3{0.0f, 0.0f, 0.0f});  // plane normal +Y

    // Order 1: hull = A, plane = B. sep dir for A (hull) = +Y (push hull up).
    {
        ShapeProxyView g;
        g.type_a = ShapeType::ConvexHull;
        g.type_b = ShapeType::Plane;
        g.geom_a = &H;
        g.prim_b = P;
        ContactManifold m;
        std::memset(&m, 0, sizeof(m));
        ResolveNarrowphase(g.type_a, g.type_b, false)(MakePair(), g, &m);
        ASSERT_GT(m.point_count, 0u);
        EXPECT_EQ(m.point_count, 4u) << "4 bottom corners below plane";
        for (uint32_t i = 0; i < m.point_count; ++i) {
            EXPECT_NEAR(m.points[i].penetration, 0.1f, kTol);
            EXPECT_TRUE(Vec3Near(m.points[i].normal, Vec3{0.0f, 1.0f, 0.0f}, 1.0e-2f));
        }
    }
    // Order 2: plane = A, hull = B. sep dir for A (plane) = -Y (plane pushes the
    // OTHER way; the swapped-slot sign test).
    {
        ShapeProxyView g;
        g.type_a = ShapeType::Plane;
        g.type_b = ShapeType::ConvexHull;
        g.geom_b = &H;
        g.prim_a = P;
        ContactManifold m;
        std::memset(&m, 0, sizeof(m));
        ResolveNarrowphase(g.type_a, g.type_b, false)(MakePair(), g, &m);
        ASSERT_GT(m.point_count, 0u);
        for (uint32_t i = 0; i < m.point_count; ++i) {
            EXPECT_NEAR(m.points[i].penetration, 0.1f, kTol);
            EXPECT_TRUE(Vec3Near(m.points[i].normal, Vec3{0.0f, -1.0f, 0.0f}, 1.0e-2f))
                << "plane-as-A normal must be -Y (swapped sign)";
        }
    }
}

// ===========================================================================
// D1: 2-run bit-identity + N>=32 cross-replica on a SYMMETRIC tie-prone config.
// ===========================================================================
// Symmetric face-face box-box-as-hull stack: ALL clip points tie on penetration,
// so the EPA face scan + the FPS reducer are the sole determinants. A heap-based
// EPA would diverge here; the array-scan EPA must be byte-identical run-to-run.
TEST(GjkEpaConvex, Determinism_TwoRunAndCrossReplica) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const auto vb = BoxVerts(he);
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    const ConvexHullView B = MakeHull(vb, Vec3{0.0f, 0.9f, 0.0f});  // overlap Y=0.1

    // 2-run bit identity (memset both, memcmp the whole struct incl. padding).
    ContactManifold m1, m2;
    std::memset(&m1, 0, sizeof(m1));
    std::memset(&m2, 0, sizeof(m2));
    ResolveNarrowphase(ShapeType::ConvexHull, ShapeType::ConvexHull, false)(
        MakePair(), [&] { ShapeProxyView g; g.type_a = ShapeType::ConvexHull;
                          g.type_b = ShapeType::ConvexHull; g.geom_a = &A;
                          g.geom_b = &B; return g; }(), &m1);
    ResolveNarrowphase(ShapeType::ConvexHull, ShapeType::ConvexHull, false)(
        MakePair(), [&] { ShapeProxyView g; g.type_a = ShapeType::ConvexHull;
                          g.type_b = ShapeType::ConvexHull; g.geom_a = &A;
                          g.geom_b = &B; return g; }(), &m2);
    ASSERT_GT(m1.point_count, 0u);
    EXPECT_EQ(std::memcmp(&m1, &m2, sizeof(ContactManifold)), 0)
        << "2-run manifold must be byte-identical (EPA determinism)";

    // N>=32 cross-replica: 40 identical pairs in one loop -> all byte-identical.
    constexpr int kReplicas = 40;
    ContactManifold ref;
    std::memset(&ref, 0, sizeof(ref));
    {
        ShapeProxyView g; g.type_a = ShapeType::ConvexHull;
        g.type_b = ShapeType::ConvexHull; g.geom_a = &A; g.geom_b = &B;
        ResolveNarrowphase(g.type_a, g.type_b, false)(MakePair(), g, &ref);
    }
    for (int r = 0; r < kReplicas; ++r) {
        ContactManifold mr;
        std::memset(&mr, 0, sizeof(mr));
        ShapeProxyView g; g.type_a = ShapeType::ConvexHull;
        g.type_b = ShapeType::ConvexHull; g.geom_a = &A; g.geom_b = &B;
        ResolveNarrowphase(g.type_a, g.type_b, false)(MakePair(), g, &mr);
        EXPECT_EQ(std::memcmp(&ref, &mr, sizeof(ContactManifold)), 0)
            << "replica " << r << " diverged (EPA determinism gate)";
    }
}

// ===========================================================================
// REGRESSION (C3c adversarial review): ORIGIN-ON-BOUNDARY EPA false-negative.
// ===========================================================================
// The MOST COMMON sim contact (flat-faced resting/stacking, origin exactly on the
// CSO boundary of the GJK seed tetra) used to return an EMPTY manifold from a
// DETERMINISTIC EPA churn (the seed tetra did not strictly enclose the origin ->
// dist==0 seed faces with inward normals -> non-monotone polytope -> face overflow
// -> stale depth 0 -> dropped). These tests pin the EXACT failing config + a SWEEP
// that exposes the data-dependence a single hand-picked config hides.

// (1) THE EXACT REPRODUCER: two unit boxes-as-hulls, A at origin, B at z=0.90
// (overlap 0.10 along +Z). Pre-fix: point_count==0. Post-fix: pen≈0.10, normal≈±Z.
TEST(GjkEpaConvex, OriginOnBoundary_ExactRepro_Z090) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const auto vb = BoxVerts(he);
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    const ConvexHullView B = MakeHull(vb, Vec3{0.0f, 0.0f, 0.90f});  // overlap Z=0.10
    const ContactManifold m = RouteHullHull(A, B);
    ASSERT_GT(m.point_count, 0u)
        << "origin-on-boundary face-flush contact must NOT be an empty manifold";
    for (uint32_t i = 0; i < m.point_count; ++i) {
        EXPECT_NEAR(m.points[i].penetration, 0.10f, kTol);
        // sep dir for A = -Z (A must move -Z to escape B above it on +Z).
        EXPECT_TRUE(Vec3Near(m.points[i].normal, Vec3{0.0f, 0.0f, -1.0f}, 1.0e-2f))
            << "n=(" << m.points[i].normal.x << "," << m.points[i].normal.y << ","
            << m.points[i].normal.z << ")";
    }
    EXPECT_EQ(m.point_count, 4u) << "axis-aligned face-face -> 4-pt patch";
}

// (2) THE SWEEP (the real anti-regression): boxes-as-hulls overlapping by
// {0.05,0.10,0.15,0.20,0.30} along EACH of +X,+Y,+Z, PLUS a diagonal offset.
// Each -> correct depth (the brute-force support-difference oracle along the
// emitted normal) + a unit normal aligned with the contact axis. A single config
// hides the GJK-seed data-dependence; the whole grid is what catches it.
TEST(GjkEpaConvex, OriginOnBoundary_OverlapSweep_AllAxes) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const auto vb = BoxVerts(he);
    const float overlaps[5] = {0.05f, 0.10f, 0.15f, 0.20f, 0.30f};
    // Axis-aligned: the +X,+Y,+Z separations (center offset = 1.0 - overlap).
    const Vec3 axis_dir[3] = {Vec3{1, 0, 0}, Vec3{0, 1, 0}, Vec3{0, 0, 1}};
    const Vec3 expect_n[3] = {Vec3{-1, 0, 0}, Vec3{0, -1, 0}, Vec3{0, 0, -1}};
    int ok = 0;
    for (int ax = 0; ax < 3; ++ax) {
        for (int oi = 0; oi < 5; ++oi) {
            const float ov = overlaps[oi];
            const float sep = 1.0f - ov;                  // center-to-center
            const Vec3 pos = axis_dir[ax] * sep;
            const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
            const ConvexHullView B = MakeHull(vb, pos);
            const ContactManifold m = RouteHullHull(A, B);
            ASSERT_GT(m.point_count, 0u)
                << "axis " << ax << " overlap " << ov << " -> EMPTY (regression!)";
            // Depth oracle along the emitted normal (the general support-difference).
            const Vec3 n = m.points[0].normal;
            const float d_oracle = OverlapAlong(va, Vec3{0, 0, 0}, vb, pos, n);
            EXPECT_NEAR(m.points[0].penetration, d_oracle, kTol)
                << "axis " << ax << " overlap " << ov << " depth != oracle";
            EXPECT_NEAR(m.points[0].penetration, ov, kTol)
                << "axis " << ax << " overlap " << ov << " wrong depth";
            EXPECT_TRUE(Vec3Near(n, expect_n[ax], 2.0e-2f))
                << "axis " << ax << " overlap " << ov << " wrong normal";
            ++ok;
        }
    }
    // (2b) A DIAGONAL offset (the on-EDGE/on-vertex degeneracy is orientation-
    // dependent; an axis-only sweep can miss the diagonal variant). Two unit boxes
    // overlapping along the body diagonal: each axis overlaps 0.20 (sep 0.80 on all
    // three axes). The MTV is still an axis face (depth 0.20), but the GJK seed
    // straddles the origin differently than the pure-axis cases.
    {
        const Vec3 pos{0.80f, 0.80f, 0.80f};
        const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
        const ConvexHullView B = MakeHull(vb, pos);
        const ContactManifold m = RouteHullHull(A, B);
        ASSERT_GT(m.point_count, 0u) << "diagonal offset -> EMPTY (regression!)";
        const Vec3 n = m.points[0].normal;
        const float d_oracle = OverlapAlong(va, Vec3{0, 0, 0}, vb, pos, n);
        EXPECT_NEAR(m.points[0].penetration, d_oracle, kTol)
            << "diagonal depth != support-difference oracle along the normal";
        // The MTV depth equals the min per-axis overlap (0.20 on each axis here).
        EXPECT_NEAR(m.points[0].penetration, 0.20f, kTol);
        // No sampled axis separates tighter than the emitted depth (n is the MTV).
        const Vec3 probe[6] = {Vec3{1,0,0}, Vec3{0,1,0}, Vec3{0,0,1},
                               Vec3{-1,0,0}, Vec3{0,-1,0}, Vec3{0,0,-1}};
        for (const Vec3& u : probe) {
            const float ovx = OverlapAlong(va, Vec3{0, 0, 0}, vb, pos, u);
            EXPECT_GE(ovx, m.points[0].penetration - kTol)
                << "diagonal: an axis separates tighter than the EPA normal";
        }
        ++ok;
    }
    EXPECT_EQ(ok, 16) << "expected 15 axis configs + 1 diagonal";
}

// (3) FLUSH-FACE convex x convex: two NON-box convex hulls (hexagonal prisms)
// with a fully coplanar contact face overlapping slightly along +Z. This is the
// general (not-a-box) origin-on-boundary class -> must be non-empty with a +Z/-Z
// normal and the right depth. Proves the fix is geometry-general, not box-special.
TEST(GjkEpaConvex, OriginOnBoundary_FlushFaceConvexConvex) {
    // A hexagonal prism (6 side verts top + 6 bottom), half-height 0.5 along Z,
    // circumradius 0.5 in XY. Two coplanar hex caps -> a flush +Z/-Z contact.
    std::vector<float> hex;
    for (int z = -1; z <= 1; z += 2) {
        const float hz = 0.5f * static_cast<float>(z);
        for (int k = 0; k < 6; ++k) {
            const float ang = static_cast<float>(k) * (3.14159265f / 3.0f);
            hex.push_back(0.5f * std::cos(ang));
            hex.push_back(0.5f * std::sin(ang));
            hex.push_back(hz);
        }
    }
    // A at origin (top cap z=0.5), B at z=0.90 (bottom cap z=0.40) -> overlap 0.10
    // along Z, the two hex caps flush/coplanar in the overlap band.
    const ConvexHullView A = MakeHull(hex, Vec3{0.0f, 0.0f, 0.0f});
    const ConvexHullView B = MakeHull(hex, Vec3{0.0f, 0.0f, 0.90f});
    const ContactManifold m = RouteHullHull(A, B);
    ASSERT_GT(m.point_count, 0u)
        << "flush-face convex x convex must NOT be an empty manifold";
    for (uint32_t i = 0; i < m.point_count; ++i) {
        EXPECT_NEAR(m.points[i].penetration, 0.10f, kTol);
        EXPECT_TRUE(Vec3Near(m.points[i].normal, Vec3{0.0f, 0.0f, -1.0f}, 2.0e-2f))
            << "n=(" << m.points[i].normal.x << "," << m.points[i].normal.y << ","
            << m.points[i].normal.z << ")";
    }
}

// (4) D1 on the FAILING class: 2-run byte-identity + N>=32 cross-replica on the
// z=0.90 origin-on-boundary config (the exact bug class, which now exercises the
// new enclosure pre-loop + sliver-skip path -- code the OLD determinism gate at
// y=0.9 also exercises, but pin it on the SPECIFIC failing seed too).
TEST(GjkEpaConvex, OriginOnBoundary_DeterminismOnFailingClass) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const auto vb = BoxVerts(he);
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    const ConvexHullView B = MakeHull(vb, Vec3{0.0f, 0.0f, 0.90f});  // the repro seed

    ContactManifold m1, m2;
    std::memset(&m1, 0, sizeof(m1));
    std::memset(&m2, 0, sizeof(m2));
    m1 = RouteHullHull(A, B);
    m2 = RouteHullHull(A, B);
    ASSERT_GT(m1.point_count, 0u);
    EXPECT_EQ(std::memcmp(&m1, &m2, sizeof(ContactManifold)), 0)
        << "2-run on the failing origin-on-boundary class must be byte-identical";

    constexpr int kReplicas = 40;  // N >= 32 cross-replica
    ContactManifold ref = RouteHullHull(A, B);
    for (int r = 0; r < kReplicas; ++r) {
        const ContactManifold mr = RouteHullHull(A, B);
        EXPECT_EQ(std::memcmp(&ref, &mr, sizeof(ContactManifold)), 0)
            << "replica " << r << " diverged (origin-on-boundary D1 gate)";
    }
}

// (5) GENUINE TOUCH (depth 0) MUST NOT become a spurious deep contact. Two unit
// boxes EXACTLY face-flush (B at z=1.0, overlap 0): the origin sits ON the CSO
// surface. The robust-EPA fix must report NO contact here -- NOT a far +/-axis
// face of the (flat, ~2-deep-on-the-other-side) box CSO. A tiny POSITIVE overlap
// just inside the boundary must still produce a real contact (the fix discriminates
// depth-0 touching from a real overlap, it does not blanket-drop near-zero depth).
TEST(GjkEpaConvex, GenuineTouchAndNearBoundary_NoSpuriousContact) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const auto vb = BoxVerts(he);
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});

    // Exact touch (overlap 0) -> NO contact (depth 0).
    {
        const ConvexHullView B = MakeHull(vb, Vec3{0.0f, 0.0f, 1.0f});
        const ContactManifold m = RouteHullHull(A, B);
        EXPECT_EQ(m.point_count, 0u)
            << "exact face-flush touch (overlap 0) must be NO contact, "
               "not a spurious deep manifold from the flat CSO's far side";
    }
    // Just SEPARATED (gap 0.001 and 0.05) -> NO contact.
    for (float gap : {0.001f, 0.05f}) {
        const ConvexHullView B = MakeHull(vb, Vec3{0.0f, 0.0f, 1.0f + gap});
        const ContactManifold m = RouteHullHull(A, B);
        EXPECT_EQ(m.point_count, 0u) << "separated (gap=" << gap << ") -> no contact";
    }
    // Just OVERLAPPING (overlap 0.001) -> a real small contact, normal -Z.
    {
        const ConvexHullView B = MakeHull(vb, Vec3{0.0f, 0.0f, 0.999f});
        const ContactManifold m = RouteHullHull(A, B);
        ASSERT_GT(m.point_count, 0u) << "tiny real overlap must still be a contact";
        EXPECT_NEAR(m.points[0].penetration, 0.001f, 1.0e-3f);
        EXPECT_TRUE(Vec3Near(m.points[0].normal, Vec3{0.0f, 0.0f, -1.0f}, 2.0e-2f));
    }
}

// (6) ROTATED / OBLIQUE origin-on-boundary: the bug is GJK-SEED-ORIENTATION-
// dependent, and (1)-(5) are all axis-aligned. A box B ROTATED off-axis and
// overlapping A produces a DIFFERENT seed straddle of the origin. The centroid-
// oriented fix is geometry-general -> each must give the right depth (the
// brute-force support-difference oracle along the emitted normal). Locks
// orientation coverage so an axis-only sweep can't hide an oblique regression.
TEST(GjkEpaConvex, OriginOnBoundary_RotatedOblique) {
    const Vec3 he{0.5f, 0.5f, 0.5f};
    const auto va = BoxVerts(he);
    const auto vb = BoxVerts(he);
    const ConvexHullView A = MakeHull(va, Vec3{0.0f, 0.0f, 0.0f});
    // (axis, degrees, center-z) chosen to give a clear positive overlap.
    struct Cfg { Vec3 axis; float deg; float z; };
    const Cfg cfgs[5] = {
        {Vec3{1, 1, 0}, 30.0f, 1.00f}, {Vec3{0, 1, 1}, 30.0f, 0.95f},
        {Vec3{1, 1, 1}, 20.0f, 1.05f}, {Vec3{1, 2, 0}, 35.0f, 0.90f},
        {Vec3{2, 1, 1}, 15.0f, 1.00f},
    };
    for (const Cfg& c : cfgs) {
        const float r = c.deg * 3.14159265f / 180.0f;
        const float s = std::sin(r * 0.5f);
        const float l = std::sqrt(c.axis.x * c.axis.x + c.axis.y * c.axis.y +
                                  c.axis.z * c.axis.z);
        const Quat rot{c.axis.x / l * s, c.axis.y / l * s, c.axis.z / l * s,
                       std::cos(r * 0.5f)};
        const Vec3 pos{0.0f, 0.0f, c.z};
        const ConvexHullView B = MakeHull(vb, pos, rot);
        const ContactManifold m = RouteHullHull(A, B);
        ASSERT_GT(m.point_count, 0u)
            << "rotated " << c.deg << "deg @z=" << c.z << " -> EMPTY (regression!)";
        // Depth oracle along the emitted normal, using the WORLD-TRANSFORMED verts
        // of both hulls (B is rotated, so support must use the view's frame).
        const Vec3 n = m.points[0].normal;
        float ma = -3.4e38f, mb = -3.4e38f;
        for (uint32_t i = 0; i < A.vcount; ++i) {
            const float d = A.Vertex(i).Dot(Vec3{-n.x, -n.y, -n.z});
            if (d > ma) ma = d;
        }
        for (uint32_t i = 0; i < B.vcount; ++i) {
            const float d = B.Vertex(i).Dot(n);
            if (d > mb) mb = d;
        }
        const float oracle = ma + mb;
        EXPECT_NEAR(m.points[0].penetration, oracle, kTol)
            << "rotated " << c.deg << "deg @z=" << c.z
            << " depth != support-difference oracle along the normal";
        EXPECT_GT(m.points[0].penetration, 0.0f);
    }
}
