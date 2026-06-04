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
