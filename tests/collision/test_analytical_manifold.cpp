// ---------------------------------------------------------------------------
// v0.8 C3b -- analytical primitive narrowphase (multi-point/face-face) tests.
// ---------------------------------------------------------------------------
// Drives the REAL Analytical-tier handlers registered in the C3a dispatch table
// (narrowphase_dispatch.hpp + analytical_manifold.hpp):
//   - box-on-plane resting -> exactly 4 coplanar contact points at the box's
//     bottom corners; penetration correct; normal = plane normal for side A.
//   - box-box face-face -> a stable 4-point patch; point SET (order-insensitive)
//     vs a hand oracle.
//   - sphere x {sphere,box,plane} -> 1 point, analytic position/normal/penetration.
//   - capsule x plane (2 endpoints) / capsule x sphere (1 pt).
//   - D1: run a handler TWICE on identical inputs (padding zeroed) -> byte-identical
//     manifold (memcmp). A SYMMETRIC box-box stack (penetration ties) -> stable
//     4-pt reduction matching the oracle SET (the determinism-trap gate).
//   - both (a,b) orderings: normal sign stays "separation dir for side A".
//   - routing sanity: ResolveNarrowphase(Box,Plane,false) is a real handler.
// ---------------------------------------------------------------------------

#include "collision/narrowphase_dispatch.hpp"
#include "collision/analytical_manifold.hpp"
#include "constraint/collidable.hpp"
#include "constraint/contact_manifold.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

using nuka::collision::CandidatePair;
using nuka::collision::NarrowphaseStubAnalytical;
using nuka::collision::ResolveNarrowphase;
using nuka::collision::ShapeProxyView;
using nuka::collision::amf::PrimParams;
using nuka::collision::amf::BuildPrimFrame;
using nuka::constraint::CollidableRef;
using nuka::constraint::CollidableType;
using nuka::constraint::ContactManifold;
using nuka::constraint::ReactionProviderKind;
using nuka::math::Quat;
using nuka::math::Transform;
using nuka::math::Vec3;
using nuka::scene::ShapeType;

namespace {

constexpr float kTol = 1.0e-4f;

PrimParams MakeBox(Vec3 he, Vec3 pos, Quat rot = Quat::Identity()) {
    PrimParams p;
    p.half_extents = he;
    p.frame = BuildPrimFrame(Transform{pos, rot});
    return p;
}
PrimParams MakeSphere(float r, Vec3 pos) {
    PrimParams p;
    p.radius = r;
    p.frame = BuildPrimFrame(Transform{pos, Quat::Identity()});
    return p;
}
PrimParams MakePlane(Vec3 pos, Quat rot = Quat::Identity()) {
    PrimParams p;
    p.frame = BuildPrimFrame(Transform{pos, rot});
    return p;
}
PrimParams MakeCapsule(float r, float hh, Vec3 pos, Quat rot = Quat::Identity()) {
    PrimParams p;
    p.radius = r;
    p.half_height = hh;
    p.frame = BuildPrimFrame(Transform{pos, rot});
    return p;
}

// Build a ShapeProxyView + CandidatePair and route through the resolved handler.
// `pa`/`pb` are the prim params for side A/B respectively.
ContactManifold Route(ShapeType ta, ShapeType tb, const PrimParams& pa,
                      const PrimParams& pb, bool has_sdf = false) {
    CandidatePair pair;
    pair.a = CollidableRef{CollidableType::RigidBody,
                           ReactionProviderKind::RigidInvMass, 1u};
    pair.b = CollidableRef{CollidableType::RigidBody,
                           ReactionProviderKind::RigidInvMass, 2u};
    ShapeProxyView g;
    g.type_a = ta;
    g.type_b = tb;
    g.prim_a = pa;
    g.prim_b = pb;
    ContactManifold out;
    ResolveNarrowphase(ta, tb, has_sdf)(pair, g, &out);
    return out;
}

bool Vec3Near(Vec3 a, Vec3 b, float tol = kTol) {
    return std::fabs(a.x - b.x) < tol && std::fabs(a.y - b.y) < tol &&
           std::fabs(a.z - b.z) < tol;
}

// Order-insensitive set compare: for each expected position, find a manifold
// point within tol. point_count must match the expected set size.
void ExpectPositionSet(const ContactManifold& m, std::vector<Vec3> expected,
                       float tol = kTol) {
    ASSERT_EQ(m.point_count, expected.size());
    std::vector<bool> used(m.point_count, false);
    for (const Vec3& e : expected) {
        bool found = false;
        for (uint32_t i = 0; i < m.point_count; ++i) {
            if (!used[i] && Vec3Near(m.points[i].position, e, tol)) {
                used[i] = true;
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "expected contact point not present: (" << e.x
                           << "," << e.y << "," << e.z << ")";
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Routing sanity.
// ---------------------------------------------------------------------------
TEST(AnalyticalManifold, BoxPlaneResolvesRealHandler) {
    EXPECT_NE(ResolveNarrowphase(ShapeType::Box, ShapeType::Plane, false),
              &NarrowphaseStubAnalytical);
    EXPECT_NE(ResolveNarrowphase(ShapeType::Plane, ShapeType::Box, false),
              &NarrowphaseStubAnalytical);
    EXPECT_NE(ResolveNarrowphase(ShapeType::Box, ShapeType::Box, false),
              &NarrowphaseStubAnalytical);
}

// ---------------------------------------------------------------------------
// Sphere x sphere -> 1 point.
// ---------------------------------------------------------------------------
TEST(AnalyticalManifold, SphereSphereSinglePoint) {
    // A at x=0, B at x=1.5, radii 1 each -> overlap 0.5 along +x.
    PrimParams a = MakeSphere(1.0f, {0.0f, 0.0f, 0.0f});
    PrimParams b = MakeSphere(1.0f, {1.5f, 0.0f, 0.0f});
    ContactManifold m = Route(ShapeType::Sphere, ShapeType::Sphere, a, b);
    ASSERT_EQ(m.point_count, 1u);
    EXPECT_NEAR(m.points[0].penetration, 0.5f, kTol);
    // normal = separation dir for A == -x (A must move toward -x away from B).
    EXPECT_TRUE(Vec3Near(m.points[0].normal, {-1.0f, 0.0f, 0.0f}));
    // contact on B's surface toward A: B.center + n*rB = (1.5-1, 0,0)=(0.5,0,0).
    EXPECT_TRUE(Vec3Near(m.points[0].position, {0.5f, 0.0f, 0.0f}));
}

// ---------------------------------------------------------------------------
// Sphere x box -> 1 point (OBB clamp). Box at origin he=1; sphere above face.
// ---------------------------------------------------------------------------
TEST(AnalyticalManifold, SphereBoxSinglePoint) {
    PrimParams sph = MakeSphere(0.6f, {0.0f, 1.5f, 0.0f});
    PrimParams box = MakeBox({1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f});
    ContactManifold m = Route(ShapeType::Sphere, ShapeType::Box, sph, box);
    ASSERT_EQ(m.point_count, 1u);
    // closest box point = (0,1,0); dist=0.5; pen = 0.6-0.5 = 0.1.
    EXPECT_NEAR(m.points[0].penetration, 0.1f, kTol);
    EXPECT_TRUE(Vec3Near(m.points[0].normal, {0.0f, 1.0f, 0.0f}));  // sep dir for sphere(A)
    EXPECT_TRUE(Vec3Near(m.points[0].position, {0.0f, 1.0f, 0.0f}));
}

// Swapped order: box is A, sphere is B -> normal flips to point box away (-y).
TEST(AnalyticalManifold, BoxSphereNormalFlips) {
    PrimParams box = MakeBox({1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f});
    PrimParams sph = MakeSphere(0.6f, {0.0f, 1.5f, 0.0f});
    ContactManifold m = Route(ShapeType::Box, ShapeType::Sphere, box, sph);
    ASSERT_EQ(m.point_count, 1u);
    EXPECT_NEAR(m.points[0].penetration, 0.1f, kTol);
    // separation dir for A(box) is DOWN (box pushed away from sphere above).
    EXPECT_TRUE(Vec3Near(m.points[0].normal, {0.0f, -1.0f, 0.0f}));
}

// ---------------------------------------------------------------------------
// Sphere x plane -> 1 point.
// ---------------------------------------------------------------------------
TEST(AnalyticalManifold, SpherePlaneSinglePoint) {
    PrimParams sph = MakeSphere(1.0f, {0.0f, 0.8f, 0.0f});  // bottom at y=-0.2
    PrimParams plane = MakePlane({0.0f, 0.0f, 0.0f});       // y=0 plane, normal +y
    ContactManifold m = Route(ShapeType::Sphere, ShapeType::Plane, sph, plane);
    ASSERT_EQ(m.point_count, 1u);
    EXPECT_NEAR(m.points[0].penetration, 0.2f, kTol);
    EXPECT_TRUE(Vec3Near(m.points[0].normal, {0.0f, 1.0f, 0.0f}));
    EXPECT_TRUE(Vec3Near(m.points[0].position, {0.0f, -0.2f, 0.0f}));
}

// ---------------------------------------------------------------------------
// Box on plane resting -> exactly 4 coplanar bottom corners.
// ---------------------------------------------------------------------------
TEST(AnalyticalManifold, BoxOnPlaneFourCorners) {
    // Box he=1, center at y=0.9 -> bottom face at y=-0.1 (penetration 0.1).
    PrimParams box = MakeBox({1.0f, 1.0f, 1.0f}, {0.0f, 0.9f, 0.0f});
    PrimParams plane = MakePlane({0.0f, 0.0f, 0.0f});
    ContactManifold m = Route(ShapeType::Box, ShapeType::Plane, box, plane);
    ASSERT_EQ(m.point_count, 4u);
    for (uint32_t i = 0; i < 4u; ++i) {
        EXPECT_NEAR(m.points[i].penetration, 0.1f, kTol);
        EXPECT_TRUE(Vec3Near(m.points[i].normal, {0.0f, 1.0f, 0.0f}));
        EXPECT_NEAR(m.points[i].position.y, -0.1f, kTol);  // coplanar at box bottom
    }
    // The 4 bottom corners (x,z in {-1,1}) at y=-0.1.
    ExpectPositionSet(m, {{-1.0f, -0.1f, -1.0f}, {1.0f, -0.1f, -1.0f},
                          {-1.0f, -0.1f, 1.0f}, {1.0f, -0.1f, 1.0f}});
}

// Swapped order: plane is A, box is B -> normal flips to -y (separation for plane).
TEST(AnalyticalManifold, PlaneBoxNormalFlips) {
    PrimParams plane = MakePlane({0.0f, 0.0f, 0.0f});
    PrimParams box = MakeBox({1.0f, 1.0f, 1.0f}, {0.0f, 0.9f, 0.0f});
    ContactManifold m = Route(ShapeType::Plane, ShapeType::Box, plane, box);
    ASSERT_EQ(m.point_count, 4u);
    for (uint32_t i = 0; i < 4u; ++i) {
        EXPECT_TRUE(Vec3Near(m.points[i].normal, {0.0f, -1.0f, 0.0f}));
    }
}

// ---------------------------------------------------------------------------
// Box-box face-face -> stable 4-point patch vs hand oracle (order-insensitive).
// ---------------------------------------------------------------------------
TEST(AnalyticalManifold, BoxBoxFaceFacePatch) {
    // Box A he=1 centered at origin. Box B he=1 centered at y=1.9 -> overlap 0.1
    // along y. A's top face is the reference; B's bottom face is incident.
    PrimParams A = MakeBox({1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f});
    PrimParams B = MakeBox({1.0f, 1.0f, 1.0f}, {0.0f, 1.9f, 0.0f});
    ContactManifold m = Route(ShapeType::Box, ShapeType::Box, A, B);
    ASSERT_EQ(m.point_count, 4u);
    for (uint32_t i = 0; i < 4u; ++i) {
        EXPECT_NEAR(m.points[i].penetration, 0.1f, kTol);
        // separation dir for A == DOWN (A pushed away from B above).
        EXPECT_TRUE(Vec3Near(m.points[i].normal, {0.0f, -1.0f, 0.0f}));
    }
    // Incident-face corners (B bottom at y=0.9), midpointed with their projection
    // onto A's top face (y=1.0): contact y = (0.9+1.0)/2 = 0.95; x,z in {-1,1}.
    ExpectPositionSet(m, {{-1.0f, 0.95f, -1.0f}, {1.0f, 0.95f, -1.0f},
                          {-1.0f, 0.95f, 1.0f}, {1.0f, 0.95f, 1.0f}});
}

// Box-box where B is offset so the overlap region is a sub-rectangle (clip trims).
TEST(AnalyticalManifold, BoxBoxOffsetClipsToOverlapRect) {
    PrimParams A = MakeBox({1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f});
    // B shifted +0.5 in x: overlap in x is [-0.5, 1.0] on A's top face.
    PrimParams B = MakeBox({1.0f, 1.0f, 1.0f}, {0.5f, 1.9f, 0.0f});
    ContactManifold m = Route(ShapeType::Box, ShapeType::Box, A, B);
    ASSERT_EQ(m.point_count, 4u);
    // All contact x within A's [-1,1] AND B's incident-face x [-0.5,1.5] -> [-0.5,1].
    for (uint32_t i = 0; i < 4u; ++i) {
        EXPECT_GE(m.points[i].position.x, -0.5f - kTol);
        EXPECT_LE(m.points[i].position.x, 1.0f + kTol);
        EXPECT_NEAR(m.points[i].penetration, 0.1f, kTol);
    }
}

// ---------------------------------------------------------------------------
// Capsule x plane -> 2 endpoints (axis-aligned along Y? no -- lay it along X).
// ---------------------------------------------------------------------------
TEST(AnalyticalManifold, CapsulePlaneTwoEndpoints) {
    // Capsule radius 0.5, half_height 1, laid along X (rotate local Y -> world X)
    // centered at y=0.4 -> the cylinder surface bottom at y = 0.4-0.5 = -0.1.
    const Quat lay = Quat::FromAxisAngle(Vec3::UnitZ(), -static_cast<float>(M_PI) / 2.0f);
    PrimParams cap = MakeCapsule(0.5f, 1.0f, {0.0f, 0.4f, 0.0f}, lay);
    PrimParams plane = MakePlane({0.0f, 0.0f, 0.0f});
    ContactManifold m = Route(ShapeType::Capsule, ShapeType::Plane, cap, plane);
    ASSERT_EQ(m.point_count, 2u);
    for (uint32_t i = 0; i < 2u; ++i) {
        EXPECT_NEAR(m.points[i].penetration, 0.1f, kTol);
        EXPECT_TRUE(Vec3Near(m.points[i].normal, {0.0f, 1.0f, 0.0f}));
        EXPECT_NEAR(m.points[i].position.y, -0.1f, kTol);
    }
    // endpoints at x = +/-1 (half_height), surface point y = -0.1.
    ExpectPositionSet(m, {{1.0f, -0.1f, 0.0f}, {-1.0f, -0.1f, 0.0f}});
}

// ---------------------------------------------------------------------------
// Capsule x sphere -> 1 point.
// ---------------------------------------------------------------------------
TEST(AnalyticalManifold, CapsuleSphereSinglePoint) {
    // Capsule along Y (default), radius 0.5 hh 1 at origin. Sphere r 0.5 at
    // x=0.8 (near the cylinder mid). dist from axis = 0.8; pen = (0.5+0.5)-0.8=0.2.
    PrimParams cap = MakeCapsule(0.5f, 1.0f, {0.0f, 0.0f, 0.0f});
    PrimParams sph = MakeSphere(0.5f, {0.8f, 0.0f, 0.0f});
    ContactManifold m = Route(ShapeType::Capsule, ShapeType::Sphere, cap, sph);
    ASSERT_EQ(m.point_count, 1u);
    EXPECT_NEAR(m.points[0].penetration, 0.2f, kTol);
    // separation dir for capsule(A) == -x: the sphere sits at +x, so the capsule
    // must move toward -x to separate (witness = closest axis pt - sphere center).
    EXPECT_TRUE(Vec3Near(m.points[0].normal, {-1.0f, 0.0f, 0.0f}));
    // contact point on capsule surface toward the sphere: (0,0,0) + (1,0,0)*r.
    EXPECT_TRUE(Vec3Near(m.points[0].position, {0.5f, 0.0f, 0.0f}));
}

// ---------------------------------------------------------------------------
// D1: two-run byte-identical manifold (padding zeroed).
// ---------------------------------------------------------------------------
namespace {
// Run handler into a ZEROED manifold buffer (so padding bytes are deterministic).
void RouteIntoZeroed(ShapeType ta, ShapeType tb, const PrimParams& pa,
                     const PrimParams& pb, ContactManifold* out) {
    std::memset(out, 0, sizeof(ContactManifold));
    CandidatePair pair;
    pair.a = CollidableRef{CollidableType::RigidBody,
                           ReactionProviderKind::RigidInvMass, 1u};
    pair.b = CollidableRef{CollidableType::RigidBody,
                           ReactionProviderKind::RigidInvMass, 2u};
    ShapeProxyView g;
    g.type_a = ta; g.type_b = tb; g.prim_a = pa; g.prim_b = pb;
    ResolveNarrowphase(ta, tb, false)(pair, g, out);
}
}  // namespace

TEST(AnalyticalManifold, BoxBoxTwoRunByteIdentical) {
    PrimParams A = MakeBox({1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f});
    PrimParams B = MakeBox({1.0f, 1.0f, 1.0f}, {0.0f, 1.9f, 0.0f});
    ContactManifold m1, m2;
    RouteIntoZeroed(ShapeType::Box, ShapeType::Box, A, B, &m1);
    RouteIntoZeroed(ShapeType::Box, ShapeType::Box, A, B, &m2);
    EXPECT_EQ(std::memcmp(&m1, &m2, sizeof(ContactManifold)), 0)
        << "box-box manifold must be byte-identical across two runs (D1)";
}

TEST(AnalyticalManifold, BoxPlaneTwoRunByteIdentical) {
    PrimParams box = MakeBox({1.0f, 1.0f, 1.0f}, {0.0f, 0.9f, 0.0f});
    PrimParams plane = MakePlane({0.0f, 0.0f, 0.0f});
    ContactManifold m1, m2;
    RouteIntoZeroed(ShapeType::Box, ShapeType::Plane, box, plane, &m1);
    RouteIntoZeroed(ShapeType::Box, ShapeType::Plane, box, plane, &m2);
    EXPECT_EQ(std::memcmp(&m1, &m2, sizeof(ContactManifold)), 0);
}

// ---------------------------------------------------------------------------
// D1 determinism-trap: SYMMETRIC box-box stack (every clip point ties on
// penetration) -> the deepest-4 reduction MUST be stable + match the oracle SET,
// AND the slot order is fixed (so the two-run memcmp above already proved
// stability). Here we assert the SET is the 4 corners (not a degenerate cluster).
// ---------------------------------------------------------------------------
TEST(AnalyticalManifold, BoxBoxSymmetricStackStableFourCorners) {
    // Perfectly concentric (in x,z) symmetric stack -> all 4 corners tie on
    // penetration (0.1). The feature-id key (not penetration) must order them.
    PrimParams A = MakeBox({1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f});
    PrimParams B = MakeBox({1.0f, 1.0f, 1.0f}, {0.0f, 1.9f, 0.0f});
    ContactManifold m = Route(ShapeType::Box, ShapeType::Box, A, B);
    ASSERT_EQ(m.point_count, 4u);
    // All 4 distinct corners present (no cluster) at the contact plane y=0.95.
    ExpectPositionSet(m, {{-1.0f, 0.95f, -1.0f}, {1.0f, 0.95f, -1.0f},
                          {-1.0f, 0.95f, 1.0f}, {1.0f, 0.95f, 1.0f}});
    // stable_key per point is the feature id -> 4 DISTINCT keys (warm-start ids).
    std::vector<uint64_t> keys;
    for (uint32_t i = 0; i < 4u; ++i) keys.push_back(m.points[i].stable_key);
    std::sort(keys.begin(), keys.end());
    EXPECT_TRUE(std::unique(keys.begin(), keys.end()) == keys.end())
        << "symmetric box-box must yield 4 DISTINCT stable_keys (no cluster)";
    // Emitted slot order is ascending feature id (deterministic).
    for (uint32_t i = 1; i < 4u; ++i) {
        EXPECT_LT(m.points[i - 1].stable_key, m.points[i].stable_key);
    }
}

// ---------------------------------------------------------------------------
// D1 determinism-trap (the >4-clip-point reduction): B rotated 45 deg about the
// stacking axis -> B's bottom face is a diamond over A's square -> the clip
// produces an OCTAGON (8 candidates), all at EQUAL penetration. The deepest-4
// reduction's tie-break is now LOAD-BEARING (it decides WHICH 4 survive). Assert
// 4 points, equal pen, normal, two-run byte-identity, AND 4 DISTINCT stable_keys
// spread symmetrically (not a degenerate clustered pair).
// ---------------------------------------------------------------------------
TEST(AnalyticalManifold, BoxBoxRotated45OctagonReductionStable) {
    PrimParams A = MakeBox({1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f});
    const Quat spin = Quat::FromAxisAngle(Vec3::UnitY(),
                                          static_cast<float>(M_PI) / 4.0f);
    PrimParams B = MakeBox({1.0f, 1.0f, 1.0f}, {0.0f, 1.9f, 0.0f}, spin);
    ContactManifold m = Route(ShapeType::Box, ShapeType::Box, A, B);
    ASSERT_EQ(m.point_count, 4u);
    for (uint32_t i = 0; i < 4u; ++i) {
        EXPECT_NEAR(m.points[i].penetration, 0.1f, kTol);
        EXPECT_TRUE(Vec3Near(m.points[i].normal, {0.0f, -1.0f, 0.0f}));
    }
    // 4 DISTINCT stable_keys (no clustered duplicates from the reduction).
    std::vector<uint64_t> keys;
    for (uint32_t i = 0; i < 4u; ++i) keys.push_back(m.points[i].stable_key);
    std::sort(keys.begin(), keys.end());
    EXPECT_TRUE(std::unique(keys.begin(), keys.end()) == keys.end())
        << "rotated box-box octagon reduction must yield 4 DISTINCT keys";
    // The 4 kept points must SPREAD around the centroid (not cluster on one edge):
    // the bounding box of the kept x/z must be reasonably wide (a clustered pair
    // would collapse one dimension). Centroid is (0,*,0); require points on both
    // sides in BOTH x and z.
    bool px = false, nx = false, pz = false, nz = false;
    for (uint32_t i = 0; i < 4u; ++i) {
        if (m.points[i].position.x > 0.2f) px = true;
        if (m.points[i].position.x < -0.2f) nx = true;
        if (m.points[i].position.z > 0.2f) pz = true;
        if (m.points[i].position.z < -0.2f) nz = true;
    }
    EXPECT_TRUE(px && nx && pz && nz)
        << "kept 4 points must spread on both sides of the centroid in x AND z";
    // Two-run byte identity (the reduction tie-break must be deterministic).
    ContactManifold m1, m2;
    RouteIntoZeroed(ShapeType::Box, ShapeType::Box, A, B, &m1);
    RouteIntoZeroed(ShapeType::Box, ShapeType::Box, A, B, &m2);
    EXPECT_EQ(std::memcmp(&m1, &m2, sizeof(ContactManifold)), 0);
}

// ---------------------------------------------------------------------------
// Sign guard across ALL box-box configs (catches a flipped normal that would
// suck bodies together): for a penetrating pair, the separation dir for A must
// have a POSITIVE component along (A.center - B.center).
// ---------------------------------------------------------------------------
TEST(AnalyticalManifold, BoxBoxNormalPushesAApartFromB) {
    struct Cfg { Vec3 bpos; Quat brot; };
    const Quat spin45 = Quat::FromAxisAngle(Vec3::UnitY(),
                                            static_cast<float>(M_PI) / 4.0f);
    std::vector<Cfg> cfgs = {
        {{0.0f, 1.9f, 0.0f}, Quat::Identity()},   // A reference (B above)
        {{0.0f, -1.9f, 0.0f}, Quat::Identity()},  // B below -> normal must be +y
        {{1.9f, 0.0f, 0.0f}, Quat::Identity()},   // B to +x
        {{0.0f, 1.9f, 0.0f}, spin45},             // rotated stack
    };
    PrimParams A = MakeBox({1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f});
    for (const Cfg& c : cfgs) {
        PrimParams B = MakeBox({1.0f, 1.0f, 1.0f}, c.bpos, c.brot);
        ContactManifold m = Route(ShapeType::Box, ShapeType::Box, A, B);
        ASSERT_GT(m.point_count, 0u);
        const Vec3 ab = A.frame.t - B.frame.t;
        EXPECT_GT(m.points[0].normal.Dot(ab), 0.0f)
            << "normal for A must point A away from B (no attraction)";
    }
}

// ---------------------------------------------------------------------------
// Non-overlap -> empty manifold (no spurious contacts).
// ---------------------------------------------------------------------------
TEST(AnalyticalManifold, SeparatedBoxesNoContact) {
    PrimParams A = MakeBox({1.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 0.0f});
    PrimParams B = MakeBox({1.0f, 1.0f, 1.0f}, {0.0f, 3.0f, 0.0f});  // gap 1.0
    ContactManifold m = Route(ShapeType::Box, ShapeType::Box, A, B);
    EXPECT_EQ(m.point_count, 0u);
}
