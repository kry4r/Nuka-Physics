// ---------------------------------------------------------------------------
// v0.8 C5a -- contact stream DRIVER tests (BuildContactManifolds).
// ---------------------------------------------------------------------------
// Validates the resolver-decoupled host driver that turns CandidatePairs into
// ContactManifolds (contact_stream_driver.hpp). It proves:
//   (1) box-on-plane (rigid x static): driver geometry == a DIRECT
//       ResolveNarrowphase(Box,Plane,false) call (same points/normals/pen) AND
//       manifold_key + a/b + merged friction/solimp/solref are populated.
//   (2) foot-style sphere-on-plane (articulation-link x static): a 1-point
//       manifold with the right penetration + side refs -- the C5c foot-ground
//       shape, proving the driver handles an articulation-link side.
//   (3) box-box (rigid x rigid): multi-point manifold == the direct analytical
//       box-box call.
//   (4) param merge: two sides with different friction/solref/priority ->
//       m.friction/m.solimp/point.solref match an INDEPENDENT MergeContactParams
//       call (cross-check, not echo).
//   (5) has_sdf BOTH-sides: one-side SDF -> has_sdf=false -> routes to the real
//       analytical handler (non-empty, appended); both-side SDF with a null
//       SdfProxyView seam -> NarrowphaseSdf null-guards to empty -> skipped.
//       Same geometry, opposite output -> isolates the `&&`. Plus a direct
//       ResolveNarrowphase routing assertion.
//   (6) skip: a pair whose resolver returns false for a side -> no manifold.
//   (7) D1: two BuildContactManifolds runs on the same pairs+resolver are
//       byte-identical (memcmp on value-initialised manifold vectors).
//
// The geometry MATH is validated in test_analytical_manifold (the SDF-tier forward
// accuracy gate is deferred to M11 with the kept SDF/IFT family);
// here we validate the DRIVER (resolve -> route -> invoke -> merge -> append),
// and the geometry checks are anchored to a DIRECT dispatch-table call so they
// can never silently disagree with the table.
// ---------------------------------------------------------------------------

#include "collision/candidate_pair.hpp"
#include "collision/contact_stream_driver.hpp"
#include "collision/narrowphase_dispatch.hpp"
#include "constraint/collidable.hpp"
#include "constraint/contact_manifold.hpp"
#include "math/vec3.hpp"
#include "scene/canonical_types.hpp"
#include "scene/contact_filter.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <map>
#include <vector>

using nuka::collision::BuildContactManifolds;
using nuka::collision::CandidatePair;
using nuka::collision::MakeStableKey;
using nuka::collision::NarrowphaseSdf;
using nuka::collision::ResolvedShape;
using nuka::collision::ResolveNarrowphase;
using nuka::collision::ShapeProxyView;
using nuka::collision::ShapeResolver;
using nuka::constraint::CollidableRef;
using nuka::constraint::CollidableType;
using nuka::constraint::ContactManifold;
using nuka::constraint::ReactionProviderKind;
using nuka::math::Vec3;
using nuka::scene::ShapeType;

namespace amf = nuka::collision::amf;

namespace {

// --- prim builders (identity rotation: PrimFrame defaults are the world axes) --
amf::PrimParams MakeBoxPrim(float hx, float hy, float hz, Vec3 pos) {
    amf::PrimParams p;
    p.half_extents = Vec3{hx, hy, hz};
    p.frame.t = pos;  // cx/cy/cz default to the world axes (identity rotation)
    return p;
}

amf::PrimParams MakeSpherePrim(float r, Vec3 pos) {
    amf::PrimParams p;
    p.radius = r;
    p.frame.t = pos;
    return p;
}

amf::PrimParams MakePlanePrim(Vec3 pos) {
    amf::PrimParams p;  // plane normal = local +Y == frame.cy == world +Y
    p.frame.t = pos;
    return p;
}

// Side refs used across the tests.
CollidableRef RigidRef(uint32_t h) {
    return CollidableRef{CollidableType::RigidBody,
                         ReactionProviderKind::RigidInvMass, h};
}
CollidableRef LinkRef(uint32_t h) {
    return CollidableRef{CollidableType::ArticulationLink,
                         ReactionProviderKind::ArticulationChainJ, h};
}
CollidableRef StaticRef(uint32_t h) {
    return CollidableRef{CollidableType::StaticWorld,
                         ReactionProviderKind::StaticNull, h};
}

CandidatePair MakePair(const CollidableRef& a, const CollidableRef& b) {
    CandidatePair p;
    p.a = a;
    p.b = b;
    p.stable_key = MakeStableKey(a, b);
    return p;
}

// A table-driven resolver: maps a CollidableRef.handle -> a pre-built
// ResolvedShape. This is the test's stand-in for the C5c rigid-BodyState /
// articulation-link resolvers -- it fabricates the prim + world frame directly
// (the DECOUPLING: the driver never reads a cooked table; the resolver supplies
// everything). A handle not in the map resolves to false (the skip case).
class MapResolver {
public:
    void Set(uint32_t handle, const ResolvedShape& s) { map_[handle] = s; }
    ShapeResolver Fn() {
        return [this](const CollidableRef& ref, ResolvedShape* out) -> bool {
            auto it = map_.find(ref.handle);
            if (it == map_.end()) return false;
            *out = it->second;
            return true;
        };
    }

private:
    std::map<uint32_t, ResolvedShape> map_;
};

ResolvedShape MakeBoxShape(float hx, float hy, float hz, Vec3 pos) {
    ResolvedShape s;
    s.type = ShapeType::Box;
    s.prim = MakeBoxPrim(hx, hy, hz, pos);
    return s;
}
ResolvedShape MakeSphereShape(float r, Vec3 pos) {
    ResolvedShape s;
    s.type = ShapeType::Sphere;
    s.prim = MakeSpherePrim(r, pos);
    return s;
}
ResolvedShape MakePlaneShape(Vec3 pos) {
    ResolvedShape s;
    s.type = ShapeType::Plane;
    s.prim = MakePlanePrim(pos);
    return s;
}

// Geometry-only equality vs a direct dispatch-table call (positions/normals/
// penetration/point_count + manifold_key + a/b). Param fields are NOT compared
// here: the driver merges params (friction/solimp/solref), the direct call does
// not -- those are asserted separately (see ParamMergeMatchesIndependentCall).
void ExpectGeometryMatchesDirect(const ContactManifold& got,
                                 const ContactManifold& direct,
                                 const CandidatePair& pair) {
    ASSERT_EQ(got.point_count, direct.point_count);
    EXPECT_EQ(got.manifold_key, direct.manifold_key);
    EXPECT_EQ(got.manifold_key, MakeStableKey(pair.a, pair.b));
    EXPECT_EQ(got.a.handle, pair.a.handle);
    EXPECT_EQ(got.b.handle, pair.b.handle);
    EXPECT_EQ(got.a.type, pair.a.type);
    EXPECT_EQ(got.b.type, pair.b.type);
    for (uint32_t i = 0; i < got.point_count; ++i) {
        EXPECT_FLOAT_EQ(got.points[i].position.x, direct.points[i].position.x);
        EXPECT_FLOAT_EQ(got.points[i].position.y, direct.points[i].position.y);
        EXPECT_FLOAT_EQ(got.points[i].position.z, direct.points[i].position.z);
        EXPECT_FLOAT_EQ(got.points[i].normal.x, direct.points[i].normal.x);
        EXPECT_FLOAT_EQ(got.points[i].normal.y, direct.points[i].normal.y);
        EXPECT_FLOAT_EQ(got.points[i].normal.z, direct.points[i].normal.z);
        EXPECT_FLOAT_EQ(got.points[i].penetration, direct.points[i].penetration);
    }
}

// Direct dispatch-table call for the geometry anchor (mirrors what the driver
// does internally -- but WITHOUT the param merge, so it is an independent
// reference for the geometry fields only).
ContactManifold DirectCall(const CandidatePair& pair, ShapeType ta, ShapeType tb,
                           const amf::PrimParams& pa, const amf::PrimParams& pb,
                           bool has_sdf) {
    ShapeProxyView view;
    view.type_a = ta;
    view.type_b = tb;
    view.prim_a = pa;
    view.prim_b = pb;
    ContactManifold m{};
    ResolveNarrowphase(ta, tb, has_sdf)(pair, view, &m);
    return m;
}

}  // namespace

// ---------------------------------------------------------------------------
// (1) Box-on-plane (rigid x static): driver geometry == direct call; merged
//     params + key + sides populated.
// ---------------------------------------------------------------------------
TEST(ContactStreamDriver, BoxOnPlaneMatchesDirectCallAndMerges) {
    // Box centered at y=0.4, half-height 0.5 -> bottom face at y=-0.1, below the
    // plane at y=0 -> 4-corner penetrating contact.
    const ResolvedShape box = MakeBoxShape(0.5f, 0.5f, 0.5f, {0.0f, 0.4f, 0.0f});
    const ResolvedShape plane = MakePlaneShape({0.0f, 0.0f, 0.0f});

    MapResolver res;
    res.Set(10u, box);
    res.Set(20u, plane);

    const CandidatePair pair = MakePair(RigidRef(10u), StaticRef(20u));
    std::vector<CandidatePair> pairs{pair};

    std::vector<ContactManifold> out;
    BuildContactManifolds(pairs, res.Fn(), &out);

    ASSERT_EQ(out.size(), 1u);
    const ContactManifold direct =
        DirectCall(pair, ShapeType::Box, ShapeType::Plane, box.prim, plane.prim, false);
    ExpectGeometryMatchesDirect(out[0], direct, pair);

    // Box bottom is 0.1 below the plane -> 4 corners, each penetration 0.1, normal
    // = world +Y (separation dir for box == side A).
    EXPECT_EQ(out[0].point_count, 4u);
    for (uint32_t i = 0; i < out[0].point_count; ++i) {
        EXPECT_NEAR(out[0].points[i].penetration, 0.1f, 1e-5f);
        EXPECT_NEAR(out[0].points[i].normal.y, 1.0f, 1e-5f);
    }

    // Merged params populated (default-vs-default merge: friction max(1,1)=1).
    EXPECT_FLOAT_EQ(out[0].friction, 1.0f);
    EXPECT_FLOAT_EQ(out[0].solimp[0], 0.9f);
    for (uint32_t i = 0; i < out[0].point_count; ++i) {
        EXPECT_FLOAT_EQ(out[0].points[i].solref_timeconst, 0.02f);
        EXPECT_FLOAT_EQ(out[0].points[i].solref_dampratio, 1.0f);
    }
}

// ---------------------------------------------------------------------------
// (2) Foot-style sphere-on-plane (articulation-link x static): 1-point manifold,
//     right penetration + side refs (the C5c foot-ground shape).
// ---------------------------------------------------------------------------
TEST(ContactStreamDriver, FootSphereOnPlaneArticulationLinkSide) {
    // Foot sphere radius 0.05 whose center is 0.03 above the ground -> sphere
    // bottom is 0.02 below the plane -> penetration 0.02.
    const ResolvedShape foot = MakeSphereShape(0.05f, {0.3f, 0.03f, -0.1f});
    const ResolvedShape ground = MakePlaneShape({0.0f, 0.0f, 0.0f});

    MapResolver res;
    res.Set(101u, foot);    // articulation link handle (global link id, C5c)
    res.Set(0u, ground);

    const CandidatePair pair = MakePair(LinkRef(101u), StaticRef(0u));
    std::vector<CandidatePair> pairs{pair};

    std::vector<ContactManifold> out;
    BuildContactManifolds(pairs, res.Fn(), &out);

    ASSERT_EQ(out.size(), 1u);
    EXPECT_EQ(out[0].point_count, 1u);
    EXPECT_NEAR(out[0].points[0].penetration, 0.02f, 1e-5f);
    EXPECT_NEAR(out[0].points[0].normal.y, 1.0f, 1e-5f);  // sep dir for sphere==A
    // Side refs: A is the articulation link, B is the static world.
    EXPECT_EQ(out[0].a.type, CollidableType::ArticulationLink);
    EXPECT_EQ(out[0].a.handle, 101u);
    EXPECT_EQ(out[0].b.type, CollidableType::StaticWorld);
    EXPECT_EQ(out[0].b.handle, 0u);
}

// ---------------------------------------------------------------------------
// (3) Box-box (rigid x rigid): multi-point manifold == the direct analytical
//     box-box call.
// ---------------------------------------------------------------------------
TEST(ContactStreamDriver, BoxBoxMatchesDirectCall) {
    // Two unit boxes stacked with a small overlap along Y.
    const ResolvedShape boxA = MakeBoxShape(0.5f, 0.5f, 0.5f, {0.0f, 0.9f, 0.0f});
    const ResolvedShape boxB = MakeBoxShape(0.5f, 0.5f, 0.5f, {0.0f, 0.0f, 0.0f});

    MapResolver res;
    res.Set(5u, boxA);
    res.Set(6u, boxB);

    const CandidatePair pair = MakePair(RigidRef(5u), RigidRef(6u));
    std::vector<CandidatePair> pairs{pair};

    std::vector<ContactManifold> out;
    BuildContactManifolds(pairs, res.Fn(), &out);

    ASSERT_EQ(out.size(), 1u);
    const ContactManifold direct = DirectCall(pair, ShapeType::Box, ShapeType::Box,
                                              boxA.prim, boxB.prim, false);
    ASSERT_GT(direct.point_count, 1u);  // a face-clip patch, not a single point
    ExpectGeometryMatchesDirect(out[0], direct, pair);
}

// ---------------------------------------------------------------------------
// (4) Param merge: two sides with different friction/solref/priority -> the
//     merged manifold params match an INDEPENDENT MergeContactParams call.
// ---------------------------------------------------------------------------
TEST(ContactStreamDriver, ParamMergeMatchesIndependentCall) {
    ResolvedShape box = MakeBoxShape(0.5f, 0.5f, 0.5f, {0.0f, 0.4f, 0.0f});
    box.friction  = 0.7f;
    box.solref[0] = 0.03f;
    box.solref[1] = 0.8f;
    box.solimp[0] = 0.8f;
    box.priority  = 0;
    box.solmix    = 2.0f;

    ResolvedShape plane = MakePlaneShape({0.0f, 0.0f, 0.0f});
    plane.friction  = 1.2f;
    plane.solref[0] = 0.05f;
    plane.solref[1] = 1.5f;
    plane.solimp[0] = 0.95f;
    plane.priority  = 0;
    plane.solmix    = 1.0f;

    MapResolver res;
    res.Set(10u, box);
    res.Set(20u, plane);
    const CandidatePair pair = MakePair(RigidRef(10u), StaticRef(20u));
    std::vector<CandidatePair> pairs{pair};

    std::vector<ContactManifold> out;
    BuildContactManifolds(pairs, res.Fn(), &out);
    ASSERT_EQ(out.size(), 1u);

    // INDEPENDENT MergeContactParams call (side A = box, side B = plane), built
    // straight from the same fields -> cross-check, not an echo of the driver.
    nuka::scene::ContactParamsIn ia;
    ia.priority = box.priority; ia.condim = static_cast<uint8_t>(box.condim);
    ia.friction_mu = box.friction;
    ia.solref[0] = box.solref[0]; ia.solref[1] = box.solref[1];
    for (int i = 0; i < 5; ++i) ia.solimp[i] = box.solimp[i];
    ia.solmix = box.solmix;
    nuka::scene::ContactParamsIn ib;
    ib.priority = plane.priority; ib.condim = static_cast<uint8_t>(plane.condim);
    ib.friction_mu = plane.friction;
    ib.solref[0] = plane.solref[0]; ib.solref[1] = plane.solref[1];
    for (int i = 0; i < 5; ++i) ib.solimp[i] = plane.solimp[i];
    ib.solmix = plane.solmix;
    const nuka::scene::MergedContactParams merged =
        nuka::scene::MergeContactParams(ia, ib);

    EXPECT_FLOAT_EQ(out[0].friction, merged.friction_mu);  // max(0.7,1.2)=1.2
    for (int i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(out[0].solimp[i], merged.solimp[i]);
    }
    ASSERT_GT(out[0].point_count, 0u);
    for (uint32_t p = 0; p < out[0].point_count; ++p) {
        EXPECT_FLOAT_EQ(out[0].points[p].solref_timeconst, merged.solref[0]);
        EXPECT_FLOAT_EQ(out[0].points[p].solref_dampratio, merged.solref[1]);
    }
    // Sanity: the merge actually moved the params off the bare-manifold defaults.
    EXPECT_NE(out[0].friction, 0.5f);
}

// ---------------------------------------------------------------------------
// (5) has_sdf BOTH-sides contract. Same overlapping sphere+plane geometry, flip
//     only the SDF flags: one-side SDF -> has_sdf=false -> real SpherePlane
//     (non-empty, appended); both-side SDF + null seam -> NarrowphaseSdf null-
//     guards -> empty -> skipped. Isolates the `&&`. Plus a direct routing check.
// ---------------------------------------------------------------------------
TEST(ContactStreamDriver, HasSdfBothSidesContract) {
    // Direct routing assertion: both sides SDF -> the real Sdf handler.
    EXPECT_EQ(ResolveNarrowphase(ShapeType::Sphere, ShapeType::Plane, true),
              &NarrowphaseSdf);
    // One side SDF -> NOT the Sdf handler (falls through to analytical).
    EXPECT_NE(ResolveNarrowphase(ShapeType::Sphere, ShapeType::Plane, false),
              &NarrowphaseSdf);

    // Overlapping sphere (center 0.03 above ground, r=0.05 -> pen 0.02).
    const auto base_sphere = MakeSphereShape(0.05f, {0.0f, 0.03f, 0.0f});
    const auto base_plane = MakePlaneShape({0.0f, 0.0f, 0.0f});

    // (a) one side has SDF -> has_sdf = false -> real analytical SpherePlane.
    {
        ResolvedShape sph = base_sphere; sph.has_sdf = true;   // only A
        ResolvedShape pln = base_plane;  pln.has_sdf = false;  // B no SDF
        MapResolver res; res.Set(1u, sph); res.Set(2u, pln);
        const CandidatePair pair = MakePair(RigidRef(1u), StaticRef(2u));
        std::vector<CandidatePair> pairs{pair};
        std::vector<ContactManifold> out;
        BuildContactManifolds(pairs, res.Fn(), &out);
        ASSERT_EQ(out.size(), 1u) << "one-sided SDF must route to analytical";
        EXPECT_EQ(out[0].point_count, 1u);
        EXPECT_NEAR(out[0].points[0].penetration, 0.02f, 1e-5f);
    }

    // (b) both sides have SDF -> has_sdf = true -> NarrowphaseSdf. The geom seam
    //     is null (no SdfProxyView supplied) -> the handler null-guards to an
    //     empty manifold -> the driver skips it (point_count 0). Same geometry as
    //     (a) but the opposite output, isolating the `&&`.
    {
        ResolvedShape sph = base_sphere; sph.has_sdf = true;
        ResolvedShape pln = base_plane;  pln.has_sdf = true;
        MapResolver res; res.Set(1u, sph); res.Set(2u, pln);
        const CandidatePair pair = MakePair(RigidRef(1u), StaticRef(2u));
        std::vector<CandidatePair> pairs{pair};
        std::vector<ContactManifold> out;
        BuildContactManifolds(pairs, res.Fn(), &out);
        EXPECT_EQ(out.size(), 0u)
            << "both-sided SDF routes to Sdf tier; null seam -> empty -> skipped";
    }
}

// ---------------------------------------------------------------------------
// (6) Skip: a pair whose resolver returns false for a side -> no manifold.
// ---------------------------------------------------------------------------
TEST(ContactStreamDriver, SkipsPairWhenSideFailsToResolve) {
    const ResolvedShape box = MakeBoxShape(0.5f, 0.5f, 0.5f, {0.0f, 0.4f, 0.0f});
    MapResolver res;
    res.Set(10u, box);  // side A resolves; side B (handle 99) does NOT.

    const CandidatePair pair = MakePair(RigidRef(10u), StaticRef(99u));
    std::vector<CandidatePair> pairs{pair};
    std::vector<ContactManifold> out;
    BuildContactManifolds(pairs, res.Fn(), &out);
    EXPECT_EQ(out.size(), 0u);
}

// ---------------------------------------------------------------------------
// (7) D1: two runs on the same pairs + resolver are byte-identical.
// ---------------------------------------------------------------------------
TEST(ContactStreamDriver, DeterministicTwoRunByteExact) {
    const ResolvedShape boxA = MakeBoxShape(0.5f, 0.5f, 0.5f, {0.0f, 0.9f, 0.0f});
    const ResolvedShape boxB = MakeBoxShape(0.5f, 0.5f, 0.5f, {0.0f, 0.0f, 0.0f});
    const ResolvedShape plane = MakePlaneShape({0.0f, 0.0f, 0.0f});
    const ResolvedShape sph = MakeSphereShape(0.05f, {1.0f, 0.03f, 0.0f});

    MapResolver res;
    res.Set(5u, boxA);
    res.Set(6u, boxB);
    res.Set(20u, plane);
    res.Set(7u, sph);

    // A few pairs (already in stable_key order is not required for D1 -- the
    // driver preserves the GIVEN order deterministically either way).
    std::vector<CandidatePair> pairs{
        MakePair(RigidRef(5u), RigidRef(6u)),   // box-box
        MakePair(RigidRef(5u), StaticRef(20u)), // box-plane
        MakePair(RigidRef(7u), StaticRef(20u)), // sphere-plane
    };

    std::vector<ContactManifold> out1;
    std::vector<ContactManifold> out2;
    BuildContactManifolds(pairs, res.Fn(), &out1);
    BuildContactManifolds(pairs, res.Fn(), &out2);

    ASSERT_EQ(out1.size(), out2.size());
    ASSERT_GT(out1.size(), 0u);
    ASSERT_EQ(sizeof(ContactManifold) * out1.size(),
              sizeof(ContactManifold) * out2.size());
    EXPECT_EQ(std::memcmp(out1.data(), out2.data(),
                          sizeof(ContactManifold) * out1.size()),
              0)
        << "two-run byte-exact (D1)";
}
