// ---------------------------------------------------------------------------
// v0.8 C3d -- the SDF high-precision tier WIRED into the narrowphase dispatch.
// ---------------------------------------------------------------------------
// C3d ends the orphan status of find_sdf_contact_newton (sdf_contact.hpp): until
// now it had ZERO stepping callsites (architecture §0(c) dead code). The Sdf-tier
// dispatch handler (NarrowphaseSdf, narrowphase_dispatch.hpp) is its FIRST caller.
// This test proves:
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
// newton needs the INITIAL guess in BOTH narrow bands (sdf_contact.hpp:184-187),
// and the handler seeds the guess at the MIDPOINT of the two body centers. So the
// test uses TWO EQUAL unit cubes whose centers are SYMMETRIC about y=0 and which
// overlap by ~2 voxels: the contact plane (and the midpoint) is y=0, squarely in
// both bands. (A box-on-a-big-ground-slab would pull the midpoint out of both
// bands -- the documented "midpoint may miss the bands -> empty manifold" case --
// so it is NOT a valid in-contact fixture for this seed strategy.)
//
// The handler calls the header-inline NUKA_SDF_HD find_sdf_contact_newton -- the
// SAME function the device kernel wraps -- so this host test validates the
// SHIPPING SDF path through the dispatch seam.
// ---------------------------------------------------------------------------

#include "collision/narrowphase_dispatch.hpp"

#include "constraint/collidable.hpp"
#include "constraint/contact_manifold.hpp"
#include "import/cooker/sparse_sdf_cooker.hpp"
#include "tests/import/sdf_host_sampler.hpp"
#include "tests/import/sdf_test_meshes.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>

namespace {

using nuka::collision::CandidatePair;
using nuka::collision::NarrowphaseSdf;
using nuka::collision::ResolveNarrowphase;
using nuka::collision::SdfBodyFrame;
using nuka::collision::SdfProxyView;
using nuka::collision::ShapeProxyView;
using nuka::constraint::CollidableRef;
using nuka::constraint::CollidableType;
using nuka::constraint::ContactManifold;
using nuka::constraint::ReactionProviderKind;
using nuka::import::cooker::CookSparseSdf;
using nuka::import::cooker::SparseSdfData;
using nuka::import::cooker::SparseSdfParams;
using nuka::math::Vec3;
using nuka::scene::ShapeType;

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
