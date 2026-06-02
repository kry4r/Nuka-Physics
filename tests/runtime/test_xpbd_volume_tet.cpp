// ---------------------------------------------------------------------------
// v0.7 p09-B: XPBD VOLUME (id 8) tet tests -- signed-volume preservation.
//
// The V3 FD adjoint gate (test_adjoint_fd_xpbd_volume) validates the multilinear
// XPBD multiplier law, BLIND to the determinant gradient grad C. This suite is
// where grad C is exercised:
//
//   1. Determinant gradient: a host central-difference of C = det(.) - 6*V_rest
//      wrt each tet-vertex coordinate matches the analytic cross-product gradients
//      the solver uses (rel-err < 1e-3), and the per-vertex gradients sum to ~0
//      (momentum conservation).
//   2. ISOLATED volume restoration: a tet with NO edge constraints, COMPRESSED
//      below rest volume, is driven back to its rest volume by the volume
//      constraint ALONE (the discriminating test -- 6 stiff edges would already
//      determine the shape and mask a zeroed volume gradient).
//   3. Multi-tet block: a small tet soft body under gravity keeps each tet's
//      volume near rest (drift bounded), stays finite.
//   4. D1 two-run byte-exactness of the volume forward.
//
// A full Vellum golden is DEFERRED to p15; these are ANALYTIC/physical invariants.
// ---------------------------------------------------------------------------

#include "runtime/soft/tetmesh_topology.hpp"
#include "runtime/soft/xpbd_world.hpp"

#include "math/vec3.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace {

using nuka::math::Vec3;
using nuka::runtime::soft::BuildTetMeshConstraints;
using nuka::runtime::soft::StepXpbdWorld;
using nuka::runtime::soft::TetMeshTet;
using nuka::runtime::soft::TetMeshTopologyOptions;
using nuka::runtime::soft::TetSignedVolumeTimes6;
using nuka::runtime::soft::UploadXpbdWorld;
using nuka::runtime::soft::XpbdConstraintSet;
using nuka::runtime::soft::XpbdParticleSet;
using nuka::runtime::soft::XpbdStepOptions;
using nuka::runtime::soft::XpbdWorld;
using nuka::runtime::soft::XpbdWorldState;

// A unit-ish reference tet (matches the solver's p0..p3 ordering).
struct RefTet {
    Vec3 p0{0.1f, 0.2f, 0.05f};
    Vec3 p1{1.2f, 0.1f, 0.0f};
    Vec3 p2{0.0f, 1.1f, 0.15f};
    Vec3 p3{0.05f, 0.0f, 1.3f};
};

} // namespace

// Gate 1: analytic determinant gradients == host central-difference, and the
// four per-vertex gradients sum to ~0 (momentum conservation).
TEST(XpbdVolumeTet, DeterminantGradientMatchesFdAndSumsToZero) {
    RefTet t;
    std::vector<Vec3> p = {t.p0, t.p1, t.p2, t.p3};

    // Analytic gradients (the exact cross products the solver uses):
    //   e1=p1-p0, e2=p2-p0, e3=p3-p0
    //   g1=e2 x e3 ; g2=e3 x e1 ; g3=e1 x e2 ; g0 = -(g1+g2+g3)
    const Vec3 e1 = p[1] - p[0];
    const Vec3 e2 = p[2] - p[0];
    const Vec3 e3 = p[3] - p[0];
    Vec3 g[4];
    g[1] = e2.Cross(e3);
    g[2] = e3.Cross(e1);
    g[3] = e1.Cross(e2);
    g[0] = (g[1] + g[2] + g[3]) * -1.0f;

    // Sum of gradients ~ 0.
    Vec3 sum = g[0] + g[1] + g[2] + g[3];
    EXPECT_NEAR(sum.Length(), 0.0f, 1.0e-4f);

    auto det = [&](const std::vector<Vec3>& q) {
        return TetSignedVolumeTimes6(q[0], q[1], q[2], q[3]);
    };

    const float h = 1.0e-3f;
    for (int v = 0; v < 4; ++v) {
        for (int axis = 0; axis < 3; ++axis) {
            std::vector<Vec3> q = p;
            float& comp = (axis == 0) ? q[v].x : (axis == 1) ? q[v].y : q[v].z;
            const float saved = comp;
            comp = saved + h;
            const float cp = det(q);
            comp = saved - h;
            const float cm = det(q);
            const float fd = (cp - cm) / (2.0f * h);
            const float analytic =
                (axis == 0) ? g[v].x : (axis == 1) ? g[v].y : g[v].z;
            const float denom = std::fabs(analytic) + std::fabs(fd) + 1.0e-6f;
            EXPECT_LT(std::fabs(analytic - fd) / denom, 1.0e-3f)
                << "vertex " << v << " axis " << axis << " analytic=" << analytic
                << " fd=" << fd;
        }
    }
}

// Gate 2: ISOLATED volume restoration. One tet, NO edge constraints, COMPRESSED
// to 60% of rest volume (uniform scale about the centroid), zero gravity. The
// volume constraint alone must drive the volume back toward rest. With zero edge
// constraints there is nothing else preserving the shape, so a zeroed volume
// gradient would leave the tet compressed -- this test would fail. (Stiff edges
// would mask it; that is why we omit them.)
TEST(XpbdVolumeTet, IsolatedVolumeConstraintRestoresRestVolume) {
    RefTet t;
    const std::vector<Vec3> rest = {t.p0, t.p1, t.p2, t.p3};
    const float rest_v6 = TetSignedVolumeTimes6(rest[0], rest[1], rest[2], rest[3]);

    // Compress: scale about the centroid by cbrt(0.6) so volume -> 0.6*rest.
    Vec3 centroid{0.0f, 0.0f, 0.0f};
    for (const Vec3& q : rest) {
        centroid += q;
    }
    centroid *= 0.25f;
    const float scale = std::cbrt(0.6f);

    XpbdParticleSet particles;
    for (const Vec3& q : rest) {
        particles.positions.push_back(centroid + (q - centroid) * scale);
        particles.velocities.push_back(Vec3{0.0f, 0.0f, 0.0f});
        particles.inv_masses.push_back(1.0f);
    }

    std::vector<TetMeshTet> tets = {TetMeshTet{{0u, 1u, 2u, 3u}}};
    TetMeshTopologyOptions opts;
    opts.emit_distance_constraints = false;  // ISOLATE the volume constraint.
    opts.volume_compliance_alpha = 0.0f;     // rigid volume.
    XpbdConstraintSet cs;
    BuildTetMeshConstraints(rest, tets, opts, cs);
    ASSERT_EQ(cs.volume.size(), 1u);
    ASSERT_EQ(cs.distance.size(), 0u);
    // Rest value stored by the cook builder must equal the rest det exactly.
    EXPECT_NEAR(cs.volume[0].rest_volume_times6, rest_v6, 1.0e-5f);

    XpbdWorld world = UploadXpbdWorld(particles, cs);
    XpbdStepOptions so;
    so.gravity = Vec3{0.0f, 0.0f, 0.0f};
    so.dt = 1.0f / 240.0f;
    so.step_count = 1u;
    so.solver_iterations = 10u;

    const float compressed_v6 =
        TetSignedVolumeTimes6(particles.positions[0], particles.positions[1],
                              particles.positions[2], particles.positions[3]);
    ASSERT_LT(std::fabs(compressed_v6), std::fabs(rest_v6))
        << "setup sanity: compressed volume must start below rest";

    for (uint32_t s = 0; s < 200u; ++s) {
        StepXpbdWorld(world, so);
    }
    const XpbdWorldState st = world.DownloadState();
    const float final_v6 = TetSignedVolumeTimes6(st.positions[0], st.positions[1],
                                                 st.positions[2], st.positions[3]);

    const float rel_err = std::fabs(final_v6 - rest_v6) / std::fabs(rest_v6);
    EXPECT_LT(rel_err, 1.0e-2f)
        << "volume not restored: rest6=" << rest_v6
        << " compressed6=" << compressed_v6 << " final6=" << final_v6
        << " rel_err=" << rel_err;
}

namespace {

// A 2x1x1 stack of two tets sharing a face (a small tet block) over the unit
// cube split into tets. Returns particles + tet list.
struct TetBlock {
    XpbdParticleSet particles;
    std::vector<TetMeshTet> tets;
};

TetBlock MakeTwoTetBlock() {
    TetBlock b;
    // Cube corners (z=0 .. z=1).
    const Vec3 corners[5] = {
        Vec3{0.0f, 0.0f, 1.0f},  // 0 apex top
        Vec3{0.0f, 0.0f, 0.0f},  // 1
        Vec3{1.0f, 0.0f, 0.0f},  // 2
        Vec3{0.0f, 1.0f, 0.0f},  // 3
        Vec3{1.0f, 1.0f, 0.2f},  // 4
    };
    for (const Vec3& c : corners) {
        b.particles.positions.push_back(c);
        b.particles.velocities.push_back(Vec3{0.0f, 0.0f, 0.0f});
        b.particles.inv_masses.push_back(1.0f);
    }
    // Pin the base triangle (1,2,3) so the block does not free-fall.
    b.particles.inv_masses[1] = 0.0f;
    b.particles.inv_masses[2] = 0.0f;
    b.particles.inv_masses[3] = 0.0f;
    b.tets = {TetMeshTet{{0u, 1u, 2u, 3u}}, TetMeshTet{{2u, 3u, 4u, 0u}}};
    return b;
}

} // namespace

// Gate 3 (multi-tet invariant) + Gate 4 (D1): a small tet block under gravity
// keeps each tet's volume near rest and is two-run byte-exact.
TEST(XpbdVolumeTet, BlockPreservesVolumeUnderGravityAndIsByteExact) {
    auto run = []() {
        TetBlock b = MakeTwoTetBlock();
        TetMeshTopologyOptions opts;
        opts.distance_compliance_alpha = 0.0f;  // stiff edges
        opts.volume_compliance_alpha = 0.0f;    // stiff volume
        XpbdConstraintSet cs;
        BuildTetMeshConstraints(b.particles.positions, b.tets, opts, cs);

        std::vector<float> rest_v6;
        for (const auto& vc : cs.volume) {
            rest_v6.push_back(vc.rest_volume_times6);
        }
        std::vector<TetMeshTet> tets = b.tets;

        XpbdWorld world = UploadXpbdWorld(b.particles, cs);
        XpbdStepOptions so;
        so.gravity = Vec3{0.0f, 0.0f, -9.81f};
        so.dt = 1.0f / 240.0f;
        so.step_count = 1u;
        so.solver_iterations = 20u;
        for (uint32_t s = 0; s < 300u; ++s) {
            StepXpbdWorld(world, so);
        }
        return std::make_tuple(world.DownloadState(), tets, rest_v6);
    };

    const auto a = run();
    const auto b = run();
    const XpbdWorldState& sa = std::get<0>(a);

    for (const Vec3& p : sa.positions) {
        ASSERT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z));
    }

    // Each tet's volume stays near rest.
    const auto& tets = std::get<1>(a);
    const auto& rest_v6 = std::get<2>(a);
    float max_rel = 0.0f;
    for (std::size_t c = 0; c < tets.size(); ++c) {
        const auto& tet = tets[c];
        const float v6 = TetSignedVolumeTimes6(
            sa.positions[tet.v[0]], sa.positions[tet.v[1]],
            sa.positions[tet.v[2]], sa.positions[tet.v[3]]);
        max_rel = std::max(max_rel, std::fabs(v6 - rest_v6[c]) / std::fabs(rest_v6[c]));
    }
    EXPECT_LT(max_rel, 0.05f) << "max relative volume drift " << max_rel;

    // D1 two-run byte-exact.
    const XpbdWorldState& sb = std::get<0>(b);
    EXPECT_EQ(std::memcmp(sa.positions.data(), sb.positions.data(),
                          sa.positions.size() * sizeof(Vec3)),
              0)
        << "volume forward position buffer not two-run byte-identical (D1 violation)";
    EXPECT_EQ(std::memcmp(sa.velocities.data(), sb.velocities.data(),
                          sa.velocities.size() * sizeof(Vec3)),
              0)
        << "volume forward velocity buffer not two-run byte-identical (D1 violation)";
}
