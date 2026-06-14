// ---------------------------------------------------------------------------
// v0.7 p09-B: XPBD VOLUME (id 8) tet tests -- signed-volume preservation,
// M9 T11 Phase-2b RE-POINTED to nk (the legacy XPBD soft stepper is
// deleted; the volume solve is op-ified onto the unified nk core -- particles.cu
// XpbdVolumeKernel, run inside nk::World's particle solve). The grad-C FD gate is
// PURE HOST (stepper-independent) and unchanged; the volume-restore + block-volume
// gates move from the legacy stepper to nk::World (cook -> World -> Step).
//
// The V3 FD adjoint gate (test_adjoint_fd_xpbd_volume) validates the multilinear
// XPBD multiplier law, BLIND to the determinant gradient grad C. This suite is
// where grad C is exercised:
//
//   1. Determinant gradient (HOST): a host central-difference of C = det(.) -
//      6*V_rest wrt each tet-vertex coordinate matches the analytic cross-product
//      gradients the solver uses (rel-err < 1e-3), and the per-vertex gradients
//      sum to ~0 (momentum conservation).
//   2. ISOLATED volume restoration (nk): a tet with NO edge constraints,
//      COMPRESSED below rest volume, is driven back to its rest volume by the
//      volume constraint ALONE (the discriminating test -- 6 stiff edges would
//      already determine the shape and mask a zeroed volume gradient).
//   3. Multi-tet block (nk): a small tet soft body under gravity keeps each tet's
//      volume near rest (drift bounded), stays finite.
//   4. D1 two-run byte-exactness of the volume forward (nk).
//
// A full Vellum golden is DEFERRED to p15; these are ANALYTIC/physical invariants.
// ---------------------------------------------------------------------------

#include "import/cooker/xpbd_cooker_types.hpp"  // XpbdConstraintSet/XpbdParticleSet (CUDA-free PODs)
#include "runtime/soft/tetmesh_topology.hpp"

#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/scoped_device_guard.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <tuple>
#include <vector>

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Vec3;
using nuka::runtime::soft::BuildTetMeshConstraints;
using nuka::runtime::soft::TetMeshTet;
using nuka::runtime::soft::TetMeshTopologyOptions;
using nuka::runtime::soft::TetSignedVolumeTimes6;
using nuka::runtime::soft::XpbdConstraintSet;
using nuka::runtime::soft::XpbdParticleSet;

// nk backend context (shared singleton, the nk_particle_equivalence pattern).
struct NkCtx { nphi::Device* dev = nullptr; nphi::Backend* backend = nullptr; };
NkCtx GetNkCtx() {
    static NkCtx c = [] {
        NkCtx r;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return c;
}

// Downloaded particle state (replaces the legacy soft world state).
struct XpbdState {
    std::vector<Vec3> positions;
    std::vector<Vec3> velocities;
};

// Cook an XpbdParticleSet + XpbdConstraintSet (distance + volume rows) into an
// nk::Model (mode = Xpbd), transcribing the de-interleaved SoA exactly as
// CookXpbdParticles does (dist a/b/rest/alpha; volume 4-particle + rest6 + alpha).
nk::Model BuildNkTetModel(const XpbdParticleSet& ps, const XpbdConstraintSet& cs,
                          uint16_t iters) {
    nk::Model model;
    nk::Model::ModelParticles& mp = model.particles;
    mp.mode = nk::Model::ParticleMode::Xpbd;
    mp.initial_pos = ps.positions;
    mp.initial_vel = ps.velocities;
    mp.inv_mass = ps.inv_masses;
    mp.xpbd_iters = iters == 0u ? 1u : iters;
    const uint32_t dn = static_cast<uint32_t>(cs.distance.size());
    for (uint32_t c = 0; c < dn; ++c) {
        mp.dist_a.push_back(cs.distance[c].particle_a);
        mp.dist_b.push_back(cs.distance[c].particle_b);
        mp.dist_rest.push_back(cs.distance[c].rest_length);
        mp.dist_alpha.push_back(cs.distance[c].compliance_alpha);
    }
    const uint32_t vn = static_cast<uint32_t>(cs.volume.size());
    for (uint32_t c = 0; c < vn; ++c) {
        for (uint32_t j = 0; j < 4u; ++j) {
            mp.vol_particles.push_back(cs.volume[c].particle[j]);
        }
        mp.vol_rest6.push_back(cs.volume[c].rest_volume_times6);
        mp.vol_alpha.push_back(cs.volume[c].compliance_alpha);
    }
    nk::ModelCapacities& cap = model.capacities;
    cap.env_count = 1;
    cap.particles_per_env = static_cast<uint32_t>(ps.positions.size());
    cap.dist_cons_per_env = dn;
    cap.vol_cons_per_env = vn;
    return model;
}

// Run an nk tet world for kSteps with the given gravity/dt and download state.
XpbdState RunNkTet(const XpbdParticleSet& ps, const XpbdConstraintSet& cs,
                   uint16_t iters, Vec3 gravity, float dt, uint32_t kSteps) {
    NkCtx c = GetNkCtx();
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = dt;
    cfg.gravity[0] = gravity.x; cfg.gravity[1] = gravity.y; cfg.gravity[2] = gravity.z;
    nk::World world(BuildNkTetModel(ps, cs, iters), 1u, c.dev, c.backend, cfg);
    EXPECT_TRUE(world.Ready());
    const uint32_t P = world.GetModel().capacities.particles_per_env;
    for (uint32_t s = 0; s < kSteps; ++s) world.Step();
    XpbdState st;
    st.positions.resize(P);
    st.velocities.resize(P);
    world.GetData().DownloadField(nk::FieldId::ParticlePos, st.positions.data(),
                                  P * sizeof(Vec3));
    world.GetData().DownloadField(nk::FieldId::ParticleVel, st.velocities.data(),
                                  P * sizeof(Vec3));
    return st;
}

// A unit-ish reference tet (matches the solver's p0..p3 ordering).
struct RefTet {
    Vec3 p0{0.1f, 0.2f, 0.05f};
    Vec3 p1{1.2f, 0.1f, 0.0f};
    Vec3 p2{0.0f, 1.1f, 0.15f};
    Vec3 p3{0.05f, 0.0f, 1.3f};
};

} // namespace

// Gate 1 (HOST, stepper-independent): analytic determinant gradients == host
// central-difference, and the four per-vertex gradients sum to ~0 (momentum).
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

// Gate 2 (nk): ISOLATED volume restoration. One tet, NO edge constraints,
// COMPRESSED to 60% of rest volume (uniform scale about the centroid), zero
// gravity. The volume constraint alone must drive the volume back toward rest. With
// zero edge constraints there is nothing else preserving the shape, so a zeroed
// volume gradient would leave the tet compressed -- this test would fail. (Stiff
// edges would mask it; that is why we omit them.)
TEST(XpbdVolumeTet, IsolatedVolumeConstraintRestoresRestVolume) {
    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
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

    const float compressed_v6 =
        TetSignedVolumeTimes6(particles.positions[0], particles.positions[1],
                              particles.positions[2], particles.positions[3]);
    ASSERT_LT(std::fabs(compressed_v6), std::fabs(rest_v6))
        << "setup sanity: compressed volume must start below rest";

    // nk path: gravity OFF (isolate the constraint), 10 GS iters, 200 steps.
    const XpbdState st = RunNkTet(particles, cs, /*iters=*/10u,
                                  Vec3{0.0f, 0.0f, 0.0f}, 1.0f / 240.0f,
                                  /*kSteps=*/200u);
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

// Gate 3 (nk, multi-tet invariant) + Gate 4 (nk, D1): a small tet block under
// gravity keeps each tet's volume near rest and is two-run byte-exact.
TEST(XpbdVolumeTet, BlockPreservesVolumeUnderGravityAndIsByteExact) {
    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
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

        // nk path: gravity (z-down) + stiff edges/volume, 20 GS iters, 300 steps.
        const XpbdState st = RunNkTet(b.particles, cs, /*iters=*/20u,
                                      Vec3{0.0f, 0.0f, -9.81f}, 1.0f / 240.0f,
                                      /*kSteps=*/300u);
        return std::make_tuple(st, tets, rest_v6);
    };

    const auto a = run();
    const auto b = run();
    const XpbdState& sa = std::get<0>(a);

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
    const XpbdState& sb = std::get<0>(b);
    EXPECT_EQ(std::memcmp(sa.positions.data(), sb.positions.data(),
                          sa.positions.size() * sizeof(Vec3)),
              0)
        << "volume forward position buffer not two-run byte-identical (D1 violation)";
    EXPECT_EQ(std::memcmp(sa.velocities.data(), sb.velocities.data(),
                          sa.velocities.size() * sizeof(Vec3)),
              0)
        << "volume forward velocity buffer not two-run byte-identical (D1 violation)";
}
