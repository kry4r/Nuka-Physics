#include "constraint/dihedral_bend.hpp"
#include "import/cooker/xpbd_cooker_types.hpp"  // XpbdParticleSet / XpbdConstraintSet (host POD)
#include "runtime/soft/cloth_topology.hpp"
#include "runtime/soft/tetmesh_topology.hpp"  // TetSignedVolumeTimes6 (planarity metric)

#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/scoped_device_guard.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Vec3;
using nuka::runtime::soft::BuildClothConstraints;
using nuka::runtime::soft::ClothTopologyOptions;
using nuka::runtime::soft::ClothTriangle;
using nuka::runtime::soft::TetSignedVolumeTimes6;
using nuka::runtime::soft::XpbdBendConstraint;
using nuka::runtime::soft::XpbdConstraintSet;
using nuka::runtime::soft::XpbdParticleSet;

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

struct XpbdState {
    std::vector<Vec3> positions;
    std::vector<Vec3> velocities;
};

nk::Model BuildNkClothModel(const XpbdParticleSet& ps, const XpbdConstraintSet& cs,
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
    const uint32_t bn = static_cast<uint32_t>(cs.bend.size());
    for (uint32_t c = 0; c < bn; ++c) {
        for (uint32_t j = 0; j < 4u; ++j) {
            mp.bend_particles.push_back(cs.bend[c].particle[j]);
        }
        mp.bend_rest_angle.push_back(cs.bend[c].rest_angle);
        mp.bend_alpha.push_back(cs.bend[c].compliance_alpha);
    }
    nk::ModelCapacities& cap = model.capacities;
    cap.env_count = 1;
    cap.particles_per_env = static_cast<uint32_t>(ps.positions.size());
    cap.dist_cons_per_env = dn;
    cap.bend_cons_per_env = bn;
    return model;
}

XpbdState RunNkCloth(const XpbdParticleSet& ps, const XpbdConstraintSet& cs,
                          uint16_t iters, Vec3 gravity, float dt, uint32_t kSteps) {
    NkCtx c = GetNkCtx();
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = dt;
    cfg.gravity[0] = gravity.x; cfg.gravity[1] = gravity.y; cfg.gravity[2] = gravity.z;
    nk::World world(BuildNkClothModel(ps, cs, iters), 1u, c.dev, c.backend, cfg);
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

struct Flap {
    Vec3 sa{0.0f, 0.0f, 0.0f};
    Vec3 sb{1.0f, 0.0f, 0.0f};
    Vec3 a0{0.3f, 0.8f, 0.0f};
    Vec3 a1{0.6f, -0.7f, 0.0f};
};

float BendConstraintValue(const XpbdBendConstraint& bc,
                          const std::vector<Vec3>& positions) {
    const auto g = nuka::constraint::EvaluateDihedralBend(
        positions[bc.particle[0]], positions[bc.particle[1]],
        positions[bc.particle[2]], positions[bc.particle[3]]);
    return nuka::constraint::DihedralBendError(g.angle, bc.rest_angle);
}

} // namespace

TEST(XpbdBendCloth, RigidMotionAndCurvedRestPreserveBending) {
    const std::vector<Vec3> rest{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f},
                                {0.3f, 0.8f, 0.4f}, {0.6f, -0.7f, 0.2f}};
    const std::vector<ClothTriangle> triangles{{{0u, 1u, 2u}}, {{1u, 0u, 3u}}};
    XpbdConstraintSet constraints;
    BuildClothConstraints(rest, triangles, {}, constraints);
    ASSERT_EQ(constraints.bend.size(), 1u);
    EXPECT_GT(std::abs(constraints.bend[0].rest_angle), 0.1f);
    const auto rotate = [](Vec3 p) {
        const float c = std::cos(1.2f), s = std::sin(1.2f);
        return Vec3{c*p.y - s*p.z, s*p.y + c*p.z, p.x};
    };
    std::vector<Vec3> transformed;
    for (Vec3 p : rest) transformed.push_back(rotate(p) + Vec3{2.0f, -1.0f, 0.8f});
    EXPECT_NEAR(BendConstraintValue(constraints.bend[0], transformed), 0.0f, 5.0e-7f);
    const auto g = nuka::constraint::EvaluateDihedralBend(rest[0], rest[1], rest[2], rest[3]);
    const auto r = nuka::constraint::EvaluateDihedralBend(
        transformed[0], transformed[1], transformed[2], transformed[3]);
    ASSERT_TRUE(g.valid);
    ASSERT_TRUE(r.valid);
    Vec3 force{}, torque{};
    for (uint32_t i = 0; i < 4u; ++i) {
        EXPECT_NEAR((rotate(g.gradients[i]) - r.gradients[i]).Length(), 0.0f, 2.0e-6f);
        force += g.gradients[i];
        torque += rest[i].Cross(g.gradients[i]);
    }
    EXPECT_NEAR(force.Length(), 0.0f, 2.0e-6f);
    EXPECT_NEAR(torque.Length(), 0.0f, 2.0e-6f);
    if (!GetNkCtx().backend) GTEST_SKIP() << "no CUDA backend";
    XpbdParticleSet particles;
    particles.positions = transformed;
    particles.velocities.assign(4u, Vec3::Zero());
    particles.inv_masses.assign(4u, 1.0f);
    const auto state = RunNkCloth(particles, constraints, 8u, Vec3::Zero(), 0.002f, 10u);
    for (uint32_t i = 0; i < 4u; ++i)
        EXPECT_NEAR((state.positions[i] - transformed[i]).Length(), 0.0f, 5.0e-6f);
}

TEST(XpbdBendCloth, DihedralGradientMatchesFdAtFlatAndFoldedStates) {
    const Flap f;
    const std::vector<Vec3> rest{f.sa, f.sb, f.a0, f.a1};
    const std::vector<ClothTriangle> triangles{{{0u, 1u, 2u}}, {{1u, 0u, 3u}}};
    XpbdConstraintSet constraints;
    BuildClothConstraints(rest, triangles, {}, constraints);
    ASSERT_EQ(constraints.bend.size(), 1u);
    for (const auto fold : {Vec3{0, 0, 0}, Vec3{0.5f, -0.3f, 0}, Vec3{-0.9f, 0.7f, 0}}) {
        auto points = rest;
        points[2].z = fold.x;
        points[3].z = fold.y;
        const auto g = nuka::constraint::EvaluateDihedralBend(points[0], points[1], points[2], points[3]);
        ASSERT_TRUE(g.valid);
        for (uint32_t i = 0; i < 4u; ++i)
            for (uint32_t axis = 0; axis < 3u; ++axis) {
                auto perturbed = points;
                float& value = axis == 0u ? perturbed[i].x : axis == 1u ? perturbed[i].y : perturbed[i].z;
                const float original = value;
                constexpr float h = 0.001f;
                value = original + h;
                const float plus = BendConstraintValue(constraints.bend[0], perturbed);
                value = original - h;
                const float minus = BendConstraintValue(constraints.bend[0], perturbed);
                const float analytic = axis == 0u ? g.gradients[i].x : axis == 1u ? g.gradients[i].y : g.gradients[i].z;
                EXPECT_NEAR((plus-minus)/(2.0f*h), analytic, 2.0e-4f);
            }
    }
}

namespace {

struct GridCloth {
    XpbdParticleSet particles;
    std::vector<ClothTriangle> triangles;
    uint32_t nx = 0u;
    uint32_t ny = 0u;
};

GridCloth MakeGrid(uint32_t nx, uint32_t ny, float spacing, bool pin_first_row) {
    GridCloth g;
    g.nx = nx;
    g.ny = ny;
    auto idx = [nx](uint32_t i, uint32_t j) { return j * nx + i; };
    for (uint32_t j = 0; j < ny; ++j) {
        for (uint32_t i = 0; i < nx; ++i) {
            g.particles.positions.push_back(
                Vec3{static_cast<float>(i) * spacing,
                     static_cast<float>(j) * spacing, 0.0f});
            g.particles.velocities.push_back(Vec3{0.0f, 0.0f, 0.0f});
            const bool pinned = pin_first_row && (j == 0u);
            g.particles.inv_masses.push_back(pinned ? 0.0f : 1.0f);
        }
    }
    for (uint32_t j = 0; j + 1 < ny; ++j) {
        for (uint32_t i = 0; i + 1 < nx; ++i) {
            const uint32_t v00 = idx(i, j);
            const uint32_t v10 = idx(i + 1, j);
            const uint32_t v01 = idx(i, j + 1);
            const uint32_t v11 = idx(i + 1, j + 1);
            g.triangles.push_back(ClothTriangle{{v00, v10, v11}});
            g.triangles.push_back(ClothTriangle{{v00, v11, v01}});
        }
    }
    return g;
}

} // namespace

TEST(XpbdBendCloth, StiffBendFlattensFoldedFlapWhileNoneStaysFolded) {
    const std::vector<Vec3> rest = {Vec3{0.0f, 0.0f, 0.0f}, Vec3{1.0f, 0.0f, 0.0f},
                                    Vec3{0.3f, 0.8f, 0.0f}, Vec3{0.6f, -0.7f, 0.0f}};
    const std::vector<ClothTriangle> tris = {ClothTriangle{{0u, 1u, 2u}},
                                             ClothTriangle{{1u, 0u, 3u}}};

    auto nonplanarity = [](const std::vector<Vec3>& p) {
        return std::fabs(TetSignedVolumeTimes6(p[0], p[1], p[2], p[3]));
    };

    auto folded_state = [&]() {
        std::vector<Vec3> p = rest;
        p[2].z += 0.4f;   // apex0 up
        p[3].z += -0.4f;  // apex1 down -> non-planar flap
        return p;
    };

    const std::vector<Vec3> folded = folded_state();
    const float initial = nonplanarity(folded);
    ASSERT_GT(initial, 1.0e-3f) << "setup sanity: flap must start non-planar";

    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";

    auto run = [&](bool emit_bend) {
        XpbdParticleSet particles;
        particles.positions = folded;
        particles.velocities = {Vec3{0, 0, 0}, Vec3{0, 0, 0}, Vec3{0, 0, 0},
                                Vec3{0, 0, 0}};
        particles.inv_masses = {1.0f, 1.0f, 1.0f, 1.0f};

        ClothTopologyOptions opts;
        opts.emit_distance_constraints = false;  // ISOLATE bend.
        opts.emit_bend_constraints = emit_bend;
        opts.bend_compliance_alpha = 0.0f;  // rigid bend (flatten hard).
        XpbdConstraintSet cs;
        BuildClothConstraints(rest, tris, opts, cs);
        if (emit_bend) {
            EXPECT_EQ(cs.bend.size(), 1u);
        } else {
            EXPECT_EQ(cs.bend.size(), 0u);
        }
        const XpbdState st = RunNkCloth(particles, cs, /*iters=*/10u,
                                             Vec3{0.0f, 0.0f, 0.0f}, 1.0f / 240.0f,
                                             /*kSteps=*/200u);
        return nonplanarity(st.positions);
    };

    const float stiff = run(/*emit_bend=*/true);
    const float none = run(/*emit_bend=*/false);

    ASSERT_TRUE(std::isfinite(stiff) && std::isfinite(none));
    EXPECT_LT(stiff, 0.2f * initial)
        << "stiff bend failed to flatten: initial=" << initial << " final=" << stiff;
    EXPECT_GT(none, 0.8f * initial)
        << "no-bend control should stay folded: initial=" << initial
        << " final=" << none;
}

TEST(XpbdBendCloth, DrapeIsFiniteNearInextensibleAndByteExact) {
    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";

    auto run = []() {
        GridCloth g = MakeGrid(4u, 4u, 0.25f, /*pin_first_row=*/false);
        g.particles.inv_masses[0] = 0.0f;
        g.particles.inv_masses[3] = 0.0f;

        ClothTopologyOptions opts;
        opts.distance_compliance_alpha = 0.0f;
        opts.bend_compliance_alpha = 1.0e-4f;
        XpbdConstraintSet cs;
        BuildClothConstraints(g.particles.positions, g.triangles, opts, cs);

        std::vector<std::pair<std::pair<uint32_t, uint32_t>, float>> rest_edges;
        for (const auto& dc : cs.distance) {
            rest_edges.push_back({{dc.particle_a, dc.particle_b}, dc.rest_length});
        }
        const XpbdState st = RunNkCloth(g.particles, cs, /*iters=*/20u,
                                             Vec3{0.0f, -9.81f, 0.3f}, 1.0f / 240.0f,
                                             /*kSteps=*/300u);
        return std::make_pair(st, rest_edges);
    };

    const auto a = run();
    const auto b = run();
    const XpbdState& sa = a.first;

    for (const Vec3& p : sa.positions) {
        ASSERT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z));
    }
    float max_drift = 0.0f;
    for (const auto& re : a.second) {
        const Vec3 d = sa.positions[re.first.first] - sa.positions[re.first.second];
        max_drift = std::max(max_drift, std::fabs(d.Length() - re.second) / re.second);
    }
    EXPECT_LT(max_drift, 0.05f) << "max relative edge-length drift " << max_drift;

    EXPECT_EQ(std::memcmp(sa.positions.data(), b.first.positions.data(),
                          sa.positions.size() * sizeof(Vec3)),
              0)
        << "bend forward position buffer not two-run byte-identical (D1 violation)";
    EXPECT_EQ(std::memcmp(sa.velocities.data(), b.first.velocities.data(),
                          sa.velocities.size() * sizeof(Vec3)),
              0)
        << "bend forward velocity buffer not two-run byte-identical (D1 violation)";
}
