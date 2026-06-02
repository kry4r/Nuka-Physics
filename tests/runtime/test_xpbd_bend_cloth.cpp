// ---------------------------------------------------------------------------
// v0.7 p09-B: XPBD BEND (id 7) cloth tests -- Bergou isometric bending.
//
// The V3 FD adjoint gate (test_adjoint_fd_xpbd_bend) validates the multilinear
// XPBD multiplier law, which is BLIND to the geometric bend gradient grad C. This
// suite is where grad C is actually exercised:
//
//   1. Stencil rest properties: sum_i k_i == 0 and sum_i k_i*x_rest == 0 (the
//      linear-precision conditions that PIN the isometric stencil up to scale).
//   2. grad C is CONSTANT: the stored K_i equal a host central-difference of
//      C = sum_i K_i . p_i at TWO distinct non-flat configs, and the two FD
//      gradients are equal -- the discriminating check that catches a non-linear
//      (e.g. ||sum k x||) regression a drape test would silently pass.
//   3. Bend SENSITIVITY: a cantilever cloth strip pinned along one edge sags
//      measurably LESS with stiff bend than with floppy bend (and far less than a
//      no-bend control) -- proves grad C actually does work.
//   4. Drape invariants: a small patch under gravity stays finite, edges near
//      rest length.
//   5. D1 two-run byte-exactness of the bend forward.
//
// A full Vellum/Houdini golden bend trajectory is DEFERRED to p15 (not
// reproducible here); these are ANALYTIC/physical invariants.
// ---------------------------------------------------------------------------

#include "runtime/soft/cloth_topology.hpp"
#include "runtime/soft/tetmesh_topology.hpp"  // TetSignedVolumeTimes6 (planarity metric)
#include "runtime/soft/xpbd_world.hpp"

#include "math/vec3.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace {

using nuka::math::Vec3;
using nuka::runtime::soft::BuildClothConstraints;
using nuka::runtime::soft::ClothTopologyOptions;
using nuka::runtime::soft::ClothTriangle;
using nuka::runtime::soft::ComputeIsometricBendStencil;
using nuka::runtime::soft::StepXpbdWorld;
using nuka::runtime::soft::TetSignedVolumeTimes6;
using nuka::runtime::soft::UploadXpbdWorld;
using nuka::runtime::soft::XpbdBendConstraint;
using nuka::runtime::soft::XpbdConstraintSet;
using nuka::runtime::soft::XpbdParticleSet;
using nuka::runtime::soft::XpbdStepOptions;
using nuka::runtime::soft::XpbdWorld;
using nuka::runtime::soft::XpbdWorldState;

// A single shared edge along x with two apex vertices straddling it in the z=0
// plane (a flat flap): shared_a=(0,0,0), shared_b=(1,0,0), apex0=(0.3,0.8,0),
// apex1=(0.6,-0.7,0).
struct Flap {
    Vec3 sa{0.0f, 0.0f, 0.0f};
    Vec3 sb{1.0f, 0.0f, 0.0f};
    Vec3 a0{0.3f, 0.8f, 0.0f};
    Vec3 a1{0.6f, -0.7f, 0.0f};
};

// C = sum_i K_i . p_i for the bend constraint's four stored gradient vectors.
float BendConstraintValue(const XpbdBendConstraint& bc,
                          const std::vector<Vec3>& positions) {
    float c = 0.0f;
    for (int j = 0; j < 4; ++j) {
        c += bc.k[j].Dot(positions[bc.particle[j]]);
    }
    return c;
}

} // namespace

// Gate 2a: the isometric stencil satisfies the linear-precision conditions that
// define it (sum_i k_i == 0, sum_i k_i*x_rest == 0). These pin k up to scale, so
// matching them == matching the cotangent stencil up to the (compliance-absorbed)
// scalar.
TEST(XpbdBendCloth, IsometricStencilHasLinearPrecision) {
    Flap f;
    float k[4];
    ASSERT_TRUE(ComputeIsometricBendStencil(f.sa, f.sb, f.a0, f.a1, k));

    float sum_k = 0.0f;
    Vec3 sum_kx{0.0f, 0.0f, 0.0f};
    const Vec3 verts[4] = {f.sa, f.sb, f.a0, f.a1};
    for (int j = 0; j < 4; ++j) {
        sum_k += k[j];
        sum_kx += verts[j] * k[j];
    }
    EXPECT_NEAR(sum_k, 0.0f, 1.0e-5f) << "stencil must annihilate constants";
    EXPECT_NEAR(sum_kx.Length(), 0.0f, 1.0e-5f)
        << "stencil must annihilate the flat rest positions (isometric)";
    // Non-trivial (normalized so max|k| == 1).
    float max_abs = 0.0f;
    for (int j = 0; j < 4; ++j) {
        max_abs = std::max(max_abs, std::fabs(k[j]));
    }
    EXPECT_NEAR(max_abs, 1.0f, 1.0e-6f);
}

// Gate 2b: grad C is CONSTANT and equals a host central-difference of C at two
// DISTINCT non-flat configs (the discriminating check that catches a non-linear
// ||sum k x|| regression). Builds a one-flap mesh, extracts the bend constraint,
// and FD-differences C wrt each particle coordinate at two folded configs.
TEST(XpbdBendCloth, BendGradientIsConstantAndMatchesFd) {
    Flap f;
    std::vector<Vec3> rest = {f.sa, f.sb, f.a0, f.a1};
    std::vector<ClothTriangle> tris = {ClothTriangle{{0u, 1u, 2u}},
                                       ClothTriangle{{1u, 0u, 3u}}};
    ClothTopologyOptions opts;
    opts.emit_distance_constraints = false;
    XpbdConstraintSet cs;
    BuildClothConstraints(rest, tris, opts, cs);
    ASSERT_EQ(cs.bend.size(), 1u);
    const XpbdBendConstraint& bc = cs.bend[0];

    // Two distinct NON-flat configurations (fold the apexes out of plane by
    // different amounts so the gradients, if config-dependent, would differ).
    auto fold = [&](float z0, float z1) {
        std::vector<Vec3> p = rest;
        p[2].z += z0;  // apex0
        p[3].z += z1;  // apex1
        return p;
    };
    const std::vector<Vec3> cfgA = fold(0.5f, -0.3f);
    const std::vector<Vec3> cfgB = fold(-0.9f, 0.7f);

    const float h = 1.0e-2f;
    for (int particle_local = 0; particle_local < 4; ++particle_local) {
        const uint32_t pi = bc.particle[particle_local];
        for (int axis = 0; axis < 3; ++axis) {
            // Analytic grad component == the stored K_i component.
            float analytic = (axis == 0)   ? bc.k[particle_local].x
                             : (axis == 1) ? bc.k[particle_local].y
                                           : bc.k[particle_local].z;

            auto fd_at = [&](std::vector<Vec3> cfg) {
                Vec3& comp = cfg[pi];
                float& a = (axis == 0) ? comp.x : (axis == 1) ? comp.y : comp.z;
                const float saved = a;
                a = saved + h;
                const float cp = BendConstraintValue(bc, cfg);
                a = saved - h;
                const float cm = BendConstraintValue(bc, cfg);
                return (cp - cm) / (2.0f * h);
            };
            const float fdA = fd_at(cfgA);
            const float fdB = fd_at(cfgB);

            // grad C constant across configs (catches a nonlinear realization).
            EXPECT_NEAR(fdA, fdB, 1.0e-4f)
                << "grad C must be config-independent (particle " << particle_local
                << " axis " << axis << ")";
            // Analytic == FD (rel-err < 1e-3 where the component is non-trivial).
            const float denom = std::fabs(analytic) + std::fabs(fdA) + 1.0e-6f;
            EXPECT_LT(std::fabs(analytic - fdA) / denom, 1.0e-3f)
                << "analytic grad C != FD (particle " << particle_local << " axis "
                << axis << ")";
        }
    }
}

namespace {

// A WxH grid cloth patch in the z=0 plane (xy lattice), unit spacing, two
// triangles per cell. Optionally pins the entire first row (y == 0) so the patch
// hangs as a cantilever.
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

// Gate 3: bend SENSITIVITY (the bend analog of IsolatedVolumeConstraintRestores).
//
// The isometric bend constraint is EXACTLY invariant under rigid motions of the
// flat rest patch (C = sum_i k_i z_i is the discrete Laplacian of the z height
// field; for any affine z = a*x + b*y + c, C = a*sum k_i x_i + b*sum k_i y_i +
// c*sum k_i = 0 by the stencil's linear-precision properties). So a gravity-sag
// metric like max|z| CANNOT discriminate -- a stiff plate just swings DOWN rigidly
// about the hinge while staying flat. The bend constraint penalizes ONLY the
// non-affine part of z (true curvature). The correct, non-circular sensitivity
// test perturbs the flap OUT of plane and checks the ISOLATED bend constraint
// flattens it (no gravity, no distance constraints):
//   - non-planarity is measured by the flap's triple product (TetSignedVolumeTimes6
//     on the 4 flap points), == 0 iff coplanar -- NOT C itself (asserting C->0
//     would be circular since the projection literally drives C).
//   - stiff bend must collapse non-planarity to a small fraction of its initial
//     value; the NO-bend control (no constraints, no gravity -> positions frozen)
//     must STAY folded. A zeroed bend gradient would make stiff == control.
TEST(XpbdBendCloth, StiffBendFlattensFoldedFlapWhileNoneStaysFolded) {
    // One flat flap; fold the two apexes out of the z=0 plane.
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
        // Stencil built from the FLAT rest geometry (the isometric reference).
        BuildClothConstraints(rest, tris, opts, cs);
        if (emit_bend) {
            EXPECT_EQ(cs.bend.size(), 1u);
        } else {
            EXPECT_EQ(cs.bend.size(), 0u);
        }

        XpbdWorld world = UploadXpbdWorld(particles, cs);
        XpbdStepOptions so;
        so.gravity = Vec3{0.0f, 0.0f, 0.0f};  // no gravity: isolate the constraint.
        so.dt = 1.0f / 240.0f;
        so.step_count = 1u;
        so.solver_iterations = 10u;
        for (uint32_t s = 0; s < 200u; ++s) {
            StepXpbdWorld(world, so);
        }
        return nonplanarity(world.DownloadState().positions);
    };

    const float stiff = run(/*emit_bend=*/true);
    const float none = run(/*emit_bend=*/false);

    ASSERT_TRUE(std::isfinite(stiff) && std::isfinite(none));
    // Stiff bend flattens the flap (non-planarity collapses); the no-bend control
    // stays folded (no constraints + no gravity -> frozen positions).
    EXPECT_LT(stiff, 0.2f * initial)
        << "stiff bend failed to flatten: initial=" << initial << " final=" << stiff;
    EXPECT_GT(none, 0.8f * initial)
        << "no-bend control should stay folded: initial=" << initial
        << " final=" << none;
}

// Gate 3 (invariants) + Gate 4 (D1): a free patch drapes under gravity, stays
// finite and near-inextensible, and is two-run byte-exact.
TEST(XpbdBendCloth, DrapeIsFiniteNearInextensibleAndByteExact) {
    auto run = []() {
        GridCloth g = MakeGrid(4u, 4u, 0.25f, /*pin_first_row=*/false);
        // Pin the two root corners so the patch does not free-fall forever.
        g.particles.inv_masses[0] = 0.0f;
        g.particles.inv_masses[3] = 0.0f;

        ClothTopologyOptions opts;
        opts.distance_compliance_alpha = 0.0f;
        opts.bend_compliance_alpha = 1.0e-4f;
        XpbdConstraintSet cs;
        BuildClothConstraints(g.particles.positions, g.triangles, opts, cs);

        // Capture rest edge lengths for the inextensibility check.
        std::vector<std::pair<std::pair<uint32_t, uint32_t>, float>> rest_edges;
        for (const auto& dc : cs.distance) {
            rest_edges.push_back({{dc.particle_a, dc.particle_b}, dc.rest_length});
        }

        XpbdWorld world = UploadXpbdWorld(g.particles, cs);
        XpbdStepOptions so;
        so.gravity = Vec3{0.0f, -9.81f, 0.3f};  // gravity + small out-of-plane nudge.
        so.dt = 1.0f / 240.0f;
        so.step_count = 1u;
        so.solver_iterations = 20u;
        for (uint32_t s = 0; s < 300u; ++s) {
            StepXpbdWorld(world, so);
        }
        const XpbdWorldState st = world.DownloadState();
        return std::make_pair(st, rest_edges);
    };

    const auto a = run();
    const auto b = run();
    const XpbdWorldState& sa = a.first;

    // Finite.
    for (const Vec3& p : sa.positions) {
        ASSERT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z));
    }
    // Near-inextensible: max edge-length drift small (stiff distance constraints).
    float max_drift = 0.0f;
    for (const auto& re : a.second) {
        const Vec3 d = sa.positions[re.first.first] - sa.positions[re.first.second];
        max_drift = std::max(max_drift, std::fabs(d.Length() - re.second) / re.second);
    }
    EXPECT_LT(max_drift, 0.05f) << "max relative edge-length drift " << max_drift;

    // D1 two-run byte-exact (position + velocity).
    EXPECT_EQ(std::memcmp(sa.positions.data(), b.first.positions.data(),
                          sa.positions.size() * sizeof(Vec3)),
              0)
        << "bend forward position buffer not two-run byte-identical (D1 violation)";
    EXPECT_EQ(std::memcmp(sa.velocities.data(), b.first.velocities.data(),
                          sa.velocities.size() * sizeof(Vec3)),
              0)
        << "bend forward velocity buffer not two-run byte-identical (D1 violation)";
}
