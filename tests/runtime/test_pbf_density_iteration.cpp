// ---------------------------------------------------------------------------
// v0.7 p10-A: Position-Based Fluids (Macklin & Mueller 2013) ANALYTIC / PHYSICAL
// invariants for the CORE forward density-projection fluid.
//
// These are NOT a Flex golden. The NVIDIA Flex "ball-into-water" paper oracle
// (exit-crit 3) is DEFERRED to p15 -- Flex is not reproducible in this
// environment. p10-A ships the analytic invariants below instead (and does NOT
// fabricate a Flex golden):
//
//   (a) rest-density block stays at rest      -- interior |C_i| small after a step
//   (b) over-compressed block relaxes         -- interior rho moves toward rho0
//   (c) single column under gravity           -- finite, stays above floor, settles
//   (e) D1 two-run byte-exact                 -- memcmp pos+vel over N steps
//
// (d) volume conservation lives in test_pbf_volume_conservation.cpp.
//
// rho0 CALIBRATION: rho0 is computed NUMERICALLY from this engine's own Poly6
// kernel over a deep-interior lattice particle (ComputePbfDensities). A wrong
// SPH normalization constant is therefore ABSORBED -- the PBF equilibrium is
// C_i = rho_i/rho0 - 1 == 0, which is independent of the kernel constant; only
// the convergence rate / physical units depend on it (units matter only for the
// deferred real-units Flex oracle, p15).
//
// FREE-SURFACE NOTE: a rest block is NOT uniform density -- surface particles
// have fewer neighbors -> rho_i < rho0 -> C_i < 0. So gate (a) evaluates |C_i|
// over INTERIOR particles only (calibrated to the interior); the solver also
// clamps the constraint to over-density (only C_i > 0 corrected), which kills
// spurious surface cohesion.
//
// CAP NOTE: the support radius h is sized so the rest neighbor count is well
// under the 32-cap; every fluid test asserts max_truncated_particle_count == 0
// (truncation keeps the lowest-INDEX neighbors, a spatially biased subset that
// would corrupt the density estimate). h ~ 1.6*dx gives ~20 neighbors.
// ---------------------------------------------------------------------------

#include "runtime/fluid/pbf_world.hpp"

#include "math/vec3.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace {

using nuka::math::Vec3;
using nuka::runtime::fluid::ComputePbfDensities;
using nuka::runtime::fluid::PbfBoundary;
using nuka::runtime::fluid::PbfParams;
using nuka::runtime::fluid::PbfParticleSet;
using nuka::runtime::fluid::PbfStepOptions;
using nuka::runtime::fluid::PbfStepReport;
using nuka::runtime::fluid::PbfWorld;
using nuka::runtime::fluid::PbfWorldState;
using nuka::runtime::fluid::StepPbfWorld;
using nuka::runtime::fluid::UploadPbfWorld;

// A regular cubic lattice of nx*ny*nz particles with spacing dx, lower corner at
// `origin`. The fluid spacing dx and support radius h are chosen so the rest
// neighbor count is comfortably under the 32-cap (h ~ 1.6*dx).
struct Lattice {
    std::vector<Vec3> positions;
    uint32_t nx = 0, ny = 0, nz = 0;
    float dx = 0.0f;
};

Lattice MakeLattice(uint32_t nx, uint32_t ny, uint32_t nz, float dx, Vec3 origin) {
    Lattice lat;
    lat.nx = nx;
    lat.ny = ny;
    lat.nz = nz;
    lat.dx = dx;
    lat.positions.reserve(static_cast<std::size_t>(nx) * ny * nz);
    for (uint32_t iz = 0; iz < nz; ++iz) {
        for (uint32_t iy = 0; iy < ny; ++iy) {
            for (uint32_t ix = 0; ix < nx; ++ix) {
                lat.positions.push_back(
                    Vec3{origin.x + static_cast<float>(ix) * dx,
                         origin.y + static_cast<float>(iy) * dx,
                         origin.z + static_cast<float>(iz) * dx});
            }
        }
    }
    return lat;
}

// Linear index into the lattice (matches MakeLattice's push order: x fastest).
uint32_t LatIdx(const Lattice& lat, uint32_t ix, uint32_t iy, uint32_t iz) {
    return (iz * lat.ny + iy) * lat.nx + ix;
}

// Indices of "interior" particles: at least `margin` cells from every boundary in
// each axis (so they have a full neighbor shell and a meaningful density).
std::vector<uint32_t> InteriorIndices(const Lattice& lat, uint32_t margin) {
    std::vector<uint32_t> out;
    for (uint32_t iz = margin; iz + margin < lat.nz; ++iz) {
        for (uint32_t iy = margin; iy + margin < lat.ny; ++iy) {
            for (uint32_t ix = margin; ix + margin < lat.nx; ++ix) {
                out.push_back(LatIdx(lat, ix, iy, iz));
            }
        }
    }
    return out;
}

PbfParticleSet MakeRestParticles(const Lattice& lat, float mass) {
    PbfParticleSet ps;
    ps.positions = lat.positions;
    ps.velocities.assign(lat.positions.size(), Vec3{0.0f, 0.0f, 0.0f});
    ps.particle_mass = mass;
    return ps;
}

// Calibrate rho0 = the density at a deep-interior lattice particle, measured with
// the engine's OWN Poly6 kernel. Returns (rho0, interior-center index).
float CalibrateRho0(const Lattice& lat, const PbfParams& base_params, float mass,
                    uint32_t* out_center_idx) {
    PbfParticleSet ps = MakeRestParticles(lat, mass);
    PbfWorld world = UploadPbfWorld(ps);
    const std::vector<float> rho = ComputePbfDensities(world, base_params);
    const uint32_t cx = lat.nx / 2u;
    const uint32_t cy = lat.ny / 2u;
    const uint32_t cz = lat.nz / 2u;
    const uint32_t center = LatIdx(lat, cx, cy, cz);
    if (out_center_idx != nullptr) {
        *out_center_idx = center;
    }
    return rho[center];
}

} // namespace

// Gate 2a: a rest-density block stays near rest. After one PBF step the INTERIOR
// constraint residual |C_i| = |rho_i/rho0 - 1| is small (the block does not
// explode or contract; surface particles are excluded -- they are legitimately
// under-dense, and the over-density clamp keeps them from pulling inward).
TEST(PbfDensityIteration, RestDensityBlockStaysAtRest) {
    const float dx = 0.05f;
    const float h = 1.6f * dx;
    const Lattice lat = MakeLattice(12u, 12u, 12u, dx, Vec3{0.0f, 0.0f, 0.0f});

    PbfParams params;
    params.support_radius_h = h;
    params.cfm_epsilon = 1.0e-6f;
    params.solver_iterations = 4u;
    params.clamp_to_overdensity = true;

    uint32_t center = 0u;
    const float rho0 = CalibrateRho0(lat, params, /*mass=*/1.0f, &center);
    ASSERT_GT(rho0, 0.0f);
    params.rest_density_rho0 = rho0;

    PbfParticleSet ps = MakeRestParticles(lat, 1.0f);
    PbfWorld world = UploadPbfWorld(ps);

    PbfStepOptions options;
    options.gravity = Vec3{0.0f, 0.0f, 0.0f};  // pure rest test (no gravity).
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    const PbfStepReport report = StepPbfWorld(world, params, options);

    // CAP guard: a well-sized h must NOT truncate (truncation would bias density).
    EXPECT_EQ(report.max_truncated_particle_count, 0u)
        << "rest block truncated the neighbor cap; h too large for the 32-cap";

    const PbfWorldState st = world.DownloadState();
    const std::vector<uint32_t> interior = InteriorIndices(lat, /*margin=*/3u);
    ASSERT_FALSE(interior.empty());

    float max_abs_c = 0.0f;
    float max_disp = 0.0f;
    for (uint32_t i : interior) {
        const float c = st.densities[i] / rho0 - 1.0f;
        max_abs_c = std::max(max_abs_c, std::fabs(c));
        const Vec3 d = st.positions[i] - lat.positions[i];
        max_disp = std::max(max_disp, std::sqrt(d.Dot(d)));
    }
    // The interior is already near rho0 at rest (|C| ~ a few %); after projection
    // it stays small. Generous bound -- the point is "no blow-up / no contraction".
    EXPECT_LT(max_abs_c, 0.10f)
        << "interior rest residual too large: max|C_i|=" << max_abs_c
        << " rho0=" << rho0;
    EXPECT_LT(max_disp, 0.25f * dx)
        << "interior rest particles drifted: max_disp=" << max_disp
        << " (dx=" << dx << ")";
    for (uint32_t i : interior) {
        EXPECT_TRUE(std::isfinite(st.positions[i].x) &&
                    std::isfinite(st.positions[i].y) &&
                    std::isfinite(st.positions[i].z));
    }
}

// Gate 2b: an OVER-COMPRESSED block relaxes toward rest density. We calibrate
// rho0 on a rest lattice (spacing dx), then build a denser lattice (spacing
// 0.85*dx) so the interior is over-dense (rho > rho0, C > 0). The density
// PROJECTION must push the over-packed particles apart -- so after ONE PBF step
// the interior over-density DECREASES.
//
// INVISCID NOTE: p10-A has NO viscosity (XSPH is p10-B). An over-compressed
// inviscid blob legitimately OVERSHOOTS into expansion and oscillates -- it does
// not monotonically settle to rho0 without damping. So this gate tests the
// projection's EFFECT directly: (1) one step drives the over-density strictly
// down (the constraint does the right thing), and (2) the post-step state is
// finite and bounded (no explosion). The settle-to-rest behavior is validated
// once XSPH lands (p10-B). Reports the interior mean C before/after.
TEST(PbfDensityIteration, OverCompressedBlockRelaxesTowardRest) {
    const float dx = 0.05f;
    const float h = 1.6f * dx;

    PbfParams params;
    params.support_radius_h = h;
    params.cfm_epsilon = 1.0e-6f;
    params.solver_iterations = 5u;
    params.clamp_to_overdensity = true;

    // Calibrate rho0 on the REST lattice.
    const Lattice rest = MakeLattice(12u, 12u, 12u, dx, Vec3{0.0f, 0.0f, 0.0f});
    uint32_t rest_center = 0u;
    const float rho0 = CalibrateRho0(rest, params, /*mass=*/1.0f, &rest_center);
    ASSERT_GT(rho0, 0.0f);
    params.rest_density_rho0 = rho0;

    // Compressed lattice (tighter spacing -> over-dense interior).
    const float dx_c = 0.85f * dx;
    const Lattice comp = MakeLattice(12u, 12u, 12u, dx_c, Vec3{0.0f, 0.0f, 0.0f});
    const std::vector<uint32_t> interior = InteriorIndices(comp, /*margin=*/3u);
    ASSERT_FALSE(interior.empty());

    auto interior_mean_c = [&](const std::vector<float>& rho) {
        double sum = 0.0;
        for (uint32_t i : interior) {
            sum += static_cast<double>(rho[i] / rho0 - 1.0f);
        }
        return static_cast<float>(sum / static_cast<double>(interior.size()));
    };

    // C BEFORE relaxation (density-only pass on the compressed config).
    PbfParticleSet ps = MakeRestParticles(comp, 1.0f);
    PbfWorld world = UploadPbfWorld(ps);
    const float c_before = interior_mean_c(ComputePbfDensities(world, params));
    ASSERT_GT(c_before, 0.05f)
        << "compressed lattice not over-dense enough to test relaxation: C_before="
        << c_before;

    // ONE step of the density projection (no gravity: isolate the constraint).
    PbfStepOptions options;
    options.gravity = Vec3{0.0f, 0.0f, 0.0f};
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    const PbfStepReport report = StepPbfWorld(world, params, options);
    EXPECT_EQ(report.max_truncated_particle_count, 0u)
        << "compressed block truncated the neighbor cap";

    const float c_after = interior_mean_c(ComputePbfDensities(world, params));

    // (1) The projection drives the over-density strictly down toward rest.
    EXPECT_LT(c_after, c_before)
        << "interior density did not relax after one projection: C_before="
        << c_before << " C_after=" << c_after << " rho0=" << rho0;
    // The reduction is substantial (the projection is doing real work, not a wash).
    EXPECT_LT(c_after, 0.9f * c_before)
        << "projection moved the over-density too little: C_before=" << c_before
        << " C_after=" << c_after;
    // (2) Post-step state finite + bounded (no explosion).
    const PbfWorldState st = world.DownloadState();
    for (uint32_t i = 0; i < st.positions.size(); ++i) {
        ASSERT_TRUE(std::isfinite(st.positions[i].x) &&
                    std::isfinite(st.positions[i].y) &&
                    std::isfinite(st.positions[i].z))
            << "particle " << i << " not finite after projection";
    }
}

// Gate 2c: a single fluid column under gravity falls, hits the floor, and stays a
// stable finite fluid -- the spec invariant "no explosion, finite, particles stay
// above floor". Asserts, sampled across the WHOLE run: every position finite,
// every particle at/above the floor, particle count invariant, and the peak speed
// stays under an energy-scale bound (no energy injection / explosion).
//
// INVISCID NOTE: p10-A has NO viscosity / surface tension (deferred to p10-B). On
// a frictionless infinite floor an inviscid, cohesion-free fluid does NOT settle
// to a heap -- it spreads as a dam-break front at the analytic front speed
// ~2*sqrt(2*g*H0) and ENERGY-CONSERVES (the peak speed is constant, not growing,
// which is exactly the stability signature). So the gate checks the energy bound +
// the literal spec invariants, NOT a viscous settle-to-rest (that lands with XSPH
// in p10-B). A GROWING peak speed would mean the projection injects energy (a real
// instability) -- the bound below catches that.
TEST(PbfDensityIteration, ColumnUnderGravityStaysStableAndBounded) {
    const float dx = 0.05f;
    const float h = 1.6f * dx;
    const float floor_y = 0.0f;
    // Start the column a little above the floor so it falls then piles up.
    const Lattice lat =
        MakeLattice(8u, 16u, 8u, dx, Vec3{0.0f, 0.2f, 0.0f});

    PbfParams params;
    params.support_radius_h = h;
    params.cfm_epsilon = 1.0e-6f;
    params.solver_iterations = 4u;
    params.clamp_to_overdensity = true;

    uint32_t center = 0u;
    const float rho0 = CalibrateRho0(lat, params, /*mass=*/1.0f, &center);
    ASSERT_GT(rho0, 0.0f);
    params.rest_density_rho0 = rho0;

    PbfParticleSet ps = MakeRestParticles(lat, 1.0f);
    PbfWorld world = UploadPbfWorld(ps);

    PbfStepOptions options;
    options.gravity = Vec3{0.0f, -9.81f, 0.0f};
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    options.boundary.enabled = true;
    options.boundary.floor_y = floor_y;

    const uint32_t initial_count = static_cast<uint32_t>(lat.positions.size());
    // Energy-scale speed bound: characteristic v ~ sqrt(2*g*H0); the dam-break
    // front is ~2x that; k = 2.5 margin. H0 = column top above floor.
    const float h0 = 0.2f + static_cast<float>(lat.ny - 1u) * dx;  // top of column.
    const float v_bound = 2.5f * std::sqrt(2.0f * 9.81f * h0);

    const uint32_t kSteps = 600u;  // 2.5 s of sim time.
    float run_max_speed = 0.0f;
    for (uint32_t s = 0; s < kSteps; ++s) {
        const PbfStepReport report = StepPbfWorld(world, params, options);
        ASSERT_EQ(report.max_truncated_particle_count, 0u)
            << "column truncated the neighbor cap at step " << s;

        const PbfWorldState st = world.DownloadState();
        ASSERT_EQ(static_cast<uint32_t>(st.positions.size()), initial_count)
            << "particle count changed at step " << s;
        for (uint32_t i = 0; i < st.positions.size(); ++i) {
            const Vec3 p = st.positions[i];
            ASSERT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) &&
                        std::isfinite(p.z))
                << "particle " << i << " not finite (explosion) at step " << s;
            EXPECT_GE(p.y, floor_y - 1.0e-3f)
                << "particle " << i << " sank below floor at step " << s;
            const Vec3 v = st.velocities[i];
            run_max_speed = std::max(run_max_speed, std::sqrt(v.Dot(v)));
        }
    }
    // No energy injection: the peak speed stays under the dam-break energy scale.
    EXPECT_LT(run_max_speed, v_bound)
        << "peak speed exceeded the energy bound (explosion): run_max_speed="
        << run_max_speed << " v_bound=" << v_bound << " H0=" << h0;
}

// Gate (e) / GATE 4: D1 two-run byte-exactness of the PBF forward. Two independent
// uploads + identical step schedules must yield byte-identical position AND
// velocity buffers (memcmp on the raw float storage). The neighbor reductions run
// in the grid's fixed ascending-by-index order with NO float atomics, so the
// result is deterministic regardless of warp scheduling.
TEST(PbfDensityIteration, ForwardIsByteExactAcrossRuns) {
    const float dx = 0.05f;
    const float h = 1.6f * dx;
    const Lattice lat = MakeLattice(10u, 14u, 10u, dx, Vec3{0.0f, 0.15f, 0.0f});

    PbfParams params;
    params.support_radius_h = h;
    params.cfm_epsilon = 1.0e-6f;
    params.solver_iterations = 4u;
    params.clamp_to_overdensity = true;

    uint32_t center = 0u;
    params.rest_density_rho0 = CalibrateRho0(lat, params, 1.0f, &center);

    PbfStepOptions options;
    options.gravity = Vec3{0.0f, -9.81f, 0.0f};
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    options.boundary.enabled = true;
    options.boundary.floor_y = 0.0f;

    const uint32_t kSteps = 200u;
    auto run = [&]() {
        PbfParticleSet ps = MakeRestParticles(lat, 1.0f);
        PbfWorld world = UploadPbfWorld(ps);
        for (uint32_t s = 0; s < kSteps; ++s) {
            StepPbfWorld(world, params, options);
        }
        return world.DownloadState();
    };

    const PbfWorldState a = run();
    const PbfWorldState b = run();

    ASSERT_EQ(a.positions.size(), b.positions.size());
    ASSERT_EQ(a.velocities.size(), b.velocities.size());
    EXPECT_EQ(std::memcmp(a.positions.data(), b.positions.data(),
                          a.positions.size() * sizeof(Vec3)),
              0)
        << "PBF position buffer is not two-run byte-identical (D1 violation)";
    EXPECT_EQ(std::memcmp(a.velocities.data(), b.velocities.data(),
                          a.velocities.size() * sizeof(Vec3)),
              0)
        << "PBF velocity buffer is not two-run byte-identical (D1 violation)";
}

// Soft-world inert-when-empty sanity: an empty PbfWorld steps to a no-op (zero
// particles, zero kernel launches, zero grid builds) -- the building block of the
// cuda_world_stepper inert-when-empty contract verified at the world level in
// test_cuda_world_stepper.cpp.
TEST(PbfDensityIteration, EmptyWorldStepsToNoOp) {
    PbfParticleSet ps;  // no particles.
    PbfWorld world = UploadPbfWorld(ps);

    PbfParams params;
    params.support_radius_h = 0.1f;
    params.rest_density_rho0 = 1.0f;

    PbfStepOptions options;
    options.step_count = 5u;
    const PbfStepReport report = StepPbfWorld(world, params, options);

    EXPECT_EQ(report.particle_count, 0u);
    EXPECT_EQ(report.kernel_launch_count, 0u);
    EXPECT_EQ(report.grid_build_count, 0u);
    EXPECT_EQ(report.simulated_step_count, 0u);
}
