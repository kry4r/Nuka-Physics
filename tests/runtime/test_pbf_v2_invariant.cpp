// ---------------------------------------------------------------------------
// v0.7 p10-B GATE 4: PBF V2 particle-count invariant.
//
// V2 (engine self-check, no oracle) for the fluid subsystem: the PBF substep must
// NEITHER GAIN NOR LOSE particles across a step -- the particle count is exactly
// invariant for the whole run, with the FULL p10-B path active (XSPH viscosity +
// surface-tension cohesion both ON, gravity + a floor). The downloaded state's
// position/velocity buffer lengths and the StepPbfWorld report's particle_count
// all stay equal to the uploaded count; every particle stays finite.
//
// WHY a FOCUSED test, not the central V2 InvariantSampler: the v0.1 V2 sampler
// (src/core/diagnostics/invariants*) operates ONLY on the rigid/articulation
// runtime::PhysicsWorld (bodies + joints) -- its Invariant::ParticleCount enum
// entry is UNWIRED (no sampling code path reads a particle count). PBF is a
// FORWARD-ONLY + INTERNAL GPU subsystem (master plan SS3 Round 7): it is NOT
// plumbed into PhysicsWorld, so the sampler cannot see it without a cross-cutting
// world-integration change that is out of p10-B scope. The task's blessed escape
// hatch ("add a focused PBF count-conservation test if the harness isn't easily
// extensible") applies; this is that test. (When fluid is integrated into the
// unified world -- a later coupling phase -- the ParticleCount enum can be wired
// to PbfWorld::ParticleCount() and this invariant folded into the sampler.)
// ---------------------------------------------------------------------------

#include "runtime/fluid/pbf_world.hpp"

#include "math/vec3.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

namespace {

using nuka::math::Vec3;
using nuka::runtime::fluid::ComputePbfDensities;
using nuka::runtime::fluid::PbfParams;
using nuka::runtime::fluid::PbfParticleSet;
using nuka::runtime::fluid::PbfStepOptions;
using nuka::runtime::fluid::PbfStepReport;
using nuka::runtime::fluid::PbfWorld;
using nuka::runtime::fluid::PbfWorldState;
using nuka::runtime::fluid::StepPbfWorld;
using nuka::runtime::fluid::UploadPbfWorld;

struct Lattice {
    std::vector<Vec3> positions;
    uint32_t nx = 0, ny = 0, nz = 0;
    float dx = 0.0f;
};

Lattice MakeLattice(uint32_t nx, uint32_t ny, uint32_t nz, float dx, Vec3 origin) {
    Lattice lat;
    lat.nx = nx; lat.ny = ny; lat.nz = nz; lat.dx = dx;
    for (uint32_t iz = 0; iz < nz; ++iz)
        for (uint32_t iy = 0; iy < ny; ++iy)
            for (uint32_t ix = 0; ix < nx; ++ix)
                lat.positions.push_back(
                    Vec3{origin.x + static_cast<float>(ix) * dx,
                         origin.y + static_cast<float>(iy) * dx,
                         origin.z + static_cast<float>(iz) * dx});
    return lat;
}

uint32_t LatIdx(const Lattice& lat, uint32_t ix, uint32_t iy, uint32_t iz) {
    return (iz * lat.ny + iy) * lat.nx + ix;
}

} // namespace

// Gate 4: particle count conserved across a full run with viscosity + cohesion ON.
TEST(PbfV2Invariant, ParticleCountConservedWithPolishOn) {
    const float dx = 0.05f;
    const float h = 1.6f * dx;
    const Lattice lat = MakeLattice(10u, 14u, 10u, dx, Vec3{0.0f, 0.15f, 0.0f});

    PbfParams params;
    params.support_radius_h = h;
    params.cfm_epsilon = 1.0e-6f;
    params.solver_iterations = 4u;
    params.clamp_to_overdensity = true;
    // FULL p10-B path active for the invariant (the demonstrated-stable coeffs;
    // gamma is an O(1) knob since the cohesion spline is max-normalized to 1).
    params.xsph_viscosity_c = 0.1f;
    params.surface_tension_gamma = 5.0f;

    // Calibrate rho0 (polish coeffs do not affect the density-only calibration).
    {
        PbfParticleSet cal;
        cal.positions = lat.positions;
        cal.velocities.assign(lat.positions.size(), Vec3{0.0f, 0.0f, 0.0f});
        cal.particle_mass = 1.0f;
        PbfWorld cw = UploadPbfWorld(cal);
        const std::vector<float> rho = ComputePbfDensities(cw, params);
        params.rest_density_rho0 =
            rho[LatIdx(lat, lat.nx / 2u, lat.ny / 2u, lat.nz / 2u)];
    }
    ASSERT_GT(params.rest_density_rho0, 0.0f);

    PbfParticleSet ps;
    ps.positions = lat.positions;
    ps.velocities.assign(lat.positions.size(), Vec3{0.0f, 0.0f, 0.0f});
    ps.particle_mass = 1.0f;
    const uint32_t initial_count = static_cast<uint32_t>(ps.positions.size());

    PbfWorld world = UploadPbfWorld(ps);
    ASSERT_EQ(world.ParticleCount(), initial_count);

    PbfStepOptions options;
    options.gravity = Vec3{0.0f, -9.81f, 0.0f};
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    options.boundary.enabled = true;
    options.boundary.floor_y = 0.0f;

    const uint32_t kSteps = 300u;
    for (uint32_t s = 0; s < kSteps; ++s) {
        const PbfStepReport report = StepPbfWorld(world, params, options);
        // The step report's particle count is invariant.
        ASSERT_EQ(report.particle_count, initial_count)
            << "report.particle_count changed at step " << s;
        // The world's particle count is invariant.
        ASSERT_EQ(world.ParticleCount(), initial_count)
            << "world.ParticleCount() changed at step " << s;
        // The downloaded state buffers are length-invariant + finite.
        const PbfWorldState st = world.DownloadState();
        ASSERT_EQ(static_cast<uint32_t>(st.positions.size()), initial_count)
            << "position buffer length changed at step " << s;
        ASSERT_EQ(static_cast<uint32_t>(st.velocities.size()), initial_count)
            << "velocity buffer length changed at step " << s;
        for (uint32_t i = 0; i < st.positions.size(); ++i) {
            const Vec3 p = st.positions[i];
            ASSERT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) &&
                        std::isfinite(p.z))
                << "particle " << i << " not finite at step " << s
                << " (polish-on instability)";
        }
    }
}
