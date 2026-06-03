// ---------------------------------------------------------------------------
// v0.7 p10-A GATE 3: PBF volume conservation (one-sided incompressibility).
//
// Incompressibility <=> volume conservation, but the FAITHFUL test for an
// INVISCID free-surface fluid is ONE-SIDED. Two distinct scenes prove the two
// halves of the invariant:
//
//   (A) Bulk no-gravity rest block: with NO free-surface spreading, the interior
//       bulk volume V = N*m / rho_mean must hold near its t0 value across many
//       steps (two-sided drift bounded). This is the clean "volume conserved"
//       number; reported as a %.
//
//   (B) Column under gravity: incompressibility means the fluid RESISTS
//       COMPRESSION -- the interior density must never EXCEED rho0 by more than a
//       tolerance (the density projection relieves any impact over-compression).
//       Free-surface UNDER-density from spreading is NOT a volume violation
//       (V = N*m/rho would falsely report "volume gain" from lost neighbors), so
//       this gate is deliberately one-sided. Particle count is invariant in both.
//
// Why not a geometric point-cloud volume: convex hull / occupied-cell counting is
// ill-defined for a free-surface fluid and a time sink; the density proxy is
// rigorous and well-defined. The cap guard (max_truncated_particle_count == 0)
// holds in both scenes.
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

std::vector<uint32_t> InteriorIndices(const Lattice& lat, uint32_t margin) {
    std::vector<uint32_t> out;
    for (uint32_t iz = margin; iz + margin < lat.nz; ++iz)
        for (uint32_t iy = margin; iy + margin < lat.ny; ++iy)
            for (uint32_t ix = margin; ix + margin < lat.nx; ++ix)
                out.push_back(LatIdx(lat, ix, iy, iz));
    return out;
}

PbfParticleSet MakeParticles(const Lattice& lat, float mass) {
    PbfParticleSet ps;
    ps.positions = lat.positions;
    ps.velocities.assign(lat.positions.size(), Vec3{0.0f, 0.0f, 0.0f});
    ps.particle_mass = mass;
    return ps;
}

float InteriorMeanDensity(const std::vector<float>& rho,
                          const std::vector<uint32_t>& interior) {
    double sum = 0.0;
    for (uint32_t i : interior) {
        sum += static_cast<double>(rho[i]);
    }
    return static_cast<float>(sum / static_cast<double>(interior.size()));
}

} // namespace

// Scene (A): bulk rest block, NO gravity, NO free surface dynamics -> the
// interior bulk volume V = N*m / rho_mean must hold near its t0 value. This is the
// clean two-sided "volume conserved" number. Particle count invariant throughout.
TEST(PbfVolumeConservation, BulkRestBlockHoldsVolume) {
    const float dx = 0.05f;
    const float h = 1.6f * dx;
    const Lattice lat = MakeLattice(14u, 14u, 14u, dx, Vec3{0.0f, 0.0f, 0.0f});

    PbfParams params;
    params.support_radius_h = h;
    params.cfm_epsilon = 1.0e-6f;
    params.solver_iterations = 5u;
    params.clamp_to_overdensity = true;

    // Calibrate rho0 from the deep-interior rest density.
    {
        PbfParticleSet cal = MakeParticles(lat, 1.0f);
        PbfWorld cal_world = UploadPbfWorld(cal);
        const std::vector<float> rho = ComputePbfDensities(cal_world, params);
        params.rest_density_rho0 =
            rho[LatIdx(lat, lat.nx / 2u, lat.ny / 2u, lat.nz / 2u)];
    }
    ASSERT_GT(params.rest_density_rho0, 0.0f);

    const std::vector<uint32_t> interior = InteriorIndices(lat, /*margin=*/3u);
    ASSERT_FALSE(interior.empty());
    const uint32_t initial_count = static_cast<uint32_t>(lat.positions.size());

    PbfParticleSet ps = MakeParticles(lat, 1.0f);
    PbfWorld world = UploadPbfWorld(ps);
    const float rho_mean_0 =
        InteriorMeanDensity(ComputePbfDensities(world, params), interior);

    PbfStepOptions options;
    options.gravity = Vec3{0.0f, 0.0f, 0.0f};  // bulk: no gravity, no free surface.
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;

    const uint32_t kSteps = 200u;
    float max_drift_pct = 0.0f;
    for (uint32_t s = 0; s < kSteps; ++s) {
        const auto report = StepPbfWorld(world, params, options);
        ASSERT_EQ(report.max_truncated_particle_count, 0u)
            << "neighbor cap truncated at step " << s;
        const PbfWorldState st = world.DownloadState();
        ASSERT_EQ(static_cast<uint32_t>(st.positions.size()), initial_count);
        if ((s + 1u) % 25u == 0u) {
            const float rho_mean =
                InteriorMeanDensity(ComputePbfDensities(world, params), interior);
            // V = N*m/rho_mean, so |dV/V| = |drho/rho|. Report density drift %.
            const float drift =
                std::fabs(rho_mean - rho_mean_0) / rho_mean_0 * 100.0f;
            max_drift_pct = std::max(max_drift_pct, drift);
        }
    }

    // Bulk volume conserved within tolerance (no free-surface spreading here).
    EXPECT_LT(max_drift_pct, 5.0f)
        << "bulk interior volume (density) drift exceeded tolerance: "
        << max_drift_pct << "% (rho_mean_0=" << rho_mean_0 << ")";
    RecordProperty("bulk_volume_drift_pct",
                   static_cast<int>(max_drift_pct * 1000.0f));
}

// Scene (B): column under gravity -> incompressibility is ONE-SIDED: the interior
// density must never EXCEED rho0 beyond a tolerance (the projection relieves any
// impact over-compression). Free-surface under-density from inviscid spreading is
// NOT a volume violation (so this gate does not bound the under-density). Particle
// count invariant. Reports the worst interior over-density seen.
TEST(PbfVolumeConservation, ColumnInteriorNeverOverCompresses) {
    const float dx = 0.05f;
    const float h = 1.6f * dx;
    const Lattice lat = MakeLattice(10u, 16u, 10u, dx, Vec3{0.0f, 0.15f, 0.0f});

    PbfParams params;
    params.support_radius_h = h;
    params.cfm_epsilon = 1.0e-6f;
    params.solver_iterations = 5u;
    params.clamp_to_overdensity = true;

    {
        PbfParticleSet cal = MakeParticles(lat, 1.0f);
        PbfWorld cal_world = UploadPbfWorld(cal);
        const std::vector<float> rho = ComputePbfDensities(cal_world, params);
        params.rest_density_rho0 =
            rho[LatIdx(lat, lat.nx / 2u, lat.ny / 2u, lat.nz / 2u)];
    }
    ASSERT_GT(params.rest_density_rho0, 0.0f);

    const std::vector<uint32_t> interior = InteriorIndices(lat, /*margin=*/2u);
    ASSERT_FALSE(interior.empty());
    const uint32_t initial_count = static_cast<uint32_t>(lat.positions.size());

    PbfParticleSet ps = MakeParticles(lat, 1.0f);
    PbfWorld world = UploadPbfWorld(ps);

    PbfStepOptions options;
    options.gravity = Vec3{0.0f, -9.81f, 0.0f};
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    options.boundary.enabled = true;
    options.boundary.floor_y = 0.0f;

    const uint32_t kSteps = 300u;
    float max_over_density_pct = 0.0f;  // worst interior (rho/rho0 - 1) > 0.
    for (uint32_t s = 0; s < kSteps; ++s) {
        const auto report = StepPbfWorld(world, params, options);
        ASSERT_EQ(report.max_truncated_particle_count, 0u)
            << "neighbor cap truncated at step " << s;
        const PbfWorldState st = world.DownloadState();
        ASSERT_EQ(static_cast<uint32_t>(st.positions.size()), initial_count);

        if ((s + 1u) % 20u == 0u) {
            const std::vector<float> rho = ComputePbfDensities(world, params);
            for (uint32_t i : interior) {
                const float over =
                    (rho[i] / params.rest_density_rho0 - 1.0f) * 100.0f;
                if (over > max_over_density_pct) {
                    max_over_density_pct = over;
                }
            }
        }
    }

    // Incompressibility: the interior never over-compresses beyond tolerance. The
    // density projection relieves the impact transient. (Under-density from the
    // inviscid free-surface spread is expected and intentionally NOT bounded.)
    EXPECT_LT(max_over_density_pct, 15.0f)
        << "interior over-compressed beyond tolerance: max over-density="
        << max_over_density_pct << "% (rho0=" << params.rest_density_rho0 << ")";
    RecordProperty("max_interior_over_density_pct",
                   static_cast<int>(max_over_density_pct * 1000.0f));
}
