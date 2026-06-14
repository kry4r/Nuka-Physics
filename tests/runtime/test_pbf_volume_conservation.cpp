// ---------------------------------------------------------------------------
// v0.7 p10-A GATE 3: PBF volume conservation (one-sided incompressibility),
// M9 T11 Phase-2b RE-POINTED to nk (the legacy PBF fluid stepper is
// deleted; the PBF substep is op-ified onto the unified nk core). rho0 calibration
// + the per-step density readout use the CUDA-FREE host ComputePbfDensities
// (fluid_cooker_types.hpp); the forward sim runs on nk::World (cook -> Step). The
// gravity/floor axis is the nk z-axis (legacy used y); the invariants are
// axis-agnostic.
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
// rigorous and well-defined.
// ---------------------------------------------------------------------------

#include "import/cooker/fluid_cooker_types.hpp"  // PbfParticleSet / PbfParams / host ComputePbfDensities

#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/scoped_device_guard.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Vec3;
using nuka::runtime::fluid::ComputePbfDensities;
using nuka::runtime::fluid::PbfParams;
using nuka::runtime::fluid::PbfParticleSet;

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

// nk PBF cook params (the subset the re-pointed conservation gates set).
struct NkPbfCook {
    PbfParticleSet ps;
    float rho0 = 0.0f, h = 0.0f;
    uint16_t iters = 5u;
    bool clamp = true;
    Vec3 gravity{0.0f, 0.0f, 0.0f};
    bool boundary = false;
    float floor_z = 0.0f;
};

nk::Model BuildNkPbfModel(const NkPbfCook& in) {
    nk::Model model;
    nk::Model::ModelParticles& mp = model.particles;
    mp.mode = nk::Model::ParticleMode::Pbf;
    mp.initial_pos = in.ps.positions;
    mp.initial_vel = in.ps.velocities;
    if (mp.initial_vel.size() != mp.initial_pos.size())
        mp.initial_vel.assign(mp.initial_pos.size(), Vec3::Zero());
    const float im = in.ps.particle_mass > 0.0f ? 1.0f / in.ps.particle_mass : 0.0f;
    mp.inv_mass.assign(mp.initial_pos.size(), im);
    mp.pbf_rest_density = in.rho0;
    mp.pbf_support_radius = in.h;
    mp.pbf_particle_mass = in.ps.particle_mass;
    mp.pbf_iters = in.iters == 0u ? 1u : in.iters;
    mp.pbf_clamp_overdensity = in.clamp;
    mp.pbf_surface_tension = 0.0f;
    mp.cell_size = in.h;
    mp.query_radius = in.h;
    mp.boundary_enabled = in.boundary;
    mp.floor_z = in.floor_z;
    Vec3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
    for (const Vec3& p : in.ps.positions) {
        lo.x = std::min(lo.x, p.x); lo.y = std::min(lo.y, p.y); lo.z = std::min(lo.z, p.z);
        hi.x = std::max(hi.x, p.x); hi.y = std::max(hi.y, p.y); hi.z = std::max(hi.z, p.z);
    }
    mp.grid_min = Vec3{lo.x - in.h, lo.y - in.h, lo.z - in.h};
    auto dim = [&](float span) {
        return static_cast<uint32_t>(std::floor(span / in.h)) + 4u;
    };
    mp.grid_dims[0] = dim(hi.x - lo.x);
    mp.grid_dims[1] = dim(hi.y - lo.y);
    mp.grid_dims[2] = dim(hi.z - lo.z);
    model.capacities.env_count = 1;
    model.capacities.particles_per_env = static_cast<uint32_t>(mp.initial_pos.size());
    model.capacities.max_grid_cells =
        mp.grid_dims[0] * mp.grid_dims[1] * mp.grid_dims[2];
    return model;
}

struct NkPbfRunner {
    nk::World world;
    uint32_t P = 0u;
    NkPbfRunner(const NkPbfCook& in, float dt)
        : world([&] {
              NkCtx c = GetNkCtx();
              nk::Pipeline::SolverConfig cfg;
              cfg.dt = dt;
              cfg.gravity[0] = in.gravity.x; cfg.gravity[1] = in.gravity.y;
              cfg.gravity[2] = in.gravity.z;
              return nk::World(BuildNkPbfModel(in), 1u, c.dev, c.backend, cfg);
          }()) {
        P = world.GetModel().capacities.particles_per_env;
    }
    void Step() { world.Step(); }
    void DownloadPositions(std::vector<Vec3>& pos) {
        pos.resize(P);
        world.GetData().DownloadField(nk::FieldId::ParticlePos, pos.data(),
                                      P * sizeof(Vec3));
    }
};

// Host density-only pass on a downloaded config (the rho0-calibration recipe).
std::vector<float> HostDensities(const std::vector<Vec3>& pos, float mass,
                                 const PbfParams& params) {
    PbfParticleSet ps;
    ps.positions = pos;
    ps.velocities.assign(pos.size(), Vec3{0.0f, 0.0f, 0.0f});
    ps.particle_mass = mass;
    return ComputePbfDensities(ps, params);
}

} // namespace

// Scene (A): bulk rest block, NO gravity, NO free surface dynamics -> the
// interior bulk volume V = N*m / rho_mean must hold near its t0 value. This is the
// clean two-sided "volume conserved" number. Particle count invariant throughout.
TEST(PbfVolumeConservation, BulkRestBlockHoldsVolume) {
    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    const float dx = 0.05f;
    const float h = 1.6f * dx;
    const Lattice lat = MakeLattice(14u, 14u, 14u, dx, Vec3{0.0f, 0.0f, 0.0f});

    PbfParams params;
    params.support_radius_h = h;
    params.cfm_epsilon = 1.0e-6f;
    params.solver_iterations = 5u;
    params.clamp_to_overdensity = true;

    // Calibrate rho0 from the deep-interior rest density (host).
    {
        PbfParticleSet cal = MakeParticles(lat, 1.0f);
        const std::vector<float> rho = ComputePbfDensities(cal, params);
        params.rest_density_rho0 =
            rho[LatIdx(lat, lat.nx / 2u, lat.ny / 2u, lat.nz / 2u)];
    }
    ASSERT_GT(params.rest_density_rho0, 0.0f);

    const std::vector<uint32_t> interior = InteriorIndices(lat, /*margin=*/3u);
    ASSERT_FALSE(interior.empty());
    const uint32_t initial_count = static_cast<uint32_t>(lat.positions.size());

    NkPbfCook in;
    in.ps = MakeParticles(lat, 1.0f);
    in.rho0 = params.rest_density_rho0; in.h = h; in.iters = 5u; in.clamp = true;
    in.gravity = Vec3{0.0f, 0.0f, 0.0f};  // bulk: no gravity, no free surface.
    NkPbfRunner runner(in, /*dt=*/1.0f / 240.0f);

    const float rho_mean_0 =
        InteriorMeanDensity(ComputePbfDensities(in.ps, params), interior);

    const uint32_t kSteps = 200u;
    float max_drift_pct = 0.0f;
    std::vector<Vec3> pos;
    for (uint32_t s = 0; s < kSteps; ++s) {
        runner.Step();
        runner.DownloadPositions(pos);
        ASSERT_EQ(static_cast<uint32_t>(pos.size()), initial_count);
        if ((s + 1u) % 25u == 0u) {
            const float rho_mean =
                InteriorMeanDensity(HostDensities(pos, 1.0f, params), interior);
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
// count invariant. Reports the worst interior over-density seen. The gravity/floor
// axis is z (legacy used y).
TEST(PbfVolumeConservation, ColumnInteriorNeverOverCompresses) {
    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    const float dx = 0.05f;
    const float h = 1.6f * dx;
    // Tall axis (gravity axis) is z: nx=10, ny=10, nz=16.
    const Lattice lat = MakeLattice(10u, 10u, 16u, dx, Vec3{0.0f, 0.0f, 0.15f});

    PbfParams params;
    params.support_radius_h = h;
    params.cfm_epsilon = 1.0e-6f;
    params.solver_iterations = 5u;
    params.clamp_to_overdensity = true;

    {
        PbfParticleSet cal = MakeParticles(lat, 1.0f);
        const std::vector<float> rho = ComputePbfDensities(cal, params);
        params.rest_density_rho0 =
            rho[LatIdx(lat, lat.nx / 2u, lat.ny / 2u, lat.nz / 2u)];
    }
    ASSERT_GT(params.rest_density_rho0, 0.0f);

    const std::vector<uint32_t> interior = InteriorIndices(lat, /*margin=*/2u);
    ASSERT_FALSE(interior.empty());
    const uint32_t initial_count = static_cast<uint32_t>(lat.positions.size());

    NkPbfCook in;
    in.ps = MakeParticles(lat, 1.0f);
    in.rho0 = params.rest_density_rho0; in.h = h; in.iters = 5u; in.clamp = true;
    in.gravity = Vec3{0.0f, 0.0f, -9.81f};
    in.boundary = true; in.floor_z = 0.0f;
    NkPbfRunner runner(in, /*dt=*/1.0f / 240.0f);

    const uint32_t kSteps = 300u;
    float max_over_density_pct = 0.0f;  // worst interior (rho/rho0 - 1) > 0.
    std::vector<Vec3> pos;
    for (uint32_t s = 0; s < kSteps; ++s) {
        runner.Step();
        runner.DownloadPositions(pos);
        ASSERT_EQ(static_cast<uint32_t>(pos.size()), initial_count);

        if ((s + 1u) % 20u == 0u) {
            const std::vector<float> rho = HostDensities(pos, 1.0f, params);
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
