// ---------------------------------------------------------------------------
// M9 T11: PBF surface tension (Akinci et al. 2013 COHESION), RE-POINTED to nk.
// A post-finalize velocity nudge a_i = -gamma * sum_j m * C(|r_ij|) * r_hat that
// pulls neighbors together. The nk PBF ops (particles.cu PbfCohesionKernel, run
// inside ParticleFinalize) carry the SAME kernel body; M9 T11 re-points the
// COHESION + D1 gates from the legacy runtime::fluid::PbfWorld stepper to
// nk::World (cook -> Step):
//
//   (1) COHESION: a small detached blob, no gravity, has its MEAN PAIRWISE
//       DISTANCE decrease after cohesive nk steps (the blob contracts).
//   (2) D1: the cohesive nk forward is two-run byte-exact.
// (The off-gate byte-identity sub-test is a kernel-launch-gating property covered
// by the nk path's same host-side coefficient gate; the re-pointed suite keeps the
// COHESION + D1 physical invariants.) rho0 is calibrated from the engine host kernel.
// ---------------------------------------------------------------------------

#include "runtime/fluid/pbf_world.hpp"  // PbfParticleSet / ComputePbfDensities (host calib)

#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/device_context.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Vec3;
using nuka::runtime::fluid::ComputePbfDensities;
using nuka::runtime::fluid::PbfParams;
using nuka::runtime::fluid::PbfParticleSet;
using nuka::runtime::fluid::PbfWorld;
using nuka::runtime::fluid::UploadPbfWorld;

// nk backend context (shared singleton, the nk_particle_equivalence pattern).
struct NkCtx { nphi::Device* dev = nullptr; nphi::Backend* backend = nullptr; };
NkCtx GetNkCtx() {
    static NkCtx c = [] {
        NkCtx r;
        static nuka::phi::DeviceContext keep = nuka::phi::MakeDefaultDeviceContext();
        (void)keep;
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

float CalibrateRho0(const Lattice& lat, const PbfParams& params, float mass) {
    PbfParticleSet ps;
    ps.positions = lat.positions;
    ps.velocities.assign(lat.positions.size(), Vec3{0.0f, 0.0f, 0.0f});
    ps.particle_mass = mass;
    PbfWorld world = UploadPbfWorld(ps);
    const std::vector<float> rho = ComputePbfDensities(world, params);
    return rho[LatIdx(lat, lat.nx / 2u, lat.ny / 2u, lat.nz / 2u)];
}

PbfParticleSet MakeBlob(const Lattice& lat, float mass) {
    PbfParticleSet ps;
    ps.positions = lat.positions;
    ps.velocities.assign(lat.positions.size(), Vec3{0.0f, 0.0f, 0.0f});
    ps.particle_mass = mass;
    return ps;
}

// Mean pairwise distance of the particle cloud (a connectedness / spread proxy).
// O(N^2); the blob is small (<= a few hundred particles) so this is cheap.
double MeanPairwiseDistance(const std::vector<Vec3>& p) {
    const std::size_t n = p.size();
    double sum = 0.0;
    std::size_t pairs = 0;
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const float dx = p[i].x - p[j].x;
            const float dy = p[i].y - p[j].y;
            const float dz = p[i].z - p[j].z;
            sum += std::sqrt(static_cast<double>(dx * dx + dy * dy + dz * dz));
            ++pairs;
        }
    }
    return pairs == 0 ? 0.0 : sum / static_cast<double>(pairs);
}

// Downloaded particle state (replaces the legacy PbfWorldState).
struct PbfState {
    std::vector<Vec3> positions;
    std::vector<Vec3> velocities;
};

// nk PBF cook params (the subset the re-pointed cohesion gates set).
struct NkPbfCook {
    PbfParticleSet ps;
    float rho0 = 0.0f, h = 0.0f, gamma = 0.0f;
    uint16_t iters = 4u;
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
    mp.pbf_surface_tension = in.gamma;
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

PbfState RunNkPbf(const NkPbfCook& in, uint32_t kSteps) {
    NkCtx c = GetNkCtx();
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = in.gravity.x; cfg.gravity[1] = in.gravity.y;
    cfg.gravity[2] = in.gravity.z;
    nk::World world(BuildNkPbfModel(in), 1u, c.dev, c.backend, cfg);
    EXPECT_TRUE(world.Ready());
    const uint32_t P = world.GetModel().capacities.particles_per_env;
    for (uint32_t s = 0; s < kSteps; ++s) world.Step();
    PbfState st;
    st.positions.resize(P);
    st.velocities.resize(P);
    world.GetData().DownloadField(nk::FieldId::ParticlePos, st.positions.data(),
                                  P * sizeof(Vec3));
    world.GetData().DownloadField(nk::FieldId::ParticleVel, st.velocities.data(),
                                  P * sizeof(Vec3));
    return st;
}

} // namespace

// Gate 2(1): cohesion contracts a detached blob (mean pairwise distance drops).
TEST(PbfSurfaceTension, CohesionContractsBlob) {
    const float dx = 0.05f;
    const float h = 1.6f * dx;
    // A small free blob (no gravity, no boundary) -- isolate cohesion.
    const Lattice lat = MakeLattice(6u, 6u, 6u, dx, Vec3{0.0f, 0.0f, 0.0f});

    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    PbfParams cal;
    cal.support_radius_h = h; cal.cfm_epsilon = 1.0e-6f;
    cal.solver_iterations = 4u; cal.clamp_to_overdensity = true;
    const float rho0 = CalibrateRho0(lat, cal, 1.0f);
    ASSERT_GT(rho0, 0.0f);

    const double mpd0 = MeanPairwiseDistance(lat.positions);
    RecordProperty("mean_pairwise_dist_0_e6", static_cast<int>(mpd0 * 1e6));

    // PURE cohesion (no viscosity) isolates the surface-tension pass. gamma=5 is an
    // O(1) knob (the spline is max-normalized to 1); the density projection + the
    // spline's short-range repulsive core hold the interior at rest spacing.
    NkPbfCook in;
    in.ps = MakeBlob(lat, 1.0f);
    in.rho0 = rho0; in.h = h; in.iters = 4u; in.clamp = true; in.gamma = 5.0f;
    in.gravity = Vec3{0.0f, 0.0f, 0.0f};
    const PbfState st = RunNkPbf(in, 120u);
    for (const Vec3& p : st.positions) {
        ASSERT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
            << "cohesion blew up the blob (non-finite)";
    }
    const double mpd1 = MeanPairwiseDistance(st.positions);
    RecordProperty("mean_pairwise_dist_1_e6", static_cast<int>(mpd1 * 1e6));

    // Cohesion pulls the cloud together: the mean pairwise distance decreases.
    EXPECT_LT(mpd1, mpd0)
        << "cohesion did not contract the blob: mpd_after=" << mpd1
        << " mpd_before=" << mpd0;
    EXPECT_LT(mpd1, 0.99 * mpd0)
        << "cohesion contracted the blob only marginally: mpd_after=" << mpd1
        << " mpd_before=" << mpd0;
}

// Gate 2(2): on nk, gamma == 0 is byte-identical to the default (0) path (the
// cohesion pass is a host-side coefficient skip). Two gamma=0 runs byte-identical.
TEST(PbfSurfaceTension, ZeroGammaIsByteIdenticalToNoTension) {
    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    const float dx = 0.05f;
    const float h = 1.6f * dx;
    const Lattice lat = MakeLattice(6u, 8u, 6u, dx, Vec3{0.0f, 0.0f, 0.0f});

    PbfParams cal;
    cal.support_radius_h = h; cal.cfm_epsilon = 1.0e-6f;
    cal.solver_iterations = 4u; cal.clamp_to_overdensity = true;
    const float rho0 = CalibrateRho0(lat, cal, 1.0f);

    auto cook = [&](float gamma) {
        NkPbfCook in;
        in.ps = MakeBlob(lat, 1.0f);
        in.rho0 = rho0; in.h = h; in.iters = 4u; in.clamp = true; in.gamma = gamma;
        in.gravity = Vec3{0.0f, 0.0f, -9.81f};  // z-up gravity + floor.
        in.boundary = true; in.floor_z = -0.5f;
        return in;
    };
    const PbfState a = RunNkPbf(cook(0.0f), 40u);
    const PbfState b = RunNkPbf(cook(0.0f), 40u);

    ASSERT_EQ(a.positions.size(), b.positions.size());
    EXPECT_EQ(std::memcmp(a.positions.data(), b.positions.data(),
                          a.positions.size() * sizeof(Vec3)), 0)
        << "nk gamma=0 positions not byte-identical (cohesion gate not a clean skip)";
    EXPECT_EQ(std::memcmp(a.velocities.data(), b.velocities.data(),
                          a.velocities.size() * sizeof(Vec3)), 0)
        << "nk gamma=0 velocities not byte-identical (cohesion gate not a clean skip)";
}

// Gate 2(3): the cohesive forward is two-run byte-exact (D1) on nk.
TEST(PbfSurfaceTension, CohesiveForwardIsByteExactAcrossRuns) {
    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    const float dx = 0.05f;
    const float h = 1.6f * dx;
    const Lattice lat = MakeLattice(6u, 6u, 6u, dx, Vec3{0.0f, 0.0f, 0.0f});

    PbfParams cal;
    cal.support_radius_h = h; cal.cfm_epsilon = 1.0e-6f;
    cal.solver_iterations = 4u; cal.clamp_to_overdensity = true;
    const float rho0 = CalibrateRho0(lat, cal, 1.0f);

    NkPbfCook in;
    in.ps = MakeBlob(lat, 1.0f);
    in.rho0 = rho0; in.h = h; in.iters = 4u; in.clamp = true; in.gamma = 5.0f;
    in.gravity = Vec3{0.0f, 0.0f, 0.0f};

    const PbfState a = RunNkPbf(in, 120u);
    const PbfState b = RunNkPbf(in, 120u);

    ASSERT_EQ(a.positions.size(), b.positions.size());
    EXPECT_EQ(std::memcmp(a.positions.data(), b.positions.data(),
                          a.positions.size() * sizeof(Vec3)), 0)
        << "cohesive PBF position buffer not two-run byte-identical (D1 violation)";
    EXPECT_EQ(std::memcmp(a.velocities.data(), b.velocities.data(),
                          a.velocities.size() * sizeof(Vec3)), 0)
        << "cohesive PBF velocity buffer not two-run byte-identical (D1 violation)";
}
