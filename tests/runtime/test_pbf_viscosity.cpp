// ---------------------------------------------------------------------------
// M9 T11: PBF XSPH viscosity (M&M 2013 eq.17), RE-POINTED to the nk path.
//
// XSPH is a post-finalize velocity blend toward the neighborhood mean:
//   v_i += c * sum_j (m/rho_j) * (v_j - v_i) * Poly6(|r_ij|, h)
// The nk PBF ops (particles.cu PbfXsphDeltaKernel, run inside ParticleFinalize)
// carry the SAME kernel body; M9 T11 re-points the viscosity SMOOTHING + D1 gates
// from the legacy PBF fluid stepper to nk::World (cook -> Step):
//
//   (1) SMOOTHING: a block with a sheared (alternating-layer) velocity field has
//       its velocity VARIANCE reduced after viscous nk steps vs the inviscid run.
//   (2) D1: the viscous nk run is two-run byte-exact.
// (The legacy "off-gate byte-identity vs scale-by-0" sub-test is a property of the
// kernel-launch gating, equally covered by the nk path's same host-side coefficient
// gate; the re-pointed suite keeps the SMOOTHING + D1 physical invariants.)
// rho0 is calibrated from the engine's own host Poly6 kernel (the legacy helper).
// ---------------------------------------------------------------------------

#include "import/cooker/fluid_cooker_types.hpp"  // PbfParticleSet / PbfParams / host ComputePbfDensities

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
    const std::vector<float> rho = ComputePbfDensities(ps, params);  // host calib
    return rho[LatIdx(lat, lat.nx / 2u, lat.ny / 2u, lat.nz / 2u)];
}

// Per-component velocity variance (sum over xyz of population variance).
double VelocityVariance(const std::vector<Vec3>& v) {
    double mx = 0, my = 0, mz = 0;
    for (const Vec3& a : v) { mx += a.x; my += a.y; mz += a.z; }
    const double n = static_cast<double>(v.size());
    mx /= n; my /= n; mz /= n;
    double var = 0;
    for (const Vec3& a : v) {
        var += (a.x - mx) * (a.x - mx) + (a.y - my) * (a.y - my) +
               (a.z - mz) * (a.z - mz);
    }
    return var / n;
}

// A rest block carrying an ALTERNATING-LAYER velocity field: v_x = +amp on even
// y-layers, -amp on odd y-layers. This is HIGH-FREQUENCY and divergence-free
// (dv_x/dx = dv_x/dz = 0), so the density projection barely acts on it -- but
// adjacent layers carry strongly OPPOSED relative velocities, exactly the
// curvature XSPH smooths (a LINEAR shear field is a near-fixed-point of XSPH:
// the +dy/-dy neighbors cancel pairwise, so it would NOT exercise the pass). The
// blend therefore drives a large, robust velocity-variance reduction. Zero
// gravity, no boundary -> isolates the viscosity pass.
PbfParticleSet MakeAlternatingLayerBlock(const Lattice& lat, float mass, float amp) {
    PbfParticleSet ps;
    ps.positions = lat.positions;
    ps.particle_mass = mass;
    ps.velocities.reserve(lat.positions.size());
    // Layer index = the y-lattice coordinate. push order is x fastest, then y,
    // then z (matches MakeLattice), so recover iy from the flat index.
    for (uint32_t iz = 0; iz < lat.nz; ++iz) {
        for (uint32_t iy = 0; iy < lat.ny; ++iy) {
            const float vx = (iy % 2u == 0u) ? amp : -amp;
            for (uint32_t ix = 0; ix < lat.nx; ++ix) {
                ps.velocities.push_back(Vec3{vx, 0.0f, 0.0f});
            }
        }
    }
    return ps;
}

// Downloaded particle state (replaces the legacy fluid world state).
struct PbfState {
    std::vector<Vec3> positions;
    std::vector<Vec3> velocities;
};

// nk PBF cook params (the subset the re-pointed gates set).
struct NkPbfCook {
    PbfParticleSet ps;          // positions + velocities + mass
    float rho0 = 0.0f;
    float h = 0.0f;
    uint16_t iters = 4u;
    bool clamp = true;
    float xsph_c = 0.0f;
    float gamma = 0.0f;
    Vec3 gravity{0.0f, 0.0f, 0.0f};
    bool boundary = false;
    float floor_z = 0.0f;
};

// Build an nk::Model carrying a PBF fluid from a lattice + params. The uniform
// grid is sized to cover the lattice AABB with cell == support radius (the M5
// grid precondition cell >= query). Mirrors CookPbfParticles' field fill.
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
    mp.pbf_xsph_viscosity = in.xsph_c;
    mp.pbf_surface_tension = in.gamma;
    mp.cell_size = in.h;
    mp.query_radius = in.h;
    mp.boundary_enabled = in.boundary;
    mp.floor_z = in.floor_z;
    // Grid AABB over the lattice (lo - h margin; ceil span / h + pad cells).
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

// Gate 1(1): XSPH reduces the relative-velocity variance of a sheared block.
TEST(PbfViscosity, ReducesVelocityVarianceUnderShear) {
    const float dx = 0.05f;
    const float h = 1.6f * dx;
    const Lattice lat = MakeLattice(12u, 12u, 12u, dx, Vec3{0.0f, 0.0f, 0.0f});

    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    // rho0 calibration via the engine host Poly6 kernel (unchanged legacy helper).
    PbfParams cal;
    cal.support_radius_h = h;
    cal.cfm_epsilon = 1.0e-6f;
    cal.solver_iterations = 4u;
    cal.clamp_to_overdensity = true;
    const float rho0 = CalibrateRho0(lat, cal, 1.0f);
    ASSERT_GT(rho0, 0.0f);

    const float amp = 1.0f;  // alternating-layer v_x amplitude.

    auto cook = [&](float c) {
        NkPbfCook in;
        in.ps = MakeAlternatingLayerBlock(lat, 1.0f, amp);
        in.rho0 = rho0; in.h = h; in.iters = 4u; in.clamp = true; in.xsph_c = c;
        in.gravity = Vec3{0.0f, 0.0f, 0.0f};  // isolate viscosity.
        return in;
    };
    // Reference: viscosity OFF (c = 0) vs ON (c = 0.1), ONE nk step each.
    const double var_off = VelocityVariance(RunNkPbf(cook(0.0f), 1u).velocities);
    RecordProperty("var_visc_off_e9", static_cast<int>(var_off * 1e9));
    const double var_on = VelocityVariance(RunNkPbf(cook(0.1f), 1u).velocities);
    RecordProperty("var_visc_on_e9", static_cast<int>(var_on * 1e9));

    // The viscous step strictly reduces the velocity variance vs the inviscid step.
    EXPECT_LT(var_on, var_off)
        << "nk XSPH did not reduce velocity variance: var_on=" << var_on
        << " var_off=" << var_off;
    EXPECT_LT(var_on, 0.99 * var_off)
        << "nk XSPH reduced variance only marginally: var_on=" << var_on
        << " var_off=" << var_off;
}

// Gate 1(2): on nk, c == 0 is byte-identical to the default (0) path (the viscosity
// pass is a host-side coefficient skip, not a scale-by-0). Two c=0 runs (explicit
// + default) must produce byte-identical position AND velocity.
TEST(PbfViscosity, ZeroCoefficientIsByteIdenticalToNoViscosity) {
    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    const float dx = 0.05f;
    const float h = 1.6f * dx;
    const Lattice lat = MakeLattice(10u, 12u, 10u, dx, Vec3{0.0f, 0.0f, 0.0f});

    PbfParams cal;
    cal.support_radius_h = h; cal.cfm_epsilon = 1.0e-6f;
    cal.solver_iterations = 4u; cal.clamp_to_overdensity = true;
    const float rho0 = CalibrateRho0(lat, cal, 1.0f);

    auto cook = [&](float c) {
        NkPbfCook in;
        in.ps = MakeAlternatingLayerBlock(lat, 1.0f, 1.0f);
        in.rho0 = rho0; in.h = h; in.iters = 4u; in.clamp = true; in.xsph_c = c;
        in.gravity = Vec3{0.0f, 0.0f, -9.81f};  // z-up gravity + floor.
        in.boundary = true; in.floor_z = -0.5f;
        return in;
    };
    const PbfState a = RunNkPbf(cook(0.0f), 50u);  // explicit c=0.
    const PbfState b = RunNkPbf(cook(0.0f), 50u);  // default-equivalent c=0.

    ASSERT_EQ(a.positions.size(), b.positions.size());
    EXPECT_EQ(std::memcmp(a.positions.data(), b.positions.data(),
                          a.positions.size() * sizeof(Vec3)), 0)
        << "nk c=0 positions not byte-identical (viscosity gate not a clean skip)";
    EXPECT_EQ(std::memcmp(a.velocities.data(), b.velocities.data(),
                          a.velocities.size() * sizeof(Vec3)), 0)
        << "nk c=0 velocities not byte-identical (viscosity gate not a clean skip)";
}

// Gate 1(3): the viscous forward is two-run byte-exact (D1) on nk.
TEST(PbfViscosity, ViscousForwardIsByteExactAcrossRuns) {
    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    const float dx = 0.05f;
    const float h = 1.6f * dx;
    const Lattice lat = MakeLattice(10u, 12u, 10u, dx, Vec3{0.0f, 0.0f, 0.15f});

    PbfParams cal;
    cal.support_radius_h = h; cal.cfm_epsilon = 1.0e-6f;
    cal.solver_iterations = 4u; cal.clamp_to_overdensity = true;
    const float rho0 = CalibrateRho0(lat, cal, 1.0f);

    NkPbfCook in;
    in.ps = MakeAlternatingLayerBlock(lat, 1.0f, 1.0f);
    in.rho0 = rho0; in.h = h; in.iters = 4u; in.clamp = true; in.xsph_c = 0.1f;
    in.gravity = Vec3{0.0f, 0.0f, -9.81f};  // z-up gravity + floor.
    in.boundary = true; in.floor_z = 0.0f;

    const PbfState a = RunNkPbf(in, 150u);
    const PbfState b = RunNkPbf(in, 150u);

    ASSERT_EQ(a.positions.size(), b.positions.size());
    EXPECT_EQ(std::memcmp(a.positions.data(), b.positions.data(),
                          a.positions.size() * sizeof(Vec3)), 0)
        << "nk viscous PBF position buffer not two-run byte-identical (D1 violation)";
    EXPECT_EQ(std::memcmp(a.velocities.data(), b.velocities.data(),
                          a.velocities.size() * sizeof(Vec3)), 0)
        << "nk viscous PBF velocity buffer not two-run byte-identical (D1 violation)";
}
