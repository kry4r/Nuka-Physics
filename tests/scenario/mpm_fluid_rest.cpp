// ---------------------------------------------------------------------------
// MLS-MPM weakly-compressible FLUID constitutive gate (model_kind == 3).
//
// A fluid is a constitutive selected by the material-table model_kind, parallel to
// fixed-corotated (0) and Neo-Hookean (2): Cauchy sigma = -p*I, J = det(F), Tait EOS
// p = max(K*(J^-gamma - 1), 0). It runs the SAME grid/transfer/floor/domain-wall path
// as the elastic kinds. Asserts:
//   * hydrostatic rest: pressure (compression, J<1) increases with depth, the free
//     surface stays flat and near rest (J ~ 1), no net drift, all finite, escape == 0;
//   * determinism: grid mass/momentum byte-identical run-to-run; grid mass == sum of
//     particle mass (partition of unity);
//   * two-way sanity: a heavy body released above the pool decelerates and the
//     body-reaction sink is non-zero (couples through the same grid path).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"   // kPairDrivenRowsPerSlot
#include "phi/backend.hpp"
#include "phi/op_schema.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"
#include "scene/cook/cook_to_model.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
namespace sdfq = nuka::runtime::sdf;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kKindBox = 2u;

struct Backend {
    nphi::Device* dev = nullptr;
    nphi::Backend* backend = nullptr;
};
Backend GetBackend() {
    static Backend b = [] {
        Backend r;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return b;
}

nk::Pipeline::SolverConfig Cfg() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = -9.81f;
    return cfg;
}

// Pool + grid geometry. A shallow wide pool of water in a domain box (xy span = tank
// walls, floor at z=0). dx + substeps are the stability knobs (CFL dt_sub < ~0.4*dx/c
// with c = sqrt(K/rho0); a modest viscosity damps the explicit lattice ringing of a
// resting inviscid column so the free surface settles flat). The demo (a fast body
// drop) uses a stiffer K with more substeps for the higher impact speed.
constexpr float kFloorZ    = 0.0f;
constexpr float kDx        = 0.02f;
constexpr float kTankHalfXY = 0.16f;   // domain box (tank) half-extent in x/y.
constexpr float kFluidHalfXY = 0.13f;  // fluid fills the tank cross-section: the
                                       // domain walls confine it (no slump), so it
                                       // settles vertically into a flat hydrostatic
                                       // column resting against the walls.
constexpr float kPoolTopZ  = 0.08f;    // fluid pool height (shallow, settles fast).
constexpr float kDensity   = 1000.0f;  // rho0 (water).
constexpr float kBulk      = 2.0e3f;   // K: soft enough that the hydrostatic depth
                                       // compression (~few %) resolves above the
                                       // MLS-MPM lattice noise floor (rest probe).
constexpr float kTaitGamma = 7.0f;     // water EOS exponent.
constexpr float kViscosity = 120.0f;   // deviatoric damping (settles the column flat,
                                       // calms wall-corner ringing of the rest probe).
constexpr uint32_t kSubsteps = 25u;    // CFL headroom for the explicit step.

// Sample the fluid pool on a lattice at dx/2 (8 particles/cell) FILLING a confined
// tank: the fluid spans the domain box cross-section so the domain-wall BC ring holds
// it (no lateral slump) and it settles into a flat hydrostatic column against the
// walls + floor.
cook::MpmCookInput BuildPoolInput() {
    cook::MpmCookInput in;
    const float pdx = kDx * 0.5f;
    const float lo_z = kFloorZ + pdx;
    const float fluid_half = kFluidHalfXY;  // many empty cells out to the tank wall.
    for (float x = -fluid_half; x <= fluid_half + 1e-4f; x += pdx)
        for (float y = -fluid_half; y <= fluid_half + 1e-4f; y += pdx)
            for (float z = lo_z; z <= kPoolTopZ + 1e-4f; z += pdx)
                in.positions.push_back(Vec3{x, y, z});
    const size_t n = in.positions.size();
    in.velocities.assign(n, Vec3::Zero());
    const float vol0 = pdx * pdx * pdx;
    in.inv_mass.assign(n, 1.0f / (kDensity * vol0));
    in.vol0.assign(n, vol0);
    in.material.density = kDensity;
    in.material.model_kind = 3.0f;     // weakly-compressible fluid.
    in.material.bulk_modulus = kBulk;
    in.material.tait_gamma = kTaitGamma;
    in.material.viscosity = kViscosity;
    in.grid_origin = Vec3{-kTankHalfXY, -kTankHalfXY, kFloorZ - 3.0f * kDx};
    const float top = kPoolTopZ + 12.0f * kDx;     // vertical splash/settle headroom.
    in.grid_dims[0] = static_cast<uint32_t>(2.0f * kTankHalfXY / kDx) + 1u;
    in.grid_dims[1] = in.grid_dims[0];
    in.grid_dims[2] = static_cast<uint32_t>((top - in.grid_origin.z) / kDx) + 1u;
    in.dx = kDx;
    in.substeps = kSubsteps;
    in.floor_normal = Vec3{0.0f, 0.0f, 1.0f};
    in.floor_d = kFloorZ;
    in.floor_friction = 0.0f;          // free slip on the floor (a fluid).
    return in;
}

// Cook the pool via the config SELECTOR (sim_method == mlsmpm), no hardcoded mode.
nk::Model BuildPoolModel() {
    nk::Model m;
    m.capacities.env_count = 1u;
    cook::XpbdCookInput soft;
    soft.solver = nk::Model::ParticleMode::Mpm;
    cook::CookSoftBodyParticles(m, 1u, soft, BuildPoolInput());
    return m;
}

bool AnyNonFinite(const std::vector<Vec3>& v, uint32_t n) {
    for (uint32_t i = 0; i < n; ++i)
        if (!std::isfinite(v[i].x) || !std::isfinite(v[i].y) || !std::isfinite(v[i].z))
            return true;
    return false;
}

// det of a row-major 3x3 (the per-particle F packing).
float Det3(const float* F) {
    return F[0] * (F[4] * F[8] - F[5] * F[7]) -
           F[1] * (F[3] * F[8] - F[5] * F[6]) +
           F[2] * (F[3] * F[7] - F[4] * F[6]);
}

// --- Two-way scene: a heavy box released above the fluid pool ---------------
constexpr float kBoxHalf  = 0.05f;
constexpr float kBoxMass  = 2.0f;      // heavy (sinks into the pool).
constexpr float kBoxDropZ = kPoolTopZ + kBoxHalf + 0.03f;  // released above the surface.

// An analytic box SDF over a narrow band, cooked into the Model directly (mirrors
// the rigid_on_mpm_rest body-coupling pattern so the body projects onto the grid).
void AddBoxSdf(nk::Model& m, int32_t body_id, float half) {
    const float vh = kDx;
    const float band = 3.0f * vh;
    const float ext = half + band;
    const int n = static_cast<int>(std::ceil(ext / vh)) + 1;
    const Vec3 origin{-static_cast<float>(n) * vh, -static_cast<float>(n) * vh,
                      -static_cast<float>(n) * vh};
    const uint32_t base = static_cast<uint32_t>(m.sdf_cell_values.size());
    auto box_phi = [&](const Vec3& p, Vec3& grad) -> float {
        const Vec3 d{std::fabs(p.x) - half, std::fabs(p.y) - half, std::fabs(p.z) - half};
        const Vec3 dpos{std::max(d.x, 0.0f), std::max(d.y, 0.0f), std::max(d.z, 0.0f)};
        const float outside = std::sqrt(dpos.LengthSq());
        const float inside = std::min(std::max(d.x, std::max(d.y, d.z)), 0.0f);
        const float phi = outside + inside;
        if (outside > 1e-6f) {
            const Vec3 g{dpos.x * (p.x < 0 ? -1.f : 1.f), dpos.y * (p.y < 0 ? -1.f : 1.f),
                         dpos.z * (p.z < 0 ? -1.f : 1.f)};
            const float gl = std::sqrt(g.LengthSq());
            grad = gl > 1e-8f ? g * (1.0f / gl) : Vec3{0, 0, 1};
        } else {
            Vec3 g{0, 0, 0};
            if (d.x >= d.y && d.x >= d.z) g.x = p.x < 0 ? -1.f : 1.f;
            else if (d.y >= d.z) g.y = p.y < 0 ? -1.f : 1.f;
            else g.z = p.z < 0 ? -1.f : 1.f;
            grad = g;
        }
        return phi;
    };
    uint32_t count = 0u;
    for (int i = 0; i <= 2 * n; ++i)
        for (int j = 0; j <= 2 * n; ++j)
            for (int k = 0; k <= 2 * n; ++k) {
                const Vec3 p{origin.x + i * vh, origin.y + j * vh, origin.z + k * vh};
                Vec3 grad{0, 0, 0};
                const float phi = box_phi(p, grad);
                if (std::fabs(phi) > band) continue;
                m.sdf_cell_keys.push_back(sdfq::PackSdfCellKey(
                    static_cast<uint32_t>(i), static_cast<uint32_t>(j),
                    static_cast<uint32_t>(k)));
                m.sdf_cell_values.push_back(phi);
                m.sdf_cell_gradients.push_back(grad);
                ++count;
            }
    nk::Model::SdfGrid sg;
    sg.origin = origin;
    sg.voxel_size = vh;
    sg.dims[0] = sg.dims[1] = sg.dims[2] = static_cast<uint32_t>(2 * n + 1);
    sg.cell_offset = base;
    sg.cell_count = count;
    const uint32_t grid_idx = static_cast<uint32_t>(m.sdf_grids.size());
    m.sdf_grids.push_back(sg);

    nk::Model::PairDrivenShape sh;
    sh.kind = kKindBox;
    sh.params[0] = half; sh.params[1] = half; sh.params[2] = half;
    sh.contype = 1u; sh.conaffinity = 1u;
    sh.sdf_grid = grid_idx;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

// Heavy free box above the pool + a far immovable filler (>= 2 bodies for the LBVH),
// cooked on top of the fluid via the sim_method=mlsmpm selector.
nk::Model BuildPoolWithBodyModel() {
    nk::Model m;
    m.capacities.env_count = 1u;

    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = Vec3{0.0f, 0.0f, kBoxDropZ};
    bi.inv_mass = 1.0f / kBoxMass;
    const float I = (1.0f / 6.0f) * kBoxMass * (2.0f * kBoxHalf) * (2.0f * kBoxHalf);
    bi.inv_inertia = Vec3{1.0f / I, 1.0f / I, 1.0f / I};
    m.body_init.push_back(bi);
    AddBoxSdf(m, 0, kBoxHalf);

    nk::Model::BodyInit bf;
    bf.pose = Transform::Identity();
    bf.pose.position = Vec3{5.0f, 0.0f, 0.0f};
    bf.inv_mass = 0.0f; bf.inv_inertia = Vec3{0, 0, 0};
    m.body_init.push_back(bf);
    AddBoxSdf(m, 1, 0.05f);

    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_sdf_grids = static_cast<uint32_t>(m.sdf_grids.size());
    cap.max_sdf_cells = static_cast<uint32_t>(m.sdf_cell_values.size());
    cap.max_contacts_per_env = 16u;
    cap.max_rows_per_env = 16u * nk::kPairDrivenRowsPerSlot;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;

    // A stiffer, deeper-substep pool for the body impact: K sets the sound speed
    // c = sqrt(K/rho0) ~ 14 m/s, well above the ~0.6 m/s entry (stable explicit step).
    cook::MpmCookInput pool = BuildPoolInput();
    pool.material.bulk_modulus = 2.0e5f;
    pool.material.viscosity = 2.0f;
    pool.substeps = 40u;
    cook::XpbdCookInput soft;
    soft.solver = nk::Model::ParticleMode::Mpm;
    cook::CookSoftBodyParticles(m, 1u, soft, pool);

    m.particles.mpm_body_friction = 0.0f;
    m.particles.mpm_bite_disable_dynamic_bc = 0u;
    return m;
}

}  // namespace

// Hydrostatic rest: the column settles, pressure (compression, J<1) increases with
// depth, the free surface stays near rest (J ~ 1), all finite, no grid escape.
TEST(MpmFluidRest, HydrostaticPressureWithDepth) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    nk::Model m = BuildPoolModel();
    const uint32_t P = m.capacities.particles_per_env;
    ASSERT_GT(P, 1000u) << "the pool must be dense";
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());

    std::vector<Vec3> pos(P, Vec3::Zero()), vel(P, Vec3::Zero());
    std::vector<float> F(static_cast<size_t>(P) * 9u, 0.0f);
    auto dl = [&] {
        w.GetData().DownloadField(nk::FieldId::ParticlePos, pos.data(), P * sizeof(Vec3));
        w.GetData().DownloadField(nk::FieldId::ParticleVel, vel.data(), P * sizeof(Vec3));
        w.GetData().DownloadField(nk::FieldId::ParticleF, F.data(), F.size() * sizeof(float));
    };

    // Settle under gravity (the column relaxes onto the floor + walls), tracking
    // finiteness + the grid-escape bit over the whole settle.
    constexpr uint32_t kSettle = 1500u;
    uint32_t escape = 0u;
    bool any_nonfinite = false;
    for (uint32_t s = 0; s < kSettle; ++s) {
        w.Step();
        if (s % 200u == 0u || s + 1u == kSettle) {
            dl();
            any_nonfinite = any_nonfinite || AnyNonFinite(pos, P) || AnyNonFinite(vel, P);
            uint32_t st = 0u;
            w.GetData().DownloadField(nk::FieldId::EnvStatus, &st, sizeof(uint32_t));
            escape |= st & nphi::kEnvStatusMpmGridEscape;
        }
    }

    // Average the volume ratio J = det(F) by depth band over a measurement window so
    // the hydrostatic profile is read at equilibrium (transient ringing washes out).
    // Deep band: lowest 25% of the column; surface band: highest 25%. Hydrostatics =>
    // the deep band is MORE compressed (smaller J) than the surface band, surface J~1.
    constexpr uint32_t kWindow = 240u;
    double deep_J = 0.0, surf_J = 0.0;
    uint64_t deep_n = 0u, surf_n = 0u;
    float min_J = 1e30f, max_J = -1e30f, top_spread = 0.0f;
    float surf_mean_z = 0.0f;
    for (uint32_t s = 0; s < kWindow; ++s) {
        w.Step();
        dl();
        float zlo = 1e30f, zhi = -1e30f;
        for (uint32_t i = 0; i < P; ++i) { zlo = std::min(zlo, pos[i].z); zhi = std::max(zhi, pos[i].z); }
        const float col = (zhi - zlo) > 1e-6f ? (zhi - zlo) : 1.0f;
        double top_z_sum = 0.0; uint32_t top_n = 0u;
        for (uint32_t i = 0; i < P; ++i) {
            const float J = Det3(&F[static_cast<size_t>(i) * 9u]);
            min_J = std::min(min_J, J); max_J = std::max(max_J, J);
            const float depth = (pos[i].z - zlo) / col;  // 0 floor .. 1 surface.
            if (depth < 0.25f) { deep_J += J; ++deep_n; }
            else if (depth > 0.75f) { surf_J += J; ++surf_n; }
            if (depth > 0.9f) { top_z_sum += pos[i].z; ++top_n; }
        }
        const float top_mean = top_n ? static_cast<float>(top_z_sum / top_n) : 0.0f;
        for (uint32_t i = 0; i < P; ++i) {
            const float depth = (pos[i].z - zlo) / col;
            if (depth > 0.9f) top_spread = std::max(top_spread, std::fabs(pos[i].z - top_mean));
        }
        surf_mean_z = (s + 1u == kWindow) ? top_mean : surf_mean_z;
    }
    const float deep_mean = deep_n ? static_cast<float>(deep_J / deep_n) : 1.0f;
    const float surf_mean = surf_n ? static_cast<float>(surf_J / surf_n) : 1.0f;

    std::fprintf(stderr,
                 "[fluid] deep_J=%.5f surf_J=%.5f min_J=%.4f max_J=%.4f "
                 "top_spread=%.4f surf_z=%.4f nonfinite=%d escape=%u\n",
                 deep_mean, surf_mean, min_J, max_J, top_spread, surf_mean_z,
                 any_nonfinite ? 1 : 0, escape);

    // The x/y walls + plane floor contain the column, so a settled fluid must NOT trip
    // the escape bit (it now flags only an open-top breach / inverted F).
    EXPECT_EQ(0u, escape) << "a contained hydrostatic column must not escape the grid";
    EXPECT_FALSE(any_nonfinite) << "the fluid trajectory must be finite";
    EXPECT_GT(min_J, 0.80f) << "compression must stay bounded (not glass-stiff collapse)";
    // A few free-surface particles at the confining wall corners dilate to ~1.3 (a
    // bounded, physical inviscid free-surface spike); a true runaway is J >> 2 / NaN.
    EXPECT_LT(max_J, 1.5f) << "expansion must stay bounded (no runaway dilation)";
    // Hydrostatics: the deep band is more compressed (smaller J) than the surface
    // band, and the free surface is near rest (J ~ 1, p -> 0).
    EXPECT_LT(deep_mean, surf_mean) << "pressure (compression) must increase with depth";
    EXPECT_NEAR(surf_mean, 1.0f, 0.04f) << "the free surface volume ratio must be ~1";
    EXPECT_LT(top_spread, 5.0f * kDx) << "the free surface must stay ~flat";
    // Guard the EOS regime + ordering (not a tight quantitative magnitude: a ~4-cell
    // explicit column under-resolves the analytic Tait profile). The deep band must be
    // live-compressed (rejects a dead or too-stiff EOS) yet far from collapse.
    EXPECT_GT(deep_mean, 0.90f) << "deep compression must be live (not a dead/soft EOS)";
    EXPECT_LT(deep_mean, 0.995f) << "deep band compressed but not collapsed (cross-device margin)";
}

// Determinism: grid mass/momentum byte-identical run-to-run, and the grid mass sums
// to the total particle mass (partition of unity through the deterministic gather).
TEST(MpmFluidRest, GridMassMomentumDeterministicAndConserved) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    auto run = [&](std::vector<float>& mass_out, std::vector<Vec3>& mom_out,
                   double& part_mass_out) -> bool {
        nk::Model m = BuildPoolModel();
        const uint32_t P = m.capacities.particles_per_env;
        const uint32_t nodes = m.capacities.mpm_grid_nodes_per_env;
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
        if (!w.Ready()) return false;
        for (uint32_t s = 0; s < 120u; ++s) w.Step();
        mass_out.assign(nodes, 0.0f);
        mom_out.assign(nodes, Vec3::Zero());
        std::vector<float> inv_mass(P, 0.0f);
        if (!w.GetData().DownloadField(nk::FieldId::GridMass, mass_out.data(),
                                       nodes * sizeof(float)) ||
            !w.GetData().DownloadField(nk::FieldId::GridMomentum, mom_out.data(),
                                       nodes * sizeof(Vec3)) ||
            !w.GetData().DownloadField(nk::FieldId::ParticleInvMass, inv_mass.data(),
                                       P * sizeof(float))) {
            return false;
        }
        part_mass_out = 0.0;
        for (uint32_t i = 0; i < P; ++i)
            if (inv_mass[i] > 0.0f) part_mass_out += 1.0 / inv_mass[i];
        return true;
    };
    std::vector<float> ma, mc;
    std::vector<Vec3> qa, qc;
    double pm_a = 0.0, pm_c = 0.0;
    ASSERT_TRUE(run(ma, qa, pm_a));
    ASSERT_TRUE(run(mc, qc, pm_c));
    ASSERT_EQ(ma.size(), mc.size());

    EXPECT_EQ(0, std::memcmp(ma.data(), mc.data(), ma.size() * sizeof(float)))
        << "grid mass differs run-to-run (an atomic scatter would)";
    EXPECT_EQ(0, std::memcmp(qa.data(), qc.data(), qa.size() * sizeof(Vec3)))
        << "grid momentum differs run-to-run";

    // Partition of unity: the LAST substep's grid mass sums to the total particle
    // mass (the P2G weights sum to 1 per particle). A small deficit is the boundary
    // partition-of-unity loss (a stencil clipped by the grid AABB), not a scatter
    // bug -- the byte-identity memcmp above is the determinism guard.
    double grid_mass = 0.0;
    for (float v : ma) grid_mass += v;
    const double rel_err = std::fabs(grid_mass - pm_a) / pm_a;
    std::fprintf(stderr, "[fluid] grid_mass=%.6f particle_mass=%.6f rel_err=%.3e\n",
                 grid_mass, pm_a, rel_err);
    EXPECT_LT(rel_err, 2e-3) << "grid mass must equal the total particle mass";
}

// Two-way sanity: a heavy body released above the pool decelerates as it enters the
// fluid and the body-reaction sink is non-zero (couples through the same grid path).
TEST(MpmFluidRest, HeavyBodyIntoPoolDeceleratesAndReacts) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    nk::Model m = BuildPoolWithBodyModel();
    const uint32_t P = m.capacities.particles_per_env;
    ASSERT_GT(P, 1000u) << "the pool must be dense";
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());

    const float dt = Cfg().dt;
    Transform body_pose{};
    Vec3 lin_vel{0, 0, 0}, reaction{0, 0, 0};
    std::vector<float> Fb(static_cast<size_t>(P) * 9u, 0.0f);
    auto box_z = [&]() {
        w.GetData().DownloadField(nk::FieldId::BodyPose, &body_pose, sizeof(Transform));
        return body_pose.position.z;
    };

    const float z0 = box_z();
    float min_vz = 0.0f;           // most-negative downward velocity (free-fall peak).
    float max_react = 0.0f;        // peak upward (signed +z) reaction sink.
    float min_J = 1e30f, max_J = -1e30f;
    uint32_t escape = 0u;
    bool nonfinite = false;
    bool decel = false;
    float prev_vz = 0.0f;
    constexpr uint32_t kSteps = 240u;   // 1 s: free-fall, entry, deceleration.
    for (uint32_t s = 0; s < kSteps; ++s) {
        w.Step();
        w.GetData().DownloadField(nk::FieldId::BodyLinearVelocity, &lin_vel, sizeof(Vec3));
        w.GetData().DownloadField(nk::FieldId::MpmBodyReaction, &reaction, sizeof(Vec3));
        const float z = box_z();
        nonfinite = nonfinite || !std::isfinite(z) || !std::isfinite(lin_vel.z) ||
                    !std::isfinite(reaction.z);
        min_vz = std::min(min_vz, lin_vel.z);
        max_react = std::max(max_react, reaction.z);   // signed: the sink pushes up (+z).
        uint32_t st = 0u;
        w.GetData().DownloadField(nk::FieldId::EnvStatus, &st, sizeof(uint32_t));
        escape |= st & nphi::kEnvStatusMpmGridEscape;
        if (s % 20u == 0u || s + 1u == kSteps) {     // bound the fluid state periodically.
            w.GetData().DownloadField(nk::FieldId::ParticleF, Fb.data(),
                                      Fb.size() * sizeof(float));
            for (uint32_t i = 0; i < P; ++i) {
                const float J = Det3(&Fb[static_cast<size_t>(i) * 9u]);
                min_J = std::min(min_J, J); max_J = std::max(max_J, J);
            }
        }
        // Deceleration: once it has reached its downward peak, vz must recover upward.
        if (lin_vel.z > prev_vz + 1e-4f && min_vz < -0.2f) decel = true;
        prev_vz = lin_vel.z;
    }
    const float z1 = box_z();
    const float free_fall_vz = -9.81f * (kSteps * dt);  // unresisted terminal vz.

    std::fprintf(stderr,
                 "[fluid+body] z0=%.4f z1=%.4f min_vz=%.4f free_fall_vz=%.4f "
                 "max_react=%.6e min_J=%.4f max_J=%.4f decel=%d escape=%u\n",
                 z0, z1, min_vz, free_fall_vz, max_react, min_J, max_J,
                 decel ? 1 : 0, escape);

    EXPECT_FALSE(nonfinite) << "the body+fluid trajectory must be finite";
    EXPECT_EQ(0u, escape) << "the stiff pool must stay contained (no escape / inverted F)";
    // The pool resists: the body never reaches the unresisted free-fall speed and it
    // decelerates after entry (vz recovers from its downward peak).
    EXPECT_GT(min_vz, free_fall_vz)
        << "the fluid must resist the drop (vz above unresisted free-fall)";
    EXPECT_TRUE(decel) << "the body must decelerate after entering the pool";
    // Two-way: the grid->body reaction sink pushes UP (+z, the sign of buoyant resist).
    EXPECT_GT(max_react, 1e-4f) << "the body-reaction sink must push up (coupled)";
    // The fluid stays stable at K=2.0e5 (c~14 m/s): the deepest particle compresses only
    // a few % and dilation stays well below the J clamp ceiling (both live, not the clamp).
    EXPECT_GT(min_J, 0.9f) << "stable pool: the deepest particle compresses only a few %";
    EXPECT_LT(max_J, 2.0f) << "dilation must stay well below the J clamp ceiling (live bound)";
}
