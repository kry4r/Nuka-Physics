// ---------------------------------------------------------------------------
// MLS-MPM granular Drucker-Prager constitutive gate (model_kind == 4).
//
// Granular is a constitutive selected by the material-table model_kind, parallel
// to the elastic kinds (0/2) and the fluid (3): Hencky-elastic predictor + a
// Drucker-Prager return map in principal-strain space (Klar et al. 2016). It runs
// the SAME grid/transfer/floor/domain-wall/body-BC path. Asserts:
//   * column collapse: a released sand column slumps into a PILE whose repose
//     angle tracks dp_friction (35 deg steeper than 15 deg); the fluid control
//     spreads flat and the elastic control holds its shape;
//   * footprint: a scripted capsule pressed into a sand bed and lifted leaves a
//     crater that PERSISTS (>= 3 s, no elastic rebound, no fluid backfill);
//   * determinism: particle state byte-identical run-to-run (the D1 gather).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/backend.hpp"
#include "phi/op_schema.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"
#include "scene/cook/cook_to_model.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

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

// Shared lattice/material scales. dx = grid cell; particles at dx/2 (8/cell).
// E=3e5 keeps the explicit CFL easy (c ~ 13.7 m/s, dt_sub*c << 0.4*dx) while the
// gravity stresses (~kPa) still resolve on the yield surface.
constexpr float kDx       = 0.01f;
constexpr float kDensity  = 1600.0f;   // dry sand.
constexpr float kYoungs   = 3.0e5f;
constexpr float kPoisson  = 0.3f;
constexpr uint32_t kSubsteps = 30u;

// --- pile scenes ---------------------------------------------------------------
// Cone: the direct repose probe (a below-repose cone holds; above-repose slumps).
// Column: a tall release for the determinism gate (heavy plastic flow).
constexpr float kColHalf   = 0.03f;    // column half footprint (x/y).
constexpr float kColHeight = 0.12f;    // 2:1 aspect, classic collapse.
constexpr float kConeR     = 0.09f;    // cone base radius.
constexpr float kConeSlope = 30.0f;    // authored cone flank angle (deg).
constexpr float kDomHalf   = 0.15f;    // wide open floor: the slump never walls out.

void FillMaterialAndGrid(cook::MpmCookInput& in, float friction_deg,
                         float model_kind, float top_z);

cook::MpmCookInput BuildConeInput(float friction_deg, float model_kind) {
    cook::MpmCookInput in;
    const float pdx = kDx * 0.5f;
    const float H = kConeR * std::tan(kConeSlope * static_cast<float>(M_PI) / 180.0f);
    const char* shape = std::getenv("NUKA_CONE_SHAPE");
    const bool cyl = shape != nullptr && shape[0] == 'c' && shape[1] == 'y';
    const bool disc = shape != nullptr && shape[0] == 'd';
    const float top = disc ? 0.02f : H;
    for (float z = pdx; z <= top + 1e-5f; z += pdx) {
        const float rz = (cyl || disc) ? kConeR : kConeR * (1.0f - z / H);
        for (float x = -kConeR; x <= kConeR + 1e-5f; x += pdx)
            for (float y = -kConeR; y <= kConeR + 1e-5f; y += pdx)
                if (x * x + y * y <= rz * rz) in.positions.push_back(Vec3{x, y, z});
    }
    FillMaterialAndGrid(in, friction_deg, model_kind, top);
    return in;
}

cook::MpmCookInput BuildColumnInput(float friction_deg, float model_kind) {
    cook::MpmCookInput in;
    const float pdx = kDx * 0.5f;
    for (float x = -kColHalf; x <= kColHalf + 1e-5f; x += pdx)
        for (float y = -kColHalf; y <= kColHalf + 1e-5f; y += pdx)
            for (float z = pdx; z <= kColHeight + 1e-5f; z += pdx)
                in.positions.push_back(Vec3{x, y, z});
    FillMaterialAndGrid(in, friction_deg, model_kind, kColHeight);
    return in;
}

void FillMaterialAndGrid(cook::MpmCookInput& in, float friction_deg,
                         float model_kind, float top_z) {
    const float pdx = kDx * 0.5f;
    const size_t n = in.positions.size();
    in.velocities.assign(n, Vec3::Zero());
    const float vol0 = pdx * pdx * pdx;
    in.inv_mass.assign(n, 1.0f / (kDensity * vol0));
    in.vol0.assign(n, vol0);
    in.material.youngs = kYoungs;
    in.material.poisson = kPoisson;
    in.material.density = kDensity;
    in.material.model_kind = model_kind;
    in.material.dp_friction = friction_deg;
    in.material.dp_cohesion = 0.0f;         // dry cohesionless sand.
    if (model_kind > 2.5f && model_kind < 3.5f) {  // the fluid control's EOS.
        in.material.bulk_modulus = 2.0e5f;
        in.material.tait_gamma = 7.0f;
        in.material.viscosity = 2.0f;
    }
    in.grid_origin = Vec3{-kDomHalf, -kDomHalf, -3.0f * kDx};
    in.grid_dims[0] = static_cast<uint32_t>(2.0f * kDomHalf / kDx) + 1u;
    in.grid_dims[1] = in.grid_dims[0];
    in.grid_dims[2] = static_cast<uint32_t>((top_z + 15.0f * kDx) / kDx) + 4u;
    in.dx = kDx;
    in.substeps = kSubsteps;
    in.floor_normal = Vec3{0.0f, 0.0f, 1.0f};
    in.floor_d = 0.0f;
    in.floor_friction = 0.7f;               // rough ground under the pile.
}

nk::Model BuildPileModel(cook::MpmCookInput in) {
    nk::Model m;
    m.capacities.env_count = 1u;
    cook::XpbdCookInput soft;
    soft.solver = nk::Model::ParticleMode::Mpm;
    cook::CookSoftBodyParticles(m, 1u, soft, in);
    return m;
}

struct PileStats {
    float h_max = 0.0f;       // 98th-pct settled height.
    float r_spread = 0.0f;    // 95th-pct radial extent.
    float angle_deg = 0.0f;   // flank slope fit.
    bool  finite = true;
    uint32_t escape = 0u;
};

// Run `steps`, then profile the pile: bin particles by radius, per-bin surface
// z = 95th pct, fit the flank slope between 20% and 80% of the peak height.
PileStats RunPile(Backend b, cook::MpmCookInput in, uint32_t steps) {
    nk::Model m = BuildPileModel(std::move(in));
    const uint32_t P = m.capacities.particles_per_env;
    EXPECT_GT(P, 1000u);
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    EXPECT_TRUE(w.Ready());
    PileStats st;
    for (uint32_t s = 0; s < steps; ++s) {
        w.Step();
        if (s % 120u == 0u || s + 1u == steps) {
            uint32_t es = 0u;
            w.GetData().DownloadField(nk::FieldId::EnvStatus, &es, sizeof(uint32_t));
            st.escape |= es & nphi::kEnvStatusMpmGridEscape;
        }
    }
    std::vector<Vec3> pos(P, Vec3::Zero());
    w.GetData().DownloadField(nk::FieldId::ParticlePos, pos.data(), P * sizeof(Vec3));
    for (const Vec3& p : pos)
        st.finite = st.finite && std::isfinite(p.x) && std::isfinite(p.y) &&
                    std::isfinite(p.z);
    if (!st.finite) return st;

    std::vector<float> zs(P), rs(P);
    for (uint32_t i = 0; i < P; ++i) {
        zs[i] = pos[i].z;
        rs[i] = std::sqrt(pos[i].x * pos[i].x + pos[i].y * pos[i].y);
    }
    std::vector<float> zq = zs, rq = rs;
    std::nth_element(zq.begin(), zq.begin() + static_cast<size_t>(0.98 * P), zq.end());
    st.h_max = zq[static_cast<size_t>(0.98 * P)];
    std::nth_element(rq.begin(), rq.begin() + static_cast<size_t>(0.95 * P), rq.end());
    st.r_spread = rq[static_cast<size_t>(0.95 * P)];

    // Radial surface profile: 24 bins over [0, r99]; slope fit over the flank
    // (the 99th-pct radius kills stray fringe particles stretching the bins).
    constexpr int kBins = 24;
    std::vector<float> r99v = rs;
    std::nth_element(r99v.begin(), r99v.begin() + static_cast<size_t>(0.99 * P),
                     r99v.end());
    const float r_max = std::max(r99v[static_cast<size_t>(0.99 * P)], 1e-6f);
    std::vector<std::vector<float>> bin_z(kBins);
    for (uint32_t i = 0; i < P; ++i) {
        if (rs[i] > r_max) continue;
        int bi = std::min(kBins - 1, static_cast<int>(rs[i] / r_max * kBins));
        bin_z[bi].push_back(zs[i]);
    }
    std::vector<float> surf_r, surf_z;
    for (int bi = 0; bi < kBins; ++bi) {
        if (bin_z[bi].size() < 8u) continue;
        std::vector<float>& v = bin_z[bi];
        std::nth_element(v.begin(), v.begin() + static_cast<size_t>(0.95 * v.size()),
                         v.end());
        surf_r.push_back((bi + 0.5f) * r_max / kBins);
        surf_z.push_back(v[static_cast<size_t>(0.95 * v.size())]);
    }
    const float lo = 0.20f * st.h_max, hi = 0.80f * st.h_max;
    double sr = 0, sz = 0, srr = 0, srz = 0; int nn = 0;
    for (size_t k = 0; k < surf_r.size(); ++k) {
        if (surf_z[k] < lo || surf_z[k] > hi) continue;
        sr += surf_r[k]; sz += surf_z[k];
        srr += surf_r[k] * surf_r[k]; srz += surf_r[k] * surf_z[k];
        ++nn;
    }
    if (nn >= 2 && (nn * srr - sr * sr) > 1e-12) {
        const double slope = (nn * srz - sr * sz) / (nn * srr - sr * sr);
        st.angle_deg = static_cast<float>(std::atan(-slope) * 180.0 / M_PI);
    }
    return st;
}

// --- sand bed + scripted capsule (footprint) ---------------------------------
constexpr float kBedHalf   = 0.12f;   // bed half footprint.
constexpr float kBedTop    = 0.045f;  // bed depth (9 particle layers).
constexpr float kCapR      = 0.02f;   // capsule radius (a duck-foot scale).
constexpr float kCapHalfL  = 0.03f;   // capsule half segment length (along x).
constexpr float kPress     = 0.02f;   // press depth into the bed.

cook::MpmCookInput BuildBedInput() {
    cook::MpmCookInput in;
    const float pdx = kDx * 0.5f;
    for (float x = -kBedHalf; x <= kBedHalf + 1e-5f; x += pdx)
        for (float y = -kBedHalf; y <= kBedHalf + 1e-5f; y += pdx)
            for (float z = pdx; z <= kBedTop + 1e-5f; z += pdx)
                in.positions.push_back(Vec3{x, y, z});
    const size_t n = in.positions.size();
    in.velocities.assign(n, Vec3::Zero());
    const float vol0 = pdx * pdx * pdx;
    in.inv_mass.assign(n, 1.0f / (kDensity * vol0));
    in.vol0.assign(n, vol0);
    in.material.youngs = kYoungs;
    in.material.poisson = kPoisson;
    in.material.density = kDensity;
    in.material.model_kind = 4.0f;
    in.material.dp_friction = 35.0f;
    in.material.dp_cohesion = 0.0f;
    in.grid_origin = Vec3{-kBedHalf - 4.0f * kDx, -kBedHalf - 4.0f * kDx, -3.0f * kDx};
    in.grid_dims[0] = static_cast<uint32_t>((2.0f * (kBedHalf + 4.0f * kDx)) / kDx) + 1u;
    in.grid_dims[1] = in.grid_dims[0];
    in.grid_dims[2] = static_cast<uint32_t>((kBedTop + 20.0f * kDx) / kDx) + 4u;
    in.dx = kDx;
    in.substeps = kSubsteps;
    in.floor_normal = Vec3{0.0f, 0.0f, 1.0f};
    in.floor_d = 0.0f;
    in.floor_friction = 0.7f;
    return in;
}

// Analytic capsule SDF (segment along local x, radius r) cooked into the Model's
// sparse grid tables, mirroring the analytic box pattern of the fluid gate.
void AddCapsuleSdf(nk::Model& m, int32_t body_id, float radius, float half_len) {
    const float vh = kDx;
    const float band = 3.0f * vh;
    const float ex = half_len + radius + band, eyz = radius + band;
    const int nx = static_cast<int>(std::ceil(ex / vh)) + 1;
    const int nyz = static_cast<int>(std::ceil(eyz / vh)) + 1;
    const Vec3 origin{-static_cast<float>(nx) * vh, -static_cast<float>(nyz) * vh,
                      -static_cast<float>(nyz) * vh};
    const uint32_t base = static_cast<uint32_t>(m.sdf_cell_values.size());
    uint32_t count = 0u;
    for (int i = 0; i <= 2 * nx; ++i)
        for (int j = 0; j <= 2 * nyz; ++j)
            for (int k = 0; k <= 2 * nyz; ++k) {
                const Vec3 p{origin.x + i * vh, origin.y + j * vh, origin.z + k * vh};
                const float cx = std::max(-half_len, std::min(half_len, p.x));
                const Vec3 d{p.x - cx, p.y, p.z};
                const float dist = std::sqrt(d.LengthSq());
                const float phi = dist - radius;
                if (std::fabs(phi) > band) continue;
                const Vec3 grad = dist > 1e-8f ? d * (1.0f / dist) : Vec3{0, 0, 1};
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
    sg.dims[0] = static_cast<uint32_t>(2 * nx + 1);
    sg.dims[1] = sg.dims[2] = static_cast<uint32_t>(2 * nyz + 1);
    sg.cell_offset = base;
    sg.cell_count = count;
    const uint32_t grid_idx = static_cast<uint32_t>(m.sdf_grids.size());
    m.sdf_grids.push_back(sg);

    nk::Model::PairDrivenShape sh;
    sh.kind = kKindBox;  // shape row only routes the SDF; the prim kind is unused here.
    sh.params[0] = radius; sh.params[1] = radius; sh.params[2] = half_len;
    sh.contype = 1u; sh.conaffinity = 1u;
    sh.sdf_grid = grid_idx;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

nk::Model BuildBedWithCapsuleModel(float cap_z0) {
    nk::Model m;
    m.capacities.env_count = 1u;

    // Body 0: the scripted press capsule. Immovable (inv_mass 0) -- the test drives
    // its pose/velocity from the host; the grid BC reads both.
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = Vec3{0.0f, 0.0f, cap_z0};
    bi.inv_mass = 0.0f;
    bi.inv_inertia = Vec3{0, 0, 0};
    m.body_init.push_back(bi);
    AddCapsuleSdf(m, 0, kCapR, kCapHalfL);

    // Body 1: a far immovable filler (the broadphase wants >= 2 collidables).
    nk::Model::BodyInit bf;
    bf.pose = Transform::Identity();
    bf.pose.position = Vec3{5.0f, 0.0f, 0.0f};
    bf.inv_mass = 0.0f; bf.inv_inertia = Vec3{0, 0, 0};
    m.body_init.push_back(bf);
    AddCapsuleSdf(m, 1, 0.02f, 0.02f);

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

    cook::XpbdCookInput soft;
    soft.solver = nk::Model::ParticleMode::Mpm;
    cook::CookSoftBodyParticles(m, 1u, soft, BuildBedInput());
    m.particles.mpm_body_friction = 0.5f;
    return m;
}

// Crater probe: 95th-pct surface z of the bed inside / outside the press footprint.
struct BedSurface {
    float in_z = 0.0f;    // under the capsule footprint.
    float out_z = 0.0f;   // an outer control ring.
};
BedSurface ProbeBed(const std::vector<Vec3>& pos) {
    std::vector<float> in_zs, out_zs;
    for (const Vec3& p : pos) {
        const bool in_x = std::fabs(p.x) < kCapHalfL * 0.8f;
        const bool in_y = std::fabs(p.y) < kCapR * 0.6f;
        if (in_x && in_y) in_zs.push_back(p.z);
        const float r = std::sqrt(p.x * p.x + p.y * p.y);
        if (r > kCapHalfL + 3.0f * kCapR && r < kBedHalf - 2.0f * kDx)
            out_zs.push_back(p.z);
    }
    BedSurface s;
    auto pct95 = [](std::vector<float>& v) {
        if (v.empty()) return 0.0f;
        const size_t k = static_cast<size_t>(0.95 * v.size());
        std::nth_element(v.begin(), v.begin() + k, v.end());
        return v[k];
    };
    s.in_z = pct95(in_zs);
    s.out_z = pct95(out_zs);
    return s;
}

}  // namespace

// Gate a: the direct repose probe. A 30-deg cone under 35-deg sand HOLDS its
// flank; under 15-deg sand it SLUMPS shallower; the fluid control flattens it;
// the elastic control freezes it. (A tall-column release is deliberately NOT the
// probe: inertial collapse runout settles far below repose -- Lube et al. 2004.)
TEST(MpmGranular, ConeReposeTracksFriction) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    constexpr uint32_t kSteps = 600u;   // 2.5 s settle.
    const float H = kConeR * std::tan(kConeSlope * static_cast<float>(M_PI) / 180.0f);

    const PileStats sand35 = RunPile(b, BuildConeInput(35.0f, 4.0f), kSteps);
    const PileStats sand15 = RunPile(b, BuildConeInput(15.0f, 4.0f), kSteps);
    const PileStats fluid  = RunPile(b, BuildConeInput(0.0f, 3.0f), kSteps);
    const PileStats elast  = RunPile(b, BuildConeInput(0.0f, 2.0f), kSteps);

    std::fprintf(stderr,
                 "[granular cone30] sand35: h=%.4f r=%.4f angle=%.1f esc=%u | "
                 "sand15: h=%.4f r=%.4f angle=%.1f | fluid: h=%.4f r=%.4f "
                 "angle=%.1f | elastic: h=%.4f r=%.4f angle=%.1f (H0=%.4f)\n",
                 sand35.h_max, sand35.r_spread, sand35.angle_deg, sand35.escape,
                 sand15.h_max, sand15.r_spread, sand15.angle_deg,
                 fluid.h_max, fluid.r_spread, fluid.angle_deg,
                 elast.h_max, elast.r_spread, elast.angle_deg, H);

    ASSERT_TRUE(sand35.finite && sand15.finite && fluid.finite && elast.finite);
    EXPECT_EQ(0u, sand35.escape) << "the pile must stay inside the grid";

    // 35-deg sand: a below-repose cone holds its flank (and cannot steepen).
    EXPECT_GT(sand35.angle_deg, 22.0f) << "35-deg sand must hold a 30-deg cone";
    EXPECT_LT(sand35.angle_deg, 38.0f) << "the cone must not steepen past repose";
    EXPECT_GT(sand35.h_max, 0.7f * H) << "the held cone keeps most of its height";
    // 15-deg sand: the same cone is above repose -> it slumps measurably shallower.
    EXPECT_LT(sand15.angle_deg, sand35.angle_deg - 5.0f)
        << "repose must track dp_friction";
    EXPECT_LT(sand15.angle_deg, 22.0f) << "15-deg sand cannot hold a 30-deg cone";
    EXPECT_GT(sand15.angle_deg, 3.0f) << "15-deg sand still piles (not a fluid)";
    EXPECT_GT(sand15.r_spread, sand35.r_spread) << "lower friction spreads wider";
    // Fluid control: the cone flattens out (no static flank).
    EXPECT_LT(fluid.angle_deg, 8.0f) << "a fluid holds no repose angle";
    EXPECT_LT(fluid.h_max, 0.5f * H) << "the fluid cone must collapse";
    // Elastic control: the cone freezes near its authored shape. (The 98th-pct
    // height probe reads ~0.77*H even on the pristine cone -- the apex thins to a
    // single particle -- so the height bound is on the probe's own scale.)
    EXPECT_GT(elast.angle_deg, 24.0f) << "the elastic cone must keep its flank";
    EXPECT_GT(elast.h_max, 0.70f * H) << "the elastic cone must keep its height";
    EXPECT_GT(sand35.h_max, 0.95f * elast.h_max)
        << "35-deg sand must hold the cone as the elastic control does";
}

// Gate b: a capsule pressed into the sand bed and lifted leaves a crater that
// persists >= 3 s (no elastic rebound, no fluid backfill).
TEST(MpmGranular, CapsuleFootprintPersists) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    const float z_start = kBedTop + kCapR + 0.02f;
    const float z_press = kBedTop + kCapR - kPress;
    nk::Model m = BuildBedWithCapsuleModel(z_start);
    const uint32_t P = m.capacities.particles_per_env;
    ASSERT_GT(P, 5000u);
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());

    const float dt = Cfg().dt;
    std::vector<Vec3> pos(P, Vec3::Zero());
    auto download = [&] {
        w.GetData().DownloadField(nk::FieldId::ParticlePos, pos.data(),
                                  P * sizeof(Vec3));
    };
    Transform cap = Transform::Identity();
    cap.position = Vec3{0.0f, 0.0f, z_start};
    Vec3 cap_v{0, 0, 0};
    auto drive = [&](float vz) {
        cap_v = Vec3{0.0f, 0.0f, vz};
        cap.position.z += vz * dt;
        w.GetData().UploadField(nk::FieldId::BodyPose, &cap, sizeof(Transform));
        w.GetData().UploadField(nk::FieldId::BodyLinearVelocity, &cap_v, sizeof(Vec3));
        w.Step();
    };

    // Settle the freshly sampled bed briefly, then press / hold / lift.
    for (uint32_t s = 0; s < 120u; ++s) drive(0.0f);
    download();
    const BedSurface before = ProbeBed(pos);

    const float v_down = -0.10f;
    const uint32_t press_steps =
        static_cast<uint32_t>((z_start - z_press) / (-v_down * dt));
    for (uint32_t s = 0; s < press_steps; ++s) drive(v_down);
    for (uint32_t s = 0; s < 60u; ++s) drive(0.0f);                 // hold 0.25 s.
    const float v_up = 0.25f;
    const uint32_t lift_steps = static_cast<uint32_t>(0.15f / (v_up * dt));
    for (uint32_t s = 0; s < lift_steps; ++s) drive(v_up);          // clear the bed.

    for (uint32_t s = 0; s < 48u; ++s) drive(0.0f);                 // 0.2 s settle.
    download();
    const BedSurface after_lift = ProbeBed(pos);
    const float depth0 = before.in_z - after_lift.in_z;

    for (uint32_t s = 0; s < 720u; ++s) drive(0.0f);                // 3 s persistence.
    download();
    const BedSurface after_wait = ProbeBed(pos);
    const float depth1 = before.in_z - after_wait.in_z;

    uint32_t es = 0u;
    w.GetData().DownloadField(nk::FieldId::EnvStatus, &es, sizeof(uint32_t));
    bool finite = true;
    for (const Vec3& p : pos)
        finite = finite && std::isfinite(p.x) && std::isfinite(p.y) &&
                 std::isfinite(p.z);

    std::fprintf(stderr,
                 "[granular footprint] bed_before=%.4f crater t0 depth=%.4f "
                 "t+3s depth=%.4f out_z t0=%.4f t1=%.4f escape=%u\n",
                 before.in_z, depth0, depth1, after_lift.out_z, after_wait.out_z,
                 es & nphi::kEnvStatusMpmGridEscape);

    EXPECT_TRUE(finite);
    EXPECT_EQ(0u, es & nphi::kEnvStatusMpmGridEscape);
    // A real crater: at least ~40% of the 2 cm press survives the lift.
    EXPECT_GT(depth0, 0.008f) << "the press must leave a crater";
    // Persistence: after 3 s the crater neither rebounds nor backfills.
    EXPECT_GT(depth1, 0.75f * depth0) << "the crater must persist (plastic memory)";
    // The bed outside the footprint stays put (no global slump/splash).
    EXPECT_GT(after_wait.out_z, before.out_z - 0.01f)
        << "the surrounding bed must not collapse";
}

// Determinism: the granular constitutive stays on the deterministic gather path
// (particle positions + F byte-identical run-to-run).
TEST(MpmGranular, CollapseByteIdenticalRunToRun) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    auto run = [&](std::vector<Vec3>& pos, std::vector<float>& F) -> bool {
        nk::Model m = BuildPileModel(BuildColumnInput(35.0f, 4.0f));
        const uint32_t P = m.capacities.particles_per_env;
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
        if (!w.Ready()) return false;
        for (uint32_t s = 0; s < 120u; ++s) w.Step();
        pos.assign(P, Vec3::Zero());
        F.assign(static_cast<size_t>(P) * 9u, 0.0f);
        return w.GetData().DownloadField(nk::FieldId::ParticlePos, pos.data(),
                                         P * sizeof(Vec3)) &&
               w.GetData().DownloadField(nk::FieldId::ParticleF, F.data(),
                                         F.size() * sizeof(float));
    };
    std::vector<Vec3> pa, pb;
    std::vector<float> fa, fb;
    ASSERT_TRUE(run(pa, fa));
    ASSERT_TRUE(run(pb, fb));
    EXPECT_EQ(0, std::memcmp(pa.data(), pb.data(), pa.size() * sizeof(Vec3)))
        << "granular particle positions differ run-to-run";
    EXPECT_EQ(0, std::memcmp(fa.data(), fb.data(), fa.size() * sizeof(float)))
        << "granular particle F differs run-to-run";
}

// Host-only data-model gate: Granular x MlsMpm validates + roundtrips through
// .nks with its DP material; every other granular pairing / cross-declaration
// is rejected LOUDLY.
TEST(MpmGranular, ValidatorAndNksRoundtrip) {
    namespace ns = nuka::scene;
    auto make = [](ns::MediaRecord::Kind k, ns::MediaRecord::Method me,
                   float model_kind) {
        ns::MediaRecord m;
        m.name = "bed";
        m.kind = k;
        m.method = me;
        m.fluid_box.min = Vec3{-0.8f, -0.45f, 0.0f};
        m.fluid_box.max = Vec3{0.8f, 0.45f, 0.035f};
        m.fluid_box.spacing = 0.005f;
        m.mpm.youngs = 3.0e5f;
        m.mpm.poisson = 0.3f;
        m.mpm.density = 1600.0f;
        m.mpm.dp_friction = 35.0f;
        m.mpm.dp_cohesion = 12.0f;
        m.mpm.model_kind = model_kind;
        m.mpm.dx = 0.01f;
        m.mpm.substeps = 30u;
        return m;
    };
    using K = ns::MediaRecord::Kind;
    using Me = ns::MediaRecord::Method;
    EXPECT_NO_THROW(cook::ValidateMedia({make(K::Granular, Me::MlsMpm, 4.0f)}));
    EXPECT_NO_THROW(cook::ValidateMedia({make(K::Granular, Me::MlsMpm, 0.0f)}));
    EXPECT_THROW(cook::ValidateMedia({make(K::Granular, Me::Xpbd, 4.0f)}),
                 std::runtime_error);
    EXPECT_THROW(cook::ValidateMedia({make(K::Granular, Me::Pbf, 4.0f)}),
                 std::runtime_error);
    EXPECT_THROW(cook::ValidateMedia({make(K::Fluid, Me::MlsMpm, 4.0f)}),
                 std::runtime_error) << "model_kind 4 demands a Granular medium";
    EXPECT_THROW(cook::ValidateMedia({make(K::Granular, Me::MlsMpm, 3.0f)}),
                 std::runtime_error) << "a granular bed cannot declare fluid";

    // The cook pins model_kind 4 even when the material field is left unset.
    const cook::MpmCookInput cooked =
        cook::BuildMpmInput(make(K::Granular, Me::MlsMpm, 0.0f));
    EXPECT_EQ(4.0f, cooked.material.model_kind);
    EXPECT_EQ(35.0f, cooked.material.dp_friction);
    EXPECT_GT(cooked.positions.size(), 100u) << "the box lattice must sample";

    // .nks roundtrip: kind string + the DP material survive save/load.
    ns::SceneIR scene;
    scene.AddMedia(make(K::Granular, Me::MlsMpm, 4.0f));
    char tmpl[] = "/tmp/nks_granular_XXXXXX";
    char* dir = ::mkdtemp(tmpl);
    ASSERT_NE(nullptr, dir);
    const std::string path = std::string(dir) + "/granular.nks";
    ns::nks::Save(scene, path);
    const ns::SceneIR loaded = ns::nks::Load(path);
    ASSERT_EQ(1u, loaded.Media().size());
    const ns::MediaRecord& r = loaded.Media().front();
    EXPECT_EQ(K::Granular, r.kind);
    EXPECT_EQ(Me::MlsMpm, r.method);
    EXPECT_EQ(35.0f, r.mpm.dp_friction);
    EXPECT_EQ(12.0f, r.mpm.dp_cohesion);
    EXPECT_EQ(4.0f, r.mpm.model_kind);
    EXPECT_EQ(0.005f, r.fluid_box.spacing);
    std::remove(path.c_str());
    ::rmdir(dir);
}

// Manual diagnostic: step one cone (env NUKA_CONE_KIND / NUKA_CONE_FRICTION)
// and report the first non-finite step, the max speed, and the escape bit.
TEST(MpmGranular, DISABLED_ConeProbe) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    const char* ek = std::getenv("NUKA_CONE_KIND");
    const char* ef = std::getenv("NUKA_CONE_FRICTION");
    const float kind = ek ? std::strtof(ek, nullptr) : 4.0f;
    const float fric = ef ? std::strtof(ef, nullptr) : 35.0f;
    cook::MpmCookInput ci = BuildConeInput(fric, kind);
    const char* ed = std::getenv("NUKA_CONE_DTDIV");   // substep-granular forensics.
    const uint32_t dtdiv = ed ? static_cast<uint32_t>(std::atoi(ed)) : 0u;
    if (dtdiv > 0u) ci.substeps = 1u;
    std::fprintf(stderr, "[cone probe] cook n=%zu p705=(%g %g %g) inv_mass705=%g "
                 "vol0_705=%g\n", ci.positions.size(),
                 ci.positions.size() > 705 ? ci.positions[705].x : -1.0f,
                 ci.positions.size() > 705 ? ci.positions[705].y : -1.0f,
                 ci.positions.size() > 705 ? ci.positions[705].z : -1.0f,
                 ci.inv_mass.size() > 705 ? ci.inv_mass[705] : -1.0f,
                 ci.vol0.size() > 705 ? ci.vol0[705] : -1.0f);
    nk::Model m = BuildPileModel(std::move(ci));
    const uint32_t P = m.capacities.particles_per_env;
    std::vector<float> F0(static_cast<size_t>(P) * 9u, 0.0f);
    nk::Pipeline::SolverConfig cfg = Cfg();
    if (dtdiv > 0u) cfg.dt /= static_cast<float>(dtdiv);
    nk::World w(std::move(m), 1u, b.dev, b.backend, cfg);
    ASSERT_TRUE(w.Ready());
    std::vector<Vec3> pos(P), vel(P);
    w.GetData().DownloadField(nk::FieldId::ParticlePos, pos.data(), P * sizeof(Vec3));
    w.GetData().DownloadField(nk::FieldId::ParticleF, F0.data(),
                              F0.size() * sizeof(float));
    std::fprintf(stderr, "[cone probe] pre pos705=(%g %g %g) F705=[%g %g %g %g %g %g "
                 "%g %g %g]\n", pos[705].x, pos[705].y, pos[705].z, F0[705*9+0],
                 F0[705*9+1], F0[705*9+2], F0[705*9+3], F0[705*9+4], F0[705*9+5],
                 F0[705*9+6], F0[705*9+7], F0[705*9+8]);
    for (uint32_t i = 0; i < P; ++i)
        if (!std::isfinite(pos[i].x) || !std::isfinite(pos[i].y) ||
            !std::isfinite(pos[i].z))
            std::fprintf(stderr, "[cone probe] PRE-STEP nonfinite i=%u (%g %g %g)\n",
                         i, pos[i].x, pos[i].y, pos[i].z);
    const uint32_t watch = [] {
        const char* ew = std::getenv("NUKA_CONE_WATCH");
        return ew ? static_cast<uint32_t>(std::atoi(ew)) : 705u;
    }();
    for (uint32_t s = 0; s < 600u; ++s) {
        w.Step();
        if (dtdiv > 0u) {
            w.GetData().DownloadField(nk::FieldId::ParticlePos, pos.data(),
                                      P * sizeof(Vec3));
            w.GetData().DownloadField(nk::FieldId::ParticleVel, vel.data(),
                                      P * sizeof(Vec3));
            w.GetData().DownloadField(nk::FieldId::ParticleF, F0.data(),
                                      F0.size() * sizeof(float));
            const float* Fw = F0.data() + static_cast<size_t>(watch) * 9u;
            std::fprintf(stderr, "[sub %u] p=(%.6g %.6g %.6g) v=(%.4g %.4g %.4g) "
                         "F=[%.4g %.4g %.4g; %.4g %.4g %.4g; %.4g %.4g %.4g]\n",
                         s, pos[watch].x, pos[watch].y, pos[watch].z, vel[watch].x,
                         vel[watch].y, vel[watch].z, Fw[0], Fw[1], Fw[2], Fw[3],
                         Fw[4], Fw[5], Fw[6], Fw[7], Fw[8]);
            if (!std::isfinite(pos[watch].z) || !std::isfinite(vel[watch].z) ||
                s > 60u) break;
            continue;
        }
        if (s % 10u != 0u && s != 599u) continue;
        w.GetData().DownloadField(nk::FieldId::ParticlePos, pos.data(), P * sizeof(Vec3));
        w.GetData().DownloadField(nk::FieldId::ParticleVel, vel.data(), P * sizeof(Vec3));
        uint32_t es = 0u;
        w.GetData().DownloadField(nk::FieldId::EnvStatus, &es, sizeof(uint32_t));
        float vmax = 0.0f, zmax = -1e9f;
        bool fin = true;
        for (uint32_t i = 0; i < P; ++i) {
            fin = fin && std::isfinite(pos[i].x) && std::isfinite(pos[i].z) &&
                  std::isfinite(vel[i].z);
            vmax = std::max(vmax, std::sqrt(vel[i].LengthSq()));
            zmax = std::max(zmax, pos[i].z);
        }
        std::fprintf(stderr, "[cone probe] step=%u P=%u vmax=%.3f zmax=%.4f "
                     "status=0x%x finite=%d\n", s, P, vmax, zmax, es, fin ? 1 : 0);
        if (!fin) {
            for (uint32_t i = 0; i < P; ++i)
                if (!std::isfinite(pos[i].z) || !std::isfinite(pos[i].x)) {
                    std::fprintf(stderr, "[cone probe] first bad i=%u pos=(%g %g %g) "
                                 "vel=(%g %g %g)\n", i, pos[i].x, pos[i].y, pos[i].z,
                                 vel[i].x, vel[i].y, vel[i].z);
                    break;
                }
            break;
        }
    }
    SUCCEED();
}

// Throughput probe (manual; run with --gtest_also_run_disabled_tests). ~200k
// particles on a wide bed; reports full-world steps/s (kSubsteps substeps each).
TEST(MpmGranular, DISABLED_Throughput200k) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    nk::Model m;
    m.capacities.env_count = 1u;
    cook::MpmCookInput in;
    const float pdx = kDx * 0.5f;
    const float half = 0.25f, top = 0.05f;   // 100 x 100 x 20 lattice = 200k.
    for (float x = -half; x < half - 1e-5f; x += pdx)
        for (float y = -half; y < half - 1e-5f; y += pdx)
            for (float z = pdx; z <= top + 1e-5f; z += pdx)
                in.positions.push_back(Vec3{x, y, z});
    const size_t n = in.positions.size();
    in.velocities.assign(n, Vec3::Zero());
    const float vol0 = pdx * pdx * pdx;
    in.inv_mass.assign(n, 1.0f / (kDensity * vol0));
    in.vol0.assign(n, vol0);
    in.material.youngs = kYoungs; in.material.poisson = kPoisson;
    in.material.density = kDensity;
    in.material.model_kind = 4.0f;
    in.material.dp_friction = 35.0f;
    in.grid_origin = Vec3{-half - 4.0f * kDx, -half - 4.0f * kDx, -3.0f * kDx};
    in.grid_dims[0] = static_cast<uint32_t>((2.0f * (half + 4.0f * kDx)) / kDx) + 1u;
    in.grid_dims[1] = in.grid_dims[0];
    in.grid_dims[2] = static_cast<uint32_t>((top + 12.0f * kDx) / kDx) + 4u;
    in.dx = kDx; in.substeps = kSubsteps;
    in.floor_normal = Vec3{0, 0, 1}; in.floor_d = 0.0f; in.floor_friction = 0.7f;
    cook::XpbdCookInput soft;
    soft.solver = nk::Model::ParticleMode::Mpm;
    cook::CookSoftBodyParticles(m, 1u, soft, in);
    const uint32_t P = m.capacities.particles_per_env;
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());
    for (uint32_t s = 0; s < 20u; ++s) w.Step();   // warm up.
    std::vector<Vec3> probe(1);
    w.GetData().DownloadField(nk::FieldId::ParticlePos, probe.data(), sizeof(Vec3));
    const auto t0 = std::chrono::steady_clock::now();
    constexpr uint32_t kSteps = 240u;
    for (uint32_t s = 0; s < kSteps; ++s) w.Step();
    w.GetData().DownloadField(nk::FieldId::ParticlePos, probe.data(), sizeof(Vec3));
    const auto t1 = std::chrono::steady_clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    std::fprintf(stderr,
                 "[granular perf] P=%u substeps=%u steps=%u wall=%.3fs -> %.1f "
                 "steps/s (%.1f substeps/s)\n",
                 P, kSubsteps, kSteps, sec, kSteps / sec, kSteps * kSubsteps / sec);
    SUCCEED();
}
