// ---------------------------------------------------------------------------
// Body<->particle FRICTION asymmetry — cloth grips, fluid slides, and the ONLY
// difference is the per-particle-SYSTEM friction coefficient (DATA, not code).
//
// One SoftFluid Model carries BOTH a SOFT (cloth) particle and a FLUID particle,
// each pressed straight down onto its own static box AND given the same tangential
// (+x) approach velocity. Both contacts flow through the SAME body<->particle
// narrowphase -> EmitPairDrivenRows -> ComputeRowMeff -> SolveRowsBlockIsland: same
// condim-3 pyramid (4 spokes bounded by mu*Sum(lambda_normal)), same row, same
// kernel. The soft slice reads the soft mu (finite); the fluid slice reads the
// fluid mu (~0); they mix with the static box side by the contact solmix=max rule.
//
// Asserts (read right after SolveRowsBlockIsland, before the finalize compose):
//   CLOTH (finite mu): the friction spokes carry lambda > 0 and the soft
//     particle's tangential velocity is bounded (it grips / is dragged).
//   FLUID (mu ~ 0): the friction bound collapses to ~0 (spoke lambda ~ 0) so the
//     tangential velocity slides on essentially unchanged, WHILE the normal row
//     still brakes the downward approach (non-penetration is unchanged).
// The asymmetry is therefore proven to be mu (data), not a per-medium code path.
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
#include "nk/solve/nk_row.hpp"
#include "phi/backend.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kKindBox = 2u;  // collision::ShapeKind Box

struct Backend { nphi::Device* dev = nullptr; nphi::Backend* backend = nullptr; };
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
    cfg.contact_margin = 0.0f;
    cfg.vel_iters = 64u;  // enough sweeps for the pyramid to bind tangential lambda.
    cfg.pos_iters = 0u;   // velocity-only so the row lambda is the sole correction.
    cfg.max_pairs = 32u;
    return cfg;
}

constexpr float kTangentialVx = 0.5f;  // shared +x tangential approach (both).
constexpr float kNormalVz = -1.0f;     // shared downward approach (both).

// A SoftFluid Model: a SOFT particle (index 0) pressed onto static box A AND a
// FLUID particle (index 1) pressed onto static box B, well separated in x so their
// PBF neighbor grids never couple. Both enter with the SAME (-z, +x) velocity, so
// the ONLY difference between the two contacts is the per-system friction.
// n_soft = 1 -> index 0 is soft, index 1 is fluid. soft_friction / fluid_friction
// are the per-system mu the body<->particle side reads.
nk::Model BuildModel(float soft_mu, float fluid_mu) {
    nk::Model m;
    auto add_static_box = [&](const Vec3& pos, float half, int32_t body_id) {
        nk::Model::BodyInit bi;
        bi.pose = Transform::Identity();
        bi.pose.position = pos;
        bi.inv_mass = 0.0f;            // immovable (one-way by mass).
        bi.inv_inertia = Vec3{0, 0, 0};
        m.body_init.push_back(bi);
        nk::Model::PairDrivenShape sh;
        sh.kind = kKindBox;
        sh.params[0] = half; sh.params[1] = half; sh.params[2] = half;
        sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
        sh.body_id = body_id; sh.group = 0u;
        m.shape_table_rows.push_back(sh);
    };
    // Two big-enough floor boxes (top face z = 0.10), 4 m apart in x. Each particle
    // (radius 0.02, centre z = 0.115 -> bottom 0.095 -> 0.005 below the face) overlaps
    // ONLY the box directly below it.
    add_static_box(Vec3{0.0f, 0.0f, 0.0f}, 0.10f, 0);   // box A under the soft particle.
    add_static_box(Vec3{4.0f, 0.0f, 0.0f}, 0.10f, 1);   // box B under the fluid particle.

    // The body (foot) side carries a SMOOTH authored material (mu = 0) on BOTH boxes,
    // so the solmix=max contact mu is decided by the PARTICLE system's mu alone:
    // max(0, soft_mu) vs max(0, fluid_mu). (A high-friction body would mask the
    // particle's mu under max -- the honest reading is that a fluid slides only when
    // the foot is itself low-friction; here both boxes are held at the same mu so the
    // ONLY difference between the two contacts is the per-system particle mu.)
    nk::ContactProfileV1 smooth_profile{};
    smooth_profile.mu1 = 0.0f;    // frictionless foot (mu_s = mu_d = 0).
    smooth_profile.mu2 = 0.0f;
    nk::ModelMaterialBucket smooth(smooth_profile);
    m.material_buckets = {smooth};
    m.body_material_bucket = {0u, 0u};              // both boxes -> bucket 0.
    m.capacities.num_material_buckets = 1u;

    nk::Model::ModelParticles& mp = m.particles;
    mp.mode = nk::Model::ParticleMode::SoftFluid;
    mp.n_soft_particles = 1u;             // index 0 soft, index 1 fluid.
    mp.soft_friction = soft_mu;
    mp.fluid_friction = fluid_mu;
    mp.initial_pos = {Vec3{0.0f, 0.0f, 0.115f}, Vec3{4.0f, 0.0f, 0.115f}};
    mp.initial_vel = {Vec3{kTangentialVx, 0.0f, kNormalVz},
                      Vec3{kTangentialVx, 0.0f, kNormalVz}};
    mp.inv_mass = {1.0f / 0.01f, 1.0f / 0.01f};   // 10 g each.
    mp.xpbd_iters = 1;
    // A particle is a SPHERE of d_min/2 on the ONE path; radius 0.02 -> d_min 0.04.
    mp.pp_contact_d_min = 0.04f;

    // The fluid slice's PBF params + uniform grid (covers both particles; the two
    // are 4 m apart so neither is the other's neighbor -> no cross coupling).
    const float rho0 = 1000.0f, spacing = 0.03f, h = spacing * 1.5f;
    mp.pbf_rest_density = rho0;
    mp.pbf_support_radius = h;
    mp.pbf_particle_mass = rho0 * spacing * spacing * spacing;
    mp.pbf_cfm_epsilon = 1.0e-6f;
    mp.pbf_iters = 2u;
    mp.pbf_clamp_overdensity = true;
    mp.grid_min = Vec3{-0.2f, -0.2f, -0.2f};
    mp.grid_dims[0] = 160; mp.grid_dims[1] = 16; mp.grid_dims[2] = 16;
    mp.cell_size = h;
    mp.query_radius = h;

    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_contacts_per_env = 16u;
    cap.max_rows_per_env = 16u * nk::kPairDrivenRowsPerSlot;
    cap.particles_per_env = static_cast<uint32_t>(mp.initial_pos.size());
    cap.max_grid_cells = mp.grid_dims[0] * mp.grid_dims[1] * mp.grid_dims[2];
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;
    return m;
}

// Run the pipeline ops in the fixed order but STOP right after SolveRowsBlockIsland
// (before ParticleFinalize composes/overwrites particle_vel) so the raw solved row
// lambda and the impulse-bearing particle_vel are readable.
bool StepToSolve(nk::World& w) {
    for (const nphi::OpCall& call : w.GetPipeline().Calls()) {
        if (w.DispatchOp(call.op, call.params) != nphi::Status::Ok) return false;
        if (call.op == nphi::NkOp::SolveRowsBlockIsland) return true;
    }
    return true;
}

struct Solved {
    bool ok = false;
    std::vector<float> lambda;        // [rows]
    std::vector<float> urows;         // [rows * 32]
    std::vector<Vec3>  particle_vel;  // [particles]
};

Solved Download(nk::World& w, uint32_t rows, uint32_t particles) {
    Solved s;
    s.lambda.assign(rows, 0.0f);
    s.urows.assign(static_cast<size_t>(rows) * 32u, 0.0f);
    s.particle_vel.assign(particles, Vec3::Zero());
    nk::Data& d = w.GetData();
    s.ok = d.DownloadField(nk::FieldId::Lambda, s.lambda.data(),
                           s.lambda.size() * sizeof(float)) &&
           d.DownloadField(nk::FieldId::Urows, s.urows.data(),
                           s.urows.size() * sizeof(float)) &&
           d.DownloadField(nk::FieldId::ParticleVel, s.particle_vel.data(),
                           s.particle_vel.size() * sizeof(Vec3));
    return s;
}

// NkRow packs to 32 f32 (the static_asserts in nk_row.hpp pin the layout):
// [0]=flags, side a at [16]=kind [17]=index, side b at [24]=kind [25]=index.
struct RowSides {
    uint32_t flags, a_kind, b_kind, a_index, b_index;
    bool active, friction;
};
RowSides DecodeRow(const std::vector<float>& urows, uint32_t row) {
    const float* r = urows.data() + static_cast<size_t>(row) * 32u;
    auto u = [&](int i) { uint32_t v; std::memcpy(&v, &r[i], 4); return v; };
    RowSides s;
    s.flags = u(0);
    s.active = (s.flags & nk::nk_row_flags::kActive) != 0u;
    s.friction = (s.flags & (nk::nk_row_flags::kFriction |
                             nk::nk_row_flags::kBlockTangent)) != 0u;
    s.a_kind = u(16); s.a_index = u(17);
    s.b_kind = u(24); s.b_index = u(25);
    return s;
}

// Sum the |lambda| over the normal vs friction-spoke rows whose particle side
// carries `particle_id`, for one body<->particle contact.
struct ContactLambda { float normal = 0.0f; float friction = 0.0f; uint32_t rows = 0u; };
ContactLambda SumLambda(const Solved& s, uint32_t rows, uint32_t particle_id) {
    ContactLambda c;
    for (uint32_t row = 0u; row < rows; ++row) {
        const RowSides rs = DecodeRow(s.urows, row);
        if (!rs.active) continue;
        const bool a_part = rs.a_kind == nk::kNkSideParticle && rs.a_index == particle_id;
        const bool b_part = rs.b_kind == nk::kNkSideParticle && rs.b_index == particle_id;
        if (!a_part && !b_part) continue;
        ++c.rows;
        if (rs.friction) c.friction += std::fabs(s.lambda[row]);
        else c.normal += std::fabs(s.lambda[row]);
    }
    return c;
}

}  // namespace

// CLOTH grips vs FLUID slides — same kernel/pyramid/row, only the per-system mu
// differs. Soft mu finite -> friction lambda > 0 + tangential bounded; fluid mu ~0
// -> friction lambda ~ 0 + tangential ~ free, normal braking unchanged.
TEST(BodyParticleFriction, ClothGripsFluidSlidesByMuOnly) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    nk::Model m = BuildModel(/*soft_mu=*/0.8f, /*fluid_mu=*/0.0f);
    const uint32_t rows = m.capacities.max_rows_per_env;
    const uint32_t particles = m.capacities.particles_per_env;

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());
    ASSERT_TRUE(StepToSolve(w));
    const Solved s = Download(w, rows, particles);
    ASSERT_TRUE(s.ok);

    const ContactLambda cloth = SumLambda(s, rows, /*particle_id=*/0u);  // soft slice.
    const ContactLambda fluid = SumLambda(s, rows, /*particle_id=*/1u);  // fluid slice.
    const float cloth_vx = s.particle_vel[0].x, cloth_vz = s.particle_vel[0].z;
    const float fluid_vx = s.particle_vel[1].x, fluid_vz = s.particle_vel[1].z;

    std::fprintf(stderr,
        "[body-particle-friction] cloth: rows=%u normalL=%.6f frictionL=%.6f "
        "vx=%.6f vz=%.6f | fluid: rows=%u normalL=%.6f frictionL=%.6f vx=%.6f vz=%.6f\n",
        cloth.rows, cloth.normal, cloth.friction, cloth_vx, cloth_vz,
        fluid.rows, fluid.normal, fluid.friction, fluid_vx, fluid_vz);

    // BOTH contacts exist and BOTH normal rows brake the downward approach (the
    // non-penetration behavior is the same for cloth and fluid).
    ASSERT_GT(cloth.rows, 0u) << "no body<->particle row for the soft particle";
    ASSERT_GT(fluid.rows, 0u) << "no body<->particle row for the fluid particle";
    EXPECT_GT(cloth.normal, 0.0f) << "cloth normal row carried no push-out impulse";
    EXPECT_GT(fluid.normal, 0.0f) << "fluid normal row carried no push-out impulse";
    EXPECT_GT(cloth_vz, kNormalVz + 1.0e-3f) << "cloth normal row did not brake -z";
    EXPECT_GT(fluid_vz, kNormalVz + 1.0e-3f) << "fluid normal row did not brake -z";

    // CLOTH grips: the friction pyramid carried real tangential impulse and the
    // tangential velocity is bounded well below the +x it entered with.
    EXPECT_GT(cloth.friction, 0.0f) << "cloth friction spokes carried no impulse";
    EXPECT_LT(cloth_vx, kTangentialVx - 1.0e-2f)
        << "finite-mu cloth did not drag/bound the tangential velocity";

    // FLUID slides: the friction bound collapsed to ~0, so the tangential velocity
    // is essentially unchanged and the friction impulse is negligible.
    EXPECT_NEAR(fluid.friction, 0.0f, 1.0e-6f)
        << "zero-mu fluid still developed a tangential friction impulse";
    EXPECT_NEAR(fluid_vx, kTangentialVx, 1.0e-3f)
        << "zero-mu fluid braked tangentially (it should slide freely)";

    // The ASYMMETRY is mu (data): same geometry, the cloth's tangential velocity is
    // strictly more arrested than the fluid's, and the cloth carries strictly more
    // friction impulse.
    EXPECT_LT(cloth_vx, fluid_vx - 1.0e-2f)
        << "cloth must grip more than fluid (the whole point of per-system mu)";
    EXPECT_GT(cloth.friction, fluid.friction)
        << "cloth must carry more friction impulse than fluid";
}

// CONTROL: give the FLUID system a finite mu and it grips too (so the slide above is
// the fluid mu = 0, NOT something structural about the fluid slice/index).
TEST(BodyParticleFriction, FluidGripsWhenGivenFiniteMu) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    nk::Model m = BuildModel(/*soft_mu=*/0.8f, /*fluid_mu=*/0.8f);  // fluid mu now finite.
    const uint32_t rows = m.capacities.max_rows_per_env;
    const uint32_t particles = m.capacities.particles_per_env;

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());
    ASSERT_TRUE(StepToSolve(w));
    const Solved s = Download(w, rows, particles);
    ASSERT_TRUE(s.ok);

    const ContactLambda fluid = SumLambda(s, rows, /*particle_id=*/1u);
    const float fluid_vx = s.particle_vel[1].x;
    std::fprintf(stderr,
        "[body-particle-friction] fluid@mu=0.8: frictionL=%.6f vx=%.6f\n",
        fluid.friction, fluid_vx);

    EXPECT_GT(fluid.friction, 0.0f)
        << "finite-mu fluid carried no friction impulse (slide was structural, a bug)";
    EXPECT_LT(fluid_vx, kTangentialVx - 1.0e-2f)
        << "finite-mu fluid did not grip tangentially";
}
