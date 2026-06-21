// ---------------------------------------------------------------------------
// Body<->particle CONTACT BUDGET disjointness — the additive-reserve gate.
//
// A cooked scene with a static ground box, a dynamic box resting on it (a real
// body<->body contact in the rigid slot sub-range), AND many particles (>= 64, so
// the per-particle reserve far exceeds the rigid candidate budget) is stepped on
// the ONE general PairDriven path. The cook grows max_contacts_per_env additively
// (rigid_base + particles_per_env*kBodyParticleContactSlotsPerParticle) so the
// rigid candidate slots [0, rigid_cap) and the body<->particle slots [rigid_cap,
// total) are DISJOINT. Before the additive reserve, the particle slot base clamped
// to 0 (reserve >> rigid budget), so the particle narrowphase OVERWROTE the rigid
// body<->body contact -> silent corruption.
//
// Asserts (read right after SolveRowsBlockIsland, before the finalize compose):
//   (a) the rigid body<->body contact SURVIVES: a row with BOTH sides bodies and a
//       movable (rigid, not static) side carries lambda > 0 — it was NOT clobbered;
//   (b) the body<->particle contacts ALSO exist in the SAME step: at least one row
//       carries a kNkSideParticle side with lambda > 0;
//   (c) the rigid + particle slot sub-ranges do not overlap: every particle row's
//       slot index >= rigid_cap, every rigid-body row's slot index < rigid_cap.
//   A separate case drives one particle into MANY surrounding boxes (more body
//   contacts than its reserved slots) and asserts kEnvStatusPairOverflow fires
//   LOUD (the drop is never silent).
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

#include "collision/cross_system_query.hpp"  // kBodyParticleContactSlotsPerParticle
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"
#include "phi/backend.hpp"
#include "scene/cook/cook_to_model.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
namespace cook = nuka::scene::cook;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr uint32_t kKindBox = 2u;  // collision::ShapeKind Box
constexpr uint32_t kSlotsPerParticle =
    nuka::collision::gpu::kBodyParticleContactSlotsPerParticle;

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
    cfg.vel_iters = 32u;
    cfg.pos_iters = 0u;   // velocity-only so the row lambda is the sole correction.
    cfg.max_pairs = 64u;
    return cfg;
}

void AddBox(nk::Model& m, const Vec3& pos, float half, float inv_mass,
            int32_t body_id) {
    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = pos;
    bi.inv_mass = inv_mass;
    bi.inv_inertia = inv_mass > 0.0f ? Vec3{inv_mass, inv_mass, inv_mass}
                                     : Vec3{0, 0, 0};
    m.body_init.push_back(bi);
    nk::Model::PairDrivenShape sh;
    sh.kind = kKindBox;
    sh.params[0] = half; sh.params[1] = half; sh.params[2] = half;
    sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
    sh.body_id = body_id; sh.group = 0u;
    m.shape_table_rows.push_back(sh);
}

// The cook's rigid candidate budget: (dynamic + 1 static) * kCandidatePairsPerColl.
uint32_t RigidBudget(uint32_t collidables) { return collidables * 4u; }

// A cooked scene: static ground box (body 0) + a dynamic box overlapping it (body
// 1, a real body<->body contact) + `nparts` soft particles dropped onto the ground
// top. The rigid budget is set the way CookToModel sizes it; CookXpbdParticles then
// grows it additively for the particle reserve.
nk::Model BuildBudgetModel(uint32_t nparts) {
    nk::Model m;
    // Body 0: static ground box, top face at z=0.5.
    AddBox(m, Vec3{0.0f, 0.0f, 0.0f}, 0.5f, 0.0f, 0);
    // Body 1: a dynamic box penetrating the ground top slightly (a body<->body
    // contact). Half 0.1, centred at z=0.58 -> bottom 0.48 < 0.5 (0.02 overlap).
    AddBox(m, Vec3{0.0f, 0.0f, 0.58f}, 0.1f, 1.0f / 1.0f, 1);

    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    // Rigid budget as the cook sizes it (dynamic bodies + 1 static) * 4.
    cap.max_contacts_per_env = RigidBudget(bodies);   // 2 collidables -> 8
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;

    // `nparts` soft particles in a small raft just above the ground top, each
    // approaching downward. No internal constraints (free point masses) so the
    // body<->particle row is the only thing touching particle_vel before finalize.
    cook::XpbdCookInput soft;
    const uint32_t side = 8u;  // 8x8 == 64 raft (>= 64 so reserve >> rigid budget).
    for (uint32_t iy = 0; iy < side; ++iy) {
        for (uint32_t ix = 0; ix < side && soft.positions.size() < nparts; ++ix) {
            const float x = -0.3f + 0.08f * static_cast<float>(ix);
            const float y = -0.3f + 0.08f * static_cast<float>(iy);
            soft.positions.push_back(Vec3{x, y, 0.515f});  // bottom near the top face.
        }
    }
    soft.velocities.assign(soft.positions.size(), Vec3{0.0f, 0.0f, -1.0f});
    soft.inv_mass.assign(soft.positions.size(), 1.0f / 0.01f);
    soft.solver_iterations = 1u;
    cook::CookXpbdParticles(m, 1u, soft);
    // A particle is a SPHERE of d_min/2 on the ONE path; radius 0.02 -> d_min 0.04.
    m.particles.pp_contact_d_min = 0.04f;
    return m;
}

// Stop right after SolveRowsBlockIsland (before ParticleFinalize overwrites vel).
bool StepToSolve(nk::World& w) {
    for (const nphi::OpCall& call : w.GetPipeline().Calls()) {
        if (w.DispatchOp(call.op, call.params) != nphi::Status::Ok) return false;
        if (call.op == nphi::NkOp::SolveRowsBlockIsland) return true;
    }
    return true;
}

struct Solved {
    bool ok = false;
    std::vector<float> lambda;   // [rows]
    std::vector<float> urows;    // [rows * 32]
    std::vector<uint32_t> env_status;  // [env]
};

Solved Download(nk::World& w, uint32_t rows, uint32_t envs) {
    Solved s;
    s.lambda.assign(rows, 0.0f);
    s.urows.assign(static_cast<size_t>(rows) * 32u, 0.0f);
    s.env_status.assign(envs, 0u);
    nk::Data& d = w.GetData();
    s.ok = d.DownloadField(nk::FieldId::Lambda, s.lambda.data(),
                           s.lambda.size() * sizeof(float)) &&
           d.DownloadField(nk::FieldId::Urows, s.urows.data(),
                           s.urows.size() * sizeof(float)) &&
           d.DownloadField(nk::FieldId::EnvStatus, s.env_status.data(),
                           s.env_status.size() * sizeof(uint32_t));
    return s;
}

// NkRow packs to 32 f32: [0]=flags [16]=a.kind [17]=a.index [24]=b.kind [25]=b.index.
struct RowSides { uint32_t a_kind, b_kind, a_index, b_index; bool active; };
RowSides DecodeRow(const std::vector<float>& urows, uint32_t row) {
    const float* r = urows.data() + static_cast<size_t>(row) * 32u;
    auto u = [&](int i) { uint32_t v; std::memcpy(&v, &r[i], 4); return v; };
    RowSides s;
    s.active = (u(0) & 1u) != 0u;
    s.a_kind = u(16); s.a_index = u(17);
    s.b_kind = u(24); s.b_index = u(25);
    return s;
}

}  // namespace

// (a)+(b)+(c): the rigid body<->body contact survives the particle writes, the
// body<->particle contacts also exist, and the two slot sub-ranges are disjoint.
TEST(BodyParticleBudget, RigidContactSurvivesManyParticles) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    nk::Model m = BuildBudgetModel(64u);

    // The cook must have GROWN the budget: total == rigid_base + nparts*slots.
    const uint32_t parts = m.capacities.particles_per_env;
    ASSERT_EQ(parts, 64u);
    const uint32_t rigid_cap = RigidBudget(2u);                 // 8
    const uint32_t total = m.capacities.max_contacts_per_env;
    EXPECT_EQ(total, rigid_cap + parts * kSlotsPerParticle)
        << "cook did not grow max_contacts_per_env additively";
    EXPECT_EQ(m.capacities.max_rows_per_env, total * nk::kPairDrivenRowsPerSlot);

    const uint32_t rows_per_env = m.capacities.max_rows_per_env;
    const uint32_t rows_per_slot = nk::kPairDrivenRowsPerSlot;

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());
    ASSERT_TRUE(StepToSolve(w));
    const Solved s = Download(w, rows_per_env, 1u);
    ASSERT_TRUE(s.ok);

    uint32_t rigid_body_rows = 0u;  // both sides bodies, one movable (rigid).
    float rigid_max_lambda = 0.0f;
    uint32_t particle_rows = 0u;
    float particle_max_lambda = 0.0f;
    uint32_t rigid_in_particle_range = 0u;  // a rigid-body row in [rigid_cap, total).
    uint32_t particle_in_rigid_range = 0u;  // a particle row in [0, rigid_cap).
    for (uint32_t row = 0u; row < rows_per_env; ++row) {
        const RowSides rs = DecodeRow(s.urows, row);
        if (!rs.active) continue;
        const uint32_t slot = row / rows_per_slot;
        const bool a_part = rs.a_kind == nk::kNkSideParticle;
        const bool b_part = rs.b_kind == nk::kNkSideParticle;
        if (a_part || b_part) {
            ++particle_rows;
            particle_max_lambda = std::max(particle_max_lambda, s.lambda[row]);
            if (slot < rigid_cap) ++particle_in_rigid_range;
        } else {
            // A body<->body row: side kinds are Rigid/Static (movable side present).
            const bool a_rigid = rs.a_kind == nk::kNkSideRigid;
            const bool b_rigid = rs.b_kind == nk::kNkSideRigid;
            if (a_rigid || b_rigid) {
                ++rigid_body_rows;
                rigid_max_lambda = std::max(rigid_max_lambda, s.lambda[row]);
                if (slot >= rigid_cap) ++rigid_in_particle_range;
            }
        }
    }
    std::fprintf(stderr,
                 "[budget] rigid_cap=%u total=%u rigid_body_rows=%u "
                 "rigid_max_lambda=%.6f particle_rows=%u particle_max_lambda=%.6f "
                 "rigid_in_particle_range=%u particle_in_rigid_range=%u\n",
                 rigid_cap, total, rigid_body_rows, rigid_max_lambda, particle_rows,
                 particle_max_lambda, rigid_in_particle_range, particle_in_rigid_range);

    // (a) the rigid body<->body contact survived (it would be clobbered pre-fix).
    EXPECT_GT(rigid_body_rows, 0u) << "rigid body<->body contact was clobbered";
    EXPECT_GT(rigid_max_lambda, 0.0f)
        << "the surviving rigid contact carries no push-out impulse";
    // (b) the body<->particle contacts ALSO exist in the SAME step.
    EXPECT_GT(particle_rows, 0u) << "no body<->particle coupling rows emitted";
    EXPECT_GT(particle_max_lambda, 0.0f) << "particle coupling carries no impulse";
    // (c) the two sub-ranges are disjoint (rigid below rigid_cap, particle above).
    EXPECT_EQ(rigid_in_particle_range, 0u)
        << "a rigid body row landed in the particle slot sub-range";
    EXPECT_EQ(particle_in_rigid_range, 0u)
        << "a particle row landed in the rigid slot sub-range (overlap = corruption)";
}

// LOUD overflow: one particle surrounded by more colliding boxes than its reserved
// body-contact slots must OR kEnvStatusPairOverflow (the drop is never silent).
TEST(BodyParticleBudget, ParticleOverContactSlotsFlagsOverflowLoud) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    nk::Model m;
    // Surround the origin with kSlotsPerParticle+3 small boxes all overlapping ONE
    // particle at the origin (more body contacts than the particle's reserved slots).
    const uint32_t nboxes = kSlotsPerParticle + 3u;
    const float r = 0.05f;
    for (uint32_t i = 0; i < nboxes; ++i) {
        const float ang = 6.2831853f * static_cast<float>(i) /
                          static_cast<float>(nboxes);
        AddBox(m, Vec3{0.06f * std::cos(ang), 0.06f * std::sin(ang), 0.0f}, r, 0.0f,
               static_cast<int32_t>(i));
    }
    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_contacts_per_env = RigidBudget(bodies);
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;

    cook::XpbdCookInput soft;
    soft.positions = {Vec3{0.0f, 0.0f, 0.0f}};  // one particle at the centre.
    soft.velocities = {Vec3::Zero()};
    soft.inv_mass = {1.0f / 0.01f};
    soft.solver_iterations = 1u;
    cook::CookXpbdParticles(m, 1u, soft);
    m.particles.pp_contact_d_min = 0.16f;  // radius 0.08 -> overlaps all the boxes.

    const uint32_t rows = m.capacities.max_rows_per_env;
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());
    ASSERT_TRUE(StepToSolve(w));
    const Solved s = Download(w, rows, 1u);
    ASSERT_TRUE(s.ok);

    std::fprintf(stderr, "[budget-overflow] boxes=%u slots/particle=%u env_status=0x%x\n",
                 nboxes, kSlotsPerParticle, s.env_status[0]);
    EXPECT_NE(s.env_status[0] & nphi::kEnvStatusPairOverflow, 0u)
        << "a particle with more body contacts than its slots must flag overflow LOUD";
}

// The SoftFluid both-slices cook grows the budget for the FULL [soft | fluid] count
// from the rigid base ONCE (the inner soft sub-cook's partial growth is overridden,
// no double-count). Cook-only (no CUDA backend).
TEST(BodyParticleBudget, SoftFluidCookGrowsAdditivelyNoDoubleCount) {
    nk::Model m;
    AddBox(m, Vec3{0.0f, 0.0f, 0.0f}, 0.5f, 0.0f, 0);   // a static ground body.
    AddBox(m, Vec3{0.0f, 0.0f, 0.58f}, 0.1f, 1.0f, 1);  // a dynamic body.
    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_contacts_per_env = RigidBudget(bodies);  // 8
    cap.max_rows_per_env = cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot;

    cook::XpbdCookInput soft;
    soft.positions = {Vec3{0.0f, 0.0f, 0.52f}, Vec3{0.06f, 0.0f, 0.52f},
                      Vec3{0.0f, 0.06f, 0.52f}};  // 3 soft particles.
    soft.velocities.assign(3, Vec3::Zero());
    soft.inv_mass.assign(3, 1.0f / 0.01f);
    cook::PbfCookInput fluid;
    const float spacing = 0.03f, h = spacing * 1.5f;
    fluid.positions = {Vec3{0.2f, 0.0f, 0.6f}, Vec3{0.23f, 0.0f, 0.6f}};  // 2 fluid.
    fluid.velocities.assign(2, Vec3::Zero());
    fluid.particle_mass = 1000.0f * spacing * spacing * spacing;
    fluid.rest_density = 1000.0f; fluid.support_radius = h; fluid.iters = 2u;
    fluid.grid_min = Vec3{0.0f, -0.1f, 0.5f};
    fluid.grid_dims[0] = 4; fluid.grid_dims[1] = 4; fluid.grid_dims[2] = 4;

    cook::CookSoftFluidParticles(m, 1u, soft, fluid);
    EXPECT_EQ(m.particles.mode, nk::Model::ParticleMode::SoftFluid);
    EXPECT_EQ(cap.particles_per_env, 5u);  // 3 soft + 2 fluid.
    // total == rigid_base + (n_soft + n_fluid)*slots, NOT rigid + (n_soft + total)*slots.
    EXPECT_EQ(cap.max_contacts_per_env, 8u + 5u * kSlotsPerParticle)
        << "SoftFluid cook double-counted the soft slice in the contact budget";
    EXPECT_EQ(cap.max_rows_per_env,
              cap.max_contacts_per_env * nk::kPairDrivenRowsPerSlot);
}
