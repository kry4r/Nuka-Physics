// ---------------------------------------------------------------------------
// Body/artic <-> soft/fluid particle COUPLING — the body↔particle row gate.
//
// A static collidable box overlapping ONE particle, stepped on the ONE general
// PairDriven path: the new body<->particle narrowphase detects the overlap, treats
// the particle as a sphere of its radius, and emits a COUPLED two-sided NkRow whose
// PARTICLE side flows through the SAME EmitPairDrivenRows -> ComputeRowMeff ->
// SolveRowsBlockIsland the body rows use. No special coupler: a particle is a
// sphere collidable on the one path.
//
// Asserts (observed at the SOLVE stage — finalize would overwrite particle_vel, a
// KNOWN trap fixed in a separate step, so we read right after SolveRowsBlockIsland):
//   (a) a NkRow is emitted with exactly one side kind == kNkSideParticle and
//       index == the GLOBAL particle id;
//   (b) the normal multiplier lambda > 0 AND the solve braked the particle's
//       approach velocity toward non-penetration (particle_vel.z rose toward 0);
//   (c) two runs of the detect->narrowphase->assemble->solve are BYTE-IDENTICAL
//       (the coupling candidate stream + rows + solved lambda).
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
    cfg.contact_margin = 0.0f;
    cfg.vel_iters = 32u;
    cfg.pos_iters = 0u;   // velocity-only so the row lambda is the sole correction.
    cfg.max_pairs = 32u;
    return cfg;
}

// One immovable box collidable + one particle just touching its top face, moving
// DOWN into it. The static box is body 0; a second tiny immovable box is body 1 so
// bodies_per_env >= 2 builds the arena LBVH the body<->particle op traverses.
nk::Model BuildModel() {
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
    // Box top face at z = 0.10. Particle of radius 0.02 centred at z = 0.115 (its
    // bottom at 0.095 -> 0.005 below the face -> a shallow overlap).
    add_static_box(Vec3{0.0f, 0.0f, 0.0f}, 0.10f, 0);
    add_static_box(Vec3{1.0f, 0.0f, 0.0f}, 0.05f, 1);  // far away (LBVH-only filler).

    nk::ModelCapacities& cap = m.capacities;
    const uint32_t bodies = static_cast<uint32_t>(m.body_init.size());
    cap.env_count = 1u;
    cap.bodies_per_env = bodies;
    cap.max_bodies_total = bodies;
    cap.max_contacts_per_env = 16u;
    cap.max_rows_per_env = 16u * nk::kPairDrivenRowsPerSlot;
    cap.particles_per_env = 1u;
    m.contact_family = nk::ContactFamily::PairDriven;
    m.filter_cross_env = true;

    // The particle system: ONE free point mass moving down into the box. Xpbd mode
    // with no internal constraints so the predict only folds gravity into position
    // (velocity untouched) and the body<->particle contact row is the only thing
    // touching particle_vel before finalize.
    nk::Model::ModelParticles& mp = m.particles;
    mp.mode = nk::Model::ParticleMode::Xpbd;
    mp.initial_pos = {Vec3{0.0f, 0.0f, 0.115f}};
    mp.initial_vel = {Vec3{0.0f, 0.0f, -1.0f}};   // approaching the box top, downward.
    mp.inv_mass = {1.0f / 0.01f};                 // 10 g particle.
    mp.xpbd_iters = 1;
    // A particle is a SPHERE of d_min/2 on the ONE path; radius 0.02 -> d_min 0.04.
    mp.pp_contact_d_min = 0.04f;
    return m;
}

// Run the pipeline ops in the fixed order but STOP right after SolveRowsBlockIsland
// (before ParticleFinalize overwrites particle_vel). Mirrors World::Step but truncated.
bool StepToSolve(nk::World& w) {
    for (const nphi::OpCall& call : w.GetPipeline().Calls()) {
        if (w.DispatchOp(call.op, call.params) != nphi::Status::Ok) return false;
        if (call.op == nphi::NkOp::SolveRowsBlockIsland) return true;
    }
    return true;
}

struct Solved {
    bool ok = false;
    std::vector<float> lambda;          // [rows]
    std::vector<float> urows;           // [rows * 32]
    std::vector<Vec3>  particle_vel;    // [particles]
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

// Decode the per-row side kinds from the downloaded urows. NkRow packs to 32 f32
// (the static_asserts in nk_row.hpp pin the layout): [0]=flags [6]=lower [7]=upper,
// side a at [16]=kind [17]=index, side b at [24]=kind [25]=index.
struct RowSides { uint32_t a_kind, b_kind, a_index, b_index; bool active; float lower, upper; };
RowSides DecodeRow(const std::vector<float>& urows, uint32_t row) {
    const float* r = urows.data() + static_cast<size_t>(row) * 32u;
    auto u = [&](int i) { uint32_t v; std::memcpy(&v, &r[i], 4); return v; };
    RowSides s;
    s.active = (u(0) & 1u) != 0u;
    s.lower = r[6]; s.upper = r[7];
    s.a_kind = u(16); s.a_index = u(17);
    s.b_kind = u(24); s.b_index = u(25);
    return s;
}

}  // namespace

// (a)+(b): the body<->particle row carries a kNkSideParticle side with the global
// particle id, lambda > 0, and the solve braked the particle's downward approach.
TEST(BodyParticleCoupling, ParticleSideRowSolvesAndBrakes) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    nk::Model m = BuildModel();
    const uint32_t rows = m.capacities.max_rows_per_env;
    const uint32_t particles = m.capacities.particles_per_env;

    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());
    ASSERT_TRUE(StepToSolve(w));

    const Solved s = Download(w, rows, particles);
    ASSERT_TRUE(s.ok);

    // (a) exactly one row side resolves to kNkSideParticle with index == global id 0.
    uint32_t particle_rows = 0u;
    uint32_t active_normal_with_particle = 0u;
    float max_lambda = 0.0f;
    for (uint32_t row = 0u; row < rows; ++row) {
        const RowSides rs = DecodeRow(s.urows, row);
        if (!rs.active) continue;
        const bool a_part = rs.a_kind == nk::kNkSideParticle;
        const bool b_part = rs.b_kind == nk::kNkSideParticle;
        if (a_part || b_part) {
            ++particle_rows;
            const uint32_t pidx = a_part ? rs.a_index : rs.b_index;
            EXPECT_EQ(pidx, 0u) << "particle side must carry the GLOBAL particle id";
            // The OTHER side must be the body (not also a particle): single-particle row.
            EXPECT_FALSE(a_part && b_part) << "row has two particle sides";
            // A normal row (upper == FLT_MAX) vs friction spoke (upper overridden 0).
            if (rs.upper > 1.0e30f) {
                ++active_normal_with_particle;
                max_lambda = std::max(max_lambda, s.lambda[row]);
            }
        }
    }
    std::fprintf(stderr,
                 "[body-particle] particle_rows=%u active_normal=%u max_lambda=%.6f "
                 "particle_vel.z=%.6f\n",
                 particle_rows, active_normal_with_particle, max_lambda,
                 s.particle_vel[0].z);
    EXPECT_GT(particle_rows, 0u) << "no kNkSideParticle row emitted by the coupling";
    EXPECT_GT(active_normal_with_particle, 0u) << "no active particle NORMAL row";

    // (b) the normal multiplier is positive (a real push-out impulse) AND the solve
    // braked the particle's downward approach velocity toward non-penetration.
    EXPECT_GT(max_lambda, 0.0f) << "particle normal lambda must be > 0 (push-out)";
    EXPECT_GT(s.particle_vel[0].z, -1.0f + 1.0e-4f)
        << "the contact row must reduce the particle's downward approach velocity";
}

// (c) two runs of detect->narrowphase->assemble->solve are byte-identical.
TEST(BodyParticleCoupling, CouplingStreamTwoRunByteIdentical) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();

    auto run = [&]() -> Solved {
        nk::Model m = BuildModel();
        const uint32_t rows = m.capacities.max_rows_per_env;
        const uint32_t particles = m.capacities.particles_per_env;
        nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
        if (!w.Ready() || !StepToSolve(w)) return {};
        return Download(w, rows, particles);
    };

    const Solved a = run();
    const Solved c = run();
    ASSERT_TRUE(a.ok && c.ok);

    auto same = [](const auto& x, const auto& y) {
        return x.size() == y.size() &&
               std::memcmp(x.data(), y.data(),
                           x.size() * sizeof(typename std::decay_t<decltype(x)>::value_type)) == 0;
    };
    EXPECT_TRUE(same(a.lambda, c.lambda)) << "coupling lambda differs across runs";
    EXPECT_TRUE(same(a.urows, c.urows)) << "coupling rows differ across runs";
    EXPECT_TRUE(same(a.particle_vel, c.particle_vel))
        << "solved particle_vel differs across runs";
}
