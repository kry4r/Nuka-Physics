// ---------------------------------------------------------------------------
// SoftFluid finalize composition — the body<->particle coupling impulse must
// SURVIVE a full World::Step.
//
// The contact solve writes a braking impulse into particle_vel BEFORE the
// SoftFluid finalize runs. The finalize reproduces the gravity+free-flight+
// projection velocity from the projected position AND composes the contact
// correction (v_now - v_pre) so the impulse persists, single-ownership, with no
// double count. Without the composition the finalize plainly recomputes
// v = (pos_proj - prev)/dt and silently overwrites the impulse, so the particle
// keeps falling and sinks through the floor.
//
// The model is a SoftFluid Model (the soft+fluid co-residence mode the render demo
// uses): ONE soft particle directly above a STATIC floor box, plus a tiny fluid
// blob parked far away (inert). A FULL World::Step runs the real pipeline INCLUDING
// SoftFluidFinalizeKernel. The assertion is on the SOFT particle: after the step it
// is BRAKED (downward velocity arrested) and does NOT sink through the floor.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
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
    cfg.vel_iters = 32u;
    cfg.pos_iters = 0u;
    cfg.max_pairs = 32u;
    return cfg;
}

// A SoftFluid Model: ONE soft particle just above a static box (body 0), a filler
// box (body 1) so the arena LBVH the body<->particle op traverses is non-trivial,
// and a tiny fluid blob parked far below so the SoftFluid mode carries a real fluid
// slice while staying out of the contact. n_soft = 1; the soft particle is index 0.
nk::Model BuildModel() {
    nk::Model m;
    auto add_static_box = [&](const Vec3& pos, float half, int32_t body_id) {
        nk::Model::BodyInit bi;
        bi.pose = Transform::Identity();
        bi.pose.position = pos;
        bi.inv_mass = 0.0f;
        bi.inv_inertia = Vec3{0, 0, 0};
        m.body_init.push_back(bi);
        nk::Model::PairDrivenShape sh;
        sh.kind = kKindBox;
        sh.params[0] = half; sh.params[1] = half; sh.params[2] = half;
        sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
        sh.body_id = body_id; sh.group = 0u;
        m.shape_table_rows.push_back(sh);
    };
    // Floor box top face at z = 0.10. Soft particle radius 0.02 centred at z = 0.115
    // (bottom at 0.095 -> 0.005 below the face -> a shallow overlap), moving DOWN.
    add_static_box(Vec3{0.0f, 0.0f, 0.0f}, 0.10f, 0);
    add_static_box(Vec3{1.0f, 0.0f, 0.0f}, 0.05f, 1);

    nk::Model::ModelParticles& mp = m.particles;
    mp.mode = nk::Model::ParticleMode::SoftFluid;
    mp.n_soft_particles = 1u;
    // Soft particle (index 0) above the box; two fluid particles parked far below.
    mp.initial_pos = {Vec3{0.0f, 0.0f, 0.115f},
                      Vec3{0.0f, 0.0f, -5.0f}, Vec3{0.03f, 0.0f, -5.0f}};
    mp.initial_vel = {Vec3{0.0f, 0.0f, -1.0f}, Vec3::Zero(), Vec3::Zero()};
    mp.inv_mass = {1.0f / 0.01f, 1.0f / 0.01f, 1.0f / 0.01f};
    mp.xpbd_iters = 1;
    // A particle is a SPHERE of d_min/2 on the ONE path; radius 0.02 -> d_min 0.04.
    mp.pp_contact_d_min = 0.04f;

    // The fluid slice's PBF params + uniform grid (covers only the parked blob).
    const float rho0 = 1000.0f, spacing = 0.03f, h = spacing * 1.5f;
    mp.pbf_rest_density = rho0;
    mp.pbf_support_radius = h;
    mp.pbf_particle_mass = rho0 * spacing * spacing * spacing;
    mp.pbf_cfm_epsilon = 1.0e-6f;
    mp.pbf_iters = 2u;
    mp.pbf_clamp_overdensity = true;
    mp.grid_min = Vec3{-0.1f, -0.1f, -5.2f};
    mp.grid_dims[0] = 4; mp.grid_dims[1] = 4; mp.grid_dims[2] = 4;
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

// A SoftFluid Model with the SAME static-box geometry but a FLUID particle (index
// >= n_soft) in shallow contact above the floor, the lone soft particle parked far
// below (inert). The fluid finalize branch must carry the contact displacement.
nk::Model BuildFluidContactModel() {
    nk::Model m;
    auto add_static_box = [&](const Vec3& pos, float half, int32_t body_id) {
        nk::Model::BodyInit bi;
        bi.pose = Transform::Identity();
        bi.pose.position = pos;
        bi.inv_mass = 0.0f;
        bi.inv_inertia = Vec3{0, 0, 0};
        m.body_init.push_back(bi);
        nk::Model::PairDrivenShape sh;
        sh.kind = kKindBox;
        sh.params[0] = half; sh.params[1] = half; sh.params[2] = half;
        sh.contype = 1u; sh.conaffinity = 1u; sh.sdf_grid = ~0u;
        sh.body_id = body_id; sh.group = 0u;
        m.shape_table_rows.push_back(sh);
    };
    add_static_box(Vec3{0.0f, 0.0f, 0.0f}, 0.10f, 0);
    add_static_box(Vec3{1.0f, 0.0f, 0.0f}, 0.05f, 1);

    nk::Model::ModelParticles& mp = m.particles;
    mp.mode = nk::Model::ParticleMode::SoftFluid;
    mp.n_soft_particles = 1u;
    // Soft particle (index 0) parked far below; the FLUID particle (index 1) just
    // above the floor top face (z=0.10), radius 0.02 (bottom at 0.095), moving DOWN.
    mp.initial_pos = {Vec3{0.0f, 0.0f, -5.0f}, Vec3{0.0f, 0.0f, 0.115f}};
    mp.initial_vel = {Vec3::Zero(), Vec3{0.0f, 0.0f, -1.0f}};
    mp.inv_mass = {1.0f / 0.01f, 1.0f / 0.01f};
    mp.xpbd_iters = 1;
    mp.pp_contact_d_min = 0.04f;  // sphere radius d_min/2 = 0.02.

    const float rho0 = 1000.0f, spacing = 0.03f, h = spacing * 1.5f;
    mp.pbf_rest_density = rho0;
    mp.pbf_support_radius = h;
    mp.pbf_particle_mass = rho0 * spacing * spacing * spacing;
    mp.pbf_cfm_epsilon = 1.0e-6f;
    mp.pbf_iters = 2u;
    mp.pbf_clamp_overdensity = true;
    // The grid AABB must enclose the in-contact fluid particle near z=0.10.
    mp.grid_min = Vec3{-0.1f, -0.1f, -0.05f};
    mp.grid_dims[0] = 4; mp.grid_dims[1] = 4; mp.grid_dims[2] = 4;
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

struct State { Vec3 pos; Vec3 vel; };

State StepOnceAndReadIdx(nk::World& w, uint32_t idx) {
    w.Step();  // FULL pipeline incl. SoftFluidFinalize.
    std::vector<Vec3> pos(w.GetModel().capacities.particles_per_env, Vec3::Zero());
    std::vector<Vec3> vel(pos.size(), Vec3::Zero());
    w.GetData().DownloadField(nk::FieldId::ParticlePos, pos.data(),
                              pos.size() * sizeof(Vec3));
    w.GetData().DownloadField(nk::FieldId::ParticleVel, vel.data(),
                              vel.size() * sizeof(Vec3));
    return State{pos[idx], vel[idx]};
}

State StepOnceAndRead(nk::World& w) { return StepOnceAndReadIdx(w, 0u); }

}  // namespace

// After a FULL World::Step the soft particle's downward approach must be arrested by
// the body<->particle coupling row surviving the SoftFluid finalize: its velocity
// rises well above the -1.0 m/s it entered with, and it does NOT sink through the
// box top face (z stays at/above the face minus the contact radius).
TEST(SoftFluidCouplingFinalize, SoftParticleBrakedOnStaticFloorAfterFullStep) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    nk::Model m = BuildModel();
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());

    const State s = StepOnceAndRead(w);
    std::fprintf(stderr,
                 "[soft-fluid-coupling] after Step: soft pos.z=%.6f vel.z=%.6f\n",
                 s.pos.z, s.vel.z);

    // BRAKED: the contact impulse arrested the downward approach (entered at -1.0).
    EXPECT_GT(s.vel.z, -1.0f + 1.0e-3f)
        << "the coupling impulse was overwritten by the SoftFluid finalize: the "
           "soft particle's downward velocity was NOT arrested after a full Step";
    // NON-PENETRATION: the soft particle did not sink through the box top (z=0.10).
    // Its sphere radius is d_min/2 = 0.02, so a braked particle stays near 0.10+.
    EXPECT_GT(s.pos.z, 0.10f - 0.02f)
        << "the soft particle sank through the static floor after a full Step";
}

// The SAME coupling contract for a FLUID particle (index >= n_soft). The fluid
// finalize branch must carry the contact displacement dv*dt onto the committed
// position, NOT just commit pos = predicted. The proof is committed-state
// CONSISTENCY: a particle whose committed velocity was braked by the contact row
// (v_committed > v_pre) must have a committed position that advanced by exactly
// v_committed*dt from the step-start position -- i.e. pos == p0 + v_committed*dt.
// The pre-fix `pos = predicted = p0 + v_pre*dt` lags by dv*dt and FAILS this.
TEST(SoftFluidCouplingFinalize, FluidParticleCarriesContactDisplacementAfterStep) {
    if (GetBackend().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    Backend b = GetBackend();
    nk::Model m = BuildFluidContactModel();
    nk::World w(std::move(m), 1u, b.dev, b.backend, Cfg());
    ASSERT_TRUE(w.Ready());

    constexpr float kDt = 1.0f / 240.0f;
    constexpr float kP0z = 0.115f;  // the fluid particle's step-start z (index 1).
    const State s = StepOnceAndReadIdx(w, 1u);

    // The pre-contact velocity the finalize saw (gravity+free-flight, no contact);
    // the pre-fix `pos = predicted` committed exactly p0 + v_pre*dt.
    std::vector<Vec3> v_pre(w.GetModel().capacities.particles_per_env, Vec3::Zero());
    w.GetData().DownloadField(nk::FieldId::ParticleVPre, v_pre.data(),
                              v_pre.size() * sizeof(Vec3));
    const float prefix_pos_z = kP0z + v_pre[1].z * kDt;   // what pos=predicted gives.
    const float implied_pos_z = kP0z + s.vel.z * kDt;     // post-fix expectation.
    std::fprintf(stderr,
                 "[soft-fluid-coupling] fluid after Step: committed pos.z=%.7f "
                 "vel.z=%.7f v_pre.z=%.7f | pre-fix(pos=predicted)=%.7f "
                 "post-fix(p0+v*dt)=%.7f dropped dv*dt=%.7e\n",
                 s.pos.z, s.vel.z, v_pre[1].z, prefix_pos_z, implied_pos_z,
                 implied_pos_z - prefix_pos_z);

    // BRAKED: the body<->particle row arrested the downward approach (entered -1.0),
    // so dv != 0 and the displacement-carry is observable (not a no-op dv==0 case).
    EXPECT_GT(s.vel.z, -1.0f + 1.0e-3f)
        << "the coupling impulse was overwritten: the fluid particle's downward "
           "velocity was NOT arrested after a full Step";
    // CONSISTENCY (THE FIX-1 PROOF): committed pos == step-start + v_committed*dt.
    // Pre-fix the committed pos == step-start + v_pre*dt (predicted), which is below
    // this by dv*dt; the |dv*dt| gap is the silently dropped contact displacement.
    EXPECT_NEAR(s.pos.z, implied_pos_z, 1.0e-5f)
        << "the SoftFluid fluid finalize dropped the contact displacement dv*dt: "
           "the committed position lags the committed velocity by dv*dt";
    // NON-PENETRATION: the braked fluid particle stays at/above the box top minus
    // its contact radius (d_min/2 = 0.02).
    EXPECT_GT(s.pos.z, 0.10f - 0.02f)
        << "the fluid particle sank through the static floor after a full Step";
}
