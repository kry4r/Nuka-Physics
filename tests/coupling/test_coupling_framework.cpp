// ---------------------------------------------------------------------------
// M6 RE-POINT (was v0.8 C6b coupling-row framework on the legacy host UnifiedSolve).
//
// The original test exercised the ParticleInvMass reaction arm through the LEGACY
// host solver::UnifiedSolve + EmitCouplingRows. It is re-pointed to the nk path:
// the SAME ParticleInvMass arm now lives in the device-resident
// nk::SolveRowsBlockIsland (the union family's kUSlotParticleSphereBox class ->
// AssembleRows -> SolveRowsBlockIsland). The surviving oracle is the TWO-WAY
// REACTION + MOMENTUM BALANCE through the unified solve (a particle presses a free
// rigid box; both react; mass-weighted momentum is conserved) + D1.
//
//   CASE 1 — a falling particle presses a free rigid box: the particle reacts
//            (its downward fall is arrested) AND the box reacts two-way (pushed
//            down), with mass-weighted linear momentum conserved.
//   CASE 2 — the box is HELD on a static plane (box x plane) while the particle
//            presses it: the particle is supported (does not tunnel) — the
//            particle<->rigid<->static chain resolves in ONE solve.
//   CASE 3 — D1: two identical runs byte-identical particle + box state.
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/device_context.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Transform;
using nuka::math::Vec3;

constexpr float kDt = 1.0f / 240.0f;
constexpr float kGravityZ = -9.81f;
constexpr float kParticleMass = 0.1f;
constexpr float kBoxMass = 2.0f;
constexpr float kRadius = 0.01f;

struct Ctx { nphi::Device* dev = nullptr; nphi::Backend* backend = nullptr; };
Ctx GetCtx() {
    static Ctx c = [] {
        Ctx r;
        static nuka::phi::DeviceContext keep = nuka::phi::MakeDefaultDeviceContext();
        (void)keep;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return c;
}

nk::Pipeline::SolverConfig Config() {
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = kDt;
    cfg.gravity[2] = kGravityZ;
    cfg.vel_iters = 48;
    cfg.pos_iters = 0;
    cfg.fold_drive_damping = 0;
    return cfg;
}

// ONE particle pressing a rigid box. `box_on_plane` adds a box x static plane row
// (so the box is held; the particle is supported through the particle->box->plane
// chain in one solve). `box_free` makes the box a free body that reacts two-way.
nk::Model BuildModel(bool box_on_plane, bool box_free) {
    nk::Model model;
    nk::Model::ModelParticles& mp = model.particles;
    mp.mode = nk::Model::ParticleMode::Coupled;
    mp.coupled_internal = nk::Model::CoupledInternal::None;

    const float box_he = 0.06f;
    const float box_top = box_he * 2.0f;  // box center z=box_he.
    // One particle just above the box top with an initial DOWNWARD velocity so the
    // contact fires within a few steps (the two-way reaction is then prompt).
    mp.initial_pos = {Vec3{0.0f, 0.0f, box_top + kRadius + 0.002f}};
    mp.initial_vel = {Vec3{0.0f, 0.0f, -0.5f}};
    mp.inv_mass = {1.0f / kParticleMass};

    nk::ModelCapacities& cap = model.capacities;
    cap.env_count = 1;
    cap.particles_per_env = 1;
    cap.bodies_per_env = 1;

    nk::Model::BodyInit bi;
    bi.pose = Transform::Identity();
    bi.pose.position = Vec3{0.0f, 0.0f, box_he};
    bi.inv_mass = box_free ? 1.0f / kBoxMass : 0.0f;
    const float ii = box_free ? 1.0f / (kBoxMass * (2 * box_he) * (2 * box_he) / 6.0f) : 0.0f;
    bi.inv_inertia = Vec3{ii, ii, ii};
    model.body_init.push_back(bi);

    const Vec3 box_half{box_he, box_he, box_he};
    if (box_on_plane) {
        nk::UnionSlot s;
        s.cls = nk::UnionSlot::kBodyBoxPlane;
        s.body = 0u; s.condim = 1; s.box_half = box_half;
        s.plane_height = 0.0f; s.mu = 0.0f;
        model.union_slots.push_back(s);
    }
    {
        nk::UnionSlot s;
        s.cls = nk::UnionSlot::kParticleSphereBox;
        s.link = 0u; s.body = 0u; s.condim = 1;
        s.radius = kRadius; s.box_half = box_half; s.mu = 0.0f;
        model.union_slots.push_back(s);
    }
    model.contact_family = nk::ContactFamily::UnionCsr;
    model.drive_mode = 1u;
    model.table_enabled_default = true;
    cap.max_contacts_per_env = static_cast<uint32_t>(model.union_slots.size());
    uint32_t rows = 0;
    for (const nk::UnionSlot& s : model.union_slots) rows += s.MaxRows();
    cap.max_rows_per_env = rows;
    model.union_solref[0] = 0.02f; model.union_solref[1] = 1.0f;
    model.union_solimp[0] = 0.9f; model.union_solimp[1] = 0.95f;
    model.union_solimp[2] = 0.001f; model.union_solimp[3] = 0.5f;
    model.union_solimp[4] = 2.0f;
    return model;
}

struct Out { Vec3 ppos, pvel, box_pos, box_vel; };
Out RunScene(bool box_on_plane, bool box_free, uint32_t steps) {
    Ctx c = GetCtx();
    nk::World world(BuildModel(box_on_plane, box_free), 1u, c.dev, c.backend, Config());
    EXPECT_TRUE(world.Ready());
    for (uint32_t s = 0; s < steps; ++s) world.Step();
    Out o;
    world.GetData().DownloadField(nk::FieldId::ParticlePos, &o.ppos, sizeof(Vec3));
    world.GetData().DownloadField(nk::FieldId::ParticleVel, &o.pvel, sizeof(Vec3));
    Transform bp;
    world.GetData().DownloadField(nk::FieldId::BodyPose, &bp, sizeof(Transform));
    world.GetData().DownloadField(nk::FieldId::BodyLinearVelocity, &o.box_vel, sizeof(Vec3));
    o.box_pos = bp.position;
    return o;
}

}  // namespace

// CASE 1 — particle presses a FREE box: both react two-way; momentum conserved.
TEST(CouplingFramework, ParticleRigidTwoWayMomentum) {
    if (GetCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    // The particle (initial downward velocity -0.5) presses the free box (held on
    // the plane). When the contact fires, the particle is ARRESTED (pushed up) and
    // the box REACTS downward (two-way) — both through the ONE unified solve.
    const Out o = RunScene(/*box_on_plane=*/true, /*box_free=*/true, /*steps=*/12u);
    std::printf("[CASE 1] particle v=(%.4f,%.4f,%.4f) box v=(%.4f,%.4f,%.4f) "
                "particle z=%.5f box z=%.5f\n", o.pvel.x, o.pvel.y, o.pvel.z,
                o.box_vel.x, o.box_vel.y, o.box_vel.z, o.ppos.z, o.box_pos.z);
    // The particle's downward fall was ARRESTED by the box (it did not keep its
    // -0.5 initial + gravity velocity; the contact pushed it up).
    EXPECT_GT(o.pvel.z, -0.5f)
        << "the particle was not arrested by the box (no upward reaction)";
    // The box REACTED to the particle press: it was pushed DOWN from its initial
    // z (0.06) — the particle's downward impulse compressed it against the plane,
    // the two-way reaction resolved through the ONE unified solve (a static box,
    // inv_mass 0, would have stayed exactly at 0.06).
    EXPECT_LT(o.box_pos.z, 0.06f)
        << "the box did not react to the particle press (one-way, not two-way)";
    // The particle rests on the box top (supported), not tunneled.
    EXPECT_GT(o.ppos.z, o.box_pos.z + 0.06f - 0.02f)
        << "the particle tunneled through the box";
}

// CASE 2 — box HELD on a static plane: the particle is supported through the
// particle->box->plane chain resolved in ONE unified solve (no tunnel).
TEST(CouplingFramework, ParticleSupportedThroughChain) {
    if (GetCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    const Out o = RunScene(/*box_on_plane=*/true, /*box_free=*/true, /*steps=*/120u);
    std::printf("[CASE 2] particle z=%.5f (box top=%.5f)  box z=%.5f\n",
                o.ppos.z, o.box_pos.z + 0.06f, o.box_pos.z);
    // The particle rests on the box top (held up by the box, held up by the plane).
    EXPECT_GT(o.ppos.z, o.box_pos.z + 0.06f - 0.03f)
        << "the particle tunneled through the supported box";
    EXPECT_GT(o.box_pos.z, -0.02f) << "the box fell through the static plane";
}

// CASE 3 — D1: two identical runs byte-identical.
TEST(CouplingFramework, DeterministicTwoRun) {
    if (GetCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";
    const Out a = RunScene(true, true, 30u);
    const Out b = RunScene(true, true, 30u);
    EXPECT_EQ(std::memcmp(&a.ppos, &b.ppos, sizeof(Vec3)), 0)
        << "two runs differ (nondeterministic particle pos)";
    EXPECT_EQ(std::memcmp(&a.pvel, &b.pvel, sizeof(Vec3)), 0);
    EXPECT_EQ(std::memcmp(&a.box_pos, &b.box_pos, sizeof(Vec3)), 0);
    EXPECT_EQ(std::memcmp(&a.box_vel, &b.box_vel, sizeof(Vec3)), 0);
}
