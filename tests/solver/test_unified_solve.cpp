// ---------------------------------------------------------------------------
// v0.8 C5a: end-to-end test of UnifiedSolve (src/solver/unified_solve.cpp) --
// the standalone compliant-contact solve entry. VALIDATED-NOT-WIRED.
// ---------------------------------------------------------------------------
// UnifiedSolve is a NEW entry alongside the legacy solver::SolveConstraints; it
// is NOT wired into the production world stepper (that flip + the golden
// re-baseline are C5c). These tests drive it directly, building manifolds in the
// test (like test_compliant_rows.cpp) -> EmitCompliantContactRows -> UnifiedSolve,
// and owning gravity + integration (UnifiedSolve is velocity-only for the
// compliant rows it drives, per unified_solve.hpp).
//
// PHYSICS CONVENTIONS the driver mirrors (both load-bearing -- see the C5a
// advisor notes recorded in v08-progress):
//   - aref is a reference ACCELERATION; the velocity-impulse PGS scales it by dt
//     (done in the kernel's compliant branch). The caller just supplies dt.
//   - aref's `vel` is the START-OF-STEP (pre-gravity) constraint velocity, as in
//     MuJoCo. Feeding the post-gravity velocity makes the damping term carry the
//     weight and pushes the equilibrium to a gap (contact chatter). So the driver
//     captures vel BEFORE applying gravity.
//
// What is validated:
//   (1) REST-START FORCE BALANCE: a box started at rest in slight penetration
//       settles with normal impulse lambda_n ~= m*|g|*dt (the contact supports
//       the weight) at a small POSITIVE steady penetration.
//   (2) STIFFNESS ORACLE: a stiffer solref (smaller timeconst) settles at a
//       SMALLER steady penetration. Exercises the headline C5a wiring --
//       Row.compliance_alpha (R) in the effective-mass denominator -- since the
//       steady penetration only tracks stiffness if R is wired.
//   (3) DROP STABILITY (generous): a dropped box does not blow up / sink through;
//       it ends near the surface with small velocity. (Transient stability only;
//       the precise oracles use the rest-start.)
//   (4) PYRAMID FRICTION: with the NEW unilateral coupled-pyramid bounds, a high-
//       mu sliding box is GRIPPED (vx -> 0) while a mu=0 box SLIDES freely.
//   (5) D1: two-run byte-exact; and N=32 cross-replica byte-identical to a single
//       env (the batched 32-box solve does not cross-contaminate).
//
// KNOWN LIMITATION (named, not a bug): a single velocity-solve-per-step has a
// stiffness ceiling -- the stiffest representable contact (timeconst == 2*dt, the
// REFSAFE floor) is only marginally stable under impact. Oracle contacts use
// timeconst >= ~4*dt. Raising the ceiling (sub-stepping / implicit position
// solve) is out of C5a scope.
// ---------------------------------------------------------------------------

#include "constraint/contact_manifold.hpp"
#include "constraint/row_builder.hpp"
#include "runtime/rigid/body_state.hpp"
#include "solver/unified_solve.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <span>
#include <vector>

using namespace nuka::constraint;
using nuka::runtime::rigid::BodyState;
namespace math = nuka::math;

namespace {

// Ground side handle: the sentinel static-body index (the scheduler skips it in
// conflict coloring and ValidBody() treats it as immovable -> zero reaction),
// matching solver::SolveUnitGroundContact's static-ground convention.
constexpr uint32_t kGroundHandle = kInvalidBodyIndex;

struct BoxParams {
    float half_extent = 0.5f;
    float inv_mass = 1.0f;     // m = 1
    float friction = 0.0f;
    float solref_tc = 0.02f;   // timeconst (>= ~4*dt for the oracle's stable band)
    float solref_dr = 1.0f;    // critically damped
    float dt = 0.005f;
    float gravity = -9.81f;    // signed (downward)
    uint32_t condim = 1u;      // 1 -> frictionless (no spokes); 3 -> 4 spokes
};

// Build a 1-point ground-plane contact for box `idx` at world `center`, +Y normal.
// Returns false (no manifold) when the box is not penetrating the y=0 plane.
bool MakeGroundManifold(uint32_t idx, const math::Vec3& center,
                        const BoxParams& p, ContactManifold* out) {
    const float penetration = p.half_extent - center.y;  // bottom = center.y - h
    if (penetration <= 0.0f) {
        return false;
    }
    ContactManifold m;
    m.a.type = CollidableType::RigidBody;
    m.a.react = ReactionProviderKind::RigidInvMass;
    m.a.handle = idx;
    m.b.type = CollidableType::StaticWorld;
    m.b.react = ReactionProviderKind::StaticNull;
    m.b.handle = kGroundHandle;
    m.friction = p.friction;
    m.restitution = 0.0f;
    // MuJoCo default solimp (dmin, dmax, width, mid, power).
    m.solimp[0] = 0.9f;
    m.solimp[1] = 0.95f;
    m.solimp[2] = 0.001f;
    m.solimp[3] = 0.5f;
    m.solimp[4] = 2.0f;

    ContactPoint pt;
    pt.position = {center.x, 0.0f, center.z};
    pt.normal = {0.0f, 1.0f, 0.0f};
    pt.penetration = penetration;
    pt.normal_impulse = 0.0f;
    pt.solref_timeconst = p.solref_tc;
    pt.solref_dampratio = p.solref_dr;
    m.AddPoint(pt);
    *out = m;
    return true;
}

nuka::solver::SolverConfig MakeConfig() {
    nuka::solver::SolverConfig cfg;
    cfg.velocity_iterations = 20u;
    cfg.position_iterations = 4u;  // compliant rows skip Baumgarte regardless
    cfg.slop = 0.0f;
    cfg.baumgarte = 0.2f;
    return cfg;
}

struct SettleResult {
    BodyState body;
    float last_normal_lambda = 0.0f;
    float last_penetration = 0.0f;
    bool contact_active_last = false;
};

// One box stepped under gravity; caller owns gravity + integration. aref's `vel`
// is the START-OF-STEP (pre-gravity) normal velocity.
SettleResult RunBox(const BoxParams& p, math::Vec3 start, int steps) {
    std::vector<BodyState> bodies(1);
    bodies[0].inv_mass = p.inv_mass;
    bodies[0].inv_inertia = {0.0f, 0.0f, 0.0f};  // contact J is pure-linear -> unused
    bodies[0].position = start;

    const nuka::solver::SolverConfig cfg = MakeConfig();
    SettleResult result;
    for (int s = 0; s < steps; ++s) {
        const float vel_pre = bodies[0].linear_velocity.y;  // start-of-step
        bodies[0].linear_velocity.y += p.gravity * p.dt;    // gravity

        ContactManifold m;
        result.contact_active_last = MakeGroundManifold(0u, bodies[0].position, p, &m);
        if (result.contact_active_last) {
            RowBuffers rows;
            std::vector<ContactRowSides> sides;
            ContactRowComplianceInputs in;
            in.vel = vel_pre;          // MuJoCo: aref uses start-of-step velocity
            in.invweight = p.inv_mass; // sum of both sides (ground = 0)
            in.dt = p.dt;
            in.condim = p.condim;
            in.refsafe = true;
            EmitCompliantContactRows(std::span<const ContactManifold>(&m, 1),
                                     in, &rows, &sides);

            nuka::solver::SolveContext ctx;
            ctx.rows = &rows;
            ctx.state = &bodies;
            ctx.sides = &sides;
            ctx.reactions = nullptr;  // C5a dispatches RigidInvMass inline
            ctx.dt = p.dt;
            nuka::solver::UnifiedSolve(ctx, cfg);

            result.last_normal_lambda = rows.rows[0].lambda;  // row 0 = normal row
        }

        bodies[0].position.x += bodies[0].linear_velocity.x * p.dt;
        bodies[0].position.y += bodies[0].linear_velocity.y * p.dt;
        bodies[0].position.z += bodies[0].linear_velocity.z * p.dt;
    }
    result.body = bodies[0];
    result.last_penetration = p.half_extent - bodies[0].position.y;
    return result;
}

// Rest-start: box begins at rest, slightly penetrating (deeper than the steady
// state so contact stays continuous as it relaxes up to equilibrium).
SettleResult RestStart(const BoxParams& p, int steps) {
    return RunBox(p, {0.0f, p.half_extent - 0.003f, 0.0f}, steps);
}

}  // namespace

// (1) REST-START FORCE BALANCE: the contact supports the weight, lambda_n ~=
//     m*|g|*dt, at a small positive steady penetration. (m = 1.)
TEST(UnifiedSolve, RestingContactSupportsWeight) {
    BoxParams p;  // condim=1, tc=0.02 (= 4*dt, stable band)
    const SettleResult r = RestStart(p, 600);

    ASSERT_FALSE(std::isnan(r.body.position.y));
    ASSERT_TRUE(r.contact_active_last) << "contact released (chatter)";
    EXPECT_LT(std::fabs(r.body.linear_velocity.y), 5.0e-2f)
        << "not settled: vy=" << r.body.linear_velocity.y;
    EXPECT_GT(r.last_penetration, 0.0f);
    EXPECT_LT(r.last_penetration, 0.01f) << "penetration too deep";

    const float expected = (1.0f / p.inv_mass) * std::fabs(p.gravity) * p.dt;
    EXPECT_GT(r.last_normal_lambda, 0.0f);
    EXPECT_NEAR(r.last_normal_lambda, expected, 0.15f * expected)
        << "normal impulse " << r.last_normal_lambda
        << " does not balance weight impulse " << expected;
}

// (2) STIFFNESS ORACLE: a stiffer solref (smaller timeconst) settles at a smaller
//     steady penetration. Validates Row.compliance_alpha (R) is in the denominator.
TEST(UnifiedSolve, StifferContactPenetratesLess) {
    BoxParams stiff;
    stiff.solref_tc = 0.02f;  // stiffer (4*dt)
    BoxParams soft;
    soft.solref_tc = 0.06f;   // softer (12*dt)

    const SettleResult rs = RestStart(stiff, 800);
    const SettleResult rf = RestStart(soft, 800);

    ASSERT_TRUE(rs.contact_active_last);
    ASSERT_TRUE(rf.contact_active_last);
    EXPECT_GT(rs.last_penetration, 0.0f);
    EXPECT_GT(rf.last_penetration, 0.0f);
    EXPECT_LT(rs.last_penetration, rf.last_penetration)
        << "stiffer (tc=" << stiff.solref_tc << ", pen=" << rs.last_penetration
        << ") should penetrate LESS than softer (tc=" << soft.solref_tc
        << ", pen=" << rf.last_penetration << ")";
}

// (3) DROP STABILITY (generous): a dropped box does not blow up / sink through and
//     ends near the surface. Transient stability only -- precise oracles above.
TEST(UnifiedSolve, DroppedBoxIsStable) {
    BoxParams p;
    p.solref_tc = 0.03f;  // comfortably inside the stable band
    const SettleResult r = RunBox(p, {0.0f, p.half_extent + 0.02f, 0.0f}, 800);

    EXPECT_FALSE(std::isnan(r.body.position.y));
    EXPECT_FALSE(std::isnan(r.body.linear_velocity.y));
    EXPECT_GT(r.body.position.y, 0.0f) << "box fell through the ground";
    // Ends within a few mm of the surface (not launched, not sunk).
    EXPECT_LT(std::fabs(r.body.position.y - p.half_extent), 0.02f)
        << "box settled far from the surface: y=" << r.body.position.y;
    // Velocity is small (within ~one gravity kick of rest; may micro-chatter).
    EXPECT_LT(std::fabs(r.body.linear_velocity.y), 0.1f)
        << "box still moving fast: vy=" << r.body.linear_velocity.y;
}

// (4) PYRAMID FRICTION: drive a penetrating box horizontally. High mu grips
//     (vx -> 0); mu = 0 slides freely (vx unchanged). Exercises the NEW
//     unilateral coupled-pyramid bound [0, mu*sum lambda_n].
namespace {
float SlideBox(float friction, float vx0, int steps) {
    BoxParams p;
    p.friction = friction;
    p.condim = 3u;  // 4 pyramid spokes

    std::vector<BodyState> bodies(1);
    bodies[0].inv_mass = p.inv_mass;
    bodies[0].inv_inertia = {0.0f, 0.0f, 0.0f};
    bodies[0].position = {0.0f, p.half_extent - 0.002f, 0.0f};  // penetrating 2mm
    bodies[0].linear_velocity = {vx0, 0.0f, 0.0f};

    const nuka::solver::SolverConfig cfg = MakeConfig();
    for (int s = 0; s < steps; ++s) {
        const float vel_pre = bodies[0].linear_velocity.y;
        bodies[0].linear_velocity.y += p.gravity * p.dt;  // press it down

        ContactManifold m;
        if (MakeGroundManifold(0u, bodies[0].position, p, &m)) {
            RowBuffers rows;
            std::vector<ContactRowSides> sides;
            ContactRowComplianceInputs in;
            in.vel = vel_pre;
            in.invweight = p.inv_mass;
            in.dt = p.dt;
            in.condim = p.condim;
            in.refsafe = true;
            EmitCompliantContactRows(std::span<const ContactManifold>(&m, 1),
                                     in, &rows, &sides);

            nuka::solver::SolveContext ctx;
            ctx.rows = &rows;
            ctx.state = &bodies;
            ctx.sides = &sides;
            ctx.reactions = nullptr;
            ctx.dt = p.dt;
            nuka::solver::UnifiedSolve(ctx, cfg);
        }
        bodies[0].position.x += bodies[0].linear_velocity.x * p.dt;
        bodies[0].position.y += bodies[0].linear_velocity.y * p.dt;
    }
    return bodies[0].linear_velocity.x;
}
}  // namespace

TEST(UnifiedSolve, PyramidFrictionGripsHighMuSlidesZeroMu) {
    constexpr float kVx0 = 0.1f;
    const float vx_grip = SlideBox(/*friction=*/1.0f, kVx0, 60);
    const float vx_slide = SlideBox(/*friction=*/0.0f, kVx0, 60);

    // High mu: the pyramid spokes arrest the tangential motion.
    EXPECT_LT(std::fabs(vx_grip), 0.2f * kVx0)
        << "high-mu box should be gripped, vx=" << vx_grip;
    // mu = 0: friction bound is [0,0] -> no tangential impulse -> vx unchanged.
    EXPECT_NEAR(vx_slide, kVx0, 1.0e-3f)
        << "frictionless box should slide freely, vx=" << vx_slide;
}

// (5a) D1: two-run byte-exact final state.
TEST(UnifiedSolve, TwoRunByteExact) {
    BoxParams p;
    const SettleResult a = RestStart(p, 400);
    const SettleResult b = RestStart(p, 400);
    EXPECT_EQ(std::memcmp(&a.body, &b.body, sizeof(BodyState)), 0)
        << "two identical UnifiedSolve runs diverged (non-deterministic)";
}

// (5b) D1: N=32 cross-replica. 32 identical boxes stepped together in ONE solve
//      must each be byte-identical to a single box stepped alone (no cross-env
//      contamination, deterministic batched solve).
TEST(UnifiedSolve, CrossReplicaMatchesSingleEnv) {
    constexpr uint32_t kN = 32u;
    constexpr int kSteps = 200;
    BoxParams p;  // condim=1: one normal row per box

    const SettleResult ref = RestStart(p, kSteps);

    const float start_y = p.half_extent - 0.003f;
    std::vector<BodyState> bodies(kN);
    for (uint32_t i = 0; i < kN; ++i) {
        bodies[i].inv_mass = p.inv_mass;
        bodies[i].inv_inertia = {0.0f, 0.0f, 0.0f};
        bodies[i].position = {0.0f, start_y, 0.0f};
    }
    const nuka::solver::SolverConfig cfg = MakeConfig();
    for (int s = 0; s < kSteps; ++s) {
        const float vel_pre = bodies[0].linear_velocity.y;  // all replicas identical
        for (uint32_t i = 0; i < kN; ++i) {
            bodies[i].linear_velocity.y += p.gravity * p.dt;
        }
        std::vector<ContactManifold> manifolds;
        manifolds.reserve(kN);
        for (uint32_t i = 0; i < kN; ++i) {
            ContactManifold m;
            if (MakeGroundManifold(i, bodies[i].position, p, &m)) {
                manifolds.push_back(m);
            }
        }
        if (!manifolds.empty()) {
            RowBuffers rows;
            std::vector<ContactRowSides> sides;
            ContactRowComplianceInputs in;
            in.vel = vel_pre;
            in.invweight = p.inv_mass;
            in.dt = p.dt;
            in.condim = p.condim;
            in.refsafe = true;
            EmitCompliantContactRows(std::span<const ContactManifold>(manifolds),
                                     in, &rows, &sides);
            nuka::solver::SolveContext ctx;
            ctx.rows = &rows;
            ctx.state = &bodies;
            ctx.sides = &sides;
            ctx.reactions = nullptr;
            ctx.dt = p.dt;
            nuka::solver::UnifiedSolve(ctx, cfg);
        }
        for (uint32_t i = 0; i < kN; ++i) {
            bodies[i].position.y += bodies[i].linear_velocity.y * p.dt;
        }
    }

    for (uint32_t i = 0; i < kN; ++i) {
        EXPECT_EQ(std::memcmp(&bodies[i], &ref.body, sizeof(BodyState)), 0)
            << "replica " << i << " diverged from the single-env reference";
    }
}

// ===========================================================================
// M4 RE-POINT — the SAME oracle semantics through the NEW nk core (plan M4:
// "test_unified_solve (oracle 口径) 重指"). The scenarios above drove the
// legacy UnifiedSolve directly; the cases below drive a 1-box-on-plane
// UnionCsr nk::Model through nk::World (NarrowphasePrimitives ->
// AssembleRows -> SolveRowsBlockIsland — the M4 device-resident pipeline)
// and hold the SAME physics oracles at the SAME tolerances:
//   (1) rest-start force balance  Σλ_n ≈ m·|g|·dt (±15 %),
//   (2) stiffer solref -> smaller steady penetration,
//   (4) pyramid friction: high-mu grips, mu=0 slides freely,
//   (5) D1 two-run byte-exact + N=8 cross-replica byte-identity,
//   (+) the solref/solimp ASSEMBLY re-point: the AssembleRows-emitted row's
//       (rhs, compliance_alpha) match the host ComputeCompliantRow at the
//       detected depth (the SAME shared HD header — test_solref_solimp's
//       oracle semantics carried onto the nk path).
// Geometry note: the nk union plane class is +Z-up (the production union
// world's frame); the legacy section above used +Y — same physics, the
// oracle quantities are axis-free.
// The nk pipeline feeds vel=0 / invweight=1 into the compliance terms (the
// legacy BatchedUnifiedWorld's exact production inputs) — at the settled
// equilibria the scenarios assert, the start-of-step velocity is ~0, so the
// oracle bands hold unchanged.
// ===========================================================================

#include "constraint/solref_solimp.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;

struct NkBoxFixture {
    nphi::Device* dev = nullptr;
    nphi::Backend* backend = nullptr;
    NkBoxFixture() {
        dev = nphi::InitBestDevice();
        if (dev != nullptr) backend = nphi::DeviceInitBackend(dev, nullptr);
    }
    ~NkBoxFixture() {
        if (backend != nullptr) nphi::BackendFree(backend);
    }
};

struct NkBoxResult {
    nuka::math::Transform pose;
    nuka::math::Vec3 vel;
    float normal_lambda_sum = 0.0f;
    float penetration = 0.0f;
    bool contact_active = false;
    // the first active normal row + its detected depth (assembly re-point).
    float row_rhs = 0.0f, row_R = 0.0f, row_depth = 0.0f;
    bool have_row = false;
};

nk::Model MakeNkBoxModel(const BoxParams& p, float vx0, uint32_t envs) {
    nk::Model model;
    model.contact_family = nk::ContactFamily::UnionCsr;
    model.union_solref[0] = p.solref_tc;
    model.union_solref[1] = p.solref_dr;
    nk::UnionSlot slot;
    slot.cls = nk::UnionSlot::kBodyBoxPlane;
    slot.body = 0u;
    slot.box_half = {p.half_extent, p.half_extent, p.half_extent};
    slot.plane_height = 0.0f;
    slot.mu = p.friction;
    slot.condim = p.condim;
    model.union_slots.push_back(slot);
    nk::Model::BodyInit init;
    init.pose.position = {0.0f, 0.0f, p.half_extent - 0.003f};
    init.linear_velocity = {vx0, 0.0f, 0.0f};
    init.inv_mass = p.inv_mass;
    init.inv_inertia = {0.0f, 0.0f, 0.0f};
    model.body_init.push_back(init);
    nk::ModelCapacities& cap = model.capacities;
    cap.env_count = envs;
    cap.bodies_per_env = 1;
    cap.max_contacts_per_env = 1;
    cap.max_rows_per_env = model.union_slots[0].MaxRows();
    return model;
}

NkBoxResult RunNkBox(NkBoxFixture& fx, const BoxParams& p, int steps,
                     float vx0 = 0.0f, uint32_t envs = 1, uint32_t read_env = 0) {
    nk::Model model = MakeNkBoxModel(p, vx0, envs);
    const uint32_t rows_per_env = model.capacities.max_rows_per_env;
    nk::Pipeline::SolverConfig cfg;
    cfg.dt = p.dt;
    cfg.gravity[0] = 0.0f; cfg.gravity[1] = 0.0f; cfg.gravity[2] = p.gravity;
    cfg.vel_iters = 20;  // the legacy MakeConfig() velocity_iterations
    cfg.pos_iters = 0;
    nk::World world(std::move(model), envs, fx.dev, fx.backend, cfg);
    EXPECT_TRUE(world.Ready());
    for (int s = 0; s < steps; ++s) {
        const nk::StepResult r = world.Step();
        EXPECT_TRUE(r.AllOk()) << "nk op failed at step " << s;
        if (!r.AllOk()) break;
    }
    NkBoxResult out;
    EXPECT_TRUE(world.GetData().DownloadField(
        nk::FieldId::BodyPose, &out.pose, sizeof(out.pose),
        read_env * sizeof(nuka::math::Transform)));
    EXPECT_TRUE(world.GetData().DownloadField(
        nk::FieldId::BodyLinearVelocity, &out.vel, sizeof(out.vel),
        read_env * sizeof(nuka::math::Vec3)));
    out.penetration = p.half_extent - out.pose.position.z;
    std::vector<nk::NkRow> rows(rows_per_env);
    std::vector<float> lambda(rows_per_env);
    std::vector<float> depths(4);
    uint32_t npoints = 0;
    EXPECT_TRUE(world.GetData().DownloadField(
        nk::FieldId::Urows, rows.data(), rows.size() * sizeof(nk::NkRow),
        static_cast<uint64_t>(read_env) * rows_per_env * sizeof(nk::NkRow)));
    EXPECT_TRUE(world.GetData().DownloadField(
        nk::FieldId::Lambda, lambda.data(), lambda.size() * sizeof(float),
        static_cast<uint64_t>(read_env) * rows_per_env * sizeof(float)));
    EXPECT_TRUE(world.GetData().DownloadField(
        nk::FieldId::UcontactCount, &npoints, sizeof(uint32_t),
        read_env * sizeof(uint32_t)));
    EXPECT_TRUE(world.GetData().DownloadField(
        nk::FieldId::UcontactDepth, depths.data(), 4 * sizeof(float),
        static_cast<uint64_t>(read_env) * 4 * sizeof(float)));
    out.contact_active = npoints > 0;
    for (uint32_t i = 0; i < rows_per_env; ++i) {
        if (!(rows[i].flags & nk::nk_row_flags::kActive)) continue;
        if (rows[i].flags & nk::nk_row_flags::kFriction) continue;
        out.normal_lambda_sum += lambda[i];
        if (!out.have_row) {
            out.have_row = true;
            out.row_rhs = rows[i].rhs;
            out.row_R = rows[i].compliance_alpha;
            out.row_depth = depths[i];  // normal slot i <-> point i (fixed layout)
        }
    }
    return out;
}

}  // namespace

// (1) rest-start force balance through the nk pipeline (same ±15 % band; the
// 4-corner patch's SUM of normal impulses balances the weight impulse).
TEST(UnifiedSolve, NkWorldRestingContactSupportsWeight) {
    NkBoxFixture fx;
    ASSERT_NE(fx.backend, nullptr);
    BoxParams p;  // condim=1, tc=0.02 (the legacy stable band)
    const NkBoxResult r = RunNkBox(fx, p, 600);
    ASSERT_FALSE(std::isnan(r.pose.position.z));
    ASSERT_TRUE(r.contact_active) << "contact released (chatter)";
    EXPECT_LT(std::fabs(r.vel.z), 5.0e-2f) << "not settled: vz=" << r.vel.z;
    EXPECT_GT(r.penetration, 0.0f);
    EXPECT_LT(r.penetration, 0.01f) << "penetration too deep";
    const float expected = (1.0f / p.inv_mass) * std::fabs(p.gravity) * p.dt;
    EXPECT_GT(r.normal_lambda_sum, 0.0f);
    EXPECT_NEAR(r.normal_lambda_sum, expected, 0.15f * expected)
        << "nk normal impulse sum " << r.normal_lambda_sum
        << " does not balance the weight impulse " << expected;

    // (+) the solref/solimp ASSEMBLY re-point: the emitted row terms equal the
    // host ComputeCompliantRow at the detected depth (same HD header; the
    // device TU compiles default-fmad -> EXPECT_NEAR at a 1e-5 relative bar,
    // the documented ~1-ULP contraction class).
    ASSERT_TRUE(r.have_row);
    const float solref[2] = {p.solref_tc, p.solref_dr};
    const float solimp[5] = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
    const auto expect = nuka::constraint::ComputeCompliantRow(
        solref, solimp, -r.row_depth, -r.row_depth, /*vel=*/0.0f,
        /*invweight=*/1.0f, p.dt, /*refsafe=*/true);
    EXPECT_NEAR(r.row_rhs, expect.aref_bias,
                std::max(1.0e-6f, 1.0e-5f * std::fabs(expect.aref_bias)));
    EXPECT_NEAR(r.row_R, expect.R, std::max(1.0e-9f, 1.0e-5f * expect.R));
}

// (2) stiffness oracle through the nk pipeline (same ordering assertion).
TEST(UnifiedSolve, NkWorldStifferContactPenetratesLess) {
    NkBoxFixture fx;
    ASSERT_NE(fx.backend, nullptr);
    BoxParams stiff; stiff.solref_tc = 0.02f;
    BoxParams soft;  soft.solref_tc = 0.06f;
    const NkBoxResult rs = RunNkBox(fx, stiff, 800);
    const NkBoxResult rf = RunNkBox(fx, soft, 800);
    ASSERT_TRUE(rs.contact_active);
    ASSERT_TRUE(rf.contact_active);
    EXPECT_GT(rs.penetration, 0.0f);
    EXPECT_GT(rf.penetration, 0.0f);
    EXPECT_LT(rs.penetration, rf.penetration)
        << "stiffer should penetrate LESS (nk path)";
}

// (4) pyramid friction through the nk pipeline (same bands).
TEST(UnifiedSolve, NkWorldPyramidFrictionGripsHighMuSlidesZeroMu) {
    NkBoxFixture fx;
    ASSERT_NE(fx.backend, nullptr);
    constexpr float kVx0 = 0.1f;
    BoxParams grip;  grip.friction = 1.0f; grip.condim = 3u;
    BoxParams slide; slide.friction = 0.0f; slide.condim = 3u;
    // start penetrating 2 mm (the legacy SlideBox IC).
    grip.half_extent = slide.half_extent = 0.5f;
    NkBoxResult rg, rsld;
    {
        nk::Model m = MakeNkBoxModel(grip, kVx0, 1);
        m.body_init[0].pose.position.z = grip.half_extent - 0.002f;
        nk::Pipeline::SolverConfig cfg;
        cfg.dt = grip.dt; cfg.gravity[2] = grip.gravity;
        cfg.vel_iters = 20; cfg.pos_iters = 0;
        nk::World w(std::move(m), 1, fx.dev, fx.backend, cfg);
        ASSERT_TRUE(w.Ready());
        for (int s = 0; s < 60; ++s) ASSERT_TRUE(w.Step().AllOk());
        ASSERT_TRUE(w.GetData().DownloadField(nk::FieldId::BodyLinearVelocity,
                                              &rg.vel, sizeof(rg.vel)));
    }
    {
        nk::Model m = MakeNkBoxModel(slide, kVx0, 1);
        m.body_init[0].pose.position.z = slide.half_extent - 0.002f;
        nk::Pipeline::SolverConfig cfg;
        cfg.dt = slide.dt; cfg.gravity[2] = slide.gravity;
        cfg.vel_iters = 20; cfg.pos_iters = 0;
        nk::World w(std::move(m), 1, fx.dev, fx.backend, cfg);
        ASSERT_TRUE(w.Ready());
        for (int s = 0; s < 60; ++s) ASSERT_TRUE(w.Step().AllOk());
        ASSERT_TRUE(w.GetData().DownloadField(nk::FieldId::BodyLinearVelocity,
                                              &rsld.vel, sizeof(rsld.vel)));
    }
    EXPECT_LT(std::fabs(rg.vel.x), 0.2f * kVx0)
        << "high-mu box should be gripped (nk), vx=" << rg.vel.x;
    EXPECT_NEAR(rsld.vel.x, kVx0, 1.0e-3f)
        << "frictionless box should slide freely (nk), vx=" << rsld.vel.x;
}

// (5a) D1 two-run byte-exact + (5b) N=8 cross-replica byte-identity.
TEST(UnifiedSolve, NkWorldD1AndCrossReplica) {
    NkBoxFixture fx;
    ASSERT_NE(fx.backend, nullptr);
    BoxParams p;
    const NkBoxResult a = RunNkBox(fx, p, 400);
    const NkBoxResult b = RunNkBox(fx, p, 400);
    EXPECT_EQ(std::memcmp(&a.pose, &b.pose, sizeof(a.pose)), 0)
        << "two identical nk runs diverged (non-deterministic)";
    EXPECT_EQ(std::memcmp(&a.vel, &b.vel, sizeof(a.vel)), 0);

    const NkBoxResult ref = RunNkBox(fx, p, 200);
    for (uint32_t e = 0; e < 8; ++e) {
        const NkBoxResult re = RunNkBox(fx, p, 200, 0.0f, /*envs=*/8, e);
        EXPECT_EQ(std::memcmp(&re.pose, &ref.pose, sizeof(ref.pose)), 0)
            << "replica " << e << " diverged from the single-env nk reference";
        EXPECT_EQ(std::memcmp(&re.vel, &ref.vel, sizeof(ref.vel)), 0);
    }
}
