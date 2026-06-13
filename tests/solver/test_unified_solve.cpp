// ---------------------------------------------------------------------------
// Compliant-contact solve oracle — the nk-core re-point (M9 T11).
// ---------------------------------------------------------------------------
// The legacy `solver::UnifiedSolve` host driver (src/solver/unified_solve.*) was
// deleted in M9; these tests hold the SAME compliant-contact physics oracles on
// the NEW nk core. A 1-box-on-plane UnionCsr nk::Model is driven through
// nk::World (NarrowphasePrimitives -> AssembleRows -> SolveRowsBlockIsland — the
// M4 device-resident pipeline) and asserts:
//   (1) REST-START FORCE BALANCE: Σλ_n ≈ m·|g|·dt (±15 %) at a small positive
//       steady penetration (the contact supports the weight).
//   (2) STIFFNESS ORACLE: a stiffer solref (smaller timeconst) settles at a
//       SMALLER steady penetration (Row.compliance_alpha (R) wired into the
//       effective-mass denominator).
//   (4) PYRAMID FRICTION: a high-mu sliding box is GRIPPED (vx -> 0) while a
//       mu=0 box SLIDES freely (the unilateral coupled-pyramid bounds).
//   (5) D1: two-run byte-exact + N=8 cross-replica byte-identity.
//   (+) the solref/solimp ASSEMBLY re-point: the AssembleRows-emitted row's
//       (rhs, compliance_alpha) match the host ComputeCompliantRow at the
//       detected depth (the SAME shared HD header — test_solref_solimp's oracle
//       semantics carried onto the nk path).
//
// These `NkWorld*` arms mirror EVERY assertion the deleted legacy `UnifiedSolve.*`
// cases carried (compliant-row R/rhs, penetration ordering, lambda-sum=weight,
// friction cone, D1, cross-replica). The legacy arms + the deleted
// `solver/unified_solve.hpp` include are gone.
//
// PHYSICS CONVENTIONS the oracle relies on (both load-bearing):
//   - aref is a reference ACCELERATION; the velocity-impulse PGS scales it by dt.
//   - aref's `vel` is the START-OF-STEP (pre-gravity) constraint velocity (MuJoCo).
// The nk pipeline feeds vel=0 / invweight=1 into the compliance terms (the legacy
// BatchedUnifiedWorld's exact production inputs) — at the settled equilibria these
// scenarios assert, the start-of-step velocity is ~0, so the oracle bands hold.
// Geometry note: the nk union plane class is +Z-up (the production union world's
// frame); the oracle quantities are axis-free.
// ---------------------------------------------------------------------------

#include "constraint/solref_solimp.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace math = nuka::math;

namespace {

// Box scenario parameters (shared by every nk box scenario below).
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
