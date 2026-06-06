// ---------------------------------------------------------------------------
// v0.8 C6b: the cross-system coupling-row framework + the ParticleInvMass
// reaction arm wired into UnifiedSolve. The K2 proving pair: a particle pushes a
// free rigid body, two-way, through ONE unified compliant contact solve.
// ---------------------------------------------------------------------------
// This is the FIRST exercise of the ParticleInvMass reaction arm (C5a left it
// "in-place-unexercised"; C6b wires its three legs). The arm has the SAME failure
// mode the C5b-core articulation arm taught: the compliant solve is
//     lambda = m_eff * (aref*dt - Jv - R*lambda),
// and the Jv leg for a particle side is dot(J_linear, v_particle). If that leg is
// dropped (returns 0), the solve still passes a rest / momentum-only gate but
// converges to a WRONG fixed point on any TRANSIENT (nonzero particle velocity)
// and breaks two-way. So CASE 1 drives a NONZERO particle velocity and asserts the
// solved lambda against the closed form (aref*dt - Jv0)/(A+R), with a BROKEN-Jv
// discriminator (the lambda the arm would produce if the particle Jv were 0) that
// differs by >> tol -- a passing CASE 1 PROVES the particle Jv leg is live.
//
// CASE 2 proves the FRAMEWORK (EmitCouplingRows): a manifold -> compliant rows
// re-classed to RowClassId CouplingContactRow (id13). CASE 3 runs the framework
// end-to-end (emit + solve) and asserts BOTH sides react with mass-weighted
// momentum conserved. CASE 4 puts the contact OFF the rigid COM and asserts the
// body's ANGULAR velocity reacts (the r x n contact Jacobian is live for coupling
// rows too). CASE 5 is the D1 two-run byte-identity gate.
//
// VALIDATED, NOT WIRED: UnifiedSolve is a standalone entry; no production stepper
// is touched. The PBD co-step that binds the particle SoA to the live xpbd/pbf
// velocity buffers is C6a.
// ---------------------------------------------------------------------------

#include "codegen/generated/row_class_registry.hpp"  // kCouplingContactRowId
#include "constraint/collidable.hpp"
#include "constraint/contact_manifold.hpp"
#include "constraint/contact_row_sides.hpp"
#include "constraint/row.hpp"
#include "constraint/row_buffers.hpp"
#include "constraint/row_builder.hpp"  // ContactRowComplianceInputs
#include "runtime/coupling/coupling_row_framework.hpp"
#include "runtime/rigid/body_state.hpp"
#include "solver/unified_solve.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace nuka::constraint;
using nuka::runtime::rigid::BodyState;
namespace math = nuka::math;

namespace {

// --- the K2 scenario constants (a particle pushing a free rigid body) ----------
// Particle: light point mass; Rigid box: heavier free body. Contact normal +Y
// (separation dir for side A == the particle, the manifold convention). Velocities
// are NONZERO along the normal so the Jv legs are load-bearing.
constexpr float kInvMassParticle = 10.0f;   // m_p = 0.1
constexpr float kMassParticle = 0.1f;
constexpr float kInvMassBox = 0.5f;         // m_box = 2.0
constexpr float kMassBox = 2.0f;
constexpr float kInvInertiaBox = 5.0f;      // isotropic, for the off-COM case
constexpr float kDt = 0.01f;
constexpr float kR = 0.05f;
constexpr float kAref = 400.0f;             // aref*dt = 4.0
constexpr float kTol = 1.0e-5f;

math::Vec3 Normal() { return math::Vec3{0.0f, 1.0f, 0.0f}; }
math::Vec3 ParticleVel0() { return math::Vec3{0.0f, 2.0f, 0.0f}; }
math::Vec3 BoxVel0() { return math::Vec3{0.0f, -0.5f, 0.0f}; }

CollidableRef MakeParticleRef(uint32_t handle) {
    CollidableRef ref;
    ref.type = CollidableType::Particle;
    ref.react = ReactionProviderKind::ParticleInvMass;
    ref.handle = handle;
    return ref;
}

CollidableRef MakeRigidRef(uint32_t handle) {
    CollidableRef ref;
    ref.type = CollidableType::RigidBody;
    ref.react = ReactionProviderKind::RigidInvMass;
    ref.handle = handle;
    return ref;
}

nuka::solver::SolverConfig OneIterConfig() {
    nuka::solver::SolverConfig cfg;
    cfg.velocity_iterations = 1u;  // 1 iter from lambda=0 -> exact closed form.
    cfg.position_iterations = 0u;
    cfg.slop = 0.0f;
    cfg.baumgarte = 0.0f;
    return cfg;
}

BodyState MakeBox(math::Vec3 position, math::Vec3 velocity) {
    BodyState box;
    box.inv_mass = kInvMassBox;
    box.inv_inertia = math::Vec3{kInvInertiaBox, kInvInertiaBox, kInvInertiaBox};
    box.position = position;
    box.linear_velocity = velocity;
    box.angular_velocity = math::Vec3{0.0f, 0.0f, 0.0f};
    return box;
}

// Append ONE hand-built compliant normal row (single contact). The particle side
// gets a coloring key >= body_count so the rigid arm stays inert for it (it is
// dispatched by react == ParticleInvMass via side.handle, NOT body_index --
// mirrors the C5b articulation coloring body_count + art_index). jac_b carries the
// LINEAR-only normal; the kernel fills the rigid side's angular r x n from
// contact_point - box.position in-kernel (the C5c augmentation).
void AppendCouplingNormalRow(RowBuffers* rows,
                             std::vector<ContactRowSides>* sides,
                             CollidableRef side_a, CollidableRef side_b,
                             uint32_t body_a, uint32_t body_b,
                             math::Vec3 jac_lin_a, math::Vec3 jac_lin_b,
                             float aref, float R, math::Vec3 contact_point) {
    Row row;
    row.row_class_id = kMaximalContactRowClassId;
    row.rhs = aref;
    row.lambda = 0.0f;
    row.lower = 0.0f;
    row.upper = kRowHugeLimit;
    row.compliance_alpha = R;
    row.flags = row_flags::Compliant | row_flags::Unilateral;

    RowMaterial material;
    material.kind = RowKind::Contact;
    material.group_id = rows->RowCount();
    material.normal_row_count = 1u;
    material.first_friction_row = 0u;
    material.friction_row_count = 0u;
    material.friction = 0.0f;
    material.restitution = 0.0f;
    material.position_error = 0.0f;

    const RowJacobian6 jac_a{jac_lin_a, math::Vec3::Zero()};
    const RowJacobian6 jac_b{jac_lin_b, math::Vec3::Zero()};
    rows->AddRow(row, RowBodyPair{body_a, body_b}, jac_a, jac_b, material);
    sides->push_back(ContactRowSides{side_a, side_b, contact_point});
}

}  // namespace

// ===========================================================================
// CASE 1 (PRIMARY): COM-centered particle<->rigid contact vs the closed form,
//   with a BROKEN-Jv discriminator. side A = ParticleInvMass (NONZERO velocity),
//   side B = RigidInvMass. Exercises ALL THREE particle legs (eff-mass, Jv, apply)
//   + the two-way rigid apply. COM-centered so the box angular term vanishes and
//   the closed form is a clean 1-DOF (A = invM_p + invM_box).
// ===========================================================================
TEST(CouplingFramework, ParticleRigidClosedFormAndJvDiscriminator) {
    const math::Vec3 n = Normal();
    // A box at the origin; the contact at the COM -> zero lever -> no box angular.
    std::vector<BodyState> bodies(1);
    bodies[0] = MakeBox(/*position=*/math::Vec3{0.0f, 0.0f, 0.0f}, BoxVel0());
    const uint32_t body_count = 1u;

    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    // Particle coloring key = body_count + particle_id (>= body_count -> rigid arm
    // inert for it); particle SoA index is the CollidableRef.handle (0).
    AppendCouplingNormalRow(&rows, &sides,
                            /*side_a=*/MakeParticleRef(0u),
                            /*side_b=*/MakeRigidRef(0u),
                            /*body_a=*/body_count + 0u, /*body_b=*/0u,
                            /*jac_lin_a=*/n, /*jac_lin_b=*/math::Vec3{0,-1,0},
                            kAref, kR, /*contact_point=*/bodies[0].position);

    std::vector<float> inv_mass = {kInvMassParticle};
    std::vector<math::Vec3> velocities = {ParticleVel0()};

    nuka::solver::SolveContext ctx;
    ctx.rows = &rows;
    ctx.state = &bodies;
    ctx.sides = &sides;
    ctx.dt = kDt;
    ctx.particles.inv_mass = &inv_mass;
    ctx.particles.velocities = &velocities;

    nuka::solver::UnifiedSolve(ctx, OneIterConfig());

    // --- closed form -------------------------------------------------------
    // A = invM_p*|n|^2 + invM_box*|n|^2 (no angular: COM-centered).
    const float A = kInvMassParticle + kInvMassBox;
    const float eff_mass = 1.0f / (A + kR);
    const float jv0_correct = n.Dot(ParticleVel0()) + math::Vec3{0,-1,0}.Dot(BoxVel0());
    const float lambda_correct = eff_mass * (kAref * kDt - jv0_correct);
    // The lambda the arm WOULD produce if the particle Jv leg returned 0 (the bug):
    const float jv0_broken = math::Vec3{0,-1,0}.Dot(BoxVel0());  // box term only
    const float lambda_broken = eff_mass * (kAref * kDt - jv0_broken);

    // The discriminator must be non-vacuous: the two fixed points differ >> tol.
    ASSERT_GT(std::fabs(lambda_correct - lambda_broken), 0.05f)
        << "test setup: broken-Jv discriminator is too weak";
    ASSERT_GT(kAref * kDt - jv0_correct, 0.0f) << "clamp would zero lambda";

    const float lambda = rows.rows[0].lambda;
    EXPECT_NEAR(lambda, lambda_correct, kTol)
        << "solved lambda != closed form -> the particle Jv leg is WRONG";
    EXPECT_GT(std::fabs(lambda - lambda_broken), 0.05f)
        << "solved lambda matches the BROKEN-Jv value -> particle Jv leg is dead";

    // --- two-way apply (both sides react) ----------------------------------
    const math::Vec3 dv_p = velocities[0] - ParticleVel0();
    const math::Vec3 dv_box = bodies[0].linear_velocity - BoxVel0();
    EXPECT_NEAR(dv_p.y, kInvMassParticle * n.y * lambda, kTol);
    EXPECT_NEAR(dv_box.y, kInvMassBox * (-n.y) * lambda, kTol);
    EXPECT_GT(dv_p.y, 0.0f) << "particle did not react";
    EXPECT_LT(dv_box.y, 0.0f) << "box did not react";
    // Mass-weighted momentum is conserved (equal-and-opposite impulse).
    const float dp = kMassParticle * dv_p.y + kMassBox * dv_box.y;
    EXPECT_NEAR(dp, 0.0f, kTol) << "two-way momentum not balanced";
}

// ===========================================================================
// CASE 2: the FRAMEWORK. EmitCouplingRows turns a particle<->rigid manifold into
//   compliant rows RE-CLASSED to RowClassId CouplingContactRow (id13), carrying
//   the per-side CollidableRefs + the world contact point. condim=1 -> exactly one
//   (normal) row, no friction spokes.
// ===========================================================================
TEST(CouplingFramework, EmitCouplingRowsReclassesAndTags) {
    ContactManifold m;
    m.a = MakeParticleRef(0u);
    m.b = MakeRigidRef(0u);
    m.friction = 0.0f;
    ContactPoint pt;
    pt.position = math::Vec3{0.1f, 0.2f, 0.3f};
    pt.normal = Normal();
    pt.penetration = 0.01f;
    pt.solref_timeconst = 0.02f;
    pt.solref_dampratio = 1.0f;
    m.AddPoint(pt);

    ContactRowComplianceInputs inputs;
    inputs.vel = -0.1f;
    inputs.invweight = kInvMassParticle + kInvMassBox;
    inputs.dt = kDt;
    inputs.condim = 1u;  // frictionless -> 1 normal row only.
    inputs.refsafe = true;

    const nuka::runtime::coupling::CouplingPair pair{m.a, m.b, 0u};
    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    nuka::runtime::coupling::EmitCouplingRows(
        std::span<const nuka::runtime::coupling::CouplingPair>(&pair, 1),
        std::span<const ContactManifold>(&m, 1), inputs, &rows, &sides);

    ASSERT_EQ(rows.RowCount(), 1u) << "condim=1 must emit exactly one normal row";
    ASSERT_EQ(sides.size(), 1u);
    // The framework re-classed the compliant row to the coupling class (id13).
    EXPECT_EQ(rows.rows[0].row_class_id,
              nuka::solver::generated::kCouplingContactRowId);
    // It stays a compliant unilateral row (the solve path is unchanged).
    EXPECT_TRUE(rows.rows[0].flags & row_flags::Compliant);
    EXPECT_TRUE(rows.rows[0].flags & row_flags::Unilateral);
    // The C4a compliance terms are populated (aref into rhs, R into alpha).
    EXPECT_GT(rows.rows[0].compliance_alpha, 0.0f);  // R = 1/D
    // The per-side tags + world contact point survive for the solver.
    EXPECT_EQ(sides[0].a.react, ReactionProviderKind::ParticleInvMass);
    EXPECT_EQ(sides[0].b.react, ReactionProviderKind::RigidInvMass);
    EXPECT_NEAR(sides[0].contact_point.x, 0.1f, kTol);
    EXPECT_NEAR(sides[0].contact_point.z, 0.3f, kTol);
}

// ===========================================================================
// CASE 3: end-to-end through the FRAMEWORK + the arm. EmitCouplingRows builds the
//   row from a manifold, UnifiedSolve solves it, and BOTH sides react two-way with
//   mass-weighted momentum conserved (the integration proof, not a closed form --
//   aref/R come from the C4a schedule).
// ===========================================================================
TEST(CouplingFramework, FrameworkEndToEndTwoWayMomentum) {
    ContactManifold m;
    m.a = MakeParticleRef(0u);
    m.b = MakeRigidRef(0u);
    m.friction = 0.0f;
    ContactPoint pt;
    pt.position = math::Vec3{0.0f, 0.0f, 0.0f};  // box COM -> isolate linear.
    pt.normal = Normal();
    pt.penetration = 0.05f;  // a healthy overlap -> aref pushes the pair apart.
    m.AddPoint(pt);

    // Start at REST along the normal (the closed-form Jv discriminator is CASE 1's
    // job). The compliant aref (from the penetration) drives a positive lambda ->
    // both sides separate two-way.
    ContactRowComplianceInputs inputs;
    inputs.vel = 0.0f;
    inputs.invweight = kInvMassParticle + kInvMassBox;
    inputs.dt = kDt;
    inputs.condim = 1u;
    inputs.refsafe = true;

    const nuka::runtime::coupling::CouplingPair pair{m.a, m.b, 0u};
    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    nuka::runtime::coupling::EmitCouplingRows(
        std::span<const nuka::runtime::coupling::CouplingPair>(&pair, 1),
        std::span<const ContactManifold>(&m, 1), inputs, &rows, &sides);
    ASSERT_EQ(rows.RowCount(), 1u);

    const math::Vec3 box_vel0{0.0f, 0.0f, 0.0f};
    const math::Vec3 particle_vel0{0.0f, 0.0f, 0.0f};
    std::vector<BodyState> bodies(1);
    bodies[0] = MakeBox(math::Vec3{0.0f, 0.0f, 0.0f}, box_vel0);
    std::vector<float> inv_mass = {kInvMassParticle};
    std::vector<math::Vec3> velocities = {particle_vel0};

    nuka::solver::SolveContext ctx;
    ctx.rows = &rows;
    ctx.state = &bodies;
    ctx.sides = &sides;
    ctx.dt = kDt;
    ctx.particles.inv_mass = &inv_mass;
    ctx.particles.velocities = &velocities;
    nuka::solver::UnifiedSolve(ctx, OneIterConfig());

    const float lambda = rows.rows[0].lambda;
    ASSERT_GT(lambda, 0.0f) << "no impulse -> the framework row did not engage";
    const math::Vec3 dv_p = velocities[0] - particle_vel0;
    const math::Vec3 dv_box = bodies[0].linear_velocity - box_vel0;
    EXPECT_GT(dv_p.y, 0.0f) << "particle did not react two-way";
    EXPECT_LT(dv_box.y, 0.0f) << "rigid body did not react two-way";
    const float dp = kMassParticle * dv_p.y + kMassBox * dv_box.y;
    EXPECT_NEAR(dp, 0.0f, 1.0e-4f) << "two-way momentum not balanced";
}

// ===========================================================================
// CASE 4: OFF-COM contact -> the rigid body's ANGULAR velocity reacts (the r x n
//   contact Jacobian is live for coupling rows). Hand-built so the closed form
//   includes the angular term; asserts box omega.z == invI * (r x n).z * lambda.
// ===========================================================================
TEST(CouplingFramework, OffCenterContactSpinsRigidBody) {
    const math::Vec3 n = Normal();
    std::vector<BodyState> bodies(1);
    bodies[0] = MakeBox(math::Vec3{0.0f, 0.0f, 0.0f}, BoxVel0());
    const uint32_t body_count = 1u;
    // Contact 0.2 m off the COM along +X. j_lin_b = -n; r = (0.2,0,0);
    // r x (-n) = (0.2,0,0) x (0,-1,0) = (0,0,-0.2).
    const math::Vec3 contact_point{0.2f, 0.0f, 0.0f};
    const math::Vec3 r = contact_point - bodies[0].position;
    const math::Vec3 ang_b = r.Cross(math::Vec3{0.0f, -1.0f, 0.0f});

    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    AppendCouplingNormalRow(&rows, &sides, MakeParticleRef(0u), MakeRigidRef(0u),
                            body_count + 0u, 0u, n, math::Vec3{0,-1,0},
                            kAref, kR, contact_point);

    std::vector<float> inv_mass = {kInvMassParticle};
    std::vector<math::Vec3> velocities = {ParticleVel0()};
    nuka::solver::SolveContext ctx;
    ctx.rows = &rows;
    ctx.state = &bodies;
    ctx.sides = &sides;
    ctx.dt = kDt;
    ctx.particles.inv_mass = &inv_mass;
    ctx.particles.velocities = &velocities;
    nuka::solver::UnifiedSolve(ctx, OneIterConfig());

    // Closed form WITH the angular term: A = invM_p + invM_box + invI*|r x n|^2.
    const float ang_sq = ang_b.Dot(ang_b);
    const float A = kInvMassParticle + kInvMassBox + kInvInertiaBox * ang_sq;
    const float eff_mass = 1.0f / (A + kR);
    const float jv0 = n.Dot(ParticleVel0()) + math::Vec3{0,-1,0}.Dot(BoxVel0());
    const float lambda = eff_mass * (kAref * kDt - jv0);

    EXPECT_NEAR(rows.rows[0].lambda, lambda, kTol);
    // The box SPINS: omega.z = invI * (r x n).z * lambda (the grasp-reorient path).
    const float expected_wz = kInvInertiaBox * ang_b.z * lambda;
    EXPECT_NEAR(bodies[0].angular_velocity.z, expected_wz, kTol);
    EXPECT_LT(bodies[0].angular_velocity.z, 0.0f) << "box did not spin under off-COM contact";
    EXPECT_GT(std::fabs(expected_wz), 1.0e-3f) << "angular reaction is vacuous";
}

// ===========================================================================
// CASE 5: D1. Two identical runs -> byte-identical particle velocities + box state
//   + solved lambda.
// ===========================================================================
TEST(CouplingFramework, DeterministicTwoRun) {
    auto run = [](std::vector<math::Vec3>* out_vel, BodyState* out_box, float* out_lambda) {
        std::vector<BodyState> bodies(1);
        bodies[0] = MakeBox(math::Vec3{0.0f, 0.0f, 0.0f}, BoxVel0());
        RowBuffers rows;
        std::vector<ContactRowSides> sides;
        AppendCouplingNormalRow(&rows, &sides, MakeParticleRef(0u), MakeRigidRef(0u),
                                1u, 0u, Normal(), math::Vec3{0,-1,0}, kAref, kR,
                                math::Vec3{0.15f, 0.0f, 0.0f});
        std::vector<float> inv_mass = {kInvMassParticle};
        std::vector<math::Vec3> velocities = {ParticleVel0()};
        nuka::solver::SolveContext ctx;
        ctx.rows = &rows;
        ctx.state = &bodies;
        ctx.sides = &sides;
        ctx.dt = kDt;
        ctx.particles.inv_mass = &inv_mass;
        ctx.particles.velocities = &velocities;
        nuka::solver::UnifiedSolve(ctx, OneIterConfig());
        *out_vel = velocities;
        *out_box = bodies[0];
        *out_lambda = rows.rows[0].lambda;
    };

    std::vector<math::Vec3> v1, v2;
    BodyState b1, b2;
    float l1 = 0.0f, l2 = 0.0f;
    run(&v1, &b1, &l1);
    run(&v2, &b2, &l2);
    EXPECT_EQ(std::memcmp(v1.data(), v2.data(), v1.size() * sizeof(math::Vec3)), 0);
    EXPECT_EQ(std::memcmp(&b1, &b2, sizeof(BodyState)), 0);
    EXPECT_EQ(l1, l2);
}
