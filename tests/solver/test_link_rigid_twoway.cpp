// ---------------------------------------------------------------------------
// v0.8 C5b-core: the ArticulationChainJ reaction arm wired into UnifiedSolve.
// ---------------------------------------------------------------------------
// C5b-core wires the reduced-coordinate Featherstone chain-J reaction (the C4c
// ArticulationReactionState math) into the compliant branch of the UnifiedSolve
// row_solver kernel -- the linchpin C5c (foot-ground) blocks on. C5a wired the
// rigid/static arms; this exercises the THREE articulation legs end-to-end on
// the GPU:
//   (1) Jv0      = sum_r chain_J[r] * qdot0[r]   (the NEW per-side Jv, which the
//                  shared ComputeJv would have dropped),
//   (2) denom    = J M^-1 J^T  (the effective-mass leg),
//   (3) apply    = qdot += M^-1 J^T dlambda  (the DOF apply leg).
//
// The ArticulationReactionState data is built BY HAND (no BatchedArticulatedWorld
// / device CRBA) so the M^-1 tile + chain-J row + qdot are exact + known. The
// C4c ArticulationEffectiveInvMass / ArticulationApplyImpulse math is already
// byte-proven (nuka_reaction_providers_test), so any mismatch here localizes to
// the WIRING (the threading / dispatch / the new ComputeCompliantJv).
//
// ★ ALL articulation tests START WITH NONZERO qdot. With qdot0 == 0 the missing-
//   Jv bug is INVISIBLE (Jv == 0 makes broken == correct), so the whole point of
//   wiring the third arm would be untestable.
//
// COLORING KEY (C5b design): an articulation side's row body-index is encoded as
// body_count + art_index (a distinct-per-articulation conflict key, >= body_count
// so the rigid-apply ValidBody path is inert -- dispatch is by side.react anyway).
// Same-articulation contacts therefore SERIALIZE into different graph colors
// (Gauss-Seidel correctness: no two lanes race on one qdot slice within a color).
// The TEST sets these body indices (the C5c foot-ground emitter sets them in
// production).
//
// Case 1 is a SINGLE compliant row, run with velocity_iterations == 1: the kernel
// folds R only into the denominator (no -R*lambda residual term), so it is a
// 1-iteration fixed point only at iter 1 (later iters drift lambda from
// (aref*dt-Jv0)/(A+R) toward /A). We assert against the exact closed form.
// ---------------------------------------------------------------------------

#include "constraint/collidable.hpp"
#include "constraint/contact_row_sides.hpp"
#include "constraint/row.hpp"
#include "constraint/row_articulation_refs.hpp"
#include "constraint/row_buffers.hpp"
#include "runtime/rigid/body_state.hpp"
#include "solver/gpu/row_scheduler.hpp"
#include "solver/unified_solve.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

using namespace nuka::constraint;
using nuka::runtime::rigid::BodyState;
namespace math = nuka::math;

namespace {

// A small hand-built articulation: dof_stride = 2, a symmetric positive-definite
// M^-1 tile, a chain-J row, and a NONZERO qdot. The M^-1 tile is the exact
// inverse of M = [[4,1],[1,3]] (det = 11): M^-1 = (1/11)[[3,-1],[-1,4]].
struct HandArticulation {
    static constexpr uint32_t kDof = 2u;
    // M^-1 = [[3/11, -1/11], [-1/11, 4/11]] (row-major, dof_stride x dof_stride).
    static std::vector<float> Minv() {
        return {3.0f / 11.0f, -1.0f / 11.0f, -1.0f / 11.0f, 4.0f / 11.0f};
    }
    // A chain-J row (the reduced-coordinate Jacobian for one contact).
    static std::vector<float> ChainJ() { return {0.7f, -0.4f}; }
    // NONZERO start-of-solve joint velocities.
    static std::vector<float> Qdot0() { return {0.9f, 0.3f}; }
};

// Closed-form leg-by-leg oracle for ONE articulation-vs-static compliant row.
struct ChainOracle {
    float jv0 = 0.0f;             // sum_r J[r]*qdot0[r]
    float eff_inv_mass = 0.0f;    // J M^-1 J^T  (the denominator term A)
    float lambda = 0.0f;          // (aref*dt - Jv0) / (A + R)
    std::vector<float> qdot_expected;  // qdot0 + (M^-1 J^T) * lambda
};

// n = dof_stride. Minv is n*n row-major; chain_j and qdot0 are length n.
ChainOracle ComputeChainOracle(const std::vector<float>& Minv,
                               const std::vector<float>& chain_j,
                               const std::vector<float>& qdot0,
                               float aref, float dt, float R) {
    const uint32_t n = static_cast<uint32_t>(chain_j.size());
    ChainOracle o;
    o.qdot_expected.assign(n, 0.0f);
    // Leg 1: Jv0 = sum_r J[r]*qdot0[r].
    for (uint32_t r = 0u; r < n; ++r) {
        o.jv0 += chain_j[r] * qdot0[r];
    }
    // Leg 2: A = J M^-1 J^T = sum_r J[r] * (sum_c Minv[r*n+c]*J[c]).
    std::vector<float> minv_jt(n, 0.0f);  // M^-1 J^T (reused by the apply).
    for (uint32_t r = 0u; r < n; ++r) {
        float acc = 0.0f;
        for (uint32_t c = 0u; c < n; ++c) {
            acc += Minv[static_cast<size_t>(r) * n + c] * chain_j[c];
        }
        minv_jt[r] = acc;
        o.eff_inv_mass += chain_j[r] * acc;
    }
    // lambda for a 1-iteration solve from lambda=0: (aref*dt - Jv0) / (A + R).
    const float effective_mass = (o.eff_inv_mass + R) > 1.0e-12f
                                     ? 1.0f / (o.eff_inv_mass + R)
                                     : 0.0f;
    o.lambda = effective_mass * (aref * dt - o.jv0);
    // Leg 3: qdot += M^-1 J^T * lambda.
    for (uint32_t r = 0u; r < n; ++r) {
        o.qdot_expected[r] = qdot0[r] + minv_jt[r] * o.lambda;
    }
    return o;
}

// 1-velocity-iteration config (see the header note on the R-in-denominator-only
// fixed point). No friction; baumgarte irrelevant (compliant rows skip it).
nuka::solver::SolverConfig OneIterConfig() {
    nuka::solver::SolverConfig cfg;
    cfg.velocity_iterations = 1u;
    cfg.position_iterations = 0u;
    cfg.slop = 0.0f;
    cfg.baumgarte = 0.0f;
    return cfg;
}

// Append ONE compliant normal row (single contact). side_a / side_b are the
// CollidableRefs; body_a / body_b the encoded coloring keys; jac_a / jac_b the
// per-side RowJacobian6 (the articulation side's jac is unused -- pass zero).
// rhs == aref (reference acceleration); compliance_alpha == R (dual regularizer).
void AppendCompliantNormalRow(RowBuffers* rows,
                              std::vector<ContactRowSides>* sides,
                              std::vector<RowArticulationRefs>* art_refs,
                              CollidableRef side_a, CollidableRef side_b,
                              uint32_t body_a, uint32_t body_b,
                              RowJacobian6 jac_a, RowJacobian6 jac_b,
                              float aref, float R,
                              RowArticulationSide art_a, RowArticulationSide art_b) {
    Row row;
    row.row_class_id = kMaximalContactRowClassId;
    row.rhs = aref;
    row.lambda = 0.0f;  // 1-iteration solve starts from zero impulse.
    row.lower = 0.0f;
    row.upper = kRowHugeLimit;  // unilateral normal: [0, huge]
    row.compliance_alpha = R;
    row.flags = row_flags::Compliant | row_flags::Unilateral;

    RowMaterial material;
    material.kind = RowKind::Contact;
    material.group_id = rows->RowCount();  // this row is its own normal-group head
    material.normal_row_count = 1u;
    material.first_friction_row = 0u;
    material.friction_row_count = 0u;
    material.friction = 0.0f;
    material.restitution = 0.0f;
    material.position_error = 0.0f;

    rows->AddRow(row, RowBodyPair{body_a, body_b}, jac_a, jac_b, material);
    sides->push_back(ContactRowSides{side_a, side_b});
    art_refs->push_back(RowArticulationRefs{art_a, art_b});
}

CollidableRef MakeArticulationRef(uint32_t handle) {
    CollidableRef ref;
    ref.type = CollidableType::ArticulationLink;
    ref.react = ReactionProviderKind::ArticulationChainJ;
    ref.handle = handle;
    return ref;
}

CollidableRef MakeStaticRef() {
    CollidableRef ref;
    ref.type = CollidableType::StaticWorld;
    ref.react = ReactionProviderKind::StaticNull;
    ref.handle = kInvalidBodyIndex;
    return ref;
}

CollidableRef MakeRigidRef(uint32_t handle) {
    CollidableRef ref;
    ref.type = CollidableType::RigidBody;
    ref.react = ReactionProviderKind::RigidInvMass;
    ref.handle = handle;
    return ref;
}

constexpr float kTol = 1.0e-5f;

}  // namespace

// ===========================================================================
// CASE 1: single articulation contact vs the closed form (the core oracle).
//   side A = ArticulationChainJ (2-DOF M^-1 tile + chain-J + NONZERO qdot),
//   side B = StaticWorld. One compliant normal row, velocity_iterations == 1.
//   Exercises ALL THREE legs (Jv0, denominator, apply).
// ===========================================================================
TEST(LinkRigidTwoWay, SingleArticulationContactMatchesClosedForm) {
    const std::vector<float> Minv = HandArticulation::Minv();
    const std::vector<float> chain_j = HandArticulation::ChainJ();
    const std::vector<float> qdot0 = HandArticulation::Qdot0();
    constexpr uint32_t kDof = HandArticulation::kDof;

    const float dt = 0.01f;
    const float R = 0.05f;
    // Pick aref so (aref*dt - Jv0) > 0 (so the [0, huge] clamp does NOT zero
    // lambda and qdot genuinely moves). Jv0 = 0.7*0.9 + (-0.4)*0.3 = 0.51.
    const float aref = 120.0f;  // aref*dt = 1.2 > Jv0 = 0.51

    const ChainOracle oracle = ComputeChainOracle(Minv, chain_j, qdot0, aref, dt, R);
    ASSERT_GT(aref * dt - oracle.jv0, 0.0f) << "test setup: clamp would zero lambda";
    ASSERT_NEAR(oracle.jv0, 0.51f, 1.0e-6f) << "hand Jv0 arithmetic check";

    // --- Build the row + the by-hand articulation buffers --------------------
    // A dummy, UNREFERENCED rigid body so body_count >= 1 (UnifiedSolve /
    // SolveRowsWithSides early-return on body_count == 0; the articulation-only /
    // body_count == 0 case is a C5c gap -- go2 foot-ground is link-vs-static, so
    // C5c hits it for real). No row names this body.
    std::vector<BodyState> bodies(1);
    bodies[0].inv_mass = 1.0f;
    const uint32_t body_count = static_cast<uint32_t>(bodies.size());

    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    std::vector<RowArticulationRefs> art_refs;

    // Articulation side coloring key = body_count + art_index (= 1 + 0 = 1).
    constexpr uint32_t kArtIndex = 0u;
    constexpr uint32_t kJacSlot = 0u;
    const RowJacobian6 zero_jac{math::Vec3::Zero(), math::Vec3::Zero()};
    AppendCompliantNormalRow(
        &rows, &sides, &art_refs,
        /*side_a=*/MakeArticulationRef(0u), /*side_b=*/MakeStaticRef(),
        /*body_a=*/body_count + kArtIndex, /*body_b=*/kInvalidBodyIndex,
        /*jac_a=*/zero_jac, /*jac_b=*/zero_jac, aref, R,
        RowArticulationSide{kArtIndex, kJacSlot}, RowArticulationSide{});

    std::vector<float> qdot = qdot0;  // MUTABLE; downloaded after the solve.

    nuka::solver::SolveContext ctx;
    ctx.rows = &rows;
    ctx.state = &bodies;
    ctx.sides = &sides;
    ctx.dt = dt;
    ctx.articulation.art_refs = &art_refs;
    ctx.articulation.chain_jacobians = &chain_j;  // one slot -> dof floats.
    ctx.articulation.inertia_m_inv = &Minv;
    ctx.articulation.qdot = &qdot;
    ctx.articulation.dof_stride = kDof;

    nuka::solver::UnifiedSolve(ctx, OneIterConfig());

    // --- Assertions: all three legs ----------------------------------------
    EXPECT_NEAR(rows.rows[0].lambda, oracle.lambda, kTol)
        << "solved lambda != closed form (aref*dt - Jv0)/(A + R)";
    ASSERT_EQ(qdot.size(), static_cast<size_t>(kDof));
    for (uint32_t r = 0u; r < kDof; ++r) {
        EXPECT_NEAR(qdot[r], oracle.qdot_expected[r], kTol)
            << "qdot[" << r << "] != qdot0 + (M^-1 J^T)*lambda";
    }
    // Sanity: qdot genuinely MOVED (the nonzero-qdot / wired-Jv proof).
    EXPECT_GT(std::fabs(qdot[0] - qdot0[0]) + std::fabs(qdot[1] - qdot0[1]), 1.0e-4f)
        << "qdot did not change -- the articulation apply arm did not fire";
}

// ===========================================================================
// CASE 2: two contacts on the SAME articulation (coloring).
//   Two normal rows, both side-A the same articulation (same art_index -> same
//   coloring key body_count+art_index), DIFFERENT chain_jac_slots. They must
//   land in DIFFERENT colors (serialized) so the kernel does not race on the one
//   qdot slice. Assert ValidateNoSharedBodiesPerColor holds + the solve is
//   stable/consistent with sequentially-applied single-row solves (PGS).
// ===========================================================================
TEST(LinkRigidTwoWay, TwoContactsSameArticulationSerializeAndAreConsistent) {
    const std::vector<float> Minv = HandArticulation::Minv();
    const std::vector<float> qdot0 = HandArticulation::Qdot0();
    constexpr uint32_t kDof = HandArticulation::kDof;
    const float dt = 0.01f;
    const float R = 0.05f;
    const float aref = 120.0f;

    // Two DISTINCT chain-J rows (different contacts on the same articulation),
    // packed in slot order: slot 0 then slot 1.
    const std::vector<float> j0 = {0.7f, -0.4f};
    const std::vector<float> j1 = {-0.3f, 0.6f};
    std::vector<float> chain_jacobians;  // dof_stride floats per slot.
    chain_jacobians.insert(chain_jacobians.end(), j0.begin(), j0.end());
    chain_jacobians.insert(chain_jacobians.end(), j1.begin(), j1.end());

    std::vector<BodyState> bodies(1);
    bodies[0].inv_mass = 1.0f;
    const uint32_t body_count = static_cast<uint32_t>(bodies.size());

    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    std::vector<RowArticulationRefs> art_refs;
    const RowJacobian6 zero_jac{math::Vec3::Zero(), math::Vec3::Zero()};
    constexpr uint32_t kArtIndex = 0u;
    for (uint32_t slot = 0u; slot < 2u; ++slot) {
        AppendCompliantNormalRow(
            &rows, &sides, &art_refs,
            MakeArticulationRef(0u), MakeStaticRef(),
            /*body_a=*/body_count + kArtIndex, /*body_b=*/kInvalidBodyIndex,
            zero_jac, zero_jac, aref, R,
            RowArticulationSide{kArtIndex, slot}, RowArticulationSide{});
    }

    // --- Coloring: same coloring key -> the two rows must be in different colors.
    const nuka::solver::gpu::RowColorPartitions partitions =
        nuka::solver::gpu::BuildRowColorPartitions(rows);
    EXPECT_TRUE(nuka::solver::gpu::ValidateNoSharedBodiesPerColor(rows, partitions))
        << "same-articulation rows must serialize into conflict-free colors";
    EXPECT_GE(partitions.ColorCount(), 2u)
        << "two same-articulation contacts must be in >= 2 colors (serialized)";

    // --- Run the kernel solve.
    std::vector<float> qdot = qdot0;
    nuka::solver::SolveContext ctx;
    ctx.rows = &rows;
    ctx.state = &bodies;
    ctx.sides = &sides;
    ctx.dt = dt;
    ctx.articulation.art_refs = &art_refs;
    ctx.articulation.chain_jacobians = &chain_jacobians;
    ctx.articulation.inertia_m_inv = &Minv;
    ctx.articulation.qdot = &qdot;
    ctx.articulation.dof_stride = kDof;
    nuka::solver::UnifiedSolve(ctx, OneIterConfig());

    // --- Reference: apply the two rows SEQUENTIALLY (Gauss-Seidel), each its own
    //     1-iter closed form, the second seeing the first's qdot update.
    std::vector<float> ref_qdot = qdot0;
    const ChainOracle o0 = ComputeChainOracle(Minv, j0, ref_qdot, aref, dt, R);
    ref_qdot = o0.qdot_expected;
    const ChainOracle o1 = ComputeChainOracle(Minv, j1, ref_qdot, aref, dt, R);
    ref_qdot = o1.qdot_expected;

    ASSERT_EQ(qdot.size(), ref_qdot.size());
    for (uint32_t r = 0u; r < kDof; ++r) {
        EXPECT_NEAR(qdot[r], ref_qdot[r], kTol)
            << "kernel qdot[" << r << "] != sequential Gauss-Seidel reference "
               "(coloring did not serialize the two contacts)";
    }
    // Both contacts must have produced a positive normal impulse.
    EXPECT_GT(rows.rows[0].lambda, 0.0f);
    EXPECT_GT(rows.rows[1].lambda, 0.0f);
}

// ===========================================================================
// CASE 3: mixed articulation <-> rigid (two-way).
//   ONE row, side A = ArticulationChainJ (nonzero qdot), side B = RigidInvMass
//   (a real finite-mass body with nonzero velocity). BOTH react: the rigid
//   velocity changes AND the articulation qdot changes; lambda is the shared
//   impulse, so Delta(rigid) = invM * Jr * lambda and Delta(qdot) = M^-1 J_a^T lambda.
// ===========================================================================
TEST(LinkRigidTwoWay, MixedArticulationRigidTwoWayReacts) {
    const std::vector<float> Minv = HandArticulation::Minv();
    const std::vector<float> chain_j = HandArticulation::ChainJ();
    const std::vector<float> qdot0 = HandArticulation::Qdot0();
    constexpr uint32_t kDof = HandArticulation::kDof;
    const float dt = 0.01f;
    const float R = 0.05f;
    const float aref = 120.0f;

    // Rigid side B: finite mass, NONZERO velocity. Pure-linear contact Jacobian
    // (a normal row): jac_b.linear = n, jac_b.angular = 0 (so inv_inertia unused).
    const float inv_mass_b = 0.5f;        // m = 2
    const math::Vec3 v_b0{0.2f, -0.1f, 0.05f};
    const math::Vec3 n_b{0.0f, 1.0f, 0.0f};  // contact normal on the rigid side
    const RowJacobian6 jac_b{n_b, math::Vec3::Zero()};

    std::vector<BodyState> bodies(1);
    bodies[0].inv_mass = inv_mass_b;
    bodies[0].inv_inertia = math::Vec3::Zero();
    bodies[0].linear_velocity = v_b0;
    const uint32_t kRigidBody = 0u;
    const uint32_t body_count = static_cast<uint32_t>(bodies.size());

    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    std::vector<RowArticulationRefs> art_refs;
    const RowJacobian6 zero_jac{math::Vec3::Zero(), math::Vec3::Zero()};
    constexpr uint32_t kArtIndex = 0u;
    AppendCompliantNormalRow(
        &rows, &sides, &art_refs,
        /*side_a=*/MakeArticulationRef(0u), /*side_b=*/MakeRigidRef(kRigidBody),
        /*body_a=*/body_count + kArtIndex, /*body_b=*/kRigidBody,
        /*jac_a=*/zero_jac, /*jac_b=*/jac_b, aref, R,
        RowArticulationSide{kArtIndex, 0u}, RowArticulationSide{});

    std::vector<float> qdot = qdot0;
    nuka::solver::SolveContext ctx;
    ctx.rows = &rows;
    ctx.state = &bodies;
    ctx.sides = &sides;
    ctx.dt = dt;
    ctx.articulation.art_refs = &art_refs;
    ctx.articulation.chain_jacobians = &chain_j;
    ctx.articulation.inertia_m_inv = &Minv;
    ctx.articulation.qdot = &qdot;
    ctx.articulation.dof_stride = kDof;
    nuka::solver::UnifiedSolve(ctx, OneIterConfig());

    const float lambda = rows.rows[0].lambda;
    EXPECT_GT(lambda, 0.0f) << "no impulse -> neither side reacts";

    // --- Closed-form two-way oracle: lambda = (aref*dt - Jv0_total)/(A_total + R).
    // Jv0_total = J_a . qdot0 (articulation) + dot(jac_b.lin, v_b0) (rigid).
    float jv_art = 0.0f;
    for (uint32_t r = 0u; r < kDof; ++r) jv_art += chain_j[r] * qdot0[r];
    const float jv_rigid = n_b.x * v_b0.x + n_b.y * v_b0.y + n_b.z * v_b0.z;
    const float jv0_total = jv_art + jv_rigid;
    // A_total = J M^-1 J^T (articulation) + invM |Jlin|^2 (rigid).
    std::vector<float> minv_jt(kDof, 0.0f);
    float A_art = 0.0f;
    for (uint32_t r = 0u; r < kDof; ++r) {
        float acc = 0.0f;
        for (uint32_t c = 0u; c < kDof; ++c) {
            acc += Minv[static_cast<size_t>(r) * kDof + c] * chain_j[c];
        }
        minv_jt[r] = acc;
        A_art += chain_j[r] * acc;
    }
    const float A_rigid = inv_mass_b * (n_b.x * n_b.x + n_b.y * n_b.y + n_b.z * n_b.z);
    const float A_total = A_art + A_rigid;
    const float effective_mass = 1.0f / (A_total + R);
    const float lambda_expected = effective_mass * (aref * dt - jv0_total);
    EXPECT_NEAR(lambda, lambda_expected, kTol) << "two-way shared impulse mismatch";

    // --- BOTH sides reacted with the shared impulse:
    // Rigid: Delta v = invM * Jlin * lambda (equal-and-opposite along the row dir).
    EXPECT_NEAR(bodies[0].linear_velocity.x, v_b0.x + inv_mass_b * n_b.x * lambda, kTol);
    EXPECT_NEAR(bodies[0].linear_velocity.y, v_b0.y + inv_mass_b * n_b.y * lambda, kTol);
    EXPECT_NEAR(bodies[0].linear_velocity.z, v_b0.z + inv_mass_b * n_b.z * lambda, kTol);
    EXPECT_GT(std::fabs(bodies[0].linear_velocity.y - v_b0.y), 1.0e-4f)
        << "rigid side did not react";
    // Articulation: Delta qdot = M^-1 J_a^T lambda.
    for (uint32_t r = 0u; r < kDof; ++r) {
        EXPECT_NEAR(qdot[r], qdot0[r] + minv_jt[r] * lambda, kTol);
    }
    EXPECT_GT(std::fabs(qdot[0] - qdot0[0]) + std::fabs(qdot[1] - qdot0[1]), 1.0e-4f)
        << "articulation side did not react";
}

// ===========================================================================
// CASE 4: D1 two-run byte-exact (qdot + bodies).
//   Run case 1 twice (resetting qdot to qdot0 between runs) and memcmp.
// ===========================================================================
TEST(LinkRigidTwoWay, DeterministicTwoRunByteExact) {
    const std::vector<float> Minv = HandArticulation::Minv();
    const std::vector<float> chain_j = HandArticulation::ChainJ();
    const std::vector<float> qdot0 = HandArticulation::Qdot0();
    constexpr uint32_t kDof = HandArticulation::kDof;
    const float dt = 0.01f;
    const float R = 0.05f;
    const float aref = 120.0f;

    auto run_once = [&](std::vector<float>* qdot_out, BodyState* body_out) {
        std::vector<BodyState> bodies(1);
        bodies[0].inv_mass = 1.0f;
        const uint32_t body_count = static_cast<uint32_t>(bodies.size());

        RowBuffers rows;
        std::vector<ContactRowSides> sides;
        std::vector<RowArticulationRefs> art_refs;
        const RowJacobian6 zero_jac{math::Vec3::Zero(), math::Vec3::Zero()};
        AppendCompliantNormalRow(
            &rows, &sides, &art_refs,
            MakeArticulationRef(0u), MakeStaticRef(),
            body_count + 0u, kInvalidBodyIndex,
            zero_jac, zero_jac, aref, R,
            RowArticulationSide{0u, 0u}, RowArticulationSide{});

        std::vector<float> qdot = qdot0;  // reset to qdot0 each run.
        nuka::solver::SolveContext ctx;
        ctx.rows = &rows;
        ctx.state = &bodies;
        ctx.sides = &sides;
        ctx.dt = dt;
        ctx.articulation.art_refs = &art_refs;
        ctx.articulation.chain_jacobians = &chain_j;
        ctx.articulation.inertia_m_inv = &Minv;
        ctx.articulation.qdot = &qdot;
        ctx.articulation.dof_stride = kDof;
        nuka::solver::UnifiedSolve(ctx, OneIterConfig());

        *qdot_out = qdot;
        *body_out = bodies[0];
    };

    std::vector<float> qa, qb;
    BodyState ba{}, bb{};
    run_once(&qa, &ba);
    run_once(&qb, &bb);

    ASSERT_EQ(qa.size(), qb.size());
    EXPECT_EQ(std::memcmp(qa.data(), qb.data(), qa.size() * sizeof(float)), 0)
        << "two identical UnifiedSolve runs produced different qdot (nondeterministic)";
    EXPECT_EQ(std::memcmp(&ba, &bb, sizeof(BodyState)), 0)
        << "two identical UnifiedSolve runs produced different bodies (nondeterministic)";
}
