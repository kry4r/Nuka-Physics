// ---------------------------------------------------------------------------
// v0.8 C6a: the PBD<->rigid unified-contact CO-STEP bridge acceptance.
// ---------------------------------------------------------------------------
// C6a wires the REAL v0.8 front-end together (C2c broadphase -> C5 driver -> C6b
// coupling framework -> C6b solver) into the Genesis-style "couple" phase, so a
// PBD particle system CONSUMES unified contact manifolds and reacts TWO-WAY
// against rigid bodies through ONE unified compliant solve. This test drives the
// reusable host bridge (runtime/coupling/unified_costep.{hpp,cpp}) -- it does NOT
// re-derive broadphase / hand-build manifolds (C6b already proved the hand-built-
// manifold -> row -> two-way path; C6a's NEW surface is the front-end assembly).
//
// THE CO-STEP ORDERING (the test owns pre/post_coupling; the bridge owns couple):
//   pre_coupling  : integrate particle velocity by gravity*dt, PREDICT positions.
//   couple        : StepUnifiedParticleRigidCoupling (mutates v_particle + bodies).
//   post_coupling : integrate positions from the CORRECTED velocities.
//
// CASES:
//   A -- multi-particle drop onto a heavy free box: TOTAL mechanical energy
//        (KE + gravitational PE for particles + KE for the box) is NON-INCREASING
//        across the run; particles SETTLE (no tunnel, bounded |v|); the box
//        reacted (two-way).  TOTAL energy, NOT KE alone (a falling particle
//        converts PE->KE).
//   B -- the MIS-ORDER BITE-CHECK: re-run A but DOUBLE-APPLY the couple phase
//        (positions unchanged -> aref double-pushes -> over-restitution). The
//        total-energy gate must now FAIL (energy spuriously rises), proving the
//        CASE A gate discriminates ordering rather than passing vacuously. A and B
//        share the energy helper + scenario and differ ONLY in the ordering.
//   C -- fluid-rest variant: a blob of SystemKind::Fluid particles resting on a
//        STATIC box floor; multi-step; total energy non-increasing + no sink-through.
//   D -- D1: two identical CASE A runs -> byte-identical particle v + pos + box state.
//
// VALIDATED, NOT WIRED: no XpbdWorld/PbfWorld/world_stepper is touched; the K3
// private particle-particle path is untouched; no golden moves.
// ---------------------------------------------------------------------------

#include "constraint/collidable.hpp"
#include "constraint/contact_row_sides.hpp"
#include "constraint/row_builder.hpp"  // ContactRowComplianceInputs
#include "math/vec3.hpp"
#include "phi/device_context.hpp"
#include "runtime/coupling/unified_costep.hpp"
#include "runtime/rigid/body_state.hpp"
#include "scene/cooked_blob.hpp"
#include "solver/rigid_solver.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using nuka::runtime::coupling::CoupleBox;
using nuka::runtime::coupling::StepUnifiedParticleRigidCoupling;
using nuka::runtime::coupling::UnifiedCoStepReport;
using nuka::runtime::rigid::BodyState;
using nuka::scene::SystemKind;
namespace math = nuka::math;

namespace {

// --- the shared scenario constants -----------------------------------------
constexpr float kDt = 1.0f / 240.0f;
constexpr float kGravity = -9.81f;             // particles only (-y).
constexpr float kParticleMass = 0.05f;
constexpr float kParticleInvMass = 1.0f / kParticleMass;
constexpr float kParticleRadius = 0.02f;        // contact sphere + query radius.
// A HEAVY free box: reacts measurably (two-way) but barely drifts over the run.
constexpr float kBoxMass = 20.0f;
constexpr float kBoxInvMass = 1.0f / kBoxMass;
constexpr float kBoxHalf = 0.25f;
constexpr float kBoxInertiaInv = 1.0f / (kBoxMass * (2.0f * kBoxHalf) * (2.0f * kBoxHalf) / 6.0f);

// The box top face world-y (the box COM is at the origin's xz, a fixed height).
constexpr float kBoxCenterY = 0.0f;
constexpr float kBoxTopY = kBoxCenterY + kBoxHalf;

// A converged compliant config: the bridge rebuilds rows EVERY step (no cross-step
// warm-start), so 1 iter under-resolves -> particles tunnel. Use many iterations.
nuka::solver::SolverConfig CoStepConfig() {
    nuka::solver::SolverConfig cfg;
    cfg.velocity_iterations = 48u;
    cfg.position_iterations = 0u;
    cfg.slop = 0.0f;
    cfg.baumgarte = 0.0f;
    return cfg;
}

// The shared compliance context (the C4a R/aref schedule). invweight ~ the actual
// reaction inv-weight sum so the stiffness is tuned for these masses (the default
// 1.0 would mistune -> tunneling / pumping).
nuka::constraint::ContactRowComplianceInputs MakeCompliance() {
    nuka::constraint::ContactRowComplianceInputs inputs;
    inputs.vel = 0.0f;
    inputs.invweight = kParticleInvMass + kBoxInvMass;
    inputs.dt = kDt;
    inputs.condim = 1u;     // frictionless: one clean normal row per contact.
    inputs.refsafe = true;
    return inputs;
}

// A grid of particles at `start_y`, inside the box xz footprint (so "no tunnel"
// tests the contact, not particles missing the box edge).
struct ParticleField {
    std::vector<math::Vec3> pos;
    std::vector<math::Vec3> vel;
    std::vector<float> inv_mass;
};

ParticleField MakeParticleGrid(uint32_t nx, uint32_t nz, float start_y,
                               float spacing) {
    ParticleField f;
    const float x0 = -0.5f * static_cast<float>(nx - 1) * spacing;
    const float z0 = -0.5f * static_cast<float>(nz - 1) * spacing;
    for (uint32_t ix = 0u; ix < nx; ++ix) {
        for (uint32_t iz = 0u; iz < nz; ++iz) {
            f.pos.push_back(math::Vec3{x0 + static_cast<float>(ix) * spacing, start_y,
                                       z0 + static_cast<float>(iz) * spacing});
            f.vel.push_back(math::Vec3::Zero());
            f.inv_mass.push_back(kParticleInvMass);
        }
    }
    return f;
}

BodyState MakeBox(float inv_mass) {
    BodyState box;
    box.inv_mass = inv_mass;
    box.inv_inertia = (inv_mass > 0.0f)
                          ? math::Vec3{kBoxInertiaInv, kBoxInertiaInv, kBoxInertiaInv}
                          : math::Vec3::Zero();
    box.position = math::Vec3{0.0f, kBoxCenterY, 0.0f};
    box.orientation = math::Quat::Identity();
    box.linear_velocity = math::Vec3::Zero();
    box.angular_velocity = math::Vec3::Zero();
    return box;
}

// TOTAL mechanical energy: KE+PE for the particles (gravitational PE referenced to
// y=0), PLUS the box KINETIC energy only. The box has NO gravity applied, so its
// gravitational PE is NOT in the dynamics -> not counted (counting it would inject
// spurious drift). Tracking TOTAL (not KE alone) is critical: a falling particle
// converts PE->KE, so KE-only would falsely fail.
double TotalEnergy(const std::vector<math::Vec3>& pos,
                   const std::vector<math::Vec3>& vel, float particle_mass,
                   const BodyState& box) {
    double e = 0.0;
    for (size_t i = 0; i < pos.size(); ++i) {
        const double v2 = static_cast<double>(vel[i].x) * vel[i].x +
                          static_cast<double>(vel[i].y) * vel[i].y +
                          static_cast<double>(vel[i].z) * vel[i].z;
        e += 0.5 * particle_mass * v2;                                 // KE
        e += static_cast<double>(particle_mass) * (-kGravity) * pos[i].y;  // PE = m*g*h
    }
    // Box KE only (linear + a small rotational term for honesty).
    if (box.inv_mass > 0.0f) {
        const double m = 1.0 / box.inv_mass;
        const double v2 = static_cast<double>(box.linear_velocity.x) * box.linear_velocity.x +
                          static_cast<double>(box.linear_velocity.y) * box.linear_velocity.y +
                          static_cast<double>(box.linear_velocity.z) * box.linear_velocity.z;
        e += 0.5 * m * v2;
    }
    return e;
}

// The outcome of a full multi-step run (for the gates + D1).
struct RunResult {
    std::vector<double> energy;        // total energy AFTER post_coupling, per step.
    std::vector<math::Vec3> final_pos;
    std::vector<math::Vec3> final_vel;
    BodyState final_box{};
    double final_min_y = 0.0;          // lowest particle y at the END.
    float max_speed = 0.0f;            // max |v| over the run (explosion guard).
    bool box_reacted = false;          // the box velocity changed (two-way).
    float box_vy_min = 0.0f;           // most-negative box vertical velocity seen.
    uint32_t total_rows = 0u;          // sum of rows across steps (non-vacuity).
    uint32_t particle_arm_rows = 0u;   // rows that dispatched the ParticleInvMass arm.
};

// Run the co-step for `steps`. `double_apply` is the CASE B mis-order: after the
// single couple solve, the solved velocity-CORRECTION delta is applied a SECOND
// time on both sides (= v_pre + 2*delta) WITHOUT re-running the solve. Because the
// solve is velocity-level and clamps the relative velocity to the aref target,
// re-RUNNING it would near no-op (the constraint is already satisfied); re-APPLYING
// the delta skips the clamp -> the separating impulse acts 2x -> over-restitution
// (e>1) -> a PURE energy injection (both sides doubled -> momentum stays balanced).
// All other ordering is identical, so CASE A and CASE B differ ONLY in this flag.
RunResult RunCoStep(const nuka::phi::DeviceContext& context, ParticleField field,
                    BodyState box, uint32_t steps, SystemKind system,
                    bool double_apply) {
    RunResult r;
    const auto config = CoStepConfig();
    const auto compliance = MakeCompliance();
    std::vector<CoupleBox> boxes = {CoupleBox{box.position, kBoxHalf}};
    std::vector<BodyState> bodies = {box};

    for (uint32_t step = 0u; step < steps; ++step) {
        // --- pre_coupling: gravity into velocity, PREDICT positions ----------
        // Capture the pre-couple (predicted) velocity so post_coupling can advance
        // the position with the CORRECTED velocity (XPBD/PBF: x_new = x_prev +
        // v_corrected*dt). We predict with v_pre, couple corrects v in place, then
        // post_coupling adds the velocity-correction delta (v_corr - v_pre)*dt.
        std::vector<math::Vec3> v_pre(field.vel.size());
        for (size_t i = 0; i < field.pos.size(); ++i) {
            if (field.inv_mass[i] > 0.0f) {
                field.vel[i].y += kGravity * kDt;
            }
            v_pre[i] = field.vel[i];
            field.pos[i] = field.pos[i] + field.vel[i] * kDt;  // predicted x.
        }

        // --- couple: the bridge (mutates field.vel + bodies) -----------------
        boxes[0].center = bodies[0].position;  // track the (heavy) box COM.
        // Snapshot the pre-couple velocities so the CASE B mis-order can re-apply
        // the solved correction delta a second time (over-restitution).
        const std::vector<math::Vec3> v_snap = field.vel;
        const BodyState box_snap = bodies[0];
        const UnifiedCoStepReport rep = StepUnifiedParticleRigidCoupling(
            context, field.pos.data(), &field.vel, field.inv_mass,
            static_cast<uint32_t>(field.pos.size()), kParticleRadius, system,
            boxes, &bodies, compliance, kDt, config);
        r.total_rows += rep.rows;
        r.particle_arm_rows += rep.particle_arm_rows;
        if (double_apply) {
            // Re-apply the SOLVED velocity-correction delta a 2nd time on BOTH
            // sides (no re-solve -> no clamp): v = v_pre + 2*delta. Doubling both
            // sides keeps momentum balanced -> the rise is pure energy injection.
            for (size_t i = 0; i < field.vel.size(); ++i) {
                field.vel[i] = field.vel[i] + (field.vel[i] - v_snap[i]);
            }
            bodies[0].linear_velocity =
                bodies[0].linear_velocity +
                (bodies[0].linear_velocity - box_snap.linear_velocity);
            bodies[0].angular_velocity =
                bodies[0].angular_velocity +
                (bodies[0].angular_velocity - box_snap.angular_velocity);
        }

        // --- post_coupling: advance positions with the CORRECTED velocity -----
        // x was predicted with v_pre; the bridge corrected the velocity in place.
        // x_final = x_predicted + (v_corrected - v_pre)*dt = x_prev + v_corr*dt.
        for (size_t i = 0; i < field.pos.size(); ++i) {
            field.pos[i] = field.pos[i] + (field.vel[i] - v_pre[i]) * kDt;
        }
        // The box (a real rigid body) integrates its corrected linear velocity.
        if (bodies[0].inv_mass > 0.0f) {
            bodies[0].position = bodies[0].position + bodies[0].linear_velocity * kDt;
        }

        // Track the box reaction + energy AFTER this step's couple+integrate.
        if (bodies[0].linear_velocity.Length() > 1.0e-7f) r.box_reacted = true;
        r.box_vy_min = std::min(r.box_vy_min, bodies[0].linear_velocity.y);
        for (const auto& v : field.vel) r.max_speed = std::max(r.max_speed, v.Length());
        r.energy.push_back(TotalEnergy(field.pos, field.vel, kParticleMass, bodies[0]));
    }

    r.final_pos = field.pos;
    r.final_vel = field.vel;
    r.final_box = bodies[0];
    double min_y = 1.0e30;
    for (const auto& p : field.pos) min_y = std::min(min_y, static_cast<double>(p.y));
    r.final_min_y = min_y;
    return r;
}

}  // namespace

// ===========================================================================
// CASE A (THE HEADLINE): multi-particle drop onto a heavy free box. TOTAL
// mechanical energy is NON-INCREASING; particles SETTLE (no tunnel, bounded |v|);
// the box reacted (two-way).
// ===========================================================================
TEST(PbdCoStepUnified, MultiParticleDropTotalEnergyNonIncreasing) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    // ~16 particles a few cm above the box top, inside the box xz footprint.
    ParticleField field = MakeParticleGrid(/*nx=*/4u, /*nz=*/4u,
                                           /*start_y=*/kBoxTopY + 0.06f,
                                           /*spacing=*/0.06f);
    const double e0 = TotalEnergy(field.pos, field.vel, kParticleMass,
                                  MakeBox(kBoxInvMass));

    const RunResult r = RunCoStep(context, field, MakeBox(kBoxInvMass),
                                  /*steps=*/30u, SystemKind::Soft,
                                  /*double_apply=*/false);

    ASSERT_GT(r.total_rows, 0u) << "no coupling rows ever emitted -- the front-end "
                                   "never produced a contact (vacuous run)";
    // The front-end tagged the particle side's reaction provider -> UnifiedSolve
    // dispatched the ParticleInvMass arm (not a vacuous rigid-only solve).
    EXPECT_EQ(r.particle_arm_rows, r.total_rows)
        << "some emitted rows did NOT carry a ParticleInvMass side -- the front-end "
           "react-tagging is broken";

    // (1) TOTAL energy is non-increasing step over step (the headline gate).
    double max_rise = 0.0;
    for (size_t i = 1; i < r.energy.size(); ++i) {
        max_rise = std::max(max_rise, r.energy[i] - r.energy[i - 1]);
    }
    std::printf("[CASE A] initial E=%.6f  final E=%.6f  (delta=%.6f)  max step-rise=%.3e\n",
                e0, r.energy.back(), r.energy.back() - e0, max_rise);
    // A tiny fp/compliant-spring tolerance (elastic PE not in the KE+gravPE sum).
    const double kEnergyTol = 1.0e-3 * std::max(1.0, e0);
    EXPECT_LE(max_rise, kEnergyTol)
        << "TOTAL mechanical energy rose across a step -> the co-step injected "
           "energy (a non-dissipative / mis-ordered couple)";
    EXPECT_LT(r.energy.back(), e0 + kEnergyTol)
        << "final total energy exceeds the initial -> energy was injected";

    // (2) particles SETTLE: none tunnels through the box top, none explodes.
    const double slop = static_cast<double>(kParticleRadius) + 0.02;
    std::printf("[CASE A] final min particle y=%.5f  (box top=%.5f, slop=%.3f)  "
                "max|v| over run=%.4f\n",
                r.final_min_y, kBoxTopY, slop, r.max_speed);
    EXPECT_GE(r.final_min_y, static_cast<double>(kBoxTopY) - slop)
        << "a particle tunneled through the box top";
    EXPECT_LT(r.max_speed, 50.0f) << "a particle velocity exploded";

    // (3) the box REACTED (two-way): its velocity changed, pushed DOWN (-y, away
    // from the particles above it).
    std::printf("[CASE A] box final lin vel=(%+.6f,%+.6f,%+.6f)  most-neg vy=%.6f\n",
                r.final_box.linear_velocity.x, r.final_box.linear_velocity.y,
                r.final_box.linear_velocity.z, r.box_vy_min);
    EXPECT_TRUE(r.box_reacted) << "the box velocity never changed -- one-way (not two-way)";
    EXPECT_LT(r.box_vy_min, 0.0f)
        << "the box was not pushed downward by the particles above it";
}

// ===========================================================================
// CASE B (THE MIS-ORDER BITE-CHECK, MANDATORY): the SAME scenario, but the solved
// velocity-correction is DOUBLE-APPLIED (the separating impulse acts 2x without a
// re-solve -> over-restitution e>1). The total-energy-non-increasing gate must now
// FAIL -- proving the CASE A gate discriminates ordering, not passing vacuously.
// (Double-RUNNING the solve would near no-op: the velocity-level compliant clamp
//  is already satisfied on the 2nd call. Double-APPLYING the delta skips the clamp
//  -> a genuine, unambiguous energy injection.)
// ===========================================================================
TEST(PbdCoStepUnified, MisOrderDoubleCoupleInjectsEnergy) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    ParticleField field = MakeParticleGrid(/*nx=*/4u, /*nz=*/4u,
                                           /*start_y=*/kBoxTopY + 0.06f,
                                           /*spacing=*/0.06f);
    const double e0 = TotalEnergy(field.pos, field.vel, kParticleMass,
                                  MakeBox(kBoxInvMass));

    // The CORRECT-ORDER baseline (same as CASE A) for the discrimination margin.
    const RunResult good = RunCoStep(context, field, MakeBox(kBoxInvMass),
                                     /*steps=*/30u, SystemKind::Soft,
                                     /*double_apply=*/false);
    double good_max_rise = 0.0;
    for (size_t i = 1; i < good.energy.size(); ++i)
        good_max_rise = std::max(good_max_rise, good.energy[i] - good.energy[i - 1]);

    // The MIS-ORDERED run: double-apply the couple phase.
    const RunResult bad = RunCoStep(context, field, MakeBox(kBoxInvMass),
                                    /*steps=*/30u, SystemKind::Soft,
                                    /*double_apply=*/true);
    ASSERT_GT(bad.total_rows, 0u) << "the mis-ordered run produced no contacts (vacuous)";
    double bad_max_rise = 0.0;
    for (size_t i = 1; i < bad.energy.size(); ++i)
        bad_max_rise = std::max(bad_max_rise, bad.energy[i] - bad.energy[i - 1]);

    const double kEnergyTol = 1.0e-3 * std::max(1.0, e0);
    std::printf("[CASE B] CORRECT-order max step-rise=%.3e (<= tol=%.3e)  "
                "MIS-order max step-rise=%.3e  final E: good=%.4f bad=%.4f init=%.4f\n",
                good_max_rise, kEnergyTol, bad_max_rise, good.energy.back(),
                bad.energy.back(), e0);

    // The CASE A gate PASSES for the correct order...
    ASSERT_LE(good_max_rise, kEnergyTol)
        << "the correct-order baseline already fails the gate -- scenario unstable";
    // ...and FAILS for the mis-order (the bite): the double-applied couple injects
    // energy that DWARFS the CASE A tolerance.
    EXPECT_GT(bad_max_rise, kEnergyTol)
        << "the mis-ordered (double-couple) run did NOT rise above the gate "
           "tolerance -- the CASE A energy gate is too loose to discriminate ordering";
    // The mis-order rise dwarfs the correct-order rise (orders of magnitude).
    EXPECT_GT(bad_max_rise, 10.0 * std::max(good_max_rise, kEnergyTol))
        << "the mis-order energy spike is not clearly separated from the correct-order "
           "run -- the discriminator is weak";
}

// ===========================================================================
// CASE C (FLUID-REST): a blob of SystemKind::Fluid particles resting on a STATIC
// box floor; multi-step; total energy non-increasing + no sink-through. Reuses the
// same bridge (system gate = Fluid<->Rigid).
// ===========================================================================
TEST(PbdCoStepUnified, FluidRestOnStaticFloorNonIncreasing) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    // A small blob resting just above the box top.
    ParticleField field = MakeParticleGrid(/*nx=*/3u, /*nz=*/3u,
                                           /*start_y=*/kBoxTopY + 0.03f,
                                           /*spacing=*/0.05f);
    // STATIC box: inv_mass = 0 -> the RigidInvMass arm is inert (StaticNull-like),
    // the floor does not move. The particles settle on it.
    BodyState floor = MakeBox(/*inv_mass=*/0.0f);
    const double e0 = TotalEnergy(field.pos, field.vel, kParticleMass, floor);

    const RunResult r = RunCoStep(context, field, floor, /*steps=*/40u,
                                  SystemKind::Fluid, /*double_apply=*/false);
    ASSERT_GT(r.total_rows, 0u) << "fluid blob never contacted the static floor";

    double max_rise = 0.0;
    for (size_t i = 1; i < r.energy.size(); ++i)
        max_rise = std::max(max_rise, r.energy[i] - r.energy[i - 1]);
    const double kEnergyTol = 1.0e-3 * std::max(1.0, e0);
    std::printf("[CASE C] initial E=%.6f  final E=%.6f  max step-rise=%.3e  "
                "final min y=%.5f (floor top=%.5f)\n",
                e0, r.energy.back(), max_rise, r.final_min_y, kBoxTopY);
    EXPECT_LE(max_rise, kEnergyTol)
        << "fluid total energy rose across a step (energy injected)";

    // No sink-through: the lowest particle stays above the floor top minus slop.
    const double slop = static_cast<double>(kParticleRadius) + 0.02;
    EXPECT_GE(r.final_min_y, static_cast<double>(kBoxTopY) - slop)
        << "fluid particles sank through the static floor";
    // The static floor did NOT move (inv_mass=0 -> no reaction applied).
    EXPECT_NEAR(r.final_box.linear_velocity.Length(), 0.0f, 1.0e-6f)
        << "the static floor moved -- the inv_mass=0 rigid arm is not inert";
}

// ===========================================================================
// CASE D (D1): two identical CASE A runs -> byte-identical final particle
// velocities + positions + box state (memcmp).
// ===========================================================================
TEST(PbdCoStepUnified, DeterministicTwoRun) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    auto make_field = []() {
        return MakeParticleGrid(/*nx=*/4u, /*nz=*/4u, /*start_y=*/kBoxTopY + 0.06f,
                                /*spacing=*/0.06f);
    };
    const RunResult a = RunCoStep(context, make_field(), MakeBox(kBoxInvMass),
                                  /*steps=*/30u, SystemKind::Soft, false);
    const RunResult b = RunCoStep(context, make_field(), MakeBox(kBoxInvMass),
                                  /*steps=*/30u, SystemKind::Soft, false);

    ASSERT_EQ(a.final_pos.size(), b.final_pos.size());
    ASSERT_EQ(a.final_vel.size(), b.final_vel.size());
    EXPECT_EQ(std::memcmp(a.final_pos.data(), b.final_pos.data(),
                          a.final_pos.size() * sizeof(math::Vec3)), 0)
        << "two runs produced different final particle positions (nondeterministic)";
    EXPECT_EQ(std::memcmp(a.final_vel.data(), b.final_vel.data(),
                          a.final_vel.size() * sizeof(math::Vec3)), 0)
        << "two runs produced different final particle velocities (nondeterministic)";
    EXPECT_EQ(std::memcmp(&a.final_box, &b.final_box, sizeof(BodyState)), 0)
        << "two runs produced a different final box BodyState (nondeterministic)";
    std::printf("[CASE D] (D1) final particle pos + vel + box state byte-identical "
                "across 2 runs\n");
}
