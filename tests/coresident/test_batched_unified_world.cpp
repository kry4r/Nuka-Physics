// ---------------------------------------------------------------------------
// v0.8 P2.1 -- BatchedUnifiedWorld skeleton + per-env MOVABLE RIGID BODIES under
// gravity (free-fall, NO contact, NO articulation). The first increment of the
// GENERAL, scene-driven, BATCHED world (roadmap §7). Gates:
//   (1) FreeFallMatchesSymplecticEuler : every env's body matches the EXACT
//       discrete symplectic-Euler recurrence (the scheme) AND the continuous
//       projectile within the discretization tolerance (it is REAL free-fall).
//   (2) PerEnvIndependence : N envs with DIFFERENT initial conditions each follow
//       their OWN trajectory -- the batching property (no cross-env contamination).
//   (3) DeterministicTwoRun : two worlds, same setup, byte-exact across all envs (D1).
//   (4) ImmovableBodyDoesNotMove : an inv_mass=0 body is frozen (matches the
//       co-resident IntegrateBoxPosition inv_mass<=0 skip).
// Mirrors the validated UnifiedCoResidentStepper integrator gates; the recurrence
// is BYTE-IDENTICAL to UnifiedCoResidentStepper::IntegrateBoxPosition so an N=1
// world matches the co-resident oracle.
// ---------------------------------------------------------------------------

#include "collision/analytical_manifold.hpp"   // amf::PrimParams / BuildPrimFrame
#include "collision/candidate_pair.hpp"        // CandidatePair / CollidableRef
#include "collision/contact_stream_driver.hpp" // BuildContactManifolds / ResolvedShape
#include "collision/convex_narrowphase.hpp"    // cvx::ConvexHullView (cup hull seam)
#include "constraint/contact_manifold.hpp"     // ContactManifold
#include "constraint/contact_row_sides.hpp"    // ContactRowSides
#include "constraint/reaction_provider.hpp"    // ReactionProviderKind
#include "constraint/row.hpp"                  // kInvalidBodyIndex / row_flags
#include "constraint/row_articulation_refs.hpp"  // RowArticulationRefs / RowArticulationSide
#include "constraint/row_buffers.hpp"          // RowBuffers / RowJacobian6
#include "constraint/row_builder.hpp"          // EmitCompliantContactRows / inputs
#include "import/usd_importer.hpp"             // LoadUsd (Gate A2 go2 scene + cup)
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer.hpp"             // UploadVector
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"  // UpdateWorldLinkPoses / DofCount
#include "runtime/articulation/articulation_drives.hpp"    // LaunchApplyTorqueDriveKernels
#include "runtime/articulation/articulation_jacobian.hpp"  // ComputeContactChainJacobians + M
#include "runtime/articulation/articulation_state.hpp"     // BuildArticulationHostState
#include "runtime/articulation/featherstone_aba.hpp"
#include "runtime/coresident/batched_unified_world.hpp"
#include "runtime/coresident/unified_coresident_stepper.hpp"  // Gate A2 oracle + grasp types
#include "runtime/rigid/body_state.hpp"
#include "runtime/world_builder.hpp"           // BuildWorld
#include "scene/canonical_types.hpp"           // scene::ShapeType
#include "scene/cooker.hpp"                    // CookScene
#include "solver/rigid_solver.hpp"             // SolverConfig
#include "solver/unified_solve.hpp"            // UnifiedSolve / SolveContext

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace coresident = nuka::runtime::coresident;
using nuka::math::Quat;
using nuka::math::Vec3;
using nuka::runtime::rigid::BodyState;

// A movable unit body (mirrors the co-resident MakeBox: inv_mass 1, diagonal
// inv_inertia so a constant body-frame spin stays a clean principal-axis spin).
BodyState MakeBody(const Vec3& position,
                   const Vec3& linear_velocity,
                   const Vec3& angular_velocity) {
    BodyState b;
    b.inv_mass = 1.0f;
    b.inv_inertia = Vec3{600.0f, 600.0f, 600.0f};
    b.position = position;
    b.linear_velocity = linear_velocity;
    b.angular_velocity = angular_velocity;
    b.orientation = Quat::Identity();
    return b;
}

// The EXACT discrete reference: the same stage order + float ops as
// BatchedUnifiedWorld::Step() / IntegrateBodyPosition (gravity velocity-kick, then
// symplectic-Euler position + quaternion advance). Used as the byte-exact oracle.
BodyState ReferenceFreeFall(BodyState b, float gravity_z, float dt,
                            uint32_t steps) {
    for (uint32_t s = 0u; s < steps; ++s) {
        // velocity stage
        b.linear_velocity.z += gravity_z * dt;
        // position stage
        b.position += b.linear_velocity * dt;
        const Vec3 w = b.angular_velocity;
        Quat dq;
        dq.w = 1.0f;
        dq.x = 0.5f * w.x * dt;
        dq.y = 0.5f * w.y * dt;
        dq.z = 0.5f * w.z * dt;
        b.orientation = (b.orientation * dq).Normalized();
    }
    return b;
}

void ExpectBodyEq(const BodyState& a, const BodyState& b, float tol,
                  const char* what) {
    EXPECT_NEAR(a.position.x, b.position.x, tol) << what << " position.x";
    EXPECT_NEAR(a.position.y, b.position.y, tol) << what << " position.y";
    EXPECT_NEAR(a.position.z, b.position.z, tol) << what << " position.z";
    EXPECT_NEAR(a.linear_velocity.x, b.linear_velocity.x, tol) << what << " vx";
    EXPECT_NEAR(a.linear_velocity.y, b.linear_velocity.y, tol) << what << " vy";
    EXPECT_NEAR(a.linear_velocity.z, b.linear_velocity.z, tol) << what << " vz";
    EXPECT_NEAR(a.orientation.w, b.orientation.w, tol) << what << " quat.w";
    EXPECT_NEAR(a.orientation.x, b.orientation.x, tol) << what << " quat.x";
    EXPECT_NEAR(a.orientation.y, b.orientation.y, tol) << what << " quat.y";
    EXPECT_NEAR(a.orientation.z, b.orientation.z, tol) << what << " quat.z";
}

constexpr float kGravityZ = -9.81f;
constexpr float kDt = 5.0e-3f;
constexpr uint32_t kSteps = 100u;  // 0.5 s

// ===========================================================================
// P2.2 -- batched box<->static-ground contact. Shared helpers + the STANDALONE
// single-box reference pipeline (the byte-exact oracle for Gate A1/C).
// ===========================================================================
namespace amf = nuka::collision::amf;
namespace articulation = nuka::runtime::articulation;
using nuka::collision::CandidatePair;
using nuka::collision::ResolvedShape;
using nuka::collision::ShapeResolver;
using nuka::constraint::CollidableRef;
using nuka::constraint::CollidableType;
using nuka::constraint::ContactManifold;
using nuka::constraint::ContactRowSides;
using nuka::constraint::ReactionProviderKind;
using nuka::constraint::RowBuffers;
using nuka::math::Transform;

// A cup-sized box (1 kg, half-extent 0.045 m -> a ~9 cm cube). inv_inertia of a solid
// cube of mass m, side L=2h: I = m L^2 / 6 -> 1/I = 6/(m L^2) = 6/(1*0.09^2) ~= 740.7.
constexpr float kCupHalf = 0.045f;
BodyState MakeCupBox(const Vec3& position, const Vec3& linear_velocity = Vec3{}) {
    BodyState b;
    b.inv_mass = 1.0f;
    const float side = 2.0f * kCupHalf;
    const float inv_i = 6.0f / (side * side);  // solid-cube principal inverse inertia.
    b.inv_inertia = Vec3{inv_i, inv_i, inv_i};
    b.position = position;
    b.orientation = Quat::Identity();
    b.linear_velocity = linear_velocity;
    return b;
}

// The box / ground PrimParams + collidable refs -- byte-for-byte the production
// helpers in batched_unified_world.cpp (and the oracle stepper). REPLICATED here so
// the standalone reference drives the SAME production calls (BuildContactManifolds ->
// EmitCompliantContactRows -> UnifiedSolve) the batched world drives.
constexpr uint32_t kRefBoxHandle = 9000u;
constexpr uint32_t kRefGroundHandle = 8000u;

amf::PrimParams RefBoxPrim(const Transform& pose, float half) {
    amf::PrimParams p;
    p.half_extents = Vec3{half, half, half};
    p.frame = amf::BuildPrimFrame(pose);
    return p;
}
amf::PrimParams RefGroundPrim(float height) {
    amf::PrimParams p;
    p.frame.cx = Vec3{1.0f, 0.0f, 0.0f};
    p.frame.cy = Vec3{0.0f, 0.0f, 1.0f};
    p.frame.cz = Vec3{0.0f, -1.0f, 0.0f};
    p.frame.t = Vec3{0.0f, 0.0f, height};
    return p;
}
CollidableRef RefBoxRef() {
    CollidableRef ref;
    ref.type = CollidableType::RigidBody;
    ref.react = ReactionProviderKind::RigidInvMass;
    ref.handle = kRefBoxHandle;
    return ref;
}
CollidableRef RefGroundRef() {
    CollidableRef ref;
    ref.type = CollidableType::StaticWorld;
    ref.react = ReactionProviderKind::StaticNull;
    ref.handle = kRefGroundHandle;
    return ref;
}

// The STANDALONE single-box reference: ONE box on a static +Z plane, run through the
// SAME production pipeline BatchedUnifiedWorld uses (velocity gravity-kick ->
// box x plane narrowphase -> EmitCompliantContactRows condim=1 -> UnifiedSolve
// cfg{64,0,0,0} with art_refs=nullptr -> IntegrateBodyPosition). Body index 0 (the
// N=1 case of BodyIndex(e,0)). This is the byte-exact oracle for Gate A1/C.
BodyState ReferenceBoxOnGround(BodyState box, float ground_height, float box_half,
                               float gravity_z, float dt, uint32_t steps) {
    for (uint32_t s = 0u; s < steps; ++s) {
        // ----- velocity stage: gravity kick -----
        if (box.inv_mass > 0.0f) box.linear_velocity.z += gravity_z * dt;

        // ----- contact phase: box<->ground through the unified spine -----
        CandidatePair pair;
        pair.a = RefBoxRef();
        pair.b = RefGroundRef();
        const Transform box_pose{box.position, box.orientation};
        const amf::PrimParams box_prim = RefBoxPrim(box_pose, box_half);
        const amf::PrimParams ground_prim = RefGroundPrim(ground_height);
        ShapeResolver resolve = [&](const CollidableRef& ref,
                                    ResolvedShape* out) -> bool {
            if (ref.type == CollidableType::RigidBody && ref.handle == kRefBoxHandle) {
                out->type = nuka::scene::ShapeType::Box;
                out->prim = box_prim;
                return true;
            }
            if (ref.type == CollidableType::StaticWorld &&
                ref.handle == kRefGroundHandle) {
                out->type = nuka::scene::ShapeType::Plane;
                out->prim = ground_prim;
                return true;
            }
            return false;
        };
        std::vector<ContactManifold> manifolds;
        const CandidatePair pairs[1] = {pair};
        nuka::collision::BuildContactManifolds(pairs, resolve, &manifolds);

        if (!manifolds.empty()) {
            nuka::constraint::ContactRowComplianceInputs inputs;
            inputs.vel = 0.0f;
            inputs.invweight = 1.0f;
            inputs.dt = dt;
            inputs.condim = 1u;
            inputs.refsafe = true;
            RowBuffers rows;
            std::vector<ContactRowSides> sides;
            nuka::constraint::EmitCompliantContactRows(manifolds, inputs, &rows,
                                                       &sides);
            for (std::size_t r = 0u; r < rows.RowCount(); ++r) {
                const ContactRowSides& sd = sides[r];
                const bool a_rigid = sd.a.react == ReactionProviderKind::RigidInvMass;
                const int box_local = a_rigid ? 0 : 1;
                const int static_local = a_rigid ? 1 : 0;
                rows.body_indices[2u * r + static_cast<uint32_t>(box_local)] = 0u;
                rows.body_indices[2u * r + static_cast<uint32_t>(static_local)] =
                    nuka::constraint::kInvalidBodyIndex;
            }
            if (rows.RowCount() > 0u && !sides.empty()) {
                nuka::solver::SolverConfig cfg;
                cfg.velocity_iterations = 64u;
                cfg.position_iterations = 0u;
                cfg.slop = 0.0f;
                cfg.baumgarte = 0.0f;
                std::vector<BodyState> bodies = {box};
                nuka::solver::SolveContext ctx;
                ctx.rows = &rows;
                ctx.state = &bodies;
                ctx.sides = &sides;
                ctx.dt = dt;
                // ctx.articulation default: art_refs=nullptr -> pure-rigid C5a path.
                nuka::solver::UnifiedSolve(ctx, cfg);
                box = bodies[0];
            }
        }

        // ----- position stage: symplectic-Euler integrate -----
        if (box.inv_mass > 0.0f) {
            box.position += box.linear_velocity * dt;
            const Vec3 w = box.angular_velocity;
            Quat dq;
            dq.w = 1.0f;
            dq.x = 0.5f * w.x * dt;
            dq.y = 0.5f * w.y * dt;
            dq.z = 0.5f * w.z * dt;
            box.orientation = (box.orientation * dq).Normalized();
        }
    }
    return box;
}

// Build a P2.2 scene template: ONE cup-box per env on a static +Z ground.
coresident::BatchedSceneTemplate MakeGroundedTemplate(const BodyState& cup,
                                                      float ground_height) {
    coresident::BatchedSceneTemplate tmpl;
    tmpl.bodies_per_env = {cup};
    tmpl.has_ground = true;
    tmpl.box_half_extent = Vec3{kCupHalf, kCupHalf, kCupHalf};
    tmpl.ground_height = ground_height;
    return tmpl;
}

// ---------------------------------------------------------------------------
// GATE 1: free-fall matches the discrete scheme AND the continuous projectile.
// ---------------------------------------------------------------------------
TEST(BatchedUnifiedWorld, FreeFallMatchesSymplecticEuler) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    const Vec3 p0{0.0f, 0.0f, 2.0f};
    const Vec3 v0{0.0f, 0.0f, 1.5f};       // upward toss
    const Vec3 w0{0.0f, 0.0f, 3.0f};       // 3 rad/s pure-Z spin

    coresident::BatchedSceneTemplate tmpl;
    tmpl.bodies_per_env = {MakeBody(p0, v0, w0)};

    constexpr uint32_t kEnvs = 8u;
    coresident::BatchedUnifiedWorld world(context, tmpl, kEnvs, kGravityZ, kDt);
    for (uint32_t s = 0u; s < kSteps; ++s) world.Step();

    const BodyState ref =
        ReferenceFreeFall(MakeBody(p0, v0, w0), kGravityZ, kDt, kSteps);

    // Every env equals the discrete reference (byte-exact: identical float ops).
    for (uint32_t e = 0u; e < kEnvs; ++e) {
        ExpectBodyEq(world.Body(e, 0u), ref, 1.0e-6f, "env-vs-discrete-ref");
    }

    // Physical sanity: z tracks the CONTINUOUS projectile within O(dt) -- proves it
    // is real free-fall, not a circular self-comparison.
    const double t = static_cast<double>(kDt) * kSteps;
    const double z_cont = 2.0 + 1.5 * t + 0.5 * kGravityZ * t * t;
    EXPECT_NEAR(static_cast<double>(world.Body(0u, 0u).position.z), z_cont, 2.0e-2)
        << "z must track the continuous projectile (real free-fall)";
    EXPECT_LT(world.Body(0u, 0u).position.z, p0.z)
        << "after 0.5 s the tossed body must have fallen below its start";

    // The spin stayed pure-Z (off-axis quat components ~ 0).
    const BodyState& b = world.Body(0u, 0u);
    EXPECT_LT(std::sqrt(b.orientation.x * b.orientation.x +
                        b.orientation.y * b.orientation.y),
              1.0e-4f)
        << "a pure-Z spin must not grow off-axis quaternion components";
}

// ---------------------------------------------------------------------------
// GATE 2: per-env independence -- different IC per env, each follows its own path.
// ---------------------------------------------------------------------------
TEST(BatchedUnifiedWorld, PerEnvIndependence) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    constexpr uint32_t kEnvs = 16u;
    coresident::BatchedSceneTemplate tmpl;
    tmpl.bodies_per_env = {MakeBody(Vec3{0.0f, 0.0f, 1.0f}, Vec3{}, Vec3{})};
    coresident::BatchedUnifiedWorld world(context, tmpl, kEnvs, kGravityZ, kDt);

    // Give each env a DISTINCT initial condition before stepping.
    std::vector<BodyState> ic(kEnvs);
    for (uint32_t e = 0u; e < kEnvs; ++e) {
        const float fe = static_cast<float>(e);
        ic[e] = MakeBody(Vec3{0.1f * fe, -0.2f * fe, 2.0f + 0.3f * fe},
                         Vec3{0.0f, 0.0f, 0.5f + 0.1f * fe},
                         Vec3{0.0f, 0.0f, 1.0f + 0.2f * fe});
        world.BodyMut(e, 0u) = ic[e];
    }

    for (uint32_t s = 0u; s < kSteps; ++s) world.Step();

    // Each env equals ITS OWN reference; adjacent envs must DIFFER (no contamination).
    for (uint32_t e = 0u; e < kEnvs; ++e) {
        const BodyState ref = ReferenceFreeFall(ic[e], kGravityZ, kDt, kSteps);
        ExpectBodyEq(world.Body(e, 0u), ref, 1.0e-6f, "per-env-own-ref");
    }
    for (uint32_t e = 1u; e < kEnvs; ++e) {
        EXPECT_GT(std::fabs(world.Body(e, 0u).position.z -
                            world.Body(e - 1u, 0u).position.z),
                  1.0e-4f)
            << "env " << e << " trajectory collapsed onto its neighbor";
    }
}

// ---------------------------------------------------------------------------
// GATE 3: determinism (D1) -- two worlds, same setup, byte-exact across all envs.
// ---------------------------------------------------------------------------
TEST(BatchedUnifiedWorld, DeterministicTwoRunByteExact) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    constexpr uint32_t kEnvs = 12u;
    coresident::BatchedSceneTemplate tmpl;
    tmpl.bodies_per_env = {MakeBody(Vec3{0.0f, 0.0f, 3.0f},
                                    Vec3{0.3f, -0.1f, 1.0f},
                                    Vec3{0.5f, 0.0f, 2.0f})};

    auto run = [&]() {
        coresident::BatchedUnifiedWorld w(context, tmpl, kEnvs, kGravityZ, kDt);
        for (uint32_t e = 0u; e < kEnvs; ++e) {
            w.BodyMut(e, 0u).position.x += 0.05f * static_cast<float>(e);
        }
        for (uint32_t s = 0u; s < kSteps; ++s) w.Step();
        return w.Bodies();
    };

    const std::vector<BodyState> a = run();
    const std::vector<BodyState> b = run();
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(BodyState)), 0)
        << "two identical runs were not byte-exact (D1 violated)";
}

// ---------------------------------------------------------------------------
// GATE 4: an immovable body (inv_mass=0) is frozen (matches IntegrateBoxPosition).
// ---------------------------------------------------------------------------
TEST(BatchedUnifiedWorld, ImmovableBodyDoesNotMove) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    BodyState statik = MakeBody(Vec3{1.0f, 2.0f, 3.0f},
                                Vec3{5.0f, 5.0f, 5.0f},   // nonzero vel, but...
                                Vec3{1.0f, 1.0f, 1.0f});
    statik.inv_mass = 0.0f;                                // ...immovable.

    coresident::BatchedSceneTemplate tmpl;
    tmpl.bodies_per_env = {statik};
    coresident::BatchedUnifiedWorld world(context, tmpl, 4u, kGravityZ, kDt);
    for (uint32_t s = 0u; s < kSteps; ++s) world.Step();

    for (uint32_t e = 0u; e < 4u; ++e) {
        const BodyState& b = world.Body(e, 0u);
        EXPECT_FLOAT_EQ(b.position.x, 1.0f);
        EXPECT_FLOAT_EQ(b.position.y, 2.0f);
        EXPECT_FLOAT_EQ(b.position.z, 3.0f) << "immovable body fell under gravity";
        EXPECT_FLOAT_EQ(b.orientation.w, 1.0f) << "immovable body rotated";
    }
}

// ===========================================================================
// P2.2 GATES -- batched box<->static-ground contact.
// ===========================================================================
// The byte-exact decision (decided UP FRONT, per the brief's BYTE-EXACT RULE):
//   * A1 / C / D  -> BYTE-EXACT. The batched env-e path and the standalone reference
//     BOTH take the pure-rigid art_refs=nullptr UnifiedSolve path; env e differs from
//     N=1 only by the body index (BodyIndex(e,0) vs 0). Cross-env rows share no body
//     id, so the greedy-in-row-index coloring (row_scheduler.cu RowsConflict) gives
//     every env the same color pattern -> env e's PGS sweep is identical whether N=1
//     or N=8. We assert byte-exact (0 tolerance) and have empirically confirmed it.
//   * A2 (production-oracle anchor) -> TIGHT TOLERANCE (not byte-exact). The real
//     co-resident Step() threads a NON-null (all-none) art_refs + minv/qdot into
//     UnifiedSolve (the C5b path), distinct from the reference's nullptr C5a path; a
//     bit-identical cross-path result is not guaranteed. A2 is the PHYSICS anchor at
//     tolerance, A1 is the byte-exact plumbing check.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GATE A1: an N=1 batched world matches the standalone single-box reference
// byte-exact over the full run (isolates the art_refs=nullptr path equivalence).
// ---------------------------------------------------------------------------
TEST(BatchedUnifiedWorldContact, Gate_A1_N1_MatchesStandaloneReference) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    const float kGroundZ = 0.0f;
    // Drop the cup from ~1.5 cm above the plane (bottom at ground+0.015): a modest
    // drop so one-step penetration stays well under the half-extent.
    const Vec3 p0{0.0f, 0.0f, kGroundZ + kCupHalf + 0.015f};
    const BodyState cup0 = MakeCupBox(p0);

    const auto tmpl = MakeGroundedTemplate(cup0, kGroundZ);
    coresident::BatchedUnifiedWorld world(context, tmpl, 1u, kGravityZ, kDt);
    const uint32_t kRun = 300u;  // enough to settle.
    for (uint32_t s = 0u; s < kRun; ++s) world.Step();

    const BodyState ref =
        ReferenceBoxOnGround(cup0, kGroundZ, kCupHalf, kGravityZ, kDt, kRun);

    // BYTE-EXACT: 0 tolerance. (We use a memcmp-grade ExpectBodyEq at tol 0.)
    ExpectBodyEq(world.Body(0u, 0u), ref, 0.0f, "N1-vs-standalone-ref");
    // And literally byte-exact on the whole BodyState struct.
    EXPECT_EQ(std::memcmp(&world.Body(0u, 0u), &ref, sizeof(BodyState)), 0)
        << "N=1 batched env was not byte-exact vs the standalone reference";
}

// ---------------------------------------------------------------------------
// GATE B: physical reality (non-circular). A box dropped from height settles ON
// the plane: bottom face ~ ground_height, velocity ~ 0, penetration bounded, no
// tunneling. Proves REAL contact, not a self-comparison.
// ---------------------------------------------------------------------------
TEST(BatchedUnifiedWorldContact, Gate_B_BoxSettlesOnPlane) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    const float kGroundZ = 0.2f;  // non-zero plane z (no accidental z=0 coincidence).
    // Drop from ~3 cm above so it falls, contacts, and settles.
    const Vec3 p0{0.3f, -0.4f, kGroundZ + kCupHalf + 0.03f};
    const BodyState cup0 = MakeCupBox(p0);

    const auto tmpl = MakeGroundedTemplate(cup0, kGroundZ);
    coresident::BatchedUnifiedWorld world(context, tmpl, 1u, kGravityZ, kDt);

    float max_penetration = 0.0f;  // peak (ground - bottom) over the run.
    float min_bottom = 1.0e9f;
    const uint32_t kRun = 400u;  // 2 s -- plenty to settle.
    for (uint32_t s = 0u; s < kRun; ++s) {
        world.Step();
        const BodyState& b = world.Body(0u, 0u);
        ASSERT_TRUE(std::isfinite(b.position.z)) << "box position non-finite @step " << s;
        const float bottom = b.position.z - kCupHalf;
        min_bottom = std::min(min_bottom, bottom);
        max_penetration = std::max(max_penetration, kGroundZ - bottom);
    }

    const BodyState& b = world.Body(0u, 0u);
    const float bottom = b.position.z - kCupHalf;
    // (1) Settled ON the plane: bottom ~ ground (within a few mm of steady compliant
    //     penetration; baumgarte=0/slop=0 leaves a small standing overlap).
    EXPECT_NEAR(bottom, kGroundZ, 5.0e-3f)
        << "box did not settle on the plane (bottom=" << bottom
        << " ground=" << kGroundZ << ")";
    // (2) Velocity ~ 0 at rest.
    EXPECT_LT(b.linear_velocity.Length(), 5.0e-2f)
        << "box not at rest (|v|=" << b.linear_velocity.Length() << ")";
    // (3) Penetration bounded + small (no divergence).
    EXPECT_GT(max_penetration, 0.0f) << "box never penetrated -- contact vacuous";
    EXPECT_LT(max_penetration, kCupHalf)
        << "penetration exceeded the half-extent -- box tunneling/diverging";
    // (4) No tunneling: the box never sank a full half-extent below the plane.
    EXPECT_GT(min_bottom, kGroundZ - kCupHalf)
        << "box tunneled through the plane (min bottom=" << min_bottom << ")";
}

// ---------------------------------------------------------------------------
// GATE C: per-env independence + THE BATCHING PROPERTY. N>=8 envs, DISTINCT ICs;
// each env matches ITS OWN standalone reference; adjacent envs DIFFER. This is the
// ONLY gate that actually tests batching (N=1 proves nothing about it).
// ---------------------------------------------------------------------------
TEST(BatchedUnifiedWorldContact, Gate_C_PerEnvIndependenceBatched) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    const float kGroundZ = 0.0f;
    constexpr uint32_t kEnvs = 12u;  // >= 8.

    // A nominal template (body 0 / ground) -- per-env ICs set below via BodyMut.
    const BodyState cup_seed = MakeCupBox(Vec3{0.0f, 0.0f, kCupHalf + 0.02f});
    const auto tmpl = MakeGroundedTemplate(cup_seed, kGroundZ);
    coresident::BatchedUnifiedWorld world(context, tmpl, kEnvs, kGravityZ, kDt);

    // DISTINCT IC per env: different x,y AND drop height (some drop more than others
    // so they settle at different times -> the trajectories genuinely diverge mid-run).
    std::vector<BodyState> ic(kEnvs);
    for (uint32_t e = 0u; e < kEnvs; ++e) {
        const float fe = static_cast<float>(e);
        ic[e] = MakeCupBox(Vec3{0.1f * fe, -0.15f * fe,
                                kGroundZ + kCupHalf + 0.01f + 0.01f * fe});
        world.BodyMut(e, 0u) = ic[e];
    }

    const uint32_t kRun = 300u;
    for (uint32_t s = 0u; s < kRun; ++s) world.Step();

    // (1) Each env equals ITS OWN standalone reference -- BYTE-EXACT (the batching
    //     property: disjoint per-env graphs color identically -> N independent solves).
    for (uint32_t e = 0u; e < kEnvs; ++e) {
        const BodyState ref =
            ReferenceBoxOnGround(ic[e], kGroundZ, kCupHalf, kGravityZ, kDt, kRun);
        EXPECT_EQ(std::memcmp(&world.Body(e, 0u), &ref, sizeof(BodyState)), 0)
            << "env " << e << " did not match its own reference byte-exact "
               "(batching cross-contamination)";
    }
    // (2) Adjacent envs DIFFER in x,y (distinct ICs -> no collapse / contamination).
    //     (z settles to ~the same resting height for all, so check the horizontal
    //     position which encodes the per-env IC.)
    for (uint32_t e = 1u; e < kEnvs; ++e) {
        const Vec3 d = world.Body(e, 0u).position - world.Body(e - 1u, 0u).position;
        EXPECT_GT(std::sqrt(d.x * d.x + d.y * d.y), 1.0e-4f)
            << "env " << e << " collapsed onto its neighbor (no per-env independence)";
    }
}

// ---------------------------------------------------------------------------
// GATE A2: anchor the standalone reference to VALIDATED production code. Construct
// the REAL co-resident box-mode UnifiedCoResidentStepper (go2 articulation + a
// movable box on a static +Z ground), drive it with the SAME box IC / ground height
// / half-extent / gravity / dt as the standalone reference, and assert the box
// trajectory matches. The foot is separated HORIZONTALLY from the box by several
// meters so the (free-falling) articulation never closes the vertical gap -> NO
// foot<->box manifold forms -> only the box<->ground rows drive the box (the exact
// path the standalone reference + the batched world take). This ties the reference
// to independently-validated code.
//
// Byte-exact is NOT expected here (decided up front): the co-resident Step() threads
// a NON-null all-none art_refs + minv/qdot into UnifiedSolve (the C5b code path),
// while the reference uses the nullptr C5a path; a bit-identical cross-path result is
// not guaranteed. Asserted at a TIGHT tolerance (~1e-5). Asset-gated (SKIP if the
// go2_float scene is absent).
// ---------------------------------------------------------------------------

// Source-tree path (NUKA_SOURCE_DIR is set on this target by CMake).
std::filesystem::path SourcePath(const char* rel) {
    return std::filesystem::path(NUKA_SOURCE_DIR) / rel;
}

// Forward-kinematics world poses of every link (mirrors the co-resident test helper).
std::vector<Transform> A2ForwardKinematics(
    const nuka::phi::DeviceContext& context,
    const articulation::ArticulationHostState& host) {
    const uint32_t link_count = host.TotalLinkCount();
    auto device = articulation::UploadArticulationState(context, host);
    nuka::phi::Buffer pose_buf(static_cast<size_t>(link_count) * sizeof(Transform),
                               nuka::phi::MemoryKind::Device);
    articulation::UpdateWorldLinkPoses(context, device.View(),
                                       static_cast<Transform*>(pose_buf.Data()));
    context.stream.Synchronize();
    std::vector<Transform> poses(link_count);
    pose_buf.CopyToHost(poses.data(), poses.size() * sizeof(Transform));
    return poses;
}

TEST(BatchedUnifiedWorldContact, Gate_A2_StandaloneRefMatchesCoResidentOracle) {
    const auto scene_path = SourcePath("examples/scenes/go2_float.usda");
    if (!std::filesystem::exists(scene_path)) {
        GTEST_SKIP() << "go2_float.usda not available -- A2 production anchor skipped "
                        "(A1 + B already give non-circular coverage)";
    }
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    // --- cook the go2_float articulation + extract its foot 0 (mirror the oracle) ---
    const auto scene = nuka::import::LoadUsd(scene_path.string());
    const auto blob = nuka::scene::CookScene(scene);
    const auto world = nuka::runtime::BuildWorld(blob);
    auto host = articulation::BuildArticulationHostState(
        world.template_view.articulations, world.template_view.body_table);
    const auto& shapes = world.template_view.shape_table;
    const uint32_t link_count = host.TotalLinkCount();

    coresident::CoResidentFoot foot;
    constexpr uint32_t kInvalidLink = ~0u;
    for (uint32_t s = 0u; s < shapes.types.size(); ++s) {
        if (shapes.types[s] != nuka::scene::ShapeType::Sphere) continue;
        const uint32_t body = shapes.body_ids[s];
        uint32_t calf = kInvalidLink;
        for (uint32_t l = 0u; l < link_count; ++l) {
            if (host.link_body[l] == body) { calf = l; break; }
        }
        if (calf == kInvalidLink) continue;
        foot.calf_link = calf;
        foot.local_offset = s < shapes.local_transforms.size()
                                ? shapes.local_transforms[s].position : Vec3::Zero();
        foot.radius = s < shapes.radii.size() ? shapes.radii[s] : 0.0f;
        break;
    }
    ASSERT_NE(foot.calf_link, kInvalidLink) << "no foot sphere found in go2_float";

    // Rest the articulation (identity base, zero velocities) so it free-falls cleanly.
    host.base_pose[0].rotation = Quat::Identity();
    for (auto& v : host.link_velocity) for (float& c : v.v) c = 0.0f;
    for (float& qd : host.qdot) qd = 0.0f;

    // FK foot center -- the box is placed FAR from it in X/Y so no foot<->box contact.
    const auto poses = A2ForwardKinematics(context, host);
    const Transform& calf = poses[foot.calf_link];
    const Vec3 foot_center = calf.position + calf.rotation.Rotate(foot.local_offset);

    // --- box IC / ground identical to the standalone reference path ---
    const float kGroundZ = 0.0f;
    // Box dropped ~1.5 cm above the plane, OFFSET +5 m in X and +5 m in Y from the
    // foot so the vertical free-fall never closes the (horizontal) gap -> no
    // foot<->box manifold ever forms (only box<->ground rows drive the box).
    const Vec3 box_pos{foot_center.x + 5.0f, foot_center.y + 5.0f,
                       kGroundZ + kCupHalf + 0.015f};
    const BodyState box0 = MakeCupBox(box_pos);

    const float kA2Dt = 1.0f / 240.0f;
    const uint32_t kA2Steps = 200u;

    coresident::CoResidentBox box_desc;
    box_desc.half_extent = kCupHalf;
    coresident::CoResidentGround ground_desc;
    ground_desc.height = kGroundZ;
    coresident::UnifiedCoResidentStepper stepper(context, host, shapes, foot, box_desc,
                                                 ground_desc, box0, kGravityZ, kA2Dt);

    bool any_foot_row = false;
    for (uint32_t s = 0u; s < kA2Steps; ++s) {
        const auto rep = stepper.Step();
        // The foot must NEVER form a manifold with the box (horizontal separation).
        if (rep.pair_found && rep.row_count > rep.ground_row_count) any_foot_row = true;
        ASSERT_TRUE(std::isfinite(stepper.Box().position.z))
            << "oracle box non-finite @step " << s;
    }
    EXPECT_FALSE(any_foot_row)
        << "a foot<->box manifold formed -- horizontal separation failed to isolate "
           "the box trajectory from the articulation";

    // The standalone reference run with the SAME box IC / ground / half / gravity / dt.
    const BodyState ref =
        ReferenceBoxOnGround(box0, kGroundZ, kCupHalf, kGravityZ, kA2Dt, kA2Steps);

    // The oracle box and the reference box must agree (tight tolerance -- cross-path).
    const BodyState& oracle = stepper.Box();
    const bool a2_byte_exact =
        std::memcmp(&oracle, &ref, sizeof(BodyState)) == 0;
    const Vec3 dp = oracle.position - ref.position;
    std::printf("[A2] cross-path (C5b oracle vs C5a ref): byte_exact=%d  "
                "|dpos|=%.3e  dvz=%.3e\n",
                a2_byte_exact ? 1 : 0, dp.Length(),
                oracle.linear_velocity.z - ref.linear_velocity.z);
    ExpectBodyEq(oracle, ref, 1.0e-5f, "standalone-ref-vs-coresident-oracle");
    // And both must have actually settled on the plane (a real, non-vacuous anchor).
    EXPECT_NEAR(oracle.position.z - kCupHalf, kGroundZ, 5.0e-3f)
        << "oracle box did not settle on the plane (anchor vacuous)";
}

// ---------------------------------------------------------------------------
// GATE D: D1 determinism. Two identical batched runs, byte-exact Bodies().
// ---------------------------------------------------------------------------
TEST(BatchedUnifiedWorldContact, Gate_D_DeterministicTwoRun) {
    const auto context = nuka::phi::MakeDefaultDeviceContext();

    const float kGroundZ = 0.05f;
    constexpr uint32_t kEnvs = 10u;
    const BodyState cup_seed = MakeCupBox(Vec3{0.0f, 0.0f, kGroundZ + kCupHalf + 0.02f});
    const auto tmpl = MakeGroundedTemplate(cup_seed, kGroundZ);

    auto run = [&]() {
        coresident::BatchedUnifiedWorld w(context, tmpl, kEnvs, kGravityZ, kDt);
        for (uint32_t e = 0u; e < kEnvs; ++e) {
            w.BodyMut(e, 0u).position.x += 0.03f * static_cast<float>(e);
            w.BodyMut(e, 0u).position.z += 0.01f * static_cast<float>(e);
        }
        for (uint32_t s = 0u; s < 250u; ++s) w.Step();
        return w.Bodies();
    };

    const std::vector<BodyState> a = run();
    const std::vector<BodyState> b = run();
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(BodyState)), 0)
        << "two identical batched contact runs were not byte-exact (D1 violated)";
}

// ===========================================================================
// P2.3a -- batched articulation<->rigid GRASP. A FIXED-base 2-finger gripper grips a
// movable convex-hull CUP by FINGER FRICTION ALONE (NO table); the cup is HELD against
// gravity; grip-off -> the cup FALLS (the BITE anti-green-wash discriminator). Gates:
//   * A1 (N=1 batched == in-test standalone replicated grasp ref) -- BYTE-EXACT.
//   * A2 (N=1 batched == REAL co-resident StepGrasp oracle) -- TIGHT TOL 1e-5.
//   * HOLD (Σ finger vertical impulse ~= m*g*dt; cup drift << free fall; NO-TABLE).
//   * BITE (grip=0 -> the cup falls, final vz < -1, drop >> held).
// The scene construction (BuildGraspGripper / LoadGraspCupHull / BuildGraspSceneBundle)
// is the SAME pattern as tests/coresident/test_grasp_hold_spike.cpp (test code, not
// protected); the BatchedSceneTemplate is populated from the SAME BuildGraspSceneBundle
// output the A2 oracle uses, so A2 is reachable at 1e-5 and A1 is byte-exact.
//
// THE BYTE-EXACT DECISION (decided UP FRONT, per the brief's BYTE-EXACT RULE):
//   * A1 -> BYTE-EXACT. The standalone reference is a near-verbatim transcription of
//     ResolveBatchedGraspContact (SAME replicated helpers, SAME row-iteration order,
//     SAME chain_jacobians accumulation, SAME device-op sequence -- it persists its OWN
//     gripper ArticulationDeviceBuffers across the step loop and scatters in place, NOT
//     a per-step re-upload, so the FP round-trip is identical). The only difference is
//     the body-index source (literal 0/1 vs BodyIndex(0,*)/total_body_count, which
//     COINCIDE at N=1). We assert memcmp==0 on the cup BodyState + q + qdot.
//   * A2 -> TIGHT TOLERANCE 1e-5 (not byte-exact). The real co-resident StepGrasp lives
//     in a DIFFERENT TU (cross-TU), threads its OWN replicated glue, and accumulates the
//     condim=3 friction-cone host quantities in its own order. "Correct-by-construction"
//     guarantees solve EQUIVALENCE, not FP bit-identity -- exactly the P2.2 A2 precedent.
// ===========================================================================

namespace cvx = nuka::collision::cvx;
using nuka::constraint::RowArticulationRefs;
using nuka::constraint::RowArticulationSide;
using nuka::constraint::RowJacobian6;
using nuka::runtime::coresident::CoResidentCup;
using nuka::runtime::coresident::CoResidentFingertip;
using nuka::runtime::coresident::GraspConfig;

constexpr float kGraspGravityZ = -9.81f;
constexpr float kGraspDt = 1.0f / 240.0f;
constexpr float kCupMass = 0.2f;
constexpr float kCupInvMass = 1.0f / kCupMass;

const std::string kCupModelUsda =
    ".nuka-assets/newton_assets/manipulation_objects/cup/model.usda";
bool GraspCupAvailable() { return std::filesystem::exists(kCupModelUsda); }

// Load the C7a cup -> a single cooked ConvexHull -> mesh-local hull verts + bbox.
// (Replicated from test_grasp_hold_spike.cpp::LoadCupHull -- test code.)
struct GraspCupHull {
    std::vector<float> verts;
    Vec3 lo{}, hi{};
    uint32_t vcount = 0u;
};
GraspCupHull LoadGraspCupHull() {
    auto scene = nuka::import::LoadUsd(kCupModelUsda);
    for (size_t i = 0; i < scene.ShapeCount(); ++i) {
        auto& s = scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        if (!s.mesh_vertices.empty())
            s.decompose_mode = nuka::scene::DecomposeMode::Skip;
    }
    const auto blob = nuka::scene::CookScene(scene);
    GraspCupHull out;
    const auto& cg = blob.convex_geometry;
    if (cg.vertex_counts.empty()) return out;
    const uint32_t voff = cg.vertex_offsets[0];
    const uint32_t vcnt = cg.vertex_counts[0];
    out.vcount = vcnt;
    out.verts.assign(cg.vertices.begin() + voff * 3u,
                     cg.vertices.begin() + (voff + vcnt) * 3u);
    out.lo = Vec3{out.verts[0], out.verts[1], out.verts[2]};
    out.hi = out.lo;
    for (uint32_t i = 0u; i < vcnt; ++i) {
        const Vec3 v{out.verts[i * 3u], out.verts[i * 3u + 1u], out.verts[i * 3u + 2u]};
        out.lo = Vec3{std::min(out.lo.x, v.x), std::min(out.lo.y, v.y),
                      std::min(out.lo.z, v.z)};
        out.hi = Vec3{std::max(out.hi.x, v.x), std::max(out.hi.y, v.y),
                      std::max(out.hi.z, v.z)};
    }
    return out;
}

// The synthetic FIXED-base 2-finger gripper (replicated from the spike's BuildGripper).
struct GraspGripper {
    articulation::ArticulationHostState host;
    uint32_t finger_link[2];
    Vec3     fingertip_local[2];
    float    fingertip_radius = 0.008f;
};
GraspGripper BuildGraspGripper(const Vec3& base_pos, float cup_half_x,
                               float fingertip_radius) {
    constexpr uint32_t kInvalidLink = ~0u;
    articulation::ArticulationCookedTopology topo;
    topo.root_body = 0u;
    topo.link_bodies = {0u, 1u, 2u};
    topo.parent_links = {kInvalidLink, 0u, 0u};
    topo.joint_types = {articulation::ArticulationJointType::Fixed,
                        articulation::ArticulationJointType::Prismatic,
                        articulation::ArticulationJointType::Prismatic};
    topo.joint_axes = {Vec3::UnitX(), Vec3{-1.0f, 0.0f, 0.0f}, Vec3{1.0f, 0.0f, 0.0f}};
    topo.parent_frames = {Transform::Identity(), Transform::Identity(),
                          Transform::Identity()};
    topo.child_frames = {Transform::Identity(), Transform::Identity(),
                         Transform::Identity()};
    Transform base_lp = Transform::Identity();
    base_lp.position = base_pos;
    topo.local_poses = {base_lp, Transform::Identity(), Transform::Identity()};
    topo.inertial_frames = {Transform::Identity(), Transform::Identity(),
                            Transform::Identity()};
    topo.initial_positions = {0.0f, 0.0f, 0.0f};
    topo.joint_dampings = {0.0f, 0.0f, 0.0f};
    topo.joint_armatures = {0.0f, 0.0f, 0.0f};
    topo.masses = {1.0f, 0.05f, 0.05f};
    topo.inertias = {Vec3{1e-2f, 1e-2f, 1e-2f}, Vec3{1e-4f, 1e-4f, 1e-4f},
                     Vec3{1e-4f, 1e-4f, 1e-4f}};

    nuka::scene::CookedBodyTable bodies;
    Transform base_pose = Transform::Identity();
    base_pose.position = base_pos;
    bodies.poses = {base_pose, Transform::Identity(), Transform::Identity()};
    bodies.local_poses = {base_lp, Transform::Identity(), Transform::Identity()};
    bodies.inertial_frames = {Transform::Identity(), Transform::Identity(),
                              Transform::Identity()};
    bodies.masses = {1.0f, 0.05f, 0.05f};
    bodies.inertias = {Vec3{1e-2f, 1e-2f, 1e-2f}, Vec3{1e-4f, 1e-4f, 1e-4f},
                       Vec3{1e-4f, 1e-4f, 1e-4f}};
    bodies.is_static = {0u, 0u, 0u};

    GraspGripper g;
    g.host = articulation::BuildArticulationHostState({topo}, bodies);
    g.finger_link[0] = 1u;
    g.finger_link[1] = 2u;
    g.fingertip_radius = fingertip_radius;
    const float kPenetration = 0.0015f;  // 1.5 mm shallow pre-pose -> contact live @ step 0.
    const float reach = cup_half_x + fingertip_radius - kPenetration;
    g.fingertip_local[0] = Vec3{reach, 0.0f, 0.0f};
    g.fingertip_local[1] = Vec3{-reach, 0.0f, 0.0f};
    return g;
}

// Assemble the grasp scene: gripper + cup hull + grip force + friction (replicated from
// the spike's BuildGraspScene). Returns the gripper + the GraspConfig the oracle uses.
struct GraspSceneBundle {
    GraspGripper gripper;
    GraspConfig  config;
};
GraspSceneBundle BuildGraspSceneBundle(const GraspCupHull& hull, float grip_force,
                                       float mu) {
    const Vec3 half = (hull.hi - hull.lo) * 0.5f;
    const Vec3 cup_local_center = (hull.hi + hull.lo) * 0.5f;
    const Vec3 cup_center{0.0f, 0.0f, 0.20f};
    const Vec3 base_pos{0.0f, 0.0f, 0.30f};

    GraspSceneBundle gs;
    gs.gripper = BuildGraspGripper(base_pos, half.x, 0.008f);
    const float drop = cup_center.z - base_pos.z;
    gs.gripper.fingertip_local[0].z += drop;
    gs.gripper.fingertip_local[1].z += drop;

    CoResidentFingertip ft0;
    ft0.link = gs.gripper.finger_link[0];
    ft0.broadphase_handle = gs.gripper.finger_link[0];
    ft0.local_offset = gs.gripper.fingertip_local[0];
    ft0.radius = gs.gripper.fingertip_radius;
    CoResidentFingertip ft1;
    ft1.link = gs.gripper.finger_link[1];
    ft1.broadphase_handle = gs.gripper.finger_link[1];
    ft1.local_offset = gs.gripper.fingertip_local[1];
    ft1.radius = gs.gripper.fingertip_radius;
    gs.config.fingertips = {ft0, ft1};

    gs.config.cup.hull_verts = hull.verts;
    for (uint32_t i = 0u; i < hull.vcount; ++i) {
        gs.config.cup.hull_verts[i * 3u + 0u] -= cup_local_center.x;
        gs.config.cup.hull_verts[i * 3u + 1u] -= cup_local_center.y;
        gs.config.cup.hull_verts[i * 3u + 2u] -= cup_local_center.z;
    }
    gs.config.cup.broadphase_body_id = 7000u;

    BodyState cup;
    cup.inv_mass = kCupInvMass;
    const float ix = kCupMass * (half.y * half.y + half.z * half.z) / 3.0f;
    const float iy = kCupMass * (half.x * half.x + half.z * half.z) / 3.0f;
    const float iz = kCupMass * (half.x * half.x + half.y * half.y) / 3.0f;
    cup.inv_inertia = Vec3{1.0f / ix, 1.0f / iy, 1.0f / iz};
    cup.position = cup_center;
    cup.orientation = Quat::Identity();
    cup.linear_velocity = Vec3::Zero();
    cup.angular_velocity = Vec3::Zero();
    gs.config.cup_state = cup;

    const uint32_t link_count = gs.gripper.host.TotalLinkCount();
    gs.config.grip_torque.assign(link_count, 0.0f);
    gs.config.grip_torque[gs.gripper.finger_link[0]] = grip_force;
    gs.config.grip_torque[gs.gripper.finger_link[1]] = grip_force;
    gs.config.drive_force_limits.assign(link_count, 0.0f);
    gs.config.friction_mu = mu;
    gs.config.condim = 3u;
    return gs;
}

// Populate a BatchedSceneTemplate from a GraspConfig + gripper (the cup is the single
// per-env body; cup_local_index=0). Identical inputs to the A2 oracle -> A1 byte-exact.
coresident::BatchedSceneTemplate MakeGraspTemplate(const GraspSceneBundle& gs) {
    coresident::BatchedSceneTemplate tmpl;
    tmpl.bodies_per_env = {gs.config.cup_state};
    tmpl.has_grasp = true;
    tmpl.gripper_proto = gs.gripper.host;
    tmpl.fingertips = gs.config.fingertips;
    tmpl.cup = gs.config.cup;
    tmpl.cup_local_index = 0u;
    tmpl.grip_torque = gs.config.grip_torque;
    tmpl.drive_force_limits = gs.config.drive_force_limits;
    tmpl.friction_mu = gs.config.friction_mu;
    tmpl.condim = gs.config.condim;
    return tmpl;
}

// ---------------------------------------------------------------------------
// The STANDALONE single-env grasp reference (the byte-exact oracle for Gate A1).
// A near-verbatim transcription of BatchedUnifiedWorld::ResolveBatchedGraspContact +
// the Step() grasp integrate, persisting its OWN gripper ArticulationDeviceBuffers
// across the step loop (NOT a per-step re-upload), so the device-op sequence is
// structurally identical to the world -> A1 is a clean byte-exact plumbing check.
// ---------------------------------------------------------------------------

// Replicated grasp helpers (byte-for-byte from the production TU; here so the standalone
// reference drives the SAME device-op sequence the batched world drives).
std::vector<Transform> RefDownloadWorldPoses(
    const nuka::phi::DeviceContext& context,
    articulation::ArticulationDeviceBuffers& device, uint32_t link_count) {
    auto view = device.View();
    nuka::phi::Buffer pose_buf(static_cast<size_t>(link_count) * sizeof(Transform),
                               nuka::phi::MemoryKind::Device);
    articulation::UpdateWorldLinkPoses(context, view,
                                       static_cast<Transform*>(pose_buf.Data()));
    context.stream.Synchronize();
    std::vector<Transform> poses(link_count);
    pose_buf.CopyToHost(poses.data(), poses.size() * sizeof(Transform));
    return poses;
}
std::vector<float> RefFootChainJ(const nuka::phi::DeviceContext& context,
                                 articulation::ArticulationHostState host,
                                 const std::vector<Transform>& fk_world_poses,
                                 uint32_t contact_link, const Vec3& contact_point,
                                 const Vec3& contact_normal, uint32_t dof_stride) {
    if (fk_world_poses.size() == host.link_pose.size()) host.link_pose = fk_world_poses;
    auto device = articulation::UploadArticulationState(context, host);
    nuka::phi::Buffer link_buf(sizeof(uint32_t), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer point_buf(sizeof(Vec3), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer normal_buf(sizeof(Vec3), nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer jbuf(static_cast<size_t>(dof_stride) * sizeof(float),
                           nuka::phi::MemoryKind::Device);
    link_buf.CopyFromHost(&contact_link, sizeof(uint32_t));
    point_buf.CopyFromHost(&contact_point, sizeof(Vec3));
    normal_buf.CopyFromHost(&contact_normal, sizeof(Vec3));
    std::vector<float> zero(dof_stride, 0.0f);
    jbuf.CopyFromHost(zero.data(), zero.size() * sizeof(float));
    articulation::ComputeContactChainJacobians(
        context, device.View(), static_cast<const uint32_t*>(link_buf.Data()),
        static_cast<const Vec3*>(point_buf.Data()),
        static_cast<const Vec3*>(normal_buf.Data()), 1u, dof_stride,
        static_cast<float*>(jbuf.Data()));
    context.stream.Synchronize();
    std::vector<float> j(dof_stride);
    jbuf.CopyToHost(j.data(), j.size() * sizeof(float));
    return j;
}
std::vector<float> RefInverseInertia(const nuka::phi::DeviceContext& context,
                                     const articulation::ArticulationHostState& host,
                                     float gravity_z, uint32_t dof_stride) {
    auto device = articulation::UploadArticulationState(context, host);
    auto view = device.View();
    const uint32_t link_count = host.TotalLinkCount();
    articulation::FeatherstoneAba::ComputeAccelerations(context, view, gravity_z);
    nuka::phi::Buffer composite(
        static_cast<size_t>(link_count) * sizeof(articulation::LinkSpatialInertia),
        nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer m(static_cast<size_t>(dof_stride) * dof_stride * sizeof(float),
                        nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer m_inv(static_cast<size_t>(dof_stride) * dof_stride * sizeof(float),
                            nuka::phi::MemoryKind::Device);
    articulation::ComputeArticulationInertiaM(
        context, view, dof_stride,
        static_cast<articulation::LinkSpatialInertia*>(composite.Data()),
        static_cast<float*>(m.Data()));
    articulation::FactorArticulationInertiaM(
        context, view, dof_stride, static_cast<const float*>(m.Data()),
        static_cast<float*>(m_inv.Data()));
    context.stream.Synchronize();
    std::vector<float> out(static_cast<size_t>(dof_stride) * dof_stride);
    m_inv.CopyToHost(out.data(), out.size() * sizeof(float));
    return out;
}
amf::PrimParams RefFootSpherePrim(const Vec3& center, float radius) {
    amf::PrimParams p;
    p.radius = radius;
    p.frame.t = center;
    return p;
}
uint32_t RefDofIndexOf(const articulation::ArticulationHostState& host,
                       uint32_t root_link, uint32_t link) {
    uint32_t idx = 0u;
    for (uint32_t k = root_link; k < link; ++k)
        idx += articulation::ArticulationJointDofCount(host.joint_type[k]);
    return idx;
}

// The persistent standalone grasp reference. Mirrors the world: upload the gripper ONCE,
// step it in place. Step() advances one step; cup + the gripper host are the final state.
struct StandaloneGraspRef {
    const nuka::phi::DeviceContext& context;
    GraspGripper gripper;
    GraspConfig config;
    float gravity_z, dt;
    articulation::ArticulationDeviceBuffers device;
    nuka::phi::Buffer grip_torque_dev, grip_limits_dev;
    BodyState cup;
    uint32_t dof_stride, root_link, base_dof, link_count;

    StandaloneGraspRef(const nuka::phi::DeviceContext& ctx, const GraspSceneBundle& gs,
                       float g, float d)
        : context(ctx), gripper(gs.gripper), config(gs.config), gravity_z(g), dt(d) {
        device = articulation::UploadArticulationState(context, gripper.host);
        dof_stride = articulation::ArticulationDofCount(gripper.host, 0u);
        root_link = gripper.host.articulation_link_offset[0];
        base_dof =
            articulation::ArticulationJointDofCount(gripper.host.joint_type[root_link]);
        link_count = gripper.host.TotalLinkCount();
        cup = config.cup_state;
        std::vector<float> torque = config.grip_torque;
        torque.resize(link_count, 0.0f);
        std::vector<float> limits = config.drive_force_limits;
        limits.resize(link_count, 0.0f);
        grip_torque_dev = nuka::phi::UploadVector(torque);
        grip_limits_dev = nuka::phi::UploadVector(limits);
    }

    void IntegrateCup() {
        if (cup.inv_mass <= 0.0f) return;
        cup.position += cup.linear_velocity * dt;
        const Vec3 w = cup.angular_velocity;
        Quat dq;
        dq.w = 1.0f;
        dq.x = 0.5f * w.x * dt;
        dq.y = 0.5f * w.y * dt;
        dq.z = 0.5f * w.z * dt;
        cup.orientation = (cup.orientation * dq).Normalized();
    }

    void Step() {
        auto view = device.View();
        articulation::LaunchApplyTorqueDriveKernels(
            context, view, static_cast<const float*>(grip_torque_dev.Data()),
            static_cast<const float*>(grip_limits_dev.Data()));
        articulation::FeatherstoneAba::ComputeAccelerations(context, view, gravity_z);
        articulation::FeatherstoneAba::IntegrateVelocity(context, view, dt);
        articulation::FeatherstoneAba::IntegrateFloatingBaseVelocity(context, view, dt,
                                                                     gravity_z);
        if (cup.inv_mass > 0.0f) cup.linear_velocity.z += gravity_z * dt;

        articulation::ArticulationHostState live = gripper.host;
        articulation::DownloadArticulationState(device, &live);
        const std::vector<Transform> poses =
            RefDownloadWorldPoses(context, device, link_count);

        const Transform cup_pose{cup.position, cup.orientation};
        cvx::ConvexHullView cup_hull;
        cup_hull.verts = config.cup.hull_verts.data();
        cup_hull.vcount = config.cup.VertexCount();
        cup_hull.frame = amf::BuildPrimFrame(cup_pose);

        const bool has_geom = cup_hull.vcount != 0u;
        std::vector<Vec3> finger_centers(config.fingertips.size());
        std::vector<CandidatePair> drive_pairs;
        if (has_geom) {
            drive_pairs.reserve(config.fingertips.size());
            for (size_t f = 0u; f < config.fingertips.size(); ++f) {
                const CoResidentFingertip& ft = config.fingertips[f];
                const Transform& lp = poses[ft.link];
                finger_centers[f] = lp.position + lp.rotation.Rotate(ft.local_offset);
                CandidatePair p;
                p.a.type = CollidableType::ArticulationLink;
                p.a.react = ReactionProviderKind::ArticulationChainJ;
                p.a.handle = ft.broadphase_handle;
                p.b.type = CollidableType::RigidBody;
                p.b.react = ReactionProviderKind::RigidInvMass;
                p.b.handle = config.cup.broadphase_body_id;
                drive_pairs.push_back(p);
            }
        }

        std::vector<ContactManifold> manifolds;
        if (has_geom) {
            ShapeResolver resolve = [&](const CollidableRef& ref,
                                        ResolvedShape* out) -> bool {
                if (ref.type == CollidableType::ArticulationLink) {
                    for (size_t f = 0u; f < config.fingertips.size(); ++f) {
                        if (config.fingertips[f].broadphase_handle == ref.handle) {
                            out->type = nuka::scene::ShapeType::Sphere;
                            out->prim = RefFootSpherePrim(finger_centers[f],
                                                          config.fingertips[f].radius);
                            out->geom = nullptr;
                            return true;
                        }
                    }
                    return false;
                }
                if (ref.type == CollidableType::RigidBody &&
                    ref.handle == config.cup.broadphase_body_id) {
                    out->type = nuka::scene::ShapeType::ConvexHull;
                    out->geom = &cup_hull;
                    return true;
                }
                return false;
            };
            nuka::collision::BuildContactManifolds(drive_pairs, resolve, &manifolds);
        }
        uint32_t finger_points = 0u;
        for (const auto& m : manifolds) finger_points += m.point_count;

        if (!has_geom || manifolds.empty() || finger_points == 0u) {
            auto v2 = device.View();
            articulation::FeatherstoneAba::IntegratePosition(context, v2, dt);
            articulation::FeatherstoneAba::IntegrateFloatingBasePose(context, v2, dt);
            IntegrateCup();
            return;
        }

        nuka::constraint::ContactRowComplianceInputs inputs;
        inputs.vel = 0.0f;
        inputs.invweight = 1.0f;
        inputs.dt = dt;
        inputs.condim = config.condim;
        RowBuffers rows;
        std::vector<ContactRowSides> sides;
        nuka::constraint::EmitCompliantContactRows(manifolds, inputs, &rows, &sides);
        for (uint32_t r = 0u; r < rows.RowCount(); ++r)
            rows.materials[r].friction = config.friction_mu;

        const std::vector<float> minv =
            RefInverseInertia(context, live, gravity_z, dof_stride);
        std::vector<float> chain_jacobians;
        std::vector<float> qdot(dof_stride, 0.0f);
        for (uint32_t i = 0u; i < base_dof && i < dof_stride; ++i)
            qdot[i] = live.link_velocity[root_link].v[i];
        for (uint32_t leg = root_link + 1u; leg < link_count; ++leg) {
            const uint32_t col = RefDofIndexOf(live, root_link, leg);
            if (col < dof_stride) qdot[col] = live.qdot[leg];
        }

        const uint32_t kArtIndex = 0u;
        const uint32_t body_count = 1u;  // ctx.state = [cup] (N=1 total body count).
        std::vector<RowArticulationRefs> art_refs(rows.RowCount());
        RowArticulationSide none_side{};
        for (uint32_t r = 0u; r < rows.RowCount(); ++r) {
            const ContactRowSides& s = sides[r];
            const bool a_art = s.a.react == ReactionProviderKind::ArticulationChainJ;
            const bool b_art = s.b.react == ReactionProviderKind::ArticulationChainJ;
            const bool a_cup = s.a.react == ReactionProviderKind::RigidInvMass;
            const bool b_cup = s.b.react == ReactionProviderKind::RigidInvMass;
            art_refs[r].a = none_side;
            art_refs[r].b = none_side;
            if (!((a_art && b_cup) || (b_art && a_cup))) continue;
            const int finger_local = a_art ? 0 : 1;
            const int cup_local = a_art ? 1 : 0;
            const RowJacobian6 j_finger =
                rows.JacobianForRowBody(r, static_cast<uint32_t>(finger_local));
            const Vec3 finger_dir = j_finger.linear;
            const Vec3 contact_point = sides[r].contact_point;
            const uint32_t finger_handle = a_art ? s.a.handle : s.b.handle;
            uint32_t finger_link = finger_handle;
            for (size_t f = 0u; f < config.fingertips.size(); ++f) {
                if (config.fingertips[f].broadphase_handle == finger_handle) {
                    finger_link = config.fingertips[f].link;
                    break;
                }
            }
            const std::vector<float> chain_j = RefFootChainJ(
                context, live, poses, finger_link, contact_point, finger_dir, dof_stride);
            const uint32_t slot =
                static_cast<uint32_t>(chain_jacobians.size() / dof_stride);
            chain_jacobians.insert(chain_jacobians.end(), chain_j.begin(),
                                   chain_j.end());
            rows.body_indices[2u * r + static_cast<uint32_t>(cup_local)] = 0u;
            rows.body_indices[2u * r + static_cast<uint32_t>(finger_local)] =
                body_count + kArtIndex;
            const RowArticulationSide finger_side{kArtIndex, slot};
            art_refs[r].a = (finger_local == 0) ? finger_side : none_side;
            art_refs[r].b = (finger_local == 1) ? finger_side : none_side;
        }

        std::vector<BodyState> bodies = {cup};
        nuka::solver::SolverConfig cfg;
        cfg.velocity_iterations = 64u;
        cfg.position_iterations = 0u;
        cfg.slop = 0.0f;
        cfg.baumgarte = 0.0f;
        nuka::solver::SolveContext ctx;
        ctx.rows = &rows;
        ctx.state = &bodies;
        ctx.sides = &sides;
        ctx.dt = dt;
        ctx.articulation.art_refs = &art_refs;
        ctx.articulation.chain_jacobians =
            chain_jacobians.empty() ? nullptr : &chain_jacobians;
        ctx.articulation.inertia_m_inv = &minv;
        ctx.articulation.qdot = &qdot;
        ctx.articulation.dof_stride = dof_stride;
        nuka::solver::UnifiedSolve(ctx, cfg);
        cup = bodies[0];

        for (uint32_t i = 0u; i < base_dof && i < dof_stride; ++i)
            live.link_velocity[root_link].v[i] = qdot[i];
        for (uint32_t leg = root_link + 1u; leg < link_count; ++leg) {
            const uint32_t col = RefDofIndexOf(live, root_link, leg);
            if (col < dof_stride) live.qdot[leg] = qdot[col];
        }
        device.link_velocity.CopyFromHost(
            live.link_velocity.data(),
            live.link_velocity.size() * sizeof(articulation::LinkSpatialVel));
        device.qdot.CopyFromHost(live.qdot.data(), live.qdot.size() * sizeof(float));
        context.stream.Synchronize();

        auto v3 = device.View();
        articulation::FeatherstoneAba::IntegratePosition(context, v3, dt);
        articulation::FeatherstoneAba::IntegrateFloatingBasePose(context, v3, dt);
        IntegrateCup();
    }

    void Download(articulation::ArticulationHostState* out) const {
        *out = gripper.host;
        articulation::DownloadArticulationState(device, out);
    }
};

// ---------------------------------------------------------------------------
// GATE A1: an N=1 batched grasp world matches the standalone replicated grasp
// reference byte-exact over the full run (isolates the batched plumbing).
// ---------------------------------------------------------------------------
TEST(BatchedUnifiedWorldGrasp, Gate_A1_N1_MatchesStandaloneReference) {
    if (!GraspCupAvailable())
        GTEST_SKIP() << "newton-assets cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const GraspCupHull hull = LoadGraspCupHull();
    ASSERT_GT(hull.vcount, 0u);

    const GraspSceneBundle gs = BuildGraspSceneBundle(hull, /*grip_force=*/8.0f,
                                                      /*mu=*/0.8f);
    const auto tmpl = MakeGraspTemplate(gs);
    const uint32_t kRun = 100u;

    coresident::BatchedUnifiedWorld world(context, tmpl, 1u, kGraspGravityZ, kGraspDt);
    for (uint32_t s = 0u; s < kRun; ++s) world.Step();

    StandaloneGraspRef ref(context, gs, kGraspGravityZ, kGraspDt);
    for (uint32_t s = 0u; s < kRun; ++s) ref.Step();

    const BodyState& world_cup = world.Body(0u, 0u);
    EXPECT_EQ(std::memcmp(&world_cup, &ref.cup, sizeof(BodyState)), 0)
        << "N=1 batched cup was not byte-exact vs the standalone grasp reference";
    articulation::ArticulationHostState world_art, ref_art;
    world.DownloadGripper(&world_art);
    ref.Download(&ref_art);
    ASSERT_EQ(world_art.q.size(), ref_art.q.size());
    EXPECT_EQ(std::memcmp(world_art.q.data(), ref_art.q.data(),
                          world_art.q.size() * sizeof(float)), 0)
        << "gripper q diverged (A1 plumbing)";
    ASSERT_EQ(world_art.qdot.size(), ref_art.qdot.size());
    EXPECT_EQ(std::memcmp(world_art.qdot.data(), ref_art.qdot.data(),
                          world_art.qdot.size() * sizeof(float)), 0)
        << "gripper qdot diverged (A1 plumbing)";
    std::printf("[GRASP A1] cup + gripper q/qdot byte-identical (N=1 batched == "
                "standalone ref) over %u steps\n", kRun);
}

// ---------------------------------------------------------------------------
// GATE A2: anchor to the VALIDATED co-resident StepGrasp oracle. SAME gripper / cup /
// grip / gravity / dt; assert the cup pose/vel + gripper qdot match at a TIGHT 1e-5
// (NOT byte-exact -- cross-TU replicated glue + condim=3 friction-cone host accumulation
// order differ; correct-by-construction guarantees solve equivalence, not FP bit-identity
// -- the P2.2 A2 precedent). Asset-gated SKIP. Prints the actual delta.
// ---------------------------------------------------------------------------
TEST(BatchedUnifiedWorldGrasp, Gate_A2_N1_MatchesCoResidentStepGrasp) {
    if (!GraspCupAvailable())
        GTEST_SKIP() << "newton-assets cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const GraspCupHull hull = LoadGraspCupHull();
    ASSERT_GT(hull.vcount, 0u);

    const GraspSceneBundle gs = BuildGraspSceneBundle(hull, /*grip_force=*/8.0f,
                                                      /*mu=*/0.8f);
    const auto tmpl = MakeGraspTemplate(gs);
    const uint32_t kRun = 100u;

    coresident::BatchedUnifiedWorld world(context, tmpl, 1u, kGraspGravityZ, kGraspDt);
    for (uint32_t s = 0u; s < kRun; ++s) world.Step();
    const BodyState& world_cup = world.Body(0u, 0u);

    coresident::UnifiedCoResidentStepper oracle(context, gs.gripper.host, gs.config,
                                                kGraspGravityZ, kGraspDt);
    double oracle_last_impulse = 0.0;
    for (uint32_t s = 0u; s < kRun; ++s) {
        const auto rep = oracle.Step();
        oracle_last_impulse = rep.cup_vertical_impulse;
    }
    const BodyState oracle_cup = oracle.Cup();

    const Vec3 dp = world_cup.position - oracle_cup.position;
    const Vec3 dv = world_cup.linear_velocity - oracle_cup.linear_velocity;
    const double w_imp = world.GraspReports()[0].cup_vertical_impulse;
    std::printf("[GRASP A2] N=1 batched vs co-resident StepGrasp: |dpos|=%.3e |dvel|=%.3e "
                " cup_z(batched=%.6f oracle=%.6f) vert_impulse(batched=%.5e oracle=%.5e)\n",
                dp.Length(), dv.Length(), world_cup.position.z, oracle_cup.position.z,
                w_imp, oracle_last_impulse);

    ExpectBodyEq(world_cup, oracle_cup, 1.0e-5f, "batched-vs-coresident-StepGrasp");
    articulation::ArticulationHostState world_art, oracle_art;
    world.DownloadGripper(&world_art);
    oracle.Download(&oracle_art);
    ASSERT_EQ(world_art.qdot.size(), oracle_art.qdot.size());
    for (size_t i = 0u; i < world_art.qdot.size(); ++i)
        EXPECT_NEAR(world_art.qdot[i], oracle_art.qdot[i], 1.0e-5f)
            << "gripper qdot[" << i << "] diverged vs the oracle";
}

// ---------------------------------------------------------------------------
// GATE HOLD: the cup is HELD against gravity by FINGER FRICTION ALONE (NO table).
// Σ(cup-side finger vertical impulse) ~= m*g*dt at steady state; cup drift << free fall;
// NO emitted row carries a StaticNull side (anti-greenwash). Mirrors GraspHoldSpike.
// ---------------------------------------------------------------------------
TEST(BatchedUnifiedWorldGrasp, Gate_HOLD_FrictionOnlyNoTable) {
    if (!GraspCupAvailable())
        GTEST_SKIP() << "newton-assets cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const GraspCupHull hull = LoadGraspCupHull();
    ASSERT_GT(hull.vcount, 0u);

    constexpr float kGripForce = 8.0f;
    constexpr float kMu = 0.8f;
    const GraspSceneBundle gs = BuildGraspSceneBundle(hull, kGripForce, kMu);
    const auto tmpl = MakeGraspTemplate(gs);
    const double cup_z0 = gs.config.cup_state.position.z;

    coresident::BatchedUnifiedWorld world(context, tmpl, 1u, kGraspGravityZ, kGraspDt);

    const uint32_t kSteps = 120u;
    constexpr uint32_t kSettle = 30u;
    const double weight_kick =
        static_cast<double>(kCupMass) * (-kGraspGravityZ) * kGraspDt;

    uint32_t contact_steps = 0u, held_steps = 0u, steady_held = 0u;
    double max_balance_err = 0.0, max_crosscheck_err = 0.0;
    bool any_static = false;
    for (uint32_t s = 0u; s < kSteps; ++s) {
        world.Step();
        const auto& rep = world.GraspReports()[0];
        if (rep.finger_contacts > 0u) ++contact_steps;
        if (rep.any_static_row) any_static = true;
        ASSERT_TRUE(std::isfinite(rep.cup_z) && std::isfinite(rep.cup_vz))
            << "cup state non-finite at step " << s;
        if (rep.finger_contacts > 0u && rep.cup_vertical_impulse > 0.0) {
            ++held_steps;
            const double bal = std::fabs(rep.cup_vertical_impulse - weight_kick) /
                               std::max(weight_kick, 1e-9);
            const double cross =
                std::fabs(rep.cup_vertical_impulse - rep.cup_dvz_impulse) /
                std::max(std::fabs(rep.cup_vertical_impulse), 1e-9);
            max_crosscheck_err = std::max(max_crosscheck_err, cross);
            if (s >= kSettle) {
                ++steady_held;
                max_balance_err = std::max(max_balance_err, bal);
            }
        }
        if (s < 4u || s == kSteps - 1u) {
            std::printf("[GRASP HOLD] step %3u: cup_z=%.5f vz=%.5f vert_impulse=%.5e "
                        "(mgdt=%.5e bal=%.3f) contacts=%u\n",
                        s, rep.cup_z, rep.cup_vz, rep.cup_vertical_impulse, weight_kick,
                        std::fabs(rep.cup_vertical_impulse - weight_kick) /
                            std::max(weight_kick, 1e-9),
                        rep.finger_contacts);
        }
    }

    const BodyState cupF = world.Body(0u, 0u);
    const double total_drift = std::fabs(cupF.position.z - cup_z0);
    const double free_fall = 0.5 * static_cast<double>(-kGraspGravityZ) *
                             (kSteps * static_cast<double>(kGraspDt)) *
                             (kSteps * static_cast<double>(kGraspDt));
    std::printf("[GRASP HOLD GATE] held %u/%u, contact %u/%u; cup drift=%.5f m "
                "(free-fall=%.4f m); balance err=%.3f crosscheck err=%.3e any_static=%d\n",
                held_steps, kSteps, contact_steps, kSteps, total_drift, free_fall,
                max_balance_err, max_crosscheck_err, any_static ? 1 : 0);

    EXPECT_FALSE(any_static)
        << "a StaticNull (table/ground) row appeared -- the cup must be held by FINGER "
           "FRICTION ALONE, no support of any kind";
    EXPECT_GE(contact_steps, kSteps - 2u) << "the cup left the pinch";
    EXPECT_GE(held_steps, kSteps - 2u) << "the fingers never delivered an upward impulse";
    EXPECT_GE(steady_held, kSteps - kSettle - 2u) << "the steady-state hold did not persist";
    EXPECT_LT(total_drift, 0.1 * free_fall) << "the cup drifted like a free-fall";
    EXPECT_LT(max_balance_err, 0.05)
        << "the steady-state finger vertical impulse did NOT balance the cup weight";
    EXPECT_LT(max_crosscheck_err, 1.0e-3)
        << "the row-impulse sum and the cup-velocity-change disagree";
}

// ---------------------------------------------------------------------------
// GATE BITE (mandatory anti-green-wash): grip=0 -> friction collapses -> the cup FALLS
// (~free-fall, final vz < -1, drop >> the held case). A hold surviving grip=0 is fake.
// ---------------------------------------------------------------------------
TEST(BatchedUnifiedWorldGrasp, Gate_BITE_GripOffCupFalls) {
    if (!GraspCupAvailable())
        GTEST_SKIP() << "newton-assets cup not present (fetch-per-env)";
    const auto context = nuka::phi::MakeDefaultDeviceContext();
    const GraspCupHull hull = LoadGraspCupHull();
    ASSERT_GT(hull.vcount, 0u);

    const GraspSceneBundle gs = BuildGraspSceneBundle(hull, /*grip_force=*/0.0f,
                                                      /*mu=*/0.8f);
    const auto tmpl = MakeGraspTemplate(gs);
    const double cup_z0 = gs.config.cup_state.position.z;

    coresident::BatchedUnifiedWorld world(context, tmpl, 1u, kGraspGravityZ, kGraspDt);
    const uint32_t kSteps = 120u;
    for (uint32_t s = 0u; s < kSteps; ++s) world.Step();
    const BodyState cupF = world.Body(0u, 0u);

    const double drop = cup_z0 - cupF.position.z;
    const double free_fall = 0.5 * static_cast<double>(-kGraspGravityZ) *
                             (kSteps * static_cast<double>(kGraspDt)) *
                             (kSteps * static_cast<double>(kGraspDt));
    std::printf("[GRASP BITE] grip=0: cup dropped %.5f m over %u steps (free-fall=%.4f m); "
                "final vz=%.4f m/s\n", drop, kSteps, free_fall, cupF.linear_velocity.z);

    EXPECT_GT(drop, 0.5 * free_fall)
        << "grip=0 did NOT drop the cup -- the hold survives without the grip (FAKE)";
    EXPECT_LT(cupF.linear_velocity.z, -1.0f)
        << "grip=0 cup is not accelerating downward -- not in free fall";
}

}  // namespace
