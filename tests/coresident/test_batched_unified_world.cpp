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
#include "constraint/contact_manifold.hpp"     // ContactManifold
#include "constraint/contact_row_sides.hpp"    // ContactRowSides
#include "constraint/reaction_provider.hpp"    // ReactionProviderKind
#include "constraint/row.hpp"                  // kInvalidBodyIndex
#include "constraint/row_buffers.hpp"          // RowBuffers
#include "constraint/row_builder.hpp"          // EmitCompliantContactRows / inputs
#include "import/usd_importer.hpp"             // LoadUsd (Gate A2 go2 scene)
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"  // UpdateWorldLinkPoses
#include "runtime/articulation/articulation_state.hpp"     // BuildArticulationHostState
#include "runtime/coresident/batched_unified_world.hpp"
#include "runtime/coresident/unified_coresident_stepper.hpp"  // Gate A2 oracle
#include "runtime/rigid/body_state.hpp"
#include "runtime/world_builder.hpp"           // BuildWorld
#include "scene/canonical_types.hpp"           // scene::ShapeType
#include "scene/cooker.hpp"                    // CookScene
#include "solver/rigid_solver.hpp"             // SolverConfig
#include "solver/unified_solve.hpp"            // UnifiedSolve / SolveContext

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
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

}  // namespace
