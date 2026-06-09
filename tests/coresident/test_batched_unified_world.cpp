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

#include "math/quat.hpp"
#include "math/vec3.hpp"
#include "phi/device_context.hpp"
#include "runtime/coresident/batched_unified_world.hpp"
#include "runtime/rigid/body_state.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
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

}  // namespace
