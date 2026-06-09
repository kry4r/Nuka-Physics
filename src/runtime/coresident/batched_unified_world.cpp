// ---------------------------------------------------------------------------
// nuka::runtime::coresident::BatchedUnifiedWorld -- implementation (v0.8 P2.1).
// ---------------------------------------------------------------------------
// P2.1 scope: per-env MOVABLE RIGID BODIES under gravity (free-fall, no contact,
// no articulation). The step loop is the SAME stage order as
// UnifiedCoResidentStepper / BatchedArticulatedWorld -- velocity stage (gravity
// kick) -> contact phase (EMPTY in P2.1) -> position stage -- so P2.2 inserts the
// batched contact phase between the two body loops with NO restructuring.
// ---------------------------------------------------------------------------

#include "runtime/coresident/batched_unified_world.hpp"

namespace nuka::runtime::coresident {

BatchedUnifiedWorld::BatchedUnifiedWorld(
    const phi::DeviceContext& context,
    const BatchedSceneTemplate& scene_template,
    uint32_t env_count,
    float gravity_z,
    float dt)
    : context_(context),
      env_count_(env_count),
      bodies_per_env_(
          static_cast<uint32_t>(scene_template.bodies_per_env.size())),
      gravity_z_(gravity_z),
      dt_(dt) {
    // Replicate the per-env body template into the env-major SoA: env e's bodies
    // occupy [e*k, (e+1)*k). This is the per-env body-id offset the batched
    // UnifiedSolve relies on (cross-env body ids are disjoint -> the colored solve
    // is N independent solves). Per-env initial-condition perturbation is applied
    // by the caller through BodyMut() after construction.
    bodies_.reserve(static_cast<size_t>(env_count_) * bodies_per_env_);
    for (uint32_t e = 0u; e < env_count_; ++e) {
        for (uint32_t i = 0u; i < bodies_per_env_; ++i) {
            bodies_.push_back(scene_template.bodies_per_env[i]);
        }
    }
}

void BatchedUnifiedWorld::IntegrateBodyPosition(
    runtime::rigid::BodyState& body) const {
    // BYTE-IDENTICAL to UnifiedCoResidentStepper::IntegrateBoxPosition: pure
    // symplectic-Euler kinematics (NO floor clamp; contact support flows through
    // the unified spine in the contact phase, which P2.2 adds).
    if (body.inv_mass <= 0.0f) return;
    body.position += body.linear_velocity * dt_;
    const math::Vec3 w = body.angular_velocity;
    math::Quat dq;
    dq.w = 1.0f;
    dq.x = 0.5f * w.x * dt_;
    dq.y = 0.5f * w.y * dt_;
    dq.z = 0.5f * w.z * dt_;
    body.orientation = (body.orientation * dq).Normalized();
}

void BatchedUnifiedWorld::Step() {
    // ----- velocity stage: gravity velocity-kick, env-major order (D1) ----------
    // Matches the co-resident box gravity kick (linear_velocity.z += g*dt) applied
    // BEFORE the contact phase. Immovable bodies (inv_mass<=0) are skipped, exactly
    // as IntegrateBoxPosition skips them.
    for (auto& body : bodies_) {
        if (body.inv_mass <= 0.0f) continue;
        body.linear_velocity.z += gravity_z_ * dt_;
    }

    // ----- contact phase ---------------------------------------------------------
    // P2.1: EMPTY (free-fall). P2.2 inserts the batched per-env narrowphase ->
    // concatenated compliant rows (per-env body-id offsets) -> single UnifiedSolve
    // here, between the velocity and position stages.

    // ----- position stage: symplectic-Euler integrate, env-major order (D1) ------
    for (auto& body : bodies_) {
        IntegrateBodyPosition(body);
    }
}

}  // namespace nuka::runtime::coresident
