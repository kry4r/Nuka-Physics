#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::rigid – Symplectic Euler integrator for rigid bodies
// ---------------------------------------------------------------------------

#include "runtime/rigid/body_state.hpp"
#include "math/vec3.hpp"

namespace nuka::runtime::rigid {

/// Apply gravity to linear velocity (skipped for static bodies).
void IntegrateGravity(BodyState& body, math::Vec3 gravity, float dt);

/// Apply accumulated external force/torque to velocities (skipped for static).
void IntegrateForces(BodyState& body, float dt);

/// Integrate velocity to position and orientation – symplectic Euler.
void IntegrateVelocity(BodyState& body, float dt);

/// Full step: gravity + forces + position update.
void StepBody(BodyState& body, math::Vec3 gravity, float dt);

} // namespace nuka::runtime::rigid
