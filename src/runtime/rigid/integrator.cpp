// ---------------------------------------------------------------------------
// nuka::runtime::rigid – Integrator (CPU implementation)
// ---------------------------------------------------------------------------

#include "runtime/rigid/integrator.hpp"

namespace nuka::runtime::rigid {

void IntegrateGravity(BodyState& body, math::Vec3 gravity, float dt) {
    if (body.inv_mass == 0.0f) return;  // static body
    body.linear_velocity += gravity * dt;
}

void IntegrateForces(BodyState& body, float dt) {
    if (body.inv_mass == 0.0f) return;  // static body

    // Linear: v += F * inv_mass * dt
    body.linear_velocity += body.force * body.inv_mass * dt;

    // Angular: omega += torque * inv_inertia * dt  (component-wise)
    body.angular_velocity.x += body.torque.x * body.inv_inertia.x * dt;
    body.angular_velocity.y += body.torque.y * body.inv_inertia.y * dt;
    body.angular_velocity.z += body.torque.z * body.inv_inertia.z * dt;
}

void IntegrateVelocity(BodyState& body, float dt) {
    if (body.inv_mass == 0.0f) return;  // static body

    // Position: p += v * dt
    body.position += body.linear_velocity * dt;

    // Orientation: q += 0.5 * omega_quat * q * dt
    // where omega_quat = Quat(0, omega.x, omega.y, omega.z)
    const auto& q = body.orientation;
    const auto& w = body.angular_velocity;

    math::Quat omega_q{0.0f, w.x, w.y, w.z};
    math::Quat dq = omega_q * q;

    body.orientation.w += 0.5f * dq.w * dt;
    body.orientation.x += 0.5f * dq.x * dt;
    body.orientation.y += 0.5f * dq.y * dt;
    body.orientation.z += 0.5f * dq.z * dt;

    body.orientation = body.orientation.Normalized();
}

void StepBody(BodyState& body, math::Vec3 gravity, float dt) {
    IntegrateGravity(body, gravity, dt);
    IntegrateForces(body, dt);
    IntegrateVelocity(body, dt);
}

} // namespace nuka::runtime::rigid
