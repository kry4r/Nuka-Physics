// ---------------------------------------------------------------------------
// nuka::runtime::articulation – Joint row builders
// ---------------------------------------------------------------------------

#include "runtime/articulation/joint_constraints.hpp"

#include "constraint/row_builder.hpp"

#include <algorithm>

namespace nuka::runtime::articulation {

void AppendRevoluteRows(
    constraint::RowBuffers* out_rows,
    uint32_t body_a, uint32_t body_b,
    math::Vec3 axis,
    math::Transform parent_frame,
    math::Transform child_frame,
    float lower_limit, float upper_limit,
    float current_angle) {
    constraint::AppendRevoluteRows(out_rows,
                                   body_a,
                                   body_b,
                                   axis,
                                   parent_frame,
                                   child_frame,
                                   lower_limit,
                                   upper_limit,
                                   current_angle);
}

void AppendDriveRow(
    constraint::RowBuffers* out_rows,
    uint32_t body_a, uint32_t body_b,
    math::Vec3 axis,
    const JointDrive& drive,
    float current_velocity) {
    // The drive row is formulated with rhs = target_velocity; the velocity ERROR
    // (target - J.v) is formed by the solver from the integrated J.v at solve
    // time, so current_velocity is NOT folded into rhs here (doing so would
    // double-count it). It is unused on purpose -- see the FLAG to drop it from
    // the public signature once the constraint::AppendDriveRow seam is updated.
    (void)current_velocity;
    constraint::AppendDriveRow(out_rows,
                               body_a,
                               body_b,
                               axis,
                               drive.target_velocity,
                               drive.max_force);
}

DriveStepResult SimulateVelocityDriveStep() {
    const float dt = 1.0f / 60.0f;
    const float inertia = 1.0f;

    JointDrive drive{};
    drive.target_velocity = 1.0f;
    drive.damping = 10.0f;
    drive.max_force = 100.0f;

    float velocity = 0.0f;
    float position = 0.0f;

    for (int step = 0; step < 10; ++step) {
        float velocity_error = drive.target_velocity - velocity;
        float force = drive.damping * velocity_error;
        force = std::max(-drive.max_force, std::min(drive.max_force, force));

        float acceleration = force / inertia;
        velocity += acceleration * dt;
        position += velocity * dt;
    }

    return DriveStepResult{velocity, position, drive.damping * (drive.target_velocity - velocity)};
}

} // namespace nuka::runtime::articulation
