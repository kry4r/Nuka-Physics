// ---------------------------------------------------------------------------
// nuka::runtime::articulation – Joint constraint builders
// ---------------------------------------------------------------------------

#include "runtime/articulation/joint_constraints.hpp"

#include <algorithm>
#include <cmath>

namespace nuka::runtime::articulation {

// ---------------------------------------------------------------------------
// BuildRevoluteConstraint
// ---------------------------------------------------------------------------

constraint::ConstraintBlock BuildRevoluteConstraint(
    uint32_t body_a, uint32_t body_b,
    math::Vec3 axis,
    math::Transform parent_frame,
    math::Transform child_frame,
    float lower_limit, float upper_limit,
    float current_angle)
{
    constraint::ConstraintBlock block{};
    block.type = constraint::ConstraintType::Joint;
    block.body_a = body_a;
    block.body_b = body_b;
    block.row_count = 5; // 3 translational + 2 rotational (perpendicular to axis)
    block.anchor_local_a = parent_frame.position;
    block.anchor_local_b = child_frame.position;

    const math::Vec3 norm_axis = axis.Normalized();

    // Build two perpendicular axes to the revolute axis
    math::Vec3 perp1, perp2;
    if (std::abs(norm_axis.x) < 0.9f) {
        perp1 = norm_axis.Cross(math::Vec3::UnitX()).Normalized();
    } else {
        perp1 = norm_axis.Cross(math::Vec3::UnitY()).Normalized();
    }
    perp2 = norm_axis.Cross(perp1).Normalized();

    // Rows 0-2: translational constraints (lock position)
    const math::Vec3 tx = math::Vec3::UnitX();
    const math::Vec3 ty = math::Vec3::UnitY();
    const math::Vec3 tz = math::Vec3::UnitZ();
    const math::Vec3 dirs[3] = {tx, ty, tz};

    const math::Vec3 r_a = parent_frame.position;
    const math::Vec3 r_b = child_frame.position;

    for (uint32_t i = 0; i < 3; ++i) {
        block.jacobian_linear_a[i]  = dirs[i];
        block.jacobian_angular_a[i] = r_a.Cross(dirs[i]);
        block.jacobian_linear_b[i]  = -dirs[i];
        block.jacobian_angular_b[i] = -(r_b.Cross(dirs[i]));

        block.rhs[i] = 0.0f;
        block.lower_limit[i] = -1e6f;
        block.upper_limit[i] = 1e6f;
    }

    // Rows 3-4: rotational constraints (lock rotation around perp axes)
    const math::Vec3 rot_axes[2] = {perp1, perp2};
    for (uint32_t i = 0; i < 2; ++i) {
        uint32_t row = 3 + i;
        block.jacobian_linear_a[row]  = math::Vec3::Zero();
        block.jacobian_angular_a[row] = rot_axes[i];
        block.jacobian_linear_b[row]  = math::Vec3::Zero();
        block.jacobian_angular_b[row] = -rot_axes[i];

        // If at limits, add bias
        if (current_angle <= lower_limit) {
            block.rhs[row] = 0.0f;
            block.lower_limit[row] = -1e6f;
            block.upper_limit[row] = 1e6f;
        } else if (current_angle >= upper_limit) {
            block.rhs[row] = 0.0f;
            block.lower_limit[row] = -1e6f;
            block.upper_limit[row] = 1e6f;
        } else {
            block.rhs[row] = 0.0f;
            block.lower_limit[row] = -1e6f;
            block.upper_limit[row] = 1e6f;
        }
    }

    return block;
}

// ---------------------------------------------------------------------------
// BuildDriveConstraint
// ---------------------------------------------------------------------------

constraint::ConstraintBlock BuildDriveConstraint(
    uint32_t body_a, uint32_t body_b,
    math::Vec3 axis,
    const JointDrive& drive,
    float current_velocity)
{
    (void)current_velocity;

    constraint::ConstraintBlock block{};
    block.type = constraint::ConstraintType::Drive;
    block.body_a = body_a;
    block.body_b = body_b;
    block.row_count = 1;

    const math::Vec3 norm_axis = axis.Normalized();

    block.jacobian_linear_a[0]  = math::Vec3::Zero();
    block.jacobian_angular_a[0] = norm_axis;
    block.jacobian_linear_b[0]  = math::Vec3::Zero();
    block.jacobian_angular_b[0] = -norm_axis;

    // Drive target: rhs stores the desired relative row velocity. The solver
    // computes rhs - Jv, so current velocity must not be subtracted here.
    block.rhs[0] = drive.target_velocity;

    block.lower_limit[0] = -drive.max_force;
    block.upper_limit[0] = drive.max_force;

    return block;
}

// ---------------------------------------------------------------------------
// SimulateVelocityDriveStep
// ---------------------------------------------------------------------------

DriveStepResult SimulateVelocityDriveStep() {
    // Simple simulation: a body with a velocity drive targeting +1 rad/s
    const float dt = 1.0f / 60.0f;
    const float inertia = 1.0f;

    JointDrive drive{};
    drive.target_velocity = 1.0f;
    drive.damping = 10.0f;
    drive.max_force = 100.0f;

    float velocity = 0.0f;
    float position = 0.0f;

    // Simulate a few steps of PD-like velocity drive
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
