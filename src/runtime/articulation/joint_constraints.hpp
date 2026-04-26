#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation – Joint constraint builders
// ---------------------------------------------------------------------------

#include "constraint/constraint_block.hpp"
#include "math/vec3.hpp"
#include "math/transform.hpp"
#include "runtime/articulation/joint_drive.hpp"

#include <cstdint>

namespace nuka::runtime::articulation {

/// Build constraint blocks for a revolute joint.
/// Returns a block with 5 rows: 3 translational + 2 rotational constraints
/// (the revolute axis is left free).
constraint::ConstraintBlock BuildRevoluteConstraint(
    uint32_t body_a, uint32_t body_b,
    math::Vec3 axis,
    math::Transform parent_frame,
    math::Transform child_frame,
    float lower_limit, float upper_limit,
    float current_angle = 0.0f
);

/// Build a drive constraint as an additional constraint row.
constraint::ConstraintBlock BuildDriveConstraint(
    uint32_t body_a, uint32_t body_b,
    math::Vec3 axis,
    const JointDrive& drive,
    float current_velocity
);

/// Simulate a simple velocity drive step for testing.
DriveStepResult SimulateVelocityDriveStep();

} // namespace nuka::runtime::articulation
