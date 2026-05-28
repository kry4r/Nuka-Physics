#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation – Joint constraint builders
// ---------------------------------------------------------------------------

#include "constraint/row_buffers.hpp"
#include "math/vec3.hpp"
#include "math/transform.hpp"
#include "runtime/articulation/joint_drive.hpp"

#include <cstdint>

namespace nuka::runtime::articulation {

/// Append rows for a revolute joint: 3 translational + 2 rotational
/// constraints. The revolute axis remains free.
void AppendRevoluteRows(
    constraint::RowBuffers* out_rows,
    uint32_t body_a, uint32_t body_b,
    math::Vec3 axis,
    math::Transform parent_frame,
    math::Transform child_frame,
    float lower_limit, float upper_limit,
    float current_angle = 0.0f);

/// Append a drive constraint row.
void AppendDriveRow(
    constraint::RowBuffers* out_rows,
    uint32_t body_a, uint32_t body_b,
    math::Vec3 axis,
    const JointDrive& drive,
    float current_velocity);

/// Simulate a simple velocity drive step for testing.
DriveStepResult SimulateVelocityDriveStep();

} // namespace nuka::runtime::articulation
