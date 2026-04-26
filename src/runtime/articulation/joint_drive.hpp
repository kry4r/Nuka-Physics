#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation – Joint drive parameters and step result
// ---------------------------------------------------------------------------

#include <cfloat>
#include <cstdint>

namespace nuka::runtime::articulation {

struct JointDrive {
    float target_position = 0.0f;
    float target_velocity = 0.0f;
    float stiffness = 0.0f;
    float damping = 10.0f;
    float max_force = FLT_MAX;
};

struct DriveStepResult {
    float joint_velocity;
    float joint_position;
    float applied_force;
};

} // namespace nuka::runtime::articulation
