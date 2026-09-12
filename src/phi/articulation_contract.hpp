#pragma once

#include <cstdint>

namespace nuka::phi {

enum class ArticulationJointType : uint8_t {
    Revolute = 0,
    Prismatic = 1,
    Fixed = 2,
    FloatingBase = 3,
};

enum class ArticulationControlMode : uint8_t {
    PDPosition = 0,
    Torque = 1,
    Velocity = 2,
    ComputedTorque = 3,
    Osc = 4,
    Actuator = 5,
};

inline constexpr uint32_t kMaxArticulationDof = 64u;

}  // namespace nuka::phi
