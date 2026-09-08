#pragma once

#include <cstdint>

namespace nuka::phi {

enum class ArticulationJointType : uint8_t {
    Revolute = 0,
    Prismatic = 1,
    Fixed = 2,
    FloatingBase = 3,
};

inline constexpr uint32_t kMaxArticulationDof = 64u;
inline constexpr uint32_t kMaxOscDof = 18u;

}  // namespace nuka::phi
