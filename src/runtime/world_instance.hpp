#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::WorldInstance – mutable per-instance simulation state
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "math/vec3.hpp"

#include <cstdint>
#include <vector>

namespace nuka::runtime {

struct WorldInstance {
    uint32_t body_count = 0;

    std::vector<math::Transform> poses;
    std::vector<math::Vec3>      linear_velocities;
    std::vector<math::Vec3>      angular_velocities;
    std::vector<math::Vec3>      forces;
    std::vector<math::Vec3>      torques;
};

} // namespace nuka::runtime
