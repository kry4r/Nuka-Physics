#pragma once
// ---------------------------------------------------------------------------
// nuka::core::diagnostics - CUDA invariant reductions
// ---------------------------------------------------------------------------

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/device_context.hpp"

#include <cstdint>

namespace nuka::core::diagnostics::gpu {

struct DeviceInvariantSnapshot {
    float energy = 0.0f;
    math::Vec3 linear_momentum = math::Vec3::Zero();
    math::Vec3 angular_momentum = math::Vec3::Zero();
    uint32_t nan_inf_count = 0;
    float max_speed = 0.0f;
    float max_position_abs = 0.0f;
};

void ComputeRigidBodyInvariantSnapshot(const phi::DeviceContext& context,
                                       uint32_t body_count,
                                       const math::Transform* poses,
                                       const math::Vec3* linear_velocities,
                                       const math::Vec3* angular_velocities,
                                       const float* inverse_masses,
                                       const math::Vec3* inverse_inertias,
                                       math::Vec3 gravity,
                                       DeviceInvariantSnapshot* out_snapshot);

} // namespace nuka::core::diagnostics::gpu
