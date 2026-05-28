#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- Featherstone generalized Jacobian builder
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"

#include <cstdint>

namespace nuka::runtime::articulation {

void ComputeLinkPointJacobians(const phi::DeviceContext& context,
                               ArticulationDeviceState state,
                               const uint32_t* contact_link_indices,
                               const math::Vec3* contact_point_world,
                               uint32_t contact_count,
                               float* out_jacobian_scalars);

} // namespace nuka::runtime::articulation
