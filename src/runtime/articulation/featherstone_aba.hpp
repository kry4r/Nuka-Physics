#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- CUDA Featherstone ABA public API
// ---------------------------------------------------------------------------

#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_state.hpp"

namespace nuka::runtime::articulation {

class FeatherstoneAba {
public:
    static void ApplyPositionDrives(const phi::DeviceContext& context,
                                    ArticulationDeviceState state,
                                    const float* drive_targets,
                                    const float* drive_stiffness,
                                    const float* drive_damping,
                                    const float* drive_force_limits);
    static void ComputeAccelerations(const phi::DeviceContext& context,
                                     ArticulationDeviceState state,
                                     float gravity_z);
    static void Integrate(const phi::DeviceContext& context,
                          ArticulationDeviceState state,
                          float dt);
};

} // namespace nuka::runtime::articulation
