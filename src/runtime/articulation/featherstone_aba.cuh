#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- CUDA Featherstone ABA declarations
// ---------------------------------------------------------------------------

#include "runtime/articulation/articulation_state.hpp"

namespace nuka::runtime::articulation {

void LaunchFeatherstoneAbaKernels(const phi::DeviceContext& context,
                                  ArticulationDeviceState state,
                                  float gravity_z);
void LaunchApplyPositionDriveKernels(const phi::DeviceContext& context,
                                     ArticulationDeviceState state,
                                     const float* drive_targets,
                                     const float* drive_stiffness,
                                     const float* drive_damping,
                                     const float* drive_force_limits,
                                     bool defer_velocity_damping = false);
void LaunchIntegrateArticulationKernels(const phi::DeviceContext& context,
                                        ArticulationDeviceState state,
                                        float dt);

} // namespace nuka::runtime::articulation
