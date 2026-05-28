#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::articulation -- CUDA Featherstone ABA declarations
// ---------------------------------------------------------------------------

#include "runtime/articulation/articulation_state.hpp"

namespace nuka::runtime::articulation {

void LaunchFeatherstoneAbaKernels(const phi::DeviceContext& context,
                                  ArticulationDeviceState state,
                                  float gravity_z);
void LaunchIntegrateArticulationKernels(const phi::DeviceContext& context,
                                        ArticulationDeviceState state,
                                        float dt);

} // namespace nuka::runtime::articulation
