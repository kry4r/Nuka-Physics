#pragma once
// ---------------------------------------------------------------------------
// CUDA-visible row scheduler views
// ---------------------------------------------------------------------------

#include "solver/gpu/row_scheduler.hpp"

namespace nuka::solver::gpu {

// Device view of RowComponentPartitions (one solver CUDA block per component;
// see row_scheduler.hpp for the byte-identity argument).
struct DeviceRowComponentPartitions {
    const uint32_t* row_indices = nullptr;
    const RowColorRange* segments = nullptr;
    const RowComponentRange* components = nullptr;
    uint32_t component_count = 0u;
};

} // namespace nuka::solver::gpu
