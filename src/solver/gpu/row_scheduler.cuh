#pragma once
// ---------------------------------------------------------------------------
// CUDA-visible row scheduler views
// ---------------------------------------------------------------------------

#include "solver/gpu/row_scheduler.hpp"

namespace nuka::solver::gpu {

struct DeviceRowColorPartitions {
    const uint32_t* row_indices = nullptr;
    const RowColorRange* color_ranges = nullptr;
    uint32_t color_count = 0u;
};

} // namespace nuka::solver::gpu
