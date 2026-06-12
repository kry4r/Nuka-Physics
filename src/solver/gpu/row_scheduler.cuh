#pragma once
// DEPRECATED(M9): SURVIVES ONLY for the legacy BatchedUnifiedWorld (the M4-M9
// parity ORACLE, edited again in M5) + its co-resident/test consumers. The nk
// core consumes NONE of this: the M4 SolveRowsBlockIsland/AssembleRows ops +
// nk::SolveSchedule (src/nk/solve/) carry the migrated row math + scheduling.
// Deleted with src/runtime/coresident/ at M9. Do NOT add new consumers.
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
