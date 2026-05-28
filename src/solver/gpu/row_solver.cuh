#pragma once
// ---------------------------------------------------------------------------
// nuka::solver::gpu::row_solver -- CUDA Universal Row solver
// ---------------------------------------------------------------------------

#include "constraint/row_buffers.hpp"
#include "phi/device_context.hpp"
#include "runtime/gpu/cuda_constraint_row_buffer.hpp"
#include "runtime/rigid/body_state.hpp"

#include <cstdint>
#include <vector>

namespace nuka::solver::gpu {

struct RowSolveConfig {
    uint32_t velocity_iterations = 10u;
    uint32_t position_iterations = 4u;
    float slop = 0.005f;
    float baumgarte = 0.2f;
};

struct RowSolveReport {
    uint32_t row_count = 0u;
    uint32_t color_count = 0u;
    uint32_t velocity_iterations = 0u;
    uint32_t position_iterations = 0u;
    float max_position_error = 0.0f;
    runtime::gpu::CudaConstraintRowSchedulerReport row_scheduler_report;
};

RowSolveReport SolveRows(const phi::DeviceContext& context,
                         constraint::RowBuffers& rows,
                         std::vector<runtime::rigid::BodyState>& bodies,
                         const RowSolveConfig& config = {});
RowSolveReport SolveRows(const phi::DeviceContext& context,
                         constraint::RowBuffers& rows,
                         runtime::rigid::BodyState* bodies,
                         uint32_t body_count,
                         const RowSolveConfig& config = {});

} // namespace nuka::solver::gpu
