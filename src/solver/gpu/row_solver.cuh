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

namespace nuka::constraint {
struct ContactRowSides;  // v0.8 C5a: defined in constraint/row_builder.hpp
}  // namespace nuka::constraint

namespace nuka::solver::gpu {

struct RowSolveConfig {
    uint32_t velocity_iterations = 10u;
    uint32_t position_iterations = 4u;
    float slop = 0.005f;
    float baumgarte = 0.2f;
    // v0.8 C5a: substep dt. Used ONLY by the compliant branch (aref acceleration
    // -> velocity-target scale). Legacy SolveRows leaves it 0 (no compliant rows
    // -> never read), so the legacy solve stays byte-identical.
    float dt = 0.0f;
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

// v0.8 C5a: the SAME graph-colored kernel, but with the per-row ContactRowSides
// stream uploaded so the COMPLIANT branch can dispatch each side's reaction by
// side.react. `sides` must be one-per-row, in the SAME order as rows.rows (the
// EmitCompliantContactRows contract). Legacy SolveRows above passes no sides and
// is byte-identical (the kernel reads sides only on a Compliant row). This is the
// device entry the host UnifiedSolve adapter calls.
RowSolveReport SolveRowsWithSides(const phi::DeviceContext& context,
                                  constraint::RowBuffers& rows,
                                  runtime::rigid::BodyState* bodies,
                                  uint32_t body_count,
                                  const constraint::ContactRowSides* sides,
                                  uint32_t sides_count,
                                  const RowSolveConfig& config = {});

} // namespace nuka::solver::gpu
