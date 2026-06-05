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
struct RowArticulationRefs;  // v0.8 C5b: constraint/row_articulation_refs.hpp
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

// v0.8 C5b-core: the per-row articulation indirection + the global articulation
// device buffers the kernel resolves an ArticulationChainJ contact side against.
// All host pointers/spans; SolveRowsWithSides uploads them (same pattern as
// `sides`) and DOWNLOADS qdot afterwards (the apply mutates it). Legacy / C5a
// callers leave everything null/zero -> the articulation arms never fire ->
// byte-identical. See constraint/row_articulation_refs.hpp for the layout.
//   art_refs       : one RowArticulationRefs per row (same order as rows.rows).
//   chain_jacobians: per (row,side) articulation slot, dof_stride floats each.
//   inertia_m_inv  : per articulation, dof_stride*dof_stride floats each.
//   qdot           : per articulation, dof_stride floats each (MUTABLE, downloaded).
struct RowArticulationData {
    const constraint::RowArticulationRefs* art_refs = nullptr;
    uint32_t art_refs_count = 0u;
    const float* chain_jacobians = nullptr;
    uint32_t chain_jacobians_count = 0u;  // == slot_count * dof_stride
    const float* inertia_m_inv = nullptr;
    uint32_t inertia_m_inv_count = 0u;    // == art_count * dof_stride * dof_stride
    float* qdot = nullptr;                // host buffer, mutated in place
    uint32_t qdot_count = 0u;             // == art_count * dof_stride
    uint32_t dof_stride = 0u;
};

// v0.8 C5a: the SAME graph-colored kernel, but with the per-row ContactRowSides
// stream uploaded so the COMPLIANT branch can dispatch each side's reaction by
// side.react. `sides` must be one-per-row, in the SAME order as rows.rows (the
// EmitCompliantContactRows contract). Legacy SolveRows above passes no sides and
// is byte-identical (the kernel reads sides only on a Compliant row). This is the
// device entry the host UnifiedSolve adapter calls.
//
// v0.8 C5b-core: `articulation` additionally carries the reduced-coordinate
// chain-J reaction buffers so a compliant ArticulationChainJ side resolves its
// J M^-1 J^T + qdot apply via the C4c provider math. Empty/null -> inert.
RowSolveReport SolveRowsWithSides(const phi::DeviceContext& context,
                                  constraint::RowBuffers& rows,
                                  runtime::rigid::BodyState* bodies,
                                  uint32_t body_count,
                                  const constraint::ContactRowSides* sides,
                                  uint32_t sides_count,
                                  const RowSolveConfig& config = {},
                                  const RowArticulationData& articulation = {});

} // namespace nuka::solver::gpu
