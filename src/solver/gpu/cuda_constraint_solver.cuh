#pragma once
// ---------------------------------------------------------------------------
// nuka::solver::gpu::cuda_constraint_solver -- CUDA constraint rows and PGS
// ---------------------------------------------------------------------------

#include "constraint/constraint_block.hpp"
#include "constraint/gpu/contact_generation.cuh"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/gpu/cuda_constraint_row_buffer.hpp"
#include "runtime/gpu/device_world.hpp"

#include <cstdint>
#include <vector>

namespace nuka::solver::gpu {

struct CudaConstraintSolverConfig {
    uint32_t velocity_iterations = 10;
    uint32_t position_iterations = 4;
    float slop = 0.005f;
    float baumgarte = 0.2f;
};

struct CudaConstraintSolverReport {
    uint32_t constraint_block_count = 0;
    uint32_t constraint_row_count = 0;
    uint32_t contact_constraint_count = 0;
    uint32_t joint_constraint_count = 0;
    uint32_t drive_constraint_count = 0;
    uint32_t velocity_iterations = 0;
    uint32_t position_iterations = 0;
    float max_position_error = 0.0f;
    runtime::gpu::CudaConstraintRowSchedulerReport row_scheduler_report;
};

class CudaConstraintSolverResult {
public:
    CudaConstraintSolverResult() = default;
    CudaConstraintSolverResult(uint32_t block_capacity,
                               phi::Buffer blocks,
                               phi::Buffer block_count,
                               phi::Buffer report);

    CudaConstraintSolverResult(const CudaConstraintSolverResult&) = delete;
    CudaConstraintSolverResult& operator=(const CudaConstraintSolverResult&) = delete;
    CudaConstraintSolverResult(CudaConstraintSolverResult&&) noexcept = default;
    CudaConstraintSolverResult& operator=(CudaConstraintSolverResult&&) noexcept = default;

    CudaConstraintSolverReport DownloadReport() const;
    std::vector<constraint::ConstraintBlock> DownloadBlocks() const;
    runtime::gpu::CudaConstraintRowBufferView ConstraintRowBuffer() const;

    const constraint::ConstraintBlock* DeviceBlocks() const;
    const uint32_t* DeviceBlockCount() const;

private:
    uint32_t block_capacity_ = 0;
    phi::Buffer blocks_;
    phi::Buffer block_count_;
    phi::Buffer report_;
};

CudaConstraintSolverResult SolveCudaConstraints(
    const phi::DeviceContext& context,
    runtime::gpu::DeviceWorld& device_world,
    const constraint::gpu::CudaContactResult* contacts,
    const CudaConstraintSolverConfig& config = {});
CudaConstraintSolverResult SolveCudaConstraints(
    runtime::gpu::DeviceWorld& device_world,
    const constraint::gpu::CudaContactResult* contacts,
    const CudaConstraintSolverConfig& config = {});

CudaConstraintSolverResult SolveCudaConstraintBlocks(
    const phi::DeviceContext& context,
    runtime::gpu::DeviceWorld& device_world,
    const std::vector<constraint::ConstraintBlock>& blocks,
    const CudaConstraintSolverConfig& config = {});
CudaConstraintSolverResult SolveCudaConstraintBlocks(
    runtime::gpu::DeviceWorld& device_world,
    const std::vector<constraint::ConstraintBlock>& blocks,
    const CudaConstraintSolverConfig& config = {});

} // namespace nuka::solver::gpu
