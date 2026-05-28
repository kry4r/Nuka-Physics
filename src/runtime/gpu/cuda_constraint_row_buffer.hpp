#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::gpu::CudaConstraintRowBuffer -- CUDA row-buffer view
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>

namespace nuka::runtime::gpu {

enum class CudaConstraintRowBufferKind : uint32_t {
    Unknown = 0u,
    ParticleRigidCoupling = 1u,
    RigidConstraintBlock = 2u,
    UniversalRowCsr = 3u,
};

enum class CudaConstraintRowLayout : uint32_t {
    Unknown = 0u,
    ParticleRigidCouplingSlot = 1u,
    ConstraintBlock = 2u,
    UniversalRowCsr = 3u,
};

enum class CudaConstraintRowScheduleMode : uint32_t {
    Unknown = 0u,
    OwnerSerialSweep = 1u,
    GlobalRowSweep = 2u,
    IslandColoredSweep = 3u,
};

struct CudaConstraintRowBufferView {
    CudaConstraintRowBufferKind kind = CudaConstraintRowBufferKind::Unknown;
    CudaConstraintRowLayout layout = CudaConstraintRowLayout::Unknown;
    CudaConstraintRowScheduleMode schedule_mode =
        CudaConstraintRowScheduleMode::Unknown;
    void* device_rows = nullptr;
    uint32_t row_count = 0u;
    uint32_t owner_count = 0u;
    uint32_t rows_per_owner = 0u;
    std::size_t row_stride_bytes = 0u;
};

struct CudaConstraintRowSchedulerConfig {
    uint32_t iterations = 1u;
    bool enable_warm_start = true;
    bool reduce_diagnostics = true;
};

struct CudaConstraintRowSchedulerReport {
    CudaConstraintRowBufferKind row_kind = CudaConstraintRowBufferKind::Unknown;
    CudaConstraintRowLayout row_layout = CudaConstraintRowLayout::Unknown;
    CudaConstraintRowScheduleMode schedule_mode =
        CudaConstraintRowScheduleMode::Unknown;
    uint32_t owner_count = 0u;
    uint32_t row_count = 0u;
    uint32_t rows_per_owner = 0u;
    std::size_t row_stride_bytes = 0u;
    uint32_t configured_iterations = 0u;
    uint32_t executed_iterations = 0u;
    uint32_t solver_launch_count = 0u;
    uint32_t diagnostic_launch_count = 0u;
    uint32_t active_row_count = 0u;
    uint32_t normal_impulse_count = 0u;
    uint32_t tangent_impulse_count = 0u;
    uint32_t diagnostic_slot_count = 0u;
    float normal_delta_impulse_magnitude = 0.0f;
    float tangent_delta_impulse_magnitude = 0.0f;
    float max_normal_delta_impulse = 0.0f;
    float max_tangent_delta_impulse = 0.0f;
    float max_residual = 0.0f;
};

struct CudaConstraintRowSchedulerIterationReport {
    uint32_t active_row_count = 0u;
    uint32_t normal_impulse_count = 0u;
    uint32_t tangent_impulse_count = 0u;
    uint32_t diagnostic_slot_count = 0u;
    float normal_delta_impulse_magnitude = 0.0f;
    float tangent_delta_impulse_magnitude = 0.0f;
    float max_normal_delta_impulse = 0.0f;
    float max_tangent_delta_impulse = 0.0f;
    float max_residual = 0.0f;
};

#if defined(__CUDACC__)
#define NUKA_CUDA_CONSTRAINT_ROW_HD __host__ __device__
#else
#define NUKA_CUDA_CONSTRAINT_ROW_HD
#endif

NUKA_CUDA_CONSTRAINT_ROW_HD inline void
SetCudaConstraintRowSchedulerMetadata(CudaConstraintRowSchedulerReport& report,
                                      const CudaConstraintRowBufferView& view,
                                      const CudaConstraintRowSchedulerConfig& config) {
    report.row_kind = view.kind;
    report.row_layout = view.layout;
    report.schedule_mode = view.schedule_mode;
    report.owner_count = view.owner_count;
    report.row_count = view.row_count;
    report.rows_per_owner = view.rows_per_owner;
    report.row_stride_bytes = view.row_stride_bytes;
    report.configured_iterations = config.iterations;
}

NUKA_CUDA_CONSTRAINT_ROW_HD inline CudaConstraintRowSchedulerReport
MakeCudaConstraintRowSchedulerReport(const CudaConstraintRowBufferView& view,
                                     const CudaConstraintRowSchedulerConfig& config) {
    CudaConstraintRowSchedulerReport report;
    SetCudaConstraintRowSchedulerMetadata(report, view, config);
    return report;
}

NUKA_CUDA_CONSTRAINT_ROW_HD inline void
AccumulateCudaConstraintRowSchedulerIteration(
    CudaConstraintRowSchedulerReport& report,
    const CudaConstraintRowSchedulerIterationReport& iteration) {
    report.active_row_count = iteration.active_row_count;
    report.normal_impulse_count += iteration.normal_impulse_count;
    report.tangent_impulse_count += iteration.tangent_impulse_count;
    report.normal_delta_impulse_magnitude += iteration.normal_delta_impulse_magnitude;
    report.tangent_delta_impulse_magnitude += iteration.tangent_delta_impulse_magnitude;
    report.max_normal_delta_impulse =
        report.max_normal_delta_impulse > iteration.max_normal_delta_impulse
            ? report.max_normal_delta_impulse
            : iteration.max_normal_delta_impulse;
    report.max_tangent_delta_impulse =
        report.max_tangent_delta_impulse > iteration.max_tangent_delta_impulse
            ? report.max_tangent_delta_impulse
            : iteration.max_tangent_delta_impulse;
    report.max_residual = report.max_residual > iteration.max_residual
        ? report.max_residual
        : iteration.max_residual;
    report.diagnostic_slot_count = iteration.diagnostic_slot_count;
}

#undef NUKA_CUDA_CONSTRAINT_ROW_HD

} // namespace nuka::runtime::gpu
