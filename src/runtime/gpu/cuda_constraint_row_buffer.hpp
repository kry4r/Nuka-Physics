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
};

struct CudaConstraintRowBufferView {
    CudaConstraintRowBufferKind kind = CudaConstraintRowBufferKind::Unknown;
    void* device_rows = nullptr;
    uint32_t row_count = 0u;
    uint32_t owner_count = 0u;
    uint32_t rows_per_owner = 0u;
    std::size_t row_stride_bytes = 0u;
};

} // namespace nuka::runtime::gpu
