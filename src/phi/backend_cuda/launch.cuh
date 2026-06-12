#pragma once
// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — the SINGLE kernel-launch wrapper.
//
// `<<<...>>>` triple-chevron launches in the v2 world live ONLY behind this
// wrapper. Op implementations (phi/backend_cuda/ops/**) and the backend call
// LaunchCuda(...) instead of writing chevrons directly; this keeps every launch
// stream-aware (mandatory for the CUDA-graph stream-capture plan path) and
// gives one audit point for launch policy.
//
// Usage:
//   LaunchCuda(my_kernel, grid, block, /*smem=*/0, stream, argA, argB, n);
// expands to:
//   my_kernel<<<grid, block, smem, stream>>>(argA, argB, n);
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

namespace nuka::phi {

template <typename Kernel, typename... Args>
inline void LaunchCuda(Kernel kernel,
                       dim3 grid,
                       dim3 block,
                       size_t shared_mem,
                       cudaStream_t stream,
                       Args... args) {
    kernel<<<grid, block, shared_mem, stream>>>(args...);
}

} // namespace nuka::phi
