#pragma once
// ---------------------------------------------------------------------------
// nuka::phi::backend_cuda interop scatter -- CUDA backend declarations (M11
// §3.E INT-1/INT-2/INT-7). This header is part of the CUDA backend layer
// (phi/backend_cuda/interop) and MAY name CUDA (it is OUTSIDE the engine
// zero-CUDA red-line, which covers only src/nk + src/scene + src/render +
// src/runtime/app). It is included ONLY by cuda_vk_scatter.cu; the app/render
// layers reach the scatter exclusively through the CUDA-FREE interface
// phi/interop_scatter.hpp (CreateCudaVkScatter factory).
//
// The concrete CudaVkScatter implementing phi::InteropScatterI lives in
// cuda_vk_scatter.cu (anon namespace), mirroring rt_backend_cuda.cpp. This header
// exists so the kernel launcher declaration + the import helpers are documented in
// one place; the kernel itself is launched through phi/backend_cuda/launch.cuh
// (the sole <<<>>> wrapper).
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "phi/interop_scatter.hpp"

namespace nuka::phi {

// Device kernel launcher (defined in cuda_vk_scatter.cu): writes, for each
// instance row, the column-major 4x4 object->world matrix into `out_mat4`
// (row*16 floats) BIT-IDENTICALLY to HostDownloadPublisher's host composition
// (R14). One thread per instance row (one writer per row -> no float atomics).
// Reads the SAME FK device field buffers DownloadField reads.
void LaunchScatterTransforms(const ScatterFkSource& fk,
                             const InstanceScatterRow* d_rows,
                             uint32_t instance_count,
                             float* out_mat4,
                             cudaStream_t stream);

}  // namespace nuka::phi
