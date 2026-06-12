// ---------------------------------------------------------------------------
// PHI v2 smoke gate — test-side op implementation.
//
// A trivial vector-add op c[i] = a[i] + b[i], registered into a spare NkOp slot
// (NkOp::ApplyDrives — no real op is wired in M1, so any slot is free) via the
// runtime SetCudaOp hook. It launches through phi::LaunchCuda (the single
// chevron wrapper) on the stream the backend hands it — so it is graph-capture
// safe and exercises BOTH the dispatch path and the plan (CUDA-graph) path.
//
// The op never dereferences ModelView / DataView (the M1 contract); it reads
// only its POD params (the 3 device pointers + n) and the stream.
// ---------------------------------------------------------------------------

#include "phi/backend.hpp"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/registry.cuh"

#include "phi2_smoke_ops.hpp"

namespace nuka::phi_smoke {

namespace {

__global__ void VecAddKernel(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

::nuka::phi::Status VecAddOp(const ::nuka::phi::ModelView& /*model*/,
                             const ::nuka::phi::DataView& /*data*/,
                             const void* params,
                             cudaStream_t stream) {
    const auto* p = static_cast<const VecAddParams*>(params);
    if (p == nullptr || p->n <= 0) {
        return ::nuka::phi::Status::Failed;
    }
    const int block = 256;
    const int grid = (p->n + block - 1) / block;
    ::nuka::phi::LaunchCuda(VecAddKernel, dim3(grid), dim3(block),
                            /*shared_mem=*/0, stream,
                            p->a, p->b, p->c, p->n);
    return ::nuka::phi::Status::Ok;
}

} // namespace

void RegisterVecAddOp() {
    ::nuka::phi::SetCudaOp(::nuka::phi::NkOp::ApplyDrives, &VecAddOp);
}

} // namespace nuka::phi_smoke
