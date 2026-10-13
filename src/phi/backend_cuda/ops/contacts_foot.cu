// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — GENERAL narrowphase-dispatch ops.
//
// L1-b: the legacy FUSED foot-sphere-vs-ground kernels
// (DetectFootGroundContactsKernel / ComputeContactTangentBasisKernel) were
// DELETED here. L1-c: the UnionCsr narrowphase dispatch was DELETED too. This
// file now hosts ONLY the GENERAL narrowphase-dispatch ops that the single
// PairDriven family uses:
//   * OpNarrowphasePrimitives -> LaunchPairDrivenNarrowphase (PairDriven).
// (The file/kernel names are kept as-is; a cosmetic rename is a later milestone.)
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>
#include <cstdio>

#include "math/cuda_vec_ops.cuh"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"   // M5: pair-driven dispatch
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/op_schema.hpp"                      // NarrowphasePrimitivesParams

namespace nuka::phi {

namespace {

using namespace ::nuka::phi::nkops;

// --- op entry points ---------------------------------------------------------

Status OpNarrowphasePrimitives(const ModelView& model, const DataView& data,
                               const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const NarrowphasePrimitivesParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->family == kContactFamilyPairDriven) {
        return LaunchPairDrivenNarrowphase(model, data, *p, stream);
    }
    // L1-b deleted the legacy FUSED foot-vs-ground detection; L1-c deleted the
    // UnionCsr narrowphase. Only the PairDriven family (handled above) dispatches.
    return Status::Ok;
}

} // namespace

void RegisterNkContactsFootOps() {
    SetCudaOp(NkOp::NarrowphasePrimitives, &OpNarrowphasePrimitives);
}

} // namespace nuka::phi
