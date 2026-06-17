// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — GENERAL narrowphase-dispatch ops.
//
// L1-b: the legacy FUSED foot-sphere-vs-ground kernels
// (DetectFootGroundContactsKernel / ComputeContactTangentBasisKernel) were
// DELETED here. L1-c: the UnionCsr narrowphase dispatch was DELETED too. This
// file now hosts ONLY the GENERAL narrowphase-dispatch ops that the single
// PairDriven family uses:
//   * OpNarrowphasePrimitives -> LaunchPairDrivenNarrowphase (PairDriven).
//   * OpContactTangentBasis    -> ComputeUnionContactTangentBasisKernel over the
//                                unified contact buffer (ucontact_*). (Despite the
//                                "Union" name this is the GENERAL tangent-basis op.)
// (The file/kernel names are kept as-is; a cosmetic rename is a later milestone.)
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "math/cuda_vec_ops.cuh"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"   // M5: pair-driven dispatch
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/op_schema.hpp"                      // NarrowphasePrimitivesParams

namespace nuka::phi {

namespace {

using namespace ::nuka::phi::nkops;

namespace mg = ::nuka::math::gpu;

__device__ math::Vec3 ScaleVec(math::Vec3 v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

__forceinline__ __device__ float Dot3(math::Vec3 a, math::Vec3 b) {
    return mg::Dot(a, b);
}

__forceinline__ __device__ math::Vec3 Cross3(math::Vec3 a, math::Vec3 b) {
    return mg::Cross(a, b);
}

__forceinline__ __device__ math::Vec3 NormalizeOrUpLocal(math::Vec3 normal) {
    return mg::NormalizeOrUp(normal);
}

// Branch-stable orthonormal tangent basis (t1,t2) for a unit normal n.
__device__ void TangentBasis(math::Vec3 n, math::Vec3* t1, math::Vec3* t2) {
    math::Vec3 reference = {1.0f, 0.0f, 0.0f};
    if (fabsf(n.x) > 0.9f) {
        reference = {0.0f, 1.0f, 0.0f};
    }
    const float proj = Dot3(reference, n);
    math::Vec3 tangent_a = {reference.x - proj * n.x,
                            reference.y - proj * n.y,
                            reference.z - proj * n.z};
    const float len_sq = Dot3(tangent_a, tangent_a);
    if (len_sq > 1.0e-12f) {
        tangent_a = ScaleVec(tangent_a, rsqrtf(len_sq));
    } else {
        tangent_a = {1.0f, 0.0f, 0.0f};
    }
    const math::Vec3 tangent_b = Cross3(n, tangent_a);
    *t1 = tangent_a;
    *t2 = tangent_b;
}

// C4: tangent basis over the UNIFIED contact buffer. One thread per contact slot;
// for each of the <=4 manifold points (ucontact_count[slot]) build (t1,t2) from
// ucontact_normal (elem:4). Inactive points (i >= count) get zero tangents.
// Deterministic (per-point, element-independent), so two runs are byte-identical.
__global__ void ComputeUnionContactTangentBasisKernel(
    const uint32_t* ucontact_count,
    const math::Vec3* ucontact_normal,   // elem:4 per slot
    uint32_t slot_count,
    math::Vec3* out_tangent1,            // elem:4 per slot
    math::Vec3* out_tangent2) {
    const uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= slot_count) return;
    const uint32_t n = ucontact_count[slot];
    for (uint32_t i = 0u; i < 4u; ++i) {
        const size_t at = static_cast<size_t>(slot) * 4u + i;
        if (i < n) {
            const math::Vec3 normal = NormalizeOrUpLocal(ucontact_normal[at]);
            math::Vec3 t1, t2;
            TangentBasis(normal, &t1, &t2);
            out_tangent1[at] = t1;
            out_tangent2[at] = t2;
        } else {
            out_tangent1[at] = {0.0f, 0.0f, 0.0f};
            out_tangent2[at] = {0.0f, 0.0f, 0.0f};
        }
    }
}

// --- op entry points ---------------------------------------------------------

Status OpNarrowphasePrimitives(const ModelView& model, const DataView& data,
                               const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const NarrowphasePrimitivesParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->family == kContactFamilyPairDriven) {
        // M5 generalized family: candidate_pairs -> analytic prim dispatch
        // (narrowphase_prims.cu). The ONE general narrowphase path.
        return LaunchPairDrivenNarrowphase(model, data, *p, stream);
    }
    // L1-b deleted the legacy FUSED foot-vs-ground detection; L1-c deleted the
    // UnionCsr narrowphase. Only the PairDriven family (handled above) dispatches.
    return Status::Ok;
}

Status OpContactTangentBasis(const ModelView& /*model*/, const DataView& data,
                             const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const ContactTangentBasisParams*>(params);
    if (p == nullptr) {
        return Status::Failed;
    }
    if (p->slot_count == 0u) {
        return Status::Ok;
    }
    constexpr uint32_t kBlockSize = 128u;
    const uint32_t block_count = (p->slot_count + kBlockSize - 1u) / kBlockSize;
    // L1-b: the FUSED contact-buffer tangent kernel was deleted. This op is now
    // enqueued ONLY for the PairDriven family (the pipeline emits the UnionCsr
    // family's tangent spokes inside AssembleRows), so it unconditionally builds
    // the tangents over the UNIFIED contact buffer (ucontact_*, elem:4).
    LaunchCuda(ComputeUnionContactTangentBasisKernel, dim3(block_count),
               dim3(kBlockSize), 0u, stream,
               static_cast<const uint32_t*>(data.ucontact_count),
               static_cast<const math::Vec3*>(data.ucontact_normal),
               p->slot_count, data.ucontact_tangent1, data.ucontact_tangent2);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

} // namespace

void RegisterNkContactsFootOps() {
    SetCudaOp(NkOp::NarrowphasePrimitives, &OpNarrowphasePrimitives);
    SetCudaOp(NkOp::ContactTangentBasis, &OpContactTangentBasis);
}

} // namespace nuka::phi
