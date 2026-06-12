// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M5 GENERALIZED (pair-driven) NarrowphasePrimitives.
//
// The ADDITIVE pair-driven narrowphase: one thread per (env x candidate slot)
// consumes the candidate_pairs stream the broadphase (LbvhQueryPairs) emitted
// and dispatches by (kind_a, kind_b) over the amf:: analytic handler set
// (sphere/capsule/box/plane two-by-two) + sphere x hull, writing the SAME
// ucontact_* 4-point manifold layout AssembleRows consumes (so AssembleRows is
// UNCHANGED). The amf:: handlers are the HD-clean ones the union / fused-foot
// families already use (the SAME math, byte-for-byte) — this op only changes
// the DRIVER (pair-stream, not slot-template).
//
// FAMILY SELECTION (the seam): OpNarrowphasePrimitives lives in contacts_foot.cu
// and dispatches on params.family:
//   kContactFamilyFusedFoot  -> DetectFootGroundContactsKernel (M3b, byte-exact)
//   kContactFamilyUnionCsr   -> LaunchUnionNarrowphase (contacts_union.cu, M4)
//   kContactFamilyPairDriven -> LaunchPairDrivenNarrowphase (THIS file, M5)
// The union slot-template path is the gate-pinned grasp/NkUnionN1 production
// path; the pair-driven path is purely additive and never touches it.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "collision/analytical_manifold.hpp"   // amf:: analytic handlers (HD)
#include "math/transform.hpp"
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"

namespace nuka::phi::nkops {

namespace {

namespace amf = ::nuka::collision::amf;
using ::nuka::constraint::ContactManifold;

// Shape kinds (mirror scene::CollisionShapeComponent::Kind + the broadphase
// ShapeDev). Plane is a +Z analytic plane at the body origin (frame.cy = world Z).
constexpr uint32_t kKindSphere = 0u;
constexpr uint32_t kKindCapsule = 1u;
constexpr uint32_t kKindBox = 2u;
constexpr uint32_t kKindPlane = 3u;

// Build the amf::PrimParams (baked world frame + extents) for one body row.
__device__ amf::PrimParams MakePrim(const PrimShapeDev& s,
                                     const math::Transform& xf) {
    amf::PrimParams p;
    // Bake the world frame from the body pose (the host BuildPrimFrame columns,
    // device replica via SdfRotate-equivalent — see PrimRotate).
    p.frame.cx = PrimRotate(xf.rotation, math::Vec3{1, 0, 0});
    p.frame.cy = PrimRotate(xf.rotation, math::Vec3{0, 1, 0});
    p.frame.cz = PrimRotate(xf.rotation, math::Vec3{0, 0, 1});
    p.frame.t = xf.position;
    p.radius = s.params[0];
    p.half_height = s.params[1];
    p.half_extents = {s.params[0], s.params[1], s.params[2]};
    return p;
}

// Dispatch the analytic handler for an ORDERED (kind_a, kind_b) pair. Returns a
// manifold whose normal is the separation dir for A. Unhandled pairs return an
// empty manifold (the SDF path / future handlers cover them).
__device__ void DispatchPair(uint32_t ka, const amf::PrimParams& a,
                             uint32_t kb, const amf::PrimParams& b,
                             ContactManifold* out) {
    out->Clear();
    // Canonicalize so the handler arg order matches the amf:: API (sphere first
    // for SphereBox/SpherePlane, box first for BoxPlane, etc.). The normal is
    // flipped if we swap to keep "separation dir for A".
    if (ka == kKindSphere && kb == kKindSphere) { amf::SphereSphere(a, b, out); return; }
    if (ka == kKindSphere && kb == kKindBox)    { amf::SphereBox(a, b, out); return; }
    if (ka == kKindSphere && kb == kKindPlane)  { amf::SpherePlane(a, b, out); return; }
    if (ka == kKindBox && kb == kKindPlane)     { amf::BoxPlane(a, b, out); return; }
    if (ka == kKindBox && kb == kKindBox)       { amf::BoxBox(a, b, out); return; }
    if (ka == kKindCapsule && kb == kKindPlane) { amf::CapsulePlane(a, b, out); return; }
    if (ka == kKindCapsule && kb == kKindSphere){ amf::CapsuleSphere(a, b, out); return; }
    // Swapped orders: run the canonical handler then flip the normal for A.
    auto flip = [&]() {
        for (uint32_t i = 0; i < out->point_count; ++i) {
            out->points[i].normal = math::Vec3{-out->points[i].normal.x,
                                               -out->points[i].normal.y,
                                               -out->points[i].normal.z};
        }
    };
    if (ka == kKindBox && kb == kKindSphere)    { amf::SphereBox(b, a, out); flip(); return; }
    if (ka == kKindPlane && kb == kKindSphere)  { amf::SpherePlane(b, a, out); flip(); return; }
    if (ka == kKindPlane && kb == kKindBox)     { amf::BoxPlane(b, a, out); flip(); return; }
    if (ka == kKindPlane && kb == kKindCapsule) { amf::CapsulePlane(b, a, out); flip(); return; }
    if (ka == kKindSphere && kb == kKindCapsule){ amf::CapsuleSphere(b, a, out); flip(); return; }
    // Unhandled (e.g. hull x hull) -> empty; the SDF path or M9 GJK retirement.
}

// One thread per (env x candidate slot). Reads candidate_pairs[(env,slot)] =
// (a,b) template-local body rows, builds both prims from body_pose + the
// shape_table, dispatches, and writes the (<=4 pt) manifold into ucontact[gid].
__global__ void PairDrivenNarrowphaseKernel(
    const uint32_t* __restrict__ candidate_pairs,   // elem:2 per slot
    const uint32_t* __restrict__ pair_count,        // per env
    const float* __restrict__ shape_table,
    const math::Transform* __restrict__ body_pose,
    uint32_t env_count, uint32_t slot_stride, uint32_t bodies_per_env,
    uint32_t* __restrict__ ucount,
    math::Vec3* __restrict__ upoint,
    math::Vec3* __restrict__ unormal,
    float* __restrict__ udepth,
    uint32_t* __restrict__ contact_count) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = env_count * slot_stride;
    if (gid >= total) return;
    const uint32_t env = gid / slot_stride;
    const uint32_t slot = gid - env * slot_stride;
    const uint32_t live = pair_count[env];

    ContactManifold m;
    m.Clear();
    if (slot < live && slot < slot_stride) {
        const uint32_t a = candidate_pairs[static_cast<size_t>(gid) * 2u + 0u];
        const uint32_t b = candidate_pairs[static_cast<size_t>(gid) * 2u + 1u];
        const PrimShapeDev sa = LoadPrimShape(shape_table, a);
        const PrimShapeDev sb = LoadPrimShape(shape_table, b);
        const math::Transform xa = body_pose[env * bodies_per_env + a];
        const math::Transform xb = body_pose[env * bodies_per_env + b];
        const amf::PrimParams pa = MakePrim(sa, xa);
        const amf::PrimParams pb = MakePrim(sb, xb);
        DispatchPair(sa.kind, pa, sb.kind, pb, &m);
    }

    const uint32_t n = m.point_count;
    ucount[gid] = n;
    for (uint32_t i = 0u; i < 4u; ++i) {
        const size_t at = static_cast<size_t>(gid) * 4u + i;
        if (i < n) {
            upoint[at] = m.points[i].position;
            unormal[at] = m.points[i].normal;
            udepth[at] = m.points[i].penetration;
        } else {
            upoint[at] = {0, 0, 0}; unormal[at] = {0, 0, 0}; udepth[at] = 0.0f;
        }
    }
    if (n > 0u && contact_count != nullptr) {
        atomicAdd(&contact_count[env], n);
    }
}

__global__ void ZeroEnvKernel(uint32_t* __restrict__ contact_count,
                              uint32_t* __restrict__ row_count, uint32_t n) {
    const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= n) return;
    contact_count[e] = 0u;
    row_count[e] = 0u;
}

}  // namespace

Status LaunchPairDrivenNarrowphase(const ModelView& model, const DataView& data,
                                   const NarrowphasePrimitivesParams& p,
                                   cudaStream_t stream) {
    if (p.env_count == 0u || p.union_slot_count == 0u) return Status::Ok;
    constexpr uint32_t kBlock = 128u;
    {
        const uint32_t b = (p.env_count + kBlock - 1u) / kBlock;
        LaunchCuda(ZeroEnvKernel, dim3(b), dim3(kBlock), 0u, stream,
                   data.contact_count, data.row_count, p.env_count);
    }
    const uint32_t total = p.env_count * p.union_slot_count;
    const uint32_t blocks = (total + kBlock - 1u) / kBlock;
    LaunchCuda(PairDrivenNarrowphaseKernel, dim3(blocks), dim3(kBlock), 0u, stream,
               data.candidate_pairs, data.pair_count,
               static_cast<const float*>(model.shape_table),
               static_cast<const math::Transform*>(data.body_pose),
               p.env_count, p.union_slot_count, p.bodies_per_env,
               data.ucontact_count, data.ucontact_point, data.ucontact_normal,
               data.ucontact_depth, data.contact_count);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  // namespace nuka::phi::nkops
