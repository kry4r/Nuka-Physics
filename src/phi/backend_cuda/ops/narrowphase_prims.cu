// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M5 GENERALIZED (pair-driven) NarrowphasePrimitives.
//
// The ADDITIVE pair-driven narrowphase: one thread per (env x candidate slot)
// consumes the candidate_pairs stream the broadphase (LbvhQueryPairs) emitted
// and dispatches by (kind_a, kind_b) over the amf:: analytic handler set
// (sphere/capsule/box/plane two-by-two) + sphere x hull, writing the SAME
// ucontact_* 4-point manifold layout AssembleRows consumes (so AssembleRows is
// UNCHANGED). The amf:: handlers are the HD-clean ones the union family already
// uses (the SAME math, byte-for-byte) — this op only changes the DRIVER
// (pair-stream, not slot-template).
//
// FAMILY SELECTION (the seam): OpNarrowphasePrimitives lives in contacts_foot.cu
// and dispatches on params.family (L1-b: the FUSED family was deleted):
//   kContactFamilyUnionCsr   -> LaunchUnionNarrowphase (contacts_union.cu, M4)
//   kContactFamilyPairDriven -> LaunchPairDrivenNarrowphase (THIS file, M5)
// The union slot-template path is the gate-pinned grasp/NkUnionN1 production
// path; the pair-driven path is purely additive and never touches it.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "collision/analytical_manifold.hpp"   // amf:: analytic handlers (HD)
#include "collision/convex_narrowphase.hpp"    // cvx:: GJK/EPA/face-clip (G5)
#include "collision/shape_kind.hpp"            // nuka::collision::ShapeKind (R2)
#include "math/transform.hpp"
#include "nk/solve/nk_row.hpp"                  // kUContactSideBody (side-kind tag)
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"

namespace nuka::phi::nkops {

namespace {

namespace amf = ::nuka::collision::amf;
namespace cvx = ::nuka::collision::cvx;
using ::nuka::constraint::ContactManifold;

// Shape kinds — R2: the ONE shared enum (collision/shape_kind.hpp). Local
// aliases keep the DispatchPair text identical. Plane is a +Z analytic plane at
// the body origin (frame.cy = world Z).
constexpr uint32_t kKindSphere  = ::nuka::collision::kShapeSphere;
constexpr uint32_t kKindCapsule = ::nuka::collision::kShapeCapsule;
constexpr uint32_t kKindBox     = ::nuka::collision::kShapeBox;
constexpr uint32_t kKindPlane   = ::nuka::collision::kShapePlane;
constexpr uint32_t kKindConvexHull = ::nuka::collision::kShapeConvexHull;

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

// Map a shape_table kind (ShapeKind) to a cvx::SupportProxy for the GJK/EPA
// fallback. For a primitive side (Box/Sphere/Capsule) it points at the in-hand
// amf::PrimParams (the world frame MakePrim already baked); for a ConvexHull side
// it builds a ConvexHullView over the cooked hull-vert pool, reusing the prim's
// baked frame (the same body world transform). Plane never reaches the fallback
// (every plane pair is analytic inline). Returns false if the kind is not a cvx
// shape (so the caller leaves the manifold empty).
// L-RECON-D: hull_verts is the GLOBAL concatenated pool; (hull_off, hull_cnt) is
// THIS shape's slice (lanes 10/11 of its shape_table row), so a hull side uses
// its own verts (pool + hull_off*3) rather than one global hull.
__device__ bool MakeSupportProxy(uint32_t kind, const amf::PrimParams& prim,
                                 const float* hull_verts, uint32_t hull_off,
                                 uint32_t hull_cnt,
                                 cvx::ConvexHullView* hull_store,
                                 cvx::SupportProxy* out) {
    switch (kind) {
        case kKindBox:     out->kind = cvx::SupportKind::Box;     out->prim = &prim; return true;
        case kKindSphere:  out->kind = cvx::SupportKind::Sphere;  out->prim = &prim; return true;
        case kKindCapsule: out->kind = cvx::SupportKind::Capsule; out->prim = &prim; return true;
        case kKindConvexHull:
            // Per-shape slice of the concatenated hull pool. The hull verts are
            // MESH-LOCAL; reuse the body's baked frame so the support scan
            // transforms them to world. warp_lockstep=false (see kernel note).
            hull_store->verts = hull_verts + static_cast<size_t>(hull_off) * 3u;
            hull_store->vcount = hull_cnt;
            hull_store->frame = prim.frame;
            hull_store->warp_lockstep = false;  // thread-per-slot kernel (not full-warp)
            out->kind = cvx::SupportKind::Hull;
            out->hull = hull_store;
            return true;
        default:
            return false;  // Plane / SdfMesh / heightfield: not a cvx fallback shape.
    }
}

// Dispatch the analytic handler for an ORDERED (kind_a, kind_b) pair. Returns a
// manifold whose normal is the separation dir for A. Cheap analytic pairs
// (sphere/plane/capsule via amf::) stay INLINE; every other convex pair
// (capsule-box, box-hull, hull-hull, ...) routes to the existing GPU GJK/EPA +
// face-clip narrowphase (cvx::ConvexNarrowphase, G5). Pairs with no cvx shape on
// a side (e.g. SDF mesh) leave the manifold empty (the SDF path covers them).
__device__ void DispatchPair(uint32_t ka, const amf::PrimParams& a,
                             uint32_t kb, const amf::PrimParams& b,
                             const float* hull_verts,
                             uint32_t hull_off_a, uint32_t hull_cnt_a,
                             uint32_t hull_off_b, uint32_t hull_cnt_b,
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
    // WP6 dog-dog leg/trunk: capsule x capsule analytic (segment-segment) — the
    // ANALYTIC pair (no EPA, so it avoids the v0.8 shallow-penetration hull debt).
    if (ka == kKindCapsule && kb == kKindCapsule){ amf::CapsuleCapsule(a, b, out); return; }
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
    // G5 FALLBACK: everything the analytic ladder did not handle (capsule-box,
    // box-hull, hull-hull, capsule-hull, ...) routes to the EXISTING device GJK +
    // EPA + face-clip narrowphase. Build a cvx::SupportProxy per side from the
    // amf::PrimParams already in hand; cvx writes the <=4-pt manifold (separation
    // dir for A) the emit loop already consumes. warp_lockstep=false (the kernel
    // is thread-per-slot, NOT full-warp converged).
    cvx::ConvexHullView ha, hb;
    cvx::SupportProxy A, B;
    if (MakeSupportProxy(ka, a, hull_verts, hull_off_a, hull_cnt_a, &ha, &A) &&
        MakeSupportProxy(kb, b, hull_verts, hull_off_b, hull_cnt_b, &hb, &B)) {
        cvx::ConvexNarrowphase(A, B, out);
    }
    // else: a non-cvx side (SDF mesh / plane-vs-non-prim) -> empty; the SDF path
    // covers it. (Plane pairs are all handled analytically above.)
}

// One thread per (env x candidate slot). Reads candidate_pairs[(env,slot)] =
// (a,b) template-local body rows, builds both prims from body_pose + the
// shape_table, dispatches, and writes the (<=4 pt) manifold into ucontact[gid].
__global__ void PairDrivenNarrowphaseKernel(
    const uint32_t* __restrict__ candidate_pairs,   // elem:2 per slot
    const uint32_t* __restrict__ pair_count,        // per env
    const float* __restrict__ shape_table,
    const math::Transform* __restrict__ body_pose,
    const float* __restrict__ hull_verts,           // cooked hull pool (G5)
    uint32_t /*hull_vert_count*/,                    // L-RECON-D: per-shape slice now in shape_table
    uint32_t gen,                                    // run/generation counter (C2)
    uint32_t env_count, uint32_t slot_stride, uint32_t rigid_slot_cap,
    uint32_t bodies_per_env,
    uint32_t* __restrict__ ucount,
    math::Vec3* __restrict__ upoint,
    math::Vec3* __restrict__ unormal,
    float* __restrict__ udepth,
    uint32_t* __restrict__ ucontact_a,              // C1/C2 collidable ids
    uint32_t* __restrict__ ucontact_b,
    uint32_t* __restrict__ ucontact_a_kind,         // per-side index-kind tag
    uint32_t* __restrict__ ucontact_b_kind,
    uint32_t* __restrict__ ucontact_gen,
    uint32_t* __restrict__ contact_count) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = env_count * slot_stride;
    if (gid >= total) return;
    const uint32_t env = gid / slot_stride;
    const uint32_t slot = gid - env * slot_stride;
    const uint32_t live = pair_count[env];

    ContactManifold m;
    m.Clear();
    uint32_t a = 0u, b = 0u;
    // The body<->body sub-range is [0, rigid_slot_cap); slots above belong to the
    // body<->particle narrowphase (== slot_stride when no particles -> identical).
    if (slot < live && slot < rigid_slot_cap) {
        a = candidate_pairs[static_cast<size_t>(gid) * 2u + 0u];
        b = candidate_pairs[static_cast<size_t>(gid) * 2u + 1u];
        const PrimShapeDev sa = LoadPrimShape(shape_table, a);
        const PrimShapeDev sb = LoadPrimShape(shape_table, b);
        const math::Transform xa = body_pose[env * bodies_per_env + a];
        const math::Transform xb = body_pose[env * bodies_per_env + b];
        const amf::PrimParams pa = MakePrim(sa, xa);
        const amf::PrimParams pb = MakePrim(sb, xb);
        // L-RECON-D: each side carries its OWN hull slice (lanes 10/11) into the
        // concatenated hull_verts pool; non-hull sides keep count 0 (no-op).
        DispatchPair(sa.kind, pa, sb.kind, pb, hull_verts,
                     sa.hull_vert_offset, sa.hull_vert_count,
                     sb.hull_vert_offset, sb.hull_vert_count, &m);
    }

    const uint32_t n = m.point_count;
    ucount[gid] = n;
    for (uint32_t i = 0u; i < 4u; ++i) {
        const size_t at = static_cast<size_t>(gid) * 4u + i;
        if (i < n) {
            upoint[at] = m.points[i].position;
            unormal[at] = m.points[i].normal;
            udepth[at] = m.points[i].penetration;
            // C2: retain the candidate (a,b) collidable ids per emitted point so
            // the assembly (S5, Phase 1B) can resolve both reaction sides. gen is
            // the per-step generation counter (for collide-once/reuse later).
            ucontact_a[at] = a;
            ucontact_b[at] = b;
            ucontact_a_kind[at] = ::nuka::nk::kUContactSideBody;
            ucontact_b_kind[at] = ::nuka::nk::kUContactSideBody;
            ucontact_gen[at] = gen;
        } else {
            upoint[at] = {0, 0, 0}; unormal[at] = {0, 0, 0}; udepth[at] = 0.0f;
            ucontact_a[at] = 0u; ucontact_b[at] = 0u; ucontact_gen[at] = 0u;
            ucontact_a_kind[at] = ::nuka::nk::kUContactSideBody;
            ucontact_b_kind[at] = ::nuka::nk::kUContactSideBody;
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
    // gen: a constant ACTIVE marker (1) this step. The ucontact_gen field is for a
    // future collide-once / reuse-across-substeps optimization; a per-step
    // monotonic counter would break two-run byte-identity, so 1A stamps a stable
    // constant for active points (0 for inactive slots). C2/the assembly only need
    // (a,b); gen carries no routing meaning yet.
    constexpr uint32_t kGen = 1u;
    LaunchCuda(PairDrivenNarrowphaseKernel, dim3(blocks), dim3(kBlock), 0u, stream,
               data.candidate_pairs, data.pair_count,
               static_cast<const float*>(model.shape_table),
               static_cast<const math::Transform*>(data.body_pose),
               static_cast<const float*>(model.hull_verts), p.hull_vert_count, kGen,
               p.env_count, p.union_slot_count, p.rigid_slot_cap, p.bodies_per_env,
               data.ucontact_count, data.ucontact_point, data.ucontact_normal,
               data.ucontact_depth, data.ucontact_a, data.ucontact_b,
               data.ucontact_a_kind, data.ucontact_b_kind,
               data.ucontact_gen, data.contact_count);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  // namespace nuka::phi::nkops
