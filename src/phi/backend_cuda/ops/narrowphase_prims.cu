

#include <cuda_runtime.h>

#include "collision/analytical_manifold.hpp"   
#include "collision/convex_narrowphase.hpp"    
#include "collision/shape_kind.hpp"            
#include "math/transform.hpp"
#include "nk/contact/contact_identity.hpp"
#include "nk/solve/nk_row.hpp"                  
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"

namespace nuka::phi::nkops {

namespace {

namespace amf = ::nuka::collision::amf;
namespace cvx = ::nuka::collision::cvx;
constexpr uint32_t kBlock = 128u;
constexpr uint32_t kWarp = 32u;
using ::nuka::constraint::ContactManifold;

constexpr uint32_t kKindSphere  = ::nuka::collision::kShapeSphere;
constexpr uint32_t kKindCapsule = ::nuka::collision::kShapeCapsule;
constexpr uint32_t kKindBox     = ::nuka::collision::kShapeBox;
constexpr uint32_t kKindPlane   = ::nuka::collision::kShapePlane;
constexpr uint32_t kKindConvexHull = ::nuka::collision::kShapeConvexHull;

__device__ amf::PrimParams MakePrim(const PrimShapeDev& s,
                                     const math::Transform& xf) {
    amf::PrimParams p;

    p.frame.cx = PrimRotate(xf.rotation, math::Vec3{1, 0, 0});
    p.frame.cy = PrimRotate(xf.rotation, math::Vec3{0, 1, 0});
    p.frame.cz = PrimRotate(xf.rotation, math::Vec3{0, 0, 1});
    p.frame.t = xf.position;
    p.radius = s.params[0];
    p.half_height = s.params[1];
    p.half_extents = {s.params[0], s.params[1], s.params[2]};
    return p;
}

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

            hull_store->verts = hull_verts + static_cast<size_t>(hull_off) * 3u;
            hull_store->vcount = hull_cnt;
            hull_store->frame = prim.frame;
            hull_store->warp_lockstep = false;  
            out->kind = cvx::SupportKind::Hull;
            out->hull = hull_store;
            return true;
        default:
            return false;  
    }
}

__device__ bool DispatchAnalyticPair(uint32_t ka, const amf::PrimParams& a,
                             uint32_t kb, const amf::PrimParams& b,
                             ContactManifold* out) {
    out->Clear();

    if (ka == kKindSphere && kb == kKindSphere) { amf::SphereSphere(a, b, out); return true; }
    if (ka == kKindSphere && kb == kKindBox)    { amf::SphereBox(a, b, out); return true; }
    if (ka == kKindSphere && kb == kKindPlane)  { amf::SpherePlane(a, b, out); return true; }
    if (ka == kKindBox && kb == kKindPlane)     { amf::BoxPlane(a, b, out); return true; }
    if (ka == kKindBox && kb == kKindBox)       { amf::BoxBox(a, b, out); return true; }
    if (ka == kKindCapsule && kb == kKindPlane) { amf::CapsulePlane(a, b, out); return true; }
    if (ka == kKindCapsule && kb == kKindSphere){ amf::CapsuleSphere(a, b, out); return true; }

    if (ka == kKindCapsule && kb == kKindCapsule){ amf::CapsuleCapsule(a, b, out); return true; }
    
    auto flip = [&]() {
        for (uint32_t i = 0; i < out->point_count; ++i) {
            out->points[i].normal = math::Vec3{-out->points[i].normal.x,
                                               -out->points[i].normal.y,
                                               -out->points[i].normal.z};
        }
    };
    if (ka == kKindBox && kb == kKindSphere)    { amf::SphereBox(b, a, out); flip(); return true; }
    if (ka == kKindPlane && kb == kKindSphere)  { amf::SpherePlane(b, a, out); flip(); return true; }
    if (ka == kKindPlane && kb == kKindBox)     { amf::BoxPlane(b, a, out); flip(); return true; }
    if (ka == kKindPlane && kb == kKindCapsule) { amf::CapsulePlane(b, a, out); flip(); return true; }
    if (ka == kKindSphere && kb == kKindCapsule){ amf::CapsuleSphere(b, a, out); flip(); return true; }

    return false;
}

__device__ bool IsConvexKind(uint32_t kind) {
    return kind == kKindSphere || kind == kKindCapsule || kind == kKindBox || kind == kKindConvexHull;
}

__device__ void DispatchConvexPair(uint32_t ka, const amf::PrimParams& a,
                                  uint32_t kb, const amf::PrimParams& b,
                                  const float* hull_verts,
                                  uint32_t hull_off_a, uint32_t hull_cnt_a,
                                  uint32_t hull_off_b, uint32_t hull_cnt_b,
                                  ContactManifold* out, cvx::ConvexWorkspace& workspace) {
    cvx::ConvexHullView ha, hb;
    cvx::SupportProxy A, B;
    if (MakeSupportProxy(ka, a, hull_verts, hull_off_a, hull_cnt_a, &ha, &A) &&
        MakeSupportProxy(kb, b, hull_verts, hull_off_b, hull_cnt_b, &hb, &B)) {
        cvx::ConvexNarrowphase(A, B, out, workspace);
    }

}

__global__ void PairDrivenNarrowphaseKernel(
    const uint32_t* __restrict__ candidate_pairs,   
    const uint32_t* __restrict__ pair_count,        
    const float* __restrict__ shape_table,
    const uint8_t* __restrict__ mesh_contact_mode,
    const math::Transform* __restrict__ body_pose,
    const float* __restrict__ hull_verts,           
    uint32_t /*hull_vert_count*/,                    
    uint32_t gen,                                    
    uint32_t env_count, uint32_t slot_stride, uint32_t rigid_slot_cap,
    uint32_t bodies_per_env,
    uint32_t* __restrict__ ucount,
    math::Vec3* __restrict__ upoint,
    math::Vec3* __restrict__ unormal,
    float* __restrict__ udepth,
    uint32_t* __restrict__ ucontact_a,              
    uint32_t* __restrict__ ucontact_b,
    uint32_t* __restrict__ ucontact_a_kind,         
    uint32_t* __restrict__ ucontact_b_kind,
    uint32_t* __restrict__ ucontact_gen,
    uint64_t* __restrict__ ucontact_id_pair,
    uint64_t* __restrict__ ucontact_id_feature,
    uint32_t* __restrict__ contact_count,
    uint32_t* __restrict__ mesh_ogc_pair_count,
    uint32_t* __restrict__ mesh_sdf_pair_count) {
    __shared__ __align__(16) unsigned char storage[(kBlock / kWarp) * sizeof(cvx::ConvexWorkspace)];
    auto& workspace = reinterpret_cast<cvx::ConvexWorkspace*>(storage)[threadIdx.x / kWarp];
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t mask = __ballot_sync(0xffffffffu, item < env_count * rigid_slot_cap);
    
    if (item >= env_count * rigid_slot_cap) return;
    const uint32_t env = item / rigid_slot_cap;
    const uint32_t slot = item - env * rigid_slot_cap;
    const uint32_t gid = env * slot_stride + slot;
    const uint32_t live = pair_count[env];

    ContactManifold m;
    m.Clear();
    uint32_t a = 0u, b = 0u;
    PrimShapeDev sa, sb;
    bool convex = false;

    if (slot < live && slot < rigid_slot_cap) {
        a = candidate_pairs[static_cast<size_t>(gid) * 2u + 0u];
        b = candidate_pairs[static_cast<size_t>(gid) * 2u + 1u];
        sa = LoadPrimShape(shape_table, a);
        sb = LoadPrimShape(shape_table, b);
        const auto route = RouteMeshContact(sa.kind, sb.kind, mesh_contact_mode[a],
                                            mesh_contact_mode[b]);
        if (route == MeshContactRoute::Ogc) atomicAdd(&mesh_ogc_pair_count[env], 1u);
        if (route == MeshContactRoute::Sdf) atomicAdd(&mesh_sdf_pair_count[env], 1u);
        if (route == MeshContactRoute::Analytic) {
            const math::Transform xa = body_pose[env * bodies_per_env + a];
            const math::Transform xb = body_pose[env * bodies_per_env + b];
            const amf::PrimParams pa = MakePrim(sa, xa);
            const amf::PrimParams pb = MakePrim(sb, xb);
            convex = !DispatchAnalyticPair(sa.kind, pa, sb.kind, pb, &m) &&
                     IsConvexKind(sa.kind) && IsConvexKind(sb.kind);
        }
    }

    uint32_t pending = __ballot_sync(mask, convex);
    while (pending != 0u) {
        const uint32_t lane = static_cast<uint32_t>(__ffs(pending) - 1);
        if (threadIdx.x % kWarp == lane) {
            const amf::PrimParams pa = MakePrim(sa, body_pose[env * bodies_per_env + a]);
            const amf::PrimParams pb = MakePrim(sb, body_pose[env * bodies_per_env + b]);
            DispatchConvexPair(sa.kind, pa, sb.kind, pb, hull_verts,
                               sa.hull_vert_offset, sa.hull_vert_count,
                               sb.hull_vert_offset, sb.hull_vert_count, &m, workspace);
        }
        __syncwarp(mask);
        pending &= pending - 1u;
    }

    const uint32_t n = m.point_count;
    ucount[gid] = n;

    for (uint32_t i = 0u; i < 4u; ++i) {
        const size_t at = static_cast<size_t>(gid) * 4u + i;
        if (i < n) {
            upoint[at] = m.points[i].position;
            unormal[at] = m.points[i].normal;
            udepth[at] = m.points[i].penetration;

            ucontact_a[at] = a;
            ucontact_b[at] = b;
            ucontact_a_kind[at] = ::nuka::nk::kUContactSideBody;
            ucontact_b_kind[at] = ::nuka::nk::kUContactSideBody;
            ucontact_gen[at] = gen;
            ::nuka::nk::CanonicalContactDescriptor descriptor;
            descriptor.a.handle = a;
            descriptor.b.handle = b;
            descriptor.local_point_a = PrimInverseTransformPoint(
                body_pose[env * bodies_per_env + a], m.points[i].position);
            descriptor.local_point_b = PrimInverseTransformPoint(
                body_pose[env * bodies_per_env + b], m.points[i].position);
            descriptor.normal = m.points[i].normal;
            const uint32_t feature = m.points[i].stable_key == 0u
                ? ::nuka::nk::kContactFeatureUnavailable
                : static_cast<uint32_t>(m.points[i].stable_key);
            descriptor.feature_a = feature;
            descriptor.feature_b = feature;
            descriptor.manifold_slot = i;
            const ::nuka::nk::ContactId id = ::nuka::nk::MakeContactId(descriptor);
            ucontact_id_pair[at] = id.pair;
            ucontact_id_feature[at] = id.feature;
        }
    }
    if (n > 0u && contact_count != nullptr) {
        atomicAdd(&contact_count[env], n);
    }
}

__global__ void ZeroEnvKernel(uint32_t* __restrict__ contact_count,
                              uint32_t* __restrict__ row_count,
                              uint32_t* __restrict__ mesh_ogc_pair_count,
                              uint32_t* __restrict__ mesh_sdf_pair_count, uint32_t n,
                              uint32_t* __restrict__ ucount, uint32_t slot_stride,
                              uint32_t rigid_slots) {
    const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t compact = slot_stride - rigid_slots;
    if (e < n * compact) {
        const uint32_t env = e / compact;
        ucount[env * slot_stride + rigid_slots + (e - env * compact)] = 0u;
    }
    if (e >= n) return;
    contact_count[e] = 0u;
    row_count[e] = 0u;
    mesh_ogc_pair_count[e] = 0u;
    mesh_sdf_pair_count[e] = 0u;
}

}  

Status LaunchPairDrivenNarrowphase(const ModelView& model, const DataView& data,
                                   const NarrowphasePrimitivesParams& p,
                                   cudaStream_t stream) {
    if (p.env_count == 0u || p.union_slot_count == 0u) {
        return Status::Ok;
    }
    const uint32_t rigid_slots =
        p.rigid_slot_cap < p.union_slot_count ? p.rigid_slot_cap : p.union_slot_count;
    {
        const uint32_t zero = p.env_count * (p.union_slot_count - rigid_slots);
        const uint32_t b = ((zero > p.env_count ? zero : p.env_count) + kBlock - 1u) / kBlock;
        LaunchCuda(ZeroEnvKernel, dim3(b), dim3(kBlock), 0u, stream,
                   data.contact_count, data.row_count, data.mesh_ogc_pair_count,
                   data.mesh_sdf_pair_count, p.env_count, data.ucontact_count,
                   p.union_slot_count, rigid_slots);
    }
    if (rigid_slots == 0u) return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
    const uint32_t total = p.env_count * rigid_slots;
    const uint32_t blocks = (total + kBlock - 1u) / kBlock;

    constexpr uint32_t kGen = 1u;
    LaunchCuda(PairDrivenNarrowphaseKernel, dim3(blocks), dim3(kBlock), 0u, stream,
               data.candidate_pairs, data.pair_count,
               static_cast<const float*>(model.shape_table), model.mesh_contact_mode,
               static_cast<const math::Transform*>(data.body_pose),
               static_cast<const float*>(model.hull_verts), p.hull_vert_count, kGen,
               p.env_count, p.union_slot_count, rigid_slots, p.bodies_per_env,
               data.ucontact_count, data.ucontact_point, data.ucontact_normal,
               data.ucontact_depth, data.ucontact_a, data.ucontact_b,
               data.ucontact_a_kind, data.ucontact_b_kind,
               data.ucontact_gen, data.ucontact_id_pair,
               data.ucontact_id_feature, data.contact_count,
               data.mesh_ogc_pair_count, data.mesh_sdf_pair_count);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  
