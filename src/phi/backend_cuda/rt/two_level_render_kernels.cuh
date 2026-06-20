#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- the SHARED device structs + nested-traversal closest-hit / any-hit
// fns for the two-level (TLAS/BLAS) tracer, included by BOTH the GOLDEN TU
// (two_level_render.cu, instantiated Real=double, --fmad=false) AND the BEAUTY
// TU (two_level_render_beauty.cu, instantiated Real=float, --fmad=true). ONE
// general path: the FP32/FP64 choice is the Real template parameter (default
// double => every golden caller + host oracle byte-unchanged), not a branch.
//
// The device fns mirror the algorithm in two_level_render.hpp: TlasLeaf
// transforms the world ray into each instance's local frame, runs an INNER
// TraverseRay over that instance's once-built BLAS, and updates the SAME best_t /
// best_prim by reference. AnyOccluder is the visibility variant (first hit wins).
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "collision/lbvh_node.cuh"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend_cuda/rt/bvh_traverse_impl.cuh"
#include "phi/backend_cuda/rt/instance_transform.cuh"
#include "phi/backend_cuda/rt/intersect_primitives.cuh"
#include "phi/backend_cuda/rt/prim_id.cuh"
#include "phi/backend_cuda/rt/ray_box.cuh"
#include "rt/camera.hpp"
#include "rt/material.hpp"
#include "rt/scene_render.hpp"  // PrimKind / SparseSdfDevice (Light is in material.hpp)

#include <cuda_runtime.h>

#include <cstdint>

namespace nuka::rt {

using ::nuka::collision::gpu::LbvhNode;
using ::nuka::math::Transform;
using ::nuka::math::Vec3;

// Flat per-primitive record for ONE mesh's BLAS, indexed by the BLAS leaf's
// original (LOCAL) prim id. The material is per-instance, so it is dropped here.
struct DevPrim {
    uint32_t kind;  // PrimKind
    uint32_t sub;   // index into the per-kind array
};

// Device view of ONE mesh's BLAS geometry (raw pointers into uploaded buffers).
struct DevBlas {
    const DevPrim* prims = nullptr;
    uint32_t prim_count = 0u;

    const Vec3* tri_v0 = nullptr;
    const Vec3* tri_v1 = nullptr;
    const Vec3* tri_v2 = nullptr;

    const Vec3* sph_center = nullptr;
    const float* sph_radius = nullptr;

    const runtime::sdf::SparseSdfDevice* sdf_hdr = nullptr;
    const collision::AABB* sdf_aabb = nullptr;
    const float* sdf_eps = nullptr;
    const int* sdf_iters = nullptr;
};

// One placed instance, device-side: its rigid transform, a view of its (shared)
// BLAS geometry + retained tree, and the global ids the leaf packs / shades.
struct DevInstance {
    Transform transform;
    const LbvhNode* blas_nodes = nullptr;
    uint32_t blas_leaf_count = 0u;
    DevBlas blas;
    uint32_t instance_id = 0u;
    uint32_t material_id = 0u;
};

// Device-side mirror of rt::BeautyOptions (trivially-copyable, passed by value
// into the beauty kernel). Pure render parameters; no host-only types.
struct BeautyParams {
    uint32_t shadow_rays;
    float sun_angular_radius;
    uint32_t gi_bounces;
    uint32_t ao_samples;
    float ao_radius;
    Vec3 sky_top, sky_bottom, sky_ground, fog_color;
    float fog_density;
    float sky_intensity;
};

// AOV destination pointers for ONE frame (raw device pointers). null skips that
// channel; shared by the host-download path and the device-resident entries.
struct AovTarget {
    float* color = nullptr;
    float* depth = nullptr;
    float* normal = nullptr;
    float* albedo = nullptr;
    float* uv = nullptr;
    uint32_t* prim = nullptr;
};

// Intersect one LOCAL primitive of a BLAS (closest-hit pass): returns t only.
// LOCAL-frame ray in; rigid -> local t == world t (see instance_transform.cuh).
template <typename Real = double>
__device__ __forceinline__ bool IntersectBlasPrimT(const DevBlas& b,
                                                   uint32_t prim,
                                                   const Vec3& origin,
                                                   const Vec3& dir,
                                                   float t_min,
                                                   float* out_t) {
    const DevPrim p = b.prims[prim];
    if (p.kind == static_cast<uint32_t>(PrimKind::Triangle)) {
        float u, v;
        Vec3 n;
        return RayTriangleIntersect<Real>(origin, dir, b.tri_v0[p.sub], b.tri_v1[p.sub],
                                          b.tri_v2[p.sub], t_min, out_t, &u, &v, &n);
    } else if (p.kind == static_cast<uint32_t>(PrimKind::Sphere)) {
        Vec3 n;
        float u, v;
        return RaySphereIntersect<Real>(origin, dir, b.sph_center[p.sub],
                                        b.sph_radius[p.sub], t_min, out_t, &n, &u, &v);
    } else {  // Sdf
        Vec3 n;
        return RaySdfSphereTrace<Real>(origin, dir, b.sdf_hdr[p.sub], b.sdf_aabb[p.sub],
                                       t_min, b.sdf_eps[p.sub], b.sdf_iters[p.sub],
                                       out_t, &n);
    }
}

// BLAS leaf functor (the INNER closest-hit traversal). On a win it writes
// PackPrimId(instance_id, local_prim) so the global tie-break is a TOTAL ORDER.
template <typename Real = double>
struct BlasLeaf {
    const LbvhNode* __restrict__ nodes;
    DevBlas blas;
    uint32_t instance_id;
    float t_min;
    __device__ void operator()(int32_t leaf_node,
                               const Vec3& origin,
                               const Vec3& dir,
                               float* best_t,
                               uint32_t* best_prim) const {
        const uint32_t local_prim = static_cast<uint32_t>(nodes[leaf_node].left);
        float t;
        if (IntersectBlasPrimT<Real>(blas, local_prim, origin, dir, t_min, &t)) {
            if (t >= t_min) {
                RtClosestHitUpdate(t, PackPrimId(instance_id, local_prim), best_t,
                                   best_prim);
            }
        }
    }
};

// TLAS leaf functor: transform the WORLD ray into the instance-local frame, then
// run the INNER closest-hit TraverseRay over that instance's BLAS, updating the
// SAME best_t / best_prim by reference. max_dist prunes far BLAS subtrees.
template <typename Real = double>
struct TlasLeaf {
    const LbvhNode* __restrict__ tlas_nodes;
    const DevInstance* __restrict__ instances;
    float t_min;
    float max_dist;
    __device__ void operator()(int32_t leaf_node,
                               const Vec3& origin,
                               const Vec3& dir,
                               float* best_t,
                               uint32_t* best_prim) const {
        const uint32_t inst = static_cast<uint32_t>(tlas_nodes[leaf_node].left);
        const DevInstance& I = instances[inst];
        if (I.blas_leaf_count == 0u) {
            return;
        }
        Vec3 o_l, d_l;
        TransformRayToLocal(I.transform, origin, dir, &o_l, &d_l);

        BlasLeaf<Real> leaf{I.blas_nodes, I.blas, I.instance_id, t_min};
        if (I.blas_leaf_count == 1u) {
            leaf(0, o_l, d_l, best_t, best_prim);
        } else {
            TraverseRay<Real>(I.blas_nodes, I.blas_leaf_count - 1u, o_l, d_l, best_t,
                              best_prim, leaf, max_dist);
        }
    }
};

// Run the full TWO-LEVEL closest hit (primary or shadow). best_t carries the
// comparable world t; best_prim carries the packed (instance, local) id. max_dist
// (default +inf => golden byte-unchanged) prunes children whose entry t >= it.
template <typename Real = double>
__device__ __forceinline__ void ClosestHit(const LbvhNode* __restrict__ tlas_nodes,
                                           uint32_t tlas_leaf_count,
                                           const DevInstance* __restrict__ instances,
                                           const Vec3& origin,
                                           const Vec3& dir,
                                           float t_min,
                                           float* best_t,
                                           uint32_t* best_prim,
                                           float max_dist = RtMissDepth()) {
    *best_t = RtMissDepth();
    *best_prim = kNoPrim;
    TlasLeaf<Real> leaf{tlas_nodes, instances, t_min, max_dist};
    if (tlas_leaf_count == 1u) {
        leaf(0, origin, dir, best_t, best_prim);
    } else if (tlas_leaf_count >= 2u) {
        TraverseRay<Real>(tlas_nodes, tlas_leaf_count - 1u, origin, dir, best_t,
                          best_prim, leaf, max_dist);
    }
}

// BLAS any-hit functor: returns true at the FIRST occluder prim in [t_min, dist).
template <typename Real = double>
struct BlasAnyHit {
    const LbvhNode* __restrict__ nodes;
    DevBlas blas;
    float t_min;
    float dist;
    __device__ bool operator()(int32_t leaf_node,
                               const Vec3& origin,
                               const Vec3& dir) const {
        const uint32_t local_prim = static_cast<uint32_t>(nodes[leaf_node].left);
        float t;
        if (IntersectBlasPrimT<Real>(blas, local_prim, origin, dir, t_min, &t)) {
            return t >= t_min && t < dist;
        }
        return false;
    }
};

// TLAS any-hit functor: transform the ray into the instance-local frame and run
// the INNER any-hit traversal over the BLAS; true iff this instance occludes.
template <typename Real = double>
struct TlasAnyHit {
    const LbvhNode* __restrict__ tlas_nodes;
    const DevInstance* __restrict__ instances;
    float t_min;
    float dist;
    __device__ bool operator()(int32_t leaf_node,
                               const Vec3& origin,
                               const Vec3& dir) const {
        const uint32_t inst = static_cast<uint32_t>(tlas_nodes[leaf_node].left);
        const DevInstance& I = instances[inst];
        if (I.blas_leaf_count == 0u) {
            return false;
        }
        Vec3 o_l, d_l;
        TransformRayToLocal(I.transform, origin, dir, &o_l, &d_l);
        BlasAnyHit<Real> leaf{I.blas_nodes, I.blas, t_min, dist};
        if (I.blas_leaf_count == 1u) {
            return leaf(0, o_l, d_l);
        }
        return TraverseRayAnyHit<Real>(I.blas_nodes, I.blas_leaf_count - 1u, o_l, d_l,
                                       leaf, dist);
    }
};

// Two-level VISIBILITY query for pure-shadow rays: true iff ANY prim is hit in
// [t_min, dist). Order-invariant, so the first-hit early-out is exact. Used by
// the beauty soft-shadow + GI-shadow rays (no AOV reconstruct needed).
template <typename Real = double>
__device__ __forceinline__ bool AnyOccluder(const LbvhNode* __restrict__ tlas_nodes,
                                            uint32_t tlas_leaf_count,
                                            const DevInstance* __restrict__ instances,
                                            const Vec3& origin,
                                            const Vec3& dir,
                                            float t_min,
                                            float dist) {
    TlasAnyHit<Real> leaf{tlas_nodes, instances, t_min, dist};
    if (tlas_leaf_count == 1u) {
        return leaf(0, origin, dir);
    } else if (tlas_leaf_count >= 2u) {
        return TraverseRayAnyHit<Real>(tlas_nodes, tlas_leaf_count - 1u, origin, dir,
                                       leaf, dist);
    }
    return false;
}

// Reconstruct surface attributes (world normal, uv) for the WINNING (instance,
// local) prim: re-transform the ray to local, re-intersect for the local
// normal/uv, then rotate the normal back to world via QuatRotate (rigid only).
template <typename Real = double>
__device__ __forceinline__ void ReconstructHit(const DevInstance* __restrict__ instances,
                                               uint32_t packed_prim,
                                               const Vec3& origin,
                                               const Vec3& dir,
                                               Vec3* out_world_normal,
                                               float* out_u,
                                               float* out_v) {
    uint32_t inst, local_prim;
    UnpackPrimId(packed_prim, &inst, &local_prim);
    const DevInstance& I = instances[inst];
    const DevBlas& b = I.blas;

    Vec3 o_l, d_l;
    TransformRayToLocal(I.transform, origin, dir, &o_l, &d_l);

    const DevPrim p = b.prims[local_prim];
    Vec3 n_local{0.0f, 0.0f, 0.0f};
    float u = 0.0f, v = 0.0f;
    if (p.kind == static_cast<uint32_t>(PrimKind::Triangle)) {
        float tt;
        RayTriangleIntersect<Real>(o_l, d_l, b.tri_v0[p.sub], b.tri_v1[p.sub],
                                   b.tri_v2[p.sub], 0.0f, &tt, &u, &v, &n_local);
    } else if (p.kind == static_cast<uint32_t>(PrimKind::Sphere)) {
        float tt;
        RaySphereIntersect<Real>(o_l, d_l, b.sph_center[p.sub], b.sph_radius[p.sub],
                                 0.0f, &tt, &n_local, &u, &v);
    } else {  // Sdf
        float tt;
        RaySdfSphereTrace<Real>(o_l, d_l, b.sdf_hdr[p.sub], b.sdf_aabb[p.sub], 0.0f,
                                b.sdf_eps[p.sub], b.sdf_iters[p.sub], &tt, &n_local);
        u = 0.0f;
        v = 0.0f;
    }
    *out_world_normal = QuatRotate(I.transform.rotation, n_local);
    *out_u = u;
    *out_v = v;
}

// Launch the FP32 beauty kernel (defined in two_level_render_beauty.cu, built
// --fmad=true). The host build path (BuildFrameTlas + buffers) stays in
// two_level_render.cu and hands this the resolved device pointers + stream.
void LaunchBeautyKernel(const PinholeCamera& camera,
                        const LbvhNode* tlas_nodes,
                        uint32_t tlas_leaf_count,
                        const DevInstance* instances,
                        const Material* materials,
                        const Light& light,
                        const BeautyParams& sky,
                        uint32_t samples,
                        uint32_t base_seed,
                        const AovTarget& dst,
                        cudaStream_t stream);

}  // namespace nuka::rt
