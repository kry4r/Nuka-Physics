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

#if defined(__CUDACC__)
#define NUKA_RT_HD __host__ __device__
#else
#define NUKA_RT_HD
#endif

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

// World-space AABB of one placed instance, the ONE box arithmetic the TLAS build/
// refit + the batched scatter share. HD, bit-for-bit vs host TransformPoint+FromBox.
NUKA_RT_HD inline collision::AABB InstanceWorldAabb(uint32_t leaf_count,
                                                   const collision::AABB& local_bound,
                                                   const Transform& xf) {
    if (leaf_count == 0u) {
        collision::AABB b;
        b.min = xf.position;
        b.max = xf.position;
        return b;
    }
    const Vec3 half{0.5f * (local_bound.max.x - local_bound.min.x),
                    0.5f * (local_bound.max.y - local_bound.min.y),
                    0.5f * (local_bound.max.z - local_bound.min.z)};
    const Vec3 center{0.5f * (local_bound.min.x + local_bound.max.x),
                      0.5f * (local_bound.min.y + local_bound.max.y),
                      0.5f * (local_bound.min.z + local_bound.max.z)};
    const Vec3 center_world = QuatRotate(xf.rotation, center) + xf.position;
    collision::AABB out;
    for (int i = 0; i < 8; ++i) {
        const Vec3 corner{(i & 1) ? half.x : -half.x, (i & 2) ? half.y : -half.y,
                          (i & 4) ? half.z : -half.z};
        const Vec3 p = QuatRotate(xf.rotation, corner) + center_world;
        // std::min/std::max semantics (Expand): min=(b<a)?b:a, max=(a<b)?b:a.
        out.min.x = (p.x < out.min.x) ? p.x : out.min.x;
        out.min.y = (p.y < out.min.y) ? p.y : out.min.y;
        out.min.z = (p.z < out.min.z) ? p.z : out.min.z;
        out.max.x = (out.max.x < p.x) ? p.x : out.max.x;
        out.max.y = (out.max.y < p.y) ? p.y : out.max.y;
        out.max.z = (out.max.z < p.z) ? p.z : out.max.z;
    }
    return out;
}

// DevInstance from a resolved transform + (shared) BLAS view + ids. The single-
// camera path and the batched scatter both build through this ONE writer.
NUKA_RT_HD inline DevInstance MakeDevInstance(const Transform& xf,
                                             const LbvhNode* blas_nodes,
                                             uint32_t blas_leaf_count,
                                             const DevBlas& blas,
                                             uint32_t instance_id,
                                             uint32_t material_id) {
    DevInstance di{};  // value-init zeroes padding -> deterministic byte layout
    di.transform = xf;
    di.blas_nodes = (blas_leaf_count > 0u) ? blas_nodes : nullptr;
    di.blas_leaf_count = blas_leaf_count;
    di.blas = blas;
    di.instance_id = instance_id;
    di.material_id = material_id;
    return di;
}

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

// ---------------------------------------------------------------------------
// SHARED beauty shade: the single-camera beauty TU and the batched sensor TU both
// call these (the ONE-path move -- one shading model, two callers). RtMath2Pi /
// RtMathPi keep the literals in one place. The RNG is a template: any struct with
// a `NextF()` returning a float in [0,1) drives the stochastic sampling, so each
// caller supplies its own deterministic, stateless-seeded generator.
// ---------------------------------------------------------------------------

NUKA_RT_HD inline float RtMath2Pi() { return 6.2831853071795864769f; }
NUKA_RT_HD inline float RtMathPi() { return 3.14159265358979323846f; }

// Orthonormal basis around a unit normal (Duff et al. 2017, branchless).
NUKA_RT_HD inline void OrthoBasis(const Vec3& n, Vec3* t, Vec3* b) {
    const float sign = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float c = n.x * n.y * a;
    *t = Vec3{1.0f + sign * n.x * n.x * a, sign * c, -sign * n.x};
    *b = Vec3{c, sign + n.y * n.y * a, -n.y};
}

// Cosine-weighted hemisphere sample about `n` (Malley's method). FP32 normalize.
NUKA_RT_HD inline Vec3 CosineHemisphere(const Vec3& n, float u1, float u2) {
    const float r = sqrtf(u1);
    const float phi = RtMath2Pi() * u2;
    const float x = r * cosf(phi);
    const float y = r * sinf(phi);
    const float z = sqrtf(fmaxf(0.0f, 1.0f - u1));
    Vec3 t, b;
    OrthoBasis(n, &t, &b);
    return RtNormalize<float>(Vec3{t.x * x + b.x * y + n.x * z,
                                   t.y * x + b.y * y + n.y * z,
                                   t.z * x + b.z * y + n.z * z});
}

// Sample a cone of half-angle `ang` about the unit axis `L` (soft sun disc).
NUKA_RT_HD inline Vec3 SampleCone(const Vec3& L, float ang, float u1, float u2) {
    const float cos_max = cosf(ang);
    const float cos_t = 1.0f - u1 * (1.0f - cos_max);
    const float sin_t = sqrtf(fmaxf(0.0f, 1.0f - cos_t * cos_t));
    const float phi = RtMath2Pi() * u2;
    Vec3 t, b;
    OrthoBasis(L, &t, &b);
    const float x = sin_t * cosf(phi);
    const float y = sin_t * sinf(phi);
    return RtNormalize<float>(Vec3{t.x * x + b.x * y + L.x * cos_t,
                                   t.y * x + b.y * y + L.y * cos_t,
                                   t.z * x + b.z * y + L.z * cos_t});
}

// Procedural sky + ground dome along a ray direction (the miss shader). Smooth
// horizon blend; below-horizon fades to a neutral ground fill so bounce rays that
// escape downward still pick up plausible fill instead of pure black.
NUKA_RT_HD inline Vec3 SkyColor(const Vec3& dir, const BeautyParams& sky) {
    if (dir.z >= 0.0f) {
        const float up = dir.z;   // 0 horizon .. 1 zenith
        const float w = up * up;  // bias color toward horizon
        return Vec3{sky.sky_bottom.x + (sky.sky_top.x - sky.sky_bottom.x) * w,
                    sky.sky_bottom.y + (sky.sky_top.y - sky.sky_bottom.y) * w,
                    sky.sky_bottom.z + (sky.sky_top.z - sky.sky_bottom.z) * w};
    }
    const float w = fminf(1.0f, -dir.z * 1.5f);
    return Vec3{sky.sky_bottom.x + (sky.sky_ground.x - sky.sky_bottom.x) * w,
                sky.sky_bottom.y + (sky.sky_ground.y - sky.sky_bottom.y) * w,
                sky.sky_bottom.z + (sky.sky_ground.z - sky.sky_bottom.z) * w};
}

// Per-pixel hit at world point `hit` with viewer-faced normal `Nf`: K-ray soft sun
// shadow + cosine AO/GI bounces. Returns the lit RGB. Visibility rays use the
// any-hit traversal; the AO/GI bounce uses a closest-hit pruned to ao_radius.
// __device__-only: it rides the device-only TLAS/BLAS traversal nest.
template <typename Rng>
__device__ __forceinline__ Vec3 ShadeBeauty(const LbvhNode* __restrict__ tlas_nodes,
                                  uint32_t tlas_leaf_count,
                                  const DevInstance* __restrict__ instances,
                                  const Material* __restrict__ materials,
                                  Light light, BeautyParams sky, const Vec3& hit,
                                  const Vec3& Nf, const Vec3& V, const Material& mat,
                                  Rng* rng) {
    // Sun direction (toward the light) + a finite angular size -> penumbra.
    Vec3 Ls;
    if (light.directional) {
        Ls = RtNormalize<float>(Vec3{-light.direction.x, -light.direction.y, -light.direction.z});
    } else {
        Ls = RtNormalize<float>(Vec3{light.position.x - hit.x, light.position.y - hit.y,
                                     light.position.z - hit.z});
    }
    const Vec3 light_col{light.color.x * light.intensity,
                         light.color.y * light.intensity,
                         light.color.z * light.intensity};
    const float eps = 1.0e-3f;
    const Vec3 sorigin{hit.x + Nf.x * eps, hit.y + Nf.y * eps, hit.z + Nf.z * eps};

    // Soft shadow: average visibility over K cone-sampled sun directions (any-hit
    // visibility -- boolean occlusion, first hit wins).
    float vis = 0.0f;
    const uint32_t K = sky.shadow_rays < 1u ? 1u : sky.shadow_rays;
    for (uint32_t k = 0; k < K; ++k) {
        const Vec3 Lk = SampleCone(Ls, sky.sun_angular_radius, rng->NextF(), rng->NextF());
        if (Lk.x * Nf.x + Lk.y * Nf.y + Lk.z * Nf.z <= 0.0f) continue;
        if (!AnyOccluder<float>(tlas_nodes, tlas_leaf_count, instances, sorigin, Lk, eps,
                                RtMissDepth())) {
            vis += 1.0f;
        }
    }
    vis /= static_cast<float>(K);

    const float NoL = fmaxf(0.0f, Nf.x * Ls.x + Nf.y * Ls.y + Nf.z * Ls.z);
    // Half vector for a GGX-ish specular highlight so the shell catches the sun.
    Vec3 H = RtNormalize<float>(Vec3{V.x + Ls.x, V.y + Ls.y, V.z + Ls.z});
    const float NoH = fmaxf(0.0f, Nf.x * H.x + Nf.y * H.y + Nf.z * H.z);
    const float alpha = fmaxf(1.0e-3f, mat.roughness * mat.roughness);
    const float a2 = alpha * alpha;
    const float dterm = NoH * NoH * (a2 - 1.0f) + 1.0f;
    const float spec = (a2 / (RtMathPi() * dterm * dterm)) * (0.04f + mat.metallic);

    const float kd = (1.0f - mat.metallic) * (1.0f / RtMathPi());
    Vec3 direct{light_col.x * (mat.albedo.x * kd + spec) * NoL * vis,
                light_col.y * (mat.albedo.y * kd + spec) * NoL * vis,
                light_col.z * (mat.albedo.z * kd + spec) * NoL * vis};

    // AO + one-bounce GI: cosine-weighted hemisphere rays. A ray that escapes the
    // AO radius (or misses) samples the sky dome; a near hit darkens the crease
    // and (for GI) picks up the bounce albedo*shade of the hit surface.
    Vec3 indirect{0.0f, 0.0f, 0.0f};
    const uint32_t M = sky.ao_samples < 1u ? 1u : sky.ao_samples;
    for (uint32_t m = 0; m < M; ++m) {
        const Vec3 d = CosineHemisphere(Nf, rng->NextF(), rng->NextF());
        const Vec3 ro{hit.x + Nf.x * eps, hit.y + Nf.y * eps, hit.z + Nf.z * eps};
        float bt; uint32_t bp;
        ClosestHit<float>(tlas_nodes, tlas_leaf_count, instances, ro, d, eps, &bt, &bp,
                          sky.ao_radius);
        if (bp == kNoPrim || bt >= sky.ao_radius) {
            const Vec3 s = SkyColor(d, sky);
            indirect.x += s.x * sky.sky_intensity;
            indirect.y += s.y * sky.sky_intensity;
            indirect.z += s.z * sky.sky_intensity;
            continue;
        }
        if (sky.gi_bounces == 0u) continue;  // AO only: occluded -> no light
        Vec3 bn; float bu, bv;
        ReconstructHit<float>(instances, bp, ro, d, &bn, &bu, &bv);
        const float bnv = bn.x * (-d.x) + bn.y * (-d.y) + bn.z * (-d.z);
        Vec3 bnf = (bnv < 0.0f) ? Vec3{-bn.x, -bn.y, -bn.z} : bn;
        uint32_t bi, blp; UnpackPrimId(bp, &bi, &blp);
        const Material bmat = materials[instances[bi].material_id];
        const Vec3 bhit{ro.x + bt * d.x, ro.y + bt * d.y, ro.z + bt * d.z};
        const Vec3 bso{bhit.x + bnf.x * eps, bhit.y + bnf.y * eps, bhit.z + bnf.z * eps};
        const bool bshadow =
            AnyOccluder<float>(tlas_nodes, tlas_leaf_count, instances, bso, Ls, eps,
                               RtMissDepth());
        const float bNoL = fmaxf(0.0f, bnf.x * Ls.x + bnf.y * Ls.y + bnf.z * Ls.z);
        const float bvis = bshadow ? 0.0f : 1.0f;
        const float bfac = bNoL * bvis * (1.0f / RtMathPi());
        indirect.x += bmat.albedo.x * light_col.x * bfac;
        indirect.y += bmat.albedo.y * light_col.y * bfac;
        indirect.z += bmat.albedo.z * light_col.z * bfac;
    }
    const float inv_m = 1.0f / static_cast<float>(M);
    indirect.x *= inv_m; indirect.y *= inv_m; indirect.z *= inv_m;

    // Indirect modulates the surface albedo (Lambert response to ambient/bounce).
    return Vec3{direct.x + mat.albedo.x * indirect.x,
                direct.y + mat.albedo.y * indirect.y,
                direct.z + mat.albedo.z * indirect.z};
}

// Filmic ACES-ish tonemap of one linear channel (Narkowicz 2015 fit), clamped to
// [0,1]. Applied per-channel to the averaged fidelity color when tonemap is on.
NUKA_RT_HD inline float TonemapAces(float x) {
    x = x < 0.0f ? 0.0f : x;
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    const float y = (x * (a * x + b)) / (x * (c * x + d) + e);
    return y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
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

#undef NUKA_RT_HD
