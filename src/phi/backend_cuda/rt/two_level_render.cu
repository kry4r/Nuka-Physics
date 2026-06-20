// ---------------------------------------------------------------------------
// nuka::rt -- the self-written TWO-LEVEL (TLAS/BLAS) ray tracer (v0.7 G1-A).
// NO OptiX / NO closed SDK. See two_level_render.hpp for the algorithm + scope.
//
// THE NEST (architect + advisor verified): the p12 stackless rt::TraverseRay is
// REUSED VERBATIM at BOTH levels.
//   * TLAS = LBVH over instance WORLD-AABBs, rebuilt per frame (RenderFrame).
//   * BLAS = per-unique-mesh LBVH over LOCAL prims, built ONCE (BuildTwoLevelScene).
//   * TlasLeaf, instead of intersecting a primitive, (a) transforms the ray into
//     the instance-local frame (rt::TransformRayToLocal), (b) runs an INNER
//     rt::TraverseRay over that instance's BLAS with a p13-style BlasLeaf,
//     (c) the inner traversal updates the SAME best_t / best_prim by reference.
//   * Rigid -> |d_l| == |d| == 1 -> local-frame t == world t, so best_t is
//     directly comparable across instances with NO rescale; t_min passes through.
//
// SINGLE-LEAF GUARD (load-bearing): rt::TraverseRay assumes node 0 is an INTERNAL
// node (internal_count = leaf_count-1). For a 1-leaf tree the caller must
// intersect node 0 directly. This guard appears at BOTH levels here:
//   * the OUTER TLAS ClosestHit (leaf_count==1 -> call TlasLeaf on node 0), and
//   * INSIDE TlasLeaf for the per-instance BLAS (blas_leaf_count==1 -> call
//     BlasLeaf on node 0). A single-primitive mesh hits the inner path.
//
// prim_id AOV packs (instance, local-prim) via rt::PackPrimId (instance HIGH) ->
// the closest-hit tie-break is a TOTAL ORDER -> D1 by construction (atomic-free,
// one ray/pixel, two runs memcmp-identical across all 6 AOVs).
// ---------------------------------------------------------------------------

#include "rt/two_level_render.hpp"

#include "collision/aabb.hpp"
#include "collision/broadphase_lbvh.hpp"
#include "collision/lbvh_node.cuh"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend.hpp"             // InitBestDevice / BackendDeviceBufferType
#include "phi/buffer.hpp"              // Buffer* / BufferAlloc / BufferUpload / ...
#include "phi/buffer_transfer_v2.hpp"  // UploadVectorV2
#include "phi/scoped_device_guard.hpp"
#include "phi/backend_cuda/rt/bvh_traverse_impl.cuh"
#include "phi/backend_cuda/rt/rt_device_context.cuh"  // RtContext / OwnedBuffer
#include "phi/backend_cuda/rt/instance_transform.cuh"
#include "phi/backend_cuda/rt/intersect_primitives.cuh"
#include "phi/backend_cuda/rt/prim_id.cuh"
#include "phi/backend_cuda/rt/ray_box.cuh"
#include "phi/backend_cuda/rt/shading.cuh"
#include "rt/camera.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::rt {

namespace {

constexpr uint32_t kBlockDim = 16u;

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

using ::nuka::collision::gpu::LbvhNode;
using ::nuka::math::Transform;
using ::nuka::math::Vec3;

// Flat per-primitive record for ONE mesh's BLAS, indexed by the BLAS leaf's
// original (LOCAL) prim id. Mirrors scene_render.cu's DevPrim but the material
// is dropped (material is per-instance in the two-level model).
struct DevPrim {
    uint32_t kind;  // PrimKind
    uint32_t sub;   // index into the per-kind array
};

// Device view of ONE mesh's BLAS geometry (raw pointers into uploaded buffers).
// Same layout discipline as scene_render.cu's DevScene, minus the materials.
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

// Intersect one LOCAL primitive of a BLAS (closest-hit pass): returns t only.
// LOCAL-frame ray in; rigid -> local t == world t (see instance_transform.cuh).
// Mirrors scene_render.cu::IntersectPrimT exactly (shared intersection fns).
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
        return RayTriangleIntersect(origin, dir, b.tri_v0[p.sub], b.tri_v1[p.sub],
                                    b.tri_v2[p.sub], t_min, out_t, &u, &v, &n);
    } else if (p.kind == static_cast<uint32_t>(PrimKind::Sphere)) {
        Vec3 n;
        float u, v;
        return RaySphereIntersect(origin, dir, b.sph_center[p.sub],
                                  b.sph_radius[p.sub], t_min, out_t, &n, &u, &v);
    } else {  // Sdf
        Vec3 n;
        return RaySdfSphereTrace(origin, dir, b.sdf_hdr[p.sub], b.sdf_aabb[p.sub],
                                 t_min, b.sdf_eps[p.sub], b.sdf_iters[p.sub],
                                 out_t, &n);
    }
}

// BLAS leaf functor (the INNER traversal). origin/dir are the INSTANCE-LOCAL
// ray; t_min is the (frame-invariant) world/local t_min. On a closest-hit win it
// writes PackPrimId(instance_id, local_prim) into best_prim so the global tie-
// break is a TOTAL ORDER (instance HIGH). best_t carries the comparable world t.
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
        if (IntersectBlasPrimT(blas, local_prim, origin, dir, t_min, &t)) {
            if (t >= t_min) {
                RtClosestHitUpdate(t, PackPrimId(instance_id, local_prim), best_t,
                                   best_prim);
            }
        }
    }
};

// TLAS leaf functor (the heart of the nest). For the instance at TLAS leaf index
// nodes[leaf_node].left, transform the WORLD ray into the instance-local frame,
// then run the INNER rt::TraverseRay over that instance's BLAS, updating the SAME
// best_t / best_prim by reference. The single-leaf BLAS is handled directly (the
// inner TraverseRay assumes node 0 is an internal node).
struct TlasLeaf {
    const LbvhNode* __restrict__ tlas_nodes;
    const DevInstance* __restrict__ instances;
    float t_min;
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
        // World ray -> instance-local frame (shared with the host oracle).
        Vec3 o_l, d_l;
        TransformRayToLocal(I.transform, origin, dir, &o_l, &d_l);

        BlasLeaf leaf{I.blas_nodes, I.blas, I.instance_id, t_min};
        if (I.blas_leaf_count == 1u) {
            // Single-primitive BLAS: node 0 IS the leaf (inner TraverseRay would
            // mis-treat it as an internal node).
            leaf(0, o_l, d_l, best_t, best_prim);
        } else {
            TraverseRay(I.blas_nodes, I.blas_leaf_count - 1u, o_l, d_l, best_t,
                        best_prim, leaf);
        }
    }
};

// Run the full TWO-LEVEL closest hit (primary or shadow). best_t carries the
// comparable world t; best_prim carries the packed (instance, local) id.
__device__ __forceinline__ void ClosestHit(const LbvhNode* __restrict__ tlas_nodes,
                                           uint32_t tlas_leaf_count,
                                           const DevInstance* __restrict__ instances,
                                           const Vec3& origin,
                                           const Vec3& dir,
                                           float t_min,
                                           float* best_t,
                                           uint32_t* best_prim) {
    *best_t = RtMissDepth();
    *best_prim = kNoPrim;
    TlasLeaf leaf{tlas_nodes, instances, t_min};
    if (tlas_leaf_count == 1u) {
        leaf(0, origin, dir, best_t, best_prim);
    } else if (tlas_leaf_count >= 2u) {
        TraverseRay(tlas_nodes, tlas_leaf_count - 1u, origin, dir, best_t,
                    best_prim, leaf);
    }
}

// Reconstruct surface attributes (world normal, uv) for the WINNING (instance,
// local) prim. Re-transform the ray to that instance's local frame, re-intersect
// for the LOCAL normal/uv (p13 ReconstructHit pattern), then rotate the normal
// back to world via QuatRotate (rigid: rotation only).
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
        RayTriangleIntersect(o_l, d_l, b.tri_v0[p.sub], b.tri_v1[p.sub],
                             b.tri_v2[p.sub], 0.0f, &tt, &u, &v, &n_local);
    } else if (p.kind == static_cast<uint32_t>(PrimKind::Sphere)) {
        float tt;
        RaySphereIntersect(o_l, d_l, b.sph_center[p.sub], b.sph_radius[p.sub],
                           0.0f, &tt, &n_local, &u, &v);
    } else {  // Sdf
        float tt;
        RaySdfSphereTrace(o_l, d_l, b.sdf_hdr[p.sub], b.sdf_aabb[p.sub], 0.0f,
                          b.sdf_eps[p.sub], b.sdf_iters[p.sub], &tt, &n_local);
        u = 0.0f;
        v = 0.0f;
    }
    // Local geometric normal -> world (rotation only).
    *out_world_normal = QuatRotate(I.transform.rotation, n_local);
    *out_u = u;
    *out_v = v;
}

// One thread per pixel: primary two-level closest hit -> reconstruct -> shadow
// ray -> shade -> write ALL 6 AOVs. Exactly one writer per pixel; no atomics ->
// D1 byte-exact. Mirrors scene_render.cu::RenderSceneKernel; the ONLY difference
// is the closest-hit / reconstruct go through the TLAS/BLAS nest.
__global__ void RenderFrameKernel(PinholeCamera camera,
                                 const LbvhNode* __restrict__ tlas_nodes,
                                 uint32_t tlas_leaf_count,
                                 const DevInstance* __restrict__ instances,
                                 const Material* __restrict__ materials,
                                 Light light,
                                 AmbientTerm ambient,
                                 float* __restrict__ out_color,
                                 float* __restrict__ out_depth,
                                 float* __restrict__ out_normal,
                                 float* __restrict__ out_albedo,
                                 float* __restrict__ out_uv,
                                 uint32_t* __restrict__ out_prim) {
    const uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= camera.width || py >= camera.height) {
        return;
    }
    const uint32_t pixel = py * camera.width + px;
    const Ray ray = camera.GenerateRay(px, py);

    float best_t;
    uint32_t best_prim;
    ClosestHit(tlas_nodes, tlas_leaf_count, instances, ray.origin, ray.dir, 0.0f,
               &best_t, &best_prim);

    Vec3 color{0.0f, 0.0f, 0.0f};
    float depth = RtMissDepth();
    Vec3 normal{0.0f, 0.0f, 0.0f};
    Vec3 albedo{0.0f, 0.0f, 0.0f};
    float uv_u = 0.0f, uv_v = 0.0f;
    uint32_t prim_id = kNoPrim;

    if (best_prim != kNoPrim) {
        depth = best_t;
        prim_id = best_prim;

        uint32_t inst, local_prim;
        UnpackPrimId(best_prim, &inst, &local_prim);
        const uint32_t material_id = instances[inst].material_id;

        Vec3 n;
        ReconstructHit(instances, best_prim, ray.origin, ray.dir, &n, &uv_u, &uv_v);
        normal = n;
        const Material mat = materials[material_id];
        albedo = mat.albedo;

        // Hit point p = origin + t*dir (fp64; mirrors scene_render.cu).
        const Vec3 hit{
            static_cast<float>(static_cast<double>(ray.origin.x) + static_cast<double>(best_t) * static_cast<double>(ray.dir.x)),
            static_cast<float>(static_cast<double>(ray.origin.y) + static_cast<double>(best_t) * static_cast<double>(ray.dir.y)),
            static_cast<float>(static_cast<double>(ray.origin.z) + static_cast<double>(best_t) * static_cast<double>(ray.dir.z))};

        const Vec3 V = RtNormalize(Vec3{-ray.dir.x, -ray.dir.y, -ray.dir.z});
        Vec3 L;
        float light_dist;
        if (light.directional) {
            L = RtNormalize(Vec3{-light.direction.x, -light.direction.y, -light.direction.z});
            light_dist = RtMissDepth();
        } else {
            const Vec3 to{light.position.x - hit.x, light.position.y - hit.y,
                          light.position.z - hit.z};
            L = RtNormalize(to);
            light_dist = static_cast<float>(sqrt(
                static_cast<double>(to.x) * to.x +
                static_cast<double>(to.y) * to.y +
                static_cast<double>(to.z) * to.z));
        }

        // Hard shadow ray (SAME nested traversal): offset along the viewer-faced
        // normal to avoid acne; the NaN guard handles axis-parallel shadow rays.
        Vec3 ns = n;
        const double nv = static_cast<double>(n.x) * V.x + static_cast<double>(n.y) * V.y +
                          static_cast<double>(n.z) * V.z;
        if (nv < 0.0) {
            ns = Vec3{-n.x, -n.y, -n.z};
        }
        const float shadow_eps = 1.0e-3f;
        const Vec3 sorigin{hit.x + ns.x * shadow_eps, hit.y + ns.y * shadow_eps,
                           hit.z + ns.z * shadow_eps};

        int lit = 1;
        float st;
        uint32_t sp;
        ClosestHit(tlas_nodes, tlas_leaf_count, instances, sorigin, L, shadow_eps,
                   &st, &sp);
        if (sp != kNoPrim && st < light_dist - shadow_eps) {
            lit = 0;
        }

        const Vec3 light_col{light.color.x * light.intensity,
                             light.color.y * light.intensity,
                             light.color.z * light.intensity};
        color = ShadeDirect(n, V, L, light_col, mat, lit, ambient.color);
    }

    out_color[pixel * 3u + 0u] = color.x;
    out_color[pixel * 3u + 1u] = color.y;
    out_color[pixel * 3u + 2u] = color.z;
    out_depth[pixel] = depth;
    out_normal[pixel * 3u + 0u] = normal.x;
    out_normal[pixel * 3u + 1u] = normal.y;
    out_normal[pixel * 3u + 2u] = normal.z;
    out_albedo[pixel * 3u + 0u] = albedo.x;
    out_albedo[pixel * 3u + 1u] = albedo.y;
    out_albedo[pixel * 3u + 2u] = albedo.z;
    out_uv[pixel * 2u + 0u] = uv_u;
    out_uv[pixel * 2u + 1u] = uv_v;
    out_prim[pixel] = prim_id;
}

// ---------------------------------------------------------------------------
// BEAUTY PATH (stochastic): a SEPARATE kernel that reuses the SAME ClosestHit /
// ReconstructHit nest as RenderFrame but adds jittered MSAA, soft area-sun
// shadows, cosine AO + one-bounce GI, and a procedural sky/fog miss shader. It
// writes only a tonemap-free linear `color` (+ the center-sample primary AOVs);
// RenderFrame above is byte-untouched so the D1 sensor goldens stay green.
// ---------------------------------------------------------------------------

// PCG32-ish hash RNG: cheap, well-distributed, deterministic from (seed, counter)
// so a fixed BeautyOptions::seed makes the whole frame reproducible run-to-run.
__device__ __forceinline__ uint32_t PcgHash(uint32_t v) {
    uint32_t state = v * 747796405u + 2891336453u;
    uint32_t word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct BeautyRng {
    uint32_t s;
    __device__ __forceinline__ float NextF() {
        s = PcgHash(s);
        // 24-bit mantissa float in [0,1).
        return static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
    }
};

// Orthonormal basis around a unit normal (Duff et al. 2017, branchless).
__device__ __forceinline__ void OrthoBasis(const Vec3& n, Vec3* t, Vec3* b) {
    const float sign = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float c = n.x * n.y * a;
    *t = Vec3{1.0f + sign * n.x * n.x * a, sign * c, -sign * n.x};
    *b = Vec3{c, sign + n.y * n.y * a, -n.y};
}

// Cosine-weighted hemisphere sample about `n` (Malley's method).
__device__ __forceinline__ Vec3 CosineHemisphere(const Vec3& n, float u1, float u2) {
    const float r = sqrtf(u1);
    const float phi = 6.2831853071795864769f * u2;
    const float x = r * cosf(phi);
    const float y = r * sinf(phi);
    const float z = sqrtf(fmaxf(0.0f, 1.0f - u1));
    Vec3 t, b;
    OrthoBasis(n, &t, &b);
    return RtNormalize(Vec3{t.x * x + b.x * y + n.x * z,
                            t.y * x + b.y * y + n.y * z,
                            t.z * x + b.z * y + n.z * z});
}

// Sample a cone of half-angle `ang` about the unit axis `L` (soft sun disc).
__device__ __forceinline__ Vec3 SampleCone(const Vec3& L, float ang, float u1, float u2) {
    const float cos_max = cosf(ang);
    const float cos_t = 1.0f - u1 * (1.0f - cos_max);
    const float sin_t = sqrtf(fmaxf(0.0f, 1.0f - cos_t * cos_t));
    const float phi = 6.2831853071795864769f * u2;
    Vec3 t, b;
    OrthoBasis(L, &t, &b);
    const float x = sin_t * cosf(phi);
    const float y = sin_t * sinf(phi);
    return RtNormalize(Vec3{t.x * x + b.x * y + L.x * cos_t,
                            t.y * x + b.y * y + L.y * cos_t,
                            t.z * x + b.z * y + L.z * cos_t});
}

// Procedural sky + ground dome along a ray direction (the miss shader). Smooth
// horizon blend; below-horizon fades to a neutral ground fill so bounce rays that
// escape downward still pick up plausible fill instead of pure black.
__device__ __forceinline__ Vec3 SkyColor(const Vec3& dir, const BeautyParams& sky) {
    const float t = 0.5f * (dir.z + 1.0f);             // -1..1 -> 0..1
    if (dir.z >= 0.0f) {
        const float up = dir.z;                        // 0 horizon .. 1 zenith
        const float w = up * up;                       // bias color toward horizon
        return Vec3{sky.sky_bottom.x + (sky.sky_top.x - sky.sky_bottom.x) * w,
                    sky.sky_bottom.y + (sky.sky_top.y - sky.sky_bottom.y) * w,
                    sky.sky_bottom.z + (sky.sky_top.z - sky.sky_bottom.z) * w};
    }
    const float w = fminf(1.0f, -dir.z * 1.5f);
    (void)t;
    return Vec3{sky.sky_bottom.x + (sky.sky_ground.x - sky.sky_bottom.x) * w,
                sky.sky_bottom.y + (sky.sky_ground.y - sky.sky_bottom.y) * w,
                sky.sky_bottom.z + (sky.sky_ground.z - sky.sky_bottom.z) * w};
}

// Per-pixel hit at world point `hit` with viewer-faced normal `Nf`: K-ray soft
// sun shadow + cosine AO/GI bounces. Returns the lit RGB (diffuse + specular +
// indirect). Reuses the SAME ClosestHit nest for every secondary ray.
__device__ __forceinline__ Vec3 ShadeBeauty(const LbvhNode* __restrict__ tlas_nodes,
                                            uint32_t tlas_leaf_count,
                                            const DevInstance* __restrict__ instances,
                                            const Material* __restrict__ materials,
                                            Light light, BeautyParams sky,
                                            const Vec3& hit, const Vec3& Nf, const Vec3& V,
                                            const Material& mat, BeautyRng* rng) {
    // Sun direction (toward the light) + a finite angular size -> penumbra.
    Vec3 Ls;
    if (light.directional) {
        Ls = RtNormalize(Vec3{-light.direction.x, -light.direction.y, -light.direction.z});
    } else {
        Ls = RtNormalize(Vec3{light.position.x - hit.x, light.position.y - hit.y,
                              light.position.z - hit.z});
    }
    const Vec3 light_col{light.color.x * light.intensity,
                         light.color.y * light.intensity,
                         light.color.z * light.intensity};
    const float eps = 1.0e-3f;
    const Vec3 sorigin{hit.x + Nf.x * eps, hit.y + Nf.y * eps, hit.z + Nf.z * eps};

    // Soft shadow: average visibility over K cone-sampled sun directions.
    float vis = 0.0f;
    const uint32_t K = sky.shadow_rays < 1u ? 1u : sky.shadow_rays;
    for (uint32_t k = 0; k < K; ++k) {
        const Vec3 Lk = SampleCone(Ls, sky.sun_angular_radius, rng->NextF(), rng->NextF());
        if (Lk.x * Nf.x + Lk.y * Nf.y + Lk.z * Nf.z <= 0.0f) continue;
        float st; uint32_t sp;
        ClosestHit(tlas_nodes, tlas_leaf_count, instances, sorigin, Lk, eps, &st, &sp);
        if (sp == kNoPrim) vis += 1.0f;
    }
    vis /= static_cast<float>(K);

    const float NoL = fmaxf(0.0f, Nf.x * Ls.x + Nf.y * Ls.y + Nf.z * Ls.z);
    // Half vector for a GGX-ish specular highlight so the shell catches the sun.
    Vec3 H = RtNormalize(Vec3{V.x + Ls.x, V.y + Ls.y, V.z + Ls.z});
    const float NoH = fmaxf(0.0f, Nf.x * H.x + Nf.y * H.y + Nf.z * H.z);
    const float alpha = fmaxf(1.0e-3f, mat.roughness * mat.roughness);
    const float a2 = alpha * alpha;
    const float dterm = NoH * NoH * (a2 - 1.0f) + 1.0f;
    const float spec = (a2 / (3.14159265f * dterm * dterm)) * (0.04f + mat.metallic);

    const float kd = (1.0f - mat.metallic) * (1.0f / 3.14159265f);
    Vec3 direct{
        light_col.x * (mat.albedo.x * kd + spec) * NoL * vis,
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
        ClosestHit(tlas_nodes, tlas_leaf_count, instances, ro, d, eps, &bt, &bp);
        if (bp == kNoPrim || bt >= sky.ao_radius) {
            // Open sky above -> sky-dome ambient (cosine already in the sample pdf).
            const Vec3 s = SkyColor(d, sky);
            indirect.x += s.x * sky.sky_intensity;
            indirect.y += s.y * sky.sky_intensity;
            indirect.z += s.z * sky.sky_intensity;
            continue;
        }
        if (sky.gi_bounces == 0u) continue;  // AO only: occluded -> no light
        // One bounce: re-shade the hit surface with a direct sun term (no further
        // recursion) and add its albedo-weighted contribution.
        Vec3 bn; float bu, bv;
        ReconstructHit(instances, bp, ro, d, &bn, &bu, &bv);
        const float bnv = bn.x * (-d.x) + bn.y * (-d.y) + bn.z * (-d.z);
        Vec3 bnf = (bnv < 0.0f) ? Vec3{-bn.x, -bn.y, -bn.z} : bn;
        uint32_t bi, blp; UnpackPrimId(bp, &bi, &blp);
        const Material bmat = materials[instances[bi].material_id];
        const Vec3 bhit{ro.x + bt * d.x, ro.y + bt * d.y, ro.z + bt * d.z};
        const Vec3 bso{bhit.x + bnf.x * eps, bhit.y + bnf.y * eps, bhit.z + bnf.z * eps};
        float st2; uint32_t sp2;
        ClosestHit(tlas_nodes, tlas_leaf_count, instances, bso, Ls, eps, &st2, &sp2);
        const float bNoL = fmaxf(0.0f, bnf.x * Ls.x + bnf.y * Ls.y + bnf.z * Ls.z);
        const float bvis = (sp2 == kNoPrim) ? 1.0f : 0.0f;
        const float bfac = bNoL * bvis * (1.0f / 3.14159265f);
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

// One thread per pixel: accumulate S jittered sub-pixel samples through the
// stochastic shade. The CENTER sample also fills depth/normal/albedo/uv/prim so
// the host RGBA8 conversion's prim-based hit test still classifies background.
__global__ void RenderBeautyKernel(PinholeCamera camera,
                                   const LbvhNode* __restrict__ tlas_nodes,
                                   uint32_t tlas_leaf_count,
                                   const DevInstance* __restrict__ instances,
                                   const Material* __restrict__ materials,
                                   Light light, BeautyParams sky, uint32_t samples,
                                   uint32_t base_seed,
                                   float* __restrict__ out_color,
                                   float* __restrict__ out_depth,
                                   float* __restrict__ out_normal,
                                   float* __restrict__ out_albedo,
                                   float* __restrict__ out_uv,
                                   uint32_t* __restrict__ out_prim) {
    const uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= camera.width || py >= camera.height) return;
    const uint32_t pixel = py * camera.width + px;

    BeautyRng rng{PcgHash(base_seed ^ (pixel * 2654435761u))};
    const uint32_t S = samples < 1u ? 1u : samples;

    Vec3 accum{0.0f, 0.0f, 0.0f};
    float c_depth = RtMissDepth();
    Vec3 c_normal{0.0f, 0.0f, 0.0f}, c_albedo{0.0f, 0.0f, 0.0f};
    float c_u = 0.0f, c_v = 0.0f;
    uint32_t c_prim = kNoPrim;

    for (uint32_t s = 0; s < S; ++s) {
        // Sub-pixel jitter in [-0.5,0.5]; sample 0 uses center for the AOV fill.
        const float jx = (s == 0u) ? 0.0f : (rng.NextF() - 0.5f);
        const float jy = (s == 0u) ? 0.0f : (rng.NextF() - 0.5f);
        const Ray ray = camera.GenerateRayJitter(px, py, jx, jy);

        float bt; uint32_t bp;
        ClosestHit(tlas_nodes, tlas_leaf_count, instances, ray.origin, ray.dir, 0.0f,
                   &bt, &bp);
        if (bp == kNoPrim) {
            const Vec3 miss = SkyColor(ray.dir, sky);
            accum.x += miss.x; accum.y += miss.y; accum.z += miss.z;
            continue;
        }
        Vec3 n; float u, v;
        ReconstructHit(instances, bp, ray.origin, ray.dir, &n, &u, &v);
        const float nv = n.x * (-ray.dir.x) + n.y * (-ray.dir.y) + n.z * (-ray.dir.z);
        const Vec3 Nf = (nv < 0.0f) ? Vec3{-n.x, -n.y, -n.z} : n;
        uint32_t inst, lp; UnpackPrimId(bp, &inst, &lp);
        const Material mat = materials[instances[inst].material_id];
        const Vec3 hit{ray.origin.x + bt * ray.dir.x, ray.origin.y + bt * ray.dir.y,
                       ray.origin.z + bt * ray.dir.z};
        const Vec3 Vv = RtNormalize(Vec3{-ray.dir.x, -ray.dir.y, -ray.dir.z});
        Vec3 col = ShadeBeauty(tlas_nodes, tlas_leaf_count, instances, materials,
                               light, sky, hit, Nf, Vv, mat, &rng);
        // Height/distance fog toward the sky-horizon: blend by 1-exp(-density*t).
        if (sky.fog_density > 0.0f) {
            const float f = 1.0f - expf(-sky.fog_density * bt);
            col.x += (sky.fog_color.x - col.x) * f;
            col.y += (sky.fog_color.y - col.y) * f;
            col.z += (sky.fog_color.z - col.z) * f;
        }
        accum.x += col.x; accum.y += col.y; accum.z += col.z;

        if (s == 0u) {
            c_depth = bt; c_normal = n; c_albedo = mat.albedo;
            c_u = u; c_v = v; c_prim = bp;
        }
    }

    const float inv_s = 1.0f / static_cast<float>(S);
    out_color[pixel * 3u + 0u] = accum.x * inv_s;
    out_color[pixel * 3u + 1u] = accum.y * inv_s;
    out_color[pixel * 3u + 2u] = accum.z * inv_s;
    out_depth[pixel] = c_depth;
    out_normal[pixel * 3u + 0u] = c_normal.x;
    out_normal[pixel * 3u + 1u] = c_normal.y;
    out_normal[pixel * 3u + 2u] = c_normal.z;
    out_albedo[pixel * 3u + 0u] = c_albedo.x;
    out_albedo[pixel * 3u + 1u] = c_albedo.y;
    out_albedo[pixel * 3u + 2u] = c_albedo.z;
    out_uv[pixel * 2u + 0u] = c_u;
    out_uv[pixel * 2u + 1u] = c_v;
    out_prim[pixel] = c_prim;
}

// Per-prim AABB for a LOCAL mesh primitive (BLAS build). Same as scene_render.cu.
collision::AABB TriAabb(const TrianglePrim& t) {
    collision::AABB b;
    b.min = {fminf(fminf(t.v0.x, t.v1.x), t.v2.x),
             fminf(fminf(t.v0.y, t.v1.y), t.v2.y),
             fminf(fminf(t.v0.z, t.v1.z), t.v2.z)};
    b.max = {fmaxf(fmaxf(t.v0.x, t.v1.x), t.v2.x),
             fmaxf(fmaxf(t.v0.y, t.v1.y), t.v2.y),
             fmaxf(fmaxf(t.v0.z, t.v1.z), t.v2.z)};
    return b;
}

collision::AABB SphAabb(const SpherePrim& s) {
    collision::AABB b;
    b.min = {s.center.x - s.radius, s.center.y - s.radius, s.center.z - s.radius};
    b.max = {s.center.x + s.radius, s.center.y + s.radius, s.center.z + s.radius};
    return b;
}

// All device buffers + retained tree owning ONE mesh's BLAS. Move-only via the
// move-only members (OwnedBuffer, LbvhBroadphaseResult).
struct BlasDevice {
    // Uploaded LOCAL prim arrays (kept alive for the kernel's lifetime).
    OwnedBuffer d_prims;
    OwnedBuffer d_tri_v0, d_tri_v1, d_tri_v2;
    OwnedBuffer d_sph_c, d_sph_r;
    OwnedBuffer d_sdf_hdr, d_sdf_aabb, d_sdf_eps, d_sdf_iters;
    // SDF cell arrays (one buffer per SDF; their device pointers are baked into
    // the uploaded SparseSdfDevice headers).
    std::vector<OwnedBuffer> sdf_keys_bufs, sdf_vals_bufs, sdf_grad_bufs;

    // Retained BLAS tree (rigid -> never refit).
    collision::gpu::LbvhBroadphaseResult tree;
    uint32_t leaf_count = 0u;

    // Local-space AABB of the WHOLE mesh (union of prim AABBs) -> the TLAS world-
    // box build transforms this oriented box by each instance's pose.
    collision::AABB local_bound;

    // Reconstructed DevBlas pointing at the uploaded buffers (filled post-upload).
    DevBlas view;
};

}  // namespace

// Opaque impl: the per-mesh BLAS devices (built once). The TLAS is per-frame.
struct TwoLevelSceneDevice::Impl {
    std::vector<BlasDevice> meshes;
};

TwoLevelSceneDevice::TwoLevelSceneDevice() : impl_(std::make_unique<Impl>()) {}
TwoLevelSceneDevice::~TwoLevelSceneDevice() = default;
TwoLevelSceneDevice::TwoLevelSceneDevice(TwoLevelSceneDevice&&) noexcept = default;
TwoLevelSceneDevice& TwoLevelSceneDevice::operator=(TwoLevelSceneDevice&&) noexcept = default;

namespace {

// Build ONE mesh's BLAS: flat prim table + per-prim AABBs (decl order: triangles,
// spheres, SDFs) -> upload local prim buffers -> BuildLbvhForQuery (retain). The
// local-space mesh bound (for the TLAS world box) is the union of prim AABBs.
BlasDevice BuildBlas(const BlasMesh& mesh, const RtContext& ctx) {
    BlasDevice out;
    phi::BufferType* bt = ctx.device_bt;

    std::vector<collision::AABB> aabbs;
    std::vector<DevPrim> prims;

    std::vector<Vec3> tri_v0, tri_v1, tri_v2;
    for (const auto& t : mesh.triangles) {
        aabbs.push_back(TriAabb(t));
        prims.push_back({static_cast<uint32_t>(PrimKind::Triangle),
                         static_cast<uint32_t>(tri_v0.size())});
        tri_v0.push_back(t.v0);
        tri_v1.push_back(t.v1);
        tri_v2.push_back(t.v2);
    }

    std::vector<Vec3> sph_center;
    std::vector<float> sph_radius;
    for (const auto& s : mesh.spheres) {
        aabbs.push_back(SphAabb(s));
        prims.push_back({static_cast<uint32_t>(PrimKind::Sphere),
                         static_cast<uint32_t>(sph_center.size())});
        sph_center.push_back(s.center);
        sph_radius.push_back(s.radius);
    }

    std::vector<runtime::sdf::SparseSdfDevice> sdf_hdrs;
    std::vector<collision::AABB> sdf_aabbs;
    std::vector<float> sdf_eps;
    std::vector<int> sdf_iters;
    for (const auto& sd : mesh.sdfs) {
        aabbs.push_back(sd.aabb);
        prims.push_back({static_cast<uint32_t>(PrimKind::Sdf),
                         static_cast<uint32_t>(sdf_hdrs.size())});

        out.sdf_keys_bufs.push_back(UploadOwned(bt, sd.keys));
        out.sdf_vals_bufs.push_back(UploadOwned(bt, sd.values));
        out.sdf_grad_bufs.push_back(UploadOwned(bt, sd.gradients));

        runtime::sdf::SparseSdfDevice hdr = sd.header;
        hdr.cell_keys = static_cast<const uint64_t*>(out.sdf_keys_bufs.back().Data());
        hdr.cell_values = static_cast<const float*>(out.sdf_vals_bufs.back().Data());
        hdr.cell_gradients = static_cast<const Vec3*>(out.sdf_grad_bufs.back().Data());
        hdr.cell_count = static_cast<uint32_t>(sd.values.size());
        sdf_hdrs.push_back(hdr);
        sdf_aabbs.push_back(sd.aabb);
        sdf_eps.push_back(sd.surface_eps);
        sdf_iters.push_back(sd.max_iters);
    }

    out.leaf_count = static_cast<uint32_t>(aabbs.size());

    // prim_id cap: each BLAS prim count must fit kPrimBits (instance-high pack).
    if (out.leaf_count > kMaxBlasPrims) {
        throw std::runtime_error(
            "BuildTwoLevelScene: a BLAS mesh exceeds kMaxBlasPrims (1<<20); "
            "rebalance prim_id.cuh kInstanceBits/kPrimBits (named consumer: p16)");
    }
    if (out.leaf_count == 0u) {
        // Empty mesh: no tree, no buffers. Instances of it contribute nothing.
        out.local_bound = collision::AABB{};
        return out;
    }

    // Union of prim AABBs == the mesh's local-space bound.
    collision::AABB bound = aabbs[0];
    for (size_t i = 1; i < aabbs.size(); ++i) {
        bound.Merge(aabbs[i]);
    }
    out.local_bound = bound;

    // Upload local prim buffers (empty kinds get a 1-element dummy, as p13 does).
    out.d_prims = UploadOwned(bt, prims);
    out.d_tri_v0 = UploadOwned(bt, tri_v0.empty() ? std::vector<Vec3>{Vec3{}} : tri_v0);
    out.d_tri_v1 = UploadOwned(bt, tri_v1.empty() ? std::vector<Vec3>{Vec3{}} : tri_v1);
    out.d_tri_v2 = UploadOwned(bt, tri_v2.empty() ? std::vector<Vec3>{Vec3{}} : tri_v2);
    out.d_sph_c = UploadOwned(bt, sph_center.empty() ? std::vector<Vec3>{Vec3{}} : sph_center);
    out.d_sph_r = UploadOwned(bt, sph_radius.empty() ? std::vector<float>{0.0f} : sph_radius);
    out.d_sdf_hdr = UploadOwned(bt, sdf_hdrs.empty() ? std::vector<runtime::sdf::SparseSdfDevice>{runtime::sdf::SparseSdfDevice{}} : sdf_hdrs);
    out.d_sdf_aabb = UploadOwned(bt, sdf_aabbs.empty() ? std::vector<collision::AABB>{collision::AABB{}} : sdf_aabbs);
    out.d_sdf_eps = UploadOwned(bt, sdf_eps.empty() ? std::vector<float>{0.0f} : sdf_eps);
    out.d_sdf_iters = UploadOwned(bt, sdf_iters.empty() ? std::vector<int>{0} : sdf_iters);

    out.view.prims = static_cast<const DevPrim*>(out.d_prims.Data());
    out.view.prim_count = out.leaf_count;
    out.view.tri_v0 = static_cast<const Vec3*>(out.d_tri_v0.Data());
    out.view.tri_v1 = static_cast<const Vec3*>(out.d_tri_v1.Data());
    out.view.tri_v2 = static_cast<const Vec3*>(out.d_tri_v2.Data());
    out.view.sph_center = static_cast<const Vec3*>(out.d_sph_c.Data());
    out.view.sph_radius = static_cast<const float*>(out.d_sph_r.Data());
    out.view.sdf_hdr = static_cast<const runtime::sdf::SparseSdfDevice*>(out.d_sdf_hdr.Data());
    out.view.sdf_aabb = static_cast<const collision::AABB*>(out.d_sdf_aabb.Data());
    out.view.sdf_eps = static_cast<const float*>(out.d_sdf_eps.Data());
    out.view.sdf_iters = static_cast<const int*>(out.d_sdf_iters.Data());

    // Build the BLAS over the LOCAL per-prim AABBs (retain the tree).
    OwnedBuffer d_boxes = UploadOwned(bt, aabbs);
    const auto* dev_boxes = static_cast<const collision::AABB*>(d_boxes.Data());
    out.tree = collision::gpu::BuildLbvhForQuery(ctx.stream, ctx.device_id,
                                                 dev_boxes, out.leaf_count);
    if (!out.tree.HasNodes()) {
        throw std::runtime_error("BuildTwoLevelScene: BLAS LBVH build retained no nodes");
    }
    return out;
}

// Per-frame scene prep shared by RenderFrame + RenderBeauty: build the device
// instance table + the TLAS over the CURRENT instance world-AABBs, upload the
// materials. The retained buffers must outlive the kernel, so they are held in
// `out` (the caller keeps it alive until the stream sync).
struct FrameTlas {
    OwnedBuffer d_instances;
    OwnedBuffer d_mats;
    OwnedBuffer d_tlas_boxes;
    collision::gpu::LbvhBroadphaseResult tlas;
    const DevInstance* instances = nullptr;
    const Material* materials = nullptr;
    const LbvhNode* tlas_nodes = nullptr;
    uint32_t inst_count = 0u;
};

FrameTlas BuildFrameTlas(TwoLevelSceneDevice::Impl* impl, const TwoLevelScene& scene,
                         const RtContext& ctx) {
    FrameTlas out;
    phi::BufferType* bt = ctx.device_bt;
    out.inst_count = static_cast<uint32_t>(scene.instances.size());
    std::vector<collision::AABB> tlas_aabbs(out.inst_count);
    std::vector<DevInstance> dev_instances(out.inst_count);
    for (uint32_t i = 0; i < out.inst_count; ++i) {
        const Instance& I = scene.instances[i];
        if (I.blas_id >= impl->meshes.size()) {
            throw std::runtime_error("RenderFrame: instance.blas_id out of range");
        }
        const BlasDevice& mesh = impl->meshes[I.blas_id];
        DevInstance di;
        di.transform = I.transform;
        di.blas_nodes = (mesh.leaf_count > 0u) ? mesh.tree.DeviceNodes() : nullptr;
        di.blas_leaf_count = mesh.leaf_count;
        di.blas = mesh.view;
        di.instance_id = i;
        di.material_id = I.material_id;
        dev_instances[i] = di;
        if (mesh.leaf_count == 0u) {
            tlas_aabbs[i].min = I.transform.position;
            tlas_aabbs[i].max = I.transform.position;
            continue;
        }
        const Vec3 half{0.5f * (mesh.local_bound.max.x - mesh.local_bound.min.x),
                        0.5f * (mesh.local_bound.max.y - mesh.local_bound.min.y),
                        0.5f * (mesh.local_bound.max.z - mesh.local_bound.min.z)};
        const Vec3 center{0.5f * (mesh.local_bound.min.x + mesh.local_bound.max.x),
                          0.5f * (mesh.local_bound.min.y + mesh.local_bound.max.y),
                          0.5f * (mesh.local_bound.min.z + mesh.local_bound.max.z)};
        Transform centered = I.transform;
        centered.position = I.transform.TransformPoint(center);
        tlas_aabbs[i] = collision::AABB::FromBox(centered, half);
    }
    out.d_instances = UploadOwned(bt, dev_instances);
    out.d_mats = UploadOwned(bt,
        scene.materials.empty() ? std::vector<Material>{Material{}} : scene.materials);
    out.d_tlas_boxes = UploadOwned(bt, tlas_aabbs);
    const auto* dev_tlas_boxes = static_cast<const collision::AABB*>(out.d_tlas_boxes.Data());
    out.tlas = collision::gpu::BuildLbvhForQuery(ctx.stream, ctx.device_id,
                                                 dev_tlas_boxes, out.inst_count);
    if (!out.tlas.HasNodes()) {
        throw std::runtime_error("RenderFrame: TLAS LBVH build retained no nodes");
    }
    out.instances = static_cast<const DevInstance*>(out.d_instances.Data());
    out.materials = static_cast<const Material*>(out.d_mats.Data());
    out.tlas_nodes = out.tlas.DeviceNodes();
    return out;
}

}  // namespace

TwoLevelSceneDevice BuildTwoLevelScene(const TwoLevelScene& scene,
                                       phi::Backend* backend) {
    if (scene.instances.size() > kMaxInstances) {
        throw std::runtime_error(
            "BuildTwoLevelScene: instance_count exceeds kMaxInstances (1<<12); "
            "rebalance prim_id.cuh kInstanceBits/kPrimBits (named consumer: p16)");
    }

    // The BLAS uploads run on the selected device + stream (re-assert the active
    // device, mirroring the backend dispatch path).
    const RtContext ctx = ResolveRtContext(backend);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    (void)cudaSetDevice(ctx.device_id);

    TwoLevelSceneDevice device;
    device.GetImpl()->meshes.reserve(scene.meshes.size());
    for (const auto& mesh : scene.meshes) {
        device.GetImpl()->meshes.push_back(BuildBlas(mesh, ctx));
    }
    cudaStreamSynchronize(ctx.stream);
    return device;
}

namespace {

// AOV destination pointers for ONE frame (raw device pointers). The kernel writes
// these in place; the caller supplies either internal scratch (host-download path)
// or its own resident buffers (RenderFrameToAovs). null skips that channel.
struct AovTarget {
    float* color = nullptr;
    float* depth = nullptr;
    float* normal = nullptr;
    float* albedo = nullptr;
    float* uv = nullptr;
    uint32_t* prim = nullptr;
};

// Build the per-frame TLAS and launch the closest-hit kernel into `dst`, on the
// resolved device + stream. The ONLY thing that varies between the host-download
// and device-resident entries is `dst`; the pixel writes are identical. The
// returned FrameTlas must outlive the caller's stream sync.
FrameTlas LaunchRenderFrame(TwoLevelSceneDevice::Impl* impl,
                            const TwoLevelScene& scene,
                            const PinholeCamera& camera,
                            const RtContext& ctx,
                            const AovTarget& dst) {
    FrameTlas frame = BuildFrameTlas(impl, scene, ctx);
    const dim3 block(kBlockDim, kBlockDim);
    const dim3 grid((camera.width + kBlockDim - 1u) / kBlockDim,
                    (camera.height + kBlockDim - 1u) / kBlockDim);
    RenderFrameKernel<<<grid, block, 0, ctx.stream>>>(
        camera, frame.tlas_nodes, frame.inst_count,
        frame.instances, frame.materials,
        scene.light, scene.ambient,
        dst.color, dst.depth, dst.normal, dst.albedo, dst.uv, dst.prim);
    CheckCuda(cudaGetLastError(), "RenderFrameKernel launch");
    return frame;
}

// Map a caller RtDeviceAovs (opaque phi::Buffer*) to raw device pointers.
AovTarget DeviceAovsToTarget(const RtDeviceAovs& aov) {
    AovTarget dst;
    dst.color = aov.color != nullptr ? static_cast<float*>(phi::BufferBase(aov.color)) : nullptr;
    dst.depth = aov.depth != nullptr ? static_cast<float*>(phi::BufferBase(aov.depth)) : nullptr;
    dst.normal = aov.normal != nullptr ? static_cast<float*>(phi::BufferBase(aov.normal)) : nullptr;
    dst.albedo = aov.albedo != nullptr ? static_cast<float*>(phi::BufferBase(aov.albedo)) : nullptr;
    dst.uv = aov.uv != nullptr ? static_cast<float*>(phi::BufferBase(aov.uv)) : nullptr;
    dst.prim = aov.prim != nullptr ? static_cast<uint32_t*>(phi::BufferBase(aov.prim)) : nullptr;
    return dst;
}

}  // namespace

Framebuffer RenderFrame(TwoLevelSceneDevice& device,
                        const TwoLevelScene& scene,
                        const PinholeCamera& camera,
                        phi::Backend* backend) {
    Framebuffer fb;
    fb.width = camera.width;
    fb.height = camera.height;
    const size_t pixels = fb.Pixels();
    fb.color.assign(pixels * 3u, 0.0f);
    fb.depth.assign(pixels, RtMissDepth());
    fb.normal.assign(pixels * 3u, 0.0f);
    fb.albedo.assign(pixels * 3u, 0.0f);
    fb.uv.assign(pixels * 2u, 0.0f);
    fb.prim.assign(pixels, kNoPrim);
    if (pixels == 0u || scene.instances.empty()) {
        return fb;
    }

    const RtContext ctx = ResolveRtContext(backend);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    (void)cudaSetDevice(ctx.device_id);

    // Internal scratch AOVs, downloaded to the host Framebuffer (one D2H).
    OwnedBuffer d_color(ctx.device_bt, pixels * 3u * sizeof(float));
    OwnedBuffer d_depth(ctx.device_bt, pixels * sizeof(float));
    OwnedBuffer d_normal(ctx.device_bt, pixels * 3u * sizeof(float));
    OwnedBuffer d_albedo(ctx.device_bt, pixels * 3u * sizeof(float));
    OwnedBuffer d_uv(ctx.device_bt, pixels * 2u * sizeof(float));
    OwnedBuffer d_prim(ctx.device_bt, pixels * sizeof(uint32_t));

    AovTarget dst;
    dst.color = static_cast<float*>(d_color.Data());
    dst.depth = static_cast<float*>(d_depth.Data());
    dst.normal = static_cast<float*>(d_normal.Data());
    dst.albedo = static_cast<float*>(d_albedo.Data());
    dst.uv = static_cast<float*>(d_uv.Data());
    dst.prim = static_cast<uint32_t*>(d_prim.Data());

    FrameTlas frame = LaunchRenderFrame(device.GetImpl(), scene, camera, ctx, dst);
    (void)frame;
    cudaStreamSynchronize(ctx.stream);

    d_color.CopyToHost(fb.color.data(), pixels * 3u * sizeof(float));
    d_depth.CopyToHost(fb.depth.data(), pixels * sizeof(float));
    d_normal.CopyToHost(fb.normal.data(), pixels * 3u * sizeof(float));
    d_albedo.CopyToHost(fb.albedo.data(), pixels * 3u * sizeof(float));
    d_uv.CopyToHost(fb.uv.data(), pixels * 2u * sizeof(float));
    d_prim.CopyToHost(fb.prim.data(), pixels * sizeof(uint32_t));
    return fb;
}

void RenderFrameToAovs(TwoLevelSceneDevice& device,
                       const TwoLevelScene& scene,
                       const PinholeCamera& camera,
                       const RtDeviceAovs& aov,
                       phi::Backend* backend) {
    const size_t pixels =
        static_cast<size_t>(camera.width) * static_cast<size_t>(camera.height);
    if (pixels == 0u || scene.instances.empty()) {
        return;
    }
    const RtContext ctx = ResolveRtContext(backend);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    (void)cudaSetDevice(ctx.device_id);

    // Write directly into the caller's resident device buffers -- no host round-trip.
    const AovTarget dst = DeviceAovsToTarget(aov);
    FrameTlas frame = LaunchRenderFrame(device.GetImpl(), scene, camera, ctx, dst);
    (void)frame;
    cudaStreamSynchronize(ctx.stream);
}

namespace {

// Build the per-frame TLAS and launch the stochastic beauty kernel into `dst`, on
// the resolved device + stream. Only `dst` varies between the host-download and
// device-resident entries; the pixel writes are identical. The returned FrameTlas
// must outlive the caller's stream sync.
FrameTlas LaunchRenderBeauty(TwoLevelSceneDevice::Impl* impl,
                             const TwoLevelScene& scene,
                             const PinholeCamera& camera,
                             const BeautyOptions& opt,
                             const RtContext& ctx,
                             const AovTarget& dst) {
    FrameTlas frame = BuildFrameTlas(impl, scene, ctx);

    BeautyParams sky;
    sky.shadow_rays = opt.shadow_rays;
    sky.sun_angular_radius = opt.sun_angular_radius;
    sky.gi_bounces = opt.gi_bounces;
    sky.ao_samples = opt.ao_samples;
    sky.ao_radius = opt.ao_radius;
    sky.sky_top = opt.sky_top;
    sky.sky_bottom = opt.sky_bottom;
    sky.sky_ground = opt.sky_ground;
    sky.fog_color = opt.fog_color;
    sky.fog_density = opt.fog_density;
    sky.sky_intensity = opt.sky_intensity;

    const dim3 block(kBlockDim, kBlockDim);
    const dim3 grid((camera.width + kBlockDim - 1u) / kBlockDim,
                    (camera.height + kBlockDim - 1u) / kBlockDim);
    RenderBeautyKernel<<<grid, block, 0, ctx.stream>>>(
        camera, frame.tlas_nodes, frame.inst_count, frame.instances, frame.materials,
        scene.light, sky, opt.samples, opt.seed,
        dst.color, dst.depth, dst.normal, dst.albedo, dst.uv, dst.prim);
    CheckCuda(cudaGetLastError(), "RenderBeautyKernel launch");
    return frame;
}

}  // namespace

Framebuffer RenderBeauty(TwoLevelSceneDevice& device,
                         const TwoLevelScene& scene,
                         const PinholeCamera& camera,
                         const BeautyOptions& opt,
                         phi::Backend* backend) {
    Framebuffer fb;
    fb.width = camera.width;
    fb.height = camera.height;
    const size_t pixels = fb.Pixels();
    fb.color.assign(pixels * 3u, 0.0f);
    fb.depth.assign(pixels, RtMissDepth());
    fb.normal.assign(pixels * 3u, 0.0f);
    fb.albedo.assign(pixels * 3u, 0.0f);
    fb.uv.assign(pixels * 2u, 0.0f);
    fb.prim.assign(pixels, kNoPrim);
    if (pixels == 0u || scene.instances.empty()) {
        return fb;
    }

    const RtContext ctx = ResolveRtContext(backend);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    (void)cudaSetDevice(ctx.device_id);

    OwnedBuffer d_color(ctx.device_bt, pixels * 3u * sizeof(float));
    OwnedBuffer d_depth(ctx.device_bt, pixels * sizeof(float));
    OwnedBuffer d_normal(ctx.device_bt, pixels * 3u * sizeof(float));
    OwnedBuffer d_albedo(ctx.device_bt, pixels * 3u * sizeof(float));
    OwnedBuffer d_uv(ctx.device_bt, pixels * 2u * sizeof(float));
    OwnedBuffer d_prim(ctx.device_bt, pixels * sizeof(uint32_t));

    AovTarget dst;
    dst.color = static_cast<float*>(d_color.Data());
    dst.depth = static_cast<float*>(d_depth.Data());
    dst.normal = static_cast<float*>(d_normal.Data());
    dst.albedo = static_cast<float*>(d_albedo.Data());
    dst.uv = static_cast<float*>(d_uv.Data());
    dst.prim = static_cast<uint32_t*>(d_prim.Data());

    FrameTlas frame = LaunchRenderBeauty(device.GetImpl(), scene, camera, opt, ctx, dst);
    (void)frame;
    cudaStreamSynchronize(ctx.stream);

    d_color.CopyToHost(fb.color.data(), pixels * 3u * sizeof(float));
    d_depth.CopyToHost(fb.depth.data(), pixels * sizeof(float));
    d_normal.CopyToHost(fb.normal.data(), pixels * 3u * sizeof(float));
    d_albedo.CopyToHost(fb.albedo.data(), pixels * 3u * sizeof(float));
    d_uv.CopyToHost(fb.uv.data(), pixels * 2u * sizeof(float));
    d_prim.CopyToHost(fb.prim.data(), pixels * sizeof(uint32_t));
    return fb;
}

void RenderBeautyToAovs(TwoLevelSceneDevice& device,
                        const TwoLevelScene& scene,
                        const PinholeCamera& camera,
                        const BeautyOptions& opt,
                        const RtDeviceAovs& aov,
                        phi::Backend* backend) {
    const size_t pixels =
        static_cast<size_t>(camera.width) * static_cast<size_t>(camera.height);
    if (pixels == 0u || scene.instances.empty()) {
        return;
    }
    const RtContext ctx = ResolveRtContext(backend);
    phi::ScopedDeviceGuard guard(ctx.device_id);
    (void)cudaSetDevice(ctx.device_id);

    const AovTarget dst = DeviceAovsToTarget(aov);
    FrameTlas frame = LaunchRenderBeauty(device.GetImpl(), scene, camera, opt, ctx, dst);
    (void)frame;
    cudaStreamSynchronize(ctx.stream);
}

}  // namespace nuka::rt
