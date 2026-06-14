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
#include "phi/backend.hpp"             // InitBestDevice / DeviceBufferType
#include "phi/buffer.hpp"              // Buffer* / BufferAlloc / BufferUpload / ...
#include "phi/buffer_transfer_v2.hpp"  // UploadVectorV2
#include "phi/device_context.hpp"
#include "rt/bvh_traverse_impl.cuh"
#include "rt/camera.hpp"
#include "rt/instance_transform.cuh"
#include "rt/intersect_primitives.cuh"
#include "rt/prim_id.cuh"
#include "rt/ray_box.cuh"
#include "rt/shading.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::rt {

namespace {

constexpr uint32_t kBlockDim = 16u;

// RT-7 (M11): file-local RAII over the phi-v2 opaque Buffer*, mirroring the legacy
// phi::Buffer surface (default-ctor + sized-ctor + Data/CopyFromHost/CopyToHost,
// move-only) so the render bodies + the GpuScene buffer-holding struct below stay
// byte-identical after the BUF sweep. Bound to the stream-0
// DeviceBufferType (NULL/default stream) -> transfers are byte+ordering identical to
// the legacy synchronous memcpy. The RT kernels keep running on their existing
// stream UNCHANGED; this is the buffer-handle swap ONLY.
class OwnedBuffer {
public:
    OwnedBuffer() = default;
    explicit OwnedBuffer(size_t bytes) {
        buf_ = phi::BufferAlloc(phi::DeviceBufferType(phi::InitBestDevice()), bytes);
    }
    explicit OwnedBuffer(phi::Buffer* buf) : buf_(buf) {}
    ~OwnedBuffer() { if (buf_ != nullptr) phi::BufferFree(buf_); }
    OwnedBuffer(OwnedBuffer&& o) noexcept : buf_(o.buf_) { o.buf_ = nullptr; }
    OwnedBuffer& operator=(OwnedBuffer&& o) noexcept {
        if (this != &o) {
            if (buf_ != nullptr) phi::BufferFree(buf_);
            buf_ = o.buf_; o.buf_ = nullptr;
        }
        return *this;
    }
    OwnedBuffer(const OwnedBuffer&) = delete;
    OwnedBuffer& operator=(const OwnedBuffer&) = delete;
    void* Data() const { return buf_ != nullptr ? phi::BufferBase(buf_) : nullptr; }
    // (No CopyFromHost: this TU's uploads all go through UploadOwned; only the
    // AOV download path uses CopyToHost.)
    void CopyToHost(void* dst, size_t bytes) const {
        if (buf_ != nullptr && bytes > 0u) phi::BufferDownload(buf_, dst, 0, bytes);
    }
private:
    phi::Buffer* buf_ = nullptr;
};

template <typename T>
OwnedBuffer UploadOwned(const std::vector<T>& values) {
    return OwnedBuffer(phi::UploadVectorV2(
        phi::DeviceBufferType(phi::InitBestDevice()), values));
}

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
BlasDevice BuildBlas(const BlasMesh& mesh) {
    BlasDevice out;

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

        out.sdf_keys_bufs.push_back(UploadOwned(sd.keys));
        out.sdf_vals_bufs.push_back(UploadOwned(sd.values));
        out.sdf_grad_bufs.push_back(UploadOwned(sd.gradients));

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
    out.d_prims = UploadOwned(prims);
    out.d_tri_v0 = UploadOwned(tri_v0.empty() ? std::vector<Vec3>{Vec3{}} : tri_v0);
    out.d_tri_v1 = UploadOwned(tri_v1.empty() ? std::vector<Vec3>{Vec3{}} : tri_v1);
    out.d_tri_v2 = UploadOwned(tri_v2.empty() ? std::vector<Vec3>{Vec3{}} : tri_v2);
    out.d_sph_c = UploadOwned(sph_center.empty() ? std::vector<Vec3>{Vec3{}} : sph_center);
    out.d_sph_r = UploadOwned(sph_radius.empty() ? std::vector<float>{0.0f} : sph_radius);
    out.d_sdf_hdr = UploadOwned(sdf_hdrs.empty() ? std::vector<runtime::sdf::SparseSdfDevice>{runtime::sdf::SparseSdfDevice{}} : sdf_hdrs);
    out.d_sdf_aabb = UploadOwned(sdf_aabbs.empty() ? std::vector<collision::AABB>{collision::AABB{}} : sdf_aabbs);
    out.d_sdf_eps = UploadOwned(sdf_eps.empty() ? std::vector<float>{0.0f} : sdf_eps);
    out.d_sdf_iters = UploadOwned(sdf_iters.empty() ? std::vector<int>{0} : sdf_iters);

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
    OwnedBuffer d_boxes = UploadOwned(aabbs);
    const auto* dev_boxes = static_cast<const collision::AABB*>(d_boxes.Data());
    out.tree = collision::gpu::BuildLbvhForQuery(dev_boxes, out.leaf_count);
    if (!out.tree.HasNodes()) {
        throw std::runtime_error("BuildTwoLevelScene: BLAS LBVH build retained no nodes");
    }
    return out;
}

}  // namespace

TwoLevelSceneDevice BuildTwoLevelScene(const TwoLevelScene& scene) {
    if (scene.instances.size() > kMaxInstances) {
        throw std::runtime_error(
            "BuildTwoLevelScene: instance_count exceeds kMaxInstances (1<<12); "
            "rebalance prim_id.cuh kInstanceBits/kPrimBits (named consumer: p16)");
    }

    // Ensure the device context is live (the BLAS uploads need a device + stream).
    auto context = phi::MakeDefaultDeviceContext();
    phi::ScopedDeviceGuard guard(context.device_id);

    TwoLevelSceneDevice device;
    device.GetImpl()->meshes.reserve(scene.meshes.size());
    for (const auto& mesh : scene.meshes) {
        device.GetImpl()->meshes.push_back(BuildBlas(mesh));
    }
    context.stream.Synchronize();
    return device;
}

Framebuffer RenderFrame(TwoLevelSceneDevice& device,
                        const TwoLevelScene& scene,
                        const PinholeCamera& camera) {
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

    auto context = phi::MakeDefaultDeviceContext();
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();

    TwoLevelSceneDevice::Impl* impl = device.GetImpl();
    const uint32_t inst_count = static_cast<uint32_t>(scene.instances.size());

    // Build the per-frame TLAS world-AABBs: transform each mesh's oriented LOCAL
    // box by the instance pose, then enclose it (p14b pattern). EVERY instance
    // gets a box so the TLAS leaf's original index IS the instance index.
    std::vector<collision::AABB> tlas_aabbs(inst_count);
    std::vector<DevInstance> dev_instances(inst_count);
    for (uint32_t i = 0; i < inst_count; ++i) {
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
            // Degenerate point box at the instance origin (its BLAS is empty;
            // the TlasLeaf early-outs on blas_leaf_count==0 so it is never hit).
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
        // Transform the box CENTER by the pose, then enclose the oriented half-
        // extent box about it (collision::AABB::FromBox uses TransformPoint).
        Transform centered = I.transform;
        centered.position = I.transform.TransformPoint(center);
        tlas_aabbs[i] = collision::AABB::FromBox(centered, half);
    }

    // Upload the instance table + the materials.
    OwnedBuffer d_instances = UploadOwned(dev_instances);
    OwnedBuffer d_mats = UploadOwned(
        scene.materials.empty() ? std::vector<Material>{Material{}} : scene.materials);

    // Build the TLAS over the instance world-AABBs (retained; lives until sync).
    OwnedBuffer d_tlas_boxes = UploadOwned(tlas_aabbs);
    const auto* dev_tlas_boxes = static_cast<const collision::AABB*>(d_tlas_boxes.Data());
    auto tlas = collision::gpu::BuildLbvhForQuery(dev_tlas_boxes, inst_count);
    if (!tlas.HasNodes()) {
        throw std::runtime_error("RenderFrame: TLAS LBVH build retained no nodes");
    }
    const LbvhNode* tlas_nodes = tlas.DeviceNodes();

    // Output AOV buffers (device).
    OwnedBuffer d_color(pixels * 3u * sizeof(float));
    OwnedBuffer d_depth(pixels * sizeof(float));
    OwnedBuffer d_normal(pixels * 3u * sizeof(float));
    OwnedBuffer d_albedo(pixels * 3u * sizeof(float));
    OwnedBuffer d_uv(pixels * 2u * sizeof(float));
    OwnedBuffer d_prim(pixels * sizeof(uint32_t));

    const dim3 block(kBlockDim, kBlockDim);
    const dim3 grid((camera.width + kBlockDim - 1u) / kBlockDim,
                    (camera.height + kBlockDim - 1u) / kBlockDim);
    RenderFrameKernel<<<grid, block, 0, stream>>>(
        camera, tlas_nodes, inst_count,
        static_cast<const DevInstance*>(d_instances.Data()),
        static_cast<const Material*>(d_mats.Data()),
        scene.light, scene.ambient,
        static_cast<float*>(d_color.Data()),
        static_cast<float*>(d_depth.Data()),
        static_cast<float*>(d_normal.Data()),
        static_cast<float*>(d_albedo.Data()),
        static_cast<float*>(d_uv.Data()),
        static_cast<uint32_t*>(d_prim.Data()));
    CheckCuda(cudaGetLastError(), "RenderFrameKernel launch");
    context.stream.Synchronize();

    d_color.CopyToHost(fb.color.data(), pixels * 3u * sizeof(float));
    d_depth.CopyToHost(fb.depth.data(), pixels * sizeof(float));
    d_normal.CopyToHost(fb.normal.data(), pixels * 3u * sizeof(float));
    d_albedo.CopyToHost(fb.albedo.data(), pixels * 3u * sizeof(float));
    d_uv.CopyToHost(fb.uv.data(), pixels * 2u * sizeof(float));
    d_prim.CopyToHost(fb.prim.data(), pixels * sizeof(uint32_t));
    return fb;
}

}  // namespace nuka::rt
