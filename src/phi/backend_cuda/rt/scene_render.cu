// ---------------------------------------------------------------------------
// nuka::rt::RenderScene -- the SENSOR renderer (v0.7 p13). Camera -> LBVH ->
// closest-hit per-primitive intersection (triangle / sphere / sparse-SDF) ->
// Lambert+GGX direct shade + hard shadow ray -> full AOV framebuffer.
//
// REUSES the p12 stackless traversal UNCHANGED (rt::TraverseRay,
// bvh_traverse_impl.cuh) for BOTH the primary closest-hit ray AND the secondary
// hard-shadow ray; the ONLY p13-specific code is the generalized leaf functor
// (SceneLeaf, dispatching on PrimKind), one analytic light, attribute
// reconstruction for the winning prim, and the widened framebuffer.
//
// ZERO atomics (one writer per pixel, sequential best_t/best_prim) -> D1
// byte-exact across ALL AOVs. fp64-internal intersections + --fmad=false
// (target flag) keep depth/normal/bary bit-exact vs the CPU oracle.
// ---------------------------------------------------------------------------

#include "rt/scene_render.hpp"

#include "collision/aabb.hpp"
#include "collision/broadphase_lbvh.hpp"
#include "collision/lbvh_node.cuh"
#include "math/vec3.hpp"
#include "phi/backend.hpp"             // InitBestDevice / DeviceBufferType
#include "phi/buffer.hpp"              // Buffer* / BufferAlloc / BufferUpload / ...
#include "phi/buffer_transfer_v2.hpp"  // UploadVectorV2
#include "phi/scoped_device_guard.hpp"
#include "phi/backend_cuda/rt/bvh_traverse_impl.cuh"
#include "phi/backend_cuda/rt/intersect_primitives.cuh"
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

// RT-7 (M11): file-local RAII over the phi-v2 opaque Buffer*, mirroring the legacy
// phi::Buffer surface (default-ctor + sized-ctor + Data/CopyFromHost/CopyToHost,
// move-only) so the render bodies below stay byte-identical after the BUF
// sweep. Bound to the stream-0 DeviceBufferType (NULL/default stream) -> transfers
// are byte+ordering identical to the legacy synchronous memcpy. The RT kernels keep
// running on their existing stream UNCHANGED; this is the buffer-handle swap ONLY.
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

// Flat per-primitive record indexed by the LBVH leaf's original prim id. `kind`
// selects the intersection; `sub` indexes into the per-kind device arrays.
struct DevPrim {
    uint32_t kind;        // PrimKind
    uint32_t sub;         // index into the per-kind array
    uint32_t material_id;
};

// Device view of the whole scene (raw pointers into uploaded buffers).
struct DevScene {
    const DevPrim* prims = nullptr;
    uint32_t prim_count = 0u;

    // Triangle data (parallel arrays).
    const math::Vec3* tri_v0 = nullptr;
    const math::Vec3* tri_v1 = nullptr;
    const math::Vec3* tri_v2 = nullptr;

    // Sphere data.
    const math::Vec3* sph_center = nullptr;
    const float* sph_radius = nullptr;

    // SDF data: one SparseSdfDevice header per sdf (pointers already device-side)
    // + a parallel world AABB + per-sdf trace params.
    const runtime::sdf::SparseSdfDevice* sdf_hdr = nullptr;
    const collision::AABB* sdf_aabb = nullptr;
    const float* sdf_eps = nullptr;
    const int* sdf_iters = nullptr;

    const Material* materials = nullptr;
};

// Intersect a single primitive (by prim id) for the CLOSEST-HIT pass: returns t
// only (attribute reconstruction is deferred to the winner). t_min = 0 for the
// primary ray. Mirrors the oracle's per-prim intersect exactly (shared fns).
__device__ __forceinline__ bool IntersectPrimT(const DevScene& s,
                                               uint32_t prim,
                                               const math::Vec3& origin,
                                               const math::Vec3& dir,
                                               float t_min,
                                               float* out_t) {
    const DevPrim p = s.prims[prim];
    if (p.kind == static_cast<uint32_t>(PrimKind::Triangle)) {
        float u, v;
        math::Vec3 n;
        return RayTriangleIntersect(origin, dir, s.tri_v0[p.sub], s.tri_v1[p.sub],
                                    s.tri_v2[p.sub], t_min, out_t, &u, &v, &n);
    } else if (p.kind == static_cast<uint32_t>(PrimKind::Sphere)) {
        math::Vec3 n;
        float u, v;
        return RaySphereIntersect(origin, dir, s.sph_center[p.sub],
                                  s.sph_radius[p.sub], t_min, out_t, &n, &u, &v);
    } else { // Sdf
        math::Vec3 n;
        return RaySdfSphereTrace(origin, dir, s.sdf_hdr[p.sub], s.sdf_aabb[p.sub],
                                 t_min, s.sdf_eps[p.sub], s.sdf_iters[p.sub],
                                 out_t, &n);
    }
}

// Closest-hit leaf functor for the generic scene: reads the leaf's original prim
// id (nodes[leaf_node].left), intersects the real primitive, and updates the
// closest hit (lowest-index tie-break). t_min is captured so shadow rays can
// offset past the origin surface.
struct SceneLeaf {
    const LbvhNode* __restrict__ nodes;
    DevScene scene;
    float t_min;
    __device__ void operator()(int32_t leaf_node,
                               const math::Vec3& origin,
                               const math::Vec3& dir,
                               float* best_t,
                               uint32_t* best_prim) const {
        const uint32_t prim = static_cast<uint32_t>(nodes[leaf_node].left);
        float t;
        if (IntersectPrimT(scene, prim, origin, dir, t_min, &t)) {
            if (t >= t_min) {
                RtClosestHitUpdate(t, prim, best_t, best_prim);
            }
        }
    }
};

// Run a closest-hit traversal (primary or shadow). Returns best_t / best_prim.
__device__ __forceinline__ void ClosestHit(const LbvhNode* __restrict__ nodes,
                                           uint32_t leaf_count,
                                           const DevScene& scene,
                                           const math::Vec3& origin,
                                           const math::Vec3& dir,
                                           float t_min,
                                           float* best_t,
                                           uint32_t* best_prim) {
    *best_t = RtMissDepth();
    *best_prim = kNoPrim;
    SceneLeaf leaf{nodes, scene, t_min};
    if (leaf_count == 1u) {
        leaf(0, origin, dir, best_t, best_prim);
    } else if (leaf_count >= 2u) {
        TraverseRay(nodes, leaf_count - 1u, origin, dir, best_t, best_prim, leaf);
    }
}

// Reconstruct surface attributes (normal, uv, albedo material) for the WINNING
// prim only (one eval, OUT of the descent -- keeps traversal lean + byte-stable).
__device__ __forceinline__ void ReconstructHit(const DevScene& s,
                                               uint32_t prim,
                                               const math::Vec3& origin,
                                               const math::Vec3& dir,
                                               float t,
                                               math::Vec3* out_normal,
                                               float* out_u,
                                               float* out_v,
                                               uint32_t* out_material) {
    const DevPrim p = s.prims[prim];
    *out_material = p.material_id;
    if (p.kind == static_cast<uint32_t>(PrimKind::Triangle)) {
        float tt, u, v;
        math::Vec3 n;
        RayTriangleIntersect(origin, dir, s.tri_v0[p.sub], s.tri_v1[p.sub],
                             s.tri_v2[p.sub], 0.0f, &tt, &u, &v, &n);
        *out_normal = n;
        *out_u = u;
        *out_v = v;
    } else if (p.kind == static_cast<uint32_t>(PrimKind::Sphere)) {
        float tt, u, v;
        math::Vec3 n;
        RaySphereIntersect(origin, dir, s.sph_center[p.sub], s.sph_radius[p.sub],
                           0.0f, &tt, &n, &u, &v);
        *out_normal = n;
        *out_u = u;
        *out_v = v;
    } else { // Sdf: normal from gradient at the hit point; uv = 0 (no param).
        float tt;
        math::Vec3 n;
        RaySdfSphereTrace(origin, dir, s.sdf_hdr[p.sub], s.sdf_aabb[p.sub], 0.0f,
                          s.sdf_eps[p.sub], s.sdf_iters[p.sub], &tt, &n);
        *out_normal = n;
        *out_u = 0.0f;
        *out_v = 0.0f;
    }
}

// One thread per pixel: primary closest hit -> reconstruct -> shadow ray -> shade
// -> write ALL AOVs. Exactly one writer per pixel; no atomics -> D1 byte-exact.
__global__ void RenderSceneKernel(PinholeCamera camera,
                                  const LbvhNode* __restrict__ nodes,
                                  uint32_t leaf_count,
                                  DevScene scene,
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
    ClosestHit(nodes, leaf_count, scene, ray.origin, ray.dir, 0.0f, &best_t,
               &best_prim);

    // Defaults (miss): black color, sentinel depth, zero aux, sentinel prim.
    math::Vec3 color{0.0f, 0.0f, 0.0f};
    float depth = RtMissDepth();
    math::Vec3 normal{0.0f, 0.0f, 0.0f};
    math::Vec3 albedo{0.0f, 0.0f, 0.0f};
    float uv_u = 0.0f, uv_v = 0.0f;
    uint32_t prim_id = kNoPrim;

    if (best_prim != kNoPrim) {
        depth = best_t;
        prim_id = best_prim;

        math::Vec3 n;
        uint32_t material_id;
        ReconstructHit(scene, best_prim, ray.origin, ray.dir, best_t, &n, &uv_u,
                       &uv_v, &material_id);
        normal = n;
        const Material mat = scene.materials[material_id];
        albedo = mat.albedo;

        // Hit point p = origin + t*dir (fp64).
        const math::Vec3 hit{
            static_cast<float>(static_cast<double>(ray.origin.x) + static_cast<double>(best_t) * static_cast<double>(ray.dir.x)),
            static_cast<float>(static_cast<double>(ray.origin.y) + static_cast<double>(best_t) * static_cast<double>(ray.dir.y)),
            static_cast<float>(static_cast<double>(ray.origin.z) + static_cast<double>(best_t) * static_cast<double>(ray.dir.z))};

        // View dir (toward camera) and light dir (toward light).
        const math::Vec3 V = RtNormalize(math::Vec3{-ray.dir.x, -ray.dir.y, -ray.dir.z});
        math::Vec3 L;
        float light_dist;
        if (light.directional) {
            L = RtNormalize(math::Vec3{-light.direction.x, -light.direction.y, -light.direction.z});
            light_dist = RtMissDepth(); // infinite
        } else {
            const math::Vec3 to{light.position.x - hit.x, light.position.y - hit.y,
                                light.position.z - hit.z};
            L = RtNormalize(to);
            light_dist = static_cast<float>(sqrt(
                static_cast<double>(to.x) * to.x +
                static_cast<double>(to.y) * to.y +
                static_cast<double>(to.z) * to.z));
        }

        // Hard shadow ray: origin offset along the (viewer-faced) normal to avoid
        // self-intersection acne; t_min epsilon. The NaN guard (RtSafeInvDir)
        // handles an axis-parallel shadow ray landing on a slab plane.
        math::Vec3 ns = n;
        const double nv = static_cast<double>(n.x) * V.x + static_cast<double>(n.y) * V.y +
                          static_cast<double>(n.z) * V.z;
        if (nv < 0.0) {
            ns = math::Vec3{-n.x, -n.y, -n.z};
        }
        const float shadow_eps = 1.0e-3f;
        const math::Vec3 sorigin{hit.x + ns.x * shadow_eps, hit.y + ns.y * shadow_eps,
                                 hit.z + ns.z * shadow_eps};

        int lit = 1;
        float st;
        uint32_t sp;
        ClosestHit(nodes, leaf_count, scene, sorigin, L, shadow_eps, &st, &sp);
        if (sp != kNoPrim && st < light_dist - shadow_eps) {
            lit = 0; // an occluder between the hit point and the light
        }

        const math::Vec3 light_col{light.color.x * light.intensity,
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

// Build the per-prim AABB for the LBVH (primitive-agnostic bounds).
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

} // namespace

Framebuffer RenderScene(const Scene& scene, const PinholeCamera& camera) {
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
    if (pixels == 0u) {
        return fb;
    }

    const cudaStream_t stream = nullptr;  // default stream 0
    const int device_id = 0;
    phi::ScopedDeviceGuard guard(device_id);

    // Build the flat prim table + per-prim AABBs in declaration order:
    // triangles, then spheres, then SDFs.
    std::vector<collision::AABB> aabbs;
    std::vector<DevPrim> prims;

    std::vector<math::Vec3> tri_v0, tri_v1, tri_v2;
    for (uint32_t i = 0; i < scene.triangles.size(); ++i) {
        const auto& t = scene.triangles[i];
        aabbs.push_back(TriAabb(t));
        prims.push_back({static_cast<uint32_t>(PrimKind::Triangle),
                         static_cast<uint32_t>(tri_v0.size()), t.material_id});
        tri_v0.push_back(t.v0);
        tri_v1.push_back(t.v1);
        tri_v2.push_back(t.v2);
    }

    std::vector<math::Vec3> sph_center;
    std::vector<float> sph_radius;
    for (uint32_t i = 0; i < scene.spheres.size(); ++i) {
        const auto& s = scene.spheres[i];
        aabbs.push_back(SphAabb(s));
        prims.push_back({static_cast<uint32_t>(PrimKind::Sphere),
                         static_cast<uint32_t>(sph_center.size()), s.material_id});
        sph_center.push_back(s.center);
        sph_radius.push_back(s.radius);
    }

    // SDFs: upload each cooked SDF's cell arrays, build a device header whose
    // pointers point at the uploaded buffers. Keep the device buffers alive for
    // the kernel launch.
    std::vector<OwnedBuffer> sdf_keys_bufs, sdf_vals_bufs, sdf_grad_bufs;
    std::vector<runtime::sdf::SparseSdfDevice> sdf_hdrs;
    std::vector<collision::AABB> sdf_aabbs;
    std::vector<float> sdf_eps;
    std::vector<int> sdf_iters;
    for (uint32_t i = 0; i < scene.sdfs.size(); ++i) {
        const auto& sd = scene.sdfs[i];
        aabbs.push_back(sd.aabb);
        prims.push_back({static_cast<uint32_t>(PrimKind::Sdf),
                         static_cast<uint32_t>(sdf_hdrs.size()), sd.material_id});

        sdf_keys_bufs.push_back(UploadOwned(sd.keys));
        sdf_vals_bufs.push_back(UploadOwned(sd.values));
        sdf_grad_bufs.push_back(UploadOwned(sd.gradients));

        runtime::sdf::SparseSdfDevice hdr = sd.header;
        hdr.cell_keys = static_cast<const uint64_t*>(sdf_keys_bufs.back().Data());
        hdr.cell_values = static_cast<const float*>(sdf_vals_bufs.back().Data());
        hdr.cell_gradients = static_cast<const math::Vec3*>(sdf_grad_bufs.back().Data());
        hdr.cell_count = static_cast<uint32_t>(sd.values.size());
        sdf_hdrs.push_back(hdr);
        sdf_aabbs.push_back(sd.aabb);
        sdf_eps.push_back(sd.surface_eps);
        sdf_iters.push_back(sd.max_iters);
    }

    const uint32_t leaf_count = static_cast<uint32_t>(aabbs.size());
    if (leaf_count == 0u) {
        // No geometry: every pixel misses. Upload the host sentinels.
        return fb;
    }

    // Upload geometry buffers.
    OwnedBuffer d_prims = UploadOwned(prims);
    OwnedBuffer d_tri_v0 = UploadOwned(tri_v0.empty() ? std::vector<math::Vec3>{math::Vec3{}} : tri_v0);
    OwnedBuffer d_tri_v1 = UploadOwned(tri_v1.empty() ? std::vector<math::Vec3>{math::Vec3{}} : tri_v1);
    OwnedBuffer d_tri_v2 = UploadOwned(tri_v2.empty() ? std::vector<math::Vec3>{math::Vec3{}} : tri_v2);
    OwnedBuffer d_sph_c = UploadOwned(sph_center.empty() ? std::vector<math::Vec3>{math::Vec3{}} : sph_center);
    OwnedBuffer d_sph_r = UploadOwned(sph_radius.empty() ? std::vector<float>{0.0f} : sph_radius);
    OwnedBuffer d_sdf_hdr = UploadOwned(sdf_hdrs.empty() ? std::vector<runtime::sdf::SparseSdfDevice>{runtime::sdf::SparseSdfDevice{}} : sdf_hdrs);
    OwnedBuffer d_sdf_aabb = UploadOwned(sdf_aabbs.empty() ? std::vector<collision::AABB>{collision::AABB{}} : sdf_aabbs);
    OwnedBuffer d_sdf_eps = UploadOwned(sdf_eps.empty() ? std::vector<float>{0.0f} : sdf_eps);
    OwnedBuffer d_sdf_iters = UploadOwned(sdf_iters.empty() ? std::vector<int>{0} : sdf_iters);
    OwnedBuffer d_mats = UploadOwned(scene.materials.empty() ? std::vector<Material>{Material{}} : scene.materials);

    DevScene dev;
    dev.prims = static_cast<const DevPrim*>(d_prims.Data());
    dev.prim_count = leaf_count;
    dev.tri_v0 = static_cast<const math::Vec3*>(d_tri_v0.Data());
    dev.tri_v1 = static_cast<const math::Vec3*>(d_tri_v1.Data());
    dev.tri_v2 = static_cast<const math::Vec3*>(d_tri_v2.Data());
    dev.sph_center = static_cast<const math::Vec3*>(d_sph_c.Data());
    dev.sph_radius = static_cast<const float*>(d_sph_r.Data());
    dev.sdf_hdr = static_cast<const runtime::sdf::SparseSdfDevice*>(d_sdf_hdr.Data());
    dev.sdf_aabb = static_cast<const collision::AABB*>(d_sdf_aabb.Data());
    dev.sdf_eps = static_cast<const float*>(d_sdf_eps.Data());
    dev.sdf_iters = static_cast<const int*>(d_sdf_iters.Data());
    dev.materials = static_cast<const Material*>(d_mats.Data());

    // Build the LBVH over the per-prim AABBs (retain nodes).
    OwnedBuffer d_boxes = UploadOwned(aabbs);
    const auto* dev_boxes = static_cast<const collision::AABB*>(d_boxes.Data());
    auto tree = collision::gpu::BuildLbvhForQuery(dev_boxes, leaf_count);
    if (!tree.HasNodes()) {
        throw std::runtime_error("RenderScene: LBVH build did not retain nodes");
    }
    const LbvhNode* nodes = tree.DeviceNodes();

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
    RenderSceneKernel<<<grid, block, 0, stream>>>(
        camera, nodes, leaf_count, dev, scene.light, scene.ambient,
        static_cast<float*>(d_color.Data()),
        static_cast<float*>(d_depth.Data()),
        static_cast<float*>(d_normal.Data()),
        static_cast<float*>(d_albedo.Data()),
        static_cast<float*>(d_uv.Data()),
        static_cast<uint32_t*>(d_prim.Data()));
    CheckCuda(cudaGetLastError(), "RenderSceneKernel launch");
    cudaStreamSynchronize(stream);

    d_color.CopyToHost(fb.color.data(), pixels * 3u * sizeof(float));
    d_depth.CopyToHost(fb.depth.data(), pixels * sizeof(float));
    d_normal.CopyToHost(fb.normal.data(), pixels * 3u * sizeof(float));
    d_albedo.CopyToHost(fb.albedo.data(), pixels * 3u * sizeof(float));
    d_uv.CopyToHost(fb.uv.data(), pixels * 2u * sizeof(float));
    d_prim.CopyToHost(fb.prim.data(), pixels * sizeof(uint32_t));
    return fb;
}

} // namespace nuka::rt
