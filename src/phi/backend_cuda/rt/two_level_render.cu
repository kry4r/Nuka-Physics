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
#include "collision/lbvh_refit.cuh"
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
#include "phi/backend_cuda/rt/render_timing.hpp"
#include "phi/backend_cuda/rt/sensor_scatter.hpp"  // SensorBlasRef (batched-path accessor)
#include "phi/backend_cuda/rt/shading.cuh"
#include "phi/backend_cuda/rt/two_level_render_kernels.cuh"  // shared device structs + fns
#include "rt/camera.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::rt {

// OPT-IN timing probe state (default off -> byte-exact untouched path). The probe
// brackets the TLAS update and the kernel with CUDA events on the render stream.
bool g_render_timing_enabled = false;
RenderTiming g_render_timing;

void SetRenderTimingEnabled(bool enabled) { g_render_timing_enabled = enabled; }
bool RenderTimingEnabled() { return g_render_timing_enabled; }
RenderTiming TakeRenderTiming() {
    RenderTiming out = g_render_timing;
    g_render_timing = RenderTiming{};
    return out;
}

namespace {

constexpr uint32_t kBlockDim = 16u;

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

using ::nuka::collision::gpu::LbvhNode;
using ::nuka::math::Vec3;

// The device structs (DevPrim/DevBlas/DevInstance/BeautyParams/AovTarget) + the
// nested-traversal fns (IntersectBlasPrimT/BlasLeaf/TlasLeaf/ClosestHit/
// ReconstructHit) are SHARED with the FP32 beauty TU via two_level_render_kernels
// .cuh. The golden kernel below instantiates them Real=double (the default), so
// it stays byte-exact vs the host oracle.

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
    ClosestHit<double>(tlas_nodes, tlas_leaf_count, instances, ray.origin, ray.dir, 0.0f,
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
        ReconstructHit<double>(instances, best_prim, ray.origin, ray.dir, &n, &uv_u, &uv_v);
        normal = n;
        const Material mat = materials[material_id];
        albedo = mat.albedo;

        // Hit point p = origin + t*dir (fp64; mirrors scene_render.cu).
        const Vec3 hit{
            static_cast<float>(static_cast<double>(ray.origin.x) + static_cast<double>(best_t) * static_cast<double>(ray.dir.x)),
            static_cast<float>(static_cast<double>(ray.origin.y) + static_cast<double>(best_t) * static_cast<double>(ray.dir.y)),
            static_cast<float>(static_cast<double>(ray.origin.z) + static_cast<double>(best_t) * static_cast<double>(ray.dir.z))};

        const Vec3 V = RtNormalize<double>(Vec3{-ray.dir.x, -ray.dir.y, -ray.dir.z});
        Vec3 L;
        float light_dist;
        if (light.directional) {
            L = RtNormalize<double>(Vec3{-light.direction.x, -light.direction.y, -light.direction.z});
            light_dist = RtMissDepth();
        } else {
            const Vec3 to{light.position.x - hit.x, light.position.y - hit.y,
                          light.position.z - hit.z};
            L = RtNormalize<double>(to);
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
        ClosestHit<double>(tlas_nodes, tlas_leaf_count, instances, sorigin, L, shadow_eps,
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

// The stochastic FP32 beauty kernel (RenderBeautyKernel + ShadeBeauty + helpers)
// lives in two_level_render_beauty.cu (--fmad=true, Real=float). It shares the
// nested-traversal fns above; this TU keeps only the host build + launch path.

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
    OwnedBuffer d_tri_n0, d_tri_n1, d_tri_n2;  // smooth per-vertex normals (beauty)
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

// InstanceWorldAabb / MakeDevInstance are SHARED (two_level_render_kernels.cuh,
// __host__ __device__) so the single-camera path here and the batched sensor
// scatter build leaf boxes + DevInstances through ONE copy of the arithmetic.

// Full TLAS rebuild cadence (refit handles the frames between). Bounds traversal-
// quality drift only; correctness is identical to a rebuild either way.
constexpr uint32_t kTlasRebuildPeriod = 32u;

// PERSISTENT render device buffers reused across frames (AOV scratch + TLAS
// tables + retained tree + refit scratch); sized lazily, no per-frame malloc/free.
struct RtRenderContext {
    // AOV scratch (host-download path), sized to `aov_pixels`.
    OwnedBuffer d_color, d_depth, d_normal, d_albedo, d_uv, d_prim;
    std::size_t aov_pixels = 0u;

    // TLAS state: persistent device tables + the retained tree (built once, then
    // refit). *_bytes track each table's allocation so EnsureUpload grows on demand.
    OwnedBuffer d_instances, d_mats, d_tlas_boxes, d_refit_visit;
    std::size_t inst_bytes = 0u, mats_bytes = 0u, boxes_bytes = 0u;
    collision::gpu::LbvhBroadphaseResult tlas;
    uint32_t tlas_inst_count = 0u;
    bool topology_built = false;
    uint32_t frames_since_rebuild = 0u;

    // Ensure the AOV scratch is sized for `pixels`; (re)allocate only on growth.
    void EnsureAovs(phi::BufferType* bt, std::size_t pixels) {
        if (pixels == aov_pixels && d_color.Data() != nullptr) {
            return;
        }
        d_color = OwnedBuffer(bt, pixels * 3u * sizeof(float));
        d_depth = OwnedBuffer(bt, pixels * sizeof(float));
        d_normal = OwnedBuffer(bt, pixels * 3u * sizeof(float));
        d_albedo = OwnedBuffer(bt, pixels * 3u * sizeof(float));
        d_uv = OwnedBuffer(bt, pixels * 2u * sizeof(float));
        d_prim = OwnedBuffer(bt, pixels * sizeof(uint32_t));
        aov_pixels = pixels;
    }

    AovTarget AovScratchTarget() {
        AovTarget dst;
        dst.color = static_cast<float*>(d_color.Data());
        dst.depth = static_cast<float*>(d_depth.Data());
        dst.normal = static_cast<float*>(d_normal.Data());
        dst.albedo = static_cast<float*>(d_albedo.Data());
        dst.uv = static_cast<float*>(d_uv.Data());
        dst.prim = static_cast<uint32_t*>(d_prim.Data());
        return dst;
    }
};

// What a launch needs from the (possibly refit) TLAS: device tables + node root.
struct FrameTlasView {
    const DevInstance* instances = nullptr;
    const Material* materials = nullptr;
    const LbvhNode* tlas_nodes = nullptr;
    uint32_t inst_count = 0u;
};

}  // namespace

// Opaque impl: the per-mesh BLAS devices (built once) + the persistent render
// context (AOV scratch + TLAS buffers reused/refit across frames).
struct TwoLevelSceneDevice::Impl {
    std::vector<BlasDevice> meshes;
    RtRenderContext rt;
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

    // Smooth per-vertex normals, 1:1 with the triangles (only when the mesh ships
    // them); empty => the view tri_n* pointers stay null and the beauty path flat-falls.
    std::vector<Vec3> tri_n0, tri_n1, tri_n2;
    const bool has_tri_normals = mesh.tri_normals.size() == mesh.triangles.size() &&
                                 !mesh.tri_normals.empty();
    if (has_tri_normals) {
        tri_n0.reserve(mesh.tri_normals.size());
        tri_n1.reserve(mesh.tri_normals.size());
        tri_n2.reserve(mesh.tri_normals.size());
        for (const auto& tn : mesh.tri_normals) {
            tri_n0.push_back(tn.n0);
            tri_n1.push_back(tn.n1);
            tri_n2.push_back(tn.n2);
        }
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

    // Upload smooth normals + point the view at them ONLY when present; absent =>
    // the tri_n* view pointers stay null (the flat-normal byte-exact fallback).
    if (has_tri_normals) {
        out.d_tri_n0 = UploadOwned(bt, tri_n0);
        out.d_tri_n1 = UploadOwned(bt, tri_n1);
        out.d_tri_n2 = UploadOwned(bt, tri_n2);
        out.view.tri_n0 = static_cast<const Vec3*>(out.d_tri_n0.Data());
        out.view.tri_n1 = static_cast<const Vec3*>(out.d_tri_n1.Data());
        out.view.tri_n2 = static_cast<const Vec3*>(out.d_tri_n2.Data());
    }

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

// Upload a host vector into a persistent OwnedBuffer, reallocating only when its
// byte size changes; otherwise reuse the existing allocation (one device copy).
template <typename T>
void EnsureUpload(OwnedBuffer& dst, std::size_t& dst_bytes, phi::BufferType* bt,
                  const std::vector<T>& values) {
    const std::size_t bytes = values.size() * sizeof(T);
    if (bytes != dst_bytes || dst.Data() == nullptr) {
        dst = OwnedBuffer(bt, bytes);
        dst_bytes = bytes;
    }
    dst.CopyFromHost(values.data(), bytes);
}

// Refresh the instance/material/box tables into the PERSISTENT buffers, then
// REBUILD (first frame / count change / cadence) or REFIT the TLAS (byte-exact).
FrameTlasView EnsureFrameTlas(TwoLevelSceneDevice::Impl* impl, const TwoLevelScene& scene,
                              const RtContext& ctx) {
    RtRenderContext& rt = impl->rt;
    phi::BufferType* bt = ctx.device_bt;
    const uint32_t inst_count = static_cast<uint32_t>(scene.instances.size());

    std::vector<collision::AABB> tlas_aabbs(inst_count);
    std::vector<DevInstance> dev_instances(inst_count);
    for (uint32_t i = 0; i < inst_count; ++i) {
        const Instance& I = scene.instances[i];
        if (I.blas_id >= impl->meshes.size()) {
            throw std::runtime_error("RenderFrame: instance.blas_id out of range");
        }
        const BlasDevice& mesh = impl->meshes[I.blas_id];
        const LbvhNode* nodes = (mesh.leaf_count > 0u) ? mesh.tree.DeviceNodes() : nullptr;
        dev_instances[i] = MakeDevInstance(I.transform, nodes, mesh.leaf_count,
                                           mesh.view, i, I.material_id);
        tlas_aabbs[i] = InstanceWorldAabb(mesh.leaf_count, mesh.local_bound, I.transform);
    }

    EnsureUpload(rt.d_instances, rt.inst_bytes, bt, dev_instances);
    const std::vector<Material> mats =
        scene.materials.empty() ? std::vector<Material>{Material{}} : scene.materials;
    EnsureUpload(rt.d_mats, rt.mats_bytes, bt, mats);
    EnsureUpload(rt.d_tlas_boxes, rt.boxes_bytes, bt, tlas_aabbs);

    const auto* dev_tlas_boxes = static_cast<const collision::AABB*>(rt.d_tlas_boxes.Data());

    // Rebuild the topology on the first frame, when the instance count changes, or
    // periodically; otherwise refit the existing tree's bounds in place.
    const bool need_rebuild = !rt.topology_built || inst_count != rt.tlas_inst_count ||
                              rt.frames_since_rebuild >= kTlasRebuildPeriod;
    if (need_rebuild) {
        rt.tlas = collision::gpu::BuildLbvhForQuery(ctx.stream, ctx.device_id,
                                                    dev_tlas_boxes, inst_count);
        if (!rt.tlas.HasNodes()) {
            throw std::runtime_error("RenderFrame: TLAS LBVH build retained no nodes");
        }
        rt.tlas_inst_count = inst_count;
        rt.topology_built = true;
        rt.frames_since_rebuild = 0u;
        // Refit visit scratch: one uint32 per internal node (leaf_count-1).
        const std::size_t visit_bytes =
            (inst_count > 1u) ? (inst_count - 1u) * sizeof(uint32_t) : sizeof(uint32_t);
        rt.d_refit_visit = OwnedBuffer(bt, visit_bytes);
    } else {
        collision::gpu::RefitLbvh(ctx.stream, ctx.device_id, rt.tlas.DeviceNodesMutable(),
                                  dev_tlas_boxes, inst_count,
                                  static_cast<uint32_t*>(rt.d_refit_visit.Data()));
        ++rt.frames_since_rebuild;
    }

    FrameTlasView out;
    out.inst_count = inst_count;
    out.instances = static_cast<const DevInstance*>(rt.d_instances.Data());
    out.materials = static_cast<const Material*>(rt.d_mats.Data());
    out.tlas_nodes = rt.tlas.DeviceNodes();
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

void CollectSensorBlasRefs(const TwoLevelSceneDevice& device,
                           std::vector<SensorBlasRef>* out_refs) {
    if (out_refs == nullptr) {
        return;
    }
    const auto& meshes = device.GetImpl()->meshes;
    out_refs->clear();
    out_refs->reserve(meshes.size());
    for (const BlasDevice& mesh : meshes) {
        SensorBlasRef ref;
        ref.blas_nodes = (mesh.leaf_count > 0u) ? mesh.tree.DeviceNodes() : nullptr;
        ref.blas_leaf_count = mesh.leaf_count;
        ref.blas = mesh.view;
        ref.local_bound = mesh.local_bound;
        out_refs->push_back(ref);
    }
}

namespace {

// AovTarget (raw per-channel device pointers; null skips a channel) is shared
// with the beauty TU via two_level_render_kernels.cuh.

// Scoped CUDA events for the opt-in timing probe; destroyed even if a launch
// throws between create and the elapsed-time read.
struct ScopedTimingEvents {
    cudaEvent_t e0{}, e1{}, e2{};
    ~ScopedTimingEvents() {
        if (e0) cudaEventDestroy(e0);
        if (e1) cudaEventDestroy(e1);
        if (e2) cudaEventDestroy(e2);
    }
};

// Refresh/refit the persistent TLAS and launch the closest-hit kernel into `dst`.
// Only `dst` varies between the host-download and device-resident entries.
void LaunchRenderFrame(TwoLevelSceneDevice::Impl* impl,
                       const TwoLevelScene& scene,
                       const PinholeCamera& camera,
                       const RtContext& ctx,
                       const AovTarget& dst) {
    // Default path (probe off): no events, byte-identical to the golden tracer.
    const bool probe = g_render_timing_enabled;
    ScopedTimingEvents ev;
    if (probe) {
        cudaEventCreate(&ev.e0); cudaEventCreate(&ev.e1); cudaEventCreate(&ev.e2);
        cudaEventRecord(ev.e0, ctx.stream);
    }

    const FrameTlasView frame = EnsureFrameTlas(impl, scene, ctx);
    if (probe) cudaEventRecord(ev.e1, ctx.stream);

    const dim3 block(kBlockDim, kBlockDim);
    const dim3 grid((camera.width + kBlockDim - 1u) / kBlockDim,
                    (camera.height + kBlockDim - 1u) / kBlockDim);
    RenderFrameKernel<<<grid, block, 0, ctx.stream>>>(
        camera, frame.tlas_nodes, frame.inst_count,
        frame.instances, frame.materials,
        scene.light, scene.ambient,
        dst.color, dst.depth, dst.normal, dst.albedo, dst.uv, dst.prim);
    CheckCuda(cudaGetLastError(), "RenderFrameKernel launch");

    if (probe) {
        cudaEventRecord(ev.e2, ctx.stream);
        cudaEventSynchronize(ev.e2);
        float tlas_ms = 0.0f, kernel_ms = 0.0f;
        cudaEventElapsedTime(&tlas_ms, ev.e0, ev.e1);
        cudaEventElapsedTime(&kernel_ms, ev.e1, ev.e2);
        g_render_timing.tlas_ms += static_cast<double>(tlas_ms);
        g_render_timing.kernel_ms += static_cast<double>(kernel_ms);
        ++g_render_timing.calls;
    }
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

    // Persistent scratch AOVs (reused across frames), downloaded to the host
    // Framebuffer (one D2H). Sized lazily; no per-frame malloc/free.
    RtRenderContext& rt = device.GetImpl()->rt;
    rt.EnsureAovs(ctx.device_bt, pixels);
    const AovTarget dst = rt.AovScratchTarget();

    LaunchRenderFrame(device.GetImpl(), scene, camera, ctx, dst);
    cudaStreamSynchronize(ctx.stream);

    rt.d_color.CopyToHost(fb.color.data(), pixels * 3u * sizeof(float));
    rt.d_depth.CopyToHost(fb.depth.data(), pixels * sizeof(float));
    rt.d_normal.CopyToHost(fb.normal.data(), pixels * 3u * sizeof(float));
    rt.d_albedo.CopyToHost(fb.albedo.data(), pixels * 3u * sizeof(float));
    rt.d_uv.CopyToHost(fb.uv.data(), pixels * 2u * sizeof(float));
    rt.d_prim.CopyToHost(fb.prim.data(), pixels * sizeof(uint32_t));
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
    LaunchRenderFrame(device.GetImpl(), scene, camera, ctx, dst);
    cudaStreamSynchronize(ctx.stream);
}

namespace {

// Refresh/refit the persistent TLAS and launch the stochastic beauty kernel into
// `dst`. Only `dst` varies between the host-download and device-resident entries.
void LaunchRenderBeauty(TwoLevelSceneDevice::Impl* impl,
                        const TwoLevelScene& scene,
                        const PinholeCamera& camera,
                        const BeautyOptions& opt,
                        const RtContext& ctx,
                        const AovTarget& dst) {
    const FrameTlasView frame = EnsureFrameTlas(impl, scene, ctx);

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
    sky.transmit_bounces = opt.transmit_bounces;
    sky.smooth_normals = opt.smooth_normals ? 1u : 0u;
    // Sun disc in the environment: a crisp glint for reflective/transmissive surfaces.
    // Zero radiance keeps the sky byte-identical for every other caller.
    sky.sun_radiance = opt.sun_disc_radiance;
    const float disc_lum = opt.sun_disc_radiance.x + opt.sun_disc_radiance.y + opt.sun_disc_radiance.z;
    if (disc_lum > 0.0f) {
        const float len = std::sqrt(scene.light.direction.x * scene.light.direction.x +
                                    scene.light.direction.y * scene.light.direction.y +
                                    scene.light.direction.z * scene.light.direction.z);
        const float inv = len > 1.0e-6f ? 1.0f / len : 0.0f;
        sky.sun_dir = math::Vec3{-scene.light.direction.x * inv, -scene.light.direction.y * inv,
                                 -scene.light.direction.z * inv};
        sky.sun_cos_radius = std::cos(opt.sun_angular_radius);
    } else {
        sky.sun_dir = math::Vec3{0.0f, 0.0f, 0.0f};
        sky.sun_cos_radius = 0.0f;
    }

    // The FP32 beauty kernel launch lives in two_level_render_beauty.cu.
    LaunchBeautyKernel(camera, frame.tlas_nodes, frame.inst_count, frame.instances,
                       frame.materials, scene.light, sky, opt.samples, opt.seed, dst,
                       ctx.stream);
    CheckCuda(cudaGetLastError(), "RenderBeautyKernel launch");
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

    // Persistent scratch AOVs reused across frames (sized lazily); the kernel
    // writes all 6, downloaded into the host Framebuffer (one D2H).
    RtRenderContext& rt = device.GetImpl()->rt;
    rt.EnsureAovs(ctx.device_bt, pixels);
    const AovTarget dst = rt.AovScratchTarget();

    LaunchRenderBeauty(device.GetImpl(), scene, camera, opt, ctx, dst);
    cudaStreamSynchronize(ctx.stream);

    // Copy back only the requested channels; the rest keep their assign() default
    // (background-safe). The kernel wrote all channels, so this is byte-unchanged
    // for any channel left enabled.
    const AovDownloadMask& m = opt.download;
    if (m.color) rt.d_color.CopyToHost(fb.color.data(), pixels * 3u * sizeof(float));
    if (m.depth) rt.d_depth.CopyToHost(fb.depth.data(), pixels * sizeof(float));
    if (m.normal) rt.d_normal.CopyToHost(fb.normal.data(), pixels * 3u * sizeof(float));
    if (m.albedo) rt.d_albedo.CopyToHost(fb.albedo.data(), pixels * 3u * sizeof(float));
    if (m.uv) rt.d_uv.CopyToHost(fb.uv.data(), pixels * 2u * sizeof(float));
    if (m.prim) rt.d_prim.CopyToHost(fb.prim.data(), pixels * sizeof(uint32_t));
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
    LaunchRenderBeauty(device.GetImpl(), scene, camera, opt, ctx, dst);
    cudaStreamSynchronize(ctx.stream);
}

}  // namespace nuka::rt
