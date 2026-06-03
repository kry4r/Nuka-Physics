// ---------------------------------------------------------------------------
// nuka::sensor::RenderCameraSensor -- RGB + depth + normal + uv CAMERA SENSOR
// over the LIVE batched sim state (v0.7 p14b). See camera_render.hpp for the
// fidelity decision (SDF-instanced), the sensor conventions (camera-space normal,
// ray-distance depth, per-body-id palette albedo), and the D1 contract.
//
// REUSES p13's __host__ __device__ RaySdfSphereTrace (rt::intersect_primitives)
// and the p12 stackless traversal (rt::TraverseRay) UNCHANGED. The only new code
// is the per-instance ray transform (the body pose's inverse applied to the ray,
// quaternion rotation INLINED because math::Quat::Rotate is host-only) and the
// per-body palette. Built --fmad=false (CMake) to pair with the reused fp64 leaf.
//
// ZERO atomics: one thread per pixel, sequential best_t/best_prim, one writer per
// output element -> two runs byte-identical across all channels.
// ---------------------------------------------------------------------------

#include "sensor/camera_render.hpp"

#include "collision/aabb.hpp"
#include "collision/broadphase_lbvh.hpp"
#include "collision/lbvh_node.cuh"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer.hpp"
#include "phi/device_context.hpp"
#include "rt/bvh_traverse_impl.cuh"
#include "rt/camera.hpp"
#include "rt/instance_transform.cuh"  // QuatRotate / QuatInvRotate (HOISTED here)
#include "rt/intersect_primitives.cuh"
#include "rt/ray_box.cuh"
#include "runtime/sdf/sparse_sdf_query.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::sensor {

namespace {

constexpr uint32_t kBlockDim = 16u;

using ::nuka::collision::gpu::LbvhNode;
using ::nuka::math::Transform;
using ::nuka::math::Vec3;
using ::nuka::rt::kNoPrim;
using ::nuka::rt::QuatInvRotate;  // HOISTED to rt/instance_transform.cuh
using ::nuka::rt::QuatRotate;     // HOISTED to rt/instance_transform.cuh
using ::nuka::rt::RtMissDepth;
using ::nuka::rt::RtNormalize;

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

// Flat 8-color palette (deterministic; keyed on body palette id % size). Distinct,
// mid-bright colors so adjacent body ids are visually separable in the RGB AOV.
// Stored in __constant__-free static device array (passed by value-ish via a small
// device function to keep the kernel signature lean). Identical host/device since
// it is a compile-time table.
__device__ __host__ __forceinline__ Vec3 PaletteColor(uint32_t id) {
    // 8 fixed RGB triples (sRGB-ish flat albedos).
    switch (id % kCameraPaletteSize) {
        case 0u: return Vec3{0.85f, 0.20f, 0.20f};  // red
        case 1u: return Vec3{0.20f, 0.75f, 0.30f};  // green
        case 2u: return Vec3{0.25f, 0.45f, 0.90f};  // blue
        case 3u: return Vec3{0.90f, 0.80f, 0.20f};  // yellow
        case 4u: return Vec3{0.80f, 0.40f, 0.85f};  // magenta
        case 5u: return Vec3{0.25f, 0.80f, 0.85f};  // cyan
        case 6u: return Vec3{0.90f, 0.55f, 0.20f};  // orange
        default: return Vec3{0.70f, 0.70f, 0.70f};  // grey
    }
}

// Per-env device scene view: the LIVE poses for THIS env's bodies + the shared
// SDF/material tables. The LBVH leaf's original index is a LOCAL body index in
// [0, body_count); body_to_sdf maps it to an sdf view (kCameraNoSdf => skip).
struct DevEnvScene {
    const Transform* poses = nullptr;   // env's body poses (already offset by env)
    const runtime::sdf::SparseSdfDevice* sdf_views = nullptr;
    const uint32_t* body_to_sdf = nullptr;
    const uint32_t* body_material = nullptr;
    uint32_t body_count = 0u;
};

// QuatRotate / QuatInvRotate are HOISTED to rt/instance_transform.cuh (shared by
// the p14b camera sensor, the G1-A two-level tracer, and its host oracle). They
// are imported via the `using` declarations above; behavior-identical (same fp32
// t=2*cross(qv,v) form) -> nuka_camera_render_test is the no-regression guard.

// Local SDF bound of a view: [origin, origin + dims*voxel_size]. Used both to clip
// the local-frame march and (host-side) to build the world LBVH AABB.
__host__ __device__ __forceinline__ collision::AABB SdfLocalBound(
    const runtime::sdf::SparseSdfDevice& s) {
    collision::AABB b;
    b.min = s.origin;
    b.max = Vec3{s.origin.x + static_cast<float>(s.dims[0]) * s.voxel_size,
                 s.origin.y + static_cast<float>(s.dims[1]) * s.voxel_size,
                 s.origin.z + static_cast<float>(s.dims[2]) * s.voxel_size};
    return b;
}

// Intersect one body (LOCAL body index) for the closest-hit pass: returns the
// WORLD-space hit distance t only (rigid transform keeps dir unit, so the local t
// equals the world t). Transforms the ray into the body's local frame, then reuses
// p13's RaySdfSphereTrace against the body's cooked SDF in local space.
__device__ __forceinline__ bool IntersectBodyT(const DevEnvScene& s,
                                               uint32_t body,
                                               const Vec3& origin,
                                               const Vec3& dir,
                                               float t_min,
                                               float* out_t) {
    const uint32_t sdf_idx = s.body_to_sdf[body];
    if (sdf_idx == kCameraNoSdf) {
        return false;
    }
    const Transform& T = s.poses[body];
    // Ray into the body's local frame: o_l = R^-1*(o - p), d_l = R^-1*d. Rigid ->
    // |d_l| == |d| == 1, so the marched t is the world distance unchanged.
    const Vec3 o_rel{origin.x - T.position.x, origin.y - T.position.y,
                     origin.z - T.position.z};
    const Vec3 o_l = QuatInvRotate(T.rotation, o_rel);
    const Vec3 d_l = QuatInvRotate(T.rotation, dir);

    const runtime::sdf::SparseSdfDevice& sdf = s.sdf_views[sdf_idx];
    const collision::AABB local_bound = SdfLocalBound(sdf);
    Vec3 n_local;
    const float surface_eps = (sdf.voxel_size > 0.0f) ? (0.2f * sdf.voxel_size) : 1.0e-3f;
    return rt::RaySdfSphereTrace(o_l, d_l, sdf, local_bound, t_min, surface_eps,
                                 512, out_t, &n_local);
}

// Closest-hit leaf functor: reads the leaf's original LOCAL body index
// (nodes[leaf].left), intersects, updates closest (lowest-index tie-break).
struct EnvLeaf {
    const LbvhNode* __restrict__ nodes;
    DevEnvScene scene;
    float t_min;
    __device__ void operator()(int32_t leaf_node, const Vec3& origin,
                               const Vec3& dir, float* best_t,
                               uint32_t* best_prim) const {
        const uint32_t body = static_cast<uint32_t>(nodes[leaf_node].left);
        float t;
        if (IntersectBodyT(scene, body, origin, dir, t_min, &t)) {
            if (t >= t_min) {
                rt::RtClosestHitUpdate(t, body, best_t, best_prim);
            }
        }
    }
};

__device__ __forceinline__ void ClosestHit(const LbvhNode* __restrict__ nodes,
                                           uint32_t leaf_count,
                                           const DevEnvScene& scene,
                                           const Vec3& origin, const Vec3& dir,
                                           float t_min, float* best_t,
                                           uint32_t* best_prim) {
    *best_t = RtMissDepth();
    *best_prim = kNoPrim;
    EnvLeaf leaf{nodes, scene, t_min};
    if (leaf_count == 1u) {
        leaf(0, origin, dir, best_t, best_prim);
    } else if (leaf_count >= 2u) {
        rt::TraverseRay(nodes, leaf_count - 1u, origin, dir, best_t, best_prim, leaf);
    }
}

// Reconstruct the WORLD-space normal of the winning body (re-march in local space
// for the local gradient normal, then rotate to world). Returns the world normal.
__device__ __forceinline__ Vec3 ReconstructWorldNormal(const DevEnvScene& s,
                                                      uint32_t body,
                                                      const Vec3& origin,
                                                      const Vec3& dir) {
    const uint32_t sdf_idx = s.body_to_sdf[body];
    const Transform& T = s.poses[body];
    const Vec3 o_rel{origin.x - T.position.x, origin.y - T.position.y,
                     origin.z - T.position.z};
    const Vec3 o_l = QuatInvRotate(T.rotation, o_rel);
    const Vec3 d_l = QuatInvRotate(T.rotation, dir);
    const runtime::sdf::SparseSdfDevice& sdf = s.sdf_views[sdf_idx];
    const collision::AABB local_bound = SdfLocalBound(sdf);
    float t;
    Vec3 n_local{0.0f, 0.0f, 0.0f};
    const float surface_eps = (sdf.voxel_size > 0.0f) ? (0.2f * sdf.voxel_size) : 1.0e-3f;
    rt::RaySdfSphereTrace(o_l, d_l, sdf, local_bound, 0.0f, surface_eps, 512, &t,
                          &n_local);
    // Local gradient normal -> world (rotation only).
    return RtNormalize(QuatRotate(T.rotation, n_local));
}

// One thread per pixel. Primary closest hit -> reconstruct world normal -> emit
// CAMERA-space viewer-faced normal + ray-distance depth + palette RGB + uv(0,0).
// Writes into THIS env's slice (out_* already offset by env on the host).
__global__ void RenderEnvKernel(rt::PinholeCamera camera,
                               const LbvhNode* __restrict__ nodes,
                               uint32_t leaf_count, DevEnvScene scene,
                               float* __restrict__ out_rgb,
                               float* __restrict__ out_depth,
                               float* __restrict__ out_normal,
                               float* __restrict__ out_uv) {
    const uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= camera.width || py >= camera.height) {
        return;
    }
    const uint32_t pixel = py * camera.width + px;
    const rt::Ray ray = camera.GenerateRay(px, py);

    float best_t;
    uint32_t best_body;
    ClosestHit(nodes, leaf_count, scene, ray.origin, ray.dir, 0.0f, &best_t,
               &best_body);

    // Defaults (miss): black RGB, sentinel depth, zero normal/uv.
    Vec3 rgb{0.0f, 0.0f, 0.0f};
    float depth = RtMissDepth();
    Vec3 normal_cam{0.0f, 0.0f, 0.0f};
    float uv_u = 0.0f, uv_v = 0.0f;

    if (best_body != kNoPrim) {
        depth = best_t;
        rgb = PaletteColor(scene.body_material[best_body]);

        const Vec3 n_world = ReconstructWorldNormal(scene, best_body, ray.origin,
                                                    ray.dir);
        // World -> camera space: project onto the camera basis (right, up, forward).
        // fp64 dots, fp32 store (consistent with the leaf arithmetic).
        const double nx = static_cast<double>(n_world.x);
        const double ny = static_cast<double>(n_world.y);
        const double nz = static_cast<double>(n_world.z);
        double cx = nx * camera.right.x + ny * camera.right.y + nz * camera.right.z;
        double cy = nx * camera.up.x + ny * camera.up.y + nz * camera.up.z;
        double cz = nx * camera.forward.x + ny * camera.forward.y + nz * camera.forward.z;
        // Viewer-faced: the camera looks down +forward, so a visible surface should
        // have its normal pointing back toward the camera, i.e. camera-z component
        // negative. Flip if it faces away (cz > 0).
        if (cz > 0.0) {
            cx = -cx;
            cy = -cy;
            cz = -cz;
        }
        normal_cam = Vec3{static_cast<float>(cx), static_cast<float>(cy),
                          static_cast<float>(cz)};
    }

    out_rgb[pixel * 3u + 0u] = rgb.x;
    out_rgb[pixel * 3u + 1u] = rgb.y;
    out_rgb[pixel * 3u + 2u] = rgb.z;
    out_depth[pixel] = depth;
    out_normal[pixel * 3u + 0u] = normal_cam.x;
    out_normal[pixel * 3u + 1u] = normal_cam.y;
    out_normal[pixel * 3u + 2u] = normal_cam.z;
    out_uv[pixel * 2u + 0u] = uv_u;
    out_uv[pixel * 2u + 1u] = uv_v;
}

}  // namespace

void RenderCameraSensor(const phi::DeviceContext& context,
                        const Transform* poses,
                        const runtime::sdf::SparseSdfDevice* sdf_views,
                        const uint32_t* body_to_sdf,
                        const uint32_t* body_material,
                        const rt::PinholeCamera* cameras,
                        uint32_t env_count,
                        uint32_t body_count_per_env,
                        uint32_t width,
                        uint32_t height,
                        float* out_rgb,
                        float* out_depth,
                        float* out_normal,
                        float* out_uv) {
    if (env_count == 0u || body_count_per_env == 0u || width == 0u || height == 0u) {
        return;
    }
    if (poses == nullptr || sdf_views == nullptr || body_to_sdf == nullptr ||
        body_material == nullptr || cameras == nullptr || out_rgb == nullptr ||
        out_depth == nullptr || out_normal == nullptr || out_uv == nullptr) {
        return;
    }

    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();

    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);

    // The world LBVH AABBs are derived from the LIVE poses, so we read poses back
    // to the host to build per-env per-body world boxes. This is the build-time
    // host round-trip only (the render itself is fully on device); deterministic
    // (a pure copy + transform). One readback of env_count*body_count Transforms.
    const size_t total_bodies =
        static_cast<size_t>(env_count) * static_cast<size_t>(body_count_per_env);
    std::vector<Transform> host_poses(total_bodies);
    CheckCuda(cudaMemcpyAsync(host_poses.data(), poses,
                              total_bodies * sizeof(Transform),
                              cudaMemcpyDeviceToHost, stream),
              "RenderCameraSensor poses D2H");
    // Read the body_to_sdf / SDF view headers (origin/dims/voxel) to compute the
    // local bounds host-side. SDF views are small (one struct per sdf).
    std::vector<uint32_t> host_body_to_sdf(body_count_per_env);
    CheckCuda(cudaMemcpyAsync(host_body_to_sdf.data(), body_to_sdf,
                              body_count_per_env * sizeof(uint32_t),
                              cudaMemcpyDeviceToHost, stream),
              "RenderCameraSensor body_to_sdf D2H");
    context.stream.Synchronize();

    // Max sdf index referenced -> how many view headers to read.
    uint32_t max_sdf = 0u;
    bool any_sdf = false;
    for (uint32_t b = 0; b < body_count_per_env; ++b) {
        if (host_body_to_sdf[b] != kCameraNoSdf) {
            any_sdf = true;
            if (host_body_to_sdf[b] > max_sdf) max_sdf = host_body_to_sdf[b];
        }
    }
    std::vector<runtime::sdf::SparseSdfDevice> host_views;
    if (any_sdf) {
        host_views.resize(max_sdf + 1u);
        CheckCuda(cudaMemcpyAsync(host_views.data(), sdf_views,
                                  (max_sdf + 1u) * sizeof(runtime::sdf::SparseSdfDevice),
                                  cudaMemcpyDeviceToHost, stream),
                  "RenderCameraSensor sdf_views D2H");
        context.stream.Synchronize();
    }

    const dim3 block(kBlockDim, kBlockDim);
    const dim3 grid((width + kBlockDim - 1u) / kBlockDim,
                    (height + kBlockDim - 1u) / kBlockDim);

    // Render each env in FIXED order (D1). Build that env's LBVH over its body
    // world-AABBs (only bodies with a cooked SDF contribute a leaf), then launch.
    for (uint32_t env = 0; env < env_count; ++env) {
        // Build the leaf AABBs + a leaf-local-body map for this env. The leaf's
        // original index (nodes[leaf].left) is the LOCAL body index, so we must give
        // the LBVH a contiguous box list whose i-th entry IS local body i. We give
        // EVERY body a box (degenerate for no-SDF bodies) so leaf index == body
        // index; no-SDF bodies are skipped at the leaf (body_to_sdf == kCameraNoSdf).
        std::vector<collision::AABB> aabbs(body_count_per_env);
        for (uint32_t b = 0; b < body_count_per_env; ++b) {
            const uint32_t sdf_idx = host_body_to_sdf[b];
            const Transform& T = host_poses[static_cast<size_t>(env) * body_count_per_env + b];
            if (sdf_idx == kCameraNoSdf || sdf_idx >= host_views.size()) {
                // Degenerate point box at the body origin (never hit; skipped at leaf).
                aabbs[b].min = T.position;
                aabbs[b].max = T.position;
                continue;
            }
            const collision::AABB local = SdfLocalBound(host_views[sdf_idx]);
            const Vec3 half{0.5f * (local.max.x - local.min.x),
                            0.5f * (local.max.y - local.min.y),
                            0.5f * (local.max.z - local.min.z)};
            const Vec3 center{0.5f * (local.min.x + local.max.x),
                              0.5f * (local.min.y + local.max.y),
                              0.5f * (local.min.z + local.max.z)};
            // World AABB enclosing the oriented local box: transform the box CENTER
            // by the pose, then enclose the oriented half-extent box about it.
            Transform centered = T;
            centered.position = T.TransformPoint(center);
            aabbs[b] = collision::AABB::FromBox(centered, half);
        }

        phi::Buffer d_boxes = phi::UploadVector(aabbs);
        const auto* dev_boxes = static_cast<const collision::AABB*>(d_boxes.Data());
        auto tree = collision::gpu::BuildLbvhForQuery(dev_boxes, body_count_per_env);
        if (!tree.HasNodes()) {
            throw std::runtime_error("RenderCameraSensor: LBVH build retained no nodes");
        }
        const LbvhNode* env_nodes = tree.DeviceNodes();

        DevEnvScene scene;
        scene.poses = poses + static_cast<size_t>(env) * body_count_per_env;
        scene.sdf_views = sdf_views;
        scene.body_to_sdf = body_to_sdf;
        scene.body_material = body_material;
        scene.body_count = body_count_per_env;

        const size_t env_off = static_cast<size_t>(env) * pixels;
        RenderEnvKernel<<<grid, block, 0, stream>>>(
            cameras[env], env_nodes, body_count_per_env, scene,
            out_rgb + env_off * 3u, out_depth + env_off,
            out_normal + env_off * 3u, out_uv + env_off * 2u);
        CheckCuda(cudaGetLastError(), "RenderEnvKernel launch");
        // Sync before the per-env LBVH buffer (d_boxes / tree) is destroyed at the
        // end of this iteration -- the kernel reads tree's device nodes.
        context.stream.Synchronize();
    }
}

}  // namespace nuka::sensor
