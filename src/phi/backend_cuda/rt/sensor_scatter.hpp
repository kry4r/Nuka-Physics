#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- the device instance-transform + world-AABB scatter for the BATCHED
// sensor path. One launch over (env x instance) composes each instance's world
// transform = fk(pose) * cached_visual_local (the SAME fk*cvl as the host
// RenderWorldToTwoLevelScene + the interop scatter) straight from the World's
// device LinkPose/BodyPose/BasePose buffers -- no host scene.instances, no D2H.
//
// It writes BOTH a DevInstance[E*M] (the batched tracer's instance table) AND the
// instance world-AABB[E*M] env-major (the batched LBVH build's leaf bounds). The
// fk*cvl compose + the world-AABB arithmetic are the SAME shared helpers the
// single-camera path uses, so a single env tile is bit-identical to that path.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "phi/backend_cuda/rt/two_level_render_kernels.cuh"  // rt::DevInstance / DevBlas
#include "phi/interop_scatter.hpp"  // ScatterFkSource / InstanceScatterRow
#include "rt/camera.hpp"  // rt::PinholeCamera
#include "scene/scene_ir.hpp"  // scene::SensorDesc

#include <cstdint>
#include <vector>

#include <cmath>

#include <cuda_runtime.h>

#if defined(__CUDACC__)
#define NUKA_RT_HD __host__ __device__
#else
#define NUKA_RT_HD
#endif

namespace nuka::rt {

// Unit direction in the SENSOR-LOCAL frame for fan sample (az_index, el_index):
// az = az_min + (az_index+0.5)*step swept about local +Z, el likewise toward +Z
// (the +0.5 sample-CENTER discipline of the pinhole). az=el=0 => local +X. A 1-ray
// fan (count 1, min==max) returns the min endpoint, a pure (origin,dir) like a
// pinhole pixel. Pure fn of (index, pattern): same inputs => same bytes (D1).
NUKA_RT_HD inline math::Vec3 LidarRayDirLocal(uint32_t az_index, uint32_t el_index,
                                              uint32_t az_count, uint32_t el_count,
                                              float az_min, float az_max,
                                              float el_min, float el_max) {
    const float az_span = az_max - az_min;
    const float el_span = el_max - el_min;
    const float az = (az_count > 0u)
                         ? az_min + (static_cast<float>(az_index) + 0.5f) * az_span /
                                        static_cast<float>(az_count)
                         : az_min;
    const float el = (el_count > 0u)
                         ? el_min + (static_cast<float>(el_index) + 0.5f) * el_span /
                                        static_cast<float>(el_count)
                         : el_min;
    const float ce = cosf(el);
    return math::Vec3{ce * cosf(az), ce * sinf(az), sinf(el)};
}

// One unique mesh's shared-BLAS view (built once, env-invariant), indexed by an
// instance's blas_id. Carries the pointers MakeDevInstance copies per DevInstance.
struct SensorBlasRef {
    const collision::gpu::LbvhNode* blas_nodes = nullptr;
    uint32_t blas_leaf_count = 0u;
    DevBlas blas;
    collision::AABB local_bound;  // mesh local-space bound (for the world AABB)
};

// Compose world = fk*cvl per (env x instance) -> DevInstance table + env-major
// world-AABBs [E*M]. Per-instance arrays env-invariant; env from the global index.
void ScatterEnvInstances(cudaStream_t stream,
                         const phi::ScatterFkSource& fk,
                         const phi::InstanceScatterRow* device_rows,
                         const uint32_t* device_blas_id,
                         const uint32_t* device_material_id,
                         const SensorBlasRef* device_blas_refs,
                         uint32_t env_count,
                         uint32_t instances_per_env,
                         DevInstance* out_instances,
                         collision::AABB* out_world_aabbs);

// One sensor's mount binding + pinhole intrinsics + lidar pattern, the scatter
// kernels' per-row input (env-invariant). kind: 1=Link, 2=Body, 3=Base, 0=Static
// (no fk). row = index into the env's pose field; local_offset = pos3+quat(wxyz)
// mount->sensor. The camera scatter reads only the fov/intrinsics fields; the
// lidar scatter reads only the az/el pattern fields, so a camera row's zero lidar
// fields (and vice versa) never move the other kernel's output bytes.
// fx<=0 => default intrinsics (focal from fov_y+aspect, centered, distortion off,
// clip wide-open) => the produced camera is byte-identical to the plain pinhole.
struct SensorMountRow {
    uint32_t kind = 0u;
    uint32_t row = 0u;
    float local_offset[7] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    float fov_y = 0.0f;  // full vertical FOV, radians
    uint32_t width = 0u;
    uint32_t height = 0u;
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    float k1 = 0.0f;
    float k2 = 0.0f;
    uint32_t distortion = 0u;  // 4-byte so the row is padding-free (byte-exact memcmp)
    float near_clip = 0.01f;
    float far_clip = 1000.0f;
    // Lidar (az,el) fan: az_count*el_count rays swept az_min->az_max (about local
    // +Z) and el_min->el_max (toward local +Z); range clamped to [min,max]_range.
    uint32_t az_count = 0u;  // 0 => not a lidar row (the camera scatter ignores these)
    uint32_t el_count = 0u;
    float az_min = 0.0f;
    float az_max = 0.0f;
    float el_min = 0.0f;
    float el_max = 0.0f;
    float min_range = 0.0f;
    float max_range = 0.0f;
};

// One sensor's resolved per-env lidar pose + fan, the lidar kernel's per-ray input
// (origin + world rotation that maps the local fan into world + the pattern). The
// scatter resolves fk*local_offset once per (env,sensor) so the trace need not.
struct LidarSensor {
    float origin[3] = {0.0f, 0.0f, 0.0f};
    float rotation[4] = {1.0f, 0.0f, 0.0f, 0.0f};  // world rotation (w,x,y,z)
    uint32_t az_count = 0u;
    uint32_t el_count = 0u;
    float az_min = 0.0f;
    float az_max = 0.0f;
    float el_min = 0.0f;
    float el_max = 0.0f;
    float min_range = 0.0f;
    float max_range = 0.0f;
};

// Build a per-env PinholeCamera [E*sensors_per_env] from the mount poses: per
// (env x sensor) cam_world = fk(pose) * local_offset, then a USD/Isaac camera
// (look down -Z, +Y up). Mount table env-invariant; env from the global index.
void ScatterEnvCameras(cudaStream_t stream,
                       const phi::ScatterFkSource& fk,
                       const SensorMountRow* device_mounts,
                       uint32_t env_count,
                       uint32_t sensors_per_env,
                       PinholeCamera* out_cameras);

// Build a per-env LidarSensor [E*sensors_per_env] from the mount poses: per
// (env x sensor) origin/rotation = fk(pose) * local_offset, carrying the row's
// az/el pattern. The trace generates each ray from this (no per-ray mount compose).
// Mount table env-invariant; env from the global index. ResolveFkPose/ComposeWorld
// are the SAME helpers the camera scatter uses.
void ScatterEnvLidars(cudaStream_t stream,
                      const phi::ScatterFkSource& fk,
                      const SensorMountRow* device_mounts,
                      uint32_t env_count,
                      uint32_t sensors_per_env,
                      LidarSensor* out_lidars);

// Lower scene::SensorDesc list to the SensorMountRow table (host-side, once). The
// vfov is converted degrees->radians (the PinholeCamera FULL vertical FOV); a
// Lidar/RangeScan desc fills the az/el pattern fields instead of the intrinsics.
std::vector<SensorMountRow> BuildSensorMountRows(const std::vector<scene::SensorDesc>& sensors);

}  // namespace nuka::rt

#undef NUKA_RT_HD
