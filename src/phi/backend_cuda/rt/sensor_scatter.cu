// ---------------------------------------------------------------------------
// nuka::rt -- the BATCHED-sensor device instance-transform + world-AABB scatter.
//
// One thread per (env x instance): resolve the fk pose from the env-major device
// LinkPose/BodyPose/BasePose buffers, compose world = fk * cached_visual_local,
// and write the DevInstance table + env-major world-AABBs. The compose is the
// SAME fp32 add/mul/cross sequence as math::Transform::operator* (the host
// RenderWorldToTwoLevelScene + the interop scatter); the world-AABB +
// DevInstance build go through the shared InstanceWorldAabb / MakeDevInstance, so
// a single env tile is bit-identical to the single-camera path. Built --fmad=false
// (CMake) so the device add/mul matches the host's -ffp-contract=off exactly.
//
// math::Transform::operator* / math::Quat are host-only inline (their ctors are
// constexpr __host__, NOT __host__ __device__), so device code cannot construct
// them; per the repo's instance_transform.cuh idiom we operate on plain floats +
// the HD math::Vec3, reusing rt::QuatRotate (the position rotate) and matching
// Quat::operator* / Normalized() for the quaternion compose.
// ---------------------------------------------------------------------------

#include "phi/backend_cuda/rt/sensor_scatter.hpp"

#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "phi/backend_cuda/launch.cuh"  // LaunchCuda (the sole <<<>>> wrapper)
#include "phi/backend_cuda/rt/instance_transform.cuh"  // rt::QuatRotate (HD)

#include <cstdint>

namespace nuka::rt {

namespace {

using ::nuka::collision::AABB;
using ::nuka::math::Quat;
using ::nuka::math::Transform;
using ::nuka::math::Vec3;
using ::nuka::phi::InstanceScatterRow;
using ::nuka::phi::ScatterFkSource;

// Hamilton product a*b on plain floats. EXACTLY math::Quat::operator* (quat.hpp:36).
__device__ inline void DevQuatMul(float aw, float ax, float ay, float az,
                                  float bw, float bx, float by, float bz,
                                  float* ow, float* ox, float* oy, float* oz) {
    *ow = aw * bw - ax * bx - ay * by - az * bz;
    *ox = aw * bx + ax * bw + ay * bz - az * by;
    *oy = aw * by - ax * bz + ay * bw + az * bx;
    *oz = aw * bz + ax * by - ay * bx + az * bw;
}

// Unit-quaternion normalize. EXACTLY math::Quat::Normalized: n<1e-12 -> Identity
// else divide; sqrtf to match the host std::sqrt<float>.
__device__ inline void DevQuatNormalize(float* w, float* x, float* y, float* z) {
    const float n = sqrtf((*w) * (*w) + (*x) * (*x) + (*y) * (*y) + (*z) * (*z));
    if (n < 1e-12f) { *w = 1.0f; *x = 0.0f; *y = 0.0f; *z = 0.0f; return; }
    *w /= n; *x /= n; *y /= n; *z /= n;
}

// Resolve the fk pose for mount `kind`/`row` in env `env` (env-major offsets). A
// null field ptr / out-of-range row -> no pose (the caller keeps fk = identity).
__device__ inline const Transform* ResolveFkPose(const ScatterFkSource& fk,
                                                 uint32_t kind, uint32_t row,
                                                 uint32_t env) {
    switch (kind) {
        case 1u:  // Link
            if (fk.link_pose != nullptr && row < fk.links_per_env) {
                return static_cast<const Transform*>(fk.link_pose) +
                       (env * fk.links_per_env + row);
            }
            return nullptr;
        case 2u:  // Body
            if (fk.body_pose != nullptr && row < fk.bodies_per_env) {
                return static_cast<const Transform*>(fk.body_pose) +
                       (env * fk.bodies_per_env + row);
            }
            return nullptr;
        case 3u:  // Base
            if (fk.base_pose != nullptr) {
                return static_cast<const Transform*>(fk.base_pose) + env;
            }
            return nullptr;
        default:  // 0 = Static / unknown.
            return nullptr;
    }
}

// Compose world = fk * cvl on plain floats: pos = fk.R.Rotate(cvl.pos)+fk.pos,
// rot = (fk.R * cvl.R).Normalized() (math::Transform::operator*). fk_pose null ->
// fk = identity (world == cvl). The SAME compose the instance + camera kernels run.
__device__ inline Transform ComposeWorld(const Transform* fk_pose,
                                         const Vec3& cvl_pos, float cvl_qw,
                                         float cvl_qx, float cvl_qy, float cvl_qz) {
    Transform world;
    if (fk_pose == nullptr) {
        world.position = cvl_pos;
        world.rotation.w = cvl_qw;
        world.rotation.x = cvl_qx;
        world.rotation.y = cvl_qy;
        world.rotation.z = cvl_qz;
        return world;
    }
    const Quat fk_rot = fk_pose->rotation;
    world.position = QuatRotate(fk_rot, cvl_pos) + fk_pose->position;
    float wqw, wqx, wqy, wqz;
    DevQuatMul(fk_rot.w, fk_rot.x, fk_rot.y, fk_rot.z, cvl_qw, cvl_qx, cvl_qy,
               cvl_qz, &wqw, &wqx, &wqy, &wqz);
    DevQuatNormalize(&wqw, &wqx, &wqy, &wqz);
    world.rotation.w = wqw;
    world.rotation.x = wqx;
    world.rotation.y = wqy;
    world.rotation.z = wqz;
    return world;
}

// One thread per (env x instance). Composes world = fk * cvl, then writes the
// DevInstance + env-major world-AABB via the shared helpers.
__global__ void ScatterEnvInstancesKernel(ScatterFkSource fk,
                                          const InstanceScatterRow* __restrict__ rows,
                                          const uint32_t* __restrict__ blas_id,
                                          const uint32_t* __restrict__ material_id,
                                          const SensorBlasRef* __restrict__ blas_refs,
                                          uint32_t env_count,
                                          uint32_t instances_per_env,
                                          DevInstance* __restrict__ out_instances,
                                          AABB* __restrict__ out_world_aabbs) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = env_count * instances_per_env;
    if (i >= total) return;
    const uint32_t env = i / instances_per_env;
    const uint32_t local = i % instances_per_env;

    const InstanceScatterRow row = rows[local];
    const SensorBlasRef ref = blas_refs[blas_id[local]];

    // cached_visual_local as plain floats (pos3 + quat w,x,y,z) -- the SAME bytes
    // math::Transform carries (pos.xyz, rot.wxyz).
    const Vec3 cvl_pos{row.cached_visual_local[0], row.cached_visual_local[1],
                       row.cached_visual_local[2]};
    const float cvl_qw = row.cached_visual_local[3];
    const float cvl_qx = row.cached_visual_local[4];
    const float cvl_qy = row.cached_visual_local[5];
    const float cvl_qz = row.cached_visual_local[6];

    // Resolve fk; a missing pose => fk = identity (world == cvl).
    const Transform* fk_pose = ResolveFkPose(fk, row.kind, row.row, env);
    const Transform world =
        ComposeWorld(fk_pose, cvl_pos, cvl_qw, cvl_qx, cvl_qy, cvl_qz);

    out_instances[i] = MakeDevInstance(world, ref.blas_nodes, ref.blas_leaf_count,
                                       ref.blas, i, material_id[local]);
    out_world_aabbs[i] = InstanceWorldAabb(ref.blas_leaf_count, ref.local_bound, world);
}

// One thread per (env x sensor). Composes cam_world = fk(pose) * local_offset and
// builds a PinholeCamera. Camera-local axes are USD/Isaac: look down -Z, +Y up;
// the mount rotation maps them to world for the shared BuildPinholeBasis.
__global__ void ScatterEnvCamerasKernel(ScatterFkSource fk,
                                        const SensorMountRow* __restrict__ mounts,
                                        uint32_t env_count,
                                        uint32_t sensors_per_env,
                                        PinholeCamera* __restrict__ out_cameras) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = env_count * sensors_per_env;
    if (i >= total) return;
    const uint32_t env = i / sensors_per_env;
    const uint32_t local = i % sensors_per_env;

    const SensorMountRow row = mounts[local];
    const Vec3 off_pos{row.local_offset[0], row.local_offset[1], row.local_offset[2]};

    const Transform* fk_pose = ResolveFkPose(fk, row.kind, row.row, env);
    const Transform cam_world = ComposeWorld(fk_pose, off_pos, row.local_offset[3],
                                             row.local_offset[4], row.local_offset[5],
                                             row.local_offset[6]);

    const Quat r = cam_world.rotation;
    const Vec3 forward = QuatRotate(r, Vec3{0.0f, 0.0f, -1.0f});
    const Vec3 up = QuatRotate(r, Vec3{0.0f, 1.0f, 0.0f});
    PinholeCamera cam = BuildPinholeBasis(cam_world.position, forward, up, row.fov_y,
                                          row.width, row.height);
    // Carry the sensor's intrinsics; default rows (fx<=0, distortion off, clip
    // wide-open) leave the camera byte-identical to the plain pinhole.
    cam.fx = row.fx;
    cam.fy = row.fy;
    cam.cx = row.cx;
    cam.cy = row.cy;
    cam.k1 = row.k1;
    cam.k2 = row.k2;
    cam.distortion = row.distortion;
    cam.near_clip = row.near_clip;
    cam.far_clip = row.far_clip;
    out_cameras[i] = cam;
}

// One thread per (env x sensor). Composes sensor_world = fk(pose) * local_offset
// and stores its origin + world rotation + the row's az/el pattern. The lidar
// trace rotates each fan ray by this rotation -- the SAME mount compose the camera
// scatter runs, so a lidar shares the camera's FK-pose path exactly.
__global__ void ScatterEnvLidarsKernel(ScatterFkSource fk,
                                       const SensorMountRow* __restrict__ mounts,
                                       uint32_t env_count,
                                       uint32_t sensors_per_env,
                                       LidarSensor* __restrict__ out_lidars) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = env_count * sensors_per_env;
    if (i >= total) return;
    const uint32_t env = i / sensors_per_env;
    const uint32_t local = i % sensors_per_env;

    const SensorMountRow row = mounts[local];
    const Vec3 off_pos{row.local_offset[0], row.local_offset[1], row.local_offset[2]};
    const Transform* fk_pose = ResolveFkPose(fk, row.kind, row.row, env);
    const Transform world = ComposeWorld(fk_pose, off_pos, row.local_offset[3],
                                         row.local_offset[4], row.local_offset[5],
                                         row.local_offset[6]);
    LidarSensor s;
    s.origin[0] = world.position.x;
    s.origin[1] = world.position.y;
    s.origin[2] = world.position.z;
    s.rotation[0] = world.rotation.w;
    s.rotation[1] = world.rotation.x;
    s.rotation[2] = world.rotation.y;
    s.rotation[3] = world.rotation.z;
    s.az_count = row.az_count;
    s.el_count = row.el_count;
    s.az_min = row.az_min;
    s.az_max = row.az_max;
    s.el_min = row.el_min;
    s.el_max = row.el_max;
    s.min_range = row.min_range;
    s.max_range = row.max_range;
    out_lidars[i] = s;
}

}  // namespace

void ScatterEnvInstances(cudaStream_t stream,
                         const phi::ScatterFkSource& fk,
                         const phi::InstanceScatterRow* device_rows,
                         const uint32_t* device_blas_id,
                         const uint32_t* device_material_id,
                         const SensorBlasRef* device_blas_refs,
                         uint32_t env_count,
                         uint32_t instances_per_env,
                         DevInstance* out_instances,
                         collision::AABB* out_world_aabbs) {
    const uint32_t total = env_count * instances_per_env;
    if (total == 0u) return;
    const uint32_t kBlock = 128u;
    const uint32_t grid = (total + kBlock - 1u) / kBlock;
    phi::LaunchCuda(ScatterEnvInstancesKernel, dim3(grid), dim3(kBlock), 0u, stream,
                    fk, device_rows, device_blas_id, device_material_id,
                    device_blas_refs, env_count, instances_per_env, out_instances,
                    out_world_aabbs);
}

void ScatterEnvCameras(cudaStream_t stream,
                       const phi::ScatterFkSource& fk,
                       const SensorMountRow* device_mounts,
                       uint32_t env_count,
                       uint32_t sensors_per_env,
                       PinholeCamera* out_cameras) {
    const uint32_t total = env_count * sensors_per_env;
    if (total == 0u) return;
    const uint32_t kBlock = 128u;
    const uint32_t grid = (total + kBlock - 1u) / kBlock;
    phi::LaunchCuda(ScatterEnvCamerasKernel, dim3(grid), dim3(kBlock), 0u, stream, fk,
                    device_mounts, env_count, sensors_per_env, out_cameras);
}

void ScatterEnvLidars(cudaStream_t stream,
                      const phi::ScatterFkSource& fk,
                      const SensorMountRow* device_mounts,
                      uint32_t env_count,
                      uint32_t sensors_per_env,
                      LidarSensor* out_lidars) {
    const uint32_t total = env_count * sensors_per_env;
    if (total == 0u) return;
    const uint32_t kBlock = 128u;
    const uint32_t grid = (total + kBlock - 1u) / kBlock;
    phi::LaunchCuda(ScatterEnvLidarsKernel, dim3(grid), dim3(kBlock), 0u, stream, fk,
                    device_mounts, env_count, sensors_per_env, out_lidars);
}

std::vector<SensorMountRow> BuildSensorMountRows(
    const std::vector<scene::SensorDesc>& sensors) {
    constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;
    std::vector<SensorMountRow> rows;
    rows.reserve(sensors.size());
    for (const scene::SensorDesc& s : sensors) {
        SensorMountRow r;
        r.kind = static_cast<uint32_t>(s.mount) + 1u;  // Link=1, Body=2, Base=3
        r.row = s.mount_index;
        r.local_offset[0] = s.local_offset.position.x;
        r.local_offset[1] = s.local_offset.position.y;
        r.local_offset[2] = s.local_offset.position.z;
        r.local_offset[3] = s.local_offset.rotation.w;
        r.local_offset[4] = s.local_offset.rotation.x;
        r.local_offset[5] = s.local_offset.rotation.y;
        r.local_offset[6] = s.local_offset.rotation.z;
        if (s.type == scene::SensorType::Lidar ||
            s.type == scene::SensorType::RangeScan) {
            // The (az,el) fan rides the row; the camera intrinsics stay zero so the
            // camera scatter, if it ever sees this row, ignores it. Angles are stored
            // radians (the schema authors them radians; importers convert at parse).
            r.az_count = s.lidar.az_count;
            r.el_count = s.lidar.el_count;
            r.az_min = s.lidar.az_min;
            r.az_max = s.lidar.az_max;
            r.el_min = s.lidar.el_min;
            r.el_max = s.lidar.el_max;
            r.min_range = s.lidar.min_range;
            r.max_range = s.lidar.max_range;
        } else {
            r.fov_y = s.cam.vfov_degrees * kDegToRad;
            r.width = s.cam.width;
            r.height = s.cam.height;
            // The schema carries focal/principal-point via vfov+centered today; k1/k2/
            // distortion + clip ride straight through (defaults => byte-identical).
            r.k1 = s.cam.k1;
            r.k2 = s.cam.k2;
            r.distortion = s.cam.distortion;
            r.near_clip = s.cam.near_clip;
            r.far_clip = s.cam.far_clip;
        }
        rows.push_back(r);
    }
    return rows;
}

}  // namespace nuka::rt
