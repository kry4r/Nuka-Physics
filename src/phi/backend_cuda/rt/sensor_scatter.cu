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

// Resolve the fk pose for one row in env `env` (env-major offsets). A null field
// ptr / out-of-range row -> no pose (the instance keeps cvl as world, fk=identity).
__device__ inline const Transform* ResolveFkPose(const ScatterFkSource& fk,
                                                 const InstanceScatterRow& row,
                                                 uint32_t env) {
    switch (row.kind) {
        case 1u:  // Link
            if (fk.link_pose != nullptr && row.row < fk.links_per_env) {
                return static_cast<const Transform*>(fk.link_pose) +
                       (env * fk.links_per_env + row.row);
            }
            return nullptr;
        case 2u:  // Body
            if (fk.body_pose != nullptr && row.row < fk.bodies_per_env) {
                return static_cast<const Transform*>(fk.body_pose) +
                       (env * fk.bodies_per_env + row.row);
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
    const Transform* fk_pose = ResolveFkPose(fk, row, env);
    Transform world;
    if (fk_pose == nullptr) {
        world.position = cvl_pos;
        world.rotation.w = cvl_qw;
        world.rotation.x = cvl_qx;
        world.rotation.y = cvl_qy;
        world.rotation.z = cvl_qz;
    } else {
        // world = fk * cvl: pos = fk.R.Rotate(cvl.pos)+fk.pos,
        // rot = (fk.R * cvl.R).Normalized() (math::Transform::operator*).
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
    }

    out_instances[i] = MakeDevInstance(world, ref.blas_nodes, ref.blas_leaf_count,
                                       ref.blas, i, material_id[local]);
    out_world_aabbs[i] = InstanceWorldAabb(ref.blas_leaf_count, ref.local_bound, world);
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

}  // namespace nuka::rt
