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

#include <cuda_runtime.h>

namespace nuka::rt {

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

// One sensor's mount binding + pinhole intrinsics, the camera kernel's per-row
// input (env-invariant). kind: 1=Link, 2=Body, 3=Base, 0=Static (no fk). row =
// index into the env's pose field; local_offset = pos3+quat(wxyz) mount->camera.
struct SensorMountRow {
    uint32_t kind = 0u;
    uint32_t row = 0u;
    float local_offset[7] = {0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    float fov_y = 0.0f;  // full vertical FOV, radians
    uint32_t width = 0u;
    uint32_t height = 0u;
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

// Lower scene::SensorDesc list to the SensorMountRow table (host-side, once). The
// vfov is converted degrees->radians (the PinholeCamera FULL vertical FOV).
std::vector<SensorMountRow> BuildSensorMountRows(const std::vector<scene::SensorDesc>& sensors);

}  // namespace nuka::rt
