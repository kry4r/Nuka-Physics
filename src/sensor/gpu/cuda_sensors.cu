// ---------------------------------------------------------------------------
// nuka::sensor::gpu::cuda_sensors implementation
// ---------------------------------------------------------------------------

#include "sensor/gpu/cuda_sensors.cuh"

#include "math/cuda_vec_ops.cuh"
#include "phi/buffer_transfer.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nuka::sensor::gpu {

namespace {

constexpr uint32_t kInvalidBody = ~0u;

// Small-vector / quaternion primitives now come from the shared device math
// library (math/cuda_vec_ops.cuh). Bodies are bit-identical to the former local
// copies. The former local `Normalize` maps to NormalizeSqrt(., 1e-8f,
// MakeVec3(1,0,0)); `Rotate` -> RotateShort; `Mul` -> QuatMul (renamed at call
// sites below). Buffer helpers come from phi/buffer_transfer.hpp.
namespace mg = ::nuka::math::gpu;
using mg::Add;
using mg::Cross;
using mg::Dot;
using mg::Length;
using mg::LengthSq;
using mg::MakeQuat;
using mg::MakeVec3;
using mg::QuatMul;
using mg::RotateShort;
using mg::Scale;
using mg::Sub;

__device__ math::Vec3 TransformPoint(math::Transform transform, math::Vec3 point) {
    return Add(RotateShort(transform.rotation, point), transform.position);
}

__device__ math::Transform Compose(math::Transform a, math::Transform b) {
    math::Transform result;
    result.position = TransformPoint(a, b.position);
    result.rotation = QuatMul(a.rotation, b.rotation);
    return result;
}

__global__ void QueryImuKernel(uint32_t sample_count,
                               const scene::BodyId* body_ids,
                               uint32_t body_count,
                               const math::Transform* poses,
                               const math::Vec3* angular_velocities,
                               const math::Vec3* forces,
                               const float* inv_masses,
                               CudaImuSample* samples) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= sample_count) {
        return;
    }

    CudaImuSample sample;
    const scene::BodyId body_id = body_ids[index];
    if (body_id < body_count && body_id != kInvalidBody) {
        sample.position = poses[body_id].position;
        sample.angular_velocity = angular_velocities[body_id];
        sample.linear_acceleration = Scale(forces[body_id], inv_masses[body_id]);
    }
    samples[index] = sample;
}

__global__ void QueryBatchedImuKernel(uint32_t sample_count,
                                      const BatchedCudaBodyRequest* body_requests,
                                      uint32_t instance_count,
                                      uint32_t body_count_per_instance,
                                      const math::Transform* poses,
                                      const math::Vec3* angular_velocities,
                                      const math::Vec3* forces,
                                      const float* inv_masses,
                                      CudaImuSample* samples) {
    const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= sample_count) {
        return;
    }

    CudaImuSample sample;
    const BatchedCudaBodyRequest request = body_requests[index];
    if (request.instance_index < instance_count
        && request.body_id < body_count_per_instance
        && request.body_id != kInvalidBody) {
        const uint32_t flat_body =
            request.instance_index * body_count_per_instance + request.body_id;
        sample.position = poses[flat_body].position;
        sample.angular_velocity = angular_velocities[flat_body];
        sample.linear_acceleration =
            Scale(forces[flat_body], inv_masses[request.body_id]);
    }
    samples[index] = sample;
}

__device__ bool RaySphere(math::Vec3 origin,
                          math::Vec3 direction,
                          math::Vec3 center,
                          float radius,
                          float max_range,
                          float* out_t) {
    const math::Vec3 oc = Sub(origin, center);
    const float b = Dot(oc, direction);
    const float c = Dot(oc, oc) - radius * radius;
    const float discriminant = b * b - c;
    if (discriminant < 0.0f) {
        return false;
    }

    const float sqrt_disc = sqrtf(discriminant);
    float t = -b - sqrt_disc;
    if (t < 0.0f) {
        t = -b + sqrt_disc;
    }
    if (t < 0.0f || t > max_range) {
        return false;
    }
    *out_t = t;
    return true;
}

__device__ bool RayPlane(math::Vec3 origin,
                         math::Vec3 direction,
                         float plane_y,
                         float max_range,
                         float* out_t) {
    if (fabsf(direction.y) < 1.0e-6f) {
        return false;
    }
    const float t = (plane_y - origin.y) / direction.y;
    if (t < 0.0f || t > max_range) {
        return false;
    }
    *out_t = t;
    return true;
}

__device__ bool RayAabb(math::Vec3 origin,
                        math::Vec3 direction,
                        math::Vec3 min_point,
                        math::Vec3 max_point,
                        float max_range,
                        float* out_t) {
    float t_min = 0.0f;
    float t_max = max_range;

    const float origins[3] = {origin.x, origin.y, origin.z};
    const float dirs[3] = {direction.x, direction.y, direction.z};
    const float mins[3] = {min_point.x, min_point.y, min_point.z};
    const float maxs[3] = {max_point.x, max_point.y, max_point.z};

    for (int axis = 0; axis < 3; ++axis) {
        if (fabsf(dirs[axis]) < 1.0e-6f) {
            if (origins[axis] < mins[axis] || origins[axis] > maxs[axis]) {
                return false;
            }
            continue;
        }

        const float inv_dir = 1.0f / dirs[axis];
        float near_t = (mins[axis] - origins[axis]) * inv_dir;
        float far_t = (maxs[axis] - origins[axis]) * inv_dir;
        if (near_t > far_t) {
            const float tmp = near_t;
            near_t = far_t;
            far_t = tmp;
        }
        t_min = fmaxf(t_min, near_t);
        t_max = fminf(t_max, far_t);
        if (t_min > t_max) {
            return false;
        }
    }

    *out_t = t_min;
    return t_min >= 0.0f && t_min <= max_range;
}

__device__ math::Vec3 RayDirectionForIndex(math::Vec3 forward,
                                           math::Vec3 up,
                                           uint32_t index,
                                           uint32_t ray_count,
                                           float horizontal_fov) {
    const math::Vec3 f = mg::NormalizeSqrt(forward, 1.0e-8f, MakeVec3(1.0f, 0.0f, 0.0f));
    math::Vec3 right = Cross(f, up);
    if (LengthSq(right) < 1.0e-8f) {
        right = MakeVec3(0.0f, 0.0f, 1.0f);
    }
    right = mg::NormalizeSqrt(right, 1.0e-8f, MakeVec3(1.0f, 0.0f, 0.0f));

    if (ray_count <= 1u || horizontal_fov <= 0.0f) {
        return f;
    }

    const float normalized =
        (static_cast<float>(index) / static_cast<float>(ray_count - 1u)) * 2.0f - 1.0f;
    const float angle = normalized * horizontal_fov * 0.5f;
    return mg::NormalizeSqrt(Add(Scale(f, cosf(angle)), Scale(right, sinf(angle))),
                             1.0e-8f, MakeVec3(1.0f, 0.0f, 0.0f));
}

__global__ void QueryLidarKernel(uint32_t ray_count,
                                 math::Vec3 origin,
                                 math::Vec3 direction,
                                 math::Vec3 up,
                                 float range,
                                 float horizontal_fov,
                                 uint32_t shape_count,
                                 const scene::ShapeType* shape_types,
                                 const scene::BodyId* shape_body_ids,
                                 const math::Transform* shape_local_transforms,
                                 const math::Vec3* shape_half_extents,
                                 const float* shape_radii,
                                 const math::Transform* poses,
                                 float* depths) {
    const uint32_t ray_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray_index >= ray_count) {
        return;
    }

    const math::Vec3 ray_direction =
        RayDirectionForIndex(direction, up, ray_index, ray_count, horizontal_fov);
    float nearest = range;

    for (uint32_t shape_index = 0; shape_index < shape_count; ++shape_index) {
        const scene::BodyId body_id = shape_body_ids[shape_index];
        if (body_id == kInvalidBody) {
            continue;
        }
        const math::Transform shape_world =
            Compose(poses[body_id], shape_local_transforms[shape_index]);

        float hit = range;
        bool did_hit = false;
        const auto type = shape_types[shape_index];
        if (type == scene::ShapeType::Sphere) {
            did_hit = RaySphere(origin,
                                ray_direction,
                                shape_world.position,
                                shape_radii[shape_index],
                                nearest,
                                &hit);
        } else if (type == scene::ShapeType::Plane) {
            did_hit = RayPlane(origin, ray_direction, shape_world.position.y, nearest, &hit);
        } else {
            const math::Vec3 half_extents = shape_half_extents[shape_index];
            did_hit = RayAabb(origin,
                              ray_direction,
                              Sub(shape_world.position, half_extents),
                              Add(shape_world.position, half_extents),
                              nearest,
                              &hit);
        }

        if (did_hit && hit < nearest) {
            nearest = hit;
        }
    }

    depths[ray_index] = nearest;
}

__device__ uint32_t FindBatchedLidarQueryIndex(uint32_t flat_ray,
                                               uint32_t query_count,
                                               const uint32_t* ray_offsets) {
    for (uint32_t query_index = 0; query_index < query_count; ++query_index) {
        if (flat_ray >= ray_offsets[query_index] && flat_ray < ray_offsets[query_index + 1u]) {
            return query_index;
        }
    }
    return query_count;
}

__global__ void QueryBatchedLidarKernel(
    uint32_t total_ray_count,
    uint32_t query_count,
    const BatchedCudaLidarOptions* options,
    const uint32_t* ray_offsets,
    uint32_t instance_count,
    uint32_t body_count_per_instance,
    uint32_t shape_count_per_instance,
    const scene::ShapeType* shape_types,
    const scene::BodyId* shape_body_ids,
    const math::Transform* shape_local_transforms,
    const math::Vec3* shape_half_extents,
    const float* shape_radii,
    const math::Transform* poses,
    float* depths) {
    const uint32_t flat_ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (flat_ray >= total_ray_count) {
        return;
    }

    const uint32_t query_index = FindBatchedLidarQueryIndex(flat_ray, query_count, ray_offsets);
    if (query_index >= query_count) {
        return;
    }

    const BatchedCudaLidarOptions query = options[query_index];
    const uint32_t local_ray = flat_ray - ray_offsets[query_index];
    const math::Vec3 ray_direction =
        RayDirectionForIndex(query.direction,
                             query.up,
                             local_ray,
                             query.ray_count,
                             query.horizontal_fov_radians);
    float nearest = query.range;

    if (query.instance_index < instance_count) {
        for (uint32_t shape_index = 0; shape_index < shape_count_per_instance; ++shape_index) {
            const scene::BodyId local_body = shape_body_ids[shape_index];
            if (local_body == kInvalidBody || local_body >= body_count_per_instance) {
                continue;
            }
            const uint32_t flat_body =
                query.instance_index * body_count_per_instance + local_body;
            const math::Transform shape_world =
                Compose(poses[flat_body], shape_local_transforms[shape_index]);

            float hit = query.range;
            bool did_hit = false;
            const auto type = shape_types[shape_index];
            if (type == scene::ShapeType::Sphere) {
                did_hit = RaySphere(query.origin,
                                    ray_direction,
                                    shape_world.position,
                                    shape_radii[shape_index],
                                    nearest,
                                    &hit);
            } else if (type == scene::ShapeType::Plane) {
                did_hit = RayPlane(query.origin,
                                   ray_direction,
                                   shape_world.position.y,
                                   nearest,
                                   &hit);
            } else {
                const math::Vec3 half_extents = shape_half_extents[shape_index];
                did_hit = RayAabb(query.origin,
                                  ray_direction,
                                  Sub(shape_world.position, half_extents),
                                  Add(shape_world.position, half_extents),
                                  nearest,
                                  &hit);
            }

            if (did_hit && hit < nearest) {
                nearest = hit;
            }
        }
    }

    depths[flat_ray] = nearest;
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

// UploadVector / DownloadVector(buf, count) now come from the shared host
// buffer-transfer header (phi/buffer_transfer.hpp); the former local copies were
// byte-identical.
using ::nuka::phi::DownloadVector;
using ::nuka::phi::UploadVector;

} // namespace

CudaImuResult::CudaImuResult(uint32_t sample_count, phi::Buffer samples)
    : sample_count_(sample_count)
    , samples_(std::move(samples)) {}

std::vector<CudaImuSample> CudaImuResult::DownloadSamples() const {
    return DownloadVector<CudaImuSample>(samples_, sample_count_);
}

CudaLidarResult::CudaLidarResult(uint32_t ray_count, phi::Buffer depths)
    : ray_count_(ray_count)
    , depths_(std::move(depths)) {}

std::vector<float> CudaLidarResult::DownloadDepths() const {
    return DownloadVector<float>(depths_, ray_count_);
}

BatchedCudaLidarResult::BatchedCudaLidarResult(uint32_t query_count,
                                               uint32_t total_ray_count,
                                               phi::Buffer ray_offsets,
                                               phi::Buffer depths)
    : query_count_(query_count)
    , total_ray_count_(total_ray_count)
    , ray_offsets_(std::move(ray_offsets))
    , depths_(std::move(depths)) {}

std::vector<uint32_t> BatchedCudaLidarResult::DownloadRayOffsets() const {
    return DownloadVector<uint32_t>(ray_offsets_, query_count_ + 1u);
}

std::vector<float> BatchedCudaLidarResult::DownloadDepths() const {
    return DownloadVector<float>(depths_, total_ray_count_);
}

CudaImuResult QueryCudaImuSensor(const phi::DeviceContext& context,
                                 const runtime::gpu::DeviceWorld& device_world,
                                 const std::vector<scene::BodyId>& body_ids) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    if (!device_world.HasUploadedState()) {
        throw std::runtime_error("QueryCudaImuSensor requires uploaded DeviceWorld state");
    }

    const uint32_t sample_count = static_cast<uint32_t>(body_ids.size());
    auto body_id_buffer = UploadVector(body_ids);
    phi::Buffer sample_buffer(sample_count * sizeof(CudaImuSample), phi::MemoryKind::Device);

    if (sample_count > 0u) {
        constexpr uint32_t kBlockSize = 128u;
        const uint32_t grid = (sample_count + kBlockSize - 1u) / kBlockSize;
        QueryImuKernel<<<grid, kBlockSize, 0, stream>>>(
            sample_count,
            static_cast<const scene::BodyId*>(body_id_buffer.Data()),
            device_world.BodyCount(),
            device_world.DevicePoses(),
            device_world.DeviceAngularVelocities(),
            device_world.DeviceForces(),
            device_world.DeviceInvMasses(),
            static_cast<CudaImuSample*>(sample_buffer.Data()));
        CheckCuda(cudaGetLastError(), "QueryImuKernel launch");
        context.stream.Synchronize();
    }

    return CudaImuResult(sample_count, std::move(sample_buffer));
}

CudaImuResult QueryCudaImuSensor(const runtime::gpu::DeviceWorld& device_world,
                                 const std::vector<scene::BodyId>& body_ids) {
    auto context = phi::MakeDefaultDeviceContext();
    return QueryCudaImuSensor(context, device_world, body_ids);
}

CudaLidarResult QueryCudaLidarSensor(const phi::DeviceContext& context,
                                     const runtime::gpu::DeviceWorld& device_world,
                                     const CudaLidarOptions& options) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    if (!device_world.HasUploadedState()) {
        throw std::runtime_error("QueryCudaLidarSensor requires uploaded DeviceWorld state");
    }

    phi::Buffer depth_buffer(options.ray_count * sizeof(float), phi::MemoryKind::Device);
    if (options.ray_count > 0u) {
        constexpr uint32_t kBlockSize = 128u;
        const uint32_t grid = (options.ray_count + kBlockSize - 1u) / kBlockSize;
        QueryLidarKernel<<<grid, kBlockSize, 0, stream>>>(
            options.ray_count,
            options.origin,
            options.direction,
            options.up,
            options.range,
            options.horizontal_fov_radians,
            device_world.ShapeCount(),
            device_world.DeviceShapeTypes(),
            device_world.DeviceShapeBodyIds(),
            device_world.DeviceShapeLocalTransforms(),
            device_world.DeviceShapeHalfExtents(),
            device_world.DeviceShapeRadii(),
            device_world.DevicePoses(),
            static_cast<float*>(depth_buffer.Data()));
        CheckCuda(cudaGetLastError(), "QueryLidarKernel launch");
        context.stream.Synchronize();
    }

    return CudaLidarResult(options.ray_count, std::move(depth_buffer));
}

CudaLidarResult QueryCudaLidarSensor(const runtime::gpu::DeviceWorld& device_world,
                                     const CudaLidarOptions& options) {
    auto context = phi::MakeDefaultDeviceContext();
    return QueryCudaLidarSensor(context, device_world, options);
}

CudaImuResult QueryBatchedCudaImuSensor(
    const phi::DeviceContext& context,
    const runtime::gpu::BatchedDeviceWorld& batch,
    const std::vector<BatchedCudaBodyRequest>& body_requests) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    if (!batch.HasUploadedState()) {
        throw std::runtime_error(
            "QueryBatchedCudaImuSensor requires uploaded BatchedDeviceWorld state");
    }

    const uint32_t sample_count = static_cast<uint32_t>(body_requests.size());
    auto request_buffer = UploadVector(body_requests);
    phi::Buffer sample_buffer(sample_count * sizeof(CudaImuSample), phi::MemoryKind::Device);

    if (sample_count > 0u) {
        constexpr uint32_t kBlockSize = 128u;
        const uint32_t grid = (sample_count + kBlockSize - 1u) / kBlockSize;
        QueryBatchedImuKernel<<<grid, kBlockSize, 0, stream>>>(
            sample_count,
            static_cast<const BatchedCudaBodyRequest*>(request_buffer.Data()),
            batch.InstanceCount(),
            batch.BodyCountPerInstance(),
            batch.DevicePoses(),
            batch.DeviceAngularVelocities(),
            batch.DeviceForces(),
            batch.DeviceInvMasses(),
            static_cast<CudaImuSample*>(sample_buffer.Data()));
        CheckCuda(cudaGetLastError(), "QueryBatchedImuKernel launch");
        context.stream.Synchronize();
    }

    return CudaImuResult(sample_count, std::move(sample_buffer));
}

CudaImuResult QueryBatchedCudaImuSensor(
    const runtime::gpu::BatchedDeviceWorld& batch,
    const std::vector<BatchedCudaBodyRequest>& body_requests) {
    auto context = phi::MakeDefaultDeviceContext();
    return QueryBatchedCudaImuSensor(context, batch, body_requests);
}

BatchedCudaLidarResult QueryBatchedCudaLidarSensor(
    const phi::DeviceContext& context,
    const runtime::gpu::BatchedDeviceWorld& batch,
    const std::vector<BatchedCudaLidarOptions>& options) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    if (!batch.HasUploadedState() || !batch.HasUploadedShapeTables()) {
        throw std::runtime_error(
            "QueryBatchedCudaLidarSensor requires uploaded BatchedDeviceWorld state and shape tables");
    }

    std::vector<uint32_t> ray_offsets(options.size() + 1u, 0u);
    for (size_t index = 0; index < options.size(); ++index) {
        ray_offsets[index + 1u] = ray_offsets[index] + options[index].ray_count;
    }

    const uint32_t query_count = static_cast<uint32_t>(options.size());
    const uint32_t total_ray_count = ray_offsets.back();
    auto option_buffer = UploadVector(options);
    auto offset_buffer = UploadVector(ray_offsets);
    phi::Buffer depth_buffer(total_ray_count * sizeof(float), phi::MemoryKind::Device);

    if (total_ray_count > 0u) {
        constexpr uint32_t kBlockSize = 128u;
        const uint32_t grid = (total_ray_count + kBlockSize - 1u) / kBlockSize;
        QueryBatchedLidarKernel<<<grid, kBlockSize, 0, stream>>>(
            total_ray_count,
            query_count,
            static_cast<const BatchedCudaLidarOptions*>(option_buffer.Data()),
            static_cast<const uint32_t*>(offset_buffer.Data()),
            batch.InstanceCount(),
            batch.BodyCountPerInstance(),
            batch.ShapeCountPerInstance(),
            batch.DeviceShapeTypes(),
            batch.DeviceShapeBodyIds(),
            batch.DeviceShapeLocalTransforms(),
            batch.DeviceShapeHalfExtents(),
            batch.DeviceShapeRadii(),
            batch.DevicePoses(),
            static_cast<float*>(depth_buffer.Data()));
        CheckCuda(cudaGetLastError(), "QueryBatchedLidarKernel launch");
        context.stream.Synchronize();
    }

    return BatchedCudaLidarResult(query_count,
                                  total_ray_count,
                                  std::move(offset_buffer),
                                  std::move(depth_buffer));
}

BatchedCudaLidarResult QueryBatchedCudaLidarSensor(
    const runtime::gpu::BatchedDeviceWorld& batch,
    const std::vector<BatchedCudaLidarOptions>& options) {
    auto context = phi::MakeDefaultDeviceContext();
    return QueryBatchedCudaLidarSensor(context, batch, options);
}

} // namespace nuka::sensor::gpu
