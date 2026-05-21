// ---------------------------------------------------------------------------
// nuka::sensor::gpu::cuda_sensors implementation
// ---------------------------------------------------------------------------

#include "sensor/gpu/cuda_sensors.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace nuka::sensor::gpu {

namespace {

constexpr uint32_t kInvalidBody = ~0u;

__device__ math::Vec3 MakeVec3(float x, float y, float z) {
    math::Vec3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

__device__ math::Quat MakeQuat(float w, float x, float y, float z) {
    math::Quat q;
    q.w = w;
    q.x = x;
    q.y = y;
    q.z = z;
    return q;
}

__device__ math::Vec3 Add(math::Vec3 a, math::Vec3 b) {
    return MakeVec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ math::Vec3 Sub(math::Vec3 a, math::Vec3 b) {
    return MakeVec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ math::Vec3 Scale(math::Vec3 v, float s) {
    return MakeVec3(v.x * s, v.y * s, v.z * s);
}

__device__ float Dot(math::Vec3 a, math::Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ math::Vec3 Cross(math::Vec3 a, math::Vec3 b) {
    return MakeVec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

__device__ float LengthSq(math::Vec3 v) {
    return Dot(v, v);
}

__device__ float Length(math::Vec3 v) {
    return sqrtf(LengthSq(v));
}

__device__ math::Vec3 Normalize(math::Vec3 v) {
    const float length = Length(v);
    if (length < 1.0e-8f) {
        return MakeVec3(1.0f, 0.0f, 0.0f);
    }
    return Scale(v, 1.0f / length);
}

__device__ math::Vec3 Rotate(math::Quat q, math::Vec3 v) {
    const math::Vec3 qv = MakeVec3(q.x, q.y, q.z);
    const math::Vec3 t = Scale(Cross(qv, v), 2.0f);
    return Add(Add(v, Scale(t, q.w)), Cross(qv, t));
}

__device__ math::Quat Mul(math::Quat a, math::Quat b) {
    return MakeQuat(
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w);
}

__device__ math::Vec3 TransformPoint(math::Transform transform, math::Vec3 point) {
    return Add(Rotate(transform.rotation, point), transform.position);
}

__device__ math::Transform Compose(math::Transform a, math::Transform b) {
    math::Transform result;
    result.position = TransformPoint(a, b.position);
    result.rotation = Mul(a.rotation, b.rotation);
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
    const math::Vec3 f = Normalize(forward);
    math::Vec3 right = Cross(f, up);
    if (LengthSq(right) < 1.0e-8f) {
        right = MakeVec3(0.0f, 0.0f, 1.0f);
    }
    right = Normalize(right);

    if (ray_count <= 1u || horizontal_fov <= 0.0f) {
        return f;
    }

    const float normalized =
        (static_cast<float>(index) / static_cast<float>(ray_count - 1u)) * 2.0f - 1.0f;
    const float angle = normalized * horizontal_fov * 0.5f;
    return Normalize(Add(Scale(f, cosf(angle)), Scale(right, sinf(angle))));
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

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

template <typename T>
phi::Buffer UploadVector(const std::vector<T>& values) {
    phi::Buffer buffer(values.size() * sizeof(T), phi::MemoryKind::Device);
    if (!values.empty()) {
        buffer.CopyFromHost(values.data(), values.size() * sizeof(T));
    }
    return buffer;
}

template <typename T>
std::vector<T> DownloadVector(const phi::Buffer& buffer, uint32_t count) {
    std::vector<T> values(count);
    if (!values.empty()) {
        buffer.CopyToHost(values.data(), values.size() * sizeof(T));
    }
    return values;
}

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

CudaImuResult QueryCudaImuSensor(const runtime::gpu::DeviceWorld& device_world,
                                 const std::vector<scene::BodyId>& body_ids) {
    if (!device_world.HasUploadedState()) {
        throw std::runtime_error("QueryCudaImuSensor requires uploaded DeviceWorld state");
    }

    const uint32_t sample_count = static_cast<uint32_t>(body_ids.size());
    auto body_id_buffer = UploadVector(body_ids);
    phi::Buffer sample_buffer(sample_count * sizeof(CudaImuSample), phi::MemoryKind::Device);

    if (sample_count > 0u) {
        constexpr uint32_t kBlockSize = 128u;
        const uint32_t grid = (sample_count + kBlockSize - 1u) / kBlockSize;
        QueryImuKernel<<<grid, kBlockSize>>>(
            sample_count,
            static_cast<const scene::BodyId*>(body_id_buffer.Data()),
            device_world.BodyCount(),
            device_world.DevicePoses(),
            device_world.DeviceAngularVelocities(),
            device_world.DeviceForces(),
            device_world.DeviceInvMasses(),
            static_cast<CudaImuSample*>(sample_buffer.Data()));
        CheckCuda(cudaGetLastError(), "QueryImuKernel launch");
        CheckCuda(cudaDeviceSynchronize(), "QueryImuKernel synchronize");
    }

    return CudaImuResult(sample_count, std::move(sample_buffer));
}

CudaLidarResult QueryCudaLidarSensor(const runtime::gpu::DeviceWorld& device_world,
                                     const CudaLidarOptions& options) {
    if (!device_world.HasUploadedState()) {
        throw std::runtime_error("QueryCudaLidarSensor requires uploaded DeviceWorld state");
    }

    phi::Buffer depth_buffer(options.ray_count * sizeof(float), phi::MemoryKind::Device);
    if (options.ray_count > 0u) {
        constexpr uint32_t kBlockSize = 128u;
        const uint32_t grid = (options.ray_count + kBlockSize - 1u) / kBlockSize;
        QueryLidarKernel<<<grid, kBlockSize>>>(
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
        CheckCuda(cudaDeviceSynchronize(), "QueryLidarKernel synchronize");
    }

    return CudaLidarResult(options.ray_count, std::move(depth_buffer));
}

} // namespace nuka::sensor::gpu
