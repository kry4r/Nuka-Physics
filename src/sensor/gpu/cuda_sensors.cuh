#pragma once
// ---------------------------------------------------------------------------
// nuka::sensor::gpu::cuda_sensors -- CUDA sensor query kernels
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "runtime/gpu/device_world.hpp"

#include <cstdint>
#include <vector>

namespace nuka::sensor::gpu {

struct CudaImuSample {
    math::Vec3 position = math::Vec3::Zero();
    math::Vec3 angular_velocity = math::Vec3::Zero();
    math::Vec3 linear_acceleration = math::Vec3::Zero();
};

class CudaImuResult {
public:
    CudaImuResult() = default;
    CudaImuResult(uint32_t sample_count, phi::Buffer samples);

    CudaImuResult(const CudaImuResult&) = delete;
    CudaImuResult& operator=(const CudaImuResult&) = delete;
    CudaImuResult(CudaImuResult&&) noexcept = default;
    CudaImuResult& operator=(CudaImuResult&&) noexcept = default;

    uint32_t SampleCount() const { return sample_count_; }
    std::vector<CudaImuSample> DownloadSamples() const;

private:
    uint32_t sample_count_ = 0;
    phi::Buffer samples_;
};

struct CudaLidarOptions {
    math::Vec3 origin = math::Vec3::Zero();
    math::Vec3 direction = math::Vec3::UnitX();
    math::Vec3 up = math::Vec3::UnitY();
    uint32_t ray_count = 0;
    float range = 100.0f;
    float horizontal_fov_radians = 0.0f;
};

class CudaLidarResult {
public:
    CudaLidarResult() = default;
    CudaLidarResult(uint32_t ray_count, phi::Buffer depths);

    CudaLidarResult(const CudaLidarResult&) = delete;
    CudaLidarResult& operator=(const CudaLidarResult&) = delete;
    CudaLidarResult(CudaLidarResult&&) noexcept = default;
    CudaLidarResult& operator=(CudaLidarResult&&) noexcept = default;

    uint32_t RayCount() const { return ray_count_; }
    std::vector<float> DownloadDepths() const;

private:
    uint32_t ray_count_ = 0;
    phi::Buffer depths_;
};

CudaImuResult QueryCudaImuSensor(const runtime::gpu::DeviceWorld& device_world,
                                 const std::vector<scene::BodyId>& body_ids);

CudaLidarResult QueryCudaLidarSensor(const runtime::gpu::DeviceWorld& device_world,
                                     const CudaLidarOptions& options);

} // namespace nuka::sensor::gpu
