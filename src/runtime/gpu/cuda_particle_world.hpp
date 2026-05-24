#pragma once
// ---------------------------------------------------------------------------
// nuka::runtime::gpu::CudaParticleWorld -- CUDA-resident particle coupling
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "runtime/gpu/device_world.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nuka::runtime::gpu {

struct CudaParticleSet {
    std::vector<math::Vec3> positions;
    std::vector<math::Vec3> velocities;
    std::vector<float> inv_masses;
    std::vector<float> radii;
    std::vector<uint32_t> phases;
};

using CudaParticleState = CudaParticleSet;

struct CudaParticlePlaneCollider {
    bool enabled = true;
    math::Vec3 normal = math::Vec3::UnitY();
    float offset = 0.0f;
    float friction = 0.0f;
    float restitution = 0.0f;
};

struct CudaParticleSphereCollider {
    bool enabled = false;
    math::Vec3 center = math::Vec3::Zero();
    float radius = 1.0f;
    float friction = 0.0f;
    float restitution = 0.0f;
};

struct CudaParticleStepOptions {
    math::Vec3 gravity = {0.0f, -9.81f, 0.0f};
    float dt = 1.0f / 60.0f;
    uint32_t step_count = 1u;
    CudaParticlePlaneCollider plane;
    CudaParticleSphereCollider sphere;
};

struct CudaParticleStepReport {
    uint32_t particle_count = 0;
    uint32_t simulated_step_count = 0;
    uint32_t kernel_launch_count = 0;
    uint32_t contact_count = 0;
    uint32_t rigid_impulse_count = 0;
    float max_penetration = 0.0f;
    float max_penetration_after_solve = 0.0f;
    float max_speed = 0.0f;
    float kinetic_energy = 0.0f;
    float rigid_impulse_magnitude = 0.0f;
};

struct CudaParticleDeviceWorldCouplingOptions {
    math::Vec3 gravity = {0.0f, -9.81f, 0.0f};
    float dt = 1.0f / 60.0f;
    uint32_t step_count = 1u;
    float friction = 0.0f;
    float restitution = 0.0f;
    bool accumulate_rigid_impulses = true;
};

class CudaParticleWorld {
public:
    CudaParticleWorld() = default;
    CudaParticleWorld(uint32_t particle_count,
                      phi::Buffer positions,
                      phi::Buffer velocities,
                      phi::Buffer inv_masses,
                      phi::Buffer radii,
                      phi::Buffer phases);

    CudaParticleWorld(const CudaParticleWorld&) = delete;
    CudaParticleWorld& operator=(const CudaParticleWorld&) = delete;
    CudaParticleWorld(CudaParticleWorld&&) noexcept = default;
    CudaParticleWorld& operator=(CudaParticleWorld&&) noexcept = default;

    uint32_t ParticleCount() const { return particle_count_; }
    std::size_t DeviceBytes() const;
    bool HasUploadedState() const;
    CudaParticleState DownloadState() const;

    math::Vec3* DevicePositions();
    const math::Vec3* DevicePositions() const;
    math::Vec3* DeviceVelocities();
    const math::Vec3* DeviceVelocities() const;
    const float* DeviceInvMasses() const;
    const float* DeviceRadii() const;
    const uint32_t* DevicePhases() const;

private:
    uint32_t particle_count_ = 0;
    phi::Buffer positions_;
    phi::Buffer velocities_;
    phi::Buffer inv_masses_;
    phi::Buffer radii_;
    phi::Buffer phases_;
};

CudaParticleWorld UploadCudaParticleWorld(const CudaParticleSet& particles);

CudaParticleStepReport StepCudaParticleWorld(
    CudaParticleWorld& particle_world,
    const CudaParticleStepOptions& options = {});

CudaParticleStepReport StepCudaParticlesAgainstDeviceWorld(
    CudaParticleWorld& particle_world,
    DeviceWorld& device_world,
    const CudaParticleDeviceWorldCouplingOptions& options = {});

} // namespace nuka::runtime::gpu
