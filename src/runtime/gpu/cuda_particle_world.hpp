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

inline constexpr uint32_t kInvalidCudaParticleCouplingShape = 0xffffffffu;
inline constexpr uint32_t kCudaParticleCouplingSlotsPerParticle = 4u;

struct CudaParticleCouplingState {
    uint32_t slot_count_per_particle = kCudaParticleCouplingSlotsPerParticle;
    std::vector<float> normal_impulses;
    std::vector<uint32_t> shape_indices;
};

struct CudaParticleCouplingConstraintRow {
    bool active = false;
    uint32_t particle_index = 0;
    uint32_t shape_index = kInvalidCudaParticleCouplingShape;
    scene::BodyId body_id = scene::kInvalidBody;
    math::Vec3 normal = math::Vec3::Zero();
    math::Vec3 contact_point = math::Vec3::Zero();
    math::Vec3 particle_linear_jacobian = math::Vec3::Zero();
    math::Vec3 body_linear_jacobian = math::Vec3::Zero();
    math::Vec3 body_angular_jacobian = math::Vec3::Zero();
    float rhs = 0.0f;
    float position_error = 0.0f;
    float effective_mass = 0.0f;
    float normal_impulse = 0.0f;
};

struct CudaParticleCouplingRowsState {
    uint32_t slot_count_per_particle = kCudaParticleCouplingSlotsPerParticle;
    std::vector<CudaParticleCouplingConstraintRow> rows;
};

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
    float rigid_angular_impulse_magnitude = 0.0f;
    uint32_t coupling_active_slot_count = 0;
    uint32_t coupling_warm_start_count = 0;
    float coupling_warm_start_impulse_magnitude = 0.0f;
    float max_coupling_normal_impulse = 0.0f;
    float coupling_force_magnitude = 0.0f;
    float coupling_torque_magnitude = 0.0f;
};

struct CudaParticleDeviceWorldCouplingOptions {
    math::Vec3 gravity = {0.0f, -9.81f, 0.0f};
    float dt = 1.0f / 60.0f;
    uint32_t step_count = 1u;
    float friction = 0.0f;
    float restitution = 0.0f;
    bool accumulate_rigid_impulses = true;
    bool enable_coupling_warm_start = true;
};

class CudaParticleWorld {
public:
    CudaParticleWorld() = default;
    CudaParticleWorld(uint32_t particle_count,
                      phi::Buffer positions,
                      phi::Buffer velocities,
                      phi::Buffer inv_masses,
                      phi::Buffer radii,
                      phi::Buffer phases,
                      phi::Buffer coupling_normal_impulses,
                      phi::Buffer coupling_shape_indices,
                      phi::Buffer coupling_rows);

    CudaParticleWorld(const CudaParticleWorld&) = delete;
    CudaParticleWorld& operator=(const CudaParticleWorld&) = delete;
    CudaParticleWorld(CudaParticleWorld&&) noexcept = default;
    CudaParticleWorld& operator=(CudaParticleWorld&&) noexcept = default;

    uint32_t ParticleCount() const { return particle_count_; }
    std::size_t DeviceBytes() const;
    bool HasUploadedState() const;
    CudaParticleState DownloadState() const;
    CudaParticleCouplingState DownloadCouplingState() const;
    CudaParticleCouplingRowsState DownloadCouplingRows() const;

    math::Vec3* DevicePositions();
    const math::Vec3* DevicePositions() const;
    math::Vec3* DeviceVelocities();
    const math::Vec3* DeviceVelocities() const;
    const float* DeviceInvMasses() const;
    const float* DeviceRadii() const;
    const uint32_t* DevicePhases() const;
    float* DeviceCouplingNormalImpulses();
    const float* DeviceCouplingNormalImpulses() const;
    uint32_t* DeviceCouplingShapeIndices();
    const uint32_t* DeviceCouplingShapeIndices() const;
    CudaParticleCouplingConstraintRow* DeviceCouplingRows();
    const CudaParticleCouplingConstraintRow* DeviceCouplingRows() const;

private:
    uint32_t particle_count_ = 0;
    phi::Buffer positions_;
    phi::Buffer velocities_;
    phi::Buffer inv_masses_;
    phi::Buffer radii_;
    phi::Buffer phases_;
    phi::Buffer coupling_normal_impulses_;
    phi::Buffer coupling_shape_indices_;
    phi::Buffer coupling_rows_;
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
