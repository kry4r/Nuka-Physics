// ---------------------------------------------------------------------------
// CUDA particle/deformable coupling foundation tests.
// ---------------------------------------------------------------------------

#include "runtime/gpu/cuda_particle_world.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

using namespace nuka;

namespace {

runtime::gpu::CudaParticleSet BuildParticleSet() {
    runtime::gpu::CudaParticleSet particles;
    particles.positions = {
        {0.0f, 2.0f, 0.0f},
        {0.5f, 1.0f, 0.0f},
        {0.0f, 0.5f, 0.0f},
    };
    particles.velocities = {
        {0.0f, 0.0f, 0.0f},
        {0.0f, -1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
    };
    particles.inv_masses = {1.0f, 2.0f, 0.0f};
    particles.radii = {0.05f, 0.1f, 0.15f};
    particles.phases = {1u, 2u, 3u};
    return particles;
}

} // namespace

TEST(CudaParticleWorld, UploadsParticleStateAndDownloadsSnapshot) {
    auto device_particles = runtime::gpu::UploadCudaParticleWorld(BuildParticleSet());

    EXPECT_EQ(device_particles.ParticleCount(), 3u);
    EXPECT_GE(device_particles.DeviceBytes(),
              3u * (sizeof(math::Vec3) * 2u + sizeof(float) * 2u + sizeof(uint32_t)));

    const auto snapshot = device_particles.DownloadState();
    ASSERT_EQ(snapshot.positions.size(), 3u);
    ASSERT_EQ(snapshot.velocities.size(), 3u);
    ASSERT_EQ(snapshot.inv_masses.size(), 3u);
    ASSERT_EQ(snapshot.radii.size(), 3u);
    ASSERT_EQ(snapshot.phases.size(), 3u);

    EXPECT_FLOAT_EQ(snapshot.positions[0].y, 2.0f);
    EXPECT_FLOAT_EQ(snapshot.velocities[1].y, -1.0f);
    EXPECT_FLOAT_EQ(snapshot.inv_masses[1], 2.0f);
    EXPECT_FLOAT_EQ(snapshot.radii[2], 0.15f);
    EXPECT_EQ(snapshot.phases[2], 3u);
}

TEST(CudaParticleWorld, IntegratesGravityAndReportsEnergyOnDevice) {
    runtime::gpu::CudaParticleSet particles;
    particles.positions = {{0.0f, 1.0f, 0.0f}};
    particles.velocities = {{0.0f, 0.0f, 0.0f}};
    particles.inv_masses = {2.0f};
    particles.radii = {0.05f};
    particles.phases = {7u};

    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleStepOptions options;
    options.gravity = {0.0f, -10.0f, 0.0f};
    options.dt = 0.1f;
    options.step_count = 1u;
    options.plane.enabled = false;
    options.sphere.enabled = false;

    const auto report = runtime::gpu::StepCudaParticleWorld(device_particles, options);
    const auto snapshot = device_particles.DownloadState();

    ASSERT_EQ(snapshot.positions.size(), 1u);
    ASSERT_EQ(snapshot.velocities.size(), 1u);
    EXPECT_FLOAT_EQ(snapshot.velocities[0].y, -1.0f);
    EXPECT_FLOAT_EQ(snapshot.positions[0].y, 0.9f);

    EXPECT_EQ(report.particle_count, 1u);
    EXPECT_EQ(report.simulated_step_count, 1u);
    EXPECT_EQ(report.kernel_launch_count, 2u);
    EXPECT_FLOAT_EQ(report.max_speed, 1.0f);
    EXPECT_NEAR(report.kinetic_energy, 0.25f, 1.0e-5f);
    EXPECT_EQ(report.contact_count, 0u);
    EXPECT_FLOAT_EQ(report.max_penetration, 0.0f);
}

TEST(CudaParticleWorld, SolvesPlaneCouplingWithoutCpuStepper) {
    runtime::gpu::CudaParticleSet particles;
    particles.positions = {
        {0.0f, -0.25f, 0.0f},
        {0.0f, 0.25f, 0.0f},
    };
    particles.velocities = {
        {0.0f, -2.0f, 0.0f},
        {0.0f, -1.0f, 0.0f},
    };
    particles.inv_masses = {1.0f, 1.0f};
    particles.radii = {0.1f, 0.1f};
    particles.phases = {1u, 1u};

    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleStepOptions options;
    options.gravity = {0.0f, 0.0f, 0.0f};
    options.dt = 1.0f / 60.0f;
    options.step_count = 1u;
    options.plane.enabled = true;
    options.plane.normal = {0.0f, 1.0f, 0.0f};
    options.plane.offset = 0.0f;
    options.plane.friction = 0.25f;
    options.plane.restitution = 0.5f;
    options.sphere.enabled = false;

    const auto report = runtime::gpu::StepCudaParticleWorld(device_particles, options);
    const auto snapshot = device_particles.DownloadState();

    ASSERT_EQ(snapshot.positions.size(), 2u);
    EXPECT_GE(snapshot.positions[0].y, snapshot.radii[0] - 1.0e-5f);
    EXPECT_GT(snapshot.velocities[0].y, 0.0f);
    EXPECT_LT(std::fabs(snapshot.velocities[0].x), 1.0e-5f);

    EXPECT_EQ(report.contact_count, 1u);
    EXPECT_GT(report.max_penetration, 0.2f);
    EXPECT_EQ(report.kernel_launch_count, 2u);
}

TEST(CudaParticleWorld, SolvesSphereCouplingAndBoundsPenetration) {
    runtime::gpu::CudaParticleSet particles;
    particles.positions = {
        {0.0f, 0.0f, 0.0f},
        {2.0f, 0.0f, 0.0f},
    };
    particles.velocities = {
        {-1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f},
    };
    particles.inv_masses = {1.0f, 1.0f};
    particles.radii = {0.1f, 0.1f};
    particles.phases = {4u, 4u};

    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleStepOptions options;
    options.gravity = {0.0f, 0.0f, 0.0f};
    options.dt = 1.0f / 120.0f;
    options.step_count = 1u;
    options.plane.enabled = false;
    options.sphere.enabled = true;
    options.sphere.center = {0.0f, 0.0f, 0.0f};
    options.sphere.radius = 1.0f;
    options.sphere.friction = 0.1f;
    options.sphere.restitution = 0.25f;

    const auto report = runtime::gpu::StepCudaParticleWorld(device_particles, options);
    const auto snapshot = device_particles.DownloadState();

    ASSERT_EQ(snapshot.positions.size(), 2u);
    EXPECT_NEAR(snapshot.positions[0].x, -1.1f, 1.0e-4f);
    EXPECT_LT(snapshot.velocities[0].x, 0.0f);
    EXPECT_NEAR(snapshot.positions[1].x, 2.0f, 1.0e-4f);

    EXPECT_EQ(report.contact_count, 1u);
    EXPECT_NEAR(report.max_penetration, 1.1f, 1.0e-4f);
    EXPECT_LE(report.max_penetration_after_solve, 1.0e-4f);
}
