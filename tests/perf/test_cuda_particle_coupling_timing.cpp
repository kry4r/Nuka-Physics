// ---------------------------------------------------------------------------
// Performance test: CUDA particle/deformable coupling timing
// ---------------------------------------------------------------------------

#include "runtime/gpu/cuda_particle_world.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <vector>

using namespace nuka;

TEST(CudaParticleCouplingTiming, PlaneAndSphereCouplingUnderOneSecond) {
    constexpr uint32_t kParticleCount = 8192u;
    constexpr uint32_t kIterationCount = 80u;

    runtime::gpu::CudaParticleSet particles;
    particles.positions.reserve(kParticleCount);
    particles.velocities.reserve(kParticleCount);
    particles.inv_masses.reserve(kParticleCount);
    particles.radii.reserve(kParticleCount);
    particles.phases.reserve(kParticleCount);

    for (uint32_t index = 0; index < kParticleCount; ++index) {
        const float x = static_cast<float>(index % 64u) * 0.025f - 0.8f;
        const float y = 0.05f + static_cast<float>((index / 64u) % 32u) * 0.015f;
        const float z = static_cast<float>(index / (64u * 32u)) * 0.025f - 0.05f;
        particles.positions.push_back({x, y, z});
        particles.velocities.push_back({0.1f, -0.25f, 0.0f});
        particles.inv_masses.push_back(1.0f);
        particles.radii.push_back(0.01f);
        particles.phases.push_back(index % 4u);
    }

    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleStepOptions options;
    options.gravity = {0.0f, -9.81f, 0.0f};
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    options.plane.enabled = true;
    options.plane.normal = {0.0f, 1.0f, 0.0f};
    options.plane.offset = 0.0f;
    options.plane.friction = 0.05f;
    options.plane.restitution = 0.0f;
    options.sphere.enabled = true;
    options.sphere.center = {0.0f, 0.12f, 0.0f};
    options.sphere.radius = 0.18f;
    options.sphere.friction = 0.05f;
    options.sphere.restitution = 0.0f;

    runtime::gpu::CudaParticleStepReport report;
    const auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t iteration = 0; iteration < kIterationCount; ++iteration) {
        report = runtime::gpu::StepCudaParticleWorld(device_particles, options);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(report.particle_count, kParticleCount);
    EXPECT_GT(report.contact_count, 0u);
    EXPECT_LE(report.max_penetration_after_solve, 1.0e-4f);
    EXPECT_LT(ms, 1000) << "CUDA particle coupling took " << ms << " ms";
}
