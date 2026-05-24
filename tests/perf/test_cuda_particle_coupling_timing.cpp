// ---------------------------------------------------------------------------
// Performance test: CUDA particle/deformable coupling timing
// ---------------------------------------------------------------------------

#include "runtime/gpu/cuda_particle_world.hpp"
#include "runtime/gpu/device_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <utility>
#include <vector>

using namespace nuka;

namespace {

runtime::gpu::DeviceWorld BuildDeviceSphereWorld(runtime::BuiltWorld& world) {
    scene::SceneIR scene;

    scene::RigidBodyRecord sphere_body;
    sphere_body.name = "coupled_sphere";
    sphere_body.mass = 8.0f;
    sphere_body.inertia = {1.0f, 1.0f, 1.0f};
    sphere_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    const auto body_id = scene.AddRigidBody(std::move(sphere_body));

    scene::CollisionShapeRecord sphere;
    sphere.body_id = body_id;
    sphere.type = scene::ShapeType::Sphere;
    sphere.local_transform.position = {0.0f, 0.12f, 0.0f};
    sphere.radius = 0.18f;
    scene.AddCollisionShape(std::move(sphere));

    world = runtime::BuildWorld(scene::CookScene(scene));
    auto device_world = runtime::gpu::UploadDeviceWorld(world.template_view);
    runtime::gpu::UploadDeviceState(device_world, world.instance);
    return device_world;
}

} // namespace

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

TEST(CudaParticleCouplingTiming, DeviceWorldRigidImpulseCouplingUnderOneSecond) {
    constexpr uint32_t kParticleCount = 4096u;
    constexpr uint32_t kIterationCount = 60u;

    runtime::gpu::CudaParticleSet particles;
    particles.positions.reserve(kParticleCount);
    particles.velocities.reserve(kParticleCount);
    particles.inv_masses.reserve(kParticleCount);
    particles.radii.reserve(kParticleCount);
    particles.phases.reserve(kParticleCount);

    for (uint32_t index = 0; index < kParticleCount; ++index) {
        const float x = static_cast<float>(index % 64u) * 0.006f - 0.19f;
        const float y = 0.12f + static_cast<float>((index / 64u) % 16u) * 0.004f;
        const float z = static_cast<float>(index / (64u * 16u)) * 0.006f - 0.02f;
        particles.positions.push_back({x, y, z});
        particles.velocities.push_back({-0.1f, 0.0f, 0.0f});
        particles.inv_masses.push_back(1.0f);
        particles.radii.push_back(0.004f);
        particles.phases.push_back(index % 4u);
    }

    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);
    runtime::BuiltWorld world;
    auto device_world = BuildDeviceSphereWorld(world);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    options.friction = 0.02f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = true;

    runtime::gpu::CudaParticleStepReport report;
    const auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t iteration = 0; iteration < kIterationCount; ++iteration) {
        report = runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
            device_particles, device_world, options);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(report.particle_count, kParticleCount);
    const auto rigid_state = device_world.DownloadState();
    ASSERT_FALSE(rigid_state.linear_velocities.empty());
    ASSERT_FALSE(rigid_state.angular_velocities.empty());
    EXPECT_LT(rigid_state.linear_velocities[0].x, 0.0f);
    EXPECT_GT(std::abs(rigid_state.angular_velocities[0].z), 1.0e-4f);
    EXPECT_LT(ms, 1000) << "CUDA DeviceWorld particle coupling took " << ms << " ms";
}
