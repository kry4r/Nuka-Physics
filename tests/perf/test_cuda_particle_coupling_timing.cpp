// ---------------------------------------------------------------------------
// Performance test: CUDA particle/deformable coupling timing
// ---------------------------------------------------------------------------

#include "runtime/gpu/cuda_particle_world.hpp"
#include "runtime/gpu/device_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
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

runtime::gpu::DeviceWorld BuildDeviceBoxWorld(runtime::BuiltWorld& world) {
    scene::SceneIR scene;

    scene::RigidBodyRecord box_body;
    box_body.name = "coupled_box";
    box_body.mass = 8.0f;
    box_body.inertia = {1.5f, 1.5f, 1.5f};
    box_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    const auto body_id = scene.AddRigidBody(std::move(box_body));

    scene::CollisionShapeRecord box;
    box.body_id = body_id;
    box.type = scene::ShapeType::Box;
    box.half_extents = {0.2f, 0.25f, 0.2f};
    scene.AddCollisionShape(std::move(box));

    world = runtime::BuildWorld(scene::CookScene(scene));
    auto device_world = runtime::gpu::UploadDeviceWorld(world.template_view);
    runtime::gpu::UploadDeviceState(device_world, world.instance);
    return device_world;
}

runtime::gpu::DeviceWorld BuildDeviceCornerBoxWorld(runtime::BuiltWorld& world) {
    scene::SceneIR scene;

    scene::RigidBodyRecord floor_body;
    floor_body.name = "coupled_corner_floor";
    floor_body.mass = 8.0f;
    floor_body.inertia = {1.5f, 1.5f, 1.5f};
    floor_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    const auto floor_body_id = scene.AddRigidBody(std::move(floor_body));

    scene::CollisionShapeRecord floor_box;
    floor_box.body_id = floor_body_id;
    floor_box.type = scene::ShapeType::Box;
    floor_box.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(floor_box));

    scene::RigidBodyRecord side_body;
    side_body.name = "coupled_corner_side";
    side_body.mass = 8.0f;
    side_body.inertia = {1.5f, 1.5f, 1.5f};
    side_body.local_transform.position = {0.36f, 0.10f, 0.0f};
    const auto side_body_id = scene.AddRigidBody(std::move(side_body));

    scene::CollisionShapeRecord side_box;
    side_box.body_id = side_body_id;
    side_box.type = scene::ShapeType::Box;
    side_box.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(side_box));

    world = runtime::BuildWorld(scene::CookScene(scene));
    auto device_world = runtime::gpu::UploadDeviceWorld(world.template_view);
    runtime::gpu::UploadDeviceState(device_world, world.instance);
    return device_world;
}

runtime::gpu::DeviceWorld BuildDeviceCapsuleWorld(runtime::BuiltWorld& world) {
    scene::SceneIR scene;

    scene::RigidBodyRecord capsule_body;
    capsule_body.name = "coupled_capsule";
    capsule_body.mass = 8.0f;
    capsule_body.inertia = {1.5f, 1.5f, 1.5f};
    capsule_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    const auto body_id = scene.AddRigidBody(std::move(capsule_body));

    scene::CollisionShapeRecord capsule;
    capsule.body_id = body_id;
    capsule.type = scene::ShapeType::Capsule;
    capsule.radius = 0.18f;
    capsule.half_height = 0.35f;
    scene.AddCollisionShape(std::move(capsule));

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
    uint64_t row_solver_impulse_count = 0;
    float row_solver_impulse_magnitude = 0.0f;
    float row_solver_angular_impulse_magnitude = 0.0f;
    const auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t iteration = 0; iteration < kIterationCount; ++iteration) {
        report = runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
            device_particles, device_world, options);
        row_solver_impulse_count += report.coupling_row_solver_impulse_count;
        row_solver_impulse_magnitude += report.coupling_row_solver_impulse_magnitude;
        row_solver_angular_impulse_magnitude += report.rigid_angular_impulse_magnitude;
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(report.particle_count, kParticleCount);
    EXPECT_EQ(report.coupling_row_solver_launch_count, 1u);
    EXPECT_GT(row_solver_impulse_count, 0u);
    EXPECT_GT(row_solver_impulse_magnitude, 0.0f);
    EXPECT_GT(row_solver_angular_impulse_magnitude, 0.0f);
    const auto rigid_state = device_world.DownloadState();
    ASSERT_FALSE(rigid_state.linear_velocities.empty());
    ASSERT_FALSE(rigid_state.angular_velocities.empty());
    EXPECT_TRUE(std::isfinite(rigid_state.linear_velocities[0].x));
    EXPECT_TRUE(std::isfinite(rigid_state.angular_velocities[0].z));
    EXPECT_LT(ms, 1000) << "CUDA DeviceWorld particle coupling took " << ms << " ms";
}

TEST(CudaParticleCouplingTiming, DeviceWorldBoxRigidImpulseCouplingUnderOneSecond) {
    constexpr uint32_t kParticleCount = 4096u;
    constexpr uint32_t kIterationCount = 60u;

    runtime::gpu::CudaParticleSet particles;
    particles.positions.reserve(kParticleCount);
    particles.velocities.reserve(kParticleCount);
    particles.inv_masses.reserve(kParticleCount);
    particles.radii.reserve(kParticleCount);
    particles.phases.reserve(kParticleCount);

    for (uint32_t index = 0; index < kParticleCount; ++index) {
        const float x = 0.04f + static_cast<float>(index % 64u) * 0.002f;
        const float y = 0.251f + static_cast<float>((index / 64u) % 16u) * 0.0001f;
        const float z = static_cast<float>(index / (64u * 16u)) * 0.006f - 0.02f;
        particles.positions.push_back({x, y, z});
        particles.velocities.push_back({0.05f, -0.05f, 0.0f});
        particles.inv_masses.push_back(1.0f);
        particles.radii.push_back(0.004f);
        particles.phases.push_back(index % 4u);
    }

    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);
    runtime::BuiltWorld world;
    auto device_world = BuildDeviceBoxWorld(world);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    options.friction = 0.2f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = true;

    runtime::gpu::CudaParticleStepReport report;
    uint64_t row_solver_impulse_count = 0;
    uint64_t row_solver_friction_impulse_count = 0;
    float row_solver_impulse_magnitude = 0.0f;
    float row_solver_friction_impulse_magnitude = 0.0f;
    float row_solver_angular_impulse_magnitude = 0.0f;
    const auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t iteration = 0; iteration < kIterationCount; ++iteration) {
        report = runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
            device_particles, device_world, options);
        row_solver_impulse_count += report.coupling_row_solver_impulse_count;
        row_solver_friction_impulse_count +=
            report.coupling_row_solver_friction_impulse_count;
        row_solver_impulse_magnitude += report.coupling_row_solver_impulse_magnitude;
        row_solver_friction_impulse_magnitude +=
            report.coupling_row_solver_friction_impulse_magnitude;
        row_solver_angular_impulse_magnitude += report.rigid_angular_impulse_magnitude;
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(report.particle_count, kParticleCount);
    EXPECT_GT(report.contact_count, 0u);
    EXPECT_LE(report.max_penetration_after_solve, 1.0e-4f);
    EXPECT_EQ(report.coupling_row_solver_launch_count, 1u);
    EXPECT_GT(row_solver_impulse_count, 0u);
    EXPECT_GT(row_solver_impulse_magnitude, 0.0f);
    EXPECT_GT(row_solver_friction_impulse_count, 0u);
    EXPECT_GT(row_solver_friction_impulse_magnitude, 0.0f);
    EXPECT_GT(row_solver_angular_impulse_magnitude, 0.0f);

    const auto rigid_state = device_world.DownloadState();
    ASSERT_FALSE(rigid_state.linear_velocities.empty());
    ASSERT_FALSE(rigid_state.angular_velocities.empty());
    EXPECT_TRUE(std::isfinite(rigid_state.linear_velocities[0].y));
    EXPECT_TRUE(std::isfinite(rigid_state.angular_velocities[0].z));
    EXPECT_LT(ms, 1000) << "CUDA DeviceWorld box particle coupling took " << ms << " ms";
}

TEST(CudaParticleCouplingTiming, DeviceWorldCapsuleRigidImpulseCouplingUnderOneSecond) {
    constexpr uint32_t kParticleCount = 4096u;
    constexpr uint32_t kIterationCount = 60u;

    runtime::gpu::CudaParticleSet particles;
    particles.positions.reserve(kParticleCount);
    particles.velocities.reserve(kParticleCount);
    particles.inv_masses.reserve(kParticleCount);
    particles.radii.reserve(kParticleCount);
    particles.phases.reserve(kParticleCount);

    for (uint32_t index = 0; index < kParticleCount; ++index) {
        const float x = 0.16f + static_cast<float>(index % 64u) * 0.0003f;
        const float y = 0.05f + static_cast<float>((index / 64u) % 16u) * 0.014f;
        const float z = static_cast<float>(index / (64u * 16u)) * 0.006f - 0.02f;
        particles.positions.push_back({x, y, z});
        particles.velocities.push_back({-0.05f, 0.0f, 0.0f});
        particles.inv_masses.push_back(1.0f);
        particles.radii.push_back(0.004f);
        particles.phases.push_back(index % 4u);
    }

    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);
    runtime::BuiltWorld world;
    auto device_world = BuildDeviceCapsuleWorld(world);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    options.friction = 0.02f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = true;

    runtime::gpu::CudaParticleStepReport report;
    uint64_t row_solver_impulse_count = 0;
    float row_solver_impulse_magnitude = 0.0f;
    float row_solver_angular_impulse_magnitude = 0.0f;
    const auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t iteration = 0; iteration < kIterationCount; ++iteration) {
        report = runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
            device_particles, device_world, options);
        row_solver_impulse_count += report.coupling_row_solver_impulse_count;
        row_solver_impulse_magnitude += report.coupling_row_solver_impulse_magnitude;
        row_solver_angular_impulse_magnitude += report.rigid_angular_impulse_magnitude;
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(report.particle_count, kParticleCount);
    EXPECT_GT(report.contact_count, 0u);
    EXPECT_LE(report.max_penetration_after_solve, 1.0e-4f);
    EXPECT_EQ(report.coupling_row_solver_launch_count, 1u);
    EXPECT_GT(row_solver_impulse_count, 0u);
    EXPECT_GT(row_solver_impulse_magnitude, 0.0f);
    EXPECT_GT(row_solver_angular_impulse_magnitude, 0.0f);

    const auto rigid_state = device_world.DownloadState();
    ASSERT_FALSE(rigid_state.linear_velocities.empty());
    ASSERT_FALSE(rigid_state.angular_velocities.empty());
    EXPECT_TRUE(std::isfinite(rigid_state.linear_velocities[0].x));
    EXPECT_TRUE(std::isfinite(rigid_state.angular_velocities[0].z));
    EXPECT_LT(ms, 1000) << "CUDA DeviceWorld capsule particle coupling took " << ms << " ms";
}

TEST(CudaParticleCouplingTiming, DeviceWorldWarmStartDiagnosticsUnderOneSecond) {
    constexpr uint32_t kParticleCount = 4096u;
    constexpr uint32_t kIterationCount = 60u;

    runtime::gpu::CudaParticleSet particles;
    particles.positions.reserve(kParticleCount);
    particles.velocities.reserve(kParticleCount);
    particles.inv_masses.reserve(kParticleCount);
    particles.radii.reserve(kParticleCount);
    particles.phases.reserve(kParticleCount);

    for (uint32_t index = 0; index < kParticleCount; ++index) {
        const float x = 0.04f + static_cast<float>(index % 64u) * 0.002f;
        const float y = 0.251f + static_cast<float>((index / 64u) % 16u) * 0.0001f;
        const float z = static_cast<float>(index / (64u * 16u)) * 0.006f - 0.02f;
        particles.positions.push_back({x, y, z});
        particles.velocities.push_back({0.03f, -0.02f, 0.0f});
        particles.inv_masses.push_back(1.0f);
        particles.radii.push_back(0.004f);
        particles.phases.push_back(index % 4u);
    }

    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);
    runtime::BuiltWorld world;
    auto device_world = BuildDeviceBoxWorld(world);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    options.friction = 0.25f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = true;
    options.enable_coupling_warm_start = true;

    runtime::gpu::CudaParticleStepReport report;
    uint64_t row_solver_impulse_count = 0;
    float row_solver_impulse_magnitude = 0.0f;
    uint64_t warm_start_count = 0;
    uint64_t tangent_warm_start_count = 0;
    float warm_start_impulse_magnitude = 0.0f;
    float tangent_warm_start_impulse_magnitude = 0.0f;
    float max_coupling_normal_impulse = 0.0f;
    float coupling_force_magnitude = 0.0f;
    float coupling_torque_magnitude = 0.0f;
    const auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t iteration = 0; iteration < kIterationCount; ++iteration) {
        report = runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
            device_particles, device_world, options);
        row_solver_impulse_count += report.coupling_row_solver_impulse_count;
        row_solver_impulse_magnitude += report.coupling_row_solver_impulse_magnitude;
        warm_start_count += report.coupling_warm_start_count;
        tangent_warm_start_count += report.coupling_tangent_warm_start_count;
        warm_start_impulse_magnitude += report.coupling_warm_start_impulse_magnitude;
        tangent_warm_start_impulse_magnitude +=
            report.coupling_tangent_warm_start_impulse_magnitude;
        max_coupling_normal_impulse =
            std::max(max_coupling_normal_impulse, report.max_coupling_normal_impulse);
        coupling_force_magnitude += report.coupling_force_magnitude;
        coupling_torque_magnitude += report.coupling_torque_magnitude;
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    const auto coupling_state = device_particles.DownloadCouplingState();

    EXPECT_EQ(report.particle_count, kParticleCount);
    EXPECT_GT(report.contact_count, 0u);
    EXPECT_GT(warm_start_count, 0u);
    EXPECT_GT(tangent_warm_start_count, 0u);
    EXPECT_GT(warm_start_impulse_magnitude, 0.0f);
    EXPECT_GT(tangent_warm_start_impulse_magnitude, 0.0f);
    EXPECT_GT(max_coupling_normal_impulse, 0.0f);
    EXPECT_GT(report.coupling_active_slot_count, 0u);
    EXPECT_GT(coupling_force_magnitude, 0.0f);
    EXPECT_GT(coupling_torque_magnitude, 0.0f);
    EXPECT_EQ(report.coupling_row_solver_launch_count, 1u);
    EXPECT_GT(row_solver_impulse_count, 0u);
    EXPECT_GT(row_solver_impulse_magnitude, 0.0f);
    EXPECT_EQ(coupling_state.slot_count_per_particle,
              runtime::gpu::kCudaParticleCouplingSlotsPerParticle);
    ASSERT_EQ(coupling_state.normal_impulses.size(),
              kParticleCount *
                  static_cast<size_t>(runtime::gpu::kCudaParticleCouplingSlotsPerParticle));
    EXPECT_LT(ms, 1000) << "CUDA DeviceWorld warm-start coupling took " << ms << " ms";
}

TEST(CudaParticleCouplingTiming, DeviceWorldMultiSlotDiagnosticsUnderOneSecond) {
    constexpr uint32_t kParticleCount = 2048u;
    constexpr uint32_t kIterationCount = 60u;

    runtime::gpu::CudaParticleSet particles;
    particles.positions.reserve(kParticleCount);
    particles.velocities.reserve(kParticleCount);
    particles.inv_masses.reserve(kParticleCount);
    particles.radii.reserve(kParticleCount);
    particles.phases.reserve(kParticleCount);

    for (uint32_t index = 0; index < kParticleCount; ++index) {
        const float z = static_cast<float>(index % 64u) * 0.0002f - 0.0064f;
        const float x = 0.05f + static_cast<float>((index / 64u) % 8u) * 0.0003f;
        const float y = 0.34f + static_cast<float>(index / (64u * 8u)) * 0.00005f;
        particles.positions.push_back({x, y, z});
        particles.velocities.push_back({0.05f, -0.05f, 0.0f});
        particles.inv_masses.push_back(1.0f);
        particles.radii.push_back(0.1f);
        particles.phases.push_back(index % 4u);
    }

    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);
    runtime::BuiltWorld world;
    auto device_world = BuildDeviceCornerBoxWorld(world);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = {0.0f, -9.81f, 0.0f};
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    options.friction = 0.0f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = true;
    options.enable_coupling_warm_start = true;
    options.coupling_row_solver_iterations = 3u;

    runtime::gpu::CudaParticleStepReport report;
    uint64_t row_solver_impulse_count = 0;
    float row_solver_impulse_magnitude = 0.0f;
    float row_solver_max_normal_delta = 0.0f;
    float row_solver_max_tangent_delta = 0.0f;
    float row_solver_max_residual = 0.0f;
    uint64_t warm_start_count = 0;
    float coupling_force_magnitude = 0.0f;
    float coupling_torque_magnitude = 0.0f;
    const auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t iteration = 0; iteration < kIterationCount; ++iteration) {
        report = runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
            device_particles, device_world, options);
        row_solver_impulse_count += report.coupling_row_solver_impulse_count;
        row_solver_impulse_magnitude += report.coupling_row_solver_impulse_magnitude;
        row_solver_max_normal_delta =
            std::max(row_solver_max_normal_delta,
                     report.coupling_row_solver_max_iteration_normal_delta_impulse);
        row_solver_max_tangent_delta =
            std::max(row_solver_max_tangent_delta,
                     report.coupling_row_solver_max_iteration_tangent_delta_impulse);
        row_solver_max_residual =
            std::max(row_solver_max_residual,
                     report.coupling_row_solver_max_residual);
        warm_start_count += report.coupling_warm_start_count;
        coupling_force_magnitude += report.coupling_force_magnitude;
        coupling_torque_magnitude += report.coupling_torque_magnitude;
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    const auto coupling_state = device_particles.DownloadCouplingState();
    const auto rows = device_particles.DownloadCouplingRows();

    EXPECT_EQ(report.particle_count, kParticleCount);
    EXPECT_GE(report.contact_count, kParticleCount * 2u);
    EXPECT_GE(report.coupling_active_slot_count, kParticleCount * 2u);
    EXPECT_GT(warm_start_count, 0u);
    EXPECT_GT(coupling_force_magnitude, 0.0f);
    EXPECT_GT(coupling_torque_magnitude, 0.0f);
    EXPECT_EQ(report.coupling_row_solver_launch_count, 3u);
    EXPECT_EQ(report.coupling_row_solver_iteration_count, 3u);
    EXPECT_EQ(report.kernel_launch_count, 8u);
    EXPECT_EQ(report.coupling_scheduler_report.row_kind,
              runtime::gpu::CudaConstraintRowBufferKind::ParticleRigidCoupling);
    EXPECT_EQ(report.coupling_scheduler_report.row_layout,
              runtime::gpu::CudaConstraintRowLayout::ParticleRigidCouplingSlot);
    EXPECT_EQ(report.coupling_scheduler_report.schedule_mode,
              runtime::gpu::CudaConstraintRowScheduleMode::OwnerSerialSweep);
    EXPECT_EQ(report.coupling_scheduler_report.owner_count, kParticleCount);
    EXPECT_EQ(report.coupling_scheduler_report.row_count,
              kParticleCount *
                  static_cast<uint32_t>(runtime::gpu::kCudaParticleCouplingSlotsPerParticle));
    EXPECT_EQ(report.coupling_scheduler_report.configured_iterations, 3u);
    EXPECT_EQ(report.coupling_scheduler_report.executed_iterations, 3u);
    EXPECT_EQ(report.coupling_scheduler_report.solver_launch_count, 3u);
    EXPECT_EQ(report.coupling_scheduler_report.diagnostic_launch_count, 3u);
    EXPECT_GE(report.coupling_scheduler_report.active_row_count, kParticleCount * 2u);
    EXPECT_GT(row_solver_impulse_count, 0u);
    EXPECT_GT(row_solver_impulse_magnitude, 0.0f);
    EXPECT_EQ(report.coupling_row_solver_diagnostic_slot_count, 3u);
    EXPECT_EQ(report.coupling_scheduler_report.diagnostic_slot_count, 3u);
    EXPECT_EQ(report.coupling_scheduler_report.normal_impulse_count,
              report.coupling_row_solver_impulse_count);
    EXPECT_EQ(report.coupling_scheduler_report.tangent_impulse_count,
              report.coupling_row_solver_friction_impulse_count);
    EXPECT_NEAR(report.coupling_scheduler_report.max_normal_delta_impulse,
                report.coupling_row_solver_max_iteration_normal_delta_impulse,
                1.0e-5f);
    EXPECT_NEAR(report.coupling_scheduler_report.max_residual,
                report.coupling_row_solver_max_residual,
                1.0e-5f);
    EXPECT_GT(row_solver_max_normal_delta, 0.0f);
    EXPECT_GE(row_solver_max_tangent_delta, 0.0f);
    EXPECT_GE(row_solver_max_residual, 0.0f);
    EXPECT_TRUE(std::isfinite(row_solver_max_normal_delta));
    EXPECT_TRUE(std::isfinite(row_solver_max_tangent_delta));
    EXPECT_TRUE(std::isfinite(row_solver_max_residual));
    EXPECT_LE(report.coupling_row_solver_iteration_normal_delta_impulses[2],
              report.coupling_row_solver_iteration_normal_delta_impulses[0] + 1.0e-5f);
    EXPECT_LE(report.coupling_row_solver_iteration_max_residuals[2],
              report.coupling_row_solver_iteration_max_residuals[0] + 1.0e-5f);
    EXPECT_EQ(coupling_state.slot_count_per_particle,
              runtime::gpu::kCudaParticleCouplingSlotsPerParticle);
    ASSERT_EQ(coupling_state.normal_impulses.size(),
              kParticleCount *
                  static_cast<size_t>(runtime::gpu::kCudaParticleCouplingSlotsPerParticle));
    ASSERT_EQ(rows.rows.size(),
              kParticleCount *
                  static_cast<size_t>(runtime::gpu::kCudaParticleCouplingSlotsPerParticle));
    EXPECT_GE(coupling_state.normal_impulses[0], 0.0f);
    EXPECT_GE(coupling_state.normal_impulses[1], 0.0f);
    EXPECT_EQ(coupling_state.shape_indices[0], 0u);
    EXPECT_EQ(coupling_state.shape_indices[1], 1u);
    EXPECT_TRUE(rows.rows[0].active);
    EXPECT_TRUE(rows.rows[1].active);
    EXPECT_EQ(rows.rows[0].particle_index, 0u);
    EXPECT_EQ(rows.rows[1].particle_index, 0u);
    EXPECT_EQ(rows.rows[0].shape_index, 0u);
    EXPECT_EQ(rows.rows[1].shape_index, 1u);
    EXPECT_GT(rows.rows[0].effective_mass, 0.0f);
    EXPECT_GT(rows.rows[1].effective_mass, 0.0f);
    EXPECT_GE(rows.rows[0].normal_impulse, 0.0f);
    EXPECT_GE(rows.rows[1].normal_impulse, 0.0f);
    EXPECT_GT(rows.rows[0].position_error, 0.0f);
    EXPECT_GT(rows.rows[1].position_error, 0.0f);
    EXPECT_LT(ms, 1000) << "CUDA DeviceWorld multi-slot coupling took " << ms << " ms";
}
