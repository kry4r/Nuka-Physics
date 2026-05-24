// ---------------------------------------------------------------------------
// CUDA particle/deformable coupling foundation tests.
// ---------------------------------------------------------------------------

#include "runtime/gpu/cuda_particle_world.hpp"
#include "runtime/gpu/device_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <utility>
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

runtime::gpu::DeviceWorld UploadScene(scene::SceneIR scene, runtime::BuiltWorld& world) {
    world = runtime::BuildWorld(scene::CookScene(scene));
    auto device_world = runtime::gpu::UploadDeviceWorld(world.template_view);
    runtime::gpu::UploadDeviceState(device_world, world.instance);
    return device_world;
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

TEST(CudaParticleWorld, CouplesAgainstCookedPlaneShapeFromDeviceWorld) {
    scene::SceneIR scene;

    scene::RigidBodyRecord ground;
    ground.name = "raised_ground";
    ground.is_static = true;
    ground.local_transform.position = {0.0f, 0.25f, 0.0f};
    const auto ground_id = scene.AddRigidBody(std::move(ground));

    scene::CollisionShapeRecord plane;
    plane.body_id = ground_id;
    plane.type = scene::ShapeType::Plane;
    scene.AddCollisionShape(std::move(plane));

    runtime::BuiltWorld world;
    auto device_world = UploadScene(std::move(scene), world);

    runtime::gpu::CudaParticleSet particles;
    particles.positions = {{0.0f, 0.2f, 0.0f}};
    particles.velocities = {{0.0f, -1.0f, 0.0f}};
    particles.inv_masses = {1.0f};
    particles.radii = {0.05f};
    particles.phases = {1u};
    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 120.0f;
    options.step_count = 1u;
    options.friction = 0.0f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = false;

    const auto report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);
    const auto particle_state = device_particles.DownloadState();

    ASSERT_EQ(particle_state.positions.size(), 1u);
    EXPECT_NEAR(particle_state.positions[0].y, 0.3f, 1.0e-4f);
    EXPECT_GE(particle_state.velocities[0].y, -1.0e-5f);
    EXPECT_EQ(report.contact_count, 1u);
    EXPECT_GT(report.max_penetration, 0.05f);
    EXPECT_LE(report.max_penetration_after_solve, 1.0e-4f);
    EXPECT_EQ(report.rigid_impulse_count, 0u);
}

TEST(CudaParticleWorld, AccumulatesImpulseIntoDynamicSphereBodyOnDevice) {
    scene::SceneIR scene;

    scene::RigidBodyRecord sphere_body;
    sphere_body.name = "robot_link_sphere";
    sphere_body.mass = 1.0f;
    sphere_body.inertia = {1.0f, 1.0f, 1.0f};
    sphere_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    const auto sphere_body_id = scene.AddRigidBody(std::move(sphere_body));

    scene::CollisionShapeRecord sphere;
    sphere.body_id = sphere_body_id;
    sphere.type = scene::ShapeType::Sphere;
    sphere.radius = 0.5f;
    scene.AddCollisionShape(std::move(sphere));

    runtime::BuiltWorld world;
    auto device_world = UploadScene(std::move(scene), world);

    runtime::gpu::CudaParticleSet particles;
    particles.positions = {{0.4f, 0.0f, 0.0f}};
    particles.velocities = {{-2.0f, 0.0f, 0.0f}};
    particles.inv_masses = {1.0f};
    particles.radii = {0.1f};
    particles.phases = {2u};
    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 120.0f;
    options.step_count = 1u;
    options.friction = 0.0f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = true;

    const auto report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);
    const auto particle_state = device_particles.DownloadState();
    const auto rigid_state = device_world.DownloadState();

    ASSERT_EQ(particle_state.positions.size(), 1u);
    ASSERT_EQ(rigid_state.linear_velocities.size(), 1u);
    EXPECT_NEAR(particle_state.positions[0].x, 0.6f, 1.0e-4f);
    EXPECT_GT(particle_state.velocities[0].x, -2.0f);
    EXPECT_LT(particle_state.velocities[0].x, 0.0f);
    EXPECT_LT(rigid_state.linear_velocities[sphere_body_id].x, -0.9f);
    EXPECT_EQ(report.contact_count, 1u);
    EXPECT_EQ(report.rigid_impulse_count, 1u);
    EXPECT_GT(report.rigid_impulse_magnitude, 0.0f);
    EXPECT_LE(report.max_penetration_after_solve, 1.0e-4f);
}

TEST(CudaParticleWorld, OffCenterParticleImpulseChangesRigidAngularVelocity) {
    scene::SceneIR scene;

    scene::RigidBodyRecord sphere_body;
    sphere_body.name = "off_center_robot_link";
    sphere_body.mass = 2.0f;
    sphere_body.inertia = {0.5f, 0.5f, 0.5f};
    sphere_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    const auto sphere_body_id = scene.AddRigidBody(std::move(sphere_body));

    scene::CollisionShapeRecord sphere;
    sphere.body_id = sphere_body_id;
    sphere.type = scene::ShapeType::Sphere;
    sphere.local_transform.position = {0.0f, 0.2f, 0.0f};
    sphere.radius = 0.5f;
    scene.AddCollisionShape(std::move(sphere));

    runtime::BuiltWorld world;
    auto device_world = UploadScene(std::move(scene), world);

    runtime::gpu::CudaParticleSet particles;
    particles.positions = {{0.4f, 0.2f, 0.0f}};
    particles.velocities = {{-2.0f, 0.0f, 0.0f}};
    particles.inv_masses = {1.0f};
    particles.radii = {0.1f};
    particles.phases = {3u};
    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 120.0f;
    options.step_count = 1u;
    options.friction = 0.0f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = true;

    const auto report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);
    const auto rigid_state = device_world.DownloadState();

    ASSERT_EQ(rigid_state.angular_velocities.size(), 1u);
    EXPECT_EQ(report.contact_count, 1u);
    EXPECT_EQ(report.rigid_impulse_count, 1u);
    EXPECT_GT(report.rigid_impulse_magnitude, 0.0f);
    EXPECT_GT(std::abs(rigid_state.angular_velocities[sphere_body_id].z), 1.0e-4f);
}

TEST(CudaParticleWorld, CouplesAgainstCookedBoxShapeFromDeviceWorld) {
    scene::SceneIR scene;

    scene::RigidBodyRecord box_body;
    box_body.name = "robot_link_box";
    box_body.mass = 2.0f;
    box_body.inertia = {0.75f, 0.75f, 0.75f};
    box_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    const auto box_body_id = scene.AddRigidBody(std::move(box_body));

    scene::CollisionShapeRecord box_shape;
    box_shape.body_id = box_body_id;
    box_shape.type = scene::ShapeType::Box;
    box_shape.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(box_shape));

    runtime::BuiltWorld world;
    auto device_world = UploadScene(std::move(scene), world);

    runtime::gpu::CudaParticleSet particles;
    particles.positions = {{0.2f, 0.34f, 0.0f}};
    particles.velocities = {{0.0f, -2.0f, 0.0f}};
    particles.inv_masses = {1.0f};
    particles.radii = {0.1f};
    particles.phases = {4u};
    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 120.0f;
    options.step_count = 1u;
    options.friction = 0.0f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = true;

    const auto report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);
    const auto particle_state = device_particles.DownloadState();
    const auto rigid_state = device_world.DownloadState();

    ASSERT_EQ(particle_state.positions.size(), 1u);
    ASSERT_EQ(rigid_state.linear_velocities.size(), 1u);
    ASSERT_EQ(rigid_state.angular_velocities.size(), 1u);
    EXPECT_NEAR(particle_state.positions[0].y, 0.35f, 1.0e-4f);
    EXPECT_GT(particle_state.velocities[0].y, -2.0f);
    EXPECT_LT(rigid_state.linear_velocities[box_body_id].y, 0.0f);

    const float contact_x = 0.2f;
    const float rigid_contact_normal_speed =
        rigid_state.linear_velocities[box_body_id].y +
        rigid_state.angular_velocities[box_body_id].z * contact_x;
    EXPECT_NEAR(particle_state.velocities[0].y - rigid_contact_normal_speed,
                0.0f,
                1.0e-4f);
    EXPECT_GT(std::abs(rigid_state.angular_velocities[box_body_id].z), 1.0e-4f);
    EXPECT_EQ(report.contact_count, 1u);
    EXPECT_EQ(report.rigid_impulse_count, 1u);
    EXPECT_GT(report.rigid_impulse_magnitude, 0.0f);
    EXPECT_GT(report.rigid_angular_impulse_magnitude, 0.0f);
    EXPECT_LE(report.max_penetration_after_solve, 1.0e-4f);
}

TEST(CudaParticleWorld, ReportsInteriorCookedBoxPenetrationToNearestFace) {
    scene::SceneIR scene;

    scene::RigidBodyRecord box_body;
    box_body.name = "static_box_boundary";
    box_body.is_static = true;
    const auto box_body_id = scene.AddRigidBody(std::move(box_body));

    scene::CollisionShapeRecord box_shape;
    box_shape.body_id = box_body_id;
    box_shape.type = scene::ShapeType::Box;
    box_shape.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(box_shape));

    runtime::BuiltWorld world;
    auto device_world = UploadScene(std::move(scene), world);

    runtime::gpu::CudaParticleSet particles;
    particles.positions = {{0.2f, 0.0f, 0.0f}};
    particles.velocities = {{0.0f, 0.0f, 0.0f}};
    particles.inv_masses = {1.0f};
    particles.radii = {0.05f};
    particles.phases = {5u};
    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 120.0f;
    options.step_count = 1u;
    options.friction = 0.0f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = false;

    const auto report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);
    const auto particle_state = device_particles.DownloadState();

    ASSERT_EQ(particle_state.positions.size(), 1u);
    EXPECT_NEAR(particle_state.positions[0].x, 0.3f, 1.0e-4f);
    EXPECT_GE(report.max_penetration, 0.099f);
    EXPECT_LE(report.max_penetration_after_solve, 1.0e-4f);
    EXPECT_EQ(report.contact_count, 1u);
}

TEST(CudaParticleWorld, CouplesAgainstCookedCapsuleShapeFromDeviceWorld) {
    scene::SceneIR scene;

    scene::RigidBodyRecord capsule_body;
    capsule_body.name = "robot_link_capsule";
    capsule_body.mass = 2.0f;
    capsule_body.inertia = {0.75f, 0.75f, 0.75f};
    capsule_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    const auto capsule_body_id = scene.AddRigidBody(std::move(capsule_body));

    scene::CollisionShapeRecord capsule_shape;
    capsule_shape.body_id = capsule_body_id;
    capsule_shape.type = scene::ShapeType::Capsule;
    capsule_shape.radius = 0.20f;
    capsule_shape.half_height = 0.50f;
    scene.AddCollisionShape(std::move(capsule_shape));

    runtime::BuiltWorld world;
    auto device_world = UploadScene(std::move(scene), world);

    runtime::gpu::CudaParticleSet particles;
    particles.positions = {{0.15f, 0.30f, 0.0f}};
    particles.velocities = {{-2.0f, 0.0f, 0.0f}};
    particles.inv_masses = {1.0f};
    particles.radii = {0.05f};
    particles.phases = {6u};
    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 120.0f;
    options.step_count = 1u;
    options.friction = 0.0f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = true;

    const auto report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);
    const auto particle_state = device_particles.DownloadState();
    const auto rigid_state = device_world.DownloadState();

    ASSERT_EQ(particle_state.positions.size(), 1u);
    ASSERT_EQ(rigid_state.linear_velocities.size(), 1u);
    ASSERT_EQ(rigid_state.angular_velocities.size(), 1u);
    EXPECT_NEAR(particle_state.positions[0].x, 0.25f, 1.0e-4f);
    EXPECT_NEAR(particle_state.positions[0].y, 0.30f, 1.0e-4f);
    EXPECT_GT(particle_state.velocities[0].x, -2.0f);
    EXPECT_LT(rigid_state.linear_velocities[capsule_body_id].x, 0.0f);

    const float contact_y = 0.30f;
    const float rigid_contact_normal_speed =
        rigid_state.linear_velocities[capsule_body_id].x -
        rigid_state.angular_velocities[capsule_body_id].z * contact_y;
    EXPECT_NEAR(particle_state.velocities[0].x - rigid_contact_normal_speed,
                0.0f,
                1.0e-4f);
    EXPECT_GT(std::abs(rigid_state.angular_velocities[capsule_body_id].z), 1.0e-4f);
    EXPECT_EQ(report.contact_count, 1u);
    EXPECT_EQ(report.rigid_impulse_count, 1u);
    EXPECT_GT(report.rigid_impulse_magnitude, 0.0f);
    EXPECT_GT(report.rigid_angular_impulse_magnitude, 0.0f);
    EXPECT_LE(report.max_penetration_after_solve, 1.0e-4f);
}
