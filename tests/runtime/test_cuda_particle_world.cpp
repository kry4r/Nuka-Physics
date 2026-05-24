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
    EXPECT_EQ(report.rigid_impulse_count, 1u);
    EXPECT_EQ(report.coupling_row_solver_launch_count, 1u);
    EXPECT_EQ(report.coupling_row_solver_impulse_count, 1u);
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

TEST(CudaParticleWorld, PersistsDeviceWorldCouplingNormalImpulseForWarmStart) {
    scene::SceneIR scene;

    scene::RigidBodyRecord box_body;
    box_body.name = "warm_start_robot_link";
    box_body.mass = 4.0f;
    box_body.inertia = {1.0f, 1.0f, 1.0f};
    const auto box_body_id = scene.AddRigidBody(std::move(box_body));

    scene::CollisionShapeRecord box_shape;
    box_shape.body_id = box_body_id;
    box_shape.type = scene::ShapeType::Box;
    box_shape.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(box_shape));

    runtime::BuiltWorld world;
    auto device_world = UploadScene(std::move(scene), world);

    runtime::gpu::CudaParticleSet particles;
    particles.positions = {{0.05f, 0.34f, 0.0f}};
    particles.velocities = {{0.0f, -0.05f, 0.0f}};
    particles.inv_masses = {1.0f};
    particles.radii = {0.1f};
    particles.phases = {8u};
    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    const auto initial_coupling = device_particles.DownloadCouplingState();
    EXPECT_EQ(initial_coupling.slot_count_per_particle,
              runtime::gpu::kCudaParticleCouplingSlotsPerParticle);
    ASSERT_EQ(initial_coupling.normal_impulses.size(),
              runtime::gpu::kCudaParticleCouplingSlotsPerParticle);
    ASSERT_EQ(initial_coupling.shape_indices.size(),
              runtime::gpu::kCudaParticleCouplingSlotsPerParticle);
    EXPECT_FLOAT_EQ(initial_coupling.normal_impulses[0], 0.0f);
    EXPECT_EQ(initial_coupling.shape_indices[0],
              runtime::gpu::kInvalidCudaParticleCouplingShape);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = {0.0f, -9.81f, 0.0f};
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    options.friction = 0.0f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = true;
    options.enable_coupling_warm_start = true;

    const auto first_report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);
    const auto stored_after_first = device_particles.DownloadCouplingState();

    ASSERT_EQ(stored_after_first.normal_impulses.size(),
              runtime::gpu::kCudaParticleCouplingSlotsPerParticle);
    ASSERT_EQ(stored_after_first.shape_indices.size(),
              runtime::gpu::kCudaParticleCouplingSlotsPerParticle);
    EXPECT_EQ(first_report.contact_count, 1u);
    EXPECT_EQ(first_report.coupling_active_slot_count, 1u);
    EXPECT_EQ(first_report.coupling_warm_start_count, 0u);
    EXPECT_FLOAT_EQ(first_report.coupling_warm_start_impulse_magnitude, 0.0f);
    EXPECT_GT(first_report.max_coupling_normal_impulse, 0.0f);
    EXPECT_GT(first_report.coupling_force_magnitude, 0.0f);
    EXPECT_GT(first_report.coupling_torque_magnitude, 0.0f);
    EXPECT_GT(stored_after_first.normal_impulses[0], 0.0f);
    EXPECT_EQ(stored_after_first.shape_indices[0], 0u);

    const auto second_report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);
    const auto stored_after_second = device_particles.DownloadCouplingState();

    EXPECT_EQ(second_report.contact_count, 1u);
    EXPECT_EQ(second_report.coupling_active_slot_count, 1u);
    EXPECT_EQ(second_report.coupling_warm_start_count, 1u);
    EXPECT_GT(second_report.coupling_warm_start_impulse_magnitude, 0.0f);
    EXPECT_GT(second_report.max_coupling_normal_impulse, 0.0f);
    EXPECT_EQ(stored_after_second.shape_indices[0], 0u);
    EXPECT_GT(stored_after_second.normal_impulses[0], 0.0f);
}

TEST(CudaParticleWorld, ClearsDeviceWorldCouplingImpulseWhenContactSeparates) {
    scene::SceneIR scene;

    scene::RigidBodyRecord sphere_body;
    sphere_body.name = "separating_robot_link";
    sphere_body.mass = 3.0f;
    sphere_body.inertia = {1.0f, 1.0f, 1.0f};
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
    particles.velocities = {{-3.0f, 0.0f, 0.0f}};
    particles.inv_masses = {1.0f};
    particles.radii = {0.1f};
    particles.phases = {9u};
    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 30.0f;
    options.step_count = 1u;
    options.friction = 0.0f;
    options.restitution = 1.0f;
    options.accumulate_rigid_impulses = true;
    options.enable_coupling_warm_start = true;

    const auto first_report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);
    auto coupling_state = device_particles.DownloadCouplingState();

    ASSERT_EQ(coupling_state.normal_impulses.size(),
              runtime::gpu::kCudaParticleCouplingSlotsPerParticle);
    ASSERT_EQ(coupling_state.shape_indices.size(),
              runtime::gpu::kCudaParticleCouplingSlotsPerParticle);
    EXPECT_EQ(first_report.contact_count, 1u);
    EXPECT_GT(coupling_state.normal_impulses[0], 0.0f);
    EXPECT_EQ(coupling_state.shape_indices[0], 0u);

    const auto second_report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);
    coupling_state = device_particles.DownloadCouplingState();

    EXPECT_EQ(second_report.contact_count, 0u);
    EXPECT_EQ(second_report.coupling_active_slot_count, 0u);
    EXPECT_EQ(second_report.coupling_warm_start_count, 0u);
    EXPECT_FLOAT_EQ(second_report.coupling_warm_start_impulse_magnitude, 0.0f);
    for (uint32_t slot = 0; slot < runtime::gpu::kCudaParticleCouplingSlotsPerParticle; ++slot) {
        EXPECT_FLOAT_EQ(coupling_state.normal_impulses[slot], 0.0f);
        EXPECT_EQ(coupling_state.shape_indices[slot],
                  runtime::gpu::kInvalidCudaParticleCouplingShape);
    }
}

TEST(CudaParticleWorld, StoresMultipleDeviceWorldCouplingSlotsForOneParticle) {
    scene::SceneIR scene;

    scene::RigidBodyRecord floor_box_body;
    floor_box_body.name = "floor_like_robot_link";
    floor_box_body.mass = 4.0f;
    floor_box_body.inertia = {1.0f, 1.0f, 1.0f};
    floor_box_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    const auto floor_body_id = scene.AddRigidBody(std::move(floor_box_body));

    scene::CollisionShapeRecord floor_box;
    floor_box.body_id = floor_body_id;
    floor_box.type = scene::ShapeType::Box;
    floor_box.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(floor_box));

    scene::RigidBodyRecord side_box_body;
    side_box_body.name = "side_wall_robot_link";
    side_box_body.mass = 4.0f;
    side_box_body.inertia = {1.0f, 1.0f, 1.0f};
    side_box_body.local_transform.position = {0.36f, 0.10f, 0.0f};
    const auto side_body_id = scene.AddRigidBody(std::move(side_box_body));

    scene::CollisionShapeRecord side_box;
    side_box.body_id = side_body_id;
    side_box.type = scene::ShapeType::Box;
    side_box.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(side_box));

    runtime::BuiltWorld world;
    auto device_world = UploadScene(std::move(scene), world);

    runtime::gpu::CudaParticleSet particles;
    particles.positions = {{0.05f, 0.34f, 0.0f}};
    particles.velocities = {{0.05f, -0.05f, 0.0f}};
    particles.inv_masses = {1.0f};
    particles.radii = {0.1f};
    particles.phases = {10u};
    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = {0.0f, -9.81f, 0.0f};
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    options.friction = 0.0f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = true;
    options.enable_coupling_warm_start = true;

    const auto first_report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);
    const auto first_coupling = device_particles.DownloadCouplingState();
    const auto first_rigid_state = device_world.DownloadState();

    ASSERT_EQ(first_coupling.normal_impulses.size(),
              runtime::gpu::kCudaParticleCouplingSlotsPerParticle);
    ASSERT_EQ(first_coupling.shape_indices.size(),
              runtime::gpu::kCudaParticleCouplingSlotsPerParticle);
    ASSERT_EQ(first_rigid_state.linear_velocities.size(), 2u);
    ASSERT_EQ(first_rigid_state.angular_velocities.size(), 2u);

    EXPECT_EQ(first_report.contact_count, 2u);
    EXPECT_EQ(first_report.coupling_active_slot_count, 2u);
    EXPECT_EQ(first_report.coupling_warm_start_count, 0u);
    EXPECT_GT(first_report.coupling_force_magnitude, 0.0f);
    EXPECT_GT(first_report.coupling_torque_magnitude, 0.0f);
    EXPECT_GT(first_coupling.normal_impulses[0], 0.0f);
    EXPECT_GT(first_coupling.normal_impulses[1], 0.0f);
    EXPECT_EQ(first_coupling.shape_indices[0], 0u);
    EXPECT_EQ(first_coupling.shape_indices[1], 1u);
    EXPECT_LT(first_rigid_state.linear_velocities[floor_body_id].y, 0.0f);
    EXPECT_GT(first_rigid_state.linear_velocities[side_body_id].x, 0.0f);
    EXPECT_GT(std::abs(first_rigid_state.angular_velocities[floor_body_id].z), 1.0e-4f);

    const auto second_report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);
    const auto second_coupling = device_particles.DownloadCouplingState();

    EXPECT_EQ(second_report.contact_count, 2u);
    EXPECT_EQ(second_report.coupling_active_slot_count, 2u);
    EXPECT_EQ(second_report.coupling_warm_start_count, 2u);
    EXPECT_GT(second_report.coupling_warm_start_impulse_magnitude, 0.0f);
    EXPECT_GT(second_report.coupling_force_magnitude, 0.0f);
    EXPECT_GT(second_report.coupling_torque_magnitude, 0.0f);
    EXPECT_EQ(second_coupling.shape_indices[0], 0u);
    EXPECT_EQ(second_coupling.shape_indices[1], 1u);
}

TEST(CudaParticleWorld, AssemblesDeviceWorldCouplingConstraintRowsOnCuda) {
    scene::SceneIR scene;

    scene::RigidBodyRecord floor_box_body;
    floor_box_body.name = "floor_constraint_row_link";
    floor_box_body.mass = 4.0f;
    floor_box_body.inertia = {1.0f, 1.0f, 1.0f};
    floor_box_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    const auto floor_body_id = scene.AddRigidBody(std::move(floor_box_body));

    scene::CollisionShapeRecord floor_box;
    floor_box.body_id = floor_body_id;
    floor_box.type = scene::ShapeType::Box;
    floor_box.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(floor_box));

    scene::RigidBodyRecord side_box_body;
    side_box_body.name = "side_constraint_row_link";
    side_box_body.mass = 4.0f;
    side_box_body.inertia = {1.0f, 1.0f, 1.0f};
    side_box_body.local_transform.position = {0.36f, 0.10f, 0.0f};
    const auto side_body_id = scene.AddRigidBody(std::move(side_box_body));

    scene::CollisionShapeRecord side_box;
    side_box.body_id = side_body_id;
    side_box.type = scene::ShapeType::Box;
    side_box.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(side_box));

    runtime::BuiltWorld world;
    auto device_world = UploadScene(std::move(scene), world);

    runtime::gpu::CudaParticleSet particles;
    particles.positions = {{0.05f, 0.34f, 0.0f}};
    particles.velocities = {{0.05f, -0.05f, 0.0f}};
    particles.inv_masses = {1.0f};
    particles.radii = {0.1f};
    particles.phases = {11u};
    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = {0.0f, -9.81f, 0.0f};
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    options.friction = 0.0f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = true;
    options.enable_coupling_warm_start = true;

    const auto report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);
    const auto rows = device_particles.DownloadCouplingRows();

    ASSERT_EQ(rows.slot_count_per_particle,
              runtime::gpu::kCudaParticleCouplingSlotsPerParticle);
    ASSERT_EQ(rows.rows.size(), runtime::gpu::kCudaParticleCouplingSlotsPerParticle);
    EXPECT_EQ(report.contact_count, 2u);
    EXPECT_EQ(report.coupling_active_slot_count, 2u);

    const auto& floor_row = rows.rows[0];
    const auto& side_row = rows.rows[1];
    EXPECT_TRUE(floor_row.active);
    EXPECT_TRUE(side_row.active);
    EXPECT_EQ(floor_row.particle_index, 0u);
    EXPECT_EQ(side_row.particle_index, 0u);
    EXPECT_EQ(floor_row.shape_index, 0u);
    EXPECT_EQ(side_row.shape_index, 1u);
    EXPECT_EQ(floor_row.body_id, floor_body_id);
    EXPECT_EQ(side_row.body_id, side_body_id);
    EXPECT_GT(floor_row.normal_impulse, 0.0f);
    EXPECT_GT(side_row.normal_impulse, 0.0f);
    EXPECT_GT(floor_row.effective_mass, 0.0f);
    EXPECT_GT(side_row.effective_mass, 0.0f);
    EXPECT_GT(floor_row.position_error, 0.0f);
    EXPECT_GT(side_row.position_error, 0.0f);
    EXPECT_NEAR(floor_row.normal.Length(), 1.0f, 1.0e-4f);
    EXPECT_NEAR(side_row.normal.Length(), 1.0f, 1.0e-4f);
    EXPECT_GT(std::abs(floor_row.body_angular_jacobian.z), 1.0e-4f);
    EXPECT_GT(std::abs(side_row.body_angular_jacobian.z), 1.0e-4f);

    for (uint32_t slot = 2u; slot < runtime::gpu::kCudaParticleCouplingSlotsPerParticle; ++slot) {
        EXPECT_FALSE(rows.rows[slot].active);
        EXPECT_EQ(rows.rows[slot].shape_index,
                  runtime::gpu::kInvalidCudaParticleCouplingShape);
    }
}

TEST(CudaParticleWorld, ExposesCudaConstraintRowBufferViewForScheduler) {
    scene::SceneIR scene;

    scene::RigidBodyRecord floor_box_body;
    floor_box_body.name = "floor_row_buffer_view_link";
    floor_box_body.mass = 4.0f;
    floor_box_body.inertia = {1.0f, 1.0f, 1.0f};
    floor_box_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    scene.AddRigidBody(std::move(floor_box_body));

    scene::CollisionShapeRecord floor_box;
    floor_box.body_id = 0u;
    floor_box.type = scene::ShapeType::Box;
    floor_box.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(floor_box));

    scene::RigidBodyRecord side_box_body;
    side_box_body.name = "side_row_buffer_view_link";
    side_box_body.mass = 4.0f;
    side_box_body.inertia = {1.0f, 1.0f, 1.0f};
    side_box_body.local_transform.position = {0.36f, 0.10f, 0.0f};
    scene.AddRigidBody(std::move(side_box_body));

    scene::CollisionShapeRecord side_box;
    side_box.body_id = 1u;
    side_box.type = scene::ShapeType::Box;
    side_box.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(side_box));

    runtime::BuiltWorld world;
    auto device_world = UploadScene(std::move(scene), world);

    runtime::gpu::CudaParticleSet particles;
    particles.positions = {{0.05f, 0.34f, 0.0f}};
    particles.velocities = {{0.05f, -0.05f, 0.0f}};
    particles.inv_masses = {1.0f};
    particles.radii = {0.1f};
    particles.phases = {17u};
    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = {0.0f, -9.81f, 0.0f};
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    options.friction = 0.0f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = true;
    options.enable_coupling_warm_start = true;
    options.solve_coupling_rows_on_cuda = true;

    const auto report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);
    const auto row_buffer = device_particles.ConstraintRowBuffer();
    const auto rows = device_particles.DownloadCouplingRows();

    EXPECT_EQ(row_buffer.kind,
              runtime::gpu::CudaConstraintRowBufferKind::ParticleRigidCoupling);
    EXPECT_EQ(row_buffer.device_rows,
              static_cast<void*>(device_particles.DeviceCouplingRows()));
    EXPECT_EQ(row_buffer.owner_count, device_particles.ParticleCount());
    EXPECT_EQ(row_buffer.rows_per_owner,
              runtime::gpu::kCudaParticleCouplingSlotsPerParticle);
    EXPECT_EQ(row_buffer.row_count, rows.rows.size());
    EXPECT_EQ(row_buffer.row_stride_bytes,
              sizeof(runtime::gpu::CudaParticleCouplingConstraintRow));
    EXPECT_EQ(report.coupling_active_slot_count, 2u);
    ASSERT_GE(rows.rows.size(), 2u);
    EXPECT_TRUE(rows.rows[0].active);
    EXPECT_TRUE(rows.rows[1].active);
}

TEST(CudaParticleWorld, ExecutesDeviceWorldCouplingRowsWithCudaSolverKernel) {
    scene::SceneIR scene;

    scene::RigidBodyRecord box_body;
    box_body.name = "row_solver_robot_link";
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
    options.solve_coupling_rows_on_cuda = true;

    const auto report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);
    const auto particle_state = device_particles.DownloadState();
    const auto rigid_state = device_world.DownloadState();
    const auto rows = device_particles.DownloadCouplingRows();

    ASSERT_EQ(particle_state.positions.size(), 1u);
    ASSERT_EQ(particle_state.velocities.size(), 1u);
    ASSERT_EQ(rigid_state.linear_velocities.size(), 1u);
    ASSERT_EQ(rigid_state.angular_velocities.size(), 1u);
    ASSERT_GE(rows.rows.size(), 1u);

    const auto& row = rows.rows[0];
    EXPECT_TRUE(row.active);
    EXPECT_EQ(row.body_id, box_body_id);
    EXPECT_GT(row.normal_impulse, 0.0f);
    EXPECT_EQ(report.contact_count, 1u);
    EXPECT_EQ(report.rigid_impulse_count, 1u);
    EXPECT_EQ(report.coupling_row_solver_launch_count, 1u);
    EXPECT_EQ(report.coupling_row_solver_impulse_count, 1u);
    EXPECT_GT(report.coupling_row_solver_impulse_magnitude, 0.0f);
    EXPECT_GT(report.rigid_impulse_magnitude, 0.0f);
    EXPECT_GT(report.rigid_angular_impulse_magnitude, 0.0f);

    const float contact_x = 0.2f;
    const float rigid_contact_normal_speed =
        rigid_state.linear_velocities[box_body_id].y +
        rigid_state.angular_velocities[box_body_id].z * contact_x;
    EXPECT_NEAR(particle_state.velocities[0].y - rigid_contact_normal_speed,
                0.0f,
                1.0e-4f);
    EXPECT_LT(rigid_state.linear_velocities[box_body_id].y, 0.0f);
    EXPECT_GT(std::abs(rigid_state.angular_velocities[box_body_id].z), 1.0e-4f);
}

TEST(CudaParticleWorld, AccumulatesMultipleCudaCouplingRowsIntoOneParticleVelocity) {
    scene::SceneIR scene;

    scene::RigidBodyRecord floor_box_body;
    floor_box_body.name = "floor_particle_row_velocity_link";
    floor_box_body.mass = 4.0f;
    floor_box_body.inertia = {1.0f, 1.0f, 1.0f};
    floor_box_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    scene.AddRigidBody(std::move(floor_box_body));

    scene::CollisionShapeRecord floor_box;
    floor_box.body_id = 0u;
    floor_box.type = scene::ShapeType::Box;
    floor_box.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(floor_box));

    scene::RigidBodyRecord side_box_body;
    side_box_body.name = "side_particle_row_velocity_link";
    side_box_body.mass = 4.0f;
    side_box_body.inertia = {1.0f, 1.0f, 1.0f};
    side_box_body.local_transform.position = {0.36f, 0.10f, 0.0f};
    scene.AddRigidBody(std::move(side_box_body));

    scene::CollisionShapeRecord side_box;
    side_box.body_id = 1u;
    side_box.type = scene::ShapeType::Box;
    side_box.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(side_box));

    runtime::BuiltWorld world;
    auto device_world = UploadScene(std::move(scene), world);

    runtime::gpu::CudaParticleSet particles;
    particles.positions = {{0.05f, 0.34f, 0.0f}};
    particles.velocities = {{0.05f, -1.0f, 0.0f}};
    particles.inv_masses = {1.0f};
    particles.radii = {0.1f};
    particles.phases = {12u};
    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    options.friction = 0.0f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = true;
    options.enable_coupling_warm_start = true;
    options.solve_coupling_rows_on_cuda = true;

    const auto report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);
    const auto particle_state = device_particles.DownloadState();
    const auto rows = device_particles.DownloadCouplingRows();

    ASSERT_EQ(particle_state.velocities.size(), 1u);
    ASSERT_GE(rows.rows.size(), 2u);
    ASSERT_TRUE(rows.rows[0].active);
    ASSERT_TRUE(rows.rows[1].active);

    EXPECT_EQ(report.contact_count, 2u);
    EXPECT_EQ(report.coupling_row_solver_launch_count, 1u);
    EXPECT_EQ(report.coupling_row_solver_impulse_count, 2u);
    EXPECT_GT(rows.rows[0].normal_impulse, 0.0f);
    EXPECT_GT(rows.rows[1].normal_impulse, 0.0f);

    const math::Vec3 expected_delta =
        rows.rows[0].particle_linear_jacobian * rows.rows[0].normal_impulse +
        rows.rows[1].particle_linear_jacobian * rows.rows[1].normal_impulse;
    EXPECT_NEAR(particle_state.velocities[0].x,
                particles.velocities[0].x + expected_delta.x,
                1.0e-4f);
    EXPECT_NEAR(particle_state.velocities[0].y,
                particles.velocities[0].y + expected_delta.y,
                1.0e-4f);
    EXPECT_NEAR(particle_state.velocities[0].z,
                particles.velocities[0].z + expected_delta.z,
                1.0e-4f);
}

TEST(CudaParticleWorld, IteratesDeviceWorldCouplingRowsWithCudaScheduler) {
    scene::SceneIR scene;

    scene::RigidBodyRecord floor_box_body;
    floor_box_body.name = "floor_scheduler_row_link";
    floor_box_body.mass = 4.0f;
    floor_box_body.inertia = {1.0f, 1.0f, 1.0f};
    floor_box_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    scene.AddRigidBody(std::move(floor_box_body));

    scene::CollisionShapeRecord floor_box;
    floor_box.body_id = 0u;
    floor_box.type = scene::ShapeType::Box;
    floor_box.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(floor_box));

    scene::RigidBodyRecord side_box_body;
    side_box_body.name = "side_scheduler_row_link";
    side_box_body.mass = 4.0f;
    side_box_body.inertia = {1.0f, 1.0f, 1.0f};
    side_box_body.local_transform.position = {0.36f, 0.10f, 0.0f};
    scene.AddRigidBody(std::move(side_box_body));

    scene::CollisionShapeRecord side_box;
    side_box.body_id = 1u;
    side_box.type = scene::ShapeType::Box;
    side_box.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(side_box));

    runtime::BuiltWorld world;
    auto device_world = UploadScene(std::move(scene), world);

    runtime::gpu::CudaParticleSet particles;
    particles.positions = {{0.05f, 0.34f, 0.0f}};
    particles.velocities = {{0.25f, -1.0f, 0.0f}};
    particles.inv_masses = {1.0f};
    particles.radii = {0.1f};
    particles.phases = {15u};
    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 240.0f;
    options.step_count = 1u;
    options.friction = 0.5f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = true;
    options.enable_coupling_warm_start = true;
    options.solve_coupling_rows_on_cuda = true;
    options.coupling_row_solver_iterations = 3u;

    const auto report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);
    const auto rows = device_particles.DownloadCouplingRows();

    ASSERT_GE(rows.rows.size(), 2u);
    ASSERT_TRUE(rows.rows[0].active);
    ASSERT_TRUE(rows.rows[1].active);

    EXPECT_EQ(report.contact_count, 2u);
    EXPECT_EQ(report.coupling_active_slot_count, 2u);
    EXPECT_EQ(report.coupling_row_solver_launch_count, 3u);
    EXPECT_EQ(report.coupling_row_solver_iteration_count, 3u);
    EXPECT_EQ(report.kernel_launch_count, 8u);
    EXPECT_GT(report.coupling_row_solver_impulse_count, 0u);
    EXPECT_GT(report.coupling_row_solver_impulse_magnitude, 0.0f);
    EXPECT_EQ(report.coupling_row_solver_diagnostic_slot_count, 3u);
    EXPECT_GT(report.coupling_row_solver_max_iteration_normal_delta_impulse, 0.0f);
    EXPECT_GE(report.coupling_row_solver_max_iteration_tangent_delta_impulse, 0.0f);
    EXPECT_GE(report.coupling_row_solver_max_residual, 0.0f);

    const float first_delta =
        report.coupling_row_solver_iteration_normal_delta_impulses[0] +
        report.coupling_row_solver_iteration_tangent_delta_impulses[0];
    const float last_delta =
        report.coupling_row_solver_iteration_normal_delta_impulses[2] +
        report.coupling_row_solver_iteration_tangent_delta_impulses[2];
    EXPECT_GT(first_delta, 0.0f);
    EXPECT_LE(last_delta, first_delta + 1.0e-5f);
    EXPECT_LE(report.coupling_row_solver_iteration_max_residuals[2],
              report.coupling_row_solver_iteration_max_residuals[0] + 1.0e-5f);
    for (uint32_t iteration = 0u; iteration < 3u; ++iteration) {
        EXPECT_TRUE(std::isfinite(
            report.coupling_row_solver_iteration_normal_delta_impulses[iteration]));
        EXPECT_TRUE(std::isfinite(
            report.coupling_row_solver_iteration_tangent_delta_impulses[iteration]));
        EXPECT_TRUE(std::isfinite(
            report.coupling_row_solver_iteration_max_residuals[iteration]));
        EXPECT_GE(report.coupling_row_solver_iteration_max_residuals[iteration], 0.0f);
    }

    for (uint32_t slot = 0u; slot < 2u; ++slot) {
        const auto& row = rows.rows[slot];
        EXPECT_EQ(row.particle_index, 0u);
        EXPECT_EQ(row.shape_index, slot);
        EXPECT_GE(row.normal_impulse, 0.0f);
        EXPECT_LE(std::abs(row.tangent_impulse_0),
                  options.friction * row.normal_impulse + 1.0e-5f);
        EXPECT_LE(std::abs(row.tangent_impulse_1),
                  options.friction * row.normal_impulse + 1.0e-5f);
    }
}

TEST(CudaParticleWorld, RecordsMultiStepRowSolverDiagnosticsBeforeFixedSlotCap) {
    scene::SceneIR scene;

    scene::RigidBodyRecord floor_box_body;
    floor_box_body.name = "floor_many_scheduler_rows_link";
    floor_box_body.mass = 4.0f;
    floor_box_body.inertia = {1.0f, 1.0f, 1.0f};
    floor_box_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    scene.AddRigidBody(std::move(floor_box_body));

    scene::CollisionShapeRecord floor_box;
    floor_box.body_id = 0u;
    floor_box.type = scene::ShapeType::Box;
    floor_box.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(floor_box));

    runtime::BuiltWorld world;
    auto device_world = UploadScene(std::move(scene), world);

    runtime::gpu::CudaParticleSet particles;
    particles.positions = {{0.0f, 0.34f, 0.0f}};
    particles.velocities = {{0.0f, -1.0f, 0.0f}};
    particles.inv_masses = {1.0f};
    particles.radii = {0.1f};
    particles.phases = {16u};
    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 240.0f;
    options.step_count = 2u;
    options.friction = 0.25f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = true;
    options.enable_coupling_warm_start = true;
    options.solve_coupling_rows_on_cuda = true;
    options.coupling_row_solver_iterations = 3u;

    const auto report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(device_particles, device_world, options);

    EXPECT_EQ(report.coupling_row_solver_iteration_count,
              options.step_count * options.coupling_row_solver_iterations);
    EXPECT_EQ(report.coupling_row_solver_diagnostic_slot_count,
              options.step_count * options.coupling_row_solver_iterations);
    EXPECT_GT(report.coupling_row_solver_iteration_normal_delta_impulses[0], 0.0f);
    for (uint32_t slot = 0u;
         slot < report.coupling_row_solver_diagnostic_slot_count;
         ++slot) {
        EXPECT_TRUE(std::isfinite(
            report.coupling_row_solver_iteration_normal_delta_impulses[slot]));
        EXPECT_TRUE(std::isfinite(
            report.coupling_row_solver_iteration_tangent_delta_impulses[slot]));
        EXPECT_TRUE(std::isfinite(
            report.coupling_row_solver_iteration_max_residuals[slot]));
        EXPECT_GE(report.coupling_row_solver_iteration_max_residuals[slot], 0.0f);
    }

    auto capped_device_particles = runtime::gpu::UploadCudaParticleWorld(particles);
    options.step_count = 3u;
    const auto capped_report = runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
        capped_device_particles, device_world, options);

    EXPECT_GT(capped_report.coupling_row_solver_iteration_count,
              runtime::gpu::kCudaParticleRowSolverDiagnosticSlots);
    EXPECT_EQ(capped_report.coupling_row_solver_diagnostic_slot_count,
              runtime::gpu::kCudaParticleRowSolverDiagnosticSlots);
}

TEST(CudaParticleWorld, SolvesDeviceWorldCouplingFrictionRowsOnCuda) {
    scene::SceneIR scene;

    scene::RigidBodyRecord box_body;
    box_body.name = "friction_row_robot_link";
    box_body.mass = 2.0f;
    box_body.inertia = {0.75f, 0.75f, 0.75f};
    box_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    scene.AddRigidBody(std::move(box_body));

    scene::CollisionShapeRecord box_shape;
    box_shape.body_id = 0u;
    box_shape.type = scene::ShapeType::Box;
    box_shape.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(box_shape));

    runtime::gpu::CudaParticleSet particles;
    particles.positions = {{0.0f, 0.34f, 0.0f}};
    particles.velocities = {{1.0f, -2.0f, 0.0f}};
    particles.inv_masses = {1.0f};
    particles.radii = {0.1f};
    particles.phases = {13u};

    runtime::BuiltWorld no_friction_world;
    auto no_friction_device_world = UploadScene(scene, no_friction_world);
    auto no_friction_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions no_friction_options;
    no_friction_options.gravity = math::Vec3::Zero();
    no_friction_options.dt = 1.0f / 120.0f;
    no_friction_options.step_count = 1u;
    no_friction_options.friction = 0.0f;
    no_friction_options.restitution = 0.0f;
    no_friction_options.accumulate_rigid_impulses = true;
    no_friction_options.solve_coupling_rows_on_cuda = true;

    const auto no_friction_report = runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
        no_friction_particles, no_friction_device_world, no_friction_options);
    const auto no_friction_state = no_friction_particles.DownloadState();

    runtime::BuiltWorld friction_world;
    auto friction_device_world = UploadScene(scene, friction_world);
    auto friction_particles = runtime::gpu::UploadCudaParticleWorld(particles);
    auto friction_options = no_friction_options;
    friction_options.friction = 0.75f;

    const auto friction_report = runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
        friction_particles, friction_device_world, friction_options);
    const auto friction_state = friction_particles.DownloadState();
    const auto friction_rows = friction_particles.DownloadCouplingRows();

    ASSERT_EQ(no_friction_state.velocities.size(), 1u);
    ASSERT_EQ(friction_state.velocities.size(), 1u);
    ASSERT_GE(friction_rows.rows.size(), 1u);
    const auto& row = friction_rows.rows[0];

    EXPECT_EQ(no_friction_report.coupling_row_solver_launch_count, 1u);
    EXPECT_EQ(friction_report.coupling_row_solver_launch_count, 1u);
    EXPECT_EQ(friction_report.coupling_row_solver_friction_impulse_count, 1u);
    EXPECT_GT(friction_report.coupling_row_solver_friction_impulse_magnitude, 0.0f);
    EXPECT_TRUE(row.active);
    EXPECT_GT(row.normal_impulse, 0.0f);
    EXPECT_FLOAT_EQ(row.friction, friction_options.friction);
    EXPECT_NEAR(row.tangent0.Length(), 1.0f, 1.0e-4f);
    EXPECT_NEAR(row.tangent1.Length(), 1.0f, 1.0e-4f);
    EXPECT_GT(std::abs(row.tangent_impulse_0), 0.0f);
    EXPECT_LE(std::abs(row.tangent_impulse_0),
              friction_options.friction * row.normal_impulse + 1.0e-5f);
    EXPECT_LE(std::abs(row.tangent_impulse_1),
              friction_options.friction * row.normal_impulse + 1.0e-5f);
    EXPECT_LT(std::abs(friction_state.velocities[0].x),
              std::abs(no_friction_state.velocities[0].x));
}

TEST(CudaParticleWorld, WarmStartsDeviceWorldCouplingFrictionRowsOnCuda) {
    scene::SceneIR scene;

    scene::RigidBodyRecord box_body;
    box_body.name = "friction_warm_start_robot_link";
    box_body.mass = 2.0f;
    box_body.inertia = {0.75f, 0.75f, 0.75f};
    box_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    scene.AddRigidBody(std::move(box_body));

    scene::CollisionShapeRecord box_shape;
    box_shape.body_id = 0u;
    box_shape.type = scene::ShapeType::Box;
    box_shape.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(box_shape));

    runtime::BuiltWorld world;
    auto device_world = UploadScene(scene, world);

    runtime::gpu::CudaParticleSet particles;
    particles.positions = {{0.0f, 0.34f, 0.0f}};
    particles.velocities = {{1.0f, 0.0f, 0.0f}};
    particles.inv_masses = {1.0f};
    particles.radii = {0.1f};
    particles.phases = {14u};
    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions options;
    options.gravity = {0.0f, -240.0f, 0.0f};
    options.dt = 1.0f / 120.0f;
    options.step_count = 1u;
    options.friction = 0.75f;
    options.restitution = 0.0f;
    options.accumulate_rigid_impulses = true;
    options.enable_coupling_warm_start = true;
    options.solve_coupling_rows_on_cuda = true;

    const auto first_report = runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
        device_particles, device_world, options);
    const auto rows_after_first = device_particles.DownloadCouplingRows();

    ASSERT_GE(rows_after_first.rows.size(), 1u);
    const auto& first_row = rows_after_first.rows[0];
    EXPECT_EQ(first_report.coupling_warm_start_count, 0u);
    EXPECT_EQ(first_report.coupling_row_solver_friction_impulse_count, 1u);
    EXPECT_GT(std::abs(first_row.tangent_impulse_0), 0.0f);

    const auto second_report = runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
        device_particles, device_world, options);
    const auto rows_after_second = device_particles.DownloadCouplingRows();

    ASSERT_GE(rows_after_second.rows.size(), 1u);
    const auto& second_row = rows_after_second.rows[0];
    EXPECT_EQ(second_report.coupling_warm_start_count, 1u);
    EXPECT_EQ(second_report.coupling_tangent_warm_start_count, 1u);
    EXPECT_GT(second_report.coupling_warm_start_impulse_magnitude,
              first_report.coupling_row_solver_impulse_magnitude);
    EXPECT_GT(second_report.coupling_tangent_warm_start_impulse_magnitude,
              0.0f);
    EXPECT_TRUE(second_row.active);
    EXPECT_EQ(second_row.shape_index, first_row.shape_index);
    const float first_tangent_cache_magnitude =
        std::abs(first_row.tangent_impulse_0) + std::abs(first_row.tangent_impulse_1);
    const float second_tangent_impulse_magnitude =
        std::abs(second_row.tangent_impulse_0) + std::abs(second_row.tangent_impulse_1);
    EXPECT_NEAR(second_report.coupling_tangent_warm_start_impulse_magnitude,
                first_tangent_cache_magnitude,
                1.0e-4f);
    EXPECT_GT(second_row.normal_impulse, 0.0f);
    EXPECT_GT(second_tangent_impulse_magnitude, 0.0f);
    EXPECT_LE(second_tangent_impulse_magnitude,
              first_tangent_cache_magnitude + 1.0e-5f);
    EXPECT_LE(std::abs(second_row.tangent_impulse_0),
              options.friction * second_row.normal_impulse + 1.0e-5f);
    EXPECT_LE(std::abs(second_row.tangent_impulse_1),
              options.friction * second_row.normal_impulse + 1.0e-5f);
}

TEST(CudaParticleWorld, ReportsNoRowSolverDiagnosticsWhenCudaRowSolverDisabled) {
    scene::SceneIR scene;

    scene::RigidBodyRecord box_body;
    box_body.name = "direct_coupling_transition_link";
    box_body.mass = 2.0f;
    box_body.inertia = {0.75f, 0.75f, 0.75f};
    box_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    scene.AddRigidBody(std::move(box_body));

    scene::CollisionShapeRecord box_shape;
    box_shape.body_id = 0u;
    box_shape.type = scene::ShapeType::Box;
    box_shape.half_extents = {0.25f, 0.25f, 0.25f};
    scene.AddCollisionShape(std::move(box_shape));

    runtime::BuiltWorld warmup_world;
    auto warmup_device_world = UploadScene(scene, warmup_world);

    runtime::gpu::CudaParticleSet warmup_particles;
    warmup_particles.positions = {{0.2f, 0.34f, 0.0f}};
    warmup_particles.velocities = {{0.0f, -2.0f, 0.0f}};
    warmup_particles.inv_masses = {1.0f};
    warmup_particles.radii = {0.1f};
    warmup_particles.phases = {4u};
    auto warmup_device_particles =
        runtime::gpu::UploadCudaParticleWorld(warmup_particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions warmup_options;
    warmup_options.gravity = math::Vec3::Zero();
    warmup_options.dt = 1.0f / 120.0f;
    warmup_options.step_count = 1u;
    warmup_options.friction = 0.0f;
    warmup_options.restitution = 0.0f;
    warmup_options.accumulate_rigid_impulses = true;
    warmup_options.solve_coupling_rows_on_cuda = true;

    const auto warmup_report = runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
        warmup_device_particles, warmup_device_world, warmup_options);
    EXPECT_EQ(warmup_report.coupling_row_solver_launch_count, 1u);
    EXPECT_EQ(warmup_report.coupling_row_solver_impulse_count, 1u);

    runtime::BuiltWorld direct_world;
    auto direct_device_world = UploadScene(scene, direct_world);
    runtime::gpu::CudaParticleSet direct_particles = warmup_particles;
    auto direct_device_particles =
        runtime::gpu::UploadCudaParticleWorld(direct_particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions direct_options = warmup_options;
    direct_options.solve_coupling_rows_on_cuda = false;

    const auto direct_report = runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
        direct_device_particles, direct_device_world, direct_options);

    EXPECT_EQ(direct_report.contact_count, 1u);
    EXPECT_EQ(direct_report.rigid_impulse_count, 1u);
    EXPECT_EQ(direct_report.kernel_launch_count, 2u);
    EXPECT_EQ(direct_report.coupling_row_solver_launch_count, 0u);
    EXPECT_EQ(direct_report.coupling_row_solver_impulse_count, 0u);
    EXPECT_FLOAT_EQ(direct_report.coupling_row_solver_impulse_magnitude, 0.0f);
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
