// ---------------------------------------------------------------------------
// CUDA particle coupling demo.
// ---------------------------------------------------------------------------

#include "runtime/gpu/cuda_particle_world.hpp"
#include "runtime/gpu/device_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <cstdint>
#include <cmath>
#include <iostream>
#include <utility>

using namespace nuka;

int main() {
    runtime::gpu::CudaParticleSet particles;
    constexpr uint32_t kParticleCount = 512u;
    particles.positions.reserve(kParticleCount);
    particles.velocities.reserve(kParticleCount);
    particles.inv_masses.reserve(kParticleCount);
    particles.radii.reserve(kParticleCount);
    particles.phases.reserve(kParticleCount);

    for (uint32_t index = 0; index < kParticleCount; ++index) {
        const float x = static_cast<float>(index % 32u) * 0.035f - 0.55f;
        const float y = 0.25f + static_cast<float>(index / 32u) * 0.02f;
        particles.positions.push_back({x, y, 0.0f});
        particles.velocities.push_back({0.05f, -0.5f, 0.0f});
        particles.inv_masses.push_back(1.0f);
        particles.radii.push_back(0.0125f);
        particles.phases.push_back(0u);
    }

    auto device_particles = runtime::gpu::UploadCudaParticleWorld(particles);

    runtime::gpu::CudaParticleStepOptions options;
    options.dt = 1.0f / 240.0f;
    options.step_count = 120u;
    options.plane.enabled = true;
    options.plane.normal = math::Vec3::UnitY();
    options.plane.offset = 0.0f;
    options.plane.friction = 0.08f;
    options.plane.restitution = 0.0f;
    options.sphere.enabled = true;
    options.sphere.center = {0.0f, 0.18f, 0.0f};
    options.sphere.radius = 0.14f;
    options.sphere.friction = 0.05f;
    options.sphere.restitution = 0.0f;

    const auto report = runtime::gpu::StepCudaParticleWorld(device_particles, options);
    const auto state = device_particles.DownloadState();

    std::cout << "cuda_particle_demo\n";
    std::cout << "particles=" << report.particle_count
              << " steps=" << report.simulated_step_count
              << " contacts=" << report.contact_count
              << " max_penetration=" << report.max_penetration
              << " residual=" << report.max_penetration_after_solve
              << " max_speed=" << report.max_speed
              << " kinetic_energy=" << report.kinetic_energy
              << " kernel_launches=" << report.kernel_launch_count
              << "\n";

    if (!state.positions.empty()) {
        std::cout << "first_particle_position=("
                  << state.positions.front().x << ", "
                  << state.positions.front().y << ", "
                  << state.positions.front().z << ")\n";
    }

    scene::SceneIR coupled_scene;
    scene::RigidBodyRecord sphere_body;
    sphere_body.name = "cuda_coupled_robot_link";
    sphere_body.mass = 4.0f;
    sphere_body.inertia = {1.0f, 1.0f, 1.0f};
    sphere_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    const auto body_id = coupled_scene.AddRigidBody(std::move(sphere_body));

    scene::CollisionShapeRecord sphere_shape;
    sphere_shape.body_id = body_id;
    sphere_shape.type = scene::ShapeType::Sphere;
    sphere_shape.local_transform.position = {0.0f, 0.18f, 0.0f};
    sphere_shape.radius = 0.18f;
    coupled_scene.AddCollisionShape(std::move(sphere_shape));

    scene::RigidBodyRecord box_body;
    box_body.name = "cuda_coupled_box_link";
    box_body.mass = 4.0f;
    box_body.inertia = {1.0f, 1.0f, 1.0f};
    box_body.local_transform.position = {0.45f, 0.0f, 0.0f};
    const auto box_body_id = coupled_scene.AddRigidBody(std::move(box_body));

    scene::CollisionShapeRecord box_shape;
    box_shape.body_id = box_body_id;
    box_shape.type = scene::ShapeType::Box;
    box_shape.half_extents = {0.15f, 0.18f, 0.15f};
    coupled_scene.AddCollisionShape(std::move(box_shape));

    scene::RigidBodyRecord capsule_body;
    capsule_body.name = "cuda_coupled_capsule_link";
    capsule_body.mass = 4.0f;
    capsule_body.inertia = {1.0f, 1.0f, 1.0f};
    capsule_body.local_transform.position = {0.90f, 0.0f, 0.0f};
    const auto capsule_body_id = coupled_scene.AddRigidBody(std::move(capsule_body));

    scene::CollisionShapeRecord capsule_shape;
    capsule_shape.body_id = capsule_body_id;
    capsule_shape.type = scene::ShapeType::Capsule;
    capsule_shape.radius = 0.16f;
    capsule_shape.half_height = 0.32f;
    coupled_scene.AddCollisionShape(std::move(capsule_shape));

    auto world = runtime::BuildWorld(scene::CookScene(coupled_scene));
    auto device_world = runtime::gpu::UploadDeviceWorld(world.template_view);
    runtime::gpu::UploadDeviceState(device_world, world.instance);

    runtime::gpu::CudaParticleSet coupled_particles;
    coupled_particles.positions = {{0.15f, 0.18f, 0.0f}, {0.0f, 0.18f, 0.0f}};
    coupled_particles.velocities = {{-0.4f, 0.0f, 0.0f}, {-0.1f, 0.0f, 0.0f}};
    coupled_particles.inv_masses = {1.0f, 1.0f};
    coupled_particles.radii = {0.02f, 0.02f};
    coupled_particles.phases = {1u, 1u};
    auto device_coupled_particles =
        runtime::gpu::UploadCudaParticleWorld(coupled_particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions coupling_options;
    coupling_options.gravity = math::Vec3::Zero();
    coupling_options.dt = 1.0f / 240.0f;
    coupling_options.step_count = 1u;
    coupling_options.friction = 0.0f;
    coupling_options.restitution = 0.0f;
    coupling_options.accumulate_rigid_impulses = true;
    coupling_options.enable_coupling_warm_start = true;

    runtime::gpu::CudaParticleSet friction_particles;
    friction_particles.positions = {{0.53f, 0.195f, 0.0f}};
    friction_particles.velocities = {{0.30f, -0.30f, 0.0f}};
    friction_particles.inv_masses = {1.0f};
    friction_particles.radii = {0.02f};
    friction_particles.phases = {5u};
    auto no_friction_particles =
        runtime::gpu::UploadCudaParticleWorld(friction_particles);
    auto friction_row_particles =
        runtime::gpu::UploadCudaParticleWorld(friction_particles);

    auto friction_reference_world = runtime::BuildWorld(scene::CookScene(coupled_scene));
    auto no_friction_device_world =
        runtime::gpu::UploadDeviceWorld(friction_reference_world.template_view);
    runtime::gpu::UploadDeviceState(no_friction_device_world, friction_reference_world.instance);

    auto friction_world = runtime::BuildWorld(scene::CookScene(coupled_scene));
    auto friction_device_world =
        runtime::gpu::UploadDeviceWorld(friction_world.template_view);
    runtime::gpu::UploadDeviceState(friction_device_world, friction_world.instance);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions no_friction_options =
        coupling_options;
    no_friction_options.friction = 0.0f;
    no_friction_options.enable_coupling_warm_start = false;
    runtime::gpu::CudaParticleDeviceWorldCouplingOptions friction_row_options =
        no_friction_options;
    friction_row_options.friction = 0.75f;

    const auto no_friction_row_report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
            no_friction_particles,
            no_friction_device_world,
            no_friction_options);
    const auto friction_row_report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
            friction_row_particles,
            friction_device_world,
            friction_row_options);
    const auto no_friction_state = no_friction_particles.DownloadState();
    const auto friction_row_state = friction_row_particles.DownloadState();
    const auto friction_rows = friction_row_particles.DownloadCouplingRows();
    const auto& friction_row = friction_rows.rows.front();
    const float friction_limit =
        friction_row_options.friction * friction_row.normal_impulse;

    std::cout << "deviceworld_box_friction_rows contacts="
              << friction_row_report.contact_count
              << " row_solver_launches="
              << friction_row_report.coupling_row_solver_launch_count
              << " row_solver_normal_impulses="
              << friction_row_report.coupling_row_solver_impulse_count
              << " row_solver_friction_impulses="
              << friction_row_report.coupling_row_solver_friction_impulse_count
              << " row_solver_friction_impulse_magnitude="
              << friction_row_report.coupling_row_solver_friction_impulse_magnitude
              << " tangent0_impulse=" << friction_row.tangent_impulse_0
              << " tangent1_impulse=" << friction_row.tangent_impulse_1
              << " friction_limit=" << friction_limit
              << " no_friction_vx=" << no_friction_state.velocities.front().x
              << " friction_vx=" << friction_row_state.velocities.front().x
              << "\n";

    const auto coupling_first_report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
        device_coupled_particles,
        device_world,
        coupling_options);
    const auto coupling_report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
            device_coupled_particles,
            device_world,
            coupling_options);
    const auto coupling_state = device_coupled_particles.DownloadCouplingState();
    const auto rigid_state = device_world.DownloadState();

    std::cout << "deviceworld_coupling contacts=" << coupling_report.contact_count
              << " rigid_impulses=" << coupling_report.rigid_impulse_count
              << " active_slots=" << coupling_report.coupling_active_slot_count
              << " warm_starts=" << coupling_report.coupling_warm_start_count
              << " rigid_impulse_magnitude=" << coupling_report.rigid_impulse_magnitude
              << " rigid_angular_impulse_magnitude="
              << coupling_report.rigid_angular_impulse_magnitude
              << " coupling_force=" << coupling_report.coupling_force_magnitude
              << " coupling_torque=" << coupling_report.coupling_torque_magnitude
              << " warm_start_impulse_magnitude="
              << coupling_report.coupling_warm_start_impulse_magnitude
              << " row_solver_launches="
              << coupling_report.coupling_row_solver_launch_count
              << " row_solver_impulses="
              << coupling_report.coupling_row_solver_impulse_count
              << " row_solver_impulse_magnitude="
              << coupling_report.coupling_row_solver_impulse_magnitude
              << " row_solver_total_impulses="
              << (coupling_first_report.coupling_row_solver_impulse_count +
                  coupling_report.coupling_row_solver_impulse_count)
              << " row_solver_total_impulse_magnitude="
              << (coupling_first_report.coupling_row_solver_impulse_magnitude +
                  coupling_report.coupling_row_solver_impulse_magnitude)
              << " max_coupling_normal_impulse="
              << coupling_report.max_coupling_normal_impulse
              << " slots_per_particle=" << coupling_state.slot_count_per_particle
              << " residual=" << coupling_report.max_penetration_after_solve
              << " cached_impulse=" << coupling_state.normal_impulses.front()
              << " body_velocity=("
              << rigid_state.linear_velocities[body_id].x << ", "
              << rigid_state.linear_velocities[body_id].y << ", "
              << rigid_state.linear_velocities[body_id].z << ")"
              << " body_angular_velocity=("
              << rigid_state.angular_velocities[body_id].x << ", "
              << rigid_state.angular_velocities[body_id].y << ", "
              << rigid_state.angular_velocities[body_id].z << ")\n";

    runtime::gpu::CudaParticleSet box_particles;
    box_particles.positions = {{0.53f, 0.195f, 0.0f}};
    box_particles.velocities = {{0.0f, -0.4f, 0.0f}};
    box_particles.inv_masses = {1.0f};
    box_particles.radii = {0.02f};
    box_particles.phases = {2u};
    auto device_box_particles = runtime::gpu::UploadCudaParticleWorld(box_particles);

    const auto box_first_report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
        device_box_particles,
        device_world,
        coupling_options);
    const auto box_report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
            device_box_particles,
            device_world,
            coupling_options);
    const auto box_coupling_state = device_box_particles.DownloadCouplingState();
    const auto box_rigid_state = device_world.DownloadState();

    std::cout << "deviceworld_box_coupling contacts=" << box_report.contact_count
              << " rigid_impulses=" << box_report.rigid_impulse_count
              << " active_slots=" << box_report.coupling_active_slot_count
              << " warm_starts=" << box_report.coupling_warm_start_count
              << " rigid_impulse_magnitude=" << box_report.rigid_impulse_magnitude
              << " rigid_angular_impulse_magnitude="
              << box_report.rigid_angular_impulse_magnitude
              << " coupling_force=" << box_report.coupling_force_magnitude
              << " coupling_torque=" << box_report.coupling_torque_magnitude
              << " warm_start_impulse_magnitude="
              << box_report.coupling_warm_start_impulse_magnitude
              << " row_solver_launches="
              << box_report.coupling_row_solver_launch_count
              << " row_solver_impulses="
              << box_report.coupling_row_solver_impulse_count
              << " row_solver_impulse_magnitude="
              << box_report.coupling_row_solver_impulse_magnitude
              << " row_solver_total_impulses="
              << (box_first_report.coupling_row_solver_impulse_count +
                  box_report.coupling_row_solver_impulse_count)
              << " row_solver_total_impulse_magnitude="
              << (box_first_report.coupling_row_solver_impulse_magnitude +
                  box_report.coupling_row_solver_impulse_magnitude)
              << " max_coupling_normal_impulse="
              << box_report.max_coupling_normal_impulse
              << " slots_per_particle=" << box_coupling_state.slot_count_per_particle
              << " residual=" << box_report.max_penetration_after_solve
              << " cached_impulse=" << box_coupling_state.normal_impulses.front()
              << " body_velocity=("
              << box_rigid_state.linear_velocities[box_body_id].x << ", "
              << box_rigid_state.linear_velocities[box_body_id].y << ", "
              << box_rigid_state.linear_velocities[box_body_id].z << ")"
              << " body_angular_velocity=("
              << box_rigid_state.angular_velocities[box_body_id].x << ", "
              << box_rigid_state.angular_velocities[box_body_id].y << ", "
              << box_rigid_state.angular_velocities[box_body_id].z << ")\n";

    runtime::gpu::CudaParticleSet capsule_particles;
    capsule_particles.positions = {{1.02f, 0.22f, 0.0f}};
    capsule_particles.velocities = {{-0.4f, 0.0f, 0.0f}};
    capsule_particles.inv_masses = {1.0f};
    capsule_particles.radii = {0.02f};
    capsule_particles.phases = {3u};
    auto device_capsule_particles =
        runtime::gpu::UploadCudaParticleWorld(capsule_particles);

    const auto capsule_first_report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
        device_capsule_particles,
        device_world,
        coupling_options);
    const auto capsule_report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
            device_capsule_particles,
            device_world,
            coupling_options);
    const auto capsule_coupling_state = device_capsule_particles.DownloadCouplingState();
    const auto capsule_rigid_state = device_world.DownloadState();

    std::cout << "deviceworld_capsule_coupling contacts=" << capsule_report.contact_count
              << " rigid_impulses=" << capsule_report.rigid_impulse_count
              << " active_slots=" << capsule_report.coupling_active_slot_count
              << " warm_starts=" << capsule_report.coupling_warm_start_count
              << " rigid_impulse_magnitude=" << capsule_report.rigid_impulse_magnitude
              << " rigid_angular_impulse_magnitude="
              << capsule_report.rigid_angular_impulse_magnitude
              << " coupling_force=" << capsule_report.coupling_force_magnitude
              << " coupling_torque=" << capsule_report.coupling_torque_magnitude
              << " warm_start_impulse_magnitude="
              << capsule_report.coupling_warm_start_impulse_magnitude
              << " row_solver_launches="
              << capsule_report.coupling_row_solver_launch_count
              << " row_solver_impulses="
              << capsule_report.coupling_row_solver_impulse_count
              << " row_solver_impulse_magnitude="
              << capsule_report.coupling_row_solver_impulse_magnitude
              << " row_solver_total_impulses="
              << (capsule_first_report.coupling_row_solver_impulse_count +
                  capsule_report.coupling_row_solver_impulse_count)
              << " row_solver_total_impulse_magnitude="
              << (capsule_first_report.coupling_row_solver_impulse_magnitude +
                  capsule_report.coupling_row_solver_impulse_magnitude)
              << " max_coupling_normal_impulse="
              << capsule_report.max_coupling_normal_impulse
              << " slots_per_particle=" << capsule_coupling_state.slot_count_per_particle
              << " residual=" << capsule_report.max_penetration_after_solve
              << " cached_impulse=" << capsule_coupling_state.normal_impulses.front()
              << " body_velocity=("
              << capsule_rigid_state.linear_velocities[capsule_body_id].x << ", "
              << capsule_rigid_state.linear_velocities[capsule_body_id].y << ", "
              << capsule_rigid_state.linear_velocities[capsule_body_id].z << ")"
              << " body_angular_velocity=("
              << capsule_rigid_state.angular_velocities[capsule_body_id].x << ", "
              << capsule_rigid_state.angular_velocities[capsule_body_id].y << ", "
              << capsule_rigid_state.angular_velocities[capsule_body_id].z << ")\n";

    scene::SceneIR corner_scene;
    scene::RigidBodyRecord floor_body;
    floor_body.name = "cuda_corner_floor_link";
    floor_body.mass = 4.0f;
    floor_body.inertia = {1.0f, 1.0f, 1.0f};
    floor_body.local_transform.position = {0.0f, 0.0f, 0.0f};
    const auto corner_floor_id = corner_scene.AddRigidBody(std::move(floor_body));

    scene::CollisionShapeRecord floor_box;
    floor_box.body_id = corner_floor_id;
    floor_box.type = scene::ShapeType::Box;
    floor_box.half_extents = {0.25f, 0.25f, 0.25f};
    corner_scene.AddCollisionShape(std::move(floor_box));

    scene::RigidBodyRecord side_body;
    side_body.name = "cuda_corner_side_link";
    side_body.mass = 4.0f;
    side_body.inertia = {1.0f, 1.0f, 1.0f};
    side_body.local_transform.position = {0.36f, 0.10f, 0.0f};
    const auto corner_side_id = corner_scene.AddRigidBody(std::move(side_body));

    scene::CollisionShapeRecord side_box;
    side_box.body_id = corner_side_id;
    side_box.type = scene::ShapeType::Box;
    side_box.half_extents = {0.25f, 0.25f, 0.25f};
    corner_scene.AddCollisionShape(std::move(side_box));

    auto corner_world = runtime::BuildWorld(scene::CookScene(corner_scene));
    auto corner_device_world = runtime::gpu::UploadDeviceWorld(corner_world.template_view);
    runtime::gpu::UploadDeviceState(corner_device_world, corner_world.instance);

    runtime::gpu::CudaParticleSet corner_particles;
    corner_particles.positions = {{0.05f, 0.34f, 0.0f}};
    corner_particles.velocities = {{0.05f, -0.05f, 0.0f}};
    corner_particles.inv_masses = {1.0f};
    corner_particles.radii = {0.1f};
    corner_particles.phases = {4u};
    auto device_corner_particles = runtime::gpu::UploadCudaParticleWorld(corner_particles);

    runtime::gpu::CudaParticleDeviceWorldCouplingOptions corner_options = coupling_options;
    corner_options.gravity = {0.0f, -9.81f, 0.0f};

    const auto corner_first_report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
        device_corner_particles,
        corner_device_world,
        corner_options);
    const auto corner_report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
            device_corner_particles,
            corner_device_world,
            corner_options);
    const auto corner_coupling_state =
        device_corner_particles.DownloadCouplingState();
    const auto corner_rows = device_corner_particles.DownloadCouplingRows();
    const auto corner_rigid_state = corner_device_world.DownloadState();

    std::cout << "deviceworld_corner_multislot contacts=" << corner_report.contact_count
              << " active_slots=" << corner_report.coupling_active_slot_count
              << " warm_starts=" << corner_report.coupling_warm_start_count
              << " coupling_force=" << corner_report.coupling_force_magnitude
              << " coupling_torque=" << corner_report.coupling_torque_magnitude
              << " row_solver_launches="
              << corner_report.coupling_row_solver_launch_count
              << " row_solver_impulses="
              << corner_report.coupling_row_solver_impulse_count
              << " row_solver_impulse_magnitude="
              << corner_report.coupling_row_solver_impulse_magnitude
              << " row_solver_total_impulses="
              << (corner_first_report.coupling_row_solver_impulse_count +
                  corner_report.coupling_row_solver_impulse_count)
              << " row_solver_total_impulse_magnitude="
              << (corner_first_report.coupling_row_solver_impulse_magnitude +
                  corner_report.coupling_row_solver_impulse_magnitude)
              << " slots_per_particle=" << corner_coupling_state.slot_count_per_particle
              << " row0_active=" << corner_rows.rows[0].active
              << " row1_active=" << corner_rows.rows[1].active
              << " row0_effective_mass=" << corner_rows.rows[0].effective_mass
              << " row1_effective_mass=" << corner_rows.rows[1].effective_mass
              << " row0_position_error=" << corner_rows.rows[0].position_error
              << " row1_position_error=" << corner_rows.rows[1].position_error
              << " row0_normal_impulse=" << corner_rows.rows[0].normal_impulse
              << " row1_normal_impulse=" << corner_rows.rows[1].normal_impulse
              << " cached_slot0_impulse=" << corner_coupling_state.normal_impulses[0]
              << " cached_slot1_impulse=" << corner_coupling_state.normal_impulses[1]
              << " cached_slot0_shape=" << corner_coupling_state.shape_indices[0]
              << " cached_slot1_shape=" << corner_coupling_state.shape_indices[1]
              << " floor_velocity=("
              << corner_rigid_state.linear_velocities[corner_floor_id].x << ", "
              << corner_rigid_state.linear_velocities[corner_floor_id].y << ", "
              << corner_rigid_state.linear_velocities[corner_floor_id].z << ")"
              << " side_velocity=("
              << corner_rigid_state.linear_velocities[corner_side_id].x << ", "
              << corner_rigid_state.linear_velocities[corner_side_id].y << ", "
              << corner_rigid_state.linear_velocities[corner_side_id].z << ")\n";

    const auto coupling_rigid_impulses =
        coupling_first_report.rigid_impulse_count + coupling_report.rigid_impulse_count;
    const auto coupling_row_impulses =
        coupling_first_report.coupling_row_solver_impulse_count +
        coupling_report.coupling_row_solver_impulse_count;
    const auto coupling_row_impulse_magnitude =
        coupling_first_report.coupling_row_solver_impulse_magnitude +
        coupling_report.coupling_row_solver_impulse_magnitude;
    const auto coupling_angular_impulse_magnitude =
        coupling_first_report.rigid_angular_impulse_magnitude +
        coupling_report.rigid_angular_impulse_magnitude;
    const bool friction_row_limit_ok =
        std::fabs(friction_row.tangent_impulse_0) <= friction_limit + 1.0e-5f &&
        std::fabs(friction_row.tangent_impulse_1) <= friction_limit + 1.0e-5f;
    const bool friction_row_reduced_tangent_speed =
        std::fabs(friction_row_state.velocities.front().x) <
        std::fabs(no_friction_state.velocities.front().x);

    const auto box_rigid_impulses =
        box_first_report.rigid_impulse_count + box_report.rigid_impulse_count;
    const auto box_row_impulses =
        box_first_report.coupling_row_solver_impulse_count +
        box_report.coupling_row_solver_impulse_count;
    const auto box_row_impulse_magnitude =
        box_first_report.coupling_row_solver_impulse_magnitude +
        box_report.coupling_row_solver_impulse_magnitude;
    const auto box_angular_impulse_magnitude =
        box_first_report.rigid_angular_impulse_magnitude +
        box_report.rigid_angular_impulse_magnitude;

    const auto capsule_rigid_impulses =
        capsule_first_report.rigid_impulse_count + capsule_report.rigid_impulse_count;
    const auto capsule_row_impulses =
        capsule_first_report.coupling_row_solver_impulse_count +
        capsule_report.coupling_row_solver_impulse_count;
    const auto capsule_row_impulse_magnitude =
        capsule_first_report.coupling_row_solver_impulse_magnitude +
        capsule_report.coupling_row_solver_impulse_magnitude;
    const auto capsule_angular_impulse_magnitude =
        capsule_first_report.rigid_angular_impulse_magnitude +
        capsule_report.rigid_angular_impulse_magnitude;

    const auto corner_row_impulses =
        corner_first_report.coupling_row_solver_impulse_count +
        corner_report.coupling_row_solver_impulse_count;
    const auto corner_row_impulse_magnitude =
        corner_first_report.coupling_row_solver_impulse_magnitude +
        corner_report.coupling_row_solver_impulse_magnitude;

    return report.max_penetration_after_solve <= 1.0e-4f &&
                   coupling_report.contact_count > 0u &&
                   coupling_rigid_impulses > 0u &&
                   coupling_report.coupling_active_slot_count > 0u &&
                   coupling_report.coupling_warm_start_count > 0u &&
                   coupling_angular_impulse_magnitude > 0.0f &&
                   coupling_report.coupling_row_solver_launch_count > 0u &&
                   coupling_row_impulses > 0u &&
                   coupling_row_impulse_magnitude > 0.0f &&
                   coupling_report.coupling_force_magnitude > 0.0f &&
                   coupling_report.coupling_torque_magnitude > 0.0f &&
                   friction_row_report.contact_count > 0u &&
                   friction_row_report.coupling_row_solver_launch_count > 0u &&
                   friction_row_report.coupling_row_solver_impulse_count > 0u &&
                   friction_row_report.coupling_row_solver_friction_impulse_count > 0u &&
                   friction_row_report.coupling_row_solver_friction_impulse_magnitude > 0.0f &&
                   friction_row.active &&
                   friction_row.normal_impulse > 0.0f &&
                   friction_row_limit_ok &&
                   friction_row_reduced_tangent_speed &&
                   box_report.contact_count > 0u &&
                   box_rigid_impulses > 0u &&
                   box_report.coupling_active_slot_count > 0u &&
                   box_report.coupling_warm_start_count > 0u &&
                   box_report.max_penetration_after_solve <= 1.0e-4f &&
                   box_angular_impulse_magnitude > 0.0f &&
                   box_report.coupling_row_solver_launch_count > 0u &&
                   box_row_impulses > 0u &&
                   box_row_impulse_magnitude > 0.0f &&
                   box_report.coupling_force_magnitude > 0.0f &&
                   box_report.coupling_torque_magnitude > 0.0f &&
                   capsule_report.contact_count > 0u &&
                   capsule_rigid_impulses > 0u &&
                   capsule_report.coupling_active_slot_count > 0u &&
                   capsule_report.coupling_warm_start_count > 0u &&
                   capsule_report.max_penetration_after_solve <= 1.0e-4f &&
                   capsule_angular_impulse_magnitude > 0.0f &&
                   capsule_report.coupling_row_solver_launch_count > 0u &&
                   capsule_row_impulses > 0u &&
                   capsule_row_impulse_magnitude > 0.0f &&
                   capsule_report.coupling_force_magnitude > 0.0f &&
                   capsule_report.coupling_torque_magnitude > 0.0f &&
                   corner_report.contact_count >= 2u &&
                   corner_report.coupling_active_slot_count >= 2u &&
                   corner_report.coupling_warm_start_count >= 2u &&
                   corner_report.coupling_force_magnitude > 0.0f &&
                   corner_report.coupling_torque_magnitude > 0.0f &&
                   corner_report.coupling_row_solver_launch_count > 0u &&
                   corner_row_impulses >= 2u &&
                   corner_row_impulse_magnitude > 0.0f &&
                   corner_coupling_state.slot_count_per_particle ==
                       runtime::gpu::kCudaParticleCouplingSlotsPerParticle &&
                   corner_rows.rows.size() >=
                       runtime::gpu::kCudaParticleCouplingSlotsPerParticle &&
                   corner_rows.rows[0].active &&
                   corner_rows.rows[1].active &&
                   corner_rows.rows[0].effective_mass > 0.0f &&
                   corner_rows.rows[1].effective_mass > 0.0f &&
                   corner_rows.rows[0].position_error > 0.0f &&
                   corner_rows.rows[1].position_error > 0.0f &&
                   corner_rows.rows[0].normal_impulse > 0.0f &&
                   corner_rows.rows[1].normal_impulse > 0.0f &&
                   corner_coupling_state.normal_impulses[0] > 0.0f &&
                   corner_coupling_state.normal_impulses[1] > 0.0f &&
                   corner_coupling_state.shape_indices[0] == 0u &&
                   corner_coupling_state.shape_indices[1] == 1u
               ? 0
               : 1;
}
