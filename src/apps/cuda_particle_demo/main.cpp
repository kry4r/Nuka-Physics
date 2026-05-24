// ---------------------------------------------------------------------------
// CUDA particle coupling demo.
// ---------------------------------------------------------------------------

#include "runtime/gpu/cuda_particle_world.hpp"
#include "runtime/gpu/device_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"

#include <cstdint>
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
    sphere_body.local_transform.position = {0.0f, 0.18f, 0.0f};
    const auto body_id = coupled_scene.AddRigidBody(std::move(sphere_body));

    scene::CollisionShapeRecord sphere_shape;
    sphere_shape.body_id = body_id;
    sphere_shape.type = scene::ShapeType::Sphere;
    sphere_shape.radius = 0.18f;
    coupled_scene.AddCollisionShape(std::move(sphere_shape));

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

    const auto coupling_report =
        runtime::gpu::StepCudaParticlesAgainstDeviceWorld(
            device_coupled_particles,
            device_world,
            coupling_options);
    const auto rigid_state = device_world.DownloadState();

    std::cout << "deviceworld_coupling contacts=" << coupling_report.contact_count
              << " rigid_impulses=" << coupling_report.rigid_impulse_count
              << " rigid_impulse_magnitude=" << coupling_report.rigid_impulse_magnitude
              << " residual=" << coupling_report.max_penetration_after_solve
              << " body_velocity=("
              << rigid_state.linear_velocities[body_id].x << ", "
              << rigid_state.linear_velocities[body_id].y << ", "
              << rigid_state.linear_velocities[body_id].z << ")\n";

    return report.max_penetration_after_solve <= 1.0e-4f &&
                   coupling_report.contact_count > 0u &&
                   coupling_report.rigid_impulse_count > 0u
               ? 0
               : 1;
}
