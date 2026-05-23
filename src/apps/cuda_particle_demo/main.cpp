// ---------------------------------------------------------------------------
// CUDA particle coupling demo.
// ---------------------------------------------------------------------------

#include "runtime/gpu/cuda_particle_world.hpp"

#include <cstdint>
#include <iostream>

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

    return report.max_penetration_after_solve <= 1.0e-4f ? 0 : 1;
}
