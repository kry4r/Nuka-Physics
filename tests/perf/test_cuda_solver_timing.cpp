// ---------------------------------------------------------------------------
// Performance test: CUDA contact assembly and solver timing
// ---------------------------------------------------------------------------

#include "collision/gpu/broadphase.cuh"
#include "constraint/gpu/contact_generation.cuh"
#include "runtime/gpu/device_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"
#include "solver/gpu/cuda_constraint_solver.cuh"

#include <gtest/gtest.h>

#include <chrono>
#include <utility>

using namespace nuka;

namespace {

scene::SceneIR BuildContactGrid(int box_count) {
    scene::SceneIR scene;

    scene::RigidBodyRecord ground;
    ground.name = "ground";
    ground.is_static = true;
    const auto ground_id = scene.AddRigidBody(std::move(ground));

    scene::CollisionShapeRecord plane;
    plane.body_id = ground_id;
    plane.type = scene::ShapeType::Plane;
    scene.AddCollisionShape(std::move(plane));

    for (int index = 0; index < box_count; ++index) {
        scene::RigidBodyRecord body;
        body.name = "box";
        body.mass = 1.0f;
        body.inertia = {1.0f, 1.0f, 1.0f};
        body.local_transform.position = {
            static_cast<float>(index % 16) * 2.0f,
            0.45f,
            static_cast<float>(index / 16) * 2.0f
        };
        const auto body_id = scene.AddRigidBody(std::move(body));

        scene::CollisionShapeRecord shape;
        shape.body_id = body_id;
        shape.type = scene::ShapeType::Box;
        shape.half_extents = {0.5f, 0.5f, 0.5f};
        scene.AddCollisionShape(std::move(shape));
    }

    return scene;
}

} // namespace

TEST(CudaSolverTiming, ContactAssemblyAndPgsUnderOneSecond) {
    constexpr int kBoxCount = 128;
    constexpr int kIterationCount = 30;

    auto world = runtime::BuildWorld(scene::CookScene(BuildContactGrid(kBoxCount)));
    for (uint32_t body = 1; body < world.instance.body_count; ++body) {
        world.instance.linear_velocities[body] = {0.0f, -1.0f, 0.0f};
    }

    auto device_world = runtime::gpu::UploadDeviceWorld(world.template_view);
    runtime::gpu::UploadDeviceState(device_world, world.instance);

    solver::gpu::CudaConstraintSolverConfig config;
    config.velocity_iterations = 8u;
    config.position_iterations = 2u;

    solver::gpu::CudaConstraintSolverReport last_report;
    const auto start = std::chrono::high_resolution_clock::now();
    for (int iteration = 0; iteration < kIterationCount; ++iteration) {
        auto broadphase = collision::gpu::BuildCudaBroadphase(device_world);
        auto contacts = constraint::gpu::GenerateCudaContacts(device_world, broadphase);
        auto result = solver::gpu::SolveCudaConstraints(device_world, &contacts, config);
        if (iteration == kIterationCount - 1) {
            last_report = result.DownloadReport();
        }
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(last_report.contact_constraint_count, static_cast<uint32_t>(kBoxCount));
    EXPECT_GE(last_report.constraint_row_count, static_cast<uint32_t>(kBoxCount * 3));
    EXPECT_EQ(last_report.velocity_iterations, config.velocity_iterations);
    EXPECT_EQ(last_report.position_iterations, config.position_iterations);
    EXPECT_LT(ms, 1000) << "CUDA contact solver pipeline took " << ms << " ms";
}
