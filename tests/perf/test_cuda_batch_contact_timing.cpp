// ---------------------------------------------------------------------------
// Performance test: CUDA batched contact solve timing
// ---------------------------------------------------------------------------

#include "runtime/gpu/batched_device_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <utility>
#include <vector>

using namespace nuka;

namespace {

struct PlaneBoxScene {
    scene::SceneIR scene;
    scene::BodyId box_body = scene::kInvalidBody;
};

PlaneBoxScene BuildPlaneBoxScene() {
    PlaneBoxScene result;

    scene::RigidBodyRecord ground;
    ground.name = "ground";
    ground.is_static = true;
    const auto ground_body = result.scene.AddRigidBody(std::move(ground));

    scene::CollisionShapeRecord plane;
    plane.body_id = ground_body;
    plane.type = scene::ShapeType::Plane;
    result.scene.AddCollisionShape(std::move(plane));

    scene::RigidBodyRecord box;
    box.name = "box";
    box.mass = 1.0f;
    box.inertia = {1.0f, 1.0f, 1.0f};
    box.local_transform.position = {0.0f, 0.45f, 0.0f};
    result.box_body = result.scene.AddRigidBody(std::move(box));

    scene::CollisionShapeRecord box_shape;
    box_shape.body_id = result.box_body;
    box_shape.type = scene::ShapeType::Box;
    box_shape.half_extents = {0.5f, 0.5f, 0.5f};
    result.scene.AddCollisionShape(std::move(box_shape));

    return result;
}

} // namespace

TEST(CudaBatchContactTiming, BatchedContactSolveUnderOneSecond) {
    constexpr uint32_t kInstanceCount = 128;
    constexpr uint32_t kStepCount = 60;

    const auto fixture = BuildPlaneBoxScene();
    auto world = runtime::BuildWorld(scene::CookScene(fixture.scene));
    std::vector<runtime::WorldInstance> instances(kInstanceCount, world.instance);
    for (uint32_t instance_index = 0; instance_index < kInstanceCount; ++instance_index) {
        auto& instance = instances[instance_index];
        instance.poses[fixture.box_body].position.x =
            static_cast<float>(instance_index % 16) * 2.0f;
        instance.poses[fixture.box_body].position.z =
            static_cast<float>(instance_index / 16) * 2.0f;
        instance.linear_velocities[fixture.box_body] = {0.0f, -1.0f, 0.0f};
    }

    auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);

    runtime::gpu::CudaBatchedWorldStepOptions options;
    options.gravity = math::Vec3::Zero();
    options.dt = 1.0f / 240.0f;
    options.step_count = kStepCount;
    options.enable_contacts = true;
    options.solver_velocity_iterations = 8u;
    options.solver_position_iterations = 4u;

    const auto start = std::chrono::high_resolution_clock::now();
    const auto report = runtime::gpu::StepBatchedCudaWorld(batch, options);
    const auto end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    EXPECT_EQ(report.instance_count, kInstanceCount);
    EXPECT_EQ(report.shape_count_per_instance, 2u);
    EXPECT_EQ(report.simulated_step_count, kStepCount);
    EXPECT_EQ(report.contact_constraint_count, kInstanceCount * kStepCount);
    EXPECT_GE(report.constraint_row_count, kInstanceCount * kStepCount * 3u);
    EXPECT_LT(ms, 1000) << "CUDA batched contact solve took " << ms << " ms";
}
