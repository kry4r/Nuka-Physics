// ---------------------------------------------------------------------------
// CUDA sensor timing coverage
// ---------------------------------------------------------------------------

#include "runtime/gpu/device_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"
#include "sensor/gpu/cuda_sensors.cuh"

#include <gtest/gtest.h>

#include <chrono>
#include <algorithm>
#include <utility>
#include <vector>

using namespace nuka;

namespace {

scene::SceneIR BuildSensorGrid(int sphere_count) {
    scene::SceneIR scene;

    scene::RigidBodyRecord ground;
    ground.name = "ground";
    ground.is_static = true;
    const auto ground_id = scene.AddRigidBody(std::move(ground));

    scene::CollisionShapeRecord plane;
    plane.body_id = ground_id;
    plane.type = scene::ShapeType::Plane;
    scene.AddCollisionShape(std::move(plane));

    for (int index = 0; index < sphere_count; ++index) {
        scene::RigidBodyRecord body;
        body.name = "sensor_target";
        body.mass = 1.0f;
        body.inertia = {1.0f, 1.0f, 1.0f};
        body.local_transform.position = {
            4.0f + static_cast<float>(index % 32) * 1.5f,
            1.0f,
            static_cast<float>(index / 32) * 1.5f
        };
        const auto body_id = scene.AddRigidBody(std::move(body));

        scene::CollisionShapeRecord sphere;
        sphere.body_id = body_id;
        sphere.type = scene::ShapeType::Sphere;
        sphere.radius = 0.35f;
        scene.AddCollisionShape(std::move(sphere));
    }

    return scene;
}

} // namespace

TEST(CudaSensorTiming, ImuAndLidarQueriesUnderOneSecond) {
    constexpr int kSphereCount = 256;
    constexpr int kIterationCount = 60;

    auto world = runtime::BuildWorld(scene::CookScene(BuildSensorGrid(kSphereCount)));
    for (uint32_t body = 1; body < world.instance.body_count; ++body) {
        world.instance.forces[body] = {0.0f, -9.81f, 0.0f};
        world.instance.angular_velocities[body] = {0.0f, 0.0f, 0.25f};
    }

    auto device_world = runtime::gpu::UploadDeviceWorld(world.template_view);
    runtime::gpu::UploadDeviceState(device_world, world.instance);

    std::vector<scene::BodyId> body_ids;
    body_ids.reserve(world.instance.body_count - 1u);
    for (uint32_t body = 1; body < world.instance.body_count; ++body) {
        body_ids.push_back(body);
    }

    sensor::gpu::CudaLidarOptions lidar_options;
    lidar_options.origin = {0.0f, 1.0f, 0.0f};
    lidar_options.direction = math::Vec3::UnitX();
    lidar_options.ray_count = 512u;
    lidar_options.range = 128.0f;
    lidar_options.horizontal_fov_radians = 1.0f;

    sensor::gpu::CudaImuResult last_imu;
    sensor::gpu::CudaLidarResult last_lidar;
    const auto start = std::chrono::high_resolution_clock::now();
    for (int iteration = 0; iteration < kIterationCount; ++iteration) {
        last_imu = sensor::gpu::QueryCudaImuSensor(device_world, body_ids);
        last_lidar = sensor::gpu::QueryCudaLidarSensor(device_world, lidar_options);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    const auto imu_samples = last_imu.DownloadSamples();
    const auto depths = last_lidar.DownloadDepths();

    EXPECT_EQ(imu_samples.size(), body_ids.size());
    EXPECT_EQ(depths.size(), lidar_options.ray_count);
    EXPECT_NEAR(imu_samples.front().linear_acceleration.y, -9.81f, 1.0e-5f);
    EXPECT_LT(*std::min_element(depths.begin(), depths.end()), lidar_options.range);
    EXPECT_LT(ms, 1000) << "CUDA sensor query pipeline took " << ms << " ms";
}
