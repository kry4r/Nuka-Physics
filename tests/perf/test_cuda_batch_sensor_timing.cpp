// ---------------------------------------------------------------------------
// Performance test: CUDA batched sensor query timing
// ---------------------------------------------------------------------------

#include "runtime/gpu/batched_device_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"
#include "sensor/gpu/cuda_sensors.cuh"

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <utility>
#include <vector>

using namespace nuka;

namespace {

scene::SceneIR BuildBatchedSensorScene() {
    scene::SceneIR scene;

    scene::RigidBodyRecord ground;
    ground.name = "ground";
    ground.is_static = true;
    const auto ground_id = scene.AddRigidBody(std::move(ground));

    scene::CollisionShapeRecord plane;
    plane.body_id = ground_id;
    plane.type = scene::ShapeType::Plane;
    scene.AddCollisionShape(std::move(plane));

    scene::RigidBodyRecord sphere_body;
    sphere_body.name = "sensor_target";
    sphere_body.mass = 2.0f;
    sphere_body.inertia = {1.0f, 1.0f, 1.0f};
    sphere_body.local_transform.position = {2.0f, 1.0f, 0.0f};
    const auto sphere_body_id = scene.AddRigidBody(std::move(sphere_body));

    scene::CollisionShapeRecord sphere;
    sphere.body_id = sphere_body_id;
    sphere.type = scene::ShapeType::Sphere;
    sphere.radius = 0.5f;
    scene.AddCollisionShape(std::move(sphere));

    return scene;
}

} // namespace

TEST(CudaBatchSensorTiming, BatchedImuAndLidarQueriesUnderOneSecond) {
    constexpr uint32_t kInstanceCount = 128;
    constexpr uint32_t kRayCountPerFan = 64;
    constexpr uint32_t kIterationCount = 60;

    auto world = runtime::BuildWorld(scene::CookScene(BuildBatchedSensorScene()));
    std::vector<runtime::WorldInstance> instances(kInstanceCount, world.instance);
    for (uint32_t instance_index = 0; instance_index < kInstanceCount; ++instance_index) {
        auto& instance = instances[instance_index];
        instance.forces[1] = {0.0f, -9.81f, 0.0f};
        instance.angular_velocities[1] = {0.0f, 0.0f, 0.25f};
        instance.poses[1].position.z = static_cast<float>(instance_index % 8) * 0.05f;
    }

    auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);

    std::vector<sensor::gpu::BatchedCudaBodyRequest> body_requests;
    body_requests.reserve(kInstanceCount);
    std::vector<sensor::gpu::BatchedCudaLidarOptions> lidar_options;
    lidar_options.reserve(kInstanceCount);
    for (uint32_t instance_index = 0; instance_index < kInstanceCount; ++instance_index) {
        body_requests.push_back({instance_index, 1u});

        sensor::gpu::BatchedCudaLidarOptions options;
        options.instance_index = instance_index;
        options.origin = {0.0f, 1.0f, static_cast<float>(instance_index % 8) * 0.05f};
        options.direction = math::Vec3::UnitX();
        options.ray_count = kRayCountPerFan;
        options.range = 16.0f;
        options.horizontal_fov_radians = 0.75f;
        lidar_options.push_back(options);
    }

    sensor::gpu::CudaImuResult last_imu;
    sensor::gpu::BatchedCudaLidarResult last_lidar;
    const auto start = std::chrono::high_resolution_clock::now();
    for (uint32_t iteration = 0; iteration < kIterationCount; ++iteration) {
        last_imu = sensor::gpu::QueryBatchedCudaImuSensor(batch, body_requests);
        last_lidar = sensor::gpu::QueryBatchedCudaLidarSensor(batch, lidar_options);
    }
    const auto end = std::chrono::high_resolution_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    const auto imu_samples = last_imu.DownloadSamples();
    const auto offsets = last_lidar.DownloadRayOffsets();
    const auto depths = last_lidar.DownloadDepths();

    EXPECT_EQ(imu_samples.size(), kInstanceCount);
    EXPECT_EQ(offsets.size(), kInstanceCount + 1u);
    EXPECT_EQ(depths.size(), kInstanceCount * kRayCountPerFan);
    EXPECT_NEAR(imu_samples.front().linear_acceleration.y, -4.905f, 1.0e-5f);
    EXPECT_LT(*std::min_element(depths.begin(), depths.end()), 16.0f);
    EXPECT_LT(ms, 1000) << "CUDA batched sensor query pipeline took " << ms << " ms";
}
