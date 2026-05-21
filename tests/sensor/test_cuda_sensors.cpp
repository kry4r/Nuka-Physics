// ---------------------------------------------------------------------------
// CUDA sensor query tests.
// ---------------------------------------------------------------------------

#include "runtime/gpu/device_world.hpp"
#include "runtime/gpu/batched_device_world.hpp"
#include "runtime/world_builder.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"
#include "sensor/gpu/cuda_sensors.cuh"
#include "sensor/state_sensor.hpp"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

using namespace nuka;

namespace {

scene::SceneIR BuildSensorScene() {
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
    sphere_body.name = "sphere";
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

runtime::gpu::DeviceWorld UploadScene(runtime::BuiltWorld& world) {
    world = runtime::BuildWorld(scene::CookScene(BuildSensorScene()));
    world.instance.forces[1] = {0.0f, -9.81f, 0.0f};
    world.instance.angular_velocities[1] = {0.1f, 0.2f, 0.3f};
    auto device_world = runtime::gpu::UploadDeviceWorld(world.template_view);
    runtime::gpu::UploadDeviceState(device_world, world.instance);
    return device_world;
}

void ExpectVec3Near(math::Vec3 actual, math::Vec3 expected, float tolerance) {
    EXPECT_NEAR(actual.x, expected.x, tolerance);
    EXPECT_NEAR(actual.y, expected.y, tolerance);
    EXPECT_NEAR(actual.z, expected.z, tolerance);
}

} // namespace

TEST(CudaSensor, SamplesImuFromDeviceWorldState) {
    runtime::BuiltWorld world;
    auto device_world = UploadScene(world);

    const auto result = sensor::gpu::QueryCudaImuSensor(device_world, {1u});
    const auto samples = result.DownloadSamples();

    ASSERT_EQ(samples.size(), 1u);

    runtime::rigid::BodyState body;
    body.inv_mass = 0.5f;
    body.position = world.instance.poses[1].position;
    body.angular_velocity = world.instance.angular_velocities[1];
    body.force = world.instance.forces[1];
    const auto reference = sensor::QueryImuSensor(body);

    ExpectVec3Near(samples[0].position, reference.position, 1.0e-5f);
    ExpectVec3Near(samples[0].angular_velocity, reference.angular_velocity, 1.0e-5f);
    ExpectVec3Near(samples[0].linear_acceleration,
                   reference.linear_acceleration,
                   1.0e-5f);
}

TEST(CudaSensor, LidarReturnsNearestSphereHitAndMaxRangeMisses) {
    runtime::BuiltWorld world;
    auto device_world = UploadScene(world);

    sensor::gpu::CudaLidarOptions options;
    options.origin = {0.0f, 1.0f, 0.0f};
    options.direction = math::Vec3::UnitX();
    options.ray_count = 3u;
    options.range = 10.0f;
    options.horizontal_fov_radians = 1.2f;

    const auto result = sensor::gpu::QueryCudaLidarSensor(device_world, options);
    const auto depths = result.DownloadDepths();

    ASSERT_EQ(depths.size(), 3u);
    EXPECT_NEAR(depths[1], 1.5f, 1.0e-3f);
    EXPECT_FLOAT_EQ(depths[0], options.range);
    EXPECT_FLOAT_EQ(depths[2], options.range);
}

TEST(CudaSensor, SamplesBatchedImuFromBatchedDeviceWorldState) {
    auto world = runtime::BuildWorld(scene::CookScene(BuildSensorScene()));
    std::vector<runtime::WorldInstance> instances(2, world.instance);
    instances[0].forces[1] = {0.0f, -9.81f, 0.0f};
    instances[0].angular_velocities[1] = {0.1f, 0.2f, 0.3f};
    instances[1].poses[1].position = {7.0f, 2.0f, 0.5f};
    instances[1].forces[1] = {4.0f, 0.0f, 0.0f};
    instances[1].angular_velocities[1] = {0.0f, 1.0f, 0.0f};

    auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);
    const std::vector<sensor::gpu::BatchedCudaBodyRequest> body_requests = {
        {0u, 1u},
        {1u, 1u}
    };

    const auto result = sensor::gpu::QueryBatchedCudaImuSensor(batch, body_requests);
    const auto samples = result.DownloadSamples();

    ASSERT_EQ(samples.size(), 2u);
    ExpectVec3Near(samples[0].position, instances[0].poses[1].position, 1.0e-5f);
    ExpectVec3Near(samples[0].angular_velocity,
                   instances[0].angular_velocities[1],
                   1.0e-5f);
    ExpectVec3Near(samples[0].linear_acceleration, {0.0f, -4.905f, 0.0f}, 1.0e-5f);
    ExpectVec3Near(samples[1].position, instances[1].poses[1].position, 1.0e-5f);
    ExpectVec3Near(samples[1].angular_velocity,
                   instances[1].angular_velocities[1],
                   1.0e-5f);
    ExpectVec3Near(samples[1].linear_acceleration, {2.0f, 0.0f, 0.0f}, 1.0e-5f);
}

TEST(CudaSensor, BatchedLidarQueriesStayWithinEachInstance) {
    auto world = runtime::BuildWorld(scene::CookScene(BuildSensorScene()));
    std::vector<runtime::WorldInstance> instances(2, world.instance);
    instances[1].poses[1].position.x = 20.0f;

    auto batch = runtime::gpu::UploadBatchedDeviceWorld(world.template_view, instances);

    std::vector<sensor::gpu::BatchedCudaLidarOptions> options(2);
    options[0].instance_index = 0u;
    options[0].origin = {0.0f, 1.0f, 0.0f};
    options[0].direction = math::Vec3::UnitX();
    options[0].ray_count = 3u;
    options[0].range = 10.0f;
    options[0].horizontal_fov_radians = 1.2f;

    options[1].instance_index = 1u;
    options[1].origin = {18.0f, 1.0f, 0.0f};
    options[1].direction = math::Vec3::UnitX();
    options[1].ray_count = 3u;
    options[1].range = 10.0f;
    options[1].horizontal_fov_radians = 1.2f;

    const auto result = sensor::gpu::QueryBatchedCudaLidarSensor(batch, options);
    const auto depths = result.DownloadDepths();

    ASSERT_EQ(depths.size(), 6u);
    EXPECT_NEAR(depths[1], 1.5f, 1.0e-3f);
    EXPECT_FLOAT_EQ(depths[0], options[0].range);
    EXPECT_FLOAT_EQ(depths[2], options[0].range);
    EXPECT_NEAR(depths[4], 1.5f, 1.0e-3f);
    EXPECT_FLOAT_EQ(depths[3], options[1].range);
    EXPECT_FLOAT_EQ(depths[5], options[1].range);
}
