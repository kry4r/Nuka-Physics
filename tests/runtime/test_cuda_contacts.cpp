// ---------------------------------------------------------------------------
// CUDA broadphase and contact generation tests.
// ---------------------------------------------------------------------------

#include "collision/gpu/broadphase.cuh"
#include "constraint/gpu/contact_generation.cuh"
#include "runtime/gpu/device_world.hpp"
#include "runtime/world_builder.hpp"
#include "runtime/world_stepper.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>

using namespace nuka;

namespace {

scene::SceneIR BuildPlaneBoxScene() {
    scene::SceneIR scene;

    scene::RigidBodyRecord ground;
    ground.name = "ground";
    ground.is_static = true;
    const auto ground_id = scene.AddRigidBody(std::move(ground));

    scene::CollisionShapeRecord plane;
    plane.body_id = ground_id;
    plane.type = scene::ShapeType::Plane;
    scene.AddCollisionShape(std::move(plane));

    scene::RigidBodyRecord box;
    box.name = "box";
    box.mass = 1.0f;
    box.inertia = {1.0f, 1.0f, 1.0f};
    box.local_transform.position = {0.0f, 0.45f, 0.0f};
    const auto box_id = scene.AddRigidBody(std::move(box));

    scene::CollisionShapeRecord box_shape;
    box_shape.body_id = box_id;
    box_shape.type = scene::ShapeType::Box;
    box_shape.half_extents = {0.5f, 0.5f, 0.5f};
    scene.AddCollisionShape(std::move(box_shape));

    return scene;
}

scene::SceneIR BuildSphereSphereScene() {
    scene::SceneIR scene;

    scene::RigidBodyRecord first;
    first.name = "first";
    first.mass = 1.0f;
    first.local_transform.position = {0.0f, 0.0f, 0.0f};
    const auto first_id = scene.AddRigidBody(std::move(first));

    scene::CollisionShapeRecord first_shape;
    first_shape.body_id = first_id;
    first_shape.type = scene::ShapeType::Sphere;
    first_shape.radius = 0.75f;
    scene.AddCollisionShape(std::move(first_shape));

    scene::RigidBodyRecord second;
    second.name = "second";
    second.mass = 1.0f;
    second.local_transform.position = {1.0f, 0.0f, 0.0f};
    const auto second_id = scene.AddRigidBody(std::move(second));

    scene::CollisionShapeRecord second_shape;
    second_shape.body_id = second_id;
    second_shape.type = scene::ShapeType::Sphere;
    second_shape.radius = 0.75f;
    scene.AddCollisionShape(std::move(second_shape));

    return scene;
}

scene::SceneIR BuildThreeBoxPairOrderingScene() {
    scene::SceneIR scene;

    const math::Vec3 positions[] = {
        {0.0f, 0.0f, 0.0f},
        {0.75f, 0.0f, 0.0f},
        {1.5f, 0.0f, 0.0f}
    };

    for (math::Vec3 position : positions) {
        scene::RigidBodyRecord body;
        body.name = "box";
        body.mass = 1.0f;
        body.local_transform.position = position;
        const auto body_id = scene.AddRigidBody(std::move(body));

        scene::CollisionShapeRecord shape;
        shape.body_id = body_id;
        shape.type = scene::ShapeType::Box;
        shape.half_extents = {0.5f, 0.5f, 0.5f};
        scene.AddCollisionShape(std::move(shape));
    }

    return scene;
}

runtime::gpu::DeviceWorld UploadSceneToDevice(const scene::SceneIR& scene,
                                              runtime::BuiltWorld& world) {
    world = runtime::BuildWorld(scene::CookScene(scene));
    auto device_world = runtime::gpu::UploadDeviceWorld(world.template_view);
    runtime::gpu::UploadDeviceState(device_world, world.instance);
    return device_world;
}

} // namespace

TEST(CudaContacts, GeneratesAabbsPairsAndPlaneBoxContactOnDevice) {
    runtime::BuiltWorld world;
    auto device_world = UploadSceneToDevice(BuildPlaneBoxScene(), world);

    auto broadphase = collision::gpu::BuildCudaBroadphase(device_world);
    const auto aabbs = broadphase.DownloadAabbs();
    const auto pairs = broadphase.DownloadPairs();

    ASSERT_EQ(aabbs.size(), 2u);
    EXPECT_NEAR(aabbs[1].min.y, -0.05f, 1.0e-6f);
    EXPECT_NEAR(aabbs[1].max.y, 0.95f, 1.0e-6f);

    ASSERT_EQ(pairs.size(), 1u);
    EXPECT_EQ(pairs[0].body_a, 0u);
    EXPECT_EQ(pairs[0].body_b, 1u);

    auto contacts = constraint::gpu::GenerateCudaContacts(device_world, broadphase);
    const auto report = contacts.DownloadReport();
    const auto manifolds = contacts.DownloadManifolds();

    EXPECT_EQ(report.pair_count, 1u);
    EXPECT_EQ(report.contact_manifold_count, 1u);
    EXPECT_EQ(report.contact_point_count, 1u);

    ASSERT_EQ(manifolds.size(), 1u);
    EXPECT_EQ(manifolds[0].a.handle, 1u);
    EXPECT_EQ(manifolds[0].b.handle, 0u);
    ASSERT_EQ(manifolds[0].point_count, 1u);
    EXPECT_NEAR(manifolds[0].points[0].position.y, 0.0f, 1.0e-6f);
    EXPECT_EQ(manifolds[0].points[0].normal, math::Vec3::UnitY());
    EXPECT_NEAR(manifolds[0].points[0].penetration, 0.05f, 1.0e-6f);

    runtime::WorldStepOptions cpu_options;
    cpu_options.gravity = math::Vec3::Zero();
    const auto cpu_report = runtime::StepWorldInstance(world.template_view,
                                                       world.instance,
                                                       cpu_options);
    EXPECT_EQ(report.pair_count, cpu_report.broadphase_pair_count);
    EXPECT_EQ(report.contact_manifold_count, cpu_report.contact_manifold_count);
    EXPECT_EQ(report.contact_point_count, cpu_report.contact_point_count);
}

TEST(CudaContacts, GeneratesSphereSphereContactOnDevice) {
    runtime::BuiltWorld world;
    auto device_world = UploadSceneToDevice(BuildSphereSphereScene(), world);

    auto broadphase = collision::gpu::BuildCudaBroadphase(device_world);
    auto contacts = constraint::gpu::GenerateCudaContacts(device_world, broadphase);
    const auto manifolds = contacts.DownloadManifolds();

    ASSERT_EQ(manifolds.size(), 1u);
    EXPECT_EQ(manifolds[0].a.handle, 0u);
    EXPECT_EQ(manifolds[0].b.handle, 1u);
    ASSERT_EQ(manifolds[0].point_count, 1u);
    EXPECT_NEAR(manifolds[0].points[0].normal.x, -1.0f, 1.0e-6f);
    EXPECT_NEAR(manifolds[0].points[0].normal.y, 0.0f, 1.0e-6f);
    EXPECT_NEAR(manifolds[0].points[0].normal.z, 0.0f, 1.0e-6f);
    EXPECT_NEAR(manifolds[0].points[0].penetration, 0.5f, 1.0e-6f);
}

TEST(CudaContacts, EmitsDeterministicPairOrderFromDeviceBroadphase) {
    runtime::BuiltWorld world;
    auto device_world = UploadSceneToDevice(BuildThreeBoxPairOrderingScene(), world);

    auto broadphase = collision::gpu::BuildCudaBroadphase(device_world);
    const auto pairs = broadphase.DownloadPairs();

    ASSERT_EQ(pairs.size(), 2u);
    EXPECT_EQ(pairs[0].body_a, 0u);
    EXPECT_EQ(pairs[0].body_b, 1u);
    EXPECT_EQ(pairs[1].body_a, 1u);
    EXPECT_EQ(pairs[1].body_b, 2u);
}
