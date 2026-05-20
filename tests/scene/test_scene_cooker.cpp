// ---------------------------------------------------------------------------
// Tests for nuka::scene::CookScene
// ---------------------------------------------------------------------------

#include "scene/cooker.hpp"

#include <gtest/gtest.h>

using namespace nuka::scene;

// ---------------------------------------------------------------------------
// Minimal scene
// ---------------------------------------------------------------------------

TEST(SceneCooker, EmitsBodyTableForMinimalScene) {
    SceneIR scene;
    scene.AddRigidBody("root");
    const auto blob = CookScene(scene);
    EXPECT_EQ(blob.body_count, 1u);
    EXPECT_EQ(blob.bodies.poses.size(), 1u);
    EXPECT_EQ(blob.bodies.inv_masses.size(), 1u);
}

// ---------------------------------------------------------------------------
// Static body gets inv_mass = 0
// ---------------------------------------------------------------------------

TEST(SceneCooker, StaticBodyHasZeroInvMass) {
    SceneIR scene;
    RigidBodyRecord rec;
    rec.name      = "ground";
    rec.mass      = 100.0f;
    rec.is_static = true;
    scene.AddRigidBody(std::move(rec));

    const auto blob = CookScene(scene);
    EXPECT_FLOAT_EQ(blob.bodies.inv_masses[0], 0.0f);
    EXPECT_EQ(blob.bodies.is_static[0], 1u);

    // Inverse inertia should also be zero for static bodies.
    const auto& inv_i = blob.bodies.inv_inertias[0];
    EXPECT_FLOAT_EQ(inv_i.x, 0.0f);
    EXPECT_FLOAT_EQ(inv_i.y, 0.0f);
    EXPECT_FLOAT_EQ(inv_i.z, 0.0f);
}

// ---------------------------------------------------------------------------
// Dynamic body inverse mass
// ---------------------------------------------------------------------------

TEST(SceneCooker, DynamicBodyInverseMass) {
    SceneIR scene;
    RigidBodyRecord rec;
    rec.name    = "box";
    rec.mass    = 4.0f;
    rec.inertia = {2.0f, 4.0f, 8.0f};
    scene.AddRigidBody(std::move(rec));

    const auto blob = CookScene(scene);
    EXPECT_FLOAT_EQ(blob.bodies.inv_masses[0], 0.25f);
    EXPECT_FLOAT_EQ(blob.bodies.inv_inertias[0].x, 0.5f);
    EXPECT_FLOAT_EQ(blob.bodies.inv_inertias[0].y, 0.25f);
    EXPECT_FLOAT_EQ(blob.bodies.inv_inertias[0].z, 0.125f);
    EXPECT_EQ(blob.bodies.is_static[0], 0u);
}

// ---------------------------------------------------------------------------
// Joint cooking
// ---------------------------------------------------------------------------

TEST(SceneCooker, CooksJoints) {
    SceneIR scene;
    const auto b0 = scene.AddRigidBody("parent");
    const auto b1 = scene.AddRigidBody("child");

    JointRecord jrec;
    jrec.name        = "hinge";
    jrec.parent_body = b0;
    jrec.child_body  = b1;
    jrec.type        = JointType::Prismatic;
    jrec.axis        = {1.0f, 0.0f, 0.0f};
    jrec.parent_frame.position = {0.25f, 0.0f, 0.0f};
    jrec.child_frame.position = {-0.75f, 0.0f, 0.0f};
    jrec.lower_limit = -1.0f;
    jrec.upper_limit =  2.0f;
    scene.AddJoint(std::move(jrec));

    const auto blob = CookScene(scene);
    EXPECT_EQ(blob.joint_count, 1u);
    EXPECT_EQ(blob.joints.types[0], JointType::Prismatic);
    EXPECT_EQ(blob.joints.parent_bodies[0], b0);
    EXPECT_EQ(blob.joints.child_bodies[0], b1);
    EXPECT_FLOAT_EQ(blob.joints.axes[0].x, 1.0f);
    EXPECT_FLOAT_EQ(blob.joints.parent_frames[0].position.x, 0.25f);
    EXPECT_FLOAT_EQ(blob.joints.child_frames[0].position.x, -0.75f);
    EXPECT_FLOAT_EQ(blob.joints.lower_limits[0], -1.0f);
    EXPECT_FLOAT_EQ(blob.joints.upper_limits[0], 2.0f);
}

// ---------------------------------------------------------------------------
// Shape cooking
// ---------------------------------------------------------------------------

TEST(SceneCooker, CooksShapes) {
    SceneIR scene;
    const auto body = scene.AddRigidBody("obj");

    CollisionShapeRecord srec;
    srec.body_id      = body;
    srec.type         = ShapeType::Capsule;
    srec.radius       = 2.0f;
    srec.half_height  = 3.0f;
    srec.half_extents = {0.0f, 0.0f, 0.0f};
    scene.AddCollisionShape(std::move(srec));

    const auto blob = CookScene(scene);
    EXPECT_EQ(blob.shape_count, 1u);
    EXPECT_EQ(blob.shapes.types[0], ShapeType::Capsule);
    EXPECT_EQ(blob.shapes.body_ids[0], body);
    EXPECT_FLOAT_EQ(blob.shapes.radii[0], 2.0f);
    EXPECT_FLOAT_EQ(blob.shapes.half_heights[0], 3.0f);
}

// ---------------------------------------------------------------------------
// Render, sensor, and actuator metadata cooking
// ---------------------------------------------------------------------------

TEST(SceneCooker, CooksImportMetadataTables) {
    SceneIR scene;
    const auto body = scene.AddRigidBody("robot");
    const auto child = scene.AddRigidBody("tool");
    const auto joint = scene.AddJoint("tool_joint", body, child);
    scene.AddSensor("imu", child);

    MaterialRecord material;
    material.name = "blue";
    material.base_color = {0.0f, 0.1f, 1.0f};
    scene.AddMaterial(std::move(material));

    CameraRecord camera;
    camera.name = "cam";
    camera.attached_body = child;
    camera.vertical_fov_degrees = 70.0f;
    scene.AddCamera(std::move(camera));

    LightRecord light;
    light.name = "sun";
    light.type = LightType::Directional;
    light.intensity = 1200.0f;
    scene.AddLight(std::move(light));

    ActuatorRecord actuator;
    actuator.name = "motor";
    actuator.joint_id = joint;
    actuator.type = ActuatorType::Velocity;
    actuator.gain = 10.0f;
    scene.AddActuator(std::move(actuator));

    const auto blob = CookScene(scene);
    EXPECT_EQ(blob.sensor_count, 1u);
    EXPECT_EQ(blob.material_count, 1u);
    EXPECT_EQ(blob.camera_count, 1u);
    EXPECT_EQ(blob.light_count, 1u);
    EXPECT_EQ(blob.actuator_count, 1u);
    EXPECT_EQ(blob.materials.base_colors[0], nuka::math::Vec3(0.0f, 0.1f, 1.0f));
    EXPECT_EQ(blob.cameras.attached_bodies[0], child);
    EXPECT_EQ(blob.lights.types[0], LightType::Directional);
    EXPECT_EQ(blob.actuators.joint_ids[0], joint);
}

// ---------------------------------------------------------------------------
// Multiple bodies
// ---------------------------------------------------------------------------

TEST(SceneCooker, MultipleBodiesCooked) {
    SceneIR scene;
    scene.AddRigidBody("a");
    scene.AddRigidBody("b");
    scene.AddRigidBody("c");

    const auto blob = CookScene(scene);
    EXPECT_EQ(blob.body_count, 3u);
    EXPECT_EQ(blob.bodies.poses.size(), 3u);
    EXPECT_EQ(blob.bodies.inv_masses.size(), 3u);
    EXPECT_EQ(blob.bodies.inv_inertias.size(), 3u);
    EXPECT_EQ(blob.bodies.is_static.size(), 3u);
}

TEST(SceneCooker, CooksBodyPosesInWorldSpaceForRuntimeStepping) {
    SceneIR scene;

    RigidBodyRecord base;
    base.name = "base";
    base.local_transform.position = {2.0f, 0.0f, 0.0f};
    const auto base_id = scene.AddRigidBody(std::move(base));

    RigidBodyRecord child;
    child.name = "child";
    child.parent_id = base_id;
    child.local_transform.position = {0.0f, 3.0f, 0.0f};
    scene.AddRigidBody(std::move(child));

    const auto blob = CookScene(scene);

    ASSERT_EQ(blob.bodies.poses.size(), 2u);
    EXPECT_FLOAT_EQ(blob.bodies.poses[0].position.x, 2.0f);
    EXPECT_FLOAT_EQ(blob.bodies.poses[0].position.y, 0.0f);
    EXPECT_FLOAT_EQ(blob.bodies.poses[1].position.x, 2.0f);
    EXPECT_FLOAT_EQ(blob.bodies.poses[1].position.y, 3.0f);
}

// ---------------------------------------------------------------------------
// Empty scene
// ---------------------------------------------------------------------------

TEST(SceneCooker, EmptySceneProducesEmptyBlob) {
    SceneIR scene;
    const auto blob = CookScene(scene);
    EXPECT_EQ(blob.body_count, 0u);
    EXPECT_EQ(blob.joint_count, 0u);
    EXPECT_EQ(blob.shape_count, 0u);
    EXPECT_EQ(blob.sensor_count, 0u);
    EXPECT_EQ(blob.material_count, 0u);
    EXPECT_EQ(blob.camera_count, 0u);
    EXPECT_EQ(blob.light_count, 0u);
    EXPECT_EQ(blob.actuator_count, 0u);
}
