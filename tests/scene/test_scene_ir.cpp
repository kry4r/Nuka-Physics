// ---------------------------------------------------------------------------
// Tests for nuka::scene::SceneIR
// ---------------------------------------------------------------------------

#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

using namespace nuka::scene;

// ---------------------------------------------------------------------------
// Basic body + joint
// ---------------------------------------------------------------------------

TEST(SceneIr, SupportsRigidBodyAndJointRecords) {
    SceneIR scene;
    const auto body  = scene.AddRigidBody("base_link");
    const auto joint = scene.AddJoint("joint0", body, body);
    EXPECT_EQ(scene.RigidBodyCount(), 1u);
    EXPECT_EQ(scene.JointCount(), 1u);
    EXPECT_EQ(body, 0u);
    EXPECT_EQ(joint, 0u);
}

// ---------------------------------------------------------------------------
// Collision shapes
// ---------------------------------------------------------------------------

TEST(SceneIr, AddCollisionShapeSimple) {
    SceneIR scene;
    const auto body  = scene.AddRigidBody("box_body");
    const auto shape = scene.AddCollisionShape(body, ShapeType::Box);
    EXPECT_EQ(scene.ShapeCount(), 1u);
    EXPECT_EQ(scene.GetShape(shape).body_id, body);
    EXPECT_EQ(scene.GetShape(shape).type, ShapeType::Box);
}

TEST(SceneIr, AddCollisionShapeWithRecord) {
    SceneIR scene;
    const auto body = scene.AddRigidBody("sphere_body");

    CollisionShapeRecord rec;
    rec.body_id = body;
    rec.type    = ShapeType::Sphere;
    rec.radius  = 1.5f;
    const auto shape = scene.AddCollisionShape(std::move(rec));

    EXPECT_EQ(scene.ShapeCount(), 1u);
    EXPECT_FLOAT_EQ(scene.GetShape(shape).radius, 1.5f);
    EXPECT_EQ(scene.GetShape(shape).type, ShapeType::Sphere);
}

// ---------------------------------------------------------------------------
// Sensors
// ---------------------------------------------------------------------------

TEST(SceneIr, AddSensor) {
    SceneIR scene;
    const auto body   = scene.AddRigidBody("sensor_body");
    const auto sensor = scene.AddSensor("imu0", body);
    EXPECT_EQ(scene.SensorCount(), 1u);
    EXPECT_EQ(scene.GetSensor(sensor).attached_body, body);
    EXPECT_EQ(scene.GetSensor(sensor).name, "imu0");
}

// ---------------------------------------------------------------------------
// Record field access
// ---------------------------------------------------------------------------

TEST(SceneIr, BodyRecordFieldDefaults) {
    SceneIR scene;
    const auto id = scene.AddRigidBody("link");
    const auto& body = scene.GetBody(id);
    EXPECT_EQ(body.name, "link");
    EXPECT_EQ(body.parent_id, kInvalidBody);
    EXPECT_FLOAT_EQ(body.mass, 1.0f);
    EXPECT_FALSE(body.is_static);
}

TEST(SceneIr, BodyRecordWithCustomFields) {
    SceneIR scene;
    RigidBodyRecord rec;
    rec.name      = "ground";
    rec.mass      = 0.0f;
    rec.is_static = true;
    rec.inertia   = {0.0f, 0.0f, 0.0f};
    const auto id = scene.AddRigidBody(std::move(rec));

    const auto& body = scene.GetBody(id);
    EXPECT_TRUE(body.is_static);
    EXPECT_FLOAT_EQ(body.mass, 0.0f);
}

TEST(SceneIr, JointRecordFieldAccess) {
    SceneIR scene;
    const auto b0 = scene.AddRigidBody("parent");
    const auto b1 = scene.AddRigidBody("child");
    const auto jid = scene.AddJoint("hinge", b0, b1);

    const auto& j = scene.GetJoint(jid);
    EXPECT_EQ(j.name, "hinge");
    EXPECT_EQ(j.parent_body, b0);
    EXPECT_EQ(j.child_body, b1);
    EXPECT_EQ(j.type, JointType::Revolute);
}

// ---------------------------------------------------------------------------
// Multiple bodies
// ---------------------------------------------------------------------------

TEST(SceneIr, MultipleBodies) {
    SceneIR scene;
    const auto a = scene.AddRigidBody("a");
    const auto b = scene.AddRigidBody("b");
    const auto c = scene.AddRigidBody("c");

    EXPECT_EQ(scene.RigidBodyCount(), 3u);
    EXPECT_EQ(a, 0u);
    EXPECT_EQ(b, 1u);
    EXPECT_EQ(c, 2u);

    EXPECT_EQ(scene.GetBody(a).name, "a");
    EXPECT_EQ(scene.GetBody(b).name, "b");
    EXPECT_EQ(scene.GetBody(c).name, "c");
}

// ---------------------------------------------------------------------------
// Bulk accessors
// ---------------------------------------------------------------------------

TEST(SceneIr, BulkAccessors) {
    SceneIR scene;
    scene.AddRigidBody("r0");
    scene.AddRigidBody("r1");
    scene.AddJoint("j0", 0, 1);
    scene.AddCollisionShape(0, ShapeType::Sphere);
    scene.AddSensor("s0", 0);

    EXPECT_EQ(scene.Bodies().size(), 2u);
    EXPECT_EQ(scene.Joints().size(), 1u);
    EXPECT_EQ(scene.Shapes().size(), 1u);
    EXPECT_EQ(scene.Sensors().size(), 1u);
}

// ---------------------------------------------------------------------------
// Out-of-range access throws
// ---------------------------------------------------------------------------

TEST(SceneIr, OutOfRangeThrows) {
    SceneIR scene;
    EXPECT_THROW(scene.GetBody(0), std::out_of_range);
    EXPECT_THROW(scene.GetJoint(0), std::out_of_range);
    EXPECT_THROW(scene.GetShape(0), std::out_of_range);
    EXPECT_THROW(scene.GetSensor(0), std::out_of_range);
}
