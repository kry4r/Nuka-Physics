// ---------------------------------------------------------------------------
// Tests for nuka::import::LoadUsd
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "scene/cooker.hpp"

#include <gtest/gtest.h>
#include <stdexcept>

TEST(UsdImporter, LoadsUsdPhysicsRigidBodyAndJoint) {
    const auto scene = nuka::import::LoadUsd("tests/data/minimal_scene.usda");

    EXPECT_EQ(scene.RigidBodyCount(), 2u);
    EXPECT_EQ(scene.JointCount(), 1u);
    EXPECT_EQ(scene.ShapeCount(), 2u);
}

TEST(UsdImporter, ParsesRigidBodyNamesAndMass) {
    const auto scene = nuka::import::LoadUsd("tests/data/minimal_scene.usda");

    ASSERT_EQ(scene.RigidBodyCount(), 2u);
    EXPECT_EQ(scene.GetBody(0).name, "base");
    EXPECT_EQ(scene.GetBody(1).name, "link1");
    EXPECT_TRUE(scene.GetBody(0).is_static);
    EXPECT_FLOAT_EQ(scene.GetBody(1).mass, 0.5f);
}

TEST(UsdImporter, ParsesUsdPhysicsJoint) {
    const auto scene = nuka::import::LoadUsd("tests/data/minimal_scene.usda");

    ASSERT_EQ(scene.JointCount(), 1u);
    const auto& joint = scene.GetJoint(0);
    EXPECT_EQ(joint.name, "joint0");
    EXPECT_EQ(joint.type, nuka::scene::JointType::Revolute);
    EXPECT_EQ(joint.parent_body, 0u);
    EXPECT_EQ(joint.child_body, 1u);
    EXPECT_FLOAT_EQ(joint.axis.z, 1.0f);
}

TEST(UsdImporter, CookedBlobContainsUsdImportedTables) {
    const auto scene = nuka::import::LoadUsd("tests/data/minimal_scene.usda");
    const auto blob = nuka::scene::CookScene(scene);

    EXPECT_EQ(blob.body_count, 2u);
    EXPECT_EQ(blob.joint_count, 1u);
    EXPECT_EQ(blob.shape_count, 2u);
    EXPECT_EQ(blob.material_count, 1u);
    EXPECT_EQ(blob.camera_count, 1u);
    EXPECT_EQ(blob.light_count, 1u);
    EXPECT_EQ(blob.actuator_count, 1u);
    EXPECT_EQ(blob.sensor_count, 1u);
}

TEST(UsdImporter, ParsesRenderControlAndSensorRecords) {
    const auto scene = nuka::import::LoadUsd("tests/data/minimal_scene.usda");

    EXPECT_EQ(scene.MaterialCount(), 1u);
    EXPECT_EQ(scene.CameraCount(), 1u);
    EXPECT_EQ(scene.LightCount(), 1u);
    EXPECT_EQ(scene.ActuatorCount(), 1u);
    EXPECT_EQ(scene.SensorCount(), 1u);

    EXPECT_EQ(scene.GetMaterial(0).name, "mat_robot");
    EXPECT_FLOAT_EQ(scene.GetMaterial(0).base_color.x, 0.8f);
    EXPECT_EQ(scene.GetCamera(0).name, "track");
    EXPECT_EQ(scene.GetLight(0).type, nuka::scene::LightType::Directional);
    EXPECT_EQ(scene.GetActuator(0).type, nuka::scene::ActuatorType::Velocity);
    EXPECT_EQ(scene.GetSensor(0).type, nuka::scene::SensorType::Imu);
}

TEST(UsdImporter, ThrowsOnMissingFile) {
    EXPECT_THROW(nuka::import::LoadUsd("tests/data/missing_scene.usda"), std::runtime_error);
}
