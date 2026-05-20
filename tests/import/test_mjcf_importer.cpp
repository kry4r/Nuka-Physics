// ---------------------------------------------------------------------------
// Tests for nuka::import::LoadMjcf
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"
#include <gtest/gtest.h>

TEST(MjcfImporter, LoadsMinimalBodyAndJoint) {
    const auto scene = nuka::import::LoadMjcf("tests/data/minimal_arm.xml");
    EXPECT_EQ(scene.RigidBodyCount(), 2u);
    EXPECT_EQ(scene.JointCount(), 1u);
}

TEST(MjcfImporter, BodyNamesAreCorrect) {
    const auto scene = nuka::import::LoadMjcf("tests/data/minimal_arm.xml");
    ASSERT_GE(scene.RigidBodyCount(), 2u);
    EXPECT_EQ(scene.GetBody(0).name, "base");
    EXPECT_EQ(scene.GetBody(1).name, "link1");
}

TEST(MjcfImporter, MassValuesAreParsed) {
    const auto scene = nuka::import::LoadMjcf("tests/data/minimal_arm.xml");
    ASSERT_GE(scene.RigidBodyCount(), 2u);
    EXPECT_FLOAT_EQ(scene.GetBody(0).mass, 1.0f);
    EXPECT_FLOAT_EQ(scene.GetBody(1).mass, 0.5f);
}

TEST(MjcfImporter, ShapesAreParsed) {
    const auto scene = nuka::import::LoadMjcf("tests/data/minimal_arm.xml");
    // Two geoms: one box for "base", one capsule for "link1"
    EXPECT_EQ(scene.ShapeCount(), 2u);
}

TEST(MjcfImporter, ShapeTypesAreCorrect) {
    const auto scene = nuka::import::LoadMjcf("tests/data/minimal_arm.xml");
    ASSERT_GE(scene.ShapeCount(), 2u);
    EXPECT_EQ(scene.GetShape(0).type, nuka::scene::ShapeType::Box);
    EXPECT_EQ(scene.GetShape(1).type, nuka::scene::ShapeType::Capsule);
}

TEST(MjcfImporter, JointTypeIsParsed) {
    const auto scene = nuka::import::LoadMjcf("tests/data/minimal_arm.xml");
    ASSERT_EQ(scene.JointCount(), 1u);
    // "hinge" maps to Revolute
    EXPECT_EQ(scene.GetJoint(0).type, nuka::scene::JointType::Revolute);
}

TEST(MjcfImporter, JointNameIsParsed) {
    const auto scene = nuka::import::LoadMjcf("tests/data/minimal_arm.xml");
    ASSERT_EQ(scene.JointCount(), 1u);
    EXPECT_EQ(scene.GetJoint(0).name, "joint0");
}

TEST(MjcfImporter, ParsesSceneAuthoringRecords) {
    const auto scene = nuka::import::LoadMjcf("tests/data/complete_robot.xml");

    EXPECT_EQ(scene.RigidBodyCount(), 2u);
    EXPECT_EQ(scene.JointCount(), 1u);
    EXPECT_EQ(scene.ShapeCount(), 2u);
    EXPECT_EQ(scene.MaterialCount(), 1u);
    EXPECT_EQ(scene.CameraCount(), 1u);
    EXPECT_EQ(scene.LightCount(), 1u);
    EXPECT_EQ(scene.ActuatorCount(), 1u);
    EXPECT_EQ(scene.SensorCount(), 1u);

    EXPECT_EQ(scene.GetMaterial(0).name, "mat_robot");
    EXPECT_FLOAT_EQ(scene.GetMaterial(0).base_color.x, 0.8f);
    EXPECT_EQ(scene.GetCamera(0).name, "track");
    EXPECT_EQ(scene.GetLight(0).type, nuka::scene::LightType::Directional);
    EXPECT_EQ(scene.GetActuator(0).joint_id, 0u);
    EXPECT_EQ(scene.GetSensor(0).type, nuka::scene::SensorType::Imu);
}

TEST(MjcfImporter, ThrowsOnMissingFile) {
    EXPECT_THROW(nuka::import::LoadMjcf("nonexistent.xml"), std::runtime_error);
}
