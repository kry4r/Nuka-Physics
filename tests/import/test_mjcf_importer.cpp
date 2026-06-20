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
    // <framepos> maps to the FramePose kind; mount resolves to link1's body row.
    EXPECT_EQ(scene.GetSensor(0).type, nuka::scene::SensorType::FramePose);
    EXPECT_EQ(scene.GetSensor(0).mount, nuka::scene::MountFrame::Body);
    EXPECT_EQ(scene.GetSensor(0).mount_index, 1u);
}

TEST(MjcfImporter, ThrowsOnMissingFile) {
    EXPECT_THROW(nuka::import::LoadMjcf("nonexistent.xml"), std::runtime_error);
}

TEST(MjcfImporter, ParsesSensorSuiteIntoUnifiedDescs) {
    using nuka::scene::SensorType;
    using nuka::scene::MountFrame;
    const auto scene = nuka::import::LoadMjcf("tests/data/sensors_suite.xml");

    ASSERT_EQ(scene.SensorCount(), 4u);
    EXPECT_EQ(scene.CameraCount(), 1u);
    EXPECT_EQ(scene.GetCamera(0).name, "onboard");
    EXPECT_FLOAT_EQ(scene.GetCamera(0).vertical_fov_degrees, 60.0f);

    // rangefinder -> RangeScan, a single forward ray, mounted on the site's body
    // with the site's local offset.
    const auto& ray = scene.GetSensor(0);
    EXPECT_EQ(ray.type, SensorType::RangeScan);
    EXPECT_EQ(ray.mount, MountFrame::Body);
    EXPECT_EQ(ray.mount_index, 0u);
    EXPECT_EQ(ray.lidar.az_count, 1u);
    EXPECT_EQ(ray.lidar.el_count, 1u);
    EXPECT_FLOAT_EQ(ray.local_offset.position.z, 0.2f);

    EXPECT_EQ(scene.GetSensor(1).type, SensorType::Imu);          // accelerometer
    EXPECT_FLOAT_EQ(scene.GetSensor(1).local_offset.position.z, 0.2f);
    EXPECT_EQ(scene.GetSensor(2).type, SensorType::Contact);      // touch
    EXPECT_FLOAT_EQ(scene.GetSensor(2).local_offset.position.z, -0.1f);
    EXPECT_EQ(scene.GetSensor(3).type, SensorType::FramePose);    // framepos
}

TEST(MjcfImporter, ThrowsOnUnsupportedSensor) {
    // An unmapped <sensor> element errors loudly rather than silently dropping.
    EXPECT_THROW(nuka::import::LoadMjcf("tests/data/sensor_unsupported.xml"),
                 std::runtime_error);
}
