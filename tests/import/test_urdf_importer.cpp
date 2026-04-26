// ---------------------------------------------------------------------------
// Tests for nuka::import::LoadUrdf
// ---------------------------------------------------------------------------

#include "import/urdf_importer.hpp"
#include <gtest/gtest.h>

TEST(UrdfImporter, LoadsMinimalBodyAndJoint) {
    const auto scene = nuka::import::LoadUrdf("tests/data/minimal_arm.urdf");
    EXPECT_EQ(scene.RigidBodyCount(), 2u);
    EXPECT_EQ(scene.JointCount(), 1u);
}

TEST(UrdfImporter, LinkNamesAreCorrect) {
    const auto scene = nuka::import::LoadUrdf("tests/data/minimal_arm.urdf");
    ASSERT_GE(scene.RigidBodyCount(), 2u);
    EXPECT_EQ(scene.GetBody(0).name, "base");
    EXPECT_EQ(scene.GetBody(1).name, "link1");
}

TEST(UrdfImporter, MassValuesAreParsed) {
    const auto scene = nuka::import::LoadUrdf("tests/data/minimal_arm.urdf");
    ASSERT_GE(scene.RigidBodyCount(), 2u);
    EXPECT_FLOAT_EQ(scene.GetBody(0).mass, 1.0f);
    EXPECT_FLOAT_EQ(scene.GetBody(1).mass, 0.5f);
}

TEST(UrdfImporter, JointAxisIsParsed) {
    const auto scene = nuka::import::LoadUrdf("tests/data/minimal_arm.urdf");
    ASSERT_EQ(scene.JointCount(), 1u);
    const auto& j = scene.GetJoint(0);
    EXPECT_FLOAT_EQ(j.axis.x, 0.0f);
    EXPECT_FLOAT_EQ(j.axis.y, 0.0f);
    EXPECT_FLOAT_EQ(j.axis.z, 1.0f);
}

TEST(UrdfImporter, JointLimitsAreParsed) {
    const auto scene = nuka::import::LoadUrdf("tests/data/minimal_arm.urdf");
    ASSERT_EQ(scene.JointCount(), 1u);
    const auto& j = scene.GetJoint(0);
    EXPECT_FLOAT_EQ(j.lower_limit, -3.14f);
    EXPECT_FLOAT_EQ(j.upper_limit,  3.14f);
}

TEST(UrdfImporter, JointTypeIsParsed) {
    const auto scene = nuka::import::LoadUrdf("tests/data/minimal_arm.urdf");
    ASSERT_EQ(scene.JointCount(), 1u);
    EXPECT_EQ(scene.GetJoint(0).type, nuka::scene::JointType::Revolute);
}

TEST(UrdfImporter, JointNameIsParsed) {
    const auto scene = nuka::import::LoadUrdf("tests/data/minimal_arm.urdf");
    ASSERT_EQ(scene.JointCount(), 1u);
    EXPECT_EQ(scene.GetJoint(0).name, "joint0");
}

TEST(UrdfImporter, CollisionShapeIsParsed) {
    const auto scene = nuka::import::LoadUrdf("tests/data/minimal_arm.urdf");
    // Only "base" has a <collision> element
    EXPECT_EQ(scene.ShapeCount(), 1u);
    EXPECT_EQ(scene.GetShape(0).type, nuka::scene::ShapeType::Box);
}

TEST(UrdfImporter, ThrowsOnMissingFile) {
    EXPECT_THROW(nuka::import::LoadUrdf("nonexistent.urdf"), std::runtime_error);
}
