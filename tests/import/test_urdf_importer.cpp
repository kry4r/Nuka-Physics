// ---------------------------------------------------------------------------
// Tests for nuka::import::LoadUrdf
// ---------------------------------------------------------------------------

#include "import/urdf_importer.hpp"
#include <gtest/gtest.h>
#include "scene/format/nks.hpp"
#include "scene/cooker.hpp"
#include <filesystem>
#include <fstream>
#include <chrono>
#include <cmath>

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


namespace {
class UrdfStrict : public ::testing::Test {
protected:
    std::filesystem::path directory;
    void SetUp() override {
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        for (unsigned int attempt = 0; attempt < 100u; ++attempt) {
            const auto candidate = std::filesystem::temp_directory_path() /
                ("nuka-urdf-" + std::to_string(stamp) + "-" + std::to_string(attempt));
            if (std::filesystem::create_directory(candidate)) {
                directory = candidate;
                return;
            }
        }
        FAIL() << "Could not allocate a private URDF fixture directory";
    }
    void TearDown() override {
        if (!directory.empty()) std::filesystem::remove_all(directory);
    }
    std::string Write(const std::string& xml) {
        const auto path = directory / "source.urdf";
        std::ofstream(path) << xml;
        return path.string();
    }
    std::string Source() {
        std::ifstream file("tests/data/minimal_arm.urdf");
        return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
    }
    static void Replace(std::string& text, const std::string& before, const std::string& after) {
        const auto position = text.find(before);
        ASSERT_NE(position, std::string::npos);
        text.replace(position, before.size(), after);
    }
};
}

TEST_F(UrdfStrict, DefaultAxisAndProductionRoundtrip) {
    auto xml = Source();
    Replace(xml, "<axis xyz=\"0 0 1\"/>", "<!-- no authored axis -->");
    Replace(xml, "<robot name=", "<robot xmlns:note=\"urn:example:metadata\" name=");
    const auto path = Write(xml);
    auto result = nuka::import::ImportUrdf(path);
    ASSERT_TRUE(result.StrictSuccess());
    EXPECT_FLOAT_EQ(result.scene->GetJoint(0).axis.x, 1.0f);
    EXPECT_FLOAT_EQ(result.scene->GetJoint(0).axis.z, 0.0f);
    EXPECT_FLOAT_EQ(nuka::import::LoadUrdf(path).GetJoint(0).axis.x, 1.0f);
    const auto output = (directory / "scene.nks").string();
    nuka::scene::nks::Save(*result.scene, output);
    const auto reloaded = nuka::scene::nks::Load(output);
    EXPECT_EQ(reloaded.RigidBodyCount(), 2u);
    EXPECT_EQ(reloaded.ShapeCount(), 1u);
    EXPECT_FLOAT_EQ(reloaded.GetJoint(0).axis.x, 1.0f);
    const auto cooked = nuka::scene::CookScene(reloaded);
    ASSERT_EQ(cooked.joint_count, 1u);
    EXPECT_FLOAT_EQ(cooked.joints.axes[0].x, 1.0f);
    EXPECT_FLOAT_EQ(cooked.joints.lower_limits[0], -3.14f);
    EXPECT_FLOAT_EQ(cooked.joints.upper_limits[0], 3.14f);
    EXPECT_EQ(cooked.joints.limit_flags[0], 3u);
    EXPECT_FLOAT_EQ(cooked.bodies.masses[0], 1.0f);
    EXPECT_FLOAT_EQ(cooked.bodies.masses[1], 0.5f);
    EXPECT_FLOAT_EQ(cooked.bodies.inertias[0].x, 0.1f);
    for (float value : cooked.bodies.inv_masses) EXPECT_TRUE(std::isfinite(value));
    for (const auto& value : cooked.bodies.inv_inertias) {
        EXPECT_TRUE(std::isfinite(value.x));
        EXPECT_TRUE(std::isfinite(value.y));
        EXPECT_TRUE(std::isfinite(value.z));
    }
}

TEST_F(UrdfStrict, RejectsLossAndInvalidInputAtSourceLine) {
    struct Case { const char* before; const char* after; const char* code; };
    const Case cases[] = {
        {"type=\"revolute\"", "type=\"planar\"", "URDF_JOINT_UNSUPPORTED"},
        {"type=\"revolute\"", "type=\"fixed revolute\"", "URDF_JOINT_UNSUPPORTED"},
        {"lower=\"-3.14\"", "", "URDF_LIMIT_DEFAULT_UNREPRESENTED"},
        {"ixy=\"0\"", "ixy=\"0.02\"", "URDF_FULL_INERTIA_UNSUPPORTED"},
        {"<box size=\"0.2 0.2 0.2\"/>", "<cylinder radius=\"0.2\" length=\"0.2\"/>", "URDF_ELEMENT_UNSUPPORTED"},
        {"<axis xyz=\"0 0 1\"/>", "<axis xyz=\"0 0 0\"/>", "URDF_AXIS_INVALID"},
        {"<axis xyz=\"0 0 1\"/>", "<axis xyz=\"nan 0 1\"/>", "URDF_NUMBER_INVALID"},
        {"<axis xyz=\"0 0 1\"/>", "<axis xyz=\"2 0 0\"/>", "URDF_AXIS_INVALID"},
        {"<axis xyz=\"0 0 1\"/>", "<axis xyz=\"0 0 1 2e\"/>", "URDF_NUMBER_INVALID"},
        {"value=\"1.0\"", "value=\"1 2e\"", "URDF_NUMBER_INVALID"},
        {"value=\"1.0\"", "value=\"1e-40\"", "URDF_INVERSE_NONFINITE"},
        {"izz=\"0.1\"", "izz=\"3\"", "URDF_INERTIA_INVALID"},
        {"size=\"0.2 0.2 0.2\"", "size=\"1.40129846e-45 0.2 0.2\"", "URDF_DIMENSION_UNDERFLOW"},
        {"<parent link=\"base\"/>", "<parent link=\"missing\"/>", "URDF_LINK_UNRESOLVED"},
        {"<limit lower=", "<limit velocity=\"3\" lower=", "URDF_ATTRIBUTE_UNSUPPORTED"},
        {"</robot>", "<gazebo/>\n</robot>", "URDF_ELEMENT_UNSUPPORTED"},
        {"<robot name=", "<robot xmlns=\"urn:foreign\" name=", "URDF_NAMESPACE_UNSUPPORTED"},
        {"name=\"link1\"", "name=\"base\"", "URDF_DUPLICATE_NAME"},
        {"lower=\"-3.14\"", "lower=\"4\"", "URDF_LIMIT_INVALID"},
    };
    for (const auto& test : cases) {
        SCOPED_TRACE(test.code);
        auto xml = Source();
        Replace(xml, test.before, test.after);
        const auto path = Write(xml);
        auto result = nuka::import::ImportUrdf(path);
        EXPECT_FALSE(result.scene.has_value());
        EXPECT_FALSE(result.StrictSuccess());
        bool found = false;
        for (const auto& diagnostic : result.diagnostics) {
            if (diagnostic.code == test.code) {
                found = true;
                EXPECT_GT(diagnostic.line, 0);
                EXPECT_EQ(diagnostic.source, path);
            }
        }
        EXPECT_TRUE(found);
    }
}

TEST_F(UrdfStrict, LegacyProfileReportsLossAndPreservesOutput) {
    auto xml = Source();
    Replace(xml, "type=\"revolute\"", "type=\"planar\"");
    const auto path = Write(xml);
    auto result = nuka::import::ImportUrdf(path, nuka::import::UrdfImportProfile::LegacyCompatible);
    ASSERT_TRUE(result.scene.has_value());
    ASSERT_FALSE(result.diagnostics.empty());
    EXPECT_FALSE(result.StrictSuccess());
    EXPECT_EQ(result.diagnostics[0].kind, nuka::import::UrdfDiagnosticKind::Unsupported);
    EXPECT_EQ(result.scene->GetJoint(0).type, nuka::scene::JointType::Free);
    EXPECT_EQ(nuka::import::LoadUrdf(path).GetJoint(0).type, result.scene->GetJoint(0).type);
}

TEST_F(UrdfStrict, SeparatesIoAndSyntaxFailures) {
    auto absent = nuka::import::ImportUrdf((directory / "missing.urdf").string());
    ASSERT_EQ(absent.diagnostics.size(), 1u);
    EXPECT_EQ(absent.diagnostics[0].kind, nuka::import::UrdfDiagnosticKind::Io);
    auto malformed = nuka::import::ImportUrdf(Write("<robot>"));
    ASSERT_EQ(malformed.diagnostics.size(), 1u);
    EXPECT_EQ(malformed.diagnostics[0].kind, nuka::import::UrdfDiagnosticKind::Syntax);
}

TEST_F(UrdfStrict, RejectsOverflowingMeshProjection) {
    std::ofstream(directory / "triangle.obj") << "v 2 0 0\nv 0 2 0\nv 0 0 2\nf 1 2 3\n";
    auto xml = Source();
    Replace(xml, "<box size=\"0.2 0.2 0.2\"/>",
            "<mesh filename=\"triangle.obj\" scale=\"3e38 1 1\"/>");
    auto result = nuka::import::ImportUrdf(Write(xml));
    EXPECT_FALSE(result.scene.has_value());
    ASSERT_FALSE(result.diagnostics.empty());
    EXPECT_EQ(result.diagnostics.back().code, "URDF_MESH_NONFINITE");
}
