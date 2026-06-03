// ---------------------------------------------------------------------------
// Tests for nuka:decompose attribute parsing in all three importers
// ---------------------------------------------------------------------------
// Exercises deliverable 5: LoadUsd / LoadUrdf / LoadMjcf read the decomposition
// mode + max_pieces onto the (mesh) CollisionShapeRecord. Snippets are written
// to temp files so no shared test fixture is touched.
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"
#include "import/urdf_importer.hpp"
#include "import/usd_importer.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>

namespace {

std::filesystem::path WriteTemp(const char* name, const std::string& contents) {
    const auto path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path, std::ios::trunc);
    out << contents;
    out.close();
    return path;
}

// A minimal single-triangle ASCII STL fixture. The MJCF mesh-file wiring now
// loads the geometry referenced by a `type="mesh"` geom, so a mesh geom must
// resolve to a real on-disk asset. These decompose tests only assert the
// nuka:decompose attribute parsing (which happens before geometry load), so any
// valid mesh suffices.
std::filesystem::path WriteFingerStl(const char* name) {
    const std::string stl =
        "solid finger\n"
        "  facet normal 0 0 1\n"
        "    outer loop\n"
        "      vertex 0 0 0\n"
        "      vertex 1 0 0\n"
        "      vertex 0 1 0\n"
        "    endloop\n"
        "  endfacet\n"
        "endsolid finger\n";
    return WriteTemp(name, stl);
}

// --- USD --------------------------------------------------------------------
// A Mesh collision shape carrying the custom nuka:decompose token + max_pieces.
TEST(DecomposeAttribute, UsdParsesForceAndMaxPieces) {
    const std::string usd = R"USD(#usda 1.0
def Xform "robot"
{
    def Xform "link1" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        bool physics:rigidBodyEnabled = true
        float physics:mass = 1.0
        def Mesh "finger" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
            custom token nuka:decompose = "force"
            custom int nuka:decompose:max_pieces = 12
        }
    }
}
)USD";
    const auto path = WriteTemp("nuka_decompose_test.usda", usd);
    const auto scene = nuka::import::LoadUsd(path.string());

    ASSERT_EQ(scene.ShapeCount(), 1u);
    const auto& shape = scene.GetShape(0);
    EXPECT_EQ(shape.type, nuka::scene::ShapeType::TriMesh);
    EXPECT_EQ(shape.decompose_mode, nuka::scene::DecomposeMode::Force);
    EXPECT_EQ(shape.decompose_max_pieces, 12u);

    std::filesystem::remove(path);
}

TEST(DecomposeAttribute, UsdDefaultsToAutoWhenAbsent) {
    const std::string usd = R"USD(#usda 1.0
def Xform "robot"
{
    def Xform "link1" (
        prepend apiSchemas = ["PhysicsRigidBodyAPI"]
    )
    {
        bool physics:rigidBodyEnabled = true
        float physics:mass = 1.0
        def Mesh "finger" (
            prepend apiSchemas = ["PhysicsCollisionAPI"]
        )
        {
        }
    }
}
)USD";
    const auto path = WriteTemp("nuka_decompose_default.usda", usd);
    const auto scene = nuka::import::LoadUsd(path.string());

    ASSERT_EQ(scene.ShapeCount(), 1u);
    EXPECT_EQ(scene.GetShape(0).decompose_mode, nuka::scene::DecomposeMode::Auto);

    std::filesystem::remove(path);
}

// --- URDF -------------------------------------------------------------------
// NOTE on the attribute form: the spec example is
// `<nuka:decompose max_pieces="8"/>` with no mode attribute. We additionally
// accept an explicit `decompose="auto|force|skip"` attribute; when absent (the
// spec's form), the mode defaults to Auto and max_pieces is still honored.
TEST(DecomposeAttribute, UrdfParsesMaxPiecesSpecForm) {
    const std::string urdf = R"URDF(<?xml version="1.0"?>
<robot name="r">
  <link name="base">
    <collision>
      <geometry><mesh filename="finger.obj"/></geometry>
      <nuka:decompose max_pieces="8"/>
    </collision>
  </link>
</robot>
)URDF";
    const auto path = WriteTemp("nuka_decompose_test.urdf", urdf);
    const auto scene = nuka::import::LoadUrdf(path.string());

    ASSERT_EQ(scene.ShapeCount(), 1u);
    const auto& shape = scene.GetShape(0);
    EXPECT_EQ(shape.type, nuka::scene::ShapeType::TriMesh);
    EXPECT_EQ(shape.decompose_mode, nuka::scene::DecomposeMode::Auto);  // no mode attr
    EXPECT_EQ(shape.decompose_max_pieces, 8u);

    std::filesystem::remove(path);
}

TEST(DecomposeAttribute, UrdfParsesExplicitModeAttribute) {
    const std::string urdf = R"URDF(<?xml version="1.0"?>
<robot name="r">
  <link name="base">
    <collision>
      <geometry><mesh filename="finger.obj"/></geometry>
      <nuka:decompose decompose="skip" max_pieces="4"/>
    </collision>
  </link>
</robot>
)URDF";
    const auto path = WriteTemp("nuka_decompose_skip.urdf", urdf);
    const auto scene = nuka::import::LoadUrdf(path.string());

    ASSERT_EQ(scene.ShapeCount(), 1u);
    const auto& shape = scene.GetShape(0);
    EXPECT_EQ(shape.decompose_mode, nuka::scene::DecomposeMode::Skip);
    EXPECT_EQ(shape.decompose_max_pieces, 4u);

    std::filesystem::remove(path);
}

// --- MJCF -------------------------------------------------------------------
TEST(DecomposeAttribute, MjcfParsesAutoAndMaxPieces) {
    WriteFingerStl("finger.stl");
    const std::string mjcf = R"MJCF(<mujoco>
  <compiler meshdir="."/>
  <asset>
    <mesh name="finger" file="finger.stl"/>
  </asset>
  <worldbody>
    <body name="finger">
      <geom name="fingergeom" type="mesh" mesh="finger"
            nuka:decompose="force" nuka:decompose:max_pieces="6"/>
    </body>
  </worldbody>
</mujoco>
)MJCF";
    const auto path = WriteTemp("nuka_decompose_test.xml", mjcf);
    const auto scene = nuka::import::LoadMjcf(path.string());

    ASSERT_EQ(scene.ShapeCount(), 1u);
    const auto& shape = scene.GetShape(0);
    EXPECT_EQ(shape.type, nuka::scene::ShapeType::TriMesh);
    EXPECT_EQ(shape.decompose_mode, nuka::scene::DecomposeMode::Force);
    EXPECT_EQ(shape.decompose_max_pieces, 6u);

    std::filesystem::remove(path);
}

TEST(DecomposeAttribute, MjcfDefaultsToAutoWhenAbsent) {
    WriteFingerStl("finger.stl");
    const std::string mjcf = R"MJCF(<mujoco>
  <compiler meshdir="."/>
  <asset>
    <mesh name="finger" file="finger.stl"/>
  </asset>
  <worldbody>
    <body name="finger">
      <geom name="fingergeom" type="mesh" mesh="finger"/>
    </body>
  </worldbody>
</mujoco>
)MJCF";
    const auto path = WriteTemp("nuka_decompose_default.xml", mjcf);
    const auto scene = nuka::import::LoadMjcf(path.string());

    ASSERT_EQ(scene.ShapeCount(), 1u);
    EXPECT_EQ(scene.GetShape(0).decompose_mode, nuka::scene::DecomposeMode::Auto);

    std::filesystem::remove(path);
}

} // namespace
