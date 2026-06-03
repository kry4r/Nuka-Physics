// ---------------------------------------------------------------------------
// Tests for nuka::import::LoadUsd MESH GEOMETRY support (#32):
//   * inline `def Mesh` points + faceVertexIndices + faceVertexCounts (fan)
//   * multi-line array values (tokens split across physical lines)
//   * `references = @file.usda@</Prim>` graft + xformOp:scale bake
//   * physics-less cup-like Xform->Mesh (no PhysicsRigidBodyAPI) still loads
//   * determinism (two parses byte-identical)
//
// Fixtures are authored at runtime into a process-unique temp directory so the
// suite stays hermetic (no reliance on a checked-in / owner-provided cup file).
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

namespace fs = std::filesystem;

// A unique temp directory for this test process.
fs::path TempDir() {
    static const fs::path dir = [] {
        fs::path d = fs::temp_directory_path() /
                     ("nuka_usd_mesh_test_" + std::to_string(::getpid()));
        fs::create_directories(d);
        return d;
    }();
    return dir;
}

void WriteText(const fs::path& p, const std::string& text) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

// Return the first shape that carries mesh geometry, or nullptr.
const nuka::scene::CollisionShapeRecord* FirstMeshShape(const nuka::scene::SceneIR& scene) {
    for (const auto& shape : scene.Shapes()) {
        if (!shape.mesh_vertices.empty()) {
            return &shape;
        }
    }
    return nullptr;
}

} // namespace

// ---------------------------------------------------------------------------
// Test 1: inline def Mesh with a quad face (count 4 -> 2 fan tris).
// ---------------------------------------------------------------------------
TEST(UsdMeshGeometry, InlineMeshQuadFanTriangulation) {
    // 4 verts: a unit quad in z=0; one quad face (count 4).
    const std::string usda =
        "#usda 1.0\n"
        "def Xform \"root\"\n"
        "{\n"
        "    def Mesh \"quad\"\n"
        "    {\n"
        "        point3f[] points = [ (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0) ]\n"
        "        int[] faceVertexIndices = [ 0, 1, 2, 3 ]\n"
        "        int[] faceVertexCounts = [ 4 ]\n"
        "    }\n"
        "}\n";
    const fs::path p = TempDir() / "inline_quad.usda";
    WriteText(p, usda);

    const auto scene = nuka::import::LoadUsd(p.string());
    const auto* shape = FirstMeshShape(scene);
    ASSERT_NE(shape, nullptr);
    EXPECT_EQ(shape->type, nuka::scene::ShapeType::TriMesh);

    // 4 verts -> 12 flat floats, file order, no dedup.
    ASSERT_EQ(shape->mesh_vertices.size(), 12u);
    const float expected_v[12] = {0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0};
    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(shape->mesh_vertices[static_cast<size_t>(i)], expected_v[i]) << "i=" << i;
    }

    // quad (count 4) -> fan: (0,1,2) and (0,2,3) => 6 indices.
    ASSERT_EQ(shape->mesh_indices.size(), 6u);
    const uint32_t expected_i[6] = {0, 1, 2, 0, 2, 3};
    for (int i = 0; i < 6; ++i) {
        EXPECT_EQ(shape->mesh_indices[static_cast<size_t>(i)], expected_i[i]) << "i=" << i;
    }
}

// ---------------------------------------------------------------------------
// Test 2: the SAME data, but arrays split across many physical lines.
// ---------------------------------------------------------------------------
TEST(UsdMeshGeometry, MultiLineArraysMatchSingleLine) {
    const std::string single =
        "#usda 1.0\n"
        "def Xform \"root\"\n"
        "{\n"
        "    def Mesh \"quad\"\n"
        "    {\n"
        "        point3f[] points = [ (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0) ]\n"
        "        int[] faceVertexIndices = [ 0, 1, 2, 3 ]\n"
        "        int[] faceVertexCounts = [ 4 ]\n"
        "    }\n"
        "}\n";
    const std::string multi =
        "#usda 1.0\n"
        "def Xform \"root\"\n"
        "{\n"
        "    def Mesh \"quad\"\n"
        "    {\n"
        "        point3f[] points = [\n"
        "            (0, 0, 0),\n"
        "            (1, 0, 0),\n"
        "            (1, 1, 0),\n"
        "            (0, 1, 0)\n"
        "        ]\n"
        "        int[] faceVertexIndices = [\n"
        "            0, 1,\n"
        "            2, 3\n"
        "        ]\n"
        "        int[] faceVertexCounts = [\n"
        "            4\n"
        "        ]\n"
        "    }\n"
        "}\n";
    const fs::path sp = TempDir() / "single_line.usda";
    const fs::path mp = TempDir() / "multi_line.usda";
    WriteText(sp, single);
    WriteText(mp, multi);

    const auto scene_s = nuka::import::LoadUsd(sp.string());
    const auto scene_m = nuka::import::LoadUsd(mp.string());
    const auto* shape_s = FirstMeshShape(scene_s);
    const auto* shape_m = FirstMeshShape(scene_m);
    ASSERT_NE(shape_s, nullptr);
    ASSERT_NE(shape_m, nullptr);

    EXPECT_EQ(shape_s->mesh_vertices, shape_m->mesh_vertices);
    EXPECT_EQ(shape_s->mesh_indices, shape_m->mesh_indices);

    // Guard against an empty-and-equal false pass: the multi-line reader must
    // actually populate the same non-trivial geometry the single-line one does
    // (4 verts -> 12 floats; one quad -> 6 fan indices).
    ASSERT_EQ(shape_m->mesh_vertices.size(), 12u);
    ASSERT_EQ(shape_m->mesh_indices.size(), 6u);
}

// ---------------------------------------------------------------------------
// Test 3: references graft + xformOp:scale bake.
// ---------------------------------------------------------------------------
TEST(UsdMeshGeometry, ReferencesGraftWithScaleBake) {
    // mesh.usda: a unit triangle under prim path /Model.
    const std::string mesh =
        "#usda 1.0\n"
        "def Xform \"Model\"\n"
        "{\n"
        "    def Mesh \"tri\"\n"
        "    {\n"
        "        point3f[] points = [ (0, 0, 0), (1, 0, 0), (0, 1, 0) ]\n"
        "        int[] faceVertexIndices = [ 0, 1, 2 ]\n"
        "        int[] faceVertexCounts = [ 3 ]\n"
        "    }\n"
        "}\n";
    // wrapper.usda: references the Model prim with a per-axis scale of (2,3,4).
    const std::string wrapper =
        "#usda 1.0\n"
        "def Xform \"root\"\n"
        "{\n"
        "    def Xform \"Model\" (\n"
        "        prepend references = @./mesh.usda@</Model>\n"
        "    )\n"
        "    {\n"
        "        float3 xformOp:translate = (0, 0, 0)\n"
        "        float3 xformOp:scale = (2, 3, 4)\n"
        "        uniform token[] xformOpOrder = [\"xformOp:translate\", \"xformOp:scale\"]\n"
        "    }\n"
        "}\n";
    const fs::path dir = TempDir() / "refs";
    fs::create_directories(dir);
    WriteText(dir / "mesh.usda", mesh);
    WriteText(dir / "wrapper.usda", wrapper);

    const auto scene = nuka::import::LoadUsd((dir / "wrapper.usda").string());
    const auto* shape = FirstMeshShape(scene);
    ASSERT_NE(shape, nullptr);

    // Triangle (0,0,0),(1,0,0),(0,1,0) scaled by (2,3,4):
    //   v0 = (0,0,0), v1 = (2,0,0), v2 = (0,3,0).
    ASSERT_EQ(shape->mesh_vertices.size(), 9u);
    const float expected[9] = {0, 0, 0, 2, 0, 0, 0, 3, 0};
    for (int i = 0; i < 9; ++i) {
        EXPECT_FLOAT_EQ(shape->mesh_vertices[static_cast<size_t>(i)], expected[i]) << "i=" << i;
    }
    ASSERT_EQ(shape->mesh_indices.size(), 3u);
    EXPECT_EQ(shape->mesh_indices[0], 0u);
    EXPECT_EQ(shape->mesh_indices[1], 1u);
    EXPECT_EQ(shape->mesh_indices[2], 2u);
}

// ---------------------------------------------------------------------------
// Test 4: physics-less cup-like Xform->Mesh (no PhysicsRigidBodyAPI) must load.
// ---------------------------------------------------------------------------
TEST(UsdMeshGeometry, PhysicsLessMeshDoesNotThrowAndHasGeometry) {
    const std::string usda =
        "#usda 1.0\n"
        "def Xform \"cup\"\n"
        "{\n"
        "    def Mesh \"cup_mesh\"\n"
        "    {\n"
        "        point3f[] points = [ (0, 0, 0), (1, 0, 0), (0, 1, 0) ]\n"
        "        int[] faceVertexIndices = [ 0, 1, 2 ]\n"
        "        int[] faceVertexCounts = [ 3 ]\n"
        "    }\n"
        "}\n";
    const fs::path p = TempDir() / "physicsless_cup.usda";
    WriteText(p, usda);

    // Must NOT throw even though there is no PhysicsRigidBodyAPI anywhere.
    nuka::scene::SceneIR scene;
    ASSERT_NO_THROW(scene = nuka::import::LoadUsd(p.string()));

    const auto* shape = FirstMeshShape(scene);
    ASSERT_NE(shape, nullptr);
    EXPECT_EQ(shape->type, nuka::scene::ShapeType::TriMesh);
    ASSERT_EQ(shape->mesh_vertices.size(), 9u);
    ASSERT_EQ(shape->mesh_indices.size(), 3u);
    // An implicit body must back the shape so the cooker has a target.
    EXPECT_NE(shape->body_id, nuka::scene::kInvalidBody);
    EXPECT_GE(scene.RigidBodyCount(), 1u);
}

// ---------------------------------------------------------------------------
// Test 5: determinism — two parses of the same fixture are byte-identical.
// ---------------------------------------------------------------------------
TEST(UsdMeshGeometry, DeterministicAcrossLoads) {
    const std::string usda =
        "#usda 1.0\n"
        "def Xform \"root\"\n"
        "{\n"
        "    def Mesh \"quad\"\n"
        "    {\n"
        "        point3f[] points = [ (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (2, 2, 2) ]\n"
        "        int[] faceVertexIndices = [ 0, 1, 2, 3, 0, 2, 4 ]\n"
        "        int[] faceVertexCounts = [ 4, 3 ]\n"
        "    }\n"
        "}\n";
    const fs::path p = TempDir() / "determinism.usda";
    WriteText(p, usda);

    const auto a = nuka::import::LoadUsd(p.string());
    const auto b = nuka::import::LoadUsd(p.string());
    const auto* sa = FirstMeshShape(a);
    const auto* sb = FirstMeshShape(b);
    ASSERT_NE(sa, nullptr);
    ASSERT_NE(sb, nullptr);
    EXPECT_EQ(sa->mesh_vertices, sb->mesh_vertices);
    EXPECT_EQ(sa->mesh_indices, sb->mesh_indices);
    // quad(4)->2 tris + tri(3)->1 tri = 3 tris => 9 indices.
    EXPECT_EQ(sa->mesh_indices.size(), 9u);
}
