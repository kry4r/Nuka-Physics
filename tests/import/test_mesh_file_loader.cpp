// ---------------------------------------------------------------------------
// Tests for nuka::import mesh-file loader (STL / OBJ) + MJCF mesh wiring.
//
// Fixtures are authored at runtime into a process-unique temp directory so the
// suite stays hermetic (no reliance on a checked-in data file / network).
// ---------------------------------------------------------------------------

#include "import/mesh_file_loader.hpp"
#include "import/mjcf_importer.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
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
                     ("nuka_mesh_loader_test_" + std::to_string(::getpid()));
        fs::create_directories(d);
        return d;
    }();
    return dir;
}

void WriteBytes(const fs::path& p, const std::vector<char>& bytes) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

void WriteText(const fs::path& p, const std::string& text) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

void PushF32(std::vector<char>& buf, float v) {
    char b[4];
    std::memcpy(b, &v, 4);
    buf.insert(buf.end(), b, b + 4);
}

void PushU32(std::vector<char>& buf, std::uint32_t v) {
    char b[4];
    std::memcpy(b, &v, 4);
    buf.insert(buf.end(), b, b + 4);
}

// Two triangles forming a unit quad in the z=0 plane:
//   tri 0: (0,0,0) (1,0,0) (1,1,0)
//   tri 1: (0,0,0) (1,1,0) (0,1,0)
const float kTri[2][3][3] = {
    {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {1.f, 1.f, 0.f}},
    {{0.f, 0.f, 0.f}, {1.f, 1.f, 0.f}, {0.f, 1.f, 0.f}},
};

// Build a valid binary STL (2 triangles) into a byte buffer.
std::vector<char> MakeBinaryStl() {
    std::vector<char> buf;
    buf.resize(80, 0);                  // 80-byte header (zeroed)
    PushU32(buf, 2u);                   // triangle count
    for (int t = 0; t < 2; ++t) {
        PushF32(buf, 0.f);              // normal x
        PushF32(buf, 0.f);              // normal y
        PushF32(buf, 1.f);              // normal z
        for (int v = 0; v < 3; ++v) {
            PushF32(buf, kTri[t][v][0]);
            PushF32(buf, kTri[t][v][1]);
            PushF32(buf, kTri[t][v][2]);
        }
        buf.push_back(0);               // attribute byte count (2 bytes)
        buf.push_back(0);
    }
    return buf;
}

std::string MakeAsciiStl() {
    std::string s = "solid test\n";
    for (int t = 0; t < 2; ++t) {
        s += "  facet normal 0 0 1\n";
        s += "    outer loop\n";
        for (int v = 0; v < 3; ++v) {
            s += "      vertex " + std::to_string(kTri[t][v][0]) + " " +
                 std::to_string(kTri[t][v][1]) + " " +
                 std::to_string(kTri[t][v][2]) + "\n";
        }
        s += "    endloop\n";
        s += "  endfacet\n";
    }
    s += "endsolid test\n";
    return s;
}

void ExpectQuadGeometry(const nuka::import::MeshGeometry& g) {
    ASSERT_EQ(g.VertexCount(), 6u);   // 2 tris * 3 verts
    ASSERT_EQ(g.TriangleCount(), 2u);
    ASSERT_EQ(g.indices.size(), 6u);
    // Vertices in file order, no dedup.
    for (int t = 0; t < 2; ++t) {
        for (int v = 0; v < 3; ++v) {
            const std::size_t base = static_cast<std::size_t>(t * 3 + v) * 3;
            EXPECT_FLOAT_EQ(g.vertices[base + 0], kTri[t][v][0]);
            EXPECT_FLOAT_EQ(g.vertices[base + 1], kTri[t][v][1]);
            EXPECT_FLOAT_EQ(g.vertices[base + 2], kTri[t][v][2]);
        }
    }
    // Index list is sequential face order.
    for (std::uint32_t i = 0; i < 6; ++i) {
        EXPECT_EQ(g.indices[i], i);
    }
}

} // namespace

TEST(MeshFileLoader, BinaryStl) {
    const fs::path p = TempDir() / "binary.stl";
    WriteBytes(p, MakeBinaryStl());
    const auto g = nuka::import::LoadStl(p.string());
    ExpectQuadGeometry(g);
}

TEST(MeshFileLoader, AsciiStl) {
    const fs::path p = TempDir() / "ascii.stl";
    WriteText(p, MakeAsciiStl());
    const auto g = nuka::import::LoadStl(p.string());
    ExpectQuadGeometry(g);
}

TEST(MeshFileLoader, AsciiAndBinaryStlMatch) {
    const fs::path bp = TempDir() / "match_bin.stl";
    const fs::path ap = TempDir() / "match_ascii.stl";
    WriteBytes(bp, MakeBinaryStl());
    WriteText(ap, MakeAsciiStl());
    const auto gb = nuka::import::LoadStl(bp.string());
    const auto ga = nuka::import::LoadStl(ap.string());
    EXPECT_EQ(gb.vertices, ga.vertices);
    EXPECT_EQ(gb.indices, ga.indices);
}

TEST(MeshFileLoader, ObjOneBasedFanTriangulation) {
    // 4 vertices, one tri face and one quad face. The quad fan-triangulates
    // into 2 triangles, so total = 3 tris => 9 indices.
    const std::string obj =
        "# comment\n"
        "o quad_and_tri\n"
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "v 0 1 0\n"
        "vt 0 0\n"
        "vn 0 0 1\n"
        "f 1 2 3\n"            // triangle (1-based)
        "f 1/1/1 2/1/1 3/1/1 4/1/1\n"; // quad with slash form -> fan
    const fs::path p = TempDir() / "shape.obj";
    WriteText(p, obj);
    const auto g = nuka::import::LoadObj(p.string());

    ASSERT_EQ(g.VertexCount(), 4u);     // 4 'v' lines, no dedup
    ASSERT_EQ(g.TriangleCount(), 3u);   // tri + (quad -> 2)
    ASSERT_EQ(g.indices.size(), 9u);

    // First face: 1-based 1,2,3 -> 0-based 0,1,2.
    EXPECT_EQ(g.indices[0], 0u);
    EXPECT_EQ(g.indices[1], 1u);
    EXPECT_EQ(g.indices[2], 2u);
    // Quad 1,2,3,4 -> fan (0,1,2) and (0,2,3).
    EXPECT_EQ(g.indices[3], 0u);
    EXPECT_EQ(g.indices[4], 1u);
    EXPECT_EQ(g.indices[5], 2u);
    EXPECT_EQ(g.indices[6], 0u);
    EXPECT_EQ(g.indices[7], 2u);
    EXPECT_EQ(g.indices[8], 3u);

    // Vertex coordinates preserved in order.
    EXPECT_FLOAT_EQ(g.vertices[0], 0.f);
    EXPECT_FLOAT_EQ(g.vertices[3], 1.f);   // v2.x
    EXPECT_FLOAT_EQ(g.vertices[9], 0.f);   // v4.x
    EXPECT_FLOAT_EQ(g.vertices[10], 1.f);  // v4.y
}

TEST(MeshFileLoader, ObjNegativeRelativeIndices) {
    // Negative indices are relative to the current vertex count.
    const std::string obj =
        "v 0 0 0\n"
        "v 1 0 0\n"
        "v 1 1 0\n"
        "f -3 -2 -1\n";  // -> 0,1,2
    const fs::path p = TempDir() / "neg.obj";
    WriteText(p, obj);
    const auto g = nuka::import::LoadObj(p.string());
    ASSERT_EQ(g.indices.size(), 3u);
    EXPECT_EQ(g.indices[0], 0u);
    EXPECT_EQ(g.indices[1], 1u);
    EXPECT_EQ(g.indices[2], 2u);
}

TEST(MeshFileLoader, DeterministicAcrossLoads) {
    const fs::path p = TempDir() / "determinism.stl";
    WriteBytes(p, MakeBinaryStl());
    const auto a = nuka::import::LoadStl(p.string());
    const auto b = nuka::import::LoadStl(p.string());
    EXPECT_EQ(a.vertices, b.vertices);
    EXPECT_EQ(a.indices, b.indices);
}

TEST(MeshFileLoader, DispatchByExtension) {
    const fs::path sp = TempDir() / "dispatch.stl";
    const fs::path op = TempDir() / "dispatch.obj";
    WriteBytes(sp, MakeBinaryStl());
    WriteText(op, "v 0 0 0\nv 1 0 0\nv 1 1 0\nf 1 2 3\n");
    EXPECT_EQ(nuka::import::LoadMeshFile(sp.string()).TriangleCount(), 2u);
    EXPECT_EQ(nuka::import::LoadMeshFile(op.string()).TriangleCount(), 1u);
    // Case-insensitive extension.
    const fs::path up = TempDir() / "upper.STL";
    WriteBytes(up, MakeBinaryStl());
    EXPECT_EQ(nuka::import::LoadMeshFile(up.string()).TriangleCount(), 2u);
}

TEST(MeshFileLoader, ThrowsOnMissingFile) {
    EXPECT_THROW(nuka::import::LoadMeshFile((TempDir() / "does_not_exist.stl").string()),
                 std::runtime_error);
}

TEST(MeshFileLoader, ThrowsOnTruncatedBinaryStl) {
    // Header claims 2 triangles but only 1 worth of data is present.
    auto buf = MakeBinaryStl();
    buf.resize(84 + 50);  // header + count + 1 triangle (claims 2)
    const fs::path p = TempDir() / "truncated.stl";
    WriteBytes(p, buf);
    EXPECT_THROW(nuka::import::LoadStl(p.string()), std::runtime_error);
}

TEST(MeshFileLoader, ThrowsOnUnsupportedExtension) {
    const fs::path p = TempDir() / "thing.ply";
    WriteText(p, "ply\n");
    EXPECT_THROW(nuka::import::LoadMeshFile(p.string()), std::runtime_error);
}

// ---------------------------------------------------------------------------
// MJCF wiring: a mesh geom must load its referenced .stl into the shape record.
// ---------------------------------------------------------------------------

TEST(MjcfMeshWiring, PopulatesMeshVerticesAndIndices) {
    const fs::path dir = TempDir() / "mjcf_wiring";
    fs::create_directories(dir);
    const fs::path stl = dir / "tiny.stl";
    WriteBytes(stl, MakeBinaryStl());

    const std::string mjcf =
        "<mujoco>\n"
        "  <compiler meshdir=\".\"/>\n"
        "  <asset>\n"
        "    <mesh name=\"m\" file=\"tiny.stl\"/>\n"
        "  </asset>\n"
        "  <worldbody>\n"
        "    <body name=\"b\">\n"
        "      <geom type=\"mesh\" mesh=\"m\"/>\n"
        "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n";
    const fs::path xml = dir / "scene.xml";
    WriteText(xml, mjcf);

    const auto scene = nuka::import::LoadMjcf(xml.string());
    ASSERT_EQ(scene.ShapeCount(), 1u);
    const auto& shape = scene.GetShape(0);
    EXPECT_EQ(shape.type, nuka::scene::ShapeType::TriMesh);
    ASSERT_EQ(shape.mesh_vertices.size(), 18u);  // 6 verts * 3
    ASSERT_EQ(shape.mesh_indices.size(), 6u);
    // Vertices are laid out in file order: triangle 1 vertex 1 = kTri[1][1] =
    // (1,1,0) lives at flat offsets [12..14] (4th vertex overall).
    EXPECT_FLOAT_EQ(shape.mesh_vertices[12], 1.f);
    EXPECT_FLOAT_EQ(shape.mesh_vertices[13], 1.f);
    EXPECT_FLOAT_EQ(shape.mesh_vertices[14], 0.f);
}

TEST(MjcfMeshWiring, AppliesMeshScale) {
    const fs::path dir = TempDir() / "mjcf_scale";
    fs::create_directories(dir);
    const fs::path stl = dir / "tiny.stl";
    WriteBytes(stl, MakeBinaryStl());

    const std::string mjcf =
        "<mujoco>\n"
        "  <compiler meshdir=\".\"/>\n"
        "  <asset>\n"
        "    <mesh name=\"m\" file=\"tiny.stl\" scale=\"2 3 4\"/>\n"
        "  </asset>\n"
        "  <worldbody>\n"
        "    <body name=\"b\">\n"
        "      <geom type=\"mesh\" mesh=\"m\"/>\n"
        "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n";
    const fs::path xml = dir / "scene.xml";
    WriteText(xml, mjcf);

    const auto scene = nuka::import::LoadMjcf(xml.string());
    ASSERT_EQ(scene.ShapeCount(), 1u);
    const auto& shape = scene.GetShape(0);
    ASSERT_EQ(shape.mesh_vertices.size(), 18u);
    // Triangle 1 vertex 1 = (1,1,0) at flat offsets [12..14], scaled by
    // (2,3,4) => (2,3,0).
    EXPECT_FLOAT_EQ(shape.mesh_vertices[12], 2.f);
    EXPECT_FLOAT_EQ(shape.mesh_vertices[13], 3.f);
    EXPECT_FLOAT_EQ(shape.mesh_vertices[14], 0.f);
}

TEST(MjcfMeshWiring, ThrowsOnUnknownMeshAsset) {
    const fs::path dir = TempDir() / "mjcf_unknown";
    fs::create_directories(dir);
    const std::string mjcf =
        "<mujoco>\n"
        "  <worldbody>\n"
        "    <body name=\"b\">\n"
        "      <geom type=\"mesh\" mesh=\"missing\"/>\n"
        "    </body>\n"
        "  </worldbody>\n"
        "</mujoco>\n";
    const fs::path xml = dir / "scene.xml";
    WriteText(xml, mjcf);
    EXPECT_THROW(nuka::import::LoadMjcf(xml.string()), std::runtime_error);
}
