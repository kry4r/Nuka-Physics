// ---------------------------------------------------------------------------
// The USD importer on the REAL newton-assets cup: a usda wrapper referencing a USDC crate.
//
// Validates that .nuka-assets/.../cup/{model.usda -> mesh.usd} imports through OpenUSD into
// the expected mesh, with parser-INDEPENDENT oracles that genuinely bite:
//   * point count + bbox + hand-pinned vertex values (from a raw hex dump),
//     UNSCALED (the crate alone) and SCALED (the wrapper, its scale baked),
//   * topology invariants: 1794 quads -> 3588 triangles, every index in [0, nPoints),
//   * watertight / 2-manifold: every edge shared by exactly 2 triangles (a buggy
//     integer decode FAILS this -- a cup is a closed surface),
//   * D1: two loads are byte-identical; two cooks are byte-identical,
//   * the cup cooks to a COLLIDABLE convex hull (the C7b grasp-gate need).
//
// The asset's Mesh has no PhysicsCollisionAPI, so it imports as a visual; collider checks go through a
// wrapper that applies the API on a guide-purpose prim, as Isaac and Newton author collision meshes.
//
// The asset path resolves relative to ${CMAKE_SOURCE_DIR} (the import test's
// WORKING_DIRECTORY). The cup is a fetch-per-env asset (.nuka-assets is
// gitignored, per the newton-assets policy), so this suite SKIPS when the asset
// is absent -- matching the repo convention (e.g. foot_ground_mjx_parity). When
// the asset IS present every assertion above runs.
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <utility>

namespace {

namespace fs = std::filesystem;

const std::string kCupDir =
    ".nuka-assets/newton_assets/manipulation_objects/cup";
const std::string kModelUsda = kCupDir + "/model.usda";   // ASCII wrapper
const std::string kMeshUsd = kCupDir + "/mesh.usd";       // USDC crate

// Authored xformOp:scale on the cup's parent Xform (model.usda).
constexpr float kScaleX = 0.03f;
constexpr float kScaleY = 0.03f;
constexpr float kScaleZ = 0.0423f;

struct Bbox {
    float lo[3] = {1e30f, 1e30f, 1e30f};
    float hi[3] = {-1e30f, -1e30f, -1e30f};
    void Add(float x, float y, float z) {
        lo[0] = std::min(lo[0], x); hi[0] = std::max(hi[0], x);
        lo[1] = std::min(lo[1], y); hi[1] = std::max(hi[1], y);
        lo[2] = std::min(lo[2], z); hi[2] = std::max(hi[2], z);
    }
};

Bbox BboxOf(const std::vector<float>& v) {
    Bbox b;
    for (size_t i = 0; i + 2 < v.size(); i += 3) b.Add(v[i], v[i + 1], v[i + 2]);
    return b;
}

const nuka::scene::CollisionShapeRecord* FirstMeshShape(
    const nuka::scene::SceneIR& scene) {
    for (const auto& s : scene.Shapes()) {
        if (!s.mesh_vertices.empty()) return &s;
    }
    return nullptr;
}

std::string WriteWrapper(const char* name, const std::string& text) {
    const fs::path path = fs::temp_directory_path() / name;
    std::ofstream(path) << text;
    return path.string();
}

// The crate's mesh as a collision-only mesh. The crate authors no stage metadata, so the wrapper
// declares the metres and +Z up model.usda declares for it.
std::string CrateCollider() {
    return WriteWrapper("nuka_cup_crate_collider.usda",
        "#usda 1.0\n(\n    metersPerUnit = 1\n    upAxis = \"Z\"\n)\n\ndef Mesh \"Model\" (\n"
        "    prepend apiSchemas = [\"PhysicsCollisionAPI\"]\n"
        "    prepend references = @" + fs::absolute(kMeshUsd).generic_string() + "@</Model>\n"
        ")\n{\n    uniform token purpose = \"guide\"\n}\n");
}

// The wrapper asset with its mesh made collision-only.
std::string ModelCollider() {
    return WriteWrapper("nuka_cup_model_collider.usda",
        "#usda 1.0\n(\n    defaultPrim = \"root\"\n    metersPerUnit = 1\n    upAxis = \"Z\"\n)\n\n"
        "def Xform \"root\" (\n"
        "    prepend references = @" + fs::absolute(kModelUsda).generic_string() + "@</root>\n"
        ")\n{\n    over \"Model\"\n    {\n        over \"Model\" (\n"
        "            prepend apiSchemas = [\"PhysicsCollisionAPI\"]\n        )\n        {\n"
        "            uniform token purpose = \"guide\"\n        }\n    }\n}\n");
}

} // namespace

// ---------------------------------------------------------------------------
// Fixture: SKIP (loudly) when the newton-assets cup is not present. The cup is a
// fetch-per-env asset (.nuka-assets is gitignored, per the newton-assets policy),
// so this mirrors the foot_ground_mjx_parity convention: run + assert where the
// asset is provided (build-cuda128 dev env), skip portably elsewhere (clean
// checkout / CI without the fetch) instead of a confusing hard failure.
// ---------------------------------------------------------------------------
class UsdcCup : public ::testing::Test {
protected:
    void SetUp() override {
        if (!fs::exists(kMeshUsd) || !fs::exists(kModelUsda)) {
            GTEST_SKIP()
                << "newton-assets cup not present (fetch-per-env; .nuka-assets is "
                   "gitignored): " << fs::absolute(kMeshUsd)
                << " -- skipping the C7a usdc-cup reader test.";
        }
    }
};

// ---------------------------------------------------------------------------
// The crate alone: counts, bbox, hand-pinned UNSCALED vertices.
// ---------------------------------------------------------------------------
TEST_F(UsdcCup, DirectCrateReadCountsAndPinnedVerts) {
    const auto scene = nuka::import::LoadUsd(CrateCollider());
    ASSERT_EQ(scene.ShapeCount(), 1u);
    const auto* shape = FirstMeshShape(scene);
    ASSERT_NE(shape, nullptr);
    EXPECT_EQ(shape->name, "Model");
    const std::vector<float>& points = shape->mesh_vertices;
    ASSERT_EQ(points.size(), 1796u * 3u);
    ASSERT_EQ(shape->mesh_indices.size(), 3588u * 3u);

    // UNSCALED bbox is the unit envelope [-1,1]^3.
    const Bbox b = BboxOf(points);
    for (int a = 0; a < 3; ++a) {
        EXPECT_NEAR(b.lo[a], -1.0f, 1e-3f) << "axis " << a;
        EXPECT_NEAR(b.hi[a], 1.0f, 1e-3f) << "axis " << a;
    }

    // Hand-pinned vertices read straight from a raw hex dump at the resolved
    // points payload (parser-independent ground truth).
    EXPECT_NEAR(points[0], -0.7071071863f, 1e-6f);
    EXPECT_NEAR(points[1], -0.7071071863f, 1e-6f);
    EXPECT_NEAR(points[2], -1.0000017881f, 1e-6f);
    EXPECT_NEAR(points[3], -0.7071071863f, 1e-6f);  // v1.x
    EXPECT_NEAR(points[5], -0.9027777910f, 1e-6f);  // v1.z
    // last vertex (index 1795).
    EXPECT_NEAR(points[1795 * 3 + 0], 0.7071051598f, 1e-6f);
    EXPECT_NEAR(points[1795 * 3 + 2], 0.9999995232f, 1e-6f);
}

// ---------------------------------------------------------------------------
// Topology oracle (parser-independent): a buggy integer decode of either int[]
// array fails one of these. NONE of these are loosened.
// ---------------------------------------------------------------------------
TEST_F(UsdcCup, TopologyInvariantsAndWatertight) {
    const auto scene = nuka::import::LoadUsd(CrateCollider());
    const auto* shape = FirstMeshShape(scene);
    ASSERT_NE(shape, nullptr);
    const uint32_t nPoints = static_cast<uint32_t>(shape->mesh_vertices.size() / 3u);
    const std::vector<uint32_t>& tris = shape->mesh_indices;
    // 1794 all-quad faces fan into exactly two triangles each.
    ASSERT_EQ(tris.size(), 1794u * 2u * 3u);

    // every index in [0, nPoints).
    EXPECT_LT(*std::max_element(tris.begin(), tris.end()), nPoints);

    // Watertight / 2-manifold: every undirected edge shared by exactly 2 triangles.
    std::map<std::pair<uint32_t, uint32_t>, int> edges;
    for (size_t t = 0; t + 2 < tris.size(); t += 3) {
        for (size_t k = 0; k < 3; ++k) {
            const uint32_t a = tris[t + k];
            const uint32_t b = tris[t + (k + 1) % 3];
            edges[{std::min(a, b), std::max(a, b)}] += 1;
        }
    }
    int non_manifold = 0;
    for (const auto& [edge, count] : edges) {
        (void)edge;
        if (count != 2) ++non_manifold;
    }
    EXPECT_GT(edges.size(), 0u);
    EXPECT_EQ(non_manifold, 0)
        << "cup must be a closed 2-manifold; non-2 edges => bad face decode";
}

// ---------------------------------------------------------------------------
// Full importer path (model.usda -> crate reference -> scale bake), with the
// hand-pinned SCALED vertex values and the few-cm cup bbox.
// ---------------------------------------------------------------------------
TEST_F(UsdcCup, ImporterScaleBakedAndPinnedVerts) {
    const auto scene = nuka::import::LoadUsd(ModelCollider());
    const auto* shape = FirstMeshShape(scene);
    ASSERT_NE(shape, nullptr);
    EXPECT_EQ(shape->type, nuka::scene::ShapeType::TriMesh);

    ASSERT_EQ(shape->mesh_vertices.size(), 1796u * 3u);
    // 1794 quads -> 3588 tris -> 10764 indices.
    ASSERT_EQ(shape->mesh_indices.size(), 3588u * 3u);

    // SCALED bbox ~ a few-cm cup: 0.06 x 0.06 x 0.0846.
    const Bbox b = BboxOf(shape->mesh_vertices);
    EXPECT_NEAR(b.lo[0], -kScaleX, 2e-4f);
    EXPECT_NEAR(b.hi[0], kScaleX, 2e-4f);
    EXPECT_NEAR(b.lo[1], -kScaleY, 2e-4f);
    EXPECT_NEAR(b.hi[1], kScaleY, 2e-4f);
    EXPECT_NEAR(b.lo[2], -kScaleZ, 2e-4f);
    EXPECT_NEAR(b.hi[2], kScaleZ, 2e-4f);

    // Hand-pinned SCALED v0 = unscaled * (0.03, 0.03, 0.0423).
    EXPECT_NEAR(shape->mesh_vertices[0], -0.7071071863f * kScaleX, 1e-6f);
    EXPECT_NEAR(shape->mesh_vertices[1], -0.7071071863f * kScaleY, 1e-6f);
    EXPECT_NEAR(shape->mesh_vertices[2], -1.0000017881f * kScaleZ, 1e-6f);
}

// ---------------------------------------------------------------------------
// D1: two independent loads are byte-identical (vertices AND indices).
// ---------------------------------------------------------------------------
TEST_F(UsdcCup, DeterministicAcrossLoads) {
    const std::string path = ModelCollider();
    const auto a = nuka::import::LoadUsd(path);
    const auto b = nuka::import::LoadUsd(path);
    const auto* sa = FirstMeshShape(a);
    const auto* sb = FirstMeshShape(b);
    ASSERT_NE(sa, nullptr);
    ASSERT_NE(sb, nullptr);
    EXPECT_EQ(sa->mesh_vertices, sb->mesh_vertices);
    EXPECT_EQ(sa->mesh_indices, sb->mesh_indices);
    // non-trivial (guard against an empty-and-equal false pass).
    EXPECT_EQ(sa->mesh_vertices.size(), 1796u * 3u);
}

// Preserve the cup's concavity and its original collision triangles.
TEST_F(UsdcCup, CooksToCollidableTriangleSurface) {
    auto scene = nuka::import::LoadUsd(ModelCollider());
    for (size_t i = 0; i < scene.ShapeCount(); ++i) {
        auto& s = scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        if (!s.mesh_vertices.empty()) {
            s.decompose_mode = nuka::scene::DecomposeMode::Auto;
        }
    }

    const auto blob = nuka::scene::CookScene(scene);

    ASSERT_EQ(blob.shapes.types.size(), 1u);
    EXPECT_EQ(blob.shapes.types[0], nuka::scene::ShapeType::TriMesh);
    ASSERT_FALSE(blob.convex_geometry.vertex_counts.empty());
    EXPECT_GT(blob.convex_geometry.vertex_counts[0], 0u);
    EXPECT_GT(blob.convex_geometry.index_counts[0], 0u);
    EXPECT_FALSE(blob.convex_geometry.vertices.empty());
    ASSERT_EQ(blob.convex_geometry.surface_info.size(), 1u);
    ASSERT_EQ(blob.convex_geometry.surface_covers.size(), 1u);
    const auto& cover = blob.convex_geometry.surface_covers[0];
    EXPECT_EQ(cover.status, nuka::import::cooker::ConvexCoverStatus::Complete) << cover.reason;
    EXPECT_GT(cover.parts.size(), 1u);
    std::fprintf(stderr, "[cup cook] backend=%s parts=%zu cells=%u queries=%llu hierarchy=%u\n",
        cover.backend.c_str(), cover.parts.size(), cover.distance_cells,
        static_cast<unsigned long long>(cover.query_points),
        unsigned(blob.convex_geometry.surface_cover_hierarchy[0]));
    EXPECT_EQ(blob.convex_geometry.vertices, scene.GetShape(0u).mesh_vertices);
    EXPECT_EQ(blob.convex_geometry.indices, scene.GetShape(0u).mesh_indices);

    // The collision mesh retains the imported scale.
    const Bbox hb = BboxOf(blob.convex_geometry.vertices);
    EXPECT_LE(hb.hi[2], kScaleZ + 2e-4f);
    EXPECT_GE(hb.lo[2], -kScaleZ - 2e-4f);

    // Repeated cooking preserves vertex and triangle order.
    const auto blob2 = nuka::scene::CookScene(scene);
    EXPECT_EQ(blob.convex_geometry.vertices, blob2.convex_geometry.vertices);
    EXPECT_EQ(blob.convex_geometry.indices, blob2.convex_geometry.indices);
}

// ---------------------------------------------------------------------------
// The asset as authored: a visual record with one vertex per distinct
// (point, normal) corner, kept in the authored point order.
// ---------------------------------------------------------------------------
TEST_F(UsdcCup, PlainMeshImportsAsVisual) {
    const auto scene = nuka::import::LoadUsd(kModelUsda);
    ASSERT_EQ(scene.ShapeCount(), 1u);
    const auto& shape = scene.GetShape(0u);
    EXPECT_EQ(shape.contype, 0u);
    EXPECT_EQ(shape.conaffinity, 0u);
    ASSERT_EQ(shape.mesh_indices.size(), 3588u * 3u);
    ASSERT_EQ(shape.mesh_vertices.size(), 6307u * 3u);
    EXPECT_EQ(shape.mesh_normals.size(), shape.mesh_vertices.size());
    EXPECT_NEAR(shape.mesh_vertices[0], -0.7071071863f * kScaleX, 1e-6f);
    EXPECT_NEAR(shape.mesh_vertices[2], -1.0000017881f * kScaleZ, 1e-6f);
    const Bbox b = BboxOf(shape.mesh_vertices);
    EXPECT_NEAR(b.lo[2], -kScaleZ, 2e-4f);
    EXPECT_NEAR(b.hi[2], kScaleZ, 2e-4f);
}
