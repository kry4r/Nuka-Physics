// ---------------------------------------------------------------------------
// v0.8 C7a: the self-written USDC (crate) reader on the REAL newton-assets cup.
//
// Validates that .nuka-assets/.../cup/{model.usda -> mesh.usd} parses through
// the from-scratch crate reader (NO OpenUSD SDK) into the expected mesh, with
// parser-INDEPENDENT oracles that genuinely bite:
//   * point count + bbox + hand-pinned vertex values (from a raw hex dump),
//     UNSCALED (direct crate read) and SCALED (full importer, scale baked),
//   * topology invariants: sum(faceVertexCounts) == len(faceVertexIndices),
//     every index in [0, nPoints), all faces are quads,
//   * watertight / 2-manifold: every edge shared by exactly 2 faces (a buggy
//     integer decode FAILS this -- a cup is a closed surface),
//   * D1: two loads are byte-identical; two cooks are byte-identical,
//   * the cup cooks to a COLLIDABLE convex hull (the C7b grasp-gate need).
//
// The asset path resolves relative to ${CMAKE_SOURCE_DIR} (the import test's
// WORKING_DIRECTORY). The cup is a fetch-per-env asset (.nuka-assets is
// gitignored, per the newton-assets policy), so this suite SKIPS when the asset
// is absent -- matching the repo convention (e.g. foot_ground_mjx_parity). When
// the asset IS present every assertion above runs.
// ---------------------------------------------------------------------------

#include "import/usd_crate_reader.hpp"
#include "import/usd_importer.hpp"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
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
// Direct crate read: counts, bbox, hand-pinned UNSCALED vertices.
// ---------------------------------------------------------------------------
TEST_F(UsdcCup, DirectCrateReadCountsAndPinnedVerts) {
    const auto mesh = nuka::import::ReadUsdcMesh(kMeshUsd, "/Model");

    EXPECT_EQ(mesh.prim_path, "/Model");
    ASSERT_EQ(mesh.PointCount(), 1796u);
    ASSERT_EQ(mesh.face_vertex_indices.size(), 7176u);
    ASSERT_EQ(mesh.face_vertex_counts.size(), 1794u);

    // UNSCALED bbox is the unit envelope [-1,1]^3.
    const Bbox b = BboxOf(mesh.points);
    for (int a = 0; a < 3; ++a) {
        EXPECT_NEAR(b.lo[a], -1.0f, 1e-3f) << "axis " << a;
        EXPECT_NEAR(b.hi[a], 1.0f, 1e-3f) << "axis " << a;
    }

    // Hand-pinned vertices read straight from a raw hex dump at the resolved
    // points payload (parser-independent ground truth).
    EXPECT_NEAR(mesh.points[0], -0.7071071863f, 1e-6f);
    EXPECT_NEAR(mesh.points[1], -0.7071071863f, 1e-6f);
    EXPECT_NEAR(mesh.points[2], -1.0000017881f, 1e-6f);
    EXPECT_NEAR(mesh.points[3], -0.7071071863f, 1e-6f);  // v1.x
    EXPECT_NEAR(mesh.points[5], -0.9027777910f, 1e-6f);  // v1.z
    // last vertex (index 1795).
    EXPECT_NEAR(mesh.points[1795 * 3 + 0], 0.7071051598f, 1e-6f);
    EXPECT_NEAR(mesh.points[1795 * 3 + 2], 0.9999995232f, 1e-6f);
}

// ---------------------------------------------------------------------------
// Topology oracle (parser-independent): a buggy integer decode of either int[]
// array fails one of these. NONE of these are loosened.
// ---------------------------------------------------------------------------
TEST_F(UsdcCup, TopologyInvariantsAndWatertight) {
    const auto mesh = nuka::import::ReadUsdcMesh(kMeshUsd, "/Model");
    const int nPoints = static_cast<int>(mesh.PointCount());

    // sum(faceVertexCounts) == len(faceVertexIndices).
    long sum_counts = 0;
    for (int c : mesh.face_vertex_counts) {
        EXPECT_EQ(c, 4) << "cup is an all-quad mesh";  // genuinely all quads
        sum_counts += c;
    }
    EXPECT_EQ(sum_counts, static_cast<long>(mesh.face_vertex_indices.size()));

    // every index in [0, nPoints).
    int idx_min = nPoints, idx_max = -1;
    for (int i : mesh.face_vertex_indices) {
        idx_min = std::min(idx_min, i);
        idx_max = std::max(idx_max, i);
    }
    EXPECT_GE(idx_min, 0);
    EXPECT_LT(idx_max, nPoints);

    // Watertight / 2-manifold: every undirected edge shared by exactly 2 faces.
    std::map<std::pair<int, int>, int> edges;
    size_t cur = 0;
    for (int c : mesh.face_vertex_counts) {
        for (int k = 0; k < c; ++k) {
            const int a = mesh.face_vertex_indices[cur + static_cast<size_t>(k)];
            const int b =
                mesh.face_vertex_indices[cur + static_cast<size_t>((k + 1) % c)];
            edges[{std::min(a, b), std::max(a, b)}] += 1;
        }
        cur += static_cast<size_t>(c);
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
// Full importer path (model.usda -> crate ref graft -> scale bake), with the
// hand-pinned SCALED vertex values and the few-cm cup bbox.
// ---------------------------------------------------------------------------
TEST_F(UsdcCup, ImporterScaleBakedAndPinnedVerts) {
    const auto scene = nuka::import::LoadUsd(kModelUsda);
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
    const auto a = nuka::import::LoadUsd(kModelUsda);
    const auto b = nuka::import::LoadUsd(kModelUsda);
    const auto* sa = FirstMeshShape(a);
    const auto* sb = FirstMeshShape(b);
    ASSERT_NE(sa, nullptr);
    ASSERT_NE(sb, nullptr);
    EXPECT_EQ(sa->mesh_vertices, sb->mesh_vertices);
    EXPECT_EQ(sa->mesh_indices, sb->mesh_indices);
    // non-trivial (guard against an empty-and-equal false pass).
    EXPECT_EQ(sa->mesh_vertices.size(), 1796u * 3u);
}

// ---------------------------------------------------------------------------
// The cup cooks to a COLLIDABLE convex hull (the C7b grasp-gate need). With
// DecomposeMode::Skip the source mesh becomes exactly one ConvexHull carrying
// its own geometry; cook-twice is byte-identical (D1).
// ---------------------------------------------------------------------------
TEST_F(UsdcCup, CooksToCollidableConvexHull) {
    auto scene = nuka::import::LoadUsd(kModelUsda);
    // Force the single-hull path so the assertion is deterministic and the
    // result is unambiguously one collidable ConvexHull.
    for (size_t i = 0; i < scene.ShapeCount(); ++i) {
        auto& s = scene.GetShapeMut(static_cast<nuka::scene::ShapeId>(i));
        if (!s.mesh_vertices.empty()) {
            s.decompose_mode = nuka::scene::DecomposeMode::Skip;
        }
    }

    const auto blob = nuka::scene::CookScene(scene);

    // Exactly one shape, of ConvexHull type, with non-empty hull geometry.
    ASSERT_EQ(blob.shapes.types.size(), 1u);
    EXPECT_EQ(blob.shapes.types[0], nuka::scene::ShapeType::ConvexHull);
    ASSERT_FALSE(blob.convex_geometry.vertex_counts.empty());
    EXPECT_GT(blob.convex_geometry.vertex_counts[0], 0u);
    EXPECT_GT(blob.convex_geometry.index_counts[0], 0u);
    EXPECT_FALSE(blob.convex_geometry.vertices.empty());

    // The cooked hull lives in the few-cm scaled envelope (collidable cup).
    const Bbox hb = BboxOf(blob.convex_geometry.vertices);
    EXPECT_LE(hb.hi[2], kScaleZ + 2e-4f);
    EXPECT_GE(hb.lo[2], -kScaleZ - 2e-4f);

    // D1: cook twice -> byte-identical hull geometry.
    const auto blob2 = nuka::scene::CookScene(scene);
    EXPECT_EQ(blob.convex_geometry.vertices, blob2.convex_geometry.vertices);
    EXPECT_EQ(blob.convex_geometry.indices, blob2.convex_geometry.indices);
}
