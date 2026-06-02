// ---------------------------------------------------------------------------
// Tests for the cooker integration of V-HACD (CookScene -> ConvexHull rows)
// ---------------------------------------------------------------------------
// Validates deliverable 6: a mesh CollisionShapeRecord flagged for
// decomposition, when cooked, expands into N ShapeType::ConvexHull rows whose
// geometry lands in CookedBlob::convex_geometry. skip => 1 ConvexHull (its own
// hull); non-mesh shapes pass through unchanged.
// ---------------------------------------------------------------------------

#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"
#include "tests/import/vhacd_test_meshes.hpp"

#include <gtest/gtest.h>

namespace {

using nuka::scene::CollisionShapeRecord;
using nuka::scene::CookScene;
using nuka::scene::DecomposeMode;
using nuka::scene::kNoConvexGeometry;
using nuka::scene::RigidBodyRecord;
using nuka::scene::SceneIR;
using nuka::scene::ShapeType;
using nuka::test::LShapeMesh;
using nuka::test::UnitCubeMesh;

nuka::scene::BodyId AddDynamicBody(SceneIR& scene) {
    RigidBodyRecord body;
    body.name = "body";
    body.mass = 1.0f;
    return scene.AddRigidBody(std::move(body));
}

TEST(VhacdCookerIntegration, ForceDecomposeExpandsToMultipleConvexHullRows) {
    SceneIR scene;
    const auto body = AddDynamicBody(scene);

    const auto mesh = LShapeMesh();
    CollisionShapeRecord shape;
    shape.body_id = body;
    shape.type = ShapeType::TriMesh;
    shape.decompose_mode = DecomposeMode::Force;
    shape.decompose_max_pieces = 16;
    shape.mesh_vertices = mesh.vertices;
    shape.mesh_indices = mesh.indices;
    scene.AddCollisionShape(std::move(shape));

    const auto blob = CookScene(scene);

    // The single source mesh expanded into >= 2 ConvexHull rows.
    EXPECT_GE(blob.shape_count, 2u);
    EXPECT_EQ(blob.shape_count, blob.shapes.types.size());
    for (const auto t : blob.shapes.types) {
        EXPECT_EQ(t, ShapeType::ConvexHull);
    }

    // Every parallel array stays the same length.
    EXPECT_EQ(blob.shapes.body_ids.size(), blob.shape_count);
    EXPECT_EQ(blob.shapes.convex_geometry_indices.size(), blob.shape_count);

    // Each row references real geometry in the convex_geometry table.
    EXPECT_EQ(blob.convex_geometry.Count(), blob.shape_count);
    for (uint32_t row = 0; row < blob.shape_count; ++row) {
        const uint32_t gi = blob.shapes.convex_geometry_indices[row];
        ASSERT_NE(gi, kNoConvexGeometry);
        ASSERT_LT(gi, blob.convex_geometry.Count());
        EXPECT_GT(blob.convex_geometry.vertex_counts[gi], 0u);
        EXPECT_GT(blob.convex_geometry.index_counts[gi], 0u);
    }

    // The flat geometry slices are consistent with the offsets/counts.
    const auto& g = blob.convex_geometry;
    for (uint32_t i = 0; i < g.Count(); ++i) {
        EXPECT_LE((g.vertex_offsets[i] + g.vertex_counts[i]) * 3u, g.vertices.size());
        EXPECT_LE(g.index_offsets[i] + g.index_counts[i], g.indices.size());
    }
}

TEST(VhacdCookerIntegration, SkipStoresSingleConvexHullWithSourceGeometry) {
    SceneIR scene;
    const auto body = AddDynamicBody(scene);

    const auto mesh = UnitCubeMesh();
    CollisionShapeRecord shape;
    shape.body_id = body;
    shape.type = ShapeType::TriMesh;
    shape.decompose_mode = DecomposeMode::Skip;
    shape.mesh_vertices = mesh.vertices;
    shape.mesh_indices = mesh.indices;
    scene.AddCollisionShape(std::move(shape));

    const auto blob = CookScene(scene);

    ASSERT_EQ(blob.shape_count, 1u);
    EXPECT_EQ(blob.shapes.types[0], ShapeType::ConvexHull);
    const uint32_t gi = blob.shapes.convex_geometry_indices[0];
    ASSERT_NE(gi, kNoConvexGeometry);
    // Skip stores the source mesh verbatim (no V-HACD), so vertex count matches.
    EXPECT_EQ(blob.convex_geometry.vertex_counts[gi],
              static_cast<uint32_t>(mesh.vertices.size() / 3));
    EXPECT_EQ(blob.convex_geometry.index_counts[gi],
              static_cast<uint32_t>(mesh.indices.size()));
}

TEST(VhacdCookerIntegration, NonMeshShapesPassThroughUnchanged) {
    SceneIR scene;
    const auto body = AddDynamicBody(scene);

    CollisionShapeRecord box;
    box.body_id = body;
    box.type = ShapeType::Box;
    box.half_extents = {0.5f, 0.5f, 0.5f};
    scene.AddCollisionShape(std::move(box));

    const auto blob = CookScene(scene);

    ASSERT_EQ(blob.shape_count, 1u);
    EXPECT_EQ(blob.shapes.types[0], ShapeType::Box);
    EXPECT_EQ(blob.shapes.convex_geometry_indices[0], kNoConvexGeometry);
    EXPECT_EQ(blob.convex_geometry.Count(), 0u);
}

TEST(VhacdCookerIntegration, MeshWithoutGeometryPassesThroughAsIs) {
    SceneIR scene;
    const auto body = AddDynamicBody(scene);

    // A TriMesh shape flagged Force but carrying no geometry (the importer path
    // today): must NOT crash and must pass through as a single TriMesh row.
    CollisionShapeRecord shape;
    shape.body_id = body;
    shape.type = ShapeType::TriMesh;
    shape.decompose_mode = DecomposeMode::Force;
    scene.AddCollisionShape(std::move(shape));

    const auto blob = CookScene(scene);

    ASSERT_EQ(blob.shape_count, 1u);
    EXPECT_EQ(blob.shapes.types[0], ShapeType::TriMesh);
    EXPECT_EQ(blob.shapes.convex_geometry_indices[0], kNoConvexGeometry);
}

} // namespace
