// Mesh cooking preserves source surfaces or emits explicitly requested convex pieces.

#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"
#include "tests/import/vhacd_test_meshes.hpp"
#include "collision/mesh_surface.hpp"
#include "collision/primitive_surface.hpp"

#include <gtest/gtest.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <stdexcept>

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

// Every decomposed piece inherits the source shape's contact parameters.
TEST(VhacdCookerIntegration, ContactParamsParallelAndPropagatedAcrossPieces) {
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
    // Non-default per-shape contact metadata that must reach EVERY piece row.
    shape.contype     = 7u;
    shape.conaffinity = 5u;
    shape.condim      = 1u;
    shape.friction_mu = 0.42f;   // explicit per-shape override
    shape.solref[0]   = 0.015f;
    scene.AddCollisionShape(std::move(shape));

    const auto blob = CookScene(scene);
    ASSERT_GE(blob.shape_count, 2u);  // genuinely decomposed

    const auto& cp = blob.contact_params;
    // Parallelism: every contact-param vector has exactly shape_count rows.
    ASSERT_EQ(cp.contypes.size(),      blob.shape_count);
    EXPECT_EQ(cp.conaffinities.size(), blob.shape_count);
    EXPECT_EQ(cp.groups.size(),        blob.shape_count);
    EXPECT_EQ(cp.solref0.size(),       blob.shape_count);
    EXPECT_EQ(cp.solref1.size(),       blob.shape_count);
    EXPECT_EQ(cp.solimp.size(),        blob.shape_count * 5u);
    EXPECT_EQ(cp.frictions.size(),     blob.shape_count);
    EXPECT_EQ(cp.condims.size(),       blob.shape_count);

    // Propagation: each piece inherits the source geom's metadata + resolved mu.
    for (uint32_t row = 0; row < blob.shape_count; ++row) {
        EXPECT_EQ(cp.contypes[row],      7u);
        EXPECT_EQ(cp.conaffinities[row], 5u);
        EXPECT_EQ(cp.condims[row],       1u);
        EXPECT_FLOAT_EQ(cp.frictions[row], 0.42f);
        EXPECT_FLOAT_EQ(cp.solref0[row],   0.015f);
    }
}

TEST(VhacdCookerIntegration, SkipPreservesTriangleSurfaceAndInterior) {
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
    EXPECT_EQ(blob.shapes.types[0], ShapeType::TriMesh);
    const uint32_t gi = blob.shapes.convex_geometry_indices[0];
    ASSERT_NE(gi, kNoConvexGeometry);
    // Skip stores the source mesh verbatim (no V-HACD), so vertex count matches.
    EXPECT_EQ(blob.convex_geometry.vertex_counts[gi],
              static_cast<uint32_t>(mesh.vertices.size() / 3));
    EXPECT_EQ(blob.convex_geometry.index_counts[gi],
              static_cast<uint32_t>(mesh.indices.size()));
    const auto& geometry = blob.convex_geometry;
    const nuka::collision::MeshSurfaceView view{geometry.vertices.data(), geometry.indices.data(),
        geometry.surface_nodes.data(), {static_cast<uint32_t>(geometry.vertices.size() / 3u),
            static_cast<uint32_t>(geometry.indices.size() / 3u),
            static_cast<uint32_t>(geometry.surface_nodes.size())}};
    ASSERT_EQ(geometry.surface_info[gi].flags, nuka::collision::kMeshSurfaceClosed);
    for (float x : {-0.75f, -0.5f, -0.25f, 0.0f, 0.25f, 0.5f, 0.75f})
        for (float y : {-0.5f, -0.25f, 0.0f, 0.25f, 0.5f})
            for (float z : {-0.5f, -0.25f, 0.0f, 0.25f, 0.5f}) {
                const nuka::math::Vec3 p{x, y, z};
                const auto exact = nuka::collision::QueryPrimitiveSurface(
                    nuka::collision::kShapeBox, {0.5f, 0.5f, 0.5f}, p);
                const auto surface = nuka::collision::QueryMeshSurface(view, geometry.surface_info[gi], p);
                ASSERT_TRUE(surface.valid);
                EXPECT_NEAR(surface.distance, exact.distance, 2e-6f) << x << ',' << y << ',' << z;
                EXPECT_NEAR(surface.normal.Length(), 1.0f, 2e-6f);
                EXPECT_LT((surface.point + surface.normal * surface.distance - p).Length(), 2e-6f);
                nuka::math::Vec3 a, b, c;
                ASSERT_TRUE(nuka::collision::MeshSurfaceTriangle(view, geometry.surface_info[gi],
                    surface.triangle, a, b, c));
                EXPECT_LT((a * surface.barycentric.x + b * surface.barycentric.y +
                    c * surface.barycentric.z - surface.point).Length(), 2e-6f);
            }
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

TEST(VhacdCookerIntegration, RejectsMissingCollisionMeshGeometry) {
    for (auto mode : {DecomposeMode::Auto, DecomposeMode::Force, DecomposeMode::Skip}) {
        SceneIR scene;
        CollisionShapeRecord shape;
        shape.body_id = AddDynamicBody(scene);
        shape.type = ShapeType::TriMesh;
        shape.decompose_mode = mode;
        const auto id = scene.AddCollisionShape(std::move(shape));
        EXPECT_THROW(CookScene(scene), std::invalid_argument);
        scene.GetShapeMut(id).mesh_vertices = {0, 0, 0, 1, 0, 0, 0, 1, 0};
        EXPECT_THROW(CookScene(scene), std::invalid_argument);
        scene.GetShapeMut(id).mesh_vertices.clear();
        scene.GetShapeMut(id).mesh_indices = {0u, 1u, 2u};
        EXPECT_THROW(CookScene(scene), std::invalid_argument);
        scene.GetShapeMut(id).contype = 0u;
        scene.GetShapeMut(id).conaffinity = 0u;
        const auto blob = CookScene(scene);
        ASSERT_EQ(blob.shape_count, 1u);
        EXPECT_EQ(blob.shapes.convex_geometry_indices[0], kNoConvexGeometry);
    }
}

TEST(VhacdCookerIntegration, AutoCookPreservesExactSurfaceAcrossDevicesCachesAndBudgets) {
    namespace cooker = nuka::import::cooker;
    namespace collision = nuka::collision;
    nuka::test::TestMesh mesh;
    const float polygon[6][2] = {{0, 0}, {3, 0}, {3, 1}, {1, 1}, {1, 3}, {0, 3}};
    for (float z : {0.0f, 1.0f})
        for (const auto& point : polygon)
            mesh.vertices.insert(mesh.vertices.end(), {point[0], point[1], z});
    for (uint32_t i = 1u; i < 5u; ++i)
        mesh.indices.insert(mesh.indices.end(), {0u, i + 1u, i, 6u, i + 6u, i + 7u});
    for (uint32_t i = 0u; i < 6u; ++i) {
        const uint32_t j = (i + 1u) % 6u;
        mesh.indices.insert(mesh.indices.end(), {i, j, j + 6u, i, j + 6u, i + 6u});
    }
    const auto raw = cooker::CookMeshSurface(mesh.vertices.data(),
        static_cast<uint32_t>(mesh.vertices.size() / 3u), mesh.indices.data(),
        static_cast<uint32_t>(mesh.indices.size() / 3u));
    const collision::MeshSurfaceView reference{mesh.vertices.data(), mesh.indices.data(),
        raw.nodes.data(), {raw.info.vertex_count, raw.info.triangle_count, raw.info.node_count}};
    SceneIR scene;
    for (uint32_t instance = 0u; instance < 2u; ++instance) {
        CollisionShapeRecord shape;
        shape.body_id = AddDynamicBody(scene);
        shape.type = ShapeType::TriMesh;
        shape.mesh_vertices = mesh.vertices;
        shape.mesh_indices = mesh.indices;
        scene.AddCollisionShape(std::move(shape));
    }
    struct CacheDirectory {
        std::filesystem::path path;
        ~CacheDirectory() {
            std::error_code error;
            for (const auto& file : std::filesystem::directory_iterator(path, error))
                if (file.is_regular_file(error)) std::filesystem::remove(file.path(), error);
            std::filesystem::remove(path, error);
        }
    };
    for (bool allow_device : {false, true}) {
        SCOPED_TRACE(allow_device);
        const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
        CacheDirectory cache{std::filesystem::temp_directory_path() /
            ("nuka_surface_cook_" + std::to_string(stamp))};
        nuka::scene::CookSceneOptions options;
        options.bake_sdf = false;
        options.mesh_surface.allow_device = allow_device;
        options.mesh_surface.cache_directory = cache.path.string();
        const auto cold = CookScene(scene, options);
        ASSERT_EQ(cold.body_count, 2u);
        ASSERT_EQ(cold.shape_count, 2u);
        ASSERT_EQ(cold.convex_geometry.Count(), 1u);
        EXPECT_NE(cold.shapes.body_ids[0], cold.shapes.body_ids[1]);
        EXPECT_EQ(cold.shapes.convex_geometry_indices[0], cold.shapes.convex_geometry_indices[1]);
        const auto verify = [&](const nuka::scene::CookedBlob& blob) {
            const auto& geometry = blob.convex_geometry;
            EXPECT_EQ(geometry.vertices, mesh.vertices);
            EXPECT_EQ(geometry.indices, mesh.indices);
            EXPECT_EQ(blob.shapes.types[0], ShapeType::TriMesh);
            const auto& info = geometry.surface_info[0];
            const collision::MeshSurfaceView view{geometry.vertices.data(), geometry.indices.data(),
                geometry.surface_nodes.data(), {info.vertex_count, info.triangle_count, info.node_count}};
            ASSERT_TRUE(cooker::MeshSurfaceTreeValid(view, info));
            for (int x = -2; x <= 10; ++x)
                for (int y = -2; y <= 10; ++y)
                    for (int z = -2; z <= 10; ++z) {
                        const nuka::math::Vec3 point{float(x) * 0.25f, float(y) * 0.25f, float(z) * 0.25f};
                        const auto expected = collision::QueryMeshSurface(reference, raw.info, point);
                        const auto actual = collision::QueryMeshSurface(view, info, point);
                        ASSERT_TRUE(actual.valid);
                        EXPECT_FLOAT_EQ(actual.distance, expected.distance);
                        EXPECT_EQ(actual.triangle, expected.triangle);
                        EXPECT_EQ(actual.feature, expected.feature);
                        EXPECT_FLOAT_EQ(actual.normal.x, expected.normal.x);
                        EXPECT_FLOAT_EQ(actual.normal.y, expected.normal.y);
                        EXPECT_FLOAT_EQ(actual.normal.z, expected.normal.z);
                    }
        };
        const auto& cover = cold.convex_geometry.surface_covers[0];
        ASSERT_EQ(cover.status, cooker::ConvexCoverStatus::Complete) << cover.reason;
        EXPECT_GE(cover.parts.size(), 2u);
        EXPECT_EQ(cover.backend, cooker::MeshQueryBackendName(allow_device));
        EXPECT_GT(cover.query_points, 0u);
        verify(cold);
        const auto warm = CookScene(scene, options);
        ASSERT_EQ(warm.convex_geometry.surface_cache_hits[0], 1u);
        verify(warm);
        {
            std::ofstream corrupt(cache.path / (cold.convex_geometry.surface_cache_keys[0] + ".nukasurf"),
                                  std::ios::binary | std::ios::trunc);
            corrupt << "invalid surface artifact";
        }
        const auto repaired = CookScene(scene, options);
        EXPECT_EQ(repaired.convex_geometry.surface_cache_hits[0], 0u);
        verify(repaired);
        EXPECT_EQ(CookScene(scene, options).convex_geometry.surface_cache_hits[0], 1u);
        options.mesh_surface.cover.max_operations = 1u;
        const auto bounded = CookScene(scene, options);
        ASSERT_EQ(bounded.convex_geometry.surface_covers[0].status,
                  cooker::ConvexCoverStatus::BudgetExceeded);
        EXPECT_TRUE(bounded.convex_geometry.surface_covers[0].parts.empty());
        EXPECT_NE(bounded.convex_geometry.surface_cache_keys[0], cold.convex_geometry.surface_cache_keys[0]);
        verify(bounded);
    }
}

} // namespace
