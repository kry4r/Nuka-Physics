// ---------------------------------------------------------------------------
// p07 sparse narrow-band SDF: determinism + per-mesh dedup (exit-crit 4 & 7).
// ---------------------------------------------------------------------------
// Determinism: N cooks of the same mesh + params produce a byte-identical
// SparseSdfData (cell_keys + values + gradients). Mirrors
// test_vhacd_determinism.cpp's serialize-and-compare. Host single-thread + fixed
// traversal order + double->float-once => bit-reproducible.
//
// Dedup: two identical mesh pieces in a scene cook to ONE stored SDF (content
// hash). We verify CookedSdfTable::Count() (unique) vs piece count (total), the
// shared SDF index, and report narrow-band bytes vs the ~80MB cap (R-C).
//
// Sign-on-real-piece: a Skip-mode cube piece (the cook path's actual ConvexHull)
// samples phi<0 at an interior point, phi>0 outside -> the convex-hull sign path
// is exercised on shipping geometry, not just hand-wound test meshes.
// ---------------------------------------------------------------------------

#include "import/cooker/sparse_sdf_cooker.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"
#include "scene/cooker.hpp"
#include "scene/scene_ir.hpp"
#include "tests/import/sdf_test_meshes.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

using nuka::import::cooker::CookSparseSdf;
using nuka::import::cooker::SparseSdfData;
using nuka::import::cooker::SparseSdfParams;
using nuka::scene::CollisionShapeRecord;
using nuka::scene::CookScene;
using nuka::scene::DecomposeMode;
using nuka::scene::kNoSdf;
using nuka::scene::RigidBodyRecord;
using nuka::scene::SceneIR;
using nuka::scene::ShapeType;

// Serialize an SDF to bytes for exact comparison (header + sorted cells).
std::string SdfBytes(const SparseSdfData& sdf) {
    std::string out;
    auto append = [&out](const void* p, size_t n) {
        out.append(static_cast<const char*>(p), n);
    };
    append(sdf.origin, sizeof(sdf.origin));
    append(&sdf.voxel_size, sizeof(sdf.voxel_size));
    append(sdf.dims, sizeof(sdf.dims));
    const uint32_t n = sdf.CellCount();
    append(&n, sizeof(n));
    if (n) {
        append(sdf.cell_keys.data(), sdf.cell_keys.size() * sizeof(uint64_t));
        append(sdf.cell_values.data(), sdf.cell_values.size() * sizeof(float));
        append(sdf.cell_gradients.data(), sdf.cell_gradients.size() * sizeof(float));
    }
    return out;
}

SparseSdfParams DetParams() {
    SparseSdfParams p;
    p.voxel_size = 0.08f;  // coarse -> cheap; determinism is resolution-agnostic
    p.band_voxels = 2u;
    return p;
}

TEST(SparseSdfDeterminism, RepeatedCooksAreByteIdentical) {
    const auto mesh = nuka::test::SphereMesh(0.0f, 0.0f, 0.0f, 1.0f, 32, 64);
    const uint32_t vc = static_cast<uint32_t>(mesh.vertices.size() / 3);
    const uint32_t tc = static_cast<uint32_t>(mesh.indices.size() / 3);

    const auto first = CookSparseSdf(mesh.vertices.data(), vc, mesh.indices.data(), tc, DetParams());
    ASSERT_GT(first.CellCount(), 0u);
    const std::string golden = SdfBytes(first);

    for (int run = 0; run < 8; ++run) {
        const auto r = CookSparseSdf(mesh.vertices.data(), vc, mesh.indices.data(), tc, DetParams());
        EXPECT_EQ(r.CellCount(), first.CellCount()) << "run " << run;
        EXPECT_EQ(SdfBytes(r), golden) << "run " << run << " differs byte-wise";
    }
}

// Gate #4 names CookedSdfTable specifically: two cooks of the same scene must
// produce a byte-identical table (cell_keys + values + gradients). The
// SparseSdfData test above covers the generator; this covers the concatenation
// path (AppendSdf in deterministic piece order) end-to-end.
TEST(SparseSdfDeterminism, CookedSdfTableIsByteIdenticalAcrossCooks) {
    SceneIR scene;
    {
        RigidBodyRecord body;
        body.name = "body";
        body.mass = 1.0f;
        const auto id = scene.AddRigidBody(std::move(body));
        const auto cube = nuka::test::UnitCubeMesh();
        CollisionShapeRecord shape;
        shape.body_id = id;
        shape.type = ShapeType::TriMesh;
        shape.decompose_mode = DecomposeMode::Skip;
        shape.mesh_vertices = cube.vertices;
        shape.mesh_indices = cube.indices;
        scene.AddCollisionShape(std::move(shape));
    }
    const auto a = CookScene(scene);
    const auto b = CookScene(scene);
    ASSERT_GT(a.sdfs.Count(), 0u);
    EXPECT_EQ(a.sdfs.Count(), b.sdfs.Count());
    EXPECT_EQ(a.sdfs.cell_keys, b.sdfs.cell_keys);
    EXPECT_EQ(a.sdfs.cell_values, b.sdfs.cell_values);
    // Gradients are math::Vec3; compare component-wise as a byte run.
    ASSERT_EQ(a.sdfs.cell_gradients.size(), b.sdfs.cell_gradients.size());
    EXPECT_EQ(std::memcmp(a.sdfs.cell_gradients.data(), b.sdfs.cell_gradients.data(),
                          a.sdfs.cell_gradients.size() * sizeof(nuka::math::Vec3)),
              0);
}

// Build a scene with `n` IDENTICAL Skip-mode cube mesh shapes (each -> one
// ConvexHull piece). All pieces share one geometry -> one deduped SDF.
SceneIR BuildIdenticalCubes(int n) {
    SceneIR scene;
    const auto cube = nuka::test::UnitCubeMesh();
    for (int i = 0; i < n; ++i) {
        RigidBodyRecord body;
        body.name = "body" + std::to_string(i);
        body.mass = 1.0f;
        const auto id = scene.AddRigidBody(std::move(body));

        CollisionShapeRecord shape;
        shape.body_id = id;
        shape.type = ShapeType::TriMesh;
        shape.decompose_mode = DecomposeMode::Skip;  // 1 ConvexHull piece each
        shape.mesh_vertices = cube.vertices;
        shape.mesh_indices = cube.indices;
        scene.AddCollisionShape(std::move(shape));
    }
    return scene;
}

TEST(SparseSdfDedup, IdenticalMeshesShareOneSdf) {
    constexpr int kInstances = 5;
    const auto scene = BuildIdenticalCubes(kInstances);
    const auto blob = CookScene(scene);

    // 5 ConvexHull pieces (one per instance), but only ONE unique SDF.
    ASSERT_EQ(blob.convex_geometry.Count(), static_cast<uint32_t>(kInstances));
    EXPECT_EQ(blob.sdfs.Count(), 1u) << "identical meshes must dedup to one SDF";

    // Every piece references the SAME (valid) SDF index.
    ASSERT_EQ(blob.sdfs.piece_sdf_indices.size(), static_cast<size_t>(kInstances));
    const uint32_t shared = blob.sdfs.piece_sdf_indices[0];
    ASSERT_NE(shared, kNoSdf);
    for (int i = 0; i < kInstances; ++i) {
        EXPECT_EQ(blob.sdfs.piece_sdf_indices[i], shared);
    }

    // Dedup + memory stats (R-C ~80MB cap; one small cube is far under).
    uint64_t band_bytes = 0;
    for (uint32_t s = 0; s < blob.sdfs.Count(); ++s) {
        band_bytes += static_cast<uint64_t>(blob.sdfs.key_counts[s]) *
                      (sizeof(uint64_t) + sizeof(float) + 3u * sizeof(float));
    }
    constexpr uint64_t kCap = 80ull * 1024ull * 1024ull;
    std::printf("[dedup] pieces=%u unique_sdfs=%u shared_cells=%u band_bytes=%llu (cap=%lluMB)\n",
                blob.convex_geometry.Count(), blob.sdfs.Count(),
                blob.sdfs.key_counts.empty() ? 0u : blob.sdfs.key_counts[0],
                static_cast<unsigned long long>(band_bytes),
                static_cast<unsigned long long>(kCap / (1024 * 1024)));
    EXPECT_LT(band_bytes, kCap);
}

TEST(SparseSdfDedup, SignCorrectOnCookPathConvexPiece) {
    // The Skip-mode cube is exactly the ConvexHull geometry the cook path feeds
    // the SDF cooker. Verify sign on this shipping-path piece.
    const auto scene = BuildIdenticalCubes(1);
    const auto blob = CookScene(scene);
    ASSERT_EQ(blob.sdfs.Count(), 1u);

    // Reconstruct a SparseSdfDevice view over the (single) cooked SDF slice.
    const auto& T = blob.sdfs;
    nuka::runtime::sdf::SparseSdfDevice dev;
    dev.origin = T.origins[0];
    dev.voxel_size = T.voxel_sizes[0];
    dev.dims[0] = T.dims_x[0];
    dev.dims[1] = T.dims_y[0];
    dev.dims[2] = T.dims_z[0];
    const uint32_t off = T.key_offsets[0];
    dev.cell_keys = T.cell_keys.data() + off;
    dev.cell_values = T.cell_values.data() + off;
    dev.cell_gradients = T.cell_gradients.data() + off;
    dev.cell_count = T.key_counts[0];

    nuka::math::Vec3 g;
    // Cube spans [-0.5,0.5]^3. Interior point near a face but inside -> phi<0.
    const float voxel = dev.voxel_size;
    const float phi_in = nuka::runtime::sdf::sparse_sdf_sample(dev, {0.5f - voxel, 0.0f, 0.0f}, g);
    EXPECT_LT(phi_in, 0.0f);
    const float phi_out = nuka::runtime::sdf::sparse_sdf_sample(dev, {0.5f + voxel, 0.0f, 0.0f}, g);
    EXPECT_GT(phi_out, 0.0f);
    EXPECT_GT(g.x, 0.5f);  // outward +X gradient
}

}  // namespace
