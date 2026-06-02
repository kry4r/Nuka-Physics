// ---------------------------------------------------------------------------
// p07 sparse narrow-band SDF: narrow-band memory savings vs dense grid.
// ---------------------------------------------------------------------------
// Narrow-band storage must be << a dense grid. The spec's "~10%" (Task 7.7.4:
// "50K of 1M voxels = 5%, 16 bytes/voxel") is a VOXEL-COUNT ratio with NO
// per-cell key, so the apples-to-apples metric is voxel_ratio = band_cells /
// dense_voxels. We assert that < 10% and ALSO report byte_ratio, which adds this
// sorted-key layout's 8-byte key on top of value(4)+gradient(12): byte_ratio =
// 24/16 * voxel_ratio ~ 1.5x.
//
// The band fraction ~ C*N/(L/h) where C ~ 2*pi for a compact (sphere) object
// and ~12 for a cube (6 flat faces). So we use a SPHERE (compact, matches the
// spec's implicit model), band_voxels=2, voxel ~ diameter/200 -> object ~200
// voxels across. Low poly keeps the cook cheap (band-cell count, not accuracy,
// drives this test).
// ---------------------------------------------------------------------------

#include "import/cooker/sparse_sdf_cooker.hpp"
#include "tests/import/sdf_test_meshes.hpp"

#include <gtest/gtest.h>

#include <cstdio>

namespace {

using nuka::import::cooker::CookSparseSdf;
using nuka::import::cooker::SparseSdfParams;

TEST(SparseSdfMemory, NarrowBandIsFractionOfDense) {
    // Sphere radius 1 => diameter 2. voxel = 2/200 => ~200 voxels across.
    const auto mesh = nuka::test::SphereMesh(0.0f, 0.0f, 0.0f, 1.0f, 32, 64);
    SparseSdfParams p;
    p.voxel_size = 2.0f / 200.0f;
    p.band_voxels = 2u;
    const auto sdf = CookSparseSdf(mesh.vertices.data(),
                                   static_cast<uint32_t>(mesh.vertices.size() / 3),
                                   mesh.indices.data(),
                                   static_cast<uint32_t>(mesh.indices.size() / 3),
                                   p);
    ASSERT_GT(sdf.CellCount(), 0u);

    const uint64_t dense_voxels = sdf.DenseVoxelCount();
    const uint64_t band_bytes = sdf.NarrowBandBytes();
    const uint64_t dense_bytes = sdf.DenseBytes();
    const double voxel_ratio = static_cast<double>(sdf.CellCount()) / static_cast<double>(dense_voxels);
    const double byte_ratio = static_cast<double>(band_bytes) / static_cast<double>(dense_bytes);

    std::printf("[memory] sphere voxel=%g dims=%u,%u,%u dense=%llu cells=%u\n"
                "         voxel_ratio=%.4f (spec ~10%%, key-free)  byte_ratio=%.4f "
                "(band=%lluB dense=%lluB; +8B key/cell => ~1.5x voxel_ratio)\n",
                p.voxel_size, sdf.dims[0], sdf.dims[1], sdf.dims[2],
                static_cast<unsigned long long>(dense_voxels), sdf.CellCount(),
                voxel_ratio, byte_ratio,
                static_cast<unsigned long long>(band_bytes),
                static_cast<unsigned long long>(dense_bytes));

    // Spec's "~10%" is the voxel-count ratio (no per-cell key). Assert that.
    EXPECT_LT(voxel_ratio, 0.10);
}

}  // namespace
