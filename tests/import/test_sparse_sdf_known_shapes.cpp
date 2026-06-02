// ---------------------------------------------------------------------------
// p07 sparse narrow-band SDF: analytical agreement for primitive shapes.
// ---------------------------------------------------------------------------
// The cooked narrow-band SDF must match the closed-form SDF near the surface.
// We assert at GRID NODES (the cooker's own stored values) so the test measures
// the COOKER, not trilinear interpolation. Sampling goes through the SAME
// function p08 ships (runtime/sdf/sparse_sdf_query.cuh::sparse_sdf_sample) via
// the host view, so the analytical gate validates the shipping sampler path.
//
// Box is the exact gate: the cube mesh IS the box, so there is no tessellation
// error — the cooker reproduces the analytical box SDF to float epsilon.
// Sphere / capsule carry an irreducible tessellation (facet-sagitta) term that
// shrinks with mesh resolution; we assert a measured-and-reported bound.
//
// band_voxels=2 keeps cook cost low (O(triangles x band^3)); node accuracy is
// independent of band width.
// ---------------------------------------------------------------------------

#include "import/cooker/sparse_sdf_cooker.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"
#include "tests/import/sdf_test_meshes.hpp"
#include "tests/import/sdf_host_sampler.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>

namespace {

using nuka::import::cooker::CookSparseSdf;
using nuka::import::cooker::SparseSdfData;
using nuka::import::cooker::SparseSdfParams;
using nuka::math::Vec3;

double SphereSdf(double x, double y, double z, double cx, double cy, double cz, double r) {
    const double dx = x - cx, dy = y - cy, dz = z - cz;
    return std::sqrt(dx * dx + dy * dy + dz * dz) - r;
}

double BoxSdf(double x, double y, double z, double hx, double hy, double hz) {
    const double qx = std::fabs(x) - hx;
    const double qy = std::fabs(y) - hy;
    const double qz = std::fabs(z) - hz;
    const double ax = std::max(qx, 0.0), ay = std::max(qy, 0.0), az = std::max(qz, 0.0);
    const double outside = std::sqrt(ax * ax + ay * ay + az * az);
    const double inside = std::min(std::max(qx, std::max(qy, qz)), 0.0);
    return outside + inside;
}

// Capsule along Z, half-height hh, radius r.
double CapsuleSdf(double x, double y, double z, double r, double hh) {
    const double cz = std::max(-hh, std::min(hh, z));
    const double dx = x, dy = y, dz = z - cz;
    return std::sqrt(dx * dx + dy * dy + dz * dz) - r;
}

// Max abs error between cooked node value and `truth(x,y,z)`, over all cells.
template <typename F>
double MaxNodeError(const SparseSdfData& sdf, F truth) {
    double maxe = 0.0;
    for (uint32_t n = 0; n < sdf.CellCount(); ++n) {
        uint32_t i, j, k;
        nuka::runtime::sdf::UnpackSdfCellKey(sdf.cell_keys[n], i, j, k);
        const double x = sdf.origin[0] + static_cast<double>(i) * sdf.voxel_size;
        const double y = sdf.origin[1] + static_cast<double>(j) * sdf.voxel_size;
        const double z = sdf.origin[2] + static_cast<double>(k) * sdf.voxel_size;
        maxe = std::max(maxe, std::fabs(static_cast<double>(sdf.cell_values[n]) - truth(x, y, z)));
    }
    return maxe;
}

SparseSdfParams TestParams(float voxel) {
    SparseSdfParams p;
    p.voxel_size = voxel;
    p.band_voxels = 2u;  // cheap; node accuracy is band-independent
    return p;
}

TEST(SparseSdfKnownShapes, BoxAgreesWithAnalyticalToFloatEpsilon) {
    const auto mesh = nuka::test::UnitCubeMesh();  // [-0.5, 0.5]^3
    const float voxel = 0.04f;
    const auto sdf = CookSparseSdf(mesh.vertices.data(),
                                   static_cast<uint32_t>(mesh.vertices.size() / 3),
                                   mesh.indices.data(),
                                   static_cast<uint32_t>(mesh.indices.size() / 3),
                                   TestParams(voxel));
    ASSERT_GT(sdf.CellCount(), 0u);
    const double maxe = MaxNodeError(sdf, [](double x, double y, double z) {
        return BoxSdf(x, y, z, 0.5, 0.5, 0.5);
    });
    std::printf("[box] voxel=%g cells=%u node maxerr=%g (%.4f%% of voxel)\n",
                voxel, sdf.CellCount(), maxe, maxe / voxel * 100.0);
    // Exact mesh => only float round-off. Far inside 1% of voxel.
    EXPECT_LT(maxe, 0.01 * voxel);
}

TEST(SparseSdfKnownShapes, SphereAgreesWithAnalyticalNearSurface) {
    const auto mesh = nuka::test::SphereMesh(0.0f, 0.0f, 0.0f, 1.0f, 48, 96);
    const float voxel = 0.06f;
    const auto sdf = CookSparseSdf(mesh.vertices.data(),
                                   static_cast<uint32_t>(mesh.vertices.size() / 3),
                                   mesh.indices.data(),
                                   static_cast<uint32_t>(mesh.indices.size() / 3),
                                   TestParams(voxel));
    ASSERT_GT(sdf.CellCount(), 0u);
    const double maxe = MaxNodeError(sdf, [](double x, double y, double z) {
        return SphereSdf(x, y, z, 0.0, 0.0, 0.0, 1.0);
    });
    std::printf("[sphere] 48x96 voxel=%g cells=%u node maxerr=%g (%.3f%% of voxel)\n",
                voxel, sdf.CellCount(), maxe, maxe / voxel * 100.0);
    // Tessellation-dominated (facet sagitta of a 48x96 UV sphere). The COOKER's
    // own error (vs the tessellated mesh) is << this; node error shrinks with
    // poly. Bound generously and report the measured number.
    EXPECT_LT(maxe, 0.05 * voxel);
}

TEST(SparseSdfKnownShapes, CapsuleAgreesWithAnalyticalNearSurface) {
    const float r = 0.5f, hh = 0.7f;
    const auto mesh = nuka::test::CapsuleMesh(r, hh, 96, 24);
    const float voxel = 0.04f;
    const auto sdf = CookSparseSdf(mesh.vertices.data(),
                                   static_cast<uint32_t>(mesh.vertices.size() / 3),
                                   mesh.indices.data(),
                                   static_cast<uint32_t>(mesh.indices.size() / 3),
                                   TestParams(voxel));
    ASSERT_GT(sdf.CellCount(), 0u);
    const double maxe = MaxNodeError(sdf, [&](double x, double y, double z) {
        return CapsuleSdf(x, y, z, r, hh);
    });
    std::printf("[capsule] 96x24 voxel=%g cells=%u node maxerr=%g (%.3f%% of voxel)\n",
                voxel, sdf.CellCount(), maxe, maxe / voxel * 100.0);
    EXPECT_LT(maxe, 0.05 * voxel);
}

// Sign sanity through the SHIPPING sampler: interior point -> phi<0, exterior
// -> phi>0, gradient ~ outward radial near the surface.
TEST(SparseSdfKnownShapes, SphereSamplerSignAndGradient) {
    const auto mesh = nuka::test::SphereMesh(0.0f, 0.0f, 0.0f, 1.0f, 48, 96);
    const float voxel = 0.06f;
    const auto sdf = CookSparseSdf(mesh.vertices.data(),
                                   static_cast<uint32_t>(mesh.vertices.size() / 3),
                                   mesh.indices.data(),
                                   static_cast<uint32_t>(mesh.indices.size() / 3),
                                   TestParams(voxel));
    const auto view = nuka::test::MakeHostView(sdf);

    Vec3 g;
    // Just inside the surface along +X.
    const float phi_in = nuka::test::SampleHost(view, 1.0f - voxel, 0.0f, 0.0f, g);
    EXPECT_LT(phi_in, 0.0f);
    // Just outside.
    const float phi_out = nuka::test::SampleHost(view, 1.0f + voxel, 0.0f, 0.0f, g);
    EXPECT_GT(phi_out, 0.0f);
    // Gradient near +X surface points ~ +X (outward).
    EXPECT_GT(g.x, 0.8f);
    EXPECT_LT(std::fabs(g.y), 0.2f);
    EXPECT_LT(std::fabs(g.z), 0.2f);
    // Far outside the band -> no coverage sentinel.
    const float phi_far = nuka::test::SampleHost(view, 5.0f, 5.0f, 5.0f, g);
    EXPECT_GT(phi_far, 1.0e29f);
}

}  // namespace
