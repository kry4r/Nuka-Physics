// ---------------------------------------------------------------------------
// p07 sparse narrow-band SDF: thin-shell preservation (R-E).
// ---------------------------------------------------------------------------
// A 0.5 mm-thick panel cooked at a 0.1 mm voxel must be resolved on BOTH sides:
//   * interior of the panel samples phi < 0 (negative, inside the slab),
//   * both faces sample phi > 0 outside,
//   * the SDF gradient FLIPS sign across the panel (+Z normal on the +Z face,
//     -Z normal on the -Z face) — the slab is genuinely two-sided, not bled
//     through. Exact nearest-triangle distance (no fast sweeping) is what makes
//     a sub-voxel-resolved two-sided thin shell possible.
//
// All in meters: 0.5 mm = 5e-4, 0.1 mm voxel = 1e-4. The panel is large in x,y
// and thin in z (half_thick = 2.5e-4).
// ---------------------------------------------------------------------------

#include "import/cooker/sparse_sdf_cooker.hpp"
#include "tests/import/sdf_test_meshes.hpp"
#include "tests/import/sdf_host_sampler.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>

namespace {

using nuka::import::cooker::CookSparseSdf;
using nuka::import::cooker::SparseSdfParams;
using nuka::math::Vec3;

TEST(SparseSdfThinShell, PanelResolvedBothSidesGradientFlips) {
    const float half_thick = 2.5e-4f;  // 0.5 mm panel
    const float voxel = 1.0e-4f;        // 0.1 mm voxel (panel = 5 voxels thick)
    // Keep the panel modest in extent so the cook stays cheap (few triangles,
    // but many band voxels across the faces -> use band_voxels=2).
    const auto mesh = nuka::test::ThinPanelMesh(2.0e-3f, 2.0e-3f, half_thick);

    SparseSdfParams p;
    p.voxel_size = voxel;
    // band=4 (production default) so the band covers the panel's 2.5-voxel-deep
    // center (|phi|<=4*voxel). The panel stays sub-(2*band) thick, so it remains
    // a genuine thin-shell test (a dense grid would still resolve only ~5 voxels
    // across the slab).
    p.band_voxels = 4u;
    const auto sdf = CookSparseSdf(mesh.vertices.data(),
                                   static_cast<uint32_t>(mesh.vertices.size() / 3),
                                   mesh.indices.data(),
                                   static_cast<uint32_t>(mesh.indices.size() / 3),
                                   p);
    ASSERT_GT(sdf.CellCount(), 0u);
    const auto view = nuka::test::MakeHostView(sdf);

    Vec3 g_center, g_plus, g_minus, g_in_plus, g_in_minus;

    // Center of the slab (z=0): deepest interior -> negative, ~ -half_thick.
    // Tolerance 1 voxel: the medial axis falls between grid nodes (the panel is
    // an odd 5 voxels thick), so trilinear interp at z=0 reads slightly shallower
    // than the true -half_thick. The SIGN (inside) is the load-bearing check.
    const float phi_center = nuka::test::SampleHost(view, 0.0f, 0.0f, 0.0f, g_center);
    EXPECT_LT(phi_center, 0.0f) << "panel interior must be inside (phi<0)";
    EXPECT_NEAR(phi_center, -half_thick, 1.0f * voxel);

    // Interior near the +Z face (still inside): gradient points +Z (toward the
    // nearer face); near the -Z face it points -Z. This is the cross-panel flip.
    const float phi_ip = nuka::test::SampleHost(view, 0.0f, 0.0f, half_thick - voxel, g_in_plus);
    const float phi_im = nuka::test::SampleHost(view, 0.0f, 0.0f, -(half_thick - voxel), g_in_minus);
    EXPECT_LT(phi_ip, 0.0f);
    EXPECT_LT(phi_im, 0.0f);
    EXPECT_GT(g_in_plus.z, 0.5f)  << "interior gradient near +Z face must point +Z";
    EXPECT_LT(g_in_minus.z, -0.5f) << "interior gradient near -Z face must point -Z";

    // Just outside each face: phi>0, outward gradient on the correct side.
    const float phi_plus = nuka::test::SampleHost(view, 0.0f, 0.0f, half_thick + voxel, g_plus);
    const float phi_minus = nuka::test::SampleHost(view, 0.0f, 0.0f, -(half_thick + voxel), g_minus);
    EXPECT_GT(phi_plus, 0.0f);
    EXPECT_GT(phi_minus, 0.0f);
    EXPECT_GT(g_plus.z, 0.5f)  << "+Z exterior gradient must point +Z";
    EXPECT_LT(g_minus.z, -0.5f) << "-Z exterior gradient must point -Z";

    std::printf("[thin-shell] panel=%.1emm voxel=%.1emm cells=%u  "
                "phi_center=%.2emm g_in+.z=%.2f g_in-.z=%.2f\n",
                static_cast<double>(2.0f * half_thick) * 1e3,
                static_cast<double>(voxel) * 1e3, sdf.CellCount(),
                static_cast<double>(phi_center) * 1e3,
                g_in_plus.z, g_in_minus.z);
}

}  // namespace
