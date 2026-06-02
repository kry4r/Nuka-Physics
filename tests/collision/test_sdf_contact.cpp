// ---------------------------------------------------------------------------
// p08-B: Newton-on-summed-SDF contact -- forward correctness + D1.
// ---------------------------------------------------------------------------
// Cooks (p07) a sphere SDF + a box SDF, places them at a known overlap, and:
//   1. ContactAccuracyVsAnalytical: find_sdf_contact_newton recovers the
//      analytical sphere-box penetration depth + SIGNED contact normal
//      (voxel-limited). The summed-SDF minimum is a FLAT valley (the point is
//      non-unique along the overlap segment), so the depth + normal are the
//      load-bearing checks; the witness point is asserted loosely (on-axis) with
//      the initial guess seeded squarely in the overlap lens.
//   2. RowSolvePushesApart: emit RigidSDFContactRow rows for a penetrating pair
//      and run the EXISTING PGS solve (SolveConstraints) -- the dynamic body's
//      separation-axis velocity must increase (it moves APART). This is the
//      check that catches a flipped normal (a flip pins lambda to 0 -> no
//      motion), so we assert real motion, not lambda>0.
//   3. ThinShell: a 0.5mm panel at 0.1mm voxel -> contact detected from inside
//      the band, the per-step clamp <= voxel keeps the iterate from tunneling.
//   4. DeterminismTwoRunByteExact: same query twice -> bit-identical contact
//      output AND bit-identical emitted row buffers.
//   5. FeatherstoneSdfRowConstruction: id-5 emission + D1 (construction only;
//      articulated separation dynamics are exercised elsewhere).
//
// find_sdf_contact_newton is the header-inline NUKA_SDF_HD function the device
// kernel also wraps; calling it here on the host validates the SHIPPING path.
// ---------------------------------------------------------------------------

#include "collision/sdf_contact.hpp"

#include "constraint/row_buffers.hpp"
#include "import/cooker/sparse_sdf_cooker.hpp"
#include "runtime/rigid/body_state.hpp"
#include "solver/rigid_solver.hpp"
#include "tests/import/sdf_host_sampler.hpp"
#include "tests/import/sdf_test_meshes.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

using nuka::collision::AppendFeatherstoneSdfContactGroup;
using nuka::collision::AppendSdfContactGroup;
using nuka::collision::find_sdf_contact_newton;
using nuka::collision::SdfBodyFrame;
using nuka::collision::SdfContactParams;
using nuka::collision::SdfContactResult;
using nuka::import::cooker::CookSparseSdf;
using nuka::import::cooker::SparseSdfData;
using nuka::import::cooker::SparseSdfParams;
using nuka::math::Vec3;

SparseSdfData CookBox(float voxel, float band_voxels) {
    const auto mesh = nuka::test::UnitCubeMesh();  // [-0.5, 0.5]^3
    SparseSdfParams p;
    p.voxel_size = voxel;
    p.band_voxels = static_cast<uint32_t>(band_voxels);
    return CookSparseSdf(mesh.vertices.data(),
                         static_cast<uint32_t>(mesh.vertices.size() / 3),
                         mesh.indices.data(),
                         static_cast<uint32_t>(mesh.indices.size() / 3), p);
}

SparseSdfData CookSphere(float r, float voxel, float band_voxels) {
    const auto mesh = nuka::test::SphereMesh(0.0f, 0.0f, 0.0f, r, 48, 96);
    SparseSdfParams p;
    p.voxel_size = voxel;
    p.band_voxels = static_cast<uint32_t>(band_voxels);
    return CookSparseSdf(mesh.vertices.data(),
                         static_cast<uint32_t>(mesh.vertices.size() / 3),
                         mesh.indices.data(),
                         static_cast<uint32_t>(mesh.indices.size() / 3), p);
}

// Identity-rotation frame at a world translation.
SdfBodyFrame FrameAt(Vec3 t) {
    SdfBodyFrame f;
    f.rot_col0 = Vec3{1.0f, 0.0f, 0.0f};
    f.rot_col1 = Vec3{0.0f, 1.0f, 0.0f};
    f.rot_col2 = Vec3{0.0f, 0.0f, 1.0f};
    f.translation = t;
    return f;
}

bool SameResult(const SdfContactResult& a, const SdfContactResult& b) {
    return std::memcmp(&a, &b, sizeof(SdfContactResult)) == 0;
}

}  // namespace

// 1. Contact accuracy vs analytical sphere-box.
//    Box A = unit cube at origin (its +X face at x=0.5). Sphere B = r=0.3 at
//    x=0.7, so its -X surface reaches x=0.4: it overlaps the box face by 0.1.
//    Analytical: penetration depth = 0.1; separation normal for A (the box) is
//    -X (the box must move -X to escape the sphere on its +X side).
TEST(SdfContact, ContactAccuracyVsAnalytical) {
    const float voxel = 0.02f;
    const float band = 4.0f;
    const float r = 0.3f;
    const float sphere_x = 0.7f;  // overlap 0.1 with the box +X face at 0.5

    const auto box = CookBox(voxel, band);
    const auto sphere = CookSphere(r, voxel, band);
    ASSERT_GT(box.CellCount(), 0u);
    ASSERT_GT(sphere.CellCount(), 0u);
    const auto box_view = nuka::test::MakeHostView(box);
    const auto sphere_view = nuka::test::MakeHostView(sphere);

    const SdfBodyFrame fa = FrameAt(Vec3{0.0f, 0.0f, 0.0f});       // box A
    const SdfBodyFrame fb = FrameAt(Vec3{sphere_x, 0.0f, 0.0f});   // sphere B

    SdfContactParams params;
    params.max_iterations = 48u;
    // Seed squarely in the overlap lens (midway: x=0.45, y=z=0) -- inside both
    // narrow bands and on the flat valley.
    const Vec3 guess{0.45f, 0.0f, 0.0f};

    const auto c = find_sdf_contact_newton(box_view.device, fa,
                                           sphere_view.device, fb, guess, params);
    ASSERT_TRUE(c.valid);
    ASSERT_TRUE(c.penetrating);

    const float expected_depth = 0.1f;
    const float depth_err = std::fabs(c.penetration - expected_depth);
    // Voxel-limited (trilinear + cook + tessellation of the sphere). Bound ~1.5
    // voxels and report.
    EXPECT_LT(depth_err, 1.5f * voxel);

    // Signed normal: separation direction for the BOX (A) is -X. Catch a flip.
    EXPECT_LT(c.normal.x, -0.9f) << "box separation normal must point -X";
    EXPECT_LT(std::fabs(c.normal.y), 0.15f);
    EXPECT_LT(std::fabs(c.normal.z), 0.15f);

    // Point loosely on-axis (the valley is flat in x; y,z pinned near 0).
    EXPECT_LT(std::fabs(c.point.y), 2.0f * voxel);
    EXPECT_LT(std::fabs(c.point.z), 2.0f * voxel);
    EXPECT_GT(c.point.x, 0.4f - 2.0f * voxel);
    EXPECT_LT(c.point.x, 0.5f + 2.0f * voxel);

    std::printf("[accuracy] depth=%.4f (exp %.4f, err %.4f = %.2f vox)  "
                "normal=(%.3f,%.3f,%.3f)  point=(%.3f,%.3f,%.3f)  iters=%u\n",
                c.penetration, expected_depth, depth_err, depth_err / voxel,
                c.normal.x, c.normal.y, c.normal.z,
                c.point.x, c.point.y, c.point.z, c.iterations);
}

// 1b. Newton DESCENT from an off-contact seed. The summed-SDF minimum is FLAT
//     along the penetration axis (the witness is non-unique there) but it is NOT
//     flat OFF-AXIS: away from the box-sphere centerline the sphere gradient
//     tilts, so grad_a+grad_b != 0 and the solve must actually descend. Seeding
//     off-axis (y offset) thus exercises the step direction + per-step clamp the
//     seed-on-answer tests skip. We assert the descent pulls the witness BACK
//     toward the centerline (y -> ~0) and recovers depth + signed normal.
//
//     HONEST NOTE: the step is fixed-magnitude gradient descent (kappa = 2/voxel,
//     no true Hessian -- the field has none along the flat valley), so it
//     converges slowly and bottoms out at the cooked-gradient interpolation noise
//     floor (residual ~5e-5 > the 1e-5 tol), running all max_iterations. That is
//     a property of the scheme, surfaced here, not a test artifact.
TEST(SdfContact, NewtonDescentFromOffContactSeed) {
    const float voxel = 0.02f;
    const float band = 6.0f;
    const float r = 0.3f;
    const auto box = CookBox(voxel, band);
    const auto sphere = CookSphere(r, voxel, band);
    const auto box_view = nuka::test::MakeHostView(box);
    const auto sphere_view = nuka::test::MakeHostView(sphere);
    const SdfBodyFrame fa = FrameAt(Vec3{0.0f, 0.0f, 0.0f});
    const SdfBodyFrame fb = FrameAt(Vec3{0.7f, 0.0f, 0.0f});

    SdfContactParams params;
    params.max_iterations = 200u;
    // Off the centerline (y = 4 voxels) but inside both narrow bands: a real
    // residual (~0.4) there, so the solve must descend toward y -> 0.
    const float seed_y = 4.0f * voxel;  // 0.08
    const Vec3 guess{0.46f, seed_y, 0.0f};

    const auto c = find_sdf_contact_newton(box_view.device, fa,
                                           sphere_view.device, fb, guess, params);
    ASSERT_TRUE(c.valid);
    ASSERT_TRUE(c.penetrating);
    // It TOOK steps (not a seed-on-answer no-op).
    EXPECT_GT(c.iterations, 1u) << "off-seed must require descent steps";
    // Descended back toward the centerline: |y| shrank well below the seed.
    EXPECT_LT(std::fabs(c.point.y), 0.5f * seed_y) << "descent must reduce off-axis offset";
    // Depth + signed normal recovered at the valley.
    EXPECT_LT(std::fabs(c.penetration - 0.1f), 2.0f * voxel);
    EXPECT_LT(c.normal.x, -0.85f) << "box separation normal must point -X";
    // Clamp held: each step <= voxel, so the witness never leapt outside the
    // overlap lens (x in [0.4-,0.5+]).
    EXPECT_GT(c.point.x, 0.4f - 3.0f * voxel);
    EXPECT_LT(c.point.x, 0.5f + 3.0f * voxel);
    std::printf("[descent] guess=(%.3f,%.3f,%.3f) -> point=(%.3f,%.3f,%.3f)  "
                "depth=%.4f normal.x=%.3f iters=%u (residual=%.2e)\n",
                guess.x, guess.y, guess.z, c.point.x, c.point.y, c.point.z,
                c.penetration, c.normal.x, c.iterations, c.residual);
}

// 2. Row solve pushes apart. Emit rigid-SDF contact rows for the same overlap;
//    body A (box) is dynamic, body B (sphere) is static (inv_mass=0). Run the
//    EXISTING PGS solve and assert A's velocity along the separation normal
//    increases -> it moves apart. A flipped normal pins lambda to 0 (no motion).
TEST(SdfContact, RowSolvePushesApart) {
    const float voxel = 0.02f;
    const float band = 4.0f;
    const float r = 0.3f;
    const auto box = CookBox(voxel, band);
    const auto sphere = CookSphere(r, voxel, band);
    const auto box_view = nuka::test::MakeHostView(box);
    const auto sphere_view = nuka::test::MakeHostView(sphere);

    const SdfBodyFrame fa = FrameAt(Vec3{0.0f, 0.0f, 0.0f});
    const SdfBodyFrame fb = FrameAt(Vec3{0.7f, 0.0f, 0.0f});
    SdfContactParams params;
    params.max_iterations = 48u;
    const auto c = find_sdf_contact_newton(box_view.device, fa,
                                           sphere_view.device, fb,
                                           Vec3{0.45f, 0.0f, 0.0f}, params);
    ASSERT_TRUE(c.valid && c.penetrating);

    nuka::constraint::RowBuffers rows;
    const uint32_t emitted =
        AppendSdfContactGroup(0u, 1u, c, /*friction=*/0.0f, /*restitution=*/0.0f,
                              &rows);
    EXPECT_EQ(emitted, 3u);  // 1 normal + 2 friction
    EXPECT_EQ(rows.rows[0].row_class_id,
              nuka::constraint::kRigidSdfContactRowClassId);

    const Vec3 n = c.normal;  // box separation normal (~ -X)

    std::vector<nuka::runtime::rigid::BodyState> bodies(2);
    bodies[0].inv_mass = 1.0f;             // box A dynamic
    bodies[0].inv_inertia = {6.0f, 6.0f, 6.0f};
    bodies[0].position = {0.0f, 0.0f, 0.0f};
    // Approaching velocity along -n (INTO the sphere): jv<0 so the velocity solve
    // must produce a positive normal impulse that removes the approach -- a
    // flipped normal would instead clamp lambda to 0 and leave the approach.
    bodies[0].linear_velocity = n * -0.5f;
    bodies[1].inv_mass = 0.0f;             // sphere B static
    bodies[1].position = {0.7f, 0.0f, 0.0f};

    const Vec3 pos_before = bodies[0].position;
    const float vel_before = bodies[0].linear_velocity.x * n.x +
                             bodies[0].linear_velocity.y * n.y +
                             bodies[0].linear_velocity.z * n.z;  // = -0.5 (approaching)

    nuka::solver::SolverConfig cfg;
    cfg.velocity_iterations = 16u;
    cfg.position_iterations = 8u;
    cfg.baumgarte = 0.5f;
    nuka::solver::SolveConstraints(rows, bodies, cfg);

    const Vec3 pos_after = bodies[0].position;
    const Vec3 disp = pos_after - pos_before;
    const float along = disp.x * n.x + disp.y * n.y + disp.z * n.z;
    const float vel_after = bodies[0].linear_velocity.x * n.x +
                            bodies[0].linear_velocity.y * n.y +
                            bodies[0].linear_velocity.z * n.z;
    std::printf("[pushapart] disp=(%.4f,%.4f,%.4f)  along-normal=%.4f  "
                "lambda=%.4f  v.n before=%.3f after=%.3f\n",
                disp.x, disp.y, disp.z, along, rows.rows[0].lambda,
                vel_before, vel_after);
    // Position-solve: the box must move in the separation direction.
    EXPECT_GT(along, 1.0e-4f) << "penetrating bodies must be pushed APART";
    // Velocity-solve: the approach (v.n<0) must be removed (v.n driven to >=0),
    // and a real normal impulse must have been applied (lambda>0). Both fail with
    // a flipped contact normal.
    EXPECT_GT(vel_after, vel_before) << "approach velocity must be reduced";
    EXPECT_GE(vel_after, -1.0e-3f) << "normal approach must be (nearly) removed";
    EXPECT_GT(rows.rows[0].lambda, 0.0f) << "a separating normal impulse must be applied";
}

// 3. Thin-shell: a 0.5mm panel cooked at 0.1mm voxel pressed by a (curved)
//    sphere. Two sub-claims of deliverable #4:
//    (a) DETECTION at 0.1mm voxel: a sub-mm panel is resolved and a contact with
//        the opposing body is found from inside the band.
//    (b) NO PUNCH-THROUGH / step clamp holds: the sphere gives a non-flat
//        residual (curvature), so seeding off the contact forces real descent
//        steps; the per-step clamp (<= the SMALLER body's voxel = the panel's
//        0.1mm) bounds each step so the witness cannot tunnel across the panel
//        into its far interior. We assert the converged witness stays on the
//        contact side of the panel (z does NOT cross to the panel's far face).
TEST(SdfContact, ThinShellNoPunchThrough) {
    const float half_thick = 2.5e-4f;   // 0.5 mm panel (half = 0.25 mm)
    const float panel_voxel = 1.0e-4f;  // 0.1 mm voxel (panel = 5 voxels thick)
    const auto panel_mesh = nuka::test::ThinPanelMesh(4.0e-3f, 4.0e-3f, half_thick);
    SparseSdfParams pp;
    pp.voxel_size = panel_voxel;
    pp.band_voxels = 4u;
    const auto panel = CookSparseSdf(panel_mesh.vertices.data(),
                                     static_cast<uint32_t>(panel_mesh.vertices.size() / 3),
                                     panel_mesh.indices.data(),
                                     static_cast<uint32_t>(panel_mesh.indices.size() / 3),
                                     pp);
    ASSERT_GT(panel.CellCount(), 0u);
    const auto panel_view = nuka::test::MakeHostView(panel);

    // Body B: a sphere (radius 1mm) cooked at its own coarser voxel, pressed onto
    // the +Z face of the panel so its -Z surface dips ~1.5 panel-voxels into the
    // panel. The curved sphere SDF makes the summed field non-flat off the z axis,
    // so the solve must descend (exercising the step + clamp).
    const float r = 1.0e-3f;            // 1 mm sphere
    const float sphere_voxel = 1.0e-4f;
    const auto sphere = CookSphere(r, sphere_voxel, 4.0f);
    ASSERT_GT(sphere.CellCount(), 0u);
    const auto sphere_view = nuka::test::MakeHostView(sphere);

    const SdfBodyFrame fa = FrameAt(Vec3{0.0f, 0.0f, 0.0f});  // panel
    // Sphere center so its -Z pole (z = center - r) sits ~1.5 panel-voxels inside
    // the panel's +Z face (at z = +half_thick): center = half_thick + r - 1.5*vox.
    const float sphere_z = half_thick + r - 1.5f * panel_voxel;
    const SdfBodyFrame fb = FrameAt(Vec3{0.0f, 0.0f, sphere_z});  // sphere

    SdfContactParams params;
    params.max_iterations = 128u;
    params.step_clamp_scale = 1.0f;  // step <= the smaller voxel (panel's 0.1mm)
    // Seed off-axis (x,y a few panel-voxels off the pole) AND a bit below the
    // contact, inside both bands, so a real descent toward the witness is needed.
    const Vec3 guess{3.0f * panel_voxel, 2.0f * panel_voxel, half_thick - 1.0f * panel_voxel};

    const auto c = find_sdf_contact_newton(panel_view.device, fa,
                                           sphere_view.device, fb, guess, params);
    ASSERT_TRUE(c.valid) << "thin-shell contact must be detected from inside the band";
    EXPECT_TRUE(c.penetrating);
    // The descent actually moved (off-seed): more than 1 iteration.
    EXPECT_GT(c.iterations, 1u) << "off-seed must require real descent steps";
    // NO PUNCH-THROUGH: the witness stays on the +Z (contact) side of the panel
    // -- the clamped step cannot tunnel it across the 5-voxel slab to z < 0 (the
    // panel's mid-plane) let alone past the far -Z face at z = -half_thick.
    EXPECT_GT(c.point.z, 0.0f)
        << "step clamp must keep the witness on the contact side (no tunneling)";
    std::printf("[thin-shell] panel=%.1emm@%.1emm-vox sphere=%.1emm  "
                "guess.z=%.3emm point=(%.3e,%.3e,%.3e)mm pen=%.3emm iters=%u\n",
                static_cast<double>(2.0f * half_thick) * 1e3,
                static_cast<double>(panel_voxel) * 1e3,
                static_cast<double>(2.0f * r) * 1e3,
                static_cast<double>(guess.z) * 1e3,
                static_cast<double>(c.point.x) * 1e3,
                static_cast<double>(c.point.y) * 1e3,
                static_cast<double>(c.point.z) * 1e3,
                static_cast<double>(c.penetration) * 1e3, c.iterations);
}

// 4. D1: two runs of the same query -> byte-identical contact output AND
//    byte-identical emitted row buffers.
TEST(SdfContact, DeterminismTwoRunByteExact) {
    const float voxel = 0.02f;
    const float band = 4.0f;
    const auto box = CookBox(voxel, band);
    const auto sphere = CookSphere(0.3f, voxel, band);
    const auto box_view = nuka::test::MakeHostView(box);
    const auto sphere_view = nuka::test::MakeHostView(sphere);
    const SdfBodyFrame fa = FrameAt(Vec3{0.0f, 0.0f, 0.0f});
    const SdfBodyFrame fb = FrameAt(Vec3{0.7f, 0.0f, 0.0f});
    SdfContactParams params;
    params.max_iterations = 48u;
    const Vec3 guess{0.45f, 0.0f, 0.0f};

    const auto c1 = find_sdf_contact_newton(box_view.device, fa,
                                            sphere_view.device, fb, guess, params);
    const auto c2 = find_sdf_contact_newton(box_view.device, fa,
                                            sphere_view.device, fb, guess, params);
    ASSERT_TRUE(c1.valid);
    EXPECT_TRUE(SameResult(c1, c2)) << "Newton contact output must be byte-identical";

    nuka::constraint::RowBuffers r1, r2;
    AppendSdfContactGroup(0u, 1u, c1, 0.5f, 0.0f, &r1);
    AppendSdfContactGroup(0u, 1u, c2, 0.5f, 0.0f, &r2);
    ASSERT_EQ(r1.rows.size(), r2.rows.size());
    ASSERT_EQ(r1.jacobian_data.size(), r2.jacobian_data.size());
    EXPECT_EQ(0, std::memcmp(r1.rows.data(), r2.rows.data(),
                             r1.rows.size() * sizeof(nuka::constraint::Row)));
    EXPECT_EQ(0, std::memcmp(r1.jacobian_data.data(), r2.jacobian_data.data(),
                             r1.jacobian_data.size() * sizeof(nuka::constraint::RowJacobian6)));
    std::printf("[D1] two-run byte-exact: contact + %zu rows + %zu jacobians\n",
                r1.rows.size(), r1.jacobian_data.size());
}

// 5. FeatherstoneSDFContactRow (id 5) construction + emission + D1. The chain
//    Jacobian is supplied (evaluated at the contact point, held for the row);
//    p08-B does the minimal correct construction. No articulated solve here.
TEST(SdfContact, FeatherstoneSdfRowConstruction) {
    const float voxel = 0.02f;
    const float band = 4.0f;
    const auto box = CookBox(voxel, band);
    const auto sphere = CookSphere(0.3f, voxel, band);
    const auto box_view = nuka::test::MakeHostView(box);
    const auto sphere_view = nuka::test::MakeHostView(sphere);
    const auto c = find_sdf_contact_newton(box_view.device,
                                           FrameAt(Vec3{0.0f, 0.0f, 0.0f}),
                                           sphere_view.device,
                                           FrameAt(Vec3{0.7f, 0.0f, 0.0f}),
                                           Vec3{0.45f, 0.0f, 0.0f}, {});
    ASSERT_TRUE(c.valid && c.penetrating);

    const Vec3 chain_lin{c.normal.x, c.normal.y, c.normal.z};  // projected normal
    const Vec3 chain_ang{0.0f, 0.0f, 1.0f};                    // a link DOF axis

    nuka::constraint::RowBuffers r1, r2;
    const uint32_t n1 = AppendFeatherstoneSdfContactGroup(
        0u, 1u, c, chain_lin, chain_ang, 0.5f, 0.0f, &r1);
    const uint32_t n2 = AppendFeatherstoneSdfContactGroup(
        0u, 1u, c, chain_lin, chain_ang, 0.5f, 0.0f, &r2);
    EXPECT_EQ(n1, 1u);
    EXPECT_EQ(n2, 1u);
    EXPECT_EQ(r1.rows[0].row_class_id,
              nuka::constraint::kFeatherstoneSdfContactRowClassId);
    // D1: identical construction.
    ASSERT_EQ(r1.rows.size(), r2.rows.size());
    EXPECT_EQ(0, std::memcmp(r1.rows.data(), r2.rows.data(),
                             r1.rows.size() * sizeof(nuka::constraint::Row)));
    EXPECT_EQ(0, std::memcmp(r1.jacobian_data.data(), r2.jacobian_data.data(),
                             r1.jacobian_data.size() * sizeof(nuka::constraint::RowJacobian6)));
    std::printf("[id5] FeatherstoneSDFContactRow emitted, class_id=%u, D1 ok\n",
                r1.rows[0].row_class_id);
}
