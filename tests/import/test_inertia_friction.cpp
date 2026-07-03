// ---------------------------------------------------------------------------
// MJCF fullinertia diagonalization + joint frictionloss import + nks roundtrip.
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"
#include "scene/format/nks.hpp"
#include "math/quat.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <string>

namespace {

// Row-major rotation from a w-first quaternion (matches articulation_state
// RotationMatrixFromQuat, the convention RotateDiagonalInertia consumes).
void RotMat(const nuka::math::Quat& q, double R[3][3]) {
    const double w = q.w, x = q.x, y = q.y, z = q.z;
    R[0][0] = 1 - 2 * (y * y + z * z); R[0][1] = 2 * (x * y - w * z); R[0][2] = 2 * (x * z + w * y);
    R[1][0] = 2 * (x * y + w * z); R[1][1] = 1 - 2 * (x * x + z * z); R[1][2] = 2 * (y * z - w * x);
    R[2][0] = 2 * (x * z - w * y); R[2][1] = 2 * (y * z + w * x); R[2][2] = 1 - 2 * (x * x + y * y);
}

// Reconstruct R * diag(d) * R^T (the spatial-inertia rebuild the engine performs).
void Reconstruct(const nuka::math::Quat& q, const nuka::math::Vec3& d, double M[3][3]) {
    double R[3][3];
    RotMat(q, R);
    const double dd[3] = {d.x, d.y, d.z};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double v = 0.0;
            for (int k = 0; k < 3; ++k) v += R[i][k] * dd[k] * R[j][k];
            M[i][j] = v;
        }
}

}  // namespace

// fullinertia="Ixx Iyy Izz Ixy Ixz Iyz" = 2 3 4 0.5 0.3 0.2 must diagonalize into
// principal moments + an inertial-frame rotation that rebuilds the ORIGINAL tensor.
TEST(InertiaFriction, FullInertiaDiagonalizesAndReconstructs) {
    const auto scene = nuka::import::LoadMjcf("tests/data/inertia_friction.xml");
    ASSERT_GE(scene.RigidBodyCount(), 2u);
    const auto& link = scene.GetBody(1);  // link1 carries the fullinertia.

    double M[3][3];
    Reconstruct(link.inertial_transform.rotation, link.inertia, M);
    const double expect[3][3] = {{2.0, 0.5, 0.3}, {0.5, 3.0, 0.2}, {0.3, 0.2, 4.0}};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            EXPECT_NEAR(M[i][j], expect[i][j], 1e-4) << "M[" << i << "][" << j << "]";

    // A tensor with off-diagonal products has a non-identity principal frame.
    const auto& q = link.inertial_transform.rotation;
    const double off = std::abs(q.x) + std::abs(q.y) + std::abs(q.z);
    EXPECT_GT(off, 1e-3);
    // COM offset is preserved verbatim (pos="0.01 0.02 0.1").
    EXPECT_NEAR(link.inertial_transform.position.z, 0.1f, 1e-6);
}

// frictionloss flows from a <default> class (j1) and an element attr overrides it
// (j2 = 0.02 over the inherited class 0.075), mirroring the BDX sts3215 class path.
TEST(InertiaFriction, JointFrictionlossParsed) {
    const auto scene = nuka::import::LoadMjcf("tests/data/inertia_friction.xml");
    ASSERT_GE(scene.JointCount(), 2u);
    const auto& j1 = scene.GetJoint(0);
    EXPECT_NEAR(j1.frictionloss, 0.075f, 1e-6);   // inherited from class "act".
    EXPECT_NEAR(j1.damping, 0.3f, 1e-6);
    EXPECT_NEAR(j1.armature, 0.01f, 1e-6);
    const auto& j2 = scene.GetJoint(1);
    EXPECT_NEAR(j2.frictionloss, 0.02f, 1e-6);     // element attr overrides the class.
}

// The diagonalized inertia (principal diag + inertial rotation) and frictionloss
// survive an nks Save/Load round-trip byte-for-value.
TEST(InertiaFriction, NksRoundTripPreservesInertiaAndFriction) {
    const auto scene = nuka::import::LoadMjcf("tests/data/inertia_friction.xml");
    const std::string path =
        "/data/xtzhang25/_work/activate/tmp/inertia_friction_roundtrip.nks";
    nuka::scene::nks::Save(scene, path);
    const auto loaded = nuka::scene::nks::Load(path);

    ASSERT_GE(loaded.RigidBodyCount(), 2u);
    ASSERT_GE(loaded.JointCount(), 1u);
    const auto& a = scene.GetBody(1);
    const auto& b = loaded.GetBody(1);
    EXPECT_NEAR(a.inertia.x, b.inertia.x, 1e-6);
    EXPECT_NEAR(a.inertia.y, b.inertia.y, 1e-6);
    EXPECT_NEAR(a.inertia.z, b.inertia.z, 1e-6);
    EXPECT_NEAR(a.inertial_transform.rotation.w, b.inertial_transform.rotation.w, 1e-6);
    EXPECT_NEAR(a.inertial_transform.rotation.x, b.inertial_transform.rotation.x, 1e-6);
    EXPECT_NEAR(a.inertial_transform.rotation.y, b.inertial_transform.rotation.y, 1e-6);
    EXPECT_NEAR(a.inertial_transform.rotation.z, b.inertial_transform.rotation.z, 1e-6);
    EXPECT_NEAR(loaded.GetJoint(0).frictionloss, 0.075f, 1e-6);
}
