// ---------------------------------------------------------------------------
// Tests for nuka::math – Vec3, Quat, Transform, Spatial algebra
// ---------------------------------------------------------------------------

#include <gtest/gtest.h>
#include "math/vec3.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/spatial.hpp"

#include <cmath>

namespace {

using namespace nuka::math;

constexpr float kPi = 3.14159265358979323846f;
constexpr float kEps = 1e-4f;

// =========================================================================
// Vec3 tests
// =========================================================================

TEST(Vec3, DefaultConstructor) {
    constexpr Vec3 v;
    EXPECT_FLOAT_EQ(v.x, 0.0f);
    EXPECT_FLOAT_EQ(v.y, 0.0f);
    EXPECT_FLOAT_EQ(v.z, 0.0f);
}

TEST(Vec3, Arithmetic) {
    constexpr Vec3 a{1.0f, 2.0f, 3.0f};
    constexpr Vec3 b{4.0f, 5.0f, 6.0f};

    constexpr auto sum = a + b;
    EXPECT_FLOAT_EQ(sum.x, 5.0f);
    EXPECT_FLOAT_EQ(sum.y, 7.0f);
    EXPECT_FLOAT_EQ(sum.z, 9.0f);

    constexpr auto diff = a - b;
    EXPECT_FLOAT_EQ(diff.x, -3.0f);
    EXPECT_FLOAT_EQ(diff.y, -3.0f);
    EXPECT_FLOAT_EQ(diff.z, -3.0f);

    constexpr auto scaled = a * 2.0f;
    EXPECT_FLOAT_EQ(scaled.x, 2.0f);
    EXPECT_FLOAT_EQ(scaled.y, 4.0f);
    EXPECT_FLOAT_EQ(scaled.z, 6.0f);

    constexpr auto div = b / 2.0f;
    EXPECT_FLOAT_EQ(div.x, 2.0f);
    EXPECT_FLOAT_EQ(div.y, 2.5f);
    EXPECT_FLOAT_EQ(div.z, 3.0f);
}

TEST(Vec3, DotProduct) {
    constexpr Vec3 a{1.0f, 2.0f, 3.0f};
    constexpr Vec3 b{4.0f, -5.0f, 6.0f};
    constexpr float d = a.Dot(b);
    EXPECT_FLOAT_EQ(d, 1.0f * 4.0f + 2.0f * (-5.0f) + 3.0f * 6.0f); // 12
}

TEST(Vec3, CrossProduct) {
    constexpr Vec3 x = Vec3::UnitX();
    constexpr Vec3 y = Vec3::UnitY();
    constexpr Vec3 z = x.Cross(y);
    EXPECT_FLOAT_EQ(z.x, 0.0f);
    EXPECT_FLOAT_EQ(z.y, 0.0f);
    EXPECT_FLOAT_EQ(z.z, 1.0f);

    // Anti-commutativity: a x b = -(b x a)
    constexpr Vec3 yx = y.Cross(x);
    EXPECT_FLOAT_EQ(yx.z, -1.0f);
}

TEST(Vec3, LengthAndNormalize) {
    Vec3 v{3.0f, 4.0f, 0.0f};
    EXPECT_FLOAT_EQ(v.Length(), 5.0f);

    Vec3 n = v.Normalized();
    EXPECT_NEAR(n.x, 0.6f, kEps);
    EXPECT_NEAR(n.y, 0.8f, kEps);
    EXPECT_NEAR(n.z, 0.0f, kEps);
    EXPECT_NEAR(n.Length(), 1.0f, kEps);
}

TEST(Vec3, Equality) {
    constexpr Vec3 a{1.0f, 2.0f, 3.0f};
    constexpr Vec3 b{1.0f, 2.0f, 3.0f};
    constexpr Vec3 c{1.0f, 2.0f, 4.0f};
    static_assert(a == b);
    static_assert(a != c);
}

// =========================================================================
// Quat tests
// =========================================================================

TEST(Quat, Identity) {
    const auto q = Quat::Identity();
    EXPECT_FLOAT_EQ(q.w, 1.0f);
    EXPECT_FLOAT_EQ(q.x, 0.0f);
    EXPECT_FLOAT_EQ(q.y, 0.0f);
    EXPECT_FLOAT_EQ(q.z, 0.0f);
}

TEST(Quat, IdentityRotatesNothing) {
    const Vec3 v{1.0f, 2.0f, 3.0f};
    const Vec3 r = Quat::Identity().Rotate(v);
    EXPECT_NEAR(r.x, v.x, kEps);
    EXPECT_NEAR(r.y, v.y, kEps);
    EXPECT_NEAR(r.z, v.z, kEps);
}

TEST(SpatialMath, RotateAxisAngleVector) {
    const auto q = Quat::FromAxisAngle({0.0f, 0.0f, 1.0f}, kPi * 0.5f);
    const auto v = q.Rotate({1.0f, 0.0f, 0.0f});
    EXPECT_NEAR(v.x, 0.0f, kEps);
    EXPECT_NEAR(v.y, 1.0f, kEps);
    EXPECT_NEAR(v.z, 0.0f, kEps);
}

TEST(Quat, Rotate180) {
    const auto q = Quat::FromAxisAngle({0.0f, 1.0f, 0.0f}, kPi);
    const auto v = q.Rotate({1.0f, 0.0f, 0.0f});
    EXPECT_NEAR(v.x, -1.0f, kEps);
    EXPECT_NEAR(v.y, 0.0f, kEps);
    EXPECT_NEAR(v.z, 0.0f, kEps);
}

TEST(Quat, Multiplication) {
    // Two 90-degree rotations around Z should give 180-degree rotation
    const auto q90 = Quat::FromAxisAngle(Vec3::UnitZ(), kPi * 0.5f);
    const auto q180 = q90 * q90;
    const auto v = q180.Rotate({1.0f, 0.0f, 0.0f});
    EXPECT_NEAR(v.x, -1.0f, kEps);
    EXPECT_NEAR(v.y, 0.0f, kEps);
    EXPECT_NEAR(v.z, 0.0f, kEps);
}

TEST(Quat, Conjugate) {
    const auto q = Quat::FromAxisAngle(Vec3::UnitZ(), kPi * 0.5f);
    const auto qc = q.Conjugate();

    // q * q^* should be identity
    const auto prod = q * qc;
    EXPECT_NEAR(prod.w, 1.0f, kEps);
    EXPECT_NEAR(prod.x, 0.0f, kEps);
    EXPECT_NEAR(prod.y, 0.0f, kEps);
    EXPECT_NEAR(prod.z, 0.0f, kEps);
}

TEST(Quat, Normalized) {
    Quat q{2.0f, 0.0f, 0.0f, 0.0f};
    auto n = q.Normalized();
    EXPECT_NEAR(n.Norm(), 1.0f, kEps);
    EXPECT_NEAR(n.w, 1.0f, kEps);
}

// =========================================================================
// Transform tests
// =========================================================================

TEST(Transform, Identity) {
    const auto t = Transform::Identity();
    const Vec3 p{1.0f, 2.0f, 3.0f};
    const Vec3 tp = t.TransformPoint(p);
    EXPECT_NEAR(tp.x, p.x, kEps);
    EXPECT_NEAR(tp.y, p.y, kEps);
    EXPECT_NEAR(tp.z, p.z, kEps);
}

TEST(Transform, TranslateOnly) {
    Transform t;
    t.position = {10.0f, 0.0f, 0.0f};
    t.rotation = Quat::Identity();
    const Vec3 p = t.TransformPoint({1.0f, 2.0f, 3.0f});
    EXPECT_NEAR(p.x, 11.0f, kEps);
    EXPECT_NEAR(p.y, 2.0f, kEps);
    EXPECT_NEAR(p.z, 3.0f, kEps);
}

TEST(Transform, RotateOnly) {
    Transform t;
    t.position = Vec3::Zero();
    t.rotation = Quat::FromAxisAngle(Vec3::UnitZ(), kPi * 0.5f);
    const Vec3 p = t.TransformPoint({1.0f, 0.0f, 0.0f});
    EXPECT_NEAR(p.x, 0.0f, kEps);
    EXPECT_NEAR(p.y, 1.0f, kEps);
    EXPECT_NEAR(p.z, 0.0f, kEps);
}

TEST(Transform, TransformDirection) {
    Transform t;
    t.position = {100.0f, 200.0f, 300.0f}; // should be ignored
    t.rotation = Quat::FromAxisAngle(Vec3::UnitZ(), kPi * 0.5f);
    const Vec3 d = t.TransformDirection({1.0f, 0.0f, 0.0f});
    EXPECT_NEAR(d.x, 0.0f, kEps);
    EXPECT_NEAR(d.y, 1.0f, kEps);
    EXPECT_NEAR(d.z, 0.0f, kEps);
}

TEST(Transform, Composition) {
    // T1: translate by (1,0,0)
    Transform t1;
    t1.position = {1.0f, 0.0f, 0.0f};
    t1.rotation = Quat::Identity();

    // T2: rotate 90 degrees around Z
    Transform t2;
    t2.position = Vec3::Zero();
    t2.rotation = Quat::FromAxisAngle(Vec3::UnitZ(), kPi * 0.5f);

    // t1 * t2 means: apply t2 first (rotate), then t1 (translate)
    const auto composed = t1 * t2;
    const Vec3 p = composed.TransformPoint({1.0f, 0.0f, 0.0f});
    // rotate (1,0,0) by 90 around Z -> (0,1,0), then translate by (1,0,0) -> (1,1,0)
    EXPECT_NEAR(p.x, 1.0f, kEps);
    EXPECT_NEAR(p.y, 1.0f, kEps);
    EXPECT_NEAR(p.z, 0.0f, kEps);
}

TEST(Transform, Inverse) {
    Transform t;
    t.position = {3.0f, 4.0f, 5.0f};
    t.rotation = Quat::FromAxisAngle(Vec3{1.0f, 1.0f, 0.0f}.Normalized(), kPi / 3.0f);

    const auto inv = t.Inverse();
    const Vec3 original{7.0f, -2.0f, 1.0f};

    // T^-1(T(p)) should give back p
    const Vec3 roundTrip = inv.TransformPoint(t.TransformPoint(original));
    EXPECT_NEAR(roundTrip.x, original.x, kEps);
    EXPECT_NEAR(roundTrip.y, original.y, kEps);
    EXPECT_NEAR(roundTrip.z, original.z, kEps);
}

// =========================================================================
// SpatialVector tests
// =========================================================================

TEST(SpatialVector, DefaultIsZero) {
    constexpr SpatialVector sv;
    EXPECT_FLOAT_EQ(sv.angular.x, 0.0f);
    EXPECT_FLOAT_EQ(sv.linear.x, 0.0f);
}

TEST(SpatialVector, Addition) {
    constexpr SpatialVector a{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    constexpr SpatialVector b{{7.0f, 8.0f, 9.0f}, {10.0f, 11.0f, 12.0f}};
    constexpr auto c = a + b;
    EXPECT_FLOAT_EQ(c.angular.x, 8.0f);
    EXPECT_FLOAT_EQ(c.linear.z, 18.0f);
}

TEST(SpatialVector, ScalarMultiply) {
    constexpr SpatialVector a{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    constexpr auto b = a * 2.0f;
    EXPECT_FLOAT_EQ(b.angular.y, 4.0f);
    EXPECT_FLOAT_EQ(b.linear.x, 8.0f);

    constexpr auto c = 3.0f * a;
    EXPECT_FLOAT_EQ(c.angular.z, 9.0f);
}

TEST(SpatialVector, Dot) {
    constexpr SpatialVector a{{1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}};
    constexpr SpatialVector b{{2.0f, 0.0f, 0.0f}, {0.0f, 3.0f, 0.0f}};
    constexpr float d = a.Dot(b);
    EXPECT_FLOAT_EQ(d, 5.0f); // 1*2 + 1*3
}

TEST(SpatialVector, CrossMotion) {
    // Pure angular velocity crossing a pure angular velocity
    constexpr SpatialVector w{{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}};
    constexpr SpatialVector v{{1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
    constexpr auto result = w.CrossMotion(v);
    // angular part: (0,0,1) x (1,0,0) = (0,1,0)
    EXPECT_NEAR(result.angular.y, 1.0f, kEps);
    EXPECT_NEAR(result.angular.x, 0.0f, kEps);
}

TEST(SpatialVector, CrossForce) {
    constexpr SpatialVector m{{0.0f, 0.0f, 1.0f}, {0.0f, 0.0f, 0.0f}};
    constexpr SpatialVector f{{0.0f, 0.0f, 0.0f}, {1.0f, 0.0f, 0.0f}};
    constexpr auto result = m.CrossForce(f);
    // angular part: w x n + v0 x f0 = 0 + 0 = 0  ... wait, n=0, f0=(1,0,0)
    // angular: (0,0,1)x(0,0,0) + (0,0,0)x(1,0,0) = 0
    // linear:  (0,0,1)x(1,0,0) = (0,1,0)... wait that's  -(0,1,0)? No:
    // (0,0,1) x (1,0,0) = (0*0 - 1*0, 1*1 - 0*0, 0*0 - 0*1) = (0,1,0)
    EXPECT_NEAR(result.linear.x, 0.0f, kEps);
    EXPECT_NEAR(result.linear.y, 1.0f, kEps);
    EXPECT_NEAR(result.linear.z, 0.0f, kEps);
}

// =========================================================================
// SpatialTransform tests
// =========================================================================

TEST(SpatialTransform, IdentityMotion) {
    const auto X = SpatialTransform::Identity();
    const SpatialVector v{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    const auto result = X.TransformMotion(v);
    EXPECT_NEAR(result.angular.x, 1.0f, kEps);
    EXPECT_NEAR(result.angular.y, 2.0f, kEps);
    EXPECT_NEAR(result.angular.z, 3.0f, kEps);
    EXPECT_NEAR(result.linear.x, 4.0f, kEps);
    EXPECT_NEAR(result.linear.y, 5.0f, kEps);
    EXPECT_NEAR(result.linear.z, 6.0f, kEps);
}

TEST(SpatialTransform, IdentityForce) {
    const auto X = SpatialTransform::Identity();
    const SpatialVector f{{1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}};
    const auto result = X.TransformForce(f);
    EXPECT_NEAR(result.angular.x, 1.0f, kEps);
    EXPECT_NEAR(result.linear.x, 4.0f, kEps);
}

} // anonymous namespace
