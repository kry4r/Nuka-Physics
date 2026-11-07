#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>

#include "nk/solve/augmented_row.hpp"

TEST(ContactTangent, RelaxationFindsLinearRootAfterExtrapolation) {
    namespace augmented = nuka::nk::augmented;
    using nuka::math::Vec3;
    for (float beta : {0.0f, 0.5f}) {
        augmented::ContactRelaxation history{};
        const Vec3 start{10.0f, 1.0f, 0.0f};
        const auto first = augmented::RelaxContactImpulse(start, {10.0f, -2.0f, 0.0f}, &history);
        const Vec3 point = first + (first - start) * beta;
        const auto root = augmented::RelaxContactImpulse(point, {10.0f, -2.0f * point.y, 0.0f}, &history);
        EXPECT_FLOAT_EQ(root.x, 10.0f);
        EXPECT_NEAR(root.y, 0.0f, 5.0e-7f);
        EXPECT_FLOAT_EQ(root.z, 0.0f);
        EXPECT_NEAR(history.weight, 1.0f / 3.0f, 1.0e-7f);
        EXPECT_LE(root.y * root.y + root.z * root.z, root.x * root.x);
    }
}

TEST(ContactTangent, CollapsedFrictionSetsHaveZeroResponse) {
    namespace augmented = nuka::nk::augmented;
    using nuka::math::Vec3;
    for (float mu : {1.1e-5f, 0.5f, 2.0f}) {
        for (float value : {0.0f, 0.1f}) {
            const Vec3 residual{0.0f, value, -value};
            for (const auto& term : {augmented::EvaluateTangent({}, 10.0f, residual, 0.0f, mu, mu),
                                    augmented::EvaluateTangentImpulse({}, 10.0f, residual, 0.0f, mu, mu)}) {
                EXPECT_FLOAT_EQ(term.impulse.y, 0.0f);
                EXPECT_FLOAT_EQ(term.impulse.z, 0.0f);
                EXPECT_FLOAT_EQ(term.curvature.yy, 0.0f);
                EXPECT_FLOAT_EQ(term.curvature.zz, 0.0f);
                EXPECT_FLOAT_EQ(term.curvature.yz, 0.0f);
                EXPECT_DOUBLE_EQ(term.potential, 0.0);
            }
            EXPECT_FLOAT_EQ(augmented::TangentPotentialChange({}, 10.0f, residual,
                {0.0f, 1.0f, -1.0f}, 0.0f, mu, mu), 0.0f);
            const double dual[2]{}, components[2]{value, -value};
            const auto ball = augmented::EvaluateBall(dual, 10.0, components, 0.0, 0.0);
            for (int i = 0; i < 2; ++i) {
                EXPECT_FLOAT_EQ(ball.impulse[i], 0.0f);
                for (int j = 0; j < 2; ++j) EXPECT_FLOAT_EQ(ball.curvature[i][j], 0.0f);
            }
            EXPECT_DOUBLE_EQ(ball.potential, 0.0);
        }
    }
}

TEST(ContactTangent, FrictionMagnitudePreservesPhysicalCurvatureAndCoulombLimit) {
    namespace augmented = nuka::nk::augmented;
    using nuka::math::Vec3;
    for (float mu : {1.1e-5f, 0.5f, 2.0f}) {
        const float normal = 0.05f, rho = 10.0f;
        const float radius = mu * normal;
        for (float fraction : {0.25f, 2.0f}) {
            const Vec3 residual{0.0f, 0.6f * fraction * radius / rho, 0.8f * fraction * radius / rho};
            const float magnitude = radius * std::min(fraction, 1.0f);
            const auto exact = augmented::EvaluateTangent({}, rho, residual, normal, mu, mu);
            const auto native = augmented::EvaluateTangentImpulse({}, rho, residual, normal, mu, mu);
            for (const auto& term : {exact, native}) {
                EXPECT_NEAR(term.impulse.y, 0.6f * magnitude, radius * 2.0e-6f);
                EXPECT_NEAR(term.impulse.z, 0.8f * magnitude, radius * 2.0e-6f);
                if (fraction < 1.0f) {
                    EXPECT_NEAR(term.curvature.yy, rho, 2.0e-5f);
                    EXPECT_NEAR(term.curvature.zz, rho, 2.0e-5f);
                }
            }
            const Vec3 drop{0.0f, 0.2f * residual.y, 0.2f * residual.z};
            const auto moved = augmented::EvaluateTangent({}, rho, residual - drop, normal, mu, mu);
            const float change = augmented::TangentPotentialChange({}, rho, residual, drop, normal, mu, mu);
            EXPECT_NEAR(change, moved.potential - exact.potential, radius * radius / rho * 2.0e-6);
        }
    }
}

TEST(ContactTangent, EllipticMetricPreservesAnisotropicConeFixedPoints) {
    namespace augmented = nuka::nk::augmented;
    using nuka::math::Vec3;
    for (float ratio : {1.0f, 1000.0f, 1000000.0f}) for (float cap : {0.15f, 10.0f, 1000.0f}) {
        const Vec3 compliance{2.0f, 2.0f / ratio, 8.0f / ratio};
        for (float mu : {0.2f, 1.0f, 2.0f}) {
            const Vec3 impulse{1.0f, 0.6f * mu, 0.4f * mu};
            const Vec3 outward{-1.0f, 0.6f / mu, 1.6f / mu};
            const Vec3 residual = Vec3{compliance.x * impulse.x, compliance.y * impulse.y,
                                       compliance.z * impulse.z} + outward * 0.25f;
            const auto term = augmented::EvaluateEllipticImpulse(impulse, cap, residual, compliance, mu);
            EXPECT_NEAR(term.impulse.x, impulse.x, 2.0e-6f);
            EXPECT_NEAR(term.impulse.y, impulse.y, 2.0e-6f);
            EXPECT_NEAR(term.impulse.z, impulse.z, 2.0e-6f);
            const auto normal = augmented::EvaluateEllipticImpulse({}, cap, {1, 0, 0}, compliance, mu);
            EXPECT_NEAR(normal.impulse.x, cap / (1.0f + 2.0f * cap), 1.0e-7f);
        }
    }
}

TEST(ContactTangent, EllipticPotentialGradientAndCurvatureMatchImpulse) {
    namespace augmented = nuka::nk::augmented;
    using nuka::math::Vec3;
    const Vec3 axes[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    const Vec3 dual{0.3f, 0.05f, -0.1f};
    for (float ratio : {1.0f, 1000.0f}) for (float cap : {0.15f, 10.0f}) {
      for (float mu : {0.2f, 1.0f, 2.0f}) {
        const Vec3 compliance{2.0f, 2.0f / ratio, 8.0f / ratio};
        const auto response = augmented::ComputeEllipticResponse(cap, compliance);
        const auto k = response.residual_scale, s = response.dual_scale;
        for (Vec3 trial : {Vec3{1.0f, 0.2f * mu, 0.0f}, Vec3{-1.0f, 0.0f, 0.0f},
                           Vec3{0.5f, 1.5f * mu, 0.2f * mu}}) {
            const Vec3 residual{(trial.x - s.x * dual.x) / k.x, (trial.y - s.y * dual.y) / k.y,
                                (trial.z - s.z * dual.z) / k.z};
            const auto term = augmented::EvaluateEllipticImpulse(dual, cap, residual, compliance, mu);
            const auto c = term.curvature;
            const Vec3 columns[3] = {{c.xx, c.xy, c.xz}, {c.xy, c.yy, c.yz}, {c.xz, c.yz, c.zz}};
            for (uint32_t axis = 0u; axis < 3u; ++axis) {
                const float h = 0.001f * std::max(1.0f, std::abs(residual.Dot(axes[axis])));
                const Vec3 step = axes[axis] * h;
                const auto plus = augmented::EvaluateEllipticImpulse(dual, cap, residual + step, compliance, mu);
                const auto minus = augmented::EvaluateEllipticImpulse(dual, cap, residual - step, compliance, mu);
                const Vec3 derivative = (plus.impulse - minus.impulse) / (2.0f * h);
                EXPECT_NEAR(derivative.x, columns[axis].x, 0.001f * cap);
                EXPECT_NEAR(derivative.y, columns[axis].y, 0.001f * cap);
                EXPECT_NEAR(derivative.z, columns[axis].z, 0.001f * cap);
                const double plus_energy = augmented::EllipticPotentialChange(dual, cap, residual, -step, compliance, mu);
                const double minus_energy = augmented::EllipticPotentialChange(dual, cap, residual, step, compliance, mu);
                EXPECT_NEAR((plus_energy - minus_energy) / (2.0 * h), term.impulse.Dot(axes[axis]), 0.001);
            }
            for (Vec3 v : {Vec3{1, 2, -3}, Vec3{-2, 1, 1}, Vec3{1, 1, 1}}) {
                const double curvature = v.Dot(columns[0] * v.x + columns[1] * v.y + columns[2] * v.z);
                EXPECT_GE(curvature, -1.0e-5 * cap);
                EXPECT_LE(curvature, cap * v.LengthSq() + 1.0e-5 * cap);
            }
        }
      }
    }
}
