#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "nk/material/hencky_j2.hpp"

namespace {
namespace material = nuka::nk::material;
using Matrix = std::array<float, 9>;
constexpr Matrix kIdentity{1, 0, 0, 0, 1, 0, 0, 0, 1};
constexpr material::HenckyJ2Parameters kMaterial{30000.0f, 0.3f, 200.0f, 4000.0f};

Matrix Multiply(const Matrix& a, const Matrix& b) {
    Matrix result{};
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            for (int k = 0; k < 3; ++k) result[3 * r + c] += a[3 * r + k] * b[3 * k + c];
    return result;
}

double Determinant(const float* a) {
    return double(a[0]) * (double(a[4]) * a[8] - double(a[5]) * a[7]) -
           double(a[1]) * (double(a[3]) * a[8] - double(a[5]) * a[6]) +
           double(a[2]) * (double(a[3]) * a[7] - double(a[4]) * a[6]);
}

Matrix Rotation(float angle) {
    const float c = std::cos(angle), s = std::sin(angle), t = 1.0f - c;
    const float x = 2.0f / 3.0f, y = -1.0f / 3.0f, z = 2.0f / 3.0f;
    return {c + x*x*t, x*y*t - z*s, x*z*t + y*s,
            y*x*t + z*s, c + y*y*t, y*z*t - x*s,
            z*x*t - y*s, z*y*t + x*s, c + z*z*t};
}

void ExpectMatrixNear(const Matrix& actual, const Matrix& expected, float relative) {
    float scale = 1.0e-12f;
    for (float v : expected) scale = std::max(scale, std::abs(v));
    for (int i = 0; i < 9; ++i) EXPECT_NEAR(actual[i], expected[i], relative * scale);
}
}  // namespace

TEST(HenckyJ2, ClosedFormYieldHardeningAndElasticUnload) {
    const double mu = double(kMaterial.youngs) / (2.0 * (1.0 + kMaterial.poisson));
    for (float strain : {0.003f, 0.03f, 0.2f, 0.7f}) {
        const Matrix trial{std::exp(strain), 0, 0, 0, std::exp(-0.5f * strain), 0,
                           0, 0, std::exp(-0.5f * strain)};
        Matrix elastic = kIdentity, plastic = kIdentity;
        float alpha = 0.0f;
        ASSERT_EQ(material::ReturnHenckyJ2(trial.data(), kMaterial, elastic.data(),
                                           plastic.data(), alpha), material::ConstitutiveStatus::Ok);
        const double log_axial = std::log(double(trial[0])), log_lateral = std::log(double(trial[4]));
        const double trial_q = 2.0 * mu * (log_axial - log_lateral);
        const double expected_alpha = std::max(0.0, (trial_q - kMaterial.yield_stress) /
                                                        (3.0 * mu + kMaterial.hardening_modulus));
        EXPECT_NEAR(alpha, expected_alpha, std::max(expected_alpha, 1.0e-4) * 2.0e-5);
        material::HenckyResponse response;
        ASSERT_EQ(material::EvaluateHenckyJ2(elastic.data(), kMaterial, response),
                  material::ConstitutiveStatus::Ok);
        const double expected_q = trial_q - 3.0 * mu * expected_alpha;
        const double trace = log_axial + 2.0 * log_lateral;
        const double bulk = kMaterial.youngs / (3.0 * (1.0 - 2.0 * kMaterial.poisson));
        const double expected_energy = expected_q * expected_q / (6.0 * mu) + 0.5 * bulk * trace * trace;
        EXPECT_NEAR(response.equivalent_stress, expected_q, expected_q * 2.0e-5);
        EXPECT_NEAR(response.elastic_energy, expected_energy, expected_energy * 2.0e-5);
        ExpectMatrixNear(Multiply(elastic, plastic), trial, 2.0e-4f);
        EXPECT_NEAR(Determinant(plastic.data()), 1.0, 2.0e-4);
        const Matrix held_plastic = plastic;
        const float held_alpha = alpha;
        const Matrix unloaded = Rotation(0.81f);
        ASSERT_EQ(material::ReturnHenckyJ2(unloaded.data(), kMaterial, elastic.data(),
                                           plastic.data(), alpha), material::ConstitutiveStatus::Ok);
        EXPECT_NEAR(alpha, held_alpha, 2.0e-6f);
        ExpectMatrixNear(plastic, held_plastic, 2.0e-6f);
    }
}

TEST(HenckyJ2, RotationObjectivityAndIsochoricPlasticHistory) {
    Matrix trial{1.25f, 0.21f, -0.13f, 0.04f, 0.88f, 0.19f, 0.08f, -0.09f, 0.98f};
    Matrix base_elastic{}, base_plastic = kIdentity;
    float base_alpha = 0.0f;
    ASSERT_EQ(material::ReturnHenckyJ2(trial.data(), kMaterial, base_elastic.data(),
                                       base_plastic.data(), base_alpha), material::ConstitutiveStatus::Ok);
    material::HenckyResponse base;
    ASSERT_EQ(material::EvaluateHenckyJ2(base_elastic.data(), kMaterial, base),
              material::ConstitutiveStatus::Ok);
    for (float angle : {-2.1f, -0.77f, 0.43f, 1.62f, 2.9f}) {
        const Matrix rotation = Rotation(angle), rotated_trial = Multiply(rotation, trial);
        Matrix elastic{}, plastic = kIdentity;
        float alpha = 0.0f;
        ASSERT_EQ(material::ReturnHenckyJ2(rotated_trial.data(), kMaterial, elastic.data(),
                                           plastic.data(), alpha), material::ConstitutiveStatus::Ok);
        material::HenckyResponse response;
        ASSERT_EQ(material::EvaluateHenckyJ2(elastic.data(), kMaterial, response),
                  material::ConstitutiveStatus::Ok);
        EXPECT_NEAR(alpha, base_alpha, base_alpha * 2.0e-5f);
        EXPECT_NEAR(response.equivalent_stress, base.equivalent_stress, base.equivalent_stress * 2.0e-5f);
        EXPECT_NEAR(response.elastic_energy, base.elastic_energy, base.elastic_energy * 2.0e-5f);
        ExpectMatrixNear(elastic, Multiply(rotation, base_elastic), 2.0e-5f);
        ExpectMatrixNear(plastic, base_plastic, 2.0e-5f);
        ExpectMatrixNear(Multiply(elastic, plastic), rotated_trial, 2.0e-4f);
        EXPECT_NEAR(Determinant(plastic.data()), 1.0, 2.0e-4);
    }
    Matrix elastic = kIdentity, plastic = kIdentity;
    float alpha = 0.0f;
    const Matrix hydrostatic{0.8f, 0, 0, 0, 0.8f, 0, 0, 0, 0.8f};
    ASSERT_EQ(material::ReturnHenckyJ2(hydrostatic.data(), kMaterial, elastic.data(),
                                       plastic.data(), alpha), material::ConstitutiveStatus::Ok);
    EXPECT_EQ(alpha, 0.0f);
    EXPECT_EQ(plastic, kIdentity);
}

TEST(HenckyJ2, InvalidStatesAndParametersDoNotCommit) {
    for (float invalid : {0.0f, -1.0f, std::numeric_limits<float>::quiet_NaN(),
                          std::numeric_limits<float>::infinity()}) {
        Matrix trial = kIdentity, elastic = Rotation(0.3f), plastic = kIdentity;
        trial[0] = invalid;
        const Matrix original = elastic;
        float alpha = 0.125f;
        EXPECT_NE(material::ReturnHenckyJ2(trial.data(), kMaterial, elastic.data(),
                                           plastic.data(), alpha), material::ConstitutiveStatus::Ok);
        EXPECT_EQ(elastic, original);
        EXPECT_EQ(plastic, kIdentity);
        EXPECT_EQ(alpha, 0.125f);
    }
    auto parameters = kMaterial;
    parameters.poisson = 0.5f;
    EXPECT_FALSE(material::ValidHenckyJ2(parameters));
    parameters.poisson = 0.0f;
    EXPECT_TRUE(material::ValidHenckyJ2(parameters));
    parameters.poisson = -0.3f;
    EXPECT_TRUE(material::ValidHenckyJ2(parameters));
    parameters.hardening_modulus = -1.0f;
    EXPECT_FALSE(material::ValidHenckyJ2(parameters));
}
