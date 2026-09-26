#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#include "nk/material/hencky_j2.hpp"
#include "nk/material/mpm_constitutive.hpp"

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

TEST(MpmMaterialTrial, RepeatedTrialsPreserveAcceptedPlasticHistory) {
    nuka::nk::MpmMaterial parameters;
    parameters.model_kind = nuka::nk::MpmMaterial::kHenckyJ2;
    parameters.youngs = kMaterial.youngs;
    parameters.poisson = kMaterial.poisson;
    parameters.yield_stress = kMaterial.yield_stress;
    parameters.hardening_modulus = kMaterial.hardening_modulus;
    Matrix elastic = kIdentity, plastic = kIdentity;
    float alpha = 0.125f;
    const material::MpmMaterialHistory history{elastic.data(), plastic.data(), alpha};
    const Matrix loading{12, 3, 0, 0, -6, 0, 0, 0, -6}, held{};
    material::MpmMaterialTrial loaded, unchanged, repeated;
    ASSERT_EQ(material::EvaluateMpmMaterialTrial(parameters, history, loading.data(), 0.01f, loaded),
              material::ConstitutiveStatus::Ok);
    ASSERT_GT(loaded.equivalent_plastic_strain, alpha);
    ASSERT_EQ(material::EvaluateMpmMaterialTrial(parameters, history, held.data(), 0.01f, unchanged),
              material::ConstitutiveStatus::Ok);
    EXPECT_EQ(unchanged.equivalent_plastic_strain, alpha);
    EXPECT_EQ(elastic, kIdentity);
    EXPECT_EQ(plastic, kIdentity);
    EXPECT_EQ(alpha, 0.125f);
    ASSERT_EQ(material::EvaluateMpmMaterialTrial(parameters, history, loading.data(), 0.01f, repeated),
              material::ConstitutiveStatus::Ok);
    for (int k = 0; k < 9; ++k) {
        EXPECT_EQ(repeated.elastic_f[k], loaded.elastic_f[k]);
        EXPECT_EQ(repeated.plastic_f[k], loaded.plastic_f[k]);
    }
    EXPECT_EQ(repeated.equivalent_plastic_strain, loaded.equivalent_plastic_strain);
    EXPECT_FALSE(material::CommitMpmMaterialTrial(loaded, elastic.data(), nullptr, &alpha));
    EXPECT_EQ(elastic, kIdentity);
    EXPECT_EQ(alpha, 0.125f);
    ASSERT_TRUE(material::CommitMpmMaterialTrial(loaded, elastic.data(), plastic.data(), &alpha));
    Matrix expected = kIdentity;
    for (int k = 0; k < 9; ++k) expected[k] += 0.01f * loading[k];
    ExpectMatrixNear(Multiply(elastic, plastic), expected, 2.0e-4f);
    EXPECT_NEAR(Determinant(plastic.data()), 1.0, 2.0e-4);
    EXPECT_EQ(alpha, loaded.equivalent_plastic_strain);
}

TEST(MpmMaterialTrial, FluidPressureAndRejectedVolumeAreTransactional) {
    nuka::nk::MpmMaterial parameters;
    parameters.model_kind = 3.0f;
    parameters.bulk_modulus = 100000.0f;
    parameters.tait_gamma = 7.0f;
    const Matrix compression{-2, 3, 0, 0, -1, 0, 0, 0, -1};
    material::MpmMaterialTrial trial;
    ASSERT_EQ(material::EvaluateMpmMaterialTrial(parameters, {kIdentity.data()},
        compression.data(), 0.01f, trial), material::ConstitutiveStatus::Ok);
    const double volume = Determinant(trial.elastic_f);
    EXPECT_NEAR(volume, 0.96, 2.0e-7);
    Matrix stress;
    ASSERT_EQ(material::EvaluateMpmKirchhoff(parameters, trial.elastic_f, compression.data(),
        stress.data()), material::ConstitutiveStatus::Ok);
    const double expected = -volume * parameters.bulk_modulus * (std::pow(volume, -7.0) - 1.0);
    for (int k = 0; k < 9; ++k)
        EXPECT_NEAR(stress[k], k % 4 == 0 ? expected : 0.0, std::abs(expected) * 2.0e-5);
    const material::MpmMaterialTrial accepted = trial;
    const Matrix inverted{-100, 0, 0, 0, -100, 0, 0, 0, -100};
    EXPECT_EQ(material::EvaluateMpmMaterialTrial(parameters, {kIdentity.data()},
        inverted.data(), 0.01f, trial), material::ConstitutiveStatus::SingularDeformation);
    for (int k = 0; k < 9; ++k) EXPECT_EQ(trial.elastic_f[k], accepted.elastic_f[k]);
    EXPECT_EQ(trial.has_plastic_history, accepted.has_plastic_history);
    EXPECT_EQ(trial.equivalent_plastic_strain, accepted.equivalent_plastic_strain);
}
