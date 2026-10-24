#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <tuple>
#include <utility>

#include "nk/material/field_response.hpp"
#include "nk/solve/field_block.hpp"

namespace {
namespace nk = nuka::nk;
namespace mf = nuka::nk::mf;
using nuka::math::SymmetricMat3;
using nuka::math::Vec3;
using Points = std::array<Vec3, nk::kMfMaxVertices>;

// A 1 mm polymer sheet and a 2 cm rest triangle (0,0), (a,0), (c,d) in the material frame.
constexpr float kA = 0.02f, kC = 0.007f, kD = 0.017f;
constexpr float kStep = 1.0f / 240.0f;

nk::MfMaterial Sheet() {
    nk::MfMaterial m;
    m.density = 1000.0f;
    m.thickness = 1.0e-3f;
    m.youngs = 2.0e9f;
    m.poisson = 0.3f;
    return m;
}

// The sheet with membrane and bend yield, hardening and Maxwell branches.
nk::MfMaterial Yielding() {
    nk::MfMaterial m = Sheet();
    m.yield_stress = 5.0e6f;
    m.hardening = 2.0e8f;
    m.yield_moment = 0.5f;
    m.bend_hardening = 0.02f;
    m.maxwell_modulus[0] = 0.5f;
    m.maxwell_time[0] = 0.01f;
    m.maxwell_modulus[1] = 0.2f;
    m.maxwell_time[1] = 0.5f;
    m.bend_maxwell_modulus[0] = 0.3f;
    m.bend_maxwell_time[0] = 0.02f;
    return m;
}

// A membrane law at the sheet's stiffness that yields near 0.2% strain, with optional Maxwell branches.
mf::PlaneStressLaw Law(bool yielding, bool viscous) {
    const float modulus = 2.2e6f;
    return {modulus, 0.3f, yielding ? 0.002f * modulus : 0.0f, yielding ? 0.1f * modulus : 0.0f,
            {viscous ? 0.5f : 0.0f, viscous ? 0.2f : 0.0f}, {0.01f, 0.5f}, 0.05f};
}

// Plastic strain, equivalent plastic strain and the two Maxwell strains of a loaded element.
constexpr float kCommitted[nk::kMfResponseStateCapacity] = {0.002f, -0.001f, 0.0005f, 0.001f, 0.001f,
                                              0.0f,   0.0002f, -0.0005f, 0.0003f, 0.0f};

nk::MfElement Triangle() {
    nk::MfElement e;
    e.kind = nk::kMfTriangle;
    for (uint32_t j = 0u; j < 3u; ++j) e.vertex[j] = j;
    const float det = kA * kD;
    e.rest[0] = kD / det;
    e.rest[1] = -kC / det;
    e.rest[2] = 0.0f;
    e.rest[3] = kA / det;
    e.rest[4] = 0.5f * det;
    return e;
}

// Bend over the rest triangle and the far vertices beyond its three edges: a turn-to-curvature map, the
// area and the rest turns.
nk::MfElement Bend(bool flat = false) {
    nk::MfElement e;
    e.kind = nk::kMfBend;
    e.hinges = 7u;
    for (uint32_t j = 0u; j < nk::kMfMaxVertices; ++j) e.vertex[j] = j;
    const float map[9] = {40.0f, -10.0f, 5.0f, -8.0f, 35.0f, 12.0f, 15.0f, 20.0f, -30.0f};
    for (uint32_t k = 0u; k < 9u; ++k) e.rest[k] = map[k];
    e.rest[9] = 0.5f * kA * kD;
    e.rest[10] = 0.0f;
    e.rest[11] = flat ? 0.0f : 0.05f;
    e.rest[12] = flat ? 0.0f : -0.03f;
    if (!mf::InitializeBendDual(e, Sheet())) throw std::runtime_error("Invalid bend curvature map");
    return e;
}

Points RestTriangle() {
    return {Vec3{0.0f, 0.0f, 0.0f}, Vec3{kA, 0.0f, 0.0f}, Vec3{kC, kD, 0.0f}, Vec3{}, Vec3{}, Vec3{}};
}

// The rest triangle with far vertices beyond edges (1, 2), (2, 0) and (0, 1), lifted by `angle`.
Points BentPatch(float angle) {
    const float s = std::sin(angle);
    Points x = RestTriangle();
    const Vec3 far[3] = {{0.022f, 0.019f, 0.012f * s}, {-0.012f, 0.012f, -0.009f * s},
                         {0.011f, -0.015f, 0.015f * s}};
    for (uint32_t i = 0u; i < 3u; ++i) {
        x[3u + 3u * i] = x[(i + 1u) % 3u];
        x[4u + 3u * i] = x[(i + 2u) % 3u];
        x[5u + 3u * i] = far[i];
    }
    return x;
}

// Rotation about the unit axis (2, -1, 2) / 3.
std::array<float, 9> Rotation(float angle) {
    const float c = std::cos(angle), s = std::sin(angle), t = 1.0f - c;
    const float x = 2.0f / 3.0f, y = -1.0f / 3.0f, z = 2.0f / 3.0f;
    return {c + x * x * t, x * y * t - z * s, x * z * t + y * s,
            y * x * t + z * s, c + y * y * t, y * z * t - x * s,
            z * x * t - y * s, z * y * t + x * s, c + z * z * t};
}

Vec3 Apply(const std::array<float, 9>& r, Vec3 v) {
    return {r[0] * v.x + r[1] * v.y + r[2] * v.z, r[3] * v.x + r[4] * v.y + r[5] * v.z,
            r[6] * v.x + r[7] * v.y + r[8] * v.z};
}

float& Axis(Vec3& v, int axis) { return axis == 0 ? v.x : axis == 1 ? v.y : v.z; }

// In-plane stretch and shear plus a lift of vertex 2, then a rigid rotation and offset.
Points Deformed(const std::array<float, 9>& rotation, Vec3 offset) {
    const Points rest = RestTriangle();
    Points x{};
    for (uint32_t j = 0u; j < 3u; ++j) {
        const Vec3 p = rest[j];
        const Vec3 local{1.012f * p.x + 0.006f * p.y, 0.994f * p.y, j == 2u ? 0.0004f : 0.0f};
        x[j] = Apply(rotation, local) + offset;
    }
    return x;
}

mf::FieldGeometry Geometry(const Points& x) {
    mf::FieldGeometry g{};
    for (uint32_t j = 0u; j < nk::kMfMaxVertices; ++j) g.start[j] = x[j];
    return g;
}

double Energy(const nk::MfElement& e, const Points& x) {
    return mf::ElementEnergy(e, Sheet(), nullptr, Geometry(x));
}

Vec3 Gradient(const nk::MfElement& e, const nk::MfMaterial& m, const Points& x, uint32_t local) {
    Vec3 gradient;
    SymmetricMat3 hessian;
    mf::ElementBlock(e, m, nullptr, local, Geometry(x), nullptr, kStep, &gradient, &hessian);
    return gradient;
}

Vec3 Gradient(const nk::MfElement& e, const Points& x, uint32_t local) { return Gradient(e, Sheet(), x, local); }

float Largest(const nk::MfElement& e, const Points& x, uint32_t count) {
    float largest = 0.0f;
    for (uint32_t local = 0u; local < count; ++local) {
        const Vec3 g = Gradient(e, x, local);
        largest = std::max({largest, std::abs(g.x), std::abs(g.y), std::abs(g.z)});
    }
    return largest;
}

uint32_t TangentIndex(int i, int j) {
    if (i == j) return static_cast<uint32_t>(i);
    return static_cast<uint32_t>(i + j + 2);
}

// The J2 measure S^T P S of a stress triple, which reaches 2/3 k^2 on the yield surface.
double YieldMeasure(const double (&s)[3]) {
    const double a = mf::kHalfSqrt2 * (s[0] + s[1]), b = mf::kHalfSqrt2 * (s[0] - s[1]);
    return a * a / 3.0 + b * b + 2.0 * s[2] * s[2];
}

// A paper seam over a 5 mm edge: cohesive stiffness 100 E / L, strength 2 MPa, toughness 200 J/m^2.
constexpr float kPaperEdge = 0.005f;

nk::MfMaterial Paper() {
    nk::MfMaterial m = Sheet();
    m.youngs = 2.0e8f;
    m.thickness = 5.0e-4f;
    m.tear_strength = 2.0e6f;
    m.tear_energy = 200.0f;
    return m;
}

nk::MfElement Seam() {
    nk::MfElement e;
    e.kind = nk::kMfSeam;
    e.hinges = 2u << 4u;
    for (uint32_t j = 0u; j < 4u; ++j) e.vertex[j] = j;
    e.rest[0] = 0.5f * kPaperEdge * Paper().thickness;
    e.rest[1] = 100.0f * Paper().youngs / kPaperEdge;
    e.rest[2] = 1.0f;
    e.rest[3] = 1.0f;
    return e;
}

// Pairs 5 mm apart whose far sides sit `first` and `second` from their near sides.
mf::FieldGeometry SeamGeometry(Vec3 first, Vec3 second) {
    mf::FieldGeometry g{};
    g.start[2] = g.start[3] = Vec3{kPaperEdge, 0.0f, 0.0f};
    g.delta[1] = first;
    g.delta[3] = second;
    return g;
}
}  // namespace

TEST(MaterialFieldResponse, StressAndTangentMatchFiniteDifferences) {
    const double strains[3][3] = {{0.01, -0.004, 0.003}, {-0.02, 0.015, -0.008}, {0.0023, -0.0009, 0.0004}};
    for (const bool yielding : {false, true}) {
        for (const bool viscous : {false, true}) {
            const mf::PlaneStressLaw law = Law(yielding, viscous);
            const double modulus = law.modulus * (1.0 + law.maxwell[0] + law.maxwell[1]), scale = modulus * 0.02;
            int yielded = 0;
            for (const auto& strain : strains) {
                const double E[3] = {strain[0], strain[1], strain[2]};
                mf::FieldResponse<double> r{};
                ASSERT_EQ(mf::EvaluatePlaneStress(law, kCommitted, E, 1.0 / 240.0, r), nk::MfStatus::Ok);
                yielded += r.yielded ? 1 : 0;
                const double h = 1.0e-6;
                for (int k = 0; k < 3; ++k) {
                    double up[3] = {E[0], E[1], E[2]}, down[3] = {E[0], E[1], E[2]};
                    up[k] += h;
                    down[k] -= h;
                    mf::FieldResponse<double> a{}, b{};
                    ASSERT_EQ(mf::EvaluatePlaneStress(law, kCommitted, up, 1.0 / 240.0, a), nk::MfStatus::Ok);
                    ASSERT_EQ(mf::EvaluatePlaneStress(law, kCommitted, down, 1.0 / 240.0, b), nk::MfStatus::Ok);
                    EXPECT_NEAR((a.psi - b.psi) / (2.0 * h), r.stress[k], 1.0e-7 * scale);
                    for (int i = 0; i < 3; ++i)
                        EXPECT_NEAR((a.stress[i] - b.stress[i]) / (2.0 * h), r.tangent[TangentIndex(i, k)],
                                    1.0e-6 * modulus);
                }
                mf::FieldResponse<float> f{};
                const float Ef[3] = {float(E[0]), float(E[1]), float(E[2])};
                ASSERT_EQ(mf::EvaluatePlaneStress(law, kCommitted, Ef, kStep, f), nk::MfStatus::Ok);
                for (int k = 0; k < 3; ++k) EXPECT_NEAR(f.stress[k], r.stress[k], 1.0e-4 * scale);
            }
            EXPECT_EQ(yielded, yielding ? 2 : 0);
        }
    }
}

TEST(MaterialFieldResponse, DirectionalMetricEliminationHasAConsistentTangent) {
    const auto law = Law(true, true);
    const double metrics[3][6] = {{2.2e6, 3.0e6, 0.3e6, 0.4e6, 0.05e6, 0.08e6},
                                  {0.0, 2.0e6, 1.0e6, 0.0, 0.0, 0.0}, {}};
    const double strain[3] = {0.01, -0.004, 0.003};
    for (const auto& metric : metrics) {
        mf::FieldResponse<double> response;
        ASSERT_EQ(mf::EvaluatePlaneStress<double>(law, kCommitted, strain, 1.0 / 240.0,
                                                  response, nullptr, metric), nk::MfStatus::Ok);
        EXPECT_TRUE(response.local.converged);
        EXPECT_GE(response.plastic, -1e-12);
        EXPECT_GE(response.viscous, -1e-12);
        for (uint32_t axis = 0u; axis < 3u; ++axis) {
            double up[3] = {strain[0], strain[1], strain[2]}, down[3] = {strain[0], strain[1], strain[2]};
            const double delta = 1e-7;
            up[axis] += delta;
            down[axis] -= delta;
            mf::FieldResponse<double> a, b;
            ASSERT_EQ(mf::EvaluatePlaneStress<double>(law, kCommitted, up, 1.0 / 240.0, a, nullptr, metric),
                      nk::MfStatus::Ok);
            ASSERT_EQ(mf::EvaluatePlaneStress<double>(law, kCommitted, down, 1.0 / 240.0, b, nullptr, metric),
                      nk::MfStatus::Ok);
            EXPECT_NEAR((a.psi - b.psi) / (2 * delta), response.stress[axis], 1e-4);
            for (uint32_t row = 0u; row < 3u; ++row)
                EXPECT_NEAR((a.stress[row] - b.stress[row]) / (2 * delta),
                            response.tangent[TangentIndex(row, axis)], 0.02);
        }
    }
}

TEST(MaterialFieldResponse, TrialsAreReadOnlyAndFailuresLeaveTheOutput) {
    const mf::PlaneStressLaw law = Law(true, true);
    float committed[nk::kMfResponseStateCapacity];
    std::memcpy(committed, kCommitted, sizeof(committed));
    const float E[3] = {0.01f, -0.002f, 0.004f};
    for (int trial = 0; trial < 100; ++trial) {
        mf::FieldResponse<float> r{};
        ASSERT_EQ(mf::EvaluatePlaneStress(law, committed, E, kStep, r), nk::MfStatus::Ok);
        ASSERT_TRUE(r.yielded);
    }
    EXPECT_EQ(std::memcmp(kCommitted, committed, sizeof(committed)), 0);

    mf::FieldResponse<float> out{};
    out.psi = 7.0f;
    out.stress[1] = -3.0f;
    out.history[4] = 0.5f;
    mf::LocalMaterialReport<float> report;
    EXPECT_EQ(mf::EvaluatePlaneStress(law, committed, E, 0.0f, out, &report), nk::MfStatus::InvalidParameters);
    EXPECT_FALSE(report.converged);
    EXPECT_EQ(report.status, nk::MfStatus::InvalidParameters);
    EXPECT_EQ(mf::EvaluatePlaneStress(law, committed, E, std::numeric_limits<float>::infinity(), out),
              nk::MfStatus::InvalidParameters);
    const float broken[3] = {std::nanf(""), 0.0f, 0.0f};
    EXPECT_EQ(mf::EvaluatePlaneStress(law, committed, broken, kStep, out), nk::MfStatus::NonFinite);
    EXPECT_EQ(out.psi, 7.0f);
    EXPECT_EQ(out.stress[1], -3.0f);
    EXPECT_EQ(out.history[4], 0.5f);

    nk::MfMaterial invalid = Yielding();
    EXPECT_TRUE(mf::ValidMaterial(invalid));
    invalid.poisson = 0.5f;
    EXPECT_FALSE(mf::ValidMaterial(invalid));
    invalid = Yielding();
    invalid.maxwell_time[1] = 0.0f;
    EXPECT_FALSE(mf::ValidMaterial(invalid));
    invalid = Yielding();
    invalid.yield_moment = -1.0f;
    EXPECT_FALSE(mf::ValidMaterial(invalid));
    invalid = Yielding();
    invalid.hardening = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(mf::ValidMaterial(invalid));
}

TEST(MaterialFieldResponse, UniaxialTensionFollowsTheBilinearCurve) {
    // S22 = S12 = 0 by Newton on E22; past yield S11 = Y + Et (E11 - Y/E) with Et = E H / (E + H).
    const mf::PlaneStressLaw law = Law(true, false);
    const double nu = law.poisson, young = law.modulus * (1.0 - nu * nu), H = law.hardening;
    float history[nk::kMfResponseStateCapacity] = {};
    double e22 = 0.0;
    int yielded = 0;
    for (int n = 1; n <= 40; ++n) {
        const double e11 = 2.5e-4 * n;
        mf::FieldResponse<double> r{};
        for (int it = 0; it < 20; ++it) {
            const double E[3] = {e11, e22, 0.0};
            ASSERT_EQ(mf::EvaluatePlaneStress(law, history, E, 1.0 / 240.0, r), nk::MfStatus::Ok);
            if (std::abs(r.stress[1]) < 1.0e-9 * law.yield) break;
            e22 -= r.stress[1] / r.tangent[1];
        }
        const double first = law.yield / young;
        const double expected = e11 <= first ? young * e11 : law.yield + young * H / (young + H) * (e11 - first);
        EXPECT_NEAR(r.stress[0], expected, 1.0e-5 * law.yield) << "step " << n;
        EXPECT_NEAR(r.stress[1], 0.0, 1.0e-8 * law.yield);
        EXPECT_NEAR(r.stress[2], 0.0, 1.0e-8 * law.yield);
        yielded += r.yielded ? 1 : 0;
        for (uint32_t k = 0u; k < nk::kMfResponseStateCapacity; ++k) history[k] = float(r.history[k]);
    }
    EXPECT_EQ(yielded, 32);
    EXPECT_NEAR(history[3], history[0], 1.0e-6f);
}

TEST(MaterialFieldResponse, HeldStrainRelaxesLikeAStandardLinearSolid) {
    mf::PlaneStressLaw law = Law(false, false);
    law.maxwell[0] = 0.8f;
    law.time[0] = 0.05f;
    const double h = 0.05 / 50.0, relax = h / law.time[0];
    const double E[3] = {0.003, 0.001, 0.0};
    mf::FieldResponse<double> elastic{};
    ASSERT_EQ(mf::EvaluatePlaneStress(Law(false, false), nullptr, E, h, elastic), nk::MfStatus::Ok);
    float history[nk::kMfResponseStateCapacity] = {};
    double dissipated = 0.0;
    for (int n = 1; n <= 400; ++n) {
        mf::FieldResponse<double> r{};
        ASSERT_EQ(mf::EvaluatePlaneStress(law, history, E, h, r), nk::MfStatus::Ok);
        const double discrete = 1.0 + law.maxwell[0] * std::pow(1.0 + relax, -n);
        const double continuous = 1.0 + law.maxwell[0] * std::exp(-n * relax);
        for (int k = 0; k < 2; ++k) {
            EXPECT_NEAR(r.stress[k] / elastic.stress[k], discrete, 1.0e-5) << "step " << n;
            EXPECT_NEAR(r.stress[k] / elastic.stress[k], continuous, 1.0e-2) << "step " << n;
        }
        dissipated += r.viscous;
        for (uint32_t k = 0u; k < nk::kMfResponseStateCapacity; ++k) history[k] = float(r.history[k]);
    }
    // The branch's loaded energy is dissipated in full; backward Euler keeps 2 / (2 + h / tau) of it.
    const double loaded = law.modulus * law.maxwell[0] * mf::UnitEnergy(double(law.poisson), E);
    EXPECT_NEAR(dissipated, loaded * 2.0 / (2.0 + relax), 1.0e-4 * loaded);
}

TEST(MaterialFieldResponse, ReturnedStressesAreAdmissibleAndDissipationIsNonnegative) {
    const mf::PlaneStressLaw law = Law(true, true);
    uint32_t seed = 12345u;
    const auto next = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return double(seed >> 8) / double(1u << 24) * 2.0 - 1.0;
    };
    int yielded = 0;
    for (int sample = 0; sample < 500; ++sample) {
        float committed[nk::kMfResponseStateCapacity];
        for (uint32_t k = 0u; k < nk::kMfResponseStateCapacity; ++k) committed[k] = float(0.003 * next());
        committed[3] = float(0.002 * (next() + 1.0));
        const double E[3] = {0.02 * next(), 0.02 * next(), 0.02 * next()};
        mf::FieldResponse<double> r{};
        ASSERT_EQ(mf::EvaluatePlaneStress(law, committed, E, 1.0 / 240.0, r), nk::MfStatus::Ok);
        const double kappa = law.yield + law.hardening * r.history[3], surface = 2.0 / 3.0 * kappa * kappa;
        const double measure = YieldMeasure(r.stress);
        EXPECT_LE(measure, surface * (1.0 + 1.0e-9)) << "sample " << sample;
        if (r.yielded) {
            ++yielded;
            EXPECT_NEAR(measure, surface, 1.0e-8 * surface) << "sample " << sample;
        }
        EXPECT_GE(r.history[3], double(committed[3]));
        EXPECT_GE(r.plastic, 0.0);
        EXPECT_GE(r.viscous, 0.0);
    }
    EXPECT_GT(yielded, 100);
}

TEST(MaterialFieldResponse, ElementGradientsMatchStoredEnergy) {
    const Points triangle = Deformed(Rotation(0.6f), Vec3{0.3f, -0.1f, 0.25f});
    const Points patch = BentPatch(0.35f);
    Points opened = patch;
    for (uint32_t i = 0u; i < 3u; ++i)
        for (uint32_t j = 0u; j < 3u; ++j)
            opened[3u + 3u * i + j] = Apply(Rotation(0.2f * (i + 1u)), patch[3u + 3u * i + j]) +
                                       Vec3{0.003f * i, -0.002f, 0.001f};
    for (const auto& [element, x, count] : {std::make_tuple(Triangle(), triangle, 3u),
                                            std::make_tuple(Bend(), patch, nk::kMfMaxVertices),
                                            std::make_tuple(Bend(), opened, nk::kMfMaxVertices)}) {
        const float largest = Largest(element, x, count);
        ASSERT_GT(largest, 0.0f);
        const float h = 2.0e-6f;
        for (uint32_t local = 0u; local < count; ++local) {
            Vec3 analytic = Gradient(element, x, local);
            for (int axis = 0; axis < 3; ++axis) {
                Points up = x, down = x;
                Axis(up[local], axis) += h;
                Axis(down[local], axis) -= h;
                const double numeric = (Energy(element, up) - Energy(element, down)) /
                                       (double(Axis(up[local], axis)) - double(Axis(down[local], axis)));
                EXPECT_NEAR(numeric, Axis(analytic, axis), 1.0e-4 * largest) << "local " << local << " axis " << axis;
            }
        }
    }
}

TEST(MaterialFieldResponse, EnergyChangeMatchesTheEnergyDifference) {
    const Points triangle = Deformed(Rotation(-0.4f), Vec3{0.3f, -0.1f, 0.25f});
    const Points patch = BentPatch(-0.25f);
    const Vec3 direction{1.5e-5f, -0.8e-5f, 1.1e-5f};
    for (const auto& [element, x, count] : {std::make_tuple(Triangle(), triangle, 3u),
                                            std::make_tuple(Bend(), patch, nk::kMfMaxVertices)}) {
        for (uint32_t trial = 0u; trial < (element.kind == nk::kMfBend ? 2u : 1u); ++trial) {
            const Vec3 move = direction * (trial == 0u ? 1.0f : 1.0e-5f);
            for (uint32_t local = 0u; local < count; ++local) {
                const mf::FieldGeometry g = Geometry(x);
                mf::FieldGeometry moved = g;
                moved.delta[local] = move;
                const float change = mf::ElementEnergyChange(element, Sheet(), nullptr, local, g, nullptr, kStep, move);
                const double expected = mf::ElementEnergy(element, Sheet(), nullptr, moved) -
                                        mf::ElementEnergy(element, Sheet(), nullptr, g);
                const double magnitude = std::abs(double(Gradient(element, x, local).Dot(move))) + std::abs(expected);
                EXPECT_NEAR(change, expected, 1.0e-3 * magnitude) << "local " << local;
            }
        }
    }
}

TEST(MaterialFieldResponse, YieldingElementChangeFollowsItsGradient) {
    const nk::MfMaterial m = Yielding();
    const Points triangle = Deformed(Rotation(0.6f), Vec3{0.3f, -0.1f, 0.25f});
    const Points patch = BentPatch(0.35f);
    // Small against the strain, whose direction sets the plastic tangent.
    const Vec3 move{1.0e-6f, -0.6e-6f, 0.8e-6f};
    for (const auto& [element, x, count] : {std::make_tuple(Triangle(), triangle, 3u),
                                            std::make_tuple(Bend(), patch, nk::kMfMaxVertices)}) {
        const mf::FieldGeometry g = Geometry(x);
        mf::FieldResponse<float> trial{};
        ASSERT_EQ(mf::ElementTrial(element, m, nullptr, g, kStep, trial), nk::MfStatus::Ok);
        ASSERT_TRUE(trial.yielded);
        for (uint32_t local = 0u; local < count; ++local) {
            const Vec3 gradient = Gradient(element, m, x, local);
            const float up = mf::ElementEnergyChange(element, m, nullptr, local, g, nullptr, kStep, move);
            const float down = mf::ElementEnergyChange(element, m, nullptr, local, g, nullptr, kStep, move * -1.0f);
            const double slope = gradient.Dot(move);
            EXPECT_NEAR(0.5 * (double(up) - double(down)), slope,
                        5.0e-3 * gradient.Length() * move.Length()) << "local " << local;
        }
    }
}

TEST(MaterialFieldKinematics, RigidMotionHasNoStrainOrTurn) {
    const Points rest = RestTriangle();
    const Vec3 offset{0.1f, 0.2f, -0.05f};
    for (float angle : {0.3f, 1.7f, -2.9f}) {
        const auto r = Rotation(angle);
        Points x{};
        for (uint32_t j = 0u; j < 3u; ++j) x[j] = Apply(r, rest[j]) + offset;
        const auto t = mf::Strain(Triangle(), Geometry(x).Corners(), nullptr);
        EXPECT_NEAR(t.g00, 0.0f, 1.0e-5f);
        EXPECT_NEAR(t.g11, 0.0f, 1.0e-5f);
        EXPECT_NEAR(t.g01, 0.0f, 1.0e-5f);
        EXPECT_NEAR(t.area_change, 0.0f, 1.0e-5f);
        EXPECT_NEAR(Energy(Triangle(), x), 0.0, 1.0e-6);
        const Points flat = BentPatch(0.0f);
        Points moved{};
        for (uint32_t j = 0u; j < nk::kMfMaxVertices; ++j) moved[j] = Apply(r, flat[j]) + offset;
        EXPECT_NEAR(Energy(Bend(true), moved), 0.0, 1.0e-10);
        for (uint32_t local = 0u; local < nk::kMfMaxVertices; ++local) {
            const Vec3 g = Gradient(Bend(true), moved, local);
            EXPECT_NEAR(g.Length(), 0.0f, 1.0e-5f) << "local " << local;
        }
    }
}

TEST(MaterialFieldKinematics, UniformStretchStrainIsExact) {
    const Points rest = RestTriangle();
    for (const auto& [a, b] : {std::make_pair(0.01f, -0.004f), std::make_pair(0.05f, 0.02f),
                               std::make_pair(-0.03f, 0.0f)}) {
        Points x{};
        for (uint32_t j = 0u; j < 3u; ++j) x[j] = Vec3{(1.0f + a) * rest[j].x, (1.0f + b) * rest[j].y, 0.0f};
        const auto t = mf::Strain(Triangle(), Geometry(x).Corners(), nullptr);
        const double e00 = 0.5 * ((1.0 + a) * (1.0 + a) - 1.0), e11 = 0.5 * ((1.0 + b) * (1.0 + b) - 1.0);
        const double scale = std::max(std::abs(e00), std::abs(e11));
        EXPECT_NEAR(t.g00, e00, 1.0e-5 * scale);
        EXPECT_NEAR(t.g11, e11, 1.0e-5 * scale);
        EXPECT_NEAR(t.g01, 0.0, 1.0e-5 * scale);
    }
}

TEST(MaterialFieldKinematics, RotationKeepsStrainEnergyAndRotatesTheGradient) {
    const auto r = Rotation(1.1f);
    const Points triangle = Deformed(Rotation(0.0f), Vec3{});
    Points turned_triangle{};
    for (uint32_t j = 0u; j < 3u; ++j) turned_triangle[j] = Apply(r, triangle[j]);
    const auto t = mf::Strain(Triangle(), Geometry(triangle).Corners(), nullptr);
    const auto u = mf::Strain(Triangle(), Geometry(turned_triangle).Corners(), nullptr);
    EXPECT_NEAR(u.g00, t.g00, 1.0e-5f);
    EXPECT_NEAR(u.g11, t.g11, 1.0e-5f);
    EXPECT_NEAR(u.g01, t.g01, 1.0e-5f);

    const Points patch = BentPatch(0.5f);
    for (const auto& [element, x, count] : {std::make_tuple(Triangle(), triangle, 3u),
                                            std::make_tuple(Bend(), patch, nk::kMfMaxVertices)}) {
        Points turned{};
        for (uint32_t j = 0u; j < count; ++j) turned[j] = Apply(r, x[j]);
        const double energy = Energy(element, x);
        ASSERT_GT(energy, 0.0);
        EXPECT_NEAR(Energy(element, turned), energy, 1.0e-4 * energy);
        const float largest = Largest(element, x, count);
        for (uint32_t local = 0u; local < count; ++local) {
            const Vec3 expected = Apply(r, Gradient(element, x, local));
            const Vec3 actual = Gradient(element, turned, local);
            EXPECT_NEAR(actual.x, expected.x, 1.0e-4f * largest);
            EXPECT_NEAR(actual.y, expected.y, 1.0e-4f * largest);
            EXPECT_NEAR(actual.z, expected.z, 1.0e-4f * largest);
        }
    }
}

TEST(MaterialFieldSeam, PartingSpendsTheToughnessAndClosingPullsNothing) {
    const mf::CohesiveLaw law = mf::SeamLaw(Seam(), Paper());
    const double onset = law.Onset(), parting = law.Parting();
    ASSERT_GT(parting, 10.0 * onset);
    EXPECT_NEAR(mf::EvaluateSeam(law, 0.0, onset).secant * onset, law.strength, 1.0e-9 * law.strength);
    // Steps that commit their largest separation dissipate G in all and store nothing once parted.
    double reached = 0.0, loss = 0.0;
    for (int k = 1; k <= 400; ++k) {
        const mf::SeamResponse s = mf::EvaluateSeam(law, reached, 1.2 * parting * k / 400.0);
        EXPECT_GE(s.loss, -1.0e-12 * law.toughness) << "step " << k;
        loss += s.loss;
        reached = s.reached;
    }
    EXPECT_NEAR(loss, law.toughness, 1.0e-9 * law.toughness);
    const mf::SeamResponse closed = mf::EvaluateSeam(law, reached, 0.3 * parting);
    EXPECT_EQ(closed.psi, 0.0);
    EXPECT_EQ(closed.secant, 0.0);
    EXPECT_EQ(closed.stored, 0.0);
}

TEST(MaterialFieldSeam, UnloadingFollowsTheDamagedStiffnessWithoutLoss) {
    const mf::CohesiveLaw law = mf::SeamLaw(Seam(), Paper());
    const double onset = law.Onset(), parting = law.Parting(), reached = 0.3 * parting;
    const double damage = mf::SeamDamage(law, reached);
    ASSERT_GT(damage, 0.0);
    ASSERT_LT(damage, 1.0);
    // The damaged secant meets the softening line at the reached separation.
    EXPECT_NEAR((1.0 - damage) * law.stiffness * reached, law.strength * (parting - reached) / (parting - onset),
                1.0e-9 * law.strength);
    for (const double r : {0.0, 0.2 * reached, reached}) {
        const mf::SeamResponse s = mf::EvaluateSeam(law, reached, r);
        EXPECT_NEAR(s.secant, (1.0 - damage) * law.stiffness, 1.0e-9 * law.stiffness);
        EXPECT_NEAR(s.loss, 0.0, 1.0e-12 * law.toughness);
        EXPECT_EQ(s.reached, reached);
    }
}

TEST(MaterialFieldSeam, PairGradientsFollowTheEnergyChange) {
    const nk::MfElement seam = Seam();
    const nk::MfMaterial m = Paper();
    EXPECT_EQ(nk::MfElementVertexCount(seam), 4u);
    const float onset = float(mf::SeamLaw(seam, m).Onset());
    // Pair 0 holds elastically, pair 1 softens; a committed reach makes pair 0 unload.
    const mf::FieldGeometry g = SeamGeometry(Vec3{0.3f, -0.2f, 0.25f} * onset, Vec3{-12.0f, 30.0f, 20.0f} * onset);
    float history[nk::kMfResponseStateCapacity]{};
    history[0] = 4.0f * onset;
    for (const float* committed : {static_cast<const float*>(nullptr), static_cast<const float*>(history)}) {
        const Vec3 move = Vec3{1.0f, -0.6f, 0.8f} * (0.01f * onset);
        for (uint32_t local = 0u; local < 4u; ++local) {
            Vec3 gradient;
            SymmetricMat3 hessian;
            mf::ElementBlock(seam, m, committed, local, g, nullptr, kStep, &gradient, &hessian);
            ASSERT_GT(gradient.Length(), 0.0f);
            ASSERT_GT(hessian.xx, 0.0f);
            const float up = mf::ElementEnergyChange(seam, m, committed, local, g, nullptr, kStep, move);
            const float down = mf::ElementEnergyChange(seam, m, committed, local, g, nullptr, kStep, move * -1.0f);
            EXPECT_NEAR(0.5 * (double(up) - double(down)), gradient.Dot(move),
                        1.0e-3 * gradient.Length() * move.Length()) << "local " << local;
        }
        mf::FieldResponse<float> trial{};
        ASSERT_EQ(mf::ElementTrial(seam, m, committed, g, kStep, trial), nk::MfStatus::Ok);
        const float softened = (Vec3{-12.0f, 30.0f, 20.0f} * onset).Length();
        EXPECT_NEAR(trial.history[1], softened, 1.0e-4f * softened);
        EXPECT_GT(trial.plastic, 0.0f);
        EXPECT_NEAR(mf::ElementEnergy(seam, m, trial.history, g), seam.rest[0] * trial.stored,
                    1.0e-4 * seam.rest[0] * trial.stored);
    }
}

TEST(MaterialFieldSeam, PartedHingesStopBending) {
    const Points patch = BentPatch(0.35f);
    float parted[nk::kMfResponseStateCapacity]{};
    for (uint32_t i = 0u; i < 3u; ++i) parted[nk::kMfHingeDamage + i] = 1.0f;
    ASSERT_GT(Energy(Bend(), patch), 0.0);
    EXPECT_EQ(mf::ElementEnergy(Bend(), Sheet(), parted, Geometry(patch)), 0.0);
    for (uint32_t local = 0u; local < nk::kMfMaxVertices; ++local) {
        Vec3 gradient;
        SymmetricMat3 hessian;
        mf::ElementBlock(Bend(), Sheet(), parted, local, Geometry(patch), nullptr, kStep, &gradient, &hessian);
        EXPECT_EQ(gradient.LengthSq(), 0.0f) << "local " << local;
    }
}

TEST(MaterialFieldSeam, DamageDegradesElasticStorageWithCommittedPlasticAndMaxwellHistory) {
    for (uint32_t mask = 1u; mask < 8u; ++mask) {
        auto element = Bend();
        element.hinges = mask;
        const auto material = Sheet();
        ASSERT_TRUE(mf::InitializeBendDual(element, material));
        double entries[6], projection[3][3];
        mf::BendMetric(element, material, nullptr, entries, projection);
        const double elastic[3] = {0.7, -0.4, 0.3};
        double full[3], condensed[3];
        mf::MetricProduct(mf::ElasticMetric<double>(mf::BendLaw(material)), elastic, full);
        mf::TangentProduct(entries, elastic, condensed);
        for (uint32_t i = 0u; i < 3u; ++i) {
            if (((mask >> i) & 1u) == 0u) continue;
            double expected = 0.0, actual = 0.0;
            for (uint32_t r = 0u; r < 3u; ++r) {
                expected += element.rest[3u * r + i] * full[r];
                actual += element.rest[3u * r + i] * condensed[r];
            }
            EXPECT_NEAR(actual, expected, 2e-6 * std::abs(expected) + 1e-8) << "hinges=" << mask;
        }
    }
    const auto patch = Geometry(BentPatch(0.35f));
    auto material = Yielding();
    material.bend_hardening = 0.0f;
    float history[nk::kMfResponseStateCapacity];
    std::memcpy(history, kCommitted, sizeof(history));
    for (uint32_t i = 0u; i < 3u; ++i) history[nk::kMfHingeDamage + i] = 1.0f;
    const auto bend = Bend();
    EXPECT_EQ(mf::ElementEnergy(bend, material, history, patch), 0.0);
    mf::FieldResponse<float> trial;
    ASSERT_EQ(mf::ElementTrial(bend, material, history, patch, kStep, trial), nk::MfStatus::Ok);
    EXPECT_EQ(trial.stored, 0.0f);
    EXPECT_EQ(trial.plastic, 0.0f);
    EXPECT_EQ(trial.viscous, 0.0f);
    EXPECT_FALSE(trial.beyond_range);
    for (uint32_t local = 0u; local < nk::kMfMaxVertices; ++local) {
        Vec3 gradient;
        SymmetricMat3 hessian;
        mf::ElementBlock(bend, material, history, local, patch, nullptr, kStep, &gradient, &hessian);
        EXPECT_EQ(gradient.LengthSq(), 0.0f);
    }
    history[nk::kMfHingeDamage] = 0.25f;
    history[nk::kMfHingeDamage + 1u] = 0.8f;
    const Vec3 move{1e-7f, -1.5e-7f, 0.5e-7f};
    for (uint32_t local = 0u; local < nk::kMfMaxVertices; ++local) {
        Vec3 gradient;
        SymmetricMat3 hessian;
        mf::ElementBlock(bend, material, history, local, patch, nullptr, kStep, &gradient, &hessian);
        const auto forward = mf::ElementEnergyChange(bend, material, history, local, patch, nullptr, kStep, move);
        const auto backward = mf::ElementEnergyChange(bend, material, history, local, patch, nullptr, kStep, -move);
        EXPECT_NEAR(0.5 * (forward - backward), gradient.Dot(move),
                    0.01 * gradient.Length() * move.Length() + 1e-12);
    }
}

TEST(MaterialFieldKinematics, SplitFacesKeepTheirOwnGeometry) {
    const nk::MfElement e = Bend();
    const Points x = BentPatch(0.35f);
    const double energy = Energy(e, x);
    Points moved = x;
    for (uint32_t i = 0u; i < 3u; ++i) {
        const uint32_t first = mf::HingeFace(e, i);
        const Vec3 shift = x[(i + 1u) % 3u] - x[first + 2u];
        for (uint32_t j = 0u; j < 3u; ++j) moved[first + j] += shift;
    }
    EXPECT_NEAR(Energy(e, moved), energy, 1.0e-6 * energy);
    const float largest = Largest(e, x, nk::kMfMaxVertices);
    for (uint32_t local = 0u; local < nk::kMfMaxVertices; ++local)
        EXPECT_NEAR((Gradient(e, moved, local) - Gradient(e, x, local)).Length(), 0.0f, 2.0e-6f * largest);
}

TEST(MaterialFieldKinematics, JoinedHingeAliasesGiveTheSharedEdgeDerivative) {
    nk::MfElement e = Bend();
    const Points x = BentPatch(0.35f);
    for (uint32_t i = 0u; i < 3u; ++i) {
        const uint32_t first = mf::HingeFace(e, i);
        e.vertex[first] = (i + 1u) % 3u;
        e.vertex[first + 1u] = (i + 2u) % 3u;
        nk::vbd::Point joined[6];
        mf::HingePoints(e, Geometry(x), i, joined);
        Vec3 q[6], original[4];
        const Vec3 shared[4] = {x[(i + 1u) % 3u], x[(i + 2u) % 3u], x[i], x[first + 2u]};
        float theta, reference;
        ASSERT_TRUE(mf::HingeAngleGradient(joined, q, &theta));
        ASSERT_TRUE(nk::vbd::HingeAngleGradient(shared, original, &reference));
        EXPECT_NEAR(theta, reference, 1.0e-6f);
        EXPECT_NEAR((q[0] + q[3] - original[0]).Length(), 0.0f, 1.0e-4f);
        EXPECT_NEAR((q[1] + q[4] - original[1]).Length(), 0.0f, 1.0e-4f);
        EXPECT_NEAR((q[2] - original[2]).Length(), 0.0f, 1.0e-4f);
        EXPECT_NEAR((q[5] - original[3]).Length(), 0.0f, 1.0e-4f);
    }
    const Vec3 move{1.5e-5f, -0.8e-5f, 1.1e-5f};
    const mf::FieldGeometry g = Geometry(x);
    for (uint32_t local = 0u; local < 3u; ++local) {
        mf::FieldGeometry moved = g;
        Vec3 shifts[nk::kMfMaxVertices]{};
        for (uint32_t j = 0u; j < nk::MfElementVertexCount(e); ++j)
            if (e.vertex[j] == e.vertex[local]) moved.delta[j] = shifts[j] = move;
        const double expected = mf::ElementEnergy(e, Sheet(), nullptr, moved) - mf::ElementEnergy(e, Sheet(), nullptr, g);
        const float single = mf::ElementEnergyChange(e, Sheet(), nullptr, local, g, nullptr, kStep, move);
        const float together = mf::ElementMovesEnergyChange(e, Sheet(), nullptr, g, nullptr, kStep, shifts);
        EXPECT_NEAR(single, expected, 1.0e-4 * std::abs(expected));
        EXPECT_NEAR(together, expected, 1.0e-4 * std::abs(expected));
        const float h = 1.0e-6f;
        for (int axis = 0; axis < 3; ++axis) {
            mf::FieldGeometry up = g, down = g;
            for (uint32_t j = 0u; j < nk::MfElementVertexCount(e); ++j)
                if (e.vertex[j] == e.vertex[local]) {
                    Axis(up.delta[j], axis) = h;
                    Axis(down.delta[j], axis) = -h;
                }
            const double numeric = (mf::ElementEnergy(e, Sheet(), nullptr, up) -
                                    mf::ElementEnergy(e, Sheet(), nullptr, down)) / (2.0 * h);
            Vec3 gradient = Gradient(e, x, local);
            EXPECT_NEAR(Axis(gradient, axis), numeric, 1.0e-4 * gradient.Length());
        }
    }
}
