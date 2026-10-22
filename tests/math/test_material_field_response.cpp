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
using nuka::math::SymmetricMat3;
using nuka::math::Vec3;
using Points = std::array<Vec3, 6>;

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

// Bend over the rest triangle and the far vertices beyond its three edges, with a hinge-space stiffness.
nk::MfElement Bend(bool flat = false) {
    nk::MfElement e;
    e.kind = nk::kMfBend;
    e.hinges = 7u;
    for (uint32_t j = 0u; j < 6u; ++j) e.vertex[j] = j;
    const float stiffness[6] = {0.04f, 0.05f, 0.03f, 0.01f, -0.008f, 0.006f};
    for (uint32_t k = 0u; k < 6u; ++k) e.rest[k] = stiffness[k];
    e.rest[6] = 0.0f;
    e.rest[7] = flat ? 0.0f : 0.05f;
    e.rest[8] = flat ? 0.0f : -0.03f;
    return e;
}

Points RestTriangle() {
    return {Vec3{0.0f, 0.0f, 0.0f}, Vec3{kA, 0.0f, 0.0f}, Vec3{kC, kD, 0.0f}, Vec3{}, Vec3{}, Vec3{}};
}

// The rest triangle with far vertices beyond edges (1, 2), (2, 0) and (0, 1), lifted by `angle`.
Points BentPatch(float angle) {
    const float s = std::sin(angle);
    return {Vec3{0.0f, 0.0f, 0.0f}, Vec3{kA, 0.0f, 0.0f}, Vec3{kC, kD, 0.0f},
            Vec3{0.022f, 0.019f, 0.012f * s}, Vec3{-0.012f, 0.012f, -0.009f * s}, Vec3{0.011f, -0.015f, 0.015f * s}};
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

nk::mf::FieldGeometry Geometry(const Points& x) {
    nk::mf::FieldGeometry g{};
    for (uint32_t j = 0u; j < 6u; ++j) g.start[j] = x[j];
    return g;
}

double Energy(const nk::MfElement& e, const Points& x) {
    return nk::mf::ElementEnergy(e, Sheet(), nullptr, Geometry(x), kStep);
}

Vec3 Gradient(const nk::MfElement& e, const Points& x, uint32_t local) {
    Vec3 gradient;
    SymmetricMat3 hessian;
    nk::mf::ElementBlock(e, Sheet(), nullptr, local, Geometry(x), nullptr, kStep, &gradient, &hessian);
    return gradient;
}

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
}  // namespace

TEST(MaterialFieldResponse, MembraneStressAndTangentMatchFiniteDifferences) {
    const nk::MfMaterial m = Sheet();
    const float committed[nk::kMfTriangleHistory] = {0.002f, -0.001f, 0.0005f};
    const double strains[3][3] = {{0.01, -0.004, 0.003}, {-0.02, 0.015, -0.008}, {0.0003, 0.0001, 0.0}};
    const double modulus = nk::mf::MembraneModulus<double>(m), scale = modulus * 0.02;
    for (const auto& strain : strains) {
        nk::mf::MembraneResult<double> r{};
        const double E[3] = {strain[0], strain[1], strain[2]};
        ASSERT_EQ(nk::mf::EvaluateMembrane(m, committed, E, 1.0 / 240.0, r), nk::MfStatus::Ok);
        const double h = 1.0e-6;
        for (int k = 0; k < 3; ++k) {
            double up[3] = {E[0], E[1], E[2]}, down[3] = {E[0], E[1], E[2]};
            up[k] += h;
            down[k] -= h;
            nk::mf::MembraneResult<double> a{}, b{};
            ASSERT_EQ(nk::mf::EvaluateMembrane(m, committed, up, 1.0 / 240.0, a), nk::MfStatus::Ok);
            ASSERT_EQ(nk::mf::EvaluateMembrane(m, committed, down, 1.0 / 240.0, b), nk::MfStatus::Ok);
            EXPECT_NEAR((a.psi - b.psi) / (2.0 * h), r.stress[k], 1.0e-8 * scale);
            for (int i = 0; i < 3; ++i)
                EXPECT_NEAR((a.stress[i] - b.stress[i]) / (2.0 * h), r.tangent[TangentIndex(i, k)], 1.0e-8 * modulus);
        }
        nk::mf::MembraneResult<float> f{};
        const float Ef[3] = {float(E[0]), float(E[1]), float(E[2])};
        ASSERT_EQ(nk::mf::EvaluateMembrane(m, committed, Ef, kStep, f), nk::MfStatus::Ok);
        for (int k = 0; k < 3; ++k) EXPECT_NEAR(f.stress[k], r.stress[k], 1.0e-4 * scale);
    }
}

TEST(MaterialFieldResponse, BendMomentAndTangentMatchFiniteDifferences) {
    const nk::MfMaterial m = Sheet();
    const float committed[nk::kMfBendHistory] = {0.05f, -0.02f, 0.01f};
    const nk::MfElement e = Bend();
    const double turns[3][3] = {{-0.7, 0.2, 0.01}, {0.01, -0.03, 0.4}, {2.5, 1.2, -0.9}};
    for (const auto& turn : turns) {
        nk::mf::BendResult<double> r{};
        const double t[3] = {turn[0], turn[1], turn[2]};
        ASSERT_EQ(nk::mf::EvaluateBend(m, committed, t, e.rest, 1.0 / 240.0, r), nk::MfStatus::Ok);
        const double h = 1.0e-6, scale = 0.1 * std::max({std::abs(t[0]), std::abs(t[1]), std::abs(t[2]), 0.1});
        for (int k = 0; k < 3; ++k) {
            double up[3] = {t[0], t[1], t[2]}, down[3] = {t[0], t[1], t[2]};
            up[k] += h;
            down[k] -= h;
            nk::mf::BendResult<double> a{}, b{};
            ASSERT_EQ(nk::mf::EvaluateBend(m, committed, up, e.rest, 1.0 / 240.0, a), nk::MfStatus::Ok);
            ASSERT_EQ(nk::mf::EvaluateBend(m, committed, down, e.rest, 1.0 / 240.0, b), nk::MfStatus::Ok);
            EXPECT_NEAR((a.psi - b.psi) / (2.0 * h), r.moment[k], 1.0e-8 * scale);
            for (int i = 0; i < 3; ++i)
                EXPECT_NEAR((a.moment[i] - b.moment[i]) / (2.0 * h), r.tangent[TangentIndex(i, k)], 1.0e-8 * 0.1);
        }
    }
}

TEST(MaterialFieldResponse, TrialsAreReadOnlyAndFailuresLeaveTheOutput) {
    const nk::MfMaterial m = Sheet();
    float committed[nk::kMfTriangleHistory] = {0.001f, 0.002f, -0.003f, 0.1f};
    float before[nk::kMfTriangleHistory];
    std::memcpy(before, committed, sizeof(committed));
    const float E[3] = {0.01f, -0.002f, 0.004f};
    for (int trial = 0; trial < 100; ++trial) {
        nk::mf::MembraneResult<float> r{};
        ASSERT_EQ(nk::mf::EvaluateMembrane(m, committed, E, kStep, r), nk::MfStatus::Ok);
    }
    EXPECT_EQ(std::memcmp(before, committed, sizeof(committed)), 0);

    nk::mf::MembraneResult<float> sentinel{};
    sentinel.psi = 7.0f;
    sentinel.stress[1] = -3.0f;
    nk::MfMaterial invalid = m;
    invalid.youngs = 0.0f;
    nk::mf::MembraneResult<float> out = sentinel;
    EXPECT_EQ(nk::mf::EvaluateMembrane(invalid, committed, E, kStep, out), nk::MfStatus::InvalidParameters);
    invalid = m;
    invalid.poisson = 0.5f;
    EXPECT_EQ(nk::mf::EvaluateMembrane(invalid, committed, E, kStep, out), nk::MfStatus::InvalidParameters);
    EXPECT_EQ(nk::mf::EvaluateMembrane(m, committed, E, 0.0f, out), nk::MfStatus::InvalidParameters);
    EXPECT_EQ(std::memcmp(&out, &sentinel, sizeof(out)), 0);
    nk::mf::BendResult<float> bend{};
    bend.psi = 5.0f;
    const nk::mf::BendResult<float> bend_before = bend;
    const float turn[3] = {0.2f, -0.1f, 0.0f};
    invalid = m;
    invalid.thickness = -1.0f;
    EXPECT_EQ(nk::mf::EvaluateBend(invalid, nullptr, turn, Bend().rest, kStep, bend), nk::MfStatus::InvalidParameters);
    EXPECT_EQ(std::memcmp(&bend, &bend_before, sizeof(bend)), 0);
}

TEST(MaterialFieldResponse, ElementGradientsMatchStoredEnergy) {
    const Points triangle = Deformed(Rotation(0.6f), Vec3{0.3f, -0.1f, 0.25f});
    const Points patch = BentPatch(0.35f);
    for (const auto& [element, x, count] : {std::make_tuple(Triangle(), triangle, 3u),
                                            std::make_tuple(Bend(), patch, 6u)}) {
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
    const Vec3 move{1.5e-5f, -0.8e-5f, 1.1e-5f};
    for (const auto& [element, x, count] : {std::make_tuple(Triangle(), triangle, 3u),
                                            std::make_tuple(Bend(), patch, 6u)}) {
        for (uint32_t local = 0u; local < count; ++local) {
            const nk::mf::FieldGeometry g = Geometry(x);
            nk::mf::FieldGeometry moved = g;
            moved.delta[local] = move;
            const float change = nk::mf::ElementEnergyChange(element, Sheet(), nullptr, local, g, nullptr, kStep, move);
            const double expected = nk::mf::ElementEnergy(element, Sheet(), nullptr, moved, kStep) -
                                    nk::mf::ElementEnergy(element, Sheet(), nullptr, g, kStep);
            const double scale = std::abs(double(Gradient(element, x, local).Dot(move))) + std::abs(expected);
            EXPECT_NEAR(change, expected, 1.0e-3 * scale) << "local " << local;
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
        const auto t = nk::mf::Strain(Triangle(), Geometry(x).Corners(), nullptr);
        EXPECT_NEAR(t.g00, 0.0f, 1.0e-5f);
        EXPECT_NEAR(t.g11, 0.0f, 1.0e-5f);
        EXPECT_NEAR(t.g01, 0.0f, 1.0e-5f);
        EXPECT_NEAR(t.area_change, 0.0f, 1.0e-5f);
        EXPECT_NEAR(Energy(Triangle(), x), 0.0, 1.0e-6);
        const Points flat = BentPatch(0.0f);
        Points moved{};
        for (uint32_t j = 0u; j < 6u; ++j) moved[j] = Apply(r, flat[j]) + offset;
        EXPECT_NEAR(Energy(Bend(true), moved), 0.0, 1.0e-10);
        for (uint32_t local = 0u; local < 6u; ++local) {
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
        const auto t = nk::mf::Strain(Triangle(), Geometry(x).Corners(), nullptr);
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
    const auto t = nk::mf::Strain(Triangle(), Geometry(triangle).Corners(), nullptr);
    const auto u = nk::mf::Strain(Triangle(), Geometry(turned_triangle).Corners(), nullptr);
    EXPECT_NEAR(u.g00, t.g00, 1.0e-5f);
    EXPECT_NEAR(u.g11, t.g11, 1.0e-5f);
    EXPECT_NEAR(u.g01, t.g01, 1.0e-5f);

    const Points patch = BentPatch(0.5f);
    for (const auto& [element, x, count] : {std::make_tuple(Triangle(), triangle, 3u),
                                            std::make_tuple(Bend(), patch, 6u)}) {
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
