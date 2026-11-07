#include <algorithm>
#include <array>
#include <cmath>

#include <gtest/gtest.h>

#include "nk/solve/vertex_block_forms.hpp"

namespace {
using namespace nuka;
using Point = std::array<long double, 3>;
using Points = std::array<Point, 3>;

long double Dot(const Point& a, const Point& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

Point Cross(const Point& a, const Point& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]};
}

struct Reference {
    long double energy;
    Points gradient;
};

Reference Evaluate(const Points& x) {
    Point f0{}, f1{};
    for (unsigned a = 0; a < 3; ++a) {
        f0[a] = x[1][a] - x[0][a];
        f1[a] = x[2][a] - x[0][a];
    }
    Point normal = Cross(f0, f1);
    const long double area = std::sqrt(Dot(normal, normal));
    for (auto& value : normal) value /= area;
    const Point c0 = Cross(f1, normal), c1 = Cross(normal, f0);
    Reference result{};
    result.energy = 500 * (Dot(f0, f0) + Dot(f1, f1) - 2 - 2 * (area - 1)) / 2 +
                    500 * (area - 1) * (area - 1);
    for (unsigned a = 0; a < 3; ++a) {
        result.gradient[1][a] = 500 * (f0[a] + (2 * (area - 1) - 1) * c0[a]);
        result.gradient[2][a] = 500 * (f1[a] + (2 * (area - 1) - 1) * c1[a]);
        result.gradient[0][a] = -result.gradient[1][a] - result.gradient[2][a];
    }
    return result;
}

nk::VbdElement Element() {
    nk::VbdElement e;
    e.kind = nk::kVbdTriangle;
    e.law = nk::kVbdMembraneNeoHookean;
    e.rest[0] = e.rest[3] = 1.0f;
    e.rest[4] = 0.5f;
    e.rest[5] = e.rest[6] = 1000.0f;
    return e;
}

nk::vbd::ElementGeometry Geometry(unsigned angle, float strain) {
    const double a = angle * 0.071, b = angle * 0.037;
    nk::vbd::ElementGeometry g;
    g.start[1] = {float((1 + double(strain)) * std::cos(a)), float(std::sin(a)), 0.0f};
    g.start[2] = {float(-std::sin(a) * std::cos(b)), float(std::cos(a) * std::cos(b)), float(std::sin(b))};
    return g;
}

Points Precise(const nk::vbd::ElementGeometry& g, const math::Vec3* move = nullptr) {
    Points points{};
    for (unsigned j = 0; j < 3; ++j) {
        points[j] = {g.start[j].x, g.start[j].y, g.start[j].z};
        if (move != nullptr) {
            points[j][0] += move[j].x;
            points[j][1] += move[j].y;
            points[j][2] += move[j].z;
        }
    }
    return points;
}

TEST(MembraneEnergy, SmallStrainForceRetainsRotationalAccuracy) {
    const auto e = Element();
    for (unsigned angle = 1; angle <= 64; ++angle) {
        const auto g = Geometry(angle, 1.0e-6f);
        const auto reference = Evaluate(Precise(g));
        for (unsigned local = 0; local < 3; ++local) {
            math::Vec3 force;
            math::SymmetricMat3 hessian;
            nk::vbd::ElementBlock(e, local, g, &force, &hessian);
            const float values[] = {force.x, force.y, force.z};
            for (unsigned a = 0; a < 3; ++a)
                EXPECT_NEAR(values[a], double(reference.gradient[local][a]),
                    5e-6 * std::max(1e-3, double(std::abs(reference.gradient[local][a]))));
        }
    }
}

TEST(MembraneEnergy, SmallMovesMatchIndependentStoredEnergy) {
    const auto e = Element();
    for (float strain : {1e-6f, -0.2f, 0.3f}) {
        for (unsigned angle = 1; angle <= 64; ++angle) {
            const auto g = Geometry(angle, strain);
            const long double before = Evaluate(Precise(g)).energy;
            math::Vec3 move[3]{{}, {1e-7f, -2e-7f, 3e-7f}, {}};
            const double expected = double(Evaluate(Precise(g, move)).energy - before);
            const double tolerance = 1e-13 + 2e-5 * std::abs(expected);
            EXPECT_NEAR(nk::vbd::ElementEnergyChange(e, 1u, g, move[1]), expected, tolerance);
            EXPECT_NEAR(nk::vbd::ElementMovesEnergyChange(e, g, move), expected, tolerance);
            move[0] = {-2e-7f, 3e-7f, 1e-7f};
            move[2] = {3e-7f, 1e-7f, -2e-7f};
            const double all = double(Evaluate(Precise(g, move)).energy - before);
            EXPECT_NEAR(nk::vbd::ElementMovesEnergyChange(e, g, move), all,
                        1e-13 + 2e-5 * std::abs(all));
        }
    }
}
}  // namespace
