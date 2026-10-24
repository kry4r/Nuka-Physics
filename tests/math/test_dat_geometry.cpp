#include <gtest/gtest.h>

#include "collision/dat_geometry.hpp"

TEST(DatGeometry, RoundedIndependentPrefixesKeepSeparatingPlane) {
    using nuka::math::Vec3;
    using namespace nuka::collision;
    const Vec3 start[4] = {
        {0.489197641611f, -0.002697866876f, 0.746767997742f},
        {0.490289270878f, 0.000723661156f, 0.749464452267f},
        {0.485279142857f, -0.001923454460f, 0.746918261051f},
        {0.491967558861f, 0.000391944486f, 0.749400377274f}};
    const Vec3 end[4] = {
        {0.489197611809f, -0.002697878517f, 0.746768057346f},
        {0.490289211273f, 0.000723625883f, 0.749464511871f},
        {0.485277801752f, -0.001924418029f, 0.746917247772f},
        {0.491967469454f, 0.000391969836f, 0.749400258064f}};
    DatPrimitive a, b;
    a.count = b.count = 2u;
    for (int i = 0; i < 2; ++i) { a.vertex[i] = start[i]; b.vertex[i] = start[i + 2]; }
    const auto separator = DatFindSeparator(a, b);
    ASSERT_GT(separator.gap, 0.0f);
    double minimum_a = DBL_MAX, maximum_b = -DBL_MAX;
    for (int i = 0; i < 4; ++i) {
        const float sign = i < 2 ? 1.0f : -1.0f;
        const float beta = DatLinearPlaneFraction(start[i], end[i], separator.negative_support,
            separator.normal * sign, sign * 0.5f * separator.gap, 0.9f);
        ASSERT_GE(beta, 0.0f);
        ASSERT_LE(beta, 1.0f);
        for (int j = 0; j <= 1024; ++j) {
            const Vec3 point = DatLinearPoint(start[i], end[i], beta * (float(j) / 1024.0f));
            const Vec3 origin = separator.negative_support, normal = separator.normal;
            const double projection = (double(point.x) - origin.x) * normal.x +
                (double(point.y) - origin.y) * normal.y + (double(point.z) - origin.z) * normal.z;
            if (i < 2) minimum_a = fmin(minimum_a, projection);
            else maximum_b = fmax(maximum_b, projection);
        }
    }
    EXPECT_GT(minimum_a, maximum_b);
    EXPECT_EQ(DatLinearPlaneFraction({0, 0, 1}, {1, 0, 2}, {}, {0, 0, 1}, 0, 0.9f), 1.0f);
}

TEST(OgcGeometry, ExtrapolatedPlaneContactPreservesTorqueAndGapDerivative) {
    using nuka::math::Vec3;
    const Vec3 face[3] = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    const Vec3 wing = {-0.4f, 1.0f, 0.001f};
    Vec3 weights;
    ASSERT_TRUE(nuka::collision::OgcPlaneCoordinates(wing, face[0], face[1], face[2], &weights));
    ASSERT_LT(weights.y, 0.0f);
    const Vec3 normal{0, 0, 1};
    const Vec3 torque = wing.Cross(normal) - face[0].Cross(normal * weights.x) -
        face[1].Cross(normal * weights.y) - face[2].Cross(normal * weights.z);
    EXPECT_LT(torque.Length(), 1.0e-6f);
    const Vec3 velocity[3] = {{0.2f, 0.1f, -0.3f}, {-0.1f, 0.3f, 0.7f}, {0.1f, -0.2f, 0.2f}};
    const auto gap = [&](float time) {
        const Vec3 a = face[0] + velocity[0] * time, b = face[1] + velocity[1] * time;
        const Vec3 c = face[2] + velocity[2] * time, n = (b - a).Cross(c - a);
        return (wing - a).Dot(n) / n.Length();
    };
    const float derivative = -(velocity[0] * weights.x + velocity[1] * weights.y + velocity[2] * weights.z).Dot(normal);
    EXPECT_NEAR((gap(1.0e-4f) - gap(-1.0e-4f)) / 2.0e-4f, derivative, 2.0e-6f);
}

TEST(OgcGeometry, IncidentFaceCannotSuppressSeparatedEdgeContact) {
    using nuka::math::Vec3;
    using namespace nuka::collision;
    const Vec3 points[] = {
        {0.4815663695f, -0.00716330949f, 0.7497904301f},
        {0.4783380628f, -0.00974297244f, 0.7478183508f},
        {0.4814806879f, -0.00724639697f, 0.7497213483f},
        {0.4786486626f, -0.00586205162f, 0.7521821260f},
        {0.4786316156f, -0.00547571667f, 0.7516648769f},
        points[1], {0.4846546352f, -0.00596351828f, 0.7521480918f}};
    uint32_t roots[] = {0, 1, 2, 3, 4, 1, 6};
    const MeshEdge edges[] = {{0, 1, 4, ~0u, 0, ~0u}, {2, 3, 5, 6, 1, 2}};
    MeshSurfaceView view;
    view.vertices = reinterpret_cast<const float*>(points);
    view.counts.vertices = 7u;
    MeshSurfaceInfo surface;
    surface.vertex_count = 7u;
    MeshEdgeInfo edge_info;
    edge_info.edge_count = 2u;
    MeshVertexExclusion excluded_a{roots, nullptr, 0u, {2, 3, ~0u}, 2u};
    MeshVertexExclusion excluded_b{roots, nullptr, 0u, {0, 1, ~0u}, 2u};
    const auto contact = [&](MeshVertexExclusion a, MeshVertexExclusion b) {
        return OgcEdgeContact(view, surface, edges, {}, edge_info, 0u,
            view, surface, edges, {}, edge_info, 1u, 1.0e-4f, a, b);
    };
    EXPECT_FALSE(contact({}, {}).feasible);
    const auto pair = contact(excluded_a, excluded_b);
    EXPECT_TRUE(pair.feasible);
    EXPECT_GT(pair.distance, 0.0f);
    EXPECT_LT(pair.distance, 1.0e-6f);
    roots[5] = 5u;
    EXPECT_FALSE(contact(excluded_a, excluded_b).feasible);
}

TEST(DatGeometry, SharedTranslationRetainsThinGap) {
    using nuka::math::Vec3;
    using namespace nuka::collision;
    DatSweptPrimitive a, b;
    a.count = b.count = 4u;
    a.vertex[0] = {0, 0, 0.7501f}; a.vertex[1] = {1, 0, 0.7501f};
    b.vertex[0] = {0, 0, 0.75f}; b.vertex[1] = {1, 0, 0.75f};
    for (uint32_t i = 0; i < 2u; ++i) {
        a.vertex[i + 2u] = a.vertex[i] + Vec3{0, 0, 0.001f};
        b.vertex[i + 2u] = b.vertex[i] + Vec3{0, 0, 0.001f};
    }
    EXPECT_FALSE(DatSweptSeparated(a, b, {0, 0, 1}, 1.0e-5f));
    EXPECT_TRUE(DatSharedLinearSeparated(a, b, {0, 0, 1}, 1.0e-5f));
    b.vertex[2].z += 0.0002f;
    EXPECT_FALSE(DatSharedLinearSeparated(a, b, {0, 0, 1}, 1.0e-5f));
}

TEST(DatGeometry, SeparatingVertexKeepsRoomForRounding) {
    using nuka::math::Vec3;
    using namespace nuka::collision;
    const Vec3 start{0.496709943f, -0.00258707535f, 0.74065119f};
    const Vec3 end{0.496660709f, -0.00239218911f, 0.740837514f};
    const Vec3 origin{0.496737838f, -0.00255289744f, 0.740645051f};
    const Vec3 normal{-0.0792273805f, 0.182122365f, 0.980078816f};
    const Vec3 other_end = origin + normal * 0.0003f;
    const float gap = 2.00243084691e-6f;
    const float offset = DatLinearPlaneOffset(gap, 0.95f, DatLinearRoundoff(start, end, normal),
                                              DatLinearRoundoff(origin, other_end, normal));
    EXPECT_EQ(DatLinearPlaneFraction(start, end, origin, normal, offset, 0.9f), 1.0f);
    const float beta = DatLinearPlaneFraction(origin, other_end, origin, -normal, -offset, 0.9f);
    EXPECT_GT(beta, 0.0f);
    EXPECT_LT(beta, 1.0f);
}
