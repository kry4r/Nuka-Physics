#pragma once

#include "collision/ogc_geometry.hpp"

#if defined(__CUDACC__)
#define NUKA_SEGMENT_HD __host__ __device__
#else
#define NUKA_SEGMENT_HD
#endif

namespace nuka::collision {

struct MeshSegmentPoint {
    math::Vec3 axis_point{};
    MeshSurfacePoint surface{};
};

// The minimum of segment endpoints, triangle edges and an interior plane crossing.
NUKA_SEGMENT_HD inline math::Vec3 ClosestSegmentTriangleAxis(
    math::Vec3 p, math::Vec3 q, math::Vec3 a, math::Vec3 b, math::Vec3 c) {
    const math::Vec3 direction = q - p, normal = (b - a).Cross(c - a);
    const float denominator = normal.Dot(direction);
    if (denominator != 0.0f) {
        const float t = normal.Dot(a - p) / denominator;
        if (t >= 0.0f && t <= 1.0f) {
            const math::Vec3 hit = p + direction * t;
            if (normal.Dot((b - a).Cross(hit - a)) >= 0.0f &&
                normal.Dot((c - b).Cross(hit - b)) >= 0.0f &&
                normal.Dot((a - c).Cross(hit - c)) >= 0.0f) return hit;
        }
    }
    math::Vec3 best = p;
    float distance_sq = (p - ClosestTrianglePoint(p, a, b, c).point).LengthSq();
    const float end_sq = (q - ClosestTrianglePoint(q, a, b, c).point).LengthSq();
    if (end_sq < distance_sq) { best = q; distance_sq = end_sq; }
    const math::Vec3 vertices[3] = {a, b, c};
    for (uint32_t edge = 0u; edge < 3u; ++edge) {
        const auto pair = OgcClosestSegments(p, q, vertices[edge], vertices[(edge + 1u) % 3u]);
        const float sq = (pair.a - pair.b).LengthSq();
        if (sq < distance_sq) { best = pair.a; distance_sq = sq; }
    }
    return best;
}

// Uses local mesh coordinates; the point query supplies the same sign and seam normals as spheres.
NUKA_SEGMENT_HD inline MeshSegmentPoint QueryMeshSegment(
    const MeshSurfaceView& view, const MeshSurfaceInfo& info,
    math::Vec3 p, math::Vec3 q, float reach) {
    MeshSegmentPoint result;
    if (!MeshSurfaceRangeValid(view, info) || view.transformed || !(reach >= 0.0f)) return result;
    float best_sq = reach * reach;
    uint32_t best_triangle = ~0u;
    for (uint32_t cursor = 0u; cursor < info.node_count;) {
        const auto& node = view.nodes[info.node_offset + cursor];
        if (node.escape <= cursor || node.escape > info.node_count) return {};
        if (!SegmentWithinBox(p, q, node.lower, node.upper, sqrtf(best_sq))) {
            cursor = node.escape;
            continue;
        }
        if (node.triangle != ~0u) {
            math::Vec3 a, b, c;
            if (!MeshSurfaceTriangle(view, info, node.triangle, a, b, c)) return {};
            if ((b - a).Cross(c - a).LengthSq() > 0.0f) {
                const auto axis = ClosestSegmentTriangleAxis(p, q, a, b, c);
                const float sq = (axis - ClosestTrianglePoint(axis, a, b, c).point).LengthSq();
                if (sq < best_sq || (sq == best_sq && node.triangle < best_triangle)) {
                    best_sq = sq;
                    best_triangle = node.triangle;
                    result.axis_point = axis;
                }
            }
        }
        ++cursor;
    }
    if (best_triangle != ~0u) result.surface = QueryMeshSurface(view, info, result.axis_point, reach);
    else { result.surface.valid = true; result.surface.distance = FLT_MAX; }
    return result;
}

}  // namespace nuka::collision

#undef NUKA_SEGMENT_HD
