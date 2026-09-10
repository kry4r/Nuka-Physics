#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>

#include "collision/mesh_surface_types.hpp"

#if defined(__CUDACC__)
#define NUKA_MESH_HD __host__ __device__
#else
#define NUKA_MESH_HD
#endif

namespace nuka::collision {

struct TriangleClosestPoint {
    math::Vec3 point{};
    math::Vec3 barycentric{};
    uint32_t feature = 0u;
};

NUKA_MESH_HD inline TriangleClosestPoint ClosestTrianglePoint(
    math::Vec3 p, math::Vec3 a, math::Vec3 b, math::Vec3 c) {
    using math::Vec3;
    const Vec3 ab = b - a, ac = c - a, ap = p - a;
    const float d1 = ab.Dot(ap), d2 = ac.Dot(ap);
    const Vec3 bp = p - b;
    const float d3 = ab.Dot(bp), d4 = ac.Dot(bp);
    const Vec3 cp = p - c;
    const float d5 = ab.Dot(cp), d6 = ac.Dot(cp);
    const float va = d3 * d6 - d5 * d4;
    const float vb = d5 * d2 - d1 * d6;
    const float vc = d1 * d4 - d3 * d2;
    if (!(ab.Cross(ac).LengthSq() > 0.0f) || !(va + vb + vc > 0.0f)) {
        const Vec3 vertices[3] = {a, b, c};
        const Vec3 weights[3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        TriangleClosestPoint best{a, weights[0], 0u};
        float best_sq = (p - a).LengthSq();
        for (uint32_t i = 0u; i < 3u; ++i) {
            const uint32_t j = (i + 1u) % 3u;
            const Vec3 edge = vertices[j] - vertices[i];
            const float length_sq = edge.LengthSq();
            const float t = length_sq > 0.0f
                ? fmaxf(0.0f, fminf(1.0f, (p - vertices[i]).Dot(edge) / length_sq)) : 0.0f;
            const Vec3 q = vertices[i] + edge * t;
            const float sq = (p - q).LengthSq();
            if (sq < best_sq) {
                best_sq = sq;
                best = {q, weights[i] * (1.0f - t) + weights[j] * t,
                        t == 0.0f ? i : t == 1.0f ? j : 3u + i};
            }
        }
        return best;
    }
    if (d1 <= 0.0f && d2 <= 0.0f) return {a, {1, 0, 0}, 0u};
    if (d3 >= 0.0f && d4 <= d3) return {b, {0, 1, 0}, 1u};
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        const float v = d1 / (d1 - d3);
        return {a + ab * v, {1.0f - v, v, 0}, 3u};
    }
    if (d6 >= 0.0f && d5 <= d6) return {c, {0, 0, 1}, 2u};
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        const float w = d2 / (d2 - d6);
        return {a + ac * w, {1.0f - w, 0, w}, 5u};
    }
    if (va <= 0.0f && d4 >= d3 && d5 >= d6) {
        const float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return {b + (c - b) * w, {0, 1.0f - w, w}, 4u};
    }
    const float inverse = 1.0f / (va + vb + vc);
    const float v = vb * inverse, w = vc * inverse;
    return {a + ab * v + ac * w, {1.0f - v - w, v, w}, 6u};
}

NUKA_MESH_HD inline bool MeshSurfaceRangeValid(
    const MeshSurfaceView& view, const MeshSurfaceInfo& info) {
    return view.vertices && view.triangles && view.nodes &&
        info.vertex_count > 0u && info.triangle_count > 0u && info.node_count > 0u &&
        info.vertex_offset <= view.counts.vertices &&
        info.vertex_count <= view.counts.vertices - info.vertex_offset &&
        info.triangle_offset <= view.counts.triangles &&
        info.triangle_count <= view.counts.triangles - info.triangle_offset &&
        info.node_offset <= view.counts.nodes &&
        info.node_count <= view.counts.nodes - info.node_offset;
}

NUKA_MESH_HD inline math::Vec3 MeshSurfaceVertex(
    const MeshSurfaceView& view, const MeshSurfaceInfo& info, uint32_t vertex) {
    const size_t at = (static_cast<size_t>(info.vertex_offset) + vertex) * 3u;
    return {view.vertices[at], view.vertices[at + 1u], view.vertices[at + 2u]};
}

NUKA_MESH_HD inline bool MeshSurfaceTriangle(
    const MeshSurfaceView& view, const MeshSurfaceInfo& info, uint32_t triangle,
    math::Vec3& a, math::Vec3& b, math::Vec3& c) {
    if (triangle >= info.triangle_count) return false;
    const size_t at = (static_cast<size_t>(info.triangle_offset) + triangle) * 3u;
    const uint32_t ia = view.triangles[at], ib = view.triangles[at + 1u],
                   ic = view.triangles[at + 2u];
    if (ia >= info.vertex_count || ib >= info.vertex_count || ic >= info.vertex_count)
        return false;
    a = MeshSurfaceVertex(view, info, ia);
    b = MeshSurfaceVertex(view, info, ib);
    c = MeshSurfaceVertex(view, info, ic);
    return true;
}

NUKA_MESH_HD inline float MeshBoundsDistanceSquared(
    math::Vec3 p, const MeshBvhNode& node) {
    const math::Vec3 delta{
        fmaxf(0.0f, fmaxf(node.lower.x - p.x, p.x - node.upper.x)),
        fmaxf(0.0f, fmaxf(node.lower.y - p.y, p.y - node.upper.y)),
        fmaxf(0.0f, fmaxf(node.lower.z - p.z, p.z - node.upper.z))};
    return delta.LengthSq();
}

// Half-open projected edges count a shared edge once, including ray/vertex ties.
NUKA_MESH_HD inline bool MeshRayEdgeOwned(math::Vec3 a, math::Vec3 b) {
    return b.z > a.z || (b.z == a.z && b.y < a.y);
}

NUKA_MESH_HD inline bool MeshPositiveXRayCrosses(
    math::Vec3 p, math::Vec3 a, math::Vec3 b, math::Vec3 c) {
    const double ay = double(a.y) - p.y, az = double(a.z) - p.z;
    const double by = double(b.y) - p.y, bz = double(b.z) - p.z;
    const double cy = double(c.y) - p.y, cz = double(c.z) - p.z;
    double u = by * cz - bz * cy, v = cy * az - cz * ay, w = ay * bz - az * by;
    double determinant = u + v + w;
    if (determinant == 0.0) return false;
    const bool reverse = determinant < 0.0;
    if (reverse) { u = -u; v = -v; w = -w; determinant = -determinant; }
    if (u < 0.0 || v < 0.0 || w < 0.0) return false;
    if (u == 0.0 && !(reverse ? MeshRayEdgeOwned(c, b) : MeshRayEdgeOwned(b, c)))
        return false;
    if (v == 0.0 && !(reverse ? MeshRayEdgeOwned(a, c) : MeshRayEdgeOwned(c, a)))
        return false;
    if (w == 0.0 && !(reverse ? MeshRayEdgeOwned(b, a) : MeshRayEdgeOwned(a, b)))
        return false;
    return u * (double(a.x) - p.x) + v * (double(b.x) - p.x) +
           w * (double(c.x) - p.x) > 0.0;
}

NUKA_MESH_HD inline bool MeshSurfaceContains(
    const MeshSurfaceView& view, const MeshSurfaceInfo& info, math::Vec3 p, bool& valid) {
    bool inside = false;
    uint32_t cursor = 0u;
    while (cursor < info.node_count) {
        const MeshBvhNode& node = view.nodes[info.node_offset + cursor];
        if (node.escape <= cursor || node.escape > info.node_count) {
            valid = false;
            return false;
        }
        if (node.upper.x < p.x || node.lower.y > p.y || node.upper.y < p.y ||
            node.lower.z > p.z || node.upper.z < p.z) {
            cursor = node.escape;
            continue;
        }
        if (node.triangle != ~0u) {
            math::Vec3 a, b, c;
            if (!MeshSurfaceTriangle(view, info, node.triangle, a, b, c)) {
                valid = false;
                return false;
            }
            if (MeshPositiveXRayCrosses(p, a, b, c)) inside = !inside;
        }
        ++cursor;
    }
    return inside;
}

NUKA_MESH_HD inline MeshSurfacePoint QueryMeshSurface(
    const MeshSurfaceView& view, const MeshSurfaceInfo& info, math::Vec3 p,
    float max_distance = FLT_MAX) {
    using math::Vec3;
    MeshSurfacePoint result;
    if (!MeshSurfaceRangeValid(view, info)) return result;
    const MeshBvhNode& root = view.nodes[info.node_offset];
    if (max_distance >= 0.0f &&
        MeshBoundsDistanceSquared(p, root) > max_distance * max_distance) {
        result.distance = FLT_MAX;
        result.valid = true;
        return result;
    }
    float best_sq = FLT_MAX;
    Vec3 face_normal{1, 0, 0};
    uint32_t cursor = 0u;
    while (cursor < info.node_count) {
        const MeshBvhNode& node = view.nodes[info.node_offset + cursor];
        if (node.escape <= cursor || node.escape > info.node_count) return {};
        if (MeshBoundsDistanceSquared(p, node) > best_sq) {
            cursor = node.escape;
            continue;
        }
        if (node.triangle != ~0u) {
            Vec3 a, b, c;
            if (!MeshSurfaceTriangle(view, info, node.triangle, a, b, c)) return {};
            const auto closest = ClosestTrianglePoint(p, a, b, c);
            const float sq = (p - closest.point).LengthSq();
            if (sq < best_sq || (sq == best_sq && node.triangle < result.triangle)) {
                best_sq = sq;
                result.point = closest.point;
                result.barycentric = closest.barycentric;
                result.triangle = node.triangle;
                result.feature = closest.feature;
                face_normal = (b - a).Cross(c - a);
            }
        }
        ++cursor;
    }
    if (result.triangle == ~0u) return result;
    result.valid = true;
    const bool closed = (info.flags & kMeshSurfaceClosed) != 0u;
    const float distance = sqrtf(best_sq);
    if (distance > 0.0f) {
        const bool inside = closed && MeshSurfaceContains(view, info, p, result.valid);
        result.distance = inside ? -distance : distance;
        result.normal = (p - result.point) * ((inside ? -1.0f : 1.0f) / distance);
    } else {
        const float length = sqrtf(face_normal.LengthSq());
        result.normal = length > 0.0f ? face_normal / length : Vec3{1, 0, 0};
        if (closed) {
            const Vec3 n = result.normal;
            const Vec3 probe{
                n.x == 0.0f ? p.x : nextafterf(p.x, n.x > 0.0f ? INFINITY : -INFINITY),
                n.y == 0.0f ? p.y : nextafterf(p.y, n.y > 0.0f ? INFINITY : -INFINITY),
                n.z == 0.0f ? p.z : nextafterf(p.z, n.z > 0.0f ? INFINITY : -INFINITY)};
            if (MeshSurfaceContains(view, info, probe, result.valid))
                result.normal = result.normal * -1.0f;
        }
    }
    return result;
}

}  // namespace nuka::collision

#undef NUKA_MESH_HD
