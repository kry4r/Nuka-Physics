#pragma once

#include "collision/mesh_surface.hpp"

#if defined(__CUDACC__)
#define NUKA_OGC_HD __host__ __device__
#else
#define NUKA_OGC_HD
#endif

namespace nuka::collision {

struct OgcSegmentPair {
    math::Vec3 a{};
    math::Vec3 b{};
    float weight_a = 0.0f;
    float weight_b = 0.0f;
};

struct OgcTriangleFeature {
    math::Vec3 point{};
    math::Vec3 barycentric{};
    math::Vec3 normal{};
    float distance = 0.0f;
    uint32_t kind = ~0u;
    uint32_t index = ~0u;
    uint32_t owner_triangle = ~0u;
    bool feasible = false;
};

struct OgcEdgePair {
    math::Vec3 point_a{};
    math::Vec3 point_b{};
    math::Vec3 normal{};
    float weight_a = 0.0f;
    float weight_b = 0.0f;
    float distance = 0.0f;
    uint32_t owner_edge_a = ~0u;
    uint32_t owner_edge_b = ~0u;
    bool feasible = false;
};

NUKA_OGC_HD inline bool OgcEdgeInterior(math::Vec3 query, math::Vec3 a,
                                          math::Vec3 b) {
    const math::Vec3 edge = b - a;
    const float length_sq = edge.LengthSq();
    if (!(length_sq > 0.0f)) return false;
    const float projection = (query - a).Dot(edge);
    return projection > 0.0f && projection < length_sq;
}

NUKA_OGC_HD inline bool OgcOutsideAdjacentFace(math::Vec3 query,
                                                 math::Vec3 a, math::Vec3 edge,
                                                 float inverse_length_sq,
                                                 math::Vec3 opposite) {
    const math::Vec3 foot = a + edge * ((opposite - a).Dot(edge) * inverse_length_sq);
    return (query - foot).Dot(foot - opposite) >= 0.0f;
}

NUKA_OGC_HD inline bool OgcEdgeFeasible(math::Vec3 query, math::Vec3 a,
                                          math::Vec3 b, math::Vec3 opposite0,
                                          math::Vec3 opposite1, bool has_opposite1) {
    if (!OgcEdgeInterior(query, a, b)) return false;
    const math::Vec3 edge = b - a;
    const float inverse_length_sq = 1.0f / edge.LengthSq();
    return OgcOutsideAdjacentFace(query, a, edge, inverse_length_sq, opposite0) &&
           (!has_opposite1 || OgcOutsideAdjacentFace(query, a, edge,
                                                      inverse_length_sq, opposite1));
}

NUKA_OGC_HD inline bool OgcWireVertexFeasible(math::Vec3 query,
                                                math::Vec3 vertex,
                                                math::Vec3 neighbor0,
                                                math::Vec3 neighbor1) {
    const math::Vec3 offset = query - vertex;
    return offset.Dot(vertex - neighbor0) >= 0.0f &&
           offset.Dot(vertex - neighbor1) >= 0.0f;
}

NUKA_OGC_HD inline bool OgcWireVertexFeasible(
    const MeshSurfaceView& view, const MeshSurfaceInfo& surface,
    const MeshEdge* edges, const MeshBvhNode* nodes, const MeshEdgeInfo& info,
    uint32_t vertex, math::Vec3 query, bool weld_by_position,
    uint32_t* owner_edge = nullptr) {
    if (!edges || !nodes || vertex >= surface.vertex_count ||
        info.node_count == 0u) return false;
    const math::Vec3 center = MeshSurfaceVertex(view, surface, vertex);
    const math::Vec3 offset = query - center;
    bool found = false;
    uint32_t cursor = 0u;
    while (cursor < info.node_count) {
        const MeshBvhNode& node = nodes[info.node_offset + cursor];
        if (node.escape <= cursor || node.escape > info.node_count) return false;
        if (MeshBoundsDistanceSquared(MeshSurfaceLocalPoint(view, center), node) > 0.0f) {
            cursor = node.escape;
            continue;
        }
        if (node.triangle != ~0u) {
            if (node.triangle >= info.edge_count) return false;
            const MeshEdge& edge = edges[info.edge_offset + node.triangle];
            const uint32_t ids[2] = {edge.vertex0, edge.vertex1};
            if (ids[0] >= surface.vertex_count || ids[1] >= surface.vertex_count)
                return false;
            for (uint32_t side = 0u; side < 2u; ++side) {
                const math::Vec3 endpoint = MeshSurfaceVertex(view, surface, ids[side]);
                const bool same = weld_by_position
                    ? endpoint.x == center.x && endpoint.y == center.y &&
                      endpoint.z == center.z : ids[side] == vertex;
                if (!same) continue;
                found = true;
                if (owner_edge && node.triangle < *owner_edge)
                    *owner_edge = node.triangle;
                const math::Vec3 neighbor = MeshSurfaceVertex(view, surface, ids[1u - side]);
                if (offset.Dot(center - neighbor) < 0.0f) return false;
            }
        }
        ++cursor;
    }
    return found;
}

NUKA_OGC_HD inline bool OgcSurfaceVertexFeasible(
    const MeshSurfaceView& view, const MeshSurfaceInfo& info,
    uint32_t vertex, math::Vec3 query, bool weld_by_position,
    uint32_t* owner_triangle = nullptr) {
    if (!MeshSurfaceRangeValid(view, info) || vertex >= info.vertex_count) return false;
    const math::Vec3 center = MeshSurfaceVertex(view, info, vertex);
    const math::Vec3 offset = query - center;
    bool found = false;
    uint32_t cursor = 0u;
    while (cursor < info.node_count) {
        const MeshBvhNode& node = view.nodes[info.node_offset + cursor];
        if (node.escape <= cursor || node.escape > info.node_count) return false;
        if (MeshBoundsDistanceSquared(MeshSurfaceLocalPoint(view, center), node) > 0.0f) {
            cursor = node.escape;
            continue;
        }
        if (node.triangle != ~0u) {
            const size_t first = (static_cast<size_t>(info.triangle_offset) + node.triangle) * 3u;
            if (node.triangle >= info.triangle_count) return false;
            const uint32_t ids[3] = {view.triangles[first], view.triangles[first + 1u],
                                     view.triangles[first + 2u]};
            for (uint32_t i = 0u; i < 3u; ++i) {
                if (ids[i] >= info.vertex_count) return false;
                const math::Vec3 face_vertex = MeshSurfaceVertex(view, info, ids[i]);
                const bool same = weld_by_position
                    ? (face_vertex.x == center.x && face_vertex.y == center.y &&
                       face_vertex.z == center.z) : ids[i] == vertex;
                if (!same) continue;
                found = true;
                if (owner_triangle && node.triangle < *owner_triangle)
                    *owner_triangle = node.triangle;
                for (uint32_t step = 1u; step < 3u; ++step) {
                    const math::Vec3 neighbor = MeshSurfaceVertex(view, info, ids[(i + step) % 3u]);
                    if (offset.Dot(center - neighbor) < 0.0f) return false;
                }
            }
        }
        ++cursor;
    }
    return found;
}

NUKA_OGC_HD inline OgcTriangleFeature OgcFacetFeature(
    const MeshSurfaceView& view, const MeshSurfaceInfo& info,
    const MeshEdge* edges, const uint32_t* triangle_edges,
    const MeshEdgeInfo& edge_info,
    uint32_t triangle, math::Vec3 query, bool weld_by_position) {
    OgcTriangleFeature result;
    if (!MeshSurfaceRangeValid(view, info) || triangle >= info.triangle_count)
        return result;
    const size_t first = (static_cast<size_t>(info.triangle_offset) + triangle) * 3u;
    const uint32_t ids[3] = {view.triangles[first], view.triangles[first + 1u],
                             view.triangles[first + 2u]};
    for (uint32_t id : ids)
        if (id >= info.vertex_count) return result;
    const math::Vec3 p[3] = {MeshSurfaceVertex(view, info, ids[0]),
                             MeshSurfaceVertex(view, info, ids[1]),
                             MeshSurfaceVertex(view, info, ids[2])};
    const math::Vec3 face_normal = (p[1] - p[0]).Cross(p[2] - p[0]);
    if (!(face_normal.LengthSq() > 0.0f)) return result;
    const TriangleClosestPoint closest = ClosestTrianglePoint(query, p[0], p[1], p[2]);
    const math::Vec3 separation = query - closest.point;
    result.point = closest.point;
    result.barycentric = closest.barycentric;
    result.distance = sqrtf(separation.LengthSq());
    result.normal = result.distance > 0.0f
        ? separation / result.distance : face_normal / sqrtf(face_normal.LengthSq());
    if (closest.feature < 3u) {
        result.kind = 0u;
        result.index = ids[closest.feature];
        result.feasible = OgcSurfaceVertexFeasible(
            view, info, result.index, query, weld_by_position,
            &result.owner_triangle);
    } else if (closest.feature < 6u) {
        if (!edges || !triangle_edges) return result;
        const uint32_t side = closest.feature - 3u;
        const uint32_t a = ids[side], b = ids[(side + 1u) % 3u];
        const uint32_t edge_id = triangle_edges[first + side];
        if (edge_id >= edge_info.edge_count) return result;
        const MeshEdge& edge = edges[edge_info.edge_offset + edge_id];
        if (edge.vertex0 >= info.vertex_count || edge.vertex1 >= info.vertex_count ||
            edge.opposite0 >= info.vertex_count ||
            (edge.triangle1 != ~0u && edge.opposite1 >= info.vertex_count) ||
            (edge.triangle0 != triangle && edge.triangle1 != triangle)) return result;
        const math::Vec3 e0 = MeshSurfaceVertex(view, info, edge.vertex0);
        const math::Vec3 e1 = MeshSurfaceVertex(view, info, edge.vertex1);
        const bool forward = weld_by_position
            ? (e0.x == p[side].x && e0.y == p[side].y && e0.z == p[side].z &&
               e1.x == p[(side + 1u) % 3u].x && e1.y == p[(side + 1u) % 3u].y &&
               e1.z == p[(side + 1u) % 3u].z)
            : (edge.vertex0 == a && edge.vertex1 == b);
        const bool reverse = weld_by_position
            ? (e1.x == p[side].x && e1.y == p[side].y && e1.z == p[side].z &&
               e0.x == p[(side + 1u) % 3u].x && e0.y == p[(side + 1u) % 3u].y &&
               e0.z == p[(side + 1u) % 3u].z)
            : (edge.vertex1 == a && edge.vertex0 == b);
        if (!forward && !reverse) return result;
        result.kind = 1u;
        result.index = edge_id;
        result.owner_triangle = edge.triangle1 == ~0u
            ? edge.triangle0 : edge.triangle0 < edge.triangle1
                ? edge.triangle0 : edge.triangle1;
        result.feasible = OgcEdgeFeasible(query, e0, e1,
            MeshSurfaceVertex(view, info, edge.opposite0),
            edge.triangle1 != ~0u
                ? MeshSurfaceVertex(view, info, edge.opposite1) : math::Vec3{},
            edge.triangle1 != ~0u);
    } else {
        result.kind = 2u;
        result.index = triangle;
        result.owner_triangle = triangle;
        result.feasible = true;
    }
    return result;
}

NUKA_OGC_HD inline OgcSegmentPair OgcClosestSegments(
    math::Vec3 a0, math::Vec3 a1, math::Vec3 b0, math::Vec3 b1) {
    const math::Vec3 u = a1 - a0, v = b1 - b0, w = a0 - b0;
    const float a = u.Dot(u), b = u.Dot(v), c = v.Dot(v);
    const float d = u.Dot(w), e = v.Dot(w);
    float s = 0.0f, t = 0.0f;
    if (a <= 0.0f && c <= 0.0f) return {a0, b0, s, t};
    if (a <= 0.0f) {
        t = fmaxf(0.0f, fminf(1.0f, e / c));
    } else if (c <= 0.0f) {
        s = fmaxf(0.0f, fminf(1.0f, -d / a));
    } else {
        const float determinant = a * c - b * b;
        if (determinant > 0.0f)
            s = fmaxf(0.0f, fminf(1.0f, (b * e - c * d) / determinant));
        t = (b * s + e) / c;
        if (t < 0.0f) {
            t = 0.0f;
            s = fmaxf(0.0f, fminf(1.0f, -d / a));
        } else if (t > 1.0f) {
            t = 1.0f;
            s = fmaxf(0.0f, fminf(1.0f, (b - d) / a));
        }
    }
    return {a0 + u * s, b0 + v * t, s, t};
}

NUKA_OGC_HD inline OgcEdgePair OgcEdgeContact(
    const MeshSurfaceView& surface_a, const MeshSurfaceInfo& info_a,
    const MeshEdge* edges_a, const MeshBvhNode* nodes_a,
    const MeshEdgeInfo& edge_info_a, uint32_t edge_a,
    const MeshSurfaceView& surface_b, const MeshSurfaceInfo& info_b,
    const MeshEdge* edges_b, const MeshBvhNode* nodes_b,
    const MeshEdgeInfo& edge_info_b, uint32_t edge_b,
    bool weld_a, bool weld_b) {
    OgcEdgePair result;
    if (!edges_a || !edges_b || edge_a >= edge_info_a.edge_count ||
        edge_b >= edge_info_b.edge_count) return result;
    const MeshEdge& ea = edges_a[edge_info_a.edge_offset + edge_a];
    const MeshEdge& eb = edges_b[edge_info_b.edge_offset + edge_b];
    if (ea.vertex0 >= info_a.vertex_count || ea.vertex1 >= info_a.vertex_count ||
        eb.vertex0 >= info_b.vertex_count || eb.vertex1 >= info_b.vertex_count)
        return result;
    const math::Vec3 a0 = MeshSurfaceVertex(surface_a, info_a, ea.vertex0);
    const math::Vec3 a1 = MeshSurfaceVertex(surface_a, info_a, ea.vertex1);
    const math::Vec3 b0 = MeshSurfaceVertex(surface_b, info_b, eb.vertex0);
    const math::Vec3 b1 = MeshSurfaceVertex(surface_b, info_b, eb.vertex1);
    const OgcSegmentPair pair = OgcClosestSegments(a0, a1, b0, b1);
    result.point_a = pair.a;
    result.point_b = pair.b;
    result.weight_a = pair.weight_a;
    result.weight_b = pair.weight_b;
    const math::Vec3 separation = pair.a - pair.b;
    result.distance = sqrtf(separation.LengthSq());
    const math::Vec3 cross = (a1 - a0).Cross(b1 - b0);
    const float cross_length = sqrtf(cross.LengthSq());
    result.normal = result.distance > 0.0f
        ? separation / result.distance
        : cross_length > 0.0f ? cross / cross_length : math::Vec3{};
    result.owner_edge_a = edge_a;
    result.owner_edge_b = edge_b;
    const bool a_feasible = pair.weight_a > 0.0f && pair.weight_a < 1.0f
        ? true : OgcWireVertexFeasible(surface_a, info_a, edges_a, nodes_a,
            edge_info_a, pair.weight_a <= 0.0f ? ea.vertex0 : ea.vertex1,
            pair.b, weld_a, &result.owner_edge_a);
    const bool b_feasible = pair.weight_b > 0.0f && pair.weight_b < 1.0f
        ? true : OgcWireVertexFeasible(surface_b, info_b, edges_b, nodes_b,
            edge_info_b, pair.weight_b <= 0.0f ? eb.vertex0 : eb.vertex1,
            pair.a, weld_b, &result.owner_edge_b);
    result.feasible = a_feasible && b_feasible &&
        result.normal.LengthSq() > 0.0f;
    return result;
}

}  // namespace nuka::collision

#undef NUKA_OGC_HD
