#include "import/cooker/mesh_surface_cooker.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

namespace nuka::import::cooker {
namespace {

using math::Vec3;
using collision::MeshBvhNode;

Vec3 Min(Vec3 a, Vec3 b) {
    return {std::min(a.x, b.x), std::min(a.y, b.y), std::min(a.z, b.z)};
}

Vec3 Max(Vec3 a, Vec3 b) {
    return {std::max(a.x, b.x), std::max(a.y, b.y), std::max(a.z, b.z)};
}

uint32_t BuildNodes(std::vector<MeshBvhNode>& nodes, std::vector<uint32_t>& order,
    const std::vector<MeshBvhNode>& leaves, uint32_t begin, uint32_t end) {
    const uint32_t index = static_cast<uint32_t>(nodes.size());
    nodes.emplace_back();
    Vec3 lo{FLT_MAX, FLT_MAX, FLT_MAX}, hi{-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (uint32_t i = begin; i < end; ++i) {
        lo = Min(lo, leaves[order[i]].lower);
        hi = Max(hi, leaves[order[i]].upper);
    }
    nodes[index].lower = lo;
    nodes[index].upper = hi;
    if (end - begin == 1u) {
        nodes[index].triangle = order[begin];
    } else {
        const Vec3 extent = hi - lo;
        const uint32_t axis = extent.x >= extent.y && extent.x >= extent.z ? 0u
            : extent.y >= extent.z ? 1u : 2u;
        const auto center = [&](uint32_t triangle) {
            const Vec3 c = leaves[triangle].lower * 0.5f + leaves[triangle].upper * 0.5f;
            return axis == 0u ? c.x : axis == 1u ? c.y : c.z;
        };
        const uint32_t middle = begin + (end - begin) / 2u;
        std::nth_element(order.begin() + begin, order.begin() + middle, order.begin() + end,
            [&](uint32_t a, uint32_t b) {
                const float ca = center(a), cb = center(b);
                return ca < cb || (ca == cb && a < b);
            });
        BuildNodes(nodes, order, leaves, begin, middle);
        BuildNodes(nodes, order, leaves, middle, end);
    }
    nodes[index].escape = static_cast<uint32_t>(nodes.size());
    return index;
}

}  // namespace

void ValidateMeshSurfaceInput(const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t triangle_count) {
    if (!vertices || !indices || vertex_count == 0u || triangle_count == 0u ||
        uint64_t(triangle_count) * 2u - 1u > std::numeric_limits<uint32_t>::max())
        throw std::invalid_argument("Collision mesh has invalid geometry counts");
    for (size_t i = 0u; i < static_cast<size_t>(vertex_count) * 3u; ++i)
        if (!std::isfinite(vertices[i]))
            throw std::invalid_argument("Collision mesh contains a non-finite vertex");
    for (size_t i = 0u; i < static_cast<size_t>(triangle_count) * 3u; ++i)
        if (indices[i] >= vertex_count)
            throw std::invalid_argument("Collision mesh triangle has an invalid vertex index");
}

CookedMeshSurface CookMeshSurface(const float* vertices, uint32_t vertex_count,
    const uint32_t* indices, uint32_t triangle_count, bool require_convex) {
    ValidateMeshSurfaceInput(vertices, vertex_count, indices, triangle_count);
    const auto vertex = [&](uint32_t i) -> Vec3 {
        const size_t at = static_cast<size_t>(i) * 3u;
        return {vertices[at], vertices[at + 1u], vertices[at + 2u]};
    };
    std::map<std::array<float, 3>, uint32_t> unique_vertices;
    std::vector<uint32_t> welded(vertex_count);
    Vec3 centroid{};
    float scale = 0.0f;
    for (uint32_t i = 0u; i < vertex_count; ++i) {
        const Vec3 v = vertex(i);
        const auto key = std::array<float, 3>{v.x, v.y, v.z};
        welded[i] = unique_vertices.emplace(key, static_cast<uint32_t>(unique_vertices.size()))
            .first->second;
        centroid += v / static_cast<float>(vertex_count);
        scale = std::max(scale, std::max(std::fabs(v.x),
                         std::max(std::fabs(v.y), std::fabs(v.z))));
    }
    std::map<std::pair<uint32_t, uint32_t>, uint32_t> edges;
    std::vector<MeshBvhNode> leaves(triangle_count);
    bool closed = true;
    const float tolerance = 64.0f * std::numeric_limits<float>::epsilon() * scale;
    for (uint32_t i = 0u; i < triangle_count; ++i) {
        const size_t at = static_cast<size_t>(i) * 3u;
        const uint32_t ids[3] = {indices[at], indices[at + 1u], indices[at + 2u]};
        const Vec3 a = vertex(ids[0]), b = vertex(ids[1]), c = vertex(ids[2]);
        leaves[i].lower = Min(a, Min(b, c));
        leaves[i].upper = Max(a, Max(b, c));
        Vec3 normal = (b - a).Cross(c - a);
        const float length = std::sqrt(normal.LengthSq());
        if (!(length > 0.0f) || !std::isfinite(length)) closed = false;
        for (uint32_t j = 0u; j < 3u; ++j) {
            const uint32_t u = welded[ids[j]], v = welded[ids[(j + 1u) % 3u]];
            if (u == v) closed = false;
            ++edges[std::minmax(u, v)];
        }
        if (require_convex && length > 0.0f) {
            normal = normal / length;
            if (normal.Dot(centroid - a) > 0.0f) normal = normal * -1.0f;
            for (uint32_t j = 0u; j < vertex_count; ++j)
                if (normal.Dot(vertex(j) - a) > tolerance)
                    throw std::invalid_argument("ConvexHull requires convex faces; use TriMesh or decomposition");
        }
    }
    for (const auto& edge : edges) if (edge.second != 2u) closed = false;
    if (require_convex && !closed)
        throw std::invalid_argument("ConvexHull requires a closed, nondegenerate surface");
    CookedMeshSurface result;
    result.info.vertex_count = vertex_count;
    result.info.triangle_count = triangle_count;
    result.info.flags = closed ? collision::kMeshSurfaceClosed : 0u;
    if (require_convex) result.info.flags |= collision::kMeshSurfaceConvex;
    result.nodes.reserve(static_cast<size_t>(triangle_count) * 2u - 1u);
    std::vector<uint32_t> order(triangle_count);
    std::iota(order.begin(), order.end(), 0u);
    BuildNodes(result.nodes, order, leaves, 0u, triangle_count);
    result.info.node_count = static_cast<uint32_t>(result.nodes.size());
    return result;
}

}  // namespace nuka::import::cooker
