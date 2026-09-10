#include "import/cooker/mesh_surface_cooker.hpp"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

#include "collision/mesh_surface.hpp"

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

void AppendTree(std::vector<MeshBvhNode>& destination, const std::vector<MeshBvhNode>& source) {
    const auto offset = static_cast<uint32_t>(destination.size());
    for (auto node : source) { node.escape += offset; destination.push_back(node); }
}

void BuildGroups(std::vector<MeshBvhNode>& nodes, std::vector<uint32_t>& order,
    const std::vector<std::vector<MeshBvhNode>>& groups, uint32_t begin, uint32_t end) {
    if (end - begin == 1u) { AppendTree(nodes, groups[order[begin]]); return; }
    const auto at = static_cast<uint32_t>(nodes.size());
    nodes.emplace_back();
    Vec3 lower{FLT_MAX, FLT_MAX, FLT_MAX}, upper{-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (uint32_t i = begin; i < end; ++i) {
        lower = Min(lower, groups[order[i]][0].lower);
        upper = Max(upper, groups[order[i]][0].upper);
    }
    const Vec3 extent = upper - lower;
    const uint32_t axis = extent.x >= extent.y && extent.x >= extent.z ? 0u
        : extent.y >= extent.z ? 1u : 2u;
    const auto center = [&](uint32_t group) {
        const auto& root = groups[group][0];
        const Vec3 value = root.lower * 0.5f + root.upper * 0.5f;
        return axis == 0u ? value.x : axis == 1u ? value.y : value.z;
    };
    const uint32_t middle = begin + (end - begin) / 2u;
    std::nth_element(order.begin() + begin, order.begin() + middle, order.begin() + end,
        [&](uint32_t a, uint32_t b) {
            const float ca = center(a), cb = center(b);
            return ca < cb || (ca == cb && a < b);
        });
    BuildGroups(nodes, order, groups, begin, middle);
    BuildGroups(nodes, order, groups, middle, end);
    nodes[at].lower = lower;
    nodes[at].upper = upper;
    nodes[at].escape = static_cast<uint32_t>(nodes.size());
}

double TreeArea(const std::vector<MeshBvhNode>& nodes) {
    double total = 0.0;
    for (const auto& node : nodes) {
        const auto extent = node.upper - node.lower;
        total += double(extent.x) * extent.y + double(extent.x) * extent.z + double(extent.y) * extent.z;
    }
    return total;
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

bool MeshSurfaceTreeValid(collision::MeshSurfaceView source, collision::MeshSurfaceInfo info) {
    if (!collision::MeshSurfaceRangeValid(source, info) ||
        uint64_t(info.node_count) != uint64_t(info.triangle_count) * 2u - 1u ||
        source.nodes[info.node_offset].escape != info.node_count) return false;
    std::vector<uint8_t> seen(info.triangle_count, 0u);
    uint32_t leaves = 0u;
    const auto contains = [](const MeshBvhNode& node, Vec3 p) {
        return p.x >= node.lower.x && p.x <= node.upper.x && p.y >= node.lower.y &&
               p.y <= node.upper.y && p.z >= node.lower.z && p.z <= node.upper.z;
    };
    for (uint32_t n = 0u; n < info.node_count; ++n) {
        const auto& node = source.nodes[info.node_offset + n];
        if (node.escape <= n || node.escape > info.node_count ||
            !std::isfinite(node.lower.x) || !std::isfinite(node.lower.y) ||
            !std::isfinite(node.lower.z) || !std::isfinite(node.upper.x) ||
            !std::isfinite(node.upper.y) || !std::isfinite(node.upper.z) ||
            !contains(node, node.lower) || !contains(node, node.upper)) return false;
        if (node.triangle == ~0u) {
            if (n + 1u >= info.node_count) return false;
            const auto& left = source.nodes[info.node_offset + n + 1u];
            if (left.escape <= n + 1u || left.escape >= node.escape) return false;
            const auto& right = source.nodes[info.node_offset + left.escape];
            if (right.escape != node.escape || !contains(node, left.lower) ||
                !contains(node, left.upper) || !contains(node, right.lower) ||
                !contains(node, right.upper)) return false;
        } else {
            Vec3 a, b, c;
            if (node.escape != n + 1u ||
                !collision::MeshSurfaceTriangle(source, info, node.triangle, a, b, c) ||
                seen[node.triangle] || !contains(node, a) || !contains(node, b) || !contains(node, c))
                return false;
            seen[node.triangle] = 1u;
            ++leaves;
        }
    }
    return leaves == info.triangle_count;
}

void GroupMeshSurfaceByCover(CookedMeshSurface& surface, const float* vertices,
    const uint32_t* indices) {
    if (surface.cover.status != ConvexCoverStatus::Complete || surface.cover.parts.size() < 2u) return;
    const auto& info = surface.info;
    const collision::MeshSurfaceView source{vertices, indices, surface.nodes.data(),
        {info.vertex_count, info.triangle_count, info.node_count}};
    std::vector<MeshBvhNode> leaves(info.triangle_count);
    std::vector<std::vector<uint32_t>> assigned(surface.cover.parts.size());
    for (uint32_t triangle = 0u; triangle < info.triangle_count; ++triangle) {
        Vec3 a, b, c;
        if (!collision::MeshSurfaceTriangle(source, info, triangle, a, b, c))
            throw std::invalid_argument("Convex cover grouping has an invalid source triangle");
        leaves[triangle].lower = Min(a, Min(b, c));
        leaves[triangle].upper = Max(a, Max(b, c));
        const Vec3 center = a / 3.0f + b / 3.0f + c / 3.0f;
        double nearest = DBL_MAX;
        size_t owner = 0u;
        for (size_t part = 0u; part < surface.cover.parts.size(); ++part) {
            double distance = -DBL_MAX;
            for (const auto& plane : surface.cover.parts[part].planes)
                distance = std::max(distance, plane.normal.x * center.x + plane.normal.y * center.y +
                    plane.normal.z * center.z - plane.offset);
            if (distance < nearest) { nearest = distance; owner = part; }
        }
        assigned[owner].push_back(triangle);
    }
    std::vector<std::vector<MeshBvhNode>> groups;
    for (auto& triangles : assigned) {
        if (triangles.empty()) continue;
        groups.emplace_back();
        BuildNodes(groups.back(), triangles, leaves, 0u, static_cast<uint32_t>(triangles.size()));
    }
    std::vector<uint32_t> order(groups.size());
    std::iota(order.begin(), order.end(), 0u);
    std::vector<MeshBvhNode> nodes;
    nodes.reserve(surface.nodes.size());
    BuildGroups(nodes, order, groups, 0u, static_cast<uint32_t>(groups.size()));
    const collision::MeshSurfaceView grouped{vertices, indices, nodes.data(), source.counts};
    if (!MeshSurfaceTreeValid(grouped, info))
        throw std::runtime_error("Convex cover grouping failed source surface coverage");
    if (TreeArea(nodes) < TreeArea(surface.nodes)) {
        surface.nodes = std::move(nodes);
        surface.cover_hierarchy = true;
    }
}

}  // namespace nuka::import::cooker
