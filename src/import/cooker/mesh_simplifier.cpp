#include "import/cooker/mesh_simplifier.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "import/cooker/mesh_surface_cooker.hpp"
#include "math/vec3.hpp"

namespace nuka::import::cooker {
namespace {

using math::Vec3;

// Collapsed faces keep this altitude (m) so world-space float32 rounding cannot flatten them.
constexpr float kMinAltitude = 1.0e-5f;

struct Quadric {
    double a[4][4]{};

    void AddPlane(Vec3 n, double d, double weight) {
        const double p[4] = {n.x, n.y, n.z, d};
        for (uint32_t i = 0u; i < 4u; ++i)
            for (uint32_t j = 0u; j < 4u; ++j)
                a[i][j] += weight * p[i] * p[j];
    }

    Quadric& operator+=(const Quadric& other) {
        for (uint32_t i = 0u; i < 4u; ++i)
            for (uint32_t j = 0u; j < 4u; ++j)
                a[i][j] += other.a[i][j];
        return *this;
    }

    double Cost(Vec3 x) const {
        const double p[4] = {x.x, x.y, x.z, 1.0};
        double value = 0.0;
        for (uint32_t i = 0u; i < 4u; ++i)
            for (uint32_t j = 0u; j < 4u; ++j)
                value += p[i] * a[i][j] * p[j];
        return value;
    }
};

struct Vertex {
    Vec3 position{};
    Quadric quadric{};
    std::set<uint32_t> faces;
    std::set<uint32_t> neighbors;
    uint32_t version = 0u;
    bool boundary = false;
    bool alive = true;
};

struct Face {
    std::array<uint32_t, 3> vertices{};
    bool alive = true;
};

struct Candidate {
    uint32_t a = 0u;
    uint32_t b = 0u;
    uint32_t version_a = 0u;
    uint32_t version_b = 0u;
    Vec3 position{};
    double cost = 0.0;

    bool operator<(const Candidate& other) const {
        if (cost != other.cost) return cost > other.cost;
        if (a != other.a) return a > other.a;
        return b > other.b;
    }
};

using Edge = std::pair<uint32_t, uint32_t>;

Vec3 Position(const float* vertices, uint32_t index) {
    const size_t at = static_cast<size_t>(index) * 3u;
    return {vertices[at], vertices[at + 1u], vertices[at + 2u]};
}

bool SolveOptimal(const Quadric& q, Vec3* position) {
    double a[3][4];
    for (uint32_t row = 0u; row < 3u; ++row) {
        for (uint32_t column = 0u; column < 3u; ++column)
            a[row][column] = q.a[row][column];
        a[row][3] = -q.a[row][3];
    }
    for (uint32_t column = 0u; column < 3u; ++column) {
        uint32_t pivot = column;
        for (uint32_t row = column + 1u; row < 3u; ++row)
            if (std::fabs(a[row][column]) > std::fabs(a[pivot][column])) pivot = row;
        if (!(std::fabs(a[pivot][column]) >
              std::numeric_limits<double>::epsilon() * std::fabs(q.a[column][column])))
            return false;
        if (pivot != column)
            for (uint32_t j = column; j < 4u; ++j)
                std::swap(a[pivot][j], a[column][j]);
        const double inverse = 1.0 / a[column][column];
        for (uint32_t j = column; j < 4u; ++j) a[column][j] *= inverse;
        for (uint32_t row = 0u; row < 3u; ++row) {
            if (row == column) continue;
            const double scale = a[row][column];
            for (uint32_t j = column; j < 4u; ++j)
                a[row][j] -= scale * a[column][j];
        }
    }
    const Vec3 result{static_cast<float>(a[0][3]),
                      static_cast<float>(a[1][3]),
                      static_cast<float>(a[2][3])};
    if (!std::isfinite(result.x) || !std::isfinite(result.y) ||
        !std::isfinite(result.z)) return false;
    *position = result;
    return true;
}

Candidate MakeCandidate(const std::vector<Vertex>& vertices, uint32_t a, uint32_t b) {
    if (a > b) std::swap(a, b);
    Candidate result;
    result.a = a;
    result.b = b;
    result.version_a = vertices[a].version;
    result.version_b = vertices[b].version;
    Quadric q = vertices[a].quadric;
    q += vertices[b].quadric;
    const Vec3 p0 = vertices[a].position, p1 = vertices[b].position;
    const Vec3 midpoint = (p0 + p1) * 0.5f;
    result.position = p0;
    result.cost = q.Cost(p0);
    for (Vec3 candidate : {p1, midpoint}) {
        const double cost = q.Cost(candidate);
        if (cost < result.cost) {
            result.position = candidate;
            result.cost = cost;
        }
    }
    Vec3 optimum{};
    if (!vertices[a].boundary && !vertices[b].boundary && SolveOptimal(q, &optimum)) {
        const double cost = q.Cost(optimum);
        if (cost < result.cost) {
            result.position = optimum;
            result.cost = cost;
        }
    }
    return result;
}

void RebuildNeighbors(std::vector<Vertex>& vertices,
                      const std::vector<Face>& faces, uint32_t index) {
    auto& vertex = vertices[index];
    vertex.neighbors.clear();
    vertex.boundary = false;
    if (!vertex.alive) {
        ++vertex.version;
        return;
    }
    for (uint32_t face_id : vertex.faces) {
        const Face& face = faces[face_id];
        if (!face.alive) continue;
        for (uint32_t other : face.vertices)
            if (other != index && vertices[other].alive)
                vertex.neighbors.insert(other);
    }
    for (uint32_t neighbor : vertex.neighbors) {
        uint32_t incident = 0u;
        for (uint32_t face_id : vertex.faces)
            if (faces[face_id].alive && vertices[neighbor].faces.contains(face_id))
                ++incident;
        if (incident != 2u) vertex.boundary = true;
    }
    ++vertex.version;
}

bool ValidCollapse(const std::vector<Vertex>& vertices,
                   const std::vector<Face>& faces, const Candidate& candidate,
                   std::set<uint32_t>* removed) {
    const uint32_t a = candidate.a, b = candidate.b;
    if (!vertices[a].alive || !vertices[b].alive ||
        vertices[a].boundary != vertices[b].boundary ||
        !vertices[a].neighbors.contains(b)) return false;
    removed->clear();
    std::set<uint32_t> opposite;
    for (uint32_t face_id : vertices[a].faces) {
        const Face& face = faces[face_id];
        if (!face.alive || !vertices[b].faces.contains(face_id)) continue;
        removed->insert(face_id);
        for (uint32_t vertex : face.vertices)
            if (vertex != a && vertex != b) opposite.insert(vertex);
    }
    if (removed->size() == 1u) {
        if (!vertices[a].boundary || !vertices[b].boundary) return false;
    } else if (removed->size() == 2u) {
        if (vertices[a].boundary || vertices[b].boundary) return false;
    } else {
        return false;
    }
    if (opposite.size() != removed->size()) return false;
    std::set<uint32_t> common;
    std::set_intersection(vertices[a].neighbors.begin(), vertices[a].neighbors.end(),
                          vertices[b].neighbors.begin(), vertices[b].neighbors.end(),
                          std::inserter(common, common.end()));
    if (common != opposite) return false;
    for (uint32_t vertex : {a, b})
        for (uint32_t face_id : vertices[vertex].faces) {
            const Face& face = faces[face_id];
            if (!face.alive || removed->contains(face_id)) continue;
            const Vec3 old[3] = {vertices[face.vertices[0]].position,
                                 vertices[face.vertices[1]].position,
                                 vertices[face.vertices[2]].position};
            Vec3 next[3] = {old[0], old[1], old[2]};
            for (uint32_t i = 0u; i < 3u; ++i)
                if (face.vertices[i] == a || face.vertices[i] == b)
                    next[i] = candidate.position;
            const Vec3 old_normal = (old[1] - old[0]).Cross(old[2] - old[0]);
            const Vec3 next_normal = (next[1] - next[0]).Cross(next[2] - next[0]);
            float longest = 0.0f;
            for (uint32_t i = 0u; i < 3u; ++i)
                longest = std::max(longest, (next[(i + 1u) % 3u] - next[i]).LengthSq());
            if (!(old_normal.Dot(next_normal) > 0.0f) ||
                !(next_normal.LengthSq() > kMinAltitude * kMinAltitude * longest)) return false;
        }
    return true;
}

bool FeasibleOnEdge(const std::vector<Vertex>& vertices,
                    const std::vector<Face>& faces, Candidate* candidate,
                    std::set<uint32_t>* removed) {
    const uint32_t a = candidate->a, b = candidate->b;
    const Vec3 start = vertices[a].position;
    const Vec3 direction = vertices[b].position - start;
    float lower = 0.0f, upper = 1.0f;
    for (uint32_t vertex : {a, b})
        for (uint32_t face_id : vertices[vertex].faces) {
            const Face& face = faces[face_id];
            if (!face.alive ||
                (vertices[a].faces.contains(face_id) &&
                 vertices[b].faces.contains(face_id))) continue;
            const Vec3 old[3] = {vertices[face.vertices[0]].position,
                                 vertices[face.vertices[1]].position,
                                 vertices[face.vertices[2]].position};
            const Vec3 normal = (old[1] - old[0]).Cross(old[2] - old[0]);
            float values[2]{};
            for (uint32_t endpoint = 0u; endpoint < 2u; ++endpoint) {
                Vec3 next[3] = {old[0], old[1], old[2]};
                for (uint32_t i = 0u; i < 3u; ++i)
                    if (face.vertices[i] == a || face.vertices[i] == b)
                        next[i] = start + direction * static_cast<float>(endpoint);
                values[endpoint] = normal.Dot((next[1] - next[0]).Cross(next[2] - next[0]));
            }
            if (values[0] <= 0.0f && values[1] <= 0.0f) return false;
            if (values[0] <= 0.0f)
                lower = std::max(lower, -values[0] / (values[1] - values[0]));
            if (values[1] <= 0.0f)
                upper = std::min(upper, values[0] / (values[0] - values[1]));
        }
    if (!(lower < upper)) return false;
    Quadric quadric = vertices[a].quadric;
    quadric += vertices[b].quadric;
    const double p[4] = {start.x, start.y, start.z, 1.0};
    const double d[4] = {direction.x, direction.y, direction.z, 0.0};
    double quadratic = 0.0, linear = 0.0;
    for (uint32_t i = 0u; i < 4u; ++i)
        for (uint32_t j = 0u; j < 4u; ++j) {
            quadratic += d[i] * quadric.a[i][j] * d[j];
            linear += 2.0 * p[i] * quadric.a[i][j] * d[j];
        }
    const float optimum = quadratic > 0.0
        ? static_cast<float>(-linear / (2.0 * quadratic)) : lower;
    const float first = std::nextafter(lower, upper);
    const float last = std::nextafter(upper, lower);
    if (first > last) return false;
    bool found = false;
    for (float t : {first, last, std::clamp(optimum, first, last),
                    (first + last) * 0.5f}) {
        Candidate trial = *candidate;
        trial.position = start + direction * t;
        trial.cost = quadric.Cost(trial.position);
        std::set<uint32_t> trial_removed;
        if (!ValidCollapse(vertices, faces, trial, &trial_removed)) continue;
        if (!found || trial.cost < candidate->cost) {
            *candidate = trial;
            *removed = std::move(trial_removed);
            found = true;
        }
    }
    return found;
}

void Collapse(std::vector<Vertex>& vertices, std::vector<Face>& faces,
              const Candidate& candidate, const std::set<uint32_t>& removed,
              std::priority_queue<Candidate>* queue) {
    const uint32_t a = candidate.a, b = candidate.b;
    std::set<uint32_t> affected = vertices[a].neighbors;
    affected.insert(vertices[b].neighbors.begin(), vertices[b].neighbors.end());
    affected.insert(a);
    affected.insert(b);
    for (uint32_t face_id : removed) {
        Face& face = faces[face_id];
        face.alive = false;
        for (uint32_t vertex : face.vertices) vertices[vertex].faces.erase(face_id);
    }
    const std::vector<uint32_t> moved(vertices[b].faces.begin(), vertices[b].faces.end());
    for (uint32_t face_id : moved) {
        Face& face = faces[face_id];
        for (uint32_t& vertex : face.vertices)
            if (vertex == b) vertex = a;
        vertices[a].faces.insert(face_id);
    }
    vertices[b].faces.clear();
    vertices[b].alive = false;
    vertices[a].position = candidate.position;
    vertices[a].quadric += vertices[b].quadric;
    for (uint32_t vertex : affected)
        RebuildNeighbors(vertices, faces, vertex);
    std::set<Edge> refreshed;
    for (uint32_t vertex : affected) {
        if (!vertices[vertex].alive) continue;
        for (uint32_t neighbor : vertices[vertex].neighbors)
            if (vertices[neighbor].alive)
                refreshed.insert(std::minmax(vertex, neighbor));
    }
    for (const auto& edge : refreshed)
        queue->push(MakeCandidate(vertices, edge.first, edge.second));
}

}  // namespace

SimplifiedMesh SimplifyMeshQem(const float* positions, uint32_t vertex_count,
                               const uint32_t* indices, uint32_t triangle_count,
                               uint32_t triangle_limit) {
    ValidateMeshSurfaceInput(positions, vertex_count, indices, triangle_count);
    if (triangle_limit == 0u)
        throw std::invalid_argument("Collision mesh triangle limit must be positive");
    const auto oriented = OrientMeshWinding(positions, vertex_count, indices, triangle_count);
    if (triangle_count <= triangle_limit) {
        SimplifiedMesh unchanged;
        unchanged.vertices.assign(positions, positions + static_cast<size_t>(vertex_count) * 3u);
        unchanged.indices = oriented;
        return unchanged;
    }
    std::map<std::array<float, 3>, uint32_t> welded;
    std::vector<uint32_t> remap(vertex_count);
    std::vector<Vertex> vertices;
    for (uint32_t index = 0u; index < vertex_count; ++index) {
        const Vec3 point = Position(positions, index);
        const auto key = std::array<float, 3>{point.x, point.y, point.z};
        const auto [it, inserted] = welded.emplace(key, static_cast<uint32_t>(vertices.size()));
        if (inserted) vertices.push_back(Vertex{point});
        remap[index] = it->second;
    }
    std::vector<Face> faces;
    faces.reserve(triangle_count);
    std::map<Edge, uint32_t> edge_counts;
    for (uint32_t triangle = 0u; triangle < triangle_count; ++triangle) {
        const size_t base = static_cast<size_t>(triangle) * 3u;
        const std::array<uint32_t, 3> ids = {remap[oriented[base]],
                                             remap[oriented[base + 1u]],
                                             remap[oriented[base + 2u]]};
        if (ids[0] == ids[1] || ids[1] == ids[2] || ids[2] == ids[0])
            throw std::invalid_argument("Collision mesh contains a degenerate triangle");
        const Vec3 a = vertices[ids[0]].position, b = vertices[ids[1]].position,
                   c = vertices[ids[2]].position;
        const Vec3 normal = (b - a).Cross(c - a);
        const float twice_area = std::sqrt(normal.LengthSq());
        if (!(twice_area > 0.0f))
            throw std::invalid_argument("Collision mesh contains a zero-area triangle");
        const Vec3 unit = normal / twice_area;
        const double distance = -unit.Dot(a);
        for (uint32_t vertex : ids) {
            vertices[vertex].faces.insert(triangle);
            vertices[vertex].quadric.AddPlane(unit, distance, twice_area);
        }
        for (uint32_t i = 0u; i < 3u; ++i)
            ++edge_counts[std::minmax(ids[i], ids[(i + 1u) % 3u])];
        faces.push_back({ids});
    }
    for (const auto& [edge, count] : edge_counts) {
        vertices[edge.first].neighbors.insert(edge.second);
        vertices[edge.second].neighbors.insert(edge.first);
        if (count == 1u) {
            vertices[edge.first].boundary = true;
            vertices[edge.second].boundary = true;
            const Vec3 a = vertices[edge.first].position;
            const Vec3 direction = vertices[edge.second].position - a;
            for (uint32_t face_id : vertices[edge.first].faces) {
                const Face& face = faces[face_id];
                if (!vertices[edge.second].faces.contains(face_id)) continue;
                const Vec3 p0 = vertices[face.vertices[0]].position;
                const Vec3 p1 = vertices[face.vertices[1]].position;
                const Vec3 p2 = vertices[face.vertices[2]].position;
                const Vec3 face_normal = (p1 - p0).Cross(p2 - p0);
                const Vec3 boundary_normal = direction.Cross(face_normal);
                const float length = std::sqrt(boundary_normal.LengthSq());
                if (length > 0.0f) {
                    const Vec3 unit = boundary_normal / length;
                    const double weight = direction.LengthSq();
                    const double distance = -unit.Dot(a);
                    vertices[edge.first].quadric.AddPlane(unit, distance, weight);
                    vertices[edge.second].quadric.AddPlane(unit, distance, weight);
                }
                break;
            }
        }
    }
    std::priority_queue<Candidate> queue;
    for (const auto& [edge, count] : edge_counts)
        queue.push(MakeCandidate(vertices, edge.first, edge.second));
    uint32_t remaining = triangle_count;
    while (remaining > triangle_limit && !queue.empty()) {
        Candidate candidate = queue.top();
        queue.pop();
        if (candidate.version_a != vertices[candidate.a].version ||
            candidate.version_b != vertices[candidate.b].version) continue;
        std::set<uint32_t> removed;
        if (!ValidCollapse(vertices, faces, candidate, &removed) &&
            !FeasibleOnEdge(vertices, faces, &candidate, &removed)) continue;
        remaining -= static_cast<uint32_t>(removed.size());
        Collapse(vertices, faces, candidate, removed, &queue);
    }
    SimplifiedMesh result;
    std::vector<uint32_t> compact(vertices.size(), ~0u);
    for (const Face& face : faces) {
        if (!face.alive) continue;
        for (uint32_t vertex : face.vertices) {
            if (compact[vertex] == ~0u) {
                compact[vertex] = static_cast<uint32_t>(result.vertices.size() / 3u);
                const Vec3 p = vertices[vertex].position;
                result.vertices.insert(result.vertices.end(), {p.x, p.y, p.z});
            }
            result.indices.push_back(compact[vertex]);
        }
    }
    return result;
}

}  // namespace nuka::import::cooker
