#include "nk/solve/vertex_block_hierarchy.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "nk/model/model.hpp"

namespace nuka::nk {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
// Nodes are added until every vertex lies within this fraction of the level spacing.
constexpr double kCover = 0.577;
// Node distance fields reach this many spacings, past any coarse triangle a vertex may use.
constexpr double kFieldReach = 3.0;
// Coarse triangles below this shape quality (1 when equilateral) do not interpolate.
constexpr double kSliverQuality = 0.2;
// Smallest barycentric weight a vertex accepts before it interpolates along a coarse edge.
constexpr double kInsideTolerance = -0.1;
// Boundary vertices turning by more than this on the rest surface are nodes on every level.
constexpr double kCornerTurn = 0.52359877559829887;
constexpr double kPi = 3.14159265358979323846;
constexpr uint32_t kMaxLevels = 15u;
constexpr uint32_t kNone = ~0u;

using Entry = std::pair<double, uint32_t>;
using MinQueue = std::priority_queue<Entry, std::vector<Entry>, std::greater<Entry>>;
using Field = std::vector<std::pair<uint32_t, double>>;

uint32_t Count(size_t size) { return static_cast<uint32_t>(size); }

// Rest metric of the vertex graph: triangle and spring edges with their rest lengths.
struct RestMesh {
    uint32_t vertices = 0u;
    std::vector<uint32_t> edge_offsets, edge_to;
    std::vector<double> edge_length;
    std::vector<std::array<uint32_t, 3>> triangles;
    std::vector<uint32_t> triangle_offsets, vertex_triangles;

    bool Active(uint32_t v) const { return edge_offsets[v + 1u] > edge_offsets[v]; }
    double Length(uint32_t a, uint32_t b) const {
        const auto first = edge_to.begin() + edge_offsets[a];
        const auto last = edge_to.begin() + edge_offsets[a + 1u];
        const auto it = std::lower_bound(first, last, b);
        return it != last && *it == b ? edge_length[static_cast<size_t>(it - edge_to.begin())] : kInf;
    }
};

RestMesh BuildRestMesh(const Model& model) {
    const auto& p = model.particles;
    const uint32_t n = model.capacities.vbd_vertices_per_env;
    const uint32_t begin = model.capacities.vbd_particle_begin;
    RestMesh mesh;
    mesh.vertices = n;
    // (a, b, length) with a < b; the first element defining an edge sets its length.
    std::vector<std::tuple<uint32_t, uint32_t, double>> defined;
    auto define = [&](uint32_t a, uint32_t b, double length) {
        if (a != b && std::isfinite(length) && length >= 0.0)
            defined.emplace_back(std::min(a, b), std::max(a, b), length);
    };
    for (const VbdElement& e : p.vbd_elements) {
        if (e.kind == kVbdTriangle) {
            // The rest edges are the columns of the inverse of the stored inverse edge matrix.
            const double a = e.rest[0], b = e.rest[1], c = e.rest[2], d = e.rest[3];
            const double det = std::abs(a * d - b * c);
            if (!(det > 0.0) || !std::isfinite(det)) continue;
            define(e.vertex[0], e.vertex[1], std::hypot(d, c) / det);
            define(e.vertex[0], e.vertex[2], std::hypot(b, a) / det);
            define(e.vertex[1], e.vertex[2], std::hypot(b + d, a + c) / det);
            mesh.triangles.push_back({e.vertex[0], e.vertex[1], e.vertex[2]});
        } else if (e.kind == kVbdSpring) {
            define(e.vertex[0], e.vertex[1], e.rest[0]);
        }
    }
    // Rod segments that no spring defines take their cooked length.
    for (const VbdElement& e : p.vbd_elements)
        if (e.kind == kVbdRodBend)
            for (uint32_t j = 0u; j < 2u; ++j) {
                const math::Vec3 x0 = p.initial_pos[begin + e.vertex[j]];
                const math::Vec3 x1 = p.initial_pos[begin + e.vertex[j + 1u]];
                const double dx = double(x1.x) - x0.x, dy = double(x1.y) - x0.y, dz = double(x1.z) - x0.z;
                define(e.vertex[j], e.vertex[j + 1u], std::sqrt(dx * dx + dy * dy + dz * dz));
            }
    std::stable_sort(defined.begin(), defined.end(), [](const auto& x, const auto& y) {
        return std::get<0>(x) != std::get<0>(y) ? std::get<0>(x) < std::get<0>(y)
                                                : std::get<1>(x) < std::get<1>(y);
    });
    std::vector<std::tuple<uint32_t, uint32_t, double>> edges;
    std::vector<uint32_t> degree(n, 0u);
    for (const auto& edge : defined)
        if (edges.empty() || std::get<0>(edges.back()) != std::get<0>(edge) ||
            std::get<1>(edges.back()) != std::get<1>(edge)) {
            edges.push_back(edge);
            ++degree[std::get<0>(edge)];
            ++degree[std::get<1>(edge)];
        }
    mesh.edge_offsets.assign(n + 1u, 0u);
    for (uint32_t v = 0u; v < n; ++v) mesh.edge_offsets[v + 1u] = mesh.edge_offsets[v] + degree[v];
    mesh.edge_to.resize(mesh.edge_offsets[n]);
    mesh.edge_length.resize(mesh.edge_offsets[n]);
    std::vector<uint32_t> cursor(mesh.edge_offsets.begin(), mesh.edge_offsets.end() - 1);
    for (const auto& [a, b, length] : edges) {
        mesh.edge_to[cursor[a]] = b;
        mesh.edge_length[cursor[a]++] = length;
        mesh.edge_to[cursor[b]] = a;
        mesh.edge_length[cursor[b]++] = length;
    }
    std::vector<uint32_t> incident(n, 0u);
    for (const auto& t : mesh.triangles)
        for (uint32_t v : t) ++incident[v];
    mesh.triangle_offsets.assign(n + 1u, 0u);
    for (uint32_t v = 0u; v < n; ++v)
        mesh.triangle_offsets[v + 1u] = mesh.triangle_offsets[v] + incident[v];
    mesh.vertex_triangles.resize(mesh.triangle_offsets[n]);
    std::vector<uint32_t> slot(mesh.triangle_offsets.begin(), mesh.triangle_offsets.end() - 1);
    for (uint32_t t = 0u; t < Count(mesh.triangles.size()); ++t)
        for (uint32_t v : mesh.triangles[t]) mesh.vertex_triangles[slot[v]++] = t;
    return mesh;
}

// Rest-surface distances: Dijkstra over edges, with paths across triangles measured straight by
// unfolding each triangle beside a settled edge (first-order fast marching).
class Geodesic {
public:
    explicit Geodesic(const RestMesh& mesh)
        : mesh_(mesh), local_(mesh.vertices, kInf), state_(mesh.vertices, kUnseen) {}

    // One source within `radius`. Given `dist`, the run advances only where it improves those
    // distances and records them with `label`; `reached` receives every vertex the run settles.
    void Run(uint32_t source, uint32_t label, double radius, std::vector<double>* dist,
             std::vector<uint32_t>* labels, Field* reached) {
        if (dist != nullptr && (0.0 < (*dist)[source] || (*labels)[source] != label)) {
            (*dist)[source] = 0.0;
            (*labels)[source] = label;
        }
        Offer(source, 0.0, radius, dist);
        while (!queue_.empty()) {
            const Entry top = queue_.top();
            queue_.pop();
            const double d = top.first;
            const uint32_t v = top.second;
            if (state_[v] == kSettled || d > local_[v]) continue;
            state_[v] = kSettled;
            for (uint32_t i = mesh_.edge_offsets[v]; i < mesh_.edge_offsets[v + 1u]; ++i)
                Offer(mesh_.edge_to[i], d + mesh_.edge_length[i], radius, dist);
            for (uint32_t i = mesh_.triangle_offsets[v]; i < mesh_.triangle_offsets[v + 1u]; ++i) {
                const std::array<uint32_t, 3>& t = mesh_.triangles[mesh_.vertex_triangles[i]];
                const uint32_t o0 = t[0] == v ? t[1] : t[0];
                const uint32_t o1 = t[2] == v ? t[1] : t[2];
                if (state_[o0] == kSettled && state_[o1] != kSettled)
                    Offer(o1, Unfold(v, o0, o1, d, local_[o0]), radius, dist);
                if (state_[o1] == kSettled && state_[o0] != kSettled)
                    Offer(o0, Unfold(v, o1, o0, d, local_[o1]), radius, dist);
            }
        }
        for (uint32_t v : touched_) {
            if (dist != nullptr && local_[v] < (*dist)[v]) {
                (*dist)[v] = local_[v];
                (*labels)[v] = label;
            }
            if (reached != nullptr) reached->emplace_back(v, local_[v]);
            local_[v] = kInf;
            state_[v] = kUnseen;
        }
        touched_.clear();
    }

private:
    static constexpr uint8_t kUnseen = 0u;
    static constexpr uint8_t kQueued = 1u;
    static constexpr uint8_t kSettled = 2u;

    void Offer(uint32_t v, double d, double radius, const std::vector<double>* dist) {
        if (d > radius || state_[v] == kSettled || !(d < local_[v])) return;
        if (dist != nullptr && !(d < (*dist)[v] + 1e-15)) return;
        if (state_[v] == kUnseen) {
            state_[v] = kQueued;
            touched_.push_back(v);
        }
        local_[v] = d;
        queue_.emplace(d, v);
    }

    // Distance to c from a source at distances da, db of a and b, unfolded across edge ab.
    double Unfold(uint32_t a, uint32_t b, uint32_t c, double da, double db) const {
        const double lab = mesh_.Length(a, b), lac = mesh_.Length(a, c), lbc = mesh_.Length(b, c);
        if (!(lab > 0.0) || !std::isfinite(lab + lac + lbc)) return kInf;
        const double cx = (lab * lab + lac * lac - lbc * lbc) / (2.0 * lab);
        const double cy = std::sqrt(std::max(lac * lac - cx * cx, 0.0));
        const double sx = (lab * lab + da * da - db * db) / (2.0 * lab);
        const double s2 = da * da - sx * sx;
        if (s2 < 0.0) return kInf;
        const double sy = -std::sqrt(s2);
        if (cy - sy <= 0.0) return kInf;
        const double x0 = sx + (cx - sx) * (-sy) / (cy - sy);
        if (x0 < 0.0 || x0 > lab) return kInf;
        return std::hypot(cx - sx, cy - sy);
    }

    const RestMesh& mesh_;
    std::vector<double> local_;
    std::vector<uint8_t> state_;
    std::vector<uint32_t> touched_;
    MinQueue queue_;
};

// Open boundary edges and rod edges; their vertices interpolate along these chains.
struct Boundary {
    std::vector<uint32_t> offsets, to;
    std::vector<double> length;
    std::vector<uint32_t> vertices;
    std::vector<uint8_t> corner;

    uint32_t Degree(uint32_t v) const { return offsets[v + 1u] - offsets[v]; }
};

Boundary BuildBoundary(const RestMesh& mesh) {
    const uint32_t n = mesh.vertices;
    std::vector<uint8_t> surface(n, 0u);
    std::vector<std::pair<uint32_t, uint32_t>> sides;
    for (const auto& t : mesh.triangles)
        for (uint32_t j = 0u; j < 3u; ++j) {
            surface[t[j]] = 1u;
            sides.emplace_back(std::min(t[j], t[(j + 1u) % 3u]), std::max(t[j], t[(j + 1u) % 3u]));
        }
    std::sort(sides.begin(), sides.end());
    std::vector<std::pair<uint32_t, uint32_t>> open;
    for (size_t i = 0u; i < sides.size();) {
        size_t j = i + 1u;
        while (j < sides.size() && sides[j] == sides[i]) ++j;
        if (j == i + 1u) open.push_back(sides[i]);
        i = j;
    }
    for (uint32_t v = 0u; v < n; ++v)
        for (uint32_t i = mesh.edge_offsets[v]; i < mesh.edge_offsets[v + 1u]; ++i)
            if (mesh.edge_to[i] > v && (surface[v] == 0u || surface[mesh.edge_to[i]] == 0u))
                open.emplace_back(v, mesh.edge_to[i]);
    std::sort(open.begin(), open.end());
    Boundary result;
    result.offsets.assign(n + 1u, 0u);
    for (const auto& [a, b] : open) {
        ++result.offsets[a + 1u];
        ++result.offsets[b + 1u];
    }
    for (uint32_t v = 0u; v < n; ++v) result.offsets[v + 1u] += result.offsets[v];
    result.to.resize(result.offsets[n]);
    result.length.resize(result.offsets[n]);
    std::vector<uint32_t> cursor(result.offsets.begin(), result.offsets.end() - 1);
    for (const auto& [a, b] : open) {
        const double length = mesh.Length(a, b);
        result.to[cursor[a]] = b;
        result.length[cursor[a]++] = length;
        result.to[cursor[b]] = a;
        result.length[cursor[b]++] = length;
    }
    // Chain ends, junctions, rod attachments and sharp boundary turns are corners.
    result.corner.assign(n, 0u);
    for (uint32_t v = 0u; v < n; ++v) {
        const uint32_t degree = result.Degree(v);
        if (degree == 0u) continue;
        result.vertices.push_back(v);
        bool corner = degree != 2u;
        if (!corner && surface[v] != 0u) {
            for (uint32_t i = result.offsets[v]; i < result.offsets[v + 1u]; ++i)
                if (surface[result.to[i]] == 0u) corner = true;
            double angle = 0.0;
            for (uint32_t i = mesh.triangle_offsets[v]; i < mesh.triangle_offsets[v + 1u]; ++i) {
                const std::array<uint32_t, 3>& t = mesh.triangles[mesh.vertex_triangles[i]];
                const uint32_t w = t[0] == v ? t[1] : t[0];
                const uint32_t c = t[2] == v ? t[1] : t[2];
                const double lw = mesh.Length(v, w), lc = mesh.Length(v, c), lo = mesh.Length(w, c);
                angle += std::acos(std::clamp((lw * lw + lc * lc - lo * lo) / (2.0 * lw * lc), -1.0, 1.0));
            }
            // The geodesic turn of a boundary is pi less the triangle angles meeting there.
            if (!corner) corner = std::cos(kPi - angle) < std::cos(kCornerTurn);
        }
        result.corner[v] = static_cast<uint8_t>(corner);
    }
    return result;
}

void BoundaryDistances(const Boundary& boundary, const std::vector<uint32_t>& sources,
                       std::vector<double>* dist) {
    MinQueue queue;
    for (uint32_t s : sources) {
        (*dist)[s] = 0.0;
        queue.emplace(0.0, s);
    }
    while (!queue.empty()) {
        const Entry top = queue.top();
        queue.pop();
        if (top.first > (*dist)[top.second]) continue;
        for (uint32_t i = boundary.offsets[top.second]; i < boundary.offsets[top.second + 1u]; ++i) {
            const double d = top.first + boundary.length[i];
            if (d < (*dist)[boundary.to[i]]) {
                (*dist)[boundary.to[i]] = d;
                queue.emplace(d, boundary.to[i]);
            }
        }
    }
}

// Arc lengths from v to the nodes ending its chain in both directions.
bool BoundaryArc(const Boundary& boundary, const std::vector<uint32_t>& node_of, uint32_t v,
                 std::array<std::pair<uint32_t, double>, 2>* ends) {
    const uint32_t limit = Count(node_of.size());
    uint32_t found = 0u;
    for (uint32_t i = boundary.offsets[v]; i < boundary.offsets[v + 1u]; ++i) {
        uint32_t previous = v;
        uint32_t current = boundary.to[i];
        double length = boundary.length[i];
        for (uint32_t steps = 0u;
             node_of[current] == kNone && boundary.Degree(current) == 2u && steps < limit; ++steps) {
            const uint32_t first = boundary.offsets[current];
            const uint32_t next = boundary.to[first] != previous ? first : first + 1u;
            length += boundary.length[next];
            previous = current;
            current = boundary.to[next];
        }
        if (node_of[current] == kNone || found == 2u) return false;
        (*ends)[found++] = {node_of[current], length};
    }
    return found == 2u;
}

double FieldDistance(const Field& field, uint32_t node) {
    const auto it = std::lower_bound(
        field.begin(), field.end(), node,
        [](const std::pair<uint32_t, double>& entry, uint32_t key) { return entry.first < key; });
    return it != field.end() && it->first == node ? it->second : kInf;
}

struct CoarseTriangle {
    std::array<uint32_t, 3> node{};
    double l12 = 0.0, x3 = 0.0, y3 = 0.0, quality = 0.0;
};

struct Level {
    std::vector<uint32_t> nodes;  // vertex of each node
    std::vector<std::array<uint32_t, kVbdCoarseParents>> parent;
    std::vector<std::array<double, kVbdCoarseParents>> weight;
};

Level BuildLevel(const RestMesh& mesh, const Boundary& boundary, Geodesic& geodesic, double spacing) {
    const uint32_t n = mesh.vertices;
    Level level;
    std::vector<uint32_t>& nodes = level.nodes;
    std::vector<double> dist(n, kInf);
    std::vector<uint32_t> label(n, kNone);
    auto add_node = [&](uint32_t vertex) {
        nodes.push_back(vertex);
        geodesic.Run(vertex, Count(nodes.size()) - 1u, kInf, &dist, &label, nullptr);
    };
    // Farthest-point sampling at cover * spacing: corners, then boundaries, then surfaces.
    if (!boundary.vertices.empty()) {
        std::vector<double> along(n, kInf);
        std::vector<uint32_t> boundary_nodes;
        for (uint32_t v : boundary.vertices)
            if (boundary.corner[v] != 0u) boundary_nodes.push_back(v);
        if (!boundary_nodes.empty()) BoundaryDistances(boundary, boundary_nodes, &along);
        for (;;) {
            uint32_t far = kNone;
            double key = -1.0;
            for (uint32_t v : boundary.vertices) {
                const double value = std::isfinite(along[v]) ? along[v] : 1e30;
                if (value > key) {
                    key = value;
                    far = v;
                }
            }
            if (std::isfinite(along[far]) && along[far] < kCover * spacing) break;
            boundary_nodes.push_back(far);
            BoundaryDistances(boundary, {far}, &along);
        }
        for (uint32_t v : boundary_nodes) add_node(v);
    }
    if (nodes.empty()) {
        // A closed surface starts from the vertex farthest from its first vertex.
        uint32_t first = 0u;
        while (first < n && !mesh.Active(first)) ++first;
        if (first == n) return level;
        Field reached;
        geodesic.Run(first, 0u, kInf, nullptr, nullptr, &reached);
        uint32_t seed = first;
        double best = -1.0;
        for (const auto& [v, d] : reached)
            if (d > best || (d == best && v < seed)) {
                best = d;
                seed = v;
            }
        add_node(seed);
    }
    uint32_t cursor = 0u;
    for (;;) {
        while (cursor < n && (!mesh.Active(cursor) || std::isfinite(dist[cursor]))) ++cursor;
        uint32_t pick = cursor;
        if (pick == n) {
            double best = -1.0;
            for (uint32_t v = 0u; v < n; ++v)
                if (mesh.Active(v) && dist[v] > best) {
                    best = dist[v];
                    pick = v;
                }
            if (pick == n || best < kCover * spacing) break;
        }
        add_node(pick);
    }

    const uint32_t count = Count(nodes.size());
    std::vector<uint32_t> node_of(n, kNone);
    for (uint32_t k = 0u; k < count; ++k) node_of[nodes[k]] = k;
    std::vector<Field> field(n);
    Field reached;
    for (uint32_t k = 0u; k < count; ++k) {
        reached.clear();
        geodesic.Run(nodes[k], k, kFieldReach * spacing, nullptr, nullptr, &reached);
        for (const auto& [v, d] : reached) field[v].emplace_back(k, d);
    }
    auto distance = [&](uint32_t node, uint32_t vertex) { return FieldDistance(field[vertex], node); };

    // Coarse triangles and edges join the nodes of neighboring geodesic Voronoi cells.
    std::vector<std::array<uint32_t, 3>> keys;
    for (const auto& t : mesh.triangles) {
        std::array<uint32_t, 3> cell{label[t[0]], label[t[1]], label[t[2]]};
        std::sort(cell.begin(), cell.end());
        if (cell[0] != cell[1] && cell[1] != cell[2]) keys.push_back(cell);
    }
    std::sort(keys.begin(), keys.end());
    keys.erase(std::unique(keys.begin(), keys.end()), keys.end());
    std::vector<CoarseTriangle> triangles(keys.size());
    std::vector<std::vector<uint32_t>> node_triangles(count);
    for (uint32_t t = 0u; t < Count(keys.size()); ++t) {
        CoarseTriangle& ct = triangles[t];
        ct.node = keys[t];
        for (uint32_t k : ct.node) node_triangles[k].push_back(t);
        const double l12 = distance(ct.node[0], nodes[ct.node[1]]);
        const double l13 = distance(ct.node[0], nodes[ct.node[2]]);
        const double l23 = distance(ct.node[1], nodes[ct.node[2]]);
        if (!std::isfinite(l12 + l13 + l23) || !(l12 > 0.0)) continue;
        ct.l12 = l12;
        ct.x3 = (l12 * l12 + l13 * l13 - l23 * l23) / (2.0 * l12);
        ct.y3 = std::sqrt(std::max(l13 * l13 - ct.x3 * ct.x3, 0.0));
        ct.quality = 2.0 * std::sqrt(3.0) * l12 * ct.y3 / (l12 * l12 + l13 * l13 + l23 * l23);
    }
    std::vector<std::array<uint32_t, 2>> edges;
    for (uint32_t v = 0u; v < n; ++v)
        for (uint32_t i = mesh.edge_offsets[v]; i < mesh.edge_offsets[v + 1u]; ++i) {
            const uint32_t w = mesh.edge_to[i];
            if (w > v && label[v] != label[w])
                edges.push_back({std::min(label[v], label[w]), std::max(label[v], label[w])});
        }
    std::sort(edges.begin(), edges.end());
    edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
    std::vector<std::vector<uint32_t>> node_edges(count);
    for (uint32_t e = 0u; e < Count(edges.size()); ++e)
        for (uint32_t k : edges[e]) node_edges[k].push_back(e);

    // Nodes keep their own value; boundary and rod vertices interpolate by arc length; surface
    // vertices take barycentric weights in a nearby well-shaped coarse triangle, else a coarse edge.
    level.parent.assign(n, {kNone, kNone, kNone});
    level.weight.assign(n, {0.0, 0.0, 0.0});
    std::vector<uint32_t> near_triangles, near_edges;
    std::array<std::pair<uint32_t, double>, 2> ends{};
    for (uint32_t i = 0u; i < n; ++i) {
        if (!mesh.Active(i)) continue;
        uint32_t used = 0u;
        auto emit = [&](uint32_t node, double w) {
            if (!(w > 0.0)) return;
            level.parent[i][used] = node;
            level.weight[i][used] = w;
            ++used;
        };
        if (node_of[i] != kNone) {
            emit(node_of[i], 1.0);
            continue;
        }
        if (boundary.Degree(i) > 0u && BoundaryArc(boundary, node_of, i, &ends)) {
            const double a = ends[0].second, b = ends[1].second;
            if (ends[0].first == ends[1].first || !(a + b > 0.0)) {
                emit(ends[0].first, 1.0);
            } else {
                emit(ends[0].first, b / (a + b));
                emit(ends[1].first, a / (a + b));
            }
            continue;
        }
        const uint32_t own = label[i];
        near_triangles.assign(node_triangles[own].begin(), node_triangles[own].end());
        near_edges.assign(node_edges[own].begin(), node_edges[own].end());
        for (uint32_t e : node_edges[own])
            for (uint32_t k : edges[e]) {
                near_triangles.insert(near_triangles.end(), node_triangles[k].begin(), node_triangles[k].end());
                near_edges.insert(near_edges.end(), node_edges[k].begin(), node_edges[k].end());
            }
        std::sort(near_triangles.begin(), near_triangles.end());
        near_triangles.erase(std::unique(near_triangles.begin(), near_triangles.end()), near_triangles.end());
        std::sort(near_edges.begin(), near_edges.end());
        near_edges.erase(std::unique(near_edges.begin(), near_edges.end()), near_edges.end());
        // Prefer the triangle whose smallest weight is largest, then the vertex's own cell, then shape.
        uint32_t best = kNone;
        double best_score = -kInf;
        std::array<double, 3> best_lambda{};
        std::tuple<double, bool, double> best_key{};
        for (uint32_t t : near_triangles) {
            const CoarseTriangle& ct = triangles[t];
            if (ct.quality < kSliverQuality) continue;
            const double d1 = distance(ct.node[0], i), d2 = distance(ct.node[1], i);
            const double d3 = distance(ct.node[2], i);
            if (!std::isfinite(d1 + d2 + d3)) continue;
            const double bx = (d1 * d1 - d2 * d2 + ct.l12 * ct.l12) / (2.0 * ct.l12);
            const double by =
                (d1 * d1 - d3 * d3 + ct.x3 * ct.x3 + ct.y3 * ct.y3 - 2.0 * bx * ct.x3) / (2.0 * ct.y3);
            const double l3 = by / ct.y3;
            const double l2 = (bx - ct.x3 * l3) / ct.l12;
            const std::array<double, 3> lambda{1.0 - l2 - l3, l2, l3};
            const double score = std::min({lambda[0], lambda[1], lambda[2]});
            const bool in_own = ct.node[0] == own || ct.node[1] == own || ct.node[2] == own;
            const std::tuple<double, bool, double> key{std::round(score * 1e9), in_own, ct.quality};
            if (best == kNone || key > best_key) {
                best = t;
                best_score = score;
                best_lambda = lambda;
                best_key = key;
            }
        }
        if (best != kNone && best_score > kInsideTolerance) {
            double sum = 0.0;
            for (double& l : best_lambda) {
                l = std::max(l, 0.0);
                sum += l;
            }
            for (uint32_t j = 0u; j < 3u; ++j) emit(triangles[best].node[j], best_lambda[j] / sum);
            continue;
        }
        uint32_t edge = kNone;
        double offset = kInf, along = 0.0;
        for (uint32_t e : near_edges) {
            const double l = distance(edges[e][0], nodes[edges[e][1]]);
            const double d1 = distance(edges[e][0], i), d2 = distance(edges[e][1], i);
            if (!std::isfinite(l + d1 + d2) || !(l > 0.0)) continue;
            const double s = (d1 * d1 - d2 * d2 + l * l) / (2.0 * l);
            const double t = s / l;
            const double off = std::sqrt(std::max(d1 * d1 - s * s, 0.0)) + l * std::max({-t, t - 1.0, 0.0});
            if (off < offset) {
                offset = off;
                edge = e;
                along = std::clamp(t, 0.0, 1.0);
            }
        }
        if (edge != kNone) {
            emit(edges[edge][0], 1.0 - along);
            emit(edges[edge][1], along);
            continue;
        }
        emit(own, 1.0);
    }
    return level;
}

}  // namespace

void BuildVertexBlockHierarchy(Model* model) {
    if (model == nullptr) return;
    auto& p = model->particles;
    auto& cap = model->capacities;
    p.vbd_coarse_level_nodes.clear();
    p.vbd_coarse_parents.clear();
    p.vbd_coarse_weights.clear();
    p.vbd_coarse_child_offsets.clear();
    p.vbd_coarse_children.clear();
    p.vbd_coarse_child_weights.clear();
    cap.vbd_coarse_levels = 0u;
    cap.vbd_coarse_nodes = 0u;
    cap.vbd_coarse_children = 0u;
    cap.vbd_coarse_dense_nodes = 0u;
    const uint32_t n = cap.vbd_vertices_per_env;
    if (n == 0u) return;
    if (uint64_t{cap.vbd_particle_begin} + n > p.initial_pos.size())
        throw std::invalid_argument("vertex-block range exceeds the particle positions");
    const RestMesh mesh = BuildRestMesh(*model);
    std::vector<double> lengths;
    uint32_t active = 0u;
    for (uint32_t v = 0u; v < n; ++v) {
        if (mesh.Active(v)) ++active;
        for (uint32_t i = mesh.edge_offsets[v]; i < mesh.edge_offsets[v + 1u]; ++i)
            if (mesh.edge_to[i] > v) lengths.push_back(mesh.edge_length[i]);
    }
    if (lengths.empty()) return;
    std::sort(lengths.begin(), lengths.end());
    const size_t middle = lengths.size() / 2u;
    const double h = lengths.size() % 2u == 1u ? lengths[middle]
                                               : 0.5 * (lengths[middle - 1u] + lengths[middle]);
    if (!(h > 0.0)) return;
    const Boundary boundary = BuildBoundary(mesh);
    Geodesic geodesic(mesh);
    // Levels sample at twice the previous spacing until a level is dense or stops shrinking.
    std::vector<Level> levels;
    uint32_t previous = active;
    for (uint32_t l = 1u; l <= kMaxLevels; ++l) {
        Level level = BuildLevel(mesh, boundary, geodesic, std::ldexp(h, static_cast<int>(l)));
        const uint32_t count = Count(level.nodes.size());
        if (count == 0u || count >= previous) break;
        levels.push_back(std::move(level));
        if (count <= kVbdCoarseDenseNodes) break;
        previous = count;
    }
    if (levels.empty()) return;
    const uint32_t level_count = Count(levels.size());
    p.vbd_coarse_level_nodes.assign(level_count + 1u, 0u);
    for (uint32_t l = 0u; l < level_count; ++l)
        p.vbd_coarse_level_nodes[l + 1u] = p.vbd_coarse_level_nodes[l] + Count(levels[l].nodes.size());
    const uint32_t nodes = p.vbd_coarse_level_nodes.back();
    p.vbd_coarse_parents.assign(size_t{level_count} * n * kVbdCoarseParents, kNone);
    p.vbd_coarse_weights.assign(size_t{level_count} * n * kVbdCoarseParents, 0.0f);
    std::vector<uint32_t> children(nodes, 0u);
    for (uint32_t l = 0u; l < level_count; ++l)
        for (uint32_t v = 0u; v < n; ++v)
            for (uint32_t j = 0u; j < kVbdCoarseParents; ++j) {
                if (levels[l].parent[v][j] == kNone) continue;
                const size_t slot = (size_t{l} * n + v) * kVbdCoarseParents + j;
                const uint32_t node = p.vbd_coarse_level_nodes[l] + levels[l].parent[v][j];
                p.vbd_coarse_parents[slot] = node;
                p.vbd_coarse_weights[slot] = static_cast<float>(levels[l].weight[v][j]);
                ++children[node];
            }
    p.vbd_coarse_child_offsets.assign(nodes + 1u, 0u);
    for (uint32_t k = 0u; k < nodes; ++k)
        p.vbd_coarse_child_offsets[k + 1u] = p.vbd_coarse_child_offsets[k] + children[k];
    p.vbd_coarse_children.resize(p.vbd_coarse_child_offsets.back());
    p.vbd_coarse_child_weights.resize(p.vbd_coarse_child_offsets.back());
    std::vector<uint32_t> cursor(p.vbd_coarse_child_offsets.begin(), p.vbd_coarse_child_offsets.end() - 1);
    for (size_t slot = 0u; slot < p.vbd_coarse_parents.size(); ++slot) {
        const uint32_t node = p.vbd_coarse_parents[slot];
        if (node == kNone) continue;
        p.vbd_coarse_children[cursor[node]] = static_cast<uint32_t>((slot / kVbdCoarseParents) % n);
        p.vbd_coarse_child_weights[cursor[node]++] = p.vbd_coarse_weights[slot];
    }
    cap.vbd_coarse_levels = level_count;
    cap.vbd_coarse_nodes = nodes;
    cap.vbd_coarse_children = Count(p.vbd_coarse_children.size());
    const uint32_t last = Count(levels.back().nodes.size());
    cap.vbd_coarse_dense_nodes = last <= kVbdCoarseDenseNodes ? last : 0u;
}

}  // namespace nuka::nk
