#include "runtime/app/viewer/editor_collision.hpp"

#include "collision/convex_narrowphase.hpp"
#include "import/cooker/convex_decomposition.hpp"
#include "runtime/app/viewer/editor_import.hpp"
#include "scene/graph/scene_graph.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <unordered_map>
#include <utility>

namespace nuka::runtime::app::viewer {
namespace {

namespace json = scene::json;
namespace cooker = import::cooker;
namespace cvx = collision::cvx;

constexpr uint32_t kHullVertices = 64u;
// The cook's decomposition cache, so a preview's decomposition is the one the cook reuses.
constexpr const char* kDecomposeCacheDir = ".nuka_cache";

bool Visual(const scene::CollisionShapeRecord& sh) { return sh.contype == 0u && sh.conaffinity == 0u; }

bool HasCollider(const scene::SceneIR& s, scene::BodyId b) {
    return std::any_of(s.Shapes().begin(), s.Shapes().end(),
                       [&](const scene::CollisionShapeRecord& sh) { return sh.body_id == b && !Visual(sh); });
}

// The body's visual triangles in its own frame, and the material of the first of them.
struct BodyMesh {
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    scene::MaterialId material = scene::kInvalidMaterial;
};

BodyMesh VisualMeshOf(const scene::SceneIR& s, scene::BodyId b) {
    BodyMesh mesh;
    for (const scene::CollisionShapeRecord& sh : s.Shapes()) {
        if (sh.body_id != b || !Visual(sh) || sh.mesh_vertices.empty() || sh.mesh_indices.empty()) continue;
        if (mesh.material == scene::kInvalidMaterial) mesh.material = sh.material_id;
        const auto base = static_cast<uint32_t>(mesh.vertices.size() / 3u);
        for (size_t i = 0; i + 2u < sh.mesh_vertices.size(); i += 3u) {
            const math::Vec3 p = sh.local_transform.TransformPoint(
                math::Vec3{sh.mesh_vertices[i], sh.mesh_vertices[i + 1u], sh.mesh_vertices[i + 2u]});
            mesh.vertices.insert(mesh.vertices.end(), {p.x, p.y, p.z});
        }
        for (const uint32_t index : sh.mesh_indices) mesh.indices.push_back(base + index);
    }
    return mesh;
}

using Vec3d = std::array<double, 3>;

Vec3d PointOf(const BodyMesh& mesh, uint32_t v) {
    return {mesh.vertices[v * 3u], mesh.vertices[v * 3u + 1u], mesh.vertices[v * 3u + 2u]};
}

Vec3d Sub(const Vec3d& a, const Vec3d& b) { return {a[0] - b[0], a[1] - b[1], a[2] - b[2]}; }

Vec3d Cross(const Vec3d& a, const Vec3d& b) {
    return {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]};
}

double Dot(const Vec3d& a, const Vec3d& b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// The grid cell of every vertex, numbered in the order the vertices first reach them.
std::vector<uint32_t> CellsOf(const BodyMesh& mesh, const Vec3d& origin, double cell, uint32_t* count) {
    std::vector<uint32_t> out(mesh.vertices.size() / 3u);
    std::unordered_map<uint64_t, uint32_t> ids;
    ids.reserve(out.size());
    for (uint32_t v = 0; v < out.size(); ++v) {
        const Vec3d p = Sub(PointOf(mesh, v), origin);
        const uint64_t key = static_cast<uint64_t>(p[0] / cell) << 42u | static_cast<uint64_t>(p[1] / cell) << 21u |
                             static_cast<uint64_t>(p[2] / cell);
        out[v] = ids.emplace(key, static_cast<uint32_t>(ids.size())).first->second;
    }
    *count = static_cast<uint32_t>(ids.size());
    return out;
}

// The triangles left when each vertex merges into its cell: those on three cells, the first of any that
// share their cells.
std::vector<uint32_t> MergedTriangles(const BodyMesh& mesh, const std::vector<uint32_t>& cell_of) {
    std::vector<std::array<uint32_t, 4>> keys;  // the sorted cells, then the triangle
    keys.reserve(mesh.indices.size() / 3u);
    for (size_t t = 0; t + 2u < mesh.indices.size(); t += 3u) {
        std::array<uint32_t, 3> c = {cell_of[mesh.indices[t]], cell_of[mesh.indices[t + 1u]],
                                     cell_of[mesh.indices[t + 2u]]};
        if (c[0] == c[1] || c[1] == c[2] || c[0] == c[2]) continue;
        std::sort(c.begin(), c.end());
        keys.push_back({c[0], c[1], c[2], static_cast<uint32_t>(t / 3u)});
    }
    std::sort(keys.begin(), keys.end());
    std::vector<uint32_t> kept;
    for (size_t i = 0; i < keys.size(); ++i) {
        const bool repeat = i != 0u && std::equal(keys[i].begin(), keys[i].begin() + 3, keys[i - 1u].begin());
        if (!repeat) kept.push_back(keys[i][3]);
    }
    std::sort(kept.begin(), kept.end());
    return kept;
}

// The eigenvalues `w` and eigenvectors (the columns of `v`) of the symmetric matrix `a`, by Jacobi sweeps.
void SymmetricEigen(std::array<double, 9> a, Vec3d& w, std::array<double, 9>& v) {
    v = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    for (int sweep = 0; sweep < 16; ++sweep) {
        const double off = std::abs(a[1]) + std::abs(a[2]) + std::abs(a[5]);
        if (off <= 1e-12 * (std::abs(a[0]) + std::abs(a[4]) + std::abs(a[8]))) break;
        for (int p = 0; p < 2; ++p) {
            for (int q = p + 1; q < 3; ++q) {
                if (a[p * 3 + q] == 0.0) continue;
                const double theta = (a[q * 3 + q] - a[p * 3 + p]) / (2.0 * a[p * 3 + q]);
                const double t = (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
                const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
                for (int k = 0; k < 3; ++k) {
                    const double kp = a[k * 3 + p], kq = a[k * 3 + q];
                    a[k * 3 + p] = c * kp - s * kq;
                    a[k * 3 + q] = s * kp + c * kq;
                }
                for (int k = 0; k < 3; ++k) {
                    const double pk = a[p * 3 + k], qk = a[q * 3 + k];
                    a[p * 3 + k] = c * pk - s * qk;
                    a[q * 3 + k] = s * pk + c * qk;
                }
                for (int k = 0; k < 3; ++k) {
                    const double kp = v[k * 3 + p], kq = v[k * 3 + q];
                    v[k * 3 + p] = c * kp - s * kq;
                    v[k * 3 + q] = s * kp + c * kq;
                }
            }
        }
    }
    w = {a[0], a[4], a[8]};
}

// A cell's vertices and the area-weighted plane quadrics of their triangles.
struct CellFit {
    std::array<double, 9> a{};
    Vec3d b{};
    Vec3d sum{};
    uint32_t count = 0u;
};

// The point that fits the cell's planes best, moved from its vertices' mean only along the directions the
// planes pin down, so floors stay flat and corners stay sharp; it stays inside the cell.
Vec3d Representative(const CellFit& f, const Vec3d& origin, double cell) {
    const double n = static_cast<double>(f.count);
    const Vec3d mean = {f.sum[0] / n, f.sum[1] / n, f.sum[2] / n};
    Vec3d x = mean;
    Vec3d r = {-f.b[0], -f.b[1], -f.b[2]};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r[i] -= f.a[i * 3 + j] * x[j];
    Vec3d w;
    std::array<double, 9> v;
    SymmetricEigen(f.a, w, v);
    const double top = std::max({w[0], w[1], w[2]});
    for (int e = 0; e < 3; ++e) {
        if (w[e] <= 1e-3 * top) continue;
        const Vec3d axis = {v[e], v[3 + e], v[6 + e]};
        const double step = Dot(axis, r) / w[e];
        for (int k = 0; k < 3; ++k) x[k] += step * axis[k];
    }
    for (int k = 0; k < 3; ++k) {
        const double lo = origin[k] + std::floor((mean[k] - origin[k]) / cell) * cell;
        x[k] = std::clamp(x[k], lo, lo + cell);
    }
    return x;
}

// Drops triangles until each patch joined across edges that two triangles share can take one winding, as the
// cook requires; merging can fold a surface into a Moebius-like loop.
void DropFolds(BodyMesh& mesh) {
    for (bool dropped = true; dropped;) {
        const auto count = static_cast<uint32_t>(mesh.indices.size() / 3u);
        // Each edge, then the triangle and whether it runs from the lower index.
        std::vector<std::pair<uint64_t, uint32_t>> sides;
        sides.reserve(mesh.indices.size());
        for (uint32_t t = 0; t < count; ++t) {
            for (uint32_t k = 0; k < 3u; ++k) {
                const uint32_t a = mesh.indices[t * 3u + k], b = mesh.indices[t * 3u + (k + 1u) % 3u];
                sides.push_back({uint64_t{std::min(a, b)} << 32u | std::max(a, b), t << 1u | uint32_t{a < b}});
            }
        }
        std::sort(sides.begin(), sides.end());
        std::vector<std::vector<std::pair<uint32_t, int8_t>>> next(count);
        for (size_t i = 0, j = 0; i < sides.size(); i = j) {
            while (j < sides.size() && sides[j].first == sides[i].first) ++j;
            if (j - i != 2u) continue;
            const uint32_t p = sides[i].second, q = sides[i + 1u].second;
            const auto same = static_cast<int8_t>((p & 1u) == (q & 1u));
            next[p >> 1u].push_back({q >> 1u, same});
            next[q >> 1u].push_back({p >> 1u, same});
        }
        std::vector<int8_t> flip(count, -1);
        std::vector<uint8_t> fold(count, 0u);
        dropped = false;
        for (uint32_t seed = 0; seed < count; ++seed) {
            if (flip[seed] >= 0) continue;
            flip[seed] = 0;
            std::vector<uint32_t> patch = {seed};
            for (size_t head = 0; head < patch.size(); ++head) {
                for (const auto& [other, parity] : next[patch[head]]) {
                    const auto want = static_cast<int8_t>(flip[patch[head]] ^ parity);
                    if (flip[other] < 0) {
                        flip[other] = want;
                        patch.push_back(other);
                    } else if (flip[other] != want) {
                        fold[other] = 1u;
                        dropped = true;
                    }
                }
            }
        }
        std::vector<uint32_t> kept;
        for (uint32_t t = 0; t < count; ++t) {
            const auto first = mesh.indices.begin() + t * 3u;
            if (fold[t] == 0u) kept.insert(kept.end(), first, first + 3);
        }
        mesh.indices = std::move(kept);
    }
}

// Merges the mesh's vertices on about the finest grid that leaves at most `limit` triangles, so a large
// environment's collider fits the contact budget.
void ClusterMesh(BodyMesh& mesh, uint32_t limit) {
    if (limit == 0u || mesh.indices.size() / 3u <= limit) return;
    Vec3d origin;
    origin.fill(std::numeric_limits<double>::max());
    for (uint32_t v = 0; v < mesh.vertices.size() / 3u; ++v)
        for (int k = 0; k < 3; ++k) origin[k] = std::min(origin[k], static_cast<double>(mesh.vertices[v * 3u + k]));
    double area = 0.0;
    for (size_t t = 0; t + 2u < mesh.indices.size(); t += 3u) {
        const Vec3d p = PointOf(mesh, mesh.indices[t]);
        const Vec3d n = Cross(Sub(PointOf(mesh, mesh.indices[t + 1u]), p), Sub(PointOf(mesh, mesh.indices[t + 2u]), p));
        area += 0.5 * std::sqrt(Dot(n, n));
    }
    // About two triangles per cell a surface crosses; the search brackets the cell between one that keeps
    // too many and one that fits.
    double cell = std::sqrt(2.0 * area / limit), over = 0.0, fits = 0.0;
    uint32_t cells = 0u;
    for (int pass = 0; pass < 6 || fits == 0.0; ++pass) {
        const size_t kept = MergedTriangles(mesh, CellsOf(mesh, origin, cell, &cells)).size();
        if (kept > limit) over = std::max(over, cell);
        else if (fits == 0.0 || cell < fits) fits = cell;
        if (kept <= limit && kept * 5u >= size_t{limit} * 4u) break;
        const double ratio = std::sqrt(static_cast<double>(std::max<size_t>(kept, 1u)) / limit);
        cell *= kept > limit ? std::max(ratio, 1.1) : ratio;
        if (cell <= over || (fits != 0.0 && cell >= fits))
            cell = over == 0.0 ? fits * 0.5 : fits == 0.0 ? over * 2.0 : std::sqrt(over * fits);
    }

    const std::vector<uint32_t> cell_of = CellsOf(mesh, origin, fits, &cells);
    std::vector<CellFit> fit(cells);
    for (uint32_t v = 0; v < cell_of.size(); ++v) {
        CellFit& f = fit[cell_of[v]];
        const Vec3d p = PointOf(mesh, v);
        for (int k = 0; k < 3; ++k) f.sum[k] += p[k];
        ++f.count;
    }
    for (size_t t = 0; t + 2u < mesh.indices.size(); t += 3u) {
        const Vec3d p = PointOf(mesh, mesh.indices[t]);
        Vec3d n = Cross(Sub(PointOf(mesh, mesh.indices[t + 1u]), p), Sub(PointOf(mesh, mesh.indices[t + 2u]), p));
        const double len = std::sqrt(Dot(n, n));
        if (len == 0.0) continue;
        for (double& c : n) c /= len;
        const double d = -Dot(n, p), weight = 0.5 * len;
        for (size_t k = 0; k < 3u; ++k) {
            CellFit& f = fit[cell_of[mesh.indices[t + k]]];
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) f.a[i * 3 + j] += weight * n[i] * n[j];
                f.b[i] += weight * d * n[i];
            }
        }
    }
    std::vector<uint32_t> out_of(cells, std::numeric_limits<uint32_t>::max());
    std::map<std::array<float, 3>, uint32_t> welded;  // the cook welds equal points, so they share an index here
    std::vector<float> vertices;
    std::vector<uint32_t> indices;
    for (const uint32_t t : MergedTriangles(mesh, cell_of)) {
        std::array<uint32_t, 3> tri;
        for (uint32_t k = 0; k < 3u; ++k) {
            const uint32_t c = cell_of[mesh.indices[t * 3u + k]];
            if (out_of[c] == std::numeric_limits<uint32_t>::max()) {
                const Vec3d p = Representative(fit[c], origin, fits);
                const std::array<float, 3> point = {static_cast<float>(p[0]), static_cast<float>(p[1]),
                                                    static_cast<float>(p[2])};
                out_of[c] = welded.emplace(point, static_cast<uint32_t>(welded.size())).first->second;
                if (out_of[c] == vertices.size() / 3u) vertices.insert(vertices.end(), point.begin(), point.end());
            }
            tri[k] = out_of[c];
        }
        if (tri[0] != tri[1] && tri[1] != tri[2] && tri[0] != tri[2])
            indices.insert(indices.end(), tri.begin(), tri.end());
    }
    mesh.vertices = std::move(vertices);
    mesh.indices = std::move(indices);
    DropFolds(mesh);
}

std::string PathOfBody(const scene::SceneIR& s, scene::BodyId b) {
    return s.Tree().PathOf(s.Ecs().NodeOf(s.EntityOfBody(b)));
}

std::string Leaf(const std::string& name) {
    const size_t slash = name.rfind('/');
    return slash == std::string::npos ? name : name.substr(slash + 1u);
}

const char* Suffix(CollisionMethod method) {
    switch (method) {
        case CollisionMethod::Hull: return "_hull";
        case CollisionMethod::Decompose: return "_vhacd";
        case CollisionMethod::Sdf: return "_sdf";
        case CollisionMethod::Mesh: return "_mesh";
    }
    return "";
}

// The reason `b` is left alone, or empty.
std::string SkipReason(const scene::SceneIR& s, scene::BodyId b, const BodyMesh& mesh, const CollisionOptions& o) {
    if (mesh.indices.empty()) return "no visual mesh";
    if (o.only_missing && HasCollider(s, b)) return "has a collider";
    return std::string();
}

// The body's convex-hull colliders in world space, at the pose its joints start from.
std::vector<float> HullPoints(const scene::SceneIR& s, scene::BodyId b) {
    std::vector<float> out;
    const math::Transform world = BodyWorld(s, b);
    for (const scene::CollisionShapeRecord& sh : s.Shapes()) {
        if (sh.body_id != b || sh.type != scene::ShapeType::ConvexHull) continue;
        for (size_t i = 0; i + 2u < sh.mesh_vertices.size(); i += 3u) {
            const math::Vec3 p = world.TransformPoint(sh.local_transform.TransformPoint(
                math::Vec3{sh.mesh_vertices[i], sh.mesh_vertices[i + 1u], sh.mesh_vertices[i + 2u]}));
            out.insert(out.end(), {p.x, p.y, p.z});
        }
    }
    return out;
}

bool HullsOverlap(const std::vector<float>& a, const std::vector<float>& b) {
    cvx::ConvexHullView ha, hb;
    ha.verts = a.data();
    ha.vcount = static_cast<uint32_t>(a.size() / 3u);
    hb.verts = b.data();
    hb.vcount = static_cast<uint32_t>(b.size() / 3u);
    cvx::SupportProxy pa, pb;
    pa.hull = &ha;
    pb.hull = &hb;
    return cvx::GjkIntersect(pa, pb).intersect;
}

// Hulls of one articulation's links that overlap where the joints start would fail the contact pass on
// the first step, so each such pair stops colliding; jointed neighbours never collide anyway.
void ExcludeOverlappingLinks(scene::SceneIR& s, const std::vector<scene::BodyId>& made) {
    std::unordered_map<scene::BodyId, scene::BodyId> parent;
    for (const scene::JointRecord& j : s.Joints()) parent[j.child_body] = j.parent_body;
    const auto root = [&](scene::BodyId b) {
        for (auto it = parent.find(b); it != parent.end() && it->second != scene::kInvalidBody; it = parent.find(b))
            b = it->second;
        return b;
    };
    std::unordered_map<scene::BodyId, std::vector<float>> points;
    const auto hull = [&](scene::BodyId b) -> const std::vector<float>& {
        auto it = points.find(b);
        if (it == points.end()) it = points.emplace(b, HullPoints(s, b)).first;
        return it->second;
    };
    const auto joined = [&](scene::BodyId a, scene::BodyId b) {
        const auto pa = parent.find(a), pb = parent.find(b);
        return (pa != parent.end() && pa->second == b) || (pb != parent.end() && pb->second == a);
    };
    for (const scene::BodyId a : made) {
        if (hull(a).empty()) continue;
        for (scene::BodyId b = 0; b < s.RigidBodyCount(); ++b) {
            const bool seen = b < a && std::binary_search(made.begin(), made.end(), b);
            if (b == a || seen || joined(a, b) || root(b) != root(a) || hull(b).empty()) continue;
            if (HullsOverlap(hull(a), hull(b))) s.AddExcludePair(a, b);
        }
    }
}

}  // namespace

bool ParseCollisionMethod(const std::string& name, CollisionMethod* out) {
    for (const CollisionMethod m : {CollisionMethod::Hull, CollisionMethod::Decompose, CollisionMethod::Sdf, CollisionMethod::Mesh}) {
        if (name == CollisionMethodName(m)) {
            *out = m;
            return true;
        }
    }
    return false;
}

const char* CollisionMethodName(CollisionMethod method) {
    switch (method) {
        case CollisionMethod::Hull: return "hull";
        case CollisionMethod::Decompose: return "vhacd";
        case CollisionMethod::Sdf: return "sdf";
        case CollisionMethod::Mesh: return "mesh";
    }
    return "";
}

CollisionOptions CollisionOptionsFromJson(const json::Value& v) {
    CollisionOptions o;
    if (const json::Value* m = v.Find("method")) ParseCollisionMethod(m->AsString(), &o.method);
    if (const json::Value* n = v.Find("max_pieces")) o.max_pieces = static_cast<uint32_t>(std::max<int64_t>(n->AsInt(), 1));
    if (const json::Value* n = v.Find("max_triangles"))
        o.max_triangles = static_cast<uint32_t>(std::max<int64_t>(n->AsInt(), 0));
    if (const json::Value* r = v.Find("replace")) o.replace = r->AsBool();
    if (const json::Value* m = v.Find("only_missing")) o.only_missing = m->AsBool();
    if (const json::Value* t = v.Find("subtree")) o.subtree = t->AsBool();
    return o;
}

json::Value CollisionOptionsToJson(const CollisionOptions& o) {
    json::Value v = json::Value::Object();
    v.Set("method", json::Value::Str(CollisionMethodName(o.method)));
    v.Set("max_pieces", json::Value::Int(o.max_pieces));
    v.Set("max_triangles", json::Value::Int(o.max_triangles));
    v.Set("replace", json::Value::Bool(o.replace));
    v.Set("only_missing", json::Value::Bool(o.only_missing));
    v.Set("subtree", json::Value::Bool(o.subtree));
    return v;
}

std::vector<scene::BodyId> CollisionBodies(const scene::SceneIR& s, const std::vector<std::string>& paths, bool subtree) {
    std::unordered_map<scene::EntityId, scene::BodyId, scene::EntityIdHash> body_of;
    for (scene::BodyId b = 0; b < s.RigidBodyCount(); ++b) body_of.emplace(s.EntityOfBody(b), b);
    std::vector<scene::BodyId> bodies;
    const auto add = [&](scene::BodyId b) {
        if (std::find(bodies.begin(), bodies.end(), b) == bodies.end()) bodies.push_back(b);
    };
    for (const std::string& path : paths) {
        const auto node = s.Tree().NodeOf(path);
        if (!node) continue;
        // A node inside a body names that body, unless it is the body and its subtree counts.
        auto carrier = node;
        while (carrier && body_of.count(carrier->entity) == 0u) carrier = carrier->parent.lock();
        if (carrier && !(carrier == node && subtree)) {
            add(body_of.at(carrier->entity));
            continue;
        }
        std::vector<std::shared_ptr<scene::SceneNode>> stack = {node};
        while (!stack.empty()) {
            const auto n = stack.back();
            stack.pop_back();
            if (const auto it = body_of.find(n->entity); it != body_of.end()) add(it->second);
            for (auto c = n->first_child; c; c = c->next_sibling) stack.push_back(c);
        }
    }
    std::sort(bodies.begin(), bodies.end());
    return bodies;
}

std::vector<CollisionPreview> PreviewCollision(const scene::SceneIR& s, const std::vector<std::string>& paths,
                                               const CollisionOptions& o) {
    std::vector<CollisionPreview> out;
    for (const scene::BodyId b : CollisionBodies(s, paths, o.subtree)) {
        BodyMesh mesh = VisualMeshOf(s, b);
        CollisionPreview p;
        p.path = PathOfBody(s, b);
        p.triangles = static_cast<uint32_t>(mesh.indices.size() / 3u);
        p.had_collider = HasCollider(s, b);
        p.skipped = SkipReason(s, b, mesh, o);
        if (p.skipped.empty()) {
            const auto vertex_count = static_cast<uint32_t>(mesh.vertices.size() / 3u);
            switch (o.method) {
                case CollisionMethod::Hull:
                    p.pieces = 1u;
                    p.vertices = static_cast<uint32_t>(
                        cooker::ComputeConvexHull(mesh.vertices.data(), vertex_count, kHullVertices).vertices.size() / 3u);
                    break;
                case CollisionMethod::Decompose: {
                    cooker::ConvexDecompositionParams params;
                    params.max_pieces = o.max_pieces;
                    bool hit = false;
                    const cooker::ConvexDecompositionResult r = cooker::DecomposeMeshCached(
                        mesh.vertices.data(), vertex_count, mesh.indices.data(), p.triangles, params, kDecomposeCacheDir, &hit);
                    p.pieces = static_cast<uint32_t>(r.pieces.size());
                    for (const cooker::ConvexPiece& piece : r.pieces) p.vertices += static_cast<uint32_t>(piece.vertices.size() / 3u);
                    if (!r.succeeded) p.skipped = r.error_message;
                    break;
                }
                case CollisionMethod::Sdf:
                    p.pieces = 1u;
                    p.vertices = vertex_count;
                    break;
                case CollisionMethod::Mesh:
                    ClusterMesh(mesh, o.max_triangles);
                    p.pieces = 1u;
                    p.vertices = static_cast<uint32_t>(mesh.vertices.size() / 3u);
                    break;
            }
        }
        out.push_back(std::move(p));
    }
    return out;
}

json::Value PreviewToJson(const std::vector<CollisionPreview>& preview) {
    json::Value a = json::Value::Array();
    for (const CollisionPreview& p : preview) {
        json::Value v = json::Value::Object();
        v.Set("path", json::Value::Str(p.path));
        v.Set("triangles", json::Value::Int(p.triangles));
        v.Set("pieces", json::Value::Int(p.pieces));
        v.Set("vertices", json::Value::Int(p.vertices));
        v.Set("had_collider", json::Value::Bool(p.had_collider));
        v.Set("skipped", json::Value::Str(p.skipped));
        a.PushBack(std::move(v));
    }
    return a;
}

size_t GenerateCollision(scene::SceneIR& s, const std::vector<std::string>& paths, const CollisionOptions& o) {
    std::vector<scene::BodyId> made;
    for (const scene::BodyId b : CollisionBodies(s, paths, o.subtree)) {
        BodyMesh mesh = VisualMeshOf(s, b);
        if (!SkipReason(s, b, mesh, o).empty()) continue;
        if (o.replace)
            for (scene::ShapeId i = static_cast<scene::ShapeId>(s.ShapeCount()); i-- > 0u;)
                if (s.GetShape(i).body_id == b && !Visual(s.GetShape(i))) s.RemoveShape(i);
        scene::CollisionShapeRecord col;
        col.body_id = b;
        col.material_id = mesh.material;
        col.name = Leaf(s.GetBody(b).name) + Suffix(o.method);
        if (o.method == CollisionMethod::Hull) {
            cooker::ConvexPiece hull = cooker::ComputeConvexHull(
                mesh.vertices.data(), static_cast<uint32_t>(mesh.vertices.size() / 3u), kHullVertices);
            col.type = scene::ShapeType::ConvexHull;
            col.mesh_vertices = std::move(hull.vertices);
            col.mesh_indices = std::move(hull.indices);
        } else {
            if (o.method == CollisionMethod::Mesh) ClusterMesh(mesh, o.max_triangles);
            col.type = scene::ShapeType::TriMesh;
            col.mesh_vertices = std::move(mesh.vertices);
            col.mesh_indices = std::move(mesh.indices);
            if (o.method == CollisionMethod::Decompose) {
                col.decompose_mode = scene::DecomposeMode::Force;
                col.decompose_max_pieces = o.max_pieces;
            }
            if (o.method == CollisionMethod::Sdf) col.mesh_contact = scene::CollisionShapeRecord::MeshContact::Sdf;
        }
        s.InsertCollisionShape(std::move(col));
        made.push_back(b);
    }
    ExcludeOverlappingLinks(s, made);
    return made.size();
}

}  // namespace nuka::runtime::app::viewer
