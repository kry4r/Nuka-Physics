#include "runtime/app/viewer/editor_collision.hpp"

#include "collision/convex_narrowphase.hpp"
#include "import/cooker/convex_decomposition.hpp"
#include "runtime/app/viewer/editor_import.hpp"
#include "scene/graph/scene_graph.hpp"

#include <algorithm>
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
    if (const json::Value* r = v.Find("replace")) o.replace = r->AsBool();
    if (const json::Value* m = v.Find("only_missing")) o.only_missing = m->AsBool();
    if (const json::Value* t = v.Find("subtree")) o.subtree = t->AsBool();
    return o;
}

json::Value CollisionOptionsToJson(const CollisionOptions& o) {
    json::Value v = json::Value::Object();
    v.Set("method", json::Value::Str(CollisionMethodName(o.method)));
    v.Set("max_pieces", json::Value::Int(o.max_pieces));
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
        const BodyMesh mesh = VisualMeshOf(s, b);
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
                case CollisionMethod::Mesh:
                    p.pieces = 1u;
                    p.vertices = vertex_count;
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
