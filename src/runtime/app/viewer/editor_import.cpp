#include "runtime/app/viewer/editor_import.hpp"

#include "import/cooker/convex_decomposition.hpp"
#include "import/mesh_file_loader.hpp"
#include "import/mjcf_importer.hpp"
#include "import/urdf_importer.hpp"
#include "import/usd_importer.hpp"
#include "runtime/app/viewer/editor_materials.hpp"
#include "runtime/app/viewer/editor_profile.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_compose.hpp"

#include "tinyxml2.h"

#include <algorithm>
#include <cctype>
#include <cfloat>
#include <cmath>
#include <exception>
#include <fstream>
#include <string>
#include <unordered_map>
#include <utility>

namespace nuka::runtime::app::viewer {
namespace {

namespace json = scene::json;

constexpr float kWaterDensity = 1000.0f;  // kg/m^3
constexpr uint32_t kHullVertices = 64u;
constexpr size_t kSniffBytes = 4096u;

std::string Lower(std::string text) {
    for (char& c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return text;
}

// The name of the first element in the file's opening bytes.
std::string RootElement(const std::string& path) {
    std::ifstream in(Utf8Path(path), std::ios::binary);
    std::string head(kSniffBytes, '\0');
    in.read(head.data(), static_cast<std::streamsize>(head.size()));
    head.resize(static_cast<size_t>(in.gcount()));
    for (size_t at = head.find('<'); at != std::string::npos; at = head.find('<', at + 1u)) {
        if (at + 1u >= head.size() || !std::isalpha(static_cast<unsigned char>(head[at + 1u]))) continue;
        size_t end = at + 1u;
        while (end < head.size() && (std::isalnum(static_cast<unsigned char>(head[end])) || head[end] == '_')) ++end;
        return head.substr(at + 1u, end - at - 1u);
    }
    return std::string();
}

// The top-level elements of an XML file by their first line, for naming a diagnostic's owner.
struct Owner {
    int line = 0;
    std::string text;
};

std::vector<Owner> Owners(const std::string& path) {
    std::vector<Owner> owners;
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS) return owners;
    for (const tinyxml2::XMLElement* e = doc.RootElement()->FirstChildElement(); e != nullptr; e = e->NextSiblingElement()) {
        const char* name = e->Attribute("name");
        owners.push_back(Owner{e->GetLineNum(), name != nullptr ? std::string(e->Name()) + " " + name : std::string(e->Name())});
    }
    return owners;
}

std::string OwnerOf(const std::vector<Owner>& owners, int line) {
    std::string owner;
    for (const Owner& o : owners) {
        if (o.line > line) break;
        owner = o.text;
    }
    return owner;
}

void Grow(const math::Vec3& p, math::Vec3* lo, math::Vec3* hi) {
    lo->x = std::min(lo->x, p.x);
    lo->y = std::min(lo->y, p.y);
    lo->z = std::min(lo->z, p.z);
    hi->x = std::max(hi->x, p.x);
    hi->y = std::max(hi->y, p.y);
    hi->z = std::max(hi->z, p.z);
}

// Grows the box by a shape posed at `world`; planes and height fields have no extent here.
void GrowShape(const scene::CollisionShapeRecord& sh, const math::Transform& world, math::Vec3* lo, math::Vec3* hi) {
    switch (sh.type) {
        case scene::ShapeType::Box:
            for (int i = 0; i < 8; ++i) {
                const math::Vec3 c{(i & 1) != 0 ? sh.half_extents.x : -sh.half_extents.x,
                                   (i & 2) != 0 ? sh.half_extents.y : -sh.half_extents.y,
                                   (i & 4) != 0 ? sh.half_extents.z : -sh.half_extents.z};
                Grow(world.TransformPoint(c), lo, hi);
            }
            break;
        case scene::ShapeType::Sphere:
        case scene::ShapeType::Capsule: {
            const float h = sh.type == scene::ShapeType::Capsule ? sh.half_height : 0.0f;
            const math::Vec3 r{sh.radius, sh.radius, sh.radius};
            for (const float z : {-h, h}) {
                const math::Vec3 c = world.TransformPoint(math::Vec3{0.0f, 0.0f, z});
                Grow(c - r, lo, hi);
                Grow(c + r, lo, hi);
            }
            break;
        }
        case scene::ShapeType::Cylinder: {
            // Eight rim points per cap bound the disk closely enough for placement.
            for (const float z : {-sh.half_height, sh.half_height})
                for (int k = 0; k < 8; ++k) {
                    const float t = 0.78539816f * static_cast<float>(k);
                    Grow(world.TransformPoint(math::Vec3{sh.radius * std::cos(t), sh.radius * std::sin(t), z}), lo, hi);
                }
            break;
        }
        case scene::ShapeType::ConvexHull:
        case scene::ShapeType::TriMesh:
            for (size_t i = 0; i + 2u < sh.mesh_vertices.size(); i += 3u)
                Grow(world.TransformPoint(math::Vec3{sh.mesh_vertices[i], sh.mesh_vertices[i + 1u], sh.mesh_vertices[i + 2u]}),
                     lo, hi);
            break;
        case scene::ShapeType::Plane:
        case scene::ShapeType::HeightField:
            break;
    }
}

// The bounds of `s` with its root frame placed at `root`.
bool BoundsUnder(const scene::SceneIR& s, const math::Transform& root, math::Vec3* lo, math::Vec3* hi) {
    *lo = math::Vec3{FLT_MAX, FLT_MAX, FLT_MAX};
    *hi = math::Vec3{-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (const scene::CollisionShapeRecord& sh : s.Shapes()) {
        const math::Transform body = sh.body_id != scene::kInvalidBody ? BodyWorld(s, sh.body_id) : math::Transform::Identity();
        GrowShape(sh, root * body * sh.local_transform, lo, hi);
    }
    return lo->x <= hi->x;
}

ImportStats StatsOf(const scene::SceneIR& s) {
    ImportStats st;
    st.links = static_cast<uint32_t>(s.RigidBodyCount());
    st.joints = static_cast<uint32_t>(s.JointCount());
    for (const scene::RigidBodyRecord& b : s.Bodies())
        if (!b.is_static) st.mass += b.mass;
    for (const scene::CollisionShapeRecord& sh : s.Shapes()) {
        const bool mesh = sh.type == scene::ShapeType::TriMesh || sh.type == scene::ShapeType::ConvexHull;
        if (sh.contype == 0u && sh.conaffinity == 0u) {
            if (!sh.mesh_vertices.empty()) ++st.visual_meshes;
        } else {
            ++st.colliders;
            if (mesh) ++st.mesh_colliders;
        }
    }
    if (!SceneBounds(s, &st.lo, &st.hi)) st.lo = st.hi = math::Vec3{};
    return st;
}

json::Value Vec(const math::Vec3& v) {
    json::Value a = json::Value::Array();
    for (const float x : {v.x, v.y, v.z}) a.PushBack(json::Value::Float(x));
    return a;
}

std::string Stem(const std::string& path) { return PathText(Utf8Path(path).stem()); }

// A root-level node name: no path separators, and not one a sibling already has.
std::string UniqueRootName(const scene::SceneIR& doc, std::string base) {
    std::replace(base.begin(), base.end(), '/', '_');
    std::replace(base.begin(), base.end(), '\\', '_');
    if (base.empty()) base = "asset";
    const auto taken = [&](const std::string& n) {
        for (auto c = doc.Tree().Root()->first_child; c; c = c->next_sibling)
            if (c->name == n) return true;
        return false;
    };
    std::string name = base;
    for (int i = 2; taken(name); ++i) name = base + "_" + std::to_string(i);
    return name;
}

}  // namespace

math::Transform BodyWorld(const scene::SceneIR& s, scene::BodyId b) {
    for (const scene::JointRecord& j : s.Joints()) {
        if (j.child_body != b || j.parent_body == scene::kInvalidBody) continue;
        math::Transform motion = math::Transform::Identity();
        if (j.type == scene::JointType::Revolute) motion.rotation = math::Quat::FromAxisAngle(j.axis, j.initial_position);
        if (j.type == scene::JointType::Prismatic) motion.position = j.axis * j.initial_position;
        return BodyWorld(s, j.parent_body) * j.parent_frame * s.GetBody(b).local_transform * motion;
    }
    math::Transform t = s.GetBody(b).local_transform;
    for (scene::BodyId p = s.GetBody(b).parent_id; p != scene::kInvalidBody; p = s.GetBody(p).parent_id)
        t = s.GetBody(p).local_transform * t;
    return t;
}

AssetKind AssetKindOf(const std::string& path) {
    const std::string ext = Lower(PathText(Utf8Path(path).extension()));
    if (ext == ".nks") return AssetKind::Scene;
    if (ext == ".urdf") return AssetKind::Urdf;
    if (ext == ".usd" || ext == ".usda" || ext == ".usdc") return AssetKind::Usd;
    if (ext == ".stl" || ext == ".obj" || ext == ".dae" || ext == ".gltf" || ext == ".glb") return AssetKind::Mesh;
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".tga" || ext == ".bmp") return AssetKind::Image;
    if (ext != ".xml") return AssetKind::Unknown;
    const std::string root = RootElement(path);
    return root == "mujoco" ? AssetKind::Mjcf : root == "robot" ? AssetKind::Urdf : AssetKind::Unknown;
}

const char* AssetKindName(AssetKind kind) {
    switch (kind) {
        case AssetKind::Scene: return "scene";
        case AssetKind::Urdf: return "urdf";
        case AssetKind::Mjcf: return "mjcf";
        case AssetKind::Usd: return "usd";
        case AssetKind::Mesh: return "mesh";
        case AssetKind::Image: return "image";
        case AssetKind::Unknown: break;
    }
    return "unknown";
}

ImportOptions ImportOptionsOf(const json::Value& fields, const std::string& fallback) {
    ImportOptions o;
    const json::Value* profile = fields.Find("profile");
    o.profile = profile != nullptr ? profile->AsString() : fallback;
    if (const json::Value* fix = fields.Find("fix_base")) o.fix_base = fix->AsBool();
    if (const json::Value* paths = fields.Find("package_paths"))
        for (const json::Value& p : paths->Elements()) o.package_paths.push_back(p.AsString());
    return o;
}

void SetImportOptions(json::Value& fields, const ImportOptions& options) {
    fields.Set("profile", json::Value::Str(options.profile));
    fields.Set("fix_base", json::Value::Bool(options.fix_base));
    json::Value paths = json::Value::Array();
    for (const std::string& p : options.package_paths) paths.PushBack(json::Value::Str(p));
    fields.Set("package_paths", std::move(paths));
}

// Importers throw on files they cannot read; the failure becomes the analysis's one error.
ImportAnalysis AnalyzeImport(const std::string& path, const ImportOptions& options) {
    ImportAnalysis a;
    a.path = path;
    a.kind = AssetKindOf(path);
    try {
        switch (a.kind) {
            case AssetKind::Urdf: {
                const bool legacy = options.profile == "legacy";
                a.profile = legacy ? "legacy" : "strict";
                a.fix_base = options.fix_base;
                a.package_paths = options.package_paths;
                const import::UrdfImportProfile mode =
                    legacy ? import::UrdfImportProfile::LegacyCompatible : import::UrdfImportProfile::Strict;
                import::UrdfImportResult r =
                    import::ImportUrdf(path, mode, import::UrdfImportOptions{options.fix_base, options.package_paths});
                const std::vector<Owner> owners = Owners(path);
                // Legacy projects despite the schema's losses, so only its last diagnostic, the failure, blocks.
                for (const import::UrdfDiagnostic& d : r.diagnostics) {
                    const bool error = !r.scene && (!legacy || &d == &r.diagnostics.back());
                    a.diagnostics.push_back(ImportDiagnostic{error ? "error" : "warning", d.code, d.element,
                                                             OwnerOf(owners, d.line), d.message, d.line});
                }
                a.scene = std::move(r.scene);
                break;
            }
            case AssetKind::Mjcf: a.scene = import::LoadMjcf(path); break;
            case AssetKind::Usd: a.scene = import::LoadUsd(path); break;
            case AssetKind::Scene: a.scene = scene::nks::Load(path); break;
            case AssetKind::Mesh: a.scene = MeshAssetScene(path); break;
            case AssetKind::Image:
                a.diagnostics.push_back(
                    ImportDiagnostic{"error", "ASSET_IMAGE", "", "", "an image is a material's map, not a scene", 0});
                break;
            case AssetKind::Unknown:
                a.diagnostics.push_back(ImportDiagnostic{"error", "ASSET_UNKNOWN", "", "", "not a scene, robot or mesh file", 0});
                break;
        }
    } catch (const std::exception& e) {
        a.diagnostics.push_back(ImportDiagnostic{"error", "ASSET_LOAD_FAILED", "", "", e.what(), 0});
        a.scene.reset();
    }
    if (!a.scene) return a;
    // The world refuses a sensor mounted on no body; the records still load for a picture.
    for (const scene::SensorDesc& s : a.scene->Sensors())
        if (s.mount == scene::MountFrame::Body && s.joint_id == scene::kInvalidJoint && s.mount_index >= a.scene->RigidBodyCount())
            a.diagnostics.push_back(
                ImportDiagnostic{"error", "SENSOR_UNMOUNTED", "sensor", s.name, "the sensor names no body or site of the file", 0});
    a.stats = StatsOf(*a.scene);
    return a;
}

bool Importable(const ImportAnalysis& a) {
    return a.scene && std::none_of(a.diagnostics.begin(), a.diagnostics.end(),
                                   [](const ImportDiagnostic& d) { return d.severity == "error"; });
}

std::string ImportError(const ImportAnalysis& a) {
    for (const ImportDiagnostic& d : a.diagnostics)
        if (d.severity == "error") return "cannot import '" + a.path + "': " + d.message;
    return "cannot import '" + a.path + "'";
}

json::Value AnalysisToJson(const ImportAnalysis& a) {
    json::Value diagnostics = json::Value::Array();
    for (const ImportDiagnostic& d : a.diagnostics) {
        json::Value v = json::Value::Object();
        v.Set("severity", json::Value::Str(d.severity));
        v.Set("code", json::Value::Str(d.code));
        v.Set("element", json::Value::Str(d.element));
        v.Set("owner", json::Value::Str(d.owner));
        v.Set("message", json::Value::Str(d.message));
        v.Set("line", json::Value::Int(d.line));
        diagnostics.PushBack(std::move(v));
    }
    json::Value stats = json::Value::Object();
    stats.Set("links", json::Value::Int(a.stats.links));
    stats.Set("joints", json::Value::Int(a.stats.joints));
    stats.Set("colliders", json::Value::Int(a.stats.colliders));
    stats.Set("mesh_colliders", json::Value::Int(a.stats.mesh_colliders));
    stats.Set("visual_meshes", json::Value::Int(a.stats.visual_meshes));
    stats.Set("mass", json::Value::Float(a.stats.mass));
    stats.Set("lo", Vec(a.stats.lo));
    stats.Set("hi", Vec(a.stats.hi));
    json::Value v = json::Value::Object();
    v.Set("path", json::Value::Str(a.path));
    v.Set("kind", json::Value::Str(AssetKindName(a.kind)));
    SetImportOptions(v, ImportOptions{a.profile, a.fix_base, a.package_paths});
    v.Set("ok", json::Value::Bool(Importable(a)));
    v.Set("loaded", json::Value::Bool(a.scene.has_value()));
    v.Set("name", json::Value::Str(Stem(a.path)));
    v.Set("diagnostics", std::move(diagnostics));
    v.Set("stats", std::move(stats));
    return v;
}

scene::SceneIR MeshAssetScene(const std::string& path) {
    import::MeshGeometry mesh = import::LoadMeshFile(path);
    const import::cooker::ConvexPiece hull =
        import::cooker::ComputeConvexHull(mesh.vertices.data(), static_cast<uint32_t>(mesh.VertexCount()), kHullVertices);
    const std::string name = Stem(path);
    const math::Vec3 lo{hull.aabb_min[0], hull.aabb_min[1], hull.aabb_min[2]};
    const math::Vec3 hi{hull.aabb_max[0], hull.aabb_max[1], hull.aabb_max[2]};
    const math::Vec3 size = hi - lo;

    scene::SceneIR s;
    scene::RigidBodyRecord b;
    b.name = name;
    b.mass = kWaterDensity * hull.volume;
    b.inertia = math::Vec3{b.mass / 12.0f * (size.y * size.y + size.z * size.z),
                           b.mass / 12.0f * (size.x * size.x + size.z * size.z),
                           b.mass / 12.0f * (size.x * size.x + size.y * size.y)};
    b.inertial_transform.position = (lo + hi) * 0.5f;
    // A flat mesh has no hull, so it stays a static visual with nothing to collide.
    b.is_static = hull.indices.empty();
    const scene::BodyId body = s.AddRigidBody(b);

    scene::MaterialRecord m;
    m.name = name + "_mat";
    m.base_color = math::Vec3{0.72f, 0.72f, 0.75f};
    const scene::MaterialId mid = s.AddMaterial(m);

    const auto add_visual = [&](import::MeshGeometry&& piece, scene::MaterialId material, const std::string& vis_name) {
        scene::CollisionShapeRecord vis;
        vis.body_id = body;
        vis.type = scene::ShapeType::TriMesh;
        vis.material_id = material;
        vis.name = vis_name;
        vis.contype = 0u;
        vis.conaffinity = 0u;
        vis.mesh_vertices = std::move(piece.vertices);
        vis.mesh_indices = std::move(piece.indices);
        vis.mesh_normals = std::move(piece.normals);
        vis.mesh_uvs = std::move(piece.uvs);
        s.AddCollisionShape(std::move(vis));
    };
    // A mesh file's material runs keep their materials, one visual per run.
    if (mesh.parts.empty()) {
        add_visual(std::move(mesh), mid, name + "_visual");
    } else {
        std::unordered_map<std::string, scene::MaterialId> colors;
        for (size_t p = 0; p < mesh.parts.size(); ++p) {
            const import::MeshPart& part = mesh.parts[p];
            scene::MaterialId material = mid;
            if (part.has_color) {
                const auto [it, fresh] = colors.emplace(part.material, mid);
                if (fresh) it->second = s.AddMaterial(import::PartMaterial(part, name + "/" + part.material));
                material = it->second;
            }
            add_visual(import::ExtractPart(mesh, p), material, name + "_visual_" + std::to_string(p));
        }
    }

    if (b.is_static) return s;
    scene::CollisionShapeRecord col;
    col.body_id = body;
    col.type = scene::ShapeType::ConvexHull;
    col.material_id = mid;
    col.name = name + "_hull";
    col.mesh_vertices = hull.vertices;
    col.mesh_indices = hull.indices;
    s.AddCollisionShape(std::move(col));
    return s;
}

bool SceneBounds(const scene::SceneIR& scene, math::Vec3* lo, math::Vec3* hi) {
    return BoundsUnder(scene, math::Transform::Identity(), lo, hi);
}

std::string ComposeAsset(scene::SceneIR& doc, const scene::SceneIR& addon, AssetKind kind, const std::string& name,
                         math::Transform placement, bool ground) {
    math::Vec3 lo, hi;
    // An asset that brings its own floor plane keeps its authored heights.
    const bool floored = std::any_of(addon.Shapes().begin(), addon.Shapes().end(), [](const auto& s) {
        return s.type == scene::ShapeType::Plane;
    });
    if (ground && !floored && BoundsUnder(addon, placement, &lo, &hi))
        placement.position.z += placement.position.z - lo.z;
    const std::string leaf = UniqueRootName(doc, name);
    const size_t first = doc.RigidBodyCount();
    const auto first_material = static_cast<scene::MaterialId>(doc.MaterialCount());
    doc = scene::Compose(doc, addon, placement);
    UniqueMaterialNames(doc, first_material);
    for (scene::BodyId b = static_cast<scene::BodyId>(first); b < doc.RigidBodyCount(); ++b) {
        if (doc.GetBody(b).parent_id != scene::kInvalidBody) continue;
        scene::RigidBodyRecord& rec = doc.GetBodyMut(b);
        rec.name = kind == AssetKind::Mesh ? leaf : leaf + "/" + rec.name;
    }
    return leaf;
}

}  // namespace nuka::runtime::app::viewer
