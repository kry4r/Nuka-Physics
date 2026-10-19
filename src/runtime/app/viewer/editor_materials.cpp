#include "runtime/app/viewer/editor_materials.hpp"

#include "render/render_world.hpp"
#include "scene/graph/scene_graph.hpp"

#include <cctype>
#include <cstdio>
#include <functional>
#include <initializer_list>
#include <unordered_set>

namespace nuka::runtime::app::viewer {
namespace {

namespace json = scene::json;

// `base` less a ".001" style suffix, then the first suffix `taken` leaves free.
std::string NextFreeName(std::string base, const std::function<bool(const std::string&)>& taken) {
    const size_t dot = base.rfind('.');
    const auto digit = [&](size_t k) { return std::isdigit(static_cast<unsigned char>(base[dot + k])) != 0; };
    if (dot != std::string::npos && dot + 4u == base.size() && digit(1u) && digit(2u) && digit(3u)) base.resize(dot);
    for (int i = 1;; ++i) {
        char suffix[16];
        std::snprintf(suffix, sizeof(suffix), ".%03d", i);
        if (!taken(base + suffix)) return base + suffix;
    }
}

json::Value Floats(std::initializer_list<float> values) {
    json::Value a = json::Value::Array();
    for (const float v : values) a.PushBack(json::Value::Float(v));
    return a;
}

}  // namespace

std::vector<MaterialPart> ShapesUnder(const EditorScene& es, const EntityRecordIndex& idx, const std::string& path,
                                      bool rendering) {
    std::unordered_set<scene::EntityId, scene::EntityIdHash> drawn;
    for (const render::RenderInstance& inst : es.sim->GetRenderWorld().instances) drawn.insert(inst.entity);
    std::vector<MaterialPart> parts;
    const scene::SceneGraph& tree = es.scene.Tree();
    tree.Traverse(tree.NodeOf(path), [&](const std::shared_ptr<scene::SceneNode>& node) {
        const auto it = idx.shape.find(node->entity);
        if (it != idx.shape.end() && (!rendering || drawn.count(node->entity) != 0u))
            parts.push_back(MaterialPart{tree.PathOf(node), it->second});
    });
    return parts;
}

scene::MaterialId FindMaterial(const scene::SceneIR& scene, const std::string& name) {
    for (const scene::MaterialRecord& m : scene.Materials())
        if (m.name == name) return m.id;
    return scene::kInvalidMaterial;
}

std::string UniqueMaterialName(const scene::SceneIR& scene, const std::string& base) {
    const auto taken = [&](const std::string& name) { return FindMaterial(scene, name) != scene::kInvalidMaterial; };
    return taken(base) ? NextFreeName(base, taken) : base;
}

void UniqueMaterialNames(scene::SceneIR& scene, scene::MaterialId first) {
    std::unordered_set<std::string> all, claimed;
    for (const scene::MaterialRecord& m : scene.Materials()) {
        all.insert(m.name);
        if (m.id < first) claimed.insert(m.name);
    }
    for (scene::MaterialId i = first; i < scene.MaterialCount(); ++i) {
        const std::string name = scene.GetMaterial(i).name;
        if (claimed.insert(name).second) continue;
        const std::string made = NextFreeName(name, [&](const std::string& n) { return all.count(n) != 0u; });
        all.insert(made);
        claimed.insert(made);
        scene.GetMaterialMut(i).name = made;
    }
}

uint32_t MaterialUsers(const scene::SceneIR& scene, scene::MaterialId id) {
    uint32_t users = 0u;
    for (const scene::CollisionShapeRecord& s : scene.Shapes()) users += s.material_id == id ? 1u : 0u;
    for (const scene::MediaRecord& m : scene.Media()) {
        users += m.render_material_id == id ? 1u : 0u;
        users += m.cable_line.slab.render_material_id == id ? 1u : 0u;
        for (const scene::MediaRecord::MpmFill& fill : m.mpm_fills) users += fill.render_material_id == id ? 1u : 0u;
    }
    return users;
}

MaterialBinding BindMaterial(const EditorScene& es, scene::MaterialId id) {
    MaterialBinding bind;
    bind.valid = true;
    bind.material_id = id;
    // A material made since the world was built has no slot yet; the default one follows the last.
    const uint32_t slot = es.scene.Ecs().FindRenderMaterial(es.scene.GetMaterial(id).name);
    if (slot < es.sim->GetRenderWorld().default_material_id) bind.render_slot = slot;
    return bind;
}

scene::RenderMaterial BoundMaterial(const EditorScene& es, const MaterialBinding& bind) {
    const render::RenderWorld& rw = es.sim->GetRenderWorld();
    if (bind.render_slot != render::kNoId && bind.render_slot < rw.MaterialCount())
        return rw.materials[bind.render_slot];
    return scene::RenderMaterialOf(es.scene.GetMaterial(bind.material_id));
}

json::Value MaterialPreset(const std::string& preset, const std::string& textures) {
    json::Value v = json::Value::Object();
    const auto set = [&](std::initializer_list<float> color, float roughness, float metallic, float friction) {
        v.Set("base_color", Floats(color));
        v.Set("roughness", json::Value::Float(roughness));
        v.Set("metallic", json::Value::Float(metallic));
        v.Set("friction", json::Value::Float(friction));
    };
    if (preset == "plastic") {
        set({0.80f, 0.80f, 0.80f, 1.0f}, 0.45f, 0.0f, 0.6f);
    } else if (preset == "metal") {
        set({0.91f, 0.92f, 0.92f, 1.0f}, 0.25f, 1.0f, 0.4f);
    } else if (preset == "rubber") {
        set({0.06f, 0.06f, 0.06f, 1.0f}, 0.9f, 0.0f, 1.2f);
    } else if (preset == "glass") {
        set({0.95f, 0.97f, 1.0f, 1.0f}, 0.05f, 0.0f, 0.4f);
        v.Set("transmission", json::Value::Float(1.0f));
        v.Set("ior", json::Value::Float(1.5f));
    } else if (preset == "emissive") {
        set({0.05f, 0.05f, 0.05f, 1.0f}, 0.5f, 0.0f, 0.6f);
        v.Set("emissive", Floats({2.0f, 1.2f, 0.4f}));
    } else if (preset == "textured") {
        set({1.0f, 1.0f, 1.0f, 1.0f}, 1.0f, 0.0f, 0.8f);
        v.Set("albedo_map", json::Value::Str(textures + "/grid_floor_albedo.png"));
        v.Set("roughness_map", json::Value::Str(textures + "/grid_floor_roughness.png"));
        v.Set("triplanar", json::Value::Bool(true));
    } else {
        return json::Value::Null();
    }
    return v;
}

}  // namespace nuka::runtime::app::viewer
