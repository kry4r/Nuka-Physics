#include "runtime/app/viewer/editor_document.hpp"

#include "render/render_world.hpp"
#include "runtime/app/viewer/editor_collision.hpp"
#include "runtime/app/viewer/editor_import.hpp"
#include "runtime/app/viewer/editor_materials.hpp"
#include "runtime/app/viewer/editor_sensors.hpp"
#include "runtime/app/viewer/robot_tools.hpp"
#include "scene/scene_fields.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace nuka::runtime::app::viewer {
namespace {

namespace json = scene::json;

constexpr uint32_t kNoParam = ~0u;

bool IsStructural(const std::string& field) {
    return field == "spawn" || field == "delete" || field == "reparent" || field == "script" || field == "import" ||
           field == "reimport" || field == "collision" || field == "static" || field == "pin" || field == "drive" ||
           field == "initial_position" || field == "attach" || field == "sensor" || field == "material_new" ||
           field == "material_rename" || field == "material_delete" || field == "material_assign" ||
           field == "material_unique" || field == "material_friction" || field == "terrain";
}

// Moves the import sources recorded at node `from` or under it to `to`; an empty `to` drops them.
void MoveSources(scene::SceneIR& scene, const std::string& from, const std::string& to) {
    std::vector<scene::SourceRecord>& sources = scene.SourcesMut();
    for (auto it = sources.begin(); it != sources.end();) {
        const bool under = !from.empty() && (it->node == from || it->node.rfind(from + "/", 0) == 0);
        if (under && to.empty()) {
            it = sources.erase(it);
            continue;
        }
        if (under) it->node = to + it->node.substr(from.size());
        ++it;
    }
}

json::Value Floats(const float* v, size_t n) {
    json::Value a = json::Value::Array();
    for (size_t i = 0; i < n; ++i) a.PushBack(json::Value::Float(v[i]));
    return a;
}

void ReadFloats(const json::Value& a, float* v, size_t n) {
    if (a.Size() != n) throw std::runtime_error("expected " + std::to_string(n) + " numbers");
    for (size_t i = 0; i < n; ++i) v[i] = a.Elements()[i].AsFloat();
}

// Overwrites the material fields the object names and keeps the rest.
void MaterialFromJson(const json::Value& v, scene::RenderMaterial* m) {
    if (const json::Value* f = v.Find("base_color")) ReadFloats(*f, m->base_color, 4u);
    if (const json::Value* f = v.Find("metallic")) m->metallic = f->AsFloat();
    if (const json::Value* f = v.Find("roughness")) m->roughness = f->AsFloat();
    if (const json::Value* f = v.Find("emissive")) ReadFloats(*f, m->emissive, 3u);
    if (const json::Value* f = v.Find("opacity")) m->opacity = f->AsFloat();
    if (const json::Value* f = v.Find("sheen")) m->sheen = f->AsFloat();
    if (const json::Value* f = v.Find("transmission")) m->transmission = f->AsFloat();
    if (const json::Value* f = v.Find("ior")) m->ior = f->AsFloat();
    if (const json::Value* f = v.Find("absorption")) ReadFloats(*f, m->absorption, 3u);
    for (size_t k = 0; k < 5u; ++k)
        if (const json::Value* f = v.Find(kMaterialMaps[k])) m->*kMaterialMapFields[k] = f->AsString();
    const auto channel = [&](const char* key, uint32_t* out) {
        const json::Value* f = v.Find(key);
        if (f == nullptr) return;
        if (f->AsInt() < 0 || f->AsInt() > 2) throw std::runtime_error("a channel is 0 (R), 1 (G) or 2 (B)");
        *out = static_cast<uint32_t>(f->AsInt());
    };
    channel("roughness_channel", &m->roughness_channel);
    channel("metallic_channel", &m->metallic_channel);
    if (const json::Value* f = v.Find("uv_scale")) m->uv_scale = f->AsFloat();
    if (const json::Value* f = v.Find("triplanar")) m->triplanar = f->AsBool();
}

// The material `after.material` names, as `error` says when there is none.
scene::MaterialId NamedMaterial(const scene::SceneIR& scene, const json::Value& after, std::string* error) {
    const std::string& name = after.At("material").AsString();
    const scene::MaterialId id = FindMaterial(scene, name);
    if (id == scene::kInvalidMaterial) *error = "no material '" + name + "'";
    return id;
}

std::string MaterialName(const scene::SceneIR& scene, scene::MaterialId id) {
    return id == scene::kInvalidMaterial ? std::string() : scene.GetMaterial(id).name;
}

json::Value Obj(std::initializer_list<std::pair<const char*, json::Value>> items) {
    json::Value v = json::Value::Object();
    for (const auto& [key, value] : items) v.Set(key, value);
    return v;
}

// `v` with `key` set to `value` in place of what it had.
json::Value Replaced(const json::Value& v, const std::string& key, json::Value value) {
    json::Value out = json::Value::Object();
    for (const auto& [k, item] : v.Items())
        if (k != key) out.Set(k, item);
    out.Set(key, std::move(value));
    return out;
}

std::string Leaf(const std::string& path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1u);
}

// The first of `roots` whose joints lead to the body named `mount` (a path or a leaf), else the first root
// when `mount` is empty; false when none does.
bool FindMount(const scene::SceneIR& scene, const std::vector<scene::BodyId>& roots, const std::string& mount,
               scene::BodyId* root, scene::BodyId* at) {
    for (const scene::BodyId r : roots) {
        for (const scene::BodyId b : JointSubtree(scene, r)) {
            const std::string& name = scene.GetBody(b).name;
            if (!mount.empty() && name != mount && Leaf(name) != mount) continue;
            *root = r;
            *at = mount.empty() ? r : b;
            return true;
        }
    }
    return false;
}

bool HasParentJoint(const scene::SceneIR& scene, scene::BodyId body) {
    for (const scene::JointRecord& j : scene.Joints())
        if (j.child_body == body) return true;
    return false;
}

}  // namespace

json::Value TxnToJson(const EditTxn& txn) {
    json::Value ops = json::Value::Array();
    for (const EditOp& op : txn.ops) {
        json::Value v = json::Value::Object();
        v.Set("target", json::Value::Str(op.target));
        v.Set("field", json::Value::Str(op.field));
        v.Set("before", op.before);
        v.Set("after", op.after);
        ops.PushBack(std::move(v));
    }
    json::Value v = json::Value::Object();
    v.Set("label", json::Value::Str(txn.label));
    v.Set("ops", std::move(ops));
    return v;
}

EditTxn TxnFromJson(const json::Value& v) {
    EditTxn txn;
    if (const json::Value* label = v.Find("label")) txn.label = label->AsString();
    for (const json::Value& o : v.At("ops").Elements()) {
        EditOp op;
        if (const json::Value* target = o.Find("target")) op.target = target->AsString();
        op.field = o.At("field").AsString();
        if (const json::Value* before = o.Find("before")) op.before = *before;
        if (const json::Value* after = o.Find("after")) op.after = *after;
        txn.ops.push_back(std::move(op));
    }
    return txn;
}

json::Value TransformToJson(const math::Transform& t) {
    const float p[3] = {t.position.x, t.position.y, t.position.z};
    const float q[4] = {t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z};
    json::Value v = json::Value::Object();
    v.Set("pos", Floats(p, 3u));
    v.Set("quat", Floats(q, 4u));
    return v;
}

math::Transform TransformFromJson(const json::Value& v) {
    float p[3];
    float q[4];
    ReadFloats(v.At("pos"), p, 3u);
    ReadFloats(v.At("quat"), q, 4u);
    math::Transform t;
    t.position = math::Vec3{p[0], p[1], p[2]};
    t.rotation = math::Quat(q[0], q[1], q[2], q[3]);
    return t;
}

json::Value MaterialToJson(const scene::RenderMaterial& m) {
    json::Value v = json::Value::Object();
    v.Set("base_color", Floats(m.base_color, 4u));
    v.Set("metallic", json::Value::Float(m.metallic));
    v.Set("roughness", json::Value::Float(m.roughness));
    v.Set("emissive", Floats(m.emissive, 3u));
    v.Set("opacity", json::Value::Float(m.opacity));
    v.Set("sheen", json::Value::Float(m.sheen));
    v.Set("transmission", json::Value::Float(m.transmission));
    v.Set("ior", json::Value::Float(m.ior));
    v.Set("absorption", Floats(m.absorption, 3u));
    for (size_t k = 0; k < 5u; ++k) v.Set(kMaterialMaps[k], json::Value::Str(m.*kMaterialMapFields[k]));
    v.Set("roughness_channel", json::Value::Int(m.roughness_channel));
    v.Set("metallic_channel", json::Value::Int(m.metallic_channel));
    v.Set("uv_scale", json::Value::Float(m.uv_scale));
    v.Set("triplanar", json::Value::Bool(m.triplanar));
    return v;
}

bool ParsePrimitiveKind(const std::string& s, PrimitiveKind* out) {
    for (const PrimitiveKind kind : {PrimitiveKind::Box, PrimitiveKind::Sphere, PrimitiveKind::Capsule,
                                     PrimitiveKind::Plane, PrimitiveKind::Cylinder}) {
        if (s == PrimitiveKindName(kind)) {
            *out = kind;
            return true;
        }
    }
    return false;
}

const char* PrimitiveKindName(PrimitiveKind kind) {
    switch (kind) {
        case PrimitiveKind::Box: return "box";
        case PrimitiveKind::Sphere: return "sphere";
        case PrimitiveKind::Capsule: return "capsule";
        case PrimitiveKind::Plane: return "plane";
        case PrimitiveKind::Cylinder: return "cylinder";
    }
    return "";
}

namespace {

// What a terrain call leaves out: g1_rough's bump field and the tiled field's steps.
constexpr float kTerrainSize = 20.0f, kTerrainCell = 0.1f, kBumpHeight = 0.05f, kBumpCell = 0.4f, kGrade = 0.1f;
constexpr float kStepRise = 0.08f, kStepWidth = 0.3f, kPlatform = 1.0f, kTile = 5.0f, kTileMargin = 0.6f;
constexpr int64_t kSteps = 8;
constexpr uint32_t kMaxTerrainCells = 4096u;

std::string TerrainKind(const scene::TerrainRecord& t) {
    if (!t.image_path.empty()) return "image";
    if (t.feature_cell > 0.0f) return "mixed";
    if (t.ring_rise != 0.0f) return "stairs";
    if (t.grade_x != 0.0f || t.grade_y != 0.0f) return "slope";
    return t.bump_height > 0.0f ? "bumps" : "flat";
}

}  // namespace

bool TerrainFromArgs(const json::Value& args, scene::TerrainRecord* out, std::string* error) {
    const json::Value* kind_arg = args.Find("kind");
    const std::string kind = kind_arg != nullptr ? kind_arg->AsString() : std::string();
    if (kind != "flat" && kind != "bumps" && kind != "slope" && kind != "stairs" && kind != "mixed") {
        *error = "the terrain kind is flat, bumps, slope, stairs or mixed";
        return false;
    }
    const auto num = [&](const char* key, float fallback) {
        const json::Value* v = args.Find(key);
        return v != nullptr ? v->AsFloat() : fallback;
    };
    const auto count = [&](const char* key, int64_t fallback) {
        const json::Value* v = args.Find(key);
        return v != nullptr ? v->AsInt() : fallback;
    };
    float size[2] = {kTerrainSize, kTerrainSize}, center[2] = {0.0f, 0.0f};
    if (const json::Value* v = args.Find("size")) ReadFloats(*v, size, 2u);
    if (const json::Value* v = args.Find("center")) ReadFloats(*v, center, 2u);
    const float cell = num("cell", kTerrainCell);
    const float cells[2] = {size[0] / cell, size[1] / cell};
    if (!(cell >= 0.02f) || !(cells[0] >= 4.0f && cells[1] >= 4.0f) ||
        !(cells[0] <= kMaxTerrainCells && cells[1] <= kMaxTerrainCells)) {
        *error = "a terrain side spans 4 to 4096 cells of 0.02 m or more";
        return false;
    }
    scene::TerrainRecord t;
    t.name = "terrain";
    t.cell = cell;
    t.ncol = static_cast<uint32_t>(std::lround(cells[0])) + 1u;
    t.nrow = static_cast<uint32_t>(std::lround(cells[1])) + 1u;
    t.origin = math::Vec3{center[0] - 0.5f * cell * static_cast<float>(t.ncol - 1u),
                          center[1] - 0.5f * cell * static_cast<float>(t.nrow - 1u), 0.0f};
    t.base_z = num("base", 0.0f);
    if (kind == "bumps" || kind == "mixed") {
        t.bump_height = num("height", kBumpHeight);
        t.bump_cell = num("bump", kBumpCell);
        if (!(t.bump_height > 0.0f) || !(t.bump_cell >= cell)) {
            *error = "bumps need a height above 0 and a footprint of a cell or more";
            return false;
        }
    }
    if (kind == "slope") {
        t.grade_x = num("grade", kGrade);
        if (t.grade_x == 0.0f || !std::isfinite(t.grade_x)) {
            *error = "a slope needs a grade other than 0";
            return false;
        }
    }
    if (kind == "stairs" || kind == "mixed") {
        t.ring_rise = num("rise", kStepRise);
        t.ring_width = num("width", kStepWidth);
        t.ring_platform = num("platform", kPlatform);
        const int64_t steps = count("count", kSteps);
        if (t.ring_rise == 0.0f || !std::isfinite(t.ring_rise) || !(t.ring_width >= cell) ||
            !(t.ring_platform >= 0.0f) || steps < 1 || steps > 64) {
            *error = "stairs need a rise other than 0 (below 0 digs a pit), steps a cell wide or more, and 1 to 64 "
                     "of them";
            return false;
        }
        t.ring_count = static_cast<uint32_t>(steps);
    }
    if (kind == "mixed") {
        t.feature_cell = num("tile", kTile);
        t.feature_margin = num("margin", kTileMargin);
        const int64_t seed = count("seed", 0);
        if (!(t.feature_margin >= 0.0f) || !(t.feature_cell > 2.0f * t.feature_margin) || seed < 0) {
            *error = "a mixed field's tiles are wider than twice their rim, and its seed is 0 or more";
            return false;
        }
        t.feature_seed = static_cast<uint32_t>(seed);
    }
    *out = std::move(t);
    return true;
}

json::Value TerrainArgs(const scene::TerrainRecord& t) {
    const std::string kind = TerrainKind(t);
    const bool bumps = kind == "bumps" || kind == "mixed", stairs = kind == "stairs" || kind == "mixed";
    const bool mixed = kind == "mixed";
    const float size[2] = {t.cell * static_cast<float>(std::max(t.ncol, 1u) - 1u),
                           t.cell * static_cast<float>(std::max(t.nrow, 1u) - 1u)};
    const float center[2] = {t.origin.x + 0.5f * size[0], t.origin.y + 0.5f * size[1]};
    json::Value v = Obj({{"kind", json::Value::Str(kind)},
                         {"size", Floats(size, 2u)},
                         {"center", Floats(center, 2u)},
                         {"cell", json::Value::Float(t.cell)},
                         {"base", json::Value::Float(t.base_z)},
                         {"height", json::Value::Float(bumps ? t.bump_height : kBumpHeight)},
                         {"bump", json::Value::Float(bumps ? t.bump_cell : kBumpCell)},
                         {"grade", json::Value::Float(kind == "slope" ? t.grade_x : kGrade)},
                         {"rise", json::Value::Float(stairs ? t.ring_rise : kStepRise)},
                         {"width", json::Value::Float(stairs ? t.ring_width : kStepWidth)},
                         {"platform", json::Value::Float(stairs ? t.ring_platform : kPlatform)},
                         {"count", json::Value::Int(stairs ? static_cast<int64_t>(t.ring_count) : kSteps)},
                         {"tile", json::Value::Float(mixed ? t.feature_cell : kTile)},
                         {"margin", json::Value::Float(mixed ? t.feature_margin : kTileMargin)},
                         {"seed", json::Value::Int(mixed ? static_cast<int64_t>(t.feature_seed) : 0)}});
    if (kind == "image") v.Set("image", json::Value::Str(t.image_path));
    return v;
}

void EditorDocument::Attach(EditorScene* es, DocumentHost* host) {
    es_ = es;
    host_ = host;
    records_.Build(es_->scene);
    undo_.clear();
    redo_.clear();
    branches_.clear();
    saved_ = 0u;
    saved_scene_ = es_->scene;
    ++revision_;
}

void EditorDocument::Detach() {
    es_ = nullptr;
    records_ = EntityRecordIndex{};
    undo_.clear();
    redo_.clear();
    branches_.clear();
    saved_ = 0u;
    saved_scene_ = scene::SceneIR{};
    ++revision_;
}

void EditorDocument::MarkSaved() {
    saved_ = StateId();
    saved_scene_ = es_->scene;
}

void EditorDocument::SetParams(std::vector<nuka_param_info_t> infos, std::vector<double> world_values) {
    infos_ = std::move(infos);
    world_values_ = std::move(world_values);
}

scene::EntityId EditorDocument::EntityAt(const std::string& path) const {
    if (const auto node = es_->scene.Tree().NodeOf(path)) return node->entity;
    return scene::kInvalidEntity;
}

// The registry parameter named by its key or by the scene field it writes.
uint32_t EditorDocument::ParamOf(const std::string& field) const {
    for (uint32_t i = 0; i < infos_.size(); ++i)
        if (field == infos_[i].key || (infos_[i].scene_field[0] != '\0' && field == infos_[i].scene_field))
            return i;
    return kNoParam;
}

// A live parameter reaches the world without a rebuild when its target is a link body.
bool EditorDocument::LiveElement(const std::string& target, uint32_t param, uint32_t* element) const {
    if (param == kNoParam || infos_[param].tier != NUKA_PARAM_LIVE) return false;
    const auto it = records_.body.find(EntityAt(target));
    if (it == records_.body.end()) return false;
    const WorldLayout& layout = es_->Layout();
    for (uint32_t i = 0; i < layout.links_per_env; ++i) {
        if (layout.links[i].body_index == it->second) {
            *element = i;
            return true;
        }
    }
    return false;
}

bool EditorDocument::NeedsSnapshot(const EditOp& op) const {
    if (IsStructural(op.field)) return true;
    if (op.field == "transform" || op.field == "material" || op.field == "name") return false;
    const uint32_t p = ParamOf(op.field);
    if (p != kNoParam && infos_[p].scene_field[0] == '\0') return false;
    uint32_t element = 0u;
    return !LiveElement(op.target, p, &element);
}

bool EditorDocument::LiveField(const std::string& target, const std::string& field) const {
    const uint32_t p = ParamOf(field);
    if (p != kNoParam && infos_[p].scene_field[0] == '\0') return infos_[p].tier == NUKA_PARAM_LIVE;
    uint32_t element = 0u;
    return LiveElement(target, p, &element);
}

bool EditorDocument::ApplyOp(const EditOp& op, EditOp* done, EditOp* inverse, OpOutcome* out, std::string* error) {
    *done = op;
    const scene::EntityId e = EntityAt(op.target);
    const bool needs_node = !op.target.empty() || op.field == "transform" || op.field == "name" ||
                            op.field == "delete" || op.field == "reparent" || op.field == "static" ||
                            op.field == "pin" || op.field == "drive" || op.field == "initial_position" ||
                            op.field == "material_assign" || op.field == "material_unique";
    if (needs_node && e == scene::kInvalidEntity) {
        *error = "no node at '" + op.target + "'";
        return false;
    }
    if (op.field == "transform") {
        if (done->before.IsNull()) {
            const render::RenderInstance* inst = InstanceOfEntity(es_->sim->GetRenderWorld(), e);
            done->before = TransformToJson(inst != nullptr ? inst->world_xform : AuthoredWorldTransform(es_->scene, e));
        }
        ApplyTransformEdit(*es_, records_, e, TransformFromJson(op.after), /*commit=*/true);
    } else if (op.field == "material") {
        // The target node's material, or without a target the one `after.material` names.
        MaterialBinding mb;
        if (op.target.empty()) {
            const scene::MaterialId id = NamedMaterial(es_->scene, op.after, error);
            if (id == scene::kInvalidMaterial) return false;
            mb = BindMaterial(*es_, id);
        } else {
            mb = ResolveMaterial(*es_, records_, e);
        }
        if (!mb.valid) {
            *error = "'" + op.target + "' has no material";
            return false;
        }
        scene::RenderMaterial m = BoundMaterial(*es_, mb);
        if (done->before.IsNull()) done->before = MaterialToJson(m);
        MaterialFromJson(op.after, &m);
        done->after = MaterialToJson(m);
        if (op.target.empty()) {
            done->before = Replaced(done->before, "material", op.after.At("material"));
            done->after.Set("material", op.after.At("material"));
        }
        ApplyMaterialEdit(*es_, mb, m, /*commit=*/true);
    } else if (op.field == "material_new") {
        // A copy of `after.from`, else a fresh material, with `after.fields` over it; the name is made unique.
        scene::MaterialRecord rec;
        rec.name = "Material";
        if (const json::Value* from = op.after.Find("from")) {
            const scene::MaterialId src = FindMaterial(es_->scene, from->AsString());
            if (src == scene::kInvalidMaterial) {
                *error = "no material '" + from->AsString() + "'";
                return false;
            }
            rec = es_->scene.GetMaterial(src);
        }
        if (const json::Value* name = op.after.Find("name"); name != nullptr && !name->AsString().empty())
            rec.name = name->AsString();
        rec.name = UniqueMaterialName(es_->scene, rec.name);
        if (const json::Value* fields = op.after.Find("fields")) {
            scene::RenderMaterial m = scene::RenderMaterialOf(rec);
            MaterialFromJson(*fields, &m);
            WriteMaterialRecord(m, &rec);
            if (const json::Value* friction = fields->Find("friction")) rec.friction_mu = friction->AsFloat();
        }
        es_->scene.AddMaterial(rec);
        done->after = Replaced(op.after, "name", json::Value::Str(rec.name));
    } else if (op.field == "material_rename") {
        const scene::MaterialId id = NamedMaterial(es_->scene, op.after, error);
        if (id == scene::kInvalidMaterial) return false;
        const std::string from = es_->scene.GetMaterial(id).name, to = op.after.At("name").AsString();
        if (to.empty() || (to != from && FindMaterial(es_->scene, to) != scene::kInvalidMaterial)) {
            *error = to.empty() ? "a material needs a name" : "a material is named '" + to + "' already";
            return false;
        }
        es_->scene.GetMaterialMut(id).name = to;
        done->before = Obj({{"material", json::Value::Str(to)}, {"name", json::Value::Str(from)}});
    } else if (op.field == "material_delete") {
        // What used the material takes `after.replace` ("" for the default one); one in use needs it.
        const scene::MaterialId id = NamedMaterial(es_->scene, op.after, error);
        if (id == scene::kInvalidMaterial) return false;
        const json::Value* replace = op.after.Find("replace");
        scene::MaterialId to = scene::kInvalidMaterial;
        if (replace != nullptr && !replace->AsString().empty()) {
            to = FindMaterial(es_->scene, replace->AsString());
            if (to == scene::kInvalidMaterial || to == id) {
                *error = to == id ? "a material cannot replace itself" : "no material '" + replace->AsString() + "'";
                return false;
            }
        }
        const uint32_t users = MaterialUsers(es_->scene, id);
        if (users > 0u && replace == nullptr) {
            const std::string count = users == 1u ? "a shape uses" : std::to_string(users) + " shapes use";
            *error = count + " '" + es_->scene.GetMaterial(id).name + "'; name the material they take instead";
            return false;
        }
        es_->scene.RemoveMaterial(id, to);
        out->rebuild = true;
    } else if (op.field == "material_assign") {
        // The shapes at or under the target take the material `after` names ("" for the default one).
        const std::string name = op.after.AsString();
        const scene::MaterialId id = name.empty() ? scene::kInvalidMaterial : FindMaterial(es_->scene, name);
        if (!name.empty() && id == scene::kInvalidMaterial) {
            *error = "no material '" + name + "'";
            return false;
        }
        const std::vector<MaterialPart> shapes = ShapesUnder(*es_, records_, op.target, /*rendering=*/false);
        if (shapes.empty()) {
            *error = "'" + op.target + "' has no shape";
            return false;
        }
        const scene::MaterialId was = es_->scene.GetShape(shapes.front().shape).material_id;
        done->before = json::Value::Str(MaterialName(es_->scene, was));
        for (const MaterialPart& s : shapes) es_->scene.GetShapeMut(s.shape).material_id = id;
        out->rebuild = true;
    } else if (op.field == "material_unique") {
        // The part takes a copy of its material, under the next free name, that no other shape shares.
        const auto it = records_.shape.find(e);
        if (it == records_.shape.end()) {
            *error = "'" + op.target + "' is not a part";
            return false;
        }
        const scene::MaterialId from = es_->scene.GetShape(it->second).material_id;
        if (from != scene::kInvalidMaterial && MaterialUsers(es_->scene, from) == 1u) {
            *error = "'" + op.target + "' has '" + MaterialName(es_->scene, from) + "' to itself already";
            return false;
        }
        scene::MaterialRecord rec;
        if (from != scene::kInvalidMaterial) {
            rec = es_->scene.GetMaterial(from);
        } else {
            // The default material's look.
            rec.name = "Material";
            rec.base_color = math::Vec3{0.8f, 0.8f, 0.8f};
        }
        rec.name = UniqueMaterialName(es_->scene, rec.name);
        es_->scene.GetShapeMut(it->second).material_id = es_->scene.AddMaterial(rec);
        done->before = json::Value::Str(MaterialName(es_->scene, from));
        done->after = json::Value::Str(rec.name);
        out->rebuild = true;
    } else if (op.field == "material_friction") {
        const scene::MaterialId id = NamedMaterial(es_->scene, op.after, error);
        if (id == scene::kInvalidMaterial) return false;
        const float friction = op.after.At("friction").AsFloat();
        if (!(friction >= 0.0f)) {
            *error = "friction is 0 or more";
            return false;
        }
        scene::MaterialRecord& rec = es_->scene.GetMaterialMut(id);
        done->before = Obj({{"material", op.after.At("material")}, {"friction", json::Value::Float(rec.friction_mu)}});
        rec.friction_mu = friction;
        out->rebuild = true;
    } else if (op.field == "name") {
        out->focus = RenameNode(*es_, op.target, op.after.AsString());
        if (out->focus.empty()) {
            *error = "cannot rename '" + op.target + "'";
            return false;
        }
        MoveSources(es_->scene, op.target, out->focus);
        done->before = json::Value::Str(Leaf(op.target));
        inverse->target = out->focus;
    } else if (op.field == "spawn") {
        PrimitiveKind kind;
        if (!ParsePrimitiveKind(op.after.At("kind").AsString(), &kind)) {
            *error = "unknown primitive '" + op.after.At("kind").AsString() + "'";
            return false;
        }
        math::Transform placement;
        if (const json::Value* pose = op.after.Find("pose")) placement = TransformFromJson(*pose);
        out->focus = SpawnPrimitive(*es_, kind, placement, op.target);
        if (out->focus.empty()) {
            *error = "cannot spawn under '" + op.target + "'";
            return false;
        }
        out->rebuild = true;
    } else if (op.field == "terrain") {
        // `after` describes the field the cook bakes (TerrainFromArgs); null removes it.
        const bool had = es_->scene.TerrainCount() > 0u;
        if (op.after.IsNull() && !had) {
            *error = "the scene has no terrain";
            return false;
        }
        done->before = had ? TerrainArgs(es_->scene.Terrain().front()) : json::Value::Null();
        if (op.after.IsNull()) {
            es_->scene.RemoveTerrain(0u);
        } else {
            scene::TerrainRecord t;
            if (!TerrainFromArgs(op.after, &t, error)) return false;
            if (had) es_->scene.GetTerrainMut(0u) = std::move(t);
            else es_->scene.AddTerrain(std::move(t));
        }
        out->rebuild = true;
    } else if (op.field == "import") {
        const ImportAnalysis a = AnalyzeImport(op.after.At("path").AsString(), ImportOptionsOf(op.after));
        if (!Importable(a)) {
            *error = ImportError(a);
            return false;
        }
        math::Transform placement;
        if (const json::Value* pose = op.after.Find("pose")) placement = TransformFromJson(*pose);
        const json::Value* ground = op.after.Find("ground");
        const bool rest = ground == nullptr ? DefaultGroundPlacement(a.kind) : ground->AsBool();
        out->focus = ComposeAsset(es_->scene, *a.scene, a.kind, op.after.At("name").AsString(), placement, rest,
                                  SourceOf(a, rest));
        out->rebuild = true;
    } else if (op.field == "reimport") {
        // The node's source imports again where the node's root stands; edits made inside the node are lost.
        const std::vector<scene::SourceRecord>& sources = es_->scene.Sources();
        const auto found = std::find_if(sources.begin(), sources.end(),
                                        [&](const scene::SourceRecord& s) { return s.node == op.target; });
        if (op.target.empty() || found == sources.end()) {
            *error = "'" + op.target + "' was not imported from a file";
            return false;
        }
        const scene::SourceRecord source = *found;
        const ImportAnalysis a = AnalyzeImport(source.file, OptionsOf(source));
        if (!Importable(a)) {
            *error = ImportError(a);
            return false;
        }
        // Cameras and lights the import set apart from its bodies stayed in the scene on their own; they don't
        // come in twice.
        scene::SceneIR addon = *a.scene;
        for (size_t i = addon.CameraCount(); i-- > 0;)
            if (addon.GetCamera(static_cast<scene::CameraId>(i)).attached_body == scene::kInvalidBody)
                addon.RemoveCamera(static_cast<scene::CameraId>(i));
        for (size_t i = addon.LightCount(); i-- > 0;)
            if (addon.GetLight(static_cast<scene::LightId>(i)).attached_body == scene::kInvalidBody)
                addon.RemoveLight(static_cast<scene::LightId>(i));
        const math::Transform placement = ReimportPlacement(es_->scene, addon, source);
        DeleteSubtree(*es_, op.target);
        MoveSources(es_->scene, op.target, "");
        out->focus = ComposeAsset(es_->scene, addon, a.kind, Leaf(op.target), placement, false, source);
        out->rebuild = true;
    } else if (op.field == "collision") {
        std::vector<std::string> paths;
        for (const json::Value& p : op.after.At("paths").Elements()) paths.push_back(p.AsString());
        if (GenerateCollision(es_->scene, paths, CollisionOptionsFromJson(op.after)) == 0u) {
            *error = "no selected body has a visual mesh to build a collider from";
            return false;
        }
        out->rebuild = true;
    } else if (op.field == "static") {
        const auto it = records_.body.find(e);
        if (it == records_.body.end()) {
            *error = "'" + op.target + "' is not a body";
            return false;
        }
        // A static root is a fixed base; the other links move with their joints.
        for (const nuka_link_binding_t& link : es_->Layout().links) {
            if (link.body_index == it->second && link.is_root == 0u) {
                *error = "only an articulation's root link can be static";
                return false;
            }
        }
        scene::RigidBodyRecord& body = es_->scene.GetBodyMut(it->second);
        done->before = json::Value::Bool(body.is_static);
        body.is_static = op.after.AsBool();
        out->rebuild = true;
    } else if (op.field == "drive") {
        const auto it = records_.body.find(e);
        const scene::JointRecord* joint = nullptr;
        for (const scene::JointRecord& j : es_->scene.Joints())
            if (it != records_.body.end() && j.child_body == it->second) joint = &j;
        scene::ActuatorType type = scene::ActuatorType::Position;
        if (joint == nullptr || !ParseActuatorType(op.after.At("type").AsString(), &type)) {
            *error = joint == nullptr ? "'" + op.target + "' is no joint's child link"
                                      : "the drive type is position, motor, velocity or force";
            return false;
        }
        scene::ActuatorId act = static_cast<scene::ActuatorId>(es_->scene.Actuators().size());
        for (const scene::ActuatorRecord& a : es_->scene.Actuators())
            if (a.joint_id == joint->id) act = a.id;
        if (act == es_->scene.Actuators().size()) {
            done->before = json::Value::Null();
            scene::ActuatorRecord rec;
            rec.name = joint->name;
            rec.joint_id = joint->id;
            act = es_->scene.AddActuator(rec);
        } else {
            const scene::ActuatorRecord& a = es_->scene.GetActuator(act);
            json::Value before = json::Value::Object();
            before.Set("type", json::Value::Str(ActuatorTypeName(a.type)));
            before.Set("kp", json::Value::Float(a.gain));
            done->before = std::move(before);
        }
        scene::ActuatorRecord& a = es_->scene.GetActuatorMut(act);
        a.type = type;
        if (const json::Value* kp = op.after.Find("kp")) a.gain = kp->AsFloat();
        out->rebuild = true;
    } else if (op.field == "initial_position") {
        // The joint's position at reset, in rad or m; the target is its child link.
        const auto it = records_.body.find(e);
        scene::JointId joint = scene::kInvalidJoint;
        for (const scene::JointRecord& j : es_->scene.Joints())
            if (it != records_.body.end() && j.child_body == it->second) joint = j.id;
        if (joint == scene::kInvalidJoint) {
            *error = "'" + op.target + "' is no joint's child link";
            return false;
        }
        scene::JointRecord& j = es_->scene.GetJointMut(joint);
        done->before = json::Value::Float(j.initial_position);
        j.initial_position = op.after.AsFloat();
        out->rebuild = true;
    } else if (op.field == "attach") {
        // The target body, or the root of the asset `after.asset` adds, hangs off the body `after.parent`.
        const std::string parent_path = op.after.At("parent").AsString();
        const auto parent = records_.body.find(EntityAt(parent_path));
        if (parent == records_.body.end()) {
            *error = "no body at '" + parent_path + "'";
            return false;
        }
        scene::JointType type = scene::JointType::Fixed;
        const json::Value* joint = op.after.Find("joint");
        if (joint != nullptr && (!ParseJointType(joint->AsString(), &type) ||
                                 (type != scene::JointType::Fixed && type != scene::JointType::Revolute &&
                                  type != scene::JointType::Prismatic))) {
            *error = "the joint is fixed, revolute or prismatic";
            return false;
        }
        float axis[3] = {0.0f, 0.0f, 1.0f};
        if (const json::Value* a = op.after.Find("axis")) ReadFloats(*a, axis, 3u);
        if (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2] < 1e-12f) {
            *error = "the joint axis is zero";
            return false;
        }
        const json::Value* mount_name = op.after.Find("mount");
        const std::string mount_path = mount_name != nullptr ? mount_name->AsString() : std::string();
        const json::Value* asset = op.after.Find("asset");
        std::optional<ImportAnalysis> analysis;
        const scene::SceneIR* source = &es_->scene;
        std::vector<scene::BodyId> roots;
        if (asset != nullptr) {
            analysis = AnalyzeImport(asset->At("path").AsString(), ImportOptionsOf(*asset));
            if (!Importable(*analysis)) {
                *error = ImportError(*analysis);
                return false;
            }
            source = &*analysis->scene;
            for (scene::BodyId b = 0; b < source->RigidBodyCount(); ++b)
                if (!HasParentJoint(*source, b)) roots.push_back(b);
        } else if (const auto it = records_.body.find(e); it != records_.body.end()) {
            roots.push_back(it->second);
        } else {
            *error = "'" + op.target + "' is not a body";
            return false;
        }
        scene::BodyId child = scene::kInvalidBody, mount = scene::kInvalidBody;
        if (!FindMount(*source, roots, mount_path, &child, &mount)) {
            *error = "no link '" + mount_path + "' in what is attached";
            return false;
        }
        if (asset == nullptr && HasParentJoint(es_->scene, child)) {
            *error = "'" + op.target + "' already hangs off a joint";
            return false;
        }
        if (asset == nullptr) {
            const std::vector<scene::BodyId> moved = JointSubtree(es_->scene, child);
            if (std::find(moved.begin(), moved.end(), parent->second) != moved.end()) {
                *error = "'" + parent_path + "' moves with '" + op.target + "'";
                return false;
            }
        }
        if (mount != child && type != scene::JointType::Fixed) {
            *error = "only a fixed joint mounts a link other than the attached root";
            return false;
        }
        if (asset != nullptr) {
            const scene::BodyId first = static_cast<scene::BodyId>(es_->scene.RigidBodyCount());
            ComposeAsset(es_->scene, *source, analysis->kind, asset->At("name").AsString(), math::Transform::Identity(),
                         false);
            child += first;
            mount += first;
        }
        std::optional<math::Transform> frame;
        if (const json::Value* f = op.after.Find("frame")) frame = TransformFromJson(*f);
        const json::Value* name = op.after.Find("name");
        const std::string made =
            AttachBody(es_->scene, child, parent->second, mount, type, math::Vec3{axis[0], axis[1], axis[2]},
                       frame ? &*frame : nullptr, name != nullptr ? name->AsString() : std::string());
        done->after.Set("name", json::Value::Str(made));
        out->focus = es_->scene.Tree().PathOf(es_->scene.Ecs().NodeOf(es_->scene.EntityOfBody(child)));
        out->rebuild = true;
    } else if (op.field == "pin") {
        const auto it = records_.media.find(e);
        if (it == records_.media.end() || es_->scene.GetMedia(it->second).kind != scene::MediaRecord::Kind::Cloth) {
            *error = "'" + op.target + "' is not a cloth";
            return false;
        }
        const uint32_t vertex = static_cast<uint32_t>(op.after.At("vertex").AsInt());
        uint32_t count = 0u;
        for (const ParticleSurfaceBinding& s : es_->Layout().surfaces)
            if (s.info.medium == it->second) count = s.info.particle_count;
        if (vertex >= count) {
            *error = "'" + op.target + "' has no vertex " + std::to_string(vertex);
            return false;
        }
        std::vector<uint32_t>& pins = es_->scene.GetMediaMut(it->second).cloth_mesh.pinned_vertices;
        const auto at = std::find(pins.begin(), pins.end(), vertex);
        json::Value before = json::Value::Object();
        before.Set("vertex", json::Value::Int(vertex));
        before.Set("on", json::Value::Bool(at != pins.end()));
        done->before = std::move(before);
        if (at != pins.end()) pins.erase(at);
        if (op.after.At("on").AsBool()) pins.push_back(vertex);
        out->rebuild = true;
    } else if (op.field == "sensor") {
        // `after.index` names the sensor `after.sensor` changes or `after.remove` drops; without it `after.sensor`
        // adds one. All cameras share one image size and all lidars one fan, so an edit of one carries to the rest.
        std::vector<scene::SensorDesc> sensors = es_->scene.Sensors();
        const size_t count = sensors.size();
        const json::Value* index = op.after.Find("index");
        const json::Value* remove = op.after.Find("remove");
        const bool removing = remove != nullptr && remove->AsBool();
        const size_t at = index != nullptr ? static_cast<size_t>(index->AsInt()) : count;
        if ((index != nullptr && (index->AsInt() < 0 || at >= count)) || (removing && index == nullptr)) {
            *error = index != nullptr ? "no sensor " + std::to_string(index->AsInt()) : "a removal names its sensor's index";
            return false;
        }
        if (at < count) done->before = SensorToJson(es_->scene, sensors[at]);
        if (removing) {
            sensors.erase(sensors.begin() + static_cast<std::ptrdiff_t>(at));
        } else {
            const json::Value* fields = op.after.Find("sensor");
            scene::SensorDesc s;
            if (at < count) {
                s = sensors[at];
            } else {
                s.mount = scene::MountFrame::World;
                s.mount_index = scene::kInvalidBody;
            }
            // A new sensor, or one that changes type, starts from its type's defaults.
            const json::Value* type = fields != nullptr ? fields->Find("type") : nullptr;
            scene::SensorType next = s.type;
            if (type != nullptr && ParseSensorType(type->AsString(), &next) && (at == count || next != s.type)) {
                s.type = next;
                NewSensorDefaults(sensors, &s);
            }
            if (!SensorFromJson(fields != nullptr ? *fields : json::Value::Object(), es_->scene, &s, error)) return false;
            if (s.name.empty()) s.name = UniqueSensorName(sensors, SensorTypeName(s.type));
            if (at < count) {
                ShareSensorShape(s, &sensors);
                sensors[at] = s;
            } else {
                sensors.push_back(s);
            }
        }
        if (const std::string shape = SensorShapeError(sensors); !shape.empty()) {
            *error = shape;
            return false;
        }
        if (removing) {
            es_->scene.RemoveSensor(static_cast<scene::SensorId>(at));
        } else if (at < count) {
            for (size_t i = 0; i < count; ++i) es_->scene.GetSensorMut(static_cast<scene::SensorId>(i)) = sensors[i];
        } else {
            es_->scene.AddSensor(sensors[at]);
            done->after.Set("index", json::Value::Int(static_cast<int64_t>(at)));
        }
        out->rebuild = true;
    } else if (op.field == "delete") {
        const size_t bodies = es_->scene.RigidBodyCount();
        const size_t shapes = es_->scene.ShapeCount();
        const size_t scripts = es_->scene.ScriptCount();
        out->focus = DeleteSubtree(*es_, op.target);
        const bool body_deleted = es_->scene.RigidBodyCount() < bodies;
        const bool shape_deleted = es_->scene.ShapeCount() < shapes;
        if (!body_deleted && !shape_deleted && es_->scene.ScriptCount() >= scripts) {
            *error = "nothing to delete at '" + op.target + "'";
            return false;
        }
        MoveSources(es_->scene, op.target, "");
        out->rebuild = out->rebuild || body_deleted || shape_deleted;
    } else if (op.field == "reparent") {
        out->focus = ReparentNode(*es_, op.target, op.after.AsString());
        if (out->focus.empty()) {
            *error = "cannot move '" + op.target + "' under '" + op.after.AsString() + "'";
            return false;
        }
        MoveSources(es_->scene, op.target, out->focus);
        out->rebuild = true;
    } else if (op.field == "script") {
        scene::ScriptId id = scene::kInvalidScript;
        if (op.target.empty()) {
            uint64_t next_id = 1u;
            for (const scene::ScriptRecord& sr : es_->scene.Scripts())
                if (sr.stable_id >= next_id) next_id = sr.stable_id + 1u;
            scene::ScriptRecord rec;
            rec.stable_id = next_id;
            rec.source = op.after.AsString();
            id = es_->scene.AddScript(rec);
        } else {
            for (size_t i = 0; i < es_->scene.ScriptCount(); ++i)
                if (e != scene::kInvalidEntity && es_->scene.EntityOfScript(static_cast<scene::ScriptId>(i)) == e)
                    id = static_cast<scene::ScriptId>(i);
            if (id == scene::kInvalidScript) {
                *error = "no script at '" + op.target + "'";
                return false;
            }
            done->before = json::Value::Str(es_->scene.GetScript(id).source);
            es_->scene.GetScriptMut(id).source = op.after.AsString();
        }
        if (const auto node = es_->scene.Ecs().NodeOf(es_->scene.EntityOfScript(id)))
            out->focus = es_->scene.Tree().PathOf(node);
    } else if (!ApplyParam(op, done, out, error)) {
        return false;
    }
    if (inverse->target.empty()) inverse->target = op.target;
    inverse->field = op.field;
    inverse->before = done->after;
    inverse->after = done->before;
    return true;
}

bool EditorDocument::ApplyParam(const EditOp& op, EditOp* done, OpOutcome* out, std::string* error) {
    const uint32_t p = ParamOf(op.field);
    const std::string scene_field = p == kNoParam ? op.field : std::string(infos_[p].scene_field);
    if (p == kNoParam && !scene::IsSceneField(scene_field)) {
        *error = "unknown field '" + op.field + "'";
        return false;
    }
    const double value = op.after.AsDouble();
    if (p != kNoParam) {
        const nuka_param_info_t& info = infos_[p];
        if (!std::isfinite(value) || value < info.min_value || value > info.max_value ||
            (info.type == NUKA_PARAM_U32 && value != std::floor(value))) {
            *error = "'" + op.field + "' is outside its range";
            return false;
        }
    }
    if (p != kNoParam && scene_field.empty()) {
        if (!op.target.empty()) {
            *error = "'" + op.field + "' is a world option; its target is \"\"";
            return false;
        }
        done->before = json::Value::Float(world_values_[p]);
        world_values_[p] = value;
        out->writes.push_back(ParamWrite{p, 0u, value});
        return true;
    }
    const scene::EntityId e = EntityAt(op.target);
    scene::SceneTarget target;
    if (const auto it = records_.body.find(e); it != records_.body.end()) target.body = it->second;
    if (const auto it = records_.shape.find(e); it != records_.shape.end()) target.shape = it->second;
    float before = 0.0f;
    if (!scene::GetSceneField(es_->scene, target, scene_field, &before)) {
        *error = "'" + op.target + "' has no " + scene_field;
        return false;
    }
    done->before = json::Value::Float(before);
    scene::SetSceneField(es_->scene, target, scene_field, static_cast<float>(value));
    uint32_t element = 0u;
    if (LiveElement(op.target, p, &element)) out->writes.push_back(ParamWrite{p, element, value});
    else out->rebuild = true;
    return true;
}

ApplyResult EditorDocument::Apply(const EditTxn& txn) {
    ApplyResult result;
    Entry entry;
    entry.forward.label = entry.inverse.label = txn.label;
    for (const EditOp& op : txn.ops) entry.snapshot = entry.snapshot || NeedsSnapshot(op);
    if (entry.snapshot) entry.before = es_->scene;
    OpOutcome out;
    for (const EditOp& op : txn.ops) {
        EditOp done;
        EditOp inverse;
        if (!ApplyOp(op, &done, &inverse, &out, &result.error)) {
            result.ok = false;
            break;
        }
        entry.forward.ops.push_back(std::move(done));
        entry.inverse.ops.insert(entry.inverse.ops.begin(), std::move(inverse));
    }
    // Re-project so the tree order is the one a reload of these records gives.
    if (entry.snapshot) es_->scene = scene::SceneIR(es_->scene);
    records_.Build(es_->scene);
    result.focus = out.focus;
    if (entry.forward.ops.empty()) return result;
    entry.rebuild = out.rebuild;
    entry.focus = out.focus;
    if (const std::string why = entry.rebuild ? host_->Rebuild(txn.label) : std::string(); !why.empty()) {
        // The world kept its scene: take back the live previews, then the document.
        ReplayProperties(entry.inverse, false);
        es_->scene = std::move(entry.before);
        records_.Build(es_->scene);
        host_->ScriptsChanged();
        result.ok = false;
        result.error = "the world refused the rebuild (" + why + "); the edit was rolled back";
        return result;
    }
    // A rebuild cooked the written values already.
    if (!entry.rebuild)
        for (const ParamWrite& write : out.writes) host_->PushParam(write);
    if (entry.snapshot) {
        entry.after = es_->scene;
        host_->ScriptsChanged();
    }
    entry.id = ++next_id_;
    undo_.push_back(std::move(entry));
    if (undo_.size() > kCap) undo_.erase(undo_.begin());
    redo_.clear();
    ++revision_;
    return result;
}

void EditorDocument::ReplayProperties(const EditTxn& txn, bool rebuilt) {
    OpOutcome out;
    std::string error;
    for (const EditOp& op : txn.ops) {
        if (IsStructural(op.field) || op.field == "name") continue;
        EditOp done;
        EditOp inverse;
        ApplyOp(op, &done, &inverse, &out, &error);
    }
    if (!rebuilt)
        for (const ParamWrite& write : out.writes) host_->PushParam(write);
}

ApplyResult EditorDocument::Restore(const Entry& entry, bool forward) {
    ApplyResult result;
    const EditTxn& txn = forward ? entry.forward : entry.inverse;
    if (!entry.snapshot) {
        OpOutcome out;
        for (const EditOp& op : txn.ops) {
            EditOp done;
            EditOp inverse;
            if (!ApplyOp(op, &done, &inverse, &out, &result.error)) {
                result.ok = false;
                break;
            }
        }
        records_.Build(es_->scene);
        for (const ParamWrite& write : out.writes) host_->PushParam(write);
        result.focus = out.focus;
        return result;
    }
    scene::SceneIR prior = std::exchange(es_->scene, forward ? entry.after : entry.before);
    records_.Build(es_->scene);
    const std::string why = entry.rebuild ? host_->Rebuild((forward ? "redo " : "undo ") + txn.label) : std::string();
    if (!why.empty()) {
        es_->scene = std::move(prior);
        records_.Build(es_->scene);
        result.ok = false;
        result.error = "the world refused the rebuild (" + why + ")";
        return result;
    }
    ReplayProperties(txn, entry.rebuild);
    records_.Build(es_->scene);
    host_->ScriptsChanged();
    if (forward) result.focus = entry.focus;
    return result;
}

ApplyResult EditorDocument::Undo() {
    if (undo_.empty()) return ApplyResult{false, "", "nothing to undo"};
    Entry entry = std::move(undo_.back());
    undo_.pop_back();
    ApplyResult result = Restore(entry, /*forward=*/false);
    if (!result.ok) {
        undo_.push_back(std::move(entry));
        return result;
    }
    redo_.push_back(std::move(entry));
    ++revision_;
    return result;
}

ApplyResult EditorDocument::Redo() {
    if (redo_.empty()) return ApplyResult{false, "", "nothing to redo"};
    Entry entry = std::move(redo_.back());
    redo_.pop_back();
    ApplyResult result = Restore(entry, /*forward=*/true);
    if (!result.ok) {
        redo_.push_back(std::move(entry));
        return result;
    }
    undo_.push_back(std::move(entry));
    ++revision_;
    return result;
}

}  // namespace nuka::runtime::app::viewer
