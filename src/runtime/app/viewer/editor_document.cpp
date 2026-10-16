#include "runtime/app/viewer/editor_document.hpp"

#include "render/render_world.hpp"
#include "scene/scene_fields.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace nuka::runtime::app::viewer {
namespace {

namespace json = scene::json;

constexpr uint32_t kNoParam = ~0u;

bool IsStructural(const std::string& field) {
    return field == "spawn" || field == "delete" || field == "reparent" || field == "script";
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
}

std::string Leaf(const std::string& path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1u);
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
    return v;
}

bool ParsePrimitiveKind(const std::string& s, PrimitiveKind* out) {
    for (const PrimitiveKind kind : {PrimitiveKind::Box, PrimitiveKind::Sphere, PrimitiveKind::Capsule,
                                     PrimitiveKind::Plane}) {
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
    }
    return "";
}

void EditorDocument::Attach(EditorScene* es, DocumentHost* host) {
    es_ = es;
    host_ = host;
    records_.Build(es_->scene);
    undo_.clear();
    redo_.clear();
    branches_.clear();
    saved_ = 0u;
    ++revision_;
}

void EditorDocument::Detach() {
    es_ = nullptr;
    records_ = EntityRecordIndex{};
    undo_.clear();
    redo_.clear();
    branches_.clear();
    saved_ = 0u;
    ++revision_;
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

bool EditorDocument::ApplyOp(const EditOp& op, EditOp* done, EditOp* inverse, OpOutcome* out, std::string* error) {
    *done = op;
    const scene::EntityId e = EntityAt(op.target);
    const bool needs_node = !op.target.empty() || op.field == "transform" || op.field == "material" ||
                            op.field == "name" || op.field == "delete" || op.field == "reparent";
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
        const MaterialBinding mb = ResolveMaterial(*es_, records_, e);
        if (!mb.valid) {
            *error = "'" + op.target + "' has no material";
            return false;
        }
        scene::RenderMaterial m;
        const render::RenderWorld& rw = es_->sim->GetRenderWorld();
        if (mb.render_slot != render::kNoId && mb.render_slot < rw.MaterialCount()) m = rw.materials[mb.render_slot];
        if (done->before.IsNull()) done->before = MaterialToJson(m);
        MaterialFromJson(op.after, &m);
        done->after = MaterialToJson(m);
        ApplyMaterialEdit(*es_, mb, m, /*commit=*/true);
    } else if (op.field == "name") {
        out->focus = RenameNode(*es_, op.target, op.after.AsString());
        if (out->focus.empty()) {
            *error = "cannot rename '" + op.target + "'";
            return false;
        }
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
    } else if (op.field == "delete") {
        const size_t bodies = es_->scene.RigidBodyCount();
        const size_t scripts = es_->scene.ScriptCount();
        out->focus = DeleteSubtree(*es_, op.target);
        const bool body_deleted = es_->scene.RigidBodyCount() < bodies;
        if (!body_deleted && es_->scene.ScriptCount() >= scripts) {
            *error = "nothing to delete at '" + op.target + "'";
            return false;
        }
        out->rebuild = out->rebuild || body_deleted;
    } else if (op.field == "reparent") {
        out->focus = ReparentNode(*es_, op.target, op.after.AsString());
        if (out->focus.empty()) {
            *error = "cannot move '" + op.target + "' under '" + op.after.AsString() + "'";
            return false;
        }
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
    if (entry.rebuild && !host_->Rebuild(txn.label)) {
        // The world kept its scene: take back the live previews, then the document.
        ReplayProperties(entry.inverse, false);
        es_->scene = std::move(entry.before);
        records_.Build(es_->scene);
        host_->ScriptsChanged();
        result.ok = false;
        result.error = "the world refused the rebuild; the edit was rolled back";
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
    if (entry.rebuild && !host_->Rebuild((forward ? "redo " : "undo ") + txn.label)) {
        es_->scene = std::move(prior);
        records_.Build(es_->scene);
        result.ok = false;
        result.error = "the world refused the rebuild";
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
