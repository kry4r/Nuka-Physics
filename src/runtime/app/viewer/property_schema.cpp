#include "runtime/app/viewer/property_schema.hpp"

#include "runtime/app/viewer/editor_document.hpp"

#include <algorithm>
#include <cctype>
#include <cfloat>

namespace nuka::runtime::app::viewer {
namespace {

// A field and its units: `unit` for rotation, `linear` along a prismatic joint.
struct Spec {
    const char* field;
    const char* label;
    const char* unit;
    const char* linear;
    float speed;
    float min;
    float max;
};

constexpr Spec kBody[] = {{"body.mass", "Mass", "kg", "kg", 0.01f, 0.0f, 0.0f},
                          {"body.inertia.x", "Inertia X", "kg\xC2\xB7m\xC2\xB2", "", 0.0005f, 0.0f, 0.0f},
                          {"body.inertia.y", "Inertia Y", "kg\xC2\xB7m\xC2\xB2", "", 0.0005f, 0.0f, 0.0f},
                          {"body.inertia.z", "Inertia Z", "kg\xC2\xB7m\xC2\xB2", "", 0.0005f, 0.0f, 0.0f}};
constexpr Spec kJoint[] = {{"joint.damping", "Damping", "N\xC2\xB7m\xC2\xB7s/rad", "N\xC2\xB7s/m", 0.01f, 0.0f, 0.0f},
                           {"joint.stiffness", "Stiffness", "N\xC2\xB7m/rad", "N/m", 0.1f, 0.0f, 0.0f},
                           {"joint.armature", "Armature", "kg\xC2\xB7m\xC2\xB2", "kg", 0.001f, 0.0f, 0.0f},
                           {"joint.frictionloss", "Friction loss", "N\xC2\xB7m", "N", 0.01f, 0.0f, 0.0f},
                           {"joint.lower_limit", "Lower limit", "rad", "m", 0.01f, 0.0f, 0.0f},
                           {"joint.upper_limit", "Upper limit", "rad", "m", 0.01f, 0.0f, 0.0f}};
constexpr Spec kDrive[] = {{"actuator.gain", "Gain", "N\xC2\xB7m/rad", "N/m", 0.1f, 0.0f, 0.0f},
                           {"actuator.force_limit", "Force limit", "N\xC2\xB7m", "N", 0.1f, 0.0f, 0.0f}};
constexpr Spec kFriction = {"shape.friction_mu", "Friction", "", "", 0.01f, 0.0f, 0.0f};
constexpr Spec kRadius = {"shape.radius", "Radius", "m", "", 0.001f, 0.0f, 0.0f};
constexpr Spec kHalfHeight = {"shape.half_height", "Half height", "m", "", 0.001f, 0.0f, 0.0f};
constexpr Spec kHalfExtents[] = {{"shape.half_extents.x", "Half X", "m", "", 0.001f, 0.0f, 0.0f},
                                 {"shape.half_extents.y", "Half Y", "m", "", 0.001f, 0.0f, 0.0f},
                                 {"shape.half_extents.z", "Half Z", "m", "", 0.001f, 0.0f, 0.0f}};
constexpr Spec kMargin = {"shape.margin", "Margin", "m", "", 0.0005f, 0.0f, 0.0f};
constexpr Spec kGap = {"shape.gap", "Gap", "m", "", 0.0005f, 0.0f, 0.0f};

bool Visual(const scene::CollisionShapeRecord& sh) { return sh.contype == 0u && sh.conaffinity == 0u; }

// A negative shape friction takes its material's, as the cooker resolves it.
float Resolved(const scene::SceneIR& s, const scene::SceneTarget& t, const std::string& field, float value) {
    if (field != kFriction.field || value >= 0.0f) return value;
    const scene::MaterialId m = s.GetShape(t.shape).material_id;
    return m < s.Materials().size() ? s.Materials()[m].friction_mu : 1.0f;
}

std::string NodePath(const scene::SceneIR& s, scene::EntityId e) {
    const auto node = s.Ecs().NodeOf(e);
    return node ? s.Tree().PathOf(node) : std::string();
}

std::string Leaf(const std::string& path) {
    const size_t slash = path.rfind('/');
    return slash == std::string::npos ? path : path.substr(slash + 1u);
}

float Bound(double v) { return static_cast<float>(std::clamp(v, -static_cast<double>(FLT_MAX), static_cast<double>(FLT_MAX))); }

// "gravity_z" reads "Gravity z".
std::string Title(std::string key) {
    std::replace(key.begin(), key.end(), '_', ' ');
    if (!key.empty()) key[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(key[0])));
    return key;
}

class Builder {
public:
    Builder(const scene::SceneIR& s, const EditorDocument& doc) : s_(s), doc_(doc) {}

    std::vector<PropertyGroup> Groups(const std::string& path) {
        groups_.clear();
        const scene::SceneTarget t = TargetAt(s_, path);
        if (t.body != scene::kInvalidBody) {
            const std::string body = NodePath(s_, s_.EntityOfBody(t.body));
            Group("Body", "", "body", body, scene::SceneTarget{t.body, scene::kInvalidShape}, kBody, false);
            for (scene::JointId j = 0; j < s_.JointCount(); ++j) {
                const scene::JointRecord& joint = s_.Joints()[j];
                if (joint.child_body != t.body ||
                    (joint.type != scene::JointType::Revolute && joint.type != scene::JointType::Prismatic))
                    continue;
                const bool linear = joint.type == scene::JointType::Prismatic;
                Group("Joint", "", "joint", body, scene::SceneTarget{t.body, scene::kInvalidShape}, kJoint, linear);
                if (std::any_of(s_.Actuators().begin(), s_.Actuators().end(),
                                [&](const scene::ActuatorRecord& a) { return a.joint_id == j; }))
                    Group("Drive", "", "drive", body, scene::SceneTarget{t.body, scene::kInvalidShape}, kDrive, linear);
            }
            for (scene::ShapeId i = 0; i < s_.ShapeCount(); ++i)
                if (s_.GetShape(i).body_id == t.body && !Visual(s_.GetShape(i))) Collision(i, true);
        } else if (t.shape != scene::kInvalidShape && !Visual(s_.GetShape(t.shape))) {
            Collision(t.shape, false);
        }
        return std::move(groups_);
    }

    std::vector<PropertyGroup> World() {
        PropertyGroup g{"World", "", "world", {std::string()}, {scene::SceneTarget{}}, {}};
        const std::vector<nuka_param_info_t>& infos = doc_.Params();
        for (size_t i = 0; i < infos.size(); ++i) {
            const nuka_param_info_t& info = infos[i];
            if (info.scene_field[0] != '\0') continue;
            PropertyField f;
            f.field = info.key;
            f.label = Title(info.key);
            f.unit = info.unit;
            f.integer = info.type == NUKA_PARAM_U32;
            f.speed = f.integer ? 0.1f : 0.01f;
            f.min = Bound(info.min_value);
            f.max = Bound(info.max_value);
            f.live = info.tier == NUKA_PARAM_LIVE;
            f.world = true;
            f.value = static_cast<float>(doc_.WorldValues()[i]);
            g.fields.push_back(std::move(f));
        }
        return {std::move(g)};
    }

private:
    template <size_t N>
    void Group(const char* title, const std::string& detail, const char* kind, const std::string& target,
               const scene::SceneTarget& record, const Spec (&specs)[N], bool linear) {
        PropertyGroup g{title, detail, kind, {target}, {record}, {}};
        for (const Spec& spec : specs) Add(g, spec, linear);
        groups_.push_back(std::move(g));
    }

    // A collision group per colliding shape, with the size fields its type has.
    void Collision(scene::ShapeId id, bool named) {
        const scene::CollisionShapeRecord& sh = s_.GetShape(id);
        const std::string path = NodePath(s_, s_.EntityOfShape(id));
        PropertyGroup g{"Collision", named ? Leaf(path) : std::string(), "collision", {path},
                        {scene::SceneTarget{scene::kInvalidBody, id}}, {}};
        Add(g, kFriction, false);
        if (sh.type == scene::ShapeType::Box)
            for (const Spec& spec : kHalfExtents) Add(g, spec, false);
        const bool axial = sh.type == scene::ShapeType::Capsule || sh.type == scene::ShapeType::Cylinder;
        if (sh.type == scene::ShapeType::Sphere || axial) Add(g, kRadius, false);
        if (axial) Add(g, kHalfHeight, false);
        Add(g, kMargin, false);
        Add(g, kGap, false);
        groups_.push_back(std::move(g));
    }

    // A registry parameter writing this field gives its unit and range.
    void Add(PropertyGroup& g, const Spec& spec, bool linear) {
        PropertyField f;
        f.field = spec.field;
        f.label = spec.label;
        f.unit = linear ? spec.linear : spec.unit;
        f.speed = spec.speed;
        f.min = spec.min;
        f.max = spec.max;
        for (const nuka_param_info_t& info : doc_.Params()) {
            if (f.field != info.scene_field) continue;
            if (info.unit[0] != '\0' && !linear) f.unit = info.unit;
            f.min = Bound(info.min_value);
            f.max = Bound(info.max_value);
        }
        float value = 0.0f;
        if (!scene::GetSceneField(s_, g.records.front(), f.field, &value)) return;
        f.value = Resolved(s_, g.records.front(), f.field, value);
        f.inherited = f.value != value;
        f.live = doc_.LiveField(g.targets.front(), f.field);
        const scene::SceneTarget saved = TargetAt(doc_.Saved(), g.targets.front());
        f.has_saved = scene::GetSceneField(doc_.Saved(), saved, f.field, &f.saved_raw);
        if (f.has_saved) f.saved = Resolved(doc_.Saved(), saved, f.field, f.saved_raw);
        g.fields.push_back(std::move(f));
    }

    const scene::SceneIR& s_;
    const EditorDocument& doc_;
    std::vector<PropertyGroup> groups_;
};

}  // namespace

scene::SceneTarget TargetAt(const scene::SceneIR& s, const std::string& path) {
    scene::SceneTarget t;
    const auto node = s.Tree().NodeOf(path);
    if (!node) return t;
    for (scene::BodyId b = 0; b < s.RigidBodyCount(); ++b)
        if (s.EntityOfBody(b) == node->entity) t.body = b;
    for (scene::ShapeId i = 0; i < s.ShapeCount(); ++i)
        if (s.EntityOfShape(i) == node->entity) t.shape = i;
    for (scene::JointId j = 0; j < s.JointCount(); ++j)
        if (s.EntityOfJoint(j) == node->entity) t.body = s.Joints()[j].child_body;
    return t;
}

std::vector<PropertyGroup> PropertySchema(const scene::SceneIR& scene, const EditorDocument& doc,
                                          const std::vector<std::string>& paths) {
    Builder builder(scene, doc);
    if (paths.empty()) return builder.World();
    std::vector<PropertyGroup> common = builder.Groups(paths.front());
    for (size_t k = 1; k < paths.size(); ++k) {
        const std::vector<PropertyGroup> other = builder.Groups(paths[k]);
        for (auto g = common.begin(); g != common.end();) {
            const auto o = std::find_if(other.begin(), other.end(),
                                        [&](const PropertyGroup& x) { return x.title == g->title && x.detail == g->detail; });
            if (o == other.end()) {
                g = common.erase(g);
                continue;
            }
            g->fields.erase(std::remove_if(g->fields.begin(), g->fields.end(),
                                           [&](PropertyField& f) {
                                               const auto of = std::find_if(o->fields.begin(), o->fields.end(),
                                                                            [&](const PropertyField& x) { return x.field == f.field; });
                                               if (of == o->fields.end()) return true;
                                               f.live = f.live && of->live;
                                               f.mixed = f.mixed || of->value != f.value;
                                               return false;
                                           }),
                            g->fields.end());
            g->targets.push_back(o->targets.front());
            g->records.push_back(o->records.front());
            ++g;
        }
    }
    return common;
}

}  // namespace nuka::runtime::app::viewer
