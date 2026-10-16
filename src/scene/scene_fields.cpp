#include "scene/scene_fields.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nuka::scene {

namespace {

template <class Record>
struct Field {
    const char* name;
    float Record::*member;
};

constexpr Field<RigidBodyRecord> kBodyFields[] = {{"mass", &RigidBodyRecord::mass}};
constexpr Field<JointRecord> kJointFields[] = {
    {"damping", &JointRecord::damping},         {"armature", &JointRecord::armature},
    {"stiffness", &JointRecord::stiffness},     {"frictionloss", &JointRecord::frictionloss},
    {"lower_limit", &JointRecord::lower_limit}, {"upper_limit", &JointRecord::upper_limit}};
constexpr Field<ActuatorRecord> kActuatorFields[] = {
    {"gain", &ActuatorRecord::gain}, {"force_limit", &ActuatorRecord::force_limit}};
constexpr Field<CollisionShapeRecord> kShapeFields[] = {
    {"friction_mu", &CollisionShapeRecord::friction_mu}, {"radius", &CollisionShapeRecord::radius},
    {"half_height", &CollisionShapeRecord::half_height}, {"margin", &CollisionShapeRecord::margin},
    {"gap", &CollisionShapeRecord::gap}};
constexpr Field<MaterialRecord> kMaterialFields[] = {{"friction_mu", &MaterialRecord::friction_mu}};

template <class Record, size_t N>
float Record::*Member(const Field<Record> (&fields)[N], std::string_view name) {
    for (const Field<Record>& f : fields)
        if (name == f.name) return f.member;
    return nullptr;
}

enum class Kind : uint8_t { Body, Joint, Actuator, Shape, Material };

struct Selector {
    Kind kind = Kind::Body;
    std::string_view name;
};

bool Select(std::string_view field, Selector* out) {
    const size_t dot = field.find('.');
    if (dot == std::string_view::npos) return false;
    const std::string_view record = field.substr(0, dot);
    out->name = field.substr(dot + 1);
    if (record == "body") out->kind = Kind::Body;
    else if (record == "joint") out->kind = Kind::Joint;
    else if (record == "actuator") out->kind = Kind::Actuator;
    else if (record == "shape") out->kind = Kind::Shape;
    else if (record == "material") out->kind = Kind::Material;
    else return false;
    switch (out->kind) {
        case Kind::Body: return Member(kBodyFields, out->name) != nullptr;
        case Kind::Joint: return Member(kJointFields, out->name) != nullptr;
        case Kind::Actuator: return Member(kActuatorFields, out->name) != nullptr;
        case Kind::Shape: return Member(kShapeFields, out->name) != nullptr;
        case Kind::Material: return Member(kMaterialFields, out->name) != nullptr;
    }
    return false;
}

// The ids of the records a field reaches from the target, in record order. A gain is
// the drive stiffness only of position actuators, so it reaches only those.
std::vector<uint32_t> Reach(const SceneIR& scene, const SceneTarget& target, const Selector& s) {
    const Kind kind = s.kind;
    std::vector<uint32_t> ids;
    switch (kind) {
        case Kind::Body:
            if (target.body < scene.RigidBodyCount()) ids.push_back(target.body);
            break;
        case Kind::Joint:
        case Kind::Actuator:
            if (target.body == kInvalidBody) break;
            for (const JointRecord& joint : scene.Joints()) {
                if (joint.child_body != target.body) continue;
                if (kind == Kind::Joint) ids.push_back(joint.id);
                for (const ActuatorRecord& act : scene.Actuators())
                    if (kind == Kind::Actuator && act.joint_id == joint.id &&
                        (s.name != "gain" || act.type == ActuatorType::Position))
                        ids.push_back(act.id);
                break;
            }
            break;
        case Kind::Shape:
            if (target.shape < scene.ShapeCount()) ids.push_back(target.shape);
            break;
        case Kind::Material:
            if (target.shape < scene.ShapeCount() &&
                scene.GetShape(target.shape).material_id < scene.MaterialCount())
                ids.push_back(scene.GetShape(target.shape).material_id);
            break;
    }
    return ids;
}

}  // namespace

bool IsSceneField(std::string_view field) {
    Selector s;
    return Select(field, &s);
}

bool GetSceneField(const SceneIR& scene, const SceneTarget& target, std::string_view field, float* value) {
    Selector s;
    if (!Select(field, &s)) return false;
    const std::vector<uint32_t> ids = Reach(scene, target, s);
    if (ids.empty()) return false;
    switch (s.kind) {
        case Kind::Body: *value = scene.GetBody(ids[0]).*Member(kBodyFields, s.name); break;
        case Kind::Joint: *value = scene.GetJoint(ids[0]).*Member(kJointFields, s.name); break;
        case Kind::Actuator: *value = scene.GetActuator(ids[0]).*Member(kActuatorFields, s.name); break;
        case Kind::Shape: *value = scene.GetShape(ids[0]).*Member(kShapeFields, s.name); break;
        case Kind::Material: *value = scene.GetMaterial(ids[0]).*Member(kMaterialFields, s.name); break;
    }
    return true;
}

bool SetSceneField(SceneIR& scene, const SceneTarget& target, std::string_view field, float value) {
    Selector s;
    if (!Select(field, &s)) return false;
    const std::vector<uint32_t> ids = Reach(scene, target, s);
    for (const uint32_t id : ids) {
        switch (s.kind) {
            case Kind::Body: scene.GetBodyMut(id).*Member(kBodyFields, s.name) = value; break;
            case Kind::Joint: scene.GetJointMut(id).*Member(kJointFields, s.name) = value; break;
            case Kind::Actuator: scene.GetActuatorMut(id).*Member(kActuatorFields, s.name) = value; break;
            case Kind::Shape: scene.GetShapeMut(id).*Member(kShapeFields, s.name) = value; break;
            case Kind::Material: scene.GetMaterialMut(id).*Member(kMaterialFields, s.name) = value; break;
        }
    }
    return !ids.empty();
}

}  // namespace nuka::scene
