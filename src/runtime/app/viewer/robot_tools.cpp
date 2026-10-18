#include "runtime/app/viewer/robot_tools.hpp"

#include <cstddef>
#include <utility>

namespace nuka::runtime::app::viewer {
namespace {

namespace json = scene::json;

constexpr std::pair<scene::ActuatorType, const char*> kActuatorTypes[] = {
    {scene::ActuatorType::Position, "position"},
    {scene::ActuatorType::Motor, "motor"},
    {scene::ActuatorType::Velocity, "velocity"},
    {scene::ActuatorType::Force, "force"}};

const scene::JointRecord* IncomingJoint(const scene::SceneIR& scene, scene::BodyId body) {
    for (const scene::JointRecord& j : scene.Joints())
        if (j.child_body == body) return &j;
    return nullptr;
}

// One environment's row of a per-link field.
void DownloadLinks(nuka_world_handle world, nuka_state_field_t field, const WorldLayout& layout, uint32_t env,
                   std::vector<float>* out) {
    out->resize(layout.links_per_env);
    CheckResult(nuka_world_download_field(world, field, out->data(), out->size() * sizeof(float),
                                          static_cast<size_t>(env) * layout.links_per_env * sizeof(float)),
                "nuka_world_download_field");
}

float At(const std::vector<float>& v, uint32_t i) { return i < v.size() ? v[i] : 0.0f; }

}  // namespace

const char* ActuatorTypeName(scene::ActuatorType type) {
    for (const auto& [t, name] : kActuatorTypes)
        if (t == type) return name;
    return "";
}

bool ParseActuatorType(const std::string& name, scene::ActuatorType* out) {
    for (const auto& [t, n] : kActuatorTypes) {
        if (name == n) {
            *out = t;
            return true;
        }
    }
    return false;
}

std::vector<RobotJoint> RobotJoints(const scene::SceneIR& scene, const WorldLayout& layout,
                                    const std::function<std::string(scene::EntityId)>& path_of) {
    std::vector<RobotJoint> joints;
    for (uint32_t l = 0; l < layout.links_per_env; ++l) {
        const nuka_link_binding_t& link = layout.links[l];
        const scene::JointRecord* j = link.is_root != 0u ? nullptr : IncomingJoint(scene, link.body_index);
        if (j == nullptr || (j->type != scene::JointType::Revolute && j->type != scene::JointType::Prismatic)) continue;
        RobotJoint r;
        r.link = l;
        r.name = j->name;
        r.path = path_of(scene.EntityOfBody(link.body_index));
        r.prismatic = j->type == scene::JointType::Prismatic;
        r.limited = j->has_lower_limit && j->has_upper_limit;
        r.lower = j->lower_limit;
        r.upper = j->upper_limit;
        for (const scene::ActuatorRecord& a : scene.Actuators()) {
            if (a.joint_id != j->id) continue;
            r.drive = ActuatorTypeName(a.type);
            r.kp = a.type == scene::ActuatorType::Position ? a.gain : 0.0f;
            r.force_limit = a.force_limit;
        }
        joints.push_back(std::move(r));
    }
    return joints;
}

std::vector<std::string> RobotRoots(const scene::SceneIR& scene, const WorldLayout& layout,
                                    const std::function<std::string(scene::EntityId)>& path_of) {
    std::vector<std::string> roots;
    for (uint32_t l = 0; l < layout.links_per_env; ++l)
        if (layout.links[l].is_root != 0u) roots.push_back(path_of(scene.EntityOfBody(layout.links[l].body_index)));
    return roots;
}

int FindJoint(const std::vector<RobotJoint>& joints, const std::string& name) {
    for (size_t i = 0; i < joints.size(); ++i)
        if (joints[i].name == name) return static_cast<int>(i);
    return -1;
}

void DownloadJointFrame(nuka_world_handle world, const WorldLayout& layout, uint32_t env, JointFrame* out) {
    DownloadLinks(world, NUKA_FIELD_JOINT_POSITION, layout, env, &out->q);
    DownloadLinks(world, NUKA_FIELD_JOINT_VELOCITY, layout, env, &out->qd);
    DownloadLinks(world, NUKA_FIELD_DRIVE_TARGET, layout, env, &out->target);
    DownloadLinks(world, NUKA_FIELD_ACTUATOR_EFFORT, layout, env, &out->effort);
    DownloadLinks(world, NUKA_FIELD_ACTUATOR_SATURATED, layout, env, &out->saturated);
}

void DownloadJointSample(nuka_world_handle world, const WorldLayout& layout, uint32_t env, JointSample* out) {
    DownloadLinks(world, NUKA_FIELD_JOINT_POSITION, layout, env, &out->q);
    DownloadLinks(world, NUKA_FIELD_DRIVE_TARGET, layout, env, &out->target);
    DownloadLinks(world, NUKA_FIELD_ACTUATOR_EFFORT, layout, env, &out->effort);
}

json::Value JointJson(const RobotJoint& joint, const JointFrame& live) {
    json::Value v = json::Value::Object();
    v.Set("name", json::Value::Str(joint.name));
    v.Set("link", json::Value::Int(joint.link));
    v.Set("path", json::Value::Str(joint.path));
    v.Set("type", json::Value::Str(joint.prismatic ? "prismatic" : "revolute"));
    v.Set("lower", joint.limited ? json::Value::Float(joint.lower) : json::Value::Null());
    v.Set("upper", joint.limited ? json::Value::Float(joint.upper) : json::Value::Null());
    v.Set("drive", joint.drive.empty() ? json::Value::Null() : json::Value::Str(joint.drive));
    v.Set("kp", json::Value::Float(joint.kp));
    v.Set("force_limit", json::Value::Float(joint.force_limit));
    v.Set("q", json::Value::Float(At(live.q, joint.link)));
    v.Set("qd", json::Value::Float(At(live.qd, joint.link)));
    v.Set("target", json::Value::Float(At(live.target, joint.link)));
    v.Set("effort", json::Value::Float(At(live.effort, joint.link)));
    v.Set("saturated", json::Value::Bool(At(live.saturated, joint.link) != 0.0f));
    return v;
}

}  // namespace nuka::runtime::app::viewer
