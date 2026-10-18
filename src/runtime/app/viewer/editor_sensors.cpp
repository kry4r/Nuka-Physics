#include "runtime/app/viewer/editor_sensors.hpp"

#include "runtime/app/viewer/editor_document.hpp"
#include "runtime/app/viewer/editor_edits.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <utility>

namespace nuka::runtime::app::viewer {
namespace {

namespace json = scene::json;
using scene::SensorType;

constexpr std::pair<SensorType, const char*> kTypeNames[] = {
    {SensorType::Imu, "imu"},
    {SensorType::Contact, "contact"},
    {SensorType::ForceTorque, "force_torque"},
    {SensorType::FramePose, "frame_pose"},
    {SensorType::JointState, "joint_state"},
    {SensorType::Camera, "camera"},
    {SensorType::Depth, "depth"},
    {SensorType::Lidar, "lidar"},
    {SensorType::RangeScan, "range_scan"},
    {SensorType::LinearVelocity, "linear_velocity"},
    {SensorType::Tactile, "tactile"}};

constexpr std::pair<::nuka::sensor::ContactRegionShape, const char*> kRegionNames[] = {
    {::nuka::sensor::ContactRegionShape::Box, "box"},
    {::nuka::sensor::ContactRegionShape::Sphere, "sphere"},
    {::nuka::sensor::ContactRegionShape::Ellipsoid, "ellipsoid"},
    {::nuka::sensor::ContactRegionShape::Capsule, "capsule"},
    {::nuka::sensor::ContactRegionShape::Cylinder, "cylinder"}};

bool CameraSensor(SensorType type) { return type == SensorType::Camera || type == SensorType::Depth; }

json::Value Vec3Json(const math::Vec3& v) {
    json::Value a = json::Value::Array();
    for (const float c : {v.x, v.y, v.z}) a.PushBack(json::Value::Float(c));
    return a;
}

math::Vec3 Vec3FromJson(const json::Value& v) {
    return math::Vec3{v.Elements()[0].AsFloat(), v.Elements()[1].AsFloat(), v.Elements()[2].AsFloat()};
}

void ReadFloat(const json::Value& v, const char* key, float* out) {
    if (const json::Value* f = v.Find(key)) *out = f->AsFloat();
}

void ReadCount(const json::Value& v, const char* key, uint16_t* out) {
    if (const json::Value* f = v.Find(key)) *out = static_cast<uint16_t>(f->AsInt());
}

std::string Size(uint32_t a, uint32_t b) { return std::to_string(a) + "x" + std::to_string(b); }

}  // namespace

const char* SensorTypeName(SensorType type) {
    for (const auto& [t, name] : kTypeNames)
        if (t == type) return name;
    return "";
}

bool ParseSensorType(const std::string& name, SensorType* out) {
    for (const auto& [t, n] : kTypeNames) {
        if (name == n) {
            *out = t;
            return true;
        }
    }
    return false;
}

bool RenderSensor(SensorType type) { return CameraSensor(type) || LidarSensor(type); }

bool LidarSensor(SensorType type) { return type == SensorType::Lidar || type == SensorType::RangeScan; }

scene::BodyId SensorBody(const scene::SceneIR& scene, const scene::SensorDesc& s) {
    if (s.joint_id != scene::kInvalidJoint) return scene.GetJoint(s.joint_id).child_body;
    return s.mount == scene::MountFrame::Body ? s.mount_index : scene::kInvalidBody;
}

std::string BodyPath(const scene::SceneIR& scene, scene::BodyId body) {
    if (body == scene::kInvalidBody) return std::string();
    return scene.Tree().PathOf(scene.Ecs().NodeOf(scene.EntityOfBody(body)));
}

scene::BodyId BodyAtOrAbove(const scene::SceneIR& scene, const std::string& path) {
    for (auto node = scene.Tree().NodeOf(path); node; node = scene.Tree().ParentOf(node))
        for (scene::BodyId b = 0; b < scene.RigidBodyCount(); ++b)
            if (scene.EntityOfBody(b) == node->entity) return b;
    return scene::kInvalidBody;
}

json::Value SensorToJson(const scene::SceneIR& scene, const scene::SensorDesc& s) {
    const scene::BodyId body = SensorBody(scene, s);
    const char* frame = body != scene::kInvalidBody             ? "body"
                        : s.mount == scene::MountFrame::World ? "world"
                        : s.mount == scene::MountFrame::Base  ? "base"
                                                              : "link";
    json::Value v = json::Value::Object();
    v.Set("index", json::Value::Int(s.id));
    v.Set("name", json::Value::Str(s.name));
    v.Set("type", json::Value::Str(SensorTypeName(s.type)));
    v.Set("mount", json::Value::Str(BodyPath(scene, body)));
    v.Set("frame", json::Value::Str(frame));
    v.Set("offset", TransformToJson(s.local_offset));
    v.Set("rate_hz", json::Value::Float(s.sample_rate_hz));
    v.Set("update_period", json::Value::Int(s.update_period));
    if (RenderSensor(s.type)) v.Set("aov_mask", json::Value::Int(s.aov_mask));
    if (CameraSensor(s.type)) {
        json::Value c = json::Value::Object();
        c.Set("width", json::Value::Int(s.cam.width));
        c.Set("height", json::Value::Int(s.cam.height));
        c.Set("vfov", json::Value::Float(s.cam.vfov_degrees));
        c.Set("near", json::Value::Float(s.cam.near_clip));
        c.Set("far", json::Value::Float(s.cam.far_clip));
        v.Set("camera", std::move(c));
    }
    if (LidarSensor(s.type)) {
        json::Value l = json::Value::Object();
        l.Set("az_count", json::Value::Int(s.lidar.az_count));
        l.Set("el_count", json::Value::Int(s.lidar.el_count));
        l.Set("az_min", json::Value::Float(s.lidar.az_min));
        l.Set("az_max", json::Value::Float(s.lidar.az_max));
        l.Set("el_min", json::Value::Float(s.lidar.el_min));
        l.Set("el_max", json::Value::Float(s.lidar.el_max));
        l.Set("min_range", json::Value::Float(s.lidar.min_range));
        l.Set("max_range", json::Value::Float(s.lidar.max_range));
        v.Set("lidar", std::move(l));
    }
    if (s.type == SensorType::Contact || s.type == SensorType::Tactile) {
        json::Value t = json::Value::Object();
        for (const auto& [shape, name] : kRegionNames)
            if (shape == s.tactile.shape) t.Set("shape", json::Value::Str(name));
        t.Set("size", Vec3Json(s.tactile.size));
        v.Set("tactile", std::move(t));
    }
    return v;
}

bool SensorFromJson(const json::Value& v, const scene::SceneIR& scene, scene::SensorDesc* s, std::string* error) {
    if (const json::Value* f = v.Find("name")) s->name = f->AsString();
    if (const json::Value* f = v.Find("type"); f != nullptr && !ParseSensorType(f->AsString(), &s->type)) {
        *error = "unknown sensor type '" + f->AsString() + "'";
        return false;
    }
    if (const json::Value* f = v.Find("mount")) {
        const std::string& path = f->AsString();
        const auto node = path.empty() ? nullptr : scene.Tree().NodeOf(path);
        scene::BodyId body = scene::kInvalidBody;
        for (scene::BodyId b = 0; node && b < scene.RigidBodyCount(); ++b)
            if (scene.EntityOfBody(b) == node->entity) body = b;
        if (!path.empty() && body == scene::kInvalidBody) {
            *error = "no body at '" + path + "'";
            return false;
        }
        s->mount = path.empty() ? scene::MountFrame::World : scene::MountFrame::Body;
        s->mount_index = body;
        s->joint_id = scene::kInvalidJoint;
    }
    if (const json::Value* f = v.Find("offset")) s->local_offset = TransformFromJson(*f);
    if (const json::Value* f = v.Find("position")) s->local_offset.position = Vec3FromJson(*f);
    if (const json::Value* f = v.Find("rotation_deg")) s->local_offset.rotation = QuatFromEulerDeg(Vec3FromJson(*f));
    ReadFloat(v, "rate_hz", &s->sample_rate_hz);
    if (const json::Value* f = v.Find("update_period")) s->update_period = static_cast<uint32_t>(f->AsInt());
    if (const json::Value* f = v.Find("aov_mask")) s->aov_mask = static_cast<uint32_t>(f->AsInt());
    if (const json::Value* c = v.Find("camera")) {
        ReadCount(*c, "width", &s->cam.width);
        ReadCount(*c, "height", &s->cam.height);
        ReadFloat(*c, "vfov", &s->cam.vfov_degrees);
        ReadFloat(*c, "near", &s->cam.near_clip);
        ReadFloat(*c, "far", &s->cam.far_clip);
    }
    if (const json::Value* l = v.Find("lidar")) {
        ReadCount(*l, "az_count", &s->lidar.az_count);
        ReadCount(*l, "el_count", &s->lidar.el_count);
        ReadFloat(*l, "az_min", &s->lidar.az_min);
        ReadFloat(*l, "az_max", &s->lidar.az_max);
        ReadFloat(*l, "el_min", &s->lidar.el_min);
        ReadFloat(*l, "el_max", &s->lidar.el_max);
        ReadFloat(*l, "min_range", &s->lidar.min_range);
        ReadFloat(*l, "max_range", &s->lidar.max_range);
    }
    if (const json::Value* t = v.Find("tactile")) {
        if (const json::Value* shape = t->Find("shape")) {
            const auto it = std::find_if(std::begin(kRegionNames), std::end(kRegionNames),
                                         [&](const auto& r) { return shape->AsString() == r.second; });
            if (it == std::end(kRegionNames)) {
                *error = "the contact region is box, sphere, ellipsoid, capsule or cylinder";
                return false;
            }
            s->tactile.shape = it->first;
        }
        if (const json::Value* size = t->Find("size")) s->tactile.size = Vec3FromJson(*size);
    }
    // State sensors read a body's motion; a joint state reads the joint its body hangs off.
    const std::string type = SensorTypeName(s->type);
    if (!RenderSensor(s->type) && s->mount == scene::MountFrame::World) {
        *error = type + " sensors mount on a body";
        return false;
    }
    if (s->type == SensorType::JointState && s->joint_id == scene::kInvalidJoint) {
        const scene::BodyId body = SensorBody(scene, *s);
        for (const scene::JointRecord& j : scene.Joints())
            if (body != scene::kInvalidBody && j.child_body == body) s->joint_id = j.id;
        if (s->joint_id == scene::kInvalidJoint) {
            *error = "a joint_state sensor mounts on a joint's child link";
            return false;
        }
    }
    return true;
}

std::string SensorShapeError(const std::vector<scene::SensorDesc>& sensors) {
    const scene::SensorDesc* camera = nullptr;
    const scene::SensorDesc* lidar = nullptr;
    for (const scene::SensorDesc& s : sensors) {
        if (CameraSensor(s.type)) {
            if (s.cam.width == 0u || s.cam.height == 0u) return "camera '" + s.name + "' has no image size";
            if (camera != nullptr && (s.cam.width != camera->cam.width || s.cam.height != camera->cam.height))
                return "every camera renders one image size: '" + s.name + "' is " + Size(s.cam.width, s.cam.height) +
                       ", '" + camera->name + "' is " + Size(camera->cam.width, camera->cam.height);
            if (camera == nullptr) camera = &s;
        } else if (LidarSensor(s.type)) {
            if (s.lidar.az_count == 0u || s.lidar.el_count == 0u) return "lidar '" + s.name + "' has no rays";
            if (lidar != nullptr && (s.lidar.az_count != lidar->lidar.az_count || s.lidar.el_count != lidar->lidar.el_count))
                return "every lidar casts one fan: '" + s.name + "' is " + Size(s.lidar.az_count, s.lidar.el_count) +
                       ", '" + lidar->name + "' is " + Size(lidar->lidar.az_count, lidar->lidar.el_count);
            if (lidar == nullptr) lidar = &s;
        }
    }
    return std::string();
}

void ShareSensorShape(const scene::SensorDesc& s, std::vector<scene::SensorDesc>* sensors) {
    for (scene::SensorDesc& o : *sensors) {
        if (CameraSensor(s.type) && CameraSensor(o.type)) {
            o.cam.width = s.cam.width;
            o.cam.height = s.cam.height;
        } else if (LidarSensor(s.type) && LidarSensor(o.type)) {
            o.lidar.az_count = s.lidar.az_count;
            o.lidar.el_count = s.lidar.el_count;
        }
    }
}

void NewSensorDefaults(const std::vector<scene::SensorDesc>& sensors, scene::SensorDesc* s) {
    constexpr float kPi = 3.14159265f;
    if (CameraSensor(s->type)) s->local_offset.rotation = math::Quat(0.5f, 0.5f, -0.5f, -0.5f);
    s->cam.width = 160u;
    s->cam.height = 120u;
    s->cam.vfov_degrees = 60.0f;
    s->cam.near_clip = 0.02f;
    s->cam.far_clip = 50.0f;
    s->lidar.az_count = 64u;
    s->lidar.el_count = 16u;
    s->lidar.az_min = -kPi;
    s->lidar.az_max = kPi;
    s->lidar.el_min = -0.26f;
    s->lidar.el_max = 0.26f;
    s->lidar.min_range = 0.05f;
    s->lidar.max_range = 30.0f;
    s->tactile.size = math::Vec3{0.05f, 0.05f, 0.05f};
    for (const scene::SensorDesc& o : sensors) {
        if (CameraSensor(o.type)) {
            s->cam.width = o.cam.width;
            s->cam.height = o.cam.height;
        } else if (LidarSensor(o.type)) {
            s->lidar.az_count = o.lidar.az_count;
            s->lidar.el_count = o.lidar.el_count;
        }
    }
}

std::string UniqueSensorName(const std::vector<scene::SensorDesc>& sensors, const std::string& base) {
    for (uint32_t n = 1u;; ++n) {
        const std::string name = base + "_" + std::to_string(n);
        if (std::none_of(sensors.begin(), sensors.end(), [&](const scene::SensorDesc& s) { return s.name == name; }))
            return name;
    }
}

}  // namespace nuka::runtime::app::viewer
