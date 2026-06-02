// ---------------------------------------------------------------------------
// nuka::import - MJCF (MuJoCo XML) importer implementation
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "scene/canonical_types.hpp"

#include <tinyxml2.h>

#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace nuka::import {

namespace {

math::Vec3 ParseVec3(const char* text) {
    math::Vec3 v{};
    if (!text) {
        return v;
    }
    std::istringstream ss(text);
    ss >> v.x >> v.y >> v.z;
    return v;
}

math::Quat ParseQuat(const char* text) {
    math::Quat q = math::Quat::Identity();
    if (!text) {
        return q;
    }
    std::istringstream ss(text);
    ss >> q.w >> q.x >> q.y >> q.z;
    return q.Normalized();
}

void ParseRange(const char* text, float& lower, float& upper) {
    if (!text) {
        return;
    }
    std::istringstream ss(text);
    ss >> lower >> upper;
}

struct MjcfJointDefaults {
    scene::JointType type = scene::JointType::Revolute;
    math::Vec3 axis = {0.0f, 1.0f, 0.0f};
    float lower_limit = -3.14159f;
    float upper_limit = 3.14159f;
    float damping = 0.0f;
    float armature = 0.0f;
};

struct MjcfGeneralDefaults {
    float gain = 1.0f;
    float force_limit = 0.0f;
};

struct MjcfDefaultClass {
    MjcfJointDefaults joint;
    MjcfGeneralDefaults general;
};

struct MjcfDefaults {
    MjcfDefaultClass root;
    std::unordered_map<std::string, MjcfDefaultClass> classes;
};

struct MjcfParseContext {
    std::unordered_map<std::string, scene::BodyId> body_ids;
    std::unordered_map<std::string, scene::JointId> joint_ids;
    std::unordered_map<std::string, scene::MaterialId> material_ids;
    MjcfDefaults defaults;
};

scene::ShapeType MjcfGeomType(const char* type_str) {
    if (!type_str) {
        return scene::ShapeType::Box;
    }
    const std::string t(type_str);
    if (t == "sphere") {
        return scene::ShapeType::Sphere;
    }
    if (t == "capsule" || t == "cylinder") {
        return scene::ShapeType::Capsule;
    }
    if (t == "box") {
        return scene::ShapeType::Box;
    }
    if (t == "plane") {
        return scene::ShapeType::Plane;
    }
    if (t == "mesh") {
        return scene::ShapeType::TriMesh;
    }
    return scene::ShapeType::Box;
}

scene::DecomposeMode DecomposeModeFromToken(const char* token) {
    if (!token) {
        return scene::DecomposeMode::Auto;
    }
    const std::string t(token);
    if (t == "force") {
        return scene::DecomposeMode::Force;
    }
    if (t == "skip") {
        return scene::DecomposeMode::Skip;
    }
    return scene::DecomposeMode::Auto;
}

scene::JointType MjcfJointType(const char* type_str) {
    if (!type_str) {
        return scene::JointType::Revolute;
    }
    const std::string t(type_str);
    if (t == "hinge") {
        return scene::JointType::Revolute;
    }
    if (t == "slide") {
        return scene::JointType::Prismatic;
    }
    if (t == "ball") {
        return scene::JointType::Spherical;
    }
    if (t == "free") {
        return scene::JointType::Free;
    }
    return scene::JointType::Revolute;
}

const MjcfDefaultClass& DefaultClassOrRoot(const MjcfDefaults& defaults,
                                           const char* class_name) {
    if (class_name) {
        const auto it = defaults.classes.find(class_name);
        if (it != defaults.classes.end()) {
            return it->second;
        }
    }
    return defaults.root;
}

void ApplyJointDefault(tinyxml2::XMLElement* joint_elem, MjcfDefaultClass* defaults) {
    if (joint_elem == nullptr || defaults == nullptr) {
        return;
    }
    if (joint_elem->Attribute("type")) {
        defaults->joint.type = MjcfJointType(joint_elem->Attribute("type"));
    }
    if (const char* axis = joint_elem->Attribute("axis")) {
        defaults->joint.axis = ParseVec3(axis);
    }
    ParseRange(joint_elem->Attribute("range"),
               defaults->joint.lower_limit,
               defaults->joint.upper_limit);
    joint_elem->QueryFloatAttribute("damping", &defaults->joint.damping);
    joint_elem->QueryFloatAttribute("armature", &defaults->joint.armature);
}

void ApplyGeneralDefault(tinyxml2::XMLElement* general_elem, MjcfDefaultClass* defaults) {
    if (general_elem == nullptr || defaults == nullptr) {
        return;
    }
    general_elem->QueryFloatAttribute("gear", &defaults->general.gain);
    ParseRange(general_elem->Attribute("forcerange"),
               defaults->general.force_limit,
               defaults->general.force_limit);
}

void ParseDefaultElement(tinyxml2::XMLElement* default_elem,
                         const MjcfDefaultClass& inherited,
                         MjcfDefaults* defaults) {
    if (default_elem == nullptr || defaults == nullptr) {
        return;
    }

    MjcfDefaultClass current = inherited;
    ApplyJointDefault(default_elem->FirstChildElement("joint"), &current);
    ApplyGeneralDefault(default_elem->FirstChildElement("general"), &current);

    if (const char* class_name = default_elem->Attribute("class")) {
        defaults->classes[class_name] = current;
    } else {
        defaults->root = current;
    }

    for (auto* child = default_elem->FirstChildElement("default");
         child != nullptr;
         child = child->NextSiblingElement("default")) {
        ParseDefaultElement(child, current, defaults);
    }
}

void ParseDefaults(tinyxml2::XMLElement* mujoco, MjcfParseContext& context) {
    for (auto* default_elem = mujoco->FirstChildElement("default");
         default_elem != nullptr;
         default_elem = default_elem->NextSiblingElement("default")) {
        ParseDefaultElement(default_elem, context.defaults.root, &context.defaults);
    }
}

scene::SensorType MjcfSensorType(const std::string& tag) {
    if (tag == "rangefinder") {
        return scene::SensorType::Lidar;
    }
    if (tag == "camera") {
        return scene::SensorType::Camera;
    }
    if (tag == "force" || tag == "torque") {
        return scene::SensorType::ForceTorque;
    }
    if (tag == "touch") {
        return scene::SensorType::Contact;
    }
    if (tag == "framepos" || tag == "framequat") {
        return scene::SensorType::Imu;
    }
    return scene::SensorType::Imu;
}

void ParseMaterials(tinyxml2::XMLElement* mujoco,
                    scene::SceneIR& scene,
                    MjcfParseContext& context) {
    auto* asset = mujoco->FirstChildElement("asset");
    if (!asset) {
        return;
    }

    for (auto* material = asset->FirstChildElement("material");
         material != nullptr;
         material = material->NextSiblingElement("material")) {

        scene::MaterialRecord record;
        const char* name = material->Attribute("name");
        record.name = name ? name : "material";

        if (const char* rgba = material->Attribute("rgba")) {
            std::istringstream ss(rgba);
            ss >> record.base_color.x >> record.base_color.y >> record.base_color.z >> record.alpha;
        }
        material->QueryFloatAttribute("roughness", &record.roughness);
        material->QueryFloatAttribute("metallic", &record.metallic);

        const scene::MaterialId id = scene.AddMaterial(std::move(record));
        context.material_ids[scene.GetMaterial(id).name] = id;
    }
}

scene::BodyId ResolveBody(const char* name, const MjcfParseContext& context) {
    if (!name) {
        return scene::kInvalidBody;
    }
    const auto it = context.body_ids.find(name);
    return it == context.body_ids.end() ? scene::kInvalidBody : it->second;
}

scene::JointId ResolveJoint(const char* name, const MjcfParseContext& context) {
    if (!name) {
        return scene::kInvalidJoint;
    }
    const auto it = context.joint_ids.find(name);
    return it == context.joint_ids.end() ? scene::kInvalidJoint : it->second;
}

void ParseBody(tinyxml2::XMLElement* body_elem,
               scene::BodyId parent_id,
               scene::SceneIR& scene,
               MjcfParseContext& context,
               const char* inherited_child_class) {

    const char* name_attr = body_elem->Attribute("name");
    const std::string body_name = name_attr ? name_attr : "unnamed";
    const char* child_class = body_elem->Attribute("childclass");
    if (!child_class) {
        child_class = inherited_child_class;
    }

    scene::RigidBodyRecord rec;
    rec.name = body_name;
    rec.parent_id = parent_id;
    if (const char* pos_attr = body_elem->Attribute("pos")) {
        rec.local_transform.position = ParseVec3(pos_attr);
    }
    if (const char* quat_attr = body_elem->Attribute("quat")) {
        rec.local_transform.rotation = ParseQuat(quat_attr);
    }

    if (auto* inertial = body_elem->FirstChildElement("inertial")) {
        inertial->QueryFloatAttribute("mass", &rec.mass);
        if (const char* pos = inertial->Attribute("pos")) {
            rec.inertial_transform.position = ParseVec3(pos);
        }
        if (const char* quat = inertial->Attribute("quat")) {
            rec.inertial_transform.rotation = ParseQuat(quat);
        }
        if (const char* diag = inertial->Attribute("diaginertia")) {
            rec.inertia = ParseVec3(diag);
        }
    }

    const scene::BodyId body_id = scene.AddRigidBody(std::move(rec));
    context.body_ids[body_name] = body_id;

    for (auto* geom = body_elem->FirstChildElement("geom");
         geom != nullptr;
         geom = geom->NextSiblingElement("geom")) {

        scene::CollisionShapeRecord shape;
        shape.body_id = body_id;
        shape.name = geom->Attribute("name") ? geom->Attribute("name") : "";
        shape.type = MjcfGeomType(geom->Attribute("type"));

        if (const char* material_name = geom->Attribute("material")) {
            const auto it = context.material_ids.find(material_name);
            if (it != context.material_ids.end()) {
                shape.material_id = it->second;
            }
        }

        if (const char* pos = geom->Attribute("pos")) {
            shape.local_transform.position = ParseVec3(pos);
        }
        if (const char* quat = geom->Attribute("quat")) {
            shape.local_transform.rotation = ParseQuat(quat);
        }

        if (const char* size_attr = geom->Attribute("size")) {
            const math::Vec3 sz = ParseVec3(size_attr);
            if (shape.type == scene::ShapeType::Box || shape.type == scene::ShapeType::Plane) {
                shape.half_extents = sz;
            } else if (shape.type == scene::ShapeType::Sphere) {
                shape.radius = sz.x;
            } else if (shape.type == scene::ShapeType::Capsule) {
                shape.radius = sz.x;
                if (sz.y > 0.0f) {
                    shape.half_height = sz.y;
                }
            }
        }

        // nuka:decompose="auto|force|skip" + optional nuka:decompose:max_pieces
        // on a mesh geom (v0.7 p06).
        if (shape.type == scene::ShapeType::TriMesh) {
            shape.decompose_mode = DecomposeModeFromToken(geom->Attribute("nuka:decompose"));
            int max_pieces = 0;
            if (geom->QueryIntAttribute("nuka:decompose:max_pieces", &max_pieces) ==
                    tinyxml2::XML_SUCCESS && max_pieces > 0) {
                shape.decompose_max_pieces = static_cast<uint32_t>(max_pieces);
            }
        }

        scene.AddCollisionShape(std::move(shape));
    }

    for (auto* joint = body_elem->FirstChildElement("joint");
         joint != nullptr;
         joint = joint->NextSiblingElement("joint")) {

        const char* jname = joint->Attribute("name");
        const char* joint_class = joint->Attribute("class");
        const MjcfDefaultClass& defaults =
            DefaultClassOrRoot(context.defaults, joint_class ? joint_class : child_class);
        scene::JointRecord jrec;
        jrec.name = jname ? jname : "unnamed_joint";
        jrec.parent_body = parent_id;
        jrec.child_body = body_id;
        jrec.type = defaults.joint.type;
        jrec.axis = defaults.joint.axis;
        jrec.lower_limit = defaults.joint.lower_limit;
        jrec.upper_limit = defaults.joint.upper_limit;
        jrec.damping = defaults.joint.damping;
        jrec.armature = defaults.joint.armature;
        if (joint->Attribute("type")) {
            jrec.type = MjcfJointType(joint->Attribute("type"));
        }
        if (const char* axis_attr = joint->Attribute("axis")) {
            jrec.axis = ParseVec3(axis_attr);
        }
        if (const char* pos_attr = joint->Attribute("pos")) {
            jrec.parent_frame.position = ParseVec3(pos_attr);
        }
        ParseRange(joint->Attribute("range"), jrec.lower_limit, jrec.upper_limit);

        const scene::JointId joint_id = scene.AddJoint(std::move(jrec));
        context.joint_ids[scene.GetJoint(joint_id).name] = joint_id;
    }

    for (auto* camera = body_elem->FirstChildElement("camera");
         camera != nullptr;
         camera = camera->NextSiblingElement("camera")) {

        scene::CameraRecord record;
        record.name = camera->Attribute("name") ? camera->Attribute("name") : "camera";
        record.attached_body = body_id;
        if (const char* pos = camera->Attribute("pos")) {
            record.local_transform.position = ParseVec3(pos);
        }
        if (const char* quat = camera->Attribute("quat")) {
            record.local_transform.rotation = ParseQuat(quat);
        }
        camera->QueryFloatAttribute("fovy", &record.vertical_fov_degrees);
        scene.AddCamera(std::move(record));
    }

    for (auto* child = body_elem->FirstChildElement("body");
         child != nullptr;
         child = child->NextSiblingElement("body")) {
        ParseBody(child, body_id, scene, context, child_class);
    }
}

void ParseLights(tinyxml2::XMLElement* worldbody, scene::SceneIR& scene) {
    for (auto* light = worldbody->FirstChildElement("light");
         light != nullptr;
         light = light->NextSiblingElement("light")) {

        scene::LightRecord record;
        record.name = light->Attribute("name") ? light->Attribute("name") : "light";
        bool directional = false;
        light->QueryBoolAttribute("directional", &directional);
        record.type = directional ? scene::LightType::Directional : scene::LightType::Point;
        if (const char* pos = light->Attribute("pos")) {
            record.local_transform.position = ParseVec3(pos);
        }
        if (const char* quat = light->Attribute("quat")) {
            record.local_transform.rotation = ParseQuat(quat);
        }
        if (const char* diffuse = light->Attribute("diffuse")) {
            record.color = ParseVec3(diffuse);
        }
        light->QueryFloatAttribute("intensity", &record.intensity);
        scene.AddLight(std::move(record));
    }
}

void ParseActuators(tinyxml2::XMLElement* mujoco,
                    scene::SceneIR& scene,
                    const MjcfParseContext& context) {
    auto* actuator_root = mujoco->FirstChildElement("actuator");
    if (!actuator_root) {
        return;
    }

    for (auto* actuator = actuator_root->FirstChildElement();
         actuator != nullptr;
         actuator = actuator->NextSiblingElement()) {

        scene::ActuatorRecord record;
        record.name = actuator->Attribute("name") ? actuator->Attribute("name") : "actuator";
        const MjcfDefaultClass& defaults =
            DefaultClassOrRoot(context.defaults, actuator->Attribute("class"));
        record.gain = defaults.general.gain;
        record.force_limit = defaults.general.force_limit;
        const std::string tag = actuator->Name() ? actuator->Name() : "";
        if (tag == "position") {
            record.type = scene::ActuatorType::Position;
        } else if (tag == "velocity") {
            record.type = scene::ActuatorType::Velocity;
        } else if (tag == "motor" || tag == "general") {
            record.type = scene::ActuatorType::Motor;
        } else {
            record.type = scene::ActuatorType::Force;
        }
        record.joint_id = ResolveJoint(actuator->Attribute("joint"), context);
        actuator->QueryFloatAttribute("gear", &record.gain);
        ParseRange(actuator->Attribute("forcerange"), record.force_limit, record.force_limit);
        scene.AddActuator(std::move(record));
    }
}

void ParseSensors(tinyxml2::XMLElement* mujoco,
                  scene::SceneIR& scene,
                  const MjcfParseContext& context) {
    auto* sensor_root = mujoco->FirstChildElement("sensor");
    if (!sensor_root) {
        return;
    }

    for (auto* sensor = sensor_root->FirstChildElement();
         sensor != nullptr;
         sensor = sensor->NextSiblingElement()) {

        const std::string tag = sensor->Name() ? sensor->Name() : "";
        scene::SensorRecord record;
        record.name = sensor->Attribute("name") ? sensor->Attribute("name") : "sensor";
        record.type = MjcfSensorType(tag);

        const char* body_name = sensor->Attribute("objname");
        if (!body_name) {
            body_name = sensor->Attribute("body");
        }
        record.attached_body = ResolveBody(body_name, context);
        scene.AddSensor(std::move(record));
    }
}

} // namespace

scene::SceneIR LoadMjcf(const std::string& path) {
    tinyxml2::XMLDocument doc;
    const tinyxml2::XMLError err = doc.LoadFile(path.c_str());
    if (err != tinyxml2::XML_SUCCESS) {
        throw std::runtime_error("MJCF: failed to load file: " + path +
                                 " (error " + std::to_string(static_cast<int>(err)) + ")");
    }

    auto* mujoco = doc.FirstChildElement("mujoco");
    if (!mujoco) {
        throw std::runtime_error("MJCF: missing <mujoco> root element in " + path);
    }

    auto* worldbody = mujoco->FirstChildElement("worldbody");
    if (!worldbody) {
        throw std::runtime_error("MJCF: missing <worldbody> element in " + path);
    }

    scene::SceneIR scene;
    MjcfParseContext context;

    ParseDefaults(mujoco, context);
    ParseMaterials(mujoco, scene, context);
    ParseLights(worldbody, scene);

    for (auto* body = worldbody->FirstChildElement("body");
         body != nullptr;
         body = body->NextSiblingElement("body")) {
        ParseBody(body, scene::kInvalidBody, scene, context, nullptr);
    }

    ParseActuators(mujoco, scene, context);
    ParseSensors(mujoco, scene, context);

    return scene;
}

} // namespace nuka::import
