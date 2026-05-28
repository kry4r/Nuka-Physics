// ---------------------------------------------------------------------------
// nuka::import - USD / USDA importer implementation
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"

#include "import/usd_stage_adapter.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace nuka::import {

namespace {

struct UsdPrim {
    std::string type;
    std::string name;
    std::string path;
    std::string parent_path;
    bool has_rigid_body_api = false;
    bool has_collision_api = false;
    bool rigid_body_enabled = false;
    bool kinematic_enabled = false;
    float mass = 1.0f;
    math::Vec3 diagonal_inertia = {1.0f, 1.0f, 1.0f};
    math::Vec3 translate = math::Vec3::Zero();
    float cube_size = 1.0f;
    float radius = 0.5f;
    float height = 1.0f;
    math::Vec3 color = {1.0f, 1.0f, 1.0f};
    float alpha = 1.0f;
    float roughness = 0.5f;
    float metallic = 0.0f;
    float intensity = 1.0f;
    float focal_length = 50.0f;
    float near_clip = 0.01f;
    float far_clip = 1000.0f;
    std::string body0_path;
    std::string body1_path;
    std::string joint_path;
    std::string body_path;
    std::string axis_token = "Z";
    std::string nuka_type;
    float lower_limit = -3.14159f;
    float upper_limit = 3.14159f;
    float gain = 1.0f;
    float force_limit = 0.0f;
    float sample_rate_hz = 0.0f;
};

std::string Trim(std::string_view text) {
    size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin]))) {
        ++begin;
    }
    size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1]))) {
        --end;
    }
    return std::string(text.substr(begin, end - begin));
}

bool StartsWith(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

std::string Lowercase(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

bool ParseDefLine(const std::string& line, std::string& type, std::string& name) {
    if (!StartsWith(line, "def ")) {
        return false;
    }
    std::istringstream stream(line);
    std::string def_token;
    stream >> def_token >> type;
    const size_t first_quote = line.find('"');
    const size_t second_quote = line.find('"', first_quote + 1);
    if (first_quote == std::string::npos || second_quote == std::string::npos) {
        return false;
    }
    name = line.substr(first_quote + 1, second_quote - first_quote - 1);
    return !type.empty() && !name.empty();
}

void ApplyApiSchemas(const std::string& line, UsdPrim& prim) {
    if (line.find("apiSchemas") == std::string::npos) {
        return;
    }
    prim.has_rigid_body_api =
        prim.has_rigid_body_api || line.find("PhysicsRigidBodyAPI") != std::string::npos;
    prim.has_collision_api =
        prim.has_collision_api || line.find("PhysicsCollisionAPI") != std::string::npos;
}

bool ParseBoolValue(const std::string& line, std::string_view key, bool& value) {
    if (line.find(key) == std::string::npos) {
        return false;
    }
    const size_t equals = line.find('=');
    if (equals == std::string::npos) {
        return false;
    }
    const std::string rhs = Lowercase(Trim(std::string_view(line).substr(equals + 1)));
    if (StartsWith(rhs, "true")) {
        value = true;
        return true;
    }
    if (StartsWith(rhs, "false")) {
        value = false;
        return true;
    }
    return false;
}

bool ParseFloatValue(const std::string& line, std::string_view key, float& value) {
    if (line.find(key) == std::string::npos) {
        return false;
    }
    const size_t equals = line.find('=');
    if (equals == std::string::npos) {
        return false;
    }
    std::istringstream stream(Trim(std::string_view(line).substr(equals + 1)));
    stream >> value;
    return !stream.fail();
}

bool ParseVec3Value(const std::string& line, std::string_view key, math::Vec3& value) {
    if (line.find(key) == std::string::npos) {
        return false;
    }
    const size_t open = line.find('(');
    const size_t close = line.find(')', open + 1);
    if (open == std::string::npos || close == std::string::npos) {
        return false;
    }
    std::string tuple = line.substr(open + 1, close - open - 1);
    std::replace(tuple.begin(), tuple.end(), ',', ' ');
    std::istringstream stream(tuple);
    stream >> value.x >> value.y >> value.z;
    return !stream.fail();
}

bool ParseRelationship(const std::string& line, std::string_view key, std::string& path) {
    if (line.find(key) == std::string::npos) {
        return false;
    }
    const size_t open = line.find('<');
    const size_t close = line.find('>', open + 1);
    if (open == std::string::npos || close == std::string::npos) {
        return false;
    }
    path = line.substr(open + 1, close - open - 1);
    return !path.empty();
}

bool ParseTokenValue(const std::string& line, std::string_view key, std::string& value) {
    if (line.find(key) == std::string::npos) {
        return false;
    }
    const size_t first_quote = line.find('"');
    const size_t second_quote = line.find('"', first_quote + 1);
    if (first_quote == std::string::npos || second_quote == std::string::npos) {
        return false;
    }
    value = line.substr(first_quote + 1, second_quote - first_quote - 1);
    return !value.empty();
}

void ApplyPropertyLine(const std::string& line, UsdPrim& prim) {
    ApplyApiSchemas(line, prim);
    (void)ParseBoolValue(line, "physics:rigidBodyEnabled", prim.rigid_body_enabled);
    (void)ParseBoolValue(line, "physics:kinematicEnabled", prim.kinematic_enabled);
    (void)ParseFloatValue(line, "physics:mass", prim.mass);
    (void)ParseVec3Value(line, "physics:diagonalInertia", prim.diagonal_inertia);
    (void)ParseVec3Value(line, "xformOp:translate", prim.translate);
    (void)ParseFloatValue(line, "double size", prim.cube_size);
    (void)ParseFloatValue(line, "float size", prim.cube_size);
    (void)ParseFloatValue(line, "double radius", prim.radius);
    (void)ParseFloatValue(line, "float radius", prim.radius);
    (void)ParseFloatValue(line, "double height", prim.height);
    (void)ParseFloatValue(line, "float height", prim.height);
    (void)ParseVec3Value(line, "inputs:diffuseColor", prim.color);
    (void)ParseVec3Value(line, "inputs:color", prim.color);
    (void)ParseFloatValue(line, "inputs:alpha", prim.alpha);
    (void)ParseFloatValue(line, "inputs:roughness", prim.roughness);
    (void)ParseFloatValue(line, "inputs:metallic", prim.metallic);
    (void)ParseFloatValue(line, "inputs:intensity", prim.intensity);
    (void)ParseFloatValue(line, "focalLength", prim.focal_length);
    (void)ParseFloatValue(line, "clippingRange:min", prim.near_clip);
    (void)ParseFloatValue(line, "clippingRange:max", prim.far_clip);
    (void)ParseRelationship(line, "physics:body0", prim.body0_path);
    (void)ParseRelationship(line, "physics:body1", prim.body1_path);
    (void)ParseRelationship(line, "nuka:joint", prim.joint_path);
    (void)ParseRelationship(line, "nuka:body", prim.body_path);
    (void)ParseTokenValue(line, "physics:axis", prim.axis_token);
    (void)ParseTokenValue(line, "nuka:type", prim.nuka_type);
    (void)ParseFloatValue(line, "physics:lowerLimit", prim.lower_limit);
    (void)ParseFloatValue(line, "physics:upperLimit", prim.upper_limit);
    (void)ParseFloatValue(line, "nuka:gain", prim.gain);
    (void)ParseFloatValue(line, "nuka:forceLimit", prim.force_limit);
    (void)ParseFloatValue(line, "nuka:sampleRate", prim.sample_rate_hz);
}

std::string JoinPath(const std::string& parent, const std::string& name) {
    if (parent.empty() || parent == "/") {
        return "/" + name;
    }
    return parent + "/" + name;
}

std::vector<UsdPrim> ParseUsdaText(const std::string& text) {
    constexpr size_t kNoPendingPrim = static_cast<size_t>(-1);
    std::vector<UsdPrim> prims;
    std::vector<size_t> stack;
    size_t pending_prim = kNoPendingPrim;

    std::istringstream input(text);
    std::string raw_line;
    while (std::getline(input, raw_line)) {
        const std::string line = Trim(raw_line);
        if (line.empty() || StartsWith(line, "#")) {
            continue;
        }

        std::string prim_type;
        std::string prim_name;
        if (ParseDefLine(line, prim_type, prim_name)) {
            const std::string parent_path = stack.empty() ? std::string{} : prims[stack.back()].path;
            UsdPrim prim;
            prim.type = std::move(prim_type);
            prim.name = std::move(prim_name);
            prim.parent_path = parent_path;
            prim.path = JoinPath(parent_path, prim.name);
            prims.push_back(std::move(prim));
            pending_prim = prims.size() - 1;
            ApplyPropertyLine(line, prims[pending_prim]);
            if (line.find('{') != std::string::npos) {
                stack.push_back(pending_prim);
                pending_prim = kNoPendingPrim;
            }
            continue;
        }

        if (pending_prim != kNoPendingPrim) {
            ApplyPropertyLine(line, prims[pending_prim]);
        } else if (!stack.empty()) {
            ApplyPropertyLine(line, prims[stack.back()]);
        }
        if (line.find('{') != std::string::npos && pending_prim != kNoPendingPrim) {
            stack.push_back(pending_prim);
            pending_prim = kNoPendingPrim;
        }
        if (line.find('}') != std::string::npos && !stack.empty()) {
            stack.pop_back();
        }
    }
    return prims;
}

scene::BodyId FindNearestBodyAncestor(
    std::string path,
    const std::unordered_map<std::string, scene::BodyId>& body_map) {

    while (!path.empty()) {
        const auto it = body_map.find(path);
        if (it != body_map.end()) {
            return it->second;
        }
        const size_t slash = path.find_last_of('/');
        if (slash == std::string::npos || slash == 0) {
            break;
        }
        path = path.substr(0, slash);
    }
    return scene::kInvalidBody;
}

bool IsRigidBodyPrim(const UsdPrim& prim) {
    return prim.has_rigid_body_api && prim.rigid_body_enabled;
}

bool ShapeTypeFromUsdPrim(const UsdPrim& prim, scene::ShapeType& shape_type) {
    if (!prim.has_collision_api) {
        return false;
    }
    if (prim.type == "Cube") {
        shape_type = scene::ShapeType::Box;
        return true;
    }
    if (prim.type == "Sphere") {
        shape_type = scene::ShapeType::Sphere;
        return true;
    }
    if (prim.type == "Capsule" || prim.type == "Cylinder") {
        shape_type = scene::ShapeType::Capsule;
        return true;
    }
    if (prim.type == "Mesh") {
        shape_type = scene::ShapeType::TriMesh;
        return true;
    }
    return false;
}

bool IsJointPrim(const UsdPrim& prim) {
    return StartsWith(prim.type, "Physics") && prim.type.find("Joint") != std::string::npos;
}

scene::JointType JointTypeFromUsdType(const std::string& type) {
    if (type == "PhysicsRevoluteJoint") {
        return scene::JointType::Revolute;
    }
    if (type == "PhysicsPrismaticJoint") {
        return scene::JointType::Prismatic;
    }
    if (type == "PhysicsSphericalJoint") {
        return scene::JointType::Spherical;
    }
    if (type == "PhysicsFixedJoint") {
        return scene::JointType::Fixed;
    }
    return scene::JointType::Fixed;
}

math::Vec3 AxisFromUsdToken(const std::string& axis) {
    if (axis == "X" || axis == "x") {
        return math::Vec3::UnitX();
    }
    if (axis == "Y" || axis == "y") {
        return math::Vec3::UnitY();
    }
    return math::Vec3::UnitZ();
}

scene::ActuatorType ActuatorTypeFromToken(const std::string& token) {
    const std::string lower = Lowercase(token);
    if (lower == "position") {
        return scene::ActuatorType::Position;
    }
    if (lower == "velocity") {
        return scene::ActuatorType::Velocity;
    }
    if (lower == "force") {
        return scene::ActuatorType::Force;
    }
    return scene::ActuatorType::Motor;
}

scene::SensorType SensorTypeFromToken(const std::string& token) {
    const std::string lower = Lowercase(token);
    if (lower == "lidar") {
        return scene::SensorType::Lidar;
    }
    if (lower == "camera") {
        return scene::SensorType::Camera;
    }
    if (lower == "force_torque" || lower == "forcetorque") {
        return scene::SensorType::ForceTorque;
    }
    if (lower == "contact") {
        return scene::SensorType::Contact;
    }
    if (lower == "framepose") {
        return scene::SensorType::FramePose;
    }
    return scene::SensorType::Imu;
}

scene::LightType LightTypeFromUsdType(const std::string& type) {
    if (type == "DistantLight") {
        return scene::LightType::Directional;
    }
    if (type == "RectLight" || type == "DiskLight") {
        return scene::LightType::Area;
    }
    return scene::LightType::Point;
}

scene::SceneIR BuildSceneFromUsdPrims(const std::vector<UsdPrim>& prims) {
    scene::SceneIR scene;
    std::unordered_map<std::string, scene::BodyId> body_map;
    std::unordered_map<std::string, scene::JointId> joint_map;

    for (const auto& prim : prims) {
        if (prim.type == "Material") {
            scene::MaterialRecord material;
            material.name = prim.name;
            material.base_color = prim.color;
            material.alpha = prim.alpha;
            material.roughness = prim.roughness;
            material.metallic = prim.metallic;
            scene.AddMaterial(std::move(material));
        }
    }

    for (const auto& prim : prims) {
        if (!IsRigidBodyPrim(prim)) {
            continue;
        }
        scene::RigidBodyRecord body;
        body.name = prim.name;
        body.parent_id = FindNearestBodyAncestor(prim.parent_path, body_map);
        body.local_transform = math::Transform{prim.translate, math::Quat::Identity()};
        body.mass = prim.mass;
        body.inertia = prim.diagonal_inertia;
        body.is_static = prim.kinematic_enabled || prim.mass <= 0.0f;
        const scene::BodyId body_id = scene.AddRigidBody(std::move(body));
        body_map[prim.path] = body_id;
    }

    for (const auto& prim : prims) {
        scene::ShapeType shape_type = scene::ShapeType::Box;
        if (!ShapeTypeFromUsdPrim(prim, shape_type)) {
            continue;
        }
        const scene::BodyId body_id = FindNearestBodyAncestor(prim.parent_path, body_map);
        if (body_id == scene::kInvalidBody) {
            continue;
        }
        scene::CollisionShapeRecord shape;
        shape.name = prim.name;
        shape.body_id = body_id;
        shape.type = shape_type;
        shape.local_transform = math::Transform{prim.translate, math::Quat::Identity()};
        if (shape_type == scene::ShapeType::Box) {
            const float half_size = prim.cube_size * 0.5f;
            shape.half_extents = {half_size, half_size, half_size};
        } else if (shape_type == scene::ShapeType::Sphere) {
            shape.radius = prim.radius;
        } else if (shape_type == scene::ShapeType::Capsule) {
            shape.radius = prim.radius;
            shape.half_height = prim.height * 0.5f;
        }
        scene.AddCollisionShape(std::move(shape));
    }

    for (const auto& prim : prims) {
        if (!IsJointPrim(prim)) {
            continue;
        }
        const auto parent_it = body_map.find(prim.body0_path);
        const auto child_it = body_map.find(prim.body1_path);
        if (parent_it == body_map.end() || child_it == body_map.end()) {
            throw std::runtime_error("USD: joint '" + prim.name + "' references an unknown rigid body");
        }
        scene::JointRecord joint;
        joint.name = prim.name;
        joint.type = JointTypeFromUsdType(prim.type);
        joint.parent_body = parent_it->second;
        joint.child_body = child_it->second;
        joint.axis = AxisFromUsdToken(prim.axis_token);
        joint.lower_limit = prim.lower_limit;
        joint.upper_limit = prim.upper_limit;
        const scene::JointId joint_id = scene.AddJoint(std::move(joint));
        joint_map[prim.path] = joint_id;
    }

    for (const auto& prim : prims) {
        if (prim.type == "Camera") {
            scene::CameraRecord camera;
            camera.name = prim.name;
            camera.attached_body = FindNearestBodyAncestor(prim.parent_path, body_map);
            camera.local_transform = math::Transform{prim.translate, math::Quat::Identity()};
            camera.vertical_fov_degrees = prim.focal_length > 0.0f ? 50.0f : 45.0f;
            camera.near_clip = prim.near_clip;
            camera.far_clip = prim.far_clip;
            scene.AddCamera(std::move(camera));
        } else if (prim.type.find("Light") != std::string::npos) {
            scene::LightRecord light;
            light.name = prim.name;
            light.type = LightTypeFromUsdType(prim.type);
            light.local_transform = math::Transform{prim.translate, math::Quat::Identity()};
            light.color = prim.color;
            light.intensity = prim.intensity;
            scene.AddLight(std::move(light));
        } else if (prim.type == "NukaActuator") {
            const auto joint_it = joint_map.find(prim.joint_path);
            if (joint_it == joint_map.end()) {
                throw std::runtime_error("USD: actuator '" + prim.name + "' references an unknown joint");
            }
            scene::ActuatorRecord actuator;
            actuator.name = prim.name;
            actuator.joint_id = joint_it->second;
            actuator.type = ActuatorTypeFromToken(prim.nuka_type);
            actuator.gain = prim.gain;
            actuator.force_limit = prim.force_limit;
            scene.AddActuator(std::move(actuator));
        } else if (prim.type == "NukaSensor") {
            const auto body_it = body_map.find(prim.body_path);
            if (body_it == body_map.end()) {
                throw std::runtime_error("USD: sensor '" + prim.name + "' references an unknown body");
            }
            scene::SensorRecord sensor;
            sensor.name = prim.name;
            sensor.attached_body = body_it->second;
            sensor.type = SensorTypeFromToken(prim.nuka_type);
            sensor.sample_rate_hz = prim.sample_rate_hz;
            scene.AddSensor(std::move(sensor));
        }
    }
    return scene;
}

} // namespace

scene::SceneIR LoadUsd(const std::string& path) {
    const UsdStageData stage = LoadUsdStageData(path);
    auto scene = BuildSceneFromUsdPrims(ParseUsdaText(stage.text));
    if (scene.RigidBodyCount() == 0u) {
        throw std::runtime_error("USD: no enabled UsdPhysics rigid bodies found in " + path);
    }
    return scene;
}

} // namespace nuka::import
