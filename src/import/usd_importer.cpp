// ---------------------------------------------------------------------------
// nuka::import - USD importer: reads the stage OpenUSD composed into SceneIR
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"

#include "import/usd_stage_adapter.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"

#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3h.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/value.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/relationship.h>
#include <pxr/usd/usdGeom/xformCache.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace nuka::import {

namespace {

// Everything the scene builder reads from one composed prim; unauthored values keep these defaults.
struct PrimData {
    std::string type;
    std::string name;
    std::string path;
    std::string parent_path;
    bool has_rigid_body_api = false;
    bool has_collision_api = false;
    bool rigid_body_enabled = false;
    bool kinematic_enabled = false;
    bool exclude_from_articulation = false;  // physics:excludeFromArticulation
    bool joint_enabled = true;               // physics:jointEnabled
    float mass = 1.0f;
    math::Vec3 diagonal_inertia = {1.0f, 1.0f, 1.0f};
    math::Vec3 translate = math::Vec3::Zero();
    math::Vec3 scale = {1.0f, 1.0f, 1.0f};
    // xformOp:rotateXYZ Euler angles (degrees, applied X then Y then Z).
    math::Vec3 rotate_xyz = math::Vec3::Zero();
    float cube_size = 1.0f;
    float radius = 0.5f;
    float height = 1.0f;
    math::Vec3 color = {1.0f, 1.0f, 1.0f};
    float alpha = 1.0f;
    float roughness = 0.5f;
    float metallic = 0.0f;
    bool has_friction = false;           // physics:dynamicFriction authored?
    float friction_mu = 1.0f;            // UsdPhysics PhysicsMaterialAPI friction
    float intensity = 1.0f;
    float exposure = 0.0f;               // UsdLux inputs:exposure (stops)
    float focal_length = 50.0f;
    bool has_focal_length = false;        // UsdGeomCamera focalLength authored?
    // UsdGeomCamera verticalAperture (mm); USD default 15.2908 == 0.602362"*25.4.
    float vertical_aperture = 15.2908f;
    float near_clip = 0.01f;
    float far_clip = 1000.0f;
    std::string body0_path;
    std::string body1_path;
    std::string joint_path;
    std::string body_path;
    std::string axis_token = "Z";
    std::string geometry_axis_token = "Z";
    std::string nuka_type;
    std::string purpose;                 // USD `purpose` token: default|render|proxy|guide
    std::string decompose_mode;          // nuka:decompose token
    int decompose_max_pieces = 32;       // nuka:decompose:max_pieces
    bool has_initial_position = false;
    bool has_lower_limit = false;
    bool has_upper_limit = false;
    float initial_position = 0.0f;
    float lower_limit = -3.14159f;
    float upper_limit = 3.14159f;
    float gain = 1.0f;
    float force_limit = 0.0f;
    float sample_rate_hz = 0.0f;

    // NukaSensor lidar / depth payload; defaults match SensorDesc.
    float sensor_h_fov = 0.0f;           // horizontal fan extent (rad)
    float sensor_v_fov = 0.0f;           // vertical fan extent (rad)
    int sensor_h_res = 0;                // horizontal ray count
    int sensor_v_res = 0;                // vertical ray count
    float sensor_min_range = 0.0f;
    float sensor_max_range = 100.0f;
    int sensor_width = 0;                // depth image width
    int sensor_height = 0;               // depth image height

    // Mesh geometry, triangulated at SceneIR-build time: flat x,y,z points in authored order.
    std::vector<float> mesh_points;
    std::vector<int> face_vertex_indices;
    std::vector<int> face_vertex_counts;
    pxr::GfMatrix4d world{1.0};          // a Mesh's full local-to-world transform
};

// Time-sampled data (Alembic, animated USD) imports at its first sample.
const pxr::UsdTimeCode kTime = pxr::UsdTimeCode::EarliestTime();

std::string Lowercase(std::string_view text) {
    std::string out(text);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

bool StartsWith(std::string_view text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.substr(0, prefix.size()) == prefix;
}

// The authored value of `name` on `prim`; empty when the attribute is absent or never authored.
pxr::VtValue Authored(const pxr::UsdPrim& prim, const char* name) {
    pxr::VtValue value;
    const pxr::UsdAttribute attr = prim.GetAttribute(pxr::TfToken(name));
    if (attr && attr.HasAuthoredValue()) {
        attr.Get(&value, kTime);
    }
    return value;
}

template <typename T>
bool Read(const pxr::UsdPrim& prim, const char* name, T& out) {
    const pxr::VtValue value = pxr::VtValue::Cast<T>(Authored(prim, name));
    if (value.IsEmpty()) {
        return false;
    }
    out = value.UncheckedGet<T>();
    return true;
}

bool Read(const pxr::UsdPrim& prim, const char* name, float& out) {
    double value = 0.0;
    if (!Read<double>(prim, name, value)) {
        return false;
    }
    out = static_cast<float>(value);
    return true;
}

template <typename V>
bool AsVec3(const pxr::VtValue& value, math::Vec3& out) {
    if (!value.IsHolding<V>()) {
        return false;
    }
    const V& v = value.UncheckedGet<V>();
    out = {static_cast<float>(v[0]), static_cast<float>(v[1]), static_cast<float>(v[2])};
    return true;
}

bool Read(const pxr::UsdPrim& prim, const char* name, math::Vec3& out) {
    const pxr::VtValue value = Authored(prim, name);
    return AsVec3<pxr::GfVec3f>(value, out) || AsVec3<pxr::GfVec3d>(value, out) ||
           AsVec3<pxr::GfVec3h>(value, out);
}

bool Read(const pxr::UsdPrim& prim, const char* name, std::string& out) {
    const pxr::VtValue value = Authored(prim, name);
    if (value.IsHolding<pxr::TfToken>()) {
        out = value.UncheckedGet<pxr::TfToken>().GetString();
    } else if (value.IsHolding<std::string>()) {
        out = value.UncheckedGet<std::string>();
    } else {
        return false;
    }
    return !out.empty();
}

// The first target of relationship `name`, as an absolute prim path.
void ReadTarget(const pxr::UsdPrim& prim, const char* name, std::string& out) {
    pxr::SdfPathVector targets;
    const pxr::UsdRelationship rel = prim.GetRelationship(pxr::TfToken(name));
    if (rel && rel.GetTargets(&targets) && !targets.empty()) {
        out = targets.front().GetString();
    }
}

PrimData ReadPrim(const pxr::UsdPrim& prim, pxr::UsdGeomXformCache& xforms) {
    PrimData d;
    d.type = prim.GetTypeName().GetString();
    d.name = prim.GetName().GetString();
    d.path = prim.GetPath().GetString();
    d.parent_path = prim.GetParent().GetPath().GetString();
    for (const pxr::TfToken& api : prim.GetAppliedSchemas()) {
        d.has_rigid_body_api = d.has_rigid_body_api || api == "PhysicsRigidBodyAPI";
        d.has_collision_api = d.has_collision_api || api == "PhysicsCollisionAPI";
    }
    Read(prim, "physics:rigidBodyEnabled", d.rigid_body_enabled);
    Read(prim, "physics:kinematicEnabled", d.kinematic_enabled);
    Read(prim, "physics:excludeFromArticulation", d.exclude_from_articulation);
    Read(prim, "physics:jointEnabled", d.joint_enabled);
    Read(prim, "physics:mass", d.mass);
    Read(prim, "physics:diagonalInertia", d.diagonal_inertia);
    Read(prim, "xformOp:translate", d.translate);
    Read(prim, "xformOp:scale", d.scale);
    Read(prim, "xformOp:rotateXYZ", d.rotate_xyz);
    Read(prim, "size", d.cube_size);
    Read(prim, "radius", d.radius);
    Read(prim, "height", d.height);
    if (!Read(prim, "inputs:color", d.color)) {
        Read(prim, "inputs:diffuseColor", d.color);
    }
    Read(prim, "inputs:alpha", d.alpha);
    Read(prim, "inputs:roughness", d.roughness);
    Read(prim, "inputs:metallic", d.metallic);
    d.has_friction = Read(prim, "physics:dynamicFriction", d.friction_mu);
    Read(prim, "inputs:intensity", d.intensity);
    Read(prim, "inputs:exposure", d.exposure);
    d.has_focal_length = Read(prim, "focalLength", d.focal_length);
    Read(prim, "verticalAperture", d.vertical_aperture);
    Read(prim, "clippingRange:min", d.near_clip);
    Read(prim, "clippingRange:max", d.far_clip);
    ReadTarget(prim, "physics:body0", d.body0_path);
    ReadTarget(prim, "physics:body1", d.body1_path);
    ReadTarget(prim, "nuka:joint", d.joint_path);
    ReadTarget(prim, "nuka:body", d.body_path);
    Read(prim, "physics:axis", d.axis_token);
    Read(prim, "axis", d.geometry_axis_token);
    Read(prim, "nuka:type", d.nuka_type);
    Read(prim, "purpose", d.purpose);
    Read(prim, "nuka:decompose", d.decompose_mode);
    Read(prim, "nuka:decompose:max_pieces", d.decompose_max_pieces);
    d.has_lower_limit = Read(prim, "physics:lowerLimit", d.lower_limit);
    d.has_upper_limit = Read(prim, "physics:upperLimit", d.upper_limit);
    d.has_initial_position = Read(prim, "nuka:initialPosition", d.initial_position);
    Read(prim, "nuka:gain", d.gain);
    Read(prim, "nuka:forceLimit", d.force_limit);
    Read(prim, "nuka:sampleRate", d.sample_rate_hz);
    Read(prim, "nuka:horizontalFov", d.sensor_h_fov);
    Read(prim, "nuka:verticalFov", d.sensor_v_fov);
    Read(prim, "nuka:horizontalResolution", d.sensor_h_res);
    Read(prim, "nuka:verticalResolution", d.sensor_v_res);
    Read(prim, "nuka:minRange", d.sensor_min_range);
    Read(prim, "nuka:maxRange", d.sensor_max_range);
    Read(prim, "nuka:width", d.sensor_width);
    Read(prim, "nuka:height", d.sensor_height);
    if (d.type == "Mesh") {
        pxr::VtArray<pxr::GfVec3f> points;
        pxr::VtArray<int> indices;
        pxr::VtArray<int> counts;
        prim.GetAttribute(pxr::TfToken("points")).Get(&points, kTime);
        prim.GetAttribute(pxr::TfToken("faceVertexIndices")).Get(&indices, kTime);
        prim.GetAttribute(pxr::TfToken("faceVertexCounts")).Get(&counts, kTime);
        d.mesh_points.reserve(points.size() * 3u);
        for (const pxr::GfVec3f& p : points) {
            d.mesh_points.insert(d.mesh_points.end(), {p[0], p[1], p[2]});
        }
        d.face_vertex_indices.assign(indices.begin(), indices.end());
        d.face_vertex_counts.assign(counts.begin(), counts.end());
        d.world = xforms.GetLocalToWorldTransform(prim);
    }
    return d;
}

// A prim's local rotation from its rotateXYZ Euler (degrees, R = Rz*Ry*Rx).
math::Quat LocalRotation(const PrimData& prim) {
    constexpr float kDegToRad = 0.017453292519943295f;  // pi / 180
    const math::Quat qx =
        math::Quat::FromAxisAngle({1.0f, 0.0f, 0.0f}, prim.rotate_xyz.x * kDegToRad);
    const math::Quat qy =
        math::Quat::FromAxisAngle({0.0f, 1.0f, 0.0f}, prim.rotate_xyz.y * kDegToRad);
    const math::Quat qz =
        math::Quat::FromAxisAngle({0.0f, 0.0f, 1.0f}, prim.rotate_xyz.z * kDegToRad);
    return (qz * qy * qx).Normalized();
}

// Mesh points in world space, for a mesh no rigid body carries.
std::vector<float> WorldPoints(const PrimData& prim) {
    std::vector<float> out(prim.mesh_points.size());
    for (size_t i = 0; i + 2 < out.size(); i += 3) {
        const pxr::GfVec3d p = prim.world.Transform(
            pxr::GfVec3d(prim.mesh_points[i], prim.mesh_points[i + 1], prim.mesh_points[i + 2]));
        out[i] = static_cast<float>(p[0]);
        out[i + 1] = static_cast<float>(p[1]);
        out[i + 2] = static_cast<float>(p[2]);
    }
    return out;
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

bool IsRigidBodyPrim(const PrimData& prim) {
    return prim.has_rigid_body_api && prim.rigid_body_enabled;
}

bool MeshHasGeometry(const PrimData& prim) {
    return prim.type == "Mesh" && !prim.mesh_points.empty();
}

bool ShapeTypeFromPrim(const PrimData& prim, scene::ShapeType& shape_type) {
    // A Mesh carrying geometry is always a TriMesh shape, even without a PhysicsCollisionAPI;
    // the primitive shapes need the collision API.
    if (MeshHasGeometry(prim)) {
        shape_type = scene::ShapeType::TriMesh;
        return true;
    }
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
    if (prim.type == "Capsule") {
        shape_type = scene::ShapeType::Capsule;
        return true;
    }
    if (prim.type == "Cylinder") {
        shape_type = scene::ShapeType::Cylinder;
        return true;
    }
    if (prim.type == "Mesh") {
        shape_type = scene::ShapeType::TriMesh;
        return true;
    }
    return false;
}

// Fan-triangulates a USD face list into triangle indices in authored order; without counts the
// indices are already triangles.
std::vector<uint32_t> FanTriangulate(const std::vector<int>& indices,
                                     const std::vector<int>& counts) {
    std::vector<uint32_t> out;
    if (counts.empty()) {
        out.reserve(indices.size());
        for (const int idx : indices) {
            if (idx >= 0) {
                out.push_back(static_cast<uint32_t>(idx));
            }
        }
        return out;
    }
    size_t cursor = 0;
    for (const int raw_count : counts) {
        if (raw_count < 3) {
            cursor += static_cast<size_t>(raw_count < 0 ? 0 : raw_count);
            continue;
        }
        const size_t count = static_cast<size_t>(raw_count);
        if (cursor + count > indices.size()) {
            break;  // malformed: not enough indices for this face.
        }
        for (size_t v = 1; v + 1 < count; ++v) {
            out.push_back(static_cast<uint32_t>(indices[cursor]));
            out.push_back(static_cast<uint32_t>(indices[cursor + v]));
            out.push_back(static_cast<uint32_t>(indices[cursor + v + 1]));
        }
        cursor += count;
    }
    return out;
}

bool IsJointPrim(const PrimData& prim) {
    return (StartsWith(prim.type, "Physics") || StartsWith(prim.type, "Physx")) &&
           prim.type.find("Joint") != std::string::npos;
}

scene::JointType JointTypeFromUsdType(const std::string& type, const std::string& name) {
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
    throw std::runtime_error("USD: joint '" + name + "' has unsupported type '" + type + "'");
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
    if (lower == "linear_velocity" || lower == "velocimeter") return scene::SensorType::LinearVelocity;
    if (lower == "lidar") {
        return scene::SensorType::Lidar;
    }
    if (lower == "rangescan" || lower == "range_scan") {
        return scene::SensorType::RangeScan;
    }
    if (lower == "depth") {
        return scene::SensorType::Depth;
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
    if (lower == "jointstate" || lower == "joint_state") {
        return scene::SensorType::JointState;
    }
    return scene::SensorType::Imu;
}

scene::DecomposeMode DecomposeModeFromToken(const std::string& token) {
    const std::string lower = Lowercase(token);
    if (lower == "force") {
        return scene::DecomposeMode::Force;
    }
    if (lower == "skip") {
        return scene::DecomposeMode::Skip;
    }
    return scene::DecomposeMode::Auto;
}

// Fallback vertical FOV (deg) used only when no focalLength is authored on the
// UsdGeomCamera -- USD has no implied FOV without focal length + aperture.
constexpr float kDefaultCameraVFovDegrees = 45.0f;

// UsdLux intensity (times 2^exposure) per Nuka radiance unit: an Isaac Lab sun of 3000 is a
// Nuka sun of 3.
constexpr float kUsdLuxToNuka = 1.0f / 1000.0f;

// Vertical FOV (degrees) from a UsdGeomCamera, per the physCamera convention:
//   vFOV = 2 * atan(verticalAperture / (2 * focalLength)).
// Aperture and focal length share units (mm), so the ratio is unitless. Falls
// back to a fixed default only when focalLength was not authored.
float CameraVFovDegreesFromUsd(const PrimData& prim) {
    if (!prim.has_focal_length || prim.focal_length <= 0.0f ||
        prim.vertical_aperture <= 0.0f) {
        return kDefaultCameraVFovDegrees;
    }
    constexpr float kRadToDeg = 57.29577951308232f;  // 180 / pi
    const float half = std::atan(prim.vertical_aperture /
                                 (2.0f * prim.focal_length));
    return 2.0f * half * kRadToDeg;
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

scene::SceneIR BuildSceneFromPrims(const std::vector<PrimData>& prims) {
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
            if (prim.has_friction) {
                material.friction_mu = prim.friction_mu;
            }
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
        // USD `purpose`: guide (viewport-only annotations) and proxy (low-detail
        // collision stand-in) geometry are not part of the render/physics scene
        // here -- skip them. render-purpose geometry is kept but forced
        // visual-only below (contype=0, so the facade projects a VisualMesh).
        const std::string purpose = Lowercase(prim.purpose);
        if (purpose == "guide" || purpose == "proxy") {
            continue;
        }
        scene::ShapeType shape_type = scene::ShapeType::Box;
        if (!ShapeTypeFromPrim(prim, shape_type)) {
            continue;
        }
        scene::CollisionShapeRecord shape;
        shape.name = prim.name;
        // Carry the prim's authored ORIENTATION (rotateXYZ), not only translate, so
        // a rotated capsule axis / tilted box collides in its true pose.
        shape.local_transform = math::Transform{prim.translate, LocalRotation(prim)};
        scene::BodyId body_id = FindNearestBodyAncestor(prim.parent_path, body_map);
        const bool implicit = body_id == scene::kInvalidBody;
        if (implicit) {
            // A mesh no rigid body carries (a geometry-only asset) rides an implicit static body
            // at the origin, with its whole local-to-world transform baked into the points.
            if (!MeshHasGeometry(prim)) {
                continue;
            }
            scene::RigidBodyRecord implicit_body;
            implicit_body.name = prim.name + "_body";
            implicit_body.parent_id = scene::kInvalidBody;
            implicit_body.is_static = true;
            body_id = scene.AddRigidBody(std::move(implicit_body));
            body_map[prim.path] = body_id;
            shape.local_transform = math::Transform{};
        }
        shape.body_id = body_id;
        shape.type = shape_type;
        // render-purpose geometry is visual-only: turn collision off so the
        // SceneIR facade projects it to a VisualMeshComponent (M2b).
        if (purpose == "render") {
            shape.contype = 0;
            shape.conaffinity = 0;
        }
        if (shape_type == scene::ShapeType::Box) {
            // A non-cube box is a Cube scaled per-axis (xformOp:scale); bake the
            // scale into the half-extents so the box keeps its true dimensions.
            const float half = prim.cube_size * 0.5f;
            shape.half_extents = {half * prim.scale.x, half * prim.scale.y,
                                  half * prim.scale.z};
        } else if (shape_type == scene::ShapeType::Sphere) {
            shape.radius = prim.radius;
        } else if (shape_type == scene::ShapeType::Capsule || shape_type == scene::ShapeType::Cylinder) {
            shape.radius = prim.radius;
            shape.half_height = prim.height * 0.5f;
            const auto axis = Lowercase(prim.geometry_axis_token);
            math::Quat alignment;
            if (axis == "x") alignment = math::Quat::FromAxisAngle({0, 1, 0}, 1.5707963267948966f);
            else if (axis == "y") alignment = math::Quat::FromAxisAngle({1, 0, 0}, -1.5707963267948966f);
            else if (axis != "z")
                throw std::runtime_error("USD: invalid " + prim.type + " axis '" + prim.geometry_axis_token + "'");
            shape.local_transform.rotation = (shape.local_transform.rotation * alignment).Normalized();
        }
        if (shape_type == scene::ShapeType::TriMesh) {
            shape.decompose_mode = DecomposeModeFromToken(prim.decompose_mode);
            if (prim.decompose_max_pieces > 0) {
                shape.decompose_max_pieces =
                    static_cast<uint32_t>(prim.decompose_max_pieces);
            }
            shape.mesh_vertices = implicit ? WorldPoints(prim) : prim.mesh_points;
            shape.mesh_indices =
                FanTriangulate(prim.face_vertex_indices, prim.face_vertex_counts);
        }
        scene.AddCollisionShape(std::move(shape));
    }

    for (const auto& prim : prims) {
        if (!IsJointPrim(prim)) {
            continue;
        }
        if (prim.exclude_from_articulation) {
            throw std::runtime_error("USD: joint '" + prim.name +
                                     "' is excluded from the articulation; loop joints are unsupported");
        }
        if (!prim.joint_enabled) {
            throw std::runtime_error("USD: joint '" + prim.name + "' is disabled; disabled joints are unsupported");
        }
        const auto parent_it = body_map.find(prim.body0_path);
        const auto child_it = body_map.find(prim.body1_path);
        if (parent_it == body_map.end() || child_it == body_map.end()) {
            throw std::runtime_error("USD: joint '" + prim.name + "' references an unknown rigid body");
        }
        scene::JointRecord joint;
        joint.name = prim.name;
        joint.type = JointTypeFromUsdType(prim.type, prim.name);
        joint.parent_body = parent_it->second;
        joint.child_body = child_it->second;
        joint.axis = AxisFromUsdToken(prim.axis_token);
        if (prim.has_initial_position) {
            joint.initial_position = prim.initial_position;
        }
        joint.lower_limit = prim.lower_limit;
        joint.upper_limit = prim.upper_limit;
        joint.has_lower_limit = prim.has_lower_limit;
        joint.has_upper_limit = prim.has_upper_limit;
        const scene::JointId joint_id = scene.AddJoint(std::move(joint));
        joint_map[prim.path] = joint_id;
    }

    for (const auto& prim : prims) {
        if (prim.type == "Camera") {
            scene::CameraRecord camera;
            camera.name = prim.name;
            camera.attached_body = FindNearestBodyAncestor(prim.parent_path, body_map);
            camera.local_transform = math::Transform{prim.translate, LocalRotation(prim)};
            camera.vertical_fov_degrees = CameraVFovDegreesFromUsd(prim);
            camera.near_clip = prim.near_clip;
            camera.far_clip = prim.far_clip;
            scene.AddCamera(std::move(camera));
        } else if (prim.type.find("Light") != std::string::npos) {
            scene::LightRecord light;
            light.name = prim.name;
            light.type = LightTypeFromUsdType(prim.type);
            light.local_transform = math::Transform{prim.translate, LocalRotation(prim)};
            light.color = prim.color;
            light.intensity = prim.intensity * std::exp2(prim.exposure) * kUsdLuxToNuka;
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
            scene::SensorDesc sensor;
            sensor.name = prim.name;
            sensor.type = SensorTypeFromToken(prim.nuka_type);
            sensor.mount = scene::MountFrame::Body;
            sensor.mount_index = body_it->second;
            sensor.local_offset = math::Transform{prim.translate, LocalRotation(prim)};
            sensor.sample_rate_hz = prim.sample_rate_hz;
            // Render-sensor payloads: a lidar/range fan + range, or a depth image
            // size. Absent attributes leave the descriptor's defaults in place.
            if (sensor.type == scene::SensorType::Lidar ||
                sensor.type == scene::SensorType::RangeScan) {
                sensor.lidar.az_count = static_cast<uint16_t>(prim.sensor_h_res);
                sensor.lidar.el_count = static_cast<uint16_t>(prim.sensor_v_res);
                sensor.lidar.az_min = -0.5f * prim.sensor_h_fov;
                sensor.lidar.az_max = 0.5f * prim.sensor_h_fov;
                sensor.lidar.el_min = -0.5f * prim.sensor_v_fov;
                sensor.lidar.el_max = 0.5f * prim.sensor_v_fov;
                sensor.lidar.min_range = prim.sensor_min_range;
                if (prim.sensor_max_range > 0.0f) {
                    sensor.lidar.max_range = prim.sensor_max_range;
                }
            } else if (sensor.type == scene::SensorType::Depth ||
                       sensor.type == scene::SensorType::Camera) {
                sensor.cam.width = static_cast<uint16_t>(prim.sensor_width);
                sensor.cam.height = static_cast<uint16_t>(prim.sensor_height);
                if (prim.sensor_v_fov > 0.0f) {
                    constexpr float kRadToDeg = 57.29577951308232f;
                    sensor.cam.vfov_degrees = prim.sensor_v_fov * kRadToDeg;
                }
            }
            scene.AddSensor(std::move(sensor));
        }
    }
    return scene;
}

} // namespace

scene::SceneIR LoadUsd(const std::string& path) {
    const pxr::UsdStageRefPtr stage = OpenUsdStage(path);
    pxr::UsdGeomXformCache xforms(kTime);
    std::vector<PrimData> prims;
    for (const pxr::UsdPrim& prim : stage->Traverse(pxr::UsdTraverseInstanceProxies())) {
        prims.push_back(ReadPrim(prim, xforms));
    }
    auto scene = BuildSceneFromPrims(prims);

    // A USD scene must yield rigid bodies (a physics scene) or at least one mesh shape (a
    // geometry-only asset).
    const bool has_mesh_geometry = std::any_of(scene.Shapes().begin(), scene.Shapes().end(),
                                               [](const auto& s) { return !s.mesh_vertices.empty(); });
    if (scene.RigidBodyCount() == 0u && !has_mesh_geometry) {
        throw std::runtime_error(
            "USD: no enabled UsdPhysics rigid bodies and no mesh geometry found in " + path);
    }
    return scene;
}

} // namespace nuka::import
