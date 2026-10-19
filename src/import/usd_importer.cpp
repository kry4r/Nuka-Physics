// ---------------------------------------------------------------------------
// nuka::import - USD importer: reads the stage OpenUSD composed into SceneIR
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"

#include "import/geom_inertia.hpp"
#include "import/usd_stage_adapter.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"

#include <pxr/base/gf/matrix3d.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/quatd.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/quath.h>
#include <pxr/base/gf/rotation.h>
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
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdPhysics/metrics.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace nuka::import {

namespace {

// Time-sampled data (Alembic, animated USD) imports at its first sample.
const pxr::UsdTimeCode kTime = pxr::UsdTimeCode::EarliestTime();

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kInf = std::numeric_limits<double>::infinity();
// SI density (kg/m^3) of a collider no mass, density or material describes.
constexpr double kDefaultDensity = 1000.0;
// Joint frames further apart than this at rest are reported; the authored pose still wins.
constexpr double kFrameGapMetres = 1e-4;
constexpr double kFrameGapRadians = 1e-3;

// The UsdPhysics D6 degrees of freedom, as the LimitAPI and DriveAPI instances name them.
constexpr const char* kDofNames[6] = {"transX", "transY", "transZ", "rotX", "rotY", "rotZ"};

float F(double value) {
    return static_cast<float>(value);
}

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
pxr::VtValue Authored(const pxr::UsdPrim& prim, const std::string& name) {
    pxr::VtValue value;
    const pxr::UsdAttribute attr = prim.GetAttribute(pxr::TfToken(name));
    if (attr && attr.HasAuthoredValue()) {
        attr.Get(&value, kTime);
    }
    return value;
}

template <typename T>
bool Read(const pxr::UsdPrim& prim, const std::string& name, T& out) {
    const pxr::VtValue value = pxr::VtValue::Cast<T>(Authored(prim, name));
    if (value.IsEmpty()) {
        return false;
    }
    out = value.UncheckedGet<T>();
    return true;
}

// The attribute's value, authored or the fallback its prim's schema defines (Cube size 2, ...).
template <typename T>
bool ReadValue(const pxr::UsdPrim& prim, const char* name, T& out) {
    pxr::VtValue value;
    const pxr::UsdAttribute attr = prim.GetAttribute(pxr::TfToken(name));
    if (!attr || !attr.Get(&value, kTime)) {
        return false;
    }
    value = pxr::VtValue::Cast<T>(value);
    if (value.IsEmpty()) {
        return false;
    }
    out = value.UncheckedGet<T>();
    return true;
}

// Float and double attributes alike; the value stays in double until a record stores it.
bool Read(const pxr::UsdPrim& prim, const std::string& name, double& out) {
    const pxr::VtValue value = Authored(prim, name);
    if (value.IsHolding<float>()) {
        out = value.UncheckedGet<float>();
    } else if (value.IsHolding<double>()) {
        out = value.UncheckedGet<double>();
    } else if (value.IsHolding<pxr::GfHalf>()) {
        out = value.UncheckedGet<pxr::GfHalf>();
    } else {
        const pxr::VtValue cast = pxr::VtValue::Cast<double>(value);
        if (cast.IsEmpty()) {
            return false;
        }
        out = cast.UncheckedGet<double>();
    }
    return true;
}

bool Read(const pxr::UsdPrim& prim, const std::string& name, float& out) {
    double value = 0.0;
    if (!Read(prim, name, value)) {
        return false;
    }
    out = static_cast<float>(value);
    return true;
}

template <typename V>
bool AsVec3(const pxr::VtValue& value, pxr::GfVec3d& out) {
    if (!value.IsHolding<V>()) {
        return false;
    }
    const V& v = value.UncheckedGet<V>();
    out = pxr::GfVec3d(v[0], v[1], v[2]);
    return true;
}

bool Read(const pxr::UsdPrim& prim, const std::string& name, pxr::GfVec3d& out) {
    const pxr::VtValue value = Authored(prim, name);
    return AsVec3<pxr::GfVec3f>(value, out) || AsVec3<pxr::GfVec3d>(value, out) ||
           AsVec3<pxr::GfVec3h>(value, out);
}

bool Read(const pxr::UsdPrim& prim, const std::string& name, math::Vec3& out) {
    pxr::GfVec3d v;
    if (!Read(prim, name, v)) {
        return false;
    }
    out = {F(v[0]), F(v[1]), F(v[2])};
    return true;
}

template <typename Q>
bool AsQuat(const pxr::VtValue& value, pxr::GfQuatd& out) {
    if (!value.IsHolding<Q>()) {
        return false;
    }
    const Q& q = value.UncheckedGet<Q>();
    const auto& i = q.GetImaginary();
    out = pxr::GfQuatd(q.GetReal(), pxr::GfVec3d(i[0], i[1], i[2]));
    return true;
}

bool Read(const pxr::UsdPrim& prim, const std::string& name, pxr::GfQuatd& out) {
    const pxr::VtValue value = Authored(prim, name);
    return AsQuat<pxr::GfQuatf>(value, out) || AsQuat<pxr::GfQuatd>(value, out) ||
           AsQuat<pxr::GfQuath>(value, out);
}

bool Read(const pxr::UsdPrim& prim, const std::string& name, std::string& out) {
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

// ---------------------------------------------------------------------------
// Poses
// ---------------------------------------------------------------------------

// A rigid transform kept in double until a record stores it.
struct Pose {
    pxr::GfQuatd rotation = pxr::GfQuatd::GetIdentity();
    pxr::GfVec3d position = pxr::GfVec3d(0.0);

    pxr::GfVec3d Apply(const pxr::GfVec3d& p) const { return rotation.Transform(p) + position; }
    Pose operator*(const Pose& other) const { return {rotation * other.rotation, Apply(other.position)}; }
    Pose Inverse() const {
        const pxr::GfQuatd inverse = rotation.GetConjugate();
        return {inverse, -inverse.Transform(position)};
    }
};

math::Transform ToTransform(const Pose& pose) {
    const pxr::GfVec3d& p = pose.position;
    const pxr::GfVec3d& v = pose.rotation.GetImaginary();
    return math::Transform{math::Vec3{F(p[0]), F(p[1]), F(p[2])},
                           math::Quat{F(pose.rotation.GetReal()), F(v[0]), F(v[1]), F(v[2])}};
}

// A local-to-world matrix split into a rigid pose and the linear part left in the pose's own axes
// (scale, a mirror, and any shear): M = residual * rotation * translation in USD's row-vector order.
struct Placement {
    Pose pose;
    pxr::GfMatrix3d residual{1.0};
};

Placement Place(const pxr::GfMatrix4d& m) {
    // Gram-Schmidt leaves an orthonormal frame untouched, so rotation-only authoring stays exact.
    pxr::GfVec3d x(m[0][0], m[0][1], m[0][2]);
    pxr::GfVec3d y(m[1][0], m[1][1], m[1][2]);
    x.Normalize();
    y -= pxr::GfDot(y, x) * x;
    y.Normalize();
    const pxr::GfVec3d z = pxr::GfCross(x, y);
    pxr::GfMatrix4d rotation(1.0);
    rotation.SetRow3(0, x);
    rotation.SetRow3(1, y);
    rotation.SetRow3(2, z);
    Placement out;
    out.pose.rotation = rotation.ExtractRotationQuat();
    if (out.pose.rotation.GetReal() < 0.0) {
        out.pose.rotation = pxr::GfQuatd(-out.pose.rotation.GetReal(), -out.pose.rotation.GetImaginary());
    }
    out.pose.position = m.ExtractTranslation();
    const pxr::GfMatrix3d linear(m[0][0], m[0][1], m[0][2], m[1][0], m[1][1], m[1][2],
                                 m[2][0], m[2][1], m[2][2]);
    const pxr::GfMatrix3d axes(x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]);
    out.residual = linear * axes.GetTranspose();
    return out;
}

bool IsIdentity(const pxr::GfMatrix3d& m) {
    return m == pxr::GfMatrix3d(1.0);
}

// Lengths of the residual's rows: the scale along each of the pose's axes.
pxr::GfVec3d ScaleOf(const pxr::GfMatrix3d& residual) {
    return {residual.GetRow(0).GetLength(), residual.GetRow(1).GetLength(), residual.GetRow(2).GetLength()};
}

bool Sheared(const pxr::GfMatrix3d& residual) {
    const pxr::GfVec3d s = ScaleOf(residual);
    const double tolerance = 1e-6 * std::max({s[0], s[1], s[2]});
    return std::abs(residual[0][1]) > tolerance || std::abs(residual[0][2]) > tolerance ||
           std::abs(residual[1][0]) > tolerance || std::abs(residual[1][2]) > tolerance ||
           std::abs(residual[2][0]) > tolerance || std::abs(residual[2][1]) > tolerance;
}

// Stage distance and mass units and the up axis, mapped onto Nuka's metres, kilograms and +Z up.
struct StageUnits {
    double meters = 1.0;
    double kilograms = 1.0;
    pxr::GfMatrix4d correction{1.0};
    bool identity = true;
};

StageUnits UnitsOf(const pxr::UsdStageRefPtr& stage) {
    StageUnits units;
    units.meters = pxr::UsdGeomGetStageMetersPerUnit(stage);
    units.kilograms = pxr::UsdPhysicsGetStageKilogramsPerUnit(stage);
    const bool y_up = pxr::UsdGeomGetStageUpAxis(stage) == pxr::UsdGeomTokens->y;
    units.identity = units.meters == 1.0 && !y_up;
    pxr::GfMatrix4d scale(1.0);
    scale.SetScale(units.meters);
    units.correction = scale;
    if (y_up) {
        pxr::GfMatrix4d turn(1.0);
        turn.SetRotate(pxr::GfRotation(pxr::GfVec3d(1.0, 0.0, 0.0), 90.0));
        units.correction = scale * turn;
    }
    return units;
}

// ---------------------------------------------------------------------------
// Prims
// ---------------------------------------------------------------------------

// Everything the scene builder reads from one composed prim; unauthored values keep these defaults.
struct PrimData {
    pxr::UsdPrim prim;
    std::string type;
    std::string name;
    std::string path;
    std::string parent_path;
    std::vector<std::string> apis;       // applied schemas, multiple-apply ones as "PhysicsLimitAPI:rotX"
    pxr::GfMatrix4d world{1.0};          // local-to-world in Nuka's units and axes
    math::Vec3 translate = math::Vec3::Zero();
    // xformOp:rotateXYZ Euler angles (degrees, applied X then Y then Z).
    math::Vec3 rotate_xyz = math::Vec3::Zero();
    double cube_size = 1.0;
    double radius = 0.5;
    double height = 1.0;
    std::string geometry_axis_token = "Z";
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
    std::string joint_path;
    std::string body_path;
    std::string nuka_type;
    std::string purpose;                 // USD `purpose` token: default|render|proxy|guide
    std::string decompose_mode;          // nuka:decompose token
    int decompose_max_pieces = 32;       // nuka:decompose:max_pieces
    bool has_initial_position = false;
    float initial_position = 0.0f;
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

    bool HasApi(std::string_view api) const {
        return std::find(apis.begin(), apis.end(), api) != apis.end();
    }
};

PrimData ReadPrim(const pxr::UsdPrim& prim, pxr::UsdGeomXformCache& xforms, const StageUnits& units) {
    PrimData d;
    d.prim = prim;
    d.type = prim.GetTypeName().GetString();
    d.name = prim.GetName().GetString();
    d.path = prim.GetPath().GetString();
    d.parent_path = prim.GetParent().GetPath().GetString();
    for (const pxr::TfToken& api : prim.GetAppliedSchemas()) {
        d.apis.push_back(api.GetString());
    }
    d.world = xforms.GetLocalToWorldTransform(prim);
    if (!units.identity) {
        d.world = d.world * units.correction;
    }
    Read(prim, "xformOp:translate", d.translate);
    Read(prim, "xformOp:rotateXYZ", d.rotate_xyz);
    ReadValue(prim, "size", d.cube_size);
    ReadValue(prim, "radius", d.radius);
    ReadValue(prim, "height", d.height);
    pxr::TfToken axis;
    if (ReadValue(prim, "axis", axis) && !axis.IsEmpty()) {
        d.geometry_axis_token = axis.GetString();
    }
    if (!Read(prim, "inputs:color", d.color) && !Read(prim, "inputs:diffuseColor", d.color)) {
        Read(prim, "color", d.color);
    }
    Read(prim, "inputs:alpha", d.alpha);
    Read(prim, "inputs:roughness", d.roughness);
    Read(prim, "inputs:metallic", d.metallic);
    d.has_friction = Read(prim, "physics:dynamicFriction", d.friction_mu);
    if (!Read(prim, "inputs:intensity", d.intensity)) {
        Read(prim, "intensity", d.intensity);
    }
    if (!Read(prim, "inputs:exposure", d.exposure)) {
        Read(prim, "exposure", d.exposure);
    }
    d.has_focal_length = Read(prim, "focalLength", d.focal_length);
    Read(prim, "verticalAperture", d.vertical_aperture);
    Read(prim, "clippingRange:min", d.near_clip);
    Read(prim, "clippingRange:max", d.far_clip);
    ReadTarget(prim, "nuka:joint", d.joint_path);
    ReadTarget(prim, "nuka:body", d.body_path);
    Read(prim, "nuka:type", d.nuka_type);
    Read(prim, "purpose", d.purpose);
    Read(prim, "nuka:decompose", d.decompose_mode);
    Read(prim, "nuka:decompose:max_pieces", d.decompose_max_pieces);
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
    }
    return d;
}

// A NukaSensor's authored offset from its mount body (degrees, R = Rz*Ry*Rx).
math::Quat LocalRotation(const PrimData& prim) {
    constexpr float kDegToRadF = 0.017453292519943295f;  // pi / 180
    const math::Quat qx =
        math::Quat::FromAxisAngle({1.0f, 0.0f, 0.0f}, prim.rotate_xyz.x * kDegToRadF);
    const math::Quat qy =
        math::Quat::FromAxisAngle({0.0f, 1.0f, 0.0f}, prim.rotate_xyz.y * kDegToRadF);
    const math::Quat qz =
        math::Quat::FromAxisAngle({0.0f, 0.0f, 1.0f}, prim.rotate_xyz.z * kDegToRadF);
    return (qz * qy * qx).Normalized();
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
    if (!prim.HasApi("PhysicsCollisionAPI")) {
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
    if (prim.type == "Plane") {
        shape_type = scene::ShapeType::Plane;
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

// A unit vector along USD axis token X, Y or Z.
pxr::GfVec3d AxisVector(const std::string& axis) {
    const std::string lower = Lowercase(axis);
    if (lower == "y") {
        return pxr::GfVec3d(0.0, 1.0, 0.0);
    }
    if (lower == "z") {
        return pxr::GfVec3d(0.0, 0.0, 1.0);
    }
    return pxr::GfVec3d(1.0, 0.0, 0.0);
}

// Turns a shape's local +Z onto USD axis token `axis` (Capsule, Cylinder and Plane axes).
bool AxisAlignment(const std::string& axis, math::Quat& alignment) {
    const std::string lower = Lowercase(axis);
    if (lower == "x") {
        alignment = math::Quat::FromAxisAngle({0, 1, 0}, 1.5707963267948966f);
    } else if (lower == "y") {
        alignment = math::Quat::FromAxisAngle({1, 0, 0}, -1.5707963267948966f);
    } else if (lower != "z") {
        return false;
    }
    return true;
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

// ---------------------------------------------------------------------------
// Scene builder
// ---------------------------------------------------------------------------

struct Body {
    size_t prim = 0;                     // index into the prims
    Pose frame;                          // the body frame in the world, after any re-origin
    bool is_static = false;
    int joint = -1;                      // the accepted joint this body hangs from
};

// One UsdPhysics joint reduced to the single-DOF (or ball, or weld) joint the engine has.
struct Joint {
    size_t prim = 0;
    scene::JointType type = scene::JointType::Revolute;
    int parent = -1;                     // body index; -1 is the world
    int child = -1;
    pxr::GfVec3d axis{0.0, 0.0, 1.0};    // in the child body frame
    bool angular = true;
    bool has_lower = false;
    bool has_upper = false;
    double lower = 0.0;                  // Nuka units, measured from the authored pose
    double upper = 0.0;
    double rest = 0.0;                   // USD joint coordinate of the authored pose
    pxr::GfVec3d anchor{0.0};            // joint point on the child side, world
    std::string drive;                   // DriveAPI and JointStateAPI instance of the free DOF
    std::vector<std::string> other_drives;
    scene::JointId id = scene::kInvalidJoint;
};

class SceneBuilder {
public:
    SceneBuilder(const StageUnits& units, std::vector<PrimData> prims, std::vector<UsdDiagnostic>& out)
        : units_(units), prims_(std::move(prims)), out_(out) {}

    scene::SceneIR Build();

private:
    void Report(UsdDiagnosticSeverity severity, const char* code, const PrimData& prim,
                std::string message) const {
        out_.push_back(UsdDiagnostic{severity, code, prim.path, std::move(message)});
    }
    void Error(const char* code, const PrimData& prim, std::string message) const {
        Report(UsdDiagnosticSeverity::Error, code, prim, std::move(message));
    }
    void Warn(const char* code, const PrimData& prim, std::string message) const {
        Report(UsdDiagnosticSeverity::Warning, code, prim, std::move(message));
    }

    int BodyAt(std::string path) const;
    bool ResolveSide(const PrimData& prim, const char* rel, int& body, std::string& target);
    Pose AnchorPose(const std::string& target, const pxr::GfVec3d& position, const pxr::GfQuatd& rotation) const;
    void ReadJoint(size_t index);
    bool ReadD6(const PrimData& prim, Joint& joint);
    void BreakCycles();
    void AddBodies(scene::SceneIR& scene);
    void AddShapes(scene::SceneIR& scene);
    void SetMass(const Body& body, scene::RigidBodyRecord& record) const;
    void AddJoints(scene::SceneIR& scene);
    void AddDrive(const Joint& joint, scene::SceneIR& scene);
    scene::BodyId WorldBody(scene::SceneIR& scene);
    Pose FrameOf(int body) const { return body < 0 ? Pose{} : bodies_[static_cast<size_t>(body)].frame; }
    scene::BodyId IdOf(int body) const { return static_cast<scene::BodyId>(body); }

    StageUnits units_;
    std::vector<PrimData> prims_;
    std::vector<UsdDiagnostic>& out_;
    std::unordered_map<std::string, size_t> prim_at_;
    std::vector<Body> bodies_;
    std::unordered_map<std::string, int> body_at_;
    std::vector<Joint> joints_;
    std::unordered_map<std::string, scene::JointId> joint_ids_;
    std::vector<scene::CollisionShapeRecord> shapes_;
    std::vector<std::pair<size_t, int>> shape_sources_;  // prim index and body index of each shape
    std::vector<scene::BodyId> anchors_;                 // per joint: the static body a world joint holds
    scene::BodyId world_body_ = scene::kInvalidBody;
    scene::BodyId next_body_ = 0;
};

// The rigid body at or above `path`; -1 when none is.
int SceneBuilder::BodyAt(std::string path) const {
    while (!path.empty()) {
        const auto it = body_at_.find(path);
        if (it != body_at_.end()) {
            return it->second;
        }
        const size_t slash = path.find_last_of('/');
        if (slash == std::string::npos || slash == 0) {
            break;
        }
        path = path.substr(0, slash);
    }
    return -1;
}

// One side of a joint: the body its relationship targets, or the world (-1) when it has no target.
bool SceneBuilder::ResolveSide(const PrimData& prim, const char* rel, int& body, std::string& target) {
    ReadTarget(prim.prim, rel, target);
    body = target.empty() ? -1 : BodyAt(target);
    if (!target.empty() && body < 0) {
        Error("USD_JOINT_BODY_UNKNOWN", prim, std::string(rel) + " <" + target +
              "> is not a rigid body or inside one; the joint is left out");
        return false;
    }
    return true;
}

// A joint frame in the world: localPos sits in the target's scaled space, localRot turns its axes.
Pose SceneBuilder::AnchorPose(const std::string& target, const pxr::GfVec3d& position,
                              const pxr::GfQuatd& rotation) const {
    const pxr::GfMatrix4d& world = target.empty() ? units_.correction : prims_[prim_at_.at(target)].world;
    return Pose{Place(world).pose.rotation * rotation, world.Transform(position)};
}

// A generic PhysicsJoint: the LimitAPI instances that lock axes (low > high) leave its free DOFs.
bool SceneBuilder::ReadD6(const PrimData& prim, Joint& joint) {
    std::vector<int> free;
    for (int dof = 0; dof < 6; ++dof) {
        const std::string name = kDofNames[dof];
        double low = -kInf, high = kInf;
        Read(prim.prim, "limit:" + name + ":physics:low", low);
        Read(prim.prim, "limit:" + name + ":physics:high", high);
        if (!(low > high)) {
            free.push_back(dof);
        }
    }
    const auto rotations = std::count_if(free.begin(), free.end(), [](int dof) { return dof >= 3; });
    std::vector<std::string> names;
    for (const int dof : free) {
        names.push_back(kDofNames[dof]);
    }
    if (free.empty()) {
        joint.type = scene::JointType::Fixed;
    } else if (free.size() == 1u) {
        const int dof = free.front();
        joint.angular = dof >= 3;
        joint.type = joint.angular ? scene::JointType::Revolute : scene::JointType::Prismatic;
        joint.axis = AxisVector(std::string(1, kDofNames[dof][joint.angular ? 3 : 5]));
        joint.drive = kDofNames[dof];
    } else if (free.size() == 3u && rotations == 3) {
        joint.type = scene::JointType::Spherical;
    } else {
        std::string list;
        for (const std::string& n : names) {
            list += (list.empty() ? "" : ", ") + n;
        }
        Error("USD_JOINT_DOF_UNSUPPORTED", prim, "the joint frees " + list +
              "; a joint with more than one free axis needs multi-DOF joints, which the engine "
              "does not have yet");
        return false;
    }
    for (const int dof : free) {
        if (names.size() == 3u || kDofNames[dof] != joint.drive) {
            double low = -kInf, high = kInf;
            Read(prim.prim, std::string("limit:") + kDofNames[dof] + ":physics:low", low);
            Read(prim.prim, std::string("limit:") + kDofNames[dof] + ":physics:high", high);
            if (std::isfinite(low) || std::isfinite(high)) {
                Warn("USD_JOINT_LIMIT_UNSUPPORTED", prim, std::string("the ") + kDofNames[dof] +
                     " limit of a ball joint is not represented and is dropped");
            }
        }
    }
    for (int dof = 0; dof < 6; ++dof) {
        if (prim.HasApi(std::string("PhysicsDriveAPI:") + kDofNames[dof]) && kDofNames[dof] != joint.drive) {
            joint.other_drives.push_back(kDofNames[dof]);
        }
    }
    if (!joint.drive.empty()) {
        Read(prim.prim, "limit:" + joint.drive + ":physics:low", joint.lower);
        Read(prim.prim, "limit:" + joint.drive + ":physics:high", joint.upper);
    }
    return true;
}

void SceneBuilder::ReadJoint(size_t index) {
    const PrimData& prim = prims_[index];
    bool enabled = true;
    Read(prim.prim, "physics:jointEnabled", enabled);
    if (!enabled) {
        Warn("USD_JOINT_DISABLED", prim, "the joint is disabled and is left out");
        return;
    }
    bool excluded = false;
    Read(prim.prim, "physics:excludeFromArticulation", excluded);
    if (excluded) {
        Error("USD_JOINT_EXCLUDED", prim, "joints outside the articulation (loop closures) are not "
              "supported yet; the joint is left out");
        return;
    }
    Joint joint;
    joint.prim = index;
    std::string target0, target1;
    if (!ResolveSide(prim, "physics:body0", joint.parent, target0) ||
        !ResolveSide(prim, "physics:body1", joint.child, target1)) {
        return;
    }
    if (joint.parent == joint.child) {
        Error("USD_JOINT_BODY_UNKNOWN", prim, joint.child < 0 ? "the joint names no body"
              : "the joint connects a body to itself");
        return;
    }
    joint.lower = -kInf;
    joint.upper = kInf;
    std::string axis = "X";
    Read(prim.prim, "physics:axis", axis);
    if (prim.type == "PhysicsRevoluteJoint" || prim.type == "PhysicsPrismaticJoint") {
        joint.angular = prim.type == "PhysicsRevoluteJoint";
        joint.type = joint.angular ? scene::JointType::Revolute : scene::JointType::Prismatic;
        joint.axis = AxisVector(axis);
        joint.drive = joint.angular ? "angular" : "linear";
        Read(prim.prim, "physics:lowerLimit", joint.lower);
        Read(prim.prim, "physics:upperLimit", joint.upper);
    } else if (prim.type == "PhysicsSphericalJoint") {
        joint.type = scene::JointType::Spherical;
        double cone0 = kInf, cone1 = kInf;
        Read(prim.prim, "physics:coneAngle0Limit", cone0);
        Read(prim.prim, "physics:coneAngle1Limit", cone1);
        if ((std::isfinite(cone0) && cone0 >= 0.0) || (std::isfinite(cone1) && cone1 >= 0.0)) {
            Warn("USD_JOINT_LIMIT_UNSUPPORTED", prim, "the cone limit of a ball joint is not represented "
                 "and is dropped");
        }
    } else if (prim.type == "PhysicsFixedJoint") {
        joint.type = scene::JointType::Fixed;
    } else if (prim.type == "PhysicsJoint") {
        if (!ReadD6(prim, joint)) {
            return;
        }
    } else {
        Error("USD_JOINT_TYPE_UNSUPPORTED", prim, "joint type " + prim.type + " is not supported; "
              "the joint is left out");
        return;
    }
    if (joint.type == scene::JointType::Spherical || joint.type == scene::JointType::Fixed) {
        for (const char* drive : {"angular", "linear"}) {
            if (prim.HasApi(std::string("PhysicsDriveAPI:") + drive)) {
                joint.other_drives.push_back(drive);
            }
        }
    }
    if (joint.child < 0) {
        if (joint.type == scene::JointType::Fixed) {
            bodies_[static_cast<size_t>(joint.parent)].is_static = true;
            return;
        }
        Error("USD_JOINT_WORLD_BODY1", prim, "the joint holds the world as body1; swap body0 and body1");
        return;
    }
    if (joint.parent < 0 && joint.type == scene::JointType::Fixed) {
        bodies_[static_cast<size_t>(joint.child)].is_static = true;
        return;
    }
    Body& child = bodies_[static_cast<size_t>(joint.child)];
    if (child.joint >= 0) {
        Error("USD_JOINT_LOOP", prim, "body <" + prims_[child.prim].path + "> already hangs from joint <" +
              prims_[joints_[static_cast<size_t>(child.joint)].prim].path +
              ">; closed loops are not supported yet, so this joint is left out");
        return;
    }

    pxr::GfVec3d pos0(0.0), pos1(0.0);
    pxr::GfQuatd rot0 = pxr::GfQuatd::GetIdentity(), rot1 = pxr::GfQuatd::GetIdentity();
    Read(prim.prim, "physics:localPos0", pos0);
    Read(prim.prim, "physics:localPos1", pos1);
    Read(prim.prim, "physics:localRot0", rot0);
    Read(prim.prim, "physics:localRot1", rot1);
    rot0.Normalize();
    rot1.Normalize();
    const Pose frame0 = AnchorPose(target0, pos0, rot0);
    const Pose frame1 = AnchorPose(target1, pos1, rot1);
    joint.anchor = frame1.position;

    // The axis in the child body frame; a target that is the body itself adds no turn of its own.
    pxr::GfQuatd turn = rot1;
    if (target1 != prims_[child.prim].path) {
        turn = child.frame.rotation.GetConjugate() * frame1.rotation;
    }
    const pxr::GfVec3d joint_axis = joint.axis;
    joint.axis = turn.Transform(joint_axis);

    // The joint coordinate USD measures at the authored pose: twist about the axis, or slide along it.
    const pxr::GfQuatd relative = frame0.rotation.GetConjugate() * frame1.rotation;
    const pxr::GfVec3d gap = frame1.position - frame0.position;
    const pxr::GfVec3d world_axis = frame1.rotation.Transform(joint_axis);
    double swing = 0.0;
    double offset = gap.GetLength();
    if (joint.type == scene::JointType::Revolute) {
        joint.rest = 2.0 * std::atan2(pxr::GfDot(relative.GetImaginary(), joint_axis), relative.GetReal());
        const pxr::GfQuatd twist(std::cos(0.5 * joint.rest), std::sin(0.5 * joint.rest) * joint_axis);
        const pxr::GfQuatd rest_swing = relative * twist.GetConjugate();
        swing = 2.0 * std::acos(std::min(1.0, std::abs(rest_swing.GetReal())));
    } else if (joint.type == scene::JointType::Prismatic) {
        joint.rest = pxr::GfDot(gap, world_axis);
        offset = (gap - joint.rest * world_axis).GetLength();
        swing = 2.0 * std::acos(std::min(1.0, std::abs(relative.GetReal())));
    } else if (joint.type == scene::JointType::Fixed) {
        swing = 2.0 * std::acos(std::min(1.0, std::abs(relative.GetReal())));
    }
    if (offset > kFrameGapMetres || swing > kFrameGapRadians) {
        Warn("USD_JOINT_FRAME_MISMATCH", prim, "the two joint frames differ at rest by " +
             std::to_string(offset) + " m and " + std::to_string(swing) +
             " rad off the joint's free motion; the authored body poses are kept");
    }

    const double unit = joint.angular ? kDegToRad : units_.meters;
    joint.has_lower = std::isfinite(joint.lower);
    joint.has_upper = std::isfinite(joint.upper);
    if (joint.type == scene::JointType::Revolute || joint.type == scene::JointType::Prismatic) {
        joint.lower = joint.lower * unit - joint.rest;
        joint.upper = joint.upper * unit - joint.rest;
    }
    if (joint.type == scene::JointType::Spherical) {
        joint.has_lower = joint.has_upper = false;
    }
    if (joint.type == scene::JointType::Fixed) {
        joint.has_lower = joint.has_upper = false;
    }
    bool collide = false;
    Read(prim.prim, "physics:collisionEnabled", collide);
    if (collide) {
        Warn("USD_JOINT_COLLISION", prim, "the engine never lets jointed bodies collide; "
             "collisionEnabled is ignored");
    }
    child.joint = static_cast<int>(joints_.size());
    joints_.push_back(std::move(joint));
}

// Drops the joint that closes each kinematic cycle a chain of single parents forms.
void SceneBuilder::BreakCycles() {
    for (size_t b = 0; b < bodies_.size(); ++b) {
        int at = static_cast<int>(b);
        for (size_t step = 0; step <= bodies_.size() && at >= 0; ++step) {
            const int j = bodies_[static_cast<size_t>(at)].joint;
            at = j < 0 ? -1 : joints_[static_cast<size_t>(j)].parent;
            if (at == static_cast<int>(b)) {
                const int closing = bodies_[b].joint;
                Error("USD_JOINT_LOOP", prims_[joints_[static_cast<size_t>(closing)].prim],
                      "the joint closes a kinematic loop; closed loops are not supported yet");
                bodies_[b].joint = -1;
                joints_[static_cast<size_t>(closing)].child = -1;
                break;
            }
        }
    }
}

scene::BodyId SceneBuilder::WorldBody(scene::SceneIR& scene) {
    if (world_body_ == scene::kInvalidBody) {
        scene::RigidBodyRecord body;
        body.name = "world";
        body.is_static = true;
        body.mass = 0.0f;
        body.inertia = math::Vec3::Zero();
        world_body_ = scene.AddRigidBody(std::move(body));
    }
    return world_body_;
}

// Mass properties of a moving body: MassAPI values where authored, the colliders' volume otherwise.
void SceneBuilder::SetMass(const Body& body, scene::RigidBodyRecord& record) const {
    const PrimData& prim = prims_[body.prim];
    const double density_unit = units_.kilograms / (units_.meters * units_.meters * units_.meters);
    double body_density = 0.0;
    Read(prim.prim, "physics:density", body_density);
    BodyMass sum;
    const int self = body_at_.at(prim.path);
    for (size_t s = 0; s < shapes_.size(); ++s) {
        if (shape_sources_[s].second != self || shapes_[s].contype == 0u) {
            continue;
        }
        const PrimData& source = prims_[shape_sources_[s].first];
        double density = 0.0, mass = 0.0;
        Read(source.prim, "physics:density", density);
        Read(source.prim, "physics:mass", mass);
        const double rho = density > 0.0 ? density * density_unit
                         : body_density > 0.0 ? body_density * density_unit : kDefaultDensity;
        AddGeomMass(shapes_[s], F(rho), mass > 0.0 ? F(mass * units_.kilograms) : -1.0f, sum);
    }
    const bool from_shapes = sum.mass > 0.0;
    if (from_shapes) {
        SetBodyMass(sum, record);
    }
    double mass = 0.0;
    Read(prim.prim, "physics:mass", mass);
    if (mass > 0.0) {
        const double total = mass * units_.kilograms;
        if (from_shapes) {
            record.inertia = record.inertia * F(total / sum.mass);
        }
        record.mass = F(total);
    }
    pxr::GfVec3d com;
    if (Read(prim.prim, "physics:centerOfMass", com) && std::isfinite(com[0]) && std::isfinite(com[1]) &&
        std::isfinite(com[2])) {
        const pxr::GfVec3d at = body.frame.Inverse().Apply(prim.world.Transform(com));
        record.inertial_transform.position = math::Vec3{F(at[0]), F(at[1]), F(at[2])};
    }
    pxr::GfVec3d diagonal(0.0);
    Read(prim.prim, "physics:diagonalInertia", diagonal);
    if (diagonal[0] > 0.0 || diagonal[1] > 0.0 || diagonal[2] > 0.0) {
        const double scale = units_.kilograms * units_.meters * units_.meters;
        record.inertia = math::Vec3{F(diagonal[0] * scale), F(diagonal[1] * scale), F(diagonal[2] * scale)};
        pxr::GfQuatd axes(0.0);
        Read(prim.prim, "physics:principalAxes", axes);
        record.inertial_transform.rotation = math::Quat::Identity();
        if (axes.GetLength() > 0.0) {
            axes.Normalize();
            const pxr::GfVec3d& v = axes.GetImaginary();
            record.inertial_transform.rotation = math::Quat{F(axes.GetReal()), F(v[0]), F(v[1]), F(v[2])};
        }
    } else if (!from_shapes) {
        Warn("USD_MASS_DEFAULT", prim, "the body has no inertia and no colliders to derive it from; "
             "it gets a unit inertia");
    }
    if (mass <= 0.0 && !from_shapes) {
        Warn("USD_MASS_DEFAULT", prim, "the body has no mass and no colliders to derive it from; "
             "it weighs 1 kg");
    }
}

void SceneBuilder::AddBodies(scene::SceneIR& scene) {
    for (const Body& body : bodies_) {
        const PrimData& prim = prims_[body.prim];
        scene::RigidBodyRecord record;
        record.name = prim.name;
        Pose local = body.frame;
        if (body.joint >= 0) {
            const Joint& joint = joints_[static_cast<size_t>(body.joint)];
            local = FrameOf(joint.parent).Inverse() * body.frame;
            const std::string& parent_path = prims_[static_cast<size_t>(body.prim)].parent_path;
            if (joint.parent >= 0 && BodyAt(parent_path) == joint.parent) {
                record.parent_id = IdOf(joint.parent);
            }
        }
        record.local_transform = ToTransform(local);
        record.is_static = body.is_static;
        if (body.is_static) {
            double mass = 0.0;
            if (Read(prim.prim, "physics:mass", mass)) {
                record.mass = F(mass * units_.kilograms);
            }
            pxr::GfVec3d diagonal;
            if (Read(prim.prim, "physics:diagonalInertia", diagonal)) {
                const double scale = units_.kilograms * units_.meters * units_.meters;
                record.inertia = math::Vec3{F(diagonal[0] * scale), F(diagonal[1] * scale),
                                            F(diagonal[2] * scale)};
            }
        } else {
            SetMass(body, record);
        }
        scene.AddRigidBody(std::move(record));
    }
    next_body_ = static_cast<scene::BodyId>(bodies_.size());
    anchors_.assign(joints_.size(), scene::kInvalidBody);
    for (size_t j = 0; j < joints_.size(); ++j) {
        if (joints_[j].child >= 0 && joints_[j].parent < 0) {
            scene::RigidBodyRecord anchor;
            anchor.name = prims_[joints_[j].prim].name + "_anchor";
            anchor.is_static = true;
            anchor.mass = 0.0f;
            anchor.inertia = math::Vec3::Zero();
            anchors_[j] = scene.AddRigidBody(std::move(anchor));
            scene.GetBodyMut(IdOf(joints_[j].child)).parent_id = anchors_[j];
        }
    }
}

void SceneBuilder::AddShapes(scene::SceneIR& scene) {
    for (size_t i = 0; i < shapes_.size(); ++i) {
        scene::CollisionShapeRecord shape = shapes_[i];
        if (shape_sources_[i].second < 0) {
            shape.body_id = WorldBody(scene);
        }
        scene.AddCollisionShape(std::move(shape));
    }
}

void SceneBuilder::AddJoints(scene::SceneIR& scene) {
    for (size_t j = 0; j < joints_.size(); ++j) {
        Joint& joint = joints_[j];
        if (joint.child < 0) {
            continue;
        }
        const PrimData& prim = prims_[joint.prim];
        scene::JointRecord record;
        record.name = prim.name;
        record.type = joint.type;
        record.parent_body = joint.parent < 0 ? anchors_[j] : IdOf(joint.parent);
        record.child_body = IdOf(joint.child);
        record.axis = math::Vec3{F(joint.axis[0]), F(joint.axis[1]), F(joint.axis[2])};
        record.has_lower_limit = joint.has_lower;
        record.has_upper_limit = joint.has_upper;
        if (joint.has_lower) {
            record.lower_limit = F(joint.lower);
        }
        if (joint.has_upper) {
            record.upper_limit = F(joint.upper);
        }
        double state = 0.0;
        if (prim.has_initial_position) {
            record.initial_position = prim.initial_position;
        } else if (!joint.drive.empty() &&
                   Read(prim.prim, "state:" + joint.drive + ":physics:position", state)) {
            record.initial_position = F(state * (joint.angular ? kDegToRad : units_.meters) - joint.rest);
        }
        joint.id = scene.AddJoint(std::move(record));
        joint_ids_[prim.path] = joint.id;
    }
}

// The engine's actuator for a joint's DriveAPI: stiffness drives position, damping alone velocity.
void SceneBuilder::AddDrive(const Joint& joint, scene::SceneIR& scene) {
    const PrimData& prim = prims_[joint.prim];
    for (const std::string& other : joint.other_drives) {
        Warn("USD_DRIVE_UNSUPPORTED", prim, "the " + other + " drive acts on a locked or ball axis and is dropped");
    }
    if (joint.drive.empty() || !prim.HasApi("PhysicsDriveAPI:" + joint.drive)) {
        return;
    }
    const std::string base = "drive:" + joint.drive + ":physics:";
    double stiffness = 0.0, damping = 0.0, max_force = kInf, target_position = 0.0, target_velocity = 0.0;
    std::string type = "force";
    Read(prim.prim, base + "stiffness", stiffness);
    Read(prim.prim, base + "damping", damping);
    Read(prim.prim, base + "maxForce", max_force);
    Read(prim.prim, base + "targetPosition", target_position);
    Read(prim.prim, base + "targetVelocity", target_velocity);
    Read(prim.prim, base + "type", type);
    if (stiffness <= 0.0 && damping <= 0.0) {
        return;
    }
    // Angular gains are per degree; Nuka's are per radian.
    const double length = joint.angular ? units_.meters * units_.meters : units_.meters;
    const double per_unit = joint.angular ? 1.0 / kDegToRad : 1.0 / units_.meters;
    scene::ActuatorRecord actuator;
    actuator.name = prim.name;
    actuator.joint_id = joint.id;
    actuator.type = stiffness > 0.0 ? scene::ActuatorType::Position : scene::ActuatorType::Velocity;
    actuator.gain = F((stiffness > 0.0 ? stiffness : damping) * units_.kilograms * length * per_unit);
    if (std::isfinite(max_force) && max_force > 0.0) {
        actuator.force_limit = F(max_force * units_.kilograms * length);
    }
    if (stiffness > 0.0 && damping > 0.0) {
        Warn("USD_DRIVE_DAMPING_IGNORED", prim, "position drives damp at 2*sqrt(stiffness); the authored "
             "damping of " + std::to_string(damping) + " is not represented");
    }
    if (target_position != 0.0 || target_velocity != 0.0) {
        Warn("USD_DRIVE_TARGET_IGNORED", prim, "drive targets are set at run time; the authored target "
             "is not imported");
    }
    if (Lowercase(type) == "acceleration") {
        Warn("USD_DRIVE_ACCELERATION", prim, "an acceleration drive is imported as a force drive");
    }
    scene.AddActuator(std::move(actuator));
}

scene::SceneIR SceneBuilder::Build() {
    scene::SceneIR scene;
    for (size_t i = 0; i < prims_.size(); ++i) {
        prim_at_[prims_[i].path] = i;
    }

    for (const PrimData& prim : prims_) {
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

    // A RigidBodyAPI prim is a body unless it is disabled; a disabled or kinematic body stays put.
    for (size_t i = 0; i < prims_.size(); ++i) {
        const PrimData& prim = prims_[i];
        if (!prim.HasApi("PhysicsRigidBodyAPI")) {
            continue;
        }
        bool enabled = true, kinematic = false;
        Read(prim.prim, "physics:rigidBodyEnabled", enabled);
        Read(prim.prim, "physics:kinematicEnabled", kinematic);
        Body body;
        body.prim = i;
        body.frame = Place(prim.world).pose;
        body.is_static = kinematic || !enabled;
        body_at_[prim.path] = static_cast<int>(bodies_.size());
        bodies_.push_back(body);
    }

    for (size_t i = 0; i < prims_.size(); ++i) {
        if (IsJointPrim(prims_[i])) {
            ReadJoint(i);
        }
    }
    BreakCycles();
    // The engine turns a joint about the child origin, so a child whose joint point sits elsewhere
    // moves its origin there; everything the body carries is placed against the moved frame.
    for (Body& body : bodies_) {
        if (body.joint >= 0) {
            const Joint& joint = joints_[static_cast<size_t>(body.joint)];
            if (joint.type != scene::JointType::Fixed && joint.anchor != body.frame.position) {
                body.frame.position = joint.anchor;
            }
        }
    }

    for (size_t i = 0; i < prims_.size(); ++i) {
        const PrimData& prim = prims_[i];
        // Purpose only steers drawing: guide and proxy geometry stays out unless it is a collider
        // (Isaac and Newton author colliders as guides), and render-only geometry never collides.
        const std::string purpose = Lowercase(prim.purpose);
        const bool collider = prim.HasApi("PhysicsCollisionAPI");
        if ((purpose == "guide" || purpose == "proxy") && !collider) {
            continue;
        }
        if (prim.type == "Cone" && collider) {
            Warn("USD_SHAPE_UNSUPPORTED", prim, "cone colliders are not supported; the cone is left out");
            continue;
        }
        scene::ShapeType shape_type = scene::ShapeType::Box;
        if (!ShapeTypeFromPrim(prim, shape_type)) {
            continue;
        }
        bool collide = true;
        Read(prim.prim, "physics:collisionEnabled", collide);
        if (!collide) {
            continue;
        }
        const int owner = BodyAt(prim.path);
        const Placement placement = Place(prim.world);
        const pxr::GfVec3d scale = ScaleOf(placement.residual);
        if (Sheared(placement.residual)) {
            Warn("USD_SHAPE_SHEARED", prim, "the shape's transform shears it; only its axis scales are kept");
        }
        scene::CollisionShapeRecord shape;
        shape.name = prim.name;
        shape.body_id = owner < 0 ? scene::kInvalidBody : IdOf(owner);
        shape.type = shape_type;
        shape.local_transform = ToTransform(FrameOf(owner).Inverse() * placement.pose);
        if (purpose == "render" && !collider) {
            shape.contype = 0;
            shape.conaffinity = 0;
        }
        const auto uniform = [&](double a, double b) { return std::abs(a - b) <= 1e-6 * std::max(a, b); };
        if (shape_type == scene::ShapeType::Box) {
            const double half = prim.cube_size * 0.5;
            shape.half_extents = {F(half * scale[0]), F(half * scale[1]), F(half * scale[2])};
        } else if (shape_type == scene::ShapeType::Sphere) {
            if (!uniform(scale[0], scale[1]) || !uniform(scale[0], scale[2])) {
                Warn("USD_SCALE_NONUNIFORM", prim, "a sphere scaled unevenly keeps its largest radius");
            }
            shape.radius = F(prim.radius * std::max({scale[0], scale[1], scale[2]}));
        } else if (shape_type == scene::ShapeType::Capsule || shape_type == scene::ShapeType::Cylinder ||
                   shape_type == scene::ShapeType::Plane) {
            math::Quat alignment;
            if (!AxisAlignment(prim.geometry_axis_token, alignment)) {
                Error("USD_SHAPE_AXIS_INVALID", prim, "invalid " + prim.type + " axis '" +
                      prim.geometry_axis_token + "'; the shape is left out");
                continue;
            }
            const int along = Lowercase(prim.geometry_axis_token) == "x" ? 0
                            : Lowercase(prim.geometry_axis_token) == "y" ? 1 : 2;
            const double across0 = scale[(along + 1) % 3], across1 = scale[(along + 2) % 3];
            if (shape_type != scene::ShapeType::Plane && !uniform(across0, across1)) {
                Warn("USD_SCALE_NONUNIFORM", prim, "a " + prim.type + " scaled unevenly across its axis "
                     "keeps its largest radius");
            }
            shape.radius = F(prim.radius * std::max(across0, across1));
            shape.half_height = F(prim.height * 0.5 * scale[along]);
            shape.local_transform.rotation = (shape.local_transform.rotation * alignment).Normalized();
        } else if (shape_type == scene::ShapeType::TriMesh) {
            shape.decompose_mode = DecomposeModeFromToken(prim.decompose_mode);
            if (prim.decompose_max_pieces > 0) {
                shape.decompose_max_pieces =
                    static_cast<uint32_t>(prim.decompose_max_pieces);
            }
            shape.mesh_vertices = prim.mesh_points;
            shape.mesh_indices = FanTriangulate(prim.face_vertex_indices, prim.face_vertex_counts);
            if (!IsIdentity(placement.residual)) {
                for (size_t v = 0; v + 2 < shape.mesh_vertices.size(); v += 3) {
                    float* p = &shape.mesh_vertices[v];
                    const pxr::GfVec3d q = pxr::GfVec3d(p[0], p[1], p[2]) * placement.residual;
                    p[0] = F(q[0]);
                    p[1] = F(q[1]);
                    p[2] = F(q[2]);
                }
                if (placement.residual.GetDeterminant() < 0.0) {
                    for (size_t t = 0; t + 2 < shape.mesh_indices.size(); t += 3) {
                        std::swap(shape.mesh_indices[t + 1], shape.mesh_indices[t + 2]);
                    }
                }
            }
        }
        shapes_.push_back(std::move(shape));
        shape_sources_.emplace_back(i, owner);
    }

    AddBodies(scene);
    AddShapes(scene);
    AddJoints(scene);

    for (size_t i = 0; i < prims_.size(); ++i) {
        const PrimData& prim = prims_[i];
        if (prim.type == "Camera") {
            const int owner = BodyAt(prim.parent_path);
            scene::CameraRecord camera;
            camera.name = prim.name;
            camera.attached_body = owner < 0 ? scene::kInvalidBody : IdOf(owner);
            camera.local_transform = ToTransform(FrameOf(owner).Inverse() * Place(prim.world).pose);
            camera.vertical_fov_degrees = CameraVFovDegreesFromUsd(prim);
            camera.near_clip = prim.near_clip;
            camera.far_clip = prim.far_clip;
            scene.AddCamera(std::move(camera));
        } else if (prim.type.find("Light") != std::string::npos) {
            scene::LightRecord light;
            light.name = prim.name;
            light.type = LightTypeFromUsdType(prim.type);
            light.local_transform = ToTransform(Place(prim.world).pose);
            light.color = prim.color;
            light.intensity = prim.intensity * std::exp2(prim.exposure) * kUsdLuxToNuka;
            scene.AddLight(std::move(light));
        } else if (IsJointPrim(prim)) {
            for (const Joint& joint : joints_) {
                if (joint.prim == i && joint.id != scene::kInvalidJoint) {
                    AddDrive(joint, scene);
                }
            }
        } else if (prim.type == "NukaActuator") {
            const auto joint_it = joint_ids_.find(prim.joint_path);
            if (joint_it == joint_ids_.end()) {
                Error("USD_ACTUATOR_JOINT_UNKNOWN", prim, "nuka:joint <" + prim.joint_path +
                      "> names no imported joint; the actuator is left out");
                continue;
            }
            scene::ActuatorRecord actuator;
            actuator.name = prim.name;
            actuator.joint_id = joint_it->second;
            actuator.type = ActuatorTypeFromToken(prim.nuka_type);
            actuator.gain = prim.gain;
            actuator.force_limit = prim.force_limit;
            scene.AddActuator(std::move(actuator));
        } else if (prim.type == "NukaSensor") {
            const auto body_it = body_at_.find(prim.body_path);
            if (body_it == body_at_.end()) {
                Error("USD_SENSOR_BODY_UNKNOWN", prim, "nuka:body <" + prim.body_path +
                      "> is not a rigid body; the sensor is left out");
                continue;
            }
            const Body& mount = bodies_[static_cast<size_t>(body_it->second)];
            // The offset is authored against the body prim; a moved body origin shifts it back.
            const Pose authored = Place(prims_[mount.prim].world).pose;
            const pxr::GfVec3d shift = mount.frame.Inverse().Apply(authored.position);
            const math::Vec3 offset{F(prim.translate.x * units_.meters + shift[0]),
                                    F(prim.translate.y * units_.meters + shift[1]),
                                    F(prim.translate.z * units_.meters + shift[2])};
            scene::SensorDesc sensor;
            sensor.name = prim.name;
            sensor.type = SensorTypeFromToken(prim.nuka_type);
            sensor.mount = scene::MountFrame::Body;
            sensor.mount_index = IdOf(body_it->second);
            sensor.local_offset = math::Transform{offset, LocalRotation(prim)};
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

bool UsdImportResult::HasErrors() const {
    return std::any_of(diagnostics.begin(), diagnostics.end(), [](const UsdDiagnostic& d) {
        return d.severity == UsdDiagnosticSeverity::Error;
    });
}

UsdImportResult ImportUsd(const std::string& path) {
    const pxr::UsdStageRefPtr stage = OpenUsdStage(path);
    const StageUnits units = UnitsOf(stage);
    pxr::UsdGeomXformCache xforms(kTime);
    std::vector<PrimData> prims;
    for (const pxr::UsdPrim& prim : stage->Traverse(pxr::UsdTraverseInstanceProxies())) {
        prims.push_back(ReadPrim(prim, xforms, units));
    }
    UsdImportResult result;
    scene::SceneIR scene = SceneBuilder(units, std::move(prims), result.diagnostics).Build();
    // A USD scene must yield rigid bodies (a physics scene) or at least one shape (an environment or
    // a geometry-only asset).
    if (scene.RigidBodyCount() == 0u && scene.ShapeCount() == 0u) {
        result.diagnostics.push_back(UsdDiagnostic{UsdDiagnosticSeverity::Error, "USD_EMPTY", "/",
            "no UsdPhysics rigid bodies, colliders or meshes found in " + path});
    }
    result.scene = std::move(scene);
    return result;
}

scene::SceneIR LoadUsd(const std::string& path) {
    UsdImportResult result = ImportUsd(path);
    for (const UsdDiagnostic& d : result.diagnostics) {
        if (d.severity == UsdDiagnosticSeverity::Error) {
            throw std::runtime_error("USD: " + d.prim + ": " + d.message);
        }
    }
    return std::move(*result.scene);
}

} // namespace nuka::import
