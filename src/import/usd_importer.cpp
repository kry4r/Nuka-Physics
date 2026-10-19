// ---------------------------------------------------------------------------
// nuka::import - USD importer: reads the stage OpenUSD composed into SceneIR
// ---------------------------------------------------------------------------

#include "import/usd_importer.hpp"

#include "import/cooker/convex_decomposition.hpp"
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
#include <pxr/base/gf/vec2d.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3h.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/usd/ar/packageUtils.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/zipFile.h>
#include <pxr/usd/usd/attribute.h>
#include <pxr/usd/usd/collectionAPI.h>
#include <pxr/usd/usd/collectionMembershipQuery.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usd/relationship.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/gprim.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdPhysics/collisionGroup.h>
#include <pxr/usd/usdPhysics/metrics.h>
#include <pxr/usd/usdShade/input.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/output.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/tokens.h>
#include <pxr/usd/usdShade/utils.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
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

// The value a shader input's source attribute holds; false for outputs and unauthored inputs.
template <typename T>
bool ReadSource(const pxr::UsdAttribute& attr, T& out) {
    return attr && Read(attr.GetPrim(), attr.GetName().GetString(), out);
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
    double plane_width = 2.0;            // Plane width and length, along its local x and y
    double plane_length = 2.0;
    std::string geometry_axis_token = "Z";
    math::Vec3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    bool intensity_authored = false;     // false: the UsdLux schema's fallback intensity
    float exposure = 0.0f;               // UsdLux inputs:exposure (stops)
    float light_angle = 0.53f;           // DistantLight inputs:angle (degrees)
    float focal_length = 50.0f;
    bool has_focal_length = false;        // UsdGeomCamera focalLength authored?
    // UsdGeomCamera verticalAperture (mm); USD default 15.2908 == 0.602362"*25.4.
    float vertical_aperture = 15.2908f;
    float near_clip = 0.01f;
    float far_clip = 1000.0f;
    std::string joint_path;
    std::string body_path;
    std::string nuka_type;
    std::string purpose = "default";    // inherited USD purpose: default|render|proxy|guide
    bool visible = true;                 // inherited USD visibility
    bool left_handed = false;            // Mesh orientation leftHanded: clockwise faces
    std::string approximation;           // MeshCollisionAPI physics:approximation
    int max_hulls = 0;                   // physxConvexDecompositionCollision:maxConvexHulls
    int hull_vertex_limit = 64;          // physxConvexHullCollision:hullVertexLimit
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
    ReadValue(prim, "width", d.plane_width);
    ReadValue(prim, "length", d.plane_length);
    pxr::TfToken axis;
    if (ReadValue(prim, "axis", axis) && !axis.IsEmpty()) {
        d.geometry_axis_token = axis.GetString();
    }
    if (!Read(prim, "inputs:color", d.color) && !Read(prim, "inputs:diffuseColor", d.color)) {
        Read(prim, "color", d.color);
    }
    d.intensity_authored = Read(prim, "inputs:intensity", d.intensity) || Read(prim, "intensity", d.intensity);
    if (!d.intensity_authored) {
        ReadValue(prim, "inputs:intensity", d.intensity);
    }
    if (!Read(prim, "inputs:exposure", d.exposure)) {
        Read(prim, "exposure", d.exposure);
    }
    ReadValue(prim, "inputs:angle", d.light_angle);
    d.has_focal_length = Read(prim, "focalLength", d.focal_length);
    Read(prim, "verticalAperture", d.vertical_aperture);
    Read(prim, "clippingRange:min", d.near_clip);
    Read(prim, "clippingRange:max", d.far_clip);
    ReadTarget(prim, "nuka:joint", d.joint_path);
    ReadTarget(prim, "nuka:body", d.body_path);
    Read(prim, "nuka:type", d.nuka_type);
    if (prim.IsA<pxr::UsdGeomImageable>()) {
        const pxr::UsdGeomImageable imageable(prim);
        d.purpose = imageable.ComputePurpose().GetString();
        d.visible = imageable.ComputeVisibility(kTime) != pxr::UsdGeomTokens->invisible;
    }
    pxr::TfToken orientation;
    d.left_handed = ReadValue(prim, "orientation", orientation) && orientation == pxr::UsdGeomTokens->leftHanded;
    Read(prim, "physics:approximation", d.approximation);
    Read(prim, "physxConvexDecompositionCollision:maxConvexHulls", d.max_hulls);
    Read(prim, "physxConvexHullCollision:hullVertexLimit", d.hull_vertex_limit);
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

bool IsGprim(const std::string& type) {
    return type == "Mesh" || type == "Cube" || type == "Sphere" || type == "Capsule" || type == "Cylinder" ||
           type == "Plane" || type == "Cone";
}

// The engine shape of a primitive gprim; false for meshes and cones.
bool PrimitiveType(const std::string& type, scene::ShapeType& shape_type) {
    if (type == "Cube") {
        shape_type = scene::ShapeType::Box;
    } else if (type == "Sphere") {
        shape_type = scene::ShapeType::Sphere;
    } else if (type == "Capsule") {
        shape_type = scene::ShapeType::Capsule;
    } else if (type == "Cylinder") {
        shape_type = scene::ShapeType::Cylinder;
    } else if (type == "Plane") {
        shape_type = scene::ShapeType::Plane;
    } else {
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Meshes
// ---------------------------------------------------------------------------

// A mesh primvar as authored: `width` floats per element, optional indices, and its interpolation.
struct Primvar {
    std::vector<float> values;
    std::vector<int> indices;
    pxr::TfToken interpolation;
    size_t width = 0;
};

template <typename V>
bool AppendElements(const pxr::VtValue& value, size_t width, std::vector<float>& out) {
    if (!value.IsHolding<pxr::VtArray<V>>()) {
        return false;
    }
    for (const V& v : value.UncheckedGet<pxr::VtArray<V>>()) {
        for (size_t k = 0; k < width; ++k) {
            out.push_back(static_cast<float>(v[k]));
        }
    }
    return true;
}

// Primvar `name` as `width`-float elements (2 for UVs, 3 for normals); empty when absent.
Primvar ReadPrimvar(const pxr::UsdPrim& prim, const pxr::TfToken& name, size_t width) {
    Primvar out;
    out.width = width;
    const pxr::UsdGeomPrimvar primvar = pxr::UsdGeomPrimvarsAPI(prim).GetPrimvar(name);
    pxr::VtValue value;
    if (!primvar || !primvar.HasAuthoredValue() || !primvar.Get(&value, kTime)) {
        return out;
    }
    const bool read = width == 2u ? AppendElements<pxr::GfVec2f>(value, 2u, out.values) ||
                                        AppendElements<pxr::GfVec2d>(value, 2u, out.values)
                                  : AppendElements<pxr::GfVec3f>(value, 3u, out.values) ||
                                        AppendElements<pxr::GfVec3d>(value, 3u, out.values);
    if (!read) {
        return out;
    }
    out.interpolation = primvar.GetInterpolation();
    pxr::VtIntArray indices;
    if (primvar.GetIndices(&indices, kTime)) {
        out.indices.assign(indices.begin(), indices.end());
    }
    return out;
}

// The normals primvar, else the plain normals attribute, as UsdGeomPointBased orders them.
Primvar NormalsOf(const pxr::UsdPrim& prim) {
    Primvar out = ReadPrimvar(prim, pxr::TfToken("normals"), 3u);
    const pxr::UsdGeomMesh mesh(prim);
    pxr::VtVec3fArray normals;
    if (out.values.empty() && mesh.GetNormalsAttr().Get(&normals, kTime)) {
        AppendElements<pxr::GfVec3f>(pxr::VtValue(normals), 3u, out.values);
        out.interpolation = mesh.GetNormalsInterpolation();
    }
    return out;
}

// The element a face corner reads: by point, face or corner as the interpolation says.
size_t ElementOf(const Primvar& primvar, size_t point, size_t face, size_t corner) {
    const pxr::TfToken& mode = primvar.interpolation;
    const size_t element = mode == pxr::UsdGeomTokens->constant      ? 0u
                         : mode == pxr::UsdGeomTokens->uniform       ? face
                         : mode == pxr::UsdGeomTokens->faceVarying   ? corner
                                                                     : point;
    return primvar.indices.empty() ? element : static_cast<size_t>(primvar.indices[element]);
}

// Whether the primvar has an element for every point, face or corner its interpolation addresses.
bool Covers(const Primvar& primvar, size_t points, size_t faces, size_t corners) {
    const pxr::TfToken& mode = primvar.interpolation;
    const size_t needed = mode == pxr::UsdGeomTokens->constant      ? 1u
                        : mode == pxr::UsdGeomTokens->uniform       ? faces
                        : mode == pxr::UsdGeomTokens->faceVarying   ? corners
                                                                    : points;
    const size_t elements = primvar.values.size() / primvar.width;
    if (primvar.indices.empty()) {
        return elements >= needed;
    }
    return primvar.indices.size() >= needed &&
           std::all_of(primvar.indices.begin(), primvar.indices.end(),
                       [&](int i) { return i >= 0 && static_cast<size_t>(i) < elements; });
}

// Triangles ready for a shape record, with one vertex per distinct (point, normal, UV) corner.
struct MeshBuffers {
    std::vector<float> vertices;
    std::vector<float> normals;
    std::vector<float> uvs;
    std::vector<uint32_t> indices;
    size_t dropped_faces = 0;            // faces naming points the mesh does not have
};

// A welded vertex: its point, then the normal and UV it carries (zero when the mesh has none).
struct CornerKey {
    int point = 0;
    float attributes[5] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    bool operator==(const CornerKey& other) const { return std::memcmp(this, &other, sizeof(CornerKey)) == 0; }
};

struct CornerHash {
    size_t operator()(const CornerKey& key) const {
        const auto* bytes = reinterpret_cast<const unsigned char*>(&key);
        uint64_t hash = 14695981039346656037ull;
        for (size_t b = 0; b < sizeof(CornerKey); ++b) {
            hash = (hash ^ bytes[b]) * 1099511628211ull;
        }
        return static_cast<size_t>(hash);
    }
};

// Fan-triangulates `faces`, welding corners that share a point, normal and UV into vertices kept in authored
// point order, so points no face uses drop out; `flip` reverses the winding.
MeshBuffers AssembleMesh(const PrimData& prim, const std::vector<int>& faces, const Primvar& normals,
                         const Primvar& uvs, bool flip) {
    const std::vector<int>& counts = prim.face_vertex_counts;
    const std::vector<int>& corners = prim.face_vertex_indices;
    std::vector<size_t> starts(counts.size() + 1u, 0u);
    for (size_t f = 0; f < counts.size(); ++f) {
        starts[f + 1u] = starts[f] + static_cast<size_t>(std::max(counts[f], 0));
    }
    const int points = static_cast<int>(prim.mesh_points.size() / 3u);
    MeshBuffers out;
    std::unordered_map<CornerKey, uint32_t, CornerHash> welded;
    std::vector<int> point_of;
    const auto vertex = [&](size_t face, size_t corner) {
        CornerKey key;
        key.point = corners[corner];
        const size_t point = static_cast<size_t>(key.point);
        if (!normals.values.empty()) {
            const float* n = &normals.values[3u * ElementOf(normals, point, face, corner)];
            std::copy(n, n + 3, key.attributes);
        }
        if (!uvs.values.empty()) {
            const float* uv = &uvs.values[2u * ElementOf(uvs, point, face, corner)];
            std::copy(uv, uv + 2, key.attributes + 3);
        }
        const auto [it, added] = welded.try_emplace(key, static_cast<uint32_t>(welded.size()));
        if (added) {
            point_of.push_back(key.point);
            const float* p = &prim.mesh_points[3u * point];
            out.vertices.insert(out.vertices.end(), p, p + 3);
            if (!normals.values.empty()) {
                out.normals.insert(out.normals.end(), key.attributes, key.attributes + 3);
            }
            if (!uvs.values.empty()) {
                out.uvs.insert(out.uvs.end(), key.attributes + 3, key.attributes + 5);
            }
        }
        return it->second;
    };
    for (const int face : faces) {
        const size_t f = static_cast<size_t>(face);
        if (counts[f] < 3) {
            continue;
        }
        const size_t start = starts[f];
        const size_t end = starts[f + 1u];
        if (end > corners.size() || std::any_of(corners.begin() + static_cast<std::ptrdiff_t>(start),
                                                corners.begin() + static_cast<std::ptrdiff_t>(end),
                                                [&](int p) { return p < 0 || p >= points; })) {
            ++out.dropped_faces;
            continue;
        }
        for (size_t k = start + 1u; k + 1u < end; ++k) {
            const uint32_t a = vertex(f, start), b = vertex(f, k), c = vertex(f, k + 1u);
            out.indices.insert(out.indices.end(), {a, flip ? c : b, flip ? b : c});
        }
    }
    std::vector<uint32_t> order(point_of.size());
    std::iota(order.begin(), order.end(), 0u);
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) { return point_of[a] < point_of[b]; });
    std::vector<uint32_t> rank(order.size());
    for (uint32_t i = 0; i < order.size(); ++i) {
        rank[order[i]] = i;
    }
    const auto permute = [&](std::vector<float>& values, size_t width) {
        std::vector<float> sorted(values.size());
        for (size_t i = 0; i < order.size(); ++i) {
            std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(order[i] * width), width,
                        sorted.begin() + static_cast<std::ptrdiff_t>(i * width));
        }
        values.swap(sorted);
    };
    permute(out.vertices, 3u);
    if (!out.normals.empty()) {
        permute(out.normals, 3u);
    }
    if (!out.uvs.empty()) {
        permute(out.uvs, 2u);
    }
    for (uint32_t& index : out.indices) {
        index = rank[index];
    }
    return out;
}

// Bakes the scale, mirror or shear a placement leaves over into the points, and its inverse
// transpose into the normals.
void BakeResidual(const pxr::GfMatrix3d& residual, MeshBuffers& mesh) {
    if (IsIdentity(residual)) {
        return;
    }
    for (size_t v = 0; v + 2u < mesh.vertices.size(); v += 3u) {
        float* p = &mesh.vertices[v];
        const pxr::GfVec3d q = pxr::GfVec3d(p[0], p[1], p[2]) * residual;
        p[0] = F(q[0]);
        p[1] = F(q[1]);
        p[2] = F(q[2]);
    }
    const pxr::GfMatrix3d cotangent = residual.GetInverse().GetTranspose();
    for (size_t v = 0; v + 2u < mesh.normals.size(); v += 3u) {
        float* n = &mesh.normals[v];
        const pxr::GfVec3d m = (pxr::GfVec3d(n[0], n[1], n[2]) * cotangent).GetNormalized();
        n[0] = F(m[0]);
        n[1] = F(m[1]);
        n[2] = F(m[2]);
    }
}

// The gprim's displayColor and displayOpacity, inherited from ancestors as constant primvars are, and
// averaged when they vary over the surface.
bool DisplayColorOf(const pxr::UsdPrim& prim, math::Vec3& color, float& opacity) {
    const pxr::UsdGeomPrimvarsAPI primvars(prim);
    pxr::VtVec3fArray colors;
    if (!primvars.FindPrimvarWithInheritance(pxr::TfToken("displayColor")).ComputeFlattened(&colors, kTime) ||
        colors.empty()) {
        return false;
    }
    pxr::GfVec3d sum(0.0);
    for (const pxr::GfVec3f& c : colors) {
        sum += pxr::GfVec3d(c);
    }
    sum /= static_cast<double>(colors.size());
    color = {F(sum[0]), F(sum[1]), F(sum[2])};
    pxr::VtFloatArray opacities;
    if (primvars.FindPrimvarWithInheritance(pxr::TfToken("displayOpacity")).ComputeFlattened(&opacities, kTime) &&
        !opacities.empty()) {
        opacity = F(std::accumulate(opacities.begin(), opacities.end(), 0.0) / static_cast<double>(opacities.size()));
    }
    return true;
}

// "rrggbb" of a colour, then "_aa" when it is not opaque.
std::string Hex(const math::Vec3& color, float alpha) {
    const auto byte = [](float v) { return static_cast<int>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f)); };
    char text[16];
    std::snprintf(text, sizeof(text), "%02x%02x%02x", byte(color.x), byte(color.y), byte(color.z));
    std::string out(text);
    if (alpha < 1.0f) {
        std::snprintf(text, sizeof(text), "_%02x", byte(alpha));
        out += text;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Shading networks
// ---------------------------------------------------------------------------

// The attribute that gives a shader input its value: its own authored value or the output it connects to.
pxr::UsdAttribute SourceOf(const pxr::UsdShadeInput& input) {
    if (!input) {
        return {};
    }
    const pxr::UsdShadeAttributeVector sources = input.GetValueProducingAttributes();
    return sources.empty() ? pxr::UsdAttribute() : sources.front();
}

bool IsOutput(const pxr::UsdAttribute& attr) {
    return attr && pxr::UsdShadeUtils::GetType(attr.GetName()) == pxr::UsdShadeAttributeType::Output;
}

std::string ShaderIdOf(const pxr::UsdPrim& prim) {
    pxr::TfToken id;
    pxr::UsdShadeShader(prim).GetShaderId(&id);
    return id.GetString();
}

// One UsdUVTexture channel feeding a material input.
struct Texture {
    std::string path;                    // resolved file, or "package#offset+length"
    int channel = 0;                     // 0 R, 1 G, 2 B, 3 A; rgb reads from R
    pxr::GfVec4f scale{1.0f};
    pxr::GfVec4f bias{0.0f};
    std::string uv_set = "st";
    float uv_scale = 1.0f;
};

// What the builder keeps about an imported material besides its record.
struct MaterialInfo {
    double density = 0.0;                // PhysicsMaterialAPI density, stage units
    bool display_color = false;          // diffuseColor reads the gprim's displayColor
    std::string uv_set;                  // primvar its textures sample; empty when untextured
};

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

// Authored UsdLux intensity (times 2^exposure) per Nuka radiance unit in Isaac Lab units: an Isaac Lab
// sun of 3000 is a Nuka sun of 3.
constexpr float kUsdLuxToNuka = 1.0f / 1000.0f;

constexpr double kSchemaSunAngle = 0.53;  // degrees

// A light's Nuka intensity at EV0: a sun's nits light a facing surface pi * sin^2(angle / 2) times over, its
// authored angle only under Nits, else 0.53 degrees (50000 gives 3.36); other lights' nits are Nuka units.
float LightIntensity(const PrimData& prim, UsdLightUnits units) {
    const float value = prim.intensity * std::exp2(prim.exposure);
    if (prim.intensity_authored && units != UsdLightUnits::Nits) {
        return units == UsdLightUnits::Isaac ? value * kUsdLuxToNuka : value;
    }
    if (prim.type != "DistantLight") {
        return value;
    }
    const double half = 0.5 * (units == UsdLightUnits::Nits ? prim.light_angle : kSchemaSunAngle) * kDegToRad;
    return F(value * kPi * std::sin(half) * std::sin(half));
}

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
    SceneBuilder(pxr::UsdStageRefPtr stage, const StageUnits& units, const UsdImportOptions& options,
                 std::vector<PrimData> prims, std::vector<UsdDiagnostic>& out)
        : stage_(std::move(stage)), units_(units), options_(options), prims_(std::move(prims)), out_(out) {}

    scene::SceneIR Build();

private:
    void Report(UsdDiagnosticSeverity severity, const char* code, const std::string& path,
                std::string message) const {
        out_.push_back(UsdDiagnostic{severity, code, path, std::move(message)});
    }
    void Error(const char* code, const PrimData& prim, std::string message) const {
        Report(UsdDiagnosticSeverity::Error, code, prim.path, std::move(message));
    }
    void Warn(const char* code, const PrimData& prim, std::string message) const {
        Report(UsdDiagnosticSeverity::Warning, code, prim.path, std::move(message));
    }
    void Warn(const char* code, const std::string& path, std::string message) const {
        Report(UsdDiagnosticSeverity::Warning, code, path, std::move(message));
    }

    int BodyAt(std::string path) const;
    bool ResolveSide(const PrimData& prim, const char* rel, int& body, std::string& target);
    Pose AnchorPose(const std::string& target, const pxr::GfVec3d& position, const pxr::GfQuatd& rotation) const;
    void ReadJoint(size_t index);
    bool ReadD6(const PrimData& prim, Joint& joint);
    void BreakCycles();
    scene::MaterialId MaterialFor(const pxr::UsdPrim& material, scene::SceneIR& scene);
    scene::MaterialRecord ReadMaterial(const pxr::UsdPrim& prim, MaterialInfo& info) const;
    void ReadPreviewSurface(const pxr::UsdShadeShader& surface, const std::string& material,
                            scene::MaterialRecord& record, MaterialInfo& info) const;
    void ReadOmniPbr(const pxr::UsdShadeShader& surface, const std::string& material,
                     scene::MaterialRecord& record, MaterialInfo& info) const;
    bool TextureOf(const pxr::UsdAttribute& source, const char* input, const std::string& material, Texture& out,
                   MaterialInfo& info) const;
    std::string TexturePath(const pxr::SdfAssetPath& file, const std::string& material) const;
    scene::MaterialId LookOf(const pxr::UsdPrim& bound, const PrimData& gprim, scene::SceneIR& scene);
    scene::MaterialId ColliderMaterial(const PrimData& prim, bool drawn, scene::SceneIR& scene);
    scene::MaterialId ColorMaterial(scene::MaterialRecord record, scene::MaterialId from, scene::SceneIR& scene);
    double MaterialDensity(scene::MaterialId id) const;
    void AddGprim(size_t index, scene::SceneIR& scene);
    bool SizePrimitive(const PrimData& prim, const Placement& placement, scene::CollisionShapeRecord& shape) const;
    void ReadContact(const PrimData& prim, scene::CollisionShapeRecord& shape) const;
    void AddMeshCollider(size_t index, int owner, scene::CollisionShapeRecord shape, const Placement& placement,
                         bool flip, scene::SceneIR& scene);
    void AddMeshVisual(size_t index, int owner, scene::CollisionShapeRecord shape, const Placement& placement,
                       bool flip, bool collider, scene::SceneIR& scene);
    void Push(size_t index, int owner, scene::CollisionShapeRecord shape);
    void ApplyCollisionGroups();
    void AddFilteredPairs(scene::SceneIR& scene) const;
    void AddBodies(scene::SceneIR& scene);
    void AddShapes(scene::SceneIR& scene);
    void SetMass(const Body& body, scene::RigidBodyRecord& record) const;
    void AddJoints(scene::SceneIR& scene);
    void AddDrive(const Joint& joint, scene::SceneIR& scene);
    void AddMjcActuator(const PrimData& prim, scene::SceneIR& scene) const;
    void AddDome(const PrimData& prim, scene::SceneIR& scene);
    scene::BodyId WorldBody(scene::SceneIR& scene);
    Pose FrameOf(int body) const { return body < 0 ? Pose{} : bodies_[static_cast<size_t>(body)].frame; }
    scene::BodyId IdOf(int body) const { return static_cast<scene::BodyId>(body); }

    pxr::UsdStageRefPtr stage_;
    StageUnits units_;
    UsdImportOptions options_;
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
    std::unordered_map<std::string, scene::MaterialId> material_at_;     // by Material prim path
    std::unordered_map<scene::MaterialId, MaterialInfo> material_info_;
    std::unordered_map<std::string, scene::MaterialId> color_materials_; // displayColor materials by name
    bool dome_ = false;                                                   // a DomeLight set the environment
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

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------

// The record of a Material prim, added the first time anything asks for it.
scene::MaterialId SceneBuilder::MaterialFor(const pxr::UsdPrim& material, scene::SceneIR& scene) {
    const std::string path = material.GetPath().GetString();
    const auto found = material_at_.find(path);
    if (found != material_at_.end()) {
        return found->second;
    }
    MaterialInfo info;
    const scene::MaterialId id = scene.AddMaterial(ReadMaterial(material, info));
    material_at_[path] = id;
    material_info_[id] = std::move(info);
    return id;
}

// A Material prim's UsdPreviewSurface, or without one the inputs written on the prim itself (Nuka's
// own scenes), plus its UsdPhysics material values.
scene::MaterialRecord SceneBuilder::ReadMaterial(const pxr::UsdPrim& prim, MaterialInfo& info) const {
    const std::string path = prim.GetPath().GetString();
    scene::MaterialRecord record;
    record.name = prim.GetName().GetString();
    const pxr::UsdShadeMaterial material(prim);
    pxr::UsdShadeShader surface = material.ComputeSurfaceSource();
    if (!surface) {
        surface = material.ComputeSurfaceSource(pxr::TfToken("mdl"));
    }
    pxr::SdfAssetPath mdl;
    if (surface && ShaderIdOf(surface.GetPrim()) == "UsdPreviewSurface") {
        ReadPreviewSurface(surface, path, record, info);
    } else if (surface && surface.GetSourceAsset(&mdl, pxr::TfToken("mdl"))) {
        if (std::filesystem::path(mdl.GetAssetPath()).filename() == "OmniPBR.mdl") {
            ReadOmniPbr(surface, path, record, info);
        } else {
            Warn("USD_MATERIAL_MDL_UNSUPPORTED", path, "MDL material '" + mdl.GetAssetPath() +
                 "' is not supported; the material keeps default values");
        }
    } else {
        if (!Read(prim, "inputs:color", record.base_color) && !Read(prim, "inputs:diffuseColor", record.base_color)) {
            Read(prim, "color", record.base_color);
        }
        Read(prim, "inputs:alpha", record.alpha);
        Read(prim, "inputs:roughness", record.roughness);
        Read(prim, "inputs:metallic", record.metallic);
    }
    Read(prim, "physics:dynamicFriction", record.friction_mu);
    Read(prim, "physics:density", info.density);
    double restitution = 0.0;
    if (Read(prim, "physics:restitution", restitution) && restitution > 0.0) {
        Warn("USD_MATERIAL_RESTITUTION", path, "restitution is not represented; contacts stay inelastic");
    }
    return record;
}

// UsdPreviewSurface inputs, each a constant or a UsdUVTexture channel; unauthored inputs keep the spec's
// defaults.
void SceneBuilder::ReadPreviewSurface(const pxr::UsdShadeShader& surface, const std::string& material,
                                      scene::MaterialRecord& record, MaterialInfo& info) const {
    const auto source = [&](const char* name) { return SourceOf(surface.GetInput(pxr::TfToken(name))); };
    record.base_color = {0.18f, 0.18f, 0.18f};
    record.ior = 1.5f;
    Texture map;
    const pxr::UsdAttribute diffuse = source("diffuseColor");
    if (IsOutput(diffuse) && StartsWith(ShaderIdOf(diffuse.GetPrim()), "UsdPrimvarReader")) {
        const pxr::UsdShadeShader reader(diffuse.GetPrim());
        std::string varname;
        ReadSource(SourceOf(reader.GetInput(pxr::TfToken("varname"))), varname);
        ReadSource(SourceOf(reader.GetInput(pxr::TfToken("fallback"))), record.base_color);
        info.display_color = varname == "displayColor";
        if (!info.display_color) {
            Warn("USD_MATERIAL_PRIMVAR", material, "diffuseColor reads primvar '" + varname +
                 "', which is not supported; its fallback colour is used");
        }
    } else if (TextureOf(diffuse, "diffuseColor", material, map, info)) {
        record.albedo_map = map.path;
        record.base_color = {map.scale[0], map.scale[1], map.scale[2]};
        record.uv_scale = map.uv_scale;
    } else {
        ReadSource(diffuse, record.base_color);
    }
    const auto scalar = [&](const char* input, float& value, std::string& path, uint32_t& channel) {
        const pxr::UsdAttribute attr = source(input);
        if (!TextureOf(attr, input, material, map, info)) {
            ReadSource(attr, value);
        } else if (map.channel == 3) {
            Warn("USD_TEXTURE_CHANNEL", material, std::string(input) + " reads a texture's alpha channel, "
                 "which is not supported; the map is left out");
        } else {
            path = map.path;
            channel = static_cast<uint32_t>(map.channel);
            value = map.scale[map.channel];
        }
    };
    scalar("roughness", record.roughness, record.roughness_map, record.roughness_channel);
    scalar("metallic", record.metallic, record.metallic_map, record.metallic_channel);
    const pxr::UsdAttribute emissive = source("emissiveColor");
    if (TextureOf(emissive, "emissiveColor", material, map, info)) {
        record.emissive_map = map.path;
        record.emissive = {map.scale[0], map.scale[1], map.scale[2]};
    } else {
        ReadSource(emissive, record.emissive);
    }
    if (TextureOf(source("normal"), "normal", material, map, info)) {
        record.normal_map = map.path;
        const bool standard = map.scale[0] == 2.0f && map.scale[1] == 2.0f && map.scale[2] == 2.0f &&
                              map.bias[0] == -1.0f && map.bias[1] == -1.0f && map.bias[2] == -1.0f;
        if (!standard) {
            Warn("USD_NORMAL_MAP_ENCODING", material, "the normal map's scale and bias are not (2, 2, 2) and "
                 "(-1, -1, -1); it is read as a standard tangent-space map");
        }
    }
    const pxr::UsdAttribute opacity = source("opacity");
    if (IsOutput(opacity)) {
        Warn("USD_MATERIAL_INPUT_UNSUPPORTED", material, "a textured opacity is not supported; the material "
             "stays opaque");
    } else {
        ReadSource(opacity, record.alpha);
    }
    ReadSource(source("ior"), record.ior);
    int specular = 0;
    if (ReadSource(source("useSpecularWorkflow"), specular) && specular != 0) {
        Warn("USD_MATERIAL_SPECULAR", material, "the specular workflow is not supported; the material is read "
             "as metallic-roughness");
    }
    record.triplanar = record.albedo_map.empty() && record.roughness_map.empty() && record.metallic_map.empty() &&
                       record.normal_map.empty() && record.emissive_map.empty();
}

// OmniPBR.mdl parameters, unauthored ones at NVIDIA's documented defaults; a map that replaces a value
// leaves the value at 1, a multiplier.
void SceneBuilder::ReadOmniPbr(const pxr::UsdShadeShader& surface, const std::string& material,
                               scene::MaterialRecord& record, MaterialInfo& info) const {
    const auto read = [&](const char* name, auto& value) {
        return ReadSource(SourceOf(surface.GetInput(pxr::TfToken(name))), value);
    };
    const auto map = [&](const char* name) {
        pxr::SdfAssetPath file;
        read(name, file);
        return TexturePath(file, material);
    };
    math::Vec3 constant{0.2f, 0.2f, 0.2f}, tint{1.0f, 1.0f, 1.0f};
    float brightness = 1.0f;
    read("diffuse_color_constant", constant);
    read("diffuse_tint", tint);
    read("albedo_brightness", brightness);
    record.albedo_map = map("diffuse_texture");
    const math::Vec3 albedo =
        record.albedo_map.empty() ? math::Vec3{constant.x * tint.x, constant.y * tint.y, constant.z * tint.z} : tint;
    record.base_color = albedo * brightness;

    // Roughness and metallic blend a constant toward a map by an influence; with ORM on, the map is ORM's G or B.
    bool orm = false;
    read("enable_ORM_texture", orm);
    const auto scalar = [&](const char* name, float& value, const char* texture, uint32_t orm_channel,
                            std::string& path, uint32_t& channel) {
        read((std::string(name) + "_constant").c_str(), value);
        float influence = 0.0f;
        read((std::string(name) + "_texture_influence").c_str(), influence);
        if (influence <= 0.0f) {
            return;
        }
        const std::string file = map(orm ? "ORM_texture" : texture);
        if (file.empty()) {
            return;
        }
        if (influence != 1.0f) {
            Warn("USD_TEXTURE_INFLUENCE", material, std::string(name) + "_texture_influence " +
                 std::to_string(influence) + " is read as 1; the map replaces the constant");
        }
        path = file;
        channel = orm ? orm_channel : 0u;
        value = 1.0f;
    };
    record.roughness = 0.5f;
    scalar("reflection_roughness", record.roughness, "reflectionroughness_texture", 1u, record.roughness_map,
           record.roughness_channel);
    record.metallic = 0.0f;
    scalar("metallic", record.metallic, "metallic_texture", 2u, record.metallic_map, record.metallic_channel);
    record.normal_map = map("normalmap_texture");

    // Emission reads in the lights' units.
    bool emission = false;
    read("enable_emission", emission);
    if (emission) {
        math::Vec3 color{1.0f, 1.0f, 0.1f};
        float intensity = 1.0f;
        read("emissive_color", color);
        read("emissive_intensity", intensity);
        record.emissive_map = map("emissive_color_texture");
        const float scale = intensity * (options_.light_units == UsdLightUnits::Isaac ? kUsdLuxToNuka : 1.0f);
        record.emissive = record.emissive_map.empty() ? color * scale : math::Vec3{scale, scale, scale};
    }
    bool opacity = false, opacity_texture = false;
    read("enable_opacity", opacity);
    read("enable_opacity_texture", opacity_texture);
    if (opacity) {
        read("opacity_constant", record.alpha);
        if (opacity_texture) {
            Warn("USD_MATERIAL_INPUT_UNSUPPORTED", material, "a textured opacity is not supported; "
                 "opacity_constant is used");
        }
    }

    // A larger texture_scale draws the texture larger: OmniPBR divides the UVs by it.
    pxr::GfVec2f scale(1.0f), translate(0.0f);
    float rotate = 0.0f;
    read("texture_scale", scale);
    read("texture_translate", translate);
    read("texture_rotate", rotate);
    if (scale[0] != scale[1] || translate != pxr::GfVec2f(0.0f) || rotate != 0.0f) {
        Warn("USD_TEXTURE_TRANSFORM", material, "only a uniform texture_scale is kept from the OmniPBR UV "
             "transform");
    }
    record.uv_scale = 1.0f / scale[0];
    bool project = false;
    read("project_uvw", project);
    const bool textured = !record.albedo_map.empty() || !record.roughness_map.empty() ||
                          !record.metallic_map.empty() || !record.normal_map.empty() || !record.emissive_map.empty();
    record.triplanar = project || !textured;
    if (textured && !project) {
        info.uv_set = "st";
    }
}

// The UsdUVTexture channel `source` names, with the UV set its st reads and a uniform UsdTransform2d scale.
bool SceneBuilder::TextureOf(const pxr::UsdAttribute& source, const char* input, const std::string& material,
                             Texture& out, MaterialInfo& info) const {
    if (!IsOutput(source)) {
        return false;
    }
    const pxr::UsdShadeShader shader(source.GetPrim());
    const std::string id = ShaderIdOf(source.GetPrim());
    if (id != "UsdUVTexture") {
        Warn("USD_MATERIAL_INPUT_UNSUPPORTED", material, std::string(input) + " comes from a '" + id +
             "' shader, which is not supported; the default is used");
        return false;
    }
    out = Texture{};
    const std::string channel = pxr::UsdShadeOutput(source).GetBaseName().GetString();
    out.channel = channel == "g" ? 1 : channel == "b" ? 2 : channel == "a" ? 3 : 0;
    ReadSource(SourceOf(shader.GetInput(pxr::TfToken("scale"))), out.scale);
    ReadSource(SourceOf(shader.GetInput(pxr::TfToken("bias"))), out.bias);
    pxr::SdfAssetPath file;
    ReadSource(SourceOf(shader.GetInput(pxr::TfToken("file"))), file);
    out.path = TexturePath(file, material);
    pxr::UsdAttribute st = SourceOf(shader.GetInput(pxr::TfToken("st")));
    if (IsOutput(st) && ShaderIdOf(st.GetPrim()) == "UsdTransform2d") {
        const pxr::UsdShadeShader transform(st.GetPrim());
        pxr::GfVec2f scale(1.0f), translation(0.0f);
        float rotation = 0.0f;
        ReadSource(SourceOf(transform.GetInput(pxr::TfToken("scale"))), scale);
        ReadSource(SourceOf(transform.GetInput(pxr::TfToken("translation"))), translation);
        ReadSource(SourceOf(transform.GetInput(pxr::TfToken("rotation"))), rotation);
        if (scale[0] != scale[1] || rotation != 0.0f || translation != pxr::GfVec2f(0.0f)) {
            Warn("USD_TEXTURE_TRANSFORM", material, "only a uniform scale is kept from UV transform <" +
                 transform.GetPath().GetString() + ">");
        }
        out.uv_scale = scale[0];
        st = SourceOf(transform.GetInput(pxr::TfToken("in")));
    }
    if (IsOutput(st) && StartsWith(ShaderIdOf(st.GetPrim()), "UsdPrimvarReader")) {
        ReadSource(SourceOf(pxr::UsdShadeShader(st.GetPrim()).GetInput(pxr::TfToken("varname"))), out.uv_set);
    }
    if (out.path.empty()) {
        return false;
    }
    if (info.uv_set.empty()) {
        info.uv_set = out.uv_set;
    } else if (info.uv_set != out.uv_set) {
        Warn("USD_TEXTURE_UV_SETS", material, "textures sample both '" + info.uv_set + "' and '" + out.uv_set +
             "'; all of them use '" + info.uv_set + "'");
    }
    return true;
}

// A texture file the renderer can open: the resolved path, or "package#offset+length" for an image stored
// in a .usdz; missing files, UDIM sets and formats it cannot decode give "".
std::string SceneBuilder::TexturePath(const pxr::SdfAssetPath& file, const std::string& material) const {
    const std::string& authored = file.GetAssetPath();
    if (authored.empty()) {
        return {};
    }
    if (authored.find("<UDIM>") != std::string::npos) {
        Warn("USD_TEXTURE_UDIM", material, "UDIM texture '" + authored + "' is not supported; it is left out");
        return {};
    }
    const std::string& resolved = file.GetResolvedPath();
    if (resolved.empty()) {
        Warn("USD_TEXTURE_MISSING", material, "texture '" + authored + "' was not found; it is left out");
        return {};
    }
    std::string package, inner = resolved;
    if (pxr::ArIsPackageRelativePath(resolved)) {
        std::tie(package, inner) = pxr::ArSplitPackageRelativePathOuter(resolved);
    }
    const std::string extension = Lowercase(std::filesystem::path(inner).extension().string());
    if (extension != ".png" && extension != ".jpg" && extension != ".jpeg" && extension != ".tga" &&
        extension != ".bmp" && extension != ".hdr") {
        Warn("USD_TEXTURE_FORMAT", material, "texture '" + authored + "' is a " + extension +
             " file, which the renderer cannot decode; it is left out");
        return {};
    }
    if (package.empty()) {
        return resolved;
    }
    const pxr::SdfZipFile zip = pxr::SdfZipFile::Open(package);
    if (zip) {
        const pxr::SdfZipFile::Iterator entry = zip.Find(inner);
        if (entry != zip.end()) {
            const pxr::SdfZipFile::FileInfo stored = entry.GetFileInfo();
            if (stored.compressionMethod == 0 && !stored.encrypted) {
                return package + "#" + std::to_string(stored.dataOffset) + "+" + std::to_string(stored.size);
            }
            Warn("USD_TEXTURE_FORMAT", material, "texture '" + authored + "' is compressed inside the package; "
                 "it is left out");
            return {};
        }
    }
    Warn("USD_TEXTURE_MISSING", material, "texture '" + authored + "' is not in " + package + "; it is left out");
    return {};
}

// What a drawn gprim, or one of its subsets, looks like: its preview binding (copied per colour when the
// material reads displayColor), else a material per displayColor.
scene::MaterialId SceneBuilder::LookOf(const pxr::UsdPrim& bound, const PrimData& gprim, scene::SceneIR& scene) {
    const pxr::UsdShadeMaterial material =
        pxr::UsdShadeMaterialBindingAPI(bound).ComputeBoundMaterial(pxr::UsdShadeTokens->preview);
    math::Vec3 color;
    float opacity = 1.0f;
    const bool colored = DisplayColorOf(gprim.prim, color, opacity);
    if (material) {
        const scene::MaterialId id = MaterialFor(material.GetPrim(), scene);
        if (!colored || !material_info_.at(id).display_color) {
            return id;
        }
        scene::MaterialRecord copy = scene.GetMaterial(id);
        copy.name += "_" + Hex(color, opacity);
        copy.base_color = color;
        copy.alpha *= opacity;
        return ColorMaterial(std::move(copy), id, scene);
    }
    if (!colored) {
        return scene::kInvalidMaterial;
    }
    scene::MaterialRecord record;
    record.name = "displayColor_" + Hex(color, opacity);
    record.base_color = color;
    record.alpha = opacity;
    return ColorMaterial(std::move(record), scene::kInvalidMaterial, scene);
}

// One record per colour name; a copy of `from` keeps its textures' UV set and its density.
scene::MaterialId SceneBuilder::ColorMaterial(scene::MaterialRecord record, scene::MaterialId from,
                                              scene::SceneIR& scene) {
    const auto found = color_materials_.find(record.name);
    if (found != color_materials_.end()) {
        return found->second;
    }
    MaterialInfo info = from == scene::kInvalidMaterial ? MaterialInfo{} : material_info_.at(from);
    info.display_color = false;
    const std::string name = record.name;
    const scene::MaterialId id = scene.AddMaterial(std::move(record));
    color_materials_[name] = id;
    material_info_[id] = std::move(info);
    return id;
}

// A collider's physics binding, which falls back to its allPurpose one; a collider drawn as itself takes
// its look when it has neither.
scene::MaterialId SceneBuilder::ColliderMaterial(const PrimData& prim, bool drawn, scene::SceneIR& scene) {
    const pxr::UsdShadeMaterial material =
        pxr::UsdShadeMaterialBindingAPI(prim.prim).ComputeBoundMaterial(pxr::TfToken("physics"));
    if (material) {
        return MaterialFor(material.GetPrim(), scene);
    }
    return drawn ? LookOf(prim.prim, prim, scene) : scene::kInvalidMaterial;
}

double SceneBuilder::MaterialDensity(scene::MaterialId id) const {
    const auto found = material_info_.find(id);
    return found == material_info_.end() ? 0.0 : found->second.density;
}

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

// A gprim gives a collider when it has CollisionAPI and a visual when it is drawn; a primitive collider
// is drawn as itself, so only meshes and primitives that never collide add a visual record.
void SceneBuilder::AddGprim(size_t index, scene::SceneIR& scene) {
    const PrimData& prim = prims_[index];
    bool enabled = true;
    Read(prim.prim, "physics:collisionEnabled", enabled);
    const bool collider = prim.HasApi("PhysicsCollisionAPI") && enabled;
    const bool drawn = prim.visible && (prim.purpose == "default" || prim.purpose == "render");
    if (!collider && !drawn) {
        return;
    }
    if (prim.type == "Cone") {
        Warn("USD_SHAPE_UNSUPPORTED", prim, "cones are not supported; the cone is left out");
        return;
    }
    const int owner = BodyAt(prim.path);
    const Placement placement = Place(prim.world);
    scene::CollisionShapeRecord shape;
    shape.name = prim.name;
    shape.body_id = owner < 0 ? scene::kInvalidBody : IdOf(owner);
    shape.local_transform = ToTransform(FrameOf(owner).Inverse() * placement.pose);
    if (prim.type == "Mesh") {
        const bool flip = prim.left_handed != (placement.residual.GetDeterminant() < 0.0);
        if (collider) {
            AddMeshCollider(index, owner, shape, placement, flip, scene);
        }
        if (drawn) {
            AddMeshVisual(index, owner, shape, placement, flip, collider, scene);
        }
        return;
    }
    if (!SizePrimitive(prim, placement, shape)) {
        return;
    }
    if (collider) {
        ReadContact(prim, shape);
        shape.material_id = ColliderMaterial(prim, drawn, scene);
    } else {
        shape.contype = 0;
        shape.conaffinity = 0;
        shape.material_id = LookOf(prim.prim, prim, scene);
    }
    Push(index, owner, std::move(shape));
}

// A primitive's type and size; scale off its axes is kept per axis, shear is dropped.
bool SceneBuilder::SizePrimitive(const PrimData& prim, const Placement& placement,
                                 scene::CollisionShapeRecord& shape) const {
    PrimitiveType(prim.type, shape.type);
    const pxr::GfVec3d scale = ScaleOf(placement.residual);
    if (Sheared(placement.residual)) {
        Warn("USD_SHAPE_SHEARED", prim, "the shape's transform shears it; only its axis scales are kept");
    }
    const auto uniform = [&](double a, double b) { return std::abs(a - b) <= 1e-6 * std::max(a, b); };
    if (shape.type == scene::ShapeType::Box) {
        const double half = prim.cube_size * 0.5;
        shape.half_extents = {F(half * scale[0]), F(half * scale[1]), F(half * scale[2])};
    } else if (shape.type == scene::ShapeType::Sphere) {
        if (!uniform(scale[0], scale[1]) || !uniform(scale[0], scale[2])) {
            Warn("USD_SCALE_NONUNIFORM", prim, "a sphere scaled unevenly keeps its largest radius");
        }
        shape.radius = F(prim.radius * std::max({scale[0], scale[1], scale[2]}));
    } else {
        math::Quat alignment;
        if (!AxisAlignment(prim.geometry_axis_token, alignment)) {
            Error("USD_SHAPE_AXIS_INVALID", prim, "invalid " + prim.type + " axis '" + prim.geometry_axis_token +
                  "'; the shape is left out");
            return false;
        }
        const int along = Lowercase(prim.geometry_axis_token) == "x" ? 0
                        : Lowercase(prim.geometry_axis_token) == "y" ? 1 : 2;
        if (shape.type == scene::ShapeType::Plane) {
            // USD runs width along X unless the axis is X, and length along Y unless the axis is Y.
            const int width_axis = along == 0 ? 2 : 0, length_axis = along == 1 ? 2 : 1;
            shape.half_extents = {F(prim.plane_width * 0.5 * scale[width_axis]),
                                  F(prim.plane_length * 0.5 * scale[length_axis]), 0.0f};
        } else {
            const double across0 = scale[(along + 1) % 3], across1 = scale[(along + 2) % 3];
            if (!uniform(across0, across1)) {
                Warn("USD_SCALE_NONUNIFORM", prim, "a " + prim.type + " scaled unevenly across its axis "
                     "keeps its largest radius");
            }
            shape.radius = F(prim.radius * std::max(across0, across1));
            shape.half_height = F(prim.height * 0.5 * scale[along]);
        }
        shape.local_transform.rotation = (shape.local_transform.rotation * alignment).Normalized();
    }
    return true;
}

// MjcCollisionAPI contact parameters, copied as the MJCF importer copies a geom's.
void SceneBuilder::ReadContact(const PrimData& prim, scene::CollisionShapeRecord& shape) const {
    int condim = 0;
    if (Read(prim.prim, "mjc:condim", condim)) {
        shape.condim = static_cast<uint8_t>(condim);
    }
    Read(prim.prim, "mjc:group", shape.collision_group);
    Read(prim.prim, "mjc:priority", shape.priority);
    Read(prim.prim, "mjc:solmix", shape.solmix);
    double length = 0.0;
    if (Read(prim.prim, "mjc:margin", length)) {
        shape.margin = F(length * units_.meters);
    }
    if (Read(prim.prim, "mjc:gap", length)) {
        shape.gap = F(length * units_.meters);
    }
    pxr::VtArray<double> values;
    if (Read(prim.prim, "mjc:solref", values)) {
        for (size_t k = 0; k < std::min<size_t>(values.size(), 2u); ++k) {
            shape.solref[k] = F(values[k]);
        }
    }
    if (Read(prim.prim, "mjc:solimp", values)) {
        for (size_t k = 0; k < std::min<size_t>(values.size(), 5u); ++k) {
            shape.solimp[k] = F(values[k]);
        }
    }
}

// A mesh collider as its MeshCollisionAPI approximation asks; nuka:decompose wins over the approximation.
void SceneBuilder::AddMeshCollider(size_t index, int owner, scene::CollisionShapeRecord shape,
                                   const Placement& placement, bool flip, scene::SceneIR& scene) {
    const PrimData& prim = prims_[index];
    std::vector<int> faces(prim.face_vertex_counts.size());
    std::iota(faces.begin(), faces.end(), 0);
    MeshBuffers mesh = AssembleMesh(prim, faces, Primvar{}, Primvar{}, flip);
    if (mesh.dropped_faces > 0u) {
        Warn("USD_MESH_INVALID", prim, std::to_string(mesh.dropped_faces) + " faces name points the mesh does "
             "not have; they are left out");
    }
    if (mesh.indices.empty()) {
        return;
    }
    BakeResidual(placement.residual, mesh);
    shape.type = scene::ShapeType::TriMesh;
    shape.mesh_vertices = std::move(mesh.vertices);
    shape.mesh_indices = std::move(mesh.indices);
    shape.decompose_mode = DecomposeModeFromToken(prim.decompose_mode);
    if (prim.decompose_max_pieces > 0) {
        shape.decompose_max_pieces = static_cast<uint32_t>(prim.decompose_max_pieces);
    }
    const std::string& approximation = prim.approximation;
    if (approximation == "convexHull") {
        // PhysX cooks the hull of the points, so a concave source mesh collides as its hull.
        cooker::ConvexPiece hull = cooker::ComputeConvexHull(
            shape.mesh_vertices.data(), static_cast<uint32_t>(shape.mesh_vertices.size() / 3u),
            static_cast<uint32_t>(prim.hull_vertex_limit));
        if (hull.volume > 0.0f) {
            shape.type = scene::ShapeType::ConvexHull;
            shape.mesh_vertices = std::move(hull.vertices);
            shape.mesh_indices = std::move(hull.indices);
        } else {
            Warn("USD_HULL_DEGENERATE", prim, "the mesh's points span no volume; it collides as a triangle mesh");
        }
    } else if (approximation == "convexDecomposition") {
        if (prim.decompose_mode.empty()) {
            shape.decompose_mode = scene::DecomposeMode::Force;
            if (prim.max_hulls > 0) {
                shape.decompose_max_pieces = static_cast<uint32_t>(prim.max_hulls);
            }
        }
    } else if (approximation == "boundingSphere" || approximation == "boundingCube") {
        pxr::GfVec3d low(kInf), high(-kInf);
        for (size_t v = 0; v + 2u < shape.mesh_vertices.size(); v += 3u) {
            for (int k = 0; k < 3; ++k) {
                low[k] = std::min(low[k], static_cast<double>(shape.mesh_vertices[v + k]));
                high[k] = std::max(high[k], static_cast<double>(shape.mesh_vertices[v + k]));
            }
        }
        const pxr::GfVec3d center = 0.5 * (low + high);
        if (approximation == "boundingSphere") {
            double radius = 0.0;
            for (size_t v = 0; v + 2u < shape.mesh_vertices.size(); v += 3u) {
                const float* p = &shape.mesh_vertices[v];
                radius = std::max(radius, (pxr::GfVec3d(p[0], p[1], p[2]) - center).GetLength());
            }
            shape.type = scene::ShapeType::Sphere;
            shape.radius = F(radius);
        } else {
            const pxr::GfVec3d half = 0.5 * (high - low);
            shape.type = scene::ShapeType::Box;
            shape.half_extents = {F(half[0]), F(half[1]), F(half[2])};
        }
        const math::Vec3 offset{F(center[0]), F(center[1]), F(center[2])};
        shape.local_transform.position = shape.local_transform.position + shape.local_transform.rotation.Rotate(offset);
        shape.mesh_vertices.clear();
        shape.mesh_indices.clear();
    } else if (approximation == "sdf") {
        shape.mesh_contact = scene::CollisionShapeRecord::MeshContact::Sdf;
    } else if (!approximation.empty() && approximation != "none" && approximation != "meshSimplification") {
        Warn("USD_APPROXIMATION_UNSUPPORTED", prim, "collision approximation '" + approximation +
             "' is not supported; the mesh collides as authored");
    }
    ReadContact(prim, shape);
    shape.material_id = ColliderMaterial(prim, false, scene);
    Push(index, owner, std::move(shape));
}

// A drawn mesh as visual records: one per materialBind subset with its own material, and one for the
// faces no subset takes.
void SceneBuilder::AddMeshVisual(size_t index, int owner, scene::CollisionShapeRecord shape,
                                 const Placement& placement, bool flip, bool collider, scene::SceneIR& scene) {
    const PrimData& prim = prims_[index];
    const size_t points = prim.mesh_points.size() / 3u;
    const size_t faces = prim.face_vertex_counts.size();
    const size_t corners = prim.face_vertex_indices.size();
    Primvar normals = NormalsOf(prim.prim);
    if (!normals.values.empty() && !Covers(normals, points, faces, corners)) {
        Warn("USD_PRIMVAR_MISMATCH", prim, "the normals do not cover the mesh; they are recomputed");
        normals = Primvar{};
    }
    std::vector<std::pair<pxr::UsdPrim, std::vector<int>>> parts;
    std::vector<char> taken(faces, 0);
    for (const pxr::UsdGeomSubset& subset : pxr::UsdShadeMaterialBindingAPI(prim.prim).GetMaterialBindSubsets()) {
        pxr::VtIntArray indices;
        subset.GetIndicesAttr().Get(&indices, kTime);
        std::vector<int> list;
        for (const int f : indices) {
            if (f >= 0 && static_cast<size_t>(f) < faces && !taken[static_cast<size_t>(f)]) {
                taken[static_cast<size_t>(f)] = 1;
                list.push_back(f);
            }
        }
        parts.emplace_back(subset.GetPrim(), std::move(list));
    }
    std::vector<int> rest;
    for (size_t f = 0; f < faces; ++f) {
        if (!taken[f]) {
            rest.push_back(static_cast<int>(f));
        }
    }
    parts.emplace_back(prim.prim, std::move(rest));
    const std::string base = collider ? prim.name + "_visual" : prim.name;
    std::unordered_map<std::string, Primvar> uv_sets;
    size_t dropped = 0;
    for (const auto& [bound, list] : parts) {
        if (list.empty()) {
            continue;
        }
        scene::CollisionShapeRecord part = shape;
        part.name = bound == prim.prim ? base : base + "_" + bound.GetName().GetString();
        part.contype = 0;
        part.conaffinity = 0;
        part.material_id = LookOf(bound, prim, scene);
        const std::string uv_set = part.material_id == scene::kInvalidMaterial
                                 ? std::string() : material_info_.at(part.material_id).uv_set;
        auto uvs = uv_sets.find(uv_set);
        if (uvs == uv_sets.end()) {
            Primvar read = uv_set.empty() ? Primvar{} : ReadPrimvar(prim.prim, pxr::TfToken(uv_set), 2u);
            if (!uv_set.empty() && read.values.empty()) {
                Warn("USD_PRIMVAR_MISSING", prim, "the mesh has no '" + uv_set + "' primvar for its textures");
            } else if (!read.values.empty() && !Covers(read, points, faces, corners)) {
                Warn("USD_PRIMVAR_MISMATCH", prim, "primvar '" + uv_set + "' does not cover the mesh; it is "
                     "left out");
                read = Primvar{};
            }
            uvs = uv_sets.emplace(uv_set, std::move(read)).first;
        }
        MeshBuffers mesh = AssembleMesh(prim, list, normals, uvs->second, flip);
        dropped += mesh.dropped_faces;
        if (mesh.indices.empty()) {
            continue;
        }
        BakeResidual(placement.residual, mesh);
        part.type = scene::ShapeType::TriMesh;
        part.mesh_vertices = std::move(mesh.vertices);
        part.mesh_normals = std::move(mesh.normals);
        part.mesh_uvs = std::move(mesh.uvs);
        part.mesh_indices = std::move(mesh.indices);
        Push(index, owner, std::move(part));
    }
    if (dropped > 0u && !collider) {
        Warn("USD_MESH_INVALID", prim, std::to_string(dropped) + " faces name points the mesh does not have; "
             "they are left out");
    }
}

void SceneBuilder::Push(size_t index, int owner, scene::CollisionShapeRecord shape) {
    shapes_.push_back(std::move(shape));
    shape_sources_.emplace_back(index, owner);
}

// UsdPhysics collision groups as contype and conaffinity bits: bit 0 stands for every collider outside the
// groups and bit g + 1 for group g, so a stage without groups keeps 1 and 1.
void SceneBuilder::ApplyCollisionGroups() {
    const pxr::UsdPhysicsCollisionGroup::CollisionGroupTable table =
        pxr::UsdPhysicsCollisionGroup::ComputeCollisionGroupTable(*stage_);
    const pxr::SdfPathVector& groups = table.GetCollisionGroups();
    const size_t count = std::min<size_t>(groups.size(), 31u);
    if (groups.size() > count) {
        Warn("USD_COLLISION_GROUP_LIMIT", groups[count].GetString(), "only 31 collision groups fit the "
             "contact bits; this group and the ones after it collide with everything");
    }
    std::vector<pxr::UsdCollectionMembershipQuery> members;
    std::vector<uint32_t> accepts(count, 1u);
    for (size_t g = 0; g < count; ++g) {
        const pxr::UsdPhysicsCollisionGroup group(stage_->GetPrimAtPath(groups[g]));
        members.push_back(group.GetCollidersCollectionAPI().ComputeMembershipQuery());
        for (size_t h = 0; h < count; ++h) {
            if (table.IsCollisionEnabled(static_cast<unsigned int>(g), static_cast<unsigned int>(h))) {
                accepts[g] |= 1u << (h + 1u);
            }
        }
    }
    for (size_t s = 0; s < shapes_.size(); ++s) {
        if (shapes_[s].contype == 0u) {
            continue;
        }
        const pxr::SdfPath path = prims_[shape_sources_[s].first].prim.GetPath();
        uint32_t type = 0u, accept = ~0u;
        for (size_t g = 0; g < count; ++g) {
            if (members[g].IsPathIncluded(path)) {
                type |= 1u << (g + 1u);
                accept &= accepts[g];
            }
        }
        if (type != 0u) {
            shapes_[s].contype = type;
            shapes_[s].conaffinity = accept;
        }
    }
}

// FilteredPairsAPI as body exclude pairs; the engine filters whole bodies, so a filter set on a collider
// widens to its body.
void SceneBuilder::AddFilteredPairs(scene::SceneIR& scene) const {
    for (const PrimData& prim : prims_) {
        if (!prim.HasApi("PhysicsFilteredPairsAPI")) {
            continue;
        }
        pxr::SdfPathVector targets;
        prim.prim.GetRelationship(pxr::TfToken("physics:filteredPairs")).GetTargets(&targets);
        const int a = BodyAt(prim.path);
        for (const pxr::SdfPath& target : targets) {
            const int b = BodyAt(target.GetString());
            if (a < 0 || b < 0) {
                Warn("USD_FILTER_PAIR_UNSUPPORTED", prim, "the filter with <" + target.GetString() +
                     "> does not join two rigid bodies; it is dropped");
                continue;
            }
            if (a == b) {
                continue;
            }
            if (prims_[bodies_[static_cast<size_t>(a)].prim].path != prim.path ||
                prims_[bodies_[static_cast<size_t>(b)].prim].path != target.GetString()) {
                Warn("USD_FILTER_PAIR_WIDENED", prim, "the filter with <" + target.GetString() +
                     "> applies to the whole bodies on both sides");
            }
            scene.AddExcludePair(IdOf(a), IdOf(b));
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
        const double material_density = MaterialDensity(shapes_[s].material_id);
        const double rho = density > 0.0 ? density * density_unit
                         : body_density > 0.0 ? body_density * density_unit
                         : material_density > 0.0 ? material_density * density_unit : kDefaultDensity;
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
        // Passive dynamics, MuJoCo's attributes first; armature and damping scale as inertia, friction as force.
        const double inertia = units_.kilograms * (joint.angular ? units_.meters * units_.meters : 1.0);
        const double force = units_.kilograms * units_.meters * (joint.angular ? units_.meters : 1.0);
        double armature = 0.0, friction = 0.0, damping = 0.0, velocity = 0.0;
        if (Read(prim.prim, "mjc:armature", armature) || Read(prim.prim, "newton:armature", armature) ||
            Read(prim.prim, "physxJoint:armature", armature)) {
            record.armature = F(armature * inertia);
        }
        if (Read(prim.prim, "mjc:frictionloss", friction) || Read(prim.prim, "newton:friction", friction)) {
            record.frictionloss = F(friction * force);
        }
        if (Read(prim.prim, "mjc:damping", damping)) {
            record.damping = F(damping * inertia);
        }
        if (Read(prim.prim, "physxJoint:maxJointVelocity", velocity)) {
            Warn("USD_JOINT_VELOCITY_LIMIT", prim, "joint velocity limits are not represented; the limit of " +
                 std::to_string(velocity) + " is dropped");
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

// A MuJoCo actuator on a joint: a motor, a position servo (gain kp, affine bias (0, -kp, b2)) or a velocity
// servo (gain kv, bias (0, 0, -kv)). The joint feels gear^2 times the servo gain.
void SceneBuilder::AddMjcActuator(const PrimData& prim, scene::SceneIR& scene) const {
    const auto params = [&](const char* name, double first) {
        std::vector<double> values(10, 0.0);
        values[0] = first;
        pxr::VtArray<double> authored;
        if (Read(prim.prim, name, authored)) {
            std::copy_n(authored.begin(), std::min<size_t>(authored.size(), values.size()), values.begin());
        }
        return values;
    };
    const std::vector<double> gear = params("mjc:gear", 1.0);
    const std::vector<double> gain = params("mjc:gainPrm", 1.0);
    const std::vector<double> bias = params("mjc:biasPrm", 0.0);
    std::string target, gain_type = "fixed", bias_type = "none", dyn_type = "none";
    ReadTarget(prim.prim, "mjc:target", target);
    Read(prim.prim, "mjc:gainType", gain_type);
    Read(prim.prim, "mjc:biasType", bias_type);
    Read(prim.prim, "mjc:dynType", dyn_type);
    const auto at = prim_at_.find(target);
    const auto joint = at == prim_at_.end() ? joints_.end() : std::find_if(joints_.begin(), joints_.end(),
        [&](const Joint& j) { return j.prim == at->second && j.id != scene::kInvalidJoint; });

    scene::ActuatorType type = scene::ActuatorType::Motor;
    double value = gear[0] * gain[0];
    bool supported = joint != joints_.end() && dyn_type == "none" && gain_type == "fixed";
    if (bias_type == "affine" && bias[0] == 0.0 && bias[1] < 0.0 && gain[0] == -bias[1]) {
        type = scene::ActuatorType::Position;
        value = gear[0] * gear[0] * gain[0];
    } else if (bias_type == "affine" && bias[0] == 0.0 && bias[1] == 0.0 && bias[2] < 0.0 && gain[0] == -bias[2]) {
        type = scene::ActuatorType::Velocity;
        value = gear[0] * gear[0] * gain[0];
    } else if (bias_type != "none") {
        supported = false;
    }
    if (!supported) {
        Warn("USD_MJC_ACTUATOR_UNSUPPORTED", prim, "only motors and position or velocity servos on a joint, with "
             "dynType none and gainType fixed, are imported; the actuator is left out");
        return;
    }
    if (type == scene::ActuatorType::Position && bias[2] != 0.0) {
        Warn("USD_DRIVE_DAMPING_IGNORED", prim, "position drives damp at 2*sqrt(stiffness); the authored "
             "damping term " + std::to_string(bias[2]) + " is not represented");
    }

    // MuJoCo limits a range when its flag is "true", or "auto" with the range authored.
    const auto range = [&](const pxr::UsdPrim& on, const std::string& base, const char* flag) {
        std::string limited = "auto";
        Read(on, flag, limited);
        double lo = 0.0, hi = 0.0;
        const bool has_lo = Read(on, base + ":min", lo);
        const bool has_hi = Read(on, base + ":max", hi);
        const bool on_range = limited == "true" || (limited == "auto" && (has_lo || has_hi));
        return on_range ? std::max(std::abs(lo), std::abs(hi)) : kInf;
    };
    const double limit = std::min(std::abs(gear[0]) * range(prim.prim, "mjc:forceRange", "mjc:forceLimited"),
        range(prims_[joint->prim].prim, "mjc:actuatorfrcrange", "mjc:actuatorfrclimited"));
    const double length = joint->angular ? units_.meters * units_.meters : 1.0;
    const double torque = units_.kilograms * units_.meters * (joint->angular ? units_.meters : 1.0);
    scene::ActuatorRecord actuator;
    actuator.name = prim.name;
    actuator.joint_id = joint->id;
    actuator.type = type;
    actuator.gain = F(value * (type == scene::ActuatorType::Motor ? torque : units_.kilograms * length));
    if (std::isfinite(limit) && limit > 0.0) {
        actuator.force_limit = F(limit * torque);
    }
    scene.AddActuator(std::move(actuator));
}

// The first DomeLight is the scene's environment: its latlong texture the HDRI, its intensity the HDRI's scale.
void SceneBuilder::AddDome(const PrimData& prim, scene::SceneIR& scene) {
    if (dome_) {
        Warn("USD_DOME_LIGHT_EXTRA", prim, "only the first DomeLight becomes the environment; this one is left out");
        return;
    }
    dome_ = true;
    scene::EnvironmentRecord& env = scene.EnvironmentMut();
    env.intensity = LightIntensity(prim, options_.light_units);
    pxr::SdfAssetPath file;
    if (!ReadValue(prim.prim, "inputs:texture:file", file) || file.GetAssetPath().empty()) {
        return;
    }
    pxr::TfToken format;
    ReadValue(prim.prim, "inputs:texture:format", format);
    if (format.GetString() != "automatic" && format.GetString() != "latlong") {
        Warn("USD_DOME_LIGHT_FORMAT", prim, "a " + format.GetString() + " dome texture is not supported; "
             "it is left out");
        return;
    }
    env.hdri = TexturePath(file, prim.path);
    if (!env.hdri.empty()) {
        Warn("USD_DOME_LIGHT_ORIENTATION", prim, "the dome texture keeps the environment's own latlong orientation; "
             "the light's rotation is not imported");
    }
}

scene::SceneIR SceneBuilder::Build() {
    scene::SceneIR scene;
    for (size_t i = 0; i < prims_.size(); ++i) {
        prim_at_[prims_[i].path] = i;
    }

    // Every Material prim gets a record in prim order, bound or not.
    for (const PrimData& prim : prims_) {
        if (prim.type == "Material") {
            MaterialFor(prim.prim, scene);
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
        if (IsGprim(prims_[i].type)) {
            AddGprim(i, scene);
        }
    }
    ApplyCollisionGroups();

    AddBodies(scene);
    AddShapes(scene);
    AddFilteredPairs(scene);
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
        } else if (prim.type == "DomeLight" || prim.type == "DomeLight_1") {
            AddDome(prim, scene);
        } else if (prim.type.find("Light") != std::string::npos) {
            scene::LightRecord light;
            light.name = prim.name;
            light.type = LightTypeFromUsdType(prim.type);
            light.local_transform = ToTransform(Place(prim.world).pose);
            light.color = prim.color;
            light.intensity = LightIntensity(prim, options_.light_units);
            scene.AddLight(std::move(light));
        } else if (IsJointPrim(prim)) {
            for (const Joint& joint : joints_) {
                if (joint.prim == i && joint.id != scene::kInvalidJoint) {
                    AddDrive(joint, scene);
                }
            }
        } else if (prim.type == "MjcActuator") {
            AddMjcActuator(prim, scene);
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

UsdImportResult ImportUsd(const std::string& path, const UsdImportOptions& options) {
    const pxr::UsdStageRefPtr stage = OpenUsdStage(path);
    const StageUnits units = UnitsOf(stage);
    pxr::UsdGeomXformCache xforms(kTime);
    std::vector<PrimData> prims;
    for (const pxr::UsdPrim& prim : stage->Traverse(pxr::UsdTraverseInstanceProxies())) {
        prims.push_back(ReadPrim(prim, xforms, units));
    }
    UsdImportResult result;
    scene::SceneIR scene = SceneBuilder(stage, units, options, std::move(prims), result.diagnostics).Build();
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
