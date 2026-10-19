// ---------------------------------------------------------------------------
// nuka::import - USD exporter: writes the UsdPhysics stage ImportUsd reads back as the same scene
// ---------------------------------------------------------------------------

#include "import/usd_exporter.hpp"

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "scene/asset/asset_bytes.hpp"
#include "scene/terrain/terrain_field.hpp"

#include <pxr/base/gf/quatd.h>
#include <pxr/base/gf/quatf.h>
#include <pxr/base/gf/rotation.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/tf/stringUtils.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/vt/types.h>
#include <pxr/base/vt/value.h>
#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/layer.h>
#include <pxr/usd/sdf/path.h>
#include <pxr/usd/sdf/types.h>
#include <pxr/usd/usd/collectionAPI.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/references.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/boundable.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/capsule.h>
#include <pxr/usd/usdGeom/cube.h>
#include <pxr/usd/usdGeom/cylinder.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/plane.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/scope.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/usd/usdLux/lightAPI.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/sphereLight.h>
#include <pxr/usd/usdLux/tokens.h>
#include <pxr/usd/usdPhysics/articulationRootAPI.h>
#include <pxr/usd/usdPhysics/collisionAPI.h>
#include <pxr/usd/usdPhysics/collisionGroup.h>
#include <pxr/usd/usdPhysics/driveAPI.h>
#include <pxr/usd/usdPhysics/filteredPairsAPI.h>
#include <pxr/usd/usdPhysics/fixedJoint.h>
#include <pxr/usd/usdPhysics/massAPI.h>
#include <pxr/usd/usdPhysics/materialAPI.h>
#include <pxr/usd/usdPhysics/meshCollisionAPI.h>
#include <pxr/usd/usdPhysics/metrics.h>
#include <pxr/usd/usdPhysics/prismaticJoint.h>
#include <pxr/usd/usdPhysics/revoluteJoint.h>
#include <pxr/usd/usdPhysics/rigidBodyAPI.h>
#include <pxr/usd/usdPhysics/scene.h>
#include <pxr/usd/usdPhysics/sphericalJoint.h>
#include <pxr/usd/usdPhysics/tokens.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdUtils/usdzPackage.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nuka::import {

namespace {

using pxr::SdfPath;
using pxr::TfToken;
using pxr::VtValue;

constexpr double kPi = 3.14159265358979323846;
constexpr double kDegToRad = kPi / 180.0;
constexpr double kRadToDeg = 180.0 / kPi;
// ImportUsd reads an authored intensity in Isaac Lab units as a thousandth of Nuka's.
constexpr double kIsaacPerNuka = 1000.0;
// The UsdLux sun disc in degrees; under nits ImportUsd scales a sun by the angle it authors.
constexpr double kSunAngle = 0.53;
// USD's default vertical aperture (mm); the focal length carries the field of view.
constexpr double kVerticalAperture = 15.2908;
// The hull vertex count ImportUsd keeps when a collider authors no hullVertexLimit.
constexpr size_t kHullVertexLimit = 64;

constexpr const char* kChannels[3] = {"r", "g", "b"};

std::string Lowercase(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

// A prim name for `name`: characters USD does not allow become '_', and an empty name takes `fallback`.
std::string Identifier(const std::string& name, const std::string& fallback) {
    return pxr::TfMakeValidIdentifier(name.empty() ? fallback : name);
}

// The last segment of a '/'-separated record name; the groups before it are Xforms of their own.
std::string Leaf(const std::string& name) {
    const size_t slash = name.rfind('/');
    return slash == std::string::npos ? name : name.substr(slash + 1);
}

// The image type a file's leading bytes name, as the extension ImportUsd decodes by.
std::string ImageExtension(const std::vector<uint8_t>& bytes) {
    const auto same = [](char m, uint8_t b) { return static_cast<uint8_t>(m) == b; };
    const auto starts = [&](std::string_view magic) {
        return bytes.size() >= magic.size() && std::equal(magic.begin(), magic.end(), bytes.begin(), same);
    };
    if (starts("\x89PNG")) return ".png";
    if (starts("\xFF\xD8\xFF")) return ".jpg";
    if (starts("BM")) return ".bmp";
    if (starts("#?RADIANCE") || starts("#?RGBE")) return ".hdr";
    return ".tga";
}

// A rigid transform kept in double until a USD attribute stores it.
struct Pose {
    pxr::GfQuatd rotation = pxr::GfQuatd::GetIdentity();
    pxr::GfVec3d position = pxr::GfVec3d(0.0);

    Pose operator*(const Pose& other) const {
        return {rotation * other.rotation, rotation.Transform(other.position) + position};
    }
    Pose Inverse() const {
        const pxr::GfQuatd inverse = rotation.GetConjugate();
        return {inverse, -inverse.Transform(position)};
    }
};

Pose PoseOf(const math::Transform& t) {
    const math::Quat& q = t.rotation;
    return {pxr::GfQuatd(q.w, q.x, q.y, q.z).GetNormalized(),
            pxr::GfVec3d(t.position.x, t.position.y, t.position.z)};
}

bool IsIdentity(const math::Transform& t) {
    return t.position.x == 0.0f && t.position.y == 0.0f && t.position.z == 0.0f && t.rotation == math::Quat::Identity();
}

void Place(const pxr::UsdGeomXformable& xform, const Pose& pose) {
    xform.AddTranslateOp(pxr::UsdGeomXformOp::PrecisionDouble).Set(pose.position);
    xform.AddOrientOp(pxr::UsdGeomXformOp::PrecisionDouble).Set(pose.rotation);
}

void AuthorExtent(const pxr::UsdPrim& prim) {
    const pxr::UsdGeomBoundable boundable(prim);
    pxr::VtVec3fArray extent;
    if (pxr::UsdGeomBoundable::ComputeExtentFromPlugins(boundable, pxr::UsdTimeCode::Default(), &extent)) {
        boundable.CreateExtentAttr(VtValue(extent));
    }
}

bool HasMesh(const scene::CollisionShapeRecord& shape) {
    return (shape.type == scene::ShapeType::TriMesh || shape.type == scene::ShapeType::ConvexHull) &&
           !shape.mesh_vertices.empty() && !shape.mesh_indices.empty();
}

// A hash of a mesh's points, triangles, normals and uvs; equal meshes hash equal.
size_t MeshHash(const scene::CollisionShapeRecord& shape) {
    const auto bytes = [](const auto& values) {
        return std::hash<std::string_view>{}(std::string_view(reinterpret_cast<const char*>(values.data()),
                                                              values.size() * sizeof(values[0])));
    };
    size_t hash = bytes(shape.mesh_vertices);
    for (const size_t part : {bytes(shape.mesh_indices), bytes(shape.mesh_normals), bytes(shape.mesh_uvs)}) {
        hash ^= part + 0x9e3779b97f4a7c15ull + (hash << 6) + (hash >> 2);
    }
    return hash;
}

bool SameMesh(const scene::CollisionShapeRecord& a, const scene::CollisionShapeRecord& b) {
    return a.mesh_vertices == b.mesh_vertices && a.mesh_indices == b.mesh_indices &&
           a.mesh_normals == b.mesh_normals && a.mesh_uvs == b.mesh_uvs;
}

const char* SensorToken(scene::SensorType type) {
    switch (type) {
        case scene::SensorType::Imu: return "imu";
        case scene::SensorType::Contact: return "contact";
        case scene::SensorType::ForceTorque: return "force_torque";
        case scene::SensorType::FramePose: return "framepose";
        case scene::SensorType::JointState: return "joint_state";
        case scene::SensorType::Camera: return "camera";
        case scene::SensorType::Depth: return "depth";
        case scene::SensorType::Lidar: return "lidar";
        case scene::SensorType::RangeScan: return "range_scan";
        case scene::SensorType::LinearVelocity: return "linear_velocity";
        case scene::SensorType::Tactile: return "";
    }
    return "";
}

const char* ActuatorToken(scene::ActuatorType type) {
    switch (type) {
        case scene::ActuatorType::Position: return "position";
        case scene::ActuatorType::Velocity: return "velocity";
        case scene::ActuatorType::Force: return "force";
        case scene::ActuatorType::Motor: return "motor";
    }
    return "motor";
}

// How a body is written.
enum class BodyKind {
    Skipped,    // outside the exported node
    Anchor,     // a static, shapeless joint parent: its joints hold the world instead
    Dynamic,    // a rigid body
    Rooted,     // a static joint parent: a rigid body a fixed joint holds to the world
    Kinematic,  // a static body hanging from a joint: a kinematic rigid body
    Static,     // static colliders under an Xform ImportUsd reads back as one static body
};

class StageWriter {
public:
    StageWriter(const scene::SceneIR& scene, const UsdExportOptions& options, pxr::UsdStageRefPtr stage,
                std::filesystem::path textures, std::vector<UsdDiagnostic>& out)
        : scene_(scene), options_(options), whole_(options.node.empty()), stage_(std::move(stage)),
          textures_(std::move(textures)), out_(out) {}

    void Write();

private:
    void Warn(const char* code, const std::string& prim, std::string message) {
        out_.push_back(UsdDiagnostic{UsdDiagnosticSeverity::Warning, code, prim, std::move(message)});
    }
    SdfPath Child(const SdfPath& parent, const std::string& name, const std::string& fallback) const;
    SdfPath GroupOf(const std::string& name);
    void Classify();
    Pose WorldOf(scene::BodyId body);
    std::pair<SdfPath, scene::BodyId> ParentOf(scene::BodyId body);
    SdfPath BodyPrim(scene::BodyId body);
    bool Written(scene::BodyId body) const;
    SdfPath MaterialPath(scene::MaterialId id);
    SdfPath WriteMaterial(const scene::MaterialRecord& record, const std::string& name);
    pxr::UsdShadeOutput Texture(const SdfPath& material, const std::string& map, const std::string& slot,
                                const char* channel, const pxr::GfVec4f& scale, bool color, float uv_scale);
    pxr::UsdShadeOutput StOf(const SdfPath& material, float uv_scale);
    std::string TextureFile(const std::string& source, const std::string& hint, const std::string& prim);
    SdfPath ColliderMaterial(const scene::CollisionShapeRecord& shape);
    void Bind(const pxr::UsdPrim& prim, const SdfPath& material);
    bool Writes(const scene::CollisionShapeRecord& shape) const;
    void WriteShape(const scene::CollisionShapeRecord& shape);
    pxr::UsdPrim WriteMesh(const SdfPath& path, const scene::CollisionShapeRecord& shape);
    void WriteMeshData(pxr::UsdGeomMesh& mesh, const scene::CollisionShapeRecord& shape);
    void WriteCollider(const scene::CollisionShapeRecord& shape, pxr::UsdPrim& prim);
    void WriteJoint(const scene::JointRecord& joint);
    void WriteRoots();
    void WriteActuator(const scene::ActuatorRecord& actuator);
    void WriteSensor(const scene::SensorDesc& sensor);
    void WriteCamera(const scene::CameraRecord& camera);
    void WriteLight(const scene::LightRecord& light);
    void WriteEnvironment();
    void WriteTerrain(const scene::TerrainRecord& terrain);
    void WriteFilteredPairs();
    void WriteCollisionGroups();
    void ReportDropped();
    float Intensity(float nuka, bool sun) const;

    const scene::SceneIR& scene_;
    const UsdExportOptions& options_;
    const bool whole_;
    pxr::UsdStageRefPtr stage_;
    std::filesystem::path textures_;
    std::vector<UsdDiagnostic>& out_;
    const SdfPath root_{"/World"};
    const SdfPath prototypes_{"/World/Prototypes"};

    std::vector<char> in_scope_;
    std::vector<int> incoming_;          // per body: the joint it hangs from, -1 for none
    std::vector<char> jointed_;          // per body: some joint names it
    std::vector<char> has_visual_;
    std::vector<char> has_collider_;
    std::vector<BodyKind> kind_;
    std::vector<Pose> world_;
    std::vector<char> world_known_;
    std::vector<SdfPath> body_path_;
    std::vector<SdfPath> joint_path_;
    std::vector<SdfPath> material_path_;
    std::map<std::pair<scene::MaterialId, float>, SdfPath> friction_materials_;
    std::unordered_map<std::string, std::string> texture_files_;  // source -> asset path
    std::set<std::string> texture_names_;
    std::set<scene::JointId> driven_;
    // Colliders by (contype, conaffinity), the classes collision groups stand for.
    std::map<std::pair<uint32_t, uint32_t>, std::vector<SdfPath>> classes_;
    // Written shapes per mesh hash, and the prototype prims of the meshes more than one shape carries.
    std::unordered_map<size_t, int> mesh_uses_;
    std::unordered_map<size_t, std::vector<std::pair<const scene::CollisionShapeRecord*, SdfPath>>> prototypes_of_;
};

// `parent`'s child named after the last segment of `name`, made unique against the prims already there.
SdfPath StageWriter::Child(const SdfPath& parent, const std::string& name, const std::string& fallback) const {
    const std::string base = Identifier(Leaf(name), fallback);
    SdfPath path = parent.AppendChild(TfToken(base));
    for (int k = 1; stage_->GetPrimAtPath(path); ++k) {
        path = parent.AppendChild(TfToken(base + "_" + std::to_string(k)));
    }
    return path;
}

// The Xform groups the '/'-separated prefix of a record name names, made on first use and marked nuka:group so
// ImportUsd gives the records under them the same prefix.
SdfPath StageWriter::GroupOf(const std::string& name) {
    SdfPath path = root_;
    size_t start = 0;
    for (size_t slash = name.find('/'); slash != std::string::npos; slash = name.find('/', start)) {
        path = path.AppendChild(TfToken(Identifier(name.substr(start, slash - start), "group")));
        if (!stage_->GetPrimAtPath(path)) {
            pxr::UsdGeomXform::Define(stage_, path)
                .GetPrim()
                .CreateAttribute(TfToken("nuka:group"), pxr::SdfValueTypeNames->Bool)
                .Set(true);
        }
        start = slash + 1;
    }
    return path;
}

void StageWriter::Classify() {
    const size_t count = scene_.RigidBodyCount();
    in_scope_.assign(count, whole_ ? 1 : 0);
    incoming_.assign(count, -1);
    jointed_.assign(count, 0);
    has_visual_.assign(count, 0);
    has_collider_.assign(count, 0);
    kind_.assign(count, BodyKind::Skipped);
    world_.assign(count, Pose{});
    world_known_.assign(count, 0);
    body_path_.assign(count, SdfPath());
    joint_path_.assign(scene_.JointCount(), SdfPath());
    material_path_.assign(scene_.MaterialCount(), SdfPath());

    std::vector<char> parent_of_joint(count, 0), has_shape(count, 0), carries(count, 0);
    for (const scene::JointRecord& joint : scene_.Joints()) {
        if (joint.type == scene::JointType::Free || joint.child_body >= count) {
            continue;
        }
        jointed_[joint.child_body] = 1;
        if (joint.parent_body < count) {
            incoming_[joint.child_body] = static_cast<int>(joint.id);
            parent_of_joint[joint.parent_body] = 1;
            jointed_[joint.parent_body] = 1;
        }
    }
    for (const scene::CollisionShapeRecord& shape : scene_.Shapes()) {
        if (shape.body_id < count) {
            has_shape[shape.body_id] = 1;
            if (shape.contype == 0 && shape.conaffinity == 0) {
                has_visual_[shape.body_id] = 1;
            } else {
                has_collider_[shape.body_id] = 1;
            }
        }
    }
    for (const scene::CameraRecord& camera : scene_.Cameras()) {
        if (camera.attached_body < count) carries[camera.attached_body] = 1;
    }
    for (const scene::LightRecord& light : scene_.Lights()) {
        if (light.attached_body < count) carries[light.attached_body] = 1;
    }
    for (const scene::SensorDesc& sensor : scene_.Sensors()) {
        if (sensor.mount == scene::MountFrame::Body && sensor.mount_index < count) carries[sensor.mount_index] = 1;
    }

    if (!whole_) {
        const std::shared_ptr<scene::SceneNode> node = scene_.Tree().NodeOf(options_.node);
        if (!node || node == scene_.Tree().Root()) {
            throw std::runtime_error("USD export: the scene has no node at '" + options_.node + "'");
        }
        std::unordered_set<scene::EntityId, scene::EntityIdHash> under;
        scene_.Tree().Traverse(node, [&](const std::shared_ptr<scene::SceneNode>& n) { under.insert(n->entity); });
        bool any = false;
        for (scene::BodyId b = 0; b < count; ++b) {
            if (under.count(scene_.EntityOfBody(b)) != 0u) {
                in_scope_[b] = 1;
                any = true;
            }
        }
        if (!any) {
            throw std::runtime_error("USD export: '" + options_.node + "' holds no body");
        }
        // Everything hanging from the node by hierarchy or by joint goes with it.
        for (bool grew = true; grew;) {
            grew = false;
            for (scene::BodyId b = 0; b < count; ++b) {
                const scene::BodyId parent = scene_.GetBody(b).parent_id;
                const bool held = (parent < count && in_scope_[parent]) ||
                                  (incoming_[b] >= 0 && in_scope_[scene_.GetJoint(incoming_[b]).parent_body]);
                if (!in_scope_[b] && held) {
                    in_scope_[b] = 1;
                    grew = true;
                }
            }
        }
    }

    for (scene::BodyId b = 0; b < count; ++b) {
        const scene::RigidBodyRecord& body = scene_.GetBody(b);
        if (!in_scope_[b]) {
            kind_[b] = BodyKind::Skipped;
        } else if (!body.is_static) {
            kind_[b] = BodyKind::Dynamic;
        } else if (!has_shape[b] && !carries[b] && incoming_[b] < 0 && parent_of_joint[b] &&
                   body.parent_id == scene::kInvalidBody) {
            kind_[b] = BodyKind::Anchor;
        } else if (incoming_[b] >= 0) {
            kind_[b] = BodyKind::Kinematic;
        } else if (parent_of_joint[b]) {
            kind_[b] = BodyKind::Rooted;
        } else {
            kind_[b] = BodyKind::Static;
        }
    }
}

// The body's world pose with every joint at zero.
Pose StageWriter::WorldOf(scene::BodyId body) {
    if (!world_known_[body]) {
        const scene::RigidBodyRecord& record = scene_.GetBody(body);
        Pose pose = PoseOf(record.local_transform);
        if (incoming_[body] >= 0) {
            const scene::JointRecord& joint = scene_.GetJoint(static_cast<scene::JointId>(incoming_[body]));
            pose = WorldOf(joint.parent_body) * PoseOf(joint.parent_frame) * pose;
        } else if (record.parent_id < scene_.RigidBodyCount()) {
            pose = WorldOf(record.parent_id) * pose;
        }
        world_[body] = pose;
        world_known_[body] = 1;
    }
    return world_[body];
}

bool StageWriter::Written(scene::BodyId body) const {
    return body < kind_.size() && kind_[body] != BodyKind::Skipped && kind_[body] != BodyKind::Anchor;
}

// The prim a body nests under: its parent body's, an anchor's place, or the group its name names.
std::pair<SdfPath, scene::BodyId> StageWriter::ParentOf(scene::BodyId body) {
    const scene::BodyId parent = scene_.GetBody(body).parent_id;
    if (parent < scene_.RigidBodyCount() && in_scope_[parent]) {
        if (kind_[parent] == BodyKind::Anchor) {
            return ParentOf(parent);
        }
        return {BodyPrim(parent), parent};
    }
    return {GroupOf(scene_.GetBody(body).name), scene::kInvalidBody};
}

SdfPath StageWriter::BodyPrim(scene::BodyId body) {
    if (!Written(body) || !body_path_[body].IsEmpty()) {
        return body < body_path_.size() ? body_path_[body] : SdfPath();
    }
    const scene::RigidBodyRecord& record = scene_.GetBody(body);
    const auto [parent, frame] = ParentOf(body);
    const SdfPath path = Child(parent, record.name, "body_" + std::to_string(body));
    const pxr::UsdGeomXform xform = pxr::UsdGeomXform::Define(stage_, path);
    Place(xform, frame == scene::kInvalidBody ? WorldOf(body) : WorldOf(frame).Inverse() * WorldOf(body));
    const pxr::UsdPrim prim = xform.GetPrim();
    if (kind_[body] == BodyKind::Static) {
        prim.CreateAttribute(TfToken("nuka:staticBody"), pxr::SdfValueTypeNames->Bool).Set(true);
    } else {
        const pxr::UsdPhysicsRigidBodyAPI rigid = pxr::UsdPhysicsRigidBodyAPI::Apply(prim);
        if (kind_[body] == BodyKind::Kinematic) {
            rigid.CreateKinematicEnabledAttr(VtValue(true));
        }
        if (record.mass == 0.0f && !has_collider_[body]) {
            // UsdPhysics has no massless body (a URDF sensor link); Nuka's own attribute keeps it massless.
            prim.CreateAttribute(TfToken("nuka:massless"), pxr::SdfValueTypeNames->Bool).Set(true);
            Warn("USD_EXPORT_MASSLESS", path.GetString(), "the body has no mass and no colliders; ImportUsd reads it "
                 "back massless, other readers give it their default mass");
        } else {
            const pxr::UsdPhysicsMassAPI mass = pxr::UsdPhysicsMassAPI::Apply(prim);
            const math::Transform& inertial = record.inertial_transform;
            const math::Quat& axes = inertial.rotation;
            mass.CreateMassAttr(VtValue(record.mass));
            mass.CreateCenterOfMassAttr(VtValue(pxr::GfVec3f(inertial.position.x, inertial.position.y,
                                                             inertial.position.z)));
            mass.CreateDiagonalInertiaAttr(VtValue(pxr::GfVec3f(record.inertia.x, record.inertia.y,
                                                                record.inertia.z)));
            mass.CreatePrincipalAxesAttr(VtValue(pxr::GfQuatf(axes.w, axes.x, axes.y, axes.z)));
        }
    }
    body_path_[body] = path;
    return path;
}

// ---------------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------------

SdfPath StageWriter::MaterialPath(scene::MaterialId id) {
    if (id >= scene_.MaterialCount()) {
        return {};
    }
    if (material_path_[id].IsEmpty()) {
        const scene::MaterialRecord& record = scene_.GetMaterial(id);
        material_path_[id] = WriteMaterial(record, Identifier(Leaf(record.name), "material_" + std::to_string(id)));
    }
    return material_path_[id];
}

// A UsdPreviewSurface material with the record's maps and factors, and its friction as a physics material, in the
// Looks scope of the group its record name names.
SdfPath StageWriter::WriteMaterial(const scene::MaterialRecord& m, const std::string& name) {
    const SdfPath looks = GroupOf(m.name).AppendChild(TfToken("Looks"));
    if (!stage_->GetPrimAtPath(looks)) {
        pxr::UsdGeomScope::Define(stage_, looks);
    }
    const SdfPath path = Child(looks, name, "material");
    const pxr::UsdShadeMaterial material = pxr::UsdShadeMaterial::Define(stage_, path);
    pxr::UsdShadeShader surface =
        pxr::UsdShadeShader::Define(stage_, path.AppendChild(TfToken("PreviewSurface")));
    surface.CreateIdAttr(VtValue(TfToken("UsdPreviewSurface")));
    material.CreateSurfaceOutput().ConnectToSource(
        surface.CreateOutput(TfToken("surface"), pxr::SdfValueTypeNames->Token));
    const auto input = [&](const char* id, const pxr::SdfValueTypeName& type) {
        return surface.CreateInput(TfToken(id), type);
    };
    const float uv = m.uv_scale;

    const pxr::GfVec3f base(m.base_color.x, m.base_color.y, m.base_color.z);
    const pxr::UsdShadeInput diffuse = input("diffuseColor", pxr::SdfValueTypeNames->Color3f);
    if (const pxr::UsdShadeOutput map = Texture(path, m.albedo_map, "albedo", "rgb",
                                                pxr::GfVec4f(base[0], base[1], base[2], 1.0f), true, uv)) {
        diffuse.ConnectToSource(map);
    } else {
        diffuse.Set(base);
    }
    const auto scalar = [&](const char* id, float value, const std::string& file, uint32_t channel,
                            const char* slot) {
        const pxr::UsdShadeInput in = input(id, pxr::SdfValueTypeNames->Float);
        if (const pxr::UsdShadeOutput map = Texture(path, file, slot, kChannels[std::min(channel, 2u)],
                                                    pxr::GfVec4f(value, value, value, 1.0f), false, uv)) {
            in.ConnectToSource(map);
        } else {
            in.Set(value);
        }
    };
    scalar("roughness", m.roughness, m.roughness_map, m.roughness_channel, "roughness");
    scalar("metallic", m.metallic, m.metallic_map, m.metallic_channel, "metallic");
    const pxr::GfVec3f glow(m.emissive.x, m.emissive.y, m.emissive.z);
    if (const pxr::UsdShadeOutput map = Texture(path, m.emissive_map, "emissive", "rgb",
                                                pxr::GfVec4f(glow[0], glow[1], glow[2], 1.0f), true, uv)) {
        input("emissiveColor", pxr::SdfValueTypeNames->Color3f).ConnectToSource(map);
    } else if (glow != pxr::GfVec3f(0.0f)) {
        input("emissiveColor", pxr::SdfValueTypeNames->Color3f).Set(glow);
    }
    if (const pxr::UsdShadeOutput map = Texture(path, m.normal_map, "normal", "rgb",
                                                pxr::GfVec4f(2.0f, 2.0f, 2.0f, 1.0f), false, uv)) {
        pxr::UsdShadeShader(map.GetPrim())
            .CreateInput(TfToken("bias"), pxr::SdfValueTypeNames->Float4)
            .Set(pxr::GfVec4f(-1.0f, -1.0f, -1.0f, 0.0f));
        input("normal", pxr::SdfValueTypeNames->Normal3f).ConnectToSource(map);
    }
    if (m.alpha != 1.0f) {
        input("opacity", pxr::SdfValueTypeNames->Float).Set(m.alpha);
    }
    input("ior", pxr::SdfValueTypeNames->Float).Set(m.ior);

    const pxr::UsdPhysicsMaterialAPI physics = pxr::UsdPhysicsMaterialAPI::Apply(material.GetPrim());
    physics.CreateDynamicFrictionAttr(VtValue(m.friction_mu));
    physics.CreateStaticFrictionAttr(VtValue(m.friction_mu));
    const bool textured = !m.albedo_map.empty() || !m.roughness_map.empty() || !m.metallic_map.empty() ||
                          !m.normal_map.empty() || !m.emissive_map.empty();
    if (textured && m.triplanar) {
        material.GetPrim().CreateAttribute(TfToken("nuka:triplanar"), pxr::SdfValueTypeNames->Bool).Set(true);
    }
    if (m.sheen != 0.0f || m.transmission != 0.0f || m.absorption.x != 0.0f || m.absorption.y != 0.0f ||
        m.absorption.z != 0.0f || m.weave.pattern != 0u) {
        Warn("USD_EXPORT_MATERIAL_EXTRA", path.GetString(), "sheen, transmission, absorption and the cloth weave "
             "have no UsdPreviewSurface input; they are left out");
    }
    return path;
}

// A UsdUVTexture channel of `map` through the material's st, scaled by `scale`; invalid when there is no image.
pxr::UsdShadeOutput StageWriter::Texture(const SdfPath& material, const std::string& map, const std::string& slot,
                                         const char* channel, const pxr::GfVec4f& scale, bool color,
                                         float uv_scale) {
    if (map.empty()) {
        return {};
    }
    const std::string file = TextureFile(map, material.GetName() + "_" + slot, material.GetString());
    if (file.empty()) {
        return {};
    }
    pxr::UsdShadeShader texture =
        pxr::UsdShadeShader::Define(stage_, material.AppendChild(TfToken(slot + "_texture")));
    texture.CreateIdAttr(VtValue(TfToken("UsdUVTexture")));
    texture.CreateInput(TfToken("file"), pxr::SdfValueTypeNames->Asset).Set(pxr::SdfAssetPath(file));
    texture.CreateInput(TfToken("sourceColorSpace"), pxr::SdfValueTypeNames->Token)
        .Set(TfToken(color ? "sRGB" : "raw"));
    texture.CreateInput(TfToken("wrapS"), pxr::SdfValueTypeNames->Token).Set(TfToken("repeat"));
    texture.CreateInput(TfToken("wrapT"), pxr::SdfValueTypeNames->Token).Set(TfToken("repeat"));
    texture.CreateInput(TfToken("scale"), pxr::SdfValueTypeNames->Float4).Set(scale);
    texture.CreateInput(TfToken("st"), pxr::SdfValueTypeNames->Float2).ConnectToSource(StOf(material, uv_scale));
    const bool rgb = std::string_view(channel) == "rgb";
    return texture.CreateOutput(TfToken(channel), rgb ? pxr::SdfValueTypeNames->Float3 : pxr::SdfValueTypeNames->Float);
}

// The material's st reader, behind a UsdTransform2d when the maps tile.
pxr::UsdShadeOutput StageWriter::StOf(const SdfPath& material, float uv_scale) {
    const SdfPath reader_path = material.AppendChild(TfToken("st_reader"));
    const SdfPath transform_path = material.AppendChild(TfToken("st_transform"));
    if (!stage_->GetPrimAtPath(reader_path)) {
        pxr::UsdShadeShader reader = pxr::UsdShadeShader::Define(stage_, reader_path);
        reader.CreateIdAttr(VtValue(TfToken("UsdPrimvarReader_float2")));
        reader.CreateInput(TfToken("varname"), pxr::SdfValueTypeNames->String).Set(std::string("st"));
        const pxr::UsdShadeOutput result = reader.CreateOutput(TfToken("result"), pxr::SdfValueTypeNames->Float2);
        if (uv_scale != 1.0f) {
            pxr::UsdShadeShader transform = pxr::UsdShadeShader::Define(stage_, transform_path);
            transform.CreateIdAttr(VtValue(TfToken("UsdTransform2d")));
            transform.CreateInput(TfToken("in"), pxr::SdfValueTypeNames->Float2).ConnectToSource(result);
            transform.CreateInput(TfToken("scale"), pxr::SdfValueTypeNames->Float2)
                .Set(pxr::GfVec2f(uv_scale, uv_scale));
            transform.CreateOutput(TfToken("result"), pxr::SdfValueTypeNames->Float2);
        }
    }
    const SdfPath source = stage_->GetPrimAtPath(transform_path) ? transform_path : reader_path;
    return pxr::UsdShadeShader(stage_->GetPrimAtPath(source)).GetOutput(TfToken("result"));
}

// The image `source` names, copied once into the textures folder; "" when it cannot be read.
std::string StageWriter::TextureFile(const std::string& source, const std::string& hint, const std::string& prim) {
    const auto found = texture_files_.find(source);
    if (found != texture_files_.end()) {
        return found->second;
    }
    const std::vector<uint8_t> bytes = scene::ReadAssetBytes(source);
    std::string file;
    if (bytes.empty()) {
        Warn("USD_EXPORT_TEXTURE_MISSING", prim, "image '" + source + "' cannot be read; it is left out");
    } else {
        // A plain file keeps its name; a chunk of a .nka, .glb or .usdz is named after its slot.
        const std::filesystem::path named = source.find('#') == std::string::npos
                                          ? std::filesystem::path(source).filename()
                                          : std::filesystem::path(Identifier(hint, "image") + ImageExtension(bytes));
        const std::string stem = named.stem().string(), extension = named.extension().string();
        std::string chosen = stem + extension;
        for (int k = 1; !texture_names_.insert(Lowercase(chosen)).second; ++k) {
            chosen = stem + "_" + std::to_string(k) + extension;
        }
        std::filesystem::create_directories(textures_);
        std::ofstream stream(textures_ / chosen, std::ios::binary);
        stream.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        if (!stream) {
            throw std::runtime_error("USD export: cannot write " + (textures_ / chosen).string());
        }
        file = "./" + textures_.filename().string() + "/" + chosen;
    }
    texture_files_[source] = file;
    return file;
}

// A collider's material; a friction of its own that differs from the material's takes a copy with that friction.
SdfPath StageWriter::ColliderMaterial(const scene::CollisionShapeRecord& shape) {
    const bool has = shape.material_id < scene_.MaterialCount();
    if (shape.friction_mu < 0.0f || (has && scene_.GetMaterial(shape.material_id).friction_mu == shape.friction_mu)) {
        return MaterialPath(shape.material_id);
    }
    const auto key = std::make_pair(has ? shape.material_id : scene::kInvalidMaterial, shape.friction_mu);
    auto found = friction_materials_.find(key);
    if (found == friction_materials_.end()) {
        scene::MaterialRecord record = has ? scene_.GetMaterial(shape.material_id) : scene::MaterialRecord{};
        record.friction_mu = shape.friction_mu;
        char mu[32];
        std::snprintf(mu, sizeof(mu), "_mu%g", static_cast<double>(shape.friction_mu));
        const std::string base = has ? Identifier(Leaf(record.name), "material_" + std::to_string(shape.material_id))
                                     : std::string("friction");
        found = friction_materials_.emplace(key, WriteMaterial(record, Identifier(base + mu, "friction"))).first;
    }
    return found->second;
}

void StageWriter::Bind(const pxr::UsdPrim& prim, const SdfPath& material) {
    if (!material.IsEmpty()) {
        pxr::UsdShadeMaterialBindingAPI::Apply(prim).Bind(pxr::UsdShadeMaterial(stage_->GetPrimAtPath(material)));
    }
}

// ---------------------------------------------------------------------------
// Shapes
// ---------------------------------------------------------------------------

bool StageWriter::Writes(const scene::CollisionShapeRecord& shape) const {
    return shape.body_id >= scene_.RigidBodyCount() ? whole_ : in_scope_[shape.body_id] != 0;
}

void StageWriter::WriteShape(const scene::CollisionShapeRecord& shape) {
    if (!Writes(shape)) {
        return;
    }
    const bool bodiless = shape.body_id >= scene_.RigidBodyCount();
    const SdfPath parent = bodiless ? GroupOf(shape.name) : BodyPrim(shape.body_id);
    const bool collider = shape.contype != 0u || shape.conaffinity != 0u;
    const SdfPath path = Child(parent, shape.name, "geom_" + std::to_string(shape.id));
    pxr::UsdPrim prim;
    switch (shape.type) {
        case scene::ShapeType::Box:
            prim = pxr::UsdGeomCube::Define(stage_, path).GetPrim();
            pxr::UsdGeomCube(prim).CreateSizeAttr(VtValue(2.0));
            break;
        case scene::ShapeType::Sphere:
            prim = pxr::UsdGeomSphere::Define(stage_, path).GetPrim();
            pxr::UsdGeomSphere(prim).CreateRadiusAttr(VtValue(static_cast<double>(shape.radius)));
            break;
        case scene::ShapeType::Capsule: {
            const pxr::UsdGeomCapsule capsule = pxr::UsdGeomCapsule::Define(stage_, path);
            capsule.CreateRadiusAttr(VtValue(static_cast<double>(shape.radius)));
            capsule.CreateHeightAttr(VtValue(2.0 * static_cast<double>(shape.half_height)));
            capsule.CreateAxisAttr(VtValue(pxr::UsdGeomTokens->z));
            prim = capsule.GetPrim();
            break;
        }
        case scene::ShapeType::Cylinder: {
            const pxr::UsdGeomCylinder cylinder = pxr::UsdGeomCylinder::Define(stage_, path);
            cylinder.CreateRadiusAttr(VtValue(static_cast<double>(shape.radius)));
            cylinder.CreateHeightAttr(VtValue(2.0 * static_cast<double>(shape.half_height)));
            cylinder.CreateAxisAttr(VtValue(pxr::UsdGeomTokens->z));
            prim = cylinder.GetPrim();
            break;
        }
        case scene::ShapeType::Plane: {
            const pxr::UsdGeomPlane plane = pxr::UsdGeomPlane::Define(stage_, path);
            plane.CreateWidthAttr(VtValue(2.0 * static_cast<double>(shape.half_extents.x)));
            plane.CreateLengthAttr(VtValue(2.0 * static_cast<double>(shape.half_extents.y)));
            plane.CreateAxisAttr(VtValue(pxr::UsdGeomTokens->z));
            prim = plane.GetPrim();
            break;
        }
        case scene::ShapeType::TriMesh:
        case scene::ShapeType::ConvexHull:
            if (!HasMesh(shape)) {
                Warn("USD_EXPORT_SHAPE_EMPTY", path.GetString(), "the mesh has no triangles; it is left out");
                return;
            }
            prim = WriteMesh(path, shape);
            break;
        case scene::ShapeType::HeightField:
            Warn("USD_EXPORT_SHAPE_UNSUPPORTED", path.GetString(), "a heightfield shape has no USD form; it is "
                 "left out");
            return;
    }
    const pxr::UsdGeomXformable xform(prim);
    Place(xform, PoseOf(shape.local_transform));
    if (shape.type == scene::ShapeType::Box) {
        xform.AddScaleOp(pxr::UsdGeomXformOp::PrecisionFloat)
            .Set(pxr::GfVec3f(shape.half_extents.x, shape.half_extents.y, shape.half_extents.z));
    }
    AuthorExtent(prim);
    if (collider) {
        WriteCollider(shape, prim);
    } else {
        Bind(prim, MaterialPath(shape.material_id));
    }
}

// A triangle mesh; one that several shapes carry is written once under the abstract Prototypes scope and
// referenced from each.
pxr::UsdPrim StageWriter::WriteMesh(const SdfPath& path, const scene::CollisionShapeRecord& shape) {
    pxr::UsdGeomMesh mesh = pxr::UsdGeomMesh::Define(stage_, path);
    const size_t hash = MeshHash(shape);
    if (mesh_uses_[hash] < 2) {
        WriteMeshData(mesh, shape);
        return mesh.GetPrim();
    }
    auto& known = prototypes_of_[hash];
    auto found = std::find_if(known.begin(), known.end(), [&](const auto& k) { return SameMesh(*k.first, shape); });
    if (found == known.end()) {
        if (!stage_->GetPrimAtPath(prototypes_)) {
            stage_->CreateClassPrim(prototypes_);
        }
        pxr::UsdGeomMesh prototype = pxr::UsdGeomMesh::Define(stage_, Child(prototypes_, path.GetName(), "mesh"));
        WriteMeshData(prototype, shape);
        found = known.insert(known.end(), {&shape, prototype.GetPath()});
    }
    mesh.GetPrim().GetReferences().AddInternalReference(found->second);
    return mesh.GetPrim();
}

// The mesh's points and triangles, with its normals and st per vertex.
void StageWriter::WriteMeshData(pxr::UsdGeomMesh& mesh, const scene::CollisionShapeRecord& shape) {
    const size_t count = shape.mesh_vertices.size() / 3u;
    pxr::VtVec3fArray points(count);
    for (size_t v = 0; v < count; ++v) {
        points[v] = pxr::GfVec3f(&shape.mesh_vertices[3u * v]);
    }
    pxr::VtIntArray indices(shape.mesh_indices.size());
    for (size_t i = 0; i < indices.size(); ++i) {
        indices[i] = static_cast<int>(shape.mesh_indices[i]);
    }
    mesh.CreatePointsAttr(VtValue(points));
    mesh.CreateFaceVertexCountsAttr(VtValue(pxr::VtIntArray(indices.size() / 3u, 3)));
    mesh.CreateFaceVertexIndicesAttr(VtValue(indices));
    mesh.CreateSubdivisionSchemeAttr(VtValue(pxr::UsdGeomTokens->none));
    if (shape.mesh_normals.size() == 3u * count) {
        pxr::VtVec3fArray normals(count);
        for (size_t v = 0; v < count; ++v) {
            normals[v] = pxr::GfVec3f(&shape.mesh_normals[3u * v]);
        }
        mesh.CreateNormalsAttr(VtValue(normals));
        mesh.SetNormalsInterpolation(pxr::UsdGeomTokens->vertex);
    }
    if (shape.mesh_uvs.size() == 2u * count) {
        pxr::VtVec2fArray st(count);
        for (size_t v = 0; v < count; ++v) {
            st[v] = pxr::GfVec2f(&shape.mesh_uvs[2u * v]);
        }
        pxr::UsdGeomPrimvarsAPI(mesh)
            .CreatePrimvar(TfToken("st"), pxr::SdfValueTypeNames->TexCoord2fArray, pxr::UsdGeomTokens->vertex)
            .Set(st);
    }
}

// CollisionAPI with the mesh approximation, the MuJoCo contact parameters that differ from the defaults, and
// the material. A body with visuals hides its colliders, as Nuka draws only bare colliders; meshes always.
void StageWriter::WriteCollider(const scene::CollisionShapeRecord& shape, pxr::UsdPrim& prim) {
    pxr::UsdPhysicsCollisionAPI::Apply(prim);
    const bool mesh = shape.type == scene::ShapeType::TriMesh || shape.type == scene::ShapeType::ConvexHull;
    if (mesh) {
        TfToken approximation = pxr::UsdPhysicsTokens->none;
        const size_t vertices = shape.mesh_vertices.size() / 3u;
        if (shape.type == scene::ShapeType::ConvexHull) {
            approximation = pxr::UsdPhysicsTokens->convexHull;
            if (vertices > kHullVertexLimit) {
                prim.CreateAttribute(TfToken("physxConvexHullCollision:hullVertexLimit"), pxr::SdfValueTypeNames->Int)
                    .Set(static_cast<int>(vertices));
            }
        } else if (shape.decompose_mode == scene::DecomposeMode::Force) {
            approximation = pxr::UsdPhysicsTokens->convexDecomposition;
            prim.CreateAttribute(TfToken("physxConvexDecompositionCollision:maxConvexHulls"),
                                 pxr::SdfValueTypeNames->Int)
                .Set(static_cast<int>(shape.decompose_max_pieces));
        } else if (shape.mesh_contact == scene::CollisionShapeRecord::MeshContact::Sdf) {
            approximation = TfToken("sdf");
        }
        pxr::UsdPhysicsMeshCollisionAPI::Apply(prim).CreateApproximationAttr(VtValue(approximation));
        if (shape.decompose_mode != scene::DecomposeMode::Auto) {
            const bool force = shape.decompose_mode == scene::DecomposeMode::Force;
            prim.CreateAttribute(TfToken("nuka:decompose"), pxr::SdfValueTypeNames->Token)
                .Set(TfToken(force ? "force" : "skip"));
            prim.CreateAttribute(TfToken("nuka:decompose:max_pieces"), pxr::SdfValueTypeNames->Int)
                .Set(static_cast<int>(shape.decompose_max_pieces));
        }
    }
    if (mesh || (shape.body_id < has_visual_.size() && has_visual_[shape.body_id])) {
        pxr::UsdGeomImageable(prim).CreatePurposeAttr(VtValue(pxr::UsdGeomTokens->guide));
    }

    const scene::CollisionShapeRecord defaults;
    bool applied = false;
    const auto mjc = [&](const char* name, const pxr::SdfValueTypeName& type, const VtValue& value) {
        if (!applied) {
            prim.AddAppliedSchema(TfToken("MjcCollisionAPI"));
            applied = true;
        }
        prim.CreateAttribute(TfToken(name), type).Set(value);
    };
    if (shape.condim != defaults.condim) {
        mjc("mjc:condim", pxr::SdfValueTypeNames->Int, VtValue(static_cast<int>(shape.condim)));
    }
    if (shape.collision_group != defaults.collision_group) {
        mjc("mjc:group", pxr::SdfValueTypeNames->Int, VtValue(static_cast<int>(shape.collision_group)));
    }
    if (shape.priority != defaults.priority) {
        mjc("mjc:priority", pxr::SdfValueTypeNames->Int, VtValue(static_cast<int>(shape.priority)));
    }
    if (shape.solmix != defaults.solmix) {
        mjc("mjc:solmix", pxr::SdfValueTypeNames->Double, VtValue(static_cast<double>(shape.solmix)));
    }
    if (shape.margin != defaults.margin) {
        mjc("mjc:margin", pxr::SdfValueTypeNames->Double, VtValue(static_cast<double>(shape.margin)));
    }
    if (shape.gap != defaults.gap) {
        mjc("mjc:gap", pxr::SdfValueTypeNames->Double, VtValue(static_cast<double>(shape.gap)));
    }
    if (!std::equal(shape.solref, shape.solref + 2, defaults.solref)) {
        mjc("mjc:solref", pxr::SdfValueTypeNames->DoubleArray,
            VtValue(pxr::VtDoubleArray{shape.solref[0], shape.solref[1]}));
    }
    if (!std::equal(shape.solimp, shape.solimp + 5, defaults.solimp)) {
        mjc("mjc:solimp", pxr::SdfValueTypeNames->DoubleArray,
            VtValue(pxr::VtDoubleArray{shape.solimp[0], shape.solimp[1], shape.solimp[2], shape.solimp[3],
                                       shape.solimp[4]}));
    }
    classes_[{shape.contype, shape.conaffinity}].push_back(prim.GetPath());
    Bind(prim, ColliderMaterial(shape));
}

// ---------------------------------------------------------------------------
// Joints and actuators
// ---------------------------------------------------------------------------

// A joint at the child origin: localPos1 is zero and both frames turn X onto the axis, so the authored pose
// is the joint's zero and ImportUsd measures no rest offset.
void StageWriter::WriteJoint(const scene::JointRecord& joint) {
    const size_t count = scene_.RigidBodyCount();
    if (joint.type == scene::JointType::Free || joint.child_body >= count || !Written(joint.child_body)) {
        return;
    }
    const SdfPath child = BodyPrim(joint.child_body);
    const SdfPath parent = joint.parent_body < count ? BodyPrim(joint.parent_body) : SdfPath();
    const bool revolute = joint.type == scene::JointType::Revolute;
    const bool prismatic = joint.type == scene::JointType::Prismatic;
    TfToken axis = pxr::UsdPhysicsTokens->x;
    pxr::GfQuatd align = pxr::GfQuatd::GetIdentity();
    if (revolute || prismatic) {
        const pxr::GfVec3d direction = pxr::GfVec3d(joint.axis.x, joint.axis.y, joint.axis.z).GetNormalized();
        const auto along = [&](const pxr::GfVec3d& unit) { return (direction - unit).GetLength() < 1e-6; };
        if (along(pxr::GfVec3d::YAxis())) {
            axis = pxr::UsdPhysicsTokens->y;
        } else if (along(pxr::GfVec3d::ZAxis())) {
            axis = pxr::UsdPhysicsTokens->z;
        } else if (!along(pxr::GfVec3d::XAxis())) {
            align = pxr::GfRotation(pxr::GfVec3d::XAxis(), direction).GetQuat();
        }
    }
    const Pose holder = parent.IsEmpty() ? Pose{} : WorldOf(joint.parent_body);
    const Pose local = holder.Inverse() * WorldOf(joint.child_body);
    const SdfPath path = Child(child, joint.name, "joint_" + std::to_string(joint.id));
    pxr::UsdPhysicsJoint usd;
    const double unit = revolute ? kRadToDeg : 1.0;
    if (revolute) {
        const pxr::UsdPhysicsRevoluteJoint hinge = pxr::UsdPhysicsRevoluteJoint::Define(stage_, path);
        hinge.CreateAxisAttr(VtValue(axis));
        if (joint.has_lower_limit) hinge.CreateLowerLimitAttr(VtValue(static_cast<float>(joint.lower_limit * unit)));
        if (joint.has_upper_limit) hinge.CreateUpperLimitAttr(VtValue(static_cast<float>(joint.upper_limit * unit)));
        usd = hinge;
    } else if (prismatic) {
        const pxr::UsdPhysicsPrismaticJoint slider = pxr::UsdPhysicsPrismaticJoint::Define(stage_, path);
        slider.CreateAxisAttr(VtValue(axis));
        if (joint.has_lower_limit) slider.CreateLowerLimitAttr(VtValue(joint.lower_limit));
        if (joint.has_upper_limit) slider.CreateUpperLimitAttr(VtValue(joint.upper_limit));
        usd = slider;
    } else if (joint.type == scene::JointType::Spherical) {
        usd = pxr::UsdPhysicsSphericalJoint::Define(stage_, path);
    } else {
        usd = pxr::UsdPhysicsFixedJoint::Define(stage_, path);
    }
    if (!parent.IsEmpty()) {
        usd.CreateBody0Rel().SetTargets({parent});
    }
    usd.CreateBody1Rel().SetTargets({child});
    usd.CreateLocalPos0Attr(VtValue(pxr::GfVec3f(local.position)));
    usd.CreateLocalRot0Attr(VtValue(pxr::GfQuatf(local.rotation * align)));
    usd.CreateLocalPos1Attr(VtValue(pxr::GfVec3f(0.0f)));
    usd.CreateLocalRot1Attr(VtValue(pxr::GfQuatf(align)));

    pxr::UsdPrim prim = usd.GetPrim();
    if ((revolute || prismatic) && joint.initial_position != 0.0f) {
        const std::string dof = revolute ? "angular" : "linear";
        prim.AddAppliedSchema(TfToken("PhysicsJointStateAPI:" + dof));
        prim.CreateAttribute(TfToken("state:" + dof + ":physics:position"), pxr::SdfValueTypeNames->Float)
            .Set(static_cast<float>(joint.initial_position * unit));
    }
    if (joint.armature != 0.0f || joint.frictionloss != 0.0f || joint.damping != 0.0f) {
        prim.AddAppliedSchema(TfToken("MjcJointAPI"));
        const auto set = [&](const char* name, float value) {
            if (value != 0.0f) {
                prim.CreateAttribute(TfToken(name), pxr::SdfValueTypeNames->Double).Set(static_cast<double>(value));
            }
        };
        set("mjc:armature", joint.armature);
        set("mjc:frictionloss", joint.frictionloss);
        set("mjc:damping", joint.damping);
    }
    if (joint.armature != 0.0f) {
        prim.AddAppliedSchema(TfToken("PhysxJointAPI"));
        prim.CreateAttribute(TfToken("physxJoint:armature"), pxr::SdfValueTypeNames->Float).Set(joint.armature);
    }
    if (joint.stiffness != 0.0f) {
        Warn("USD_EXPORT_JOINT_SPRING", path.GetString(), "the joint spring has no UsdPhysics form; it is left out");
    }
    if (joint.mimic_source != scene::kInvalidJoint) {
        Warn("USD_EXPORT_JOINT_MIMIC", path.GetString(), "the mimic coupling has no UsdPhysics form; it is left out");
    }
    if (!IsIdentity(joint.child_frame)) {
        Warn("USD_EXPORT_JOINT_CHILD_FRAME", path.GetString(), "the joint turns about the child origin; its child "
             "frame is left out");
    }
    joint_path_[joint.id] = path;
}

// A fixed joint to the world under each static joint parent, and ArticulationRootAPI on each tree's root.
void StageWriter::WriteRoots() {
    for (scene::BodyId b = 0; b < kind_.size(); ++b) {
        if (!Written(b) || kind_[b] == BodyKind::Static || !jointed_[b]) {
            continue;
        }
        const SdfPath body = BodyPrim(b);
        if (kind_[b] == BodyKind::Rooted) {
            const Pose world = WorldOf(b);
            const pxr::UsdPhysicsFixedJoint fixed =
                pxr::UsdPhysicsFixedJoint::Define(stage_, Child(body, "fixed_base", "fixed_base"));
            fixed.CreateBody1Rel().SetTargets({body});
            fixed.CreateLocalPos0Attr(VtValue(pxr::GfVec3f(world.position)));
            fixed.CreateLocalRot0Attr(VtValue(pxr::GfQuatf(world.rotation)));
            fixed.CreateLocalPos1Attr(VtValue(pxr::GfVec3f(0.0f)));
            fixed.CreateLocalRot1Attr(VtValue(pxr::GfQuatf(1.0f)));
        }
        const int joint = incoming_[b];
        if (joint < 0 || !Written(scene_.GetJoint(static_cast<scene::JointId>(joint)).parent_body)) {
            pxr::UsdPhysicsArticulationRootAPI::Apply(stage_->GetPrimAtPath(body));
        }
    }
}

// Position and velocity actuators as the joint's DriveAPI, in degrees for angular joints; position drives
// carry the 2*sqrt(stiffness) damping the engine gives them. Others become NukaActuator prims.
void StageWriter::WriteActuator(const scene::ActuatorRecord& actuator) {
    if (actuator.joint_id >= joint_path_.size() || joint_path_[actuator.joint_id].IsEmpty()) {
        if (whole_) {
            Warn("USD_EXPORT_ACTUATOR_JOINT", root_.GetString(), "actuator '" + actuator.name + "' drives no "
                 "written joint; it is left out");
        }
        return;
    }
    const scene::JointRecord& joint = scene_.GetJoint(actuator.joint_id);
    const SdfPath& joint_path = joint_path_[actuator.joint_id];
    const bool revolute = joint.type == scene::JointType::Revolute;
    const bool servo = actuator.type == scene::ActuatorType::Position || actuator.type == scene::ActuatorType::Velocity;
    if ((revolute || joint.type == scene::JointType::Prismatic) && servo && actuator.gain > 0.0f &&
        driven_.insert(actuator.joint_id).second) {
        const double unit = revolute ? kDegToRad : 1.0;
        const double gain = actuator.gain;
        const pxr::UsdPhysicsDriveAPI drive =
            pxr::UsdPhysicsDriveAPI::Apply(stage_->GetPrimAtPath(joint_path), TfToken(revolute ? "angular" : "linear"));
        drive.CreateTypeAttr(VtValue(pxr::UsdPhysicsTokens->force));
        if (actuator.type == scene::ActuatorType::Position) {
            drive.CreateStiffnessAttr(VtValue(static_cast<float>(gain * unit)));
            drive.CreateDampingAttr(VtValue(static_cast<float>(2.0 * std::sqrt(gain) * unit)));
        } else {
            drive.CreateDampingAttr(VtValue(static_cast<float>(gain * unit)));
        }
        if (actuator.force_limit > 0.0f) {
            drive.CreateMaxForceAttr(VtValue(actuator.force_limit));
        }
        return;
    }
    const SdfPath path = Child(BodyPrim(joint.child_body), actuator.name, "actuator_" + std::to_string(actuator.id));
    const pxr::UsdPrim prim = stage_->DefinePrim(path, TfToken("NukaActuator"));
    prim.CreateRelationship(TfToken("nuka:joint")).SetTargets({joint_path});
    prim.CreateAttribute(TfToken("nuka:type"), pxr::SdfValueTypeNames->Token)
        .Set(TfToken(ActuatorToken(actuator.type)));
    prim.CreateAttribute(TfToken("nuka:gain"), pxr::SdfValueTypeNames->Float).Set(actuator.gain);
    prim.CreateAttribute(TfToken("nuka:forceLimit"), pxr::SdfValueTypeNames->Float).Set(actuator.force_limit);
}

// ---------------------------------------------------------------------------
// Sensors, cameras, lights, environment, terrain
// ---------------------------------------------------------------------------

void StageWriter::WriteSensor(const scene::SensorDesc& sensor) {
    const bool on_body = sensor.mount == scene::MountFrame::Body && sensor.mount_index < scene_.RigidBodyCount();
    if (!on_body || sensor.type == scene::SensorType::Tactile) {
        if (whole_) {
            Warn("USD_EXPORT_SENSOR", root_.GetString(), "only non-tactile sensors on a body are written; '" +
                 sensor.name + "' is left out");
        }
        return;
    }
    if (!in_scope_[sensor.mount_index]) {
        return;
    }
    const SdfPath body = BodyPrim(sensor.mount_index);
    const pxr::UsdPrim prim =
        stage_->DefinePrim(Child(body, sensor.name, "sensor_" + std::to_string(sensor.id)), TfToken("NukaSensor"));
    prim.CreateRelationship(TfToken("nuka:body")).SetTargets({body});
    prim.CreateAttribute(TfToken("nuka:type"), pxr::SdfValueTypeNames->Token).Set(TfToken(SensorToken(sensor.type)));
    if (sensor.joint_id < joint_path_.size() && !joint_path_[sensor.joint_id].IsEmpty()) {
        prim.CreateRelationship(TfToken("nuka:joint")).SetTargets({joint_path_[sensor.joint_id]});
    }
    if (sensor.sample_rate_hz > 0.0f) {
        prim.CreateAttribute(TfToken("nuka:sampleRate"), pxr::SdfValueTypeNames->Float).Set(sensor.sample_rate_hz);
    }
    // ImportUsd reads the offset as written: translate, then degrees applied X, Y, Z.
    const Pose offset = PoseOf(sensor.local_offset);
    const pxr::GfVec3d angles = pxr::GfRotation(offset.rotation)
                                    .Decompose(pxr::GfVec3d::XAxis(), pxr::GfVec3d::YAxis(), pxr::GfVec3d::ZAxis());
    prim.CreateAttribute(TfToken("xformOp:translate"), pxr::SdfValueTypeNames->Double3).Set(offset.position);
    prim.CreateAttribute(TfToken("xformOp:rotateXYZ"), pxr::SdfValueTypeNames->Float3).Set(pxr::GfVec3f(angles));
    prim.CreateAttribute(TfToken("xformOpOrder"), pxr::SdfValueTypeNames->TokenArray)
        .Set(pxr::VtTokenArray{TfToken("xformOp:translate"), TfToken("xformOp:rotateXYZ")});
    const auto number = [&](const char* name, float value) {
        prim.CreateAttribute(TfToken(name), pxr::SdfValueTypeNames->Float).Set(value);
    };
    const auto count = [&](const char* name, int value) {
        prim.CreateAttribute(TfToken(name), pxr::SdfValueTypeNames->Int).Set(value);
    };
    if (sensor.type == scene::SensorType::Lidar || sensor.type == scene::SensorType::RangeScan) {
        const scene::LidarPattern& fan = sensor.lidar;
        number("nuka:horizontalFov", fan.az_max - fan.az_min);
        number("nuka:verticalFov", fan.el_max - fan.el_min);
        count("nuka:horizontalResolution", fan.az_count);
        count("nuka:verticalResolution", fan.el_count);
        number("nuka:minRange", fan.min_range);
        number("nuka:maxRange", fan.max_range);
        if (fan.az_min != -fan.az_max || fan.el_min != -fan.el_max) {
            Warn("USD_EXPORT_SENSOR_FAN", prim.GetPath().GetString(), "the ray fan is written centred; its offset is "
                 "left out");
        }
    } else if (sensor.type == scene::SensorType::Depth || sensor.type == scene::SensorType::Camera) {
        count("nuka:width", sensor.cam.width);
        count("nuka:height", sensor.cam.height);
        number("nuka:verticalFov", static_cast<float>(sensor.cam.vfov_degrees * kDegToRad));
    }
}

void StageWriter::WriteCamera(const scene::CameraRecord& camera) {
    const bool attached = camera.attached_body < scene_.RigidBodyCount();
    if (attached ? !in_scope_[camera.attached_body] : !whole_) {
        return;
    }
    const SdfPath parent = attached ? BodyPrim(camera.attached_body) : GroupOf(camera.name);
    const pxr::UsdGeomCamera usd =
        pxr::UsdGeomCamera::Define(stage_, Child(parent, camera.name, "camera_" + std::to_string(camera.id)));
    Place(usd, PoseOf(camera.local_transform));
    const double half = 0.5 * camera.vertical_fov_degrees * kDegToRad;
    usd.CreateFocalLengthAttr(VtValue(static_cast<float>(kVerticalAperture / (2.0 * std::tan(half)))));
    usd.CreateVerticalApertureAttr(VtValue(static_cast<float>(kVerticalAperture)));
    usd.CreateClippingRangeAttr(VtValue(pxr::GfVec2f(camera.near_clip, camera.far_clip)));
    usd.CreateFocusDistanceAttr(VtValue(camera.focus_distance));
}

// A Nuka intensity in the export's light units; under nits a sun spreads over the schema's 0.53 degree disc.
float StageWriter::Intensity(float nuka, bool sun) const {
    if (options_.light_units == UsdLightUnits::Isaac) {
        return static_cast<float>(nuka * kIsaacPerNuka);
    }
    if (options_.light_units == UsdLightUnits::Raw || !sun) {
        return nuka;
    }
    const double half = 0.5 * kSunAngle * kDegToRad;
    return static_cast<float>(nuka / (kPi * std::sin(half) * std::sin(half)));
}

void StageWriter::WriteLight(const scene::LightRecord& light) {
    const bool attached = light.attached_body < scene_.RigidBodyCount();
    if (attached ? !in_scope_[light.attached_body] : !whole_) {
        return;
    }
    const SdfPath parent = attached ? BodyPrim(light.attached_body) : GroupOf(light.name);
    const SdfPath path = Child(parent, light.name, "light_" + std::to_string(light.id));
    pxr::UsdPrim prim;
    if (light.type == scene::LightType::Directional) {
        const pxr::UsdLuxDistantLight sun = pxr::UsdLuxDistantLight::Define(stage_, path);
        sun.CreateAngleAttr(VtValue(static_cast<float>(kSunAngle)));
        prim = sun.GetPrim();
    } else if (light.type == scene::LightType::Area) {
        prim = pxr::UsdLuxRectLight::Define(stage_, path).GetPrim();
    } else {
        const pxr::UsdLuxSphereLight bulb = pxr::UsdLuxSphereLight::Define(stage_, path);
        bulb.CreateTreatAsPointAttr(VtValue(true));
        prim = bulb.GetPrim();
        if (light.type == scene::LightType::Spot) {
            Warn("USD_EXPORT_SPOT_LIGHT", path.GetString(), "a spot light records no cone; it is written as a point "
                 "light");
        }
    }
    Place(pxr::UsdGeomXformable(prim), PoseOf(light.local_transform));
    const pxr::UsdLuxLightAPI api(prim);
    api.CreateIntensityAttr(VtValue(Intensity(light.intensity, light.type == scene::LightType::Directional)));
    api.CreateColorAttr(VtValue(pxr::GfVec3f(light.color.x, light.color.y, light.color.z)));
}

// The HDRI as a latlong DomeLight; the look parameters of Nuka's own sky have no UsdLux form.
void StageWriter::WriteEnvironment() {
    const scene::EnvironmentRecord& env = scene_.Environment();
    std::vector<std::string> dropped;
    if (env.yaw_deg != 0.0f) dropped.push_back("yaw");
    if (env.hdri.empty() && env.intensity != 1.0f) dropped.push_back("intensity");
    if (env.use_scene_materials || env.ibl_full_fill || env.specular_env) dropped.push_back("render switches");
    if (env.exposure_ev != 0.0f || env.grade != 0.0f || env.sun_disc != 0.0f) dropped.push_back("grading");
    if (env.sky.enabled) dropped.push_back("sky");
    if (env.shadow.enabled) dropped.push_back("shadow bounds");
    if (!dropped.empty()) {
        std::string list;
        for (const std::string& item : dropped) {
            list += (list.empty() ? "" : ", ") + item;
        }
        Warn("USD_EXPORT_ENVIRONMENT", root_.GetString(), "the environment's " + list + " have no UsdLux form; "
             "they are left out");
    }
    if (env.hdri.empty()) {
        return;
    }
    const std::string file = TextureFile(env.hdri, "environment", root_.GetString());
    if (file.empty()) {
        return;
    }
    const pxr::UsdLuxDomeLight dome = pxr::UsdLuxDomeLight::Define(stage_, Child(root_, "Environment", "Environment"));
    dome.CreateTextureFileAttr(VtValue(pxr::SdfAssetPath(file)));
    dome.CreateTextureFormatAttr(VtValue(pxr::UsdLuxTokens->latlong));
    pxr::UsdLuxLightAPI(dome.GetPrim()).CreateIntensityAttr(VtValue(Intensity(env.intensity, false)));
}

// The terrain's height grid as a static triangle-mesh collider, two triangles per cell.
void StageWriter::WriteTerrain(const scene::TerrainRecord& terrain) {
    const SdfPath path = Child(GroupOf(terrain.name), terrain.name, "terrain");
    terrain::HeightField field;
    try {
        field = terrain::TerrainField(terrain);
    } catch (const std::runtime_error& error) {
        Warn("USD_EXPORT_TERRAIN", path.GetString(), std::string(error.what()) + "; the terrain is left out");
        return;
    }
    const pxr::UsdPrim holder = pxr::UsdGeomXform::Define(stage_, path).GetPrim();
    holder.CreateAttribute(TfToken("nuka:staticBody"), pxr::SdfValueTypeNames->Bool).Set(true);
    const pxr::UsdGeomMesh mesh = pxr::UsdGeomMesh::Define(stage_, path.AppendChild(TfToken("surface")));
    pxr::VtVec3fArray points(static_cast<size_t>(field.nrow) * field.ncol);
    for (uint32_t r = 0; r < field.nrow; ++r) {
        for (uint32_t c = 0; c < field.ncol; ++c) {
            points[static_cast<size_t>(r) * field.ncol + c] =
                pxr::GfVec3f(field.origin.x + static_cast<float>(c) * field.cell_x,
                             field.origin.y + static_cast<float>(r) * field.cell_y, terrain::WorldZAt(field, r, c));
        }
    }
    pxr::VtIntArray indices;
    indices.reserve(static_cast<size_t>(field.nrow - 1u) * (field.ncol - 1u) * 6u);
    for (uint32_t r = 0; r + 1u < field.nrow; ++r) {
        for (uint32_t c = 0; c + 1u < field.ncol; ++c) {
            const int a = static_cast<int>(r * field.ncol + c), b = a + 1;
            const int d = a + static_cast<int>(field.ncol), e = d + 1;
            for (const int i : {a, b, e, a, e, d}) {
                indices.push_back(i);
            }
        }
    }
    mesh.CreatePointsAttr(VtValue(points));
    mesh.CreateFaceVertexCountsAttr(VtValue(pxr::VtIntArray(indices.size() / 3u, 3)));
    mesh.CreateFaceVertexIndicesAttr(VtValue(indices));
    mesh.CreateSubdivisionSchemeAttr(VtValue(pxr::UsdGeomTokens->none));
    AuthorExtent(mesh.GetPrim());
    pxr::UsdPhysicsCollisionAPI::Apply(mesh.GetPrim());
    pxr::UsdPhysicsMeshCollisionAPI::Apply(mesh.GetPrim())
        .CreateApproximationAttr(VtValue(pxr::UsdPhysicsTokens->none));
    Warn("USD_EXPORT_TERRAIN_MESH", path.GetString(), "the terrain is written as a triangle-mesh collider and reads "
         "back as one, not as a terrain");
}

// ---------------------------------------------------------------------------
// Filters and collision groups
// ---------------------------------------------------------------------------

void StageWriter::WriteFilteredPairs() {
    for (const auto& [a, b] : scene_.ExcludePairs()) {
        if (a >= kind_.size() || b >= kind_.size() || !in_scope_[a] || !in_scope_[b]) {
            continue;
        }
        if (!Written(a) || !Written(b)) {
            Warn("USD_EXPORT_FILTER_PAIR", root_.GetString(), "a collision filter with an anchor body has no prim "
                 "to hold it; it is left out");
            continue;
        }
        pxr::UsdPhysicsFilteredPairsAPI::Apply(stage_->GetPrimAtPath(BodyPrim(a)))
            .CreateFilteredPairsRel()
            .AddTarget(BodyPrim(b));
    }
}

// One group per (contype, conaffinity) class once any collider leaves the default 1, 1; two classes MuJoCo would
// not let touch (neither contype meets the other's conaffinity) filter each other.
void StageWriter::WriteCollisionGroups() {
    if (classes_.empty() || (classes_.size() == 1u && classes_.begin()->first == std::make_pair(1u, 1u))) {
        return;
    }
    if (classes_.size() > 31u) {
        Warn("USD_EXPORT_COLLISION_GROUPS", root_.GetString(), std::to_string(classes_.size()) + " collision "
             "classes are written; ImportUsd reads only the first 31");
    }
    const SdfPath scope = root_.AppendChild(TfToken("CollisionGroups"));
    pxr::UsdGeomScope::Define(stage_, scope);
    std::vector<std::pair<std::pair<uint32_t, uint32_t>, pxr::UsdPhysicsCollisionGroup>> groups;
    for (const auto& [bits, colliders] : classes_) {
        const std::string name =
            "contype_" + std::to_string(bits.first) + "_conaffinity_" + std::to_string(bits.second);
        const pxr::UsdPhysicsCollisionGroup group =
            pxr::UsdPhysicsCollisionGroup::Define(stage_, scope.AppendChild(TfToken(name)));
        const pxr::UsdRelationship includes = group.GetCollidersCollectionAPI().CreateIncludesRel();
        for (const SdfPath& collider : colliders) {
            includes.AddTarget(collider);
        }
        groups.emplace_back(bits, group);
    }
    for (size_t k = 0; k < groups.size(); ++k) {
        for (size_t l = k; l < groups.size(); ++l) {
            const auto& [ck, ak] = groups[k].first;
            const auto& [cl, al] = groups[l].first;
            if ((ck & al) != 0u || (cl & ak) != 0u) {
                continue;
            }
            groups[k].second.CreateFilteredGroupsRel().AddTarget(groups[l].second.GetPath());
            if (l != k) {
                groups[l].second.CreateFilteredGroupsRel().AddTarget(groups[k].second.GetPath());
            }
        }
    }
}

void StageWriter::ReportDropped() {
    const std::string world = root_.GetString();
    if (scene_.MediaCount() > 0u) {
        Warn("USD_EXPORT_MEDIA", world, std::to_string(scene_.MediaCount()) + " cloth, soft and MPM media have no "
             "UsdPhysics form; they are left out");
    }
    if (scene_.ScriptCount() > 0u) {
        Warn("USD_EXPORT_SCRIPTS", world, std::to_string(scene_.ScriptCount()) + " scripts are left out");
    }
    if (!scene_.ContactPairs().empty()) {
        Warn("USD_EXPORT_CONTACT_PAIRS", world, "explicit contact pairs have no UsdPhysics form; they are left out");
    }
    if (!scene_.InitialState().empty()) {
        Warn("USD_EXPORT_INITIAL_STATE", world, "the settled initial state is left out; joints start at their "
             "initial positions");
    }
    if (scene_.Settle().steps > 0 || !scene_.Settle().holds.empty()) {
        Warn("USD_EXPORT_SETTLE", world, "the settle directive is left out");
    }
}

void StageWriter::Write() {
    pxr::UsdGeomSetStageUpAxis(stage_, pxr::UsdGeomTokens->z);
    pxr::UsdGeomSetStageMetersPerUnit(stage_, 1.0);
    pxr::UsdPhysicsSetStageKilogramsPerUnit(stage_, 1.0);
    stage_->GetRootLayer()->SetDocumentation("Nuka scene written as UsdPhysics");
    stage_->SetDefaultPrim(pxr::UsdGeomXform::Define(stage_, root_).GetPrim());
    const pxr::UsdPhysicsScene physics =
        pxr::UsdPhysicsScene::Define(stage_, root_.AppendChild(TfToken("PhysicsScene")));
    physics.CreateGravityDirectionAttr(VtValue(pxr::GfVec3f(0.0f, 0.0f, -1.0f)));
    physics.CreateGravityMagnitudeAttr(VtValue(9.81f));

    Classify();
    if (whole_) {
        for (scene::MaterialId m = 0; m < scene_.MaterialCount(); ++m) {
            MaterialPath(m);
        }
    }
    for (scene::BodyId b = 0; b < scene_.RigidBodyCount(); ++b) {
        BodyPrim(b);
    }
    for (const scene::CollisionShapeRecord& shape : scene_.Shapes()) {
        if (Writes(shape) && HasMesh(shape)) {
            ++mesh_uses_[MeshHash(shape)];
        }
    }
    for (const scene::CollisionShapeRecord& shape : scene_.Shapes()) WriteShape(shape);
    for (const scene::JointRecord& joint : scene_.Joints()) WriteJoint(joint);
    WriteRoots();
    for (const scene::ActuatorRecord& actuator : scene_.Actuators()) WriteActuator(actuator);
    for (const scene::SensorDesc& sensor : scene_.Sensors()) WriteSensor(sensor);
    for (const scene::CameraRecord& camera : scene_.Cameras()) WriteCamera(camera);
    for (const scene::LightRecord& light : scene_.Lights()) WriteLight(light);
    if (whole_) {
        WriteEnvironment();
        for (const scene::TerrainRecord& terrain : scene_.Terrain()) WriteTerrain(terrain);
        ReportDropped();
    }
    WriteFilteredPairs();
    WriteCollisionGroups();
}

}  // namespace

UsdExportResult ExportUsd(const scene::SceneIR& scene, const std::string& path, const UsdExportOptions& options) {
    const std::filesystem::path target(path);
    const std::string extension = Lowercase(target.extension().string());
    if (extension != ".usda" && extension != ".usdc" && extension != ".usd" && extension != ".usdz") {
        throw std::runtime_error("USD export: '" + path + "' is not a .usd, .usda, .usdc or .usdz path");
    }
    // A package is staged as usdc plus its textures in a scratch folder beside it, then zipped.
    const bool package = extension == ".usdz";
    const std::string stem = target.stem().string();
    const std::filesystem::path dir = target.parent_path();
    const std::filesystem::path work = package ? dir / ("." + stem + "_usdz") : dir;
    const std::filesystem::path layer = package ? work / (stem + ".usdc") : target;
    if (package) {
        std::filesystem::remove_all(work);
    }
    if (!work.empty()) {
        std::filesystem::create_directories(work);
    }
    UsdExportResult result;
    const pxr::UsdStageRefPtr stage = pxr::UsdStage::CreateInMemory();
    StageWriter(scene, options, stage, work / (stem + "_textures"), result.diagnostics).Write();
    if (!stage->GetRootLayer()->Export(layer.string())) {
        throw std::runtime_error("USD export: cannot write " + layer.string());
    }
    if (package) {
        const bool packed = pxr::UsdUtilsCreateNewUsdzPackage(pxr::SdfAssetPath(layer.string()), target.string());
        std::filesystem::remove_all(work);
        if (!packed) {
            throw std::runtime_error("USD export: cannot write the package " + path);
        }
    }
    return result;
}

}  // namespace nuka::import
