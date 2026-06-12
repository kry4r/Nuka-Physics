// ---------------------------------------------------------------------------
// nuka::scene::nks - .nks scene Save/Load implementation (M2c, §3.7 tree form).
// ---------------------------------------------------------------------------
// HOST-ONLY. The .nks JSON shape follows the owner-amended spec §3.7:
//
//   { "nks_version": 1,
//     "physics_materials": { "<name>": {...} },   // keyed by material name
//     "render_materials":  { "<name>": {...} },   // keyed by material name
//     "tree": [ <node>, ... ],                    // SceneGraph pre-order, nested
//     "sensors": [...], "exclude_pairs": [...], "contact_pairs": [...] }
//
// The `tree` section is the SceneGraph serialized in PRE-ORDER via
// SceneGraph::Traverse: each node is { "name": <single segment>, optional
// "transform", optional component objects (rigid_body / collision_shape /
// visual_mesh / joint / actuator / camera / light), "children": [...] }. Sibling
// order == tree (tail-append) order, so a second Save is byte-identical. Group
// nodes (path-prefix nodes carrying no component) serialize as {name, children}.
//
// FIDELITY: the SceneIR RECORDS remain the cook-fidelity authority. Save walks
// the tree for STRUCTURE/NAMES/ORDER but reads field VALUES from the record the
// node maps to (via EntityOfBody/Shape/Joint reverse lookup), so the cook-
// critical legacy keys (contype/conaffinity/solref/solimp/condim/priority/
// friction_mu/solmix/margin/gap/decompose_mode + the mesh AssetRef into the
// sibling .nka) stay on each collision_shape object verbatim.
//
// INVERSE (Save<->Load) DESIGN — record_name vs derived path:
//   ProjectBody splits a body RECORD name on '/' into path-prefix group nodes +
//   a leaf body node, hung under the parent BODY's node. So a body's DERIVED
//   path = parent-body-path + '/' + (the record-name segments). On Load we
//   reconstruct a body's record name by walking up from the body node to (but
//   excluding) the nearest ancestor BODY node, joining the single segments with
//   '/'. That reconstructed suffix == the original record name in every case the
//   importers/Compose produce — EXCEPT when sibling-name dedup renamed a node
//   (AddEntity auto-suffixes _1/_2). When the derived suffix differs from the
//   stored original we emit an explicit "record_name" on the body node and Load
//   prefers it; otherwise name = derived suffix. The SecondSaveByteIdentical and
//   cook-byte-equality gates enforce this is a true inverse pair.
//
// ID ORDER: cook reads Bodies()/Shapes()/Joints() each in dense-id order, so the
// loaded records must land in the SAME id order as the original. Load replays in
// the SAME phase order RebuildFacade projects in (materials, then bodies, then
// shapes, then joints, then cameras/lights/actuators), each phase a pre-order
// tree walk. Because the importers/Compose add a body then its own shapes/joints
// before descending, the shapes/joints are grouped by body in body-id order,
// which equals the per-body pre-order encounter — so the phase-walk reproduces
// the original dense-id order exactly.
//
// Mesh geometry (record mesh_vertices/mesh_indices) is offloaded to the sibling
// .nka as CMSH (collision) / MESH (visual) chunks, deduped by content hash, and
// referenced from the JSON via AssetRef text. Same SceneIR => byte-identical
// .nks + .nka (the roundtrip gate asserts this).
// ---------------------------------------------------------------------------

#include "scene/format/nks.hpp"

#include "import/mjcf_importer.hpp"
#include "import/urdf_importer.hpp"
#include "import/usd_importer.hpp"
#include "scene/asset/asset_ref.hpp"
#include "scene/asset/nka.hpp"
#include "scene/format/json.hpp"
#include "scene/scene_compose.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace nuka::scene::nks {

namespace {

using json::Value;

// ---------------------------------------------------------------------------
// small helpers: enums <-> strings (own format; readable + stable)
// ---------------------------------------------------------------------------
const char* ShapeTypeName(ShapeType t) {
    switch (t) {
        case ShapeType::Sphere:      return "sphere";
        case ShapeType::Capsule:     return "capsule";
        case ShapeType::Box:         return "box";
        case ShapeType::Plane:       return "plane";
        case ShapeType::ConvexHull:  return "convex_hull";
        case ShapeType::TriMesh:     return "trimesh";
        case ShapeType::HeightField: return "heightfield";
    }
    return "box";
}
ShapeType ShapeTypeFromName(const std::string& s) {
    if (s == "sphere") return ShapeType::Sphere;
    if (s == "capsule") return ShapeType::Capsule;
    if (s == "box") return ShapeType::Box;
    if (s == "plane") return ShapeType::Plane;
    if (s == "convex_hull") return ShapeType::ConvexHull;
    if (s == "trimesh") return ShapeType::TriMesh;
    if (s == "heightfield") return ShapeType::HeightField;
    return ShapeType::Box;
}

const char* JointTypeName(JointType t) {
    switch (t) {
        case JointType::Revolute:  return "revolute";
        case JointType::Prismatic: return "prismatic";
        case JointType::Fixed:     return "fixed";
        case JointType::Spherical: return "spherical";
        case JointType::Free:      return "free";
    }
    return "revolute";
}
JointType JointTypeFromName(const std::string& s) {
    if (s == "revolute") return JointType::Revolute;
    if (s == "prismatic") return JointType::Prismatic;
    if (s == "fixed") return JointType::Fixed;
    if (s == "spherical") return JointType::Spherical;
    if (s == "free") return JointType::Free;
    return JointType::Revolute;
}

const char* SensorTypeName(SensorType t) {
    switch (t) {
        case SensorType::Imu:         return "imu";
        case SensorType::Lidar:       return "lidar";
        case SensorType::Camera:      return "camera";
        case SensorType::ForceTorque: return "force_torque";
        case SensorType::Contact:     return "contact";
        case SensorType::FramePose:   return "frame_pose";
    }
    return "imu";
}
SensorType SensorTypeFromName(const std::string& s) {
    if (s == "imu") return SensorType::Imu;
    if (s == "lidar") return SensorType::Lidar;
    if (s == "camera") return SensorType::Camera;
    if (s == "force_torque") return SensorType::ForceTorque;
    if (s == "contact") return SensorType::Contact;
    if (s == "frame_pose") return SensorType::FramePose;
    return SensorType::Imu;
}

LightType LightTypeFromName(const std::string& s) {
    if (s == "point") return LightType::Point;
    if (s == "directional") return LightType::Directional;
    if (s == "spot") return LightType::Spot;
    if (s == "area") return LightType::Area;
    return LightType::Point;
}

const char* ActuatorTypeName(ActuatorType t) {
    switch (t) {
        case ActuatorType::Motor:    return "motor";
        case ActuatorType::Position: return "position";
        case ActuatorType::Velocity: return "velocity";
        case ActuatorType::Force:    return "force";
    }
    return "motor";
}
ActuatorType ActuatorTypeFromName(const std::string& s) {
    if (s == "motor") return ActuatorType::Motor;
    if (s == "position") return ActuatorType::Position;
    if (s == "velocity") return ActuatorType::Velocity;
    if (s == "force") return ActuatorType::Force;
    return ActuatorType::Motor;
}

const char* DecomposeModeName(DecomposeMode m) {
    switch (m) {
        case DecomposeMode::Auto:  return "auto";
        case DecomposeMode::Force: return "force";
        case DecomposeMode::Skip:  return "skip";
    }
    return "auto";
}
DecomposeMode DecomposeModeFromName(const std::string& s) {
    if (s == "auto") return DecomposeMode::Auto;
    if (s == "force") return DecomposeMode::Force;
    if (s == "skip") return DecomposeMode::Skip;
    return DecomposeMode::Auto;
}

// ---------------------------------------------------------------------------
// math <-> json
// ---------------------------------------------------------------------------
// Checked element access: Value::Elements() on a non-array returns an empty
// vector, so unchecked [i] would be UB on malformed input (override overlays
// are hand-authored, so this path IS reachable). Throws a json::ParseError
// instead.
float FloatAt(const Value& a, size_t i, const char* what) {
    if (!a.IsArray() || i >= a.Elements().size()) {
        throw json::ParseError(std::string("nks: expected a float array of at least ") +
                               std::to_string(i + 1) + " elements for " + what);
    }
    return a.Elements()[i].AsFloat();
}

Value Vec3Json(const math::Vec3& v) {
    Value a = Value::Array();
    a.PushBack(Value::Float(v.x));
    a.PushBack(Value::Float(v.y));
    a.PushBack(Value::Float(v.z));
    return a;
}
math::Vec3 Vec3FromJson(const Value& a) {
    return math::Vec3{FloatAt(a, 0, "vec3"), FloatAt(a, 1, "vec3"),
                      FloatAt(a, 2, "vec3")};
}

Value QuatJson(const math::Quat& q) {  // (w,x,y,z)
    Value a = Value::Array();
    a.PushBack(Value::Float(q.w));
    a.PushBack(Value::Float(q.x));
    a.PushBack(Value::Float(q.y));
    a.PushBack(Value::Float(q.z));
    return a;
}
math::Quat QuatFromJson(const Value& a) {
    return math::Quat{FloatAt(a, 0, "quat"), FloatAt(a, 1, "quat"),
                      FloatAt(a, 2, "quat"), FloatAt(a, 3, "quat")};
}

Value TransformJson(const math::Transform& t) {
    Value o = Value::Object();
    o.Set("pos", Vec3Json(t.position));
    o.Set("quat", QuatJson(t.rotation));
    return o;
}
math::Transform TransformFromJson(const Value& o) {
    math::Transform t;
    t.position = Vec3FromJson(o.At("pos"));
    t.rotation = QuatFromJson(o.At("quat"));
    return t;
}

Value FloatArray(const float* p, int n) {
    Value a = Value::Array();
    for (int i = 0; i < n; ++i) a.PushBack(Value::Float(p[i]));
    return a;
}

// ---------------------------------------------------------------------------
// Save
// ---------------------------------------------------------------------------

// One shared mesh-chunk dedup table for a single Save: content-hash -> AssetRef
// text. The .nka file basename is captured so the AssetRef path matches the
// sibling file.
struct MeshSink {
    NkaWriter writer;
    std::string nka_basename;
    std::unordered_map<uint64_t, std::string> by_hash;  // content hash -> assetref text

    // Append a collision mesh (CMSH) deduped by content hash; return AssetRef text.
    std::string AddCollisionMesh(const std::vector<float>& verts,
                                 const std::vector<uint32_t>& indices) {
        const std::vector<uint8_t> payload = EncodeCollisionMesh(verts, indices);
        const uint64_t hash = NkaContentHash(payload);
        const auto it = by_hash.find(hash);
        if (it != by_hash.end()) return it->second;
        const uint32_t index = writer.AddChunk(NkaTagCmsh(), payload);
        AssetRef ref;
        ref.nka_path = nka_basename;
        ref.fourcc = NkaTagCmsh();
        ref.index = index;
        const std::string text = ToString(ref);
        by_hash.emplace(hash, text);
        return text;
    }
};

// Serialize a collision/visual shape RECORD into the per-node component object
// (full cook fidelity: every legacy field stays here verbatim). The node `name`
// is the geom node's tree name and is set by the caller, not here.
Value SaveShape(const CollisionShapeRecord& s, MeshSink& sink) {
    Value o = Value::Object();
    o.Set("type", Value::Str(ShapeTypeName(s.type)));
    o.Set("local", TransformJson(s.local_transform));
    o.Set("half_extents", Vec3Json(s.half_extents));
    o.Set("radius", Value::Float(s.radius));
    o.Set("half_height", Value::Float(s.half_height));
    o.Set("material_id", Value::Int(static_cast<int64_t>(s.material_id)));
    // Legacy-fidelity contact metadata (we own this format; these reproduce the
    // cook-identical record).
    o.Set("contype", Value::Int(s.contype));
    o.Set("conaffinity", Value::Int(s.conaffinity));
    o.Set("collision_group", Value::Int(s.collision_group));
    o.Set("solref", FloatArray(s.solref, 2));
    o.Set("solimp", FloatArray(s.solimp, 5));
    o.Set("friction_mu", Value::Float(s.friction_mu));
    o.Set("priority", Value::Int(s.priority));
    o.Set("solmix", Value::Float(s.solmix));
    o.Set("margin", Value::Float(s.margin));
    o.Set("gap", Value::Float(s.gap));
    o.Set("condim", Value::Int(s.condim));
    o.Set("decompose_mode", Value::Str(DecomposeModeName(s.decompose_mode)));
    o.Set("decompose_max_pieces", Value::Int(s.decompose_max_pieces));
    // Inline mesh geometry -> .nka CMSH chunk (deduped); store the AssetRef text.
    if (!s.mesh_vertices.empty() && !s.mesh_indices.empty()) {
        o.Set("mesh", Value::Str(sink.AddCollisionMesh(s.mesh_vertices, s.mesh_indices)));
    }
    return o;
}

// A body's "rigid_body" component object (mass / inertia / parent_id / inertial
// frame / static flag). parent_id is stored explicitly: the tree carries the
// kinematic hierarchy structurally, but the cook reads RigidBodyRecord.parent_id
// directly (ResolveWorldTransform), so it must round-trip verbatim.
Value SaveRigidBody(const RigidBodyRecord& b) {
    Value o = Value::Object();
    o.Set("parent_id", Value::Int(static_cast<int64_t>(b.parent_id)));
    o.Set("inertial", TransformJson(b.inertial_transform));
    o.Set("mass", Value::Float(b.mass));
    o.Set("inertia", Vec3Json(b.inertia));
    o.Set("is_static", Value::Bool(b.is_static));
    return o;
}

Value SaveJoint(const JointRecord& j) {
    Value o = Value::Object();
    // The joint name is NOT the tree node's name (a joint rides on the CHILD
    // body's node, or on an auxiliary joint node named after the joint), so it
    // is stored on the component object itself for an exact record round-trip.
    o.Set("name", Value::Str(j.name));
    o.Set("type", Value::Str(JointTypeName(j.type)));
    o.Set("parent_body", Value::Int(static_cast<int64_t>(j.parent_body)));
    o.Set("child_body", Value::Int(static_cast<int64_t>(j.child_body)));
    o.Set("axis", Vec3Json(j.axis));
    o.Set("parent_frame", TransformJson(j.parent_frame));
    o.Set("child_frame", TransformJson(j.child_frame));
    o.Set("lower_limit", Value::Float(j.lower_limit));
    o.Set("upper_limit", Value::Float(j.upper_limit));
    o.Set("damping", Value::Float(j.damping));
    o.Set("armature", Value::Float(j.armature));
    o.Set("stiffness", Value::Float(j.stiffness));
    o.Set("initial_position", Value::Float(j.initial_position));
    return o;
}

Value SaveActuator(const ActuatorRecord& a) {
    Value o = Value::Object();
    o.Set("name", Value::Str(a.name));
    o.Set("type", Value::Str(ActuatorTypeName(a.type)));
    o.Set("joint_id", Value::Int(static_cast<int64_t>(a.joint_id)));
    o.Set("gain", Value::Float(a.gain));
    o.Set("force_limit", Value::Float(a.force_limit));
    return o;
}

Value SaveSensor(const SensorRecord& s) {
    Value o = Value::Object();
    o.Set("name", Value::Str(s.name));
    o.Set("type", Value::Str(SensorTypeName(s.type)));
    o.Set("attached_body", Value::Int(static_cast<int64_t>(s.attached_body)));
    o.Set("local", TransformJson(s.local_transform));
    o.Set("sample_rate_hz", Value::Float(s.sample_rate_hz));
    return o;
}

// Cameras / lights serialize from their ECS COMPONENT inline on their own tree
// node (see SaveNode): the component is a 1:1 projection of the record, the node
// carries the name, and the attached body is the parent body node — so there is
// no record-array Save helper for them under the §3.7 tree form.

// The authored MaterialRecord (the cook-fidelity authority) carries BOTH render
// fields (base_color/alpha/roughness/metallic) and a physics field (friction_mu).
// §3.7 splits these into two keyed sections. We store the record's raw values so
// Load reconstructs the identical MaterialRecord:
//   render_materials[name] = {base_color:[r,g,b,a], metallic, roughness}
//   physics_materials[name] = {static_friction, dynamic_friction}  (== friction_mu)
// ProjectMaterial maps friction_mu -> both static/dynamic equally, so storing the
// record value under both (and reading static_friction back into friction_mu) is
// an exact round-trip — including the <0 "inherit material μ" sentinel.
Value SaveRenderMaterial(const MaterialRecord& m) {
    Value o = Value::Object();
    Value bc = Value::Array();
    bc.PushBack(Value::Float(m.base_color.x));
    bc.PushBack(Value::Float(m.base_color.y));
    bc.PushBack(Value::Float(m.base_color.z));
    bc.PushBack(Value::Float(m.alpha));
    o.Set("base_color", std::move(bc));
    o.Set("metallic", Value::Float(m.metallic));
    o.Set("roughness", Value::Float(m.roughness));
    return o;
}

Value SavePhysicsMaterial(const MaterialRecord& m) {
    Value o = Value::Object();
    o.Set("static_friction", Value::Float(m.friction_mu));
    o.Set("dynamic_friction", Value::Float(m.friction_mu));
    return o;
}

}  // namespace

namespace {

// Reverse maps entity -> record id, plus the record kind a node carries. Built
// once per Save so the tree walk can fetch the exact record (the cook-fidelity
// authority) behind each node.
struct NodeIndex {
    std::unordered_map<EntityId, BodyId, EntityIdHash>   body;
    std::unordered_map<EntityId, ShapeId, EntityIdHash>  shape;
    std::unordered_map<EntityId, JointId, EntityIdHash>  joint;
    std::unordered_map<EntityId, ActuatorId, EntityIdHash> actuator;
    std::unordered_map<EntityId, CameraId, EntityIdHash> camera;
    std::unordered_map<EntityId, LightId, EntityIdHash>  light;
};

NodeIndex BuildNodeIndex(const SceneIR& scene) {
    NodeIndex idx;
    for (BodyId i = 0; i < scene.RigidBodyCount(); ++i) {
        const EntityId e = scene.EntityOfBody(i);
        if (e != kInvalidEntity) idx.body.emplace(e, i);
    }
    for (ShapeId i = 0; i < scene.ShapeCount(); ++i) {
        const EntityId e = scene.EntityOfShape(i);
        if (e != kInvalidEntity) idx.shape.emplace(e, i);
    }
    for (JointId i = 0; i < scene.JointCount(); ++i) {
        const EntityId e = scene.EntityOfJoint(i);
        if (e != kInvalidEntity) idx.joint.emplace(e, i);
    }
    // Actuators / cameras / lights have no EntityOf* accessor; match by component
    // presence during the tree walk (their fields are self-contained on the node).
    return idx;
}

// Walk up from `node` to (but excluding) the nearest ancestor node that maps to
// a BODY record; join the single segments between them with '/'. That suffix is
// the body's DERIVED record name (group prefix segments included). Returns ""
// for a node directly under root with no group prefix only when node==body leaf.
std::string DerivedBodyName(const SceneGraph& tree, const NodeIndex& idx,
                            const std::shared_ptr<SceneNode>& body_node) {
    std::vector<std::string> segs;
    segs.push_back(body_node->name);
    auto cur = tree.ParentOf(body_node);
    while (cur && cur != tree.Root()) {
        const EntityId e = cur->entity;
        if (idx.body.find(e) != idx.body.end()) break;  // ancestor body: stop
        segs.push_back(cur->name);
        cur = tree.ParentOf(cur);
    }
    std::string out;
    for (size_t i = segs.size(); i-- > 0;) {
        if (!out.empty()) out += '/';
        out += segs[i];
    }
    return out;
}

// Serialize one tree node (pre-order, recursive). Reads VALUES from the records
// the node maps to; structure/names/order from the tree.
Value SaveNode(const SceneIR& scene, const NodeIndex& idx, MeshSink& sink,
               const std::shared_ptr<SceneNode>& node) {
    const SceneGraph& tree = scene.Tree();
    const Registry& ecs = scene.Ecs();
    const EntityId e = node->entity;

    Value o = Value::Object();
    o.Set("name", Value::Str(node->name));

    // -- rigid_body (+ optional record_name when dedup diverged) ------------
    const auto bit = idx.body.find(e);
    if (bit != idx.body.end()) {
        const RigidBodyRecord& b = scene.GetBody(bit->second);
        // record_name only when the tree-derived suffix differs from the stored
        // record name (sibling-dedup case). Else the derived path reconstructs it.
        const std::string derived = DerivedBodyName(tree, idx, node);
        if (derived != b.name) {
            o.Set("record_name", Value::Str(b.name));
        }
        o.Set("transform", TransformJson(b.local_transform));
        o.Set("rigid_body", SaveRigidBody(b));
    }

    // -- collision_shape / visual_mesh -------------------------------------
    const auto sit = idx.shape.find(e);
    if (sit != idx.shape.end()) {
        const CollisionShapeRecord& s = scene.GetShape(sit->second);
        Value shape = SaveShape(s, sink);
        // The geom node name = StableAutoName("geom", id, record.name) = the
        // record name when non-empty (deduped), else "geom_<id>". Store the raw
        // record name explicitly when it differs from the node name (unnamed
        // shapes have an empty record name; dedup may rename) so it round-trips
        // EXACTLY. (ProjectShape re-derives "geom_<id>" from the same id, so an
        // empty record name reconstructs to the same node name.)
        if (s.name != node->name) {
            shape.Set("record_name", Value::Str(s.name));
        }
        // A non-colliding geom projects to a VisualMeshComponent (h1 fingers);
        // everything else to a CollisionShapeComponent. The full-fidelity record
        // is identical either way — only the key name reflects the projection so
        // the on-disk node mirrors the facade.
        if (ecs.Has<VisualMeshComponent>(e)) {
            o.Set("visual_mesh", std::move(shape));
        } else {
            o.Set("collision_shape", std::move(shape));
        }
    }

    // -- joint (on the child body node, or an auxiliary joint child node) ---
    const auto jit = idx.joint.find(e);
    if (jit != idx.joint.end()) {
        o.Set("joint", SaveJoint(scene.GetJoint(jit->second)));
    }

    // -- actuator (component on the joint-bearing entity) -------------------
    if (ecs.Has<ActuatorComponent>(e)) {
        for (ActuatorId i = 0; i < scene.ActuatorCount(); ++i) {
            const ActuatorRecord& a = scene.GetActuator(i);
            if (a.joint_id != kInvalidJoint && a.joint_id < scene.JointCount() &&
                scene.EntityOfJoint(a.joint_id) == e) {
                o.Set("actuator", SaveActuator(a));
                break;
            }
        }
    }

    // -- camera / light (own node under the attached body) ------------------
    if (ecs.Has<CameraComponent>(e)) {
        const CameraComponent* c = ecs.Get<CameraComponent>(e);
        Value cam = Value::Object();
        cam.Set("local", TransformJson(c->local_transform));
        cam.Set("vertical_fov_degrees", Value::Float(c->vertical_fov_degrees));
        cam.Set("near_clip", Value::Float(c->near_clip));
        cam.Set("far_clip", Value::Float(c->far_clip));
        o.Set("camera", std::move(cam));
    }
    if (ecs.Has<LightComponent>(e)) {
        const LightComponent* l = ecs.Get<LightComponent>(e);
        Value light = Value::Object();
        const char* tname = "point";
        switch (l->type) {
            case LightComponent::Type::Point:       tname = "point"; break;
            case LightComponent::Type::Directional: tname = "directional"; break;
            case LightComponent::Type::Spot:        tname = "spot"; break;
            case LightComponent::Type::Area:        tname = "area"; break;
        }
        light.Set("type", Value::Str(tname));
        light.Set("local", TransformJson(l->local_transform));
        light.Set("color", Vec3Json(l->color));
        light.Set("intensity", Value::Float(l->intensity));
        o.Set("light", std::move(light));
    }

    // -- children (sibling order == tree tail-append order) -----------------
    Value children = Value::Array();
    for (auto child = node->first_child; child; child = child->next_sibling) {
        children.PushBack(SaveNode(scene, idx, sink, child));
    }
    o.Set("children", std::move(children));
    return o;
}

}  // namespace

void Save(const SceneIR& scene, const std::string& nks_path) {
    namespace fs = std::filesystem;
    const fs::path p(nks_path);
    const std::string stem = p.stem().string();
    const std::string nka_name = stem + ".nka";
    const fs::path nka_path = p.parent_path() / nka_name;

    MeshSink sink;
    sink.nka_basename = nka_name;

    Value root = Value::Object();
    root.Set("nks_version", Value::Int(1));

    // -- split materials (keyed by name) ------------------------------------
    {
        Value phys = Value::Object();
        Value rend = Value::Object();
        for (const MaterialRecord& m : scene.Materials()) {
            phys.Set(m.name, SavePhysicsMaterial(m));
            rend.Set(m.name, SaveRenderMaterial(m));
        }
        root.Set("physics_materials", std::move(phys));
        root.Set("render_materials", std::move(rend));
    }

    // -- tree (SceneGraph pre-order, nested) --------------------------------
    {
        const NodeIndex idx = BuildNodeIndex(scene);
        const SceneGraph& tree = scene.Tree();
        Value nodes = Value::Array();
        for (auto child = tree.Root()->first_child; child; child = child->next_sibling) {
            nodes.PushBack(SaveNode(scene, idx, sink, child));
        }
        root.Set("tree", std::move(nodes));
    }

    // -- sensors (record-fidelity section: no tree home) --------------------
    {
        Value sensors = Value::Array();
        for (const SensorRecord& s : scene.Sensors()) sensors.PushBack(SaveSensor(s));
        root.Set("sensors", std::move(sensors));
    }

    // -- filters: exclude pairs + contact-pair overrides --------------------
    {
        Value excludes = Value::Array();
        for (const auto& e : scene.ExcludePairs()) {
            Value pr = Value::Array();
            pr.PushBack(Value::Int(static_cast<int64_t>(e.first)));
            pr.PushBack(Value::Int(static_cast<int64_t>(e.second)));
            excludes.PushBack(std::move(pr));
        }
        root.Set("exclude_pairs", std::move(excludes));
    }
    {
        Value pairs = Value::Array();
        for (const ContactPairOverride& cp : scene.ContactPairs()) {
            Value o = Value::Object();
            o.Set("geom1", Value::Int(static_cast<int64_t>(cp.geom1)));
            o.Set("geom2", Value::Int(static_cast<int64_t>(cp.geom2)));
            o.Set("condim", Value::Int(cp.condim));
            o.Set("friction_mu", Value::Float(cp.friction_mu));
            o.Set("solref", FloatArray(cp.solref, 2));
            o.Set("solimp", FloatArray(cp.solimp, 5));
            o.Set("margin", Value::Float(cp.margin));
            o.Set("gap", Value::Float(cp.gap));
            pairs.PushBack(std::move(o));
        }
        root.Set("contact_pairs", std::move(pairs));
    }

    // -- write .nks + sibling .nka ------------------------------------------
    const std::string text = root.Dump(2);
    std::ofstream out(nks_path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("nks: cannot open for write: " + nks_path);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!out) throw std::runtime_error("nks: write failed: " + nks_path);

    // Always write the sibling .nka (possibly empty: a valid 0-chunk container),
    // so a scene with no inline mesh geometry still round-trips byte-identically.
    sink.writer.Write(nka_path.string());
}

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------
namespace {

// Apply an "imports" list (resolved BEFORE the saved records, so import bodies
// land first; but Save never emits imports, so a Saved scene is flat). Each
// import loads by extension and Composes with prefix attach_at + "/".
SceneIR ApplyImports(SceneIR scene, const Value& imports,
                     const std::filesystem::path& base_dir);

void LoadInto(SceneIR& scene, const Value& root, const std::filesystem::path& base_dir,
              const std::string& nka_path) {
    // Optional .nka (mesh chunks). Open lazily on the first mesh reference.
    std::unique_ptr<NkaFile> nka;
    auto ensure_nka = [&]() -> NkaFile& {
        if (!nka) {
            nka = std::make_unique<NkaFile>(NkaFile::Open(nka_path));
        }
        return *nka;
    };

    // -- imports (compose first, deterministic) -----------------------------
    if (const Value* imports = root.Find("imports")) {
        scene = ApplyImports(std::move(scene), *imports, base_dir);
    }

    // -- split materials (keyed by name; physics + render sections) ---------
    // ProjectMaterial maps friction_mu -> static/dynamic friction equally, so
    // physics_materials[name].static_friction round-trips the record value. The
    // two sections share the same key set; iterate render_materials for the name
    // order (insertion-ordered JSON) and pull friction from physics_materials.
    {
        const Value* phys = root.Find("physics_materials");
        const Value* rend = root.Find("render_materials");
        if (rend) {
            for (const auto& kv : rend->Items()) {
                const Value& rm = kv.second;
                MaterialRecord rec;
                rec.name = kv.first;
                const Value& bc = rm.At("base_color");
                rec.base_color = math::Vec3{FloatAt(bc, 0, "base_color"),
                                            FloatAt(bc, 1, "base_color"),
                                            FloatAt(bc, 2, "base_color")};
                rec.alpha = FloatAt(bc, 3, "base_color");
                rec.metallic = rm.At("metallic").AsFloat();
                rec.roughness = rm.At("roughness").AsFloat();
                if (phys) {
                    if (const Value* pm = phys->Find(kv.first)) {
                        rec.friction_mu = pm->At("static_friction").AsFloat();
                    }
                }
                scene.AddMaterial(std::move(rec));
            }
        }
    }

    // -- tree: reconstruct records in their ORIGINAL dense-id order. The .nks
    //    tree groups a body's child BODIES before the body's own geom nodes (the
    //    facade projection order), whereas the importers add a body then its OWN
    //    shapes/joints before descending. So a single pre-order pass would NOT
    //    reproduce the shape/joint id order. Instead we: (1) collect body nodes
    //    in pre-order (== body-id order for importer/compose scenes — parents
    //    precede children, sibling order is tail-append), reconstructing each
    //    body's DERIVED name; then (2) walk bodies in that id order and emit each
    //    body's OWN direct geom / joint / camera / light / actuator children.
    //    That body-grouped emission reproduces the import dense-id order exactly,
    //    which the cook-byte-equality + second-save gates enforce.
    const Value* tree = root.Find("tree");
    if (tree && tree->IsArray()) {
        // A body node + the resolved BodyId Phase 1 assigned it (id order).
        struct BodyNode {
            const Value* node;
            BodyId       id;
        };
        std::vector<BodyNode> body_nodes;

        // -- Phase 1: bodies (pre-order). The record name is "record_name" when
        //    present (sibling-dedup case), else the DERIVED suffix: the group-
        //    prefix segments accumulated since the last body ancestor, plus this
        //    node's single-segment name. `prefix` threads that suffix down.
        std::function<void(const Value&, const std::string&)> walk_bodies =
            [&](const Value& node, const std::string& prefix) {
                const std::string seg = node.At("name").AsString();
                const Value* rb = node.Find("rigid_body");
                std::string child_prefix;
                if (rb) {
                    RigidBodyRecord rec;
                    if (const Value* rn = node.Find("record_name")) {
                        rec.name = rn->AsString();
                    } else {
                        rec.name = prefix.empty() ? seg : (prefix + seg);
                    }
                    rec.parent_id = static_cast<BodyId>(rb->At("parent_id").AsInt());
                    rec.local_transform = TransformFromJson(node.At("transform"));
                    rec.inertial_transform = TransformFromJson(rb->At("inertial"));
                    rec.mass = rb->At("mass").AsFloat();
                    rec.inertia = Vec3FromJson(rb->At("inertia"));
                    rec.is_static = rb->At("is_static").AsBool();
                    const BodyId id = scene.AddRigidBody(std::move(rec));
                    body_nodes.push_back(BodyNode{&node, id});
                    // A body resets the group prefix for ITS subtree (its node is
                    // the body node; descendant bodies are placed under it).
                    child_prefix.clear();
                } else {
                    // A group node contributes its segment to the prefix of the
                    // body nodes beneath it (until the next body resets it).
                    child_prefix = prefix.empty() ? (seg + "/") : (prefix + seg + "/");
                }
                if (const Value* kids = node.Find("children")) {
                    for (const Value& c : kids->Elements()) {
                        walk_bodies(c, child_prefix);
                    }
                }
            };
        for (const Value& top : tree->Elements()) {
            walk_bodies(top, std::string());
        }

        // -- Phase 2: shapes. Walk bodies in id order; for each, emit its OWN
        //    direct geom children (sibling order) -> reproduces shape-id order.
        auto load_shape = [&](const Value& node, BodyId body_id) {
            const Value* s = node.Find("collision_shape");
            if (!s) s = node.Find("visual_mesh");
            if (!s) return;
            CollisionShapeRecord rec;
            if (const Value* rn = s->Find("record_name")) {
                rec.name = rn->AsString();
            } else {
                rec.name = node.At("name").AsString();
            }
            rec.body_id = body_id;
            rec.material_id = static_cast<MaterialId>(s->At("material_id").AsInt());
            rec.type = ShapeTypeFromName(s->At("type").AsString());
            rec.local_transform = TransformFromJson(s->At("local"));
            rec.half_extents = Vec3FromJson(s->At("half_extents"));
            rec.radius = s->At("radius").AsFloat();
            rec.half_height = s->At("half_height").AsFloat();
            rec.contype = static_cast<uint32_t>(s->At("contype").AsInt());
            rec.conaffinity = static_cast<uint32_t>(s->At("conaffinity").AsInt());
            rec.collision_group = static_cast<int32_t>(s->At("collision_group").AsInt());
            rec.solref[0] = FloatAt(s->At("solref"), 0, "solref");
            rec.solref[1] = FloatAt(s->At("solref"), 1, "solref");
            for (int k = 0; k < 5; ++k) rec.solimp[k] = FloatAt(s->At("solimp"), k, "solimp");
            rec.friction_mu = s->At("friction_mu").AsFloat();
            rec.priority = static_cast<int32_t>(s->At("priority").AsInt());
            rec.solmix = s->At("solmix").AsFloat();
            rec.margin = s->At("margin").AsFloat();
            rec.gap = s->At("gap").AsFloat();
            rec.condim = static_cast<uint8_t>(s->At("condim").AsInt());
            rec.decompose_mode = DecomposeModeFromName(s->At("decompose_mode").AsString());
            rec.decompose_max_pieces =
                static_cast<uint32_t>(s->At("decompose_max_pieces").AsInt());
            if (const Value* mesh = s->Find("mesh")) {
                const AssetRef ref = ParseAssetRef(mesh->AsString());
                const std::vector<uint8_t> bytes =
                    ensure_nka().LoadChunk(ref.fourcc, ref.index);
                DecodeCollisionMesh(bytes, rec.mesh_vertices, rec.mesh_indices);
            }
            scene.AddCollisionShape(std::move(rec));
        };
        // Direct children of root that are geoms (orphan shapes: no body) come
        // first, mirroring how an orphan shape projects under root.
        auto direct_children = [](const Value& node) -> const std::vector<Value>* {
            const Value* kids = node.Find("children");
            return kids ? &kids->Elements() : nullptr;
        };
        for (const BodyNode& bn : body_nodes) {
            if (const auto* kids = direct_children(*bn.node)) {
                for (const Value& c : *kids) load_shape(c, bn.id);
            }
        }

        // -- Phase 3: joints. A joint rides on its CHILD body node (a "joint"
        //    component) or on an auxiliary joint-only child node (multi-joint
        //    body). Walk bodies in id order; emit the body's own joint, then any
        //    auxiliary joint-only direct children -> reproduces joint-id order.
        auto load_joint = [&](const Value* j) {
            JointRecord rec;
            rec.name = j->At("name").AsString();
            rec.type = JointTypeFromName(j->At("type").AsString());
            rec.parent_body = static_cast<BodyId>(j->At("parent_body").AsInt());
            rec.child_body = static_cast<BodyId>(j->At("child_body").AsInt());
            rec.axis = Vec3FromJson(j->At("axis"));
            rec.parent_frame = TransformFromJson(j->At("parent_frame"));
            rec.child_frame = TransformFromJson(j->At("child_frame"));
            rec.lower_limit = j->At("lower_limit").AsFloat();
            rec.upper_limit = j->At("upper_limit").AsFloat();
            rec.damping = j->At("damping").AsFloat();
            rec.armature = j->At("armature").AsFloat();
            rec.stiffness = j->At("stiffness").AsFloat();
            rec.initial_position = j->At("initial_position").AsFloat();
            scene.AddJoint(std::move(rec));
        };
        for (const BodyNode& bn : body_nodes) {
            if (const Value* j = bn.node->Find("joint")) load_joint(j);
            if (const auto* kids = direct_children(*bn.node)) {
                for (const Value& c : *kids) {
                    // Auxiliary joint nodes carry a "joint" but no "rigid_body".
                    if (!c.Find("rigid_body")) {
                        if (const Value* j = c.Find("joint")) load_joint(j);
                    }
                }
            }
        }

        // -- Phase 4: cameras + lights (own direct child node under the attached
        //    body). attached_body = the owning body id; name = node name. -----
        for (const BodyNode& bn : body_nodes) {
            if (const auto* kids = direct_children(*bn.node)) {
                for (const Value& c : *kids) {
                    if (const Value* cam = c.Find("camera")) {
                        CameraRecord rec;
                        rec.name = c.At("name").AsString();
                        rec.attached_body = bn.id;
                        rec.local_transform = TransformFromJson(cam->At("local"));
                        rec.vertical_fov_degrees = cam->At("vertical_fov_degrees").AsFloat();
                        rec.near_clip = cam->At("near_clip").AsFloat();
                        rec.far_clip = cam->At("far_clip").AsFloat();
                        scene.AddCamera(std::move(rec));
                    }
                }
            }
        }
        for (const BodyNode& bn : body_nodes) {
            if (const auto* kids = direct_children(*bn.node)) {
                for (const Value& c : *kids) {
                    if (const Value* lt = c.Find("light")) {
                        LightRecord rec;
                        rec.name = c.At("name").AsString();
                        rec.type = LightTypeFromName(lt->At("type").AsString());
                        rec.attached_body = bn.id;
                        rec.local_transform = TransformFromJson(lt->At("local"));
                        rec.color = Vec3FromJson(lt->At("color"));
                        rec.intensity = lt->At("intensity").AsFloat();
                        scene.AddLight(std::move(rec));
                    }
                }
            }
        }
        // Root-level (unattached) cameras / lights project under root.
        for (const Value& top : tree->Elements()) {
            if (top.Find("rigid_body")) continue;  // body subtrees handled above
            if (const Value* cam = top.Find("camera")) {
                CameraRecord rec;
                rec.name = top.At("name").AsString();
                rec.local_transform = TransformFromJson(cam->At("local"));
                rec.vertical_fov_degrees = cam->At("vertical_fov_degrees").AsFloat();
                rec.near_clip = cam->At("near_clip").AsFloat();
                rec.far_clip = cam->At("far_clip").AsFloat();
                scene.AddCamera(std::move(rec));
            }
            if (const Value* lt = top.Find("light")) {
                LightRecord rec;
                rec.name = top.At("name").AsString();
                rec.type = LightTypeFromName(lt->At("type").AsString());
                rec.local_transform = TransformFromJson(lt->At("local"));
                rec.color = Vec3FromJson(lt->At("color"));
                rec.intensity = lt->At("intensity").AsFloat();
                scene.AddLight(std::move(rec));
            }
        }

        // -- Phase 5: actuators (a component beside the joint it drives). Walk
        //    bodies in id order; emit the body's own actuator, then auxiliary
        //    joint-node actuators -> reproduces actuator-id order.
        auto load_actuator = [&](const Value* a) {
            ActuatorRecord rec;
            rec.name = a->At("name").AsString();
            rec.type = ActuatorTypeFromName(a->At("type").AsString());
            rec.joint_id = static_cast<JointId>(a->At("joint_id").AsInt());
            rec.gain = a->At("gain").AsFloat();
            rec.force_limit = a->At("force_limit").AsFloat();
            scene.AddActuator(std::move(rec));
        };
        for (const BodyNode& bn : body_nodes) {
            if (const Value* a = bn.node->Find("actuator")) load_actuator(a);
            if (const auto* kids = direct_children(*bn.node)) {
                for (const Value& c : *kids) {
                    if (!c.Find("rigid_body")) {
                        if (const Value* a = c.Find("actuator")) load_actuator(a);
                    }
                }
            }
        }
    }

    // -- sensors ------------------------------------------------------------
    if (const Value* sensors = root.Find("sensors")) {
        for (const Value& s : sensors->Elements()) {
            SensorRecord rec;
            rec.name = s.At("name").AsString();
            rec.type = SensorTypeFromName(s.At("type").AsString());
            rec.attached_body = static_cast<BodyId>(s.At("attached_body").AsInt());
            rec.local_transform = TransformFromJson(s.At("local"));
            rec.sample_rate_hz = s.At("sample_rate_hz").AsFloat();
            scene.AddSensor(std::move(rec));
        }
    }

    // -- cameras ------------------------------------------------------------
    if (const Value* cameras = root.Find("cameras")) {
        for (const Value& c : cameras->Elements()) {
            CameraRecord rec;
            rec.name = c.At("name").AsString();
            rec.attached_body = static_cast<BodyId>(c.At("attached_body").AsInt());
            rec.local_transform = TransformFromJson(c.At("local"));
            rec.vertical_fov_degrees = c.At("vertical_fov_degrees").AsFloat();
            rec.near_clip = c.At("near_clip").AsFloat();
            rec.far_clip = c.At("far_clip").AsFloat();
            scene.AddCamera(std::move(rec));
        }
    }

    // -- lights -------------------------------------------------------------
    if (const Value* lights = root.Find("lights")) {
        for (const Value& l : lights->Elements()) {
            LightRecord rec;
            rec.name = l.At("name").AsString();
            rec.type = LightTypeFromName(l.At("type").AsString());
            rec.attached_body = static_cast<BodyId>(l.At("attached_body").AsInt());
            rec.local_transform = TransformFromJson(l.At("local"));
            rec.color = Vec3FromJson(l.At("color"));
            rec.intensity = l.At("intensity").AsFloat();
            scene.AddLight(std::move(rec));
        }
    }

    // -- filters ------------------------------------------------------------
    if (const Value* excludes = root.Find("exclude_pairs")) {
        for (const Value& e : excludes->Elements()) {
            scene.AddExcludePair(static_cast<BodyId>(FloatAt(e, 0, "exclude_pair")),
                                 static_cast<BodyId>(FloatAt(e, 1, "exclude_pair")));
        }
    }
    if (const Value* pairs = root.Find("contact_pairs")) {
        for (const Value& cp : pairs->Elements()) {
            ContactPairOverride ov;
            ov.geom1 = static_cast<ShapeId>(cp.At("geom1").AsInt());
            ov.geom2 = static_cast<ShapeId>(cp.At("geom2").AsInt());
            ov.condim = static_cast<uint8_t>(cp.At("condim").AsInt());
            ov.friction_mu = cp.At("friction_mu").AsFloat();
            ov.solref[0] = FloatAt(cp.At("solref"), 0, "solref");
            ov.solref[1] = FloatAt(cp.At("solref"), 1, "solref");
            for (int k = 0; k < 5; ++k) ov.solimp[k] = FloatAt(cp.At("solimp"), k, "solimp");
            ov.margin = cp.At("margin").AsFloat();
            ov.gap = cp.At("gap").AsFloat();
            scene.AddContactPair(ov);
        }
    }
}

SceneIR ApplyImports(SceneIR scene, const Value& imports,
                     const std::filesystem::path& base_dir) {
    namespace fs = std::filesystem;
    const bool cache_on = [] {
        const char* env = std::getenv("NUKA_IMPORT_CACHE");
        return env && std::string(env) == "1";
    }();
    (void)cache_on;  // Import lazy-cache hook: default OFF (de-risk). When enabled
                     // the source-hash -> .nuka_cache mapping would memoize the
                     // imported SceneIR; left as a documented seam for M-later.

    for (const Value& imp : imports.Elements()) {
        std::string file = imp.At("file").AsString();
        const fs::path src = base_dir / file;
        const std::string ext = src.extension().string();

        SceneIR addon;
        if (ext == ".xml") {
            addon = nuka::import::LoadMjcf(src.string());
        } else if (ext == ".usda" || ext == ".usd" || ext == ".usdc") {
            addon = nuka::import::LoadUsd(src.string());
        } else if (ext == ".urdf") {
            addon = nuka::import::LoadUrdf(src.string());
        } else {
            throw std::runtime_error("nks import: unsupported extension: " + ext);
        }

        math::Transform placement = math::Transform::Identity();
        if (const Value* t = imp.Find("transform")) {
            placement = TransformFromJson(*t);
        }
        std::string prefix;
        if (const Value* at = imp.Find("attach_at")) {
            prefix = at->AsString() + "/";
        }
        scene = Compose(scene, addon, placement, prefix);
    }
    return scene;
}

// -- override application ----------------------------------------------------
// Apply a {"overrides": {"<derived/path>": {<partial component objects>}}}
// overlay to an already-loaded scene. The override key is the body's DERIVED
// tree path (parent-body chain + group prefix + record name), which under §3.7
// is the addressing scheme the `tree` section uses — NOT the (now suffix-only)
// record name. Patches the matched record fields in place.
void ApplyOverrides(SceneIR& scene, const Value& overlay) {
    const Value* overrides = overlay.Find("overrides");
    if (!overrides) return;

    // Build derived-path -> body id by walking each body's projected node up to
    // the root (SceneGraph::PathOf). For a root body the derived path == its
    // record name, so a root-keyed overlay keeps working unchanged.
    std::unordered_map<std::string, BodyId> body_by_path;
    const SceneGraph& tree = scene.Tree();
    for (BodyId i = 0; i < scene.RigidBodyCount(); ++i) {
        const EntityId e = scene.EntityOfBody(i);
        if (e == kInvalidEntity) continue;
        const auto node = scene.Ecs().NodeOf(e);
        if (!node) continue;
        body_by_path.emplace(tree.PathOf(node), i);
    }

    for (const auto& kv : overrides->Items()) {
        const std::string& path = kv.first;
        const Value& patch = kv.second;

        const auto bit = body_by_path.find(path);
        if (bit != body_by_path.end()) {
            RigidBodyRecord& b = scene.GetBodyMut(bit->second);
            if (const Value* tr = patch.Find("transform")) {
                if (const Value* pos = tr->Find("pos")) b.local_transform.position = Vec3FromJson(*pos);
                if (const Value* q = tr->Find("quat")) b.local_transform.rotation = QuatFromJson(*q);
            }
            if (const Value* rb = patch.Find("rigid_body")) {
                if (const Value* mass = rb->Find("mass")) b.mass = mass->AsFloat();
                if (const Value* inertia = rb->Find("inertia")) b.inertia = Vec3FromJson(*inertia);
                if (const Value* st = rb->Find("is_static")) b.is_static = st->AsBool();
            }
        }

        // material override (key matches a material name)
        for (MaterialId i = 0; i < scene.MaterialCount(); ++i) {
            if (scene.GetMaterial(i).name != path) continue;
            MaterialRecord& m = scene.GetMaterialMut(i);
            if (const Value* mat = patch.Find("material")) {
                if (const Value* bc = mat->Find("base_color")) m.base_color = Vec3FromJson(*bc);
                if (const Value* a = mat->Find("alpha")) m.alpha = a->AsFloat();
                if (const Value* r = mat->Find("roughness")) m.roughness = r->AsFloat();
                if (const Value* me = mat->Find("metallic")) m.metallic = me->AsFloat();
                if (const Value* fr = mat->Find("friction_mu")) m.friction_mu = fr->AsFloat();
            }
        }
    }
    // Mutating records via Get*Mut marks the facade dirty; the next facade read
    // (Tree/Ecs/EntityOf*) lazily re-projects, so no manual rebuild is needed.
}

std::string ReadTextFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("nks: cannot open: " + path);
    return std::string((std::istreambuf_iterator<char>(in)),
                       std::istreambuf_iterator<char>());
}

}  // namespace

SceneIR Load(const std::string& nks_path) {
    namespace fs = std::filesystem;
    const fs::path p(nks_path);
    const std::string stem = p.stem().string();
    const fs::path nka_path = p.parent_path() / (stem + ".nka");

    const Value root = Value::Parse(ReadTextFile(nks_path));

    SceneIR scene;
    LoadInto(scene, root, p.parent_path(), nka_path.string());
    return scene;
}

SceneIR Load(const std::string& base_path, const std::string& overlay_path) {
    SceneIR scene = Load(base_path);
    const Value overlay = Value::Parse(ReadTextFile(overlay_path));
    ApplyOverrides(scene, overlay);
    return scene;
}

} // namespace nuka::scene::nks
