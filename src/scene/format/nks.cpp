// ---------------------------------------------------------------------------
// nuka::scene::nks - .nks scene Save/Load implementation (M2c, §3.7 tree form).
// ---------------------------------------------------------------------------
// HOST-ONLY. The .nks JSON shape follows the owner-amended spec §3.7:
//
//   { "nks_version": 1,
//     "physics_materials": { "<name>": {...} },   // keyed by material name
//     "render_materials":  { "<name>": {...} },   // keyed by material name
//     "tree": [ <node>, ... ],                    // SceneGraph pre-order, nested
//     "sensors": [...], "exclude_pairs": [...], "contact_pairs": [...],
//     "media": [ <media>, ... ] }                  // cloth/soft-tet/fluid records
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
        case SensorType::Contact:     return "contact";
        case SensorType::ForceTorque: return "force_torque";
        case SensorType::FramePose:   return "frame_pose";
        case SensorType::JointState:  return "joint_state";
        case SensorType::Camera:      return "camera";
        case SensorType::Depth:       return "depth";
        case SensorType::Lidar:       return "lidar";
        case SensorType::RangeScan:   return "range_scan";
    }
    return "imu";
}
SensorType SensorTypeFromName(const std::string& s) {
    if (s == "imu") return SensorType::Imu;
    if (s == "contact") return SensorType::Contact;
    if (s == "force_torque") return SensorType::ForceTorque;
    if (s == "frame_pose") return SensorType::FramePose;
    if (s == "joint_state") return SensorType::JointState;
    if (s == "camera") return SensorType::Camera;
    if (s == "depth") return SensorType::Depth;
    if (s == "lidar") return SensorType::Lidar;
    if (s == "range_scan") return SensorType::RangeScan;
    return SensorType::Imu;
}

const char* MountFrameName(MountFrame m) {
    switch (m) {
        case MountFrame::Link: return "link";
        case MountFrame::Body: return "body";
        case MountFrame::Base: return "base";
    }
    return "body";
}
MountFrame MountFrameFromName(const std::string& s) {
    if (s == "link") return MountFrame::Link;
    if (s == "body") return MountFrame::Body;
    if (s == "base") return MountFrame::Base;
    return MountFrame::Body;
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

const char* MediaKindName(MediaRecord::Kind k) {
    switch (k) {
        case MediaRecord::Kind::Cloth:   return "cloth";
        case MediaRecord::Kind::SoftTet: return "soft_tet";
        case MediaRecord::Kind::Fluid:   return "fluid";
    }
    return "cloth";
}
MediaRecord::Kind MediaKindFromName(const std::string& s) {
    if (s == "cloth") return MediaRecord::Kind::Cloth;
    if (s == "soft_tet") return MediaRecord::Kind::SoftTet;
    if (s == "fluid") return MediaRecord::Kind::Fluid;
    throw std::runtime_error("nks: unknown media kind '" + s + "'");
}

const char* MediaMethodName(MediaRecord::Method m) {
    switch (m) {
        case MediaRecord::Method::Xpbd:   return "xpbd";
        case MediaRecord::Method::Pbf:    return "pbf";
        case MediaRecord::Method::MlsMpm: return "mlsmpm";
    }
    return "xpbd";
}
MediaRecord::Method MediaMethodFromName(const std::string& s) {
    if (s == "xpbd") return MediaRecord::Method::Xpbd;
    if (s == "pbf") return MediaRecord::Method::Pbf;
    if (s == "mlsmpm") return MediaRecord::Method::MlsMpm;
    throw std::runtime_error("nks: unknown media method '" + s + "'");
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

// An AssetRef serializes as its "<nka>#TAG/idx" text (empty string for an empty
// ref); the .nka chunk bytes a media baked-mesh ref points at are not read here.
Value AssetRefJson(const AssetRef& r) { return Value::Str(ToString(r)); }
AssetRef AssetRefFromJson(const Value& v) { return ParseAssetRef(v.AsString()); }

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

    // Append a VISUAL mesh (MESH) deduped by content hash; return AssetRef text.
    // M8.5 T5: a non-colliding geom's triangles are routed here (vs CMSH) so the
    // render consumer (render_world.cpp's NkaTagMesh() branch) decodes real
    // geometry. EncodeMesh carries positions + normal/uv streams; the decoded
    // positions/indices round-trip the source triangles byte-exactly
    // (SceneRoundtrip ExpectShapeRecordsEqual still holds). Keyed in the SAME
    // by_hash map as CMSH/SAMP -- the encoded payloads differ by fourcc/layout so
    // there is no cross-family hash collision (a MESH payload is never equal to a
    // CMSH payload of the same triangles: MESH interleaves the normal/uv streams).
    //
    // NORMALS: this sink threads authored normals when the record carries them.
    // An EMPTY `normals` argument writes a zero-filled normal stream (the render
    // side synthesizes normals from positions at load); a NON-EMPTY argument
    // round-trips the authored normals verbatim. The argument defaults to empty
    // so a record with no normals produces byte-identical .nka output.
    // CASCADE FLAG: SaveShape now passes CollisionShapeRecord.mesh_normals here and
    // LoadObj parses `vn`, but the importer copy MeshGeometry.normals ->
    // record.mesh_normals lives in mjcf_importer.cpp (not owned) and is still
    // missing -- until that line lands, mesh_normals is empty and the .nka bytes
    // are byte-identical. Once wired, a non-empty stream CHANGES the cooked .nka
    // (D1 visual-mesh goldens move; owner regenerates).
    std::string AddVisualMesh(const std::vector<float>& verts,
                              const std::vector<uint32_t>& indices,
                              const std::vector<float>& normals = {}) {
        NkaMesh mesh;
        mesh.positions = verts;
        mesh.normals = normals;
        mesh.indices = indices;
        const std::vector<uint8_t> payload = EncodeMesh(mesh);
        const uint64_t hash = NkaContentHash(payload);
        const auto it = by_hash.find(hash);
        if (it != by_hash.end()) return it->second;
        const uint32_t index = writer.AddChunk(NkaTagMesh(), payload);
        AssetRef ref;
        ref.nka_path = nka_basename;
        ref.fourcc = NkaTagMesh();
        ref.index = index;
        const std::string text = ToString(ref);
        by_hash.emplace(hash, text);
        return text;
    }

};

// Serialize a collision/visual shape RECORD into the per-node component object
// (full cook fidelity: every legacy field stays here verbatim). The node `name`
// is the geom node's tree name and is set by the caller, not here.
// `is_visual` mirrors the facade projection (the node carries a
// VisualMeshComponent vs a CollisionShapeComponent): a VISUAL geom's triangles
// route to a .nka MESH chunk (M8.5 T5 visual-mesh cook), a colliding geom's to
// CMSH -- ALL OTHER fields, including the "mesh" AssetRef key name, are written
// identically, so a collision shape's on-disk bytes are UNCHANGED by this.
Value SaveShape(const CollisionShapeRecord& s, MeshSink& sink, bool is_visual) {
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
    // Inline mesh geometry -> .nka MESH (visual-only) / CMSH (colliding) chunk,
    // deduped; store the AssetRef text under the same "mesh" key. Routing matches
    // the facade projection so a non-colliding geom (the h1/go2 visual meshes)
    // becomes a render-decodable MESH chunk, while every colliding mesh's CMSH
    // routing -- and its on-disk bytes -- stay exactly as before.
    if (!s.mesh_vertices.empty() && !s.mesh_indices.empty()) {
        o.Set("mesh", Value::Str(is_visual
                                     ? sink.AddVisualMesh(s.mesh_vertices, s.mesh_indices,
                                                          s.mesh_normals)
                                     : sink.AddCollisionMesh(s.mesh_vertices, s.mesh_indices)));
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

// Legacy keys (attached_body=mount_index, local) plus additive mount/camera/lidar
// payloads emitted only when non-default, so an Imu sensor's bytes are unchanged.
Value SaveSensor(const SensorDesc& s) {
    Value o = Value::Object();
    o.Set("name", Value::Str(s.name));
    o.Set("type", Value::Str(SensorTypeName(s.type)));
    o.Set("attached_body", Value::Int(static_cast<int64_t>(s.mount_index)));
    o.Set("local", TransformJson(s.local_offset));
    o.Set("sample_rate_hz", Value::Float(s.sample_rate_hz));
    if (s.mount != MountFrame::Body) {
        o.Set("mount", Value::Str(MountFrameName(s.mount)));
    }
    if (s.update_period != 1u) {
        o.Set("update_period", Value::Int(static_cast<int64_t>(s.update_period)));
    }
    if (s.aov_mask != 0u) {
        o.Set("aov_mask", Value::Int(static_cast<int64_t>(s.aov_mask)));
    }
    if (s.type == SensorType::Camera || s.type == SensorType::Depth) {
        Value c = Value::Object();
        c.Set("width", Value::Int(s.cam.width));
        c.Set("height", Value::Int(s.cam.height));
        c.Set("vfov_degrees", Value::Float(s.cam.vfov_degrees));
        c.Set("near_clip", Value::Float(s.cam.near_clip));
        c.Set("far_clip", Value::Float(s.cam.far_clip));
        c.Set("distortion", Value::Int(s.cam.distortion));
        c.Set("k1", Value::Float(s.cam.k1));
        c.Set("k2", Value::Float(s.cam.k2));
        o.Set("camera", std::move(c));
    }
    if (s.type == SensorType::Lidar || s.type == SensorType::RangeScan) {
        Value l = Value::Object();
        l.Set("az_count", Value::Int(s.lidar.az_count));
        l.Set("el_count", Value::Int(s.lidar.el_count));
        l.Set("az_min", Value::Float(s.lidar.az_min));
        l.Set("az_max", Value::Float(s.lidar.az_max));
        l.Set("el_min", Value::Float(s.lidar.el_min));
        l.Set("el_max", Value::Float(s.lidar.el_max));
        l.Set("min_range", Value::Float(s.lidar.min_range));
        l.Set("max_range", Value::Float(s.lidar.max_range));
        o.Set("lidar", std::move(l));
    }
    return o;
}

// A media entry (cloth / soft-tet / fluid). Every block is written verbatim (like
// SaveShape) so any MediaRecord round-trips field-for-field whatever populated it.
Value SaveMedia(const MediaRecord& m) {
    Value o = Value::Object();
    o.Set("name", Value::Str(m.name));
    o.Set("kind", Value::Str(MediaKindName(m.kind)));
    o.Set("method", Value::Str(MediaMethodName(m.method)));
    o.Set("render_material_id",
          Value::Int(static_cast<int64_t>(m.render_material_id)));
    o.Set("baked", AssetRefJson(m.baked));

    Value cg = Value::Object();
    cg.Set("nx", Value::Int(m.cloth_grid.nx));
    cg.Set("ny", Value::Int(m.cloth_grid.ny));
    cg.Set("spacing", Value::Float(m.cloth_grid.spacing));
    cg.Set("origin", Vec3Json(m.cloth_grid.origin));
    cg.Set("free", Value::Bool(m.cloth_grid.free));
    o.Set("cloth_grid", std::move(cg));

    Value ts = Value::Object();
    ts.Set("center", Vec3Json(m.tet_sphere.center));
    ts.Set("radius", Value::Float(m.tet_sphere.radius));
    ts.Set("cells", Value::Int(m.tet_sphere.cells));
    ts.Set("cell_len", Value::Float(m.tet_sphere.cell_len));
    o.Set("tet_sphere", std::move(ts));

    Value fb = Value::Object();
    fb.Set("min", Vec3Json(m.fluid_box.min));
    fb.Set("max", Vec3Json(m.fluid_box.max));
    fb.Set("spacing", Value::Float(m.fluid_box.spacing));
    o.Set("fluid_box", std::move(fb));

    Value xp = Value::Object();
    xp.Set("particle_mass", Value::Float(m.xpbd.particle_mass));
    xp.Set("friction", Value::Float(m.xpbd.friction));
    xp.Set("distance_alpha", Value::Float(m.xpbd.distance_alpha));
    xp.Set("bend_alpha", Value::Float(m.xpbd.bend_alpha));
    xp.Set("volume_alpha", Value::Float(m.xpbd.volume_alpha));
    xp.Set("iters", Value::Int(m.xpbd.iters));
    xp.Set("aero_drag_normal", Value::Float(m.xpbd.aero_drag_normal));
    xp.Set("aero_drag_tangent", Value::Float(m.xpbd.aero_drag_tangent));
    xp.Set("aero_drag_max_dv", Value::Float(m.xpbd.aero_drag_max_dv));
    o.Set("xpbd", std::move(xp));

    Value pb = Value::Object();
    pb.Set("rest_density", Value::Float(m.pbf.rest_density));
    pb.Set("support_scale", Value::Float(m.pbf.support_scale));
    pb.Set("iters", Value::Int(m.pbf.iters));
    pb.Set("friction", Value::Float(m.pbf.friction));
    pb.Set("clamp_overdensity", Value::Bool(m.pbf.clamp_overdensity));
    pb.Set("walls_enabled", Value::Bool(m.pbf.walls_enabled));
    pb.Set("walls_min", Vec3Json(m.pbf.walls_min));
    pb.Set("walls_max", Vec3Json(m.pbf.walls_max));
    pb.Set("floor_z", Value::Float(m.pbf.floor_z));
    pb.Set("boundary_layers", Value::Int(m.pbf.boundary_layers));
    o.Set("pbf", std::move(pb));

    Value mp = Value::Object();
    mp.Set("youngs", Value::Float(m.mpm.youngs));
    mp.Set("poisson", Value::Float(m.mpm.poisson));
    mp.Set("density", Value::Float(m.mpm.density));
    mp.Set("dp_friction", Value::Float(m.mpm.dp_friction));
    mp.Set("dp_cohesion", Value::Float(m.mpm.dp_cohesion));
    mp.Set("model_kind", Value::Float(m.mpm.model_kind));
    mp.Set("bulk_modulus", Value::Float(m.mpm.bulk_modulus));
    mp.Set("tait_gamma", Value::Float(m.mpm.tait_gamma));
    mp.Set("viscosity", Value::Float(m.mpm.viscosity));
    mp.Set("dx", Value::Float(m.mpm.dx));
    mp.Set("substeps", Value::Int(m.mpm.substeps));
    mp.Set("floor_normal", Vec3Json(m.mpm.floor_normal));
    mp.Set("floor_d", Value::Float(m.mpm.floor_d));
    mp.Set("floor_friction", Value::Float(m.mpm.floor_friction));
    o.Set("mpm", std::move(mp));

    Value rs = Value::Object();
    rs.Set("normal_offset", Value::Float(m.render_skin.normal_offset));
    rs.Set("smooth_iters", Value::Int(m.render_skin.smooth_iters));
    rs.Set("smooth_lambda", Value::Float(m.render_skin.smooth_lambda));
    rs.Set("skin_mesh", AssetRefJson(m.render_skin.skin_mesh));
    o.Set("render_skin", std::move(rs));
    return o;
}

// A terrain field. Every parametric amplitude is written verbatim so any
// TerrainRecord round-trips field-for-field (the engine has no terrain types).
Value SaveTerrain(const TerrainRecord& t) {
    Value o = Value::Object();
    o.Set("name", Value::Str(t.name));
    o.Set("nrow", Value::Int(t.nrow));
    o.Set("ncol", Value::Int(t.ncol));
    o.Set("cell", Value::Float(t.cell));
    o.Set("origin", Vec3Json(t.origin));
    o.Set("base_z", Value::Float(t.base_z));
    o.Set("grade_x", Value::Float(t.grade_x));
    o.Set("grade_y", Value::Float(t.grade_y));
    o.Set("ring_rise", Value::Float(t.ring_rise));
    o.Set("ring_width", Value::Float(t.ring_width));
    o.Set("ring_platform", Value::Float(t.ring_platform));
    o.Set("ring_count", Value::Int(t.ring_count));
    o.Set("bump_height", Value::Float(t.bump_height));
    o.Set("bump_cell", Value::Float(t.bump_cell));
    o.Set("feature_cell", Value::Float(t.feature_cell));
    o.Set("feature_margin", Value::Float(t.feature_margin));
    o.Set("feature_seed", Value::Int(t.feature_seed));
    o.Set("curric_levels", Value::Int(t.curric_levels));
    o.Set("curric_types", Value::Int(t.curric_types));
    o.Set("image_path", Value::Str(t.image_path));
    o.Set("image_radius_x", Value::Float(t.image_radius_x));
    o.Set("image_radius_y", Value::Float(t.image_radius_y));
    o.Set("image_elevation_z", Value::Float(t.image_elevation_z));
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

// ---------------------------------------------------------------------------
// M7 metadata sections: initial_state / settle (authored, not records).
// All emitted in a DETERMINISTIC order (std::map keys are sorted; vector order
// is the authored order) so a second Save is byte-identical.
// ---------------------------------------------------------------------------

// initial_state: { "<node-path>": { "qpos":[...], "root":{pos,quat} }, ... }.
// SceneInitialState is a std::map, so Items()-style iteration is key-sorted.
Value SaveInitialState(const SceneInitialState& is) {
    Value o = Value::Object();
    for (const auto& kv : is) {
        Value e = Value::Object();
        Value q = Value::Array();
        for (float v : kv.second.qpos) q.PushBack(Value::Float(v));
        e.Set("qpos", std::move(q));
        e.Set("root", TransformJson(kv.second.root));
        o.Set(kv.first, std::move(e));
    }
    return o;
}

// settle: { "steps":N, "dt":d, "holds":[{ "dofs":pat, "mode":"pd", "kp":, "kd":}] }.
Value SaveSettle(const cook::SettleSpec& s) {
    Value o = Value::Object();
    o.Set("steps", Value::Int(static_cast<int64_t>(s.steps)));
    o.Set("dt", Value::Float(s.dt));
    Value holds = Value::Array();
    for (const cook::SettleSpec::Hold& h : s.holds) {
        Value ho = Value::Object();
        ho.Set("dofs", Value::Str(h.dof_pattern));
        ho.Set("mode", Value::Str("pd"));  // the only Mode today.
        ho.Set("kp", Value::Float(h.kp));
        ho.Set("kd", Value::Float(h.kd));
        holds.PushBack(std::move(ho));
    }
    o.Set("holds", std::move(holds));
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
        const bool is_visual = ecs.Has<VisualMeshComponent>(e);
        Value shape = SaveShape(s, sink, is_visual);
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
        if (is_visual) {
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
        for (const SensorDesc& s : scene.Sensors()) sensors.PushBack(SaveSensor(s));
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

    // -- media (cloth / soft-tet / fluid) -----------------------------------
    // First-class media records the cook reads; emitted only when present so a
    // media-free scene's bytes are unchanged. Authoring order == Media() order.
    if (!scene.Media().empty()) {
        Value media = Value::Array();
        for (const MediaRecord& m : scene.Media()) media.PushBack(SaveMedia(m));
        root.Set("media", std::move(media));
    }

    // -- terrain (parametric heightfield field) -----------------------------
    // Emitted only when present so a terrain-free scene's bytes are unchanged.
    if (!scene.Terrain().empty()) {
        Value terrain = Value::Array();
        for (const TerrainRecord& t : scene.Terrain()) terrain.PushBack(SaveTerrain(t));
        root.Set("terrain", std::move(terrain));
    }

    // -- M7 metadata: initial_state / settle (only when present) ------------
    // Emitted only when authored so a scene without IC/settle is unchanged (the
    // existing SecondSaveByteIdentical fixtures stay byte-equal). A legacy
    // `grasp` block (GraspConfig, removed) is NEVER re-emitted; on Load it is
    // silently skipped (forward-compat).
    if (!scene.InitialState().empty()) {
        root.Set("initial_state", SaveInitialState(scene.InitialState()));
    }
    if (scene.Settle().steps != 0 || !scene.Settle().holds.empty()) {
        root.Set("settle", SaveSettle(scene.Settle()));
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
            const bool is_visual = (s == nullptr);  // visual_mesh nodes have no collision_shape
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
                if (ref.fourcc == NkaTagMesh()) {
                    // M8.5 T5: a VISUAL geom's triangles live in a MESH chunk.
                    // Decode the source triangles back into the record (so the
                    // SceneRoundtrip mesh-equality + cook gates hold) AND remember
                    // the resolved AssetRef so ProjectShape can wire it onto the
                    // VisualMeshComponent for the render consumer. The on-disk ref
                    // stores only the basename (deterministic, location-free); we
                    // rewrite nka_path to the full sibling .nka path here so the
                    // render-time open_nka() resolves regardless of the CWD.
                    const NkaMesh m = DecodeMesh(bytes);
                    rec.mesh_vertices = m.positions;
                    rec.mesh_indices = m.indices;
                    // Authored normals (when non-zero) round-trip back so a
                    // re-Save re-emits the identical MESH stream.
                    if (!m.normals.empty() &&
                        std::any_of(m.normals.begin(), m.normals.end(),
                                    [](float n) { return n != 0.0f; })) {
                        rec.mesh_normals = m.normals;
                    }
                    AssetRef resolved = ref;
                    resolved.nka_path = nka_path;
                    rec.visual_mesh_ref = ToString(resolved);
                } else {
                    DecodeCollisionMesh(bytes, rec.mesh_vertices, rec.mesh_indices);
                }
            }
            (void)is_visual;  // routing is driven by the chunk fourcc above
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
    // The legacy 5-key form loads as a Body mount; the unified extras (mount /
    // camera / lidar / aov_mask / update_period) are read additively when present.
    if (const Value* sensors = root.Find("sensors")) {
        for (const Value& s : sensors->Elements()) {
            SensorDesc rec;
            rec.name = s.At("name").AsString();
            rec.type = SensorTypeFromName(s.At("type").AsString());
            rec.mount_index = static_cast<uint32_t>(s.At("attached_body").AsInt());
            rec.local_offset = TransformFromJson(s.At("local"));
            rec.sample_rate_hz = s.At("sample_rate_hz").AsFloat();
            rec.mount = MountFrame::Body;
            if (const Value* m = s.Find("mount")) rec.mount = MountFrameFromName(m->AsString());
            if (const Value* up = s.Find("update_period")) {
                rec.update_period = static_cast<uint32_t>(up->AsInt());
            }
            if (const Value* am = s.Find("aov_mask")) {
                rec.aov_mask = static_cast<uint32_t>(am->AsInt());
            }
            if (const Value* c = s.Find("camera")) {
                rec.cam.width = static_cast<uint16_t>(c->At("width").AsInt());
                rec.cam.height = static_cast<uint16_t>(c->At("height").AsInt());
                rec.cam.vfov_degrees = c->At("vfov_degrees").AsFloat();
                rec.cam.near_clip = c->At("near_clip").AsFloat();
                rec.cam.far_clip = c->At("far_clip").AsFloat();
                rec.cam.distortion = static_cast<uint8_t>(c->At("distortion").AsInt());
                rec.cam.k1 = c->At("k1").AsFloat();
                rec.cam.k2 = c->At("k2").AsFloat();
            }
            if (const Value* l = s.Find("lidar")) {
                rec.lidar.az_count = static_cast<uint16_t>(l->At("az_count").AsInt());
                rec.lidar.el_count = static_cast<uint16_t>(l->At("el_count").AsInt());
                rec.lidar.az_min = l->At("az_min").AsFloat();
                rec.lidar.az_max = l->At("az_max").AsFloat();
                rec.lidar.el_min = l->At("el_min").AsFloat();
                rec.lidar.el_max = l->At("el_max").AsFloat();
                rec.lidar.min_range = l->At("min_range").AsFloat();
                rec.lidar.max_range = l->At("max_range").AsFloat();
            }
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

    // -- media (cloth / soft-tet / fluid) -----------------------------------
    // Each read keeps the record's struct default when absent, so a hand-authored
    // .nks may omit any block/field it does not use (kind + method are required).
    if (const Value* media = root.Find("media")) {
        auto f = [](const Value& o, const char* k, float d) {
            const Value* v = o.Find(k); return v ? v->AsFloat() : d; };
        auto u = [](const Value& o, const char* k, uint32_t d) {
            const Value* v = o.Find(k);
            return v ? static_cast<uint32_t>(v->AsInt()) : d; };
        auto b = [](const Value& o, const char* k, bool d) {
            const Value* v = o.Find(k); return v ? v->AsBool() : d; };
        auto v3 = [](const Value& o, const char* k, math::Vec3 d) {
            const Value* v = o.Find(k); return v ? Vec3FromJson(*v) : d; };

        for (const Value& mv : media->Elements()) {
            MediaRecord rec;
            if (const Value* nm = mv.Find("name")) rec.name = nm->AsString();
            rec.kind = MediaKindFromName(mv.At("kind").AsString());
            rec.method = MediaMethodFromName(mv.At("method").AsString());
            rec.render_material_id = u(mv, "render_material_id", rec.render_material_id);
            if (const Value* bk = mv.Find("baked")) rec.baked = AssetRefFromJson(*bk);

            if (const Value* cg = mv.Find("cloth_grid")) {
                rec.cloth_grid.nx = u(*cg, "nx", rec.cloth_grid.nx);
                rec.cloth_grid.ny = u(*cg, "ny", rec.cloth_grid.ny);
                rec.cloth_grid.spacing = f(*cg, "spacing", rec.cloth_grid.spacing);
                rec.cloth_grid.origin = v3(*cg, "origin", rec.cloth_grid.origin);
                rec.cloth_grid.free = b(*cg, "free", rec.cloth_grid.free);
            }
            if (const Value* ts = mv.Find("tet_sphere")) {
                rec.tet_sphere.center = v3(*ts, "center", rec.tet_sphere.center);
                rec.tet_sphere.radius = f(*ts, "radius", rec.tet_sphere.radius);
                rec.tet_sphere.cells = u(*ts, "cells", rec.tet_sphere.cells);
                rec.tet_sphere.cell_len = f(*ts, "cell_len", rec.tet_sphere.cell_len);
            }
            if (const Value* fb = mv.Find("fluid_box")) {
                rec.fluid_box.min = v3(*fb, "min", rec.fluid_box.min);
                rec.fluid_box.max = v3(*fb, "max", rec.fluid_box.max);
                rec.fluid_box.spacing = f(*fb, "spacing", rec.fluid_box.spacing);
            }
            if (const Value* xp = mv.Find("xpbd")) {
                rec.xpbd.particle_mass = f(*xp, "particle_mass", rec.xpbd.particle_mass);
                rec.xpbd.friction = f(*xp, "friction", rec.xpbd.friction);
                rec.xpbd.distance_alpha = f(*xp, "distance_alpha", rec.xpbd.distance_alpha);
                rec.xpbd.bend_alpha = f(*xp, "bend_alpha", rec.xpbd.bend_alpha);
                rec.xpbd.volume_alpha = f(*xp, "volume_alpha", rec.xpbd.volume_alpha);
                rec.xpbd.iters = static_cast<uint16_t>(u(*xp, "iters", rec.xpbd.iters));
                rec.xpbd.aero_drag_normal = f(*xp, "aero_drag_normal", rec.xpbd.aero_drag_normal);
                rec.xpbd.aero_drag_tangent = f(*xp, "aero_drag_tangent", rec.xpbd.aero_drag_tangent);
                rec.xpbd.aero_drag_max_dv = f(*xp, "aero_drag_max_dv", rec.xpbd.aero_drag_max_dv);
            }
            if (const Value* pb = mv.Find("pbf")) {
                rec.pbf.rest_density = f(*pb, "rest_density", rec.pbf.rest_density);
                rec.pbf.support_scale = f(*pb, "support_scale", rec.pbf.support_scale);
                rec.pbf.iters = static_cast<uint16_t>(u(*pb, "iters", rec.pbf.iters));
                rec.pbf.friction = f(*pb, "friction", rec.pbf.friction);
                rec.pbf.clamp_overdensity = b(*pb, "clamp_overdensity", rec.pbf.clamp_overdensity);
                rec.pbf.walls_enabled = b(*pb, "walls_enabled", rec.pbf.walls_enabled);
                rec.pbf.walls_min = v3(*pb, "walls_min", rec.pbf.walls_min);
                rec.pbf.walls_max = v3(*pb, "walls_max", rec.pbf.walls_max);
                rec.pbf.floor_z = f(*pb, "floor_z", rec.pbf.floor_z);
                rec.pbf.boundary_layers = u(*pb, "boundary_layers", rec.pbf.boundary_layers);
            }
            if (const Value* mp = mv.Find("mpm")) {
                rec.mpm.youngs = f(*mp, "youngs", rec.mpm.youngs);
                rec.mpm.poisson = f(*mp, "poisson", rec.mpm.poisson);
                rec.mpm.density = f(*mp, "density", rec.mpm.density);
                rec.mpm.dp_friction = f(*mp, "dp_friction", rec.mpm.dp_friction);
                rec.mpm.dp_cohesion = f(*mp, "dp_cohesion", rec.mpm.dp_cohesion);
                rec.mpm.model_kind = f(*mp, "model_kind", rec.mpm.model_kind);
                rec.mpm.bulk_modulus = f(*mp, "bulk_modulus", rec.mpm.bulk_modulus);
                rec.mpm.tait_gamma = f(*mp, "tait_gamma", rec.mpm.tait_gamma);
                rec.mpm.viscosity = f(*mp, "viscosity", rec.mpm.viscosity);
                rec.mpm.dx = f(*mp, "dx", rec.mpm.dx);
                rec.mpm.substeps = u(*mp, "substeps", rec.mpm.substeps);
                rec.mpm.floor_normal = v3(*mp, "floor_normal", rec.mpm.floor_normal);
                rec.mpm.floor_d = f(*mp, "floor_d", rec.mpm.floor_d);
                rec.mpm.floor_friction = f(*mp, "floor_friction", rec.mpm.floor_friction);
            }
            if (const Value* rs = mv.Find("render_skin")) {
                rec.render_skin.normal_offset = f(*rs, "normal_offset", rec.render_skin.normal_offset);
                rec.render_skin.smooth_iters = u(*rs, "smooth_iters", rec.render_skin.smooth_iters);
                rec.render_skin.smooth_lambda = f(*rs, "smooth_lambda", rec.render_skin.smooth_lambda);
                if (const Value* sm = rs->Find("skin_mesh"))
                    rec.render_skin.skin_mesh = AssetRefFromJson(*sm);
            }
            scene.AddMedia(std::move(rec));
        }
    }

    // -- terrain (parametric heightfield field) -----------------------------
    // Each read keeps the record's struct default when absent (hand-authorable).
    if (const Value* terrain = root.Find("terrain")) {
        auto f = [](const Value& o, const char* k, float d) {
            const Value* v = o.Find(k); return v ? v->AsFloat() : d; };
        auto u = [](const Value& o, const char* k, uint32_t d) {
            const Value* v = o.Find(k);
            return v ? static_cast<uint32_t>(v->AsInt()) : d; };
        auto v3 = [](const Value& o, const char* k, math::Vec3 d) {
            const Value* v = o.Find(k); return v ? Vec3FromJson(*v) : d; };
        for (const Value& tv : terrain->Elements()) {
            TerrainRecord rec;
            if (const Value* nm = tv.Find("name")) rec.name = nm->AsString();
            rec.nrow = u(tv, "nrow", rec.nrow);
            rec.ncol = u(tv, "ncol", rec.ncol);
            rec.cell = f(tv, "cell", rec.cell);
            rec.origin = v3(tv, "origin", rec.origin);
            rec.base_z = f(tv, "base_z", rec.base_z);
            rec.grade_x = f(tv, "grade_x", rec.grade_x);
            rec.grade_y = f(tv, "grade_y", rec.grade_y);
            rec.ring_rise = f(tv, "ring_rise", rec.ring_rise);
            rec.ring_width = f(tv, "ring_width", rec.ring_width);
            rec.ring_platform = f(tv, "ring_platform", rec.ring_platform);
            rec.ring_count = u(tv, "ring_count", rec.ring_count);
            rec.bump_height = f(tv, "bump_height", rec.bump_height);
            rec.bump_cell = f(tv, "bump_cell", rec.bump_cell);
            rec.feature_cell = f(tv, "feature_cell", rec.feature_cell);
            rec.feature_margin = f(tv, "feature_margin", rec.feature_margin);
            rec.feature_seed = u(tv, "feature_seed", rec.feature_seed);
            rec.curric_levels = u(tv, "curric_levels", rec.curric_levels);
            rec.curric_types = u(tv, "curric_types", rec.curric_types);
            if (const Value* ip = tv.Find("image_path")) rec.image_path = ip->AsString();
            rec.image_radius_x = f(tv, "image_radius_x", rec.image_radius_x);
            rec.image_radius_y = f(tv, "image_radius_y", rec.image_radius_y);
            rec.image_elevation_z = f(tv, "image_elevation_z", rec.image_elevation_z);
            scene.AddTerrain(std::move(rec));
        }
    }

    // -- M7 metadata: initial_state / settle --------------------------------
    if (const Value* is = root.Find("initial_state")) {
        SceneInitialState& dst = scene.InitialStateMut();
        for (const auto& kv : is->Items()) {
            ArticulationInitialState rec;
            if (const Value* q = kv.second.Find("qpos")) {
                for (const Value& v : q->Elements()) rec.qpos.push_back(v.AsFloat());
            }
            rec.root = TransformFromJson(kv.second.At("root"));
            dst.emplace(kv.first, std::move(rec));
        }
    }
    if (const Value* st = root.Find("settle")) {
        cook::SettleSpec& spec = scene.SettleMut();
        spec.steps = static_cast<int>(st->At("steps").AsInt());
        spec.dt = st->At("dt").AsFloat();
        if (const Value* holds = st->Find("holds")) {
            for (const Value& h : holds->Elements()) {
                cook::SettleSpec::Hold hold;
                hold.dof_pattern = h.At("dofs").AsString();
                hold.mode = cook::SettleSpec::Hold::Mode::PD;  // "pd" is the only mode.
                if (const Value* kp = h.Find("kp")) hold.kp = kp->AsFloat();
                if (const Value* kd = h.Find("kd")) hold.kd = kd->AsFloat();
                spec.holds.push_back(std::move(hold));
            }
        }
    }
    // Legacy `grasp` block (GraspConfig, removed): forward-compat skip. An older
    // committed .nks (e.g. examples/scenes/h1_cup_table.nks) may still carry a
    // top-level "grasp" JSON object plus a sibling .nka SAMP chunk for the cup
    // hull. We simply do NOT read either -- the JSON key is ignored and the SAMP
    // chunk is never requested -- so a legacy file still loads without error.
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
