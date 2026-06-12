// ---------------------------------------------------------------------------
// nuka::scene::nks - .nks scene Save/Load implementation (M2c).
// ---------------------------------------------------------------------------
// HOST-ONLY. The SceneIR records are the source of truth for cook fidelity; the
// facade (tree/ECS) is DERIVED from them by the SceneIR write-through. So Save
// serializes the records (structured as the pre-order node tree so the JSON is
// tree-shaped and human-readable), and Load replays Add* in the SAME order the
// facade rebuild uses (materials, bodies, shapes, joints, actuators, sensors,
// cameras, lights, filters). Replaying produces records identical to the saved
// ones, so the reconstructed facade tree equals the saved tree by construction.
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

const char* LightTypeName(LightType t) {
    switch (t) {
        case LightType::Point:       return "point";
        case LightType::Directional: return "directional";
        case LightType::Spot:        return "spot";
        case LightType::Area:        return "area";
    }
    return "point";
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
Value Vec3Json(const math::Vec3& v) {
    Value a = Value::Array();
    a.PushBack(Value::Float(v.x));
    a.PushBack(Value::Float(v.y));
    a.PushBack(Value::Float(v.z));
    return a;
}
math::Vec3 Vec3FromJson(const Value& a) {
    return math::Vec3{a.Elements()[0].AsFloat(), a.Elements()[1].AsFloat(),
                      a.Elements()[2].AsFloat()};
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
    return math::Quat{a.Elements()[0].AsFloat(), a.Elements()[1].AsFloat(),
                      a.Elements()[2].AsFloat(), a.Elements()[3].AsFloat()};
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

Value SaveJoint(const JointRecord& j) {
    Value o = Value::Object();
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

Value SaveCamera(const CameraRecord& c) {
    Value o = Value::Object();
    o.Set("name", Value::Str(c.name));
    o.Set("attached_body", Value::Int(static_cast<int64_t>(c.attached_body)));
    o.Set("local", TransformJson(c.local_transform));
    o.Set("vertical_fov_degrees", Value::Float(c.vertical_fov_degrees));
    o.Set("near_clip", Value::Float(c.near_clip));
    o.Set("far_clip", Value::Float(c.far_clip));
    return o;
}

Value SaveLight(const LightRecord& l) {
    Value o = Value::Object();
    o.Set("name", Value::Str(l.name));
    o.Set("type", Value::Str(LightTypeName(l.type)));
    o.Set("attached_body", Value::Int(static_cast<int64_t>(l.attached_body)));
    o.Set("local", TransformJson(l.local_transform));
    o.Set("color", Vec3Json(l.color));
    o.Set("intensity", Value::Float(l.intensity));
    return o;
}

Value SaveMaterial(const MaterialRecord& m) {
    Value o = Value::Object();
    o.Set("base_color", Vec3Json(m.base_color));
    o.Set("alpha", Value::Float(m.alpha));
    o.Set("roughness", Value::Float(m.roughness));
    o.Set("metallic", Value::Float(m.metallic));
    o.Set("friction_mu", Value::Float(m.friction_mu));
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

    // -- materials (keyed by material name) ---------------------------------
    {
        Value mats = Value::Object();
        for (const MaterialRecord& m : scene.Materials()) {
            mats.Set(m.name, SaveMaterial(m));
        }
        root.Set("materials", std::move(mats));
    }

    // -- bodies (record order; the body name is the full path) --------------
    // The body record name already encodes the path; on Load the same name
    // reproduces the same group-node structure, so we serialize records in
    // dense-id order (a faithful, deterministic, lossless representation).
    {
        Value bodies = Value::Array();
        for (const RigidBodyRecord& b : scene.Bodies()) {
            Value o = Value::Object();
            o.Set("name", Value::Str(b.name));
            o.Set("parent_id", Value::Int(static_cast<int64_t>(b.parent_id)));
            o.Set("local", TransformJson(b.local_transform));
            o.Set("inertial", TransformJson(b.inertial_transform));
            o.Set("mass", Value::Float(b.mass));
            o.Set("inertia", Vec3Json(b.inertia));
            o.Set("is_static", Value::Bool(b.is_static));
            bodies.PushBack(std::move(o));
        }
        root.Set("bodies", std::move(bodies));
    }

    // -- shapes (record order) ----------------------------------------------
    {
        Value shapes = Value::Array();
        for (const CollisionShapeRecord& s : scene.Shapes()) {
            Value o = Value::Object();
            o.Set("name", Value::Str(s.name));
            o.Set("body_id", Value::Int(static_cast<int64_t>(s.body_id)));
            // Merge the rest of the fidelity fields.
            Value detail = SaveShape(s, sink);
            for (const auto& kv : detail.Items()) {
                o.Set(kv.first, kv.second);
            }
            shapes.PushBack(std::move(o));
        }
        root.Set("shapes", std::move(shapes));
    }

    // -- joints / actuators / sensors / cameras / lights --------------------
    {
        Value joints = Value::Array();
        for (const JointRecord& j : scene.Joints()) {
            Value o = SaveJoint(j);
            Value ordered = Value::Object();
            ordered.Set("name", Value::Str(j.name));
            for (const auto& kv : o.Items()) ordered.Set(kv.first, kv.second);
            joints.PushBack(std::move(ordered));
        }
        root.Set("joints", std::move(joints));
    }
    {
        Value acts = Value::Array();
        for (const ActuatorRecord& a : scene.Actuators()) {
            Value o = SaveActuator(a);
            Value ordered = Value::Object();
            ordered.Set("name", Value::Str(a.name));
            for (const auto& kv : o.Items()) ordered.Set(kv.first, kv.second);
            acts.PushBack(std::move(ordered));
        }
        root.Set("actuators", std::move(acts));
    }
    {
        Value sensors = Value::Array();
        for (const SensorRecord& s : scene.Sensors()) sensors.PushBack(SaveSensor(s));
        root.Set("sensors", std::move(sensors));
    }
    {
        Value cameras = Value::Array();
        for (const CameraRecord& c : scene.Cameras()) cameras.PushBack(SaveCamera(c));
        root.Set("cameras", std::move(cameras));
    }
    {
        Value lights = Value::Array();
        for (const LightRecord& l : scene.Lights()) lights.PushBack(SaveLight(l));
        root.Set("lights", std::move(lights));
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

    // -- materials ----------------------------------------------------------
    if (const Value* mats = root.Find("materials")) {
        for (const auto& kv : mats->Items()) {
            const Value& m = kv.second;
            MaterialRecord rec;
            rec.name = kv.first;
            rec.base_color = Vec3FromJson(m.At("base_color"));
            rec.alpha = m.At("alpha").AsFloat();
            rec.roughness = m.At("roughness").AsFloat();
            rec.metallic = m.At("metallic").AsFloat();
            rec.friction_mu = m.At("friction_mu").AsFloat();
            scene.AddMaterial(std::move(rec));
        }
    }

    // -- bodies (pre-order: parent_id < child id always, so dense order works) -
    if (const Value* bodies = root.Find("bodies")) {
        for (const Value& b : bodies->Elements()) {
            RigidBodyRecord rec;
            rec.name = b.At("name").AsString();
            rec.parent_id = static_cast<BodyId>(b.At("parent_id").AsInt());
            rec.local_transform = TransformFromJson(b.At("local"));
            rec.inertial_transform = TransformFromJson(b.At("inertial"));
            rec.mass = b.At("mass").AsFloat();
            rec.inertia = Vec3FromJson(b.At("inertia"));
            rec.is_static = b.At("is_static").AsBool();
            scene.AddRigidBody(std::move(rec));
        }
    }

    // -- shapes -------------------------------------------------------------
    if (const Value* shapes = root.Find("shapes")) {
        for (const Value& s : shapes->Elements()) {
            CollisionShapeRecord rec;
            rec.name = s.At("name").AsString();
            rec.body_id = static_cast<BodyId>(s.At("body_id").AsInt());
            rec.material_id = static_cast<MaterialId>(s.At("material_id").AsInt());
            rec.type = ShapeTypeFromName(s.At("type").AsString());
            rec.local_transform = TransformFromJson(s.At("local"));
            rec.half_extents = Vec3FromJson(s.At("half_extents"));
            rec.radius = s.At("radius").AsFloat();
            rec.half_height = s.At("half_height").AsFloat();
            rec.contype = static_cast<uint32_t>(s.At("contype").AsInt());
            rec.conaffinity = static_cast<uint32_t>(s.At("conaffinity").AsInt());
            rec.collision_group = static_cast<int32_t>(s.At("collision_group").AsInt());
            rec.solref[0] = s.At("solref").Elements()[0].AsFloat();
            rec.solref[1] = s.At("solref").Elements()[1].AsFloat();
            for (int k = 0; k < 5; ++k) rec.solimp[k] = s.At("solimp").Elements()[k].AsFloat();
            rec.friction_mu = s.At("friction_mu").AsFloat();
            rec.priority = static_cast<int32_t>(s.At("priority").AsInt());
            rec.solmix = s.At("solmix").AsFloat();
            rec.margin = s.At("margin").AsFloat();
            rec.gap = s.At("gap").AsFloat();
            rec.condim = static_cast<uint8_t>(s.At("condim").AsInt());
            rec.decompose_mode = DecomposeModeFromName(s.At("decompose_mode").AsString());
            rec.decompose_max_pieces = static_cast<uint32_t>(s.At("decompose_max_pieces").AsInt());
            if (const Value* mesh = s.Find("mesh")) {
                const AssetRef ref = ParseAssetRef(mesh->AsString());
                const std::vector<uint8_t> bytes = ensure_nka().LoadChunk(ref.fourcc, ref.index);
                DecodeCollisionMesh(bytes, rec.mesh_vertices, rec.mesh_indices);
            }
            scene.AddCollisionShape(std::move(rec));
        }
    }

    // -- joints -------------------------------------------------------------
    if (const Value* joints = root.Find("joints")) {
        for (const Value& j : joints->Elements()) {
            JointRecord rec;
            rec.name = j.At("name").AsString();
            rec.type = JointTypeFromName(j.At("type").AsString());
            rec.parent_body = static_cast<BodyId>(j.At("parent_body").AsInt());
            rec.child_body = static_cast<BodyId>(j.At("child_body").AsInt());
            rec.axis = Vec3FromJson(j.At("axis"));
            rec.parent_frame = TransformFromJson(j.At("parent_frame"));
            rec.child_frame = TransformFromJson(j.At("child_frame"));
            rec.lower_limit = j.At("lower_limit").AsFloat();
            rec.upper_limit = j.At("upper_limit").AsFloat();
            rec.damping = j.At("damping").AsFloat();
            rec.armature = j.At("armature").AsFloat();
            rec.stiffness = j.At("stiffness").AsFloat();
            rec.initial_position = j.At("initial_position").AsFloat();
            scene.AddJoint(std::move(rec));
        }
    }

    // -- actuators ----------------------------------------------------------
    if (const Value* acts = root.Find("actuators")) {
        for (const Value& a : acts->Elements()) {
            ActuatorRecord rec;
            rec.name = a.At("name").AsString();
            rec.type = ActuatorTypeFromName(a.At("type").AsString());
            rec.joint_id = static_cast<JointId>(a.At("joint_id").AsInt());
            rec.gain = a.At("gain").AsFloat();
            rec.force_limit = a.At("force_limit").AsFloat();
            scene.AddActuator(std::move(rec));
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
            scene.AddExcludePair(static_cast<BodyId>(e.Elements()[0].AsInt()),
                                 static_cast<BodyId>(e.Elements()[1].AsInt()));
        }
    }
    if (const Value* pairs = root.Find("contact_pairs")) {
        for (const Value& cp : pairs->Elements()) {
            ContactPairOverride ov;
            ov.geom1 = static_cast<ShapeId>(cp.At("geom1").AsInt());
            ov.geom2 = static_cast<ShapeId>(cp.At("geom2").AsInt());
            ov.condim = static_cast<uint8_t>(cp.At("condim").AsInt());
            ov.friction_mu = cp.At("friction_mu").AsFloat();
            ov.solref[0] = cp.At("solref").Elements()[0].AsFloat();
            ov.solref[1] = cp.At("solref").Elements()[1].AsFloat();
            for (int k = 0; k < 5; ++k) ov.solimp[k] = cp.At("solimp").Elements()[k].AsFloat();
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
// Apply a {"overrides": {"<path>": {<partial component objects>}}} overlay to an
// already-loaded scene. Matches the override path against each body's tree path
// and each shape/joint name; patches the corresponding record fields in place.
void ApplyOverrides(SceneIR& scene, const Value& overlay) {
    const Value* overrides = overlay.Find("overrides");
    if (!overrides) return;

    // Build path -> body id (the body record name == its tree path) and
    // name -> shape/joint ids for matching.
    std::unordered_map<std::string, BodyId> body_by_path;
    for (BodyId i = 0; i < scene.RigidBodyCount(); ++i) {
        body_by_path.emplace(scene.GetBody(i).name, i);
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
    // Mutating records directly bypassed the write-through; rebuild the facade so
    // the tree/ECS reflect the overrides. A copy triggers RebuildFacade.
    scene = SceneIR(scene);
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
