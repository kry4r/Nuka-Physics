// ---------------------------------------------------------------------------
// nuka::import - MJCF (MuJoCo XML) importer implementation
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"

#include "import/mesh_file_loader.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "scene/canonical_types.hpp"

#include <tinyxml2.h>

#include <filesystem>
#include <iostream>
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

// Fill up to `count` floats from a whitespace-separated MuJoCo list into `out`,
// assigning out[i] ONLY for entries that successfully extract. Entries beyond
// what the text provides are LEFT UNCHANGED (so a partial solref="0.01" keeps
// the default solref[1]). NOTE: a failed std::istream extraction zeroes its
// target since C++11, so we extract into a temporary and copy only on success —
// reusing `ss >> out[i]` directly would clobber the trailing defaults with 0.
// Returns the number of values actually parsed.
int ParseFloatList(const char* text, float* out, int count) {
    if (!text || count <= 0) {
        return 0;
    }
    std::istringstream ss(text);
    int n = 0;
    for (; n < count; ++n) {
        float v;
        if (!(ss >> v)) {
            break;
        }
        out[n] = v;
    }
    return n;
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

// Per-class <geom> contact-attribute defaults (v0.8 C1b). Each field carries a
// `has_*` presence flag so the geom loop applies a default ONLY when it was
// explicitly authored in the <default><geom>; an unmentioned default attr must
// NOT clobber the CollisionShapeRecord's own (MuJoCo-matching) C1a default.
// (The joint-default path can apply unconditionally because its struct inits
// match JointRecord's; that invariant does NOT hold here — notably the record's
// friction_mu default is the -1 "inherit material μ" sentinel, not MuJoCo's
// 1.0 — so per-field presence tracking is required.)
struct MjcfGeomDefaults {
    bool has_contype = false;       uint32_t contype = 1;
    bool has_conaffinity = false;   uint32_t conaffinity = 1;
    bool has_group = false;         int32_t group = 0;
    bool has_condim = false;        uint8_t condim = 3;
    bool has_priority = false;      int32_t priority = 0;
    bool has_solmix = false;        float solmix = 1.0f;
    bool has_margin = false;        float margin = 0.0f;
    bool has_gap = false;           float gap = 0.0f;
    bool has_solref = false;        float solref[2] = {0.02f, 1.0f};
    bool has_solimp = false;        float solimp[5] = {0.9f, 0.95f, 0.001f, 0.5f, 2.0f};
    bool has_friction = false;      float friction_mu = 1.0f;  // first component only
};

struct MjcfDefaultClass {
    MjcfJointDefaults joint;
    MjcfGeneralDefaults general;
    MjcfGeomDefaults geom;
};

struct MjcfDefaults {
    MjcfDefaultClass root;
    std::unordered_map<std::string, MjcfDefaultClass> classes;
};

// A resolved <asset><mesh> entry: the on-disk path (meshdir-resolved) plus the
// optional per-axis scale applied to the loaded geometry.
struct MjcfMeshAsset {
    std::string resolved_path;
    math::Vec3  scale = {1.0f, 1.0f, 1.0f};
};

struct MjcfParseContext {
    std::unordered_map<std::string, scene::BodyId> body_ids;
    std::unordered_map<std::string, scene::JointId> joint_ids;
    std::unordered_map<std::string, scene::MaterialId> material_ids;
    std::unordered_map<std::string, MjcfMeshAsset> mesh_assets;
    // Named geoms only: resolves a <contact><pair geom1=/geom2=> name to the
    // ShapeId returned by AddCollisionShape (v0.8 C1b).
    std::unordered_map<std::string, scene::ShapeId> geom_ids;
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

// Read the first component of a MuJoCo `friction` list (slide,spin,roll[,...]).
// MuJoCo geom/pair friction is a vector; v0.8 models tangential friction as a
// single isotropic μ, so we take ONLY friction[0] and DROP the spin/roll (and
// any further) components. Returns true if at least one value was present.
bool ParseFrictionFirst(const char* text, float& out_mu) {
    if (!text) {
        return false;
    }
    float v;
    if (ParseFloatList(text, &v, 1) == 1) {
        out_mu = v;
        return true;
    }
    return false;
}

// Read the C1b contact attributes from a <default>'s <geom> child into the
// class's geom-default struct, setting the matching has_* flag for each attr
// that is actually present (absent attrs leave both the value and flag alone, so
// they inherit from the parent class via the `current = inherited` copy).
void ApplyGeomDefault(const tinyxml2::XMLElement* geom_elem, MjcfDefaultClass* defaults) {
    if (geom_elem == nullptr || defaults == nullptr) {
        return;
    }
    MjcfGeomDefaults& g = defaults->geom;

    if (geom_elem->QueryUnsignedAttribute("contype", &g.contype) == tinyxml2::XML_SUCCESS) {
        g.has_contype = true;
    }
    if (geom_elem->QueryUnsignedAttribute("conaffinity", &g.conaffinity) ==
            tinyxml2::XML_SUCCESS) {
        g.has_conaffinity = true;
    }
    if (geom_elem->QueryIntAttribute("group", &g.group) == tinyxml2::XML_SUCCESS) {
        g.has_group = true;
    }
    int condim = 0;
    if (geom_elem->QueryIntAttribute("condim", &condim) == tinyxml2::XML_SUCCESS) {
        g.condim = static_cast<uint8_t>(condim);
        g.has_condim = true;
    }
    if (geom_elem->QueryIntAttribute("priority", &g.priority) == tinyxml2::XML_SUCCESS) {
        g.has_priority = true;
    }
    if (geom_elem->QueryFloatAttribute("solmix", &g.solmix) == tinyxml2::XML_SUCCESS) {
        g.has_solmix = true;
    }
    if (geom_elem->QueryFloatAttribute("margin", &g.margin) == tinyxml2::XML_SUCCESS) {
        g.has_margin = true;
    }
    if (geom_elem->QueryFloatAttribute("gap", &g.gap) == tinyxml2::XML_SUCCESS) {
        g.has_gap = true;
    }
    if (geom_elem->Attribute("solref") &&
        ParseFloatList(geom_elem->Attribute("solref"), g.solref, 2) > 0) {
        g.has_solref = true;
    }
    if (geom_elem->Attribute("solimp") &&
        ParseFloatList(geom_elem->Attribute("solimp"), g.solimp, 5) > 0) {
        g.has_solimp = true;
    }
    if (ParseFrictionFirst(geom_elem->Attribute("friction"), g.friction_mu)) {
        g.has_friction = true;
    }
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
    ApplyGeomDefault(default_elem->FirstChildElement("geom"), &current);

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

// Resolve <asset><mesh name= file= scale=> entries into the context, honoring
// <compiler meshdir="...">. Mesh paths are resolved relative to the MJCF file's
// own directory (base_dir) + meshdir. The actual mesh bytes are loaded lazily
// when a geom references the asset (see ParseBody) so unused assets cost nothing.
void ParseMeshAssets(tinyxml2::XMLElement* mujoco,
                     MjcfParseContext& context,
                     const std::filesystem::path& base_dir) {
    std::filesystem::path meshdir;
    if (auto* compiler = mujoco->FirstChildElement("compiler")) {
        if (const char* md = compiler->Attribute("meshdir")) {
            meshdir = md;
        }
    }

    auto* asset = mujoco->FirstChildElement("asset");
    if (!asset) {
        return;
    }

    for (auto* mesh = asset->FirstChildElement("mesh");
         mesh != nullptr;
         mesh = mesh->NextSiblingElement("mesh")) {

        const char* name = mesh->Attribute("name");
        const char* file = mesh->Attribute("file");
        if (!name || !file) {
            continue;
        }

        MjcfMeshAsset record;
        // std::filesystem::operator/ resets to the right operand if it is
        // absolute, so an absolute meshdir or file path is handled correctly.
        record.resolved_path = (base_dir / meshdir / file).string();
        if (const char* scale = mesh->Attribute("scale")) {
            record.scale = ParseVec3(scale);
        }
        context.mesh_assets[name] = std::move(record);
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

            // Load the referenced mesh file (STL/OBJ) into the shape's source
            // geometry so the cooker (V-HACD/SDF) and renderer have triangles.
            if (const char* mesh_name = geom->Attribute("mesh")) {
                const auto it = context.mesh_assets.find(mesh_name);
                if (it == context.mesh_assets.end()) {
                    throw std::runtime_error(
                        "MJCF: geom references unknown mesh asset '" +
                        std::string(mesh_name) + "'");
                }
                MeshGeometry geo = LoadMeshFile(it->second.resolved_path);
                const math::Vec3 s = it->second.scale;
                for (std::size_t i = 0; i + 2 < geo.vertices.size(); i += 3) {
                    geo.vertices[i + 0] *= s.x;
                    geo.vertices[i + 1] *= s.y;
                    geo.vertices[i + 2] *= s.z;
                }
                shape.mesh_vertices = std::move(geo.vertices);
                shape.mesh_indices = std::move(geo.indices);
            }
        }

        // -- C1b collision/contact attributes --------------------------------
        // Precedence (low -> high): the CollisionShapeRecord's C1a defaults
        // (already MuJoCo's) < the resolved <default> class's <geom> attrs
        // (presence-gated) < the geom's own explicit attrs. Geom class resolves
        // exactly like joints: the geom's own `class` attr, else the body's
        // (inherited) `childclass`.
        const char* geom_class = geom->Attribute("class");
        const MjcfGeomDefaults& gd =
            DefaultClassOrRoot(context.defaults, geom_class ? geom_class : child_class).geom;
        if (gd.has_contype)     { shape.contype = gd.contype; }
        if (gd.has_conaffinity) { shape.conaffinity = gd.conaffinity; }
        if (gd.has_group)       { shape.collision_group = gd.group; }
        if (gd.has_condim)      { shape.condim = gd.condim; }
        if (gd.has_priority)    { shape.priority = gd.priority; }
        if (gd.has_solmix)      { shape.solmix = gd.solmix; }
        if (gd.has_margin)      { shape.margin = gd.margin; }
        if (gd.has_gap)         { shape.gap = gd.gap; }
        if (gd.has_solref)      { shape.solref[0] = gd.solref[0]; shape.solref[1] = gd.solref[1]; }
        if (gd.has_solimp) {
            for (int k = 0; k < 5; ++k) { shape.solimp[k] = gd.solimp[k]; }
        }
        // friction_mu stays the -1 "inherit material μ" sentinel unless the
        // default class explicitly set a friction; only friction[0] is kept.
        if (gd.has_friction)    { shape.friction_mu = gd.friction_mu; }

        // The geom's own explicit attrs override the resolved class default.
        // QueryUnsignedAttribute/QueryIntAttribute/QueryFloatAttribute write the
        // target ONLY on XML_SUCCESS, so an ABSENT attr leaves the value from
        // the default/record untouched (this also makes contype="0" read as a
        // literal 0, not the default 1 -- the h1 visual-only finger pattern).
        geom->QueryUnsignedAttribute("contype", &shape.contype);
        geom->QueryUnsignedAttribute("conaffinity", &shape.conaffinity);
        geom->QueryIntAttribute("group", &shape.collision_group);
        {
            int condim = 0;
            if (geom->QueryIntAttribute("condim", &condim) == tinyxml2::XML_SUCCESS) {
                shape.condim = static_cast<uint8_t>(condim);
            }
        }
        geom->QueryIntAttribute("priority", &shape.priority);
        geom->QueryFloatAttribute("solmix", &shape.solmix);
        geom->QueryFloatAttribute("margin", &shape.margin);
        geom->QueryFloatAttribute("gap", &shape.gap);
        // solref (<=2) / solimp (<=5): fill what's present, leave the rest at the
        // already-resolved value (ParseFloatList only writes successfully-parsed
        // entries, so a partial list does not zero the trailing defaults).
        ParseFloatList(geom->Attribute("solref"), shape.solref, 2);
        ParseFloatList(geom->Attribute("solimp"), shape.solimp, 5);
        // friction: take ONLY the first (isotropic tangential) component;
        // spin/roll (and any further) friction are DROPPED. Set the per-shape μ
        // override only when friction is explicitly present here (so an
        // unspecified friction keeps the -1 sentinel for the cooker to resolve).
        ParseFrictionFirst(geom->Attribute("friction"), shape.friction_mu);

        const std::string geom_name = shape.name;
        const scene::ShapeId shape_id = scene.AddCollisionShape(std::move(shape));
        if (!geom_name.empty()) {
            context.geom_ids[geom_name] = shape_id;
        }
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

// Parse the top-level <contact> section (a sibling of <worldbody>), which holds
// <exclude> (per-body-pair collision exclusion) and <pair> (explicit per-geom-
// pair contact overrides). Run AFTER the body tree so body/geom names resolve.
// Name resolution is non-fatal: an unresolved name is SKIPPED with a warning
// (matching the importer's "skip-not-crash" intent for authoring slips). The
// filter/merge POLICY that consumes excludes + pair overrides is C1c.
void ParseContact(tinyxml2::XMLElement* mujoco,
                  scene::SceneIR& scene,
                  const MjcfParseContext& context) {
    auto* contact_root = mujoco->FirstChildElement("contact");
    if (!contact_root) {
        return;
    }

    for (auto* exclude = contact_root->FirstChildElement("exclude");
         exclude != nullptr;
         exclude = exclude->NextSiblingElement("exclude")) {

        const char* b1 = exclude->Attribute("body1");
        const char* b2 = exclude->Attribute("body2");
        const scene::BodyId id1 = ResolveBody(b1, context);
        const scene::BodyId id2 = ResolveBody(b2, context);
        if (id1 == scene::kInvalidBody || id2 == scene::kInvalidBody) {
            std::cerr << "MJCF: <contact><exclude> references unknown body '"
                      << (b1 ? b1 : "(null)") << "' / '" << (b2 ? b2 : "(null)")
                      << "' -- skipped\n";
            continue;
        }
        scene.AddExcludePair(id1, id2);
    }

    for (auto* pair = contact_root->FirstChildElement("pair");
         pair != nullptr;
         pair = pair->NextSiblingElement("pair")) {

        const char* g1 = pair->Attribute("geom1");
        const char* g2 = pair->Attribute("geom2");
        const auto it1 = g1 ? context.geom_ids.find(g1) : context.geom_ids.end();
        const auto it2 = g2 ? context.geom_ids.find(g2) : context.geom_ids.end();
        if (it1 == context.geom_ids.end() || it2 == context.geom_ids.end()) {
            std::cerr << "MJCF: <contact><pair> references unknown geom '"
                      << (g1 ? g1 : "(null)") << "' / '" << (g2 ? g2 : "(null)")
                      << "' -- skipped\n";
            continue;
        }

        // A <pair> is an explicit authored override: ContactPairOverride is
        // pre-initialized to MuJoCo's concrete pair defaults (NOT the -1 friction
        // sentinel), so absent attrs keep those defaults.
        scene::ContactPairOverride ov;
        ov.geom1 = it1->second;
        ov.geom2 = it2->second;
        {
            int condim = 0;
            if (pair->QueryIntAttribute("condim", &condim) == tinyxml2::XML_SUCCESS) {
                ov.condim = static_cast<uint8_t>(condim);
            }
        }
        pair->QueryFloatAttribute("margin", &ov.margin);
        pair->QueryFloatAttribute("gap", &ov.gap);
        ParseFloatList(pair->Attribute("solref"), ov.solref, 2);
        ParseFloatList(pair->Attribute("solimp"), ov.solimp, 5);
        // friction: first (isotropic tangential) component only; spin/roll dropped.
        ParseFrictionFirst(pair->Attribute("friction"), ov.friction_mu);
        scene.AddContactPair(ov);
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
    ParseMeshAssets(mujoco, context, std::filesystem::path(path).parent_path());
    ParseLights(worldbody, scene);

    for (auto* body = worldbody->FirstChildElement("body");
         body != nullptr;
         body = body->NextSiblingElement("body")) {
        ParseBody(body, scene::kInvalidBody, scene, context, nullptr);
    }

    ParseActuators(mujoco, scene, context);
    ParseSensors(mujoco, scene, context);
    // <contact> is parsed LAST so all body + geom names already resolve.
    ParseContact(mujoco, scene, context);

    return scene;
}

} // namespace nuka::import
