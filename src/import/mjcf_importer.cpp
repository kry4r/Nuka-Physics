// ---------------------------------------------------------------------------
// nuka::import - MJCF (MuJoCo XML) importer implementation
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"

#include "import/cooker/convex_decomposition.hpp"
#include "import/geom_inertia.hpp"
#include "import/mesh_file_loader.hpp"
#include "import/principal_inertia.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "scene/canonical_types.hpp"

#include <tinyxml2.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

// MuJoCo bounds a joint's summed actuator force when actuatorfrclimited is true, or auto with a range.
void ParseActuatorForceRange(const tinyxml2::XMLElement* element, float& limit) {
    if (const char* range = element->Attribute("actuatorfrcrange")) {
        float lower = 0.0f, upper = 0.0f;
        ParseRange(range, lower, upper);
        limit = std::max(std::fabs(lower), std::fabs(upper));
    }
    if (const char* limited = element->Attribute("actuatorfrclimited"); limited && std::string(limited) == "false") {
        limit = 0.0f;
    }
}

int ParseLimited(const char* text, int fallback) {
    if (text == nullptr) return fallback;
    const std::string value(text);
    if (value == "true" || value == "1") return 1;
    if (value == "false" || value == "0") return 0;
    return -1;
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
    // MJCF spec default joint axis is +Z when the attribute is omitted.
    math::Vec3 axis = {0.0f, 0.0f, 1.0f};
    float lower_limit = -3.14159f;
    float upper_limit = 3.14159f;
    bool has_range = false;
    int limited = -1;  // -1 auto, 0 false, 1 true
    float damping = 0.0f;
    float armature = 0.0f;
    float frictionloss = 0.0f;
    float actuator_force_limit = 0.0f;
    float stiffness = 0.0f;
    float spring_reference = 0.0f;
    constraint::SolverReference limit_reference;
    constraint::SolverReference friction_reference;
    float limit_margin = 0.0f;
};

struct MjcfGeneralDefaults {
    float gear = 1.0f;
    float force_limit = 0.0f;
    float gain = 1.0f;
    float bias[3] = {0.0f, 0.0f, 0.0f};
    bool affine = false;
};

// Per-class <geom> contact-attribute defaults (v0.8 C1b). Each field carries a
// `has_*` presence flag so the geom loop applies a default ONLY when it was
// explicitly authored in the <default><geom>; an unmentioned default attr must
// NOT clobber the CollisionShapeRecord's own (MuJoCo-matching) C1a default.
// (The joint-default path can apply unconditionally because its struct inits
// match JointRecord's; that invariant does NOT hold here — notably the record's
// friction_mu default is the -1 "inherit material μ" sentinel, not MuJoCo's
// 1.0 — so per-field presence tracking is required.)
struct MjcfFrameDefaults {
    math::Vec3 position;
    std::string orientation_tag;
    std::string orientation_value;
    std::string fromto;
};

struct MjcfGeomDefaults : MjcfFrameDefaults {
    // M8.5 T5: the geom `type` can be set by a <default><geom type=...> class
    // (e.g. go2's class="visual" -> type="mesh"); the geom element itself then
    // omits the attr. Capture it so the geom loop resolves mesh geoms whose type
    // lives ONLY on the class default -- otherwise their STL/OBJ never loads and
    // the visual-mesh cook produces an empty .nka.
    bool has_type = false;          std::string type;
    bool has_size = false;          math::Vec3 size{0.5f, 0.5f, 0.5f};
    std::string material;
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
    float friction[3] = {1.0f, 0.005f, 0.0001f};
    float density = 1000.0f;
    float mass = -1.0f;  // negative: the geom's mass comes from its density
};

struct MjcfSiteDefaults : MjcfFrameDefaults {
    std::string type = "sphere";
    math::Vec3 size{0.005f, 0.005f, 0.005f};
};

struct MjcfDefaultClass {
    MjcfJointDefaults joint;
    MjcfGeneralDefaults general;
    MjcfGeomDefaults geom;
    MjcfSiteDefaults site;
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

// An MJCF <hfield> asset: its grid, size (radius_x radius_y elevation_z base_z) and inline elevation data.
struct MjcfHeightField {
    uint32_t           nrow = 0u, ncol = 0u;
    float              size[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> elevation;
};

// A named MJCF <site>: the body it rides on plus its body-local frame. A sensor
// that references the site mounts on that body with this frame as its offset.
struct MjcfSite {
    scene::BodyId body = scene::kInvalidBody;
    math::Transform local = math::Transform::Identity();
    sensor::TactileConfig region;
};

struct MjcfParseContext {
    std::unordered_map<std::string, scene::BodyId> body_ids;
    std::unordered_map<std::string, scene::JointId> joint_ids;
    std::unordered_map<std::string, scene::MaterialId> material_ids;
    std::unordered_map<std::string, std::string> texture_files;
    std::unordered_map<std::string, MjcfMeshAsset> mesh_assets;
    std::unordered_map<std::string, MjcfHeightField> height_fields;
    // The static body <worldbody> geoms ride, once there are any.
    scene::BodyId world_body = scene::kInvalidBody;
    // Named geoms only: resolves a <contact><pair geom1=/geom2=> name to the
    // ShapeId returned by AddCollisionShape (v0.8 C1b).
    std::unordered_map<std::string, scene::ShapeId> geom_ids;
    std::unordered_map<std::string, MjcfSite> site_ids;
    // A hinged body's link frame sits on its joint anchor (MJCF joint pos), so what it authors in its
    // own frame (geoms, sites, inertial, children, ...) moves by this shift: minus the anchor.
    std::unordered_map<scene::BodyId, math::Vec3> anchor_shift;
    MjcfDefaults defaults;
    float angle_scale = 0.017453292519943295f;
    std::string euler_sequence = "xyz";
    int inertia_from_geom = -1;  // -1 auto, 0 false, 1 true
    int inertia_groups[2] = {0, 5};
};

math::Vec3 AnchorShift(const MjcfParseContext& context, scene::BodyId body) {
    const auto it = context.anchor_shift.find(body);
    return it == context.anchor_shift.end() ? math::Vec3::Zero() : it->second;
}

scene::ShapeType MjcfGeomType(const char* type_str) {
    if (!type_str) {
        return scene::ShapeType::Sphere;
    }
    const std::string t(type_str);
    if (t == "sphere") {
        return scene::ShapeType::Sphere;
    }
    if (t == "capsule") {
        return scene::ShapeType::Capsule;
    }
    if (t == "cylinder") {
        return scene::ShapeType::Cylinder;
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
    throw std::runtime_error("MJCF: unsupported geom type '" + t + "'");
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
    throw std::runtime_error("MJCF: unsupported joint type '" + t + "'");
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
    if (joint_elem->Attribute("range")) {
        ParseRange(joint_elem->Attribute("range"),
                   defaults->joint.lower_limit,
                   defaults->joint.upper_limit);
        defaults->joint.has_range = true;
    }
    defaults->joint.limited =
        ParseLimited(joint_elem->Attribute("limited"), defaults->joint.limited);
    joint_elem->QueryFloatAttribute("damping", &defaults->joint.damping);
    joint_elem->QueryFloatAttribute("armature", &defaults->joint.armature);
    joint_elem->QueryFloatAttribute("frictionloss", &defaults->joint.frictionloss);
    ParseActuatorForceRange(joint_elem, defaults->joint.actuator_force_limit);
    joint_elem->QueryFloatAttribute("stiffness", &defaults->joint.stiffness);
    joint_elem->QueryFloatAttribute("springref", &defaults->joint.spring_reference);
    joint_elem->QueryFloatAttribute("margin", &defaults->joint.limit_margin);
    ParseFloatList(joint_elem->Attribute("solreflimit"), defaults->joint.limit_reference.solref, 2);
    ParseFloatList(joint_elem->Attribute("solimplimit"), defaults->joint.limit_reference.solimp, 5);
    ParseFloatList(joint_elem->Attribute("solreffriction"), defaults->joint.friction_reference.solref, 2);
    ParseFloatList(joint_elem->Attribute("solimpfriction"), defaults->joint.friction_reference.solimp, 5);
}

void ApplyGeneralDefault(tinyxml2::XMLElement* general_elem, MjcfDefaultClass* defaults) {
    if (general_elem == nullptr || defaults == nullptr) {
        return;
    }
    general_elem->QueryFloatAttribute("gear", &defaults->general.gear);
    ParseRange(general_elem->Attribute("forcerange"),
               defaults->general.force_limit,
               defaults->general.force_limit);
    ParseFloatList(general_elem->Attribute("gainprm"), &defaults->general.gain, 1);
    ParseFloatList(general_elem->Attribute("biasprm"), defaults->general.bias, 3);
    if (const char* bias = general_elem->Attribute("biastype"))
        defaults->general.affine = std::string(bias) == "affine";
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
void ApplyFrameDefault(const tinyxml2::XMLElement* element, MjcfFrameDefaults* frame) {
    if (const char* pos = element->Attribute("pos")) frame->position = ParseVec3(pos);
    uint32_t rotations = 0u;
    for (const char* tag : {"quat", "axisangle", "euler", "xyaxes", "zaxis"}) {
        if (const char* value = element->Attribute(tag)) {
            frame->orientation_tag = tag;
            frame->orientation_value = value;
            ++rotations;
        }
    }
    if (rotations > 1u) throw std::runtime_error("MJCF: multiple orientation attributes");
    if (const char* fromto = element->Attribute("fromto")) frame->fromto = fromto;
}

void ApplyGeomDefault(const tinyxml2::XMLElement* geom_elem, MjcfDefaultClass* defaults) {
    if (geom_elem == nullptr || defaults == nullptr) {
        return;
    }
    MjcfGeomDefaults& g = defaults->geom;
    ApplyFrameDefault(geom_elem, &g);
    if (const char* size = geom_elem->Attribute("size")) {
        float values[] = {g.size.x, g.size.y, g.size.z};
        if (ParseFloatList(size, values, 3) == 0) throw std::runtime_error("MJCF: invalid geom size");
        g.size = {values[0], values[1], values[2]};
        g.has_size = true;
    }
    if (const char* material = geom_elem->Attribute("material")) g.material = material;

    if (const char* type_str = geom_elem->Attribute("type")) {
        g.type = type_str;
        g.has_type = true;
    }
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
    ParseFloatList(geom_elem->Attribute("friction"), g.friction, 3);
    geom_elem->QueryFloatAttribute("density", &g.density);
    geom_elem->QueryFloatAttribute("mass", &g.mass);
}

void ApplySiteDefault(const tinyxml2::XMLElement* element, MjcfSiteDefaults* site) {
    if (!element) return;
    if (const char* type = element->Attribute("type")) site->type = type;
    float size[] = {site->size.x, site->size.y, site->size.z};
    ParseFloatList(element->Attribute("size"), size, 3);
    site->size = {size[0], size[1], size[2]};
    ApplyFrameDefault(element, site);
}

math::Vec3 FrameUnitVector(math::Vec3 v) {
    const float length = v.Length();
    if (!(length > 0.0f) || !std::isfinite(length)) throw std::runtime_error("MJCF: invalid orientation axis");
    return v / length;
}

math::Quat FrameZRotation(math::Vec3 direction) {
    const auto z = FrameUnitVector(direction);
    return z.z <= -1.0f ? math::Quat{0.0f, 1.0f, 0.0f, 0.0f} :
        math::Quat{1.0f + z.z, -z.y, z.x, 0.0f}.Normalized();
}

math::Quat FrameOrientation(const MjcfFrameDefaults& site, const MjcfParseContext& context) {
    if (site.orientation_tag.empty()) return math::Quat::Identity();
    float values[6]{};
    const int required = site.orientation_tag == "xyaxes" ? 6 :
        site.orientation_tag == "euler" || site.orientation_tag == "zaxis" ? 3 : 4;
    if (ParseFloatList(site.orientation_value.c_str(), values, required) != required)
        throw std::runtime_error("MJCF: incomplete orientation");
    for (float value : values) if (!std::isfinite(value)) throw std::runtime_error("MJCF: invalid orientation");
    if (site.orientation_tag == "quat") {
        const math::Quat q{values[0], values[1], values[2], values[3]};
        if (!(q.Norm() > 0.0f)) throw std::runtime_error("MJCF: zero quaternion");
        return q.Normalized();
    }
    if (site.orientation_tag == "axisangle")
        return math::Quat::FromAxisAngle(FrameUnitVector({values[0], values[1], values[2]}), values[3] * context.angle_scale);
    if (site.orientation_tag == "zaxis") return FrameZRotation({values[0], values[1], values[2]});
    if (site.orientation_tag == "xyaxes") {
        const auto x = FrameUnitVector({values[0], values[1], values[2]});
        const math::Vec3 raw_y{values[3], values[4], values[5]};
        const auto y = FrameUnitVector(raw_y - x * raw_y.Dot(x));
        const auto z = x.Cross(y);
        const double matrix[3][3] = {{x.x, y.x, z.x}, {x.y, y.y, z.y}, {x.z, y.z, z.z}};
        return QuatFromMatrix(matrix);
    }
    if (context.euler_sequence.size() != 3u) throw std::runtime_error("MJCF: invalid eulerseq");
    auto q = math::Quat::Identity();
    for (uint32_t axis = 0u; axis < 3u; ++axis) {
        const char code = context.euler_sequence[axis];
        math::Vec3 vector{};
        if (code == 'x' || code == 'X') vector.x = 1.0f;
        else if (code == 'y' || code == 'Y') vector.y = 1.0f;
        else if (code == 'z' || code == 'Z') vector.z = 1.0f;
        else throw std::runtime_error("MJCF: invalid eulerseq axis");
        const auto rotation = math::Quat::FromAxisAngle(vector, values[axis] * context.angle_scale);
        q = code >= 'a' && code <= 'z' ? q * rotation : rotation * q;
    }
    return q.Normalized();
}

MjcfSite ResolveSite(const MjcfSiteDefaults& description, scene::BodyId body, const MjcfParseContext& context) {
    MjcfSite site;
    site.body = body;
    site.local.position = description.position;
    site.local.rotation = FrameOrientation(description, context);
    auto& region = site.region;
    region.size = description.size;
    if (description.type == "sphere") region.shape = sensor::ContactRegionShape::Sphere;
    else if (description.type == "box") region.shape = sensor::ContactRegionShape::Box;
    else if (description.type == "ellipsoid") region.shape = sensor::ContactRegionShape::Ellipsoid;
    else if (description.type == "capsule") region.shape = sensor::ContactRegionShape::Capsule;
    else if (description.type == "cylinder") region.shape = sensor::ContactRegionShape::Cylinder;
    else throw std::runtime_error("MJCF: unsupported site type '" + description.type + "'");
    if (!description.fromto.empty()) {
        float coordinates[6];
        if (ParseFloatList(description.fromto.c_str(), coordinates, 6) != 6 || description.type == "sphere")
            throw std::runtime_error("MJCF: invalid site fromto");
        const math::Vec3 a{coordinates[0], coordinates[1], coordinates[2]};
        const math::Vec3 b{coordinates[3], coordinates[4], coordinates[5]};
        site.local.position = (a + b) * 0.5f;
        site.local.rotation = FrameZRotation(b - a);
        const float half_height = (b - a).Length() * 0.5f;
        if (description.type == "capsule" || description.type == "cylinder") region.size.y = half_height;
        else region.size.z = half_height;
    }
    if (description.type == "sphere") region.size.y = region.size.z = 0.0f;
    if (description.type == "capsule" || description.type == "cylinder") region.size.z = 0.0f;
    site.local.position = site.local.position + AnchorShift(context, body);
    return site;
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
    ApplySiteDefault(default_elem->FirstChildElement("site"), &current.site);

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

// Map an MJCF <sensor> tag to the unified sensor kind. An unrecognized tag is
// rejected by the caller, not silently coerced.
bool MjcfSensorType(const std::string& tag, scene::SensorType& out) {
    if (tag == "rangefinder") { out = scene::SensorType::RangeScan; return true; }
    if (tag == "velocimeter") { out = scene::SensorType::LinearVelocity; return true; }
    if (tag == "accelerometer" || tag == "gyro") {
        out = scene::SensorType::Imu; return true;
    }
    if (tag == "force" || tag == "torque") { out = scene::SensorType::ForceTorque; return true; }
    if (tag == "touch") { out = scene::SensorType::Contact; return true; }
    if (tag == "framepos" || tag == "framequat" ||
        tag == "framelinvel" || tag == "frameangvel" ||
        tag == "framexaxis" || tag == "frameyaxis" || tag == "framezaxis") {
        out = scene::SensorType::FramePose; return true;
    }
    if (tag == "jointpos" || tag == "jointvel") {
        out = scene::SensorType::JointState; return true;
    }
    return false;
}

void ParseMaterials(tinyxml2::XMLElement* mujoco,
                    scene::SceneIR& scene,
                    MjcfParseContext& context,
                    const std::filesystem::path& base_dir) {
    auto* asset = mujoco->FirstChildElement("asset");
    if (!asset) {
        return;
    }

    for (auto* texture = asset->FirstChildElement("texture");
         texture != nullptr;
         texture = texture->NextSiblingElement("texture")) {
        const char* name = texture->Attribute("name");
        const char* file = texture->Attribute("file");
        if (name == nullptr || file == nullptr || file[0] == '\0') {
            continue;
        }
        context.texture_files[name] = (base_dir / file).lexically_normal().string();
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
        if (const char* texture_name = material->Attribute("texture")) {
            const auto it = context.texture_files.find(texture_name);
            if (it != context.texture_files.end()) {
                record.albedo_map = it->second;
                // Image materials prefer authored UVs; meshes without UVs still
                // use the shader's triplanar fallback.
                record.triplanar = false;
            }
        }

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
        if (!file) {
            continue;
        }
        // MuJoCo rule: when <mesh> omits `name`, the mesh name DEFAULTS to the
        // file's basename without extension (so a geom's `mesh="base_0"` resolves
        // a <mesh file="base_0.obj"/>). go2's asset block relies on this; without
        // it the 16 visual meshes never load and the visual-mesh cook is empty.
        std::string mesh_name = name ? std::string(name)
                                     : std::filesystem::path(file).stem().string();

        MjcfMeshAsset record;
        // std::filesystem::operator/ resets to the right operand if it is
        // absolute, so an absolute meshdir or file path is handled correctly.
        record.resolved_path = (base_dir / meshdir / file).string();
        if (const char* scale = mesh->Attribute("scale")) {
            record.scale = ParseVec3(scale);
        }
        context.mesh_assets[mesh_name] = std::move(record);
    }
}

// Resolve <asset><hfield> entries with inline elevation data; file-backed and runtime-filled fields fail loud.
void ParseHeightFieldAssets(tinyxml2::XMLElement* mujoco, MjcfParseContext& context) {
    auto* asset = mujoco->FirstChildElement("asset");
    if (!asset) {
        return;
    }
    for (auto* field = asset->FirstChildElement("hfield"); field != nullptr;
         field = field->NextSiblingElement("hfield")) {
        const std::string name = field->Attribute("name") ? field->Attribute("name") : "";
        if (field->Attribute("file") || !field->Attribute("elevation")) {
            throw std::runtime_error("MJCF: hfield '" + name +
                                     "' needs inline elevation data; file and runtime fields are not supported");
        }
        MjcfHeightField record;
        record.nrow = field->UnsignedAttribute("nrow", 0u);
        record.ncol = field->UnsignedAttribute("ncol", 0u);
        ParseFloatList(field->Attribute("size"), record.size, 4);
        std::istringstream ss(field->Attribute("elevation"));
        for (float v; ss >> v;) record.elevation.push_back(v);
        if (record.nrow < 2u || record.ncol < 2u || record.size[0] <= 0.0f || record.size[1] <= 0.0f ||
            record.elevation.size() != static_cast<size_t>(record.nrow) * record.ncol) {
            throw std::runtime_error("MJCF: hfield '" + name + "' has an invalid grid");
        }
        context.height_fields[name] = std::move(record);
    }
}

// A worldbody hfield geom becomes the scene's terrain as MuJoCo places it: the grid spans twice the radii
// around the geom, row r at y = -radius_y + r * cell_y, and the elevation, normalized to [0,1], rises
// elevation_z above the geom's z.
void AddHeightFieldTerrain(tinyxml2::XMLElement* geom, const std::string& name, scene::SceneIR& scene,
                           const MjcfParseContext& context) {
    const char* field_name = geom->Attribute("hfield");
    const auto it = field_name ? context.height_fields.find(field_name) : context.height_fields.end();
    if (it == context.height_fields.end()) {
        throw std::runtime_error("MJCF: hfield geom '" + name + "' references no inline <hfield>");
    }
    const math::Quat q = ParseQuat(geom->Attribute("quat"));
    if (geom->Attribute("euler") || geom->Attribute("axisangle") || geom->Attribute("xyaxes") ||
        geom->Attribute("zaxis") || std::fabs(std::fabs(q.w) - 1.0f) > 1e-6f) {
        throw std::runtime_error("MJCF: hfield geom '" + name + "' is rotated; the terrain grid is axis-aligned");
    }
    if (scene.TerrainCount() != 0u) {
        throw std::runtime_error("MJCF: hfield geom '" + name + "' is a second terrain; a scene holds one");
    }
    const MjcfHeightField& field = it->second;
    const auto [low, high] = std::minmax_element(field.elevation.begin(), field.elevation.end());
    const float span = *high - *low;
    const math::Vec3 pos = ParseVec3(geom->Attribute("pos"));
    scene::TerrainRecord terrain;
    terrain.name = name.empty() ? std::string(field_name) : name;
    terrain.nrow = field.nrow;
    terrain.ncol = field.ncol;
    terrain.cell = 2.0f * field.size[0] / static_cast<float>(field.ncol - 1u);
    terrain.cell_y = 2.0f * field.size[1] / static_cast<float>(field.nrow - 1u);
    terrain.origin = {pos.x - field.size[0], pos.y - field.size[1], pos.z};
    terrain.base_z = pos.z;
    terrain.height_scale = field.size[2];
    terrain.heights.reserve(field.elevation.size());
    for (const float v : field.elevation) terrain.heights.push_back(span > 0.0f ? (v - *low) / span : 0.0f);
    scene.AddTerrain(std::move(terrain));
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

scene::BodyId AddStaticBody(std::string name, scene::SceneIR& scene) {
    scene::RigidBodyRecord body;
    body.name = std::move(name);
    body.is_static = true;
    body.mass = 0.0f;
    body.inertia = math::Vec3::Zero();
    return scene.AddRigidBody(std::move(body));
}

// True when a root body's joints hinge or slide against the world instead of floating.
bool JointedToWorld(tinyxml2::XMLElement* body_elem, const MjcfParseContext& context, const char* child_class) {
    for (auto* joint = body_elem->FirstChildElement("joint"); joint != nullptr;
         joint = joint->NextSiblingElement("joint")) {
        const char* type = joint->Attribute("type");
        const char* joint_class = joint->Attribute("class");
        const scene::JointType kind = type ? MjcfJointType(type)
            : DefaultClassOrRoot(context.defaults, joint_class ? joint_class : child_class).joint.type;
        if (kind != scene::JointType::Free) return true;
    }
    return false;
}

// Defined below ParseBody; ParseBody nests body-attached <light> elements.
void ParseLightElement(tinyxml2::XMLElement* light,
                       scene::BodyId attached_body,
                       scene::SceneIR& scene);

// A body's <geom> children become its shapes; a non-null `mass` also sums their inertia.
void ParseGeoms(tinyxml2::XMLElement* body_elem,
                scene::BodyId body_id,
                scene::SceneIR& scene,
                MjcfParseContext& context,
                const char* child_class,
                BodyMass* mass) {
    for (auto* geom = body_elem->FirstChildElement("geom");
         geom != nullptr;
         geom = geom->NextSiblingElement("geom")) {

        scene::CollisionShapeRecord shape;
        shape.law_source = constraint::LawSource::MuJoCo;
        shape.body_id = body_id;
        shape.name = geom->Attribute("name") ? geom->Attribute("name") : "";

        // Resolve the geom class FIRST so the type can fall back to the class
        // default's type (M8.5 T5: e.g. go2's class="visual" sets type="mesh"
        // while the geom element omits it). The contact-attr precedence below
        // re-reads `gd` for the rest of the fields (unchanged).
        const char* geom_class = geom->Attribute("class");
        const MjcfGeomDefaults& gd =
            DefaultClassOrRoot(context.defaults, geom_class ? geom_class : child_class).geom;
        // A worldbody hfield is the terrain, not a collision shape of its body.
        const char* type_name = geom->Attribute("type") ? geom->Attribute("type")
                              : gd.has_type ? gd.type.c_str() : nullptr;
        if (type_name != nullptr && std::string(type_name) == "hfield") {
            if (body_id != context.world_body) {
                throw std::runtime_error("MJCF: hfield geom '" + shape.name +
                                         "' must sit in <worldbody>; the terrain is static");
            }
            AddHeightFieldTerrain(geom, shape.name, scene, context);
            continue;
        }
        // An explicit type overrides the class default; MJCF defaults to a sphere.
        if (const char* geom_type = geom->Attribute("type")) {
            shape.type = MjcfGeomType(geom_type);
        } else if (gd.has_type) {
            shape.type = MjcfGeomType(gd.type.c_str());
        } else {
            shape.type = MjcfGeomType(nullptr);
        }

        const char* material_name = geom->Attribute("material");
        if (!material_name && !gd.material.empty()) material_name = gd.material.c_str();
        if (material_name) {
            const auto it = context.material_ids.find(material_name);
            if (it != context.material_ids.end()) {
                shape.material_id = it->second;
            }
        } else if (const char* geom_rgba = geom->Attribute("rgba")) {
            // A geom with an inline rgba but no named <material>: synthesize a
            // MaterialRecord so the visual color survives the import (it reaches
            // RenderMaterial through the SceneIR facade). Deduped by the rgba
            // string so repeated identical colors share one material.
            const auto found = context.material_ids.find(geom_rgba);
            if (found != context.material_ids.end()) {
                shape.material_id = found->second;
            } else {
                scene::MaterialRecord record;
                record.name = std::string("geom_rgba:") + geom_rgba;
                std::istringstream rgba_ss(geom_rgba);
                rgba_ss >> record.base_color.x >> record.base_color.y
                        >> record.base_color.z >> record.alpha;
                const scene::MaterialId mid = scene.AddMaterial(std::move(record));
                context.material_ids[geom_rgba] = mid;
                shape.material_id = mid;
            }
        }

        MjcfFrameDefaults frame = gd;
        ApplyFrameDefault(geom, &frame);
        shape.local_transform.position = frame.position;
        shape.local_transform.rotation = FrameOrientation(frame, context);

        const char* size_attr = geom->Attribute("size");
        if (size_attr || gd.has_size) {
            float size[] = {gd.size.x, gd.size.y, gd.size.z};
            if (size_attr && ParseFloatList(size_attr, size, 3) == 0)
                throw std::runtime_error("MJCF: invalid geom size");
            const math::Vec3 sz{size[0], size[1], size[2]};
            if (shape.type == scene::ShapeType::Box || shape.type == scene::ShapeType::Plane) {
                shape.half_extents = sz;
            } else if (shape.type == scene::ShapeType::Sphere) {
                shape.radius = sz.x;
            } else if (shape.type == scene::ShapeType::Capsule || shape.type == scene::ShapeType::Cylinder) {
                shape.radius = sz.x;
                if (sz.y > 0.0f) {
                    shape.half_height = sz.y;
                }
            }
        }
        if (!frame.fromto.empty()) {
            float points[6];
            const bool axial = shape.type == scene::ShapeType::Capsule || shape.type == scene::ShapeType::Cylinder;
            if (ParseFloatList(frame.fromto.c_str(), points, 6) != 6 ||
                (!axial && shape.type != scene::ShapeType::Box))
                throw std::runtime_error("MJCF: invalid geom fromto");
            const math::Vec3 a{points[0], points[1], points[2]};
            const math::Vec3 b{points[3], points[4], points[5]};
            shape.local_transform.position = (a + b) * 0.5f;
            shape.local_transform.rotation = FrameZRotation(b - a);
            const float half_length = (b - a).Length() * 0.5f;
            if (axial) shape.half_height = half_length;
            else shape.half_extents.z = half_length;
        }
        shape.local_transform.position = shape.local_transform.position + AnchorShift(context, body_id);

        // nuka:decompose="auto|force|skip" + optional nuka:decompose:max_pieces
        // on a mesh geom (v0.7 p06).
        if (shape.type == scene::ShapeType::TriMesh) {
            bool as_hull = false;
            if (const char* mode = geom->Attribute("nuka:mesh_contact")) {
                if (std::string(mode) == "sdf")
                    shape.mesh_contact = scene::CollisionShapeRecord::MeshContact::Sdf;
                else if (std::string(mode) == "hull")
                    as_hull = true;
                else if (std::string(mode) != "ogc")
                    throw std::runtime_error("MJCF: invalid mesh contact mode");
            }
            if (geom->Attribute("nuka:mesh_triangle_limit")) {
                unsigned int count = 0u;
                if (geom->QueryUnsignedAttribute("nuka:mesh_triangle_limit", &count) !=
                        tinyxml2::XML_SUCCESS || count == 0u)
                    throw std::runtime_error("MJCF: invalid mesh triangle limit");
                shape.mesh_triangle_limit = count;
            }
            if (geom->Attribute("nuka:mesh_error_limit")) {
                float limit = 0.0f;
                if (geom->QueryFloatAttribute("nuka:mesh_error_limit", &limit) !=
                        tinyxml2::XML_SUCCESS || !(limit > 0.0f && limit <= 0.001f))
                    throw std::runtime_error("MJCF: invalid mesh error limit");
                shape.mesh_error_limit = limit;
            }
            shape.decompose_mode = DecomposeModeFromToken(geom->Attribute("nuka:decompose"));
            if (const char* orientation = geom->Attribute("nuka:mesh_orientation")) {
                if (std::string(orientation) != "automatic" && std::string(orientation) != "outward")
                    throw std::runtime_error("MJCF mesh orientation must be automatic or outward");
                shape.mesh_oriented = std::string(orientation) == "outward";
            }
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
                // Normals transform by the inverse-transpose of the scale, then renormalize.
                for (std::size_t i = 0; i + 2 < geo.normals.size(); i += 3) {
                    float nx = geo.normals[i + 0] * (s.x != 0.0f ? 1.0f / s.x : 0.0f);
                    float ny = geo.normals[i + 1] * (s.y != 0.0f ? 1.0f / s.y : 0.0f);
                    float nz = geo.normals[i + 2] * (s.z != 0.0f ? 1.0f / s.z : 0.0f);
                    const float len = std::sqrt(nx * nx + ny * ny + nz * nz);
                    const float inv = len > 0.0f ? 1.0f / len : 0.0f;
                    geo.normals[i + 0] = nx * inv;
                    geo.normals[i + 1] = ny * inv;
                    geo.normals[i + 2] = nz * inv;
                }
                shape.mesh_vertices = std::move(geo.vertices);
                shape.mesh_normals = std::move(geo.normals);
                shape.mesh_uvs = std::move(geo.uvs);
                shape.mesh_indices = std::move(geo.indices);
            }
            // MuJoCo collides a mesh geom as the hull of its vertices, capped at maxhullvert corners.
            if (as_hull) {
                int limit = 0x7fffffff;
                geom->QueryIntAttribute("nuka:hull_vertex_limit", &limit);
                cooker::ConvexPiece hull = cooker::ComputeConvexHull(shape.mesh_vertices.data(),
                    static_cast<uint32_t>(shape.mesh_vertices.size() / 3u), static_cast<uint32_t>(limit));
                if (hull.volume > 0.0f) {
                    shape.type = scene::ShapeType::ConvexHull;
                    shape.mesh_vertices = std::move(hull.vertices);
                    shape.mesh_indices = std::move(hull.indices);
                    shape.mesh_normals.clear();
                    shape.mesh_uvs.clear();
                }
            }
        }

        // -- C1b collision/contact attributes --------------------------------
        // Precedence (low -> high): the CollisionShapeRecord's C1a defaults
        // (already MuJoCo's) < the resolved <default> class's <geom> attrs
        // (presence-gated) < the geom's own explicit attrs. `gd` (the resolved
        // geom-class default) was bound above for the type fallback; reuse it.
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
        shape.friction_mu = gd.friction[0];
        shape.friction_spin = gd.friction[1];
        shape.friction_roll = gd.friction[2];

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
        float friction[3] = {shape.friction_mu, shape.friction_spin, shape.friction_roll};
        ParseFloatList(geom->Attribute("friction"), friction, 3);
        shape.friction_mu = friction[0];
        shape.friction_spin = friction[1];
        shape.friction_roll = friction[2];

        float density = gd.density;
        float geom_mass = gd.mass;
        geom->QueryFloatAttribute("density", &density);
        geom->QueryFloatAttribute("mass", &geom_mass);
        if (mass && shape.collision_group >= context.inertia_groups[0] &&
            shape.collision_group <= context.inertia_groups[1])
            AddGeomMass(shape, density, geom_mass, *mass);

        const std::string geom_name = shape.name;
        const scene::ShapeId shape_id = scene.AddCollisionShape(std::move(shape));
        if (!geom_name.empty()) {
            context.geom_ids[geom_name] = shape_id;
        }
    }
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
    // Articulations root at static bodies, so a root hinge or slide turns against a static anchor.
    if (parent_id == scene::kInvalidBody && JointedToWorld(body_elem, context, child_class)) {
        parent_id = AddStaticBody(body_name + "_anchor", scene);
    }

    scene::RigidBodyRecord rec;
    rec.name = body_name;
    rec.parent_id = parent_id;
    rec.is_static = parent_id == scene::kInvalidBody &&
        !body_elem->FirstChildElement("joint") && !body_elem->FirstChildElement("freejoint");
    // Without <inertial> the geoms supply the mass below; never leak the SceneIR
    // construction placeholder into imported physics.
    rec.mass = 0.0f;
    rec.inertia = math::Vec3::Zero();
    if (const char* pos_attr = body_elem->Attribute("pos")) {
        rec.local_transform.position = ParseVec3(pos_attr);
    }
    if (const char* quat_attr = body_elem->Attribute("quat")) {
        rec.local_transform.rotation = ParseQuat(quat_attr);
    }
    // MuJoCo turns a hinge or ball about its pos in the body frame; the link frame moves onto that anchor.
    // Several joints apply in order (mj_kinematics): every joint but the last moves a massless link on its
    // own anchor, and the body hangs off the last one. A slide keeps the previous anchor (its pos moves nothing).
    std::vector<std::pair<tinyxml2::XMLElement*, math::Vec3>> chain;
    math::Vec3 anchor = math::Vec3::Zero();
    for (auto* joint = body_elem->FirstChildElement("joint"); joint != nullptr;
         joint = joint->NextSiblingElement("joint")) {
        const char* joint_class = joint->Attribute("class");
        const scene::JointType type = joint->Attribute("type") ? MjcfJointType(joint->Attribute("type"))
            : DefaultClassOrRoot(context.defaults, joint_class ? joint_class : child_class).joint.type;
        if (type == scene::JointType::Free || type == scene::JointType::Fixed) continue;
        if (type != scene::JointType::Prismatic && joint->Attribute("pos")) anchor = ParseVec3(joint->Attribute("pos"));
        chain.emplace_back(joint, anchor);
    }
    rec.local_transform.position = rec.local_transform.position + AnchorShift(context, parent_id);
    std::unordered_map<tinyxml2::XMLElement*, std::pair<scene::BodyId, scene::BodyId>> joint_bodies;
    math::Vec3 previous = math::Vec3::Zero();
    for (size_t k = 0; k + 1 < chain.size(); ++k) {
        scene::RigidBodyRecord link;
        const char* jname = chain[k].first->Attribute("name");
        link.name = body_name + "/" + (jname ? jname : "joint" + std::to_string(k));
        link.parent_id = parent_id;
        link.local_transform.position = k == 0 ? rec.local_transform.position +
            rec.local_transform.rotation.Rotate(chain[k].second) : chain[k].second - previous;
        link.local_transform.rotation = k == 0 ? rec.local_transform.rotation : math::Quat::Identity();
        link.mass = 1.0e-6f;
        link.inertia = math::Vec3{1.0e-9f, 1.0e-9f, 1.0e-9f};
        const scene::BodyId link_id = scene.AddRigidBody(std::move(link));
        joint_bodies[chain[k].first] = {parent_id, link_id};
        parent_id = link_id;
        previous = chain[k].second;
    }
    if (chain.size() > 1) {
        rec.parent_id = parent_id;
        rec.local_transform.position = anchor - previous;
        rec.local_transform.rotation = math::Quat::Identity();
    } else {
        rec.local_transform.position = rec.local_transform.position + rec.local_transform.rotation.Rotate(anchor);
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
        } else if (const char* full = inertial->Attribute("fullinertia")) {
            // fullinertia = [Ixx Iyy Izz Ixy Ixz Iyz] in the body frame.
            // Diagonalize to principal moments + a body->principal rotation
            // (MuJoCo semantics); the rotation folds into the inertial frame so
            // the FULL tensor is physical (RotateDiagonalInertia rebuilds it).
            float v[6] = {1.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
            const int n = ParseFloatList(full, v, 6);
            if (n >= 6) {
                math::Vec3 principal;
                math::Quat rot;
                DiagonalizeInertia(v, principal, rot);
                rec.inertia = principal;
                rec.inertial_transform.rotation = rot;
            } else if (n >= 3) {
                rec.inertia = math::Vec3{v[0], v[1], v[2]};
            }
        }
    }
    rec.inertial_transform.position = rec.inertial_transform.position - anchor;

    // compiler inertiafromgeom: auto fills bodies without <inertial>, true replaces it.
    const bool from_geoms = !rec.is_static && (context.inertia_from_geom == 1 ||
        (context.inertia_from_geom == -1 && !body_elem->FirstChildElement("inertial")));
    const scene::BodyId body_id = scene.AddRigidBody(std::move(rec));
    context.body_ids[body_name] = body_id;
    if (!chain.empty()) joint_bodies[chain.back().first] = {parent_id, body_id};
    if (anchor.x != 0.0f || anchor.y != 0.0f || anchor.z != 0.0f) {
        context.anchor_shift[body_id] = math::Vec3::Zero() - anchor;
    }
    BodyMass mass;
    ParseGeoms(body_elem, body_id, scene, context, child_class, from_geoms ? &mass : nullptr);
    if (from_geoms) SetBodyMass(mass, scene.GetBodyMut(body_id));

    if (const auto* free_joint = body_elem->FirstChildElement("freejoint")) {
        if (parent_id != scene::kInvalidBody || body_elem->FirstChildElement("joint") ||
            free_joint->NextSiblingElement("freejoint"))
            throw std::runtime_error("MJCF freejoint must be the only joint of a root body");
        scene::JointRecord joint;
        const char* name = free_joint->Attribute("name");
        joint.name = name ? name : body_name + "/free";
        joint.type = scene::JointType::Free;
        joint.child_body = body_id;
        const scene::JointId id = scene.AddJoint(std::move(joint));
        context.joint_ids[scene.GetJoint(id).name] = id;
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
        const auto link = joint_bodies.find(joint);
        jrec.parent_body = link != joint_bodies.end() ? link->second.first : parent_id;
        jrec.child_body = link != joint_bodies.end() ? link->second.second : body_id;
        jrec.type = defaults.joint.type;
        jrec.axis = defaults.joint.axis;
        jrec.lower_limit = defaults.joint.lower_limit;
        jrec.upper_limit = defaults.joint.upper_limit;
        bool has_range = defaults.joint.has_range;
        int limited = defaults.joint.limited;
        jrec.damping = defaults.joint.damping;
        jrec.armature = defaults.joint.armature;
        jrec.frictionloss = defaults.joint.frictionloss;
        jrec.actuator_force_limit = defaults.joint.actuator_force_limit;
        jrec.stiffness = defaults.joint.stiffness;
        jrec.spring_reference = defaults.joint.spring_reference;
        jrec.law_source = constraint::LawSource::MuJoCo;
        jrec.limit_reference = defaults.joint.limit_reference;
        jrec.friction_reference = defaults.joint.friction_reference;
        jrec.limit_margin = defaults.joint.limit_margin;
        // Nuka extension: author the cooked q0 explicitly. MJCF keyframes are
        // intentionally not imported, so this keeps initial FK/colliders aligned
        // before the first physics step without changing MuJoCo joint semantics.
        joint->QueryFloatAttribute("nuka:initial_position", &jrec.initial_position);
        // Explicit dynamics override class defaults; omitted attributes retain them.
        joint->QueryFloatAttribute("damping", &jrec.damping);
        joint->QueryFloatAttribute("armature", &jrec.armature);
        joint->QueryFloatAttribute("frictionloss", &jrec.frictionloss);
        ParseActuatorForceRange(joint, jrec.actuator_force_limit);
        joint->QueryFloatAttribute("stiffness", &jrec.stiffness);
        joint->QueryFloatAttribute("springref", &jrec.spring_reference);
        joint->QueryFloatAttribute("margin", &jrec.limit_margin);
        ParseFloatList(joint->Attribute("solreflimit"), jrec.limit_reference.solref, 2);
        ParseFloatList(joint->Attribute("solimplimit"), jrec.limit_reference.solimp, 5);
        ParseFloatList(joint->Attribute("solreffriction"), jrec.friction_reference.solref, 2);
        ParseFloatList(joint->Attribute("solimpfriction"), jrec.friction_reference.solimp, 5);
        if (joint->Attribute("type")) {
            jrec.type = MjcfJointType(joint->Attribute("type"));
        }
        if (const char* axis_attr = joint->Attribute("axis")) {
            jrec.axis = ParseVec3(axis_attr);
        }
        // The anchor is the link origin (see above); a slide's pos does not change its motion.
        if (joint->Attribute("range")) {
            ParseRange(joint->Attribute("range"), jrec.lower_limit, jrec.upper_limit);
            has_range = true;
        }
        limited = ParseLimited(joint->Attribute("limited"), limited);
        const bool limits_enabled = has_range && limited != 0;
        jrec.has_lower_limit = limits_enabled;
        jrec.has_upper_limit = limits_enabled;
        // A ball joint's cone limit, quaternion spring and dry friction have no articulation rows yet.
        if (jrec.type == scene::JointType::Spherical &&
            (limits_enabled || jrec.stiffness != 0.0f || jrec.frictionloss != 0.0f))
            throw std::runtime_error("MJCF: ball joint '" + jrec.name +
                                     "' has a range, stiffness or frictionloss the articulation does not model");

        const scene::JointId joint_id = scene.AddJoint(std::move(jrec));
        context.joint_ids[scene.GetJoint(joint_id).name] = joint_id;
    }

    // An omitted joint welds a child body to its parent; retain its inertia and sensor frame.
    if (parent_id != scene::kInvalidBody && !body_elem->FirstChildElement("joint") &&
        !body_elem->FirstChildElement("freejoint")) {
        scene::JointRecord joint;
        joint.name = body_name + "/fixed";
        joint.parent_body = parent_id;
        joint.child_body = body_id;
        joint.type = scene::JointType::Fixed;
        scene.AddJoint(std::move(joint));
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
        record.local_transform.position = record.local_transform.position + AnchorShift(context, body_id);
        if (const char* quat = camera->Attribute("quat")) {
            record.local_transform.rotation = ParseQuat(quat);
        }
        camera->QueryFloatAttribute("fovy", &record.vertical_fov_degrees);
        scene.AddCamera(std::move(record));
    }

    // Named sites retain their sensing volume and body-local frame.
    for (auto* site = body_elem->FirstChildElement("site");
         site != nullptr;
         site = site->NextSiblingElement("site")) {
        const char* site_name = site->Attribute("name");
        if (!site_name) {
            continue;
        }
        const char* site_class = site->Attribute("class");
        auto description = DefaultClassOrRoot(context.defaults, site_class ? site_class : child_class).site;
        ApplySiteDefault(site, &description);
        context.site_ids[site_name] = ResolveSite(description, body_id, context);
    }

    // Body-nested lights attach to this body: their local transform composes with
    // the body's live pose every frame (the general attachment path).
    for (auto* light = body_elem->FirstChildElement("light");
         light != nullptr;
         light = light->NextSiblingElement("light")) {
        ParseLightElement(light, body_id, scene);
    }

    for (auto* child = body_elem->FirstChildElement("body");
         child != nullptr;
         child = child->NextSiblingElement("body")) {
        ParseBody(child, body_id, scene, context, child_class);
    }
}

void ParseLightElement(tinyxml2::XMLElement* light,
                       scene::BodyId attached_body,
                       scene::SceneIR& scene) {
    scene::LightRecord record;
    record.name = light->Attribute("name") ? light->Attribute("name") : "light";
    record.attached_body = attached_body;
    bool directional = false;
    light->QueryBoolAttribute("directional", &directional);
    record.type = directional ? scene::LightType::Directional : scene::LightType::Point;
    if (const char* pos = light->Attribute("pos")) {
        record.local_transform.position = ParseVec3(pos);
    }
    if (const char* quat = light->Attribute("quat")) {
        record.local_transform.rotation = ParseQuat(quat);
    }
    if (const char* dir = light->Attribute("dir")) {
        // MuJoCo aims a light with `dir` (target direction of travel); convert to
        // the equivalent local rotation so downstream consumers share one frame.
        const math::Vec3 d = ParseVec3(dir);
        const float len = std::sqrt(d.x * d.x + d.y * d.y + d.z * d.z);
        if (len > 1.0e-6f) {
            const math::Vec3 n{d.x / len, d.y / len, d.z / len};
            // MuJoCo light frame: -Z aims along `dir`, +Y up (camera convention).
            math::Vec3 up_ref{0.0f, 0.0f, 1.0f};
            if (std::fabs(n.x * up_ref.x + n.y * up_ref.y + n.z * up_ref.z) > 0.99f) {
                up_ref = math::Vec3{0.0f, 1.0f, 0.0f};
            }
            math::Vec3 right{n.y * up_ref.z - n.z * up_ref.y,
                             n.z * up_ref.x - n.x * up_ref.z,
                             n.x * up_ref.y - n.y * up_ref.x};
            const float rlen = std::sqrt(right.x * right.x + right.y * right.y +
                                         right.z * right.z);
            if (rlen > 1.0e-6f) {
                right = math::Vec3{right.x / rlen, right.y / rlen, right.z / rlen};
                const math::Vec3 up{right.y * n.z - right.z * n.y,
                                    right.z * n.x - right.x * n.z,
                                    right.x * n.y - right.y * n.x};
                // Trace-based rotation-matrix -> quaternion extraction with
                // columns X->right, Y->up, Z->-dir (the -Z aims convention).
                const float m11 = right.x, m22 = up.y, m33 = n.z;
                const float tr = m11 + m22 + m33;
                math::Quat q;
                if (tr > 0.0f) {
                    const float s = std::sqrt(tr + 1.0f) * 2.0f;
                    q = math::Quat{0.25f * s,
                                   (up.z - n.y) / s,
                                   (n.x - right.z) / s,
                                   (right.y - up.x) / s};
                } else if (m11 > m22 && m11 > m33) {
                    const float s = std::sqrt(1.0f + m11 - m22 - m33) * 2.0f;
                    q = math::Quat{(up.z - n.y) / s, 0.25f * s,
                                   (up.x + right.y) / s,
                                   (n.x + right.z) / s};
                } else if (m22 > m33) {
                    const float s = std::sqrt(1.0f + m22 - m11 - m33) * 2.0f;
                    q = math::Quat{(n.x - right.z) / s, (up.x + right.y) / s,
                                   0.25f * s,
                                   (right.y + up.z) / s};
                } else {
                    const float s = std::sqrt(1.0f + m33 - m11 - m22) * 2.0f;
                    q = math::Quat{(right.y - up.x) / s, (n.x + right.z) / s,
                                   (right.y + up.z) / s, 0.25f * s};
                }
                record.local_transform.rotation = q.Normalized();
            }
        }
        // A positional light's dir attribute aims it without changing its type.
    }
    if (const char* diffuse = light->Attribute("diffuse")) {
        record.color = ParseVec3(diffuse);
    }
    light->QueryFloatAttribute("intensity", &record.intensity);
    scene.AddLight(std::move(record));
}

void ParseLights(tinyxml2::XMLElement* worldbody, scene::SceneIR& scene) {
    for (auto* light = worldbody->FirstChildElement("light");
         light != nullptr;
         light = light->NextSiblingElement("light")) {
        ParseLightElement(light, scene::kInvalidBody, scene);
    }
}

// MuJoCo's dof_M0: the joint's subtree inertia about its axis at the authored pose, plus armature.
float JointInertiaAtRest(const scene::SceneIR& scene, scene::JointId joint_id) {
    const scene::JointRecord& joint = scene.GetJoint(joint_id);
    std::vector<math::Transform> world(scene.RigidBodyCount());
    for (scene::BodyId b = 0; b < world.size(); ++b) {
        const scene::RigidBodyRecord& body = scene.GetBody(b);
        world[b] = body.parent_id == scene::kInvalidBody ? body.local_transform
                                                         : world[body.parent_id] * body.local_transform;
    }
    const math::Vec3 anchor = world[joint.child_body].position;
    const math::Vec3 axis = world[joint.child_body].rotation.Rotate(
        joint.axis * (1.0f / std::sqrt(joint.axis.LengthSq())));
    float inertia = joint.armature;
    for (scene::BodyId b = 0; b < world.size(); ++b) {
        scene::BodyId walk = b;
        while (walk != scene::kInvalidBody && walk != joint.child_body) walk = scene.GetBody(walk).parent_id;
        if (walk == scene::kInvalidBody) continue;
        const scene::RigidBodyRecord& body = scene.GetBody(b);
        if (joint.type == scene::JointType::Prismatic) {
            inertia += body.mass;
            continue;
        }
        const math::Transform frame = world[b] * body.inertial_transform;
        const math::Vec3 local = frame.rotation.Conjugate().Rotate(axis);
        const math::Vec3 arm = (frame.position - anchor).Cross(axis);
        inertia += body.inertia.x * local.x * local.x + body.inertia.y * local.y * local.y +
                   body.inertia.z * local.z * local.z + body.mass * arm.LengthSq();
    }
    return inertia;
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
        record.force_limit = defaults.general.force_limit;
        float gear = defaults.general.gear;
        float gain = defaults.general.gain;
        float bias[3] = {defaults.general.bias[0], defaults.general.bias[1], defaults.general.bias[2]};
        bool affine = defaults.general.affine;
        const std::string tag = actuator->Name() ? actuator->Name() : "";
        actuator->QueryFloatAttribute("gear", &gear);
        if (tag == "position") {
            float kp = 1.0f, kv = 0.0f;
            actuator->QueryFloatAttribute("kp", &kp);
            actuator->QueryFloatAttribute("kv", &kv);
            gain = kp;
            bias[0] = 0.0f; bias[1] = -kp; bias[2] = -kv;
            actuator->QueryFloatAttribute("dampratio", &bias[2]);
            affine = true;
        } else if (tag == "velocity") {
            float kv = 1.0f;
            actuator->QueryFloatAttribute("kv", &kv);
            gain = kv;
            bias[0] = 0.0f; bias[1] = 0.0f; bias[2] = -kv;
            affine = true;
        } else if (tag == "motor") {
            gain = 1.0f;
            affine = false;
        } else if (tag == "general") {
            ParseFloatList(actuator->Attribute("gainprm"), &gain, 1);
            ParseFloatList(actuator->Attribute("biasprm"), bias, 3);
            if (const char* type = actuator->Attribute("biastype")) affine = std::string(type) == "affine";
        }
        record.joint_id = ResolveJoint(actuator->Attribute("joint"), context);
        if (record.joint_id == scene::kInvalidJoint)
            throw std::runtime_error("MJCF: actuator '" + record.name + "' needs a known joint transmission");
        if (scene.GetJoint(record.joint_id).type == scene::JointType::Spherical)
            throw std::runtime_error("MJCF: actuator '" + record.name + "' drives a ball joint, which has no drive rows");
        // A positive velocity bias on a position servo is MuJoCo's damping ratio over the reflected inertia.
        if (affine && gain > 0.0f && gain == -bias[1] && bias[2] > 0.0f)
            bias[2] = -bias[2] * 2.0f * std::sqrt(gain * JointInertiaAtRest(scene, record.joint_id) / (gear * gear));
        // A joint transmission applies gear (gain u + b0 + b1 gear q + b2 gear qdot) to its joint.
        if (tag != "position" && tag != "velocity" && tag != "motor" && tag != "general") {
            record.type = scene::ActuatorType::Force;
            record.gain = gear;
        } else if (!affine || (bias[1] == 0.0f && bias[2] == 0.0f)) {
            record.type = scene::ActuatorType::Motor;
            record.gain = gear * gain;
        } else if (bias[1] != 0.0f) {
            if (bias[1] > 0.0f || bias[0] != 0.0f)
                throw std::runtime_error("MJCF: actuator '" + record.name + "' bias is not a position servo");
            // The bias sets the servo's stiffness and damping; the control gain only scales its target.
            record.type = scene::ActuatorType::Position;
            record.gain = -gear * gear * bias[1];
            record.damping = -gear * gear * bias[2];
        } else {
            if (bias[2] > 0.0f || bias[0] != 0.0f)
                throw std::runtime_error("MJCF: actuator '" + record.name + "' bias is not a velocity servo");
            record.type = scene::ActuatorType::Velocity;
            record.gain = -gear * gear * bias[2];
            record.damping = record.gain;
        }
        ParseRange(actuator->Attribute("forcerange"), record.force_limit, record.force_limit);
        scene.AddActuator(std::move(record));
    }
}

void ParseJointEqualities(tinyxml2::XMLElement* mujoco, scene::SceneIR& scene,
                          const MjcfParseContext& context) {
    auto* equality = mujoco->FirstChildElement("equality");
    if (equality == nullptr) return;
    for (auto* item = equality->FirstChildElement(); item != nullptr; item = item->NextSiblingElement()) {
        bool active = true;
        item->QueryBoolAttribute("active", &active);
        if (active && std::string(item->Name()) != "joint")
            throw std::runtime_error(std::string("MJCF: <equality><") + item->Name() + "> is unsupported");
    }
    for (auto* item = equality->FirstChildElement("joint");
         item != nullptr; item = item->NextSiblingElement("joint")) {
        bool active = true;
        item->QueryBoolAttribute("active", &active);
        if (!active) continue;
        const scene::JointId target = ResolveJoint(item->Attribute("joint1"), context);
        const scene::JointId source = ResolveJoint(item->Attribute("joint2"), context);
        if (target == scene::kInvalidJoint || source == scene::kInvalidJoint || target == source)
            throw std::runtime_error("MJCF: joint equality needs two distinct scalar joints");
        float coefficient[5] = {0.0f, 1.0f, 0.0f, 0.0f, 0.0f};
        if (const char* values = item->Attribute("polycoef")) {
            if (ParseFloatList(values, coefficient, 5) == 0)
                throw std::runtime_error("MJCF: invalid joint equality coefficients");
        }
        for (float value : coefficient)
            if (!std::isfinite(value))
                throw std::runtime_error("MJCF: nonfinite joint equality coefficient");
        if (coefficient[2] != 0.0f || coefficient[3] != 0.0f || coefficient[4] != 0.0f)
            throw std::runtime_error("MJCF: nonlinear joint equality is unsupported");
        const auto& source_joint = scene.GetJoint(source);
        auto& target_joint = scene.GetJointMut(target);
        if (target_joint.mimic_source != scene::kInvalidJoint)
            throw std::runtime_error("MJCF: duplicate joint equality target");
        target_joint.mimic_source = source;
        target_joint.mimic_multiplier = coefficient[1];
        target_joint.mimic_offset = target_joint.initial_position + coefficient[0] -
                                    coefficient[1] * source_joint.initial_position;
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
        scene::SensorDesc record;
        record.name = sensor->Attribute("name") ? sensor->Attribute("name") : "sensor";
        if (!MjcfSensorType(tag, record.type)) {
            throw std::runtime_error("MJCF: unsupported <sensor> element '" + tag + "'");
        }

        // A site (site=, or objtype="site" objname=) carries (body, local frame); else objname/body
        // name a body. An unresolved name stays kInvalidBody for the runtime to flag.
        record.mount = scene::MountFrame::Body;
        if (record.type == scene::SensorType::JointState) {
            record.joint_id = ResolveJoint(sensor->Attribute("joint"), context);
            if (record.joint_id == scene::kInvalidJoint)
                throw std::runtime_error("MJCF: joint sensor has an unresolved joint");
            record.mount_index = scene.GetJoint(record.joint_id).child_body;
            scene.AddSensor(std::move(record));
            continue;
        }
        const char* site_name = sensor->Attribute("site");
        if (!site_name && sensor->Attribute("objtype", "site")) site_name = sensor->Attribute("objname");
        if (site_name) {
            const auto it = context.site_ids.find(site_name);
            if (it != context.site_ids.end()) {
                record.mount_index = it->second.body;
                record.local_offset = it->second.local;
                record.tactile = it->second.region;
            }
        } else {
            const char* body_name = sensor->Attribute("objname");
            if (!body_name) {
                body_name = sensor->Attribute("body");
            }
            record.mount_index = ResolveBody(body_name, context);
        }
        if (record.type == scene::SensorType::Contact &&
            (record.mount_index == scene::kInvalidBody || !sensor::ValidTactileConfig(record.tactile, true)))
            throw std::runtime_error("MJCF: touch requires a valid body-mounted site volume");

        // A rangefinder is a single forward ray; the runtime ray-gen reads this
        // 1-ray pattern through the same lidar path the multi-ray scanners use.
        if (record.type == scene::SensorType::RangeScan) {
            record.lidar.az_count = 1;
            record.lidar.el_count = 1;
        }

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

void ParseNukaEnvironment(tinyxml2::XMLElement* mujoco, scene::SceneIR& scene) {
    // Nuka-only beauty metadata for generated MJCF scenes. It is deliberately a
    // root child so source MuJoCo assets stay untouched and physics import is
    // independent from presentation settings.
    auto* element = mujoco->FirstChildElement("nuka_environment");
    if (!element) {
        return;
    }

    scene::EnvironmentRecord record;
    bool enabled = false;
    if (element->QueryBoolAttribute("use_scene_materials", &enabled) == tinyxml2::XML_SUCCESS) {
        record.use_scene_materials = enabled;
    }
    element->QueryFloatAttribute("exposure_ev", &record.exposure_ev);
    element->QueryFloatAttribute("grade", &record.grade);
    element->QueryFloatAttribute("sun_disc", &record.sun_disc);
    bool specular = false;
    if (element->QueryBoolAttribute("specular_env", &specular) == tinyxml2::XML_SUCCESS) {
        record.specular_env = specular;
    }
    scene.EnvironmentMut() = std::move(record);
}

// Elements that add bodies, joints or couplings must be supported, never skipped.
void RejectUnsupportedElements(const tinyxml2::XMLElement* element) {
    for (const auto* child = element->FirstChildElement(); child != nullptr;
         child = child->NextSiblingElement()) {
        const std::string tag = child->Name();
        if (tag == "include" || tag == "frame" || tag == "replicate" || tag == "attach" ||
            tag == "composite" || tag == "flexcomp")
            throw std::runtime_error("MJCF: <" + tag + "> is unsupported");
        RejectUnsupportedElements(child);
    }
    if (std::string(element->Name()) == "tendon" && element->FirstChildElement() != nullptr)
        throw std::runtime_error("MJCF: tendons are unsupported");
}

} // namespace

scene::SceneIR LoadMjcf(const std::string& path) {
    tinyxml2::XMLDocument doc;
    const tinyxml2::XMLError err = doc.LoadFile(path.c_str());
    if (err != tinyxml2::XML_SUCCESS) {
        throw std::runtime_error("MJCF: failed to load file: " + path + " (" +
                                 tinyxml2::XMLDocument::ErrorIDToName(err) + ")");
    }

    auto* mujoco = doc.FirstChildElement("mujoco");
    if (!mujoco) {
        throw std::runtime_error("MJCF: missing <mujoco> root element in " + path);
    }

    auto* worldbody = mujoco->FirstChildElement("worldbody");
    if (!worldbody) {
        throw std::runtime_error("MJCF: missing <worldbody> element in " + path);
    }
    RejectUnsupportedElements(mujoco);

    scene::SceneIR scene;
    MjcfParseContext context;
    scene.SolverOptionsMut().law_source = constraint::LawSource::MuJoCo;

    if (const auto* option = mujoco->FirstChildElement("option")) {
        auto& solver = scene.SolverOptionsMut();
        if (const char* cone = option->Attribute("cone"))
            solver.cone = std::unordered_map<std::string, constraint::FrictionCone>{
                {"elliptic", constraint::FrictionCone::Elliptic},
                {"pyramidal", constraint::FrictionCone::Pyramidal}}.at(cone);
        option->QueryFloatAttribute("impratio", &solver.impratio);
        if (const char* integrator = option->Attribute("integrator")) {
            solver.integrator = std::unordered_map<std::string, constraint::Integrator>{
                {"Euler", constraint::Integrator::Euler}, {"RK4", constraint::Integrator::Rk4},
                {"implicit", constraint::Integrator::Implicit},
                {"implicitfast", constraint::Integrator::ImplicitFast}}.at(integrator);
        }
        if (const auto* flag = option->FirstChildElement("flag"))
            if (const char* refsafe = flag->Attribute("refsafe"))
                solver.refsafe = std::unordered_map<std::string, bool>{
                    {"enable", true}, {"disable", false}}.at(refsafe);
    }

    if (const auto* compiler = mujoco->FirstChildElement("compiler")) {
        if (const char* angle = compiler->Attribute("angle")) {
            if (std::string(angle) == "radian") context.angle_scale = 1.0f;
            else if (std::string(angle) != "degree") throw std::runtime_error("MJCF: invalid compiler angle");
        }
        if (const char* sequence = compiler->Attribute("eulerseq")) context.euler_sequence = sequence;
        if (const char* from_geom = compiler->Attribute("inertiafromgeom")) {
            const std::string mode(from_geom);
            if (mode != "auto" && mode != "true" && mode != "false")
                throw std::runtime_error("MJCF: invalid compiler inertiafromgeom");
            context.inertia_from_geom = mode == "auto" ? -1 : mode == "true" ? 1 : 0;
        }
        float groups[2] = {0.0f, 5.0f};
        ParseFloatList(compiler->Attribute("inertiagrouprange"), groups, 2);
        context.inertia_groups[0] = static_cast<int>(groups[0]);
        context.inertia_groups[1] = static_cast<int>(groups[1]);
    }

    ParseDefaults(mujoco, context);
    ParseMaterials(mujoco, scene, context, std::filesystem::path(path).parent_path());
    ParseNukaEnvironment(mujoco, scene);
    ParseMeshAssets(mujoco, context, std::filesystem::path(path).parent_path());
    ParseHeightFieldAssets(mujoco, context);
    ParseLights(worldbody, scene);

    // Geoms placed straight in <worldbody> ride a static world body.
    if (worldbody->FirstChildElement("geom")) {
        context.world_body = AddStaticBody("world", scene);
        ParseGeoms(worldbody, context.world_body, scene, context, nullptr, nullptr);
    }
    for (auto* body = worldbody->FirstChildElement("body");
         body != nullptr;
         body = body->NextSiblingElement("body")) {
        ParseBody(body, scene::kInvalidBody, scene, context, nullptr);
    }

    ParseJointEqualities(mujoco, scene, context);
    ParseActuators(mujoco, scene, context);
    ParseSensors(mujoco, scene, context);
    // <contact> is parsed LAST so all body + geom names already resolve.
    ParseContact(mujoco, scene, context);

    return scene;
}

} // namespace nuka::import
