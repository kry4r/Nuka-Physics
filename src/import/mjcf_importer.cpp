// ---------------------------------------------------------------------------
// nuka::import – MJCF (MuJoCo XML) importer implementation
// ---------------------------------------------------------------------------

#include "import/mjcf_importer.hpp"
#include "scene/canonical_types.hpp"
#include "math/vec3.hpp"
#include "math/transform.hpp"

#include <tinyxml2.h>

#include <stdexcept>
#include <string>
#include <sstream>
#include <unordered_map>

namespace nuka::import {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

/// Parse a whitespace-separated list of 3 floats into a Vec3.
math::Vec3 ParseVec3(const char* text) {
    math::Vec3 v{};
    if (!text) return v;
    std::istringstream ss(text);
    ss >> v.x >> v.y >> v.z;
    return v;
}

/// Map MJCF geom type strings to ShapeType.
scene::ShapeType MjcfGeomType(const char* type_str) {
    if (!type_str) return scene::ShapeType::Box; // MJCF default
    const std::string t(type_str);
    if (t == "sphere")  return scene::ShapeType::Sphere;
    if (t == "capsule") return scene::ShapeType::Capsule;
    if (t == "box")     return scene::ShapeType::Box;
    if (t == "plane")   return scene::ShapeType::Plane;
    return scene::ShapeType::Box;
}

/// Map MJCF joint type strings to JointType.
scene::JointType MjcfJointType(const char* type_str) {
    if (!type_str) return scene::JointType::Revolute; // "hinge" is default
    const std::string t(type_str);
    if (t == "hinge")  return scene::JointType::Revolute;
    if (t == "slide")  return scene::JointType::Prismatic;
    if (t == "ball")   return scene::JointType::Spherical;
    if (t == "free")   return scene::JointType::Free;
    return scene::JointType::Revolute;
}

/// Recursively parse <body> elements under worldbody.
/// @param parent_id  The BodyId of the parent (kInvalidBody for worldbody root).
void ParseBody(tinyxml2::XMLElement* body_elem,
               scene::BodyId parent_id,
               scene::SceneIR& scene) {

    // -- Create the rigid body ------------------------------------------------
    const char* name_attr = body_elem->Attribute("name");
    const std::string body_name = name_attr ? name_attr : "unnamed";

    scene::RigidBodyRecord rec;
    rec.name = body_name;
    rec.parent_id = parent_id;

    // Parse position from "pos" attribute
    const char* pos_attr = body_elem->Attribute("pos");
    if (pos_attr) {
        const math::Vec3 pos = ParseVec3(pos_attr);
        rec.local_transform = math::Transform{pos, math::Quat::Identity()};
    }

    // Parse <inertial> if present
    if (auto* inertial = body_elem->FirstChildElement("inertial")) {
        inertial->QueryFloatAttribute("mass", &rec.mass);

        const char* inertia_pos = inertial->Attribute("pos");
        if (inertia_pos) {
            // inertial pos is the center-of-mass offset; we store it but
            // keep the body transform as-is for this skeleton importer.
            (void)ParseVec3(inertia_pos);
        }

        const char* diag = inertial->Attribute("diaginertia");
        if (diag) {
            rec.inertia = ParseVec3(diag);
        }
    }

    const scene::BodyId body_id = scene.AddRigidBody(std::move(rec));

    // -- Parse <geom> children ------------------------------------------------
    for (auto* geom = body_elem->FirstChildElement("geom");
         geom != nullptr;
         geom = geom->NextSiblingElement("geom")) {

        scene::CollisionShapeRecord shape;
        shape.body_id = body_id;
        shape.type = MjcfGeomType(geom->Attribute("type"));

        // Parse size attribute (interpretation depends on type)
        const char* size_attr = geom->Attribute("size");
        if (size_attr) {
            math::Vec3 sz = ParseVec3(size_attr);
            if (shape.type == scene::ShapeType::Box) {
                shape.half_extents = sz;
            } else if (shape.type == scene::ShapeType::Sphere) {
                shape.radius = sz.x;
            } else if (shape.type == scene::ShapeType::Capsule) {
                shape.radius = sz.x;
                // half_height from fromto or second size component
                if (sz.y > 0.0f) {
                    shape.half_height = sz.y;
                }
            }
        }

        scene.AddCollisionShape(std::move(shape));
    }

    // -- Parse <joint> children -----------------------------------------------
    for (auto* joint = body_elem->FirstChildElement("joint");
         joint != nullptr;
         joint = joint->NextSiblingElement("joint")) {

        const char* jname = joint->Attribute("name");
        const std::string joint_name = jname ? jname : "unnamed_joint";

        scene::JointRecord jrec;
        jrec.name = joint_name;
        jrec.parent_body = parent_id;
        jrec.child_body = body_id;
        jrec.type = MjcfJointType(joint->Attribute("type"));

        const char* axis_attr = joint->Attribute("axis");
        if (axis_attr) {
            jrec.axis = ParseVec3(axis_attr);
        }

        scene.AddJoint(std::move(jrec));
    }

    // -- Recurse into child <body> elements -----------------------------------
    for (auto* child = body_elem->FirstChildElement("body");
         child != nullptr;
         child = child->NextSiblingElement("body")) {
        ParseBody(child, body_id, scene);
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

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

    // Iterate over top-level <body> elements inside <worldbody>
    for (auto* body = worldbody->FirstChildElement("body");
         body != nullptr;
         body = body->NextSiblingElement("body")) {
        ParseBody(body, scene::kInvalidBody, scene);
    }

    return scene;
}

} // namespace nuka::import
