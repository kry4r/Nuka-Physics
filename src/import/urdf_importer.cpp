// ---------------------------------------------------------------------------
// nuka::import – URDF importer implementation
// ---------------------------------------------------------------------------

#include "import/urdf_importer.hpp"
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

/// Map URDF joint type strings to JointType.
scene::JointType UrdfJointType(const char* type_str) {
    if (!type_str) return scene::JointType::Fixed;
    const std::string t(type_str);
    if (t == "revolute"   || t == "continuous") return scene::JointType::Revolute;
    if (t == "prismatic")                       return scene::JointType::Prismatic;
    if (t == "fixed")                           return scene::JointType::Fixed;
    if (t == "floating")                        return scene::JointType::Free;
    if (t == "planar")                          return scene::JointType::Free;
    return scene::JointType::Fixed;
}

/// Map URDF collision geometry to ShapeType.
scene::ShapeType UrdfGeomType(const char* tag_name) {
    if (!tag_name) return scene::ShapeType::Box;
    const std::string t(tag_name);
    if (t == "box")      return scene::ShapeType::Box;
    if (t == "sphere")   return scene::ShapeType::Sphere;
    if (t == "cylinder")  return scene::ShapeType::Capsule; // approximate
    if (t == "mesh")     return scene::ShapeType::TriMesh;
    return scene::ShapeType::Box;
}

/// Map a nuka:decompose token to DecomposeMode.
scene::DecomposeMode DecomposeModeFromToken(const char* token) {
    if (!token) return scene::DecomposeMode::Auto;
    const std::string t(token);
    if (t == "force") return scene::DecomposeMode::Force;
    if (t == "skip")  return scene::DecomposeMode::Skip;
    return scene::DecomposeMode::Auto;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

scene::SceneIR LoadUrdf(const std::string& path) {
    tinyxml2::XMLDocument doc;
    const tinyxml2::XMLError err = doc.LoadFile(path.c_str());
    if (err != tinyxml2::XML_SUCCESS) {
        throw std::runtime_error("URDF: failed to load file: " + path +
                                 " (error " + std::to_string(static_cast<int>(err)) + ")");
    }

    auto* robot = doc.FirstChildElement("robot");
    if (!robot) {
        throw std::runtime_error("URDF: missing <robot> root element in " + path);
    }

    scene::SceneIR scene;

    // Map from link name -> BodyId for joint resolution
    std::unordered_map<std::string, scene::BodyId> link_map;
    // Dedup map for synthesized visual materials (URDF <material name=> is a
    // named, potentially shared, color) -> the MaterialId it was first added as.
    std::unordered_map<std::string, scene::MaterialId> material_names;

    // -- Parse <link> elements ------------------------------------------------
    for (auto* link = robot->FirstChildElement("link");
         link != nullptr;
         link = link->NextSiblingElement("link")) {

        const char* name_attr = link->Attribute("name");
        const std::string link_name = name_attr ? name_attr : "unnamed";

        scene::RigidBodyRecord rec;
        rec.name = link_name;

        // Parse <inertial>
        if (auto* inertial = link->FirstChildElement("inertial")) {
            // <mass value="..."/>
            if (auto* mass_elem = inertial->FirstChildElement("mass")) {
                mass_elem->QueryFloatAttribute("value", &rec.mass);
            }

            // <origin xyz="..."/>
            if (auto* origin = inertial->FirstChildElement("origin")) {
                const char* xyz = origin->Attribute("xyz");
                if (xyz) {
                    rec.inertial_transform.position = ParseVec3(xyz);
                }
            }

            // <inertia ixx="..." iyy="..." izz="..."/>
            if (auto* inertia = inertial->FirstChildElement("inertia")) {
                float ixx = 1.0f, iyy = 1.0f, izz = 1.0f;
                inertia->QueryFloatAttribute("ixx", &ixx);
                inertia->QueryFloatAttribute("iyy", &iyy);
                inertia->QueryFloatAttribute("izz", &izz);
                rec.inertia = math::Vec3{ixx, iyy, izz};
            }
        }

        const scene::BodyId body_id = scene.AddRigidBody(std::move(rec));
        link_map[link_name] = body_id;

        // Parse <collision><geometry> for shapes
        for (auto* collision = link->FirstChildElement("collision");
             collision != nullptr;
             collision = collision->NextSiblingElement("collision")) {

            auto* geometry = collision->FirstChildElement("geometry");
            if (!geometry) continue;

            // Find the first child element of <geometry> (box, sphere, cylinder, etc.)
            auto* shape_elem = geometry->FirstChildElement();
            if (!shape_elem) continue;

            scene::CollisionShapeRecord shape;
            shape.body_id = body_id;
            shape.type = UrdfGeomType(shape_elem->Name());

            if (shape.type == scene::ShapeType::Box) {
                const char* size_attr = shape_elem->Attribute("size");
                if (size_attr) {
                    math::Vec3 full_size = ParseVec3(size_attr);
                    shape.half_extents = math::Vec3{
                        full_size.x * 0.5f,
                        full_size.y * 0.5f,
                        full_size.z * 0.5f
                    };
                }
            } else if (shape.type == scene::ShapeType::Sphere) {
                shape_elem->QueryFloatAttribute("radius", &shape.radius);
            } else if (shape.type == scene::ShapeType::Capsule) {
                shape_elem->QueryFloatAttribute("radius", &shape.radius);
                float length = 1.0f;
                shape_elem->QueryFloatAttribute("length", &length);
                shape.half_height = length * 0.5f;
            }

            // Parse collision origin if present
            if (auto* coll_origin = collision->FirstChildElement("origin")) {
                const char* xyz = coll_origin->Attribute("xyz");
                if (xyz) {
                    shape.local_transform = math::Transform{
                        ParseVec3(xyz), math::Quat::Identity()
                    };
                }
            }

            // Parse <nuka:decompose [decompose="auto|force|skip"] [max_pieces="N"]/>
            // on a mesh collision element (v0.7 p06).
            if (auto* decomp = collision->FirstChildElement("nuka:decompose")) {
                shape.decompose_mode = DecomposeModeFromToken(decomp->Attribute("decompose"));
                int max_pieces = 0;
                if (decomp->QueryIntAttribute("max_pieces", &max_pieces) ==
                        tinyxml2::XML_SUCCESS && max_pieces > 0) {
                    shape.decompose_max_pieces = static_cast<uint32_t>(max_pieces);
                }
            }

            scene.AddCollisionShape(std::move(shape));
        }

        // Parse <visual><geometry> as non-colliding (contype=0/conaffinity=0)
        // shape records. These project to VisualMeshComponent through the SceneIR
        // facade (M2b) so render-only geometry survives import; physics is
        // untouched (collision shapes above own contact). A <visual><material>
        // with an inline <color rgba="..."/> is synthesized into a MaterialRecord
        // so the visual color reaches RenderMaterial.
        for (auto* visual = link->FirstChildElement("visual");
             visual != nullptr;
             visual = visual->NextSiblingElement("visual")) {

            auto* geometry = visual->FirstChildElement("geometry");
            if (!geometry) continue;
            auto* shape_elem = geometry->FirstChildElement();
            if (!shape_elem) continue;

            scene::CollisionShapeRecord shape;
            shape.body_id = body_id;
            shape.type = UrdfGeomType(shape_elem->Name());
            shape.contype = 0;       // visual-only: no collision (facade -> VisualMesh)
            shape.conaffinity = 0;
            shape.name = visual->Attribute("name") ? visual->Attribute("name") : "";

            if (shape.type == scene::ShapeType::Box) {
                if (const char* size_attr = shape_elem->Attribute("size")) {
                    math::Vec3 full_size = ParseVec3(size_attr);
                    shape.half_extents = math::Vec3{
                        full_size.x * 0.5f, full_size.y * 0.5f, full_size.z * 0.5f};
                }
            } else if (shape.type == scene::ShapeType::Sphere) {
                shape_elem->QueryFloatAttribute("radius", &shape.radius);
            } else if (shape.type == scene::ShapeType::Capsule) {
                shape_elem->QueryFloatAttribute("radius", &shape.radius);
                float length = 1.0f;
                shape_elem->QueryFloatAttribute("length", &length);
                shape.half_height = length * 0.5f;
            }

            if (auto* vis_origin = visual->FirstChildElement("origin")) {
                if (const char* xyz = vis_origin->Attribute("xyz")) {
                    shape.local_transform = math::Transform{
                        ParseVec3(xyz), math::Quat::Identity()};
                }
            }

            // Inline visual material color -> synthesized MaterialRecord. URDF
            // <material name="..."><color rgba="r g b a"/></material>. A bare
            // <material name="x"/> reference (no color) is skipped (no color to
            // carry); duplicate names are deduped by reusing the existing record.
            if (auto* mat = visual->FirstChildElement("material")) {
                if (auto* color = mat->FirstChildElement("color")) {
                    if (const char* rgba = color->Attribute("rgba")) {
                        std::istringstream ss(rgba);
                        scene::MaterialRecord mrec;
                        const char* mname = mat->Attribute("name");
                        mrec.name = mname ? mname : (link_name + "_vis_mat");
                        ss >> mrec.base_color.x >> mrec.base_color.y
                           >> mrec.base_color.z >> mrec.alpha;
                        const auto found = material_names.find(mrec.name);
                        if (found != material_names.end()) {
                            shape.material_id = found->second;
                        } else {
                            const scene::MaterialId mid =
                                scene.AddMaterial(std::move(mrec));
                            material_names[mat->Attribute("name")
                                               ? mat->Attribute("name")
                                               : (link_name + "_vis_mat")] = mid;
                            shape.material_id = mid;
                        }
                    }
                }
            }

            scene.AddCollisionShape(std::move(shape));
        }
    }

    // -- Parse <joint> elements -----------------------------------------------
    for (auto* joint = robot->FirstChildElement("joint");
         joint != nullptr;
         joint = joint->NextSiblingElement("joint")) {

        const char* jname = joint->Attribute("name");
        const std::string joint_name = jname ? jname : "unnamed_joint";

        scene::JointRecord jrec;
        jrec.name = joint_name;
        jrec.type = UrdfJointType(joint->Attribute("type"));

        // <parent link="..."/>
        if (auto* parent = joint->FirstChildElement("parent")) {
            const char* plink = parent->Attribute("link");
            if (plink) {
                auto it = link_map.find(plink);
                if (it != link_map.end()) {
                    jrec.parent_body = it->second;
                }
            }
        }

        // <child link="..."/>
        if (auto* child = joint->FirstChildElement("child")) {
            const char* clink = child->Attribute("link");
            if (clink) {
                auto it = link_map.find(clink);
                if (it != link_map.end()) {
                    jrec.child_body = it->second;
                }
            }
        }

        // <origin xyz="..."/>
        if (auto* origin = joint->FirstChildElement("origin")) {
            const char* xyz = origin->Attribute("xyz");
            if (xyz) {
                jrec.parent_frame = math::Transform{
                    ParseVec3(xyz), math::Quat::Identity()
                };
            }
        }

        // <axis xyz="..."/>
        if (auto* axis = joint->FirstChildElement("axis")) {
            const char* xyz = axis->Attribute("xyz");
            if (xyz) {
                jrec.axis = ParseVec3(xyz);
            }
        }

        float effort_limit = 0.0f;
        // <limit lower="..." upper="..." effort="..."/>
        if (auto* limit = joint->FirstChildElement("limit")) {
            jrec.has_lower_limit =
                limit->QueryFloatAttribute("lower", &jrec.lower_limit) == tinyxml2::XML_SUCCESS;
            jrec.has_upper_limit =
                limit->QueryFloatAttribute("upper", &jrec.upper_limit) == tinyxml2::XML_SUCCESS;
            (void)limit->QueryFloatAttribute("effort", &effort_limit);
        }

        const scene::JointId joint_id = scene.AddJoint(std::move(jrec));
        if (effort_limit > 0.0f) {
            scene::ActuatorRecord actuator;
            actuator.name = joint_name + "_effort";
            actuator.type = scene::ActuatorType::Motor;
            actuator.joint_id = joint_id;
            actuator.force_limit = effort_limit;
            scene.AddActuator(std::move(actuator));
        }
    }

    return scene;
}

} // namespace nuka::import
