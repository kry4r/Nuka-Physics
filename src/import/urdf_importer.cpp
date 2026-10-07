// ---------------------------------------------------------------------------
// nuka::import – URDF importer implementation
// ---------------------------------------------------------------------------

#include "import/urdf_importer.hpp"
#include "import/mesh_file_loader.hpp"
#include "import/principal_inertia.hpp"
#include "scene/canonical_types.hpp"
#include "math/vec3.hpp"
#include "math/transform.hpp"

#include <tinyxml2.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
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

math::Transform ParseOrigin(const tinyxml2::XMLElement* element) {
    math::Transform result = math::Transform::Identity();
    if (element == nullptr) return result;
    if (const char* xyz = element->Attribute("xyz")) result.position = ParseVec3(xyz);
    if (const char* rpy = element->Attribute("rpy")) {
        const math::Vec3 angle = ParseVec3(rpy);
        result.rotation = (math::Quat::FromAxisAngle({0, 0, 1}, angle.z) *
            math::Quat::FromAxisAngle({0, 1, 0}, angle.y) *
            math::Quat::FromAxisAngle({1, 0, 0}, angle.x)).Normalized();
    }
    return result;
}

std::filesystem::path ResolveMeshPath(const std::filesystem::path& urdf,
                                      const std::string& filename) {
    namespace fs = std::filesystem;
    if (filename.empty()) throw std::runtime_error("URDF: mesh filename is empty");
    if (filename.rfind("package://", 0u) == 0u) {
        const fs::path package_path = filename.substr(10u);
        if (package_path.empty()) throw std::runtime_error("URDF: mesh package path is empty");
        for (fs::path root = urdf.parent_path(); !root.empty(); root = root.parent_path()) {
            const fs::path inside = root / package_path;
            if (fs::is_regular_file(inside)) return inside;
            if (root.filename() == *package_path.begin()) {
                const fs::path sibling = root.parent_path() / package_path;
                if (fs::is_regular_file(sibling)) return sibling;
            }
            if (root == root.parent_path()) break;
        }
        throw std::runtime_error("URDF: mesh package path was not found: " + filename);
    }
    const fs::path path = filename.rfind("file://", 0u) == 0u
        ? fs::path(filename.substr(7u)) : fs::path(filename);
    const fs::path resolved = path.is_absolute() ? path : urdf.parent_path() / path;
    if (!fs::is_regular_file(resolved))
        throw std::runtime_error("URDF: mesh file was not found: " + resolved.string());
    return resolved;
}

void LoadUrdfMesh(const tinyxml2::XMLElement* element, const std::filesystem::path& urdf,
                  scene::CollisionShapeRecord& shape) {
    if (shape.type != scene::ShapeType::TriMesh) return;
    const char* filename = element->Attribute("filename");
    if (filename == nullptr) throw std::runtime_error("URDF: mesh has no filename");
    MeshGeometry mesh = LoadMeshFile(ResolveMeshPath(urdf, filename).string());
    math::Vec3 scale{1.0f, 1.0f, 1.0f};
    if (const char* value = element->Attribute("scale")) scale = ParseVec3(value);
    if (!std::isfinite(scale.x) || !std::isfinite(scale.y) || !std::isfinite(scale.z) ||
        scale.x == 0.0f || scale.y == 0.0f || scale.z == 0.0f)
        throw std::runtime_error("URDF: mesh scale must be finite and nonzero");
    for (size_t i = 0u; i < mesh.vertices.size(); i += 3u) {
        mesh.vertices[i] *= scale.x;
        mesh.vertices[i + 1u] *= scale.y;
        mesh.vertices[i + 2u] *= scale.z;
    }
    for (size_t i = 0u; i + 2u < mesh.normals.size(); i += 3u) {
        const math::Vec3 normal{mesh.normals[i] / scale.x,
                                mesh.normals[i + 1u] / scale.y,
                                mesh.normals[i + 2u] / scale.z};
        const math::Vec3 unit = normal.Normalized();
        mesh.normals[i] = unit.x;
        mesh.normals[i + 1u] = unit.y;
        mesh.normals[i + 2u] = unit.z;
    }
    if (scale.x * scale.y * scale.z < 0.0f)
        for (size_t i = 0u; i + 2u < mesh.indices.size(); i += 3u)
            std::swap(mesh.indices[i + 1u], mesh.indices[i + 2u]);
    shape.mesh_vertices = std::move(mesh.vertices);
    shape.mesh_indices = std::move(mesh.indices);
    shape.mesh_normals = std::move(mesh.normals);
    shape.mesh_uvs = std::move(mesh.uvs);
}

/// Map URDF joint type strings to JointType.
scene::JointType UrdfJointType(const char* type_str, const std::string& joint_name) {
    const std::string t = type_str ? type_str : "";
    if (t == "revolute"   || t == "continuous") return scene::JointType::Revolute;
    if (t == "prismatic")                       return scene::JointType::Prismatic;
    if (t == "fixed")                           return scene::JointType::Fixed;
    if (t == "floating")                        return scene::JointType::Free;
    throw std::runtime_error("URDF: joint '" + joint_name + "' has unsupported type '" + t + "'");
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
        // A link without <inertial> is massless, as the URDF format defines.
        rec.mass = 0.0f;
        rec.inertia = math::Vec3::Zero();

        // Parse <inertial>
        if (auto* inertial = link->FirstChildElement("inertial")) {
            // <mass value="..."/>
            if (auto* mass_elem = inertial->FirstChildElement("mass")) {
                mass_elem->QueryFloatAttribute("value", &rec.mass);
            }

            rec.inertial_transform = ParseOrigin(inertial->FirstChildElement("origin"));

            // The full tensor in the inertial frame becomes principal moments about rotated axes.
            if (auto* inertia = inertial->FirstChildElement("inertia")) {
                const char* keys[6] = {"ixx", "iyy", "izz", "ixy", "ixz", "iyz"};
                float full[6] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
                for (int k = 0; k < 6; ++k) inertia->QueryFloatAttribute(keys[k], &full[k]);
                math::Quat axes;
                DiagonalizeInertia(full, rec.inertia, axes);
                rec.inertial_transform.rotation = rec.inertial_transform.rotation * axes;
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
            LoadUrdfMesh(shape_elem, path, shape);

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
            shape.local_transform = ParseOrigin(collision->FirstChildElement("origin"));

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
            if (auto* contact = collision->FirstChildElement("nuka:mesh_contact")) {
                const char* mode = contact->Attribute("mode");
                if (mode != nullptr && std::string(mode) == "sdf")
                    shape.mesh_contact = scene::CollisionShapeRecord::MeshContact::Sdf;
                else if (mode != nullptr && std::string(mode) != "ogc")
                    throw std::runtime_error("URDF: invalid mesh contact mode");
            }
            if (auto* simplify = collision->FirstChildElement("nuka:mesh_triangle_limit")) {
                unsigned int limit = 0u;
                if (simplify->QueryUnsignedAttribute("value", &limit) != tinyxml2::XML_SUCCESS ||
                    limit == 0u)
                    throw std::runtime_error("URDF: invalid mesh triangle limit");
                shape.mesh_triangle_limit = limit;
            }
            if (auto* error = collision->FirstChildElement("nuka:mesh_error_limit")) {
                float limit = 0.0f;
                if (error->QueryFloatAttribute("value", &limit) != tinyxml2::XML_SUCCESS ||
                    !(limit > 0.0f && limit <= 0.001f))
                    throw std::runtime_error("URDF: invalid mesh error limit");
                shape.mesh_error_limit = limit;
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
            LoadUrdfMesh(shape_elem, path, shape);
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

            shape.local_transform = ParseOrigin(visual->FirstChildElement("origin"));

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

    std::unordered_map<std::string, scene::JointId> joint_names;
    scene::JointId next_joint = static_cast<scene::JointId>(scene.JointCount());
    for (auto* joint = robot->FirstChildElement("joint");
         joint != nullptr; joint = joint->NextSiblingElement("joint")) {
        const char* name = joint->Attribute("name");
        if (name == nullptr || name[0] == '\0' ||
            !joint_names.emplace(name, next_joint++).second)
            throw std::runtime_error("URDF: joint names must be unique and nonempty");
    }

    // -- Parse <joint> elements -----------------------------------------------
    for (auto* joint = robot->FirstChildElement("joint");
         joint != nullptr;
         joint = joint->NextSiblingElement("joint")) {

        const char* jname = joint->Attribute("name");
        const std::string joint_name = jname ? jname : "unnamed_joint";

        scene::JointRecord jrec;
        jrec.name = joint_name;
        jrec.type = UrdfJointType(joint->Attribute("type"), joint_name);

        // <parent link="..."/> and <child link="..."/> must name declared links.
        auto link_body = [&](const char* tag) {
            const auto* element = joint->FirstChildElement(tag);
            const char* link = element ? element->Attribute("link") : nullptr;
            const auto it = link ? link_map.find(link) : link_map.end();
            if (it == link_map.end())
                throw std::runtime_error("URDF: joint '" + joint_name + "' has no declared " + tag + " link");
            return it->second;
        };
        jrec.parent_body = link_body("parent");
        jrec.child_body = link_body("child");

        // <origin xyz="..."/>
        jrec.parent_frame = ParseOrigin(joint->FirstChildElement("origin"));

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

        if (auto* mimic = joint->FirstChildElement("mimic")) {
            const char* source = mimic->Attribute("joint");
            const auto found = source ? joint_names.find(source) : joint_names.end();
            if (found == joint_names.end() || found->second == joint_names.at(joint_name))
                throw std::runtime_error("URDF: mimic source joint is missing or self-referential");
            jrec.mimic_source = found->second;
            if (mimic->Attribute("multiplier") &&
                mimic->QueryFloatAttribute("multiplier", &jrec.mimic_multiplier) != tinyxml2::XML_SUCCESS)
                throw std::runtime_error("URDF: invalid mimic multiplier");
            if (mimic->Attribute("offset") &&
                mimic->QueryFloatAttribute("offset", &jrec.mimic_offset) != tinyxml2::XML_SUCCESS)
                throw std::runtime_error("URDF: invalid mimic offset");
            if (!std::isfinite(jrec.mimic_multiplier) || !std::isfinite(jrec.mimic_offset))
                throw std::runtime_error("URDF: nonfinite mimic coefficient");
        }

        const scene::JointId joint_id = scene.AddJoint(std::move(jrec));
        if (effort_limit > 0.0f && scene.GetJoint(joint_id).mimic_source == scene::kInvalidJoint) {
            scene::ActuatorRecord actuator;
            actuator.name = joint_name + "_effort";
            actuator.type = scene::ActuatorType::Motor;
            actuator.joint_id = joint_id;
            actuator.force_limit = effort_limit;
            scene.AddActuator(std::move(actuator));
        }
    }

    // Robot-level <nuka:exclude link1="..." link2="..."/> disables contact between two links,
    // as MJCF <contact><exclude> does; unknown or identical links are an error.
    for (auto* exclude = robot->FirstChildElement("nuka:exclude"); exclude != nullptr;
         exclude = exclude->NextSiblingElement("nuka:exclude")) {
        const char* first = exclude->Attribute("link1");
        const char* second = exclude->Attribute("link2");
        const auto a = first ? link_map.find(first) : link_map.end();
        const auto b = second ? link_map.find(second) : link_map.end();
        if (a == link_map.end() || b == link_map.end() || a->second == b->second)
            throw std::runtime_error("URDF: nuka:exclude needs two distinct existing links");
        scene.AddExcludePair(a->second, b->second);
    }

    return scene;
}

} // namespace nuka::import
