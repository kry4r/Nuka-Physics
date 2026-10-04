// ---------------------------------------------------------------------------
// nuka::import – URDF importer implementation
// ---------------------------------------------------------------------------

#include "import/urdf_importer.hpp"
#include "import/mesh_file_loader.hpp"
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
#include <unordered_set>
#include <functional>
#include <limits>

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

static scene::SceneIR ParseUrdf(tinyxml2::XMLElement* robot, const std::string& path) {
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

            rec.inertial_transform = ParseOrigin(inertial->FirstChildElement("origin"));

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
        jrec.type = UrdfJointType(joint->Attribute("type"));
        if (jrec.type == scene::JointType::Revolute || jrec.type == scene::JointType::Prismatic)
            jrec.axis = math::Vec3::UnitX();

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

namespace {
using Element = tinyxml2::XMLElement;

void Diagnose(UrdfImportResult& r, const Element* e, UrdfDiagnosticKind kind,
              const std::string& code, const std::string& message) {
    r.diagnostics.push_back({kind, code, {}, e ? e->GetLineNum() : 0, e ? e->Name() : "", message});
}

bool Listed(const std::string& value, const std::string& list) {
    std::istringstream tokens(list);
    std::string token;
    while (tokens >> token) if (token == value) return true;
    return false;
}

void ValidateUrdf(Element* robot, UrdfImportResult& result) {
    const std::unordered_map<std::string, std::pair<std::string, std::string>> schema{
        {"robot", {"name", "link joint"}}, {"link", {"name", "inertial collision visual"}},
        {"inertial", {"", "origin mass inertia"}}, {"mass", {"value", ""}},
        {"inertia", {"ixx iyy izz ixy ixz iyz", ""}}, {"origin", {"xyz rpy", ""}},
        {"collision", {"", "origin geometry"}}, {"visual", {"name", "origin geometry"}},
        {"geometry", {"", "box sphere mesh"}}, {"box", {"size", ""}},
        {"sphere", {"radius", ""}}, {"mesh", {"filename scale", ""}},
        {"joint", {"name type", "parent child origin axis limit"}},
        {"parent", {"link", ""}}, {"child", {"link", ""}},
        {"axis", {"xyz", ""}}, {"limit", {"lower upper", ""}}
    };
    const std::unordered_map<std::string, std::string> required_attributes{
        {"robot", "name"}, {"link", "name"}, {"joint", "name type"}, {"mass", "value"},
        {"inertia", "ixx iyy izz"}, {"box", "size"}, {"sphere", "radius"},
        {"mesh", "filename"}, {"axis", "xyz"}, {"parent", "link"}, {"child", "link"}
    };
    const std::unordered_map<std::string, std::string> required_children{
        {"link", "inertial"}, {"inertial", "mass inertia"}, {"joint", "parent child"},
        {"visual", "geometry"}, {"collision", "geometry"}
    };
    std::function<void(Element*)> walk = [&](Element* e) {
        if (std::string(e->Name()) == "inertia") {
            double x = 0, y = 0, z = 0;
            if (e->QueryDoubleAttribute("ixx", &x) == tinyxml2::XML_SUCCESS &&
                e->QueryDoubleAttribute("iyy", &y) == tinyxml2::XML_SUCCESS &&
                e->QueryDoubleAttribute("izz", &z) == tinyxml2::XML_SUCCESS &&
                std::isfinite(x) && std::isfinite(y) && std::isfinite(z)) {
                const double tolerance = 1e-6 * std::max({std::abs(x), std::abs(y), std::abs(z)});
                if (x > y + z + tolerance || y > x + z + tolerance || z > x + y + tolerance)
                    Diagnose(result, e, UrdfDiagnosticKind::Invalid, "URDF_INERTIA_INVALID", "Principal inertia violates triangle inequalities");
            }
        }
        const auto found = schema.find(e->Name());
        if (found == schema.end()) return;
        for (auto* a = e->FirstAttribute(); a; a = a->Next()) {
            const std::string key = a->Name();
            if (key == "xmlns") {
                if (*a->Value()) Diagnose(result, e, UrdfDiagnosticKind::Unsupported,
                    "URDF_NAMESPACE_UNSUPPORTED", "Default element namespace is not interpreted by this profile");
                continue;
            }
            if (key.rfind("xmlns:", 0) == 0) continue;
            if (!Listed(key, found->second.first)) {
                Diagnose(result, e, UrdfDiagnosticKind::Unsupported, "URDF_ATTRIBUTE_UNSUPPORTED",
                         "Unrepresented attribute: " + key);
                continue;
            }
            if (Listed(key, "name type link filename")) continue;
            const size_t count = Listed(key, "xyz rpy size scale") ? 3u : 1u;
            std::istringstream input(a->Value());
            std::vector<double> values(count);
            bool valid = true;
            for (auto& value : values) {
                if (!(input >> value) || !std::isfinite(value) ||
                    std::abs(value) > std::numeric_limits<float>::max()) valid = false;
            }
            input >> std::ws;
            if (!valid || input.peek() != std::char_traits<char>::eof()) {
                Diagnose(result, e, UrdfDiagnosticKind::Invalid, "URDF_NUMBER_INVALID",
                         "Expected finite numeric value(s): " + key);
                continue;
            }
            if (Listed(key, "ixy ixz iyz") && values[0] != 0.0)
                Diagnose(result, e, UrdfDiagnosticKind::Unsupported, "URDF_FULL_INERTIA_UNSUPPORTED",
                         "Nonzero inertia cross terms are not represented");
            if (Listed(key, "value ixx iyy izz radius size") &&
                std::any_of(values.begin(), values.end(), [](double v) { return static_cast<float>(v) <= 0.0f; }))
                Diagnose(result, e, UrdfDiagnosticKind::Invalid, "URDF_POSITIVE_VALUE_REQUIRED",
                         "Expected strictly positive value(s): " + key);
            if (Listed(key, "value ixx iyy izz") &&
                !std::isfinite(1.0f / static_cast<float>(values[0])))
                Diagnose(result, e, UrdfDiagnosticKind::Invalid, "URDF_INVERSE_NONFINITE",
                         "Value has no finite float inverse: " + key);
            if (key == "size" && std::any_of(values.begin(), values.end(), [](double v) {
                    return static_cast<float>(v) * 0.5f <= 0.0f;
                }))
                Diagnose(result, e, UrdfDiagnosticKind::Invalid, "URDF_DIMENSION_UNDERFLOW", "Half extents must remain positive");
            if (key == "scale" && std::any_of(values.begin(), values.end(), [](double v) { return static_cast<float>(v) == 0.0f; }))
                Diagnose(result, e, UrdfDiagnosticKind::Invalid, "URDF_SCALE_INVALID", "Mesh scale cannot be zero");
            if (std::string(e->Name()) == "axis" && key == "xyz") {
                const double squared = values[0] * values[0] + values[1] * values[1] + values[2] * values[2];
                if (std::abs(squared - 1.0) > 1e-5)
                    Diagnose(result, e, UrdfDiagnosticKind::Invalid, "URDF_AXIS_INVALID", "Strict joint axis must have unit length");
            }
        }
        std::unordered_set<std::string> children;
        for (auto* c = e->FirstChildElement(); c; c = c->NextSiblingElement()) {
            const std::string tag = c->Name();
            if (!Listed(tag, found->second.second)) {
                Diagnose(result, c, UrdfDiagnosticKind::Unsupported, "URDF_ELEMENT_UNSUPPORTED",
                         "Unrepresented element under " + std::string(e->Name()) + ": " + tag);
                continue;
            }
            if (!children.insert(tag).second && std::string(e->Name()) != "robot" &&
                !(std::string(e->Name()) == "link" && Listed(tag, "visual collision")))
                Diagnose(result, c, UrdfDiagnosticKind::Invalid, "URDF_DUPLICATE_ELEMENT", "Repeated singleton element");
            walk(c);
        }
        auto required = required_attributes.find(e->Name());
        if (required != required_attributes.end()) {
            std::istringstream keys(required->second);
            std::string key;
            while (keys >> key)
                if (!e->Attribute(key.c_str()) || !*e->Attribute(key.c_str()))
                    Diagnose(result, e, UrdfDiagnosticKind::Invalid, "URDF_ATTRIBUTE_REQUIRED", "Missing attribute: " + key);
        }
        auto required_child = required_children.find(e->Name());
        if (required_child != required_children.end()) {
            std::istringstream tags(required_child->second);
            std::string tag;
            while (tags >> tag)
                if (!e->FirstChildElement(tag.c_str()))
                    Diagnose(result, e, UrdfDiagnosticKind::Unsupported, "URDF_DEFAULT_UNREPRESENTED",
                             "Strict profile requires explicit " + tag + " to avoid synthetic defaults");
        }
        if (std::string(e->Name()) == "geometry" &&
            (!e->FirstChildElement() || e->FirstChildElement()->NextSiblingElement()))
            Diagnose(result, e, UrdfDiagnosticKind::Invalid, "URDF_GEOMETRY_INVALID", "Expected exactly one geometry");
    };
    walk(robot);
    if (robot->NextSiblingElement() || robot->PreviousSiblingElement())
        Diagnose(result, robot, UrdfDiagnosticKind::Invalid, "URDF_ROOT_INVALID", "Expected a single robot root");
    std::unordered_set<std::string> links, joints;
    for (auto* link = robot->FirstChildElement("link"); link; link = link->NextSiblingElement("link")) {
        const char* name = link->Attribute("name");
        if (name && !links.insert(name).second)
            Diagnose(result, link, UrdfDiagnosticKind::Invalid, "URDF_DUPLICATE_NAME", "Duplicate link name");
    }
    std::unordered_map<std::string, std::string> parents;
    for (auto* joint = robot->FirstChildElement("joint"); joint; joint = joint->NextSiblingElement("joint")) {
        const char* name = joint->Attribute("name");
        if (name && !joints.insert(name).second)
            Diagnose(result, joint, UrdfDiagnosticKind::Invalid, "URDF_DUPLICATE_NAME", "Duplicate joint name");
        const char* type = joint->Attribute("type");
        if (type && !Listed(type, "fixed revolute continuous prismatic floating"))
            Diagnose(result, joint, UrdfDiagnosticKind::Unsupported, "URDF_JOINT_UNSUPPORTED", "Joint type is not represented exactly");
        std::string endpoints[2];
        for (int i = 0; i < 2; ++i) {
            auto* endpoint = joint->FirstChildElement(i == 0 ? "parent" : "child");
            const char* link = endpoint ? endpoint->Attribute("link") : nullptr;
            if (link) endpoints[i] = link;
            if (!link || !links.count(link))
                Diagnose(result, endpoint ? endpoint : joint, UrdfDiagnosticKind::Invalid,
                         "URDF_LINK_UNRESOLVED", "Joint endpoint does not resolve to a link");
        }
        if (endpoints[0] == endpoints[1] || !parents.emplace(endpoints[1], endpoints[0]).second)
            Diagnose(result, joint, UrdfDiagnosticKind::Invalid, "URDF_TOPOLOGY_INVALID", "Self joint or multiple parents");
        auto* limit = joint->FirstChildElement("limit");
        if (type && Listed(type, "revolute prismatic") &&
            (!limit || !limit->Attribute("lower") || !limit->Attribute("upper")))
            Diagnose(result, limit ? limit : joint, UrdfDiagnosticKind::Unsupported,
                     "URDF_LIMIT_DEFAULT_UNREPRESENTED", "Strict scalar joints require explicit lower and upper limits");
        double lower = 0, upper = 0;
        if (limit && limit->QueryDoubleAttribute("lower", &lower) == tinyxml2::XML_SUCCESS &&
            limit->QueryDoubleAttribute("upper", &upper) == tinyxml2::XML_SUCCESS && lower > upper)
            Diagnose(result, limit, UrdfDiagnosticKind::Invalid, "URDF_LIMIT_INVALID", "Lower limit exceeds upper limit");
        if (type && std::string(type) == "continuous" && limit &&
            (limit->Attribute("lower") || limit->Attribute("upper")))
            Diagnose(result, limit, UrdfDiagnosticKind::Unsupported, "URDF_CONTINUOUS_LIMIT_UNSUPPORTED",
                     "Continuous joints cannot acquire position limits in a strict projection");
    }
    size_t roots = 0;
    for (const auto& link : links) {
        if (!parents.count(link)) ++roots;
        std::unordered_set<std::string> seen;
        std::string cursor = link;
        while (parents.count(cursor)) {
            if (!seen.insert(cursor).second) {
                Diagnose(result, robot, UrdfDiagnosticKind::Invalid, "URDF_TOPOLOGY_CYCLE", "Link hierarchy contains a cycle");
                break;
            }
            cursor = parents.at(cursor);
        }
    }
    if (!links.empty() && roots != 1u)
        Diagnose(result, robot, UrdfDiagnosticKind::Invalid, "URDF_ROOT_COUNT_INVALID", "Expected one root link");
    if (links.empty()) Diagnose(result, robot, UrdfDiagnosticKind::Invalid, "URDF_LINK_REQUIRED", "Robot has no links");
}
} // namespace

UrdfImportResult ImportUrdf(const std::string& path, UrdfImportProfile profile) {
    UrdfImportResult result;
    result.profile = profile;
    tinyxml2::XMLDocument doc;
    const auto error = doc.LoadFile(path.c_str());
    if (error != tinyxml2::XML_SUCCESS) {
        const bool io = error == tinyxml2::XML_ERROR_FILE_NOT_FOUND ||
                        error == tinyxml2::XML_ERROR_FILE_COULD_NOT_BE_OPENED ||
                        error == tinyxml2::XML_ERROR_FILE_READ_ERROR;
        result.diagnostics.push_back({io ? UrdfDiagnosticKind::Io : UrdfDiagnosticKind::Syntax,
            io ? "URDF_IO_ERROR" : "URDF_XML_ERROR", path, doc.ErrorLineNum(), "", doc.ErrorStr()});
        return result;
    }
    auto* robot = doc.FirstChildElement("robot");
    if (!robot) {
        result.diagnostics.push_back({UrdfDiagnosticKind::Invalid, "URDF_ROOT_REQUIRED", path, 0, "", "Missing robot root"});
        return result;
    }
    ValidateUrdf(robot, result);
    for (auto& diagnostic : result.diagnostics) diagnostic.source = path;
    if (profile == UrdfImportProfile::Strict && !result.diagnostics.empty()) return result;
    try {
        result.scene = ParseUrdf(robot, path);
        bool finite_mesh = true;
        for (const auto& shape : result.scene->Shapes()) {
            for (const auto* stream : {&shape.mesh_vertices, &shape.mesh_normals, &shape.mesh_uvs})
                for (float value : *stream) if (!std::isfinite(value)) finite_mesh = false;
        }
        if (!finite_mesh) {
            result.diagnostics.push_back({UrdfDiagnosticKind::Invalid, "URDF_MESH_NONFINITE", path,
                robot->GetLineNum(), "robot", "Decoded or scaled mesh contains nonfinite values"});
            if (profile == UrdfImportProfile::Strict) result.scene.reset();
        }
    } catch (const std::exception& failure) {
        result.diagnostics.push_back({UrdfDiagnosticKind::Invalid, "URDF_PROJECTION_FAILED", path,
                                      robot->GetLineNum(), "robot", failure.what()});
    }
    return result;
}

scene::SceneIR LoadUrdf(const std::string& path) {
    tinyxml2::XMLDocument doc;
    if (doc.LoadFile(path.c_str()) != tinyxml2::XML_SUCCESS)
        throw std::runtime_error("URDF: failed to load file: " + path);
    auto* robot = doc.FirstChildElement("robot");
    if (!robot) throw std::runtime_error("URDF: missing <robot> root element in " + path);
    return ParseUrdf(robot, path);
}

} // namespace nuka::import
