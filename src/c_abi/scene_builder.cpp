// ---------------------------------------------------------------------------
// nuka::c_abi -- the BUILT-SCENE authoring surface (program a SceneIR of rigid
// primitives + media records, then cook it through the SAME CookSceneToModel a
// file scene uses). This is the GENERAL media authoring substrate: a tet-soft
// body / a fluid box / a second cloth are added as TAGGED MediaRecords via
// nuka_scene_add_media, never by accreting flat slots onto the coupled desc.
//
// It REUSES the existing scene handle (SceneRecord / SceneTable, shared with the
// load/edit surface in scene.cpp) and the shared world-create helpers
// (ValidateWorldDescAndDevice / MakeWorldRecipe / CreateWorld in
// world.cpp) -- so the built-scene world is ONE nk::World on the ONE cook path,
// not a second world type. Host-only (zero CUDA tokens); only the world create
// touches a device (via the shared helpers).
// ---------------------------------------------------------------------------

#include "nuka/nuka_scene.h"

#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "runtime/articulation/control_mode.hpp"
#include "scene/canonical_types.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/scene_ir.hpp"
#include "nk/material/hencky_j2.hpp"
#include "scene/terrain/heightfield.hpp"

#include <exception>
#include <cmath>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using nuka::c_abi::DeviceRecord;
using nuka::c_abi::SceneRecord;
using nuka::c_abi::SceneTable;  // the shared table (defined in scene.cpp).
namespace cook = nuka::scene::cook;
namespace articulation = nuka::runtime::articulation;
namespace nscene = nuka::scene;
namespace nmath = nuka::math;

bool ApplyMpmPlasticity(const nuka_mpm_plasticity_desc_t* desc,
                       nscene::MediaMpmMaterial& material) {
    if (desc == nullptr) return material.model_kind != nuka::nk::MpmMaterial::kHenckyJ2;
    if (desc->struct_size < sizeof(*desc) ||
        material.model_kind != nuka::nk::MpmMaterial::kHenckyJ2 ||
        !std::isfinite(material.density) || !(material.density > 0.0f) ||
        !nuka::nk::material::ValidHenckyJ2({material.youngs, material.poisson,
                                         desc->yield_stress, desc->hardening_modulus})) return false;
    material.yield_stress = desc->yield_stress;
    material.hardening_modulus = desc->hardening_modulus;
    return true;
}

// A primitive's world pose; an all-zero quat reads as identity (a zero-init desc
// then sits at the origin upright, the ergonomic default).
nmath::Transform PrimitivePose(const nuka_rigid_primitive_desc_t& d) {
    nmath::Transform t = nmath::Transform::Identity();
    t.position = nmath::Vec3{d.pos[0], d.pos[1], d.pos[2]};
    const float qn = d.quat[0] * d.quat[0] + d.quat[1] * d.quat[1] +
                     d.quat[2] * d.quat[2] + d.quat[3] * d.quat[3];
    if (qn > 1.0e-12f) {
        t.rotation = nmath::Quat{d.quat[0], d.quat[1], d.quat[2], d.quat[3]};
    }
    return t;
}

// Diagonal inertia of a uniform solid primitive about its center of mass.
nmath::Vec3 PrimitiveInertia(uint32_t kind, const float dims[3], float mass) {
    switch (kind) {
        case NUKA_PRIMITIVE_BOX: {
            const float hx = dims[0], hy = dims[1], hz = dims[2];
            const float c = mass / 3.0f;  // solid box about its centre.
            return nmath::Vec3{c * (hy * hy + hz * hz), c * (hx * hx + hz * hz),
                               c * (hx * hx + hy * hy)};
        }
        case NUKA_PRIMITIVE_SPHERE: {
            const float i = 0.4f * mass * dims[0] * dims[0];  // 2/5 m r^2.
            return nmath::Vec3{i, i, i};
        }
        case NUKA_PRIMITIVE_CAPSULE: {
            const float r = dims[0], hh = dims[1];
            const float cylinder_mass = mass * (3.0f * hh) / (3.0f * hh + 2.0f * r);
            const float cap_mass = mass - cylinder_mass;
            const float axial = (0.5f * cylinder_mass + 0.4f * cap_mass) * r * r;
            const float trans = cylinder_mass * (0.25f * r * r + hh * hh / 3.0f) +
                cap_mass * (0.4f * r * r + hh * hh + 0.75f * hh * r);
            return nmath::Vec3{trans, trans, axial};
        }
        case NUKA_PRIMITIVE_CYLINDER: {
            const float r = dims[0], hh = dims[1];
            const float trans = mass * (3.0f * r * r + 4.0f * hh * hh) / 12.0f;
            return nmath::Vec3{trans, trans, 0.5f * mass * r * r};
        }
        default:  // PLANE: static, inertia irrelevant.
            return nmath::Vec3{1.0f, 1.0f, 1.0f};
    }
}

// Fill a scene MediaRecord from the TAGGED desc (kind/method select the geometry +
// material block read). Returns false on an unknown kind/method (caller -> INVALID_ARG).
bool MediaFromDesc(const nuka_media_desc_t& d, nscene::MediaRecord* out) {
    using Kind = nscene::MediaRecord::Kind;
    using Method = nscene::MediaRecord::Method;
    nscene::MediaRecord m;
    switch (d.kind) {
        case NUKA_MEDIA_CLOTH:    m.kind = Kind::Cloth; break;
        case NUKA_MEDIA_SOFT_TET: m.kind = Kind::SoftTet; break;
        case NUKA_MEDIA_FLUID:    m.kind = Kind::Fluid; break;
        case NUKA_MEDIA_GRANULAR: m.kind = Kind::Granular; break;
        case NUKA_MEDIA_CABLE:    m.kind = Kind::Cable; break;
        default: return false;
    }
    switch (d.method) {
        case NUKA_MEDIA_METHOD_MLSMPM: m.method = Method::MlsMpm; break;
        case NUKA_MEDIA_METHOD_VBD:    m.method = Method::Vbd; break;
        case NUKA_MEDIA_METHOD_FIELD:  m.method = Method::Field; break;
        default: return false;
    }
    // Geometry (one block read per kind).
    m.cloth_grid.nx = d.cloth_nx;
    m.cloth_grid.ny = d.cloth_ny;
    m.cloth_grid.spacing = d.cloth_spacing;
    m.cloth_grid.origin = nmath::Vec3{d.cloth_origin[0], d.cloth_origin[1],
                                      d.cloth_origin[2]};
    m.cloth_grid.free = (d.cloth_free != 0u);
    if (d.cloth_pin >
        static_cast<uint32_t>(nscene::MediaRecord::ClothPin::EdgeY1)) {
        return false;
    }
    m.cloth_grid.pin = static_cast<nscene::MediaRecord::ClothPin>(d.cloth_pin);
    m.tet_sphere.center = nmath::Vec3{d.tet_center[0], d.tet_center[1],
                                      d.tet_center[2]};
    m.tet_sphere.radius = d.tet_radius;
    m.tet_sphere.cells = d.tet_cells;
    m.tet_sphere.cell_len = d.tet_cell_len;
    m.fluid_box.min = nmath::Vec3{d.fluid_min[0], d.fluid_min[1], d.fluid_min[2]};
    m.fluid_box.max = nmath::Vec3{d.fluid_max[0], d.fluid_max[1], d.fluid_max[2]};
    m.fluid_box.spacing = d.fluid_spacing;
    m.fluid_box.position_jitter = d.fluid_position_jitter;
    m.cable_line.start = nmath::Vec3{d.cable_start[0], d.cable_start[1], d.cable_start[2]};
    m.cable_line.end = nmath::Vec3{d.cable_end[0], d.cable_end[1], d.cable_end[2]};
    m.cable_line.segments = d.cable_segments;
    m.cable_line.radius = d.cable_radius;
    if (d.cable_pin >
        static_cast<uint32_t>(nscene::MediaRecord::CableLine::Pin::None)) {
        return false;
    }
    m.cable_line.pin = static_cast<nscene::MediaRecord::CableLine::Pin>(d.cable_pin);
    m.cable_line.slab.half_extents = nmath::Vec3{d.cable_slab_half_extents[0],
        d.cable_slab_half_extents[1], d.cable_slab_half_extents[2]};
    m.cable_line.slab.mass = d.cable_slab_mass;
    m.cable_line.slab.stiffness = d.cable_slab_stiffness;
    m.cable_line.slab.render_material_id = d.cable_slab_render_material_id;
    // Constitutive material (one block read per method).
    m.vbd.particle_mass = d.vbd_particle_mass;
    m.vbd.friction = d.vbd_friction;
    m.vbd.half_thickness = d.vbd_half_thickness;
    m.vbd.surface_density = d.vbd_surface_density;
    m.vbd.stretch_stiffness = d.vbd_stretch_stiffness;
    m.vbd.poisson = d.vbd_poisson;
    m.vbd.bend_stiffness = d.vbd_bend_stiffness;
    m.vbd.damping = d.vbd_damping;
    m.vbd.aero_drag_normal = d.vbd_aero_drag_normal;
    m.vbd.aero_drag_tangent = d.vbd_aero_drag_tangent;
    m.vbd.aero_drag_max_dv = d.vbd_aero_drag_max_dv;
    m.mpm.youngs = d.mpm_youngs;
    m.mpm.poisson = d.mpm_poisson;
    m.mpm.density = d.mpm_density;
    m.mpm.dp_friction = d.mpm_dp_friction;
    m.mpm.dp_cohesion = d.mpm_dp_cohesion;
    m.mpm.model_kind = d.mpm_model_kind;
    m.mpm.bulk_modulus = d.mpm_bulk_modulus;
    m.mpm.tait_gamma = d.mpm_tait_gamma;
    m.mpm.viscosity = d.mpm_viscosity;
    m.mpm.dx = d.mpm_dx;
    m.mpm.substeps = d.mpm_substeps;
    m.mpm.floor_normal = nmath::Vec3{d.mpm_floor_normal[0], d.mpm_floor_normal[1],
                                     d.mpm_floor_normal[2]};
    m.mpm.floor_d = d.mpm_floor_d;
    m.mpm.floor_friction = d.mpm_floor_friction;
    m.mpm.loft_headroom = d.mpm_loft_headroom;
    // Render skin.
    m.render_skin.normal_offset = d.skin_normal_offset;
    m.render_skin.smooth_iters = d.skin_smooth_iters;
    m.render_skin.smooth_lambda = d.skin_smooth_lambda;
    m.render_skin.grain_round = d.skin_grain_round;
    m.render_skin.grain_radius_jitter = d.skin_grain_radius_jitter;
    m.render_skin.grain_tint_jitter = d.skin_grain_tint_jitter;
    m.render_material_id = d.render_material_id;
    *out = std::move(m);
    return true;
}

}  // namespace

extern "C" {

nuka_scene_handle nuka_scene_create(const char* base_scene_path) {
    // A non-empty base path imports that scene through the SAME load path
    // nuka_scene_load uses (one entry, any format); NULL/"" starts an empty scene.
    if (base_scene_path != nullptr && base_scene_path[0] != '\0') {
        nuka_scene_handle handle = nullptr;
        if (nuka_scene_load(base_scene_path, &handle) != NUKA_RESULT_OK) {
            return nullptr;
        }
        return handle;
    }
    try {
        auto record = std::make_unique<SceneRecord>();
        record->scene = std::make_unique<nscene::SceneIR>();
        return SceneTable().Insert(std::move(record));
    } catch (...) {
        return nullptr;
    }
}

nuka_result_t nuka_scene_add_rigid_primitive(
    nuka_scene_handle scene, const nuka_rigid_primitive_desc_t* desc) {
    if (desc == nullptr) return NUKA_RESULT_INVALID_ARG;
    SceneRecord* sr = SceneTable().Get(scene);
    if (sr == nullptr || !sr->scene) return NUKA_RESULT_NULL_HANDLE;
    try {
        nscene::ShapeType type;
        const char* kind_name;
        switch (desc->kind) {
            case NUKA_PRIMITIVE_PLANE:
                type = nscene::ShapeType::Plane; kind_name = "plane"; break;
            case NUKA_PRIMITIVE_BOX:
                type = nscene::ShapeType::Box; kind_name = "box"; break;
            case NUKA_PRIMITIVE_SPHERE:
                type = nscene::ShapeType::Sphere; kind_name = "sphere"; break;
            case NUKA_PRIMITIVE_CAPSULE:
                type = nscene::ShapeType::Capsule; kind_name = "capsule"; break;
            case NUKA_PRIMITIVE_CYLINDER:
                type = nscene::ShapeType::Cylinder; kind_name = "cylinder"; break;
            default:
                return NUKA_RESULT_INVALID_ARG;
        }
        // A box/sphere/capsule/cylinder needs positive extents (zero/negative cooks degenerate
        // geometry); a plane carries no extent.
        if (type == nscene::ShapeType::Sphere && desc->dims[0] <= 0.0f) {
            return NUKA_RESULT_INVALID_ARG;
        }
        if ((type == nscene::ShapeType::Capsule || type == nscene::ShapeType::Cylinder) &&
            (desc->dims[0] <= 0.0f || desc->dims[1] <= 0.0f)) {
            return NUKA_RESULT_INVALID_ARG;
        }
        if (type == nscene::ShapeType::Box &&
            (desc->dims[0] <= 0.0f || desc->dims[1] <= 0.0f || desc->dims[2] <= 0.0f)) {
            return NUKA_RESULT_INVALID_ARG;
        }
        // A plane is inherently static (an infinite movable plane is nonsensical).
        const bool is_static =
            (type == nscene::ShapeType::Plane) ? true : (desc->is_static != 0u);
        const float mass = desc->mass > 0.0f ? desc->mass : 1.0f;

        nscene::RigidBodyRecord body;
        body.name = std::string(kind_name) + "_" +
                    std::to_string(sr->scene->RigidBodyCount());
        body.is_static = is_static;
        body.local_transform = PrimitivePose(*desc);
        body.mass = mass;
        body.inertia = PrimitiveInertia(desc->kind, desc->dims, mass);
        const nscene::BodyId bid = sr->scene->AddRigidBody(std::move(body));

        nscene::CollisionShapeRecord shape;
        shape.body_id = bid;
        shape.name = std::string(kind_name) + "_shape_" +
                     std::to_string(sr->scene->ShapeCount());
        shape.type = type;
        if (type == nscene::ShapeType::Box) {
            shape.half_extents =
                nmath::Vec3{desc->dims[0], desc->dims[1], desc->dims[2]};
        } else if (type == nscene::ShapeType::Sphere) {
            shape.radius = desc->dims[0];
        } else if (type == nscene::ShapeType::Capsule || type == nscene::ShapeType::Cylinder) {
            shape.radius = desc->dims[0];
            shape.half_height = desc->dims[1];
        }
        // friction < 0 keeps the inherit-material sentinel; >= 0 is an explicit
        // per-shape override (the SAME precedence the cooker resolves).
        if (desc->friction >= 0.0f) shape.friction_mu = desc->friction;

        // An authored render material adds a VISUAL-ONLY twin shape carrying it
        // (contype/conaffinity 0 -- the imported-scene visual-geom pattern), so
        // the primitive renders with that appearance. No material => today's
        // collision-only record, byte-identical.
        nscene::MaterialId material_id = nscene::kInvalidMaterial;
        if (desc->material_name != nullptr && desc->material_name[0] != '\0') {
            for (const nscene::MaterialRecord& m : sr->scene->Materials()) {
                if (m.name == desc->material_name) { material_id = m.id; break; }
            }
            if (material_id == nscene::kInvalidMaterial) {
                return NUKA_RESULT_INVALID_ARG;  // unknown material name (add first).
            }
        }
        nscene::CollisionShapeRecord visual;
        if (material_id != nscene::kInvalidMaterial) {
            visual = shape;
            visual.name = std::string(kind_name) + "_visual_" +
                          std::to_string(sr->scene->ShapeCount() + 1u);
            visual.contype = 0;
            visual.conaffinity = 0;
            visual.material_id = material_id;
        }
        sr->scene->AddCollisionShape(std::move(shape));
        if (material_id != nscene::kInvalidMaterial) {
            sr->scene->AddCollisionShape(std::move(visual));
        }
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_scene_add_material(nuka_scene_handle scene,
                                      const nuka_render_material_desc_t* desc,
                                      uint32_t* out_material_id) {
    if (desc == nullptr || desc->name == nullptr || desc->name[0] == '\0') {
        return NUKA_RESULT_INVALID_ARG;
    }
    SceneRecord* sr = SceneTable().Get(scene);
    if (sr == nullptr || !sr->scene) return NUKA_RESULT_NULL_HANDLE;
    try {
        nscene::MaterialRecord rec;
        rec.name = desc->name;
        // An all-zero base color reads as opaque white (the zero-init ergonomic
        // default; a black material authors an explicit tiny non-zero color).
        const bool zero_color = desc->base_color[0] == 0.0f &&
                                desc->base_color[1] == 0.0f &&
                                desc->base_color[2] == 0.0f &&
                                desc->base_color[3] == 0.0f;
        rec.base_color = zero_color
            ? nmath::Vec3{1.0f, 1.0f, 1.0f}
            : nmath::Vec3{desc->base_color[0], desc->base_color[1], desc->base_color[2]};
        rec.alpha = zero_color ? 1.0f : desc->base_color[3];
        rec.metallic = desc->metallic;
        rec.roughness = desc->roughness > 0.0f ? desc->roughness : 0.5f;
        rec.sheen = desc->sheen;
        if (desc->albedo_map != nullptr) rec.albedo_map = desc->albedo_map;
        if (desc->roughness_map != nullptr) rec.roughness_map = desc->roughness_map;
        if (desc->normal_map != nullptr) rec.normal_map = desc->normal_map;
        rec.uv_scale = desc->uv_scale > 0.0f ? desc->uv_scale : 1.0f;
        rec.triplanar = desc->triplanar != 0u;
        const nscene::MaterialId id = sr->scene->AddMaterial(std::move(rec));
        if (out_material_id != nullptr) *out_material_id = id;
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_scene_set_environment(nuka_scene_handle scene,
                                         const nuka_environment_desc_t* desc) {
    if (desc == nullptr) return NUKA_RESULT_INVALID_ARG;
    SceneRecord* sr = SceneTable().Get(scene);
    if (sr == nullptr || !sr->scene) return NUKA_RESULT_NULL_HANDLE;
    try {
        nscene::EnvironmentRecord rec;
        if (desc->hdri != nullptr) rec.hdri = desc->hdri;
        rec.yaw_deg = desc->yaw_deg;
        rec.intensity = desc->intensity > 0.0f ? desc->intensity : 1.0f;
        rec.use_scene_materials = desc->use_scene_materials != 0u;
        rec.ibl_full_fill = desc->ibl_full_fill != 0u;
        rec.exposure_ev = desc->exposure_ev;
        rec.grade = desc->grade;
        rec.sun_disc = desc->sun_disc;
        rec.specular_env = desc->specular_env != 0u;
        sr->scene->EnvironmentMut() = std::move(rec);
        return NUKA_RESULT_OK;
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_scene_add_media(nuka_scene_handle scene,
                                   const nuka_media_desc_t* desc,
                                   uint32_t* out_media_id) {
    return nuka_scene_add_media_ex(scene, desc, nullptr, out_media_id);
}

nuka_result_t nuka_scene_add_media_ex(nuka_scene_handle scene,
                                    const nuka_media_desc_t* desc,
                                    const nuka_mpm_plasticity_desc_t* plasticity,
                                    uint32_t* out_media_id) {
    if (desc == nullptr) return NUKA_RESULT_INVALID_ARG;
    SceneRecord* sr = SceneTable().Get(scene);
    if (sr == nullptr || !sr->scene) return NUKA_RESULT_NULL_HANDLE;
    try {
        nscene::MediaRecord media;
        if (!MediaFromDesc(*desc, &media)) {
            return NUKA_RESULT_INVALID_ARG;  // unknown kind / method.
        }
        if (!ApplyMpmPlasticity(plasticity, media.mpm) ||
            (plasticity != nullptr && media.method != nscene::MediaRecord::Method::MlsMpm))
            return NUKA_RESULT_INVALID_ARG;
        // Reject an illegal (kind x method) immediately via the SAME cook validator
        // (a single-record list; the cross-medium mix is caught at cook).
        cook::ValidateMedia(std::vector<nscene::MediaRecord>{media});
        media.name = "media_" + std::to_string(sr->scene->MediaCount());
        const nscene::MediaId id = sr->scene->AddMedia(std::move(media));
        if (out_media_id != nullptr) *out_media_id = id;
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception&) {
        // ValidateMedia throws on an illegal pair -> a caller error (LOUD).
        return NUKA_RESULT_INVALID_ARG;
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_scene_set_mpm_contact_capacity(nuka_scene_handle scene,
                                                  uint32_t media_id, uint32_t capacity) {
    SceneRecord* sr = SceneTable().Get(scene);
    if (sr == nullptr || !sr->scene) return NUKA_RESULT_NULL_HANDLE;
    if (media_id >= sr->scene->MediaCount()) return NUKA_RESULT_INVALID_ARG;
    auto& media = sr->scene->GetMediaMut(media_id);
    if (media.method != nscene::MediaRecord::Method::MlsMpm) return NUKA_RESULT_INVALID_ARG;
    media.mpm.contact_capacity = capacity;
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_scene_add_field_media(nuka_scene_handle scene,
                                         const nuka_media_desc_t* desc,
                                         const nuka_field_material_desc_t* field,
                                         uint32_t* out_media_id) {
    static_assert(NUKA_FIELD_HISTORY_VERSION == nuka::nk::kMfStateVersion, "public history schema must match the engine");
    static_assert(NUKA_FIELD_LAW_SMALL_STRAIN == nuka::nk::kMfSmallStrainLaw &&
                  NUKA_FIELD_LAW_LOG_STRAIN_ORTHOTROPIC == nuka::nk::kMfLogStrainOrthotropicLaw,
                  "public field laws must match the engine");
    static_assert(NUKA_FIELD_FRACTURE_ISOTROPIC == nuka::nk::kMfIsotropicCohesive &&
                  NUKA_FIELD_FRACTURE_MIXED_MODE == nuka::nk::kMfMixedModeCohesive,
                  "public fracture laws must match the engine");
    const bool logarithmic = field != nullptr && field->struct_size >= offsetof(nuka_field_material_desc_t, fracture_law) &&
                             field->law == NUKA_FIELD_LAW_LOG_STRAIN_ORTHOTROPIC;
    const bool mixed = field != nullptr && field->struct_size == sizeof(nuka_field_material_desc_t) &&
                       field->fracture_law == NUKA_FIELD_FRACTURE_MIXED_MODE;
    const nuka_field_contract_desc_t contract{sizeof(nuka_field_contract_desc_t), 1u, logarithmic ? 2u : 1u, 1u,
                                              mixed ? 2u : 1u, NUKA_FIELD_HISTORY_VERSION};
    return nuka_scene_add_field_media_v1(scene, desc, field, &contract, out_media_id);
}

nuka_result_t nuka_scene_add_field_media_v1(nuka_scene_handle scene,
    const nuka_media_desc_t* desc, const nuka_field_material_desc_t* field,
    const nuka_field_contract_desc_t* contract, uint32_t* out_media_id) {
    if (!contract || contract->struct_size != sizeof(nuka_field_contract_desc_t)) return NUKA_RESULT_INVALID_ARG;
    const nuka::nk::ContractVersions versions{contract->schema_version, contract->material_version,
        contract->section_version, contract->fracture_version, contract->history_version};
    if (!versions.Supported()) return NUKA_RESULT_NOT_SUPPORTED;
    if (desc == nullptr || field == nullptr ||
        (field->struct_size != offsetof(nuka_field_material_desc_t, shear_strength) &&
         field->struct_size != offsetof(nuka_field_material_desc_t, law) &&
         field->struct_size != offsetof(nuka_field_material_desc_t, fracture_law) &&
         field->struct_size != sizeof(nuka_field_material_desc_t)) ||
        desc->method != NUKA_MEDIA_METHOD_FIELD)
        return NUKA_RESULT_INVALID_ARG;
    SceneRecord* sr = SceneTable().Get(scene);
    if (sr == nullptr || !sr->scene) return NUKA_RESULT_NULL_HANDLE;
    try {
        nscene::MediaRecord media;
        if (!MediaFromDesc(*desc, &media)) return NUKA_RESULT_INVALID_ARG;
        nscene::MediaFieldMaterial& m = media.field;
        m.contract = versions;
        m.density = field->density;
        m.thickness = field->thickness;
        m.youngs = field->youngs;
        m.poisson = field->poisson;
        m.yield_stress = field->yield_stress;
        m.hardening = field->hardening;
        m.yield_moment = field->yield_moment;
        m.bend_hardening = field->bend_hardening;
        for (int i = 0; i < 2; ++i) {
            m.maxwell_modulus[i] = field->maxwell_modulus[i];
            m.maxwell_time[i] = field->maxwell_time[i];
            m.bend_maxwell_modulus[i] = field->bend_maxwell_modulus[i];
            m.bend_maxwell_time[i] = field->bend_maxwell_time[i];
        }
        m.tear_strength = field->tear_strength;
        m.tear_energy = field->tear_energy;
        m.weibull_modulus = field->weibull_modulus;
        for (int i = 0; i < 3; ++i) m.grain[i] = field->grain[i];
        m.grain_ratio = field->grain_ratio;
        if (field->struct_size >= offsetof(nuka_field_material_desc_t, law)) {
            m.shear_strength = field->shear_strength;
            m.weibull_reference_area = field->weibull_reference_area;
        }
        if (field->struct_size >= offsetof(nuka_field_material_desc_t, fracture_law)) {
            m.law = field->law == 0u ? NUKA_FIELD_LAW_SMALL_STRAIN : field->law;
            m.youngs_cd = field->youngs_cd;
            m.shear_modulus = field->shear_modulus;
            m.yield_stress_cd = field->yield_stress_cd;
            m.yield_shear = field->yield_shear;
            m.yield_biaxial = field->yield_biaxial;
            m.strain_range = field->strain_range;
        }
        if (field->struct_size >= offsetof(nuka_field_material_desc_t, section_law)) {
            m.fracture_law = field->fracture_law == 0u ? NUKA_FIELD_FRACTURE_ISOTROPIC : field->fracture_law;
            m.shear_tear_energy = field->shear_tear_energy;
            m.mixity_exponent = field->mixity_exponent;
        }
        if (field->struct_size == sizeof(nuka_field_material_desc_t)) {
            m.section_law = field->section_law == 0u ? NUKA_FIELD_SECTION_PLATE : field->section_law;
            m.bend_stiffness = field->bend_stiffness;
            m.curvature_range = field->curvature_range;
        }
        cook::ValidateMedia(std::vector<nscene::MediaRecord>{media});
        media.name = "media_" + std::to_string(sr->scene->MediaCount());
        const nscene::MediaId id = sr->scene->AddMedia(std::move(media));
        if (out_media_id != nullptr) *out_media_id = id;
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception&) {
        return NUKA_RESULT_INVALID_ARG;
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_scene_add_mpm_fill(nuka_scene_handle scene, uint32_t media_id,
                                      const nuka_mpm_fill_desc_t* desc) {
    return nuka_scene_add_mpm_fill_ex(scene, media_id, desc, nullptr);
}

nuka_result_t nuka_scene_add_mpm_fill_ex(nuka_scene_handle scene, uint32_t media_id,
                                       const nuka_mpm_fill_desc_t* desc,
                                       const nuka_mpm_plasticity_desc_t* plasticity) {
    if (desc == nullptr) return NUKA_RESULT_INVALID_ARG;
    SceneRecord* sr = SceneTable().Get(scene);
    if (sr == nullptr || !sr->scene) return NUKA_RESULT_NULL_HANDLE;
    if (media_id >= sr->scene->MediaCount()) return NUKA_RESULT_INVALID_ARG;
    try {
        nscene::MediaRecord& rec = sr->scene->GetMediaMut(media_id);
        // Fills append only to a box MLS-MPM medium (fluid / granular).
        const bool box_mpm =
            rec.method == nscene::MediaRecord::Method::MlsMpm &&
            (rec.kind == nscene::MediaRecord::Kind::Fluid ||
             rec.kind == nscene::MediaRecord::Kind::Granular);
        if (!box_mpm) return NUKA_RESULT_INVALID_ARG;
        nscene::MediaRecord::MpmFill fl;
        fl.box.min = nmath::Vec3{desc->min[0], desc->min[1], desc->min[2]};
        fl.box.max = nmath::Vec3{desc->max[0], desc->max[1], desc->max[2]};
        fl.box.spacing = desc->spacing;
        fl.box.position_jitter = desc->position_jitter;
        fl.material.youngs = desc->youngs;
        fl.material.poisson = desc->poisson;
        fl.material.density = desc->density;
        fl.material.dp_friction = desc->dp_friction;
        fl.material.dp_cohesion = desc->dp_cohesion;
        fl.material.model_kind = desc->model_kind;
        fl.material.bulk_modulus = desc->bulk_modulus;
        fl.material.tait_gamma = desc->tait_gamma;
        fl.material.viscosity = desc->viscosity;
        if (!ApplyMpmPlasticity(plasticity, fl.material) ||
            (plasticity != nullptr && rec.kind == nscene::MediaRecord::Kind::Granular))
            return NUKA_RESULT_INVALID_ARG;
        fl.render_material_id = desc->render_material_id;
        rec.mpm_fills.push_back(std::move(fl));
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_scene_add_cloth_region(nuka_scene_handle scene, uint32_t media_id,
                                          const nuka_cloth_region_desc_t* desc) {
    if (desc == nullptr || (desc->vertices == nullptr && desc->vertex_count != 0u))
        return NUKA_RESULT_INVALID_ARG;
    SceneRecord* sr = SceneTable().Get(scene);
    if (sr == nullptr || !sr->scene) return NUKA_RESULT_NULL_HANDLE;
    if (media_id >= sr->scene->MediaCount()) return NUKA_RESULT_INVALID_ARG;
    try {
        nscene::MediaRecord& rec = sr->scene->GetMediaMut(media_id);
        if (rec.kind != nscene::MediaRecord::Kind::Cloth) return NUKA_RESULT_INVALID_ARG;
        nscene::MediaRecord::ClothRegion region;
        region.vertices.assign(desc->vertices, desc->vertices + desc->vertex_count);
        region.material.particle_mass = desc->particle_mass;
        region.material.surface_density = desc->surface_density;
        region.material.half_thickness = desc->half_thickness;
        region.material.friction = desc->friction;
        region.material.stretch_stiffness = desc->stretch_stiffness;
        region.material.poisson = desc->poisson;
        region.material.bend_stiffness = desc->bend_stiffness;
        region.material.damping = desc->damping;
        rec.cloth_regions.push_back(std::move(region));
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_scene_add_cloth_seam(nuka_scene_handle scene, uint32_t media_id,
                                        const uint32_t* vertices, uint32_t count) {
    if (vertices == nullptr || count < 2u) return NUKA_RESULT_INVALID_ARG;
    SceneRecord* sr = SceneTable().Get(scene);
    if (sr == nullptr || !sr->scene) return NUKA_RESULT_NULL_HANDLE;
    if (media_id >= sr->scene->MediaCount()) return NUKA_RESULT_INVALID_ARG;
    try {
        nscene::MediaRecord& rec = sr->scene->GetMediaMut(media_id);
        if (rec.kind != nscene::MediaRecord::Kind::Cloth || rec.method != nscene::MediaRecord::Method::Field)
            return NUKA_RESULT_INVALID_ARG;
        std::vector<nscene::MediaRecord> seamed{rec};
        seamed[0].cloth_mesh.seams.emplace_back(vertices, vertices + count);
        cook::ValidateMedia(seamed);
        rec.cloth_mesh.seams = std::move(seamed[0].cloth_mesh.seams);
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_scene_add_cloth_attachment(nuka_scene_handle scene, uint32_t media_id,
                                              uint32_t vertex, float stiffness) {
    if (!(stiffness >= 0.0f) || !std::isfinite(stiffness)) return NUKA_RESULT_INVALID_ARG;
    SceneRecord* sr = SceneTable().Get(scene);
    if (sr == nullptr || !sr->scene) return NUKA_RESULT_NULL_HANDLE;
    if (media_id >= sr->scene->MediaCount()) return NUKA_RESULT_INVALID_ARG;
    try {
        auto& rec = sr->scene->GetMediaMut(media_id);
        if (rec.kind != nscene::MediaRecord::Kind::Cloth) return NUKA_RESULT_INVALID_ARG;
        rec.cloth_mesh.attachments.push_back({vertex, stiffness});
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_create_from_built_scene(
    nuka_device_handle device, const nuka_world_desc_t* desc,
    nuka_scene_handle scene, const nuka_built_scene_options_t* options,
    nuka_world_handle* out) {
    if (out == nullptr) return NUKA_RESULT_INVALID_ARG;
    *out = nullptr;
    if (desc == nullptr) return NUKA_RESULT_INVALID_ARG;
    try {
        SceneRecord* sr = SceneTable().Get(scene);
        if (sr == nullptr || !sr->scene) return NUKA_RESULT_NULL_HANDLE;

        // Validate + resolve device/control mode. scene_path is NOT required (the
        // scene comes from the handle, not desc->scene_path -- which is ignored).
        DeviceRecord* device_record = nullptr;
        articulation::ControlMode control_mode =
            articulation::ControlMode::PDPosition;
        const nuka_result_t valid = nuka::c_abi::ValidateWorldDescAndDevice(
            device, desc, /*require_scene_path=*/false, &device_record,
            &control_mode);
        if (valid != NUKA_RESULT_OK) return valid;

        // The handle keeps ownership; the world cooks a copy, its media before the terrain.
        nscene::SceneIR built = *sr->scene;
        nuka::c_abi::WorldRecipe recipe = nuka::c_abi::MakeWorldRecipe(*desc);
        recipe.media_before_terrain = true;
        if (options != nullptr) {
            // Opt-in per-link visual-mesh SDF bake (a foot then engages the MPM grid BC).
            recipe.bake_link_sdf = options->bake_link_sdf != 0u;
            recipe.baumgarte_max_velocity = options->baumgarte_max_velocity;
            recipe.overlap_recovery_velocity = options->overlap_recovery_velocity;
            recipe.solver_vel_iters = options->solver_vel_iters;
            recipe.solver_pos_iters = options->solver_pos_iters;
            recipe.solver_contact_margin = options->solver_contact_margin;
            recipe.solver_max_pairs = options->solver_max_pairs;
            recipe.cloth_integrator = options->cloth_integrator;
        }
        return nuka::c_abi::CreateWorld(device_record, std::move(recipe), std::move(built),
                                        std::string(), out);
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

}  // extern "C"
