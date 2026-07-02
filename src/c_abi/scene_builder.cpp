// ---------------------------------------------------------------------------
// nuka::c_abi -- the BUILT-SCENE authoring surface (program a SceneIR of rigid
// primitives + media records, then cook it through the SAME CookSceneToModel a
// file scene uses). This is the GENERAL media authoring substrate: a tet-soft
// body / a fluid box / a second cloth are added as TAGGED MediaRecords via
// nuka_scene_add_media, never by accreting flat slots onto the coupled desc.
//
// It REUSES the existing scene handle (SceneRecord / SceneTable, shared with the
// load/edit surface in scene.cpp) and the shared world-create helpers
// (ValidateWorldDescAndDevice / ApplyControlTerrainGravity / FinishWorldCreate in
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
#include "scene/terrain/heightfield.hpp"

#include <exception>
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

// Diagonal solid-body inertia from a primitive's dims + mass (a MOVABLE body then
// responds to torque; a cylinder approximates the capsule; a static body ignores it).
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
            const float r = dims[0], hh = dims[1];  // cylinder about local Z.
            const float axial = 0.5f * mass * r * r;
            const float trans = mass * (3.0f * r * r + 4.0f * hh * hh) / 12.0f;
            return nmath::Vec3{trans, trans, axial};
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
        default: return false;
    }
    switch (d.method) {
        case NUKA_MEDIA_METHOD_XPBD:   m.method = Method::Xpbd; break;
        case NUKA_MEDIA_METHOD_PBF:    m.method = Method::Pbf; break;
        case NUKA_MEDIA_METHOD_MLSMPM: m.method = Method::MlsMpm; break;
        default: return false;
    }
    // Geometry (one block read per kind).
    m.cloth_grid.nx = d.cloth_nx;
    m.cloth_grid.ny = d.cloth_ny;
    m.cloth_grid.spacing = d.cloth_spacing;
    m.cloth_grid.origin = nmath::Vec3{d.cloth_origin[0], d.cloth_origin[1],
                                      d.cloth_origin[2]};
    m.cloth_grid.free = (d.cloth_free != 0u);
    m.tet_sphere.center = nmath::Vec3{d.tet_center[0], d.tet_center[1],
                                      d.tet_center[2]};
    m.tet_sphere.radius = d.tet_radius;
    m.tet_sphere.cells = d.tet_cells;
    m.tet_sphere.cell_len = d.tet_cell_len;
    m.fluid_box.min = nmath::Vec3{d.fluid_min[0], d.fluid_min[1], d.fluid_min[2]};
    m.fluid_box.max = nmath::Vec3{d.fluid_max[0], d.fluid_max[1], d.fluid_max[2]};
    m.fluid_box.spacing = d.fluid_spacing;
    // Constitutive material (one block read per method).
    m.xpbd.particle_mass = d.xpbd_particle_mass;
    m.xpbd.friction = d.xpbd_friction;
    m.xpbd.distance_alpha = d.xpbd_distance_alpha;
    m.xpbd.bend_alpha = d.xpbd_bend_alpha;
    m.xpbd.volume_alpha = d.xpbd_volume_alpha;
    m.xpbd.iters = static_cast<uint16_t>(d.xpbd_iters);
    m.xpbd.aero_drag_normal = d.xpbd_aero_drag_normal;
    m.xpbd.aero_drag_tangent = d.xpbd_aero_drag_tangent;
    m.xpbd.aero_drag_max_dv = d.xpbd_aero_drag_max_dv;
    m.pbf.rest_density = d.pbf_rest_density;
    m.pbf.support_scale = d.pbf_support_scale;
    m.pbf.iters = static_cast<uint16_t>(d.pbf_iters);
    m.pbf.friction = d.pbf_friction;
    m.pbf.clamp_overdensity = (d.pbf_clamp_overdensity != 0u);
    m.pbf.walls_enabled = (d.pbf_walls_enabled != 0u);
    m.pbf.walls_min = nmath::Vec3{d.pbf_walls_min[0], d.pbf_walls_min[1],
                                  d.pbf_walls_min[2]};
    m.pbf.walls_max = nmath::Vec3{d.pbf_walls_max[0], d.pbf_walls_max[1],
                                  d.pbf_walls_max[2]};
    m.pbf.floor_z = d.pbf_floor_z;
    m.pbf.boundary_layers = d.pbf_boundary_layers;
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
    // Render skin.
    m.render_skin.normal_offset = d.skin_normal_offset;
    m.render_skin.smooth_iters = d.skin_smooth_iters;
    m.render_skin.smooth_lambda = d.skin_smooth_lambda;
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
            default:
                return NUKA_RESULT_INVALID_ARG;
        }
        // A box/sphere/capsule needs positive extents (zero/negative cooks degenerate
        // geometry); a plane carries no extent.
        if (type == nscene::ShapeType::Sphere && desc->dims[0] <= 0.0f) {
            return NUKA_RESULT_INVALID_ARG;
        }
        if (type == nscene::ShapeType::Capsule &&
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
        } else if (type == nscene::ShapeType::Capsule) {
            shape.radius = desc->dims[0];
            shape.half_height = desc->dims[1];
        }
        // friction < 0 keeps the inherit-material sentinel; >= 0 is an explicit
        // per-shape override (the SAME precedence the cooker resolves).
        if (desc->friction >= 0.0f) shape.friction_mu = desc->friction;
        sr->scene->AddCollisionShape(std::move(shape));
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_scene_add_media(nuka_scene_handle scene,
                                   const nuka_media_desc_t* desc) {
    if (desc == nullptr) return NUKA_RESULT_INVALID_ARG;
    SceneRecord* sr = SceneTable().Get(scene);
    if (sr == nullptr || !sr->scene) return NUKA_RESULT_NULL_HANDLE;
    try {
        nscene::MediaRecord media;
        if (!MediaFromDesc(*desc, &media)) {
            return NUKA_RESULT_INVALID_ARG;  // unknown kind / method.
        }
        // Reject an illegal (kind x method) immediately via the SAME cook validator
        // (a single-record list; the cross-medium MPM+XPBD/PBF mix is caught at cook).
        cook::ValidateMedia(std::vector<nscene::MediaRecord>{media});
        media.name = "media_" + std::to_string(sr->scene->MediaCount());
        sr->scene->AddMedia(std::move(media));
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

        // ONE cook: rigid + media through the SAME orchestrator a file scene uses. A
        // media-free / primitive-free built scene cooks byte-identically to
        // CookToModel (CookSceneMedia is a no-op on empty media). The handle keeps
        // ownership; the cook consumes a copy.
        nscene::SceneIR built = *sr->scene;
        cook::CookToModelResult cooked = cook::CookSceneToModel(
            built, static_cast<int>(desc->env_count), cook::CookToModelOptions{});

        // The SHARED post-cook step (drive_mode + optional heightfield + gravity).
        nuka::terrain::HeightField cooked_terrain;
        nmath::Vec3 gravity{0.0f, 0.0f, 0.0f};
        const nuka_result_t terrain = nuka::c_abi::ApplyControlTerrainGravity(
            desc, control_mode, cooked.model, &cooked_terrain, &gravity);
        if (terrain != NUKA_RESULT_OK) return terrain;

        // A positive baumgarte_max_velocity bounds the contact recovery push-out (the
        // SAME model override the coupled entry applies); 0 keeps the cooked default.
        if (options != nullptr && options->baumgarte_max_velocity > 0.0f) {
            cooked.model.baumgarte_max_velocity = options->baumgarte_max_velocity;
        }

        // Every cooked medium's render surface (built from the SAME media list before
        // the scene moves into the record), so a beauty render rebuilds each surface.
        std::vector<cook::MediaRenderSurface> media_surfaces =
            cook::BuildSceneMediaRenderSurfaces(built.Media());

        // Build + insert the live world through the SAME record-assembly path. The
        // SolverConfig overrides ride `options`; all-zero/NULL keeps today's cfg.
        const nuka_result_t result = nuka::c_abi::FinishWorldCreate(
            std::move(cooked.model), std::move(built), std::move(cooked_terrain),
            device_record, desc->fixed_dt, desc->env_count, control_mode, gravity,
            out, options ? options->solver_vel_iters : 0u,
            options ? options->solver_pos_iters : 0u,
            options ? options->solver_contact_margin : 0.0f,
            options ? options->solver_max_pairs : 0u);
        if (result == NUKA_RESULT_OK) {
            if (auto* record = nuka::c_abi::WorldTable().Get(*out); record != nullptr)
                record->particle_surfaces = std::move(media_surfaces);
        }
        return result;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

}  // extern "C"
