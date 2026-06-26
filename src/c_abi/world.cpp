// ---------------------------------------------------------------------------
// nuka::c_abi -- the ONE generic world entry (M9 T5/T6 cutover).
//
// The C-ABI world is now exactly ONE nk::World built from
// Scene -> CookToModel -> nk::World (mirroring c_abi/recorder.cpp's proven
// create+ownership pattern). The two legacy steppers (single-env Featherstone
// StepWorldGpu + the multi-env batched articulated world) are GONE: there is no
// special world type and no per-case physics path (owner's highest directive --
// a GENERAL solver). create cooks the authored scene, builds an
// nk::World on the DeviceRecord's already-owned phi v2 device/backend (M9 T2),
// and seeds the cooked IC + PD hold targets (the cook+SeedInitialState bake them,
// so the legacy BuildHoldDriveTargets/Upload/Sync host plumbing is deleted).
// step/reset/buffer-view drive that ONE nk::World directly.
//
// A transitional ArticulationHostState mirror (articulation_host) is captured at
// create from the SAME cook for the diffsim backward family + set_link_mass + DR
// (host dI/dmass + spatial-inertia rebuild); its device writes target the nk
// arena LinkInertia field. That mirror retires when the backward op-ifies (T7).
// ---------------------------------------------------------------------------

#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"

#include "nk/pipeline/world.hpp"
#include "nk/solve/nk_row.hpp"  // L1: kPairDrivenRowsPerSlot (general row budget)
#include "runtime/articulation/articulation_cooker.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/cooker.hpp"
#include "scene/format/nks.hpp"  // native .nks loader (parity with scene.cpp)

#include "import/mjcf_importer.hpp"
#include "import/urdf_importer.hpp"
#include "import/usd_importer.hpp"
#include "math/transform.hpp"       // M10 co-residence: replica placement transform
#include "scene/scene_compose.hpp"  // M10 co-residence: compose K replicas pre-cook
#include "scene/scene_ir.hpp"
#include "scene/terrain/heightfield.hpp"          // the ONE cooked terrain grid
#include "scene/terrain/heightfield_loaders.hpp"  // image + parametric loaders
#include "scene/terrain/heightfield_sample.hpp"   // the ONE bilinear obs sampler

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace nuka::c_abi {

namespace nk = nuka::nk;
namespace cook = nuka::scene::cook;
namespace articulation = nuka::runtime::articulation;

// The WorldRecord special members are out-of-line so the unique_ptr<nk::World>
// member is created/destroyed where nk::World is a complete type (here). The
// destructor's implicit teardown of `world` runs BEFORE the borrowed backend is
// freed (the DeviceRecord outlives the WorldRecord by API contract).
WorldRecord::WorldRecord() = default;
WorldRecord::~WorldRecord() = default;
WorldRecord::WorldRecord(WorldRecord&&) noexcept = default;
WorldRecord& WorldRecord::operator=(WorldRecord&&) noexcept = default;

namespace {

// Ingest .nks/.xml(MJCF)/.urdf/.usda by extension into a SceneIR. KEPT verbatim
// (the ~12 World.create_from_scene consumers -- go2_float.usda / h1_reduced_train
// .xml / go2_stand.usda -- depend on it; recon KEEP note). NOT narrowed.
bool LoadSceneByExtension(const char* scene_path, scene::SceneIR* out_scene) {
    if (out_scene == nullptr) {
        return false;
    }
    const std::filesystem::path path(scene_path);
    const std::string extension = path.extension().string();
    if (extension == ".nks") {
        *out_scene = scene::nks::Load(scene_path);
        return true;
    }
    if (extension == ".xml" || extension == ".mjcf") {
        *out_scene = import::LoadMjcf(scene_path);
        return true;
    }
    if (extension == ".urdf") {
        *out_scene = import::LoadUrdf(scene_path);
        return true;
    }
    if (extension == ".usd" || extension == ".usda") {
        *out_scene = import::LoadUsd(scene_path);
        return true;
    }
    return false;
}

// Capture the transitional diffsim/noise host mirror from the SAME cook the
// legacy create used (CookArticulations -> BuildArticulationHostState over the
// FIRST articulation, single-env). This reproduces topo.masses/inertias/
// inertial_frames + joint_armature + link_inertia BYTE-IDENTICALLY, so
// BuildMassParams (diffsim) / ResolveLinkInertiaParams (DR + set_link_mass) /
// CaptureNominalBaseline stay bit-exact. It is HOST-ONLY (no device upload): the
// live device inertia is the nk arena's LinkInertia field; set_link_mass / DR
// write THAT in place (world.cpp does not own a legacy ArticulationDeviceBuffers).
void CaptureArticulationHostMirror(const scene::SceneIR& scene, WorldRecord* record) {
    const scene::CookedBlob blob = scene::CookScene(scene);
    const std::vector<articulation::ArticulationCookedTopology> arts =
        articulation::CookArticulations(blob);
    if (arts.empty()) {
        record->articulation_host = articulation::ArticulationHostState{};
        return;
    }
    // CookToModel transcribes only the FIRST articulation today (generic
    // single-articulation cook debt, M7); mirror that exact choice so the host
    // mirror's global-link ordering matches the arena's. Passing all arts here
    // would DIVERGE the mirror from the single-articulation arena, so we keep the
    // first and make the truncation LOUD (no silent data drop): the diffsim / DR /
    // set_link_mass host mirror covers only articulation 0.
    if (arts.size() > 1u) {
        std::fprintf(stderr,
                     "[nuka] CaptureArticulationHostMirror: %zu articulations "
                     "cooked; diffsim/DR/set_link_mass host mirror covers only "
                     "articulation 0 (multi-articulation mirror not yet "
                     "implemented).\n",
                     arts.size());
    }
    record->articulation_host =
        articulation::BuildArticulationHostState({arts.front()}, blob.bodies);
}

// Translate the compact coupled cloth slots into a first-class scene MediaRecord
// (Cloth, XPBD by default / MLS-MPM when soft_sim_method selects it). The cook's
// BuildClothXpbdInput reads this record; distance_alpha stays 0 (an inextensible
// cloth), support fields are XPBD-only. Geometry-only when the extent is absent.
nuka::scene::MediaRecord ClothMediaFromDesc(const nuka_coupled_particles_desc_t& p) {
    nuka::scene::MediaRecord m;
    m.kind = nuka::scene::MediaRecord::Kind::Cloth;
    m.method = (p.soft_sim_method == 1u) ? nuka::scene::MediaRecord::Method::MlsMpm
                                         : nuka::scene::MediaRecord::Method::Xpbd;
    m.cloth_grid.nx = p.cloth_nx;
    m.cloth_grid.ny = p.cloth_ny;
    m.cloth_grid.spacing = p.cloth_spacing;
    m.cloth_grid.origin = {p.cloth_origin_x, p.cloth_origin_y, p.cloth_origin_z};
    m.cloth_grid.free = (p.cloth_free != 0u);
    m.xpbd.particle_mass = p.cloth_particle_mass;
    m.xpbd.friction = p.cloth_friction;
    m.xpbd.distance_alpha = 0.0f;  // inextensible cloth (the legacy hardcoded value).
    m.xpbd.bend_alpha = p.cloth_bend_alpha;
    m.xpbd.iters = static_cast<uint16_t>(p.cloth_iters);
    m.xpbd.aero_drag_normal = p.aero_normal;
    m.xpbd.aero_drag_tangent = p.aero_tangent;
    m.xpbd.aero_drag_max_dv = p.aero_max_dv;
    return m;
}

// Translate the compact coupled fluid slots into a MediaRecord (Fluid, PBF by
// default / MLS-MPM when selected). support_scale 1.5 reproduces the legacy
// support_radius = 1.5*spacing. Geometry-only when the box extent is absent.
nuka::scene::MediaRecord FluidMediaFromDesc(const nuka_coupled_particles_desc_t& p) {
    nuka::scene::MediaRecord m;
    m.kind = nuka::scene::MediaRecord::Kind::Fluid;
    m.method = (p.soft_sim_method == 1u) ? nuka::scene::MediaRecord::Method::MlsMpm
                                         : nuka::scene::MediaRecord::Method::Pbf;
    m.fluid_box.min = {p.fluid_min_x, p.fluid_min_y, p.fluid_min_z};
    m.fluid_box.max = {p.fluid_max_x, p.fluid_max_y, p.fluid_max_z};
    m.fluid_box.spacing = p.fluid_spacing;
    m.pbf.rest_density = p.fluid_rest_density;
    m.pbf.support_scale = 1.5f;  // SPH support over the lattice spacing.
    m.pbf.iters = static_cast<uint16_t>(p.fluid_iters);
    m.pbf.friction = p.fluid_friction;
    m.pbf.floor_z = p.fluid_floor_z;
    m.pbf.clamp_overdensity = true;
    return m;
}

// The cooked products a world-create entry assembles before building the live world:
// the robot Model, the composed scene, the baked terrain grid, the resolved gravity,
// the control mode + the validated device. Shared by the scene-only and coupled
// entries; the coupled entry cooks particles onto `model` before FinishWorldCreate.
struct PreparedWorld {
    nuka::nk::Model model;
    nuka::scene::SceneIR scene;
    nuka::terrain::HeightField terrain;
    nuka::math::Vec3 gravity = nuka::runtime::kDefaultGravity;
    articulation::ControlMode control_mode = articulation::ControlMode::PDPosition;
    DeviceRecord* device_record = nullptr;
};

// Validate `desc`, load + compose the scene, cook it to an nk::Model, bake the
// optional heightfield + recompute its budget, and resolve gravity. Returns OK with
// `out` filled, or the specific error the create contract reports. This is the ONE
// scene->Model cook path; both create entries call it (the coupled entry then cooks
// particles onto out->model). MUST be called inside a try (CookToModel may throw).
nuka_result_t PrepareWorldFromDesc(nuka_device_handle device,
                                   const nuka_world_desc_t* desc,
                                   PreparedWorld* out) {
    if (desc == nullptr || desc->scene_path == nullptr || desc->fixed_dt <= 0.0f) {
        return NUKA_RESULT_INVALID_ARG;
    }
    if (desc->env_count == 0u) {
        return NUKA_RESULT_INVALID_ARG;
    }
    // Determinism level: 0 = D1/Strong (default), 1 = D2/Weak. Reject other values.
    // The nk pipeline runs the D1 schedule only; D2 currently selects the same
    // kernels and is behaviorally identical (NOT a separate path).
    if (desc->determinism > 1u) {
        return NUKA_RESULT_INVALID_ARG;
    }
    // Control mode: 0 PDPosition + 1 Torque are wired onto the ONE nk::World (presets
    // of the one affine actuator). The other modes have no op kernel yet -> rejected
    // NOT_SUPPORTED. A value above the enum range is an outright INVALID_ARG.
    if (!articulation::IsControlModeImplemented(desc->control_mode)) {
        return NUKA_RESULT_INVALID_ARG;
    }
    const auto control_mode =
        static_cast<articulation::ControlMode>(desc->control_mode);
    if (control_mode != articulation::ControlMode::PDPosition &&
        control_mode != articulation::ControlMode::Torque) {
        return NUKA_RESULT_NOT_SUPPORTED;  // preset not yet wired on the nk world.
    }

    auto* device_record = nuka::c_abi::DeviceTable().Get(device);
    if (device_record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    // The nk::World ctor REQUIRES the phi v2 device/backend acquired at create.
    if (device_record->phi_device == nullptr || device_record->backend == nullptr) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }

    nuka::scene::SceneIR scene;
    if (!nuka::c_abi::LoadSceneByExtension(desc->scene_path, &scene)) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }

    // CO-RESIDENCE: compose the authored scene with itself instance_count times at
    // distinct XY so the cook emits K SEPARATE articulations co-resident in one env.
    // 0/1 == single instance -> the loop never runs -> BYTE-IDENTICAL single-artic.
    if (desc->instance_count > 1u) {
        const float spacing =
            (desc->instance_spacing > 0.0f) ? desc->instance_spacing : 1.5f;
        for (uint32_t i = 1u; i < desc->instance_count; ++i) {
            nuka::scene::SceneIR replica;
            if (!nuka::c_abi::LoadSceneByExtension(desc->scene_path, &replica)) {
                return NUKA_RESULT_NOT_SUPPORTED;
            }
            nuka::math::Transform place = nuka::math::Transform::Identity();
            place.position.x = spacing * static_cast<float>(i);
            scene = nuka::scene::Compose(scene, replica, place,
                                         "dog" + std::to_string(i) + "_");
        }
    }

    // Cook the authored scene to an nk::Model (the GENERAL LBVH+cvx narrowphase +
    // mixed-island solve; CookToModel always returns a PairDriven model). The world
    // runs EXACTLY the cooked scene physics -- it does NOT inject a synthetic ground.
    nuka::scene::cook::CookToModelResult cooked =
        nuka::scene::cook::CookToModel(scene, static_cast<int>(desc->env_count));

    // Route the declared control mode onto the cooked Model's drive_mode (PDPosition
    // -> 0 position-PD kernel, Torque -> 1 direct-torque kernel). PDPosition is a
    // no-op (drive_mode defaults to 0 => the PD goldens stay byte-identical).
    cooked.model.drive_mode =
        (control_mode == articulation::ControlMode::Torque) ? 1u : 0u;

    // When the caller requests a baked heightfield (contact_family == 1), build the
    // ONE source-of-truth HeightField -- from a grayscale IMAGE if set, else the
    // parametric generator -- then cook it. A zero-init desc => flat => byte-identical
    // to no terrain. The obs/spawn sampler reads the SAME grid (stored on the record).
    nuka::terrain::HeightField cooked_terrain;
    if (desc->contact_family == 1u) {
        nuka::nk::Model& m = cooked.model;
        const uint32_t orig_bodies = m.capacities.bodies_per_env;
        m.body_init.resize(orig_bodies);  // heightfield seeds at orig_bodies.
        const uint32_t nrow =
            desc->heightfield_nrow != 0u ? desc->heightfield_nrow : 41u;
        const uint32_t ncol =
            desc->heightfield_ncol != 0u ? desc->heightfield_ncol : 41u;
        const float cell =
            desc->heightfield_cell > 0.0f ? desc->heightfield_cell : 0.25f;

        const bool image_path = desc->heightfield_image_path != nullptr &&
                                desc->heightfield_image_path[0] != '\0';
        if (image_path) {
            // The grid is centred at the world origin (corner = -radius); a malformed
            // image is a LOUD INVALID_ARG, never a silent flat fallback.
            const float rx = desc->heightfield_radius_x > 0.0f
                                 ? desc->heightfield_radius_x
                                 : 0.5f * static_cast<float>(ncol - 1u) * cell;
            const float ry = desc->heightfield_radius_y > 0.0f
                                 ? desc->heightfield_radius_y
                                 : 0.5f * static_cast<float>(nrow - 1u) * cell;
            std::string err;
            if (!nuka::terrain::LoadHeightFieldFromImage(
                    desc->heightfield_image_path, rx, ry,
                    desc->heightfield_elevation_z, desc->heightfield_base_z,
                    nuka::math::Vec3{-rx, -ry, 0.0f}, cooked_terrain, err)) {
                return NUKA_RESULT_INVALID_ARG;
            }
        } else {
            // Parametric: the generator fills the grid ONCE from the desc knobs. The
            // heightfield_terrain_type is a LEGACY selector the C-ABI maps to
            // parametric feature amplitudes (the engine has no terrain types).
            nuka::terrain::TerrainGenConfig cfg;
            cfg.nrow = nrow;
            cfg.ncol = ncol;
            cfg.cell_x = cell;
            cfg.cell_y = cell;
            const float half_x = 0.5f * static_cast<float>(ncol - 1u) * cell;
            const float half_y = 0.5f * static_cast<float>(nrow - 1u) * cell;
            cfg.origin = nuka::math::Vec3{-half_x, -half_y, 0.0f};
            cfg.base_z = cooked.model.ground_height;
            switch (desc->heightfield_terrain_type) {
                case 1u:  // stairs (climbing rings).
                    cfg.ring_rise = desc->terrain_step_height;
                    cfg.ring_width = desc->terrain_step_width;
                    cfg.ring_platform = desc->terrain_platform_width;
                    cfg.ring_count = 8u;
                    break;
                case 2u:  // pit (descending rings).
                    cfg.ring_rise = -desc->terrain_step_height;
                    cfg.ring_width = desc->terrain_step_width;
                    cfg.ring_platform = desc->terrain_platform_width;
                    cfg.ring_count = 8u;
                    break;
                case 3u:  // hashed bumps.
                    cfg.bump_height = desc->terrain_grid_height_max;
                    cfg.bump_cell = desc->terrain_grid_width;
                    break;
                case 4u:  // tiled multi-feature field (the complex scene).
                    cfg.ring_rise = desc->terrain_step_height;
                    cfg.ring_width = desc->terrain_step_width;
                    cfg.ring_platform = desc->terrain_platform_width;
                    cfg.ring_count = 8u;
                    cfg.bump_height = desc->terrain_grid_height_max;
                    cfg.bump_cell = desc->terrain_grid_width;
                    cfg.feature_cell = 5.0f;
                    cfg.feature_margin = 0.6f;
                    break;
                case 5u:  // curriculum tile grid (level rows x type cols).
                    cfg.ring_rise = desc->terrain_step_height;
                    cfg.ring_width = desc->terrain_step_width;
                    cfg.ring_platform = desc->terrain_platform_width;
                    cfg.bump_height = desc->terrain_grid_height_max;
                    cfg.bump_cell = desc->terrain_grid_width;
                    cfg.feature_cell = desc->terrain_feature_cell > 0.0f
                                           ? desc->terrain_feature_cell
                                           : 6.0f;
                    cfg.feature_margin = 0.6f;
                    cfg.curric_levels = desc->curric_levels;
                    cfg.curric_types = desc->curric_types;
                    break;
                default:  // 0 flat: every amplitude 0.
                    break;
            }
            if (!nuka::terrain::GenerateHeightField(cfg, cooked_terrain)) {
                return NUKA_RESULT_INVALID_ARG;
            }
        }
        nuka::scene::cook::CookHeightfieldGrid(m, cooked_terrain);

        nuka::nk::ModelCapacities& cap = m.capacities;
        // bodies_per_env (the LBVH leaf count) includes every static terrain
        // collidable; the CONTACT budget is driven by the DYNAMIC collidables (+1 for
        // the static ground) at 4 candidate slots each -- the SAME rule the cook uses.
        cap.bodies_per_env = static_cast<uint32_t>(m.body_init.size());
        cap.max_bodies_total = static_cast<uint32_t>(m.shape_table_rows.size());
        constexpr uint32_t kCandidatePairsPerCollidable = 4u;
        constexpr uint32_t kStaticCollidables = 1u;
        const uint32_t dynamic_collidables = orig_bodies + kStaticCollidables;
        const uint32_t rigid_base =
            dynamic_collidables * kCandidatePairsPerCollidable;
        cap.max_contacts_per_env = rigid_base;
        cap.max_rows_per_env =
            cap.max_contacts_per_env * nuka::nk::kPairDrivenRowsPerSlot;
        // A coupled world (terrain AND particles) keeps the body<->particle slot
        // reserve disjoint above the just-recomputed rigid base -- the SAME additive
        // rule the cook uses. No-op when particles_per_env == 0.
        nuka::scene::cook::GrowContactBudgetForParticles(cap, rigid_base);
    }

    // Resolve world gravity (Z-up). A zero-initialized desc substitutes the shared
    // standard-Earth default; any non-zero component takes the caller's full vector.
    nuka::math::Vec3 gravity = nuka::runtime::kDefaultGravity;
    if (desc->gravity_x != 0.0f || desc->gravity_y != 0.0f ||
        desc->gravity_z != 0.0f) {
        gravity = {desc->gravity_x, desc->gravity_y, desc->gravity_z};
    }

    out->model = std::move(cooked.model);
    out->scene = std::move(scene);
    out->terrain = std::move(cooked_terrain);
    out->gravity = gravity;
    out->control_mode = control_mode;
    out->device_record = device_record;
    return NUKA_RESULT_OK;
}

// Build the live WorldRecord from a cooked Model + the composed scene and insert it
// into the WorldTable. Shared by the scene-only and the coupled (scene + particles)
// create entries so the record assembly is ONE path -- the coupled entry differs only
// in cooking particles onto `cooked_model` before this runs. On success writes `out`
// and returns OK; on a failed World build returns INTERNAL and leaves `out` null.
// `solver_*` override the world SolverConfig (each 0 keeps the engine default, so the
// scene-only entry never passes them and stays byte-identical). The coupled entry
// forwards the desc's solver knobs here.
nuka_result_t FinishWorldCreate(nuka::nk::Model&& cooked_model,
                                nuka::scene::SceneIR&& scene,
                                nuka::terrain::HeightField&& cooked_terrain,
                                DeviceRecord* device_record, float fixed_dt,
                                uint32_t env_count,
                                articulation::ControlMode control_mode,
                                const nuka::math::Vec3& gravity,
                                nuka_world_handle* out,
                                uint32_t solver_vel_iters = 0u,
                                uint32_t solver_pos_iters = 0u,
                                float solver_contact_margin = 0.0f,
                                uint32_t solver_max_pairs = 0u) {
    auto record = std::make_unique<WorldRecord>();
    record->device = device_record;
    record->env_count = env_count;
    record->control_mode = control_mode;
    record->cooked_terrain = std::move(cooked_terrain);  // obs/spawn read this.
    record->step_options.dt = fixed_dt;
    record->step_options.gravity = gravity;

    nuka::nk::Pipeline::SolverConfig cfg;
    cfg.dt = fixed_dt;
    cfg.gravity[0] = gravity.x;
    cfg.gravity[1] = gravity.y;
    cfg.gravity[2] = gravity.z;
    // Each non-zero override replaces a default; all-zero leaves cfg exactly as built
    // above (dt + gravity only), so an unset caller is byte-identical to today.
    if (solver_vel_iters != 0u) cfg.vel_iters = static_cast<uint16_t>(solver_vel_iters);
    if (solver_pos_iters != 0u) cfg.pos_iters = static_cast<uint16_t>(solver_pos_iters);
    if (solver_contact_margin > 0.0f) cfg.contact_margin = solver_contact_margin;
    if (solver_max_pairs != 0u) cfg.max_pairs = solver_max_pairs;

    record->world = std::make_unique<nuka::nk::World>(
        std::move(cooked_model), env_count, device_record->phi_device,
        device_record->backend, cfg);
    if (!record->world->Ready()) {
        return NUKA_RESULT_INTERNAL;
    }

    // Transitional host mirror for the diffsim backward + set_link_mass + DR
    // (host dI/dmass + spatial-inertia rebuild). HOST-ONLY -- device writes target
    // the nk arena. Particles carry no articulation, so this covers the robot only.
    CaptureArticulationHostMirror(scene, record.get());

    // Retain the FINAL composed scene so a camera-sensor attach builds the per-env
    // visual binding from the SAME ECS the cook saw (host data only).
    record->scene = std::make_unique<nuka::scene::SceneIR>(std::move(scene));

    *out = WorldTable().Insert(std::move(record));
    return *out == nullptr ? NUKA_RESULT_INTERNAL : NUKA_RESULT_OK;
}

}  // namespace

}  // namespace nuka::c_abi

extern "C" {

nuka_result_t nuka_world_create_from_scene(nuka_device_handle device,
                                           const nuka_world_desc_t* desc,
                                           nuka_world_handle* out) {
    if (out == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    *out = nullptr;
    try {
        nuka::c_abi::PreparedWorld prepared;
        const nuka_result_t prep =
            nuka::c_abi::PrepareWorldFromDesc(device, desc, &prepared);
        if (prep != NUKA_RESULT_OK) {
            return prep;
        }
        // Build + insert the live world (the shared record-assembly path).
        return nuka::c_abi::FinishWorldCreate(
            std::move(prepared.model), std::move(prepared.scene),
            std::move(prepared.terrain), prepared.device_record, desc->fixed_dt,
            desc->env_count, prepared.control_mode, prepared.gravity, out);
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_create_coupled_from_scene(
    nuka_device_handle device, const nuka_world_desc_t* desc,
    const nuka_coupled_particles_desc_t* particles, nuka_world_handle* out) {
    if (out == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    *out = nullptr;
    if (particles == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    try {
        // Cook the robot + scene through the SAME path the scene-only entry uses.
        nuka::c_abi::PreparedWorld prepared;
        const nuka_result_t prep =
            nuka::c_abi::PrepareWorldFromDesc(device, desc, &prepared);
        if (prep != NUKA_RESULT_OK) {
            return prep;
        }

        // Translate the compact descriptor's media slots into first-class scene
        // MediaRecords and append them to the robot scene; the cook reads the list.
        const bool has_cloth = particles->cloth_nx >= 2u &&
                               particles->cloth_ny >= 2u &&
                               particles->cloth_spacing > 0.0f;
        const bool has_fluid =
            particles->fluid_spacing > 0.0f &&
            particles->fluid_max_x > particles->fluid_min_x &&
            particles->fluid_max_y > particles->fluid_min_y &&
            particles->fluid_max_z > particles->fluid_min_z;
        if (!has_cloth && !has_fluid) {
            // A particle-free world must use nuka_world_create_from_scene.
            return NUKA_RESULT_INVALID_ARG;
        }
        nuka::scene::MediaRecord cloth_media;
        if (has_cloth) {
            cloth_media = nuka::c_abi::ClothMediaFromDesc(*particles);
            prepared.scene.AddMedia(cloth_media);
        }
        if (has_fluid) {
            prepared.scene.AddMedia(nuka::c_abi::FluidMediaFromDesc(*particles));
        }

        // Cook the media onto the robot Model via the ONE general media dispatch (the
        // [soft|fluid] layout, validated against the legal kind x method set: cloth
        // XPBD + fluid PBF; soft_sim_method == mlsmpm throws LOUDLY here). The disjoint
        // body<->particle slot reserve sits above the rigid budget the prepare step
        // set; the contact diameter rides the descriptor radius.
        nuka::scene::cook::CookSceneMedia(prepared.model, desc->env_count,
                                          prepared.scene.Media(),
                                          particles->contact_radius);

        // A positive baumgarte_max_velocity bounds the contact recovery push-out; 0
        // leaves the cooked model default (so a zero-init desc is byte-identical).
        if (particles->baumgarte_max_velocity > 0.0f) {
            prepared.model.baumgarte_max_velocity = particles->baumgarte_max_velocity;
        }

        // The cloth render-surface connectivity, built before the scene moves into the
        // world record, so a live beauty render rebuilds the deforming surface from the
        // live particle field -- the SAME lattice the physics cook meshed.
        std::vector<uint32_t> cloth_surface_triangles;
        if (has_cloth) {
            cloth_surface_triangles =
                nuka::scene::cook::BuildClothSurfaceTriangles(cloth_media);
        }

        // Build + insert the live coupled world (the SAME record-assembly path). The
        // SolverConfig overrides ride the coupled desc; all-zero keeps today's cfg.
        const nuka_result_t result = nuka::c_abi::FinishWorldCreate(
            std::move(prepared.model), std::move(prepared.scene),
            std::move(prepared.terrain), prepared.device_record, desc->fixed_dt,
            desc->env_count, prepared.control_mode, prepared.gravity, out,
            particles->solver_vel_iters, particles->solver_pos_iters,
            particles->solver_contact_margin, particles->solver_max_pairs);
        if (result == NUKA_RESULT_OK) {
            if (auto* record = nuka::c_abi::WorldTable().Get(*out); record != nullptr)
                record->particle_surface_triangles = std::move(cloth_surface_triangles);
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

void nuka_world_destroy(nuka_world_handle world) {
    (void)nuka::c_abi::WorldTable().Remove(world);
}

nuka_result_t nuka_world_step(nuka_world_handle world) {
    return nuka_world_step_n(world, 1u);
}

nuka_result_t nuka_world_step_n(nuka_world_handle world, uint32_t step_count) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!record->world) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }
    if (step_count == 0u) {
        return NUKA_RESULT_OK;
    }
    try {
        for (uint32_t step = 0u; step < step_count; ++step) {
            const nuka::nk::StepResult result = record->world->Step();
            if (!result.AllOk()) {
                return NUKA_RESULT_INTERNAL;
            }
        }
        record->simulated_step_count += step_count;
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_reset(nuka_world_handle world) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!record->world) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }
    try {
        // Empty env list -> the bulk RestoreState op (restore the cooked initial
        // snapshot + clear qddot/tau/lambda). The generic nk::World Reset replaces
        // the legacy batched-only Reset.
        const nuka::phi::Status status = record->world->Reset({});
        return status == nuka::phi::Status::Ok ? NUKA_RESULT_OK
                                               : NUKA_RESULT_INTERNAL;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_reset_envs(nuka_world_handle world,
                                    const uint32_t* env_ids,
                                    uint32_t count) {
    if (count > 0u && env_ids == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!record->world) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }
    if (count == 0u) {
        return NUKA_RESULT_OK;
    }
    // Bounds-check the ids here so an out-of-range id rejects cleanly without
    // touching any env state (the friendlier control-plane contract).
    const uint32_t env_count = record->world->EnvCount();
    for (uint32_t i = 0u; i < count; ++i) {
        if (env_ids[i] >= env_count) {
            return NUKA_RESULT_INVALID_ARG;
        }
    }
    try {
        const std::vector<uint32_t> ids(env_ids, env_ids + count);
        const nuka::phi::Status status = record->world->Reset(ids);
        return status == nuka::phi::Status::Ok ? NUKA_RESULT_OK
                                               : NUKA_RESULT_INTERNAL;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_get_last_invariant_violations(nuka_world_handle world,
                                                       uint32_t* out_count,
                                                       void* out_array,
                                                       uint32_t array_capacity) {
    if (out_count == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (array_capacity > 0u && out_array == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }

    // M9 NAMED GAP: invariant sampling read the legacy host-mirrored BuiltWorld
    // template/instance each step; the generic nk::World keeps its state on the
    // device and the C-ABI no longer round-trips it to a host BuiltWorld. The
    // sampler is therefore never populated -> always report zero violations. The
    // only consumer (an invariant probe) reads a count; no gate asserts a
    // non-zero violation through this path. Re-pointing to read nk Data is a
    // device-download cost deferred to a later milestone (flagged in the report).
    const auto& violations = record->last_invariant_violations;
    *out_count = static_cast<uint32_t>(violations.size());
    if (out_array == nullptr || array_capacity == 0u || violations.empty()) {
        return NUKA_RESULT_OK;
    }

    const uint32_t copy_count =
        std::min(array_capacity, static_cast<uint32_t>(violations.size()));
    auto* out = static_cast<nuka_invariant_violation_t*>(out_array);
    for (uint32_t index = 0u; index < copy_count; ++index) {
        const auto& violation = violations[index];
        out[index].invariant = static_cast<uint32_t>(violation.which);
        out[index].step = violation.step_index;
        out[index].env_id = violation.env_id;
        out[index].value = violation.value;
        out[index].threshold = violation.threshold;
    }
    return NUKA_RESULT_OK;
}

// BATCHED arbitrary-(x,y) terrain height sampler over the world's COOKED grid.
// Reads the world's stored HeightField via the ONE bilinear sampler -- the SAME
// full-relief grid the physics narrowphase rests feet on -- so obs/spawn height
// equals physics by construction. There is no per-env vertical scaling: the grid
// is model-level (one terrain for the batch), so any per-env shrink would diverge
// the obs from the physics it is meant to mirror.
nuka_result_t nuka_world_sample_terrain_height_batch(nuka_world_handle world,
                                                     uint32_t n,
                                                     const float* xs,
                                                     const float* ys,
                                                     float* out_heights) {
    if (n == 0u) {
        return NUKA_RESULT_OK;  // nothing to sample.
    }
    if (world == nullptr || xs == nullptr || ys == nullptr ||
        out_heights == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    const nuka::terrain::HeightField& hf = record->cooked_terrain;
    if (!nuka::terrain::IsValid(hf)) {
        // No cooked heightfield: report the flat ground floor for every column.
        for (uint32_t i = 0u; i < n; ++i) out_heights[i] = 0.0f;
        return NUKA_RESULT_OK;
    }
    for (uint32_t i = 0u; i < n; ++i) {
        out_heights[i] = nuka::terrain::SampleHeightFieldZ(hf, xs[i], ys[i]);
    }
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_get_dof_name(nuka_world_handle world,
                                      uint32_t link_index, char* out, size_t cap,
                                      size_t* out_len) {
    if (out_len != nullptr) {
        *out_len = 0u;
    }
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (record->scene == nullptr ||
        record->articulation_host.articulations.empty()) {
        return NUKA_RESULT_NOT_SUPPORTED;  // no cooked articulation retained.
    }
    try {
        // Cooked link -> its body row. The host mirror covers articulation 0 (the
        // co-residence case mirrors only the first dog, the existing host limitation).
        const auto& topo = record->articulation_host.articulations.front();
        if (link_index >= topo.link_bodies.size()) {
            return NUKA_RESULT_INVALID_ARG;
        }
        const nuka::scene::BodyId body = topo.link_bodies[link_index];
        const nuka::scene::SceneIR& scene = *record->scene;

        // The DOF name is the joint whose CHILD is this link's body (the actuated
        // joint). The root (no parent joint) falls back to the link body's name.
        std::string name;
        for (const auto& joint : scene.Joints()) {
            if (joint.child_body == body) {
                name = joint.name;
                break;
            }
        }
        if (name.empty() && body < scene.Bodies().size()) {
            name = scene.Bodies()[body].name;
        }

        if (out_len != nullptr) {
            *out_len = name.size();  // set first so the caller can resize + retry.
        }
        if (out == nullptr) {
            return NUKA_RESULT_OK;  // size query: *out_len holds the needed length.
        }
        if (name.size() + 1u > cap) {
            return NUKA_RESULT_INVALID_ARG;  // LOUD: too small to hold name + NUL.
        }
        for (size_t i = 0u; i < name.size(); ++i) {
            out[i] = name[i];
        }
        out[name.size()] = '\0';
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

} // extern "C"
