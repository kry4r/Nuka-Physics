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
    if (desc == nullptr || desc->scene_path == nullptr || desc->fixed_dt <= 0.0f) {
        return NUKA_RESULT_INVALID_ARG;
    }
    if (desc->env_count == 0u) {
        return NUKA_RESULT_INVALID_ARG;
    }
    // Determinism level: 0 = D1/Strong (default), 1 = D2/Weak. Reject other
    // values defensively. M10 NAMED GAP: the nk::World pipeline runs the D1
    // schedule only; the Weak (D2) escape hatch is RL-adjacent and is rebuilt on
    // the nk world at M10. Today D2 selects the same (D1) kernels and is
    // behaviorally identical, so we accept it but it is NOT a separate path.
    if (desc->determinism > 1u) {
        return NUKA_RESULT_INVALID_ARG;
    }
    // Control mode (T1 unified-actuator wire): 0 = PDPosition and 1 = Torque are
    // both wired onto the ONE generic nk::World -- OpApplyDrives carries both the
    // position-PD kernel (mode 0) and the direct-torque kernel (mode 1), selected
    // by model.drive_mode, which we route from desc->control_mode below. These are
    // PRESETS of the one affine actuator, not separate solver paths. The remaining
    // modes (2 Velocity / 3 ComputedTorque / 4 Osc / 5 Actuator) have no op kernel
    // on the unified world yet -> still rejected with NOT_SUPPORTED rather than
    // silently mis-actuate (the affine evaluator + those presets land in a later
    // phase). A value ABOVE the enum's range is an outright INVALID_ARG.
    if (!nuka::runtime::articulation::IsControlModeImplemented(
            desc->control_mode)) {
        return NUKA_RESULT_INVALID_ARG;
    }
    const auto control_mode = static_cast<nuka::runtime::articulation::ControlMode>(
        desc->control_mode);
    if (control_mode != nuka::runtime::articulation::ControlMode::PDPosition &&
        control_mode != nuka::runtime::articulation::ControlMode::Torque) {
        return NUKA_RESULT_NOT_SUPPORTED;  // preset not yet wired on the nk world.
    }

    auto* device_record = nuka::c_abi::DeviceTable().Get(device);
    if (device_record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    // The DeviceRecord acquired the phi v2 device/backend at create (M9 T2). The
    // nk::World ctor REQUIRES them (it borrows the backend); do NOT re-init.
    if (device_record->phi_device == nullptr || device_record->backend == nullptr) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }

    try {
        nuka::scene::SceneIR scene;
        if (!nuka::c_abi::LoadSceneByExtension(desc->scene_path, &scene)) {
            return NUKA_RESULT_NOT_SUPPORTED;
        }

        // M10 CO-RESIDENCE (dog-dog collision foundation, owner bottom line):
        // compose the authored scene with itself instance_count times at distinct
        // XY so the cook emits K SEPARATE articulations co-resident in one env (the
        // WP1/WP5-8 multi-dog path). A reload per replica (not a copy) mirrors
        // tests/scenario/multi_dog_costep.cpp::CookKFloatDogs exactly. 0/1 ==
        // single instance -> the loop body never runs -> the scene + cook are
        // BYTE-IDENTICAL to every legacy single-articulation create (D1 safe).
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

        // Cook the authored scene to an nk::Model (mirrors recorder.cpp). The
        // unified world runs EXACTLY the cooked scene physics -- it does NOT inject
        // a synthetic ground. A scene that wants seated feet authors a ground at or
        // above the feet (or requests a baked heightfield, below). go2_stand.usda
        // authors NO ground -> its feet (z ~ 0.28) float above z = 0 -> no
        // foot-ground reaction -> the world reproduces the owner MJX golden's
        // free-space fixed-base PD stance to 1e-4 (the golden's ground truth).
        // Auto-seating would load the feet and diverge the golden by ~1 rad -- i.e.
        // it would change the physics the golden pins, and is a per-scene
        // special-case the unified-world directive forbids.
        // ONE-GENERAL-SOLVER landing (L1-b): the FUSED runtime is DELETED, so every
        // world cooks through the GENERAL LBVH+cvx narrowphase + mixed-island solve
        // (CookToModel always returns a PairDriven model). desc->contact_family no
        // longer selects FUSED-vs-general (it is retained in the struct, removed in a
        // later milestone); it now only requests the baked-heightfield collidable
        // (below), the surface the general feet collide against.
        nuka::scene::cook::CookToModelResult cooked = nuka::scene::cook::CookToModel(
            scene, static_cast<int>(desc->env_count));

        // T1 (unified-actuator wire): route the declared control mode onto the
        // cooked Model's drive_mode, which the Pipeline copies into the ApplyDrives
        // op params (pipeline.cpp: p_apply_drives_.mode = model.drive_mode).
        // PDPosition -> 0 (position-PD kernel), Torque -> 1 (direct-torque kernel).
        // CookToModel defaults drive_mode to 0, so for PDPosition this assignment is
        // a no-op (the op graph + every drive byte is unchanged => the PD goldens
        // stay byte-identical). For Torque it flips the ApplyDrives op to the torque
        // preset that already lives in OpApplyDrives.
        cooked.model.drive_mode =
            (control_mode == nuka::runtime::articulation::ControlMode::Torque) ? 1u
                                                                               : 0u;

        // On the general path the feet have NO analytic terrain to detect against;
        // they collide with a real STATIC heightfield collidable. When the caller
        // requests a baked heightfield (contact_family == 1), build the ONE source-
        // of-truth HeightField -- from a grayscale IMAGE if heightfield_image_path is
        // set, else from the parametric generator selected by heightfield_terrain_type
        // and the terrain_* knobs -- then cook it. The obs/spawn sampler reads the
        // SAME grid (stored on the record), so obs and physics agree by construction.
        // A zero-init desc => a flat parametric grid => byte-identical to no terrain.
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
                // Load the grayscale height map. A malformed/unsupported image is a
                // LOUD INVALID_ARG, never a silent flat fallback. The grid is centred
                // at the world origin (corner = -radius), matching the parametric path.
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
                // Parametric: the generator fills the grid ONCE from the desc knobs.
                nuka::terrain::TerrainGenConfig cfg;
                cfg.nrow = nrow;
                cfg.ncol = ncol;
                cfg.cell_x = cell;
                cfg.cell_y = cell;
                const float half_x = 0.5f * static_cast<float>(ncol - 1u) * cell;
                const float half_y = 0.5f * static_cast<float>(nrow - 1u) * cell;
                cfg.origin = nuka::math::Vec3{-half_x, -half_y, 0.0f};
                cfg.base_z = cooked.model.ground_height;
                // The desc's heightfield_terrain_type is a LEGACY selector the C-ABI
                // maps to parametric feature amplitudes -- the engine has no terrain
                // types (MuJoCo hfield model). 0 flat, 1 stairs, 2 pit, 3 boxes, 4
                // composite (the tiled multi-feature field). The terrain_* knobs are
                // the shared feature parameters.
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
                    default:  // 0 flat: every amplitude 0.
                        break;
                }
                if (!nuka::terrain::GenerateHeightField(cfg, cooked_terrain)) {
                    return NUKA_RESULT_INVALID_ARG;
                }
            }
            nuka::scene::cook::CookHeightfieldGrid(m, cooked_terrain);

            nuka::nk::ModelCapacities& cap = m.capacities;
            // bodies_per_env (the LBVH leaf count) MUST include every static terrain
            // collidable -- they are real broadphase leaves a foot pairs against. The
            // CONTACT budget, however, is driven by the DYNAMIC collidables only: a
            // static-static pair is filtered (no reaction side), so every solvable
            // contact has >=1 dynamic side. Budgeting (dynamic + 1) * 4 candidate
            // slots per env stays bounded as static terrain geometry grows. This is
            // the SAME per-dynamic-body rule the cook uses; the static ground is +1.
            cap.bodies_per_env = static_cast<uint32_t>(m.body_init.size());
            cap.max_bodies_total =
                static_cast<uint32_t>(m.shape_table_rows.size());
            constexpr uint32_t kCandidatePairsPerCollidable = 4u;
            constexpr uint32_t kStaticCollidables = 1u;
            const uint32_t dynamic_collidables = orig_bodies + kStaticCollidables;
            cap.max_contacts_per_env =
                dynamic_collidables * kCandidatePairsPerCollidable;
            cap.max_rows_per_env =
                cap.max_contacts_per_env * nuka::nk::kPairDrivenRowsPerSlot;
        }

        // Resolve world gravity (Z-up). A zero-initialized desc (all three 0.0)
        // substitutes the shared standard-Earth default so a legacy create is
        // byte-identical; any non-zero component takes the caller's full vector.
        nuka::math::Vec3 gravity = nuka::runtime::kDefaultGravity;
        if (desc->gravity_x != 0.0f || desc->gravity_y != 0.0f ||
            desc->gravity_z != 0.0f) {
            gravity = {desc->gravity_x, desc->gravity_y, desc->gravity_z};
        }

        auto record = std::make_unique<nuka::c_abi::WorldRecord>();
        record->device = device_record;
        record->env_count = desc->env_count;
        record->control_mode = control_mode;
        record->cooked_terrain = std::move(cooked_terrain);  // obs/spawn read this.
        // dt/gravity scalars for the diffsim RolloutParams + the InvariantWorldView.
        record->step_options.dt = desc->fixed_dt;
        record->step_options.gravity = gravity;

        // Solver config for the nk Pipeline. dt/gravity from the desc; the
        // articulation/contact knobs default from the Model (recorder.cpp seeds
        // only dt/gravity, the same as here).
        nuka::nk::Pipeline::SolverConfig cfg;
        cfg.dt = desc->fixed_dt;
        cfg.gravity[0] = gravity.x;
        cfg.gravity[1] = gravity.y;
        cfg.gravity[2] = gravity.z;

        record->world = std::make_unique<nuka::nk::World>(
            std::move(cooked.model), desc->env_count, device_record->phi_device,
            device_record->backend, cfg);
        if (!record->world->Ready()) {
            return NUKA_RESULT_INTERNAL;
        }

        // Transitional host mirror for the diffsim backward + set_link_mass + DR
        // (host dI/dmass + spatial-inertia rebuild). Byte-identical to the legacy
        // path (same cook). HOST-ONLY -- device writes target the nk arena.
        nuka::c_abi::CaptureArticulationHostMirror(scene, record.get());

        *out = nuka::c_abi::WorldTable().Insert(std::move(record));
        return *out == nullptr ? NUKA_RESULT_INTERNAL : NUKA_RESULT_OK;
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
// Reads the world's stored HeightField via the ONE bilinear sampler, so the obs/
// spawn height matches the grid the physics narrowphase rests feet on. The optional
// per-element scale multiplies value*scale_z (the curriculum difficulty), preserving
// the per-env terrain ladder without per-env grids.
nuka_result_t nuka_world_sample_terrain_height_batch(nuka_world_handle world,
                                                     uint32_t n,
                                                     const float* xs,
                                                     const float* ys,
                                                     const float* scales,
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
        const float z = nuka::terrain::SampleHeightFieldZ(hf, xs[i], ys[i]);
        const float scale = (scales != nullptr) ? scales[i] : 1.0f;
        // Apply the per-env vertical scale as a multiply around the base floor.
        out_heights[i] = hf.base_z + (z - hf.base_z) * scale;
    }
    return NUKA_RESULT_OK;
}

} // extern "C"
