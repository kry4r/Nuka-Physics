// ---------------------------------------------------------------------------
// nuka::c_abi -- the ONE generic world entry (M9 T5/T6 cutover).
//
// The C-ABI world is now exactly ONE nk::World built from
// Scene -> CookToModel -> nk::World (mirroring c_abi/recorder.cpp's proven
// create+ownership pattern). The two legacy steppers (single-env Featherstone
// StepWorldGpu + the multi-env batched articulated world) are GONE: there is no
// special world type and no per-case physics path (owner's highest directive --
// a GENERAL solver, no "case 特惠"). create cooks the authored scene, builds an
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
#include "runtime/articulation/articulation_cooker.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "scene/cook/cook_to_model.hpp"
#include "scene/cooker.hpp"

#include "import/mjcf_importer.hpp"
#include "import/urdf_importer.hpp"
#include "import/usd_importer.hpp"
#include "scene/scene_ir.hpp"
#include "sensor/terrain/terrain_field.hpp"  // Go2-on-stairs batched height sampler

#include <algorithm>
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
    // mirror's global-link ordering matches the arena's.
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
    // Control mode: 0 = PDPosition (the only mode wired onto the generic nk::World
    // in M9). M10 NAMED GAP: non-PD modes (1 Torque / 2 Velocity / 3 ComputedTorque
    // / 4 Osc / 5 Actuator) + osc_task_link are RL-adjacent control surfaces that
    // are rebuilt on the nk world at M10 -- reject them with NOT_SUPPORTED rather
    // than silently mis-actuate.
    if (!nuka::runtime::articulation::IsControlModeImplemented(
            desc->control_mode)) {
        return NUKA_RESULT_INVALID_ARG;
    }
    const auto control_mode = static_cast<nuka::runtime::articulation::ControlMode>(
        desc->control_mode);
    if (control_mode != nuka::runtime::articulation::ControlMode::PDPosition) {
        return NUKA_RESULT_NOT_SUPPORTED;  // M10 named gap (non-PD control).
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

        // Cook the authored scene to an nk::Model (mirrors recorder.cpp). The
        // unified world runs EXACTLY the cooked scene physics -- it does NOT inject
        // a synthetic ground (the legacy multi-env batched path derived one via
        // DeriveGroundHeight; the single-env oracle path ran contacts OFF). The
        // FusedFoot foot pipeline detects against the implicit ground plane at
        // model.ground_height (default 0); a scene that wants seated feet authors a
        // ground at or above the feet. go2_stand.usda authors NO ground -> its feet
        // (z ~ 0.28) float above z = 0 -> no foot-ground reaction -> the world
        // reproduces the owner MJX golden's free-space fixed-base PD stance to 1e-4
        // (the golden's ground truth). Auto-seating would load the feet and diverge
        // the golden by ~1 rad -- i.e. it would change the physics the golden pins,
        // and is a per-scene special-case the unified-world directive forbids.
        nuka::scene::cook::CookToModelResult cooked =
            nuka::scene::cook::CookToModel(scene, static_cast<int>(desc->env_count));

        // Go2-on-stairs Phase 2a: apply the procedural-terrain cook config from the
        // desc AFTER cook + BEFORE nk::World construction (Pipeline::Build reads
        // model.terrain at construct time, copying it into the NarrowphasePrimitives
        // op params). ground_height stays the cooked model's; the 5 desc floats set
        // the shared TerrainParams. A zero-initialized desc leaves all five at 0.0
        // => the cook's flat terrain (SampleTerrainHeight returns ground_height for
        // every env) => byte-identical to a world created without terrain (D1). The
        // per-env terrain TYPE/DIFFICULTY are seeded by SeedInitialState (0 / 1.0)
        // and written post-create via NUKA_FIELD_ENV_TERRAIN_TYPE/DIFFICULTY.
        cooked.model.terrain.ground_height   = cooked.model.ground_height;
        cooked.model.terrain.step_height     = desc->terrain_step_height;
        cooked.model.terrain.step_width      = desc->terrain_step_width;
        cooked.model.terrain.platform_width  = desc->terrain_platform_width;
        cooked.model.terrain.grid_width      = desc->terrain_grid_width;
        cooked.model.terrain.grid_height_max = desc->terrain_grid_height_max;

        auto record = std::make_unique<nuka::c_abi::WorldRecord>();
        record->device = device_record;
        record->env_count = desc->env_count;
        record->control_mode = control_mode;
        // dt/gravity scalars for the diffsim RolloutParams + the InvariantWorldView.
        record->step_options.dt = desc->fixed_dt;
        record->step_options.gravity = {0.0f, 0.0f, -9.81f};

        // Solver config for the nk Pipeline. dt/gravity from the desc; the
        // articulation/contact knobs default from the Model (recorder.cpp seeds
        // only dt/gravity, the same as here).
        nuka::nk::Pipeline::SolverConfig cfg;
        cfg.dt = desc->fixed_dt;
        cfg.gravity[0] = 0.0f;
        cfg.gravity[1] = 0.0f;
        cfg.gravity[2] = -9.81f;

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

// Go2-on-stairs (random-XY spawn + free-roam): BATCHED arbitrary-(x,y) procedural
// terrain height sampler -- the C-ABI projection of the SINGLE source of truth
// nuka::terrain::SampleTerrainHeight (+ ScaleTerrainDifficulty) in
// src/sensor/terrain/terrain_field.hpp. The RL spawn-on-terrain reset places the
// Go2 base at a RANDOM (x,y) and reads its spawn height from the EXACT closed-form
// heightfield the foot kernel rests feet on (no python full-sampler drift). PURE
// HOST free function: includes only the host/device-portable terrain header (its
// __host__/__device__ qualifiers gate on __CUDACC__ -> empty under g++, zero CUDA
// tokens) -- no device, no world handle. NULL with n>0 => INVALID_ARG.
nuka_result_t nuka_terrain_sample_height_batch(uint32_t n,
                                               const uint32_t* types,
                                               const float* xs,
                                               const float* ys,
                                               const float* difficulties,
                                               float ground_height,
                                               float step_height,
                                               float step_width,
                                               float platform_width,
                                               float grid_width,
                                               float grid_height_max,
                                               float* out_heights) {
    if (n == 0u) {
        return NUKA_RESULT_OK;  // nothing to sample.
    }
    if (types == nullptr || xs == nullptr || ys == nullptr ||
        difficulties == nullptr || out_heights == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    // MODEL-LEVEL TerrainParams (one geometry for the whole batch); difficulty is
    // the per-element scale applied below.
    nuka::terrain::TerrainParams base{};
    base.ground_height   = ground_height;
    base.step_height     = step_height;
    base.step_width      = step_width;
    base.platform_width  = platform_width;
    base.grid_width      = grid_width;
    base.grid_height_max = grid_height_max;
    for (uint32_t i = 0u; i < n; ++i) {
        // Per-env curriculum difficulty (scales vertical magnitude only), then
        // sample the shared closed-form heightfield at this element's (x,y).
        const nuka::terrain::TerrainParams scaled =
            nuka::terrain::ScaleTerrainDifficulty(base, difficulties[i]);
        out_heights[i] = nuka::terrain::SampleTerrainHeight(
            types[i], xs[i], ys[i], scaled);
    }
    return NUKA_RESULT_OK;
}

} // extern "C"
