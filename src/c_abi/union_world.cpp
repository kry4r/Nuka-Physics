// ---------------------------------------------------------------------------
// nuka::c_abi -- C ABI for the H1 WHOLE-BODY UNION WORLD (v0.8 G2). A MINIMAL
// extern "C" shim over nuka::runtime::coresident::BatchedUnifiedWorld carrying
// the COMPLETE G1d union scene (floating-base whole-body H1 + settled in-hand
// cup + ground + table). The scene's BatchedSceneTemplate is COOKED from the
// AUTHORED examples/scenes/h1_cup_table.nks (M7: scene::nks::Load ->
// scene::cook::CookSceneToUnionTemplate), which reads the baked settled IC +
// grasp config -- REPLACING the deleted 1026-line h1_union_scene_factory's
// FK/placement/200-step-settle authoring (proven equivalent to ~1e-8 by the
// h1_grasp_lift gate). Mirrors the grasp_world.cpp pattern: a local HandleTable,
// a record holding the world + the borrowed DeviceRecord (the device must
// outlive the union world), and the shared try/catch -> nuka_result_t convention.
// ---------------------------------------------------------------------------

#include "nuka/nuka_union.h"

#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "runtime/coresident/batched_unified_world.hpp"
#include "runtime/rigid/body_state.hpp"
#include "scene/cook/union_cook.hpp"
#include "scene/cook/union_scene_constants.hpp"
#include "scene/format/nks.hpp"
#include "scene/scene_ir.hpp"

#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace nuka::c_abi {

namespace coresident = nuka::runtime::coresident;
namespace cook = nuka::scene::cook;

// One batched union world bound to a device (the device MUST outlive this
// record -- the grasp/diffsim contract). Holds the world + the cooked scene
// metadata (drive tables / limits / grip columns) + the last_action echo.
struct UnionWorldRecord {
    DeviceRecord* device = nullptr;
    std::unique_ptr<coresident::BatchedUnifiedWorld> world;
    cook::CookedUnionScene scene;  // template + metadata (kept whole; small).
    uint32_t env_count = 0u;
    uint32_t action_dim = 0u;        // == dof_stride (51).
    uint32_t num_fingertips = 0u;
    uint32_t num_feet = 0u;
    uint32_t bodies_per_env = 0u;
    uint32_t cup_local_index = 0u;
    float gravity_z = 0.0f;
    float dt = 0.0f;
    // The exact bytes of the last successful set_actions (zeros before the
    // first call) -- the `last_action` obs field.
    std::vector<float> last_action;
    // Reused obs scratch so export_obs is allocation-free per step.
    coresident::ObsStateBatch obs;
};

}  // namespace nuka::c_abi

namespace {

using nuka::c_abi::DeviceRecord;
using nuka::c_abi::UnionWorldRecord;
namespace coresident = nuka::runtime::coresident;
namespace cook = nuka::scene::cook;

nuka::c_abi::HandleTable<nuka_union_world_t, UnionWorldRecord>& UnionWorldTable() {
    static nuka::c_abi::HandleTable<nuka_union_world_t, UnionWorldRecord> table;
    return table;
}

// The drive-table selector (which: 0 hold / 1 rest / 2 close).
const std::vector<coresident::H1UnionDriveEntry>* SelectDriveTable(
    const UnionWorldRecord& r, uint32_t which) {
    switch (which) {
        case 0u: return &r.scene.drive_hold;
        case 1u: return &r.scene.drive_rest;
        case 2u: return &r.scene.drive_close;
        default: return nullptr;
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// create / destroy
// ---------------------------------------------------------------------------
nuka_result_t nuka_union_world_create(nuka_device_handle device,
                                      const nuka_union_world_desc_t* desc,
                                      nuka_union_world_handle* out) {
    if (out == nullptr) return NUKA_RESULT_INVALID_ARG;
    *out = nullptr;
    if (desc == nullptr || desc->env_count == 0u) return NUKA_RESULT_INVALID_ARG;
    DeviceRecord* device_record = nuka::c_abi::DeviceTable().Get(device);
    if (device_record == nullptr) return NUKA_RESULT_NULL_HANDLE;

    // Sentinel defaults (the SAME constants on every host language -- no float-
    // literal seam): 0 gravity -> -9.81f; <=0 dt -> 1/240f. The scene itself is
    // the AUTHORED .nks (it imports h1_with_hand + the cup hull); the desc's
    // mjcf/cup paths now serve only as the file-existence guard for the assets
    // that scene references (empty -> the repo-relative defaults).
    const std::string mjcf =
        (desc->h1_mjcf_path != nullptr && desc->h1_mjcf_path[0] != '\0')
            ? desc->h1_mjcf_path
            : cook::kH1UnionMjcfDefault;
    const std::string cup =
        (desc->cup_usda_path != nullptr && desc->cup_usda_path[0] != '\0')
            ? desc->cup_usda_path
            : cook::kH1UnionCupDefault;
    const std::string nks_path = cook::kH1UnionNksDefault;
    const float gravity_z =
        (desc->gravity_z == 0.0f) ? cook::kH1UnionGravityZ : desc->gravity_z;
    const float dt =
        (desc->fixed_dt <= 0.0f) ? cook::kH1UnionDt : desc->fixed_dt;

    if (!std::filesystem::exists(nks_path) ||
        !std::filesystem::exists(mjcf) || !std::filesystem::exists(cup)) {
        return NUKA_RESULT_FILE_NOT_FOUND;
    }

    try {
        auto record = std::make_unique<UnionWorldRecord>();
        record->device = device_record;
        // Load the AUTHORED union scene + COOK it to the BatchedSceneTemplate
        // (reads the baked settled IC + grasp config -- the M7 replacement for
        // the factory's FK/placement/200-step-settle). Throws LOUDLY on a
        // degenerate scene (missing GraspConfig / initial_state).
        const nuka::scene::SceneIR scene = nuka::scene::nks::Load(nks_path);
        record->scene = cook::CookSceneToUnionTemplate(
            scene, static_cast<int>(desc->env_count));
        record->world = std::make_unique<coresident::BatchedUnifiedWorld>(
            device_record->context, record->scene.tmpl, desc->env_count,
            gravity_z, dt);
        record->env_count = record->world->EnvCount();
        record->action_dim = record->world->ActionDim();
        record->num_fingertips = record->world->NumFingertips();
        record->num_feet = static_cast<uint32_t>(record->scene.tmpl.feet.size());
        record->bodies_per_env = record->world->BodiesPerEnv();
        record->cup_local_index = record->scene.tmpl.cup_local_index;
        record->gravity_z = gravity_z;
        record->dt = dt;
        record->last_action.assign(
            static_cast<size_t>(record->env_count) * record->action_dim, 0.0f);

        *out = UnionWorldTable().Insert(std::move(record));
        return *out == nullptr ? NUKA_RESULT_INTERNAL : NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

void nuka_union_world_destroy(nuka_union_world_handle world) {
    (void)UnionWorldTable().Remove(world);
}

// ---------------------------------------------------------------------------
// metadata
// ---------------------------------------------------------------------------
nuka_result_t nuka_union_world_dims(nuka_union_world_handle world,
                                    nuka_union_world_dims_t* out) {
    if (out == nullptr) return NUKA_RESULT_INVALID_ARG;
    std::memset(out, 0, sizeof(*out));
    UnionWorldRecord* r = UnionWorldTable().Get(world);
    if (r == nullptr) return NUKA_RESULT_NULL_HANDLE;
    out->env_count = r->env_count;
    out->action_dim = r->action_dim;
    out->base_dof = r->scene.base_dof;
    out->num_fingertips = r->num_fingertips;
    out->num_feet = r->num_feet;
    out->link_count = r->scene.link_count;
    out->bodies_per_env = r->bodies_per_env;
    out->cup_local_index = r->cup_local_index;
    out->num_grip_dofs = static_cast<uint32_t>(r->scene.grip_dofs.size());
    out->drive_table_len = static_cast<uint32_t>(r->scene.drive_hold.size());
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_union_world_scene_info(nuka_union_world_handle world,
                                          nuka_union_scene_info_t* out) {
    if (out == nullptr) return NUKA_RESULT_INVALID_ARG;
    std::memset(out, 0, sizeof(*out));
    UnionWorldRecord* r = UnionWorldTable().Get(world);
    if (r == nullptr) return NUKA_RESULT_NULL_HANDLE;
    out->table_height = r->scene.table_height;
    out->cup_mass = r->scene.cup_mass;
    out->total_mass = static_cast<float>(r->scene.total_mass);
    out->poly_cx = r->scene.poly_cx;
    out->dt = r->dt;
    out->gravity_z = r->gravity_z;
    out->ankle_settle_tau[0] = r->scene.ankle_settle_tau[0];
    out->ankle_settle_tau[1] = r->scene.ankle_settle_tau[1];
    out->place_found = r->scene.place_found ? 1u : 0u;
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_union_world_drive_table(nuka_union_world_handle world,
                                           uint32_t which,
                                           uint32_t* dof, float* target,
                                           float* kp, float* kd, float* tlim,
                                           uint8_t* grip, size_t len) {
    UnionWorldRecord* r = UnionWorldTable().Get(world);
    if (r == nullptr) return NUKA_RESULT_NULL_HANDLE;
    const auto* table = SelectDriveTable(*r, which);
    if (table == nullptr || len != table->size()) return NUKA_RESULT_INVALID_ARG;
    for (size_t i = 0u; i < table->size(); ++i) {
        const coresident::H1UnionDriveEntry& e = (*table)[i];
        if (dof != nullptr) dof[i] = e.dof;
        if (target != nullptr) target[i] = e.target;
        if (kp != nullptr) kp[i] = e.kp;
        if (kd != nullptr) kd[i] = e.kd;
        if (tlim != nullptr) tlim[i] = e.tlim;
        if (grip != nullptr) grip[i] = e.grip;
    }
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_union_world_grip_dofs(nuka_union_world_handle world,
                                         uint32_t* dofs, size_t len) {
    UnionWorldRecord* r = UnionWorldTable().Get(world);
    if (r == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (dofs == nullptr || len != r->scene.grip_dofs.size())
        return NUKA_RESULT_INVALID_ARG;
    std::memcpy(dofs, r->scene.grip_dofs.data(), len * sizeof(uint32_t));
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_union_world_dof_limits(nuka_union_world_handle world,
                                          float* limits, size_t len) {
    UnionWorldRecord* r = UnionWorldTable().Get(world);
    if (r == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (limits == nullptr || len != r->scene.dof_torque_limit.size())
        return NUKA_RESULT_INVALID_ARG;
    std::memcpy(limits, r->scene.dof_torque_limit.data(), len * sizeof(float));
    return NUKA_RESULT_OK;
}

// ---------------------------------------------------------------------------
// action injection + step
// ---------------------------------------------------------------------------
nuka_result_t nuka_union_world_set_actions(nuka_union_world_handle world,
                                           const float* actions, size_t count) {
    UnionWorldRecord* r = UnionWorldTable().Get(world);
    if (r == nullptr) return NUKA_RESULT_NULL_HANDLE;
    const size_t want = static_cast<size_t>(r->env_count) * r->action_dim;
    if (actions == nullptr || count != want) return NUKA_RESULT_INVALID_ARG;
    try {
        r->world->SetActions(actions, count);
        std::memcpy(r->last_action.data(), actions, count * sizeof(float));
        return NUKA_RESULT_OK;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_union_world_step(nuka_union_world_handle world) {
    UnionWorldRecord* r = UnionWorldTable().Get(world);
    if (r == nullptr) return NUKA_RESULT_NULL_HANDLE;
    try {
        r->world->Step();
        return NUKA_RESULT_OK;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_union_world_step_with_actions(nuka_union_world_handle world,
                                                 const float* actions, size_t count) {
    const nuka_result_t r = nuka_union_world_set_actions(world, actions, count);
    if (r != NUKA_RESULT_OK) return r;
    return nuka_union_world_step(world);
}

// ---------------------------------------------------------------------------
// per-env seams
// ---------------------------------------------------------------------------
nuka_result_t nuka_union_world_set_table_enabled(nuka_union_world_handle world,
                                                 int32_t env, uint32_t enabled) {
    UnionWorldRecord* r = UnionWorldTable().Get(world);
    if (r == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (env >= 0 && static_cast<uint32_t>(env) >= r->env_count)
        return NUKA_RESULT_INVALID_ARG;
    try {
        if (env < 0) {
            r->world->SetTableEnabled(enabled != 0u);
        } else {
            r->world->SetTableEnabled(static_cast<uint32_t>(env), enabled != 0u);
        }
        return NUKA_RESULT_OK;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_union_world_table_enabled(nuka_union_world_handle world,
                                             uint32_t env, uint32_t* out) {
    if (out == nullptr) return NUKA_RESULT_INVALID_ARG;
    *out = 0u;
    UnionWorldRecord* r = UnionWorldTable().Get(world);
    if (r == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (env >= r->env_count) return NUKA_RESULT_INVALID_ARG;
    *out = r->world->TableEnabled(env) ? 1u : 0u;
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_union_world_set_base_pose(nuka_union_world_handle world,
                                             uint32_t env, const float* pose7) {
    UnionWorldRecord* r = UnionWorldTable().Get(world);
    if (r == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (pose7 == nullptr || env >= r->env_count) return NUKA_RESULT_INVALID_ARG;
    try {
        nuka::math::Transform pose;
        pose.position = nuka::math::Vec3{pose7[0], pose7[1], pose7[2]};
        pose.rotation = nuka::math::Quat{pose7[3], pose7[4], pose7[5], pose7[6]};
        r->world->SetGripperBasePose(env, pose);
        return NUKA_RESULT_OK;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_union_world_reset_envs(nuka_union_world_handle world,
                                          const uint32_t* env_ids, uint32_t count,
                                          uint64_t seed) {
    UnionWorldRecord* r = UnionWorldTable().Get(world);
    if (r == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (count == 0u || env_ids == nullptr) return NUKA_RESULT_OK;  // no-op OK.
    try {
        std::vector<uint32_t> ids;
        ids.reserve(count);
        for (uint32_t i = 0u; i < count; ++i) {
            if (env_ids[i] >= r->env_count) return NUKA_RESULT_INVALID_ARG;
            ids.push_back(env_ids[i]);
        }
        r->world->ResetEnvs(ids, seed);
        // A reset env's table toggle returns to the template state (ON) -- the
        // RL episode-boundary contract (mirrors the oracle's construction init).
        for (uint32_t e : ids) r->world->SetTableEnabled(e, r->scene.tmpl.has_table);
        return NUKA_RESULT_OK;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// batched obs export
// ---------------------------------------------------------------------------
nuka_result_t nuka_union_world_export_obs(nuka_union_world_handle world,
                                          float* q, size_t q_len,
                                          float* qdot, size_t qdot_len,
                                          float* base_pose, size_t bp_len,
                                          float* base_vel, size_t bv_len,
                                          float* fingertip_world_pos, size_t ft_len,
                                          float* finger_normal_impulse, size_t fi_len,
                                          float* last_action, size_t la_len) {
    UnionWorldRecord* r = UnionWorldTable().Get(world);
    if (r == nullptr) return NUKA_RESULT_NULL_HANDLE;
    const size_t n = r->env_count;
    const size_t want_q = n * r->action_dim;
    const size_t want_ft = n * 3u * r->num_fingertips;
    const size_t want_fi = n * r->num_fingertips;
    if (q != nullptr && q_len != want_q) return NUKA_RESULT_INVALID_ARG;
    if (qdot != nullptr && qdot_len != want_q) return NUKA_RESULT_INVALID_ARG;
    if (base_pose != nullptr && bp_len != n * 7u) return NUKA_RESULT_INVALID_ARG;
    if (base_vel != nullptr && bv_len != n * 6u) return NUKA_RESULT_INVALID_ARG;
    if (fingertip_world_pos != nullptr && ft_len != want_ft) return NUKA_RESULT_INVALID_ARG;
    if (finger_normal_impulse != nullptr && fi_len != want_fi) return NUKA_RESULT_INVALID_ARG;
    if (last_action != nullptr && la_len != want_q) return NUKA_RESULT_INVALID_ARG;
    try {
        r->world->ExportObsState(r->obs);
        const coresident::ObsStateBatch& o = r->obs;
        if (q != nullptr) std::memcpy(q, o.q.data(), o.q.size() * sizeof(float));
        if (qdot != nullptr)
            std::memcpy(qdot, o.qdot.data(), o.qdot.size() * sizeof(float));
        if (base_pose != nullptr)
            std::memcpy(base_pose, o.base_pose.data(),
                        o.base_pose.size() * sizeof(float));
        if (base_vel != nullptr)
            std::memcpy(base_vel, o.base_vel.data(),
                        o.base_vel.size() * sizeof(float));
        if (fingertip_world_pos != nullptr)
            std::memcpy(fingertip_world_pos, o.fingertip_world_pos.data(),
                        o.fingertip_world_pos.size() * sizeof(float));
        if (finger_normal_impulse != nullptr)
            std::memcpy(finger_normal_impulse, o.finger_normal_impulse.data(),
                        o.finger_normal_impulse.size() * sizeof(float));
        if (last_action != nullptr)
            std::memcpy(last_action, r->last_action.data(),
                        r->last_action.size() * sizeof(float));
        return NUKA_RESULT_OK;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_union_world_read_cup_contacts(nuka_union_world_handle world,
                                                 float* cup_pose, size_t cp_len,
                                                 float* cup_vel, size_t cv_len,
                                                 uint32_t* foot_rows, size_t fr_len,
                                                 float* foot_impulse, size_t fim_len,
                                                 uint32_t* table_rows, size_t tr_len,
                                                 float* table_impulse, size_t tim_len,
                                                 uint32_t* finger_contacts, size_t fc_len) {
    UnionWorldRecord* r = UnionWorldTable().Get(world);
    if (r == nullptr) return NUKA_RESULT_NULL_HANDLE;
    const size_t n = r->env_count;
    if (cup_pose != nullptr && cp_len != n * 7u) return NUKA_RESULT_INVALID_ARG;
    if (cup_vel != nullptr && cv_len != n * 6u) return NUKA_RESULT_INVALID_ARG;
    if (foot_rows != nullptr && fr_len != n) return NUKA_RESULT_INVALID_ARG;
    if (foot_impulse != nullptr && fim_len != n) return NUKA_RESULT_INVALID_ARG;
    if (table_rows != nullptr && tr_len != n) return NUKA_RESULT_INVALID_ARG;
    if (table_impulse != nullptr && tim_len != n) return NUKA_RESULT_INVALID_ARG;
    if (finger_contacts != nullptr && fc_len != n) return NUKA_RESULT_INVALID_ARG;
    try {
        const auto& reports = r->world->GraspReports();
        for (uint32_t e = 0u; e < n; ++e) {
            const nuka::runtime::rigid::BodyState& cup =
                r->world->Body(e, r->cup_local_index);
            if (cup_pose != nullptr) {
                float* p = cup_pose + static_cast<size_t>(e) * 7u;
                p[0] = cup.position.x; p[1] = cup.position.y; p[2] = cup.position.z;
                p[3] = cup.orientation.w; p[4] = cup.orientation.x;
                p[5] = cup.orientation.y; p[6] = cup.orientation.z;
            }
            if (cup_vel != nullptr) {
                float* v = cup_vel + static_cast<size_t>(e) * 6u;
                v[0] = cup.linear_velocity.x; v[1] = cup.linear_velocity.y;
                v[2] = cup.linear_velocity.z; v[3] = cup.angular_velocity.x;
                v[4] = cup.angular_velocity.y; v[5] = cup.angular_velocity.z;
            }
            const bool have = e < reports.size();
            if (foot_rows != nullptr)
                foot_rows[e] = have ? reports[e].foot_normal_rows : 0u;
            if (foot_impulse != nullptr)
                foot_impulse[e] =
                    have ? static_cast<float>(reports[e].foot_normal_impulse_sum) : 0.0f;
            if (table_rows != nullptr)
                table_rows[e] = have ? reports[e].table_row_count : 0u;
            if (table_impulse != nullptr)
                table_impulse[e] =
                    have ? static_cast<float>(reports[e].table_vertical_impulse) : 0.0f;
            if (finger_contacts != nullptr)
                finger_contacts[e] = have ? reports[e].finger_contacts : 0u;
        }
        return NUKA_RESULT_OK;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}
