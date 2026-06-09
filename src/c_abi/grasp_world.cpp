// ---------------------------------------------------------------------------
// nuka::c_abi -- C ABI for the BATCHED GRASP WORLD (v0.8 A2). A MINIMAL extern "C"
// shim over nuka::runtime::coresident::BatchedUnifiedWorld so a Python PPO env can
// drive the validated batched grasp world. Mirrors the diffsim.cpp pattern: a
// local HandleTable, a record holding the world + the DeviceRecord it borrows the
// CUDA context from (the device must outlive the grasp world, like a Tape), and the
// shared try/catch -> nuka_result_t error convention.
//
// The scene is built by the SHARED, VALIDATED grasp_scene_factory (the SAME
// BatchedSceneTemplate the 21 nuka_batched_unified_world_test gates exercise), so
// the Python side trains on a certified scene.
// ---------------------------------------------------------------------------

#include "nuka/nuka_grasp.h"

#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"
#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"
#include "runtime/coresident/batched_unified_world.hpp"
#include "runtime/coresident/grasp_scene_factory.hpp"
#include "runtime/rigid/body_state.hpp"

#include <cstring>
#include <exception>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace nuka::c_abi {

namespace coresident = nuka::runtime::coresident;

// One batched grasp world bound to a device. Owns the BatchedUnifiedWorld and holds
// a pointer to the DeviceRecord whose phi::DeviceContext the world borrows by const
// reference -- the device MUST outlive this record (the diffsim TapeRecord contract).
// The validated scene template is built once via the shared grasp_scene_factory.
struct GraspWorldRecord {
    DeviceRecord* device = nullptr;
    std::unique_ptr<coresident::BatchedUnifiedWorld> world;
    // Cached scalar shapes (the world's getters are O(1) but cheap to mirror; the
    // export buffers are sized from these so we keep them on the record).
    uint32_t env_count = 0u;
    uint32_t action_dim = 0u;
    uint32_t num_fingertips = 0u;
    uint32_t bodies_per_env = 0u;
    uint32_t cup_local_index = 0u;
    // Reused obs scratch (resized once) so export_obs is allocation-free per step.
    coresident::ObsStateBatch obs;
};

}  // namespace nuka::c_abi

namespace {

using nuka::c_abi::DeviceRecord;
using nuka::c_abi::GraspWorldRecord;
namespace coresident = nuka::runtime::coresident;

nuka::c_abi::HandleTable<nuka_grasp_world_t, GraspWorldRecord>& GraspWorldTable() {
    static nuka::c_abi::HandleTable<nuka_grasp_world_t, GraspWorldRecord> table;
    return table;
}

// Common scalar-getter body: validate handle + out, then write *out.
nuka_result_t ScalarGet(nuka_grasp_world_handle world, uint32_t* out, uint32_t value) {
    if (out == nullptr) return NUKA_RESULT_INVALID_ARG;
    *out = 0u;
    GraspWorldRecord* record = GraspWorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    *out = value;
    return NUKA_RESULT_OK;
}

}  // namespace

// ---------------------------------------------------------------------------
// create / destroy
// ---------------------------------------------------------------------------
nuka_result_t nuka_grasp_world_create(nuka_device_handle device,
                                      const nuka_grasp_world_desc_t* desc,
                                      nuka_grasp_world_handle* out) {
    if (out == nullptr) return NUKA_RESULT_INVALID_ARG;
    *out = nullptr;
    if (desc == nullptr || desc->cup_asset_path == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    if (desc->env_count == 0u || desc->fixed_dt <= 0.0f) {
        return NUKA_RESULT_INVALID_ARG;
    }
    DeviceRecord* device_record = nuka::c_abi::DeviceTable().Get(device);
    if (device_record == nullptr) return NUKA_RESULT_NULL_HANDLE;

    try {
        // Load + cook the cup via the SHARED, VALIDATED factory (the SAME path the
        // 21 gates exercise). An absent / empty-hull asset -> FILE_NOT_FOUND.
        const coresident::GraspCupHull hull =
            coresident::LoadGraspCupHull(desc->cup_asset_path);
        if (hull.vcount == 0u) {
            return NUKA_RESULT_FILE_NOT_FOUND;
        }
        const coresident::GraspSceneBundle bundle =
            coresident::BuildGraspSceneBundle(hull, desc->grip_force, desc->friction_mu);
        const coresident::BatchedSceneTemplate tmpl = coresident::MakeGraspTemplate(bundle);

        auto record = std::make_unique<GraspWorldRecord>();
        record->device = device_record;
        record->world = std::make_unique<coresident::BatchedUnifiedWorld>(
            device_record->context, tmpl, desc->env_count, desc->gravity_z,
            desc->fixed_dt);
        record->env_count = record->world->EnvCount();
        record->action_dim = record->world->ActionDim();
        record->num_fingertips = record->world->NumFingertips();
        record->bodies_per_env = record->world->BodiesPerEnv();
        record->cup_local_index = tmpl.cup_local_index;

        *out = GraspWorldTable().Insert(std::move(record));
        return *out == nullptr ? NUKA_RESULT_INTERNAL : NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

void nuka_grasp_world_destroy(nuka_grasp_world_handle world) {
    (void)GraspWorldTable().Remove(world);
}

// ---------------------------------------------------------------------------
// scalar getters
// ---------------------------------------------------------------------------
nuka_result_t nuka_grasp_world_env_count(nuka_grasp_world_handle world, uint32_t* out) {
    if (out == nullptr) return NUKA_RESULT_INVALID_ARG;
    *out = 0u;
    GraspWorldRecord* record = GraspWorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    return ScalarGet(world, out, record->env_count);
}
nuka_result_t nuka_grasp_world_action_dim(nuka_grasp_world_handle world, uint32_t* out) {
    if (out == nullptr) return NUKA_RESULT_INVALID_ARG;
    *out = 0u;
    GraspWorldRecord* record = GraspWorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    return ScalarGet(world, out, record->action_dim);
}
nuka_result_t nuka_grasp_world_num_fingertips(nuka_grasp_world_handle world, uint32_t* out) {
    if (out == nullptr) return NUKA_RESULT_INVALID_ARG;
    *out = 0u;
    GraspWorldRecord* record = GraspWorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    return ScalarGet(world, out, record->num_fingertips);
}
nuka_result_t nuka_grasp_world_bodies_per_env(nuka_grasp_world_handle world, uint32_t* out) {
    if (out == nullptr) return NUKA_RESULT_INVALID_ARG;
    *out = 0u;
    GraspWorldRecord* record = GraspWorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    return ScalarGet(world, out, record->bodies_per_env);
}

// ---------------------------------------------------------------------------
// action injection + step
// ---------------------------------------------------------------------------
nuka_result_t nuka_grasp_world_set_actions(nuka_grasp_world_handle world,
                                           const float* actions, size_t count) {
    GraspWorldRecord* record = GraspWorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    const size_t want = static_cast<size_t>(record->env_count) * record->action_dim;
    if (actions == nullptr || count != want) return NUKA_RESULT_INVALID_ARG;
    try {
        record->world->SetActions(actions, count);
        return NUKA_RESULT_OK;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_grasp_world_step(nuka_grasp_world_handle world) {
    GraspWorldRecord* record = GraspWorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    try {
        record->world->Step();
        return NUKA_RESULT_OK;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_grasp_world_step_with_actions(nuka_grasp_world_handle world,
                                                 const float* actions, size_t count) {
    GraspWorldRecord* record = GraspWorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    const size_t want = static_cast<size_t>(record->env_count) * record->action_dim;
    if (actions == nullptr || count != want) return NUKA_RESULT_INVALID_ARG;
    try {
        record->world->Step(actions, count);  // SetActions + Step.
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
nuka_result_t nuka_grasp_world_export_obs(nuka_grasp_world_handle world,
                                          float* q, size_t q_len,
                                          float* qdot, size_t qdot_len,
                                          float* fingertip_world_pos, size_t ft_len,
                                          float* finger_normal_impulse, size_t imp_len) {
    GraspWorldRecord* record = GraspWorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    const size_t n = record->env_count;
    const size_t want_q = n * record->action_dim;
    const size_t want_ft = n * 3u * record->num_fingertips;
    const size_t want_imp = n * record->num_fingertips;
    // Validate every NON-NULL buffer's length up front (a NULL buffer is skipped).
    if (q != nullptr && q_len != want_q) return NUKA_RESULT_INVALID_ARG;
    if (qdot != nullptr && qdot_len != want_q) return NUKA_RESULT_INVALID_ARG;
    if (fingertip_world_pos != nullptr && ft_len != want_ft) return NUKA_RESULT_INVALID_ARG;
    if (finger_normal_impulse != nullptr && imp_len != want_imp) return NUKA_RESULT_INVALID_ARG;
    try {
        record->world->ExportObsState(record->obs);
        const coresident::ObsStateBatch& o = record->obs;
        if (q != nullptr) std::memcpy(q, o.q.data(), o.q.size() * sizeof(float));
        if (qdot != nullptr) std::memcpy(qdot, o.qdot.data(), o.qdot.size() * sizeof(float));
        if (fingertip_world_pos != nullptr)
            std::memcpy(fingertip_world_pos, o.fingertip_world_pos.data(),
                        o.fingertip_world_pos.size() * sizeof(float));
        if (finger_normal_impulse != nullptr)
            std::memcpy(finger_normal_impulse, o.finger_normal_impulse.data(),
                        o.finger_normal_impulse.size() * sizeof(float));
        return NUKA_RESULT_OK;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// cup pose / velocity + per-env contact signals
// ---------------------------------------------------------------------------
nuka_result_t nuka_grasp_world_read_cup(nuka_grasp_world_handle world,
                                        float* cup_pose, size_t pose_len,
                                        float* cup_vel, size_t vel_len,
                                        float* cup_z, size_t z_len,
                                        float* cup_vz, size_t vz_len,
                                        uint32_t* finger_contacts, size_t fc_len) {
    GraspWorldRecord* record = GraspWorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    const size_t n = record->env_count;
    if (cup_pose != nullptr && pose_len != n * 7u) return NUKA_RESULT_INVALID_ARG;
    if (cup_vel != nullptr && vel_len != n * 6u) return NUKA_RESULT_INVALID_ARG;
    if (cup_z != nullptr && z_len != n) return NUKA_RESULT_INVALID_ARG;
    if (cup_vz != nullptr && vz_len != n) return NUKA_RESULT_INVALID_ARG;
    if (finger_contacts != nullptr && fc_len != n) return NUKA_RESULT_INVALID_ARG;
    try {
        const auto& reports = record->world->GraspReports();
        for (uint32_t e = 0u; e < n; ++e) {
            const nuka::runtime::rigid::BodyState& cup =
                record->world->Body(e, record->cup_local_index);
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
            if (cup_z != nullptr) cup_z[e] = cup.position.z;
            if (cup_vz != nullptr) cup_vz[e] = cup.linear_velocity.z;
            if (finger_contacts != nullptr) {
                finger_contacts[e] =
                    (e < reports.size()) ? reports[e].finger_contacts : 0u;
            }
        }
        return NUKA_RESULT_OK;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

// ---------------------------------------------------------------------------
// per-env autoreset
// ---------------------------------------------------------------------------
nuka_result_t nuka_grasp_world_reset_envs(nuka_grasp_world_handle world,
                                          const uint32_t* env_ids, uint32_t count,
                                          uint64_t seed) {
    GraspWorldRecord* record = GraspWorldTable().Get(world);
    if (record == nullptr) return NUKA_RESULT_NULL_HANDLE;
    if (count == 0u || env_ids == nullptr) return NUKA_RESULT_OK;  // no-op OK.
    try {
        std::vector<uint32_t> ids;
        ids.reserve(count);
        for (uint32_t i = 0u; i < count; ++i) {
            if (env_ids[i] >= record->env_count) return NUKA_RESULT_INVALID_ARG;
            ids.push_back(env_ids[i]);
        }
        record->world->ResetEnvs(ids, seed);
        return NUKA_RESULT_OK;
    } catch (const std::exception& e) {
        return nuka::c_abi::MapExceptionToResult(e);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}
