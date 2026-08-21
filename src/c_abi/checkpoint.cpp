#include "nuka/nuka.h"

#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"
#include "nk/pipeline/world.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace nuka::c_abi {

struct WorldCheckpointRecord {
    nuka_world_handle owner = nullptr;
    std::vector<uint8_t> persistent;
    uint32_t simulated_step_count = 0u;
    runtime::WorldStepOptions step_options;
    uint32_t sparse_solver_backend = 0u;
    std::array<sensor::noise::SensorNoiseConfig, WorldRecord::kNoiseFieldCount>
        noise_config{};
    std::array<uint64_t, WorldRecord::kNoiseFieldCount> noise_seq{};
    sensor::noise::DomainRandomizationConfig dr_config;
    bool dr_baseline_captured = false;
    std::vector<float> dr_nominal_link_mass;
    std::vector<float> dr_nominal_joint_armature;
    float dr_nominal_gravity_z = 0.0f;
    float dr_nominal_friction = 0.0f;
};

HandleTable<nuka_checkpoint_t, WorldCheckpointRecord>& CheckpointTable() {
    static HandleTable<nuka_checkpoint_t, WorldCheckpointRecord> table;
    return table;
}

namespace {

constexpr uint64_t kFnvOffset = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

void HashBytes(uint64_t* hash, const void* bytes, size_t count) {
    const auto* data = static_cast<const uint8_t*>(bytes);
    for (size_t i = 0u; i < count; ++i) {
        *hash ^= data[i];
        *hash *= kFnvPrime;
    }
}

template <typename T>
void HashValue(uint64_t* hash, const T& value) {
    HashBytes(hash, &value, sizeof(value));
}

void HashFloatVector(uint64_t* hash, const std::vector<float>& values) {
    const uint64_t count = values.size();
    HashValue(hash, count);
    if (!values.empty()) {
        HashBytes(hash, values.data(), values.size() * sizeof(float));
    }
}

void CopyHostState(const WorldRecord& source, WorldCheckpointRecord* target) {
    target->simulated_step_count = source.simulated_step_count;
    target->step_options = source.step_options;
    target->sparse_solver_backend = source.sparse_solver_backend;
    for (uint32_t i = 0u; i < WorldRecord::kNoiseFieldCount; ++i) {
        target->noise_config[i] = source.noise_config[i];
        target->noise_seq[i] = source.noise_seq[i];
    }
    target->dr_config = source.dr_config;
    target->dr_baseline_captured = source.dr_baseline_captured;
    target->dr_nominal_link_mass = source.dr_nominal_link_mass;
    target->dr_nominal_joint_armature = source.dr_nominal_joint_armature;
    target->dr_nominal_gravity_z = source.dr_nominal_gravity_z;
    target->dr_nominal_friction = source.dr_nominal_friction;
}

void RestoreHostState(const WorldCheckpointRecord& source, WorldRecord* target) {
    target->simulated_step_count = source.simulated_step_count;
    target->step_options = source.step_options;
    target->sparse_solver_backend = source.sparse_solver_backend;
    for (uint32_t i = 0u; i < WorldRecord::kNoiseFieldCount; ++i) {
        target->noise_config[i] = source.noise_config[i];
        target->noise_seq[i] = source.noise_seq[i];
    }
    target->dr_config = source.dr_config;
    target->dr_baseline_captured = source.dr_baseline_captured;
    target->dr_nominal_link_mass = source.dr_nominal_link_mass;
    target->dr_nominal_joint_armature = source.dr_nominal_joint_armature;
    target->dr_nominal_gravity_z = source.dr_nominal_gravity_z;
    target->dr_nominal_friction = source.dr_nominal_friction;
    target->last_invariant_violations.clear();
    target->invariant_sampler.Reset();
}

void HashHostState(uint64_t* hash, const WorldRecord& record) {
    HashValue(hash, record.env_count);
    HashValue(hash, record.simulated_step_count);
    const uint8_t mode = static_cast<uint8_t>(record.control_mode);
    HashValue(hash, mode);
    HashValue(hash, record.sparse_solver_backend);
    HashValue(hash, record.step_options.gravity.x);
    HashValue(hash, record.step_options.gravity.y);
    HashValue(hash, record.step_options.gravity.z);
    HashValue(hash, record.step_options.dt);
    HashValue(hash, record.step_options.step_count);
    const uint8_t clear_forces = record.step_options.clear_forces_after_step ? 1u : 0u;
    const uint8_t contacts = record.step_options.enable_contacts ? 1u : 0u;
    HashValue(hash, clear_forces);
    HashValue(hash, contacts);
    HashValue(hash, record.step_options.solver_velocity_iterations);
    HashValue(hash, record.step_options.solver_position_iterations);
    HashValue(hash, record.step_options.solver_slop);
    HashValue(hash, record.step_options.solver_baumgarte);
    for (uint32_t i = 0u; i < WorldRecord::kNoiseFieldCount; ++i) {
        const uint32_t kind = static_cast<uint32_t>(record.noise_config[i].kind);
        HashValue(hash, kind);
        HashValue(hash, record.noise_config[i].param1);
        HashValue(hash, record.noise_config[i].param2);
        HashValue(hash, record.noise_config[i].seed);
        HashValue(hash, record.noise_seq[i]);
    }
    const auto hash_range = [hash](const sensor::noise::DomainRandomizationConfig::Range& range) {
        HashValue(hash, range.lo);
        HashValue(hash, range.hi);
    };
    hash_range(record.dr_config.mass_multiplier);
    hash_range(record.dr_config.friction_multiplier);
    hash_range(record.dr_config.restitution_offset);
    hash_range(record.dr_config.joint_armature_offset);
    hash_range(record.dr_config.gravity_z_offset);
    const uint8_t dr_enabled = record.dr_config.enabled ? 1u : 0u;
    const uint8_t baseline = record.dr_baseline_captured ? 1u : 0u;
    HashValue(hash, dr_enabled);
    HashValue(hash, record.dr_config.seed);
    HashValue(hash, baseline);
    HashFloatVector(hash, record.dr_nominal_link_mass);
    HashFloatVector(hash, record.dr_nominal_joint_armature);
    HashValue(hash, record.dr_nominal_gravity_z);
    HashValue(hash, record.dr_nominal_friction);
}

}  // namespace

}  // namespace nuka::c_abi

extern "C" {

nuka_result_t nuka_world_checkpoint_capture(nuka_world_handle world,
                                            nuka_checkpoint_handle* out) {
    if (out == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    *out = nullptr;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!record->world) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }
    try {
        auto checkpoint = std::make_unique<nuka::c_abi::WorldCheckpointRecord>();
        checkpoint->owner = world;
        if (!record->world->GetData().DownloadPersistent(&checkpoint->persistent)) {
            return NUKA_RESULT_INTERNAL;
        }
        nuka::c_abi::CopyHostState(*record, checkpoint.get());
        *out = nuka::c_abi::CheckpointTable().Insert(std::move(checkpoint));
        return *out == nullptr ? NUKA_RESULT_INTERNAL : NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

nuka_result_t nuka_world_checkpoint_restore(nuka_world_handle world,
                                            nuka_checkpoint_handle checkpoint) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    auto* saved = nuka::c_abi::CheckpointTable().Get(checkpoint);
    if (record == nullptr || saved == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (saved->owner != world) {
        return NUKA_RESULT_INVALID_ARG;
    }
    if (!record->world) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }
    try {
        if (!record->world->GetData().UploadPersistent(saved->persistent)) {
            return NUKA_RESULT_INVALID_ARG;
        }
        nuka::c_abi::RestoreHostState(*saved, record);
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

void nuka_checkpoint_destroy(nuka_checkpoint_handle checkpoint) {
    (void)nuka::c_abi::CheckpointTable().Remove(checkpoint);
}

nuka_result_t nuka_world_state_hash(nuka_world_handle world, uint64_t* out_hash) {
    if (out_hash == nullptr) {
        return NUKA_RESULT_INVALID_ARG;
    }
    *out_hash = 0u;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    if (!record->world) {
        return NUKA_RESULT_NOT_SUPPORTED;
    }
    try {
        std::vector<uint8_t> persistent;
        if (!record->world->GetData().DownloadPersistent(&persistent)) {
            return NUKA_RESULT_INTERNAL;
        }
        uint64_t hash = nuka::c_abi::kFnvOffset;
        constexpr char domain[] = "NukaStateHashV1";
        nuka::c_abi::HashBytes(&hash, domain, sizeof(domain));
        nuka::c_abi::HashHostState(&hash, *record);
        const uint64_t byte_count = persistent.size();
        nuka::c_abi::HashValue(&hash, byte_count);
        nuka::c_abi::HashBytes(&hash, persistent.data(), persistent.size());
        *out_hash = hash;
        return NUKA_RESULT_OK;
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

}  // extern "C"
