// Sensor observations keep measured values separate from the physics state.

#include "nuka/nuka_noise.h"

#include "c_abi/dlpack_table.hpp"
#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/scoped_device_guard.hpp"
#include "runtime/articulation/articulation_state.hpp"
#include "sensor/noise/n2_domain_randomization.hpp"
#include "sensor/noise/noise_config.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <cmath>
#include <exception>
#include <limits>
#include <new>
#include <vector>

namespace {

namespace noise = nuka::sensor::noise;

// Public field descriptors define the supported field indices.
bool FieldInRange(nuka_state_field_t field) {
    return nuka::c_abi::FindDlpackFieldRow(field) != nullptr;
}

nuka_result_t ResolveObservation(nuka_world_handle world, nuka::c_abi::WorldRecord& record,
    nuka_state_field_t field, nuka::sensor::Observation** observation, nuka_buffer_view_t* source) {
    const auto* descriptor = nuka::c_abi::FindDlpackFieldRow(field);
    if (!descriptor) return NUKA_RESULT_INVALID_ARG;
    if (descriptor->dtype != nuka::c_abi::kWireDtypeF32 ||
        descriptor->element_stride_bytes != sizeof(float)) return NUKA_RESULT_NOT_SUPPORTED;
    auto result = nuka_world_get_buffer_view(world, field, source);
    if (result != NUKA_RESULT_OK) return result;
    if (source->dtype != nuka::c_abi::kWireDtypeF32 ||
        source->element_stride_bytes != sizeof(float)) return NUKA_RESULT_NOT_SUPPORTED;
    if (!record.world) return NUKA_RESULT_NOT_SUPPORTED;
    const uint32_t env_count = record.world->EnvCount();
    if (!env_count || !source->element_count || source->element_count % env_count != 0u ||
        source->element_count > std::numeric_limits<uint32_t>::max()) return NUKA_RESULT_NOT_SUPPORTED;
    auto found = record.field_observations.find(field);
    if (found == record.field_observations.end()) {
        auto created = std::make_unique<nuka::sensor::Observation>();
        const auto status = created->Initialize(record.world->Backend(), env_count,
            static_cast<uint32_t>(source->element_count / env_count), static_cast<uint32_t>(field));
        if (status != nuka::phi::Status::Ok) return nuka::c_abi::MapStatusToResult(status);
        found = record.field_observations.emplace(field, std::move(created)).first;
    }
    *observation = found->second.get();
    return NUKA_RESULT_OK;
}

namespace articulation = nuka::runtime::articulation;

// Reads a global link's NOMINAL scalar mass from the host inertia mirror. In the
// 6x6 spatial inertia row-major layout MakeSpatialInertia produces, the bottom-
// right 3x3 block is mass*Identity, so the (3,3) element is the scalar mass. The
// static_assert pins the 6x6 row-major layout so a block reorder fails to compile
// instead of silently extracting the wrong scalar. This is the LIVE truth
// (set_link_mass keeps it in sync) and sidesteps the topo.masses vs bodies.masses
// ambiguity in the cooker fallback.
float MassFromInertia(const articulation::LinkSpatialInertia& inertia) {
    static_assert(sizeof(inertia.I) / sizeof(inertia.I[0]) == 36u,
                  "LinkSpatialInertia is not a flat 6x6 (row-major) matrix");
    constexpr uint32_t kSpatialDim = 6u;          // 6x6 spatial inertia.
    constexpr uint32_t kMassRow = 3u;             // bottom-right 3x3 == mass*I.
    return inertia.I[kMassRow * kSpatialDim + kMassRow];  // (3,3) == scalar mass.
}

// Resolves a GLOBAL link index to its (diagonal_inertia, inertial_frame) by
// walking the articulations in concatenation order -- the SAME resolution
// nuka_world_set_link_mass uses to reproduce the global-link layout
// BuildArticulationHostState emits. Returns false if out of range.
bool ResolveLinkInertiaParams(const articulation::ArticulationHostState& host,
                              uint32_t link_index,
                              nuka::math::Vec3* diagonal_inertia,
                              nuka::math::Transform* inertial_frame) {
    uint32_t global = 0u;
    for (const auto& topo : host.articulations) {
        const uint32_t link_count =
            static_cast<uint32_t>(topo.link_bodies.size());
        if (link_index < global + link_count) {
            const uint32_t local = link_index - global;
            *diagonal_inertia = (local < topo.inertias.size())
                                    ? topo.inertias[local]
                                    : nuka::math::Vec3{0.0f, 0.0f, 0.0f};
            *inertial_frame = (local < topo.inertial_frames.size())
                                  ? topo.inertial_frames[local]
                                  : nuka::math::Transform::Identity();
            return true;
        }
        global += link_count;
    }
    return false;
}

// Snapshots the NOMINAL baseline ONCE (per-link mass, per-DOF armature,
// gravity.z) so a repeated apply re-randomizes AROUND nominal (idempotent)
// instead of compounding a random walk. Idempotent itself (guarded by
// dr_baseline_captured).
void CaptureNominalBaseline(nuka::c_abi::WorldRecord& record) {
    if (record.dr_baseline_captured) {
        return;
    }
    const auto& host = record.articulation_host;
    const uint32_t total_links = host.TotalLinkCount();
    record.dr_nominal_link_mass.resize(total_links);
    for (uint32_t l = 0u; l < total_links && l < host.link_inertia.size(); ++l) {
        record.dr_nominal_link_mass[l] = MassFromInertia(host.link_inertia[l]);
    }
    record.dr_nominal_joint_armature = host.joint_armature;  // per-DOF copy
    record.dr_nominal_gravity_z = record.step_options.gravity.z;
    // M9: the legacy batched contact step params are gone; contact friction lives
    // on the nk Model material buckets. The DR friction multiplier has no host
    // scalar to poke -> the nominal stays 0 (inert). M10 named gap (RL contact DR
    // rebuilt on the nk world at M10).
    record.dr_nominal_friction = 0.0f;
    record.dr_baseline_captured = true;
}

// Applies the sampled per-episode randomization to the world's engine buffers.
// Single-env (env_count == 1) is the tested contact-free diff-sim path. mass +
// gravity are TAPE-VISIBLE (they change the contact-free ABA forward + gradient);
// armature is present but inert in that forward; friction/restitution have no
// contact buffer here (sampled into `sampled` for RL completeness, inert). See
// nuka_noise.h for the full per-param buffer mapping.
nuka_result_t ApplyPerEpisodeRandomization(
    nuka::c_abi::WorldRecord& record,
    const noise::DomainRandomizationConfig& cfg) {
    if (!cfg.enabled) {
        return NUKA_RESULT_OK;  // byte no-op -> oracle safe
    }
    if (!record.world || record.articulation_host.TotalLinkCount() == 0u) {
        return NUKA_RESULT_NOT_SUPPORTED;  // no link inertia to scale
    }
    if (record.device == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    CaptureNominalBaseline(record);

    auto& host = record.articulation_host;
    // The host mirror is SINGLE-ENV (CaptureArticulationHostMirror cooks the first
    // articulation only); the live nk arena LinkInertia/JointArmature fields are
    // env-replicated (links_per_env * env_count). The DR multiplier is per-env, so
    // we iterate the arena's full env-major span: per-env link count == the host
    // mirror's link count, which is the env-major tile stride.
    const uint32_t links_per_env = host.TotalLinkCount();
    const uint32_t env_count = (record.env_count == 0u) ? 1u : record.env_count;
    const uint32_t total_links = links_per_env * env_count;

    const cudaStream_t stream = nullptr;  // BUF-14: stream 0
    const int device_id = record.device->device_id;
    nuka::phi::ScopedDeviceGuard guard(device_id);

    auto* device_inertia =
        record.world->FieldPtr<articulation::LinkSpatialInertia>(
            nuka::nk::FieldId::LinkInertia);

    // --- mass: rebuild each link's spatial inertia from nominal*mult ----------
    // Per-env multiplier; applied to every link in that env replica. The host
    // mirror + nominal baseline are SINGLE-ENV (indexed by `local`); the live nk
    // arena LinkInertia field is env-replicated (indexed by the global `link`).
    for (uint32_t env = 0u; env < env_count; ++env) {
        const noise::SampledRandomization sampled =
            noise::SampleEpisodeRandomization(cfg, env);
        for (uint32_t local = 0u; local < links_per_env; ++local) {
            const uint32_t link = env * links_per_env + local;
            if (link >= total_links || local >= host.link_inertia.size()) {
                continue;
            }
            const float nominal =
                (local < record.dr_nominal_link_mass.size())
                    ? record.dr_nominal_link_mass[local]
                    : MassFromInertia(host.link_inertia[local]);
            const float new_mass = nominal * sampled.mass_multiplier;
            if (!(new_mass > 0.0f)) {
                continue;  // MakeSpatialInertia zeroes a non-positive mass
            }
            nuka::math::Vec3 diag{0.0f, 0.0f, 0.0f};
            nuka::math::Transform frame = nuka::math::Transform::Identity();
            if (!ResolveLinkInertiaParams(host, local, &diag, &frame)) {
                continue;
            }
            const articulation::LinkSpatialInertia new_inertia =
                articulation::MakeSpatialInertia(new_mass, diag, frame);
            // Keep the single-env host mirror in sync (overwritten each env, last
            // env's value wins -- the legacy behavior, the mirror is only the
            // diffsim dI/dmass source which DR does not gate on).
            host.link_inertia[local] = new_inertia;
            if (device_inertia != nullptr) {
                cudaError_t copy_status = cudaMemcpyAsync(
                    device_inertia + link, &new_inertia,
                    sizeof(articulation::LinkSpatialInertia),
                    cudaMemcpyHostToDevice, stream);
                if (copy_status != cudaSuccess) {
                    return NUKA_RESULT_INTERNAL;
                }
            }
        }
    }

    // --- joint armature: nominal + offset, per-link (env-tiled) --------------
    // The arena JointArmature field is per:link (env-replicated). The host mirror
    // + nominal baseline are single-env (indexed by `local`). Present on the
    // articulation state but INERT in the contact-free tape forward; set for RL
    // completeness + a consistent host/device mirror.
    auto* device_armature =
        record.world->FieldPtr<float>(nuka::nk::FieldId::JointArmature);
    if (!host.joint_armature.empty() && device_armature != nullptr) {
        const uint32_t dof_per_env =
            static_cast<uint32_t>(host.joint_armature.size());
        bool armature_changed = false;
        for (uint32_t env = 0u; env < env_count; ++env) {
            const noise::SampledRandomization sampled =
                noise::SampleEpisodeRandomization(cfg, env);
            for (uint32_t local = 0u; local < dof_per_env; ++local) {
                const float nominal_n =
                    (local < record.dr_nominal_joint_armature.size())
                        ? record.dr_nominal_joint_armature[local]
                        : 0.0f;
                const float value = nominal_n + sampled.joint_armature_offset;
                host.joint_armature[local] = value;  // single-env mirror sync
                const uint32_t dof = env * dof_per_env + local;
                cudaError_t copy_status = cudaMemcpyAsync(
                    device_armature + dof, &value, sizeof(float),
                    cudaMemcpyHostToDevice, stream);
                if (copy_status != cudaSuccess) {
                    return NUKA_RESULT_INTERNAL;
                }
                armature_changed = true;
            }
        }
        (void)armature_changed;
    }

    // --- gravity.z: nominal + offset (one world scalar; env 0's sample) ------
    // Gravity is a single world property -> the SAME offset applies to all envs
    // (it cannot be per-env). Read by nuka_tape_create into RolloutParams.
    {
        const noise::SampledRandomization sampled =
            noise::SampleEpisodeRandomization(cfg, 0u);
        record.step_options.gravity.z =
            record.dr_nominal_gravity_z + sampled.gravity_z_offset;
        // friction_multiplier: M9 named gap. The legacy batched contact step
        // params (the only friction host scalar) are gone; contact friction now
        // lives on the nk Model material buckets, which DR does not yet poke. The
        // single-env contact-free diffsim tape has no contact solve -> inert
        // either way. RL contact DR (incl. friction) is rebuilt on the nk world
        // at M10. restitution_offset: likewise no buffer in this path -> inert.
        (void)sampled;
    }

    cudaStreamSynchronize(stream);
    return NUKA_RESULT_OK;
}

}  // namespace

extern "C" {

nuka_result_t nuka_world_set_sensor_noise(nuka_world_handle world,
    nuka_state_field_t field, const nuka_sensor_noise_desc_t* desc) {
    if (!FieldInRange(field)) return NUKA_RESULT_INVALID_ARG;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (!record) return NUKA_RESULT_NULL_HANDLE;
    try {
        nuka::sensor::ObservationConfig config;
        if (desc) {
            config.noise.kind = static_cast<noise::NoiseKind>(desc->kind);
            config.noise.param1 = desc->param1;
            config.noise.param2 = desc->param2;
            config.noise.seed = desc->seed;
        }
        if (!nuka::sensor::ValidObservationConfig(config)) return NUKA_RESULT_INVALID_ARG;
        if (!desc && record->field_observations.find(field) == record->field_observations.end())
            return NUKA_RESULT_OK;
        nuka_buffer_view_t source{};
        nuka::sensor::Observation* observation = nullptr;
        const auto result = ResolveObservation(world, *record, field, &observation, &source);
        if (result != NUKA_RESULT_OK) return result;
        return nuka::c_abi::MapStatusToResult(observation->Configure(config));
    } catch (const std::bad_alloc&) { return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) { return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) { return NUKA_RESULT_INTERNAL; }
}

nuka_result_t nuka_world_set_sensor_error(nuka_world_handle world,
    nuka_state_field_t field, const nuka_sensor_error_desc_t* desc) {
    if (!FieldInRange(field) || (desc && desc->struct_size < sizeof(*desc)))
        return NUKA_RESULT_INVALID_ARG;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (!record) return NUKA_RESULT_NULL_HANDLE;
    if (!desc) return nuka_world_set_sensor_noise(world, field, nullptr);
    try {
        nuka::sensor::ObservationConfig config;
        config.noise.seed = desc->seed;
        auto& error = config.error;
        error.bias = desc->bias;
        error.scale_error = desc->scale_error;
        error.noise_density = desc->noise_density;
        error.initial_bias_stddev = desc->initial_bias_stddev;
        error.bias_random_walk = desc->bias_random_walk;
        error.correlated_bias_stddev = desc->correlated_bias_stddev;
        error.correlation_time = desc->correlation_time;
        error.quantization = desc->quantization;
        error.minimum = desc->minimum;
        error.maximum = desc->maximum;
        error.response_time = desc->response_time;
        error.temperature_coefficient = desc->temperature_coefficient;
        error.reference_temperature = desc->reference_temperature;
        error.saturation_enabled = desc->saturation_enabled;
        if (!nuka::sensor::ValidObservationConfig(config)) return NUKA_RESULT_INVALID_ARG;
        nuka_buffer_view_t source{};
        nuka::sensor::Observation* observation = nullptr;
        const auto result = ResolveObservation(world, *record, field, &observation, &source);
        if (result != NUKA_RESULT_OK) return result;
        return nuka::c_abi::MapStatusToResult(observation->Configure(config));
    } catch (const std::bad_alloc&) { return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) { return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) { return NUKA_RESULT_INTERNAL; }
}

nuka_result_t nuka_world_sample_observation(nuka_world_handle world,
    nuka_state_field_t field, double sample_interval, float temperature) {
    if (!FieldInRange(field) || !(sample_interval > 0.0) || !std::isfinite(sample_interval) ||
        sample_interval > std::numeric_limits<float>::max() || !std::isfinite(temperature))
        return NUKA_RESULT_INVALID_ARG;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (!record) return NUKA_RESULT_NULL_HANDLE;
    try {
        nuka_buffer_view_t source{};
        nuka::sensor::Observation* observation = nullptr;
        const auto result = ResolveObservation(world, *record, field, &observation, &source);
        if (result != NUKA_RESULT_OK) return result;
        const auto status = observation->Sample(static_cast<const float*>(source.device_ptr),
                                                sample_interval, temperature);
        if (status != nuka::phi::Status::Ok) return nuka::c_abi::MapStatusToResult(status);
        return nuka::c_abi::MapStatusToResult(record->world->Synchronize());
    } catch (const std::bad_alloc&) { return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) { return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) { return NUKA_RESULT_INTERNAL; }
}

nuka_result_t nuka_world_apply_sensor_noise(nuka_world_handle world, nuka_state_field_t field) {
    if (!FieldInRange(field)) return NUKA_RESULT_INVALID_ARG;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (!record) return NUKA_RESULT_NULL_HANDLE;
    const auto found = record->field_observations.find(field);
    const float temperature = found == record->field_observations.end() ? 25.0f :
        found->second->Config().error.reference_temperature;
    return nuka_world_sample_observation(world, field, record->step_options.dt, temperature);
}

nuka_result_t nuka_world_get_observation_view(nuka_world_handle world,
    nuka_state_field_t field, nuka_buffer_view_t* out) {
    if (!out) return NUKA_RESULT_INVALID_ARG;
    *out = {};
    if (!FieldInRange(field)) return NUKA_RESULT_INVALID_ARG;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (!record) return NUKA_RESULT_NULL_HANDLE;
    const auto found = record->field_observations.find(field);
    if (found == record->field_observations.end() || !found->second->Active()) return NUKA_RESULT_NOT_SUPPORTED;
    out->device_ptr = found->second->Values();
    out->element_count = found->second->ValueBytes() / sizeof(float);
    out->element_stride_bytes = sizeof(float);
    out->dtype = nuka::c_abi::kWireDtypeF32;
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_download_observation(nuka_world_handle world,
    nuka_state_field_t field, void* bytes, size_t nbytes, size_t byte_offset) {
    if (!FieldInRange(field)) return NUKA_RESULT_INVALID_ARG;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (!record) return NUKA_RESULT_NULL_HANDLE;
    const auto found = record->field_observations.find(field);
    if (found == record->field_observations.end() || !found->second->Active()) return NUKA_RESULT_NOT_SUPPORTED;
    try {
        return nuka::c_abi::MapStatusToResult(found->second->Download(bytes, nbytes, byte_offset));
    } catch (const std::exception& error) { return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) { return NUKA_RESULT_INTERNAL; }
}

nuka_result_t nuka_world_get_observation_stamp(nuka_world_handle world,
    nuka_state_field_t field, uint32_t env, nuka_observation_stamp_t* out) {
    if (!out) return NUKA_RESULT_INVALID_ARG;
    *out = {};
    if (!FieldInRange(field)) return NUKA_RESULT_INVALID_ARG;
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (!record) return NUKA_RESULT_NULL_HANDLE;
    const auto found = record->field_observations.find(field);
    if (found == record->field_observations.end() || !found->second->Active()) return NUKA_RESULT_NOT_SUPPORTED;
    try {
        nuka::sensor::ObservationStamp stamp;
        const auto status = found->second->ReadStamp(env, &stamp);
        if (status != nuka::phi::Status::Ok) return nuka::c_abi::MapStatusToResult(status);
        out->sequence = stamp.sequence;
        out->elapsed_time = stamp.elapsed_time;
        out->valid = stamp.valid;
        return NUKA_RESULT_OK;
    } catch (const std::exception& error) { return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) { return NUKA_RESULT_INTERNAL; }
}

nuka_result_t nuka_world_set_domain_randomization(
    nuka_world_handle world, const nuka_domain_randomization_desc_t* desc) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    // NULL desc clears DR back to disabled (apply becomes a byte no-op).
    if (desc == nullptr) {
        record->dr_config = noise::DomainRandomizationConfig{};
        record->dr_config.enabled = false;
        return NUKA_RESULT_OK;
    }
    noise::DomainRandomizationConfig cfg;
    cfg.mass_multiplier = {desc->mass_mul_lo, desc->mass_mul_hi};
    cfg.friction_multiplier = {desc->friction_mul_lo, desc->friction_mul_hi};
    cfg.restitution_offset = {desc->restitution_off_lo, desc->restitution_off_hi};
    cfg.joint_armature_offset = {desc->armature_off_lo, desc->armature_off_hi};
    cfg.gravity_z_offset = {desc->gravity_off_lo, desc->gravity_off_hi};
    cfg.seed = desc->seed;
    cfg.enabled = (desc->enabled != 0);
    record->dr_config = cfg;
    // NOTE: recording does NOT reset dr_baseline_captured -- the nominal baseline
    // is the world's true cooked state, independent of which DR ranges are set.
    return NUKA_RESULT_OK;
}

nuka_result_t nuka_world_apply_domain_randomization(nuka_world_handle world) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (record == nullptr) {
        return NUKA_RESULT_NULL_HANDLE;
    }
    try {
        return ApplyPerEpisodeRandomization(*record, record->dr_config);
    } catch (const std::bad_alloc&) {
        return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) {
        return NUKA_RESULT_INTERNAL;
    }
}

}  // extern "C"
