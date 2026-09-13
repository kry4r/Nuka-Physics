#include "nuka/nuka.h"

#include "c_abi/handle_table.hpp"
#include "c_abi/internal.hpp"
#include "nk/pipeline/world.hpp"
#include "render/sensor_backend.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

namespace nuka::c_abi {

struct WorldCheckpointRecord {
    nuka_world_handle owner = nullptr;
    std::vector<uint8_t> persistent;
    std::vector<runtime::articulation::LinkSpatialInertia> link_inertia;
    std::vector<float> joint_armature;
    uint32_t simulated_step_count = 0u;
    runtime::WorldStepOptions step_options;
    uint32_t sparse_solver_backend = 0u;
    std::map<nuka_state_field_t, sensor::ObservationSnapshot> observations;
    sensor::StateSensorBankSnapshot state_sensors;
    uint64_t sensor_revision = 0u;
    bool sensor_rendered = false;
    rt::SensorStateSnapshot imaging;
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

void HashObservationConfig(uint64_t* hash, const sensor::ObservationConfig& config) {
    HashValue(hash, static_cast<uint32_t>(config.noise.kind));
    HashValue(hash, config.noise.param1);
    HashValue(hash, config.noise.param2);
    HashValue(hash, config.noise.seed);
    const auto& e = config.error;
    const float parameters[] = {e.bias, e.scale_error, e.noise_density, e.initial_bias_stddev,
        e.bias_random_walk, e.correlated_bias_stddev, e.correlation_time, e.quantization,
        e.minimum, e.maximum, e.response_time, e.temperature_coefficient, e.reference_temperature};
    HashBytes(hash, parameters, sizeof(parameters));
    HashValue(hash, e.saturation_enabled);
}

void HashImagingState(uint64_t* hash, const rt::SensorStateSnapshot& snapshot) {
    const auto& state = snapshot.imaging;
    HashValue(hash, state.env_count);
    HashValue(hash, uint64_t{state.cameras.size()});
    for (const auto& c : state.cameras) {
        HashValue(hash, c.enabled);
        HashValue(hash, c.shot_noise);
        HashValue(hash, c.adc_bits);
        HashValue(hash, c.exposure_time);
        HashValue(hash, c.electrons_per_unit_second);
        HashValue(hash, c.full_well_electrons);
        HashValue(hash, c.read_noise_electrons);
        HashValue(hash, c.row_noise_electrons);
        HashValue(hash, c.dark_current);
        HashValue(hash, c.dark_doubling_temperature);
        HashValue(hash, c.temperature);
        HashValue(hash, c.reference_temperature);
        HashValue(hash, c.pixel_gain_stddev);
        HashValue(hash, c.pixel_offset_stddev_electrons);
        HashValue(hash, c.analog_gain);
        HashValue(hash, c.black_level_electrons);
        HashValue(hash, c.dead_pixel_probability);
        HashValue(hash, c.hot_pixel_probability);
        HashValue(hash, c.hot_pixel_current);
        HashValue(hash, c.seed);
    }
    for (const auto* configs : {&state.depths, &state.lidars}) {
        HashValue(hash, uint64_t{configs->size()});
        for (const auto& c : *configs) {
            HashValue(hash, c.enabled);
            HashValue(hash, c.bias);
            HashValue(hash, c.scale_error);
            HashValue(hash, c.distance_stddev);
            HashValue(hash, c.quadratic_stddev);
            HashValue(hash, c.incidence_bias);
            HashValue(hash, c.quantization);
            HashValue(hash, c.return_photons);
            HashValue(hash, c.reference_distance);
            HashValue(hash, c.background_photons);
            HashValue(hash, c.precision);
            HashValue(hash, c.minimum_return);
            HashValue(hash, c.dropout_probability);
            HashValue(hash, c.seed);
        }
    }
    for (const auto* stamps : {&state.camera_stamps, &state.lidar_stamps}) {
        HashValue(hash, uint64_t{stamps->size()});
        for (const auto& stamp : *stamps) {
            HashValue(hash, stamp.acquisitions);
            HashValue(hash, stamp.sample_time);
            HashValue(hash, stamp.valid);
        }
    }
    for (double time : state.sample_times) HashValue(hash, time);
    HashValue(hash, snapshot.aov_mask);
    HashValue(hash, snapshot.width);
    HashValue(hash, snapshot.height);
    HashValue(hash, snapshot.lidar_az);
    HashValue(hash, snapshot.lidar_el);
    for (const auto* values : {&snapshot.color, &snapshot.depth, &snapshot.normal, &snapshot.albedo, &snapshot.range})
        HashFloatVector(hash, *values);
    HashValue(hash, uint64_t{snapshot.prim.size()});
    HashBytes(hash, snapshot.prim.data(), snapshot.prim.size() * sizeof(uint32_t));
    const auto& f = snapshot.fidelity;
    HashValue(hash, f.spp);
    HashValue(hash, f.shadow_samples);
    HashValue(hash, f.sun_angular_radius);
    HashValue(hash, f.ao_enabled);
    HashValue(hash, f.ao_samples);
    HashValue(hash, f.ao_radius);
    HashValue(hash, f.gi_enabled);
    HashValue(hash, f.tonemap_enabled);
    HashValue(hash, f.srgb_enabled);
    for (const auto& color : {f.sky_top, f.sky_bottom, f.sky_ground, f.fog_color}) {
        HashValue(hash, color.x);
        HashValue(hash, color.y);
        HashValue(hash, color.z);
    }
    HashValue(hash, f.fog_density);
    HashValue(hash, f.sky_intensity);
    HashValue(hash, f.seed);
    const auto& d = snapshot.render_dr;
    HashValue(hash, d.enabled);
    HashValue(hash, d.seed);
    HashValue(hash, d.color_jitter);
    HashValue(hash, d.roughness_jitter);
    HashValue(hash, d.metallic_jitter);
    HashValue(hash, d.light_dir_jitter);
    HashValue(hash, d.light_intensity_jitter);
    HashValue(hash, d.light_color_jitter);
    HashValue(hash, d.ambient_intensity_jitter);
}

phi::Status CopyHostState(const WorldRecord& source, WorldCheckpointRecord* target) {
    const auto sensor_status = source.world->StateSensors().Capture(&target->state_sensors);
    if (sensor_status != phi::Status::Ok) return sensor_status;
    if (source.sensor) {
        target->sensor_revision = source.sensor->revision;
        target->sensor_rendered = source.sensor->rendered;
        target->imaging = source.sensor->backend->CaptureSensorState(source.sensor->handle);
    }
    target->simulated_step_count = source.simulated_step_count;
    target->step_options = source.step_options;
    target->sparse_solver_backend = source.sparse_solver_backend;
    target->link_inertia = source.articulation_host.link_inertia;
    target->joint_armature = source.articulation_host.joint_armature;
    for (const auto& entry : source.field_observations) {
        if (!entry.second->Active()) continue;
        const auto status = entry.second->Capture(&target->observations[entry.first]);
        if (status != phi::Status::Ok) return status;
    }
    target->dr_config = source.dr_config;
    target->dr_baseline_captured = source.dr_baseline_captured;
    target->dr_nominal_link_mass = source.dr_nominal_link_mass;
    target->dr_nominal_joint_armature = source.dr_nominal_joint_armature;
    target->dr_nominal_gravity_z = source.dr_nominal_gravity_z;
    target->dr_nominal_friction = source.dr_nominal_friction;
    return phi::Status::Ok;
}

void RestoreHostState(const WorldCheckpointRecord& source, WorldRecord* target) {
    target->simulated_step_count = source.simulated_step_count;
    target->step_options = source.step_options;
    target->sparse_solver_backend = source.sparse_solver_backend;
    target->dr_config = source.dr_config;
    target->dr_baseline_captured = source.dr_baseline_captured;
    target->dr_nominal_link_mass = source.dr_nominal_link_mass;
    target->dr_nominal_joint_armature = source.dr_nominal_joint_armature;
    target->dr_nominal_gravity_z = source.dr_nominal_gravity_z;
    target->dr_nominal_friction = source.dr_nominal_friction;
    target->articulation_host.link_inertia = source.link_inertia;
    target->articulation_host.joint_armature = source.joint_armature;
    target->last_invariant_violations.clear();
    target->invariant_sampler.Reset();
    if (target->sensor) {
        auto& s = *target->sensor;
        s.backend->RestoreSensorState(s.handle, source.imaging);
        s.rendered = source.sensor_rendered;
        s.fidelity = source.imaging.fidelity;
        s.render_dr = source.imaging.render_dr;
        s.aov_mask = source.imaging.aov_mask;
        s.camera_responses = source.imaging.imaging.cameras;
        s.depth_responses = source.imaging.imaging.depths;
        s.lidar_responses = source.imaging.imaging.lidars;
    }
}

phi::Status HashHostState(uint64_t* hash, const WorldRecord& record) {
    HashValue(hash, record.env_count);
    HashValue(hash, record.simulated_step_count);
    HashValue(hash, record.sensor ? record.sensor->revision : uint64_t{0u});
    if (record.sensor) {
        HashValue(hash, record.sensor->rendered);
        HashImagingState(hash, record.sensor->backend->CaptureSensorState(record.sensor->handle));
    }
    sensor::StateSensorBankSnapshot state_sensors;
    const auto sensor_status = record.world->StateSensors().Capture(&state_sensors);
    if (sensor_status != phi::Status::Ok) return sensor_status;
    HashValue(hash, state_sensors.step);
    for (uint64_t reset : state_sensors.reset_steps) HashValue(hash, reset);
    uint64_t sensor_count = 0u;
    for (const auto& sensor : state_sensors.sensors) if (sensor.active) ++sensor_count;
    HashValue(hash, sensor_count);
    if (sensor_count) for (double time : state_sensors.times) HashValue(hash, time);
    for (uint32_t id = 0u; id < state_sensors.sensors.size(); ++id) {
        const auto& sensor = state_sensors.sensors[id];
        if (!sensor.active) continue;
        HashValue(hash, id);
        const auto& d = sensor.desc;
        HashValue(hash, static_cast<uint32_t>(d.kind));
        HashValue(hash, static_cast<uint32_t>(d.mount));
        HashValue(hash, d.index);
        HashValue(hash, d.update_period);
        HashBytes(hash, &d.local_offset, sizeof(d.local_offset));
        HashValue(hash, d.sample_period);
        HashValue(hash, d.latency);
        HashValue(hash, d.latency_jitter);
        HashValue(hash, d.dropout_probability);
        HashValue(hash, d.temperature);
        HashValue(hash, d.seed);
        for (const auto& error : d.errors) HashObservationConfig(hash, error);
        if (sensor::IsContactRegionSensor(d.kind)) {
            const auto& tactile = d.tactile;
            HashValue(hash, static_cast<uint32_t>(tactile.shape));
            HashValue(hash, tactile.size.x);
            HashValue(hash, tactile.size.y);
            HashValue(hash, tactile.size.z);
            HashValue(hash, tactile.spread_fraction);
            HashValue(hash, tactile.spread_sigma);
            HashValue(hash, tactile.hysteresis_strength);
            HashValue(hash, tactile.hysteresis_time);
        }
        HashBytes(hash, sensor.bytes.data(), sensor.bytes.size());
    }
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
    uint64_t observation_count = 0u;
    for (const auto& entry : record.field_observations) if (entry.second->Active()) ++observation_count;
    HashValue(hash, observation_count);
    for (const auto& entry : record.field_observations) {
        if (!entry.second->Active()) continue;
        sensor::ObservationSnapshot snapshot;
        const auto status = entry.second->Capture(&snapshot);
        if (status != phi::Status::Ok) return status;
        HashValue(hash, static_cast<uint32_t>(entry.first));
        const uint32_t kind = static_cast<uint32_t>(snapshot.config.noise.kind);
        HashValue(hash, kind);
        HashValue(hash, snapshot.config.noise.param1);
        HashValue(hash, snapshot.config.noise.param2);
        HashValue(hash, snapshot.config.noise.seed);
        const auto& e = snapshot.config.error;
        const float parameters[] = {e.bias, e.scale_error, e.noise_density, e.initial_bias_stddev,
            e.bias_random_walk, e.correlated_bias_stddev, e.correlation_time, e.quantization,
            e.minimum, e.maximum, e.response_time, e.temperature_coefficient, e.reference_temperature};
        HashBytes(hash, parameters, sizeof(parameters));
        HashValue(hash, e.saturation_enabled);
        HashValue(hash, snapshot.env_count);
        HashValue(hash, snapshot.values_per_env);
        HashBytes(hash, snapshot.bytes.data(), snapshot.bytes.size());
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
    HashFloatVector(hash, record.articulation_host.joint_armature);
    for (const auto& inertia : record.articulation_host.link_inertia) {
        HashBytes(hash, &inertia, sizeof(inertia));
    }
    return phi::Status::Ok;
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
        const auto status = nuka::c_abi::CopyHostState(*record, checkpoint.get());
        if (status != nuka::phi::Status::Ok) return nuka::c_abi::MapStatusToResult(status);
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
        if ((record->sensor ? record->sensor->revision : 0u) != saved->sensor_revision)
            return NUKA_RESULT_INVALID_ARG;
        if (!record->world->StateSensors().Compatible(saved->state_sensors)) return NUKA_RESULT_INVALID_ARG;
        for (const auto& entry : saved->observations) {
            const auto found = record->field_observations.find(entry.first);
            if (found == record->field_observations.end() || !found->second->Compatible(entry.second))
                return NUKA_RESULT_INVALID_ARG;
        }
        if (!record->world->GetData().UploadPersistent(saved->persistent)) {
            return NUKA_RESULT_INVALID_ARG;
        }
        nuka::c_abi::RestoreHostState(*saved, record);
        for (auto& entry : record->field_observations) {
            const auto found = saved->observations.find(entry.first);
            const auto status = found == saved->observations.end() ? entry.second->Deactivate() :
                entry.second->Restore(found->second);
            if (status != nuka::phi::Status::Ok) return nuka::c_abi::MapStatusToResult(status);
        }
        return nuka::c_abi::MapStatusToResult(record->world->RestoreStateSensors(saved->state_sensors));
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
        constexpr char domain[] = "NukaStateHashV5";
        nuka::c_abi::HashBytes(&hash, domain, sizeof(domain));
        const auto status = nuka::c_abi::HashHostState(&hash, *record);
        if (status != nuka::phi::Status::Ok) return nuka::c_abi::MapStatusToResult(status);
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
