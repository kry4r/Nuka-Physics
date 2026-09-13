#include "nuka/nuka_state_sensor.h"

#include <cmath>
#include <exception>
#include <new>

#include "c_abi/handle_table.hpp"
#include "c_abi/dlpack_table.hpp"
#include "c_abi/internal.hpp"
#include "c_abi/measurement_error.hpp"
#include "nk/pipeline/world.hpp"

namespace {

template <class Function>
nuka_result_t WithSensorWorld(nuka_world_handle world, Function function) {
    auto* record = nuka::c_abi::WorldTable().Get(world);
    if (!record) return NUKA_RESULT_NULL_HANDLE;
    if (!record->world) return NUKA_RESULT_NOT_SUPPORTED;
    try { return function(*record->world);
    } catch (const std::bad_alloc&) { return NUKA_RESULT_OUT_OF_MEMORY;
    } catch (const std::exception& error) { return nuka::c_abi::MapExceptionToResult(error);
    } catch (...) { return NUKA_RESULT_INTERNAL; }
}

}  // namespace

extern "C" {

nuka_result_t nuka_world_attach_state_sensor(nuka_world_handle world,
    const nuka_state_sensor_desc_t* desc, uint32_t* out_sensor) {
    if (!desc || !out_sensor || desc->struct_size < sizeof(*desc) ||
        !std::isfinite(desc->sample_rate_hz) || desc->sample_rate_hz < 0.0) return NUKA_RESULT_INVALID_ARG;
    *out_sensor = ~0u;
    return WithSensorWorld(world, [&](nuka::nk::World& target) {
        nuka::sensor::StateSensorDesc config;
        config.kind = static_cast<nuka::sensor::StateSensorKind>(desc->kind);
        config.mount = static_cast<nuka::sensor::StateSensorMount>(desc->mount);
        config.index = desc->mount_index;
        config.local_offset.position = {desc->local_offset[0], desc->local_offset[1], desc->local_offset[2]};
        config.local_offset.rotation = {desc->local_offset[3], desc->local_offset[4], desc->local_offset[5], desc->local_offset[6]};
        config.sample_period = desc->sample_rate_hz > 0.0 ? 1.0 / desc->sample_rate_hz : 0.0;
        config.update_period = desc->update_period;
        config.latency = desc->latency;
        config.latency_jitter = desc->latency_jitter;
        config.dropout_probability = desc->dropout_probability;
        config.temperature = desc->temperature;
        config.seed = desc->seed;
        return nuka::c_abi::MapStatusToResult(target.AttachStateSensor(config, out_sensor));
    });
}

nuka_result_t nuka_world_get_state_sensor_count(nuka_world_handle world, uint32_t* out_count) {
    if (!out_count) return NUKA_RESULT_INVALID_ARG;
    *out_count = 0u;
    return WithSensorWorld(world, [&](nuka::nk::World& target) {
        *out_count = target.StateSensors().Count();
        return NUKA_RESULT_OK;
    });
}

nuka_result_t nuka_world_get_state_sensor_active(nuka_world_handle world, uint32_t sensor, uint32_t* out_active) {
    if (!out_active) return NUKA_RESULT_INVALID_ARG;
    *out_active = 0u;
    return WithSensorWorld(world, [&](nuka::nk::World& target) {
        if (sensor >= target.StateSensors().Count()) return NUKA_RESULT_INVALID_ARG;
        *out_active = target.StateSensors().Active(sensor) ? 1u : 0u;
        return NUKA_RESULT_OK;
    });
}

nuka_result_t nuka_world_get_state_sensor_dims(nuka_world_handle world, uint32_t sensor,
    uint32_t* out_envs, uint32_t* out_values) {
    if (!out_envs || !out_values) return NUKA_RESULT_INVALID_ARG;
    *out_envs = *out_values = 0u;
    return WithSensorWorld(world, [&](nuka::nk::World& target) {
        if (!target.StateSensors().Active(sensor)) return NUKA_RESULT_INVALID_ARG;
        *out_envs = target.EnvCount();
        *out_values = target.StateSensors().ValueCount(sensor);
        return NUKA_RESULT_OK;
    });
}

nuka_result_t nuka_world_set_state_sensor_error(nuka_world_handle world, uint32_t sensor,
    uint32_t component, const nuka_sensor_error_desc_t* desc) {
    if (desc && desc->struct_size < sizeof(*desc)) return NUKA_RESULT_INVALID_ARG;
    return WithSensorWorld(world, [&](nuka::nk::World& target) {
        return nuka::c_abi::MapStatusToResult(target.ConfigureStateSensorError(
            sensor, component, nuka::c_abi::MeasurementErrorConfig(desc)));
    });
}

nuka_result_t nuka_world_get_state_sensor_view(nuka_world_handle world, uint32_t sensor, nuka_buffer_view_t* out) {
    if (!out) return NUKA_RESULT_INVALID_ARG;
    *out = {};
    return WithSensorWorld(world, [&](nuka::nk::World& target) {
        if (!target.StateSensors().Active(sensor)) return NUKA_RESULT_INVALID_ARG;
        out->device_ptr = target.StateSensors().Values(sensor);
        out->element_count = uint64_t{target.EnvCount()} * target.StateSensors().ValueCount(sensor);
        out->element_stride_bytes = sizeof(float);
        out->dtype = nuka::c_abi::kWireDtypeF32;
        return NUKA_RESULT_OK;
    });
}

nuka_result_t nuka_world_download_state_sensor(nuka_world_handle world, uint32_t sensor,
    void* bytes, size_t nbytes, size_t byte_offset) {
    return WithSensorWorld(world, [&](nuka::nk::World& target) {
        return nuka::c_abi::MapStatusToResult(target.StateSensors().Download(sensor, bytes, nbytes, byte_offset));
    });
}

nuka_result_t nuka_world_get_state_sensor_stamp(nuka_world_handle world, uint32_t sensor,
    uint32_t env, nuka_state_sensor_stamp_t* out) {
    if (!out) return NUKA_RESULT_INVALID_ARG;
    *out = {};
    return WithSensorWorld(world, [&](nuka::nk::World& target) {
        nuka::sensor::StateSensorStamp stamp;
        const auto status = target.StateSensors().ReadStamp(sensor, env, &stamp);
        if (status != nuka::phi::Status::Ok) return nuka::c_abi::MapStatusToResult(status);
        out->sequence = stamp.sequence;
        out->acquisitions = stamp.acquisitions;
        out->dropped = stamp.dropped;
        out->sample_time = stamp.sample_time;
        out->delivery_time = stamp.delivery_time;
        out->valid = stamp.valid;
        return NUKA_RESULT_OK;
    });
}

}  // extern "C"
