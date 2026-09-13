#pragma once

#include "nuka/nuka_noise.h"
#include "sensor/observation_types.hpp"

namespace nuka::c_abi {

inline sensor::ObservationConfig MeasurementErrorConfig(const nuka_sensor_error_desc_t* desc) {
    sensor::ObservationConfig config;
    if (!desc) return config;
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
    return config;
}

}  // namespace nuka::c_abi
