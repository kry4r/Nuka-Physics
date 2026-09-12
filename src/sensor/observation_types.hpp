#pragma once

#include <cstdint>

#include "sensor/noise/noise_config.hpp"

namespace nuka::sensor {

struct MeasurementError {
    float bias = 0.0f;
    float scale_error = 0.0f;
    float noise_density = 0.0f;
    float initial_bias_stddev = 0.0f;
    float bias_random_walk = 0.0f;
    float correlated_bias_stddev = 0.0f;
    float correlation_time = 0.0f;
    float quantization = 0.0f;
    float minimum = 0.0f;
    float maximum = 0.0f;
    float response_time = 0.0f;
    float temperature_coefficient = 0.0f;
    float reference_temperature = 25.0f;
    uint32_t saturation_enabled = 0u;
};

struct ObservationConfig {
    noise::SensorNoiseConfig noise;
    MeasurementError error;
};

struct ObservationNoiseState {
    float fixed_bias = 0.0f;
    float random_walk = 0.0f;
    float correlated_bias = 0.0f;
    float filtered_value = 0.0f;
};

struct ObservationStamp {
    uint64_t sequence = 0u;
    double elapsed_time = 0.0;
    uint32_t valid = 0u;
    uint32_t reserved = 0u;
};

bool ValidObservationConfig(const ObservationConfig& config);

}  // namespace nuka::sensor
