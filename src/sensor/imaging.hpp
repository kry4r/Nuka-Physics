#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace nuka::sensor {

struct CameraResponse {
    uint32_t enabled = 0u;
    uint32_t shot_noise = 1u;
    uint32_t adc_bits = 12u;
    float exposure_time = 0.01f;
    float electrons_per_unit_second = 1000000.0f;
    float full_well_electrons = 10000.0f;
    float read_noise_electrons = 0.0f;
    float row_noise_electrons = 0.0f;
    float dark_current = 0.0f;
    float dark_doubling_temperature = 0.0f;
    float temperature = 25.0f;
    float reference_temperature = 25.0f;
    float pixel_gain_stddev = 0.0f;
    float pixel_offset_stddev_electrons = 0.0f;
    float analog_gain = 1.0f;
    float black_level_electrons = 0.0f;
    float dead_pixel_probability = 0.0f;
    float hot_pixel_probability = 0.0f;
    float hot_pixel_current = 0.0f;
    uint64_t seed = 0u;
};

struct RangeResponse {
    uint32_t enabled = 0u;
    float bias = 0.0f;
    float scale_error = 0.0f;
    float distance_stddev = 0.0f;
    float quadratic_stddev = 0.0f;
    float incidence_bias = 0.0f;
    float quantization = 0.0f;
    float return_photons = 0.0f;
    float reference_distance = 1.0f;
    float background_photons = 0.0f;
    float precision = 0.0f;
    uint32_t minimum_return = 1u;
    float dropout_probability = 0.0f;
    uint64_t seed = 0u;
};

struct ImagingStamp {
    uint64_t acquisitions = 0u;
    double sample_time = 0.0;
    uint32_t valid = 0u;
    uint32_t reserved = 0u;
};

struct ImagingState {
    uint32_t env_count = 0u;
    std::vector<CameraResponse> cameras;
    std::vector<RangeResponse> depths;
    std::vector<RangeResponse> lidars;
    std::vector<ImagingStamp> camera_stamps;
    std::vector<ImagingStamp> lidar_stamps;
    std::vector<double> sample_times;
};

inline bool ValidCameraResponse(const CameraResponse& c) {
    const float nonnegative[] = {c.read_noise_electrons, c.row_noise_electrons, c.dark_current,
        c.dark_doubling_temperature, c.pixel_gain_stddev, c.pixel_offset_stddev_electrons,
        c.dead_pixel_probability, c.hot_pixel_probability, c.hot_pixel_current};
    for (float value : nonnegative) if (!std::isfinite(value) || value < 0.0f) return false;
    const float positive[] = {c.exposure_time, c.electrons_per_unit_second,
        c.full_well_electrons, c.analog_gain};
    for (float value : positive) if (!std::isfinite(value) || value <= 0.0f) return false;
    if (c.enabled > 1u || c.shot_noise > 1u || c.adc_bits > 24u ||
        c.dead_pixel_probability + c.hot_pixel_probability > 1.0f ||
        !std::isfinite(c.temperature) || !std::isfinite(c.reference_temperature) ||
        !std::isfinite(c.black_level_electrons)) return false;
    const double exponent = c.dark_doubling_temperature > 0.0f
        ? (double{c.temperature} - c.reference_temperature) / c.dark_doubling_temperature : 0.0;
    const double dark = (double{c.dark_current} + c.hot_pixel_current) * std::exp2(exponent);
    return std::isfinite(dark * c.exposure_time);
}

inline bool ValidRangeResponse(const RangeResponse& c) {
    const float nonnegative[] = {c.distance_stddev, c.quadratic_stddev, c.quantization,
        c.return_photons, c.background_photons, c.precision, c.dropout_probability};
    for (float value : nonnegative) if (!std::isfinite(value) || value < 0.0f) return false;
    return c.enabled <= 1u && std::isfinite(c.bias) && std::isfinite(c.scale_error) &&
        c.scale_error > -1.0f && std::isfinite(c.incidence_bias) &&
        std::isfinite(c.reference_distance) && c.reference_distance > 0.0f &&
        c.dropout_probability <= 1.0f && c.minimum_return > 0u &&
        (c.return_photons > 0.0f || (c.precision == 0.0f && c.background_photons == 0.0f));
}

}  // namespace nuka::sensor
