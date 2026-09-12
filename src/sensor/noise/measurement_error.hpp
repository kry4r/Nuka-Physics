#pragma once

#include <cmath>

#include "sensor/noise/philox.cuh"
#include "sensor/observation_types.hpp"

#if defined(__CUDACC__)
#define NUKA_MEASUREMENT_HD __host__ __device__
#else
#define NUKA_MEASUREMENT_HD
#endif

namespace nuka::sensor::noise {

NUKA_MEASUREMENT_HD inline float TaggedNormal(uint64_t seed, uint32_t element,
                                              uint64_t sequence, uint32_t cause) {
    auto counter = MakeCounter(element, sequence);
    counter.v[3] = cause;
    const auto random = Philox4x32_10(counter, SplitSeed(seed));
    return sqrtf(-2.0f * logf(Uint32ToUniform01(random.v[0]))) *
        cosf(6.28318530717958648f * Uint32ToUniform01(random.v[1]));
}

// Density is the square root of two-sided power spectral density for a boxcar sample.
// Bias diffusion and the stationary Ornstein-Uhlenbeck process use exact interval variances.
NUKA_MEASUREMENT_HD inline float MeasureValue(float truth, const ObservationConfig& config,
    ObservationNoiseState* state, uint32_t element, uint32_t channel, uint64_t sequence,
    double interval, float temperature) {
    const uint64_t seed = config.noise.seed ^ (0x9e3779b97f4a7c15ull * (uint64_t{channel} + 1u));
    const auto& e = config.error;
    if (sequence == 0u) {
        state->fixed_bias = e.initial_bias_stddev > 0.0f ?
            e.initial_bias_stddev * TaggedNormal(seed, element, 0u, 1u) : 0.0f;
        state->correlated_bias = e.correlated_bias_stddev > 0.0f ?
            e.correlated_bias_stddev * TaggedNormal(seed, element, 0u, 2u) : 0.0f;
        state->random_walk = 0.0f;
        state->filtered_value = truth;
    }
    if (e.response_time > 0.0f) {
        const float amount = static_cast<float>(-expm1(-interval / e.response_time));
        state->filtered_value += amount * (truth - state->filtered_value);
    } else {
        state->filtered_value = truth;
    }
    if (e.bias_random_walk > 0.0f)
        state->random_walk += static_cast<float>(e.bias_random_walk * sqrt(interval)) *
            TaggedNormal(seed, element, sequence, 3u);
    if (e.correlated_bias_stddev > 0.0f && sequence != 0u) {
        const double decay = exp(-interval / e.correlation_time);
        const double variance = -expm1(-2.0 * interval / e.correlation_time);
        state->correlated_bias = static_cast<float>(decay * state->correlated_bias +
            e.correlated_bias_stddev * sqrt(variance) * TaggedNormal(seed, element, sequence, 4u));
    }
    float value = state->filtered_value;
    if (e.scale_error != 0.0f) value *= 1.0f + e.scale_error;
    const float bias = e.bias + state->fixed_bias + state->random_walk + state->correlated_bias +
        e.temperature_coefficient * (temperature - e.reference_temperature);
    if (bias != 0.0f) value += bias;
    if (e.noise_density > 0.0f)
        value += static_cast<float>(e.noise_density / sqrt(interval)) *
            TaggedNormal(seed, element, sequence, 5u);
    switch (config.noise.kind) {
        case NoiseKind::Gaussian:
            value += config.noise.param1 + config.noise.param2 *
                TaggedNormal(seed, element, sequence, 6u);
            break;
        case NoiseKind::Poisson:
            value += static_cast<float>(PoissonSample(seed ^ 0xd2b74407b1ce6e93ull,
                element, sequence, config.noise.param1));
            break;
        case NoiseKind::None: break;
    }
    if (e.quantization > 0.0f) value = static_cast<float>(round(double{value} / e.quantization) * e.quantization);
    if (e.saturation_enabled && !std::isnan(value)) value = fminf(e.maximum, fmaxf(e.minimum, value));
    return value;
}

}  // namespace nuka::sensor::noise

#undef NUKA_MEASUREMENT_HD
