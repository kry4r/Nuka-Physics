#pragma once

#include "sensor/imaging.hpp"
#include "sensor/noise/philox.cuh"

#if defined(__CUDACC__)
#define NUKA_IMAGE_HD __host__ __device__ inline
#else
#define NUKA_IMAGE_HD inline
#endif

namespace nuka::sensor::noise {

enum class ImagingStream : uint32_t {
    PixelGain, PixelOffset, PixelFailure, PhotonCount, ReadNoise, RowNoise,
    ReturnCount, BackgroundCount, RangeNoise, Dropout
};

NUKA_IMAGE_HD uint64_t ImagingKey(uint64_t seed, uint32_t env, uint32_t sensor,
    uint32_t channel, ImagingStream stream) {
    const auto value = Philox4x32_10({{env, sensor, channel, static_cast<uint32_t>(stream)}}, SplitSeed(seed));
    return uint64_t{value.v[0]} | (uint64_t{value.v[1]} << 32u);
}

struct CameraSampleKeys {
    uint64_t gain[3], offset[3], photon[3], read[3];
    uint64_t failure, row;
};

struct RangeSampleKeys {
    uint64_t dropout, photon, background, distance;
};

NUKA_IMAGE_HD CameraSampleKeys MakeCameraSampleKeys(uint64_t seed, uint32_t env, uint32_t sensor) {
    CameraSampleKeys keys;
    for (uint32_t c = 0u; c < 3u; ++c) {
        keys.gain[c] = ImagingKey(seed, env, sensor, c, ImagingStream::PixelGain);
        keys.offset[c] = ImagingKey(seed, env, sensor, c, ImagingStream::PixelOffset);
        keys.photon[c] = ImagingKey(seed, env, sensor, c, ImagingStream::PhotonCount);
        keys.read[c] = ImagingKey(seed, env, sensor, c, ImagingStream::ReadNoise);
    }
    keys.failure = ImagingKey(seed, env, sensor, 0u, ImagingStream::PixelFailure);
    keys.row = ImagingKey(seed, env, sensor, 0u, ImagingStream::RowNoise);
    return keys;
}

NUKA_IMAGE_HD RangeSampleKeys MakeRangeSampleKeys(uint64_t seed, uint32_t env,
    uint32_t sensor, uint32_t channel) {
    return {ImagingKey(seed, env, sensor, channel, ImagingStream::Dropout),
        ImagingKey(seed, env, sensor, channel, ImagingStream::ReturnCount),
        ImagingKey(seed, env, sensor, channel, ImagingStream::BackgroundCount),
        ImagingKey(seed, env, sensor, channel, ImagingStream::RangeNoise)};
}

NUKA_IMAGE_HD double PhotonCount(uint64_t seed, uint32_t element, uint64_t sequence, double mean) {
    if (!(mean > 0.0)) return 0.0;
    if (mean <= 1.0e8) return PoissonSample(seed, element, sequence, static_cast<float>(mean));
    // Above 1e8 expected electrons, the normal limit avoids an unbounded count representation.
    return fmax(0.0, mean + sqrt(mean) * NormalSample(seed, element, sequence));
}

NUKA_IMAGE_HD float ImagingUniform(uint64_t seed, uint32_t element, uint64_t sequence) {
    return Uint32ToUniform01(Philox4x32_10(MakeCounter(element, sequence), SplitSeed(seed)).v[0]);
}

NUKA_IMAGE_HD float FormCameraChannel(float radiance, const CameraResponse& c,
    const CameraSampleKeys& keys, uint32_t pixel, uint32_t row, uint32_t component, uint64_t sequence) {
    if (!c.enabled) return radiance;
    const float failure = c.dead_pixel_probability + c.hot_pixel_probability > 0.0f
        ? ImagingUniform(keys.failure, pixel, 0u) : 1.0f;
    const bool dead = failure <= c.dead_pixel_probability;
    const bool hot = !dead && failure <= c.dead_pixel_probability + c.hot_pixel_probability;
    const double gain = c.pixel_gain_stddev > 0.0f ? fmax(0.0, 1.0 + double{c.pixel_gain_stddev} *
        NormalSample(keys.gain[component], pixel, 0u)) : 1.0;
    const double exponent = c.dark_doubling_temperature > 0.0f
        ? (double{c.temperature} - c.reference_temperature) / c.dark_doubling_temperature : 0.0;
    const double dark = (double{c.dark_current} + (hot ? c.hot_pixel_current : 0.0f)) * exp2(exponent);
    const double signal = dead ? 0.0 : fmax(0.0, double{radiance}) * c.electrons_per_unit_second * gain;
    const double mean = (signal + dark) * c.exposure_time;
    double charge = c.shot_noise
        ? PhotonCount(keys.photon[component], pixel, sequence, mean) : mean;
    charge = fmin(charge, double{c.full_well_electrons});
    if (c.pixel_offset_stddev_electrons > 0.0f)
        charge += double{c.pixel_offset_stddev_electrons} * NormalSample(keys.offset[component], pixel, 0u);
    if (c.read_noise_electrons > 0.0f)
        charge += double{c.read_noise_electrons} * NormalSample(keys.read[component], pixel, sequence);
    if (c.row_noise_electrons > 0.0f)
        charge += double{c.row_noise_electrons} * NormalSample(keys.row, row, sequence);
    double normalized = (charge * c.analog_gain + c.black_level_electrons) / c.full_well_electrons;
    normalized = fmin(1.0, fmax(0.0, normalized));
    if (c.adc_bits) {
        const double levels = static_cast<double>((uint32_t{1u} << c.adc_bits) - 1u);
        normalized = floor(normalized * levels + 0.5) / levels;
    }
    return static_cast<float>(normalized);
}

NUKA_IMAGE_HD float FormRange(float distance, float incidence, float reflectance,
    float minimum, float maximum, float miss, const RangeResponse& c,
    const RangeSampleKeys& keys, uint32_t element, uint64_t sequence) {
    if (!c.enabled) return distance;
    if (!(distance >= minimum && distance <= maximum)) return miss;
    if (c.dropout_probability > 0.0f &&
        ImagingUniform(keys.dropout, element, sequence) <= c.dropout_probability) return miss;
    const double cosine = fmin(1.0, fmax(0.0, double{incidence}));
    double variance = double{c.distance_stddev} * c.distance_stddev;
    const double quadratic = double{c.quadratic_stddev} * distance * distance;
    variance += quadratic * quadratic;
    if (c.return_photons > 0.0f) {
        if (!(distance > 0.0f)) return miss;
        const double ratio = double{c.reference_distance} / distance;
        const double mean = double{c.return_photons} * fmax(0.0, double{reflectance}) * cosine * ratio * ratio;
        const double count = PhotonCount(keys.photon, element, sequence, mean);
        if (count < c.minimum_return) return miss;
        const double background = PhotonCount(keys.background, element, sequence, c.background_photons);
        variance += double{c.precision} * c.precision * (count + background) / (count * count);
    }
    double measured = double{distance} * (1.0 + c.scale_error) + c.bias + c.incidence_bias * (1.0 - cosine);
    if (variance > 0.0) measured += sqrt(variance) * NormalSample(keys.distance, element, sequence);
    if (c.quantization > 0.0f) measured = floor(measured / c.quantization + 0.5) * c.quantization;
    return measured >= minimum && measured <= maximum ? static_cast<float>(measured) : miss;
}

}  // namespace nuka::sensor::noise

#undef NUKA_IMAGE_HD
