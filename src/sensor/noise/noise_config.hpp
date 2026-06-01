#pragma once
// ---------------------------------------------------------------------------
// noise_config.hpp -- sensor-noise kind + descriptor (v0.5 p04 N1)
// ---------------------------------------------------------------------------
//
// Internal (C++ side) noise configuration. The C-ABI mirror lives in
// src/include/nuka/nuka_noise.h (nuka_sensor_noise_desc_t). Default is None so
// V1 oracle scenes are byte-identical unless noise is explicitly enabled.
// ---------------------------------------------------------------------------

#include <cstdint>

namespace nuka::sensor::noise {

enum class NoiseKind : uint32_t {
    None = 0,
    Gaussian = 1,
    Poisson = 2,
};

struct SensorNoiseConfig {
    NoiseKind kind = NoiseKind::None;
    float param1 = 0.0f;     // Gaussian: mean        / Poisson: lambda
    float param2 = 0.0f;     // Gaussian: stddev      / (unused for Poisson)
    uint64_t seed = 0u;      // RNG seed (key)
};

}  // namespace nuka::sensor::noise
