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

// ---------------------------------------------------------------------------
// N2 per-episode domain randomization (v0.5 p04 Task 5.4.7).
//
// Each [lo, hi] range is sampled ONCE per env per episode-reset to a per-env
// multiplier / offset that is a PURE FUNCTION of (seed, env_idx, param_idx) via
// the host Philox path (see n2_domain_randomization.hpp). There is no mutable
// sampled state to checkpoint -- the deterministic seed IS the recorded state,
// so the reverse/replay pass re-derives identical values and the diff-sim
// backward stays D1 byte-exact two-run (advisor: "N2 checkpoint writes must NOT
// perturb backward D1").
//
// DEFAULT IS DISABLED (`enabled == false`): a world with DR off is byte-
// unchanged by ApplyPerEpisodeRandomization, so V1 oracle scenes stay byte-
// identical unless DR is explicitly enabled.
struct DomainRandomizationConfig {
    struct Range {
        float lo = 0.0f;
        float hi = 0.0f;
    };
    // Spec Task 5.4.7 defaults.
    Range mass_multiplier{0.8f, 1.2f};         // scales each link's mass
    Range friction_multiplier{0.5f, 1.5f};     // scales contact friction
    Range restitution_offset{-0.1f, 0.1f};     // adds to contact restitution
    Range joint_armature_offset{0.0f, 0.05f};  // adds to per-DOF armature
    Range gravity_z_offset{-0.5f, 0.5f};       // adds to world gravity.z (m/s^2)

    bool enabled = false;  // OFF by default -> apply is a byte no-op (oracle safe)
    uint64_t seed = 0u;    // Philox key; the recorded deterministic state
};

}  // namespace nuka::sensor::noise
