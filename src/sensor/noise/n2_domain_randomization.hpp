#pragma once
// ---------------------------------------------------------------------------
// n2_domain_randomization.hpp -- per-episode domain randomization sampling
// (v0.5 p04 Task 5.4.7 sim-to-real N2)
// ---------------------------------------------------------------------------
//
// SAMPLING IS COUNTER-BASED + PURE (philox-only dependency; no WorldRecord, no
// runtime). A per-env multiplier / offset is a PURE FUNCTION of
//   (config.seed, env_idx, param_idx)
// via the HOST path of the N1 Philox4x32-10 generator: distinct param lanes give
// independent yet reproducible draws. There is NO mutable RNG state and NO float
// atomics -- the deterministic seed IS the recorded state, so the diff-sim
// reverse/replay pass re-derives identical values and the backward stays D1
// byte-exact two-run (the advisor-flagged headline: "N2 checkpoint writes must
// NOT perturb backward D1").
//
// This header is sampling-only so it stays unit-testable in isolation (it links
// against nothing but philox.cuh). The buffer-touching half --
// ApplyPerEpisodeRandomization -- lives in c_abi/noise.cpp where the WorldRecord
// + runtime (MakeSpatialInertia, the live device buffers) are already linked.
// ---------------------------------------------------------------------------

#include "sensor/noise/noise_config.hpp"
#include "sensor/noise/philox.cuh"

#include <cstdint>

namespace nuka::sensor::noise {

// The result of sampling one env's per-episode randomization. mass / friction
// are MULTIPLIERS (nominal * mult); restitution / armature / gravity are
// OFFSETS (nominal + offset). Snapshotting against a NOMINAL baseline (not the
// live value) is what makes a repeated apply idempotent -- re-randomizing around
// nominal every reset instead of a compounding random walk.
struct SampledRandomization {
    float mass_multiplier = 1.0f;
    float friction_multiplier = 1.0f;
    float restitution_offset = 0.0f;
    float joint_armature_offset = 0.0f;
    float gravity_z_offset = 0.0f;
};

// Stable param-lane indices. Each occupies a distinct Philox counter lane (the
// `seq` argument), so the draws are independent across params yet reproducible.
// NEVER reorder/renumber these without bumping a seed convention: they are the
// counter coordinates that make replay byte-exact.
enum class DomainRandomizationParam : uint32_t {
    Mass = 0u,
    Friction = 1u,
    Restitution = 2u,
    Armature = 3u,
    GravityZ = 4u,
};

// One uniform draw in [lo, hi], a pure function of (seed, env_idx, param). Uses
// the raw Philox output word (lane 0) mapped through Uint32ToUniform01 in (0,1]
// -- NOT NormalSample: we want a bounded uniform, and lo + u*(hi-lo) is
// guaranteed within [lo, hi] for u in (0,1] (and degenerates to lo==hi when the
// range is a point, e.g. an unset/zero range).
inline float SampleUniformInRange(uint64_t seed, uint32_t env_idx,
                                  DomainRandomizationParam param,
                                  const DomainRandomizationConfig::Range& range) {
    const Philox4x32Counter out = Philox4x32_10(
        MakeCounter(env_idx, static_cast<uint64_t>(param)), SplitSeed(seed));
    const float u = Uint32ToUniform01(out.v[0]);  // in (0, 1]
    return range.lo + u * (range.hi - range.lo);
}

// Samples ALL params for one env. Multipliers default to 1.0 / offsets to 0.0
// when DR is disabled (so a disabled apply is an exact no-op without sampling).
inline SampledRandomization SampleEpisodeRandomization(
    const DomainRandomizationConfig& cfg, uint32_t env_idx) {
    SampledRandomization s;
    if (!cfg.enabled) {
        return s;  // identity: mult==1, offset==0
    }
    s.mass_multiplier = SampleUniformInRange(
        cfg.seed, env_idx, DomainRandomizationParam::Mass, cfg.mass_multiplier);
    s.friction_multiplier =
        SampleUniformInRange(cfg.seed, env_idx, DomainRandomizationParam::Friction,
                             cfg.friction_multiplier);
    s.restitution_offset =
        SampleUniformInRange(cfg.seed, env_idx,
                             DomainRandomizationParam::Restitution,
                             cfg.restitution_offset);
    s.joint_armature_offset =
        SampleUniformInRange(cfg.seed, env_idx,
                             DomainRandomizationParam::Armature,
                             cfg.joint_armature_offset);
    s.gravity_z_offset =
        SampleUniformInRange(cfg.seed, env_idx,
                             DomainRandomizationParam::GravityZ,
                             cfg.gravity_z_offset);
    return s;
}

}  // namespace nuka::sensor::noise
