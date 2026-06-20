#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- per-env render domain-randomization config for the batched sensor
// render. Each enabled axis randomizes one appearance term per env (material
// base_color / roughness / metallic; light direction / intensity / color;
// ambient intensity) as a PURE function of (seed, env, axis) via the same
// counter-based Philox the per-episode DR uses -- no mutable RNG state, so the
// same seed yields the same bytes (RL-reproducible).
//
// DEFAULT IS DISABLED (`enabled == false`): the per-env tables are then exact
// replicas of the base (env 0) set, so the batched render's cross-env tiles stay
// byte-identical -- there is ONE trace path that always reads BY ENV; DR off is a
// table of replicas, not a separate code path. CUDA-free POD (host + device).
// ---------------------------------------------------------------------------

#include <cstdint>

namespace nuka::rt {

// A symmetric jitter band: each randomized scalar is base + u*range with u in
// [-1, 1] (additive) or base * (1 + u*range) (multiplicative). A zero range
// leaves that scalar at its base value (an unset axis is a byte no-op).
struct RenderDrConfig {
    // Material axes (per material slot, per env). color_jitter adds to each RGB
    // channel; roughness/metallic_jitter add to the scalar; all clamped to valid.
    float color_jitter = 0.0f;      // +/- per RGB channel of base_color
    float roughness_jitter = 0.0f;  // +/- added to roughness, clamped (0,1]
    float metallic_jitter = 0.0f;   // +/- added to metallic, clamped [0,1]

    // Light axes (per env). dir_jitter perturbs each direction component then the
    // vector renormalizes; intensity/color are multiplicative about the base.
    float light_dir_jitter = 0.0f;        // +/- per direction component
    float light_intensity_jitter = 0.0f;  // fractional, base*(1 +/- this)
    float light_color_jitter = 0.0f;      // fractional per RGB, base*(1 +/- this)

    // Ambient axis (per env): multiplicative about the base ambient color.
    float ambient_intensity_jitter = 0.0f;  // fractional, base*(1 +/- this)

    bool enabled = false;  // OFF -> per-env tables are exact base replicas
    uint64_t seed = 0u;    // Philox key; the recorded deterministic state
};

// Stable per-axis Philox lanes. Each axis (and material RGB component) draws on a
// distinct counter lane so the draws are independent yet reproducible. NEVER
// reorder without bumping a seed convention -- these are the replay coordinates.
enum class RenderDrAxis : uint32_t {
    MaterialColorR = 0u,
    MaterialColorG = 1u,
    MaterialColorB = 2u,
    MaterialRoughness = 3u,
    MaterialMetallic = 4u,
    LightDirX = 5u,
    LightDirY = 6u,
    LightDirZ = 7u,
    LightIntensity = 8u,
    LightColorR = 9u,
    LightColorG = 10u,
    LightColorB = 11u,
    AmbientIntensity = 12u,
};

}  // namespace nuka::rt
