#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- opt-in shading-fidelity profile for the batched sensor render. The
// batched trace shades flat (Lambert+GGX, 1 light, 1 hard shadow, 1 spp, no AA);
// this profile lifts it to the single-camera beauty look (MSAA spp + soft sun
// shadow + AO + one-bounce GI + procedural sky/fog + tonemap) WITHOUT a second
// tracer -- the SAME BatchedSensorTraceKernel takes the richer shade gated on the
// profile being non-default. TEXTURES are out of scope (a UV-plumbing follow-on).
//
// DEFAULT IS THE CHEAP SHADE (`Enabled()==false`): spp=1, ao/gi off, one hard
// shadow, no tonemap -> the trace takes the EXACT current arithmetic, so the AOV
// bytes are unchanged. The fidelity path is stochastic but DETERMINISTIC: every
// sample seeds from Philox(seed, env, sensor, pixel, sample) (no mutable RNG
// state), so the same seed yields the same bytes run-to-run. CUDA-free POD.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#include <cstdint>

namespace nuka::rt {

// Shading-fidelity knobs for the batched sensor render. Default == today's cheap
// shade (Enabled() false). The sky_* / fog_* defaults mirror rt::BeautyOptions so
// the fidelity miss shader + AO/GI sky-dome ambient match the single-camera look.
struct SensorFidelityConfig {
    uint32_t spp = 1u;             // jittered sub-pixel samples per pixel (MSAA)
    uint32_t shadow_samples = 1u;  // soft-shadow rays across the sun disc (1 = hard)
    float sun_angular_radius = 0.04f;  // sun half-angle (rad) -> penumbra width

    bool ao_enabled = false;       // ambient occlusion / sky-dome ambient bounces
    uint32_t ao_samples = 3u;      // hemisphere rays when ao_enabled
    float ao_radius = 0.6f;        // AO ray max distance (world units)
    bool gi_enabled = false;       // add the one-bounce diffuse GI term on AO rays

    bool tonemap_enabled = false;  // ACES-ish filmic tonemap of the averaged color

    // Procedural sky + height fog for the fidelity miss shader + secondary-ray
    // ambient (the single-camera atmosphere). Unused when Enabled() is false.
    math::Vec3 sky_top{0.55f, 0.62f, 0.72f};     // zenith (up)
    math::Vec3 sky_bottom{0.86f, 0.89f, 0.93f};  // horizon
    math::Vec3 sky_ground{0.34f, 0.35f, 0.37f};  // below-horizon (down) fill
    math::Vec3 fog_color{0.84f, 0.88f, 0.93f};   // height/distance fog tint
    float fog_density = 0.0f;                     // per-metre extinction (0 = off)
    float sky_intensity = 1.0f;                   // scales the sky-dome ambient

    uint64_t seed = 0x9e3779b9u;   // Philox key; the recorded deterministic state

    // True iff any knob lifts the shade above the cheap default. The trace reads
    // this: false -> the exact current 1-spp/1-hard-shadow arithmetic (byte no-op).
    bool Enabled() const {
        return spp > 1u || shadow_samples > 1u || ao_enabled || gi_enabled ||
               tonemap_enabled;
    }
};

}  // namespace nuka::rt
