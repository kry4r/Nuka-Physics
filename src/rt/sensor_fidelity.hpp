#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- default high-quality shading profile for the batched sensor render.
// The sensor uses the same persistent batched trace for textured materials,
// stochastic anti-aliasing, soft shadows, AO/GI, ACES tonemapping, and sRGB output.
// Every field is CUDA-free POD and samples are deterministic for a fixed seed.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#include <cstdint>

namespace nuka::rt {

// Default high-quality sensor shading. The sky_* / fog_* defaults mirror
// rt::BeautyOptions so the miss shader and AO/GI sky-dome ambient match the
// single-camera look.
struct SensorFidelityConfig {
    uint32_t spp = 16u;            // jittered sub-pixel samples per pixel (MSAA)
    uint32_t shadow_samples = 12u; // soft-shadow rays across the sun disc
    float sun_angular_radius = 0.04f;  // sun half-angle (rad) -> penumbra width

    bool ao_enabled = true;        // ambient occlusion / sky-dome ambient bounces
    uint32_t ao_samples = 8u;      // hemisphere rays when ao_enabled
    float ao_radius = 0.6f;         // AO ray max distance (world units)
    bool gi_enabled = true;        // add the one-bounce diffuse GI term on AO rays

    bool tonemap_enabled = true;   // ACES-ish filmic tonemap of averaged color
    bool srgb_enabled = true;      // encode linear color to the RGB sensor contract
    uint32_t transmit_bounces = 2u;
    bool smooth_normals = false;

    // Procedural sky + height fog for the miss shader + secondary-ray ambient.
    math::Vec3 sky_top{0.55f, 0.62f, 0.72f};     // zenith (up)
    math::Vec3 sky_bottom{0.86f, 0.89f, 0.93f};  // horizon
    math::Vec3 sky_ground{0.34f, 0.35f, 0.37f};  // below-horizon (down) fill
    math::Vec3 fog_color{0.84f, 0.88f, 0.93f};   // height/distance fog tint
    float fog_density = 0.0f;                     // per-metre extinction (0 = off)
    float sky_intensity = 1.0f;                   // scales the sky-dome ambient

    uint64_t seed = 0x9e3779b9u;   // Philox key; the recorded deterministic state

};

}  // namespace nuka::rt
