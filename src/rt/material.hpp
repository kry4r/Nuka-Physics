#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- material + analytic light types for the sensor renderer (v0.7 p13).
//
// SIMPLE shading data: per-primitive Lambert+GGX material (albedo / metallic /
// roughness / emission) indexed by the primitive's material id, plus an analytic
// directional OR point light. This is the BOUNDED sensor renderer -- NOT the A1
// photoreal path (no area lights / IBL / multi-bounce; those are G2 in
// docs/plans/2026-06-03-post-v07-roadmap.md). HD-trivial structs so the same
// definitions are used by the kernel and the CPU shading oracle.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"

#include <cstdint>

namespace nuka::rt {

// Lambert + GGX material. albedo is base color; metallic in [0,1] tints the
// specular and removes diffuse; roughness in (0,1] is the GGX alpha source.
//
// transmission/ior/absorption are the dielectric extension consumed ONLY by the
// stochastic beauty path (Fresnel reflect+refract + Beer-Lambert tint). They
// DEFAULT to a strict no-op (transmission==0 => opaque): the FP64 golden/sensor
// shade (ShadeDirect) never reads them, so an opaque material is byte-unchanged.
struct Material {
    math::Vec3 albedo{0.8f, 0.8f, 0.8f};
    float metallic = 0.0f;
    float roughness = 0.5f;
    math::Vec3 emission{0.0f, 0.0f, 0.0f};
    float transmission = 0.0f;            // 0 = opaque; 1 = fully transmissive
    float ior = 1.0f;                     // refractive index (water ~1.33)
    math::Vec3 absorption{0.0f, 0.0f, 0.0f};  // per-metre Beer-Lambert extinction
    // Grazing retroreflective sheen weight (silk/satin), beauty-only. 0 (default) is a
    // strict no-op: ShadeDirect ignores it and ShadeBeauty skips the term byte-exactly.
    float sheen = 0.0f;
};

// One analytic light. directional == true: `direction` is the (unit) direction
// the light TRAVELS (so the vector TO the light is -direction); distance is
// infinite (shadow rays use a large t_max). directional == false: `position` is
// a world point; the vector to the light is (position - hit) and the shadow ray
// t_max is that distance.
struct Light {
    math::Vec3 color{1.0f, 1.0f, 1.0f};
    math::Vec3 direction{0.0f, -1.0f, 0.0f}; // travel dir (directional)
    math::Vec3 position{0.0f, 0.0f, 0.0f};   // world point (point light)
    float intensity = 1.0f;
    bool directional = true;
};

// Ambient term added once (flat, NOT ambient occlusion -- AO is a G2 photoreal
// feature, explicitly out of p13 scope). Small constant fill so unshadowed-but-
// backfacing geometry is not pure black in the sensor image.
struct AmbientTerm {
    math::Vec3 color{0.03f, 0.03f, 0.03f};
};

} // namespace nuka::rt
