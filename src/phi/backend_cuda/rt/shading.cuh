#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- SIMPLE Lambert + GGX direct shading for the sensor renderer
// (v0.7 p13). ONE analytic light + a flat ambient fill. This is the BOUNDED
// sensor shade, NOT the A1 photoreal path: NO multi-bounce GI, NO ambient
// occlusion, NO anti-aliasing, NO denoise, NO tonemap, NO area lights (all G2,
// docs/plans/2026-06-03-post-v07-roadmap.md). One ray per pixel, no RNG.
//
// Hard shadows are computed by the CALLER (it has the BVH / oracle to trace the
// secondary ray); ShadeDirect takes a `lit` flag (0/1) so the SAME shade math is
// shared by the kernel AND the CPU oracle (the p12 pattern). HD, fp64-internal in
// the load-bearing terms; RGB is a tolerance gate (analytic N.L / known BRDF), so
// it is less ULP-sensitive than depth/normal -- but kept deterministic + FMA-free
// so the D1 two-run memcmp on color is also exact.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "rt/material.hpp"

#include <cstdint>

#if defined(__CUDACC__)
#define NUKA_RT_HD __host__ __device__
#else
#define NUKA_RT_HD
#endif

namespace nuka::rt {

NUKA_RT_HD inline double RtMax(double a, double b) { return a > b ? a : b; }

// GGX (Trowbridge-Reitz) normal-distribution term D, the specular lobe shape.
// alpha = roughness^2. Returns D(NoH). fp64 internal, no FMA.
NUKA_RT_HD inline double GgxD(double NoH, double alpha) {
    const double a2 = alpha * alpha;
    const double d = NoH * NoH * (a2 - 1.0) + 1.0;
    const double pi = 3.14159265358979323846;
    // Guard d==0 (NoH==0 && a2==0): denom -> 0; clamp to tiny.
    const double denom = pi * d * d;
    if (denom < 1.0e-12) {
        return 0.0;
    }
    return a2 / denom;
}

// Direct shading at a hit point. Inputs:
//   N         unit surface normal (geometric; flipped to face the viewer below)
//   V         unit view direction (toward the camera, == -ray.dir)
//   L         unit light direction (toward the light)
//   light_col light color * intensity
//   mat       material
//   lit       1 if the light is visible (shadow ray unoccluded), else 0
//   ambient   flat ambient fill color
// Returns linear RGB radiance. Lambert diffuse (energy-normalised by 1/pi) +
// GGX specular (D-only simplified lobe per the p13 plan, scaled by metallic) *
// N.L, plus emission + a flat ambient term. Deterministic, FMA-free.
NUKA_RT_HD inline math::Vec3 ShadeDirect(math::Vec3 N,
                                         math::Vec3 V,
                                         math::Vec3 L,
                                         math::Vec3 light_col,
                                         const Material& mat,
                                         int lit,
                                         math::Vec3 ambient) {
    // Face the normal toward the viewer (two-sided shading for sensor use).
    const double nv = static_cast<double>(N.x) * V.x +
                      static_cast<double>(N.y) * V.y +
                      static_cast<double>(N.z) * V.z;
    math::Vec3 Nf = N;
    if (nv < 0.0) {
        Nf = math::Vec3{-N.x, -N.y, -N.z};
    }

    const double NoL = RtMax(static_cast<double>(Nf.x) * L.x +
                             static_cast<double>(Nf.y) * L.y +
                             static_cast<double>(Nf.z) * L.z, 0.0);

    // Lambert diffuse (1 - metallic) * albedo * NoL / pi.
    const double pi = 3.14159265358979323846;
    const double kd = (1.0 - static_cast<double>(mat.metallic)) / pi;

    // Half vector H = normalize(V + L). fp64; true sqrt.
    const double hx = static_cast<double>(V.x) + L.x;
    const double hy = static_cast<double>(V.y) + L.y;
    const double hz = static_cast<double>(V.z) + L.z;
    const double hlen2 = hx * hx + hy * hy + hz * hz;
    double NoH = 0.0;
    if (hlen2 > 0.0) {
        const double inv = 1.0 / sqrt(hlen2);
        NoH = RtMax((static_cast<double>(Nf.x) * hx +
                     static_cast<double>(Nf.y) * hy +
                     static_cast<double>(Nf.z) * hz) * inv, 0.0);
    }
    const double alpha = static_cast<double>(mat.roughness) * static_cast<double>(mat.roughness);
    const double spec = GgxD(NoH, alpha) * static_cast<double>(mat.metallic);

    const double l = lit ? 1.0 : 0.0;
    // radiance = light * (diffuse_albedo*kd + spec) * NoL * lit  + emission + ambient*albedo
    math::Vec3 out;
    out.x = static_cast<float>(static_cast<double>(light_col.x) *
                (static_cast<double>(mat.albedo.x) * kd + spec) * NoL * l +
                static_cast<double>(mat.emission.x) +
                static_cast<double>(ambient.x) * static_cast<double>(mat.albedo.x));
    out.y = static_cast<float>(static_cast<double>(light_col.y) *
                (static_cast<double>(mat.albedo.y) * kd + spec) * NoL * l +
                static_cast<double>(mat.emission.y) +
                static_cast<double>(ambient.y) * static_cast<double>(mat.albedo.y));
    out.z = static_cast<float>(static_cast<double>(light_col.z) *
                (static_cast<double>(mat.albedo.z) * kd + spec) * NoL * l +
                static_cast<double>(mat.emission.z) +
                static_cast<double>(ambient.z) * static_cast<double>(mat.albedo.z));
    return out;
}

} // namespace nuka::rt

#undef NUKA_RT_HD
