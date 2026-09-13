#pragma once

#include <cmath>
#include <cstdint>

#include "math/vec3.hpp"

#if defined(__CUDACC__)
#define NUKA_TACTILE_HD __host__ __device__
#else
#define NUKA_TACTILE_HD
#endif

namespace nuka::sensor {

enum class ContactRegionShape : uint32_t { Box, Sphere, Ellipsoid, Capsule, Cylinder };

struct TactileConfig {
    ContactRegionShape shape = ContactRegionShape::Box;
    math::Vec3 size{};
    float spread_fraction = 0.0f;
    float spread_sigma = 0.0f;
    float hysteresis_strength = 0.0f;
    float hysteresis_time = 0.0f;
};

struct TactileState {
    math::Vec3 impulse;
    uint32_t reserved = 0u;
    double previous_force[3] = {};
    double relaxation[3] = {};
};

inline bool ValidTactileConfig(const TactileConfig& config, bool normal_only) {
    const auto s = config.size;
    if (!std::isfinite(s.x) || !std::isfinite(s.y) || !std::isfinite(s.z) ||
        !(s.x > 0.0f) || s.y < 0.0f || s.z < 0.0f ||
        !std::isfinite(config.spread_fraction) || config.spread_fraction < 0.0f || config.spread_fraction > 1.0f ||
        !std::isfinite(config.spread_sigma) || config.spread_sigma < 0.0f ||
        !std::isfinite(config.hysteresis_strength) || config.hysteresis_strength < 0.0f ||
        !std::isfinite(config.hysteresis_time) || config.hysteresis_time < 0.0f ||
        (config.hysteresis_strength > 0.0f && config.hysteresis_time == 0.0f) ||
        (config.spread_fraction > 0.0f && config.spread_sigma == 0.0f)) return false;
    if (normal_only && (config.spread_fraction != 0.0f || config.spread_sigma != 0.0f)) return false;
    if (!normal_only && config.shape != ContactRegionShape::Box) return false;
    switch (config.shape) {
        case ContactRegionShape::Box:
        case ContactRegionShape::Ellipsoid: return s.y > 0.0f && s.z > 0.0f;
        case ContactRegionShape::Sphere: return s.y == 0.0f && s.z == 0.0f;
        case ContactRegionShape::Capsule: return s.z == 0.0f;
        case ContactRegionShape::Cylinder: return s.y > 0.0f && s.z == 0.0f;
    }
    return false;
}

namespace tactile_detail {

NUKA_TACTILE_HD inline bool Slab(double p, double d, double half, double& near, double& far) {
    if (d == 0.0) return fabs(p) <= half;
    const double a = (-half - p) / d;
    const double b = (half - p) / d;
    near = fmax(near, fmin(a, b));
    far = fmin(far, fmax(a, b));
    return far >= near;
}

NUKA_TACTILE_HD inline bool SphereRay(math::Vec3 p, math::Vec3 d, double radius) {
    const double c = double{p.x} * p.x + double{p.y} * p.y + double{p.z} * p.z - radius * radius;
    if (c <= 0.0) return true;
    const double b = double{p.x} * d.x + double{p.y} * d.y + double{p.z} * d.z;
    const double a = double{d.x} * d.x + double{d.y} * d.y + double{d.z} * d.z;
    return a > 0.0 && b < 0.0 && b * b - a * c >= 0.0;
}

NUKA_TACTILE_HD inline bool CylinderRay(math::Vec3 p, math::Vec3 d, double radius, double half_height) {
    double near = 0.0, far = INFINITY;
    if (!Slab(p.z, d.z, half_height, near, far)) return false;
    const double a = double{d.x} * d.x + double{d.y} * d.y;
    const double b = double{p.x} * d.x + double{p.y} * d.y;
    const double c = double{p.x} * p.x + double{p.y} * p.y - radius * radius;
    if (a == 0.0) return c <= 0.0;
    const double discriminant = b * b - a * c;
    if (discriminant < 0.0) return false;
    const double root = sqrt(discriminant);
    return fmin(far, (-b + root) / a) >= fmax(near, (-b - root) / a);
}

NUKA_TACTILE_HD inline float GaussianInterval(float center, float half, float sigma) {
    const double scale = 0.7071067811865475244 / sigma;
    const double lower = (-double{half} - center) * scale;
    const double upper = (double{half} - center) * scale;
    const double mass = lower >= 0.0 ? erfc(lower) - erfc(upper) :
        upper <= 0.0 ? erfc(-upper) - erfc(-lower) : erf(upper) - erf(lower);
    return static_cast<float>(fmax(0.0, fmin(1.0, 0.5 * mass)));
}

}  // namespace tactile_detail

// A touch volume collects contacts whose outward normal ray reaches its sensing region.
// Box/ellipsoid sizes are half-axes; sphere uses radius; local-Z capsule/cylinder use radius and half-height.
NUKA_TACTILE_HD inline bool TouchRegionIntersectsRay(const TactileConfig& config, math::Vec3 p, math::Vec3 d) {
    using namespace tactile_detail;
    const auto s = config.size;
    switch (config.shape) {
        case ContactRegionShape::Box: {
            double near = 0.0, far = INFINITY;
            return Slab(p.x, d.x, s.x, near, far) && Slab(p.y, d.y, s.y, near, far) &&
                Slab(p.z, d.z, s.z, near, far);
        }
        case ContactRegionShape::Sphere: return SphereRay(p, d, s.x);
        case ContactRegionShape::Ellipsoid:
            return SphereRay({p.x / s.x, p.y / s.y, p.z / s.z}, {d.x / s.x, d.y / s.y, d.z / s.z}, 1.0);
        case ContactRegionShape::Capsule:
            return CylinderRay(p, d, s.x, s.y) || SphereRay({p.x, p.y, p.z - s.y}, d, s.x) ||
                SphereRay({p.x, p.y, p.z + s.y}, d, s.x);
        case ContactRegionShape::Cylinder: return CylinderRay(p, d, s.x, s.y);
    }
    return false;
}

// The rectangle is half-open in x/y, with a finite sensing depth along local Z.
// Gaussian mass is integrated over its area; finite array edges retain their physical spill loss.
NUKA_TACTILE_HD inline float TaxelWeight(const TactileConfig& config, math::Vec3 p) {
    const auto s = config.size;
    if (fabsf(p.z) > s.z) return 0.0f;
    const float ideal = p.x >= -s.x && p.x < s.x && p.y >= -s.y && p.y < s.y ? 1.0f : 0.0f;
    if (config.spread_fraction == 0.0f) return ideal;
    const float spread = tactile_detail::GaussianInterval(p.x, s.x, config.spread_sigma) *
        tactile_detail::GaussianInterval(p.y, s.y, config.spread_sigma);
    return (1.0f - config.spread_fraction) * ideal + config.spread_fraction * spread;
}

// A Maxwell observation branch has equilibrium gain one and exact piecewise-constant substep integration.
NUKA_TACTILE_HD inline math::Vec3 IntegrateTactileResponse(
    const TactileConfig& config, TactileState& state, double interval) {
    if (config.hysteresis_strength == 0.0f) return state.impulse;
    const double decay = exp(-interval / config.hysteresis_time);
    const double integral_scale = config.hysteresis_time * -expm1(-interval / config.hysteresis_time);
    const float input[] = {state.impulse.x, state.impulse.y, state.impulse.z};
    float output[3];
    for (uint32_t c = 0u; c < 3u; ++c) {
        const double force = input[c] / interval;
        const double transient = state.relaxation[c] + force - state.previous_force[c];
        output[c] = static_cast<float>(input[c] + config.hysteresis_strength * transient * integral_scale);
        state.previous_force[c] = force;
        state.relaxation[c] = transient * decay;
    }
    return {output[0], output[1], output[2]};
}

}  // namespace nuka::sensor

#undef NUKA_TACTILE_HD
