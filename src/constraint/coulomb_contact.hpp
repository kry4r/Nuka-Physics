#pragma once

#include <cfloat>
#include <cmath>
#include <cstdint>

#include "math/symmetric_mat3.hpp"

#if defined(__CUDACC__)
#define NUKA_COULOMB_HD __host__ __device__
#else
#define NUKA_COULOMB_HD
#endif

namespace nuka::constraint {

enum class ContactSolveMetric : uint32_t {
    NormalVelocity, TangentVelocity, NormalVelocityViolation, NormalImpulseViolation,
    ComplementarityWork, FrictionImpulseViolation, FrictionWorkViolation, FrictionDissipationWork,
    Count
};
inline constexpr uint32_t kContactSolveMetricCount = static_cast<uint32_t>(ContactSolveMetric::Count);
inline constexpr uint32_t kContactSolveCountSize = 2u;

NUKA_COULOMB_HD inline float ProjectedContactNormal(float response, float residual, float impulse) {
    if (!(response > 0.0f) || !(response <= FLT_MAX)) return 0.0f;
    return fmaxf(impulse + residual / response, 0.0f);
}

NUKA_COULOMB_HD inline float CoulombTangentSpectralResponse(
    const math::SymmetricMat3& response, float mu_first, float mu_second) {
    const float mu = fmaxf(mu_first, mu_second);
    if (!(mu > 0.0f) || !(mu <= FLT_MAX)) return 0.0f;
    const float first_scale = fmaxf(mu_first, 0.0f) / mu;
    const float second_scale = fmaxf(mu_second, 0.0f) / mu;
    const float h11 = response.yy * first_scale * first_scale;
    const float h22 = response.zz * second_scale * second_scale;
    const float h12 = response.yz * first_scale * second_scale;
    const float largest = 0.5f * (h11 + h22 + hypotf(h11 - h22, 2.0f * h12));
    return largest > FLT_MIN && largest <= FLT_MAX ? largest : 0.0f;
}

NUKA_COULOMB_HD inline math::Vec3 ProjectedCoulombTangentStepWithResponse(
    const math::SymmetricMat3& response, math::Vec3 residual, math::Vec3 impulse,
    float normal, float mu_first, float mu_second, float largest) {
    const float mu = fmaxf(mu_first, mu_second);
    if (!(normal > 0.0f) || !(mu > 0.0f)) return {normal, 0.0f, 0.0f};
    const float first_scale = fmaxf(mu_first, 0.0f) / mu;
    const float second_scale = fmaxf(mu_second, 0.0f) / mu;
    if (!(largest > 0.0f)) return {normal, 0.0f, 0.0f};
    float first = first_scale > 0.0f ? impulse.y / first_scale +
        first_scale * residual.y / largest : 0.0f;
    float second = second_scale > 0.0f ? impulse.z / second_scale +
        second_scale * residual.z / largest : 0.0f;
    const float length = hypotf(first, second);
    const float radius = normal * mu;
    if (length > radius) {
        const float scale = radius / length;
        first *= scale;
        second *= scale;
    }
    return {normal, first * first_scale, second * second_scale};
}

// Projects tangent impulse against a fixed normal impulse and friction ellipse.
NUKA_COULOMB_HD inline math::Vec3 ProjectedCoulombTangentStep(
    const math::SymmetricMat3& response, math::Vec3 residual, math::Vec3 impulse,
    float normal, float mu_first, float mu_second) {
    const float largest = CoulombTangentSpectralResponse(response, mu_first, mu_second);
    return ProjectedCoulombTangentStepWithResponse(
        response, residual, impulse, normal, mu_first, mu_second, largest);
}

NUKA_COULOMB_HD inline math::Vec3 ProjectedCoulombStepWithResponse(
    const math::SymmetricMat3& response, math::Vec3 residual, math::Vec3 impulse,
    float mu_first, float mu_second, float tangent_response) {
    const float normal = ProjectedContactNormal(response.xx, residual.x, impulse.x);
    const float delta_normal = normal - impulse.x;
    residual.y -= response.xy * delta_normal;
    residual.z -= response.xz * delta_normal;
    return ProjectedCoulombTangentStepWithResponse(
        response, residual, impulse, normal, mu_first, mu_second, tangent_response);
}

// Normal complementarity and tangential maximum dissipation have separate projections.
// A scalar spectral step in scaled tangent coordinates preserves the friction ellipse.
NUKA_COULOMB_HD inline math::Vec3 ProjectedCoulombStep(
    const math::SymmetricMat3& response, math::Vec3 residual, math::Vec3 impulse,
    float mu_first, float mu_second) {
    const float normal = ProjectedContactNormal(response.xx, residual.x, impulse.x);
    const float delta_normal = normal - impulse.x;
    residual.y -= response.xy * delta_normal;
    residual.z -= response.xz * delta_normal;
    return ProjectedCoulombTangentStep(
        response, residual, impulse, normal, mu_first, mu_second);
}

struct CoulombContactResidual {
    float normal_natural_velocity = FLT_MAX;
    float tangent_natural_velocity = FLT_MAX;
    float normal_velocity_violation = FLT_MAX;
    float normal_impulse_violation = FLT_MAX;
    float normal_complementarity = FLT_MAX;
    float friction_impulse_violation = FLT_MAX;
    float friction_power_violation = FLT_MAX;
    float friction_dissipation_work = FLT_MAX;
    bool valid = false;

    NUKA_COULOMB_HD bool Finite() const {
        return normal_natural_velocity >= 0.0f && normal_natural_velocity <= FLT_MAX &&
            tangent_natural_velocity >= 0.0f && tangent_natural_velocity <= FLT_MAX &&
            normal_velocity_violation >= 0.0f && normal_velocity_violation <= FLT_MAX &&
            normal_impulse_violation >= 0.0f && normal_impulse_violation <= FLT_MAX &&
            normal_complementarity >= 0.0f && normal_complementarity <= FLT_MAX &&
            friction_impulse_violation >= 0.0f && friction_impulse_violation <= FLT_MAX &&
            friction_power_violation >= 0.0f && friction_power_violation <= FLT_MAX &&
            friction_dissipation_work >= 0.0f && friction_dissipation_work <= FLT_MAX;
    }

    NUKA_COULOMB_HD bool Within(float velocity_tolerance, float impulse_tolerance,
                                float work_tolerance) const {
        return valid && Finite() && velocity_tolerance >= 0.0f && velocity_tolerance <= FLT_MAX &&
            impulse_tolerance >= 0.0f && impulse_tolerance <= FLT_MAX &&
            work_tolerance >= 0.0f && work_tolerance <= FLT_MAX &&
            normal_natural_velocity <= velocity_tolerance &&
            tangent_natural_velocity <= velocity_tolerance &&
            normal_velocity_violation <= velocity_tolerance &&
            normal_impulse_violation <= impulse_tolerance &&
            friction_impulse_violation <= impulse_tolerance &&
            normal_complementarity <= work_tolerance &&
            friction_power_violation <= work_tolerance &&
            friction_dissipation_work <= work_tolerance;
    }
};

// Evaluates the accepted state; velocity is w = Jv - target + R*lambda.
NUKA_COULOMB_HD inline CoulombContactResidual EvaluateCoulombContactResidual(
    const math::SymmetricMat3& response, math::Vec3 velocity, math::Vec3 impulse,
    float mu_first, float mu_second) {
    CoulombContactResidual result;
    const bool finite = response.xx >= 0.0f && response.xx <= FLT_MAX &&
        response.yy >= 0.0f && response.yy <= FLT_MAX &&
        response.zz >= 0.0f && response.zz <= FLT_MAX &&
        response.xy >= -FLT_MAX && response.xy <= FLT_MAX &&
        response.xz >= -FLT_MAX && response.xz <= FLT_MAX &&
        response.yz >= -FLT_MAX && response.yz <= FLT_MAX &&
        velocity.x >= -FLT_MAX && velocity.x <= FLT_MAX &&
        velocity.y >= -FLT_MAX && velocity.y <= FLT_MAX &&
        velocity.z >= -FLT_MAX && velocity.z <= FLT_MAX &&
        impulse.x >= -FLT_MAX && impulse.x <= FLT_MAX &&
        impulse.y >= -FLT_MAX && impulse.y <= FLT_MAX &&
        impulse.z >= -FLT_MAX && impulse.z <= FLT_MAX &&
        mu_first >= 0.0f && mu_first <= FLT_MAX &&
        mu_second >= 0.0f && mu_second <= FLT_MAX;
    if (!finite) return result;
    if (double(response.yz) * response.yz > double(response.yy) * response.zz)
        return result;

    const float normal_projected =
        ProjectedContactNormal(response.xx, -velocity.x, impulse.x);
    result.normal_natural_velocity = response.xx > 0.0f ?
        fabsf(impulse.x - normal_projected) * response.xx :
        (impulse.x > 0.0f ? fabsf(velocity.x) : fmaxf(-velocity.x, 0.0f));
    result.normal_velocity_violation = fmaxf(-velocity.x, 0.0f);
    result.normal_impulse_violation = fmaxf(-impulse.x, 0.0f);
    result.normal_complementarity = fabsf(impulse.x * velocity.x);
    const double tangent_work = double(impulse.y) * velocity.y + double(impulse.z) * velocity.z;
    result.friction_power_violation = static_cast<float>(fmax(tangent_work, 0.0));
    const double support_work = double(fmaxf(impulse.x, 0.0f)) *
        hypot(double(mu_first) * velocity.y, double(mu_second) * velocity.z);
    result.friction_dissipation_work = static_cast<float>(fabs(tangent_work + support_work));

    const float mu = fmaxf(mu_first, mu_second);
    float scaled_first = 0.0f;
    float scaled_second = 0.0f;
    float zero_axis_violation = 0.0f;
    if (mu_first > 0.0f) scaled_first = impulse.y / mu_first;
    else zero_axis_violation = fabsf(impulse.y);
    if (mu_second > 0.0f) scaled_second = impulse.z / mu_second;
    else zero_axis_violation = fmaxf(zero_axis_violation, fabsf(impulse.z));
    result.friction_impulse_violation = fmaxf(
        zero_axis_violation, fmaxf(hypotf(scaled_first, scaled_second) - impulse.x, 0.0f));

    if (!(mu > 0.0f)) {
        result.tangent_natural_velocity = 0.0f;
        result.valid = result.Finite();
        return result;
    }
    const float largest = CoulombTangentSpectralResponse(response, mu_first, mu_second);
    if (!(largest > 0.0f)) {
        if ((mu_first > 0.0f && response.yy != 0.0f) ||
            (mu_second > 0.0f && response.zz != 0.0f)) return result;
        result.tangent_natural_velocity = 0.0f;
        result.valid = result.Finite();
        return result;
    }
    const math::Vec3 projected = ProjectedCoulombTangentStep(
        response, -velocity, impulse, impulse.x, mu_first, mu_second);
    const float first_scale = mu_first / mu;
    const float second_scale = mu_second / mu;
    const float first_delta = first_scale > 0.0f ?
        (impulse.y - projected.y) / first_scale : 0.0f;
    const float second_delta = second_scale > 0.0f ?
        (impulse.z - projected.z) / second_scale : 0.0f;
    result.tangent_natural_velocity = largest * hypotf(first_delta, second_delta);
    result.valid = result.Finite();
    return result;
}

}  // namespace nuka::constraint

#undef NUKA_COULOMB_HD
