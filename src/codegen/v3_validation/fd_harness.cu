// V3 finite-difference validator for the maximal_drive (dense_adjoint) slice.
//
// Generates random away-from-clamp test cases, central-differences the
// generated primal forward-update, and compares to the generated analytical
// adjoint (seed v=1 -> the adjoint returns the row of partials directly).

#include "codegen/v3_validation/fd_harness.hpp"

#include "codegen/v3_validation/numerical_jacobian.cuh"

#include "codegen/generated/maximal_drive_adjoint.cuh"

#include <cmath>
#include <cstdint>
#include <random>

namespace nuka::codegen::v3 {

using nuka::solver::generated::MaximalDriveAdjGrads;
using nuka::solver::generated::MaximalDriveAdjInputs;
using nuka::solver::generated::maximal_drive_adjoint_eval;
using nuka::solver::generated::maximal_drive_clamp_active;

namespace {

// Sign-preserving magnitude sampler in [floor, ceil] (|value| >= floor).
float SampleAwayFromZero(std::mt19937_64& rng, float floor, float ceil) {
    std::uniform_real_distribution<float> mag(floor, ceil);
    std::uniform_int_distribution<int> sign(0, 1);
    const float m = mag(rng);
    return sign(rng) == 0 ? m : -m;
}

float SampleSymmetric(std::mt19937_64& rng, float abs_max) {
    std::uniform_real_distribution<float> dist(-abs_max, abs_max);
    return dist(rng);
}

// Pull the analytical partials out of the grad bundle (seed=1) in field order.
void AnalyticPartials(const MaximalDriveAdjGrads& g, float out[kMaximalDriveGradCount]) {
    out[0] = g.grad_q;
    out[1] = g.grad_qdot;
    out[2] = g.grad_drive_target;
    out[3] = g.grad_drive_stiffness;
    out[4] = g.grad_drive_damping;
}

// Symmetric relative error with a denominator floor (R9: keeps near-zero
// partials from inflating relative error toward infinity).
double RelError(double analytic, double numeric) {
    constexpr double kEps = 1.0e-6;
    return std::fabs(analytic - numeric) /
           (std::fabs(analytic) + std::fabs(numeric) + kEps);
}

} // namespace

FdValidationResult ValidateMaximalDriveFd(const MaximalDriveFdConfig& config) {
    std::mt19937_64 rng(config.seed);
    FdValidationResult result;
    result.case_count = config.num_cases;

    for (uint32_t i = 0; i < config.num_cases; ++i) {
        MaximalDriveAdjInputs in;
        in.q = SampleSymmetric(rng, config.q_abs_max);
        in.qdot = SampleAwayFromZero(rng, config.qdot_abs_min, config.qdot_abs_max);
        // Sample the position gap (target - q) away from zero, then derive target.
        const float gap = SampleAwayFromZero(rng, config.target_gap_min, config.target_gap_max);
        in.drive_target = in.q + gap;
        {
            std::uniform_real_distribution<float> kp(config.kp_min, config.kp_max);
            std::uniform_real_distribution<float> kd(config.kd_min, config.kd_max);
            in.drive_stiffness = kp(rng);
            in.drive_damping = kd(rng);
        }
        in.force_limit = config.force_limit;

        // By construction the base point and its central-difference stencil stay
        // strictly inside the band; assert the clamp is inactive so a sampling
        // regression surfaces here rather than as a misleading FD slope.
        if (maximal_drive_clamp_active(in) != 1.0f) {
            // Treat an unexpectedly-saturated case as max error so the test fails
            // loudly instead of silently measuring across the clamp.
            result.max_rel_err = 1.0f;
            result.worst_case_index = i;
            continue;
        }

        const MaximalDriveAdjGrads g = maximal_drive_adjoint_eval(in, 1.0f);
        float analytic[kMaximalDriveGradCount];
        AnalyticPartials(g, analytic);

        double numeric[kMaximalDriveGradCount];
        NumericalJacobianMaximalDrive(in, static_cast<double>(config.fd_delta), numeric);

        for (int p = 0; p < kMaximalDriveGradCount; ++p) {
            const double rel = RelError(static_cast<double>(analytic[p]), numeric[p]);
            if (rel > static_cast<double>(result.max_rel_err)) {
                result.max_rel_err = static_cast<float>(rel);
                result.worst_case_index = i;
                result.worst_param_index = static_cast<uint32_t>(p);
            }
        }
    }

    return result;
}

float MaxSaturatedAdjointMagnitude(const MaximalDriveFdConfig& config) {
    std::mt19937_64 rng(config.seed ^ 0x5A7U);
    float worst = 0.0f;

    for (uint32_t i = 0; i < config.num_cases; ++i) {
        MaximalDriveAdjInputs in;
        in.q = SampleSymmetric(rng, config.q_abs_max);
        in.qdot = SampleAwayFromZero(rng, config.qdot_abs_min, config.qdot_abs_max);
        in.drive_stiffness = config.kp_max;
        in.drive_damping = config.kd_max;
        // Force saturation: pick a small force_limit and a large position gap so
        // |u| = Kp*|gap| is strictly past the limit (with margin).
        in.force_limit = 1.0f;
        const float gap_sign = (i % 2 == 0) ? 1.0f : -1.0f;
        in.drive_target = in.q + gap_sign * 10.0f; // Kp(>=5) * 10 >> 1

        // Confirm we are actually saturated (defensive against config drift).
        if (maximal_drive_clamp_active(in) != 0.0f) {
            return 1.0e9f; // forces the caller's == 0 assertion to fail loudly
        }

        const MaximalDriveAdjGrads g = maximal_drive_adjoint_eval(in, 1.0f);
        worst = std::fmax(worst, std::fabs(g.grad_q));
        worst = std::fmax(worst, std::fabs(g.grad_qdot));
        worst = std::fmax(worst, std::fabs(g.grad_drive_target));
        worst = std::fmax(worst, std::fabs(g.grad_drive_stiffness));
        worst = std::fmax(worst, std::fabs(g.grad_drive_damping));
    }

    return worst;
}

} // namespace nuka::codegen::v3
