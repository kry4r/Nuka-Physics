// V3 finite-difference validator for the XPBDBendRow reverse-mode adjoint
// (v0.7 p09-B). Central-differences the GENERATED primal forward-update and
// compares to the GENERATED analytical adjoint (seed v=1), over a batch sampled
// strictly inside the (unbounded) equality box so clamp_active is always 1.

#include "codegen/v3_validation/xpbd_bend_fd.hpp"

#include "codegen/generated/xpbd_bend_adjoint.cuh"

#include <cmath>
#include <cstdint>
#include <random>

namespace nuka::codegen::v3 {

namespace {

using nuka::solver::generated::XPBDBendAdjGrads;
using nuka::solver::generated::XPBDBendAdjInputs;
using nuka::solver::generated::xpbd_bend_adjoint_eval;
using nuka::solver::generated::xpbd_bend_clamp_active;
using nuka::solver::generated::xpbd_bend_forward_eval;

// Four differentiable fields: C, lambda, effective_mass, alpha_tilde.
inline constexpr int kXpbdBendGradCount = 4;

float SampleSymmetric(std::mt19937_64& rng, float abs_max) {
    std::uniform_real_distribution<float> dist(-abs_max, abs_max);
    return dist(rng);
}

float SampleRange(std::mt19937_64& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

double RelError(double analytic, double numeric) {
    constexpr double kEps = 1.0e-6;
    return std::fabs(analytic - numeric) /
           (std::fabs(analytic) + std::fabs(numeric) + kEps);
}

void Partials(const XPBDBendAdjGrads& g, float out[kXpbdBendGradCount]) {
    out[0] = g.grad_C;
    out[1] = g.grad_lambda;
    out[2] = g.grad_effective_mass;
    out[3] = g.grad_alpha_tilde;
}

XPBDBendAdjInputs Perturb(const XPBDBendAdjInputs& base, int idx, float delta) {
    XPBDBendAdjInputs in = base;
    switch (idx) {
    case 0: in.C += delta; break;
    case 1: in.lambda += delta; break;
    case 2: in.effective_mass += delta; break;
    case 3: in.alpha_tilde += delta; break;
    default: break;
    }
    return in;
}

} // namespace

XpbdBendFdResult ValidateXpbdBendFd(const XpbdBendFdConfig& config) {
    std::mt19937_64 rng(config.seed);
    XpbdBendFdResult result;
    result.case_count = config.num_cases;

    for (uint32_t i = 0; i < config.num_cases; ++i) {
        XPBDBendAdjInputs in;
        in.C = SampleSymmetric(rng, config.C_abs_max);
        // lambda away from zero (magnitude in [lambda_min, lambda_max], random
        // sign) so grad_alpha_tilde = -effective_mass*lambda stays well-
        // conditioned against the float32 FD cancellation floor.
        {
            const float magnitude = SampleRange(rng, config.lambda_min, config.lambda_max);
            const float sign = (rng() & 1u) ? 1.0f : -1.0f;
            in.lambda = sign * magnitude;
        }
        in.effective_mass = SampleRange(rng, config.meff_min, config.meff_max);
        in.alpha_tilde = SampleRange(rng, config.alpha_tilde_min, config.alpha_tilde_max);
        in.lower = config.lower;
        in.upper = config.upper;

        // Equality row: the wide box must leave the row unclamped. A regression in
        // the band sampling surfaces here rather than as a misleading zero slope.
        if (xpbd_bend_clamp_active(in) != 1.0f) {
            result.max_rel_err = 1.0f;
            result.worst_case_index = i;
            continue;
        }

        const XPBDBendAdjGrads g = xpbd_bend_adjoint_eval(in, 1.0f);
        float analytic[kXpbdBendGradCount];
        Partials(g, analytic);

        const double delta = static_cast<double>(config.fd_delta);
        for (int p = 0; p < kXpbdBendGradCount; ++p) {
            const XPBDBendAdjInputs plus = Perturb(in, p, static_cast<float>(delta));
            const XPBDBendAdjInputs minus = Perturb(in, p, static_cast<float>(-delta));
            const double numeric =
                (static_cast<double>(xpbd_bend_forward_eval(plus)) -
                 static_cast<double>(xpbd_bend_forward_eval(minus))) /
                (2.0 * delta);
            const double rel = RelError(static_cast<double>(analytic[p]), numeric);
            if (rel > static_cast<double>(result.max_rel_err)) {
                result.max_rel_err = static_cast<float>(rel);
                result.worst_case_index = i;
                result.worst_param_index = static_cast<uint32_t>(p);
            }
        }
    }
    return result;
}

} // namespace nuka::codegen::v3
