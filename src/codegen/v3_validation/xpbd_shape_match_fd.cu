// V3 finite-difference validator for the XPBDShapeMatchRow reverse-mode adjoint
// (v0.7 p09-C). Central-differences the GENERATED primal forward-update and
// compares to the GENERATED analytical adjoint (seed v=1), over a batch sampled
// strictly inside the (unbounded) equality box so clamp_active is always 1.

#include "codegen/v3_validation/xpbd_shape_match_fd.hpp"

#include "codegen/generated/xpbd_shape_match_adjoint.cuh"

#include <cmath>
#include <cstdint>
#include <random>

namespace nuka::codegen::v3 {

namespace {

using nuka::solver::generated::XPBDShapeMatchAdjGrads;
using nuka::solver::generated::XPBDShapeMatchAdjInputs;
using nuka::solver::generated::xpbd_shape_match_adjoint_eval;
using nuka::solver::generated::xpbd_shape_match_clamp_active;
using nuka::solver::generated::xpbd_shape_match_forward_eval;

// Three differentiable fields: position, goal, stiffness.
inline constexpr int kXpbdShapeMatchGradCount = 3;

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

void Partials(const XPBDShapeMatchAdjGrads& g, float out[kXpbdShapeMatchGradCount]) {
    out[0] = g.grad_position;
    out[1] = g.grad_goal;
    out[2] = g.grad_stiffness;
}

XPBDShapeMatchAdjInputs Perturb(const XPBDShapeMatchAdjInputs& base, int idx,
                                float delta) {
    XPBDShapeMatchAdjInputs in = base;
    switch (idx) {
    case 0: in.position += delta; break;
    case 1: in.goal += delta; break;
    case 2: in.stiffness += delta; break;
    default: break;
    }
    return in;
}

} // namespace

XpbdShapeMatchFdResult ValidateXpbdShapeMatchFd(const XpbdShapeMatchFdConfig& config) {
    std::mt19937_64 rng(config.seed);
    XpbdShapeMatchFdResult result;
    result.case_count = config.num_cases;

    for (uint32_t i = 0; i < config.num_cases; ++i) {
        XPBDShapeMatchAdjInputs in;
        in.position = SampleSymmetric(rng, config.position_abs_max);
        // goal away from position (|gap| in [gap_min, gap_max], random sign) so
        // grad_stiffness = (goal - position) stays well-conditioned against the
        // float32 FD cancellation floor.
        {
            const float magnitude = SampleRange(rng, config.gap_min, config.gap_max);
            const float sign = (rng() & 1u) ? 1.0f : -1.0f;
            in.goal = in.position + sign * magnitude;
        }
        in.stiffness = SampleRange(rng, config.stiffness_min, config.stiffness_max);
        in.lower = config.lower;
        in.upper = config.upper;

        // Equality row: the wide box must leave the row unclamped. A regression in
        // the band sampling surfaces here rather than as a misleading zero slope.
        if (xpbd_shape_match_clamp_active(in) != 1.0f) {
            result.max_rel_err = 1.0f;
            result.worst_case_index = i;
            continue;
        }

        const XPBDShapeMatchAdjGrads g = xpbd_shape_match_adjoint_eval(in, 1.0f);
        float analytic[kXpbdShapeMatchGradCount];
        Partials(g, analytic);

        const double delta = static_cast<double>(config.fd_delta);
        for (int p = 0; p < kXpbdShapeMatchGradCount; ++p) {
            const XPBDShapeMatchAdjInputs plus =
                Perturb(in, p, static_cast<float>(delta));
            const XPBDShapeMatchAdjInputs minus =
                Perturb(in, p, static_cast<float>(-delta));
            const double numeric =
                (static_cast<double>(xpbd_shape_match_forward_eval(plus)) -
                 static_cast<double>(xpbd_shape_match_forward_eval(minus))) /
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
