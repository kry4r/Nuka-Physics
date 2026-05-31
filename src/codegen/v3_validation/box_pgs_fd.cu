// V3 finite-difference validator for the box-clamp PGS row classes.
//
// One generic core (templated over the generated primal/adjoint pair) drives all
// three classes, since maximal_joint / maximal_contact / featherstone_contact
// share the single-row PGS law and the same four differentiable fields
// (jv, rhs, lambda, effective_mass). Each public entry point binds the core to a
// class's GENERATED header (the single source of truth for that class's law).

#include "codegen/v3_validation/box_pgs_fd.hpp"

#include "codegen/generated/featherstone_contact_adjoint.cuh"
#include "codegen/generated/maximal_contact_adjoint.cuh"
#include "codegen/generated/maximal_joint_adjoint.cuh"

#include <cmath>
#include <cstdint>
#include <random>

namespace nuka::codegen::v3 {

namespace {

// Four differentiable fields shared by every box-PGS row class.
inline constexpr int kBoxPgsGradCount = 4;

float SampleSymmetric(std::mt19937_64& rng, float abs_max) {
    std::uniform_real_distribution<float> dist(-abs_max, abs_max);
    return dist(rng);
}

float SampleRange(std::mt19937_64& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

// Symmetric relative error with a denominator floor (keeps near-zero partials
// from inflating relative error toward infinity).
double RelError(double analytic, double numeric) {
    constexpr double kEps = 1.0e-6;
    return std::fabs(analytic - numeric) /
           (std::fabs(analytic) + std::fabs(numeric) + kEps);
}

// Trait binding a generated row class's pair to the generic core. Each row class
// has the SAME field set, so the trait just forwards to the class's symbols.
template <typename Inputs, typename Grads>
struct BoxPgsBinding {
    // Pull the analytical partials out of the grad bundle (seed=1) in field order
    // (jv, rhs, lambda, effective_mass).
    static void Partials(const Grads& g, float out[kBoxPgsGradCount]) {
        out[0] = g.grad_jv;
        out[1] = g.grad_rhs;
        out[2] = g.grad_lambda;
        out[3] = g.grad_effective_mass;
    }
    // Perturb field `idx` of a copy of `base` by `delta`.
    static Inputs Perturb(const Inputs& base, int idx, float delta) {
        Inputs in = base;
        switch (idx) {
        case 0: in.jv += delta; break;
        case 1: in.rhs += delta; break;
        case 2: in.lambda += delta; break;
        case 3: in.effective_mass += delta; break;
        default: break;
        }
        return in;
    }
};

// Generic away-from-boundary FD batch. ForwardEval/AdjointEval/ClampActive are
// the GENERATED __host__ __device__ functions for one class.
template <typename Inputs, typename Grads, typename ForwardFn, typename AdjointFn,
          typename ClampFn>
BoxPgsFdResult ValidateBoxPgs(const BoxPgsFdConfig& config,
                              ForwardFn forward_eval,
                              AdjointFn adjoint_eval,
                              ClampFn clamp_active) {
    using Bind = BoxPgsBinding<Inputs, Grads>;
    std::mt19937_64 rng(config.seed);
    BoxPgsFdResult result;
    result.case_count = config.num_cases;

    for (uint32_t i = 0; i < config.num_cases; ++i) {
        Inputs in;
        in.jv = SampleSymmetric(rng, config.jv_abs_max);
        in.rhs = SampleSymmetric(rng, config.rhs_abs_max);
        in.lambda = SampleRange(rng, config.lambda_min, config.lambda_max);
        in.effective_mass = SampleRange(rng, config.meff_min, config.meff_max);
        in.lower = config.lower;
        in.upper = config.upper;

        // The base point and its central-difference stencil are constructed to
        // stay strictly inside the box; assert the row is unclamped so a sampling
        // regression surfaces here rather than as a misleading FD slope.
        if (clamp_active(in) != 1.0f) {
            result.max_rel_err = 1.0f;
            result.worst_case_index = i;
            continue;
        }

        const Grads g = adjoint_eval(in, 1.0f);
        float analytic[kBoxPgsGradCount];
        Bind::Partials(g, analytic);

        const double delta = static_cast<double>(config.fd_delta);
        for (int p = 0; p < kBoxPgsGradCount; ++p) {
            const Inputs plus = Bind::Perturb(in, p, static_cast<float>(delta));
            const Inputs minus = Bind::Perturb(in, p, static_cast<float>(-delta));
            const double numeric =
                (static_cast<double>(forward_eval(plus)) -
                 static_cast<double>(forward_eval(minus))) /
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

// Generic AT/over-boundary event batch (stop_grad_on_event). Drives u strictly
// past the upper box edge so the row is clamped (event); every analytical grad
// component must be exactly 0 (the stop-gradient). Returns max |grad|.
template <typename Inputs, typename Grads, typename AdjointFn, typename ClampFn>
float MaxEventStoppedMagnitude(const BoxPgsFdConfig& config,
                              AdjointFn adjoint_eval,
                              ClampFn clamp_active) {
    using Bind = BoxPgsBinding<Inputs, Grads>;
    std::mt19937_64 rng(config.seed ^ 0x5A7U);
    float worst = 0.0f;

    for (uint32_t i = 0; i < config.num_cases; ++i) {
        Inputs in;
        in.jv = SampleSymmetric(rng, config.jv_abs_max);
        in.rhs = SampleSymmetric(rng, config.rhs_abs_max);
        in.lambda = SampleRange(rng, config.lambda_min, config.lambda_max);
        in.effective_mass = SampleRange(rng, config.meff_min, config.meff_max);
        // Tight box so u = lambda + m_eff*(rhs - jv) (>= ~0.0) sits strictly past
        // an edge. Alternate which edge to exercise both clamp branches.
        if (i % 2u == 0u) {
            in.lower = -1.0e-3f;
            in.upper = 1.0e-3f;  // u >= ~0.25 >> upper -> clamped at upper edge
        } else {
            in.lower = 1.0e3f;   // u << lower -> clamped at lower edge
            in.upper = 2.0e3f;
        }

        // Confirm the row really is event-clamped (defensive against config drift).
        if (clamp_active(in) != 0.0f) {
            return 1.0e9f; // forces the caller's == 0 assertion to fail loudly
        }

        const Grads g = adjoint_eval(in, 1.0f);
        float analytic[kBoxPgsGradCount];
        Bind::Partials(g, analytic);
        for (int p = 0; p < kBoxPgsGradCount; ++p) {
            worst = std::fmax(worst, std::fabs(analytic[p]));
        }
    }
    return worst;
}

} // namespace

using nuka::solver::generated::FeatherstoneContactAdjGrads;
using nuka::solver::generated::FeatherstoneContactAdjInputs;
using nuka::solver::generated::MaximalContactAdjGrads;
using nuka::solver::generated::MaximalContactAdjInputs;
using nuka::solver::generated::MaximalJointAdjGrads;
using nuka::solver::generated::MaximalJointAdjInputs;

BoxPgsFdResult ValidateMaximalJointFd(const BoxPgsFdConfig& config) {
    using namespace nuka::solver::generated;
    return ValidateBoxPgs<MaximalJointAdjInputs, MaximalJointAdjGrads>(
        config, maximal_joint_forward_eval, maximal_joint_adjoint_eval,
        maximal_joint_clamp_active);
}

BoxPgsFdResult ValidateMaximalContactFd(const BoxPgsFdConfig& config) {
    using namespace nuka::solver::generated;
    return ValidateBoxPgs<MaximalContactAdjInputs, MaximalContactAdjGrads>(
        config, maximal_contact_forward_eval, maximal_contact_adjoint_eval,
        maximal_contact_clamp_active);
}

float MaxEventStoppedMaximalContactMagnitude(const BoxPgsFdConfig& config) {
    using namespace nuka::solver::generated;
    return MaxEventStoppedMagnitude<MaximalContactAdjInputs, MaximalContactAdjGrads>(
        config, maximal_contact_adjoint_eval, maximal_contact_clamp_active);
}

BoxPgsFdResult ValidateFeatherstoneContactFd(const BoxPgsFdConfig& config) {
    using namespace nuka::solver::generated;
    return ValidateBoxPgs<FeatherstoneContactAdjInputs, FeatherstoneContactAdjGrads>(
        config, featherstone_contact_forward_eval, featherstone_contact_adjoint_eval,
        featherstone_contact_clamp_active);
}

float MaxEventStoppedFeatherstoneContactMagnitude(const BoxPgsFdConfig& config) {
    using namespace nuka::solver::generated;
    return MaxEventStoppedMagnitude<FeatherstoneContactAdjInputs,
                                    FeatherstoneContactAdjGrads>(
        config, featherstone_contact_adjoint_eval, featherstone_contact_clamp_active);
}

} // namespace nuka::codegen::v3
