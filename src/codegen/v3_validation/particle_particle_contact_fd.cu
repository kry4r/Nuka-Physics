// V3 finite-difference validator for the ParticleParticleContactRow reverse-mode
// adjoint (v0.7 p11, K3 cross-system coupling infrastructure). Central-differences
// the GENERATED primal forward-update and compares to the GENERATED analytical
// adjoint (seed v=1), over a batch of ACTIVE penetrating contacts (C < 0) sampled
// strictly inside the unilateral box [0, +huge) so clamp_active is always 1.
//
// The dispatchable adjoint is the IDENTICAL multilinear XPBD multiplier law as the
// distance row (only the geometric grad C / the unilateral box differ), so this
// mirrors xpbd_distance_fd exactly apart from the active-contact sampling regime.

#include "codegen/v3_validation/particle_particle_contact_fd.hpp"

#include "codegen/generated/particle_particle_contact_adjoint.cuh"

#include <cmath>
#include <cstdint>
#include <random>

namespace nuka::codegen::v3 {

namespace {

using nuka::solver::generated::ParticleParticleContactAdjGrads;
using nuka::solver::generated::ParticleParticleContactAdjInputs;
using nuka::solver::generated::particle_particle_contact_adjoint_eval;
using nuka::solver::generated::particle_particle_contact_clamp_active;
using nuka::solver::generated::particle_particle_contact_forward_eval;

// Four differentiable fields: C, lambda, effective_mass, alpha_tilde.
inline constexpr int kPpcGradCount = 4;

float SampleRange(std::mt19937_64& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

double RelError(double analytic, double numeric) {
    constexpr double kEps = 1.0e-6;
    return std::fabs(analytic - numeric) /
           (std::fabs(analytic) + std::fabs(numeric) + kEps);
}

void Partials(const ParticleParticleContactAdjGrads& g, float out[kPpcGradCount]) {
    out[0] = g.grad_C;
    out[1] = g.grad_lambda;
    out[2] = g.grad_effective_mass;
    out[3] = g.grad_alpha_tilde;
}

ParticleParticleContactAdjInputs Perturb(const ParticleParticleContactAdjInputs& base,
                                         int idx, float delta) {
    ParticleParticleContactAdjInputs in = base;
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

ParticleParticleContactFdResult
ValidateParticleParticleContactFd(const ParticleParticleContactFdConfig& config) {
    std::mt19937_64 rng(config.seed);
    ParticleParticleContactFdResult result;
    result.case_count = config.num_cases;

    for (uint32_t i = 0; i < config.num_cases; ++i) {
        ParticleParticleContactAdjInputs in;
        // ACTIVE contact: C strongly negative (penetrating), |C| in [pen_min,
        // pen_max], so u = effective_mass*(-C)+... is comfortably positive and
        // the row stays inside the unilateral box under +/- fd_delta.
        in.C = -SampleRange(rng, config.pen_min, config.pen_max);
        // lambda small + POSITIVE (a contact multiplier is non-negative -- a
        // unilateral contact pushes, never sticks; warm-started at 0). Positive
        // lambda keeps u = lambda + effective_mass*(-C - a~*lambda) strictly inside
        // the box [0, +huge) (both the lambda and the dominant effective_mass*(-C)
        // terms are positive). A signed lambda -- valid for the equality distance
        // row -- could drive u below 0 here (the lower active-set edge), which is a
        // non-smooth point the contact row deliberately does not sample. lambda is
        // kept away from zero so grad_alpha_tilde = -effective_mass*lambda stays
        // well-conditioned against the f32 FD cancellation floor.
        in.lambda = SampleRange(rng, config.lambda_min, config.lambda_max);
        in.effective_mass = SampleRange(rng, config.meff_min, config.meff_max);
        in.alpha_tilde = SampleRange(rng, config.alpha_tilde_min, config.alpha_tilde_max);
        in.lower = config.lower;
        in.upper = config.upper;

        // Unilateral contact row: the active penetrating contact must be strictly
        // inside the box [0, +huge). A regression in the active-contact sampling
        // surfaces here rather than as a misleading zero slope at the edge.
        if (particle_particle_contact_clamp_active(in) != 1.0f) {
            result.max_rel_err = 1.0f;
            result.worst_case_index = i;
            continue;
        }

        const ParticleParticleContactAdjGrads g =
            particle_particle_contact_adjoint_eval(in, 1.0f);
        float analytic[kPpcGradCount];
        Partials(g, analytic);

        const double delta = static_cast<double>(config.fd_delta);
        for (int p = 0; p < kPpcGradCount; ++p) {
            const ParticleParticleContactAdjInputs plus =
                Perturb(in, p, static_cast<float>(delta));
            const ParticleParticleContactAdjInputs minus =
                Perturb(in, p, static_cast<float>(-delta));
            const double numeric =
                (static_cast<double>(particle_particle_contact_forward_eval(plus)) -
                 static_cast<double>(particle_particle_contact_forward_eval(minus))) /
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
