// Central-difference numerical Jacobian of the maximal_drive primal law.
//
// Differences the generated __host__ __device__ primal forward-update
// (maximal_drive_forward_eval) wrt each differentiable input/param, in double
// precision for headroom, so the V3 harness can compare it against the
// generated analytical adjoint. This file deliberately depends ONLY on the
// generated header (the single source of truth for the law).

#include "codegen/v3_validation/numerical_jacobian.cuh"

#include "codegen/generated/maximal_drive_adjoint.cuh"

namespace nuka::codegen::v3 {

using nuka::solver::generated::MaximalDriveAdjInputs;
using nuka::solver::generated::maximal_drive_forward_eval;

namespace {

// Evaluate the generated primal with x[field] perturbed by `delta` (double in,
// rounded into the float law the engine actually runs).
double EvalPrimalPerturbed(const MaximalDriveAdjInputs& base, int field, double delta) {
    MaximalDriveAdjInputs in = base;
    switch (field) {
    case 0: in.q += static_cast<float>(delta); break;
    case 1: in.qdot += static_cast<float>(delta); break;
    case 2: in.drive_target += static_cast<float>(delta); break;
    case 3: in.drive_stiffness += static_cast<float>(delta); break;
    case 4: in.drive_damping += static_cast<float>(delta); break;
    default: break;
    }
    return static_cast<double>(maximal_drive_forward_eval(in));
}

} // namespace

void NumericalJacobianMaximalDrive(const MaximalDriveAdjInputs& base,
                                   double delta,
                                   double out_partials[kMaximalDriveGradCount]) {
    for (int field = 0; field < kMaximalDriveGradCount; ++field) {
        const double plus = EvalPrimalPerturbed(base, field, delta);
        const double minus = EvalPrimalPerturbed(base, field, -delta);
        out_partials[field] = (plus - minus) / (2.0 * delta);
    }
}

} // namespace nuka::codegen::v3
