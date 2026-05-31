// Central-difference numerical Jacobian declarations for the V3 FD harness.

#pragma once

#include "codegen/generated/maximal_drive_adjoint.cuh"

namespace nuka::codegen::v3 {

// Number of differentiable inputs/params for the maximal_drive law
// (q, qdot, drive_target, drive_stiffness, drive_damping).
inline constexpr int kMaximalDriveGradCount = 5;

// Central-difference ∂tau/∂x_i for each of the kMaximalDriveGradCount inputs of
// the generated primal forward-update. Computed in double for headroom.
void NumericalJacobianMaximalDrive(
    const nuka::solver::generated::MaximalDriveAdjInputs& base,
    double delta,
    double out_partials[kMaximalDriveGradCount]);

} // namespace nuka::codegen::v3
