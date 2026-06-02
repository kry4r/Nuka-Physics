#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- SDF-contact IFT: the d/dM, d/dJ channel (v0.7 p08-C, the
//                  §4.B acceptance)
// ---------------------------------------------------------------------------
//
// WHAT THIS IS (and what the v0.5 ift_runner DEFERRED). The v0.5 IFT path
// (ift_runner.{hpp,cu}) ships grad_qdot_free and grad_b_c for the converged
// contact solve but explicitly DEFERS the d/dM and d/dJ channels (see the
// ift_runner.cu LIMITATIONS comment, "d(A^-1) = -A^-1 dA A^-1 + the M^-1 J^T
// product rule ... are DEFERRED"). v0.7 §4.B rules that the SDF-contact rows are
// THE differentiable contact path and that the deferred d/dM, d/dJ channels are
// OWNED HERE (the PGS path's d/dM,d/dJ are SUPERSEDED, not retrofitted). This
// header implements the system-level IFT-at-convergence of the GLOBAL constraint
// solve and FD-validates it by PERTURBING ACTUAL ENTRIES OF M / J AND RE-SOLVING
// THE CONSTRAINT SYSTEM TO CONVERGENCE (see tests/diffsim/test_sdf_contact_ift_fd.cpp).
//
// THE SYSTEM. At the converged active set the contact solve is the linear system
//
//      A lambda = r,      A = J M^-1 J^T  (SPD Delassus),   r = b_c - J qdot_free
//
// where J is the active-row chain Jacobian (sticking-equality rows, the
// kkt_builder active set), M^-1 the articulation inverse-inertia tile, b_c the
// per-row target separating velocity, and qdot_free the pre-solve velocity. The
// SDF-contact rows (id 4 / id 5) feed J, M^-1, b_c through this SAME contact
// solve (they emit RowKind::Contact rows resolved by the existing PGS path), so
// their d/dM, d/dJ ARE the system-IFT derivatives of this A lambda = r.
//
// THE IFT (differentiate A lambda = r at the converged lambda, A held SPD):
//
//      dA lambda + A dlambda = dr
//      => dlambda = A^-1 ( dr - dA lambda )                         (the IFT, one A^-1 apply)
//
// This is exactly ONE EXTRA A^-1 apply on the vector (dr - dA lambda) -- the same
// A^-1 the runner already forms (the BlockJacobi exact island solve / a dense
// LDLT here). The two channels differ ONLY in dA and dr:
//
//   * d/dM channel.   M^-1 is the tile the solve consumes; perturbing M^-1 by
//                     dMinv gives  dA = J dMinv J^T  and  dr = 0  (b_c, qdot_free,
//                     J held). (We perturb M^-1 directly -- the tile the Delassus
//                     contraction reads -- so there is NO -M^-1 dM M^-1 layer.
//                     d/dM follows by chaining dMinv = -M^-1 dM M^-1 if a caller
//                     ever wants the M-entry derivative; the FD perturbs M^-1.)
//
//   * d/dJ channel.   Perturbing J by dJ gives  dA = dJ M^-1 J^T + J M^-1 dJ^T
//                     (the product rule -- BOTH terms) and  dr = -dJ qdot_free
//                     (r = b_c - J qdot_free, so the dr term is NONZERO and must
//                     NOT be dropped; forgetting it makes the d/dJ FD fail).
//
// NO step_curvature. This file NEVER touches the forward's `step_curvature`
// (the 2/voxel descent constant). That scalar is a forward descent-step proxy
// ONLY; the along-valley curvature of the summed SDF is ~0 and feeding 2/voxel
// into any inversion manufactures a fictitious gradient. The d/dM, d/dJ channel
// is VALLEY-INDEPENDENT: it differentiates the SPD Delassus system A lambda = r,
// whose A = J M^-1 J^T has nothing to do with the witness-point curvature.
//
// HOST / DENSE. This is a host (CUDA-free) dense reference implementation: the
// active block is tiny (n <= kMaxBlockDim = 12, dof <= 18). It is the analytical
// oracle the §4.B FD validates against, and the structure the GPU runner mirrors
// (the GPU ift_runner already owns the A^-1 apply; this adds the dA/dr assembly).
// fp64 throughout (the analytical reference; D1 byte-exactness is gated on the
// GPU runner's existing fp64-accumulate / fp32-store path, unchanged here).
// ---------------------------------------------------------------------------

#include <cstdint>
#include <vector>

namespace nuka::diffsim {

// A converged active-set contact system, dense, ROW-MAJOR, fp64. `n` active rows
// (normal + 2 friction per engaged slot, kkt order) over `dof` articulation DOFs.
//   j         : n*dof   active chain-Jacobian rows
//   m_inv     : dof*dof inverse-inertia tile (the tile the contraction consumes)
//   b_c       : n       per-row target separating velocity
//   qdot_free : dof     pre-solve velocity
//   lambda    : n       the CONVERGED impulses (A lambda = r). If empty, solved here.
struct SdfContactDenseSystem {
    uint32_t n = 0u;     // active rows
    uint32_t dof = 0u;   // articulation DOFs
    std::vector<double> j;          // n * dof, row-major
    std::vector<double> m_inv;      // dof * dof, row-major (symmetric)
    std::vector<double> b_c;        // n
    std::vector<double> qdot_free;  // dof
    std::vector<double> lambda;     // n (converged); filled by SolveLambda if empty
};

// Assemble A = J M^-1 J^T (n x n, row-major) from the system's J, M^-1.
std::vector<double> AssembleDelassus(const SdfContactDenseSystem& s);

// r = b_c - J qdot_free (n).
std::vector<double> AssembleRhs(const SdfContactDenseSystem& s);

// Solve A lambda = r to convergence (dense Cholesky / LDLT). Used both to obtain
// the converged lambda the IFT linearizes around AND, in the FD test, to RE-SOLVE
// a perturbed system. `a` is n*n row-major SPD; returns the n-vector lambda.
std::vector<double> SolveSpd(const std::vector<double>& a, const std::vector<double>& r,
                             uint32_t n);

// Convenience: solve the system's own A lambda = r (fills/returns lambda).
std::vector<double> SolveLambda(const SdfContactDenseSystem& s);

// ---------------------------------------------------------------------------
// THE d/dM channel.  Directional derivative dlambda for a perturbation dMinv of
// the inverse-inertia tile (dof*dof, row-major). dA = J dMinv J^T, dr = 0, so
//      dlambda = A^-1 ( - (J dMinv J^T) lambda ).
// `lambda` must be the converged impulses (s.lambda, or SolveLambda(s)).
// ---------------------------------------------------------------------------
std::vector<double> DLambda_dMinv(const SdfContactDenseSystem& s,
                                  const std::vector<double>& dminv);

// ---------------------------------------------------------------------------
// THE d/dJ channel.  Directional derivative dlambda for a perturbation dJ of the
// active chain-Jacobian (n*dof, row-major). BOTH the product-rule dA terms AND
// the dr term:
//      dA = dJ M^-1 J^T + J M^-1 dJ^T
//      dr = - dJ qdot_free                     (NOT zero -- r = b_c - J qdot_free)
//      dlambda = A^-1 ( dr - dA lambda ).
// ---------------------------------------------------------------------------
std::vector<double> DLambda_dJ(const SdfContactDenseSystem& s,
                               const std::vector<double>& dj);

}  // namespace nuka::diffsim
