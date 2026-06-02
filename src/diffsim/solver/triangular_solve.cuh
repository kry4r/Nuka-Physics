#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- sparse triangular solve (device inline helpers) -- v0.7 p02
// ---------------------------------------------------------------------------
//
// Forward / backward substitution for the ILU(0) preconditioner apply. On the
// batched-dense seam a "block" is a dense n x n tile (kMaxBlockDim stride) whose
// STRUCTURAL ZEROS are honored: the substitution skips structurally-zero factor
// entries (the "sparse" in sparse triangular solve), so for a genuinely sparse-
// within-the-tile factor the work matches the nonzero pattern.
//
// The packed ILU(0) factor `lu` stores (see ilu0_preconditioner.hpp): strict lower
// = L off-diagonals (L has IMPLICIT UNIT diagonal), diagonal + upper = U.
//
//   Forward  : solve L y = r,  y_i = r_i - sum_{k<i} L[i][k] y_k   (unit L diag)
//   Backward : solve U z = y,  z_i = (y_i - sum_{k>i} U[i][k] z_k) / U[i][i]
//
// DETERMINISM: a FIXED level-schedule order. On the dense per-block seam the
// "levels" are simply the natural row order processed SEQUENTIALLY by lane 0 of the
// warp (within a level rows are independent; here every row is its own level along
// the dependency chain, which is the fixed natural order). NO atomics, NO parallel
// reduction inside a block, fp64 accumulation in fixed k-order -> byte-exact. Across
// blocks the warps are independent (that IS the across-level parallelism the spec
// describes; it needs no nondeterministic level assignment because the partition is
// fixed by the block structure).
// ---------------------------------------------------------------------------

#include <cstdint>

#include <cuda_runtime.h>

namespace nuka::diffsim {

// Forward substitution L y = rhs (unit lower diagonal), in place on `rhs` -> y.
// Skips structurally-zero L entries. Fixed (i ascending, k ascending) order. fp64.
__device__ __forceinline__ void TriSolveLowerUnitDevice(const double* __restrict__ lu,
                                                        uint32_t n, uint32_t md,
                                                        double* __restrict__ rhs) {
    for (uint32_t i = 0u; i < n; ++i) {
        double s = rhs[i];
        for (uint32_t k = 0u; k < i; ++k) {
            const double lik = lu[i * md + k];
            if (lik == 0.0) continue;  // structural zero -> skip
            s -= lik * rhs[k];
        }
        rhs[i] = s;  // unit diagonal: no divide
    }
}

// Backward substitution U z = rhs, in place on `rhs` -> z. Skips structurally-zero
// U entries. Fixed (i descending, k ascending over k>i) order. fp64. U[i][i] is
// pivot-guarded by the ILU(0) factorization so the divide is finite.
__device__ __forceinline__ void TriSolveUpperDevice(const double* __restrict__ lu,
                                                    uint32_t n, uint32_t md,
                                                    double* __restrict__ rhs) {
    for (uint32_t ii = 0u; ii < n; ++ii) {
        const uint32_t i = n - 1u - ii;
        double s = rhs[i];
        for (uint32_t k = i + 1u; k < n; ++k) {
            const double uik = lu[i * md + k];
            if (uik == 0.0) continue;  // structural zero -> skip
            s -= uik * rhs[k];
        }
        rhs[i] = s / lu[i * md + i];
    }
}

// Full ILU(0) apply on the device: z = (L U)^-1 r. Forward then backward, in place
// on `scratch` (which must already hold r on entry; on exit holds z). Lane-0 call.
__device__ __forceinline__ void Ilu0ApplyDevice(const double* __restrict__ lu,
                                                uint32_t n, uint32_t md,
                                                double* __restrict__ scratch) {
    TriSolveLowerUnitDevice(lu, n, md, scratch);
    TriSolveUpperDevice(lu, n, md, scratch);
}

}  // namespace nuka::diffsim
