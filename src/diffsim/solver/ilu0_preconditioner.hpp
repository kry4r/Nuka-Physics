#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- ILU(0) preconditioner (symbolic + numeric) -- v0.7 p02
// ---------------------------------------------------------------------------
//
// ILU(0) = Incomplete LU with ZERO fill-in: factor A ~= L U where L (unit lower)
// and U (upper) are constrained to A's EXISTING sparsity pattern -- any update
// that would write into a structurally-zero position of A is DROPPED (the "zero
// fill-in" rule). On the batched-dense seam a "block" is a dense n x n tile with a
// fixed stride (kMaxBlockDim), but a block can still carry STRUCTURAL ZEROS (exact
// 0.0 entries): ILU(0) keeps L/U zero exactly where A is zero, so for a genuinely
// sparse-within-the-tile A it is a true INCOMPLETE factorization (L U != A). For a
// FULLY DENSE block (no structural zeros) ILU(0) degenerates to the COMPLETE LU
// factorization (nothing to drop) -- documented honestly; the preconditioner-
// quality test uses structurally-sparse blocks so the incompleteness is real.
//
// We store the combined factor in a single n x n (kMaxBlockDim-strided) array
// `lu`: the strict lower triangle holds L's off-diagonals (L has implicit unit
// diagonal), the diagonal + upper triangle holds U. This is the standard packed
// ILU storage; applying M^-1 = (L U)^-1 is a forward solve L y = r (unit diagonal)
// then a backward solve U z = y (see triangular_solve.cu).
//
// SYMBOLIC PHASE: the pattern IS A's pattern (zero fill-in) -- no separate symbolic
// data structure is needed on the dense seam; "in pattern" == "A[i][j] != 0". The
// sparsity mask is derived from the input block at factor time.
//
// NUMERIC PHASE (classic IKJ ILU(0), fixed loop order -> D1):
//   for i = 1..n-1:
//     for k = 0..i-1 with (i,k) in pattern:
//       lu[i][k] /= lu[k][k]                       # L multiplier
//       for j = k+1..n-1 with (i,j) in pattern:    # drop fill outside pattern
//         lu[i][j] -= lu[i][k] * lu[k][j]
//
// DETERMINISM: a single sequential fixed-order pass (lane 0 of the warp, or the
// host reference). No atomics, no parallel reduction, fp64 accumulation. Same
// input bytes => byte-identical factor.
//
// This header exposes (a) __device__ helpers the MINRES warp kernel calls (factor
// in shared, apply via triangular solves) and (b) a HOST reference factorization
// used by the factorization unit test as the known-good oracle.
// ---------------------------------------------------------------------------

#include <cstdint>

#include <cuda_runtime.h>  // cudaStream_t (Launch*Test signatures, BUF-14)

namespace nuka {
namespace diffsim {

// Relative pivot floor for ILU(0): a |U[k][k]| at or below kIluPivotRelEps *
// max|diag| is treated as a structurally-degenerate pivot and replaced by a safe
// signed unit so the division never blows up (mirrors the CG modified-Cholesky
// null-direction guard's RELATIVE-threshold philosophy). Fixed constant => the
// branch is a deterministic data-dependent compare => D1.
constexpr double kIluPivotRelEps = 1.0e-12;

// ---------------------------------------------------------------------------
// HOST reference ILU(0) factorization (TEST-ONLY oracle; never a production path).
//
// `a` is a row-major n x n dense block with stride `stride` (>= n). `lu_out` is
// written row-major with the SAME stride: strict-lower = L off-diagonals (unit
// diagonal implicit), diagonal+upper = U. Entries OUTSIDE the structural pattern of
// `a` (where a[i][j] == 0.0 exactly, i != j) are left 0.0 (zero fill-in). fp64.
//
// This is the SAME fixed-order IKJ recurrence the device path runs, so the device
// factor matches it byte-for-byte (both fp64); the unit test also checks it against
// a hand-computed reference on a small structured matrix.
// ---------------------------------------------------------------------------
void Ilu0FactorizeHost(const double* a, uint32_t n, uint32_t stride,
                       double* lu_out);

// HOST reference application of the ILU(0) preconditioner: solve (L U) z = r for z
// given the packed `lu` factor (forward L y = r with unit L diagonal, then backward
// U z = y). Row-major, stride `stride`. fp64. TEST-ONLY oracle.
void Ilu0ApplyHost(const double* lu, uint32_t n, uint32_t stride, const double* r,
                   double* z_out);

#ifdef __CUDACC__
// ---------------------------------------------------------------------------
// DEVICE ILU(0) factorization (lane 0 of a warp-per-block solve). Reads the dense
// block `a` (row-major, kMaxBlockDim stride, fp64-in-shared) and writes the packed
// LU factor into `lu` (same stride). FIXED IKJ loop order -> D1. Structural zeros
// of `a` are NOT filled (zero fill-in). The pivot guard (kIluPivotRelEps) keeps a
// degenerate diagonal finite. Call from lane 0 only, then __syncwarp before apply.
//
// `md` is the row stride (== kMaxBlockDim) so this header stays independent of the
// backend header's constant.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void Ilu0FactorizeDevice(const double* __restrict__ a,
                                                     uint32_t n, uint32_t md,
                                                     double* __restrict__ lu) {
    // Block scale for the relative pivot guard: largest |diagonal| (fixed order).
    double max_diag = 0.0;
    for (uint32_t d = 0u; d < n; ++d) {
        const double ad = a[d * md + d];
        const double abs_ad = (ad < 0.0) ? -ad : ad;
        if (abs_ad > max_diag) max_diag = abs_ad;
    }
    const double pivot_floor = kIluPivotRelEps * (max_diag > 0.0 ? max_diag : 1.0);

    // Copy A into LU (in pattern; structural zeros stay zero -> never filled).
    for (uint32_t i = 0u; i < n; ++i)
        for (uint32_t j = 0u; j < n; ++j) lu[i * md + j] = a[i * md + j];

    // Classic IKJ ILU(0): drop any update into a structurally-zero position.
    for (uint32_t i = 1u; i < n; ++i) {
        for (uint32_t k = 0u; k < i; ++k) {
            // (i,k) in pattern?  (structural zero -> no L multiplier here)
            if (a[i * md + k] == 0.0) continue;
            double ukk = lu[k * md + k];
            // Pivot guard: degenerate U diagonal -> safe signed unit (keep sign so
            // the indefinite structure is not silently flipped). Deterministic.
            if (ukk <= pivot_floor && ukk >= -pivot_floor) {
                ukk = (ukk < 0.0) ? -1.0 : 1.0;
                lu[k * md + k] = ukk;
            }
            const double lik = lu[i * md + k] / ukk;
            lu[i * md + k] = lik;
            for (uint32_t j = k + 1u; j < n; ++j) {
                // Drop fill outside A's pattern (zero fill-in rule).
                if (a[i * md + j] == 0.0) continue;
                lu[i * md + j] -= lik * lu[k * md + j];
            }
        }
        // Final pivot guard on U[i][i] (used by the backward solve / next rows).
        double uii = lu[i * md + i];
        if (uii <= pivot_floor && uii >= -pivot_floor) {
            lu[i * md + i] = (uii < 0.0) ? -1.0 : 1.0;
        }
    }
    // Guard the very first pivot too (i==0 had no k-loop above).
    {
        double u00 = lu[0];
        if (u00 <= pivot_floor && u00 >= -pivot_floor) {
            lu[0] = (u00 < 0.0) ? -1.0 : 1.0;
        }
    }
}
#endif  // __CUDACC__

// ---------------------------------------------------------------------------
// Batched DEVICE test launchers (host-callable, defined in the .cu). These run the
// SAME __device__ ILU(0) factor / apply as the MINRES backend so the device path
// can be unit-tested + two-run D1 byte-exact checked in isolation. Device pointers;
// the caller owns + synchronizes. `lu`/`a` are block-major (stride kMaxBlockDim^2,
// row-major n x n); `r`/`z` are block-major (stride kMaxBlockDim).
// ---------------------------------------------------------------------------
void LaunchIlu0FactorizeTest(cudaStream_t stream, int device_id, const float* a,
                             const uint32_t* block_dim, float* lu,
                             uint32_t block_count);

void LaunchIlu0ApplyTest(cudaStream_t stream, int device_id, const float* lu,
                         const uint32_t* block_dim, const float* r, float* z,
                         uint32_t block_count);

}  // namespace diffsim
}  // namespace nuka
