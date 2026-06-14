// ---------------------------------------------------------------------------
// nuka::diffsim -- self-written deterministic MINRES backend kernel (v0.7 p02)
// ---------------------------------------------------------------------------
//
// Warp-per-block preconditioned MINRES over batched-dense SYMMETRIC (possibly
// INDEFINITE) blocks. See minres_backend.hpp for the design contract. The structure
// mirrors sparse_solver_cg.cu exactly (one warp per block, lane L owns row L + the
// L-th component of every vector, fp64 internal, fixed-order WarpButterflySum
// reductions, fixed-order SpMV gather, NO atomics / NO cuBLAS) so the D1 byte-exact
// guarantee + the conventions carry over unchanged.
//
// ALGORITHM (Paige & Saunders 1975, preconditioned MINRES with SPD preconditioner
// M; here M = absolute-Jacobi or ILU(0)). Per block, all scalars fixed-order fp64:
//
//   r0 = b - A x0   (x0 = 0 => r0 = b)
//   z0 = M^-1 r0 ;  beta1 = sqrt(<r0, z0>)        (M-inner-product norm)
//   v1 = r0 / beta1 (in the A-space) ; pz = z0 / beta1 (the M^-1-applied direction)
//   Lanczos: each step k generates v_{k+1}, the tridiagonal alpha_k / beta_{k+1};
//   Givens rotations reduce the tridiagonal to upper-triangular R; the solution is
//   updated x += tau_k * w_k with w_k the rotated search direction. ||r|| is tracked
//   by the running product phi_k. We use the standard "Algorithm 4" variable names.
//
// Determinism: every <.,.> and the beta = sqrt(<.,.>) use the fixed-order butterfly;
// the Givens c/s and the recurrence are scalar fp64 in fixed order; the per-lane
// update x_i += tau * w_i is independent. Same input + same GPU => byte-identical x.
// ---------------------------------------------------------------------------

#include "diffsim/solver/ilu0_preconditioner.hpp"
#include "diffsim/solver/minres_backend.hpp"
#include "diffsim/solver/triangular_solve.cuh"
#include "diffsim/sparse_solver_backend.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::diffsim {

namespace {

constexpr uint32_t kWarpSize = 32u;
constexpr uint32_t kBlockStride = kMaxBlockDim * kMaxBlockDim;  // 144 floats/block

// Deterministic floor: once a Lanczos beta (the M-norm of the new residual) drops
// to/below this the Krylov space is exhausted (converged or breakdown) -> freeze
// the iteration (no division by ~0). Fixed constant => same input => same break.
__device__ constexpr double kBetaFloor = 1.0e-30;

// Fixed-order warp-shuffle butterfly sum (identical to the CG backend). fp64, full
// mask, steps 16..1, lane-0 result broadcast to all lanes -> bit-exact.
__device__ __forceinline__ double WarpButterflySum(double v) {
    constexpr unsigned kFull = 0xffffffffu;
#pragma unroll
    for (uint32_t offset = kWarpSize / 2u; offset > 0u; offset >>= 1u) {
        v += __shfl_down_sync(kFull, v, offset);
    }
    return __shfl_sync(kFull, v, 0);
}

// One warp solves block `blk` with preconditioned MINRES. Lane `lane` owns matrix
// row `lane` and the `lane`-th component of every vector. Lanes >= n are inactive
// (carry reduction identities). `precond` selects absolute-Jacobi (Preconditioner::
// Jacobi) or ILU(0) (Preconditioner::BlockJacobi, reinterpreted -- see header).
__device__ __forceinline__ void SolveOneBlockMinres(
    const float* __restrict__ values, uint32_t n, const float* __restrict__ b_blk,
    float* __restrict__ x_blk, Preconditioner precond, uint32_t max_iter, double tol,
    bool run_to_fixed_iters, double* __restrict__ lu_shared,
    double* __restrict__ tri_scratch, uint32_t lane) {
    const bool active = (lane < n);

    // --- Load this lane's matrix row into registers (fp64). ------------------
    double row[kMaxBlockDim];
#pragma unroll
    for (uint32_t c = 0u; c < kMaxBlockDim; ++c) {
        double aij = 0.0;
        if (active && c < n) aij = static_cast<double>(values[lane * kMaxBlockDim + c]);
        row[c] = aij;
    }
    const double a_diag = active ? row[lane] : 0.0;
    const double b_i = active ? static_cast<double>(b_blk[lane]) : 0.0;

    // --- Optional ILU(0) factor (BlockJacobi key). All lanes stage the fp64 A tile
    //     into the [kBlockStride, 2*kBlockStride) half of lu_shared (coalesced),
    //     then lane 0 factors it (fixed IKJ order) into the [0, kBlockStride) half
    //     and publishes the packed L/U to the warp. The A-source and factor-target
    //     regions are DISJOINT so the in-shared factor never overwrites its input.
    //     For absolute-Jacobi this whole block is skipped. -----------------------
    double* a_view = lu_shared + kBlockStride;  // A staging (ILU(0) only)
    if (precond == Preconditioner::BlockJacobi) {
        for (uint32_t idx = lane; idx < kBlockStride; idx += kWarpSize) {
            const uint32_t r = idx / kMaxBlockDim;
            const uint32_t c = idx % kMaxBlockDim;
            a_view[idx] = (r < n && c < n)
                              ? static_cast<double>(values[r * kMaxBlockDim + c])
                              : 0.0;
        }
        __syncwarp(0xffffffffu);
        if (lane == 0u) Ilu0FactorizeDevice(a_view, n, kMaxBlockDim, lu_shared);
        __syncwarp(0xffffffffu);
    }

    // Applies the preconditioner z = M^-1 r and returns this lane's z_i. ABSOLUTE-
    // Jacobi: z_i = r_i / max(|a_ii|, floor) (an SPD preconditioner for an indefinite
    // A -- the standard choice). ILU(0): z = (L U)^-1 r via lane-0 forward/back
    // substitution on the cached factor, scattered back through tri_scratch.
    auto apply_precond = [&](double r_i) -> double {
        if (precond == Preconditioner::Jacobi) {
            if (!active) return 0.0;
            double d = (a_diag < 0.0) ? -a_diag : a_diag;  // |diag|
            if (d <= kBetaFloor) d = 1.0;
            return r_i / d;
        }
        // ILU(0): stage r into tri_scratch, lane 0 solves L U z = r, read back.
        if (active) tri_scratch[lane] = r_i;
        __syncwarp(0xffffffffu);
        if (lane == 0u) Ilu0ApplyDevice(lu_shared, n, kMaxBlockDim, tri_scratch);
        __syncwarp(0xffffffffu);
        const double z_i = active ? tri_scratch[lane] : 0.0;
        __syncwarp(0xffffffffu);
        return z_i;
    };

    // SpMV: (A p)_i = dot(row_i, p) over the warp (fixed k order). Identical to CG.
    auto spmv = [&](const double p_self) -> double {
        double acc = 0.0;
#pragma unroll
        for (uint32_t k = 0u; k < kMaxBlockDim; ++k) {
            const double p_k = __shfl_sync(0xffffffffu, p_self, k);
            acc += row[k] * p_k;
        }
        return active ? acc : 0.0;
    };

    // --- Preconditioned MINRES (faithful transcription of the SOL/Stanford
    //     minres.m, Paige & Saunders 1975). x0 = 0 => r0 = b. All scalars are
    //     fixed-order fp64; r1/r2 (un-preconditioned Lanczos residual vectors) and
    //     y (preconditioned) + w/w1/w2 (search directions) are DISTRIBUTED (lane i
    //     owns component i). Variable names match the reference 1:1. -------------
    //
    //   y     = M^-1 b ;  beta1 = sqrt(<b, y>)        (preconditioned M-norm of r0)
    //   r1 = r2 = b ;  x = 0 ;  w = w2 = 0
    //   oldb=0; beta=beta1; dbar=0; epsln=0; phibar=beta1; cs=-1; sn=0
    //   each step k:
    //     v   = (1/beta) y                            (M-orthonormal Lanczos vector)
    //     y   = A v
    //     if k>=2: y -= (beta/oldb) r1
    //     alfa = <v, y>
    //     y   -= (alfa/beta) r2
    //     r1 = r2 ; r2 = y ; y = M^-1 r2
    //     oldb = beta ; beta = sqrt(<r2, y>)
    //     (previous rotation) oldeps=epsln; delta=cs*dbar+sn*alfa;
    //         gbar=sn*dbar-cs*alfa; epsln=sn*beta; dbar=-cs*beta
    //     (new rotation)      gamma=norm([gbar beta]); cs=gbar/gamma; sn=beta/gamma
    //         phi=cs*phibar; phibar=sn*phibar
    //     (update x) w1=w2; w2=w; w=(v - oldeps*w1 - delta*w2)/gamma; x += phi*w
    //   phibar tracks ||r_k|| (the MINRES residual estimate) -> the stop test.
    double r1_i = b_i;
    double r2_i = b_i;
    double y2_i = apply_precond(b_i);                 // y = M^-1 b
    double beta1_sq = WarpButterflySum(b_i * y2_i);
    double beta1 = sqrt(beta1_sq > 0.0 ? beta1_sq : 0.0);  // M-norm of r0; <b,M^-1 b>
                                                           // < 0 would mean M not SPD

    const double stop = tol * (beta1 > kBetaFloor ? beta1 : 1.0);

    double oldb = 0.0;
    double beta = beta1;
    double dbar = 0.0;
    double epsln = 0.0;
    double phibar = beta1;
    double cs = -1.0;
    double sn = 0.0;

    double w_i = 0.0;
    double w2_i = 0.0;
    double x_i = 0.0;

    bool converged = (beta1 <= kBetaFloor);  // zero rhs => x = 0 exact

    for (uint32_t iter = 0u; iter < max_iter && !converged; ++iter) {
        // v = (1/beta) y2   (y2 holds M^-1 r2; this is the M-orthonormal vector)
        const double inv_beta = (beta > kBetaFloor) ? (1.0 / beta) : 0.0;
        const double v_i = inv_beta * y2_i;

        // y = A v
        double yy_i = spmv(v_i);
        // if k>=2: y -= (beta/oldb) r1
        if (iter >= 1u && oldb > kBetaFloor) {
            yy_i -= (beta / oldb) * r1_i;
        }
        // alfa = <v, y>
        const double alfa = WarpButterflySum(v_i * yy_i);
        // y -= (alfa/beta) r2
        if (beta > kBetaFloor) yy_i -= (alfa / beta) * r2_i;

        // r1 = r2 ; r2 = y ; y2 = M^-1 r2
        r1_i = r2_i;
        r2_i = yy_i;
        y2_i = apply_precond(r2_i);
        oldb = beta;
        double beta_sq = WarpButterflySum(r2_i * y2_i);
        beta = sqrt(beta_sq > 0.0 ? beta_sq : 0.0);  // M-norm of the new residual

        // Apply the previous plane rotation Q_{k-1}.
        const double oldeps = epsln;
        const double delta = cs * dbar + sn * alfa;
        const double gbar = sn * dbar - cs * alfa;
        epsln = sn * beta;
        dbar = -cs * beta;

        // Compute the next plane rotation Q_k.
        double gamma = sqrt(gbar * gbar + beta * beta);
        if (gamma <= kBetaFloor) gamma = kBetaFloor;
        cs = gbar / gamma;
        sn = beta / gamma;
        const double phi = cs * phibar;
        phibar = sn * phibar;  // == |sn| * phibar; tracks ||r_k||

        // Update x: w = (v - oldeps*w1 - delta*w2) / gamma ; x += phi*w
        const double w1_i = w2_i;
        w2_i = w_i;
        w_i = (v_i - oldeps * w1_i - delta * w2_i) / gamma;
        x_i += phi * w_i;

        // Convergence: phibar is the running ||r_k|| estimate.
        if (!run_to_fixed_iters && (phibar <= stop || phibar <= kBetaFloor)) {
            converged = true;
            break;
        }
        // Lanczos breakdown: beta ~ 0 -> Krylov space exhausted, x is the min-resid
        // solution; freeze (the remaining fixed iters are no-ops).
        if (beta <= kBetaFloor) {
            converged = true;
            break;
        }
    }

    // --- Write back (capped at kMaxBlockDim; padding lanes write 0). -------------
    if (lane < kMaxBlockDim) {
        x_blk[lane] = active ? static_cast<float>(x_i) : 0.0f;
    }
}

__global__ void SolveMinresKernel(const float* __restrict__ values,
                                  const uint32_t* __restrict__ block_dim,
                                  const float* __restrict__ b, float* __restrict__ x,
                                  uint32_t block_count, Preconditioner precond,
                                  uint32_t max_iter, double tol,
                                  bool run_to_fixed_iters) {
    const uint32_t blk = blockIdx.x;
    if (blk >= block_count) return;
    const uint32_t lane = threadIdx.x;

    // Shared scratch: lu_shared needs [0,kBlockStride) for the ILU(0) factor +
    // [kBlockStride, 2*kBlockStride) for the fp64 A staging (ILU(0) path only).
    // tri_scratch[kMaxBlockDim] is the triangular-solve rhs/z staging.
    __shared__ double lu_shared[2u * kBlockStride];
    __shared__ double tri_scratch[kMaxBlockDim];

    const uint32_t n = block_dim[blk];
    const float* __restrict__ values_blk =
        values + static_cast<size_t>(blk) * kBlockStride;
    const float* __restrict__ b_blk = b + static_cast<size_t>(blk) * kMaxBlockDim;
    float* __restrict__ x_blk = x + static_cast<size_t>(blk) * kMaxBlockDim;

    if (n == 0u) {
        if (lane < kMaxBlockDim) x_blk[lane] = 0.0f;
        return;
    }

    SolveOneBlockMinres(values_blk, n, b_blk, x_blk, precond, max_iter, tol,
                        run_to_fixed_iters, lu_shared, tri_scratch, lane);
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string("nuka::diffsim::SelfWrittenMinresBackend ") + operation +
            ": " + cudaGetErrorString(result));
    }
}

}  // namespace

SelfWrittenMinresBackend::SelfWrittenMinresBackend(cudaStream_t stream, int device_id,
                                                   DeterminismLevel determinism)
    : stream_(stream), device_id_(device_id), determinism_(determinism) {}

void SelfWrittenMinresBackend::Solve(const BatchedDenseSpdSystem& system,
                                     const float* b, float* x,
                                     const SolveParams& params) {
    if (system.block_count == 0u) return;
    if (system.values == nullptr || system.block_dim == nullptr || b == nullptr ||
        x == nullptr) {
        throw std::invalid_argument(
            "nuka::diffsim::SelfWrittenMinresBackend::Solve: null system/vector "
            "pointer");
    }
    phi::ScopedDeviceGuard guard(device_id_);
    const cudaStream_t stream = stream_;

    SolveMinresKernel<<<system.block_count, kWarpSize, 0u, stream>>>(
        system.values, system.block_dim, b, x, system.block_count,
        params.preconditioner, params.max_iter, static_cast<double>(params.tol),
        params.run_to_fixed_iters);
    CheckCuda(cudaGetLastError(), "SolveMinresKernel launch");
}

}  // namespace nuka::diffsim
