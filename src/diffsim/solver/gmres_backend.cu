// ---------------------------------------------------------------------------
// nuka::diffsim -- self-written deterministic GMRES(m) backend kernel (v0.7 p03-A)
// ---------------------------------------------------------------------------
//
// Warp-per-block restarted GMRES(m) over batched-dense GENERAL (NON-SYMMETRIC)
// blocks. See gmres_backend.hpp for the design contract. The warp/lane layout, the
// fp64 row load, the fixed-order SpMV gather, and the WarpButterflySum reductions
// are IDENTICAL to sparse_solver_cg.cu / minres_backend.cu (so the D1 byte-exact
// guarantee + the < 1e-5 agreement carry over). The ONE structural difference from
// the CG/MINRES short recurrences: GMRES retains the FULL Krylov basis, so the basis
// V and the Hessenberg/Givens scalars live in SHARED memory (not per-lane registers
// -- m=30 would spill catastrophically). The basis is warp-DISTRIBUTED (V_sh[k*md +
// L] = component L of basis vector k); lane 0 runs the sequential Arnoldi-Hessenberg
// Givens rotations + the least-squares back-substitution on the warp-uniform H; the
// whole warp cooperates on the butterfly dots and the per-lane x += V y update.
//
// ALGORITHM (Saad & Schultz 1986, restarted GMRES(m) with LEFT preconditioning M):
//   outer (restart) cycle, x carried across cycles:
//     r0 = M^-1 (b - A x)                       (true residual, recomputed per cycle)
//     beta = ||r0|| ; v_0 = r0 / beta ; g[0] = beta
//     for j = 0..m-1 (Arnoldi, MODIFIED Gram-Schmidt):
//       w = M^-1 (A v_j)
//       for i = 0..j:  h[i][j] = <w, v_i> ; w -= h[i][j] v_i   (MGS, fixed i order)
//       h[j+1][j] = ||w||
//       if h[j+1][j] <= floor: HAPPY BREAKDOWN (exact in this subspace) -> stop adding
//       else v_{j+1} = w / h[j+1][j]
//       apply the j prior Givens rotations to column j of H, then form rotation j to
//         zero h[j+1][j]; rotate g; |g[j+1]| is the running ||r|| estimate
//       if |g[j+1]| <= tol*||b|| : converged -> break the inner loop
//     solve the (k x k) upper-triangular least squares R y = g[0..k-1] (back-sub)
//     x += sum_{i<k} y_i v_i
//   repeat until converged or max outer cycles (max_iter / m, >= 1)
//
// DETERMINISM: every <.,.> / ||.|| is the fixed-order butterfly; MGS subtracts in
// fixed i order; the Givens c/s, g-rotation, and back-substitution are fp64 scalar
// ops on lane 0 in fixed loop order; x += V y is per-lane. Same input + same GPU =>
// byte-identical x. fp64 internal (loads fp32->fp64, stores fp64->fp32).
// ---------------------------------------------------------------------------

#include "diffsim/solver/gmres_backend.hpp"
#include "diffsim/sparse_solver_backend.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::diffsim {

namespace {

constexpr uint32_t kWarpSize = 32u;
constexpr uint32_t kBlockStride = kMaxBlockDim * kMaxBlockDim;  // 144 floats/block

// Compile-time cap on the restart length m. The runtime restart_ is clamped to this
// so the shared-memory basis (m+1 vectors) + Hessenberg are fixed-size. m=50 is the
// top of Saad's recommended 30-50 band; for n<=12 blocks any m>=n converges in one
// cycle, smaller m exercises the restart path (the test uses m=4).
constexpr uint32_t kMaxRestart = 50u;

// Deterministic floor mirroring MINRES's kBetaFloor: an Arnoldi norm (||w|| after
// orthogonalization, or the initial ||r0||) at/below this is a HAPPY BREAKDOWN (the
// Krylov space is exhausted -- the current subspace already contains the exact
// solution) -> stop extending the basis, solve the least squares with what we have,
// and freeze. Also guards every division by a norm. Fixed constant => same input =>
// same break => D1.
__device__ constexpr double kNormFloor = 1.0e-30;

// Fixed-order warp-shuffle butterfly sum (identical to CG/MINRES). fp64, full mask,
// steps 16..1, lane-0 result broadcast to all lanes -> bit-exact across lanes + runs.
__device__ __forceinline__ double WarpButterflySum(double v) {
    constexpr unsigned kFull = 0xffffffffu;
#pragma unroll
    for (uint32_t offset = kWarpSize / 2u; offset > 0u; offset >>= 1u) {
        v += __shfl_down_sync(kFull, v, offset);
    }
    return __shfl_sync(kFull, v, 0);
}

// One warp solves block `blk` with restarted GMRES(m). Lane `lane` owns matrix row
// `lane` and the `lane`-th component of every vector. Lanes >= n are inactive (carry
// reduction identities, write 0 padding). Shared scratch (kernel-allocated):
//   v_sh   [(kMaxRestart+1) * kMaxBlockDim] -- the warp-distributed Krylov basis:
//          v_sh[k*kMaxBlockDim + L] = component L of basis vector v_k.
//   h_sh   [(kMaxRestart+1) * kMaxRestart] -- upper-Hessenberg (col-major: column j
//          at h_sh[j*(kMaxRestart+1) + i]); lane-0-owned after the dots publish.
//   gv_sh  [kMaxRestart+1] -- the rotated rhs g (Givens-applied beta e1); lane 0.
//   cs_sh,sn_sh [kMaxRestart] -- the Givens cosines/sines; lane 0.
//   y_sh   [kMaxRestart] -- the least-squares solution coefficients; lane 0 -> warp.
__device__ __forceinline__ void SolveOneBlockGmres(
    const float* __restrict__ values, uint32_t n, const float* __restrict__ b_blk,
    float* __restrict__ x_blk, Preconditioner precond, uint32_t restart,
    uint32_t max_iter, double tol, bool run_to_fixed_iters,
    double* __restrict__ v_sh, double* __restrict__ h_sh, double* __restrict__ gv_sh,
    double* __restrict__ cs_sh, double* __restrict__ sn_sh,
    double* __restrict__ y_sh, uint32_t lane) {
    const bool active = (lane < n);
    const uint32_t md = kMaxBlockDim;
    const uint32_t hrows = kMaxRestart + 1u;  // Hessenberg column stride

    // --- Load this lane's matrix row into registers (fp64). -------------------
    double row[kMaxBlockDim];
#pragma unroll
    for (uint32_t c = 0u; c < kMaxBlockDim; ++c) {
        double aij = 0.0;
        if (active && c < n) aij = static_cast<double>(values[lane * md + c]);
        row[c] = aij;
    }
    const double a_diag = active ? row[lane] : 0.0;
    const double b_i = active ? static_cast<double>(b_blk[lane]) : 0.0;

    // LEFT preconditioner apply z = M^-1 r (this lane's z_i). Jacobi: z_i = r_i/a_ii
    // (non-symmetric A -> the diagonal can have either sign; we keep the SIGNED
    // diagonal -- left Jacobi for a general matrix, NOT absolute value). The
    // BlockJacobi key is the reserved hook for ILU(0)/AMG (p03-B); until wired it
    // falls through to the safe diagonal apply (documented in the header).
    auto apply_precond = [&](double r_i) -> double {
        if (!active) return 0.0;
        // Jacobi (and the not-yet-wired ILU0/AMG hook fall back here).
        double d = a_diag;
        if (d > -kNormFloor && d < kNormFloor) d = 1.0;  // guard near-zero diagonal
        return r_i / d;
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

    // ||b|| (preconditioned-residual stop reference). The stop test uses the
    // un-preconditioned ||b|| as the relative scale (matches CG/MINRES). Fixed order.
    const double bb = WarpButterflySum(b_i * b_i);
    const double bnorm = sqrt(bb > 0.0 ? bb : 0.0);
    const double stop = tol * (bnorm > kNormFloor ? bnorm : 1.0);

    // x0 = 0. The solution component this lane owns, carried across restart cycles.
    double x_i = 0.0;
    bool converged = (bnorm <= kNormFloor);  // zero rhs => x = 0 exact

    // Outer (restart) cycles: ceil(max_iter / restart), at least 1.
    const uint32_t m = (restart < 1u) ? 1u : restart;
    const uint32_t max_cycles =
        (max_iter < 1u) ? 1u : ((max_iter + m - 1u) / m);

    for (uint32_t cycle = 0u; cycle < max_cycles && !converged; ++cycle) {
        // r0 = M^-1 (b - A x). x0=0 on cycle 0 -> A x = 0; otherwise recompute the
        // TRUE residual via the deterministic SpMV (so multi-cycle convergence is
        // measured against the real residual, not just the carried Givens estimate).
        const double ax_i = (cycle == 0u) ? 0.0 : spmv(x_i);
        const double r_i = b_i - ax_i;
        const double z_i = apply_precond(r_i);
        const double beta_sq = WarpButterflySum(z_i * z_i);
        double beta = sqrt(beta_sq > 0.0 ? beta_sq : 0.0);
        if (beta <= kNormFloor) { converged = true; break; }

        // v_0 = z / beta ; g[0] = beta.
        const double inv_beta = 1.0 / beta;
        if (active) v_sh[0u * md + lane] = z_i * inv_beta;
        __syncwarp(0xffffffffu);
        if (lane == 0u) gv_sh[0] = beta;

        uint32_t k = 0u;          // number of Arnoldi vectors built this cycle
        bool inner_converged = false;
        bool happy = false;       // happy breakdown (h_{j+1,j} ~ 0)

        for (uint32_t j = 0u; j < m && !inner_converged && !happy; ++j) {
            // w = M^-1 (A v_j). v_j is warp-distributed in v_sh[j*md + .].
            const double vj_i = active ? v_sh[j * md + lane] : 0.0;
            const double av_i = spmv(vj_i);
            double w_i = apply_precond(av_i);

            // MODIFIED Gram-Schmidt: for i=0..j (FIXED order), h[i][j]=<w,v_i>,
            // w -= h[i][j] v_i. Each dot is the fixed-order butterfly; the subtraction
            // is per-lane. Storing into the lane-0-owned Hessenberg column j.
            for (uint32_t i = 0u; i <= j; ++i) {
                const double vi_i = active ? v_sh[i * md + lane] : 0.0;
                const double hij = WarpButterflySum(w_i * vi_i);
                w_i -= hij * vi_i;
                if (lane == 0u) h_sh[j * hrows + i] = hij;
            }
            // h[j+1][j] = ||w||.
            const double ww = WarpButterflySum(w_i * w_i);
            const double hjp1 = sqrt(ww > 0.0 ? ww : 0.0);
            if (lane == 0u) h_sh[j * hrows + (j + 1u)] = hjp1;

            if (hjp1 <= kNormFloor) {
                // Happy breakdown: the new direction vanished -> the exact solution
                // lies in span(v_0..v_j). Build NO v_{j+1}; this column closes the
                // least-squares system. k = j+1 includes column j.
                happy = true;
            } else {
                const double inv_h = 1.0 / hjp1;
                if (active) v_sh[(j + 1u) * md + lane] = w_i * inv_h;
                __syncwarp(0xffffffffu);
            }

            // Lane 0: apply the j PRIOR Givens rotations to column j of H, form the
            // new rotation to zero h[j+1][j], rotate g. All fp64, fixed order -> D1.
            if (lane == 0u) {
                double* hcol = h_sh + j * hrows;
                for (uint32_t i = 0u; i < j; ++i) {
                    const double t1 = hcol[i];
                    const double t2 = hcol[i + 1u];
                    hcol[i] = cs_sh[i] * t1 + sn_sh[i] * t2;
                    hcol[i + 1u] = -sn_sh[i] * t1 + cs_sh[i] * t2;
                }
                // New rotation over (h[j][j], h[j+1][j]).
                const double hj = hcol[j];
                const double hp = hcol[j + 1u];
                double rr = sqrt(hj * hj + hp * hp);
                if (rr <= kNormFloor) rr = kNormFloor;
                const double cj = hj / rr;
                const double sj = hp / rr;
                cs_sh[j] = cj;
                sn_sh[j] = sj;
                hcol[j] = cj * hj + sj * hp;  // == rr (the rotated diagonal)
                hcol[j + 1u] = 0.0;           // zeroed sub-diagonal
                // Rotate g: g[j+1] = -sj*g[j]; g[j] = cj*g[j].
                const double gj = gv_sh[j];
                gv_sh[j] = cj * gj;
                gv_sh[j + 1u] = -sj * gj;
            }
            __syncwarp(0xffffffffu);

            k = j + 1u;
            // Convergence: |g[k]| is the GMRES residual estimate (||r_k||). Broadcast
            // lane-0's value to the warp for a uniform branch (no lane split).
            double res_est = (lane == 0u) ? gv_sh[k] : 0.0;
            res_est = __shfl_sync(0xffffffffu, res_est, 0);
            res_est = (res_est < 0.0) ? -res_est : res_est;
            if (!run_to_fixed_iters && res_est <= stop) inner_converged = true;
        }

        // Least squares: solve the k x k upper-triangular R y = g[0..k-1] (R is the
        // rotated Hessenberg's leading triangle, col-major in h_sh) by back-sub on
        // lane 0, then publish y to the warp. Fixed loop order -> D1.
        if (lane == 0u) {
            for (uint32_t ii = 0u; ii < k; ++ii) {
                const uint32_t i = k - 1u - ii;
                double s = gv_sh[i];
                for (uint32_t c = i + 1u; c < k; ++c) {
                    s -= h_sh[c * hrows + i] * y_sh[c];
                }
                double rii = h_sh[i * hrows + i];
                if (rii > -kNormFloor && rii < kNormFloor)
                    rii = (rii < 0.0) ? -kNormFloor : kNormFloor;
                y_sh[i] = s / rii;
            }
        }
        __syncwarp(0xffffffffu);

        // x += sum_{i<k} y_i v_i  (per-lane; v_i warp-distributed). Fixed i order.
        for (uint32_t i = 0u; i < k; ++i) {
            const double yi = y_sh[i];
            const double vi_i = active ? v_sh[i * md + lane] : 0.0;
            x_i += yi * vi_i;
        }

        // If the inner loop signalled convergence (or happy breakdown gave the exact
        // subspace solution and the residual is at floor), we are done.
        if (inner_converged) { converged = true; }
        // Happy breakdown means x is exact in span; recompute is unnecessary, the next
        // cycle (if any) would produce r0 ~ 0 and exit. Loop continues otherwise to
        // restart with the updated x.
    }

    // --- Write back (capped at kMaxBlockDim; padding lanes write 0). -------------
    if (lane < kMaxBlockDim) {
        x_blk[lane] = active ? static_cast<float>(x_i) : 0.0f;
    }
}

__global__ void SolveGmresKernel(const float* __restrict__ values,
                                 const uint32_t* __restrict__ block_dim,
                                 const float* __restrict__ b, float* __restrict__ x,
                                 uint32_t block_count, Preconditioner precond,
                                 uint32_t restart, uint32_t max_iter, double tol,
                                 bool run_to_fixed_iters) {
    const uint32_t blk = blockIdx.x;
    if (blk >= block_count) return;
    const uint32_t lane = threadIdx.x;

    // Shared scratch (sized to the compile-time kMaxRestart so it is fixed per warp).
    __shared__ double v_sh[(kMaxRestart + 1u) * kMaxBlockDim];
    __shared__ double h_sh[(kMaxRestart + 1u) * kMaxRestart];
    __shared__ double gv_sh[kMaxRestart + 1u];
    __shared__ double cs_sh[kMaxRestart];
    __shared__ double sn_sh[kMaxRestart];
    __shared__ double y_sh[kMaxRestart];

    const uint32_t n = block_dim[blk];
    const float* __restrict__ values_blk =
        values + static_cast<size_t>(blk) * kBlockStride;
    const float* __restrict__ b_blk = b + static_cast<size_t>(blk) * kMaxBlockDim;
    float* __restrict__ x_blk = x + static_cast<size_t>(blk) * kMaxBlockDim;

    if (n == 0u) {
        if (lane < kMaxBlockDim) x_blk[lane] = 0.0f;
        return;
    }

    // Clamp restart to the compile-time cap.
    uint32_t m = (restart < 1u) ? 1u : restart;
    if (m > kMaxRestart) m = kMaxRestart;

    SolveOneBlockGmres(values_blk, n, b_blk, x_blk, precond, m, max_iter, tol,
                       run_to_fixed_iters, v_sh, h_sh, gv_sh, cs_sh, sn_sh, y_sh,
                       lane);
}

// Non-symmetric detector kernel (read-only): lane 0 scans the strict-upper vs
// strict-lower off-diagonal pairs; if any pair differs by more than a RELATIVE
// margin the block is non-symmetric -> OR the uint flag (uint atomicOr => D1; NOT a
// float atomic). A is read-only. A bit-symmetric A (the Delassus) gives exactly-zero
// difference => never flags => symmetric path byte-identical. See header for the
// byte-safety argument.
__device__ constexpr double kSymRelEps = 1.0e-5;  // above ~1e-7 fp32 floor, at gate

__global__ void DetectNonSymmetricKernel(const float* __restrict__ values,
                                         const uint32_t* __restrict__ block_dim,
                                         uint32_t block_count,
                                         uint32_t* __restrict__ flag) {
    const uint32_t blk = blockIdx.x;
    if (blk >= block_count) return;
    const uint32_t lane = threadIdx.x;
    const uint32_t n = block_dim[blk];
    if (n == 0u) return;
    if (lane != 0u) return;  // lane 0 does the sequential scan

    const float* __restrict__ vb = values + static_cast<size_t>(blk) * kBlockStride;
    // Block scale: max |entry| over the active n x n sub-block (fixed order).
    double max_abs = 0.0;
    for (uint32_t i = 0u; i < n; ++i) {
        for (uint32_t j = 0u; j < n; ++j) {
            const double a = static_cast<double>(vb[i * kMaxBlockDim + j]);
            const double aa = (a < 0.0) ? -a : a;
            if (aa > max_abs) max_abs = aa;
        }
    }
    const double scale = (max_abs > 0.0) ? max_abs : 1.0;
    const double thresh = kSymRelEps * scale;

    bool nonsym = false;
    for (uint32_t i = 0u; i < n && !nonsym; ++i) {
        for (uint32_t j = i + 1u; j < n; ++j) {
            const double aij = static_cast<double>(vb[i * kMaxBlockDim + j]);
            const double aji = static_cast<double>(vb[j * kMaxBlockDim + i]);
            double d = aij - aji;
            if (d < 0.0) d = -d;
            if (d > thresh) { nonsym = true; break; }
        }
    }
    if (nonsym) atomicOr(flag, 1u);
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string("nuka::diffsim::SelfWrittenGmresBackend ") + operation +
            ": " + cudaGetErrorString(result));
    }
}

}  // namespace

SelfWrittenGmresBackend::SelfWrittenGmresBackend(const phi::DeviceContext& context,
                                                 uint32_t restart,
                                                 DeterminismLevel determinism)
    : context_(context),
      restart_((restart < 1u) ? 1u : restart),
      determinism_(determinism) {}

void SelfWrittenGmresBackend::Solve(const BatchedDenseSpdSystem& system,
                                    const float* b, float* x,
                                    const SolveParams& params) {
    if (system.block_count == 0u) return;
    if (system.values == nullptr || system.block_dim == nullptr || b == nullptr ||
        x == nullptr) {
        throw std::invalid_argument(
            "nuka::diffsim::SelfWrittenGmresBackend::Solve: null system/vector "
            "pointer");
    }
    phi::ScopedDeviceGuard guard(context_.device_id);
    const cudaStream_t stream = context_.stream.Native();

    SolveGmresKernel<<<system.block_count, kWarpSize, 0u, stream>>>(
        system.values, system.block_dim, b, x, system.block_count,
        params.preconditioner, restart_, params.max_iter,
        static_cast<double>(params.tol), params.run_to_fixed_iters);
    CheckCuda(cudaGetLastError(), "SolveGmresKernel launch");
}

void DetectBatchedNonSymmetric(const phi::DeviceContext& context,
                               const BatchedDenseSpdSystem& system,
                               uint32_t* out_nonsymmetric_flag) {
    if (system.block_count == 0u || out_nonsymmetric_flag == nullptr) return;
    if (system.values == nullptr || system.block_dim == nullptr) {
        throw std::invalid_argument(
            "nuka::diffsim::DetectBatchedNonSymmetric: null system pointer");
    }
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    DetectNonSymmetricKernel<<<system.block_count, kWarpSize, 0u, stream>>>(
        system.values, system.block_dim, system.block_count, out_nonsymmetric_flag);
    CheckCuda(cudaGetLastError(), "DetectNonSymmetricKernel launch");
}

}  // namespace nuka::diffsim
