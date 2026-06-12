// ---------------------------------------------------------------------------
// v0.5 p03 R1 -- self-written CG solver: Eigen-LDLT cross-check + D1 bit-exact
// ---------------------------------------------------------------------------
//
// Validates nuka::diffsim::SelfWrittenCgBackend (warp-per-articulation batched
// dense SPD CG, the IFT solver backend) against a DENSE HOST REFERENCE
// (Eigen::LDLT, < 1e-6 componentwise) -- NOT against any closed-source SDK -- and
// proves the R6 D1 gate: same input + same GPU => byte-identical solution.
//
//   CgVsDense.AgreesWithEigenLdlt : a batch of GENUINELY COUPLED (off-diagonal)
//     random SPD blocks of varied size n in {1..12}, random b; per-block
//     x_ref = A.ldlt().solve(b) on host; EXPECT max componentwise rel-err < 1e-6
//     for BOTH preconditioners (Jacobi runs the full CG loop; BlockJacobi is the
//     exact island solve -> near machine-exact).
//   CgDeterminism.BitExactSameInputs : same A,b, two Solve calls on the same GPU
//     -> EXPECT byte-identical raw bits (memcmp == 0).
// ---------------------------------------------------------------------------

#include "diffsim/sparse_solver_backend.hpp"
#include "diffsim/sparse_solver_cg.hpp"
#include "phi/buffer_legacy.hpp"
#include "phi/device_context.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

namespace diffsim = nuka::diffsim;
constexpr uint32_t kMd = diffsim::kMaxBlockDim;  // 12
constexpr uint32_t kStride = kMd * kMd;          // 144

// One synthetic SPD block of active dimension n, stored row-major into a
// kStride-padded float span, plus its rhs into a kMd-padded float span. SPD by
// construction: A = L L^T + n*I with L lower-triangular random in [-1,1], which
// guarantees a genuinely COUPLED (non-diagonal) SPD matrix (the L L^T term fills
// the off-diagonals; +n*I keeps it well-conditioned without making it diagonal).
struct Block {
    uint32_t n = 0u;
    Eigen::MatrixXd A;  // n x n, the dense reference (fp64)
    Eigen::VectorXd b;  // n
};

Block MakeSpdBlock(uint32_t n, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(n, n);
    for (uint32_t i = 0u; i < n; ++i) {
        for (uint32_t j = 0u; j <= i; ++j) {
            L(i, j) = u(rng);
        }
    }
    Eigen::MatrixXd A = L * L.transpose();
    A += static_cast<double>(n) * Eigen::MatrixXd::Identity(n, n);  // SPD, coupled
    // Symmetrize against fp drift so A is exactly symmetric (CG/LDLT both assume it).
    A = 0.5 * (A + A.transpose()).eval();
    Eigen::VectorXd b(n);
    for (uint32_t i = 0u; i < n; ++i) b(i) = u(rng);
    return Block{n, A, b};
}

// Pack a batch of blocks into the device layout (values: block-major, stride
// kStride, row-major within block; b: block-major, stride kMd). Padding is 0.
void PackBatch(const std::vector<Block>& blocks, std::vector<float>& values,
               std::vector<uint32_t>& dims, std::vector<float>& rhs) {
    const size_t bc = blocks.size();
    values.assign(bc * kStride, 0.0f);
    dims.assign(bc, 0u);
    rhs.assign(bc * kMd, 0.0f);
    for (size_t blk = 0u; blk < bc; ++blk) {
        const Block& B = blocks[blk];
        dims[blk] = B.n;
        for (uint32_t i = 0u; i < B.n; ++i) {
            for (uint32_t j = 0u; j < B.n; ++j) {
                values[blk * kStride + i * kMd + j] = static_cast<float>(B.A(i, j));
            }
            rhs[blk * kMd + i] = static_cast<float>(B.b(i));
        }
    }
}

// Upload host vectors to device phi::Buffers (returns by move via out-params).
template <typename T>
nuka::phi::Buffer UploadBuffer(const std::vector<T>& host) {
    nuka::phi::Buffer buf(host.size() * sizeof(T), nuka::phi::MemoryKind::Device);
    buf.CopyFromHost(host.data(), host.size() * sizeof(T));
    return buf;
}

// Runs the CG backend on a packed batch and returns the solution x (host,
// block-major stride kMd, padding included).
std::vector<float> RunSolve(const nuka::phi::DeviceContext& ctx,
                            const std::vector<float>& values,
                            const std::vector<uint32_t>& dims,
                            const std::vector<float>& rhs,
                            diffsim::Preconditioner precond, uint32_t max_iter,
                            bool fixed_iters) {
    const uint32_t bc = static_cast<uint32_t>(dims.size());
    nuka::phi::Buffer d_values = UploadBuffer(values);
    nuka::phi::Buffer d_dims = UploadBuffer(dims);
    nuka::phi::Buffer d_b = UploadBuffer(rhs);
    nuka::phi::Buffer d_x(static_cast<size_t>(bc) * kMd * sizeof(float),
                          nuka::phi::MemoryKind::Device);

    diffsim::BatchedDenseSpdSystem system;
    system.block_count = bc;
    system.values = static_cast<const float*>(d_values.Data());
    system.block_dim = static_cast<const uint32_t*>(d_dims.Data());

    diffsim::SolveParams params;
    params.preconditioner = precond;
    params.max_iter = max_iter;
    params.tol = 1.0e-6f;
    params.run_to_fixed_iters = fixed_iters;

    auto solver = diffsim::MakeSparseSolverBackend("cg", ctx);
    solver->Solve(system, static_cast<const float*>(d_b.Data()),
                  static_cast<float*>(d_x.Data()), params);
    ctx.stream.Synchronize();

    std::vector<float> x(static_cast<size_t>(bc) * kMd, 0.0f);
    d_x.CopyToHost(x.data(), x.size() * sizeof(float));
    return x;
}

// Builds a representative batch: every size n in {1..12} (so the n<12 padding
// paths are exercised) plus a few extra coupled blocks, deterministic seed.
std::vector<Block> MakeBatch() {
    std::mt19937 rng(0xC0FFEEu);
    std::vector<Block> blocks;
    for (uint32_t n = 1u; n <= kMd; ++n) blocks.push_back(MakeSpdBlock(n, rng));
    // A handful of full-size coupled blocks to bulk up the batch / parallelism.
    for (uint32_t k = 0u; k < 8u; ++k) blocks.push_back(MakeSpdBlock(12u, rng));
    // An explicit empty (n==0) block to exercise the skip path.
    blocks.push_back(Block{0u, Eigen::MatrixXd(0, 0), Eigen::VectorXd(0)});
    return blocks;
}

// Max componentwise relative error of the CG solution vs Eigen::LDLT per block.
double MaxRelErr(const std::vector<Block>& blocks, const std::vector<float>& x) {
    double max_rel = 0.0;
    for (size_t blk = 0u; blk < blocks.size(); ++blk) {
        const Block& B = blocks[blk];
        if (B.n == 0u) continue;
        Eigen::VectorXd x_ref = B.A.ldlt().solve(B.b);
        // Reference scale = max |x_ref| component (componentwise rel-err with a
        // tiny floor so a near-zero reference component doesn't blow up the ratio).
        double scale = 0.0;
        for (uint32_t i = 0u; i < B.n; ++i)
            scale = std::max(scale, std::abs(x_ref(i)));
        const double denom = std::max(scale, 1.0e-12);
        for (uint32_t i = 0u; i < B.n; ++i) {
            const double xi = static_cast<double>(x[blk * kMd + i]);
            max_rel = std::max(max_rel, std::abs(xi - x_ref(i)) / denom);
        }
    }
    return max_rel;
}

TEST(CgVsDense, AgreesWithEigenLdlt) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const std::vector<Block> blocks = MakeBatch();
    std::vector<float> values, rhs;
    std::vector<uint32_t> dims;
    PackBatch(blocks, values, dims, rhs);

    // Jacobi: genuine CG iteration on coupled blocks. Run to a FIXED 64 iters
    // (>> the <=12 dim CG needs to finish in exact arithmetic) so the solve fully
    // CONVERGES against the dense reference, rather than honoring the production
    // tol=1e-6 early-exit (which would stop CG at relative-residual 1e-6 -> a
    // solution error ~kappa*1e-6, i.e. right at the gate -- that tests tol, not the
    // solver). fp64 internal arithmetic keeps the converged componentwise rel-err
    // well under 1e-6. (BlockJacobi below is the exact island solve regardless.)
    const std::vector<float> x_jacobi =
        RunSolve(ctx, values, dims, rhs, diffsim::Preconditioner::Jacobi, 64u, true);
    const double rel_jacobi = MaxRelErr(blocks, x_jacobi);
    std::printf("[CgVsDense] Jacobi      max componentwise rel-err = %.3e\n", rel_jacobi);
    EXPECT_LT(rel_jacobi, 1.0e-6) << "Jacobi-preconditioned CG vs Eigen::LDLT";

    // BlockJacobi: the exact per-block Cholesky island solve -> converges in <=1
    // iteration; expect near machine-exact.
    const std::vector<float> x_block =
        RunSolve(ctx, values, dims, rhs, diffsim::Preconditioner::BlockJacobi, 64u, false);
    const double rel_block = MaxRelErr(blocks, x_block);
    std::printf("[CgVsDense] BlockJacobi max componentwise rel-err = %.3e\n", rel_block);
    EXPECT_LT(rel_block, 1.0e-6) << "BlockJacobi (exact Cholesky) CG vs Eigen::LDLT";
}

TEST(CgDeterminism, BitExactSameInputs) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const std::vector<Block> blocks = MakeBatch();
    std::vector<float> values, rhs;
    std::vector<uint32_t> dims;
    PackBatch(blocks, values, dims, rhs);

    // Two independent solves of the SAME system on the SAME GPU. Test BOTH
    // preconditioners AND a forced fixed-iteration path (the strictest D1 demo).
    for (auto precond :
         {diffsim::Preconditioner::Jacobi, diffsim::Preconditioner::BlockJacobi}) {
        for (bool fixed : {false, true}) {
            const std::vector<float> x1 =
                RunSolve(ctx, values, dims, rhs, precond, 64u, fixed);
            const std::vector<float> x2 =
                RunSolve(ctx, values, dims, rhs, precond, 64u, fixed);
            ASSERT_EQ(x1.size(), x2.size());
            // Raw-bit comparison: byte-identical output is the R6 D1 gate.
            const int cmp =
                std::memcmp(x1.data(), x2.data(), x1.size() * sizeof(float));
            EXPECT_EQ(cmp, 0)
                << "D1 bit-exact violated: precond="
                << static_cast<int>(precond) << " fixed_iters=" << fixed;
        }
    }
    std::printf("[CgDeterminism] all 4 (precond x fixed) paths byte-identical\n");
}

// ---------------------------------------------------------------------------
// MUST-FIX 1 regression (p03 adversarial review): an OBLIQUE rank-deficient PSD
// block must produce a FINITE, range-space-correct BlockJacobi solution -- NOT a
// NaN. The old absolute pivot floor (1e-300 -> sqrt -> 1e-150 -> divide) overflowed
// to NaN on an oblique null direction (a PSD block whose post-elimination Schur
// pivot is tiny-but-NONZERO, unlike a structurally dead row whose pivot is exactly
// 0 and happened to stay finite). The modified-Cholesky null-direction handling
// (kNullPivotRelEps) must keep it finite.
//
// We build A = Q diag(lam0, lam1, ~0) Q^T with a NON-axis-aligned rotation Q, so the
// null direction is OBLIQUE (not a coordinate row). rhs = A y is then guaranteed in
// range(A), so a correct range-space solve satisfies A x = rhs exactly on the range.
// We assert (1) every solution component is std::isfinite, and (2) the residual
// ||A x - rhs|| is small (range-space-correct). We do NOT compare to a Moore-Penrose
// pseudo-inverse: the modified-Cholesky least-norm solution is pivot-order-dependent
// and is not exactly A^+ rhs, but it IS an exact solution on the range (which is what
// the IFT path needs -- the null direction carries zero gradient).
TEST(CgVsDense, ObliqueRankDeficientBlockIsFiniteAndRangeCorrect) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    constexpr uint32_t n = 3u;

    // Non-axis-aligned orthonormal Q (a rotation built from a fixed generic axis/angle
    // via Gram-Schmidt on three generic seed vectors -> guaranteed oblique columns).
    Eigen::Matrix3d M;
    M << 0.6, 0.2, -0.3,
         0.1, 0.7,  0.4,
        -0.5, 0.3,  0.8;
    Eigen::HouseholderQR<Eigen::Matrix3d> qr(M);
    Eigen::Matrix3d Q = qr.householderQ();  // orthonormal, oblique (non-axis) columns

    // One tiny eigenvalue along an OBLIQUE direction (Q's 3rd column), two healthy.
    // lam2 = 1e-9 * lam_max is well below kNullPivotRelEps*max_diag (the null cutoff)
    // yet NONZERO -- exactly the case the old absolute floor turned into NaN.
    Eigen::Vector3d lam(21.0, 4.0, 21.0e-9);
    Eigen::Matrix3d A = Q * lam.asDiagonal() * Q.transpose();
    A = 0.5 * (A + A.transpose()).eval();  // exact symmetry

    // rhs = A y for a generic y -> rhs in range(A) (the null component of y is killed
    // by A, so rhs has zero projection on the null direction; a range-space solve is
    // therefore exact). y picks up all three directions to make the test non-trivial.
    Eigen::Vector3d y(0.9, -0.4, 1.3);
    Eigen::Vector3d rhs_vec = A * y;

    // Pack the single block into the device layout.
    std::vector<float> values(kStride, 0.0f);
    std::vector<uint32_t> dims(1u, n);
    std::vector<float> rhs(kMd, 0.0f);
    for (uint32_t i = 0u; i < n; ++i) {
        for (uint32_t j = 0u; j < n; ++j)
            values[i * kMd + j] = static_cast<float>(A(i, j));
        rhs[i] = static_cast<float>(rhs_vec(i));
    }

    // BlockJacobi = the modified-Cholesky island solve (the IFT solve path).
    const std::vector<float> x =
        RunSolve(ctx, values, dims, rhs, diffsim::Preconditioner::BlockJacobi, 64u, false);

    // (1) FINITENESS: every active component finite (the NaN lock-out). Padding too.
    for (uint32_t i = 0u; i < kMd; ++i)
        ASSERT_TRUE(std::isfinite(x[i]))
            << "oblique-PSD BlockJacobi produced non-finite x[" << i << "] (the "
               "MUST-FIX 1 NaN regressed)";

    // (2) RANGE-SPACE CORRECTNESS: ||A x - rhs|| small (A x must reproduce the
    // in-range rhs). Computed on host in fp64 from the GPU solution.
    Eigen::Vector3d x_dev;
    for (uint32_t i = 0u; i < n; ++i) x_dev(i) = static_cast<double>(x[i]);
    const Eigen::Vector3d resid = A * x_dev - rhs_vec;
    const double resid_norm = resid.norm();
    const double rhs_norm = std::max(rhs_vec.norm(), 1.0e-12);
    std::printf("[CgVsDense] oblique-PSD: lam=(%.3e,%.3e,%.3e) ||Ax-rhs||/||rhs|| = %.3e "
                "(x finite)\n",
                lam(0), lam(1), lam(2), resid_norm / rhs_norm);
    // fp32 store of A + the solve -> a few-eps relative residual on the healthy
    // directions; 1e-4 is ample headroom and far from a NaN/garbage result.
    EXPECT_LT(resid_norm, 1.0e-4 * rhs_norm)
        << "oblique-PSD BlockJacobi solution not range-space-correct";

    // The null component must be SUPPRESSED (least-norm-style): x has no blow-up
    // along the oblique null direction -- assert it stays O(1), not 1/tiny ~ 1e9.
    const double null_comp = std::abs(Q.col(2).dot(x_dev));
    std::printf("[CgVsDense] oblique-PSD: |null-direction component of x| = %.3e\n",
                null_comp);
    EXPECT_LT(null_comp, 1.0e2)
        << "oblique-PSD solution blew up along the null direction (divide-by-tiny)";
}

}  // namespace
