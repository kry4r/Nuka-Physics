// ---------------------------------------------------------------------------
// v0.7 p03-A -- self-written GMRES vs a DENSE reference on GENERAL NON-SYMMETRIC
// systems. < 1e-5 componentwise + D1 bit-exact + the restart(m<n) path + the
// non-symmetric detector byte-safety.
// ---------------------------------------------------------------------------
//
// GMRES (Saad & Schultz 1986) solves GENERAL non-symmetric systems where neither CG
// (needs SPD) nor MINRES (needs A == A^T) applies. We build genuinely non-symmetric
// blocks A = Q (D + N) Q^T-ish via a random well-conditioned matrix with a forced
// positive-real spectrum (diagonally dominant + a non-symmetric off-diagonal) so the
// system is solvable and GMRES converges, across the block-size range n in {2..12}.
// The self-written warp-per-block GMRES (fp64 internal, Jacobi preconditioner) is
// cross-checked against Eigen PartialPivLU, the APPROVED TEST-ONLY dense oracle for
// general non-symmetric systems (LDLT/FullPivLU symmetric assumptions do not apply;
// PartialPivLU is the standard general dense solver).
//
// Tolerance honesty: the gate is 1e-5 (the phase exit criterion); agreement is
// measured against the dense LU on the SAME fp32-quantized (A,b) the GPU consumed
// (isolates solver agreement from input quantization, exactly as the CG/MINRES tests
// do). We report the achieved max rel-err + residual so the margin is visible.
//
// RESTART COVERAGE (the load-bearing test the n<=12 default m=30 would otherwise
// leave dead): GMRES converges in <= n Arnoldi steps, so m=30 > n always converges
// in ONE cycle and never restarts. To exercise + gate the genuine restart path we
// add a backend with m < n (m=4 on n up to 12) and assert it still agrees with the
// dense oracle -- i.e. multiple restart cycles converge.
// ---------------------------------------------------------------------------

#include "diffsim/solver/gmres_backend.hpp"
#include "diffsim/sparse_solver_backend.hpp"
#include "phi/buffer.hpp"
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

// A genuinely NON-SYMMETRIC block of dimension n: a random matrix made DIAGONALLY
// DOMINANT (so it is non-singular + Jacobi-preconditionable + GMRES-convergent)
// while keeping A[i][j] != A[j][i] (non-symmetric off-diagonal). The diagonal carries
// a positive bias so the spectrum stays in the right half-plane.
struct Block {
    uint32_t n = 0u;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
};

Block MakeNonSymmetricBlock(uint32_t n, double offdiag_scale, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Eigen::MatrixXd A(n, n);
    for (uint32_t i = 0u; i < n; ++i)
        for (uint32_t j = 0u; j < n; ++j)
            A(i, j) = offdiag_scale * u(rng);  // independent -> NON-symmetric
    // Diagonal dominance: a_ii = (sum_j!=i |a_ij|) + positive bias. Guarantees a
    // non-singular, GMRES-convergent, Jacobi-preconditionable matrix.
    for (uint32_t i = 0u; i < n; ++i) {
        double row_abs = 0.0;
        for (uint32_t j = 0u; j < n; ++j)
            if (j != i) row_abs += std::abs(A(i, j));
        A(i, i) = row_abs + 1.0 + std::abs(u(rng));
    }
    Eigen::VectorXd b(n);
    for (uint32_t i = 0u; i < n; ++i) b(i) = u(rng);
    return Block{n, A, b};
}

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
            for (uint32_t j = 0u; j < B.n; ++j)
                values[blk * kStride + i * kMd + j] = static_cast<float>(B.A(i, j));
            rhs[blk * kMd + i] = static_cast<float>(B.b(i));
        }
    }
}

template <typename T>
nuka::phi::Buffer UploadBuffer(const std::vector<T>& host) {
    nuka::phi::Buffer buf(host.size() * sizeof(T), nuka::phi::MemoryKind::Device);
    buf.CopyFromHost(host.data(), host.size() * sizeof(T));
    return buf;
}

// Runs GMRES via the named backend factory. `restart` selects the GMRES(m) length:
// the default backend uses m=30 (one cycle for n<=12); the restart-coverage test
// uses the "self_gmres" backend with the SolveParams unchanged and relies on the
// factory default of m=30, so for the small-m path we use the explicit backend ctor.
std::vector<float> RunGmres(const nuka::phi::DeviceContext& ctx,
                            diffsim::SparseLinearSolver& solver,
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
    params.tol = 1.0e-7f;
    params.run_to_fixed_iters = fixed_iters;

    solver.Solve(system, static_cast<const float*>(d_b.Data()),
                 static_cast<float*>(d_x.Data()), params);
    ctx.stream.Synchronize();

    std::vector<float> x(static_cast<size_t>(bc) * kMd, 0.0f);
    d_x.CopyToHost(x.data(), x.size() * sizeof(float));
    return x;
}

Eigen::MatrixXd Fp32(const Eigen::MatrixXd& A) {
    Eigen::MatrixXd out(A.rows(), A.cols());
    for (Eigen::Index i = 0; i < A.rows(); ++i)
        for (Eigen::Index j = 0; j < A.cols(); ++j)
            out(i, j) = static_cast<double>(static_cast<float>(A(i, j)));
    return out;
}
Eigen::VectorXd Fp32(const Eigen::VectorXd& v) {
    Eigen::VectorXd out(v.size());
    for (Eigen::Index i = 0; i < v.size(); ++i)
        out(i) = static_cast<double>(static_cast<float>(v(i)));
    return out;
}

// Max componentwise rel-err of the GPU GMRES solution vs Eigen PartialPivLU on the
// SAME fp32-quantized (A,b). PartialPivLU is the general non-symmetric dense oracle.
double MaxRelErr(const std::vector<Block>& blocks, const std::vector<float>& x,
                 double& worst_resid_out) {
    double max_rel = 0.0;
    double worst_resid = 0.0;
    for (size_t blk = 0u; blk < blocks.size(); ++blk) {
        const Block& B = blocks[blk];
        if (B.n == 0u) continue;
        const Eigen::MatrixXd A32 = Fp32(B.A);
        const Eigen::VectorXd b32 = Fp32(B.b);
        Eigen::VectorXd x_ref = A32.partialPivLu().solve(b32);
        double scale = 0.0;
        for (uint32_t i = 0u; i < B.n; ++i)
            scale = std::max(scale, std::abs(x_ref(i)));
        const double denom = std::max(scale, 1.0e-12);
        Eigen::VectorXd x_gpu(B.n);
        for (uint32_t i = 0u; i < B.n; ++i) {
            const double xi = static_cast<double>(x[blk * kMd + i]);
            x_gpu(i) = xi;
            max_rel = std::max(max_rel, std::abs(xi - x_ref(i)) / denom);
        }
        const double resid =
            (A32 * x_gpu - b32).norm() / std::max(b32.norm(), 1.0e-12);
        worst_resid = std::max(worst_resid, resid);
    }
    worst_resid_out = worst_resid;
    return max_rel;
}

std::vector<Block> MakeNonSymmetricBatch() {
    std::mt19937 rng(0xC0FFEEu);
    std::vector<Block> blocks;
    for (double scale : {0.3, 1.0, 3.0}) {
        for (uint32_t n = 2u; n <= kMd; ++n)
            blocks.push_back(MakeNonSymmetricBlock(n, scale, rng));
    }
    blocks.push_back(Block{0u, Eigen::MatrixXd(0, 0), Eigen::VectorXd(0)});
    return blocks;
}

// GATE 2: self-written GMRES (default m=30) agrees with the dense PartialPivLU oracle
// on genuinely NON-SYMMETRIC systems within 1e-5. For n<=12 the m=30 cycle converges
// without restarting (exact in <= n fp64 Arnoldi steps).
TEST(GmresVsDenseNonSymmetric, AgreesWithPartialPivLu) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const std::vector<Block> blocks = MakeNonSymmetricBatch();
    std::vector<float> values, rhs;
    std::vector<uint32_t> dims;
    PackBatch(blocks, values, dims, rhs);

    // Default factory backend (restart m=30).
    auto solver = diffsim::MakeSparseSolverBackend("self_gmres", ctx);
    const std::vector<float> x =
        RunGmres(ctx, *solver, values, dims, rhs, diffsim::Preconditioner::Jacobi,
                 200u, true);
    double worst_resid = 0.0;
    const double rel = MaxRelErr(blocks, x, worst_resid);
    std::printf("[GmresVsDenseNonSymmetric] m=30 Jacobi vs fp32-PartialPivLU: "
                "max rel-err = %.3e, worst ||Ax-b||/||b|| = %.3e\n",
                rel, worst_resid);
    EXPECT_LT(rel, 1.0e-5)
        << "self-written GMRES vs Eigen PartialPivLU on non-symmetric blocks";
    EXPECT_LT(worst_resid, 1.0e-5)
        << "GMRES residual ||Ax-b||/||b|| on non-symmetric blocks";
}

// GATE 2b (RESTART COVERAGE): GMRES(m) with m < n MUST restart across multiple cycles
// and STILL converge to the dense reference. m=4 forces restarts on blocks up to
// n=12 (3 cycles). This is the only test that exercises the restart machinery (the
// default m=30 always converges in one cycle for n<=12).
TEST(GmresVsDenseNonSymmetric, RestartSmallMConverges) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const std::vector<Block> blocks = MakeNonSymmetricBatch();
    std::vector<float> values, rhs;
    std::vector<uint32_t> dims;
    PackBatch(blocks, values, dims, rhs);

    // Explicit small restart length m=4 (< n for most blocks). Many outer cycles are
    // budgeted (max_iter=200 -> ceil(200/4)=50 cycles >> what convergence needs).
    diffsim::SelfWrittenGmresBackend gmres4(ctx, /*restart=*/4u);
    const std::vector<float> x =
        RunGmres(ctx, gmres4, values, dims, rhs, diffsim::Preconditioner::Jacobi,
                 200u, false);
    double worst_resid = 0.0;
    const double rel = MaxRelErr(blocks, x, worst_resid);
    std::printf("[GmresVsDenseNonSymmetric] RESTART m=4 (m<n) vs fp32-PartialPivLU: "
                "max rel-err = %.3e, worst ||Ax-b||/||b|| = %.3e\n",
                rel, worst_resid);
    EXPECT_LT(rel, 1.0e-5)
        << "GMRES(m=4) restart path failed to converge to the dense reference";
    EXPECT_LT(worst_resid, 1.0e-5)
        << "GMRES(m=4) restart residual on non-symmetric blocks";
}

// A SYMMETRIC SPD block (the byte-safety baseline for the detector: must NOT flag).
Block MakeSymmetricBlock(uint32_t n, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Eigen::MatrixXd M(n, n);
    for (uint32_t i = 0u; i < n; ++i)
        for (uint32_t j = 0u; j < n; ++j) M(i, j) = u(rng);
    Eigen::MatrixXd A = M * M.transpose();  // symmetric by construction
    A = 0.5 * (A + A.transpose()).eval();   // exactly symmetric
    for (uint32_t i = 0u; i < n; ++i) A(i, i) += static_cast<double>(n);  // SPD
    Eigen::VectorXd b(n);
    for (uint32_t i = 0u; i < n; ++i) b(i) = u(rng);
    return Block{n, A, b};
}

uint32_t RunDetect(const nuka::phi::DeviceContext& ctx,
                   const std::vector<Block>& blocks) {
    std::vector<float> values, rhs;
    std::vector<uint32_t> dims;
    PackBatch(blocks, values, dims, rhs);
    nuka::phi::Buffer d_values = UploadBuffer(values);
    nuka::phi::Buffer d_dims = UploadBuffer(dims);
    nuka::phi::Buffer d_flag(sizeof(uint32_t), nuka::phi::MemoryKind::Device);
    uint32_t zero = 0u;
    d_flag.CopyFromHost(&zero, sizeof(uint32_t));

    diffsim::BatchedDenseSpdSystem system;
    system.block_count = static_cast<uint32_t>(dims.size());
    system.values = static_cast<const float*>(d_values.Data());
    system.block_dim = static_cast<const uint32_t*>(d_dims.Data());

    diffsim::DetectBatchedNonSymmetric(ctx, system,
                                       static_cast<uint32_t*>(d_flag.Data()));
    ctx.stream.Synchronize();
    uint32_t flag = 0u;
    d_flag.CopyToHost(&flag, sizeof(uint32_t));
    return flag;
}

// THE BYTE-SAFETY EVIDENCE: the non-symmetric detector that gates ift_runner's GMRES
// auto-routing must return 0 (-> the symmetric CG/MINRES path runs untouched ->
// byte-identical) on every SYMMETRIC batch, and 1 on a genuinely non-symmetric one.
// This is what proves the symmetric IFT path is unchanged -- the regression tests
// alone cannot (GMRES would solve a symmetric system too); only flag==0 guarantees
// the symmetric branch is taken.
TEST(GmresVsDenseNonSymmetric, NonSymmetricDetectorBytesafe) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    std::mt19937 rng(0x5A5Au);

    // (a) exactly-symmetric SPD batch -> MUST NOT flag.
    std::vector<Block> sym;
    for (uint32_t n = 1u; n <= kMd; ++n) sym.push_back(MakeSymmetricBlock(n, rng));
    EXPECT_EQ(RunDetect(ctx, sym), 0u)
        << "non-symmetric detector FALSE-POSITIVE on a symmetric batch (would mis-"
           "route the symmetric IFT path to GMRES and break byte-identity)";

    // (b) genuinely non-symmetric batch -> MUST flag.
    std::vector<Block> nonsym;
    for (uint32_t n = 2u; n <= kMd; ++n)
        nonsym.push_back(MakeNonSymmetricBlock(n, 1.0, rng));
    EXPECT_EQ(RunDetect(ctx, nonsym), 1u)
        << "non-symmetric detector FALSE-NEGATIVE on a genuinely non-symmetric batch";

    // (c) a single non-symmetric block mixed into an otherwise-symmetric batch ->
    //     flag (the whole-batch routing trigger).
    std::vector<Block> mixed = sym;
    mixed.push_back(MakeNonSymmetricBlock(6u, 1.0, rng));
    EXPECT_EQ(RunDetect(ctx, mixed), 1u)
        << "one non-symmetric block in a symmetric batch must flag (whole-batch route)";

    std::printf("[GmresVsDenseNonSymmetric] detector byte-safe: symmetric->0, "
                "non-symmetric->1\n");
}

// D1: two GMRES solves of the same non-symmetric batch are byte-identical (both the
// one-cycle m=30 path and the multi-cycle m=4 restart path).
TEST(GmresVsDenseNonSymmetric, BitExact) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const std::vector<Block> blocks = MakeNonSymmetricBatch();
    std::vector<float> values, rhs;
    std::vector<uint32_t> dims;
    PackBatch(blocks, values, dims, rhs);

    for (uint32_t m : {30u, 4u}) {
        diffsim::SelfWrittenGmresBackend g1(ctx, m);
        diffsim::SelfWrittenGmresBackend g2(ctx, m);
        for (bool fixed : {false, true}) {
            const std::vector<float> x1 = RunGmres(
                ctx, g1, values, dims, rhs, diffsim::Preconditioner::Jacobi, 200u,
                fixed);
            const std::vector<float> x2 = RunGmres(
                ctx, g2, values, dims, rhs, diffsim::Preconditioner::Jacobi, 200u,
                fixed);
            ASSERT_EQ(x1.size(), x2.size());
            EXPECT_EQ(std::memcmp(x1.data(), x2.data(), x1.size() * sizeof(float)), 0)
                << "GMRES D1 bit-exact violated (non-symmetric batch), m=" << m
                << " fixed_iters=" << fixed;
        }
    }
    std::printf("[GmresVsDenseNonSymmetric] GMRES two-run byte-identical "
                "(m=30 + m=4 restart)\n");
}

}  // namespace
