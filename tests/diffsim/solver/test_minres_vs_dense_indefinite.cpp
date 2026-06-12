// ---------------------------------------------------------------------------
// v0.7 p02 -- self-written MINRES vs a DENSE reference on SYMMETRIC INDEFINITE
// systems. < 1e-5 componentwise + D1 bit-exact.
// ---------------------------------------------------------------------------
//
// MINRES (Paige & Saunders 1975) solves SYMMETRIC INDEFINITE systems where CG
// breaks down. We build blocks A = Q diag(lambda) Q^T with lambda containing BOTH
// positive AND negative eigenvalues (genuinely indefinite, NOT SPD), across the
// block-size range n in {2..12} (n==1 has no room for a sign split). The self-
// written warp-per-block MINRES (fp64 internal, absolute-Jacobi preconditioner --
// the SPD preconditioner required for indefinite MINRES) is cross-checked against
// Eigen FullPivLU, the APPROVED TEST-ONLY dense oracle for INDEFINITE systems
// (LDLT assumes (semi)definite and is INVALID here).
//
// Tolerance honesty: indefinite systems are worse-conditioned than the SPD blocks
// the CG test gates at 1e-6. The gate here is 1e-5 (the phase exit criterion);
// agreement is measured against the dense LU on the SAME fp32-quantized (A,b) the
// GPU consumed (isolates solver agreement from input quantization, exactly as the
// CG test does). We report the achieved max rel-err so the margin is visible.
// ---------------------------------------------------------------------------

#include "diffsim/solver/minres_backend.hpp"
#include "diffsim/sparse_solver_backend.hpp"
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

// A symmetric INDEFINITE block of dimension n: A = Q diag(lambda) Q^T with lambda
// log-spaced in magnitude over [1, kappa] but with ALTERNATING SIGNS so the
// spectrum straddles zero (genuinely indefinite). Q is Householder-orthonormal.
struct Block {
    uint32_t n = 0u;
    double kappa = 1.0;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
};

Block MakeIndefiniteBlock(uint32_t n, double kappa, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Eigen::MatrixXd M(n, n);
    for (uint32_t i = 0u; i < n; ++i)
        for (uint32_t j = 0u; j < n; ++j) M(i, j) = u(rng);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
    Eigen::MatrixXd Q = qr.householderQ();

    Eigen::VectorXd lam(n);
    for (uint32_t i = 0u; i < n; ++i) {
        const double t = (n == 1u) ? 0.0
                                   : static_cast<double>(i) /
                                         static_cast<double>(n - 1u);
        const double mag = std::pow(10.0, t * std::log10(kappa));  // [1, kappa]
        lam(i) = (i % 2u == 0u) ? mag : -mag;  // alternating sign -> indefinite
    }
    Eigen::MatrixXd A = Q * lam.asDiagonal() * Q.transpose();
    A = 0.5 * (A + A.transpose()).eval();
    Eigen::VectorXd b(n);
    for (uint32_t i = 0u; i < n; ++i) b(i) = u(rng);
    return Block{n, kappa, A, b};
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

std::vector<float> RunMinres(const nuka::phi::DeviceContext& ctx,
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

    auto solver = diffsim::MakeSparseSolverBackend("self_minres", ctx);
    solver->Solve(system, static_cast<const float*>(d_b.Data()),
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

// Max componentwise rel-err of the GPU MINRES solution vs Eigen FullPivLU on the
// SAME fp32-quantized (A,b). FullPivLU is the indefinite-valid dense oracle.
double MaxRelErr(const std::vector<Block>& blocks, const std::vector<float>& x,
                 double& worst_resid_out) {
    double max_rel = 0.0;
    double worst_resid = 0.0;
    for (size_t blk = 0u; blk < blocks.size(); ++blk) {
        const Block& B = blocks[blk];
        if (B.n == 0u) continue;
        const Eigen::MatrixXd A32 = Fp32(B.A);
        const Eigen::VectorXd b32 = Fp32(B.b);
        Eigen::VectorXd x_ref = A32.fullPivLu().solve(b32);
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
        // Also report the true relative residual ||A x - b|| / ||b|| (the MINRES
        // minimized quantity) -- the most honest correctness measure.
        const double resid = (A32 * x_gpu - b32).norm() /
                             std::max(b32.norm(), 1.0e-12);
        worst_resid = std::max(worst_resid, resid);
    }
    worst_resid_out = worst_resid;
    return max_rel;
}

std::vector<Block> MakeIndefiniteBatch() {
    std::mt19937 rng(0xBEEFu);
    std::vector<Block> blocks;
    for (double kappa : {1.0e1, 1.0e2, 1.0e3}) {
        for (uint32_t n = 2u; n <= kMd; ++n)
            blocks.push_back(MakeIndefiniteBlock(n, kappa, rng));
    }
    blocks.push_back(Block{0u, 1.0, Eigen::MatrixXd(0, 0), Eigen::VectorXd(0)});
    return blocks;
}

// GATE 2: self-written MINRES agrees with the dense FullPivLU oracle on genuinely
// INDEFINITE symmetric systems within 1e-5. Absolute-Jacobi preconditioner (SPD,
// the valid choice for indefinite MINRES), run to enough fixed iters to converge.
TEST(MinresVsDenseIndefinite, AgreesWithFullPivLu) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const std::vector<Block> blocks = MakeIndefiniteBatch();
    std::vector<float> values, rhs;
    std::vector<uint32_t> dims;
    PackBatch(blocks, values, dims, rhs);

    // n<=12 so 200 fixed iters >> n guarantees MINRES exhausts the Krylov space
    // (exact in <= n steps in fp64; the extra iters are no-ops after breakdown).
    const std::vector<float> x =
        RunMinres(ctx, values, dims, rhs, diffsim::Preconditioner::Jacobi, 200u,
                  true);
    double worst_resid = 0.0;
    const double rel = MaxRelErr(blocks, x, worst_resid);
    std::printf("[MinresVsDenseIndefinite] absJacobi vs fp32-FullPivLU: "
                "max rel-err = %.3e, worst ||Ax-b||/||b|| = %.3e\n",
                rel, worst_resid);
    EXPECT_LT(rel, 1.0e-5)
        << "self-written MINRES vs Eigen FullPivLU on indefinite symmetric blocks";
    EXPECT_LT(worst_resid, 1.0e-5)
        << "MINRES residual ||Ax-b||/||b|| on indefinite symmetric blocks";
}

// A DEFINITE (SPD) block: A = Q diag(lambda) Q^T, lambda all POSITIVE, log-spaced
// over [1, kappa]. The byte-safety baseline for the detector (must NOT flag).
Block MakeSpdBlock(uint32_t n, double kappa, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Eigen::MatrixXd M(n, n);
    for (uint32_t i = 0u; i < n; ++i)
        for (uint32_t j = 0u; j < n; ++j) M(i, j) = u(rng);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
    Eigen::MatrixXd Q = qr.householderQ();
    Eigen::VectorXd lam(n);
    for (uint32_t i = 0u; i < n; ++i) {
        const double t = (n == 1u) ? 0.0
                                   : static_cast<double>(i) /
                                         static_cast<double>(n - 1u);
        lam(i) = std::pow(10.0, t * std::log10(kappa));  // all > 0 -> SPD
    }
    Eigen::MatrixXd A = Q * lam.asDiagonal() * Q.transpose();
    A = 0.5 * (A + A.transpose()).eval();
    Eigen::VectorXd b(n);
    for (uint32_t i = 0u; i < n; ++i) b(i) = u(rng);
    return Block{n, kappa, A, b};
}

// A PSD block with EXACTLY ONE zero eigenvalue (a null direction): the byte-safety-
// critical case -- the detector's no-pivoting LDLT yields a pivot ~0 there, which
// the STRICT-negative test must NOT flag (else the SPD/PSD IFT path would mis-route).
Block MakePsdNullBlock(uint32_t n, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Eigen::MatrixXd M(n, n);
    for (uint32_t i = 0u; i < n; ++i)
        for (uint32_t j = 0u; j < n; ++j) M(i, j) = u(rng);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
    Eigen::MatrixXd Q = qr.householderQ();
    Eigen::VectorXd lam(n);
    for (uint32_t i = 0u; i < n; ++i) lam(i) = 1.0 + static_cast<double>(i);
    lam(n - 1u) = 0.0;  // one exact null direction -> PSD, NOT indefinite
    Eigen::MatrixXd A = Q * lam.asDiagonal() * Q.transpose();
    A = 0.5 * (A + A.transpose()).eval();
    Eigen::VectorXd b(n);
    for (uint32_t i = 0u; i < n; ++i) b(i) = u(rng);
    return Block{n, 1.0, A, b};
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

    diffsim::DetectBatchedIndefinite(ctx, system,
                                     static_cast<uint32_t*>(d_flag.Data()));
    ctx.stream.Synchronize();
    uint32_t flag = 0u;
    d_flag.CopyToHost(&flag, sizeof(uint32_t));
    return flag;
}

// THE BYTE-SAFETY EVIDENCE: the indefinite detector that gates ift_runner's auto-
// routing must return 0 (-> CG path runs untouched -> byte-identical) on every
// SPD/PSD batch, and 1 on a genuinely indefinite one. This is what proves the SPD
// IFT path is unchanged -- the regression tests alone cannot (MINRES would pass
// them too); only flag==0 guarantees the CG branch is taken.
TEST(MinresVsDenseIndefinite, IndefiniteDetectorBytesafe) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    std::mt19937 rng(0xD37u);

    // (a) SPD batch incl. an ill-conditioned kappa=1e6 block -> MUST NOT flag.
    std::vector<Block> spd;
    for (double kappa : {1.0e0, 1.0e3, 1.0e6})
        for (uint32_t n = 1u; n <= kMd; ++n) spd.push_back(MakeSpdBlock(n, kappa, rng));
    EXPECT_EQ(RunDetect(ctx, spd), 0u)
        << "indefinite detector FALSE-POSITIVE on an SPD batch (would mis-route the "
           "SPD IFT path to MINRES and break the byte-identical guarantee)";

    // (b) PSD null-direction blocks (one zero eigenvalue) -> MUST NOT flag (the
    //     byte-safety-critical case: a ~0 pivot is NOT strictly negative).
    std::vector<Block> psd;
    for (uint32_t n = 2u; n <= kMd; ++n) psd.push_back(MakePsdNullBlock(n, rng));
    EXPECT_EQ(RunDetect(ctx, psd), 0u)
        << "indefinite detector FALSE-POSITIVE on a PSD null-direction batch";

    // (c) genuinely indefinite batch -> MUST flag.
    std::vector<Block> indef;
    for (uint32_t n = 2u; n <= kMd; ++n) indef.push_back(MakeIndefiniteBlock(n, 1.0e2, rng));
    EXPECT_EQ(RunDetect(ctx, indef), 1u)
        << "indefinite detector FALSE-NEGATIVE on a genuinely indefinite batch";

    // (d) a single indefinite block mixed into an otherwise-SPD batch -> flag (the
    //     whole-batch routing trigger).
    std::vector<Block> mixed = spd;
    mixed.push_back(MakeIndefiniteBlock(6u, 1.0e2, rng));
    EXPECT_EQ(RunDetect(ctx, mixed), 1u)
        << "one indefinite block in an SPD batch must flag (whole-batch routing)";

    std::printf("[MinresVsDenseIndefinite] detector byte-safe: SPD/PSD->0, "
                "indefinite->1\n");
}

// D1: two MINRES solves of the same indefinite batch are byte-identical.
TEST(MinresVsDenseIndefinite, BitExact) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const std::vector<Block> blocks = MakeIndefiniteBatch();
    std::vector<float> values, rhs;
    std::vector<uint32_t> dims;
    PackBatch(blocks, values, dims, rhs);

    for (bool fixed : {false, true}) {
        const std::vector<float> x1 =
            RunMinres(ctx, values, dims, rhs, diffsim::Preconditioner::Jacobi, 200u,
                      fixed);
        const std::vector<float> x2 =
            RunMinres(ctx, values, dims, rhs, diffsim::Preconditioner::Jacobi, 200u,
                      fixed);
        ASSERT_EQ(x1.size(), x2.size());
        EXPECT_EQ(std::memcmp(x1.data(), x2.data(), x1.size() * sizeof(float)), 0)
            << "MINRES D1 bit-exact violated (indefinite batch), fixed_iters="
            << fixed;
    }
    std::printf("[MinresVsDenseIndefinite] MINRES two-run byte-identical\n");
}

}  // namespace
