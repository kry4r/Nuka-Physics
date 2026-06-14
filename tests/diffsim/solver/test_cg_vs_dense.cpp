// ---------------------------------------------------------------------------
// v0.7 p01 -- self-written CG vs a DENSE reference (Eigen LDLT) on the
// GENERAL / WORSE-CONDITIONED path. < 1e-6 componentwise + D1 bit-exact.
// ---------------------------------------------------------------------------
//
// The v0.5 test (tests/diffsim/test_cg_vs_dense.cpp) checks WELL-conditioned
// coupled SPD blocks (A = L L^T + n I, kappa ~ O(10)). This p01 test HARDENS the
// SAME warp-per-block self-written CG against deliberately WORSE-CONDITIONED SPD
// blocks built with an explicit eigenvalue spread A = Q diag(lambda) Q^T, kappa up
// to ~1e6, across the full block-size range n in {1..12} (the kMaxBlockDim cap).
//
// Eigen LDLT is the APPROVED TEST-ONLY dense oracle (never a production path); the
// self-written CG (fp64 internal) is the production solver. We assert:
//   (1) Jacobi-preconditioned CG run to enough fixed iters CONVERGES to the dense
//       LDLT solution within 1e-6 componentwise even at kappa ~ 1e6 (fp64 internal
//       arithmetic keeps the converged solution accurate; this is the honest
//       "hardened for worse-conditioned" claim);
//   (2) BlockJacobi (exact per-block Cholesky island solve) is near machine-exact;
//   (3) two solves of the same worse-conditioned batch are byte-identical (D1).
//
// HONESTY: kMaxBlockDim = 12 is the architectural block cap (the warp-per-block
// IFT layout stride). "Larger systems" means worse-conditioned 12x12 blocks, NOT
// bigger n -- that bound is documented here and in the report, not papered over.
// ---------------------------------------------------------------------------

#include "diffsim/sparse_solver_backend.hpp"
#include "diffsim/sparse_solver_cg.hpp"
#include "phi/backend.hpp"
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

// Test-only RAII device buffer over the phi v2 opaque Buffer*. Mirrors the legacy
// phi::Buffer surface (Data/Size/CopyFromHost/CopyToHost, move-only) so the test
// bodies stay byte-identical after the buffer sweep. Device-level DEFAULT
// (stream-0) allocation; CopyFromHost/CopyToHost run on stream 0 exactly like the
// legacy synchronous cudaMemcpy this replaces.
class OwnedDeviceBuffer {
public:
    OwnedDeviceBuffer() = default;
    explicit OwnedDeviceBuffer(size_t bytes) : bytes_(bytes) {
        buf_ = nuka::phi::BufferAlloc(
            nuka::phi::DeviceBufferType(nuka::phi::InitBestDevice()), bytes);
    }
    ~OwnedDeviceBuffer() { if (buf_ != nullptr) nuka::phi::BufferFree(buf_); }
    OwnedDeviceBuffer(OwnedDeviceBuffer&& o) noexcept
        : buf_(o.buf_), bytes_(o.bytes_) { o.buf_ = nullptr; o.bytes_ = 0u; }
    OwnedDeviceBuffer& operator=(OwnedDeviceBuffer&& o) noexcept {
        if (this != &o) {
            if (buf_ != nullptr) nuka::phi::BufferFree(buf_);
            buf_ = o.buf_; bytes_ = o.bytes_; o.buf_ = nullptr; o.bytes_ = 0u;
        }
        return *this;
    }
    OwnedDeviceBuffer(const OwnedDeviceBuffer&) = delete;
    OwnedDeviceBuffer& operator=(const OwnedDeviceBuffer&) = delete;
    void* Data() { return buf_ != nullptr ? nuka::phi::BufferBase(buf_) : nullptr; }
    const void* Data() const { return buf_ != nullptr ? nuka::phi::BufferBase(buf_) : nullptr; }
    size_t Size() const { return bytes_; }
    void CopyFromHost(const void* src, size_t bytes) {
        if (buf_ != nullptr && bytes > 0u) nuka::phi::BufferUpload(buf_, src, 0, bytes);
    }
    void CopyToHost(void* dst, size_t bytes) const {
        if (buf_ != nullptr && bytes > 0u) nuka::phi::BufferDownload(buf_, dst, 0, bytes);
    }
private:
    nuka::phi::Buffer* buf_ = nullptr;
    size_t bytes_ = 0u;
};

namespace diffsim = nuka::diffsim;
constexpr uint32_t kMd = diffsim::kMaxBlockDim;  // 12 (architectural block cap)
constexpr uint32_t kStride = kMd * kMd;          // 144

// One worse-conditioned SPD block of dimension n with a CONTROLLED condition
// number kappa. A = Q diag(lambda) Q^T with Q a Householder-orthonormal (oblique,
// non-axis-aligned) basis and lambda log-spaced in [lam_min, lam_min*kappa], so
// cond(A) == kappa exactly (up to fp). n==1 degenerates to a 1x1 (kappa irrelevant).
struct Block {
    uint32_t n = 0u;
    double kappa = 1.0;
    Eigen::MatrixXd A;  // n x n dense reference (fp64)
    Eigen::VectorXd b;  // n
};

Block MakeIllConditionedSpdBlock(uint32_t n, double kappa, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    // Orthonormal Q from a random n x n via Householder QR -> generic (oblique) axes.
    Eigen::MatrixXd M(n, n);
    for (uint32_t i = 0u; i < n; ++i)
        for (uint32_t j = 0u; j < n; ++j) M(i, j) = u(rng);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
    Eigen::MatrixXd Q = qr.householderQ();

    // Log-spaced eigenvalues spanning the requested kappa, smallest = 1.
    Eigen::VectorXd lam(n);
    if (n == 1u) {
        lam(0) = 1.0;
    } else {
        const double log_lo = 0.0;                // log10(1)
        const double log_hi = std::log10(kappa);  // log10(kappa)
        for (uint32_t i = 0u; i < n; ++i) {
            const double t = static_cast<double>(i) / static_cast<double>(n - 1u);
            lam(i) = std::pow(10.0, log_lo + t * (log_hi - log_lo));
        }
    }
    Eigen::MatrixXd A = Q * lam.asDiagonal() * Q.transpose();
    A = 0.5 * (A + A.transpose()).eval();  // exact symmetry against fp drift
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
OwnedDeviceBuffer UploadBuffer(const std::vector<T>& host) {
    OwnedDeviceBuffer buf(host.size() * sizeof(T));
    buf.CopyFromHost(host.data(), host.size() * sizeof(T));
    return buf;
}

std::vector<float> RunSolve(const nuka::phi::DeviceContext& ctx,
                            const std::vector<float>& values,
                            const std::vector<uint32_t>& dims,
                            const std::vector<float>& rhs,
                            diffsim::Preconditioner precond, uint32_t max_iter,
                            bool fixed_iters) {
    const uint32_t bc = static_cast<uint32_t>(dims.size());
    OwnedDeviceBuffer d_values = UploadBuffer(values);
    OwnedDeviceBuffer d_dims = UploadBuffer(dims);
    OwnedDeviceBuffer d_b = UploadBuffer(rhs);
    OwnedDeviceBuffer d_x(static_cast<size_t>(bc) * kMd * sizeof(float));

    diffsim::BatchedDenseSpdSystem system;
    system.block_count = bc;
    system.values = static_cast<const float*>(d_values.Data());
    system.block_dim = static_cast<const uint32_t*>(d_dims.Data());

    diffsim::SolveParams params;
    params.preconditioner = precond;
    params.max_iter = max_iter;
    params.tol = 1.0e-6f;
    params.run_to_fixed_iters = fixed_iters;

    auto solver = diffsim::MakeSparseSolverBackend("self_cg", ctx);
    solver->Solve(system, static_cast<const float*>(d_b.Data()),
                  static_cast<float*>(d_x.Data()), params);
    ctx.stream.Synchronize();

    std::vector<float> x(static_cast<size_t>(bc) * kMd, 0.0f);
    d_x.CopyToHost(x.data(), x.size() * sizeof(float));
    return x;
}

// fp32-roundtrip a matrix / vector: the EXACT bytes the GPU solver sees (the
// device stores A, b in fp32). Promoting these back to fp64 for the Eigen oracle
// is the correct experimental control -- it isolates SOLVER agreement (fp64 CG vs
// fp64 LDLT on identical bytes) from INPUT QUANTIZATION (fp64 truth -> fp32),
// which at kappa K perturbs the solution by ~K*eps_fp32 regardless of solver.
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

// Max componentwise rel-err of the GPU solution vs the Eigen LDLT oracle on the
// SAME fp32-quantized (A,b) the GPU consumed -> pure solver agreement.
double MaxRelErr(const std::vector<Block>& blocks, const std::vector<float>& x) {
    double max_rel = 0.0;
    for (size_t blk = 0u; blk < blocks.size(); ++blk) {
        const Block& B = blocks[blk];
        if (B.n == 0u) continue;
        const Eigen::MatrixXd A32 = Fp32(B.A);
        const Eigen::VectorXd b32 = Fp32(B.b);
        Eigen::VectorXd x_ref = A32.ldlt().solve(b32);
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

// Max componentwise rel-err of the GPU solution vs the TRUE-fp64 LDLT solution
// (the un-quantized A,b). This is dominated by input quantization (~kappa*eps);
// used only to DOCUMENT the conditioning bound, not as a solver-agreement gate.
double MaxRelErrVsTrueFp64(const std::vector<Block>& blocks,
                           const std::vector<float>& x) {
    double max_rel = 0.0;
    for (size_t blk = 0u; blk < blocks.size(); ++blk) {
        const Block& B = blocks[blk];
        if (B.n == 0u) continue;
        Eigen::VectorXd x_ref = B.A.ldlt().solve(B.b);  // un-quantized truth
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

// Worst kappa over a batch (for the documented kappa*eps input-precision bound).
double MaxKappa(const std::vector<Block>& blocks) {
    double k = 1.0;
    for (const Block& B : blocks)
        if (B.n > 0u) k = std::max(k, B.kappa);
    return k;
}

// A worse-conditioned batch: every n in {1..12} at three kappa tiers (1e2, 1e4,
// 1e6), plus an explicit empty block. Deterministic seed.
std::vector<Block> MakeWorseConditionedBatch() {
    std::mt19937 rng(0x5EEDu);
    std::vector<Block> blocks;
    for (double kappa : {1.0e2, 1.0e4, 1.0e6}) {
        for (uint32_t n = 1u; n <= kMd; ++n)
            blocks.push_back(MakeIllConditionedSpdBlock(n, kappa, rng));
    }
    blocks.push_back(Block{0u, 1.0, Eigen::MatrixXd(0, 0), Eigen::VectorXd(0)});
    return blocks;
}

TEST(CgVsDenseGeneral, AgreesWithEigenLdltOnWorseConditioned) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const std::vector<Block> blocks = MakeWorseConditionedBatch();
    std::vector<float> values, rhs;
    std::vector<uint32_t> dims;
    PackBatch(blocks, values, dims, rhs);

    // GATE (1e-6): SOLVER AGREEMENT against the dense LDLT oracle on the SAME
    // fp32-quantized (A,b) the GPU consumed. This isolates the solver (fp64 CG vs
    // fp64 LDLT on identical input bytes) from input quantization, so the gate is
    // meaningful at ALL tested kappa (up to 1e6) rather than collapsing into the
    // ~kappa*eps input-precision floor. Jacobi runs to a fixed 200 iters (>> n) so
    // CG fully converges even at kappa~1e6; BlockJacobi is the exact island solve.
    const std::vector<float> x_jacobi =
        RunSolve(ctx, values, dims, rhs, diffsim::Preconditioner::Jacobi, 200u, true);
    const double rel_jacobi = MaxRelErr(blocks, x_jacobi);
    std::printf("[CgVsDenseGeneral] Jacobi      vs fp32-LDLT (kappa up to 1e6) "
                "max rel-err = %.3e\n", rel_jacobi);
    EXPECT_LT(rel_jacobi, 1.0e-6)
        << "Jacobi CG vs Eigen LDLT (fp32-sourced) on worse-conditioned SPD blocks";

    const std::vector<float> x_block = RunSolve(
        ctx, values, dims, rhs, diffsim::Preconditioner::BlockJacobi, 64u, false);
    const double rel_block = MaxRelErr(blocks, x_block);
    std::printf("[CgVsDenseGeneral] BlockJacobi vs fp32-LDLT (kappa up to 1e6) "
                "max rel-err = %.3e\n", rel_block);
    EXPECT_LT(rel_block, 1.0e-6)
        << "BlockJacobi (exact Cholesky) vs Eigen LDLT (fp32-sourced) "
           "on worse-conditioned blocks";

    // DOCUMENTED CONDITIONING BOUND (informational, NOT a solver gate): against
    // the TRUE-fp64 (un-quantized) LDLT, the end-to-end error is dominated by
    // input quantization ~kappa*eps_fp32, NOT solver defect. We assert it stays
    // within a generous constant C * kappa_max * eps_fp32 -- this PROVES the limit
    // is fp32-input precision (the GPU stores A,b as fp32) rather than a bad solve.
    constexpr double kEpsFp32 = 1.1920929e-7;
    const double kappa_max = MaxKappa(blocks);
    const double cond_bound = 8.0 * kappa_max * kEpsFp32;  // C=8 generous headroom
    const double true_block = MaxRelErrVsTrueFp64(blocks, x_block);
    std::printf("[CgVsDenseGeneral] BlockJacobi vs TRUE-fp64 = %.3e  (kappa*eps "
                "bound %.3e, kappa_max=%.0e) -- input-precision-limited\n",
                true_block, cond_bound, kappa_max);
    EXPECT_LT(true_block, cond_bound)
        << "end-to-end error exceeds the documented kappa*eps_fp32 input-precision "
           "bound -- would indicate a solver defect, not input quantization";
}

TEST(CgVsDenseGeneral, BitExactOnWorseConditioned) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const std::vector<Block> blocks = MakeWorseConditionedBatch();
    std::vector<float> values, rhs;
    std::vector<uint32_t> dims;
    PackBatch(blocks, values, dims, rhs);

    for (auto precond :
         {diffsim::Preconditioner::Jacobi, diffsim::Preconditioner::BlockJacobi}) {
        for (bool fixed : {false, true}) {
            const std::vector<float> x1 =
                RunSolve(ctx, values, dims, rhs, precond, 200u, fixed);
            const std::vector<float> x2 =
                RunSolve(ctx, values, dims, rhs, precond, 200u, fixed);
            ASSERT_EQ(x1.size(), x2.size());
            const int cmp =
                std::memcmp(x1.data(), x2.data(), x1.size() * sizeof(float));
            EXPECT_EQ(cmp, 0)
                << "D1 bit-exact violated on worse-conditioned batch: precond="
                << static_cast<int>(precond) << " fixed_iters=" << fixed;
        }
    }
    std::printf("[CgVsDenseGeneral] worse-conditioned: all 4 paths byte-identical\n");
}

// Bundles all outputs of a diagnostics-ON solve for the two-run byte-exact gate.
struct DiagSolveOutputs {
    std::vector<float> x;
    std::vector<uint32_t> iters;
    std::vector<float> resid;
    std::vector<uint8_t> status;
    std::vector<float> history;
};

DiagSolveOutputs RunSolveWithDiag(const nuka::phi::DeviceContext& ctx,
                                  const std::vector<float>& values,
                                  const std::vector<uint32_t>& dims,
                                  const std::vector<float>& rhs,
                                  diffsim::Preconditioner precond,
                                  uint32_t max_iter, uint32_t cap) {
    const uint32_t bc = static_cast<uint32_t>(dims.size());
    OwnedDeviceBuffer d_values = UploadBuffer(values);
    OwnedDeviceBuffer d_dims = UploadBuffer(dims);
    OwnedDeviceBuffer d_b = UploadBuffer(rhs);
    OwnedDeviceBuffer d_x(static_cast<size_t>(bc) * kMd * sizeof(float));
    OwnedDeviceBuffer d_iters(static_cast<size_t>(bc) * sizeof(uint32_t));
    OwnedDeviceBuffer d_resid(static_cast<size_t>(bc) * sizeof(float));
    OwnedDeviceBuffer d_status(static_cast<size_t>(bc) * sizeof(uint8_t));
    OwnedDeviceBuffer d_hist(static_cast<size_t>(bc) * cap * sizeof(float));
    // Pre-fill the history with a fixed sentinel so untouched tail entries are
    // deterministic across runs (the kernel only writes the iters it runs).
    std::vector<float> hist_init(static_cast<size_t>(bc) * cap, -1.0f);
    d_hist.CopyFromHost(hist_init.data(), hist_init.size() * sizeof(float));

    diffsim::BatchedDenseSpdSystem system;
    system.block_count = bc;
    system.values = static_cast<const float*>(d_values.Data());
    system.block_dim = static_cast<const uint32_t*>(d_dims.Data());

    diffsim::SolveParams params;
    params.preconditioner = precond;
    params.max_iter = max_iter;
    params.tol = 1.0e-6f;
    params.diagnostics.iters_run = static_cast<uint32_t*>(d_iters.Data());
    params.diagnostics.final_rel_resid = static_cast<float*>(d_resid.Data());
    params.diagnostics.status = static_cast<uint8_t*>(d_status.Data());
    params.diagnostics.residual_history = static_cast<float*>(d_hist.Data());
    params.diagnostics.history_cap = cap;

    auto solver = diffsim::MakeSparseSolverBackend("self_cg", ctx);
    solver->Solve(system, static_cast<const float*>(d_b.Data()),
                  static_cast<float*>(d_x.Data()), params);
    ctx.stream.Synchronize();

    DiagSolveOutputs o;
    o.x.assign(static_cast<size_t>(bc) * kMd, 0.0f);
    o.iters.assign(bc, 0u);
    o.resid.assign(bc, 0.0f);
    o.status.assign(bc, 0u);
    o.history.assign(static_cast<size_t>(bc) * cap, 0.0f);
    d_x.CopyToHost(o.x.data(), o.x.size() * sizeof(float));
    d_iters.CopyToHost(o.iters.data(), o.iters.size() * sizeof(uint32_t));
    d_resid.CopyToHost(o.resid.data(), o.resid.size() * sizeof(float));
    d_status.CopyToHost(o.status.data(), o.status.size() * sizeof(uint8_t));
    d_hist.CopyToHost(o.history.data(), o.history.size() * sizeof(float));
    return o;
}

// GATE 1 (D1) for the NEW kernel outputs: a diagnostics-ON solve must be two-run
// byte-exact in BOTH x AND every diagnostic buffer (iters_run / final_rel_resid /
// status / residual_history). The diagnostics are new kernel writes, so the D1 gate
// must cover them, not just the diagnostics-off x path.
TEST(CgVsDenseGeneral, DiagnosticsOnIsBitExact) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const std::vector<Block> blocks = MakeWorseConditionedBatch();
    std::vector<float> values, rhs;
    std::vector<uint32_t> dims;
    PackBatch(blocks, values, dims, rhs);
    constexpr uint32_t cap = 32u;

    for (auto precond :
         {diffsim::Preconditioner::Jacobi, diffsim::Preconditioner::BlockJacobi}) {
        const DiagSolveOutputs a =
            RunSolveWithDiag(ctx, values, dims, rhs, precond, 200u, cap);
        const DiagSolveOutputs b =
            RunSolveWithDiag(ctx, values, dims, rhs, precond, 200u, cap);
        EXPECT_EQ(std::memcmp(a.x.data(), b.x.data(), a.x.size() * sizeof(float)), 0)
            << "diag-on x not bit-exact, precond=" << static_cast<int>(precond);
        EXPECT_EQ(std::memcmp(a.iters.data(), b.iters.data(),
                              a.iters.size() * sizeof(uint32_t)), 0)
            << "iters_run not bit-exact, precond=" << static_cast<int>(precond);
        EXPECT_EQ(std::memcmp(a.resid.data(), b.resid.data(),
                              a.resid.size() * sizeof(float)), 0)
            << "final_rel_resid not bit-exact, precond=" << static_cast<int>(precond);
        EXPECT_EQ(std::memcmp(a.status.data(), b.status.data(),
                              a.status.size() * sizeof(uint8_t)), 0)
            << "status not bit-exact, precond=" << static_cast<int>(precond);
        EXPECT_EQ(std::memcmp(a.history.data(), b.history.data(),
                              a.history.size() * sizeof(float)), 0)
            << "residual_history not bit-exact, precond=" << static_cast<int>(precond);
    }

    // The diagnostics-ON x must ALSO be byte-identical to the diagnostics-OFF x
    // (the opt-in side-channel must not perturb the solution -- the no-IFT-
    // regression guarantee). RunSolve (no diag) vs the diag-on x above.
    const std::vector<float> x_off =
        RunSolve(ctx, values, dims, rhs, diffsim::Preconditioner::Jacobi, 200u, false);
    const DiagSolveOutputs on =
        RunSolveWithDiag(ctx, values, dims, rhs, diffsim::Preconditioner::Jacobi,
                         200u, cap);
    ASSERT_EQ(x_off.size(), on.x.size());
    EXPECT_EQ(std::memcmp(x_off.data(), on.x.data(), x_off.size() * sizeof(float)), 0)
        << "diagnostics-ON perturbed x vs diagnostics-OFF (side-channel not inert)";
    std::printf("[CgVsDenseGeneral] diagnostics-ON: two-run byte-exact (x+all diag "
                "buffers) AND diag-on x == diag-off x\n");
}

}  // namespace
