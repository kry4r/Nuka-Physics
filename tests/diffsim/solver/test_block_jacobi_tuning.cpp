// ---------------------------------------------------------------------------
// v0.7 p01 -- Block-Jacobi vs Jacobi preconditioner sweep (the "tuning" test).
// ---------------------------------------------------------------------------
//
// HONEST SCOPE NOTE (the spec asks for a "block-size / island-grouping tuning
// sweep"): under the warp-per-block architecture there is NO sub-block size to
// tune and NO island grouping to sweep. Each constraint island IS exactly one
// dense block (n <= kMaxBlockDim = 12), solved by ONE warp; the island partition
// is fixed by the row scheduler upstream (deterministic), not a solver knob. And
// "Block-Jacobi" here is the EXACT per-block Cholesky of the whole block (the
// block diagonal == the whole island), so it is the exact island solve, not a
// block-SIZE-parameterized approximate preconditioner. There is therefore no
// fabricated block-size knob to sweep.
//
// What IS meaningful -- and what this test measures -- is the preconditioner-
// QUALITY tuning decision the general path actually makes: Jacobi (diagonal,
// genuine multi-iteration CG) vs Block-Jacobi (exact Cholesky, <= 1 iteration) as
// CONDITIONING worsens. We sweep kappa and, using the opt-in convergence
// diagnostics, report Jacobi's iterations-to-tol vs Block-Jacobi's (1), i.e. the
// convergence-rate speedup that justifies preferring Block-Jacobi on tightly
// coupled / worse-conditioned blocks. We assert (a) Block-Jacobi converges in <= 1
// iteration regardless of kappa, (b) Jacobi's iteration count grows with kappa,
// (c) Block-Jacobi is at least as fast as Jacobi (the spec's "2-5x better"
// expectation -- here unboundedly better as kappa grows because BlockJacobi is the
// exact solve). Both solves agree with the dense LDLT oracle.
// ---------------------------------------------------------------------------

#include "diffsim/sparse_solver_backend.hpp"
#include "diffsim/sparse_solver_cg.hpp"
#include "diffsim/solver/cg_diagnostics.hpp"
#include "phi/backend.hpp"
#include "phi/buffer.hpp"
#include "phi/scoped_device_guard.hpp"
#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
constexpr uint32_t kMd = diffsim::kMaxBlockDim;  // 12
constexpr uint32_t kStride = kMd * kMd;          // 144

template <typename T>
OwnedDeviceBuffer UploadBuffer(const std::vector<T>& host) {
    OwnedDeviceBuffer buf(host.size() * sizeof(T));
    buf.CopyFromHost(host.data(), host.size() * sizeof(T));
    return buf;
}

Eigen::MatrixXd MakeSpd(uint32_t n, double kappa, std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Eigen::MatrixXd M(n, n);
    for (uint32_t i = 0u; i < n; ++i)
        for (uint32_t j = 0u; j < n; ++j) M(i, j) = u(rng);
    Eigen::MatrixXd Q = Eigen::HouseholderQR<Eigen::MatrixXd>(M).householderQ();
    Eigen::VectorXd lam(n);
    for (uint32_t i = 0u; i < n; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(n - 1u);
        lam(i) = std::pow(10.0, t * std::log10(kappa));
    }
    Eigen::MatrixXd A = Q * lam.asDiagonal() * Q.transpose();
    return (0.5 * (A + A.transpose())).eval();
}

// Solve one block with diagnostics; return (iters_run, final_rel_resid, x).
struct SolveOut {
    uint32_t iters = 0u;
    float final_rel_resid = 0.0f;
    uint8_t status = 0u;
    Eigen::VectorXd x;
};

SolveOut SolveWithDiag(cudaStream_t ctx, int ctx_dev,
                       const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
                       diffsim::Preconditioner precond, uint32_t max_iter) {
    const uint32_t n = static_cast<uint32_t>(A.rows());
    std::vector<float> values(kStride, 0.0f);
    std::vector<uint32_t> dims(1u, n);
    std::vector<float> rhs(kMd, 0.0f);
    for (uint32_t i = 0u; i < n; ++i) {
        for (uint32_t j = 0u; j < n; ++j)
            values[i * kMd + j] = static_cast<float>(A(i, j));
        rhs[i] = static_cast<float>(b(i));
    }
    OwnedDeviceBuffer d_values = UploadBuffer(values);
    OwnedDeviceBuffer d_dims = UploadBuffer(dims);
    OwnedDeviceBuffer d_b = UploadBuffer(rhs);
    OwnedDeviceBuffer d_x(static_cast<size_t>(kMd) * sizeof(float));
    OwnedDeviceBuffer d_iters(sizeof(uint32_t));
    OwnedDeviceBuffer d_resid(sizeof(float));
    OwnedDeviceBuffer d_status(sizeof(uint8_t));

    diffsim::BatchedDenseSpdSystem system;
    system.block_count = 1u;
    system.values = static_cast<const float*>(d_values.Data());
    system.block_dim = static_cast<const uint32_t*>(d_dims.Data());

    diffsim::SolveParams params;
    params.preconditioner = precond;
    params.max_iter = max_iter;
    params.tol = 1.0e-6f;
    params.run_to_fixed_iters = false;  // tol-gated: iters_run = actual cost
    params.diagnostics.iters_run = static_cast<uint32_t*>(d_iters.Data());
    params.diagnostics.final_rel_resid = static_cast<float*>(d_resid.Data());
    params.diagnostics.status = static_cast<uint8_t*>(d_status.Data());

    auto solver = diffsim::MakeSparseSolverBackend("self_cg", ctx, ctx_dev);
    solver->Solve(system, static_cast<const float*>(d_b.Data()),
                  static_cast<float*>(d_x.Data()), params);
    cudaStreamSynchronize(ctx);

    SolveOut out;
    std::vector<float> x(kMd, 0.0f);
    d_x.CopyToHost(x.data(), x.size() * sizeof(float));
    d_iters.CopyToHost(&out.iters, sizeof(out.iters));
    d_resid.CopyToHost(&out.final_rel_resid, sizeof(out.final_rel_resid));
    d_status.CopyToHost(&out.status, sizeof(out.status));
    out.x = Eigen::VectorXd(n);
    for (uint32_t i = 0u; i < n; ++i) out.x(i) = static_cast<double>(x[i]);
    return out;
}

// fp32-roundtrip (the EXACT bytes the GPU consumed) so the rel-err measures
// SOLVER agreement (fp64 CG vs fp64 LDLT on identical input bytes), not the
// ~kappa*eps_fp32 input quantization. See test_cg_vs_dense.cpp for the rationale.
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

double RelErr(const Eigen::MatrixXd& A, const Eigen::VectorXd& b,
              const Eigen::VectorXd& x) {
    const Eigen::VectorXd xr = Fp32(A).ldlt().solve(Fp32(b));
    return (x - xr).norm() / std::max(xr.norm(), 1.0e-12);
}

TEST(BlockJacobiTuning, QualitySweepVsConditioning) {
    const cudaStream_t ctx = nullptr;  // BUF-14: stream 0
    const int ctx_dev = 0;
    constexpr uint32_t n = kMd;  // 12: the largest island the cap allows
    constexpr uint32_t max_iter = 200u;
    std::mt19937 rng(0xB10C4u);

    std::printf("[BlockJacobiTuning] n=%u (island == 1 block == 1 warp; no block-"
                "size/grouping knob under warp-per-block)\n", n);
    std::printf("[BlockJacobiTuning] %-10s %-14s %-14s %-12s %-12s\n", "kappa",
                "Jacobi_iters", "Block_iters", "Jac_relerr", "Blk_relerr");

    uint32_t prev_jacobi_iters = 0u;
    bool jacobi_grew = false;
    for (double kappa : {1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5}) {
        const Eigen::MatrixXd A = MakeSpd(n, kappa, rng);
        Eigen::VectorXd b(n);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for (uint32_t i = 0u; i < n; ++i) b(i) = u(rng);

        const SolveOut jac =
            SolveWithDiag(ctx, ctx_dev, A, b, diffsim::Preconditioner::Jacobi, max_iter);
        const SolveOut blk =
            SolveWithDiag(ctx, ctx_dev, A, b, diffsim::Preconditioner::BlockJacobi, max_iter);
        const double jac_err = RelErr(A, b, jac.x);
        const double blk_err = RelErr(A, b, blk.x);

        std::printf("[BlockJacobiTuning] %-10.0e %-14u %-14u %-12.2e %-12.2e\n",
                    kappa, jac.iters, blk.iters, jac_err, blk_err);

        // (a) Block-Jacobi (exact Cholesky island solve) converges in <= 1 iter
        //     regardless of conditioning.
        EXPECT_LE(blk.iters, 1u)
            << "BlockJacobi should be the exact island solve (<=1 iter), kappa="
            << kappa;
        // (c) Block-Jacobi is at least as fast as Jacobi (the convergence-rate win).
        EXPECT_LE(blk.iters, jac.iters)
            << "BlockJacobi must not need more iters than Jacobi, kappa=" << kappa;
        // Both agree with the dense oracle on the SAME fp32 input bytes (pure
        // solver agreement -- correct, not just fast). fp64 internal arithmetic
        // keeps this ~1e-6 across the whole kappa sweep.
        EXPECT_LT(blk_err, 1.0e-6) << "BlockJacobi vs fp32-LDLT, kappa=" << kappa;
        EXPECT_LT(jac_err, 1.0e-6) << "Jacobi vs fp32-LDLT, kappa=" << kappa;

        if (kappa > 1.0e1 && jac.iters > prev_jacobi_iters) jacobi_grew = true;
        prev_jacobi_iters = jac.iters;
    }
    // (b) Jacobi's iteration count grows as conditioning worsens (the very reason
    //     Block-Jacobi is preferred on tightly coupled / worse-conditioned blocks).
    EXPECT_TRUE(jacobi_grew)
        << "Jacobi iterations did not grow with kappa -- the conditioning sweep "
           "should make the diagonal preconditioner work harder";
}

// Exercises the host-side diagnostics summarizer (SummarizeCgDiagnostics) over a
// multi-block batch: it must aggregate iters / residual / status deterministically.
TEST(BlockJacobiTuning, HostSummarizerAggregatesBatch) {
    const cudaStream_t ctx = nullptr;  // BUF-14: stream 0
    const int ctx_dev = 0;
    std::mt19937 rng(0x5A1ADu);
    constexpr uint32_t bc = 6u;
    std::vector<float> values(static_cast<size_t>(bc) * kStride, 0.0f);
    std::vector<uint32_t> dims(bc, 0u);
    std::vector<float> rhs(static_cast<size_t>(bc) * kMd, 0.0f);
    for (uint32_t blk = 0u; blk < bc; ++blk) {
        const uint32_t n = (blk == bc - 1u) ? 0u : (4u + blk);  // last is empty
        dims[blk] = n;
        if (n == 0u) continue;
        const double kappa = std::pow(10.0, 1.0 + blk);  // 1e1..1e5
        const Eigen::MatrixXd A = MakeSpd(n, kappa, rng);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        for (uint32_t i = 0u; i < n; ++i) {
            for (uint32_t j = 0u; j < n; ++j)
                values[blk * kStride + i * kMd + j] = static_cast<float>(A(i, j));
            rhs[blk * kMd + i] = static_cast<float>(u(rng));
        }
    }
    OwnedDeviceBuffer d_values = UploadBuffer(values);
    OwnedDeviceBuffer d_dims = UploadBuffer(dims);
    OwnedDeviceBuffer d_b = UploadBuffer(rhs);
    OwnedDeviceBuffer d_x(static_cast<size_t>(bc) * kMd * sizeof(float));
    OwnedDeviceBuffer d_iters(static_cast<size_t>(bc) * sizeof(uint32_t));
    OwnedDeviceBuffer d_resid(static_cast<size_t>(bc) * sizeof(float));
    OwnedDeviceBuffer d_status(static_cast<size_t>(bc) * sizeof(uint8_t));

    diffsim::BatchedDenseSpdSystem system;
    system.block_count = bc;
    system.values = static_cast<const float*>(d_values.Data());
    system.block_dim = static_cast<const uint32_t*>(d_dims.Data());

    diffsim::SolveParams params;
    params.preconditioner = diffsim::Preconditioner::Jacobi;
    params.max_iter = 200u;
    params.tol = 1.0e-6f;
    params.diagnostics.iters_run = static_cast<uint32_t*>(d_iters.Data());
    params.diagnostics.final_rel_resid = static_cast<float*>(d_resid.Data());
    params.diagnostics.status = static_cast<uint8_t*>(d_status.Data());

    auto solver = diffsim::MakeSparseSolverBackend("self_cg", ctx, ctx_dev);
    solver->Solve(system, static_cast<const float*>(d_b.Data()),
                  static_cast<float*>(d_x.Data()), params);
    cudaStreamSynchronize(ctx);

    std::vector<uint32_t> iters(bc, 0u);
    std::vector<float> resid(bc, 0.0f);
    std::vector<uint8_t> status(bc, 0u);
    d_iters.CopyToHost(iters.data(), iters.size() * sizeof(uint32_t));
    d_resid.CopyToHost(resid.data(), resid.size() * sizeof(float));
    d_status.CopyToHost(status.data(), status.size() * sizeof(uint8_t));

    const diffsim::CgDiagnosticsSummary s = diffsim::SummarizeCgDiagnostics(
        bc, iters.data(), resid.data(), status.data(), dims.data());

    std::printf("[BlockJacobiTuning] summary: blocks=%u max_iters=%u total_iters=%u "
                "max_resid=%.2e converged=%u maxiter=%u stalled=%u diverged=%u\n",
                s.blocks, s.max_iters_run, s.total_iters_run, s.max_final_rel_resid,
                s.converged_blocks, s.maxiter_blocks, s.stalled_blocks,
                s.diverged_blocks);

    // bc-1 non-empty blocks (the last is n==0, skipped by block_dim).
    EXPECT_EQ(s.blocks, bc - 1u);
    EXPECT_GT(s.max_iters_run, 0u);
    EXPECT_GE(s.total_iters_run, s.max_iters_run);
    // All non-empty blocks converged within the 200-iter budget (SPD, fp64).
    EXPECT_EQ(s.converged_blocks, bc - 1u);
    EXPECT_EQ(s.diverged_blocks, 0u) << "SPD CG must not diverge";
    EXPECT_LT(s.max_final_rel_resid, 1.0e-5f);
}

// POSITIVE divergence test: the detector must actually FIRE on a genuinely non-SPD
// operator -- the other tests only prove it does NOT false-fire on healthy SPD
// input. We feed a NEGATIVE-DEFINITE block A = -(L L^T + n I). At iteration 0 the
// CG scalar pAp = p^T A p < 0, so the solve-time SPD guard trips (broke_down) BEFORE
// convergence -> kCgStatusDiverged is set. The solution must also stay FINITE (the
// alpha=0 freeze prevents a NaN blow-up). This is the runtime SPD-failure detection
// the brief requires ("keep symmetry/SPD detection"), proven to engage.
TEST(BlockJacobiTuning, DivergenceFlagFiresOnNonSpd) {
    const cudaStream_t ctx = nullptr;  // BUF-14: stream 0
    const int ctx_dev = 0;
    constexpr uint32_t n = 6u;
    std::mt19937 rng(0xDEADu);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    // Negative-definite: A = -(L L^T + n I) -> all eigenvalues < 0 -> NOT SPD.
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(n, n);
    for (uint32_t i = 0u; i < n; ++i)
        for (uint32_t j = 0u; j <= i; ++j) L(i, j) = u(rng);
    Eigen::MatrixXd A =
        -(L * L.transpose() + static_cast<double>(n) * Eigen::MatrixXd::Identity(n, n));
    A = 0.5 * (A + A.transpose()).eval();
    Eigen::VectorXd b(n);
    for (uint32_t i = 0u; i < n; ++i) b(i) = u(rng);

    // Jacobi here: BlockJacobi's modified Cholesky treats a negative pivot as a null
    // direction (a different, also-finite path); the divergence SIGNAL we are
    // validating is the CG SPD guard (pAp<0), exercised by genuine CG iteration.
    const SolveOut out = SolveWithDiag(ctx, ctx_dev, A, b, diffsim::Preconditioner::Jacobi, 64u);

    std::printf("[BlockJacobiTuning] non-SPD (neg-def): status=0x%02x iters=%u "
                "final_rel_resid=%.3e\n", out.status, out.iters, out.final_rel_resid);

    // (1) The divergence/breakdown flag fired (the detector engaged on real non-SPD).
    EXPECT_NE(out.status & diffsim::kCgStatusDiverged, 0)
        << "divergence flag did NOT fire on a negative-definite (non-SPD) operator";
    // (2) The solution stayed finite (the alpha/beta freeze prevented a NaN).
    for (uint32_t i = 0u; i < n; ++i)
        EXPECT_TRUE(std::isfinite(out.x(i)))
            << "non-SPD solve produced non-finite x[" << i << "]";
}

}  // namespace
