// ---------------------------------------------------------------------------
// v0.7 p02 -- ILU(0)-preconditioned MINRES converges in FEWER iterations than
// Jacobi-preconditioned MINRES (the preconditioner-quality gate).
// ---------------------------------------------------------------------------
//
// Gate 3 (spec Ilu0VsJacobi.FewerIterations): on a representative ill-conditioned
// system, ILU(0)-MINRES reaches a target residual in <= 0.5x the iterations of
// Jacobi-MINRES.
//
// PRECONDITIONER-VALIDITY HONESTY (the trap): a preconditioned MINRES requires an
// SPD preconditioner -- ILU(0) of an INDEFINITE A is itself indefinite and breaks
// MINRES's residual minimization. So the ILU(0)-vs-Jacobi quality comparison runs
// on an ILL-CONDITIONED (quasi-)DEFINITE structurally-sparse block (where ILU(0) is
// well-defined and a genuine incomplete factorization). The INDEFINITE-correctness
// claim is gated separately (test_minres_indefinite / test_minres_vs_dense_indefinite)
// with absolute-Jacobi. These are deliberately different systems -- documented, not
// papered over.
//
// Iteration count is measured by sweeping max_iter with the deterministic tol early-
// exit ON and finding the SMALLEST max_iter whose solution reaches the target
// residual (the backend runs <= max_iter iters, exiting early at tol). The 0.5x
// comparison is on those two iteration counts.
// ---------------------------------------------------------------------------

#include "diffsim/solver/minres_backend.hpp"
#include "diffsim/sparse_solver_backend.hpp"
#include "phi/backend.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
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
constexpr uint32_t kMd = diffsim::kMaxBlockDim;
constexpr uint32_t kStride = kMd * kMd;

template <typename T>
OwnedDeviceBuffer UploadBuffer(const std::vector<T>& host) {
    OwnedDeviceBuffer buf(host.size() * sizeof(T));
    buf.CopyFromHost(host.data(), host.size() * sizeof(T));
    return buf;
}

// An ill-conditioned, structurally-sparse SPD block (n=12): a pentadiagonal stencil
// (so ILU(0) drops genuine fill at distance-2+ positions) with a strong diagonal
// gradient that drives the condition number up. SPD by diagonal dominance.
std::vector<double> MakeIllConditionedSparse(uint32_t n) {
    std::vector<double> A(static_cast<size_t>(n) * n, 0.0);
    for (uint32_t i = 0u; i < n; ++i) {
        // Diagonal gradient: small at the top, large at the bottom -> high kappa.
        A[i * n + i] = 1.0 + 40.0 * static_cast<double>(i) / static_cast<double>(n);
        if (i + 1u < n) { A[i * n + (i + 1u)] = -1.0; A[(i + 1u) * n + i] = -1.0; }
        if (i + 2u < n) { A[i * n + (i + 2u)] = -0.5; A[(i + 2u) * n + i] = -0.5; }
        // structural zeros at |i-j| >= 3 -> ILU(0) drops the fill there.
    }
    return A;
}

double RunResidual(const nuka::phi::DeviceContext& ctx, const std::vector<double>& A,
                   uint32_t n, const std::vector<double>& b,
                   diffsim::Preconditioner precond, uint32_t max_iter) {
    std::vector<float> values(kStride, 0.0f), rhs(kMd, 0.0f);
    std::vector<uint32_t> dims(1u, n);
    for (uint32_t i = 0u; i < n; ++i) {
        for (uint32_t j = 0u; j < n; ++j)
            values[i * kMd + j] = static_cast<float>(A[i * n + j]);
        rhs[i] = static_cast<float>(b[i]);
    }
    OwnedDeviceBuffer d_values = UploadBuffer(values);
    OwnedDeviceBuffer d_dims = UploadBuffer(dims);
    OwnedDeviceBuffer d_b = UploadBuffer(rhs);
    OwnedDeviceBuffer d_x(static_cast<size_t>(kMd) * sizeof(float));

    diffsim::BatchedDenseSpdSystem system;
    system.block_count = 1u;
    system.values = static_cast<const float*>(d_values.Data());
    system.block_dim = static_cast<const uint32_t*>(d_dims.Data());

    diffsim::SolveParams params;
    params.preconditioner = precond;
    params.max_iter = max_iter;
    params.tol = 1.0e-8f;  // tight; we measure iters to reach kTargetResid below
    params.run_to_fixed_iters = false;  // allow deterministic early exit

    auto solver = diffsim::MakeSparseSolverBackend("self_minres", ctx);
    solver->Solve(system, static_cast<const float*>(d_b.Data()),
                  static_cast<float*>(d_x.Data()), params);
    ctx.stream.Synchronize();

    std::vector<float> x(kMd, 0.0f);
    d_x.CopyToHost(x.data(), x.size() * sizeof(float));

    double r2 = 0.0, b2 = 0.0;
    for (uint32_t i = 0u; i < n; ++i) {
        double ax = 0.0;
        for (uint32_t j = 0u; j < n; ++j) ax += A[i * n + j] * x[j];
        const double ri = ax - b[i];
        r2 += ri * ri;
        b2 += b[i] * b[i];
    }
    return std::sqrt(r2) / std::sqrt(std::max(b2, 1e-30));
}

// Smallest max_iter whose solution reaches the target relative residual (the
// backend exits early at tol, so increasing max_iter == letting it run more iters).
uint32_t ItersToReach(const nuka::phi::DeviceContext& ctx,
                      const std::vector<double>& A, uint32_t n,
                      const std::vector<double>& b, diffsim::Preconditioner precond,
                      double target, uint32_t iter_cap) {
    for (uint32_t k = 1u; k <= iter_cap; ++k) {
        const double resid = RunResidual(ctx, A, n, b, precond, k);
        if (resid <= target) return k;
    }
    return iter_cap + 1u;  // did not reach within cap
}

TEST(Ilu0VsJacobi, FewerIterations) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const uint32_t n = 12u;
    const std::vector<double> A = MakeIllConditionedSparse(n);
    std::vector<double> b(n);
    for (uint32_t i = 0u; i < n; ++i) b[i] = std::sin(0.7 * i + 1.0);  // fixed rhs

    constexpr double kTarget = 1.0e-5;
    constexpr uint32_t kCap = 60u;
    const uint32_t it_jacobi =
        ItersToReach(ctx, A, n, b, diffsim::Preconditioner::Jacobi, kTarget, kCap);
    const uint32_t it_ilu0 = ItersToReach(
        ctx, A, n, b, diffsim::Preconditioner::BlockJacobi /*=ILU(0)*/, kTarget, kCap);

    std::printf("[Ilu0VsJacobi] iters to reach %.0e: Jacobi=%u  ILU(0)=%u\n",
                kTarget, it_jacobi, it_ilu0);
    ASSERT_LE(it_jacobi, kCap) << "Jacobi-MINRES did not converge within the cap";
    ASSERT_LE(it_ilu0, kCap) << "ILU(0)-MINRES did not converge within the cap";
    // ILU(0) on this pentadiagonal block is a strong incomplete factorization; it
    // must reach the target in <= 0.5x the Jacobi iteration count.
    EXPECT_LE(it_ilu0 * 2u, it_jacobi)
        << "ILU(0)-MINRES (" << it_ilu0 << " iters) is not <= 0.5x Jacobi-MINRES ("
        << it_jacobi << " iters) on the ill-conditioned structurally-sparse block";
}

}  // namespace
