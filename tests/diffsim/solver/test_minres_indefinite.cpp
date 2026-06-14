// ---------------------------------------------------------------------------
// v0.7 p02 -- self-written MINRES on a KNOWN symmetric INDEFINITE system.
// ---------------------------------------------------------------------------
//
// A focused correctness test (the spec's MinresIndefinite.KnownTestProblem): build
// a small symmetric matrix with BOTH positive and negative eigenvalues, solve with
// the self-written MINRES, and verify the residual ||A x - b|| is small. Also
// exercises a 2x2 saddle-point KKT block [[H, b^T],[b, 0]] -- POSITIVE diagonal but
// INDEFINITE (eigenvalues straddle zero), the canonical case CG cannot handle and a
// raw-diagonal indefinite detector would mis-classify. D1 bit-exact two-run.
// ---------------------------------------------------------------------------

#include "diffsim/sparse_solver_backend.hpp"
#include "phi/backend.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
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

// Solve a single dense symmetric block (n x n) with self-written MINRES.
std::vector<float> SolveOne(const nuka::phi::DeviceContext& ctx,
                            const std::vector<float>& dense_nn, uint32_t n,
                            const std::vector<float>& b) {
    std::vector<float> values(kStride, 0.0f);
    std::vector<uint32_t> dims(1u, n);
    std::vector<float> rhs(kMd, 0.0f);
    for (uint32_t i = 0u; i < n; ++i) {
        for (uint32_t j = 0u; j < n; ++j) values[i * kMd + j] = dense_nn[i * n + j];
        rhs[i] = b[i];
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
    params.preconditioner = diffsim::Preconditioner::Jacobi;  // absolute-Jacobi
    params.max_iter = 100u;
    params.tol = 1.0e-7f;
    params.run_to_fixed_iters = true;

    auto solver = diffsim::MakeSparseSolverBackend("self_minres", ctx);
    solver->Solve(system, static_cast<const float*>(d_b.Data()),
                  static_cast<float*>(d_x.Data()), params);
    ctx.stream.Synchronize();

    std::vector<float> x(kMd, 0.0f);
    d_x.CopyToHost(x.data(), x.size() * sizeof(float));
    return x;
}

double Residual(const std::vector<float>& dense_nn, uint32_t n,
                const std::vector<float>& b, const std::vector<float>& x) {
    double r2 = 0.0, b2 = 0.0;
    for (uint32_t i = 0u; i < n; ++i) {
        double ax = 0.0;
        for (uint32_t j = 0u; j < n; ++j)
            ax += static_cast<double>(dense_nn[i * n + j]) *
                  static_cast<double>(x[j]);
        const double ri = ax - static_cast<double>(b[i]);
        r2 += ri * ri;
        b2 += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    return std::sqrt(r2) / std::sqrt(std::max(b2, 1.0e-30));
}

TEST(MinresIndefinite, KnownTestProblem) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    // Symmetric indefinite 3x3: eigenvalues {+4, +1, -2} (mixed sign).
    //   diag(1, 3, -1) rotated a little to couple the rows.
    const uint32_t n = 3u;
    const std::vector<float> A = {
        2.0f,  1.0f,  0.0f,
        1.0f,  1.0f,  1.0f,
        0.0f,  1.0f, -2.0f};  // symmetric, det<0 -> indefinite
    const std::vector<float> b = {1.0f, -2.0f, 0.5f};
    const std::vector<float> x = SolveOne(ctx, A, n, b);
    const double rel = Residual(A, n, b, x);
    std::printf("[MinresIndefinite] 3x3 indefinite: ||Ax-b||/||b|| = %.3e\n", rel);
    EXPECT_LT(rel, 1.0e-5) << "MINRES failed on a known indefinite 3x3";
}

TEST(MinresIndefinite, SaddlePointKkt) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    // 2x2 saddle point [[H, c],[c, 0]] with H>0: POSITIVE+ZERO diagonal but
    // INDEFINITE (det = -c^2 < 0). The case a raw-diagonal detector mis-routes.
    const uint32_t n = 2u;
    const std::vector<float> A = {3.0f, 2.0f, 2.0f, 0.0f};
    const std::vector<float> b = {1.0f, 1.0f};
    const std::vector<float> x = SolveOne(ctx, A, n, b);
    const double rel = Residual(A, n, b, x);
    std::printf("[MinresIndefinite] 2x2 saddle (pos diag, indefinite): "
                "||Ax-b||/||b|| = %.3e\n", rel);
    EXPECT_LT(rel, 1.0e-5) << "MINRES failed on a 2x2 KKT saddle point";
}

TEST(MinresIndefinite, BitExact) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const uint32_t n = 3u;
    const std::vector<float> A = {2.0f, 1.0f, 0.0f, 1.0f, 1.0f, 1.0f,
                                  0.0f, 1.0f, -2.0f};
    const std::vector<float> b = {1.0f, -2.0f, 0.5f};
    const std::vector<float> x1 = SolveOne(ctx, A, n, b);
    const std::vector<float> x2 = SolveOne(ctx, A, n, b);
    EXPECT_EQ(std::memcmp(x1.data(), x2.data(), x1.size() * sizeof(float)), 0)
        << "MINRES not byte-identical two-run on the indefinite 3x3";
    std::printf("[MinresIndefinite] two-run byte-identical\n");
}

}  // namespace
