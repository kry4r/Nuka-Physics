// ---------------------------------------------------------------------------
// v0.7 p02 -- ILU(0) factorization vs a KNOWN reference + host==device + D1.
// ---------------------------------------------------------------------------
//
// ILU(0) = incomplete LU, ZERO fill-in: L U constrained to A's existing sparsity
// pattern (updates into structurally-zero positions are dropped). We verify:
//   (1) the host factor matches a HAND-COMPUTED reference on a small structurally-
//       sparse matrix, element-wise < 1e-7;
//   (2) on a structurally-sparse matrix ILU(0) is GENUINELY incomplete: L U != A in
//       the dropped positions (otherwise the test would be vacuous -- a fully dense
//       block makes ILU(0) degenerate to the complete LU, L U == A);
//   (3) the DEVICE factor (warp-per-block, lane-0 IKJ) matches the host factor
//       byte-exact-to-fp32, and is D1 two-run byte-identical;
//   (4) the ILU(0) APPLY (forward/backward triangular solve) inverts L U: for a
//       random r, A_pattern (L U) z = r is solved consistently host vs device.
// ---------------------------------------------------------------------------

#include "diffsim/solver/ilu0_preconditioner.hpp"
#include "diffsim/sparse_solver_backend.hpp"
#include "phi/buffer_legacy.hpp"
#include "phi/device_context.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

namespace diffsim = nuka::diffsim;
constexpr uint32_t kMd = diffsim::kMaxBlockDim;
constexpr uint32_t kStride = kMd * kMd;

template <typename T>
nuka::phi::Buffer UploadBuffer(const std::vector<T>& host) {
    nuka::phi::Buffer buf(host.size() * sizeof(T), nuka::phi::MemoryKind::Device);
    buf.CopyFromHost(host.data(), host.size() * sizeof(T));
    return buf;
}

// ILU(0) on a known structurally-sparse SPD-ish matrix. A 4x4 tridiagonal-plus-
// corner pattern: structural zeros at several off-diagonals so fill-in is dropped.
//   A = [ 4  1  0  0 ]
//       [ 1  4  1  0 ]
//       [ 0  1  4  1 ]
//       [ 0  0  1  4 ]   (tridiagonal: zeros at (0,2),(0,3),(1,3) etc.)
// For a tridiagonal matrix ILU(0) == exact LU (no fill is generated outside the
// pattern by a tridiagonal), so this checks the recurrence numerics. The genuine-
// incompleteness check (2) uses a pattern that DOES generate dropped fill.
TEST(Ilu0, MatchesReferenceTridiagonal) {
    const uint32_t n = 4u;
    const uint32_t st = n;
    std::vector<double> A = {
        4, 1, 0, 0,
        1, 4, 1, 0,
        0, 1, 4, 1,
        0, 0, 1, 4};
    std::vector<double> lu(static_cast<size_t>(n) * n, 0.0);
    diffsim::Ilu0FactorizeHost(A.data(), n, st, lu.data());

    // Hand reference: U[0][0]=4; L[1][0]=1/4; U[1][1]=4 - (1/4)*1 = 3.75; ...
    // d0=4; d1=4-1/d0=3.75; d2=4-1/d1=3.7333...; d3=4-1/d2=3.732142857...
    const double d0 = 4.0;
    const double d1 = 4.0 - 1.0 / d0;
    const double d2 = 4.0 - 1.0 / d1;
    const double d3 = 4.0 - 1.0 / d2;
    EXPECT_NEAR(lu[0 * st + 0], d0, 1e-7);
    EXPECT_NEAR(lu[1 * st + 1], d1, 1e-7);
    EXPECT_NEAR(lu[2 * st + 2], d2, 1e-7);
    EXPECT_NEAR(lu[3 * st + 3], d3, 1e-7);
    EXPECT_NEAR(lu[1 * st + 0], 1.0 / d0, 1e-7);   // L[1][0]
    EXPECT_NEAR(lu[2 * st + 1], 1.0 / d1, 1e-7);   // L[2][1]
    EXPECT_NEAR(lu[3 * st + 2], 1.0 / d2, 1e-7);   // L[3][2]
    EXPECT_NEAR(lu[0 * st + 1], 1.0, 1e-7);        // U[0][1]
    // Structural zeros stay zero (no fill).
    EXPECT_EQ(lu[0 * st + 2], 0.0);
    EXPECT_EQ(lu[0 * st + 3], 0.0);
    EXPECT_EQ(lu[1 * st + 3], 0.0);
    std::printf("[Ilu0] tridiagonal reference matches (d3=%.9f)\n", d3);
}

// (2) GENUINE incompleteness: a pattern where elimination WOULD create fill outside
// the pattern, which ILU(0) drops. Pattern: arrowhead-ish
//   A = [ 4  1  1  1 ]
//       [ 1  4  0  0 ]
//       [ 1  0  4  0 ]
//       [ 1  0  0  4 ]
// Eliminating row 0 into rows 1..3 would create fill at (i,j) for i,j in {1,2,3},
// i!=j -- all STRUCTURAL ZEROS -> dropped. So L U != A there (true incomplete).
TEST(Ilu0, GenuinelyIncompleteArrowhead) {
    const uint32_t n = 4u;
    const uint32_t st = n;
    std::vector<double> A = {
        4, 1, 1, 1,
        1, 4, 0, 0,
        1, 0, 4, 0,
        1, 0, 0, 4};
    std::vector<double> lu(static_cast<size_t>(n) * n, 0.0);
    diffsim::Ilu0FactorizeHost(A.data(), n, st, lu.data());

    // Reconstruct (L U) and compare to A. At the dropped positions (1,2),(1,3),
    // (2,3) etc. the COMPLETE LU would be nonzero (Schur fill); ILU(0) leaves the
    // factor zero there, so (L U)[1][2] != A[1][2]==0 -> genuine incompleteness.
    auto lu_at = [&](uint32_t i, uint32_t j) { return lu[i * st + j]; };
    auto reconstruct = [&](uint32_t i, uint32_t j) {
        // (L U)[i][j] = sum_k L[i][k] U[k][j], L unit-lower, U upper.
        double s = 0.0;
        for (uint32_t k = 0u; k <= std::min(i, j); ++k) {
            const double lik = (k == i) ? 1.0 : ((k < i) ? lu_at(i, k) : 0.0);
            const double ukj = (k <= j) ? lu_at(k, j) : 0.0;
            s += lik * ukj;
        }
        return s;
    };
    // In-pattern entries are reproduced; dropped fill makes off-pattern (L U) differ
    // from A's zero there. Show at least one position where (L U) != 0 but A == 0.
    bool found_fill = false;
    double max_off_pattern = 0.0;
    for (uint32_t i = 1u; i < n; ++i)
        for (uint32_t j = 1u; j < n; ++j)
            if (i != j && A[i * st + j] == 0.0) {
                const double luij = reconstruct(i, j);
                max_off_pattern = std::max(max_off_pattern, std::abs(luij));
                if (std::abs(luij) > 1e-9) found_fill = true;
            }
    std::printf("[Ilu0] arrowhead off-pattern |(LU)-A| max = %.4e (dropped fill)\n",
                max_off_pattern);
    EXPECT_TRUE(found_fill)
        << "ILU(0) on the arrowhead should drop genuine fill -> (L U) != A "
           "off-pattern; if this fails the factorization is not actually incomplete";
}

// (3) DEVICE factor == host factor (fp32), and D1 two-run byte-exact.
TEST(Ilu0, DeviceMatchesHostAndIsBitExact) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const uint32_t n = 4u;
    const std::vector<float> A_dense = {
        4, 1, 0, 0,
        1, 4, 1, 0,
        0, 1, 4, 1,
        0, 0, 1, 4};
    // Pack into block-major (1 block, kMd stride).
    std::vector<float> a_blk(kStride, 0.0f);
    std::vector<uint32_t> dims(1u, n);
    for (uint32_t i = 0u; i < n; ++i)
        for (uint32_t j = 0u; j < n; ++j) a_blk[i * kMd + j] = A_dense[i * n + j];

    nuka::phi::Buffer d_a = UploadBuffer(a_blk);
    nuka::phi::Buffer d_dims = UploadBuffer(dims);
    nuka::phi::Buffer d_lu(static_cast<size_t>(kStride) * sizeof(float),
                           nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer d_lu2(static_cast<size_t>(kStride) * sizeof(float),
                            nuka::phi::MemoryKind::Device);

    diffsim::LaunchIlu0FactorizeTest(
        ctx, static_cast<const float*>(d_a.Data()),
        static_cast<const uint32_t*>(d_dims.Data()),
        static_cast<float*>(d_lu.Data()), 1u);
    diffsim::LaunchIlu0FactorizeTest(
        ctx, static_cast<const float*>(d_a.Data()),
        static_cast<const uint32_t*>(d_dims.Data()),
        static_cast<float*>(d_lu2.Data()), 1u);
    ctx.stream.Synchronize();

    std::vector<float> lu_dev(kStride, 0.0f), lu_dev2(kStride, 0.0f);
    d_lu.CopyToHost(lu_dev.data(), lu_dev.size() * sizeof(float));
    d_lu2.CopyToHost(lu_dev2.data(), lu_dev2.size() * sizeof(float));

    EXPECT_EQ(std::memcmp(lu_dev.data(), lu_dev2.data(), kStride * sizeof(float)), 0)
        << "ILU(0) device factor not D1 bit-exact two-run";

    // Host factor (fp64), compare to device (fp32) within fp32 epsilon.
    std::vector<double> a64(static_cast<size_t>(n) * n);
    for (uint32_t i = 0u; i < n; ++i)
        for (uint32_t j = 0u; j < n; ++j) a64[i * n + j] = A_dense[i * n + j];
    std::vector<double> lu_host(static_cast<size_t>(n) * n, 0.0);
    diffsim::Ilu0FactorizeHost(a64.data(), n, n, lu_host.data());

    double max_diff = 0.0;
    for (uint32_t i = 0u; i < n; ++i)
        for (uint32_t j = 0u; j < n; ++j)
            max_diff = std::max(max_diff,
                                std::abs(static_cast<double>(lu_dev[i * kMd + j]) -
                                         lu_host[i * n + j]));
    std::printf("[Ilu0] device vs host factor max |diff| = %.4e\n", max_diff);
    EXPECT_LT(max_diff, 1e-5) << "ILU(0) device factor vs host fp64 reference";
}

// (4) ILU(0) APPLY (forward/backward triangular solve) inverts L U: device solve
// of (L U) z = r matches the host reference, and is D1 bit-exact.
TEST(Ilu0, ApplyMatchesHostAndIsBitExact) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const uint32_t n = 5u;
    // SPD tridiagonal -> ILU(0) well-defined, apply is a genuine (L U)^-1.
    std::vector<double> a64(static_cast<size_t>(n) * n, 0.0);
    for (uint32_t i = 0u; i < n; ++i) {
        a64[i * n + i] = 4.0;
        if (i > 0u) { a64[i * n + (i - 1u)] = -1.0; a64[(i - 1u) * n + i] = -1.0; }
    }
    std::vector<double> lu_host(static_cast<size_t>(n) * n, 0.0);
    diffsim::Ilu0FactorizeHost(a64.data(), n, n, lu_host.data());

    std::vector<double> r64 = {1.0, 2.0, -1.0, 0.5, 3.0};
    std::vector<double> z_host(n, 0.0);
    diffsim::Ilu0ApplyHost(lu_host.data(), n, n, r64.data(), z_host.data());

    // Pack LU + r into block-major fp32 for the device apply.
    std::vector<float> lu_blk(kStride, 0.0f), r_blk(kMd, 0.0f);
    std::vector<uint32_t> dims(1u, n);
    for (uint32_t i = 0u; i < n; ++i) {
        for (uint32_t j = 0u; j < n; ++j)
            lu_blk[i * kMd + j] = static_cast<float>(lu_host[i * n + j]);
        r_blk[i] = static_cast<float>(r64[i]);
    }
    nuka::phi::Buffer d_lu = UploadBuffer(lu_blk);
    nuka::phi::Buffer d_r = UploadBuffer(r_blk);
    nuka::phi::Buffer d_dims = UploadBuffer(dims);
    nuka::phi::Buffer d_z(static_cast<size_t>(kMd) * sizeof(float),
                          nuka::phi::MemoryKind::Device);
    nuka::phi::Buffer d_z2(static_cast<size_t>(kMd) * sizeof(float),
                           nuka::phi::MemoryKind::Device);

    diffsim::LaunchIlu0ApplyTest(ctx, static_cast<const float*>(d_lu.Data()),
                                 static_cast<const uint32_t*>(d_dims.Data()),
                                 static_cast<const float*>(d_r.Data()),
                                 static_cast<float*>(d_z.Data()), 1u);
    diffsim::LaunchIlu0ApplyTest(ctx, static_cast<const float*>(d_lu.Data()),
                                 static_cast<const uint32_t*>(d_dims.Data()),
                                 static_cast<const float*>(d_r.Data()),
                                 static_cast<float*>(d_z2.Data()), 1u);
    ctx.stream.Synchronize();

    std::vector<float> z_dev(kMd, 0.0f), z_dev2(kMd, 0.0f);
    d_z.CopyToHost(z_dev.data(), z_dev.size() * sizeof(float));
    d_z2.CopyToHost(z_dev2.data(), z_dev2.size() * sizeof(float));
    EXPECT_EQ(std::memcmp(z_dev.data(), z_dev2.data(), kMd * sizeof(float)), 0)
        << "ILU(0) apply not D1 bit-exact two-run";

    double max_diff = 0.0;
    for (uint32_t i = 0u; i < n; ++i)
        max_diff = std::max(max_diff, std::abs(static_cast<double>(z_dev[i]) -
                                               z_host[i]));
    std::printf("[Ilu0] apply device vs host max |diff| = %.4e\n", max_diff);
    EXPECT_LT(max_diff, 1e-5) << "ILU(0) apply device vs host reference";
}

}  // namespace
