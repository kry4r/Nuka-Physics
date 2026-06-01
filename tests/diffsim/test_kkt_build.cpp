// ---------------------------------------------------------------------------
// v0.5 p03 R2 -- contact KKT (Delassus) builder: host Gram cross-check + D1
// ---------------------------------------------------------------------------
//
// Validates nuka::diffsim::BuildContactDelassusSystem -- the warp-per-
// articulation assembly of the batched dense SPD Delassus operator
//      A_b = J_b M_b^-1 J_b^T
// from the engine's real contact buffer layout (ArticulatedContactRow rows +
// jac_normal/tangent1/tangent2 chain rows + per-articulation M^-1 tile) into the
// locked BatchedDenseSpdSystem the CG backend consumes.
//
// Primary cross-check is SYNTHETIC and decoupled from contact detection: we hand-
// populate the device buffers with known rows + a known SPD M^-1 tile + genuinely
// COUPLED (non-orthogonal) J rows, run the builder, and INDEPENDENTLY recompute
// A = J M^-1 J^T on the host with Eigen (fp64), replicating the builder's exact
// active-set ordering (slot ascending, then normal -> t1 -> t2). We assert:
//   * per-component builder-vs-host rel-err < 1e-5,
//   * EXACT bit-symmetry  A[i][j] == A[j][i],
//   * the diagonal identity  A[i][i] == J_i M^-1 J_i^T  (the engine stores
//     effective_mass = 1 / that; we cross-check against 1/effective_mass too),
//   * D1: two builds -> byte-identical `values`.
//
// The batch deliberately exercises: an articulation with a SKIPPED slot (depth<=0)
// between two active slots (so the row->J gather is NOT a trivial identity), an
// EMPTY articulation (n_b == 0), and a full 12-row articulation.
// ---------------------------------------------------------------------------

#include "diffsim/kkt_builder.hpp"
#include "diffsim/sparse_solver_backend.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"
#include "runtime/articulation/articulation_contacts.hpp"

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
namespace art = nuka::runtime::articulation;

constexpr uint32_t kMd = diffsim::kMaxBlockDim;          // 12 (output block stride dim)
constexpr uint32_t kStride = kMd * kMd;                  // 144
constexpr uint32_t kSlots = art::kMaxFootContactsPerEnv; // 4
constexpr uint32_t kDof = 18u;                           // dof_stride (== max_dof cap)

// One synthetic articulation's worth of host-side ground truth: which of its 4
// slots are engaged (active normal row) plus a per-slot, per-direction J row and
// the articulation's SPD M^-1 tile. Active rows are emitted in builder order
// (slot asc, normal -> t1 -> t2) for the host Gram reference.
struct Articulation {
    bool slot_active[kSlots] = {false, false, false, false};
    // J rows: [slot][dir(0=n,1=t1,2=t2)] each length kDof.
    Eigen::MatrixXd jac[kSlots][3];   // each 1 x kDof, only for active slots
    Eigen::MatrixXd minv;             // kDof x kDof SPD
};

// Build a coupled SPD M^-1 tile = L L^T + kDof*I (genuinely off-diagonal).
Eigen::MatrixXd MakeSpdTile(std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Eigen::MatrixXd L = Eigen::MatrixXd::Zero(kDof, kDof);
    for (uint32_t i = 0u; i < kDof; ++i)
        for (uint32_t j = 0u; j <= i; ++j) L(i, j) = u(rng);
    Eigen::MatrixXd M = L * L.transpose() + static_cast<double>(kDof) *
                                                Eigen::MatrixXd::Identity(kDof, kDof);
    return 0.5 * (M + M.transpose()).eval();  // exactly symmetric
}

// A dense, non-orthogonal J row (so the resulting Gram has nonzero off-diagonals).
Eigen::MatrixXd MakeJRow(std::mt19937& rng) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    Eigen::MatrixXd j(1, kDof);
    for (uint32_t c = 0u; c < kDof; ++c) j(0, c) = u(rng);
    return j;
}

// The flat host-side active-row list of an articulation, in builder order.
struct ActiveRows {
    std::vector<Eigen::MatrixXd> rows;  // each 1 x kDof
};

ActiveRows GatherActive(const Articulation& a) {
    ActiveRows out;
    for (uint32_t s = 0u; s < kSlots; ++s) {
        if (!a.slot_active[s]) continue;
        if (out.rows.size() + 3u > kMd) break;  // builder caps at 12 rows
        out.rows.push_back(a.jac[s][0]);  // normal
        out.rows.push_back(a.jac[s][1]);  // tangent1
        out.rows.push_back(a.jac[s][2]);  // tangent2
    }
    return out;
}

// Host reference Gram A = J M^-1 J^T for one articulation (fp64, exact ordering).
Eigen::MatrixXd HostGram(const Articulation& a) {
    const ActiveRows ar = GatherActive(a);
    const uint32_t n = static_cast<uint32_t>(ar.rows.size());
    Eigen::MatrixXd A(n, n);
    for (uint32_t i = 0u; i < n; ++i)
        for (uint32_t j = 0u; j < n; ++j)
            A(i, j) = (ar.rows[i] * a.minv * ar.rows[j].transpose())(0, 0);
    return A;
}

// Builds the device-side input buffers (env-major, fixed strides) from a batch of
// articulations, then returns the builder's `values` (host copy) + `dims`.
struct BuildResult {
    std::vector<float> values;       // block_count * 144
    std::vector<uint32_t> dims;      // block_count
};

BuildResult RunBuild(const nuka::phi::DeviceContext& ctx,
                     const std::vector<Articulation>& batch) {
    const uint32_t bc = static_cast<uint32_t>(batch.size());
    const uint32_t slot_count = bc * kSlots;

    std::vector<art::ArticulatedContactRow> rows(slot_count);
    std::vector<float> jn(static_cast<size_t>(slot_count) * kDof, 0.0f);
    std::vector<float> jt1(static_cast<size_t>(slot_count) * kDof, 0.0f);
    std::vector<float> jt2(static_cast<size_t>(slot_count) * kDof, 0.0f);
    std::vector<float> minv(static_cast<size_t>(bc) * kDof * kDof, 0.0f);

    for (uint32_t b = 0u; b < bc; ++b) {
        const Articulation& a = batch[b];
        // M^-1 tile (row-major, stride kDof^2).
        for (uint32_t r = 0u; r < kDof; ++r)
            for (uint32_t c = 0u; c < kDof; ++c)
                minv[static_cast<size_t>(b) * kDof * kDof + r * kDof + c] =
                    static_cast<float>(a.minv(r, c));
        for (uint32_t s = 0u; s < kSlots; ++s) {
            const uint32_t slot = b * kSlots + s;
            art::ArticulatedContactRow& row = rows[slot];
            row.articulation = b;
            row.contact_link = b;  // unused by the builder
            if (a.slot_active[s]) {
                row.row_type = art::ContactRowType::kRowNormal;
                row.depth = 0.01f;  // depth > 0 -> engaged
                for (uint32_t c = 0u; c < kDof; ++c) {
                    jn[static_cast<size_t>(slot) * kDof + c] =
                        static_cast<float>(a.jac[s][0](0, c));
                    jt1[static_cast<size_t>(slot) * kDof + c] =
                        static_cast<float>(a.jac[s][1](0, c));
                    jt2[static_cast<size_t>(slot) * kDof + c] =
                        static_cast<float>(a.jac[s][2](0, c));
                }
            } else {
                row.row_type = art::ContactRowType::kRowInactive;
                row.depth = 0.0f;  // depth == 0 -> NOT engaged (skipped)
            }
        }
    }

    nuka::phi::Buffer d_rows(rows.size() * sizeof(art::ArticulatedContactRow));
    d_rows.CopyFromHost(rows.data(), rows.size() * sizeof(art::ArticulatedContactRow));
    nuka::phi::Buffer d_jn(jn.size() * sizeof(float));
    d_jn.CopyFromHost(jn.data(), jn.size() * sizeof(float));
    nuka::phi::Buffer d_jt1(jt1.size() * sizeof(float));
    d_jt1.CopyFromHost(jt1.data(), jt1.size() * sizeof(float));
    nuka::phi::Buffer d_jt2(jt2.size() * sizeof(float));
    d_jt2.CopyFromHost(jt2.data(), jt2.size() * sizeof(float));
    nuka::phi::Buffer d_minv(minv.size() * sizeof(float));
    d_minv.CopyFromHost(minv.data(), minv.size() * sizeof(float));

    diffsim::BatchedDenseSpdSystem system;
    diffsim::ContactDelassusBuffers buffers;
    diffsim::BuildContactDelassusSystem(
        ctx, static_cast<const art::ArticulatedContactRow*>(d_rows.Data()),
        static_cast<const float*>(d_jn.Data()),
        static_cast<const float*>(d_jt1.Data()),
        static_cast<const float*>(d_jt2.Data()),
        static_cast<const float*>(d_minv.Data()), bc, kDof, system, buffers);
    ctx.stream.Synchronize();

    BuildResult res;
    res.values.assign(static_cast<size_t>(bc) * kStride, 0.0f);
    res.dims.assign(bc, 0u);
    buffers.values.CopyToHost(res.values.data(), res.values.size() * sizeof(float));
    buffers.block_dim.CopyToHost(res.dims.data(), res.dims.size() * sizeof(uint32_t));
    return res;
}

// A representative batch (fixed seed). Articulation 0: slots {0,2} active (slot 1
// skipped between them -> non-identity gather, n=6). Art 1: all 4 active (n=12,
// full block). Art 2: empty (no active slot, n=0). Art 3: only slot 3 active (n=3).
std::vector<Articulation> MakeBatch() {
    std::mt19937 rng(0xDE1A55u);
    std::vector<Articulation> batch(4);

    auto fill_slot = [&](Articulation& a, uint32_t s) {
        a.slot_active[s] = true;
        a.jac[s][0] = MakeJRow(rng);
        a.jac[s][1] = MakeJRow(rng);
        a.jac[s][2] = MakeJRow(rng);
    };

    batch[0].minv = MakeSpdTile(rng);
    fill_slot(batch[0], 0u);
    fill_slot(batch[0], 2u);  // slot 1 skipped

    batch[1].minv = MakeSpdTile(rng);
    for (uint32_t s = 0u; s < kSlots; ++s) fill_slot(batch[1], s);  // full 12 rows

    batch[2].minv = MakeSpdTile(rng);  // no active slot -> n == 0

    batch[3].minv = MakeSpdTile(rng);
    fill_slot(batch[3], 3u);  // only last slot

    return batch;
}

TEST(KktBuild, MatchesHostGram) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const std::vector<Articulation> batch = MakeBatch();
    const BuildResult r = RunBuild(ctx, batch);

    double max_rel = 0.0;
    double max_sym = 0.0;
    double max_diag = 0.0;
    for (uint32_t b = 0u; b < batch.size(); ++b) {
        const Eigen::MatrixXd A = HostGram(batch[b]);
        const uint32_t n = static_cast<uint32_t>(A.rows());
        ASSERT_EQ(r.dims[b], n) << "block_dim mismatch at articulation " << b;

        // Reference scale for componentwise rel-err (per block, with a floor).
        double scale = 1.0e-12;
        for (uint32_t i = 0u; i < n; ++i)
            for (uint32_t j = 0u; j < n; ++j)
                scale = std::max(scale, std::abs(A(i, j)));

        for (uint32_t i = 0u; i < n; ++i) {
            for (uint32_t j = 0u; j < n; ++j) {
                const float a_dev = r.values[b * kStride + i * kMd + j];
                max_rel = std::max(max_rel, std::abs(static_cast<double>(a_dev) - A(i, j)) / scale);
            }
        }

        // EXACT bit-symmetry: A[i][j] == A[j][i] bitwise (mirror-stored, not recomputed).
        for (uint32_t i = 0u; i < n; ++i) {
            for (uint32_t j = i + 1u; j < n; ++j) {
                const float aij = r.values[b * kStride + i * kMd + j];
                const float aji = r.values[b * kStride + j * kMd + i];
                ASSERT_EQ(std::memcmp(&aij, &aji, sizeof(float)), 0)
                    << "bit-symmetry violated at block " << b << " (" << i << "," << j << ")";
                max_sym = std::max(max_sym, std::abs(static_cast<double>(aij) - aji));
            }
        }

        // Diagonal identity: A[i][i] == J_i M^-1 J_i^T (== 1/effective_mass).
        const ActiveRows ar = GatherActive(batch[b]);
        for (uint32_t i = 0u; i < n; ++i) {
            const double diag_ref = (ar.rows[i] * batch[b].minv * ar.rows[i].transpose())(0, 0);
            const float diag_dev = r.values[b * kStride + i * kMd + i];
            double dscale = std::max(std::abs(diag_ref), 1.0e-12);
            max_diag = std::max(max_diag, std::abs(static_cast<double>(diag_dev) - diag_ref) / dscale);
            // The engine stores effective_mass = 1/(J_i M^-1 J_i^T); cross-check that
            // the diagonal equals the reciprocal of that effective mass.
            const double eff_mass = 1.0 / diag_ref;
            EXPECT_NEAR(static_cast<double>(diag_dev), 1.0 / eff_mass,
                        1.0e-4 * dscale)
                << "diagonal != 1/effective_mass at block " << b << " row " << i;
        }

        // Padding (lanes >= n, and the j>n columns) must be exactly 0.
        for (uint32_t i = 0u; i < kMd; ++i)
            for (uint32_t j = 0u; j < kMd; ++j)
                if (i >= n || j >= n)
                    EXPECT_EQ(r.values[b * kStride + i * kMd + j], 0.0f)
                        << "nonzero padding at block " << b << " (" << i << "," << j << ")";
    }

    std::printf("[KktBuild] max builder-vs-host rel-err = %.3e\n", max_rel);
    std::printf("[KktBuild] max |A[i][j]-A[j][i]|       = %.3e (must be 0)\n", max_sym);
    std::printf("[KktBuild] max diagonal rel-err        = %.3e\n", max_diag);
    EXPECT_LT(max_rel, 1.0e-5) << "builder A = J M^-1 J^T vs host Eigen Gram";
    EXPECT_EQ(max_sym, 0.0) << "Gram block must be exactly bit-symmetric";
    EXPECT_LT(max_diag, 1.0e-5) << "diagonal == J_i M^-1 J_i^T (= 1/effective_mass)";
}

TEST(KktBuild, BitExactSameInputs) {
    auto ctx = nuka::phi::MakeDefaultDeviceContext();
    const std::vector<Articulation> batch = MakeBatch();
    const BuildResult r1 = RunBuild(ctx, batch);
    const BuildResult r2 = RunBuild(ctx, batch);
    ASSERT_EQ(r1.values.size(), r2.values.size());
    ASSERT_EQ(r1.dims.size(), r2.dims.size());
    // Raw-bit comparison over the FULL values span (incl. all padding + empty
    // blocks) -- the R6 D1 gate.
    const int cmp_v =
        std::memcmp(r1.values.data(), r2.values.data(), r1.values.size() * sizeof(float));
    const int cmp_d =
        std::memcmp(r1.dims.data(), r2.dims.data(), r1.dims.size() * sizeof(uint32_t));
    EXPECT_EQ(cmp_v, 0) << "D1 bit-exact violated: values differ across two builds";
    EXPECT_EQ(cmp_d, 0) << "D1 bit-exact violated: block_dim differs across two builds";
    std::printf("[KktBuild] two-run byte-identical (values + block_dim)\n");
}

}  // namespace
