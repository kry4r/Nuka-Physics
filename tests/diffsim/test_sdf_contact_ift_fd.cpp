// ---------------------------------------------------------------------------
// p08-C Channel 1 (the §4.B acceptance): d/dM, d/dJ of the SDF-contact solve.
// ---------------------------------------------------------------------------
// THE SYSTEM-LEVEL FD (the trap the task warns about). We do NOT validate the
// per-row codegen derivative rule (id 0 already has that; the SDF rows inherit it
// for free, and a green per-row FD does NOT satisfy §4.B). Instead, for each
// perturbed entry of M^-1 / J we:
//     (1) PERTURB the actual M^-1 / J entry by +/- eps,
//     (2) RE-ASSEMBLE A = J M^-1 J^T and r = b_c - J qdot_free,
//     (3) RE-SOLVE the constraint system A lambda = r TO CONVERGENCE
//         (dense Cholesky -- the exact island solve, == the converged PGS fixed
//          point), for BOTH the + and - perturbation,
//     (4) central-difference   (lambda_+ - lambda_-) / (2 eps)
// and compare to the analytical IFT directional derivative  dlambda = A^-1
// (dr - dA lambda)  (DLambda_dMinv / DLambda_dJ). The re-solve in step (3) is the
// load-bearing part: it is the GLOBAL constraint solve, not a row rule.
//
// d/dM: dr = 0, dA = J dMinv J^T.
// d/dJ: dA = dJ M^-1 J^T + J M^-1 dJ^T, AND dr = -dJ qdot_free (nonzero).
//
// The system is the Delassus of a real, coupled, SPD active set with a converged
// lambda; valley-independent (no step_curvature anywhere).
// ---------------------------------------------------------------------------

#include "diffsim/sdf_contact_ift.hpp"

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using nuka::diffsim::AssembleDelassus;
using nuka::diffsim::AssembleRhs;
using nuka::diffsim::DLambda_dJ;
using nuka::diffsim::DLambda_dMinv;
using nuka::diffsim::SdfContactDenseSystem;
using nuka::diffsim::SolveLambda;

double RelErr(double a, double b) {
    return std::fabs(a - b) / (std::fabs(a) + std::fabs(b) + 1.0e-9);
}

// INDEPENDENT ORACLE (the anti-vacuity fix). The §4.B FD must re-run the GLOBAL
// constraint solve to convergence for each perturbation -- AND it must do so with
// an oracle that does NOT share the module's AssembleDelassus / CholeskySolve, or
// a transpose/index bug in either would pass on both sides. So we assemble
// A = J M^-1 J^T and re-solve A lambda = r with EIGEN's LDLT (the exact island
// solve == the converged PGS/CG fixed point; the same dense-reference pattern as
// test_ift_vs_tape_backward). Analytical and FD now share ONLY the inputs + the
// linearization-point lambda (also Eigen-sourced).
Eigen::MatrixXd EigenJ(const SdfContactDenseSystem& s) {
    Eigen::MatrixXd J(s.n, s.dof);
    for (uint32_t r = 0u; r < s.n; ++r)
        for (uint32_t c = 0u; c < s.dof; ++c) J(r, c) = s.j[static_cast<size_t>(r) * s.dof + c];
    return J;
}
Eigen::MatrixXd EigenMinv(const SdfContactDenseSystem& s) {
    Eigen::MatrixXd M(s.dof, s.dof);
    for (uint32_t i = 0u; i < s.dof; ++i)
        for (uint32_t k = 0u; k < s.dof; ++k) M(i, k) = s.m_inv[static_cast<size_t>(i) * s.dof + k];
    return M;
}
Eigen::VectorXd EigenRhs(const SdfContactDenseSystem& s) {
    Eigen::MatrixXd J = EigenJ(s);
    Eigen::VectorXd qf(s.dof), b(s.n);
    for (uint32_t c = 0u; c < s.dof; ++c) qf(c) = s.qdot_free[c];
    for (uint32_t r = 0u; r < s.n; ++r) b(r) = s.b_c[r];
    return b - J * qf;
}
// Re-solve to convergence with Eigen LDLT -- the independent oracle.
std::vector<double> ReSolveToConvergence(const SdfContactDenseSystem& s) {
    const Eigen::MatrixXd J = EigenJ(s);
    const Eigen::MatrixXd M = EigenMinv(s);
    const Eigen::MatrixXd A = J * M * J.transpose();  // independent A = J M^-1 J^T
    const Eigen::VectorXd r = EigenRhs(s);
    const Eigen::VectorXd lam = A.ldlt().solve(r);     // re-solve to convergence
    std::vector<double> out(s.n);
    for (uint32_t i = 0u; i < s.n; ++i) out[i] = lam(i);
    return out;
}

// Build a controlled, well-conditioned coupled active-set system: `n` active rows
// (e.g. 3 = one engaged slot: normal + 2 friction) over `dof` articulation DOFs.
// M^-1 = symmetric SPD tile; J = full-row-rank random; b_c, qdot_free random. The
// linearization-point lambda is set from the INDEPENDENT Eigen oracle (not the
// module's SolveLambda) so a wrong lambda0 is also caught by the FD.
SdfContactDenseSystem MakeSystem(uint32_t n, uint32_t dof, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    SdfContactDenseSystem s;
    s.n = n;
    s.dof = dof;
    s.j.resize(static_cast<size_t>(n) * dof);
    for (auto& v : s.j) v = u(rng);
    // M^-1 = B B^T + dof*I  (SPD, well-conditioned).
    std::vector<double> b(static_cast<size_t>(dof) * dof);
    for (auto& v : b) v = u(rng);
    s.m_inv.assign(static_cast<size_t>(dof) * dof, 0.0);
    for (uint32_t i = 0u; i < dof; ++i)
        for (uint32_t k = 0u; k < dof; ++k) {
            double acc = 0.0;
            for (uint32_t t = 0u; t < dof; ++t)
                acc += b[static_cast<size_t>(i) * dof + t] * b[static_cast<size_t>(k) * dof + t];
            s.m_inv[static_cast<size_t>(i) * dof + k] = acc + (i == k ? double(dof) : 0.0);
        }
    s.b_c.resize(n);
    for (auto& v : s.b_c) v = 0.5 * u(rng);
    s.qdot_free.resize(dof);
    for (auto& v : s.qdot_free) v = 0.3 * u(rng);
    s.lambda = ReSolveToConvergence(s);  // Eigen-sourced linearization point
    return s;
}

}  // namespace

// ---------------------------------------------------------------------------
// Correspondence: the module's AssembleDelassus must equal the independent Eigen
// A = J M^-1 J^T (ties the host reference's assembly to a non-shared oracle; a
// transpose/index bug surfaces here too, not only via the FD).
// ---------------------------------------------------------------------------
TEST(SdfContactIft, AssembleDelassusMatchesEigen) {
    const SdfContactDenseSystem s = MakeSystem(/*n=*/6u, /*dof=*/8u, /*seed=*/0xA11CEu);
    const std::vector<double> a = AssembleDelassus(s);
    const Eigen::MatrixXd Ae = EigenJ(s) * EigenMinv(s) * EigenJ(s).transpose();
    double maxd = 0.0, sca = 1.0e-12;
    for (uint32_t i = 0u; i < s.n; ++i)
        for (uint32_t k = 0u; k < s.n; ++k) {
            sca = std::max(sca, std::fabs(Ae(i, k)));
            maxd = std::max(maxd, std::fabs(a[static_cast<size_t>(i) * s.n + k] - Ae(i, k)));
        }
    std::printf("[Ch1 assemble] module-A vs Eigen-A max abs diff = %.3e (scale %.3e)\n", maxd, sca);
    EXPECT_LT(maxd, 1.0e-9 * sca);
    // And the module's own solve matches Eigen LDLT on the same A,r.
    const std::vector<double> lam_mod = SolveLambda(s);
    const std::vector<double> lam_eig = ReSolveToConvergence(s);
    double maxl = 0.0;
    for (uint32_t i = 0u; i < s.n; ++i) maxl = std::max(maxl, std::fabs(lam_mod[i] - lam_eig[i]));
    EXPECT_LT(maxl, 1.0e-9) << "module CholeskySolve vs Eigen LDLT";
}

// ---------------------------------------------------------------------------
// d/dM channel: perturb M^-1 entries, RE-SOLVE, compare to A^-1(-J dMinv J^T lambda).
// ---------------------------------------------------------------------------
TEST(SdfContactIft, DMatchesPerturbAndResolve) {
    const SdfContactDenseSystem s = MakeSystem(/*n=*/3u, /*dof=*/4u, /*seed=*/0xD1Fu);
    const double eps = 1.0e-6;
    double max_rel = 0.0;
    uint32_t worst_i = 0u, worst_k = 0u, worst_row = 0u;

    // Perturb each M^-1 entry. M^-1 is symmetric: perturb the symmetric pair
    // (i,k)+(k,i) together (the analytical dMinv direction is that symmetric
    // pattern -- it is the tile the contraction actually consumes).
    for (uint32_t i = 0u; i < s.dof; ++i) {
        for (uint32_t k = i; k < s.dof; ++k) {
            // Analytical: directional derivative for dMinv = symmetric unit bump.
            std::vector<double> dminv(static_cast<size_t>(s.dof) * s.dof, 0.0);
            dminv[static_cast<size_t>(i) * s.dof + k] = 1.0;
            if (i != k) dminv[static_cast<size_t>(k) * s.dof + i] = 1.0;
            const std::vector<double> dlam = DLambda_dMinv(s, dminv);

            // FD: PERTURB the M^-1 entries (symmetric), RE-SOLVE to convergence.
            SdfContactDenseSystem sp = s, sm = s;
            sp.m_inv[static_cast<size_t>(i) * s.dof + k] += eps;
            sm.m_inv[static_cast<size_t>(i) * s.dof + k] -= eps;
            if (i != k) {
                sp.m_inv[static_cast<size_t>(k) * s.dof + i] += eps;
                sm.m_inv[static_cast<size_t>(k) * s.dof + i] -= eps;
            }
            const std::vector<double> lp = ReSolveToConvergence(sp);
            const std::vector<double> lm = ReSolveToConvergence(sm);

            for (uint32_t r = 0u; r < s.n; ++r) {
                const double fd = (lp[r] - lm[r]) / (2.0 * eps);
                const double rel = RelErr(dlam[r], fd);
                if (rel > max_rel) { max_rel = rel; worst_i = i; worst_k = k; worst_row = r; }
            }
        }
    }
    std::printf("[Ch1 d/dM] perturb-M^-1 + RE-SOLVE: max rel err = %.3e "
                "(M^-1[%u,%u], lambda row %u)\n", max_rel, worst_i, worst_k, worst_row);
    EXPECT_LT(max_rel, 1.0e-3) << "d/dM IFT vs perturb-M^-1-and-re-solve";
}

// ---------------------------------------------------------------------------
// d/dJ channel: perturb J entries, RE-SOLVE, compare to the FULL IFT
// (product-rule dA BOTH terms + the dr = -dJ qdot_free term).
// ---------------------------------------------------------------------------
TEST(SdfContactIft, DJMatchesPerturbAndResolve) {
    const SdfContactDenseSystem s = MakeSystem(/*n=*/3u, /*dof=*/4u, /*seed=*/0x5EEDu);
    const double eps = 1.0e-6;
    double max_rel = 0.0;
    uint32_t worst_r = 0u, worst_c = 0u, worst_row = 0u;

    for (uint32_t pr = 0u; pr < s.n; ++pr) {
        for (uint32_t pc = 0u; pc < s.dof; ++pc) {
            // Analytical directional derivative for dJ = unit bump at (pr,pc).
            std::vector<double> dj(static_cast<size_t>(s.n) * s.dof, 0.0);
            dj[static_cast<size_t>(pr) * s.dof + pc] = 1.0;
            const std::vector<double> dlam = DLambda_dJ(s, dj);

            // FD: PERTURB the J entry, RE-SOLVE to convergence (A AND r both change).
            SdfContactDenseSystem sp = s, sm = s;
            sp.j[static_cast<size_t>(pr) * s.dof + pc] += eps;
            sm.j[static_cast<size_t>(pr) * s.dof + pc] -= eps;
            const std::vector<double> lp = ReSolveToConvergence(sp);
            const std::vector<double> lm = ReSolveToConvergence(sm);

            for (uint32_t r = 0u; r < s.n; ++r) {
                const double fd = (lp[r] - lm[r]) / (2.0 * eps);
                const double rel = RelErr(dlam[r], fd);
                if (rel > max_rel) { max_rel = rel; worst_r = pr; worst_c = pc; worst_row = r; }
            }
        }
    }
    std::printf("[Ch1 d/dJ] perturb-J + RE-SOLVE: max rel err = %.3e "
                "(J[%u,%u], lambda row %u)\n", max_rel, worst_r, worst_c, worst_row);
    EXPECT_LT(max_rel, 1.0e-3) << "d/dJ IFT vs perturb-J-and-re-solve";
}

// ---------------------------------------------------------------------------
// Guard: dropping the dr = -dJ qdot_free term must BREAK the d/dJ match (proves
// the FD is system-level / re-solves r, not a per-row dA-only check). We rebuild
// the analytical dlambda WITHOUT the dr term and confirm it diverges from FD.
// ---------------------------------------------------------------------------
TEST(SdfContactIft, DJDropDrTermFailsAsExpected) {
    SdfContactDenseSystem s = MakeSystem(/*n=*/3u, /*dof=*/4u, /*seed=*/0xBADu);
    // Make qdot_free large so the dr term dominates and the omission is unmissable.
    for (auto& v : s.qdot_free) v *= 50.0;
    s.lambda = ReSolveToConvergence(s);  // re-linearize at the new qdot_free (Eigen oracle)

    const double eps = 1.0e-6;
    const uint32_t pr = 0u, pc = 0u;
    std::vector<double> dj(static_cast<size_t>(s.n) * s.dof, 0.0);
    dj[static_cast<size_t>(pr) * s.dof + pc] = 1.0;

    // Correct analytical (with dr).
    const std::vector<double> dlam_full = DLambda_dJ(s, dj);

    // FD re-solve.
    SdfContactDenseSystem sp = s, sm = s;
    sp.j[static_cast<size_t>(pr) * s.dof + pc] += eps;
    sm.j[static_cast<size_t>(pr) * s.dof + pc] -= eps;
    const std::vector<double> lp = ReSolveToConvergence(sp);
    const std::vector<double> lm = ReSolveToConvergence(sm);

    // "dA-only" (the WRONG per-row-flavored answer): A^-1(-dA lambda), dr dropped.
    // Reconstruct it by zeroing qdot_free (kills dr) in a clone.
    SdfContactDenseSystem s_nodr = s;
    for (auto& v : s_nodr.qdot_free) v = 0.0;
    s_nodr.lambda = s.lambda;  // same linearization point lambda
    const std::vector<double> dlam_nodr = DLambda_dJ(s_nodr, dj);

    double max_full = 0.0, max_nodr = 0.0;
    for (uint32_t r = 0u; r < s.n; ++r) {
        const double fd = (lp[r] - lm[r]) / (2.0 * eps);
        max_full = std::max(max_full, RelErr(dlam_full[r], fd));
        max_nodr = std::max(max_nodr, RelErr(dlam_nodr[r], fd));
    }
    std::printf("[Ch1 d/dJ guard] full(with dr) rel=%.3e   dr-dropped rel=%.3e\n",
                max_full, max_nodr);
    EXPECT_LT(max_full, 1.0e-3) << "full IFT (with dr) must match FD re-solve";
    EXPECT_GT(max_nodr, 1.0e-2) << "dropping dr=-dJ*qdot_free must visibly break the match";
}

// ---------------------------------------------------------------------------
// Coupled multi-slot case (n=6 = two engaged slots) over a larger dof, to gate
// the genuinely coupled Delassus (not just a single 3x3 block).
// ---------------------------------------------------------------------------
TEST(SdfContactIft, CoupledMultiSlotDMandDJ) {
    const SdfContactDenseSystem s = MakeSystem(/*n=*/6u, /*dof=*/8u, /*seed=*/0xC0FFEEu);
    const double eps = 1.0e-6;

    double max_m = 0.0;
    for (uint32_t i = 0u; i < s.dof; ++i)
        for (uint32_t k = i; k < s.dof; ++k) {
            std::vector<double> dminv(static_cast<size_t>(s.dof) * s.dof, 0.0);
            dminv[static_cast<size_t>(i) * s.dof + k] = 1.0;
            if (i != k) dminv[static_cast<size_t>(k) * s.dof + i] = 1.0;
            const std::vector<double> dlam = DLambda_dMinv(s, dminv);
            SdfContactDenseSystem sp = s, sm = s;
            sp.m_inv[static_cast<size_t>(i) * s.dof + k] += eps;
            sm.m_inv[static_cast<size_t>(i) * s.dof + k] -= eps;
            if (i != k) {
                sp.m_inv[static_cast<size_t>(k) * s.dof + i] += eps;
                sm.m_inv[static_cast<size_t>(k) * s.dof + i] -= eps;
            }
            const std::vector<double> lp = ReSolveToConvergence(sp), lm = ReSolveToConvergence(sm);
            for (uint32_t r = 0u; r < s.n; ++r)
                max_m = std::max(max_m, RelErr(dlam[r], (lp[r] - lm[r]) / (2.0 * eps)));
        }

    double max_j = 0.0;
    for (uint32_t pr = 0u; pr < s.n; ++pr)
        for (uint32_t pc = 0u; pc < s.dof; ++pc) {
            std::vector<double> dj(static_cast<size_t>(s.n) * s.dof, 0.0);
            dj[static_cast<size_t>(pr) * s.dof + pc] = 1.0;
            const std::vector<double> dlam = DLambda_dJ(s, dj);
            SdfContactDenseSystem sp = s, sm = s;
            sp.j[static_cast<size_t>(pr) * s.dof + pc] += eps;
            sm.j[static_cast<size_t>(pr) * s.dof + pc] -= eps;
            const std::vector<double> lp = ReSolveToConvergence(sp), lm = ReSolveToConvergence(sm);
            for (uint32_t r = 0u; r < s.n; ++r)
                max_j = std::max(max_j, RelErr(dlam[r], (lp[r] - lm[r]) / (2.0 * eps)));
        }

    std::printf("[Ch1 coupled n=6 dof=8] d/dM max rel=%.3e   d/dJ max rel=%.3e\n", max_m, max_j);
    EXPECT_LT(max_m, 1.0e-3);
    EXPECT_LT(max_j, 1.0e-3);
}

// ---------------------------------------------------------------------------
// D1 (checklist item 7): two runs of the d/dM and d/dJ directional derivatives
// are byte-identical. (Host fp64 reference -> deterministic by construction; this
// makes the byte-exactness on the SDF-contact gradients explicit.)
// ---------------------------------------------------------------------------
TEST(SdfContactIft, D1TwoRunByteExact) {
    const SdfContactDenseSystem s = MakeSystem(/*n=*/6u, /*dof=*/8u, /*seed=*/0xD17u);
    std::vector<double> dminv(static_cast<size_t>(s.dof) * s.dof, 0.0);
    dminv[0] = 1.0; dminv[s.dof + 1] = 1.0;  // a couple of symmetric bumps
    std::vector<double> dj(static_cast<size_t>(s.n) * s.dof, 0.0);
    dj[3] = 1.0; dj[s.dof + 2] = 1.0;

    const std::vector<double> m1 = DLambda_dMinv(s, dminv);
    const std::vector<double> m2 = DLambda_dMinv(s, dminv);
    const std::vector<double> j1 = DLambda_dJ(s, dj);
    const std::vector<double> j2 = DLambda_dJ(s, dj);
    ASSERT_EQ(m1.size(), m2.size());
    ASSERT_EQ(j1.size(), j2.size());
    EXPECT_EQ(0, std::memcmp(m1.data(), m2.data(), m1.size() * sizeof(double)))
        << "D1: d/dM directional derivative differs across two runs";
    EXPECT_EQ(0, std::memcmp(j1.data(), j2.data(), j1.size() * sizeof(double)))
        << "D1: d/dJ directional derivative differs across two runs";
    std::printf("[Ch1 D1] d/dM + d/dJ byte-identical across two runs\n");
}
