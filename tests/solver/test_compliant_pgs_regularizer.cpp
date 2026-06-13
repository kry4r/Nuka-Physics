// ---------------------------------------------------------------------------
// v0.8 C5b-core: compliant-PGS regularizer R-SWEEP guard -- the nk-core re-point
// (M9 T11-articulated-fold).
// ---------------------------------------------------------------------------
// The legacy `solver::UnifiedSolve` host driver + `solver/gpu/row_solver.cu` are
// deleted in M9 T11-core; the regularized-PGS compliant update (the 210c11f
// -R*old_impulse feedback term) was migrated into the nk SolveRowsBlockIsland op
// (src/phi/backend_cuda/ops/solve_rows.cu). This guard now drives the SAME
// coupled full-rank-SPD compliant system through the nk op via the M4 re-point
// harness (tests/solver/nk_solve_harness.hpp) and holds the SAME R-sweep
// gates: kernel lambda == double-precision dense (A+diag(R))lambda=b across R/A
// from ~1e-3 to ~1e+1, the large-R discriminator vs the pre-fix A*lambda=b fixed
// point, and D1 -- the regularizer's R-independence is asserted on the nk path.
//
// Commit 210c11f fixed the regularized Gauss-Seidel update so it carries the
// matching -R*old_impulse feedback term:
//     lambda = effective_mass * (rhs_v - jv - compliance_alpha*old_impulse)
// WITHOUT that term the per-row increment's fixed point is A*lambda = b (the R
// in the denominator cancels), so the kernel converges to a point ~R/A above the
// EXACT solve of its OWN assembled (A + diag(R))lambda = b system. The bug is
// problem-dependent (the bias is exactly R/A), and it was only ever verified at
// ONE operating point (the go2 foot-ground MJX-parity test, R/A ~= 0.0057).
//
// This test is the MJX-FREE, R-INDEPENDENT proof: it constructs a small COUPLED
// compliant contact system with a KNOWN, full-rank SPD effective matrix A, drives
// it through the SAME UnifiedSolve compliant PGS kernel the production-flip will
// use, and compares the kernel's converged lambda to a DOUBLE-PRECISION dense
// direct solve of (A + diag(R))lambda = b -- across R spanning >=4 orders of
// magnitude (R/A from ~1e-3 to ~1e+1, including R comparable to and far larger
// than A, where the OLD bug's R/A error would have been enormous).
//
// HOW A IS MADE FULL-RANK SPD (the crux):
//   The contact-space matrix is A = J M^-1 J^T (N x N), rank <= min(N, dof). For
//   A to be full-rank SPD we MUST have dof_stride >= N AND N linearly-independent
//   chain-J rows on ONE articulation. So: dof_stride = N = 4, a fixed 4x4 SPD
//   M^-1 (symmetric, diagonally dominant), and 4 independent chain-J rows. We
//   prove A is SPD on the host by checking POSITIVE PIVOTS in the elimination.
//
// COUPLING: all N rows share ONE articulation (same art_index -> same coloring
// key body_count+art_index, distinct chain_jac_slot per contact). They serialize
// into N graph colors (Gauss-Seidel), all reading/writing the SHARED qdot slice,
// so the off-diagonal A_ij = J_i M^-1 J_j^T genuinely couples the rows -- exactly
// the regime co-residence (box/arm mass ratios -> different A, different R/A)
// will hit. The unique fixed point of converged GS over a color-serialized SPD
// system is the (A + diag(R))lambda = b direct solve, independent of sweep order.
//
// DISCRIMINATION (the guard's permanent teeth): we ALSO dense-solve the pre-fix
// fixed point A*lambda = b (R cancels) and assert at the large-R case that the
// kernel matches (A + diag(R)) and is FAR from A*lambda=b -- so the test would
// fail under the pre-210c11f code without needing the revert. (A manual revert
// was ALSO run once to confirm empirically; see the report. The fix is restored.)
// ---------------------------------------------------------------------------

#include "constraint/collidable.hpp"
#include "constraint/contact_row_sides.hpp"
#include "constraint/row.hpp"
#include "constraint/row_articulation_refs.hpp"
#include "constraint/row_buffers.hpp"
#include "nk_solve_harness.hpp"
#include "runtime/rigid/body_state.hpp"
#include "solver/solver_config.hpp"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <vector>

using namespace nuka::constraint;
using nuka::runtime::rigid::BodyState;
namespace math = nuka::math;

namespace {

// ----------------------------------------------------------------------------
// The synthetic coupled articulation. N contacts on ONE N-DOF articulation.
// ----------------------------------------------------------------------------
constexpr uint32_t kN = 4u;  // contact rows == DOFs (dof_stride == N -> A full rank)

// A fixed, symmetric, strictly diagonally-dominant (=> SPD) 4x4 M (mass matrix),
// row-major. We invert it ON THE HOST in double precision below; the kernel
// consumes the fp32 M^-1 tile. Diagonal dominance guarantees SPD, and makes the
// resulting A well-conditioned so fp32 PGS converges to 1e-5 even at small R.
const std::array<double, kN * kN>& MassMatrix() {
    static const std::array<double, kN * kN> kM = {
        // strongly diagonally dominant, symmetric
        12.0,  1.5, -0.8,  0.6,
         1.5, 10.0,  1.1, -0.7,
        -0.8,  1.1,  9.0,  1.3,
         0.6, -0.7,  1.3,  8.0,
    };
    return kM;
}

// N linearly-independent chain-J rows (each dof_stride == N floats). The matrix
// whose ROWS are these is well-conditioned (close to a perturbed identity), so
// J has full row rank => A = J M^-1 J^T is full-rank SPD.
const std::array<std::array<double, kN>, kN>& ChainJRows() {
    static const std::array<std::array<double, kN>, kN> kJ = {{
        {{ 1.00,  0.20, -0.10,  0.05}},
        {{ 0.15,  0.90,  0.25, -0.10}},
        {{-0.05,  0.10,  1.10,  0.20}},
        {{ 0.10, -0.15,  0.05,  0.95}},
    }};
    return kJ;
}

// Nonzero start-of-solve joint velocities (so Jv0 != 0 -- the missing-Jv path is
// only exercised with nonzero qdot, mirroring test_link_rigid_twoway's note).
const std::array<double, kN>& Qdot0() {
    static const std::array<double, kN> kQ = {{0.9, -0.3, 0.5, 0.2}};
    return kQ;
}

// ----------------------------------------------------------------------------
// Host double-precision dense linear algebra (test-only oracle).
// ----------------------------------------------------------------------------

// Invert an n x n matrix (row-major) via Gauss-Jordan with partial pivoting.
// Returns false if singular. Pure double precision (oracle).
bool InvertDense(const std::vector<double>& in, uint32_t n, std::vector<double>* out) {
    std::vector<double> a = in;            // working copy
    out->assign(static_cast<size_t>(n) * n, 0.0);
    for (uint32_t i = 0; i < n; ++i) (*out)[static_cast<size_t>(i) * n + i] = 1.0;
    for (uint32_t col = 0; col < n; ++col) {
        // partial pivot
        uint32_t pivot = col;
        double best = std::fabs(a[static_cast<size_t>(col) * n + col]);
        for (uint32_t r = col + 1; r < n; ++r) {
            const double v = std::fabs(a[static_cast<size_t>(r) * n + col]);
            if (v > best) { best = v; pivot = r; }
        }
        if (best < 1e-14) return false;  // singular
        if (pivot != col) {
            for (uint32_t k = 0; k < n; ++k) {
                std::swap(a[static_cast<size_t>(col) * n + k], a[static_cast<size_t>(pivot) * n + k]);
                std::swap((*out)[static_cast<size_t>(col) * n + k], (*out)[static_cast<size_t>(pivot) * n + k]);
            }
        }
        const double diag = a[static_cast<size_t>(col) * n + col];
        for (uint32_t k = 0; k < n; ++k) {
            a[static_cast<size_t>(col) * n + k] /= diag;
            (*out)[static_cast<size_t>(col) * n + k] /= diag;
        }
        for (uint32_t r = 0; r < n; ++r) {
            if (r == col) continue;
            const double f = a[static_cast<size_t>(r) * n + col];
            if (f == 0.0) continue;
            for (uint32_t k = 0; k < n; ++k) {
                a[static_cast<size_t>(r) * n + k] -= f * a[static_cast<size_t>(col) * n + k];
                (*out)[static_cast<size_t>(r) * n + k] -= f * (*out)[static_cast<size_t>(col) * n + k];
            }
        }
    }
    return true;
}

// Solve an n x n SPD system M x = rhs via Gaussian elimination with partial
// pivoting (double). ALSO records the n pivot magnitudes so the caller can prove
// SPD-ness (all pivots strictly positive for an SPD matrix run without pivot
// swaps -- we additionally assert each chosen pivot is positive). Returns false
// if singular.
bool SolveDense(const std::vector<double>& M_in, const std::vector<double>& rhs,
                uint32_t n, std::vector<double>* x, std::vector<double>* pivots_out) {
    std::vector<double> a = M_in;
    std::vector<double> b = rhs;
    if (pivots_out != nullptr) pivots_out->assign(n, 0.0);
    for (uint32_t col = 0; col < n; ++col) {
        uint32_t pivot = col;
        double best = std::fabs(a[static_cast<size_t>(col) * n + col]);
        for (uint32_t r = col + 1; r < n; ++r) {
            const double v = std::fabs(a[static_cast<size_t>(r) * n + col]);
            if (v > best) { best = v; pivot = r; }
        }
        if (best < 1e-14) return false;
        if (pivot != col) {
            for (uint32_t k = 0; k < n; ++k)
                std::swap(a[static_cast<size_t>(col) * n + k], a[static_cast<size_t>(pivot) * n + k]);
            std::swap(b[col], b[pivot]);
        }
        const double diag = a[static_cast<size_t>(col) * n + col];
        if (pivots_out != nullptr) (*pivots_out)[col] = diag;
        for (uint32_t r = col + 1; r < n; ++r) {
            const double f = a[static_cast<size_t>(r) * n + col] / diag;
            for (uint32_t k = col; k < n; ++k)
                a[static_cast<size_t>(r) * n + k] -= f * a[static_cast<size_t>(col) * n + k];
            b[r] -= f * b[col];
        }
    }
    x->assign(n, 0.0);
    for (int col = static_cast<int>(n) - 1; col >= 0; --col) {
        double s = b[col];
        for (uint32_t k = col + 1; k < n; ++k)
            s -= a[static_cast<size_t>(col) * n + k] * (*x)[k];
        (*x)[col] = s / a[static_cast<size_t>(col) * n + col];
    }
    return true;
}

// Build the contact-space effective matrix A = J M^-1 J^T (N x N, row-major) and
// rhs b_i = aref_i*dt - Jv0_i from the SAME fp32 values the kernel uploads, but
// accumulate in double. `Minv_f` is the fp32 M^-1 tile (row-major, N*N).
struct DenseSystem {
    std::vector<double> A;   // N*N effective matrix
    std::vector<double> b;   // N rhs
    std::vector<double> jv0; // N (informational)
};

DenseSystem BuildDenseSystem(const std::vector<float>& Minv_f,
                             const std::vector<float>& chain_jacobians_f,
                             const std::vector<float>& qdot0_f,
                             const std::vector<float>& aref_f,
                             float dt, uint32_t n) {
    DenseSystem s;
    s.A.assign(static_cast<size_t>(n) * n, 0.0);
    s.b.assign(n, 0.0);
    s.jv0.assign(n, 0.0);
    // Promote to double from the EXACT fp32 inputs.
    std::vector<double> Minv(static_cast<size_t>(n) * n);
    for (size_t i = 0; i < Minv.size(); ++i) Minv[i] = static_cast<double>(Minv_f[i]);
    auto Jrow = [&](uint32_t i, uint32_t c) {
        return static_cast<double>(chain_jacobians_f[static_cast<size_t>(i) * n + c]);
    };
    // A_ij = J_i M^-1 J_j^T.
    for (uint32_t i = 0; i < n; ++i) {
        // M^-1 J_i^T
        std::vector<double> minv_jit(n, 0.0);
        for (uint32_t r = 0; r < n; ++r) {
            double acc = 0.0;
            for (uint32_t c = 0; c < n; ++c) acc += Minv[static_cast<size_t>(r) * n + c] * Jrow(i, c);
            minv_jit[r] = acc;
        }
        for (uint32_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (uint32_t r = 0; r < n; ++r) acc += Jrow(j, r) * minv_jit[r];
            s.A[static_cast<size_t>(j) * n + i] = acc;  // A_ji; A symmetric
        }
        // Jv0_i = sum_r J_i[r] * qdot0[r]
        double jv = 0.0;
        for (uint32_t r = 0; r < n; ++r) jv += Jrow(i, r) * static_cast<double>(qdot0_f[r]);
        s.jv0[i] = jv;
        s.b[i] = static_cast<double>(aref_f[i]) * static_cast<double>(dt) - jv;
    }
    return s;
}

// ----------------------------------------------------------------------------
// Row / context construction helpers (mirror test_link_rigid_twoway).
// ----------------------------------------------------------------------------
CollidableRef MakeArticulationRef(uint32_t handle) {
    CollidableRef ref;
    ref.type = CollidableType::ArticulationLink;
    ref.react = ReactionProviderKind::ArticulationChainJ;
    ref.handle = handle;
    return ref;
}

CollidableRef MakeStaticRef() {
    CollidableRef ref;
    ref.type = CollidableType::StaticWorld;
    ref.react = ReactionProviderKind::StaticNull;
    ref.handle = kInvalidBodyIndex;
    return ref;
}

// Append ONE compliant normal row (one contact) on the shared articulation,
// chain_jac_slot == slot, compliance_alpha == R, rhs == aref.
void AppendRow(RowBuffers* rows, std::vector<ContactRowSides>* sides,
               std::vector<RowArticulationRefs>* art_refs, uint32_t body_count,
               uint32_t slot, float aref, float R) {
    Row row;
    row.row_class_id = kMaximalContactRowClassId;
    row.rhs = aref;
    row.lambda = 0.0f;
    row.lower = 0.0f;
    row.upper = kRowHugeLimit;  // unilateral normal [0, huge]
    row.compliance_alpha = R;
    row.flags = row_flags::Compliant | row_flags::Unilateral;

    RowMaterial material;
    material.kind = RowKind::Contact;
    material.group_id = rows->RowCount();  // each row its own normal-group head
    material.normal_row_count = 1u;
    material.first_friction_row = 0u;
    material.friction_row_count = 0u;
    material.friction = 0.0f;
    material.restitution = 0.0f;
    material.position_error = 0.0f;

    const RowJacobian6 zero_jac{math::Vec3::Zero(), math::Vec3::Zero()};
    constexpr uint32_t kArtIndex = 0u;  // ALL rows share one articulation -> coupled
    rows->AddRow(row, RowBodyPair{body_count + kArtIndex, kInvalidBodyIndex},
                 zero_jac, zero_jac, material);
    sides->push_back(ContactRowSides{MakeArticulationRef(0u), MakeStaticRef()});
    art_refs->push_back(RowArticulationRefs{RowArticulationSide{kArtIndex, slot},
                                            RowArticulationSide{}});
}

// Many-iteration config: drive the coupled GS to convergence.
nuka::solver::SolverConfig ConvergedConfig() {
    nuka::solver::SolverConfig cfg;
    cfg.velocity_iterations = 100u;
    cfg.position_iterations = 0u;
    cfg.slop = 0.0f;
    cfg.baumgarte = 0.0f;
    return cfg;
}

constexpr float kDt = 0.01f;

// Run the kernel for one R value; returns the converged per-row lambda. If
// qdot_out != nullptr, also returns the MUTATED live qdot slice (the apply path)
// so the determinism guard can cover both outputs.
std::vector<float> RunKernel(const std::vector<float>& Minv_f,
                            const std::vector<float>& chain_jacobians_f,
                            const std::vector<float>& qdot0_f,
                            const std::vector<float>& aref_f, float R,
                            std::vector<float>* qdot_out = nullptr) {
    std::vector<BodyState> bodies(1);  // one UNREFERENCED rigid body so body_count>=1
    bodies[0].inv_mass = 1.0f;
    const uint32_t body_count = static_cast<uint32_t>(bodies.size());

    RowBuffers rows;
    std::vector<ContactRowSides> sides;
    std::vector<RowArticulationRefs> art_refs;
    for (uint32_t slot = 0; slot < kN; ++slot) {
        AppendRow(&rows, &sides, &art_refs, body_count, slot, aref_f[slot], R);
    }

    // nk re-point: drive the assembled coupled compliant rows through the nk
    // SolveRowsBlockIsland op (the migrated regularized-PGS update).
    const auto res = nk_harness::NkSolveRows(
        rows, sides, art_refs, chain_jacobians_f, Minv_f, qdot0_f, &bodies, kN,
        static_cast<uint16_t>(ConvergedConfig().velocity_iterations), kDt);
    EXPECT_TRUE(res.ok) << "nk solve harness failed";

    std::vector<float> lambda(kN, 0.0f);
    if (res.ok) {
        for (uint32_t i = 0; i < kN; ++i) lambda[i] = res.lambda[i];
        if (qdot_out != nullptr) *qdot_out = res.qdot;  // mutated joint velocities
    } else if (qdot_out != nullptr) {
        *qdot_out = qdot0_f;
    }
    return lambda;
}

// Build the fp32 inputs once (the kernel and the oracle both consume these).
struct Inputs {
    std::vector<float> Minv;            // N*N fp32 M^-1 tile (row-major)
    std::vector<float> chain_jacobians; // N rows * N floats
    std::vector<float> qdot0;           // N
};

Inputs MakeInputs() {
    Inputs in;
    // Invert the (double) mass matrix, then downcast to fp32.
    std::vector<double> M(MassMatrix().begin(), MassMatrix().end());
    std::vector<double> Minv_d;
    EXPECT_TRUE(InvertDense(M, kN, &Minv_d)) << "mass matrix not invertible";
    in.Minv.resize(Minv_d.size());
    for (size_t i = 0; i < Minv_d.size(); ++i) in.Minv[i] = static_cast<float>(Minv_d[i]);

    in.chain_jacobians.resize(static_cast<size_t>(kN) * kN);
    for (uint32_t i = 0; i < kN; ++i)
        for (uint32_t c = 0; c < kN; ++c)
            in.chain_jacobians[static_cast<size_t>(i) * kN + c] =
                static_cast<float>(ChainJRows()[i][c]);

    in.qdot0.resize(kN);
    for (uint32_t r = 0; r < kN; ++r) in.qdot0[r] = static_cast<float>(Qdot0()[r]);
    return in;
}

double RelErr(double a, double b) {
    const double denom = std::fabs(b) > 1e-12 ? std::fabs(b) : 1.0;
    return std::fabs(a - b) / denom;
}

}  // namespace

// ===========================================================================
// THE R-SWEEP GUARD: kernel lambda == dense (A + diag(R)) lambda = b, every R.
// ===========================================================================
TEST(CompliantPgsRegularizer, KernelMatchesAplusRDenseSolveAcrossRSweep) {
    const Inputs in = MakeInputs();

    // Choose aref so b (and hence the exact lambda) is comfortably positive at
    // every R, so the unilateral [0,huge] clamp stays INACTIVE -> the converged
    // PGS solves a pure linear system, comparable to the dense direct solve.
    // aref*dt is the velocity-target; pick large enough aref that b_i > 0.
    const std::vector<float> aref = {150.0f, 130.0f, 170.0f, 140.0f};

    // Build A, b ONCE (R-independent); prove A is full-rank SPD via positive
    // pivots in its own Cholesky-equivalent elimination.
    const DenseSystem sys = BuildDenseSystem(in.Minv, in.chain_jacobians, in.qdot0,
                                             aref, kDt, kN);
    {
        std::vector<double> x, pivots;
        ASSERT_TRUE(SolveDense(sys.A, sys.b, kN, &x, &pivots))
            << "effective matrix A is singular -- dof_stride < N or rank-deficient J";
        for (uint32_t i = 0; i < kN; ++i) {
            EXPECT_GT(pivots[i], 1e-9)
                << "A pivot[" << i << "] not strictly positive -- A is not SPD";
        }
    }

    // Sweep R across >= 4 orders of magnitude. trace(A)/N ~ A scale; pick R from
    // ~1e-3*A up to ~1e+1*A so we hit R << A, R ~ A, and R >> A.
    double a_scale = 0.0;
    for (uint32_t i = 0; i < kN; ++i) a_scale += sys.A[static_cast<size_t>(i) * kN + i];
    a_scale /= kN;  // mean diagonal of A (a representative A magnitude)

    const std::vector<double> r_over_a = {1e-3, 1e-2, 1e-1, 1.0, 3.0, 1e+1};

    double sweep_max_rel_err = 0.0;
    double sweep_max_rel_err_at = 0.0;
    for (double ratio : r_over_a) {
        const float R = static_cast<float>(ratio * a_scale);

        // Dense oracle: (A + diag(R)) lambda = b in double.
        std::vector<double> A_plus_R = sys.A;
        for (uint32_t i = 0; i < kN; ++i) A_plus_R[static_cast<size_t>(i) * kN + i] += R;
        std::vector<double> lambda_exact, pivots;
        ASSERT_TRUE(SolveDense(A_plus_R, sys.b, kN, &lambda_exact, &pivots))
            << "(A+R) singular at R/A=" << ratio;

        // SETUP PRECONDITION: exact solution strictly positive componentwise, so
        // the [0,huge] clamp is inactive at convergence (linear-vs-PGS compare).
        for (uint32_t i = 0; i < kN; ++i) {
            ASSERT_GT(lambda_exact[i], 1e-4)
                << "exact lambda[" << i << "] <= 0 at R/A=" << ratio
                << " -- clamp would corrupt the linear-vs-PGS comparison";
        }

        const std::vector<float> lambda_kernel =
            RunKernel(in.Minv, in.chain_jacobians, in.qdot0, aref, R);

        double max_rel = 0.0;
        for (uint32_t i = 0; i < kN; ++i) {
            const double rel = RelErr(static_cast<double>(lambda_kernel[i]), lambda_exact[i]);
            max_rel = std::max(max_rel, rel);
        }
        if (max_rel > sweep_max_rel_err) {
            sweep_max_rel_err = max_rel;
            sweep_max_rel_err_at = ratio;
        }
        EXPECT_LT(max_rel, 1e-5)
            << "kernel lambda != dense (A+diag(R)) solve at R/A=" << ratio
            << " (R=" << R << "), max rel err=" << max_rel
            << " -- the compliant-PGS regularizer fix (210c11f) is NOT R-independent";
    }
    // Surface the worst-case number (the report quotes this).
    RecordProperty("sweep_max_rel_err", std::to_string(sweep_max_rel_err));
    RecordProperty("sweep_max_rel_err_at_R_over_A", std::to_string(sweep_max_rel_err_at));
    EXPECT_LT(sweep_max_rel_err, 1e-5)
        << "worst-case sweep rel err " << sweep_max_rel_err
        << " at R/A=" << sweep_max_rel_err_at;
}

// ===========================================================================
// DISCRIMINATION: at LARGE R/A the kernel must match (A+diag(R)) and be FAR from
// the PRE-FIX fixed point A*lambda=b (which the broken update converged to). This
// bakes the guard's teeth in permanently -- it fails under pre-210c11f code.
// ===========================================================================
TEST(CompliantPgsRegularizer, LargeRDiscriminatesAgainstPreFixFixedPoint) {
    const Inputs in = MakeInputs();
    const std::vector<float> aref = {150.0f, 130.0f, 170.0f, 140.0f};
    const DenseSystem sys = BuildDenseSystem(in.Minv, in.chain_jacobians, in.qdot0,
                                             aref, kDt, kN);

    double a_scale = 0.0;
    for (uint32_t i = 0; i < kN; ++i) a_scale += sys.A[static_cast<size_t>(i) * kN + i];
    a_scale /= kN;

    // Large R/A: the bug's bias ~ R/A would be order-1 here.
    const float R = static_cast<float>(10.0 * a_scale);

    // Correct fixed point: (A + diag(R)) lambda = b.
    std::vector<double> A_plus_R = sys.A;
    for (uint32_t i = 0; i < kN; ++i) A_plus_R[static_cast<size_t>(i) * kN + i] += R;
    std::vector<double> lambda_AR, lambda_A, pivots;
    ASSERT_TRUE(SolveDense(A_plus_R, sys.b, kN, &lambda_AR, &pivots));
    // Pre-fix fixed point: A lambda = b (the R-in-denominator-only bug; R cancels).
    ASSERT_TRUE(SolveDense(sys.A, sys.b, kN, &lambda_A, &pivots))
        << "A singular -- cannot form the pre-fix discriminator";

    // The two fixed points must differ by a LARGE margin at this R (else the test
    // would not discriminate the fix from the bug).
    double max_rel_gap = 0.0;
    for (uint32_t i = 0; i < kN; ++i)
        max_rel_gap = std::max(max_rel_gap, RelErr(lambda_A[i], lambda_AR[i]));
    EXPECT_GT(max_rel_gap, 0.5)
        << "the pre-fix (A*lambda=b) and post-fix (A+R) fixed points are too close "
           "at R/A=10 (gap=" << max_rel_gap << ") -- test does not discriminate";

    const std::vector<float> lambda_kernel =
        RunKernel(in.Minv, in.chain_jacobians, in.qdot0, aref, R);

    double max_rel_to_AR = 0.0, max_rel_to_A = 0.0;
    for (uint32_t i = 0; i < kN; ++i) {
        max_rel_to_AR = std::max(max_rel_to_AR, RelErr(static_cast<double>(lambda_kernel[i]), lambda_AR[i]));
        max_rel_to_A  = std::max(max_rel_to_A,  RelErr(static_cast<double>(lambda_kernel[i]), lambda_A[i]));
    }
    RecordProperty("largeR_rel_to_AplusR", std::to_string(max_rel_to_AR));
    RecordProperty("largeR_rel_to_Aonly", std::to_string(max_rel_to_A));
    RecordProperty("largeR_AR_vs_A_gap", std::to_string(max_rel_gap));

    // Kernel matches the CORRECT (A+R) solve...
    EXPECT_LT(max_rel_to_AR, 1e-5)
        << "kernel does not match the (A+diag(R)) solve at R/A=10 (rel=" << max_rel_to_AR << ")";
    // ...and is FAR from the PRE-FIX A*lambda=b fixed point (the bug's target).
    EXPECT_GT(max_rel_to_A, 0.1)
        << "kernel matches the PRE-FIX A*lambda=b fixed point at R/A=10 (rel=" << max_rel_to_A
        << ") -- the 210c11f regularizer fix appears to be REVERTED/absent";
}

// ===========================================================================
// DETERMINISM (D1): two identical kernel runs -> byte-identical lambda.
// ===========================================================================
TEST(CompliantPgsRegularizer, DeterministicTwoRunByteExact) {
    const Inputs in = MakeInputs();
    const std::vector<float> aref = {150.0f, 130.0f, 170.0f, 140.0f};
    const DenseSystem sys = BuildDenseSystem(in.Minv, in.chain_jacobians, in.qdot0,
                                             aref, kDt, kN);
    double a_scale = 0.0;
    for (uint32_t i = 0; i < kN; ++i) a_scale += sys.A[static_cast<size_t>(i) * kN + i];
    a_scale /= kN;
    const float R = static_cast<float>(1.0 * a_scale);  // R ~ A

    std::vector<float> qa, qb;
    const std::vector<float> a = RunKernel(in.Minv, in.chain_jacobians, in.qdot0, aref, R, &qa);
    const std::vector<float> b = RunKernel(in.Minv, in.chain_jacobians, in.qdot0, aref, R, &qb);
    ASSERT_EQ(a.size(), b.size());
    EXPECT_EQ(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), 0)
        << "two identical nk solve runs produced different lambda (nondeterministic)";
    // Also cover the apply path: the MUTATED live qdot slice must be byte-identical.
    ASSERT_EQ(qa.size(), qb.size());
    EXPECT_EQ(std::memcmp(qa.data(), qb.data(), qa.size() * sizeof(float)), 0)
        << "two identical nk solve runs produced different qdot (nondeterministic apply)";
}
