// ---------------------------------------------------------------------------
// nuka::diffsim -- SDF-contact IFT d/dM, d/dJ channel (v0.7 p08-C, §4.B).
// See sdf_contact_ift.hpp for the full derivation. Dense fp64 host reference.
// ---------------------------------------------------------------------------

#include "diffsim/sdf_contact_ift.hpp"

#include <cmath>
#include <stdexcept>

namespace nuka::diffsim {

namespace {

// Dense SPD solve via Cholesky (A = L L^T), forward/back substitution. A is
// n*n row-major SPD. Falls back to a tiny diagonal jitter only if a pivot is
// non-positive (the active SDF-contact Delassus is SPD when the active rows are
// independent; a structurally dead row would make it PSD -- the jitter keeps the
// reference finite, mirroring the GPU BlockJacobi modified-Cholesky null-handling).
std::vector<double> CholeskySolve(const std::vector<double>& a_in,
                                  const std::vector<double>& r, uint32_t n) {
    std::vector<double> a = a_in;  // working copy
    std::vector<double> l(static_cast<size_t>(n) * n, 0.0);
    constexpr double kPivotFloor = 1.0e-300;
    for (uint32_t i = 0u; i < n; ++i) {
        for (uint32_t j = 0u; j <= i; ++j) {
            double sum = a[static_cast<size_t>(i) * n + j];
            for (uint32_t k = 0u; k < j; ++k) {
                sum -= l[static_cast<size_t>(i) * n + k] * l[static_cast<size_t>(j) * n + k];
            }
            if (i == j) {
                if (sum <= kPivotFloor) {
                    // Null direction: keep finite (least-norm), zero this factor row.
                    l[static_cast<size_t>(i) * n + j] = 1.0;
                } else {
                    l[static_cast<size_t>(i) * n + j] = std::sqrt(sum);
                }
            } else {
                const double ljj = l[static_cast<size_t>(j) * n + j];
                l[static_cast<size_t>(i) * n + j] = (ljj != 0.0) ? sum / ljj : 0.0;
            }
        }
    }
    // Solve L y = r.
    std::vector<double> y(n, 0.0);
    for (uint32_t i = 0u; i < n; ++i) {
        double sum = r[i];
        for (uint32_t k = 0u; k < i; ++k) sum -= l[static_cast<size_t>(i) * n + k] * y[k];
        const double lii = l[static_cast<size_t>(i) * n + i];
        y[i] = (lii != 0.0) ? sum / lii : 0.0;
    }
    // Solve L^T x = y.
    std::vector<double> x(n, 0.0);
    for (uint32_t ii = 0u; ii < n; ++ii) {
        const uint32_t i = n - 1u - ii;
        double sum = y[i];
        for (uint32_t k = i + 1u; k < n; ++k) sum -= l[static_cast<size_t>(k) * n + i] * x[k];
        const double lii = l[static_cast<size_t>(i) * n + i];
        x[i] = (lii != 0.0) ? sum / lii : 0.0;
    }
    return x;
}

// y = M^-1 v  (dof). M^-1 is dof*dof row-major symmetric.
std::vector<double> ApplyMinv(const SdfContactDenseSystem& s, const std::vector<double>& v) {
    std::vector<double> y(s.dof, 0.0);
    for (uint32_t i = 0u; i < s.dof; ++i) {
        double acc = 0.0;
        for (uint32_t k = 0u; k < s.dof; ++k)
            acc += s.m_inv[static_cast<size_t>(i) * s.dof + k] * v[k];
        y[i] = acc;
    }
    return y;
}

// y = J w  (n), J is n*dof row-major.
std::vector<double> ApplyJ(const SdfContactDenseSystem& s, const std::vector<double>& w) {
    std::vector<double> y(s.n, 0.0);
    for (uint32_t r = 0u; r < s.n; ++r) {
        double acc = 0.0;
        for (uint32_t c = 0u; c < s.dof; ++c)
            acc += s.j[static_cast<size_t>(r) * s.dof + c] * w[c];
        y[r] = acc;
    }
    return y;
}

void CheckShapes(const SdfContactDenseSystem& s) {
    if (s.n == 0u || s.dof == 0u)
        throw std::invalid_argument("SdfContactDenseSystem: n and dof must be > 0");
    if (s.j.size() != static_cast<size_t>(s.n) * s.dof ||
        s.m_inv.size() != static_cast<size_t>(s.dof) * s.dof || s.b_c.size() != s.n ||
        s.qdot_free.size() != s.dof)
        throw std::invalid_argument("SdfContactDenseSystem: shape mismatch");
}

}  // namespace

std::vector<double> AssembleDelassus(const SdfContactDenseSystem& s) {
    CheckShapes(s);
    // A = J M^-1 J^T. Form W = M^-1 J^T (dof x n), then A = J W (n x n).
    // W column r = M^-1 * (J row r).
    std::vector<double> a(static_cast<size_t>(s.n) * s.n, 0.0);
    std::vector<std::vector<double>> w(s.n);  // w[r] = M^-1 J_r  (dof)
    for (uint32_t r = 0u; r < s.n; ++r) {
        std::vector<double> jr(s.dof);
        for (uint32_t c = 0u; c < s.dof; ++c) jr[c] = s.j[static_cast<size_t>(r) * s.dof + c];
        w[r] = ApplyMinv(s, jr);
    }
    for (uint32_t i = 0u; i < s.n; ++i) {
        for (uint32_t k = 0u; k < s.n; ++k) {
            double acc = 0.0;
            for (uint32_t c = 0u; c < s.dof; ++c)
                acc += s.j[static_cast<size_t>(i) * s.dof + c] * w[k][c];
            a[static_cast<size_t>(i) * s.n + k] = acc;
        }
    }
    return a;
}

std::vector<double> AssembleRhs(const SdfContactDenseSystem& s) {
    CheckShapes(s);
    std::vector<double> jqf = ApplyJ(s, s.qdot_free);
    std::vector<double> r(s.n);
    for (uint32_t i = 0u; i < s.n; ++i) r[i] = s.b_c[i] - jqf[i];
    return r;
}

std::vector<double> SolveSpd(const std::vector<double>& a, const std::vector<double>& r,
                             uint32_t n) {
    return CholeskySolve(a, r, n);
}

std::vector<double> SolveLambda(const SdfContactDenseSystem& s) {
    const std::vector<double> a = AssembleDelassus(s);
    const std::vector<double> r = AssembleRhs(s);
    return CholeskySolve(a, r, s.n);
}

std::vector<double> DLambda_dMinv(const SdfContactDenseSystem& s,
                                  const std::vector<double>& dminv) {
    CheckShapes(s);
    if (dminv.size() != static_cast<size_t>(s.dof) * s.dof)
        throw std::invalid_argument("DLambda_dMinv: dminv shape mismatch");
    std::vector<double> lambda = s.lambda.empty() ? SolveLambda(s) : s.lambda;
    if (lambda.size() != s.n)
        throw std::invalid_argument("DLambda_dMinv: lambda size mismatch");

    const std::vector<double> a = AssembleDelassus(s);

    // dA lambda = (J dMinv J^T) lambda. Compute right-to-left:
    //   t = J^T lambda      (dof)
    //   u = dMinv t         (dof)
    //   dA_lambda = J u     (n)
    std::vector<double> jt_lambda(s.dof, 0.0);
    for (uint32_t c = 0u; c < s.dof; ++c) {
        double acc = 0.0;
        for (uint32_t r = 0u; r < s.n; ++r)
            acc += s.j[static_cast<size_t>(r) * s.dof + c] * lambda[r];
        jt_lambda[c] = acc;
    }
    std::vector<double> u(s.dof, 0.0);
    for (uint32_t i = 0u; i < s.dof; ++i) {
        double acc = 0.0;
        for (uint32_t k = 0u; k < s.dof; ++k)
            acc += dminv[static_cast<size_t>(i) * s.dof + k] * jt_lambda[k];
        u[i] = acc;
    }
    std::vector<double> da_lambda = ApplyJ(s, u);  // n

    // dr = 0. dlambda = A^-1 ( -dA lambda ).
    std::vector<double> rhs(s.n);
    for (uint32_t i = 0u; i < s.n; ++i) rhs[i] = -da_lambda[i];
    return CholeskySolve(a, rhs, s.n);
}

std::vector<double> DLambda_dJ(const SdfContactDenseSystem& s,
                               const std::vector<double>& dj) {
    CheckShapes(s);
    if (dj.size() != static_cast<size_t>(s.n) * s.dof)
        throw std::invalid_argument("DLambda_dJ: dj shape mismatch");
    std::vector<double> lambda = s.lambda.empty() ? SolveLambda(s) : s.lambda;
    if (lambda.size() != s.n)
        throw std::invalid_argument("DLambda_dJ: lambda size mismatch");

    const std::vector<double> a = AssembleDelassus(s);

    // dA = dJ M^-1 J^T + J M^-1 dJ^T.  dA lambda has two terms.
    // Precompute the M^-1 J^T columns: w[r] = M^-1 J_r (dof). (Same as in A.)
    std::vector<std::vector<double>> w(s.n);
    for (uint32_t r = 0u; r < s.n; ++r) {
        std::vector<double> jr(s.dof);
        for (uint32_t c = 0u; c < s.dof; ++c) jr[c] = s.j[static_cast<size_t>(r) * s.dof + c];
        w[r] = ApplyMinv(s, jr);
    }

    // TERM 1: (dJ M^-1 J^T) lambda = dJ * ( M^-1 J^T lambda ) = dJ * (sum_r lambda[r] w[r]).
    std::vector<double> minv_jt_lambda(s.dof, 0.0);
    for (uint32_t r = 0u; r < s.n; ++r)
        for (uint32_t c = 0u; c < s.dof; ++c) minv_jt_lambda[c] += lambda[r] * w[r][c];
    std::vector<double> term1(s.n, 0.0);  // dJ * minv_jt_lambda
    for (uint32_t r = 0u; r < s.n; ++r) {
        double acc = 0.0;
        for (uint32_t c = 0u; c < s.dof; ++c)
            acc += dj[static_cast<size_t>(r) * s.dof + c] * minv_jt_lambda[c];
        term1[r] = acc;
    }

    // TERM 2: (J M^-1 dJ^T) lambda = J * M^-1 * (dJ^T lambda).
    //   dJ^T lambda (dof) = sum_r lambda[r] * dJ_r
    std::vector<double> djt_lambda(s.dof, 0.0);
    for (uint32_t r = 0u; r < s.n; ++r)
        for (uint32_t c = 0u; c < s.dof; ++c)
            djt_lambda[c] += lambda[r] * dj[static_cast<size_t>(r) * s.dof + c];
    std::vector<double> minv_djt_lambda = ApplyMinv(s, djt_lambda);  // dof
    std::vector<double> term2 = ApplyJ(s, minv_djt_lambda);          // n

    // dr = -dJ qdot_free  (NONZERO -- r = b_c - J qdot_free).
    std::vector<double> dr(s.n, 0.0);
    for (uint32_t r = 0u; r < s.n; ++r) {
        double acc = 0.0;
        for (uint32_t c = 0u; c < s.dof; ++c)
            acc += dj[static_cast<size_t>(r) * s.dof + c] * s.qdot_free[c];
        dr[r] = -acc;
    }

    // dlambda = A^-1 ( dr - dA lambda ) = A^-1 ( dr - term1 - term2 ).
    std::vector<double> rhs(s.n);
    for (uint32_t i = 0u; i < s.n; ++i) rhs[i] = dr[i] - term1[i] - term2[i];
    return CholeskySolve(a, rhs, s.n);
}

}  // namespace nuka::diffsim
