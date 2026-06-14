// ---------------------------------------------------------------------------
// M9 T11: XPBD SHAPE-MATCH (id 9) cluster tests, RE-POINTED to the nk path.
// The meshless cluster regularizer (Mueller et al. 2005), the 4th XPBD row class.
//
// The shape-match constraint is the one XPBD family that lived ONLY in the legacy
// legacy XPBD soft stepper; M9 T11 ports it to the nk particle ops
// (src/phi/backend_cuda/ops/particles.cu XpbdShapeMatchKernel + the CookXpbd
// shape-match cook). This suite asserts the SAME analytic/physical invariants the
// legacy p09-C suite asserted, now through nk::World (cook -> World -> Step):
//
//   1. POLAR dR/dA: a host replication of the device Higham Newton polar
//      decomposition, with the analytic derivative dR/dA via the skew-generator
//      solve (tr(S)I - S) w = axial(R^T dA - (R^T dA)^T), dR = R [w]_x, matched to
//      a host central-difference of R(A) -- rel-err < 1e-3. PURE host math (the
//      polar machinery is identical between the legacy + nk kernels), unchanged.
//   2. DISCRIMINATING cluster relaxation (on nk): a cluster initialized NON-rigidly
//      (anisotropic stretch + shear) relaxes under the shape-match constraint
//      ALONE toward a RIGID transform of the rest shape: final pairwise distances
//      return to the REST pairwise distances AND the recovered rotation equals
//      polar(A_init) (forces the nk device polar to be correct) AND the initial
//      sheared config did NOT already match (non-vacuous).
//   3. D1 two-run byte-exactness of the nk shape-match forward.
//
// A full Vellum golden is DEFERRED; these are ANALYTIC/physical invariants.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "nk/model/generated/field_ids.hpp"
#include "nk/model/model.hpp"
#include "nk/pipeline/world.hpp"
#include "phi/backend.hpp"
#include "phi/device_context.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>
#include <vector>

namespace {

namespace nk = nuka::nk;
namespace nphi = nuka::phi;
using nuka::math::Vec3;

// Downloaded particle state holder (replaces the legacy XpbdState).
struct XpbdState {
    std::vector<Vec3> positions;
    std::vector<Vec3> velocities;
};

// nk backend context (shared singleton, the nk_particle_equivalence pattern).
struct NkCtx { nphi::Device* dev = nullptr; nphi::Backend* backend = nullptr; };
NkCtx GetNkCtx() {
    static NkCtx c = [] {
        NkCtx r;
        static nuka::phi::DeviceContext keep = nuka::phi::MakeDefaultDeviceContext();
        (void)keep;
        r.dev = nphi::InitBestDevice();
        if (r.dev) r.backend = nphi::DeviceInitBackend(r.dev, nullptr);
        return r;
    }();
    return c;
}

// -------------------------------------------------------------------------
// Host replica of the device 3x3 polar machinery (row-major m[3r+c]). Kept
// here -- NOT in production math headers -- because it exists only to host-FD
// validate the device PolarRotation + its analytic derivative dR/dA.
// -------------------------------------------------------------------------
// DOUBLE precision. The device PolarRotation runs in float32; this host replica is
// in double for ONE reason: the polar dR/dA gate validates the MATHEMATICAL
// CORRECTNESS of the analytic derivative FORMULA (its derivation + axial-vector
// convention), which is precision-independent. A central-difference of the polar
// rotation R(A) -- whose individual dR entries can be O(1e-4) -- is dominated by
// float32 cancellation (~1.5e-2 floor, NOT a math error), so a float host FD would
// validate the float roundoff, not the formula. The exact analog is the volume
// row's grad-C gate, which host-FDs det(.) in exact host arithmetic. The float32
// device forward solve is exercised by Gate 2's relaxation (which lands rest
// pairwise distances at < 1e-2) -- so float convergence is covered there. The
// double polar here is algorithmically identical to the device (same Higham Newton
// iteration + det-flip), just at higher working precision.
struct M3 {
    double m[9];
};

M3 Zero3() {
    M3 a;
    for (int i = 0; i < 9; ++i) a.m[i] = 0.0;
    return a;
}
M3 Ident3() {
    M3 a = Zero3();
    a.m[0] = a.m[4] = a.m[8] = 1.0;
    return a;
}
M3 Mul3(const M3& a, const M3& b) {
    M3 c;
    for (int r = 0; r < 3; ++r)
        for (int col = 0; col < 3; ++col) {
            double s = 0.0;
            for (int k = 0; k < 3; ++k) s += a.m[3 * r + k] * b.m[3 * k + col];
            c.m[3 * r + col] = s;
        }
    return c;
}
M3 Transpose3(const M3& a) {
    M3 c;
    for (int r = 0; r < 3; ++r)
        for (int col = 0; col < 3; ++col) c.m[3 * r + col] = a.m[3 * col + r];
    return c;
}
double Det3(const M3& a) {
    return a.m[0] * (a.m[4] * a.m[8] - a.m[5] * a.m[7]) -
           a.m[1] * (a.m[3] * a.m[8] - a.m[5] * a.m[6]) +
           a.m[2] * (a.m[3] * a.m[7] - a.m[4] * a.m[6]);
}
// (a^-1)^T = cofactor(a)/det(a) -- algorithmically identical to the device
// Mat3InvTranspose.
M3 InvTranspose3(const M3& a, double eps) {
    const double det = Det3(a);
    if (std::fabs(det) < eps) return Ident3();
    const double id = 1.0 / det;
    M3 c;
    c.m[0] = (a.m[4] * a.m[8] - a.m[5] * a.m[7]) * id;
    c.m[1] = -(a.m[3] * a.m[8] - a.m[5] * a.m[6]) * id;
    c.m[2] = (a.m[3] * a.m[7] - a.m[4] * a.m[6]) * id;
    c.m[3] = -(a.m[1] * a.m[8] - a.m[2] * a.m[7]) * id;
    c.m[4] = (a.m[0] * a.m[8] - a.m[2] * a.m[6]) * id;
    c.m[5] = -(a.m[0] * a.m[7] - a.m[1] * a.m[6]) * id;
    c.m[6] = (a.m[1] * a.m[5] - a.m[2] * a.m[4]) * id;
    c.m[7] = -(a.m[0] * a.m[5] - a.m[2] * a.m[3]) * id;
    c.m[8] = (a.m[0] * a.m[4] - a.m[1] * a.m[3]) * id;
    return c;
}
// Higham Newton polar rotation R = polar(A), det-corrected -- the SAME iteration
// the device PolarRotation runs (R <- 0.5(R + R^-T), det-flip), at double working
// precision. 32 iterations (vs the device's 24; both far past convergence here).
M3 PolarR(const M3& A) {
    const int kPolarIters = 32;
    const double kDetEps = 1.0e-14;
    M3 R = A;
    for (int it = 0; it < kPolarIters; ++it) {
        const M3 RinvT = InvTranspose3(R, kDetEps);
        for (int i = 0; i < 9; ++i) R.m[i] = 0.5 * (R.m[i] + RinvT.m[i]);
    }
    if (Det3(R) < 0.0) {
        R.m[2] = -R.m[2];
        R.m[5] = -R.m[5];
        R.m[8] = -R.m[8];
    }
    return R;
}

// Analytic derivative of the polar rotation: dR = R [w]_x with
//     (tr(S) I - S) w = axial(B - B^T),  B = R^T dA,  S = sym(R^T A).
// Derivation: A = R S, RtR = I => W := R^T dR is skew; R^T dA = W S + dS, and
// since dS is symmetric the skew part gives skew(R^T dA) = skew(W S) = (1/2)(W S +
// S W^T) = (1/2)(W S - S W) (W skew). In axial coords W=[w]_x this is exactly
// (tr(S) I - S) w on the left and axial(B - B^T) on the right (factor 1; verified
// numerically to ratio 1.0 in double, see commit notes). Returns dR for the dA
// perturbation matrix.
M3 PolarDR(const M3& A, const M3& dA) {
    const M3 R = PolarR(A);
    const M3 Rt = Transpose3(R);
    // S = sym(R^T A) (symmetrize: the Newton R is not bit-exactly orthogonal).
    const M3 RtA = Mul3(Rt, A);
    M3 S;
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            S.m[3 * r + c] = 0.5 * (RtA.m[3 * r + c] + RtA.m[3 * c + r]);
    const double trS = S.m[0] + S.m[4] + S.m[8];
    // M = tr(S) I - S.
    M3 Mm = S;
    for (int i = 0; i < 9; ++i) Mm.m[i] = -Mm.m[i];
    Mm.m[0] += trS;
    Mm.m[4] += trS;
    Mm.m[8] += trS;
    // B = R^T dA ; rhs = axial(B - B^T) with axial(X)_x = X[2][1], _y = X[0][2],
    // _z = X[1][0] (so the rhs picks up 2*the off-diagonal of B).
    const M3 B = Mul3(Rt, dA);
    const double rhs[3] = {B.m[7] - B.m[5], B.m[2] - B.m[6], B.m[3] - B.m[1]};
    // Solve Mm w = rhs. Mm is symmetric, so Mm^-1 = (Mm^-1)^T = InvTranspose3(Mm).
    const M3 MmInv = InvTranspose3(Mm, 1.0e-14);
    const double w[3] = {
        MmInv.m[0] * rhs[0] + MmInv.m[1] * rhs[1] + MmInv.m[2] * rhs[2],
        MmInv.m[3] * rhs[0] + MmInv.m[4] * rhs[1] + MmInv.m[5] * rhs[2],
        MmInv.m[6] * rhs[0] + MmInv.m[7] * rhs[1] + MmInv.m[8] * rhs[2]};
    // [w]_x (skew-symmetric generator), then dR = R W.
    M3 W = Zero3();
    W.m[1] = -w[2];
    W.m[2] = w[1];
    W.m[3] = w[2];
    W.m[5] = -w[0];
    W.m[6] = -w[1];
    W.m[7] = w[0];
    return Mul3(R, W);
}

} // namespace

// Gate 1: analytic polar derivative dR/dA == host central-difference at a
// well-conditioned, non-degenerate cluster covariance (det A > 0, healthy
// singular values), and R is a proper rotation (det +1).
TEST(XpbdShapeMatch, PolarDerivativeMatchesFdAndIsProperRotation) {
    // A well-conditioned covariance: rotation * anisotropic stretch (no
    // reflection => det A > 0; distinct healthy singular values => the
    // (tr(S)I - S) solve is nonsingular, the det-flip branch inactive).
    M3 A;
    A.m[0] = 1.4; A.m[1] = 0.20; A.m[2] = -0.10;
    A.m[3] = 0.15; A.m[4] = 0.90; A.m[5] = 0.25;
    A.m[6] = -0.05; A.m[7] = 0.18; A.m[8] = 1.10;
    ASSERT_GT(Det3(A), 0.0) << "test cluster must be non-inverted (det A > 0)";

    const M3 R = PolarR(A);
    // Proper rotation: R^T R == I and det +1.
    const M3 RtR = Mul3(Transpose3(R), R);
    EXPECT_NEAR(RtR.m[0], 1.0, 1.0e-9);
    EXPECT_NEAR(RtR.m[4], 1.0, 1.0e-9);
    EXPECT_NEAR(RtR.m[8], 1.0, 1.0e-9);
    EXPECT_NEAR(RtR.m[1], 0.0, 1.0e-9);
    EXPECT_NEAR(RtR.m[5], 0.0, 1.0e-9);
    EXPECT_NEAR(Det3(R), 1.0, 1.0e-9) << "polar factor must be a proper rotation";

    // Central-difference dR/dA element-by-element vs the analytic derivative
    // contracted with the same single-entry perturbation. Double precision (see
    // the M3 note): this validates the derivative FORMULA, so the only residual is
    // the O(h^2) central-difference truncation error, not float32 cancellation.
    const double h = 1.0e-5;
    double max_rel = 0.0;
    for (int e = 0; e < 9; ++e) {
        M3 dA = Zero3();
        dA.m[e] = 1.0;  // unit perturbation of entry e
        const M3 analytic = PolarDR(A, dA);

        M3 Ap = A, Am = A;
        Ap.m[e] += h;
        Am.m[e] -= h;
        const M3 Rp = PolarR(Ap);
        const M3 Rm = PolarR(Am);
        for (int k = 0; k < 9; ++k) {
            const double fd = (Rp.m[k] - Rm.m[k]) / (2.0 * h);
            const double denom = std::fabs(analytic.m[k]) + std::fabs(fd) + 1.0e-9;
            const double rel = std::fabs(analytic.m[k] - fd) / denom;
            max_rel = std::max(max_rel, rel);
        }
    }
    EXPECT_LT(max_rel, 1.0e-3)
        << "polar dR/dA analytic vs FD max rel-err " << max_rel;
}

namespace {

// Pairwise distance matrix flattened (i<j) for an n-particle config.
std::vector<float> PairwiseDistances(const std::vector<Vec3>& p) {
    std::vector<float> d;
    for (std::size_t i = 0; i < p.size(); ++i)
        for (std::size_t j = i + 1; j < p.size(); ++j)
            d.push_back((p[i] - p[j]).Length());
    return d;
}

// An 8-particle rest cube cluster (unit cube corners, slightly perturbed off the
// regular grid so the covariance is full-rank and well-conditioned).
std::vector<Vec3> RestCubeCluster() {
    return {
        Vec3{0.0f, 0.0f, 0.0f}, Vec3{1.0f, 0.0f, 0.05f},
        Vec3{0.0f, 1.0f, 0.0f}, Vec3{1.0f, 1.0f, 0.0f},
        Vec3{0.05f, 0.0f, 1.0f}, Vec3{1.0f, 0.0f, 1.0f},
        Vec3{0.0f, 1.0f, 1.0f}, Vec3{1.0f, 1.0f, 1.0f}};
}

} // namespace

// Gate 2 (DISCRIMINATING relaxation) + Gate 3 (D1): a cluster initialized
// NON-rigidly (anisotropic stretch + shear) relaxes under the shape-match
// constraint ALONE toward a rigid transform of the rest shape (rest pairwise
// distances recovered), and is two-run byte-exact.
TEST(XpbdShapeMatch, NonRigidClusterRelaxesToRigidShapeAndIsByteExact) {
    const std::vector<Vec3> rest = RestCubeCluster();
    const std::vector<float> rest_d = PairwiseDistances(rest);

    // A KNOWN reference rotation R0: angle theta0 about the tilted unit axis
    // a = (1,1,1)/sqrt(3) (Rodrigues). R0 is composed with the (sheared) stretch to
    // build a non-rigid initial deformation; it is the rigid PART of that
    // deformation, NOT the rotation the device recovers (the shear contributes its
    // own rotational component -- see the discrimination block below). The
    // discrimination matters because the device PolarRotation (float, in
    // nk soft ops) is a SEPARATE transcription from the host PolarR (double, in
    // this file): a relaxation that only checked rest pairwise distances would pass
    // even if the device polar returned IDENTITY (a pure rest-translation also
    // preserves pairwise distances). The block below recovers the rotation from the
    // relaxed config and asserts it equals polar(A_init) -- the rotation the device
    // MUST compute -- which forces the device polar to be correct.
    const float theta0 = 0.5f;  // ~28.6 deg
    const float ax = 1.0f / std::sqrt(3.0f);
    auto rotateR0 = [&](const Vec3& v) {
        const Vec3 k{ax, ax, ax};
        const float ct = std::cos(theta0), st = std::sin(theta0);
        // Rodrigues: v*ct + (k x v)*st + k*(k.v)*(1-ct).
        return v * ct + k.Cross(v) * st + k * (k.Dot(v) * (1.0f - ct));
    };

    // Deform NON-rigidly THEN apply the known rotation R0: anisotropic stretch
    // (sx,sy,sz) + shear (the symmetric part the shape-match removes), composed
    // with R0 (the rigid part it must recover). NOT a rigid transform overall, so
    // it does NOT already match rest pairwise distances.
    auto deform = [&](const Vec3& q) {
        const float sx = 1.6f, sy = 0.7f, sz = 1.25f, shear = 0.35f;
        const Vec3 stretched{sx * q.x + shear * q.z, sy * q.y, sz * q.z};
        return rotateR0(stretched);
    };

    if (GetNkCtx().backend == nullptr) GTEST_SKIP() << "no CUDA backend";

    // The initial (deformed) positions -- the soft particle rest state seeded into
    // the nk Model. Used both for the setup-sanity check and for the per-run cook.
    std::vector<Vec3> init_positions;
    for (const Vec3& q : rest) init_positions.push_back(deform(q));

    // Build an nk::Model carrying ONE shape-match cluster over the deformed cube
    // (the CSR cook: c0 + q_i are computed inside CookShapeMatch via the Model
    // staging). The cook here is inline (the nk_particle_equivalence pattern) so
    // the test exercises the SAME ModelParticles fields the cook produces.
    auto build_model = [&]() {
        nk::Model model;
        nk::Model::ModelParticles& mp = model.particles;
        mp.mode = nk::Model::ParticleMode::Xpbd;
        mp.initial_pos = init_positions;
        mp.initial_vel.assign(rest.size(), Vec3::Zero());
        mp.inv_mass.assign(rest.size(), 1.0f);
        mp.xpbd_iters = 30u;  // within-step re-projection converges.
        // ONE cluster over all 8 corners. Cook c0 = (sum m_i x_i^0)/sum m_i and
        // q_i = x_i^0 - c0 (the legacy soft-upload shape-match flatten 1:1).
        const float n = static_cast<float>(rest.size());
        Vec3 c0{0, 0, 0};
        for (const Vec3& q : rest) c0 += q;
        c0 /= n;
        mp.sm_cluster_offset = {0u};
        mp.sm_cluster_size   = {static_cast<uint32_t>(rest.size())};
        mp.sm_stiffness      = {1.0f};  // s=1: goal config has rest pairwise dists.
        mp.sm_rest_centroid  = {c0};
        for (uint32_t i = 0; i < rest.size(); ++i) {
            mp.sm_particles.push_back(i);
            mp.sm_rest_q.push_back(rest[i] - c0);
            mp.sm_mass.push_back(1.0f);
        }
        nk::ModelCapacities& cap = model.capacities;
        cap.env_count = 1;
        cap.particles_per_env = static_cast<uint32_t>(rest.size());
        cap.shape_match_slots_per_env = 1u;
        cap.shape_match_members_per_env = static_cast<uint32_t>(rest.size());
        return model;
    };

    nk::Pipeline::SolverConfig cfg;
    cfg.dt = 1.0f / 240.0f;
    cfg.gravity[0] = cfg.gravity[1] = cfg.gravity[2] = 0.0f;  // isolate shape-match.
    NkCtx c = GetNkCtx();
    auto run = [&]() {
        nk::World world(build_model(), 1u, c.dev, c.backend, cfg);
        EXPECT_TRUE(world.Ready());
        const uint32_t P = world.GetModel().capacities.particles_per_env;
        for (uint32_t s = 0; s < 60u; ++s) world.Step();
        std::vector<Vec3> p(P), v(P);
        world.GetData().DownloadField(nk::FieldId::ParticlePos, p.data(),
                                      P * sizeof(Vec3));
        world.GetData().DownloadField(nk::FieldId::ParticleVel, v.data(),
                                      P * sizeof(Vec3));
        XpbdState st;
        st.positions = p;
        st.velocities = v;
        return st;
    };

    // Setup sanity: the sheared init must NOT already match rest distances (else
    // the relaxation assertion would be vacuous).
    const std::vector<float> init_d = PairwiseDistances(init_positions);
    float init_max_drift = 0.0f;
    for (std::size_t k = 0; k < rest_d.size(); ++k)
        init_max_drift = std::max(init_max_drift, std::fabs(init_d[k] - rest_d[k]));
    ASSERT_GT(init_max_drift, 0.1f)
        << "setup sanity: sheared init must NOT already match rest distances";

    const XpbdState sa = run();
    for (const Vec3& p : sa.positions) {
        ASSERT_TRUE(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z));
    }

    // Final pairwise distances must return to the REST pairwise distances: the
    // cluster relaxed to a RIGID transform of the rest shape.
    const std::vector<float> final_d = PairwiseDistances(sa.positions);
    float max_drift = 0.0f;
    for (std::size_t k = 0; k < rest_d.size(); ++k)
        max_drift = std::max(max_drift, std::fabs(final_d[k] - rest_d[k]));
    EXPECT_LT(max_drift, 1.0e-2f)
        << "cluster did not relax to a rigid transform of rest: max pairwise "
           "distance drift " << max_drift;

    // DISCRIMINATING rotation check: the rotation the device shape-match converges
    // to is NOT the construction rotation R0. With stiffness=1 the fixed point sets
    // p_i = c + R_dev*q_i, so A_final = R_dev*Q (Q = sum m_i q_i q_i^T, symmetric
    // PD) and polar(A_final) = R_dev. But the rotation the cluster MUST settle on is
    // polar(A_init), where A_init = sum (p_i^init - c_init) q_i^T is the covariance
    // of the INITIAL deformed positions: since p_i^init = R0*M*q_i (+centroid) with
    // M the (non-symmetric, sheared) stretch, A_init = R0*M*Q and
    //     polar(A_init) = R0 * polar(M*Q) != R0
    // because the SHEAR in M carries a rotational component (~7 deg here) that the
    // construction rotation R0 omits. So the correct reference is R_expected =
    // polar(A_init), and the device's recovered R (= polar(A_final)) must equal it.
    // This is STRICTLY STRONGER than comparing to R0: it would catch a device polar
    // that returned identity, R0, or any other wrong rotation -- the anti-identity
    // guard below keeps it non-vacuous. (Comparing to R0, as a prior attempt did,
    // was mathematically inconsistent: it would falsely PASS a device that ignored
    // the shear's rotational part and returned R0.)
    {
        Vec3 c0{0, 0, 0}, c{0, 0, 0};
        for (const Vec3& q : rest) c0 += q;
        c0 /= static_cast<float>(rest.size());
        for (const Vec3& p : sa.positions) c += p;
        c /= static_cast<float>(sa.positions.size());
        M3 A = Zero3();
        for (std::size_t i = 0; i < rest.size(); ++i) {
            const Vec3 d = sa.positions[i] - c;
            const Vec3 q = rest[i] - c0;
            A.m[0] += d.x * q.x; A.m[1] += d.x * q.y; A.m[2] += d.x * q.z;
            A.m[3] += d.y * q.x; A.m[4] += d.y * q.y; A.m[5] += d.y * q.z;
            A.m[6] += d.z * q.x; A.m[7] += d.z * q.y; A.m[8] += d.z * q.z;
        }
        const M3 R = PolarR(A);
        EXPECT_NEAR(Det3(R), 1.0, 1.0e-2)
            << "recovered shape-match rotation must be proper (det +1)";

        // R_expected = polar(A_init): covariance of the INITIAL deformed positions
        // (built from the SAME deform() the world was seeded with) against the rest
        // offsets. This is the rotation the device fixed point provably converges to.
        Vec3 ci{0, 0, 0};
        for (const Vec3& p : init_positions) ci += p;
        ci /= static_cast<float>(init_positions.size());
        M3 Ai = Zero3();
        for (std::size_t i = 0; i < rest.size(); ++i) {
            const Vec3 d = init_positions[i] - ci;
            const Vec3 q = rest[i] - c0;
            Ai.m[0] += d.x * q.x; Ai.m[1] += d.x * q.y; Ai.m[2] += d.x * q.z;
            Ai.m[3] += d.y * q.x; Ai.m[4] += d.y * q.y; Ai.m[5] += d.y * q.z;
            Ai.m[6] += d.z * q.x; Ai.m[7] += d.z * q.y; Ai.m[8] += d.z * q.z;
        }
        const M3 R_expected = PolarR(Ai);
        EXPECT_GT(Det3(Ai), 0.0)
            << "initial covariance must be non-inverted (no det-flip case)";

        // Anti-vacuity guard: R_expected must be FAR from identity, else the check
        // could pass on a device polar that returned identity.
        double exp_from_ident = 0.0;
        for (int k = 0; k < 9; ++k) {
            const double idk = (k == 0 || k == 4 || k == 8) ? 1.0 : 0.0;
            exp_from_ident = std::max(exp_from_ident, std::fabs(R_expected.m[k] - idk));
        }
        ASSERT_GT(exp_from_ident, 0.1)
            << "test sanity: expected rotation must differ from identity";

        // The recovered (final-config) rotation must equal polar(A_init). A broken /
        // identity / wrong device polar fails here.
        double max_dr = 0.0;
        for (int k = 0; k < 9; ++k)
            max_dr = std::max(max_dr, std::fabs(R.m[k] - R_expected.m[k]));
        EXPECT_LT(max_dr, 1.0e-2)
            << "recovered rotation does not match polar(A_init) (device polar may "
               "be wrong / identity): max element drift " << max_dr;
    }

    // D1 two-run byte-exact.
    const XpbdState sb = run();
    EXPECT_EQ(std::memcmp(sa.positions.data(), sb.positions.data(),
                          sa.positions.size() * sizeof(Vec3)),
              0)
        << "shape-match forward position buffer not two-run byte-identical (D1)";
    EXPECT_EQ(std::memcmp(sa.velocities.data(), sb.velocities.data(),
                          sa.velocities.size() * sizeof(Vec3)),
              0)
        << "shape-match forward velocity buffer not two-run byte-identical (D1)";
}
