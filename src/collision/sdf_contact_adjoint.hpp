#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::sdf_contact_adjoint -- the ENVELOPE-THEOREM contact-geometry
//                  adjoint of the Newton-summed-SDF contact (v0.7 p08-C, Ch2/Ch3)
// ---------------------------------------------------------------------------
//
// THE SINGULARITY (why this design, not the naive IFT). The summed-SDF minimum
// f(p) = phi_a(p) + phi_b(p) is a FLAT VALLEY over the convex-convex overlap: at
// the witness p*, d f / d p ~= 0 across the whole overlap, so the witness POINT
// is NON-UNIQUE and the naive point-IFT dr/dp is RANK-DEFICIENT/singular along
// the valley. DEPTH and NORMAL are well-defined; the POINT is not. (The forward's
// `step_curvature` = 2/voxel is a forward descent-step scalar that does NOT model
// the true along-valley curvature, which is ~0 -- feeding it into any inversion
// would manufacture a FICTITIOUS point gradient. This file NEVER uses it.)
//
// THE ENVELOPE THEOREM (valley-proof). Because d f / d p ~= 0 at p*, the depth
//      V(theta) = (phi_a + phi_b)(p*(theta), theta)
// has   dV/dtheta = (partial f / partial theta)|_{p* FROZEN}  +  (df/dp)|_{p*} . (dp*/dtheta)
//                 = (partial f / partial theta)|_{p* FROZEN}   (the second term
//                   drops: df/dp ~= 0 multiplies the ill-defined dp*/dtheta).
// So the depth gradient is the EXPLICIT partial of the summed field w.r.t. the
// parameters theta (SDF cell values; body pose) at the FROZEN witness p*. We do
// NOT differentiate the descent iteration. The normal n = normalize(grad_b -
// grad_a)|_{p*} is likewise evaluated at the frozen p*: its valley-tangent
// component multiplies the ~0 along-valley curvature, so it is stable.
//
// THE CONTACT POINT IS DEGENERATE (Channel 3) -> NOT differentiated. The witness
// is non-unique along the valley, so there is NO ground-truth difference quotient
// for a point gradient; any "passing" point-gradient FD would be validating
// against solver noise (the residual ~5e-5 descent floor). We therefore RESTRICT
// the differentiable contact outputs to DEPTH + NORMAL and document the point's
// degeneracy. The envelope theorem is the PRINCIPLE behind this restriction (the
// point drops out of the depth/normal gradients precisely because df/dp ~= 0), not
// a punt. This header provides NO contact-point gradient. (If a downstream path
// ever needs one, SURFACE it -- do not fabricate it from step_curvature.)
//
// PARAMETERS theta differentiated:
//   (a) SDF CELL VALUES the trilinear sampler interpolated at p* (depth channel).
//       partial phi_a / partial cell_value[corner] = w[corner] (the trilinear
//       weight). Since cell VALUES and cell GRADIENTS are stored independently in
//       the cooked SDF, perturbing a cell value does NOT move p* -> the envelope
//       error for this channel is EXACTLY 0 (machine-precision FD).
//   (b) NORMAL n = normalize(grad_b - grad_a)|_{p*}. SHIPPED as the witness-FREE
//       normalize-Jacobian J_norm(v) = (I - n n^T)/|grad_b - grad_a| (dn for a
//       perturbation of the un-normalized diff). The upstream theta-channels
//       (cell-gradient, pose) are DEGENERATE: at convergence grad_a = -grad_b, so
//       n is PINNED to the gradient field at the witness, and ANY theta that
//       changes a gradient MOVES the witness first-order -- the witness-motion term
//       is the SAME ORDER as the frozen partial and cannot be dropped (the depth's
//       envelope protection does NOT extend to the normal, since n is not the
//       stationary objective f). EMPIRICALLY FALSIFIED + demonstrated in the test;
//       NO normal-theta gradient is shipped (deferred: reduced-space witness-motion
//       IFT). This contradicts the task's "differentiate the normal at frozen p*"
//       assumption -- surfaced, not fabricated.
//   (c) BODY POSE (translation; rotation). depth-vs-translation:
//       partial phi_a / partial t_a = grad_a_local . partial(local)/partial t_a
//       = grad_a_local . (-R_a^T) = -grad_a_world, so partial depth / partial t_a
//       = +grad_a_world (depth = -(phi_a+phi_b)). normal-vs-pose needs the FIELD
//       HESSIAN H_local = d(interpolated grad)/d(local p) (even at FROZEN world
//       p*, moving the body shifts the LOCAL sample point), provided by
//       sample_sdf_with_hessian below.
//
// HOST. CUDA-free host header; uses the SAME p07 trilinear sampler internals.
// fp32 to match the forward (the gradients are validated against fp64 FD with the
// envelope O(residual) error confirmed within tolerance, see test).
// ---------------------------------------------------------------------------

#include "collision/sdf_contact.hpp"
#include "math/vec3.hpp"
#include "runtime/sdf/sparse_sdf_query.cuh"

#include <cstdint>

namespace nuka::collision {

// 3x3 matrix (row-major) for the field Hessian + normalize Jacobian. Host-only.
struct Mat3 {
    float m[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};  // row-major: m[r*3+c]
    static Mat3 Zero() { return Mat3{}; }
    float& at(int r, int c) { return m[r * 3 + c]; }
    float at(int r, int c) const { return m[r * 3 + c]; }
    math::Vec3 Mul(const math::Vec3& v) const {
        return {m[0] * v.x + m[1] * v.y + m[2] * v.z, m[3] * v.x + m[4] * v.y + m[5] * v.z,
                m[6] * v.x + m[7] * v.y + m[8] * v.z};
    }
};

// Extended sampler that ALSO returns the local field Hessian H_local =
// d(interpolated gradient)/d(local p). The trilinear gradient blend is
//   g_local(p) = sum_corner w[corner](p) * cell_grad[corner],
// so H_local[a][b] = d g_local[a] / d p_b = sum_corner (dw[corner]/dp_b) *
// cell_grad[corner][a], where dw/dp carries the +/-(1/h) weight derivatives.
// This is the SPATIAL derivative of the (already-interpolated) cell gradients --
// distinct from the cooked second-derivative of phi; it is what the normal-vs-pose
// envelope partial needs. Returns kOutsideBand (and leaves H at 0) out of band.
// Fixed-order accumulation (D1); host-only (the adjoint runs on the host oracle).
float sample_sdf_with_hessian(const runtime::sdf::SparseSdfDevice& sdf, math::Vec3 p_local,
                              math::Vec3& out_grad_local, Mat3& out_hessian_local);

// Per-corner trilinear weights at p* (in the body's LOCAL frame), in the canonical
// (di,dj,dk) lexicographic order -- the SAME order as sparse_sdf_sample. Used by
// the cell-value depth channel (weight == partial phi / partial cell_value).
struct TrilinearWeights {
    bool in_band = false;
    float w[8] = {0, 0, 0, 0, 0, 0, 0, 0};  // (0,0,0),(0,0,1),(0,1,0),...,(1,1,1)
};
TrilinearWeights trilinear_weights(const runtime::sdf::SparseSdfDevice& sdf, math::Vec3 p_local);

// ---------------------------------------------------------------------------
// CHANNEL 2 outputs. All evaluated at the FROZEN witness p* (envelope theorem).
// ---------------------------------------------------------------------------

// depth = max(-(phi_a+phi_b), 0). When penetrating (the differentiable regime),
// partial depth / partial cell_value_a[corner] = -w_a[corner]  (and similarly for
// b). Returns the 8 per-corner partials for body A and body B at p*.
struct DepthCellValueGrad {
    bool valid = false;
    float d_depth_d_cellvalue_a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    float d_depth_d_cellvalue_b[8] = {0, 0, 0, 0, 0, 0, 0, 0};
};
DepthCellValueGrad depth_grad_wrt_cell_values(const runtime::sdf::SparseSdfDevice& sdf_a,
                                              const SdfBodyFrame& frame_a,
                                              const runtime::sdf::SparseSdfDevice& sdf_b,
                                              const SdfBodyFrame& frame_b,
                                              const SdfContactResult& contact);

// partial depth / partial t_a = +grad_a_world, partial depth / partial t_b =
// +grad_b_world (both at p*; depth = -(phi_a+phi_b), and partial phi/partial t =
// -grad_world). 3-vectors. Valid only when penetrating.
struct DepthPoseGrad {
    bool valid = false;
    math::Vec3 d_depth_d_translation_a{};
    math::Vec3 d_depth_d_translation_b{};
};
DepthPoseGrad depth_grad_wrt_translation(const SdfContactResult& contact);

// normal n = normalize(grad_b_world - grad_a_world)|_{p*}. The normalize Jacobian
// J_norm(v) = (I - n n^T)/|v| maps a perturbation of the un-normalized diff d =
// grad_b_world - grad_a_world to dn. This is the witness-FREE local building block
// (validated directly: normalize(d + eps v) vs J_norm v) -- p11 reuses it to push
// a normal-cotangent back onto whatever produced grad_b - grad_a.
struct NormalNormalizeJacobian {
    bool valid = false;
    Mat3 j_norm{};        // (I - n n^T)/|grad_b - grad_a|
    float inv_len = 0.0f; // 1/|grad_b - grad_a|
};
NormalNormalizeJacobian normal_normalize_jacobian(const SdfContactResult& contact);

// DEGENERATE (Channel-3-adjacent): the normal-vs-CELL-GRADIENT FROZEN partial
// J_norm * R_b * w_b[corner]. EMPIRICALLY this is NOT the cell-gradient theta-
// derivative either (see test). REASON (exact): at the converged witness,
// stationarity gives grad_a(p*) = -grad_b(p*), so the normal n = normalize(grad_b
// - grad_a) = -normalize(grad_a(p*)) is PINNED to the gradient FIELD at the
// witness. ANY theta that changes a gradient also changes the descent direction
// r = grad_a + grad_b and therefore MOVES the witness first-order; the witness-
// motion term (partial n/partial p*).(dp*/dtheta) is the SAME ORDER as this frozen
// term (partial n/partial p* ~ 1/voxel curvature; dp* ~ voxel) and can NEVER be
// dropped. So this frozen partial is shipped ONLY as the demonstration term the
// NormalVsCellGradientIsDegenerate test uses to prove the frozen model diverges
// from the re-converged FD. DO NOT consume it as a theta-gradient. The correct
// normal-theta derivative needs the reduced-space (valley-null-projected) witness-
// motion IFT (using the curved body's REAL curvature -- NOT step_curvature);
// deferred. (The validated, witness-FREE normal building block is
// normal_normalize_jacobian above -- that is the genuine p11 reuse hook.)
struct NormalCellGradGrad {
    bool valid = false;
    // d n / d (cell_grad_b[corner] along world axis), for corner in 0..7, axis 0..2.
    // dn_b[corner][axis] is a Vec3.
    math::Vec3 dn_b[8][3];
};
NormalCellGradGrad normal_grad_wrt_cell_gradient_b(const runtime::sdf::SparseSdfDevice& sdf_b,
                                                   const SdfBodyFrame& frame_b,
                                                   const SdfContactResult& contact);

// ---------------------------------------------------------------------------
// DEGENERATE (Channel-3-adjacent): the normal-vs-POSE derivative. EMPIRICALLY the
// frozen-p* model below is WRONG (see test): translating the curved body makes the
// witness TRACK it ACROSS the valley to a new radial contact, so the TRUE
// d n / d (translation) ~= 0 (a sphere-on-plane normal is invariant to tangential
// translation). This frozen-Hessian term  J_norm * (+R_a H_a R_a^T) (and the b
// analog) is NOT that derivative -- it OMITS the d n / d p* . d p*/d theta term,
// which does NOT drop out for the normal (unlike depth, n is not the stationary
// objective f, so d n / d p* != 0). It is shipped ONLY as the demonstration term
// the NormalVsTranslationIsDegenerate test uses to prove the frozen model diverges
// from the re-converged FD. DO NOT consume this as a normal-pose theta-gradient
// (a reduced-space valley-null-projected IFT using the curved body's REAL
// curvature -- NOT step_curvature -- is the correct future construction; deferred).
struct NormalPoseFrozenTerm {
    bool valid = false;
    Mat3 frozen_d_normal_d_translation_a{};  // J_norm * (+R_a H_a R_a^T) -- NOT the gradient
    Mat3 frozen_d_normal_d_translation_b{};  // J_norm * (-R_b H_b R_b^T) -- NOT the gradient
};
NormalPoseFrozenTerm normal_pose_frozen_term(const runtime::sdf::SparseSdfDevice& sdf_a,
                                             const SdfBodyFrame& frame_a,
                                             const runtime::sdf::SparseSdfDevice& sdf_b,
                                             const SdfBodyFrame& frame_b,
                                             const SdfContactResult& contact);

}  // namespace nuka::collision
