#pragma once
// ---------------------------------------------------------------------------
// nuka::diffsim -- reverse-mode (adjoint) spatial-algebra primitives
// ---------------------------------------------------------------------------
//
// Each forward spatial op in math/cuda_spatial_ops.cuh is, for fixed matrix
// operands, a LINEAR map of its float inputs; its reverse-mode adjoint is the
// transpose of that map. These helpers implement exactly those transposes,
// reusing the SAME left-to-right `value +=` accumulation order as the forward
// (the determinism contract: no reassociation, no FMA insertion).
//
// Convention: a forward op  out = f(in)  has adjoint  grad_in += (df/din)^T * grad_out.
// "grad" buffers are the cotangents (dL/d<value>). We ADD into grad_in (multiple
// forward uses of the same value accumulate) -- the single-thread-per-articulation
// reverse makes that plain `+=` (no atomics, fixed order, D1).
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "math/cuda_vec_ops.cuh"

#include <cuda_runtime.h>

namespace nuka::diffsim {

// --- 3-vector cross-product adjoints ----------------------------------------
// For c = a x b:  grad_a += b x g ;  grad_b += g x a   (g = dL/dc).
__forceinline__ __device__ nuka::math::Vec3 CrossR(nuka::math::Vec3 a,
                                                   nuka::math::Vec3 b) {
    return ::nuka::math::gpu::Cross(a, b);
}

__forceinline__ __device__ void Zero6Rev(float* v) {
    for (unsigned int i = 0u; i < 6u; ++i) v[i] = 0.0f;
}
__forceinline__ __device__ void Zero36Rev(float* m) {
    for (unsigned int i = 0u; i < 36u; ++i) m[i] = 0.0f;
}

// Forward: out[i] = a[i]*b[i] elementwise? No -- generic accumulation helpers.

// --- Mat66MulVec6 -----------------------------------------------------------
// Forward: out = M * vec       (out[r] = sum_c M[r*6+c]*vec[c]).
// Adjoint w.r.t. vec: grad_vec += M^T * grad_out  -> grad_vec[c] += sum_r M[r*6+c]*g[r].
// Adjoint w.r.t. M  : grad_M[r*6+c] += g[r]*vec[c]  (outer product).
__forceinline__ __device__ void Mat66MulVec6_RevVec(const float* M,
                                                    const float* grad_out,
                                                    float* grad_vec_inout) {
    for (unsigned int c = 0u; c < 6u; ++c) {
        float s = 0.0f;
        for (unsigned int r = 0u; r < 6u; ++r) {
            s += M[r * 6u + c] * grad_out[r];
        }
        grad_vec_inout[c] += s;
    }
}
__forceinline__ __device__ void Mat66MulVec6_RevMat(const float* grad_out,
                                                    const float* vec,
                                                    float* grad_M_inout) {
    for (unsigned int r = 0u; r < 6u; ++r) {
        for (unsigned int c = 0u; c < 6u; ++c) {
            grad_M_inout[r * 6u + c] += grad_out[r] * vec[c];
        }
    }
}

// --- Dot6 -------------------------------------------------------------------
// Forward: s = sum_i a[i]*b[i].  Adjoint: grad_a[i] += g*b[i]; grad_b[i] += g*a[i].
__forceinline__ __device__ void Dot6_Rev(const float* a, const float* b, float g,
                                         float* grad_a_inout, float* grad_b_inout) {
    for (unsigned int i = 0u; i < 6u; ++i) {
        grad_a_inout[i] += g * b[i];
        grad_b_inout[i] += g * a[i];
    }
}

// --- TransformMotion --------------------------------------------------------
// Forward: out = X * in.  Adjoint(in): grad_in += X^T * grad_out (== Mat66MulVec6
// with X^T). Adjoint(X): grad_X[r*6+c] += grad_out[r]*in[c].
__forceinline__ __device__ void TransformMotion_RevIn(const float* X,
                                                      const float* grad_out,
                                                      float* grad_in_inout) {
    for (unsigned int c = 0u; c < 6u; ++c) {
        float s = 0.0f;
        for (unsigned int r = 0u; r < 6u; ++r) {
            s += X[r * 6u + c] * grad_out[r];
        }
        grad_in_inout[c] += s;
    }
}
__forceinline__ __device__ void TransformMotion_RevX(const float* grad_out,
                                                     const float* in,
                                                     float* grad_X_inout) {
    for (unsigned int r = 0u; r < 6u; ++r) {
        for (unsigned int c = 0u; c < 6u; ++c) {
            grad_X_inout[r * 6u + c] += grad_out[r] * in[c];
        }
    }
}

// --- TransformForceTranspose ------------------------------------------------
// Forward: out = X^T * in   (out[r] = sum_c X[c*6+r]*in[c]).
// Adjoint(in): grad_in[c] += sum_r X[c*6+r]*grad_out[r]  == X * grad_out.
// Adjoint(X) : grad_X[c*6+r] += in[c]*grad_out[r].
__forceinline__ __device__ void TransformForceTranspose_RevIn(const float* X,
                                                              const float* grad_out,
                                                              float* grad_in_inout) {
    for (unsigned int c = 0u; c < 6u; ++c) {
        float s = 0.0f;
        for (unsigned int r = 0u; r < 6u; ++r) {
            s += X[c * 6u + r] * grad_out[r];
        }
        grad_in_inout[c] += s;
    }
}
__forceinline__ __device__ void TransformForceTranspose_RevX(const float* in,
                                                             const float* grad_out,
                                                             float* grad_X_inout) {
    for (unsigned int c = 0u; c < 6u; ++c) {
        for (unsigned int r = 0u; r < 6u; ++r) {
            grad_X_inout[c * 6u + r] += in[c] * grad_out[r];
        }
    }
}

// --- TransformInertiaToParent ----------------------------------------------
// Forward: P = X^T * C * X    via  temp = C * X ; P = X^T * temp.
//   temp[r*6+c] = sum_k C[r*6+k] * X[k*6+c]
//   P[r*6+c]    = sum_k X[k*6+r] * temp[k*6+c]
// Adjoint w.r.t. C (X held fixed):
//   grad_temp from P-step: grad_temp[k*6+c] += sum_r X[k*6+r]*grad_P[r*6+c]
//                          (P[r,c] uses X[k,r]*temp[k,c]; d/d temp[k,c] = X[k,r])
//   grad_C from temp-step : grad_C[r*6+k] += sum_c grad_temp[r*6+c]*X[k*6+c]
// We only need the C-adjoint (X is held fixed for the tau/qdot/mass spine); the
// X-adjoint is added by the q-channel pass below (RevX variant).
__forceinline__ __device__ void TransformInertiaToParent_RevC(const float* X,
                                                              const float* grad_P,
                                                              float* grad_C_inout) {
    float grad_temp[36];
    for (unsigned int i = 0u; i < 36u; ++i) grad_temp[i] = 0.0f;
    // grad_temp[k*6+c] = sum_r X[k*6+r] * grad_P[r*6+c]
    for (unsigned int k = 0u; k < 6u; ++k) {
        for (unsigned int c = 0u; c < 6u; ++c) {
            float s = 0.0f;
            for (unsigned int r = 0u; r < 6u; ++r) {
                s += X[k * 6u + r] * grad_P[r * 6u + c];
            }
            grad_temp[k * 6u + c] += s;
        }
    }
    // grad_C[r*6+k] += sum_c grad_temp[r*6+c] * X[k*6+c]
    for (unsigned int r = 0u; r < 6u; ++r) {
        for (unsigned int k = 0u; k < 6u; ++k) {
            float s = 0.0f;
            for (unsigned int c = 0u; c < 6u; ++c) {
                s += grad_temp[r * 6u + c] * X[k * 6u + c];
            }
            grad_C_inout[r * 6u + k] += s;
        }
    }
}

// Full X-adjoint of P = X^T C X (C held at its forward value): needed only for
// the q-channel. dP/dX has two contributions (temp = C X, P = X^T temp):
//   from temp-step: grad_X[k*6+c] += sum_r C[r*6+k]^... handled by recomputing
//   We compute grad_X by the standard two-factor product rule:
//     grad_X (via temp = C*X)  : grad_X[k*6+c] += sum_r C[r*6+k]*grad_temp[r*6+c]
//     grad_X (via P = X^T*temp): grad_X[k*6+r] += sum_c grad_P[r*6+c]*temp[k*6+c]
// where grad_temp[k*6+c] = sum_r X[k*6+r]*grad_P[r*6+c] (same as RevC's grad_temp)
// and temp[k*6+c] = sum_j C[k*6+j]*X[j*6+c] (forward temp).
__forceinline__ __device__ void TransformInertiaToParent_RevX(const float* X,
                                                              const float* C,
                                                              const float* grad_P,
                                                              float* grad_X_inout) {
    float grad_temp[36];
    float temp[36];
    for (unsigned int k = 0u; k < 6u; ++k) {
        for (unsigned int c = 0u; c < 6u; ++c) {
            float gt = 0.0f;
            float t = 0.0f;
            for (unsigned int r = 0u; r < 6u; ++r) {
                gt += X[k * 6u + r] * grad_P[r * 6u + c];
            }
            for (unsigned int j = 0u; j < 6u; ++j) {
                t += C[k * 6u + j] * X[j * 6u + c];
            }
            grad_temp[k * 6u + c] = gt;
            temp[k * 6u + c] = t;
        }
    }
    // via P = X^T * temp:  grad_X[k*6+r] += sum_c grad_P[r*6+c]*temp[k*6+c]
    for (unsigned int k = 0u; k < 6u; ++k) {
        for (unsigned int r = 0u; r < 6u; ++r) {
            float s = 0.0f;
            for (unsigned int c = 0u; c < 6u; ++c) {
                s += grad_P[r * 6u + c] * temp[k * 6u + c];
            }
            grad_X_inout[k * 6u + r] += s;
        }
    }
    // via temp = C * X:    grad_X[j*6+c] += sum_k C[k*6+j]*grad_temp[k*6+c]
    for (unsigned int j = 0u; j < 6u; ++j) {
        for (unsigned int c = 0u; c < 6u; ++c) {
            float s = 0.0f;
            for (unsigned int k = 0u; k < 6u; ++k) {
                s += C[k * 6u + j] * grad_temp[k * 6u + c];
            }
            grad_X_inout[j * 6u + c] += s;
        }
    }
}

// --- SubtractOuterProduct36 -------------------------------------------------
// Forward: M[r,c] -= u[r]*u[c]*inv_d,  inv_d = 1/max(diagonal,eps).  (M aliases
// the reduced_inertia; here "in" is the pre-subtract value, "out" the result.)
// We treat M_out = M_in - (u u^T) inv_d, so for the reverse:
//   grad_M_in[r,c] += grad_M_out[r,c]                         (identity passthrough)
//   grad_u[r]      += -inv_d * sum_c (grad_M_out[r,c]*u[c] + grad_M_out[c,r]*u[c])
//   grad_inv_d     += -sum_{r,c} grad_M_out[r,c]*u[r]*u[c]
// (inv_d depends on `diagonal`; when diagonal>=eps, d(inv_d)/d(diagonal)=-inv_d^2.)
// grad_M_in passthrough is handled by the caller reusing the same grad buffer.
__forceinline__ __device__ void SubtractOuterProduct36_RevU(const float* grad_M_out,
                                                            const float* u,
                                                            float inv_d,
                                                            float* grad_u_inout) {
    for (unsigned int r = 0u; r < 6u; ++r) {
        float s = 0.0f;
        for (unsigned int c = 0u; c < 6u; ++c) {
            // symmetric: contribution to u[r] from (r,c) and (c,r)
            s += grad_M_out[r * 6u + c] * u[c] + grad_M_out[c * 6u + r] * u[c];
        }
        grad_u_inout[r] += -inv_d * s;
    }
}
__forceinline__ __device__ float SubtractOuterProduct36_RevInvD(const float* grad_M_out,
                                                               const float* u) {
    float s = 0.0f;
    for (unsigned int r = 0u; r < 6u; ++r) {
        for (unsigned int c = 0u; c < 6u; ++c) {
            s += grad_M_out[r * 6u + c] * u[r] * u[c];
        }
    }
    return -s;
}

// --- MotionCross adjoint ----------------------------------------------------
// Forward: out = lhs (motion) x* rhs (motion).
//   lw=lhs[0:3] lv=lhs[3:6] rw=rhs[0:3] rv=rhs[3:6]
//   angular = lw x rw ; linear = lw x rv + lv x rw
// g = dL/d out = [ga(0:3), gl(3:6)].
//   grad_lw = rw x ga + rv x gl ; grad_lv = rw x gl
//   grad_rw = ga x lw + gl x lv ; grad_rv = gl x lw
__forceinline__ __device__ void MotionCross_Rev(const float* lhs, const float* rhs,
                                                const float* g,
                                                float* grad_lhs_inout,
                                                float* grad_rhs_inout) {
    using nuka::math::Vec3;
    const Vec3 lw{lhs[0], lhs[1], lhs[2]}, lv{lhs[3], lhs[4], lhs[5]};
    const Vec3 rw{rhs[0], rhs[1], rhs[2]}, rv{rhs[3], rhs[4], rhs[5]};
    const Vec3 ga{g[0], g[1], g[2]}, gl{g[3], g[4], g[5]};
    const Vec3 g_lw = ::nuka::math::gpu::Add(CrossR(rw, ga), CrossR(rv, gl));
    const Vec3 g_lv = CrossR(rw, gl);
    const Vec3 g_rw = ::nuka::math::gpu::Add(CrossR(ga, lw), CrossR(gl, lv));
    const Vec3 g_rv = CrossR(gl, lw);
    grad_lhs_inout[0] += g_lw.x; grad_lhs_inout[1] += g_lw.y; grad_lhs_inout[2] += g_lw.z;
    grad_lhs_inout[3] += g_lv.x; grad_lhs_inout[4] += g_lv.y; grad_lhs_inout[5] += g_lv.z;
    grad_rhs_inout[0] += g_rw.x; grad_rhs_inout[1] += g_rw.y; grad_rhs_inout[2] += g_rw.z;
    grad_rhs_inout[3] += g_rv.x; grad_rhs_inout[4] += g_rv.y; grad_rhs_inout[5] += g_rv.z;
}

// --- ForceCross adjoint -----------------------------------------------------
// Forward: out = motion x force.
//   w=motion[0:3] v=motion[3:6] n=force[0:3] f=force[3:6]
//   angular = w x n + v x f ; linear = w x f
// g = [ga(0:3), gl(3:6)].
//   grad_w = n x ga + f x gl ; grad_v = f x ga
//   grad_n = ga x w          ; grad_f = ga x v + gl x w
__forceinline__ __device__ void ForceCross_Rev(const float* motion, const float* force,
                                               const float* g,
                                               float* grad_motion_inout,
                                               float* grad_force_inout) {
    using nuka::math::Vec3;
    const Vec3 w{motion[0], motion[1], motion[2]}, v{motion[3], motion[4], motion[5]};
    const Vec3 n{force[0], force[1], force[2]}, f{force[3], force[4], force[5]};
    const Vec3 ga{g[0], g[1], g[2]}, gl{g[3], g[4], g[5]};
    const Vec3 g_w = ::nuka::math::gpu::Add(CrossR(n, ga), CrossR(f, gl));
    const Vec3 g_v = CrossR(f, ga);
    const Vec3 g_n = CrossR(ga, w);
    const Vec3 g_f = ::nuka::math::gpu::Add(CrossR(ga, v), CrossR(gl, w));
    grad_motion_inout[0] += g_w.x; grad_motion_inout[1] += g_w.y; grad_motion_inout[2] += g_w.z;
    grad_motion_inout[3] += g_v.x; grad_motion_inout[4] += g_v.y; grad_motion_inout[5] += g_v.z;
    grad_force_inout[0] += g_n.x; grad_force_inout[1] += g_n.y; grad_force_inout[2] += g_n.z;
    grad_force_inout[3] += g_f.x; grad_force_inout[4] += g_f.y; grad_force_inout[5] += g_f.z;
}

}  // namespace nuka::diffsim
