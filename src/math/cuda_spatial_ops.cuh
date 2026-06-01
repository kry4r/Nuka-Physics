#pragma once
// ---------------------------------------------------------------------------
// nuka::math::gpu - shared __device__ spatial (6-vector / 6x6) algebra
//
// Consolidates the Featherstone spatial-algebra helpers that are currently
// duplicated between:
//   src/runtime/articulation/featherstone_aba.cu        (Zero6, Zero36, Dot6,
//       Add6InPlace, Mat66MulVec6, Copy36, TransformMotion, TransformForceTranspose,
//       TransformInertiaToParent, MotionCross, ForceCross, SubtractOuterProduct36)
//   src/runtime/articulation/articulation_contacts.cu   (Dot6Local, Copy36Local,
//       Mat66MulVec6Local, TransformInertiaToParentLocal, TransformForceTransposeLocal)
//
// All operands are flat float arrays in row-major order:
//   * spatial 6-vector  : float[6]  (angular[0..2], linear[3..5])
//   * spatial 6x6 matrix: float[36] (row-major, index = row*6 + col)
//
// DESIGN NOTE (T8b isolation): featherstone_aba.cu passes these the Plücker
// matrix via `const LinkSpatialTransform&` and reads only `transform.X`
// (a float[36]). To avoid any dependency on articulation_state.hpp (owned and
// being edited by T8b), the transform-style functions here take the raw
// `const float* X` directly. The Phase-2 migration at the call site is the
// mechanical substitution `f(transform.X, ...)`; the body is byte-identical.
//
// DETERMINISM CONTRACT: float op order is reproduced exactly (left-to-right
// accumulation in the `value +=` loops). Do not reassociate, do not insert FMA
// intrinsics, do not reorder the reductions. `__forceinline__` is safe (no float
// reorder), but the Phase-2 owner-golden must be re-verified byte-for-byte.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "math/cuda_vec_ops.cuh"  // Cross3 == gpu::Cross, Add == gpu::Add, MakeVec3

#if !defined(__CUDACC__)
#error "math/cuda_spatial_ops.cuh is a CUDA device header; compile with nvcc."
#endif

namespace nuka::math::gpu {

// ---------------------------------------------------------------------------
// Fills / copies
// ---------------------------------------------------------------------------

__forceinline__ __device__ void Zero6(float* value) {
    for (unsigned int i = 0u; i < 6u; ++i) {
        value[i] = 0.0f;
    }
}

__forceinline__ __device__ void Zero36(float* value) {
    for (unsigned int i = 0u; i < 36u; ++i) {
        value[i] = 0.0f;
    }
}

// Copy36 == Copy36Local (byte-identical).
__forceinline__ __device__ void Copy36(const float* __restrict__ src,
                                       float* __restrict__ dst) {
    for (unsigned int i = 0u; i < 36u; ++i) {
        dst[i] = src[i];
    }
}

// ---------------------------------------------------------------------------
// 6-vector arithmetic
// ---------------------------------------------------------------------------

// Dot6 == Dot6Local (left-associated sum, identical).
__forceinline__ __device__ float Dot6(const float* a, const float* b) {
    float sum = 0.0f;
    for (unsigned int i = 0u; i < 6u; ++i) {
        sum += a[i] * b[i];
    }
    return sum;
}

__forceinline__ __device__ void Add6InPlace(float* lhs, const float* rhs) {
    for (unsigned int i = 0u; i < 6u; ++i) {
        lhs[i] += rhs[i];
    }
}

// ---------------------------------------------------------------------------
// 6x6 matrix-vector / outer-product
// ---------------------------------------------------------------------------

// Mat66MulVec6 == Mat66MulVec6Local. out = matrix * vector.
__forceinline__ __device__ void Mat66MulVec6(const float* matrix,
                                             const float* vector,
                                             float* out) {
    for (unsigned int row = 0u; row < 6u; ++row) {
        float value = 0.0f;
        for (unsigned int col = 0u; col < 6u; ++col) {
            value += matrix[row * 6u + col] * vector[col];
        }
        out[row] = value;
    }
}

// matrix -= (u u^T) / max(diagonal, min_diagonal). Featherstone rank-1 downdate.
// featherstone_aba.cu uses kMinDiagonal = 1.0e-6f; parameterized here so the
// constant stays a literal at the call site (bit-identical fmaxf path).
__forceinline__ __device__ void SubtractOuterProduct36(float* matrix,
                                                       const float* u,
                                                       float diagonal,
                                                       float min_diagonal) {
    const float inv_diagonal = 1.0f / fmaxf(diagonal, min_diagonal);
    for (unsigned int row = 0u; row < 6u; ++row) {
        for (unsigned int col = 0u; col < 6u; ++col) {
            matrix[row * 6u + col] -= u[row] * u[col] * inv_diagonal;
        }
    }
}

// ---------------------------------------------------------------------------
// Plücker transform applications (take the raw float[36] X matrix; see header
// note on T8b isolation).
// ---------------------------------------------------------------------------

// out = X * in   (TransformMotion in featherstone_aba.cu).
__forceinline__ __device__ void TransformMotion(const float* X,
                                                const float* in,
                                                float* out) {
    for (unsigned int row = 0u; row < 6u; ++row) {
        float value = 0.0f;
        for (unsigned int col = 0u; col < 6u; ++col) {
            value += X[row * 6u + col] * in[col];
        }
        out[row] = value;
    }
}

// out = X^T * in  (TransformForceTranspose / TransformForceTransposeLocal).
__forceinline__ __device__ void TransformForceTranspose(const float* X,
                                                        const float* in,
                                                        float* out) {
    for (unsigned int row = 0u; row < 6u; ++row) {
        float value = 0.0f;
        for (unsigned int col = 0u; col < 6u; ++col) {
            value += X[col * 6u + row] * in[col];
        }
        out[row] = value;
    }
}

// parent_delta = X^T * child_inertia * X  (TransformInertiaToParent /
// TransformInertiaToParentLocal). Two left-associated 6x6 multiplies via a
// 36-float scratch; order preserved exactly.
__forceinline__ __device__ void TransformInertiaToParent(const float* X,
                                                         const float* child_inertia,
                                                         float* parent_delta) {
    float temp[36];
    for (unsigned int row = 0u; row < 6u; ++row) {
        for (unsigned int col = 0u; col < 6u; ++col) {
            float value = 0.0f;
            for (unsigned int k = 0u; k < 6u; ++k) {
                value += child_inertia[row * 6u + k] * X[k * 6u + col];
            }
            temp[row * 6u + col] = value;
        }
    }
    for (unsigned int row = 0u; row < 6u; ++row) {
        for (unsigned int col = 0u; col < 6u; ++col) {
            float value = 0.0f;
            for (unsigned int k = 0u; k < 6u; ++k) {
                value += X[k * 6u + row] * temp[k * 6u + col];
            }
            parent_delta[row * 6u + col] = value;
        }
    }
}

// ---------------------------------------------------------------------------
// Spatial cross products (Featherstone motion/force cross), float[6] in/out.
// MotionCross / ForceCross in featherstone_aba.cu. Build through Vec3 Cross/Add
// to keep the exact same float op order as the originals.
// ---------------------------------------------------------------------------

// out = lhs (motion) x* rhs (motion).
__forceinline__ __device__ void MotionCross(const float* lhs,
                                            const float* rhs,
                                            float* out) {
    const Vec3 lw = MakeVec3(lhs[0], lhs[1], lhs[2]);
    const Vec3 lv = MakeVec3(lhs[3], lhs[4], lhs[5]);
    const Vec3 rw = MakeVec3(rhs[0], rhs[1], rhs[2]);
    const Vec3 rv = MakeVec3(rhs[3], rhs[4], rhs[5]);
    const Vec3 angular = Cross(lw, rw);
    const Vec3 linear = Add(Cross(lw, rv), Cross(lv, rw));
    out[0] = angular.x;
    out[1] = angular.y;
    out[2] = angular.z;
    out[3] = linear.x;
    out[4] = linear.y;
    out[5] = linear.z;
}

// ---------------------------------------------------------------------------
// Deterministic fixed-pivot (unpivoted) 6x6 LDL^T solve of A x = b, A SPD.
// Identical body in:
//   featherstone_aba.cu     Solve6x6Ldlt
//   src/diffsim/step_backward.cu  Solve6x6LdltRev
// (the two differed only in brace style; bodies are arithmetically identical --
// same kMinDiagonal floor, same fixed loop order -> D1-safe). The `min_diagonal`
// floor is a parameter so the constant stays a literal at each call site (the
// callers both pass kMinDiagonal = 1.0e-6f), bit-identical to the inlined form.
// A is copied into local scratch and factored there; the caller's `matrix`
// buffer is untouched. NOTE the triple/double products (a*a*d, a*y, a*out) are
// single inline expressions here exactly as in the originals -- no stored,
// multiply-used product is introduced, so FMA contraction is preserved.
__forceinline__ __device__ void Solve6x6Ldlt(const float* __restrict__ matrix,
                                             const float* __restrict__ rhs,
                                             float* __restrict__ out,
                                             float min_diagonal) {
    float a[36];
    for (unsigned int i = 0u; i < 36u; ++i) {
        a[i] = matrix[i];
    }
    float d[6];
    for (unsigned int j = 0u; j < 6u; ++j) {
        float djj = a[j * 6u + j];
        for (unsigned int k = 0u; k < j; ++k) {
            djj -= a[j * 6u + k] * a[j * 6u + k] * d[k];
        }
        if (djj < min_diagonal) {
            djj = min_diagonal;  // SPD floor; guards a degenerate config.
        }
        d[j] = djj;
        for (unsigned int i = j + 1u; i < 6u; ++i) {
            float lij = a[i * 6u + j];
            for (unsigned int k = 0u; k < j; ++k) {
                lij -= a[i * 6u + k] * a[j * 6u + k] * d[k];
            }
            a[i * 6u + j] = lij / djj;
        }
    }
    // Forward solve L y = b.
    float y[6];
    for (unsigned int i = 0u; i < 6u; ++i) {
        float value = rhs[i];
        for (unsigned int k = 0u; k < i; ++k) {
            value -= a[i * 6u + k] * y[k];
        }
        y[i] = value;
    }
    // Diagonal solve D z = y.
    for (unsigned int i = 0u; i < 6u; ++i) {
        y[i] /= d[i];
    }
    // Backward solve L^T x = z.
    for (unsigned int ii = 6u; ii > 0u; --ii) {
        const unsigned int i = ii - 1u;
        float value = y[i];
        for (unsigned int k = i + 1u; k < 6u; ++k) {
            value -= a[k * 6u + i] * out[k];
        }
        out[i] = value;
    }
}

// out = motion x force (force cross).
__forceinline__ __device__ void ForceCross(const float* motion,
                                           const float* force,
                                           float* out) {
    const Vec3 w = MakeVec3(motion[0], motion[1], motion[2]);
    const Vec3 v = MakeVec3(motion[3], motion[4], motion[5]);
    const Vec3 n = MakeVec3(force[0], force[1], force[2]);
    const Vec3 f = MakeVec3(force[3], force[4], force[5]);
    const Vec3 angular = Add(Cross(w, n), Cross(v, f));
    const Vec3 linear = Cross(w, f);
    out[0] = angular.x;
    out[1] = angular.y;
    out[2] = angular.z;
    out[3] = linear.x;
    out[4] = linear.y;
    out[5] = linear.z;
}

} // namespace nuka::math::gpu
