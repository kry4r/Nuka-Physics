#pragma once
// ---------------------------------------------------------------------------
// nuka::math::gpu - shared __device__ small-vector / quaternion primitives
//
// This header consolidates the per-TU `__device__` helper functions that are
// currently re-declared inside the anonymous namespace of nearly every CUDA
// physics translation unit (broadphase.cu, contact_generation.cu,
// invariants_gpu.cu, cuda_particle_world.cu, batched_device_world.cu,
// row_solver.cu, cuda_sensors.cu, articulation_jacobian.cu,
// articulation_contacts.cu, featherstone_aba.cu, ...).
//
// DETERMINISM CONTRACT (D1 strong determinism):
//   Every function body below reproduces the EXACT floating-point expression
//   and operation order of the existing duplicated copies. Do NOT reassociate
//   floats, do NOT hand-insert __fmaf_rn / intrinsics, do NOT swap
//   `rsqrtf(x)` for `1.0f/sqrtf(x)` (they are not bit-identical). `__forceinline__`
//   is applied because it does not change float op order; however the Phase-2
//   migration MUST re-verify the owner-golden trajectory byte-for-byte because
//   force-inlining can newly enable cross-call FMA contraction that the
//   separately-compiled `__device__` calls previously prevented.
//
// Why MakeVec3 / MakeQuat exist: math::Vec3 / math::Quat have `constexpr`
// constructors that are effectively __host__ in CUDA. Device code therefore
// builds values either by brace-init of the aggregate or by field assignment.
// Both forms are reproduced verbatim so each call site keeps its original body.
//
// NAMING / VARIANTS: where existing copies differ only in epsilon / fallback
// constants but share the EXACT same float recipe (same sqrt-vs-rsqrt, same op
// order), they are collapsed into one function parameterized by those constants
// (passing a literal compiles to the same FP ops with a different immediate, so
// it is bit-identical to the inlined constant). Where copies differ in the float
// recipe itself (sqrt vs rsqrt, op order), SEPARATE functions are provided.
// See docs/plans/2026-05-30-v03-cuda-math-lib.md for the full divergence table.
// ---------------------------------------------------------------------------

#include "math/vec3.hpp"
#include "math/quat.hpp"

#if !defined(__CUDACC__)
#error "math/cuda_vec_ops.cuh is a CUDA device header; compile the including TU with nvcc."
#endif

namespace nuka::math::gpu {

// ---------------------------------------------------------------------------
// Constructors (device aggregate builders)
// ---------------------------------------------------------------------------

// Brace-init form (featherstone_aba.cu style).
// Marked __host__ __device__: batched_device_world.cu's MakeVec3 is the only
// copy declared __host__ __device__ (the others are __device__-only); making the
// shared symbol HD covers that signature so the batched migration compiles, and
// device-only callers are unaffected. Vec3's constructor is already HD.
__forceinline__ __host__ __device__ Vec3 MakeVec3(float x, float y, float z) {
    return {x, y, z};
}

// Field-assignment form (broadphase / contact_generation / sensors / solver /
// particle / batched style). Bit-identical result to MakeVec3; kept as a named
// alias so a call site that used the field-assignment copy migrates to the same
// symbol. Both copies produce identical SASS. HD for symmetry with MakeVec3 and
// to cover any host-side construction in the batched TU.
__forceinline__ __host__ __device__ Quat MakeQuat(float w, float x, float y, float z) {
    Quat q;
    q.w = w;
    q.x = x;
    q.y = y;
    q.z = z;
    return q;
}

// ---------------------------------------------------------------------------
// Vec3 component-wise arithmetic
// (broadphase / contact_generation / sensors / batched / particle / solver)
// ---------------------------------------------------------------------------

__forceinline__ __device__ Vec3 Add(Vec3 a, Vec3 b) {
    return MakeVec3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__forceinline__ __device__ Vec3 Sub(Vec3 a, Vec3 b) {
    return MakeVec3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__forceinline__ __device__ Vec3 Neg(Vec3 v) {
    return MakeVec3(-v.x, -v.y, -v.z);
}

__forceinline__ __device__ Vec3 Scale(Vec3 v, float s) {
    return MakeVec3(v.x * s, v.y * s, v.z * s);
}

// ---------------------------------------------------------------------------
// Vec3 products
// ---------------------------------------------------------------------------

// Dot == Dot3 across all copies (broadphase, invariants, contact_generation,
// sensors, particle, batched, solver, jacobian, contacts, featherstone).
// Op order: x*x + y*y + z*z (left-associated). IDENTICAL everywhere.
__forceinline__ __device__ float Dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Cross == Cross3 across all copies. Op order per component is fixed:
//   { y*z - z*y, z*x - x*z, x*y - y*x }. IDENTICAL everywhere (both brace-init
// and MakeVec3 copies expand to the same expression).
__forceinline__ __device__ Vec3 Cross(Vec3 a, Vec3 b) {
    return MakeVec3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x);
}

// ---------------------------------------------------------------------------
// Vec3 length helpers
// ---------------------------------------------------------------------------

__forceinline__ __device__ float LengthSq(Vec3 v) {
    return Dot(v, v);
}

// sqrtf(Dot(v,v)) family (contact_generation / sensors / batched / solver).
__forceinline__ __device__ float Length(Vec3 v) {
    return sqrtf(Dot(v, v));
}

// ---------------------------------------------------------------------------
// Named unit vectors (contact_generation / batched copies are identical)
// ---------------------------------------------------------------------------

__forceinline__ __device__ Vec3 ZeroVec3() { return MakeVec3(0.0f, 0.0f, 0.0f); }
__forceinline__ __device__ Vec3 UnitX()    { return MakeVec3(1.0f, 0.0f, 0.0f); }
__forceinline__ __device__ Vec3 UnitY()    { return MakeVec3(0.0f, 1.0f, 0.0f); }
__forceinline__ __device__ Vec3 UnitZ()    { return MakeVec3(0.0f, 0.0f, 1.0f); }

// ---------------------------------------------------------------------------
// Normalization families.
//
// These DIFFER in float recipe and are kept SEPARATE. Do not merge across
// families; only the eps / fallback constants within a family are parameters.
// ---------------------------------------------------------------------------

// Family A: sqrt-then-(1.0f/length)-scale.
//   Used by:  contact_generation.cu Normalize  (eps 1e-6, fallback UnitY)
//             batched_device_world.cu Normalize (eps 1e-8, fallback UnitY)
//             cuda_sensors.cu Normalize         (eps 1e-8, fallback (1,0,0))
// Recipe is bit-identical across the three; only (eps, fallback) differ, so
// they collapse into one parameterized function.
__forceinline__ __device__ Vec3 NormalizeSqrt(Vec3 v, float eps, Vec3 fallback) {
    const float length = Length(v);
    if (length < eps) {
        return fallback;
    }
    return Scale(v, 1.0f / length);
}

// Family B: rsqrtf(Dot)-scale, branch on length_sq, fallback +Z (1e-10).
//   Used by:  articulation_jacobian.cu NormalizeOrUp
//             articulation_contacts.cu NormalizeOrUpLocal
// Body is byte-identical between the two copies (rsqrtf, 1e-10, {0,0,1}).
__forceinline__ __device__ Vec3 NormalizeOrUp(Vec3 normal) {
    const float length_sq = Dot(normal, normal);
    if (length_sq > 1.0e-10f) {
        const float inv_length = rsqrtf(length_sq);
        return {normal.x * inv_length, normal.y * inv_length, normal.z * inv_length};
    }
    return {0.0f, 0.0f, 1.0f};
}

// Family C: rsqrtf(LengthSq)-scale, branch length_sq <= eps, caller fallback.
//   Used by:  cuda_particle_world.cu NormalizeOr (eps 1e-12, caller fallback)
// rsqrtf path => distinct from Family A; keep separate.
__forceinline__ __device__ Vec3 NormalizeOr(Vec3 v, Vec3 fallback) {
    const float len_sq = LengthSq(v);
    if (len_sq <= 1.0e-12f) {
        return fallback;
    }
    return Scale(v, rsqrtf(len_sq));
}

// ---------------------------------------------------------------------------
// Quaternion helpers
// ---------------------------------------------------------------------------

__forceinline__ __device__ Quat QuatIdentity() {
    return MakeQuat(1.0f, 0.0f, 0.0f, 0.0f);
}

// Hamilton product a * b (w-first), mirroring math::Quat::operator*.
// Identical body in articulation_contacts.cu (QuatMul), batched (Mul),
// row_solver (Multiply); op order is fixed across all copies.
__forceinline__ __device__ Quat QuatMul(Quat a, Quat b) {
    return MakeQuat(
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w);
}

// Quaternion normalization families. These DIFFER in float recipe / epsilon and
// stay SEPARATE.

// Family Q-A: norm = sqrtf(w^2+x^2+y^2+z^2); branch norm <= eps; scale by
// (1.0f/norm). Used by row_solver.cu Normalize (eps 1e-12) and
// batched_device_world.cu NormalizeQuat (eps 1e-8, uses `< eps`).
//   NOTE the comparison operator differs between the two copies:
//     row_solver:  norm <= 1e-12   batched: norm < 1e-8
//   Both are reproduced by parameterizing eps; the `<=` form is the canonical
//   one here and is bit-identical to row_solver. The batched copy used `<`; for
//   a degenerate-to-eps input the boundary differs. They are functionally the
//   same recipe, so this single function serves both, but Phase 2 must map the
//   batched site with care (see migration plan; provided variant below).
__forceinline__ __device__ Quat QuatNormalizeSqrtLe(Quat q, float eps) {
    const float norm = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (norm <= eps) {
        return MakeQuat(1.0f, 0.0f, 0.0f, 0.0f);
    }
    const float inv_norm = 1.0f / norm;
    return MakeQuat(q.w * inv_norm, q.x * inv_norm, q.y * inv_norm, q.z * inv_norm);
}

// batched_device_world.cu NormalizeQuat used `norm < eps` (strict). Provided as a
// distinct symbol so its boundary semantics are preserved byte-for-byte.
__forceinline__ __device__ Quat QuatNormalizeSqrtLt(Quat q, float eps) {
    const float norm = sqrtf(q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z);
    if (norm < eps) {
        return MakeQuat(1.0f, 0.0f, 0.0f, 0.0f);
    }
    const float inv_norm = 1.0f / norm;
    return MakeQuat(q.w * inv_norm, q.x * inv_norm, q.y * inv_norm, q.z * inv_norm);
}

// Family Q-B: norm_sq then rsqrtf, branch norm_sq <= eps, fallback identity.
// Used by articulation_contacts.cu QuatNormalize (eps 1e-12).
__forceinline__ __device__ Quat QuatNormalizeRsqrt(Quat q, float eps) {
    const float norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (norm_sq <= eps) {
        return QuatIdentity();
    }
    const float inv_norm = rsqrtf(norm_sq);
    return MakeQuat(q.w * inv_norm, q.x * inv_norm, q.y * inv_norm, q.z * inv_norm);
}

// Family Q-C: norm_sq then rsqrtf in-place, branch norm_sq < eps (strict),
// fallback identity. Used by featherstone_aba.cu QuatNormalizeForward
// (eps 1e-24, `<`). Distinct boundary => separate symbol.
__forceinline__ __device__ Quat QuatNormalizeForwardRsqrt(Quat q, float eps) {
    const float norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (norm_sq < eps) {
        Quat identity;
        identity.w = 1.0f;
        identity.x = 0.0f;
        identity.y = 0.0f;
        identity.z = 0.0f;
        return identity;
    }
    const float inv = rsqrtf(norm_sq);
    q.w *= inv;
    q.x *= inv;
    q.y *= inv;
    q.z *= inv;
    return q;
}

// Rotate a vector by a quaternion, DEFENSIVELY NORMALIZING q first (rsqrtf,
// branch norm_sq > 1e-12), then the explicit-component q*v*q^-1 short form with
// the intermediate t2 = 2*(qv x v). Identical body in:
//   articulation_jacobian.cu  RotateByQuat
//   articulation_drives.cu     RotateByQuatDrive   (OSC task-Jacobian path)
// (character-identical; jacobian uses `Cross` via `using mg::Cross`, drives uses
// the qualified `mg::Cross` -- same symbol). This is a DISTINCT recipe from
// RotateShort below (which takes an already-normalized q and folds the 2x into t
// before the second cross, a different float expression) -- the two are kept
// SEPARATE so neither call site's bytes shift. NormalizeOrUp's contacts copy uses
// yet another (ScaleVec) form and is intentionally NOT merged here.
__forceinline__ __device__ Vec3 RotateByQuatNormalized(Quat q, Vec3 v) {
    const float norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (norm_sq > 1.0e-12f) {
        const float inv_norm = rsqrtf(norm_sq);
        q.w *= inv_norm;
        q.x *= inv_norm;
        q.y *= inv_norm;
        q.z *= inv_norm;
    }
    const Vec3 qv{q.x, q.y, q.z};
    const Vec3 t = Cross(qv, v);
    const Vec3 t2{2.0f * t.x, 2.0f * t.y, 2.0f * t.z};
    const Vec3 wt{q.w * t2.x, q.w * t2.y, q.w * t2.z};
    const Vec3 qvt = Cross(qv, t2);
    return {v.x + wt.x + qvt.x, v.y + wt.y + qvt.y, v.z + wt.z + qvt.z};
}

// Rotate a vector by a (possibly-unnormalized) quaternion using the
//   t = 2*(qv x v);  v + w*t + qv x t
// short form. Identical body in broadphase / contact_generation / sensors /
// batched / row_solver (named Rotate). Depends on Add/Scale/Cross above.
__forceinline__ __device__ Vec3 RotateShort(Quat q, Vec3 v) {
    const Vec3 qv = MakeVec3(q.x, q.y, q.z);
    const Vec3 t = Scale(Cross(qv, v), 2.0f);
    return Add(Add(v, Scale(t, q.w)), Cross(qv, t));
}

} // namespace nuka::math::gpu
