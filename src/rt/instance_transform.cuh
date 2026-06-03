#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- the SHARED rigid-instance ray transform for the two-level tracer
// (v0.7 G1-A). HOISTED from sensor/camera_render.cu's anon namespace so the
// SAME quaternion-vector rotation is used by:
//   * the p14b camera sensor (IntersectBodyT / ReconstructWorldNormal),
//   * the G1-A two-level tracer (TLAS leaf transforms the ray into each
//     instance's local frame; the BLAS local normal is rotated back to world),
//   * the G1-A host brute-force ORACLE (byte-exact host==device parity).
//
// The rotation is the exact `t = 2*cross(qv,v)` fp32 form math::Quat::Rotate
// uses on the host (math/quat.hpp), so this is a behavior-IDENTICAL hoist: the
// p14b camera path produces bit-for-bit the same image (nuka_camera_render_test
// is the no-regression guard). math::Quat::Rotate itself is host-only `inline`
// (it is NOT __host__ __device__), so we cannot call it from device code; this
// header provides the HD equivalents WITHOUT touching the shared math headers.
//
// __host__ __device__ (NUKA_RT_HD): the host oracle calls TransformRayToLocal
// VERBATIM, so the only arithmetic that can diverge GPU-vs-oracle is fp32
// add/mul/cross -- IEEE round-to-nearest identical once FMA contraction is off
// (--fmad=false on the .cu, -ffp-contract=off on the oracle TU). NO atomics, no
// RNG -> D1-safe.
//
// Rigid invariant (load-bearing for the two-level tracer): a unit-quaternion
// rotation preserves vector length, so |d_l| == |d| == 1. Hence the BLAS-local
// march/intersection distance t EQUALS the world distance unchanged -> best_t
// is directly comparable across instances at the TLAS level with NO rescale,
// and t_min is frame-invariant (the same t_min/shadow_eps is passed into the
// BLAS leaf untouched).
// ---------------------------------------------------------------------------

#include "math/quat.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"

#if defined(__CUDACC__)
#define NUKA_RT_HD __host__ __device__
#else
#define NUKA_RT_HD
#endif

namespace nuka::rt {

// Rotate a vector v by a unit quaternion q (q * v * q^-1). Same form as
// math::Quat::Rotate; HD so it runs on device. fp32 throughout (matches Quat).
NUKA_RT_HD inline math::Vec3 QuatRotate(const math::Quat& q, const math::Vec3& v) {
    const math::Vec3 qv{q.x, q.y, q.z};
    const math::Vec3 t = 2.0f * qv.Cross(v);
    return v + q.w * t + qv.Cross(t);
}

// Inverse-rotate a vector v by a unit quaternion q (q^-1 * v * q). The conjugate
// of a unit quaternion is its inverse: negate the vector part. fp32 throughout.
NUKA_RT_HD inline math::Vec3 QuatInvRotate(const math::Quat& q, const math::Vec3& v) {
    const math::Vec3 qv{-q.x, -q.y, -q.z};
    const math::Vec3 t = 2.0f * qv.Cross(v);
    return v + q.w * t + qv.Cross(t);
}

// Transform a world-space ray (origin o, unit direction d) into the LOCAL frame
// of a rigid transform T: o_l = R^-1*(o - p), d_l = R^-1*d. This is the EXACT
// sequence the kernel and the oracle must share -- do NOT use Transform::Inverse()
// (its Conjugate().Normalized() introduces an extra sqrt/divide and different
// bits, breaking host==device parity). Rigid -> |d_l| == |d|, so the local-frame
// hit distance is the world distance unchanged.
NUKA_RT_HD inline void TransformRayToLocal(const math::Transform& T,
                                           const math::Vec3& o,
                                           const math::Vec3& d,
                                           math::Vec3* o_l,
                                           math::Vec3* d_l) {
    const math::Vec3 o_rel{o.x - T.position.x, o.y - T.position.y,
                           o.z - T.position.z};
    *o_l = QuatInvRotate(T.rotation, o_rel);
    *d_l = QuatInvRotate(T.rotation, d);
}

} // namespace nuka::rt

#undef NUKA_RT_HD
