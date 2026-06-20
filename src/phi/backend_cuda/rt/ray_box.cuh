#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- ray-AABB slab intersection (v0.7 p12).
//
// THIS is the p12 leaf-primitive intersection: a ray vs an axis-aligned BOX,
// returning a GENUINE hit distance t_near (not a "closest AABB-entry" candidate).
// p13 will swap RayBoxIntersect -> triangle/sphere/sparse-SDF and add shading;
// NOTHING in the BVH descent changes because the leaf t is computed here.
//
// CRITICAL for the oracle-match gate (D1 is automatic; matching the oracle is the
// work): this is ONE __host__ __device__ function included by BOTH the render
// kernel AND the CPU brute-force oracle. Arithmetic is fp64 internally, stored
// fp32, formulated as (box.min - origin) * inv_dir -- subtract-then-multiply,
// NO a*b+c pattern -> no FMA-contraction divergence between host and device.
// (Belt-and-suspenders: the .cu is also built with --fmad=false and the oracle
// TU with -ffp-contract=off so the build itself forbids contraction.)
//
// Returns true on hit and writes t_near (the world-space distance to the first
// intersection with the box's surface). A ray that ORIGINATES inside the box
// reports the entry t which is <= 0; we treat such hits as valid only when
// t_near >= 0 (the camera is outside all boxes in p12 scenes), and the standard
// slab early-out (t_near <= t_far, t_far >= 0) is applied. Inputs: ray origin,
// unit direction, and the box min/max (a collision::AABB).
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "math/vec3.hpp"

#include <cstdint>

#if defined(__CUDACC__)
#define NUKA_RT_HD __host__ __device__
#else
#define NUKA_RT_HD
#endif

namespace nuka::rt {

// Sentinel depth for a miss: +inf. The framebuffer is initialized to this; a
// pixel that hits nothing keeps it. prim-id sentinel is kNoPrim.
NUKA_RT_HD inline float RtMissDepth() {
    // 0x7f800000 == +inf in fp32; byte-stable across host/device.
    return __builtin_huge_valf();
}

inline constexpr uint32_t kNoPrim = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// NaN-GUARD (p13, p12 reviewer carry-over now LIVE): a ray exactly axis-parallel
// to a slab (dir component == 0) while ORIGINATING on that slab's plane gives
// (plane - origin) == 0 and inv_dir == +/-inf, so 0 * inf == NaN -- and a NaN
// poisons the fmin/fmax slab reduction (a comparison against NaN is false, so
// the wrong branch is taken). p12 was immune (camera outside all boxes, never on
// a plane); p13's SHADOW / secondary rays originate ON surfaces, so this is live.
//
// Guard: replace 1/d for a (near-)zero d with a LARGE FINITE value of the right
// sign (copysign(kHuge, d)). Then 0 * large == 0 (finite, not NaN), and a true
// axis-parallel-but-off-plane ray still gets a huge +/- t that fmin/fmax select
// away identically to the +/-inf it replaced -- so this is bit-exact-preserving
// for every p12 ray (where the in-plane case never occurred) AND NaN-free for
// the p13 surface-origin case. kHuge is 1e300: comfortably larger than any scene
// extent yet finite, so 0*kHuge == 0 exactly. ONE helper, used by BOTH the leaf
// box test here AND NodeEntryT in the traversal .cu, so the guard covers the
// node-pruning math too (a NaN there would mis-prune a subtree => wrong result).
// ---------------------------------------------------------------------------
// Real defaults to double: every existing caller (golden kernel + host oracle)
// is byte-unchanged. The beauty path instantiates Real=float for FP32 speed.
template <typename Real = double>
NUKA_RT_HD inline Real RtSafeInvDir(float d) {
    const Real dd = static_cast<Real>(d);
    // |d| below this -> treat as axis-parallel; clamp inv to a large finite of
    // the same sign. The threshold scales with the precision so float never
    // underflows it. fp64: 1e-300; fp32: a denormal-safe small value.
    const Real kTinyDir = sizeof(Real) >= 8u ? static_cast<Real>(1.0e-300)
                                             : static_cast<Real>(1.0e-30);
    const Real kHugeInv = sizeof(Real) >= 8u ? static_cast<Real>(1.0e300)
                                             : static_cast<Real>(1.0e30);
    const Real one = static_cast<Real>(1);
    if (dd > kTinyDir) {
        return one / dd;
    }
    if (dd < -kTinyDir) {
        return one / dd;
    }
    // dd == 0 (or denormally tiny): return a large finite with the sign of dd.
    // copysign(x, +0)=+x, copysign(x, -0)=-x; for a literal 0.0f the sign is +.
    return dd < static_cast<Real>(0) ? -kHugeInv : kHugeInv;
}

// Shared slab reduction: computes the [tmin, tmax] interval of the ray vs the
// box. Uses RtSafeInvDir (NaN-guarded). Subtract-then-multiply only (no FMA).
// Returns false on a miss (interval empty or box entirely behind the ray).
template <typename Real = double>
NUKA_RT_HD inline bool RtSlabInterval(const math::Vec3& origin,
                                      const math::Vec3& dir,
                                      const collision::AABB& box,
                                      Real* out_tmin,
                                      Real* out_tmax) {
    const Real inv_x = RtSafeInvDir<Real>(dir.x);
    const Real inv_y = RtSafeInvDir<Real>(dir.y);
    const Real inv_z = RtSafeInvDir<Real>(dir.z);

    // Per-axis plane distances: (plane - origin) * inv_dir. Subtract then
    // multiply -> no a*b+c -> no FMA contraction.
    const Real tx0 = (static_cast<Real>(box.min.x) - static_cast<Real>(origin.x)) * inv_x;
    const Real tx1 = (static_cast<Real>(box.max.x) - static_cast<Real>(origin.x)) * inv_x;
    const Real ty0 = (static_cast<Real>(box.min.y) - static_cast<Real>(origin.y)) * inv_y;
    const Real ty1 = (static_cast<Real>(box.max.y) - static_cast<Real>(origin.y)) * inv_y;
    const Real tz0 = (static_cast<Real>(box.min.z) - static_cast<Real>(origin.z)) * inv_z;
    const Real tz1 = (static_cast<Real>(box.max.z) - static_cast<Real>(origin.z)) * inv_z;

    const Real txmin = tx0 < tx1 ? tx0 : tx1;
    const Real txmax = tx0 < tx1 ? tx1 : tx0;
    const Real tymin = ty0 < ty1 ? ty0 : ty1;
    const Real tymax = ty0 < ty1 ? ty1 : ty0;
    const Real tzmin = tz0 < tz1 ? tz0 : tz1;
    const Real tzmax = tz0 < tz1 ? tz1 : tz0;

    Real tmin = txmin > tymin ? txmin : tymin;
    tmin = tmin > tzmin ? tmin : tzmin;
    Real tmax = txmax < tymax ? txmax : tymax;
    tmax = tmax < tzmax ? tmax : tzmax;

    if (tmax < tmin || tmax < static_cast<Real>(0)) {
        return false;
    }
    *out_tmin = tmin;
    *out_tmax = tmax;
    return true;
}

// Ray vs AABB slab test. origin/dir are world-space (dir unit length). On a hit
// returns true and sets *t_near to the entry distance along the ray (>= 0 for a
// camera outside the box). fp64 internally, fp32 stored. NaN-guarded via the
// shared RtSlabInterval. The ONLY operations are subtract, multiply (by inv_dir),
// and min/max -- no fused multiply-add anywhere.
template <typename Real = double>
NUKA_RT_HD inline bool RayBoxIntersect(const math::Vec3& origin,
                                       const math::Vec3& dir,
                                       const collision::AABB& box,
                                       float* t_near) {
    Real tmin, tmax;
    if (!RtSlabInterval<Real>(origin, dir, box, &tmin, &tmax)) {
        return false;
    }
    // Entry distance: if the origin is inside the box tmin < 0; clamp to the
    // first surface crossing in front of the ray (tmin if >= 0 else tmax).
    const Real t = tmin >= static_cast<Real>(0) ? tmin : tmax;
    *t_near = static_cast<float>(t);
    return true;
}

// Ray vs AABB ENTRY t for node pruning: the entry distance clamped to 0 when the
// ray starts inside the box, plus a hit flag. Same NaN-guarded slab math as
// RayBoxIntersect (shares RtSlabInterval). Used by the traversal's NodeEntryT so
// the NaN guard covers the node-pruning test too.
template <typename Real = double>
NUKA_RT_HD inline bool RtNodeEntryT(const math::Vec3& origin,
                                    const math::Vec3& dir,
                                    const collision::AABB& box,
                                    float* entry_t) {
    Real tmin, tmax;
    if (!RtSlabInterval<Real>(origin, dir, box, &tmin, &tmax)) {
        return false;
    }
    *entry_t = static_cast<float>(tmin >= static_cast<Real>(0) ? tmin : static_cast<Real>(0));
    return true;
}

// Closest-hit update with the EXACT tie-break the oracle must mirror: on equal
// t keep the LOWEST primitive index. Used identically by kernel and oracle so
// depth+prim-id match bit-for-bit. Returns true if (t,prim) became the new best.
NUKA_RT_HD inline bool RtClosestHitUpdate(float t, uint32_t prim,
                                          float* best_t, uint32_t* best_prim) {
    if (t < *best_t || (t == *best_t && prim < *best_prim)) {
        *best_t = t;
        *best_prim = prim;
        return true;
    }
    return false;
}

} // namespace nuka::rt

#undef NUKA_RT_HD
