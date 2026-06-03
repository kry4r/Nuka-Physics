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

// Ray vs AABB slab test. origin/dir are world-space (dir unit length). On a hit
// returns true and sets *t_near to the entry distance along the ray (>= 0 for a
// camera outside the box). fp64 internally, fp32 stored. The branchless slab
// formulation uses fmin/fmax of the two per-axis plane distances; division by a
// (possibly zero) dir component yields +/-inf which fmin/fmax handle correctly
// (a ray parallel to a slab is inside that slab iff origin is between the
// planes, which the inf arithmetic encodes). The ONLY operations are subtract,
// multiply (by inv_dir), and min/max -- no fused multiply-add anywhere.
NUKA_RT_HD inline bool RayBoxIntersect(const math::Vec3& origin,
                                       const math::Vec3& dir,
                                       const collision::AABB& box,
                                       float* t_near) {
    // inv_dir in fp64; a zero component -> +/-inf, intentional.
    const double inv_x = 1.0 / static_cast<double>(dir.x);
    const double inv_y = 1.0 / static_cast<double>(dir.y);
    const double inv_z = 1.0 / static_cast<double>(dir.z);

    // Per-axis plane distances: (plane - origin) * inv_dir. Subtract then
    // multiply -> no a*b+c -> no FMA contraction.
    const double tx0 = (static_cast<double>(box.min.x) - static_cast<double>(origin.x)) * inv_x;
    const double tx1 = (static_cast<double>(box.max.x) - static_cast<double>(origin.x)) * inv_x;
    const double ty0 = (static_cast<double>(box.min.y) - static_cast<double>(origin.y)) * inv_y;
    const double ty1 = (static_cast<double>(box.max.y) - static_cast<double>(origin.y)) * inv_y;
    const double tz0 = (static_cast<double>(box.min.z) - static_cast<double>(origin.z)) * inv_z;
    const double tz1 = (static_cast<double>(box.max.z) - static_cast<double>(origin.z)) * inv_z;

    const double txmin = tx0 < tx1 ? tx0 : tx1;
    const double txmax = tx0 < tx1 ? tx1 : tx0;
    const double tymin = ty0 < ty1 ? ty0 : ty1;
    const double tymax = ty0 < ty1 ? ty1 : ty0;
    const double tzmin = tz0 < tz1 ? tz0 : tz1;
    const double tzmax = tz0 < tz1 ? tz1 : tz0;

    double tmin = txmin > tymin ? txmin : tymin;
    tmin = tmin > tzmin ? tmin : tzmin;
    double tmax = txmax < tymax ? txmax : tymax;
    tmax = tmax < tzmax ? tmax : tzmax;

    // Miss if the slabs don't overlap, or the box is entirely behind the ray.
    if (tmax < tmin || tmax < 0.0) {
        return false;
    }
    // Entry distance: if the origin is inside the box tmin < 0; clamp to the
    // first surface crossing in front of the ray (tmin if >= 0 else tmax). In
    // p12 the camera is outside all boxes so tmin >= 0; this clamp is a guard.
    const double t = tmin >= 0.0 ? tmin : tmax;
    *t_near = static_cast<float>(t);
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
