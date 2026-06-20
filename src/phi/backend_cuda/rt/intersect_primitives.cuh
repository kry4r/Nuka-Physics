#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- real ray/primitive leaf intersections (v0.7 p13).
//
// p12 shipped a box-only leaf (ray_box.cuh::RayBoxIntersect). p13 GENERALIZES the
// leaf to real primitives: TRIANGLE (Moller-Trumbore), SPHERE (analytic), and
// SPARSE-SDF (sphere-tracing the p07 narrow-band SDF). Each returns a genuine hit
// distance t PLUS the surface attributes the AOV framebuffer needs (normal, uv /
// barycentric). The BVH descent is UNCHANGED: these functions are called at the
// leaf exactly where RayBoxIntersect was, and the closest-hit + lowest-index
// tie-break (RtClosestHitUpdate) is reused verbatim.
//
// SHARED __host__ __device__ (NUKA_RT_HD): ONE definition included by BOTH the
// render kernel AND the CPU brute-force oracle (the p12 pattern). To hold the
// "depth/normal/bary EXACT vs oracle" gate, the math is fp64-internal / fp32
// stored, subtract-then-multiply (no a*b+c -> no FMA-contraction divergence),
// TRUE sqrt + TRUE divide (no rsqrtf/__fdividef fast-math). Belt-and-suspenders:
// the .cu is built --fmad=false, the oracle TU -ffp-contract=off.
//
// DETERMINISM: no atomics, no RNG, fixed evaluation order -> two runs identical.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "math/vec3.hpp"
#include "phi/backend_cuda/rt/ray_box.cuh"  // RtSafeInvDir (shared NaN-guarded inverse direction)
#include "runtime/sdf/sparse_sdf_query.cuh"

#include <cstdint>

#if defined(__CUDACC__)
#define NUKA_RT_HD __host__ __device__
#else
#define NUKA_RT_HD
#endif

namespace nuka::rt {

// Normalize a Vec3 (TRUE sqrt + TRUE divide, no rsqrt). Returns {0,0,0} for a
// (near-)zero vector. Real defaults to double (golden + oracle byte-unchanged);
// the beauty path instantiates Real=float. Host==device for a fixed Real.
template <typename Real = double>
NUKA_RT_HD inline math::Vec3 RtNormalize(const math::Vec3& v) {
    const Real x = static_cast<Real>(v.x);
    const Real y = static_cast<Real>(v.y);
    const Real z = static_cast<Real>(v.z);
    const Real len2 = x * x + y * y + z * z;
    if (len2 <= static_cast<Real>(0)) {
        return math::Vec3{0.0f, 0.0f, 0.0f};
    }
    const Real inv = static_cast<Real>(1) / sqrt(len2);
    return math::Vec3{static_cast<float>(x * inv),
                      static_cast<float>(y * inv),
                      static_cast<float>(z * inv)};
}

// -------------------------------------------------------------------------
// TRIANGLE -- Moller-Trumbore. origin/dir world-space (dir unit). On a hit
// returns true and sets t (world distance), barycentric (u,v) [w = 1-u-v at v0],
// and the GEOMETRIC normal (normalize(cross(e1,e2)), NOT flipped to the viewer --
// the AOV stores the geometric normal; the shader flips for lighting). fp64
// internal. The 1/a divide is a true divide; cross/dot are subtract-then-mul
// chains (no FMA). t must be in front of t_min to count.
// -------------------------------------------------------------------------
template <typename Real = double>
NUKA_RT_HD inline bool RayTriangleIntersect(const math::Vec3& origin,
                                            const math::Vec3& dir,
                                            const math::Vec3& v0,
                                            const math::Vec3& v1,
                                            const math::Vec3& v2,
                                            float t_min,
                                            float* out_t,
                                            float* out_u,
                                            float* out_v,
                                            math::Vec3* out_normal) {
    // Edges.
    const Real e1x = static_cast<Real>(v1.x) - static_cast<Real>(v0.x);
    const Real e1y = static_cast<Real>(v1.y) - static_cast<Real>(v0.y);
    const Real e1z = static_cast<Real>(v1.z) - static_cast<Real>(v0.z);
    const Real e2x = static_cast<Real>(v2.x) - static_cast<Real>(v0.x);
    const Real e2y = static_cast<Real>(v2.y) - static_cast<Real>(v0.y);
    const Real e2z = static_cast<Real>(v2.z) - static_cast<Real>(v0.z);

    const Real dx = static_cast<Real>(dir.x);
    const Real dy = static_cast<Real>(dir.y);
    const Real dz = static_cast<Real>(dir.z);

    // h = cross(dir, e2)
    const Real hx = dy * e2z - dz * e2y;
    const Real hy = dz * e2x - dx * e2z;
    const Real hz = dx * e2y - dy * e2x;
    // a = dot(e1, h)
    const Real a = e1x * hx + e1y * hy + e1z * hz;
    const Real kEps = static_cast<Real>(1.0e-12);
    if (a > -kEps && a < kEps) {
        return false; // ray parallel to triangle plane
    }
    const Real f = static_cast<Real>(1) / a;

    // s = origin - v0
    const Real sx = static_cast<Real>(origin.x) - static_cast<Real>(v0.x);
    const Real sy = static_cast<Real>(origin.y) - static_cast<Real>(v0.y);
    const Real sz = static_cast<Real>(origin.z) - static_cast<Real>(v0.z);

    // u = f * dot(s, h)
    const Real u = f * (sx * hx + sy * hy + sz * hz);
    if (u < static_cast<Real>(0) || u > static_cast<Real>(1)) {
        return false;
    }
    // q = cross(s, e1)
    const Real qx = sy * e1z - sz * e1y;
    const Real qy = sz * e1x - sx * e1z;
    const Real qz = sx * e1y - sy * e1x;
    // v = f * dot(dir, q)
    const Real v = f * (dx * qx + dy * qy + dz * qz);
    if (v < static_cast<Real>(0) || (u + v) > static_cast<Real>(1)) {
        return false;
    }
    // t = f * dot(e2, q)
    const Real t = f * (e2x * qx + e2y * qy + e2z * qz);
    if (t < static_cast<Real>(t_min)) {
        return false;
    }

    *out_t = static_cast<float>(t);
    *out_u = static_cast<float>(u);
    *out_v = static_cast<float>(v);
    // geometric normal = normalize(cross(e1, e2))
    const math::Vec3 ng{static_cast<float>(e1y * e2z - e1z * e2y),
                        static_cast<float>(e1z * e2x - e1x * e2z),
                        static_cast<float>(e1x * e2y - e1y * e2x)};
    *out_normal = RtNormalize<Real>(ng);
    return true;
}

// -------------------------------------------------------------------------
// SPHERE -- analytic. Solves the quadratic for the nearest root >= t_min. On a
// hit returns t, outward normal (unit), and spherical uv (u in [0,1) about +Y,
// v in [0,1] pole-to-pole). fp64 internal; true sqrt + true divide.
// -------------------------------------------------------------------------
template <typename Real = double>
NUKA_RT_HD inline bool RaySphereIntersect(const math::Vec3& origin,
                                          const math::Vec3& dir,
                                          const math::Vec3& center,
                                          float radius,
                                          float t_min,
                                          float* out_t,
                                          math::Vec3* out_normal,
                                          float* out_u,
                                          float* out_v) {
    const Real ocx = static_cast<Real>(origin.x) - static_cast<Real>(center.x);
    const Real ocy = static_cast<Real>(origin.y) - static_cast<Real>(center.y);
    const Real ocz = static_cast<Real>(origin.z) - static_cast<Real>(center.z);
    const Real dx = static_cast<Real>(dir.x);
    const Real dy = static_cast<Real>(dir.y);
    const Real dz = static_cast<Real>(dir.z);
    const Real r = static_cast<Real>(radius);

    // dir is unit so a == 1; keep it general (a = dot(d,d)) for robustness.
    const Real a = dx * dx + dy * dy + dz * dz;
    const Real b = ocx * dx + ocy * dy + ocz * dz;       // half-b
    const Real c = (ocx * ocx + ocy * ocy + ocz * ocz) - r * r;
    const Real disc = b * b - a * c;                     // (half-b)^2 - a*c
    if (disc < static_cast<Real>(0)) {
        return false;
    }
    const Real sqrt_d = sqrt(disc);
    Real t = (-b - sqrt_d) / a;
    if (t < static_cast<Real>(t_min)) {
        t = (-b + sqrt_d) / a;                             // far root (inside)
        if (t < static_cast<Real>(t_min)) {
            return false;
        }
    }

    *out_t = static_cast<float>(t);
    // hit point p = origin + t*dir; normal = (p - center)/r
    const Real px = static_cast<Real>(origin.x) + t * dx;
    const Real py = static_cast<Real>(origin.y) + t * dy;
    const Real pz = static_cast<Real>(origin.z) + t * dz;
    const math::Vec3 n{static_cast<float>(px - static_cast<Real>(center.x)),
                       static_cast<float>(py - static_cast<Real>(center.y)),
                       static_cast<float>(pz - static_cast<Real>(center.z))};
    const math::Vec3 nn = RtNormalize<Real>(n);
    *out_normal = nn;

    // Spherical uv from the unit normal. atan2/acos: these feed the uv AOV
    // (tolerance-checked downstream), not the EXACT depth/bary gate.
    const Real pi = static_cast<Real>(3.14159265358979323846);
    const Real ny = static_cast<Real>(nn.y) < static_cast<Real>(-1) ? static_cast<Real>(-1)
                  : (static_cast<Real>(nn.y) > static_cast<Real>(1) ? static_cast<Real>(1)
                  : static_cast<Real>(nn.y));
    const Real uu = static_cast<Real>(0.5) + atan2(static_cast<Real>(nn.z),
                                  static_cast<Real>(nn.x)) / (static_cast<Real>(2) * pi);
    const Real vv = static_cast<Real>(0.5) - asin(ny) / pi;
    *out_u = static_cast<float>(uu);
    *out_v = static_cast<float>(vv);
    return true;
}

// -------------------------------------------------------------------------
// SPARSE-SDF -- sphere-trace the p07 narrow-band SDF (sparse_sdf_sample). The
// SDF lives in its OWN local frame; for p13 we render it at IDENTITY transform
// (world == local) -- the SDF test cooks at identity, keeping the normal gate
// clean (no transform_normal). The narrow band returns kOutsideBand (1e30)
// OUTSIDE the band, so a naive `t += phi` from the camera (outside the band)
// would jump 1e30 and instantly "miss". The fix (per the p13 design + advisor):
//   1) Clip the march to the SDF AABB [aabb_t_min, aabb_t_max] (slab entry/exit).
//   2) Step t += min(phi, max_step) with max_step ~= voxel_size, so an
//      OUT-OF-BAND sample steps ONE voxel forward (fixed-march) instead of 1e30,
//      tightening to a true sphere-trace once inside the band.
//   3) Terminate phi <= surface_eps BEFORE stepping (never step on a hit, never
//      march backward on a tiny negative phi).
// Returns t at the surface crossing + the surface normal (normalize(gradient)).
// The hit is "within tolerance" (state the measured bound), NOT bit-exact -- the
// iterative march is the one gate the task allows as tolerance-matched.
//
// aabb is the SDF's world-space bound (== local, identity); pass it so we can
// clip the march. surface_eps / max_iters / max_step are explicit knobs.
// -------------------------------------------------------------------------
template <typename Real = double>
NUKA_RT_HD inline bool RaySdfSphereTrace(const math::Vec3& origin,
                                         const math::Vec3& dir,
                                         const runtime::sdf::SparseSdfDevice& sdf,
                                         const collision::AABB& aabb,
                                         float t_min,
                                         float surface_eps,
                                         int max_iters,
                                         float* out_t,
                                         math::Vec3* out_normal) {
    // Clip the march to the SDF AABB; outside it there is no band coverage.
    Real aabb_tmin, aabb_tmax;
    {
        // Inline slab (NaN-guarded) interval against the SDF bound.
        const Real inv_x = RtSafeInvDir<Real>(dir.x);
        const Real inv_y = RtSafeInvDir<Real>(dir.y);
        const Real inv_z = RtSafeInvDir<Real>(dir.z);
        const Real tx0 = (static_cast<Real>(aabb.min.x) - static_cast<Real>(origin.x)) * inv_x;
        const Real tx1 = (static_cast<Real>(aabb.max.x) - static_cast<Real>(origin.x)) * inv_x;
        const Real ty0 = (static_cast<Real>(aabb.min.y) - static_cast<Real>(origin.y)) * inv_y;
        const Real ty1 = (static_cast<Real>(aabb.max.y) - static_cast<Real>(origin.y)) * inv_y;
        const Real tz0 = (static_cast<Real>(aabb.min.z) - static_cast<Real>(origin.z)) * inv_z;
        const Real tz1 = (static_cast<Real>(aabb.max.z) - static_cast<Real>(origin.z)) * inv_z;
        const Real txmin = tx0 < tx1 ? tx0 : tx1, txmax = tx0 < tx1 ? tx1 : tx0;
        const Real tymin = ty0 < ty1 ? ty0 : ty1, tymax = ty0 < ty1 ? ty1 : ty0;
        const Real tzmin = tz0 < tz1 ? tz0 : tz1, tzmax = tz0 < tz1 ? tz1 : tz0;
        Real a = txmin > tymin ? txmin : tymin; a = a > tzmin ? a : tzmin;
        Real b = txmax < tymax ? txmax : tymax; b = b < tzmax ? b : tzmax;
        if (b < a || b < static_cast<Real>(0)) {
            return false;
        }
        aabb_tmin = a > static_cast<Real>(t_min) ? a : static_cast<Real>(t_min);
        aabb_tmax = b;
    }
    if (aabb_tmin > aabb_tmax) {
        return false;
    }

    const float max_step = sdf.voxel_size > 0.0f ? sdf.voxel_size : 1.0f;
    Real t = aabb_tmin;
    for (int i = 0; i < max_iters; ++i) {
        if (t > aabb_tmax) {
            return false;
        }
        const math::Vec3 p{static_cast<float>(static_cast<Real>(origin.x) + t * static_cast<Real>(dir.x)),
                           static_cast<float>(static_cast<Real>(origin.y) + t * static_cast<Real>(dir.y)),
                           static_cast<float>(static_cast<Real>(origin.z) + t * static_cast<Real>(dir.z))};
        math::Vec3 grad;
        const float phi = runtime::sdf::sparse_sdf_sample(sdf, p, grad);
        if (phi <= surface_eps && phi > -1.0e29f) {
            // Surface crossing (and a real in-band sample, not kOutsideBand).
            *out_t = static_cast<float>(t);
            *out_normal = RtNormalize<Real>(grad);
            return true;
        }
        // Step: in-band -> sphere-trace by phi (clamped to a voxel so a large
        // positive band-edge phi cannot overshoot the surface); out-of-band
        // (phi == kOutsideBand) -> one voxel fixed-march.
        Real step = static_cast<Real>(phi);
        if (step > static_cast<Real>(max_step)) {
            step = static_cast<Real>(max_step);
        }
        // Guarantee FORWARD progress: at least a small fraction of a voxel. This
        // also subsumes "never march backward" -- a negative phi (we slightly
        // overshot the surface but phi > surface_eps did not fire because we
        // landed just inside) is bumped up to min_progress, never negative.
        const Real min_progress = static_cast<Real>(max_step) * static_cast<Real>(1.0e-3);
        if (step < min_progress) {
            step = min_progress;
        }
        t += step;
    }
    return false;
}

} // namespace nuka::rt

#undef NUKA_RT_HD
