#pragma once
// ---------------------------------------------------------------------------
// nuka::collision -- analytical primitive narrowphase handlers (v0.8 C3b)
// ---------------------------------------------------------------------------
// The Analytical-tier geometry math for the C3a dispatch table: closed-form,
// multi-point / face-face manifolds for primitive x primitive pairs (Q2 hybrid
// narrowphase). Ports + UPGRADES the v0.7 single-point generators
// (world_stepper.cpp GeneratePlaneContact/SphereSphere/SphereBox/BoxBox):
//   - box vs plane  -> up to 4 deepest box corners below the plane (4-pt patch).
//   - box vs box    -> SAT min-penetration axis + Sutherland-Hodgman face clip,
//                      reduced to <=4 by a DETERMINISTIC feature-id key.
//   - sphere vs sphere / sphere vs box / sphere vs plane -> 1 point.
//   - capsule vs plane (<=2 endpoints) / capsule vs sphere (1 pt).
// (capsule vs box / capsule vs capsule DEFERRED to the stub -- see header end.)
//
// NORMAL CONVENTION (matches ContactPoint + row_builder.cpp:88): point.normal is
// the SEPARATION DIRECTION FOR SIDE A of the manifold -- i.e. the direction in
// which side A must move to resolve the overlap. We honour the manifold's ACTUAL
// (a,b) sides (the CandidatePair is ordered by packed (type<<28|handle), NOT by
// shape type, so the box may be a OR b): each handler is written for one
// (type_a,type_b) ordering and the table also registers the swapped slot; the
// swapped wrapper FLIPS the normal so it always points "the way A must move".
//
// ---------------------------------------------------------------------------
// WHY HD-CLEAN MATH (device-safe pattern, sdf_contact.hpp:61-69):
// ---------------------------------------------------------------------------
// These handlers are authored as __host__ __device__ (NUKA_AMF_HD) free
// functions so the SAME geometry compiles for the C5 device dispatch. Like
// find_sdf_contact_newton, they use ONLY Vec3 HD-constexpr ops + sqrtf and a
// BAKED rotation frame (PrimFrame, columns = world axes) -- never Vec3::Length /
// Vec3::Normalized / Quat::Rotate, which are plain `inline` (NOT HD) on the
// production include path (vec3.hpp/quat.hpp). The caller bakes the quaternion
// into PrimFrame columns on the host (BuildPrimFrame). We do NOT add HD to
// vec3.hpp -- keeping HD blast radius off the golden include path.
//
// CAVEAT (honest): no .cu includes this header yet (C3a is validated-not-wired,
// device dispatch is the C5 seam). So the HOST build proves host-compilability;
// nvcc-compilability is STRUCTURED-FOR but UNVERIFIED until C5 wires the device
// dispatch. The code uses only the device-safe primitives above to make that
// wiring a no-op, but the report flags it as unverified.
// ---------------------------------------------------------------------------
// D1 (the face-clip determinism trap, C3b risk):
//   - Sutherland-Hodgman clips edges in a FIXED vertex order (0->1->2->3),
//     against the reference face's 4 side planes in a FIXED plane order. No
//     data-dependent reordering of the FP accumulation.
//   - When >4 clip points survive, the deepest-4 reduction sorts the <=8
//     candidates by a STABLE key whose PRIMARY ordering is a packed integer
//     FEATURE ID (incident-vertex index in the box-box case / corner bitmask in
//     the box-plane case), NOT the float penetration. Float penetration is a
//     SECONDARY de-prioritiser only where features cannot tie (it never decides
//     ties in the symmetric resting case, where every point ties on penetration
//     -- the feature id is the SOLE determinant there, the classic trap).
//   - The same feature id is written into ContactPoint.stable_key (warm-start
//     identity for free; deterministic per-point identity per the C3b spec).
//   - Emitted point SLOT ORDER is the sorted (ascending feature-id) order, so a
//     two-run memcmp of the manifold is byte-identical (D1 gate).
// ---------------------------------------------------------------------------

#include "constraint/contact_manifold.hpp"
#include "math/transform.hpp"
#include "math/vec3.hpp"

#include <cstdint>

#if defined(__CUDACC__)
#define NUKA_AMF_HD __host__ __device__
#else
#define NUKA_AMF_HD
#endif

namespace nuka::collision::amf {

using constraint::ContactManifold;
using constraint::ContactPoint;
using math::Vec3;

// ---------------------------------------------------------------------------
// PrimFrame -- a baked world<->local frame for one primitive (HD-clean).
// ---------------------------------------------------------------------------
// Columns are the body axes in world (like SdfBodyFrame). Local->world is a
// column combination; world->local is the orthonormal transpose (a dot per
// column). Lets the HD handlers do OBB math with NO non-HD quaternion helper:
// the host bakes Quat -> columns once (BuildPrimFrame).
struct PrimFrame {
    Vec3 cx{1.0f, 0.0f, 0.0f};  // world-space local X axis
    Vec3 cy{0.0f, 1.0f, 0.0f};  // world-space local Y axis
    Vec3 cz{0.0f, 0.0f, 1.0f};  // world-space local Z axis
    Vec3 t {0.0f, 0.0f, 0.0f};  // world-space origin

    NUKA_AMF_HD Vec3 LocalToWorld(Vec3 l) const {
        return {cx.x * l.x + cy.x * l.y + cz.x * l.z + t.x,
                cx.y * l.x + cy.y * l.y + cz.y * l.z + t.y,
                cx.z * l.x + cy.z * l.y + cz.z * l.z + t.z};
    }
    NUKA_AMF_HD Vec3 WorldToLocal(Vec3 w) const {
        const Vec3 d{w.x - t.x, w.y - t.y, w.z - t.z};
        return {d.x * cx.x + d.y * cx.y + d.z * cx.z,
                d.x * cy.x + d.y * cy.y + d.z * cy.z,
                d.x * cz.x + d.y * cz.y + d.z * cz.z};
    }
    // Rotate a LOCAL direction into world (no translation).
    NUKA_AMF_HD Vec3 DirLocalToWorld(Vec3 l) const {
        return {cx.x * l.x + cy.x * l.y + cz.x * l.z,
                cx.y * l.x + cy.y * l.y + cz.y * l.z,
                cx.z * l.x + cy.z * l.y + cz.z * l.z};
    }
    // World-space axis k in {0,1,2}.
    NUKA_AMF_HD Vec3 Axis(int k) const { return k == 0 ? cx : (k == 1 ? cy : cz); }
};

// Bake a Transform (quaternion + position) into a PrimFrame on the HOST. Uses
// Quat::Rotate (host `inline`) -- this is the host-side fill, NOT device code.
inline PrimFrame BuildPrimFrame(const math::Transform& xf) {
    PrimFrame f;
    f.cx = xf.rotation.Rotate(Vec3::UnitX());
    f.cy = xf.rotation.Rotate(Vec3::UnitY());
    f.cz = xf.rotation.Rotate(Vec3::UnitZ());
    f.t  = xf.position;
    return f;
}

// ---------------------------------------------------------------------------
// DEVICE-SAFE frame baking (WP5/WP6 dog-dog narrowphase). The same math as
// BuildPrimFrame but written HD-clean (no Quat::Rotate, which is host-only):
// rotate the 3 unit axes by the quaternion with the explicit v + 2w(qxv) +
// 2 qx(qxv) formula (the same recipe contacts_foot.cu RotateByQuat uses), so it
// compiles for both host and device. Used by the dog_dog_contact.cu kernel.
// ---------------------------------------------------------------------------
NUKA_AMF_HD inline Vec3 RotateByQuatHD(const math::Quat& q_in, Vec3 v) {
    // Normalize defensively (matches RotateByQuat's 1e-12 guard).
    math::Quat q = q_in;
    const float norm_sq = q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z;
    if (norm_sq > 1.0e-12f) {
        const float inv = 1.0f / sqrtf(norm_sq);
        q.w *= inv; q.x *= inv; q.y *= inv; q.z *= inv;
    }
    const Vec3 qv{q.x, q.y, q.z};
    // t = 2 * (qv x v)
    const Vec3 t{2.0f * (qv.y * v.z - qv.z * v.y),
                 2.0f * (qv.z * v.x - qv.x * v.z),
                 2.0f * (qv.x * v.y - qv.y * v.x)};
    // result = v + w*t + qv x t
    const Vec3 qvt{qv.y * t.z - qv.z * t.y,
                   qv.z * t.x - qv.x * t.z,
                   qv.x * t.y - qv.y * t.x};
    return {v.x + q.w * t.x + qvt.x,
            v.y + q.w * t.y + qvt.y,
            v.z + q.w * t.z + qvt.z};
}

// Hamilton product (w-first), HD-clean (mirrors math::Quat::operator*).
NUKA_AMF_HD inline math::Quat QuatMulHD(const math::Quat& a, const math::Quat& b) {
    math::Quat r;
    r.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    r.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    r.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
    r.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
    return r;
}

// Compose two transforms HD-clean: (parent o child) = apply child first, then
// parent. position = parent.pos + R(parent.rot) * child.pos; rotation = parent.rot
// * child.rot. (math::Transform::operator* is host-only `inline`.)
NUKA_AMF_HD inline math::Transform ComposeTransformHD(const math::Transform& parent,
                                                     const math::Transform& child) {
    math::Transform out;
    out.position = parent.position + RotateByQuatHD(parent.rotation, child.position);
    out.rotation = QuatMulHD(parent.rotation, child.rotation);
    return out;
}

// Bake a Transform into a PrimFrame on host OR device (HD-clean).
NUKA_AMF_HD inline PrimFrame BuildPrimFrameDevice(const math::Transform& xf) {
    PrimFrame f;
    f.cx = RotateByQuatHD(xf.rotation, Vec3{1.0f, 0.0f, 0.0f});
    f.cy = RotateByQuatHD(xf.rotation, Vec3{0.0f, 1.0f, 0.0f});
    f.cz = RotateByQuatHD(xf.rotation, Vec3{0.0f, 0.0f, 1.0f});
    f.t  = xf.position;
    return f;
}

// ---------------------------------------------------------------------------
// HD-clean vector helpers (no Vec3::Length/Normalized -- those are not HD).
// ---------------------------------------------------------------------------
NUKA_AMF_HD inline float Len(Vec3 v) {
    return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}
// Component access by axis index k in {0,1,2} (Vec3 has no operator[]).
NUKA_AMF_HD inline float Comp(Vec3 v, int k) {
    return k == 0 ? v.x : (k == 1 ? v.y : v.z);
}
// Normalize with a fallback for near-zero input (matches Vec3::Normalized, HD).
NUKA_AMF_HD inline Vec3 Norm(Vec3 v, Vec3 fallback) {
    const float len = Len(v);
    if (len < 1.0e-12f) return fallback;
    return {v.x / len, v.y / len, v.z / len};
}

// ---------------------------------------------------------------------------
// Sutherland-Hodgman polygon clip (REUSED by C3c convex narrowphase).
// ---------------------------------------------------------------------------
// Clips a polygon `poly` (count <= kClipMax verts, CCW or CW -- order preserved)
// against a single half-space {p : (p - plane_pt).dot(plane_n) <= 0} (KEEP the
// side the normal points AWAY from -> "inside" = behind the plane). FIXED edge
// order (i -> i+1, wrapping), so the FP accumulation order is data-independent
// (D1). Writes the clipped polygon into `out`, returns its vertex count.
//
// Carries a parallel `feat` array (a per-vertex feature id) so the box-box face
// clip can track which incident-face vertex produced each survivor -- intersection
// vertices inherit the id of the edge's ORIGIN vertex (fixed rule -> D1).
inline constexpr int kClipMax = 8;  // 4 input + up to 4 new intersections

struct ClipPoly {
    Vec3      v[kClipMax];
    uint32_t  feat[kClipMax];
    int       count = 0;
    NUKA_AMF_HD void Add(Vec3 p, uint32_t f) {
        if (count < kClipMax) { v[count] = p; feat[count] = f; ++count; }
    }
};

NUKA_AMF_HD inline ClipPoly ClipAgainstPlane(const ClipPoly& in, Vec3 plane_pt,
                                             Vec3 plane_n) {
    ClipPoly out;
    if (in.count == 0) return out;
    // Signed distances (positive == OUTSIDE the half-space we discard).
    for (int i = 0; i < in.count; ++i) {
        const int j = (i + 1) % in.count;           // FIXED edge order
        const Vec3 cur = in.v[i];
        const Vec3 nxt = in.v[j];
        const float dc = (cur - plane_pt).Dot(plane_n);
        const float dn = (nxt - plane_pt).Dot(plane_n);
        const bool cur_in = dc <= 0.0f;
        const bool nxt_in = dn <= 0.0f;
        if (cur_in) {
            out.Add(cur, in.feat[i]);               // keep inside vertex
        }
        if (cur_in != nxt_in) {
            // Edge crosses the plane: add the intersection. dc - dn != 0 here.
            const float t = dc / (dc - dn);
            const Vec3 ip{cur.x + (nxt.x - cur.x) * t,
                          cur.y + (nxt.y - cur.y) * t,
                          cur.z + (nxt.z - cur.z) * t};
            out.Add(ip, in.feat[i]);                // inherit ORIGIN feature id
        }
    }
    return out;
}

// ---------------------------------------------------------------------------
// Deepest-4 reduction (the determinism-trap-safe selection).
// ---------------------------------------------------------------------------
// From `n` (<= kClipMax) candidate points with parallel penetration + feature
// id, select <=4 and write them into the manifold in a FIXED slot order. D1 key:
//   PRIMARY  : feature id (uint32) ASCENDING  -- the SOLE determinant on ties.
//   (penetration is NOT used as an ordering key: in the symmetric resting case
//    all penetrations tie, so a float key would be nondeterministic. The feature
//    id is unique per surviving clip vertex, so it fully orders the set.)
// When n > 4 we keep the 4 with the LARGEST penetration, but break penetration
// ties by SMALLEST feature id, then emit those 4 in ASCENDING feature-id order.
// stable_key = feature id (warm-start identity).
//
// `normal` is the (already correctly-signed for side A) contact normal, applied
// to every emitted point.
struct ManifoldPointCand {
    Vec3     position;
    float    penetration;
    uint32_t feature;
};

NUKA_AMF_HD inline void ReduceAndEmit(ManifoldPointCand* cand, int n, Vec3 normal,
                                      ContactManifold* out) {
    if (n <= 0) return;
    // 1) Selection: if more than 4, mark the 4 deepest (tie -> smallest feature).
    //    Selection sort by (penetration DESC, feature ASC) for the first
    //    min(n,4) slots -- fixed iteration, O(n^2) on n<=8 (cheap, branch-stable).
    const int keep = n < 4 ? n : 4;
    for (int s = 0; s < keep; ++s) {
        int best = s;
        for (int i = s + 1; i < n; ++i) {
            const bool deeper  = cand[i].penetration >  cand[best].penetration;
            const bool tie_pen = cand[i].penetration == cand[best].penetration;
            const bool smaller_feat = cand[i].feature < cand[best].feature;
            if (deeper || (tie_pen && smaller_feat)) best = i;
        }
        // swap s,best (fixed-order)
        ManifoldPointCand tmp = cand[s]; cand[s] = cand[best]; cand[best] = tmp;
    }
    // 2) Emit order: ASCENDING feature id among the kept `keep` (insertion sort,
    //    fixed iteration) -> deterministic slot order independent of selection.
    for (int i = 1; i < keep; ++i) {
        ManifoldPointCand key = cand[i];
        int j = i - 1;
        while (j >= 0 && cand[j].feature > key.feature) {
            cand[j + 1] = cand[j];
            --j;
        }
        cand[j + 1] = key;
    }
    out->Clear();
    for (int i = 0; i < keep; ++i) {
        ContactPoint pt;
        pt.position    = cand[i].position;
        pt.normal      = normal;
        pt.penetration = cand[i].penetration;
        pt.stable_key  = static_cast<uint64_t>(cand[i].feature);
        out->AddPoint(pt);
    }
}

// ---------------------------------------------------------------------------
// Box-box face-clip reduction: KEEP-DEEPEST-THEN-SPREAD (farthest-point sampling).
// ---------------------------------------------------------------------------
// The deepest-4-by-penetration ReduceAndEmit above is WRONG for a face-clip
// octagon: in the symmetric resting case ALL clip points tie on penetration, so
// "take 4 deepest" keeps 4 CONSECUTIVE clip verts (a cluster on two adjacent
// edges) -- no spread -> no resting torque support -> jitter. The standard fix
// (Bullet btPersistentManifold) is farthest-point sampling:
//   1) seed = the DEEPEST point (tie -> lowest clip index).
//   2) repeat: add the candidate whose MIN distance to the already-kept set is
//      LARGEST (tie -> lowest clip index). -> a maximally-spread <=4 patch.
// D1: the float distance is tie-broken by the unique clip-output index (the
// pattern the SAT axis pick already uses: strict `>` improvement, lowest index
// wins). NO float key reorders on a tie. stable_key = clip-output index (unique
// per survivor, deterministic; per-point identity for C3b -- NOT a durable
// cross-frame warm-start id, see header note / C3c durable-feature follow-up).
//
// NOTE (scope): the task text suggests "take the first 4 by (penetration, fixed
// index)"; that is exactly the algorithm that CLUSTERS the symmetric octagon.
// The determinism-trap GATE ("symmetric case -> stable AND matches the oracle
// SET") requires the spread square, so box-box is upgraded to FPS. The shared
// clip+reduce util also feeds C3c (general convex), which needs spread for grasp
// stability. ReduceAndEmit (deepest-by-penetration) is RETAINED unchanged for
// box-plane, where deepest selection is correct (grab the bottom face corners).
NUKA_AMF_HD inline void ReduceAndEmitSpread(ManifoldPointCand* cand, int n,
                                            Vec3 normal, ContactManifold* out) {
    out->Clear();
    if (n <= 0) return;
    if (n <= 4) {
        // <=4: emit all, in clip-index (feature) ascending order for D1 slots.
        for (int i = 1; i < n; ++i) {  // insertion sort by feature
            ManifoldPointCand key = cand[i];
            int j = i - 1;
            while (j >= 0 && cand[j].feature > key.feature) { cand[j+1]=cand[j]; --j; }
            cand[j + 1] = key;
        }
        for (int i = 0; i < n; ++i) {
            ContactPoint pt;
            pt.position = cand[i].position; pt.normal = normal;
            pt.penetration = cand[i].penetration;
            pt.stable_key = static_cast<uint64_t>(cand[i].feature);
            out->AddPoint(pt);
        }
        return;
    }
    // n > 4: farthest-point sampling.
    int kept[4];
    bool used[kClipMax] = {false};
    // 1) seed = deepest (tie -> lowest index, via strict > improvement).
    int seed = 0;
    for (int i = 1; i < n; ++i) {
        if (cand[i].penetration > cand[seed].penetration) seed = i;
    }
    kept[0] = seed; used[seed] = true;
    // 2) iteratively add the farthest-from-kept-set candidate.
    for (int s = 1; s < 4; ++s) {
        int best = -1; float best_mind = -1.0f;
        for (int i = 0; i < n; ++i) {
            if (used[i]) continue;
            // min distance from cand[i] to the current kept set.
            float mind = 3.4e38f;
            for (int t = 0; t < s; ++t) {
                const Vec3 d = cand[i].position - cand[kept[t]].position;
                const float dd = d.Dot(d);            // squared distance (cheaper)
                if (dd < mind) mind = dd;
            }
            // strict improvement -> first (lowest) index wins ties (D1).
            if (mind > best_mind) { best_mind = mind; best = i; }
        }
        if (best < 0) break;
        kept[s] = best; used[best] = true;
    }
    // 3) emit the kept set in ASCENDING clip-index order (deterministic slots).
    //    (kept[] currently in selection order; sort the small array by feature.)
    for (int i = 1; i < 4; ++i) {
        int key = kept[i]; int j = i - 1;
        while (j >= 0 && cand[kept[j]].feature > cand[key].feature) {
            kept[j + 1] = kept[j]; --j;
        }
        kept[j + 1] = key;
    }
    for (int i = 0; i < 4; ++i) {
        ContactPoint pt;
        pt.position = cand[kept[i]].position; pt.normal = normal;
        pt.penetration = cand[kept[i]].penetration;
        pt.stable_key = static_cast<uint64_t>(cand[kept[i]].feature);
        out->AddPoint(pt);
    }
}

// ===========================================================================
// PRIMITIVE PARAMS carried in ShapeProxyView (filled by the caller from SoA).
// ===========================================================================
struct PrimParams {
    Vec3      half_extents{0.5f, 0.5f, 0.5f};  // Box
    float     radius      = 0.5f;              // Sphere / Capsule
    float     half_height = 0.5f;              // Capsule (along local Y)
    PrimFrame frame;                           // baked world frame
};

// ---------------------------------------------------------------------------
// SPHERE x SPHERE -> 1 point. normal = separation dir for A (a's center - b's).
// ---------------------------------------------------------------------------
NUKA_AMF_HD inline void SphereSphere(const PrimParams& a, const PrimParams& b,
                                     ContactManifold* out) {
    out->Clear();
    const Vec3 delta = a.frame.t - b.frame.t;          // A relative to B
    const float dist = Len(delta);
    const float pen = (a.radius + b.radius) - dist;
    if (pen <= 0.0f) return;
    const Vec3 n = Norm(delta, Vec3::UnitY());         // points A away from B
    const Vec3 p = b.frame.t + n * b.radius;           // on B's surface
    ContactPoint pt;
    pt.position = p; pt.normal = n; pt.penetration = pen; pt.stable_key = 0ull;
    out->AddPoint(pt);
}

// ---------------------------------------------------------------------------
// SPHERE (a) x BOX (b) -> 1 point. OBB clamp in box-LOCAL frame (not AABB).
// normal = separation dir for A (sphere). 1-pt.
// ---------------------------------------------------------------------------
NUKA_AMF_HD inline void SphereBox(const PrimParams& sph, const PrimParams& box,
                                  ContactManifold* out) {
    out->Clear();
    const Vec3 cl = box.frame.WorldToLocal(sph.frame.t);     // sphere center, box-local
    const Vec3 he = box.half_extents;
    const Vec3 q{cl.x < -he.x ? -he.x : (cl.x > he.x ? he.x : cl.x),
                 cl.y < -he.y ? -he.y : (cl.y > he.y ? he.y : cl.y),
                 cl.z < -he.z ? -he.z : (cl.z > he.z ? he.z : cl.z)};
    const Vec3 closest_world = box.frame.LocalToWorld(q);
    const Vec3 delta = sph.frame.t - closest_world;          // sphere away from box
    const float dist = Len(delta);
    const float pen = sph.radius - dist;
    if (pen <= 0.0f) return;
    const Vec3 n = Norm(delta, Vec3::UnitY());               // separation dir for sphere(=A)
    ContactPoint pt;
    pt.position = closest_world; pt.normal = n; pt.penetration = pen;
    pt.stable_key = 0ull;
    out->AddPoint(pt);
}

// ---------------------------------------------------------------------------
// SPHERE (a) x PLANE (b) -> 1 point. Plane normal = b's local +Y in world.
// normal = separation dir for A (sphere) = plane normal (sphere pushed up off).
// ---------------------------------------------------------------------------
NUKA_AMF_HD inline void SpherePlane(const PrimParams& sph, const PrimParams& plane,
                                    ContactManifold* out) {
    out->Clear();
    const Vec3 n = Norm(plane.frame.cy, Vec3::UnitY());      // plane normal (world)
    const float signed_dist = (sph.frame.t - plane.frame.t).Dot(n);
    const float pen = sph.radius - signed_dist;
    if (pen <= 0.0f) return;
    const Vec3 p = sph.frame.t - n * sph.radius;             // sphere bottom
    ContactPoint pt;
    pt.position = p; pt.normal = n; pt.penetration = pen; pt.stable_key = 0ull;
    out->AddPoint(pt);
}

// ---------------------------------------------------------------------------
// BOX (a) x PLANE (b) -> up to 4 deepest corners below the plane (4-pt patch).
// normal = separation dir for A (box) = plane normal (box pushed up). feature id
// = the corner's signed-axis bitmask (0..7), deterministic + warm-start identity.
// ---------------------------------------------------------------------------
NUKA_AMF_HD inline void BoxPlane(const PrimParams& box, const PrimParams& plane,
                                 ContactManifold* out) {
    out->Clear();
    const Vec3 n = Norm(plane.frame.cy, Vec3::UnitY());      // plane normal (world)
    const Vec3 he = box.half_extents;
    ManifoldPointCand cand[8];
    int ncand = 0;
    // 8 corners in a FIXED order: bit0=x sign, bit1=y sign, bit2=z sign.
    for (uint32_t c = 0; c < 8u; ++c) {
        const float sx = (c & 1u) ? 1.0f : -1.0f;
        const float sy = (c & 2u) ? 1.0f : -1.0f;
        const float sz = (c & 4u) ? 1.0f : -1.0f;
        const Vec3 corner_world =
            box.frame.LocalToWorld({sx * he.x, sy * he.y, sz * he.z});
        const float signed_dist = (corner_world - plane.frame.t).Dot(n);
        if (signed_dist < 0.0f) {                            // below plane -> overlap
            cand[ncand].position    = corner_world;
            cand[ncand].penetration = -signed_dist;          // >0 == depth
            cand[ncand].feature     = c;                     // 0..7 corner id
            ++ncand;
        }
    }
    ReduceAndEmit(cand, ncand, n, out);
}

// ---------------------------------------------------------------------------
// BOX x BOX -> SAT min-penetration axis + face-clip (Sutherland-Hodgman),
// reduced to <=4 by feature id. normal = separation dir for A.
// ---------------------------------------------------------------------------
// D1: the 15 SAT axes are tested in a FIXED order (A's 3 faces, B's 3 faces, then
// the 9 edge-edge crosses). The first axis WINS ties (we only replace on strict
// improvement), so a symmetric stack picks the SAME axis every run. Face axes are
// preferred over edge axes by an EPS bias (btBoxBox trick) so a parallel-face
// resting stack uses face-clip (4 pts), not a 1-pt edge result.
NUKA_AMF_HD inline void BoxBox(const PrimParams& A, const PrimParams& B,
                               ContactManifold* out) {
    out->Clear();
    const Vec3 ha = A.half_extents;
    const Vec3 hb = B.half_extents;
    const Vec3 dpos = B.frame.t - A.frame.t;   // B center relative to A center

    // Cache the 3x3 relative rotation R[i][j] = A.axis(i) . B.axis(j) and abs.
    Vec3 ax[3] = {A.frame.Axis(0), A.frame.Axis(1), A.frame.Axis(2)};
    Vec3 bx[3] = {B.frame.Axis(0), B.frame.Axis(1), B.frame.Axis(2)};
    float R[3][3], AbsR[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            R[i][j] = ax[i].Dot(bx[j]);
            AbsR[i][j] = (R[i][j] < 0.0f ? -R[i][j] : R[i][j]) + 1.0e-6f;  // EPS for parallel
        }
    const float ea[3] = {ha.x, ha.y, ha.z};
    const float eb[3] = {hb.x, hb.y, hb.z};
    // d in A's frame.
    const float dA[3] = {dpos.Dot(ax[0]), dpos.Dot(ax[1]), dpos.Dot(ax[2])};

    float min_pen = 3.4e38f;
    int   best_axis = -1;        // 0..2 = A face, 3..5 = B face, 6..14 = edge cross
    Vec3  best_n{0.0f, 1.0f, 0.0f};
    const float kFaceBias = 1.0e-5f;  // prefer face axes over edge axes on tie

    // Helper lambda-free axis test (HD): given a world axis L (need not be unit
    // for projection ratio? -> we DO normalize so penetration is a true length).
    // We compute projected radii using the relative-rotation form for face axes
    // (exact, no per-axis renorm), and explicit projection for edge axes.

    // --- A's 3 face axes (i) ---
    for (int i = 0; i < 3; ++i) {
        const float ra = ea[i];
        const float rb = eb[0]*AbsR[i][0] + eb[1]*AbsR[i][1] + eb[2]*AbsR[i][2];
        const float s = dA[i] < 0.0f ? -dA[i] : dA[i];   // |projection of d|
        const float pen = (ra + rb) - s;
        if (pen <= 0.0f) return;                           // separating axis found
        if (pen < min_pen - kFaceBias) {
            min_pen = pen; best_axis = i;
            best_n = (dA[i] < 0.0f) ? -ax[i] : ax[i];      // dir A away from B
        }
    }
    // --- B's 3 face axes (j) ---
    for (int j = 0; j < 3; ++j) {
        const float ra = ea[0]*AbsR[0][j] + ea[1]*AbsR[1][j] + ea[2]*AbsR[2][j];
        const float rb = eb[j];
        const float dBj = dpos.Dot(bx[j]);
        const float s = dBj < 0.0f ? -dBj : dBj;
        const float pen = (ra + rb) - s;
        if (pen <= 0.0f) return;
        if (pen < min_pen - kFaceBias) {
            min_pen = pen; best_axis = 3 + j;
            best_n = (dBj < 0.0f) ? -bx[j] : bx[j];
        }
    }
    // --- 9 edge-edge cross axes (i x j) ---  (FIXED order i outer, j inner)
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            Vec3 L = ax[i].Cross(bx[j]);
            const float Llen = Len(L);
            if (Llen < 1.0e-6f) continue;                  // parallel edges -> skip
            L = L / Llen;
            const float ra = ea[0]*( (ax[0].Dot(L)<0?-ax[0].Dot(L):ax[0].Dot(L)) )
                           + ea[1]*( (ax[1].Dot(L)<0?-ax[1].Dot(L):ax[1].Dot(L)) )
                           + ea[2]*( (ax[2].Dot(L)<0?-ax[2].Dot(L):ax[2].Dot(L)) );
            const float rb = eb[0]*( (bx[0].Dot(L)<0?-bx[0].Dot(L):bx[0].Dot(L)) )
                           + eb[1]*( (bx[1].Dot(L)<0?-bx[1].Dot(L):bx[1].Dot(L)) )
                           + eb[2]*( (bx[2].Dot(L)<0?-bx[2].Dot(L):bx[2].Dot(L)) );
            const float s = dpos.Dot(L);
            const float sa = s < 0.0f ? -s : s;
            const float pen = (ra + rb) - sa;
            if (pen <= 0.0f) return;
            // Edge axes need a STRICT improvement beyond kFaceBias to beat a face.
            if (pen < min_pen - kFaceBias) {
                min_pen = pen; best_axis = 6 + i * 3 + j;
                best_n = (s < 0.0f) ? -L : L;              // dir A away from B
            }
        }
    }
    if (best_axis < 0) return;

    // --- Edge-edge winner: 1-point closest-segment contact. ---
    if (best_axis >= 6) {
        const int i = (best_axis - 6) / 3;
        const int j = (best_axis - 6) % 3;
        // Closest points between A's edge dir ax[i] and B's edge dir bx[j].
        // Pick the edge midpoints on each box along the OTHER two axes by the
        // sign of the separation normal (standard box edge selection).
        Vec3 pa = A.frame.t;
        for (int k = 0; k < 3; ++k) {
            if (k == i) continue;
            const float sgn = ax[k].Dot(best_n) >= 0.0f ? 1.0f : -1.0f;
            pa += ax[k] * (sgn * ea[k]);
        }
        Vec3 pb = B.frame.t;
        for (int k = 0; k < 3; ++k) {
            if (k == j) continue;
            const float sgn = bx[k].Dot(best_n) >= 0.0f ? -1.0f : 1.0f;
            pb += bx[k] * (sgn * eb[k]);
        }
        // Closest points on the two infinite lines (pa + s*ea_dir, pb + t*eb_dir).
        const Vec3 ea_dir = ax[i], eb_dir = bx[j];
        const Vec3 r = pa - pb;
        const float aa = 1.0f, bb = ea_dir.Dot(eb_dir), cc = 1.0f;
        const float dd = ea_dir.Dot(r), ee = eb_dir.Dot(r);
        const float denom = aa * cc - bb * bb;
        float sN = 0.0f, tN = 0.0f;
        if (denom > 1.0e-9f) {
            sN = (bb * ee - cc * dd) / denom;
            tN = (aa * ee - bb * dd) / denom;
        }
        const float sCl = sN < -ea[i] ? -ea[i] : (sN > ea[i] ? ea[i] : sN);
        const float tCl = tN < -eb[j] ? -eb[j] : (tN > eb[j] ? eb[j] : tN);
        const Vec3 ca = pa + ea_dir * sCl;
        const Vec3 cb = pb + eb_dir * tCl;
        ContactPoint pt;
        pt.position = (ca + cb) * 0.5f;
        pt.normal = best_n;
        pt.penetration = min_pen;
        pt.stable_key = 0xE000'0000ull | (static_cast<uint64_t>(i) << 4) | j;
        out->AddPoint(pt);
        return;
    }

    // --- Face winner: identify reference box (R) + incident box (I). ---
    // The reference face is the winning face axis's face on its box; the
    // incident face is the most anti-parallel face on the other box.
    const bool ref_is_A = best_axis < 3;
    const int ref_axis = ref_is_A ? best_axis : (best_axis - 3);
    // Frames/params for ref + inc.
    const PrimFrame& Rf = ref_is_A ? A.frame : B.frame;
    const PrimFrame& If = ref_is_A ? B.frame : A.frame;
    const Vec3 rhe = ref_is_A ? ha : hb;   // reference half-extents
    const Vec3 ihe = ref_is_A ? hb : ha;   // incident half-extents

    // Reference face outward normal (world), pointing from ref toward inc.
    Vec3 refN = Rf.Axis(ref_axis);
    const Vec3 ref_to_inc = If.t - Rf.t;
    if (ref_to_inc.Dot(refN) < 0.0f) refN = -refN;

    // Incident box: the face whose normal is MOST anti-parallel to refN.
    int inc_axis = 0; float best_dot = 3.4e38f; float inc_sign = 1.0f;
    for (int k = 0; k < 3; ++k) {
        const float d = If.Axis(k).Dot(refN);
        const float ad = d < 0.0f ? -d : d;
        // most anti-parallel == largest |d|; among those choose; FIXED order ties
        if (-ad < best_dot) { best_dot = -ad; inc_axis = k; inc_sign = d < 0.0f ? 1.0f : -1.0f; }
    }
    // Incident face center + its 4 corners (the face pointing toward ref).
    // The incident face is at +inc_sign*ihe[inc_axis] along inc_axis. Its two
    // in-plane axes are the other two incident axes, FIXED order (u then v).
    int iu = (inc_axis + 1) % 3, iv = (inc_axis + 2) % 3;
    const Vec3 inc_face_center =
        If.t + If.Axis(inc_axis) * (inc_sign * Comp(ihe, inc_axis));
    const Vec3 iu_w = If.Axis(iu), iv_w = If.Axis(iv);
    const float iu_e = Comp(ihe, iu), iv_e = Comp(ihe, iv);
    // 4 incident-face corners in a FIXED order; feature id packs (inc_axis, k).
    ClipPoly poly;
    const float su[4] = {-1.0f, +1.0f, +1.0f, -1.0f};
    const float sv[4] = {-1.0f, -1.0f, +1.0f, +1.0f};
    for (uint32_t k = 0; k < 4u; ++k) {
        const Vec3 corner = inc_face_center + iu_w * (su[k] * iu_e)
                                            + iv_w * (sv[k] * iv_e);
        const uint32_t feat = (static_cast<uint32_t>(inc_axis) << 4) | k;
        poly.Add(corner, feat);
    }

    // Reference face: clip against its 4 side planes. The reference face's two
    // in-plane axes are the other two ref axes (FIXED order). Each side plane
    // passes through the ref-face extent, normal pointing OUTWARD; we keep the
    // half-space BEHIND it (Sutherland-Hodgman discards the outside).
    int ru = (ref_axis + 1) % 3, rv = (ref_axis + 2) % 3;
    const Vec3 ru_w = Rf.Axis(ru), rv_w = Rf.Axis(rv);
    const float ru_e = Comp(rhe, ru), rv_e = Comp(rhe, rv);
    const Vec3 ref_face_center = Rf.t + refN * Comp(rhe, ref_axis);
    // 4 side planes in FIXED order: +u, -u, +v, -v.
    {
        const Vec3 pu = ref_face_center + ru_w * ru_e;  // +u side
        poly = ClipAgainstPlane(poly, pu, ru_w);
        const Vec3 mu = ref_face_center - ru_w * ru_e;  // -u side
        poly = ClipAgainstPlane(poly, mu, -ru_w);
        const Vec3 pv = ref_face_center + rv_w * rv_e;  // +v side
        poly = ClipAgainstPlane(poly, pv, rv_w);
        const Vec3 mv = ref_face_center - rv_w * rv_e;  // -v side
        poly = ClipAgainstPlane(poly, mv, -rv_w);
    }

    // Keep only points BELOW the reference face (penetrating); project onto the
    // ref face for the contact position; penetration = depth below ref face.
    // Feature id = the clip-OUTPUT index k (0..count-1): UNIQUE per survivor and
    // deterministic (fixed clip iteration). The incident-vertex feat (poly.feat)
    // is NOT unique after clipping -- intersection verts inherit their edge's
    // origin id, so an octagon carries only 4 distinct origin ids (clustered +
    // duplicate keys). The clip-output index is the D1-safe unique key here.
    ManifoldPointCand cand[kClipMax];
    int ncand = 0;
    for (int k = 0; k < poly.count; ++k) {
        const float depth = (ref_face_center - poly.v[k]).Dot(refN);  // >0 below face
        if (depth >= 0.0f) {
            // contact point: the incident-face vertex projected onto ref face.
            const Vec3 onref = poly.v[k] + refN * depth;
            cand[ncand].position    = (poly.v[k] + onref) * 0.5f;  // midpoint (stable)
            cand[ncand].penetration = depth;
            cand[ncand].feature     = static_cast<uint32_t>(k);    // unique clip index
            ++ncand;
        }
    }

    // Normal for side A: refN points ref->inc. If ref IS A, A must move AWAY from
    // B == along -refN... but our convention is "separation dir for A". refN
    // points from ref toward inc; the separation dir for the reference box is
    // -refN (push ref away from inc). For side A:
    //   ref_is_A: A==ref, separation for A = -refN
    //   !ref_is_A: A==inc, separation for A = +refN (push inc away from ref)
    const Vec3 normalA = ref_is_A ? (-refN) : refN;

    if (ncand == 0) {
        // Degenerate (no surviving penetrating point): fall back to 1-pt at the
        // deepest incident corner along refN (rare; keeps a non-empty manifold).
        // Use the incident face center projected; feature = inc_axis high bits.
        const float depth = min_pen;
        ContactPoint pt;
        pt.position = inc_face_center + refN * (depth * 0.5f);
        pt.normal = normalA;
        pt.penetration = depth;
        pt.stable_key = (static_cast<uint64_t>(inc_axis) << 4);
        out->AddPoint(pt);
        return;
    }
    ReduceAndEmitSpread(cand, ncand, normalA, out);  // box-box: keep-deepest+spread
}

// ---------------------------------------------------------------------------
// CAPSULE x PLANE -> up to 2 points (the two endpoints below the plane).
// Capsule axis = local Y (half_height). normal = separation dir for A (capsule).
// ---------------------------------------------------------------------------
NUKA_AMF_HD inline void CapsulePlane(const PrimParams& cap, const PrimParams& plane,
                                     ContactManifold* out) {
    out->Clear();
    const Vec3 n = Norm(plane.frame.cy, Vec3::UnitY());      // plane normal (world)
    const Vec3 axis = cap.frame.cy;                          // capsule local Y in world
    const Vec3 e0 = cap.frame.t + axis * cap.half_height;
    const Vec3 e1 = cap.frame.t - axis * cap.half_height;
    const Vec3 ends[2] = {e0, e1};
    ManifoldPointCand cand[2];
    int ncand = 0;
    for (uint32_t k = 0; k < 2u; ++k) {
        const float signed_dist = (ends[k] - plane.frame.t).Dot(n) - cap.radius;
        if (signed_dist < 0.0f) {
            cand[ncand].position    = ends[k] - n * cap.radius;  // surface point
            cand[ncand].penetration = -signed_dist;
            cand[ncand].feature     = k;                          // endpoint id
            ++ncand;
        }
    }
    ReduceAndEmit(cand, ncand, n, out);
}

// ---------------------------------------------------------------------------
// CAPSULE (a) x SPHERE (b) -> 1 point. Closest point on capsule segment to the
// sphere center. normal = separation dir for A (capsule).
// ---------------------------------------------------------------------------
NUKA_AMF_HD inline void CapsuleSphere(const PrimParams& cap, const PrimParams& sph,
                                      ContactManifold* out) {
    out->Clear();
    const Vec3 axis = cap.frame.cy;
    const Vec3 seg_a = cap.frame.t - axis * cap.half_height;
    const Vec3 ab = axis * (2.0f * cap.half_height);
    const float ab_len2 = ab.Dot(ab);
    float tparam = 0.0f;
    if (ab_len2 > 1.0e-12f) {
        tparam = (sph.frame.t - seg_a).Dot(ab) / ab_len2;
        tparam = tparam < 0.0f ? 0.0f : (tparam > 1.0f ? 1.0f : tparam);
    }
    const Vec3 closest = seg_a + ab * tparam;                // on capsule axis
    const Vec3 delta = closest - sph.frame.t;                // capsule away from sphere
    const float dist = Len(delta);
    const float pen = (cap.radius + sph.radius) - dist;
    if (pen <= 0.0f) return;
    const Vec3 n = Norm(delta, Vec3::UnitY());               // separation dir for capsule(=A)
    const Vec3 p = closest - n * cap.radius;                 // on capsule surface
    ContactPoint pt;
    pt.position = p; pt.normal = n; pt.penetration = pen; pt.stable_key = 0ull;
    out->AddPoint(pt);
}

// ---------------------------------------------------------------------------
// Closest points between two segments [p1,q1] and [p2,q2] (Ericson, Real-Time
// Collision Detection 5.1.9, HD-clean). Returns the two closest points via
// out_c1/out_c2 and the squared distance. Deterministic: a FIXED branch order,
// no data-dependent reordering, so the FP accumulation matches host==device.
// ---------------------------------------------------------------------------
NUKA_AMF_HD inline float ClosestPtSegmentSegment(Vec3 p1, Vec3 q1, Vec3 p2,
                                                 Vec3 q2, Vec3* out_c1,
                                                 Vec3* out_c2) {
    const Vec3 d1 = q1 - p1;          // direction of segment S1
    const Vec3 d2 = q2 - p2;          // direction of segment S2
    const Vec3 r  = p1 - p2;
    const float a = d1.Dot(d1);       // squared length of S1
    const float e = d2.Dot(d2);       // squared length of S2
    const float f = d2.Dot(r);
    float s = 0.0f;
    float t = 0.0f;
    constexpr float kEps = 1.0e-12f;
    if (a <= kEps && e <= kEps) {
        // Both segments degenerate to points.
        s = 0.0f; t = 0.0f;
    } else if (a <= kEps) {
        // First segment degenerates to a point.
        s = 0.0f;
        t = f / e;
        t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    } else {
        const float c = d1.Dot(r);
        if (e <= kEps) {
            // Second segment degenerates to a point.
            t = 0.0f;
            s = -c / a;
            s = s < 0.0f ? 0.0f : (s > 1.0f ? 1.0f : s);
        } else {
            // The general nondegenerate case.
            const float b = d1.Dot(d2);
            const float denom = a * e - b * b;   // always >= 0
            if (denom > kEps) {
                s = (b * f - c * e) / denom;
                s = s < 0.0f ? 0.0f : (s > 1.0f ? 1.0f : s);
            } else {
                s = 0.0f;  // parallel segments: pick s = 0 (deterministic).
            }
            t = (b * s + f) / e;
            if (t < 0.0f) {
                t = 0.0f;
                s = -c / a;
                s = s < 0.0f ? 0.0f : (s > 1.0f ? 1.0f : s);
            } else if (t > 1.0f) {
                t = 1.0f;
                s = (b - c) / a;
                s = s < 0.0f ? 0.0f : (s > 1.0f ? 1.0f : s);
            }
        }
    }
    const Vec3 c1 = p1 + d1 * s;
    const Vec3 c2 = p2 + d2 * t;
    if (out_c1) *out_c1 = c1;
    if (out_c2) *out_c2 = c2;
    const Vec3 dv = c1 - c2;
    return dv.Dot(dv);
}

// ---------------------------------------------------------------------------
// CAPSULE (a) x CAPSULE (b) -> 1 point. Closest segment-to-segment between the
// two capsule axes; inflate by the two radii. normal = separation dir for A
// (capsule a). The single deepest point is emitted at the midpoint of the two
// closest surface points (matches the SphereSphere convention applied to the
// closest features). This is the WP6 dog-dog leg/trunk narrowphase.
// ---------------------------------------------------------------------------
NUKA_AMF_HD inline void CapsuleCapsule(const PrimParams& a, const PrimParams& b,
                                       ContactManifold* out) {
    out->Clear();
    const Vec3 axa = a.frame.cy;                  // capsule a local Y in world
    const Vec3 axb = b.frame.cy;
    const Vec3 a0 = a.frame.t + axa * a.half_height;
    const Vec3 a1 = a.frame.t - axa * a.half_height;
    const Vec3 b0 = b.frame.t + axb * b.half_height;
    const Vec3 b1 = b.frame.t - axb * b.half_height;
    Vec3 ca;
    Vec3 cb;
    const float dist2 = ClosestPtSegmentSegment(a0, a1, b0, b1, &ca, &cb);
    const float dist = sqrtf(dist2);
    const float pen = (a.radius + b.radius) - dist;
    if (pen <= 0.0f) return;
    const Vec3 delta = ca - cb;                   // a away from b
    // Fallback normal: if the axes intersect (dist ~ 0), use a's axis-perp guess.
    const Vec3 n = Norm(delta, Norm(axa.Cross(axb), Vec3::UnitY()));
    // Contact point: the midpoint of the two surface points along the normal.
    const Vec3 sa = ca - n * a.radius;            // a's surface point
    const Vec3 sb = cb + n * b.radius;            // b's surface point
    ContactPoint pt;
    pt.position = (sa + sb) * 0.5f;
    pt.normal = n;
    pt.penetration = pen;
    pt.stable_key = 0ull;
    out->AddPoint(pt);
}

} // namespace nuka::collision::amf
