// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — general contact pipeline Phase 2 (H3): the GENERAL
// heightfield narrowphase (the per-cell TRIANGLE_PRISM midphase).
//
// THE ONE PRINCIPLE: a heightfield is NOT a special terrain contact — it is a
// data-driven set of collidable triangles that ride the SAME cvx GJK/EPA
// narrowphase + the SAME mixed-island solver every other body uses. This op
// turns each overlapped grid cell into 2 downward-solid TRIANGLE_PRISM colliders
// and routes each through cvx::ConvexNarrowphase (the EXISTING device GJK/EPA +
// face-clip), exactly like a box-vs-hull pair. No SampleTerrainHeight in any
// kernel: the prism corner heights come from the COOKED `heights` grid (H2).
//
// FLOW (one thread per (env x candidate slot)):
//   1. Read the broadphase candidate (a,b). If neither side is the heightfield
//      collidable (kind == Heightfield), the slot is not ours -> leave it (the
//      analytic/cvx NarrowphasePrimitives already handled non-heightfield pairs).
//   2. The OTHER side is the convex (box / sphere / capsule / hull). Build its
//      cvx::SupportProxy (reusing the in-hand world PrimParams / cooked hull pool).
//   3. Project the convex's world AABB into heightfield-LOCAL space, map to the
//      overlapped cell range (Newton heightfield.py:288-390 OBB->AABB->cell walk).
//   4. For EACH overlapped cell, for EACH of its 2 triangles, build the prism
//      ConvexHullView (3 LOCAL corners + the heightfield's world frame + a -Z
//      extrude through the field's z-range so the prism is SOLID), run
//      cvx::ConvexNarrowphase(convex, prism), and collect every contact point
//      (each keeps its OWN world position/normal/depth) into a fixed candidate
//      buffer.  A body straddling a STEP picks up tread-normal contacts from the
//      upper-tread prisms AND riser-normal contacts from the riser prisms.
//   5. Farthest-point-spread reduce to <=4 points (PER-POINT normals preserved,
//      so tread + riser both survive) and write them into THIS slot's ucontact_*
//      with side a = the convex body, side b = the heightfield (static, body_id
//      == -1 -> the assembly resolves only the convex's reaction side).
//
// The corner-height extraction mirrors Newton get_triangle_shape_from_heightfield
// (heightfield.py:204-287): cell (r,c) corners p00/p10/p01/p11 at LOCAL
// (origin + c*cell, origin + r*cell, min_z + h*range); tri 0 = (p00,p10,p11),
// tri 1 = (p00,p11,p01). The prism support extrudes along the heightfield frame's
// local -Z (SupportTrianglePrism, G2) so a body resting on the top face is pushed
// up and never tunnels the back.
//
// DETERMINISM (D1 for the new path): the cell sweep is a FIXED nested (r,c,sub)
// loop; cvx is D1 by construction; the cross-prism reduction is the same
// farthest-point-spread (deepest seed, strict-> max-min-distance, fixed-order)
// used by amf::ReduceAndEmitSpread -> two-run byte-identical contact streams.
//
// FAMILY GATING (D1): EARLY-EXITS unless family == kContactFamilyPairDriven, so
// the FusedFoot/UnionCsr graphs (which enqueue it when has_collidables) do NO
// work -> their goldens are byte-untouched.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "collision/analytical_manifold.hpp"   // amf:: (HD) PrimParams/PrimFrame
#include "collision/convex_narrowphase.hpp"    // cvx:: GJK/EPA/face-clip + prism
#include "collision/shape_kind.hpp"            // nuka::collision::ShapeKind
#include "math/transform.hpp"
#include "nk/model/generated/views.hpp"        // ModelView / DataView
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"  // PrimShapeDev / LoadPrimShape / PrimRotate
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/op_schema.hpp"

namespace nuka::phi {

namespace {

namespace amf = ::nuka::collision::amf;
namespace cvx = ::nuka::collision::cvx;
using ::nuka::constraint::ContactManifold;
using ::nuka::math::Vec3;
using ::nuka::phi::nkops::PrimShapeDev;
using ::nuka::phi::nkops::LoadPrimShape;
using ::nuka::phi::nkops::PrimRotate;

constexpr uint32_t kBlockSize = 128u;
constexpr uint32_t kKindHeightfield = ::nuka::collision::kShapeHeightfield;
constexpr uint32_t kKindBox         = ::nuka::collision::kShapeBox;
constexpr uint32_t kKindSphere      = ::nuka::collision::kShapeSphere;
constexpr uint32_t kKindCapsule     = ::nuka::collision::kShapeCapsule;
constexpr uint32_t kKindConvexHull  = ::nuka::collision::kShapeConvexHull;

// Max cross-prism contact candidates we accumulate before the <=4 reduction. A
// body's AABB overlaps a bounded cell range; we cap conservatively and keep the
// DEEPEST candidates if the cap is hit (deterministic, never UB).
constexpr int kMaxCand = 32;

// One contact candidate (world position + world normal + depth), kept per point
// so tread + riser normals both survive the reduction.
struct Cand {
    Vec3  pos;
    Vec3  nrm;
    float depth;
    uint32_t feat;  // a fixed insertion order id (cell*2+sub)*4 + point -> D1 key.
};

// Build the convex side's world PrimParams (the same MakePrim the prim
// narrowphase uses).
__device__ amf::PrimParams MakePrim(const PrimShapeDev& s,
                                    const math::Transform& xf) {
    amf::PrimParams p;
    p.frame.cx = PrimRotate(xf.rotation, Vec3{1, 0, 0});
    p.frame.cy = PrimRotate(xf.rotation, Vec3{0, 1, 0});
    p.frame.cz = PrimRotate(xf.rotation, Vec3{0, 0, 1});
    p.frame.t = xf.position;
    p.radius = s.params[0];
    p.half_height = s.params[1];
    p.half_extents = {s.params[0], s.params[1], s.params[2]};
    return p;
}

// Convex-side support proxy from the in-hand world PrimParams (Box/Sphere/Capsule)
// or the cooked hull pool (ConvexHull). Returns false for a non-cvx kind.
// L-RECON-D: (hull_off, hull_cnt) is the convex shape's OWN slice into the
// concatenated hull_verts pool (lanes 10/11 of its shape_table row).
__device__ bool MakeConvexProxy(uint32_t kind, const amf::PrimParams& prim,
                                const float* hull_verts, uint32_t hull_off,
                                uint32_t hull_cnt,
                                cvx::ConvexHullView* hull_store,
                                cvx::SupportProxy* out) {
    switch (kind) {
        case kKindBox:     out->kind = cvx::SupportKind::Box;     out->prim = &prim; return true;
        case kKindSphere:  out->kind = cvx::SupportKind::Sphere;  out->prim = &prim; return true;
        case kKindCapsule: out->kind = cvx::SupportKind::Capsule; out->prim = &prim; return true;
        case kKindConvexHull:
            hull_store->verts = hull_verts + static_cast<size_t>(hull_off) * 3u;
            hull_store->vcount = hull_cnt;
            hull_store->frame = prim.frame;
            hull_store->warp_lockstep = false;  // thread-per-slot (not full-warp)
            out->kind = cvx::SupportKind::Hull;
            out->hull = hull_store;
            return true;
        default:
            return false;
    }
}

// Heightfield local AABB (in the heightfield frame) from an OBB->AABB projection
// of the convex's world AABB. Mirrors Newton heightfield_vs_convex_midphase: we
// build the convex's world AABB (centre +- half via the rotated extents), then
// express its corner span in the heightfield frame. We use the convex's WORLD
// AABB centre/half (computed from the convex prim) transformed by the heightfield
// frame's WorldToLocal — a conservative AABB-of-the-transformed-AABB.
__device__ void ConvexLocalAabb(uint32_t kind, const amf::PrimParams& prim,
                                const float* hull_verts, uint32_t hull_off,
                                uint32_t hull_cnt,
                                const amf::PrimFrame& hf_frame,
                                Vec3* out_lo, Vec3* out_hi) {
    // World AABB half-extents of the convex (rotation-baked).
    Vec3 c = prim.frame.t;
    Vec3 he{0, 0, 0};
    auto rotabs = [&](Vec3 v) -> Vec3 {
        const Vec3 cx = prim.frame.cx, cy = prim.frame.cy, cz = prim.frame.cz;
        return {fabsf(cx.x) * v.x + fabsf(cy.x) * v.y + fabsf(cz.x) * v.z,
                fabsf(cx.y) * v.x + fabsf(cy.y) * v.y + fabsf(cz.y) * v.z,
                fabsf(cx.z) * v.x + fabsf(cy.z) * v.y + fabsf(cz.z) * v.z};
    };
    switch (kind) {
        case kKindSphere:  { const float r = prim.radius; he = {r, r, r}; break; }
        case kKindBox:     he = rotabs(prim.half_extents); break;
        case kKindCapsule: he = rotabs(Vec3{prim.radius, prim.radius + prim.half_height,
                                            prim.radius}); break;
        case kKindConvexHull: {
            // Bound-sphere from this shape's slice of the hull pool (rot-invariant).
            const float* hv = hull_verts + static_cast<size_t>(hull_off) * 3u;
            float max_sq = 0.0f;
            for (uint32_t v = 0; v < hull_cnt; ++v) {
                const float x = hv[v * 3u + 0u];
                const float y = hv[v * 3u + 1u];
                const float z = hv[v * 3u + 2u];
                const float d = x * x + y * y + z * z;
                if (d > max_sq) max_sq = d;
            }
            const float r = sqrtf(max_sq);
            he = {r, r, r};
            break;
        }
        default: he = {0, 0, 0}; break;
    }
    // World AABB corners -> heightfield-local AABB (transform all 8 corners and
    // take the local min/max; conservative + exact for the cell-walk input).
    Vec3 lo{3.4e38f, 3.4e38f, 3.4e38f};
    Vec3 hi{-3.4e38f, -3.4e38f, -3.4e38f};
    for (int sx = -1; sx <= 1; sx += 2)
        for (int sy = -1; sy <= 1; sy += 2)
            for (int sz = -1; sz <= 1; sz += 2) {
                const Vec3 wc{c.x + sx * he.x, c.y + sy * he.y, c.z + sz * he.z};
                const Vec3 lc = hf_frame.WorldToLocal(wc);
                lo.x = fminf(lo.x, lc.x); lo.y = fminf(lo.y, lc.y); lo.z = fminf(lo.z, lc.z);
                hi.x = fmaxf(hi.x, lc.x); hi.y = fmaxf(hi.y, lc.y); hi.z = fmaxf(hi.z, lc.z);
            }
    *out_lo = lo;
    *out_hi = hi;
}

// Read a grid corner's LOCAL z from the cooked normalized height grid.
// heights stores h in [0,1]; local z = min_z + h*(max_z - min_z). row-major.
__device__ __forceinline__ float CornerZ(const float* heights, uint32_t data_offset,
                                         uint32_t ncol, float min_z, float range,
                                         uint32_t r, uint32_t c) {
    const float h = heights[data_offset + r * ncol + c];
    return min_z + h * range;
}

// FACE-ONLY sphere-vs-triangle contact (world space). A sphere resting on a
// heightfield must contact the FACE it sits over, NOT the internal triangulation
// edges that split each flat cell into two coplanar triangles: the closest point
// on such a shared edge to a shallow-resting sphere center is nearly IN-PLANE, so
// an edge-based query (EPA / SphereHull closest-point) emits a near-HORIZONTAL
// normal -- a spurious sideways force that kicks the foot off the stance. We emit a
// contact ONLY when the sphere center projects INSIDE the triangle (closest feature
// == the face); then the normal is the true face normal (oriented toward the
// center) and the depth is radius - |signed distance to the plane|. On flat ground
// this is EXACTLY the analytic vertical-support contact (point = center projected to
// the surface, normal +Z, depth = (surface+r) - center_z); on a step riser the
// steep face yields the correct near-horizontal normal. Real CONVEX edges (a step's
// top lip) are a rare secondary feature handled by the neighbouring face the center
// projects into; internal coplanar edges are correctly suppressed. Deterministic
// (pure arithmetic, no data-dependent ordering).
__device__ bool SphereTriangleFace(Vec3 center, float radius, Vec3 a, Vec3 b, Vec3 c,
                                   Vec3* out_pos, Vec3* out_nrm, float* out_pen) {
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    Vec3 n{ab.y * ac.z - ab.z * ac.y,
           ab.z * ac.x - ab.x * ac.z,
           ab.x * ac.y - ab.y * ac.x};
    const float nl2 = n.Dot(n);
    if (nl2 < 1.0e-20f) return false;             // degenerate triangle
    n = n * (1.0f / sqrtf(nl2));
    // Orient the face normal toward the sphere center (outward for a resting foot).
    float s = (center - a).Dot(n);
    if (s < 0.0f) { n = n * -1.0f; s = -s; }
    if (s > radius) return false;                 // sphere clears the plane
    // Closest point = center projected onto the triangle plane.
    const Vec3 proj = center - n * s;
    // Barycentric inside test (proj within triangle abc).
    const Vec3 v2 = proj - a;
    const float d00 = ab.Dot(ab);
    const float d01 = ab.Dot(ac);
    const float d11 = ac.Dot(ac);
    const float d20 = v2.Dot(ab);
    const float d21 = v2.Dot(ac);
    const float denom = d00 * d11 - d01 * d01;
    if (fabsf(denom) < 1.0e-20f) return false;
    const float inv = 1.0f / denom;
    const float vv = (d11 * d20 - d01 * d21) * inv;
    const float ww = (d00 * d21 - d01 * d20) * inv;
    const float uu = 1.0f - vv - ww;
    // Epsilon-tolerant inside test: a foot whose center projects onto a cell/
    // diagonal BOUNDARY (go2's symmetric feet can land exactly on the grid lines)
    // would FP-fail a strict `< 0` test in BOTH adjacent coplanar triangles and
    // drop the contact entirely. Admit boundary projections (they then register in
    // one or both triangles; ClusterByNormal collapses the duplicate by normal).
    constexpr float kBaryEps = 1.0e-4f;
    if (uu < -kBaryEps || vv < -kBaryEps || ww < -kBaryEps) return false;
    *out_pos = proj;          // contact point on the face surface
    *out_nrm = n;             // separation dir for the sphere (side a == convex)
    *out_pen = radius - s;    // > 0 (s <= radius checked above)
    return true;
}

// Add the contact points of a manifold into the candidate buffer (each point
// keeps its OWN world position/normal/depth so distinct prism normals survive).
__device__ void AccumManifold(const ContactManifold& m, uint32_t feat_base,
                              Cand* cand, int* n) {
    for (uint32_t i = 0u; i < m.point_count; ++i) {
        if (m.points[i].penetration <= 0.0f) continue;  // only real overlaps.
        if (*n >= kMaxCand) {
            // Cap hit: replace the SHALLOWEST kept candidate if this is deeper
            // (deterministic keep-deepest; fixed scan).
            int shallow = 0;
            for (int k = 1; k < kMaxCand; ++k)
                if (cand[k].depth < cand[shallow].depth) shallow = k;
            if (m.points[i].penetration > cand[shallow].depth) {
                cand[shallow].pos = m.points[i].position;
                cand[shallow].nrm = m.points[i].normal;
                cand[shallow].depth = m.points[i].penetration;
                cand[shallow].feat = feat_base + i;
            }
            continue;
        }
        cand[*n].pos = m.points[i].position;
        cand[*n].nrm = m.points[i].normal;
        cand[*n].depth = m.points[i].penetration;
        cand[*n].feat = feat_base + i;
        ++(*n);
    }
}

// Farthest-point-spread select <=4 candidates, PRESERVING per-point normals.
// Mirrors amf::ReduceAndEmitSpread (deepest seed, strict-> max-min-distance,
// fixed-order, feature-ascending emit) but keeps each point's own normal.
__device__ int ReducePerPoint(const Cand* cand, int n, Cand* out4) {
    if (n <= 0) return 0;
    if (n <= 4) {
        // copy, then insertion-sort by feature (D1 slot order).
        for (int i = 0; i < n; ++i) out4[i] = cand[i];
        for (int i = 1; i < n; ++i) {
            Cand key = out4[i]; int j = i - 1;
            while (j >= 0 && out4[j].feat > key.feat) { out4[j + 1] = out4[j]; --j; }
            out4[j + 1] = key;
        }
        return n;
    }
    int kept[4];
    bool used[kMaxCand];
    for (int i = 0; i < kMaxCand; ++i) used[i] = false;
    int seed = 0;
    for (int i = 1; i < n; ++i)
        if (cand[i].depth > cand[seed].depth) seed = i;  // strict -> lowest idx.
    kept[0] = seed; used[seed] = true;
    for (int s = 1; s < 4; ++s) {
        int best = -1; float best_mind = -1.0f;
        for (int i = 0; i < n; ++i) {
            if (used[i]) continue;
            float mind = 3.4e38f;
            for (int t = 0; t < s; ++t) {
                const Vec3 d = cand[i].pos - cand[kept[t]].pos;
                const float dd = d.Dot(d);
                if (dd < mind) mind = dd;
            }
            if (mind > best_mind) { best_mind = mind; best = i; }  // strict -> lowest idx.
        }
        if (best < 0) break;
        kept[s] = best; used[best] = true;
    }
    // feature-ascending order over the kept set (D1 slot order).
    for (int i = 1; i < 4; ++i) {
        int key = kept[i]; int j = i - 1;
        while (j >= 0 && cand[kept[j]].feat > cand[key].feat) { kept[j + 1] = kept[j]; --j; }
        kept[j + 1] = key;
    }
    for (int i = 0; i < 4; ++i) out4[i] = cand[kept[i]];
    return 4;
}

// Cluster a single convex's heightfield candidates by NORMAL direction and keep
// the DEEPEST per cluster. The per-cell triangle split (2 sub-triangles per cell)
// + a convex straddling several cells produce MANY candidates for one surface --
// on FLAT ground a resting foot picks up its true +Z face contact PLUS shallower
// diagonal-edge / adjacent-cell artifacts (slightly tilted normals). Those extra
// contacts over-constrain the foot and kick the stance off the analytic rest pose.
// Clustering by normal collapses each distinct CONTACT SURFACE to one representative
// (the deepest), so a foot on flat ground yields ONE clean +Z contact (matching the
// analytic vertical-support detector), WHILE a foot straddling a STEP keeps BOTH the
// tread (+Z) and the near-horizontal riser cluster (the physically-correct two-
// surface contact the analytic model misses). DETERMINISM: fixed feat-order scan,
// first-match cluster assignment (lowest cluster index), strict-`>` deepest keep.
__device__ int ClusterByNormal(const Cand* in, int n, Cand* out) {
    // Merge candidates whose unit normals lie within ~60 deg (cos > 0.5): a flat-
    // cell diagonal artifact (a few deg off +Z) merges into the +Z surface, while a
    // step riser (~90 deg off +Z) stays its own cluster.
    constexpr float kMergeCos = 0.5f;
    int m = 0;
    for (int i = 0; i < n; ++i) {
        int hit = -1;
        for (int j = 0; j < m; ++j) {
            if (in[i].nrm.Dot(out[j].nrm) > kMergeCos) { hit = j; break; }
        }
        if (hit < 0) {
            if (m < kMaxCand) { out[m] = in[i]; ++m; }
        } else if (in[i].depth > out[hit].depth) {
            out[hit] = in[i];  // keep the deepest contact on this surface
        }
    }
    return m;
}

__global__ void NarrowphaseHeightfieldKernel(
    const uint32_t* __restrict__ candidate_pairs,   // elem:2 per slot
    const uint32_t* __restrict__ pair_count,        // per env
    const float* __restrict__ shape_table,
    const math::Transform* __restrict__ body_pose,
    const float* __restrict__ hull_verts,
    uint32_t /*hull_vert_count*/,   // L-RECON-D: per-shape slice now in shape_table
    const float* __restrict__ heights,
    NarrowphaseHeightfieldParams hp,
    uint32_t* __restrict__ ucount,
    Vec3* __restrict__ upoint,
    Vec3* __restrict__ unormal,
    float* __restrict__ udepth,
    uint32_t* __restrict__ ucontact_a,
    uint32_t* __restrict__ ucontact_b,
    uint32_t* __restrict__ ucontact_gen,
    uint32_t* __restrict__ contact_count) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t total = hp.env_count * hp.slot_stride;
    if (gid >= total) return;
    const uint32_t env = gid / hp.slot_stride;
    const uint32_t slot = gid - env * hp.slot_stride;
    const uint32_t live = pair_count[env];
    if (slot >= live || slot >= hp.slot_stride) return;  // not a live candidate.

    const uint32_t a = candidate_pairs[static_cast<size_t>(gid) * 2u + 0u];
    const uint32_t b = candidate_pairs[static_cast<size_t>(gid) * 2u + 1u];
    const PrimShapeDev sa = LoadPrimShape(shape_table, a);
    const PrimShapeDev sb = LoadPrimShape(shape_table, b);

    // Identify the heightfield side + the convex side. If neither is the
    // heightfield, this slot belongs to NarrowphasePrimitives -> leave it.
    uint32_t conv_local = 0u, hf_local = 0u;
    PrimShapeDev conv_sh;
    bool a_is_hf = (sa.kind == kKindHeightfield);
    bool b_is_hf = (sb.kind == kKindHeightfield);
    if (a_is_hf == b_is_hf) return;  // both or neither -> not our pair.
    if (a_is_hf) { hf_local = a; conv_local = b; conv_sh = sb; }
    else         { hf_local = b; conv_local = a; conv_sh = sa; }

    // The convex must be a cvx shape (box/sphere/capsule/hull).
    const math::Transform xconv = body_pose[env * hp.bodies_per_env + conv_local];
    const amf::PrimParams cprim = MakePrim(conv_sh, xconv);
    cvx::ConvexHullView conv_hull;
    cvx::SupportProxy conv_proxy;
    if (!MakeConvexProxy(conv_sh.kind, cprim, hull_verts,
                         conv_sh.hull_vert_offset, conv_sh.hull_vert_count,
                         &conv_hull, &conv_proxy)) {
        return;  // non-cvx convex side -> nothing to do.
    }

    // The heightfield world frame (its body pose). The grid is laid out in this
    // LOCAL frame; prism corners are LOCAL, the frame transforms to world.
    const math::Transform xhf = body_pose[env * hp.bodies_per_env + hf_local];
    amf::PrimFrame hf_frame;
    hf_frame.cx = PrimRotate(xhf.rotation, Vec3{1, 0, 0});
    hf_frame.cy = PrimRotate(xhf.rotation, Vec3{0, 1, 0});
    hf_frame.cz = PrimRotate(xhf.rotation, Vec3{0, 0, 1});
    hf_frame.t = xhf.position;

    // Project the convex's world AABB into heightfield-local space + map to cells.
    Vec3 lo, hi;
    ConvexLocalAabb(conv_sh.kind, cprim, hull_verts,
                    conv_sh.hull_vert_offset, conv_sh.hull_vert_count, hf_frame,
                    &lo, &hi);
    const float margin = hp.contact_margin;
    lo.x -= margin; lo.y -= margin; lo.z -= margin;
    hi.x += margin; hi.y += margin; hi.z += margin;

    const float origin_x = hp.origin_x;  // local (0,0) corner X
    const float origin_y = hp.origin_y;
    const float cell = hp.cell_size;
    const float min_z = hp.min_z;
    const float range = hp.max_z - hp.min_z;
    const uint32_t ncol = hp.ncol;
    const uint32_t nrow = hp.nrow;

    // cell index = floor((local - origin) / cell). Clamp to [0, n-2] (a cell is
    // bounded by corners c..c+1, so the last valid cell index is n-2).
    auto clampi = [](int v, int lo_, int hi_) -> int {
        return v < lo_ ? lo_ : (v > hi_ ? hi_ : v);
    };
    int c_lo = static_cast<int>(floorf((lo.x - origin_x) / cell));
    int c_hi = static_cast<int>(floorf((hi.x - origin_x) / cell));
    int r_lo = static_cast<int>(floorf((lo.y - origin_y) / cell));
    int r_hi = static_cast<int>(floorf((hi.y - origin_y) / cell));
    c_lo = clampi(c_lo, 0, static_cast<int>(ncol) - 2);
    c_hi = clampi(c_hi, 0, static_cast<int>(ncol) - 2);
    r_lo = clampi(r_lo, 0, static_cast<int>(nrow) - 2);
    r_hi = clampi(r_hi, 0, static_cast<int>(nrow) - 2);

    // If the convex's local-z AABB sits entirely above the field's max OR below
    // its min by more than margin, the broadphase still flagged the big-leaf, but
    // the cell sweep below still tests cvx (cheap, correct). Sweep the cells.
    Cand cand[kMaxCand];
    int ncand = 0;
    // Solid -Z extrusion depth: through the whole field z-span + a fixed pad so
    // the prism is solid down to the field floor (a body never tunnels the back).
    const float extrude = (range > 0.0f ? range : 0.0f) + 1.0f;

    for (int r = r_lo; r <= r_hi; ++r) {
        for (int c = c_lo; c <= c_hi; ++c) {
            const uint32_t rc = static_cast<uint32_t>(r);
            const uint32_t cc = static_cast<uint32_t>(c);
            // Corner LOCAL positions (Newton layout). x grows with col, y with row.
            const float x0 = origin_x + static_cast<float>(c) * cell;
            const float x1 = x0 + cell;
            const float y0 = origin_y + static_cast<float>(r) * cell;
            const float y1 = y0 + cell;
            const float z00 = CornerZ(heights, hp.data_offset, ncol, min_z, range, rc, cc);
            const float z10 = CornerZ(heights, hp.data_offset, ncol, min_z, range, rc, cc + 1u);
            const float z01 = CornerZ(heights, hp.data_offset, ncol, min_z, range, rc + 1u, cc);
            const float z11 = CornerZ(heights, hp.data_offset, ncol, min_z, range, rc + 1u, cc + 1u);
            const Vec3 p00{x0, y0, z00};
            const Vec3 p10{x1, y0, z10};
            const Vec3 p01{x0, y1, z01};
            const Vec3 p11{x1, y1, z11};
            // The cell's 2 triangles (Newton: tri0=(p00,p10,p11), tri1=(p00,p11,p01)).
            const uint32_t cell_idx = rc * (ncol - 1u) + cc;
            for (int sub = 0; sub < 2; ++sub) {
                Vec3 tv[3];
                if (sub == 0) { tv[0] = p00; tv[1] = p10; tv[2] = p11; }
                else          { tv[0] = p00; tv[1] = p11; tv[2] = p01; }
                ContactManifold m;
                const uint32_t feat_base = (cell_idx * 2u + static_cast<uint32_t>(sub)) * 4u;
                if (conv_sh.kind == kKindSphere) {
                    // ROBUST shallow sphere-vs-heightfield. A sphere is the one
                    // analytic-support shape whose CSO curvature breaks EPA's shallow-
                    // penetration monotonicity (the documented detect->drop->detect
                    // dead band ~1-1.4mm; see cvx::SphereHull) -- on a FLAT field that
                    // DROPPED a resting foot (3 contacts vs the analytic 4). But a
                    // closest-point query (EPA or SphereHull) on a per-triangle prism
                    // returns the INTERNAL triangulation EDGE as the closest feature
                    // for a shallow-resting sphere, whose near-HORIZONTAL normal kicks
                    // the foot sideways (3.45 rad). A heightfield's internal coplanar
                    // edges are not real geometric edges, so we take the FACE-ONLY
                    // contact: emit iff the center projects INSIDE this triangle, with
                    // the true face normal. On flat this is EXACTLY the analytic
                    // vertical-support contact; on a riser the steep face gives the
                    // correct near-horizontal normal. Box/capsule/hull keep the EPA
                    // path below (EPA is robust for flat-faced supports).
                    const Vec3 wa = hf_frame.LocalToWorld(tv[0]);
                    const Vec3 wb = hf_frame.LocalToWorld(tv[1]);
                    const Vec3 wc = hf_frame.LocalToWorld(tv[2]);
                    Vec3 cpos, cnrm;
                    float cpen;
                    if (SphereTriangleFace(cprim.frame.t, cprim.radius, wa, wb, wc,
                                           &cpos, &cnrm, &cpen)) {
                        ::nuka::constraint::ContactPoint pt;
                        pt.position = cpos;
                        pt.normal = cnrm;       // sep dir for side a (the convex/foot)
                        pt.penetration = cpen;
                        pt.stable_key = 0ull;
                        m.AddPoint(pt);
                    }
                } else {
                    // Build the prism ConvexHullView: 3 LOCAL verts, heightfield world
                    // frame, -Z extrude. cvx transforms verts via frame.LocalToWorld.
                    float prism_verts[9];
                    prism_verts[0] = tv[0].x; prism_verts[1] = tv[0].y; prism_verts[2] = tv[0].z;
                    prism_verts[3] = tv[1].x; prism_verts[4] = tv[1].y; prism_verts[5] = tv[1].z;
                    prism_verts[6] = tv[2].x; prism_verts[7] = tv[2].y; prism_verts[8] = tv[2].z;
                    cvx::ConvexHullView prism;
                    prism.verts = prism_verts;
                    prism.vcount = 3u;
                    prism.frame = hf_frame;
                    prism.extrude = extrude;
                    prism.warp_lockstep = false;
                    cvx::SupportProxy prism_proxy;
                    prism_proxy.kind = cvx::SupportKind::TrianglePrism;
                    prism_proxy.hull = &prism;
                    // Convex side first (A), prism second (B): the manifold normal is
                    // the separation dir for A (the convex). Side a == convex, side b
                    // == the static heightfield -> the assembly scatters reaction into
                    // the convex only (the heightfield is body_id == -1).
                    cvx::ConvexNarrowphase(conv_proxy, prism_proxy, &m);
                }
                AccumManifold(m, feat_base, cand, &ncand);
            }
        }
    }

    // Reduce to <=4 (per-point normals preserved) and write into THIS slot.
    // SPHERE convex only: cluster by contact surface (normal) keeping the deepest
    // per surface FIRST, so a foot collapses its per-cell triangle-split / multi-
    // cell duplicates to ONE +Z contact on flat (analytic-equivalent) while keeping
    // a step's distinct tread + riser surfaces. Box/capsule/hull are NOT clustered:
    // a box FACE legitimately needs its multi-corner contact spread for a stable
    // rest (merging same-normal corners to one point collapses that support), so
    // they keep the EPA manifold + farthest-point reduce unchanged (Phase 2A).
    Cand out4[4];
    int kept;
    if (conv_sh.kind == kKindSphere) {
        Cand clustered[kMaxCand];
        const int nclust = ClusterByNormal(cand, ncand, clustered);
        kept = ReducePerPoint(clustered, nclust, out4);
    } else {
        kept = ReducePerPoint(cand, ncand, out4);
    }
    ucount[gid] = static_cast<uint32_t>(kept);
    for (uint32_t i = 0u; i < 4u; ++i) {
        const size_t at = static_cast<size_t>(gid) * 4u + i;
        if (i < static_cast<uint32_t>(kept)) {
            upoint[at] = out4[i].pos;
            unormal[at] = out4[i].nrm;
            udepth[at] = out4[i].depth;
            ucontact_a[at] = conv_local;   // side a = the convex body row.
            ucontact_b[at] = hf_local;     // side b = the heightfield (static).
            ucontact_gen[at] = 1u;
        } else {
            upoint[at] = {0, 0, 0}; unormal[at] = {0, 0, 0}; udepth[at] = 0.0f;
            ucontact_a[at] = 0u; ucontact_b[at] = 0u; ucontact_gen[at] = 0u;
        }
    }
    if (kept > 0 && contact_count != nullptr) {
        atomicAdd(&contact_count[env], static_cast<uint32_t>(kept));
    }
}

Status OpNarrowphaseHeightfield(const ModelView& model, const DataView& data,
                                const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const NarrowphaseHeightfieldParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->family != kContactFamilyPairDriven) return Status::Ok;  // early-exit.
    if (p->has_heightfield == 0u) return Status::Ok;               // no field cooked.
    if (p->env_count == 0u || p->slot_stride == 0u || p->bodies_per_env == 0u)
        return Status::Ok;
    if (data.candidate_pairs == nullptr || data.body_pose == nullptr ||
        model.shape_table == nullptr || model.heights == nullptr) {
        return Status::Ok;
    }
    const uint32_t total = p->env_count * p->slot_stride;
    const uint32_t blocks = (total + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(NarrowphaseHeightfieldKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               data.candidate_pairs, data.pair_count,
               static_cast<const float*>(model.shape_table),
               static_cast<const math::Transform*>(data.body_pose),
               static_cast<const float*>(model.hull_verts), p->hull_vert_count,
               static_cast<const float*>(model.heights), *p,
               data.ucontact_count, data.ucontact_point, data.ucontact_normal,
               data.ucontact_depth, data.ucontact_a, data.ucontact_b,
               data.ucontact_gen, data.contact_count);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkNarrowphaseHeightfieldOps() {
    SetCudaOp(NkOp::NarrowphaseHeightfield, &OpNarrowphaseHeightfield);
}

}  // namespace nuka::phi
