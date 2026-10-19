

#include <cuda_runtime.h>

#include "collision/analytical_manifold.hpp"   
#include "collision/convex_narrowphase.hpp"    
#include "collision/shape_kind.hpp"            
#include "math/transform.hpp"
#include "nk/contact/contact_identity.hpp"
#include "nk/model/generated/views.hpp"        
#include "nk/solve/nk_row.hpp"                  
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"  
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
constexpr uint32_t kKindCylinder    = ::nuka::collision::kShapeCylinder;

constexpr int kMaxCand = 32;

constexpr float kHeightfieldPrismExtrudePad = 1.0f;

struct Cand {
    Vec3  pos;
    Vec3  nrm;
    float depth;
    uint32_t feat;  
};

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

__device__ bool MakeConvexProxy(uint32_t kind, const amf::PrimParams& prim,
                                const float* hull_verts, uint32_t hull_off,
                                uint32_t hull_cnt,
                                cvx::ConvexHullView* hull_store,
                                cvx::SupportProxy* out) {
    switch (kind) {
        case kKindBox:     out->kind = cvx::SupportKind::Box;     out->prim = &prim; return true;
        case kKindSphere:  out->kind = cvx::SupportKind::Sphere;  out->prim = &prim; return true;
        case kKindCapsule: out->kind = cvx::SupportKind::Capsule; out->prim = &prim; return true;
        case kKindCylinder: out->kind = cvx::SupportKind::Cylinder; out->prim = &prim; return true;
        case kKindConvexHull:
            hull_store->verts = hull_verts + static_cast<size_t>(hull_off) * 3u;
            hull_store->vcount = hull_cnt;
            hull_store->frame = prim.frame;
            hull_store->warp_lockstep = false;  
            out->kind = cvx::SupportKind::Hull;
            out->hull = hull_store;
            return true;
        default:
            return false;
    }
}

__device__ void ConvexLocalAabb(uint32_t kind, const amf::PrimParams& prim,
                                const float* hull_verts, uint32_t hull_off,
                                uint32_t hull_cnt,
                                const amf::PrimFrame& hf_frame,
                                Vec3* out_lo, Vec3* out_hi) {
    
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
        case kKindCapsule: he = rotabs(Vec3{prim.radius, prim.radius,
                                            prim.radius + prim.half_height}); break;
        case kKindCylinder: {
            const Vec3 a = prim.frame.cz;
            he = {prim.half_height * fabsf(a.x) + prim.radius * sqrtf(fmaxf(0.0f, 1.0f - a.x * a.x)),
                  prim.half_height * fabsf(a.y) + prim.radius * sqrtf(fmaxf(0.0f, 1.0f - a.y * a.y)),
                  prim.half_height * fabsf(a.z) + prim.radius * sqrtf(fmaxf(0.0f, 1.0f - a.z * a.z))};
            break;
        }
        case kKindConvexHull: {
            
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

__device__ __forceinline__ float CornerZ(const float* heights, uint32_t data_offset,
                                         uint32_t ncol, float min_z, float range,
                                         uint32_t r, uint32_t c) {
    const float h = heights[data_offset + r * ncol + c];
    return min_z + h * range;
}

__device__ bool SphereTriangleFace(Vec3 center, float radius, Vec3 a, Vec3 b, Vec3 c,
                                   Vec3 solid_out,
                                   Vec3* out_pos, Vec3* out_nrm, float* out_pen) {
    const Vec3 ab = b - a;
    const Vec3 ac = c - a;
    Vec3 n{ab.y * ac.z - ab.z * ac.y,
           ab.z * ac.x - ab.x * ac.z,
           ab.x * ac.y - ab.y * ac.x};
    const float nl2 = n.Dot(n);
    if (nl2 < 1.0e-20f) return false;             
    n = n * (1.0f / sqrtf(nl2));

    if (n.Dot(solid_out) < 0.0f) n = n * -1.0f;
    const float s = (center - a).Dot(n);
    if (s > radius) return false;                 
    
    const Vec3 proj = center - n * s;
    
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

    constexpr float kBaryEps = 1.0e-4f;
    if (uu < -kBaryEps || vv < -kBaryEps || ww < -kBaryEps) return false;
    *out_pos = proj;          
    *out_nrm = n;             
    *out_pen = radius - s;    
    return true;
}

__device__ void SphereHeightfieldTri(const amf::PrimParams& cprim,
                                     const amf::PrimFrame& hf_frame,
                                     const Vec3 tv[3], ContactManifold* m) {
    const Vec3 wa = hf_frame.LocalToWorld(tv[0]);
    const Vec3 wb = hf_frame.LocalToWorld(tv[1]);
    const Vec3 wc = hf_frame.LocalToWorld(tv[2]);
    Vec3 cpos, cnrm;
    float cpen;

    if (SphereTriangleFace(cprim.frame.t, cprim.radius, wa, wb, wc, hf_frame.cz,
                           &cpos, &cnrm, &cpen)) {
        ::nuka::constraint::ContactPoint pt;
        pt.position = cpos;
        pt.normal = cnrm;       
        pt.penetration = cpen;
        pt.stable_key = 0ull;
        m->AddPoint(pt);
    }
}

constexpr int kCapsuleSegSamples = 9;
__device__ void CapsuleHeightfieldTri(const amf::PrimParams& cprim,
                                     const amf::PrimFrame& hf_frame,
                                     const Vec3 tv[3], ContactManifold* m) {
    const Vec3 wa = hf_frame.LocalToWorld(tv[0]);
    const Vec3 wb = hf_frame.LocalToWorld(tv[1]);
    const Vec3 wc = hf_frame.LocalToWorld(tv[2]);
    const Vec3 e0 = cprim.frame.t + cprim.frame.cz * cprim.half_height;
    const Vec3 e1 = cprim.frame.t - cprim.frame.cz * cprim.half_height;
    for (int i = 0; i < kCapsuleSegSamples; ++i) {
        const float t = static_cast<float>(i) /
                        static_cast<float>(kCapsuleSegSamples - 1);
        const Vec3 center = e0 * (1.0f - t) + e1 * t;
        Vec3 cpos, cnrm;
        float cpen;
        if (SphereTriangleFace(center, cprim.radius, wa, wb, wc, hf_frame.cz,
                               &cpos, &cnrm, &cpen)) {
            ::nuka::constraint::ContactPoint pt;
            pt.position = cpos;
            pt.normal = cnrm;     
            pt.penetration = cpen;
            pt.stable_key = 0ull;
            m->AddPoint(pt);
        }
    }
}

__device__ void PrismCvxTri(const cvx::SupportProxy& conv_proxy,
                            const amf::PrimFrame& hf_frame, const Vec3 tv[3],
                            float extrude, ContactManifold* m, cvx::ConvexWorkspace& workspace) {
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

    cvx::ConvexNarrowphase(conv_proxy, prism_proxy, m, workspace);
}

__device__ void AccumManifold(const ContactManifold& m, uint32_t feat_base,
                              Cand* cand, int* n) {
    for (uint32_t i = 0u; i < m.point_count; ++i) {
        if (m.points[i].penetration <= 0.0f) continue;  
        if (*n >= kMaxCand) {

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

__device__ int ReducePerPoint(const Cand* cand, int n, Cand* out4) {
    if (n <= 0) return 0;
    if (n <= 4) {
        
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
        if (cand[i].depth > cand[seed].depth) seed = i;  
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
            if (mind > best_mind) { best_mind = mind; best = i; }  
        }
        if (best < 0) break;
        kept[s] = best; used[best] = true;
    }
    
    for (int i = 1; i < 4; ++i) {
        int key = kept[i]; int j = i - 1;
        while (j >= 0 && cand[kept[j]].feat > cand[key].feat) { kept[j + 1] = kept[j]; --j; }
        kept[j + 1] = key;
    }
    for (int i = 0; i < 4; ++i) out4[i] = cand[kept[i]];
    return 4;
}

__device__ int ClusterByNormal(const Cand* in, int n, Cand* out) {

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
            out[hit] = in[i];  
        }
    }
    return m;
}

struct HeightfieldWorkspace {
    cvx::ConvexWorkspace convex;
    Cand candidates[kMaxCand];
    Cand clustered[kMaxCand];
};

__global__ void NarrowphaseHeightfieldKernel(
    const uint32_t* __restrict__ candidate_pairs,   
    const uint32_t* __restrict__ pair_count,        
    const float* __restrict__ shape_table,
    const math::Transform* __restrict__ body_pose,
    const float* __restrict__ hull_verts,
    uint32_t /*hull_vert_count*/,   
    const float* __restrict__ heights,
    NarrowphaseHeightfieldParams hp,
    uint32_t* __restrict__ ucount,
    Vec3* __restrict__ upoint,
    Vec3* __restrict__ unormal,
    float* __restrict__ udepth,
    uint32_t* __restrict__ ucontact_a,
    uint32_t* __restrict__ ucontact_b,
    uint32_t* __restrict__ ucontact_a_kind,
    uint32_t* __restrict__ ucontact_b_kind,
    uint32_t* __restrict__ ucontact_gen,
    uint64_t* __restrict__ ucontact_id_pair,
    uint64_t* __restrict__ ucontact_id_feature,
    uint32_t* __restrict__ contact_count) {
    constexpr uint32_t kWarp = 32u;
    __shared__ __align__(16) unsigned char storage[(kBlockSize / kWarp) * sizeof(HeightfieldWorkspace)];
    if (threadIdx.x % kWarp != 0u) return;
    auto& workspace = reinterpret_cast<HeightfieldWorkspace*>(storage)[threadIdx.x / kWarp];
    const uint32_t gid = (blockIdx.x * blockDim.x + threadIdx.x) / kWarp;
    const uint32_t total = hp.env_count * hp.slot_stride;
    if (gid >= total) return;
    const uint32_t env = gid / hp.slot_stride;
    const uint32_t slot = gid - env * hp.slot_stride;
    const uint32_t live = pair_count[env];

    if (slot >= live || slot >= hp.rigid_slot_cap) return;  

    const uint32_t a = candidate_pairs[static_cast<size_t>(gid) * 2u + 0u];
    const uint32_t b = candidate_pairs[static_cast<size_t>(gid) * 2u + 1u];
    const PrimShapeDev sa = LoadPrimShape(shape_table, a);
    const PrimShapeDev sb = LoadPrimShape(shape_table, b);

    uint32_t conv_local = 0u, hf_local = 0u;
    PrimShapeDev conv_sh;
    bool a_is_hf = (sa.kind == kKindHeightfield);
    bool b_is_hf = (sb.kind == kKindHeightfield);
    if (a_is_hf == b_is_hf) return;  
    if (a_is_hf) { hf_local = a; conv_local = b; conv_sh = sb; }
    else         { hf_local = b; conv_local = a; conv_sh = sa; }

    const math::Transform xconv = body_pose[env * hp.bodies_per_env + conv_local];
    const amf::PrimParams cprim = MakePrim(conv_sh, xconv);
    cvx::ConvexHullView conv_hull;
    cvx::SupportProxy conv_proxy;
    if (!MakeConvexProxy(conv_sh.kind, cprim, hull_verts,
                         conv_sh.hull_vert_offset, conv_sh.hull_vert_count,
                         &conv_hull, &conv_proxy)) {
        return;  
    }

    const math::Transform xhf = body_pose[env * hp.bodies_per_env + hf_local];
    amf::PrimFrame hf_frame;
    hf_frame.cx = PrimRotate(xhf.rotation, Vec3{1, 0, 0});
    hf_frame.cy = PrimRotate(xhf.rotation, Vec3{0, 1, 0});
    hf_frame.cz = PrimRotate(xhf.rotation, Vec3{0, 0, 1});
    hf_frame.t = xhf.position;

    Vec3 lo, hi;
    ConvexLocalAabb(conv_sh.kind, cprim, hull_verts,
                    conv_sh.hull_vert_offset, conv_sh.hull_vert_count, hf_frame,
                    &lo, &hi);
    const float margin = hp.contact_margin;
    lo.x -= margin; lo.y -= margin; lo.z -= margin;
    hi.x += margin; hi.y += margin; hi.z += margin;

    const float origin_x = hp.origin_x;  
    const float origin_y = hp.origin_y;
    const float cell = hp.cell_size;
    const float min_z = hp.min_z;
    const float range = hp.max_z - hp.min_z;
    const uint32_t ncol = hp.ncol;
    const uint32_t nrow = hp.nrow;

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

    Cand* cand = workspace.candidates;
    int ncand = 0;

    const float extrude =
        (range > 0.0f ? range : 0.0f) + kHeightfieldPrismExtrudePad;

    for (int r = r_lo; r <= r_hi; ++r) {
        for (int c = c_lo; c <= c_hi; ++c) {
            const uint32_t rc = static_cast<uint32_t>(r);
            const uint32_t cc = static_cast<uint32_t>(c);
            
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
            
            const uint32_t cell_idx = rc * (ncol - 1u) + cc;
            for (int sub = 0; sub < 2; ++sub) {
                Vec3 tv[3];
                if (sub == 0) { tv[0] = p00; tv[1] = p10; tv[2] = p11; }
                else          { tv[0] = p00; tv[1] = p11; tv[2] = p01; }
                ContactManifold m;
                const uint32_t feat_base = (cell_idx * 2u + static_cast<uint32_t>(sub)) * 4u;

                if (conv_sh.kind == kKindSphere) {
                    SphereHeightfieldTri(cprim, hf_frame, tv, &m);
                } else if (conv_sh.kind == kKindCapsule) {
                    CapsuleHeightfieldTri(cprim, hf_frame, tv, &m);
                } else {
                    PrismCvxTri(conv_proxy, hf_frame, tv, extrude, &m, workspace.convex);
                }
                AccumManifold(m, feat_base, cand, &ncand);
            }
        }
    }

    Cand out4[4];
    int kept;
    if (conv_sh.kind == kKindSphere) {
        Cand* clustered = workspace.clustered;
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
            ucontact_a[at] = conv_local;   
            ucontact_b[at] = hf_local;     
            ucontact_a_kind[at] = ::nuka::nk::kUContactSideBody;
            ucontact_b_kind[at] = ::nuka::nk::kUContactSideBody;
            ucontact_gen[at] = 1u;
            ::nuka::nk::CanonicalContactDescriptor descriptor;
            descriptor.a.handle = conv_local;
            descriptor.b.handle = hf_local;
            descriptor.local_point_a =
                ::nuka::phi::nkops::PrimInverseTransformPoint(
                    xconv, out4[i].pos);
            descriptor.local_point_b =
                ::nuka::phi::nkops::PrimInverseTransformPoint(
                    xhf, out4[i].pos);
            descriptor.normal = out4[i].nrm;
            descriptor.feature_a = ::nuka::nk::kContactFeatureUnavailable;
            descriptor.feature_b = out4[i].feat;
            descriptor.manifold_slot = i;
            const ::nuka::nk::ContactId id = ::nuka::nk::MakeContactId(descriptor);
            ucontact_id_pair[at] = id.pair;
            ucontact_id_feature[at] = id.feature;
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
    if (p->family != kContactFamilyPairDriven) return Status::Ok;  
    if (p->has_heightfield == 0u) return Status::Ok;               
    if (p->env_count == 0u || p->slot_stride == 0u || p->bodies_per_env == 0u)
        return Status::Ok;
    if (data.candidate_pairs == nullptr || data.body_pose == nullptr ||
        model.shape_table == nullptr || model.heights == nullptr) {
        return Status::Ok;
    }
    const uint32_t total = p->env_count * p->slot_stride;
    const uint32_t blocks = (total + kBlockSize / 32u - 1u) / (kBlockSize / 32u);
    LaunchCuda(NarrowphaseHeightfieldKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               data.candidate_pairs, data.pair_count,
               static_cast<const float*>(model.shape_table),
               static_cast<const math::Transform*>(data.body_pose),
               static_cast<const float*>(model.hull_verts), p->hull_vert_count,
               static_cast<const float*>(model.heights), *p,
               data.ucontact_count, data.ucontact_point, data.ucontact_normal,
               data.ucontact_depth, data.ucontact_a, data.ucontact_b,
               data.ucontact_a_kind, data.ucontact_b_kind,
               data.ucontact_gen, data.ucontact_id_pair,
               data.ucontact_id_feature, data.contact_count);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  

void RegisterNkNarrowphaseHeightfieldOps() {
    SetCudaOp(NkOp::NarrowphaseHeightfield, &OpNarrowphaseHeightfield);
}

}  
