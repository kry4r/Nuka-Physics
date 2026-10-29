

#include <cfloat>
#include <cuda_runtime.h>

#include "collision/analytical_manifold.hpp"   
#include "collision/convex_narrowphase.hpp"    
#include "collision/dat_geometry.hpp"
#include "collision/shape_kind.hpp"            
#include "math/transform.hpp"
#include "nk/contact/contact_identity.hpp"
#include "nk/contact/contact_profile.hpp"
#include "nk/solve/nk_row.hpp"                  
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/prims_types.cuh"

namespace nuka::phi::nkops {

namespace {

namespace amf = ::nuka::collision::amf;
namespace cvx = ::nuka::collision::cvx;
constexpr uint32_t kBlock = 128u;
constexpr uint32_t kWarp = 32u;
using ::nuka::constraint::ContactManifold;

constexpr uint32_t kKindSphere  = ::nuka::collision::kShapeSphere;
constexpr uint32_t kKindCapsule = ::nuka::collision::kShapeCapsule;
constexpr uint32_t kKindBox     = ::nuka::collision::kShapeBox;
constexpr uint32_t kKindPlane   = ::nuka::collision::kShapePlane;
constexpr uint32_t kKindConvexHull = ::nuka::collision::kShapeConvexHull;
constexpr uint32_t kKindCylinder = ::nuka::collision::kShapeCylinder;

__device__ amf::PrimParams MakePrim(const PrimShapeDev& s,
                                     const math::Transform& xf) {
    amf::PrimParams p;

    p.frame.cx = PrimRotate(xf.rotation, math::Vec3{1, 0, 0});
    p.frame.cy = PrimRotate(xf.rotation, math::Vec3{0, 1, 0});
    p.frame.cz = PrimRotate(xf.rotation, math::Vec3{0, 0, 1});
    p.frame.t = xf.position;
    p.radius = s.params[0];
    p.half_height = s.params[1];
    p.half_extents = {s.params[0], s.params[1], s.params[2]};
    return p;
}

__device__ bool MakeSupportProxy(uint32_t kind, const amf::PrimParams& prim,
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

__device__ bool DispatchAnalyticPair(uint32_t ka, const amf::PrimParams& a,
                             uint32_t kb, const amf::PrimParams& b,
                             float query, ContactManifold* out) {
    out->Clear();

    if (ka == kKindSphere && kb == kKindSphere) { amf::SphereSphere(a, b, out, query); return true; }
    if (ka == kKindSphere && kb == kKindBox)    { amf::SphereBox(a, b, out, query); return true; }
    if (ka == kKindSphere && kb == kKindPlane)  { amf::SpherePlane(a, b, out, query); return true; }
    if (ka == kKindBox && kb == kKindPlane)     { amf::BoxPlane(a, b, out, query); return true; }
    if (ka == kKindBox && kb == kKindBox)       { amf::BoxBox(a, b, out, query); return true; }
    if (ka == kKindCapsule && kb == kKindPlane) { amf::CapsulePlane(a, b, out, query); return true; }
    if (ka == kKindCapsule && kb == kKindSphere){ amf::CapsuleSphere(a, b, out, query); return true; }

    if (ka == kKindCapsule && kb == kKindCapsule){ amf::CapsuleCapsule(a, b, out, query); return true; }
    if (ka == kKindCylinder && kb == kKindPlane) { amf::CylinderPlane(a, b, out, query); return true; }
    if (ka == kKindCylinder && kb == kKindSphere) { amf::CylinderSphere(a, b, out, query); return true; }
    
    auto flip = [&]() {
        for (uint32_t i = 0; i < out->point_count; ++i) {
            out->points[i].normal = math::Vec3{-out->points[i].normal.x,
                                               -out->points[i].normal.y,
                                               -out->points[i].normal.z};
        }
    };
    if (ka == kKindBox && kb == kKindSphere)    { amf::SphereBox(b, a, out, query); flip(); return true; }
    if (ka == kKindPlane && kb == kKindSphere)  { amf::SpherePlane(b, a, out, query); flip(); return true; }
    if (ka == kKindPlane && kb == kKindBox)     { amf::BoxPlane(b, a, out, query); flip(); return true; }
    if (ka == kKindPlane && kb == kKindCapsule) { amf::CapsulePlane(b, a, out, query); flip(); return true; }
    if (ka == kKindSphere && kb == kKindCapsule){ amf::CapsuleSphere(b, a, out, query); flip(); return true; }
    if (ka == kKindPlane && kb == kKindCylinder) { amf::CylinderPlane(b, a, out, query); flip(); return true; }
    if (ka == kKindSphere && kb == kKindCylinder) { amf::CylinderSphere(b, a, out, query); flip(); return true; }

    return false;
}

// Mesh vertices inside the query band meet the plane; the deepest one and the band's tangent
// extremes span the manifold. Features are vertex index + 1.
__device__ void HullPlanePair(const cvx::ConvexHullView& hull, const amf::PrimParams& plane,
                              bool hull_first, float separation, float query, ContactManifold* out,
                              math::Vec3* witness_a, math::Vec3* witness_b) {
    constexpr uint32_t kKeys = 9u;
    const math::Vec3 n = amf::Norm(plane.frame.cz, math::Vec3::UnitZ());
    const math::Vec3 t = amf::Norm(plane.frame.cx, math::Vec3::UnitX());
    const math::Vec3 s = n.Cross(t);
    const math::Vec3 axes[kKeys] = {-n, t, -t, s, -s, t + s, t - s, s - t, -t - s};
    float best[kKeys];
    uint32_t index[kKeys];
    for (uint32_t k = 0u; k < kKeys; ++k) { best[k] = -FLT_MAX; index[k] = ~0u; }
    const uint32_t lane = threadIdx.x % kWarp;
    for (uint32_t i = lane; i < hull.vcount; i += kWarp) {
        const math::Vec3 v = hull.Vertex(i) - plane.frame.t;
        if (!(v.Dot(n) < query)) continue;
        for (uint32_t k = 0u; k < kKeys; ++k) {
            const float key = v.Dot(axes[k]);
            if (key > best[k]) { best[k] = key; index[k] = i; }
        }
    }
    for (uint32_t k = 0u; k < kKeys; ++k) {
        for (uint32_t offset = kWarp / 2u; offset > 0u; offset >>= 1u) {
            const float other = __shfl_down_sync(~0u, best[k], offset);
            const uint32_t at = __shfl_down_sync(~0u, index[k], offset);
            if (other > best[k] || (other == best[k] && at < index[k])) { best[k] = other; index[k] = at; }
        }
    }
    if (lane != 0u || index[0] == ~0u) return;
    amf::ManifoldPointCand cand[kKeys];
    int count = 0;
    for (uint32_t k = 0u; k < kKeys; ++k) {
        bool seen = false;
        for (int j = 0; j < count; ++j) seen |= cand[j].feature == index[k] + 1u;
        if (seen) continue;
        const math::Vec3 v = hull.Vertex(index[k]);
        const float d = (v - plane.frame.t).Dot(n);
        cand[count++] = {v - n * (0.5f * d), separation - d, index[k] + 1u};
    }
    amf::ReduceAndEmitSpread(cand, count, hull_first ? n : -n, out);
    for (uint32_t i = 0u; i < out->point_count; ++i) {
        const math::Vec3 v = hull.Vertex(static_cast<uint32_t>(out->points[i].stable_key) - 1u);
        const math::Vec3 foot = v - n * (v - plane.frame.t).Dot(n);
        witness_a[i] = hull_first ? v : foot;
        witness_b[i] = hull_first ? foot : v;
    }
}

__device__ void DispatchConvexPair(uint32_t ka, const amf::PrimParams& a,
                                  uint32_t kb, const amf::PrimParams& b,
                                  const float* hull_verts,
                                  uint32_t hull_off_a, uint32_t hull_cnt_a,
                                  uint32_t hull_off_b, uint32_t hull_cnt_b,
                                  float separation, float query, ContactManifold* out,
                                  math::Vec3* witness_a, math::Vec3* witness_b,
                                  cvx::ConvexWorkspace& workspace) {
    if (ka == kKindPlane || kb == kKindPlane) {
        const bool hull_first = kb == kKindPlane;
        cvx::ConvexHullView hull;
        hull.verts = hull_verts + static_cast<size_t>(hull_first ? hull_off_a : hull_off_b) * 3u;
        hull.vcount = hull_first ? hull_cnt_a : hull_cnt_b;
        hull.frame = (hull_first ? a : b).frame;
        HullPlanePair(hull, hull_first ? b : a, hull_first, separation, query, out, witness_a, witness_b);
        return;
    }
    cvx::ConvexHullView ha, hb;
    cvx::SupportProxy A, B;
    if (MakeSupportProxy(ka, a, hull_verts, hull_off_a, hull_cnt_a, &ha, &A) &&
        MakeSupportProxy(kb, b, hull_verts, hull_off_b, hull_cnt_b, &hb, &B)) {
        ha.warp_lockstep = hb.warp_lockstep = true;
        const auto gjk = cvx::GjkIntersect(A, B);
        if (!gjk.intersect) {
            const auto closest = cvx::ClosestBetweenProxies(A, B);
            if (!(closest.dist > 0.0f && closest.dist < query)) return;
            if (threadIdx.x % kWarp != 0u) return;
            constraint::ContactPoint point;
            point.position = (closest.wa + closest.wb) * 0.5f;
            point.normal = (closest.wa - closest.wb) / closest.dist;
            point.penetration = separation - closest.dist;
            point.stable_key = 0ull;
            out->AddPoint(point);
            witness_a[0] = closest.wa;
            witness_b[0] = closest.wb;
        } else {
            if (threadIdx.x % kWarp != 0u) return;
            ha.warp_lockstep = hb.warp_lockstep = false;
            const bool rounded = ka == kKindCapsule || kb == kKindCapsule ||
                                 ka == kKindCylinder || kb == kKindCylinder;
            if (rounded) cvx::ConvexNarrowphase(A, B, out, workspace);
            else {
                const auto epa = cvx::EpaExpand(A, B, gjk, workspace.epa);
                cvx::BuildConvexManifold(A, B, epa, out, workspace);
            }
            for (uint32_t i = 0u; i < out->point_count; ++i) {
                auto& point = out->points[i];
                witness_a[i] = point.position - point.normal * (0.5f * point.penetration);
                witness_b[i] = point.position + point.normal * (0.5f * point.penetration);
                point.penetration += separation;
            }
        }
    }

}

// Analytic pairs use one thread; convex pairs share their support queries across a warp.
// Only the leader expands EPA and writes the manifold, keeping its workspace race free.
template <bool kConvex>
__global__ void PairDrivenNarrowphaseKernel(
    const uint32_t* __restrict__ candidate_pairs,   
    const uint32_t* __restrict__ pair_count,        
    const float* __restrict__ shape_table,
    const float* __restrict__ mat_buckets, uint32_t num_material_buckets,
    const uint64_t* __restrict__ authored_keys, const uint32_t* __restrict__ authored_profiles,
    uint32_t authored_count,
    const uint8_t* __restrict__ mesh_contact_mode,
    const math::Transform* __restrict__ body_pose,
    const math::Vec3* __restrict__ linear_velocity,
    const math::Vec3* __restrict__ angular_velocity,
    const math::Vec3* __restrict__ aabb_lo,
    const math::Vec3* __restrict__ aabb_hi,
    const float* __restrict__ previous_motion,
    const float* __restrict__ hull_verts,           
    uint32_t /*hull_vert_count*/,                    
    uint32_t gen, float separation, float dt,
    uint32_t env_count, uint32_t slot_stride, uint32_t rigid_slot_cap,
    uint32_t bodies_per_env,
    uint32_t* __restrict__ ucount,
    math::Vec3* __restrict__ upoint,
    math::Vec3* __restrict__ unormal,
    float* __restrict__ udepth,
    math::Vec3* __restrict__ uwitness_a,
    math::Vec3* __restrict__ uwitness_b,
    uint32_t* __restrict__ ulaw,
    uint32_t* __restrict__ ucontact_a,              
    uint32_t* __restrict__ ucontact_b,
    uint32_t* __restrict__ ucontact_a_kind,         
    uint32_t* __restrict__ ucontact_b_kind,
    uint32_t* __restrict__ ucontact_gen,
    uint64_t* __restrict__ ucontact_id_pair,
    uint64_t* __restrict__ ucontact_id_feature,
    uint32_t* __restrict__ contact_count,
    uint32_t* __restrict__ mesh_ogc_pair_count,
    uint32_t* __restrict__ mesh_sdf_pair_count) {
    const uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t item = kConvex ? thread / kWarp : thread;
    
    if (item >= env_count * rigid_slot_cap) return;
    const uint32_t env = item / rigid_slot_cap;
    const uint32_t slot = item - env * rigid_slot_cap;
    const uint32_t gid = env * slot_stride + slot;
    const uint32_t live = pair_count[env];

    ContactManifold m;
    m.Clear();
    math::Vec3 witness_a[4], witness_b[4];
    uint32_t a = 0u, b = 0u;
    PrimShapeDev sa, sb;
    bool convex = false;
    uint32_t law = ::nuka::nk::kContactLawSpeculative;
    float query_margin = separation;
    const auto speed = [&](uint32_t body) {
        const uint32_t at = env * bodies_per_env + body;
        const auto center = body_pose[at].position;
        const auto lo = aabb_lo[at] - center, hi = aabb_hi[at] - center;
        const math::Vec3 extent{fmaxf(fabsf(lo.x), fabsf(hi.x)),
            fmaxf(fabsf(lo.y), fabsf(hi.y)), fmaxf(fabsf(lo.z), fabsf(hi.z))};
        return fmaxf(previous_motion != nullptr ? previous_motion[at] : 0.0f,
            sqrtf(linear_velocity[at].LengthSq()) +
            sqrtf(angular_velocity[at].LengthSq() * extent.LengthSq()));
    };
    // A pair is queried over its profile band or the gap both bodies can close within the step; a
    // second shell width keeps contacts resting on the separation shell detected.
    const auto query_radius = [&]() {
        return fmaxf(query_margin, collision::DatQueryRadius(
            collision::DatMotionRadius(2.0f * separation, dt, speed(a), speed(b), kDatRelaxation)));
    };

    if (slot < live && slot < rigid_slot_cap) {
        a = candidate_pairs[static_cast<size_t>(gid) * 2u + 0u];
        b = candidate_pairs[static_cast<size_t>(gid) * 2u + 1u];
        sa = LoadPrimShape(shape_table, a);
        sb = LoadPrimShape(shape_table, b);
        const float* profiles = mat_buckets + size_t{env} * num_material_buckets * nk::kContactProfileWordCount;
        // An authored pair's single profile replaces both shape profiles.
        const uint32_t authored = AuthoredPairProfile(authored_keys, authored_profiles, authored_count, a, b);
        const float* profile_a = profiles +
            size_t{authored != ~0u ? authored : sa.contact_profile_index} * nk::kContactProfileWordCount;
        const float* profile_b = profiles +
            size_t{authored != ~0u ? authored : sb.contact_profile_index} * nk::kContactProfileWordCount;
        const uint32_t source_a = __float_as_uint(profile_a[nk::kContactProfileLawSource]);
        const uint32_t source_b = __float_as_uint(profile_b[nk::kContactProfileLawSource]);
        if (source_a == static_cast<uint32_t>(constraint::LawSource::MuJoCo) ||
            source_b == static_cast<uint32_t>(constraint::LawSource::MuJoCo))
            law = nk::kContactLawCompliant;
        if (law == nk::kContactLawCompliant && authored != ~0u)
            query_margin = profile_a[nk::kContactProfileMargin] + profile_a[nk::kContactProfileGap];
        else if (law == nk::kContactLawCompliant)
            query_margin = profile_a[nk::kContactProfileMargin] + profile_b[nk::kContactProfileMargin] +
                           profile_a[nk::kContactProfileGap] + profile_b[nk::kContactProfileGap];
        const auto route = RouteMeshContact(sa.kind, sb.kind, mesh_contact_mode[a],
                                            mesh_contact_mode[b]);
        if (!kConvex && route == MeshContactRoute::Ogc) atomicAdd(&mesh_ogc_pair_count[env], 1u);
        if (!kConvex && route == MeshContactRoute::Sdf) atomicAdd(&mesh_sdf_pair_count[env], 1u);
        if (kConvex)
            convex = route == MeshContactRoute::Analytic &&
                     ::nuka::collision::TakesConvexManifold(sa.kind, sb.kind);
        else if (route == MeshContactRoute::Analytic) {
            const math::Transform xa = body_pose[env * bodies_per_env + a];
            const math::Transform xb = body_pose[env * bodies_per_env + b];
            const amf::PrimParams pa = MakePrim(sa, xa);
            const amf::PrimParams pb = MakePrim(sb, xb);
            DispatchAnalyticPair(sa.kind, pa, sb.kind, pb, query_radius(), &m);
            if (law != nk::kContactLawCompliant)
                for (uint32_t i = 0u; i < m.point_count; ++i) m.points[i].penetration += separation;
        }
    }

    if constexpr (kConvex) {
        if (!convex) return;
        __shared__ __align__(16) unsigned char storage[(kBlock / kWarp) * sizeof(cvx::ConvexWorkspace)];
        auto& workspace = reinterpret_cast<cvx::ConvexWorkspace*>(storage)[threadIdx.x / kWarp];
        auto pa = MakePrim(sa, body_pose[env * bodies_per_env + a]);
        auto pb = MakePrim(sb, body_pose[env * bodies_per_env + b]);
        const float query = query_radius();
        const auto origin = pa.frame.t;
        pa.frame.t = {};
        pb.frame.t -= origin;
        DispatchConvexPair(sa.kind, pa, sb.kind, pb, hull_verts,
                           sa.hull_vert_offset, sa.hull_vert_count,
                           sb.hull_vert_offset, sb.hull_vert_count,
                           law == nk::kContactLawCompliant ? 0.0f : separation, query,
                           &m, witness_a, witness_b, workspace);
        if (threadIdx.x % kWarp != 0u) return;
        for (uint32_t i = 0u; i < m.point_count; ++i) {
            m.points[i].position += origin;
            witness_a[i] += origin;
            witness_b[i] += origin;
        }
    }

    const uint32_t n = m.point_count;
    ucount[gid] = n;
    ulaw[gid] = law;

    for (uint32_t i = 0u; i < 4u; ++i) {
        const size_t at = static_cast<size_t>(gid) * 4u + i;
        if (i < n) {
            upoint[at] = m.points[i].position;
            unormal[at] = m.points[i].normal;
            udepth[at] = m.points[i].penetration;
            // Witness buffers exist only with OGC slots.
            if (kConvex && uwitness_a != nullptr) {
                uwitness_a[at] = witness_a[i];
                uwitness_b[at] = witness_b[i];
            }

            ucontact_a[at] = a;
            ucontact_b[at] = b;
            ucontact_a_kind[at] = ::nuka::nk::kUContactSideBody;
            ucontact_b_kind[at] = ::nuka::nk::kUContactSideBody;
            ucontact_gen[at] = gen;
            ::nuka::nk::CanonicalContactDescriptor descriptor;
            descriptor.a.handle = a;
            descriptor.b.handle = b;
            descriptor.local_point_a = PrimInverseTransformPoint(
                body_pose[env * bodies_per_env + a], m.points[i].position);
            descriptor.local_point_b = PrimInverseTransformPoint(
                body_pose[env * bodies_per_env + b], m.points[i].position);
            descriptor.normal = m.points[i].normal;
            const uint32_t feature = m.points[i].stable_key == 0u
                ? ::nuka::nk::kContactFeatureUnavailable
                : static_cast<uint32_t>(m.points[i].stable_key);
            descriptor.feature_a = feature;
            descriptor.feature_b = feature;
            descriptor.manifold_slot = i;
            const ::nuka::nk::ContactId id = ::nuka::nk::MakeContactId(descriptor);
            ucontact_id_pair[at] = id.pair;
            ucontact_id_feature[at] = id.feature;
        }
    }
    if (n > 0u && contact_count != nullptr) {
        atomicAdd(&contact_count[env], n);
    }
}

__global__ void ZeroEnvKernel(uint32_t* __restrict__ contact_count,
                              uint32_t* __restrict__ row_count,
                              uint32_t* __restrict__ mesh_ogc_pair_count,
                              uint32_t* __restrict__ mesh_sdf_pair_count, uint32_t n,
                              uint32_t* __restrict__ ucount, uint32_t slot_stride,
                              uint32_t rigid_slots) {
    const uint32_t e = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t compact = slot_stride - rigid_slots;
    if (e < n * compact) {
        const uint32_t env = e / compact;
        ucount[env * slot_stride + rigid_slots + (e - env * compact)] = 0u;
    }
    if (e >= n) return;
    contact_count[e] = 0u;
    row_count[e] = 0u;
    mesh_ogc_pair_count[e] = 0u;
    mesh_sdf_pair_count[e] = 0u;
}

}  

Status LaunchPairDrivenNarrowphase(const ModelView& model, const DataView& data,
                                   const NarrowphasePrimitivesParams& p,
                                   cudaStream_t stream) {
    if (p.env_count == 0u || p.union_slot_count == 0u) {
        return Status::Ok;
    }
    const uint32_t rigid_slots =
        p.rigid_slot_cap < p.union_slot_count ? p.rigid_slot_cap : p.union_slot_count;
    {
        const uint32_t zero = p.env_count * (p.union_slot_count - rigid_slots);
        const uint32_t b = ((zero > p.env_count ? zero : p.env_count) + kBlock - 1u) / kBlock;
        LaunchCuda(ZeroEnvKernel, dim3(b), dim3(kBlock), 0u, stream,
                   data.contact_count, data.row_count, data.mesh_ogc_pair_count,
                   data.mesh_sdf_pair_count, p.env_count, data.ucontact_count,
                   p.union_slot_count, rigid_slots);
    }
    if (rigid_slots == 0u) return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
    const uint32_t total = p.env_count * rigid_slots;

    constexpr uint32_t kGen = 1u;
    for (const bool convex : {false, true}) {
        if (convex && p.convex_pairs == 0u) break;
        const uint32_t threads = convex ? kWarp : 1u;
        const uint32_t blocks = (uint64_t{total} * threads + kBlock - 1u) / kBlock;
        LaunchCuda(convex ? PairDrivenNarrowphaseKernel<true> : PairDrivenNarrowphaseKernel<false>,
                   dim3(blocks), dim3(kBlock), 0u, stream,
                   data.candidate_pairs, data.pair_count,
                   static_cast<const float*>(model.shape_table),
                   static_cast<const float*>(data.mat_buckets), p.num_material_buckets,
                   static_cast<const uint64_t*>(model.contact_pair_keys),
                   static_cast<const uint32_t*>(model.contact_pair_profiles), p.contact_pairs,
                   model.mesh_contact_mode,
                   static_cast<const math::Transform*>(data.body_pose),
                   data.body_linear_velocity, data.body_angular_velocity,
                   data.body_aabb_lo, data.body_aabb_hi, data.dat_body_motion,
                   static_cast<const float*>(model.hull_verts), p.hull_vert_count, kGen,
                   fmaxf(p.contact_margin, ::nuka::collision::kRigidContactSeparation),
                   p.dt,
                   p.env_count, p.union_slot_count, rigid_slots, p.bodies_per_env,
                   data.ucontact_count, data.ucontact_point, data.ucontact_normal,
                   data.ucontact_depth, data.ucontact_witness_a, data.ucontact_witness_b,
                   data.ucontact_law, data.ucontact_a, data.ucontact_b,
                   data.ucontact_a_kind, data.ucontact_b_kind,
                   data.ucontact_gen, data.ucontact_id_pair,
                   data.ucontact_id_feature, data.contact_count,
                   data.mesh_ogc_pair_count, data.mesh_sdf_pair_count);
    }
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  
