// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — M4 UNION-family narrowphase (NarrowphasePrimitives
// union branch): one thread per (env x union contact slot), running the SAME
// HD-clean analytic handlers the legacy BatchedUnifiedWorld used —
//   foot   : amf::SpherePlane  (the legacy HOST C3b foot narrowphase)
//   finger : cvx::SphereHull   (the legacy GPU grasp narrowphase kernel body)
//   table  : amf::BoxPlane     (the legacy HOST C3b cup-proxy x table)
// — writing the per-slot manifold (<=4 points) into the ucontact_* Data scratch.
//
// Geometry inputs are the LIVE device state: link_pose (refreshed by
// FkWorldPoses this step — the same UpdateWorldLinkPoses output the legacy
// downloads fed) and body_pose. The fingertip / foot sphere center and the box
// proxy pose are formed with the EXACT host expressions (RotateQuatHostExpr ==
// math::Quat::Rotate; the PrimFrame bake == amf::BuildPrimFrame columns), so a
// device-detected manifold matches the legacy host-detected one at the FP
// floor (the same ULP class the G1c gates absorbed for the GPU finger seam).
//
// Gating: a slot with flags bit0 (kUSlotGatedOnTable) only emits while the
// per-env table_enabled Data field is non-zero — the SetTableEnabled lift
// choreography is a field write, never a graph rebuild (plan §3.2).
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "collision/analytical_manifold.hpp"   // amf::SpherePlane/BoxPlane (HD)
#include "collision/convex_narrowphase.hpp"    // cvx::SphereHull (HD)
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/union_types.cuh"

namespace nuka::phi::nkops {

namespace {

namespace amf = ::nuka::collision::amf;
namespace cvx = ::nuka::collision::cvx;
using ::nuka::constraint::ContactManifold;

// Device PrimFrame bake — EXACT amf::BuildPrimFrame (host) expressions, with
// Quat::Rotate replicated by RotateQuatHostExpr.
__device__ amf::PrimFrame BuildPrimFrameDev(const math::Transform& xf) {
    amf::PrimFrame f;
    f.cx = RotateQuatHostExpr(xf.rotation, math::Vec3{1.0f, 0.0f, 0.0f});
    f.cy = RotateQuatHostExpr(xf.rotation, math::Vec3{0.0f, 1.0f, 0.0f});
    f.cz = RotateQuatHostExpr(xf.rotation, math::Vec3{0.0f, 0.0f, 1.0f});
    f.t = xf.position;
    return f;
}

// GroundPrim — the legacy +Z plane prim (frame.cy = world +Z), verbatim.
__device__ amf::PrimParams GroundPrimDev(float height) {
    amf::PrimParams p;
    p.frame.cx = math::Vec3{1.0f, 0.0f, 0.0f};
    p.frame.cy = math::Vec3{0.0f, 0.0f, 1.0f};
    p.frame.cz = math::Vec3{0.0f, -1.0f, 0.0f};
    p.frame.t = math::Vec3{0.0f, 0.0f, height};
    return p;
}

// The shared-memory hull-vertex cache capacity (verts x 3 floats; 2048 x 12 B
// = 24 KB). The C7a cup hull is 1796 verts; the GJK support scans walk the
// WHOLE pool per iteration, so caching it in shared is the dominant
// narrowphase win (measured 2.16 ms -> sub-ms at N=1). A hull beyond the cap
// fails LOUDLY at the op entry (capacity is a Model property; M5 revisits).
constexpr uint32_t kMaxSharedHullVerts = 2048u;

// One BLOCK (a single warp) per (env x union slot): the slot's analytic /
// GJK narrowphase is a SERIAL per-slot computation (the fixed-cap GJK loops),
// so spreading slots across blocks puts every slot on its own SM partition
// (the former all-slots-in-one-block layout serialized the warp on inter-slot
// branch divergence — measured ~2 ms at N=1). The warp cooperatively stages
// the hull-vertex pool into shared; lane 0 runs the slot. Each slot owns its
// ucontact_* entries — no atomics, fixed order, fixed iteration caps -> D1.
__global__ void UnionNarrowphaseKernel(const float* __restrict__ union_slots,
                                       const float* __restrict__ hull_verts,
                                       uint32_t hull_vert_count,
                                       const math::Transform* __restrict__ link_pose,
                                       const math::Transform* __restrict__ body_pose,
                                       const math::Vec3* __restrict__ particle_pos,
                                       uint32_t particles_per_env,
                                       const uint32_t* __restrict__ table_enabled,
                                       uint32_t env_count,
                                       uint32_t slot_count,
                                       uint32_t base_link_count,
                                       uint32_t bodies_per_env,
                                       uint32_t* __restrict__ out_count,
                                       math::Vec3* __restrict__ out_point,
                                       math::Vec3* __restrict__ out_normal,
                                       float* __restrict__ out_depth,
                                       uint32_t* __restrict__ contact_count) {
    const uint32_t gid = blockIdx.x;  // one block per (env x slot)
    const uint32_t total = env_count * slot_count;
    if (gid >= total) {
        return;
    }
    const uint32_t env = gid / slot_count;
    const uint32_t slot = gid - env * slot_count;
    const UnionSlotDev u = LoadUnionSlot(union_slots, slot);

    // Cooperative SHARED hull-vertex cache + bounding radius (finger slots
    // only — the analytic classes never scan the pool). The bound is fmaxf
    // over |v|^2 — order-independent (fmax is associative/commutative on
    // finite floats), warp-reduced, used ONLY for a CONSERVATIVE early-out
    // (|center - hull_origin| > bound + radius => SphereHull would return
    // no-contact anyway: every hull point is within `bound` of the origin).
    __shared__ float hull_sh[kMaxSharedHullVerts * 3u];
    __shared__ float bound_sh[32u];
    float hull_bound = 0.0f;
    const uint32_t cached = (u.cls == kUSlotFingerSphereHull &&
                             hull_vert_count <= kMaxSharedHullVerts)
                                ? hull_vert_count : 0u;
    if (cached > 0u) {
        float local_max_sq = 0.0f;
        for (uint32_t v = threadIdx.x; v < cached; v += blockDim.x) {
            const float x = hull_verts[3u * v + 0u];
            const float y = hull_verts[3u * v + 1u];
            const float z = hull_verts[3u * v + 2u];
            hull_sh[3u * v + 0u] = x;
            hull_sh[3u * v + 1u] = y;
            hull_sh[3u * v + 2u] = z;
            local_max_sq = fmaxf(local_max_sq, x * x + y * y + z * z);
        }
        bound_sh[threadIdx.x] = local_max_sq;
        __syncthreads();
        if (threadIdx.x == 0u) {
            float m = 0.0f;
            for (uint32_t t = 0u; t < blockDim.x; ++t) m = fmaxf(m, bound_sh[t]);
            bound_sh[0] = sqrtf(m);
        }
        __syncthreads();
        hull_bound = bound_sh[0];
    }
    const float* const hull_pool = cached > 0u ? hull_sh : hull_verts;

    // The FINGER class runs ALL 32 lanes in LOCKSTEP (identical inputs ->
    // identical control flow) so SupportHull's warp-cooperative scan can split
    // the 1796-vert pool across the warp (the measured GJK bottleneck); the
    // analytic classes are lane-0 scalar work. Lane 0 alone writes outputs.
    if (u.cls != kUSlotFingerSphereHull && threadIdx.x != 0u) {
        return;
    }

    ContactManifold m;
    m.Clear();
    bool emitted = false;
    switch (u.cls) {
        case kUSlotFootSpherePlane: {
            const math::Transform lp =
                link_pose[static_cast<size_t>(env) * base_link_count + u.link];
            // center = lp.position + lp.rotation.Rotate(offset) — the legacy
            // host foot_centers expression verbatim.
            amf::PrimParams sphere;
            sphere.radius = u.radius;
            sphere.frame.t =
                lp.position + RotateQuatHostExpr(lp.rotation, u.offset);
            const amf::PrimParams plane = GroundPrimDev(u.plane_height);
            amf::SpherePlane(sphere, plane, &m);
            emitted = true;
            break;
        }
        case kUSlotFingerSphereHull: {
            if (hull_vert_count == 0u) break;
            const math::Transform lp =
                link_pose[static_cast<size_t>(env) * base_link_count + u.link];
            amf::PrimParams sphere;
            sphere.radius = u.radius;
            sphere.frame.t =
                lp.position + RotateQuatHostExpr(lp.rotation, u.offset);
            const math::Transform cup_pose =
                body_pose[static_cast<size_t>(env) * bodies_per_env + u.body];
            // CONSERVATIVE bound early-out: every hull point lies within
            // hull_bound of the hull origin, so a center farther than
            // bound + radius cannot touch — SphereHull would return the
            // no-contact empty manifold anyway (pure skip, no physics change).
            {
                const math::Vec3 d = sphere.frame.t - cup_pose.position;
                const float dist_sq = d.Dot(d);
                const float reach = hull_bound + u.radius;
                if (dist_sq > reach * reach) {
                    emitted = true;  // empty manifold (m stays cleared).
                    break;
                }
            }
            cvx::ConvexHullView hull;
            hull.verts = hull_pool;  // shared-cached pool (same bytes).
            hull.vcount = hull_vert_count;
            hull.frame = BuildPrimFrameDev(cup_pose);
            // FULL-WARP lockstep call (all lanes, identical inputs): the
            // support scans run warp-cooperatively (bit-identical argmax);
            // every lane computes the same manifold, lane 0 writes it out.
            hull.warp_lockstep = true;
            // sphere_is_a == true: pair = (fingertip=A=sphere, cup=B=hull) —
            // the legacy grasp kernel's exact call.
            cvx::SphereHull(sphere, hull, /*sphere_is_a=*/true, &m);
            emitted = true;
            break;
        }
        case kUSlotBodyBoxPlane: {
            const bool gated = (u.flags & kUSlotGatedOnTable) != 0u;
            if (gated && table_enabled[env] == 0u) break;
            const math::Transform bp =
                body_pose[static_cast<size_t>(env) * bodies_per_env + u.body];
            // proxy pose: position offset in the body frame (legacy table
            // branch verbatim); rotation = the body's.
            math::Transform proxy = bp;
            proxy.position = bp.position + RotateQuatHostExpr(bp.rotation, u.offset);
            amf::PrimParams box;
            box.half_extents = u.box_half;
            box.frame = BuildPrimFrameDev(proxy);
            const amf::PrimParams plane = GroundPrimDev(u.plane_height);
            amf::BoxPlane(box, plane, &m);
            emitted = true;
            break;
        }
        case kUSlotParticleSpherePlane: {
            // particle sphere (slot.link == LOCAL particle index) x +Z plane.
            const math::Vec3 pc =
                particle_pos[static_cast<size_t>(env) * particles_per_env + u.link];
            amf::PrimParams sphere;
            sphere.radius = u.radius;
            sphere.frame.t = pc;  // particle world position (identity rotation).
            const amf::PrimParams plane = GroundPrimDev(u.plane_height);
            amf::SpherePlane(sphere, plane, &m);
            emitted = true;
            break;
        }
        case kUSlotParticleSphereBox: {
            // particle sphere (slot.link == LOCAL particle index) x rigid box
            // (slot.body + offset + half_extents). The box pose is the body's.
            const bool gated = (u.flags & kUSlotGatedOnTable) != 0u;
            if (gated && table_enabled[env] == 0u) break;
            const math::Vec3 pc =
                particle_pos[static_cast<size_t>(env) * particles_per_env + u.link];
            amf::PrimParams sphere;
            sphere.radius = u.radius;
            sphere.frame.t = pc;
            const math::Transform bp =
                body_pose[static_cast<size_t>(env) * bodies_per_env + u.body];
            math::Transform proxy = bp;
            proxy.position = bp.position + RotateQuatHostExpr(bp.rotation, u.offset);
            amf::PrimParams box;
            box.half_extents = u.box_half;
            box.frame = BuildPrimFrameDev(proxy);
            amf::SphereBox(sphere, box, &m);
            emitted = true;
            break;
        }
        default:
            break;
    }

    if (threadIdx.x != 0u) {
        return;  // every lane holds the same manifold; lane 0 writes it.
    }
    const uint32_t n = emitted ? m.point_count : 0u;
    out_count[gid] = n;
    for (uint32_t i = 0u; i < 4u; ++i) {
        const size_t at = static_cast<size_t>(gid) * 4u + i;
        if (i < n) {
            out_point[at] = m.points[i].position;
            out_normal[at] = m.points[i].normal;
            out_depth[at] = m.points[i].penetration;
        } else {
            out_point[at] = math::Vec3{0.0f, 0.0f, 0.0f};
            out_normal[at] = math::Vec3{0.0f, 0.0f, 0.0f};
            out_depth[at] = 0.0f;
        }
    }
    if (n > 0u) {
        atomicAdd(&contact_count[env], n);  // order-independent sum (D1).
    }
}

__global__ void ZeroEnvCountsKernel(uint32_t* __restrict__ contact_count,
                                    uint32_t* __restrict__ row_count,
                                    uint32_t env_count) {
    const uint32_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= env_count) return;
    contact_count[env] = 0u;
    row_count[env] = 0u;
}

}  // namespace

Status LaunchUnionNarrowphase(const ModelView& model, const DataView& data,
                              const NarrowphasePrimitivesParams& p,
                              cudaStream_t stream) {
    if (p.env_count == 0u || p.union_slot_count == 0u) {
        return Status::Ok;
    }
    constexpr uint32_t kBlock = 128u;
    {
        const uint32_t blocks = (p.env_count + kBlock - 1u) / kBlock;
        LaunchCuda(ZeroEnvCountsKernel, dim3(blocks), dim3(kBlock), 0u, stream,
                   data.contact_count, data.row_count, p.env_count);
    }
    const uint32_t total = p.env_count * p.union_slot_count;
    // One single-warp block per slot (see the kernel header note).
    LaunchCuda(UnionNarrowphaseKernel, dim3(total), dim3(32u), 0u, stream,
               static_cast<const float*>(model.union_slots),
               static_cast<const float*>(model.hull_verts),
               p.hull_vert_count,
               static_cast<const math::Transform*>(data.link_pose),
               static_cast<const math::Transform*>(data.body_pose),
               static_cast<const math::Vec3*>(data.particle_pos),
               p.particles_per_env,
               static_cast<const uint32_t*>(data.table_enabled),
               p.env_count, p.union_slot_count, p.base_link_count,
               p.bodies_per_env,
               data.ucontact_count, data.ucontact_point, data.ucontact_normal,
               data.ucontact_depth, data.contact_count);
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  // namespace nuka::phi::nkops
