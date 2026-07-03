// ---------------------------------------------------------------------------
// PHI v2 CUDA backend — general contact pipeline Phase 0 (B2): SyncLinkBodyPose.
//
// Copies every articulation link's FK world pose (link_pose, written by
// FkWorldPoses) into its OWNING body_pose row, so articulation links become
// first-class collidables visible to the LBVH broadphase (BuildAabbsKernel reads
// body_pose[gid]). Without this op the per-link body rows hold a stale pose and
// an articulated robot is invisible to general body<->body contact — this op is
// the load-bearing prerequisite for the whole general path (design §2.6).
//
// FAMILY GATING (D1): the op EARLY-EXITS unless family == kContactFamilyPairDriven
// — exactly like the broadphase ops it feeds. The H1 grasp (UnionCsr) graph
// enqueues this op (the pipeline always inserts it when there are collidables) but
// it does NO work for the union family, so its captured graph / golden is
// byte-untouched. (L1-b: the FUSED family is gone; PairDriven is the general
// default, so this op now does real work for the locomotion/general cook.)
//
// INDEXING: link_body is the MODEL table (one TEMPLATE-LOCAL body row per link,
// tiled env-major without offset; staged by StampPerLink). For a GLOBAL link
// gl = env*L + l the owning GLOBAL body row is env*B + link_body[gl]. The
// collidable's offset from the link frame is the cooked link_geom_local; we
// compose it (HD-clean) ONLY when the link carries a cooked collision shape
// (link_geom_kind != 0), because an un-cooked link_geom_local is a zero-filled
// Transform (rotation == {0,0,0,0}, NOT identity) and composing it would corrupt
// the pose. A link with no cooked shape writes its raw FK link_pose.
// ---------------------------------------------------------------------------

#include <cuda_runtime.h>

#include "collision/analytical_manifold.hpp"   // amf::ComposeTransformHD (HD)
#include "math/transform.hpp"
#include "nk/model/generated/views.hpp"        // ModelView / DataView
#include "phi/backend_cuda/launch.cuh"
#include "phi/backend_cuda/ops/nk_op_registrations.cuh"
#include "phi/backend_cuda/ops/registry.cuh"
#include "phi/op_schema.hpp"

namespace nuka::phi {

namespace {

namespace amf = ::nuka::collision::amf;
constexpr uint32_t kBlockSize = 128u;

// One thread per (env x link). body_pose[env*B + link_body[gl]] =
// link_pose[gl] (o link_geom_local[gl] when the link has a cooked shape).
__global__ void SyncLinkBodyPoseKernel(
    const math::Transform* __restrict__ link_pose,        // FK world poses (env*L)
    const uint32_t* __restrict__ link_body,               // template-local body row / link
    const uint32_t* __restrict__ link_geom_kind,          // 0 == no cooked shape
    const math::Transform* __restrict__ link_geom_local,  // shape's link-local xform
    uint32_t total_links,                                  // env_count * links_per_env
    uint32_t links_per_env,
    uint32_t bodies_per_env,
    math::Transform* __restrict__ body_pose) {
    const uint32_t gl = blockIdx.x * blockDim.x + threadIdx.x;
    if (gl >= total_links) return;
    const uint32_t env = gl / links_per_env;
    const uint32_t b_local = link_body[gl];
    if (b_local >= bodies_per_env) return;  // link owns no movable body row.
    const uint32_t gb = env * bodies_per_env + b_local;

    math::Transform world = link_pose[gl];
    if (link_geom_kind != nullptr && link_geom_kind[gl] != 0u &&
        link_geom_local != nullptr) {
        world = amf::ComposeTransformHD(world, link_geom_local[gl]);
    }
    body_pose[gb] = world;
}

// One thread per (env x body). A COLLIDABLE PROXY body row (an extra collision
// geom of an articulation link) carries its source template-local link in
// body_collidable_link (~0u == not a proxy) and the geom's link-local offset in
// body_collidable_local; pose it from the source link's FK world pose. Multiple
// proxies per link give a link multiple collidables (multi-geom feet). Non-proxy
// rows (~0u) are skipped, so a single-geom world writes nothing here (no-op).
__global__ void SyncProxyCollidablePoseKernel(
    const math::Transform* __restrict__ link_pose,          // FK world poses (env*L)
    const uint32_t* __restrict__ body_collidable_link,      // template-local link, or ~0u
    const math::Transform* __restrict__ body_collidable_local,  // geom offset in the link frame
    uint32_t total_bodies,                                   // env_count * bodies_per_env
    uint32_t links_per_env,
    uint32_t bodies_per_env,
    math::Transform* __restrict__ body_pose) {
    const uint32_t gb = blockIdx.x * blockDim.x + threadIdx.x;
    if (gb >= total_bodies) return;
    const uint32_t link_local = body_collidable_link[gb];
    if (link_local == ~uint32_t(0) || link_local >= links_per_env) return;  // not a proxy.
    const uint32_t env = gb / bodies_per_env;
    const uint32_t gl = env * links_per_env + link_local;
    body_pose[gb] = amf::ComposeTransformHD(link_pose[gl], body_collidable_local[gb]);
}

Status OpSyncLinkBodyPose(const ModelView& model, const DataView& data,
                          const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const SyncLinkBodyPoseParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->family != kContactFamilyPairDriven) return Status::Ok;  // early-exit.
    if (p->env_count == 0u || p->links_per_env == 0u ||
        p->bodies_per_env == 0u) {
        return Status::Ok;
    }
    if (data.link_pose == nullptr || model.link_body == nullptr ||
        data.body_pose == nullptr) {
        return Status::Ok;  // nothing to sync.
    }
    const uint32_t total = p->env_count * p->links_per_env;
    const uint32_t blocks = (total + kBlockSize - 1u) / kBlockSize;
    LaunchCuda(SyncLinkBodyPoseKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
               static_cast<const math::Transform*>(data.link_pose),
               static_cast<const uint32_t*>(model.link_body),
               static_cast<const uint32_t*>(model.link_geom_kind),
               static_cast<const math::Transform*>(model.link_geom_local),
               total, p->links_per_env, p->bodies_per_env,
               static_cast<math::Transform*>(data.body_pose));
    // Extra collidable proxies (multi-geom feet): pose the appended proxy body
    // rows from their source link. A no-op when the model authored none (every
    // body_collidable_link == ~0u), so single-geom worlds are byte-untouched.
    if (model.body_collidable_link != nullptr &&
        model.body_collidable_local != nullptr) {
        const uint32_t total_bodies = p->env_count * p->bodies_per_env;
        const uint32_t body_blocks = (total_bodies + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(SyncProxyCollidablePoseKernel, dim3(body_blocks), dim3(kBlockSize),
                   0u, stream,
                   static_cast<const math::Transform*>(data.link_pose),
                   static_cast<const uint32_t*>(model.body_collidable_link),
                   static_cast<const math::Transform*>(model.body_collidable_local),
                   total_bodies, p->links_per_env, p->bodies_per_env,
                   static_cast<math::Transform*>(data.body_pose));
    }
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkSyncBodyPoseOps() {
    SetCudaOp(NkOp::SyncLinkBodyPose, &OpSyncLinkBodyPose);
}

}  // namespace nuka::phi
