// Synchronize collidable poses from their articulation or free-body owners.
// Owner indices are environment-local; an optional mask selects environments.

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
    uint32_t env_count,
    const uint32_t* env_ids,
    math::Transform* __restrict__ body_pose) {
    const uint32_t work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= total_links) return;
    const uint32_t env_slot = work / links_per_env;
    const uint32_t env = env_ids ? env_ids[env_slot] : env_slot;
    if (env >= env_count) return;
    const uint32_t gl = env * links_per_env + work % links_per_env;
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

// Compose each proxy's local shape offset with its owner's world pose.
// Owners are never proxies, so the kernel cannot overwrite another thread's input.
__global__ void SyncProxyCollidablePoseKernel(
    const math::Transform* __restrict__ link_pose,          // FK world poses (env*L), may be null
    const uint32_t* __restrict__ body_collidable_link,      // template-local link, or ~0u
    const math::Transform* __restrict__ body_collidable_local,  // shape offset in the owner frame
    const uint32_t* __restrict__ body_collidable_body,      // template-local body row, or ~0u
    uint32_t total_bodies,                                   // env_count * bodies_per_env
    uint32_t links_per_env,
    uint32_t bodies_per_env,
    uint32_t env_count,
    const uint32_t* env_ids,
    math::Transform* __restrict__ body_pose) {
    const uint32_t work = blockIdx.x * blockDim.x + threadIdx.x;
    if (work >= total_bodies) return;
    const uint32_t env_slot = work / bodies_per_env;
    const uint32_t env = env_ids ? env_ids[env_slot] : env_slot;
    if (env >= env_count) return;
    const uint32_t gb = env * bodies_per_env + work % bodies_per_env;
    const uint32_t link_local = body_collidable_link[gb];
    if (link_local != ~uint32_t(0)) {
        if (link_pose == nullptr || link_local >= links_per_env) return;
        const uint32_t gl = env * links_per_env + link_local;
        body_pose[gb] = amf::ComposeTransformHD(link_pose[gl], body_collidable_local[gb]);
        return;
    }
    if (body_collidable_body == nullptr) return;
    const uint32_t body_local = body_collidable_body[gb];
    if (body_local == ~uint32_t(0) || body_local >= bodies_per_env) return;  // not a proxy.
    const uint32_t owner = env * bodies_per_env + body_local;
    body_pose[gb] = amf::ComposeTransformHD(body_pose[owner], body_collidable_local[gb]);
}

Status OpSyncLinkBodyPose(const ModelView& model, const DataView& data,
                          const void* params, cudaStream_t stream) {
    const auto* p = static_cast<const SyncLinkBodyPoseParams*>(params);
    if (p == nullptr) return Status::Failed;
    if (p->family != kContactFamilyPairDriven) return Status::Ok;  // early-exit.
    if (p->env_count == 0u || p->bodies_per_env == 0u) return Status::Ok;
    if (data.body_pose == nullptr) return Status::Ok;  // nothing to sync.
    if (p->selected_env_count > p->env_count ||
        (p->selected_env_count > 0u && data.reset_env_ids == nullptr)) return Status::Failed;
    const uint32_t selected_count = p->selected_env_count > 0u ? p->selected_env_count : p->env_count;
    const uint32_t* env_ids = p->selected_env_count > 0u ? data.reset_env_ids : nullptr;
    // Link collidables use the selected environment's FK state.
    if (data.link_pose != nullptr && model.link_body != nullptr &&
        p->links_per_env > 0u) {
        const uint32_t total = selected_count * p->links_per_env;
        const uint32_t blocks = (total + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(SyncLinkBodyPoseKernel, dim3(blocks), dim3(kBlockSize), 0u, stream,
                   static_cast<const math::Transform*>(data.link_pose),
                   static_cast<const uint32_t*>(model.link_body),
                   static_cast<const uint32_t*>(model.link_geom_kind),
                   static_cast<const math::Transform*>(model.link_geom_local),
                   total, p->links_per_env, p->bodies_per_env, p->env_count, env_ids,
                   static_cast<math::Transform*>(data.body_pose));
    }
    // Additional collision shapes use the same owner state and environment mask.
    if (model.body_collidable_link != nullptr &&
        model.body_collidable_local != nullptr) {
        const uint32_t total_bodies = selected_count * p->bodies_per_env;
        const uint32_t body_blocks = (total_bodies + kBlockSize - 1u) / kBlockSize;
        LaunchCuda(SyncProxyCollidablePoseKernel, dim3(body_blocks), dim3(kBlockSize),
                   0u, stream,
                   static_cast<const math::Transform*>(data.link_pose),
                   static_cast<const uint32_t*>(model.body_collidable_link),
                   static_cast<const math::Transform*>(model.body_collidable_local),
                   static_cast<const uint32_t*>(model.body_collidable_body),
                   total_bodies, p->links_per_env, p->bodies_per_env, p->env_count, env_ids,
                   static_cast<math::Transform*>(data.body_pose));
    }
    return (cudaGetLastError() == cudaSuccess) ? Status::Ok : Status::Failed;
}

}  // namespace

void RegisterNkSyncBodyPoseOps() {
    SetCudaOp(NkOp::SyncLinkBodyPose, &OpSyncLinkBodyPose);
}

}  // namespace nuka::phi
