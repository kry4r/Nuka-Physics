// ---------------------------------------------------------------------------
// nuka::collision::gpu -- LBVH per-frame refit (Task 7.4.5).
//
// When body count + Morton ordering are stable across frames, the tree TOPOLOGY
// (child/parent links, leaf->body mapping) can be reused; only the AABB bounds
// need refreshing. Refit re-loads each leaf's new AABB and re-propagates bounds
// bottom-up using the SAME integer-atomic visit-counter scheme as the build, so
// it is dramatically cheaper than a full rebuild while preserving D1 (the merge
// is order-independent; the only atomics are uint32).
//
// Heuristic (caller-owned): refit for N frames, then rebuild (N ~ 10-50) to
// bound tree-quality degradation. This TU provides the refit primitive; the
// rebuild decision lives with the caller.
// ---------------------------------------------------------------------------

#include "collision/lbvh_node.cuh"
#include "collision/lbvh_refit.cuh"
#include "phi/device_context.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace nuka::collision::gpu {

namespace {

constexpr uint32_t kBlockSize = 128u;

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " refit failed: " +
                                 cudaGetErrorString(result));
    }
}

// Re-load each leaf's bound from the fresh AABB array. Topology (parent links,
// leaf->body mapping in `left`) is untouched.
__global__ void RefitLeavesKernel(uint32_t leaf_count,
                                  const collision::AABB* __restrict__ new_aabbs,
                                  LbvhNode* __restrict__ nodes) {
    const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= leaf_count) {
        return;
    }
    const uint32_t internal_count = leaf_count - 1u;
    const uint32_t node_idx = internal_count + lane;
    const int32_t body = nodes[node_idx].left; // original body id stored at build
    nodes[node_idx].aabb = new_aabbs[body];
}

// Bottom-up merge identical to the build's PropagateAabbsKernel: the second
// child to reach a node (uint32 atomic counter) merges both child bounds. D1:
// merge is order-independent.
__global__ void RefitPropagateKernel(uint32_t leaf_count,
                                     LbvhNode* __restrict__ nodes,
                                     uint32_t* __restrict__ visit) {
    const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= leaf_count) {
        return;
    }
    const uint32_t internal_count = leaf_count - 1u;
    int32_t node = nodes[internal_count + lane].parent;
    while (node >= 0) {
        // Same Karras bottom-up visibility fence as the build's propagation:
        // make this thread's child-bound write visible before signaling arrival,
        // so the second arriver never merges a stale child AABB.
        __threadfence();
        const uint32_t prev = atomicAdd(&visit[node], 1u);
        if (prev == 0u) {
            return;
        }
        const LbvhNode self = nodes[node];
        nodes[node].aabb = LbvhMerge(nodes[self.left].aabb, nodes[self.right].aabb);
        node = nodes[node].parent;
    }
}

} // namespace

void RefitLbvh(const phi::DeviceContext& context,
               LbvhNode* device_nodes,
               const collision::AABB* device_new_aabbs,
               uint32_t leaf_count,
               uint32_t* device_visit_scratch) {
    if (leaf_count < 2u) {
        return;
    }
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();
    const uint32_t internal_count = leaf_count - 1u;
    const uint32_t blocks = (leaf_count + kBlockSize - 1u) / kBlockSize;

    CheckCuda(cudaMemsetAsync(device_visit_scratch, 0,
                              internal_count * sizeof(uint32_t), stream),
              "cudaMemsetAsync visit");

    RefitLeavesKernel<<<blocks, kBlockSize, 0, stream>>>(
        leaf_count, device_new_aabbs, device_nodes);
    CheckCuda(cudaGetLastError(), "RefitLeavesKernel launch");

    RefitPropagateKernel<<<blocks, kBlockSize, 0, stream>>>(
        leaf_count, device_nodes, device_visit_scratch);
    CheckCuda(cudaGetLastError(), "RefitPropagateKernel launch");
}

} // namespace nuka::collision::gpu
