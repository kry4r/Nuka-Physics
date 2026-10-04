#pragma once

#include <cstdint>

#include <cuda_runtime.h>

#include "collision/mesh_surface_types.hpp"

namespace nuka::phi {
namespace {

constexpr uint32_t kWarpLanes = 32u;
constexpr uint32_t kTreeShareDepth = 2u;
constexpr uint32_t kTreeShares = 1u << kTreeShareDepth;

// Share `share` of a preorder tree: one subtree kTreeShareDepth levels below the root, or a leaf met
// above that depth for the share whose remaining path bits are zero. Parent bounds contain their
// children, so the shares reach exactly the leaves one whole-tree walk reaches.
__device__ bool TreeShare(const collision::MeshBvhNode* nodes, uint32_t count, uint32_t share,
                          uint32_t* begin, uint32_t* end) {
    *begin = *end = 0u;
    if (count == 0u) return true;
    uint32_t cursor = 0u;
    for (uint32_t level = 0u; level < kTreeShareDepth; ++level) {
        const auto& node = nodes[cursor];
        if (node.escape <= cursor || node.escape > count) return false;
        if (node.triangle != ~0u) {
            if ((share & ((1u << (kTreeShareDepth - level)) - 1u)) != 0u) return true;
            break;
        }
        const uint32_t left = cursor + 1u;
        if (left >= node.escape) return false;
        const uint32_t right = nodes[left].escape;
        if (right <= left || right >= node.escape) return false;
        cursor = (share >> (kTreeShareDepth - 1u - level)) & 1u ? right : left;
    }
    const uint32_t escape = nodes[cursor].escape;
    if (escape <= cursor || escape > count) return false;
    *begin = cursor;
    *end = escape;
    return true;
}

// Lanes queue the pairs their traversals reach and the warp tests them a full warp at a time, so the
// costly pair tests never run on just the few lanes that found one. Every lane of the warp must call.
template <typename Candidate, typename Next, typename Test>
__device__ void TestInWarpBatches(Candidate* queue, bool active, Next&& next, Test&& test) {
    const uint32_t lane = threadIdx.x % kWarpLanes;
    uint32_t queued = 0u;
    bool searching = active;
    for (;;) {
        Candidate candidate;
        searching = searching && next(&candidate);
        const uint32_t found = __ballot_sync(~0u, searching);
        if (searching) queue[queued + __popc(found & ((1u << lane) - 1u))] = candidate;
        queued += __popc(found);
        __syncwarp();
        if (queued >= kWarpLanes || (found == 0u && queued > 0u)) {
            const uint32_t batch = min(queued, kWarpLanes);
            queued -= batch;
            if (lane < batch) test(queue[queued + lane]);
            __syncwarp();
        }
        if (found == 0u && queued == 0u) return;
    }
}

}  // namespace
}  // namespace nuka::phi
