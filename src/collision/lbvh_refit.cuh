#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::gpu -- LBVH refit primitive (Task 7.4.5) declaration.
//
// Re-propagates fresh leaf AABBs up an EXISTING LBVH (topology reused). See
// lbvh_refit.cu for the contract. Header is .cuh because LbvhNode is a CUDA
// type; included only from .cu TUs.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "collision/lbvh_node.cuh"
#include "phi/device_context.hpp"

#include <cstdint>

namespace nuka::collision::gpu {

// Refit `leaf_count` leaves of an existing tree in `device_nodes` (the flat
// 2N-1 node array produced by BuildLbvhBroadphase) using the fresh per-body
// bounds in `device_new_aabbs` (indexed by ORIGINAL body id). `device_visit_
// scratch` must hold at least (leaf_count-1) uint32 (zeroed internally).
void RefitLbvh(const phi::DeviceContext& context,
               LbvhNode* device_nodes,
               const collision::AABB* device_new_aabbs,
               uint32_t leaf_count,
               uint32_t* device_visit_scratch);

} // namespace nuka::collision::gpu
