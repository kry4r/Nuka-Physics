#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::gpu::broadphase -- CUDA AABB and pair generation
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "collision/dynamic_broadphase.hpp"
#include "phi/buffer.hpp"
#include "phi/device_context.hpp"

#include <cstdint>
#include <vector>

namespace nuka::collision::gpu {

class CudaBroadphaseResult {
public:
    CudaBroadphaseResult() = default;
    CudaBroadphaseResult(uint32_t shape_count,
                         uint32_t pair_slot_count,
                         phi::Buffer* aabbs,
                         phi::Buffer* pairs,
                         phi::Buffer* pair_active_flags,
                         phi::Buffer* pair_count);
    ~CudaBroadphaseResult();

    CudaBroadphaseResult(const CudaBroadphaseResult&) = delete;
    CudaBroadphaseResult& operator=(const CudaBroadphaseResult&) = delete;
    CudaBroadphaseResult(CudaBroadphaseResult&& other) noexcept;
    CudaBroadphaseResult& operator=(CudaBroadphaseResult&& other) noexcept;

    uint32_t ShapeCount() const { return shape_count_; }
    uint32_t PairSlotCount() const { return pair_slot_count_; }
    uint32_t PairCount() const;

    const collision::AABB* DeviceAabbs() const;
    const collision::CollisionPair* DevicePairs() const;
    const uint8_t* DevicePairActiveFlags() const;
    const uint32_t* DevicePairCount() const;

    std::vector<collision::AABB> DownloadAabbs() const;
    std::vector<collision::CollisionPair> DownloadPairs() const;

private:
    uint32_t shape_count_ = 0;
    uint32_t pair_slot_count_ = 0;
    phi::Buffer* aabbs_ = nullptr;  // phi v2 handles; freed in the dtor.
    phi::Buffer* pairs_ = nullptr;
    phi::Buffer* pair_active_flags_ = nullptr;
    phi::Buffer* pair_count_ = nullptr;
};

// Run the SAP O(n^2) pair-slot broadphase directly over a device AABB buffer
// (the oracle entry for the LBVH differential gate). The AABBs are the caller's;
// the result borrows nothing from them (its own device pair buffers are built).
CudaBroadphaseResult BuildCudaBroadphase(const phi::DeviceContext& context,
                                         const collision::AABB* device_aabbs,
                                         uint32_t shape_count);
CudaBroadphaseResult BuildCudaBroadphase(const collision::AABB* device_aabbs,
                                         uint32_t shape_count);

} // namespace nuka::collision::gpu
