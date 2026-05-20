#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::gpu::broadphase -- CUDA AABB and pair generation
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "collision/dynamic_broadphase.hpp"
#include "phi/buffer.hpp"
#include "runtime/gpu/device_world.hpp"

#include <cstdint>
#include <vector>

namespace nuka::collision::gpu {

class CudaBroadphaseResult {
public:
    CudaBroadphaseResult() = default;
    CudaBroadphaseResult(uint32_t shape_count,
                         uint32_t pair_slot_count,
                         phi::Buffer aabbs,
                         phi::Buffer pairs,
                         phi::Buffer pair_active_flags,
                         phi::Buffer pair_count);

    CudaBroadphaseResult(const CudaBroadphaseResult&) = delete;
    CudaBroadphaseResult& operator=(const CudaBroadphaseResult&) = delete;
    CudaBroadphaseResult(CudaBroadphaseResult&&) noexcept = default;
    CudaBroadphaseResult& operator=(CudaBroadphaseResult&&) noexcept = default;

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
    phi::Buffer aabbs_;
    phi::Buffer pairs_;
    phi::Buffer pair_active_flags_;
    phi::Buffer pair_count_;
};

CudaBroadphaseResult BuildCudaBroadphase(const runtime::gpu::DeviceWorld& device_world);

} // namespace nuka::collision::gpu
