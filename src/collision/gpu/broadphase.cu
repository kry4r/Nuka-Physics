// ---------------------------------------------------------------------------
// nuka::collision::gpu::broadphase implementation
// ---------------------------------------------------------------------------

#include "collision/gpu/broadphase.cuh"

#include "phi/buffer_transfer.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nuka::collision::gpu {

namespace {

// Buffer helpers come from phi/buffer_transfer.hpp. This SAP path now consumes a
// caller-provided device AABB buffer directly (the per-shape AABB derivation that
// used to live here was tied to the removed whole-world device container), so the
// only device math needed is the AABB overlap test below.

__device__ bool Overlaps(collision::AABB a, collision::AABB b) {
    if (a.max.x < b.min.x || a.min.x > b.max.x) return false;
    if (a.max.y < b.min.y || a.min.y > b.max.y) return false;
    if (a.max.z < b.min.z || a.min.z > b.max.z) return false;
    return true;
}

__device__ uint32_t PairSlot(uint32_t shape_count, uint32_t i, uint32_t j) {
    return (i * (2u * shape_count - i - 1u)) / 2u + (j - i - 1u);
}

__global__ void GeneratePairSlotsKernel(uint32_t shape_count,
                                        const collision::AABB* aabbs,
                                        collision::CollisionPair* pairs,
                                        uint8_t* pair_active_flags,
                                        uint32_t* pair_count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= shape_count) {
        return;
    }

    for (uint32_t j = i + 1u; j < shape_count; ++j) {
        const uint32_t slot = PairSlot(shape_count, i, j);
        pairs[slot] = {i, j};
        const bool overlaps = Overlaps(aabbs[i], aabbs[j]);
        pair_active_flags[slot] = overlaps ? 1u : 0u;
        if (overlaps) {
            atomicAdd(pair_count, 1u);
        }
    }
}

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

// DownloadVector(buf, count) now comes from the shared host buffer-transfer
// header (phi/buffer_transfer.hpp); the former local copy was byte-identical.
using ::nuka::phi::DownloadVector;

} // namespace

CudaBroadphaseResult::CudaBroadphaseResult(uint32_t shape_count,
                                           uint32_t pair_slot_count,
                                           phi::Buffer aabbs,
                                           phi::Buffer pairs,
                                           phi::Buffer pair_active_flags,
                                           phi::Buffer pair_count)
    : shape_count_(shape_count)
    , pair_slot_count_(pair_slot_count)
    , aabbs_(std::move(aabbs))
    , pairs_(std::move(pairs))
    , pair_active_flags_(std::move(pair_active_flags))
    , pair_count_(std::move(pair_count)) {}

uint32_t CudaBroadphaseResult::PairCount() const {
    if (pair_count_.Size() == 0u) {
        return 0u;
    }

    uint32_t pair_count = 0;
    pair_count_.CopyToHost(&pair_count, sizeof(pair_count));
    return pair_count;
}

const collision::AABB* CudaBroadphaseResult::DeviceAabbs() const {
    return static_cast<const collision::AABB*>(aabbs_.Data());
}

const collision::CollisionPair* CudaBroadphaseResult::DevicePairs() const {
    return static_cast<const collision::CollisionPair*>(pairs_.Data());
}

const uint8_t* CudaBroadphaseResult::DevicePairActiveFlags() const {
    return static_cast<const uint8_t*>(pair_active_flags_.Data());
}

const uint32_t* CudaBroadphaseResult::DevicePairCount() const {
    return static_cast<const uint32_t*>(pair_count_.Data());
}

std::vector<collision::AABB> CudaBroadphaseResult::DownloadAabbs() const {
    return DownloadVector<collision::AABB>(aabbs_, shape_count_);
}

std::vector<collision::CollisionPair> CudaBroadphaseResult::DownloadPairs() const {
    const auto pairs = DownloadVector<collision::CollisionPair>(pairs_, pair_slot_count_);
    const auto flags = DownloadVector<uint8_t>(pair_active_flags_, pair_slot_count_);

    std::vector<collision::CollisionPair> active_pairs;
    active_pairs.reserve(PairCount());
    for (uint32_t slot = 0; slot < pair_slot_count_; ++slot) {
        if (flags[slot] != 0u) {
            active_pairs.push_back(pairs[slot]);
        }
    }
    return active_pairs;
}

CudaBroadphaseResult BuildCudaBroadphase(const phi::DeviceContext& context,
                                         const collision::AABB* device_aabbs,
                                         uint32_t shape_count) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();

    const uint32_t pair_slot_count = shape_count > 1u
        ? (shape_count * (shape_count - 1u)) / 2u
        : 0u;

    // The result owns its own AABB copy so DeviceAabbs()/DownloadAabbs() stay
    // valid independent of the caller's buffer lifetime.
    phi::Buffer aabbs(shape_count * sizeof(collision::AABB), phi::MemoryKind::Device);
    phi::Buffer pairs(pair_slot_count * sizeof(collision::CollisionPair),
                      phi::MemoryKind::Device);
    phi::Buffer active_flags(pair_slot_count * sizeof(uint8_t), phi::MemoryKind::Device);
    phi::Buffer pair_count(sizeof(uint32_t), phi::MemoryKind::Device);
    CheckCuda(cudaMemsetAsync(pair_count.Data(), 0, sizeof(uint32_t), stream),
              "cudaMemsetAsync pair count");

    if (shape_count == 0u) {
        context.stream.Synchronize();
        return CudaBroadphaseResult(0, 0,
                                    std::move(aabbs),
                                    std::move(pairs),
                                    std::move(active_flags),
                                    std::move(pair_count));
    }

    CheckCuda(cudaMemcpyAsync(aabbs.Data(), device_aabbs,
                              shape_count * sizeof(collision::AABB),
                              cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpyAsync broadphase AABBs");

    constexpr uint32_t kBlockSize = 128u;
    const uint32_t shape_blocks = (shape_count + kBlockSize - 1u) / kBlockSize;
    if (pair_slot_count > 0u) {
        GeneratePairSlotsKernel<<<shape_blocks, kBlockSize, 0, stream>>>(
            shape_count,
            static_cast<const collision::AABB*>(aabbs.Data()),
            static_cast<collision::CollisionPair*>(pairs.Data()),
            static_cast<uint8_t*>(active_flags.Data()),
            static_cast<uint32_t*>(pair_count.Data()));
        CheckCuda(cudaGetLastError(), "GeneratePairSlotsKernel launch");
    }

    context.stream.Synchronize();
    return CudaBroadphaseResult(shape_count,
                                pair_slot_count,
                                std::move(aabbs),
                                std::move(pairs),
                                std::move(active_flags),
                                std::move(pair_count));
}

CudaBroadphaseResult BuildCudaBroadphase(const collision::AABB* device_aabbs,
                                         uint32_t shape_count) {
    auto context = phi::MakeDefaultDeviceContext();
    return BuildCudaBroadphase(context, device_aabbs, shape_count);
}

} // namespace nuka::collision::gpu
