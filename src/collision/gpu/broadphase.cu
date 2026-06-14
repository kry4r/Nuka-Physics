// ---------------------------------------------------------------------------
// nuka::collision::gpu::broadphase implementation
// ---------------------------------------------------------------------------

#include "collision/gpu/broadphase.cuh"

#include "phi/backend.hpp"            // InitBestDevice, DeviceBufferType
#include "phi/buffer_transfer_v2.hpp" // DownloadVectorV2

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

// File-local RAII over a phi v2 opaque Buffer*. Stream-0 device buffer type:
// NULL-stream transfers are byte+ordering identical to the legacy synchronous
// memcpy. Frees on every path (incl. CheckCuda throw paths). Base()/Handle()
// mirror the legacy Data(); Release() hands the long-lived result members their
// owned handle.
class OwnedBuffer {
public:
    OwnedBuffer() = default;
    explicit OwnedBuffer(size_t bytes) : bytes_(bytes) {
        buf_ = phi::BufferAlloc(phi::DeviceBufferType(phi::InitBestDevice()), bytes);
    }
    ~OwnedBuffer() { if (buf_ != nullptr) phi::BufferFree(buf_); }
    OwnedBuffer(OwnedBuffer&& o) noexcept : buf_(o.buf_), bytes_(o.bytes_) {
        o.buf_ = nullptr; o.bytes_ = 0u;
    }
    OwnedBuffer& operator=(OwnedBuffer&& o) noexcept {
        if (this != &o) {
            if (buf_ != nullptr) phi::BufferFree(buf_);
            buf_ = o.buf_; bytes_ = o.bytes_; o.buf_ = nullptr; o.bytes_ = 0u;
        }
        return *this;
    }
    OwnedBuffer(const OwnedBuffer&) = delete;
    OwnedBuffer& operator=(const OwnedBuffer&) = delete;
    void*  Base()   const { return buf_ != nullptr ? phi::BufferBase(buf_) : nullptr; }
    size_t Bytes()  const { return bytes_; }
    phi::Buffer* Handle() const { return buf_; }
    phi::Buffer* Release() { phi::Buffer* b = buf_; buf_ = nullptr; bytes_ = 0u; return b; }
private:
    phi::Buffer* buf_ = nullptr;
    size_t bytes_ = 0u;
};

} // namespace

CudaBroadphaseResult::CudaBroadphaseResult(uint32_t shape_count,
                                           uint32_t pair_slot_count,
                                           phi::Buffer* aabbs,
                                           phi::Buffer* pairs,
                                           phi::Buffer* pair_active_flags,
                                           phi::Buffer* pair_count)
    : shape_count_(shape_count)
    , pair_slot_count_(pair_slot_count)
    , aabbs_(aabbs)
    , pairs_(pairs)
    , pair_active_flags_(pair_active_flags)
    , pair_count_(pair_count) {}

CudaBroadphaseResult::~CudaBroadphaseResult() {
    if (aabbs_             != nullptr) { phi::BufferFree(aabbs_); }
    if (pairs_             != nullptr) { phi::BufferFree(pairs_); }
    if (pair_active_flags_ != nullptr) { phi::BufferFree(pair_active_flags_); }
    if (pair_count_        != nullptr) { phi::BufferFree(pair_count_); }
}

CudaBroadphaseResult::CudaBroadphaseResult(CudaBroadphaseResult&& other) noexcept
    : shape_count_(other.shape_count_)
    , pair_slot_count_(other.pair_slot_count_)
    , aabbs_(other.aabbs_)
    , pairs_(other.pairs_)
    , pair_active_flags_(other.pair_active_flags_)
    , pair_count_(other.pair_count_) {
    other.shape_count_ = 0u;
    other.pair_slot_count_ = 0u;
    other.aabbs_ = nullptr;
    other.pairs_ = nullptr;
    other.pair_active_flags_ = nullptr;
    other.pair_count_ = nullptr;
}

CudaBroadphaseResult& CudaBroadphaseResult::operator=(CudaBroadphaseResult&& other) noexcept {
    if (this != &other) {
        if (aabbs_             != nullptr) { phi::BufferFree(aabbs_); }
        if (pairs_             != nullptr) { phi::BufferFree(pairs_); }
        if (pair_active_flags_ != nullptr) { phi::BufferFree(pair_active_flags_); }
        if (pair_count_        != nullptr) { phi::BufferFree(pair_count_); }
        shape_count_ = other.shape_count_;
        pair_slot_count_ = other.pair_slot_count_;
        aabbs_ = other.aabbs_;
        pairs_ = other.pairs_;
        pair_active_flags_ = other.pair_active_flags_;
        pair_count_ = other.pair_count_;
        other.shape_count_ = 0u;
        other.pair_slot_count_ = 0u;
        other.aabbs_ = nullptr;
        other.pairs_ = nullptr;
        other.pair_active_flags_ = nullptr;
        other.pair_count_ = nullptr;
    }
    return *this;
}

uint32_t CudaBroadphaseResult::PairCount() const {
    if (pair_count_ == nullptr) {
        return 0u;
    }

    uint32_t pair_count = 0;
    phi::BufferDownload(pair_count_, &pair_count, 0, sizeof(pair_count));
    return pair_count;
}

const collision::AABB* CudaBroadphaseResult::DeviceAabbs() const {
    return static_cast<const collision::AABB*>(
        aabbs_ != nullptr ? phi::BufferBase(aabbs_) : nullptr);
}

const collision::CollisionPair* CudaBroadphaseResult::DevicePairs() const {
    return static_cast<const collision::CollisionPair*>(
        pairs_ != nullptr ? phi::BufferBase(pairs_) : nullptr);
}

const uint8_t* CudaBroadphaseResult::DevicePairActiveFlags() const {
    return static_cast<const uint8_t*>(
        pair_active_flags_ != nullptr ? phi::BufferBase(pair_active_flags_) : nullptr);
}

const uint32_t* CudaBroadphaseResult::DevicePairCount() const {
    return static_cast<const uint32_t*>(
        pair_count_ != nullptr ? phi::BufferBase(pair_count_) : nullptr);
}

std::vector<collision::AABB> CudaBroadphaseResult::DownloadAabbs() const {
    return phi::DownloadVectorV2<collision::AABB>(aabbs_, shape_count_);
}

std::vector<collision::CollisionPair> CudaBroadphaseResult::DownloadPairs() const {
    const auto pairs = phi::DownloadVectorV2<collision::CollisionPair>(pairs_, pair_slot_count_);
    const auto flags = phi::DownloadVectorV2<uint8_t>(pair_active_flags_, pair_slot_count_);

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
    OwnedBuffer aabbs(shape_count * sizeof(collision::AABB));
    OwnedBuffer pairs(pair_slot_count * sizeof(collision::CollisionPair));
    OwnedBuffer active_flags(pair_slot_count * sizeof(uint8_t));
    OwnedBuffer pair_count(sizeof(uint32_t));
    CheckCuda(cudaMemsetAsync(pair_count.Base(), 0, sizeof(uint32_t), stream),
              "cudaMemsetAsync pair count");

    if (shape_count == 0u) {
        context.stream.Synchronize();
        return CudaBroadphaseResult(0, 0,
                                    aabbs.Release(),
                                    pairs.Release(),
                                    active_flags.Release(),
                                    pair_count.Release());
    }

    CheckCuda(cudaMemcpyAsync(aabbs.Base(), device_aabbs,
                              shape_count * sizeof(collision::AABB),
                              cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpyAsync broadphase AABBs");

    constexpr uint32_t kBlockSize = 128u;
    const uint32_t shape_blocks = (shape_count + kBlockSize - 1u) / kBlockSize;
    if (pair_slot_count > 0u) {
        GeneratePairSlotsKernel<<<shape_blocks, kBlockSize, 0, stream>>>(
            shape_count,
            static_cast<const collision::AABB*>(aabbs.Base()),
            static_cast<collision::CollisionPair*>(pairs.Base()),
            static_cast<uint8_t*>(active_flags.Base()),
            static_cast<uint32_t*>(pair_count.Base()));
        CheckCuda(cudaGetLastError(), "GeneratePairSlotsKernel launch");
    }

    context.stream.Synchronize();
    return CudaBroadphaseResult(shape_count,
                                pair_slot_count,
                                aabbs.Release(),
                                pairs.Release(),
                                active_flags.Release(),
                                pair_count.Release());
}

CudaBroadphaseResult BuildCudaBroadphase(const collision::AABB* device_aabbs,
                                         uint32_t shape_count) {
    auto context = phi::MakeDefaultDeviceContext();
    return BuildCudaBroadphase(context, device_aabbs, shape_count);
}

} // namespace nuka::collision::gpu
