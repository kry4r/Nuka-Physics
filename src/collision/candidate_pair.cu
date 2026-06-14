// ---------------------------------------------------------------------------
// nuka::collision::BuildCandidatePairStream implementation (v0.8 C2a)
// ---------------------------------------------------------------------------
// Takes an UNSORTED CandidatePair set, stamps each pair's canonical stable_key
// (MakeStableKey, pure-integer -> deterministic), uploads, and sorts ASCENDING
// by stable_key with thrust::stable_sort_by_key. Radix on the 64-bit integral
// key is stable + deterministic, so the output stream is D1 byte-exact run to
// run -- the SAME sort discipline broadphase_lbvh.cu uses to remove its
// nondeterministic emit order. NO append atomics: C2a sorts a fixed input;
// C2b/C2c produce that input via private-slice count->scan->fill (no global
// append atomic, cross_system_query.hpp posture) and call THIS builder.
// ---------------------------------------------------------------------------

#include "collision/candidate_pair.hpp"

#include "phi/backend.hpp"            // InitBestDevice, DeviceBufferType
#include "phi/buffer_transfer_v2.hpp" // UploadVectorV2

#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/sort.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nuka::collision {

namespace {

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

} // namespace

CandidatePairStream::CandidatePairStream(uint32_t count, phi::Buffer* pairs)
    : count_(count), pairs_(pairs) {}

CandidatePairStream::~CandidatePairStream() {
    if (pairs_ != nullptr) {
        phi::BufferFree(pairs_);
    }
}

CandidatePairStream::CandidatePairStream(CandidatePairStream&& other) noexcept
    : count_(other.count_), pairs_(other.pairs_) {
    other.count_ = 0u;
    other.pairs_ = nullptr;
}

CandidatePairStream& CandidatePairStream::operator=(CandidatePairStream&& other) noexcept {
    if (this != &other) {
        if (pairs_ != nullptr) {
            phi::BufferFree(pairs_);
        }
        count_ = other.count_;
        pairs_ = other.pairs_;
        other.count_ = 0u;
        other.pairs_ = nullptr;
    }
    return *this;
}

const CandidatePair* CandidatePairStream::DevicePairs() const {
    return static_cast<const CandidatePair*>(
        pairs_ != nullptr ? phi::BufferBase(pairs_) : nullptr);
}

std::vector<CandidatePair> CandidatePairStream::DownloadPairs() const {
    std::vector<CandidatePair> out(count_);
    if (count_ > 0u && pairs_ != nullptr) {
        phi::BufferDownload(pairs_, out.data(), 0,
                            static_cast<size_t>(count_) * sizeof(CandidatePair));
    }
    return out;
}

CandidatePairStream BuildCandidatePairStream(const phi::DeviceContext& context,
                                             const std::vector<CandidatePair>& unsorted_pairs) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();

    // Stream-0 device buffer type: NULL-stream transfers are byte+ordering
    // identical to the legacy synchronous memcpy (the upload completes before the
    // work-stream thrust sort reads it, exactly as the legacy path relied on).
    phi::BufferType* bt = phi::DeviceBufferType(phi::InitBestDevice());

    const uint32_t count = static_cast<uint32_t>(unsorted_pairs.size());
    if (count == 0u) {
        return CandidatePairStream(0u, phi::BufferAlloc(bt, 0u));
    }

    // Stamp each pair's canonical stable_key on the host (MakeStableKey is HD; on
    // the host it is the same pure-integer packing). Keys are computed from
    // (a,b) regardless of any value the caller left in stable_key.
    std::vector<CandidatePair> staged = unsorted_pairs;
    std::vector<uint64_t> keys(count);
    for (uint32_t i = 0u; i < count; ++i) {
        staged[i].stable_key = MakeStableKey(staged[i].a, staged[i].b);
        keys[i] = staged[i].stable_key;
    }

    // Upload keys + pairs, then radix stable_sort_by_key ascending on the key.
    phi::Buffer* d_keys = phi::UploadVectorV2(bt, keys);
    phi::Buffer* d_pairs = phi::UploadVectorV2(bt, staged);

    thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream),
        static_cast<uint64_t*>(phi::BufferBase(d_keys)),
        static_cast<uint64_t*>(phi::BufferBase(d_keys)) + count,
        static_cast<CandidatePair*>(phi::BufferBase(d_pairs)));
    CheckCuda(cudaGetLastError(), "stable_sort_by_key candidate pairs");

    context.stream.Synchronize();
    phi::BufferFree(d_keys);
    return CandidatePairStream(count, d_pairs);
}

CandidatePairStream BuildCandidatePairStream(const std::vector<CandidatePair>& unsorted_pairs) {
    auto context = phi::MakeDefaultDeviceContext();
    return BuildCandidatePairStream(context, unsorted_pairs);
}

} // namespace nuka::collision
