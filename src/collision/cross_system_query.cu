// ---------------------------------------------------------------------------
// nuka::collision::gpu::QueryParticlesAgainstRigidLbvh implementation
// (v0.7 p05, Task 7.5.3).
//
// Two passes over the particles, each traversing the rigid LBVH with an explicit
// stack (mirrors LbvhPairQueryKernel from p04):
//   pass 1: count in-AABB rigid candidates per particle (capped) -> counts
//   exclusive scan: counts -> CSR offsets
//   pass 2: write each particle's candidates into its PRIVATE CSR slice,
//           sorted ascending by rigid body index (insertion sort, cap-bounded)
//
// D1: the only sort is the per-thread insertion sort into a private slice (no
// cross-thread ordering); the offset scan is thrust::exclusive_scan
// (deterministic); the only atomic is a uint32 truncation counter. The AABB
// overlap test compares pos +/- radius against node AABBs with plain </> (same
// IEEE op as a CPU oracle on the same inputs -- no MAD), so the candidate SET is
// exact vs brute force when under the cap. NO float atomics.
// ---------------------------------------------------------------------------

#include "collision/cross_system_query.hpp"

#include "collision/aabb.hpp"
#include "collision/lbvh_node.cuh"
#include "phi/backend.hpp"            // InitBestDevice, DeviceBufferType
#include "phi/buffer_transfer_v2.hpp" // DownloadVectorV2

#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nuka::collision::gpu {

namespace {

constexpr uint32_t kBlockSize = 128u;

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

// File-local RAII over a phi v2 opaque Buffer*. Stream-0 device buffer type:
// NULL-stream transfers are byte+ordering identical to the legacy synchronous
// memcpy. Frees on every path (incl. the CheckCuda throw paths). Base()/Handle()
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

// AABB overlap (device). Same predicate as collision::AABB::Overlaps and the
// LBVH traversal's LbvhOverlaps; inlined here so this TU does not include
// lbvh_traversal.cuh (which defines a non-static __global__ that would collide
// at link time with broadphase_lbvh.cu's copy).
__device__ __forceinline__ bool AabbOverlaps(const collision::AABB& a,
                                              const collision::AABB& b) {
    if (a.max.x < b.min.x || a.min.x > b.max.x) return false;
    if (a.max.y < b.min.y || a.min.y > b.max.y) return false;
    if (a.max.z < b.min.z || a.min.z > b.max.z) return false;
    return true;
}

// Build a particle's query AABB (sphere of `radius`) from its position.
__device__ __forceinline__ collision::AABB ParticleQueryAabb(math::Vec3 p,
                                                             float radius) {
    collision::AABB box;
    box.min = math::Vec3{p.x - radius, p.y - radius, p.z - radius};
    box.max = math::Vec3{p.x + radius, p.y + radius, p.z + radius};
    return box;
}

// Traverse the rigid LBVH for one particle. Collects overlapping rigid leaf body
// indices into `out` (a private slice), keeping them sorted ascending and capped
// at `max_count` (lowest body ids). Returns the count written; sets *overflow if
// more than max_count candidates overlapped. Single-leaf tree (leaf_count==1) is
// handled by the caller via root==the leaf node.
__device__ uint32_t TraverseRigidLbvh(const collision::AABB& query,
                                       const LbvhNode* __restrict__ nodes,
                                       uint32_t leaf_count,
                                       uint32_t* __restrict__ out,
                                       uint32_t max_count,
                                       bool* __restrict__ overflow) {
    uint32_t count = 0u;
    bool of = false;

    auto insert = [&](uint32_t body) {
        if (count < max_count) {
            uint32_t pos = count;
            while (pos > 0u && out[pos - 1u] > body) {
                out[pos] = out[pos - 1u];
                --pos;
            }
            out[pos] = body;
            ++count;
        } else {
            of = true;
            if (body < out[count - 1u]) {
                uint32_t pos = count - 1u;
                while (pos > 0u && out[pos - 1u] > body) {
                    out[pos] = out[pos - 1u];
                    --pos;
                }
                out[pos] = body;
            }
        }
    };

    if (leaf_count == 1u) {
        // Single node is the leaf itself (node 0). Test directly.
        const LbvhNode leaf = nodes[0];
        if (AabbOverlaps(query, leaf.aabb)) {
            insert(static_cast<uint32_t>(leaf.left));
        }
        if (overflow != nullptr) {
            *overflow = of;
        }
        return count;
    }

    const uint32_t internal_count = leaf_count - 1u;
    int32_t stack[64];
    int32_t top = 0;
    stack[top++] = 0; // root internal node

    while (top > 0) {
        const int32_t node_idx = stack[--top];
        const LbvhNode node = nodes[node_idx];
        const int32_t children[2] = {node.left, node.right};
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            const int32_t child = children[c];
            const bool is_leaf = (static_cast<uint32_t>(child) >= internal_count);
            const collision::AABB child_box = nodes[child].aabb;
            if (!AabbOverlaps(query, child_box)) {
                continue;
            }
            if (is_leaf) {
                insert(static_cast<uint32_t>(nodes[child].left));
            } else if (top < 63) {
                stack[top++] = child;
            }
        }
    }

    if (overflow != nullptr) {
        *overflow = of;
    }
    return count;
}

// --- Pass 1: count candidates -----------------------------------------------
__global__ void CountCandidatesKernel(uint32_t particle_count,
                                       const math::Vec3* __restrict__ positions,
                                       float radius,
                                       const LbvhNode* __restrict__ nodes,
                                       uint32_t leaf_count,
                                       uint32_t max_candidates,
                                       uint32_t* __restrict__ out_counts) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const collision::AABB query = ParticleQueryAabb(positions[i], radius);
    uint32_t scratch[kCrossSystemMaxCandidates];
    const uint32_t n = TraverseRigidLbvh(query, nodes, leaf_count, scratch,
                                         max_candidates, /*overflow=*/nullptr);
    out_counts[i] = n;
}

// --- Pass 2: fill candidates ------------------------------------------------
__global__ void FillCandidatesKernel(uint32_t particle_count,
                                      const math::Vec3* __restrict__ positions,
                                      float radius,
                                      const LbvhNode* __restrict__ nodes,
                                      uint32_t leaf_count,
                                      uint32_t max_candidates,
                                      const uint32_t* __restrict__ offsets,
                                      uint32_t* __restrict__ out_candidates,
                                      uint32_t* __restrict__ out_truncated) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= particle_count) {
        return;
    }
    const collision::AABB query = ParticleQueryAabb(positions[i], radius);
    bool overflow = false;
    uint32_t* slice = out_candidates + offsets[i];
    (void)TraverseRigidLbvh(query, nodes, leaf_count, slice, max_candidates,
                            &overflow);
    if (overflow) {
        atomicAdd(out_truncated, 1u); // uint32 atomic -> D1
    }
}

} // namespace

// --- Result accessors -------------------------------------------------------
CrossSystemQueryResult::CrossSystemQueryResult(uint32_t particle_count,
                                               uint32_t candidate_capacity,
                                               uint32_t total_candidates,
                                               uint32_t truncated_particle_count,
                                               phi::Buffer* candidate_offsets,
                                               phi::Buffer* candidate_counts,
                                               phi::Buffer* candidate_indices)
    : particle_count_(particle_count)
    , candidate_capacity_(candidate_capacity)
    , total_candidates_(total_candidates)
    , truncated_particle_count_(truncated_particle_count)
    , candidate_offsets_(candidate_offsets)
    , candidate_counts_(candidate_counts)
    , candidate_indices_(candidate_indices) {}

CrossSystemQueryResult::~CrossSystemQueryResult() {
    if (candidate_offsets_ != nullptr) { phi::BufferFree(candidate_offsets_); }
    if (candidate_counts_  != nullptr) { phi::BufferFree(candidate_counts_); }
    if (candidate_indices_ != nullptr) { phi::BufferFree(candidate_indices_); }
}

CrossSystemQueryResult::CrossSystemQueryResult(CrossSystemQueryResult&& other) noexcept
    : particle_count_(other.particle_count_)
    , candidate_capacity_(other.candidate_capacity_)
    , total_candidates_(other.total_candidates_)
    , truncated_particle_count_(other.truncated_particle_count_)
    , candidate_offsets_(other.candidate_offsets_)
    , candidate_counts_(other.candidate_counts_)
    , candidate_indices_(other.candidate_indices_) {
    other.particle_count_ = 0u;
    other.candidate_capacity_ = 0u;
    other.total_candidates_ = 0u;
    other.truncated_particle_count_ = 0u;
    other.candidate_offsets_ = nullptr;
    other.candidate_counts_ = nullptr;
    other.candidate_indices_ = nullptr;
}

CrossSystemQueryResult& CrossSystemQueryResult::operator=(CrossSystemQueryResult&& other) noexcept {
    if (this != &other) {
        if (candidate_offsets_ != nullptr) { phi::BufferFree(candidate_offsets_); }
        if (candidate_counts_  != nullptr) { phi::BufferFree(candidate_counts_); }
        if (candidate_indices_ != nullptr) { phi::BufferFree(candidate_indices_); }
        particle_count_ = other.particle_count_;
        candidate_capacity_ = other.candidate_capacity_;
        total_candidates_ = other.total_candidates_;
        truncated_particle_count_ = other.truncated_particle_count_;
        candidate_offsets_ = other.candidate_offsets_;
        candidate_counts_ = other.candidate_counts_;
        candidate_indices_ = other.candidate_indices_;
        other.particle_count_ = 0u;
        other.candidate_capacity_ = 0u;
        other.total_candidates_ = 0u;
        other.truncated_particle_count_ = 0u;
        other.candidate_offsets_ = nullptr;
        other.candidate_counts_ = nullptr;
        other.candidate_indices_ = nullptr;
    }
    return *this;
}

const uint32_t* CrossSystemQueryResult::DeviceCandidateOffsets() const {
    return static_cast<const uint32_t*>(
        candidate_offsets_ != nullptr ? phi::BufferBase(candidate_offsets_) : nullptr);
}
const uint32_t* CrossSystemQueryResult::DeviceCandidateCounts() const {
    return static_cast<const uint32_t*>(
        candidate_counts_ != nullptr ? phi::BufferBase(candidate_counts_) : nullptr);
}
const uint32_t* CrossSystemQueryResult::DeviceCandidateIndices() const {
    return static_cast<const uint32_t*>(
        candidate_indices_ != nullptr ? phi::BufferBase(candidate_indices_) : nullptr);
}
std::vector<uint32_t> CrossSystemQueryResult::DownloadCandidateOffsets() const {
    return phi::DownloadVectorV2<uint32_t>(candidate_offsets_, particle_count_);
}
std::vector<uint32_t> CrossSystemQueryResult::DownloadCandidateCounts() const {
    return phi::DownloadVectorV2<uint32_t>(candidate_counts_, particle_count_);
}
std::vector<uint32_t> CrossSystemQueryResult::DownloadCandidateIndices() const {
    return phi::DownloadVectorV2<uint32_t>(candidate_indices_, total_candidates_);
}

CrossSystemQueryResult QueryParticlesAgainstRigidLbvh(
    const phi::DeviceContext& context,
    const math::Vec3* particle_positions,
    uint32_t particle_count,
    float query_radius,
    const LbvhBroadphaseResult& rigid_tree) {
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();

    const uint32_t leaf_count = rigid_tree.LeafCount();
    const LbvhNode* nodes = rigid_tree.DeviceNodes();

    auto make_empty = [&]() {
        OwnedBuffer e0(0u);
        OwnedBuffer e1(0u);
        OwnedBuffer e2(0u);
        return CrossSystemQueryResult(particle_count, kCrossSystemMaxCandidates,
                                      0u, 0u, e0.Release(), e1.Release(),
                                      e2.Release());
    };

    if (particle_count == 0u || leaf_count == 0u || nodes == nullptr) {
        // No particles or an empty/non-retained rigid tree -> no candidates.
        if (particle_count == 0u) {
            return make_empty();
        }
        // Still emit zero-count CSR for every particle so offsets are well-formed.
        OwnedBuffer d_offsets(particle_count * sizeof(uint32_t));
        OwnedBuffer d_counts(particle_count * sizeof(uint32_t));
        CheckCuda(cudaMemsetAsync(d_offsets.Base(), 0, particle_count * sizeof(uint32_t), stream),
                  "memset offsets (empty tree)");
        CheckCuda(cudaMemsetAsync(d_counts.Base(), 0, particle_count * sizeof(uint32_t), stream),
                  "memset counts (empty tree)");
        context.stream.Synchronize();
        OwnedBuffer d_cand(sizeof(uint32_t));
        return CrossSystemQueryResult(particle_count, kCrossSystemMaxCandidates,
                                      0u, 0u, d_offsets.Release(),
                                      d_counts.Release(), d_cand.Release());
    }

    const uint32_t blocks = (particle_count + kBlockSize - 1u) / kBlockSize;

    // --- Pass 1: count ------------------------------------------------------
    OwnedBuffer d_counts(particle_count * sizeof(uint32_t));
    CountCandidatesKernel<<<blocks, kBlockSize, 0, stream>>>(
        particle_count, particle_positions, query_radius, nodes, leaf_count,
        kCrossSystemMaxCandidates,
        static_cast<uint32_t*>(d_counts.Base()));
    CheckCuda(cudaGetLastError(), "CountCandidatesKernel launch");

    // --- Exclusive scan -> offsets ------------------------------------------
    OwnedBuffer d_offsets(particle_count * sizeof(uint32_t));
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        static_cast<const uint32_t*>(d_counts.Base()),
        static_cast<const uint32_t*>(d_counts.Base()) + particle_count,
        static_cast<uint32_t*>(d_offsets.Base()));
    CheckCuda(cudaGetLastError(), "exclusive_scan candidate counts");

    context.stream.Synchronize();
    uint32_t last_offset = 0u;
    uint32_t last_count = 0u;
    {
        const auto* off_dev = static_cast<const uint32_t*>(d_offsets.Base());
        const auto* cnt_dev = static_cast<const uint32_t*>(d_counts.Base());
        CheckCuda(cudaMemcpy(&last_offset, off_dev + (particle_count - 1u),
                             sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  "copy last offset");
        CheckCuda(cudaMemcpy(&last_count, cnt_dev + (particle_count - 1u),
                             sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  "copy last count");
    }
    const uint32_t total_candidates = last_offset + last_count;

    // --- Pass 2: fill -------------------------------------------------------
    OwnedBuffer d_cand(
        (total_candidates == 0u ? 1u : total_candidates) * sizeof(uint32_t));
    OwnedBuffer d_truncated(sizeof(uint32_t));
    CheckCuda(cudaMemsetAsync(d_truncated.Base(), 0, sizeof(uint32_t), stream),
              "memset truncated");
    FillCandidatesKernel<<<blocks, kBlockSize, 0, stream>>>(
        particle_count, particle_positions, query_radius, nodes, leaf_count,
        kCrossSystemMaxCandidates,
        static_cast<const uint32_t*>(d_offsets.Base()),
        static_cast<uint32_t*>(d_cand.Base()),
        static_cast<uint32_t*>(d_truncated.Base()));
    CheckCuda(cudaGetLastError(), "FillCandidatesKernel launch");

    context.stream.Synchronize();
    uint32_t truncated = 0u;
    phi::BufferDownload(d_truncated.Handle(), &truncated, 0, sizeof(truncated));

    return CrossSystemQueryResult(particle_count, kCrossSystemMaxCandidates,
                                  total_candidates, truncated,
                                  d_offsets.Release(), d_counts.Release(),
                                  d_cand.Release());
}

CrossSystemQueryResult QueryParticlesAgainstRigidLbvh(
    const math::Vec3* particle_positions,
    uint32_t particle_count,
    float query_radius,
    const LbvhBroadphaseResult& rigid_tree) {
    auto context = phi::MakeDefaultDeviceContext();
    return QueryParticlesAgainstRigidLbvh(context, particle_positions,
                                          particle_count, query_radius,
                                          rigid_tree);
}

} // namespace nuka::collision::gpu
