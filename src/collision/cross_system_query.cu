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
#include "phi/buffer_transfer.hpp"

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

using ::nuka::phi::DownloadVector;

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
                                               phi::Buffer candidate_offsets,
                                               phi::Buffer candidate_counts,
                                               phi::Buffer candidate_indices)
    : particle_count_(particle_count)
    , candidate_capacity_(candidate_capacity)
    , total_candidates_(total_candidates)
    , truncated_particle_count_(truncated_particle_count)
    , candidate_offsets_(std::move(candidate_offsets))
    , candidate_counts_(std::move(candidate_counts))
    , candidate_indices_(std::move(candidate_indices)) {}

const uint32_t* CrossSystemQueryResult::DeviceCandidateOffsets() const {
    return static_cast<const uint32_t*>(candidate_offsets_.Data());
}
const uint32_t* CrossSystemQueryResult::DeviceCandidateCounts() const {
    return static_cast<const uint32_t*>(candidate_counts_.Data());
}
const uint32_t* CrossSystemQueryResult::DeviceCandidateIndices() const {
    return static_cast<const uint32_t*>(candidate_indices_.Data());
}
std::vector<uint32_t> CrossSystemQueryResult::DownloadCandidateOffsets() const {
    return DownloadVector<uint32_t>(candidate_offsets_, particle_count_);
}
std::vector<uint32_t> CrossSystemQueryResult::DownloadCandidateCounts() const {
    return DownloadVector<uint32_t>(candidate_counts_, particle_count_);
}
std::vector<uint32_t> CrossSystemQueryResult::DownloadCandidateIndices() const {
    return DownloadVector<uint32_t>(candidate_indices_, total_candidates_);
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
        phi::Buffer e0(0u, phi::MemoryKind::Device);
        phi::Buffer e1(0u, phi::MemoryKind::Device);
        phi::Buffer e2(0u, phi::MemoryKind::Device);
        return CrossSystemQueryResult(particle_count, kCrossSystemMaxCandidates,
                                      0u, 0u, std::move(e0), std::move(e1),
                                      std::move(e2));
    };

    if (particle_count == 0u || leaf_count == 0u || nodes == nullptr) {
        // No particles or an empty/non-retained rigid tree -> no candidates.
        if (particle_count == 0u) {
            return make_empty();
        }
        // Still emit zero-count CSR for every particle so offsets are well-formed.
        phi::Buffer d_offsets(particle_count * sizeof(uint32_t), phi::MemoryKind::Device);
        phi::Buffer d_counts(particle_count * sizeof(uint32_t), phi::MemoryKind::Device);
        CheckCuda(cudaMemsetAsync(d_offsets.Data(), 0, particle_count * sizeof(uint32_t), stream),
                  "memset offsets (empty tree)");
        CheckCuda(cudaMemsetAsync(d_counts.Data(), 0, particle_count * sizeof(uint32_t), stream),
                  "memset counts (empty tree)");
        context.stream.Synchronize();
        phi::Buffer d_cand(sizeof(uint32_t), phi::MemoryKind::Device);
        return CrossSystemQueryResult(particle_count, kCrossSystemMaxCandidates,
                                      0u, 0u, std::move(d_offsets),
                                      std::move(d_counts), std::move(d_cand));
    }

    const uint32_t blocks = (particle_count + kBlockSize - 1u) / kBlockSize;

    // --- Pass 1: count ------------------------------------------------------
    phi::Buffer d_counts(particle_count * sizeof(uint32_t), phi::MemoryKind::Device);
    CountCandidatesKernel<<<blocks, kBlockSize, 0, stream>>>(
        particle_count, particle_positions, query_radius, nodes, leaf_count,
        kCrossSystemMaxCandidates,
        static_cast<uint32_t*>(d_counts.Data()));
    CheckCuda(cudaGetLastError(), "CountCandidatesKernel launch");

    // --- Exclusive scan -> offsets ------------------------------------------
    phi::Buffer d_offsets(particle_count * sizeof(uint32_t), phi::MemoryKind::Device);
    thrust::exclusive_scan(
        thrust::cuda::par.on(stream),
        static_cast<const uint32_t*>(d_counts.Data()),
        static_cast<const uint32_t*>(d_counts.Data()) + particle_count,
        static_cast<uint32_t*>(d_offsets.Data()));
    CheckCuda(cudaGetLastError(), "exclusive_scan candidate counts");

    context.stream.Synchronize();
    uint32_t last_offset = 0u;
    uint32_t last_count = 0u;
    {
        const auto* off_dev = static_cast<const uint32_t*>(d_offsets.Data());
        const auto* cnt_dev = static_cast<const uint32_t*>(d_counts.Data());
        CheckCuda(cudaMemcpy(&last_offset, off_dev + (particle_count - 1u),
                             sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  "copy last offset");
        CheckCuda(cudaMemcpy(&last_count, cnt_dev + (particle_count - 1u),
                             sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  "copy last count");
    }
    const uint32_t total_candidates = last_offset + last_count;

    // --- Pass 2: fill -------------------------------------------------------
    phi::Buffer d_cand(
        (total_candidates == 0u ? 1u : total_candidates) * sizeof(uint32_t),
        phi::MemoryKind::Device);
    phi::Buffer d_truncated(sizeof(uint32_t), phi::MemoryKind::Device);
    CheckCuda(cudaMemsetAsync(d_truncated.Data(), 0, sizeof(uint32_t), stream),
              "memset truncated");
    FillCandidatesKernel<<<blocks, kBlockSize, 0, stream>>>(
        particle_count, particle_positions, query_radius, nodes, leaf_count,
        kCrossSystemMaxCandidates,
        static_cast<const uint32_t*>(d_offsets.Data()),
        static_cast<uint32_t*>(d_cand.Data()),
        static_cast<uint32_t*>(d_truncated.Data()));
    CheckCuda(cudaGetLastError(), "FillCandidatesKernel launch");

    context.stream.Synchronize();
    uint32_t truncated = 0u;
    d_truncated.CopyToHost(&truncated, sizeof(truncated));

    return CrossSystemQueryResult(particle_count, kCrossSystemMaxCandidates,
                                  total_candidates, truncated,
                                  std::move(d_offsets), std::move(d_counts),
                                  std::move(d_cand));
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
