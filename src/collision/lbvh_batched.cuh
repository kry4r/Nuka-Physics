#pragma once
// Collision and rendering share this batched LBVH build with stable CUB sorting.
// Explicit workspace keeps allocations outside capture; leaf indices are environment-local.

#include "collision/lbvh_node.cuh"      // LbvhNode / LbvhDelta / LbvhMerge
#include "collision/morton_codes.cuh"   // Morton3D30
#include "math/vec3.hpp"

#include <cfloat>
#include <cstdint>
#include <limits>

#include <cuda_runtime.h>

#include <cub/device/device_radix_sort.cuh>

namespace nuka::collision::gpu {

struct LbvhWorkspaceLayout {
    size_t index_offset;
    size_t temp_offset;
    explicit LbvhWorkspaceLayout(uint32_t count)
        : index_offset((size_t{count} * sizeof(uint64_t) + 255u) & ~size_t{255u}),
          temp_offset((index_offset + size_t{count} * sizeof(uint32_t) + 255u) & ~size_t{255u}) {}
};

inline cudaError_t QueryLbvhWorkspaceBytes(uint32_t envs, uint32_t leaves,
                                           size_t* bytes) {
    if (bytes == nullptr) return cudaErrorInvalidValue;
    *bytes = 0u;
    if (envs == 0u || leaves < 2u) return cudaSuccess;
    const uint64_t count = uint64_t{envs} * leaves;
    if (envs > 65535u || count > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        uint64_t{envs} * (uint64_t{leaves} * 2u - 1u) > std::numeric_limits<uint32_t>::max())
        return cudaErrorInvalidValue;
    size_t temp = 0u;
    const auto status = cub::DeviceRadixSort::SortPairs<uint64_t, uint32_t>(
        nullptr, temp, static_cast<const uint64_t*>(nullptr), static_cast<uint64_t*>(nullptr),
        static_cast<const uint32_t*>(nullptr), static_cast<uint32_t*>(nullptr),
        static_cast<int>(count));
    if (status != cudaSuccess) return status;
    *bytes = LbvhWorkspaceLayout(static_cast<uint32_t>(count)).temp_offset + temp;
    return cudaSuccess;
}

// Interleaved-array AABB source: one collision::AABB per leaf (render / test).
struct AabbArraySource {
    const collision::AABB* __restrict__ aabbs;
    __device__ __forceinline__ collision::AABB Load(uint32_t i) const {
        return aabbs[i];
    }
};

// Split lo/hi AABB source (the collision arena's body_aabb_lo / body_aabb_hi).
// Load reassembles the SAME interleaved AABB (min==lo[i], max==hi[i]).
struct AabbSplitSource {
    const math::Vec3* __restrict__ lo;
    const math::Vec3* __restrict__ hi;
    __device__ __forceinline__ collision::AABB Load(uint32_t i) const {
        collision::AABB box;
        box.min = lo[i];
        box.max = hi[i];
        return box;
    }
};

namespace lbvh_batched_detail {

constexpr uint32_t kBlockSize = 128u;

// One thread folds the env's own AABB-center bound (fixed order) -> env-LOCAL
// Morton + index + the (env<<32)|morton compound key (envs never mix).
template <typename Src>
__global__ void EnvMortonKernel(Src src,
                                uint32_t leaves_per_env,
                                uint32_t* __restrict__ out_morton,
                                uint32_t* __restrict__ out_index,
                                uint64_t* __restrict__ out_sortkey) {
    const uint32_t env = blockIdx.x;
    const uint32_t base = env * leaves_per_env;
    if (threadIdx.x != 0u) return;
    float mn[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
    float mx[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    for (uint32_t i = 0; i < leaves_per_env; ++i) {
        const collision::AABB box = src.Load(base + i);
        const float c[3] = {0.5f * (box.min.x + box.max.x),
                            0.5f * (box.min.y + box.max.y),
                            0.5f * (box.min.z + box.max.z)};
        for (int k = 0; k < 3; ++k) {
            mn[k] = fminf(mn[k], c[k]);
            mx[k] = fmaxf(mx[k], c[k]);
        }
    }
    const float inv[3] = {
        (mx[0] > mn[0]) ? 1.0f / (mx[0] - mn[0]) : 0.0f,
        (mx[1] > mn[1]) ? 1.0f / (mx[1] - mn[1]) : 0.0f,
        (mx[2] > mn[2]) ? 1.0f / (mx[2] - mn[2]) : 0.0f};
    for (uint32_t i = 0; i < leaves_per_env; ++i) {
        const collision::AABB box = src.Load(base + i);
        const float nx = (0.5f * (box.min.x + box.max.x) - mn[0]) * inv[0];
        const float ny = (0.5f * (box.min.y + box.max.y) - mn[1]) * inv[1];
        const float nz = (0.5f * (box.min.z + box.max.z) - mn[2]) * inv[2];
        const uint32_t code = Morton3D30(nx, ny, nz);
        out_morton[base + i] = code;
        out_index[base + i] = i;  // env-LOCAL leaf index (0..N-1)
        out_sortkey[base + i] =
            (static_cast<uint64_t>(env) << 32) | static_cast<uint64_t>(code);
    }
}

// Init one env's tree (node slice base = env*(2N-1)). Internal parents pre-seed
// -1; leaves store the env-LOCAL index in `.left` + the leaf AABB.
template <typename Src>
__global__ void EnvInitNodesKernel(Src src,
                                   const uint32_t* __restrict__ sorted_index,
                                   uint32_t leaves_per_env,
                                   LbvhNode* __restrict__ nodes) {
    const uint32_t env = blockIdx.y;
    const uint32_t N = leaves_per_env;
    const uint32_t node_count = 2u * N - 1u;
    const uint32_t internal = N - 1u;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= node_count) return;
    const uint32_t nbase = env * node_count;
    const uint32_t bbase = env * N;
    if (idx < internal) {
        nodes[nbase + idx].parent = -1;
        return;
    }
    const uint32_t lane = idx - internal;
    const uint32_t local_leaf = sorted_index ? sorted_index[bbase + lane] : lane;
    LbvhNode leaf;
    leaf.left = static_cast<int32_t>(local_leaf);
    leaf.right = -1;
    leaf.parent = -1;
    leaf.aabb = src.Load(bbase + local_leaf);
    nodes[nbase + idx] = leaf;
}

// Karras internal-node build for one env (node slice base = env*(2N-1)).
static __global__ void EnvBuildInternalKernel(const uint32_t* __restrict__ morton_sorted,
                                              uint32_t leaves_per_env,
                                              LbvhNode* __restrict__ nodes) {
    const uint32_t env = blockIdx.y;
    const uint32_t N = leaves_per_env;
    const uint32_t internal = N - 1u;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= internal) return;
    const uint32_t nbase = env * (2u * N - 1u);
    const uint32_t mbase = env * N;
    const uint32_t* morton = morton_sorted + mbase;
    const int i = static_cast<int>(idx);

    const int d_l = LbvhDelta(i, i - 1, N, morton);
    const int d_r = LbvhDelta(i, i + 1, N, morton);
    const int d = (d_r >= d_l) ? 1 : -1;
    const int delta_min = LbvhDelta(i, i - d, N, morton);
    int l_max = 2;
    while (LbvhDelta(i, i + l_max * d, N, morton) > delta_min) l_max *= 2;
    int l = 0;
    for (int t = l_max >> 1; t >= 1; t >>= 1) {
        if (LbvhDelta(i, i + (l + t) * d, N, morton) > delta_min) l += t;
    }
    const int j = i + l * d;
    const int delta_node = LbvhDelta(i, j, N, morton);
    int s = 0;
    int t_div = l;
    do {
        t_div = (t_div + 1) >> 1;
        if (LbvhDelta(i, i + (s + t_div) * d, N, morton) > delta_node) s += t_div;
    } while (t_div > 1);
    const int gamma = i + s * d + min(d, 0);
    const int first = min(i, j);
    const int last = max(i, j);
    const int32_t left_child = (first == gamma)
        ? static_cast<int32_t>(internal + gamma) : static_cast<int32_t>(gamma);
    const int32_t right_child = (last == gamma + 1)
        ? static_cast<int32_t>(internal + gamma + 1) : static_cast<int32_t>(gamma + 1);
    nodes[nbase + i].left = left_child;
    nodes[nbase + i].right = right_child;
    nodes[nbase + left_child].parent = i;
    nodes[nbase + right_child].parent = i;
}

// Bottom-up AABB propagate for one env (uint32 visit atomics, order-independent
// merge -> D1; __threadfence guards the child-bound write; visit base env*N).
static __global__ void EnvPropagateKernel(uint32_t leaves_per_env,
                                          LbvhNode* __restrict__ nodes,
                                          uint32_t* __restrict__ visit) {
    const uint32_t env = blockIdx.y;
    const uint32_t N = leaves_per_env;
    const uint32_t internal = N - 1u;
    const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= N) return;
    const uint32_t nbase = env * (2u * N - 1u);
    const uint32_t vbase = env * N;
    int32_t node = nodes[nbase + internal + lane].parent;
    while (node >= 0) {
        __threadfence();
        const uint32_t prev = atomicAdd(&visit[vbase + node], 1u);
        if (prev == 0u) return;
        const LbvhNode self = nodes[nbase + node];
        nodes[nbase + node].aabb =
            LbvhMerge(nodes[nbase + self.left].aabb,
                      nodes[nbase + self.right].aabb);
        node = nodes[nbase + node].parent;
    }
}

// Re-load each leaf's bound from the fresh per-env AABB source using the env-
// LOCAL leaf index stored in `.left` at build (topology untouched).
template <typename Src>
__global__ void EnvRefitLeavesKernel(Src src,
                                     uint32_t leaves_per_env,
                                     LbvhNode* __restrict__ nodes) {
    const uint32_t env = blockIdx.y;
    const uint32_t N = leaves_per_env;
    const uint32_t internal = N - 1u;
    const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= N) return;
    const uint32_t nbase = env * (2u * N - 1u);
    const uint32_t bbase = env * N;
    const uint32_t node_idx = nbase + internal + lane;
    const int32_t local_leaf = nodes[node_idx].left;  // env-local id at build
    nodes[node_idx].aabb = src.Load(bbase + local_leaf);
}

static __global__ void ZeroU32Kernel(uint32_t* __restrict__ a, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = 0u;
}

static __global__ void SortedMortonKernel(const uint64_t* __restrict__ keys,
                                          uint32_t* __restrict__ morton, uint32_t count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) morton[i] = static_cast<uint32_t>(keys[i]);
}

// Shared launch body over any AABB source. Both public launchers forward here.
template <typename Src>
inline cudaError_t BuildLbvhBatchedNodesImpl(cudaStream_t stream, Src src,
                                      uint32_t env_count, uint32_t leaves_per_env,
                                      LbvhNode* out_nodes,
                                      uint32_t* morton, uint32_t* index,
                                      uint64_t* sortkey, uint32_t* visit,
                                      void* workspace, size_t workspace_bytes) {
    const uint32_t E = env_count;
    const uint32_t N = leaves_per_env;
    if (E == 0u || N == 0u) return cudaSuccess;
    const uint64_t count = uint64_t{E} * N;
    if (E > 65535u || count > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
        uint64_t{E} * (uint64_t{N} * 2u - 1u) > std::numeric_limits<uint32_t>::max())
        return cudaErrorInvalidValue;
    if (N == 1u) {
        EnvInitNodesKernel<Src><<<dim3(1u, E), dim3(kBlockSize), 0, stream>>>(
            src, nullptr, N, out_nodes);
        return cudaGetLastError();
    }
    const LbvhWorkspaceLayout layout(static_cast<uint32_t>(count));
    if (workspace == nullptr || workspace_bytes <= layout.temp_offset)
        return cudaErrorInvalidValue;
    auto* bytes = static_cast<uint8_t*>(workspace);
    auto* keys_in = reinterpret_cast<uint64_t*>(bytes);
    auto* indices_in = reinterpret_cast<uint32_t*>(bytes + layout.index_offset);
    size_t temp_bytes = workspace_bytes - layout.temp_offset;
    {  // zero the per-leaf visit counters.
        const uint32_t n = E * N;
        const uint32_t b = (n + kBlockSize - 1u) / kBlockSize;
        ZeroU32Kernel<<<b, kBlockSize, 0, stream>>>(visit, n);
    }
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
    EnvMortonKernel<Src><<<dim3(E), dim3(kBlockSize), 0, stream>>>(
        src, N, morton, indices_in, keys_in);
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
    const auto sorted = cub::DeviceRadixSort::SortPairs(
        bytes + layout.temp_offset, temp_bytes, keys_in, sortkey,
        indices_in, index, static_cast<int>(count), 0, 64, stream);
    if (sorted != cudaSuccess) return sorted;
    SortedMortonKernel<<<(static_cast<uint32_t>(count) + kBlockSize - 1u) / kBlockSize,
                          kBlockSize, 0, stream>>>(sortkey, morton, static_cast<uint32_t>(count));
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
    const uint32_t node_count = 2u * N - 1u;
    const uint32_t node_blocks = (node_count + kBlockSize - 1u) / kBlockSize;
    EnvInitNodesKernel<Src><<<dim3(node_blocks, E), dim3(kBlockSize), 0, stream>>>(
        src, index, N, out_nodes);
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
    const uint32_t internal_blocks = ((N - 1u) + kBlockSize - 1u) / kBlockSize;
    EnvBuildInternalKernel<<<dim3(internal_blocks, E), dim3(kBlockSize), 0, stream>>>(
        morton, N, out_nodes);
    if (const auto status = cudaGetLastError(); status != cudaSuccess) return status;
    const uint32_t leaf_blocks = (N + kBlockSize - 1u) / kBlockSize;
    EnvPropagateKernel<<<dim3(leaf_blocks, E), dim3(kBlockSize), 0, stream>>>(
        N, out_nodes, visit);
    return cudaGetLastError();
}

template <typename Src>
inline void RefitLbvhBatchedImpl(cudaStream_t stream, LbvhNode* nodes, Src src,
                                 uint32_t env_count, uint32_t leaves_per_env,
                                 uint32_t* visit) {
    const uint32_t E = env_count;
    const uint32_t N = leaves_per_env;
    if (E == 0u || N == 0u) return;
    if (N == 1u) {
        EnvRefitLeavesKernel<Src><<<dim3(1u, E), dim3(kBlockSize), 0, stream>>>(src, N, nodes);
        return;
    }
    {
        const uint32_t n = E * N;
        const uint32_t b = (n + kBlockSize - 1u) / kBlockSize;
        ZeroU32Kernel<<<b, kBlockSize, 0, stream>>>(visit, n);
    }
    const uint32_t leaf_blocks = (N + kBlockSize - 1u) / kBlockSize;
    EnvRefitLeavesKernel<Src><<<dim3(leaf_blocks, E), dim3(kBlockSize), 0, stream>>>(
        src, N, nodes);
    EnvPropagateKernel<<<dim3(leaf_blocks, E), dim3(kBlockSize), 0, stream>>>(
        N, nodes, visit);
}

}  // namespace lbvh_batched_detail

// Build E per-env trees over env-major `device_aabbs` into `device_out_nodes`
// (env-offset env*(2N-1)); scratch each E*N; leaf `.left` = env-LOCAL index.
inline cudaError_t BuildLbvhBatchedNodes(cudaStream_t stream, int /*device_id*/,
                                  const collision::AABB* device_aabbs,
                                  uint32_t env_count,
                                  uint32_t leaves_per_env,
                                  LbvhNode* device_out_nodes,
                                  uint32_t* device_morton_scratch,
                                  uint32_t* device_index_scratch,
                                  uint64_t* device_sortkey_scratch,
                                  uint32_t* device_visit_scratch,
                                  void* workspace, size_t workspace_bytes) {
    return lbvh_batched_detail::BuildLbvhBatchedNodesImpl(
        stream, AabbArraySource{device_aabbs}, env_count, leaves_per_env,
        device_out_nodes, device_morton_scratch, device_index_scratch,
        device_sortkey_scratch, device_visit_scratch, workspace, workspace_bytes);
}

// Split lo/hi overload for the collision arena (body_aabb_lo / body_aabb_hi).
// Same algorithm; the source reassembles the interleaved AABB per leaf.
inline cudaError_t BuildLbvhBatchedNodes(cudaStream_t stream, int /*device_id*/,
                                  const math::Vec3* device_aabb_lo,
                                  const math::Vec3* device_aabb_hi,
                                  uint32_t env_count,
                                  uint32_t leaves_per_env,
                                  LbvhNode* device_out_nodes,
                                  uint32_t* device_morton_scratch,
                                  uint32_t* device_index_scratch,
                                  uint64_t* device_sortkey_scratch,
                                  uint32_t* device_visit_scratch,
                                  void* workspace, size_t workspace_bytes) {
    return lbvh_batched_detail::BuildLbvhBatchedNodesImpl(
        stream, AabbSplitSource{device_aabb_lo, device_aabb_hi}, env_count,
        leaves_per_env, device_out_nodes, device_morton_scratch,
        device_index_scratch, device_sortkey_scratch, device_visit_scratch,
        workspace, workspace_bytes);
}

// Refit E per-env trees in place: env-offset leaf reload from `device_new_aabbs`
// (env-major) + env bottom-up propagate; topology reused, visit scratch E*N.
inline void RefitLbvhBatched(cudaStream_t stream, int /*device_id*/,
                             LbvhNode* device_nodes,
                             const collision::AABB* device_new_aabbs,
                             uint32_t env_count,
                             uint32_t leaves_per_env,
                             uint32_t* device_visit_scratch) {
    lbvh_batched_detail::RefitLbvhBatchedImpl(
        stream, device_nodes, AabbArraySource{device_new_aabbs}, env_count,
        leaves_per_env, device_visit_scratch);
}

// Split lo/hi refit overload for the collision arena.
inline void RefitLbvhBatched(cudaStream_t stream, int /*device_id*/,
                             LbvhNode* device_nodes,
                             const math::Vec3* device_new_aabb_lo,
                             const math::Vec3* device_new_aabb_hi,
                             uint32_t env_count,
                             uint32_t leaves_per_env,
                             uint32_t* device_visit_scratch) {
    lbvh_batched_detail::RefitLbvhBatchedImpl(
        stream, device_nodes,
        AabbSplitSource{device_new_aabb_lo, device_new_aabb_hi}, env_count,
        leaves_per_env, device_visit_scratch);
}

}  // namespace nuka::collision::gpu
