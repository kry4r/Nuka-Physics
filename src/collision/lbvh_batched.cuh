#pragma once
// ---------------------------------------------------------------------------
// nuka::collision::gpu -- shared BATCHED, single-launch, per-env LBVH build.
//
// The ONE copy of the env-indexed Karras build: N contiguous per-env trees in
// one launch over dim3(node_blocks, env_count), env-offset node slices
// env*(2N-1), plus ONE cross-env thrust::stable_sort_by_key on the u64 key
// (env<<32)|morton30 (bit-identical to E independent stable sorts). Both the
// pair-driven collision broadphase op and the render TLAS call these launchers;
// the env-kernels are templated on an AABB SOURCE so the SAME kernel body reads
// either an interleaved collision::AABB array (render / standalone) or the
// collision arena's split lo/hi vec3 arrays -- one algorithm, two thin adapters.
//
// Leaf `.left` is the env-LOCAL leaf index; the caller maps it to a body /
// instance id. The arithmetic is byte-identical to the single-tree build/refit
// so per-env results equal independent builds. Included only from .cu TUs.
// ---------------------------------------------------------------------------

#include "collision/lbvh_node.cuh"      // LbvhNode / LbvhDelta / LbvhMerge
#include "collision/morton_codes.cuh"   // Morton3D30
#include "math/vec3.hpp"

#include <cfloat>
#include <cstdint>

#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>

namespace nuka::collision::gpu {

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
    const uint32_t local_leaf = sorted_index[bbase + lane];  // env-local id
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

// Shared launch body over any AABB source. Both public launchers forward here.
template <typename Src>
inline void BuildLbvhBatchedNodesImpl(cudaStream_t stream, Src src,
                                      uint32_t env_count, uint32_t leaves_per_env,
                                      LbvhNode* out_nodes,
                                      uint32_t* morton, uint32_t* index,
                                      uint64_t* sortkey, uint32_t* visit) {
    const uint32_t E = env_count;
    const uint32_t N = leaves_per_env;
    if (E == 0u || N < 2u) return;
    {  // zero the per-leaf visit counters.
        const uint32_t n = E * N;
        const uint32_t b = (n + kBlockSize - 1u) / kBlockSize;
        ZeroU32Kernel<<<b, kBlockSize, 0, stream>>>(visit, n);
    }
    EnvMortonKernel<Src><<<dim3(E), dim3(kBlockSize), 0, stream>>>(
        src, N, morton, index, sortkey);
    {  // ONE cross-env stable sort: high env bits keep envs disjoint, low morton
       // bits order within an env, stability breaks ties by ascending local id.
        const size_t total = static_cast<size_t>(E) * N;
        auto vals = thrust::make_zip_iterator(thrust::make_tuple(index, morton));
        thrust::stable_sort_by_key(thrust::cuda::par.on(stream),
                                   sortkey, sortkey + total, vals);
    }
    const uint32_t node_count = 2u * N - 1u;
    const uint32_t node_blocks = (node_count + kBlockSize - 1u) / kBlockSize;
    EnvInitNodesKernel<Src><<<dim3(node_blocks, E), dim3(kBlockSize), 0, stream>>>(
        src, index, N, out_nodes);
    const uint32_t internal_blocks = ((N - 1u) + kBlockSize - 1u) / kBlockSize;
    EnvBuildInternalKernel<<<dim3(internal_blocks, E), dim3(kBlockSize), 0, stream>>>(
        morton, N, out_nodes);
    const uint32_t leaf_blocks = (N + kBlockSize - 1u) / kBlockSize;
    EnvPropagateKernel<<<dim3(leaf_blocks, E), dim3(kBlockSize), 0, stream>>>(
        N, out_nodes, visit);
}

template <typename Src>
inline void RefitLbvhBatchedImpl(cudaStream_t stream, LbvhNode* nodes, Src src,
                                 uint32_t env_count, uint32_t leaves_per_env,
                                 uint32_t* visit) {
    const uint32_t E = env_count;
    const uint32_t N = leaves_per_env;
    if (E == 0u || N < 2u) return;
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
inline void BuildLbvhBatchedNodes(cudaStream_t stream, int /*device_id*/,
                                  const collision::AABB* device_aabbs,
                                  uint32_t env_count,
                                  uint32_t leaves_per_env,
                                  LbvhNode* device_out_nodes,
                                  uint32_t* device_morton_scratch,
                                  uint32_t* device_index_scratch,
                                  uint64_t* device_sortkey_scratch,
                                  uint32_t* device_visit_scratch) {
    lbvh_batched_detail::BuildLbvhBatchedNodesImpl(
        stream, AabbArraySource{device_aabbs}, env_count, leaves_per_env,
        device_out_nodes, device_morton_scratch, device_index_scratch,
        device_sortkey_scratch, device_visit_scratch);
}

// Split lo/hi overload for the collision arena (body_aabb_lo / body_aabb_hi).
// Same algorithm; the source reassembles the interleaved AABB per leaf.
inline void BuildLbvhBatchedNodes(cudaStream_t stream, int /*device_id*/,
                                  const math::Vec3* device_aabb_lo,
                                  const math::Vec3* device_aabb_hi,
                                  uint32_t env_count,
                                  uint32_t leaves_per_env,
                                  LbvhNode* device_out_nodes,
                                  uint32_t* device_morton_scratch,
                                  uint32_t* device_index_scratch,
                                  uint64_t* device_sortkey_scratch,
                                  uint32_t* device_visit_scratch) {
    lbvh_batched_detail::BuildLbvhBatchedNodesImpl(
        stream, AabbSplitSource{device_aabb_lo, device_aabb_hi}, env_count,
        leaves_per_env, device_out_nodes, device_morton_scratch,
        device_index_scratch, device_sortkey_scratch, device_visit_scratch);
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
