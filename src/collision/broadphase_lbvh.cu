// ---------------------------------------------------------------------------
// nuka::collision::gpu::BroadphaseLbvh implementation -- Karras 2012 LBVH.
//
// v0.7 p04. See broadphase_lbvh.hpp for the determinism / set-equality
// contract. Reference: Tero Karras, "Maximizing Parallelism in the
// Construction of BVHs, Octrees, and k-d Trees" (2012).
// ---------------------------------------------------------------------------

#include "collision/broadphase_lbvh.hpp"

#include "collision/lbvh_node.cuh"
#include "collision/lbvh_traversal.cuh"
#include "collision/morton_codes.cuh"
#include "phi/backend.hpp"            // InitBestDevice, DeviceBufferType
#include "phi/buffer_transfer_v2.hpp" // DownloadVectorV2

#include <cuda_runtime.h>

#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/sort.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nuka::collision::gpu {

namespace {

constexpr uint32_t kBlockSize = 128u;

// Default compact-pair output sizing. When the caller passes max_pairs_hint==0 we
// guess kDefaultPairFanout * count (typical broadphase fan-out) clamped to
// [kMinPairCapacity, kMaxPairCapacity] so the buffer stays small even at 50k.
constexpr uint64_t kDefaultPairFanout = 32ull;
constexpr uint64_t kMaxPairCapacity   = 64ull * 1024ull * 1024ull;  // 64M pairs cap
constexpr uint32_t kMinPairCapacity   = 1024u;

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

// File-local RAII over a phi v2 opaque Buffer*. Wraps the stream-0 device buffer
// type (NULL-stream transfers are byte+ordering identical to the legacy
// synchronous memcpy this replaces) so the scoped allocations below still free
// on every path (including the CheckCuda throw paths) exactly as the legacy RAII
// Buffer did. Move-only; Base()/Bytes() mirror the legacy Data()/Size().
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
    void*  Base()  const { return buf_ != nullptr ? phi::BufferBase(buf_) : nullptr; }
    size_t Bytes() const { return bytes_; }
    phi::Buffer* Handle() const { return buf_; }
    // Release ownership of the underlying handle to the caller (for the result's
    // long-lived pairs_/nodes_ members).
    phi::Buffer* Release() { phi::Buffer* b = buf_; buf_ = nullptr; bytes_ = 0u; return b; }
private:
    phi::Buffer* buf_ = nullptr;
    size_t bytes_ = 0u;
};

// Download `count` elements from an OwnedBuffer into a host vector. BufferDownload
// self-syncs the NULL/default stream (stream 0) -- byte+ordering identical to a
// synchronous stream-0 cudaMemcpy.
template <typename T>
std::vector<T> DownloadOwned(const OwnedBuffer& buf, uint32_t count) {
    return phi::DownloadVectorV2<T>(buf.Handle(), count);
}

// --- Scene-bound reduction --------------------------------------------------
// Block-reduced min/max of AABB centers, written out per-block; the host folds
// the small per-block array. Using a fixed-size, fixed-order host fold keeps the
// scene bound a deterministic function of the input (D1): the per-block partials
// are an associative min/max and we fold them in a fixed index order.
__global__ void ReduceCenterBoundsKernel(uint32_t count,
                                         const collision::AABB* __restrict__ aabbs,
                                         float3* __restrict__ block_min,
                                         float3* __restrict__ block_max) {
    __shared__ float3 s_min[kBlockSize];
    __shared__ float3 s_max[kBlockSize];

    const uint32_t tid = threadIdx.x;
    const uint32_t gid = blockIdx.x * blockDim.x + tid;

    float3 local_min = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    float3 local_max = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    if (gid < count) {
        const collision::AABB box = aabbs[gid];
        const float cx = 0.5f * (box.min.x + box.max.x);
        const float cy = 0.5f * (box.min.y + box.max.y);
        const float cz = 0.5f * (box.min.z + box.max.z);
        local_min = make_float3(cx, cy, cz);
        local_max = local_min;
    }
    s_min[tid] = local_min;
    s_max[tid] = local_max;
    __syncthreads();

    for (uint32_t stride = blockDim.x >> 1; stride > 0u; stride >>= 1) {
        if (tid < stride) {
            s_min[tid].x = fminf(s_min[tid].x, s_min[tid + stride].x);
            s_min[tid].y = fminf(s_min[tid].y, s_min[tid + stride].y);
            s_min[tid].z = fminf(s_min[tid].z, s_min[tid + stride].z);
            s_max[tid].x = fmaxf(s_max[tid].x, s_max[tid + stride].x);
            s_max[tid].y = fmaxf(s_max[tid].y, s_max[tid + stride].y);
            s_max[tid].z = fmaxf(s_max[tid].z, s_max[tid + stride].z);
        }
        __syncthreads();
    }

    if (tid == 0u) {
        block_min[blockIdx.x] = s_min[0];
        block_max[blockIdx.x] = s_max[0];
    }
}

// --- Morton codes -----------------------------------------------------------
__global__ void ComputeMortonKernel(uint32_t count,
                                     const collision::AABB* __restrict__ aabbs,
                                     float3 scene_min,
                                     float3 scene_inv_extent,
                                     uint32_t* __restrict__ out_morton,
                                     uint32_t* __restrict__ out_index) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const collision::AABB box = aabbs[i];
    const float cx = 0.5f * (box.min.x + box.max.x);
    const float cy = 0.5f * (box.min.y + box.max.y);
    const float cz = 0.5f * (box.min.z + box.max.z);
    const float nx = (cx - scene_min.x) * scene_inv_extent.x;
    const float ny = (cy - scene_min.y) * scene_inv_extent.y;
    const float nz = (cz - scene_min.z) * scene_inv_extent.z;
    out_morton[i] = Morton3D30(nx, ny, nz);
    out_index[i] = i;
}

// --- Node init: leaves get body id + AABB; internal nodes get parent=-1 ------
// Launched over ALL node_count nodes. CRITICAL: internal-node parents MUST be
// initialized to -1 here, because the Karras build only writes a node's parent
// when ANOTHER internal node names it as a child -- the ROOT is never anyone's
// child, so without this its parent stays garbage and the bottom-up propagation
// walks off the end of the visit array (OOB / illegal access).
__global__ void InitNodesKernel(uint32_t leaf_count,
                                const collision::AABB* __restrict__ aabbs,
                                const uint32_t* __restrict__ sorted_index,
                                LbvhNode* __restrict__ nodes) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t internal_count = leaf_count - 1u;
    const uint32_t node_count = 2u * leaf_count - 1u;
    if (idx >= node_count) {
        return;
    }
    if (idx < internal_count) {
        // Internal node: parent set by the build (except the root); pre-seed -1.
        nodes[idx].parent = -1;
        return;
    }
    const uint32_t lane = idx - internal_count;
    const uint32_t body = sorted_index[lane];
    LbvhNode leaf;
    leaf.left = static_cast<int32_t>(body); // leaf stores ORIGINAL body id
    leaf.right = -1;
    leaf.parent = -1;
    leaf.aabb = aabbs[body];
    nodes[idx] = leaf;
}

// --- Karras internal-node build --------------------------------------------
// One thread per internal node i in [0, leaf_count-1). Determines the range of
// leaves it covers and its split position, sets children + parent links.
__global__ void BuildInternalNodesKernel(uint32_t leaf_count,
                                         const uint32_t* __restrict__ morton_sorted,
                                         LbvhNode* __restrict__ nodes) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t internal_count = leaf_count - 1u;
    if (idx >= internal_count) {
        return;
    }
    const int i = static_cast<int>(idx);
    const int n = static_cast<int>(leaf_count);

    // Determine direction of the range (+1 or -1).
    const int d_l = LbvhDelta(i, i - 1, leaf_count, morton_sorted);
    const int d_r = LbvhDelta(i, i + 1, leaf_count, morton_sorted);
    const int d = (d_r >= d_l) ? 1 : -1;

    // Compute upper bound for the length of the range.
    const int delta_min = LbvhDelta(i, i - d, leaf_count, morton_sorted);
    int l_max = 2;
    while (LbvhDelta(i, i + l_max * d, leaf_count, morton_sorted) > delta_min) {
        l_max *= 2;
    }

    // Binary search the actual length using LCP.
    int l = 0;
    for (int t = l_max >> 1; t >= 1; t >>= 1) {
        if (LbvhDelta(i, i + (l + t) * d, leaf_count, morton_sorted) > delta_min) {
            l += t;
        }
    }
    const int j = i + l * d; // other end of the range

    // Find the split position via the in-range LCP.
    const int delta_node = LbvhDelta(i, j, leaf_count, morton_sorted);
    int s = 0;
    int t_div = l;
    do {
        t_div = (t_div + 1) >> 1;
        if (LbvhDelta(i, i + (s + t_div) * d, leaf_count, morton_sorted) > delta_node) {
            s += t_div;
        }
    } while (t_div > 1);
    const int gamma = i + s * d + min(d, 0);

    const int first = min(i, j);
    const int last = max(i, j);

    // Children: leaf if the range end coincides with the split boundary.
    const int32_t left_child =
        (first == gamma) ? static_cast<int32_t>(internal_count + gamma)
                         : static_cast<int32_t>(gamma);
    const int32_t right_child =
        (last == gamma + 1) ? static_cast<int32_t>(internal_count + gamma + 1)
                            : static_cast<int32_t>(gamma + 1);

    nodes[i].left = left_child;
    nodes[i].right = right_child;
    nodes[left_child].parent = i;
    nodes[right_child].parent = i;
    (void)n;
}

// --- Bottom-up AABB propagation (integer atomic visit-counters) -------------
// One thread per leaf. Each walks toward the root; the SECOND child to arrive
// at a node (tracked by an atomicAdd on a uint32 counter -> integer atomic, D1
// safe) merges both child bounds into the node and continues upward. The first
// arrival stops. The MERGE itself (fminf/fmaxf) is order-independent, so the
// result is identical regardless of which child fires the atomic first.
__global__ void PropagateAabbsKernel(uint32_t leaf_count,
                                     LbvhNode* __restrict__ nodes,
                                     uint32_t* __restrict__ visit) {
    const uint32_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= leaf_count) {
        return;
    }
    const uint32_t internal_count = leaf_count - 1u;
    int32_t node = nodes[internal_count + lane].parent;
    while (node >= 0) {
        // Make THIS thread's child-AABB write (the leaf bound on the first
        // hop, or the merged bound on subsequent hops) globally visible BEFORE
        // we signal arrival via the atomic. Without this fence the second
        // arriving thread can observe the incremented counter while the child
        // bound write is still in flight -> it merges a stale/garbage child
        // AABB. That race is invisible at n=2/3 (propagation is single-threaded
        // up each path) but corrupts bounds at scale (run-to-run-varying pair
        // counts + missed overlaps). The fence is the Karras bottom-up fix.
        __threadfence();
        const uint32_t prev = atomicAdd(&visit[node], 1u);
        if (prev == 0u) {
            return; // first child here; second will do the merge
        }
        // Second arrival: both children's bounds are now visible.
        const LbvhNode self = nodes[node];
        nodes[node].aabb = LbvhMerge(nodes[self.left].aabb, nodes[self.right].aabb);
        node = nodes[node].parent;
    }
}

// Pack canonical (a,b) into a uint64 sort key = (a<<32)|b. Used to make the
// compact emitted pair list deterministically sorted (D1).
__global__ void PackKeysKernel(const collision::CollisionPair* __restrict__ pairs,
                               uint64_t* __restrict__ keys,
                               uint32_t count) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) {
        return;
    }
    const collision::CollisionPair p = pairs[i];
    keys[i] = (static_cast<uint64_t>(p.body_a) << 32) | static_cast<uint64_t>(p.body_b);
}

float3 Fold(const std::vector<float3>& mins, bool is_max) {
    float3 acc = is_max ? make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX)
                        : make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
    for (const float3& v : mins) {
        if (is_max) {
            acc.x = std::max(acc.x, v.x);
            acc.y = std::max(acc.y, v.y);
            acc.z = std::max(acc.z, v.z);
        } else {
            acc.x = std::min(acc.x, v.x);
            acc.y = std::min(acc.y, v.y);
            acc.z = std::min(acc.z, v.z);
        }
    }
    return acc;
}

} // namespace

LbvhBroadphaseResult::LbvhBroadphaseResult(uint32_t leaf_count,
                                           uint32_t pair_count,
                                           uint32_t pair_capacity,
                                           phi::Buffer* pairs,
                                           bool truncated)
    : leaf_count_(leaf_count)
    , pair_count_(pair_count)
    , pair_capacity_(pair_capacity)
    , truncated_(truncated)
    , pairs_(pairs) {}

LbvhBroadphaseResult::LbvhBroadphaseResult(uint32_t leaf_count,
                                           uint32_t pair_count,
                                           uint32_t pair_capacity,
                                           phi::Buffer* pairs,
                                           phi::Buffer* nodes,
                                           bool truncated)
    : leaf_count_(leaf_count)
    , pair_count_(pair_count)
    , pair_capacity_(pair_capacity)
    , truncated_(truncated)
    , pairs_(pairs)
    , nodes_(nodes) {}

LbvhBroadphaseResult::~LbvhBroadphaseResult() {
    if (pairs_ != nullptr) { phi::BufferFree(pairs_); }
    if (nodes_ != nullptr) { phi::BufferFree(nodes_); }
}

LbvhBroadphaseResult::LbvhBroadphaseResult(LbvhBroadphaseResult&& other) noexcept
    : leaf_count_(other.leaf_count_)
    , pair_count_(other.pair_count_)
    , pair_capacity_(other.pair_capacity_)
    , truncated_(other.truncated_)
    , pairs_(other.pairs_)
    , nodes_(other.nodes_) {
    other.leaf_count_ = 0u;
    other.pair_count_ = 0u;
    other.pair_capacity_ = 0u;
    other.truncated_ = false;
    other.pairs_ = nullptr;
    other.nodes_ = nullptr;
}

LbvhBroadphaseResult& LbvhBroadphaseResult::operator=(LbvhBroadphaseResult&& other) noexcept {
    if (this != &other) {
        if (pairs_ != nullptr) { phi::BufferFree(pairs_); }
        if (nodes_ != nullptr) { phi::BufferFree(nodes_); }
        leaf_count_ = other.leaf_count_;
        pair_count_ = other.pair_count_;
        pair_capacity_ = other.pair_capacity_;
        truncated_ = other.truncated_;
        pairs_ = other.pairs_;
        nodes_ = other.nodes_;
        other.leaf_count_ = 0u;
        other.pair_count_ = 0u;
        other.pair_capacity_ = 0u;
        other.truncated_ = false;
        other.pairs_ = nullptr;
        other.nodes_ = nullptr;
    }
    return *this;
}

const collision::CollisionPair* LbvhBroadphaseResult::DevicePairs() const {
    return static_cast<const collision::CollisionPair*>(
        pairs_ != nullptr ? phi::BufferBase(pairs_) : nullptr);
}

const LbvhNode* LbvhBroadphaseResult::DeviceNodes() const {
    if (nodes_ == nullptr || NodeCount() == 0u) {
        return nullptr;
    }
    return static_cast<const LbvhNode*>(phi::BufferBase(nodes_));
}

bool LbvhBroadphaseResult::HasNodes() const {
    return nodes_ != nullptr && NodeCount() != 0u;
}

std::vector<collision::CollisionPair> LbvhBroadphaseResult::DownloadPairs() const {
    const uint32_t n = std::min(pair_count_, pair_capacity_);
    return phi::DownloadVectorV2<collision::CollisionPair>(pairs_, n);
}

// Shared build implementation. `retain_nodes` selects whether the built tree is
// kept in the result (cross-system query) or destroyed (default pair-only path).
// The compute pipeline is IDENTICAL in both cases -- the flag only affects what
// the result HOLDS at the end -- so the default path stays byte-identical to p04.
static LbvhBroadphaseResult BuildLbvhImpl(cudaStream_t stream, int device_id,
                                          const collision::AABB* device_aabbs,
                                          uint32_t count,
                                          uint32_t max_pairs_hint,
                                          bool retain_nodes) {
    phi::ScopedDeviceGuard guard(device_id);

    if (count < 2u) {
        // 0 or 1 body => no pairs possible.
        OwnedBuffer empty(0u);
        if (retain_nodes && count == 1u) {
            // A 1-leaf tree has a single node (the leaf itself). Retain it so the
            // cross-system query can still report that one body as a candidate.
            // The single sorted-index is trivially {0}; reuse InitNodesKernel.
            OwnedBuffer nodes(sizeof(LbvhNode));
            OwnedBuffer index(sizeof(uint32_t));
            CheckCuda(cudaMemsetAsync(index.Base(), 0, sizeof(uint32_t), stream),
                      "cudaMemsetAsync n=1 index");
            InitNodesKernel<<<1u, kBlockSize, 0, stream>>>(
                1u, device_aabbs,
                static_cast<uint32_t*>(index.Base()),
                static_cast<LbvhNode*>(nodes.Base()));
            CheckCuda(cudaGetLastError(), "InitNodesKernel launch (n=1)");
            cudaStreamSynchronize(stream);
            OwnedBuffer empty_pairs(0u);
            return LbvhBroadphaseResult(count, 0u, 0u, empty_pairs.Release(),
                                        nodes.Release());
        }
        return LbvhBroadphaseResult(count, 0u, 0u, empty.Release());
    }

    // Compact-pair output capacity. Default heuristic (named constants above):
    // kDefaultPairFanout * count clamped to [kMinPairCapacity, kMaxPairCapacity].
    // Callers that know the true overlap bound pass it explicitly.
    uint32_t pair_capacity = max_pairs_hint;
    if (pair_capacity == 0u) {
        const uint64_t guess = static_cast<uint64_t>(count) * kDefaultPairFanout;
        pair_capacity = static_cast<uint32_t>(std::min(guess, kMaxPairCapacity));
        if (pair_capacity < kMinPairCapacity) {
            pair_capacity = kMinPairCapacity;
        }
    }

    const uint32_t blocks = (count + kBlockSize - 1u) / kBlockSize;

    // --- 1. Scene-bound reduction over AABB centers -------------------------
    OwnedBuffer d_block_min(blocks * sizeof(float3));
    OwnedBuffer d_block_max(blocks * sizeof(float3));
    ReduceCenterBoundsKernel<<<blocks, kBlockSize, 0, stream>>>(
        count, device_aabbs,
        static_cast<float3*>(d_block_min.Base()),
        static_cast<float3*>(d_block_max.Base()));
    CheckCuda(cudaGetLastError(), "ReduceCenterBoundsKernel launch");

    // The reduction kernel runs on the work stream; synchronize before the
    // host-side (default-stream) download so the per-block partials are valid.
    cudaStreamSynchronize(stream);
    const auto h_block_min = DownloadOwned<float3>(d_block_min, blocks);
    const auto h_block_max = DownloadOwned<float3>(d_block_max, blocks);

    const float3 scene_min = Fold(h_block_min, /*is_max=*/false);
    const float3 scene_max = Fold(h_block_max, /*is_max=*/true);
    float3 inv_extent;
    inv_extent.x = (scene_max.x > scene_min.x) ? 1.0f / (scene_max.x - scene_min.x) : 0.0f;
    inv_extent.y = (scene_max.y > scene_min.y) ? 1.0f / (scene_max.y - scene_min.y) : 0.0f;
    inv_extent.z = (scene_max.z > scene_min.z) ? 1.0f / (scene_max.z - scene_min.z) : 0.0f;

    // --- 2. Morton codes + index ------------------------------------------
    OwnedBuffer d_morton(count * sizeof(uint32_t));
    OwnedBuffer d_index(count * sizeof(uint32_t));
    ComputeMortonKernel<<<blocks, kBlockSize, 0, stream>>>(
        count, device_aabbs, scene_min, inv_extent,
        static_cast<uint32_t*>(d_morton.Base()),
        static_cast<uint32_t*>(d_index.Base()));
    CheckCuda(cudaGetLastError(), "ComputeMortonKernel launch");

    // --- 3. Deterministic radix sort by Morton key ------------------------
    // thrust::stable_sort_by_key uses radix sort for integral keys (stable,
    // deterministic across runs). This gives D1 byte-exact sorted order.
    thrust::stable_sort_by_key(
        thrust::cuda::par.on(stream),
        static_cast<uint32_t*>(d_morton.Base()),
        static_cast<uint32_t*>(d_morton.Base()) + count,
        static_cast<uint32_t*>(d_index.Base()));
    CheckCuda(cudaGetLastError(), "stable_sort_by_key");

    // --- 4. Allocate nodes + visit counters --------------------------------
    const uint32_t node_count = 2u * count - 1u;
    const uint32_t internal_count = count - 1u;
    OwnedBuffer d_nodes(node_count * sizeof(LbvhNode));
    OwnedBuffer d_visit(internal_count * sizeof(uint32_t));
    CheckCuda(cudaMemsetAsync(d_visit.Base(), 0, internal_count * sizeof(uint32_t), stream),
              "cudaMemsetAsync visit");

    // --- 5. Init nodes, build internal nodes, propagate AABBs --------------
    const uint32_t node_blocks = (node_count + kBlockSize - 1u) / kBlockSize;
    InitNodesKernel<<<node_blocks, kBlockSize, 0, stream>>>(
        count, device_aabbs,
        static_cast<uint32_t*>(d_index.Base()),
        static_cast<LbvhNode*>(d_nodes.Base()));
    CheckCuda(cudaGetLastError(), "InitNodesKernel launch");

    const uint32_t internal_blocks = (internal_count + kBlockSize - 1u) / kBlockSize;
    BuildInternalNodesKernel<<<internal_blocks, kBlockSize, 0, stream>>>(
        count,
        static_cast<uint32_t*>(d_morton.Base()),
        static_cast<LbvhNode*>(d_nodes.Base()));
    CheckCuda(cudaGetLastError(), "BuildInternalNodesKernel launch");

    PropagateAabbsKernel<<<blocks, kBlockSize, 0, stream>>>(
        count,
        static_cast<LbvhNode*>(d_nodes.Base()),
        static_cast<uint32_t*>(d_visit.Base()));
    CheckCuda(cudaGetLastError(), "PropagateAabbsKernel launch");

    // --- 6. Overlap traversal -> compact pair list -------------------------
    OwnedBuffer d_pairs(pair_capacity * sizeof(collision::CollisionPair));
    OwnedBuffer d_pair_count(sizeof(uint32_t));
    CheckCuda(cudaMemsetAsync(d_pair_count.Base(), 0, sizeof(uint32_t), stream),
              "cudaMemsetAsync pair count");

    LbvhPairQueryKernel<<<blocks, kBlockSize, 0, stream>>>(
        count,
        static_cast<const LbvhNode*>(d_nodes.Base()),
        static_cast<collision::CollisionPair*>(d_pairs.Base()),
        static_cast<uint32_t*>(d_pair_count.Base()),
        pair_capacity);
    CheckCuda(cudaGetLastError(), "LbvhPairQueryKernel launch");

    cudaStreamSynchronize(stream);

    uint32_t pair_count = 0u;
    phi::BufferDownload(d_pair_count.Handle(), &pair_count, 0, sizeof(pair_count));
    // Truncation is silent data loss: the emit kernel atomically counts EVERY
    // overlap but only the first pair_capacity fit the buffer. Surface it loudly
    // AND structurally (truncated flag on the result) so callers can branch.
    const bool truncated = pair_count > pair_capacity;
    if (truncated) {
        std::fprintf(stderr,
                     "[nuka_lbvh] WARNING: broadphase pair overflow -- %u overlaps "
                     "exceed capacity %u; %u pairs DROPPED. Pass a larger "
                     "max_pairs_hint.\n",
                     pair_count, pair_capacity, pair_count - pair_capacity);
    }
    const uint32_t valid = std::min(pair_count, pair_capacity);

    // --- 7. Deterministic sort of the compact pair list -------------------
    // Pack (a,b) into a uint64 key = (a<<32)|b and stable-sort; this removes the
    // nondeterministic emit order so the output is byte-identical across runs.
    if (valid > 1u) {
        OwnedBuffer d_keys(valid * sizeof(uint64_t));
        const uint32_t pack_blocks = (valid + kBlockSize - 1u) / kBlockSize;
        PackKeysKernel<<<pack_blocks, kBlockSize, 0, stream>>>(
            static_cast<const collision::CollisionPair*>(d_pairs.Base()),
            static_cast<uint64_t*>(d_keys.Base()),
            valid);
        CheckCuda(cudaGetLastError(), "PackKeysKernel launch");
        thrust::stable_sort_by_key(
            thrust::cuda::par.on(stream),
            static_cast<uint64_t*>(d_keys.Base()),
            static_cast<uint64_t*>(d_keys.Base()) + valid,
            static_cast<collision::CollisionPair*>(d_pairs.Base()));
        CheckCuda(cudaGetLastError(), "stable_sort pairs");
        cudaStreamSynchronize(stream);
    }

    if (retain_nodes) {
        return LbvhBroadphaseResult(count, valid, pair_capacity,
                                    d_pairs.Release(), d_nodes.Release(), truncated);
    }
    return LbvhBroadphaseResult(count, valid, pair_capacity, d_pairs.Release(),
                                truncated);
}

LbvhBroadphaseResult BuildLbvhBroadphase(cudaStream_t stream, int device_id,
                                         const collision::AABB* device_aabbs,
                                         uint32_t count,
                                         uint32_t max_pairs_hint) {
    return BuildLbvhImpl(stream, device_id, device_aabbs, count, max_pairs_hint,
                         /*retain_nodes=*/false);
}

LbvhBroadphaseResult BuildLbvhForQuery(cudaStream_t stream, int device_id,
                                       const collision::AABB* device_aabbs,
                                       uint32_t count,
                                       uint32_t max_pairs_hint) {
    return BuildLbvhImpl(stream, device_id, device_aabbs, count, max_pairs_hint,
                         /*retain_nodes=*/true);
}

LbvhBroadphaseResult BuildLbvhForQuery(const collision::AABB* device_aabbs,
                                       uint32_t count,
                                       uint32_t max_pairs_hint) {
        return BuildLbvhForQuery(/*stream=*/nullptr, /*device_id=*/0, device_aabbs, count, max_pairs_hint);
}

LbvhBroadphaseResult BuildLbvhBroadphase(const collision::AABB* device_aabbs,
                                         uint32_t count,
                                         uint32_t max_pairs_hint) {
        return BuildLbvhBroadphase(/*stream=*/nullptr, /*device_id=*/0, device_aabbs, count, max_pairs_hint);
}

} // namespace nuka::collision::gpu
