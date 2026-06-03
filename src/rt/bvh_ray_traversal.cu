// ---------------------------------------------------------------------------
// nuka::rt::RenderDepth / DebugMaxTreeDepth implementation (v0.7 p12).
//
// STACKLESS parent-pointer (Hapala 2011) ray traversal over the LBVH retained by
// BuildLbvhForQuery. One thread per pixel; a sequential local best_t/best_prim
// register holds the closest hit. NO STACK at all -> immune to the latent
// stack[64] overflow the existing pair-traversal carries (constraint 3), and the
// degenerate-deep-tree gate passes by construction. ZERO atomics on anything
// (each pixel has exactly one writer) -> trivially two-run byte-exact (D1).
//
// Why stackless over a guarded stack (per the design review): a runtime-sized
// register array is impossible, a global stack costs rays*depth*4B, and a fixed
// stack[256]+guard is the same latent risk the task warns against. The state
// machine needs only left/right/parent (all present in LbvhNode) and zero build
// changes.
//
// The leaf intersection is rt::RayBoxIntersect (rt/ray_box.cuh) -- a REAL slab
// test giving a genuine t; the SAME __host__ __device__ function the CPU oracle
// calls, so depth+prim-id match bit-for-bit. Tie-break (lowest prim on equal t)
// is RtClosestHitUpdate, also shared.
//
// D1 / no-float-atomic: covered by the lint physics_path scope (src/rt/**),
// which this phase adds. There are NO atomics here.
// ---------------------------------------------------------------------------

#include "rt/bvh_ray_traversal.hpp"

#include "collision/broadphase_lbvh.hpp"
#include "collision/lbvh_node.cuh"
#include "math/vec3.hpp"
#include "phi/buffer.hpp"
#include "phi/buffer_transfer.hpp"
#include "phi/device_context.hpp"
#include "rt/camera.hpp"
#include "rt/ray_box.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace nuka::rt {

namespace {

constexpr uint32_t kBlockDim = 16u; // 16x16 = 256 threads/block

void CheckCuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " +
                                 cudaGetErrorString(result));
    }
}

using ::nuka::collision::gpu::LbvhNode;

// Ray-AABB entry t for the NODE-pruning test (not a surface hit -- a node bound
// is an internal volume the ray may enter). Returns the entry distance (tmin
// clamped to 0 when the ray starts inside the node) and whether the ray
// intersects the node's AABB at all with tmax >= 0. We prune a subtree iff it is
// missed OR its entry t >= best_t (a closer hit already exists). Same fp64 slab
// math as RayBoxIntersect; node test only needs the entry distance + a hit flag.
__device__ __forceinline__ bool NodeEntryT(const math::Vec3& origin,
                                           const math::Vec3& dir,
                                           const collision::AABB& box,
                                           float* entry_t) {
    const double inv_x = 1.0 / static_cast<double>(dir.x);
    const double inv_y = 1.0 / static_cast<double>(dir.y);
    const double inv_z = 1.0 / static_cast<double>(dir.z);

    const double tx0 = (static_cast<double>(box.min.x) - static_cast<double>(origin.x)) * inv_x;
    const double tx1 = (static_cast<double>(box.max.x) - static_cast<double>(origin.x)) * inv_x;
    const double ty0 = (static_cast<double>(box.min.y) - static_cast<double>(origin.y)) * inv_y;
    const double ty1 = (static_cast<double>(box.max.y) - static_cast<double>(origin.y)) * inv_y;
    const double tz0 = (static_cast<double>(box.min.z) - static_cast<double>(origin.z)) * inv_z;
    const double tz1 = (static_cast<double>(box.max.z) - static_cast<double>(origin.z)) * inv_z;

    const double txmin = tx0 < tx1 ? tx0 : tx1;
    const double txmax = tx0 < tx1 ? tx1 : tx0;
    const double tymin = ty0 < ty1 ? ty0 : ty1;
    const double tymax = ty0 < ty1 ? ty1 : ty0;
    const double tzmin = tz0 < tz1 ? tz0 : tz1;
    const double tzmax = tz0 < tz1 ? tz1 : tz0;

    double tmin = txmin > tymin ? txmin : tymin;
    tmin = tmin > tzmin ? tmin : tzmin;
    double tmax = txmax < tymax ? txmax : tymax;
    tmax = tmax < tzmax ? tmax : tzmax;

    if (tmax < tmin || tmax < 0.0) {
        return false;
    }
    *entry_t = static_cast<float>(tmin >= 0.0 ? tmin : 0.0);
    return true;
}

// Test+possibly-update the closest hit against the leaf box at flat node index
// `leaf_node`. nodes[leaf_node].left holds the ORIGINAL box index (the prim id
// the oracle uses). Uses the shared RayBoxIntersect + RtClosestHitUpdate.
__device__ __forceinline__ void IntersectLeaf(const LbvhNode* __restrict__ nodes,
                                              int32_t leaf_node,
                                              const math::Vec3& origin,
                                              const math::Vec3& dir,
                                              float* best_t,
                                              uint32_t* best_prim) {
    const collision::AABB box = nodes[leaf_node].aabb;
    float t;
    if (RayBoxIntersect(origin, dir, box, &t)) {
        // Surface hit must be in front of the ray and not farther than best.
        if (t >= 0.0f) {
            const uint32_t prim = static_cast<uint32_t>(nodes[leaf_node].left);
            RtClosestHitUpdate(t, prim, best_t, best_prim);
        }
    }
}

// Stackless Hapala traversal of one ray. `internal_count` = N-1 (a flat index
// >= internal_count is a leaf). For N==1 the single node IS the leaf (handled by
// the caller). Returns closest (best_t, best_prim) via out-params.
//
// NOTE (premise correction, see tests): this is stackless NOT because the LBVH
// emits deep trees -- it does NOT. LbvhDelta's index tie-break (32+clz(i^j))
// keeps the Karras tree O(log N) even for fully-coincident Morton codes, so the
// existing stack[64] is safe to ~2^64 leaves. We are stackless because it is
// robust REGARDLESS of any depth assumption (no global stack memory, no
// runtime-sized register array, no compile-time stack+guard) -- the right shape
// for one-thread-per-pixel over a full framebuffer.
//
// State machine: we move between a node and its parent. `from_child` encodes
// which child we just RETURNED from (0=came from above/start, 1=returned from
// left, 2=returned from right). At an internal node we decide the next node:
// descend left, then right, then go up -- pruning any child whose bound is
// missed or whose entry t >= best_t.
__device__ void TraverseRay(const LbvhNode* __restrict__ nodes,
                            uint32_t internal_count,
                            const math::Vec3& origin,
                            const math::Vec3& dir,
                            float* best_t,
                            uint32_t* best_prim) {
    // Should we descend into `child` (prune if missed or entry t >= best_t)?
    auto want = [&](int32_t child) -> bool {
        float et;
        if (!NodeEntryT(origin, dir, nodes[child].aabb, &et)) {
            return false;
        }
        return et < *best_t;
    };

    constexpr int kFromParent = 0;
    constexpr int kFromLeft = 1;
    constexpr int kFromRight = 2;

    int32_t current = 0;      // start at root (internal node 0)
    int from = kFromParent;   // how we arrived at `current`

    // Root pruning: if the root bound is entirely missed, nothing to do.
    {
        float et;
        if (!NodeEntryT(origin, dir, nodes[0].aabb, &et)) {
            return;
        }
    }

    while (true) {
        const LbvhNode node = nodes[current];
        const int32_t left = node.left;
        const int32_t right = node.right;
        const int32_t parent = node.parent;

        int32_t next = -1;
        int next_from = kFromParent;

        if (from == kFromParent) {
            // Just arrived from above: try left child first.
            const bool left_leaf = (static_cast<uint32_t>(left) >= internal_count);
            if (left_leaf) {
                IntersectLeaf(nodes, left, origin, dir, best_t, best_prim);
                // Then try right (as if we returned from left).
                from = kFromLeft;
                continue;
            } else if (want(left)) {
                next = left;
                next_from = kFromParent;
            } else {
                // Left pruned/leaf-done: fall through to right.
                from = kFromLeft;
                continue;
            }
        } else if (from == kFromLeft) {
            // Returned from (or skipped) left: try right child.
            const bool right_leaf = (static_cast<uint32_t>(right) >= internal_count);
            if (right_leaf) {
                IntersectLeaf(nodes, right, origin, dir, best_t, best_prim);
                from = kFromRight;
                continue;
            } else if (want(right)) {
                next = right;
                next_from = kFromParent;
            } else {
                from = kFromRight;
                continue;
            }
        } else { // kFromRight: both children done -> go up.
            if (parent < 0) {
                return; // returned to root from its right subtree: done.
            }
            // Determine whether `current` was the parent's left or right child,
            // so the parent resumes from the correct state.
            const int32_t pleft = nodes[parent].left;
            next = parent;
            next_from = (pleft == current) ? kFromLeft : kFromRight;
        }

        if (next >= 0) {
            current = next;
            from = next_from;
        }
    }
}

// One thread per pixel. Generates the ray (deterministic), traverses, writes the
// closest-hit depth + prim into the framebuffer. Exactly one writer per pixel ->
// no atomics, byte-exact.
__global__ void RenderDepthKernel(PinholeCamera camera,
                                  const LbvhNode* __restrict__ nodes,
                                  uint32_t leaf_count,
                                  float* __restrict__ out_depth,
                                  uint32_t* __restrict__ out_prim) {
    const uint32_t px = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= camera.width || py >= camera.height) {
        return;
    }
    const uint32_t pixel = py * camera.width + px;

    float best_t = RtMissDepth();
    uint32_t best_prim = kNoPrim;

    const Ray ray = camera.GenerateRay(px, py);

    if (leaf_count == 1u) {
        // Single node IS the leaf (node 0); intersect directly.
        IntersectLeaf(nodes, 0, ray.origin, ray.dir, &best_t, &best_prim);
    } else if (leaf_count >= 2u) {
        TraverseRay(nodes, leaf_count - 1u, ray.origin, ray.dir, &best_t,
                    &best_prim);
    }

    out_depth[pixel] = best_t;
    out_prim[pixel] = best_prim;
}

} // namespace

DepthRender RenderDepth(const std::vector<collision::AABB>& boxes,
                        const PinholeCamera& camera) {
    DepthRender out;
    out.width = camera.width;
    out.height = camera.height;
    const size_t pixels = static_cast<size_t>(camera.width) *
                          static_cast<size_t>(camera.height);
    out.depth.assign(pixels, RtMissDepth());
    out.prim.assign(pixels, kNoPrim);

    if (pixels == 0u) {
        return out;
    }

    auto context = phi::MakeDefaultDeviceContext();
    phi::ScopedDeviceGuard guard(context.device_id);
    const cudaStream_t stream = context.stream.Native();

    // Output framebuffers (device). RenderDepth is NOT a per-step hot path (it
    // renders a sensor frame, like cross_system_query's allocations) -> these
    // allocations are out of the hot_path lint scope by design.
    phi::Buffer d_depth(pixels * sizeof(float), phi::MemoryKind::Device);
    phi::Buffer d_prim(pixels * sizeof(uint32_t), phi::MemoryKind::Device);

    const uint32_t leaf_count = static_cast<uint32_t>(boxes.size());

    if (leaf_count == 0u) {
        // No geometry: every pixel misses. Initialize device buffers to the
        // sentinels and download (keeps the device path consistent).
        // float +inf and 0xFFFFFFFF are not memset-able patterns, so just upload
        // the host-initialized sentinel vectors.
        d_depth.CopyFromHost(out.depth.data(), pixels * sizeof(float));
        d_prim.CopyFromHost(out.prim.data(), pixels * sizeof(uint32_t));
        return out;
    }

    // Build the box-primitive LBVH, RETAINING the node tree (BuildLbvhForQuery).
    phi::Buffer d_boxes = phi::UploadVector(boxes);
    const auto* dev_boxes = static_cast<const collision::AABB*>(d_boxes.Data());
    auto tree = collision::gpu::BuildLbvhForQuery(dev_boxes, leaf_count);
    if (!tree.HasNodes()) {
        throw std::runtime_error("RenderDepth: LBVH build did not retain nodes");
    }
    const LbvhNode* nodes = tree.DeviceNodes();

    const dim3 block(kBlockDim, kBlockDim);
    const dim3 grid((camera.width + kBlockDim - 1u) / kBlockDim,
                    (camera.height + kBlockDim - 1u) / kBlockDim);
    RenderDepthKernel<<<grid, block, 0, stream>>>(
        camera, nodes, leaf_count,
        static_cast<float*>(d_depth.Data()),
        static_cast<uint32_t*>(d_prim.Data()));
    CheckCuda(cudaGetLastError(), "RenderDepthKernel launch");
    context.stream.Synchronize();

    d_depth.CopyToHost(out.depth.data(), pixels * sizeof(float));
    d_prim.CopyToHost(out.prim.data(), pixels * sizeof(uint32_t));
    return out;
}

uint32_t DebugMaxTreeDepth(const std::vector<collision::AABB>& boxes) {
    const uint32_t leaf_count = static_cast<uint32_t>(boxes.size());
    if (leaf_count == 0u) {
        return 0u;
    }

    phi::Buffer d_boxes = phi::UploadVector(boxes);
    const auto* dev_boxes = static_cast<const collision::AABB*>(d_boxes.Data());
    auto tree = collision::gpu::BuildLbvhForQuery(dev_boxes, leaf_count);
    if (!tree.HasNodes()) {
        throw std::runtime_error("DebugMaxTreeDepth: LBVH build did not retain nodes");
    }

    const uint32_t node_count = tree.NodeCount(); // 2N-1
    // Download the node array via cudaMemcpy from the device node pointer (the
    // retained tree exposes DeviceNodes() but not its backing Buffer).
    std::vector<LbvhNode> host_nodes(node_count);
    CheckCuda(cudaMemcpy(host_nodes.data(), tree.DeviceNodes(),
                         node_count * sizeof(LbvhNode), cudaMemcpyDeviceToHost),
              "download LBVH nodes for depth");

    const uint32_t internal_count = leaf_count - 1u;
    uint32_t max_depth = 0u;
    if (leaf_count == 1u) {
        return 0u; // single leaf is the root
    }
    // Walk parents from each leaf up to the root, counting edges.
    for (uint32_t leaf = 0u; leaf < leaf_count; ++leaf) {
        const uint32_t leaf_node = internal_count + leaf;
        uint32_t depth = 0u;
        int32_t cur = static_cast<int32_t>(leaf_node);
        // Guard against a malformed tree (should never loop): cap at node_count.
        for (uint32_t steps = 0u; steps <= node_count; ++steps) {
            const int32_t parent = host_nodes[static_cast<size_t>(cur)].parent;
            if (parent < 0) {
                break;
            }
            ++depth;
            cur = parent;
        }
        if (depth > max_depth) {
            max_depth = depth;
        }
    }
    return max_depth;
}

} // namespace nuka::rt
