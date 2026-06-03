#pragma once
// ---------------------------------------------------------------------------
// nuka::rt -- self-written CUDA ray tracer: camera -> LBVH -> closest-hit depth
// (v0.7 p12). NO OptiX, NO closed SDK -- pure self-written CUDA RT (master plan
// Round 10): full determinism + full differentiability + zero closed-SDK dep.
//
// This is the NAMED CONSUMER that wires the retained LBVH (BuildLbvhForQuery,
// p04+p05) into the runtime renderer (resolves the LBVH->runtime orphan): build
// box-primitive AABBs on device, BuildLbvhForQuery over them, then traverse
// DeviceNodes() with a STACKLESS parent-pointer (Hapala) ray traversal -- one
// thread per pixel, sequential local best_t/best_prim register, ZERO atomics
// (atomic-free => trivially two-run byte-exact). The leaf intersection is a REAL
// ray-box slab test (rt/ray_box.cuh) producing a genuine hit distance t; p13
// swaps box -> triangle/sphere/SDF + shading with NO traversal change.
//
// HOST header: included from g++ TUs (tests). The LBVH node type stays inside
// the .cu (forward-declared pointer in broadphase_lbvh.hpp). RenderDepth takes
// host-side box geometry, builds the device AABBs + tree itself, and returns the
// downloaded depth + prim-id framebuffers.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "rt/camera.hpp"

#include <cstdint>
#include <vector>

namespace nuka::rt {

// Result of a depth render: row-major framebuffers of size width*height.
//   depth[y*width + x]  : world-space distance to the closest box surface, or
//                         +inf (RtMissDepth) if the pixel's ray hit nothing.
//   prim[y*width + x]   : ORIGINAL box index of the closest hit, or kNoPrim on a
//                         miss. (NOT the leaf node index -- the original index,
//                         so it matches the oracle's boxes[i] indexing.)
struct DepthRender {
    uint32_t width = 0u;
    uint32_t height = 0u;
    std::vector<float> depth;
    std::vector<uint32_t> prim;
};

// Render a closest-hit depth + prim-id buffer for `camera` against the AABB
// boxes in `boxes` (PROGRAMMATIC geometry only -- built in-test; the mesh-file
// importer is a SEPARATE orphan, deliberately NOT a p12 dependency). Builds the
// device AABB array, the LBVH (BuildLbvhForQuery, RETAINING the node tree),
// launches the stackless ray-traversal kernel (one thread per pixel), and
// downloads the framebuffers. Deterministic; no RNG; atomic-free.
DepthRender RenderDepth(const std::vector<collision::AABB>& boxes,
                        const PinholeCamera& camera);

// Debug helper for the degenerate-deep-tree gate: build the LBVH over `boxes`
// and return the ACTUAL maximum leaf depth (root depth = 0), computed by
// downloading the node array and walking parent pointers from each leaf. Keeps
// the CUDA-only LbvhNode inside the .cu while letting a .cpp test assert the
// tree is genuinely deeper than a naive stack[64] could hold.
uint32_t DebugMaxTreeDepth(const std::vector<collision::AABB>& boxes);

} // namespace nuka::rt
