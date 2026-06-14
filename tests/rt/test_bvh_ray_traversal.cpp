// ---------------------------------------------------------------------------
// v0.7 p12 GATE: the self-written CUDA ray tracer (camera -> LBVH -> closest-hit
// depth, box leaves) must match a CPU brute-force ray-vs-all-boxes oracle, be
// two-run byte-exact (D1), and survive a degenerate-DEEP tree (deeper than a
// naive stack[64]) without overflow.
//
// The oracle reuses the SAME __host__ __device__ slab test (rt::RayBoxIntersect)
// + closest-hit update (rt::RtClosestHitUpdate) + camera ray-gen
// (PinholeCamera::GenerateRay) that the GPU kernel uses, so depth+prim-id match
// bit-for-bit (the only thing that could diverge is FMA contraction, which is
// forbidden in both TUs: --fmad=false on the .cu, -ffp-contract=off here).
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "math/vec3.hpp"
#include "rt/bvh_ray_traversal.hpp"
#include "rt/camera.hpp"
#include "phi/backend_cuda/rt/ray_box.cuh" // host-callable: RayBoxIntersect / RtClosestHitUpdate / kNoPrim

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace nuka;

namespace {

constexpr float kPi = 3.14159265358979323846f;

// CPU brute-force oracle: for each pixel, generate the same ray, intersect it
// against EVERY box (original index = box index), keep the closest with the
// lowest-index tie-break. Identical arithmetic to the kernel (shared functions).
rt::DepthRender CpuOracle(const std::vector<collision::AABB>& boxes,
                          const rt::PinholeCamera& cam) {
    rt::DepthRender out;
    out.width = cam.width;
    out.height = cam.height;
    const size_t pixels = static_cast<size_t>(cam.width) * cam.height;
    out.depth.assign(pixels, rt::RtMissDepth());
    out.prim.assign(pixels, rt::kNoPrim);

    for (uint32_t py = 0; py < cam.height; ++py) {
        for (uint32_t px = 0; px < cam.width; ++px) {
            const rt::Ray ray = cam.GenerateRay(px, py);
            float best_t = rt::RtMissDepth();
            uint32_t best_prim = rt::kNoPrim;
            for (uint32_t i = 0; i < boxes.size(); ++i) {
                float t;
                if (rt::RayBoxIntersect(ray.origin, ray.dir, boxes[i], &t)) {
                    if (t >= 0.0f) {
                        rt::RtClosestHitUpdate(t, i, &best_t, &best_prim);
                    }
                }
            }
            const size_t pixel = static_cast<size_t>(py) * cam.width + px;
            out.depth[pixel] = best_t;
            out.prim[pixel] = best_prim;
        }
    }
    return out;
}

collision::AABB Box(math::Vec3 c, float hx, float hy, float hz) {
    collision::AABB b;
    b.min = {c.x - hx, c.y - hy, c.z - hz};
    b.max = {c.x + hx, c.y + hy, c.z + hz};
    return b;
}

// Compare a GPU render against the CPU oracle: prim-ids must match EXACTLY;
// depth must be fp32-EXACT (identical arithmetic). Returns max |depth diff| over
// pixels that both agree are hits. Misses must agree (both +inf / both kNoPrim).
struct CompareResult {
    bool prim_exact = true;
    float max_depth_err = 0.0f;
    size_t hit_pixels = 0;
    size_t mismatched_prim = 0;
};

CompareResult Compare(const rt::DepthRender& got, const rt::DepthRender& oracle) {
    CompareResult r;
    EXPECT_EQ(got.width, oracle.width);
    EXPECT_EQ(got.height, oracle.height);
    const size_t n = got.depth.size();
    for (size_t i = 0; i < n; ++i) {
        if (got.prim[i] != oracle.prim[i]) {
            r.prim_exact = false;
            ++r.mismatched_prim;
        }
        const bool got_hit = (got.prim[i] != rt::kNoPrim);
        const bool ora_hit = (oracle.prim[i] != rt::kNoPrim);
        if (got_hit && ora_hit) {
            ++r.hit_pixels;
            const float d = std::fabs(got.depth[i] - oracle.depth[i]);
            if (d > r.max_depth_err) r.max_depth_err = d;
        }
    }
    return r;
}

} // namespace

// --- Single box: a wall straight ahead. Some pixels hit, edges miss. ---------
TEST(RtBvhTraversal, SingleBoxMatchesOracle) {
    std::vector<collision::AABB> boxes = {Box({0.0f, 0.0f, 5.0f}, 1.0f, 1.0f, 1.0f)};
    auto cam = rt::BuildPinhole({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
                                {0.0f, 1.0f, 0.0f}, 0.6f * kPi, 64u, 48u);
    auto got = rt::RenderDepth(boxes, cam);
    auto oracle = CpuOracle(boxes, cam);
    auto cmp = Compare(got, oracle);
    EXPECT_TRUE(cmp.prim_exact) << "prim-id mismatches: " << cmp.mismatched_prim;
    EXPECT_EQ(cmp.max_depth_err, 0.0f);
    EXPECT_GT(cmp.hit_pixels, 0u) << "scene produced no hits -- camera mis-aimed";
}

// --- A grid of boxes: exercises real BVH descent + many leaves. --------------
TEST(RtBvhTraversal, GridOfBoxesMatchesOracle) {
    std::vector<collision::AABB> boxes;
    for (int gx = -3; gx <= 3; ++gx) {
        for (int gy = -2; gy <= 2; ++gy) {
            boxes.push_back(Box({static_cast<float>(gx) * 1.5f,
                                 static_cast<float>(gy) * 1.5f, 8.0f},
                                0.5f, 0.5f, 0.5f));
        }
    }
    auto cam = rt::BuildPinhole({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
                                {0.0f, 1.0f, 0.0f}, 0.7f * kPi, 96u, 72u);
    auto got = rt::RenderDepth(boxes, cam);
    auto oracle = CpuOracle(boxes, cam);
    auto cmp = Compare(got, oracle);
    EXPECT_TRUE(cmp.prim_exact) << "prim-id mismatches: " << cmp.mismatched_prim;
    EXPECT_EQ(cmp.max_depth_err, 0.0f);
    EXPECT_GT(cmp.hit_pixels, 100u);
}

// --- Overlapping boxes sharing a front plane: forces an EQUAL-t tie; the lower
// index must win in BOTH the kernel and the oracle. We aim a camera so a column
// of pixels hits the shared front face of two boxes at the same z. -----------
TEST(RtBvhTraversal, EqualTTieBreakLowestIndex) {
    // Two boxes with the SAME front face at z=4 (shared plane on the hit axis),
    // overlapping in the image. Box 0 and box 1 both present their min.z=4 face.
    // index 0 is added second-from-left so we can verify lowest-index wins.
    std::vector<collision::AABB> boxes;
    // box 0: spans x in [-1,1]
    boxes.push_back(Box({0.0f, 0.0f, 5.0f}, 1.0f, 1.0f, 1.0f));
    // box 1: SAME front plane z=4, overlaps box 0 in x/y entirely (slightly
    // wider), so along any ray that hits, the entry plane (z=4) is identical ->
    // identical t. Lower index (0) must win.
    boxes.push_back(Box({0.0f, 0.0f, 5.5f}, 1.5f, 1.5f, 1.5f));
    // Both have min.z = 4.0 exactly.
    boxes[0].min.z = 4.0f;
    boxes[1].min.z = 4.0f;

    auto cam = rt::BuildPinhole({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
                                {0.0f, 1.0f, 0.0f}, 0.4f * kPi, 48u, 48u);
    auto got = rt::RenderDepth(boxes, cam);
    auto oracle = CpuOracle(boxes, cam);
    auto cmp = Compare(got, oracle);
    EXPECT_TRUE(cmp.prim_exact) << "prim-id mismatches: " << cmp.mismatched_prim;
    EXPECT_EQ(cmp.max_depth_err, 0.0f);

    // Verify the tie genuinely fired: at least one pixel hits both boxes' front
    // plane at the same t, and the kept prim is 0 (the lower index). Find a
    // central pixel that hits and assert prim==0 there.
    size_t tie_pixels = 0;
    for (uint32_t py = 0; py < cam.height; ++py) {
        for (uint32_t px = 0; px < cam.width; ++px) {
            const rt::Ray ray = cam.GenerateRay(px, py);
            float t0, t1;
            const bool h0 = rt::RayBoxIntersect(ray.origin, ray.dir, boxes[0], &t0);
            const bool h1 = rt::RayBoxIntersect(ray.origin, ray.dir, boxes[1], &t1);
            if (h0 && h1 && t0 == t1 && t0 >= 0.0f) {
                ++tie_pixels;
                const size_t pixel = static_cast<size_t>(py) * cam.width + px;
                EXPECT_EQ(got.prim[pixel], 0u)
                    << "equal-t tie should keep lowest index at pixel (" << px
                    << "," << py << ")";
            }
        }
    }
    EXPECT_GT(tie_pixels, 0u)
        << "tie scene is vacuous -- no pixel produced an equal-t hit on both boxes";
}

// --- Rays that miss -> sentinel depth (+inf) / kNoPrim. ----------------------
TEST(RtBvhTraversal, MissesProduceSentinel) {
    std::vector<collision::AABB> boxes = {Box({0.0f, 0.0f, 5.0f}, 0.2f, 0.2f, 0.2f)};
    // Aim AWAY from the box so every ray misses.
    auto cam = rt::BuildPinhole({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, -1.0f},
                                {0.0f, 1.0f, 0.0f}, 0.3f * kPi, 32u, 32u);
    auto got = rt::RenderDepth(boxes, cam);
    for (size_t i = 0; i < got.depth.size(); ++i) {
        EXPECT_EQ(got.prim[i], rt::kNoPrim);
        EXPECT_TRUE(std::isinf(got.depth[i]));
    }
    // And it still matches the oracle (all-miss).
    auto oracle = CpuOracle(boxes, cam);
    auto cmp = Compare(got, oracle);
    EXPECT_TRUE(cmp.prim_exact);
    EXPECT_EQ(cmp.hit_pixels, 0u);
}

// --- D1: two-run byte-exact depth + prim-id buffers (memcmp). ----------------
TEST(RtBvhTraversal, D1TwoRunByteExact) {
    std::vector<collision::AABB> boxes;
    for (int gx = -4; gx <= 4; ++gx) {
        for (int gy = -3; gy <= 3; ++gy) {
            boxes.push_back(Box({static_cast<float>(gx) * 1.1f,
                                 static_cast<float>(gy) * 1.1f,
                                 6.0f + 0.3f * static_cast<float>(gx + gy)},
                                0.4f, 0.4f, 0.4f));
        }
    }
    auto cam = rt::BuildPinhole({0.2f, -0.1f, 0.0f}, {0.0f, 0.0f, 6.0f},
                                {0.0f, 1.0f, 0.0f}, 0.8f * kPi, 128u, 96u);
    auto r1 = rt::RenderDepth(boxes, cam);
    auto r2 = rt::RenderDepth(boxes, cam);
    ASSERT_EQ(r1.depth.size(), r2.depth.size());
    ASSERT_EQ(r1.prim.size(), r2.prim.size());
    EXPECT_EQ(0, std::memcmp(r1.depth.data(), r2.depth.data(),
                             r1.depth.size() * sizeof(float)))
        << "depth buffer not two-run byte-exact (D1)";
    EXPECT_EQ(0, std::memcmp(r1.prim.data(), r2.prim.data(),
                             r1.prim.size() * sizeof(uint32_t)))
        << "prim-id buffer not two-run byte-exact (D1)";
}

// --- Sanity: DebugMaxTreeDepth is not under-counting. 8 well-separated boxes
// build a balanced tree of depth ~log2(8)=3. ---------------------------------
TEST(RtBvhTraversal, DebugMaxTreeDepthBalancedScene) {
    std::vector<collision::AABB> boxes;
    for (int i = 0; i < 8; ++i) {
        boxes.push_back(Box({static_cast<float>(i) * 4.0f, 0.0f, 5.0f},
                            0.5f, 0.5f, 0.5f));
    }
    const uint32_t depth = rt::DebugMaxTreeDepth(boxes);
    EXPECT_EQ(depth, 3u) << "balanced 8-leaf tree should be depth 3; got " << depth;
}

// --- Degenerate Morton-CLUSTERED tree: thousands of near-coincident boxes whose
// centroids collapse into a single Morton cell. This is the DEEPEST tree this
// Karras build can emit. PREMISE CORRECTION (reported to controller): the task
// framed stack[64] as a "latent overflow risk for Morton-clustered trees" -- but
// LbvhDelta augments equal Morton codes with `32 + clz(i^j)` (the standard Karras
// duplicate-key fix), so the effective keys become the contiguous unique indices
// 0..N-1 -> a BALANCED O(log N) tree, NOT a linear chain. Identical codes do NOT
// deepen the tree; you'd need ~2^64 leaves to reach depth 64. So `stack[64]` is
// actually safe here. Our STACKLESS traversal is still the right call (robust
// regardless of the build's depth properties), and this test proves it renders
// the deepest achievable tree correctly with NO OOB. We measure + report the
// actual depth and assert it stays small (log-bounded) while the render matches
// the oracle bit-for-bit. ------------------------------------------------------
TEST(RtBvhTraversal, DegenerateClusteredTreeMatchesOracle) {
    // ~4096 boxes clustered with sub-cell jitter so their Morton codes are
    // (largely) identical -> exercises the index tie-break + the deepest tree
    // this build emits + the stackless state machine under stress.
    std::vector<collision::AABB> boxes;
    const int cluster = 4096;
    for (int i = 0; i < cluster; ++i) {
        const float off = static_cast<float>(i) * 1e-5f;
        boxes.push_back(Box({off, off, 5.0f + off}, 0.5f, 0.5f, 0.5f));
    }

    const uint32_t depth = rt::DebugMaxTreeDepth(boxes);
    // Log-depth-bounded: ceil(log2(4096))=12; allow slack for split imbalance
    // but assert it is nowhere near 64 (the evidence that this build is safe).
    EXPECT_LT(depth, 32u)
        << "Karras build unexpectedly deep (" << depth
        << ") -- premise that it is log-bounded is wrong; revisit stack safety";
    RecordProperty("measured_max_tree_depth", static_cast<int>(depth));

    auto cam = rt::BuildPinhole({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
                                {0.0f, 1.0f, 0.0f}, 0.5f * kPi, 64u, 64u);
    auto got = rt::RenderDepth(boxes, cam);
    auto oracle = CpuOracle(boxes, cam);
    auto cmp = Compare(got, oracle);
    EXPECT_TRUE(cmp.prim_exact)
        << "clustered-tree render diverged from oracle: " << cmp.mismatched_prim
        << " prim mismatches (stackless traversal mis-walked the deep tree)";
    EXPECT_EQ(cmp.max_depth_err, 0.0f);
    EXPECT_GT(cmp.hit_pixels, 0u);
    // Surface the measured depth in the test log for the controller.
    std::printf("[ DEPTH    ] DegenerateClusteredTree: %u boxes -> max tree depth %u\n",
                static_cast<uint32_t>(boxes.size()), depth);
}

// --- Empty scene (no boxes): every pixel must be a miss (sentinel). ----------
TEST(RtBvhTraversal, EmptySceneAllMiss) {
    std::vector<collision::AABB> boxes; // empty
    auto cam = rt::BuildPinhole({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
                                {0.0f, 1.0f, 0.0f}, 0.5f * kPi, 16u, 16u);
    auto got = rt::RenderDepth(boxes, cam);
    ASSERT_EQ(got.depth.size(), static_cast<size_t>(16) * 16);
    for (size_t i = 0; i < got.depth.size(); ++i) {
        EXPECT_EQ(got.prim[i], rt::kNoPrim);
        EXPECT_TRUE(std::isinf(got.depth[i]));
    }
    // Matches the (all-miss) oracle too.
    auto oracle = CpuOracle(boxes, cam);
    EXPECT_EQ(0, std::memcmp(got.prim.data(), oracle.prim.data(),
                             got.prim.size() * sizeof(uint32_t)));
}

// --- Single box, single pixel sanity: depth equals the analytic distance. ----
TEST(RtBvhTraversal, AnalyticDepthCenterPixel) {
    // Box front face at z=4, camera at origin looking +z. The CENTER ray goes
    // straight down +z, so it hits the front face at distance exactly 4.
    std::vector<collision::AABB> boxes = {Box({0.0f, 0.0f, 5.0f}, 1.0f, 1.0f, 1.0f)};
    boxes[0].min.z = 4.0f;
    // Even dimensions: no single pixel is exactly centered, but the 4 central
    // pixels are symmetric; use an ODD resolution so pixel (w/2,h/2) is dead
    // center and its ray is exactly +z.
    auto cam = rt::BuildPinhole({0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f},
                                {0.0f, 1.0f, 0.0f}, 0.5f * kPi, 65u, 65u);
    auto got = rt::RenderDepth(boxes, cam);
    const size_t center = static_cast<size_t>(32) * 65 + 32;
    EXPECT_EQ(got.prim[center], 0u);
    EXPECT_NEAR(got.depth[center], 4.0f, 1e-4f);
}
