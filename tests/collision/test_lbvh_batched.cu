// ---------------------------------------------------------------------------
// Shared batched LBVH build primitive gate: env-batching == independent builds.
//
// One BuildLbvhBatchedNodes over E env AABB-sets must produce, per env, the
// BYTE-IDENTICAL node array that env_count=1 on that env's AABBs alone produces
// -- proving the u64 (env<<32)|morton cross-env stable_sort isolates envs bit-
// exactly (no cross-env mixing). The same is asserted for RefitLbvhBatched after
// a per-env pose nudge. Drives the interleaved collision::AABB overload (the
// render/standalone path); the split lo/hi overload runs the SAME kernels.
// ---------------------------------------------------------------------------

#include "collision/aabb.hpp"
#include "collision/lbvh_batched.cuh"
#include "collision/lbvh_node.cuh"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <cuda_runtime.h>

namespace {

namespace cg = nuka::collision::gpu;
using nuka::collision::AABB;

#define NK_CUDA_OK(call)                                                     \
    do {                                                                     \
        const cudaError_t e_ = (call);                                       \
        ASSERT_EQ(e_, cudaSuccess) << cudaGetErrorString(e_);               \
    } while (0)

// Device-scratch bundle for a build/refit over (env_count, leaves_per_env).
struct DeviceScratch {
    AABB* aabbs = nullptr;
    cg::LbvhNode* nodes = nullptr;
    uint32_t* morton = nullptr;
    uint32_t* index = nullptr;
    uint64_t* sortkey = nullptr;
    uint32_t* visit = nullptr;

    DeviceScratch(uint32_t env_count, uint32_t leaves_per_env) {
        const size_t leaves = static_cast<size_t>(env_count) * leaves_per_env;
        const size_t node_count =
            static_cast<size_t>(env_count) * (2u * leaves_per_env - 1u);
        EXPECT_EQ(cudaMalloc(&aabbs, leaves * sizeof(AABB)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&nodes, node_count * sizeof(cg::LbvhNode)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&morton, leaves * sizeof(uint32_t)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&index, leaves * sizeof(uint32_t)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&sortkey, leaves * sizeof(uint64_t)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&visit, leaves * sizeof(uint32_t)), cudaSuccess);
    }
    ~DeviceScratch() {
        cudaFree(aabbs);
        cudaFree(nodes);
        cudaFree(morton);
        cudaFree(index);
        cudaFree(sortkey);
        cudaFree(visit);
    }
    DeviceScratch(const DeviceScratch&) = delete;
    DeviceScratch& operator=(const DeviceScratch&) = delete;
};

// Deterministic synthetic per-env AABB set: small boxes scattered in a cube,
// the per-env seed shifts the layout so envs are genuinely distinct trees.
std::vector<AABB> MakeEnvAabbs(uint32_t leaves_per_env, uint32_t env_seed) {
    std::mt19937 rng(0xB7A1u + env_seed * 2654435761u);
    std::uniform_real_distribution<float> pos(-10.0f, 10.0f);
    std::uniform_real_distribution<float> ext(0.1f, 0.6f);
    std::vector<AABB> out(leaves_per_env);
    for (uint32_t i = 0; i < leaves_per_env; ++i) {
        const float cx = pos(rng), cy = pos(rng), cz = pos(rng);
        const float ex = ext(rng), ey = ext(rng), ez = ext(rng);
        out[i].min = {cx - ex, cy - ey, cz - ez};
        out[i].max = {cx + ex, cy + ey, cz + ez};
    }
    return out;
}

std::vector<cg::LbvhNode> DownloadNodes(const cg::LbvhNode* dev, size_t count) {
    std::vector<cg::LbvhNode> host(count);
    EXPECT_EQ(cudaMemcpy(host.data(), dev, count * sizeof(cg::LbvhNode),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
    return host;
}

constexpr uint32_t kEnvCount = 8u;
constexpr uint32_t kLeavesPerEnv = 37u;  // odd, > one block boundary friendliness

}  // namespace

// One batched build over E envs == E independent single-env builds, byte-exact
// per env (the cross-env u64 sort never mixes envs).
TEST(LbvhBatched, BatchedBuildEqualsIndependentBuilds) {
    const uint32_t node_count = 2u * kLeavesPerEnv - 1u;

    // Assemble the env-major AABB array + run ONE batched build.
    std::vector<AABB> all(static_cast<size_t>(kEnvCount) * kLeavesPerEnv);
    std::vector<std::vector<AABB>> per_env(kEnvCount);
    for (uint32_t e = 0; e < kEnvCount; ++e) {
        per_env[e] = MakeEnvAabbs(kLeavesPerEnv, e);
        std::memcpy(&all[static_cast<size_t>(e) * kLeavesPerEnv], per_env[e].data(),
                    kLeavesPerEnv * sizeof(AABB));
    }

    DeviceScratch batched(kEnvCount, kLeavesPerEnv);
    NK_CUDA_OK(cudaMemcpy(batched.aabbs, all.data(), all.size() * sizeof(AABB),
                          cudaMemcpyHostToDevice));
    cg::BuildLbvhBatchedNodes(/*stream=*/nullptr, /*device_id=*/0, batched.aabbs,
                              kEnvCount, kLeavesPerEnv, batched.nodes,
                              batched.morton, batched.index, batched.sortkey,
                              batched.visit);
    NK_CUDA_OK(cudaDeviceSynchronize());
    const auto batched_nodes =
        DownloadNodes(batched.nodes, static_cast<size_t>(kEnvCount) * node_count);

    // Build each env ALONE (env_count=1) and compare its node slice byte-for-byte.
    for (uint32_t e = 0; e < kEnvCount; ++e) {
        DeviceScratch one(1u, kLeavesPerEnv);
        NK_CUDA_OK(cudaMemcpy(one.aabbs, per_env[e].data(),
                              kLeavesPerEnv * sizeof(AABB), cudaMemcpyHostToDevice));
        cg::BuildLbvhBatchedNodes(/*stream=*/nullptr, /*device_id=*/0, one.aabbs,
                                  1u, kLeavesPerEnv, one.nodes, one.morton,
                                  one.index, one.sortkey, one.visit);
        NK_CUDA_OK(cudaDeviceSynchronize());
        const auto solo_nodes = DownloadNodes(one.nodes, node_count);

        const cg::LbvhNode* slice = &batched_nodes[static_cast<size_t>(e) * node_count];
        const int cmp = std::memcmp(slice, solo_nodes.data(),
                                    node_count * sizeof(cg::LbvhNode));
        EXPECT_EQ(cmp, 0) << "env " << e
                          << " batched node slice != independent build (byte-diff)";
    }
}

// A per-env pose nudge then RefitLbvhBatched over E envs == E independent
// single-env refits, byte-exact per env (topology reused; bounds re-propagated).
TEST(LbvhBatched, BatchedRefitEqualsIndependentRefits) {
    const uint32_t node_count = 2u * kLeavesPerEnv - 1u;

    std::vector<AABB> all(static_cast<size_t>(kEnvCount) * kLeavesPerEnv);
    std::vector<std::vector<AABB>> per_env(kEnvCount);
    std::vector<std::vector<AABB>> per_env_nudged(kEnvCount);
    for (uint32_t e = 0; e < kEnvCount; ++e) {
        per_env[e] = MakeEnvAabbs(kLeavesPerEnv, e);
        std::memcpy(&all[static_cast<size_t>(e) * kLeavesPerEnv], per_env[e].data(),
                    kLeavesPerEnv * sizeof(AABB));
        // Nudge: a per-env-distinct shift so each refit input differs.
        per_env_nudged[e] = per_env[e];
        const float dx = 0.07f * static_cast<float>(e + 1);
        for (auto& b : per_env_nudged[e]) {
            b.min.x += dx; b.max.x += dx;
            b.min.y -= 0.03f; b.max.y -= 0.03f;
        }
    }
    std::vector<AABB> all_nudged(all.size());
    for (uint32_t e = 0; e < kEnvCount; ++e) {
        std::memcpy(&all_nudged[static_cast<size_t>(e) * kLeavesPerEnv],
                    per_env_nudged[e].data(), kLeavesPerEnv * sizeof(AABB));
    }

    // Batched: build over the original, then refit over the nudged set in place.
    DeviceScratch batched(kEnvCount, kLeavesPerEnv);
    NK_CUDA_OK(cudaMemcpy(batched.aabbs, all.data(), all.size() * sizeof(AABB),
                          cudaMemcpyHostToDevice));
    cg::BuildLbvhBatchedNodes(/*stream=*/nullptr, /*device_id=*/0, batched.aabbs,
                              kEnvCount, kLeavesPerEnv, batched.nodes,
                              batched.morton, batched.index, batched.sortkey,
                              batched.visit);
    AABB* d_nudged = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_nudged, all_nudged.size() * sizeof(AABB)));
    NK_CUDA_OK(cudaMemcpy(d_nudged, all_nudged.data(),
                          all_nudged.size() * sizeof(AABB), cudaMemcpyHostToDevice));
    cg::RefitLbvhBatched(/*stream=*/nullptr, /*device_id=*/0, batched.nodes,
                         d_nudged, kEnvCount, kLeavesPerEnv, batched.visit);
    NK_CUDA_OK(cudaDeviceSynchronize());
    const auto batched_nodes =
        DownloadNodes(batched.nodes, static_cast<size_t>(kEnvCount) * node_count);
    cudaFree(d_nudged);

    for (uint32_t e = 0; e < kEnvCount; ++e) {
        DeviceScratch one(1u, kLeavesPerEnv);
        NK_CUDA_OK(cudaMemcpy(one.aabbs, per_env[e].data(),
                              kLeavesPerEnv * sizeof(AABB), cudaMemcpyHostToDevice));
        cg::BuildLbvhBatchedNodes(/*stream=*/nullptr, /*device_id=*/0, one.aabbs,
                                  1u, kLeavesPerEnv, one.nodes, one.morton,
                                  one.index, one.sortkey, one.visit);
        AABB* d_one_nudged = nullptr;
        NK_CUDA_OK(cudaMalloc(&d_one_nudged, kLeavesPerEnv * sizeof(AABB)));
        NK_CUDA_OK(cudaMemcpy(d_one_nudged, per_env_nudged[e].data(),
                              kLeavesPerEnv * sizeof(AABB), cudaMemcpyHostToDevice));
        cg::RefitLbvhBatched(/*stream=*/nullptr, /*device_id=*/0, one.nodes,
                             d_one_nudged, 1u, kLeavesPerEnv, one.visit);
        NK_CUDA_OK(cudaDeviceSynchronize());
        const auto solo_nodes = DownloadNodes(one.nodes, node_count);
        cudaFree(d_one_nudged);

        const cg::LbvhNode* slice = &batched_nodes[static_cast<size_t>(e) * node_count];
        const int cmp = std::memcmp(slice, solo_nodes.data(),
                                    node_count * sizeof(cg::LbvhNode));
        EXPECT_EQ(cmp, 0) << "env " << e
                          << " batched refit slice != independent refit (byte-diff)";
    }
}

// The split lo/hi overload (the collision-arena layout) reassembles the SAME
// interleaved AABB -> byte-identical node array to the interleaved overload.
TEST(LbvhBatched, SplitLoHiOverloadMatchesInterleaved) {
    const uint32_t node_count = 2u * kLeavesPerEnv - 1u;
    const size_t leaves = static_cast<size_t>(kEnvCount) * kLeavesPerEnv;

    std::vector<AABB> all(leaves);
    std::vector<nuka::math::Vec3> lo(leaves), hi(leaves);
    for (uint32_t e = 0; e < kEnvCount; ++e) {
        const auto env = MakeEnvAabbs(kLeavesPerEnv, e);
        for (uint32_t i = 0; i < kLeavesPerEnv; ++i) {
            const size_t at = static_cast<size_t>(e) * kLeavesPerEnv + i;
            all[at] = env[i];
            lo[at] = env[i].min;
            hi[at] = env[i].max;
        }
    }

    DeviceScratch inter(kEnvCount, kLeavesPerEnv);
    NK_CUDA_OK(cudaMemcpy(inter.aabbs, all.data(), leaves * sizeof(AABB),
                          cudaMemcpyHostToDevice));
    cg::BuildLbvhBatchedNodes(/*stream=*/nullptr, /*device_id=*/0, inter.aabbs,
                              kEnvCount, kLeavesPerEnv, inter.nodes, inter.morton,
                              inter.index, inter.sortkey, inter.visit);

    DeviceScratch split(kEnvCount, kLeavesPerEnv);
    nuka::math::Vec3* d_lo = nullptr;
    nuka::math::Vec3* d_hi = nullptr;
    NK_CUDA_OK(cudaMalloc(&d_lo, leaves * sizeof(nuka::math::Vec3)));
    NK_CUDA_OK(cudaMalloc(&d_hi, leaves * sizeof(nuka::math::Vec3)));
    NK_CUDA_OK(cudaMemcpy(d_lo, lo.data(), leaves * sizeof(nuka::math::Vec3),
                          cudaMemcpyHostToDevice));
    NK_CUDA_OK(cudaMemcpy(d_hi, hi.data(), leaves * sizeof(nuka::math::Vec3),
                          cudaMemcpyHostToDevice));
    cg::BuildLbvhBatchedNodes(/*stream=*/nullptr, /*device_id=*/0, d_lo, d_hi,
                              kEnvCount, kLeavesPerEnv, split.nodes, split.morton,
                              split.index, split.sortkey, split.visit);
    NK_CUDA_OK(cudaDeviceSynchronize());

    const size_t total = static_cast<size_t>(kEnvCount) * node_count;
    const auto a = DownloadNodes(inter.nodes, total);
    const auto b = DownloadNodes(split.nodes, total);
    cudaFree(d_lo);
    cudaFree(d_hi);
    EXPECT_EQ(std::memcmp(a.data(), b.data(), total * sizeof(cg::LbvhNode)), 0)
        << "split lo/hi overload diverged from the interleaved AABB overload";
}
