# Nuka Physics v0.7 – Phase 4: SAP → LBVH Rigid Broadphase Upgrade

> **Master plan reference:** §3 Round 7 (broadphase upgrade) + §7 v0.7 exit criteria
> **Prerequisites:** v0.5 closed; v0.7 Phases 1-3 (solver suite — can run in parallel)
> **Blocks:** v0.7 Phase 5 (cross-system query needs LBVH on rigid side)
> **Exit criteria gate:** v0.7
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Replace the existing Sweep-and-Prune (SAP) broadphase with a **Linear BVH (LBVH)**. SAP scales poorly for large dynamic scenes (S3 envelope is 50K rigid bodies); LBVH is O(N log N) build + O(log N) query and parallelizes naturally on GPU.

This phase only handles **rigid-rigid** broadphase. Particle-particle broadphase (uniform grid) and cross-system queries (rigid BVH × particle grid) come in Phase 5.

Reference: Tero Karras 2012 "Maximizing Parallelism in the Construction of BVHs, Octrees, and k-d Trees."

## Tech Stack

- CUDA 12+
- Morton code sorting (existing in some forms; may use Thrust's deterministic sort)
- Existing collision detection pipeline

## Files to Create

- `src/collision/broadphase_lbvh.hpp`
- `src/collision/broadphase_lbvh.cu`
- `src/collision/morton_codes.cuh`
- `src/collision/lbvh_traversal.cuh` — pair query: which AABBs overlap which
- `src/collision/lbvh_refit.cu` — per-frame refit when topology unchanged
- `tests/collision/test_lbvh_build_correctness.cpp`
- `tests/collision/test_lbvh_overlap_query.cpp`
- `tests/collision/test_lbvh_vs_sap_pair_set.cpp` — both broadphases produce same pair set
- `tests/perf/test_lbvh_50k_bodies.cpp` — scaling benchmark

## Files to Modify

- `src/collision/broadphase.hpp` (existing) — abstract interface; add `BroadphaseImpl` enum
- `src/collision/sweep_and_prune.cpp` (existing) — keep but mark deprecated
- `src/runtime/world_stepper.cpp` — choose broadphase implementation
- `src/include/nuka/nuka.h` — add broadphase selection field to `nuka_world_desc_t`
- `src/constraint/contact_builder.cpp` — consume pair list from LBVH

## Tasks

### Task 7.4.1 — Morton code generation

`src/collision/morton_codes.cuh`:

```cuda
// 30-bit Morton code (10 bits per axis) from 3D position
__device__ uint32_t morton_3d_30bit(float x, float y, float z);

// 63-bit Morton code (21 bits per axis) for larger scenes
__device__ uint64_t morton_3d_63bit(float x, float y, float z);

// One thread per body; writes morton code + body index pair
__global__ void compute_morton_codes_kernel(
    uint32_t body_count,
    const float* __restrict__ body_aabb_centers,   // 3 floats per body
    const float3 scene_min, const float3 scene_max,
    uint32_t* __restrict__ out_morton,
    uint32_t* __restrict__ out_body_idx);
```

Scene bounding box computed via deterministic reduction on body AABBs at start of frame.

### Task 7.4.2 — Sort by Morton code

Use Thrust's `sort_by_key`. Caveat: Thrust's default sort is not guaranteed deterministic. Use radix sort with stable mode (Thrust supports this) → deterministic.

```cpp
thrust::stable_sort_by_key(thrust::cuda::par.on(stream),
                            morton_codes, morton_codes + n, body_indices);
```

### Task 7.4.3 — Build LBVH (Karras algorithm)

`src/collision/broadphase_lbvh.cu`:

Karras's algorithm:
- N leaves correspond to N bodies (sorted by Morton code).
- N-1 internal nodes; each spans a range of consecutive leaves.
- Determined by longest common prefix (LCP) of Morton codes.

```cuda
struct LbvhNode {
    int32_t left;     // child index; negative = leaf
    int32_t right;
    float3 aabb_min, aabb_max;
};

__global__ void build_lbvh_internal_nodes_kernel(
    uint32_t n_leaves,
    const uint32_t* __restrict__ morton_sorted,
    LbvhNode* __restrict__ nodes);

__global__ void compute_node_aabbs_kernel(
    uint32_t n_leaves,
    LbvhNode* __restrict__ nodes,
    const float3* __restrict__ leaf_aabb_min,
    const float3* __restrict__ leaf_aabb_max,
    uint32_t* __restrict__ parent_visited);   // atomic counters bottom-up
```

Bottom-up AABB propagation uses atomic counters (one per node) to track when both children have computed AABBs. This is an integer atomic, not float, so D1 is preserved (the order in which atomics fire doesn't affect the *result*, only the order of writes).

### Task 7.4.4 — Pair overlap query

For contact generation, find all pairs (i, j) where AABB(i) overlaps AABB(j).

`src/collision/lbvh_traversal.cuh`:

```cuda
__global__ void lbvh_pair_query_kernel(
    uint32_t leaf_count,
    const LbvhNode* nodes,
    const float3* leaf_aabb_min, const float3* leaf_aabb_max,
    uint32_t* out_pairs_packed,           // (i << 32) | j packed into uint64_t
    uint32_t* out_pair_count);
```

One thread per leaf; traverse the tree from root, checking AABB overlap. When a leaf-leaf overlap is found, append (i, j) to output buffer.

Determinism: output order varies by thread scheduling. To preserve D1, sort the output pair list after collection (deterministic stable sort by packed (i,j)).

### Task 7.4.5 — Refit (per-frame update without rebuild)

When body count and pair structure are stable across frames (typical for training), rebuilding the full BVH every frame is wasteful. **Refit** propagates new body AABBs up the existing tree:

```cuda
__global__ void lbvh_refit_kernel(uint32_t leaf_count,
                                   LbvhNode* nodes,
                                   const float3* new_leaf_aabb_min,
                                   const float3* new_leaf_aabb_max);
```

This is dramatically cheaper than rebuild. Trade-off: tree quality degrades over many refits (becomes a poor partition). Heuristic: refit for N frames, then rebuild. Tune N=10-50.

### Task 7.4.6 — Backend selection

```c
typedef enum {
    NUKA_BROADPHASE_SAP = 0,        /* legacy; deprecated */
    NUKA_BROADPHASE_LBVH = 1,       /* default for v0.7+ */
} nuka_broadphase_t;
```

Default flips to LBVH at v0.7.

### Task 7.4.7 — Tests

`tests/collision/test_lbvh_vs_sap_pair_set.cpp`:

```cpp
TEST(LbvhVsSap, SamePairSetOnSmokeScene) {
    auto pairs_sap = ComputePairsSap(smoke_scene);
    auto pairs_lbvh = ComputePairsLbvh(smoke_scene);
    EXPECT_EQ(SortedSet(pairs_sap), SortedSet(pairs_lbvh));
}
```

`tests/perf/test_lbvh_50k_bodies.cpp`:

```cpp
TEST(LbvhPerf, 50kBodiesBuildAndQuery) {
    auto t0 = now_us();
    auto bvh = BuildLbvh(50000_random_aabbs);
    auto t1 = now_us();
    auto pairs = QueryPairs(bvh);
    auto t2 = now_us();
    // Targets: build < 500us, query < 200us on RTX 4090
    EXPECT_LT(t1 - t0, 500);
    EXPECT_LT(t2 - t1, 200);
}
```

## Validation

- LBVH and SAP produce the same overlap pair set on all v0.1 smoke scenes.
- LBVH builds 50K bodies in < 500 µs.
- Pair query for 50K bodies < 200 µs.
- Refit-then-N-frames-then-rebuild policy improves perf vs rebuild-every-frame by 30%+.
- D1 determinism preserved (verified by bit-exact same pair list).

## Exit Criteria for v0.7 Phase 4

1. `BroadphaseLbvh` operational; produces correct pair set.
2. LBVH pair set agrees with SAP on smoke scenes.
3. 50K-body benchmark hits budget.
4. Refit + periodic rebuild policy in place.
5. Default broadphase flipped to LBVH for new World handles.
6. SAP retained but deprecated (used for diff-test only).
7. Determinism + lint pass.

## What This Phase Does Not Do

- No particle grid (Phase 5).
- No cross-system query (Phase 5).
- No sweep-test for continuous collision (separate future phase if needed).
- No deformable body broadphase (Phase 5 brings particles; deformable surfaces use particle grid).
