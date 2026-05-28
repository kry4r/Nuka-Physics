# Nuka Physics v0.7 – Phase 12: Self-Written CUDA Ray Tracer — LBVH + Stackless Traversal

> **Master plan reference:** §3 Round 10 (self-written CUDA RT, no OptiX) + §3 Round 13 (S2 RT pipeline)
> **Prerequisites:** v0.7 Phase 4 (LBVH for rigid; some scaffolding reused)
> **Blocks:** v0.7 Phase 13 (intersections + shading) and Phase 14 (sensors)
> **Exit criteria gate:** v0.7
> **🔒 HARD CONSTRAINT (project-wide):** GPU-only simulation. No CPU physics simulation in production code paths. See master plan §5.6.

## Goal

Begin construction of the self-written CUDA ray tracing pipeline. Master plan §3 Round 10 commits to pure-CUDA RT (no OptiX dependency) with full analytical adjoint and D1 determinism.

This phase delivers:
1. **LBVH for ray-tracing acceleration** — distinct from Phase 4's collision LBVH (different traversal patterns, different memory layout optimized for rays).
2. **Stackless traversal kernel** — persistent threads, per-ray BVH walk.
3. **Ray-AABB intersection** as the primitive test.

Phase 13 adds the geometric primitive intersections (triangles, spheres, SDFs) and shading. Phase 14 wires these into sensor outputs.

## Tech Stack

- CUDA 12+
- Persistent threads pattern (work-stealing)
- Phase 4's LBVH build code (refactor + extend)

## Files to Create

- `src/rt/rt_lbvh.hpp`
- `src/rt/rt_lbvh.cu` — RT-specific BVH build (different priority than collision BVH)
- `src/rt/ray.cuh` — ray structure + intersection record
- `src/rt/traversal_stackless.cu` — stackless BVH walk kernel
- `src/rt/ray_aabb.cuh` — slab method for ray-AABB
- `src/rt/persistent_thread_scheduler.cuh` — work distribution
- `tests/rt/test_rt_lbvh_build.cpp`
- `tests/rt/test_traversal_correctness.cpp`
- `tests/perf/test_rt_traversal_throughput.cpp`

## Files to Modify

- `src/render/vulkan_renderer.cpp` — eventually integrate RT-rendered framebuffer for sensors (Phase 14)

## Tasks

### Task 7.12.1 — Ray and intersection structures

`src/rt/ray.cuh`:

```cuda
struct Ray {
    float3 origin;
    float3 direction;
    float  t_min, t_max;
};

struct Intersection {
    float t;                    // distance along ray
    uint32_t primitive_id;      // hit primitive
    float2 barycentric;         // for triangles
    float3 normal;              // surface normal at hit
    uint32_t material_id;
};

// Initial value: no hit
__device__ Intersection make_miss() { return {INFINITY, 0, {0,0}, {0,0,0}, 0}; }
```

### Task 7.12.2 — RT-optimized LBVH build

Distinct from Phase 4 collision LBVH:
- Collision: pairs query (which AABBs overlap which) — uses Karras parallel build.
- RT: per-ray traversal — wants higher-quality tree (more separation, less false positives).

Use **Surface Area Heuristic (SAH)** approximation in the build to produce a better tree. Reference: Wald 2014 — fast LBVH then top-down SAH refinement.

```cuda
// Phase 1: Karras LBVH base build (same as Phase 4)
// Phase 2: top-down SAH refit at each internal node
__global__ void rt_lbvh_sah_refit_kernel(...);
```

Build time is mostly a one-time cost per frame for dynamic scenes; SAH refit adds ~50% build cost but yields 30-50% better traversal throughput.

### Task 7.12.3 — Stackless traversal

Reference: Hapala et al. 2011 "Efficient Stack-less BVH Traversal for Ray Tracing."

Stackless avoids per-thread stack memory (saves register pressure on GPU). Uses parent/sibling/skip pointers in the BVH structure.

```cuda
__device__ Intersection traverse_lbvh_stackless(
    const Ray& ray,
    const RtLbvhNode* nodes,
    const uint32_t root_idx)
{
    Intersection best = make_miss();
    int32_t node_idx = root_idx;
    while (node_idx != -1) {
        const RtLbvhNode& n = nodes[node_idx];
        bool hit_aabb = ray_aabb_intersect(ray, n.aabb_min, n.aabb_max);
        if (hit_aabb) {
            if (n.is_leaf) {
                // Test geometry (Phase 13 fills this in; placeholder: hit at AABB)
                Intersection isect = test_primitive(ray, n.primitive_id);
                if (isect.t < best.t) best = isect;
            }
            node_idx = n.left;   // go down
        } else {
            node_idx = n.miss;   // skip-pointer to next-to-visit
        }
    }
    return best;
}
```

Skip pointers (`miss` field) are precomputed at build time.

### Task 7.12.4 — Persistent threads

For ray dispatch with varying workload (some rays terminate early, some deeply traverse), use **persistent threads** pattern:
- Launch fixed number of threads (matching GPU SM count × max threads / SM).
- Each thread pulls work from a global ray queue.
- Self-balances workload across rays.

`src/rt/persistent_thread_scheduler.cuh`:

```cuda
__global__ void rt_persistent_kernel(
    Ray* ray_queue, uint32_t* ray_count,
    Intersection* results,
    const RtLbvhNode* bvh,
    /* ... */)
{
    while (true) {
        uint32_t idx = atomicAdd(ray_count, 1);
        if (idx >= total_rays) return;
        Ray r = ray_queue[idx];
        results[idx] = traverse_lbvh_stackless(r, bvh, 0);
    }
}
```

Note: `atomicAdd` here is on `uint32_t`, not float — does not violate D1 contract. The output `results[idx]` is determined by `idx`, not by execution order, so result is bit-exact.

### Task 7.12.5 — Ray-AABB intersection

`src/rt/ray_aabb.cuh`:

```cuda
__device__ bool ray_aabb_intersect(const Ray& r, float3 aabb_min, float3 aabb_max,
                                    float& t_near, float& t_far)
{
    // Slab method
    float3 inv_dir = float3{1.f/r.direction.x, 1.f/r.direction.y, 1.f/r.direction.z};
    float3 t0 = (aabb_min - r.origin) * inv_dir;
    float3 t1 = (aabb_max - r.origin) * inv_dir;
    t_near = fmaxf(fmaxf(fminf(t0.x, t1.x), fminf(t0.y, t1.y)), fminf(t0.z, t1.z));
    t_far  = fminf(fminf(fmaxf(t0.x, t1.x), fmaxf(t0.y, t1.y)), fmaxf(t0.z, t1.z));
    return t_far >= t_near && t_far >= r.t_min && t_near < r.t_max;
}
```

### Task 7.12.6 — Tests

`tests/rt/test_rt_lbvh_build.cpp`:

```cpp
// 1000 random AABBs; build LBVH; verify all AABBs covered by traversal
```

`tests/rt/test_traversal_correctness.cpp`:

```cpp
// Known scene: 100 unit spheres; rays from camera
// Each pixel computes nearest sphere via brute force AND via LBVH; agree on hit primitive
```

`tests/perf/test_rt_traversal_throughput.cpp`:

```cpp
// 1920x1080 rays × 10K primitives
// Target: > 50 MRays/s on RTX 4090 (without RT Cores; we're not using them)
```

50 MRays/s is ~5-10× slower than OptiX with RT Cores; this is the price of self-written.

## Validation

- LBVH build covers all primitives.
- Stackless traversal matches brute force on test scenes.
- Persistent thread scheduler distributes work; no per-thread imbalance.
- Determinism: same scene + same rays → bit-exact hit results.
- 50 MRays/s throughput baseline (will improve as Phase 13 adds primitive tests).

## Exit Criteria for v0.7 Phase 12

1. RT-LBVH builder operational with SAH refit.
2. Stackless traversal kernel runs.
3. Persistent thread scheduler functional.
4. Ray-AABB intersection passes correctness.
5. Throughput baseline established.
6. Determinism tests pass.

## What This Phase Does Not Do

- No triangle / sphere / SDF intersection (Phase 13).
- No shading (Phase 13).
- No RT-rendered sensor output (Phase 14).
- No RT for shadow rays / GI (future visual fidelity phase).
- No diff-rendering — adjoint of RT lands in v2.0 Phase 7.
